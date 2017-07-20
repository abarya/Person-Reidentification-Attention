---This files calls the classes cnn and the masked/modified lstm and 
---then clubs them together to form a new class attention
require 'nn'
require 'opts.lua'
require 'utilities.lua'
require 'nngraph'
local LSTM = require 'implementlstm.lua'
--local CNN=require 'cnn_model.lua'

local layer,parent=torch.class('nn.attention','nn.Module')

function layer:__init()
	parent.__init(self)
  self.batch_size=opt.batchsize
  self.num_classes=871
	self.seq_length=opt.seq_length
  local dropout = opt.drop_out
  self.rnn_size = opt.hidden_size
  self.num_layers = opt.num_layers  
  self.cnn_op_size=6
  self.cnn_op_depth=256
  self.input_size=256---attention map
	self.lstm_output_size=36 ------size of attention map
  self.num_op_hid=3----the no. of hidden states that are concatenated for obtaining output
  self.hidden_size=opt.hidden_size
  self.lstm=LSTM.lstm(self.input_size, self.rnn_size, self.num_layers, dropout)
  self.concat_norm=self:create_concat_norm(self.num_op_hid,self.hidden_size)
end

function layer:create_concat_norm(num_hidden,hidden_size)
  inputs={nn.JoinTable(2)()}
  i1=nn.Identity()(inputs[1])
  norm=nn.CMulTable()({i1,inputs[1]})
  sum=nn.Sum(2)(norm)
  repli=nn.Replicate(num_hidden*hidden_size,2)(sum)
  out1=nn.CDivTable()({inputs[1],repli})
  prediction=nn.Linear(1536,self.num_classes)(out1)
  op={out1,prediction}
  return nn.gModule(inputs,op)
end  

function layer:create_clones()
	self.lstm_units={self.lstm}
	for t=1,self.seq_length do
        self.lstm_units[t] = self.lstm:clone('weight', 'bias', 'gradWeight', 'gradBias')
    end
end

function layer:shareClones()
    if self.lstm_units == nil then self:create_clones(); return; end
    print('resharing clones inside the attention model')
    self.lstm_units[1] = self.core
    for t=1,self.seq_length do
        self.lstm_units[t]:share(self.lstm, 'weight', 'bias', 'gradWeight', 'gradBias')
    end
end

function layer:getModulesList()
    return {self.lstm}
end

function layer:parameters()
    -- we only have two internal modules, return their params
    local p,g = self.lstm:parameters()
    local params = {}
    for k,v in pairs(p) do 
      table.insert(params, v)
    end
    local grad_params = {}
    for k,v in pairs(g) do table.insert(grad_params, v) end
    return params, grad_params
end

function layer:getParameters()
    local p,g = self.lstm:parameters()
    local p1,g1=self.concat_norm:getParameters()
    local params = {}
    for k,v in pairs(p) do table.insert(params, v)end
    for k,v in pairs(p1) do table.insert(params, v)end
    local grad_params = {}
    for k,v in pairs(g) do table.insert(grad_params, v) end
    for k,v in pairs(g1) do table.insert(grad_params, v) end
    return params, grad_params
end  

function layer:training()
    if self.lstm_units == nil then self:createClones() end -- create these lazily if needed
    for k,v in pairs(self.lstm_units) do v:training() end
end

function layer:evaluate()
    if self.lstm_units == nil then self:createClones() end -- create these lazily if needed
    for k,v in pairs(self.lstm_units) do v:evaluate() end
end

function layer:initialize_gradients(batchsize)
  local dl_dc8,dl_dconv8,dl_dloc8,dl_dh8
  dl_dc8=torch.Tensor(batchsize,self.hidden_size):zero():cuda()
  dl_dconv8=torch.Tensor(batchsize,256,6,6):zero():cuda()
  dl_dloc8=torch.Tensor(batchsize,6*6):zero():cuda()
  dl_dh8 =dl_dc8:clone():cuda()
  return dl_dc8,dl_dconv8,dl_dloc8,dl_dh8
end

function layer:updateOutput(input)
  location_map,cell_st,hid_st,self.conv_feat=unpack(input)
  if self.lstm_units == nil then self:create_clones() end -- lazily create clones on first forward pass
  
  self.fore_inputs_and_states = {[1]={location_map,cell_st,hid_st,self.conv_feat}}
  self.core_output={}
  self.hidden_states={}
  self.output_lstm={}
  for t=1,self.seq_length do
  	  local out=self.lstm_units[t]:forward(self.fore_inputs_and_states[t])
      location_map,cell_st,hid_st,self.conv_feat,out_lstm=unpack(out)
  	  table.insert(self.fore_inputs_and_states,{location_map,cell_st,hid_st,self.conv_feat})
      if(t==2 or t==4 or t==8) then
  	     table.insert(self.output_lstm,out_lstm)
      end   
  end
  -- print(self.output_lstm)
  local output=self.concat_norm:forward(self.output_lstm)
  return output
end

function layer:updateGradInput(input, gradOutput)
  --dl_dh2 means derivative of loss w.r.t. hidden state-2
  dl_dop2,dl_dop4,dl_dop8=unpack(self.concat_norm:backward(self.output_lstm,gradOutput))
  --computing gradients for lstm

  dl_dc8,dl_dconv8,dl_dloc8,dl_dh8=self:initialize_gradients(input[1]:size(1)) ---initializing grad_loss w.r.t. output of last time step 
  self.gradOutput_lstm={[self.seq_length]={dl_dloc8,dl_dc8,dl_dh8,dl_dconv8,dl_dop8}}
  for t=self.seq_length,1,-1 do
    if (t-1==2) then
      dl_dop=dl_dop2
    elseif (t-1==4) then
      dl_dop=dl_dop4
    else
      dl_dop=dl_dop8:clone():zero()  
    end 
    dl_dloc,dl_dc,dl_dh,dl_dconv=unpack(self.lstm_units[t]:backward(self.fore_inputs_and_states[t],self.gradOutput_lstm[t]))
    self.gradOutput_lstm[t-1]={dl_dloc,dl_dc,dl_dh,dl_dconv,dl_dop}
    --
  end
  self.gradInput=self.gradOutput_lstm[0]
  return self.gradInput
end




