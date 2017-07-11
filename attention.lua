---This files calls the classes cnn and the masked/modified lstm and 
---then clubs them together to form a new class attention
require 'nn'
require 'opts.lua'
require 'utilities.lua'
require 'nngraph'
require 'modified_lstm'
local CNN=require 'cnn_model.lua'

modified_lstm=nn.modified_lstm()

local layer,parent=torch.class('nn.attention','nn.Module')

function layer:__init()
	parent.__init(self)
	self.seq_length=opt.seq_length
	self.lstm=nn.modified_lstm().modified_lstm
	self.cnn=CNN.cnn()
	--self.lstm_output = torch.Tensor()
	self.batch_size=opt.batch_size
	self.seq_length=8
	self.lstm_output_size=36 ------size of attention map
  self.num_op_hid=3----the no. of hidden states that are concatenated for obtaining output
  self.hidden_size=opt.hidden_size
  self.concat_norm=self:create_concat_norm(self.num_op_hid,self.hidden_size)
  self.batch_size=opt.batchsize
end

function layer:create_concat_norm(num_hidden,hidden_size)
  inputs={nn.JoinTable(2)()}
  i1=nn.Identity()(inputs[1])
  norm=nn.CMulTable()({i1,inputs[1]})
  sum=nn.Sum(2)(norm)
  repli=nn.Replicate(num_hidden*hidden_size,2)(sum)
  out=nn.CDivTable()({inputs[1],repli})
  op={out}
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
    return {self.cnn,self.lstm}
end

function layer:parameters()
    -- we only have two internal modules, return their params
    local p1,g1 = self.cnn:parameters()
    local p3,g3 = self.lstm:parameters()

    local params = {}
    for k,v in pairs(p1) do table.insert(params, v) end
    for k,v in pairs(p3) do table.insert(params, v) end

    local grad_params = {}
    for k,v in pairs(g1) do table.insert(grad_params, v) end
    for k,v in pairs(g3) do table.insert(grad_params, v) end

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

function layer:initialize_gradients()
  local dl_dc8,dl_dconv8,dl_dloc8
  dl_dc8=torch.Tensor(self.batch_size,self.hidden_size):zero()
  dl_dconv8=torch.Tensor(self.batch_size,256,6,6):zero()
  dl_dloc8=torch.Tensor(self.batch_size,6*6):zero()
  return dl_dc8,dl_dconv8,dl_dloc8
end

function layer:updateOutput(input)

  location_map,cell_st,hid_st,self.conv_feat=unpack(self.cnn:forward(input))
  print(self.cnn:forward(input))
  if self.lstm_units == nil then self:create_clones() end -- lazily create clones on first forward pass
  
  self.fore_inputs_and_states = {[1]={location_map,cell_st,hid_st,self.conv_feat}}
  self.core_output={}
  self.hidden_states={}
  self.output_lstm={}
  for t=1,self.seq_length do
  	  local out=self.lstm_units[t]:forward(self.fore_inputs_and_states[t])
  	  table.insert(self.fore_inputs_and_states,out)
  	  location_map,cell_st,hid_st,self.conv_feat=unpack(out)
      if(t==2 or t==4 or t==8) then
  	     table.insert(self.output_lstm,hid_st)
      end   
  end
  print(self.output_lstm)
  local output=self.concat_norm:forward(self.output_lstm)
  return output
end

function layer:updateGradInput(input, gradOutput)
  --dl_dh2 means derivative of loss w.r.t. hidden state-2
  dl_dh2,dl_dh4,dl_dh8=unpack(self.concat_norm:backward(self.output_lstm,gradOutput))
  --computing gradients for lstm

  dl_dc8,dl_dconv8,dl_dloc8=self:initialize_gradients() ---initializing grad_loss w.r.t. output of last time step 
  gradOutput_lstm={[self.seq_length]={dl_dloc8,dl_dc8,dl_dh8,dl_dconv8}
  for t=self.seq_length,8,-1 do
    print(self.lstm_units[t]:backward(self.fore_inputs_and_states[t],gradOutput_lstm[t]))
  end  



  -- local batch_size = ques:size(1)

  -- local d_core_output, d_imgfeat, dummy = unpack(self.atten:backward({self.core_output, img, self.mask}, gradOutput))

  -- -- go backwards and lets compute gradients
  -- local d_core_state = {[self.tmax] = self.init_state} -- initial dstates
  -- local d_embed_core = d_embed_core or self.core_output:new()
  -- d_embed_core:resize(batch_size, self.seq_length, self.rnn_size):zero()

  -- for t=self.tmax,1,-1 do
  --   -- concat state gradients and output vector gradients at time step t
  --   local dout = {}
  --   for k=1,#d_core_state[t] do table.insert(dout, d_core_state[t][k]) end
  --   table.insert(dout, d_core_output:narrow(2,t,1):contiguous():view(-1, self.hidden_size))
  --   local dinputs = self.cores[t]:backward(self.fore_inputs[t], dout)

  --   if t > self.tmin then
  --     for k=1,self.num_state+1 do
  --       dinputs[k]:maskedFill(self.mask:narrow(2,t,1):contiguous():view(batch_size,1):expandAs(dinputs[k]), 0)
  --     end
  --   end
  --   d_core_state[t-1] = {} -- copy over rest to state grad
  --   for k=2,self.num_state+1 do table.insert(d_core_state[t-1], dinputs[k]) end
  --   d_embed_core:narrow(2,t,1):copy(dinputs[1])
  -- end
  -- self.gradInput = {d_embed_core, d_imgfeat}
  -- return self.gradInput
end




