require 'nn'
require 'opts.lua'
local LSTM = require 'implementlstm.lua'
local CNN=require 'cnn_model.lua'

local layer, parent = torch.class('nn.attention', 'nn.Module')
function layer:__init()
    parent.__init(self)
    self.num_layers = opt.num_layers--number of layers in the lstm
    local dropout = opt.drop_out
    self.rnn_size = opt.hidden_size
    self.seq_length =  opt.seq_length
    self.core = LSTM.lstm(self.rnn_size, self.rnn_size, self.num_layers, dropout)
    self.cnn  = CNN.cnn()
    self:_createInitState(1)
    self.mask = torch.Tensor()
    self.core_output = torch.Tensor()
end
--[[ This funtion will initialize the first hidden and cell state respectively]]--
function layer:_createInitState(batch_size)
  assert(batch_size ~= nil, 'batch size must be provided')
  -- construct the initial state for the LSTM
  if not self.init_state then self.init_state = {} end -- lazy init
  for h=1,self.num_layers*2 do
    -- note, the init state Must be zeros because we are using init_state to init grads in backward call too
    if self.init_state[h] then
      if self.init_state[h]:size(1) ~= batch_size then
        self.init_state[h]:resize(batch_size, self.rnn_size):zero() -- expand the memory
      end
    else
      if h%2==0 then
          self.init_state[h]=nn.Identity()(self.cnn[2][3]):annotate{
          name='initial cell state'
        }
      else
          self.init_state[h]=nn.Identity()(self.cnn[2][2]):annotate{
          name='cnn->mlp->initial hidden state'
        }
      end          
    end
  end
  self.num_state = #self.init_state
  end
-- This function creates the lstm core for different time steps..In our case num_steps=
function layer:createClones()
    print('constructing clones inside the ques_level')
    self.cores = {self.core}
    for t=1,self.seq_length do
        self.cores[t] = self.core:clone('weight', 'bias', 'gradWeight', 'gradBias')
    end
end

function layer:shareClones()
    if self.cores == nil then self:createClones(); return; end
    print('resharing clones inside the ques_level')
    self.cores[1] = self.core
    for t=1,self.seq_length do
        self.cores[t]:share(self.core, 'weight', 'bias', 'gradWeight', 'gradBias')
    end
end

function layer:getModulesList()
    return {self.core}
end

function layer:parameters()
    -- we only have two internal modules, return their params
    local p,g = self.core:parameters()

    local params = {}
    for k,v in pairs(p) do table.insert(params, v) end

    local grad_params = {}
    for k,v in pairs(g) do table.insert(grad_params, v) end

    return params, grad_params
end


function layer:training()
    if self.cores == nil then self:createClones() end -- create these lazily if needed
    for k,v in pairs(self.cores) do v:training() end
end

function layer:evaluate()
    if self.cores == nil then self:createClones() end -- create these lazily if needed
    for k,v in pairs(self.cores) do v:evaluate() end
end
-- I have doubt on how will we update this function
function layer:updateOutput(input)
  --local ques = input[1]
  --local seq_len = input[2]
  --local img = input[3]
  --self.mask = input[4]
  --cnn_output=cnn_model:forward(input)
  --if we do the above step to compute the cnn output how'll the gradients flow backward through the cnn


  if self.cores == nil then self:createClones() end -- lazily create clones on first forward pass
  local batch_size = ques:size(1)
  self.tmax = torch.max(seq_len)
  self.tmin = torch.min(seq_len)

  self:_createInitState(batch_size)
  self.fore_state = {[0] = self.init_state}
  self.fore_inputs = {}
  self.core_output:resize(batch_size, self.seq_length, self.rnn_size):zero()


  for t=1,self.tmax do
      self.fore_inputs[t] = {ques:narrow(2,t,1):contiguous():view(-1, self.rnn_size), unpack(self.fore_state[t-1])}
      local out = self.cores[t]:forward(self.fore_inputs[t])
      if t > self.tmin then
        for i=1,self.num_state+1 do
          out[i]:maskedFill(self.mask:narrow(2,t,1):contiguous():view(batch_size,1):expandAs(out[i]), 0)
        end
      end
      self.fore_state[t] = {} -- the rest is state
      for i=1,self.num_state do table.insert(self.fore_state[t], out[i]) end
      
      self.core_output:narrow(2,t,1):copy(out[self.num_state+1])
  end
  
  local w_lstm_ques, w_lstm_img, ques_atten, img_atten = unpack(self.atten:forward({self.core_output, img, self.mask}))

  return {w_lstm_ques, w_lstm_img, ques_atten, img_atten}
end

function layer:updateGradInput(input, gradOutput)
  local ques = input[1]
  local seq_len = input[2]
  local img = input[3]

  local batch_size = ques:size(1)

  local d_core_output, d_imgfeat, dummy = unpack(self.atten:backward({self.core_output, img, self.mask}, gradOutput))

  -- go backwards and lets compute gradients
  local d_core_state = {[self.tmax] = self.init_state} -- initial dstates
  local d_embed_core = d_embed_core or self.core_output:new()
  d_embed_core:resize(batch_size, self.seq_length, self.rnn_size):zero()

  for t=self.tmax,1,-1 do
    -- concat state gradients and output vector gradients at time step t
    local dout = {}
    for k=1,#d_core_state[t] do table.insert(dout, d_core_state[t][k]) end
    table.insert(dout, d_core_output:narrow(2,t,1):contiguous():view(-1, self.hidden_size))
    local dinputs = self.cores[t]:backward(self.fore_inputs[t], dout)

    if t > self.tmin then
      for k=1,self.num_state+1 do
        dinputs[k]:maskedFill(self.mask:narrow(2,t,1):contiguous():view(batch_size,1):expandAs(dinputs[k]), 0)
      end
    end
    d_core_state[t-1] = {} -- copy over rest to state grad
    for k=2,self.num_state+1 do table.insert(d_core_state[t-1], dinputs[k]) end
    d_embed_core:narrow(2,t,1):copy(dinputs[1])
  end
  self.gradInput = {d_embed_core, d_imgfeat}
  return self.gradInput
end
