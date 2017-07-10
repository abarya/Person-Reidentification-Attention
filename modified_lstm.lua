require 'nn'
require 'opts.lua'
require 'utilities.lua'
require 'nngraph'
dofile 'implementlstm.lua'
local LSTM = require 'implementlstm.lua'

local layer,parent=torch.class('nn.modified_lstm','nn.Module')

function layer:__init()
	parent.__init(self)
	self.num_layers = opt.num_layers     --number of layers in the lstm
    local dropout = opt.drop_out
    self.rnn_size = opt.hidden_size
    self.seq_length =  opt.seq_length
    self.cnn_op_size=6
    self.cnn_op_depth=256
    self.input_size=256---attention map
    self.core = LSTM.lstm(self.input_size, self.rnn_size, self.num_layers, dropout)
    --self:_createInitState(1)
    self:_join_layers(1)
    --self:create_clones()
    --self:shareClones()
end

function layer:_join_layers(batchsize)
	
	model={}	--This model will take as input a 3D feature map from the conv net and the hidden and cell state from the previous layer
	---For LSTM, the input is given as a table of 3 tensors with dimension (batchsize,input)
	---This model will take two inputs..one of them comes from cnn and the other from the output of the previous time step
	--- from the cnn, we get  a table of three tensors   --{conv_feat,initial_hidden_st,initial_cell_st}
	-- table.insert(model,nn.Identity()():annotate{name='conv_feat'})
	-- table.insert(model,nn.Identity()():annotate{name='o/p from previous time step'})--this is 1d tensor
	-- table.insert(model,nn.Identity()():annotate{name='previous hidden state'})
	-- table.insert(model,nn.Identity()():annotate{name='previous cell state'})
	-- --converting to 2D and 
	-- reshaped=nn.Reshape(self.cnn_op_size,self.cnn_op_size)(model[2])
	-- replicated=nn.Replicate(self.cnn_op_depth)(reshaped)
	-- product=nn.CMulTable()({model[1],replicated})
	-- avg_pool=nn.SpatialAveragePooling(self.cnn_op_size,self.cnn_op_size,self.cnn_op_size,self.cnn_op_size,0,0)--(product)
	-- avg_pool.divide=false
	-- input_lstm=avg_pool(product)
	-- output={input_lstm,model[3],model[4],model[1]} --conv_feat,input,hidden state,cel state
	-- gmodel=nn.gModule(model,output)
	-- extended_model=nn.Sequential()
	-- extended_model:add(gmodel)
	self.modified_lstm=self.core
end

-- function layer:create_clones()
-- 	self.lstm_units={self.modified_lstm}
-- 	for t=1,self.seq_length do
--         self.lstm_units[t] = self.modified_lstm:clone('weight', 'bias', 'gradWeight', 'gradBias')
--     end
-- end

-- function layer:shareClones()
--     if self.lstm_units == nil then self:create_clones(); return; end
--     print('resharing clones inside the attention model')
--     self.lstm_units[1] = self.core
--     for t=1,self.seq_length do
--         self.lstm_units[t]:share(self.core, 'weight', 'bias', 'gradWeight', 'gradBias')
--     end
-- end

-- mylstm=nn.modified_lstm()
-- mylstm.modified_lstm:forward(torch.ones(1,256,6,6))
-- model = localizeMemory(mylstm.modified_lstm);
--graph.dot(model.fg, 'model','test') 

function layer:updateOutput(input)
	print("hello,I'm Abhishek Arya")
	print(input)
	return self.modified_lstm:forward(input)
end	