require 'nn'
require 'opts.lua'
require 'utilities.lua'
require 'nngraph'
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
    self.core = LSTM.lstm(self.rnn_size, self.rnn_size, self.num_layers, dropout)
    --self:_createInitState(1)
    self:_join_layers(1)
    self:create_clones()
    self:shareClones()
end

function layer:_join_layers(batchsize)
	
	model={}	--This model will take as input a 3D feature map from the conv net and the hidden and cell state from the previous layer
	---For LSTM, the input is given as a table of 3 tensors with dimension (batchsize,input)
	---This model will take two inputs..one of them comes from cnn and the other from the output of the previous time step
	--- from the cnn, we get  a table of three tensors   --{conv_feat,initial_hidden_st,initial_cell_st}
	table.insert(model,nn.Identity()():annotate{name='conv_feat'})
	table.insert(model,nn.Identity()():annotate{name='o/p from previous time step'})--this is 1d tensor
	table.insert(model,nn.Identity()():annotate{name='previous hidden state'})
	table.insert(model,nn.Identity()():annotate{name='previous cell state'})
	--converting to 2D and 
	reshaped=nn.Reshape(self.cnn_op_size,self.cnn_op_size)(model[2])
	replicated=nn.Replicate(self.cnn_op_depth)(reshaped)
	product=nn.CMulTable()({model[1],replicated})
	avg_pool=nn.SpatialAveragePooling(self.cnn_op_size,self.cnn_op_size,self.cnn_op_size,self.cnn_op_size,0,0)--(product)
	avg_pool.divide=false
	input_lstm=avg_pool(product)
	output={model[1],input_lstm,model[3],model[4]} --conv_feat,input,hidden state,cel state
	gmodel=nn.gModule(model,output)
	extended_model=nn.Sequential()
	extended_model:add(gmodel)
	extended_model:add(self.core)
	self.modified_lstm=extended_model
end

function layer:create_clones()
	self.lstm_units={self.modified_lstm}
	for t=1,self.seq_length do
        self.lstm_units[t] = self.modified_lstm:clone('weight', 'bias', 'gradWeight', 'gradBias')
    end
end

function layer:shareClones()
    if self.lstm_units == nil then self:create_clones(); return; end
    print('resharing clones inside the attention model')
    self.lstm_units[1] = self.core
    for t=1,self.seq_length do
        self.lstm_units[t]:share(self.core, 'weight', 'bias', 'gradWeight', 'gradBias')
    end
end



mylstm=nn.modified_lstm()
print(mylstm.modified_lstm)
model = localizeMemory(mylstm.modified_lstm);
--graph.dot(model.fg, 'model','test') 