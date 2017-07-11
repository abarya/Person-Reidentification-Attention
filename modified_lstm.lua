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
    self:_join_layers(1)
end

function layer:_join_layers(batchsize)
	
	model={}
	self.modified_lstm=self.core
end
