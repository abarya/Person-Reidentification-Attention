-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.

require 'nn';

local L2Distance, parent = torch.class('nn.L2Distance', 'nn.Module')

function L2Distance:__init()
   parent.__init(self)   
end

function L2Distance:updateOutput(input)
	local N = input[1]:size(1)
	self.inputDifference = torch.csub(input[1], input[2])
	local squaredDifference = torch.pow(self.inputDifference, 2)
	self.output = torch.zeros(1):typeAs(input[1])
	self.output[1] = torch.sum(squaredDifference) / N
	return self.output
end

function L2Distance:updateGradInput(input, gradOutput)
	local N = input[1]:size(1)
	self.gradInput = {}
	self.gradInput[1] = 2 * self.inputDifference * gradOutput[1]/N
	self.gradInput[2] = -2 * self.inputDifference * gradOutput[1]/N
	return self.gradInput
end
