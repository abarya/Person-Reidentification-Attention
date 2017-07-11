
-- require 'nn'

-- new_model=nn.Sequential()
-- new_model:add(nn.Linear(10,5))
-- new_model:add(nn.ReLU())
-- new_model:add(nn.Linear(5,5))
-- new_model:add(nn.Linear(5,1))

-- new_parameters,new_grad=new_model:getParameters()
-- print(new_parameters:size())
-- model=torch.load('model_sample.net')

-- params,grad=model:getParameters()

-- new_parameters[{{1,85}}]=params
-- print(new_parameters:size())

-- require 'nn'
-- require 'nngraph'
-- h1 = nn.Linear(20, 10)()
-- h2 = nn.Linear(10, 1)(nn.Tanh()(nn.Linear(10, 10)(nn.Tanh()(h1))))
-- mlp = nn.gModule({h1}, {h2})

-- x = torch.rand(20)
-- dx = torch.rand(1)
-- mlp:updateOutput(x)
-- mlp:updateGradInput(x, dx)
-- mlp:accGradParameters(x, dx)

-- -- draw graph (the forward graph, '.fg')
-- graph.dot(mlp.fg, 'MLP')

-- require 'nngraph'
-- require 'nn'
-- require 'rnn'
-- require 'utilities.lua'
-- require 'cunn'
-- require 'cutorch'
-- -- nn.FastLSTM.usenngraph = true -- faster
-- -- nn.FastLSTM.bn = opt.bn
-- -- rnn = nn.FastLSTM(inputsize, hiddensize)
-- inputSize=5
-- outputSize=5
-- rho=5
-- lstm=nn.LSTM(inputSize, outputSize,3)():annotate{
-- 	name='image1 Linear unit mlp12 ',graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = 'grey'}
-- 	}
-- li=nn.Linear(outputSize,10)(lstm):annotate{
-- 	name='Linear unit mlp12 ',graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = 'grey'}
-- 	}
-- model=localizeMemory(nn.gModule({lstm},{li}))
-- --model = localizeMemory(nn.gModule({subnet1["img1_conv1"],subnet2["img2_conv1"],subnet3["img3_conv1"]},{m3}));
-- model = localizeMemory(model);
-- graph.dot(model.fg, 'model')  
-- input=torch.randn(6,5)
-- hid1=torch.ones(5)
-- hid2=torch.ones(5)
-- hiddenstate={hid1,hid2}


-- --lstm:setHiddenState(1,hiddenstate)
-- print(model:forward(input:cuda()))
-- print(model.step)
-- print(model:getHiddenState(1,input)[1])
-- --print(lstm.userPrevOutput)
require 'nngraph'
require 'nn'
require 'rnn'
dofile 'lstm_modified.lua'
require 'utilities.lua'
-- nn.FastLSTM.usenngraph = true -- faster
-- nn.FastLSTM.bn = opt.bn
-- rnn = nn.FastLSTM(inputsize, hiddensize)
inputSize=5
outputSize=5
rho=5
-- lstm=nn.LSTM(inputSize, outputSize,3)():annotate{
-- 	name='image1 Linear unit mlp12 ',graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = 'grey'}
-- 	}
-- li=nn.Linear(outputSize,10)(lstm):annotate{
-- 	name='Linear unit mlp12 ',graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = 'grey'}
-- 	}
-- model=localizeMemory(nn.gModule({lstm},{li}))
-- --model = localizeMemory(nn.gModule({subnet1["img1_conv1"],subnet2["img2_conv1"],subnet3["img3_conv1"]},{m3}));
-- model = localizeMemory(model);
-- graph.dot(model.fg, 'model')  
lstm=nn.LSTM(inputSize, outputSize,3)

input=torch.randn(6,5)
hid1=torch.ones(5)
hid2=torch.ones(5)
hiddenstate={hid1,hid2}


--lstm:setHiddenState(1,hiddenstate)
print(lstm:forward(input))
print(lstm.step)
print(lstm:getHiddenState(1,input)[1])
--print(lstm.userPrevOutput)


