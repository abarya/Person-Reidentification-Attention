require 'xlua'
require 'optim'
require 'nn'
require 'torch'
require 'pl'
require 'paths'
require 'loadcaffe'
require 'cunn'
require 'image'
local c = require 'trepl.colorize'
opt = lapp[[
    --save               (default "logs")
    -b,--batchSize       (default 128)              batch size
    -r,--learningRate    (default .1)              learning rate
    --gamma              (default .0001)            learning rate decay using inverse policy
    --p                  (default 0.75)             power
    --optimization       (default 'SGD')
    --weightDecay        (default 0.0005)           weightDecay
    --learningRateDecay  (default 1e-4)
    -m,--momentum        (default 0.9)              momentum
	-m,--model           (default alexnet)          model name
	-t,--type            (default cuda)            float/cuda
]]
--opt.model='alexnet'
print(opt.model,opt.type,opt.batchSize)

local function cast(t)
   if opt.type == 'cuda' then
      require 'cunn'
      return t:cuda()
   elseif opt.type == 'float' then
      return t:float()
   elseif opt.type == 'cl' then
      require 'clnn'
      return t:cl()
   else
      error('Unknown type '..opt.type)
   end
end

print(c.blue '==>' ..' configuring model')

print(c.blue'==>' ..' setting criterion')
criterion = cast(nn.CrossEntropyCriterion())

print(c.blue'==>' ..' configuring optimizer')

confusion = optim.ConfusionMatrix(871)

local datapath_file = io.open("datasets/cuhk01_test100.txt")
image_paths={}
labels={}
if datapath_file then
  i=0
  for line in datapath_file:lines() do
    local path,label=unpack(line:split(" "))
    image_paths[i+1]=path
    labels[i+1]=label
    i=i+1
  end
end  

pretrained_model = 'alexnet'
img_size = 227
if(pretrained_model == 'alexnet') then
	net = loadcaffe.load('deploy.prototxt', 'bvlc_alexnet.caffemodel', 'cudnn')
	--reinitialize the weights for fc layers and change the last fc layer
	fc1=net:get(17)
	--print(fc1.weight[1])
	fc1:reset()
	fc2=net:get(20)
	fc2:reset()
	net:remove(23)
	net:insert(nn.Linear(4096,871):cuda(),23)
	net:remove(24)
	print(net)
	parameters,gradParameters = net:getParameters()
	mean_image=image.load("mean_image.png"):cuda() * 255
  temp=mean_image[3]:clone()
  mean_image[3]=mean_image[1]
  mean_image[1]=temp 
  img_size = 227
else
	net = loadcaffe.load('VGG_ILSVRC_16_layers_deploy.prototxt', 'VGG_ILSVRC_16_layers.caffemodel', 'cudnn')
	print(net)
  --net:get(33):reset()
  --net:get(36):reset()
  net:remove(39)
  net:insert(nn.Linear(4096, 871):cuda(), 39)
  net:remove(40)
  print(net)
	parameters,gradParameters = net:getParameters()
	mean_image=torch.zeros(3,224,224):cuda()
	mean_image[1] = torch.Tensor({103.939}):repeatTensor(224,224)
	mean_image[2] = torch.Tensor({116.779}):repeatTensor(224,224)
	mean_image[3] = torch.Tensor({123.68}):repeatTensor(224,224)
  img_size = 224
end

lrs = torch.zeros(parameters:size(1)):fill(0.001)
lrs[{{parameters:size(1) - (4096*871 + 871) + 1, parameters:size(1)}}] = 1

optimState = {
  learningRates = lrs,
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
}
print(optimState)

epoch=0
--local trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
function train()
  net:training()
  epoch = epoch+1
  print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' ..opt.batchSize .. ']')
  local target = cast(torch.FloatTensor(opt.batchSize))
  local indices = torch.randperm(#labels):long():split(opt.batchSize)
  indices[#indices] = nil
  for t,v in ipairs(indices) do
    input=torch.CudaTensor(opt.batchSize,3,img_size,img_size)
    for i=1,opt.batchSize do
      target[i]=labels[v[i]]
      img=image.scale(image.load(image_paths[v[i]]), img_size, img_size):cuda() * 255
      temp=img[3]:clone()
      img[3]=img[1]
      img[1]=temp
      input[i]=img - mean_image
    end  
    input=input:cuda()
    target=target:cuda()  
    local feval = function(x)
      gradParameters:zero()
      local outputs = net:forward(input)
      local f = criterion:forward(outputs, target)
      --print(target)
      local df_do = criterion:backward(outputs, target)
      net:backward(input, df_do)
      confusion:batchAdd(outputs, target)
      confusion:updateValids()
      --print(outputs:size())
      print('epoch ',epoch,' loss ',f,'train accuracy  ',confusion.totalValid * 100)
      --trainLogger:add{[' % loss'] = f}
      return f,gradParameters
    end
    optim.sgd(feval, parameters, optimState)
    --print(fc1.weight[1])
  end
  
  --trainLogger:plot()
  confusion:updateValids()
  --print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
        --confusion.totalValid * 100, torch.toc)

  train_acc = confusion.totalValid * 100
  confusion:zero()
  print(c.blue'==>' ..' saving model')
  torch.save('saved_model/finetuned_'.. pretrained_model.. '.net',net)
end

for i=1,40 do
  train()
end  

