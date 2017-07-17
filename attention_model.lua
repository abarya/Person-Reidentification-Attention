--local CNN=require 'cnn_model.lua'
local model=require 'model.lua'
require 'optim'
require 'torch'
require 'cunn'
optimState = {
  learningRate = 0.4,
  weightDecay = 0.3,
  momentum =0.2,
  learningRateDecay = 0.1,
}
parameters,gradParameters=model:getParameters()
print(parameters[{{1,10}}],parameters:size())
input={torch.randn(1,3,227,227),torch.randn(1,3,227,227),torch.randn(1,3,227,227)}
--print(torch.cat(torch.cat(parameters[{{1,10}}],parameters[{{6149576+1,6149576+10}}],2),parameters[{{12299152+1,12299152+10}}],2))
print("before update")
model:forward(input)
os.exit()
for i=1,10 do
  local feval = function(x)
    gradParameters:zero()
    print("forward")
    local outputs = model:forward(input)
    local f = torch.randn(1536)
    --print(target)
    local df_do = {torch.randn(1,1536),torch.randn(1,1536),torch.randn(1,1536)}
    print("backward")
    model:backward(input, df_do)
    return f,gradParameters
  end
  optim.sgd(feval, parameters, optimState)
  print(torch.cat(torch.cat(parameters[{{1,10}}],parameters[{{6149576+1,6149576+10}}],2),parameters[{{12299152+1,12299152+10}}],2))
  model:clearState()
  print('After update')
  parameters,gradParameters=model:getParameters()
  print(model)
  os.exit()
  end  