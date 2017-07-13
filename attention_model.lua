local CNN=require 'cnn_model.lua'
require 'maptable.lua'
require 'optim'
require 'torch'
require 'cunn'
optimState = {
  learningRate = 0.4,
  weightDecay = 0.3,
  momentum =0.2,
  learningRateDecay = 0.1,
}
m=CNN.cnn()
model=nn.maptable(m,true,3)
print("hello")
parameters,gradParameters=model:getParameters()
print(parameters[{{1,10}}],parameters:size())
input={torch.randn(1,3,227,227),torch.randn(1,3,227,227),torch.randn(1,3,227,227)}

-- print(model.modules[1])
-- print(model.modules[2])
-- print(model.modules[3])
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
  --model:clearState()
  print('After update')
  parameters,gradParameters=model:getParameters()
  print(parameters[{{1,10}}],parameters:size())
end  