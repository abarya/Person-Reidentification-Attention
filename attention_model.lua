require 'attention.lua'

atten=nn.attention() 

f=atten:forward(torch.ones(1,3,227,227))

gradOutput=torch.randn(1,1536)
atten:backward(input,gradOutput)