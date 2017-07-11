require 'attention.lua'
-- local CNN=require 'cnn_model.lua'
-- require 'modified_lstm.lua'
-- modified_lstm=nn.modified_lstm()

-- cnn=CNN.cnn()

-- attention_model=nn.Sequential()

-- attention_model:add(cnn)
-- print(attention_model:forward(torch.ones(1,3,227,227)))
-- attention_model:add(modified_lstm)

-- print(attention_model:forward(torch.ones(1,3,227,227)))

atten=nn.attention() 

f=atten:forward(torch.ones(1,3,227,227))

gradOutput=torch.randn(1,1536)
atten:backward(input,gradOutput)