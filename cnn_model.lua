require 'torch';
require 'nn';
require 'io';
require 'lfs'
require 'cunn'
require 'cutorch'
require 'nngraph'
dofile 'opts.lua'
require 'utilities.lua'
require 'rnn'
logger = require 'log'
logger.outfile = opt.logFile
print(opt)
--define fillcolors for different layers
COLOR_CONV = 'cyan';
COLOR_MAXPOOL = 'grey';
COLOR_RELU = 'lightblue';
COLOR_SOFTMAX = 'green';
COLOR_FC = 'orange';
COLOR_AUGMENTS = 'brown';
COLOR_LINEAR='pink'

TEXTCOLOR = 'black';
NODESTYLE = 'filled';
lstm_hiddenstate=512
lstm_inputsize  =256

-- default options to verify cnn_model standalone
if(opt == nil) then
	opt = {}
	opt.cnn_model = "alexnet"
end

if opt.cnn_model=="alexnet" then
	input_size=227
else
	input_size=224
end		
-- input dimensions:
local nfeats = 3
local width = input_size
local height = input_size

nnpackage = nn;
CNN={}

function CNN.cnn()

	nngraph.setDebug(true)
	print("creating new model " .. LOAD_MODEL_NAME .. '\n')
	if opt.cnn_model=="alexnet" then
		require 'loadcaffe'
		pretrained_model=loadcaffe.load('deploy.prototxt','bvlc_alexnet.caffemodel','cudnn')
    end	
    cnn={}
    table.insert(cnn,nn.Identity()():annotate{name='input'})
    print('creating cnn model')
  
	for k,v in pairs(pretrained_model) do
		--print(v)
		if(type(v)=='table') then
			for i=1,table.getn(v) do
				name=v[i].name:sub(1,4)
				print(v[i].name)
				if(name=="conv" and i==1) then
					cnn['img_' .. v[i].name]=nnpackage.SpatialConvolution(v[i].nInputPlane, v[i].nOutputPlane, v[i].kW, v[i].kH, v[i].dW, v[i].dH, v[i].padW, v[i].padH)(cnn[1]):annotate{
					name='convolution unit  ' .. v[i].name,
					graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_CONV}
				    }
				elseif(name=="conv" and i>1) then
					cnn['img_' .. v[i].name]=nnpackage.SpatialConvolution(v[i].nInputPlane, v[i].nOutputPlane, v[i].kW, v[i].kH, v[i].dW, v[i].dH, v[i].padW, v[i].padH)(cnn[prev]):annotate{
					name='convolution unit  ' .. v[i].name,
					graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_CONV}
					}
				elseif(name=="relu") then
					cnn['img_' .. v[i].name]=nnpackage.ReLU()(cnn[prev]):annotate{
					name='ReLU unit ' .. v[i].name,
					graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_RELU}
					}
				elseif(name=="pool") then
					cnn['img_' .. v[i].name]=nnpackage.SpatialMaxPooling(v[i].kW, v[i].kH, v[i].dW, v[i].dH, v[i].padW, v[i].padH)(cnn[prev]):annotate{
					name='MAXPOOL unit ' .. v[i].name,
					graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_MAXPOOL}
					}	
				elseif(name=="norm") then
					cnn['img_' .. v[i].name]=nnpackage.SpatialCrossMapLRN(v[i].size,v[i].alpha,v[i].beta,v[i].k)(cnn[prev]):annotate{
					name='localResponseNormalization unit ' .. v[i].name,
					graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_MAXPOOL}
					}
				elseif(name=="torc") then	
					break	
				end
				-- if(name=="conv" and num>1) then
				-- 	print(subnet1['img1_' .. v[i].name])
				-- 	subnet['img' ..string.format("%d",num) .. '_' .. v[i].name].data.module:share(subnet1['img1_' .. v[i].name].data.module,'weight', 'bias', 'gradWeight', 'gradBias')
				-- end
				prev='img_' .. v[i].name
				--print(prev)
			end	
		end	
	end

	---------------------------------------------------------------------------------------
    --for alexnet the dimesnsion of the feature map is 256*6*6
    cnn['img_avg_pool']=nnpackage.SpatialAveragePooling(6,6,6,6,0,0)(cnn['img_pool5']):annotate{
	name='image AVGPOOL unit ',graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_MAXPOOL}
	}	

 	cnn['img_view']=nnpackage.View(-1,256)(cnn["img_avg_pool"]):annotate{
	name='image view unit ',graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_MAXPOOL}
	}

	-- for initializing hidden states

    cnn['img_mlp1_hid']=nnpackage.Linear(256,512)(cnn["img_view"]):annotate{
	name='Linear unit mlp1_hid ',graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_LINEAR}
	}

    cnn['img_mlp2_hid']=nnpackage.Linear(512,lstm_hiddenstate)(cnn['img_mlp1_hid']):annotate{
	name='Linear unit mlp12_hid ',graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_LINEAR}
	}

	-- for initializing cell states

    cnn['img_mlp1_cell']=nnpackage.Linear(256,512)(cnn["img_view"]):annotate{
	name='Linear unit mlp1 ',graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_LINEAR}
	}

    cnn['img_mlp2_cell']=nnpackage.Linear(512,lstm_hiddenstate)(cnn['img_mlp1_cell']):annotate{
	name='Linear unit mlp12 ',graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_LINEAR}
	}
	cnn['hid_to_softmax']=nnpackage.Linear(512,6*6)(cnn['img_mlp2_hid']):annotate{name='Linearhid to location_map'}
	cnn['soft_location_map']=nnpackage.SoftMax()(cnn['hid_to_softmax']):annotate{name='softmax_location_map'}
	outputs={}
	--cnn['attention']=nn.attention()({cnn['soft_location_map'],cnn['img_mlp2_cell'],cnn['img_mlp2_hid'],cnn['img_pool5']}):annotate{name='attention_module'}
	-- table.insert(outputs,cnn['soft_location_map'])
	-- table.insert(outputs,cnn['img_mlp2_cell'])
	-- table.insert(outputs,cnn['img_mlp2_hid'])
	-- table.insert(outputs,cnn['img_pool5'])
	--print(#outputs)
	final_module=nnpackage.gModule(cnn,{cnn['soft_location_map'],cnn['img_mlp2_cell'],cnn['img_mlp2_hid'],cnn['img_pool5']})
	return final_module
	--print(cnn,outputs)
	--local model=nn.gModule(cnn,outputs)
	--return model
end	

-- function create_model()
-- 	nngraph.setDebug(true)
-- 	print("creating new model " .. LOAD_MODEL_NAME .. '\n')
-- 	if opt.cnn_model=="alexnet" then
-- 		require 'loadcaffe'
-- 		cnn=loadcaffe.load('deploy.prototxt','bvlc_alexnet.caffemodel','cudnn')
--     end	
--     --------------------------------------------------------------------------------------
--     ---------------CNN for feature extraction

-- 	---------------------------------------------------------------------------------------
-- 	--Network for image 1 called as subnet-1
-- 	---------------------------------------------------------------------------------------
-- 	subnet1=cnn(opt.cnn_model,1)

-- 	-------------------------------------------------------------------
--     -- Network for the image-2 (called as subNetwork2
--     --------------------------------------------------------------------------------
--  --    subnet2=create_subnet_cnn(opt.cnn_model,2)

-- 	-- -------------------------------------------------------------------
--  --    -- Network for the image-3 (called as subNetwork3
--  --    --------------------------------------------------------------------------------
--  --    subnet3=create_subnet_cnn(opt.cnn_model,3)

--  --    m3 = nn.JoinTable(1)({subnet1["img1_pool5"],subnet2["img2_pool5"],subnet3["img3_pool5"]}):annotate{
-- 	-- 				      name='Joining unit',
-- 	-- 				      graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_AUGMENTS}
--  --        				 };
--  --    ---------------------------------------------------------------------------------------
--     -----------Attention Model
    

-- 	subnet1['img1_lstm']=nnpackage.LSTM()


    

--     model=localizeMemory(nn.gModule({subnet1["img1_conv1"]},{subnet1['img1_mlp2']}))
--     --model = localizeMemory(nn.gModule({subnet1["img1_conv1"],subnet2["img2_conv1"],subnet3["img3_conv1"]},{m3}));
-- 	model = localizeMemory(model);
-- 	graph.dot(model.fg, 'model')  
-- 	return model  
-- end		

-- model=CNN.cnn()
-- model = localizeMemory(model);
-- graph.dot(model.fg, 'model','test') 

-- output=model:forward(torch.randn(1,3,227,227))
-- print(output[2])

return CNN
	
