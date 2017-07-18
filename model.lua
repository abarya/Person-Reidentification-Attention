require 'torch';
require 'nn';
require 'cunn'
require 'cutorch'
require 'nngraph'
require 'utilities.lua'
require 'opts.lua'
require 'attention.lua'
require 'rnn'
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

if opt.model_atten_model=="alexnet" then
	input_size=227
else
	input_size=224
end		
-- input dimensions:
local nfeats = 3
local width = input_size
local height = input_size

nnpackage = nn;
local model={}

if opt.cnn_model=="alexnet" then
	require 'loadcaffe'
	pretrained_model=loadcaffe.load('deploy.prototxt','bvlc_alexnet.caffemodel','cudnn')
end	
model_atten={}
function model.attention(num)
	nngraph.setDebug(true)
	print("creating new model " .. LOAD_MODEL_NAME .. '\n')
    print('creating model................................... branch'..string.format(num))

    for k,v in pairs(pretrained_model) do
		--print(v)
		if(type(v)=='table') then
			for i=1,table.getn(v) do
				name=v[i].name:sub(1,4)
				print(v[i].name)
				if(name=="conv" and i==1) then
					model_atten['img_'..string.format(num).. v[i].name]=nnpackage.SpatialConvolution(v[i].nInputPlane, v[i].nOutputPlane, v[i].kW, v[i].kH, v[i].dW, v[i].dH, v[i].padW, v[i].padH)():annotate{
					name='convolution unit  ' .. v[i].name,
					graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_CONV}
				    }
				elseif(name=="conv" and i>1) then
					model_atten['img_'..string.format(num).. v[i].name]=nnpackage.SpatialConvolution(v[i].nInputPlane, v[i].nOutputPlane, v[i].kW, v[i].kH, v[i].dW, v[i].dH, v[i].padW, v[i].padH)(model_atten[prev]):annotate{
					name='convolution unit  ' .. v[i].name,
					graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_CONV}
					}
				elseif(name=="relu") then
					model_atten['img_'..string.format(num).. v[i].name]=nnpackage.ReLU()(model_atten[prev]):annotate{
					name='ReLU unit ' .. v[i].name,
					graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_RELU}
					}
				elseif(name=="pool") then
					model_atten['img_'..string.format(num).. v[i].name]=nnpackage.SpatialMaxPooling(v[i].kW, v[i].kH, v[i].dW, v[i].dH, v[i].padW, v[i].padH)(model_atten[prev]):annotate{
					name='MAXPOOL unit ' .. v[i].name,
					graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_MAXPOOL}
					}	
				elseif(name=="norm") then
					model_atten['img_'..string.format(num).. v[i].name]=nnpackage.SpatialCrossMapLRN(v[i].size,v[i].alpha,v[i].beta,v[i].k)(model_atten[prev]):annotate{
					name='localResponseNormalization unit ' .. v[i].name,
					graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_MAXPOOL}
					}
				elseif(name=="torc") then	
					break	
				end
				if(name=="conv" and num>1) then
					model_atten['img_1' .. v[i].name].data.module:share(model_atten['img_'..string.format(num).. v[i].name].data.module,'weight', 'bias', 'gradWeight', 'gradBias')
				end
				prev='img_'..string.format(num).. v[i].name
			end	
		end	
	end
	---------------------------------------------------------------------------------------
    --for alexnet the dimesnsion of the feature map is 256*6*6
    model_atten['img_'..string.format(num)..'avg_pool']=nnpackage.SpatialAveragePooling(6,6,6,6,0,0)(model_atten['img_'..string.format(num)..'pool5']):annotate{
	name='AVGPOOL unit ',graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_MAXPOOL}
	}	

 	model_atten['img_'..string.format(num)..'view']=nnpackage.View(-1,256)(model_atten['img_'..string.format(num)..'avg_pool']):annotate{
	name='view unit ',graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_MAXPOOL}
	}

	-- for initializing hidden states

    model_atten['img_'..string.format(num)..'mlp1_hid']=nnpackage.Linear(256,512)(model_atten['img_'..string.format(num)..'view']):annotate{
	name='Linear unit mlp1_hid ',graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_LINEAR}
	}

    model_atten['img_'..string.format(num)..'mlp2_hid']=nnpackage.Linear(512,lstm_hiddenstate)(model_atten['img_'..string.format(num)..'mlp1_hid']):annotate{
	name='Linear unit mlp12_hid ',graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_LINEAR}
	}

	-- for initializing cell states

    model_atten['img_'..string.format(num)..'mlp1_cell']=nnpackage.Linear(256,512)(model_atten['img_'..string.format(num)..'view']):annotate{
	name='Linear unit mlp1 ',graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_LINEAR}
	}

    model_atten['img_'..string.format(num)..'mlp2_cell']=nnpackage.Linear(512,lstm_hiddenstate)(model_atten['img_'..string.format(num)..'mlp1_cell']):annotate{
	name='Linear unit mlp12 ',graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_LINEAR}
	}
	model_atten['img_'..string.format(num)..'hid2soft']=nnpackage.Linear(512,6*6)(model_atten['img_'..string.format(num)..'mlp2_hid']):annotate{name='Linearhid to location_map',
	graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_LINEAR}
    }
	model_atten['img_'..string.format(num)..'loc_map']=nnpackage.SoftMax()(model_atten['img_'..string.format(num)..'hid2soft']):annotate{name='softmax_location_map'}

	-----------------Sharing for second and third sub-network

	if(num>1) then
		model_atten['img_1mlp1_hid'].data.module:share(model_atten['img_'..string.format(num)..'mlp1_hid'].data.module,'weight', 'bias', 'gradWeight', 'gradBias')
		model_atten['img_1mlp2_hid'].data.module:share(model_atten['img_'..string.format(num).. 'mlp2_hid'].data.module,'weight', 'bias', 'gradWeight', 'gradBias')
		model_atten['img_1mlp1_cell'].data.module:share(model_atten['img_'..string.format(num).. 'mlp1_cell'].data.module,'weight', 'bias', 'gradWeight', 'gradBias')
		model_atten['img_1mlp2_cell'].data.module:share(model_atten['img_'..string.format(num).. 'mlp2_cell'].data.module,'weight', 'bias', 'gradWeight', 'gradBias')
		model_atten['img_1hid2soft'].data.module:share(model_atten['img_'..string.format(num).. 'hid2soft'].data.module,'weight', 'bias', 'gradWeight', 'gradBias')
	end	

	----------------------------------------Attention using LSTM---------------------------------------------------------

	model_atten['img_'..string.format(num)..'attention']=nn.attention()({model_atten['img_'..string.format(num)..'loc_map'],model_atten['img_'..string.format(num)..'mlp2_cell']
		                                                                ,model_atten['img_'..string.format(num)..'mlp2_hid'],model_atten['img_'..string.format(num)..'pool5']}):annotate{name='attention_module'}
	
	if(num>1) then
		print("model attention share",num)
		atten1=model_atten['img_'..string.format(num-1).. 'attention'].data.module.lstm
		atten2=model_atten['img_'..string.format(num).. 'attention'].data.module.lstm
		for j=1,#atten1.forwardnodes do
		  if(atten1.forwardnodes[j].data.module~=nil and atten1.forwardnodes[j].data.module.weight~=nil) then
		    atten1.forwardnodes[j].data.module:share(atten2.forwardnodes[j].data.module, 'weight','bias','gradWeight','gradBias');
		  end
		end 
	end	
end  

model.attention(1)  
--model.attention(2)
--model.attention(3)
m1=nn.gModule({model_atten['img_1conv1']},--,model_atten['img_2conv1']},--,model_atten['img_3conv1']},
{model_atten['img_1attention']})
 --model_atten['img_2mlp2_hid'],model_atten['img_2mlp2_cell'],model_atten['img_2loc_map'],model_atten['img_2pool5']})
 --model_atten['img_3mlp2_hid'],model_atten['img_3mlp2_cell'],model_atten['img_3loc_map'],model_atten['img_3pool5']})
seq=nn.Sequencer(m1)
return seq