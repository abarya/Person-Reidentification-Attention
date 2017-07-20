require 'torch';
require 'nn';
require 'cunn'
require 'cutorch'
require 'nngraph'
dofile 'utilities.lua'
dofile 'opts.lua'
require 'attention'
require 'L2Distance'
require 'rnn'
require 'cudnn'

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


function create_base_model()
	nngraph.setDebug(true)
	print("creating new model " .. LOAD_MODEL_NAME .. '\n')
  layers = {}

  --copy the pretrained weights to the new model
  local pre_trained_model=torch.load('saved_model/finetuned_alexnet.t7'):cuda()--way to load model
  pre_trained_model:remove(23)
  pre_trained_model:remove(22)
  pre_trained_model:remove(21)
  pre_trained_model:remove(20)
  pre_trained_model:remove(19)
  pre_trained_model:remove(18)
  pre_trained_model:remove(17)
  pre_trained_model:remove(16)
  
  p1,g1 = pre_trained_model:getParameters()
  local input = nn.Identity()()
  layers['pretrained'] = pre_trained_model(input)

	---------------------------------------------------------------------------------------
  --for alexnet the dimesnsion of the feature map is 256*6*6
  layers['avg_pool']=nnpackage.SpatialAveragePooling(6,6,6,6,0,0)(layers['pretrained']):annotate{
    name='AVGPOOL unit ',graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_MAXPOOL}
	}	

 	layers['view1']=nnpackage.View(-1,256)(layers['avg_pool']):annotate{
    name='view unit ',graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_MAXPOOL}
	}

	-- for initializing hidden states
  layers['linear0_h0']=nnpackage.Linear(256,512)(layers['view1']):annotate{
    name='Linear unit mlp1_hid ',graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_LINEAR}
	}

  layers['linear1_h0'] = nnpackage.Linear(512,lstm_hiddenstate)(layers['linear0_h0']):annotate{
    name='Linear unit mlp12_hid ',graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_LINEAR}
	}

	-- for initializing cell states
  layers['linear0_c0'] = nnpackage.Linear(256,512)(layers['view1']):annotate{
    name='Linear unit mlp1 ',graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_LINEAR}
	}

  layers['linear1_c0'] = nnpackage.Linear(512, lstm_hiddenstate)(layers['linear0_c0']):annotate{
    name='Linear unit mlp12 ',graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_LINEAR}
	}
  
	layers['linear1_h2s'] = nnpackage.Linear(512,6*6)(layers['linear1_h0']):annotate{
    name='Linearhid to location_map',	graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_LINEAR}
  }
  
	layers['softmap'] = nnpackage.SoftMax()(layers['linear1_h2s']):annotate{
    name='softmax_location_map'
  }

	----------------------------------------Attention using LSTM---------------------------------------------------------

	layers['final'] = nn.attention()({layers['softmap'], 
                                      layers['linear1_c0'],
		                                  layers['linear1_h0'],
                                      layers['pretrained']}):annotate{ name='attention_module' }

  local model = nn.gModule({input}, {layers['final']})
  graph.dot(model.fg, "personreid_attention_base_model", "personreid_attention_base_model")
  return model
end  

function create_final_model()

  local input = nn.Identity()()
  seq_model = nn.Sequencer(create_base_model())(input)

  person1 = nn.SelectTable(1)(seq_model)
  person2 = nn.SelectTable(2)(seq_model)
  person3 = nn.SelectTable(3)(seq_model)

  person1_desc = nn.SelectTable(1)(person1)
  person1_idscore = nn.SelectTable(2)(person1)
  person2_desc = nn.SelectTable(1)(person2)
  person2_idscore = nn.SelectTable(2)(person2)
  person3_desc = nn.SelectTable(1)(person3)
  person3_idscore = nn.SelectTable(2)(person3)

  --new_model = nn.gModule({input}, {person1_desc, person2_desc, person3_desc, 
  --                               person1_idscore, person2_idscore, person3_idscore})

  p1_p2_distance = nn.L2Distance()({person1_desc, person2_desc})
  p2_p3_distance = nn.L2Distance()({person2_desc, person3_desc})
  difference = nn.CSubTable()({p1_p2_distance, p2_p3_distance})

  local final_model = nn.gModule({input}, {person1_idscore, person2_idscore, person3_idscore, difference})

  return final_model:cuda()
end
--torch.save('saved_model/attention_model.t7', seq_model)
--]]
