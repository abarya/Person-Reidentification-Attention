require 'pl'
require 'lfs'
opt=lapp[[
	-- usecuda           	   (default true)         		true/false
	--optimization             (default 'sgd')        		CG  | LBFGS  |  SGD   | ASGD
	-r,--learningRate          (default 1)            		learning rate
    --learningRateDecay        (default 1e-7)         		learning rate decay
    --weightDecay              (default 0.0005)       		weightDecay
    -m,--momentum              (default 0.9)          		momentum
    --dataset                  (default 'others')           cuhk03  |  others
    --datasetname              (default 'cuhk01_test100')
    --cnn_model                (default "alexnet")          alexnet |  vgg
    --hidden_size              (default 512)                hidden layer size in lstm
    --num_layers               (default 1)					mumber of layers of lstm
    --drop_out                 (default 0)
    --seq_length               (default 8)                  num of time steps for which attention model should run
    --batchsize                (default 1)
    --cnn_op_size              (default 6)
    --cnn_op_depth             (default 256)
]]
rootLogFolder = paths.concat(lfs.currentdir() .. '/../', 'scratch', opt.dataset) 
opt.save = paths.concat(rootLogFolder, os.date("%d-%b-%Y-%X-") .. 'personreid_' .. opt.datasetname)
LOAD_MODEL_NAME = paths.concat(opt.save,'_' .. opt.datasetname) 
opt.logFile = paths.concat(opt.save,opt.datasetname .. '.log')

