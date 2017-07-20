dofile 'model.lua'
require 'optim'
require 'torch'
require 'cunn'
require 'cutorch'
require 'L2Distance'
local logger=dofile 'log.lua'
dofile 'utilities.lua'
dofile 'opts.lua'
require 'xlua'
--torch.setdefaulttensortype('torch.CudaTensor')

--model=model:cuda()

--lrs = torch.zeros(p2:size(1)):fill(1)
--lrs[{{1,p1:size(1)}}] = .01
optimState = {
  --learningRates = lrs,
  learningRate = 0.001,
  weightDecay = 0.0005,
  momentum =0.9,
  learningRateDecay = 0.0001,
}


model = create_final_model()

--get data before training
fileNames, filePaths = getAllFileNamesInDir(opt.datapath);

trainfiles = loadAllImagesFromFolders(fileNames, filePaths)

trainData = {
  data = trainfiles,
  size = function() return table.map_length(trainfiles) end
}
logger_file = optim.Logger('logs/train'..sys.clock()..'.log')

confusion = optim.ConfusionMatrix(871)

function crit(out,target)
  local cross_entropy = nn.CrossEntropyCriterion()
  pc = nn.ParallelCriterion():add(cross_entropy):add(cross_entropy):add(cross_entropy):cuda()
  iden_ip = {out[1],out[2],out[3]}
  l_iden = pc:forward(iden_ip, target)
  
  --l_iden = l_iden/3
  dli_do1,dli_do2,dli_do3=unpack(pc:backward(iden_ip,target))
  
  criterion = nn.MarginCriterion():cuda()
  criterion.sizeAverage=false
  
  target_class = torch.ones(out[4]:size()) * -1
  target_class = target_class:cuda()
  l_trip=criterion:forward(out[4], target_class)
  df_do=criterion:backward(out[4], target_class)
    
  local l_total=l_iden+l_trip
  --update confusion matrix
  confusion:batchAdd(out[1][2], target[1])
  confusion:batchAdd(out[2][2], target[2])
  confusion:batchAdd(out[3][2], target[3])
  confusion:updateValids()
  return l_total, {dli_do1, dli_do2, dli_do3, df_do} 
end  

function training(epoch)

  --epoch tracker
  epoch =epoch or 1

  local time=sys.clock()

  -- set model to training mode (for modules that differ in training and testing, like Dropout)
  --model:training()

  -- get the handles for parameters and gradient parameters
  parameters,gradParameters = model:getParameters()

  datanames = table.getAllKeys(trainData['data']);
  allTrainData = trainData['data'];

  -- do one epoch
  logger.trace('==> doing epoch on training data:')
  logger.trace("==> online epoch # " .. epoch)

  totalTrainPersons = #datanames;
  print(totalTrainPersons,"toral persons")
  -- create input triplet combinations
  local inputs = {}
  local targets = {}
  totalFiles=0
  random_perm=torch.randperm(871)
  -- for all the training identites, create triplets 
  for t=1,totalTrainPersons do
    --get current train sequence & collect determine all positive and negative sequences
    currentSequenceName = datanames[random_perm[t]]; 

    --insert all positive samples in inputs, targets
    allImgNames = table.getAllKeys(allTrainData[currentSequenceName])
    for i = 1, #allImgNames do
      -- load new sample
      anchorImage = allTrainData[currentSequenceName][allImgNames[i]];

      --for the positive pair
      for j = i+1, #allImgNames do
          positiveImage = allTrainData[currentSequenceName][allImgNames[j]];
          
          -- for the negative pair
          neg_imgNum=getRandomNumber(1, #datanames, random_perm[t])
          negativeSampleSequence = allTrainData[datanames[neg_imgNum]];
          allNegImgNames = table.getAllKeys(negativeSampleSequence)
          negativeImage = negativeSampleSequence[allNegImgNames[getRandomNumber(1, #allNegImgNames)]]; 

          table.insert(inputs, {positiveImage,anchorImage,negativeImage})
          table.insert(targets,{random_perm[t],random_perm[t],neg_imgNum})

          totalFiles = totalFiles + 1;
          end  
      end

    end
  print(totalFiles)


  ---- after generating all the triplets, split them into batch size of opt.batchSize
  --   then for each batch size of opt.batchSize, do the stochastic gradient descent

  totalSamples = table.map_length(targets)
  totalBatches = totalSamples / opt.batchsize;

  -- if the totalSamples count is not divisble by opt.batchSize, then add +1
  if(totalSamples % opt.batchsize ~= 0) then
      totalBatches = math.floor(totalBatches)
  end
      
  logger.debug('total pairs of training samples : ' .. totalSamples .. ' (total : ' .. totalFiles .. 'total batches: ' .. totalBatches)
  
  -- randomize the generated inputs and outputs
  randomOrder = torch.randperm(totalSamples)

  for batchIndex = 0, totalBatches - 1 do
    -- disp progress
    --xlua.progress(batchIndex + 1, totalBatches)
    
    -- find the batchsamples start index and end index
    time = sys.clock()
    local batchStart = (batchIndex  * opt.batchsize) + 1
    local batchEnd = ((batchIndex + 1)  * opt.batchsize);

    -- make sure that index do not exceed the end index of the totalSamples
    if(batchEnd > totalSamples) then
        batchEnd = totalSamples
    end

    local currentBatchSize = batchEnd - batchStart + 1;
    local error=0
    -- create closure to evaluate f(X) and df/dX
    local feval = function(x)
        
        -- get new parameters
        if x ~= parameters then
            parameters:copy(x)
        end

        -- reset gradients
        model:zeroGradParameters()

        -- f is the average of all criterions
        local f = 0
        currentTotalImagesTrained = 0;
        input,target = getArrangedInputs(inputs, targets, opt.nGPUs, batchStart, batchEnd, randomOrder)
        -- estimate f
        local output = model:forward(input)

        local loss,dl_do=crit(output,target) 

        f = f + loss
        error=f

        model:backward(input,dl_do)
        
        currentTotalImagesTrained = currentTotalImagesTrained + currentBatchSize

        -- -- update confusion
        -- if(opt.nGPUs > 1) then
        --     confusion:batchAdd(output, target)
        -- else
        --     confusion:add(output, target)
        -- end
        --for i = batchStart, batchEnd do

        -- normalize gradients and f(X)
        --print('total images : ' .. currentTotalImagesTrained)
        --gradParameters:div(currentTotalImagesTrained)
        --f = f/currentTotalImagesTrained
        -- return f and df/dX
        return f,gradParameters
    end
    optim.sgd(feval, parameters, optimState)

    --train accuracy
    train_acc = confusion.totalValid * 100

    if (batchIndex % 20 == 0) then
      logger_file:add{['training error'] = error}
      logger.trace('error = '..error..', epoch = '..epoch..' , iterations = '..batchIndex)
      logger.trace('train accuracy = '..train_acc)
    end  
    if(batchIndex % 500 == 0) then
      --print confusion matrix
      logger.trace('batches done',batchIndex)   
    end  
  end    
end
for i=1,25 do
  training(i)
end  

logger_file:style{['training error'] = '-'}
logger_file:plot()


