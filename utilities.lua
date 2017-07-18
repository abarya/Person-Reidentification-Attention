require 'image'
require 'torch'
require 'cutorch'
require 'cunn'
require 'opts.lua'
local logger=require 'log'
function localizeMemory(tensor)
  if(opt.useCuda) then
     newTensor = cudaTheData(tensor)
  else
    newTensor = tensor;
  end
  
  return newTensor;
end

function cudaTheData(data)
    local clonedData = nil
    
    -- if the data is really a table, then recurse into table and cuda the data
    -- remember that, the criterions and models in torch are even table, 
    -- but torch.typename will give valid class labels for them.
    -- so, we can differentiate the real tables with torch.typename
    if('table' == type(data) and (nil == torch.typename(data))) then
        clonedData = {}
        for i, internalData in ipairs(data) do
            clonedData[i] = cudaTheData(internalData)
        end
    else
        clonedData = data:cuda()
    end
    return clonedData
end

function getAllFileNamesInDir(directory)

    index = 1;
    filesOrFolderNames = {};
    filesOrFolderPaths = {};
    
    if(lfs.attributes(directory) == nil) then
      logger.warn('given dorectory ' .. directory .. ' does not exist!');
      return {}, {};
    end
    
    for file in lfs.dir(directory) do
        -- print (file )
        if(file:sub(1,1) ~= '.' and file ~= '..') then
            filesOrFolderNames[index] = file;
            filesOrFolderPaths[index] = directory .. '/' .. file;
            index = index + 1;
        end
    end

    return filesOrFolderNames, filesOrFolderPaths;
end

function loadAllImagesFromFolders(fileNames, filePaths)
  loadedfiles = {};
--[[]    
    for i, foldername in ipairs(fileNames) do
        local hashKey = i .. '-' .. foldername;
        print(foldername .. ' --> ' .. filePaths[i])
        logger.trace('\tcollecting all Image file paths from '.. filePaths[i])
        allImgNames, allImgPaths = getAllFileNamesInDir(filePaths[i]);
        
        local imgTable = {};
        for j, imgname in ipairs(allImgPaths) do 
           local loadImageWithScale = loadImageWithScale
           local scale = opt.scale
           
           t:addjob(
                function() 
                    local loadedImg = (loadImageWithScale(imgname, scale))
                    return loadedImg
                end,
                
                function(img)
                    imgTable[imgname] = img; 
                end
           )
        end
        logger.trace('\tNumber of images : '.. #allImgPaths)
        loadedfiles[hashKey] = imgTable;
    end
    
    t:synchronize()
    return loadedfiles
    --]]

  loadedfiles = {}; 
    for i, foldername in ipairs(fileNames) do
        print(foldername .. ' --> ' .. filePaths[i])
        logger.trace('\tcollecting all Image file paths from '.. filePaths[i])
        allImgNames, allImgPaths = getAllFileNamesInDir(filePaths[i]);
        
        imgTable = {};
        for j, imgname in ipairs(allImgPaths) do 
           --print(imgname)
           img = (image.load(imgname)) -- '1' --
           imgTable[imgname] = img; 
        end
        logger.trace('\tNumber of images : '.. #allImgPaths,'image num='..i)
        loadedfiles[foldername] = imgTable;
    end
    
    return loadedfiles
    
end

--[[
   
   name: table.map_length
   @param
   @return the number of keys in the table (i.e., effectively the length of the table)
   
]]--

function table.map_length(t)
    local c = 0
    for k,v in pairs(t) do
         c = c+1
    end
    return c
end

--[[
   
   name: table.getAllKeys
   @param
   @return the number of keys in the table (i.e., effectively the length of the table)
   
]]--

function table.getAllKeys(tbl)
    local keyset={}
    local n=0

    for k,v in pairs(tbl) do
      n=n+1
      keyset[n]=k
    end
    
    table.sort(keyset)
    return keyset;
end

--[[
   
   name: table.getValues
   @param
   @return get all the values for given keys
   
]]--

function table.getValues(tbl, keys)
    local values={}
    local n=0

    for index = 1, keys:size(1) do
      values[index]=tbl[keys[index]]
    end
    
    table.sort(values)
    return values;
end


--[[
   
   name: getRandomNumber
   @param
   @return a random number between lowe and upper, but without the number in exclude
   
]]--

function getRandomNumber(lower, upper, exclude)
    randNumber = math.random(lower, upper);
    while(randNumber == exclude) do
        randNumber = math.random(lower, upper);
    end
    return randNumber;
end

--[[

   name: getArrangedInputsForNGPUs 
   @param
   @return split the data into batches based on number of GPUs used
]]--

function getArrangedInputs(data, target, nGPUs, beginIndex, endIndex, randomOrder)
    if(opt.cnn_model=="alexnet") then
      image_size=227
    end  
    input1=torch.CudaTensor(opt.batchsize,3,image_size,image_size)
    input2=torch.CudaTensor(opt.batchsize,3,image_size,image_size)
    input3=torch.CudaTensor(opt.batchsize,3,image_size,image_size)
    label1=torch.CudaTensor(opt.batchsize)
    label2=torch.CudaTensor(opt.batchsize)
    label3=torch.CudaTensor(opt.batchsize)
    num=1
    for i=beginIndex,endIndex do
        input1[num]=data[i][1]:cuda()
        input2[num]=data[i][2]:cuda()
        input3[num]=data[i][3]:cuda()
        label1[num]=target[i][1]
        label2[num]=target[i][2]
        label3[num]=target[i][3]
        num=num+1
    end         
    return {input1,input2,input3},{label1,label2,label3}
end

--[[
   
   name: sliceRange
   @param
   @return calculate the range of indices needed based on index and number of total splits
   
]]--
function sliceRange(nElem, idx, splits)
    -- calculate the count of common elements for all the GPUs
   local commonEltsPerMod = math.floor(nElem / splits)
   
    -- calculate the count of reamining elements for which the element-count shall be commonEltsPerMod + 1
   local remainingElts = nElem - (commonEltsPerMod * splits)
   
   -- according to current idx, how much "commonEltsPerMod + 1" elements are there?
   local commonPlusOneEltsCount = math.min(idx - 1, remainingElts)
   -- according to current idx, how much "commonEltsPerMod" elements are there?
   local commonEltsCount = (idx - 1) - commonPlusOneEltsCount 
   
   -- determine the start index
   local rangeStart = (commonPlusOneEltsCount * (commonEltsPerMod + 1)) + 
                        (commonEltsCount * commonEltsPerMod) + 1
                        
    -- determine the total elements for current index
   local currentElts = commonEltsPerMod
   if(idx <= remainingElts) then currentElts = commonEltsPerMod + 1 end

    -- return start index and elements count
   return rangeStart, currentElts
end
