require 'image'
require 'torch'
require 'cutorch'
require 'cunn'
require 'opts.lua'
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