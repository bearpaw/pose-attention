require 'paths'
paths.dofile('ref.lua')     -- Parse command line input and do global variable initialization
paths.dofile('data.lua')    -- Set up data processing
paths.dofile('model.lua')   -- Read in network model
paths.dofile('train.lua')   -- Load up training/testing functions

local DataParallelTable = nn.DataParallelTable
-- First remove any DataParallelTable
if torch.type(model) == 'nn.DataParallelTable' then
   model = model:get(1)
end

-- Wrap the model with DataParallelTable, if using more than one GPU
if opt.nGPU > 1 then
   local gpus = torch.range(1, opt.nGPU):totable()
   local fastest, benchmark = cudnn.fastest, cudnn.benchmark

   local dpt = DataParallelTable(1, true, true)
      :add(model, gpus)
      :threads(function()
         local cudnn = require 'cudnn'
         local nngraph = require 'nngraph'  -- wyang: to work with nngraph on multi-GPUs
                                        -- https://github.com/torch/cunn/issues/241
         cudnn.fastest, cudnn.benchmark = fastest, benchmark
      end)
   dpt.gradInput = nil

   model = dpt:cuda()
end

if opt.optimState ~= 'none' then
   print('==> Load optimState')
   optimState = torch.load(opt.optimState)
   print(optimState)
end

torch.setnumthreads(4)
-- cutorch.setDevice(opt.GPU) 

local Dataloader = require 'dataloader'
loader = Dataloader.create(opt)

isFinished = false -- Finish early if validation accuracy plateaus, can be adjusted with opt.threshold

-- Main training loop
for i=1,opt.nEpochs do
    train()
    valid()
    collectgarbage()
    epoch = epoch + 1
    if isFinished then break end
end

-- Update options/reference for last epoch
opt.lastEpoch = epoch - 1
torch.save(opt.save .. '/options.t7', opt)

-- Generate final predictions on validation set
if opt.finalPredictions == 1 then predict() end

-- Save model
model:clearState()
torch.save(paths.concat(opt.save,'final_model.t7'), model)
torch.save(paths.concat(opt.save,'optimState.t7'), optimState)

