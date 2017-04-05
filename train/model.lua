
--- Load up network model or initialize from scratch
paths.dofile('models/' .. opt.netType .. '.lua')

-- Continuing an experiment where it left off
if opt.continue or opt.branch ~= 'none' then
    local prevModel = opt.load .. '/final_model.t7'
    print('==> Loading model from: ' .. prevModel)
    model = torch.load(prevModel)

-- Or a path to previously trained model is provided
elseif opt.loadModel ~= 'none' then
    assert(paths.filep(opt.loadModel), 'File not found: ' .. opt.loadModel)
    print('==> Loading model from: ' .. opt.loadModel)
    model = torch.load(opt.loadModel)

-- Or we're starting fresh
else
    print('==> Creating model from file: models/' .. opt.netType .. '.lua')
    model = createModel(modelArgs)
end

-- First remove any DataParallelTable
if torch.type(model) == 'nn.DataParallelTable' then
  model = model:get(1)
end

-- Criterion (can be set in the opt.task file as well)
if not criterion then
    criterion = nn[opt.crit .. 'Criterion']()
end

if opt.GPU ~= -1 then
    -- Convert model to CUDA
    print('==> Converting model to CUDA')
    model:cuda()
    criterion:cuda()
    
    cudnn.fastest = true
    cudnn.benchmark = true
end

-- Wrap the model with DataParallelTable, if using more than one GPU
if opt.nGPU > 1 then
  local gpus = torch.range(1, opt.nGPU):totable()
  local fastest, benchmark = cudnn.fastest, cudnn.benchmark
  local dpt = nn.DataParallelTable(1, true, true)
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
