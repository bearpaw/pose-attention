--[[
Average scores

Wei Yang
2016-10-26
]]--
require 'paths'
local matio = require 'matio'
paths.dofile('util.lua')
paths.dofile('img.lua')

local isdisplay = true
local isflip = true
local minusmean = true

local hmthresh = 0.002
local scale_search = {1.0}
-- local scale_search = {0.7,0.8,0.9,1.0,1.1,1.2} -- used in our paper with NVIDIA Titan X (12 GB memory).

local outdim = 64
local partnum = 17

local modeldir = '../checkpoints/mpii/crf_parts/'
local modelname = 'model.t7'

local modelpath = paths.concat(modeldir, modelname)
local predpath = paths.concat(modeldir, 'pred_' .. paths.basename(modelname, 't7') .. '.mat')

print(modelpath)
print(predpath)

local function shuffleLR(x)
   local dim
   if x:nDimension() == 4 then
      dim = 2
   else
      assert(x:nDimension() == 3)
      dim = 1
   end

   local matched_parts = {
      {1,6},   {2,5},   {3,4},
      {11,16}, {12,15}, {13,14}
   }

   for i = 1,#matched_parts do
      local idx1, idx2 = unpack(matched_parts[i])
      local tmp = x:narrow(dim, idx1, 1):clone()
      x:narrow(dim, idx1, 1):copy(x:narrow(dim, idx2, 1))
      x:narrow(dim, idx2, 1):copy(tmp)
   end

   return x
end

local function flip(x)
   require 'image'
   local y = torch.FloatTensor(x:size())
   for i = 1, x:size(1) do
      image.hflip(y[i], x[i]:float())
   end
   return y:typeAs(x)
end



local function drawOriginalSkeleton(input, coords)

   local im = input:clone()

   local pairRef = {
      {1,2},      {2,3},      {3,7},
      {4,5},      {4,7},      {5,6},
      {7,9},      {9,10},
      {14,9},     {11,12},    {12,13},
      {13,9},     {14,15},    {15,16}
   }

   local partNames = {'RAnk','RKne','RHip','LHip','LKne','LAnk',
      'Pelv','Thrx','Neck','Head',
      'RWri','RElb','RSho','LSho','LElb','LWri'}
   local partColor = {1,1,1,2,2,2,0,0,0,0,3,3,3,4,4,4}

   local actThresh = 0.002

   -- Loop through adjacent joint pairings
   for i = 1,#pairRef do
      if coords[pairRef[i][1]][3] > actThresh and coords[pairRef[i][2]][3] > actThresh then
         -- Set appropriate line color
         local color
         if partColor[pairRef[i][1]] == 1 then color = {0,.3,1}
         elseif partColor[pairRef[i][1]] == 2 then color = {1,.3,0}
         elseif partColor[pairRef[i][1]] == 3 then color = {0,0,1}
         elseif partColor[pairRef[i][1]] == 4 then color = {1,0,0}
         else color = {.7,0,.7} end

         -- Draw line
         im = drawLine(im, coords[pairRef[i][1]], coords[pairRef[i][2]], 4, color, 0)
      end
   end

   return im
end

--------------------------------------------------------------------------------
-- Initialization
--------------------------------------------------------------------------------

if arg[1] == 'demo' or arg[1] == 'predict-test' then
   -- Test set annotations do not have ground truth part locations, but provide
   -- information about the location and scale of people in each image.
   a = loadAnnotations('test')

elseif arg[1] == 'predict-valid' then
   -- Validation set annotations on the other hand, provide part locations,
   -- visibility information, normalization factors for final evaluation, etc.
   a = loadAnnotations('valid')

else
   print("Please use one of the following input arguments:")
   print("    demo - Generate and display results on a few demo images")
   print("    predict-valid - Generate predictions on the validation set (MPII images must be available in 'images' directory)")
   print("    predict-test - Generate predictions on the test set")
   return
end

m = torch.load(modelpath)   -- Load pre-trained model

-- don't save the DataParallelTable for easier loading on other machines
if torch.type(m) == 'nn.DataParallelTable' then
   print('convert to single model')
   m = m:get(1):clearState()
   torch.save(paths.concat(modeldir, 'single_' .. modelname), m)
   print('Done')
   os.exit()
end

m:cuda()
m:evaluate()

if arg[1] == 'demo' then
   idxs = torch.Tensor({695, 3611, 2486, 7424, 10032, 5, 4829})
   -- If all the MPII images are available, use the following line to see a random sampling of images
   -- idxs = torch.randperm(a.nsamples):sub(1,10)
else
   idxs = torch.range(1,a.nsamples)
end


nsamples = idxs:nElement()
-- Displays a convenient progress bar
xlua.progress(0,nsamples)
preds = torch.Tensor(nsamples,16,3)

--------------------------------------------------------------------------------
-- Main loop
--------------------------------------------------------------------------------
local perm = torch.range(1, nsamples)


for idx = 1,nsamples do
   local i = perm[idx]
   local imname = paths.basename(a['images'][idxs[i]], 'jpg')
   local respath = paths.concat(predpath, imname ..'.mat')

   -- Set up input image
   local im = image.load('../data/mpii/images/' .. a['images'][idxs[i]])
   local center = a['center'][idxs[i]]
   local original_scale = a['scale'][idxs[i]]

   local fuseInp = torch.zeros(#scale_search, 3, 256, 256)
   local hmpyra = torch.zeros(#scale_search, 16, im:size(2), im:size(3))
   local batch = torch.Tensor(#scale_search, 3, 256, 256)
   local flipbatch = torch.Tensor(#scale_search, 3, 256, 256)

   for is, factor in ipairs(scale_search) do
      local scale = original_scale*factor
      local inp = crop(im, center, scale, 0, 256)
      batch[{is, {}, {}, {}}]:copy(inp)
   end

   -- minus mean
   if minusmean then
      batch:add(-0.5)
   end
   -- Get network output
   local out = m:forward(batch:cuda())

   -- Get flipped output
   if isflip then
      out = applyFn(function (x) return x:clone() end, out)
      local flippedOut = m:forward(flip(batch):cuda())
      flippedOut = applyFn(function (x) return flip(shuffleLR(x)) end, flippedOut)
      out = applyFn(function (x,y) return x:add(y):div(2) end, out, flippedOut)
   end

   cutorch.synchronize()
   local hm = out[#out]:float()
   hm[hm:lt(0)] = 0

   -- Get heatmaps (original image size)
   for is, scale in pairs(scale_search) do
      local hm_img = getHeatmaps(im, center, original_scale*scale, 0, 256, hm[is])
      hmpyra[{is, {}, {}, {}}]:copy(hm_img:sub(1, 16))
   end

   -- fuse heatmap
   if arg[2] == 'max' then
      fuseHm = hmpyra:max(1)
   else
      fuseHm = hmpyra:mean(1)
   end
   fuseHm = fuseHm[1]
   fuseHm[fuseHm:lt(0)] = 0


   -- get predictions
   for p = 1,16 do
      local maxy, iy = fuseHm[p]:max(2)
      local maxv, ix = maxy:max(1)
      ix = torch.squeeze(ix)

      preds[i][p][2] = ix
      preds[i][p][1] = iy[ix]
      preds[i][p][3] = maxy[ix]
   end

   -- Display the result
   if isdisplay then
      local dispImg = drawOriginalSkeleton(im, preds[i])
      w = image.display{image=dispImg,win=w}
   end

   xlua.progress(idx,nsamples)
   collectgarbage()
end

-- Save predictions
if arg[1] == 'predict-valid' then
   matio.save(paths.concat(predpath), preds)
elseif arg[1] == 'predict-test' then
   matio.save(paths.concat(predpath, 'full-test.mat'), preds)
end
