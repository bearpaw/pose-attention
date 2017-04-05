-- Get prediction coordinates
predDim = {nParts,2}

criterion = nn.ParallelCriterion()
for i = 1,opt.nStack do criterion:add(nn.MSECriterion()) end

-- Code to generate training samples from raw images.
function generateSample(set, idx)
    local pts = annot[set]['part'][idx]
    local c = annot[set]['center'][idx]
    local s = annot[set]['scale'][idx]

    local img = image.load(opt.dataDir .. '/images/' .. annot[set]['images'][idx])

    -- For single-person pose estimation with a centered/scaled figure
    local inp = crop(img, c, s, 0, opt.inputRes)
    local out
    if opt.trainBg then
        out = torch.zeros(nParts+1, opt.outputRes, opt.outputRes)
    else
        out = torch.zeros(nParts, opt.outputRes, opt.outputRes)
    end
    for i = 1,nParts do
        if pts[i][1] > 0 then -- Checks that there is a ground truth annotation
            drawGaussian(out[i], transform(torch.add(pts[i],1), c, s, 0, opt.outputRes), 1)
        end
    end

    if opt.trainBg then
        out[nParts+1], _ = torch.max(out:sub(1, nParts, 1, opt.outputRes, 1, opt.outputRes), 1)

        -- local image = require('image')
        -- image.display(out[nParts+1])
        -- sys.sleep(3)
    end

    return inp,out
end

function preprocess(input, label)
    newLabel = {}
    for i = 1,opt.nStack do newLabel[i] = label end
    return input, newLabel
end

function postprocess(set, idx, output)
    local preds = getPreds(output[#output])
    return preds
end

function accuracy(output,label)
    local jntIdxs = {mpii={1,2,3,4,5,6,11,12,15,16},flic={2,3,5,6,7,8}}
    return heatmapAccuracy(output[#output],label[#output],nil,jntIdxs[opt.dataset])
end
