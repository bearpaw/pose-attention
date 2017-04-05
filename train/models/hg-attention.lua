paths.dofile('layers/Residual.lua')
paths.dofile('layers/ResidualPool.lua')
paths.dofile('layers/AttentionPartsCRF.lua')

local function repResidual(num, inp, nRep)
    local out={}
    for i=1,nRep do
        local tmpout
        if i==1 then
            tmpout = Residual(num, num)(inp)
        else
            tmpout = ResidualPool(num, num)(out[i-1])  -------------------- here
        end
        table.insert(out,tmpout)
    end
    return out[nRep]
end

local function hourglass(n, f, inp, imsize,nModual)
    -- Upper branch
    local pool = nnlib.SpatialMaxPooling(2,2,2,2)(inp)
    local up={}
    local low={}
    for i=1, nModual do
        local tmpup
        local tmplow
        if i==1 then
            -- tmpup = Residual(f,f)(inp)
            if n>1 then
                tmpup = repResidual(f,inp,n-1)
            else
                tmpup = Residual(f,f)(inp)
            end
            tmplow = Residual(f,f)(pool)
        else
            if n>1 then 
                tmpup = repResidual(f,up[i-1],n-1)
            else
                tmpup = ResidualPool(f,f)(up[i-1]) ------------- here
            end
            -- tmpup = Residual(f,f)(up[i-1])
            tmplow = Residual(f,f)(low[i-1])
        end
        table.insert(up,tmpup)
        table.insert(low,tmplow)
    end
    -- Lower branch
    
    local low2
    if n > 1 then 
        low2 = hourglass(n-1,f,low[nModual],imsize/2,nModual)
    else 
        low2 = Residual(f,f)(low[nModual]) 
    end

    local low3 = Residual(f,f)(low2)
    local up2 = nn.SpatialUpSamplingNearest(2)(low3)

    comb =  nn.CAddTable()({up[nModual],up2})
    
    return  comb
end


local function lin(numIn,numOut,inp)
    -- Apply 1x1 convolution, stride 1, no padding
    local l = nnlib.SpatialConvolution(numIn,numOut,1,1,1,1,0,0)(inp)
    return nnlib.ReLU(true)(nn.SpatialBatchNormalization(numOut)(l))
end

function createModel()
    local inp = nn.Identity()()

    -- Initial processing of the image
    local cnv1_ = nnlib.SpatialConvolution(3,64,7,7,1,1,3,3)(inp)           -- 128
    local cnv1 = nnlib.ReLU(true)(nn.SpatialBatchNormalization(64)(cnv1_))
    local r1 = Residual(64,64)(cnv1)
    local pool1 = nnlib.SpatialMaxPooling(2,2,2,2)(r1)  
    local r2 =  Residual(64,64)(pool1)
    local r3 = Residual(64,128)(r2)

    local pool2 = nnlib.SpatialMaxPooling(2,2,2,2)(r3)  
    local r4 = Residual(128,128)(pool2)
    local r5 = Residual(128,128)(r4)
    local r6 = Residual(128,opt.nFeats)(r5)

    local out = {}
    local inter = {}
    table.insert(inter,r6)

    npool = opt.nPool;
    if npool == 3 then 
        nModual = 16/opt.nStack
    else
        nModual = 8/opt.nStack
    end

    for i = 1,opt.nStack do
        local hg = hourglass(npool,opt.nFeats,inter[i],opt.outputRes,nModual)
        local ll1
        local ll2
        local att
        local tmpOut

        if i==opt.nStack then
            -- Linear layer to produce first set of predictions
            ll1 = lin(opt.nFeats,opt.nFeats*2,hg)
            ll2 = lin(opt.nFeats*2,opt.nFeats*2,ll1)
            att = AttentionPartsCRF(opt.nFeats*2, ll2, opt.LRNKer, 3, 0)
            tmpOut = AttentionPartsCRF(opt.nFeats*2, att, opt.LRNKer, 3, 1)
        else
            ll1 = lin(opt.nFeats,opt.nFeats,hg)
            ll2 = lin(opt.nFeats,opt.nFeats,ll1)
            if i>4 then
                att = AttentionPartsCRF(opt.nFeats, ll2, opt.LRNKer, 3, 0)
                tmpOut = AttentionPartsCRF(opt.nFeats, att, opt.LRNKer, 3, 1)
            else
                att = AttentionPartsCRF(opt.nFeats, ll2, opt.LRNKer,3,0)
                tmpOut = nnlib.SpatialConvolution(opt.nFeats,outputDim[1][1],1,1,1,1,0,0)(att)
            end
        end
        table.insert(out,tmpOut)

        if i < opt.nStack then 
            local outmap = nnlib.SpatialConvolution(outputDim[1][1], 256, 1,1,1,1,0,0)(tmpOut)
            local ll3 = lin(opt.nFeats,opt.nFeats,ll1)
            local tmointer = nn.CAddTable()({inter[i], outmap,ll3}) 
            table.insert(inter,tmointer)
        end
    end
    -- Final model
    local model = nn.gModule({inp}, out)
    return model
end
