local function AttentionIter(numIn, inp , lrnSize, itersize)
    local pad = math.floor(lrnSize/2)
    local U = nnlib.SpatialConvolution(numIn,1, 3, 3, 1,1,1,1)(inp)
    local spConv = nnlib.SpatialConvolution(1,1,lrnSize,lrnSize,1,1,pad,pad)
    -- need to share the parameters and the gradParameters as well
    local spConv_clone = spConv:clone('weight','bias','gradWeight','gradBias')

    local Q={}
    local C = {}
    for i=1,itersize do
        local conv 
        local Q_tmp

        if i==1 then
            conv = spConv(U)
        else
            conv = spConv_clone(Q[i-1])
        end
        table.insert(C,conv)
        Q_tmp = nn.Sigmoid()(nn.CAddTable(true)({C[i], U}))
        table.insert(Q,Q_tmp)
    end

    local pfeat = nn.CMulTable(){inp, nn.Replicate(numIn,   2){Q[itersize]}}
    return pfeat 
end

function AttentionPartsCRF(numIn, inp , lrnSize, itersize, usepart)
    if usepart == 0 then
        return AttentionIter(numIn, inp , lrnSize, itersize)
    else
        local partnum = outputDim[1][1]
        local pre = {}
        for i=1, partnum do
            local att = AttentionIter(numIn, inp , lrnSize, itersize)
            local s = nnlib.SpatialConvolution(numIn,1,1,1,1,1,0,0)(att)
            table.insert(pre,s)
        end
        return nn.JoinTable(2)(pre)
    end

end
