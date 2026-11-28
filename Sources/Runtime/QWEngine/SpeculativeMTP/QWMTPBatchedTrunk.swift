import Metal
import QWAlloc
import QWAttn
import QWCore
import QWGEMM
import QWHAL

struct QWMTPBatchedTrunk {
    let hal: QWHal
    let geometry: QWModelGeometry
    let gdnProjection: QWTensorValue
    let attentionProjection: QWTensorValue
    let outputInputs: QWTensorValue
    let controls: QWTensorValue
    let gdnPipeline: any MTLComputePipelineState
    let gdnPipelines: [Int: any MTLComputePipelineState]
    private let gdnRowPairedPipelines: [Int: any MTLComputePipelineState]
    let rows: Int
    let qkv: [QWTensorValue]
    let z: [QWTensorValue]
    let beta: [QWTensorValue]
    let decay: [QWTensorValue]
    let q: [QWTensorValue]
    let k: [QWTensorValue]
    let v: [QWTensorValue]
    let outputs: [QWTensorValue]

    static func allocationBytes(geometry: QWModelGeometry, rows: Int) -> Int {
        precondition((2...16).contains(rows))
        func aligned(_ bytes: Int) -> Int { (bytes + 255) / 256 * 256 }
        let projCols = max(1, max(geometry.gdnTotalN, geometry.attnProjTotalN))
        let hcaOutCols = geometry.hcaConfig.map { max(1, $0.oGroups) * $0.oLoraRank } ?? 0
        let outCols = max(1, max(geometry.gdnZChannels,
                                 max(geometry.qHeads * geometry.headDim,
                                     max(geometry.qHeads * geometry.vHeadDim, hcaOutCols))))
        let ctrlCols = max(1, 2 * geometry.gdnVHeads)
        return aligned(rows * projCols * 2) +
            aligned(rows * outCols * 2) + aligned(rows * ctrlCols * 4)
    }

    init(geometry: QWModelGeometry, hal: QWHal, allocator: QWSubAllocator, rows: Int) throws {
        self.hal = hal
        self.geometry = geometry
        guard (2...16).contains(rows) else { throw QWEngineDriverError.state("batched trunk requires B=2...16") }
        self.rows = rows
        let maxKernelRows = min(8, rows)
        var pipelines: [Int: any MTLComputePipelineState] = [:]
        for count in 2...maxKernelRows {
            pipelines[count] = try hal.pipeline("qwm_gdn_projection_m\(count)")
        }
        guard let maximumPipeline = pipelines[maxKernelRows] else {
            throw QWEngineDriverError.state("batched trunk maximum-row pipeline is unavailable")
        }
        gdnPipeline = maximumPipeline
        gdnPipelines = pipelines
        var rowPaired: [Int: any MTLComputePipelineState] = [:]
        for count in 2...maxKernelRows where count >= QWMTPBatchedMLP.rowPairedMinimumRows {
            rowPaired[count] = try hal.pipeline("qwm_gdn_projection_rp_m\(count)")
        }
        gdnRowPairedPipelines = rowPaired
        let gdnVH = max(1, geometry.gdnVHeads)
        let ctrlStride = 2 * gdnVH
        let projCols = max(1, max(geometry.gdnTotalN, geometry.attnProjTotalN))
        let hcaOutCols = geometry.hcaConfig.map { max(1, $0.oGroups) * $0.oLoraRank } ?? 0
        let outCols = max(1, max(geometry.gdnZChannels,
                                 max(geometry.qHeads * geometry.headDim,
                                     max(geometry.qHeads * geometry.vHeadDim, hcaOutCols))))
        let projection = try allocator.allocate(QWTensorShape([rows * projCols], .f16))
        let inputs = try allocator.allocate(QWTensorShape([rows * outCols], .f16))
        let parameters = try allocator.allocate(QWTensorShape([rows * ctrlStride], .f32))
        gdnProjection = projection
        attentionProjection = QWTensorValue(region: projection.region, offset: projection.offset,
            shape: QWTensorShape([max(1, rows * geometry.attnProjTotalN)], .f16))
        outputInputs = inputs
        controls = parameters
        func views(_ storage: QWTensorValue, stride: Int, start: Int, count: Int, dtype: QWDType) -> [QWTensorValue] {
            let bytes = dtype == .f16 ? 2 : 4
            return (0..<rows).map {
                QWTensorValue(region: storage.region, offset: storage.offset + ($0 * stride + start) * bytes,
                              shape: QWTensorShape([max(1, count)], dtype))
            }
        }
        qkv = views(projection, stride: projCols, start: 0,
                    count: geometry.gdnQKVChannels, dtype: .f16)
        z = views(projection, stride: projCols, start: geometry.gdnQKVChannels,
                  count: geometry.gdnZChannels, dtype: .f16)
        beta = views(parameters, stride: ctrlStride, start: 0, count: gdnVH, dtype: .f32)
        decay = views(parameters, stride: ctrlStride, start: gdnVH, count: gdnVH, dtype: .f32)
        q = views(projection, stride: geometry.attnProjTotalN, start: 0,
                  count: geometry.qProjOut, dtype: .f16)
        k = views(projection, stride: geometry.attnProjTotalN, start: geometry.attnKRow0,
                  count: geometry.kvChannels, dtype: .f16)
        v = views(projection, stride: geometry.attnProjTotalN, start: geometry.attnVRow0,
                  count: geometry.kvChannels, dtype: .f16)
        outputs = views(inputs, stride: outCols, start: 0,
                        count: outCols, dtype: .f16)
    }

    func gdnPipeline(forActiveRows activeRows: Int, rowPaired: Bool = false, isAQ4: Bool = false) throws -> any MTLComputePipelineState {
        guard (2...rows).contains(activeRows) else {
            throw QWEngineDriverError.state("invalid batched trunk active row count")
        }
        if isAQ4 {
            let name = rowPaired ? "qwm_gdn_projection_rp_m\(activeRows)" : "qwm_gdn_projection_m\(activeRows)"
            return try hal.pipeline(name, isAQ4: true)
        }
        guard let pipeline = (rowPaired ? gdnRowPairedPipelines : gdnPipelines)[activeRows] else {
            throw QWEngineDriverError.state("invalid batched trunk active row count")
        }
        return pipeline
    }

    func validate(_ row: Int) throws {
        guard (0..<rows).contains(row) else { throw QWEngineDriverError.state("invalid batched trunk row") }
    }
}

extension QWEngineDriver {
    func encodeVerificationLayerInputProjection(
        _ enc: any MTLComputeCommandEncoder,
        weights: LayerW,
        activeNormalized: QWTensorValue,
        mtpBatchedTrunk: QWMTPBatchedTrunk,
        mtpBatchedMLP: QWMTPBatchedMLP,
        activeRows: Int
    ) throws {
        switch weights.kind {
        case .gdn, .gdnMoe:
            let w = weights.gdn!
            if geometry.gdnVectorDecay {
                let qkvC = geometry.gdnQKVChannels
                let fgbC = geometry.kdaFgbChannels
                let qkvOut = prefixRows(mtpBatchedTrunk.gdnProjection, rows: activeRows, cols: qkvC, dt: .f16)
                try mtpBatchedMLP.encodeProjection(enc, input: activeNormalized, weight: w.inProj,
                                                   output: qkvOut, activeRows: activeRows)
                if let fgbProj = w.fgbProj, let fgb16 = psc.fgb16 {
                    let fgbOut = prefixRows(fgb16, rows: activeRows, cols: fgbC, dt: .f16)
                    try mtpBatchedMLP.encodeProjection(enc, input: activeNormalized, weight: fgbProj,
                                                       output: fgbOut, activeRows: activeRows)
                }
            } else {
                try mtpBatchedMLP.encodeGDNProjection(enc, input: activeNormalized, weight: w.inProj,
                    output: mtpBatchedTrunk.gdnProjection,
                    controls: QWMTPBatchedMLP.GDNControls(aLog: w.aLog, bias: w.dtBias,
                                                          output: mtpBatchedTrunk.controls),
                    trunk: mtpBatchedTrunk,
                    activeRows: activeRows)
            }
        case .fullAttention, .moe:
            let wAttn = weights.attn!.qkvProj
            if let mla = mlaOperator, weights.attn?.mla != nil {
                let latentOut = prefixRows(mla.mlaLatentScratch, rows: activeRows, cols: wAttn.n, dt: .f16)
                try mtpBatchedMLP.encodeProjection(enc, input: activeNormalized, weight: wAttn,
                                                   output: latentOut, activeRows: activeRows)
            } else {
                let attnOutSlice = prefixRows(mtpBatchedTrunk.attentionProjection, rows: activeRows, cols: wAttn.n, dt: .f16)
                try mtpBatchedMLP.encodeProjection(enc, input: activeNormalized, weight: wAttn,
                                                   output: attnOutSlice, activeRows: activeRows)
            }
            if let mbsa = mbsaOperator, let idxProj = weights.attn?.mbsa?.qkvIdxProj {
                let idxCols = (geometry.kvHeads + 1) * geometry.headDim
                let idxOut = prefixRows(mbsa.idxQKVScratch, rows: activeRows, cols: idxCols, dt: .f16)
                try mtpBatchedMLP.encodeProjection(enc, input: activeNormalized, weight: idxProj,
                                                   output: idxOut, activeRows: activeRows)
            }
        case .hca, .hcaMoe:
            guard let hca = hcaOperator, let wFusedIn = weights.hca?.wFusedIn else {
                throw QWEngineDriverError.state("HCA verification layer missing operator or wFusedIn")
            }
            let fusedInOut = prefixRows(hca.fusedInScratch, rows: activeRows, cols: hca.inStride, dt: .f16)
            try mtpBatchedMLP.encodeProjection(enc, input: activeNormalized, weight: wFusedIn,
                                               output: fusedInOut, activeRows: activeRows)
        }
    }

    func encodeVerificationLayerCore(
        _ enc: any MTLComputeCommandEncoder,
        session: QWEngineSession,
        layer: Int,
        weights: LayerW,
        activeNormalized: QWTensorValue,
        checkpoint: QWGDNReplayCheckpoint,
        mtpBatchedTrunk: QWMTPBatchedTrunk,
        mtpBatchedMLP: QWMTPBatchedMLP,
        partials: QWMRegion?,
        activeRows: Int
    ) throws {
        switch weights.kind {
        case .gdn, .gdnMoe:
            try encodeVerificationGDNCore(
                enc, session: session, layer: layer, weights: weights,
                activeNormalized: activeNormalized, checkpoint: checkpoint,
                mtpBatchedTrunk: mtpBatchedTrunk, activeRows: activeRows)
        case .fullAttention, .moe:
            try encodeVerificationAttnCore(
                enc, session: session, layer: layer, weights: weights,
                activeNormalized: activeNormalized, mtpBatchedTrunk: mtpBatchedTrunk,
                mtpBatchedMLP: mtpBatchedMLP, partials: partials, activeRows: activeRows)
        case .hca, .hcaMoe:
            try encodeVerificationHCACore(
                enc, session: session, layer: layer, weights: weights,
                mtpBatchedTrunk: mtpBatchedTrunk, mtpBatchedMLP: mtpBatchedMLP, activeRows: activeRows)
        }
    }

    private func encodeVerificationGDNCore(
        _ enc: any MTLComputeCommandEncoder,
        session: QWEngineSession,
        layer: Int,
        weights: LayerW,
        activeNormalized: QWTensorValue,
        checkpoint: QWGDNReplayCheckpoint,
        mtpBatchedTrunk: QWMTPBatchedTrunk,
        activeRows: Int
    ) throws {
        let w = weights.gdn!, index = gdnIndexMap[layer]
        let gdnOut = prefixRows(mtpBatchedTrunk.outputInputs, rows: activeRows,
                                cols: geometry.gdnZChannels, dt: .f16)
        if geometry.gdnVectorDecay, let fgb16 = psc.fgb16 {
            let fgbC = geometry.kdaFgbChannels
            let fgbView = prefixRows(fgb16, rows: activeRows, cols: fgbC, dt: .f16)
            if let fB = w.fB, let gB = w.gB {
                try checkpoint.encodeBatchedKDA(
                    enc, conv: gdnConv, decoder: gdnDecode,
                    layer: index, activeRows: activeRows,
                    gdnProjection: mtpBatchedTrunk.gdnProjection,
                    fgbProjection: fgbView,
                    fBWeight: fB, gBWeight: gB, aLog: w.aLog, dtBias: w.dtBias,
                    convWeight: w.conv, normWeight: w.normW,
                    zScratch: psc.z16,
                    ring: session.ring[index], state: session.rec[index],
                    output: gdnOut)
            } else {
                try checkpoint.encodeBatchedKDADirect(
                    enc, conv: gdnConv, decoder: gdnDecode,
                    layer: index, activeRows: activeRows,
                    gdnProjection: mtpBatchedTrunk.gdnProjection,
                    fgbProjection: fgbView,
                    ln1: activeNormalized, bProjF16: w.bProjF16,
                    aLog: w.aLog, dtBias: w.dtBias,
                    convWeight: w.conv, normWeight: w.normW,
                    zScratch: psc.z16,
                    ring: session.ring[index], state: session.rec[index],
                    output: gdnOut)
            }
        } else {
            try checkpoint.encodeBatchedGDN(
                enc, conv: gdnConv, decoder: gdnDecode,
                layer: index, activeRows: activeRows,
                gdnProjection: mtpBatchedTrunk.gdnProjection, controls: mtpBatchedTrunk.controls,
                convWeight: w.conv, normWeight: w.normW,
                ring: session.ring[index], state: session.rec[index],
                output: gdnOut)
        }
    }

    private func encodeVerificationAttnCore(
        _ enc: any MTLComputeCommandEncoder,
        session: QWEngineSession,
        layer: Int,
        weights: LayerW,
        activeNormalized: QWTensorValue,
        mtpBatchedTrunk: QWMTPBatchedTrunk,
        mtpBatchedMLP: QWMTPBatchedMLP,
        partials: QWMRegion?,
        activeRows: Int
    ) throws {
        let fullAttnIdx = attnIndexMap[layer]
        let views = session.attnKVViews[fullAttnIdx]
        let attnOutCols = weights.attn!.o.k
        let attnOut = prefixRows(mtpBatchedTrunk.outputInputs, rows: activeRows,
                                 cols: attnOutCols, dt: .f16)
        if let mla = mlaOperator, let mlaW = weights.attn?.mla {
            let mropePos = session.mropeSlice(pos: session.position, count: activeRows, mropeLayout: mla.mropeLayout)
            mla.encodeLatentPrepAndPool(
                enc, ln1: activeNormalized, mlaW: mlaW,
                kv: views, fullAttnLayer: fullAttnIdx,
                pos0: session.position, rows: activeRows,
                mropePositions: mropePos)
            enc.memoryBarrier(scope: .buffers)
            let qInView = mla.isDirectQ
                ? activeNormalized
                : prefixRows(mla.qResidScratch, rows: activeRows, cols: mlaW.qBProj.k, dt: .f16)
            let qFullView = prefixRows(mla.mlaQFullScratch, rows: activeRows, cols: mlaW.qBProj.n, dt: .f16)
            try mtpBatchedMLP.encodeProjection(enc, input: qInView, weight: mlaW.qBProj,
                                               output: qFullView, activeRows: activeRows)
            mla.encodeScoreSelectAbsorbAttendAndProjUV(
                enc, mlaW: mlaW, kv: views, attOut: attnOut,
                fullAttnLayer: fullAttnIdx, pos0: session.position, rows: activeRows,
                ln1: activeNormalized, mropePositions: mropePos)
            enc.memoryBarrier(scope: .buffers)
            return
        }
        guard let partials, let attnDecode else {
            throw QWEngineDriverError.state("GQA verification requires shared attention partials and attnDecode")
        }
        let keys = prefixRows(psc.attKR16, rows: activeRows, cols: geometry.kvChannels, dt: .f16)
        let values = prefixRows(psc.attV16, rows: activeRows, cols: geometry.kvChannels, dt: .f16)
        let qgate = prefixRows(psc.attQR16, rows: activeRows, cols: geometry.qProjOut, dt: .f16)
        let qkvCols = weights.attn!.qkvProj.n
        try ropeApply.encodeExtractQK(enc, hal: hal,
            attnY: prefixRows(mtpBatchedTrunk.attentionProjection, rows: activeRows, cols: qkvCols, dt: .f16),
            qOut: qgate, kOut: keys, vOut: values,
            wq: weights.attn!.wq, wk: weights.attn!.wkTiled,
            qkvBias: weights.attn!.qkvBias,
            positionStart: max(0, session.position + session.roundRecord.mropeDelta), count: activeRows,
            inRowCols: qkvCols)
        enc.memoryBarrier(scope: .buffers)
        let layerSW = geometry.virtualLayerSlidingWindow(layer)
        let mtpSparseThreshold = lsaConfig.denseBypassThreshold
        if layerSW == 0, let lsa = lsaOperator {
            let isSparse = session.position >= mtpSparseThreshold
            lsa.encodeIndexAndCluster(enc, qgate: qgate, kIn: keys,
                                      fullAttnLayer: fullAttnIdx,
                                      pos0: session.position, rows: activeRows,
                                      qMode: isSparse ? .allRows : .none)
        }
        if let mbsa = mbsaOperator, let mbsaW = weights.attn?.mbsa,
           session.position + activeRows <= mbsa.maxContext {
            if mbsaW.qkvIdxProj != nil {
                mbsa.encodeIndexPrep(enc, weights: mbsaW, ropeTable: ropeApply.tableTensor,
                                     layerIndex: fullAttnIdx, pos0: session.position, rows: activeRows)
            } else {
                let idxSlice = QWTensorValue(
                    region: mtpBatchedTrunk.attentionProjection.region,
                    offset: mtpBatchedTrunk.attentionProjection.offset + geometry.attnTotalN * 2,
                    shape: QWTensorShape([activeRows * qkvCols - geometry.attnTotalN], .f16))
                mbsa.encodeIndexPrep(enc, weights: mbsaW, ropeTable: ropeApply.tableTensor,
                                     layerIndex: fullAttnIdx, pos0: session.position, rows: activeRows,
                                     qkvSource: idxSlice, qkvRowStrideCols: qkvCols)
            }
        }
        encodePagedKVWrite(enc, kv: views, k: keys, v: values, startPos: session.position, t: activeRows)
        enc.memoryBarrier(scope: .buffers)
        if let mbsa = mbsaOperator, weights.attn?.mbsa != nil,
           session.position + activeRows > mbsa.config.denseEquivalenceTokens,
           session.position + activeRows <= mbsa.maxContext {
            mbsa.encodeSelectAndAttend(
                enc, qgate: qgate, kv: views,
                out: prefixRows(attnOut, rows: activeRows, cols: geometry.qHeads * geometry.headDim, dt: .f16),
                layerIndex: fullAttnIdx, pos0: session.position, rows: activeRows,
                partials: partials)
            enc.memoryBarrier(scope: .buffers)
        } else if layerSW == 0, let lsa = lsaOperator, session.position >= mtpSparseThreshold {
            lsa.encodeSelectBlocks(enc, fullAttnLayer: fullAttnIdx,
                                   pos0: session.position, rows: activeRows)
            let budgetCap = lsaConfig.alignedBudget + 640
            let outCols = geometry.qHeads * geometry.headDim
            var r0 = 0
            while r0 < activeRows {
                let runSplits = QWAttnDecode.splits(
                    forKeys: min(session.position + r0 + 1, budgetCap),
                    kvHeads: geometry.kvHeads)
                var r1 = r0 + 1
                while r1 < activeRows,
                      r1 - r0 < 8,
                      QWAttnDecode.splits(
                          forKeys: min(session.position + r1 + 1, budgetCap),
                          kvHeads: geometry.kvHeads) == runSplits {
                    r1 += 1
                }
                let chunk = r1 - r0
                let fullWindow = (r0 == 0 && chunk == activeRows)
                let blockListTV = lsa.blockList(forFullAttnLayer: fullAttnIdx)
                let blockCountTV = lsa.blockCount(forFullAttnLayer: fullAttnIdx)
                try attnDecode.encodeSparseBatched(
                    enc,
                    qgate: fullWindow ? qgate : qgate.slice(byteOffset: r0 * geometry.qProjOut * 2, count: chunk * geometry.qProjOut),
                    kv: views,
                    knew: fullWindow ? keys : keys.slice(byteOffset: r0 * geometry.kvChannels * 2, count: chunk * geometry.kvChannels),
                    vnew: fullWindow ? values : values.slice(byteOffset: r0 * geometry.kvChannels * 2, count: chunk * geometry.kvChannels),
                    partials: partials,
                    out: fullWindow ? attnOut : attnOut.slice(byteOffset: r0 * outCols * 2, count: chunk * outCols),
                    blockList: fullWindow ? blockListTV : QWTensorValue(
                        region: blockListTV.region,
                        offset: blockListTV.offset + r0 * lsa.listStride * 4,
                        shape: QWTensorShape([chunk * lsa.listStride], .i32)),
                    blockCount: fullWindow ? blockCountTV : QWTensorValue(
                        region: blockCountTV.region,
                        offset: blockCountTV.offset + r0 * 4,
                        shape: QWTensorShape([chunk], .i32)),
                    cBase: session.position + r0, T: chunk, splits: runSplits,
                    listStride: lsa.listStride, writeKV: false)
                enc.memoryBarrier(scope: .buffers)
                r0 = r1
            }
        } else {
            let outCols = geometry.qHeads * geometry.headDim
            func keysForRow(_ rowOffset: Int) -> Int {
                let total = session.position + rowOffset + 1
                return layerSW > 0 ? min(total, layerSW) : total
            }
            var r0 = 0
            while r0 < activeRows {
                let runSplits = QWAttnDecode.splits(forKeys: keysForRow(r0), kvHeads: geometry.kvHeads, gpuCores: hal.caps.gpuCores)
                var r1 = r0 + 1
                while r1 < activeRows,
                      r1 - r0 < 8,
                      QWAttnDecode.splits(forKeys: keysForRow(r1), kvHeads: geometry.kvHeads, gpuCores: hal.caps.gpuCores) == runSplits {
                    r1 += 1
                }
                let chunk = r1 - r0
                let fullWindow = (r0 == 0 && chunk == activeRows)
                let qChunk = fullWindow ? qgate : qgate.slice(byteOffset: r0 * geometry.qProjOut * 2, count: chunk * geometry.qProjOut)
                let kChunk = fullWindow ? keys : keys.slice(byteOffset: r0 * geometry.kvChannels * 2, count: chunk * geometry.kvChannels)
                let vChunk = fullWindow ? values : values.slice(byteOffset: r0 * geometry.kvChannels * 2, count: chunk * geometry.kvChannels)
                let oChunk = fullWindow ? attnOut : attnOut.slice(byteOffset: r0 * outCols * 2, count: chunk * outCols)
                if chunk == 1 {
                    try attnDecode.encode(
                        enc, qgate: qChunk, kv: views, knew: kChunk, vnew: vChunk,
                        partials: partials, out: oChunk,
                        C: session.position + r0, splits: runSplits,
                        slidingWindow: layerSW)
                } else {
                    try attnDecode.encodeBatched(
                        enc, qgate: qChunk, kv: views, knew: kChunk, vnew: vChunk,
                        partials: partials, out: oChunk,
                        cBase: session.position + r0, T: chunk, splits: runSplits,
                        sharedPrefix: true, slidingWindow: layerSW)
                }
                enc.memoryBarrier(scope: .buffers)
                r0 = r1
            }
        }
    }

    private func encodeVerificationHCACore(
        _ enc: any MTLComputeCommandEncoder,
        session: QWEngineSession,
        layer: Int,
        weights: LayerW,
        mtpBatchedTrunk: QWMTPBatchedTrunk,
        mtpBatchedMLP: QWMTPBatchedMLP,
        activeRows: Int
    ) throws {
        guard let hca = hcaOperator, let hcaW = weights.hca, let wQUpGate = hcaW.wQUpGate else {
            throw QWEngineDriverError.state("HCA verification layer missing operator or wQUpGate")
        }
        let hcaIdx = hcaIndexMap[layer]
        hca.encodeInPrepAndCompress(enc, hcaW: hcaW, hcaLayer: hcaIdx, pos0: session.position, rows: activeRows)
        enc.memoryBarrier(scope: .buffers)
        let qrView = prefixRows(hca.qrScratch, rows: activeRows, cols: hca.qLoraRank, dt: .f16)
        let qUpView = prefixRows(hca.qUpGateScratch, rows: activeRows, cols: hca.qUpStride, dt: .f16)
        try mtpBatchedMLP.encodeProjection(enc, input: qrView, weight: wQUpGate,
                                           output: qUpView, activeRows: activeRows)
        let zOut = prefixRows(mtpBatchedTrunk.outputInputs, rows: activeRows,
                              cols: hca.zOutStride, dt: .f16)
        hca.encodeQPrepAttendVHAAndWoA(enc, hcaW: hcaW, zGatedOut: zOut,
                                       hcaLayer: hcaIdx, pos0: session.position, rows: activeRows)
        enc.memoryBarrier(scope: .buffers)
    }
}
