import Metal
import QWAlloc
import QWAttn
import QWCore
import QWGEMM
import QWHAL

struct QWMTPBatchedTrunk {
    let geometry: QWModelGeometry
    let gdnProjection: QWTensorValue
    let attentionProjection: QWTensorValue
    let outputInputs: QWTensorValue
    let controls: QWTensorValue
    let gdnPipeline: any MTLComputePipelineState
    let gdnPipelines: [Int: any MTLComputePipelineState]
    private let gdnRowPairedPipelines: [Int: any MTLComputePipelineState]
    let rows: Int
    private let qkv: [QWTensorValue]
    private let z: [QWTensorValue]
    private let beta: [QWTensorValue]
    private let decay: [QWTensorValue]
    private let q: [QWTensorValue]
    private let k: [QWTensorValue]
    private let v: [QWTensorValue]
    private let outputs: [QWTensorValue]

    static func allocationBytes(geometry: QWModelGeometry, rows: Int) -> Int {
        precondition((2...16).contains(rows))
        func aligned(_ bytes: Int) -> Int { (bytes + 255) / 256 * 256 }
        let projCols = max(1, max(geometry.gdnTotalN, geometry.attnTotalN))
        let hcaOutCols = geometry.hcaConfig.map { max(1, $0.oGroups) * $0.oLoraRank } ?? 0
        let outCols = max(1, max(geometry.gdnZChannels,
                                 max(geometry.qHeads * geometry.headDim,
                                     max(geometry.qHeads * geometry.vHeadDim, hcaOutCols))))
        let ctrlCols = max(1, 2 * geometry.gdnVHeads)
        return aligned(rows * projCols * 2) +
            aligned(rows * outCols * 2) + aligned(rows * ctrlCols * 4)
    }

    init(geometry: QWModelGeometry, hal: QWHal, allocator: QWSubAllocator, rows: Int) throws {
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
        let projCols = max(1, max(geometry.gdnTotalN, geometry.attnTotalN))
        let hcaOutCols = geometry.hcaConfig.map { max(1, $0.oGroups) * $0.oLoraRank } ?? 0
        let outCols = max(1, max(geometry.gdnZChannels,
                                 max(geometry.qHeads * geometry.headDim,
                                     max(geometry.qHeads * geometry.vHeadDim, hcaOutCols))))
        let projection = try allocator.allocate(QWTensorShape([rows * projCols], .f16))
        let inputs = try allocator.allocate(QWTensorShape([rows * outCols], .f16))
        let parameters = try allocator.allocate(QWTensorShape([rows * ctrlStride], .f32))
        gdnProjection = projection
        attentionProjection = QWTensorValue(region: projection.region, offset: projection.offset,
            shape: QWTensorShape([max(1, rows * geometry.attnTotalN)], .f16))
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
        q = views(projection, stride: geometry.attnTotalN, start: 0,
                  count: geometry.qProjOut, dtype: .f16)
        k = views(projection, stride: geometry.attnTotalN, start: geometry.attnKRow0,
                  count: geometry.kvChannels, dtype: .f16)
        v = views(projection, stride: geometry.attnTotalN, start: geometry.attnVRow0,
                  count: geometry.kvChannels, dtype: .f16)
        outputs = views(inputs, stride: outCols, start: 0,
                        count: outCols, dtype: .f16)
    }

    func gdnRow(_ row: Int) throws -> (qkv: QWTensorValue, z: QWTensorValue, beta: QWTensorValue, decay: QWTensorValue) {
        try validate(row)
        return (qkv[row], z[row], beta[row], decay[row])
    }

    func attentionRow(_ row: Int) throws -> (q: QWTensorValue, k: QWTensorValue, v: QWTensorValue) {
        try validate(row)
        return (q[row], k[row], v[row])
    }

    func outputRow(_ row: Int) throws -> QWTensorValue {
        try validate(row)
        return outputs[row]
    }

    func gdnPipeline(forActiveRows activeRows: Int, rowPaired: Bool = false) throws -> any MTLComputePipelineState {
        guard (2...rows).contains(activeRows),
              let pipeline = (rowPaired ? gdnRowPairedPipelines : gdnPipelines)[activeRows] else {
            throw QWEngineDriverError.state("invalid batched trunk active row count")
        }
        return pipeline
    }

    private func validate(_ row: Int) throws {
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
                let fgbC = 2 * geometry.gdnLowRankDim + geometry.gdnVHeads
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
                try mtpBatchedMLP.encodeProjection(enc, input: activeNormalized, weight: wAttn,
                                                   output: mtpBatchedTrunk.attentionProjection,
                                                   activeRows: activeRows)
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
        func rowSlice(_ base: QWTensorValue, row: Int, cols: Int, dt: QWDType) -> QWTensorValue {
            let elemBytes = (dt == .f16 || dt == .bf16) ? 2 : 4
            return QWTensorValue(region: base.region, offset: base.offset + row * cols * elemBytes,
                                 shape: QWTensorShape([cols], dt))
        }
        switch weights.kind {
        case .gdn, .gdnMoe:
            let w = weights.gdn!, index = gdnIndexMap[layer]
            let gdnOut = prefixRows(mtpBatchedTrunk.outputInputs, rows: activeRows,
                                    cols: geometry.gdnZChannels, dt: .f16)
            if geometry.gdnVectorDecay, let fB = w.fB, let gB = w.gB, let fgb16 = psc.fgb16 {
                let fgbC = 2 * geometry.gdnLowRankDim + geometry.gdnVHeads
                try checkpoint.encodeBatchedKDA(
                    enc, conv: gdnConv, decoder: gdnDecode,
                    layer: index, activeRows: activeRows,
                    gdnProjection: mtpBatchedTrunk.gdnProjection,
                    fgbProjection: prefixRows(fgb16, rows: activeRows, cols: fgbC, dt: .f16),
                    fBWeight: fB, gBWeight: gB, aLog: w.aLog, dtBias: w.dtBias,
                    convWeight: w.conv, normWeight: w.normW,
                    zScratch: psc.z16,
                    ring: session.ring[index], state: session.rec[index],
                    output: gdnOut)
            } else {
                try checkpoint.encodeBatchedGDN(
                    enc, conv: gdnConv, decoder: gdnDecode,
                    layer: index, activeRows: activeRows,
                    gdnProjection: mtpBatchedTrunk.gdnProjection, controls: mtpBatchedTrunk.controls,
                    convWeight: w.conv, normWeight: w.normW,
                    ring: session.ring[index], state: session.rec[index],
                    output: gdnOut)
            }
        case .fullAttention, .moe:
            let fullAttnIdx = attnIndexMap[layer]
            let views = session.attnKVViews[fullAttnIdx]
            let attnOutCols = weights.attn!.o.k
            let attnOut = prefixRows(mtpBatchedTrunk.outputInputs, rows: activeRows,
                                     cols: attnOutCols, dt: .f16)
            if let mla = mlaOperator, let mlaW = weights.attn?.mla {
                mla.encodeLatentPrepAndPool(
                    enc, ln1: activeNormalized, mlaW: mlaW,
                    kv: views, fullAttnLayer: fullAttnIdx,
                    pos0: session.position, rows: activeRows)
                enc.memoryBarrier(scope: .buffers)
                let qNormView = prefixRows(mla.qResidScratch, rows: activeRows, cols: mlaW.qBProj.k, dt: .f16)
                let qFullView = prefixRows(mla.mlaQFullScratch, rows: activeRows, cols: mlaW.qBProj.n, dt: .f16)
                try mtpBatchedMLP.encodeProjection(enc, input: qNormView, weight: mlaW.qBProj,
                                                   output: qFullView, activeRows: activeRows)
                mla.encodeScoreSelectAbsorbAttendAndProjUV(
                    enc, mlaW: mlaW, kv: views, attOut: attnOut,
                    fullAttnLayer: fullAttnIdx, pos0: session.position, rows: activeRows)
                enc.memoryBarrier(scope: .buffers)
            } else {
                guard let partials, let attnDecode else {
                    throw QWEngineDriverError.state("GQA verification requires shared attention partials and attnDecode")
                }
                let keys = prefixRows(psc.attKR16, rows: activeRows, cols: geometry.kvChannels, dt: .f16)
                let values = prefixRows(psc.attV16, rows: activeRows, cols: geometry.kvChannels, dt: .f16)
                let qgate = prefixRows(psc.attQR16, rows: activeRows, cols: geometry.qProjOut, dt: .f16)
                try ropeApply.encodeExtractQK(enc, hal: hal,
                    attnY: prefixRows(mtpBatchedTrunk.attentionProjection, rows: activeRows, cols: geometry.attnTotalN, dt: .f16),
                    qOut: qgate, kOut: keys, vOut: values,
                    wq: weights.attn!.wq, wk: weights.attn!.wkTiled,
                    positionStart: session.position, count: activeRows)
                enc.memoryBarrier(scope: .buffers)
                if let lsa = lsaOperator {
                    lsa.encodeIndexAndCluster(enc, qgate: qgate, kIn: keys,
                                              fullAttnLayer: fullAttnIdx,
                                              pos0: session.position, rows: activeRows)
                }
                encodePagedKVWrite(enc, kv: views, k: keys, v: values, startPos: session.position, t: activeRows)
                enc.memoryBarrier(scope: .buffers)
                let mtpSparseThreshold = max(32_768, lsaConfig.denseBypassThreshold * max(1, activeRows) * 4)
                if let lsa = lsaOperator, session.position >= mtpSparseThreshold {
                    lsa.encodeSelectBlocks(enc, fullAttnLayer: fullAttnIdx,
                                           pos0: session.position, rows: activeRows)
                    let activeKeys = min(session.position + activeRows, lsaConfig.alignedBudget + 640)
                    let splits = QWAttnDecode.splits(forKeys: activeKeys)
                    for r in 0..<activeRows {
                        let qRow = rowSlice(qgate, row: r, cols: geometry.qProjOut, dt: .f16)
                        let kRow = rowSlice(keys, row: r, cols: geometry.kvChannels, dt: .f16)
                        let vRow = rowSlice(values, row: r, cols: geometry.kvChannels, dt: .f16)
                        let oRow = rowSlice(attnOut, row: r, cols: geometry.qHeads * geometry.headDim, dt: .f16)
                        let bList = QWTensorValue(
                            region: lsa.blockListScratch.region,
                            offset: lsa.blockListScratch.offset + r * lsa.listStride * 4,
                            shape: QWTensorShape([lsa.listStride], .i32))
                        let bCount = QWTensorValue(
                            region: lsa.blockCountScratch.region,
                            offset: lsa.blockCountScratch.offset + r * 4,
                            shape: QWTensorShape([1], .i32))
                        try attnDecode.encodeSparse(
                            enc, qgate: qRow, kv: views, knew: kRow, vnew: vRow,
                            partials: partials, out: oRow,
                            blockList: bList, blockCount: bCount,
                            C: session.position + r, splits: splits, listStride: lsa.listStride)
                        enc.memoryBarrier(scope: .buffers)
                    }
                } else if activeRows <= 8 {
                    try attnDecode.encodeBatched(enc,
                        qgate: qgate,
                        kv: views, knew: keys, vnew: values, partials: partials, out: attnOut,
                        cBase: session.position, T: activeRows,
                        splits: QWAttnDecode.splits(forKeys: session.position + activeRows), sharedPrefix: true)
                    enc.memoryBarrier(scope: .buffers)
                } else {
                    let c0 = activeRows / 2
                    let c1 = activeRows - c0
                    let outCols = geometry.qHeads * geometry.headDim
                    try attnDecode.encodeBatched(enc,
                        qgate: prefixRows(qgate, rows: c0, cols: geometry.qProjOut, dt: .f16),
                        kv: views,
                        knew: prefixRows(keys, rows: c0, cols: geometry.kvChannels, dt: .f16),
                        vnew: prefixRows(values, rows: c0, cols: geometry.kvChannels, dt: .f16),
                        partials: partials,
                        out: prefixRows(attnOut, rows: c0, cols: outCols, dt: .f16),
                        cBase: session.position, T: c0,
                        splits: QWAttnDecode.splits(forKeys: session.position + c0), sharedPrefix: true)
                    enc.memoryBarrier(scope: .buffers)
                    try attnDecode.encodeBatched(enc,
                        qgate: qgate.slice(byteOffset: c0 * geometry.qProjOut * 2, count: c1 * geometry.qProjOut),
                        kv: views,
                        knew: keys.slice(byteOffset: c0 * geometry.kvChannels * 2, count: c1 * geometry.kvChannels),
                        vnew: values.slice(byteOffset: c0 * geometry.kvChannels * 2, count: c1 * geometry.kvChannels),
                        partials: partials,
                        out: attnOut.slice(byteOffset: c0 * outCols * 2, count: c1 * outCols),
                        cBase: session.position + c0, T: c1,
                        splits: QWAttnDecode.splits(forKeys: session.position + activeRows), sharedPrefix: true)
                    enc.memoryBarrier(scope: .buffers)
                }
            }
        case .hca, .hcaMoe:
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
}
