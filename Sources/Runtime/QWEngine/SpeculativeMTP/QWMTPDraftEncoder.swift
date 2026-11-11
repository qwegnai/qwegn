import Metal
import QWAlloc
import QWAttn
import QWCore
import QWGEMM
import QWHAL
import QWHead
import QWKV
import QWMLP
import QWNorm
import QWRope

struct QWMTPDraftWeights {
    let preFcNormEmbedding: QWTensorValue
    let preFcNormHidden: QWTensorValue
    let fc: QWWeightView
    let inputNorm: QWTensorValue
    let qkvProj: QWWeightView
    let q: QWWeightView, k: QWWeightView, v: QWWeightView
    let o: QWWeightView
    let wq: QWTensorValue?, wkTiled: QWTensorValue?
    let postAttnNorm: QWTensorValue
    let gu: QWMLPWeight, down: QWWeightView
    let norm: QWTensorValue
    var moe: QWEngineDriver.MoEW? = nil
    var mla: QWMLAAttnOperator.LayerWeights? = nil
}

struct QWMTPDraftScratch {
    let geometry: QWModelGeometry
    let embedding: QWTensorValue
    let concat: QWTensorValue
    let enorm: QWTensorValue, hnorm: QWTensorValue
    let fcOut32: QWTensorValue, fcOut16: QWTensorValue
    let ln1: QWTensorValue
    let qkv32: QWTensorValue, qkv16: QWTensorValue
    let q16: QWTensorValue, k16: QWTensorValue, v16: QWTensorValue
    let qRope: QWTensorValue, kRope: QWTensorValue
    let attOut: QWTensorValue, oProjOut: QWTensorValue
    let ln2: QWTensorValue
    let mlp32: QWTensorValue, mlpIntermediate: QWTensorValue
    let downOut: QWTensorValue, postNorm: QWTensorValue
    let draftLogits: QWTensorValue
    let greedySelector: QWMTPGreedySelector
    let gsFc: QWTensorValue, gsHidden: QWTensorValue
    let gsAttn: QWTensorValue, gsMLP: QWTensorValue
    /// L1: the draft head's own group sums (K = hidden, 40 f32). The head
    /// GEMV reads postNorm — the same input the bf16 head reads — so the
    /// q4g route needs its own xgroupsum output; it is never shared with
    let gsHead: QWTensorValue
    let partials: QWMRegion
    let moe: QWEngineDriver.MoEScratch?

    init(geometry: QWModelGeometry, allocator: QWSubAllocator, partials: QWMRegion,
         intermediate: Int? = nil, vocabulary: Int? = nil) throws {
        self.geometry = geometry
        let defaultInter = geometry.intermediate > 0
            ? geometry.intermediate
            : max(128, ((geometry.moeIntermediate + 127) / 128) * 128)
        let intermediate = intermediate ?? defaultInter
        let vocabulary = vocabulary ?? geometry.vocab
        let maxInter = max(geometry.intermediate, defaultInter)
        guard intermediate > 0, intermediate <= maxInter, intermediate % 128 == 0,
              vocabulary > 0, vocabulary <= geometry.vocab,
              partials.length >= QWAttnDecode.decodePartialsBytes(geometry: geometry) else {
            throw QWEngineDriverError.state("MTP scratch geometry or attention partials are invalid")
        }
        let h = geometry.hidden, kv = geometry.kvHeads * geometry.headDim
        func allocate(_ n: Int, _ type: QWDType = .f16) throws -> QWTensorValue {
            try allocator.allocate(QWTensorShape([n], type))
        }
        func view(_ tensor: QWTensorValue, start: Int, count: Int) -> QWTensorValue {
            QWTensorValue(region: tensor.region, offset: tensor.offset + start * 2,
                          shape: QWTensorShape([count], .f16))
        }
        embedding = try allocate(h)
        concat = try allocate(2 * h)
        enorm = view(concat, start: 0, count: h)
        hnorm = view(concat, start: h, count: h)
        fcOut32 = try allocate(h, .f32)
        fcOut16 = try allocate(h)
        ln1 = try allocate(h)
        qkv32 = try allocate(geometry.qProjOut + 2 * kv, .f32)
        qkv16 = try allocate(geometry.qProjOut + 2 * kv)
        q16 = view(qkv16, start: 0, count: geometry.qProjOut)
        k16 = view(qkv16, start: geometry.qProjOut, count: kv)
        v16 = view(qkv16, start: geometry.qProjOut + kv, count: kv)
        qRope = try allocate(geometry.qProjOut)
        kRope = try allocate(kv)
        attOut = try allocate(geometry.qHeads * geometry.headDim)
        oProjOut = try allocate(h)
        ln2 = try allocate(h)
        mlp32 = try allocate(intermediate, .f32)
        mlpIntermediate = try allocate(intermediate)
        downOut = try allocate(h)
        postNorm = try allocate(h)
        draftLogits = try allocate(vocabulary, .f32)
        greedySelector = try QWMTPGreedySelector(geometry: geometry, allocator: allocator, vocabulary: vocabulary)
        gsFc = try allocate(2 * h / 128, .f32)
        gsHidden = try allocate(h / 128, .f32)
        gsAttn = try allocate(geometry.qHeads * geometry.headDim / 128, .f32)
        gsMLP = try allocate(intermediate / 128, .f32)
        gsHead = try allocate(h / 128, .f32)
        self.partials = partials
        if geometry.numExperts > 0 {
            moe = try QWEngineDriver.makeMoEScratch(allocate, rows: 1, geometry: geometry)
        } else {
            moe = nil
        }
    }
}

/// One MTP block on a serial encoder. Absolute RoPE position is independent
/// of the physical pair-cache row, because all-accept rounds leave pair gaps.
final class QWMTPDraftEncoder {
    let geometry: QWModelGeometry
    let scratch: QWMTPDraftScratch
    private let hal: QWHal
    private let weights: QWMTPDraftWeights
    private let rope: QWRopeApply
    private let ropeRows: Int
    var positionCapacity: Int { ropeRows }
    private let attention: QWAttnDecode
    private let mlaOperator: QWMLAAttnOperator?
    private let mlaLayerIndex: Int
    private let fcPlan: QWGemmPlan
    private let qkvPlan: QWGemmPlan
    private let qBPlan: QWGemmPlan?
    private let outPlan: QWGemmPlan
    private let guPlan: QWGemmPlan
    private let downPlan: QWGemmPlan

    init(geometry: QWModelGeometry, hal: QWHal, weights: QWMTPDraftWeights, scratch: QWMTPDraftScratch,
         rope: QWRopeApply, ropeRows: Int, kvMode: QWKVMode = .f16,
         mlaOperator: QWMLAAttnOperator? = nil, mlaLayerIndex: Int = 0) throws {
        self.geometry = geometry
        let h = geometry.hidden, kv = geometry.kvHeads * geometry.headDim
        let i = weights.down.k
        let expectedQKVCols = weights.mla != nil
            ? (geometry.qLoraRank + geometry.kvLoraRank + geometry.indexerHeadDim)
            : (geometry.qProjOut + 2 * kv)
        let expectedOutK = weights.mla != nil
            ? (geometry.qHeads * geometry.vHeadDim)
            : (geometry.qHeads * geometry.headDim)
        guard ropeRows > 0,
              weights.fc.n == h, weights.fc.k == 2 * h,
              weights.qkvProj.n == expectedQKVCols, weights.qkvProj.k == h,
              weights.o.n == h, weights.o.k == expectedOutK,
              weights.gu.view.n == 2 * i, weights.gu.view.k == h, weights.down.n == h,
              (weights.moe != nil || scratch.mlpIntermediate.shape.elementCount == i) else {
            throw QWEngineDriverError.state("MTP projection geometry does not match scratch")
        }
        for norm in [weights.preFcNormEmbedding, weights.preFcNormHidden, weights.inputNorm,
                     weights.postAttnNorm, weights.norm] {
            guard norm.shape.dtype == .f16, norm.shape.elementCount == h else {
                throw QWEngineDriverError.state("MTP norm must be FP16 [hidden]")
            }
        }
        if weights.mla == nil {
            guard let wq = weights.wq, let wkTiled = weights.wkTiled,
                  wq.shape.dtype == .f16, wq.shape.elementCount == geometry.headDim,
                  wkTiled.shape.dtype == .f16, wkTiled.shape.elementCount == kv else {
                throw QWEngineDriverError.state("MTP Q/K norm geometry is invalid")
            }
        }
        self.hal = hal
        self.weights = weights
        self.scratch = scratch
        self.rope = rope
        self.ropeRows = ropeRows
        self.mlaOperator = mlaOperator
        self.mlaLayerIndex = mlaLayerIndex
        attention = try QWAttnDecode(hal: hal, kvMode: kvMode, geometry: geometry)
        fcPlan = QWGemm.plan(QWGemmShape(m: 1, n: h, k: 2 * h), dtype: .q4g, epi: .none, outDtype: .f16)
        qkvPlan = QWGemm.plan(QWGemmShape(m: 1, n: weights.qkvProj.n, k: h), dtype: .q4g, epi: .none, outDtype: .f16)
        if let mlaW = weights.mla {
            qBPlan = QWGemm.plan(QWGemmShape(m: 1, n: mlaW.qBProj.n, k: mlaW.qBProj.k), dtype: .q4g, epi: .none, outDtype: .f16)
        } else {
            qBPlan = nil
        }
        outPlan = QWGemm.plan(QWGemmShape(m: 1, n: h, k: weights.o.k), dtype: .q4g, epi: .none, outDtype: .f16)
        guPlan = QWGemm.plan(QWGemmShape(m: 1, n: 2 * i, k: h), dtype: .q4g, epi: .siluMulPair, outDtype: .f16)
        downPlan = QWGemm.plan(QWGemmShape(m: 1, n: h, k: i), dtype: .q4g, epi: .none, outDtype: .f16)
        for kernel in ["qwg_rmsnorm", "qwg_rmsnorm_residual", "qwg_rmsnorm_from_rot",
                       "qwg_mtp_pre_fc_norm",
                       "qwg_fwht32_f16", "qwg_gemv_q4g_v4", "qwg_gemv_q4g_v4_h16",
                       "qwg_gemv_q4g_silu_pair", "qwg_gemv_q4g_silu_pair_h16",
                       "qwg_cast_f32_to_f16", "qwg_gemv_f16_v4", "qwg_mtp_seed_kv",
                       "qwg_kv_paged_write_general",
                       "qwg_mtp_argmax_checked", "qwg_mtp_head_gemv_q4g_argmax",
                       "qwg_head_argmax_fin"] {
            _ = try hal.pipeline(kernel)
        }
        if !weights.fc.isIncumbentContainer {
            for optionalKernel in [
                QWGemmAQ.decodeKernelName(bits: weights.fc.bits, group: weights.fc.groupSize, isSHVQ: weights.fc.isSHVQ),
                QWGemmAQ.siluPairKernelName(bits: weights.gu.view.bits, group: weights.gu.view.groupSize, isSHVQ: weights.gu.view.isSHVQ),
                QWGemmAQ.headArgmaxKernelName(bits: weights.fc.bits, group: weights.fc.groupSize, isSHVQ: weights.fc.isSHVQ),
                QWGemmAQ.kernelName(bits: weights.fc.bits, group: weights.fc.groupSize, isSHVQ: weights.fc.isSHVQ)
            ].compactMap({ $0 }) {
                _ = try hal.pipeline(optionalKernel)
            }
        }
    }

    private func encodePreAttn(_ enc: any MTLComputeCommandEncoder, embedding: QWTensorValue,
                               hidden: QWTensorValue, kv: (k: QWKVLayerView, v: QWKVLayerView),
                               cacheRows: Int, absolutePosition: Int) throws {
        let h = geometry.hidden
        guard embedding.shape.dtype == .f16, embedding.shape.elementCount == h,
              hidden.shape.dtype == .f16, hidden.shape.elementCount == h,
              cacheRows >= 0, cacheRows < QWKVCache.maxTokensPerSeq,
              absolutePosition >= 0, absolutePosition < ropeRows else {
            throw QWEngineDriverError.state("MTP input shape or position is invalid")
        }
        let pages = cacheRows / QWKVCache.pageTokens + 1
        guard kv.k.table.shape.elementCount >= pages, kv.v.table.shape.elementCount >= pages,
              kv.k.pageBytes == QWKVCache.pageBytes(mode: kv.k.mode, geometry: geometry),
              kv.k.pageBytes == kv.v.pageBytes else {
            throw QWEngineDriverError.state("MTP KV capacity or dtype is invalid")
        }
        let s = scratch, w = weights
        try QWNorm.encode(enc, hal: hal, variant: .rmsnorm,
                          x: embedding, gamma: w.preFcNormEmbedding, out: s.enorm,
                          geometry: geometry)
        try QWNorm.encode(enc, hal: hal, variant: .rmsnormFromRot,
                          x: hidden, gamma: w.preFcNormHidden, out: s.hnorm,
                          geometry: geometry)
        try project(enc, x: s.concat, gs: s.gsFc, w: w.fc,
                    plan: fcPlan, y32: s.fcOut32, y16: s.fcOut16)
        try QWNorm.encode(enc, hal: hal, variant: .rmsnorm,
                          x: s.fcOut16, gamma: w.inputNorm, out: s.ln1,
                          geometry: geometry)
    }

    private func encodeQKV(_ enc: any MTLComputeCommandEncoder, embedding: QWTensorValue,
                           hidden: QWTensorValue, kv: (k: QWKVLayerView, v: QWKVLayerView),
                           cacheRows: Int, absolutePosition: Int) throws {
        try encodePreAttn(enc, embedding: embedding, hidden: hidden, kv: kv,
                          cacheRows: cacheRows, absolutePosition: absolutePosition)
        let s = scratch, w = weights
        if let mla = mlaOperator, let mlaW = w.mla {
            let latent1 = QWTensorValue(region: mla.mlaLatentScratch.region,
                                        offset: mla.mlaLatentScratch.offset,
                                        shape: QWTensorShape([w.qkvProj.n], .f16))
            try project(enc, x: s.ln1, gs: s.gsHidden, w: w.qkvProj,
                        plan: qkvPlan, y32: s.qkv32, y16: latent1)
            mla.encodeLatentPrepAndPool(enc, ln1: s.ln1, mlaW: mlaW,
                                        kv: kv, fullAttnLayer: mlaLayerIndex,
                                        pos0: cacheRows, rows: 1)
            return
        }
        guard let wq = w.wq, let wkTiled = w.wkTiled else {
            throw QWEngineDriverError.state("MTP Q/K norms are missing")
        }
        try project(enc, x: s.ln1, gs: s.gsHidden, w: w.qkvProj,
                    plan: qkvPlan, y32: s.qkv32, y16: s.qkv16)
        try rope.encodeQK(enc, hal: hal, q: s.q16, k: s.k16,
                          wq: wq, wk: wkTiled, qOut: s.qRope, kOut: s.kRope,
                          positionStart: absolutePosition, count: 1)
    }

    /// Prompt pair rows depend only on supplied trunk hidden, not earlier MTP
    /// outputs. Seed K/V without attention readout, output projection, or MLP.
    func encodePairCache(_ enc: any MTLComputeCommandEncoder, embedding: QWTensorValue,
                         hidden: QWTensorValue, kv: (k: QWKVLayerView, v: QWKVLayerView),
                         cacheRows: Int, absolutePosition: Int) throws {
        try encodeQKV(enc, embedding: embedding, hidden: hidden, kv: kv,
                       cacheRows: cacheRows, absolutePosition: absolutePosition)
        if mlaOperator != nil, weights.mla != nil {
            return
        }
        var seedQ8: UInt32
        switch kv.k.mode {
        case .f16: seedQ8 = 0
        case .q8:  seedQ8 = 1
        case .q4:  seedQ8 = 2
        @unknown default: throw QWEngineDriverError.state("MTP KV unknown mode")
        }
        if !geometry.is27BKVShape {
            enc.setComputePipelineState(try hal.pipeline("qwg_kv_paged_write_general"))
            enc.setBuffer(kv.k.pool.buffer, offset: kv.k.pool.base, index: 0)
            enc.setBuffer(kv.v.pool.buffer, offset: kv.v.pool.base, index: 1)
            kv.k.table.bind(enc, index: 2)
            kv.v.table.bind(enc, index: 3)
            scratch.kRope.bind(enc, index: 4)
            scratch.v16.bind(enc, index: 5)
            var row = UInt32(cacheRows)
            var tOne: UInt32 = 1
            enc.setBytes(&row, length: 4, index: 6)
            enc.setBytes(&tOne, length: 4, index: 7)
            enc.setBytes(&seedQ8, length: 4, index: 8)
            enc.dispatchThreadgroups(MTLSize(width: geometry.kvHeads, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            return
        }
        enc.setComputePipelineState(try hal.pipeline("qwg_mtp_seed_kv"))
        scratch.kRope.bind(enc, index: 0)
        scratch.v16.bind(enc, index: 1)
        enc.setBuffer(kv.k.pool.buffer, offset: kv.k.pool.base, index: 2)
        enc.setBuffer(kv.v.pool.buffer, offset: kv.v.pool.base, index: 3)
        kv.k.table.bind(enc, index: 4)
        kv.v.table.bind(enc, index: 5)
        var row = UInt32(cacheRows)
        enc.setBytes(&row, length: 4, index: 6)
        enc.setBytes(&seedQ8, length: 4, index: 7)
        enc.dispatchThreads(MTLSize(width: 1024, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    func encode(_ enc: any MTLComputeCommandEncoder, embedding: QWTensorValue,
                hidden: QWTensorValue, kv: (k: QWKVLayerView, v: QWKVLayerView),
                cacheRows: Int, absolutePosition: Int) throws {
        try encodeQKV(enc, embedding: embedding, hidden: hidden, kv: kv,
                       cacheRows: cacheRows, absolutePosition: absolutePosition)
        let s = scratch, w = weights
        if let mla = mlaOperator, let mlaW = w.mla, let qBPlan {
            let qResid1 = QWTensorValue(region: mla.qResidScratch.region,
                                        offset: mla.qResidScratch.offset,
                                        shape: QWTensorShape([mlaW.qBProj.k], .f16))
            let qFull1 = QWTensorValue(region: mla.mlaQFullScratch.region,
                                       offset: mla.mlaQFullScratch.offset,
                                       shape: QWTensorShape([mlaW.qBProj.n], .f16))
            try project(enc, x: qResid1, gs: s.gsHidden, w: mlaW.qBProj,
                        plan: qBPlan, y32: s.qkv32, y16: qFull1)
            mla.encodeScoreSelectAbsorbAttendAndProjUV(
                enc, mlaW: mlaW, kv: kv, attOut: s.attOut,
                fullAttnLayer: mlaLayerIndex, pos0: cacheRows, rows: 1)
        } else {
            try attention.encode(enc, qgate: s.qRope, kv: kv, knew: s.kRope, vnew: s.v16,
                                 partials: s.partials, out: s.attOut,
                                 C: cacheRows, splits: QWAttnDecode.splits(forKeys: cacheRows + 1))
        }
        try project(enc, x: s.attOut, gs: s.gsAttn, w: w.o,
                    plan: outPlan, y32: s.fcOut32, y16: s.oProjOut)
        try QWNorm.encode(enc, hal: hal, variant: .rmsnormResidual,
                          x: s.oProjOut, gamma: w.postAttnNorm, out: s.ln2, residual: s.fcOut16,
                          geometry: geometry)
        if let moe = w.moe, let moeScratch = s.moe {
            var mEnc = enc
            try QWMoEChain.encode(enc: &mEnc, hal: hal, moe: moe, s: moeScratch,
                                  x16: s.ln2, dn32: s.fcOut32, dn16: s.downOut,
                                  rows: 1, geometry: geometry)
        } else {
            try project(enc, x: s.ln2, gs: s.gsHidden, w: w.gu.view,
                        plan: guPlan, y32: s.mlp32, y16: s.mlpIntermediate)
            try QWGemm.encodeFwht32(enc, hal: hal, x: s.mlpIntermediate)
            try project(enc, x: s.mlpIntermediate, gs: s.gsMLP, w: w.down,
                        plan: downPlan, y32: s.fcOut32, y16: s.downOut)
        }
        try QWNorm.encode(enc, hal: hal, variant: .rmsnormResidual,
                          x: s.downOut, gamma: w.norm, out: s.postNorm, residual: s.fcOut16,
                          geometry: geometry)
    }

    /// A configured q4g sidecar takes precedence; otherwise use the target
    /// head's selected storage, including its q4g route.
    func encodeHead(_ enc: any MTLComputeCommandEncoder, weight: QWTensorValue,
                    draftHead: QWMTPTDraftHead? = nil) throws {
        try encodeHead(enc, storage: .f16(weight), draftHead: draftHead)
    }

    func encodeHead(_ enc: any MTLComputeCommandEncoder, storage: QWHeadStorage,
                    draftHead: QWMTPTDraftHead? = nil) throws {
        if let draftHead {
            try draftHead.encode(enc, hal: hal, x: scratch.postNorm,
                                 gs: scratch.gsHead, y: scratch.draftLogits)
            return
        }
        switch storage {
        case .f16(let weight):
            try QWHead.encodeF16(enc, hal: hal, w: weight, x: scratch.postNorm,
                                 y: scratch.draftLogits, geometry: geometry,
                                 N: scratch.draftLogits.shape.elementCount)
        case .q4g(let weight):
            try QWHead.encodeQ4g(enc, hal: hal, weight: weight,
                x: scratch.postNorm, gs: scratch.gsHead, y: scratch.draftLogits)
        }
    }

    func encodeHeadAndSelect(_ enc: any MTLComputeCommandEncoder, storage: QWHeadStorage,
                             draftHead: QWMTPTDraftHead? = nil,
                             writeLogits: Bool = true,
                             destination: QWTensorValue? = nil) throws {
        let vocabulary = scratch.draftLogits.shape.elementCount
        let outToken = destination ?? scratch.greedySelector.token
        let q4View: QWWeightView?
        if let draftHead {
            for buf in draftHead.mappedBuffers {
                guard scratch.draftLogits.region.buffer !== buf else {
                    throw QWEngineDriverError.state("draft head output aliases a mapped plane")
                }
            }
            q4View = draftHead.view
        } else {
            switch storage {
            case .f16:
                q4View = nil
            case .q4g(let weight):
                q4View = weight.view
            }
        }
        guard let view = q4View else {
            try encodeHead(enc, storage: storage, draftHead: draftHead)
            try scratch.greedySelector.encode(enc, hal: hal, logits: scratch.draftLogits,
                                              ordering: .numericalDraft, destination: outToken)
            return
        }
        if !view.isIncumbentContainer {
            let nTG = (vocabulary + 31) / 32
            let keyBytes = nTG * 4
            guard view.n == vocabulary, view.k == geometry.hidden,
                  scratch.partials.length >= 2 * keyBytes,
                  outToken.shape.dtype == .i32, outToken.shape.elementCount == 1,
                  (outToken.region.base + outToken.offset) % 4 == 0 else {
                throw QWEngineDriverError.state("draft fused head argmax geometry is invalid")
            }
            let keysHi = QWTensorValue(region: scratch.partials, offset: 0,
                                       shape: QWTensorShape([nTG], .i32))
            let keysLo = QWTensorValue(region: scratch.partials, offset: keyBytes,
                                       shape: QWTensorShape([nTG], .i32))
            try QWGemmAQ.encodeHeadArgmax(enc, hal: hal, view: view, x: scratch.postNorm,
                                          y: scratch.draftLogits, keysHi: keysHi, keysLo: keysLo,
                                          writeLogits: writeLogits)
            var nTG32 = UInt32(nTG)
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(try hal.pipeline("qwg_head_argmax_fin"))
            keysHi.bind(enc, index: 0)
            keysLo.bind(enc, index: 1)
            outToken.bind(enc, index: 2)
            enc.setBytes(&nTG32, length: 4, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
            return
        }
        let nTG = QWHead.gemvArgmaxSlots(vocabulary)
        let keyBytes = nTG * 4
        guard view.n == vocabulary, view.k == geometry.hidden,
              scratch.partials.length >= 2 * keyBytes,
              outToken.shape.dtype == .i32, outToken.shape.elementCount == 1,
              (outToken.region.base + outToken.offset) % 4 == 0 else {
            throw QWEngineDriverError.state("draft fused head argmax geometry is invalid")
        }
        let keysHi = QWTensorValue(region: scratch.partials, offset: 0,
                                   shape: QWTensorShape([nTG], .i32))
        let keysLo = QWTensorValue(region: scratch.partials, offset: keyBytes,
                                   shape: QWTensorShape([nTG], .i32))
        var nU = UInt32(vocabulary), kU = UInt32(view.k)
        var writeFlag = UInt32(writeLogits ? 1 : 0), nTG32 = UInt32(nTG)
        enc.setComputePipelineState(try hal.pipeline("qwg_mtp_head_gemv_q4g_argmax"))
        view.blocks.bind(enc, index: 0)
        scratch.postNorm.bind(enc, index: 1)
        scratch.draftLogits.bind(enc, index: 3)
        enc.setBytes(&nU, length: 4, index: 4)
        enc.setBytes(&kU, length: 4, index: 5)
        keysHi.bind(enc, index: 6)
        keysLo.bind(enc, index: 7)
        enc.setBytes(&writeFlag, length: 4, index: 8)
        enc.dispatchThreadgroups(MTLSize(width: nTG, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.memoryBarrier(scope: .buffers)
        enc.setComputePipelineState(try hal.pipeline("qwg_head_argmax_fin"))
        keysHi.bind(enc, index: 0)
        keysLo.bind(enc, index: 1)
        outToken.bind(enc, index: 2)
        enc.setBytes(&nTG32, length: 4, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
    }

    private func project(_ enc: any MTLComputeCommandEncoder, x: QWTensorValue,
                         gs: QWTensorValue, w: QWWeightView, plan: QWGemmPlan,
                         y32: QWTensorValue, y16: QWTensorValue) throws {
        _ = (gs, y32)
        try QWGemm.encodeH16(enc, hal: hal, plan: plan, x: x, wQ4: w, y16: y16)
    }
}
