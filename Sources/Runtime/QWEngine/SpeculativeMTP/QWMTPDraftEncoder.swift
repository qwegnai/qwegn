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
    var qkvBias: QWTensorValue? = nil
    var oBias: QWTensorValue? = nil
    var gatedNorms: QWEngineDriver.GatedMTPNorms? = nil
}

struct QWMTPDraftScratch {
    static let maxProposalRows = 15

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
    let draftLogitHistory: QWTensorValue
    let selector: QWTokenSelector
    let gsFc: QWTensorValue, gsHidden: QWTensorValue
    let gsAttn: QWTensorValue, gsMLP: QWTensorValue
    /// L1: the draft head's own group sums (K = hidden, 40 f32).
    let gsHead: QWTensorValue
    let partials: QWMRegion
    let moe: QWEngineDriver.MoEScratch?

    static func allocationBytes(geometry: QWModelGeometry, intermediate: Int? = nil, vocabulary: Int? = nil) -> Int {
        let mtpCfgInter = (geometry.mtpConfig?.isMoE == false) ? (geometry.mtpConfig?.intermediateSize ?? 0) : 0
        let defaultInter = mtpCfgInter > 0 ? mtpCfgInter
            : (geometry.intermediate > 0 ? geometry.intermediate : max(128, ((geometry.moeIntermediate + 127) / 128) * 128))
        let inter = intermediate ?? defaultInter
        let vocab = vocabulary ?? geometry.vocab
        let h = geometry.hidden, kv = max(1, geometry.kvHeads * geometry.headDim)
        let qOut = max(1, geometry.qProjOut), qCols = max(128, geometry.qHeads * geometry.headDim)
        func aligned(_ n: Int, _ dt: QWDType = .f16) -> Int {
            let b = Int((Double(max(1, n)) * dt.bytesPerParam).rounded())
            return ((b + 255) / 256) * 256
        }
        let tensors = aligned(h) * 6 + aligned(2 * h) + aligned(h, .f32) + aligned(qOut + 2 * kv, .f32)
            + aligned(qOut + 2 * kv) + aligned(qOut) + aligned(kv) + aligned(qCols)
            + aligned(inter, .f32) + aligned(inter) + aligned(vocab, .f32)
            + aligned(maxProposalRows * vocab, .f32)
            + aligned(2 * h / 128, .f32) + aligned(h / 128, .f32) * 2
            + aligned(max(1, qCols / 128), .f32) + aligned(inter / 128, .f32)
        let sel = QWTokenSelector.allocationBytes(vocabLimit: geometry.vocab, vocabulary: vocab, maxRows: maxProposalRows)
        let mtpIsMoE = geometry.mtpConfig?.isMoE ?? (geometry.numExperts > 0)
        let moeB = (mtpIsMoE && geometry.numExperts > 0)
            ? QWEngineDriver.moeScratchAllocationBytes(rows: 1, geometry: geometry) : 0
        return tensors + sel + moeB
    }

    init(geometry: QWModelGeometry, allocator: QWSubAllocator, partials: QWMRegion,
         intermediate: Int? = nil, vocabulary: Int? = nil) throws {
        self.geometry = geometry
        let mtpCfgInter = (geometry.mtpConfig?.isMoE == false)
            ? (geometry.mtpConfig?.intermediateSize ?? 0) : 0
        let defaultInter = mtpCfgInter > 0
            ? mtpCfgInter
            : (geometry.intermediate > 0
                ? geometry.intermediate
                : max(128, ((geometry.moeIntermediate + 127) / 128) * 128))
        let intermediate = intermediate ?? defaultInter
        let vocabulary = vocabulary ?? geometry.vocab
        let maxInter = max(geometry.intermediate, max(mtpCfgInter, defaultInter))
        let minPartials = geometry.usesLatentOrCompressedAttention
            ? QWEngineDriver.draftHeadArgmaxPartialsBytes(vocab: vocabulary)
            : QWAttnDecode.decodePartialsBytes(geometry: geometry)
        guard intermediate > 0, intermediate <= maxInter, intermediate % 128 == 0,
              vocabulary > 0, vocabulary <= geometry.vocab,
              partials.length >= minPartials else {
            throw QWEngineDriverError.state("MTP scratch geometry or attention partials are invalid")
        }
        let h = geometry.hidden, kv = max(1, geometry.kvHeads * geometry.headDim)
        let qOut = max(1, geometry.qProjOut)
        let qCols = max(128, geometry.qHeads * geometry.headDim)
        func allocate(_ n: Int, _ type: QWDType = .f16) throws -> QWTensorValue {
            try allocator.allocate(QWTensorShape([max(1, n)], type))
        }
        func view(_ tensor: QWTensorValue, start: Int, count: Int) -> QWTensorValue {
            QWTensorValue(region: tensor.region, offset: tensor.offset + start * 2,
                          shape: QWTensorShape([max(1, count)], .f16))
        }
        embedding = try allocate(h)
        concat = try allocate(2 * h)
        enorm = view(concat, start: 0, count: h)
        hnorm = view(concat, start: h, count: h)
        fcOut32 = try allocate(h, .f32)
        fcOut16 = try allocate(h)
        ln1 = try allocate(h)
        qkv32 = try allocate(qOut + 2 * kv, .f32)
        qkv16 = try allocate(qOut + 2 * kv)
        q16 = view(qkv16, start: 0, count: qOut)
        k16 = view(qkv16, start: qOut, count: kv)
        v16 = view(qkv16, start: qOut + kv, count: kv)
        qRope = try allocate(qOut)
        kRope = try allocate(kv)
        attOut = try allocate(qCols)
        oProjOut = try allocate(h)
        ln2 = try allocate(h)
        mlp32 = try allocate(intermediate, .f32)
        mlpIntermediate = try allocate(intermediate)
        downOut = try allocate(h)
        postNorm = try allocate(h)
        draftLogits = try allocate(vocabulary, .f32)
        draftLogitHistory = try allocate(Self.maxProposalRows * vocabulary, .f32)
        selector = try QWTokenSelector(
            vocabLimit: geometry.vocab, allocator: allocator,
            vocabulary: vocabulary, maxRows: Self.maxProposalRows)
        gsFc = try allocate(2 * h / 128, .f32)
        gsHidden = try allocate(h / 128, .f32)
        gsAttn = try allocate(max(1, qCols / 128), .f32)
        gsMLP = try allocate(intermediate / 128, .f32)
        gsHead = try allocate(h / 128, .f32)
        self.partials = partials
        let mtpIsMoE = geometry.mtpConfig?.isMoE ?? (geometry.numExperts > 0)
        if mtpIsMoE && geometry.numExperts > 0 {
            moe = try QWEngineDriver.makeMoEScratch(allocate, rows: 1, geometry: geometry)
        } else {
            moe = nil
        }
    }

    func draftLogitRow(_ row: Int) throws -> QWTensorValue {
        let vocab = draftLogits.shape.elementCount
        guard (0..<Self.maxProposalRows).contains(row) else {
            throw QWEngineDriverError.state("MTP draft logit history row \(row) out of range")
        }
        return QWTensorValue(
            region: draftLogitHistory.region,
            offset: draftLogitHistory.offset + row * vocab * 4,
            shape: QWTensorShape([vocab], .f32))
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
    weak var driver: QWEngineDriver?
    var hasColdMoE: Bool { weights.moe?.coldStreamer != nil }
    var nextStepHidden: QWTensorValue {
        (weights.gatedNorms != nil && geometry.useSandwichNorm) ? scratch.fcOut16 : scratch.postNorm
    }

    static func allocationBytes(geometry: QWModelGeometry, intermediate: Int? = nil, vocabulary: Int? = nil) -> Int {
        QWMTPDraftScratch.allocationBytes(geometry: geometry, intermediate: intermediate, vocabulary: vocabulary)
    }

    init(geometry: QWModelGeometry, hal: QWHal, weights: QWMTPDraftWeights, scratch: QWMTPDraftScratch,
         rope: QWRopeApply, ropeRows: Int, kvMode: QWKVMode = .f16,
         mlaOperator: QWMLAAttnOperator? = nil, mlaLayerIndex: Int = 0,
         driver: QWEngineDriver? = nil) throws {
        self.driver = driver
        self.geometry = geometry
        let h = geometry.hidden, kv = geometry.kvHeads * geometry.headDim
        let i = weights.down.k
        let ropeDim = geometry.usesDecoupledRopeMLA ? geometry.qkRopeHeadDim : 0
        let idxDim = geometry.mlaHasIndexer ? max(32, geometry.indexerHeadDim) : 0
        let expectedQKVCols: Int
        if let mlaW = weights.mla {
            let effIdx = (mlaW.denseOnly || !geometry.mlaHasIndexer) ? 0 : idxDim
            let effGate = mlaW.oGateInQKV ? (geometry.qHeads * geometry.vHeadDim) : 0
            let fallbackCols = (geometry.mlaHasDirectQ ? 0 : geometry.qLoraRank)
                + geometry.kvLoraRank + ropeDim + effIdx + effGate
            expectedQKVCols = mlaOperator?.latentStride(for: mlaW) ?? fallbackCols
        } else {
            expectedQKVCols = geometry.qProjOut + 2 * kv
        }
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
        let allowBF16Norm = weights.gatedNorms != nil
        for norm in [weights.preFcNormEmbedding, weights.preFcNormHidden, weights.inputNorm,
                     weights.postAttnNorm, weights.norm] {
            let validDtype = norm.shape.dtype == .f16 || (allowBF16Norm && norm.shape.dtype == .bf16)
            guard validDtype, norm.shape.elementCount == h else {
                throw QWEngineDriverError.state("MTP norm must be FP16/BF16 [hidden]")
            }
        }
        if weights.mla == nil {
            if geometry.descriptor.hasQKNorm {
                guard let wq = weights.wq, let wkTiled = weights.wkTiled,
                      wq.shape.dtype == .f16, wq.shape.elementCount == geometry.headDim,
                      wkTiled.shape.dtype == .f16, wkTiled.shape.elementCount == kv else {
                    throw QWEngineDriverError.state("MTP Q/K norm geometry is invalid")
                }
            } else {
                guard (weights.wq == nil) == (weights.wkTiled == nil),
                      weights.wq.map({ $0.shape.dtype == .f16 && $0.shape.elementCount == geometry.headDim }) ?? true,
                      weights.wkTiled.map({ $0.shape.dtype == .f16 && $0.shape.elementCount == kv }) ?? true else {
                    throw QWEngineDriverError.state("MTP Q/K norm geometry is invalid")
                }
            }
            if let qkvBias = weights.qkvBias {
                guard qkvBias.shape.dtype == .f16, qkvBias.shape.elementCount == geometry.attnTotalN,
                      (qkvBias.region.base + qkvBias.offset) % 16 == 0 else {
                    throw QWEngineDriverError.state("MTP QKV bias geometry is invalid")
                }
            }
            if let oBias = weights.oBias {
                guard oBias.shape.dtype == .f16, oBias.shape.elementCount == h,
                      (oBias.region.base + oBias.offset) % 16 == 0 else {
                    throw QWEngineDriverError.state("MTP O bias geometry is invalid")
                }
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
                       "qwg_add_bias_f16", "qwg_mtp_pre_fc_norm",
                       "qwg_fwht32_f16", "qwg_gemv_q4g_v4", "qwg_gemv_q4g_v4_h16",
                       "qwg_gemv_q4g_silu_pair", "qwg_gemv_q4g_silu_pair_h16",
                       "qwg_cast_f32_to_f16", "qwg_gemv_f16_v4", "qwg_mtp_seed_kv",
                       "qwg_kv_paged_write_general",
                       "qwg_mtp_argmax_checked", "qwg_mtp_head_gemv_q4g_argmax",
                       "qwg_head_argmax_fin"] {
            _ = try hal.pipeline(kernel)
        }
        let draftViews: [QWWeightView] = [
            weights.fc, weights.qkvProj, weights.o, weights.gu.view, weights.down
        ] + (weights.mla.map { [$0.qBProj] } ?? [])
        for v in draftViews where !v.isIncumbentContainer {
            for optionalKernel in [
                QWGemmAQ.decodeKernelName(bits: v.bits, group: v.groupSize, isSHVQ: v.isSHVQ),
                QWGemmAQ.headArgmaxKernelName(bits: v.bits, group: v.groupSize, isSHVQ: v.isSHVQ),
                QWGemmAQ.kernelName(bits: v.bits, group: v.groupSize, isSHVQ: v.isSHVQ)
            ].compactMap({ $0 }) {
                _ = try hal.pipeline(optionalKernel)
            }
        }
        let mtpSwiGLUClamp = geometry.routedSwiGLUClamp(layer: geometry.numLayers)
        if let mtpSwiGLUClamp {
            _ = try hal.pipeline("qwg_gemv_q4g_silu_pair_h16", swigluClamp: mtpSwiGLUClamp)
        }
        if !weights.gu.view.isIncumbentContainer,
           let siluName = QWGemmAQ.siluPairKernelName(bits: weights.gu.view.bits,
                                                      group: weights.gu.view.groupSize,
                                                      isSHVQ: weights.gu.view.isSHVQ) {
            _ = try hal.pipeline(siluName, swigluClamp: mtpSwiGLUClamp)
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
        if let gn = w.gatedNorms {
            try QWNorm.encodeGatedRMSNorm(
                enc, hal: hal, x: embedding, weights: gn.enorm,
                out: s.enorm, hidden: h, applyFwht32: w.fc.isRotated)
            try QWNorm.encodeGatedRMSNorm(
                enc, hal: hal, x: hidden, weights: gn.hnorm,
                out: s.hnorm, hidden: h, applyFwht32: w.fc.isRotated)
            try project(enc, x: s.concat, gs: s.gsFc, w: w.fc,
                        plan: fcPlan, y32: s.fcOut32, y16: s.fcOut16)
            try QWNorm.encodeGatedRMSNorm(
                enc, hal: hal, x: s.fcOut16, weights: gn.input,
                out: s.ln1, hidden: h, applyFwht32: w.qkvProj.isRotated)
            return
        }
        if geometry.hidden % 256 == 0 {
            try QWNorm.encodeMTPPreFcNorm(enc, hal: hal, embeddings: embedding, hidden: hidden,
                                          previous: hidden, gammaEmb: w.preFcNormEmbedding,
                                          gammaHid: w.preFcNormHidden, out: s.concat,
                                          tStart: 0, pairs: 1, geometry: geometry)
        } else {
            try QWNorm.encode(enc, hal: hal, variant: .rmsnorm,
                              x: embedding, gamma: w.preFcNormEmbedding, out: s.enorm,
                              geometry: geometry)
            try QWNorm.encode(enc, hal: hal, variant: .rmsnormFromRot,
                              x: hidden, gamma: w.preFcNormHidden, out: s.hnorm,
                              geometry: geometry)
        }
        try project(enc, x: s.concat, gs: s.gsFc, w: w.fc,
                    plan: fcPlan, y32: s.fcOut32, y16: s.fcOut16)
        try QWNorm.encode(enc, hal: hal, variant: .rmsnorm,
                          x: s.fcOut16, gamma: w.inputNorm, out: s.ln1,
                          geometry: geometry)
    }

    private func mropePositionsForRow(cacheRows: Int, mropeDelta: Int) -> [Int32]? {
        guard let mla = mlaOperator, mla.mropeLayout != .none1D, mropeDelta != 0 else { return nil }
        return [Int32](repeating: Int32(cacheRows + mropeDelta), count: 3)
    }

    private func encodeQKV(_ enc: any MTLComputeCommandEncoder, embedding: QWTensorValue,
                           hidden: QWTensorValue, kv: (k: QWKVLayerView, v: QWKVLayerView),
                           cacheRows: Int, absolutePosition: Int, mropeDelta: Int = 0) throws {
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
                                        pos0: cacheRows, rows: 1,
                                        mropePositions: mropePositionsForRow(cacheRows: cacheRows, mropeDelta: mropeDelta))
            return
        }
        try project(enc, x: s.ln1, gs: s.gsHidden, w: w.qkvProj,
                    plan: qkvPlan, y32: s.qkv32, y16: s.qkv16)
        let effectivePos = geometry.mropeLayout != .none1D ? max(0, absolutePosition + mropeDelta) : absolutePosition
        if let qkvBias = w.qkvBias {
            try rope.encodeExtractQK(enc, hal: hal, attnY: s.qkv16,
                                     qOut: s.qRope, kOut: s.kRope, vOut: s.v16,
                                     wq: w.wq, wk: w.wkTiled, qkvBias: qkvBias,
                                     positionStart: effectivePos, count: 1)
        } else {
            try rope.encodeQK(enc, hal: hal, q: s.q16, k: s.k16,
                              wq: w.wq, wk: w.wkTiled, qOut: s.qRope, kOut: s.kRope,
                              positionStart: effectivePos, count: 1)
        }
    }

    /// Prompt pair rows depend only on supplied trunk hidden, not earlier MTP
    /// outputs. Seed K/V without attention readout, output projection, or MLP.
    func encodePairCache(_ enc: any MTLComputeCommandEncoder, embedding: QWTensorValue,
                         hidden: QWTensorValue, kv: (k: QWKVLayerView, v: QWKVLayerView),
                         cacheRows: Int, absolutePosition: Int, mropeDelta: Int = 0) throws {
        try encodeQKV(enc, embedding: embedding, hidden: hidden, kv: kv,
                       cacheRows: cacheRows, absolutePosition: absolutePosition,
                       mropeDelta: mropeDelta)
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
        if !geometry.matchesKVStride1024Kernel {
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
                cacheRows: Int, absolutePosition: Int, mropeDelta: Int = 0) throws {
        var current = enc
        try encode(&current, embedding: embedding, hidden: hidden, kv: kv,
                   cacheRows: cacheRows, absolutePosition: absolutePosition,
                   mropeDelta: mropeDelta)
    }

    func encode(_ enc: inout any MTLComputeCommandEncoder, embedding: QWTensorValue,
                hidden: QWTensorValue, kv: (k: QWKVLayerView, v: QWKVLayerView),
                cacheRows: Int, absolutePosition: Int,
                mropeDelta: Int = 0,
                driver: QWEngineDriver? = nil, cb: (any MTLCommandBuffer)? = nil) throws {
        try encodeQKV(enc, embedding: embedding, hidden: hidden, kv: kv,
                       cacheRows: cacheRows, absolutePosition: absolutePosition,
                       mropeDelta: mropeDelta)
        let s = scratch, w = weights
        if let mla = mlaOperator, let mlaW = w.mla, let qBPlan {
            let qIn1 = mla.isDirectQ
                ? s.ln1
                : QWTensorValue(region: mla.qResidScratch.region,
                                offset: mla.qResidScratch.offset,
                                shape: QWTensorShape([mlaW.qBProj.k], .f16))
            let qFull1 = QWTensorValue(region: mla.mlaQFullScratch.region,
                                       offset: mla.mlaQFullScratch.offset,
                                       shape: QWTensorShape([mlaW.qBProj.n], .f16))
            try project(enc, x: qIn1, gs: s.gsHidden, w: mlaW.qBProj,
                        plan: qBPlan, y32: s.qkv32, y16: qFull1)
            mla.encodeScoreSelectAbsorbAttendAndProjUV(
                enc, mlaW: mlaW, kv: kv, attOut: s.attOut,
                fullAttnLayer: mlaLayerIndex, pos0: cacheRows, rows: 1,
                ln1: s.ln1,
                mropePositions: mropePositionsForRow(cacheRows: cacheRows, mropeDelta: mropeDelta))
        } else {
            try attention.encode(enc, qgate: s.qRope, kv: kv, knew: s.kRope, vnew: s.v16,
                                 partials: s.partials, out: s.attOut,
                                 C: cacheRows, splits: QWAttnDecode.splits(forKeys: cacheRows + 1, kvHeads: geometry.kvHeads, gpuCores: hal.caps.gpuCores))
        }
        try project(enc, x: s.attOut, gs: s.gsAttn, w: w.o,
                    plan: outPlan, y32: s.fcOut32, y16: s.oProjOut)
        if let oBias = w.oBias {
            try QWNorm.encodeAddBiasF16(enc, hal: hal, x: s.oProjOut, bias: oBias,
                                        rows: 1, cols: geometry.hidden)
        }
        if let gn = w.gatedNorms, geometry.useSandwichNorm {
            let emitMoE = w.moe != nil && s.moe != nil
            try QWNorm.encodeGatedSandwichResidual(
                enc, hal: hal,
                branchOut: s.oProjOut, residual: s.fcOut16,
                postWeights: gn.postSelfAttn, nextPreWeights: gn.postAttn, out: s.ln2,
                hidden: geometry.hidden, applyFwht32: w.gu.view.isRotated,
                xRot32: emitMoE ? s.moe?.xRot32 : nil,
                xPlain32: emitMoE ? s.moe?.xPlain32 : nil)
        } else {
            try QWNorm.encode(enc, hal: hal, variant: .rmsnormResidual,
                              x: s.oProjOut, gamma: w.postAttnNorm, out: s.ln2, residual: s.fcOut16,
                              geometry: geometry)
        }
        if let moe = w.moe, let moeScratch = s.moe {
            let activeDriver = driver ?? self.driver
            if moe.coldStreamer != nil && activeDriver == nil {
                throw QWEngineDriverError.state("MTP draft MoE cold experts require an attached driver")
            }
            let prevID = ObjectIdentifier(enc)
            try QWMoEChain.encode(enc: &enc, hal: hal, moe: moe, s: moeScratch,
                                  x16: s.ln2, dn32: s.fcOut32, dn16: s.downOut,
                                  rows: 1, geometry: geometry,
                                  driver: moe.coldStreamer != nil ? activeDriver : nil,
                                  cb: cb, skipOut32: true,
                                  preBridgedF32: (w.gatedNorms != nil && geometry.useSandwichNorm)
                                      ? (plain: moeScratch.xPlain32, rotated: moeScratch.xRot32) : nil)
            if ObjectIdentifier(enc) != prevID {
                hal.touchKeepAlive(enc)
            }
        } else {
            let fuseMtpFwht = !w.gu.view.isIncumbentContainer
                && w.down.isRotated
                && (w.gu.view.n / 2) > 0
                && ((w.gu.view.n / 2) % 32 == 0)
            try QWGemm.encodeH16(
                enc, hal: hal, plan: guPlan, x: s.ln2, wQ4: w.gu.view,
                y16: s.mlpIntermediate,
                swigluClamp: geometry.routedSwiGLUClamp(layer: geometry.numLayers),
                fuseFwht32: fuseMtpFwht
            )
            if w.down.isRotated && !fuseMtpFwht {
                try QWGemm.encodeFwht32(enc, hal: hal, x: s.mlpIntermediate)
            }
            try project(enc, x: s.mlpIntermediate, gs: s.gsMLP, w: w.down,
                        plan: downPlan, y32: s.fcOut32, y16: s.downOut)
        }
        if let gn = w.gatedNorms, geometry.useSandwichNorm {
            try QWNorm.encodeGatedSandwichResidual(
                enc, hal: hal,
                branchOut: s.downOut, residual: s.fcOut16,
                postWeights: gn.postFFN, nextPreWeights: gn.sharedHead, out: s.postNorm,
                hidden: geometry.hidden, applyFwht32: true)
        } else {
            try QWNorm.encode(enc, hal: hal, variant: .rmsnormResidual,
                              x: s.downOut, gamma: w.norm, out: s.postNorm, residual: s.fcOut16,
                              geometry: geometry)
        }
    }

    /// A configured q4g sidecar takes precedence; otherwise use the target
    /// head's selected storage, including its q4g route.
    func encodeHead(_ enc: any MTLComputeCommandEncoder, weight: QWTensorValue,
                    draftHead: QWMTPTDraftHead? = nil,
                    destination: QWTensorValue? = nil) throws {
        try encodeHead(enc, storage: .f16(weight), draftHead: draftHead, destination: destination)
    }

    func encodeHead(_ enc: any MTLComputeCommandEncoder, storage: QWHeadStorage,
                    draftHead: QWMTPTDraftHead? = nil,
                    destination: QWTensorValue? = nil) throws {
        let softcap = geometry.finalLogitSoftcapping ?? 0.0
        let logitsOut = destination ?? scratch.draftLogits
        let count = logitsOut.shape.elementCount
        if let draftHead {
            try draftHead.encode(enc, hal: hal, x: scratch.postNorm,
                                 gs: scratch.gsHead, y: logitsOut)
            if softcap > 0 {
                enc.memoryBarrier(scope: .buffers)
                try QWHead.encodeSoftcapF32(enc, hal: hal, logits: logitsOut,
                                            count: count, softcap: softcap)
            }
            return
        }
        let unrotateF16 = weights.gatedNorms != nil && geometry.useSandwichNorm
            && hal.weightBasis.rotatesActivations
        switch storage {
        case .f16(let weight):
            if unrotateF16 {
                try QWGemm.encodeFwht32(enc, hal: hal, x: scratch.postNorm)
                enc.memoryBarrier(scope: .buffers)
            }
            try QWHead.encodeF16(enc, hal: hal, w: weight, x: scratch.postNorm,
                                 y: logitsOut, geometry: geometry,
                                 N: count, softcap: softcap)
        case .bf16(let weight):
            if unrotateF16 {
                try QWGemm.encodeFwht32(enc, hal: hal, x: scratch.postNorm)
                enc.memoryBarrier(scope: .buffers)
            }
            try QWHead.encodeBF16(enc, hal: hal, w: weight, x: scratch.postNorm,
                                  y: logitsOut, geometry: geometry,
                                  N: count, softcap: softcap)
        case .q4g(let weight):
            try QWHead.encodeQ4g(enc, hal: hal, weight: weight,
                x: scratch.postNorm, gs: scratch.gsHead, y: logitsOut, softcap: softcap)
        }
    }

    func encodeHeadAndSelect(_ enc: any MTLComputeCommandEncoder, storage: QWHeadStorage,
                             draftHead: QWMTPTDraftHead? = nil,
                             writeLogits: Bool = true,
                             destination: QWTensorValue? = nil) throws {
        let vocabulary = scratch.draftLogits.shape.elementCount
        let outToken = destination ?? scratch.selector.token
        let softcap = geometry.finalLogitSoftcapping ?? 0.0
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
            case .f16, .bf16:
                q4View = nil
            case .q4g(let weight):
                q4View = weight.view
            }
        }
        guard softcap <= 0, let view = q4View else {
            try encodeHead(enc, storage: storage, draftHead: draftHead)
            enc.memoryBarrier(scope: .buffers)
            try scratch.selector.encodeGreedy(enc, hal: hal, logits: scratch.draftLogits,
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
