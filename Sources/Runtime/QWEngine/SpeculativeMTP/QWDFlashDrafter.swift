// QWDFlashDrafter.swift — DFlash 2 parallel block-diffusion drafter
// for Qwen3.8-27B (5 sliding-window layers, 2-tap dynamic causal conv,
// non-causal block attention, and rank-256 pairwise CandidateSelector).
import Darwin
import Foundation
import Metal
import QWAlloc
import QWCore
import QWGEMM
import QWHead
import QWHAL
import QWLoad
import QWMTP
import QWQuant

/// The drafter config and session slot types live in QWDFlashDrafter+Config.swift.
final class QWDFlashDrafter: @unchecked Sendable {
    let geometry: QWModelGeometry
    struct LayerWeights {
        let ln1: QWTensorValue
        let ln2: QWTensorValue
        let qNorm: QWTensorValue
        let kNorm: QWTensorValue
        let attnConvBase: QWTensorValue?
        let attnConvProj: QWTensorValue?
        let mlpConvBase: QWTensorValue?
        let mlpConvProj: QWTensorValue?
        let qkvProj: QWWeightView
        let kvProj: QWWeightView
        let oProj: QWWeightView
        let gateUpProj: QWWeightView
        let downProj: QWWeightView
    }

    static let defaultPackPath: String = QWDFlashConfig.defaultPackPath

    let config: QWDFlashConfig
    let packPath: String
    let targetLayerSlotMap: [Int]
    private let weightBuffer: any MTLBuffer
    private let scratchBuffer: any MTLBuffer

    private let fc: QWWeightView
    private let hiddenNorm: QWTensorValue
    let finalNorm: QWTensorValue
    private let selectorHiddenProj: QWTensorValue?
    private let predecessorCodebook: QWTensorValue?
    private let successorCodebook: QWTensorValue?
    let dspineBridge: DSpineBridgeWeights?
    private let invFreq: QWTensorValue
    let layers: [LayerWeights]

    let targetHidden: QWTensorValue
    private let hCtx: QWTensorValue
    private let kvCtx: QWTensorValue
    let propX: QWTensorValue
    let propNorm: QWTensorValue
    let propRot: QWTensorValue
    private let propDyn: QWTensorValue?
    private let qkvProp: QWTensorValue
    private let attnOut: QWTensorValue
    private let mlpInter: QWTensorValue
    let propDown: QWTensorValue
    private let draftLogits: QWTensorValue
    private let top16PartialsVal: QWTensorValue
    private let top16PartialsId: QWTensorValue
    private let unaryOut: QWTensorValue
    let candidatesOut: QWTensorValue
    private let hiddenProjOut: QWTensorValue?
    let draftTokensOut: QWTensorValue
    let dspineScratch: DSpineBridgeScratch?

    private let psoCaptureTarget: any MTLComputePipelineState
    private let psoRMSNorm: any MTLComputePipelineState
    private let psoF16GemvBatched: any MTLComputePipelineState
    private let psoConvApply: any MTLComputePipelineState
    private let psoRopeKVCtx: any MTLComputePipelineState
    private let psoRopeQKProp: any MTLComputePipelineState
    private let psoAttn: any MTLComputePipelineState
    private let psoTop16Stage1: any MTLComputePipelineState
    private let psoTop16Stage2: any MTLComputePipelineState
    private let psoSelectorGreedy: any MTLComputePipelineState
    private let psoGemvT4: any MTLComputePipelineState
    private let psoGemmBatched: any MTLComputePipelineState
    private let psoDownBatched: [Int: any MTLComputePipelineState]
    private let psoDownRowPaired: [Int: any MTLComputePipelineState]
    let psoGateUpFwht32Batched: [Int: any MTLComputePipelineState]
    let dspinePSO: DSpinePipelines
    let quantPSOs: [DrafterQuantKey: DrafterQuantPipelines]

    let sessionSlots: [QWDFlashSessionSlot]

    init(geometry: QWModelGeometry, hal: QWHal, packPath: String, drafts: Int = 7, selectorTopK: Int? = nil) throws {
        try QWDFlashConfig.validateDrafter(
            targetDescriptor: geometry.descriptor,
            packPath: packPath,
            drafts: drafts,
            selectorTopK: selectorTopK
        )
        self.geometry = geometry
        let resolvedURL = packPath.canonicalFileURL
        guard FileManager.default.fileExists(atPath: resolvedURL.path) else {
            throw QWEngineDriverError.packMissing(resolvedURL.path)
        }
        self.packPath = resolvedURL.path
        self.config = QWDFlashConfig.load(from: resolvedURL, topKOverride: selectorTopK)
        var slotMap = [Int](repeating: -1, count: geometry.numLayers)
        for (slot, lid) in config.targetLayerIds.enumerated() where (0..<geometry.numLayers).contains(lid) {
            slotMap[lid] = slot
        }
        self.targetLayerSlotMap = slotMap

        let pack = try QWQuant.Pack(url: resolvedURL, descriptor: .qwen38_27b, skipIdentityCheck: true)
        let io = PlaneIO(pack: pack, policy: LoadPolicy(descriptor: pack.descriptor))

        let wCapacity = Self.computeWeightCapacity(pack: pack, isDSpine: config.isDSpine)
        guard let wBuf = hal.device.makeBuffer(length: wCapacity, options: .storageModeShared) else {
            throw QWEngineDriverError.state("QWDFlashDrafter: failed to allocate weight buffer")
        }
        self.weightBuffer = wBuf
        let wSub = QWSubAllocator(region: QWMRegion(buffer: wBuf, base: 0, length: wCapacity))

        func loadQ4(_ name: String, n: Int, k: Int) throws -> QWWeightView {
            try Self.loadQuantWeight(pack: pack, io: io, wSub: wSub, hal: hal, name: name, n: n, k: k)
        }

        func concatQ4(_ specs: [(name: String, n: Int)], k: Int) throws -> (QWWeightView, [QWWeightView]) {
            try Self.concatQuantWeights(pack: pack, io: io, wSub: wSub, hal: hal, specs: specs, k: k)
        }

        let psoBf16F16 = try hal.pipeline("qwg_bf16_to_f16")
        let cvtTmp = hal.device.makeBuffer(length: 16 * 1024 * 1024, options: .storageModeShared)!

        func loadF16(_ name: String, _ elems: Int) throws -> QWTensorValue {
            let e = try pack.entry(name)
            let total = e.shape.reduce(1, *)
            guard total == elems else {
                throw QWEngineDriverError.weightMissing(name, "DFlash shape \(total) != \(elems)")
            }
            let t = try wSub.allocate(QWTensorShape([elems], .f16))
            if elems >= 4096, e.rawOff != nil {
                let chunk = 8 * 1024 * 1024
                var done = 0
                while done < elems {
                    let n = min(chunk, elems - done)
                    try io.loadRaw(name, elemOffset: done, elems: n, into: cvtTmp.contents())
                    guard let (cb, enc) = hal.begin() else {
                        throw QWEngineDriverError.state("DFlash bf16->f16 cmdbuffer")
                    }
                    var start = UInt32(0), nn = UInt32(n)
                    enc.setComputePipelineState(psoBf16F16)
                    enc.setBuffer(cvtTmp, offset: 0, index: 0)
                    enc.setBuffer(t.region.buffer, offset: t.region.base + t.offset + done * 2, index: 1)
                    enc.setBytes(&start, length: 4, index: 2)
                    enc.setBytes(&nn, length: 4, index: 3)
                    enc.dispatchThreadgroups(MTLSize(width: (n + 255) / 256, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    hal.end(cb, enc)
                    cb.waitUntilCompleted()
                    done += n
                }
                return t
            }
            let src = try pack.rawBf16(name)
            let dst = t.region.buffer.contents().advanced(by: t.region.base + t.offset)
                .assumingMemoryBound(to: UInt16.self)
            for i in 0..<elems { dst[i] = QWQuant.f32tof16Bits(src[i]) }
            return t
        }

        func loadRawBF16(_ name: String, _ elems: Int) throws -> QWTensorValue {
            let t = try wSub.allocate(QWTensorShape([elems], .f16))
            let dst = t.region.buffer.contents().advanced(by: t.region.base + t.offset)
            try io.loadRaw(name, elemOffset: 0, elems: elems, into: dst)
            return t
        }

        let H = config.hiddenSize
        let I = config.intermediateSize
        let V = config.vocabSize
        let R = config.selectorRank
        let qDim = config.numAttentionHeads * config.headDim
        let kvDim = config.numKeyValueHeads * config.headDim
        let hasConv = config.convKernelSize > 0
        let convProjDim = hasConv ? (2 * config.convKernelSize * (H / config.convGroupSize)) : 0

        self.fc = try loadQ4("fc.weight", n: H, k: config.targetLayerIds.count * H)
        self.hiddenNorm = try loadF16("hidden_norm.weight", H)
        self.finalNorm = try loadF16("norm.weight", H)
        if config.isDSpine {
            self.selectorHiddenProj = nil
            self.predecessorCodebook = nil
            self.successorCodebook = nil
            self.dspineBridge = try Self.loadDSpineBridgeWeights(
                config: config, wSub: wSub, loadF16: loadF16, loadRawBF16: loadRawBF16
            )
        } else {
            self.selectorHiddenProj = try loadF16("candidate_selector.hidden_projection.weight", R * H)
            self.predecessorCodebook = try loadRawBF16("candidate_selector.predecessor_codebook", V * R)
            self.successorCodebook = try loadRawBF16("candidate_selector.successor_codebook", V * R)
            self.dspineBridge = nil
        }

        let halfDim = config.headDim / 2
        let invFreqT = try wSub.allocate(QWTensorShape([halfDim], .f32))
        let invPtr = invFreqT.region.buffer.contents().advanced(by: invFreqT.region.base + invFreqT.offset)
            .assumingMemoryBound(to: Float.self)
        for i in 0..<halfDim {
            let exp = Float(2 * i) / Float(config.headDim)
            invPtr[i] = 1.0 / powf(config.ropeTheta, exp)
        }
        self.invFreq = invFreqT

        var loadedLayers: [LayerWeights] = []
        for l in 0..<config.numLayers {
            let p = "layers.\(l)."
            let ln1 = try loadF16(p + "input_layernorm.weight", H)
            let ln2 = try loadF16(p + "post_attention_layernorm.weight", H)
            let qNorm = try loadF16(p + "self_attn.q_norm.weight", config.headDim)
            let kNorm = try loadF16(p + "self_attn.k_norm.weight", config.headDim)
            let attnConvBase = hasConv ? try loadF16(p + "attention_conv.base_kernel", 2 * config.convKernelSize * H) : nil
            let attnConvProj = hasConv ? try loadF16(p + "attention_conv.kernel_projection.weight", convProjDim * H) : nil
            let mlpConvBase = hasConv ? try loadF16(p + "mlp_conv.base_kernel", 2 * config.convKernelSize * H) : nil
            let mlpConvProj = hasConv ? try loadF16(p + "mlp_conv.kernel_projection.weight", convProjDim * H) : nil

            let (qkvBig, _) = try concatQ4([
                (p + "self_attn.q_proj.weight", qDim),
                (p + "self_attn.k_proj.weight", kvDim),
                (p + "self_attn.v_proj.weight", kvDim)
            ], k: H)
            let kvBlockOffset = qDim * qkvBig.rowBytes
            let kvByteCount = 2 * kvDim * qkvBig.rowBytes
            let kvBlocks = QWTensorValue(region: qkvBig.blocks.region,
                                         offset: qkvBig.blocks.offset + kvBlockOffset,
                                         shape: QWTensorShape([kvByteCount / 2], .f16))
            guard let kvView = qkvBig.subview(blocks: kvBlocks, n: 2 * kvDim, k: H) else {
                throw QWEngineDriverError.weightMissing(p + "self_attn.k_proj.weight", "invalid kvView")
            }
            let oProj = try loadQ4(p + "self_attn.o_proj.weight", n: H, k: qDim)
            let (guBig, _) = try concatQ4([
                (p + "mlp.gate_proj.weight", I),
                (p + "mlp.up_proj.weight", I)
            ], k: H)
            let downProj = try loadQ4(p + "mlp.down_proj.weight", n: H, k: I)
            loadedLayers.append(LayerWeights(
                ln1: ln1, ln2: ln2, qNorm: qNorm, kNorm: kNorm,
                attnConvBase: attnConvBase, attnConvProj: attnConvProj,
                mlpConvBase: mlpConvBase, mlpConvProj: mlpConvProj,
                qkvProj: qkvBig, kvProj: kvView, oProj: oProj,
                gateUpProj: guBig, downProj: downProj
            ))
        }
        self.layers = loadedLayers
        wSub.markSteadyState()
        pack.adviseDontNeed()

        let sCapacity = config.maxCacheSize > 2048 ? 992_000_000 : 240_000_000
        guard let sBuf = hal.device.makeBuffer(length: sCapacity, options: .storageModeShared) else {
            throw QWEngineDriverError.state("QWDFlashDrafter: failed to allocate scratch buffer")
        }
        self.scratchBuffer = sBuf
        let sSub = QWSubAllocator(region: QWMRegion(buffer: sBuf, base: 0, length: sCapacity))
        let maxCtx = config.maxCacheSize
        let maxB = max(8, config.blockSize)
        let maxD = maxB - 1

        self.targetHidden = try sSub.allocate(QWTensorShape([maxCtx * config.targetLayerIds.count * H], .f16))
        self.hCtx = try sSub.allocate(QWTensorShape([maxCtx * H], .f16))
        self.kvCtx = try sSub.allocate(QWTensorShape([maxCtx * 2 * kvDim], .f16))
        self.propX = try sSub.allocate(QWTensorShape([maxB * H], .f32))
        self.propNorm = try sSub.allocate(QWTensorShape([maxB * H], .f16))
        self.propRot = try sSub.allocate(QWTensorShape([maxB * H], .f16))
        self.propDyn = hasConv ? try sSub.allocate(QWTensorShape([maxB * convProjDim], .f32)) : nil
        self.qkvProp = try sSub.allocate(QWTensorShape([maxB * (qDim + 2 * kvDim)], .f16))
        self.attnOut = try sSub.allocate(QWTensorShape([maxB * qDim], .f16))
        self.mlpInter = try sSub.allocate(QWTensorShape([maxB * I], .f16))
        self.propDown = try sSub.allocate(QWTensorShape([maxB * H], .f16))
        self.draftLogits = try sSub.allocate(QWTensorShape([maxD * V], .f32))
        self.top16PartialsVal = try sSub.allocate(QWTensorShape([maxD * 256 * 16], .f32))
        self.top16PartialsId = try sSub.allocate(QWTensorShape([maxD * 256 * 16], .i32))
        self.unaryOut = try sSub.allocate(QWTensorShape([maxD * 16], .f32))
        self.candidatesOut = try sSub.allocate(QWTensorShape([maxD * 16], .i32))
        self.hiddenProjOut = config.isDSpine ? nil : try sSub.allocate(QWTensorShape([maxD * R], .f32))
        self.draftTokensOut = try sSub.allocate(QWTensorShape([maxB], .i32))
        self.dspineScratch = config.isDSpine
            ? try Self.allocateDSpineScratch(config: config, sSub: sSub, maxB: maxB, maxD: maxD)
            : nil

        let layerKVElements = maxCtx * kvDim
        var slots: [QWDFlashSessionSlot] = []
        for _ in 0..<2 {
            let kSlab = try sSub.allocate(QWTensorShape([config.numLayers * layerKVElements], .f16))
            let vSlab = try sSub.allocate(QWTensorShape([config.numLayers * layerKVElements], .f16))
            let kViews = (0..<config.numLayers).map { l in
                QWTensorValue(region: kSlab.region, offset: kSlab.offset + l * layerKVElements * 2,
                              shape: QWTensorShape([layerKVElements], .f16))
            }
            let vViews = (0..<config.numLayers).map { l in
                QWTensorValue(region: vSlab.region, offset: vSlab.offset + l * layerKVElements * 2,
                              shape: QWTensorShape([layerKVElements], .f16))
            }
            slots.append(QWDFlashSessionSlot(kSlab: kSlab, vSlab: vSlab, kLayers: kViews, vLayers: vViews))
        }
        self.sessionSlots = slots
        sSub.markSteadyState()

        self.psoCaptureTarget = try hal.pipeline("qwg_dflash_capture_target_hidden")
        self.psoRMSNorm = try hal.pipeline("qwg_dflash_rmsnorm")
        self.psoF16GemvBatched = try hal.pipeline("qwg_dflash_f16_gemv_batched")
        self.psoConvApply = try hal.pipeline("qwg_dflash_conv_apply")
        self.psoRopeKVCtx = try hal.pipeline("qwg_dflash_rope_kv_ctx")
        self.psoRopeQKProp = try hal.pipeline("qwg_dflash_rope_qk_prop")
        self.psoAttn = try hal.pipeline("qwg_dflash_attn")
        self.psoTop16Stage1 = try hal.pipeline("qwg_dflash_top16_stage1")
        self.psoTop16Stage2 = try hal.pipeline("qwg_dflash_top16_stage2")
        self.psoSelectorGreedy = try hal.pipeline("qwg_dflash_selector_greedy")
        self.psoGemvT4 = try hal.pipeline("qwg_gemv_q4g_dx_h16_t4")
        self.psoGemmBatched = try hal.pipeline("qwg_gemm_q4g_h16")
        self.dspinePSO = try DSpinePipelines(hal: hal)

        var downMap: [Int: any MTLComputePipelineState] = [:]
        var guMap: [Int: any MTLComputePipelineState] = [:]
        var downRowPairedMap: [Int: any MTLComputePipelineState] = [:]
        for m in 2...8 {
            downMap[m] = try hal.pipeline("qwm_mlp_down_m\(m)")
            guMap[m] = try hal.pipeline("qwm_mlp_gate_up_fwht32_m\(m)")
            if m >= QWMTPBatchedMLP.rowPairedMinimumRows {
                downRowPairedMap[m] = try hal.pipeline("qwm_mlp_down_rp_m\(m)")
            }
        }
        self.psoDownBatched = downMap
        self.psoDownRowPaired = downRowPairedMap
        self.psoGateUpFwht32Batched = guMap
        self.quantPSOs = try Self.loadQuantPipelines(hal: hal, fc: self.fc, layers: loadedLayers)
    }

    func encodeCaptureTargetHidden(_ enc: any MTLComputeCommandEncoder,
                                   source: QWTensorValue,
                                   srcRowOffset: Int,
                                   dstRowOffset: Int,
                                   rows: Int,
                                   slot: Int) {
        guard rows > 0, (0..<5).contains(slot) else { return }
        enc.setComputePipelineState(psoCaptureTarget)
        source.bind(enc, index: 0)
        targetHidden.bind(enc, index: 1)
        var sOff = UInt32(srcRowOffset), dOff = UInt32(dstRowOffset), slotU = UInt32(slot)
        enc.setBytes(&sOff, length: 4, index: 2)
        enc.setBytes(&dOff, length: 4, index: 3)
        enc.setBytes(&slotU, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    private func encodeQ4Matmul(_ enc: any MTLComputeCommandEncoder,
                                weight: QWWeightView,
                                input: QWTensorValue,
                                output: QWTensorValue,
                                rows: Int) {
        if !QWMTPBatchedMLP.usesQ4GMKernel(weight) {
            encodeAQMatmul(enc, weight: weight, input: input, output: output, rows: rows)
            return
        }
        let n = weight.n, k = weight.k
        if rows == 1 {
            enc.setComputePipelineState(psoGemvT4)
            weight.blocks.bind(enc, index: 0)
            input.bind(enc, index: 1)
            output.bind(enc, index: 3)
            var nU = UInt32(n), kU = UInt32(k)
            enc.setBytes(&nU, length: 4, index: 4)
            enc.setBytes(&kU, length: 4, index: 5)
            output.bind(enc, index: 8)
            enc.dispatchThreadgroups(MTLSize(width: (n + 3) / 4, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        } else if let pso = psoDownBatched[rows] {
            let rowPairedPSO = QWMTPBatchedMLP.usesRowPairs(weight, activeRows: rows) ? psoDownRowPaired[rows] : nil
            enc.setComputePipelineState(rowPairedPSO ?? pso)
            weight.blocks.bind(enc, index: 0)
            input.bind(enc, index: 1)
            output.bind(enc, index: 3)
            var nU = UInt32(n) | (weight.isIncumbentContainer ? 0 : 0x8000_0000), kU = UInt32(k)
            enc.setBytes(&nU, length: 4, index: 4)
            enc.setBytes(&kU, length: 4, index: 5)
            let tptg = rowPairedPSO == nil ? 128 : 64
            enc.dispatchThreadgroups(MTLSize(width: (n + 3) / 4, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: tptg, height: 1, depth: 1))
        } else if (9...16).contains(rows) {
            let c0 = rows / 2, c1 = rows - c0
            encodeQ4Matmul(enc, weight: weight, input: input.slice(byteOffset: 0, count: c0 * k),
                           output: output.slice(byteOffset: 0, count: c0 * n), rows: c0)
            encodeQ4Matmul(enc, weight: weight, input: input.slice(byteOffset: c0 * k * 2, count: c1 * k),
                           output: output.slice(byteOffset: c0 * n * 2, count: c1 * n), rows: c1)
        } else {
            enc.setComputePipelineState(psoGemmBatched)
            weight.blocks.bind(enc, index: 0)
            input.bind(enc, index: 1)
            output.bind(enc, index: 2)
            output.bind(enc, index: 3)
            var mU = UInt32(rows), nU = UInt32(n), kU = UInt32(k)
            enc.setBytes(&mU, length: 4, index: 4)
            enc.setBytes(&nU, length: 4, index: 5)
            enc.setBytes(&kU, length: 4, index: 6)
            var xs = UInt32(k), ys = UInt32(n)
            enc.setBytes(&xs, length: 4, index: 9)
            enc.setBytes(&ys, length: 4, index: 10)
            enc.dispatchThreadgroups(MTLSize(width: (n + 63) / 64, height: (rows + 63) / 64, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        }
    }

    func encodeRMSNorm(_ enc: any MTLComputeCommandEncoder,
                       x: QWTensorValue, gamma: QWTensorValue,
                       y: QWTensorValue, yRot: QWTensorValue,
                       rows: Int, mode: UInt32, epsOverride: Float? = nil) {
        enc.setComputePipelineState(psoRMSNorm)
        x.bind(enc, index: 0)
        gamma.bind(enc, index: 1)
        y.bind(enc, index: 2)
        yRot.bind(enc, index: 3)
        var eps = epsOverride ?? config.rmsNormEps, modeU = mode
        enc.setBytes(&eps, length: 4, index: 4)
        enc.setBytes(&modeU, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    func encodeF16GemvBatched(_ enc: any MTLComputeCommandEncoder,
                              w: QWTensorValue, x: QWTensorValue, y: QWTensorValue,
                              n: Int, k: Int, m: Int) {
        enc.setComputePipelineState(psoF16GemvBatched)
        w.bind(enc, index: 0)
        x.bind(enc, index: 1)
        y.bind(enc, index: 2)
        var nU = UInt32(n), kU = UInt32(k), mU = UInt32(m)
        enc.setBytes(&nU, length: 4, index: 3)
        enc.setBytes(&kU, length: 4, index: 4)
        enc.setBytes(&mU, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: (n + 3) / 4, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    private func encodeConvApply(_ enc: any MTLComputeCommandEncoder,
                                 x: QWTensorValue, dyn: QWTensorValue,
                                 baseKernel: QWTensorValue, outOrResidual: QWTensorValue,
                                 rows: Int, mode: UInt32) {
        enc.setComputePipelineState(psoConvApply)
        x.bind(enc, index: 0)
        dyn.bind(enc, index: 1)
        baseKernel.bind(enc, index: 2)
        outOrResidual.bind(enc, index: 3)
        var modeU = mode
        enc.setBytes(&modeU, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    func encodeContextUpdate(_ cb: any MTLCommandBuffer,
                             slot: QWDFlashSessionSlot,
                             startPosition: Int,
                             totalCount: Int) throws {
        let captureRows = min(totalCount, config.maxCacheSize)
        guard captureRows > 0 else { return }
        let posOffset = startPosition + (totalCount - captureRows)
        guard let enc = cb.makeComputeCommandEncoder() else {
            throw QWEngineDriverError.state("QWDFlashDrafter: failed to create context encoder")
        }
        defer { enc.endEncoding() }

        encodeQ4Matmul(enc, weight: fc, input: targetHidden, output: hCtx, rows: captureRows)
        enc.memoryBarrier(scope: .buffers)
        encodeRMSNorm(enc, x: hCtx, gamma: hiddenNorm, y: hCtx, yRot: hCtx, rows: captureRows, mode: 1)
        enc.memoryBarrier(scope: .buffers)

        var cOff = UInt32(posOffset), eps = config.rmsNormEps, maxCacheU = UInt32(config.maxCacheSize)
        for l in 0..<config.numLayers {
            let lw = layers[l]
            encodeQ4Matmul(enc, weight: lw.kvProj, input: hCtx, output: kvCtx, rows: captureRows)
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(psoRopeKVCtx)
            kvCtx.bind(enc, index: 0)
            lw.kNorm.bind(enc, index: 1)
            invFreq.bind(enc, index: 2)
            slot.kLayers[l].bind(enc, index: 3)
            slot.vLayers[l].bind(enc, index: 4)
            enc.setBytes(&cOff, length: 4, index: 5)
            enc.setBytes(&eps, length: 4, index: 6)
            enc.setBytes(&maxCacheU, length: 4, index: 7)
            enc.dispatchThreadgroups(MTLSize(width: captureRows, height: config.numKeyValueHeads, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)
        }
    }

    func propose(hal: QWHal,
                 slot: QWDFlashSessionSlot,
                 cacheOffset: Int,
                 pendingToken: Int,
                 draftCount: Int,
                 head: QWHead,
                 headStorage: QWHeadStorage,
                 draftHead: QWMTPTDraftHead?,
                 embedding: (Int, QWTensorValue) throws -> Void,
                 stochasticOwner: QWStochasticSession? = nil,
                 completed: ((any MTLCommandBuffer) throws -> Void)? = nil) throws -> (drafts: [Int], proposals: [QWMTPGumbelProposal]) {
        let D = draftCount
        let M = D + 1
        let maxDrafts = max(7, config.blockSize - 1)
        guard (1...maxDrafts).contains(D) else {
            throw QWEngineDriverError.state("QWDFlashDrafter: invalid draftCount \(D) (expected 1...\(maxDrafts))")
        }
        let H = config.hiddenSize
        let V = config.vocabSize
        let R = config.selectorRank
        let topK = config.selectorTopK
        let hasConv = config.convKernelSize > 0
        let convProjDim = hasConv ? (2 * config.convKernelSize * (H / config.convGroupSize)) : 0

        let row0 = QWTensorValue(region: propNorm.region, offset: propNorm.offset, shape: QWTensorShape([H], .f16))
        let row1 = QWTensorValue(region: propNorm.region, offset: propNorm.offset + H * 2, shape: QWTensorShape([H], .f16))
        try embedding(pendingToken, row0)
        try embedding(config.maskTokenId, row1)
        let src16 = propNorm.region.buffer.contents()
            .advanced(by: propNorm.region.base + propNorm.offset)
            .assumingMemoryBound(to: Float16.self)
        let dst32 = propX.region.buffer.contents()
            .advanced(by: propX.region.base + propX.offset)
            .assumingMemoryBound(to: Float.self)
        for i in 0..<H {
            dst32[i] = Float(src16[i])
            dst32[H + i] = Float(src16[H + i])
        }
        if M > 2 {
            let maskRow32 = UnsafeRawPointer(dst32.advanced(by: H))
            for m in 2..<M {
                memcpy(UnsafeMutableRawPointer(dst32.advanced(by: m * H)), maskRow32, H * 4)
            }
        }

        guard let (cb, enc) = hal.begin(.serial) else {
            throw QWEngineDriverError.state("QWDFlashDrafter: command buffer creation failed")
        }

        if config.isDSpine {
            encodeRMSNorm(enc, x: propX, gamma: layers[0].ln1, y: propNorm, yRot: propRot, rows: M, mode: 2)
        }

        var maxCacheU = UInt32(config.maxCacheSize)
        var isCausalU = UInt32(config.draftBlockCausal ? 1 : 0)

        for l in 0..<config.numLayers {
            let lw = layers[l]
            if !config.isDSpine, let acProj = lw.attnConvProj, let acBase = lw.attnConvBase, let pDyn = propDyn {
                encodeRMSNorm(enc, x: propX, gamma: lw.ln1, y: propNorm, yRot: propNorm, rows: M, mode: 0)
                encodeF16GemvBatched(enc, w: acProj, x: propNorm, y: pDyn, n: convProjDim, k: H, m: M)
                encodeConvApply(enc, x: propNorm, dyn: pDyn, baseKernel: acBase, outOrResidual: propRot, rows: M, mode: 0)
            }
            encodeQ4Matmul(enc, weight: lw.qkvProj, input: propRot, output: qkvProp, rows: M)

            enc.setComputePipelineState(psoRopeQKProp)
            qkvProp.bind(enc, index: 0)
            lw.qNorm.bind(enc, index: 1)
            lw.kNorm.bind(enc, index: 2)
            invFreq.bind(enc, index: 3)
            var pOff = UInt32(cacheOffset), eps = config.rmsNormEps
            enc.setBytes(&pOff, length: 4, index: 4)
            enc.setBytes(&eps, length: 4, index: 5)
            enc.dispatchThreadgroups(MTLSize(width: M, height: config.numAttentionHeads + config.numKeyValueHeads, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

            enc.setComputePipelineState(psoAttn)
            qkvProp.bind(enc, index: 0)
            slot.kLayers[l].bind(enc, index: 1)
            slot.vLayers[l].bind(enc, index: 2)
            attnOut.bind(enc, index: 3)
            var nOff = UInt32(cacheOffset), mU = UInt32(M)
            enc.setBytes(&nOff, length: 4, index: 4)
            enc.setBytes(&mU, length: 4, index: 5)
            enc.setBytes(&maxCacheU, length: 4, index: 6)
            enc.setBytes(&isCausalU, length: 4, index: 7)
            enc.dispatchThreadgroups(MTLSize(width: config.numAttentionHeads, height: M, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

            if config.isDSpine {
                encodeQ4Matmul(enc, weight: lw.oProj, input: attnOut, output: propDown, rows: M)
                encodeRMSNorm(enc, x: propX, gamma: lw.ln2, y: propDown, yRot: propRot, rows: M, mode: 3)
            } else if let acBase = lw.attnConvBase, let mcProj = lw.mlpConvProj,
                      let mcBase = lw.mlpConvBase, let pDyn = propDyn {
                encodeQ4Matmul(enc, weight: lw.oProj, input: attnOut, output: propNorm, rows: M)
                encodeConvApply(enc, x: propNorm, dyn: pDyn, baseKernel: acBase, outOrResidual: propX, rows: M, mode: 1)
                encodeRMSNorm(enc, x: propX, gamma: lw.ln2, y: propNorm, yRot: propNorm, rows: M, mode: 0)
                encodeF16GemvBatched(enc, w: mcProj, x: propNorm, y: pDyn, n: convProjDim, k: H, m: M)
                encodeConvApply(enc, x: propNorm, dyn: pDyn, baseKernel: mcBase, outOrResidual: propRot, rows: M, mode: 0)
            }

            encodeGateUpMatmul(enc, weight: lw.gateUpProj, input: propRot, output: mlpInter, rows: M)
            encodeQ4Matmul(enc, weight: lw.downProj, input: mlpInter, output: propDown, rows: M)
            if let bw = dspineBridge, let bs = dspineScratch {
                encodeDSpineBridgeLayer(enc, layer: l, anchorToken: pendingToken, rows: M, bw: bw, bs: bs)
            } else if let mcBase = lw.mlpConvBase, let pDyn = propDyn {
                encodeConvApply(enc, x: propDown, dyn: pDyn, baseKernel: mcBase, outOrResidual: propX, rows: M, mode: 1)
            }
        }

        let propXDraftSlice = dspineScratch?.propXPost4
            ?? QWTensorValue(region: propX.region, offset: propX.offset + H * 4, shape: QWTensorShape([D * H], .f32))
        let propNormSlice = QWTensorValue(region: propNorm.region, offset: propNorm.offset,
                                          shape: QWTensorShape([D * H], .f16))
        let propRotSlice = QWTensorValue(region: propRot.region, offset: propRot.offset,
                                         shape: QWTensorShape([D * H], .f16))
        let logitsSlice = QWTensorValue(region: draftLogits.region, offset: draftLogits.offset,
                                        shape: QWTensorShape([D * V], .f32))

        encodeRMSNorm(enc, x: propXDraftSlice, gamma: finalNorm, y: propNormSlice, yRot: propRotSlice, rows: D, mode: 2)
        if !config.isDSpine, let selProj = selectorHiddenProj, let hProjOut = hiddenProjOut {
            encodeF16GemvBatched(enc, w: selProj, x: propNormSlice, y: hProjOut, n: R, k: H, m: D)
        }

        if let draftHead {
            if D == 1 {
                try draftHead.encode(enc, hal: hal, x: propRotSlice, y: logitsSlice)
            } else {
                try draftHead.encodeBatched(enc, hal: hal, x: propRotSlice, y: logitsSlice, m: D)
            }
        } else {
            switch headStorage {
            case .q4g:
                try head.encode(enc, storage: headStorage, x: propRotSlice, y: logitsSlice, M: D)
            case .f16:
                try head.encode(enc, storage: headStorage, x: propNormSlice, y: logitsSlice, M: D)
            }
        }

        var vocabU = UInt32(V), topKU = UInt32(topK)
        enc.setComputePipelineState(psoTop16Stage1)
        logitsSlice.bind(enc, index: 0)
        top16PartialsVal.bind(enc, index: 1)
        top16PartialsId.bind(enc, index: 2)
        enc.setBytes(&vocabU, length: 4, index: 3)
        enc.setBytes(&topKU, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: 256, height: D, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        enc.setComputePipelineState(psoTop16Stage2)
        top16PartialsVal.bind(enc, index: 0)
        top16PartialsId.bind(enc, index: 1)
        unaryOut.bind(enc, index: 2)
        candidatesOut.bind(enc, index: 3)
        enc.setBytes(&topKU, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: D, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        if let bw = dspineBridge, let bs = dspineScratch {
            encodeDSpineRefinementTail(
                enc, draftCount: D, topK: topK, headStorage: headStorage,
                bw: bw, bs: bs, stochastic: stochasticOwner != nil
            )
        } else if stochasticOwner == nil,
                  let hProjOut = hiddenProjOut,
                  let pBook = predecessorCodebook,
                  let sBook = successorCodebook {
            var anchorI = Int32(pendingToken), draftCountU = UInt32(D)
            enc.setComputePipelineState(psoSelectorGreedy)
            hProjOut.bind(enc, index: 0)
            unaryOut.bind(enc, index: 1)
            candidatesOut.bind(enc, index: 2)
            pBook.bind(enc, index: 3)
            sBook.bind(enc, index: 4)
            draftTokensOut.bind(enc, index: 5)
            enc.setBytes(&anchorI, length: 4, index: 6)
            enc.setBytes(&draftCountU, length: 4, index: 7)
            enc.setBytes(&topKU, length: 4, index: 8)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }

        hal.end(cb, enc)
        cb.waitUntilCompleted()
        guard cb.status == .completed, cb.error == nil else {
            throw QWEngineDriverError.state("QWDFlashDrafter GPU failure: \(String(describing: cb.error))")
        }
        try completed?(cb)

        if let owner = stochasticOwner {
            if let bs = dspineScratch {
                return try selectDSpineStochastic(owner: owner, draftCount: D, topK: topK, bs: bs)
            }
            return try selectStochastic(owner: owner, anchorToken: pendingToken, draftCount: D, topK: topK)
        } else {
            let ptr = draftTokensOut.region.buffer.contents()
                .advanced(by: draftTokensOut.region.base + draftTokensOut.offset)
                .assumingMemoryBound(to: Int32.self)
            let drafts = (0..<D).map { Int(ptr[$0]) }
            return (drafts, [])
        }
    }

    private func selectStochastic(owner: QWStochasticSession,
                                  anchorToken: Int,
                                  draftCount: Int,
                                  topK: Int) throws -> (drafts: [Int], proposals: [QWMTPGumbelProposal]) {
        guard let hiddenProjOut, let predecessorCodebook, let successorCodebook else {
            throw QWEngineDriverError.state("QWDFlashDrafter: missing CandidateSelector buffers")
        }
        let R = config.selectorRank
        let V = config.vocabSize
        let temp = max(Double(owner.config.temperature), 1e-6)
        let hPtr = hiddenProjOut.region.buffer.contents()
            .advanced(by: hiddenProjOut.region.base + hiddenProjOut.offset)
            .assumingMemoryBound(to: Float.self)
        let uPtr = unaryOut.region.buffer.contents()
            .advanced(by: unaryOut.region.base + unaryOut.offset)
            .assumingMemoryBound(to: Float.self)
        let cPtr = candidatesOut.region.buffer.contents()
            .advanced(by: candidatesOut.region.base + candidatesOut.offset)
            .assumingMemoryBound(to: Int32.self)
        let pBook = predecessorCodebook.region.buffer.contents()
            .advanced(by: predecessorCodebook.region.base + predecessorCodebook.offset)
            .assumingMemoryBound(to: UInt16.self)
        let sBook = successorCodebook.region.buffer.contents()
            .advanced(by: successorCodebook.region.base + successorCodebook.offset)
            .assumingMemoryBound(to: UInt16.self)

        func bf16(_ bits: UInt16) -> Float {
            Float(bitPattern: UInt32(bits) << 16)
        }

        var pred = anchorToken
        var drafts: [Int] = []
        var proposals: [QWMTPGumbelProposal] = []
        for pos in 0..<draftCount {
            var scores = [Double](repeating: 0, count: topK)
            var ids = [Int](repeating: 0, count: topK)
            let hBase = pos * R
            let pBase = pred * R
            var maxScore = -Double.infinity
            for k in 0..<topK {
                let cand = Int(cPtr[pos * 16 + k])
                ids[k] = cand
                let sBase = cand * R
                var edge: Float = 0
                for r in 0..<R {
                    edge += bf16(pBook[pBase + r]) * hPtr[hBase + r] * bf16(sBook[sBase + r])
                }
                let s = Double(uPtr[pos * 16 + k] + edge)
                scores[k] = s
                if s > maxScore { maxScore = s }
            }
            let weights = scores.map { exp(($0 - maxScore) / temp) }
            let dist = try QWSamplingDistribution.sparse(vocabulary: V, ids: ids, weights: weights)
            let proposal = try owner.stream.sampleGumbelProposal(dist)
            pred = proposal.token
            drafts.append(proposal.token)
            proposals.append(proposal)
        }
        return (drafts, proposals)
    }
}
