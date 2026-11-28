// QWDSparkDrafter.swift — DSpark target-tap ingestion, SWA-128 draft layers, Markov chain, and confidence scheduler.
import Foundation
import Metal
import QWAlloc
import QWAttn
import QWCore
import QWGEMM
import QWHead
import QWHAL
import QWMTP

/// DSpark runtime executor owning SWA-128 context rings, target tap buffer, and draft block/Markov kernels.
final class QWDSparkDrafter: @unchecked Sendable {
    struct PrefillCaptureSlice: Sendable {
        let ctxStartPos: Int
        let localStart: Int
        let captureRows: Int
    }

    final class SessionSlot: @unchecked Sendable {
        let index: Int
        let ringBuffer: any MTLBuffer
        let swaKV: [QWTensorValue]
        let ringBytes: Int

        init(index: Int, ringBuffer: any MTLBuffer, swaKV: [QWTensorValue], ringBytes: Int) {
            self.index = index
            self.ringBuffer = ringBuffer
            self.swaKV = swaKV
            self.ringBytes = ringBytes
        }

        func snapshotData() -> Data {
            Data(bytes: ringBuffer.contents(), count: ringBytes)
        }

        func restoreSnapshot(_ data: Data) -> Bool {
            guard data.count == ringBytes else { return false }
            let ringDst = ringBuffer.contents()
            withExtendedLifetime(ringBuffer) {
                data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                    guard let base = raw.baseAddress else { return }
                    ringDst.copyMemory(from: base, byteCount: ringBytes)
                }
            }
            return true
        }
    }

    struct DSparkCtxKVParams {
        var T: UInt32
        var dim: UInt32
        var nopeDim: UInt32
        var ropeDim: UInt32
        var pos0: UInt32
        var ringSize: UInt32
        var flags: UInt32
        var rmsEps: Float
    }

    struct DSparkBlockAttnParams {
        var B: UInt32
        var qHeads: UInt32
        var headDim: UInt32
        var qUpStride: UInt32
        var pos0: UInt32
        var ringSize: UInt32
        var slidingWindow: UInt32
        var hasSink: UInt32
        var swaMaskMode: UInt32
    }

    struct DSparkMarkovParams {
        var stepIdx: UInt32
        var V: UInt32
        var rank: UInt32
        var flags: UInt32
        var partCount: UInt32
    }

    struct DSparkConfParams {
        var stepIdx: UInt32
        var H: UInt32
        var R: UInt32
        var hiddenRowIdx: UInt32
        var V: UInt32
        var flags: UInt32
    }

    struct DSparkQPrepParams {
        var T: UInt32
        var pos0: UInt32
        var qHeads: UInt32
        var headDim: UInt32
        var nopeDim: UInt32
        var ropeDim: UInt32
        var qUpStride: UInt32
        var indexDim: UInt32
        var rmsEps: Float
        var mscale: Float
    }

    struct DSparkVhaInvRopeParams {
        var T: UInt32
        var pos0: UInt32
        var qHeads: UInt32
        var headDim: UInt32
        var nopeDim: UInt32
        var ropeDim: UInt32
        var invMscale: Float
        var hasVHA: UInt32
    }

    struct DSparkGroupedWoAParams {
        var T: UInt32
        var oGroups: UInt32
        var oLoraRank: UInt32
        var groupIn: UInt32
        var qUpStride: UInt32
        var oGateOffset: UInt32
        var bits: UInt32
        var group: UInt32
        var isIncumbent: UInt32
    }

    let weights: QWDSparkWeights
    let config: QWModelDescriptor.DSparkConfig
    let hal: QWHal
    let targetGeometry: QWModelGeometry
    let draftGeometry: QWModelGeometry
    let targetLayerSlotMap: [Int]
    let sessionSlots: [SessionSlot]
    let maxCaptureRows: Int
    let schedulerEnabled: Bool
    var confidenceScheduler: QWDSparkConfidenceScheduler
    private(set) var lastTailConfidence: Float?
    private var pendingProfileStages: (blockCB: any MTLCommandBuffer, headCB: any MTLCommandBuffer, pos0: Int)?
    static let stageProfileEnabled = ProcessInfo.processInfo.environment["QW_DSPARK_PROFILE"] == "1"

    let tapBuf: QWTensorValue
    let scratchBuffer: any MTLBuffer
    let promptToksBuf: any MTLBuffer
    let tokSeqBuf: any MTLBuffer
    let mainProjRaw: QWTensorValue
    let mainNormRot: QWTensorValue
    let ctxKVRaw: QWTensorValue

    let blkX: QWTensorValue
    let blkLn1: QWTensorValue
    let blkLn2: QWTensorValue
    let blkO16: QWTensorValue
    let blkDn16: QWTensorValue
    let blkDn32: QWTensorValue
    let blkMhcStream: QWTensorValue?
    let blkMhcControls: QWTensorValue?
    let blkHeadHidden: QWTensorValue
    let blkHeadNorm: QWTensorValue

    let blkQDownRaw: QWTensorValue
    let blkQNormRot: QWTensorValue
    let blkKVRaw: QWTensorValue
    let blkKV: QWTensorValue
    let blkQUpGate: QWTensorValue
    let blkAttOut: QWTensorValue
    let blkZVha: QWTensorValue
    let blkZGated: QWTensorValue

    let blkMoe: QWEngineDriver.MoEScratch?
    let blkDenseAct16: QWTensorValue?
    let baseLogits: QWTensorValue
    let stepLogits: QWTensorValue
    let confOut: QWTensorValue
    let headGS: QWTensorValue
    let keysHi: QWTensorValue
    let keysLo: QWTensorValue
    let invFreqBuf: QWTensorValue
    let zeroSinkBuf: QWTensorValue
    let zeroVhaBuf: QWTensorValue
    let argmaxPartCount: Int

    let psoTapMean: any MTLComputePipelineState
    let psoCtxKVWrite: any MTLComputePipelineState
    let psoBlockAttn: any MTLComputePipelineState
    let psoMarkovArgmax: any MTLComputePipelineState
    let psoMarkovLogits: any MTLComputePipelineState
    let psoConfDot: any MTLComputePipelineState
    let psoArgmaxFin: any MTLComputePipelineState
    let psoQPrep: any MTLComputePipelineState
    let psoVhaInvRope: any MTLComputePipelineState
    let psoGroupedWoAGemv: any MTLComputePipelineState
    let psoGroupedWoAGemvMR: any MTLComputePipelineState
    let psoGroupedWoAGemvSub2: any MTLComputePipelineState
    let psoGroupedWoAGemvMRSub2: any MTLComputePipelineState

    var activeBuffers: [any MTLBuffer] {
        [scratchBuffer, promptToksBuf, tokSeqBuf] + sessionSlots.map(\.ringBuffer)
    }

    init(
        geometry: QWModelGeometry,
        hal: QWHal,
        config: QWModelDescriptor.DSparkConfig,
        packURL: URL,
        draftHeadURL: URL?,
        maxPrefillChunk: Int,
        sessionSlotCount: Int,
        confidenceScheduleEnabled: Bool
    ) throws {
        self.targetGeometry = geometry
        self.draftGeometry = geometry
        self.hal = hal
        self.config = config
        self.weights = try QWDSparkWeights(
            geometry: geometry, hal: hal, config: config,
            packURL: packURL, draftHeadURL: draftHeadURL
        )

        var map = Array(repeating: -1, count: geometry.numLayers)
        for (slot, targetIdx) in config.targetZeroBasedLayers.enumerated() where (0..<geometry.numLayers).contains(targetIdx) {
            map[targetIdx] = slot
        }
        self.targetLayerSlotMap = map

        let B = config.blockSize
        let H = weights.hidden
        let V = weights.vocab
        let aGeo = weights.attnGeometry
        let capRows = max(max(B, min(16, config.slidingWindow)), min(maxPrefillChunk, config.slidingWindow))
        self.maxCaptureRows = capRows
        self.argmaxPartCount = max(1, min(1024, (V + 255) / 256))
        self.schedulerEnabled = confidenceScheduleEnabled && (weights.confidenceHead != nil)
        self.confidenceScheduler = QWDSparkConfidenceScheduler(
            maxDrafts: min(15, config.maxDrafts),
            numExperts: geometry.numExperts,
            expertsPerTok: geometry.expertsPerTok
        )

        self.psoTapMean = try hal.pipeline("qwg_dspark_mhc_tap_mean")
        self.psoCtxKVWrite = try hal.pipeline("qwg_dspark_swa_ctx_kv_write")
        self.psoBlockAttn = try hal.pipeline("qwg_dspark_block_hca_attn")
        self.psoMarkovArgmax = try hal.pipeline("qwg_dspark_markov_aq8_argmax")
        self.psoMarkovLogits = try hal.pipeline("qwg_dspark_markov_aq8_logits")
        self.psoConfDot = try hal.pipeline("qwg_dspark_confidence_dot")
        self.psoArgmaxFin = try hal.pipeline("qwg_head_argmax_fin")
        self.psoQPrep = try hal.pipeline("qwg_hca_q_prep")
        self.psoVhaInvRope = try hal.pipeline("qwg_hca_vha_invrope")
        self.psoGroupedWoAGemv = try hal.pipeline("qwg_hca_grouped_wo_a_gemv")
        self.psoGroupedWoAGemvMR = try hal.pipeline("qwg_hca_grouped_wo_a_gemv_mr")
        self.psoGroupedWoAGemvSub2 = try hal.pipeline("qwg_hca_grouped_wo_a_gemv_sub2")
        self.psoGroupedWoAGemvMRSub2 = try hal.pipeline("qwg_hca_grouped_wo_a_gemv_mr_sub2")

        let ringLayerBytes = ((config.slidingWindow * aGeo.headDim * 2) + 255) & ~255
        let ringTotalBytes = config.numHiddenLayers * ringLayerBytes
        var slots: [SessionSlot] = []
        slots.reserveCapacity(max(1, sessionSlotCount))
        for idx in 0..<max(1, sessionSlotCount) {
            guard let buf = hal.device.makeBuffer(length: max(256, ringTotalBytes), options: .storageModeShared) else {
                throw QWEngineDriverError.state("DSpark SWA ring allocation failed")
            }
            memset(buf.contents(), 0, ringTotalBytes)
            var kvLayers: [QWTensorValue] = []
            kvLayers.reserveCapacity(config.numHiddenLayers)
            for l in 0..<config.numHiddenLayers {
                kvLayers.append(QWTensorValue(
                    region: QWMRegion(buffer: buf, base: l * ringLayerBytes, length: ringLayerBytes),
                    offset: 0,
                    shape: QWTensorShape([config.slidingWindow, aGeo.headDim], .f16)
                ))
            }
            slots.append(SessionSlot(index: idx, ringBuffer: buf, swaKV: kvLayers, ringBytes: ringTotalBytes))
        }
        self.sessionSlots = slots

        guard let pTokBuf = hal.device.makeBuffer(length: max(256, B * 4), options: .storageModeShared),
              let tSeqBuf = hal.device.makeBuffer(length: max(256, (B + 1) * 4), options: .storageModeShared) else {
            throw QWEngineDriverError.state("DSpark token journal allocation failed")
        }
        self.promptToksBuf = pTokBuf
        self.tokSeqBuf = tSeqBuf

        let C = geometry.mhcConfig?.streams ?? 0
        let ctrlCols = C + C * C
        let ropePairs = max(1, aGeo.ropeDim / 2)
        let moeBytes = weights.moeExperts > 0
            ? QWEngineDriver.moeScratchAllocationBytes(
                rows: B, geometry: geometry, useGroupedNAX: hal.caps.supportsSimdgroupMMA)
            : 0

        var cursor = 0
        func align256(_ x: Int) -> Int { (x + 255) & ~255 }
        func reserve(_ count: Int, _ dt: QWDType) -> (off: Int, bytes: Int, shape: QWTensorShape) {
            let elemB: Int = (dt == .f16 || dt == .bf16) ? 2 : 4
            let b = max(256, count * elemB)
            let off = cursor
            cursor = align256(cursor + b)
            return (off, b, QWTensorShape([count], dt))
        }

        let tapSpec = reserve(capRows * weights.numTaps * H, .f16)
        let mpSpec = reserve(capRows * H, .f16)
        let mnSpec = reserve(capRows * H, .f16)
        let ckSpec = reserve(capRows * aGeo.headDim, .f16)
        let xSpec = reserve(B * H, .f16)
        let ln1Spec = reserve(B * H, .f16)
        let ln2Spec = reserve(B * H, .f16)
        let o16Spec = reserve(B * H, .f16)
        let dn16Spec = reserve(B * H, .f16)
        let dn32Spec = reserve(B * H, .f32)
        let mhcSSpec = C > 0 ? reserve(B * C * H, .f32) : nil
        let mhcCSpec = C > 0 ? reserve(B * ctrlCols, .f32) : nil
        let hhSpec = reserve(B * H, .f16)
        let hnSpec = reserve(B * H, .f16)
        let qdSpec = reserve(B * aGeo.qLoraRank, .f16)
        let qnSpec = reserve(B * aGeo.qLoraRank, .f16)
        let kvrSpec = reserve(B * aGeo.headDim, .f16)
        let kvSpec = reserve(B * aGeo.headDim, .f16)
        let quSpec = reserve(B * aGeo.qUpStride, .f16)
        let aoSpec = reserve(B * aGeo.qHeads * aGeo.headDim, .f16)
        let zvSpec = reserve(B * aGeo.qHeads * aGeo.headDim, .f16)
        let zgSpec = reserve(B * aGeo.zOutStride, .f16)
        let denseSpec = (weights.moeExperts == 0) ? reserve(B * max(1, weights.moeIntermediate), .f16) : nil
        let moeOff = cursor
        if moeBytes > 0 { cursor = align256(cursor + moeBytes) }
        let blSpec = reserve(B * V, .f32)
        let slSpec = reserve(V, .f32)
        let cfSpec = reserve(B, .f32)
        let gsSpec = reserve(B * max(1, H / 128), .f32)
        let khSpec = reserve(argmaxPartCount, .i32)
        let klSpec = reserve(argmaxPartCount, .i32)
        let invSpec = reserve(ropePairs, .f32)
        let sinkSpec = reserve(aGeo.qHeads, .f32)
        let vhaSpec = reserve(aGeo.qHeads * aGeo.qHeads, .f16)

        guard let sBuf = hal.device.makeBuffer(length: max(256, cursor), options: .storageModeShared) else {
            throw QWEngineDriverError.state("DSpark scratch buffer allocation failed (\(cursor) B)")
        }
        self.scratchBuffer = sBuf
        func bind(_ sp: (off: Int, bytes: Int, shape: QWTensorShape)) -> QWTensorValue {
            QWTensorValue(region: QWMRegion(buffer: sBuf, base: sp.off, length: sp.bytes), offset: 0, shape: sp.shape)
        }

        self.tapBuf = bind(tapSpec)
        self.mainProjRaw = bind(mpSpec)
        self.mainNormRot = bind(mnSpec)
        self.ctxKVRaw = bind(ckSpec)
        self.blkX = bind(xSpec)
        self.blkLn1 = bind(ln1Spec)
        self.blkLn2 = bind(ln2Spec)
        self.blkO16 = bind(o16Spec)
        self.blkDn16 = bind(dn16Spec)
        self.blkDn32 = bind(dn32Spec)
        self.blkMhcStream = mhcSSpec.map(bind)
        self.blkMhcControls = mhcCSpec.map(bind)
        self.blkHeadHidden = bind(hhSpec)
        self.blkHeadNorm = bind(hnSpec)
        self.blkQDownRaw = bind(qdSpec)
        self.blkQNormRot = bind(qnSpec)
        self.blkKVRaw = bind(kvrSpec)
        self.blkKV = bind(kvSpec)
        self.blkQUpGate = bind(quSpec)
        self.blkAttOut = bind(aoSpec)
        self.blkZVha = bind(zvSpec)
        self.blkZGated = bind(zgSpec)
        self.blkDenseAct16 = denseSpec.map(bind)
        self.baseLogits = bind(blSpec)
        self.stepLogits = bind(slSpec)
        self.confOut = bind(cfSpec)
        self.headGS = bind(gsSpec)
        self.keysHi = bind(khSpec)
        self.keysLo = bind(klSpec)
        self.invFreqBuf = bind(invSpec)
        self.zeroSinkBuf = bind(sinkSpec)
        self.zeroVhaBuf = bind(vhaSpec)

        if moeBytes > 0 {
            var mCur = moeOff
            let mLimit = moeOff + moeBytes
            self.blkMoe = try QWEngineDriver.makeMoEScratch({ elems, dt in
                let eb: Int = (dt == .f16 || dt == .bf16) ? 2 : 4
                let b = max(256, elems * eb)
                let off = mCur
                mCur = align256(mCur + b)
                guard mCur <= mLimit else {
                    throw QWEngineDriverError.state("DSpark MoE scratch overflow")
                }
                return QWTensorValue(
                    region: QWMRegion(buffer: sBuf, base: off, length: b),
                    offset: 0,
                    shape: QWTensorShape([elems], dt)
                )
            }, rows: B, geometry: geometry, useGroupedNAX: hal.caps.supportsSimdgroupMMA)
        } else {
            self.blkMoe = nil
        }

        let (invFreqVals, _) = QWModelDescriptor.computeRopeInvFreq(
            rotaryDim: aGeo.ropeDim, base: config.ropeTheta, yarn: nil
        )
        let invDst = sBuf.contents().advanced(by: invSpec.off).assumingMemoryBound(to: Float.self)
        for i in 0..<min(ropePairs, invFreqVals.count) { invDst[i] = invFreqVals[i] }
        memset(sBuf.contents().advanced(by: sinkSpec.off), 0, sinkSpec.bytes)
        memset(sBuf.contents().advanced(by: vhaSpec.off), 0, vhaSpec.bytes)

        hal.makeResident([sBuf, pTokBuf, tSeqBuf] + slots.map(\.ringBuffer))
        try Self.prewarmDSparkPipelines(hal: hal, weights: weights)
    }

    private static func prewarmDSparkPipelines(hal: QWHal, weights: QWDSparkWeights) throws {
        var seen = Set<String>()
        func warm(_ name: String?) throws {
            guard let name, seen.insert(name).inserted else { return }
            _ = try hal.pipeline(name)
        }
        try warm("qwg_fwht32_f16")
        try warm("qwg_multimax_f32")
        try warm("qwg_gemv_q4g_dx_h16_t4")
        let useNAX = hal.caps.isMetal4TensorCapable
        let canSG = hal.caps.supportsSimdgroupMMA
        if canSG {
            try warm(useNAX ? "qwg_gemm_q4g_h16" : "qwg_gemm_q4g_sg")
            try warm(useNAX ? "qwg_gemm_q4g_h16_swiglu" : "qwg_gemm_q4g_sg_swiglu")
        }
        if case .quantized(let dhView) = weights.draftHead, !dhView.isIncumbentContainer {
            try warm(QWGemmAQ.kernelName(bits: dhView.bits, group: dhView.groupSize, isSHVQ: dhView.isSHVQ))
        }
        for lw in weights.layers {
            for w in [weights.mainProj, lw.qDownProj, lw.kvProj, lw.qUpGateProj, lw.oProj] {
                if !w.isIncumbentContainer {
                    if canSG {
                        try warm(QWGemmAQ.gemmKernelName(bits: w.bits, group: w.groupSize, isSHVQ: w.isSHVQ, useNAX: useNAX))
                    }
                    try warm(QWGemmAQ.decodeKernelName(bits: w.bits, group: w.groupSize, isSHVQ: w.isSHVQ))
                }
            }
            if let moe = lw.moe {
                for w in [moe.sharedGateUp, moe.sharedDown, moe.latent?.fc1, moe.latent?.fc2].compactMap({ $0 }) {
                    if !w.isIncumbentContainer {
                        if canSG {
                            try warm(QWGemmAQ.gemmKernelName(bits: w.bits, group: w.groupSize, isSHVQ: w.isSHVQ, useNAX: useNAX))
                        }
                        try warm(QWGemmAQ.decodeKernelName(bits: w.bits, group: w.groupSize, isSHVQ: w.isSHVQ))
                    }
                }
            }
            if let gu = lw.denseGateUp, !gu.isIncumbentContainer, canSG {
                try warm(QWGemmAQ.swigluGemmKernelName(bits: gu.bits, group: gu.groupSize, isSHVQ: gu.isSHVQ, useNAX: useNAX))
            }
            if let dn = lw.denseDown, !dn.isIncumbentContainer {
                if canSG {
                    try warm(QWGemmAQ.gemmKernelName(bits: dn.bits, group: dn.groupSize, isSHVQ: dn.isSHVQ, useNAX: useNAX))
                }
                try warm(QWGemmAQ.decodeKernelName(bits: dn.bits, group: dn.groupSize, isSHVQ: dn.isSHVQ))
            }
        }
    }

    /// Overlap of the prior commit CB is safe only for 1-CB async greedy DSpark (`count <= maxDrafts`).
    static func allowsPriorCommitOverlap(schedulerEnabled: Bool, verifyRows: Int, maxDrafts: Int) -> Bool {
        !schedulerEnabled && (verifyRows - 1) <= maxDrafts
    }

    /// Computes trailing slice within `slidingWindow` (`[max(0, windowEnd - W) ..< pos + T)`).
    static func prefillCaptureSlice(
        startPos: Int,
        rows: Int,
        windowEnd: Int,
        slidingWindow: Int
    ) -> PrefillCaptureSlice? {
        guard rows > 0, slidingWindow > 0 else { return nil }
        let winStart = max(0, windowEnd - slidingWindow)
        let capStart = max(startPos, winStart)
        let capEnd = min(startPos + rows, windowEnd)
        guard capStart < capEnd else { return nil }
        return PrefillCaptureSlice(
            ctxStartPos: capStart,
            localStart: capStart - startPos,
            captureRows: capEnd - capStart
        )
    }

    func useResources(on enc: any MTLComputeCommandEncoder) {
        for buf in weights.allBuffers { enc.useResource(buf, usage: .read) }
        enc.useResource(scratchBuffer, usage: [.read, .write])
        enc.useResource(promptToksBuf, usage: .read)
        enc.useResource(tokSeqBuf, usage: [.read, .write])
        for s in sessionSlots { enc.useResource(s.ringBuffer, usage: [.read, .write]) }
    }

    func encodeCaptureTargetTap(
        _ enc: any MTLComputeCommandEncoder,
        base: QWTensorValue,
        srcRowOffset: Int,
        dstRowOffset: Int,
        rows: Int,
        slot: Int,
        hc: Int,
        deferredInject: (src: QWTensorValue, inj: QWTensorValue, isMHC: Bool)? = nil,
        singleStream: Bool = false
    ) {
        guard rows > 0, slot >= 0, slot < weights.numTaps,
              dstRowOffset + rows <= maxCaptureRows else { return }
        var flags: UInt32 = 0
        if singleStream {
            flags = 4
        } else if let def = deferredInject {
            flags = def.isMHC ? 2 : 1
        }
        if weights.mainProj.isRotated {
            flags |= 8
        }
        let dummy = deferredInject?.src ?? base
        let inj = deferredInject?.inj ?? base
        enc.setComputePipelineState(psoTapMean)
        enc.bindTensors(base, dummy, inj, tapBuf, startIndex: 0)
        enc.setUInt32s(srcRowOffset, dstRowOffset, slot, weights.hidden / 8, weights.numTaps, max(1, hc), Int(flags), startIndex: 4)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    func encodeContextIngest(
        _ enc: any MTLComputeCommandEncoder,
        driver: QWEngineDriver,
        slot: SessionSlot,
        pos0: Int,
        rows: Int
    ) throws {
        guard rows > 0, rows <= maxCaptureRows else { return }
        let H = weights.hidden
        let aGeo = weights.attnGeometry
        let tapRows = tapBuf.slice(byteOffset: 0, count: rows * weights.numTaps * H, dtype: .f16)
        let mpOut = mainProjRaw.slice(byteOffset: 0, count: rows * H, dtype: .f16)
        let mnOut = mainNormRot.slice(byteOffset: 0, count: rows * H, dtype: .f16)
        let kvOut = ctxKVRaw.slice(byteOffset: 0, count: rows * aGeo.headDim, dtype: .f16)

        try encodeMatrixProj(enc, driver: driver, m: rows, n: H, k: weights.numTaps * H,
                             x: tapRows, w: weights.mainProj, y16: mpOut)
        enc.memoryBarrier(scope: .buffers)
        let normFlags: UInt32 = 1 | (weights.layers.first?.kvProj.isRotated == true ? 2 : 0)
        encodeNormRoPE(enc, rawIn: mpOut, normW: weights.mainNorm, kvOut: mnOut,
                       rows: rows, dim: H, nopeDim: H, ropeDim: 0, pos0: 0,
                       ringSize: 1, flags: normFlags)
        enc.memoryBarrier(scope: .buffers)

        for l in 0..<config.numHiddenLayers {
            try encodeMatrixProj(enc, driver: driver, m: rows, n: aGeo.headDim, k: H,
                                 x: mnOut, w: weights.layers[l].kvProj, y16: kvOut)
            enc.memoryBarrier(scope: .buffers)
            encodeNormRoPE(enc, rawIn: kvOut, normW: weights.layers[l].kvNormW, kvOut: slot.swaKV[l],
                           rows: rows, dim: aGeo.headDim, nopeDim: aGeo.nopeDim, ropeDim: aGeo.ropeDim,
                           pos0: pos0, ringSize: config.slidingWindow, flags: 0)
            if l + 1 < config.numHiddenLayers {
                enc.memoryBarrier(scope: .buffers)
            }
        }
    }

    func encodeContextIngestCB(
        _ cb: any MTLCommandBuffer,
        driver: QWEngineDriver,
        slot: SessionSlot,
        pos0: Int,
        rows: Int
    ) throws {
        guard rows > 0 else { return }
        guard let enc = cb.makeComputeCommandEncoder() else {
            throw QWEngineDriverError.state("DSpark context ingest encoder creation failed")
        }
        hal.touchKeepAlive(enc)
        do {
            useResources(on: enc)
            try encodeContextIngest(enc, driver: driver, slot: slot, pos0: pos0, rows: rows)
        } catch {
            enc.endEncoding()
            throw error
        }
        enc.endEncoding()
    }

    func enqueueGreedyPropose(
        driver: QWEngineDriver,
        slot: SessionSlot,
        pos0: Int,
        anchorToken: Int,
        requestedDrafts: Int,
        journalBuffer: (any MTLBuffer)? = nil,
        copyEmbedRow: ((Int, QWTensorValue) throws -> Void)? = nil
    ) throws -> (cb: any MTLCommandBuffer, maxDrafts: Int) {
        let draftLimit = min(requestedDrafts, config.maxDrafts)
        guard draftLimit >= 1 else {
            throw QWEngineDriverError.state("DSpark draftLimit must be >= 1")
        }
        let B = config.blockSize
        let pPtr = promptToksBuf.contents().assumingMemoryBound(to: Int32.self)
        let seqPtr = tokSeqBuf.contents().assumingMemoryBound(to: Int32.self)
        pPtr[0] = Int32(anchorToken)
        seqPtr[0] = Int32(anchorToken)
        for i in 1..<B {
            pPtr[i] = Int32(config.noiseTokenId)
            seqPtr[i] = 0
        }
        seqPtr[B] = 0

        if driver.mtpDeviceEmbedding == nil, let copyEmbedRow {
            for r in 0..<B {
                let rowVal = blkX.slice(byteOffset: r * weights.hidden * 2, count: weights.hidden, dtype: .f16)
                try copyEmbedRow(Int(pPtr[r]), rowVal)
            }
        }

        let headRows = (config.sampleOffset == 0) ? draftLimit : B
        if Self.stageProfileEnabled {
            let profiled = try encodeGreedyProfiledCBs(
                driver: driver, slot: slot, pos0: pos0, draftLimit: draftLimit,
                headRows: headRows, journalBuffer: journalBuffer
            )
            pendingProfileStages = (profiled.blockCB, profiled.headCB, pos0)
            return (profiled.markovCB, draftLimit)
        }
        pendingProfileStages = nil

        guard let cb = hal.makeUnretainedCommandBuffer() else {
            throw QWEngineDriverError.state("DSpark greedy propose command buffer failed")
        }
        var enc = try makeProposeEncoder(cb)
        do {
            try encodeBlockAndBaseLogits(&enc, cb: cb, driver: driver, slot: slot, pos0: pos0,
                                         anchorToken: anchorToken, headRows: headRows)
            try encodeGreedyMarkovSteps(enc, count: draftLimit)
        } catch {
            enc.endEncoding()
            throw error
        }
        enc.endEncoding()
        if let journalBuffer {
            guard let blit = cb.makeBlitCommandEncoder() else {
                throw QWEngineDriverError.state("DSpark journal blit encoder failed")
            }
            blit.copy(from: tokSeqBuf, sourceOffset: 0, to: journalBuffer, destinationOffset: 0,
                      size: (1 + draftLimit) * 4)
            blit.endEncoding()
        }
        cb.commit()
        return (cb, draftLimit)
    }

    func readGreedyDraftsAndSelectCount(
        completedCB: any MTLCommandBuffer,
        maxDrafts: Int,
        requestedDrafts: Int
    ) throws -> [Int] {
        completedCB.waitUntilCompleted()
        let stages = pendingProfileStages
        pendingProfileStages = nil
        guard completedCB.status == .completed else {
            throw QWEngineDriverError.state("DSpark propose GPU failure: \(String(describing: completedCB.error))")
        }
        let seqPtr = tokSeqBuf.contents().assumingMemoryBound(to: Int32.self)
        let activeCount: Int
        let tailConf: Float?
        if schedulerEnabled {
            let cPtr = confOut.region.buffer.contents().advanced(by: confOut.region.base + confOut.offset)
                .assumingMemoryBound(to: Float.self)
            let probs = (0..<maxDrafts).map { cPtr[$0] }
            activeCount = confidenceScheduler.optimalDraftCount(
                confidences: probs, fallbackMax: min(maxDrafts, requestedDrafts)
            )
            tailConf = activeCount > 0 ? probs[activeCount - 1] : nil
        } else {
            activeCount = min(maxDrafts, requestedDrafts)
            tailConf = nil
        }
        lastTailConfidence = tailConf
        logProfileStagesIfNeeded(stages: stages, markovCB: completedCB, drafts: activeCount)
        return (1...activeCount).map { Int(seqPtr[$0]) }
    }

    func proposeStochastic(
        driver: QWEngineDriver,
        session: QWEngineSession,
        slot: SessionSlot,
        pos0: Int,
        anchorToken: Int,
        requestedDrafts: Int,
        owner: QWStochasticSession,
        copyEmbedRow: ((Int, QWTensorValue) throws -> Void)? = nil,
        completed: ((any MTLCommandBuffer) throws -> Void)? = nil
    ) throws -> (drafts: [Int], proposals: [QWMTPGumbelProposal]) {
        let draftLimit = min(requestedDrafts, config.maxDrafts)
        guard draftLimit >= 1 else { return ([], []) }
        let B = config.blockSize
        let pPtr = promptToksBuf.contents().assumingMemoryBound(to: Int32.self)
        let seqPtr = tokSeqBuf.contents().assumingMemoryBound(to: Int32.self)
        pPtr[0] = Int32(anchorToken)
        seqPtr[0] = Int32(anchorToken)
        for i in 1..<B { pPtr[i] = Int32(config.noiseTokenId) }

        if driver.mtpDeviceEmbedding == nil, let copyEmbedRow {
            for r in 0..<B {
                let rowVal = blkX.slice(byteOffset: r * weights.hidden * 2, count: weights.hidden, dtype: .f16)
                try copyEmbedRow(Int(pPtr[r]), rowVal)
            }
        }

        let headRows = (config.sampleOffset == 0) ? draftLimit : B
        guard let blockCB = hal.makeUnretainedCommandBuffer() else {
            throw QWEngineDriverError.state("DSpark stochastic block CB failed")
        }
        var enc = try makeProposeEncoder(blockCB)
        do {
            try encodeBlockAndBaseLogits(&enc, cb: blockCB, driver: driver, slot: slot, pos0: pos0,
                                         anchorToken: anchorToken, headRows: headRows)
        } catch {
            enc.endEncoding()
            throw error
        }
        enc.endEncoding()
        blockCB.commit()
        blockCB.waitUntilCompleted()
        guard blockCB.status == .completed else {
            throw QWEngineDriverError.state("DSpark block forward GPU failure: \(String(describing: blockCB.error))")
        }
        try completed?(blockCB)

        let previewStream = QWMTPRandomStream(cloning: try owner.stream.frozenState())
        try previewStream.beginGumbelRound(drafts: draftLimit, vocabulary: weights.vocab, maxDrafts: 15)
        var drafts: [Int] = []
        var dists: [QWSamplingDistribution] = []
        var confProbs: [Float] = []
        drafts.reserveCapacity(draftLimit)
        dists.reserveCapacity(draftLimit)
        confProbs.reserveCapacity(draftLimit)

        let sampleOff = config.sampleOffset
        let baseByteOff = sampleOff * weights.vocab * 4
        let baseSlice = baseLogits.slice(byteOffset: baseByteOff, count: draftLimit * weights.vocab, dtype: .f32)
        let confPtr = confOut.region.buffer.contents().advanced(by: confOut.region.base + confOut.offset)
            .assumingMemoryBound(to: Float.self)

        for j in 0..<draftLimit {
            guard let stepCB = hal.makeUnretainedCommandBuffer(),
                  let stepEnc = stepCB.makeComputeCommandEncoder() else {
                throw QWEngineDriverError.state("DSpark Markov stochastic step encoder failed")
            }
            hal.touchKeepAlive(stepEnc)
            do {
                useResources(on: stepEnc)
                encodeMarkovLogitsStep(stepEnc, baseLogitsSlice: baseSlice, stepIdx: j)
                stepEnc.memoryBarrier(scope: .buffers)
                if let mm = driver.multimaxParams {
                    try driver.head.encodeMultiMax(stepEnc, logits: stepLogits, params: mm, count: weights.vocab)
                }
                if schedulerEnabled, weights.confidenceHead != nil {
                    encodeConfidenceDot(stepEnc, stepIdx: j, hiddenRowIdx: sampleOff + j)
                }
            } catch {
                stepEnc.endEncoding()
                throw error
            }
            stepEnc.endEncoding()
            stepCB.commit()
            stepCB.waitUntilCompleted()
            guard stepCB.status == .completed else {
                throw QWEngineDriverError.state("DSpark Markov step GPU failure: \(String(describing: stepCB.error))")
            }
            try completed?(stepCB)

            let draftHistory = session.activeRepetitionHistory(extraTokens: drafts)
            let dist = try owner.distribution(logits: stepLogits, repetitionHistory: draftHistory)
            let sampled = try previewStream.sampleGumbelProposal(dist)
            drafts.append(sampled.token)
            dists.append(dist)
            seqPtr[j + 1] = Int32(sampled.token)
            if schedulerEnabled, weights.confidenceHead != nil {
                confProbs.append(confPtr[j])
            }
        }
        let kStar = (schedulerEnabled && confProbs.count == drafts.count)
            ? confidenceScheduler.optimalDraftCount(confidences: confProbs, fallbackMax: drafts.count)
            : drafts.count
        lastTailConfidence = (schedulerEnabled && kStar > 0 && kStar <= confProbs.count) ? confProbs[kStar - 1] : nil
        try owner.stream.beginGumbelRound(drafts: kStar, vocabulary: weights.vocab, maxDrafts: 15)
        var proposals: [QWMTPGumbelProposal] = []
        proposals.reserveCapacity(kStar)
        for dist in dists.prefix(kStar) {
            proposals.append(try owner.stream.sampleGumbelProposal(dist))
        }
        return (Array(drafts.prefix(kStar)), proposals)
    }
}
