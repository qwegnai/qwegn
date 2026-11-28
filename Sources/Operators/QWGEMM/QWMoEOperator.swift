// QWMoEOperator.swift — Unified Sparse-MoE channel-mixer operator for QWEngine.
// Serves both single-stream f16 inputs (via f16->f32 bridge) and pre-rotated f32 Hyper-Connection streams.

import Foundation
import Metal
import QWAlloc
import QWCore
import QWHAL

public enum QWMoEOperator {
    static func resolveF16NormedBridgeInputs(
        _ enc: any MTLComputeCommandEncoder, hal: QWHal, x16: QWTensorValue,
        scratch s: Scratch, router: QWTensorValue, rows T: Int, hidden H: Int,
        routerCols: Int, isGroupedNax: Bool, skipRot32: Bool
    ) throws -> (xPlain: QWTensorValue, xRot: QWTensorValue) {
        let useRouterPlainF16 = skipRot32 && isGroupedNax && T >= 64
            && hal.caps.isMetal4TensorCapable
            && (router.shape.dtype == .bf16 || router.shape.dtype == .f32)
            && routerCols % 64 == 0 && H % 32 == 0
        if useRouterPlainF16 && hal.weightBasis == .standard {
            return (x16, s.xRot32)
        }
        let plainTarget = useRouterPlainF16
            ? s.xPlain32.slice(byteOffset: 0, count: T * H, dtype: .f16)
            : s.xPlain32
        try QWGemm.encodeMoEBridgeF16ToF32(
            enc, hal: hal, x: x16, xRot32: s.xRot32, xPlain32: plainTarget,
            rows: T, hidden: H, skipRot32: skipRot32, plainF16: useRouterPlainF16)
        return (plainTarget, s.xRot32)
    }

    /// Input activation layout entering the Sparse-MoE channel mixer.
    public enum Input {
        /// Normed `f16 [rows, hidden]` activation; the operator runs `encodeMoEBridgeF16ToF32`.
        case f16Normed(QWTensorValue)
        /// Pre-rotated `f32 [rows, hidden]` (`plain` for router, `rotated` for experts, optional `f16` precast for NAX).
        case f32PreRotated(plain: QWTensorValue, rotated: QWTensorValue, prefillPrecast16: QWTensorValue?)
    }

    /// Optional decode router encoder `(enc, hal, weight, x, logits, sg, rows, experts, k)`.
    public typealias DecodeRouterEncoder = (
        any MTLComputeCommandEncoder, QWHal, QWTensorValue, QWTensorValue,
        QWTensorValue, QWTensorValue, Int, Int, Int
    ) throws -> Void

    /// Optional latent-space expert bottleneck (`fc1_latent_proj -> experts -> latent_norm -> fc2_latent_proj`).
    public struct LatentBottleneck {
        public let fc1: QWWeightView
        public let normGamma: QWTensorValue?
        public let normEps: Float
        public let fc2: QWWeightView
        public let latentDim: Int

        public init(fc1: QWWeightView, normGamma: QWTensorValue?, normEps: Float = 1e-6,
                    fc2: QWWeightView, latentDim: Int) {
            self.fc1 = fc1
            self.normGamma = normGamma
            self.normEps = normEps
            self.fc2 = fc2
            self.latentDim = latentDim
        }
    }

    /// Ping-pong latent activation scratch (`[rows, latentDim]` f32 + optional f16 precast for NAX prefill).
    public struct LatentScratch {
        public let bufA32: QWTensorValue
        public let bufB32: QWTensorValue
        public let bufB16: QWTensorValue?

        public init(bufA32: QWTensorValue, bufB32: QWTensorValue, bufB16: QWTensorValue? = nil) {
            self.bufA32 = bufA32
            self.bufB32 = bufB32
            self.bufB16 = bufB16
        }
    }

    /// Unified Sparse-MoE layer weights view.
    public struct Weights {
        public let router: QWTensorValue
        public let routerBias: QWTensorValue?
        public let routedScaleParam: QWTensorValue?
        public let splitFeatureRouting: Bool
        public let decodeRouterOverride: DecodeRouterEncoder?
        public let gateUp: QWWeightView
        public let down: QWWeightView
        public let coldGateUp: QWWeightView?
        public let coldDown: QWWeightView?
        public let sharedGateUp: QWWeightView?
        public let sharedDown: QWWeightView?
        public let sharedGateScalar: QWTensorValue?
        public let latent: LatentBottleneck?
        public let latentD: QWWeightView?
        public let latentU: QWWeightView?
        public let latentLabels: QWTensorValue?
        public let numLatentGroups: Int
        public let latentDim: Int
        public let hidden: Int
        public let experts: Int
        public let topK: Int
        public let intermediate: Int
        public let sharedIntermediate: Int
        public let normTopK: Bool
        public let reuseExpertsAcrossTokens: Bool
        public let moeSpecExpertBudget: Int
        public let fpGateUp: QWMoEExpertBank?
        public let fpUp: QWMoEExpertBank?
        public let fpDown: QWMoEExpertBank?
        public let gateSoftcap: Float
        public let upSoftcap: Float
        public let useSigmoidScore: Bool
        public let nGroup: Int
        public let topkGroup: Int
        public let groupTopM: Int
        public let routedSwigluClamp: QWSwiGLUClampSpec?
        public let sharedSwigluClamp: QWSwiGLUClampSpec?

        public init(router: QWTensorValue, routerBias: QWTensorValue? = nil,
                    routedScaleParam: QWTensorValue? = nil,
                    splitFeatureRouting: Bool = false,
                    decodeRouterOverride: DecodeRouterEncoder? = nil,
                    gateUp: QWWeightView, down: QWWeightView,
                    coldGateUp: QWWeightView? = nil, coldDown: QWWeightView? = nil,
                    sharedGateUp: QWWeightView? = nil, sharedDown: QWWeightView? = nil,
                    sharedGateScalar: QWTensorValue? = nil,
                    latent: LatentBottleneck? = nil,
                    latentD: QWWeightView? = nil, latentU: QWWeightView? = nil,
                    latentLabels: QWTensorValue? = nil,
                    numLatentGroups: Int = 0, latentDim: Int = 0,
                    hidden: Int, experts: Int, topK: Int,
                    intermediate: Int, sharedIntermediate: Int,
                    normTopK: Bool = true, reuseExpertsAcrossTokens: Bool = true,
                    moeSpecExpertBudget: Int = 0,
                    fpGateUp: QWMoEExpertBank? = nil,
                    fpUp: QWMoEExpertBank? = nil,
                    fpDown: QWMoEExpertBank? = nil,
                    gateSoftcap: Float = 0, upSoftcap: Float = 0,
                    useSigmoidScore: Bool = false,
                    nGroup: Int = 1, topkGroup: Int = 1, groupTopM: Int = 1,
                    routedSwigluClamp: QWSwiGLUClampSpec? = nil,
                    sharedSwigluClamp: QWSwiGLUClampSpec? = nil) {
            self.router = router
            self.routerBias = routerBias
            self.routedScaleParam = routedScaleParam
            self.splitFeatureRouting = splitFeatureRouting
            self.decodeRouterOverride = decodeRouterOverride
            self.gateUp = gateUp
            self.down = down
            self.coldGateUp = coldGateUp
            self.coldDown = coldDown
            self.sharedGateUp = sharedGateUp
            self.sharedDown = sharedDown
            self.sharedGateScalar = sharedGateScalar
            self.latent = latent
            self.latentD = latentD
            self.latentU = latentU
            self.latentLabels = latentLabels
            self.numLatentGroups = numLatentGroups
            self.latentDim = latentDim
            self.hidden = hidden
            self.experts = experts
            self.topK = topK
            self.intermediate = intermediate
            self.sharedIntermediate = sharedIntermediate
            self.normTopK = normTopK
            self.reuseExpertsAcrossTokens = reuseExpertsAcrossTokens
            self.moeSpecExpertBudget = moeSpecExpertBudget
            self.fpGateUp = fpGateUp
            self.fpUp = fpUp
            self.fpDown = fpDown
            self.gateSoftcap = gateSoftcap
            self.upSoftcap = upSoftcap
            self.useSigmoidScore = useSigmoidScore
            self.nGroup = nGroup
            self.topkGroup = topkGroup
            self.groupTopM = groupTopM
            self.routedSwigluClamp = routedSwigluClamp
            self.sharedSwigluClamp = sharedSwigluClamp
        }
    }

    /// Pure-FP Sparse-MoE weights (`BF16` router/shared + `NVFP4_G16`/`MXFP8_G32` experts).
    public struct FPWeights: Sendable {
        public let router: QWGemmFP.BF16Bank
        public let routerBias: QWTensorValue?
        public let routedScaleParam: QWTensorValue?
        public let splitFeatureRouting: Bool
        public let gateUp: QWMoEExpertBank
        public let up: QWMoEExpertBank?
        public let down: QWMoEExpertBank
        public let sharedGateUp: QWGemmFP.BF16Bank?
        public let sharedUp: QWGemmFP.BF16Bank?
        public let sharedDown: QWGemmFP.BF16Bank?
        public let hidden: Int
        public let experts: Int
        public let topK: Int
        public let intermediate: Int
        public let sharedIntermediate: Int
        public let gateSoftcap: Float
        public let upSoftcap: Float
        public let normTopK: Bool
        public let moeSpecExpertBudget: Int
        public let useSigmoidScore: Bool
        public let nGroup: Int
        public let topkGroup: Int
        public let groupTopM: Int

        public init(router: QWGemmFP.BF16Bank, routerBias: QWTensorValue? = nil,
                    routedScaleParam: QWTensorValue? = nil, splitFeatureRouting: Bool = false,
                    gateUp: QWMoEExpertBank, up: QWMoEExpertBank? = nil, down: QWMoEExpertBank,
                    sharedGateUp: QWGemmFP.BF16Bank? = nil, sharedUp: QWGemmFP.BF16Bank? = nil,
                    sharedDown: QWGemmFP.BF16Bank? = nil,
                    hidden: Int, experts: Int, topK: Int,
                    intermediate: Int, sharedIntermediate: Int = 0,
                    gateSoftcap: Float = 0, upSoftcap: Float = 0,
                    normTopK: Bool = true, moeSpecExpertBudget: Int = 0,
                    useSigmoidScore: Bool = false, nGroup: Int = 1,
                    topkGroup: Int = 1, groupTopM: Int = 1) {
            self.router = router; self.routerBias = routerBias; self.routedScaleParam = routedScaleParam
            self.splitFeatureRouting = splitFeatureRouting; self.gateUp = gateUp; self.up = up; self.down = down
            self.sharedGateUp = sharedGateUp; self.sharedUp = sharedUp; self.sharedDown = sharedDown
            self.hidden = hidden; self.experts = experts; self.topK = topK
            self.intermediate = intermediate; self.sharedIntermediate = sharedIntermediate
            self.gateSoftcap = gateSoftcap; self.upSoftcap = upSoftcap
            self.normTopK = normTopK; self.moeSpecExpertBudget = moeSpecExpertBudget
            self.useSigmoidScore = useSigmoidScore; self.nGroup = nGroup
            self.topkGroup = topkGroup; self.groupTopM = groupTopM
        }
    }

    /// Scratch buffers for pure-FP Sparse-MoE prefill (grouped NAX) and decode/verify (compact).
    public struct FPScratch {
        public let logits: QWTensorValue
        public let idx: QWTensorValue
        public let wts: QWTensorValue
        public let expOff: QWTensorValue
        public let sortedPair: QWTensorValue
        public let sortedTok: QWTensorValue
        public let invPerm: QWTensorValue
        public let tileE: QWTensorValue?
        public let tileR: QWTensorValue?
        public let tileCount: QWTensorValue?
        public let activeExperts: QWTensorValue?
        public let activeIndirect: QWTensorValue?
        public let gu: QWTensorValue
        public let act: QWTensorValue
        public let dnAcc: QWTensorValue
        public let guSh: QWTensorValue?
        public let actSh: QWTensorValue?
        public let shared: QWTensorValue?
        public let gate: QWTensorValue?
        public let interDtype: QWFPElemDtype

        public init(logits: QWTensorValue, idx: QWTensorValue, wts: QWTensorValue,
                    expOff: QWTensorValue, sortedPair: QWTensorValue,
                    sortedTok: QWTensorValue, invPerm: QWTensorValue,
                    tileE: QWTensorValue? = nil, tileR: QWTensorValue? = nil,
                    tileCount: QWTensorValue? = nil,
                    activeExperts: QWTensorValue? = nil, activeIndirect: QWTensorValue? = nil,
                    gu: QWTensorValue, act: QWTensorValue, dnAcc: QWTensorValue,
                    guSh: QWTensorValue? = nil, actSh: QWTensorValue? = nil,
                    shared: QWTensorValue? = nil, gate: QWTensorValue? = nil,
                    interDtype: QWFPElemDtype = .bf16) {
            self.logits = logits; self.idx = idx; self.wts = wts
            self.expOff = expOff; self.sortedPair = sortedPair
            self.sortedTok = sortedTok; self.invPerm = invPerm
            self.tileE = tileE; self.tileR = tileR; self.tileCount = tileCount
            self.activeExperts = activeExperts; self.activeIndirect = activeIndirect
            self.gu = gu; self.act = act; self.dnAcc = dnAcc
            self.guSh = guSh; self.actSh = actSh
            self.shared = shared; self.gate = gate
            self.interDtype = interDtype
        }
    }

    /// Unified Sparse-MoE scratch tensors for decode (`rows = 1`), verify (`rows <= 8`), and prefill (`rows > 8`).
    public struct Scratch {
        public let xRot32: QWTensorValue
        public let xPlain32: QWTensorValue
        public let logits: QWTensorValue
        public let idx: QWTensorValue
        public let wts: QWTensorValue
        public let gu32: QWTensorValue
        public let act32: QWTensorValue
        public let guSh32: QWTensorValue?
        public let actSh32: QWTensorValue?
        public let shared: QWTensorValue
        public let gate: QWTensorValue
        public let expOff: QWTensorValue?
        public let sortedPair: QWTensorValue?
        public let sortedTok: QWTensorValue?
        public let invPerm: QWTensorValue?
        public let tileE: QWTensorValue?
        public let tileR: QWTensorValue?
        public let tileCount: QWTensorValue?
        public let gu16: QWTensorValue?
        public let dnHi: QWTensorValue?
        public let dnAcc: QWTensorValue
        public let naxX16: QWTensorValue?
        public let naxY16: QWTensorValue?
        public let latent: LatentScratch?
        public let groupIds: QWTensorValue?
        public let onesRoute: QWTensorValue?
        public let xLatent32: QWTensorValue?
        public let xLatent16: QWTensorValue?
        public let zLatent32: QWTensorValue?
        public let maxTileWidth: Int
        public let activeExperts: QWTensorValue?
        public let activeIndirect: QWTensorValue?

        public init(xRot32: QWTensorValue, xPlain32: QWTensorValue,
                    logits: QWTensorValue, idx: QWTensorValue, wts: QWTensorValue,
                    gu32: QWTensorValue, act32: QWTensorValue,
                    guSh32: QWTensorValue?, actSh32: QWTensorValue?,
                    shared: QWTensorValue, gate: QWTensorValue,
                    expOff: QWTensorValue?, sortedPair: QWTensorValue?,
                    sortedTok: QWTensorValue?, invPerm: QWTensorValue?,
                    tileE: QWTensorValue?, tileR: QWTensorValue?, tileCount: QWTensorValue?,
                    gu16: QWTensorValue?, dnHi: QWTensorValue?, dnAcc: QWTensorValue,
                    naxX16: QWTensorValue?, naxY16: QWTensorValue? = nil,
                    latent: LatentScratch? = nil,
                    groupIds: QWTensorValue? = nil, onesRoute: QWTensorValue? = nil,
                    xLatent32: QWTensorValue? = nil, xLatent16: QWTensorValue? = nil,
                    zLatent32: QWTensorValue? = nil, maxTileWidth: Int,
                    activeExperts: QWTensorValue? = nil,
                    activeIndirect: QWTensorValue? = nil) {
            self.xRot32 = xRot32; self.xPlain32 = xPlain32
            self.logits = logits; self.idx = idx; self.wts = wts
            self.gu32 = gu32; self.act32 = act32
            self.guSh32 = guSh32; self.actSh32 = actSh32
            self.shared = shared; self.gate = gate
            self.expOff = expOff; self.sortedPair = sortedPair
            self.sortedTok = sortedTok; self.invPerm = invPerm
            self.tileE = tileE; self.tileR = tileR; self.tileCount = tileCount
            self.gu16 = gu16; self.dnHi = dnHi; self.dnAcc = dnAcc
            self.naxX16 = naxX16; self.naxY16 = naxY16
            self.latent = latent
            self.groupIds = groupIds; self.onesRoute = onesRoute
            self.xLatent32 = xLatent32; self.xLatent16 = xLatent16; self.zLatent32 = zLatent32
            self.maxTileWidth = maxTileWidth
            self.activeExperts = activeExperts
            self.activeIndirect = activeIndirect
        }
    }

    public static func canUseNaxDenseProj(
        hal: QWHal, weight: QWWeightView?, rows: Int, maxTileWidth: Int, minNaxRows: Int = 64
    ) -> Bool {
        guard let weight, hal.caps.supportsSimdgroupMMA else { return false }
        return rows >= minNaxRows
            && weight.n % 64 == 0
            && weight.k % 64 == 0
            && weight.k <= maxTileWidth
    }

    private static let envMoENaxSwigluEnabled: Bool =
        ProcessInfo.processInfo.environment["QW_MOE_NAX_SWIGLU"] != "0"

    private static let envDenseSwigluNax: String? =
        ProcessInfo.processInfo.environment["QW_DENSE_SWIGLU_NAX"]

    static func canFuseMoEGateUpNaxSwiglu(hal: QWHal, weights w: Weights, intermediate I: Int) -> Bool {
        hal.caps.supportsSimdgroupMMA
            && envMoENaxSwigluEnabled
            && I > 0 && I % 32 == 0
            && w.gateSoftcap == 0 && w.upSoftcap == 0
            && QWGemm.supportsMoEGateUpNaxSwiglu(w.gateUp)
            && (w.coldGateUp == nil || w.coldGateUp!.containerKind == w.gateUp.containerKind)
    }

    static func canFuseSharedSwiGLUNax(
        hal: QWHal,
        weights w: Weights,
        shPrecast16: QWTensorValue?,
        shAct16: QWTensorValue?
    ) -> Bool {
        if !hal.caps.supportsSimdgroupMMA || envDenseSwigluNax == "0" { return false }
        guard let shGU = w.sharedGateUp else { return false }
        let widthOK = envDenseSwigluNax == "1" || shGU.bits <= 8 || shGU.bits == 15
        let ISh = w.sharedIntermediate
        return widthOK
            && ISh > 0 && ISh % 64 == 0
            && shPrecast16 != nil && shAct16 != nil
            && QWGemm.supportsDenseSwiGLUNax(shGU)
    }

    static func encodeRoutedNaxGateUpAndAct(
        _ enc: any MTLComputeCommandEncoder,
        hal: QWHal,
        weights w: Weights,
        xe: QWTensorValue,
        expOff: QWTensorValue,
        tileE: QWTensorValue,
        tileR: QWTensorValue,
        tileCount: QWTensorValue,
        gu16: QWTensorValue?,
        dnHi: QWTensorValue,
        intermediate I: Int,
        pairs: Int,
        tilePlan: QWGemm.MoETilePlan,
        sortedTok: QWTensorValue,
        barrier: () -> Void
    ) throws {
        if canFuseMoEGateUpNaxSwiglu(hal: hal, weights: w, intermediate: I) {
            try QWGemm.encodeMoEGateUpNaxSwiglu(
                enc, hal: hal, xe: xe, bank: w.gateUp, expertOff: expOff,
                tileE: tileE, tileR: tileR, tileCount: tileCount,
                out16: dnHi, intermediate: I, tilePlan: tilePlan,
                sortedTok: sortedTok, coldBank: w.coldGateUp,
                swigluClamp: w.routedSwigluClamp)
            return
        }
        guard let gu16 else {
            throw QWGemmError.badShape("non-fused grouped NAX MoE requires gu16 scratch")
        }
        try QWGemm.encodeMoEGateUpNax(enc, hal: hal, xe: xe, bank: w.gateUp, expertOff: expOff,
                                      tileE: tileE, tileR: tileR, tileCount: tileCount,
                                      gu16: gu16, rowsPerExpert: 2 * I, tilePlan: tilePlan,
                                      sortedTok: sortedTok, coldBank: w.coldGateUp)
        barrier()
        try QWGemm.encodeSiluMulSplitF16(enc, hal: hal, gu: gu16, outHi: dnHi, outLo: dnHi,
                                         rows: pairs, intermediate: I,
                                         swigluClamp: w.routedSwigluClamp)
    }

    /// Encodes an all-resident Sparse-MoE pass over `rows` tokens into `out32` (and optional `out16`).
    public static func encodeResident(
        _ enc: inout any MTLComputeCommandEncoder,
        hal: QWHal,
        weights w: Weights,
        scratch s: Scratch,
        input: Input,
        out32: QWTensorValue,
        out16: QWTensorValue? = nil,
        skipOut32: Bool = false,
        rows T: Int,
        supportsGroupedNaxPrefill: Bool = true,
        concurrentBarriers: Bool = false,
        rotateEncoder: ((inout any MTLComputeCommandEncoder) throws -> Void)? = nil,
        profileStage: ((String) -> (any MTLComputeCommandEncoder)?)? = nil
    ) throws {
        if let latent = w.latent {
            try encodeResidentLatent(
                &enc, hal: hal, weights: w, latent: latent, scratch: s, input: input,
                out32: out32, out16: out16, skipOut32: skipOut32, rows: T,
                supportsGroupedNaxPrefill: supportsGroupedNaxPrefill,
                concurrentBarriers: concurrentBarriers,
                rotateEncoder: rotateEncoder, profileStage: profileStage
            )
            return
        }
        if w.numLatentGroups > 0 && w.latentDim > 0 && w.latentD != nil && w.latentU != nil && w.latentLabels != nil {
            try encodeResidentGroupLatent(
                &enc, hal: hal, weights: w, scratch: s, input: input,
                out32: out32, out16: out16, skipOut32: skipOut32, rows: T,
                supportsGroupedNaxPrefill: supportsGroupedNaxPrefill,
                concurrentBarriers: concurrentBarriers,
                rotateEncoder: rotateEncoder, profileStage: profileStage
            )
            return
        }
        if let fpGateUp = w.fpGateUp, let fpDown = w.fpDown,
           fpGateUp.isFPContainer, fpDown.isFPContainer {
            try encodeResidentWithFPBanks(
                &enc, hal: hal, weights: w, scratch: s, input: input,
                fpGateUp: fpGateUp, fpUp: w.fpUp, fpDown: fpDown,
                out32: out32, out16: out16, skipOut32: skipOut32, rows: T,
                supportsGroupedNaxPrefill: supportsGroupedNaxPrefill,
                concurrentBarriers: concurrentBarriers,
                rotateEncoder: rotateEncoder)
            return
        }
        @inline(__always) func barrier() {
            if concurrentBarriers { enc.memoryBarrier(scope: .buffers) }
        }
        let H = w.hidden, E = w.experts, K = w.topK, I = w.intermediate, ISh = w.sharedIntermediate
        let routerCols = w.splitFeatureRouting ? (2 * E) : E
        let useFusedMoE = Self.canFuseMoEGateUpNaxSwiglu(hal: hal, weights: w, intermediate: I)
        let isGroupedNax = T > 8 && supportsGroupedNaxPrefill && hal.caps.supportsSimdgroupMMA
            && s.expOff != nil && s.sortedPair != nil && s.sortedTok != nil
            && s.invPerm != nil && s.tileE != nil && s.tileR != nil
            && s.tileCount != nil && (useFusedMoE || s.gu16 != nil)
            && s.dnHi != nil && s.naxX16 != nil
        let skipRot32 = isGroupedNax && (
            ISh == 0 || (T >= 64 && H % 64 == 0 && (2 * ISh) % 64 == 0)
        )

        let xPlain: QWTensorValue
        let xRot: QWTensorValue
        let xe16ForNax: QWTensorValue?
        let shPrecast16: QWTensorValue?
        let needsCastXe16: Bool
        switch input {
        case .f16Normed(let x16):
            (xPlain, xRot) = try Self.resolveF16NormedBridgeInputs(
                enc, hal: hal, x16: x16, scratch: s, router: w.router,
                rows: T, hidden: H, routerCols: routerCols,
                isGroupedNax: isGroupedNax, skipRot32: skipRot32)
            xe16ForNax = x16
            shPrecast16 = T >= 64 ? x16 : nil
            needsCastXe16 = false
        case .f32PreRotated(let plain, let rotated, let precast16):
            xPlain = plain
            xRot = rotated
            xe16ForNax = precast16 ?? s.naxX16
            shPrecast16 = T >= 64 ? (precast16 ?? s.naxX16) : nil
            needsCastXe16 = (precast16 == nil)
        }

        if isGroupedNax,
           let expOff = s.expOff, let sortedPair = s.sortedPair,
           let sortedTok = s.sortedTok, let invPerm = s.invPerm,
           let tileE = s.tileE, let tileR = s.tileR, let tileCount = s.tileCount,
           let dnHi = s.dnHi,
           let naxX16 = s.naxX16, let xe16 = xe16ForNax {
            if needsCastXe16 {
                try QWGemm.encodeCast(enc, hal: hal, x32: xRot, y16: naxX16, count: T * H)
                enc.memoryBarrier(scope: .buffers)
            }
            let canAliasShAct16 = ISh > 0
                && naxX16.shape.elementCount >= T * (H + ISh)
                && canUseNaxDenseProj(hal: hal, weight: w.sharedDown, rows: T, maxTileWidth: s.maxTileWidth)
            let shAct16 = canAliasShAct16
                ? naxX16.slice(byteOffset: T * H * 2, count: T * ISh, dtype: .f16)
                : nil
            let useFusedSh = Self.canFuseSharedSwiGLUNax(hal: hal, weights: w, shPrecast16: shPrecast16, shAct16: shAct16)
            let dnAcc16 = s.dnAcc.slice(byteOffset: 0, count: T * K * H, dtype: .f16)
            try QWGemm.encodeFnRouterGemm(enc, hal: hal, weight: w.router, x: xPlain, y: s.logits,
                                          rows: T, n: routerCols, k: H,
                                          shGateWeight: ISh > 0 ? w.sharedGateScalar : nil,
                                          shGateOut: ISh > 0 ? s.gate : nil)
            if concurrentBarriers, let shGU = w.sharedGateUp, ISh > 0 {
                if useFusedSh, let x16Sh = shPrecast16, let out16Sh = shAct16 {
                    try QWGemm.encodeDenseSwiGLUNax(enc, hal: hal, weight: shGU, x16: x16Sh, out16: out16Sh,
                                                    rows: T, intermediate: ISh, swigluClamp: w.sharedSwigluClamp)
                } else if let guSh = s.guSh32 {
                    let naxY = s.naxY16 ?? naxX16
                    try QWGemm.encodeDenseProjF32(enc, hal: hal, weight: shGU, x: xRot, y: guSh,
                                                  naxX16: naxX16, naxY16: naxY, maxTileWidth: s.maxTileWidth,
                                                  rows: T, x16Precast: shPrecast16)
                }
            }
            barrier()
            try QWGemm.encodeFnMoeRoute(enc, hal: hal, logits: s.logits, idx: s.idx, wts: s.wts,
                                        rows: T, experts: E, topK: K, normTopK: w.normTopK,
                                        bias: w.routerBias, splitFeature: w.splitFeatureRouting,
                                        routedScaleParam: w.routedScaleParam,
                                        useSigmoid: w.useSigmoidScore, nGroup: w.nGroup,
                                        topkGroup: w.topkGroup, groupTopM: w.groupTopM)
            if !concurrentBarriers, let shGU = w.sharedGateUp, let shDown = w.sharedDown, ISh > 0 {
                if useFusedSh, let x16Sh = shPrecast16, let out16Sh = shAct16 {
                    try QWGemm.encodeDenseSwiGLUNax(enc, hal: hal, weight: shGU, x16: x16Sh, out16: out16Sh,
                                                    rows: T, intermediate: ISh, swigluClamp: w.sharedSwigluClamp)
                    try QWGemm.encodeDenseProjF32(enc, hal: hal, weight: shDown, x: s.actSh32 ?? s.shared,
                                                  y: s.shared, naxX16: out16Sh, naxY16: out16Sh,
                                                  maxTileWidth: s.maxTileWidth, rows: T, x16Precast: out16Sh)
                } else if let guSh = s.guSh32, let actSh = s.actSh32 {
                    let shScratch0 = naxX16.slice(byteOffset: 0, count: T * H, dtype: .f16)
                    try QWGemm.encodeDenseProjF32(enc, hal: hal, weight: shGU, x: xRot, y: guSh,
                                                  naxX16: shScratch0, naxY16: shScratch0, maxTileWidth: H,
                                                  rows: T, x16Precast: shPrecast16)
                    try QWGemm.encodeSiluMulF32(enc, hal: hal, gu: guSh, act: actSh, rows: T,
                                                intermediate: ISh, act16: shAct16,
                                                swigluClamp: w.sharedSwigluClamp)
                    let shScratch = shAct16 ?? shScratch0
                    try QWGemm.encodeDenseProjF32(enc, hal: hal, weight: shDown, x: actSh, y: s.shared,
                                                  naxX16: shScratch, naxY16: shScratch,
                                                  maxTileWidth: shAct16 != nil ? s.maxTileWidth : 0,
                                                  rows: T, x16Precast: shAct16)
                }
            } else if concurrentBarriers, !useFusedSh, let guSh = s.guSh32, let actSh = s.actSh32, ISh > 0 {
                try QWGemm.encodeSiluMulF32(enc, hal: hal, gu: guSh, act: actSh, rows: T,
                                            intermediate: ISh, act16: shAct16,
                                            swigluClamp: w.sharedSwigluClamp)
            }
            barrier()
            guard let rotateEncoder else {
                throw QWGemmError.badShape("moe grouped nax requires rotateEncoder for indirect dispatch")
            }
            let tilePlan = QWGemm.MoETilePlan(experts: E, pairs: T * K, gateUpRows: 2 * I, downRows: H)
            try QWGemm.encodeMoEGroupPairsF32(enc, hal: hal, ids: s.idx, expertOff: expOff,
                                              sortedPairIdx: sortedPair, sortedTok: sortedTok,
                                              experts: E, totalPairs: T * K, routes: K,
                                              invPerm: invPerm, tileE: tileE, tileR: tileR,
                                              tileCount: tileCount, tilePlan: tilePlan)
            if concurrentBarriers, let shDown = w.sharedDown, ISh > 0,
               let actSh = useFusedSh ? (s.actSh32 ?? s.shared) : s.actSh32 {
                let naxY = s.naxY16 ?? naxX16
                try QWGemm.encodeDenseProjF32(enc, hal: hal, weight: shDown, x: actSh, y: s.shared,
                                              naxX16: naxX16, naxY16: naxY,
                                              maxTileWidth: shAct16 != nil ? s.maxTileWidth : 0,
                                              rows: T, x16Precast: shAct16)
            }
            try rotateEncoder(&enc)
            try Self.encodeRoutedNaxGateUpAndAct(
                enc, hal: hal, weights: w, xe: xe16, expOff: expOff,
                tileE: tileE, tileR: tileR, tileCount: tileCount,
                gu16: s.gu16, dnHi: dnHi, intermediate: I, pairs: T * K,
                tilePlan: tilePlan, sortedTok: sortedTok, barrier: barrier)
            barrier()
            try QWGemm.encodeMoENaxSplitF32(enc, hal: hal, xHi: dnHi, xLo: dnHi, bank: w.down,
                                            expertOff: expOff, tileE: tileE, tileR: tileR,
                                            tileCount: tileCount, y32: dnAcc16, rowsPerExpert: H,
                                            tilePlan: tilePlan, coldBank: w.coldDown)
            barrier()
            try QWGemm.encodeMoEDownReduceCombineF32(enc, hal: hal, dnAcc: dnAcc16, invPerm: invPerm,
                                                     weights: s.wts, shared: s.shared, gate: s.gate,
                                                     out: out32, hidden: H, routes: K, tokens: T,
                                                     out16: out16, skipOut32: skipOut32 && out16 != nil)
            barrier()
            return
        }

        // Rank-1 GEMV decode / verify path (T <= 8 or non-NAX shapes)
        if T <= 8, let decodeRouter = w.decodeRouterOverride {
            try decodeRouter(enc, hal, w.router, xPlain, s.logits, s.gate, T, routerCols, H)
        } else {
            try QWGemm.encodeFnRouterGemm(enc, hal: hal, weight: w.router, x: xPlain, y: s.logits,
                                          rows: T, n: routerCols, k: H,
                                          shGateWeight: ISh > 0 ? w.sharedGateScalar : nil,
                                          shGateOut: ISh > 0 ? s.gate : nil)
        }
        let smallShNaxAct16: QWTensorValue? = {
            guard T >= 4, hal.caps.isMetal4TensorCapable, H >= 512, H % 64 == 0, ISh >= 512, ISh % 64 == 0,
                  let xe16 = xe16ForNax, xe16.shape.dtype == .f16, xe16.shape.elementCount >= T * H,
                  let shGU = w.sharedGateUp, let shDown = w.sharedDown,
                  let guSh = s.guSh32, guSh.bytes >= T * ISh * 2,
                  QWGemm.supportsDenseSwiGLUNax(shGU),
                  shDown.isIncumbentContainer
                      || QWGemmAQ.naxF32KernelName(bits: shDown.bits, group: shDown.groupSize, isSHVQ: shDown.isSHVQ) != nil else {
                return nil
            }
            return QWTensorValue(region: guSh.region, offset: guSh.offset, shape: QWTensorShape([T * ISh], .f16))
        }()
        let fuseSh = smallShNaxAct16 == nil && ISh > 0 && ISh <= 4096 && (ISh % 8 == 0)
            && w.sharedGateUp != nil && s.guSh32 != nil && s.actSh32 != nil
        if let shGU = w.sharedGateUp, ISh > 0 {
            if let shAct16 = smallShNaxAct16, let xe16 = xe16ForNax {
                let x16Sh = xe16.shape.elementCount == T * H ? xe16 : xe16.slice(byteOffset: 0, count: T * H, dtype: .f16)
                try QWGemm.encodeDenseSwiGLUNax(enc, hal: hal, weight: shGU, x16: x16Sh, out16: shAct16,
                                                rows: T, intermediate: ISh, swigluClamp: w.sharedSwigluClamp)
            } else if let guSh = s.guSh32 {
                try QWGemm.encodeHQ4F32V4(enc, hal: hal, weight: shGU, x: xRot, y: guSh, rows: T)
            }
        }
        barrier()
        try QWGemm.encodeFnMoeRoute(enc, hal: hal, logits: s.logits, idx: s.idx, wts: s.wts,
                                    rows: T, experts: E, topK: K, normTopK: w.normTopK,
                                    guSh: fuseSh ? s.guSh32 : nil,
                                    actSh: fuseSh ? s.actSh32 : nil,
                                    sharedIntermediate: fuseSh ? ISh : 0,
                                    bias: w.routerBias,
                                    expertBudget: w.moeSpecExpertBudget,
                                    splitFeature: w.splitFeatureRouting,
                                    routedScaleParam: w.routedScaleParam,
                                    useSigmoid: w.useSigmoidScore, nGroup: w.nGroup,
                                    topkGroup: w.topkGroup, groupTopM: w.groupTopM,
                                    sharedSwigluClamp: w.sharedSwigluClamp)
        barrier()
        let canCompactQWP = T >= 2 && T <= 8 && w.latent == nil && w.numLatentGroups == 0
            && QWGemm.supportsCompactQWP(gateUp: w.gateUp, down: w.down)
            && E >= 1 && E <= 1024 && H > 0 && H % 32 == 0 && I > 0 && I % 32 == 0
            && w.gateUp.k == H && w.down.k == I
            && (w.coldGateUp.map { $0.containerKind == w.gateUp.containerKind && $0.k == w.gateUp.k } ?? true)
            && (w.coldDown.map { $0.containerKind == w.down.containerKind && $0.k == w.down.k } ?? true)
            && s.act32.shape.elementCount >= T * K * I
            && s.dnAcc.shape.elementCount >= T * K * H
            && out32.shape.dtype == .f32 && out32.shape.elementCount >= T * H
        if canCompactQWP,
           let expOff = s.expOff, let sortedPair = s.sortedPair,
           let sortedTok = s.sortedTok, let actExp = s.activeExperts,
           let actInd = s.activeIndirect,
           expOff.shape.elementCount >= E + 1,
           actExp.shape.elementCount >= E,
           actInd.bytes >= QWGemmFP.MoEActiveBucketPlan.indirectBufferBytes,
           sortedPair.shape.elementCount >= T * K,
           sortedTok.shape.elementCount >= T * K {
            if !concurrentBarriers, let nextEnc = profileStage?("moe.shared_down") { enc = nextEnc }
            if let shDown = w.sharedDown, ISh > 0 {
                if let shAct16 = smallShNaxAct16 {
                    try QWGemm.encodeNaxF32(enc, hal: hal, weight: shDown, x16: shAct16, y16: shAct16, y32: s.shared, rows: T)
                    barrier()
                } else if let guSh = s.guSh32, let actSh = s.actSh32 {
                    if !fuseSh {
                        try QWGemm.encodeSiluMulF32(enc, hal: hal, gu: guSh, act: actSh, rows: T,
                                                    intermediate: ISh, swigluClamp: w.sharedSwigluClamp)
                        barrier()
                    }
                    try QWGemm.encodeHQ4F32V4(enc, hal: hal, weight: shDown, x: actSh, y: s.shared, rows: T)
                    barrier()
                }
            }
            if !concurrentBarriers, let nextEnc = profileStage?("moe.experts_gate_up") { enc = nextEnc }
            try QWGemm.encodeMoEQWPCompactRoutedCombine(
                enc, hal: hal,
                x: xRot, gateUp: w.gateUp, down: w.down,
                ids: s.idx, weights: s.wts,
                shared: s.shared, gate: s.gate, out: out32,
                expertOff: expOff, activeExperts: actExp, activeIndirect: actInd,
                sortedPairIdx: sortedPair, sortedTok: sortedTok,
                actBuf: s.act32, dnAccBuf: s.dnAcc,
                hidden: H, experts: E, routes: K, intermediate: I, tokens: T,
                out16: out16, coldGateUp: w.coldGateUp, coldDown: w.coldDown,
                gateSoftcap: w.gateSoftcap, upSoftcap: w.upSoftcap,
                applyFwht: true, skipShared: false,
                concurrentBarriers: concurrentBarriers,
                skipOut32: skipOut32, nofmaShared: true,
                swigluClamp: w.routedSwigluClamp)
            barrier()
            return
        }
        let fuseRoutedGU = I > 0 && (I % 32 == 0)
        if concurrentBarriers {
            try QWGemm.encodeMoEGateUpF32R1(enc, hal: hal, x: xRot, bank: w.gateUp, ids: s.idx,
                                            out: fuseRoutedGU ? s.act32 : s.gu32,
                                            rowsPerExpert: 2 * I, routes: K, tokens: T,
                                            fuseSiluFwht: fuseRoutedGU,
                                            coldBank: w.coldGateUp,
                                            reuseExpertsAcrossTokens: w.reuseExpertsAcrossTokens,
                                            swigluClamp: w.routedSwigluClamp)
            if let shDown = w.sharedDown, ISh > 0 {
                if let shAct16 = smallShNaxAct16 {
                    try QWGemm.encodeNaxF32(enc, hal: hal, weight: shDown, x16: shAct16, y16: shAct16, y32: s.shared, rows: T)
                } else if let guSh = s.guSh32, let actSh = s.actSh32 {
                    if !fuseSh {
                        try QWGemm.encodeSiluMulF32(enc, hal: hal, gu: guSh, act: actSh, rows: T,
                                                    intermediate: ISh, swigluClamp: w.sharedSwigluClamp)
                        barrier()
                    }
                    try QWGemm.encodeHQ4F32V4(enc, hal: hal, weight: shDown, x: actSh, y: s.shared, rows: T)
                }
            }
            barrier()
            if !fuseRoutedGU {
                try QWGemm.encodeSiluMulF32(enc, hal: hal, gu: s.gu32, act: s.act32, rows: T * K,
                                            intermediate: I, swigluClamp: w.routedSwigluClamp)
                barrier()
            }
            try QWGemm.encodeMoEDownSharedCombineF32(enc, hal: hal, x: s.act32, bank: w.down,
                                                     rowsPerRoute: H, routes: K, weights: s.wts,
                                                     shared: s.shared, gate: s.gate, out: out32,
                                                     ids: s.idx, tokens: T, out16: out16,
                                                     coldBank: w.coldDown,
                                                     reuseExpertsAcrossTokens: w.reuseExpertsAcrossTokens)
            barrier()
        } else {
            if let nextEnc = profileStage?("moe.shared_down") { enc = nextEnc }
            if let shDown = w.sharedDown, ISh > 0 {
                if let shAct16 = smallShNaxAct16 {
                    try QWGemm.encodeNaxF32(enc, hal: hal, weight: shDown, x16: shAct16, y16: shAct16, y32: s.shared, rows: T)
                } else if let guSh = s.guSh32, let actSh = s.actSh32 {
                    if !fuseSh {
                        try QWGemm.encodeSiluMulF32(enc, hal: hal, gu: guSh, act: actSh, rows: T,
                                                    intermediate: ISh, swigluClamp: w.sharedSwigluClamp)
                    }
                    try QWGemm.encodeHQ4F32V4(enc, hal: hal, weight: shDown, x: actSh, y: s.shared, rows: T)
                }
            }
            if let nextEnc = profileStage?("moe.experts_gate_up") { enc = nextEnc }
            try QWGemm.encodeMoEGateUpF32R1(enc, hal: hal, x: xRot, bank: w.gateUp, ids: s.idx,
                                            out: fuseRoutedGU ? s.act32 : s.gu32,
                                            rowsPerExpert: 2 * I, routes: K, tokens: T,
                                            fuseSiluFwht: fuseRoutedGU,
                                            coldBank: w.coldGateUp,
                                            reuseExpertsAcrossTokens: w.reuseExpertsAcrossTokens,
                                            swigluClamp: w.routedSwigluClamp)
            if !fuseRoutedGU {
                try QWGemm.encodeSiluMulF32(enc, hal: hal, gu: s.gu32, act: s.act32, rows: T * K,
                                            intermediate: I, swigluClamp: w.routedSwigluClamp)
            }
            if let nextEnc = profileStage?("moe.experts_down") { enc = nextEnc }
            try QWGemm.encodeMoEDownSharedCombineF32(enc, hal: hal, x: s.act32, bank: w.down,
                                                     rowsPerRoute: H, routes: K, weights: s.wts,
                                                     shared: s.shared, gate: s.gate, out: out32,
                                                     ids: s.idx, tokens: T, out16: out16,
                                                     coldBank: w.coldDown,
                                                     reuseExpertsAcrossTokens: w.reuseExpertsAcrossTokens)
        }
    }
}
