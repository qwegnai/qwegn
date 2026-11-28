// QWHal+Constants.swift — MTLFunctionConstantValues builder for model geometry.
import Metal
import QWCore

extension QWHal {
    public static func makeFunctionConstants(
        geometry: QWModelGeometry,
        isAQ4: Bool = false,
        swigluClamp: QWSwiGLUClampSpec? = nil,
        weightBasis: QWWeightBasis = .rotated,
        fcDrafterRotated: Bool? = nil,
        fcHeadRotated: Bool? = nil,
        gdnNormSigmoid: Bool? = nil,
        gdnGateScale: Float? = nil
    ) -> MTLFunctionConstantValues {
        let fc = MTLFunctionConstantValues()
        func setU32(_ v: Int, at idx: Int) {
            var u = UInt32(v)
            fc.setConstantValue(&u, type: .uint, index: idx)
        }
        setU32(geometry.hidden, at: 0)
        setU32(geometry.vocab, at: 1)
        setU32(geometry.intermediate, at: 2)
        setU32(geometry.numLayers, at: 3)
        setU32(geometry.fullAttentionInterval, at: 4)
        setU32(geometry.qHeads, at: 5)
        setU32(geometry.kvHeads, at: 6)
        setU32(geometry.headDim, at: 7)
        setU32(geometry.qProjOut, at: 8)
        setU32(geometry.rotaryDims, at: 9)
        var eps = Float(geometry.rmsEps)
        fc.setConstantValue(&eps, type: .float, index: 10)
        setU32(geometry.maxPositionEmbeddings, at: 11)
        setU32(geometry.gdnQKHeads, at: 12)
        setU32(geometry.gdnVHeads, at: 13)
        setU32(geometry.gdnHeadDim, at: 14)
        setU32(geometry.gdnQKVChannels, at: 15)
        setU32(geometry.gdnZChannels, at: 16)
        setU32(geometry.gdnConvKernel, at: 17)
        setU32(geometry.qHeads / max(1, geometry.kvHeads), at: 18)
        setU32(geometry.gdnVHeads / max(1, geometry.gdnQKHeads), at: 19)
        let kvRow = geometry.kvHeads * geometry.headDim
        let q4GroupsPerHead = max(1, (geometry.headDim + 127) / 128)
        setU32(kvRow, at: 20)
        setU32(256 * kvRow, at: 21)
        setU32(kvRow / 2, at: 22)
        setU32(geometry.kvHeads * 2, at: 23)
        setU32(geometry.kvHeads * q4GroupsPerHead * 4, at: 24)
        // The attention output gate: a gated model packs Q‖gate in one q row.
        let standard = Self.standardConstants(geometry: geometry)
        var hasGate = standard.hasQGate
        fc.setConstantValue(&hasGate, type: .bool, index: 25)
        setU32(Int(standard.qStride), at: 26)
        var attnScale = standard.attnScale
        fc.setConstantValue(&attnScale, type: .float, index: 27)
        // MLX's gather kernel reads three function constants (200, 201, 202)
        // out of the vendored `quantized_nax.h`: each says that one extent is
        // a whole number of tiles. Only that kernel reads them.
        var alignM = false, alignN = true, alignK = true
        fc.setConstantValue(&alignM, type: .bool, index: 200)
        fc.setConstantValue(&alignN, type: .bool, index: 201)
        fc.setConstantValue(&alignK, type: .bool, index: 202)
        // Generalized architecture function constants (220..227):
        setU32(geometry.mhcConfig?.streams ?? 1, at: 220)
        setU32(geometry.mhcConfig?.iterations ?? 20, at: 221)
        var gdnVecDecay = geometry.gdnVectorDecay
        fc.setConstantValue(&gdnVecDecay, type: .bool, index: 222)
        var gdnGateMin = Float(geometry.gdnGateLowerBound)
        fc.setConstantValue(&gdnGateMin, type: .float, index: 223)
        var swigluLim = Float(swigluClamp?.limit ?? geometry.swigluLimit ?? 0.0)
        fc.setConstantValue(&swigluLim, type: .float, index: 224)
        var routedScale = geometry.descriptor.routedScalingFactorLearnable ? Float(1.0) : Float(geometry.routedScalingFactor)
        fc.setConstantValue(&routedScale, type: .float, index: 225)
        setU32(geometry.kvLoraRank, at: 226)
        var indexKPool = geometry.indexKPoolCompress
        fc.setConstantValue(&indexKPool, type: .bool, index: 227)
        var situBeta: Float = 0.0
        var situLinBeta: Float = 0.0
        if case let .situ(beta, linearBeta) = geometry.descriptor.activationKind {
            situBeta = beta
            situLinBeta = linearBeta
        }
        fc.setConstantValue(&situBeta, type: .float, index: 228)
        fc.setConstantValue(&situLinBeta, type: .float, index: 229)
        var mhcAdditiveEps = geometry.mhcConfig?.hasExitHead == true
        fc.setConstantValue(&mhcAdditiveEps, type: .bool, index: 230)
        var mhcHcEps = geometry.mhcConfig?.hcEps ?? (geometry.mhcConfig?.streams == 4 ? Float(1e-6) : Float(0.0))
        fc.setConstantValue(&mhcHcEps, type: .float, index: 231)
        var aq4 = isAQ4
        fc.setConstantValue(&aq4, type: .bool, index: 232)
        var postSilu = swigluClamp?.postSilu ?? (geometry.swigluClampConfig?.mode == .postSilu && geometry.swigluLimit != nil)
        fc.setConstantValue(&postSilu, type: .bool, index: 233)
        var normSig = gdnNormSigmoid ?? geometry.kdaConfig.map { $0.outputNormAct == .sigmoid } ?? geometry.gdnVectorDecay
        fc.setConstantValue(&normSig, type: .bool, index: 234)
        var gateScale = gdnGateScale ?? Float(1.0)
        fc.setConstantValue(&gateScale, type: .float, index: 235)
        // The activation basis is NOT a function constant. It is a build-time
        // property of the metallib — see `QWWeightBasis.metallibFileName` —
        // because the rotation is a chain of `simd_shuffle_xor` calls and a
        // runtime branch in front of them puts the shuffle in a divergent
        // region, which cost 14.5% of prefill throughput when it was tried.
        var drafterRotated = fcDrafterRotated ?? weightBasis.rotatesActivations
        fc.setConstantValue(&drafterRotated, type: .bool, index: 236)
        var headRotated = fcHeadRotated ?? weightBasis.rotatesActivations
        fc.setConstantValue(&headRotated, type: .bool, index: 237)
        return fc
    }

}
