// DriverWeightLoading+HCAMoE.swift — HCA, Decoupled-RoPE MLA-DSA, Latent-MoE,
// mHC exit-head, MultiMax, and shared-last-layer ECHO weight loaders.

import Foundation
import Metal
import QWAlloc
import QWAttn
import QWCore
import QWGEMM
import QWHAL
import QWMLP
import QWQuant

extension QWEngineDriver.WeightLoader {

    static func fwht32OrthoInPlace(_ values: inout [Float]) {
        let scale: Float = 1.0 / sqrt(32.0)
        var base = 0
        while base + 32 <= values.count {
            var h = 1
            while h < 32 {
                var i = 0
                while i < 32 {
                    for j in i..<(i + h) {
                        let a = values[base + j]
                        let b = values[base + j + h]
                        values[base + j] = a + b
                        values[base + j + h] = a - b
                    }
                    i += h * 2
                }
                h *= 2
            }
            for k in 0..<32 { values[base + k] *= scale }
            base += 32
        }
    }

    mutating func loadMoERouterAndScale(
        _ blk: String, experts: Int, coldE: Int
    ) throws -> (router: QWTensorValue, bias: QWTensorValue?, scaleParam: QWTensorValue?, perm: [Int]?) {
        let H = geometry.hidden
        let split = geometry.moeSplitFeatureRouting
        let routerRows = split ? (2 * experts) : experts
        let hasMlpExperts = hasTensor(blk + "mlp.experts.\(experts - 1).gate_proj.weight")
        let hasBsExperts = hasTensor(blk + "block_sparse_moe.experts.\(experts - 1).w1.weight")
        let canPermute = coldE > 0 && geometry.hasBiasCorrectedRouter && (hasMlpExperts || hasBsExperts) && (geometry.moeRoutingConfig?.nGroup ?? 1) <= 1

        func readRouterFloats() throws -> [Float] {
            if hasTensor(blk + "mlp.gate.weight") {
                return try rawFloats(blk + "mlp.gate.weight", routerRows * H)
            }
            if split {
                let g0 = try rawFloats(blk + "block_sparse_moe.gate.weight", experts * H)
                let g1 = try rawFloats(blk + "block_sparse_moe.gate.weight_1", experts * H)
                return g0 + g1
            }
            return try rawFloats(blk + "block_sparse_moe.gate.weight", experts * H)
        }
        let biasName = hasTensor(blk + "mlp.gate.e_score_correction_bias")
            ? (blk + "mlp.gate.e_score_correction_bias")
            : (hasTensor(blk + "mlp.gate.expert_bias")
                ? (blk + "mlp.gate.expert_bias")
                : (hasTensor(blk + "block_sparse_moe.e_score_correction_bias")
                    ? (blk + "block_sparse_moe.e_score_correction_bias")
                    : (blk + "block_sparse_moe.expert_bias")))
        let scaleName = hasTensor(blk + "mlp.routed_scaling_factor_param")
            ? (blk + "mlp.routed_scaling_factor_param")
            : (blk + "block_sparse_moe.gate.routed_scaling_factor_param")

        if canPermute {
            let rawR = try readRouterFloats()
            let rawB = try rawFloats(biasName, experts)
            let rawS = geometry.routedScalingFactorLearnable ? try rawFloats(scaleName, experts) : nil
            let rawLn2 = (try? rawFloats(blk + "post_attention_layernorm.weight", H))
                ?? [Float](repeating: 1.0, count: H)
            var wNorm = [Float](repeating: 0, count: experts)
            for e in 0..<experts {
                var sumSq: Float = 0
                for c in 0..<H {
                    let v0 = rawR[e * H + c] * rawLn2[c]
                    sumSq += v0 * v0
                    if split {
                        let v1 = rawR[(experts + e) * H + c] * rawLn2[c]
                        sumSq += v1 * v1
                    }
                }
                wNorm[e] = sqrt(sumSq)
            }
            let perm = (0..<experts).sorted { a, b in
                if rawB[a] != rawB[b] { return rawB[a] > rawB[b] }
                if wNorm[a] != wNorm[b] { return wNorm[a] > wNorm[b] }
                return a < b
            }
            var permR = [Float](repeating: 0, count: routerRows * H)
            var permB = [Float](repeating: 0, count: experts)
            for dstIdx in 0..<experts {
                let srcIdx = perm[dstIdx]
                permB[dstIdx] = rawB[srcIdx]
                for c in 0..<H {
                    permR[dstIdx * H + c] = rawR[srcIdx * H + c]
                    if split {
                        permR[(experts + dstIdx) * H + c] = rawR[(experts + srcIdx) * H + c]
                    }
                }
            }
            let rT = try allocateF32(permR)
            let bT = try allocateF32(permB)
            let sT: QWTensorValue?
            if let rawS {
                let permS = perm.map { rawS[$0] }
                sT = try allocateF32(permS)
            } else {
                sT = nil
            }
            return (rT, bT, sT, perm)
        }

        let rT: QWTensorValue
        if hasTensor(blk + "mlp.gate.weight") {
            if plan.moeRouterIsBF16 && coldE == 0 {
                rT = try bf16wRaw(blk + "mlp.gate.weight", routerRows * H)
            } else {
                rT = try f32w(blk + "mlp.gate.weight", routerRows * H)
            }
        } else {
            rT = try allocateF32(try readRouterFloats())
        }
        let bT = geometry.hasBiasCorrectedRouter ? try f32w(biasName, experts) : nil
        let sT = geometry.routedScalingFactorLearnable ? try f32w(scaleName, experts) : nil
        return (rT, bT, sT, nil)
    }

    mutating func loadLatentBottleneckIfNeeded(_ blk: String) throws -> QWMoEOperator.LatentBottleneck? {
        guard let latL = geometry.moeLatentSize, latL > 0 else { return nil }
        let H = geometry.hidden
        let fc1Name = hasTensor(blk + "mlp.fc1.weight")
            ? (blk + "mlp.fc1.weight")
            : (blk + "block_sparse_moe.fc1_latent_proj.weight")
        let lnName = hasTensor(blk + "mlp.latent_norm.weight")
            ? (blk + "mlp.latent_norm.weight")
            : (blk + "block_sparse_moe.latent_norm.weight")
        let fc2Name = hasTensor(blk + "mlp.fc2.weight")
            ? (blk + "mlp.fc2.weight")
            : (blk + "block_sparse_moe.fc2_latent_proj.weight")
        let fc1 = try q4(fc1Name, n: latL, k: H)
        let normGamma = try f16wNorm(lnName, latL)
        let fc2 = try q4(fc2Name, n: H, k: latL)
        return QWMoEOperator.LatentBottleneck(
            fc1: fc1, normGamma: normGamma, normEps: geometry.rmsEps,
            fc2: fc2, latentDim: latL
        )
    }

    private mutating func allocateF32(_ src: [Float]) throws -> QWTensorValue {
        let t = try allocateWeight(QWTensorShape([src.count], .f32))
        let dst = t.region.buffer.contents().advanced(by: t.region.base + t.offset)
        src.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            if let base = raw.baseAddress, !raw.isEmpty {
                dst.copyMemory(from: base, byteCount: raw.count)
            }
        }
        return t
    }

    mutating func allocateF16(_ src: [Float]) throws -> QWTensorValue {
        let t = try allocateWeight(QWTensorShape([src.count], .f16))
        let bits = src.map { QWQuant.f32tof16Bits($0) }
        let dst = t.region.buffer.contents().advanced(by: t.region.base + t.offset)
        bits.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            if let base = raw.baseAddress, !raw.isEmpty {
                dst.copyMemory(from: base, byteCount: raw.count)
            }
        }
        return t
    }

    private mutating func loadVHAMix(blk: String, qHeads: Int) throws -> QWTensorValue {
        if hasTensor(blk + "self_attn.vha_mix.weight") {
            return try f16w(blk + "self_attn.vha_mix.weight", qHeads * qHeads)
        }
        let uName = blk + "self_attn.vha_postmix_U"
        let vName = blk + "self_attn.vha_postmix_V"
        let rank: Int
        if let pk, let e = try? pk.entry(resolveName(uName)), e.shape.count == 2 {
            rank = e.shape[1]
        } else if let st, let info = st.tensor(resolveName(uName)), info.shape.count == 2 {
            rank = info.shape[1]
        } else {
            rank = 16
        }
        let u = try rawFloats(uName, qHeads * rank)
        let v = try rawFloats(vName, qHeads * rank)
        var mix = [Float](repeating: 0, count: qHeads * qHeads)
        for i in 0..<qHeads {
            mix[i * qHeads + i] = 1.0
            for j in 0..<qHeads {
                var acc: Float = 0
                for r in 0..<rank { acc += v[i * rank + r] * u[j * rank + r] }
                mix[i * qHeads + j] += acc
            }
        }
        return try allocateF16(mix)
    }

    mutating func loadMHCOrContractedHC(
        _ blk: String, mhc: QWModelDescriptor.MHCConfig
    ) throws -> QWEngineDriver.MHCLayerWeights {
        if hasTensor(blk + "mhc_attn.weight") || hasTensor(blk + "hc_attn_fn") {
            return try loadMHC(blk, mhc: mhc)
        }
        func loadHCSub(_ sub: String) throws -> (w: QWTensorValue, b: QWTensorValue) {
            let c = mhc.streams
            let routerRows = 2 * c + (mhc.fixedIdentity ? 0 : c * c)
            let width = c * geometry.hidden
            let pfx = blk + sub + "."
            let rawMap = try rawFloats(pfx + "mapping_proj.weight", width * routerRows)
            let needsTranspose = tensorShape(pfx + "mapping_proj.weight") == [width, routerRows]
            let aPre = (try? rawFloats(pfx + "alpha_pre", 1).first) ?? 1.0
            let aPost = (try? rawFloats(pfx + "alpha_post", 1).first) ?? 1.0
            let aRes = (try? rawFloats(pfx + "alpha_res", 1).first) ?? 1.0
            var wOut = [Float](repeating: 0, count: routerRows * width)
            for r in 0..<routerRows {
                let s: Float = (r < c) ? aPre : ((r < 2 * c) ? aPost : aRes)
                for col in 0..<width {
                    let src = needsTranspose ? rawMap[col * routerRows + r] : rawMap[r * width + col]
                    wOut[r * width + col] = src * s
                }
            }
            let wT = try allocateF32(wOut)
            let bT = try f32w(pfx + "bias", routerRows)
            return (wT, bT)
        }
        let attnSub = hasTensor(blk + "self_attention_hyper_connection.mapping_proj.weight")
            ? "self_attention_hyper_connection" : "self_attn_hyper_connection"
        let attn = try loadHCSub(attnSub)
        let ffn = try loadHCSub("mlp_hyper_connection")
        return QWEngineDriver.MHCLayerWeights(attnW: attn.w, attnB: attn.b, ffnW: ffn.w, ffnB: ffn.b)
    }

    mutating func loadHCAAttnLayer(
        blk: String, layer: Int,
        ln1: QWTensorValue, ln2: QWTensorValue,
        gu: QWMLPWeight, down: QWWeightView,
        moe: QWEngineDriver.MoEW?,
        geometry: QWModelGeometry
    ) throws -> QWEngineDriver.LayerW {
        guard let hca = geometry.hcaConfig else {
            throw QWEngineDriverError.badConfig("missing hcaConfig on compressed layer \(layer)")
        }
        let h = geometry.hidden
        let qHeads = hca.qHeads
        let headDim = hca.headDim
        let qRank = hca.qLoraRank
        let oGroups = max(1, hca.oGroups)
        let oLoraRank = hca.oLoraRank
        let groupIn = (qHeads / oGroups) * headDim
        let compRatio = max(1, hca.defaultCompressRatio)

        let wFusedIn: QWWeightView
        if hasTensor(blk + "self_attn.fused_in_proj.weight") {
            wFusedIn = try q4(blk + "self_attn.fused_in_proj.weight", n: qRank + 3 * headDim, k: h)
        } else {
            let (big, _) = try concatQ4([
                (blk + "self_attn.linear_q_down_proj.weight", qRank),
                (blk + "self_attn.linear_kv_proj.weight", headDim),
                (blk + "self_attn.core_attention.compressor.linear_wkv.weight", headDim),
                (blk + "self_attn.core_attention.compressor.linear_wgate.weight", headDim)
            ], k: h)
            wFusedIn = big
        }
        let qNormW = try f16w(
            hasTensor(blk + "self_attn.q_norm.weight")
                ? (blk + "self_attn.q_norm.weight")
                : (blk + "self_attn.q_layernorm.weight"),
            qRank)
        let kvNormW = try f16w(
            hasTensor(blk + "self_attn.kv_norm.weight")
                ? (blk + "self_attn.kv_norm.weight")
                : (blk + "self_attn.kv_layernorm.weight"),
            headDim)
        let compNormW = try f16w(
            hasTensor(blk + "self_attn.comp_norm.weight")
                ? (blk + "self_attn.comp_norm.weight")
                : (blk + "self_attn.core_attention.compressor.norm.weight"),
            headDim)
        let ape = try f32w(
            hasTensor(blk + "self_attn.compressor.ape")
                ? (blk + "self_attn.compressor.ape")
                : (blk + "self_attn.core_attention.compressor.ape"),
            compRatio * headDim)

        let wQUpGate: QWWeightView
        if hasTensor(blk + "self_attn.q_up_gate_proj.weight") {
            wQUpGate = try q4(blk + "self_attn.q_up_gate_proj.weight",
                              n: qHeads * headDim + oGroups * oLoraRank, k: qRank)
        } else {
            let (big, _) = try concatQ4([
                (blk + "self_attn.linear_q_up_proj.weight", qHeads * headDim),
                (blk + "self_attn.gate_proj.weight", oGroups * oLoraRank)
            ], k: qRank)
            wQUpGate = big
        }
        let woA = try q4(
            hasTensor(blk + "self_attn.wo_a.weight")
                ? (blk + "self_attn.wo_a.weight")
                : (blk + "self_attn.linear_o_group_proj"),
            n: oGroups * oLoraRank, k: groupIn)
        let woB = try q4(blk + "self_attn.o_proj.weight", n: h, k: oGroups * oLoraRank)
        let vhaMix = hca.useVHA ? try loadVHAMix(blk: blk, qHeads: qHeads) : nil
        let sinkName: String
        if hasTensor(blk + "self_attn.attn_sink") {
            sinkName = blk + "self_attn.attn_sink"
        } else if hasTensor(blk + "self_attn.core_attention.attn_sink") {
            sinkName = blk + "self_attn.core_attention.attn_sink"
        } else {
            sinkName = blk + "self_attn.core_attention.softmax_offset"
        }
        let attnSink = hca.hasAttnSinkBias ? try f32w(sinkName, qHeads) : nil

        let lsaPair = hcaLsaByLayer[layer]
        let hcaW = QWHCAAttnOperator.LayerWeights(
            wFusedIn: wFusedIn, qNormW: qNormW, kvNormW: kvNormW, compNormW: compNormW,
            ape: ape, wQUpGate: wQUpGate, woA: woA, woARawF16: nil, woB: woB,
            vhaMix: vhaMix, attnSink: attnSink,
            lsaWQIdx: lsaPair?.wq, lsaWKIdx: lsaPair?.wk
        )
        let mhcW: QWEngineDriver.MHCLayerWeights?
        if let mhc = geometry.mhcConfig, layer >= mhc.start, layer < mhc.end {
            mhcW = try loadMHCOrContractedHC(blk, mhc: mhc)
        } else {
            mhcW = nil
        }
        return QWEngineDriver.LayerW(
            ln1: ln1, ln2: ln2, gu: gu, down: down,
            kind: moe == nil ? .hca : .hcaMoe,
            gdn: nil, attn: nil, moe: moe, mhc: mhcW, hca: hcaW
        )
    }

    mutating func loadDecoupledMLAAttnWeights(
        blk: String,
        geometry: QWModelGeometry,
        denseOnly: Bool = false
    ) throws -> QWEngineDriver.AttnW {
        let mla = geometry.mlaConfig
        let hmla = geometry.hybridMLAConfig
        guard mla != nil || hmla != nil else {
            throw QWEngineDriverError.badConfig("missing mlaConfig on decoupled MLA layer")
        }
        let h = geometry.hidden
        let qHeads = mla?.qHeads ?? hmla?.qHeads ?? geometry.qHeads
        let qRank = mla?.qLoraRank ?? hmla?.qLoraRank ?? 0
        let kvRank = mla?.kvLoraRank ?? hmla?.kvLoraRank ?? geometry.kvLoraRank
        let nopeDim = mla?.qkNopeHeadDim ?? hmla?.qkNopeHeadDim ?? geometry.qkNopeHeadDim
        let ropeDim = mla?.qkRopeHeadDim ?? hmla?.qkRopeHeadDim ?? geometry.qkRopeHeadDim
        let vDim = mla?.vHeadDim ?? hmla?.vHeadDim ?? geometry.vHeadDim
        let isDirectQ = mla?.isDirectQ ?? (qRank <= 0)
        let hasIndexer = !denseOnly && (mla?.hasIndexer ?? (geometry.indexerHeads > 0))
        let outputGateMode = mla?.outputGateMode
            ?? (geometry.attnOutputGate ? .elementWiseFromQLora : .none)
        let hasInlineOGate = outputGateMode == .elementWiseFromQLora
        let hasOGate = outputGateMode != .none
        let idxH = hasIndexer ? max(1, geometry.indexerHeads) : 0
        let idxD = hasIndexer ? max(32, geometry.indexerHeadDim) : 0
        let qCh = qHeads * (nopeDim + ropeDim + (hasInlineOGate ? vDim : 0))
        let useVHA = mla?.useVHA ?? hmla?.useVHA ?? false
        let hasAttnSink = mla?.hasAttnSinkBias ?? hmla?.hasAttnSinkBias ?? false

        let ap = (hasTensor(blk + "self_attn.kv_a_proj_with_mqa.weight")
            || hasTensor(blk + "self_attn.q_a_proj.weight"))
            ? (blk + "self_attn.")
            : (blk + "attention.")
        let idxPfx = hasTensor(ap + "indexer.wk.weight")
            ? (ap + "indexer.")
            : (ap + "core_attention.indexer.")
        let fuseOGateInQKV = outputGateMode == .elementWiseFromHidden && hasTensor(ap + "gate_proj.weight")
        let hasExternalOGate = outputGateMode == .headWiseFromHidden
            || (outputGateMode == .elementWiseFromHidden && !fuseOGateInQKV)

        var qkvSpecs: [(name: String, n: Int)] = []
        if !isDirectQ {
            qkvSpecs.append((ap + "q_a_proj.weight", qRank))
        }
        qkvSpecs.append((ap + "kv_a_proj_with_mqa.weight", kvRank + ropeDim))
        if hasIndexer {
            qkvSpecs.append((idxPfx + "wk.weight", idxD))
        }
        if fuseOGateInQKV {
            qkvSpecs.append((ap + "gate_proj.weight", qHeads * vDim))
        }
        let (qkvProj, qkvSub) = try concatQ4(qkvSpecs, k: h)

        let kvANormW: QWTensorValue
        let qANormW: QWTensorValue
        let qAGatedNorm: QWMLAAttnOperator.GatedLatentNorm?
        let kvAGatedNorm: QWMLAAttnOperator.GatedLatentNorm?
        if let gn = geometry.gatedNormConfig {
            if !isDirectQ {
                let qG = try loadGatedLatentNorm(ap + "q_a_layernorm.", dim: qRank, config: gn)
                let kvG = try loadGatedLatentNorm(ap + "kv_a_layernorm.", dim: kvRank, config: gn)
                qAGatedNorm = qG
                kvAGatedNorm = kvG
                qANormW = qG.weight!
                kvANormW = kvG.weight!
            } else {
                let kvG = try loadGatedLatentNorm(ap + "kv_a_layernorm.", dim: kvRank, config: gn)
                qAGatedNorm = nil
                kvAGatedNorm = kvG
                kvANormW = kvG.weight!
                qANormW = kvANormW
            }
        } else if !isDirectQ {
            qANormW = try f16w(ap + "q_a_layernorm.weight", qRank)
            kvANormW = try f16w(ap + "kv_a_layernorm.weight", kvRank)
            qAGatedNorm = nil
            kvAGatedNorm = nil
        } else {
            kvANormW = try f16w(ap + "kv_a_layernorm.weight", kvRank)
            qANormW = kvANormW
            qAGatedNorm = nil
            kvAGatedNorm = nil
        }

        let kNormW: QWTensorValue
        let kNormB: QWTensorValue
        let wGateWeights: QWTensorValue
        if hasIndexer {
            kNormW = try f16w(idxPfx + "k_norm.weight", idxD)
            kNormB = (mla?.indexerKNormIsRMS == true) ? kNormW : try f16w(idxPfx + "k_norm.bias", idxD)
            var wProjFloats = try rawFloats(idxPfx + "weights_proj.weight", idxH * h)
            if hl.weightBasis.rotatesActivations {
                Self.fwht32OrthoInPlace(&wProjFloats)
            }
            wGateWeights = try allocateF16(wProjFloats)
        } else {
            kNormW = kvANormW
            kNormB = kvANormW
            wGateWeights = kvANormW
        }

        let qBProj: QWWeightView
        if isDirectQ {
            let qDirectName = hasTensor(ap + "q_proj.weight") ? (ap + "q_proj.weight") : (ap + "q_b_proj.weight")
            if hasIndexer {
                let (big, _) = try concatQ4([
                    (qDirectName, qCh),
                    (idxPfx + "wq_b.weight", idxH * idxD)
                ], k: h)
                qBProj = big
            } else {
                qBProj = try q4(qDirectName, n: qCh, k: h)
            }
        } else if hasIndexer && hasTensor(ap + "q_b_gate_idx_proj.weight") {
            qBProj = try q4(ap + "q_b_gate_idx_proj.weight", n: qCh + idxH * idxD, k: qRank)
        } else if !hasIndexer && !hasInlineOGate && hasTensor(ap + "q_b_proj.weight") {
            qBProj = try q4(ap + "q_b_proj.weight", n: qCh, k: qRank)
        } else if hasTensor(ap + "q_b_nope_proj.weight") {
            var qbSpecs: [(name: String, n: Int)] = [
                (ap + "q_b_nope_proj.weight", qHeads * nopeDim),
                (ap + "q_b_rope_proj.weight", qHeads * ropeDim)
            ]
            if hasInlineOGate {
                qbSpecs.append((ap + "gate_proj.weight", qHeads * vDim))
            }
            if hasIndexer {
                qbSpecs.append((idxPfx + "wq_b.weight", idxH * idxD))
            }
            let (big, _) = try concatQ4(qbSpecs, k: qRank)
            qBProj = big
        } else {
            var qbSpecs: [(name: String, n: Int)] = [
                (ap + "q_b_proj.weight", qHeads * (nopeDim + ropeDim))
            ]
            if hasInlineOGate {
                qbSpecs.append((ap + "gate_proj.weight", qHeads * vDim))
            }
            if hasIndexer {
                qbSpecs.append((idxPfx + "wq_b.weight", idxH * idxD))
            }
            let (big, _) = try concatQ4(qbSpecs, k: qRank)
            qBProj = big
        }
        let kvBProj = try q4(ap + "kv_b_proj.weight", n: qHeads * (nopeDim + vDim), k: kvRank)
        let oName = hasTensor(ap + "o_proj.weight") ? (ap + "o_proj.weight") : (ap + "dense.weight")
        let o = try q4(oName, n: h, k: qHeads * vDim)

        let oGateF16: QWTensorValue?
        if hasExternalOGate {
            let gateCols = outputGateMode == .headWiseFromHidden ? qHeads : (qHeads * vDim)
            var gateFloats = try rawFloats(ap + "g_proj.weight", gateCols * h)
            if hl.weightBasis.rotatesActivations {
                Self.fwht32OrthoInPlace(&gateFloats)
            }
            oGateF16 = try allocateF16(gateFloats)
        } else {
            oGateF16 = nil
        }

        let vhaMix = useVHA ? try loadVHAMix(blk: blk, qHeads: qHeads) : nil
        let attnSink = hasAttnSink ? try f32w(
            hasTensor(ap + "attn_sink")
                ? (ap + "attn_sink")
                : (ap + "core_attention.softmax_offset"),
            qHeads) : nil
        let flashWeights = hasIndexer
            ? try loadFlashIndexerWeights(blk: blk, ap: ap, idxPfx: idxPfx, idxH: idxH, hidden: h)
            : nil

        let mlaW = QWMLAAttnOperator.LayerWeights(
            qANormW: qANormW, kvANormW: kvANormW, kNormW: kNormW, kNormB: kNormB,
            wGateWeights: wGateWeights, ape: nil, qBProj: qBProj, kvBProj: kvBProj,
            vhaMix: vhaMix, attnSink: attnSink, hasOGate: hasOGate, oGateF16: oGateF16,
            flashGateW: flashWeights?.gate, flashBias: flashWeights?.bias,
            qAGatedNorm: qAGatedNorm, kvAGatedNorm: kvAGatedNorm,
            alphaQ: mla?.alphaQ, alphaKV: mla?.alphaKV,
            indexerKNormIsRMS: mla?.indexerKNormIsRMS,
            softmaxScaleOverride: mla?.softmaxScale,
            denseOnly: denseOnly, oGateInQKV: fuseOGateInQKV
        )
        return QWEngineDriver.AttnW(
            qkvProj: qkvProj, q: qkvSub[0], k: qkvSub[min(1, qkvSub.count - 1)],
            v: qkvSub[qkvSub.count - 1],
            o: o, wq: nil, wkTiled: nil, mla: mlaW
        )
    }

    private func loadFlashIndexerWeights(
        blk: String, ap: String, idxPfx: String, idxH: Int, hidden: Int
    ) throws -> (gate: QWTensorValue, bias: QWTensorValue)? {
        guard let containerURL = pk?.url ?? st?.url else { return nil }
        let fiURL = containerURL.deletingLastPathComponent().appendingPathComponent("flashindexer.qwp")
        guard FileManager.default.fileExists(atPath: fiURL.path) else { return nil }
        let fiPack = try QWQuant.Pack(url: fiURL, descriptor: geometry.descriptor, skipIdentityCheck: true)

        var prefixes = [idxPfx, ap + "indexer.", ap + "core_attention.indexer."]
        if let range = blk.range(of: "layers.") {
            let suffix = String(blk[range.lowerBound...])
            prefixes.append("model.\(suffix)self_attn.indexer.")
            prefixes.append("model.\(suffix)self_attn.core_attention.indexer.")
        }
        var matched: String?
        for prefix in prefixes where (try? fiPack.entry(prefix + "linear_gate.weight")) != nil
            && (try? fiPack.entry(prefix + "flashindexer_bias")) != nil {
            matched = prefix
            break
        }
        guard let pfx = matched else { return nil }

        let gateName = pfx + "linear_gate.weight"
        let biasName = pfx + "flashindexer_bias"
        let readPackFloats: (String, Int) throws -> [Float] = { name, expected in
            let entry = try fiPack.entry(name)
            let out = entry.rawOff != nil ? try fiPack.rawBf16(name) : try fiPack.dequantQ4g(name)
            guard out.count == expected else {
                throw QWEngineDriverError.weightMissing(name, "shape \(out.count) ≠ \(expected)")
            }
            return out
        }

        var gateFloats = try readPackFloats(gateName, idxH * hidden)
        if hl.weightBasis.rotatesActivations {
            Self.fwht32OrthoInPlace(&gateFloats)
        }
        let biasFloats = try readPackFloats(biasName, idxH)

        let gateBytes = max(256, (gateFloats.count * 2 + 255) & ~255)
        guard let gateBuf = hl.device.makeBuffer(length: gateBytes, options: .storageModeShared) else {
            throw QWAllocError.outOfSpace(region: "flashindexer-gate", asked: gateBytes, free: 0)
        }
        gateBuf.label = "flashindexer-gate-\(blk)"
        let gatePtr = gateBuf.contents().bindMemory(to: Float16.self, capacity: gateFloats.count)
        for i in 0..<gateFloats.count {
            gatePtr[i] = Float16(gateFloats[i])
        }
        let gateTensor = QWTensorValue(
            region: QWMRegion(buffer: gateBuf, base: 0, length: gateBytes),
            offset: 0,
            shape: QWTensorShape([idxH, hidden], .f16)
        )

        let biasBytes = max(256, (biasFloats.count * 4 + 255) & ~255)
        guard let biasBuf = hl.device.makeBuffer(length: biasBytes, options: .storageModeShared) else {
            throw QWAllocError.outOfSpace(region: "flashindexer-bias", asked: biasBytes, free: 0)
        }
        biasBuf.label = "flashindexer-bias-\(blk)"
        let biasDst = biasBuf.contents()
        withExtendedLifetime(biasBuf) {
            biasFloats.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                if let src = raw.baseAddress {
                    biasDst.copyMemory(from: src, byteCount: biasFloats.count * 4)
                }
            }
        }
        let biasTensor = QWTensorValue(
            region: QWMRegion(buffer: biasBuf, base: 0, length: biasBytes),
            offset: 0,
            shape: QWTensorShape([idxH], .f32)
        )
        return (gateTensor, biasTensor)
    }

    mutating func loadDecoupledMLAAttnLayer(
        blk: String, layer: Int,
        ln1: QWTensorValue, ln2: QWTensorValue,
        gu: QWMLPWeight, down: QWWeightView,
        moe: QWEngineDriver.MoEW?,
        geometry: QWModelGeometry,
        gatedNorms: QWEngineDriver.GatedLayerNorms? = nil
    ) throws -> QWEngineDriver.LayerW {
        let denseOnly = geometry.mlaConfig?.denseAttentionLayers.contains(layer) == true
        let attnW = try loadDecoupledMLAAttnWeights(blk: blk, geometry: geometry, denseOnly: denseOnly)
        let mhcW: QWEngineDriver.MHCLayerWeights?
        if let mhc = geometry.mhcConfig, layer >= mhc.start, layer < mhc.end {
            mhcW = try loadMHCOrContractedHC(blk, mhc: mhc)
        } else {
            mhcW = nil
        }
        return QWEngineDriver.LayerW(
            ln1: ln1, ln2: ln2, gu: gu, down: down,
            kind: moe == nil ? .fullAttention : .moe,
            gdn: nil, attn: attnW, moe: moe, mhc: mhcW,
            gatedNorms: gatedNorms
        )
    }

    mutating func loadExitHeadPair(prefix: String, streams: Int) throws -> (w: QWTensorValue, b: QWTensorValue) {
        let width = streams * geometry.hidden
        if hasTensor(prefix + ".weight") {
            let wT = try f32wOrDequant(prefix + ".weight", streams * width)
            let bT = try f32wOrDequant(prefix + ".bias", streams)
            return (wT, bT)
        }
        let fnName = hasTensor(prefix + ".hc_head_fn") ? (prefix + ".hc_head_fn") : (prefix + "hc_head_fn")
        let scName = hasTensor(prefix + ".hc_head_scale") ? (prefix + ".hc_head_scale") : (prefix + "hc_head_scale")
        let bsName = hasTensor(prefix + ".hc_head_base") ? (prefix + ".hc_head_base") : (prefix + "hc_head_base")
        let rawFn = try rawFloats(fnName, width * streams)
        let needsTranspose = tensorShape(fnName) == [width, streams]
        let scVals: [Float]
        if let scShape = tensorShape(scName), scShape.reduce(1, *) == streams {
            scVals = try rawFloats(scName, streams)
        } else {
            let s0 = (try? rawFloats(scName, 1).first) ?? 1.0
            scVals = [Float](repeating: s0, count: streams)
        }
        var wOut = [Float](repeating: 0, count: streams * width)
        for s in 0..<streams {
            let sc = scVals[s]
            for col in 0..<width {
                let src = needsTranspose ? rawFn[col * streams + s] : rawFn[s * width + col]
                wOut[s * width + col] = src * sc
            }
        }
        let wT = try allocateF32(wOut)
        let bT = try f32w(bsName, streams)
        return (wT, bT)
    }

    mutating func loadMHCExitHeadIfNeeded() throws -> QWEngineDriver.MHCExitHeadWeights? {
        guard let mhc = geometry.mhcConfig, mhc.hasExitHead else { return nil }
        let prefix = hasTensor(QWEngineDriver.P + "hc_head.hc_head_fn")
            ? (QWEngineDriver.P + "hc_head")
            : (QWEngineDriver.P + "mhc_contract")
        let pair = try loadExitHeadPair(prefix: prefix, streams: mhc.streams)
        return QWEngineDriver.MHCExitHeadWeights(headW: pair.w, headB: pair.b)
    }

    mutating func loadMultimaxIfNeeded() throws -> QWTensorValue? {
        guard geometry.multimaxHead else { return nil }
        if hasTensor("lm_head.multimax") {
            return try f32w("lm_head.multimax", 8)
        }
        let ranges = try rawFloats(QWEngineDriver.P + "lm_head.multimax_ranges", 4)
        let ts = try rawFloats(QWEngineDriver.P + "lm_head.multimax_ts", 4)
        return try allocateF32(ranges + ts)
    }

    mutating func loadEchoDraftWeightsIfNeeded(drafts: Int) throws -> QWEngineDriver.EchoDraftWeights? {
        guard drafts > 0 else { return nil }
        if geometry.hasDedicatedMHCMTP {
            return try loadDedicatedEchoDraftWeights()
        }
        guard geometry.mtpSharedLastLayer else { return nil }
        let h = geometry.hidden
        let streams = geometry.mhcConfig?.streams ?? 4
        let blk = QWEngineDriver.P + "layers.\(geometry.numLayers)."
        let eNorm = try f16wNorm(blk + "enorm.weight", h)
        let hNorm = try f16wNorm(blk + "hnorm.weight", h)
        let eProj = try q4(blk + "e_proj.weight", n: h, k: h)
        let hProj = try q4(blk + "h_proj.weight", n: h, k: h)
        let hcPair: (w: QWTensorValue, b: QWTensorValue)
        if hasTensor(blk + "mhc_contract.weight") {
            hcPair = try loadExitHeadPair(prefix: blk + "mhc_contract", streams: streams)
        } else {
            hcPair = try loadExitHeadPair(prefix: blk, streams: streams)
        }
        let sharedHeadNorm = try f16wNorm(blk + "shared_head.norm.weight", h)
        let routerBias: QWTensorValue?
        if geometry.hasBiasCorrectedRouter && geometry.numExperts > 0 {
            let bName: String
            if hasTensor(blk + "mlp.gate.e_score_correction_bias") {
                bName = blk + "mlp.gate.e_score_correction_bias"
            } else if hasTensor(blk + "block_sparse_moe.e_score_correction_bias") {
                bName = blk + "block_sparse_moe.e_score_correction_bias"
            } else {
                let prevBlk = QWEngineDriver.P + "layers.\(max(0, geometry.numLayers - 1))."
                bName = hasTensor(prevBlk + "mlp.gate.e_score_correction_bias")
                    ? (prevBlk + "mlp.gate.e_score_correction_bias")
                    : (prevBlk + "block_sparse_moe.e_score_correction_bias")
            }
            routerBias = try f32w(bName, geometry.numExperts)
        } else {
            routerBias = nil
        }
        return QWEngineDriver.EchoDraftWeights(
            eNorm: eNorm, hNorm: hNorm, eProj: eProj, hProj: hProj,
            hcHeadW: hcPair.w, hcHeadB: hcPair.b,
            sharedHeadNorm: sharedHeadNorm, routerBias: routerBias
        )
    }
}
