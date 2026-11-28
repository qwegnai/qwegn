// DriverWeightLoading+Layers.swift — MoE and per-layer weight loaders on QWEngineDriver.WeightLoader.

import Foundation
import Darwin
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWNorm
import QWRope
import QWKV
import QWAttn
import QWGDN
import QWLoad
import QWMLP
import QWHead
import QWMTP
import QWQuant

extension QWEngineDriver.WeightLoader {
        typealias MoEW = QWEngineDriver.MoEW
        typealias LayerW = QWEngineDriver.LayerW
        typealias GdnW = QWEngineDriver.GdnW
        typealias AttnW = QWEngineDriver.AttnW
        typealias ColdExpertStreamer = QWEngineDriver.ColdExpertStreamer
        typealias MHCLayerWeights = QWEngineDriver.MHCLayerWeights

        /// A sparse MoE layer's weights: the router plane and the two banks.
        /// The router is the pack's RAW bf16 plane (the packer copies it
        /// verbatim — not quantised, not rotated), widened to f32; the banks
        /// are ONE contiguous view each, the experts stacked in order (C119).
        mutating func packedMoE(_ blk: String, experts: Int, topK: Int, layer: Int = 0) throws -> MoEW {
            let I = geometry.moeIntermediate
            let H = geometry.hidden
            let outDim = geometry.moeRoutedDim
            let resE = geometry.effectiveResidentExperts
            let coldE = max(0, experts - resE)
            let routed = try loadMoERouterAndScale(blk, experts: experts, coldE: coldE)
            let router = routed.router
            let routerBias = routed.bias
            let routedScaleParam = routed.scaleParam
            let expertPerm = routed.perm
            let gu: QWWeightView
            let down: QWWeightView
            if st != nil, hasTensor(blk + "mlp.switch_mlp.gate_proj.weight") {
                let guBlocks: QWTensorValue
                if format.isNativeAffine {
                    let guElems = (resE * 2 * I * format.rowBytes(k: outDim)) / 2
                    guBlocks = try allocateWeight(QWTensorShape([guElems], .f16))
                    try repackSafetensorsSwitchGateUpToAQ(
                        gateName: blk + "mlp.switch_mlp.gate_proj.weight",
                        upName: blk + "mlp.switch_mlp.up_proj.weight",
                        experts: resE, i: I, h: outDim,
                        into: guBlocks.region.buffer,
                        dstByteOffset: guBlocks.region.base + guBlocks.offset
                    )
                } else {
                    let guElems = resE * 2 * I * (outDim / 32) * 10
                    guBlocks = try allocateWeight(QWTensorShape([guElems], .f16))
                    try quantizeSafetensorsSwitchGateUp(
                        gateName: blk + "mlp.switch_mlp.gate_proj.weight",
                        upName: blk + "mlp.switch_mlp.up_proj.weight",
                        experts: resE, i: I, h: outDim,
                        into: guBlocks.region.buffer,
                        dstByteOffset: guBlocks.region.base + guBlocks.offset
                    )
                }
                let downBlocks = try planeView(
                    blk + "mlp.switch_mlp.down_proj.weight",
                    n: resE * outDim, k: I, key: \QWQTensorEntry.hq4Off, residentExperts: resE
                )
                guard let guView = QWWeightView(blocks: guBlocks, n: resE * 2 * I, k: outDim,
                                             bits: format.bits, groupSize: format.groupSize,
                                             isRotated: hl.weightBasis.rotatesActivations),
                      let downView = QWWeightView(blocks: downBlocks, n: resE * outDim, k: I,
                                                 bits: format.bits, groupSize: format.groupSize,
                                                 isRotated: hl.weightBasis.rotatesActivations) else {
                    throw QWEngineDriverError.weightMissing(blk + "mlp.switch_mlp", "moe bank view invalid")
                }
                gu = guView
                down = downView
            } else if pk != nil,
                      let fusedPair = ColdExpertStreamer.fusedExpertCandidates.first(where: {
                          hasTensor(blk + $0.gateUp) && hasTensor(blk + $0.down)
                      }) {
                let guName = blk + fusedPair.gateUp
                let dnName = blk + fusedPair.down
                let guBlocks = try planeView(
                    guName, n: resE * 2 * I, k: outDim,
                    key: \QWQTensorEntry.hq4Off, residentExperts: resE
                )
                let downBlocks = try planeView(
                    dnName, n: resE * outDim, k: I,
                    key: \QWQTensorEntry.hq4Off, residentExperts: resE
                )
                let guEntry = try? pk?.entry(resolveName(guName))
                let dnEntry = try? pk?.entry(resolveName(dnName))
                guard let guView = QWWeightView(blocks: guBlocks, n: resE * 2 * I, k: outDim,
                                                bits: guEntry?.hqBits ?? format.expertBits,
                                                groupSize: guEntry?.hqGroup ?? format.expertGroupSize,
                                                isRotated: hl.weightBasis.rotatesActivations,
                                                isIncumbentContainer: guEntry?.isIncumbentHQ4 ?? format.expertIsIncumbentContainer,
                                                isSHVQ: guEntry?.isSHVQ ?? format.expertIsSHVQ),
                      let downView = QWWeightView(blocks: downBlocks, n: resE * outDim, k: I,
                                                  bits: dnEntry?.hqBits ?? format.expertDownBits,
                                                  groupSize: dnEntry?.hqGroup ?? format.expertDownGroupSize,
                                                  isRotated: hl.weightBasis.rotatesActivations,
                                                  isIncumbentContainer: dnEntry?.isIncumbentHQ4 ?? format.expertDownIsIncumbentContainer,
                                                  isSHVQ: dnEntry?.isSHVQ ?? format.expertDownIsSHVQ) else {
                    throw QWEngineDriverError.weightMissing(guName, "moe bank view invalid")
                }
                gu = guView
                down = downView
            } else {
                let useBlockSparse = !hasTensor(blk + "mlp.experts.0.gate_proj.weight")
                    && hasTensor(blk + "block_sparse_moe.experts.0.w1.weight")
                var guSpecs: [(name: String, n: Int)] = []
                var downSpecs: [(name: String, n: Int)] = []
                for idx in 0..<resE {
                    let e = expertPerm?[idx] ?? idx
                    if useBlockSparse {
                        guSpecs.append((blk + "block_sparse_moe.experts.\(e).w1.weight", I))
                        guSpecs.append((blk + "block_sparse_moe.experts.\(e).w3.weight", I))
                        downSpecs.append((blk + "block_sparse_moe.experts.\(e).w2.weight", outDim))
                    } else {
                        guSpecs.append((blk + "mlp.experts.\(e).gate_proj.weight", I))
                        guSpecs.append((blk + "mlp.experts.\(e).up_proj.weight", I))
                        downSpecs.append((blk + "mlp.experts.\(e).down_proj.weight", outDim))
                    }
                }
                let (guBank, _) = try concatQ4(guSpecs, k: outDim)
                let (downBank, _) = try concatQ4(downSpecs, k: I)
                guard let guView = QWWeightView(blocks: guBank.blocks, n: resE * 2 * I, k: outDim,
                                             bits: format.expertBits, groupSize: format.expertGroupSize,
                                             isRotated: hl.weightBasis.rotatesActivations,
                                             isIncumbentContainer: format.expertIsIncumbentContainer,
                                             isSHVQ: format.expertIsSHVQ),
                      let downView = QWWeightView(blocks: downBank.blocks, n: resE * outDim, k: I,
                                                 bits: format.expertDownBits, groupSize: format.expertDownGroupSize,
                                                 isRotated: hl.weightBasis.rotatesActivations,
                                                 isIncumbentContainer: format.expertDownIsIncumbentContainer,
                                                 isSHVQ: format.expertDownIsSHVQ) else {
                    throw QWEngineDriverError.weightMissing(blk + "mlp.experts", "moe bank view invalid")
                }
                gu = guView
                down = downView
            }
            let coldBanks = coldE > 0 ? try ensureSharedColdBanks(coldE: coldE, i: I, h: outDim) : nil
            let coldStreamer: ColdExpertStreamer?
            if let cb = coldBanks, let io = self.io {
                let guPtr = cb.gu.blocks.region.buffer.contents().advanced(by: cb.gu.blocks.region.base + cb.gu.blocks.offset)
                let downPtr = cb.down.blocks.region.buffer.contents().advanced(by: cb.down.blocks.region.base + cb.down.blocks.offset)
                let probeSuffix = hasTensor(blk + "mlp.gate.weight") ? "mlp.gate.weight" : "block_sparse_moe.gate.weight"
                let resolvedProbe = resolveName(blk + probeSuffix)
                let resolvedLayerPrefix = resolvedProbe.hasSuffix(probeSuffix)
                    ? String(resolvedProbe.dropLast(probeSuffix.count))
                    : resolveName(blk)
                coldStreamer = ColdExpertStreamer(
                    io: io,
                    layerPrefix: resolvedLayerPrefix,
                    residentExperts: resE,
                    totalExperts: experts,
                    stagingSlots: coldE,
                    intermediate: I,
                    hidden: outDim,
                    guRowBytes: gu.rowBytes,
                    downRowBytes: down.rowBytes,
                    guBasePtr: guPtr,
                    downBasePtr: downPtr,
                    expertPermutation: expertPerm
                )
            } else {
                coldStreamer = nil
            }
            let latent = try loadLatentBottleneckIfNeeded(blk)
            let ISh = geometry.sharedExpertIntermediate
            let sharedGateUp: QWWeightView?
            let sharedDown: QWWeightView?
            let sharedGateScalar: QWTensorValue?
            if ISh > 0 {
                if !hasTensor(blk + "mlp.shared_experts.gate_proj.weight")
                    && !hasTensor(blk + "mlp.shared_expert.gate_proj.weight"),
                   let shBsPrefix = hasTensor(blk + "block_sparse_moe.shared_experts.w1.weight")
                    ? (blk + "block_sparse_moe.shared_experts.")
                    : (hasTensor(blk + "block_sparse_moe.shared_expert.w1.weight")
                        ? (blk + "block_sparse_moe.shared_expert.") : nil) {
                    let (shGU, _) = try concatQ4([
                        (shBsPrefix + "w1.weight", ISh),
                        (shBsPrefix + "w3.weight", ISh)
                    ], k: H)
                    sharedGateUp = shGU
                    sharedDown = try q4(shBsPrefix + "w2.weight", n: H, k: ISh)
                } else {
                    let shPrefix = hasTensor(blk + "mlp.shared_experts.gate_proj.weight")
                        ? (blk + "mlp.shared_experts.")
                        : (blk + "mlp.shared_expert.")
                    let (shGU, _) = try concatQ4([
                        (shPrefix + "gate_proj.weight", ISh),
                        (shPrefix + "up_proj.weight", ISh)
                    ], k: H)
                    sharedGateUp = shGU
                    sharedDown = try q4(shPrefix + "down_proj.weight", n: H, k: ISh)
                }
                sharedGateScalar = geometry.sharedExpertHasGate
                    ? try f32w(blk + "mlp.shared_expert_gate.weight", H)
                    : nil
            } else {
                sharedGateUp = nil
                sharedDown = nil
                sharedGateScalar = nil
            }
            return MoEW(router: router, gateUp: gu, down: down,
                        numExperts: experts, expertsPerTok: topK,
                        sharedGateUp: sharedGateUp, sharedDown: sharedDown,
                        sharedGateScalar: sharedGateScalar,
                        routerBias: routerBias,
                        routedScaleParam: routedScaleParam,
                        splitFeatureRouting: geometry.moeSplitFeatureRouting,
                        latent: latent,
                        coldGateUp: coldBanks?.gu, coldDown: coldBanks?.down,
                        coldStreamer: coldStreamer,
                        useSigmoidScore: geometry.descriptor.moeRoutingConfig?.scoreFunction == "sigmoid",
                        nGroup: geometry.descriptor.moeRoutingConfig?.nGroup ?? 1,
                        topkGroup: geometry.descriptor.moeRoutingConfig?.topkGroup ?? 1,
                        groupTopM: geometry.descriptor.moeRoutingConfig?.groupTopM ?? 1,
                        routedSwigluClamp: geometry.routedSwiGLUClamp(layer: layer),
                        sharedSwigluClamp: geometry.sharedSwiGLUClamp(layer: layer))
        }

        mutating func loadLayerMLP(
            _ blk: String, isMoE: Bool, geometry: QWModelGeometry, layer: Int = 0
        ) throws -> (gu: QWMLPWeight, down: QWWeightView, moe: MoEW?) {
            let experts = isMoE ? geometry.numExperts : 0
            let moe: MoEW? = experts > 0
                ? try packedMoE(blk, experts: experts, topK: geometry.expertsPerTok, layer: layer)
                : nil
            if let moe {
                let outDim = geometry.moeRoutedDim
                guard let v = QWMLPWeight(blocks: moe.gateUp.blocks,
                                         n: 2 * geometry.moeIntermediate,
                                         k: outDim,
                                         bits: moe.gateUp.bits,
                                         groupSize: moe.gateUp.groupSize,
                                         isRotated: moe.gateUp.isRotated,
                                         isIncumbentContainer: moe.gateUp.isIncumbentContainer,
                                         isSHVQ: moe.gateUp.isSHVQ) else {
                    throw QWEngineDriverError.weightMissing(blk + "mlp.experts", "moe gu view invalid")
                }
                let down = try firstExpert(moe.down, rows: outDim)
                return (v, down, moe)
            }
            let gu = try packedGU(blk)
            let downName = hasTensor(blk + "mlp.down_proj.weight")
                ? (blk + "mlp.down_proj.weight")
                : (blk + "block_sparse_moe.w2.weight")
            let down = try q4(downName, n: geometry.hidden, k: geometry.intermediate)
            return (gu, down, nil)
        }

        mutating func loadGQABiasTripletIfNeeded(
            _ attnPrefix: String, geometry: QWModelGeometry
        ) throws -> QWTensorValue? {
            let qName = attnPrefix + "q_proj.bias"
            let kName = attnPrefix + "k_proj.bias"
            let vName = attnPrefix + "v_proj.bias"
            let hasAny = hasTensor(qName) || hasTensor(kName) || hasTensor(vName)
            guard geometry.descriptor.hasQKVBias || geometry.descriptor.hasAttnOutputBias || hasAny else {
                return nil
            }
            for name in [qName, kName, vName] where !hasTensor(name) {
                throw QWEngineDriverError.weightMissing(name, "missing attention QKV bias")
            }
            let qCount = geometry.qProjOut
            let kvCount = geometry.kvHeads * geometry.headDim
            let qFloats = try rawFloats(qName, qCount)
            let kFloats = try rawFloats(kName, kvCount)
            let vFloats = try rawFloats(vName, kvCount)
            let t = try allocateWeight(QWTensorShape([qCount + 2 * kvCount], .f16))
            let dst = t.region.buffer.contents()
                .advanced(by: t.region.base + t.offset)
                .assumingMemoryBound(to: UInt16.self)
            var cursor = 0
            for floats in [qFloats, kFloats, vFloats] {
                for f in floats {
                    dst[cursor] = QWQuant.f32tof16Bits(f)
                    cursor += 1
                }
            }
            return t
        }

        mutating func loadGQAOutputBiasIfNeeded(
            _ attnPrefix: String, geometry: QWModelGeometry
        ) throws -> QWTensorValue? {
            let oName = attnPrefix + "o_proj.bias"
            let exists = hasTensor(oName)
            guard geometry.descriptor.hasAttnOutputBias || exists else { return nil }
            guard exists else {
                throw QWEngineDriverError.weightMissing(oName, "missing attention output bias")
            }
            return try f16w(oName, geometry.hidden)
        }

        mutating func loadGQAAttn(
            _ blk: String, geometry: QWModelGeometry, includeMBSA: Bool
        ) throws -> AttnW {
            var attnSpecs = [
                (blk + "self_attn.q_proj.weight", geometry.qProjOut),
                (blk + "self_attn.k_proj.weight", (geometry.kvHeads * geometry.headDim)),
                (blk + "self_attn.v_proj.weight", (geometry.kvHeads * geometry.headDim))
            ]
            let hasMBSA = includeMBSA && geometry.blockSparseGQAConfig != nil
            let idxPfx = blk + "self_attn.indexer."
            let idxH = geometry.mbsaIndexerHeads
            let idxD = geometry.mbsaIndexerHeadDim
            if hasMBSA {
                attnSpecs.append(contentsOf: [
                    (idxPfx + "linear_q_proj.weight", idxH * idxD),
                    (idxPfx + "linear_k_proj.weight", idxD)
                ])
            }
            let (qkvProj, attnSub) = try concatQ4(attnSpecs, k: geometry.hidden)
            let o = try q4(blk + "self_attn.o_proj.weight",
                           n: geometry.hidden, k: geometry.qHeads * geometry.headDim)
            let qkvBias = try loadGQABiasTripletIfNeeded(blk + "self_attn.", geometry: geometry)
            let oBias = try loadGQAOutputBiasIfNeeded(blk + "self_attn.", geometry: geometry)
            let qNormName = hasTensor(blk + "self_attn.q_norm.weight")
                ? (blk + "self_attn.q_norm.weight")
                : (hasTensor(blk + "self_attn.q_layernorm.weight")
                    ? (blk + "self_attn.q_layernorm.weight") : nil)
            let wq: QWTensorValue? = try qNormName.map { try f16wNorm($0, geometry.headDim) }
            // wk contract (C16 law): [kvHeads·headDim] — tile k_norm ×kvHeads.
            let kNormName = hasTensor(blk + "self_attn.k_norm.weight")
                ? (blk + "self_attn.k_norm.weight")
                : (hasTensor(blk + "self_attn.k_layernorm.weight")
                    ? (blk + "self_attn.k_layernorm.weight") : nil)
            var wkT: QWTensorValue?
            if let kNormName {
                let knBits = try normGammaBits(kNormName, geometry.headDim)
                let t = try allocateWeight(QWTensorShape([(geometry.kvHeads * geometry.headDim)], .f16))
                let p = t.region.buffer.contents().advanced(by: t.region.base + t.offset)
                for h in 0..<geometry.kvHeads {
                    knBits.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                        p.advanced(by: h * geometry.headDim * 2)
                         .copyMemory(from: raw.baseAddress!, byteCount: geometry.headDim * 2)
                    }
                }
                wkT = t
            }
            let mbsaW: QWMBSAAttnOperator.LayerWeights?
            if hasMBSA {
                let wqIdx = try f16wNorm(idxPfx + "q_layernorm.weight", idxD)
                let wkIdx = try f16wNorm(idxPfx + "k_layernorm.weight", idxD)
                mbsaW = QWMBSAAttnOperator.LayerWeights(qkvIdxProj: nil, wqIdx: wqIdx, wkIdx: wkIdx)
            } else {
                mbsaW = nil
            }
            return AttnW(qkvProj: qkvProj, q: attnSub[0], k: attnSub[1], v: attnSub[2],
                         o: o, wq: wq, wkTiled: wkT, mbsa: mbsaW, qkvBias: qkvBias, oBias: oBias)
        }

        /// Every layer's weights, in the plan's order.
        mutating func loadLayers(geometry: QWModelGeometry) throws -> [LayerW] {
            var ws: [LayerW] = []
            for l in 0..<geometry.numLayers {
                let blk = "\(QWEngineDriver.P)layers.\(l)."
                let norms = try loadLayerNorms(blk, geometry: geometry)
                let ln1 = norms.ln1
                let ln2 = norms.ln2
                let gatedNorms = norms.gated
                let (gu, down, moe) = try loadLayerMLP(blk, isMoE: geometry.isMoELayer(l), geometry: geometry, layer: l)
                if geometry.isCompressedAttention(l) {
                    ws.append(try loadHCAAttnLayer(
                        blk: blk, layer: l, ln1: ln1, ln2: ln2,
                        gu: gu, down: down, moe: moe, geometry: geometry))
                    continue
                }
                if geometry.isFullAttention(l) {
                    if geometry.usesNoPEMLA {
                        ws.append(try loadMLAAttnLayer(
                            blk: blk, layer: l, ln1: ln1, ln2: ln2,
                            gu: gu, down: down, moe: moe, geometry: geometry))
                        continue
                    }
                    if geometry.usesDecoupledRopeMLA {
                        ws.append(try loadDecoupledMLAAttnLayer(
                            blk: blk, layer: l, ln1: ln1, ln2: ln2,
                            gu: gu, down: down, moe: moe, geometry: geometry,
                            gatedNorms: gatedNorms))
                        continue
                    }
                    let attn = try loadGQAAttn(blk, geometry: geometry, includeMBSA: true)
                    let mhcW: MHCLayerWeights?
                    if let mhc = geometry.mhcConfig, l >= mhc.start, l < mhc.end {
                        mhcW = try loadMHCOrContractedHC(blk, mhc: mhc)
                    } else {
                        mhcW = nil
                    }
                    ws.append(LayerW(ln1: ln1, ln2: ln2, gu: gu, down: down,
                                     kind: moe == nil ? .fullAttention : .moe,
                                     gdn: nil, attn: attn, moe: moe, mhc: mhcW,
                                     gatedNorms: gatedNorms))
                } else {
                    if geometry.gdnVectorDecay {
                        ws.append(try loadKDALayer(
                            blk: blk, layer: l, ln1: ln1, ln2: ln2,
                            gu: gu, down: down, moe: moe, geometry: geometry))
                        continue
                    }
                    let gdnSpecs = [
                        (blk + "linear_attn.in_proj_qkv.weight", geometry.gdnQKVChannels),
                        (blk + "linear_attn.in_proj_z.weight", geometry.gdnZChannels),
                        (blk + "linear_attn.in_proj_a.weight", geometry.gdnVHeads),
                        (blk + "linear_attn.in_proj_b.weight", geometry.gdnVHeads)
                    ]
                    let (inProj, gdnSub) = try concatQ4(gdnSpecs, k: geometry.hidden)
                    let qkv = gdnSub[0], z = gdnSub[1], a = gdnSub[2], b = gdnSub[3]
                    let conv = try f16w(blk + "linear_attn.conv1d.weight", geometry.gdnQKVChannels * geometry.gdnConvKernel)
                    // The GDN's gated norm multiplies by `w` outright — it is
                    // not a `1 + w` norm, so its gamma is loaded verbatim.
                    let gnw = try f16w(blk + "linear_attn.norm.weight", geometry.gdnHeadDim)
                    let aLog = try f32w(blk + "linear_attn.A_log", geometry.gdnVHeads)
                    let dtB = try f32w(blk + "linear_attn.dt_bias", geometry.gdnVHeads)
                    let out = try q4(blk + "linear_attn.out_proj.weight", n: geometry.hidden, k: geometry.gdnZChannels)
                    let mhcW: MHCLayerWeights?
                    if let mhc = geometry.mhcConfig, l >= mhc.start, l < mhc.end {
                        mhcW = try loadMHC(blk, mhc: mhc)
                    } else {
                        mhcW = nil
                    }
                    ws.append(LayerW(ln1: ln1, ln2: ln2, gu: gu, down: down,
                                     kind: moe == nil ? .gdn : .gdnMoe,
                                     gdn: GdnW(inProj: inProj, qkv: qkv, z: z, a: a, b: b, conv: conv, normW: gnw,
                                               aLog: aLog, dtBias: dtB, out: out),
                                     attn: nil, moe: moe, mhc: mhcW,
                                     gatedNorms: gatedNorms))
                }
            }
            return ws
        }
    }
