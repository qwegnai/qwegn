// DriverInitPhases.swift — init-phase builders extracted from
// QWEngineDriver.init (MTP draft weights, GDN slot views,
// PSO pre-warm). Allocation order preserved. The per-layer weights
// load in DriverWeightLoading.swift, beside the loader they use.
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
import QWMLP
import QWHead
import QWMTP
import QWState

extension QWEngineDriver {
    static func loadMTPWeights(_ L: inout WeightLoader, drafts: Int, geometry: QWModelGeometry) throws -> MTPWeights? {
        guard drafts > 0, !geometry.mtpSharedLastLayer else { return nil }
        if !L.hasTensor("mtp.pre_fc_norm_embedding.weight"),
           let nextN = try L.loadNextNMTPLayer(geometry: geometry),
           let attn = nextN.layer.attn {
            return MTPWeights(
                preFcNormEmbedding: nextN.eNorm,
                preFcNormHidden: nextN.hNorm,
                fc: nextN.ehProj,
                inputNorm: nextN.layer.ln1,
                qkvProj: attn.qkvProj,
                q: attn.q, k: attn.k, v: attn.v,
                o: attn.o,
                wq: attn.wq, wkTiled: attn.wkTiled,
                postAttnNorm: nextN.layer.ln2,
                gu: nextN.layer.gu, down: nextN.layer.down,
                norm: nextN.sharedHeadNorm,
                moe: nextN.layer.moe,
                mla: attn.mla
            )
        }
            let mPreFcNormEmb = try L.f16wNorm("mtp.pre_fc_norm_embedding.weight", geometry.hidden)
            let mPreFcNormHid = try L.f16wNorm("mtp.pre_fc_norm_hidden.weight", geometry.hidden)
            let mFc = try L.q4("mtp.fc.weight", n: geometry.hidden, k: 2 * geometry.hidden)
            let mInputNorm = try L.f16wNorm("mtp.layers.0.input_layernorm.weight", geometry.hidden)
            let mAttnSpecs = [
                ("mtp.layers.0.self_attn.q_proj.weight", geometry.qProjOut),
                ("mtp.layers.0.self_attn.k_proj.weight", (geometry.kvHeads * geometry.headDim)),
                ("mtp.layers.0.self_attn.v_proj.weight", (geometry.kvHeads * geometry.headDim))
            ]
            let (mQKVProj, mAttnSub) = try L.concatQ4(mAttnSpecs, k: geometry.hidden)
            let mQ = mAttnSub[0], mK = mAttnSub[1], mV = mAttnSub[2]
            let mO = try L.q4("mtp.layers.0.self_attn.o_proj.weight",
                            n: geometry.hidden, k: geometry.qHeads * geometry.headDim)
            let mWq = try L.f16wNorm("mtp.layers.0.self_attn.q_norm.weight", geometry.headDim)
            let mKnBits = try L.normGammaBits("mtp.layers.0.self_attn.k_norm.weight", geometry.headDim)
            let mWkT = try L.allocateWeight(QWTensorShape([(geometry.kvHeads * geometry.headDim)], .f16))
            let pWkT = mWkT.region.buffer.contents().advanced(by: mWkT.region.base + mWkT.offset)
            for h in 0..<geometry.kvHeads {
                mKnBits.withUnsafeBytes {
                    pWkT.advanced(by: h * geometry.headDim * 2)
                        .copyMemory(from: $0.baseAddress!, byteCount: geometry.headDim * 2)
                }
            }
            let mPostAttnNorm = try L.f16wNorm("mtp.layers.0.post_attention_layernorm.weight", geometry.hidden)
            let experts = geometry.numExperts
            let mMoE: MoEW? = experts > 0
                ? try L.packedMoE("mtp.layers.0.", experts: experts, topK: geometry.expertsPerTok)
                : nil
            let mGu: QWMLPWeight
            let mDown: QWWeightView
            if let mMoE {
                guard let v = QWMLPWeight(blocks: mMoE.gateUp.blocks,
                                          n: 2 * geometry.moeIntermediate,
                                          k: geometry.hidden,
                                          bits: mMoE.gateUp.bits,
                                          groupSize: mMoE.gateUp.groupSize,
                                          isRotated: mMoE.gateUp.isRotated,
                                          isIncumbentContainer: mMoE.gateUp.isIncumbentContainer,
                                          isSHVQ: mMoE.gateUp.isSHVQ) else {
                    throw QWEngineDriverError.weightMissing("mtp.layers.0.mlp.experts", "mtp moe gu view invalid")
                }
                mGu = v
                mDown = try L.firstExpert(mMoE.down, rows: geometry.hidden)
            } else {
                mGu = try L.packedGU("mtp.layers.0.")
                mDown = try L.q4("mtp.layers.0.mlp.down_proj.weight", n: geometry.hidden, k: geometry.intermediate)
            }
            let mNorm = try L.f16wNorm("mtp.norm.weight", geometry.hidden)

        return MTPWeights(
                preFcNormEmbedding: mPreFcNormEmb,
                preFcNormHidden: mPreFcNormHid,
                fc: mFc,
                inputNorm: mInputNorm,
                qkvProj: mQKVProj,
                q: mQ, k: mK, v: mV,
                o: mO,
                wq: mWq, wkTiled: mWkT,
                postAttnNorm: mPostAttnNorm,
                gu: mGu, down: mDown,
                norm: mNorm,
                moe: mMoE
            )
    }

    /// Creates every door the engine may dispatch, ONCE. The doors come
    /// from the module that owns the engine: there is no global list.
    static func prewarmPipelines(hal: QWHal, doors: [String]) throws {
        for k in doors { _ = try hal.pipeline(k) }
        hal.freezePipelines()
    }

    static func slotViews(_ slab: QWTensorValue, elem: Int, dt: QWDType, layers: Int, geometry: QWModelGeometry) -> [QWTensorValue] {
        let strideBytes = Int((Double(elem) * dt.bytesPerParam).rounded())
        return (0..<layers).map { g in
            QWTensorValue(region: slab.region,
                          offset: slab.offset + g * strideBytes,
                          shape: QWTensorShape([elem], dt))
        }
    }

    static func makeKVCaches(
        hal: QWHal, arena: QWArena, attnLayers: [Int], kvMode: QWKVMode,
        geometry: QWModelGeometry, mtpDrafts: Int, trunkKVBytes: Int, mtpKVBytes: Int
    ) throws -> (kvCache: QWKVCache, mtpCache: QWKVCache?) {
        if mtpDrafts > 0 {
            if let kvV = arena.kvVSub {
                let trunkVLen = ((trunkKVBytes / 2) / 256) * 256
                let trunkKLen = trunkKVBytes - trunkVLen
                let trunkKOff = try arena.kvRegion().allocate(bytes: trunkKLen)
                let trunkVOff = try kvV.allocate(bytes: trunkVLen)
                let trunkKSub = QWSubAllocator(region: QWMRegion(buffer: arena.kvRegion().region.buffer,
                                                                 base: arena.kvRegion().region.base + trunkKOff,
                                                                 length: trunkKLen))
                let trunkVSub = QWSubAllocator(region: QWMRegion(buffer: kvV.region.buffer,
                                                                 base: kvV.region.base + trunkVOff,
                                                                 length: trunkVLen))
                let kv = try QWKVCache(hal: hal, subAllocator: trunkKSub, vSubAllocator: trunkVSub,
                                       layers: attnLayers, mode: kvMode, arena: arena, geometry: geometry)

                let mtpVLen = ((mtpKVBytes / 2) / 256) * 256
                let mtpKLen = mtpKVBytes - mtpVLen
                let mtpKOff = try arena.kvRegion().allocate(bytes: mtpKLen)
                let mtpVOff = try kvV.allocate(bytes: mtpVLen)
                let mtpKSub = QWSubAllocator(region: QWMRegion(buffer: arena.kvRegion().region.buffer,
                                                               base: arena.kvRegion().region.base + mtpKOff,
                                                               length: mtpKLen))
                let mtpVSub = QWSubAllocator(region: QWMRegion(buffer: kvV.region.buffer,
                                                               base: kvV.region.base + mtpVOff,
                                                               length: mtpVLen))
                let mtp = try QWKVCache(hal: hal, subAllocator: mtpKSub, vSubAllocator: mtpVSub,
                                        layers: [0], mode: kvMode, arena: arena, geometry: geometry)
                return (kv, mtp)
            } else {
                let trunkOff = try arena.kvRegion().allocate(bytes: trunkKVBytes)
                let trunkSub = QWSubAllocator(region: QWMRegion(buffer: arena.kvRegion().region.buffer,
                                                                base: arena.kvRegion().region.base + trunkOff,
                                                                length: trunkKVBytes))
                let kv = try QWKVCache(hal: hal, subAllocator: trunkSub, layers: attnLayers,
                                       mode: kvMode, arena: arena, geometry: geometry)

                let mtpOff = try arena.kvRegion().allocate(bytes: mtpKVBytes)
                let mtpSub = QWSubAllocator(region: QWMRegion(buffer: arena.kvRegion().region.buffer,
                                                              base: arena.kvRegion().region.base + mtpOff,
                                                              length: mtpKVBytes))
                let mtp = try QWKVCache(hal: hal, subAllocator: mtpSub, layers: [0],
                                        mode: kvMode, arena: arena, geometry: geometry)
                return (kv, mtp)
            }
        } else {
            let kv = try QWKVCache(hal: hal, arena: arena, layers: attnLayers, mode: kvMode, geometry: geometry)
            return (kv, nil)
        }
    }

    static func makeDecodeAndPrefillScratch(
        asub: QWSubAllocator, gsub: QWSubAllocator,
        geometry: QWModelGeometry, maxPrefillChunk: Int,
        pOff: Int, pCarveBytes: Int,
        useGroupedNAX: Bool = true
    ) throws -> (sc: Scratch, psc: PrefillScratch, gsOff: Int) {
        func tv(_ n: Int, _ dt: QWDType) throws -> QWTensorValue {
            try asub.allocate(QWTensorShape([max(1, n)], dt))
        }
        let H = geometry.hidden, I = geometry.intermediate, V = geometry.vocab
        let qOut = geometry.qProjOut, kvE = geometry.kvChannels
        let qkvC = geometry.gdnQKVChannels, zC = geometry.gdnZChannels
        let gdnVH = geometry.gdnVHeads, qC = geometry.qChannels
        let glogC = geometry.gdnGlogChannels
        let fgbC = 2 * geometry.gdnLowRankDim + gdnVH

        let gdnY = try tv(geometry.gdnTotalN, .f32)
        let gdnY16 = try tv(geometry.gdnTotalN, .f16)
        let qkv32 = QWTensorValue(region: gdnY.region, offset: gdnY.offset, shape: QWTensorShape([max(1, qkvC)], .f32))
        let z32 = QWTensorValue(region: gdnY.region, offset: gdnY.offset + qkvC * 4, shape: QWTensorShape([max(1, zC)], .f32))
        let a32 = QWTensorValue(region: gdnY.region, offset: gdnY.offset + geometry.gdnARow0 * 4, shape: QWTensorShape([max(1, gdnVH)], .f32))
        let b32 = QWTensorValue(region: gdnY.region, offset: gdnY.offset + geometry.gdnBRow0 * 4, shape: QWTensorShape([max(1, gdnVH)], .f32))
        let qkv16 = QWTensorValue(region: gdnY16.region, offset: gdnY16.offset, shape: QWTensorShape([max(1, qkvC)], .f16))
        let z16 = QWTensorValue(region: gdnY16.region, offset: gdnY16.offset + qkvC * 2, shape: QWTensorShape([max(1, zC)], .f16))

        let attnY = try tv(geometry.attnTotalN, .f32)
        let attnY16 = try tv(geometry.attnTotalN, .f16)
        let q32 = QWTensorValue(region: attnY.region, offset: attnY.offset, shape: QWTensorShape([qOut], .f32))
        let k32 = QWTensorValue(region: attnY.region, offset: attnY.offset + geometry.attnKRow0 * 4, shape: QWTensorShape([kvE], .f32))
        let v32 = QWTensorValue(region: attnY.region, offset: attnY.offset + geometry.attnVRow0 * 4, shape: QWTensorShape([kvE], .f32))
        let q16 = QWTensorValue(region: attnY16.region, offset: attnY16.offset, shape: QWTensorShape([qOut], .f16))
        let k16 = QWTensorValue(region: attnY16.region, offset: attnY16.offset + geometry.attnKRow0 * 2, shape: QWTensorShape([kvE], .f16))
        let v16 = QWTensorValue(region: attnY16.region, offset: attnY16.offset + geometry.attnVRow0 * 2, shape: QWTensorShape([kvE], .f16))

        let maxGsCols = max(H, max(geometry.qLoraRank, geometry.hcaConfig?.qLoraRank ?? 0))
        let maxGsICols = max(I, max(geometry.qHeads * geometry.vHeadDim,
                                    geometry.hcaConfig.map { max(1, $0.oGroups) * $0.oLoraRank } ?? 0))
        let attOutCols = max(qC, geometry.qHeads * geometry.vHeadDim)
        let sc = Scratch(
            x: try tv(H, .f16),
            ln1: try tv(H, .f16), ln2: try tv(H, .f16),
            gs: try tv(max(1, maxGsCols / 128), .f32), gs48: try tv(Self.oProjGroupSumFloats(geometry: geometry), .f32), gsI: try tv(max(1, maxGsICols / 128), .f32),
            gdnY: gdnY, gdnY16: gdnY16,
            qkv32: qkv32, qkv16: qkv16,
            z32: z32, z16: z16,
            a32: a32, b32: b32,
            betas: try tv(gdnVH, .f32), glog: try tv(glogC, .f32),
            convY: try tv(qkvC, .f16),
            oScr: try tv(zC, .f32), oOut: try tv(zC, .f16),
            attnY: attnY, attnY16: attnY16,
            q32: q32, k32: k32, v32: v32,
            q16: q16, k16: k16, v16: v16,
            qR: try tv(qOut, .f16), kR: try tv(kvE, .f16),
            attOut: try tv(attOutCols, .f16),
            o32: try tv(H, .f32), o16: try tv(H, .f16),
            gu32: try tv(I, .f32), gu16: try tv(I, .f16),
            dn32: try tv(H, .f32), dn16: try tv(H, .f16),
            logits: try tv(V, .f32),
            keysHi: try tv(QWHead.gemvArgmaxSlots(V), .i32),
            keysLo: try tv(QWHead.gemvArgmaxSlots(V), .i32),
            token: try tv(1, .i32),
            partials: QWMRegion(buffer: gsub.region.buffer, base: gsub.region.base + pOff,
                                length: pCarveBytes),
            moe: geometry.numExperts > 0 ? try Self.makeMoEScratch(tv, rows: 1, geometry: geometry, useGroupedNAX: useGroupedNAX) : nil,
            mhcStream: try geometry.mhcConfig.map { try tv($0.streams * H, .f32) },
            mhcControls: try geometry.mhcConfig.map { try tv($0.streams + $0.streams * $0.streams, .f32) },
            fgb16: geometry.gdnVectorDecay ? try tv(fgbC, .f16) : nil
        )

        let maxT = maxPrefillChunk
        let pscStartOff = asub.usedBytes
        let usesMLA = geometry.usesLatentOrCompressedAttention
        let psc = PrefillScratch(
            maxChunk: maxT,
            x: try tv(maxT * H, .f16),
            ln1: try tv(maxT * H, .f16),
            ln2: try tv(maxT * H, .f16),
            gdnY: try tv(geometry.gdnVectorDecay ? 0 : maxT * geometry.gdnTotalN, .f32),
            attnY16: try tv(usesMLA ? 0 : maxT * geometry.attnTotalN, .f16),
            qkv16: try tv(maxT * qkvC, .f16),
            z16: try tv(maxT * zC, .f16),
            betas: try tv(maxT * gdnVH, .f32), glog: try tv(maxT * glogC, .f32),
            convY: try tv(maxT * qkvC, .f16), oScr: try tv(maxT * zC, .f32),
            oOut: try tv(maxT * zC, .f16), attOut: try tv(maxT * attOutCols, .f16),
            attV16: try tv(usesMLA ? 0 : (maxT + QWAttnPrefill.padRows) * kvE, .f16),
            attQR16: try tv(usesMLA ? 0 : (maxT + QWAttnPrefill.padRows) * qOut, .f16),
            attKR16: try tv(usesMLA ? 0 : (maxT + QWAttnPrefill.padRows) * kvE, .f16),
            U: try tv(maxT * zC, .f32), W: try tv(maxT * zC, .f32),
            A: try tv(maxT * zC, .f32), B: try tv(maxT * zC, .f32),
            o16: try tv(maxT * H, .f16),
            guRaw16: try tv(maxT * max(2 * I, 2 * H), .f16),
            gu16: try tv(maxT * I, .f16), dn32: try tv(maxT * H, .f32), dn16: try tv(maxT * H, .f16),
            moe: geometry.numExperts > 0
                ? try Self.makeMoEScratch(tv, rows: maxT, geometry: geometry, useGroupedNAX: useGroupedNAX)
                : nil,
            mhcStream: try geometry.mhcConfig.map { try tv(maxT * $0.streams * H, .f32) },
            mhcControls: try geometry.mhcConfig.map { try tv(maxT * ($0.streams + $0.streams * $0.streams), .f32) },
            fgb16: geometry.gdnVectorDecay ? try tv(maxT * fgbC, .f16) : nil
        )
        return (sc, psc, pscStartOff)
    }
}
