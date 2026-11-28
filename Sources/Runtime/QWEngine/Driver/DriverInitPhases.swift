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

extension QWEngineDriver {
    static func loadMTPWeights(_ L: inout WeightLoader, drafts: Int, geometry: QWModelGeometry) throws -> MTPWeights? {
        guard drafts > 0, !geometry.mtpSharedLastLayer, !geometry.hasDedicatedMHCMTP else { return nil }
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
                mla: attn.mla,
                qkvBias: attn.qkvBias,
                oBias: attn.oBias,
                gatedNorms: nextN.gatedNorms
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
            let mQkvBias = try L.loadGQABiasTripletIfNeeded("mtp.layers.0.self_attn.", geometry: geometry)
            let mOBias = try L.loadGQAOutputBiasIfNeeded("mtp.layers.0.self_attn.", geometry: geometry)
            let mWq = try L.f16wNorm("mtp.layers.0.self_attn.q_norm.weight", geometry.headDim)
            let mKnBits = try L.normGammaBits("mtp.layers.0.self_attn.k_norm.weight", geometry.headDim)
            let mWkT = try L.allocateWeight(QWTensorShape([(geometry.kvHeads * geometry.headDim)], .f16))
            let pWkT = mWkT.region.buffer.contents().advanced(by: mWkT.region.base + mWkT.offset)
            for h in 0..<geometry.kvHeads {
                mKnBits.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                    pWkT.advanced(by: h * geometry.headDim * 2)
                        .copyMemory(from: raw.baseAddress!, byteCount: geometry.headDim * 2)
                }
            }
            let mPostAttnNorm = try L.f16wNorm("mtp.layers.0.post_attention_layernorm.weight", geometry.hidden)
            let experts = geometry.numExperts
            let mMoE: MoEW? = experts > 0
                ? try L.packedMoE("mtp.layers.0.", experts: experts, topK: geometry.expertsPerTok, layer: geometry.numLayers)
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
                moe: mMoE,
                qkvBias: mQkvBias,
                oBias: mOBias
            )
    }

    private static let lsaSelectorDoors: Set<String> = [
        "qwg_lsa_select_blocks",
        "qwg_lsa_select_score",
        "qwg_lsa_select_hist2",
        "qwg_lsa_select_expand",
        "qwg_lsa_select_finalize",
    ]

    static func prewarmDoors(_ doors: [String], hasLSAOperator: Bool) -> [String] {
        hasLSAOperator ? doors : doors.filter { !lsaSelectorDoors.contains($0) }
    }

    /// Creates every door the engine may dispatch, ONCE. The doors come
    /// from the module that owns the engine: there is no global list.
    static func prewarmPipelines(hal: QWHal, doors: [String]) throws {
        let specs = hal.geometry.swigluClampConfig?.distinctSpecs() ?? []
        let clampSubstrings = [
            "silu", "swiglu", "moe_route_decode", "moe_gateup", "moe_compact", "mlp_gate_up"
        ]
        let activeDoors = hal.caps.isMetal4TensorCapable
            ? doors
            : doors.filter { !QWNaxKernelRegistry.names.contains($0) }
        for k in activeDoors {
            _ = try hal.pipeline(k)
            if !specs.isEmpty && clampSubstrings.contains(where: { k.contains($0) }) {
                for spec in specs {
                    _ = try hal.pipeline(k, swigluClamp: spec)
                }
            }
        }
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

    private static func resolvedTrunkPoolWeights(
        attnLayers: [Int], trunkKVBytes: Int, kvMode: QWKVMode, geometry: QWModelGeometry,
        maxTokens: Int?, prefillChunk: Int
    ) -> [Int]? {
        guard geometry.hasAnySlidingWindow, attnLayers.count > 1 else { return nil }
        let windows = attnLayers.map { geometry.virtualLayerSlidingWindow($0) }
        guard windows.contains(where: { $0 > 0 }) else { return nil }
        func weightsFor(_ tok: Int, chunk: Int) -> [Int] {
            windows.map { w in
                let cap = w > 0 ? min(tok, w + chunk + 512) : tok
                return QWEngineDriver.kvPagesPerPool(maxTokens: cap)
            }
        }
        if let maxTokens { return weightsFor(maxTokens, chunk: prefillChunk) }
        let pageBytes = QWKVCache.pageBytes(mode: kvMode, geometry: geometry)
        let tablesTotal = QWKVCache.tableSlotBytes * 2 * attnLayers.count * QWKVCache.maxSeqs
        let targetPages = max(attnLayers.count, (trunkKVBytes - tablesTotal) / (2 * pageBytes))
        func solveMaxTokens(chunk: Int) -> (tok: Int, total: Int) {
            var lo = 1, hi = max(geometry.maxPositionEmbeddings, 1_048_576), best = 1
            while lo <= hi {
                let mid = lo + (hi - lo) / 2
                let sum = weightsFor(mid, chunk: chunk).reduce(0, +)
                if sum <= targetPages { best = mid; lo = mid + 1 } else { hi = mid - 1 }
            }
            return (best, weightsFor(best, chunk: chunk).reduce(0, +))
        }
        let candidates = [prefillChunk, QWPrefillLimits.chunkRows, 2048, 4096, 1024, 3584, 3072, 2560, 1536]
        for chunk in candidates {
            let solved = solveMaxTokens(chunk: chunk)
            if solved.total == targetPages,
               solved.tok == geometry.maxPositionEmbeddings || solved.tok % 4096 == 0 {
                return weightsFor(solved.tok, chunk: chunk)
            }
        }
        for chunk in candidates {
            let solved = solveMaxTokens(chunk: chunk)
            if solved.total == targetPages { return weightsFor(solved.tok, chunk: chunk) }
        }
        return weightsFor(solveMaxTokens(chunk: prefillChunk).tok, chunk: prefillChunk)
    }

    static func makeKVCaches(
        hal: QWHal, arena: QWArena, attnLayers: [Int], kvMode: QWKVMode,
        geometry: QWModelGeometry, mtpDrafts: Int, trunkKVBytes: Int, mtpKVBytes: Int,
        maxTokens: Int? = nil, prefillChunk: Int = 4096
    ) throws -> (kvCache: QWKVCache, mtpCache: QWKVCache?) {
        func carveSub(_ parent: QWSubAllocator, bytes: Int) throws -> QWSubAllocator {
            let off = try parent.allocate(bytes: bytes)
            return QWSubAllocator(region: QWMRegion(buffer: parent.region.buffer,
                                                    base: parent.region.base + off, length: bytes))
        }
        let trunkWindows = attnLayers.map { geometry.virtualLayerSlidingWindow($0) }
        let effectiveTrunkBytes = trunkKVBytes > 0 ? trunkKVBytes
            : arena.kvRegion().freeBytes + (arena.kvVSub?.freeBytes ?? 0) + (arena.kvAuxSub?.freeBytes ?? 0)
        let trunkWeights = resolvedTrunkPoolWeights(
            attnLayers: attnLayers, trunkKVBytes: effectiveTrunkBytes,
            kvMode: kvMode, geometry: geometry, maxTokens: maxTokens, prefillChunk: prefillChunk)
        guard mtpDrafts > 0 else {
            let kv = try QWKVCache(hal: hal, arena: arena, layers: attnLayers, mode: kvMode,
                                   geometry: geometry, layerSlidingWindows: trunkWindows,
                                   layerPoolWeights: trunkWeights)
            return (kv, nil)
        }
        func makeSplitOrSingle(layers: [Int], totalBytes: Int,
                               windows: [Int], weights: [Int]?) throws -> QWKVCache {
            if let kvV = arena.kvVSub {
                let vLen = ((totalBytes / 2) / 256) * 256
                let kLen = totalBytes - vLen
                let kSub = try carveSub(arena.kvRegion(), bytes: kLen)
                let vSub = try carveSub(kvV, bytes: vLen)
                return try QWKVCache(hal: hal, subAllocator: kSub, vSubAllocator: vSub,
                                     layers: layers, mode: kvMode, arena: arena, geometry: geometry,
                                     layerSlidingWindows: windows, layerPoolWeights: weights)
            }
            let sub = try carveSub(arena.kvRegion(), bytes: totalBytes)
            return try QWKVCache(hal: hal, subAllocator: sub, layers: layers,
                                 mode: kvMode, arena: arena, geometry: geometry,
                                 layerSlidingWindows: windows, layerPoolWeights: weights)
        }
        let kv = try makeSplitOrSingle(layers: attnLayers, totalBytes: trunkKVBytes,
                                       windows: trunkWindows, weights: trunkWeights)
        let mtp = try makeSplitOrSingle(layers: [0], totalBytes: mtpKVBytes,
                                        windows: [0], weights: nil)
        return (kv, mtp)
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
        let fgbC = geometry.kdaFgbChannels

        let gdnY = try tv(geometry.gdnTotalN, .f32)
        let gdnY16 = try tv(geometry.gdnTotalN, .f16)
        let qkv32 = QWTensorValue(region: gdnY.region, offset: gdnY.offset, shape: QWTensorShape([max(1, qkvC)], .f32))
        let z32 = QWTensorValue(region: gdnY.region, offset: gdnY.offset + qkvC * 4, shape: QWTensorShape([max(1, zC)], .f32))
        let a32 = QWTensorValue(region: gdnY.region, offset: gdnY.offset + geometry.gdnARow0 * 4, shape: QWTensorShape([max(1, gdnVH)], .f32))
        let b32 = QWTensorValue(region: gdnY.region, offset: gdnY.offset + geometry.gdnBRow0 * 4, shape: QWTensorShape([max(1, gdnVH)], .f32))
        let qkv16 = QWTensorValue(region: gdnY16.region, offset: gdnY16.offset, shape: QWTensorShape([max(1, qkvC)], .f16))
        let z16 = QWTensorValue(region: gdnY16.region, offset: gdnY16.offset + qkvC * 2, shape: QWTensorShape([max(1, zC)], .f16))

        let attnY = try tv(geometry.attnProjTotalN, .f32)
        let attnY16 = try tv(geometry.attnProjTotalN, .f16)
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
        let pX = try tv(maxT * H, .f16)
        let pLn1 = try tv(maxT * H, .f16)
        let pLn2 = try tv(maxT * H, .f16)
        let pO16 = try tv(maxT * H, .f16)
        let pDn32 = try tv(maxT * H, .f32)
        let pDn16 = try tv(maxT * H, .f16)
        let pMhcStream = try geometry.mhcConfig.map { try tv(maxT * $0.streams * H, .f32) }
        let pMhcControls = try geometry.mhcConfig.map { try tv(maxT * ($0.streams + $0.streams * $0.streams), .f32) }

        // Attention/GDN transients and MLP/MoE transients never overlap within a layer.
        let aliasBranches = asub.regions.count == 1
        let moePersistentBytes = geometry.numExperts > 0
            ? (qwAlignUp(max(1, maxT * H) * 4) + qwAlignUp(max(1, maxT) * 4))
            : 0
        let moePersistentSub: QWSubAllocator?
        if aliasBranches && moePersistentBytes > 0 {
            let moeOff = try asub.allocate(bytes: moePersistentBytes)
            moePersistentSub = QWSubAllocator(region: QWMRegion(
                buffer: asub.region.buffer,
                base: asub.region.base + moeOff,
                length: moePersistentBytes))
        } else {
            moePersistentSub = nil
        }
        func moePersistentTv(_ n: Int, _ dt: QWDType) throws -> QWTensorValue {
            guard let moePersistentSub else { return try tv(n, dt) }
            return try moePersistentSub.allocate(QWTensorShape([max(1, n)], dt))
        }
        let transientBase = asub.usedBytes
        let branchLen = asub.freeBytes
        let attnSub: QWSubAllocator? = aliasBranches
            ? QWSubAllocator(region: QWMRegion(buffer: asub.region.buffer,
                                               base: asub.region.base + transientBase,
                                               length: branchLen))
            : nil
        let mlpSub: QWSubAllocator? = aliasBranches
            ? QWSubAllocator(region: QWMRegion(buffer: asub.region.buffer,
                                               base: asub.region.base + transientBase,
                                               length: branchLen))
            : nil
        func attnTv(_ n: Int, _ dt: QWDType) throws -> QWTensorValue {
            guard let attnSub else { return try tv(n, dt) }
            return try attnSub.allocate(QWTensorShape([max(1, n)], dt))
        }
        func mlpTv(_ n: Int, _ dt: QWDType) throws -> QWTensorValue {
            guard let mlpSub else { return try tv(n, dt) }
            return try mlpSub.allocate(QWTensorShape([max(1, n)], dt))
        }

        let wyElems = (!geometry.gdnVectorDecay && geometry.gdnLayerCount > 0 && maxT > 4096) ? (maxT * zC) : 0
        let gdnYElems = geometry.gdnVectorDecay
            ? 0
            : max(maxT * 2 * gdnVH, QWBatchHeadScratch.maxBatchRows * geometry.gdnTotalN)
        let pGdnY = try attnTv(gdnYElems, .f32)
        let pAttnY16 = try attnTv(usesMLA ? 0 : maxT * geometry.attnProjTotalN, .f16)
        let pQkv16 = try attnTv(maxT * qkvC, .f16)
        let pZ16 = try attnTv(maxT * zC, .f16)
        let pBetas = try attnTv(maxT * gdnVH, .f32)
        let pGlog = try attnTv(maxT * glogC, .f32)
        let pConvY = try attnTv(maxT * qkvC, .f16)
        let pOScr = try attnTv(maxT * zC, .f32)
        let pOOut = try attnTv(maxT * zC, .f16)
        let pAttOut = try attnTv(maxT * attOutCols, .f16)
        let pAttV16 = try attnTv(usesMLA ? 0 : (maxT + QWAttnPrefill.padRows) * kvE, .f16)
        let pAttQR16 = try attnTv(usesMLA ? 0 : (maxT + QWAttnPrefill.padRows) * qOut, .f16)
        let pAttKR16 = try attnTv(usesMLA ? 0 : (maxT + QWAttnPrefill.padRows) * kvE, .f16)
        let pU = try attnTv(wyElems, .f32)
        let pW = try attnTv(wyElems, .f32)
        let pA = try attnTv(wyElems, .f32)
        let pB = try attnTv(wyElems, .f32)
        let pFgb16 = geometry.gdnVectorDecay ? try attnTv(maxT * fgbC, .f16) : nil

        let usesFusedSwiGLU = (I & 31) == 0 && useGroupedNAX
        let pGuRaw16 = try mlpTv(maxT * (usesFusedSwiGLU ? (2 * H) : max(2 * I, 2 * H)), .f16)
        let pGu16 = try mlpTv(maxT * I, .f16)
        let pMoe: MoEScratch? = geometry.numExperts > 0
            ? try Self.makeMoEScratch(mlpTv, persistentAlloc: moePersistentTv, rows: maxT, geometry: geometry, useGroupedNAX: useGroupedNAX)
            : nil
        if let attnSub, let mlpSub {
            let transientPeak = max(attnSub.usedBytes, mlpSub.usedBytes)
            if transientPeak > 0 { try asub.allocate(bytes: transientPeak) }
        }
        let psc = PrefillScratch(
            maxChunk: maxT,
            x: pX, ln1: pLn1, ln2: pLn2,
            gdnY: pGdnY, attnY16: pAttnY16,
            qkv16: pQkv16, z16: pZ16,
            betas: pBetas, glog: pGlog,
            convY: pConvY, oScr: pOScr,
            oOut: pOOut, attOut: pAttOut,
            attV16: pAttV16, attQR16: pAttQR16, attKR16: pAttKR16,
            U: pU, W: pW, A: pA, B: pB,
            o16: pO16, guRaw16: pGuRaw16, gu16: pGu16, dn32: pDn32, dn16: pDn16,
            moe: pMoe, mhcStream: pMhcStream, mhcControls: pMhcControls, fgb16: pFgb16
        )
        return (sc, psc, pscStartOff)
    }
}
