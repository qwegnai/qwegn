// QWMoEOperator+Latent.swift — Latent-bottleneck Sparse-MoE execution (`fc1 -> experts -> latent_norm -> fc2`).
// Keeps full-hidden shared experts on `[rows, H]` while running routed experts in `[rows, L]` with 0 empty NAX tiles.

import Foundation
import Metal
import QWAlloc
import QWCore
import QWHAL

extension QWMoEOperator {
    /// Density-aware token crossover for Grouped-NAX MoE prefill on high-expert-count models.
    public static func groupedNaxMinRows(experts: Int, topK: Int) -> Int {
        guard experts >= 256 && topK > 0 else { return 9 }
        return max(16, (experts + topK - 1) / (2 * topK))
    }

    static func encodeResidentLatent(
        _ encRef: inout any MTLComputeCommandEncoder,
        hal: QWHal,
        weights w: Weights,
        latent: LatentBottleneck,
        scratch s: Scratch,
        input: Input,
        out32: QWTensorValue,
        out16: QWTensorValue?,
        skipOut32: Bool = false,
        rows T: Int,
        supportsGroupedNaxPrefill: Bool,
        concurrentBarriers: Bool,
        rotateEncoder: ((inout any MTLComputeCommandEncoder) throws -> Void)?,
        profileStage: ((String) -> (any MTLComputeCommandEncoder)?)?
    ) throws {
        var enc = encRef
        defer { encRef = enc }

        guard let ls = s.latent else {
            throw QWGemmError.badShape("latent moe requires LatentScratch")
        }
        @inline(__always) func barrier() {
            if concurrentBarriers { enc.memoryBarrier(scope: .buffers) }
        }

        let H = w.hidden, L = latent.latentDim, E = w.experts, K = w.topK
        let I = w.intermediate, ISh = w.sharedIntermediate
        let routerCols = w.splitFeatureRouting ? (2 * E) : E
        let allowDenseNax = supportsGroupedNaxPrefill && hal.caps.supportsSimdgroupMMA
        let minNax = allowDenseNax ? 8 : Int.max
        let neededGU32 = T * K * 2 * I
        let neededAct32 = T * K * I
        let hasFallbackF32Scratch = T <= 8
            || (s.gu32.shape.elementCount >= neededGU32 && s.act32.shape.elementCount >= neededAct32)
            || ((s.gu16?.bytes ?? 0) >= neededGU32 * 4 && (s.dnHi?.bytes ?? 0) >= neededAct32 * 4)
        let minGroupedRows = hasFallbackF32Scratch ? Self.groupedNaxMinRows(experts: E, topK: K) : 9
        let useFusedMoE = Self.canFuseMoEGateUpNaxSwiglu(hal: hal, weights: w, intermediate: I)
        let canGroupedNax = supportsGroupedNaxPrefill && hal.caps.supportsSimdgroupMMA
            && L % 64 == 0 && (2 * I) % 64 == 0
            && s.expOff != nil && s.sortedPair != nil && s.sortedTok != nil
            && s.invPerm != nil && s.tileE != nil && s.tileR != nil
            && s.tileCount != nil && (useFusedMoE || s.gu16 != nil) && s.dnHi != nil
            && s.naxX16 != nil && ls.bufB16 != nil
        let isGroupedNax = T >= minGroupedRows && canGroupedNax
        let hasNaxHScratch = (s.naxX16?.shape.elementCount ?? 0) >= T * H
        let hasPrecastInput: Bool
        switch input {
        case .f16Normed:
            hasPrecastInput = true
        case .f32PreRotated(_, _, let prefillPrecast16):
            hasPrecastInput = prefillPrecast16 != nil
        }
        let hasFc1NaxInput = hasNaxHScratch || hasPrecastInput
        let canFc1PrecastStrict = T >= minNax && hasPrecastInput
            && H % 64 == 0 && L % 64 == 0 && H <= s.maxTileWidth
            && latent.fc1.n % 64 == 0 && latent.fc1.k % 64 == 0
        let canShGUPrecastStrict = ISh == 0 || w.sharedGateUp == nil
            || (T >= minNax && hasPrecastInput
                && H % 64 == 0 && (2 * ISh) % 64 == 0 && H <= s.maxTileWidth
                && (w.sharedGateUp?.n ?? 0) % 64 == 0 && (w.sharedGateUp?.k ?? 0) % 64 == 0)
        let canUseFc1NaxPrecast = T >= minNax && hasFc1NaxInput && H % 64 == 0 && L % 64 == 0 && H <= s.maxTileWidth
        let skipRot32 = canFc1PrecastStrict && canShGUPrecastStrict

        @inline(__always) func encodeDenseFallback(
            weight: QWWeightView, x: QWTensorValue, y: QWTensorValue, rows: Int,
            naxScratch: QWTensorValue? = nil, precast16: QWTensorValue? = nil
        ) throws {
            if let scratch = naxScratch, rows >= minNax {
                let naxY = s.naxY16 ?? scratch
                try QWGemm.encodeDenseProjF32(enc, hal: hal, weight: weight, x: x, y: y,
                                              naxX16: scratch, naxY16: naxY,
                                              maxTileWidth: max(weight.k, s.maxTileWidth),
                                              rows: rows, x16Precast: precast16, minNaxRows: minNax)
            } else if rows >= 32 {
                try QWGemm.encodeHQ4F32BM(enc, hal: hal, weight: weight, x: x, y: y, rows: rows)
            } else {
                try QWGemm.encodeHQ4F32V4(enc, hal: hal, weight: weight, x: x, y: y, rows: rows)
            }
        }

        let xPlain: QWTensorValue
        let xRot: QWTensorValue
        let xe16ForNax: QWTensorValue?
        switch input {
        case .f16Normed(let x16):
            (xPlain, xRot) = try Self.resolveF16NormedBridgeInputs(
                enc, hal: hal, x16: x16, scratch: s, router: w.router,
                rows: T, hidden: H, routerCols: routerCols,
                isGroupedNax: isGroupedNax, skipRot32: skipRot32)
            xe16ForNax = x16
        case .f32PreRotated(let plain, let rotated, let precast16):
            xPlain = plain
            xRot = rotated
            xe16ForNax = precast16
        }
        let fc1Precast16 = canUseFc1NaxPrecast ? xe16ForNax : nil
        let hasShNaxInput = hasNaxHScratch || xe16ForNax != nil
        let shPrecast16 = (T >= minNax && hasShNaxInput && H % 64 == 0 && (2 * ISh) % 64 == 0 && H <= s.maxTileWidth)
            ? xe16ForNax : nil
        let canNaxShDown = allowDenseNax && canUseNaxDenseProj(
            hal: hal, weight: w.sharedDown, rows: T, maxTileWidth: s.maxTileWidth, minNaxRows: minNax
        )
        let prefShAct16 = (ISh > 0 && canNaxShDown && (s.naxX16?.shape.elementCount ?? 0) >= T * (H + ISh))
            ? s.naxX16!.slice(byteOffset: T * H * 2, count: T * ISh, dtype: .f16)
            : nil
        let canAliasGuShForShAct16 = ISh > 0 && !isGroupedNax && T >= minNax && canNaxShDown
            && (s.guSh32?.bytes ?? 0) >= T * ISh * 2
        let candShAct16 = prefShAct16
            ?? (canAliasGuShForShAct16 ? s.guSh32!.slice(byteOffset: 0, count: T * ISh, dtype: .f16) : nil)
        let canFusedShNax = Self.canFuseSharedSwiGLUNax(hal: hal, weights: w, shPrecast16: shPrecast16, shAct16: candShAct16)
        let shAct16 = canFusedShNax ? candShAct16 : prefShAct16

        if isGroupedNax,
           let expOff = s.expOff, let sortedPair = s.sortedPair,
           let sortedTok = s.sortedTok, let invPerm = s.invPerm,
           let tileE = s.tileE, let tileR = s.tileR, let tileCount = s.tileCount,
           let dnHi = s.dnHi,
           let naxX16 = s.naxX16, let xLat16 = ls.bufB16 {
            let naxScratch0 = naxX16.slice(byteOffset: 0, count: T * H, dtype: .f16)
            let naxY = s.naxY16 ?? naxScratch0
            let dnAcc16 = s.dnAcc.slice(byteOffset: 0, count: T * K * L, dtype: .f16)

            if !concurrentBarriers, let nextEnc = profileStage?("moe.router") { enc = nextEnc }
            try QWGemm.encodeFnRouterGemm(enc, hal: hal, weight: w.router, x: xPlain, y: s.logits,
                                          rows: T, n: routerCols, k: H,
                                          shGateWeight: ISh > 0 ? w.sharedGateScalar : nil,
                                          shGateOut: ISh > 0 ? s.gate : nil)
            if !concurrentBarriers, let nextEnc = profileStage?("moe.latent_fc1") { enc = nextEnc }
            try QWGemm.encodeDenseProjF32(enc, hal: hal, weight: latent.fc1, x: xRot, y: ls.bufA32,
                                          naxX16: naxScratch0, naxY16: naxY, maxTileWidth: s.maxTileWidth,
                                          rows: T, x16Precast: fc1Precast16, minNaxRows: minNax)
            barrier()

            try QWGemm.encodeFnMoeRoute(enc, hal: hal, logits: s.logits, idx: s.idx, wts: s.wts,
                                        rows: T, experts: E, topK: K, normTopK: w.normTopK,
                                        bias: w.routerBias, splitFeature: w.splitFeatureRouting,
                                        routedScaleParam: w.routedScaleParam,
                                        useSigmoid: w.useSigmoidScore, nGroup: w.nGroup,
                                        topkGroup: w.topkGroup, groupTopM: w.groupTopM)
            try QWGemm.encodeLatentNormH32F32(enc, hal: hal, x: ls.bufA32, gamma: nil,
                                              out32: ls.bufB32, out16: xLat16,
                                              rows: T, latent: L, skipOut32: true)
            if !concurrentBarriers, let nextEnc = profileStage?("moe.shared_down") { enc = nextEnc }
            if let shGU = w.sharedGateUp, let shDown = w.sharedDown, ISh > 0 {
                if Self.canFuseSharedSwiGLUNax(hal: hal, weights: w, shPrecast16: shPrecast16, shAct16: shAct16),
                   let x16Sh = shPrecast16, let out16Sh = shAct16 {
                    try QWGemm.encodeDenseSwiGLUNax(enc, hal: hal, weight: shGU, x16: x16Sh, out16: out16Sh,
                                                    rows: T, intermediate: ISh)
                    barrier()
                    try QWGemm.encodeDenseProjF32(enc, hal: hal, weight: shDown, x: s.actSh32 ?? s.shared,
                                                  y: s.shared, naxX16: out16Sh, naxY16: naxY,
                                                  maxTileWidth: s.maxTileWidth,
                                                  rows: T, x16Precast: out16Sh, minNaxRows: minNax)
                } else if let guSh = s.guSh32, let actSh = s.actSh32 {
                    try QWGemm.encodeDenseProjF32(enc, hal: hal, weight: shGU, x: xRot, y: guSh,
                                                  naxX16: naxScratch0, naxY16: naxY, maxTileWidth: s.maxTileWidth,
                                                  rows: T, x16Precast: shPrecast16, minNaxRows: minNax)
                    barrier()
                    try QWGemm.encodeSiluMulF32(enc, hal: hal, gu: guSh, act: actSh, rows: T,
                                                intermediate: ISh, act16: shAct16,
                                                swigluClamp: w.sharedSwigluClamp)
                    barrier()
                    let canUseScratch0ForDown = naxScratch0.shape.elementCount >= T * ISh && naxY.shape.elementCount >= T * ISh
                    let shScratch = shAct16 ?? naxScratch0
                    let shDownMaxTile = (shAct16 != nil || canUseScratch0ForDown) ? s.maxTileWidth : 0
                    try QWGemm.encodeDenseProjF32(enc, hal: hal, weight: shDown, x: actSh, y: s.shared,
                                                  naxX16: shScratch, naxY16: naxY,
                                                  maxTileWidth: shDownMaxTile,
                                                  rows: T, x16Precast: shAct16, minNaxRows: minNax)
                }
            }
            barrier()

            guard let rotateEncoder else {
                throw QWGemmError.badShape("latent moe grouped nax requires rotateEncoder for indirect dispatch")
            }
            let tilePlan = QWGemm.MoETilePlan(experts: E, pairs: T * K, gateUpRows: 2 * I, downRows: L)
            if !concurrentBarriers, let nextEnc = profileStage?("moe.group_pairs") { enc = nextEnc }
            try QWGemm.encodeMoEGroupPairsF32(enc, hal: hal, ids: s.idx, expertOff: expOff,
                                              sortedPairIdx: sortedPair, sortedTok: sortedTok,
                                              experts: E, totalPairs: T * K, routes: K,
                                              invPerm: invPerm, tileE: tileE, tileR: tileR,
                                              tileCount: tileCount, tilePlan: tilePlan)
            try rotateEncoder(&enc)
            if !concurrentBarriers, let nextEnc = profileStage?("moe.experts_nax_gu") { enc = nextEnc }
            try Self.encodeRoutedNaxGateUpAndAct(
                enc, hal: hal, weights: w, xe: xLat16, expOff: expOff,
                tileE: tileE, tileR: tileR, tileCount: tileCount,
                gu16: s.gu16, dnHi: dnHi, intermediate: I, pairs: T * K,
                tilePlan: tilePlan, sortedTok: sortedTok, barrier: barrier)
            barrier()
            if !concurrentBarriers, let nextEnc = profileStage?("moe.experts_nax_down") { enc = nextEnc }
            try QWGemm.encodeMoENaxSplitF32(enc, hal: hal, xHi: dnHi, xLo: dnHi, bank: w.down,
                                            expertOff: expOff, tileE: tileE, tileR: tileR,
                                            tileCount: tileCount, y32: dnAcc16, rowsPerExpert: L,
                                            tilePlan: tilePlan, coldBank: w.coldDown)
            barrier()
            if !concurrentBarriers, let nextEnc = profileStage?("moe.experts_nax") { enc = nextEnc }
            try QWGemm.encodeMoEDownReduceCombineF32(enc, hal: hal, dnAcc: dnAcc16, invPerm: invPerm,
                                                     weights: s.wts, shared: s.shared, gate: s.gate,
                                                     out: ls.bufA32, hidden: L, routes: K, tokens: T,
                                                     skipShared: true)
            barrier()
            if !concurrentBarriers, let nextEnc = profileStage?("moe.latent_fc2") { enc = nextEnc }

            let useFc2Precast = T >= minNax && H % 64 == 0 && L % 64 == 0 && L <= s.maxTileWidth
            try QWGemm.encodeLatentNormH32F32(enc, hal: hal, x: ls.bufA32, gamma: latent.normGamma,
                                              out32: ls.bufB32, out16: useFc2Precast ? xLat16 : nil,
                                              rows: T, latent: L, eps: latent.normEps,
                                              skipOut32: useFc2Precast)
            barrier()
            try QWGemm.encodeDenseProjF32(enc, hal: hal, weight: latent.fc2, x: ls.bufB32, y: out32,
                                          naxX16: naxScratch0, naxY16: naxY, maxTileWidth: max(L, s.maxTileWidth),
                                          rows: T, x16Precast: useFc2Precast ? xLat16 : nil, minNaxRows: minNax)
            if ISh > 0 || out16 != nil {
                barrier()
                try QWGemm.encodeMoEDownReduceCombineF32(enc, hal: hal, dnAcc: out32, invPerm: nil,
                                                         weights: s.wts, shared: s.shared, gate: s.gate,
                                                         out: out32, hidden: H, routes: K, tokens: T,
                                                         out16: out16, skipShared: ISh == 0,
                                                         unweightedSingle: true,
                                                         skipOut32: skipOut32 && out16 != nil)
            }
            barrier()
            return
        }

        // Rank-1 GEMV / sparse high-E fallback before Grouped-NAX breaks even
        if T <= 8, let decodeRouter = w.decodeRouterOverride {
            try decodeRouter(enc, hal, w.router, xPlain, s.logits, s.gate, T, routerCols, H)
        } else {
            try QWGemm.encodeFnRouterGemm(enc, hal: hal, weight: w.router, x: xPlain, y: s.logits,
                                          rows: T, n: routerCols, k: H,
                                          shGateWeight: ISh > 0 ? w.sharedGateScalar : nil,
                                          shGateOut: ISh > 0 ? s.gate : nil)
        }
        let naxScratchH = hasNaxHScratch ? s.naxX16!.slice(byteOffset: 0, count: T * H, dtype: .f16) : nil
        try encodeDenseFallback(weight: latent.fc1, x: xRot, y: ls.bufA32, rows: T,
                                naxScratch: naxScratchH ?? fc1Precast16, precast16: fc1Precast16)
        let useFusedShFallback = Self.canFuseSharedSwiGLUNax(hal: hal, weights: w, shPrecast16: shPrecast16, shAct16: shAct16)
        let fuseSh = !useFusedShFallback && T <= 8 && ISh > 0 && ISh <= 4096 && (ISh % 8 == 0)
            && w.sharedGateUp != nil && s.guSh32 != nil && s.actSh32 != nil
        if let shGU = w.sharedGateUp, ISh > 0 {
            if useFusedShFallback, let x16Sh = shPrecast16, let out16Sh = shAct16 {
                try QWGemm.encodeDenseSwiGLUNax(enc, hal: hal, weight: shGU, x16: x16Sh, out16: out16Sh,
                                                rows: T, intermediate: ISh)
            } else if let guSh = s.guSh32 {
                try encodeDenseFallback(weight: shGU, x: xRot, y: guSh, rows: T,
                                        naxScratch: naxScratchH ?? shPrecast16, precast16: shPrecast16)
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
        try QWGemm.encodeLatentNormH32F32(enc, hal: hal, x: ls.bufA32, gamma: nil,
                                          out32: ls.bufB32, rows: T, latent: L)
        barrier()

        let fuseRoutedGU = I > 0 && (I % 32 == 0)
        if !concurrentBarriers, let nextEnc = profileStage?("moe.shared_down") { enc = nextEnc }
        if let shDown = w.sharedDown, ISh > 0 {
            if useFusedShFallback, let out16Sh = shAct16 {
                try encodeDenseFallback(weight: shDown, x: s.actSh32 ?? s.shared, y: s.shared, rows: T,
                                        naxScratch: out16Sh, precast16: out16Sh)
            } else if let guSh = s.guSh32, let actSh = s.actSh32 {
                if !fuseSh {
                    try QWGemm.encodeSiluMulF32(enc, hal: hal, gu: guSh, act: actSh, rows: T,
                                                intermediate: ISh, act16: shAct16,
                                                swigluClamp: w.sharedSwigluClamp)
                    barrier()
                }
                let canAliasShDown0 = (s.naxX16?.shape.elementCount ?? 0) >= T * ISh
                    && (s.naxY16?.shape.elementCount ?? s.naxX16?.shape.elementCount ?? 0) >= T * ISh
                    && canUseNaxDenseProj(hal: hal, weight: shDown, rows: T, maxTileWidth: s.maxTileWidth, minNaxRows: minNax)
                let shDownScratch = shAct16
                    ?? (canAliasShDown0 ? s.naxX16!.slice(byteOffset: 0, count: T * ISh, dtype: .f16) : nil)
                    ?? (canNaxShDown && guSh.bytes >= T * ISh * 2
                        ? guSh.slice(byteOffset: 0, count: T * ISh, dtype: .f16)
                        : nil)
                try encodeDenseFallback(weight: shDown, x: actSh, y: s.shared, rows: T,
                                        naxScratch: shDownScratch, precast16: fuseSh ? nil : shAct16)
            }
        }
        let actForCompact: QWTensorValue? = (s.act32.shape.elementCount >= neededAct32)
            ? s.act32
            : s.dnHi.flatMap { $0.bytes >= neededAct32 * 4 ? $0.slice(byteOffset: 0, count: neededAct32, dtype: .f32) : nil }
        let canCompactQWP = T >= 2 && T <= 16
            && QWGemm.supportsCompactQWP(gateUp: w.gateUp, down: w.down)
            && E >= 1 && E <= 1024 && L > 0 && L % 32 == 0 && I > 0 && I % 32 == 0
            && w.gateUp.k == L && w.down.k == I
            && (w.coldGateUp.map { $0.containerKind == w.gateUp.containerKind && $0.k == w.gateUp.k } ?? true)
            && (w.coldDown.map { $0.containerKind == w.down.containerKind && $0.k == w.down.k } ?? true)
            && actForCompact != nil
            && s.dnAcc.shape.elementCount >= T * K * L
        if canCompactQWP,
           let actBuf = actForCompact,
           let expOff = s.expOff, let sortedPair = s.sortedPair,
           let sortedTok = s.sortedTok, let actExp = s.activeExperts,
           let actInd = s.activeIndirect,
           expOff.shape.elementCount >= E + 1,
           actExp.shape.elementCount >= E,
           actInd.bytes >= QWGemmFP.MoEActiveBucketPlan.indirectBufferBytes,
           sortedPair.shape.elementCount >= T * K,
           sortedTok.shape.elementCount >= T * K {
            if !concurrentBarriers, let nextEnc = profileStage?("moe.experts_gate_up") { enc = nextEnc }
            try QWGemm.encodeMoEQWPCompactRoutedCombine(
                enc, hal: hal,
                x: ls.bufB32, gateUp: w.gateUp, down: w.down,
                ids: s.idx, weights: s.wts,
                shared: s.shared, gate: s.gate, out: ls.bufA32,
                expertOff: expOff, activeExperts: actExp, activeIndirect: actInd,
                sortedPairIdx: sortedPair, sortedTok: sortedTok,
                actBuf: actBuf, dnAccBuf: s.dnAcc,
                hidden: L, experts: E, routes: K, intermediate: I, tokens: T,
                coldGateUp: w.coldGateUp, coldDown: w.coldDown,
                gateSoftcap: w.gateSoftcap, upSoftcap: w.upSoftcap,
                skipShared: true,
                concurrentBarriers: concurrentBarriers,
                nofmaShared: true,
                swigluClamp: w.routedSwigluClamp)
            barrier()
        } else {
            let guBuf32: QWTensorValue = s.gu32.shape.elementCount >= neededGU32
                ? s.gu32
                : s.gu16!.slice(byteOffset: 0, count: neededGU32, dtype: .f32)
            let actBuf32: QWTensorValue = s.act32.shape.elementCount >= neededAct32
                ? s.act32
                : s.dnHi!.slice(byteOffset: 0, count: neededAct32, dtype: .f32)
            if !concurrentBarriers, let nextEnc = profileStage?("moe.experts_gate_up") { enc = nextEnc }
            try QWGemm.encodeMoEGateUpF32R1(enc, hal: hal, x: ls.bufB32, bank: w.gateUp, ids: s.idx,
                                            out: fuseRoutedGU ? actBuf32 : guBuf32,
                                            rowsPerExpert: 2 * I, routes: K, tokens: T,
                                            fuseSiluFwht: fuseRoutedGU,
                                            coldBank: w.coldGateUp,
                                            reuseExpertsAcrossTokens: w.reuseExpertsAcrossTokens,
                                            swigluClamp: w.routedSwigluClamp)
            barrier()
            if !fuseRoutedGU {
                try QWGemm.encodeSiluMulF32(enc, hal: hal, gu: guBuf32, act: actBuf32, rows: T * K,
                                            intermediate: I, swigluClamp: w.routedSwigluClamp)
                barrier()
            }
            if !concurrentBarriers, let nextEnc = profileStage?("moe.experts_down") { enc = nextEnc }
            try QWGemm.encodeMoEDownSharedCombineF32(enc, hal: hal, x: actBuf32, bank: w.down,
                                                     rowsPerRoute: L, routes: K, weights: s.wts,
                                                     shared: s.shared, gate: s.gate, out: ls.bufA32,
                                                     ids: s.idx, tokens: T,
                                                     coldBank: w.coldDown,
                                                     reuseExpertsAcrossTokens: w.reuseExpertsAcrossTokens,
                                                     skipShared: true)
            barrier()
        }

        let canFc2Nax = T >= minNax && H % 64 == 0 && L % 64 == 0 && L <= s.maxTileWidth
        let fc2PrecastBuf: QWTensorValue? = canFc2Nax
            ? (ls.bufB16 ?? (s.dnAcc.bytes >= T * L * 2 ? s.dnAcc.slice(byteOffset: 0, count: T * L, dtype: .f16) : nil))
            : nil
        let useFc2Precast = fc2PrecastBuf != nil
        try QWGemm.encodeLatentNormH32F32(enc, hal: hal, x: ls.bufA32, gamma: latent.normGamma,
                                          out32: ls.bufB32, out16: fc2PrecastBuf,
                                          rows: T, latent: L, eps: latent.normEps,
                                          skipOut32: useFc2Precast)
        barrier()
        if !concurrentBarriers, let nextEnc = profileStage?("moe.latent_fc2") { enc = nextEnc }
        let fc2Scratch = fc2PrecastBuf
            ?? ((s.naxX16?.shape.elementCount ?? 0) >= T * L ? s.naxX16!.slice(byteOffset: 0, count: T * L, dtype: .f16) : nil)
        try encodeDenseFallback(weight: latent.fc2, x: ls.bufB32, y: out32, rows: T,
                                naxScratch: fc2Scratch, precast16: fc2PrecastBuf)
        if ISh > 0 || out16 != nil {
            barrier()
            try QWGemm.encodeMoEDownReduceCombineF32(enc, hal: hal, dnAcc: out32, invPerm: nil,
                                                     weights: s.wts, shared: s.shared, gate: s.gate,
                                                     out: out32, hidden: H, routes: K, tokens: T,
                                                     out16: out16, skipShared: ISh == 0,
                                                     unweightedSingle: true,
                                                     skipOut32: skipOut32 && out16 != nil)
        }
        barrier()
    }
}
