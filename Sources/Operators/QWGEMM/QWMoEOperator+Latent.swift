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
        let allowDenseNax = supportsGroupedNaxPrefill && hal.caps.isMetal4TensorCapable
        let minNax = allowDenseNax ? 16 : Int.max
        let neededGU32 = T * K * 2 * I
        let neededAct32 = T * K * I
        let hasFallbackF32Scratch = T <= 8
            || (s.gu32.shape.elementCount >= neededGU32 && s.act32.shape.elementCount >= neededAct32)
            || ((s.gu16?.bytes ?? 0) >= neededGU32 * 4 && (s.dnHi?.bytes ?? 0) >= neededAct32 * 4)
        let minGroupedRows = hasFallbackF32Scratch ? Self.groupedNaxMinRows(experts: E, topK: K) : 9
        let canGroupedNax = supportsGroupedNaxPrefill && hal.caps.isMetal4TensorCapable
            && L % 64 == 0 && (2 * I) % 64 == 0
            && s.expOff != nil && s.sortedPair != nil && s.sortedTok != nil
            && s.invPerm != nil && s.tileE != nil && s.tileR != nil
            && s.tileCount != nil && s.gu16 != nil && s.dnHi != nil
            && s.naxX16 != nil && ls.bufB16 != nil
        let isGroupedNax = T >= minGroupedRows && canGroupedNax
        let hasNaxHScratch = (s.naxX16?.shape.elementCount ?? 0) >= T * H
        let canUseFc1NaxPrecast = T >= minNax && hasNaxHScratch && H % 64 == 0 && L % 64 == 0 && H <= s.maxTileWidth
        let skipRot32 = (isGroupedNax || T >= minNax) && canUseFc1NaxPrecast && (ISh == 0 || (2 * ISh) % 64 == 0)

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
            try QWGemm.encodeMoEBridgeF16ToF32(enc, hal: hal, x: x16,
                                               xRot32: s.xRot32, xPlain32: s.xPlain32,
                                               rows: T, hidden: H, skipRot32: skipRot32)
            xPlain = s.xPlain32
            xRot = s.xRot32
            xe16ForNax = x16
        case .f32PreRotated(let plain, let rotated, let precast16):
            xPlain = plain
            xRot = rotated
            xe16ForNax = precast16
        }
        let fc1Precast16 = canUseFc1NaxPrecast ? xe16ForNax : nil
        let shPrecast16 = (T >= minNax && hasNaxHScratch && H % 64 == 0 && (2 * ISh) % 64 == 0 && H <= s.maxTileWidth)
            ? xe16ForNax : nil
        let shAct16 = (T >= minNax && ISh > 0 && ISh % 64 == 0 && (s.naxX16?.shape.elementCount ?? 0) >= T * (H + ISh))
            ? s.naxX16!.slice(byteOffset: T * H * 2, count: T * ISh, dtype: .f16)
            : nil

        if isGroupedNax,
           let expOff = s.expOff, let sortedPair = s.sortedPair,
           let sortedTok = s.sortedTok, let invPerm = s.invPerm,
           let tileE = s.tileE, let tileR = s.tileR, let tileCount = s.tileCount,
           let gu16 = s.gu16, let dnHi = s.dnHi,
           let naxX16 = s.naxX16, let xLat16 = ls.bufB16 {
            let naxScratch0 = naxX16.slice(byteOffset: 0, count: T * H, dtype: .f16)
            let naxY = s.naxY16 ?? naxScratch0
            let dnAcc16 = s.dnAcc.slice(byteOffset: 0, count: T * K * L, dtype: .f16)

            try QWGemm.encodeFnRouterGemm(enc, hal: hal, weight: w.router, x: xPlain, y: s.logits,
                                          rows: T, n: routerCols, k: H,
                                          shGateWeight: ISh > 0 ? w.sharedGateScalar : nil,
                                          shGateOut: ISh > 0 ? s.gate : nil)
            try QWGemm.encodeDenseProjF32(enc, hal: hal, weight: latent.fc1, x: xRot, y: ls.bufA32,
                                          naxX16: naxScratch0, naxY16: naxY, maxTileWidth: s.maxTileWidth,
                                          rows: T, x16Precast: fc1Precast16, minNaxRows: minNax)
            barrier()

            try QWGemm.encodeFnMoeRoute(enc, hal: hal, logits: s.logits, idx: s.idx, wts: s.wts,
                                        rows: T, experts: E, topK: K, normTopK: w.normTopK,
                                        bias: w.routerBias, splitFeature: w.splitFeatureRouting,
                                        routedScaleParam: w.routedScaleParam)
            try QWGemm.encodeLatentNormH32F32(enc, hal: hal, x: ls.bufA32, gamma: nil,
                                              out32: ls.bufB32, out16: xLat16,
                                              rows: T, latent: L, skipOut32: true)
            if let shGU = w.sharedGateUp, let shDown = w.sharedDown,
               let guSh = s.guSh32, let actSh = s.actSh32, ISh > 0 {
                try QWGemm.encodeDenseProjF32(enc, hal: hal, weight: shGU, x: xRot, y: guSh,
                                              naxX16: naxScratch0, naxY16: naxY, maxTileWidth: max(H, s.maxTileWidth),
                                              rows: T, x16Precast: shPrecast16, minNaxRows: minNax)
                barrier()
                try QWGemm.encodeSiluMulF32(enc, hal: hal, gu: guSh, act: actSh, rows: T,
                                            intermediate: ISh, act16: shAct16)
                barrier()
                let shScratch = shAct16 ?? naxScratch0
                try QWGemm.encodeDenseProjF32(enc, hal: hal, weight: shDown, x: actSh, y: s.shared,
                                              naxX16: shScratch, naxY16: naxY,
                                              maxTileWidth: max(max(H, ISh), s.maxTileWidth),
                                              rows: T, x16Precast: shAct16, minNaxRows: minNax)
            }
            barrier()

            guard let rotateEncoder else {
                throw QWGemmError.badShape("latent moe grouped nax requires rotateEncoder for indirect dispatch")
            }
            let tilePlan = QWGemm.MoETilePlan(experts: E, pairs: T * K, gateUpRows: 2 * I, downRows: L)
            try QWGemm.encodeMoEGroupPairsF32(enc, hal: hal, ids: s.idx, expertOff: expOff,
                                              sortedPairIdx: sortedPair, sortedTok: sortedTok,
                                              experts: E, totalPairs: T * K, routes: K,
                                              invPerm: invPerm, tileE: tileE, tileR: tileR,
                                              tileCount: tileCount, tilePlan: tilePlan)
            try rotateEncoder(&enc)
            try QWGemm.encodeMoEGateUpNax(enc, hal: hal, xe: xLat16, bank: w.gateUp, expertOff: expOff,
                                          tileE: tileE, tileR: tileR, tileCount: tileCount,
                                          gu16: gu16, rowsPerExpert: 2 * I, tilePlan: tilePlan,
                                          sortedTok: sortedTok, coldBank: w.coldGateUp)
            barrier()
            try QWGemm.encodeSiluMulSplitF16(enc, hal: hal, gu: gu16, outHi: dnHi, outLo: dnHi,
                                             rows: T * K, intermediate: I)
            barrier()
            try QWGemm.encodeMoENaxSplitF32(enc, hal: hal, xHi: dnHi, xLo: dnHi, bank: w.down,
                                            expertOff: expOff, tileE: tileE, tileR: tileR,
                                            tileCount: tileCount, y32: dnAcc16, rowsPerExpert: L,
                                            tilePlan: tilePlan, coldBank: w.coldDown)
            barrier()
            try QWGemm.encodeMoEDownReduceCombineF32(enc, hal: hal, dnAcc: dnAcc16, invPerm: invPerm,
                                                     weights: s.wts, shared: s.shared, gate: s.gate,
                                                     out: ls.bufA32, hidden: L, routes: K, tokens: T,
                                                     skipShared: true)
            barrier()

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
                                                         unweightedSingle: true)
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
                                naxScratch: naxScratchH, precast16: fc1Precast16)
        let fuseSh = T <= 8 && ISh > 0 && ISh <= 4096 && w.sharedGateUp != nil && s.guSh32 != nil && s.actSh32 != nil
        if let shGU = w.sharedGateUp, let guSh = s.guSh32, ISh > 0 {
            try encodeDenseFallback(weight: shGU, x: xRot, y: guSh, rows: T,
                                    naxScratch: naxScratchH, precast16: shPrecast16)
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
                                    routedScaleParam: w.routedScaleParam)
        try QWGemm.encodeLatentNormH32F32(enc, hal: hal, x: ls.bufA32, gamma: nil,
                                          out32: ls.bufB32, rows: T, latent: L)
        barrier()

        let fuseRoutedGU = I > 0 && (I % 32 == 0)
        if !concurrentBarriers, let nextEnc = profileStage?("moe.shared_down") { enc = nextEnc }
        if let shDown = w.sharedDown, let guSh = s.guSh32, let actSh = s.actSh32, ISh > 0 {
            if !fuseSh {
                try QWGemm.encodeSiluMulF32(enc, hal: hal, gu: guSh, act: actSh, rows: T,
                                            intermediate: ISh, act16: shAct16)
                barrier()
            }
            let shDownScratch = shAct16
                ?? ((s.naxX16?.shape.elementCount ?? 0) >= T * ISh ? s.naxX16!.slice(byteOffset: 0, count: T * ISh, dtype: .f16) : nil)
            try encodeDenseFallback(weight: shDown, x: actSh, y: s.shared, rows: T,
                                    naxScratch: shDownScratch, precast16: shAct16)
        }
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
                                        reuseExpertsAcrossTokens: w.reuseExpertsAcrossTokens)
        barrier()
        if !fuseRoutedGU {
            try QWGemm.encodeSiluMulF32(enc, hal: hal, gu: guBuf32, act: actBuf32, rows: T * K, intermediate: I)
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

        let useFc2Precast = T >= minNax && H % 64 == 0 && L % 64 == 0 && L <= s.maxTileWidth && ls.bufB16 != nil
        try QWGemm.encodeLatentNormH32F32(enc, hal: hal, x: ls.bufA32, gamma: latent.normGamma,
                                          out32: ls.bufB32, out16: useFc2Precast ? ls.bufB16 : nil,
                                          rows: T, latent: L, eps: latent.normEps,
                                          skipOut32: useFc2Precast)
        barrier()
        if !concurrentBarriers, let nextEnc = profileStage?("moe.latent_fc2") { enc = nextEnc }
        let fc2Scratch = useFc2Precast
            ? ls.bufB16
            : ((s.naxX16?.shape.elementCount ?? 0) >= T * L ? s.naxX16!.slice(byteOffset: 0, count: T * L, dtype: .f16) : nil)
        try encodeDenseFallback(weight: latent.fc2, x: ls.bufB32, y: out32, rows: T,
                                naxScratch: fc2Scratch, precast16: useFc2Precast ? ls.bufB16 : nil)
        if ISh > 0 || out16 != nil {
            barrier()
            try QWGemm.encodeMoEDownReduceCombineF32(enc, hal: hal, dnAcc: out32, invPerm: nil,
                                                     weights: s.wts, shared: s.shared, gate: s.gate,
                                                     out: out32, hidden: H, routes: K, tokens: T,
                                                     out16: out16, skipShared: ISh == 0,
                                                     unweightedSingle: true)
        }
        barrier()
    }
}
