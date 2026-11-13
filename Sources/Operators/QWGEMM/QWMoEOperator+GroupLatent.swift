// QWMoEOperator+GroupLatent.swift — Multi-group latent Sparse-MoE execution (`latentD[NG] -> experts -> latentU[NG]`).
// Serves grouped-latent MoE layers (`numLatentGroups > 0`) for Grouped-NAX prefill (`T > 8`) and GEMV decode (`T <= 8`).

import Foundation
import Metal
import QWAlloc
import QWCore
import QWHAL

extension QWMoEOperator {
    static func encodeResidentGroupLatent(
        _ encRef: inout any MTLComputeCommandEncoder,
        hal: QWHal,
        weights w: Weights,
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

        @inline(__always) func barrier() {
            if concurrentBarriers { enc.memoryBarrier(scope: .buffers) }
        }
        let H = w.hidden, E = w.experts, K = w.topK, I = w.intermediate, ISh = w.sharedIntermediate
        let NG = w.numLatentGroups, L = w.latentDim
        let routerCols = w.splitFeatureRouting ? (2 * E) : E
        guard let latentD = w.latentD, let latentU = w.latentU, let latentLabels = w.latentLabels,
              let groupIds = s.groupIds, let onesRoute = s.onesRoute,
              let xLatent32 = s.xLatent32, let zLatent32 = s.zLatent32 else {
            throw QWGemmError.badShape("group-latent moe requires latentD, latentU, latentLabels, and latent scratch")
        }
        let isGroupedNax = T > 8 && supportsGroupedNaxPrefill && hal.caps.isMetal4TensorCapable
            && s.expOff != nil && s.sortedPair != nil && s.sortedTok != nil
            && s.invPerm != nil && s.tileE != nil && s.tileR != nil
            && s.tileCount != nil && s.gu16 != nil && s.dnHi != nil && s.naxX16 != nil
            && s.xLatent16 != nil && L % 64 == 0 && (NG * L) % 64 == 0 && H % 64 == 0
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
            try QWGemm.encodeMoEBridgeF16ToF32(enc, hal: hal, x: x16,
                                               xRot32: s.xRot32, xPlain32: s.xPlain32,
                                               rows: T, hidden: H, skipRot32: skipRot32)
            xPlain = s.xPlain32
            xRot = s.xRot32
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
           let gu16 = s.gu16, let dnHi = s.dnHi,
           let naxX16 = s.naxX16, let xLatent16 = s.xLatent16, let xe16 = xe16ForNax {
            let shAct16 = (T >= 64 && ISh > 0 && ISh % 64 == 0)
                ? naxX16.slice(byteOffset: T * H * 2, count: T * ISh, dtype: .f16)
                : nil
            let dnAcc16 = s.dnAcc.slice(byteOffset: 0, count: T * max(K, NG) * H, dtype: .f16)
            try QWGemm.encodeFnRouterGemm(enc, hal: hal, weight: w.router, x: xPlain, y: s.logits,
                                          rows: T, n: routerCols, k: H,
                                          shGateWeight: ISh > 0 ? w.sharedGateScalar : nil,
                                          shGateOut: ISh > 0 ? s.gate : nil)
            if concurrentBarriers, let shGU = w.sharedGateUp, let guSh = s.guSh32, ISh > 0 {
                let naxY = s.naxY16 ?? naxX16
                try QWGemm.encodeDenseProjF32(enc, hal: hal, weight: shGU, x: xRot, y: guSh,
                                              naxX16: naxX16, naxY16: naxY, maxTileWidth: s.maxTileWidth,
                                              rows: T, x16Precast: shPrecast16)
            }
            barrier()
            try QWGemm.encodeFnMoeRoute(enc, hal: hal, logits: s.logits, idx: s.idx, wts: s.wts,
                                        rows: T, experts: E, topK: K, normTopK: w.normTopK, bias: w.routerBias,
                                        splitFeature: w.splitFeatureRouting, routedScaleParam: w.routedScaleParam,
                                        latentLabels: latentLabels, groupIds: groupIds, numGroups: NG)
            if !concurrentBarriers, let shGU = w.sharedGateUp, let shDown = w.sharedDown,
               let guSh = s.guSh32, let actSh = s.actSh32, ISh > 0 {
                let shScratch0 = naxX16.slice(byteOffset: 0, count: T * H, dtype: .f16)
                try QWGemm.encodeDenseProjF32(enc, hal: hal, weight: shGU, x: xRot, y: guSh,
                                              naxX16: shScratch0, naxY16: shScratch0, maxTileWidth: H,
                                              rows: T, x16Precast: shPrecast16)
                try QWGemm.encodeSiluMulF32(enc, hal: hal, gu: guSh, act: actSh, rows: T,
                                            intermediate: ISh, act16: shAct16)
                let shScratch = shAct16 ?? shScratch0
                try QWGemm.encodeDenseProjF32(enc, hal: hal, weight: shDown, x: actSh, y: s.shared,
                                              naxX16: shScratch, naxY16: shScratch, maxTileWidth: max(H, ISh),
                                              rows: T, x16Precast: shAct16)
            } else if concurrentBarriers, let guSh = s.guSh32, let actSh = s.actSh32, ISh > 0 {
                try QWGemm.encodeSiluMulF32(enc, hal: hal, gu: guSh, act: actSh, rows: T,
                                            intermediate: ISh, act16: shAct16)
            }
            if needsCastXe16 {
                try QWGemm.encodeCast(enc, hal: hal, x32: xRot, y16: naxX16, count: T * H)
            }
            barrier()
            guard let rotateEncoder else {
                throw QWGemmError.badShape("moe grouped nax requires rotateEncoder for indirect dispatch")
            }
            let stage1Plan = QWGemm.MoETilePlan(experts: NG, pairs: T * NG, gateUpRows: L, downRows: 64)
            try QWGemm.encodeMoEGroupPairsF32(enc, hal: hal, ids: groupIds, expertOff: expOff,
                                              sortedPairIdx: sortedPair, sortedTok: sortedTok,
                                              experts: NG, totalPairs: T * NG, routes: NG,
                                              invPerm: invPerm, tileE: tileE, tileR: tileR,
                                              tileCount: tileCount, tilePlan: stage1Plan,
                                              directSlotTok: false)
            if concurrentBarriers, let shDown = w.sharedDown, let actSh = s.actSh32, ISh > 0 {
                let naxY = s.naxY16 ?? naxX16
                try QWGemm.encodeDenseProjF32(enc, hal: hal, weight: shDown, x: actSh, y: s.shared,
                                              naxX16: naxX16, naxY16: naxY, maxTileWidth: s.maxTileWidth,
                                              rows: T, x16Precast: shAct16)
            }
            try rotateEncoder(&enc)
            try QWGemm.encodeMoEGateUpNax(enc, hal: hal, xe: xe16, bank: latentD, expertOff: expOff,
                                          tileE: tileE, tileR: tileR, tileCount: tileCount,
                                          gu16: dnAcc16, rowsPerExpert: L, tilePlan: stage1Plan,
                                          sortedTok: sortedTok)
            barrier()
            try QWGemm.encodeMoEDownReduceCombineF32(enc, hal: hal, dnAcc: dnAcc16, invPerm: invPerm,
                                                     weights: onesRoute, shared: s.shared, gate: s.gate,
                                                     out: xLatent32, hidden: L, routes: NG, tokens: T,
                                                     out16: xLatent16, ids: groupIds,
                                                     scatterPerRoute: true, skipOut32: true)
            barrier()
            let tilePlan = QWGemm.MoETilePlan(experts: E, pairs: T * K, gateUpRows: 2 * I, downRows: L)
            try QWGemm.encodeMoEGroupPairsF32(enc, hal: hal, ids: s.idx, expertOff: expOff,
                                              sortedPairIdx: sortedPair, sortedTok: sortedTok,
                                              experts: E, totalPairs: T * K, routes: K,
                                              invPerm: invPerm, tileE: tileE, tileR: tileR,
                                              tileCount: tileCount, tilePlan: tilePlan)
            try rotateEncoder(&enc)
            try QWGemm.encodeMoEGateUpNax(enc, hal: hal, xe: xLatent16, bank: w.gateUp, expertOff: expOff,
                                          tileE: tileE, tileR: tileR, tileCount: tileCount,
                                          gu16: gu16, rowsPerExpert: 2 * I, tilePlan: tilePlan,
                                          sortedTok: sortedTok, coldBank: w.coldGateUp)
            barrier()
            try QWGemm.encodeSiluMulSplitF16(enc, hal: hal, gu: gu16, outHi: dnHi, outLo: dnHi,
                                             rows: T * K, intermediate: I)
            barrier()
            try QWGemm.encodeMoENaxSplitF32(enc, hal: hal, xHi: dnHi, xLo: dnHi, bank: w.down,
                                            expertOff: expOff, tileE: tileE, tileR: tileR,
                                            tileCount: tileCount, y32: zLatent32, rowsPerExpert: L,
                                            tilePlan: tilePlan, coldBank: w.coldDown)
            barrier()
            try QWGemm.encodeMoEDownReduceCombineF32(enc, hal: hal, dnAcc: zLatent32, invPerm: invPerm,
                                                     weights: s.wts, shared: s.shared, gate: s.gate,
                                                     out: xLatent32, hidden: L, routes: K, tokens: T,
                                                     out16: xLatent16, ids: s.idx, numGroups: NG)
            barrier()
            let stage4Plan = QWGemm.MoETilePlan(experts: NG, pairs: T * NG, gateUpRows: H, downRows: 64)
            try QWGemm.encodeMoEGroupPairsF32(enc, hal: hal, ids: groupIds, expertOff: expOff,
                                              sortedPairIdx: sortedPair, sortedTok: sortedTok,
                                              experts: NG, totalPairs: T * NG, routes: NG,
                                              invPerm: invPerm, tileE: tileE, tileR: tileR,
                                              tileCount: tileCount, tilePlan: stage4Plan,
                                              directSlotTok: true)
            try rotateEncoder(&enc)
            try QWGemm.encodeMoEGateUpNax(enc, hal: hal, xe: xLatent16, bank: latentU, expertOff: expOff,
                                          tileE: tileE, tileR: tileR, tileCount: tileCount,
                                          gu16: dnAcc16, rowsPerExpert: H, tilePlan: stage4Plan,
                                          sortedTok: sortedTok)
            barrier()
            try QWGemm.encodeMoEDownReduceCombineF32(enc, hal: hal, dnAcc: dnAcc16, invPerm: invPerm,
                                                     weights: onesRoute, shared: s.shared, gate: s.gate,
                                                     out: out32, hidden: H, routes: NG, tokens: T,
                                                     out16: out16, skipShared: ISh == 0,
                                                     ids: groupIds, numGroups: 0)
            barrier()
            return
        }

        if T <= 8, let decodeRouter = w.decodeRouterOverride {
            try decodeRouter(enc, hal, w.router, xPlain, s.logits, s.gate, T, routerCols, H)
        } else {
            try QWGemm.encodeFnRouterGemm(enc, hal: hal, weight: w.router, x: xPlain, y: s.logits,
                                          rows: T, n: routerCols, k: H,
                                          shGateWeight: ISh > 0 ? w.sharedGateScalar : nil,
                                          shGateOut: ISh > 0 ? s.gate : nil)
        }
        let fuseSh = ISh > 0 && ISh <= 4096 && w.sharedGateUp != nil && s.guSh32 != nil && s.actSh32 != nil
        if let shGU = w.sharedGateUp, let guSh = s.guSh32, ISh > 0 {
            try QWGemm.encodeHQ4F32V4(enc, hal: hal, weight: shGU, x: xRot, y: guSh, rows: T)
        }
        barrier()
        try QWGemm.encodeFnMoeRoute(enc, hal: hal, logits: s.logits, idx: s.idx, wts: s.wts,
                                    rows: T, experts: E, topK: K, normTopK: w.normTopK,
                                    guSh: fuseSh ? s.guSh32 : nil,
                                    actSh: fuseSh ? s.actSh32 : nil,
                                    sharedIntermediate: fuseSh ? ISh : 0,
                                    bias: w.routerBias, expertBudget: w.moeSpecExpertBudget,
                                    splitFeature: w.splitFeatureRouting, routedScaleParam: w.routedScaleParam,
                                    latentLabels: latentLabels, groupIds: groupIds, numGroups: NG)
        barrier()
        if !concurrentBarriers, let nextEnc = profileStage?("moe.latent_down") { enc = nextEnc }
        try QWGemm.encodeMoEDownSharedCombineF32(enc, hal: hal, x: xRot, bank: latentD,
                                                 rowsPerRoute: L, routes: NG, weights: onesRoute,
                                                 shared: s.shared, gate: s.gate, out: xLatent32,
                                                 ids: groupIds, tokens: T, routeOutput: true,
                                                 reuseExpertsAcrossTokens: w.reuseExpertsAcrossTokens,
                                                 skipShared: true, numGroups: 0, tokenIndexedX: true)
        barrier()
        let fuseRoutedGU = I > 0 && (I % 32 == 0)
        if concurrentBarriers {
            try QWGemm.encodeMoEGateUpF32R1(enc, hal: hal, x: xLatent32, bank: w.gateUp, ids: s.idx,
                                            out: fuseRoutedGU ? s.act32 : s.gu32,
                                            rowsPerExpert: 2 * I, routes: K, tokens: T,
                                            fuseSiluFwht: fuseRoutedGU, coldBank: w.coldGateUp,
                                            reuseExpertsAcrossTokens: false)
            if let shDown = w.sharedDown, let guSh = s.guSh32, let actSh = s.actSh32, ISh > 0 {
                if !fuseSh {
                    try QWGemm.encodeSiluMulF32(enc, hal: hal, gu: guSh, act: actSh, rows: T, intermediate: ISh)
                    barrier()
                }
                try QWGemm.encodeHQ4F32V4(enc, hal: hal, weight: shDown, x: actSh, y: s.shared, rows: T)
            }
            barrier()
            if !fuseRoutedGU {
                try QWGemm.encodeSiluMulF32(enc, hal: hal, gu: s.gu32, act: s.act32, rows: T * K, intermediate: I)
                barrier()
            }
            try QWGemm.encodeMoEDownSharedCombineF32(enc, hal: hal, x: s.act32, bank: w.down,
                                                     rowsPerRoute: L, routes: K, weights: s.wts,
                                                     shared: s.shared, gate: s.gate, out: zLatent32,
                                                     ids: s.idx, tokens: T, out16: nil,
                                                     coldBank: w.coldDown, reuseExpertsAcrossTokens: false,
                                                     numGroups: NG)
            barrier()
            try QWGemm.encodeMoEDownSharedCombineF32(enc, hal: hal, x: zLatent32, bank: latentU,
                                                     rowsPerRoute: H, routes: NG, weights: onesRoute,
                                                     shared: s.shared, gate: s.gate, out: out32,
                                                     ids: groupIds, tokens: T, out16: out16,
                                                     reuseExpertsAcrossTokens: w.reuseExpertsAcrossTokens,
                                                     skipShared: ISh == 0, numGroups: 0)
            barrier()
        } else {
            if let nextEnc = profileStage?("moe.shared_down") { enc = nextEnc }
            if let shDown = w.sharedDown, let guSh = s.guSh32, let actSh = s.actSh32, ISh > 0 {
                if !fuseSh {
                    try QWGemm.encodeSiluMulF32(enc, hal: hal, gu: guSh, act: actSh, rows: T, intermediate: ISh)
                }
                try QWGemm.encodeHQ4F32V4(enc, hal: hal, weight: shDown, x: actSh, y: s.shared, rows: T)
            }
            if let nextEnc = profileStage?("moe.experts_gate_up") { enc = nextEnc }
            try QWGemm.encodeMoEGateUpF32R1(enc, hal: hal, x: xLatent32, bank: w.gateUp, ids: s.idx,
                                            out: fuseRoutedGU ? s.act32 : s.gu32,
                                            rowsPerExpert: 2 * I, routes: K, tokens: T,
                                            fuseSiluFwht: fuseRoutedGU, coldBank: w.coldGateUp,
                                            reuseExpertsAcrossTokens: false)
            if !fuseRoutedGU {
                try QWGemm.encodeSiluMulF32(enc, hal: hal, gu: s.gu32, act: s.act32, rows: T * K, intermediate: I)
            }
            if let nextEnc = profileStage?("moe.experts_down") { enc = nextEnc }
            try QWGemm.encodeMoEDownSharedCombineF32(enc, hal: hal, x: s.act32, bank: w.down,
                                                     rowsPerRoute: L, routes: K, weights: s.wts,
                                                     shared: s.shared, gate: s.gate, out: zLatent32,
                                                     ids: s.idx, tokens: T, out16: nil,
                                                     coldBank: w.coldDown, reuseExpertsAcrossTokens: false,
                                                     numGroups: NG)
            if let nextEnc = profileStage?("moe.latent_up") { enc = nextEnc }
            try QWGemm.encodeMoEDownSharedCombineF32(enc, hal: hal, x: zLatent32, bank: latentU,
                                                     rowsPerRoute: H, routes: NG, weights: onesRoute,
                                                     shared: s.shared, gate: s.gate, out: out32,
                                                     ids: groupIds, tokens: T, out16: out16,
                                                     reuseExpertsAcrossTokens: w.reuseExpertsAcrossTokens,
                                                     skipShared: ISh == 0, numGroups: 0)
        }
    }
}
