// QWMoELayerKernel.swift — The sparse-MoE layer: the dense attention, then
// the routed experts.
//
// The attention is the dense full-attention kernel's own — composition, not
// a copy — so a MoE layer cannot drift from the dense layer it shares its
// tail with. Only the MLP differs.

import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWGEMM

/// The sparse-MoE chain: the normed hidden → the routed experts' weighted
/// sum. ONE body for a decode step and a prefill chunk — only the row count
/// differs.
enum QWMoEChain {
    /// Encode the chain for `rows` rows. `x16` is the f16 normed hidden; the
    /// result lands in `dn32` (f32) and `dn16` (f16) — the layer's MLP slot.
    static let coldTimingProxy = ProcessInfo.processInfo.environment["QW_COLD_TIMING_PROXY"] == "1"

    static func encode(enc: inout any MTLComputeCommandEncoder, hal: QWHal,
                       moe: QWEngineDriver.MoEW, s: QWEngineDriver.MoEScratch,
                       x16: QWTensorValue, dn32: QWTensorValue, dn16: QWTensorValue,
                       rows: Int, geometry: QWModelGeometry,
                       driver: QWEngineDriver? = nil,
                       cb: (any MTLCommandBuffer)? = nil,
                       skipOut32: Bool = false,
                       preBridgedF32: (plain: QWTensorValue, rotated: QWTensorValue)? = nil,
                       isVerifyWindow: Bool = false) throws {
        // QW_COLD_TIMING_PROXY=1 (benchmark only): skip SSD staging and read cold ids from the
        // stale cold bank (experts - K slots, so ids stay in bounds). Same bytes and dispatches
        // as an all-resident pack; the output is not meaningful.
        let streamer = Self.coldTimingProxy ? nil : moe.coldStreamer
        if streamer == nil {
            let weights = QWMoEOperator.Weights(
                router: moe.router,
                routerBias: moe.routerBias,
                routedScaleParam: moe.routedScaleParam,
                splitFeatureRouting: moe.splitFeatureRouting,
                gateUp: moe.gateUp,
                down: moe.down,
                coldGateUp: moe.coldGateUp,
                coldDown: moe.coldDown,
                sharedGateUp: moe.sharedGateUp,
                sharedDown: moe.sharedDown,
                sharedGateScalar: moe.sharedGateScalar,
                latent: moe.latent,
                hidden: geometry.hidden,
                experts: geometry.numExperts,
                topK: geometry.expertsPerTok,
                intermediate: geometry.moeIntermediate,
                sharedIntermediate: geometry.sharedExpertIntermediate,
                normTopK: geometry.normTopKProb,
                reuseExpertsAcrossTokens: true,
                moeSpecExpertBudget: isVerifyWindow ? (driver?.activeMoESpecExpertBudget(rows: rows) ?? 0) : 0,
                useSigmoidScore: moe.useSigmoidScore,
                nGroup: moe.nGroup,
                topkGroup: moe.topkGroup,
                groupTopM: moe.groupTopM,
                routedSwigluClamp: moe.routedSwigluClamp,
                sharedSwigluClamp: moe.sharedSwigluClamp
            )
            let scratch = QWMoEOperator.Scratch(
                xRot32: s.xRot32, xPlain32: s.xPlain32,
                logits: s.logits, idx: s.idx, wts: s.wts,
                gu32: s.gu32, act32: s.act32,
                guSh32: s.guSh32, actSh32: s.actSh32,
                shared: s.shared, gate: s.gate,
                expOff: s.expOff, sortedPair: s.sortedPair,
                sortedTok: s.sortedTok, invPerm: s.invPerm,
                tileE: s.tileE, tileR: s.tileR, tileCount: s.tileCount,
                gu16: s.gu16, dnHi: s.dnHi, dnAcc: s.dnAcc,
                naxX16: s.naxX16, latent: s.latent,
                maxTileWidth: max(geometry.hidden, geometry.sharedExpertIntermediate),
                activeExperts: s.activeExperts, activeIndirect: s.activeIndirect
            )
            let rotateFn: ((inout any MTLComputeCommandEncoder) throws -> Void)?
            if let driver {
                rotateFn = { try driver.rotateActiveEncoder(&$0) }
            } else if let cb {
                rotateFn = { try hal.rotateEncoder(cb, &$0) }
            } else {
                rotateFn = nil
            }
            let moeInput: QWMoEOperator.Input = preBridgedF32.map {
                QWMoEOperator.Input.f32PreRotated(plain: $0.plain, rotated: $0.rotated, prefillPrecast16: x16)
            } ?? .f16Normed(x16)
            try QWMoEOperator.encodeResident(
                &enc, hal: hal, weights: weights, scratch: scratch,
                input: moeInput, out32: dn32, out16: dn16, skipOut32: skipOut32,
                rows: rows, supportsGroupedNaxPrefill: geometry.supportsGroupedNAXMoEPrefill,
                concurrentBarriers: false,
                rotateEncoder: rotateFn,
                profileStage: { stage in
                    driver?.profileStage(stage)
                    return driver?.activeEnc
                }
            )
            return
        }
        if moe.latent != nil || moe.splitFeatureRouting || moe.routedScaleParam != nil {
            throw QWEngineDriverError.unsupportedModel(
                "SSD cold expert streaming does not support Latent-MoE or split-feature routers; keep experts resident")
        }
        guard let driver else {
            throw QWEngineDriverError.state("cold expert streaming requires a driver")
        }
        let H = geometry.hidden
        let I = geometry.moeIntermediate
        let E = geometry.numExperts
        let K = geometry.expertsPerTok
        let ISh = geometry.sharedExpertIntermediate
        let isGroupedNax = rows > 8 && geometry.supportsGroupedNAXMoEPrefill
            && hal.caps.supportsSimdgroupMMA
            && (streamer == nil || (moe.coldGateUp != nil && moe.coldDown != nil))
            && s.expOff != nil && s.sortedPair != nil && s.sortedTok != nil
            && s.invPerm != nil && s.tileE != nil && s.tileR != nil
            && s.tileCount != nil && s.gu16 != nil && s.dnHi != nil
            && s.naxX16 != nil
        let skipRot32 = isGroupedNax && (ISh == 0 || (rows >= 64 && H % 64 == 0 && (2 * ISh) % 64 == 0))
        // The bridge: the f16 normed hidden → the rotated and the un-rotated
        // f32 copies. The router reads the un-rotated one (its weight is the
        // packer's raw plane); the expert GEMMs read the rotated one (the
        // packer rotates every weight's input dimension).
        try QWGemm.encodeMoEBridgeF16ToF32(enc, hal: hal, x: x16,
                                           xRot32: s.xRot32, xPlain32: s.xPlain32,
                                           rows: rows, hidden: H, skipRot32: skipRot32)
        // The router: the raw plane over the un-rotated hidden (fuses shared-expert sigmoid gate).
        try QWGemm.encodeFnRouterGemm(enc, hal: hal, weight: moe.router,
                                     x: s.xPlain32, y: s.logits, rows: rows, n: E, k: H,
                                     shGateWeight: ISh > 0 ? moe.sharedGateScalar : nil,
                                     shGateOut: ISh > 0 ? s.gate : nil)

        if isGroupedNax,
           let expOff = s.expOff, let sortedPair = s.sortedPair,
           let sortedTok = s.sortedTok, let invPerm = s.invPerm,
           let tileE = s.tileE, let tileR = s.tileR, let tileCount = s.tileCount,
           let gu16 = s.gu16, let dnHi = s.dnHi,
           let naxX16 = s.naxX16 {
            // The route: the top-k experts and their weights. The normalisation is
            // the model's own flag (`norm_topk_prob`), never a constant.
            try QWGemm.encodeFnMoeRoute(enc, hal: hal, logits: s.logits, idx: s.idx, wts: s.wts,
                                        rows: rows, experts: E, topK: K,
                                        normTopK: geometry.normTopKProb,
                                        bias: moe.routerBias,
                                        useSigmoid: moe.useSigmoidScore, nGroup: moe.nGroup,
                                        topkGroup: moe.topkGroup, groupTopM: moe.groupTopM)
            let shScratch0 = naxX16.slice(byteOffset: 0, count: rows * H, dtype: .f16)
            let dnAcc16 = s.dnAcc.slice(byteOffset: 0, count: rows * K * H, dtype: .f16)

            let maxShTileWidth = max(H, ISh)
            let shAct16 = (ISh > 0
                && naxX16.shape.elementCount >= rows * (H + ISh)
                && QWMoEOperator.canUseNaxDenseProj(hal: hal, weight: moe.sharedDown, rows: rows, maxTileWidth: maxShTileWidth))
                ? naxX16.slice(byteOffset: rows * H * 2, count: rows * ISh, dtype: .f16)
                : nil
            let encodeSharedExpert: (any MTLComputeCommandEncoder) throws -> Void = { targetEnc in
                if let shGU = moe.sharedGateUp, let shDown = moe.sharedDown,
                   let guSh = s.guSh32, let actSh = s.actSh32, ISh > 0 {
                    try QWGemm.encodeDenseProjF32(targetEnc, hal: hal, weight: shGU, x: s.xRot32, y: guSh,
                                                  naxX16: shScratch0, naxY16: shScratch0, maxTileWidth: H,
                                                  rows: rows, x16Precast: rows >= 64 ? x16 : nil)
                    try QWGemm.encodeSiluMulF32(targetEnc, hal: hal, gu: guSh, act: actSh, rows: rows,
                                                intermediate: ISh, act16: shAct16,
                                                swigluClamp: moe.sharedSwigluClamp)
                    let shScratch = shAct16 ?? shScratch0
                    try QWGemm.encodeDenseProjF32(targetEnc, hal: hal, weight: shDown, x: actSh, y: s.shared,
                                                  naxX16: shScratch, naxY16: shScratch,
                                                  maxTileWidth: shAct16 != nil ? maxShTileWidth : 0,
                                                  rows: rows, x16Precast: shAct16)
                }
            }

            var shEncoded = false
            let encodeSharedOnce: (any MTLComputeCommandEncoder) throws -> Void = { targetEnc in
                guard !shEncoded else { return }
                shEncoded = true
                try encodeSharedExpert(targetEnc)
            }
            let sliceSpan: (Int, Int) -> (
                spanRows: Int, spanPairs: Int, plan: QWGemm.MoETilePlan,
                x: QWTensorValue, ids: QWTensorValue, inv: QWTensorValue,
                gu: QWTensorValue, dnHi: QWTensorValue, dnAcc: QWTensorValue
            ) = { sRow, eRow in
                let sr = eRow - sRow
                let sp = sr * K
                return (
                    sr, sp,
                    QWGemm.MoETilePlan(experts: E, pairs: sp, gateUpRows: 2 * I, downRows: H),
                    x16.slice(byteOffset: sRow * H * 2, count: sr * H, dtype: .f16),
                    s.idx.slice(byteOffset: sRow * K * 4, count: sp),
                    invPerm.slice(byteOffset: 0, count: sp),
                    gu16.slice(byteOffset: 0, count: sp * 2 * I, dtype: .f16),
                    dnHi.slice(byteOffset: 0, count: sp * I, dtype: .f16),
                    s.dnAcc.slice(byteOffset: 0, count: sp * H, dtype: .f16)
                )
            }

            if driver.canStageColdExpertsMapped(moe: moe, isVerifyWindow: isVerifyWindow) {
                let tilePlan = QWGemm.MoETilePlan(experts: E, pairs: rows * K, gateUpRows: 2 * I, downRows: H)
                let stagedCount = try driver.stageColdExpertsMapped(
                    moe: moe, s: s, rows: rows,
                    resident: { resEnc, resCB, ids in
                        try encodeSharedOnce(resEnc)
                        try QWGemm.encodeMoEGroupPairsF32(resEnc, hal: hal, ids: ids, expertOff: expOff,
                                                          sortedPairIdx: sortedPair, sortedTok: sortedTok,
                                                          experts: E, totalPairs: rows * K, routes: K,
                                                          invPerm: invPerm, tileE: tileE, tileR: tileR,
                                                          tileCount: tileCount, tilePlan: tilePlan)
                        try hal.rotateEncoder(resCB, &resEnc)
                        try QWGemm.encodeMoEGateUpNax(resEnc, hal: hal, xe: x16, bank: moe.gateUp, expertOff: expOff,
                                                      tileE: tileE, tileR: tileR, tileCount: tileCount,
                                                      gu16: gu16, rowsPerExpert: 2 * I, tilePlan: tilePlan,
                                                      sortedTok: sortedTok, coldBank: moe.coldGateUp,
                                                      tileFilter: .residentOnly)
                        try QWGemm.encodeSiluMulSplitF16(resEnc, hal: hal, gu: gu16, outHi: dnHi, outLo: dnHi,
                                                         rows: rows * K, intermediate: I,
                                                         swigluClamp: moe.routedSwigluClamp)
                        try QWGemm.encodeMoENaxSplitF32(resEnc, hal: hal, xHi: dnHi, xLo: dnHi, bank: moe.down,
                                                        expertOff: expOff, tileE: tileE, tileR: tileR,
                                                        tileCount: tileCount, y32: dnAcc16, rowsPerExpert: H,
                                                        tilePlan: tilePlan, coldBank: moe.coldDown,
                                                        tileFilter: .residentOnly)
                    },
                    coldGateUp: { guEnc in
                        try QWGemm.encodeMoEGateUpNax(guEnc, hal: hal, xe: x16, bank: moe.gateUp, expertOff: expOff,
                                                      tileE: tileE, tileR: tileR, tileCount: tileCount,
                                                      gu16: gu16, rowsPerExpert: 2 * I, tilePlan: tilePlan,
                                                      sortedTok: sortedTok, coldBank: moe.coldGateUp,
                                                      tileFilter: .coldOnly)
                    }
                )
                if let active = driver.activeEnc { enc = active }
                if stagedCount > 0 {
                    try QWGemm.encodeSiluMulSplitF16(enc, hal: hal, gu: gu16, outHi: dnHi, outLo: dnHi,
                                                     rows: rows * K, intermediate: I,
                                                     swigluClamp: moe.routedSwigluClamp)
                    try QWGemm.encodeMoENaxSplitF32(enc, hal: hal, xHi: dnHi, xLo: dnHi, bank: moe.down,
                                                    expertOff: expOff, tileE: tileE, tileR: tileR,
                                                    tileCount: tileCount, y32: dnAcc16, rowsPerExpert: H,
                                                    tilePlan: tilePlan, coldBank: moe.coldDown,
                                                    tileFilter: .coldOnly)
                }
                try QWGemm.encodeMoEDownReduceCombineF32(enc, hal: hal, dnAcc: dnAcc16, invPerm: invPerm,
                                                         weights: s.wts, shared: s.shared, gate: s.gate,
                                                         out: dn32, hidden: H, routes: K, tokens: rows,
                                                         out16: dn16, skipOut32: skipOut32)
                if stagedCount > 0 {
                    try driver.unloadColdExpertsIfNeeded(moe: moe, stagedCount: stagedCount)
                    if let active = driver.activeEnc { enc = active }
                }
            } else {
                var startRow = 0
                while startRow < rows {
                    var overlappedResident = false
                    var overlappedColdGateUp = false
                    let res = try driver.stageColdExpertsSpan(
                        moe: moe, s: s, startRow: startRow, totalRows: rows,
                        isVerifyWindow: isVerifyWindow,
                        onBeforeIO: { resEnc, resCB, spanEnd in
                            overlappedResident = true
                            let sp = sliceSpan(startRow, spanEnd)
                            try encodeSharedOnce(resEnc)
                            try QWGemm.encodeMoEGroupPairsF32(resEnc, hal: hal, ids: sp.ids, expertOff: expOff,
                                                              sortedPairIdx: sortedPair, sortedTok: sortedTok,
                                                              experts: E, totalPairs: sp.spanPairs, routes: K,
                                                              invPerm: sp.inv, tileE: tileE, tileR: tileR,
                                                              tileCount: tileCount, tilePlan: sp.plan)
                            try hal.rotateEncoder(resCB, &resEnc)
                            try QWGemm.encodeMoEGateUpNax(resEnc, hal: hal, xe: sp.x, bank: moe.gateUp, expertOff: expOff,
                                                          tileE: tileE, tileR: tileR, tileCount: tileCount,
                                                          gu16: sp.gu, rowsPerExpert: 2 * I, tilePlan: sp.plan,
                                                          sortedTok: sortedTok, coldBank: moe.coldGateUp,
                                                          tileFilter: .residentOnly)
                            try QWGemm.encodeSiluMulSplitF16(resEnc, hal: hal, gu: sp.gu, outHi: sp.dnHi, outLo: sp.dnHi,
                                                             rows: sp.spanPairs, intermediate: I,
                                                             swigluClamp: moe.routedSwigluClamp)
                            try QWGemm.encodeMoENaxSplitF32(resEnc, hal: hal, xHi: sp.dnHi, xLo: sp.dnHi, bank: moe.down,
                                                            expertOff: expOff, tileE: tileE, tileR: tileR,
                                                            tileCount: tileCount, y32: sp.dnAcc, rowsPerExpert: H,
                                                            tilePlan: sp.plan, coldBank: moe.coldDown,
                                                            tileFilter: .residentOnly)
                        },
                        onAfterGateUpIO: { guEnc, spanEnd in
                            overlappedColdGateUp = true
                            let sp = sliceSpan(startRow, spanEnd)
                            try QWGemm.encodeMoEGateUpNax(guEnc, hal: hal, xe: sp.x, bank: moe.gateUp, expertOff: expOff,
                                                          tileE: tileE, tileR: tileR, tileCount: tileCount,
                                                          gu16: sp.gu, rowsPerExpert: 2 * I, tilePlan: sp.plan,
                                                          sortedTok: sortedTok, coldBank: moe.coldGateUp,
                                                          tileFilter: .coldOnly)
                        }
                    )
                    if let active = driver.activeEnc { enc = active }
                    let nextRow = res.nextRow
                    let stagedCount = res.stagedCount
                    guard nextRow > startRow else {
                        throw QWEngineDriverError.state("cold expert span made no forward progress at row \(startRow)")
                    }
                    let sp = sliceSpan(startRow, nextRow)
                    if overlappedResident {
                        if stagedCount > 0 {
                            if !overlappedColdGateUp {
                                try QWGemm.encodeMoEGateUpNax(enc, hal: hal, xe: sp.x, bank: moe.gateUp, expertOff: expOff,
                                                              tileE: tileE, tileR: tileR, tileCount: tileCount,
                                                              gu16: sp.gu, rowsPerExpert: 2 * I, tilePlan: sp.plan,
                                                              sortedTok: sortedTok, coldBank: moe.coldGateUp,
                                                              tileFilter: .coldOnly)
                            }
                            try QWGemm.encodeSiluMulSplitF16(enc, hal: hal, gu: sp.gu, outHi: sp.dnHi, outLo: sp.dnHi,
                                                             rows: sp.spanPairs, intermediate: I,
                                                             swigluClamp: moe.routedSwigluClamp)
                            try QWGemm.encodeMoENaxSplitF32(enc, hal: hal, xHi: sp.dnHi, xLo: sp.dnHi, bank: moe.down,
                                                            expertOff: expOff, tileE: tileE, tileR: tileR,
                                                            tileCount: tileCount, y32: sp.dnAcc, rowsPerExpert: H,
                                                            tilePlan: sp.plan, coldBank: moe.coldDown,
                                                            tileFilter: .coldOnly)
                        }
                    } else {
                        try encodeSharedOnce(enc)
                        try QWGemm.encodeMoEGroupPairsF32(enc, hal: hal, ids: sp.ids, expertOff: expOff,
                                                          sortedPairIdx: sortedPair, sortedTok: sortedTok,
                                                          experts: E, totalPairs: sp.spanPairs, routes: K,
                                                          invPerm: sp.inv, tileE: tileE, tileR: tileR,
                                                          tileCount: tileCount, tilePlan: sp.plan)
                        try driver.rotateActiveEncoder(&enc)
                        try QWGemm.encodeMoEGateUpNax(enc, hal: hal, xe: sp.x, bank: moe.gateUp, expertOff: expOff,
                                                      tileE: tileE, tileR: tileR, tileCount: tileCount,
                                                      gu16: sp.gu, rowsPerExpert: 2 * I, tilePlan: sp.plan,
                                                      sortedTok: sortedTok, coldBank: moe.coldGateUp)
                        try QWGemm.encodeSiluMulSplitF16(enc, hal: hal, gu: sp.gu, outHi: sp.dnHi, outLo: sp.dnHi,
                                                         rows: sp.spanPairs, intermediate: I,
                                                         swigluClamp: moe.routedSwigluClamp)
                        try QWGemm.encodeMoENaxSplitF32(enc, hal: hal, xHi: sp.dnHi, xLo: sp.dnHi, bank: moe.down,
                                                        expertOff: expOff, tileE: tileE, tileR: tileR,
                                                        tileCount: tileCount, y32: sp.dnAcc, rowsPerExpert: H,
                                                        tilePlan: sp.plan, coldBank: moe.coldDown)
                    }
                    let wtsSpan = s.wts.slice(byteOffset: startRow * K * 4, count: sp.spanPairs, dtype: .f32)
                    let shSpan = s.shared.slice(byteOffset: startRow * H * 4, count: sp.spanRows * H, dtype: .f32)
                    let gateSpan = s.gate.slice(byteOffset: startRow * 4, count: sp.spanRows, dtype: .f32)
                    let dn32Span = dn32.slice(byteOffset: startRow * H * 4, count: sp.spanRows * H, dtype: .f32)
                    let dn16Span = dn16.slice(byteOffset: startRow * H * 2, count: sp.spanRows * H, dtype: .f16)
                    try QWGemm.encodeMoEDownReduceCombineF32(enc, hal: hal, dnAcc: sp.dnAcc, invPerm: sp.inv,
                                                             weights: wtsSpan, shared: shSpan, gate: gateSpan,
                                                             out: dn32Span, hidden: H, routes: K, tokens: sp.spanRows,
                                                             out16: dn16Span, skipOut32: skipOut32)
                    if stagedCount > 0 {
                        try driver.unloadColdExpertsIfNeeded(moe: moe, stagedCount: stagedCount)
                        if let active = driver.activeEnc { enc = active }
                    }
                    startRow = nextRow
                }
            }
        } else {
            // Rank-1 GEMV decode / cold-streaming path
            let fuseSh = ISh > 0 && ISh <= 4096 && (ISh % 8 == 0) && moe.sharedGateUp != nil && s.guSh32 != nil && s.actSh32 != nil
            if let shGU = moe.sharedGateUp, let guSh = s.guSh32, ISh > 0 {
                try QWGemm.encodeHQ4F32V4(enc, hal: hal, weight: shGU, x: s.xRot32, y: guSh, rows: rows)
            }
            let moeSpecBudget = isVerifyWindow ? driver.activeMoESpecExpertBudget(rows: rows) : 0
            try QWGemm.encodeFnMoeRoute(enc, hal: hal, logits: s.logits, idx: s.idx, wts: s.wts,
                                        rows: rows, experts: E, topK: K,
                                        normTopK: geometry.normTopKProb,
                                        guSh: fuseSh ? s.guSh32 : nil,
                                        actSh: fuseSh ? s.actSh32 : nil,
                                        sharedIntermediate: fuseSh ? ISh : 0,
                                        bias: moe.routerBias,
                                        expertBudget: moeSpecBudget,
                                        useSigmoid: moe.useSigmoidScore, nGroup: moe.nGroup,
                                        topkGroup: moe.topkGroup, groupTopM: moe.groupTopM,
                                        sharedSwigluClamp: moe.sharedSwigluClamp)
            var shDownEncoded = false
            let encodeSharedDown: (any MTLComputeCommandEncoder) throws -> Void = { targetEnc in
                guard !shDownEncoded else { return }
                shDownEncoded = true
                if let shDown = moe.sharedDown,
                   let guSh = s.guSh32, let actSh = s.actSh32, ISh > 0 {
                    if !fuseSh {
                        try QWGemm.encodeSiluMulF32(targetEnc, hal: hal, gu: guSh, act: actSh, rows: rows,
                                                    intermediate: ISh, swigluClamp: moe.sharedSwigluClamp)
                    }
                    try QWGemm.encodeHQ4F32V4(targetEnc, hal: hal, weight: shDown, x: actSh, y: s.shared, rows: rows)
                }
            }
            let fuseRoutedGU = I > 0 && (I % 32 == 0)
            if driver.coldEventsUsable(moe, rows: rows, isVerifyWindow: isVerifyWindow) {
                let xSlice = s.xRot32.slice(byteOffset: 0, count: rows * H, dtype: .f32)
                let actSlice = s.act32.slice(byteOffset: 0, count: rows * K * I, dtype: .f32)
                let guSlice = s.gu32.slice(byteOffset: 0, count: rows * K * 2 * I, dtype: .f32)
                let wtsSlice = s.wts.slice(byteOffset: 0, count: rows * K, dtype: .f32)
                let shSlice = s.shared.slice(byteOffset: 0, count: rows * H, dtype: .f32)
                let gateSlice = s.gate.slice(byteOffset: 0, count: rows, dtype: .f32)
                let dn32Slice = s.dnAcc.slice(byteOffset: 0, count: rows * K * H, dtype: .f32)
                let encodeExperts: (any MTLComputeCommandEncoder, QWTensorValue, QWGemm.MoETileFilter) throws -> Void = {
                    targetEnc, ids, filter in
                    try QWGemm.encodeMoEGateUpF32R1(targetEnc, hal: hal, x: xSlice, bank: moe.gateUp,
                                                   ids: ids, out: fuseRoutedGU ? actSlice : guSlice,
                                                   rowsPerExpert: 2 * I, routes: K, tokens: rows,
                                                   fuseSiluFwht: fuseRoutedGU, coldBank: moe.coldGateUp,
                                                   tileFilter: filter, swigluClamp: moe.routedSwigluClamp)
                    if !fuseRoutedGU {
                        try QWGemm.encodeSiluMulF32(targetEnc, hal: hal, gu: guSlice, act: actSlice,
                                                    rows: rows * K, intermediate: I,
                                                    swigluClamp: moe.routedSwigluClamp)
                    }
                    try QWGemm.encodeMoEDownSharedCombineF32(targetEnc, hal: hal, x: actSlice, bank: moe.down,
                                                            rowsPerRoute: H, routes: K, weights: wtsSlice,
                                                            shared: shSlice, gate: gateSlice, out: dn32Slice,
                                                            ids: ids, tokens: rows, out16: nil,
                                                            coldBank: moe.coldDown, tileFilter: filter,
                                                            routeOutput: true)
                }
                // Resident experts run on GPU-mapped slots while the host stages the misses.
                let overlapped = try driver.stageColdExpertsEvent(
                    moe: moe, s: s, rows: rows, isVerifyWindow: isVerifyWindow
                ) { resEnc, resIds in
                    try encodeSharedDown(resEnc)
                    try encodeExperts(resEnc, resIds, .residentOnly)
                }
                if let active = driver.activeEnc { enc = active }
                try encodeSharedDown(enc)
                try encodeExperts(enc, s.idx.slice(byteOffset: 0, count: rows * K), overlapped ? .coldOnly : .all)
                try driver.encodeEventPromotions(moe: moe)
            } else {
                var startRow = 0
                while startRow < rows {
                    var overlappedSpan = false
                    var overlappedColdGateUp = false
                    let (nextRow, stagedCount) = try driver.stageColdExpertsSpan(
                        moe: moe, s: s, startRow: startRow, totalRows: rows,
                        isVerifyWindow: isVerifyWindow,
                        onBeforeIO: { resEnc, _, spanEnd in
                            overlappedSpan = true
                            try encodeSharedDown(resEnc)
                            let spanRows = spanEnd - startRow
                            let xSlice = s.xRot32.slice(byteOffset: startRow * H * 4, count: spanRows * H, dtype: .f32)
                            let idsSlice = s.idx.slice(byteOffset: startRow * K * 4, count: spanRows * K)
                            let actSlice = s.act32.slice(byteOffset: startRow * K * I * 4, count: spanRows * K * I, dtype: .f32)
                            let guSlice = s.gu32.slice(byteOffset: startRow * K * 2 * I * 4, count: spanRows * K * 2 * I, dtype: .f32)
                            let wtsSlice = s.wts.slice(byteOffset: startRow * K * 4, count: spanRows * K, dtype: .f32)
                            let shSlice = s.shared.slice(byteOffset: startRow * H * 4, count: spanRows * H, dtype: .f32)
                            let gateSlice = s.gate.slice(byteOffset: startRow * 4, count: spanRows, dtype: .f32)
                            let dn32Slice = s.dnAcc.slice(byteOffset: startRow * K * H * 4,
                                                          count: spanRows * K * H, dtype: .f32)
                            try QWGemm.encodeMoEGateUpF32R1(resEnc, hal: hal, x: xSlice, bank: moe.gateUp,
                                                           ids: idsSlice, out: fuseRoutedGU ? actSlice : guSlice,
                                                           rowsPerExpert: 2 * I, routes: K, tokens: spanRows,
                                                           fuseSiluFwht: fuseRoutedGU, coldBank: moe.coldGateUp,
                                                           tileFilter: .residentOnly,
                                                           swigluClamp: moe.routedSwigluClamp)
                            if !fuseRoutedGU {
                                try QWGemm.encodeSiluMulF32(resEnc, hal: hal, gu: guSlice, act: actSlice,
                                                            rows: spanRows * K, intermediate: I,
                                                            swigluClamp: moe.routedSwigluClamp)
                            }
                            try QWGemm.encodeMoEDownSharedCombineF32(resEnc, hal: hal, x: actSlice, bank: moe.down,
                                                                    rowsPerRoute: H, routes: K, weights: wtsSlice,
                                                                    shared: shSlice, gate: gateSlice, out: dn32Slice,
                                                                    ids: idsSlice, tokens: spanRows,
                                                                    out16: nil, coldBank: moe.coldDown,
                                                                    tileFilter: .residentOnly, routeOutput: true)
                        },
                        onAfterGateUpIO: { guEnc, spanEnd in
                            overlappedColdGateUp = true
                            let spanRows = spanEnd - startRow
                            let xSlice = s.xRot32.slice(byteOffset: startRow * H * 4, count: spanRows * H, dtype: .f32)
                            let idsSlice = s.idx.slice(byteOffset: startRow * K * 4, count: spanRows * K)
                            let actSlice = s.act32.slice(byteOffset: startRow * K * I * 4, count: spanRows * K * I, dtype: .f32)
                            let guSlice = s.gu32.slice(byteOffset: startRow * K * 2 * I * 4, count: spanRows * K * 2 * I, dtype: .f32)
                            try QWGemm.encodeMoEGateUpF32R1(guEnc, hal: hal, x: xSlice, bank: moe.gateUp,
                                                           ids: idsSlice, out: fuseRoutedGU ? actSlice : guSlice,
                                                           rowsPerExpert: 2 * I, routes: K, tokens: spanRows,
                                                           fuseSiluFwht: fuseRoutedGU, coldBank: moe.coldGateUp,
                                                           tileFilter: .coldOnly,
                                                           swigluClamp: moe.routedSwigluClamp)
                            if !fuseRoutedGU {
                                try QWGemm.encodeSiluMulF32(guEnc, hal: hal, gu: guSlice, act: actSlice,
                                                            rows: spanRows * K, intermediate: I,
                                                            swigluClamp: moe.routedSwigluClamp)
                            }
                        }
                    )
                    if let active = driver.activeEnc { enc = active }
                    guard nextRow > startRow else {
                        throw QWEngineDriverError.state("cold expert span made no forward progress at row \(startRow)")
                    }
                    if !shDownEncoded {
                        try encodeSharedDown(enc)
                    }
                    let spanRows = nextRow - startRow
                    let xSlice = s.xRot32.slice(byteOffset: startRow * H * 4, count: spanRows * H, dtype: .f32)
                    let idsSlice = s.idx.slice(byteOffset: startRow * K * 4, count: spanRows * K)
                    let actSlice = s.act32.slice(byteOffset: startRow * K * I * 4, count: spanRows * K * I, dtype: .f32)
                    let guSlice = s.gu32.slice(byteOffset: startRow * K * 2 * I * 4, count: spanRows * K * 2 * I, dtype: .f32)
                    let wtsSlice = s.wts.slice(byteOffset: startRow * K * 4, count: spanRows * K, dtype: .f32)
                    let shSlice = s.shared.slice(byteOffset: startRow * H * 4, count: spanRows * H, dtype: .f32)
                    let gateSlice = s.gate.slice(byteOffset: startRow * 4, count: spanRows, dtype: .f32)
                    let routeDownOffset = startRow * K * H * 4
                    let routeDownCount = spanRows * K * H
                    if !overlappedSpan || stagedCount > 0 {
                        let filter: QWGemm.MoETileFilter = overlappedSpan ? .coldOnly : .all
                        if !overlappedColdGateUp {
                            try QWGemm.encodeMoEGateUpF32R1(enc, hal: hal, x: xSlice, bank: moe.gateUp,
                                                           ids: idsSlice, out: fuseRoutedGU ? actSlice : guSlice,
                                                           rowsPerExpert: 2 * I, routes: K, tokens: spanRows,
                                                           fuseSiluFwht: fuseRoutedGU, coldBank: moe.coldGateUp,
                                                           tileFilter: filter,
                                                           swigluClamp: moe.routedSwigluClamp)
                            if !fuseRoutedGU {
                                try QWGemm.encodeSiluMulF32(enc, hal: hal, gu: guSlice, act: actSlice,
                                                            rows: spanRows * K, intermediate: I,
                                                            swigluClamp: moe.routedSwigluClamp)
                            }
                        }
                        try QWGemm.encodeMoEDownSharedCombineF32(enc, hal: hal, x: actSlice, bank: moe.down,
                                                                rowsPerRoute: H, routes: K, weights: wtsSlice,
                                                                shared: shSlice, gate: gateSlice,
                                                                out: s.dnAcc.slice(byteOffset: routeDownOffset,
                                                                                   count: routeDownCount, dtype: .f32),
                                                                ids: idsSlice, tokens: spanRows,
                                                                out16: nil, coldBank: moe.coldDown,
                                                                tileFilter: filter, routeOutput: true)
                    }
                    if stagedCount > 0 {
                        try driver.unloadColdExpertsIfNeeded(moe: moe, stagedCount: stagedCount)
                        if let active = driver.activeEnc { enc = active }
                    }
                    startRow = nextRow
                }
            }
            if let active = driver.activeEnc { enc = active }
            try QWGemm.encodeMoEDownReduceCombineF32(enc, hal: hal, dnAcc: s.dnAcc, invPerm: nil,
                                                     weights: s.wts, shared: s.shared, gate: s.gate,
                                                     out: dn32, hidden: H, routes: K, tokens: rows,
                                                     out16: dn16, skipOut32: skipOut32)
        }
    }
}

/// The hybrid GDN + sparse-MoE layer: linear attention with a gated norm, then
/// the routed experts (+ shared expert when present).
final class GDNMoELayerKernel: DenseLayerKernel, QWLayerKernel, @unchecked Sendable {
    static let key = QWEnginePlan.LayerKernelKey(attention: .gdn, mlp: .moe, ple: false)
    static func make() -> any QWLayerKernel { GDNMoELayerKernel() }

    private let attention = GDNLayerKernel()

    override func encodeAttention(_ context: QWLayerDecodeContext) throws {
        try attention.encodeAttention(context)
    }

    override func encodeAttentionPrefill(_ context: QWLayerPrefillContext) throws {
        try attention.encodeAttentionPrefill(context)
    }

    override func encodeMLP(_ context: QWLayerDecodeContext) throws {
        let driver = context.driver
        let sc = driver.sc
        guard let moe = context.weights.moe else {
            throw QWEngineDriverError.state("a GDN sparse-MoE layer has no MoE weights")
        }
        guard let s = sc.moe else {
            throw QWEngineDriverError.state("the decode scratch has no MoE tensors")
        }
        var active = driver.activeEnc ?? context.enc
        let preBridged = driver.canFusePreMoENorm(weights: context.weights, rows: 1)
            ? (plain: s.xPlain32, rotated: s.xRot32) : nil
        try QWMoEChain.encode(enc: &active, hal: driver.hal, moe: moe, s: s,
                              x16: sc.ln2, dn32: sc.dn32, dn16: sc.dn16,
                              rows: 1, geometry: driver.geometry, driver: driver, skipOut32: true,
                              preBridgedF32: preBridged)
    }

    override func encodeMLPPrefill(_ context: QWLayerPrefillContext) throws {
        let driver = context.driver
        let psc = driver.psc
        let geometry = driver.geometry
        let H = geometry.hidden
        let T = context.rows
        guard let moe = context.weights.moe else {
            throw QWEngineDriverError.state("a GDN sparse-MoE layer has no MoE weights")
        }
        guard let s = psc.moe else {
            throw QWEngineDriverError.state("the prefill scratch has no MoE tensors")
        }
        var active = driver.activeEnc ?? context.enc
        let preBridged = driver.canFusePreMoENorm(weights: context.weights, rows: T)
            ? (plain: driver.prefixRows(s.xPlain32, rows: T, cols: H, dt: .f32),
               rotated: driver.prefixRows(s.xRot32, rows: T, cols: H, dt: .f32))
            : nil
        try QWMoEChain.encode(enc: &active, hal: driver.hal, moe: moe, s: s,
                              x16: driver.prefixRows(psc.ln2, rows: T, cols: H, dt: .f16),
                              dn32: psc.dn32, dn16: psc.dn16,
                              rows: T, geometry: geometry, driver: driver, skipOut32: true,
                              preBridgedF32: preBridged)
    }
}

/// The sparse-MoE layer: the dense full attention, then the routed experts.
final class MoELayerKernel: DenseLayerKernel, QWLayerKernel, @unchecked Sendable {
    static let key = QWEnginePlan.LayerKernelKey(attention: .fullAttention, mlp: .moe, ple: false)
    static func make() -> any QWLayerKernel { MoELayerKernel() }

    /// The dense full-attention kernel: a MoE layer's attention is the same
    /// code as a dense layer's, so it is reused rather than copied.
    private let attention = FullAttentionLayerKernel()

    override func encodeAttention(_ context: QWLayerDecodeContext) throws {
        try attention.encodeAttention(context)
    }

    override func encodeAttentionPrefill(_ context: QWLayerPrefillContext) throws {
        try attention.encodeAttentionPrefill(context)
    }

    override func encodeMLP(_ context: QWLayerDecodeContext) throws {
        let driver = context.driver
        let sc = driver.sc
        guard let moe = context.weights.moe else {
            throw QWEngineDriverError.state("a sparse-MoE layer has no MoE weights")
        }
        guard let s = sc.moe else {
            throw QWEngineDriverError.state("the decode scratch has no MoE tensors")
        }
        var active = driver.activeEnc ?? context.enc
        let preBridged = driver.canFusePreMoENorm(weights: context.weights, rows: 1)
            ? (plain: s.xPlain32, rotated: s.xRot32) : nil
        try QWMoEChain.encode(enc: &active, hal: driver.hal, moe: moe, s: s,
                              x16: sc.ln2, dn32: sc.dn32, dn16: sc.dn16,
                              rows: 1, geometry: driver.geometry, driver: driver, skipOut32: true,
                              preBridgedF32: preBridged)
    }

    override func encodeMLPPrefill(_ context: QWLayerPrefillContext) throws {
        let driver = context.driver
        let psc = driver.psc
        let geometry = driver.geometry
        let H = geometry.hidden
        let T = context.rows
        guard let moe = context.weights.moe else {
            throw QWEngineDriverError.state("a sparse-MoE layer has no MoE weights")
        }
        guard let s = psc.moe else {
            throw QWEngineDriverError.state("the prefill scratch has no MoE tensors")
        }
        var active = driver.activeEnc ?? context.enc
        let preBridged = driver.canFusePreMoENorm(weights: context.weights, rows: T)
            ? (plain: driver.prefixRows(s.xPlain32, rows: T, cols: H, dt: .f32),
               rotated: driver.prefixRows(s.xRot32, rows: T, cols: H, dt: .f32))
            : nil
        try QWMoEChain.encode(enc: &active, hal: driver.hal, moe: moe, s: s,
                              x16: driver.prefixRows(psc.ln2, rows: T, cols: H, dt: .f16),
                              dn32: psc.dn32, dn16: psc.dn16,
                              rows: T, geometry: geometry, driver: driver, skipOut32: true,
                              preBridgedF32: preBridged)
    }
}
