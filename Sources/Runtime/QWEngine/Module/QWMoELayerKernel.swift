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
                       cb: (any MTLCommandBuffer)? = nil) throws {
        // QW_COLD_TIMING_PROXY=1 (benchmark only): skip SSD staging and read cold ids from the
        // stale cold bank (experts - K slots, so ids stay in bounds). Same bytes and dispatches
        // as an all-resident pack; the output is not meaningful.
        let streamer = Self.coldTimingProxy ? nil : moe.coldStreamer
        if streamer == nil {
            let weights = QWMoEOperator.Weights(
                router: moe.router,
                routerBias: moe.routerBias,
                gateUp: moe.gateUp,
                down: moe.down,
                coldGateUp: moe.coldGateUp,
                coldDown: moe.coldDown,
                sharedGateUp: moe.sharedGateUp,
                sharedDown: moe.sharedDown,
                sharedGateScalar: moe.sharedGateScalar,
                hidden: geometry.hidden,
                experts: geometry.numExperts,
                topK: geometry.expertsPerTok,
                intermediate: geometry.moeIntermediate,
                sharedIntermediate: geometry.sharedExpertIntermediate,
                normTopK: geometry.normTopKProb,
                reuseExpertsAcrossTokens: true,
                moeSpecExpertBudget: driver?.activeMoESpecExpertBudget(rows: rows) ?? 0
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
                naxX16: s.naxX16, maxTileWidth: max(geometry.hidden, geometry.sharedExpertIntermediate)
            )
            let rotateFn: ((inout any MTLComputeCommandEncoder) throws -> Void)?
            if let driver {
                rotateFn = { try driver.rotateActiveEncoder(&$0) }
            } else if let cb {
                rotateFn = { try hal.rotateEncoder(cb, &$0) }
            } else {
                rotateFn = nil
            }
            try QWMoEOperator.encodeResident(
                &enc, hal: hal, weights: weights, scratch: scratch,
                input: .f16Normed(x16), out32: dn32, out16: dn16,
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
        let H = geometry.hidden
        let I = geometry.moeIntermediate
        let E = geometry.numExperts
        let K = geometry.expertsPerTok
        let ISh = geometry.sharedExpertIntermediate
        let isGroupedNax = rows > 8 && geometry.supportsGroupedNAXMoEPrefill
            && hal.caps.isMetal4TensorCapable
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
                                        bias: moe.routerBias)
            let shScratch0 = naxX16.slice(byteOffset: 0, count: rows * H, dtype: .f16)
            let dnAcc16 = s.dnAcc.slice(byteOffset: 0, count: rows * K * H, dtype: .f16)

            let shAct16 = (rows >= 64 && ISh > 0 && ISh % 64 == 0)
                ? naxX16.slice(byteOffset: rows * H * 2, count: rows * ISh, dtype: .f16)
                : nil
            let encodeSharedExpert: (any MTLComputeCommandEncoder) throws -> Void = { targetEnc in
                if let shGU = moe.sharedGateUp, let shDown = moe.sharedDown,
                   let guSh = s.guSh32, let actSh = s.actSh32, ISh > 0 {
                    try QWGemm.encodeDenseProjF32(targetEnc, hal: hal, weight: shGU, x: s.xRot32, y: guSh,
                                                  naxX16: shScratch0, naxY16: shScratch0, maxTileWidth: H,
                                                  rows: rows, x16Precast: rows >= 64 ? x16 : nil)
                    try QWGemm.encodeSiluMulF32(targetEnc, hal: hal, gu: guSh, act: actSh, rows: rows,
                                                intermediate: ISh, act16: shAct16)
                    let shScratch = shAct16 ?? shScratch0
                    try QWGemm.encodeDenseProjF32(targetEnc, hal: hal, weight: shDown, x: actSh, y: s.shared,
                                                  naxX16: shScratch, naxY16: shScratch, maxTileWidth: max(H, ISh),
                                                  rows: rows, x16Precast: shAct16)
                }
            }

            let tilePlan = QWGemm.MoETilePlan(experts: E, pairs: rows * K, gateUpRows: 2 * I, downRows: H)
            let encodeResidentPass: (inout any MTLComputeCommandEncoder, any MTLCommandBuffer, QWTensorValue) throws -> Void = {
                resEnc, resCB, ids in
                try encodeSharedExpert(resEnc)
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
                                                 rows: rows * K, intermediate: I)
                try QWGemm.encodeMoENaxSplitF32(resEnc, hal: hal, xHi: dnHi, xLo: dnHi, bank: moe.down,
                                                expertOff: expOff, tileE: tileE, tileR: tileR,
                                                tileCount: tileCount, y32: dnAcc16, rowsPerExpert: H,
                                                tilePlan: tilePlan, coldBank: moe.coldDown,
                                                tileFilter: .residentOnly)
            }
            let encodeColdGateUp: (any MTLComputeCommandEncoder) throws -> Void = { guEnc in
                try QWGemm.encodeMoEGateUpNax(guEnc, hal: hal, xe: x16, bank: moe.gateUp, expertOff: expOff,
                                              tileE: tileE, tileR: tileR, tileCount: tileCount,
                                              gu16: gu16, rowsPerExpert: 2 * I, tilePlan: tilePlan,
                                              sortedTok: sortedTok, coldBank: moe.coldGateUp,
                                              tileFilter: .coldOnly)
            }

            var stagedCount = 0
            var overlappedResident = false
            var overlappedColdGateUp = false
            if streamer != nil, let driver, driver.canStageColdExpertsMapped(moe: moe) {
                stagedCount = try driver.stageColdExpertsMapped(
                    moe: moe, s: s, rows: rows, resident: encodeResidentPass, coldGateUp: encodeColdGateUp)
                overlappedResident = true
                overlappedColdGateUp = true
                if let active = driver.activeEnc { enc = active }
            } else if streamer != nil, let driver {
                let res = try driver.stageColdExpertsSpan(
                    moe: moe, s: s, startRow: 0, totalRows: rows,
                    onBeforeIO: { resEnc, resCB, _ in
                        overlappedResident = true
                        try encodeResidentPass(&resEnc, resCB, s.idx)
                    },
                    onAfterGateUpIO: { guEnc, _ in
                        overlappedColdGateUp = true
                        try encodeColdGateUp(guEnc)
                    }
                )
                stagedCount = res.stagedCount
                if let active = driver.activeEnc { enc = active }
            }

            if overlappedResident {
                // No staged expert means the resident pass already covered every pair.
                let hasCold = stagedCount > 0
                if hasCold && !overlappedColdGateUp {
                    try encodeColdGateUp(enc)
                }
                if hasCold {
                    try QWGemm.encodeSiluMulSplitF16(enc, hal: hal, gu: gu16, outHi: dnHi, outLo: dnHi,
                                                     rows: rows * K, intermediate: I)
                    try QWGemm.encodeMoENaxSplitF32(enc, hal: hal, xHi: dnHi, xLo: dnHi, bank: moe.down,
                                                    expertOff: expOff, tileE: tileE, tileR: tileR,
                                                    tileCount: tileCount, y32: dnAcc16, rowsPerExpert: H,
                                                    tilePlan: tilePlan, coldBank: moe.coldDown,
                                                    tileFilter: .coldOnly)
                }
            } else {
                try encodeSharedExpert(enc)
                try QWGemm.encodeMoEGroupPairsF32(enc, hal: hal, ids: s.idx, expertOff: expOff,
                                                  sortedPairIdx: sortedPair, sortedTok: sortedTok,
                                                  experts: E, totalPairs: rows * K, routes: K,
                                                  invPerm: invPerm, tileE: tileE, tileR: tileR,
                                                  tileCount: tileCount, tilePlan: tilePlan)
                if let driver {
                    try driver.rotateActiveEncoder(&enc)
                } else if let cb {
                    try hal.rotateEncoder(cb, &enc)
                }
                try QWGemm.encodeMoEGateUpNax(enc, hal: hal, xe: x16, bank: moe.gateUp, expertOff: expOff,
                                              tileE: tileE, tileR: tileR, tileCount: tileCount,
                                              gu16: gu16, rowsPerExpert: 2 * I, tilePlan: tilePlan,
                                              sortedTok: sortedTok, coldBank: moe.coldGateUp)
                try QWGemm.encodeSiluMulSplitF16(enc, hal: hal, gu: gu16, outHi: dnHi, outLo: dnHi,
                                                 rows: rows * K, intermediate: I)
                try QWGemm.encodeMoENaxSplitF32(enc, hal: hal, xHi: dnHi, xLo: dnHi, bank: moe.down,
                                                expertOff: expOff, tileE: tileE, tileR: tileR,
                                                tileCount: tileCount, y32: dnAcc16, rowsPerExpert: H,
                                                tilePlan: tilePlan, coldBank: moe.coldDown)
            }

            try QWGemm.encodeMoEDownReduceCombineF32(enc, hal: hal, dnAcc: dnAcc16, invPerm: invPerm,
                                                     weights: s.wts, shared: s.shared, gate: s.gate,
                                                     out: dn32, hidden: H, routes: K, tokens: rows,
                                                     out16: dn16)
            if stagedCount > 0, let driver {
                try driver.unloadColdExpertsIfNeeded(moe: moe, stagedCount: stagedCount)
                if let active = driver.activeEnc { enc = active }
            }
        } else {
            // Rank-1 GEMV decode / cold-streaming path
            let fuseSh = ISh > 0 && ISh <= 4096 && moe.sharedGateUp != nil && s.guSh32 != nil && s.actSh32 != nil
            if let shGU = moe.sharedGateUp, let guSh = s.guSh32, ISh > 0 {
                try QWGemm.encodeHQ4F32V4(enc, hal: hal, weight: shGU, x: s.xRot32, y: guSh, rows: rows)
            }
            let moeSpecBudget = driver?.activeMoESpecExpertBudget(rows: rows) ?? 0
            try QWGemm.encodeFnMoeRoute(enc, hal: hal, logits: s.logits, idx: s.idx, wts: s.wts,
                                        rows: rows, experts: E, topK: K,
                                        normTopK: geometry.normTopKProb,
                                        guSh: fuseSh ? s.guSh32 : nil,
                                        actSh: fuseSh ? s.actSh32 : nil,
                                        sharedIntermediate: fuseSh ? ISh : 0,
                                        bias: moe.routerBias,
                                        expertBudget: moeSpecBudget)
            var shDownEncoded = false
            let encodeSharedDown: (any MTLComputeCommandEncoder) throws -> Void = { targetEnc in
                guard !shDownEncoded else { return }
                shDownEncoded = true
                if let shDown = moe.sharedDown,
                   let guSh = s.guSh32, let actSh = s.actSh32, ISh > 0 {
                    if !fuseSh {
                        try QWGemm.encodeSiluMulF32(targetEnc, hal: hal, gu: guSh, act: actSh, rows: rows, intermediate: ISh)
                    }
                    try QWGemm.encodeHQ4F32V4(targetEnc, hal: hal, weight: shDown, x: actSh, y: s.shared, rows: rows)
                }
            }
            let fuseRoutedGU = I > 0 && (I % 32 == 0)
            if let driver, driver.coldEventsUsable(moe, rows: rows) {
                let routeOutput = streamer != nil
                let xSlice = s.xRot32.slice(byteOffset: 0, count: rows * H, dtype: .f32)
                let actSlice = s.act32.slice(byteOffset: 0, count: rows * K * I, dtype: .f32)
                let guSlice = s.gu32.slice(byteOffset: 0, count: rows * K * 2 * I, dtype: .f32)
                let wtsSlice = s.wts.slice(byteOffset: 0, count: rows * K, dtype: .f32)
                let shSlice = s.shared.slice(byteOffset: 0, count: rows * H, dtype: .f32)
                let gateSlice = s.gate.slice(byteOffset: 0, count: rows, dtype: .f32)
                let dn32Slice = s.dnAcc.slice(byteOffset: 0, count: rows * K * H, dtype: .f32)
                let downOut = routeOutput ? dn32Slice : dn32.slice(byteOffset: 0, count: rows * H, dtype: .f32)
                let downOut16 = routeOutput ? nil : dn16.slice(byteOffset: 0, count: rows * H, dtype: .f16)
                let encodeExperts: (any MTLComputeCommandEncoder, QWTensorValue, QWGemm.MoETileFilter) throws -> Void = {
                    targetEnc, ids, filter in
                    try QWGemm.encodeMoEGateUpF32R1(targetEnc, hal: hal, x: xSlice, bank: moe.gateUp,
                                                   ids: ids, out: fuseRoutedGU ? actSlice : guSlice,
                                                   rowsPerExpert: 2 * I, routes: K, tokens: rows,
                                                   fuseSiluFwht: fuseRoutedGU, coldBank: moe.coldGateUp,
                                                   tileFilter: filter)
                    if !fuseRoutedGU {
                        try QWGemm.encodeSiluMulF32(targetEnc, hal: hal, gu: guSlice, act: actSlice,
                                                    rows: rows * K, intermediate: I)
                    }
                    try QWGemm.encodeMoEDownSharedCombineF32(targetEnc, hal: hal, x: actSlice, bank: moe.down,
                                                            rowsPerRoute: H, routes: K, weights: wtsSlice,
                                                            shared: shSlice, gate: gateSlice, out: downOut,
                                                            ids: ids, tokens: rows, out16: downOut16,
                                                            coldBank: moe.coldDown, tileFilter: filter,
                                                            routeOutput: routeOutput)
                }
                // Resident experts run on GPU-mapped slots while the host stages the misses.
                let overlapped = try driver.stageColdExpertsEvent(moe: moe, s: s, rows: rows) { resEnc, resIds in
                    try encodeSharedDown(resEnc)
                    try encodeExperts(resEnc, resIds, .residentOnly)
                }
                if let active = driver.activeEnc { enc = active }
                try encodeSharedDown(enc)
                try encodeExperts(enc, s.idx.slice(byteOffset: 0, count: rows * K), overlapped ? .coldOnly : .all)
                try driver.encodeEventPromotions(moe: moe)
            } else if streamer != nil, let driver {
                var startRow = 0
                while startRow < rows {
                    var overlappedSpan = false
                    var overlappedColdGateUp = false
                    let (nextRow, stagedCount) = try driver.stageColdExpertsSpan(
                        moe: moe, s: s, startRow: startRow, totalRows: rows,
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
                                                           tileFilter: .residentOnly)
                            if !fuseRoutedGU {
                                try QWGemm.encodeSiluMulF32(resEnc, hal: hal, gu: guSlice, act: actSlice,
                                                            rows: spanRows * K, intermediate: I)
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
                                                           tileFilter: .coldOnly)
                            if !fuseRoutedGU {
                                try QWGemm.encodeSiluMulF32(guEnc, hal: hal, gu: guSlice, act: actSlice,
                                                            rows: spanRows * K, intermediate: I)
                            }
                        }
                    )
                    if let active = driver.activeEnc { enc = active }
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
                                                           tileFilter: filter)
                            if !fuseRoutedGU {
                                try QWGemm.encodeSiluMulF32(enc, hal: hal, gu: guSlice, act: actSlice,
                                                            rows: spanRows * K, intermediate: I)
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
            } else {
                let draftFilter: QWGemm.MoETileFilter = streamer != nil ? .residentOnly : .all
                let routeOutput = streamer != nil
                driver?.profileStage("moe.shared_down")
                if let a = driver?.activeEnc { enc = a }
                try encodeSharedDown(enc)
                driver?.profileStage("moe.experts_gate_up")
                if let a = driver?.activeEnc { enc = a }
                try QWGemm.encodeMoEGateUpF32R1(enc, hal: hal, x: s.xRot32, bank: moe.gateUp,
                                               ids: s.idx, out: fuseRoutedGU ? s.act32 : s.gu32,
                                               rowsPerExpert: 2 * I, routes: K, tokens: rows,
                                               fuseSiluFwht: fuseRoutedGU, coldBank: moe.coldGateUp,
                                               tileFilter: draftFilter)
                if !fuseRoutedGU {
                    try QWGemm.encodeSiluMulF32(enc, hal: hal, gu: s.gu32, act: s.act32,
                                                rows: rows * K, intermediate: I)
                }
                driver?.profileStage("moe.experts_down")
                if let a = driver?.activeEnc { enc = a }
                try QWGemm.encodeMoEDownSharedCombineF32(enc, hal: hal, x: s.act32, bank: moe.down,
                                                        rowsPerRoute: H, routes: K, weights: s.wts,
                                                        shared: s.shared, gate: s.gate,
                                                        out: routeOutput ? s.dnAcc : dn32,
                                                        ids: s.idx, tokens: rows,
                                                        out16: routeOutput ? nil : dn16, coldBank: moe.coldDown,
                                                        tileFilter: draftFilter, routeOutput: routeOutput)
            }
            if streamer != nil {
                if let active = driver?.activeEnc { enc = active }
                try QWGemm.encodeMoEDownReduceCombineF32(enc, hal: hal, dnAcc: s.dnAcc, invPerm: nil,
                                                         weights: s.wts, shared: s.shared, gate: s.gate,
                                                         out: dn32, hidden: H, routes: K, tokens: rows,
                                                         out16: dn16)
            }
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
        try QWMoEChain.encode(enc: &active, hal: driver.hal, moe: moe, s: s,
                              x16: sc.ln2, dn32: sc.dn32, dn16: sc.dn16,
                              rows: 1, geometry: driver.geometry, driver: driver)
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
        try QWMoEChain.encode(enc: &active, hal: driver.hal, moe: moe, s: s,
                              x16: driver.prefixRows(psc.ln2, rows: T, cols: H, dt: .f16),
                              dn32: psc.dn32, dn16: psc.dn16,
                              rows: T, geometry: geometry, driver: driver)
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
        try QWMoEChain.encode(enc: &active, hal: driver.hal, moe: moe, s: s,
                              x16: sc.ln2, dn32: sc.dn32, dn16: sc.dn16,
                              rows: 1, geometry: driver.geometry, driver: driver)
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
        try QWMoEChain.encode(enc: &active, hal: driver.hal, moe: moe, s: s,
                              x16: driver.prefixRows(psc.ln2, rows: T, cols: H, dt: .f16),
                              dn32: psc.dn32, dn16: psc.dn16,
                              rows: T, geometry: geometry, driver: driver)
    }
}
