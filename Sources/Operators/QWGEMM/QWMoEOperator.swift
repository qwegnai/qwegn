// QWMoEOperator.swift — Unified Sparse-MoE channel-mixer operator for QWEngine.
// Serves both single-stream f16 inputs (via f16->f32 bridge) and pre-rotated f32 Hyper-Connection streams.

import Foundation
import Metal
import QWAlloc
import QWCore
import QWHAL

public enum QWMoEOperator {
    /// Input activation layout entering the Sparse-MoE channel mixer.
    public enum Input {
        /// Normed `f16 [rows, hidden]` activation; the operator runs `encodeMoEBridgeF16ToF32`.
        case f16Normed(QWTensorValue)
        /// Pre-rotated `f32 [rows, hidden]` (`plain` for router, `rotated` for experts, optional `f16` precast for NAX).
        case f32PreRotated(plain: QWTensorValue, rotated: QWTensorValue, prefillPrecast16: QWTensorValue?)
    }

    /// Optional decode router encoder `(enc, hal, weight, x, logits, sg, rows, experts, k)`.
    public typealias DecodeRouterEncoder = (
        any MTLComputeCommandEncoder, QWHal, QWTensorValue, QWTensorValue,
        QWTensorValue, QWTensorValue, Int, Int, Int
    ) throws -> Void

    /// Unified Sparse-MoE layer weights view.
    public struct Weights {
        public let router: QWTensorValue
        public let routerBias: QWTensorValue?
        public let decodeRouterOverride: DecodeRouterEncoder?
        public let gateUp: QWWeightView
        public let down: QWWeightView
        public let coldGateUp: QWWeightView?
        public let coldDown: QWWeightView?
        public let sharedGateUp: QWWeightView?
        public let sharedDown: QWWeightView?
        public let sharedGateScalar: QWTensorValue?
        public let hidden: Int
        public let experts: Int
        public let topK: Int
        public let intermediate: Int
        public let sharedIntermediate: Int
        public let normTopK: Bool
        public let reuseExpertsAcrossTokens: Bool
        public let moeSpecExpertBudget: Int

        public init(router: QWTensorValue, routerBias: QWTensorValue? = nil,
                    decodeRouterOverride: DecodeRouterEncoder? = nil,
                    gateUp: QWWeightView, down: QWWeightView,
                    coldGateUp: QWWeightView? = nil, coldDown: QWWeightView? = nil,
                    sharedGateUp: QWWeightView? = nil, sharedDown: QWWeightView? = nil,
                    sharedGateScalar: QWTensorValue? = nil,
                    hidden: Int, experts: Int, topK: Int,
                    intermediate: Int, sharedIntermediate: Int,
                    normTopK: Bool = true, reuseExpertsAcrossTokens: Bool = true,
                    moeSpecExpertBudget: Int = 0) {
            self.router = router
            self.routerBias = routerBias
            self.decodeRouterOverride = decodeRouterOverride
            self.gateUp = gateUp
            self.down = down
            self.coldGateUp = coldGateUp
            self.coldDown = coldDown
            self.sharedGateUp = sharedGateUp
            self.sharedDown = sharedDown
            self.sharedGateScalar = sharedGateScalar
            self.hidden = hidden
            self.experts = experts
            self.topK = topK
            self.intermediate = intermediate
            self.sharedIntermediate = sharedIntermediate
            self.normTopK = normTopK
            self.reuseExpertsAcrossTokens = reuseExpertsAcrossTokens
            self.moeSpecExpertBudget = moeSpecExpertBudget
        }
    }

    /// Unified Sparse-MoE scratch tensors for decode (`rows = 1`), verify (`rows <= 8`), and prefill (`rows > 8`).
    public struct Scratch {
        public let xRot32: QWTensorValue
        public let xPlain32: QWTensorValue
        public let logits: QWTensorValue
        public let idx: QWTensorValue
        public let wts: QWTensorValue
        public let gu32: QWTensorValue
        public let act32: QWTensorValue
        public let guSh32: QWTensorValue?
        public let actSh32: QWTensorValue?
        public let shared: QWTensorValue
        public let gate: QWTensorValue
        public let expOff: QWTensorValue?
        public let sortedPair: QWTensorValue?
        public let sortedTok: QWTensorValue?
        public let invPerm: QWTensorValue?
        public let tileE: QWTensorValue?
        public let tileR: QWTensorValue?
        public let tileCount: QWTensorValue?
        public let gu16: QWTensorValue?
        public let dnHi: QWTensorValue?
        public let dnAcc: QWTensorValue
        public let naxX16: QWTensorValue?
        public let naxY16: QWTensorValue?
        public let maxTileWidth: Int

        public init(xRot32: QWTensorValue, xPlain32: QWTensorValue,
                    logits: QWTensorValue, idx: QWTensorValue, wts: QWTensorValue,
                    gu32: QWTensorValue, act32: QWTensorValue,
                    guSh32: QWTensorValue?, actSh32: QWTensorValue?,
                    shared: QWTensorValue, gate: QWTensorValue,
                    expOff: QWTensorValue?, sortedPair: QWTensorValue?,
                    sortedTok: QWTensorValue?, invPerm: QWTensorValue?,
                    tileE: QWTensorValue?, tileR: QWTensorValue?, tileCount: QWTensorValue?,
                    gu16: QWTensorValue?, dnHi: QWTensorValue?, dnAcc: QWTensorValue,
                    naxX16: QWTensorValue?, naxY16: QWTensorValue? = nil, maxTileWidth: Int) {
            self.xRot32 = xRot32; self.xPlain32 = xPlain32
            self.logits = logits; self.idx = idx; self.wts = wts
            self.gu32 = gu32; self.act32 = act32
            self.guSh32 = guSh32; self.actSh32 = actSh32
            self.shared = shared; self.gate = gate
            self.expOff = expOff; self.sortedPair = sortedPair
            self.sortedTok = sortedTok; self.invPerm = invPerm
            self.tileE = tileE; self.tileR = tileR; self.tileCount = tileCount
            self.gu16 = gu16; self.dnHi = dnHi; self.dnAcc = dnAcc
            self.naxX16 = naxX16; self.naxY16 = naxY16
            self.maxTileWidth = maxTileWidth
        }
    }

    /// Encodes an all-resident Sparse-MoE pass over `rows` tokens into `out32` (and optional `out16`).
    public static func encodeResident(
        _ enc: inout any MTLComputeCommandEncoder,
        hal: QWHal,
        weights w: Weights,
        scratch s: Scratch,
        input: Input,
        out32: QWTensorValue,
        out16: QWTensorValue? = nil,
        rows T: Int,
        supportsGroupedNaxPrefill: Bool = true,
        concurrentBarriers: Bool = false,
        rotateEncoder: ((inout any MTLComputeCommandEncoder) throws -> Void)? = nil,
        profileStage: ((String) -> (any MTLComputeCommandEncoder)?)? = nil
    ) throws {
        @inline(__always) func barrier() {
            if concurrentBarriers { enc.memoryBarrier(scope: .buffers) }
        }
        let H = w.hidden, E = w.experts, K = w.topK, I = w.intermediate, ISh = w.sharedIntermediate
        let isGroupedNax = T > 8 && supportsGroupedNaxPrefill
            && s.expOff != nil && s.sortedPair != nil && s.sortedTok != nil
            && s.invPerm != nil && s.tileE != nil && s.tileR != nil
            && s.tileCount != nil && s.gu16 != nil && s.dnHi != nil && s.naxX16 != nil
        let skipRot32 = isGroupedNax && (ISh == 0 || (T >= 64 && H % 64 == 0 && (2 * ISh) % 64 == 0))

        let xPlain: QWTensorValue
        let xRot: QWTensorValue
        let xe16ForNax: QWTensorValue?
        let shPrecast16: QWTensorValue?
        switch input {
        case .f16Normed(let x16):
            try QWGemm.encodeMoEBridgeF16ToF32(enc, hal: hal, x: x16,
                                               xRot32: s.xRot32, xPlain32: s.xPlain32,
                                               rows: T, hidden: H, skipRot32: skipRot32)
            xPlain = s.xPlain32
            xRot = s.xRot32
            xe16ForNax = x16
            shPrecast16 = T >= 64 ? x16 : nil
        case .f32PreRotated(let plain, let rotated, let precast16):
            xPlain = plain
            xRot = rotated
            xe16ForNax = precast16 ?? s.naxX16
            shPrecast16 = T >= 64 ? (precast16 ?? s.naxX16) : nil
        }

        if isGroupedNax,
           let expOff = s.expOff, let sortedPair = s.sortedPair,
           let sortedTok = s.sortedTok, let invPerm = s.invPerm,
           let tileE = s.tileE, let tileR = s.tileR, let tileCount = s.tileCount,
           let gu16 = s.gu16, let dnHi = s.dnHi,
           let naxX16 = s.naxX16, let xe16 = xe16ForNax {
            let shAct16 = (T >= 64 && ISh > 0 && ISh % 64 == 0)
                ? naxX16.slice(byteOffset: T * H * 2, count: T * ISh, dtype: .f16)
                : nil
            let dnAcc16 = s.dnAcc.slice(byteOffset: 0, count: T * K * H, dtype: .f16)
            try QWGemm.encodeFnRouterGemm(enc, hal: hal, weight: w.router, x: xPlain, y: s.logits,
                                          rows: T, n: E, k: H,
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
                                        rows: T, experts: E, topK: K, normTopK: w.normTopK, bias: w.routerBias)
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
            barrier()
            guard let rotateEncoder else {
                throw QWGemmError.badShape("moe grouped nax requires rotateEncoder for indirect dispatch")
            }
            let tilePlan = QWGemm.MoETilePlan(experts: E, pairs: T * K, gateUpRows: 2 * I, downRows: H)
            try QWGemm.encodeMoEGroupPairsF32(enc, hal: hal, ids: s.idx, expertOff: expOff,
                                              sortedPairIdx: sortedPair, sortedTok: sortedTok,
                                              experts: E, totalPairs: T * K, routes: K,
                                              invPerm: invPerm, tileE: tileE, tileR: tileR,
                                              tileCount: tileCount, tilePlan: tilePlan)
            if concurrentBarriers, let shDown = w.sharedDown, let actSh = s.actSh32, ISh > 0 {
                let naxY = s.naxY16 ?? naxX16
                try QWGemm.encodeDenseProjF32(enc, hal: hal, weight: shDown, x: actSh, y: s.shared,
                                              naxX16: naxX16, naxY16: naxY, maxTileWidth: s.maxTileWidth,
                                              rows: T, x16Precast: shAct16)
            }
            try rotateEncoder(&enc)
            try QWGemm.encodeMoEGateUpNax(enc, hal: hal, xe: xe16, bank: w.gateUp, expertOff: expOff,
                                          tileE: tileE, tileR: tileR, tileCount: tileCount,
                                          gu16: gu16, rowsPerExpert: 2 * I, tilePlan: tilePlan,
                                          sortedTok: sortedTok, coldBank: w.coldGateUp)
            barrier()
            try QWGemm.encodeSiluMulSplitF16(enc, hal: hal, gu: gu16, outHi: dnHi, outLo: dnHi,
                                             rows: T * K, intermediate: I)
            barrier()
            try QWGemm.encodeMoENaxSplitF32(enc, hal: hal, xHi: dnHi, xLo: dnHi, bank: w.down,
                                            expertOff: expOff, tileE: tileE, tileR: tileR,
                                            tileCount: tileCount, y32: dnAcc16, rowsPerExpert: H,
                                            tilePlan: tilePlan, coldBank: w.coldDown)
            barrier()
            try QWGemm.encodeMoEDownReduceCombineF32(enc, hal: hal, dnAcc: dnAcc16, invPerm: invPerm,
                                                     weights: s.wts, shared: s.shared, gate: s.gate,
                                                     out: out32, hidden: H, routes: K, tokens: T,
                                                     out16: out16)
            barrier()
            return
        }

        // Rank-1 GEMV decode / verify path (T <= 8 or non-NAX shapes)
        if T <= 8, let decodeRouter = w.decodeRouterOverride {
            try decodeRouter(enc, hal, w.router, xPlain, s.logits, s.gate, T, E, H)
        } else {
            try QWGemm.encodeFnRouterGemm(enc, hal: hal, weight: w.router, x: xPlain, y: s.logits,
                                          rows: T, n: E, k: H,
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
                                    bias: w.routerBias,
                                    expertBudget: w.moeSpecExpertBudget)
        barrier()
        let fuseRoutedGU = I > 0 && (I % 32 == 0)
        if concurrentBarriers {
            try QWGemm.encodeMoEGateUpF32R1(enc, hal: hal, x: xRot, bank: w.gateUp, ids: s.idx,
                                            out: fuseRoutedGU ? s.act32 : s.gu32,
                                            rowsPerExpert: 2 * I, routes: K, tokens: T,
                                            fuseSiluFwht: fuseRoutedGU,
                                            coldBank: w.coldGateUp,
                                            reuseExpertsAcrossTokens: w.reuseExpertsAcrossTokens)
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
                                                     rowsPerRoute: H, routes: K, weights: s.wts,
                                                     shared: s.shared, gate: s.gate, out: out32,
                                                     ids: s.idx, tokens: T, out16: out16,
                                                     coldBank: w.coldDown,
                                                     reuseExpertsAcrossTokens: w.reuseExpertsAcrossTokens)
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
            try QWGemm.encodeMoEGateUpF32R1(enc, hal: hal, x: xRot, bank: w.gateUp, ids: s.idx,
                                            out: fuseRoutedGU ? s.act32 : s.gu32,
                                            rowsPerExpert: 2 * I, routes: K, tokens: T,
                                            fuseSiluFwht: fuseRoutedGU,
                                            coldBank: w.coldGateUp,
                                            reuseExpertsAcrossTokens: w.reuseExpertsAcrossTokens)
            if !fuseRoutedGU {
                try QWGemm.encodeSiluMulF32(enc, hal: hal, gu: s.gu32, act: s.act32, rows: T * K, intermediate: I)
            }
            if let nextEnc = profileStage?("moe.experts_down") { enc = nextEnc }
            try QWGemm.encodeMoEDownSharedCombineF32(enc, hal: hal, x: s.act32, bank: w.down,
                                                     rowsPerRoute: H, routes: K, weights: s.wts,
                                                     shared: s.shared, gate: s.gate, out: out32,
                                                     ids: s.idx, tokens: T, out16: out16,
                                                     coldBank: w.coldDown,
                                                     reuseExpertsAcrossTokens: w.reuseExpertsAcrossTokens)
        }
    }
}
