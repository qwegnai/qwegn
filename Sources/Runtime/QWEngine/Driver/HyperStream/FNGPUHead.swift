import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWHead

public struct FNGPUStochasticVerifyRequest: Sendable {
    public let draftProposals: QWTokenSelector
    public let draftCount: Int
    public let params: QWStochasticSelectionParams
    public let draftProposalSlotBase: Int
    public let repetitionHistoryByRow: [[Int]]

    public init(
        draftProposals: QWTokenSelector,
        draftCount: Int,
        params: QWStochasticSelectionParams,
        draftProposalSlotBase: Int = 0,
        repetitionHistoryByRow: [[Int]] = []
    ) {
        self.draftProposals = draftProposals
        self.draftCount = draftCount
        self.params = params
        self.draftProposalSlotBase = draftProposalSlotBase
        self.repetitionHistoryByRow = repetitionHistoryByRow
    }
}

public struct FNGPUStochasticDraftRequest: Sendable {
    public let params: QWStochasticSelectionParams
    public let proposalSlotBase: Int
    public let journal: QWTensorValue?
    public let journalOffset: Int
    public let repetitionHistoryByRow: [[Int]]

    public init(
        params: QWStochasticSelectionParams,
        proposalSlotBase: Int = 0,
        journal: QWTensorValue? = nil,
        journalOffset: Int = 0,
        repetitionHistoryByRow: [[Int]] = []
    ) {
        self.params = params
        self.proposalSlotBase = proposalSlotBase
        self.journal = journal
        self.journalOffset = journalOffset
        self.repetitionHistoryByRow = repetitionHistoryByRow
    }
}

/// FlashNext tail on the GPU: the final hyper-connection mixer and the LM head.
/// The mixer's down/up and the head stay HQ4 on the GPU (as the pack stores them); the activations are
/// rotated into the Walsh-Hadamard domain instead, so no weight is ever expanded to dense f32.
public final class FNGPUHead {
    public let weight: QWWeightView
    public let vocab: Int, hidden: Int, maxRows: Int
    let mixerDown: QWWeightView, mixerUp: QWWeightView
    let normWeight: QWTensorValue
    let eps: Float
    let width: Int, lowrank: Int, hc: Int

    let xn: QWTensorValue        // f32 [maxRows * width]  normed stream (natural)
    let rotW: QWTensorValue      // f32 [maxRows * width]  rotated stream
    let low: QWTensorValue       // f32 [maxRows * lowrank] silu'd down output (natural)
    let mixed: QWTensorValue     // f32 [maxRows * hidden] mixed hidden
    let rotM: QWTensorValue      // f32 [maxRows * hidden] rotated hidden (head input)
    let logits: QWTensorValue    // f32 [maxRows * vocab]
    let keysHi: QWTensorValue, keysLo: QWTensorValue
    public let token: QWTensorValue     // i32 [maxRows]
    public let selector: QWTokenSelector
    let arena: QWArena
    private var lastRows: Int = 1

    public init(arena: QWArena, hal: QWHal, head: QWWeightView, mixerDown: QWWeightView, mixerUp: QWWeightView,
                normWeight: QWTensorValue, eps: Float, hidden: Int, hc: Int, lowrank: Int,
                maxRows: Int = 2) throws {
        _ = hal
        self.arena = arena
        self.weight = head
        self.mixerDown = mixerDown
        self.mixerUp = mixerUp
        self.normWeight = normWeight
        self.eps = eps
        self.hidden = hidden
        self.hc = hc
        self.lowrank = lowrank
        self.width = hc * hidden
        self.vocab = head.n
        self.maxRows = max(1, maxRows)
        let R = self.maxRows
        let act = arena.activationsRegion()
        xn = try act.allocate(QWTensorShape([R * width], .f32))
        rotW = try act.allocate(QWTensorShape([R * width], .f32))
        low = try act.allocate(QWTensorShape([R * lowrank], .f32))
        mixed = try act.allocate(QWTensorShape([R * hidden], .f32))
        rotM = try act.allocate(QWTensorShape([R * hidden], .f32))
        logits = try act.allocate(QWTensorShape([R * head.n], .f32))
        let slots = QWGemm.blockArgmaxSlots(head.n)
        keysHi = try act.allocate(QWTensorShape([R * slots], .i32))
        keysLo = try act.allocate(QWTensorShape([R * slots], .i32))
        token = try act.allocate(QWTensorShape([R], .i32))
        selector = try QWTokenSelector(
            vocabLimit: head.n,
            allocator: act,
            vocabulary: head.n,
            maxRows: R
        )
    }

    /// The final mixer + LM head + argmax (or stochastic LC verify / draft selection) for `rows` streams.
    public func encode(_ enc: any MTLComputeCommandEncoder, hal: QWHal, stream: QWTensorValue,
                       inInject: (base: QWTensorValue, src: QWTensorValue, scale: QWTensorValue)? = nil,
                       rows: Int = 1,
                       selectRowFrom: (verified: QWTensorValue, draft: QWTensorValue, totalRows: Int)? = nil,
                       variational: QWVariationalAcceptance? = nil,
                       stochasticVerify: FNGPUStochasticVerifyRequest? = nil,
                       stochasticDraft: FNGPUStochasticDraftRequest? = nil) throws {
        precondition(rows >= 1 && rows <= maxRows)
        lastRows = rows
        let scale = Float(hc)
        if let sel = selectRowFrom, let inj = inInject, rows == 1 {
            try QWGemm.encodeFnHCInjectNormRotSel(enc, hal: hal, base: inj.base, src: inj.src, inj: inj.scale,
                                                  xOut: stream, weight: normWeight, xn: xn, rot: rotW,
                                                  verifiedTok: sel.verified, draftTok: sel.draft,
                                                  width: width, hidden: hidden, eps: eps,
                                                  totalRows: sel.totalRows)
        } else if let inj = inInject {
            try QWGemm.encodeFnHCInjectNormRot(enc, hal: hal, base: inj.base, src: inj.src, inj: inj.scale,
                                               xOut: stream, weight: normWeight, xn: xn, rot: rotW,
                                               rows: rows, width: width, hidden: hidden, eps: eps)
        } else {
            try QWGemm.encodeFnHCNormRot(enc, hal: hal, x: stream, weight: normWeight, xn: xn, rot: rotW,
                                         rows: rows, width: width, hidden: hidden, eps: eps)
        }
        enc.memoryBarrier(scope: .buffers)
        try QWGemm.encodeHQ4F32V4(enc, hal: hal, weight: mixerDown, x: rotW, y: low, rows: rows, epi: 1, scale: scale)
        enc.memoryBarrier(scope: .buffers)
        try QWGemm.encodeFnHCUpMixRot(enc, hal: hal, upWeight: mixerUp, hcLow: low, xn: xn,
                                      mixed: mixed, rotMixed: rotM, rows: rows, hidden: hidden, hc: hc)
        enc.memoryBarrier(scope: .buffers)
        let activeLogits = logits.slice(byteOffset: 0, count: rows * vocab, dtype: .f32)
        if let sv = stochasticVerify {
            try QWGemm.encodeHQ4F32V4(enc, hal: hal, weight: weight, x: rotM, y: logits, rows: rows)
            enc.memoryBarrier(scope: .buffers)
            try selector.encodeStochasticVerifyLC(
                enc, hal: hal, logits: activeLogits,
                draftProposals: sv.draftProposals,
                draftCount: sv.draftCount,
                params: sv.params,
                draftProposalSlotBase: sv.draftProposalSlotBase,
                repetitionHistoryByRow: sv.repetitionHistoryByRow
            )
            enc.memoryBarrier(scope: .buffers)
            let copy = try hal.pipeline("qwg_verify_copy")
            enc.setComputePipelineState(copy)
            selector.verifyResolvedTokens.slice(byteOffset: 0, count: rows, dtype: .i32).bind(enc, index: 0)
            token.slice(byteOffset: 0, count: rows, dtype: .i32).bind(enc, index: 1)
            var countU = UInt32(rows)
            enc.setBytes(&countU, length: 4, index: 2)
            enc.dispatchThreads(MTLSize(width: rows, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(32, rows), height: 1, depth: 1))
        } else if let sd = stochasticDraft {
            try QWGemm.encodeHQ4F32V4(enc, hal: hal, weight: weight, x: rotM, y: logits, rows: rows)
            enc.memoryBarrier(scope: .buffers)
            try selector.encodeStochasticDraft(
                enc, hal: hal, logits: activeLogits,
                params: sd.params,
                rows: rows,
                proposalSlotBase: sd.proposalSlotBase,
                destination: token,
                journal: sd.journal,
                journalOffset: sd.journalOffset,
                repetitionHistoryByRow: sd.repetitionHistoryByRow
            )
        } else {
            try QWGemm.encodeHQ4F32V4Argmax(enc, hal: hal, weight: weight, x: rotM, y: logits,
                                            keysHi: keysHi, keysLo: keysLo, rows: rows)
            enc.memoryBarrier(scope: .buffers)
            try QWGemm.encodeArgmaxFin(enc, hal: hal, keysHi: keysHi, keysLo: keysLo,
                                       token: token, slots: QWGemm.blockArgmaxSlots(vocab), rows: rows,
                                       variational: variational)
        }
    }

    /// The argmax token id of the last encoded row (read after the command buffer completes).
    public func argmaxToken() -> Int {
        Int(arena.read(token, as: Int32.self, count: lastRows)[lastRows - 1])
    }

    /// All `count` argmax token ids (row 0 ..< count) from the last multi-row encode.
    public func argmaxTokens(count: Int) -> [Int] {
        arena.read(token, as: Int32.self, count: count).map(Int.init)
    }

    /// The f32 logits for `row` (host read of the GPU buffer).
    public func readLogits(row: Int = 0) -> [Float] {
        let slice = row == 0
            ? logits
            : logits.slice(byteOffset: row * vocab * 4, count: vocab, dtype: .f32)
        return arena.read(slice, as: Float.self, count: vocab)
    }
}
