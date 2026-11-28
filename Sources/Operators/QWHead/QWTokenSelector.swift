// QWTokenSelector.swift — L5.0 GPU greedy & stochastic token selector and LC verifier.
import Foundation
import Metal
import QWAlloc
import QWCore
import QWHAL

public enum QWGreedyOrdering: Sendable, Equatable {
    case numericalDraft
    case shippingTarget
}

public struct QWStochasticSelectionParams: Sendable, Equatable {
    public var temperature: Float
    public var topK: Int
    public var topP: Float
    public var repetitionPenalty: Float
    public var seed: UInt64
    public var philoxOffsetBase: UInt64

    public init(
        temperature: Float, topK: Int, topP: Float,
        repetitionPenalty: Float = 1.0, seed: UInt64, philoxOffsetBase: UInt64
    ) {
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.seed = seed
        self.philoxOffsetBase = philoxOffsetBase
    }
}

public struct QWSparseProposalHeader: Sendable, Equatable {
    public let token: Int
    public let isSparse: Bool
    public let overflow: Bool
    public let isFinite: Bool
    public let count: Int
    public let maxLogit: Float
    public let logSumExp: Float
    public let logZ: Float
    public let temperature: Float
    public let topK: Int
    public let topP: Float
    public let gumbelToken: Int
    public let gumbelLogProb: Float

    public init(
        token: Int, isSparse: Bool, overflow: Bool, isFinite: Bool = true,
        count: Int, maxLogit: Float, logSumExp: Float, logZ: Float,
        temperature: Float, topK: Int, topP: Float,
        gumbelToken: Int = -1, gumbelLogProb: Float = 0
    ) {
        self.token = token; self.isSparse = isSparse; self.overflow = overflow; self.isFinite = isFinite
        self.count = count; self.maxLogit = maxLogit; self.logSumExp = logSumExp; self.logZ = logZ
        self.temperature = temperature; self.topK = topK; self.topP = topP
        self.gumbelToken = gumbelToken; self.gumbelLogProb = gumbelLogProb
    }
}

public struct QWSparseProposalBuffer: Sendable, Equatable {
    public let header: QWSparseProposalHeader
    public let ids: [Int]
    public let probs: [Double]

    public var token: Int { header.token }
    public var isSparse: Bool { header.isSparse }
    public var overflow: Bool { header.overflow }
    public var isFinite: Bool { header.isFinite }
    public var maxLogit: Float { header.maxLogit }
    public var logSumExp: Float { header.logSumExp }
    public var logZ: Float { header.logZ }
    public var temperature: Float { header.temperature }
    public var topK: Int { header.topK }
    public var topP: Float { header.topP }
    public var gumbelToken: Int { header.gumbelToken }
    public var gumbelLogProb: Float { header.gumbelLogProb }

    public init(header: QWSparseProposalHeader, ids: [Int], probs: [Double]) {
        self.header = header
        self.ids = ids
        self.probs = probs
    }

    public init(
        token: Int, isSparse: Bool, overflow: Bool, isFinite: Bool = true,
        ids: [Int], probs: [Double], maxLogit: Float = 0, logSumExp: Float = 0,
        logZ: Float = 0, temperature: Float = 1.0, topK: Int = 64, topP: Float = 0.9
    ) {
        self.header = QWSparseProposalHeader(
            token: token, isSparse: isSparse, overflow: overflow, isFinite: isFinite,
            count: ids.count, maxLogit: maxLogit, logSumExp: logSumExp, logZ: logZ,
            temperature: temperature, topK: topK, topP: topP
        )
        self.ids = ids
        self.probs = probs
    }
}

public struct QWGPUVerifyResolution: Sendable, Equatable {
    public let resolved: Bool
    public let acceptedCount: Int
    public let tokens: [Int]

    public init(resolved: Bool, acceptedCount: Int, tokens: [Int]) {
        self.resolved = resolved
        self.acceptedCount = acceptedCount
        self.tokens = tokens
    }
}

/// GPU greedy and stochastic token selector with sparse top-k/top-p export and LC verifier.
public struct QWTokenSelector: Sendable {
    public static let maxSparseCandidates = 512
    public static let maxRepetitionTokens = 256
    private static let statsThreadgroups = 256
    private static let headerStrideWords = 16

    public let vocabLimit: Int
    public let vocabulary: Int
    public let maxRows: Int
    public let keysHi: QWTensorValue
    public let keysLo: QWTensorValue
    public let token: QWTensorValue
    public let tgMax: QWTensorValue
    public let tgSum: QWTensorValue
    public let tgFinite: QWTensorValue
    public let tgGHi: QWTensorValue
    public let tgGLo: QWTensorValue
    public let candIds: QWTensorValue
    public let candLogits: QWTensorValue
    public let candCounts: QWTensorValue
    public let candOverflows: QWTensorValue
    public let proposalHeaders: QWTensorValue
    public let proposalIds: QWTensorValue
    public let proposalProbs: QWTensorValue
    public let repTokens: QWTensorValue
    public let repCounts: QWTensorValue
    public let verifyResolvedTokens: QWTensorValue
    public let verifyMeta: QWTensorValue

    public static func allocationBytes(vocabLimit: Int, vocabulary: Int, maxRows: Int = 1) -> Int {
        precondition(vocabLimit >= 1 && (1...vocabLimit).contains(vocabulary) && maxRows >= 1)
        let slots = (vocabulary + 255) / 256
        let statsElems = maxRows * statsThreadgroups
        let candElems = maxRows * maxSparseCandidates
        let repElems = maxRows * maxRepetitionTokens
        let sizes = [
            maxRows * slots * 4, maxRows * slots * 4, 4,
            statsElems * 4, statsElems * 4, statsElems * 4, statsElems * 4, statsElems * 4,
            candElems * 4, candElems * 4, maxRows * 4, maxRows * 4,
            maxRows * headerStrideWords * 4, candElems * 4, candElems * 4,
            repElems * 4, maxRows * 4, maxRows * 4, 4 * 4,
        ]
        return sizes.reduce(0) { $0 + qwAlignUp($1, 256) }
    }

    public init(vocabLimit: Int, allocator: QWSubAllocator, vocabulary: Int, maxRows: Int = 1) throws {
        guard vocabLimit >= 1, (1...vocabLimit).contains(vocabulary), maxRows >= 1 else {
            throw QWHeadError.badShape("token selector vocabulary or row capacity is invalid")
        }
        self.vocabLimit = vocabLimit
        self.vocabulary = vocabulary
        self.maxRows = maxRows
        let slots = (vocabulary + 255) / 256
        let statsElems = maxRows * Self.statsThreadgroups
        let candElems = maxRows * Self.maxSparseCandidates
        let repElems = maxRows * Self.maxRepetitionTokens
        keysHi = try allocator.allocate(QWTensorShape([maxRows * slots], .i32))
        keysLo = try allocator.allocate(QWTensorShape([maxRows * slots], .i32))
        token = try allocator.allocate(QWTensorShape([1], .i32))
        tgMax = try allocator.allocate(QWTensorShape([statsElems], .f32))
        tgSum = try allocator.allocate(QWTensorShape([statsElems], .f32))
        tgFinite = try allocator.allocate(QWTensorShape([statsElems], .i32))
        tgGHi = try allocator.allocate(QWTensorShape([statsElems], .i32))
        tgGLo = try allocator.allocate(QWTensorShape([statsElems], .i32))
        candIds = try allocator.allocate(QWTensorShape([candElems], .i32))
        candLogits = try allocator.allocate(QWTensorShape([candElems], .f32))
        candCounts = try allocator.allocate(QWTensorShape([maxRows], .i32))
        candOverflows = try allocator.allocate(QWTensorShape([maxRows], .i32))
        proposalHeaders = try allocator.allocate(QWTensorShape([maxRows * Self.headerStrideWords], .i32))
        proposalIds = try allocator.allocate(QWTensorShape([candElems], .i32))
        proposalProbs = try allocator.allocate(QWTensorShape([candElems], .f32))
        repTokens = try allocator.allocate(QWTensorShape([repElems], .i32))
        repCounts = try allocator.allocate(QWTensorShape([maxRows], .i32))
        verifyResolvedTokens = try allocator.allocate(QWTensorShape([maxRows], .i32))
        verifyMeta = try allocator.allocate(QWTensorShape([4], .i32))
    }

    public func encodeGreedy(
        _ enc: any MTLComputeCommandEncoder,
        hal: QWHal,
        logits: QWTensorValue,
        ordering: QWGreedyOrdering = .numericalDraft,
        destination: QWTensorValue? = nil
    ) throws {
        try encodeGreedyBatched(
            enc, hal: hal, logits: logits, rows: 1,
            ordering: ordering, destination: destination ?? token
        )
    }

    public func encodeGreedyBatched(
        _ enc: any MTLComputeCommandEncoder,
        hal: QWHal,
        logits: QWTensorValue,
        rows: Int,
        ordering: QWGreedyOrdering = .shippingTarget,
        destination: QWTensorValue
    ) throws {
        guard (1...maxRows).contains(rows),
              logits.shape.dtype == .f32, logits.shape.elementCount == rows * vocabulary,
              Self.hasValidWordExtent(logits),
              destination.shape.dtype == .i32, destination.shape.elementCount >= rows,
              Self.hasValidWordExtent(destination) else {
            throw QWHeadError.badShape("token selector greedy logits or destination extent is invalid")
        }
        try validateNoOverlap(logits, against: [keysHi, keysLo, destination], label: "greedy")
        let slots = (vocabulary + 255) / 256
        let first = try hal.pipeline(ordering == .numericalDraft
            ? "qwg_mtp_argmax_checked" : "qwg_mtp_argmax_target_checked")
        let final = try hal.pipeline("qwg_head_argmax_fin")
        enc.setComputePipelineState(first)
        enc.bindTensors(logits, keysHi, keysLo, startIndex: 0)
        enc.setUInt32s(vocabulary, startIndex: 3)
        enc.dispatchThreadgroups(MTLSize(width: slots, height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.memoryBarrier(scope: .buffers)
        enc.setComputePipelineState(final)
        enc.bindTensors(keysHi, keysLo, destination, startIndex: 0)
        enc.setUInt32s(slots, startIndex: 3)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
    }

    public func encodeStochasticDraft(
        _ enc: any MTLComputeCommandEncoder,
        hal: QWHal,
        logits: QWTensorValue,
        params: QWStochasticSelectionParams,
        rows: Int = 1,
        proposalSlotBase: Int = 0,
        destination: QWTensorValue? = nil,
        journal: QWTensorValue? = nil,
        journalOffset: Int = 0,
        repetitionHistoryByRow: [[Int]] = []
    ) throws {
        let dest = destination ?? token
        guard (1...maxRows).contains(rows),
              proposalSlotBase >= 0, proposalSlotBase + rows <= maxRows,
              logits.shape.dtype == .f32, logits.shape.elementCount == rows * vocabulary,
              Self.hasValidWordExtent(logits),
              dest.shape.dtype == .i32, dest.shape.elementCount >= rows,
              Self.hasValidWordExtent(dest) else {
            throw QWHeadError.badShape("stochastic draft selector logits, slot, or destination invalid")
        }
        if let journal {
            guard journal.shape.dtype == .i32,
                  journalOffset >= 0, journalOffset + rows <= journal.shape.elementCount,
                  Self.hasValidWordExtent(journal) else {
                throw QWHeadError.badShape("stochastic draft selector journal invalid")
            }
        }
        try validateStochasticParams(params)
        var scratchList = [
            tgMax, tgSum, tgFinite, tgGHi, tgGLo, candIds, candLogits,
            candCounts, candOverflows, proposalHeaders, proposalIds, proposalProbs, dest,
        ]
        if let journal { scratchList.append(journal) }
        try validateNoOverlap(logits, against: scratchList, label: "stochastic draft")

        let hasRep = try stageRepetitionMask(
            repetitionHistoryByRow, rows: rows, slotBase: proposalSlotBase, penalty: params.repetitionPenalty
        )
        let psoStats = try hal.pipeline("qwg_spec_stats")
        let psoStatsFin = try hal.pipeline("qwg_spec_stats_fin")
        let isPureT = params.topP >= 1.0 && params.topK >= vocabulary
        let psoCollect = isPureT ? nil : try hal.pipeline("qwg_spec_collect_trunc")
        let psoSample = isPureT ? nil : try hal.pipeline("qwg_spec_sample_trunc")
        let effTopK = UInt32(min(params.topK, vocabulary))
        let journalTarget = journal ?? dest
        let hasJournal: UInt32 = journal != nil ? 1 : 0
        let jOff = UInt32(journal != nil ? journalOffset : 0)

        encodeStatsAndFin(
            enc, psoStats: psoStats, psoStatsFin: psoStatsFin, logits: logits, rows: rows,
            params: params, effTopK: effTopK, hasRep: hasRep, repSlotBase: proposalSlotBase,
            computeGumbel: 1, isDraft: 1, proposalSlotBase: proposalSlotBase,
            draftTokenOut: dest, journalOut: journalTarget, hasJournal: hasJournal, journalOffset: jOff
        )
        if let psoCollect, let psoSample {
            enc.memoryBarrier(scope: .buffers)
            encodeCollect(
                enc, psoCollect: psoCollect, logits: logits, rows: rows,
                repPenalty: params.repetitionPenalty, hasRep: hasRep,
                repSlotBase: proposalSlotBase, proposalSlotBase: proposalSlotBase
            )
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(psoSample)
            enc.bindTensors(proposalHeaders, candIds, candLogits, candCounts, candOverflows,
                            proposalIds, proposalProbs, dest, journalTarget, startIndex: 0)
            enc.setUInt32s(vocabulary, startIndex: 9)
            var temp = params.temperature, kU = effTopK, topP = params.topP
            enc.setBytes(&temp, length: 4, index: 10)
            enc.setBytes(&kU, length: 4, index: 11)
            enc.setBytes(&topP, length: 4, index: 12)
            bindPhilox(enc, params: params, startIndex: 13)
            enc.setUInt32s(proposalSlotBase, Int(hasJournal), Int(jOff), startIndex: 17)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: rows, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
    }

    public func encodeStochasticVerifyLC(
        _ enc: any MTLComputeCommandEncoder,
        hal: QWHal,
        logits: QWTensorValue,
        draftProposals: QWTokenSelector,
        draftCount: Int,
        params: QWStochasticSelectionParams,
        draftProposalSlotBase: Int = 0,
        repetitionHistoryByRow: [[Int]] = []
    ) throws {
        let rows = draftCount + 1
        guard draftCount >= 1, rows <= maxRows,
              draftProposalSlotBase >= 0, draftProposalSlotBase + draftCount <= draftProposals.maxRows,
              logits.shape.dtype == .f32, logits.shape.elementCount == rows * vocabulary,
              Self.hasValidWordExtent(logits) else {
            throw QWHeadError.badShape("GPU LC verify logits or draft count invalid")
        }
        try validateStochasticParams(params)
        try validateNoOverlap(logits, against: [
            tgMax, tgSum, tgFinite, tgGHi, tgGLo, candIds, candLogits,
            candCounts, candOverflows, proposalHeaders, verifyResolvedTokens, verifyMeta,
        ], label: "stochastic verify")

        let hasRep = try stageRepetitionMask(
            repetitionHistoryByRow, rows: rows, slotBase: 0, penalty: params.repetitionPenalty
        )
        let psoStats = try hal.pipeline("qwg_spec_stats")
        let psoStatsFin = try hal.pipeline("qwg_spec_stats_fin")
        let psoCollect = try hal.pipeline("qwg_spec_collect_trunc")
        let psoResolve = try hal.pipeline("qwg_spec_verify_lc_resolve")
        let effTopK = UInt32(min(params.topK, vocabulary))

        encodeStatsAndFin(
            enc, psoStats: psoStats, psoStatsFin: psoStatsFin, logits: logits, rows: rows,
            params: params, effTopK: effTopK, hasRep: hasRep, repSlotBase: 0,
            computeGumbel: 0, isDraft: 0, proposalSlotBase: 0,
            draftTokenOut: verifyResolvedTokens, journalOut: verifyResolvedTokens, hasJournal: 0, journalOffset: 0
        )
        enc.memoryBarrier(scope: .buffers)
        encodeCollect(
            enc, psoCollect: psoCollect, logits: logits, rows: rows,
            repPenalty: params.repetitionPenalty, hasRep: hasRep, repSlotBase: 0, proposalSlotBase: 0
        )
        enc.memoryBarrier(scope: .buffers)
        enc.setComputePipelineState(psoResolve)
        enc.bindTensors(proposalHeaders, candIds, candLogits, candCounts, candOverflows, startIndex: 0)
        enc.bindTensors(draftProposals.proposalHeaders, draftProposals.proposalIds,
                        draftProposals.proposalProbs, verifyResolvedTokens, verifyMeta, startIndex: 5)
        enc.setUInt32s(draftCount, vocabulary, startIndex: 10)
        var temp = params.temperature, kU = effTopK, topP = params.topP
        enc.setBytes(&temp, length: 4, index: 12)
        enc.setBytes(&kU, length: 4, index: 13)
        enc.setBytes(&topP, length: 4, index: 14)
        bindPhilox(enc, params: params, startIndex: 15)
        enc.setUInt32s(draftProposalSlotBase, startIndex: 19)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    public func completedToken() throws -> Int {
        let ptr = token.region.pointer.advanced(by: token.offset).assumingMemoryBound(to: Int32.self)
        return try withExtendedLifetime(token.region.buffer) {
            let id = Int(ptr.pointee)
            guard (0..<vocabulary).contains(id) else {
                throw QWHeadError.badShape("token selector rejected non-finite logits or invalid token \(id)")
            }
            return id
        }
    }

    public func completedTokens(from destination: QWTensorValue, count: Int) throws -> [Int] {
        guard count > 0, count <= destination.shape.elementCount else {
            throw QWHeadError.badShape("token selector active row count is invalid")
        }
        let ids = destination.region.pointer.advanced(by: destination.offset).assumingMemoryBound(to: Int32.self)
        return try withExtendedLifetime(destination.region.buffer) {
            var out: [Int] = []
            out.reserveCapacity(count)
            for row in 0..<count {
                let id = Int(ids[row])
                guard (0..<vocabulary).contains(id) else {
                    throw QWHeadError.badShape("token selector row \(row) rejected non-finite logits or invalid token \(id)")
                }
                out.append(id)
            }
            return out
        }
    }

    public func readSparseProposal(row: Int = 0) throws -> QWSparseProposalBuffer {
        guard (0..<maxRows).contains(row) else {
            throw QWHeadError.badShape("sparse proposal row \(row) out of range 0..<\(maxRows)")
        }
        let base = proposalHeaders.region.pointer.advanced(by: proposalHeaders.offset)
            .assumingMemoryBound(to: UInt32.self)
            .advanced(by: row * Self.headerStrideWords)
        let header = try withExtendedLifetime(proposalHeaders.region.buffer) {
            let tok = Int(Int32(bitPattern: base[0]))
            let isFinite = base[4] != 0
            guard isFinite, (0..<vocabulary).contains(tok) else {
                throw QWHeadError.badShape("stochastic proposal row \(row) rejected non-finite logits or token \(tok)")
            }
            return QWSparseProposalHeader(
                token: tok, isSparse: base[1] == 0, overflow: base[3] != 0, isFinite: isFinite,
                count: Int(base[2]), maxLogit: Float(bitPattern: base[5]),
                logSumExp: Float(bitPattern: base[6]), logZ: Float(bitPattern: base[7]),
                temperature: Float(bitPattern: base[10]), topK: Int(base[11]), topP: Float(bitPattern: base[12]),
                gumbelToken: Int(Int32(bitPattern: base[13])), gumbelLogProb: Float(bitPattern: base[14])
            )
        }
        guard header.isSparse else {
            return QWSparseProposalBuffer(header: header, ids: [], probs: [])
        }
        guard (1...Self.maxSparseCandidates).contains(header.count) else {
            throw QWHeadError.badShape("sparse proposal count \(header.count) invalid")
        }
        let idPtr = proposalIds.region.pointer.advanced(by: proposalIds.offset)
            .assumingMemoryBound(to: UInt32.self)
            .advanced(by: row * Self.maxSparseCandidates)
        let ids: [Int] = try withExtendedLifetime(proposalIds.region.buffer) {
            var out: [Int] = []
            out.reserveCapacity(header.count)
            for i in 0..<header.count {
                let id = Int(idPtr[i])
                guard (0..<vocabulary).contains(id) else {
                    throw QWHeadError.badShape("sparse proposal token \(id) outside 0..<\(vocabulary)")
                }
                out.append(id)
            }
            return out
        }
        let pPtr = proposalProbs.region.pointer.advanced(by: proposalProbs.offset)
            .assumingMemoryBound(to: Float.self)
            .advanced(by: row * Self.maxSparseCandidates)
        let probs: [Double] = withExtendedLifetime(proposalProbs.region.buffer) {
            var out: [Double] = []
            out.reserveCapacity(header.count)
            for i in 0..<header.count { out.append(Double(pPtr[i])) }
            return out
        }
        return QWSparseProposalBuffer(header: header, ids: ids, probs: probs)
    }

    public func readGPUVerifyResolution(draftCount: Int) throws -> QWGPUVerifyResolution {
        guard draftCount >= 1, draftCount + 1 <= maxRows else {
            throw QWHeadError.badShape("GPU verify draftCount \(draftCount) exceeds maxRows \(maxRows)")
        }
        let meta = verifyMeta.region.pointer.advanced(by: verifyMeta.offset).assumingMemoryBound(to: UInt32.self)
        let (status, accepted) = withExtendedLifetime(verifyMeta.region.buffer) { () -> (UInt32, Int) in
            return (meta[0], Int(meta[1]))
        }
        if status == 2 {
            throw QWHeadError.badShape("GPU LC verification rejected non-finite logits")
        }
        guard status == 1 else {
            return QWGPUVerifyResolution(resolved: false, acceptedCount: 0, tokens: [])
        }
        guard (0...draftCount).contains(accepted) else {
            throw QWHeadError.badShape("GPU LC verification returned invalid acceptedCount \(accepted)")
        }
        let tokens = try completedTokens(from: verifyResolvedTokens, count: accepted + 1)
        return QWGPUVerifyResolution(resolved: true, acceptedCount: accepted, tokens: tokens)
    }

    private func validateStochasticParams(_ params: QWStochasticSelectionParams) throws {
        guard params.temperature.isFinite, params.temperature > 0, (1 / params.temperature).isFinite,
              params.topK > 0, params.topP.isFinite, params.topP > 0, params.topP <= 1,
              params.repetitionPenalty.isFinite, params.repetitionPenalty > 0 else {
            throw QWHeadError.badShape("invalid stochastic sampling parameters")
        }
    }

    private func stageRepetitionMask(
        _ historyByRow: [[Int]], rows: Int, slotBase: Int, penalty: Float
    ) throws -> UInt32 {
        if historyByRow.isEmpty || penalty == 1.0 { return 0 }
        guard historyByRow.count == rows else {
            throw QWHeadError.badShape("repetition mask row count \(historyByRow.count) != \(rows)")
        }
        let cPtr = repCounts.region.pointer.advanced(by: repCounts.offset)
            .assumingMemoryBound(to: UInt32.self)
        let tPtr = repTokens.region.pointer.advanced(by: repTokens.offset)
            .assumingMemoryBound(to: UInt32.self)
        var any = false
        try withExtendedLifetime(repCounts.region.buffer) {
            try withExtendedLifetime(repTokens.region.buffer) {
                for r in 0..<rows {
                    let rowHist = historyByRow[r]
                    guard rowHist.count <= Self.maxRepetitionTokens else {
                        throw QWHeadError.badShape("repetition mask exceeds \(Self.maxRepetitionTokens) tokens")
                    }
                    let slot = slotBase + r
                    cPtr[slot] = UInt32(rowHist.count)
                    let base = slot * Self.maxRepetitionTokens
                    for (i, tok) in rowHist.enumerated() {
                        guard (0..<vocabulary).contains(tok) else {
                            throw QWHeadError.badShape("repetition token \(tok) outside 0..<\(vocabulary)")
                        }
                        tPtr[base + i] = UInt32(tok)
                    }
                    if !rowHist.isEmpty { any = true }
                }
            }
        }
        return any ? 1 : 0
    }

    private func bindPhilox(
        _ enc: any MTLComputeCommandEncoder, params: QWStochasticSelectionParams, startIndex: Int
    ) {
        enc.setPod(UInt32(truncatingIfNeeded: params.seed >> 32), index: startIndex)
        enc.setPod(UInt32(truncatingIfNeeded: params.seed), index: startIndex + 1)
        enc.setPod(UInt32(truncatingIfNeeded: params.philoxOffsetBase), index: startIndex + 2)
        enc.setPod(UInt32(truncatingIfNeeded: params.philoxOffsetBase >> 32), index: startIndex + 3)
    }

    private func encodeStatsAndFin(
        _ enc: any MTLComputeCommandEncoder,
        psoStats: any MTLComputePipelineState,
        psoStatsFin: any MTLComputePipelineState,
        logits: QWTensorValue,
        rows: Int,
        params: QWStochasticSelectionParams,
        effTopK: UInt32,
        hasRep: UInt32,
        repSlotBase: Int,
        computeGumbel: UInt32,
        isDraft: UInt32,
        proposalSlotBase: Int,
        draftTokenOut: QWTensorValue,
        journalOut: QWTensorValue,
        hasJournal: UInt32,
        journalOffset: UInt32
    ) {
        var invT = 1.0 / params.temperature, repPen = params.repetitionPenalty
        var temp = params.temperature, topK = effTopK, topP = params.topP

        enc.setComputePipelineState(psoStats)
        enc.bindTensors(logits, repTokens, repCounts, tgMax, tgSum, tgFinite, tgGHi, tgGLo, startIndex: 0)
        enc.setUInt32s(vocabulary, startIndex: 8)
        enc.setBytes(&invT, length: 4, index: 9)
        enc.setBytes(&repPen, length: 4, index: 10)
        enc.setUInt32s(Int(hasRep), repSlotBase, Int(computeGumbel), startIndex: 11)
        bindPhilox(enc, params: params, startIndex: 14)
        enc.dispatchThreadgroups(MTLSize(width: Self.statsThreadgroups, height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        enc.memoryBarrier(scope: .buffers)
        enc.setComputePipelineState(psoStatsFin)
        enc.bindTensors(logits, repTokens, repCounts, tgMax, tgSum, tgFinite, tgGHi, tgGLo,
                        proposalHeaders, candCounts, candOverflows, draftTokenOut, journalOut, startIndex: 0)
        enc.setUInt32s(vocabulary, startIndex: 13)
        enc.setBytes(&temp, length: 4, index: 14)
        enc.setBytes(&topK, length: 4, index: 15)
        enc.setBytes(&topP, length: 4, index: 16)
        enc.setBytes(&repPen, length: 4, index: 17)
        enc.setUInt32s(Int(hasRep), repSlotBase, Int(computeGumbel), Int(isDraft),
                       proposalSlotBase, Int(hasJournal), Int(journalOffset), startIndex: 18)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    private func encodeCollect(
        _ enc: any MTLComputeCommandEncoder,
        psoCollect: any MTLComputePipelineState,
        logits: QWTensorValue,
        rows: Int,
        repPenalty: Float,
        hasRep: UInt32,
        repSlotBase: Int,
        proposalSlotBase: Int
    ) {
        var repPen = repPenalty
        enc.setComputePipelineState(psoCollect)
        enc.bindTensors(logits, repTokens, repCounts, proposalHeaders,
                        candIds, candLogits, candCounts, candOverflows, startIndex: 0)
        enc.setUInt32s(vocabulary, startIndex: 8)
        enc.setBytes(&repPen, length: 4, index: 9)
        enc.setUInt32s(Int(hasRep), repSlotBase, proposalSlotBase, startIndex: 10)
        enc.dispatchThreadgroups(MTLSize(width: Self.statsThreadgroups, height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    private func validateNoOverlap(_ src: QWTensorValue, against targets: [QWTensorValue], label: String) throws {
        let a = src.region.base + src.offset
        for dst in targets where src.region.buffer === dst.region.buffer {
            let b = dst.region.base + dst.offset
            guard a + src.bytes <= b || b + dst.bytes <= a else {
                throw QWHeadError.badShape("token selector \(label) logits overlap scratch")
            }
        }
    }

    private static func hasValidWordExtent(_ tensor: QWTensorValue) -> Bool {
        tensor.offset >= 0 && tensor.region.base >= 0
            && tensor.offset <= tensor.region.length
            && tensor.bytes <= tensor.region.length - tensor.offset
            && tensor.region.base <= tensor.region.buffer.length
            && tensor.region.length <= tensor.region.buffer.length - tensor.region.base
            && (tensor.region.base + tensor.offset) % 4 == 0
    }
}
