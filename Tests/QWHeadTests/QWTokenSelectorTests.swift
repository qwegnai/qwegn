// QWTokenSelectorTests.swift — Bit-exact Philox, greedy, sparse top-k/top-p, overflow, and LC verify tests.
import Foundation
import Metal
import XCTest
@testable import QWHead
import QWAlloc
import QWCore
import QWHAL

final class QWTokenSelectorTests: XCTestCase {
    private func makeHal() throws -> QWHal {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("no Metal device") }
        return try QWHal(geometry: .qwen38_27b)
    }

    private func submit(_ hal: QWHal, _ encode: (any MTLComputeCommandEncoder) throws -> Void) throws {
        guard let (cb, enc) = hal.begin() else { throw XCTSkip("no command buffer") }
        do {
            try encode(enc)
        } catch {
            enc.endEncoding()
            throw error
        }
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)
    }

    private func cpuVerifyLC(
        draftProps: [QWSparseProposalBuffer],
        targetLogits: UnsafePointer<Float>,
        vocab: Int,
        params: QWStochasticSelectionParams,
        repByRow: [[Int]]
    ) throws -> QWGPUVerifyResolution {
        let sampler = QWSampler(config: QWSamplerConfig(
            temperature: params.temperature, topK: params.topK, topP: params.topP,
            seed: params.seed, uniformVersion: .upper24, repetitionPenalty: params.repetitionPenalty
        ))
        let dCount = draftProps.count
        var tokens: [Int] = []
        for r in 0..<dCount {
            let qProp = draftProps[r]
            let rowPtr = UnsafeBufferPointer(start: targetLogits.advanced(by: r * vocab), count: vocab)
            let pDist = try sampler.distribution(logits: rowPtr, repetitionHistory: repByRow[r])
            let d = qProp.token
            let pD = zip(pDist.survivorIds, pDist.survivorProbs).first(where: { $0.0 == d })?.1 ?? 0.0
            let qD = zip(qProp.ids, qProp.probs).first(where: { $0.0 == d })?.1 ?? 0.0
            let uAcc = Double(QWPhilox.u(params.philoxOffsetBase + UInt64(2 * r), seed: params.seed, version: .upper24))
            if pD > 0.0 && (qD <= 0.0 || pD >= qD || uAcc < (pD / qD)) {
                tokens.append(d)
            } else {
                var qMap: [Int: Double] = [:]
                for (id, pr) in zip(qProp.ids, qProp.probs) { qMap[id] = pr }
                var resIds: [Int] = []
                var resW: [Double] = []
                for (id, pV) in zip(pDist.survivorIds, pDist.survivorProbs) {
                    let diff = max(0.0, pV - (qMap[id] ?? 0.0))
                    if diff > 0.0 {
                        resIds.append(id)
                        resW.append(diff)
                    }
                }
                let uRes = Double(QWPhilox.u(
                    params.philoxOffsetBase + UInt64(2 * r + 1), seed: params.seed, version: .upper24
                ))
                let resDist = try resIds.isEmpty
                    ? pDist
                    : QWSamplingDistribution.sparse(vocabulary: vocab, ids: resIds, weights: resW)
                tokens.append(try resDist.sampleToken(uniform: uRes, version: .upper24))
                return QWGPUVerifyResolution(resolved: true, acceptedCount: r, tokens: tokens)
            }
        }
        let bonusPtr = UnsafeBufferPointer(start: targetLogits.advanced(by: dCount * vocab), count: vocab)
        let bonusDist = try sampler.distribution(logits: bonusPtr, repetitionHistory: repByRow[dCount])
        let uBonus = Double(QWPhilox.u(
            params.philoxOffsetBase + UInt64(2 * dCount), seed: params.seed, version: .upper24
        ))
        tokens.append(try bonusDist.sampleToken(uniform: uBonus, version: .upper24))
        return QWGPUVerifyResolution(resolved: true, acceptedCount: dCount, tokens: tokens)
    }

    func testMetalVsSwiftPhiloxZeroULPParity() throws {
        let hal = try makeHal()
        let vocab = 2
        let arena = try QWArena(hal: hal, weights: 256, activations: 64 << 10, kv: 256, gdnStates: 256)
        let act = arena.activationsRegion()
        let selector = try QWTokenSelector(vocabLimit: vocab, allocator: act, vocabulary: vocab)
        let logits = try act.allocate(QWTensorShape([vocab], .f32))
        let p = arena.cpuPointer(of: logits).assumingMemoryBound(to: Float.self)
        act.markSteadyState()

        for trial in 0..<24 {
            let seed = UInt64(0x9E37_79B9_7F4A_7C15) &+ UInt64(trial) &* 0x517C_C1B7_2722_0A95
            let offset = UInt64(trial) &* 17 &+ (trial % 2 == 0 ? 0 : UInt64(UInt32.max))
            let u = QWPhilox.u(offset, seed: seed, version: .upper24)
            let params = QWStochasticSelectionParams(
                temperature: 1.0, topK: 2, topP: 0.99999, seed: seed, philoxOffsetBase: offset
            )

            let eps = Float(1.0 / 16_777_216.0)
            let guardBand = Float(1e-5)
            guard u > 2 * guardBand, u < 1.0 - 2 * guardBand else { continue }
            let pBelow = u - guardBand
            p[0] = 0.0
            p[1] = Foundation.log((1.0 - pBelow) / pBelow)
            try submit(hal) { enc in
                try selector.encodeStochasticDraft(enc, hal: hal, logits: logits, params: params)
            }
            let propBelow = try selector.readSparseProposal()
            let cpuBelow = try QWSamplingDistribution.sparse(
                vocabulary: 2, ids: propBelow.ids, weights: propBelow.probs
            ).sampleToken(uniform: Double(u), version: .upper24)
            XCTAssertEqual(propBelow.token, 1, "expected token 1 below Philox u at trial \(trial)")
            XCTAssertEqual(propBelow.token, cpuBelow, "0-ULP draw mismatch below at trial \(trial)")

            let pAbove = min(1.0 - eps, u + guardBand)
            p[0] = 0.0
            p[1] = Foundation.log((1.0 - pAbove) / pAbove)
            try submit(hal) { enc in
                try selector.encodeStochasticDraft(enc, hal: hal, logits: logits, params: params)
            }
            let propAbove = try selector.readSparseProposal()
            let cpuAbove = try QWSamplingDistribution.sparse(
                vocabulary: 2, ids: propAbove.ids, weights: propAbove.probs
            ).sampleToken(uniform: Double(u), version: .upper24)
            XCTAssertEqual(propAbove.token, cpuAbove, "0-ULP draw mismatch above at trial \(trial)")
        }
        XCTAssertEqual(act.steadyStateAllocs, 0)
    }

    func testGreedyParityAtTileAndVocabularyBoundaries() throws {
        let hal = try makeHal()
        for vocab in [1, 33, 256, 257, 1025] {
            let arena = try QWArena(hal: hal, weights: 256, activations: vocab * 16 + 128 << 10, kv: 256, gdnStates: 256)
            let act = arena.activationsRegion()
            let selector = try QWTokenSelector(vocabLimit: vocab, allocator: act, vocabulary: vocab, maxRows: 2)
            XCTAssertEqual(act.usedBytes, QWTokenSelector.allocationBytes(vocabLimit: vocab, vocabulary: vocab, maxRows: 2))
            let logits = try act.allocate(QWTensorShape([2 * vocab], .f32))
            let dest = try act.allocate(QWTensorShape([2], .i32))
            let p = arena.cpuPointer(of: logits).assumingMemoryBound(to: Float.self)
            act.markSteadyState()

            for i in 0..<(2 * vocab) { p[i] = -Float(i % 97 + 1) }
            p[0] = -0.0
            p[vocab - 1] = 0.0
            p[vocab] = 3.0
            p[2 * vocab - 1] = 3.0

            try submit(hal) { enc in
                try selector.encodeGreedy(
                    enc, hal: hal,
                    logits: logits.slice(byteOffset: 0, count: vocab, dtype: .f32),
                    ordering: .numericalDraft
                )
            }
            XCTAssertEqual(try selector.completedToken(), 0)

            try submit(hal) { enc in
                try selector.encodeGreedy(
                    enc, hal: hal,
                    logits: logits.slice(byteOffset: 0, count: vocab, dtype: .f32),
                    ordering: .shippingTarget
                )
            }
            XCTAssertEqual(try selector.completedToken(), vocab > 1 ? vocab - 1 : 0)

            try submit(hal) { enc in
                try selector.encodeGreedyBatched(
                    enc, hal: hal, logits: logits, rows: 2,
                    ordering: .shippingTarget, destination: dest
                )
            }
            XCTAssertEqual(try selector.completedTokens(from: dest, count: 2), [vocab > 1 ? vocab - 1 : 0, 0])

            p[vocab + vocab / 2] = .nan
            try submit(hal) { enc in
                try selector.encodeGreedyBatched(
                    enc, hal: hal, logits: logits, rows: 2,
                    ordering: .shippingTarget, destination: dest
                )
            }
            XCTAssertThrowsError(try selector.completedTokens(from: dest, count: 2))
            XCTAssertEqual(act.steadyStateAllocs, 0)
        }
    }

    func testStochasticPureTAndTopKTopPAndOverflowFallback() throws {
        let hal = try makeHal()
        let vocab = 1024
        let arena = try QWArena(hal: hal, weights: 256, activations: 256 << 10, kv: 256, gdnStates: 256)
        let act = arena.activationsRegion()
        let selector = try QWTokenSelector(vocabLimit: vocab, allocator: act, vocabulary: vocab, maxRows: 2)
        let logits = try act.allocate(QWTensorShape([vocab], .f32))
        let journal = try act.allocate(QWTensorShape([4], .i32))
        let p = arena.cpuPointer(of: logits).assumingMemoryBound(to: Float.self)
        act.markSteadyState()

        // 1) Pure-T (topP >= 1.0 && topK >= V) uses .fullLogZ and writes draftToken + journal.
        for i in 0..<vocab { p[i] = Float(i % 13) - 6.0 }
        p[77] = 9.5
        let pureParams = QWStochasticSelectionParams(
            temperature: 0.7, topK: vocab, topP: 1.0, seed: 42, philoxOffsetBase: 10
        )
        try submit(hal) { enc in
            try selector.encodeStochasticDraft(
                enc, hal: hal, logits: logits, params: pureParams, journal: journal, journalOffset: 2
            )
        }
        let pureProp = try selector.readSparseProposal()
        XCTAssertFalse(pureProp.isSparse)
        XCTAssertFalse(pureProp.overflow)
        XCTAssertEqual(try selector.completedToken(), pureProp.token)
        XCTAssertEqual(Int(arena.read(journal, as: Int32.self, count: 4)[2]), pureProp.token)

        // 2) topK-bound (candCount >= topK && sum < topP) stays .sparse.
        for i in 0..<vocab { p[i] = (i < 16) ? 2.0 : 0.0 }
        let topKBoundParams = QWStochasticSelectionParams(
            temperature: 1.0, topK: 16, topP: 0.99, seed: 99, philoxOffsetBase: 3
        )
        try submit(hal) { enc in
            try selector.encodeStochasticDraft(enc, hal: hal, logits: logits, params: topKBoundParams)
        }
        let topKProp = try selector.readSparseProposal()
        XCTAssertTrue(topKProp.isSparse)
        XCTAssertFalse(topKProp.overflow)
        XCTAssertEqual(topKProp.ids, Array(0..<16))
        XCTAssertTrue(topKProp.ids.contains(topKProp.token))

        // 3) topP-bound + tie inclusion matching QWSampler ascending-ID tie semantics.
        for i in 0..<vocab { p[i] = -30.0 }
        p[10] = 5.0
        p[20] = 5.0
        p[30] = 5.0
        p[40] = 1.0
        let tieParams = QWStochasticSelectionParams(
            temperature: 1.0, topK: 8, topP: 0.5, seed: 1234, philoxOffsetBase: 5
        )
        try submit(hal) { enc in
            try selector.encodeStochasticDraft(enc, hal: hal, logits: logits, params: tieParams)
        }
        let tieProp = try selector.readSparseProposal()
        XCTAssertTrue(tieProp.isSparse)
        XCTAssertEqual(tieProp.ids, [10, 20])
        XCTAssertEqual(tieProp.probs[0], 0.5, accuracy: 1e-5)
        XCTAssertEqual(tieProp.probs[1], 0.5, accuracy: 1e-5)

        // 4) Forced overflow (> 512 uniform candidates) falls back losslessly to .fullLogZ.
        for i in 0..<vocab { p[i] = 1.0 }
        let overflowParams = QWStochasticSelectionParams(
            temperature: 1.0, topK: 64, topP: 0.9, seed: 777, philoxOffsetBase: 11
        )
        try submit(hal) { enc in
            try selector.encodeStochasticDraft(enc, hal: hal, logits: logits, params: overflowParams)
        }
        let ovProp = try selector.readSparseProposal()
        XCTAssertFalse(ovProp.isSparse)
        XCTAssertTrue(ovProp.overflow)
        XCTAssertEqual(ovProp.token, ovProp.gumbelToken)
        XCTAssertTrue((0..<vocab).contains(ovProp.token))
        XCTAssertEqual(act.steadyStateAllocs, 0)
    }

    func testVerifyLCResolveMatchesCPUExactnessAndOverflowSentinel() throws {
        let hal = try makeHal()
        let vocab = 1024
        let draftCount = 3
        let rows = draftCount + 1
        let arena = try QWArena(hal: hal, weights: 256, activations: 512 << 10, kv: 256, gdnStates: 256)
        let act = arena.activationsRegion()
        let draftSel = try QWTokenSelector(vocabLimit: vocab, allocator: act, vocabulary: vocab, maxRows: draftCount)
        let verifySel = try QWTokenSelector(vocabLimit: vocab, allocator: act, vocabulary: vocab, maxRows: rows)

        let draftLogits = try act.allocate(QWTensorShape([draftCount * vocab], .f32))
        let targetLogits = try act.allocate(QWTensorShape([rows * vocab], .f32))
        let draftDest = try act.allocate(QWTensorShape([draftCount], .i32))
        let dp = arena.cpuPointer(of: draftLogits).assumingMemoryBound(to: Float.self)
        let tp = arena.cpuPointer(of: targetLogits).assumingMemoryBound(to: Float.self)
        act.markSteadyState()

        for r in 0..<draftCount {
            for v in 0..<vocab { dp[r * vocab + v] = -20.0 }
            dp[r * vocab + 4] = 4.0
            dp[r * vocab + 9] = 3.5
            dp[r * vocab + 15] = 2.0
        }
        for r in 0..<rows {
            for v in 0..<vocab { tp[r * vocab + v] = -20.0 }
            tp[r * vocab + 4] = 3.0
            tp[r * vocab + 9] = 4.2
            tp[r * vocab + 21] = 2.8
        }
        let repByRow: [[Int]] = [[4], [9], [], [21]]

        for trial in 0..<12 {
            let draftParams = QWStochasticSelectionParams(
                temperature: 0.8, topK: 16, topP: 0.95, seed: UInt64(2026 + trial), philoxOffsetBase: 100
            )
            let verifyParams = QWStochasticSelectionParams(
                temperature: 0.8, topK: 16, topP: 0.95, repetitionPenalty: 1.5,
                seed: UInt64(4096 + trial), philoxOffsetBase: UInt64(500 + trial * 10)
            )
            try submit(hal) { enc in
                try draftSel.encodeStochasticDraft(
                    enc, hal: hal, logits: draftLogits, params: draftParams,
                    rows: draftCount, destination: draftDest
                )
            }
            let draftProps = try (0..<draftCount).map { try draftSel.readSparseProposal(row: $0) }
            try submit(hal) { enc in
                try verifySel.encodeStochasticVerifyLC(
                    enc, hal: hal, logits: targetLogits, draftProposals: draftSel,
                    draftCount: draftCount, params: verifyParams, repetitionHistoryByRow: repByRow
                )
            }
            let gpuRes = try verifySel.readGPUVerifyResolution(draftCount: draftCount)
            let cpuRes = try cpuVerifyLC(
                draftProps: draftProps, targetLogits: tp, vocab: vocab,
                params: verifyParams, repByRow: repByRow
            )
            XCTAssertTrue(gpuRes.resolved)
            XCTAssertEqual(gpuRes.acceptedCount, cpuRes.acceptedCount, "trial \(trial)")
            XCTAssertEqual(gpuRes.tokens, cpuRes.tokens, "trial \(trial)")
        }

        // Target overflow (> 512 uniform candidates) sets resolved = false sentinel cleanly.
        for r in 0..<rows {
            for v in 0..<vocab { tp[r * vocab + v] = 1.0 }
        }
        let ovVerifyParams = QWStochasticSelectionParams(
            temperature: 1.0, topK: 64, topP: 0.9, seed: 88, philoxOffsetBase: 200
        )
        try submit(hal) { enc in
            try verifySel.encodeStochasticVerifyLC(
                enc, hal: hal, logits: targetLogits, draftProposals: draftSel,
                draftCount: draftCount, params: ovVerifyParams
            )
        }
        let ovRes = try verifySel.readGPUVerifyResolution(draftCount: draftCount)
        XCTAssertEqual(ovRes, QWGPUVerifyResolution(resolved: false, acceptedCount: 0, tokens: []))

        // Missing sparse draft (pure-T .fullLogZ draft) also returns resolved = false sentinel.
        for r in 0..<rows {
            for v in 0..<vocab { tp[r * vocab + v] = -20.0 }
            tp[r * vocab + 4] = 3.0
        }
        let pureDraftParams = QWStochasticSelectionParams(
            temperature: 0.8, topK: vocab, topP: 1.0, seed: 7, philoxOffsetBase: 10
        )
        try submit(hal) { enc in
            try draftSel.encodeStochasticDraft(
                enc, hal: hal, logits: draftLogits, params: pureDraftParams,
                rows: draftCount, destination: draftDest
            )
        }
        try submit(hal) { enc in
            try verifySel.encodeStochasticVerifyLC(
                enc, hal: hal, logits: targetLogits, draftProposals: draftSel,
                draftCount: draftCount, params: ovVerifyParams
            )
        }
        let missingSparseRes = try verifySel.readGPUVerifyResolution(draftCount: draftCount)
        XCTAssertEqual(missingSparseRes, QWGPUVerifyResolution(resolved: false, acceptedCount: 0, tokens: []))

        // Non-finite target logit throws rather than silently falling back.
        let sparseDraftParams = QWStochasticSelectionParams(
            temperature: 0.8, topK: 16, topP: 0.95, seed: 7, philoxOffsetBase: 10
        )
        try submit(hal) { enc in
            try draftSel.encodeStochasticDraft(
                enc, hal: hal, logits: draftLogits, params: sparseDraftParams,
                rows: draftCount, destination: draftDest
            )
        }
        tp[vocab + 123] = .nan
        try submit(hal) { enc in
            try verifySel.encodeStochasticVerifyLC(
                enc, hal: hal, logits: targetLogits, draftProposals: draftSel,
                draftCount: draftCount, params: ovVerifyParams
            )
        }
        XCTAssertThrowsError(try verifySel.readGPUVerifyResolution(draftCount: draftCount))
        XCTAssertEqual(act.steadyStateAllocs, 0)
    }
}
