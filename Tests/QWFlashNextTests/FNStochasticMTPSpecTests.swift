// FNStochasticMTPSpecTests.swift — Unit and Metal wiring tests for Flash-Next stochastic MTP & loop-aware penalty.

import Foundation
import Metal
import XCTest
import QWAlloc
import QWCore
@testable import QWEngine
import QWHAL
import QWHead
import QWMTP

final class FNStochasticMTPSpecTests: XCTestCase {
    func testSessionUsesAdaptiveDraftPolicyAndHonorsLaw11TimingGuard() throws {
        var greedySession = FNGPUSessionData(
            history: [1, 2, 3],
            sampler: QWSampler(config: .greedy),
            useSpec: true
        )
        var cleanSession = FNGPUSessionData(
            history: [1, 2, 3],
            sampler: QWSampler(config: .greedy),
            useSpec: true
        )
        XCTAssertEqual(greedySession.chainPolicy.mode, .autoregressiveChain)

        for round in 0..<8 {
            let p1 = greedySession.chainPolicy.selectChainPasses(maxPasses: 2)
            let p2 = cleanSession.chainPolicy.selectChainPasses(maxPasses: 2)
            XCTAssertEqual(p1, p2)
            let noisyMs = (round % 2 == 0) ? 950.0 : 0.5
            greedySession.chainPolicy.observeVerifyRound(
                rows: 2 + p1, ms: noisyMs, isGreedy: true, hasExplicitSeed: false
            )
            greedySession.chainPolicy.observeChain(
                passes: max(1, p1), ms: noisyMs, isGreedy: true, hasExplicitSeed: false
            )
            greedySession.chainPolicy.observeAcceptance(draftCount: 1 + p1, accepted: 1)
            cleanSession.chainPolicy.observeAcceptance(draftCount: 1 + p2, accepted: 1)
        }
        XCTAssertEqual(
            greedySession.chainPolicy.selectChainPasses(maxPasses: 2),
            cleanSession.chainPolicy.selectChainPasses(maxPasses: 2)
        )
        XCTAssertEqual(
            greedySession.chainPolicy.expectedTokens(drafts: 3),
            cleanSession.chainPolicy.expectedTokens(drafts: 3),
            accuracy: 1e-12
        )

        var seededSession = FNGPUSessionData(
            history: [1, 2, 3],
            sampler: QWSampler(config: QWSamplerConfig(
                temperature: 0.8, topK: 32, topP: 0.9, seed: 42,
                uniformVersion: .upper24, hasExplicitSeed: true
            )),
            useSpec: true
        )
        for _ in 0..<6 {
            let p = seededSession.chainPolicy.selectChainPasses(maxPasses: 2)
            seededSession.chainPolicy.observeVerifyRound(
                rows: 2 + p, ms: 900.0, isGreedy: false, hasExplicitSeed: true
            )
            seededSession.chainPolicy.observeChain(
                passes: max(1, p), ms: 900.0, isGreedy: false, hasExplicitSeed: true
            )
        }
        let state1 = try seededSession.ensureMTPStreamState()
        XCTAssertEqual(state1.seed, 42)
        XCTAssertEqual(state1.drawsConsumed, 0)
        let state2 = try seededSession.ensureMTPStreamState()
        XCTAssertEqual(state1, state2)
    }

    func testPrecomputeLoopRepetitionByRowNeverUsesRawUnloopedDrafts() {
        let vocab = 1024
        // Non-looping recent + non-looping drafts must produce empty loop repetition sets on every row.
        let cleanRecent = [10, 11, 12, 13, 14]
        let cleanDrafts = [15, 16, 17]
        let cleanRep = FNGPUSessionData.precomputeLoopRepetitionByRow(
            recentGenerated: cleanRecent,
            seenTokenList: QWSampler.loopRepetitionTokens(in: cleanRecent),
            drafts: cleanDrafts,
            vocab: vocab
        )
        XCTAssertEqual(cleanRep, [[], [], [], []])

        // 3 trailing identical tokens + draft[0] == 77 triggers a 4x+ period-1 loop from row 1 onward.
        let burstRecent = [1, 2, 77, 77, 77]
        let burstDrafts = [77, 77, 77]
        let burstRep = FNGPUSessionData.precomputeLoopRepetitionByRow(
            recentGenerated: burstRecent,
            seenTokenList: QWSampler.loopRepetitionTokens(in: burstRecent),
            drafts: burstDrafts,
            vocab: vocab
        )
        XCTAssertEqual(burstRep.count, 4)
        XCTAssertEqual(burstRep[0], [])
        for r in 1...3 {
            let expectedWindow = Array((burstRecent + burstDrafts.prefix(r)).suffix(FNGPUSessionData.repetitionPenaltyWindow))
            XCTAssertEqual(burstRep[r], QWSampler.loopRepetitionTokens(in: expectedWindow))
            XCTAssertTrue(burstRep[r].contains(77))
            XCTAssertFalse(burstRep[r].contains(88))
        }
    }

    func testFNGPUHeadSelectorWiringAndVerifyCopy() throws {
        try XCTSkipIf(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] == "true"
                || MTLCreateSystemDefaultDevice() == nil,
            "requires Apple Silicon Metal device"
        )
        let hal = try QWHal(geometry: .qwen38_27b)
        let arena = try QWArena(hal: hal, weights: 1 << 20, activations: 1 << 20, kv: 64 << 10, gdnStates: 0)
        let wBuf = try XCTUnwrap(hal.device.makeBuffer(length: 65_536, options: .storageModeShared))
        func tensor(_ buf: any MTLBuffer, count: Int, dtype: QWDType = .f32) -> QWTensorValue {
            QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: buf.length),
                          offset: 0, shape: QWTensorShape([count], dtype))
        }
        let vocab = 64
        let hqWeight = try XCTUnwrap(QWWeightView(
            blocks: tensor(wBuf, count: 8192, dtype: .f16),
            n: vocab, k: 32, bits: 4, groupSize: 32, isRotated: true
        ))
        let mixerW = try XCTUnwrap(QWWeightView(
            blocks: tensor(wBuf, count: 1024, dtype: .f16),
            n: 32, k: 32, bits: 4, groupSize: 32, isRotated: true
        ))
        let normW = tensor(wBuf, count: 32)
        let head = try FNGPUHead(
            arena: arena, hal: hal, head: hqWeight, mixerDown: mixerW, mixerUp: mixerW,
            normWeight: normW, eps: 1e-6, hidden: 32, hc: 1, lowrank: 32, maxRows: 4
        )
        XCTAssertEqual(head.selector.maxRows, 4)
        XCTAssertEqual(head.selector.vocabulary, vocab)

        let draftSelector = try QWTokenSelector(
            vocabLimit: vocab, allocator: arena.activationsRegion(), vocabulary: vocab, maxRows: 2
        )
        let draftLogits = try arena.activationsRegion().allocate(QWTensorShape([vocab], .f32))
        let dp = arena.cpuPointer(of: draftLogits).assumingMemoryBound(to: Float.self)
        let tp = arena.cpuPointer(of: head.logits).assumingMemoryBound(to: Float.self)
        for v in 0..<vocab {
            dp[v] = -20.0
            tp[v] = -20.0
            tp[vocab + v] = -20.0
        }
        dp[5] = 4.0; dp[9] = 3.0
        tp[5] = 4.0; tp[9] = 3.0
        tp[vocab + 12] = 5.0

        let draftParams = QWStochasticSelectionParams(
            temperature: 0.7, topK: 8, topP: 0.95, seed: 123, philoxOffsetBase: 0
        )
        let verifyParams = QWStochasticSelectionParams(
            temperature: 0.7, topK: 8, topP: 0.95, seed: 456, philoxOffsetBase: 100
        )
        guard let pair = hal.begin(.concurrent) else { return XCTFail("failed to begin CB") }
        let cb = pair.0, enc = pair.1
        var encoderClosed = false
        defer {
            if !encoderClosed { enc.endEncoding() }
        }
        try draftSelector.encodeStochasticDraft(enc, hal: hal, logits: draftLogits, params: draftParams)
        enc.memoryBarrier(scope: .buffers)
        try head.selector.encodeStochasticVerifyLC(
            enc, hal: hal, logits: head.logits.slice(byteOffset: 0, count: 2 * vocab, dtype: .f32),
            draftProposals: draftSelector,
            draftCount: 1, params: verifyParams
        )
        enc.memoryBarrier(scope: .buffers)
        let copy = try hal.pipeline("qwg_verify_copy")
        enc.setComputePipelineState(copy)
        head.selector.verifyResolvedTokens.slice(byteOffset: 0, count: 2, dtype: .i32).bind(enc, index: 0)
        head.token.slice(byteOffset: 0, count: 2, dtype: .i32).bind(enc, index: 1)
        var two: UInt32 = 2
        enc.setBytes(&two, length: 4, index: 2)
        enc.dispatchThreads(MTLSize(width: 2, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 2, height: 1, depth: 1))
        encoderClosed = true
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)

        let res = try head.selector.readGPUVerifyResolution(draftCount: 1)
        XCTAssertTrue(res.resolved)
        XCTAssertEqual(head.argmaxTokens(count: res.tokens.count), res.tokens)
        XCTAssertEqual(head.readLogits(row: 1)[12], 5.0, accuracy: 1e-6)
    }
}
