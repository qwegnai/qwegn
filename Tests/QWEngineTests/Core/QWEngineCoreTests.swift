import XCTest
import QWCore
import QWHead
@testable import QWEngine

/// The core owns the loop; a module owns the numerics. These tests use the
/// toy module, never a model pack.
final class QWEngineCoreTests: XCTestCase {
    private func makePlan(kvStorageMode: String = "f16") throws -> QWEnginePlan {
        try QWEnginePlan(geometry: QWModelGeometry.qwen38_27b, modelId: "toy",
                         geometryDigest: "d", prefillChunkRows: 512,
                         kvStorageMode: kvStorageMode)
    }

    private func makeCore(maxSessions: Int = 2,
                           kvStorageMode: String = "f16") throws -> (QWEngineCore, LoopToyModule) {
        let plan = try makePlan(kvStorageMode: kvStorageMode)
        let module = LoopToyModule(plan: plan)
        let core = try QWEngineCore(configuration: .init(plan: plan, module: module,
                                                          sampling: .greedy, maxSessions: maxSessions))
        return (core, module)
    }

    func testCoreGeneratesThroughTheModuleSeam() throws {
        let (core, module) = try makeCore()
        let result = try core.generate(prompt: [1, 2, 3], maxNew: 8, stopIDs: [],
                                       sampling: .greedy, onToken: nil)
        XCTAssertEqual(result.tokens.count, 8)
        // The toy emits token+1 each step, so the sequence is consecutive.
        XCTAssertEqual(result.tokens, Array(4...11))
        XCTAssertEqual(result.stepMs.count, 7)
        XCTAssertEqual(result.stepTokenCounts.count, 7)
        XCTAssertFalse(module.prefillChunks.isEmpty)
    }

    func testThePromptIsFedExactlyOnceAndThePositionsAreConsecutive() throws {
        let (core, module) = try makeCore()
        _ = try core.generate(prompt: [1, 2, 3], maxNew: 8, stopIDs: [],
                              sampling: .greedy, onToken: nil)
        // The prompt is prefilled once, and the first decode step feeds the
        // generated token at the position right after the prompt.
        XCTAssertEqual(module.prefillChunks, [0..<3])
        XCTAssertEqual(module.decoded.map(\.position), [3, 4, 5, 6, 7, 8, 9])
        XCTAssertEqual(module.decoded.map(\.token), [4, 5, 6, 7, 8, 9, 10])
    }

    func testTheFirstTokenComesFromThePrefillNotFromAnExtraDecode() throws {
        let (core, module) = try makeCore()
        let result = try core.generate(prompt: [1, 2, 3], maxNew: 4, stopIDs: [],
                                       sampling: .greedy, onToken: nil)
        // The head ran on the last prefill chunk: the first generated token
        // is the one that chunk produced, not a decode of the last prompt
        // token (which would consume the prompt twice).
        XCTAssertEqual(result.tokens.first, 4)
        XCTAssertEqual(module.decoded.count, 3)
        XCTAssertEqual(module.decoded.first?.token, 4)
    }

    func testPrefillIsChunkedByThePlan() throws {
        let (core, module) = try makeCore()
        let prompt = Array(1...1200)
        _ = try core.generate(prompt: prompt, maxNew: 2, stopIDs: [],
                              sampling: .greedy, onToken: nil)
        XCTAssertEqual(module.prefillChunks.map(\.count), [512, 512, 176])
        XCTAssertEqual(module.prefillChunks.first?.lowerBound, 0)
        XCTAssertEqual(module.prefillChunks.last?.upperBound, 1200)
    }

    func testSessionPoolExhaustionThrowsRatherThanBlocking() throws {
        let (core, _) = try makeCore(maxSessions: 1)
        let handle = try core.acquireSession()
        XCTAssertThrowsError(try core.acquireSession()) { error in
            XCTAssertEqual(error as? QWModuleError, .sessionExhausted)
        }
        core.releaseSession(handle)
        XCTAssertNoThrow(try core.acquireSession())
    }

    func testReleasedSessionIsResetBeforeReuse() throws {
        let (core, _) = try makeCore(maxSessions: 1)
        let handle = try core.acquireSession()
        let state = handle.state as? LoopToyModule.State
        XCTAssertNotNil(state)
        state?.activeTokens = 42
        core.releaseSession(handle)
        let again = try core.acquireSession()
        XCTAssertEqual((again.state as? LoopToyModule.State)?.activeTokens, 0)
        // The pool resets a session every time it hands it out.
        XCTAssertEqual((again.state as? LoopToyModule.State)?.resets, 2)
    }

    func testReleasingTwiceIsANoOp() throws {
        let (core, _) = try makeCore(maxSessions: 1)
        let handle = try core.acquireSession()
        core.releaseSession(handle)
        core.releaseSession(handle)
        // The slot is still free exactly once — a double release must not
        // put the same session in the pool twice.
        XCTAssertNoThrow(try core.acquireSession())
        XCTAssertThrowsError(try core.acquireSession()) { error in
            XCTAssertEqual(error as? QWModuleError, .sessionExhausted)
        }
    }

    func testAForeignHandleIsIgnoredByThePool() throws {
        let (core, _) = try makeCore(maxSessions: 1)
        let other = try core.acquireSession()
        core.releaseSession(other)
        let foreign = QWEngineSessionHandle(id: other.id, state: other.state)
        core.releaseSession(foreign)
        // The foreign handle did not put a second copy in the pool.
        XCTAssertNoThrow(try core.acquireSession())
        XCTAssertThrowsError(try core.acquireSession()) { error in
            XCTAssertEqual(error as? QWModuleError, .sessionExhausted)
        }
    }

    func testSessionIDsAreUnique() throws {
        let (core, _) = try makeCore(maxSessions: 3)
        var ids: Set<Int> = []
        var handles: [QWEngineSessionHandle] = []
        for _ in 0..<3 {
            let handle = try core.acquireSession()
            ids.insert(handle.id)
            handles.append(handle)
        }
        // Three live sessions have three different ids.
        XCTAssertEqual(ids.count, 3)
        for handle in handles { core.releaseSession(handle) }
    }

    func testStopIDTerminatesGeneration() throws {
        let (core, _) = try makeCore()
        // The toy emits token+1, so the second generated token is 5.
        let result = try core.generate(prompt: [3], maxNew: 64, stopIDs: [5],
                                       sampling: .greedy, onToken: nil)
        XCTAssertEqual(result.tokens, [4])
    }

    func testOnTokenCallbackCanCancel() throws {
        let (core, _) = try makeCore()
        var seen: [Int] = []
        let result = try core.generate(prompt: [1], maxNew: 16, stopIDs: [],
                                       sampling: .greedy, onToken: { token in
            seen.append(token)
            return seen.count < 3
        })
        XCTAssertEqual(result.tokens.count, 3)
        XCTAssertEqual(seen.count, 3)
    }

    func testTheRequestsSamplingContractReachesTheSession() throws {
        // The module owns the sampling, so the contract has to travel with
        // the session: a request that samples differently must not be served
        // with the previous request's contract.
        let (core, module) = try makeCore()
        let hot = QWSamplerConfig(temperature: 1.0, topK: 20, topP: 0.95, seed: 42)
        _ = try core.generate(prompt: [1], maxNew: 4, stopIDs: [], sampling: hot, onToken: nil)
        XCTAssertEqual(module.samplingContracts.last, hot)
    }

    func testTheSpeculationContractReachesTheSession() throws {
        let (core, module) = try makeCore()
        let drafts = QWSpeculativeParameters(enabled: true, mtpDrafts: 3, mtpBackend: "ngram")
        _ = try core.generate(prompt: [1], maxNew: 4, stopIDs: [], sampling: .greedy,
                              speculative: drafts, onToken: nil)
        XCTAssertEqual(module.speculativeContracts.last?.mtpDrafts, 3)
        XCTAssertEqual(module.speculativeContracts.last?.mtpBackend, "ngram")
    }

    func testAStopTokenEndsTheRequestAndIsReported() throws {
        // The stop token belongs to the caller: it ends the request and is
        // reported, and it is never appended to the generated tokens.
        let (core, _) = try makeCore()
        var stopped: [Int] = []
        let result = try core.generate(prompt: [1], maxNew: 8, stopIDs: [3],
                                       sampling: .greedy, onToken: nil,
                                       onStopToken: { stopped.append($0) })
        XCTAssertEqual(result.tokens, [2])
        XCTAssertEqual(stopped, [3])
    }

    func testTheModuleIsPreparedBeforeThePrefillIsMeasured() throws {
        // The work a request needs up front — the prefill stack, for
        // instance — must not land inside the measured prefill.
        let (core, module) = try makeCore()
        _ = try core.generate(prompt: [1, 2, 3], maxNew: 2, stopIDs: [], sampling: .greedy,
                              onToken: nil)
        XCTAssertEqual(module.prepared.last, [1, 2, 3])
    }

    func testTheModulesStepAttributionWinsOverTheCallersClock() throws {
        // A module that commits several tokens in one step splits the step's
        // time across them. The caller must report that attribution, not
        // its own clock — a busy GPU is not a free step.
        let (core, module) = try makeCore()
        module.attributedStepMs = 7.5
        let result = try core.generate(prompt: [1], maxNew: 4, stopIDs: [],
                                       sampling: .greedy, onToken: nil)
        XCTAssertEqual(result.stepMs.count, 3)
        XCTAssertTrue(result.stepMs.allSatisfy { abs($0 - 7.5) < 0.0001 },
                      "expected the module's attribution, got \(result.stepMs)")
    }

    func testTopKOfOneIsGreedyEvenWithATemperature() throws {
        // The repo's canonical rule: `temperature <= 0 || topK == 1`.
        let (core, _) = try makeCore()
        let greedy = QWSamplerConfig(temperature: 0.7, topK: 1, topP: 0.9, seed: 7)
        XCTAssertTrue(greedy.isGreedy)
        let result = try core.generate(prompt: [1, 2, 3], maxNew: 4, stopIDs: [],
                                       sampling: greedy, onToken: nil)
        XCTAssertEqual(result.tokens, [4, 5, 6, 7])
    }

    func testAnOutOfVocabularyPromptFailsBeforeAnyChunkRuns() throws {
        let (core, module) = try makeCore()
        XCTAssertThrowsError(try core.generate(prompt: [1, 2, 999_999], maxNew: 4, stopIDs: [],
                                               sampling: .greedy, onToken: nil)) { error in
            XCTAssertEqual(error as? QWModuleError, .invalidToken(999_999))
        }
        // Nothing was prefilled: the failure came before the first chunk.
        XCTAssertTrue(module.prefillChunks.isEmpty)
    }

    func testAnUnsupportedKVStorageModeIsRefused() throws {
        let plan = try makePlan(kvStorageMode: "q4")
        let module = LoopToyModule(plan: plan)
        XCTAssertThrowsError(try QWEngineCore(configuration: .init(plan: plan, module: module,
                                                                  sampling: .greedy, maxSessions: 1))) { error in
            guard case QWModuleError.unsupportedKVStorageMode = error else {
                return XCTFail("expected unsupportedKVStorageMode, got \(error)")
            }
        }
    }

    func testAModuleAndAPlanThatDisagreeAreRefused() throws {
        let plan = try makePlan()
        let other = try QWEnginePlan(geometry: QWModelGeometry.qwen38_27b, modelId: "toy",
                                     geometryDigest: "other", prefillChunkRows: 512,
                                     kvStorageMode: "f16")
        let module = LoopToyModule(plan: other)
        XCTAssertThrowsError(try QWEngineCore(configuration: .init(plan: plan, module: module,
                                                                  sampling: .greedy, maxSessions: 1))) { error in
            XCTAssertEqual(error as? QWModuleError, .planMismatch(model: "toy"))
        }
    }

    func testMaxNewZeroProducesNoTokens() throws {
        let (core, module) = try makeCore()
        let result = try core.generate(prompt: [1, 2, 3], maxNew: 0, stopIDs: [],
                                       sampling: .greedy, onToken: nil)
        XCTAssertTrue(result.tokens.isEmpty)
        XCTAssertTrue(module.decoded.isEmpty)
    }

    func testTheConfiguredSamplingIsUsedWhenTheRequestNamesNone() throws {
        let (core, _) = try makeCore()
        let result = try core.generate(prompt: [1, 2, 3], maxNew: 4, stopIDs: [],
                                       sampling: nil, onToken: nil)
        XCTAssertEqual(result.tokens, [4, 5, 6, 7])
    }

    func testEmptyPromptMatchesTheDenseEngine() throws {
        let (core, module) = try makeCore()
        // The dense engine's prefill returns 0 for an empty prompt, so the
        // first generated token is 0.
        let result = try core.generate(prompt: [], maxNew: 2, stopIDs: [],
                                       sampling: .greedy, onToken: nil)
        XCTAssertEqual(result.tokens, [0, 1])
        XCTAssertTrue(module.prefillChunks.isEmpty)
        // Two generated tokens means one decode step, at position 0.
        XCTAssertEqual(module.decoded.map(\.position), [0])
    }
}
