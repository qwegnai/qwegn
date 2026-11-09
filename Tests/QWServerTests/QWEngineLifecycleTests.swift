// QWEngineLifecycleTests.swift — The engine's phase is the truth.
// No test here reaches for a real model: the load is a seam, so the
// lifecycle is observed without a pack behind it.

import XCTest
@testable import QWServer
import Foundation

private enum LoadStop: Error {
    case stopped
}

/// Occupies the calling thread the way a real weight load does. It is
/// deliberately not async: a load that suspends would let the actor
/// answer even if the load ran on it, and the test would prove nothing.
private func occupyTheThread(for seconds: Double) {
    Thread.sleep(forTimeInterval: seconds)
}

final class QWEngineLifecycleTests: XCTestCase {
    private func makeConfig() -> QWEngineLifecycleActor.Config {
        // The user calls the model "test-model"; the pack is something
        // else. The engine must always report the pack.
        QWEngineLifecycleActor.Config(packPath: "/dummy/Qwen3.8-27B-hq4g32.qwp",
                                      modelName: "test-model",
                                      host: "127.0.0.1", port: 0)
    }

    func testLifecyclePhaseStartsUnloaded() async throws {
        let actor = QWEngineLifecycleActor(config: makeConfig(), logger: QWServerLogger(level: .none))
        let phase = actor.lifecyclePhase()
        let model = actor.loadedModelName()
        let info = actor.lifecycleInfo()
        XCTAssertEqual(phase, .unloaded)
        XCTAssertNil(model)
        XCTAssertEqual(info.phase, .unloaded)
        XCTAssertNil(info.model)
        XCTAssertNil(info.detail)
        XCTAssertNotNil(info.since)
    }

    func testTheActorAnswersWhileTheLoadIsRunning() async throws {
        // The load occupies its thread the way a real weight load does.
        // If the load ran on the actor, the actor could not answer while
        // it ran — and this test would fail rather than pass quietly.
        let actor = QWEngineLifecycleActor(
            config: makeConfig(),
            logger: QWServerLogger(level: .none),
            load: { _, _, _ in
                occupyTheThread(for: 0.3)
                throw LoadStop.stopped
            })

        let load = Task { try await actor.loadExplicit() }

        var sawLoading = false
        for _ in 0..<200 {
            if actor.lifecyclePhase() == .loading {
                sawLoading = true
                break
            }
            try await Task.sleep(nanoseconds: 5_000_000)
        }
        XCTAssertTrue(sawLoading, "the loading phase must be externally observable")

        _ = try? await load.value
        let phase = actor.lifecyclePhase()
        XCTAssertEqual(phase, .failed)
    }

    func testAFailedLoadReportsTheReason() async throws {
        struct LoadRefused: LocalizedError {
            var errorDescription: String? { "the pack was refused" }
        }
        let actor = QWEngineLifecycleActor(
            config: makeConfig(),
            logger: QWServerLogger(level: .none),
            load: { _, _, _ in throw LoadRefused() })

        do {
            _ = try await actor.loadExplicit()
            XCTFail("the load should have failed")
        } catch {
            // The failure is the load's to report.
        }

        let phase = actor.lifecyclePhase()
        let info = actor.lifecycleInfo()
        XCTAssertEqual(phase, .failed)
        XCTAssertEqual(info.phase, .failed)
        XCTAssertEqual(info.detail, "the pack was refused")
        // The pack that failed, not the name the user gave it.
        XCTAssertEqual(info.model, "Qwen3.8-27B-hq4g32")
    }

    func testThePhaseIsPublishedInTheLifecycleInfo() async throws {
        let actor = QWEngineLifecycleActor(config: makeConfig(), logger: QWServerLogger(level: .none))
        let info = actor.lifecycleInfo()
        let encoded = try JSONEncoder().encode(info)
        let decoded = try JSONDecoder().decode(QWEngineLifecycleInfo.self, from: encoded)
        XCTAssertEqual(decoded.phase, .unloaded)
        XCTAssertEqual(decoded.model, info.model)
        XCTAssertEqual(decoded.detail, info.detail)
    }

    func testTheMomentIsPublishedAsAnInstant() throws {
        let info = QWEngineLifecycleInfo(phase: .loading, since: Date(timeIntervalSince1970: 0),
                                          model: "m", detail: nil)
        let data = try JSONEncoder().encode(info)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(json["since"] as? String, "1970-01-01T00:00:00.000Z")
        let decoded = try JSONDecoder().decode(QWEngineLifecycleInfo.self, from: data)
        XCTAssertEqual(decoded.since, Date(timeIntervalSince1970: 0))
    }

    func testThePhaseKnowsWhenItBegan() async throws {
        let created = Date()
        let actor = QWEngineLifecycleActor(config: makeConfig(), logger: QWServerLogger(level: .none))
        let asked = Date()
        let since = actor.lifecycleSince()
        XCTAssertNotNil(since)
        guard let since else { return }
        // The phase began when the engine did, and it had begun by the
        // time the question was asked.
        XCTAssertGreaterThanOrEqual(since, created)
        XCTAssertLessThanOrEqual(since, asked)
    }

    func testAnUnloadWithNothingLoadedReportsNoReclaimedMemory() async throws {
        let actor = QWEngineLifecycleActor(config: makeConfig(), logger: QWServerLogger(level: .none))
        let reclaimed = try await actor.unloadExplicit()
        // Nothing was resident, so there is no number to report. A zero
        // would be a measurement of something that never happened.
        XCTAssertNil(reclaimed)
        let phase = actor.lifecyclePhase()
        XCTAssertEqual(phase, .unloaded)
    }

    func testAnUnloadDuringALoadLeavesTheEngineSettled() async throws {
        let actor = QWEngineLifecycleActor(
            config: makeConfig(),
            logger: QWServerLogger(level: .none),
            load: { _, _, _ in
                occupyTheThread(for: 0.2)
                throw LoadStop.stopped
            })

        let load = Task { try await actor.loadExplicit() }
        // Wait for the load to be under way before asking for the unload.
        for _ in 0..<200 {
            if actor.lifecyclePhase() == .loading { break }
            try await Task.sleep(nanoseconds: 5_000_000)
        }

        // The unload must return, and the engine must not be left
        // waiting on a load that has already ended.
        let reclaimed = try await actor.unloadExplicit()
        XCTAssertNil(reclaimed, "a failed load reclaimed nothing")
        let phase = actor.lifecyclePhase()
        XCTAssertNotEqual(phase, .loading, "the engine cannot stay in the loading phase")

        _ = try? await load.value
    }

    func testTheModelIsThePackAndNotTheNameItWasGiven() async throws {
        // The user may call the model anything. The engine must report
        // the pack that is really there.
        let actor = QWEngineLifecycleActor(
            config: makeConfig(),
            logger: QWServerLogger(level: .none),
            load: { _, _, _ in
                occupyTheThread(for: 0.2)
                throw LoadStop.stopped
            })
        let load = Task { try await actor.loadExplicit() }
        for _ in 0..<200 {
            if actor.lifecyclePhase() == .loading { break }
            try await Task.sleep(nanoseconds: 5_000_000)
        }

        let info = actor.lifecycleInfo()
        XCTAssertEqual(info.phase, .loading)
        XCTAssertEqual(info.model, "Qwen3.8-27B-hq4g32")
        // The weights are on their way, but nothing is resident yet.
        let resident = actor.loadedModelName()
        XCTAssertNil(resident)

        _ = try? await load.value
    }

    func testTheEngineStateIsOneSnapshot() async throws {
        let actor = QWEngineLifecycleActor(config: makeConfig(), logger: QWServerLogger(level: .none))
        let state = actor.engineState()
        XCTAssertEqual(state.lifecycle.phase, .unloaded)
        XCTAssertFalse(state.isLoaded)
        XCTAssertNil(state.loadedModelName)
        XCTAssertEqual(state.lifecycle.phase.rawValue, "unloaded")
    }
}
