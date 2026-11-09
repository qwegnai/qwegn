// QWSettingsCoherenceTests.swift — The settings and the server must
// agree after a change. The rule lives in QWSettingsCoherence and
// QWServerLifecycleRules; the app must use that rule and not its own.

import XCTest
@testable import QWServer
import QWCore
import Foundation

final class QWSettingsCoherenceTests: XCTestCase {
    private let menuBarDirectory = "Sources/Apps/QWMenuBar"
    private let serverDirectory = "Sources/Apps/QWServer"

    // MARK: - 1. The setting the app never showed

    func testKeepServerRunningAfterQuitIsASettingWithAnEffect() {
        let defaults = QWServerSettings()
        XCTAssertFalse(defaults.keepServerRunningAfterQuit)

        let note = QWSettingsCoherence.effectNote(for: "keepServerRunningAfterQuit")
        XCTAssertNotNil(note, "the setting must say what it does")
        XCTAssertTrue(note!.contains("server restart"),
                      "the note must say when the change takes effect")
    }

    func testTheSettingsWindowShowsKeepServerRunningAfterQuit() throws {
        let source = try MenuBarRendererParityTests.read("\(menuBarDirectory)/SettingsView.swift")
        XCTAssertTrue(source.contains("keepServerRunningAfterQuit"),
                      "the settings window must offer the setting")
        XCTAssertTrue(source.contains("Keep the server running after the app quits"),
                      "the setting must be stated in words the user reads")
    }

    func testKeepServerRunningAfterQuitIsPersisted() throws {
        let tmpDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }
        let fileURL = tmpDir.appendingPathComponent("settings.json")

        var settings = QWServerSettings()
        settings.keepServerRunningAfterQuit = true
        try QWSettingsManager().saveSettings(settings, to: fileURL)

        let reloaded = QWSettingsManager().loadSettings(from: fileURL)
        XCTAssertTrue(reloaded.keepServerRunningAfterQuit)
    }

    // MARK: - 2. Restart means one thing

    func testRestartIsTheInProcessEngineRestart() throws {
        let source = try MenuBarRendererParityTests.read("\(menuBarDirectory)/TelemetryObservable.swift")
        let body = try XCTUnwrap(MenuBarRendererParityTests.body(of: "restartEngine", in: source))
        XCTAssertTrue(body.contains("/v1/engine/restart"),
                      "restart must be the engine restart inside the running server")
        XCTAssertFalse(body.contains("stopEngineAsync"),
                       "restart must not stop the server process")
    }

    func testTheProcessRestartHasItsOwnName() throws {
        let source = try MenuBarRendererParityTests.read("\(menuBarDirectory)/TelemetryObservable.swift")
        let body = try XCTUnwrap(MenuBarRendererParityTests.body(of: "restartServerProcess", in: source))
        XCTAssertTrue(body.contains("stopEngineAsync"))
        XCTAssertTrue(body.contains("startEngine"))
    }

    func testTheSurfacesUseTheOneRestart() throws {
        let cards = try MenuBarRendererParityTests.read("\(menuBarDirectory)/TelemetryCards.swift")
        XCTAssertTrue(cards.contains("telemetry.restartEngine()"))
        XCTAssertFalse(cards.contains("telemetry.restartServerProcess()"),
                       "the card button must not restart the process")
        let app = try MenuBarRendererParityTests.read("\(menuBarDirectory)/QWMenuBarApp.swift")
        XCTAssertTrue(app.contains("telemetry.restartEngine()"))
    }

    func testTheServerRouteIsTheEngineRestart() throws {
        let source = try MenuBarRendererParityTests.read("\(serverDirectory)/Core/QWHttpServer.swift")
        XCTAssertTrue(source.contains("\"/v1/engine/restart\""))
        XCTAssertTrue(source.contains("restartExplicit()"))
    }

    // MARK: - 3. The settings that need a server restart

    func testEverySettingSaysWhenItTakesEffect() throws {
        let data = try JSONEncoder().encode(QWServerSettings())
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let classified = QWSettingsCoherence.serverRestartKeys.union(QWSettingsCoherence.liveKeys)
        XCTAssertEqual(Set(object.keys), classified,
                       "every setting must be classified; an unclassified one is a silent lie")
    }

    func testTheSettingsTheServerReadsAtStartupNeedARestart() {
        let atStartup = ["host", "port", "modelPackPath", "modelName", "maxContext", "kvMode",
                         "ropeScalingType", "ropeScalingFactor", "temperature", "topP", "topK",
                         "maxTokens", "repetitionPenalty", "seed", "enableThinking",
                         "reasoningEffort", "systemPrompt", "lazyLoad",
                         "speculativeDecodingEnabled", "mtpDrafts", "mtpBackend",
                         "mtpAdaptiveDepth", "mtpEntropyThreshold", "dflashDraftPackPath",
                         "dflashSelectorTopK", "gdnPrefill", "evictPrefillScratch",
                         "prefillChunkSize", "warmRAMLimitGB", "ssdCacheLimitGB", "cacheDir",
                         "logLevel"]
        for key in atStartup {
            XCTAssertTrue(QWSettingsCoherence.needsServerRestart(key),
                          "\(key) is read when the server starts")
        }
    }

    func testTheSettingsReadOnDemandDoNotNeedARestart() {
        for key in ["modelAlias", "modelAliases", "loadVision", "logDir", "logFileName",
                    "telemetryPollIntervalMs", "runAtStartup", "autoStartServer"] {
            XCTAssertFalse(QWSettingsCoherence.needsServerRestart(key),
                           "\(key) is read when it is needed")
        }
    }

    func testTheSettingsWindowOffersLoadVisionInPlaceWithoutRestart() throws {
        let source = try MenuBarRendererParityTests.read("\(menuBarDirectory)/SettingsView.swift")
        XCTAssertTrue(source.contains("draftSettings.loadVision"),
                      "the Model tab must offer the Load Vision Tower toggle")
        XCTAssertTrue(source.contains("Load Vision Tower (Multimodal Image Input)"),
                      "the toggle must be stated clearly in the Model tab")
        let note = QWSettingsCoherence.effectNote(for: "loadVision")
        XCTAssertNotNil(note)
        XCTAssertTrue(note!.contains("without restarting the engine"),
                      "loadVision takes effect in place without resetting the KV cache")
    }

    func testAChangedSettingThatNeedsARestartIsPending() {
        let before = QWServerSettings()
        var after = before
        after.port = 9090
        after.telemetryPollIntervalMs = 1000

        XCTAssertEqual(QWSettingsCoherence.changedKeys(from: before, to: after),
                       ["port", "telemetryPollIntervalMs"])
        XCTAssertEqual(QWSettingsCoherence.pendingServerRestartKeys(from: before, to: after),
                       ["port"])
    }

    func testTheSettingsWindowMarksAndOffersTheRestart() throws {
        let source = try ["SettingsView.swift", "SettingsView+Actions.swift"]
            .map { try MenuBarRendererParityTests.read("\(menuBarDirectory)/\($0)") }
            .joined(separator: "\n")
        XCTAssertTrue(source.contains("SettingsRestartBadge"),
                      "the settings that wait for a restart must be marked")
        XCTAssertTrue(source.contains("Restart Server"),
                      "the restart must be offered where the change waits")
        XCTAssertTrue(source.contains("restartServerProcess()"))
    }

    // MARK: - 4. The parent-pid watchdog

    func testTheWatchdogIsArmedOnlyWhenTheServerShouldDieWithTheApp() {
        XCTAssertTrue(QWServerLifecycleRules.armsParentPidWatchdog(keepServerRunningAfterQuit: false))
        XCTAssertFalse(QWServerLifecycleRules.armsParentPidWatchdog(keepServerRunningAfterQuit: true))
        XCTAssertEqual(QWServerLifecycleRules.parentPidArguments(keepServerRunningAfterQuit: true,
                                                                 parentPid: 42), [])
        XCTAssertEqual(QWServerLifecycleRules.parentPidArguments(keepServerRunningAfterQuit: false,
                                                                 parentPid: 42),
                       ["--parent-pid", "42"])
    }

    func testTheServerKeepsRunningOnlyWhenNothingWouldKillIt() {
        XCTAssertEqual(QWServerLifecycleRules.quitDecision(keepServerRunningAfterQuit: true,
                                                           parentWatchdogArmed: false),
                       .leaveServerRunning)
        XCTAssertEqual(QWServerLifecycleRules.quitDecision(keepServerRunningAfterQuit: true,
                                                           parentWatchdogArmed: true),
                       .stopServer)
        XCTAssertEqual(QWServerLifecycleRules.quitDecision(keepServerRunningAfterQuit: false,
                                                           parentWatchdogArmed: false),
                       .stopServer)
    }

    func testTheAppArmsTheWatchdogFromTheSetting() throws {
        let source = try MenuBarRendererParityTests.read("\(menuBarDirectory)/TelemetryObservable.swift")
        XCTAssertTrue(source.contains("QWServerLifecycleRules.parentPidArguments"))
        XCTAssertFalse(source.contains("\"--parent-pid\""),
                       "the app must not hard-code the watchdog argument")
    }

    func testTheAppQuitsByTheSharedRule() throws {
        let source = try MenuBarRendererParityTests.read("\(menuBarDirectory)/QWMenuBarApp.swift")
        XCTAssertTrue(source.contains("QWServerLifecycleRules.quitDecision"))
        XCTAssertFalse(source.contains("telemetry.settings.keepServerRunningAfterQuit ||"),
                       "the quit path must not decide on its own")
    }

    func testALiveServerIsNotReportedAsGone() {
        XCTAssertEqual(QWServerLifecycleRules.liveness(answered: true, processAlive: false), .serving)
        XCTAssertEqual(QWServerLifecycleRules.liveness(answered: false, processAlive: true), .unreachable)
        XCTAssertEqual(QWServerLifecycleRules.liveness(answered: false, processAlive: false), .gone)
    }

    func testTheAppAsksTheServerBeforeCallingItGone() throws {
        let source = try MenuBarRendererParityTests.read("\(menuBarDirectory)/TelemetryObservable.swift")
        let body = try XCTUnwrap(MenuBarRendererParityTests.body(of: "applyUnreachableState", in: source))
        XCTAssertTrue(body.contains("QWServerLifecycleRules.liveness"))
        XCTAssertTrue(body.contains("serverProcessIsAlive"))
    }

    // MARK: - 5. autoStartServer is read again after a change

    func testTheStartDecisionFollowsTheSettingAsItIsNow() {
        XCTAssertEqual(QWSettingsCoherence.startDecision(autoStartServer: true, serverRunning: false),
                       .start)
        XCTAssertEqual(QWSettingsCoherence.startDecision(autoStartServer: false, serverRunning: false),
                       .stayStopped)
        XCTAssertEqual(QWSettingsCoherence.startDecision(autoStartServer: true, serverRunning: true),
                       .alreadyRunning)
    }

    func testTheAppRereadsTheSettingsAfterASave() throws {
        let source = try MenuBarRendererParityTests.read("\(menuBarDirectory)/TelemetryObservable.swift")
        let body = try XCTUnwrap(MenuBarRendererParityTests.body(of: "saveSettings", in: source))
        XCTAssertTrue(body.contains("loadSettings()"),
                      "the settings are read back from the file, not from a launch-time copy")
        XCTAssertTrue(body.contains("applyAutoStartDecision()"))
    }

    func testTheManagerSeesAChangeWrittenByAnotherProcess() throws {
        let tmpDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }
        let fileURL = tmpDir.appendingPathComponent("settings.json")

        let manager = QWSettingsManager()
        var settings = manager.loadSettings(from: fileURL)
        XCTAssertEqual(settings.port, 8080)

        settings.port = 9090
        settings.modelName = "Another-Model"
        let data = try JSONEncoder().encode(settings)
        try data.write(to: fileURL, options: .atomic)

        let reloaded = manager.loadSettings(from: fileURL)
        XCTAssertEqual(reloaded.port, 9090, "a change on disk must not be masked by the cache")
        XCTAssertEqual(reloaded.modelName, "Another-Model")
    }

    // MARK: - 6. Context slider (64 to 1M) and model memory prediction

    func testContextSliderStepsSpan64To1MWithoutNativeOrYaRNLabels() {
        let steps = QWSettingsCoherence.contextSliderSteps
        XCTAssertEqual(steps.first, 64)
        XCTAssertEqual(steps.last, 1_048_576)
        XCTAssertEqual(steps.count, 15)
        for (idx, tok) in steps.enumerated() {
            XCTAssertEqual(QWSettingsCoherence.nearestContextStepIndex(for: tok), idx)
            XCTAssertEqual(QWSettingsCoherence.contextTokens(forStepIndex: idx), tok)
            let label = QWSettingsCoherence.formatContextTokens(tok)
            XCTAssertFalse(label.localizedCaseInsensitiveContains("native"))
            XCTAssertFalse(label.localizedCaseInsensitiveContains("yarn"))
        }
        XCTAssertEqual(QWSettingsCoherence.formatContextTokens(64), "64 tokens")
        XCTAssertEqual(QWSettingsCoherence.formatContextTokens(262_144), "256k (262,144 tokens)")
        XCTAssertEqual(QWSettingsCoherence.formatContextTokens(524_288), "512k (524,288 tokens)")
        XCTAssertEqual(QWSettingsCoherence.formatContextTokens(1_048_576), "1M (1,048,576 tokens)")
    }

    func testSettingsWindowUsesContextLimitSliderAndAlertsWhenExceedingMemory() throws {
        let settingsSource = try MenuBarRendererParityTests.read("\(menuBarDirectory)/SettingsView.swift")
        let coherenceViews = try MenuBarRendererParityTests.read("\(menuBarDirectory)/SettingsCoherenceViews.swift")
        for label in ["256k native", "512k YaRN", "1M YaRN"] {
            XCTAssertFalse(settingsSource.contains(label), "SettingsView must not show '\(label)'")
            XCTAssertFalse(coherenceViews.contains(label), "SettingsCoherenceViews must not show '\(label)'")
        }
        XCTAssertTrue(settingsSource.contains("ContextLimitSliderRow("))
        XCTAssertTrue(coherenceViews.contains("Context Limit Exceeds Available Memory"))
        XCTAssertTrue(coherenceViews.contains("Predicted Memory:"))
    }

    func testEstimateContextMemoryPredictsModelAndKVDemandsAndFlagsOverBudget() throws {
        let desc = QWModelDescriptor.qwen38_27b
        var settings = QWServerSettings()
        settings.kvMode = "f16"
        settings.speculativeDecodingEnabled = false

        settings.maxContext = 4_096
        let shortEst = try XCTUnwrap(settings.estimateContextMemory(
            for: desc, physicalMemoryBytes: 64 * (1 << 30)))
        XCTAssertTrue(shortEst.fits)
        XCTAssertGreaterThan(shortEst.weightsBytes, 14 * (1 << 30))

        settings.maxContext = 1_048_576
        let longEst = try XCTUnwrap(settings.estimateContextMemory(
            for: desc, physicalMemoryBytes: 64 * (1 << 30)))
        XCTAssertFalse(longEst.fits, "1M FP16 KV on a 27B model exceeds a 64 GiB RAM budget")
        XCTAssertGreaterThan(longEst.kvBytes, shortEst.kvBytes)
        XCTAssertGreaterThan(longEst.requiredBytes, longEst.availableBytes)

        settings.kvMode = "q4"
        let q4Est = try XCTUnwrap(settings.estimateContextMemory(
            for: desc, physicalMemoryBytes: 64 * (1 << 30)))
        XCTAssertLessThan(q4Est.kvBytes, longEst.kvBytes)
    }
}
