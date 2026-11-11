// QWNoProfileResolutionTests.swift — Verifies hermetic `--no-profile` bypass
// across QWSettingsManager.effectiveSettings and QWServerConfigurationResolver.
import Foundation
import XCTest
import QWCore
@testable import QWServer

final class QWNoProfileResolutionTests: XCTestCase {
    func testEffectiveSettingsIgnoresPersistedProfileWhenRequested() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let settingsURL = dir.appendingPathComponent("settings.json")
        var saved = QWServerSettings()
        saved.modelName = "TestModel-SHVQ3"
        saved.kvMode = "f16"
        saved.lsaMode = "off"
        saved.mtpDrafts = 4
        try QWSettingsManager.shared.saveSettings(saved, to: settingsURL)

        let profiled = QWSettingsManager.shared.effectiveSettings(
            forModel: "TestModel-SHVQ3",
            packPath: "",
            base: saved,
            from: settingsURL,
            ignoreProfile: false
        )
        XCTAssertNotNil(profiled.profile)
        XCTAssertEqual(profiled.settings.kvMode, "f16")
        XCTAssertEqual(profiled.settings.lsaMode, "off")

        let hermetic = QWSettingsManager.shared.effectiveSettings(
            forModel: "TestModel-SHVQ3",
            packPath: "",
            base: QWServerSettings(),
            from: settingsURL,
            ignoreProfile: true
        )
        XCTAssertNil(hermetic.profile)
        XCTAssertEqual(hermetic.settings.kvMode, "q4")
        XCTAssertEqual(hermetic.settings.lsaMode, "auto")
    }

    func testServerResolverBypassSetsExplicitNoProfileSource() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let settingsURL = dir.appendingPathComponent("settings.json")
        var saved = QWServerSettings()
        saved.modelName = "BenchModel"
        saved.kvMode = "f16"
        saved.lsaMode = "qsa-cluster"
        try QWSettingsManager.shared.saveSettings(saved, to: settingsURL)

        let resolved = try QWServerConfigurationResolver.resolve(
            cliPack: "/tmp/nonexistent-bench-pack",
            cliModelName: "BenchModel",
            cliIgnoreProfile: true,
            environment: [:]
        )
        XCTAssertEqual(resolved.sourceSummary["noProfile"], "cli (--no-profile)")
        XCTAssertEqual(QWKnobSource.normalise(resolved.sourceSummary["noProfile"] ?? ""), .explicit)
        XCTAssertEqual(resolved.kvMode, "q4")
        XCTAssertEqual(resolved.lsaMode, "auto")
    }
}
