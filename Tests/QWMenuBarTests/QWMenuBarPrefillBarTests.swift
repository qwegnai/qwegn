// QWMenuBarPrefillBarTests.swift — Tests & 2x Retina snapshots for invariant MenuBar prefill bar layout.

import AppKit
import Foundation
import QWCore
@testable import QWMenuBarUI
import QWServer
import SwiftUI
import XCTest

final class QWMenuBarPrefillBarTests: XCTestCase {
    @MainActor
    func testCupertinoTelemetryViewHeightIsInvariantAcrossIdlePrefillAndDecode() throws {
        _ = NSApplication.shared
        let telemetry = TelemetryObservable(
            startServices: false,
            enableAutoLifecycle: false,
            initialSettings: QWServerSettings()
        )
        telemetry.connection = .connected
        telemetry.engineState = .serverUp(isLoaded: true)
        telemetry.isDaemonTransitioning = false

        let now = Date()
        let modelName = "Qwen3.8-27B-Instruct-hq4g32"
        func makeDashboard(phase: QWEngineTelemetryRegistry.EngineInferencePhase) -> QWDashboardViewModel {
            let prefillProg: String? = phase == .prefill ? "26,337" : nil
            let prefillProgNum: Double? = phase == .prefill ? 26_337 : nil
            let prefillTotal: String? = phase == .prefill ? "62,707" : nil
            let prefillTotalNum: Double? = phase == .prefill ? 62_707 : nil
            let prefillTitle = phase == .prefill ? "Prefill (live)" : "Prefill (last)"
            let decodeTitle = phase == .decode ? "Decode (live)" : "Decode (last)"
            let prefillRate = phase == .prefill ? "1824.0" : "1780.0"
            let prefillRateNum = phase == .prefill ? 1824.0 : 1780.0
            let decodeRate = phase == .decode ? "84.2" : "81.6"
            let decodeRateNum = phase == .decode ? 84.2 : 81.6
            func card(
                _ id: String, _ title: String, _ val: String?, _ unit: String?, _ num: Double?,
                series: [Double?] = [], reason: String? = nil
            ) -> QWDashboardCard {
                QWDashboardCard(
                    id: id, title: title, value: val, unit: unit, number: num,
                    provenance: val == nil ? .unavailable : .measured,
                    sampledAt: val == nil ? nil : now, age: val == nil ? nil : 0.1,
                    isStale: false, reason: reason, series: series
                )
            }
            return QWDashboardViewModel(
                sampledAt: now,
                lifecycle: .init(phase: .loaded, since: now, model: modelName, detail: nil),
                modelId: modelName,
                capabilities: ["prefixCache"],
                generation: phase,
                facts: QWDashboardFacts(
                    quantization: "HQ4_G32 (native, GPU)",
                    hostLogicalCores: 18,
                    gdnPrefillMode: "chunked",
                    gpuName: "Apple M5 Max",
                    metalFamily: "Metal 4",
                    physicalMemoryGB: 128,
                    lsaMode: "auto"
                ),
                cards: [
                    card("throughput.display_decode_tokens_per_sec", decodeTitle, decodeRate, "tok/s", decodeRateNum, series: [80.1, 81.6, 84.2]),
                    card("throughput.display_prefill_tokens_per_sec", prefillTitle, prefillRate, "tok/s", prefillRateNum, series: [1780.0, nil, 1824.0]),
                    card("prefill.progress_tokens", "Prefill progress", prefillProg, "tok", prefillProgNum, reason: "no prompt prefill is currently in flight"),
                    card("prefill.total_tokens", "Prefill target", prefillTotal, "tok", prefillTotalNum, reason: "no prompt prefill is currently in flight"),
                    card("latency.ttft_ms", "Time to first token", "142", "ms", 142),
                    card("context.active_tokens", "Last turn tokens", "4096", "tok", 4096),
                    card("gpu.busy_ratio", "Engine GPU", "94.1", "%", 94.1),
                    card("gpu.device", "Device GPU", "95.0", "%", 95.0),
                    card("hardware.cpu_percent", "CPU", "18.5", "%", 18.5),
                    card("hardware.memory_percent", "Memory", "64.2", "%", 64.2),
                    card("hardware.memory_pressure_level", "Memory pressure", "Normal", nil, nil),
                    card("hardware.gpu_temp_celsius", "GPU Temp", "68.4", "°C", 68.4),
                    card("hardware.fan_rpm", "Fans", "2450", "rpm", 2450),
                    card("hardware.system_power_watts", "System Power", "78.2", "W", 78.2),
                    card("hardware.thermal_state", "Thermal State", "Nominal", nil, nil),
                    card("memory.arena_reserved_bytes", "Metal arena", "24.10", "GiB", 24.10),
                    card("memory.resident_bytes", "Resident weights", "22.50", "GiB", 22.50),
                    card("cache.warm_bytes", "Warm cache", "1.20", "GiB", 1.20),
                    card("cache.ssd_bytes", "SSD cache", "0.00", "GiB", 0.0),
                    card("cache.tokens_reused_total", "Tokens reused", "1024", "tok", 1024),
                ],
                actions: [],
                knobs: ["kvMode": .string("q4"), "contextTokens": .int(131_072)],
                knobSources: [:]
            )
        }

        func measureSize(for phase: QWEngineTelemetryRegistry.EngineInferencePhase, snapshotName: String?) throws -> NSSize {
            telemetry.dashboard = makeDashboard(phase: phase)
            let hosting = NSHostingView(
                rootView: CupertinoTelemetryView(telemetry: telemetry).background(TelemetryTheme.surface)
            )
            hosting.appearance = NSAppearance(named: .darkAqua)
            hosting.wantsLayer = true
            let fit = hosting.fittingSize
            hosting.frame = NSRect(origin: .zero, size: fit)
            hosting.layoutSubtreeIfNeeded()

            if let snapshotName, let snapDir = ProcessInfo.processInfo.environment["QWG_SNAPSHOT_DIR"], !snapDir.isEmpty {
                let dirURL = URL(fileURLWithPath: snapDir, isDirectory: true)
                try FileManager.default.createDirectory(at: dirURL, withIntermediateDirectories: true)
                let window = NSWindow(
                    contentRect: NSRect(origin: .zero, size: fit),
                    styleMask: [.borderless],
                    backing: .buffered,
                    defer: false
                )
                window.appearance = NSAppearance(named: .darkAqua)
                window.isReleasedWhenClosed = false
                window.contentView = hosting
                window.layoutIfNeeded()
                hosting.layoutSubtreeIfNeeded()
                try write2xPNG(of: hosting, size: fit, to: dirURL.appendingPathComponent(snapshotName))
                window.close()
            }
            return fit
        }

        let idleSize = try measureSize(for: .idle, snapshotName: "menubar_prefill_idle_2x.png")
        let prefillSize = try measureSize(for: .prefill, snapshotName: "menubar_prefill_active_2x.png")
        let decodeSize = try measureSize(for: .decode, snapshotName: nil)

        XCTAssertEqual(idleSize.width, 440, accuracy: 0.5)
        XCTAssertEqual(prefillSize.width, 440, accuracy: 0.5)
        XCTAssertEqual(decodeSize.width, 440, accuracy: 0.5)
        XCTAssertEqual(idleSize.height, prefillSize.height, accuracy: 0.5)
        XCTAssertEqual(decodeSize.height, prefillSize.height, accuracy: 0.5)
    }

    @MainActor
    private func write2xPNG(of view: NSView, size: NSSize, to url: URL) throws {
        guard let rep = NSBitmapImageRep(
            bitmapDataPlanes: nil,
            pixelsWide: Int((size.width * 2).rounded()),
            pixelsHigh: Int((size.height * 2).rounded()),
            bitsPerSample: 8,
            samplesPerPixel: 4,
            hasAlpha: true,
            isPlanar: false,
            colorSpaceName: .deviceRGB,
            bytesPerRow: 0,
            bitsPerPixel: 0
        ) else { return }
        rep.size = size
        view.cacheDisplay(in: view.bounds, to: rep)
        if let png = rep.representation(using: .png, properties: [:]) {
            try png.write(to: url)
        }
    }
}
