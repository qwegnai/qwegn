// QWModernWebPlatformParityTests.swift — Verify ADR-014 offline /ui Web Platform standards (G6).

import Foundation
import XCTest

final class QWModernWebPlatformParityTests: XCTestCase {
    private static func readResource(_ rel: String) throws -> String {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        return try String(
            contentsOf: root.appendingPathComponent("Sources/Apps/QWServer/Resources/\(rel)"),
            encoding: .utf8
        )
    }

    func testOfflineWebPlatformStandardsDialogsTabsContainerQueriesAndA11y() throws {
        let html = try Self.readResource("controlbar.html")
        let baseCSS = try Self.readResource("css/base.css")
        let panelsCSS = try Self.readResource("css/panels.css")
        let stateJS = try Self.readResource("js/state.js")
        let actionsJS = try Self.readResource("js/actions.js")

        // Offline guarantee & 800-line ceiling across updated WebUI resources.
        for (name, content) in [
            ("controlbar.html", html),
            ("css/base.css", baseCSS),
            ("css/panels.css", panelsCSS),
            ("js/state.js", stateJS),
            ("js/actions.js", actionsJS),
        ] {
            let lineCount = content.split(separator: "\n", omittingEmptySubsequences: false).count
            XCTAssertLessThanOrEqual(lineCount, 800, "\(name) exceeded 800 lines (\(lineCount))")
            XCTAssertFalse(content.contains("https" + "://"), "\(name) must not reference external HTTPS URLs")
        }

        // ARIA tabs & modal closedby / commandfor invokers.
        for token in [
            "role=\"tablist\"",
            "role=\"tab\"",
            "aria-controls=\"view-monitor\"",
            "role=\"tabpanel\"",
            "aria-labelledby=\"nav-tab-monitor\"",
            "<dialog id=\"modal-switch-model\" closedby=\"any\"",
            "<dialog id=\"modal-memory-pack\" closedby=\"closerequest\"",
            "<dialog id=\"modal-update\" closedby=\"closerequest\"",
            "command=\"close\" commandfor=\"modal-switch-model\"",
            "command=\"close\" commandfor=\"modal-memory-pack\"",
            "command=\"close\" commandfor=\"modal-update\"",
        ] {
            XCTAssertTrue(html.contains(token), "controlbar.html missing '\(token)'")
        }

        // Shared dialog:modal scroll containment, Forced-Colors underline, and @container queries.
        for cssTok in [
            "html:has(dialog:modal)",
            "dialog:modal",
            "dialog::backdrop",
            "overscroll-behavior: contain",
            "container-type: inline-size",
            "@container",
            "forced-colors",
            "border-block-end",
        ] {
            XCTAssertTrue(baseCSS.contains(cssTok), "css/base.css missing '\(cssTok)'")
        }

        // Roving keyboard nav, busy cancel guards, Safari pointerdown/up dismiss, commandfor fallback.
        XCTAssertTrue(stateJS.contains("aria-selected"))
        for jsTok in ["ArrowRight", "ArrowLeft", "Home", "End", "pointerdown", "pointerup", "cancel", "commandfor"] {
            XCTAssertTrue(actionsJS.contains(jsTok), "js/actions.js missing '\(jsTok)'")
        }
    }
}
