import SwiftUI
import AppKit
import Combine
import QWCore
import QWServer

public struct QWMenuBarApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    public init() {}

    public var body: some Scene {
        Settings {
            EmptyView()
        }
    }
}

@MainActor
public final class AppDelegate: NSObject, NSApplicationDelegate {
    private var statusItem: NSStatusItem!
    private var popover: NSPopover!
    private let telemetry = TelemetryObservable()
    private var cancellables = Set<AnyCancellable>()

    private var settingsWindow: NSWindow?
    private var logsWindow: NSWindow?
    private var isTerminating = false

    public func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        if isTerminating {
            return .terminateNow
        }

        // The server outlives the app only when the setting says so and
        // no watchdog would kill it; a server that would die with the app
        // anyway is stopped on the way out.
        let decision = QWServerLifecycleRules.quitDecision(
            keepServerRunningAfterQuit: telemetry.settings.keepServerRunningAfterQuit,
            parentWatchdogArmed: telemetry.parentPidWatchdogArmed
        )
        if decision == .leaveServerRunning {
            QWServerLogger.shared.log(.info, "Menu bar app quitting: the server keeps running.")
            return .terminateNow
        }
        if !telemetry.isAnyServerRunning {
            return .terminateNow
        }
        if telemetry.settings.keepServerRunningAfterQuit {
            QWServerLogger.shared.log(.warn, "Menu bar app quitting: the running server has a parent-pid watchdog and would die with the app, so it is stopped now. Restart the server to make 'Keep server running after quit' take effect.")
        }

        isTerminating = true
        QWServerLogger.shared.log(.info, "Menu bar app quitting: stopping engine server...")

        Task { @MainActor in
            await telemetry.stopEngineAsync()
            sender.reply(toApplicationShouldTerminate: true)
        }

        return .terminateLater
    }

    public func applicationDidFinishLaunching(_ notification: Notification) {
        do {
            try QWEngineInstanceLock.acquireMenuLock()
        } catch {
            QWServerLogger.shared.log(.warn, "\(error.localizedDescription)")
            NSApp.terminate(nil)
            return
        }

        NSApp.setActivationPolicy(.accessory)

        // Configure file sink logger using persistent settings
        let logPath = telemetry.settings.expandedLogFilePath
        QWServerLogger.shared.configureFileSink(path: logPath)
        QWServerLogger.shared.log(.info, "QWEGN Menu Bar Controller initialized. Logging to \(logPath)")

        // Configure status item in menu bar
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        if let button = statusItem.button {
            button.image = createMenuBarIcon(isServing: telemetry.isServing)
            button.imagePosition = .imageOnly
            button.target = self
            button.action = #selector(togglePopover)
        }

        // Wire dynamic menu bar icon observation: gray when not serving, active when serving.
        // The icon reads the same two facts the popover draws: the
        // connection and the engine's published phase.
        telemetry.$dashboard
            .combineLatest(telemetry.$connection)
            .receive(on: RunLoop.main)
            .sink { [weak self] dashboard, connection in
                guard let self else { return }
                let isServing = connection == .connected
                    && dashboard?.lifecycle.phase == .loaded
                self.statusItem.button?.image = self.createMenuBarIcon(isServing: isServing)
            }
            .store(in: &cancellables)

        // Configure popover with authentic macOS system-level liquid glass
        popover = NSPopover()
        popover.contentSize = NSSize(width: 440, height: 550)
        popover.behavior = .transient
        popover.animates = true

        let contentView = CupertinoTelemetryView(
            telemetry: telemetry,
            onOpenWebUI: { [weak self] in self?.openWebUI() },
            onOpenLogs: { [weak self] in self?.openLogsWindow() },
            onOpenSettings: { [weak self] in self?.openSettingsWindow() },
            onQuit: { NSApp.terminate(nil) }
        )
        let hostingController = NSHostingController(rootView: contentView)
        hostingController.view.wantsLayer = true
        hostingController.view.layer?.backgroundColor = .clear
        popover.contentViewController = hostingController

        // The popover can be opened at launch, for a check that does not
        // need a click: `QWG_OPEN_POPOVER=1 qwg-menu`. The status item
        // is the only way in otherwise, and a click cannot be scripted
        // without assistive access.
        if ProcessInfo.processInfo.environment["QWG_OPEN_POPOVER"] == "1" {
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { [weak self] in
                guard let self, let button = self.statusItem.button else { return }
                self.popover.show(relativeTo: button.bounds, of: button, preferredEdge: .minY)
            }
        }

        // Register global keyboard monitor for ⌘R, ⌘D, ⌘L, ⌘,, ⌘Q
        NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            if event.modifierFlags.contains(.command) {
                switch event.charactersIgnoringModifiers {
                case "r":
                    self?.telemetry.restartEngine()
                    return nil
                case "d":
                    self?.openWebUI()
                    return nil
                case "l":
                    self?.openLogsWindow()
                    return nil
                case ",":
                    self?.openSettingsWindow()
                    return nil
                case "q":
                    NSApp.terminate(nil)
                    return nil
                default:
                    break
                }
            }
            return event
        }
    }

    public func openWebUI() {
        popover.performClose(nil)
        telemetry.openWebDashboard()
    }

    @objc private func togglePopover() {
        guard let button = statusItem.button else { return }
        if popover.isShown {
            popover.performClose(nil)
        } else {
            popover.show(relativeTo: button.bounds, of: button, preferredEdge: .minY)
            popover.contentViewController?.view.window?.makeKey()
        }
    }

    public func openSettingsWindow() {
        popover.performClose(nil)
        if let window = settingsWindow {
            window.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }

        let settingsView = SettingsView(
            telemetry: telemetry,
            onClose: { [weak self] in self?.settingsWindow?.close() }
        )
        let hosting = NSHostingController(rootView: settingsView)
        let window = NSWindow(contentViewController: hosting)
        window.title = "QWEGN Settings"
        window.styleMask = [.titled, .closable, .miniaturizable, .resizable]
        window.setContentSize(NSSize(width: 580, height: 490))
        window.minSize = NSSize(width: 520, height: 420)
        window.isReleasedWhenClosed = false
        window.center()
        self.settingsWindow = window

        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    public func openLogsWindow() {
        popover.performClose(nil)
        if let window = logsWindow {
            window.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }

        let logsView = LogViewerWindow(
            telemetry: telemetry,
            onClose: { [weak self] in self?.logsWindow?.close() }
        )
        let hosting = NSHostingController(rootView: logsView)
        let window = NSWindow(contentViewController: hosting)
        window.title = "QWEGN Logs"
        window.styleMask = [.titled, .closable, .miniaturizable, .resizable]
        window.setContentSize(NSSize(width: 780, height: 500))
        window.minSize = NSSize(width: 640, height: 380)
        window.isReleasedWhenClosed = false
        window.center()
        self.logsWindow = window

        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    /// Generates the 18x18 QG processor microchip emblem for the macOS menu bar:
    /// Active template image when serving model; dimmed non-template gray when stopped or in standby.
    private func createMenuBarIcon(isServing: Bool = false) -> NSImage {
        let size = NSSize(width: 18, height: 18)
        let image = NSImage(size: size, flipped: true) { rect in
            guard let ctx = NSGraphicsContext.current?.cgContext else { return false }
            let strokeColor = isServing ? NSColor.black.cgColor : NSColor(white: 0.55, alpha: 0.85).cgColor
            QGProcessorLogoView.draw(in: ctx, size: rect.size, strokeColor: strokeColor)
            return true
        }
        image.isTemplate = isServing
        return image
    }
}
