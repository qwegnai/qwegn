// QWServerLifecycleRules.swift — The rule that keeps the app and the
// server's parent-pid watchdog from disagreeing about whether the server
// should outlive the app, and the rule that keeps a live server from
// being reported as gone.

import Foundation

/// What the app may claim about the server.
public enum QWServerLiveness: String, Sendable, CaseIterable {
    /// The server answered.
    case serving
    /// The server did not answer, but its process is still there.
    case unreachable
    /// The server did not answer and its process is gone.
    case gone
}

/// What the app does with the server when it quits.
public enum QWServerQuitDecision: String, Sendable, Equatable {
    /// Leave the server running: it outlives the app.
    case leaveServerRunning
    /// Stop the server before the app goes.
    case stopServer
}

/// The lifecycle rules the app and the server share.
public enum QWServerLifecycleRules {
    /// The rule, in one place: the parent pid is never evidence of death.
    /// A server that answers is serving; a server whose process is alive
    /// but quiet is unreachable — not gone.
    public static func liveness(answered: Bool, processAlive: Bool) -> QWServerLiveness {
        if answered { return .serving }
        return processAlive ? .unreachable : .gone
    }

    /// The parent-pid watchdog is armed when the server starts, and only
    /// then: when the server is meant to die with the app that started it,
    /// which is `keepServerRunningAfterQuit` off.
    public static func armsParentPidWatchdog(keepServerRunningAfterQuit: Bool) -> Bool {
        !keepServerRunningAfterQuit
    }

    /// The arguments that arm the watchdog, or none.
    public static func parentPidArguments(keepServerRunningAfterQuit: Bool,
                                          parentPid: Int32) -> [String] {
        guard armsParentPidWatchdog(keepServerRunningAfterQuit: keepServerRunningAfterQuit),
              parentPid > 0 else {
            return []
        }
        return ["--parent-pid", String(parentPid)]
    }

    /// What to do with the server when the app quits. It outlives the app
    /// only when the setting says so *and* no watchdog would kill it; a
    /// server that would die anyway is stopped on the way out.
    public static func quitDecision(keepServerRunningAfterQuit: Bool,
                                    parentWatchdogArmed: Bool) -> QWServerQuitDecision {
        let outlives = keepServerRunningAfterQuit && !parentWatchdogArmed
        return outlives ? .leaveServerRunning : .stopServer
    }
}
