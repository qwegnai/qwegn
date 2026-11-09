import Foundation

/// Centralized path expansion, canonicalization, and `~/.qwg` workspace layout.
public enum QWPaths {
    public static var rootDirectoryURL: URL {
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".qwg", isDirectory: true)
    }

    public static var settingsFileURL: URL {
        rootDirectoryURL.appendingPathComponent("settings.json", isDirectory: false)
    }

    public static var settingsFilePath: String { settingsFileURL.path }

    public static var modelsDirectoryURL: URL {
        rootDirectoryURL.appendingPathComponent("models", isDirectory: true)
    }

    public static var modelsDirectoryPath: String { modelsDirectoryURL.path }

    /// The Hugging Face hub cache the `hf` CLI writes to:
    /// `HF_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`.
    public static var hfHubCacheDirectoryURL: URL {
        let env = ProcessInfo.processInfo.environment
        if let cache = env["HF_HUB_CACHE"], !cache.isEmpty {
            return URL(fileURLWithPath: cache.expandingTildePath, isDirectory: true)
        }
        if let home = env["HF_HOME"], !home.isEmpty {
            return URL(fileURLWithPath: home.expandingTildePath, isDirectory: true)
                .appendingPathComponent("hub", isDirectory: true)
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub", isDirectory: true)
    }

    public static var defaultKVCacheDirectoryURL: URL {
        rootDirectoryURL.appendingPathComponent("kv-cache", isDirectory: true)
    }

    /// The directory holding one knob profile file per model.
    public static var profilesDirectoryURL: URL {
        rootDirectoryURL.appendingPathComponent("profiles", isDirectory: true)
    }

    public static var pidFilePath: String {
        rootDirectoryURL.appendingPathComponent("qwg.pid", isDirectory: false).path
    }

    public static var engineLockFilePath: String {
        rootDirectoryURL.appendingPathComponent("qwg-engine.lock", isDirectory: false).path
    }

    public static var menuLockFilePath: String {
        rootDirectoryURL.appendingPathComponent("qwg-menu.lock", isDirectory: false).path
    }

    /// Per-user Application Support folder (`~/Library/Application Support/Qwegn`).
    public static var applicationSupportDirectoryURL: URL {
        if let env = ProcessInfo.processInfo.environment["QWEGN_APP_SUPPORT_DIR"], !env.isEmpty {
            return URL(fileURLWithPath: env.expandingTildePath, isDirectory: true)
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/Qwegn", isDirectory: true)
    }

    /// Lazy Python virtual environment for `qwg quantize` (`~/Library/Application Support/Qwegn/venv`).
    public static var pythonVenvDirectoryURL: URL {
        if let env = ProcessInfo.processInfo.environment["QWEGN_VENV_DIR"], !env.isEmpty {
            return URL(fileURLWithPath: env.expandingTildePath, isDirectory: true)
        }
        return applicationSupportDirectoryURL.appendingPathComponent("venv", isDirectory: true)
    }

    public static var pythonVenvExecutableURL: URL {
        pythonVenvDirectoryURL.appendingPathComponent("bin/python", isDirectory: false)
    }

    /// Support root holding `Scripts/quantize/pack.py` (repo root in dev, `/usr/local/share/qwegn` in `.pkg`).
    public static var supportDirectoryURL: URL {
        let fm = FileManager.default
        if let env = ProcessInfo.processInfo.environment["QWEGINE_ROOT"], !env.isEmpty {
            return URL(fileURLWithPath: env.expandingTildePath, isDirectory: true)
        }
        let execDir = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath().deletingLastPathComponent()
        var d = execDir
        for _ in 0..<8 {
            if fm.fileExists(atPath: d.appendingPathComponent("Scripts/quantize/pack.py").path) { return d }
            d = d.deletingLastPathComponent()
        }
        let candidates = [
            execDir.deletingLastPathComponent().appendingPathComponent("share/qwegn", isDirectory: true),
            URL(fileURLWithPath: "/usr/local/share/qwegn", isDirectory: true),
            URL(fileURLWithPath: "/Applications/Qwegn.app/Contents/Resources", isDirectory: true),
        ]
        for cand in candidates where fm.fileExists(atPath: cand.appendingPathComponent("Scripts/quantize/pack.py").path) {
            return cand
        }
        return URL(fileURLWithPath: fm.currentDirectoryPath, isDirectory: true)
    }
}

public enum QWEngineInstanceError: Error, LocalizedError, CustomStringConvertible, Equatable {
    case engineAlreadyRunning(pid: Int32)
    case menuAlreadyRunning(pid: Int32)

    public var description: String {
        switch self {
        case .engineAlreadyRunning(let pid):
            return "Refusing to start a second engine instance: another qwg engine is already running (PID \(pid)). Stop it first before launching a new engine."
        case .menuAlreadyRunning(let pid):
            return "Another qwg-menu instance is already running (PID \(pid))."
        }
    }

    public var errorDescription: String? { description }
}

/// Process-level single-instance lock using kernel `flock(LOCK_EX | LOCK_NB)`.
/// Automatically released by the OS kernel if the holding process exits or crashes.
public enum QWEngineInstanceLock {
    private static let lock = NSLock()
    private nonisolated(unsafe) static var engineLockFD: Int32 = -1
    private nonisolated(unsafe) static var menuLockFD: Int32 = -1

    /// Acquires an exclusive kernel lock for the `qwg` engine process (`serve`, `run`, `repl`).
    /// Idempotent within the same process. Throws `QWEngineInstanceError.engineAlreadyRunning`
    /// if another OS process currently holds the engine lock.
    public static func acquireEngineLock() throws {
        lock.lock()
        defer { lock.unlock() }
        if engineLockFD >= 0 { return }

        try? FileManager.default.createDirectory(at: QWPaths.rootDirectoryURL, withIntermediateDirectories: true)
        let path = QWPaths.engineLockFilePath
        let fd = open(path, O_CREAT | O_RDWR, 0o644)
        guard fd >= 0 else { return }

        if flock(fd, LOCK_EX | LOCK_NB) != 0 {
            let existingPid = readPID(fromPath: path) ?? readPID(fromPath: QWPaths.pidFilePath) ?? -1
            close(fd)
            throw QWEngineInstanceError.engineAlreadyRunning(pid: existingPid)
        }

        let pidStr = "\(ProcessInfo.processInfo.processIdentifier)\n"
        _ = ftruncate(fd, 0)
        _ = lseek(fd, 0, SEEK_SET)
        pidStr.withCString { ptr in
            _ = write(fd, ptr, strlen(ptr))
        }
        try? "\(ProcessInfo.processInfo.processIdentifier)".write(
            toFile: QWPaths.pidFilePath, atomically: true, encoding: .utf8
        )
        engineLockFD = fd
    }

    /// Releases the engine lock held by this process, if any.
    public static func releaseEngineLock() {
        lock.lock()
        defer { lock.unlock() }
        guard engineLockFD >= 0 else { return }
        _ = flock(engineLockFD, LOCK_UN)
        close(engineLockFD)
        engineLockFD = -1
        try? FileManager.default.removeItem(atPath: QWPaths.pidFilePath)
    }

    /// Returns the PID of a currently running `qwg` engine process if another process holds the lock, or `nil`.
    public static func runningEnginePID() -> Int32? {
        lock.lock()
        let heldHere = engineLockFD >= 0
        lock.unlock()
        if heldHere {
            return ProcessInfo.processInfo.processIdentifier
        }
        let path = QWPaths.engineLockFilePath
        let fd = open(path, O_RDONLY)
        if fd >= 0 {
            if flock(fd, LOCK_EX | LOCK_NB) != 0 {
                close(fd)
                if let pid = readPID(fromPath: path), pid > 0, kill(pid, 0) == 0 {
                    return pid
                }
            } else {
                _ = flock(fd, LOCK_UN)
                close(fd)
            }
        }
        if let pid = readPID(fromPath: QWPaths.pidFilePath), pid > 0, kill(pid, 0) == 0 {
            return pid
        }
        return nil
    }

    /// Acquires an exclusive kernel lock for `qwg-menu` so two menu bar apps never run simultaneously.
    public static func acquireMenuLock() throws {
        lock.lock()
        defer { lock.unlock() }
        if menuLockFD >= 0 { return }

        try? FileManager.default.createDirectory(at: QWPaths.rootDirectoryURL, withIntermediateDirectories: true)
        let path = QWPaths.menuLockFilePath
        let fd = open(path, O_CREAT | O_RDWR, 0o644)
        guard fd >= 0 else { return }

        if flock(fd, LOCK_EX | LOCK_NB) != 0 {
            let existingPid = readPID(fromPath: path) ?? -1
            close(fd)
            throw QWEngineInstanceError.menuAlreadyRunning(pid: existingPid)
        }

        let pidStr = "\(ProcessInfo.processInfo.processIdentifier)\n"
        _ = ftruncate(fd, 0)
        _ = lseek(fd, 0, SEEK_SET)
        pidStr.withCString { ptr in
            _ = write(fd, ptr, strlen(ptr))
        }
        menuLockFD = fd
    }

    private static func readPID(fromPath path: String) -> Int32? {
        guard let str = try? String(contentsOfFile: path, encoding: .utf8) else { return nil }
        return Int32(str.trimmingCharacters(in: .whitespacesAndNewlines))
    }
}

extension String {
    /// Expands a leading `~` or `~user` to the full filesystem path.
    @inline(__always)
    public var expandingTildePath: String {
        (self as NSString).expandingTildeInPath
    }

    /// Canonicalized filesystem path expanding leading tilde (`~`) if present.
    @inline(__always)
    public var canonicalPath: String {
        expandingTildePath
    }

    /// Expands `~` and returns a file `URL`.
    @inline(__always)
    public var canonicalFileURL: URL {
        URL(fileURLWithPath: expandingTildePath)
    }
}
