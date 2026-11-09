import Foundation
import Darwin
import QWCore

public enum QWLogLevel: Int, Sendable, Comparable {
    case debug = 0
    case info = 1
    case warn = 2
    case error = 3
    case none = 4

    public static func < (lhs: QWLogLevel, rhs: QWLogLevel) -> Bool {
        lhs.rawValue < rhs.rawValue
    }

    public static func fromString(_ str: String) -> QWLogLevel {
        switch str.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "debug": return .debug
        case "info": return .info
        case "warn", "warning": return .warn
        case "error": return .error
        case "none", "off": return .none
        default: return .info
        }
    }
}

public final class QWServerLogger: @unchecked Sendable {
    public static let shared = QWServerLogger()

    private var level: QWLogLevel = .info
    private var lock = os_unfair_lock_s()
    private var fileDescriptor: Int32 = -1
    private var configuredLogPath: String?

    // In-memory ring buffer of recent log lines for streaming to UI
    private var ringBuffer: [String] = []
    private let maxRingBufferSize = 2000
    private var subscribers: [UUID: @Sendable (String) -> Void] = [:]

    public init(level: QWLogLevel = .info) {
        self.level = level
    }

    deinit {
        if fileDescriptor >= 0 {
            close(fileDescriptor)
        }
    }

    public func setLevel(_ newLevel: QWLogLevel) {
        os_unfair_lock_lock(&lock)
        self.level = newLevel
        os_unfair_lock_unlock(&lock)
    }

    public func getLevel() -> QWLogLevel {
        os_unfair_lock_lock(&lock)
        let l = self.level
        os_unfair_lock_unlock(&lock)
        return l
    }

    /// Configures a persistent log file sink, creating parent directories if needed.
    public func configureFileSink(path: String) {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }

        if fileDescriptor >= 0 {
            close(fileDescriptor)
            fileDescriptor = -1
        }

        let expanded = path.expandingTildePath
        self.configuredLogPath = expanded

        let dir = (expanded as NSString).deletingLastPathComponent
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true, attributes: [
            .posixPermissions: 0o700
        ])

        let fd = open(expanded, O_WRONLY | O_CREAT | O_APPEND, 0o644)
        if fd >= 0 {
            self.fileDescriptor = fd
        }
    }

    /// Returns a snapshot of the recent in-memory log lines.
    public func getRecentLogs() -> [String] {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        return ringBuffer
    }

    /// Subscribes a listener to receive new log lines as they arrive.
    public func subscribe(_ handler: @escaping @Sendable (String) -> Void) -> UUID {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        let id = UUID()
        subscribers[id] = handler
        return id
    }

    /// Unsubscribes a listener.
    public func unsubscribe(_ id: UUID) {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        subscribers.removeValue(forKey: id)
    }

    private func broadcast(_ line: String) {
        // Collect subscribers under lock, call outside
        var callbacks: [@Sendable (String) -> Void] = []
        os_unfair_lock_lock(&lock)
        ringBuffer.append(line)
        if ringBuffer.count > maxRingBufferSize {
            ringBuffer.removeFirst(ringBuffer.count - maxRingBufferSize)
        }
        callbacks = Array(subscribers.values)
        os_unfair_lock_unlock(&lock)

        for cb in callbacks {
            cb(line)
        }
    }

    /// Emits a single rate-governed, non-allocating structured request log line directly
    /// to STDOUT_FILENO via stack allocation and POSIX write(2).
    public func logRequest(
        method: StaticString,
        path: StaticString,
        statusCode: Int,
        promptTokens: Int,
        completionTokens: Int,
        imagesCount: Int = 0,
        cacheHitTokens: Int = 0,
        cacheTier: String? = nil,
        cacheActive: Bool = false,
        specSummary: String? = nil,
        ttftMs: Double?,
        decodeTokPerSec: Double?,
        totalSec: Double
    ) {
        guard getLevel() <= .info else { return }

        // Stack-allocated 512-byte scratch buffer — zero heap allocations
        withUnsafeTemporaryAllocation(of: UInt8.self, capacity: 512) { buf in
            guard let ptr = buf.baseAddress else { return }
            var idx = 0

            func appendByte(_ b: UInt8) {
                if idx < 510 {
                    ptr[idx] = b
                    idx += 1
                }
            }

            func appendBytes(_ bytes: UnsafeRawBufferPointer) {
                let count = min(bytes.count, 510 - idx)
                if count > 0 {
                    memcpy(ptr + idx, bytes.baseAddress!, count)
                    idx += count
                }
            }

            func appendStatic(_ str: StaticString) {
                str.withUTF8Buffer { appendBytes(UnsafeRawBufferPointer($0)) }
            }

            func appendInt(_ val: Int) {
                if val == 0 {
                    appendByte(UInt8(ascii: "0"))
                    return
                }
                var v = val
                let neg = v < 0
                if neg { v = -v }
                var tmp = [UInt8](repeating: 0, count: 20)
                var pos = 0
                while v > 0 {
                    tmp[pos] = UInt8(ascii: "0") + UInt8(v % 10)
                    v /= 10
                    pos += 1
                }
                if neg { appendByte(UInt8(ascii: "-")) }
                while pos > 0 {
                    pos -= 1
                    appendByte(tmp[pos])
                }
            }

            func appendDouble(_ val: Double, precision: Int = 1) {
                if val.isNaN || val.isInfinite {
                    appendStatic("0.0")
                    return
                }
                let intPart = Int(val)
                appendInt(intPart)
                appendByte(UInt8(ascii: "."))
                var frac = abs(val - Double(intPart))
                for _ in 0..<precision {
                    frac *= 10
                    let digit = Int(frac) % 10
                    appendByte(UInt8(ascii: "0") + UInt8(digit))
                }
            }

            var now = time(nil)
            var tmVal = tm()
            localtime_r(&now, &tmVal)
            var timeBuf = [CChar](repeating: 0, count: 32)
            let timeLen = strftime(&timeBuf, 32, "[%Y-%m-%d %H:%M:%S]", &tmVal)
            if timeLen > 0 {
                timeBuf.withUnsafeBufferPointer { tptr in
                    appendBytes(UnsafeRawBufferPointer(start: tptr.baseAddress, count: timeLen))
                }
            }
            appendByte(UInt8(ascii: " "))

            appendStatic(method)
            appendByte(UInt8(ascii: " "))
            appendStatic(path)
            appendStatic(" -> ")
            appendInt(statusCode)
            appendStatic(statusCode == 200 ? " OK | " : " ERR | ")

            appendInt(promptTokens)
            appendStatic(" prompt")
            if imagesCount > 0 {
                appendStatic(" (")
                appendInt(imagesCount)
                appendStatic(imagesCount == 1 ? " img)" : " imgs)")
            }
            if cacheHitTokens > 0 {
                appendStatic(" (hit ")
                appendInt(cacheHitTokens)
                if let tier = cacheTier {
                    appendByte(UInt8(ascii: " "))
                    appendByte(UInt8(ascii: "["))
                    for b in tier.utf8 { appendByte(b) }
                    appendByte(UInt8(ascii: "]"))
                }
                appendByte(UInt8(ascii: ")"))
            } else if cacheActive {
                appendStatic(" (miss)")
            }
            appendStatic(" + ")
            appendInt(completionTokens)
            appendStatic(" gen tok | TTFT ")
            if let ttftMs {
                appendDouble(ttftMs, precision: 1)
                appendStatic("ms | ")
            } else {
                appendStatic("— | ")
            }
            if let decodeTokPerSec {
                appendDouble(decodeTokPerSec, precision: 1)
                appendStatic(" tok/s | ")
            } else {
                appendStatic("— tok/s | ")
            }
            appendDouble(totalSec, precision: 2)
            appendStatic("s")
            if let spec = specSummary, !spec.isEmpty {
                appendStatic(" | spec: ")
                for b in spec.utf8 { appendByte(b) }
            }
            appendByte(UInt8(ascii: "\n"))

            // Write to stdout and file
            os_unfair_lock_lock(&self.lock)
            write(STDOUT_FILENO, ptr, idx)
            if self.fileDescriptor >= 0 {
                write(self.fileDescriptor, ptr, idx)
            }
            os_unfair_lock_unlock(&self.lock)

            if let str = String(bytes: UnsafeRawBufferPointer(start: ptr, count: idx), encoding: .utf8) {
                broadcast(str)
            }
        }
    }

    /// Emits simple structured log message
    public func log(_ lvl: QWLogLevel, _ message: String) {
        guard lvl >= getLevel() else { return }

        var now = time(nil)
        var tmVal = tm()
        localtime_r(&now, &tmVal)
        var timeBuf = [CChar](repeating: 0, count: 32)
        let timeLen = strftime(&timeBuf, 32, "[%Y-%m-%d %H:%M:%S]", &tmVal)
        let timeStr = timeLen > 0 ? (timeBuf.withUnsafeBufferPointer { ptr in ptr.baseAddress.map { String(cString: $0) } } ?? "") : ""

        let levelTag: String
        switch lvl {
        case .debug: levelTag = "DEBUG"
        case .info:  levelTag = "INFO "
        case .warn:  levelTag = "WARN "
        case .error: levelTag = "ERROR"
        case .none:  return
        }

        let line = "\(timeStr) \(levelTag) \(message)\n"
        line.utf8CString.withUnsafeBytes { raw in
            os_unfair_lock_lock(&self.lock)
            write(lvl >= .warn ? STDERR_FILENO : STDOUT_FILENO, raw.baseAddress, raw.count - 1)
            if self.fileDescriptor >= 0 {
                write(self.fileDescriptor, raw.baseAddress, raw.count - 1)
            }
            os_unfair_lock_unlock(&self.lock)
        }

        broadcast(line)
    }

    /// Forwards a line captured from a child `qwg` process without double-prefixing timestamps
    /// when the child line is already formatted by `QWServerLogger`.
    public func logChildOutputLine(_ rawLine: String) {
        let utf8 = Array(rawLine.utf8)
        let isPreformatted = utf8.count >= 21
            && utf8[0] == UInt8(ascii: "[")
            && utf8[5] == UInt8(ascii: "-")
            && utf8[8] == UInt8(ascii: "-")
            && utf8[11] == UInt8(ascii: " ")
            && utf8[14] == UInt8(ascii: ":")
            && utf8[17] == UInt8(ascii: ":")
            && utf8[20] == UInt8(ascii: "]")
        if isPreformatted {
            let line = rawLine.hasSuffix("\n") ? rawLine : (rawLine + "\n")
            line.utf8CString.withUnsafeBytes { raw in
                os_unfair_lock_lock(&self.lock)
                write(STDOUT_FILENO, raw.baseAddress, raw.count - 1)
                if self.fileDescriptor >= 0 {
                    write(self.fileDescriptor, raw.baseAddress, raw.count - 1)
                }
                os_unfair_lock_unlock(&self.lock)
            }
            broadcast(line)
        } else {
            log(.info, "[qwg] \(rawLine)")
        }
    }

    /// Emits the hardware and engine audit banner directly to stdout and sinks
    public func logStartupBanner(_ banner: String) {
        banner.utf8CString.withUnsafeBytes { raw in
            os_unfair_lock_lock(&self.lock)
            write(STDOUT_FILENO, raw.baseAddress, raw.count - 1)
            if self.fileDescriptor >= 0 {
                write(self.fileDescriptor, raw.baseAddress, raw.count - 1)
            }
            os_unfair_lock_unlock(&self.lock)
        }
        broadcast(banner)
    }
}
