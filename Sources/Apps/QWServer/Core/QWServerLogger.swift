import Foundation
import Darwin
import os
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

    private struct LogState: Sendable {
        var level: QWLogLevel
        var configuredLogPath: String?
        var ringBuffer: [String] = []
        var subscribers: [UUID: @Sendable (String) -> Void] = [:]
    }

    private struct FileSinkState: Sendable {
        var fd: Int32 = -1
        var activeWriters: Int = 0
        var retiredFDs: [Int32] = []
    }

    private let maxRingBufferSize = 2000
    private let stateLock: OSAllocatedUnfairLock<LogState>
    private let fileSinkLock = OSAllocatedUnfairLock<FileSinkState>(initialState: FileSinkState())

    public init(level: QWLogLevel = .info) {
        self.stateLock = OSAllocatedUnfairLock(initialState: LogState(level: level))
    }

    deinit {
        let fdsToClose = fileSinkLock.withLock { state -> [Int32] in
            var fds = state.retiredFDs
            state.retiredFDs.removeAll()
            if state.fd >= 0 {
                fds.append(state.fd)
                state.fd = -1
            }
            return fds
        }
        for fd in fdsToClose {
            close(fd)
        }
    }

    public func setLevel(_ newLevel: QWLogLevel) {
        stateLock.withLock { $0.level = newLevel }
    }

    public func getLevel() -> QWLogLevel {
        stateLock.withLock { $0.level }
    }

    /// Configures a persistent log file sink, creating parent directories if needed.
    public func configureFileSink(path: String) {
        let expanded = path.expandingTildePath
        let dir = (expanded as NSString).deletingLastPathComponent
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true, attributes: [
            .posixPermissions: 0o700
        ])
        let newFD = open(expanded, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0o644)

        stateLock.withLock { $0.configuredLogPath = expanded }
        let retired = fileSinkLock.withLock { state -> [Int32] in
            let oldFD = state.fd
            state.fd = newFD
            guard oldFD >= 0 else { return [] }
            if state.activeWriters == 0 {
                return [oldFD]
            }
            state.retiredFDs.append(oldFD)
            return []
        }
        for fd in retired {
            close(fd)
        }
    }

    /// Returns a snapshot of the recent in-memory log lines.
    public func getRecentLogs() -> [String] {
        stateLock.withLock { $0.ringBuffer }
    }

    /// Subscribes a listener to receive new log lines as they arrive.
    public func subscribe(_ handler: @escaping @Sendable (String) -> Void) -> UUID {
        stateLock.withLock { state in
            let id = UUID()
            state.subscribers[id] = handler
            return id
        }
    }

    /// Unsubscribes a listener.
    public func unsubscribe(_ id: UUID) {
        stateLock.withLock { _ = $0.subscribers.removeValue(forKey: id) }
    }

    private func broadcast(_ line: String) {
        let callbacks = stateLock.withLock { state -> [@Sendable (String) -> Void] in
            state.ringBuffer.append(line)
            if state.ringBuffer.count > maxRingBufferSize {
                state.ringBuffer.removeFirst(state.ringBuffer.count - maxRingBufferSize)
            }
            return Array(state.subscribers.values)
        }
        for cb in callbacks {
            cb(line)
        }
    }

    private static func writeAll(fd: Int32, buffer: UnsafeRawPointer, count: Int) {
        var written = 0
        while written < count {
            let rc = Darwin.write(fd, buffer.advanced(by: written), count - written)
            if rc > 0 {
                written += rc
            } else if rc < 0 && errno == EINTR {
                continue
            } else {
                break
            }
        }
    }

    private func emitOutput(streamFD: Int32, buffer: UnsafeRawPointer, count: Int) {
        guard count > 0 else { return }
        Self.writeAll(fd: streamFD, buffer: buffer, count: count)

        let fd = fileSinkLock.withLock { state -> Int32 in
            guard state.fd >= 0 else { return -1 }
            state.activeWriters += 1
            return state.fd
        }
        guard fd >= 0 else { return }
        defer {
            let retired = fileSinkLock.withLock { state -> [Int32] in
                state.activeWriters -= 1
                guard state.activeWriters == 0, !state.retiredFDs.isEmpty else { return [] }
                let out = state.retiredFDs
                state.retiredFDs.removeAll()
                return out
            }
            for oldFD in retired {
                close(oldFD)
            }
        }
        Self.writeAll(fd: fd, buffer: buffer, count: count)
    }

    /// Emits a single rate-governed, non-allocating structured request log line directly
    /// to STDOUT_FILENO via stack allocation and POSIX write(2).
    public func logRequest(
        method: String,
        path: String,
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
        totalSec: Double,
        client: String? = nil,
        requestId: String? = nil,
        sessionId: String? = nil,
        peerHost: String? = nil
    ) {
        guard getLevel() <= .info else { return }

        // Stack-allocated 768-byte scratch buffer — zero heap allocations
        withUnsafeTemporaryAllocation(of: UInt8.self, capacity: 768) { buf in
            guard let ptr = buf.baseAddress else { return }
            var idx = 0

            func appendByte(_ b: UInt8) {
                if idx < 766 {
                    ptr[idx] = b
                    idx += 1
                }
            }

            func appendBytes(_ bytes: UnsafeRawBufferPointer) {
                let count = min(bytes.count, 766 - idx)
                if count > 0, let src = bytes.baseAddress {
                    memcpy(ptr + idx, src, count)
                    idx += count
                }
            }

            func appendStatic(_ str: StaticString) {
                str.withUTF8Buffer { appendBytes(UnsafeRawBufferPointer($0)) }
            }

            func appendString(_ str: String) {
                for b in str.utf8 where b != UInt8(ascii: "\r") && b != UInt8(ascii: "\n") {
                    appendByte(b)
                }
            }

            func appendInt(_ val: Int) {
                if val == 0 {
                    appendByte(UInt8(ascii: "0"))
                    return
                }
                let neg = val < 0
                var v: UInt = val.magnitude
                var tmp = [UInt8](repeating: 0, count: 21)
                var pos = 0
                while v > 0 && pos < tmp.count {
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
                let clampedPrecision = min(12, max(0, precision))
                let truncated = val.rounded(.towardZero)
                guard let intPart = Int(exactly: truncated) else {
                    if val < 0 { appendByte(UInt8(ascii: "-")) }
                    var mag = abs(val)
                    var exp = 0
                    while mag >= 10.0 && exp < 400 {
                        mag /= 10.0
                        exp += 1
                    }
                    let lead = min(9, max(0, Int(mag)))
                    appendByte(UInt8(ascii: "0") + UInt8(lead))
                    appendByte(UInt8(ascii: "."))
                    var rem = max(0.0, mag - Double(lead))
                    for _ in 0..<max(1, clampedPrecision) {
                        rem *= 10.0
                        let digit = min(9, max(0, Int(rem)))
                        appendByte(UInt8(ascii: "0") + UInt8(digit))
                        rem = max(0.0, rem - Double(digit))
                    }
                    appendStatic("e+")
                    appendInt(exp)
                    return
                }
                if val < 0 && intPart == 0 {
                    appendByte(UInt8(ascii: "-"))
                }
                appendInt(intPart)
                appendByte(UInt8(ascii: "."))
                var frac = abs(val - truncated)
                for _ in 0..<clampedPrecision {
                    frac *= 10
                    let digit = min(9, max(0, Int(frac) % 10))
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

            appendString(method)
            appendByte(UInt8(ascii: " "))
            appendString(path)
            appendStatic(" -> ")
            appendInt(statusCode)
            appendStatic(statusCode == 200 ? " OK | " : " ERR | ")

            let hasTraceMeta = (client?.isEmpty == false)
                || (requestId?.isEmpty == false)
                || (sessionId?.isEmpty == false)
                || (peerHost?.isEmpty == false)
            if hasTraceMeta {
                var wroteField = false
                if let client, !client.isEmpty {
                    appendStatic("client=")
                    appendString(client)
                    wroteField = true
                }
                if let requestId, !requestId.isEmpty {
                    if wroteField { appendByte(UInt8(ascii: " ")) }
                    appendStatic("req=")
                    appendString(requestId)
                    wroteField = true
                }
                if let sessionId, !sessionId.isEmpty {
                    if wroteField { appendByte(UInt8(ascii: " ")) }
                    appendStatic("sess=")
                    appendString(sessionId)
                    wroteField = true
                }
                if let peerHost, !peerHost.isEmpty {
                    if wroteField { appendByte(UInt8(ascii: " ")) }
                    appendStatic("peer=")
                    appendString(peerHost)
                }
                appendStatic(" | ")
            }

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

            emitOutput(streamFD: STDOUT_FILENO, buffer: ptr, count: idx)

            if let str = String(bytes: UnsafeRawBufferPointer(start: ptr, count: idx), encoding: .utf8) {
                broadcast(str)
            }
        }
    }

    /// Emits a structured HTTP request log line for non-chat/control routes and cancellations.
    public func logSimpleRequest(
        method: String,
        path: String,
        statusCode: Int,
        client: String? = nil,
        requestId: String? = nil,
        sessionId: String? = nil,
        peerHost: String? = nil,
        detail: String? = nil,
        totalSec: Double
    ) {
        guard getLevel() <= .info else { return }

        var now = time(nil)
        var tmVal = tm()
        localtime_r(&now, &tmVal)
        var timeBuf = [CChar](repeating: 0, count: 32)
        let timeLen = strftime(&timeBuf, 32, "[%Y-%m-%d %H:%M:%S]", &tmVal)
        let timeStr = timeLen > 0 ? (timeBuf.withUnsafeBufferPointer { ptr in ptr.baseAddress.map { String(cString: $0) } } ?? "") : ""

        let okTag = (statusCode >= 200 && statusCode < 300) ? "OK" : "ERR"
        var sections: [String] = ["\(timeStr) \(method) \(path) -> \(statusCode) \(okTag)"]

        var traceParts: [String] = []
        if let client, !client.isEmpty { traceParts.append("client=\(client)") }
        if let requestId, !requestId.isEmpty { traceParts.append("req=\(requestId)") }
        if let sessionId, !sessionId.isEmpty { traceParts.append("sess=\(sessionId)") }
        if let peerHost, !peerHost.isEmpty { traceParts.append("peer=\(peerHost)") }
        if !traceParts.isEmpty { sections.append(traceParts.joined(separator: " ")) }

        if let detail, !detail.isEmpty {
            let cleanDetail = detail
                .replacingOccurrences(of: "\r", with: " ")
                .replacingOccurrences(of: "\n", with: " ")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !cleanDetail.isEmpty { sections.append(cleanDetail) }
        }
        sections.append(String(format: "%.2fs", max(0.0, totalSec)))

        let line = sections.joined(separator: " | ") + "\n"
        line.utf8CString.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            let byteCount = max(0, raw.count - 1)
            if byteCount > 0, let base = raw.baseAddress {
                emitOutput(streamFD: STDOUT_FILENO, buffer: base, count: byteCount)
            }
        }
        broadcast(line)
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
        line.utf8CString.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            let byteCount = max(0, raw.count - 1)
            if byteCount > 0, let base = raw.baseAddress {
                emitOutput(streamFD: lvl >= .warn ? STDERR_FILENO : STDOUT_FILENO, buffer: base, count: byteCount)
            }
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
            line.utf8CString.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                let byteCount = max(0, raw.count - 1)
                if byteCount > 0, let base = raw.baseAddress {
                    emitOutput(streamFD: STDOUT_FILENO, buffer: base, count: byteCount)
                }
            }
            broadcast(line)
        } else {
            log(.info, "[qwg] \(rawLine)")
        }
    }

    /// Emits the hardware and engine audit banner directly to stdout and sinks
    public func logStartupBanner(_ banner: String) {
        banner.utf8CString.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            let byteCount = max(0, raw.count - 1)
            if byteCount > 0, let base = raw.baseAddress {
                emitOutput(streamFD: STDOUT_FILENO, buffer: base, count: byteCount)
            }
        }
        broadcast(banner)
    }
}
