import Foundation
import Dispatch

/// C02 / Track 3 — QWDispatchIO: High-throughput asynchronous NVMe streaming engine.
///
/// Provides direct unbuffered DMA transfers via Darwin `F_NOCACHE` and libdispatch
public enum QWDispatchIOError: Error, CustomStringConvertible, Sendable {
    case openFailed(path: String, errno: Int32)
    case readFailed(offset: Int64, length: Int, errno: Int32)
    case writeFailed(offset: Int64, length: Int, errno: Int32)
    case channelClosed
    case invalidAlignment(offset: Int64, alignment: Int)
    case shortRead(expected: Int, actual: Int)

    public var description: String {
        switch self {
        case .openFailed(let path, let err):
            return "QWDispatchIO: open failed for '\(path)': \(String(cString: strerror(err)))"
        case .readFailed(let off, let len, let err):
            return "QWDispatchIO: read failed at offset \(off) (length \(len)): \(String(cString: strerror(err)))"
        case .writeFailed(let off, let len, let err):
            return "QWDispatchIO: write failed at offset \(off) (length \(len)): \(String(cString: strerror(err)))"
        case .channelClosed:
            return "QWDispatchIO: channel is closed"
        case .invalidAlignment(let off, let align):
            return "QWDispatchIO: offset \(off) is not aligned to \(align) bytes"
        case .shortRead(let exp, let act):
            return "QWDispatchIO: short read (expected \(exp) bytes, got \(act))"
        }
    }
}

public final class QWDispatchIO: @unchecked Sendable {
    /// Host VM page size (16 KiB on Apple Silicon) — sysconf, never hardcoded.
    public static let pageSize: Int = Int(sysconf(_SC_PAGESIZE))

    public enum AccessMode: Sendable {
        case readOnly
        case writeOnly
        case readWrite

        var posixFlags: Int32 {
            switch self {
            case .readOnly: return O_RDONLY
            case .writeOnly: return O_WRONLY | O_CREAT
            case .readWrite: return O_RDWR | O_CREAT
            }
        }
    }

    public let path: String
    public let mode: AccessMode
    public let directIO: Bool
    public let fd: Int32

    private let channel: DispatchIO
    private let queue: DispatchQueue
    private let lock = NSLock()
    private var isClosed = false

    /// Initialize asynchronous I/O channel for the specified file.
    ///
    /// - Parameters:
    public init(path: String,
                mode: AccessMode = .readOnly,
                directIO: Bool = true,
                queue: DispatchQueue? = nil) throws {
        self.path = path
        self.mode = mode
        self.directIO = directIO

        let ioQueue = queue ?? DispatchQueue(label: "qw.dispatch_io.\(UUID().uuidString)",
                                             qos: .userInitiated,
                                             attributes: .concurrent)
        self.queue = ioQueue

        let openFlags = mode.posixFlags
        let fd = open(path, openFlags, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH)
        guard fd >= 0 else {
            throw QWDispatchIOError.openFailed(path: path, errno: errno)
        }
        self.fd = fd

        if directIO {
            // Bypass page cache for direct DMA transfers from NVMe SSD
            _ = fcntl(fd, F_NOCACHE, 1)
        }

        let chan = DispatchIO(type: .random, fileDescriptor: fd, queue: ioQueue, cleanupHandler: { cleanupErr in
            Darwin.close(fd)
        })
        self.channel = chan
        chan.setLimit(lowWater: 1)
    }

    deinit {
        close()
    }

    /// Explicitly close the asynchronous I/O channel and release the file descriptor.
    public func close() {
        lock.lock()
        defer { lock.unlock() }
        if !isClosed {
            isClosed = true
            channel.close(flags: .stop)
        }
    }

    private func checkOpen() throws {
        lock.lock()
        defer { lock.unlock() }
        if isClosed {
            throw QWDispatchIOError.channelClosed
        }
    }

    /// Read raw bytes asynchronously at the specified offset.
    public func read(offset: Int64, length: Int) async throws -> DispatchData {
        try checkOpen()
        return try await withCheckedThrowingContinuation { continuation in
            var accumulated = DispatchData.empty
            channel.read(offset: off_t(offset), length: length, queue: queue) { done, data, error in
                if let data, data.count > 0 {
                    accumulated.append(data)
                }
                if error != 0 {
                    continuation.resume(throwing: QWDispatchIOError.readFailed(offset: offset, length: length, errno: error))
                    return
                }
                if done {
                    continuation.resume(returning: accumulated)
                }
            }
        }
    }

    /// Read directly into a pre-allocated raw memory buffer. Delegates to the
    /// single-copy `pread` loop on the I/O queue (no DispatchData staging).
    public func read(offset: Int64, length: Int, into dst: UnsafeMutableRawPointer) async throws -> Int {
        let dstAddr = Int(bitPattern: dst)
        return try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Int, Error>) in
            queue.async {
                do {
                    guard let ptr = UnsafeMutableRawPointer(bitPattern: dstAddr) else {
                        throw QWDispatchIOError.readFailed(offset: offset, length: length, errno: EINVAL)
                    }
                    continuation.resume(returning: try self.readSync(offset: offset, into: ptr, length: length))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    /// Asynchronously read in fixed-size chunks, invoking the handler for each chunk.
    public func readChunks(offset: Int64,
                           length: Int,
                           chunkSize: Int,
                           handler: (Int64, DispatchData) throws -> Void) async throws {
        try checkOpen()
        var currentOffset = offset
        let endOffset = offset + Int64(length)

        while currentOffset < endOffset {
            let bytesToRead = min(chunkSize, Int(endOffset - currentOffset))
            let chunk = try await read(offset: currentOffset, length: bytesToRead)
            try handler(currentOffset, chunk)
            currentOffset += Int64(bytesToRead)
        }
    }

    /// Write raw DispatchData asynchronously at the specified offset.
    public func write(offset: Int64, data: DispatchData) async throws {
        try checkOpen()
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            channel.write(offset: off_t(offset), data: data, queue: queue) { done, remainingData, error in
                if error != 0 {
                    continuation.resume(throwing: QWDispatchIOError.writeFailed(offset: offset, length: data.count, errno: error))
                    return
                }
                if done {
                    continuation.resume(returning: ())
                }
            }
        }
    }

    /// Write from a raw pointer buffer asynchronously at the specified offset.
    public func write(offset: Int64, buffer: UnsafeRawPointer, length: Int) async throws {
        let data = DispatchData(bytes: UnsafeRawBufferPointer(start: buffer, count: length))
        try await write(offset: offset, data: data)
    }

    /// Write from a byte array asynchronously at the specified offset.
    public func write(offset: Int64, bytes: [UInt8]) async throws {
        let dispatchData = bytes.withUnsafeBytes { DispatchData(bytes: $0) }
        try await write(offset: offset, data: dispatchData)
    }

    /// Write from Foundation Data asynchronously at the specified offset.
    public func write(offset: Int64, data: Data) async throws {
        let dispatchData = data.withUnsafeBytes { DispatchData(bytes: $0) }
        try await write(offset: offset, data: dispatchData)
    }

    /// Read asynchronously returning Foundation Data.
    public func readData(offset: Int64, length: Int) async throws -> Data {
        let d = try await read(offset: offset, length: length)
        return Data(d)
    }

    /// Synchronous read fallback using POSIX `pread` with an `EINTR` retry loop.
    public func readSync(offset: Int64, into dst: UnsafeMutableRawPointer, length: Int) throws -> Int {
        if isClosed {
            throw QWDispatchIOError.channelClosed
        }
        var totalRead = 0
        var ptr = dst

        while totalRead < length {
            let toRead = length - totalRead
            let bytesRead = pread(fd, ptr, toRead, off_t(offset + Int64(totalRead)))
            if bytesRead < 0 {
                if errno == EINTR { continue }
                throw QWDispatchIOError.readFailed(offset: offset + Int64(totalRead), length: toRead, errno: errno)
            }
            if bytesRead == 0 { break } // EOF
            totalRead += bytesRead
            ptr = ptr.advanced(by: bytesRead)
        }
        return totalRead
    }

    /// Synchronous write fallback using POSIX `pwrite` with an `EINTR` retry loop.
    public func writeSync(offset: Int64, buffer: UnsafeRawPointer, length: Int) throws {
        try checkOpen()
        var totalWritten = 0
        var ptr = buffer

        while totalWritten < length {
            let toWrite = length - totalWritten
            let bytesWritten = pwrite(fd, ptr, toWrite, off_t(offset + Int64(totalWritten)))
            if bytesWritten < 0 {
                if errno == EINTR { continue }
                throw QWDispatchIOError.writeFailed(offset: offset + Int64(totalWritten), length: toWrite, errno: errno)
            }
            totalWritten += bytesWritten
            ptr = ptr.advanced(by: bytesWritten)
        }
    }
}
