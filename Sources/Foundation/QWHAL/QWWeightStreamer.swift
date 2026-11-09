// QWWeightStreamer.swift — Asynchronous unbuffered direct DMA weight streamer via QWDispatchIO
import Foundation
import Dispatch

public enum QWWeightStreamerError: Error, CustomStringConvertible {
    case fileNotFound(String)
    case invalidRange(offset: Int, count: Int, fileLength: Int)
    case readFailed(String)

    public var description: String {
        switch self {
        case .fileNotFound(let path):
            return "QWWeightStreamer: file not found at '\(path)'"
        case .invalidRange(let off, let cnt, let len):
            return "QWWeightStreamer: invalid range [\(off), \(off + cnt)) in file of length \(len)"
        case .readFailed(let msg):
            return "QWWeightStreamer: read failed: \(msg)"
        }
    }
}

public final class QWWeightStreamer: Sendable {
    public let url: URL
    public let dispatchIO: QWDispatchIO
    public let fileLength: Int

    public init(url: URL, unbuffered: Bool = true) throws {
        self.url = url
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw QWWeightStreamerError.fileNotFound(url.path)
        }
        let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
        guard let size = attrs[.size] as? NSNumber else {
            throw QWWeightStreamerError.readFailed("Unable to determine file size")
        }
        self.fileLength = size.intValue
        self.dispatchIO = try QWDispatchIO(path: url.path, mode: .readOnly, directIO: unbuffered)
    }

    public func streamTensor(offset: Int, byteCount: Int, destination: UnsafeMutableRawPointer) throws {
        guard offset >= 0, byteCount >= 0, offset + byteCount <= fileLength else {
            throw QWWeightStreamerError.invalidRange(offset: offset, count: byteCount, fileLength: fileLength)
        }
        _ = try dispatchIO.readSync(offset: Int64(offset), into: destination, length: byteCount)
    }

    public func streamTensorAsync(offset: Int, byteCount: Int) async throws -> DispatchData {
        guard offset >= 0, byteCount >= 0, offset + byteCount <= fileLength else {
            throw QWWeightStreamerError.invalidRange(offset: offset, count: byteCount, fileLength: fileLength)
        }
        return try await dispatchIO.read(offset: Int64(offset), length: byteCount)
    }
}
