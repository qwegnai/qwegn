import Foundation
import Metal
import QuartzCore

/// Opt-in command-buffer attribution without changing encoders, barriers, or submission cuts.
final class FNGPUExecutionTrace {
    static let enabled = ProcessInfo.processInfo.environment["QW_FLASH_TRACE"] == "1"

    struct BufferTiming: Encodable {
        let layers: [Int]
        let encodeStartMS: Double
        let submitMS: Double
        let gpuStartMS: Double
        let gpuEndMS: Double

        var encodeMS: Double { submitMS - encodeStartMS }
        var submitToStartMS: Double { gpuStartMS - submitMS }
        var gpuMS: Double { gpuEndMS - gpuStartMS }
    }

    struct Report: Encodable {
        let schema = 1
        let phase: String
        let position: Int
        let rows: Int
        let pleRoute: String
        let mappedPLEBytes: Int
        let includesMTP: Bool
        let plePrepareMS: Double
        let pleWaitMS: Double
        let nextPLEPrepareMS: Double
        let completionWaitMS: Double
        let stackToGPUCompleteMS: Double
        let buffers: [BufferTiming]

        var encodeMS: Double { buffers.reduce(0) { $0 + $1.encodeMS } - pleWaitMS }
        var gpuMS: Double { buffers.reduce(0) { $0 + $1.gpuMS } }
        var gpuSpanMS: Double { (buffers.last?.gpuEndMS ?? 0) - (buffers.first?.gpuStartMS ?? 0) }
        var gpuIdleBetweenBuffersMS: Double {
            zip(buffers, buffers.dropFirst()).reduce(0) { $0 + max(0, $1.1.gpuStartMS - $1.0.gpuEndMS) }
        }
    }

    enum TraceError: Error, CustomStringConvertible {
        case invalidCommandBufferTimestamps

        var description: String { "Flash-Next trace: missing or invalid completed command-buffer timestamps" }
    }

    private struct Submission {
        let buffer: any MTLCommandBuffer
        let lower: Int
        let upper: Int
        let encodeStart: Double
        let submit: Double
    }

    private let started = CACurrentMediaTime()
    private var prepared = 0.0
    private var pleWait = 0.0
    private var completionWait = 0.0
    private var nextPLEPrepare = 0.0
    private var submissions: [Submission] = []

    var clock: Double { CACurrentMediaTime() }

    func preparedPLE() { prepared = clock - started }

    func waitedForPLE(since start: Double?) {
        if let start { pleWait += clock - start }
    }

    func waitedForCompletion(since start: Double?) {
        if let start { completionWait += clock - start }
    }

    func preparedNextPLE(since start: Double?) {
        if let start { nextPLEPrepare += clock - start }
    }

    func submitting(_ buffer: any MTLCommandBuffer, lower: Int, upper: Int, encodeStart: Double?) {
        guard let encodeStart else { return }
        submissions.append(Submission(buffer: buffer, lower: lower, upper: upper,
                                      encodeStart: encodeStart, submit: clock))
    }

    func report(phase: String, position: Int, rows: Int, pleRoute: String,
                mappedPLEBytes: Int, includesMTP: Bool) throws -> Report {
        let finished = clock
        let buffers = try submissions.map { entry -> BufferTiming in
            let start = entry.buffer.gpuStartTime, end = entry.buffer.gpuEndTime
            guard start.isFinite, end.isFinite, start > 0, end >= start,
                  start >= entry.submit else {
                throw TraceError.invalidCommandBufferTimestamps
            }
            return BufferTiming(layers: [entry.lower, entry.upper],
                                encodeStartMS: (entry.encodeStart - started) * 1000,
                                submitMS: (entry.submit - started) * 1000,
                                gpuStartMS: (start - started) * 1000,
                                gpuEndMS: (end - started) * 1000)
        }
        return Report(phase: phase, position: position, rows: rows, pleRoute: pleRoute,
                      mappedPLEBytes: mappedPLEBytes, includesMTP: includesMTP,
                      plePrepareMS: prepared * 1000, pleWaitMS: pleWait * 1000,
                      nextPLEPrepareMS: nextPLEPrepare * 1000,
                      completionWaitMS: completionWait * 1000,
                      stackToGPUCompleteMS: (finished - started) * 1000, buffers: buffers)
    }

    func emit(phase: String, position: Int, rows: Int, pleRoute: String,
              mappedPLEBytes: Int, includesMTP: Bool) throws {
        let result = try report(phase: phase, position: position, rows: rows, pleRoute: pleRoute,
                                mappedPLEBytes: mappedPLEBytes, includesMTP: includesMTP)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        var data = Data("[flash-execution] ".utf8)
        data.append(try encoder.encode(result))
        data.append(0x0A)
        try FileHandle.standardError.write(contentsOf: data)
    }
}
