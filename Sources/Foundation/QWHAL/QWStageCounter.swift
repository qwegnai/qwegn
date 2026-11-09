// QWStageCounter.swift — Metal stage-boundary GPU timestamp profiler (A1).
// Hoisted from the dense MTP verifier so BOTH engine classes attribute GPU time
// per encoder region without host drains (per-door drains inflate ~5x; this does
// not). Device law: one MTLCounterSampleBuffer caps at 32 KiB = 4096 timestamp
// samples (8 B each), so segments rotate across a small ring of buffers.

import Foundation
import Metal
import QWMetalCounterSupport

public struct QWGPUStageProfileResult<Stage: Hashable & Sendable>: Sendable {
    public let gpuMicrosByStage: [Stage: Double]
    public let attributedGpuMicros: Double
    public let totalGpuMicros: Double
}

/// Max timestamp samples per MTLCounterSampleBuffer (32 KiB / 8 B).
private let kQWSamplesPerBuffer = 4096

public final class QWGPUStageProfiler<Stage: Hashable & Sendable> {
    private struct Segment {
        let stage: Stage
        let buffer: Int
        let localSample: Int
    }

    private struct BufferWindow {
        let buffer: any MTLCounterSampleBuffer
        var usedSamples: Int = 0
    }

    private var windows: [BufferWindow] = []
    private let segsPerBuffer: Int
    private let capacity: Int
    private var segments: [Segment] = []

    public enum ProfileError: Error, CustomStringConvertible {
        case unsupportedSampling
        case timestampCounterUnavailable
        case sampleBufferCreation(String)
        case encoderCreation
        case capacityExceeded
        case sampleResolution
        case invalidTimestamp

        public var description: String {
            switch self {
            case .unsupportedSampling: return "QWHAL: compute stage-boundary counter sampling is unsupported"
            case .timestampCounterUnavailable: return "QWHAL: Metal timestamp counter set is unavailable"
            case .sampleBufferCreation(let message): return "QWHAL: counter sample buffer creation failed: \(message)"
            case .encoderCreation: return "QWHAL: profiled compute encoder creation failed"
            case .capacityExceeded: return "QWHAL: GPU stage profile encoder capacity exceeded"
            case .sampleResolution: return "QWHAL: Metal counter samples could not be resolved"
            case .invalidTimestamp: return "QWHAL: Metal returned invalid timestamp samples"
            }
        }
    }

    /// Device capability probe — false on hardware without stage sampling.
    public static func isSupported(on device: any MTLDevice) -> Bool {
        device.supportsCounterSampling(.atStageBoundary)
            && (device.counterSets?.contains { $0.name == "timestamp" } ?? false)
    }

    public init(device: any MTLDevice, segmentCapacity: Int) throws {
        guard device.supportsCounterSampling(.atStageBoundary) else {
            throw ProfileError.unsupportedSampling
        }
        guard let counterSet = device.counterSets?.first(where: { $0.name == "timestamp" }) else {
            throw ProfileError.timestampCounterUnavailable
        }
        segsPerBuffer = kQWSamplesPerBuffer / 2
        capacity = segmentCapacity
        var made: [BufferWindow] = []
        for _ in 0..<max(1, (segmentCapacity + segsPerBuffer - 1) / segsPerBuffer) {
            let descriptor = MTLCounterSampleBufferDescriptor()
            descriptor.counterSet = counterSet
            descriptor.storageMode = .shared
            descriptor.sampleCount = kQWSamplesPerBuffer
            do {
                made.append(BufferWindow(buffer: try device.makeCounterSampleBuffer(descriptor: descriptor)))
            } catch {
                throw ProfileError.sampleBufferCreation(error.localizedDescription)
            }
        }
        windows = made
        segments.reserveCapacity(segmentCapacity)
    }

    /// False once the segment budget is spent; callers fall back to plain doors.
    public var hasSegmentCapacity: Bool { segments.count < capacity }
    public var segmentCount: Int { segments.count }

    /// Encoder with start/end timestamp samples booked under `stage`.
    public func makeEncoder(commandBuffer: any MTLCommandBuffer,
                           stage: Stage) throws -> any MTLComputeCommandEncoder {
        guard hasSegmentCapacity else { throw ProfileError.capacityExceeded }
        let global = segments.count
        let windowIndex = global / segsPerBuffer
        let local = (global % segsPerBuffer) * 2
        let pass = MTLComputePassDescriptor()
        pass.dispatchType = .serial
        let attachment = MTLComputePassSampleBufferAttachmentDescriptor()
        attachment.sampleBuffer = windows[windowIndex].buffer
        attachment.startOfEncoderSampleIndex = local
        attachment.endOfEncoderSampleIndex = local + 1
        pass.sampleBufferAttachments[0] = attachment
        guard let encoder = commandBuffer.makeComputeCommandEncoder(descriptor: pass) else {
            throw ProfileError.encoderCreation
        }
        windows[windowIndex].usedSamples = local + 2
        segments.append(Segment(stage: stage, buffer: windowIndex, localSample: local))
        return encoder
    }

    /// GPU microseconds summed per stage. Command buffer must have completed.
    public func resolveByStage() throws -> [(stage: Stage, micros: Double)] {
        let ticks = try resolveTicksByStage()
        return ticks.map { (stage: $0.0, micros: Double($0.1) / 1000) }
    }

    /// Calibrated view kept for the dense MTP verifier call site.
    public func resolve(totalGpuMicros: Double) throws -> QWGPUStageProfileResult<Stage> {
        let pairs = try resolveByStage()
        var byStage: [Stage: Double] = [:]
        var attributed = 0.0
        for pair in pairs {
            byStage[pair.stage, default: 0] += pair.micros
            attributed += pair.micros
        }
        return QWGPUStageProfileResult(gpuMicrosByStage: byStage,
                                       attributedGpuMicros: attributed,
                                       totalGpuMicros: totalGpuMicros)
    }

    /// Clear booked segments so sample slots restart for the next command buffer.
    public func reset() {
        segments.removeAll()
        for index in windows.indices { windows[index].usedSamples = 0 }
    }

    private func timestamps(in window: BufferWindow) throws -> [UInt64] {
        guard window.usedSamples > 0 else { return [] }
        guard let resolved = QWResolveMetalCounterSamples(
            window.buffer, NSRange(location: 0, length: window.usedSamples)) else {
            throw ProfileError.sampleResolution
        }
        let stride = MemoryLayout<MTLCounterResultTimestamp>.stride
        guard resolved.count >= window.usedSamples * stride else { throw ProfileError.sampleResolution }
        return resolved.withUnsafeBytes { raw in
            (0..<window.usedSamples).map { index in
                raw.loadUnaligned(fromByteOffset: index * stride,
                                  as: MTLCounterResultTimestamp.self).timestamp
            }
        }
    }

    private func resolveTicksByStage() throws -> [(Stage, UInt64)] {
        guard !segments.isEmpty else { return [] }
        var stampArrays: [[UInt64]] = []
        for window in windows { stampArrays.append(try timestamps(in: window)) }
        var ticksByStage: [Stage: UInt64] = [:]
        var order: [Stage] = []
        for segment in segments {
            let stamps = stampArrays[segment.buffer]
            let start = stamps[segment.localSample]
            let end = stamps[segment.localSample + 1]
            guard start != MTLCounterErrorValue, end != MTLCounterErrorValue, end >= start else {
                throw ProfileError.invalidTimestamp
            }
            if ticksByStage[segment.stage] == nil { order.append(segment.stage) }
            ticksByStage[segment.stage, default: 0] += end - start
        }
        // Metal's GPUTimestamp counter is in nanoseconds.
        return order.map { stage in (stage, ticksByStage[stage]!) }
    }
}
