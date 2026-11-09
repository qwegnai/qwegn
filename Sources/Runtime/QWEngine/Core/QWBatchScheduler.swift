// QWBatchScheduler.swift — Unified Continuous Batching Scheduler for QWEngine.
// Coordinates single-stream and multi-session continuous batching across all
// model architectures, coalescing concurrent request bursts cleanly.

import Foundation
import QWCore
import QWFlashNext
import QWHead

/// Per-slot recurrent/conv/PLE state for multi-session continuous batching.
public typealias QWBatchSlotState = FNBatchSlotState

/// Output metrics of one completed batch generation job.
public typealias QWBatchJobOutput = FNBatchJobOutput

/// One generation request submitted to the GPU batch scheduler.
public typealias QWBatchJob = FNBatchJob

/// Tracks requests currently rendering chat templates or tokenizing so concurrent bursts coalesce cleanly.
public typealias QWBatchAdmissionTracker = FNBatchAdmissionTracker

/// 4-slot GPU continuous batching scheduler for multi-session execution.
public typealias QWGPUBatchScheduler = FNGPUBatchScheduler

/// Unified request scheduler over any `QWInferenceEngine`.
public final class QWBatchScheduler: @unchecked Sendable {
    public static let maxSlots = QWGPUBatchScheduler.maxSlots

    private let underlying: any QWInferenceEngine
    private let lock = NSLock()
    private var activeCount: Int = 0
    private var waitingCount: Int = 0

    /// Optional callback invoked after each decode step with `(gpuMs, wallMs)`.
    public var onStepTiming: (@Sendable (Double, Double) -> Void)?

    public init(engine: any QWInferenceEngine) {
        self.underlying = engine
    }

    /// Number of currently executing or queued generation requests on this scheduler.
    public var inFlightRequests: Int {
        lock.lock()
        defer { lock.unlock() }
        return activeCount + waitingCount
    }

    /// Underlying 4-slot GPU batch scheduler when the engine supports multi-slot hardware coalescing.
    public var gpuBatchScheduler: QWGPUBatchScheduler? {
        if let flashModule = underlying as? FlashNextModule {
            return flashModule.engine?.batchScheduler
        }
        return nil
    }

    /// Runs `request` through the unified batch scheduler, coalescing concurrent bursts when supported.
    public func generate(
        request: QWGenerationRequest,
        onToken: ((Int) -> Bool)? = nil
    ) throws -> QWGenerationResult {
        lock.lock()
        activeCount += 1
        lock.unlock()
        defer {
            lock.lock()
            activeCount = max(0, activeCount - 1)
            lock.unlock()
        }
        return try underlying.generate(request: request, onToken: onToken)
    }
}
