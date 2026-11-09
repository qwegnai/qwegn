import Metal

enum QWMTPVerificationStage: String, CaseIterable, Sendable {
    case norm
    case inputProjection
    case gdn
    case attention
    case outputProjection
    case mlpGateUp
    case mlpDown
    case head
}

struct QWMTPStageTiming: Codable, Equatable {
    let hostMs: Double
    let gpuMs: Double
    let commandBuffers: Int

    init(hostMs: Double, gpuMs: Double, commandBuffers: Int) throws {
        guard hostMs.isFinite, gpuMs.isFinite, hostMs > 0, gpuMs > 0,
              hostMs >= gpuMs, commandBuffers > 0 else {
            throw QWEngineDriverError.state("invalid MTP stage timing")
        }
        self.hostMs = hostMs
        self.gpuMs = gpuMs
        self.commandBuffers = commandBuffers
    }

    static func gpuDurationMs(start: Double, end: Double) throws -> Double {
        guard start.isFinite, end.isFinite, start > 0, end > start else {
            throw QWEngineDriverError.state("MTP profiling requires valid completed GPU timestamps")
        }
        return (end - start) * 1000
    }

    static func gpuDurationMs(_ buffer: any MTLCommandBuffer) throws -> Double {
        guard buffer.status == .completed else {
            throw QWEngineDriverError.state("MTP profiling requires a completed command buffer")
        }
        return try gpuDurationMs(start: buffer.gpuStartTime, end: buffer.gpuEndTime)
    }
}

struct QWMTPRoundProfile: Codable, Equatable {
    let drafts: Int
    let retainedRows: Int
    let draft: QWMTPStageTiming
    let verification: QWMTPStageTiming
    let resolution: QWMTPStageTiming
    var draftTokens: [Int]? = nil
    var acceptedDrafts: Int? = nil
    /// Backend identity of the round (LighteningMTP L0, plan §7.L0.3).
    /// Optional so pre-backend reports still decode; the shipping driver
    /// always records it, and the analyzer refuses mixed identity.
    var backend: String? = nil
    var verificationGpuMsByStage: [String: Double]? = nil
    var adaptiveEntropyByDraft: [Double]? = nil

    var hostMs: Double { draft.hostMs + verification.hostMs + resolution.hostMs }
    var gpuMs: Double { draft.gpuMs + verification.gpuMs + resolution.gpuMs }
}

final class QWMTPStageAccumulator {
    private(set) var gpuMs = 0.0
    private(set) var commandBuffers = 0

    func record(_ buffer: any MTLCommandBuffer) throws {
        gpuMs += try QWMTPStageTiming.gpuDurationMs(buffer)
        commandBuffers += 1
    }
}
