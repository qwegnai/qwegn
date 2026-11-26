// QWDecisionExecutionBackend.swift — Zero-Metal backend seam for Metal 4, CoreAI ANE, and test adapters.
import Foundation
import QWCore

public struct QWDecisionSequenceInput: Sendable, Hashable, Equatable {
    public let tokenIds: [Int]
    public let topology: QWDecisionModelConfig.AttentionTopology
    public let stateEndPosition: Int
    public let questionSpans: [Range<Int>]
    public let markerPositions: [Int]
    public let pairIndex: Int?
    public let chain: String
    public let images: [Data]
    public let pointerValidMask: [Bool]

    public init(
        tokenIds: [Int],
        topology: QWDecisionModelConfig.AttentionTopology,
        stateEndPosition: Int = 0,
        questionSpans: [Range<Int>] = [],
        markerPositions: [Int] = [],
        pairIndex: Int? = nil,
        chain: String = "default",
        images: [Data] = [],
        pointerValidMask: [Bool] = []
    ) {
        self.tokenIds = tokenIds
        self.topology = topology
        self.stateEndPosition = stateEndPosition
        self.questionSpans = questionSpans
        self.markerPositions = markerPositions
        self.pairIndex = pairIndex
        self.chain = chain
        self.images = images
        self.pointerValidMask = pointerValidMask
    }
}

public struct QWDecisionSlotTarget: Sendable, Hashable, Equatable {
    public let sequenceIndex: Int
    public let tokenPosition: Int
    public let candidateTokenGroups: [[Int]]
    public let isPointerSlot: Bool
    public let pointerEndPosition: Int?

    public init(
        sequenceIndex: Int,
        tokenPosition: Int,
        candidateTokenGroups: [[Int]] = [],
        isPointerSlot: Bool = false,
        pointerEndPosition: Int? = nil
    ) {
        self.sequenceIndex = sequenceIndex
        self.tokenPosition = tokenPosition
        self.candidateTokenGroups = candidateTokenGroups
        self.isPointerSlot = isPointerSlot
        self.pointerEndPosition = pointerEndPosition
    }
}

public struct QWDecisionBackendBatchRequest: Sendable, Hashable, Equatable {
    public let sequences: [QWDecisionSequenceInput]
    public let slots: [QWDecisionSlotTarget]
    public let uniqueCandidateTokenIds: [Int]

    public init(
        sequences: [QWDecisionSequenceInput],
        slots: [QWDecisionSlotTarget],
        uniqueCandidateTokenIds: [Int] = []
    ) {
        self.sequences = sequences
        self.slots = slots
        self.uniqueCandidateTokenIds = uniqueCandidateTokenIds
    }
}

public struct QWDecisionSlotOutput: Sendable, Equatable {
    public let tokenLogits: [Int: Float]
    public let fullLogits: [Float]?
    public let markerLogits: [Float]?
    public let pooledLogits: [Float]?
    public let pointerStartLogits: [Float]?
    public let pointerEndLogits: [Float]?

    public init(
        tokenLogits: [Int: Float] = [:],
        fullLogits: [Float]? = nil,
        markerLogits: [Float]? = nil,
        pooledLogits: [Float]? = nil,
        pointerStartLogits: [Float]? = nil,
        pointerEndLogits: [Float]? = nil
    ) {
        self.tokenLogits = tokenLogits
        self.fullLogits = fullLogits
        self.markerLogits = markerLogits
        self.pooledLogits = pooledLogits
        self.pointerStartLogits = pointerStartLogits
        self.pointerEndLogits = pointerEndLogits
    }

    public func groupScores(for candidateGroups: [[Int]]) -> [Double] {
        candidateGroups.map { group in
            var best = -Double.infinity
            for tid in group {
                let val: Float? = tokenLogits[tid] ?? fullLogits.flatMap { tid >= 0 && tid < $0.count ? $0[tid] : nil }
                if let val {
                    best = max(best, Double(val))
                }
            }
            return best.isFinite ? best : -1e4
        }
    }
}

public struct QWDecisionBackendOutput: Sendable, Equatable {
    public let slotOutputs: [QWDecisionSlotOutput]
    public let promptTokens: Int
    public let latencyMs: Double

    public init(slotOutputs: [QWDecisionSlotOutput], promptTokens: Int, latencyMs: Double = 0) {
        self.slotOutputs = slotOutputs
        self.promptTokens = promptTokens
        self.latencyMs = latencyMs
    }
}

/// Abstract hardware/model backend executing single-pass decision batches (`ADR-004`).
public protocol QWDecisionExecutionBackend: Sendable {
    var decisionConfig: QWDecisionModelConfig { get }
    var maxContextLength: Int { get }
    func evaluate(_ request: QWDecisionBackendBatchRequest) async throws -> QWDecisionBackendOutput
}

/// Pure-host dense embedding backend seam sharing the decision engine's Metal 4 backbone.
public protocol QWEmbeddingExecutionBackend: Sendable {
    var embeddingDimension: Int { get }
    var maxContextLength: Int { get }
    func embedBatch(
        tokenBatches: [[Int]],
        pooling: QWDecisionModelConfig.PoolingKind,
        normalize: Bool,
        truncateDimensions: Int?
    ) async throws -> (embeddings: [[Float]], promptTokens: Int, latencyMs: Double)
}

/// Shared Matryoshka truncation, L2 normalization, and IEEE-754 base64 helpers.
public enum QWEmbeddingMath {
    public static func postProcess(
        _ vector: [Float],
        normalize: Bool,
        truncateDimensions: Int?
    ) -> [Float] {
        var out: [Float]
        if let dim = truncateDimensions, dim > 0, dim < vector.count {
            out = Array(vector.prefix(dim))
        } else {
            out = vector
        }
        guard normalize, !out.isEmpty else { return out }
        var sumSq: Double = 0
        for v in out {
            let d = Double(v)
            sumSq += d * d
        }
        let norm = sumSq.squareRoot()
        guard norm > 1e-12 else { return out }
        let inv = Float(1.0 / norm)
        for i in out.indices {
            out[i] *= inv
        }
        return out
    }

    public static func encodeBase64LittleEndian(_ vector: [Float]) -> String {
        var data = Data(capacity: vector.count * 4)
        for v in vector {
            var le = v.bitPattern.littleEndian
            withUnsafeBytes(of: &le) { (raw: UnsafeRawBufferPointer) in
                if let base = raw.baseAddress {
                    data.append(base.assumingMemoryBound(to: UInt8.self), count: 4)
                }
            }
        }
        return data.base64EncodedString()
    }
}
