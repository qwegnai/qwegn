// QWEngineReport.swift — What an engine says about itself.
//
// A consumer prints these numbers without knowing which engine it got.
// The engine composes its own lines: the frame belongs to the
// consumer, the story belongs to the engine.

import Foundation

/// What the prefix cache did for one turn. Absent when no cache
/// answered: a cache that was never asked is not a cache of zero.
public struct QWCacheAnswer: Sendable, Equatable {
    /// Whether the cache served the prompt.
    public var hit: Bool
    /// How many tokens the cache reused.
    public var matchedTokens: Int
    /// How many tokens the prompt had.
    public var totalTokens: Int
    /// The tier that served the prompt, when it names one.
    public var tier: String?

    public init(hit: Bool, matchedTokens: Int, totalTokens: Int, tier: String? = nil) {
        self.hit = hit
        self.matchedTokens = matchedTokens
        self.totalTokens = totalTokens
        self.tier = tier
    }
}

/// What an engine reports about itself. The numbers here are measured;
/// the lines are the engine's own story, composed by the engine.
public struct QWEngineReport: Sendable {
    /// The engine's one-line capacity statement.
    public var capacityLog: String
    /// The bytes the arena reserved.
    public var arenaBytes: Int
    /// The bytes the weights occupy.
    public var weightUsedBytes: Int
    /// The bytes reserved for the weights.
    public var weightReservedBytes: Int
    /// The steady-state allocations the engine has measured in the arena
    /// it keeps its working set in. Absent when the engine does not
    /// measure them: unknown is not zero.
    public var steadyAllocations: Int?
    /// The seconds this process spent creating pipelines. It is the
    /// process's total, not the engine's own: a report cannot claim
    /// more precision than the measurement has.
    public var pipelineSeconds: Double?
    /// The engine's own lines: what only this family has to say.
    public var detailLines: String
    /// The context the engine holds. Absent when the engine does not
    /// report one: the requested context is not the held context.
    public var contextCapacity: Int?
    /// The container the engine holds its weights in, as its own name.
    /// Absent when the engine does not report one: a container that was
    /// guessed from the path is a claim about nothing.
    public var weightFormat: String?
    /// Why the size-preserving family was declined, when it was. Absent when
    /// the model's own bytes are the container, and when the engine does not
    /// report one.
    public var weightFormatDeclineReason: String?

    /// The detail lines, ending with a newline when there are any.
    public var detailText: String {
        guard !detailLines.isEmpty else { return "" }
        return detailLines.hasSuffix("\n") ? detailLines : detailLines + "\n"
    }

    public init(capacityLog: String, arenaBytes: Int, weightUsedBytes: Int,
                weightReservedBytes: Int, steadyAllocations: Int? = nil,
                pipelineSeconds: Double? = nil, detailLines: String = "",
                contextCapacity: Int? = nil, weightFormat: String? = nil,
                weightFormatDeclineReason: String? = nil) {
        self.capacityLog = capacityLog
        self.arenaBytes = arenaBytes
        self.weightUsedBytes = weightUsedBytes
        self.weightReservedBytes = weightReservedBytes
        self.steadyAllocations = steadyAllocations
        self.pipelineSeconds = pipelineSeconds
        self.detailLines = detailLines
        self.contextCapacity = contextCapacity
        self.weightFormat = weightFormat
        self.weightFormatDeclineReason = weightFormatDeclineReason
    }
}
