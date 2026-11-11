// QWEngineTypes.swift — The types the engine protocol speaks.
//
// They live at module level, not inside a concrete engine: a consumer
// that holds `any QWInferenceEngine` names these, and never a family.

import Foundation
import QWCore
import QWHead

/// What the prefix cache did on the last turn. The tier is the one the
/// cache served from; absence means it served from none.
public struct QWPrefixCacheStats: Sendable {
    public let hit: Bool
    public let matchedTokens: Int
    public let chargedBytes: Int
    public let evictions: Int
    public let skippedOversize: Bool
    public let tier: String?

    public init(
        hit: Bool,
        matchedTokens: Int,
        chargedBytes: Int,
        evictions: Int,
        skippedOversize: Bool,
        tier: String? = nil
    ) {
        self.hit = hit
        self.matchedTokens = matchedTokens
        self.chargedBytes = chargedBytes
        self.evictions = evictions
        self.skippedOversize = skippedOversize
        self.tier = tier
    }
}

/// What one generation cost. The seconds are what the engine measured.
public struct QWGenerationStats: Sendable {
    public var promptTokens: Int
    public var generatedTokens: Int
    public var prefillSec: Double       // streamed prefill wall (all prompt steps)
    public var decodeSec: Double        // generation-loop wall
    public var firstTokenSec: Double    // TTFT = prefill wall (first gen id ready)
    public var stepMs: [Double]         // per decode-step latency (post-sync)
}

/// What one speculative generation cost: the rounds it took and what
/// each of them produced.
public struct QWExperimentalGenerationStats: Sendable {
    public let promptTokens: Int
    public let generatedTokens: Int
    public let prefillSec: Double
    public let decodeSec: Double
    public let roundMs: [Double]
    public let roundTokenCounts: [Int]

    public var prefillTokensPerSecond: Double {
        guard prefillSec > 0, prefillSec.isFinite else { return 0 }
        return Double(promptTokens) / prefillSec
    }

    public var decodeTokensPerSecond: Double {
        guard decodeSec > 0, decodeSec.isFinite else { return 0 }
        return Double(roundTokenCounts.reduce(0, +)) / decodeSec
    }
}

/// The cache an engine holds right now. A byte count nobody measured is
/// absent: a cache that was never read is not a cache of zero bytes.
public struct QWPrefixCacheState: Sendable, Equatable {
    /// Whether a prefix cache is configured at all.
    public var isConfigured: Bool
    /// The bytes charged against the warm limit.
    public var warmBytes: Int?
    /// The bytes charged against the cold limit.
    public var ssdBytes: Int?
    /// The entries the warm tier holds. Absent when the cache counts none.
    public var warmEntries: Int?
    /// The entries the cold tier holds. Absent when the cache counts none.
    public var ssdEntries: Int?
    /// Admitted warm RAM byte cap after unified-memory admission clamping.
    public var warmLimitBytes: Int?

    public init(isConfigured: Bool, warmBytes: Int? = nil, ssdBytes: Int? = nil,
                warmEntries: Int? = nil, ssdEntries: Int? = nil, warmLimitBytes: Int? = nil) {
        self.isConfigured = isConfigured
        self.warmBytes = warmBytes
        self.ssdBytes = ssdBytes
        self.warmEntries = warmEntries
        self.ssdEntries = ssdEntries
        self.warmLimitBytes = warmLimitBytes
    }
}

/// A generation, described the way every engine understands it.
public struct QWGenerationRequest: Sendable {
    public var prompt: [Int]
    public var maxNew: Int
    public var stopIDs: Set<Int>
    public var sampling: QWSamplerConfig
    /// Whether the caller wants the prefix cache used. The dense engine
    /// honours the request; the Flash-Next family's cache is its own — it
    /// serves the request whenever it is configured.
    public var usePrefixCache: Bool
    /// The speculation contract: whether the caller wants a drafter, and
    /// which one. The engine decides whether it can serve it.
    public var speculative: QWSpeculativeParameters
    /// The prefix length the caller wants kept after the turn.
    public var retainPrefixLength: Int?
    /// Whether the caller wants the detailed per-stage timings. The
    /// engine measures them only when they are asked for.
    public var collectTimings: Bool
    /// Whether per-token log-probabilities should be collected from the LM head.
    public var collectLogprobs: Bool
    /// Number of top candidate logprobs (`0..20`) to collect per token when `collectLogprobs` is true.
    public var topLogprobsCount: Int
    /// Optional per-token logprob callback invoked immediately before `onToken`.
    public var onTokenLogprob: (@Sendable (QWTokenLogprob) -> Void)?
    /// Request-local token constraint (structured output). Only engines whose
    /// `supportsTokenConstraints` is true may be handed one.
    public var constraint: (any QWTokenConstraint)?
    /// Merged visual feature spans to inject at `<|image|>` / `<|video|>` positions during prefill.
    public var visualSpans: [QWVisualSpan]

    public init(
        prompt: [Int],
        maxNew: Int,
        stopIDs: Set<Int> = [],
        sampling: QWSamplerConfig,
        usePrefixCache: Bool = false,
        speculative: QWSpeculativeParameters = QWSpeculativeParameters(),
        retainPrefixLength: Int? = nil,
        collectTimings: Bool = false,
        collectLogprobs: Bool = false,
        topLogprobsCount: Int = 0,
        onTokenLogprob: (@Sendable (QWTokenLogprob) -> Void)? = nil,
        constraint: (any QWTokenConstraint)? = nil,
        visualSpans: [QWVisualSpan] = []
    ) {
        self.constraint = constraint
        self.prompt = prompt
        self.maxNew = maxNew
        self.stopIDs = stopIDs
        self.sampling = sampling
        self.usePrefixCache = usePrefixCache
        self.speculative = speculative
        self.retainPrefixLength = retainPrefixLength
        self.collectTimings = collectTimings
        self.collectLogprobs = collectLogprobs
        self.topLogprobsCount = topLogprobsCount
        self.onTokenLogprob = onTokenLogprob
        self.visualSpans = visualSpans
    }
}
