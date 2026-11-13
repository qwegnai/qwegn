// QWInferenceParameters.swift — Cohesive value structs for sampling and speculative decoding parameters.
// Centralizes greedy detection (`temperature <= 0 || topK == 1`) and Dense NAX kernel selection logging.

import Foundation

/// Dense NAX kernel modes for row-batched HQ4 projections (`T >= 64`).
public enum QWDenseNaxMode: String, Sendable {
    case singlePass = "single_pass"
}

/// Cohesive sampling hyperparameters shared across CLI, configuration resolver, server settings, and lifecycle actor.
public struct QWSamplingParameters: Codable, Sendable, Equatable {
    public var temperature: Double
    public var topP: Double
    public var topK: Int
    public var maxTokens: Int
    public var repetitionPenalty: Double
    public var seed: Int

    public init(
        temperature: Double = 0.7,
        topP: Double = 0.9,
        topK: Int = 40,
        maxTokens: Int = 512,
        repetitionPenalty: Double = 1.05,
        seed: Int = -1
    ) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.maxTokens = maxTokens
        self.repetitionPenalty = repetitionPenalty
        self.seed = seed
    }

    /// True when the hyperparameters specify deterministic greedy argmax decoding.
    public var isGreedy: Bool {
        temperature <= 0.0 || topK == 1
    }

    /// True when the caller supplied an explicit non-negative RNG seed.
    public var hasExplicitSeed: Bool {
        seed >= 0
    }

    /// True when greedy decoding is requested alongside a non-identity repetition penalty.
    public var isGreedyWithRepetitionPenalty: Bool {
        isGreedy && repetitionPenalty != 1.0
    }

    /// Returns a copy with per-request overrides applied.
    public func merging(
        temperature: Double? = nil,
        topP: Double? = nil,
        topK: Int? = nil,
        maxTokens: Int? = nil,
        repetitionPenalty: Double? = nil,
        seed: Int? = nil
    ) -> QWSamplingParameters {
        let resolvedTemp = temperature ?? self.temperature
        let resolvedTopP = topP ?? self.topP
        // A near-zero top_p nucleus (<= 1e-5) or temperature <= 0 collapses sampling to deterministic greedy argmax.
        let topPForcesGreedy = topP != nil && resolvedTopP <= 1e-5
        let resolvedTopK = topPForcesGreedy ? 1 : (topK ?? self.topK)
        let isRequestOverrideGreedy = (temperature != nil && resolvedTemp <= 0.0)
            || (topK != nil && resolvedTopK == 1)
            || topPForcesGreedy
        return QWSamplingParameters(
            temperature: topPForcesGreedy ? 0.0 : resolvedTemp,
            topP: resolvedTopP,
            topK: resolvedTopK,
            maxTokens: maxTokens ?? self.maxTokens,
            repetitionPenalty: repetitionPenalty ?? (isRequestOverrideGreedy ? 1.0 : self.repetitionPenalty),
            seed: seed ?? self.seed
        )
    }

    /// Resolves the effective 64-bit RNG seed (generating a random seed when `seed < 0`).
    public func resolvedSeed() -> UInt64 {
        seed >= 0 ? UInt64(seed) : UInt64.random(in: 0..<UInt64.max)
    }

    /// Resolves the active Dense NAX kernel mode for these sampling parameters.
    /// Always uses `.singlePass` (`qwg_gemm_q4g_nax_f32`) across both stochastic and greedy hyperparameters.
    public func resolveDenseNaxMode() -> QWDenseNaxMode {
        .singlePass
    }

    /// Formats the diagnostic log message describing Dense NAX kernel selection.
    /// `paramsSource` attributes where the effective hyperparameters came from.
    /// `kernelName` is printed only when the caller KNOWS the kernel it runs.
    public func denseNaxSelectionLogMessage(paramsSource: String? = nil,
                                            kernelName: String? = nil) -> String {
        let sourceNote = paramsSource.map { " [params source: \($0)]" } ?? ""
        let kernelNote = kernelName.map { " (\($0))" } ?? ""
        return String(
            format: "[DenseNAX] Hyperparameters (temperature=%.2f, top_p=%.2f, top_k=%d, greedy=%@)%@: using default Single-Pass Dense NAX%@.",
            temperature, topP, topK, isGreedy ? "true" : "false", sourceNote, kernelNote
        )
    }
}

public enum QWDraftStrategy: String, Sendable {
    case hybrid
    case ngram
    case mtp

    public enum Error: Swift.Error, CustomStringConvertible {
        case invalidStrategy(String)

        public var description: String {
            switch self {
            case .invalidStrategy(let name): return "Unknown draft strategy '\(name)'"
            }
        }
    }

    public static func resolve(_ value: String) throws -> QWDraftStrategy {
        switch value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "hybrid": return .hybrid
        case "ngram": return .ngram
        case "mtp", "neural": return .mtp
        default: throw Error.invalidStrategy(value)
        }
    }
}

/// Cohesive speculative decoding parameters shared across server settings, configuration resolver, and lifecycle actor.
public struct QWSpeculativeParameters: Codable, Sendable, Equatable {
    public var enabled: Bool
    public var mtpDrafts: Int
    public var mtpBackend: String?
    public var mtpRelaxedMargin: Double
    /// Variational acceptance floor: a non-argmax draft also needs `p_target(draft) >= mtpRelaxedMinProb` (`0` disables it).
    public var mtpRelaxedMinProb: Double
    public var mtpDraftStrategy: String
    /// Set after the module has applied its environment and strategy policy.
    public var draftStrategyIsResolved: Bool?
    public var mtpQuadraticSeeding: Bool
    public var adaptiveDepth: Bool
    public var entropyThreshold: Double
    public var dflashDraftPackPath: String?
    public var dflashSelectorTopK: Int

    public init(
        enabled: Bool = true,
        mtpDrafts: Int = 1,
        mtpBackend: String? = "variational",
        mtpRelaxedMargin: Double = 1.5,
        mtpRelaxedMinProb: Double = QWSpeculativeParameters.defaultRelaxedMinProb,
        mtpDraftStrategy: String = "hybrid",
        mtpQuadraticSeeding: Bool = true,
        adaptiveDepth: Bool = false,
        entropyThreshold: Double = 0.7,
        dflashDraftPackPath: String? = nil,
        dflashSelectorTopK: Int = 16
    ) {
        self.enabled = enabled
        self.mtpDrafts = mtpDrafts
        self.mtpBackend = mtpBackend
        self.mtpRelaxedMargin = mtpRelaxedMargin
        self.mtpRelaxedMinProb = mtpRelaxedMinProb
        self.mtpDraftStrategy = mtpDraftStrategy
        self.draftStrategyIsResolved = nil
        self.mtpQuadraticSeeding = mtpQuadraticSeeding
        self.adaptiveDepth = adaptiveDepth
        self.entropyThreshold = entropyThreshold
        self.dflashDraftPackPath = dflashDraftPackPath
        self.dflashSelectorTopK = dflashSelectorTopK
    }

    /// Effective draft count (`0` when speculative decoding is disabled).
    public var effectiveDrafts: Int {
        enabled ? mtpDrafts : 0
    }

    /// Effective relaxed margin in nats (`0` unless `mtpBackend` is variational/relaxed/plm).
    public var effectiveRelaxedMargin: Float {
        guard enabled else { return 0.0 }
        let backend = (mtpBackend ?? "variational").trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        switch backend {
        case "variational", "relaxed", "plm":
            return Float(max(0.0, mtpRelaxedMargin))
        default:
            return 0.0
        }
    }

    /// Default variational acceptance floor (measured on the 100-question llmprobe code eval).
    public static let defaultRelaxedMinProb: Double = 0.2

    /// Effective relaxed acceptance floor in `[0, 1]` (`0` when relaxation is off).
    public var effectiveRelaxedMinProb: Float {
        guard effectiveRelaxedMargin > 0 else { return 0 }
        return Float(min(1.0, max(0.0, mtpRelaxedMinProb)))
    }

    /// Top-performance speculative decoding defaults for a given model descriptor.
    public static func topPerformanceDefaults(for descriptor: QWModelDescriptor? = nil) -> QWSpeculativeParameters {
        guard let descriptor else {
            return QWSpeculativeParameters(
                enabled: true,
                mtpDrafts: 3,
                mtpBackend: "variational",
                mtpRelaxedMargin: 1.5,
                mtpDraftStrategy: "hybrid",
                mtpQuadraticSeeding: true,
                adaptiveDepth: false,
                entropyThreshold: 0.7
            )
        }
        let hasMTP = descriptor.supportsNativeMTP
        let drafts = hasMTP ? min(3, descriptor.maxSpeculativeDrafts) : 0
        return QWSpeculativeParameters(
            enabled: hasMTP,
            mtpDrafts: drafts,
            mtpBackend: "variational",
            mtpRelaxedMargin: 1.5,
            mtpDraftStrategy: "hybrid",
            mtpQuadraticSeeding: true,
            adaptiveDepth: false,
            entropyThreshold: 0.7
        )
    }
}

/// Per-token log-probability entry captured from the LM head logit distribution.
public struct QWTokenLogprob: Sendable, Equatable {
    public struct Candidate: Sendable, Equatable {
        public let tokenId: Int
        public let logprob: Float

        public init(tokenId: Int, logprob: Float) {
            self.tokenId = tokenId
            self.logprob = logprob
        }
    }

    public let tokenId: Int
    public let logprob: Float
    public let topCandidates: [Candidate]

    public init(tokenId: Int, logprob: Float, topCandidates: [Candidate] = []) {
        self.tokenId = tokenId
        self.logprob = logprob
        self.topCandidates = topCandidates
    }
}

/// Contiguous span of merged visual token embeddings in an expanded multimodal prompt.
public struct QWVisualSpan: Sendable, Equatable {
    public let tokenOffset: Int
    public let tokenCount: Int
    public let embeddings: [Float]

    public init(tokenOffset: Int, tokenCount: Int, embeddings: [Float]) {
        self.tokenOffset = tokenOffset
        self.tokenCount = tokenCount
        self.embeddings = embeddings
    }

    /// Deterministic 64-bit FNV-1a hash of the visual embedding payload and span geometry.
    public var contentHash64: UInt64 {
        var h: UInt64 = 0xcbf2_9ce4_8422_2325
        let prime: UInt64 = 0x0000_0100_0000_01b3
        h = (h ^ UInt64(bitPattern: Int64(tokenOffset))) &* prime
        h = (h ^ UInt64(bitPattern: Int64(tokenCount))) &* prime
        embeddings.withUnsafeBufferPointer { buf in
            for v in buf {
                h = (h ^ UInt64(v.bitPattern)) &* prime
            }
        }
        return h
    }

    /// Replaces visual-span token slots with deterministic hash-salted `Int64` IDs for prefix cache identity.
    public static func saltedCacheHistory(history: [Int64], visualSpans: [QWVisualSpan]) -> [Int64] {
        guard !visualSpans.isEmpty else { return history }
        var out = history
        for span in visualSpans {
            let h = span.contentHash64
            let upper = min(out.count, span.tokenOffset + span.tokenCount)
            guard span.tokenOffset >= 0, span.tokenOffset < upper else { continue }
            for pos in span.tokenOffset..<upper {
                let idx = UInt64(pos - span.tokenOffset)
                var z = h &+ idx &* 0x9E37_79B9_7F4A_7C15
                z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
                z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
                z = z ^ (z >> 31)
                out[pos] = Int64(bitPattern: 0x4000_0000_0000_0000 | (z & 0x3FFF_FFFF_FFFF_FFFF))
            }
        }
        return out
    }

    /// Replaces visual-span token slots with deterministic hash-salted token IDs in `0..<vocab`.
    public static func saltedCacheTokens(prompt: [Int], visualSpans: [QWVisualSpan], vocab: Int) -> [Int] {
        guard !visualSpans.isEmpty, vocab > 0 else { return prompt }
        var out = prompt
        let mod = UInt64(vocab)
        for span in visualSpans {
            let h = span.contentHash64
            let upper = min(out.count, span.tokenOffset + span.tokenCount)
            guard span.tokenOffset >= 0, span.tokenOffset < upper else { continue }
            for pos in span.tokenOffset..<upper {
                let idx = UInt64(pos - span.tokenOffset)
                var z = h &+ idx &* 0x9E37_79B9_7F4A_7C15
                z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
                z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
                z = z ^ (z >> 31)
                out[pos] = Int(z % mod)
            }
        }
        return out
    }
}
