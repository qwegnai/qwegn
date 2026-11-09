// QWInferenceEngine.swift — The neutral engine protocol.
//
// No concrete engine type appears in any signature here. A consumer that
// holds `any QWInferenceEngine` holds an engine: it may be the dense one,
// it may be the Flash-Next one, and it may be one nobody has written yet.

import Foundation
import QWCore
import QWHead
import QWKV

/// What an engine can do, what its cache holds, and how it generates.
/// The engine-neutral surface: it names no family and no concrete type,
/// so a consumer can hold it without knowing which engine it got.
public protocol QWInferenceEngine: AnyObject, Sendable {
    /// What this engine can do. Absence means "not available", never a zero.
    var capabilities: Set<QWKnobCapability> { get }

    /// Whether a prefix cache is configured right now.
    var hasPrefixCache: Bool { get }

    /// The cache the engine holds. A byte count nobody measured is absent.
    var prefixCache: QWPrefixCacheState { get }

    /// Where the engine reports what its cache did.
    var onCacheEvent: (@Sendable (QWPrefixCacheEvent) -> Void)? { get set }

    /// What this engine reports about itself: the numbers and the lines
    /// a consumer prints without knowing the family.
    var report: QWEngineReport { get }

    /// The last turn's measurements. Absent when the engine ran no turn.
    var lastTurn: QWGenerationResult? { get }

    /// A multi-turn conversation over this engine. Nil when the engine
    /// owns its whole loop: there is no conversation to have.
    func makeConversation() -> QWConversation?

    /// Whether `generate` honours `QWGenerationRequest.constraint`. An
    /// engine that does not must refuse a constrained request.
    var supportsTokenConstraints: Bool { get }

    /// Runs one generation.
    func generate(request: QWGenerationRequest, onToken: ((Int) -> Bool)?) throws -> QWGenerationResult

    /// Prefills `tokens`, writes a standalone `.qwcache` file at `outputURL`, and optionally pins it in warm RAM.
    func buildStandalonePrefixCache(tokens: [Int], outputURL: URL, pinAsSharedBase: Bool) throws -> Int

    /// Mounts a standalone `.qwcache` file from `fileURL` into warm RAM as a pinned shared base prefix.
    func mountStandalonePrefixCache(from fileURL: URL) throws -> (tokens: [Int], byteCount: Int)

    /// Unmounts all pinned shared-base prefixes from the warm prefix store.
    func unmountSharedBasePrefixes()
}

extension QWInferenceEngine {
    public var supportsTokenConstraints: Bool { false }

    public func buildStandalonePrefixCache(tokens: [Int], outputURL: URL, pinAsSharedBase: Bool = true) throws -> Int {
        throw QWModuleError.notImplemented("buildStandalonePrefixCache")
    }

    public func mountStandalonePrefixCache(from fileURL: URL) throws -> (tokens: [Int], byteCount: Int) {
        throw QWModuleError.notImplemented("mountStandalonePrefixCache")
    }

    public func unmountSharedBasePrefixes() {}
}

/// An engine whose caller drives the steps: it makes a session, prefills
/// it and decodes it token by token. A module that owns the whole loop
/// does not offer these, and does not conform here.
public protocol QWSessionInferenceEngine: QWInferenceEngine {
    /// The frozen plan of the model this engine serves.
    var plan: QWEnginePlan { get }
    var ropeRows: Int { get }
    var prefillChunkSize: Int { get }
    var mtpDrafts: Int { get }
    var mtpDraftProposals: Int { get }
    var mtpDraftAccepted: Int { get }

    var lastPrefixCacheStats: QWPrefixCacheStats? { get }
    var lastGeneration: QWGenerationStats? { get }
    var lastExperimentalGeneration: QWExperimentalGenerationStats? { get }

    func makeSession(promptTokens: Int) throws -> QWEngineSession
    func makeStochasticSession(promptTokens: Int, config: QWSamplerConfig) throws -> QWEngineSession
    func finishSession(_ session: QWEngineSession)
    func snapshotSession(_ session: QWEngineSession) throws -> QWEngineSessionSnapshot
    func rollbackSession(_ session: QWEngineSession, to snapshot: QWEngineSessionSnapshot) throws

    @discardableResult
    func prefill(_ session: QWEngineSession, tokens: [Int], mode: GDNPrefillMode?) throws -> Int
    func decodeStep(_ session: QWEngineSession, token: Int) throws -> Int

    func generate(
        session: QWEngineSession,
        prompt: [Int],
        maxNew: Int,
        stopIds: Set<Int>,
        onToken: ((Int) -> Bool)?
    ) throws -> [Int]

    func generateExperimentalGreedy(
        session: QWEngineSession,
        prompt: [Int],
        maxNew: Int,
        stopIds: Set<Int>,
        onToken: ((Int) -> Bool)?,
        onStopToken: ((Int) -> Void)?
    ) throws -> [Int]

    func generateExperimentalStochastic(
        session: QWEngineSession,
        prompt: [Int],
        maxNew: Int,
        stopIds: Set<Int>,
        speculative: Bool,
        onToken: ((Int) -> Bool)?
    ) throws -> [Int]

    func generateCachedGreedy(
        prompt: [Int],
        maxNew: Int,
        stopIds: Set<Int>,
        speculative: Bool,
        retainPrefixLength: Int?,
        onToken: ((Int) -> Bool)?,
        inspectCompleted: ((QWEngineSession) throws -> Void)?
    ) throws -> [Int]

    func generateCachedStochastic(
        prompt: [Int],
        maxNew: Int,
        config: QWSamplerConfig,
        stopIds: Set<Int>,
        speculative: Bool,
        retainPrefixLength: Int?,
        onToken: ((Int) -> Bool)?,
        inspectCompleted: ((QWEngineSession) throws -> Void)?
    ) throws -> [Int]
}

extension QWSessionInferenceEngine {
    public func generateExperimentalGreedy(
        session: QWEngineSession,
        prompt: [Int],
        maxNew: Int,
        stopIds: Set<Int> = [],
        onToken: ((Int) -> Bool)? = nil
    ) throws -> [Int] {
        try generateExperimentalGreedy(
            session: session, prompt: prompt, maxNew: maxNew,
            stopIds: stopIds, onToken: onToken, onStopToken: nil
        )
    }

    public func generateCachedGreedy(
        prompt: [Int],
        maxNew: Int,
        stopIds: Set<Int> = [],
        speculative: Bool = false,
        retainPrefixLength: Int? = nil,
        onToken: ((Int) -> Bool)? = nil
    ) throws -> [Int] {
        try generateCachedGreedy(
            prompt: prompt, maxNew: maxNew, stopIds: stopIds,
            speculative: speculative, retainPrefixLength: retainPrefixLength,
            onToken: onToken, inspectCompleted: nil
        )
    }

    public func generateCachedStochastic(
        prompt: [Int],
        maxNew: Int,
        config: QWSamplerConfig,
        stopIds: Set<Int> = [],
        speculative: Bool = false,
        retainPrefixLength: Int? = nil,
        onToken: ((Int) -> Bool)? = nil
    ) throws -> [Int] {
        try generateCachedStochastic(
            prompt: prompt, maxNew: maxNew, config: config, stopIds: stopIds,
            speculative: speculative, retainPrefixLength: retainPrefixLength,
            onToken: onToken, inspectCompleted: nil
        )
    }
}
