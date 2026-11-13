// QWEngineDriverConformance.swift — The dense engine, seen through the
// neutral protocol.
//
// The dispatch rule lives here, in the engine that runs it: which path a
// request takes, and what it cost. The server asks for a generation and
// reads the result; it does not decide how the engine gets there.

import Foundation
import Metal
import QWCore
import QWHAL
import QWHead

private let denseDriverGenerateLock = NSRecursiveLock()

extension QWEngineDriver: QWSessionInferenceEngine {
    public var prefixCacheChargedBytes: Int { prefixStore?.chargedBytes ?? 0 }
    public var prefixCacheEntries: Int { prefixStore?.count ?? 0 }
    public var ssdCacheChargedBytes: Int { cacheDiskManager?.chargedBytes ?? 0 }
    public var ssdCacheEntries: Int { cacheDiskManager?.entryCount ?? 0 }
    public var hasPrefixCache: Bool { prefixStore != nil }
    public var arenaBytes: Int { arena.committedBytes }
    public var weightReservedBytes: Int { arena.regionSizes.weights }
    public var weightUsedBytes: Int { arena.weightsSub.usedBytes }
    public var captureTarget: any MTLCommandQueue { hal.queue }
    public var freeKVPages: Int { kvCache.freePageCount }
    /// The dense family offers every capability its module offers, and expert streaming when the model is MoE.
    public var capabilities: Set<QWKnobCapability> {
        geometry.numExperts > 0 ? DenseModule.capabilities.union([.expertStreaming]) : DenseModule.capabilities
    }
    public var supportsTokenConstraints: Bool { true }

    /// The cache the engine holds. The bytes are what the cache says it
    /// charged; an engine with no cache reports no cache at all.
    public var prefixCache: QWPrefixCacheState {
        guard hasPrefixCache else { return QWPrefixCacheState(isConfigured: false) }
        return QWPrefixCacheState(
            isConfigured: true,
            warmBytes: prefixCacheChargedBytes,
            ssdBytes: ssdCacheChargedBytes,
            warmEntries: prefixCacheEntries,
            ssdEntries: ssdCacheEntries,
            warmLimitBytes: prefixStore?.byteLimit
        )
    }

    /// The last turn's measurements, in the shape every family reports.
    public var lastTurn: QWGenerationResult? {
        // The cache's answer belongs to the turn: the consumer that
        // asks afterwards gets the answer this turn got.
        let cache = lastPrefixCacheStats.map { stats in
            QWCacheAnswer(hit: stats.hit, matchedTokens: stats.matchedTokens,
                          totalTokens: lastPromptTokens, tier: stats.tier)
        }
        if let stats = lastGeneration {
            return QWGenerationResult(tokens: [], stepMs: stats.stepMs, stepTokenCounts: [],
                                      prefillSeconds: stats.prefillSec,
                                      firstTokenSeconds: stats.firstTokenSec,
                                      decodeSeconds: stats.decodeSec,
                                      cache: cache)
        }
        if let experimental = lastExperimentalGeneration {
            return QWGenerationResult(tokens: [], stepMs: experimental.roundMs,
                                      stepTokenCounts: experimental.roundTokenCounts,
                                      prefillSeconds: experimental.prefillSec,
                                      firstTokenSeconds: experimental.prefillSec,
                                      decodeSeconds: experimental.decodeSec,
                                      cache: cache)
        }
        return nil
    }

    /// A multi-turn conversation over this engine. The dense engine
    /// drives its own steps, so it has one.
    public func makeConversation() -> QWConversation? {
        QWConversation(engine: self)
    }

    /// What the dense engine says about itself. The frame belongs to the
    /// consumer; these lines are the dense engine's own story.
    public var report: QWEngineReport {
        var lines = ""
        if lastExperimentalGeneration != nil {
            let isDSpine = dflashDrafter?.config.isDSpine == true
            let backendLabel = isDSpine ? "dspine" : mtpBackend.rawValue
            let routeLabel = isDSpine
                ? "dspine:block-causal-5L-w8192-bridge1024-r512-refine16"
                : mtpBackend.route
            let profileLabel = isDSpine ? "dspine-block-speculative" : mtpBackend.qualificationProfile
            lines += "mtp backend:             \(backendLabel) (requested: \(lastRequestedBackend ?? "default"))\n"
            lines += "effective kernel route:  \(routeLabel)\n"
            lines += "qualification profile:   \(profileLabel)\n"
            if mtpBackend == .dflash2 {
                lines += "dflash drafter pack:     \(dflashDraftPackPath ?? "none") (selector top-k: \(dflashSelectorTopK ?? 16))\n"
            }
            if mtpBackend == .variational {
                lines += "variational margin:      \(String(format: "%.2f", mtpRelaxedMargin)) (min prob: \(String(format: "%.2f", effectiveRelaxedMinProb)), quadratic seeding: \(mtpQuadraticSeeding))\n"
            }
        }
        lines += "prefill scratch policy:  \(prefillScratchEviction.rawValue) (auto <= 36 GB RAM)\n"
        let chunkTimings = prefillTimingLedger.values()
        if !chunkTimings.isEmpty {
            var encodeMicros: Double = 0
            for chunk in chunkTimings { encodeMicros += chunk.encodeMicros }
            lines += "prefill chunks:          \(chunkTimings.count) (host elapsed: \(String(format: "%.2f", Double(encodeMicros) / 1_000_000)) s)\n"
        }
        if lastExperimentalGeneration != nil, mtpDraftProposals > 0 {
            let accepted = Double(mtpDraftAccepted) / Double(mtpDraftProposals) * 100.0
            lines += "speculative cascade:     MTP: \(mtpDraftAccepted)/\(mtpDraftProposals) accepted (\(String(format: "%.1f", accepted))%)\n"
        }
        return QWEngineReport(
            capacityLog: capacityLogMessage,
            arenaBytes: arenaBytes,
            weightUsedBytes: weightUsedBytes,
            weightReservedBytes: weightReservedBytes,
            steadyAllocations: steadyAllocs,
            pipelineSeconds: QWHalPSOStats.seconds,
            detailLines: lines,
            weightFormat: weightFormat.name,
            weightFormatDeclineReason: weightFormatDeclineReason
        )
    }

    /// Runs one generation and reports what it cost. The rule: the
    /// prefix cache when the caller wants it, greedy when the sampler
    /// asks for greedy, the drafter when the caller asked for one.
    public func generate(
        request: QWGenerationRequest,
        onToken: ((Int) -> Bool)?
    ) throws -> QWGenerationResult {
        denseDriverGenerateLock.lock()
        defer { denseDriverGenerateLock.unlock() }
        if request.usePrefixCache && !hasPrefixCache {
            throw QWEngineDriverError.state("the request wants a prefix cache this engine does not have")
        }
        // The records this generation reports must be its own: a record
        // left by an earlier turn is not this turn's measurement.
        lastGeneration = nil
        lastExperimentalGeneration = nil
        lastPrefixCacheStats = nil
        lastRequestedBackend = request.speculative.mtpBackend
        let reqBackend = request.speculative.mtpBackend?.lowercased()
        // The request's floor holds for this generation only (we are under the generate lock).
        let configuredMinProb = mtpRelaxedMinProb
        if request.sampling.hasExplicitSeed {
            activeRelaxedMarginOverride = 0.0
        } else if reqBackend == "variational" || reqBackend == "relaxed" || reqBackend == "plm" {
            activeRelaxedMarginOverride = Float(max(0.0, request.speculative.mtpRelaxedMargin))
            mtpRelaxedMinProb = request.speculative.mtpRelaxedMinProb
        } else if reqBackend == "standard" || reqBackend == "dflash2" || reqBackend == "dspine" {
            activeRelaxedMarginOverride = 0.0
        } else {
            activeRelaxedMarginOverride = nil
        }
        activeCollectLogprobs = request.collectLogprobs
        activeConstraint = request.constraint.map { ($0, QWSampler(config: request.sampling)) }
        activeVisualSpans = request.visualSpans
        defer {
            activeVisualSpans = []
            activeConstraint = nil
            activeCollectLogprobs = false
            activeRelaxedMarginOverride = nil
            mtpRelaxedMinProb = configuredMinProb
        }
        lastPromptTokens = request.prompt.count
        lastStopToken = nil
        if request.collectTimings { prefillTimingLedger.setEnabled(true) }
        let prefillMark = prefillTimingLedger.values().count
        let cacheActive = request.usePrefixCache
        let pureGreedy = (request.sampling.isGreedy && request.sampling.repetitionPenalty == 1.0)
            || request.constraint != nil
        // The engine's own drafts decide whether it can speculate: a
        // request that asks for a drafter this engine does not have is
        // served without one, never refused.
        let useSpeculative = request.speculative.enabled && mtpDrafts > 0
            && !request.collectLogprobs && (request.constraint?.allowsSpeculation ?? true)
            && !(request.sampling.hasExplicitSeed && !request.sampling.isGreedy)
        var collectedLogprobs: [QWTokenLogprob] = []
        let lpSampler = request.collectLogprobs ? QWSampler(config: request.sampling) : nil
        let topKCount = max(0, min(20, request.topLogprobsCount))
        let effectiveOnToken: ((Int) -> Bool)? = (onToken != nil || request.collectLogprobs) ? { [self] tok in
            if let lpSampler {
                let rawLogits = sc.logits.region.pointer.advanced(by: sc.logits.offset)
                    .assumingMemoryBound(to: Float.self)
                let rowBuf = UnsafeBufferPointer(start: rawLogits, count: geometry.vocab)
                let lp = lpSampler.computeTokenLogprob(logits: rowBuf, chosenToken: tok, topK: topKCount)
                collectedLogprobs.append(lp)
                request.onTokenLogprob?(lp)
            }
            return onToken?(tok) ?? true
        } : nil
        let tokens: [Int]
        // The stop token the turn ended on. The dense engine reports it
        // on the speculative path, where the drafter can reach it first.
        var stoppedToken: Int?

        if cacheActive && pureGreedy {
            tokens = try generateCachedGreedy(
                prompt: request.prompt,
                maxNew: request.maxNew,
                stopIds: request.stopIDs,
                speculative: useSpeculative,
                retainPrefixLength: request.retainPrefixLength,
                onToken: effectiveOnToken
            )
        } else if cacheActive {
            tokens = try generateCachedStochastic(
                prompt: request.prompt,
                maxNew: request.maxNew,
                config: request.sampling,
                stopIds: request.stopIDs,
                speculative: useSpeculative,
                retainPrefixLength: request.retainPrefixLength,
                onToken: effectiveOnToken
            )
        } else if pureGreedy {
            let session = try makeSession(promptTokens: request.prompt.count)
            defer { finishSession(session) }
            tokens = useSpeculative
                ? try generateExperimentalGreedy(
                    session: session,
                    prompt: request.prompt,
                    maxNew: request.maxNew,
                    stopIds: request.stopIDs,
                    onToken: effectiveOnToken,
                    onStopToken: { stoppedToken = $0 }
                )
                : try generate(
                    session: session,
                    prompt: request.prompt,
                    maxNew: request.maxNew,
                    stopIds: request.stopIDs,
                    onToken: effectiveOnToken
                )
        } else {
            let session = try makeStochasticSession(
                promptTokens: request.prompt.count,
                config: request.sampling
            )
            defer { finishSession(session) }
            tokens = try generateExperimentalStochastic(
                session: session,
                prompt: request.prompt,
                maxNew: request.maxNew,
                stopIds: request.stopIDs,
                speculative: useSpeculative,
                onToken: effectiveOnToken
            )
        }

        let stats = lastGeneration
        let experimental = lastExperimentalGeneration
        // The speculative turn measures its rounds; the plain turn
        // measures its steps. The result carries whichever ran.
        let stepMs = stats?.stepMs ?? experimental?.roundMs ?? []
        let stepTokenCounts = experimental?.roundTokenCounts ?? []
        var gpuMicros: Double = 0
        for chunk in prefillTimingLedger.values().dropFirst(prefillMark) {
            if let gpu = chunk.gpuMicros { gpuMicros += gpu }
        }
        let gpuSeconds = Double(gpuMicros) / 1_000_000
        let cache = lastPrefixCacheStats.map { stats in
            QWCacheAnswer(hit: stats.hit, matchedTokens: stats.matchedTokens,
                          totalTokens: request.prompt.count, tier: stats.tier)
        }
        var specSummary: String? = nil
        if let g = experimental, !g.roundTokenCounts.isEmpty {
            let totalTokens = g.roundTokenCounts.reduce(0, +)
            let rounds = g.roundTokenCounts.count
            specSummary = String(format: "rounds=%d, mean=%.2f tok/rnd", rounds, Double(totalTokens) / Double(rounds))
        } else if useSpeculative {
            specSummary = String(format: "mtp=%d/%d", mtpDraftAccepted, mtpDraftProposals)
        }
        return QWGenerationResult(
            tokens: tokens,
            stepMs: stepMs,
            stepTokenCounts: stepTokenCounts,
            prefillSeconds: stats?.prefillSec ?? experimental?.prefillSec,
            firstTokenSeconds: stats?.firstTokenSec ?? experimental?.prefillSec,
            decodeSeconds: stats?.decodeSec ?? experimental?.decodeSec,
            prefillGPUSeconds: gpuSeconds > 0 ? gpuSeconds : nil,
            stopTokenID: stoppedToken ?? lastStopToken,
            cache: cache,
            speculativeSummary: specSummary,
            tokenLogprobs: collectedLogprobs
        )
    }
}

extension QWEngineDriver {
    /// Effective variational acceptance floor on `p_target(draft)` (`0` disables it).
    public var effectiveRelaxedMinProb: Float {
        Float(min(1.0, max(0.0, mtpRelaxedMinProb)))
    }
}
