// FlashNextGeneration.swift — The Flash-Next façade over the shared core.
//
// The engine owns the numerics; the core owns the loop. This file is the
// seam between them: the public entry point the CLI and the server already
// call, now driven by the same core loop every family uses.

import Foundation
import QWCore
import QWFlashNext
import QWGEMM
import QWHead

/// Tracks concurrent HTTP requests currently rendering chat templates and tokenizing so the batch scheduler can coalesce bursts.
public enum QWConcurrentBatchTracker {
    public static func beginPreparing() { FNBatchAdmissionTracker.beginPreparing() }
    public static func endPreparing() { FNBatchAdmissionTracker.endPreparing() }
}

extension FNGPUEngine {
    /// Generates tokens for a prompt. The loop belongs to the core — one
    /// loop, one sampler, one session pool — and this façade hands the core
    /// a Flash-Next module. FN's numerics stay in the engine.
    public func generate(
        prompt: [Int],
        maxNew: Int,
        sampling: QWSamplingParameters = QWSamplingParameters(temperature: 0, topP: 1, topK: 1, repetitionPenalty: 1.0),
        speculative: QWSpeculativeParameters = QWSpeculativeParameters(),
        stopIds: Set<Int> = [],
        visualSpans: [QWVisualSpan] = [],
        onToken: ((Int) -> Bool)? = nil,
        onStopToken: ((Int) -> Void)? = nil
    ) throws -> GenerationResult {
        guard !prompt.isEmpty else {
            // Nothing ran, so nothing was measured: the timings are
            // absent, not zero.
            return GenerationResult(tokens: [], prefillSec: nil, prefillGPUSec: nil,
                                    decodeSec: nil, firstTokenSec: nil,
                                    stepMs: [], denseNaxLogMessage: "")
        }
        // The NAX kernel mode is chosen from the request's own
        // hyperparameters, by the one generation both doors use.
        let contract = QWSamplerConfig(sampling, vocabulary: plan.vocab)
        let module = FlashNextModule(engine: self)
        // The generation itself lives in the module: this façade and the
        // protocol surface are two doors onto one implementation.
        let result = try module.runGeneration(
            request: QWGenerationRequest(prompt: prompt, maxNew: maxNew, stopIDs: stopIds,
                                         sampling: contract, speculative: speculative,
                                         visualSpans: visualSpans),
            onToken: onToken, onStopToken: onStopToken)
        return GenerationResult(tokens: result.tokens,
                                prefillSec: result.prefillSeconds,
                                prefillGPUSec: result.prefillGPUSeconds,
                                decodeSec: result.decodeSeconds,
                                firstTokenSec: result.firstTokenSeconds,
                                stepMs: result.stepMs,
                                denseNaxLogMessage: module.naxLogMessage,
                                prefixCacheHit: module.prefixCacheHit,
                                prefixCacheMatchedTokens: module.prefixCacheMatchedTokens)
    }
}

extension FlashNextModule {
    /// The one Flash-Next generation: the shared core loop over this
    /// module's numerics. The façade and the protocol surface are two
    /// doors onto this one implementation.
    func runGeneration(
        request: QWGenerationRequest,
        onToken: ((Int) -> Bool)?,
        onStopToken: ((Int) -> Void)?
    ) throws -> QWGenerationResult {
        // An empty prompt generates nothing in this family: the loop has
        // no chunk to prefill and no token to start from. The dense
        // family answers the degenerate input differently — it throws
        // when its cache is active — so the two families differ here.
        guard !request.prompt.isEmpty else {
            return QWGenerationResult(tokens: [], stepMs: [], stepTokenCounts: [])
        }
        guard let engine = self.engine else {
            if request.constraint != nil {
                throw QWEngineError.state("token constraints need the Flash-Next GPU engine")
            }
            let core = try QWEngineCore(configuration: QWEngineCoreConfiguration(
                plan: plan, module: self, sampling: request.sampling,
                maxSessions: 1, speculative: request.speculative
            ))
            return try core.generate(prompt: request.prompt, maxNew: request.maxNew,
                                     stopIDs: request.stopIDs, sampling: request.sampling,
                                     speculative: request.speculative, onToken: onToken,
                                     onStopToken: onStopToken)
        }
        for token in request.prompt where token < 0 || token >= plan.vocab {
            throw QWModuleError.invalidToken(token)
        }
        naxLogMessage = QWGemm.configureDenseNax(
            temperature: Double(request.sampling.temperature),
            topP: Double(request.sampling.topP),
            topK: request.sampling.topK
        ).logMessage
        let useSpec = FNGPUSessionState.useSpec(sampling: request.sampling,
                                                speculative: request.speculative,
                                                engine: engine)
        let draftMode = try FNGPUSessionState.draftMode(for: request.speculative)
        engine.batchScheduler.onCacheEvaluated = { [weak self, weak engine] sessionData in
            guard let self, let engine else { return }
            self.reportCacheEvent(engine: engine, sessionData: sessionData)
        }
        engine.batchScheduler.onSlot0CacheEvaluated = nil
        engine.batchScheduler.onStepTiming = { gpuMs, wallMs in
            if gpuMs > 0 && wallMs > 0 {
                QWTelemetryRegistry.shared.record("gpu.busy_ratio", gpuMs / wallMs, provenance: .measured)
            }
        }
        let job = FNBatchJob(
            prompt: request.prompt,
            maxNew: request.maxNew,
            stopIDs: request.stopIDs,
            sampling: request.sampling,
            useSpec: useSpec && !request.collectLogprobs,
            draftMode: draftMode,
            maxDrafts: max(1, min(FNGPUSessionData.defaultMTPDrafts, request.speculative.effectiveDrafts)),
            relaxedMargin: request.sampling.hasExplicitSeed ? 0 : request.speculative.effectiveRelaxedMargin,
            relaxedMinProb: request.speculative.effectiveRelaxedMinProb,
            quadraticSeeding: request.speculative.mtpQuadraticSeeding,
            collectLogprobs: request.collectLogprobs,
            topLogprobsCount: request.topLogprobsCount,
            onToken: onToken,
            onStopToken: onStopToken,
            onTokenLogprob: request.onTokenLogprob,
            constraint: request.constraint,
            retainPrefixLength: request.retainPrefixLength,
            visualSpans: request.visualSpans
        )
        let out: FNBatchJobOutput
        do {
            out = try engine.batchScheduler.submit(job)
        } catch FNGPUError.invalidToken(let tok) {
            throw QWModuleError.invalidToken(tok)
        } catch let error as QWEngineError {
            throw QWModuleError.from(error)
        }
        lastSpecProposals = out.specProposals
        lastSpecAccepted = out.specAccepted
        var cache: QWCacheAnswer?
        if hasPrefixCache {
            cache = QWCacheAnswer(hit: out.prefixCacheHit,
                                  matchedTokens: out.prefixCacheMatchedTokens,
                                  totalTokens: request.prompt.count,
                                  tier: out.prefixCacheTier?.rawValue)
        }
        var specSummary: String? = nil
        if out.specProposals > 0 {
            let rounds = max(1, out.stepTokenCounts.count)
            let totalTok = out.stepTokenCounts.reduce(0, +)
            specSummary = String(format: "mtp=%d/%d, rounds=%d, mean=%.2f tok/rnd",
                                 out.specAccepted, out.specProposals,
                                 rounds, Double(totalTok) / Double(rounds))
        }
        let finalResult = QWGenerationResult(
            tokens: out.tokens,
            stepMs: out.stepMs,
            stepTokenCounts: out.stepTokenCounts,
            prefillSeconds: out.prefillSeconds,
            firstTokenSeconds: out.firstTokenSeconds,
            decodeSeconds: out.decodeSeconds,
            prefillGPUSeconds: out.prefillGPUSeconds,
            stopTokenID: out.stopTokenID,
            cache: cache,
            speculativeSummary: specSummary,
            tokenLogprobs: out.tokenLogprobs
        )
        lastResult = finalResult
        return finalResult
    }
}
