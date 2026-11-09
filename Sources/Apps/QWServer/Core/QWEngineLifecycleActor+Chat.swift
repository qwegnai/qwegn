// QWEngineLifecycleActor+Chat.swift — the inference handler the
// HTTP routes call. Split out so the actor file stays under the
// repository's file-size rule.
import Foundation
import QWCli
import QWCore
import QWEngine
import QWHAL
import QWHead
import QWKV
import QWMM
import QWQuant
import QWToken
import QWVision

public struct ChatCompletionExecutionResult: Sendable {
    public let content: String
    public let promptTokens: Int
    public let completionTokens: Int
    public let reasoningTokens: Int?
    public let cachedTokens: Int
    public let truncated: Bool
    public let startsInThinking: Bool
    public let matchedStopSequence: String?
    public let logprobs: [ChatLogprobTokenEntry]?
    public let prefillSec: Double?
    public let decodeSec: Double?
}

extension QWEngineLifecycleActor {

    /// Dynamically loads or unloads the vision tower in place without unloading the LLM or resetting the KV cache.
    public func setVisionLoaded(_ loadVision: Bool) throws {
        config.loadVision = loadVision
        statusBox.updateConfig(config)
        if case .loaded(let loaded) = state {
            try loaded.visionBackend?.setVisionLoaded(loadVision)
            logger.log(.info, "Vision tower \(loadVision ? "loaded into" : "unloaded from") memory in place (KV cache preserved).")
        }
    }

    nonisolated public func handleChatCompletion(
        request: ChatCompletionRequest,
        onToken: ((String) -> Void)? = nil,
        onTokenLogprob: ((ChatLogprobTokenEntry) -> Void)? = nil
    ) async throws -> ChatCompletionExecutionResult {
        try await selectModelIfNeeded(request.model, allowSameFamilyQuantSwitch: false)
        await beginInFlightRequest()
        do {
            let result = try await executeChatCompletion(
                request: request,
                onToken: onToken,
                onTokenLogprob: onTokenLogprob
            )
            await endInFlightRequest()
            return result
        } catch {
            await endInFlightRequest()
            throw error
        }
    }

    nonisolated private func executeChatCompletion(
        request: ChatCompletionRequest,
        onToken: ((String) -> Void)?,
        onTokenLogprob: ((ChatLogprobTokenEntry) -> Void)?
    ) async throws -> ChatCompletionExecutionResult {
        let memState = await autoMountAndPrepareMemoryState(for: request)
        let loaded = try await ensureLoaded()
        let config = getConfig()
        QWConcurrentBatchTracker.beginPreparing()
        var preparingActive = true
        defer {
            if preparingActive { QWConcurrentBatchTracker.endPreparing() }
        }
        let inferStart = Date()
        let tok = loaded.tokenizer

        let imageUrls = request.extractImageUrls()

        let prepared = try QWChatTurnPreparer.preparePrompt(
            request: request,
            config: config,
            chatTemplate: loaded.chatTemplate,
            tokenizer: tok,
            descriptor: loaded.descriptor,
            supportsTokenConstraints: loaded.modelBackend.supportsTokenConstraints,
            memoryManifest: memState.manifest
        )
        let shouldThink = prepared.shouldThink
        let isConstrainedJson = prepared.isConstrainedJson
        let wire = prepared.wire

        let encodedWire = try await encodeWireIncrementallyWithSystemPrefix(
            wire: wire, tokenizer: tok, packPath: config.packPath
        )
        let promptTokens = encodedWire.tokens
        let systemPrefixTokenCount = encodedWire.systemPrefixTokenCount
        let mmTurn = try await QWImageDecoder.prepareMultimodalTurn(
            promptTokens: promptTokens,
            imageUrls: imageUrls,
            hiddenDim: loaded.descriptor.hidden,
            visionBackend: loaded.visionBackend,
            onImageEncoded: { [logger] idx, total, h, w, ms in
                logger.log(.info, String(format: "vision_encode_ms=%.1f image=%d/%d patches=%dx%d",
                                         ms, idx, total, h, w))
            }
        )
        if mmTurn.loadedOnDemand {
            logger.log(.info, "vision_tower_loaded_on_demand=true")
        } else if mmTurn.unloadedOnText {
            logger.log(.info, "vision_tower_unloaded_on_text_prompt=true")
        }
        let visualSpans = mmTurn.visualSpans
        let requestedMaxNew = request.maxTokens ?? request.maxCompletionTokens
            ?? config.modelMaxLength.map { max(0, $0 - mmTurn.tokens.count) }
            ?? config.maxTokens
        let budgetGuard = QWMemoryVault.applyElasticBudgetGuard(
            mountedMemoryTokens: memState.mountedTokens,
            deltaTokens: memState.deltaTokens,
            agentPromptTokens: mmTurn.tokens.count,
            maxNewTokens: requestedMaxNew,
            maxContext: config.maxContext
        )
        if budgetGuard.bypassedOversizedCache {
            logger.log(.info, "[MemoryPack] Elastic budget guard bypassed oversized .qwcache (\(memState.mountedTokens.count) tokens) to preserve \(mmTurn.tokens.count)-token agent prompt within maxContext=\(config.maxContext)")
        }
        let memoryPrefixTokens = budgetGuard.prefixTokens
        let effectivePromptTokens: [Int]
        let effectiveSystemPrefixTokenCount: Int?
        if visualSpans.isEmpty && !memoryPrefixTokens.isEmpty {
            effectivePromptTokens = memoryPrefixTokens + mmTurn.tokens
            effectiveSystemPrefixTokenCount = systemPrefixTokenCount.map { memoryPrefixTokens.count + $0 }
        } else {
            effectivePromptTokens = mmTurn.tokens
            effectiveSystemPrefixTokenCount = systemPrefixTokenCount
        }

        var stops = prepared.stops
        let fmtType = request.responseFormatType
        let hasActiveTools = prepared.hasActiveTools
        let imEndTokenId = prepared.imEndTokenId
        let endThinkTokenId = prepared.endThinkTokenId
        let userTextContainsImEnd = prepared.userTextContainsImEnd
        let jsonConstraint = prepared.jsonConstraint
        let activeConstraint = prepared.activeConstraint
        let customStopStrings = prepared.customStopStrings

        final class StreamCapture: @unchecked Sendable {
            let tok: QWTokenizer
            var streamDecoder: QWStreamDecoder
            var fullText: String
            var emittedUTF8Count: Int = 0
            var pendingPrefix: String
            var hasReceivedPiece: Bool = false
            var ttftSec: Double? = nil
            var inThinking: Bool
            var reasoningTokenCount: Int = 0
            var matchedStopSequence: String? = nil
            var stoppedByCustomRule: Bool = false
            let onToken: ((String) -> Void)?
            let onTokenLogprob: ((ChatLogprobTokenEntry) -> Void)?
            let modelBackend: any QWModelBackend
            init(tok: QWTokenizer, initialPrefix: String, startsInThinking: Bool,
                 onToken: ((String) -> Void)?,
                 onTokenLogprob: ((ChatLogprobTokenEntry) -> Void)?,
                 modelBackend: any QWModelBackend) {
                self.tok = tok
                self.streamDecoder = QWStreamDecoder(tok)
                self.fullText = initialPrefix
                self.pendingPrefix = initialPrefix
                self.inThinking = startsInThinking
                self.onToken = onToken
                self.onTokenLogprob = onTokenLogprob
                self.modelBackend = modelBackend
            }

            func emitUpTo(_ utf8Limit: Int) {
                guard let onToken else {
                    emittedUTF8Count = utf8Limit
                    return
                }
                let views = fullText.utf8
                guard utf8Limit > emittedUTF8Count, utf8Limit <= views.count else { return }
                let startIdx = views.index(views.startIndex, offsetBy: emittedUTF8Count)
                let endIdx = views.index(views.startIndex, offsetBy: utf8Limit)
                if let str = String(views[startIdx..<endIdx]) {
                    emittedUTF8Count = utf8Limit
                    if !str.isEmpty { onToken(str) }
                }
            }
        }
        let capture = StreamCapture(
            tok: tok,
            initialPrefix: isConstrainedJson ? "{" : "",
            startsInThinking: shouldThink,
            onToken: onToken,
            onTokenLogprob: onTokenLogprob,
            modelBackend: loaded.modelBackend
        )
        let maxNew = request.maxTokens ?? request.maxCompletionTokens
            ?? config.modelMaxLength.map { max(0, $0 - effectivePromptTokens.count) }
            ?? config.maxTokens
        if request.ignoreEos == true || (request.minTokens ?? 0) >= maxNew {
            stops.removeAll()
        }

        let telemetry = QWEngineTelemetryRegistry.shared
        telemetry.beginPrefill(promptTokens: effectivePromptTokens.count)
        let promptTokCount = effectivePromptTokens.count
        let stopOnSingleTool = (request.parallelToolCalls == false) && hasActiveTools

        let tokenCallback: @Sendable (Int) -> Bool = { id in
            if capture.ttftSec == nil {
                let firstElapsed = max(0.0001, Date().timeIntervalSince(inferStart))
                capture.ttftSec = firstElapsed
                telemetry.beginDecode(promptTokens: promptTokCount)
            } else {
                telemetry.recordDecodeToken()
            }
            if capture.inThinking {
                capture.reasoningTokenCount += 1
                if id == endThinkTokenId {
                    capture.inThinking = false
                }
            }
            var piece = ""
            if userTextContainsImEnd && id == imEndTokenId {
                let lastOpen = capture.fullText.range(of: "<function=", options: .backwards)
                let lastClose = capture.fullText.range(of: "</function>", options: .backwards)
                let insideOpenToolFunction: Bool
                if let lo = lastOpen {
                    insideOpenToolFunction = (lastClose == nil) || (lo.lowerBound > lastClose!.lowerBound)
                } else {
                    insideOpenToolFunction = false
                }
                let insideOpenJson = isConstrainedJson && (Self.firstCompleteJsonObjectEnd(in: capture.fullText) == nil)
                if insideOpenToolFunction || insideOpenJson {
                    piece = "<|im_end|>"
                } else {
                    capture.stoppedByCustomRule = true
                    return false
                }
            } else if let decoded = try? capture.streamDecoder.decode(id, skipSpecialTokens: true) {
                piece = decoded
            }
            guard !piece.isEmpty else { return true }

            if !capture.hasReceivedPiece {
                capture.hasReceivedPiece = true
                if isConstrainedJson {
                    let trimmed = piece.drop(while: { $0.isWhitespace })
                    if trimmed.hasPrefix("{") {
                        piece = String(trimmed.dropFirst())
                    }
                }
            }
            capture.fullText += piece
            if capture.inThinking && capture.fullText.contains("</think>") {
                capture.inThinking = false
            }

            if !customStopStrings.isEmpty {
                var earliestRange: Range<String.Index>? = nil
                var matchedStop: String? = nil
                for stopStr in customStopStrings {
                    if let r = capture.fullText.range(of: stopStr) {
                        if earliestRange == nil || r.lowerBound < earliestRange!.lowerBound {
                            earliestRange = r
                            matchedStop = stopStr
                        }
                    }
                }
                if let range = earliestRange, let matchedStop {
                    capture.matchedStopSequence = matchedStop
                    capture.stoppedByCustomRule = true
                    capture.fullText = String(capture.fullText[..<range.lowerBound])
                    capture.emitUpTo(capture.fullText.utf8.count)
                    return false
                }
            }

            if let jsonConstraint, jsonConstraint.isComplete {
                capture.stoppedByCustomRule = true
                capture.emitUpTo(capture.fullText.utf8.count)
                return false
            }

            if isConstrainedJson, let jsonEnd = Self.firstCompleteJsonObjectEnd(in: capture.fullText) {
                capture.fullText = String(capture.fullText[..<jsonEnd])
                capture.stoppedByCustomRule = true
                capture.emitUpTo(capture.fullText.utf8.count)
                return false
            }

            if stopOnSingleTool, let closeRange = capture.fullText.range(of: "</tool_call>") {
                capture.fullText = String(capture.fullText[..<closeRange.upperBound])
                capture.stoppedByCustomRule = true
                capture.emitUpTo(capture.fullText.utf8.count)
                return false
            }

            if customStopStrings.isEmpty {
                capture.emitUpTo(capture.fullText.utf8.count)
            } else {
                let hold = Self.heldStopSuffixUTF8(capture.fullText, stops: customStopStrings)
                let safeUTF8 = max(0, capture.fullText.utf8.count - hold)
                capture.emitUpTo(safeUTF8)
            }
            return true
        }

        let collectLP = request.logprobs == true
        let topLPCount = min(max(0, request.topLogprobs ?? 0), 20)
        var logprobCallback: (@Sendable (QWTokenLogprob) -> Void)? = nil
        if collectLP && capture.onTokenLogprob != nil {
            logprobCallback = { (lp: QWTokenLogprob) in
                let entry = Self.makeLogprobEntry(lp, tokenizer: capture.tok)
                capture.onTokenLogprob?(entry)
            }
        }

        let sampling = config.samplingParameters.merging(
            temperature: request.temperature, topP: request.topP, topK: request.topK,
            repetitionPenalty: request.repetitionPenalty, seed: request.seed
        )
        let requestCarriesSampling = request.temperature != nil || request.topP != nil
            || request.topK != nil || request.repetitionPenalty != nil || request.seed != nil
        logSamplingSelectionIfChanged(sampling.denseNaxSelectionLogMessage(
            paramsSource: requestCarriesSampling ? "request override"
                : Self.samplingParamsSourceLabel(config.sourceSummary)))

        let firstVisualOffset = visualSpans.map(\.tokenOffset).min() ?? effectivePromptTokens.count
        let retainPrefixLen: Int? = ((effectiveSystemPrefixTokenCount ?? 0) >= 128
            && (effectiveSystemPrefixTokenCount ?? 0) < firstVisualOffset)
            ? effectiveSystemPrefixTokenCount
            : nil
        let genRequest = QWModelGenerationRequest(
            promptTokens: effectivePromptTokens,
            maxNewTokens: maxNew,
            sampling: sampling,
            stopTokenIds: stops,
            retainPrefixLength: retainPrefixLen,
            hasImages: !visualSpans.isEmpty,
            collectLogprobs: collectLP,
            topLogprobsCount: topLPCount,
            onTokenLogprob: logprobCallback,
            constraint: activeConstraint,
            exactSpeculation: isConstrainedJson,
            visualSpans: visualSpans
        )

        let metrics: QWModelCompletionMetrics
        preparingActive = false
        QWConcurrentBatchTracker.endPreparing()
        do {
            metrics = try await withCheckedThrowingContinuation { cont in
                DispatchQueue.global(qos: .userInitiated).async {
                    do {
                        let m = try capture.modelBackend.generate(
                            request: genRequest,
                            config: config,
                            onToken: tokenCallback
                        )
                        cont.resume(returning: m)
                    } catch {
                        cont.resume(throwing: error)
                    }
                }
            }
        } catch {
            telemetry.resetToIdle()
            QWRequestRecordStore.shared.append(QWRequestRecord(
                id: QWRequestRecordStore.shared.nextId(),
                time: Date(),
                modelId: config.packName,
                promptTokens: effectivePromptTokens.count,
                generatedTokens: 0,
                outcome: "failed"
            ))
            throw error
        }

        let flushTail = capture.streamDecoder.flush()
        if !flushTail.isEmpty && !capture.stoppedByCustomRule {
            capture.fullText += flushTail
        }
        if !customStopStrings.isEmpty && capture.matchedStopSequence == nil {
            for stopStr in customStopStrings {
                if let r = capture.fullText.range(of: stopStr) {
                    capture.matchedStopSequence = stopStr
                    capture.stoppedByCustomRule = true
                    capture.fullText = String(capture.fullText[..<r.lowerBound])
                    break
                }
            }
        }
        capture.emitUpTo(capture.fullText.utf8.count)

        var fullText = capture.fullText
        if let fmtType, (fmtType == "json_schema" || fmtType == "json_object"),
           (isConstrainedJson || !fullText.contains("<tool_call>")),
           let startBrace = fullText.firstIndex(of: "{"),
           let endBrace = Self.firstCompleteJsonObjectEnd(in: String(fullText[startBrace...])) {
            let sub = String(fullText[startBrace...])
            fullText = String(sub[..<endBrace])
        }
        let ttftSec = capture.ttftSec

        let totalDuration = Date().timeIntervalSince(inferStart)
        let ttftMs = (ttftSec ?? metrics.firstTokenSec).map { $0 * 1000.0 }
        let decodeSec = metrics.decodeSec
        let decodedTokens = metrics.tokens.count > 1 ? metrics.tokens.count - 1 : metrics.tokens.count
        let decodeTokPerSec = decodeSec.flatMap { $0 > 0 && decodedTokens > 0 ? Double(decodedTokens) / $0 : nil }

        if metrics.cacheActive {
            telemetry.recordCacheLookup(hitTier: metrics.cacheTier, matchedTokens: metrics.cacheHitTokens)
            telemetry.updateCacheUsage(
                warmBytes: metrics.warmCacheBytes,
                warmLimit: config.cacheConfig.warmByteLimit,
                ssdBytes: metrics.ssdCacheBytes,
                ssdLimit: config.cacheConfig.ssdByteLimit
            )
        }
        telemetry.endRequest(
            tokensIn: effectivePromptTokens.count,
            tokensOut: metrics.tokens.count,
            cachedTokens: metrics.cacheHitTokens,
            prefillSec: metrics.prefillSec,
            decodeSec: decodeSec,
            ttftSec: ttftSec ?? metrics.firstTokenSec
        )
        telemetry.updateContext(
            active: effectivePromptTokens.count + metrics.tokens.count,
            capacity: metrics.activeContextCapacity
        )

        logger.logRequest(
            method: "POST", path: "/v1/chat/completions", statusCode: 200,
            promptTokens: effectivePromptTokens.count, completionTokens: metrics.tokens.count,
            imagesCount: imageUrls.count, cacheHitTokens: metrics.cacheHitTokens,
            cacheTier: metrics.cacheTier, cacheActive: metrics.cacheActive,
            specSummary: metrics.speculativeSummary,
            ttftMs: ttftMs, decodeTokPerSec: decodeTokPerSec, totalSec: totalDuration
        )

        QWRequestRecordStore.shared.append(QWRequestRecord(
            id: QWRequestRecordStore.shared.nextId(),
            time: Date(),
            modelId: config.packName,
            promptTokens: effectivePromptTokens.count,
            generatedTokens: metrics.tokens.count,
            cachedTokens: metrics.cacheHitTokens,
            cacheHitTier: metrics.cacheTier,
            ttftMs: ttftMs,
            prefillMs: metrics.prefillSec.map { $0 * 1000.0 },
            decodeMs: metrics.decodeSec.map { $0 * 1000.0 },
            decodeTps: decodeTokPerSec,
            stageGPU: metrics.stageGPU,
            config: [
                "kvMode": String(describing: config.kvMode),
                "mtpDrafts": String(config.mtpDrafts),
                "speculative": config.speculativeDecodingEnabled ? "on" : "off"
            ],
            outcome: "ok"
        ))

        let truncated = !capture.stoppedByCustomRule
            && (metrics.tokens.count >= maxNew)
            && (metrics.tokens.last.map { !stops.contains($0) } ?? true)
        let logprobEntries: [ChatLogprobTokenEntry]? = collectLP
            ? metrics.tokenLogprobs.map { Self.makeLogprobEntry($0, tokenizer: tok) }
            : nil
        let reasoningToks: Int? = shouldThink ? capture.reasoningTokenCount : nil

        return ChatCompletionExecutionResult(
            content: fullText,
            promptTokens: effectivePromptTokens.count,
            completionTokens: metrics.tokens.count,
            reasoningTokens: reasoningToks,
            cachedTokens: metrics.cacheHitTokens,
            truncated: truncated,
            startsInThinking: shouldThink,
            matchedStopSequence: capture.matchedStopSequence,
            logprobs: logprobEntries,
            prefillSec: metrics.prefillSec,
            decodeSec: metrics.decodeSec
        )
    }

    nonisolated private static func makeLogprobEntry(_ lp: QWTokenLogprob, tokenizer: QWTokenizer) -> ChatLogprobTokenEntry {
        let tokStr = (try? tokenizer.decode([lp.tokenId], skipSpecialTokens: false)) ?? ""
        let tokBytes = Array(tokStr.utf8).map { Int($0) }
        let tops: [ChatLogprobTokenEntry.TopLogprob] = lp.topCandidates.map { cand in
            let cStr = (try? tokenizer.decode([cand.tokenId], skipSpecialTokens: false)) ?? ""
            let cBytes = Array(cStr.utf8).map { Int($0) }
            return ChatLogprobTokenEntry.TopLogprob(token: cStr, logprob: cand.logprob, bytes: cBytes)
        }
        return ChatLogprobTokenEntry(token: tokStr, logprob: lp.logprob, bytes: tokBytes, topLogprobs: tops)
    }

    nonisolated private static func heldStopSuffixUTF8(_ text: String, stops: [String]) -> Int {
        var best = 0
        for s in stops {
            let maxChars = min(s.count - 1, text.count)
            guard maxChars > 0 else { continue }
            for len in stride(from: maxChars, through: 1, by: -1) {
                let prefix = String(s.prefix(len))
                if text.hasSuffix(prefix) {
                    best = max(best, prefix.utf8.count)
                    break
                }
            }
        }
        return best
    }

    nonisolated private static func firstCompleteJsonObjectEnd(in text: String) -> String.Index? {
        var depth = 0
        var inString = false
        var escaped = false
        var started = false
        for idx in text.indices {
            let ch = text[idx]
            if inString {
                if escaped {
                    escaped = false
                } else if ch == "\\" {
                    escaped = true
                } else if ch == "\"" {
                    inString = false
                }
                continue
            }
            if ch == "\"" {
                inString = true
            } else if ch == "{" {
                depth += 1
                started = true
            } else if ch == "}" {
                depth -= 1
                if started && depth == 0 {
                    return text.index(after: idx)
                }
            }
        }
        return nil
    }

    /// Incrementally encodes a rendered chat wire string by caching tokenized prefixes up to
    /// `<|im_start|>` turn boundaries across concurrent sessions. Because `<|im_start|>` is an added
    /// special token starting with `<` (U+003C, ccc=0), NFC normalization and BPE chunking never cross the boundary.
    func encodeWireIncrementally(wire: String, tokenizer tok: QWTokenizer, packPath: String) throws -> [Int] {
        try encodeWireIncrementallyWithSystemPrefix(wire: wire, tokenizer: tok, packPath: packPath).tokens
    }

    func encodeWireIncrementallyWithSystemPrefix(
        wire: String,
        tokenizer tok: QWTokenizer,
        packPath: String
    ) throws -> (tokens: [Int], systemPrefixTokenCount: Int?) {
        let marker = "<|im_start|>"
        guard tok.tokenId(for: marker) != nil,
              let lastRange = wire.range(of: marker, options: .backwards),
              lastRange.lowerBound > wire.startIndex else {
            return (try tok.encode(wire), nil)
        }
        let maxEntries = max(12, config.maxConcurrentSessions * 2 + 4)
        func storePrefixEntry(_ entry: CachedTokenPrefixEntry) {
            cachedTokenPrefixes.removeAll { $0.packPath == entry.packPath && $0.prefixWire == entry.prefixWire }
            cachedTokenPrefixes.append(entry)
            if cachedTokenPrefixes.count > maxEntries {
                cachedTokenPrefixes.removeFirst(cachedTokenPrefixes.count - maxEntries)
            }
        }

        var systemWire: String? = nil
        if wire.hasPrefix("<|im_start|>system") || wire.hasPrefix("<|im_start|>developer") {
            let afterFirst = wire.index(wire.startIndex, offsetBy: marker.count)
            if let secondRange = wire.range(of: marker, range: afterFirst..<wire.endIndex),
               secondRange.lowerBound <= lastRange.lowerBound {
                systemWire = String(wire[..<secondRange.lowerBound])
            }
        }

        let newPrefixWire = String(wire[..<lastRange.lowerBound])
        let tailWire = String(wire[lastRange.lowerBound...])
        let newUTF8Count = newPrefixWire.utf8.count
        let markerUTF8 = Array(marker.utf8)

        var bestIdx: Int? = nil
        var bestLen = 0
        for (idx, entry) in cachedTokenPrefixes.enumerated() where entry.packPath == packPath && !entry.prefixWire.isEmpty {
            let count = entry.prefixWire.utf8.count
            guard count > bestLen, count <= newUTF8Count,
                  newPrefixWire.utf8.starts(with: entry.prefixWire.utf8) else { continue }
            if count == newUTF8Count {
                bestIdx = idx; bestLen = count
            } else {
                let deltaStart = newPrefixWire.utf8.index(newPrefixWire.utf8.startIndex, offsetBy: count)
                if newPrefixWire.utf8[deltaStart...].starts(with: markerUTF8) {
                    bestIdx = idx; bestLen = count
                }
            }
        }

        if let idx = bestIdx {
            let matched = cachedTokenPrefixes.remove(at: idx)
            cachedTokenPrefixes.append(matched)
            let deltaStart = newPrefixWire.utf8.index(newPrefixWire.utf8.startIndex, offsetBy: matched.prefixWire.utf8.count)
            let deltaWire = String(newPrefixWire[deltaStart...])
            if deltaWire.isEmpty {
                let tailTokens = try tok.encode(tailWire)
                return (matched.prefixTokens + tailTokens, matched.systemTokenCount)
            } else if deltaWire.utf8.starts(with: markerUTF8) {
                let deltaTokens = try tok.encode(deltaWire)
                let updatedPrefixTokens = matched.prefixTokens + deltaTokens
                let sysCount = matched.systemTokenCount
                    ?? (systemWire != nil && matched.prefixWire == systemWire ? matched.prefixTokens.count : nil)
                storePrefixEntry(CachedTokenPrefixEntry(
                    packPath: packPath, prefixWire: newPrefixWire,
                    prefixTokens: updatedPrefixTokens, systemTokenCount: sysCount))
                let tailTokens = try tok.encode(tailWire)
                return (updatedPrefixTokens + tailTokens, sysCount)
            }
        }

        var resolvedSystemTokenCount: Int? = nil
        let prefixTokens: [Int]
        if let sysW = systemWire, sysW.utf8.count < newUTF8Count, newPrefixWire.utf8.starts(with: sysW.utf8) {
            let sysTokens = try tok.encode(sysW)
            resolvedSystemTokenCount = sysTokens.count
            storePrefixEntry(CachedTokenPrefixEntry(
                packPath: packPath, prefixWire: sysW, prefixTokens: sysTokens, systemTokenCount: sysTokens.count))
            let remStart = newPrefixWire.utf8.index(newPrefixWire.utf8.startIndex, offsetBy: sysW.utf8.count)
            let remTokens = try tok.encode(String(newPrefixWire[remStart...]))
            prefixTokens = sysTokens + remTokens
        } else {
            prefixTokens = try tok.encode(newPrefixWire)
            if let sysW = systemWire, sysW.utf8.count == newUTF8Count {
                resolvedSystemTokenCount = prefixTokens.count
            }
        }
        storePrefixEntry(CachedTokenPrefixEntry(
            packPath: packPath, prefixWire: newPrefixWire,
            prefixTokens: prefixTokens, systemTokenCount: resolvedSystemTokenCount))
        let tailTokens = try tok.encode(tailWire)
        return (prefixTokens + tailTokens, resolvedSystemTokenCount)
    }
}
