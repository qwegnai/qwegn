// QwgReplAndServe.swift — Interactive `repl`, `bench`, and HTTP `serve` CLI subcommands.
// Supports interactive multi-turn chat (`qwg repl`), benchmark ratchet verification
// (`qwg bench`), and the OpenAI/Ollama-compatible HTTP server (`qwg serve`)
// across both `Qwen3.8-27B` and `Qwen3.8-Flash-Next`.

import ArgumentParser
import Foundation
import QWCli
import QWCore
import QWEngine
import QWHead
import QWKV
import QWQuant
import QWServer
import QWToken

// MARK: - Repl

struct Repl: ParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Interactive multi-turn chat conversation.")
    @OptionGroup var knobs: QWGKnobOptions
    @Argument(help: "Model directory, name in ~/.qwg/models, alias, or legacy .qwp pack.") var pack: String? = nil
    @Option(name: [.customLong("model"), .customShort("m")], help: "Model directory, name in ~/.qwg/models, or alias from settings.") var modelOption: String? = nil
    @Option(name: .customLong("image"), help: "Path to image input for multimodal vision prompt.") var image: String?
    @Option(name: .customLong("video"), help: "Path to video input for multimodal vision prompt.") var video: String?
    @Flag(name: .customLong("show-thinking"), inversion: .prefixedNo, help: "Show or hide thinking tokens in output.") var showThinking: Bool = true
    @Flag(name: .customLong("hide-thinking"), help: "Hide thinking tokens in output.") var hideThinking: Bool = false
    @Flag(name: .customLong("verbose"), help: "Print generation timing statistics after each response.") var verbose: Bool = false
    @Flag(name: .customLong("reuse-prefix"), inversion: .prefixedNo, help: "Reuse exact consumed-token conversation prefixes (default: on). Disable for fresh whole-history processing; split-prefill logits can differ.") var reusePrefix: Bool = true
    @Flag(name: .customLong("stream"), inversion: .prefixedNo, help: "Stream output while generating (default: on).") var stream: Bool = true
    @Flag(name: .customLong("color"), inversion: .prefixedNo, help: "Colorize thinking output and prompts (default: auto).") var color: Bool?
    @Option(name: .customLong("dflash-window"), help: "DFlash 2 / DSpine window B in 2...16 (auto-detects dflash2 vs dspine from drafter pack and sets --mtp-drafts B-1).") var dflashWindow: Int?
    @Flag(name: .customLong("dspark-confidence-schedule"), inversion: .prefixedNo, help: "Enable adaptive confidence-head verification budget for DSpark.") var dsparkConfidenceSchedule: Bool?
    @Option(name: .customLong("repo"), help: "Repository or knowledge directory root containing .qwg/manifest.json to mount (repeatable).") var repos: [String] = []
    @Option(name: .customLong("docs"), help: "Documentation or PDF directory root to mount into Cognitive Vault v2 (repeatable).") var docs: [String] = []
    var repo: String? {
        get { repos.first }
        set { repos = newValue.map { [$0] } ?? [] }
    }
    @Flag(name: [.customLong("no-memory-pack"), .customLong("no-memory")], help: "Disable automatic mounting of .qwg/manifest.json memory pack.") var noMemory: Bool = false
    @Flag(name: .customLong("no-profile"), help: "Bypass ~/.qwg/settings.json and per-model knob profiles.") var noProfile: Bool = false

    /// A knob the user set that this command cannot honour is refused here,
    /// where the error can name the flag, instead of being ignored later.
    func validate() throws {
        try knobs.requireHonoured("qwg repl")
        try Self.validateImageOption(image: image, allowVision: knobs.loadVision ?? true)
        try Self.validateVideoOption(video: video, allowVision: knobs.loadVision ?? true)
    }

    static func validateImageOption(image: String?, allowVision: Bool) throws {
        if image != nil && !allowVision {
            throw ValidationError("image input provided but vision loading disabled via --no-load-vision")
        }
    }

    static func validateVideoOption(video: String?, allowVision: Bool) throws {
        if video != nil && !allowVision {
            throw ValidationError("video input provided but vision loading disabled via --no-load-vision")
        }
    }

    func run() throws {
        let allowVision = knobs.loadVision ?? true
        if (image != nil || video != nil) && !allowVision {
            let kind = video != nil ? "video" : "image"
            FileHandle.standardError.write("qwg repl: error: \(kind) input provided but vision loading disabled via --no-load-vision\n".data(using: .utf8)!)
            throw ExitCode(2)
        }
        let settings = QWSettingsManager.shared.loadSettings()
        let (resolvedPack, initialMtpDrafts, effectiveMtpBackend, resolvedBackend) =
            try QWResolvedCliEngineConfig.resolvePackAndMTP(
                verb: "qwg repl", modelOption: modelOption, pack: pack,
                knobs: knobs, dflashWindow: dflashWindow, settings: settings
            )
        let cliCfg = try QWResolvedCliEngineConfig.resolveAssetsAndKnobs(
            verb: "qwg repl", resolvedPack: resolvedPack, knobs: knobs, dflashWindow: dflashWindow,
            initialMtpDrafts: initialMtpDrafts, effectiveMtpBackend: effectiveMtpBackend,
            resolvedBackend: resolvedBackend, image: image, video: video, settings: settings,
            validateDefaultContextImmediately: true, repoPath: repo,
            disableMemoryPack: noMemory, ignoreProfile: noProfile
        )
        let runtimeAssets = cliCfg.runtimeAssets
        let defaultContext = cliCfg.defaultContext
        let effectiveMtpDrafts = cliCfg.effectiveMtpDrafts
        let tok = cliCfg.tokenizer
        let specialTokens = cliCfg.specialTokens

        var isThinking = knobs.enableThinking ?? true
        var displayThinking = isThinking && showThinking && !hideThinking
        var currentReasoningEffort = ChatCompletionRequest.normalizeReasoningEffort(knobs.reasoningEffort, fallback: "xhigh")
        var currentMaxTokens = knobs.maxTokens
        var currentSystem = knobs.systemPrompt
        var showStats = verbose
        let cliColor = QWCli.Color.auto(force: color)

        FileHandle.standardError.write("Loading engine weights into GPU arena...\n".data(using: .utf8)!)
        let loadT0 = Date()
        var conversation: QWConversation? = nil

        let effSettings = cliCfg.effectiveSettings
        let effectiveChunkSize = QWPrefillLimits.clampedRows(
            cliCfg.defaultPrefillChunkSize(explicitChunkSize: knobs.prefillChunkSize)
        )
        let resolvedCacheDirURL = cliCfg.resolvedCacheDirURL
        let warmBytes = Int((knobs.warmCacheLimit ?? 4.0) * 1024 * 1024 * 1024)
        let ssdEnabled = knobs.ssdCacheLimit.map { $0 > 0 } ?? effSettings.ssdCacheEnabled
        let ssdBytes = ssdEnabled ? Int((knobs.ssdCacheLimit ?? effSettings.ssdCacheLimitGB) * 1024 * 1024 * 1024) : 0
        // The engine is built by the module that serves this pack: the
        // CLI names no engine type, and it does not pick a path.
        let options = cliCfg.makeBuildOptions(
            knobs: knobs,
            maxTokens: knobs.contextTokens ?? defaultContext,
            prefillChunkSize: effectiveChunkSize,
            prefixCacheDirectory: (reusePrefix && warmBytes > 0) ? resolvedCacheDirURL : nil,
            cacheConfig: (reusePrefix && warmBytes > 0)
                ? QWCacheConfig(warmByteLimit: warmBytes, ssdByteLimit: ssdBytes,
                                cacheDirectory: resolvedCacheDirURL, wipeOnStart: false)
                : nil,
            dsparkConfidenceSchedule: dsparkConfidenceSchedule
        )
        do {
            try QWEngineInstanceLock.acquireEngineLock()
        } catch {
            FileHandle.standardError.write("qwg repl: error: \(error)\n".data(using: .utf8)!)
            throw ExitCode(1)
        }
        defer { QWEngineInstanceLock.releaseEngineLock() }
        let engine = try QWEngine(options: options, descriptor: runtimeAssets.descriptor)
        let loadSec = Date().timeIntervalSince(loadT0)
        let report = engine.report
        FileHandle.standardError.write(
            "Engine ready (\(String(format: "%.1f", loadSec)) s, arena reserved \(String(format: "%.2f", Double(report.arenaBytes) / 1_073_741_824)) GiB, weights used/reserved \(String(format: "%.2f", Double(report.weightUsedBytes) / 1_073_741_824))/\(String(format: "%.2f", Double(report.weightReservedBytes) / 1_073_741_824)) GiB).\n\n".data(using: .utf8)!)
        let modelDisplayName = URL(fileURLWithPath: resolvedPack).lastPathComponent
        print("""
        \(cliColor.bold)qwg repl (\(modelDisplayName))\(cliColor.reset)
        Interactive multi-turn session. Type '/help' for commands, '/exit' to quit.
        """)
        fflush(stdout)
        var activeMemoryPack = cliCfg.memoryPack
        var mountedMemoryTokens: [Int] = []
        var recalledVaultPaths: Set<String> = []
        if let mem = activeMemoryPack {
            let mounted = try engine.mountStandalonePrefixCache(from: mem.cacheFileURL)
            mountedMemoryTokens = mounted.tokens
            let mibStr = QWStatFormat.mebibytes(mounted.byteCount)
            let cortexCount = mem.manifest.files.filter(\.inCortex).count
            let vaultCount = mem.manifest.files.count - cortexCount
            let filesDesc = vaultCount > 0
                ? "\(mem.manifest.files.count) files: \(cortexCount) cortex + \(vaultCount) vault"
                : "\(mem.manifest.files.count) files"
            FileHandle.standardError.write(Data(
                "qwg repl: mounted LSA memory pack '\(mem.manifest.packName)' (\(filesDesc), \(mounted.tokens.count) tokens, \(mibStr), enforced kv=\(mem.artifact.kvMode.uppercased())).\n".utf8
            ))
        }
        conversation = (reusePrefix && !engine.hasPrefixCache) ? engine.makeConversation() : nil
        if engine.hasPrefixCache {
            FileHandle.standardError.write(Data("qwg repl: two-tier prefix cache enabled (warm RAM \(String(format: "%.1f", knobs.warmCacheLimit ?? 4.0)) GiB, cold SSD \(String(format: "%.1f", knobs.ssdCacheLimit ?? 10.0)) GiB at \(knobs.cacheDir ?? "~/.qwg/kv-cache")).\n".utf8))
        } else if reusePrefix {
            FileHandle.standardError.write(Data("qwg repl: exact-prefix reuse enabled; split-prefill logits can differ from fresh whole-history processing. Use --no-reuse-prefix to disable.\n".utf8))
        }
    

        defer { conversation?.reset() }

        let visionController: QWCLIVisionController? = (allowVision && engine.capabilities.contains(.vision) && runtimeAssets.descriptor.visDepth > 0)
            ? (try? QWCLIVisionController(modelPath: resolvedPack))
            : nil
        if (image != nil || video != nil) && visionController == nil {
            let kind = video != nil ? "video" : "image"
            FileHandle.standardError.write("qwg repl: error: \(kind) input provided but this model pack has no active vision tower\n".data(using: .utf8)!)
            throw ExitCode(2)
        }
        var pendingTurnImage: String? = (visionController != nil) ? image : nil
        var pendingTurnVideo: String? = (visionController != nil) ? video : nil
        var activeImageHistory: [String] = []
        var activeVideoHistory: [QWVideoInputSpec] = []
        var replTempFiles: [URL] = []
        defer {
            for url in replTempFiles {
                try? FileManager.default.removeItem(at: url)
            }
        }

        var history: [QWJson] = []
        if let s = currentSystem, !s.isEmpty {
            history.append(.object([("role", .string("system")), ("content", .string(s))]))
        }

        replLoop: while true {
            FileHandle.standardOutput.write("\(cliColor.prompt)>>> \(cliColor.reset)".data(using: .utf8)!)
            fflush(stdout)
            guard var line = readLine() else {
                print("\nGoodbye!")
                break replLoop
            }
            while line.hasSuffix("\\") {
                line.removeLast()
                FileHandle.standardOutput.write("\(cliColor.dim)... \(cliColor.reset)".data(using: .utf8)!)
                fflush(stdout)
                guard let next = readLine() else { break }
                line += "\n" + next
            }
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty { continue }

            if trimmed.hasPrefix("/") {
                let cmdParts = trimmed.split(separator: " ", maxSplits: 1).map(String.init)
                let cmd = cmdParts[0].lowercased()
                let arg = cmdParts.count > 1 ? cmdParts[1].trimmingCharacters(in: .whitespaces) : ""
                let action = Repl.handleSlashCommand(
                    cmd: cmd, arg: arg, cliColor: cliColor, engine: engine,
                    runtimeAssets: runtimeAssets, cliCfg: cliCfg, showThinking: showThinking,
                    hideThinking: hideThinking, conversation: conversation,
                    visionController: visionController, history: &history,
                    activeImageHistory: &activeImageHistory, activeVideoHistory: &activeVideoHistory,
                    recalledVaultPaths: &recalledVaultPaths,
                    pendingTurnImage: &pendingTurnImage, pendingTurnVideo: &pendingTurnVideo,
                    replTempFiles: &replTempFiles, activeMemoryPack: &activeMemoryPack,
                    mountedMemoryTokens: &mountedMemoryTokens, currentSystem: &currentSystem,
                    isThinking: &isThinking, displayThinking: &displayThinking,
                    currentReasoningEffort: &currentReasoningEffort,
                    currentMaxTokens: &currentMaxTokens, showStats: &showStats)
                if action == .exit { break replLoop }
                continue
            }

            let turnImage = pendingTurnImage
            let turnVideo = pendingTurnVideo
            pendingTurnImage = nil
            pendingTurnVideo = nil
            var turnVideos: [QWVideoInputSpec] = []
            if let img = turnImage {
                activeImageHistory.append(img)
            }
            if let vid = turnVideo {
                let spec = QWVideoInputSpec(url: vid)
                activeVideoHistory.append(spec)
                turnVideos.append(spec)
            }
            var userContent: String
            if turnImage != nil || !turnVideos.isEmpty {
                userContent = QWVideoDecoder.formatPromptWithMedia(
                    userPrompt: trimmed,
                    imageCount: turnImage != nil ? 1 : 0,
                    videoInputs: turnVideos,
                    tokenizer: tok,
                    descriptor: runtimeAssets.descriptor
                )
            } else {
                userContent = trimmed
            }
            if let mem = activeMemoryPack {
                let vaultRecall = QWMemoryVault.recallVaultContext(
                    query: trimmed, manifest: mem.manifest, excludedPaths: recalledVaultPaths
                )
                if !vaultRecall.isEmpty {
                    userContent += vaultRecall.recallBlock
                    recalledVaultPaths.formUnion(vaultRecall.recalledFiles)
                }
                let delta = QWMemoryPack.computeDelta(for: mem.manifest, excludingPaths: recalledVaultPaths)
                if delta.hasChanges, !delta.deltaBlock.isEmpty {
                    userContent += delta.deltaBlock
                    recalledVaultPaths.formUnion(delta.modifiedFiles)
                    recalledVaultPaths.formUnion(delta.deletedFiles)
                }
            }
            history.append(.object([("role", .string("user")), ("content", .string(userContent))]))
            let effectiveMaxNew = currentMaxTokens ?? (isThinking ? 512 : 128)
            var tplOpts = QWChatTemplateOptions()
            tplOpts.addGenerationPrompt = true
            tplOpts.enableThinking = .bool(isThinking)
            if isThinking { tplOpts.reasoningEffort = .string(currentReasoningEffort) }

            do {
                let maxCtx = knobs.contextTokens ?? defaultContext
                var wire = try runtimeAssets.chatTemplate.render(messages: history, options: tplOpts)
                var promptIds = try tok.encode(wire)
                let budgetGuard = QWMemoryVault.applyElasticBudgetGuard(
                    mountedMemoryTokens: mountedMemoryTokens,
                    deltaTokens: [],
                    agentPromptTokens: promptIds.count,
                    maxNewTokens: effectiveMaxNew,
                    maxContext: maxCtx
                )
                let memoryPrefixIds = budgetGuard.prefixTokens
                while memoryPrefixIds.count + promptIds.count + effectiveMaxNew + 64 > maxCtx && history.count > 2 {
                    let removeIdx = history.first?.get("role").isString("system") == true ? 1 : 0
                    if removeIdx < history.count {
                        if case .string(let s) = history[removeIdx].get("content") {
                            if (s.contains("<|image_pad|>") || s.contains("<|image|>")) && !activeImageHistory.isEmpty {
                                activeImageHistory.removeFirst()
                            }
                            if (s.contains("<|video_pad|>") || s.contains("<|video|>")) && !activeVideoHistory.isEmpty {
                                activeVideoHistory.removeFirst()
                            }
                        }
                        history.remove(at: removeIdx)
                    }
                    if removeIdx < history.count && history[removeIdx].get("role").isString("assistant") {
                        history.remove(at: removeIdx)
                    }
                    wire = try runtimeAssets.chatTemplate.render(messages: history, options: tplOpts)
                    promptIds = try tok.encode(wire)
                    FileHandle.standardError.write("\(cliColor.systemNotice)[qwg repl: trimmed older context to fit window]\(cliColor.reset)\n".data(using: .utf8)!)
                }
                if (!activeImageHistory.isEmpty || !activeVideoHistory.isEmpty) && visionController == nil {
                    throw QWMultimodalError.badArgument("Vision tower is disabled or unavailable for this model.")
                }
                let mmTurn = try visionController?.prepareTurn(
                    promptTokens: promptIds,
                    imagePaths: activeImageHistory,
                    videoInputs: activeVideoHistory,
                    currentTurnHasImage: turnImage != nil || turnVideo != nil,
                    hiddenDim: runtimeAssets.descriptor.hidden
                ) ?? (tokens: promptIds, visualSpans: [])
                promptIds = (mmTurn.visualSpans.isEmpty ? memoryPrefixIds : []) + mmTurn.tokens
                let visualSpans = mmTurn.visualSpans

                let stops: Set<Int> = specialTokens.defaultStopTokenIds
                var streamDecoder = QWStreamDecoder(tok)
                var inThinking = isThinking
                var answerStarted = !isThinking
                var generatedIds: [Int] = []

                if stream && inThinking && displayThinking {
                    FileHandle.standardOutput.write(cliColor.thinking.data(using: .utf8)!)
                }

                let onToken: (Int) -> Bool = { (id: Int) -> Bool in
                    generatedIds.append(id)
                    if id == specialTokens.endThinkTokenId {
                        inThinking = false
                        if stream { _ = streamDecoder.flush() }
                        if stream && displayThinking {
                            FileHandle.standardOutput.write((cliColor.reset + "\n\n").data(using: .utf8)!)
                        }
                        return true
                    }
                    if stream && inThinking {
                        if displayThinking,
                           let piece = try? streamDecoder.decode(id, skipSpecialTokens: true),
                           !piece.isEmpty {
                            FileHandle.standardOutput.write(piece.data(using: .utf8)!)
                        }
                    } else if stream {
                        if let piece = try? streamDecoder.decode(id, skipSpecialTokens: true), !piece.isEmpty {
                            if !answerStarted {
                                let tr = piece.drop(while: { $0.isNewline })
                                if !tr.isEmpty {
                                    answerStarted = true
                                    FileHandle.standardOutput.write(String(tr).data(using: .utf8)!)
                                }
                            } else {
                                FileHandle.standardOutput.write(piece.data(using: .utf8)!)
                            }
                        }
                    }
                    return true
                }

                let samplingParams = cliCfg.makeSamplingParameters(knobs: knobs, maxTokens: effectiveMaxNew)
                let specParams = cliCfg.makeSpeculativeParameters(knobs: knobs)
                if let conversation, visualSpans.isEmpty {
                    // The multi-turn path: the conversation keeps what
                    // the previous turn left behind.
                    generatedIds = try conversation.generate(
                        prompt: promptIds, maxNew: effectiveMaxNew, stopIds: stops,
                        speculative: effectiveMtpDrafts > 0, onToken: onToken)
                } else {
                    let request = QWGenerationRequest(
                        prompt: promptIds, maxNew: effectiveMaxNew, stopIDs: stops,
                        sampling: QWSamplerConfig(samplingParams, vocabulary: runtimeAssets.descriptor.vocab),
                        usePrefixCache: engine.hasPrefixCache,
                        speculative: specParams,
                        visualSpans: visualSpans)
                    let result = try engine.generate(request: request, onToken: onToken)
                    generatedIds = result.tokens
                }

                if stream {
                    let tail = streamDecoder.flush()
                    if !tail.isEmpty && (!inThinking || displayThinking) {
                        if !answerStarted {
                            let tr = tail.drop(while: { $0.isNewline })
                            if !tr.isEmpty { FileHandle.standardOutput.write(String(tr).data(using: .utf8)!) }
                        } else {
                            FileHandle.standardOutput.write(tail.data(using: .utf8)!)
                        }
                    }
                    FileHandle.standardOutput.write((cliColor.reset + "\n").data(using: .utf8)!)
                }

                let split = QWTokenRouter.split(generatedIds, endThinkTokenId: specialTokens.endThinkTokenId)
                let thinkText = split.hasThinkingEnd ? QWTokenRouter.text(split.thinking, tok: tok) : ""
                let ansText = QWTokenRouter.text(split.answer, tok: tok)

                if !stream {
                    if isThinking {
                        if generatedIds.contains(specialTokens.endThinkTokenId) {
                            if displayThinking && !thinkText.isEmpty {
                                FileHandle.standardOutput.write((cliColor.thinking + thinkText + cliColor.reset + "\n\n").data(using: .utf8)!)
                            }
                            if !ansText.isEmpty { FileHandle.standardOutput.write((ansText + "\n").data(using: .utf8)!) }
                        } else if displayThinking && !ansText.isEmpty {
                            FileHandle.standardOutput.write((cliColor.thinking + ansText + cliColor.reset + "\n").data(using: .utf8)!)
                        } else if !displayThinking {
                            FileHandle.standardError.write(Data("qwg repl: note: \(QWTokenRouter.thinkingBudgetNote)\n".utf8))
                        }
                    } else if !ansText.isEmpty {
                        FileHandle.standardOutput.write((ansText + "\n").data(using: .utf8)!)
                    }
                }

                var asstMsg: [(String, QWJson)] = [("role", .string("assistant")), ("content", .string(ansText))]
                if !thinkText.isEmpty { asstMsg.append(("reasoning_content", .string(thinkText))) }
                history.append(.object(asstMsg))

                if showStats {
                    let turn = engine.lastTurn
                    let prefillTokS = QWStatFormat.tokensPerSecond(tokens: promptIds.count,
                                                                     seconds: turn?.prefillSeconds)
                    let decodeTokS = QWStatFormat.tokensPerSecond(tokens: max(0, generatedIds.count - 1),
                                                                    seconds: turn?.decodeSeconds)
                    let allocs = engine.report.steadyAllocations.map(String.init) ?? "unavailable"
                    var stats = ""
                    if let cache = turn?.cache {
                        let tier = cache.tier ?? (cache.hit ? "warm_ram" : "none")
                        stats += "(prefix cache: \(cache.hit ? "HIT [\(tier)] (\(cache.matchedTokens)/\(cache.totalTokens) tokens)" : "MISS"), warm \(QWStatFormat.mebibytes(engine.prefixCache.warmBytes)), ssd \(QWStatFormat.mebibytes(engine.prefixCache.ssdBytes)))\n"
                    }
                    stats += "\(cliColor.dim)(\(generatedIds.count) tok gen, prefill \(QWStatFormat.rate(prefillTokS)), TTFT \(QWStatFormat.seconds(turn?.firstTokenSeconds)), decode \(QWStatFormat.rate(decodeTokS)), steady allocs \(allocs))\(cliColor.reset)\n\n"
                    stats += engine.report.detailText
                    FileHandle.standardError.write(stats.data(using: .utf8)!)
                } else {
                    print("")
                }
            } catch {
                conversation?.reset()
                FileHandle.standardError.write("\(cliColor.reset)\(cliColor.systemNotice)qwg repl error: \(error)\(cliColor.reset)\n\n".data(using: .utf8)!)
            }
        }
    }
}

// MARK: - Serve

struct Serve: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Start OpenAI and Ollama compatible HTTP inference server with multimodal image support and lazy loading.")

    @OptionGroup var knobs: QWGKnobOptions
    @Argument(help: "Model directory, name in ~/.qwg/models, alias, or legacy .qwp pack (optional if specified in config file or environment).") var pack: String? = nil
    @Option(name: [.customLong("model"), .customShort("m")], help: "Model directory, name in ~/.qwg/models, or alias from settings.") var modelOption: String? = nil
    @Option(name: .customLong("config"), help: "Path to settings JSON file (default: ~/.qwg/settings.json if present).") var config: String?
    @Option(name: .customLong("host"), help: "Network host interface to bind.") var host: String?
    @Option(name: .customLong("port"), help: "Port to listen on.") var port: Int?
    @Option(name: .customLong("api-key"), help: "Bearer / x-api-key token required on HTTP requests (required for non-loopback host binds).") var apiKey: String?
    @Flag(name: .customLong("allow-local-images"), help: "Allow local file:// / filesystem image URLs in server chat requests.") var allowLocalImages: Bool = false
    @Flag(name: .customLong("allow-remote-images"), help: "Allow public http/https remote image URLs in server chat requests.") var allowRemoteImages: Bool = false
    @Flag(name: .customLong("pin-model"), help: "Pin the active primary model pack against implicit request-driven swaps (explicit POST /v1/models/load still permitted).") var pinModel: Bool = false
    @Option(name: .customLong("model-name"), help: "Model name reported in /v1/models.") var modelName: String?
    @Option(name: .customLong("model-alias"), help: "Model alias accepted in /v1/chat/completions.") var modelAlias: String?
    @Option(name: .customLong("dflash-window"), help: "DFlash 2 / DSpine window B in 2...16 (auto-detects dflash2 vs dspine from drafter pack and sets --mtp-drafts B-1).") var dflashWindow: Int?
    @Flag(name: .customLong("dspark-confidence-schedule"), inversion: .prefixedNo, help: "Enable adaptive confidence-head verification budget for DSpark.") var dsparkConfidenceSchedule: Bool?
    @Option(name: .customLong("repo"), help: "Mount repository/knowledge memory packs (<path>/.qwg/manifest.json) at server startup (repeatable).") var repos: [String] = []
    @Option(name: .customLong("docs"), help: "Mount documentation/PDF directories into Cognitive Vault v2 at server startup (repeatable).") var docs: [String] = []
    var repo: String? {
        get { repos.first }
        set { repos = newValue.map { [$0] } ?? [] }
    }
    @Option(name: .customLong("parent-pid"), help: "Parent process PID to monitor for automatic termination.") var parentPid: Int32? = nil
    @Flag(name: .customLong("no-profile"), help: "Bypass ~/.qwg/settings.json and per-model knob profiles.") var noProfile: Bool = false

    /// A knob the user set that this command cannot honour is refused here,
    /// where the error can name the flag, instead of being ignored later.
    func validate() throws {
        try knobs.requireHonoured("qwg serve")
    }

    func run() async throws {
        signal(SIGPIPE, SIG_IGN)
        if let bs = dflashWindow, !(2...16).contains(bs) {
            FileHandle.standardError.write("qwg serve: error: --dflash-window must be 2...16 (got \(bs))\n".data(using: .utf8)!)
            throw ExitCode(2)
        }
        if let tk = knobs.dflashTopK, !(1...16).contains(tk) {
            FileHandle.standardError.write("qwg serve: error: --dflash-top-k must be 1...16 (got \(tk))\n".data(using: .utf8)!)
            throw ExitCode(2)
        }
        let resolved = try QWServerConfigurationResolver.resolve(
            cliConfigPath: config, cliPack: modelOption ?? pack, cliHost: host, cliPort: port,
            cliApiKey: apiKey,
            cliAllowLocalImagePaths: allowLocalImages ? true : nil,
            cliAllowRemoteImageURLs: allowRemoteImages ? true : nil,
            cliPinModel: pinModel ? true : nil,
            cliModelName: modelName, cliModelAlias: modelAlias, cliMaxContext: knobs.contextTokens,
            cliKVMode: knobs.kvMode, cliTemperature: knobs.temperature,
            cliTopP: knobs.topP, cliTopK: knobs.topK,
            cliMaxTokens: knobs.maxTokens, cliRepetitionPenalty: knobs.repetitionPenalty,
            cliSeed: knobs.seed,
            cliEnableThinking: knobs.enableThinking, cliReasoningEffort: knobs.reasoningEffort,
            cliSystemPrompt: knobs.systemPrompt, cliMTPDrafts: knobs.mtpDrafts,
            cliMTPBackend: knobs.mtpBackend,
            cliMTPRelaxedMargin: knobs.mtpRelaxedMargin,
            cliMTPRelaxedMinProb: knobs.mtpRelaxedMinProb,
            cliMTPDraftStrategy: knobs.mtpDraftStrategy,
            cliMTPQuadraticSeeding: knobs.mtpQuadraticSeeding,
            cliMTPAdaptiveDepth: knobs.mtpAdaptiveDepth,
            cliMTPEntropyThreshold: knobs.mtpEntropyThreshold,
            cliDFlashDraftPack: knobs.dflashDraftPack, cliDFlashBlockSize: dflashWindow,
            cliDFlashTopK: knobs.dflashTopK,
            cliDSparkConfidenceSchedule: dsparkConfidenceSchedule,
            cliGDNPrefill: knobs.gdnPrefill, cliEvictScratch: knobs.evictPrefillScratch,
            cliPrefillChunkSize: knobs.prefillChunkSize,
            cliMaxConcurrentSessions: knobs.maxConcurrentSessions,
            cliLSAMode: knobs.lsaMode, cliLSABudget: knobs.lsaBudget,
            cliLSAClusterBlock: knobs.lsaClusterBlock, cliLSASidecarPath: knobs.lsaSidecarPath,
            cliLSACrossLoopReuse: knobs.lsaCrossLoopReuse,
            cliWarmRAMLimitGB: knobs.warmCacheLimit,
            cliSSDCacheLimitGB: knobs.ssdCacheLimit, cliCacheDir: knobs.cacheDir,
            cliLogLevel: knobs.logLevel, cliLazy: knobs.lazy,
            cliLoadVision: knobs.loadVision, cliStreamNgrams: knobs.streamNgrams,
            cliMemoryPolicy: knobs.memoryPolicy, cliOSReserveGiB: knobs.osReserve,
            cliResidentExpertsPerLayer: knobs.residentExpertsPerLayer, cliMoESpecBudget: knobs.moeSpecBudget,
            cliDecisionModelPackPath: knobs.decisionModel,
            cliDecisionComputeUnit: knobs.decisionComputeUnit,
            cliDecisionLoadVision: knobs.decisionLoadVision,
            cliDecisionDefaultScoreReadout: knobs.decisionScoreReadout,
            cliDecisionDefaultTruncation: knobs.decisionTruncation,
            cliDecisionAlwaysEmitBox: knobs.decisionAlwaysEmitBox,
            cliIgnoreProfile: noProfile
        )
        let isDFlash2 = ["dflash2", "dflash", "dspine"].contains(resolved.mtpBackend?.lowercased() ?? "")
        let isDSpark = resolved.mtpBackend?.lowercased() == "dspark"
        let maxDrafts = 15
        guard resolved.mtpDrafts >= 0 && resolved.mtpDrafts <= maxDrafts else {
            FileHandle.standardError.write("qwg serve: error: --mtp-drafts must be 0...\(maxDrafts) (got \(resolved.mtpDrafts))\n".data(using: .utf8)!)
            throw ExitCode(2)
        }
        let resolvedKVMode: QWKVMode
        do {
            resolvedKVMode = try QWCli.resolveKVMode(resolved.kvMode)
            try QWCli.validateContextBudget(tokens: resolved.maxContext)
        } catch {
            FileHandle.standardError.write("qwg serve: error: \(error)\n".data(using: .utf8)!)
            throw ExitCode(2)
        }

        if !resolved.modelPackPath.isEmpty {
            do {
                let runtimeAssets = try QWModelRuntimeAssets.load(packPath: resolved.modelPackPath)
                if resolved.speculativeDecodingEnabled && isDFlash2 && resolved.mtpDrafts > 0 {
                    try QWDFlashConfig.validateDrafter(
                        targetDescriptor: runtimeAssets.descriptor,
                        packPath: resolved.dflashDraftPackPath,
                        drafts: resolved.mtpDrafts,
                        selectorTopK: resolved.dflashSelectorTopK
                    )
                } else if resolved.speculativeDecodingEnabled && isDSpark && resolved.mtpDrafts > 0 {
                    guard runtimeAssets.descriptor.dsparkConfig != nil else {
                        throw QWEngineError.unsupportedModel("DSpark requires dspark_* capabilities in config.json")
                    }
                    guard QWModelDescriptor.DSparkConfig.resolvePackURL(near: resolved.modelPackPath) != nil else {
                        throw QWEngineError.packMissing("dspark.qwp")
                    }
                }
            } catch {
                FileHandle.standardError.write("qwg serve: error: \(error)\n".data(using: .utf8)!)
                throw ExitCode(2)
            }
        }

        let logger = QWServerLogger.shared
        logger.setLevel(QWLogLevel.fromString(resolved.logLevel))
        let expandedLogDir = resolved.logDir.expandingTildePath
        let trimmedLogFile = resolved.logFileName.trimmingCharacters(in: .whitespacesAndNewlines)
        if !expandedLogDir.isEmpty, !trimmedLogFile.isEmpty {
            let logPath = (expandedLogDir as NSString).appendingPathComponent(trimmedLogFile)
            logger.configureFileSink(path: logPath)
        }

        let warmBytes = Int(resolved.warmRAMLimitGB * 1024 * 1024 * 1024)
        let ssdBytes = resolved.ssdCacheEnabled ? Int(resolved.ssdCacheLimitGB * 1024 * 1024 * 1024) : 0
        let cacheConfig = QWCacheConfig(
            warmByteLimit: warmBytes, ssdByteLimit: ssdBytes,
            maxConcurrentSessions: resolved.maxConcurrentSessions,
            cacheDirectory: resolved.cacheDir.canonicalFileURL,
            wipeOnStart: false
        )

        var actorConfig = QWEngineLifecycleActor.Config(
            packPath: resolved.modelPackPath, modelName: resolved.modelName,
            maxContext: resolved.maxContext, kvMode: resolvedKVMode,
            speculativeDecodingEnabled: resolved.speculativeDecodingEnabled,
            mtpDrafts: resolved.mtpDrafts, mtpBackend: resolved.mtpBackend,
            mtpRelaxedMargin: resolved.mtpRelaxedMargin,
            mtpRelaxedMinProb: resolved.mtpRelaxedMinProb,
            mtpDraftStrategy: resolved.mtpDraftStrategy,
            mtpQuadraticSeeding: resolved.mtpQuadraticSeeding,
            mtpAdaptiveDepth: resolved.mtpAdaptiveDepth,
            mtpAdaptiveEntropyThreshold: resolved.mtpEntropyThreshold,
            dflashDraftPackPath: resolved.dflashDraftPackPath.isEmpty ? nil : resolved.dflashDraftPackPath,
            dflashSelectorTopK: resolved.dflashSelectorTopK,
            dsparkConfidenceSchedule: resolved.dsparkConfidenceSchedule,
            ropeScalingType: resolved.ropeScalingType,
            ropeScalingFactor: resolved.ropeScalingFactor,
            gdnPrefill: resolved.gdnPrefill,
            evictPrefillScratch: resolved.evictPrefillScratch,
            prefillChunkSize: resolved.prefillChunkSize,
            maxConcurrentSessions: resolved.maxConcurrentSessions,
            lsaConfig: resolved.lsaConfig,
            cacheConfig: cacheConfig,
            host: resolved.host, port: resolved.port, temperature: resolved.temperature,
            topP: resolved.topP, topK: resolved.topK, maxTokens: resolved.maxTokens,
            modelMaxLength: resolved.modelMaxLength,
            repetitionPenalty: resolved.repetitionPenalty, seed: resolved.seed,
            enableThinking: resolved.enableThinking, reasoningEffort: resolved.reasoningEffort,
            systemPrompt: resolved.systemPrompt,
            loadVision: resolved.loadVision,
            allowLocalImagePaths: resolved.allowLocalImagePaths,
            allowRemoteImageURLs: resolved.allowRemoteImageURLs,
            decisionModelPackPath: resolved.decisionModelPackPath,
            decisionComputeUnit: resolved.decisionComputeUnit,
            decisionLoadVision: resolved.decisionLoadVision,
            decisionDefaultScoreReadout: resolved.decisionDefaultScoreReadout,
            decisionDefaultTruncation: resolved.decisionDefaultTruncation,
            decisionAlwaysEmitBox: resolved.decisionAlwaysEmitBox,
            sourceSummary: resolved.sourceSummary
        )
        actorConfig.pinModel = resolved.pinModel
        actorConfig.engineTuning = resolved.engineTuning
        let actor = QWEngineLifecycleActor(config: actorConfig, logger: logger, configPath: config)

        do {
            try QWEngineInstanceLock.acquireEngineLock()
        } catch {
            logger.log(.error, "\(error)")
            FileHandle.standardError.write("qwg serve: \(error)\n".data(using: .utf8)!)
            throw ExitCode(1)
        }

        let cleanRepos = repos.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
        let cleanDocs = docs.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
        actor.configureAllowedWorkspaceRoots(cleanRepos + cleanDocs)
        if let primaryRepo = cleanRepos.first {
            do {
                _ = try await actor.mountMemoryPack(at: primaryRepo)
            } catch {
                logger.log(.warn, "Primary .qwcache mount skipped for '\(primaryRepo)' (\(error)); mounting as Vault v2 pack.")
                _ = try? await actor.mountAuxiliaryVaultPack(at: primaryRepo, autoIndexIfMissing: true)
            }
            for extraRepo in cleanRepos.dropFirst() {
                do {
                    _ = try await actor.mountAuxiliaryVaultPack(at: extraRepo, autoIndexIfMissing: true)
                } catch {
                    logger.log(.warn, "Failed to mount auxiliary --repo '\(extraRepo)': \(error)")
                }
            }
        }
        for docDir in cleanDocs {
            do {
                _ = try await actor.mountAuxiliaryVaultPack(at: docDir, autoIndexIfMissing: true)
            } catch {
                logger.log(.warn, "Failed to mount --docs '\(docDir)': \(error)")
            }
        }
        if cleanRepos.isEmpty && resolved.modelPackPath.isEmpty && resolved.decisionModelPackPath.isEmpty {
            logger.log(.warn, "No model pack specified. Starting server in idle lifecycle mode (select model via settings or API).")
        } else if !resolved.lazy {
            do {
                if !resolved.modelPackPath.isEmpty && !actor.isLoaded() {
                    logger.log(.info, "Eager loading enabled: preloading primary model...")
                    _ = try await actor.loadExplicit()
                }
                if !resolved.decisionModelPackPath.isEmpty && !actor.isDecisionSidecarLoaded() {
                    logger.log(.info, "Eager loading enabled: preloading decision model '\(resolved.decisionModelPackPath)'...")
                    let unit = QWDecisionComputeUnit(rawValue: resolved.decisionComputeUnit.lowercased()) ?? .auto
                    _ = try await actor.loadDecisionSidecar(
                        packPath: resolved.decisionModelPackPath,
                        computeUnit: unit,
                        loadVision: resolved.decisionLoadVision
                    )
                }
            } catch {
                QWEngineInstanceLock.releaseEngineLock()
                FileHandle.standardError.write("qwg serve: failed to eager load model: \(error)\n".data(using: .utf8)!)
                throw ExitCode(1)
            }
        } else if cleanRepos.isEmpty {
            let label = resolved.modelPackPath.isEmpty ? resolved.decisionModelPackPath : resolved.modelName
            logger.log(.info, "Lazy loading active: model '\(label)' will load on first request.")
        }

        let server = QWHttpServer(
            actor: actor,
            host: resolved.host,
            port: resolved.port,
            logger: logger,
            apiKey: resolved.apiKey,
            corsAllowedOrigins: resolved.corsAllowedOrigins,
            enableAutoUpdateChecks: !noProfile
        )
        do {
            try server.start()
        } catch {
            QWEngineInstanceLock.releaseEngineLock()
            FileHandle.standardError.write("qwg serve: failed to start HTTP listener on \(resolved.host):\(resolved.port): \(error)\n".data(using: .utf8)!)
            throw ExitCode(1)
        }

        let pidPath = QWPaths.pidFilePath
        try? "\(ProcessInfo.processInfo.processIdentifier)".write(toFile: pidPath, atomically: true, encoding: .utf8)

        var parentSource: DispatchSourceProcess? = nil
        if let ppid = parentPid, ppid > 0 {
            let src = DispatchSource.makeProcessSource(identifier: ppid, eventMask: .exit, queue: DispatchQueue.global(qos: .utility))
            src.setEventHandler {
                logger.log(.info, "Parent process (PID \(ppid)) exited. Shutting down qwg serve gracefully...")
                QWEngineInstanceLock.releaseEngineLock()
                Darwin.exit(0)
            }
            src.resume()
            parentSource = src
        }

        func gracefulExit() {
            logger.log(.info, "Stopping HTTP server and cleaning up...")
            server.stop()
            QWEngineInstanceLock.releaseEngineLock()
            parentSource?.cancel()
            Darwin.exit(0)
        }

        signal(SIGPIPE, SIG_IGN)
        signal(SIGINT, SIG_IGN)
        signal(SIGTERM, SIG_IGN)
        let sigintSrc = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
        let sigtermSrc = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
        sigintSrc.setEventHandler { gracefulExit() }
        sigtermSrc.setEventHandler { gracefulExit() }
        sigintSrc.resume()
        sigtermSrc.resume()

        // Park forever without leaking a continuation (empty-closure
        // withCheckedContinuation trips the Swift 6.4 continuation-misuse
        // trap); signal sources stay serviced while this task is suspended.
        try? await Task.sleep(nanoseconds: .max)
    }
}
