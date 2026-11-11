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
    @Option(name: .customLong("dflash-window"), help: "DFlash 2 / DSpine window B in 2...16 (auto-detects dflash2 vs dspine from drafter pack and sets --mtp-drafts B-1).") var dflashWindow: Int? = nil
    @Option(name: .customLong("repo"), help: "Repository or knowledge directory root containing .qwg/manifest.json to mount as LSA memory.") var repo: String? = nil
    @Flag(name: [.customLong("no-memory-pack"), .customLong("no-memory")], help: "Disable automatic mounting of .qwg/manifest.json memory pack.") var noMemory: Bool = false
    @Flag(name: .customLong("no-profile"), help: "Bypass ~/.qwg/settings.json and per-model knob profiles.") var noProfile: Bool = false

    /// A knob the user set that this command cannot honour is refused here,
    /// where the error can name the flag, instead of being ignored later.
    func validate() throws {
        try knobs.requireHonoured("qwg repl")
    }

    func run() throws {
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
        let effectiveChunkSize = QWPrefillLimits.clampedRows(knobs.prefillChunkSize ?? effSettings.prefillChunkSize)
        let resolvedCacheDirURL = cliCfg.resolvedCacheDirURL
        let warmBytes = Int((knobs.warmCacheLimit ?? 4.0) * 1024 * 1024 * 1024)
        let ssdEnabled = knobs.ssdCacheLimit != nil ? (knobs.ssdCacheLimit! > 0) : effSettings.ssdCacheEnabled
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
                : nil
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

        let allowVision = knobs.loadVision ?? true
        let visionController: QWCLIVisionController? = (allowVision && engine.capabilities.contains(.vision) && runtimeAssets.descriptor.visDepth > 0)
            ? (try? QWCLIVisionController(modelPath: resolvedPack))
            : nil
        var pendingTurnImage: String? = allowVision ? image : nil
        var activeImageHistory: [String] = []

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
                let parts = trimmed.split(separator: " ", maxSplits: 1).map(String.init)
                let cmd = parts[0].lowercased()
                let arg = parts.count > 1 ? parts[1].trimmingCharacters(in: .whitespaces) : ""
                switch cmd {
                case "/exit", "/quit", "/q":
                    print("Goodbye!")
                    break replLoop
                case "/clear", "/reset":
                    conversation?.reset()
                    history.removeAll()
                    activeImageHistory.removeAll()
                    recalledVaultPaths.removeAll()
                    visionController?.unload()
                    if let s = currentSystem, !s.isEmpty {
                        history.append(.object([("role", .string("system")), ("content", .string(s))]))
                    }
                    print("\(cliColor.systemNotice)[Conversation history cleared.]\(cliColor.reset)\n")
                    continue
                case "/image":
                    pendingTurnImage = arg.isEmpty ? nil : arg
                    print("\(cliColor.systemNotice)[Next turn image: \(pendingTurnImage ?? "none")]\(cliColor.reset)\n")
                    continue
                case "/help", "/h", "/?":
                    print("""
                    \(cliColor.bold)Commands:\(cliColor.reset)
                      /help, /h               Show this help message
                      /clear, /reset          Clear conversation history
                      /image [path]           Attach an image to the next turn
                      /system [prompt]        View or update system prompt (empty to clear)
                      /thinking [on|off]      Toggle or set thinking mode
                      /effort [low|med|xhigh] Set reasoning effort
                      /tokens [N]             Set max generated tokens per turn
                      /stats                  Toggle performance statistics display
                      /cache                  Show prefix cache memory and SSD usage
                      /memory [mount|unmount] Inspect, mount, or unmount LSA codebase memory pack
                      /history                Show conversation turn count
                      /exit, /quit, /q        Exit REPL

                    """)
                    continue
                case "/memory", "/repo":
                    if arg.lowercased() == "unmount" {
                        engine.unmountSharedBasePrefixes()
                        activeMemoryPack = nil
                        mountedMemoryTokens = []
                        recalledVaultPaths.removeAll()
                        print("\(cliColor.systemNotice)[Memory pack unmounted.]\(cliColor.reset)\n")
                    } else if arg.lowercased().hasPrefix("mount ") {
                        let targetPath = String(arg.dropFirst("mount ".count)).trimmingCharacters(in: .whitespaces)
                        if let resolvedMem = QWResolvedCliMemoryPack.resolve(
                            explicitRepoPath: targetPath, disableMemoryPack: false,
                            modelName: runtimeAssets.descriptor.family.rawValue,
                            geometryDigest: runtimeAssets.descriptor.geometryDigestV2Hex,
                            preferredKVMode: String(describing: cliCfg.resolvedKVMode)
                        ), resolvedMem.artifact.kvMode.lowercased() == String(describing: cliCfg.resolvedKVMode).lowercased(),
                           let mounted = try? engine.mountStandalonePrefixCache(from: resolvedMem.cacheFileURL) {
                            activeMemoryPack = resolvedMem
                            mountedMemoryTokens = mounted.tokens
                            recalledVaultPaths.removeAll()
                            print("\(cliColor.systemNotice)[Mounted '\(resolvedMem.manifest.packName)' (\(mounted.tokens.count) tokens, kv=\(resolvedMem.artifact.kvMode.uppercased())).]\(cliColor.reset)\n")
                        } else {
                            print("\(cliColor.systemNotice)[Could not mount memory pack at '\(targetPath)' (check .qwg/manifest.json and matching KV mode '\(cliCfg.resolvedKVMode)').]\(cliColor.reset)\n")
                        }
                    } else if let mem = activeMemoryPack {
                        let delta = QWMemoryPack.computeDelta(for: mem.manifest)
                        let cortexCount = mem.manifest.files.filter(\.inCortex).count
                        let vaultCount = mem.manifest.files.count - cortexCount
                        print("\(cliColor.bold)Active LSA Memory Pack:\(cliColor.reset) \(mem.manifest.packName) (\(mem.manifest.rootPath))")
                        print("  Files / Tokens : \(mem.manifest.files.count) files (\(cortexCount) cortex + \(vaultCount) vault), \(mountedMemoryTokens.count) cortex tokens (enforced KV: \(mem.artifact.kvMode.uppercased()))")
                        print("  Vault Recalled : \(recalledVaultPaths.count) file(s) paged into session")
                        print("  Live Delta     : \(delta.modifiedFiles.count) modified, \(delta.deletedFiles.count) deleted\n")
                    } else {
                        print("\(cliColor.dim)No LSA memory pack mounted. Use '/memory mount <path>' or run 'qwg index'.\(cliColor.reset)\n")
                    }
                    continue
                case "/cache":
                    let cacheState = engine.prefixCache
                    if cacheState.isConfigured {
                        print("\(cliColor.bold)Prefix Cache Status:\(cliColor.reset)")
                        print("  Warm RAM : \(QWStatFormat.mebibytes(cacheState.warmBytes)) (\(cacheState.warmEntries.map(String.init) ?? "unavailable") entries)")
                        print("  Cold SSD : \(QWStatFormat.mebibytes(cacheState.ssdBytes)) (\(cacheState.ssdEntries.map(String.init) ?? "unavailable") entries)")
                    } else {
                        print("\(cliColor.dim)Prefix cache is not enabled.\(cliColor.reset)")
                    }
                    print("")
                    continue
                case "/system":
                    conversation?.reset()
                    if arg.isEmpty {
                        currentSystem = nil
                        if let first = history.first, first.get("role").isString("system") { history.removeFirst() }
                        print("\(cliColor.systemNotice)[System prompt cleared.]\(cliColor.reset)\n")
                    } else {
                        currentSystem = arg
                        if let first = history.first, first.get("role").isString("system") {
                            history[0] = .object([("role", .string("system")), ("content", .string(arg))])
                        } else {
                            history.insert(.object([("role", .string("system")), ("content", .string(arg))]), at: 0)
                        }
                        print("\(cliColor.systemNotice)[System prompt updated.]\(cliColor.reset)\n")
                    }
                    continue
                case "/thinking":
                    conversation?.reset()
                    if arg.lowercased() == "on" || arg.lowercased() == "true" { isThinking = true }
                    else if arg.lowercased() == "off" || arg.lowercased() == "false" { isThinking = false }
                    else { isThinking.toggle() }
                    displayThinking = isThinking && showThinking && !hideThinking
                    print("\(cliColor.systemNotice)[Thinking \(isThinking ? "enabled" : "disabled").]\(cliColor.reset)\n")
                    continue
                case "/effort":
                    conversation?.reset()
                    let validEfforts = ["low", "medium", "xhigh"]
                    if validEfforts.contains(arg.lowercased()) {
                        currentReasoningEffort = arg.lowercased()
                        print("\(cliColor.systemNotice)[Reasoning effort set to '\(currentReasoningEffort)'.]\(cliColor.reset)\n")
                    } else {
                        print("\(cliColor.systemNotice)[Usage: /effort low|medium|xhigh (current: \(currentReasoningEffort))]\(cliColor.reset)\n")
                    }
                    continue
                case "/tokens":
                    if let n = Int(arg), n > 0 {
                        currentMaxTokens = n
                        print("\(cliColor.systemNotice)[Max generated tokens set to \(n).]\(cliColor.reset)\n")
                    } else {
                        let cur = currentMaxTokens.map(String.init) ?? (isThinking ? "512 (default)" : "128 (default)")
                        print("\(cliColor.systemNotice)[Usage: /tokens <positive integer> (current: \(cur))]\(cliColor.reset)\n")
                    }
                    continue
                case "/stats":
                    showStats.toggle()
                    print("\(cliColor.systemNotice)[Performance statistics \(showStats ? "enabled" : "disabled").]\(cliColor.reset)\n")
                    continue
                case "/history":
                    let turns = history.filter { $0.get("role").isString("user") }.count
                    print("\(cliColor.systemNotice)[History contains \(turns) user turn(s), total \(history.count) message(s).]\(cliColor.reset)\n")
                    continue
                default:
                    print("\(cliColor.systemNotice)Unknown command '\(cmd)'. Type '/help' for available commands.\(cliColor.reset)\n")
                    continue
                }
            }

            let turnImage = pendingTurnImage
            pendingTurnImage = nil
            var userContent: String
            if let img = turnImage {
                activeImageHistory.append(img)
                userContent = trimmed.contains("<|image_pad|>") ? trimmed : "<|vision_start|><|image_pad|><|vision_end|>" + trimmed
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
                        if case .string(let s) = history[removeIdx].get("content"),
                           s.contains("<|image_pad|>"), !activeImageHistory.isEmpty {
                            activeImageHistory.removeFirst()
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
                let mmTurn = try visionController?.prepareTurn(
                    promptTokens: promptIds,
                    imagePaths: activeImageHistory,
                    currentTurnHasImage: turnImage != nil,
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

// MARK: - Bench

struct Bench: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Ratchet gate: re-verify the m1 soak artifact + baselines (m1 --check); other names race Scripts/bench binaries.")
    @Argument(help: "Bench name: 'm1' (the M1 soak/ratchet gate) or a Scripts/bench/* component name.") var component: String = "m1"
    @Flag(name: .customLong("check"), help: "m1: re-verify newest soak log + ratchet vs baselines (CI merge gate; exit 1 on FAIL). For components: 3%-regression check after the run.") var check: Bool = false
    @Option(name: .customLong("soak-dir"), help: "Directory holding m1_soak_*.jsonl (default: cwd).") var soakDir: String?
    @Option(name: .customLong("baselines"), help: "Baseline JSON (default: Scripts/bench/baselines.json).") var baselines: String?
    @Flag(name: .customLong("run"), help: "Component benches: execute Scripts/bench/run_sweep.sh first (authoritative, GPU-quiet); default is re-verify only.") var runSweep: Bool = false

    func run() throws {
        guard component != "m1", component != "full" else {
            try runM1Gate()
            return
        }
        let fm = FileManager.default
        let repo = fm.currentDirectoryPath
        let script = repo + "/Scripts/bench/run_sweep.sh"
        guard fm.fileExists(atPath: script) else {
            FileHandle.standardError.write("qwg bench \(component): no run_sweep.sh at \(script)\n".data(using: .utf8)!)
            throw ExitCode(2)
        }
        if !runSweep {
            FileHandle.standardError.write("qwg bench \(component): component benches run via `Scripts/bench/run_sweep.sh \(component)` (GPU-quiet, operator; add --run to invoke from here)\n".data(using: .utf8)!)
            throw ExitCode(0)
        }
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/bin/bash")
        p.arguments = [script, component]
        p.standardOutput = FileHandle.standardOutput
        p.standardError = FileHandle.standardError
        try p.run(); p.waitUntilExit()
        throw ExitCode(p.terminationStatus)
    }

    func runM1Gate() throws {
        let dir = URL(fileURLWithPath: soakDir ?? FileManager.default.currentDirectoryPath)
        let baselineURL = URL(fileURLWithPath: baselines
            ?? FileManager.default.currentDirectoryPath + "/Scripts/bench/baselines.json")
        guard let log = QWCli.newestSoakLog(in: dir) else {
            FileHandle.standardError.write("[bench m1] FAIL no m1_soak_*.jsonl in \(dir.path) — run `make m1` first; the artifact IS the gate\n".data(using: .utf8)!)
            throw ExitCode(1)
        }
        let samples = try QWCli.SoakSample.load(log)
        let verdict = QWCli.verifySoak(samples: samples)
        for l in verdict.lines { print(l) }
        var pass = verdict.allPass
        if FileManager.default.fileExists(atPath: baselineURL.path) {
            let base = try QWCli.Baselines.load(baselineURL)
            let rows = QWCli.ratchetCheck(current: verdict.metrics, baseline: base)
            for l in QWCli.ratchetLines(rows) { print(l) }
            if !rows.allSatisfy({ $0.ok }) { pass = false }
        } else {
            print("[bench m1] SKIP ratchet (no \(baselineURL.lastPathComponent); m1 --ratchet-update writes it on the first authoritative pass)")
        }
        print(pass ? "[bench m1] VERDICT: PASS — soak verified, ratchet green"
                   : "[bench m1] VERDICT: FAIL")
        if !pass { throw ExitCode(1) }
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
    @Option(name: .customLong("model-name"), help: "Model name reported in /v1/models.") var modelName: String?
    @Option(name: .customLong("model-alias"), help: "Model alias accepted in /v1/chat/completions.") var modelAlias: String?
    @Option(name: .customLong("dflash-window"), help: "DFlash 2 / DSpine window B in 2...16 (auto-detects dflash2 vs dspine from drafter pack and sets --mtp-drafts B-1).") var dflashWindow: Int?
    @Option(name: .customLong("repo"), help: "Mount a repository/knowledge memory pack (<path>/.qwg/manifest.json) at server startup.") var repo: String? = nil
    @Option(name: .customLong("parent-pid"), help: "Parent process PID to monitor for automatic termination.") var parentPid: Int32? = nil
    @Flag(name: .customLong("no-profile"), help: "Bypass ~/.qwg/settings.json and per-model knob profiles.") var noProfile: Bool = false

    /// A knob the user set that this command cannot honour is refused here,
    /// where the error can name the flag, instead of being ignored later.
    func validate() throws {
        try knobs.requireHonoured("qwg serve")
    }

    func run() async throws {
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
            cliGDNPrefill: knobs.gdnPrefill, cliEvictScratch: knobs.evictPrefillScratch,
            cliPrefillChunkSize: knobs.prefillChunkSize,
            cliMaxConcurrentSessions: knobs.maxConcurrentSessions,
            cliLSAMode: knobs.lsaMode, cliLSABudget: knobs.lsaBudget,
            cliLSAClusterBlock: knobs.lsaClusterBlock, cliLSASidecarPath: knobs.lsaSidecarPath,
            cliWarmRAMLimitGB: knobs.warmCacheLimit,
            cliSSDCacheLimitGB: knobs.ssdCacheLimit, cliCacheDir: knobs.cacheDir,
            cliLogLevel: knobs.logLevel, cliLazy: knobs.lazy,
            cliLoadVision: knobs.loadVision, cliStreamNgrams: knobs.streamNgrams,
            cliMemoryPolicy: knobs.memoryPolicy, cliOSReserveGiB: knobs.osReserve,
            cliResidentExpertsPerLayer: knobs.residentExpertsPerLayer, cliMoESpecBudget: knobs.moeSpecBudget,
            cliIgnoreProfile: noProfile
        )
        let isDFlash2 = ["dflash2", "dflash", "dspine"].contains(resolved.mtpBackend?.lowercased() ?? "")
        let maxDrafts = isDFlash2 ? 15 : 5
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
                }
            } catch {
                FileHandle.standardError.write("qwg serve: error: \(error)\n".data(using: .utf8)!)
                throw ExitCode(2)
            }
        }

        let logger = QWServerLogger.shared
        logger.setLevel(QWLogLevel.fromString(resolved.logLevel))

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
            sourceSummary: resolved.sourceSummary
        )
        actorConfig.engineTuning = resolved.engineTuning
        let actor = QWEngineLifecycleActor(config: actorConfig, logger: logger, configPath: config)

        do {
            try QWEngineInstanceLock.acquireEngineLock()
        } catch {
            logger.log(.error, "\(error)")
            FileHandle.standardError.write("qwg serve: \(error)\n".data(using: .utf8)!)
            throw ExitCode(1)
        }

        let trimmedRepo = repo?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if !trimmedRepo.isEmpty {
            do {
                _ = try await actor.mountMemoryPack(at: trimmedRepo)
            } catch {
                logger.log(.warn, "Failed to mount --repo '\(trimmedRepo)' at startup: \(error)")
            }
        } else if resolved.modelPackPath.isEmpty {
            logger.log(.warn, "No model pack specified. Starting server in idle lifecycle mode (select model via settings or API).")
        } else if !resolved.lazy {
            logger.log(.info, "Eager loading enabled: preloading model...")
            do {
                _ = try await actor.loadExplicit()
            } catch {
                QWEngineInstanceLock.releaseEngineLock()
                FileHandle.standardError.write("qwg serve: failed to eager load model: \(error)\n".data(using: .utf8)!)
                throw ExitCode(1)
            }
        } else {
            logger.log(.info, "Lazy loading active: model '\(resolved.modelName)' will load on first inference request.")
        }

        let server = QWHttpServer(actor: actor, host: resolved.host, port: resolved.port, logger: logger)
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
