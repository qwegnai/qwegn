// qwg.swift — Command-line entry point and `plan`, `inspect`, `quantize`, `pull`, `run` verbs.
// Maps CLI options directly to `QWCli` and the engine module that serves the
// selected pack, so model families execute seamlessly based on user selection.

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


@main
struct Qwg: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "qwg",
        abstract: "QWEngine — native Swift+Metal inference for Qwen3.8-27B and Qwen3.8-Flash-Next.",
        version: QWVersion.current,
        subcommands: [Init.self, Inspect.self, Plan.self, Quantize.self, LsaConvert.self, Index.self, Pull.self, Run.self, Repl.self, Bench.self, Serve.self, Telemetry.self])
}

// MARK: - Init

struct Init: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "init",
        abstract: "Create or rebuild the Python environment in ~/Library/Application Support/Qwegn/venv/.")
    @Flag(name: .customLong("python"), help: "Create or rebuild the Python environment for quantize.") var python: Bool = false

    func validate() throws {
        guard python else {
            throw ValidationError("Pass --python to create or rebuild the Python environment.")
        }
    }

    func run() throws {
        _ = try QWCli.ensurePythonEnvironment(supportRoot: Quantize.repoRoot(), rebuild: true)
    }
}

// MARK: - Plan

struct Plan: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Admission report: price model/experts/PLE/KV/GDN state vs the working-set cap; ADMIT/REFUSE with exact bytes.")
    @OptionGroup var knobs: QWGKnobOptions
    @Option(name: .customLong("config"), help: "Model config.json (descriptor source).") var config: String
    @Option(name: .customLong("scratch-mb"), help: "Per-sequence scratch estimate in MiB (caller-owned number).") var scratchMB: Int = 0
    @Option(name: .customLong("cap-gb"), help: "Working-set cap GiB (measured; default = physical RAM as an upper bound).") var capGB: Double?

    /// A knob the user set that this command cannot honour is refused here,
    /// where the error can name the flag, instead of being ignored later.
    func validate() throws {
        try knobs.requireHonoured("qwg plan")
    }

    func run() throws {
        guard let contextTokens = knobs.contextTokens else {
            FileHandle.standardError.write("qwg plan: error: --context-tokens (or --context) is required\n".data(using: .utf8)!)
            throw ExitCode(64)
        }
        print(try QWCli.admissionPlanReport(
            configURL: URL(fileURLWithPath: config),
            contextTokens: contextTokens, kvMode: knobs.kvMode ?? "q4",
            scratchMB: scratchMB, capGB: capGB))
    }
}

// MARK: - Inspect

struct Inspect: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Dump pack provenance + tensor table; optionally dequant one tensor to .npy.")
    @Argument(help: "Model directory or legacy .qwp pack.") var pack: String
    @Option(name: .customLong("dequant"), help: "Tensor name → write dequantized f32 .npy.") var dequant: String?
    @Option(name: .customLong("out"), help: "Override the .npy output path (with --dequant).") var out: String?

    func run() throws {
        let url = URL(fileURLWithPath: pack)
        if let t = dequant {
            let dst = out.map { URL(fileURLWithPath: $0) }
                ?? url.deletingPathExtension().appendingPathExtension("npy")
            try QWCli.dequantToNpy(packURL: url, tensor: t, out: dst)
            print("wrote \(dst.path)")
            return
        }
        print(try QWCli.inspectReport(packURL: url))
    }
}

// MARK: - Quantize

struct Quantize: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Shell out to the proven Python packer (Scripts/quantize/pack.py) — Swift never reimplements the quantizer.")
    @Argument(help: "Source model directory (holds *.safetensors).") var model: String
    @Option(name: .customLong("out"), help: "New output model directory (model.qwp and assets/).") var outPack: String
    @Option(name: .customLong("quant"), help: "Quantization format (q4g, hq{2,3,4,6,8}_g{32,64,128,256}, shvq{1,15,2,3,4}_g{32,64,128,256}).") var quant: String = "q4g"
    @Option(name: .customLong("dense-quant"), help: "Optional override for 2D dense/shared tensors.") var denseQuant: String?
    @Option(name: .customLong("expert-quant"), help: "Optional override for routed MoE gate/up tensors.") var expertQuant: String?
    @Option(name: .customLong("expert-down-quant"), help: "Optional override for routed MoE down_proj tensors.") var expertDownQuant: String?
    @Option(name: .customLong("ngram-quant"), help: "Optional PLE ngram table format (q8_g32, vq2_g32, or vq4_g32).") var ngramQuant: String?

    func run() throws {
        let lowerQ = quant.lowercased()
        guard lowerQ == "q4g" || lowerQ == "hq4_g32_m5"
            || lowerQ.hasPrefix("hq") || lowerQ.hasPrefix("shvq") else {
            throw ValidationError("Unsupported quant '\(quant)'; Q8 mirrors are not used by the engine.")
        }
        let repoRoot = Self.repoRoot()
        let py = try QWCli.ensurePythonEnvironment(supportRoot: repoRoot)
        let scripts = repoRoot.appendingPathComponent("Scripts")
        let plan = QWCli.quantizePlan(repo: model, outPack: URL(fileURLWithPath: outPack),
                                      quant: quant, scriptsDir: scripts, pythonURL: py,
                                      denseQuant: denseQuant, expertQuant: expertQuant,
                                      expertDownQuant: expertDownQuant, ngramQuant: ngramQuant)
        print("provenance: \(plan.provenanceLine)")
        let p = Process()
        p.executableURL = URL(fileURLWithPath: plan.command[0])
        p.arguments = Array(plan.command.dropFirst())
        let pipe = Pipe(); p.standardOutput = pipe; p.standardError = pipe
        try p.run()
        let h = pipe.fileHandleForReading
        while true {
            let d = h.availableData
            if d.isEmpty { break }
            FileHandle.standardOutput.write(d)
        }
        p.waitUntilExit()
        if p.terminationStatus != 0 { throw ExitCode(p.terminationStatus) }
    }

    static func repoRoot() -> URL {
        QWPaths.supportDirectoryURL
    }
}

// MARK: - Pull

struct Pull: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Parallel ranged download of a checkpoint from the HF hub (plan math is QWCli-downloadPlan-tested).")
    @Argument(help: "hf repo id, e.g. org/model.") var model: String
    @Option(name: .customLong("folder"), help: "Local destination directory (default: ~/.qwg/models).") var folder: String = QWPaths.modelsDirectoryPath
    @Option(name: .customLong("parts"), help: "Parallel range chunks per file.") var parts: Int = 4

    func run() throws {
        let dest = URL(fileURLWithPath: (folder as NSString).expandingTildeInPath, isDirectory: true)
        let saved = try QWCli.pullCheckpoint(repo: model, folder: dest, parts: parts)
        print("ready: \(saved.path)")
    }
}

// MARK: - Run

struct Run: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "One-shot greedy generation on a model directory, alias, or legacy .qwp pack.")
    @OptionGroup var knobs: QWGKnobOptions
    @Argument(help: "Model directory, name in ~/.qwg/models, alias, or legacy .qwp pack.") var pack: String? = nil
    @Option(name: [.customLong("model"), .customShort("m")], help: "Model directory, name in ~/.qwg/models, or alias from settings.") var modelOption: String? = nil
    @Option(name: .customLong("prompt"), help: "User prompt text (optional if --repl is specified).") var prompt: String?
    @Option(name: .customLong("prompt-file"), help: "Path to a file containing the user prompt text.") var promptFile: String?
    @Option(name: .customLong("image"), help: "Path to image input for multimodal vision prompt.") var image: String?
    @Option(name: .customLong("video"), help: "Path to video input for multimodal vision prompt.") var video: String?
    @Flag(name: .customLong("show-thinking"), inversion: .prefixedNo, help: "Show or hide thinking tokens in output.") var showThinking: Bool = true
    @Flag(name: .customLong("hide-thinking"), help: "Hide thinking tokens in output.") var hideThinking: Bool = false
    @Flag(name: .customLong("stream"), inversion: .prefixedNo, help: "Stream tokens to stdout as they are generated.") var stream: Bool = true
    @Flag(name: .customLong("stop-think"), help: "Stop generation immediately when thinking ends (at </think>).") var stopThink: Bool = false
    @Option(name: .customLong("stop"), help: "Additional comma-separated stop token IDs or token strings.") var stopTokens: String?
    @Flag(name: .customLong("repl"), help: "Start interactive multi-turn REPL mode.") var repl: Bool = false
    @Flag(name: .customLong("color"), inversion: .prefixedNo, help: "Colorize thinking output and prompts (default: auto).") var color: Bool?
    @Flag(name: .customLong("verbose"), help: "Print a performance statistics block after the output (load, prefill/decode tok/s, TTFT, p50/p95 step latency, arena).") var verbose: Bool = false
    @Option(name: .customLong("dflash-window"), help: "DFlash 2 / DSpine window B in 2...16 (auto-detects dflash2 vs dspine from drafter pack and sets --mtp-drafts B-1).") var dflashWindow: Int? = nil
    @Option(name: .customLong("repo"), help: "Repository or knowledge directory root containing .qwg/manifest.json to mount as LSA memory.") var repo: String? = nil
    @Flag(name: [.customLong("no-memory-pack"), .customLong("no-memory")], help: "Disable automatic mounting of .qwg/manifest.json memory pack.") var noMemory: Bool = false
    @Flag(name: .customLong("no-profile"), help: "Bypass ~/.qwg/settings.json and per-model knob profiles (hermetic benchmark mode).") var noProfile: Bool = false

    /// A knob the user set that this command cannot honour is refused here,
    /// where the error can name the flag, instead of being ignored later.
    func validate() throws {
        try knobs.requireHonoured("qwg run")
    }

    func run() throws {
        let settings = QWSettingsManager.shared.loadSettings()
        let (resolvedPack, initialMtpDrafts, effectiveMtpBackend, resolvedBackend) =
            try QWResolvedCliEngineConfig.resolvePackAndMTP(
                verb: "qwg run",
                modelOption: modelOption,
                pack: pack,
                knobs: knobs,
                dflashWindow: dflashWindow,
                settings: settings
            )
        let resolvedPromptText: String?
        if let pf = promptFile {
            resolvedPromptText = try String(contentsOfFile: pf.expandingTildePath, encoding: .utf8)
        } else {
            resolvedPromptText = prompt
        }
        if repl || resolvedPromptText == nil {
            var replCmd = Repl()
            replCmd.pack = resolvedPack
            replCmd.modelOption = modelOption
            replCmd.image = image
            replCmd.video = video
            replCmd.showThinking = showThinking
            replCmd.hideThinking = hideThinking
            replCmd.verbose = verbose
            replCmd.stream = stream
            replCmd.color = color
            replCmd.dflashWindow = dflashWindow
            replCmd.repo = repo
            replCmd.noMemory = noMemory
            replCmd.noProfile = noProfile
            // The window already decided the drafts and the backend, so the
            // repl is handed the decision rather than the window. A knob the
            // user did not set stays unset, not a default wearing their clothes.
            var replKnobs = knobs
            if dflashWindow != nil {
                replKnobs.mtpDrafts = initialMtpDrafts
                replKnobs.mtpBackend = effectiveMtpBackend
            }
            replCmd.knobs = replKnobs
            try replCmd.run()
            return
        }

        guard let userPrompt = resolvedPromptText else {
            FileHandle.standardError.write("qwg run: missing --prompt (or use --repl for interactive mode)\n".data(using: .utf8)!)
            throw ExitCode(2)
        }

        let isThinking = knobs.enableThinking ?? true
        let displayThinking = isThinking && showThinking && !hideThinking
        let cliColor = QWCli.Color.auto(force: color)

        let requestedMaxNew = knobs.maxTokens
        if let req = requestedMaxNew, req <= 0 {
            FileHandle.standardError.write("qwg run: error: max generated tokens must be positive (got \(req))\n".data(using: .utf8)!)
            throw ExitCode(2)
        }
        let cliCfg = try QWResolvedCliEngineConfig.resolveAssetsAndKnobs(
            verb: "qwg run",
            resolvedPack: resolvedPack,
            knobs: knobs,
            dflashWindow: dflashWindow,
            initialMtpDrafts: initialMtpDrafts,
            effectiveMtpBackend: effectiveMtpBackend,
            resolvedBackend: resolvedBackend,
            image: image,
            video: video,
            settings: settings,
            validateDefaultContextImmediately: false,
            repoPath: repo,
            disableMemoryPack: noMemory || (repo == nil),
            ignoreProfile: noProfile
        )
        let runtimeAssets = cliCfg.runtimeAssets
        let defaultContext = cliCfg.defaultContext
        let effectiveMaxNew = requestedMaxNew ?? (isThinking ? 512 : 128)
        let tok = cliCfg.tokenizer

        var msgs: [QWJson] = []
        if let s = knobs.systemPrompt, !s.isEmpty {
            msgs.append(.object([("role", .string("system")), ("content", .string(s))]))
        }
        var effectiveUserPrompt = userPrompt
        if let memPack = cliCfg.memoryPack {
            let vaultRecall = QWMemoryVault.recallVaultContext(
                query: userPrompt,
                manifest: memPack.manifest
            )
            if !vaultRecall.isEmpty {
                effectiveUserPrompt += vaultRecall.recallBlock
            }
        }
        let allowVision = knobs.loadVision ?? true
        let hasImageInput = image != nil
        if hasImageInput && !allowVision {
            FileHandle.standardError.write("qwg run: error: image input provided but vision loading disabled via --no-load-vision\n".data(using: .utf8)!)
            throw ExitCode(2)
        }
        let formattedUserPrompt = (image != nil && !effectiveUserPrompt.contains("<|image_pad|>"))
            ? "<|vision_start|><|image_pad|><|vision_end|>" + effectiveUserPrompt
            : effectiveUserPrompt
        msgs.append(.object([("role", .string("user")), ("content", .string(formattedUserPrompt))]))

        let effectiveEffort = ChatCompletionRequest.normalizeReasoningEffort(knobs.reasoningEffort, fallback: "xhigh")

        var tplOpts = QWChatTemplateOptions()
        tplOpts.addGenerationPrompt = true
        tplOpts.enableThinking = .bool(isThinking)
        if isThinking {
            tplOpts.reasoningEffort = .string(effectiveEffort)
        }
        let wire = try runtimeAssets.chatTemplate.render(messages: msgs, options: tplOpts)
        let rawPromptIds = try tok.encode(wire)
        var promptIds: [Int]
        let visualSpans: [QWVisualSpan]
        if let imgPath = image, allowVision {
            let mm = try QWImageDecoder.prepareOneShotMultimodalTurn(
                promptTokens: rawPromptIds, imagePath: imgPath,
                modelPath: resolvedPack, hiddenDim: runtimeAssets.descriptor.hidden
            )
            promptIds = mm.tokens
            visualSpans = mm.visualSpans
        } else {
            promptIds = rawPromptIds
            visualSpans = []
        }
        let precomputedDelta = cliCfg.memoryPack?.encodeDeltaTokens(chatTemplate: runtimeAssets.chatTemplate, tokenizer: tok)
        let memoryReserve = cliCfg.memoryPack.map { $0.artifact.tokenCount + (precomputedDelta?.tokens.count ?? 0) + 8192 } ?? 0
        let prefixCacheOn = knobs.prefixCache ?? (cliCfg.memoryPack != nil)
        let requiredCapacity = promptIds.count + memoryReserve + effectiveMaxNew + (prefixCacheOn ? 512 : 64)
        let pageAlignedRequired = max(4096, ((requiredCapacity + 255) / 256) * 256)
        let oneShotDefaultContext = min(defaultContext, pageAlignedRequired)
        let effectiveContext = max(knobs.contextTokens ?? oneShotDefaultContext, requiredCapacity)
        if knobs.contextTokens == nil {
            do {
                try QWCli.validateContextBudget(tokens: effectiveContext)
            } catch {
                FileHandle.standardError.write("qwg run: error: \(error)\n".data(using: .utf8)!)
                throw ExitCode(2)
            }
        }

        let resolvedEviction: QWPrefillScratchEvictionPolicy
        do {
            resolvedEviction = try QWPrefillScratchEvictionPolicy.parse(knobs.evictPrefillScratch ?? "auto")
        } catch {
            FileHandle.standardError.write("qwg run: error: \(error)\n".data(using: .utf8)!)
            throw ExitCode(2)
        }

        let specialTokens = cliCfg.specialTokens
        var stops: Set<Int> = specialTokens.defaultStopTokenIds
        if let extraStops = stopTokens {
            for s in extraStops.split(separator: ",") {
                let trimmed = s.trimmingCharacters(in: .whitespaces)
                if let id = Int(trimmed) {
                    stops.insert(id)
                } else if let encoded = try? tok.encode(trimmed), encoded.count == 1 {
                    stops.insert(encoded[0])
                } else {
                    FileHandle.standardError.write("qwg run: warning: stop token '\(trimmed)' could not be resolved to a single token ID\n".data(using: .utf8)!)
                }
            }
        }

        var streamDecoder = QWStreamDecoder(tok)
        var inThinking = isThinking
        var reachedAnswer = !isThinking
        var answerStarted = !isThinking

        if stream && inThinking && displayThinking {
            FileHandle.standardOutput.write(cliColor.thinking.data(using: .utf8)!)
        }

        let onToken: (Int) -> Bool = { (id: Int) -> Bool in
            if id == specialTokens.endThinkTokenId {
                inThinking = false
                reachedAnswer = true
                if stream {
                    _ = streamDecoder.flush()
                    if displayThinking {
                        FileHandle.standardOutput.write((cliColor.reset + "\n\n").data(using: .utf8)!)
                    }
                }
                return !stopThink
            }
            if stream {
                if inThinking {
                    if displayThinking,
                       let piece = try? streamDecoder.decode(id, skipSpecialTokens: true),
                       !piece.isEmpty {
                        FileHandle.standardOutput.write(piece.data(using: .utf8)!)
                    }
                } else {
                    reachedAnswer = true
                    if let piece = try? streamDecoder.decode(id, skipSpecialTokens: true), !piece.isEmpty {
                        if !answerStarted {
                            let trimmed = piece.drop(while: { $0.isNewline })
                            if !trimmed.isEmpty {
                                answerStarted = true
                                FileHandle.standardOutput.write(String(trimmed).data(using: .utf8)!)
                            }
                        } else {
                            FileHandle.standardOutput.write(piece.data(using: .utf8)!)
                        }
                    }
                }
            }
            return true
        }

        let effSettings = cliCfg.effectiveSettings
        let configuredChunk = QWPrefillLimits.clampedRows(knobs.prefillChunkSize ?? effSettings.prefillChunkSize)
        let minChunkRows = cliCfg.memoryPack != nil ? 1024 : 64
        let promptAlignedChunk = max(minChunkRows, ((promptIds.count + 63) / 64) * 64)
        let effectiveChunkSize = knobs.prefillChunkSize != nil
            ? configuredChunk
            : min(configuredChunk, promptAlignedChunk)
        let samplingParams = cliCfg.makeSamplingParameters(knobs: knobs, maxTokens: effectiveMaxNew)

        let resolvedCacheDirURL = cliCfg.resolvedCacheDirURL
        let loadT0 = Date()
        let gdnPrefill = knobs.gdnPrefill ?? "chunked"
        guard ["naive", "chunked", "regres"].contains(gdnPrefill.lowercased()) else {
            FileHandle.standardError.write("qwg run: invalid --gdn-prefill mode '\(gdnPrefill)' (must be 'chunked', 'naive', or 'regres')\n".data(using: .utf8)!)
            throw ExitCode(2)
        }
        // The engine is built by the module that serves this pack: the
        // CLI names no engine type, and it does not pick a path.
        let options = cliCfg.makeBuildOptions(
            knobs: knobs,
            maxTokens: effectiveContext,
            prefillChunkSize: effectiveChunkSize,
            prefixCacheDirectory: prefixCacheOn ? resolvedCacheDirURL : nil,
            cacheConfig: prefixCacheOn
                ? QWCacheConfig(warmByteLimit: 4 << 30,
                                ssdByteLimit: effSettings.ssdCacheEnabled ? Int(effSettings.ssdCacheLimitGB * 1024 * 1024 * 1024) : 0,
                                cacheDirectory: resolvedCacheDirURL, wipeOnStart: false)
                : nil,
            prefillScratchEviction: resolvedEviction,
            gdnPrefill: gdnPrefill
        )
        do {
            try QWEngineInstanceLock.acquireEngineLock()
        } catch {
            FileHandle.standardError.write("qwg run: error: \(error)\n".data(using: .utf8)!)
            throw ExitCode(1)
        }
        defer { QWEngineInstanceLock.releaseEngineLock() }
        let engine = try QWEngine(options: options, descriptor: runtimeAssets.descriptor)
        let loadSec = Date().timeIntervalSince(loadT0)
        if let memPack = cliCfg.memoryPack, visualSpans.isEmpty {
            let mountedTokens: [Int]
            if prefixCacheOn {
                let mounted = try engine.mountStandalonePrefixCache(from: memPack.cacheFileURL)
                mountedTokens = mounted.tokens
            } else {
                let rootURL = URL(fileURLWithPath: memPack.manifest.rootPath)
                let files = try QWMemoryPack.discoverFiles(in: rootURL, includeTests: memPack.manifest.includeTests)
                let built = try QWMemoryPack.buildEnvelope(
                    packName: memPack.manifest.packName,
                    rootPath: rootURL.path,
                    files: files,
                    clusterBlockTokens: memPack.manifest.clusterBlockTokens,
                    kvMode: memPack.artifact.kvMode,
                    maxTokens: QWMemoryVault.defaultCortexTokenBudget,
                    tokenCounter: { try tok.encode($0).count }
                )
                let envWire = try Index.renderMemoryEnvelopeWire(
                    systemContent: built.systemMessageContent,
                    chatTemplate: runtimeAssets.chatTemplate
                )
                mountedTokens = try tok.encode(envWire)
            }
            let delta = precomputedDelta ?? memPack.encodeDeltaTokens(chatTemplate: runtimeAssets.chatTemplate, tokenizer: tok)
            let guardDecision = QWMemoryVault.applyElasticBudgetGuard(
                mountedMemoryTokens: mountedTokens,
                deltaTokens: delta.tokens,
                agentPromptTokens: promptIds.count,
                maxNewTokens: effectiveMaxNew,
                maxContext: effectiveContext
            )
            promptIds = guardDecision.prefixTokens + promptIds
            let modeLabel = prefixCacheOn ? "mounted memory pack" : "cold-prefill memory envelope"
            FileHandle.standardError.write(
                "qwg run: \(modeLabel) '\(memPack.manifest.packName)' (\(mountedTokens.count) tokens, kv=\(memPack.artifact.kvMode), delta=\(delta.delta.modifiedFiles.count) modified).\n".data(using: .utf8)!
            )
        }
        if stopThink { stops.insert(specialTokens.endThinkTokenId) }
        let specParams = cliCfg.makeSpeculativeParameters(knobs: knobs)
        let samplerConfig = QWSamplerConfig(samplingParams, vocabulary: runtimeAssets.descriptor.vocab)
        let request = QWGenerationRequest(
            prompt: promptIds,
            maxNew: effectiveMaxNew,
            stopIDs: stops,
            sampling: samplerConfig,
            usePrefixCache: prefixCacheOn,
            speculative: specParams,
            collectTimings: verbose,
            visualSpans: visualSpans
        )
        let result = try engine.generate(request: request, onToken: stream ? onToken : nil)
        // The turn ended on the end-of-thinking marker: the consumer
        // decides what to do with it.
        var stoppedAtThinkingEnd = false
        if let stopID = result.stopTokenID, stopID == specialTokens.endThinkTokenId {
            stoppedAtThinkingEnd = true
            _ = onToken(stopID)
        }
        var ids = result.tokens
        if stoppedAtThinkingEnd && !stream { ids.append(specialTokens.endThinkTokenId) }
        try Self.finishOutput(ids: ids, tok: tok, streamDecoder: &streamDecoder, stream: stream,
                              isThinking: isThinking, displayThinking: displayThinking,
                              inThinking: inThinking, reachedAnswer: reachedAnswer,
                              answerStarted: answerStarted, cliColor: cliColor,
                              endThinkTokenId: specialTokens.endThinkTokenId)
        let report = engine.report
        let allocs = report.steadyAllocations.map(String.init) ?? "unavailable"
        FileHandle.standardError.write("(\(ids.count) tok gen, steady allocs \(allocs))\n".data(using: .utf8)!)
        if verbose {
            let profileState = noProfile ? "bypassed" : (cliCfg.profileApplied ? "applied" : "default")
            let cfgLine = "[config] profile=\(profileState) kv_mode=\(cliCfg.resolvedKVMode) lsa_mode=\(cliCfg.lsaConfig.mode.rawValue) mtp_drafts=\(cliCfg.effectiveMtpDrafts)"
            Self.printRunVerboseStats(result: result, report: report,
                                      promptTokens: promptIds.count, generatedTokens: ids.count,
                                      loadSec: loadSec, cacheDir: resolvedCacheDirURL,
                                      configLog: cfgLine)
        }
    }

    static func finishOutput(ids: [Int], tok: QWTokenizer, streamDecoder: inout QWStreamDecoder,
                             stream: Bool, isThinking: Bool, displayThinking: Bool,
                             inThinking: Bool, reachedAnswer: Bool, answerStarted: Bool,
                             cliColor: QWCli.Color,
                             endThinkTokenId: Int = QWModelGeometry.qwen38_27b.endThinkTokenId) throws {
        if stream {
            let tail = streamDecoder.flush()
            if !tail.isEmpty && (!inThinking || displayThinking) {
                if !answerStarted {
                    let trimmed = tail.drop(while: { $0.isNewline })
                    if !trimmed.isEmpty {
                        FileHandle.standardOutput.write(String(trimmed).data(using: .utf8)!)
                    }
                } else {
                    FileHandle.standardOutput.write(tail.data(using: .utf8)!)
                }
            }
            FileHandle.standardOutput.write((cliColor.reset + "\n").data(using: .utf8)!)
            if isThinking && !reachedAnswer && !displayThinking {
                FileHandle.standardError.write("qwg run: note: \(QWTokenRouter.thinkingBudgetNote)\n".data(using: .utf8)!)
            }
        } else if isThinking && !displayThinking {
            if QWTokenRouter.split(ids, endThinkTokenId: endThinkTokenId).hasThinkingEnd {
                let ansText = QWTokenRouter.text(
                    QWTokenRouter.split(ids, endThinkTokenId: endThinkTokenId).answer, tok: tok)
                if !ansText.isEmpty { print(ansText) }
            } else {
                FileHandle.standardError.write("qwg run: note: \(QWTokenRouter.thinkingBudgetNote)\n".data(using: .utf8)!)
            }
        } else if isThinking && displayThinking {
            let split = QWTokenRouter.split(ids, endThinkTokenId: endThinkTokenId)
            if split.hasThinkingEnd {
                let thinkText = QWTokenRouter.text(split.thinking, tok: tok)
                let ansText = QWTokenRouter.text(split.answer, tok: tok)
                let formattedThink = thinkText.isEmpty ? "" : (cliColor.thinking + thinkText + cliColor.reset)
                if formattedThink.isEmpty {
                    if !ansText.isEmpty { print(ansText) }
                } else if ansText.isEmpty {
                    print(formattedThink)
                } else {
                    print(formattedThink + "\n\n" + ansText)
                }
            } else {
                print(cliColor.thinking + (try tok.decode(ids, skipSpecialTokens: true).trimmingCharacters(in: .whitespacesAndNewlines)) + cliColor.reset)
            }
        } else {
            print(try tok.decode(ids, skipSpecialTokens: true).trimmingCharacters(in: .whitespacesAndNewlines))
        }
    }

    /// The statistics block after a run. The frame belongs to the CLI;
    /// the engine's own story comes from its report.
    static func printRunVerboseStats(result: QWGenerationResult, report: QWEngineReport,
                                     promptTokens: Int, generatedTokens: Int,
                                     loadSec: Double, cacheDir: URL,
                                     configLog: String? = nil) {
        let steps = result.stepMs.sorted()
        let stepLatency = QWStatFormat.stepLatency(steps)
        let decodeTokS = QWStatFormat.tokensPerSecond(tokens: max(0, generatedTokens - 1),
                                                       seconds: result.decodeSeconds)
        let prefillTokS = QWStatFormat.tokensPerSecond(tokens: promptTokens, seconds: result.prefillSeconds)
        let prefillGPUTokS = QWStatFormat.tokensPerSecond(tokens: promptTokens,
                                                          seconds: result.prefillGPUSeconds)
        let footprint = QWProcessMemory.physicalFootprintBytes()
        var cacheLine = ""
        if let cache = result.cache {
            cacheLine = cache.hit
                ? "prefix cache:            HIT (\(cache.matchedTokens)/\(cache.totalTokens) tokens)\n"
                : "prefix cache:            MISS (saved to \(cacheDir.path))\n"
        }
        let rounds = result.stepTokenCounts
        var roundLine = ""
        if !rounds.isEmpty {
            let mean = Double(rounds.reduce(0, +)) / Double(rounds.count)
            roundLine = "speculative rounds:      \(rounds.count) (mean \(String(format: "%.2f", mean)) emitted tokens/round)\n"
        }
        let allocs = report.steadyAllocations.map { "\($0) (law: 0)" } ?? "unavailable"
        let cfgLine = configLog.map { "\($0)\n" } ?? ""
        let block = """
        === qwg run statistics ===
        \(report.capacityLog)
        \(cfgLine)pack load (weights→GPU): \(String(format: "%.1f", loadSec)) s
        resident memory (RAM):   \(String(format: "%.2f", Double(footprint) / 1_073_741_824)) GiB
        arena reserved:          \(String(format: "%.2f", Double(report.arenaBytes) / 1_073_741_824)) GiB
        weights used/reserved:   \(String(format: "%.2f", Double(report.weightUsedBytes) / 1_073_741_824)) / \(String(format: "%.2f", Double(report.weightReservedBytes) / 1_073_741_824)) GiB
        prompt tokens:           \(promptTokens)
        generated tokens:        \(generatedTokens)
        \(cacheLine)prefill:                 \(QWStatFormat.seconds(result.prefillSeconds))  (\(QWStatFormat.rate(prefillTokS)), GPU: \(QWStatFormat.seconds(result.prefillGPUSeconds)) / \(QWStatFormat.rate(prefillGPUTokS)))
        TTFT (first token):      \(QWStatFormat.seconds(result.firstTokenSeconds))
        decode:                  \(QWStatFormat.seconds(result.decodeSeconds, "%.1f"))  (\(QWStatFormat.rate(decodeTokS, "%.2f")))
        \(roundLine)step latency:            \(stepLatency)
        steady-state allocs:     \(allocs)
        \(report.detailText)
        """
        FileHandle.standardError.write(block.data(using: .utf8)!)
    }
}

// MARK: - LsaConvert

struct LsaConvert: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "lsa-convert",
        abstract: "Generate or import a .qlsa (QLSA0001) Latent Sparse Attention projection sidecar for a model directory.")

    @Argument(help: "Model directory, name in ~/.qwg/models, alias, or .qwp pack.") var model: String
    @Option(name: .customLong("method"), help: "Conversion method: svd (zero-data weight SVD), pca (activation PCA), distill (ridge distillation), or import.") var method: String = "svd"
    @Option(name: .customLong("out"), help: "Output .qlsa sidecar file path (default: <model_dir>/model.qlsa).") var out: String?
    @Option(name: .customLong("calibration-file"), help: "Text file for activation calibration (with --method pca or --method distill).") var calibrationFile: String?
    @Option(name: .customLong("import-path"), help: "External safetensors/npy file to import (with --method import).") var importPath: String?
    @Option(name: .customLong("index-dim"), help: "Latent index dimension d_idx (default: 64).") var indexDim: Int = QWLSAConfig.defaultIndexDim
    @Option(name: .customLong("recommended-budget"), help: "Recommended LSA key budget k (default: 2048).") var recommendedBudget: Int = QWLSAConfig.defaultBudget

    func run() throws {
        let settings = QWSettingsManager.shared.loadSettings()
        let resolvedPack = settings.resolveModelReference(model).packPath
        let targetPath = resolvedPack.isEmpty ? model.expandingTildePath : resolvedPack
        let supportRoot = Quantize.repoRoot()
        let scriptPath = supportRoot.appendingPathComponent("Scripts/convert_lsa_sidecar.py").path
        if FileManager.default.fileExists(atPath: scriptPath) && method.lowercased() != "svd-native" {
            let py = try QWCli.ensurePythonEnvironment(supportRoot: supportRoot)
            var args = [scriptPath, "--model", targetPath, "--method", method.lowercased(),
                        "--index-dim", "\(indexDim)", "--recommended-budget", "\(recommendedBudget)"]
            if let out { args += ["--out", out.expandingTildePath] }
            if let calibrationFile { args += ["--calibration-file", calibrationFile.expandingTildePath] }
            if let importPath { args += ["--import-path", importPath.expandingTildePath] }
            let p = Process()
            p.executableURL = py
            p.arguments = args
            p.standardOutput = FileHandle.standardOutput
            p.standardError = FileHandle.standardError
            try p.run()
            p.waitUntilExit()
            if p.terminationStatus != 0 { throw ExitCode(p.terminationStatus) }
            return
        }
        // Native Accelerate LAPACK SVD synthesis fallback for .qwp packs and .safetensors checkpoints
        let assets = try QWModelRuntimeAssets.load(packPath: targetPath)
        let desc = assets.descriptor
        let layerTensors = try QWLSASidecar.synthesizeFromModel(packPath: targetPath, descriptor: desc, indexDim: indexDim)
        let root = QWModelRoot(url: URL(fileURLWithPath: targetPath))
        let outURL = out.map { URL(fileURLWithPath: $0.expandingTildePath) }
            ?? root.directory.appendingPathComponent("model.qlsa")
        let prov = QWLSASidecar.Header.Provenance(
            modelId: root.directory.lastPathComponent,
            geometryDigest: desc.geometryDigestV2Hex,
            method: "weight_svd",
            indexDim: indexDim,
            poolSize: QWLSAConfig.poolSize,
            clusterBlockSize: QWLSAConfig.defaultClusterBlock,
            bucketsPerBlock: QWLSAConfig.bucketsPerBlock,
            maxBucketKeys: QWLSAConfig.maxBucketKeys,
            recommendedBudget: recommendedBudget
        )
        let shape = [indexDim, desc.lsaInputDim]
        try QWLSASidecar.write(to: outURL, provenance: prov, layerTensors: layerTensors, wqShape: shape, wkShape: shape)
        print("wrote \(outURL.path) (\(layerTensors.count) full-attention layers, shape=\(shape), method=weight_svd)")
    }
}
