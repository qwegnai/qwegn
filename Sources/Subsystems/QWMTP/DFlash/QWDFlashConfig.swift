// QWDFlashConfig.swift — DFlash 2 / DSpine configuration, error type,
// session KV slot layout, and capability-driven drafter pack discovery.
import Foundation
import QWAlloc
import QWCore
import QWQuant

public enum QWDFlashError: Error, CustomStringConvertible, Sendable {
    case packMissing(String)
    case weightMissing(String, String)
    case state(String)

    public var description: String {
        switch self {
        case .packMissing(let p): return "QWDFlash: pack not found at \(p)"
        case .weightMissing(let name, let detail): return "QWDFlash: weight \(name) invalid (\(detail))"
        case .state(let msg): return msg
        }
    }
}

public struct QWDFlashConfig: Sendable, Equatable {
    public enum AttentionMode: String, Sendable, Equatable {
        case gqa
        case mla
    }

    public enum SelectorSoftcapMode: UInt32, Sendable, Equatable {
        case none = 0
        case beforeCorrection = 1
        case afterCorrection = 2
    }

    public let hiddenSize: Int
    public let numLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let intermediateSize: Int
    public let vocabSize: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let slidingWindow: Int
    public let maxCacheSize: Int
    public let blockSize: Int
    public let maskTokenId: Int
    public let targetLayerIds: [Int]
    public let convKernelSize: Int
    public let convGroupSize: Int
    public let selectorRank: Int
    public let selectorTopK: Int
    public let isDSpine: Bool
    public let draftBlockCausal: Bool
    public let bridgeCodeSize: Int
    public let bridgeRank: Int
    public let bridgeEps: Float
    public let virtualLayerCapture: Bool
    public let finalLogitSoftcapping: Float
    public let selectorSoftcapMode: SelectorSoftcapMode
    public let shiftLabel: Bool

    public let attentionMode: AttentionMode
    public let qLoraRank: Int
    public let kvLoraRank: Int
    public let qkNopeHeadDim: Int
    public let qkRopeHeadDim: Int
    public let vHeadDim: Int
    public let gatedAttention: Bool
    public let useMLAScalingFactor: Bool
    public let gatedNormRank: Int
    public let gatedNormZeroCentered: Bool
    public let layernormGatingWeight: Float
    public let useSandwichNorm: Bool
    public let swigluLimit: Float
    public let ropeYarn: QWModelDescriptor.YarnRopeConfig?
    public let ropeInterleave: Bool
    public let softmaxScale: Float

    public var isMLA: Bool { attentionMode == .mla }

    public var alphaQ: Float {
        guard useMLAScalingFactor, hiddenSize > 0, qLoraRank > 0 else { return 1.0 }
        return Float((Double(hiddenSize) / Double(qLoraRank)).squareRoot())
    }

    public var alphaKV: Float {
        guard useMLAScalingFactor, hiddenSize > 0, kvLoraRank > 0 else { return 1.0 }
        return Float((Double(hiddenSize) / Double(kvLoraRank)).squareRoot())
    }

    public init(
        hiddenSize: Int = 5120, numLayers: Int = 5,
        numAttentionHeads: Int = 32, numKeyValueHeads: Int = 8, headDim: Int = 128,
        intermediateSize: Int = 17408, vocabSize: Int = 248320,
        rmsNormEps: Float = 1e-6, ropeTheta: Float = 10_000_000.0,
        slidingWindow: Int = 2048, blockSize: Int = 8, maskTokenId: Int = 248070,
        targetLayerIds: [Int] = [5, 19, 33, 47, 61],
        convKernelSize: Int = 2, convGroupSize: Int = 16,
        selectorRank: Int = 256, selectorTopK: Int = 16,
        isDSpine: Bool = false, draftBlockCausal: Bool = false,
        bridgeCodeSize: Int = 1024, bridgeRank: Int = 512, bridgeEps: Float = 1e-6,
        virtualLayerCapture: Bool = false,
        finalLogitSoftcapping: Float = 0.0,
        selectorSoftcapMode: SelectorSoftcapMode = .none,
        shiftLabel: Bool = false,
        attentionMode: AttentionMode = .gqa,
        qLoraRank: Int = 0,
        kvLoraRank: Int = 0,
        qkNopeHeadDim: Int = 0,
        qkRopeHeadDim: Int? = nil,
        vHeadDim: Int? = nil,
        gatedAttention: Bool = false,
        useMLAScalingFactor: Bool = false,
        gatedNormRank: Int = 0,
        gatedNormZeroCentered: Bool = false,
        layernormGatingWeight: Float = 2.0,
        useSandwichNorm: Bool = false,
        swigluLimit: Float = 0.0,
        ropeYarn: QWModelDescriptor.YarnRopeConfig? = nil,
        ropeInterleave: Bool? = nil,
        softmaxScale: Float? = nil
    ) {
        self.hiddenSize = hiddenSize; self.numLayers = numLayers
        self.numAttentionHeads = numAttentionHeads; self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim; self.intermediateSize = intermediateSize; self.vocabSize = vocabSize
        self.rmsNormEps = rmsNormEps; self.ropeTheta = ropeTheta; self.slidingWindow = slidingWindow
        self.maxCacheSize = max(1, slidingWindow - 1); self.blockSize = blockSize
        self.maskTokenId = maskTokenId; self.targetLayerIds = targetLayerIds
        self.convKernelSize = convKernelSize; self.convGroupSize = convGroupSize
        self.selectorRank = selectorRank; self.selectorTopK = min(max(1, selectorTopK), 16)
        self.isDSpine = isDSpine; self.draftBlockCausal = draftBlockCausal
        self.bridgeCodeSize = bridgeCodeSize; self.bridgeRank = bridgeRank; self.bridgeEps = bridgeEps
        self.virtualLayerCapture = virtualLayerCapture
        self.finalLogitSoftcapping = finalLogitSoftcapping
        self.selectorSoftcapMode = finalLogitSoftcapping > 0 ? selectorSoftcapMode : .none
        self.shiftLabel = shiftLabel
        self.attentionMode = attentionMode
        self.qLoraRank = qLoraRank
        self.kvLoraRank = kvLoraRank
        self.qkNopeHeadDim = qkNopeHeadDim
        self.qkRopeHeadDim = qkRopeHeadDim ?? (attentionMode == .mla ? max(0, headDim - qkNopeHeadDim) : headDim)
        self.vHeadDim = vHeadDim ?? (attentionMode == .mla ? max(128, qkNopeHeadDim) : headDim)
        self.gatedAttention = gatedAttention
        self.useMLAScalingFactor = useMLAScalingFactor
        self.gatedNormRank = max(0, gatedNormRank)
        self.gatedNormZeroCentered = gatedNormZeroCentered
        self.layernormGatingWeight = layernormGatingWeight
        self.useSandwichNorm = useSandwichNorm
        self.swigluLimit = max(0.0, swigluLimit)
        self.ropeYarn = ropeYarn
        self.ropeInterleave = ropeInterleave ?? (attentionMode == .mla)
        if let softmaxScale {
            self.softmaxScale = softmaxScale
        } else {
            let effDim = attentionMode == .mla ? (qkNopeHeadDim + self.qkRopeHeadDim) : headDim
            let baseScale = effDim > 0 ? (1.0 / Double(effDim).squareRoot()) : 0.0
            let mAll: Double
            if let y = ropeYarn, y.factor > 1.0, y.mscaleAllDim > 0.0 {
                mAll = 0.1 * y.mscaleAllDim * Foundation.log(y.factor) + 1.0
            } else {
                mAll = 1.0
            }
            self.softmaxScale = Float(baseScale * mAll * mAll)
        }
    }

    public struct PrefillCaptureSlice: Sendable, Equatable {
        public let localStart: Int
        public let captureRows: Int
        public let ctxStartPos: Int

        public init(localStart: Int, captureRows: Int, ctxStartPos: Int) {
            self.localStart = localStart
            self.captureRows = captureRows
            self.ctxStartPos = ctxStartPos
        }
    }

    /// Computes the live `[localStart..<localStart+captureRows]` slice intersecting `[windowEnd-maxCacheSize..<windowEnd]`.
    public static func prefillCaptureSlice(
        startPos: Int, rows: Int, windowEnd: Int, maxCacheSize: Int
    ) -> PrefillCaptureSlice? {
        guard rows > 0, maxCacheSize > 0 else { return nil }
        let windowStart = max(0, max(windowEnd, startPos + rows) - maxCacheSize)
        guard startPos + rows > windowStart else { return nil }
        let rawLocal = max(0, windowStart - startPos)
        let rawCount = rows - rawLocal
        guard rawCount > 0 else { return nil }
        let captureRows = min(rawCount, maxCacheSize)
        let localStart = rawLocal + (rawCount - captureRows)
        return PrefillCaptureSlice(
            localStart: localStart, captureRows: captureRows, ctxStartPos: startPos + localStart)
    }

    /// True when capture indices follow virtual `executionPlan` layers (`0..<executionPlan.count`) on looped targets.
    public func usesVirtualCapture(for target: QWModelDescriptor) -> Bool {
        virtualLayerCapture || (target.executionPlan.count > target.numLayers
            && targetLayerIds.contains { $0 >= target.numLayers })
    }

    private static func discoverInstalledDrafterPath(
        in modelsDirectory: URL = URL(fileURLWithPath: QWPaths.modelsDirectoryPath),
        dspineOnly: Bool
    ) -> String {
        let fm = FileManager.default
        guard fm.fileExists(atPath: modelsDirectory.path),
              let entries = try? fm.contentsOfDirectory(
                  at: modelsDirectory, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles]
              ) else {
            return ""
        }
        var dflashMatches: [String] = []
        var dspineMatches: [String] = []
        for url in entries.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            let isDir = (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) ?? false
            guard isDir || url.pathExtension == "qwp", isDrafterPack(at: url) else { continue }
            let cfg = load(from: url)
            if cfg.isDSpine {
                dspineMatches.append(url.path)
            } else if !dspineOnly {
                dflashMatches.append(url.path)
            }
        }
        if dspineOnly { return dspineMatches.first ?? "" }
        if let first = dflashMatches.sorted(by: { lhs, rhs in
            let lr = Self.drafterQuantPreferenceRank(lhs)
            let rr = Self.drafterQuantPreferenceRank(rhs)
            return lr != rr ? lr > rr : lhs < rhs
        }).first {
            return first
        }
        return dspineMatches.first ?? ""
    }

    public static let defaultDSpinePackPath: String =
        discoverInstalledDrafterPath(dspineOnly: true)

    public static let defaultPackPath: String =
        discoverInstalledDrafterPath(dspineOnly: false)

    /// Scans `<target>/`, sibling `<targetPrefix>-DFlash2-*`, and `~/.qwg/models` for a compatible drafter pack.
    public static func defaultPackPath(for target: QWModelDescriptor?, targetPackPath: String? = nil) -> String {
        resolveAutoDrafterPath(
            for: target,
            targetPackPath: targetPackPath,
            modelsDirectory: URL(fileURLWithPath: QWPaths.modelsDirectoryPath)
        )
    }

    private static func isCompatibleConfig(_ cfg: QWDFlashConfig, target: QWModelDescriptor) -> Bool {
        let layerBound = cfg.usesVirtualCapture(for: target)
            ? max(target.numLayers, target.executionPlan.count)
            : target.numLayers
        return cfg.hiddenSize == target.hidden
            && cfg.vocabSize == target.vocab
            && !cfg.targetLayerIds.isEmpty
            && cfg.targetLayerIds.allSatisfy { (0...layerBound).contains($0) }
    }

    private static func stripQuantSuffixTokens(_ dirName: String) -> String {
        let quantPrefixes = ["qwp", "fast", "shvq", "hq", "aq", "vq", "q4", "q8", "fp16", "bf16", "mxfp", "mixed", "nvfp"]
        var parts = dirName.split(separator: "-").map(String.init)
        while let last = parts.last?.lowercased(),
              parts.count > 1,
              quantPrefixes.contains(where: { last.hasPrefix($0) }) {
            parts.removeLast()
        }
        return parts.joined(separator: "-")
    }

    public static func defaultPackPath(
        for target: QWModelDescriptor?,
        modelsDirectory: URL,
        targetPackPath: String? = nil
    ) -> String {
        resolveAutoDrafterPath(for: target, targetPackPath: targetPackPath, modelsDirectory: modelsDirectory)
    }

    /// Resolves nested (`dflash_draft/`, `dflash.qwp`) or sibling (`<prefix>-DFlash2-*`) drafter packs without name hardcoding.
    public static func resolveAutoDrafterPath(
        for target: QWModelDescriptor?,
        targetPackPath: String? = nil,
        modelsDirectory: URL = URL(fileURLWithPath: QWPaths.modelsDirectoryPath)
    ) -> String {
        let fm = FileManager.default
        var targetPrefixLower: String?
        var searchRoots: [URL] = []

        if let rawTarget = targetPackPath?.trimmingCharacters(in: .whitespacesAndNewlines), !rawTarget.isEmpty {
            let tURL = rawTarget.expandingTildePath.canonicalFileURL
            var isDirObj: ObjCBool = false
            let exists = fm.fileExists(atPath: tURL.path, isDirectory: &isDirObj)
            let tDir = ((exists && !isDirObj.boolValue) || tURL.pathExtension == "qwp")
                ? tURL.deletingLastPathComponent()
                : tURL
            let localCandidates = [
                tDir.appendingPathComponent("dflash.qwp"),
                tDir.appendingPathComponent("assets/dflash.qwp"),
                tDir.appendingPathComponent("dflash_draft"),
                tDir.appendingPathComponent("assets/dflash_draft"),
                tDir.appendingPathComponent("drafter")
            ]
            for url in localCandidates where fm.fileExists(atPath: url.path) && isDrafterPack(at: url) {
                let cfg = load(from: url, target: target)
                if let target {
                    if !cfg.isDSpine && Self.isCompatibleConfig(cfg, target: target) {
                        return url.path
                    }
                } else if !cfg.isDSpine {
                    return url.path
                }
            }
            let stripped = stripQuantSuffixTokens(tDir.lastPathComponent).lowercased()
            if !stripped.isEmpty {
                targetPrefixLower = stripped
            }
            let parentDir = tDir.deletingLastPathComponent()
            if fm.fileExists(atPath: parentDir.path) {
                searchRoots.append(parentDir)
            }
        }

        guard let target else { return defaultPackPath }
        if fm.fileExists(atPath: modelsDirectory.path),
           !searchRoots.contains(where: { $0.standardizedFileURL == modelsDirectory.standardizedFileURL }) {
            searchRoots.append(modelsDirectory)
        }

        var prefixDFlash: [String] = []
        var dflashMatches: [String] = []
        var dspineMatches: [String] = []
        var seenPaths = Set<String>()
        for rootDir in searchRoots {
            guard let entries = try? fm.contentsOfDirectory(
                at: rootDir, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles]
            ) else { continue }
            for url in entries.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
                let stdPath = url.standardizedFileURL.path
                guard seenPaths.insert(stdPath).inserted else { continue }
                let isDir = (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) ?? false
                guard isDir || url.pathExtension == "qwp", isDrafterPack(at: url) else { continue }
                let cfg = load(from: url, target: target)
                guard Self.isCompatibleConfig(cfg, target: target) else { continue }
                if cfg.isDSpine {
                    dspineMatches.append(url.path)
                } else if let pfx = targetPrefixLower,
                          url.lastPathComponent.lowercased().hasPrefix(pfx + "-dflash2") {
                    prefixDFlash.append(url.path)
                } else {
                    dflashMatches.append(url.path)
                }
            }
        }
        let rankSort: (String, String) -> Bool = { lhs, rhs in
            let lr = Self.drafterQuantPreferenceRank(lhs)
            let rr = Self.drafterQuantPreferenceRank(rhs)
            return lr != rr ? lr > rr : lhs < rhs
        }
        if let first = prefixDFlash.sorted(by: rankSort).first { return first }
        if let first = dflashMatches.sorted(by: rankSort).first { return first }
        if let first = dspineMatches.first { return first }
        return ""
    }

    /// Ranks compatible drafter packs so pure 4-bit SHVQ precedes HQ4 and mixed sub-4-bit packs.
    public static func drafterQuantPreferenceRank(_ packPath: String) -> Int {
        let lower = URL(fileURLWithPath: packPath).lastPathComponent.lowercased()
        let hasSub4Bit = lower.contains("shvq3") || lower.contains("shvq2") || lower.contains("shvq15")
        if lower.contains("shvq4") && !hasSub4Bit { return 2 }
        return hasSub4Bit ? 0 : 1
    }

    static func decodeDrafterDescriptorOrFallback(
        configData: Data,
        packURL: URL,
        fallback: QWModelDescriptor
    ) throws -> QWModelDescriptor {
        guard (try? JSONSerialization.jsonObject(with: configData) as? [String: Any]) != nil else {
            throw QWDFlashError.state("DFlash2 drafter missing valid config.json at \(packURL.path)")
        }
        do {
            return try QWModelDescriptor.decodeDrafter(configJSON: configData)
        } catch QWModelDescriptorError.unsupportedModelType {
            guard isDrafterPack(at: packURL) else {
                throw QWDFlashError.state("DFlash2 drafter missing valid config.json at \(packURL.path)")
            }
            return fallback
        } catch {
            throw QWDFlashError.state("DFlash2 drafter missing valid config.json at \(packURL.path)")
        }
    }

    /// Validates that `targetDescriptor` is compatible with DFlash 2 / DSpine and that `packPath` is complete.
    public static func validateDrafter(
        targetDescriptor: QWModelDescriptor,
        packPath: String?,
        drafts: Int = 7,
        selectorTopK: Int? = nil
    ) throws {
        let rawPath = (packPath ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        let defaultForTarget = Self.defaultPackPath(for: targetDescriptor)
        let effectivePath = (rawPath.isEmpty ? defaultForTarget : rawPath).expandingTildePath
        guard !effectivePath.isEmpty else {
            throw QWDFlashError.packMissing("no compatible drafter pack found for target")
        }
        let resolvedURL = effectivePath.canonicalFileURL
        guard FileManager.default.fileExists(atPath: resolvedURL.path) else {
            throw QWDFlashError.packMissing(resolvedURL.path)
        }
        let cfg = Self.load(from: resolvedURL, topKOverride: selectorTopK, target: targetDescriptor)
        let maxDrafts = max(7, cfg.blockSize - 1)
        guard (1...maxDrafts).contains(drafts) else {
            throw QWDFlashError.state(
                "DFlash2 drafter requires mtpDrafts in 1...\(maxDrafts) (got \(drafts))"
            )
        }
        guard cfg.hiddenSize == targetDescriptor.hidden else {
            throw QWDFlashError.state(
                "DFlash2 drafter hiddenSize (\(cfg.hiddenSize)) does not match target (\(targetDescriptor.hidden))"
            )
        }
        guard cfg.vocabSize == targetDescriptor.vocab else {
            throw QWDFlashError.state(
                "DFlash2 drafter vocabSize (\(cfg.vocabSize)) does not match target (\(targetDescriptor.vocab))"
            )
        }
        let layerBound = cfg.usesVirtualCapture(for: targetDescriptor)
            ? max(targetDescriptor.numLayers, targetDescriptor.executionPlan.count)
            : targetDescriptor.numLayers
        guard !cfg.targetLayerIds.isEmpty,
              cfg.targetLayerIds.allSatisfy({ (0...layerBound).contains($0) }) else {
            throw QWDFlashError.state(
                "DFlash2 drafter targetLayerIds \(cfg.targetLayerIds) out of range for \(layerBound)-layer target"
            )
        }
        if cfg.isMLA {
            let expectedHeadDim = cfg.qkNopeHeadDim + cfg.qkRopeHeadDim
            guard cfg.headDim > 0, cfg.headDim == expectedHeadDim,
                  cfg.qLoraRank > 0, cfg.kvLoraRank > 0, cfg.vHeadDim > 0 else {
                throw QWDFlashError.state(
                    "DFlash2 MLA drafter invalid dims headDim=\(cfg.headDim) nope=\(cfg.qkNopeHeadDim) rope=\(cfg.qkRopeHeadDim)"
                )
            }
            guard cfg.hiddenSize <= 5120, cfg.hiddenSize % 32 == 0,
                  cfg.gatedNormRank <= 32, cfg.blockSize <= 8,
                  cfg.qLoraRank <= 2048, cfg.qLoraRank % 32 == 0,
                  cfg.kvLoraRank >= 128, cfg.kvLoraRank <= 512, cfg.kvLoraRank % 128 == 0,
                  cfg.qkNopeHeadDim <= 256,
                  cfg.qkRopeHeadDim <= 128, cfg.qkRopeHeadDim % 32 == 0,
                  cfg.vHeadDim % 32 == 0 else {
                throw QWDFlashError.state(
                    "DFlash2 MLA drafter exceeds SRAM ceilings (H=\(cfg.hiddenSize) rank=\(cfg.gatedNormRank) B=\(cfg.blockSize) qLora=\(cfg.qLoraRank) kvLora=\(cfg.kvLoraRank) nope=\(cfg.qkNopeHeadDim) rope=\(cfg.qkRopeHeadDim) v=\(cfg.vHeadDim))"
                )
            }
        } else {
            guard cfg.headDim == 128 else {
                throw QWDFlashError.state("DFlash2 drafter headDim must be 128 (got \(cfg.headDim))")
            }
        }
        guard cfg.vocabSize <= 393_216 else {
            throw QWDFlashError.state("DFlash2 drafter vocabSize \(cfg.vocabSize) exceeds 393216")
        }
        let root = QWModelRoot(url: resolvedURL)
        guard let cfgURL = root.configURL,
              let cfgData = try? Data(contentsOf: cfgURL) else {
            throw QWDFlashError.state("DFlash2 drafter missing valid config.json at \(resolvedURL.path)")
        }
        let drafterDesc = try decodeDrafterDescriptorOrFallback(
            configData: cfgData, packURL: resolvedURL, fallback: targetDescriptor
        )
        var required = ["fc.weight", "hidden_norm.weight", "norm.weight"]
        if cfg.gatedNormRank > 0 {
            required += [
                "hidden_norm.gate_up_projection.weight", "hidden_norm.gate_down_projection.weight",
                "norm.gate_up_projection.weight", "norm.gate_down_projection.weight"
            ]
        }
        if cfg.isMLA {
            required += [
                "layers.0.self_attn.q_a_proj.weight",
                "layers.0.self_attn.kv_a_proj_with_mqa.weight",
                "layers.0.self_attn.kv_b_proj.weight",
                "layers.0.self_attn.o_proj.weight"
            ]
            if cfg.gatedAttention {
                required.append("layers.0.self_attn.gate_proj.weight")
            }
        } else {
            required.append("layers.0.self_attn.q_proj.weight")
        }
        if cfg.isDSpine {
            required += ["selected_prefix_bridge.codebook", "selected_prefix_bridge.read_weight",
                         "selected_prefix_bridge.feature_write_weights"]
        }
        if root.kind == .qwp1 {
            let pack = try QWQuant.Pack(url: resolvedURL, descriptor: drafterDesc, skipIdentityCheck: true)
            for name in required {
                _ = try pack.entry(name)
            }
            if cfg.isMLA {
                let hasSplit = (try? pack.entry("layers.0.self_attn.q_b_nope_proj.weight")) != nil
                    && (try? pack.entry("layers.0.self_attn.q_b_rope_proj.weight")) != nil
                let hasFused = (try? pack.entry("layers.0.self_attn.q_b_proj.weight")) != nil
                guard hasSplit || hasFused else {
                    throw QWDFlashError.weightMissing("layers.0.self_attn.q_b_nope_proj.weight", "missing in QWP1 MLA drafter")
                }
            }
        } else if root.kind == .safetensorsSingle || root.kind == .safetensorsSharded {
            let st = try QWSafeTensorsContainer(url: root.weights)
            for name in required {
                guard st.tensor(name) != nil else {
                    throw QWDFlashError.weightMissing(name, "missing in safetensors drafter")
                }
            }
            if cfg.isMLA {
                let hasFused = st.tensor("layers.0.self_attn.q_b_proj.weight") != nil
                let hasSplit = st.tensor("layers.0.self_attn.q_b_nope_proj.weight") != nil
                    && st.tensor("layers.0.self_attn.q_b_rope_proj.weight") != nil
                guard hasFused || hasSplit else {
                    throw QWDFlashError.weightMissing("layers.0.self_attn.q_b_proj.weight", "missing in safetensors MLA drafter")
                }
            }
        } else {
            throw QWDFlashError.state("DFlash2 drafter unsupported pack kind at \(resolvedURL.path)")
        }
    }

    /// True when `packURL` holds a DFlash 2 or DSpine drafter `config.json`.
    public static func isDrafterPack(at packURL: URL) -> Bool {
        guard FileManager.default.fileExists(atPath: packURL.path) else {
            return false
        }
        return isDrafterPackResolved(at: packURL)
    }

    private static func isDrafterPackResolved(at packURL: URL) -> Bool {
        let root = QWModelRoot(url: packURL)
        guard let cfgURL = root.configURL,
              let data = try? Data(contentsOf: cfgURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return false
        }
        if json["dflash_config"] != nil || json["selected_prefix_code_bridge"] != nil {
            return true
        }
        if let archs = json["architectures"] as? [String],
           archs.contains(where: {
               $0.localizedCaseInsensitiveContains("DraftModel") || $0.localizedCaseInsensitiveContains("DFlash")
           }) {
            return true
        }
        return false
    }

    /// Infers `"dspine"` vs `"dflash2"` and maximum draft count (`blockSize - 1`) from the drafter pack's `config.json`.
    public static func inferBackendAndMaxDrafts(packPath: String?) -> (backend: String, maxDrafts: Int) {
        let rawPath = (packPath ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        let effectivePath = (rawPath.isEmpty ? Self.defaultPackPath : rawPath).expandingTildePath
        guard !effectivePath.isEmpty else {
            return ("dflash2", 7)
        }
        let url = effectivePath.canonicalFileURL
        guard FileManager.default.fileExists(atPath: url.path) else {
            return ("dflash2", 7)
        }
        let cfg = Self.load(from: url)
        return (cfg.isDSpine ? "dspine" : "dflash2", max(7, min(15, cfg.blockSize - 1)))
    }

    public static func load(
        from packURL: URL,
        topKOverride: Int? = nil,
        target: QWModelDescriptor? = nil
    ) -> QWDFlashConfig {
        let dir = packURL.pathExtension == "qwp" ? packURL.deletingLastPathComponent() : packURL
        let resolvedCfgURL: URL? = FileManager.default.fileExists(atPath: packURL.path)
            ? (QWModelRoot(url: packURL).configURL ?? dir.appendingPathComponent("assets/config.json"))
            : nil
        guard let cfgURL = resolvedCfgURL,
              let data = try? Data(contentsOf: cfgURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return QWDFlashConfig(
                hiddenSize: target?.hidden ?? 0,
                numLayers: 0,
                numAttentionHeads: target?.qHeads ?? 0,
                numKeyValueHeads: target?.kvHeads ?? 0,
                headDim: target?.headDim ?? 0,
                intermediateSize: target?.denseIntermediate ?? 0,
                vocabSize: target?.vocab ?? 0,
                targetLayerIds: [],
                selectorTopK: topKOverride ?? 16
            )
        }
        let dflash = json["dflash_config"] as? [String: Any] ?? [:]
        let rope = (json["rope_parameters"] as? [String: Any]) ?? (json["rope_scaling"] as? [String: Any]) ?? [:]
        let bridgeDict = (dflash["selected_prefix_code_bridge"] as? [String: Any])
            ?? (json["selected_prefix_code_bridge"] as? [String: Any])
        let isDSpine = bridgeDict != nil || (json["selected_prefix_code_bridge"] as? Bool == true)
        let refineDict = dflash["selected_prefix_last_write_refine"] as? [String: Any]
        let secondPass = json["code_bridge_second_pass"] as? [String: Any]
        let topK = topKOverride
            ?? (refineDict?["head_topk"] as? Int)
            ?? (secondPass?["head_topk"] as? Int)
            ?? (dflash["selector_top_k"] as? Int)
            ?? 16
        let rmsEps = (json["rms_norm_eps"] as? NSNumber)?.floatValue ?? target?.rmsEps ?? 1e-6
        let ropeTheta = Float((json["rope_theta"] as? NSNumber)?.doubleValue
            ?? (rope["rope_theta"] as? NSNumber)?.doubleValue
            ?? target?.ropeTheta
            ?? 10_000_000.0)
        let softcapVal = (dflash["final_logit_softcapping"] as? NSNumber)?.floatValue
            ?? (json["final_logit_softcapping"] as? NSNumber)?.floatValue
            ?? 0.0
        let softcapRaw = (dflash["selector_softcap_mode"] as? String)
            ?? (json["selector_softcap_mode"] as? String)
        let softcapMode: SelectorSoftcapMode = switch softcapRaw {
        case "before_correction": .beforeCorrection
        case "after_correction": .afterCorrection
        default: softcapVal > 0 ? .beforeCorrection : .none
        }

        let attnModeStr = ((dflash["attention_mode"] as? String) ?? (json["attention_mode"] as? String))?.lowercased()
        let isMLA = (attnModeStr == "mla") || (json["kv_lora_rank"] != nil) || (dflash["kv_lora_rank"] != nil)
        let qLora = (json["q_lora_rank"] as? Int) ?? (dflash["q_lora_rank"] as? Int) ?? 0
        let kvLora = (json["kv_lora_rank"] as? Int) ?? (dflash["kv_lora_rank"] as? Int) ?? 0
        let nopeDim = (json["qk_nope_head_dim"] as? Int) ?? (dflash["qk_nope_head_dim"] as? Int) ?? (isMLA ? 128 : 0)
        let ropeDim = (json["qk_rope_head_dim"] as? Int) ?? (dflash["qk_rope_head_dim"] as? Int) ?? (isMLA ? 64 : 128)
        let vDim = (json["v_head_dim"] as? Int) ?? (dflash["v_head_dim"] as? Int) ?? (isMLA ? 128 : 128)
        let resolvedHeadDim = isMLA
            ? ((json["qk_head_dim"] as? Int) ?? (nopeDim + ropeDim))
            : ((json["head_dim"] as? Int) ?? target?.headDim ?? 128)
        let gatedAttn = (json["gated_attention"] as? Bool) ?? (dflash["gated_attention"] as? Bool) ?? false
        let useMLAScale = (json["use_mla_scaling_factor"] as? Bool)
            ?? (dflash["use_mla_scaling_factor"] as? Bool) ?? false
        let normType = ((json["norm_type"] as? String) ?? (dflash["norm_type"] as? String)) ?? ""
        let normLower = normType.lowercased()
        let gatedNormRank: Int
        if normLower.contains("gatednorm") {
            gatedNormRank = (json["gated_norm_rank"] as? Int) ?? (dflash["gated_norm_rank"] as? Int) ?? 16
        } else {
            gatedNormRank = (json["gated_norm_rank"] as? Int) ?? 0
        }
        let gatedZeroCentered = normLower.contains("zerocentered")
        let lnGatingWeight = (json["layernorm_gating_weight"] as? NSNumber)?.floatValue ?? 2.0
        let useSandwich = ((json["layernorm_type"] as? String)?.lowercased() == "pre_post")
        let swigluLim = (json["swiglu_limit"] as? NSNumber)?.floatValue
            ?? (dflash["swiglu_limit"] as? NSNumber)?.floatValue ?? 0.0

        let ropeType = ((rope["rope_type"] as? String) ?? (rope["type"] as? String)
            ?? (json["rope_type"] as? String))?.lowercased()
        let parsedYarn: QWModelDescriptor.YarnRopeConfig?
        if ropeType == "yarn", let factor = (rope["factor"] as? NSNumber)?.doubleValue, factor > 1.0 {
            let origMax = (rope["original_max_position_embeddings"] as? Int) ?? 32_768
            let betaFast = (rope["beta_fast"] as? NSNumber)?.doubleValue ?? 32.0
            let betaSlow = (rope["beta_slow"] as? NSNumber)?.doubleValue ?? 1.0
            let mscale = (rope["mscale"] as? NSNumber)?.doubleValue ?? 1.0
            let mscaleAll = (rope["mscale_all_dim"] as? NSNumber)?.doubleValue ?? 0.0
            parsedYarn = QWModelDescriptor.YarnRopeConfig(
                factor: factor, originalMaxPositionEmbeddings: origMax,
                betaFast: betaFast, betaSlow: betaSlow, mscale: mscale, mscaleAllDim: mscaleAll
            )
        } else {
            parsedYarn = nil
        }
        let ropeInterleave = (json["rope_interleave"] as? Bool) ?? (rope["rope_interleave"] as? Bool) ?? isMLA

        return QWDFlashConfig(
            hiddenSize: (json["hidden_size"] as? Int) ?? target?.hidden ?? 0,
            numLayers: (json["num_hidden_layers"] as? Int) ?? 5,
            numAttentionHeads: (json["num_attention_heads"] as? Int) ?? target?.qHeads ?? 32,
            numKeyValueHeads: (json["num_key_value_heads"] as? Int) ?? target?.kvHeads ?? (isMLA ? 1 : 8),
            headDim: resolvedHeadDim,
            intermediateSize: (json["intermediate_size"] as? Int) ?? target?.denseIntermediate ?? 0,
            vocabSize: (json["vocab_size"] as? Int) ?? target?.vocab ?? 0,
            rmsNormEps: rmsEps,
            ropeTheta: ropeTheta,
            slidingWindow: (json["sliding_window"] as? Int) ?? (isDSpine ? 8192 : 2048),
            blockSize: (dflash["block_size"] as? Int) ?? (json["block_size"] as? Int) ?? (isDSpine ? 16 : 8),
            maskTokenId: (dflash["mask_token_id"] as? Int) ?? 248070,
            targetLayerIds: (dflash["target_layer_ids"] as? [Int]) ?? (isDSpine ? [1, 16, 31, 46, 61] : [5, 19, 33, 47, 61]),
            convKernelSize: (dflash["conv_kernel_size"] as? Int) ?? (isDSpine ? 0 : 2),
            convGroupSize: (dflash["conv_group_size"] as? Int) ?? 16,
            selectorRank: (dflash["selector_rank"] as? Int) ?? 256,
            selectorTopK: topK,
            isDSpine: isDSpine,
            draftBlockCausal: (dflash["draft_block_causal"] as? Bool) ?? isDSpine,
            bridgeCodeSize: (bridgeDict?["code_size"] as? Int) ?? (json["selected_prefix_code_size"] as? Int) ?? 1024,
            bridgeRank: (bridgeDict?["rank"] as? Int) ?? (json["selected_prefix_rank"] as? Int) ?? 512,
            bridgeEps: (bridgeDict?["eps"] as? NSNumber)?.floatValue ?? 1e-6,
            virtualLayerCapture: (dflash["virtual_layer_capture"] as? Bool)
                ?? (json["virtual_layer_capture"] as? Bool) ?? false,
            finalLogitSoftcapping: softcapVal,
            selectorSoftcapMode: softcapMode,
            shiftLabel: (dflash["shift_label"] as? Bool) ?? (json["shift_label"] as? Bool) ?? false,
            attentionMode: isMLA ? .mla : .gqa,
            qLoraRank: qLora,
            kvLoraRank: kvLora,
            qkNopeHeadDim: nopeDim,
            qkRopeHeadDim: isMLA ? ropeDim : resolvedHeadDim,
            vHeadDim: isMLA ? vDim : resolvedHeadDim,
            gatedAttention: gatedAttn,
            useMLAScalingFactor: useMLAScale,
            gatedNormRank: gatedNormRank,
            gatedNormZeroCentered: gatedZeroCentered,
            layernormGatingWeight: lnGatingWeight,
            useSandwichNorm: useSandwich,
            swigluLimit: swigluLim,
            ropeYarn: parsedYarn,
            ropeInterleave: ropeInterleave
        )
    }
}

public struct QWDFlashSessionSlot: Sendable {
    public let kSlab: QWTensorValue
    public let vSlab: QWTensorValue
    public let kLayers: [QWTensorValue]
    public let vLayers: [QWTensorValue]

    public init(kSlab: QWTensorValue, vSlab: QWTensorValue,
                kLayers: [QWTensorValue], vLayers: [QWTensorValue]) {
        self.kSlab = kSlab
        self.vSlab = vSlab
        self.kLayers = kLayers
        self.vLayers = vLayers
    }
}
