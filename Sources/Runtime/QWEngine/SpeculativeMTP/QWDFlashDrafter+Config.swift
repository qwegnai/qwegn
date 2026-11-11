// QWDFlashDrafter+Config.swift — the DFlash 2 config and the session
// slot types the drafter allocates.
import Foundation
import QWAlloc
import QWCore
import QWQuant

public struct QWDFlashConfig: Sendable, Equatable {
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
        bridgeCodeSize: Int = 1024, bridgeRank: Int = 512, bridgeEps: Float = 1e-6
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
    }

    public static let defaultDSpinePackPath: String =
        "\(FileManager.default.homeDirectoryForCurrentUser.path)/.qwg/models/Qwen3.8-27B-DSpine-b16-hq4g32"

    public static let defaultPackPath: String = {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let dflash = "\(home)/.qwg/models/Qwen3.8-27B-DFlash2-hq4g32"
        let dspine = "\(home)/.qwg/models/Qwen3.8-27B-DSpine-b16-hq4g32"
        if !FileManager.default.fileExists(atPath: dflash) && FileManager.default.fileExists(atPath: dspine) {
            return dspine
        }
        return dflash
    }()

    /// Validates that `targetDescriptor` is compatible with DFlash 2 / DSpine and that the drafter pack
    /// at `packPath` exists, has valid configuration, and contains the required weight tensors.
    public static func validateDrafter(
        targetDescriptor: QWModelDescriptor,
        packPath: String?,
        drafts: Int = 7,
        selectorTopK: Int? = nil
    ) throws {
        guard targetDescriptor.hcCount == nil && targetDescriptor.numExperts == nil else {
            throw QWEngineDriverError.state(
                "DFlash2 drafter is only compatible with dense single-stream target models (selected model is \(targetDescriptor.family.rawValue))"
            )
        }
        let rawPath = (packPath ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        let effectivePath = (rawPath.isEmpty ? Self.defaultPackPath : rawPath).expandingTildePath
        let resolvedURL = effectivePath.canonicalFileURL
        guard FileManager.default.fileExists(atPath: resolvedURL.path) else {
            throw QWEngineDriverError.packMissing(resolvedURL.path)
        }
        let cfg = Self.load(from: resolvedURL, topKOverride: selectorTopK)
        let maxDrafts = max(7, cfg.blockSize - 1)
        guard (1...maxDrafts).contains(drafts) else {
            throw QWEngineDriverError.state(
                "DFlash2 drafter requires mtpDrafts in 1...\(maxDrafts) (got \(drafts))"
            )
        }
        guard cfg.hiddenSize == targetDescriptor.hidden else {
            throw QWEngineDriverError.state(
                "DFlash2 drafter hiddenSize (\(cfg.hiddenSize)) does not match target (\(targetDescriptor.hidden))"
            )
        }
        guard cfg.vocabSize == targetDescriptor.vocab else {
            throw QWEngineDriverError.state(
                "DFlash2 drafter vocabSize (\(cfg.vocabSize)) does not match target (\(targetDescriptor.vocab))"
            )
        }
        guard cfg.targetLayerIds.count == cfg.numLayers,
              cfg.targetLayerIds.allSatisfy({ (0..<targetDescriptor.numLayers).contains($0) }) else {
            throw QWEngineDriverError.state(
                "DFlash2 drafter targetLayerIds \(cfg.targetLayerIds) out of range for \(targetDescriptor.numLayers)-layer target"
            )
        }
        let pack = try QWQuant.Pack(url: resolvedURL, descriptor: .qwen38_27b, skipIdentityCheck: true)
        var required = ["fc.weight", "hidden_norm.weight", "norm.weight", "layers.0.self_attn.q_proj.weight"]
        if cfg.isDSpine {
            required += ["selected_prefix_bridge.codebook", "selected_prefix_bridge.read_weight",
                         "selected_prefix_bridge.feature_write_weights"]
        }
        for name in required {
            _ = try pack.entry(name)
        }
    }

    /// True when `packURL` holds a DFlash 2 or DSpine drafter `config.json`.
    public static func isDrafterPack(at packURL: URL) -> Bool {
        let root = QWModelRoot(url: packURL)
        guard let cfgURL = root.configURL,
              let data = try? Data(contentsOf: cfgURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return false
        }
        if json["dflash_config"] != nil || json["selected_prefix_code_bridge"] != nil {
            return true
        }
        if let mt = (json["model_type"] as? String)?.lowercased(),
           mt.contains("dflash") || mt.contains("dspine") {
            return true
        }
        return false
    }

    /// Infers `"dspine"` vs `"dflash2"` and maximum draft count (`blockSize - 1`) from the drafter pack's `config.json`.
    public static func inferBackendAndMaxDrafts(packPath: String?) -> (backend: String, maxDrafts: Int) {
        let rawPath = (packPath ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        let effectivePath = (rawPath.isEmpty ? Self.defaultPackPath : rawPath).expandingTildePath
        let url = effectivePath.canonicalFileURL
        guard FileManager.default.fileExists(atPath: url.path) else {
            return ("dflash2", 7)
        }
        let cfg = Self.load(from: url)
        return (cfg.isDSpine ? "dspine" : "dflash2", max(7, min(15, cfg.blockSize - 1)))
    }

    public static func load(from packURL: URL, topKOverride: Int? = nil) -> QWDFlashConfig {
        let dir = packURL.pathExtension == "qwp" ? packURL.deletingLastPathComponent() : packURL
        let cfgURL = dir.appendingPathComponent("assets/config.json")
        guard let data = try? Data(contentsOf: cfgURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return QWDFlashConfig(selectorTopK: topKOverride ?? 16)
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
        return QWDFlashConfig(
            hiddenSize: (json["hidden_size"] as? Int) ?? 5120,
            numLayers: (json["num_hidden_layers"] as? Int) ?? 5,
            numAttentionHeads: (json["num_attention_heads"] as? Int) ?? 32,
            numKeyValueHeads: (json["num_key_value_heads"] as? Int) ?? 8,
            headDim: (json["head_dim"] as? Int) ?? 128,
            intermediateSize: (json["intermediate_size"] as? Int) ?? 17408,
            vocabSize: (json["vocab_size"] as? Int) ?? 248320,
            rmsNormEps: Float((json["rms_norm_eps"] as? Double) ?? 1e-6),
            ropeTheta: Float((json["rope_theta"] as? Double) ?? (rope["rope_theta"] as? Double) ?? 10_000_000.0),
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
            bridgeEps: Float((bridgeDict?["eps"] as? Double) ?? 1e-6)
        )
    }
}

struct QWDFlashSessionSlot {
    let kSlab: QWTensorValue
    let vSlab: QWTensorValue
    let kLayers: [QWTensorValue]
    let vLayers: [QWTensorValue]
}

