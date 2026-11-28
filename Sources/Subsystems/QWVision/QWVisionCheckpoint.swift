// QWVisionCheckpoint.swift — Safetensors shard & QWP checkpoint reader for ViT.
import Foundation
import QWCore
import QWQuant

public enum QWVisionError: Error, CustomStringConvertible, Equatable {
    case missingTensor(String)
    case shapeMismatch(String)
    case badLayout(String)
    case goldenMissing(String)

    public var description: String {
        switch self {
        case .missingTensor(let s): return "QWVision: checkpoint tensor missing: \(s)"
        case .shapeMismatch(let s): return "QWVision: shape mismatch: \(s)"
        case .badLayout(let s): return "QWVision: layout law violated: \(s)"
        case .goldenMissing(let s): return "QWVision: golden missing: \(s)"
        }
    }
}

/// Safetensors shard adapter backed by `QWSafeTensorsContainer` with `QWVisionError` mapping.
struct QWSafetensors {
    private let container: QWSafeTensorsContainer?
    private let deferredLayoutError: String?

    var tensors: [String: QWSafeTensorsTensor] { container?.tensors ?? [:] }

    init(path: String) throws {
        do {
            self.container = try QWSafeTensorsContainer(url: URL(fileURLWithPath: path))
            self.deferredLayoutError = nil
        } catch QWSafeTensorsError.badHeader(let detail) where detail.hasPrefix("the tensors cover ") {
            self.container = nil
            self.deferredLayoutError = detail
        } catch let err as QWSafeTensorsError {
            throw QWVisionError.badLayout(err.description)
        }
    }

    func f32(_ key: String, want: [Int]?) throws -> [Float] {
        if let deferredLayoutError { throw QWVisionError.badLayout(deferredLayoutError) }
        guard let container, let t = container.tensors[key] else {
            throw QWVisionError.missingTensor(key)
        }
        if let w = want, t.shape != w {
            throw QWVisionError.shapeMismatch("\(key): \(t.shape) ≠ \(w)")
        }
        guard t.dtype == .f16 || t.dtype == .bf16 || t.dtype == .f32 else {
            throw QWVisionError.badLayout("QWVision: unsupported dtype \(t.dtype.rawValue) for '\(key)'")
        }
        do {
            return try container.readFloats(key, expectedElementCount: t.elementCount)
        } catch QWSafeTensorsError.missingTensor(let name) {
            throw QWVisionError.missingTensor(name)
        } catch let err as QWSafeTensorsError {
            throw QWVisionError.badLayout(err.description)
        }
    }

    func shape(_ key: String) -> [Int]? { container?.tensors[key]?.shape }
    func has(_ key: String) -> Bool { container?.tensors[key] != nil }
    func adviseDontNeed() { container?.adviseDontNeed() }
}

/// Multi-shard or `.qwp` view over a model's vision tower weights.
public final class QWVisionCheckpoint: @unchecked Sendable {
    private let lock = NSLock()
    let index: [String: String]
    let externalKeys: [String: (shard: String, fullKey: String)]
    private var shards: [String: QWSafetensors] = [:]
    let dir: String
    let prefix: String
    public let config: QWVisionConfig
    private var pack: QWQuant.Pack?
    let packPrefix: String

    private static let prefixCandidates = [
        "model.vision_tower.",
        "vision_tower.vision_model.",
        "vision_model.",
        "model.visual.",
        "vision.",
        "vision_tower.",
    ]

    private static let externalMergerAliases: [(logical: String, candidates: [String])] = [
        ("merger.norm.weight", ["model.multi_modal_projector.layer_norm.weight", "multi_modal_projector.layer_norm.weight", "multi_modal_projector.norm.weight", "projector.norm.weight"]),
        ("merger.norm.bias", ["model.multi_modal_projector.layer_norm.bias", "multi_modal_projector.layer_norm.bias", "multi_modal_projector.norm.bias", "projector.norm.bias"]),
        ("merger.linear_fc1.weight", ["model.multi_modal_projector.linear_1.weight", "linear_proj.0.weight", "multi_modal_projector.linear_1.weight", "multi_modal_projector.fc1.weight", "projector.fc1.weight", "mm_projector.0.weight"]),
        ("merger.linear_fc1.bias", ["model.multi_modal_projector.linear_1.bias", "linear_proj.0.bias", "multi_modal_projector.linear_1.bias", "multi_modal_projector.fc1.bias", "projector.fc1.bias", "mm_projector.0.bias"]),
        ("merger.linear_fc2.weight", ["model.multi_modal_projector.linear_2.weight", "linear_proj.2.weight", "multi_modal_projector.linear_2.weight", "multi_modal_projector.fc2.weight", "projector.fc2.weight", "mm_projector.2.weight"]),
        ("merger.linear_fc2.bias", ["model.multi_modal_projector.linear_2.bias", "linear_proj.2.bias", "multi_modal_projector.linear_2.bias", "multi_modal_projector.fc2.bias", "projector.fc2.bias", "mm_projector.2.bias"]),
    ]

    private static func detectPrefix(in keys: some Sequence<String>) -> String {
        let all = Array(keys)
        return prefixCandidates.first { pfx in all.contains(where: { $0.hasPrefix(pfx) }) } ?? "model.visual."
    }

    public init(pack packPath: String) throws {
        self.dir = packPath
        self.prefix = "vision_tower."
        self.index = [:]
        let pk = try QWQuant.Pack(url: URL(fileURLWithPath: packPath))
        self.pack = pk
        self.packPrefix = Self.detectPrefix(in: pk.tensorNames)
        let packSet = Set(pk.tensorNames)
        var ext: [String: (shard: String, fullKey: String)] = [:]
        for alias in Self.externalMergerAliases where !packSet.contains(self.packPrefix + alias.logical) {
            if let hit = alias.candidates.first(where: { packSet.contains($0) }) {
                ext[alias.logical] = ("", hit)
            }
        }
        self.externalKeys = ext
        self.config = QWVisionConfig(descriptor: pk.descriptor)
    }

    public init(dir: String) throws {
        self.dir = dir
        self.pack = nil
        let indexPath = dir + "/model.safetensors.index.json"
        var ext: [String: (shard: String, fullKey: String)] = [:]
        if FileManager.default.fileExists(atPath: indexPath) {
            let idxData = try Data(contentsOf: URL(fileURLWithPath: indexPath))
            guard let obj = try JSONSerialization.jsonObject(with: idxData) as? [String: Any],
                  let wm = obj["weight_map"] as? [String: String] else {
                throw QWVisionError.badLayout("malformed vision index json at \(indexPath)")
            }
            let detected = Self.detectPrefix(in: wm.keys)
            self.prefix = detected
            self.packPrefix = detected
            var v: [String: String] = [:]
            for (k, shard) in wm where k.hasPrefix(detected) { v[String(k.dropFirst(detected.count))] = shard }
            for alias in Self.externalMergerAliases where v[alias.logical] == nil {
                if let hit = alias.candidates.first(where: { wm[$0] != nil }), let shard = wm[hit] {
                    ext[alias.logical] = (shard, hit)
                }
            }
            self.index = v
            self.externalKeys = ext
        } else {
            let single = try QWSafetensors(path: dir + "/model.safetensors")
            let detected = Self.detectPrefix(in: single.tensors.keys)
            self.prefix = detected
            self.packPrefix = detected
            var v: [String: String] = [:]
            for k in single.tensors.keys where k.hasPrefix(detected) { v[String(k.dropFirst(detected.count))] = "model.safetensors" }
            for alias in Self.externalMergerAliases where v[alias.logical] == nil {
                if let hit = alias.candidates.first(where: { single.tensors[$0] != nil }) {
                    ext[alias.logical] = ("model.safetensors", hit)
                }
            }
            self.index = v
            self.externalKeys = ext
            self.shards["model.safetensors"] = single
        }
        if let cfgURL = QWModelRoot(url: URL(fileURLWithPath: dir)).configURL,
           let cfgData = try? Data(contentsOf: cfgURL),
           let desc = try? QWModelDescriptor.decode(configJSON: cfgData),
           desc.visDepth > 0 {
            self.config = QWVisionConfig(descriptor: desc)
        } else {
            self.config = Self.inferConfig(
                dir: dir,
                prefix: self.prefix,
                index: self.index,
                externalKeys: self.externalKeys,
                shards: &self.shards
            )
        }
    }

    private static func inferConfig(
        dir: String,
        prefix: String,
        index: [String: String],
        externalKeys: [String: (shard: String, fullKey: String)],
        shards: inout [String: QWSafetensors]
    ) -> QWVisionConfig {
        func lookupShape(_ key: String) -> [Int]? {
            let resolved: (file: String, fullKey: String)
            if let file = index[key] {
                resolved = (file, prefix + key)
            } else if let ext = externalKeys[key] {
                resolved = (ext.shard, ext.fullKey)
            } else {
                return nil
            }
            if let s = shards[resolved.file] { return s.shape(resolved.fullKey) }
            guard let s = try? QWSafetensors(path: dir + "/" + resolved.file) else { return nil }
            shards[resolved.file] = s
            return s.shape(resolved.fullKey)
        }
        let posShape = lookupShape("pos_embed.weight")
        if lookupShape("patchifier.weight") != nil
            || lookupShape("blocks.0.conv2d.weight") != nil
            || lookupShape("layers.0.conv2d.weight") != nil
            || lookupShape("layers.0.conv.spatial_conv.weight") != nil
            || lookupShape("blocks.0.conv.spatial_conv.weight") != nil {
            let pfShape = lookupShape("patchifier.weight") ?? lookupShape("patch_embed.proj.weight")
            let dim = pfShape?.first ?? 1024
            let chan = pfShape?.dropFirst().reduce(1, *) ?? 768
            var temporal = 1
            var patch = max(1, Int(Double(chan / 3).squareRoot().rounded()))
            if 3 * patch * patch != chan && chan % 6 == 0 {
                let p2 = max(1, Int(Double(chan / 6).squareRoot().rounded()))
                if 6 * p2 * p2 == chan {
                    temporal = 2
                    patch = p2
                }
            }
            let blockPrefix = (lookupShape("blocks.0.norm1.weight") != nil
                || lookupShape("blocks.0.operator_norm.weight") != nil) ? "blocks" : "layers"
            var depth = 0
            var lTypes: [String] = []
            var convK = 3
            var kvH: Int?
            var mlpInner = dim * 4
            while lookupShape("\(blockPrefix).\(depth).norm1.weight") != nil
                    || lookupShape("\(blockPrefix).\(depth).layer_norm1.weight") != nil
                    || lookupShape("\(blockPrefix).\(depth).operator_norm.weight") != nil {
                if let cw = lookupShape("\(blockPrefix).\(depth).conv.spatial_conv.weight")
                    ?? lookupShape("\(blockPrefix).\(depth).conv2d.weight") {
                    lTypes.append("conv")
                    if let lastD = cw.last { convK = lastD }
                } else {
                    lTypes.append("full_attention")
                    if kvH == nil, let kw = lookupShape("\(blockPrefix).\(depth).self_attn.k_proj.weight"), kw.count == 2 {
                        let headsEst = max(1, dim / 64)
                        kvH = max(1, kw[0] / max(1, dim / headsEst))
                    }
                }
                if let fc1 = lookupShape("\(blockPrefix).\(depth).feed_forward.w1.weight")
                    ?? lookupShape("\(blockPrefix).\(depth).mlp.fc1.weight")
                    ?? lookupShape("\(blockPrefix).\(depth).mlp.linear_fc1.weight") {
                    mlpInner = fc1[0]
                }
                depth += 1
            }
            let heads = (dim > 0 && dim % 64 == 0) ? (dim / 64) : max(1, dim / 16)
            let fc1W = lookupShape("merger.linear_fc1.weight")
            let merged = fc1W?.last ?? (dim * 4)
            let fc1Mid = fc1W?.first ?? merged
            let merge = dim > 0 ? max(1, Int(Double(merged / dim).squareRoot().rounded())) : 2
            let outDim = lookupShape("merger.linear_fc2.weight")?.first ?? dim
            return QWVisionConfig(
                dim: dim, heads: heads, headDim: dim / max(1, heads), depth: max(1, depth),
                mlpInner: mlpInner, patch: patch, temporal: temporal, merge: merge,
                numPosEmbeds: 0, outDim: outDim, imageTokenId: QWVisionConfig.qwen3VLReference.imageTokenId,
                videoTokenId: QWVisionConfig.qwen3VLReference.videoTokenId, ropeTheta: 10_000.0,
                eps: 1e-5, projectorNormEps: 1e-6, architecture: .lfm3VL, projectionHiddenDim: fc1Mid,
                projectorBias: lookupShape("merger.linear_fc1.bias") != nil,
                mergerIntermediateDim: (fc1Mid != merged ? fc1Mid : nil),
                kvHeads: kvH ?? heads, visionLayerTypes: lTypes, convKernelSize: convK
            )
        }
        if index["embeddings.patch_embedding.weight"] != nil
            && (lookupShape("encoder.layers.0.layer_norm1.weight") != nil
                || lookupShape("layers.0.layer_norm1.weight") != nil)
            && lookupShape("embeddings.class_embedding") == nil {
            let peShape = lookupShape("embeddings.patch_embedding.weight")
            let posS = lookupShape("embeddings.position_embedding.weight") ?? lookupShape("embeddings.position_embedding")
            let dim = peShape?.first ?? 1024
            let chan = peShape?.dropFirst().reduce(1, *) ?? 768
            let patch = max(1, Int(Double(chan / 3).squareRoot().rounded()))
            let numPos = posS?.first ?? 256
            let lPfx = lookupShape("encoder.layers.0.layer_norm1.weight") != nil ? "encoder.layers" : "layers"
            var depth = 0
            while lookupShape("\(lPfx).\(depth).layer_norm1.weight") != nil { depth += 1 }
            let mlpInner = lookupShape("\(lPfx).0.mlp.fc1.weight")?.first ?? (dim * 4)
            let heads = (dim > 0 && dim % 64 == 0) ? (dim / 64) : max(1, dim / 16)
            let fc1W = lookupShape("merger.linear_fc1.weight")
            let merged = fc1W?.last ?? dim
            let fc1Mid = fc1W?.first ?? merged
            let merge = dim > 0 ? max(1, Int(Double(merged / dim).squareRoot().rounded())) : 1
            let outDim = lookupShape("merger.linear_fc2.weight")?.first ?? merged
            return QWVisionConfig(
                dim: dim, heads: heads, headDim: dim / max(1, heads), depth: max(1, depth),
                mlpInner: mlpInner, patch: patch, temporal: 1, merge: merge,
                numPosEmbeds: numPos, outDim: outDim, imageTokenId: QWVisionConfig.qwen3VLReference.imageTokenId,
                videoTokenId: QWVisionConfig.qwen3VLReference.videoTokenId, ropeTheta: 10_000.0,
                eps: 1e-6, architecture: .siglip2, projectionHiddenDim: fc1Mid,
                projectorBias: lookupShape("merger.linear_fc1.bias") != nil,
                mergerIntermediateDim: (fc1Mid != merged ? fc1Mid : nil)
            )
        }
        if index["embeddings.patch_embedding.weight"] != nil {
            let inkPos = lookupShape("embeddings.position_embedding")
            let inkPE = lookupShape("embeddings.patch_embedding.weight")
            let dim = inkPos?.dropFirst().first ?? inkPE?.first ?? 1024
            let numPos = inkPos?.first ?? 1024
            let patchChan = inkPE?.dropFirst().first ?? 1200
            let patch = max(1, Int(Double(patchChan / 3).squareRoot().rounded()))
            let mlpInner = lookupShape("layers.0.mlp.fc1.weight")?.first ?? (dim * 4)
            let projInW = lookupShape("proj_in.weight")
            let merged = projInW?.last ?? (dim * 4)
            let projHidden = projInW?.first ?? merged
            let merge = dim > 0 ? max(1, Int(Double(merged / dim).squareRoot().rounded())) : 2
            let outDim = lookupShape("proj_out.weight")?.first ?? 2304
            var depth = 0
            while lookupShape("layers.\(depth).layernorm_before.weight") != nil { depth += 1 }
            let heads = max(1, dim / 64)
            return QWVisionConfig(
                dim: dim, heads: heads, headDim: dim / heads, depth: max(1, depth),
                mlpInner: mlpInner, patch: patch, temporal: 1, merge: merge,
                numPosEmbeds: numPos, outDim: outDim, imageTokenId: 200054, videoTokenId: -1,
                ropeTheta: 10_000.0, eps: 1e-5, architecture: .inkling2,
                projectionHiddenDim: projHidden, projectorBias: lookupShape("proj_in.bias") != nil
            )
        }
        let pwShape = lookupShape("patch_embed.proj.weight")
        let ref = QWVisionConfig.qwen3VLReference
        let dim = posShape?.dropFirst().first ?? pwShape?.first ?? ref.dim
        let numPosEmbeds = posShape?.first ?? 0
        var patch = ref.patch
        var temporal = ref.temporal
        if let pwShape, pwShape.count == 5 {
            if pwShape[1] == 3 {
                temporal = pwShape[2]
                patch = pwShape[3]
            } else if pwShape[4] == 3 {
                temporal = pwShape[1]
                patch = pwShape[2]
            }
        }
        let mlpInner = lookupShape("blocks.0.mlp.linear_fc1.weight")?.first ?? (dim * 4)
        let fc1Shape = lookupShape("merger.linear_fc1.weight")
        let merged = fc1Shape?.last ?? (dim * 4)
        let fc1Mid = fc1Shape?.first ?? merged
        let merge = dim > 0 ? max(1, Int(Double(merged / dim).squareRoot().rounded())) : 2
        let outDim = lookupShape("merger.linear_fc2.bias")?.first
            ?? lookupShape("merger.linear_fc2.weight")?.first
            ?? ref.outDim
        let hasProjBias = (lookupShape("merger.linear_fc1.bias") != nil)
            || (lookupShape("merger.linear_fc2.bias") != nil)
        var depth = 0
        while lookupShape("blocks.\(depth).norm1.weight") != nil { depth += 1 }
        let heads = (dim > 0 && dim % 72 == 0) ? (dim / 72) : max(1, dim / 64)
        let headDim = heads > 0 ? dim / heads : 0
        return QWVisionConfig(
            dim: dim,
            heads: heads,
            headDim: headDim,
            depth: max(depth, 1),
            mlpInner: mlpInner,
            patch: patch,
            temporal: temporal,
            merge: merge,
            numPosEmbeds: numPosEmbeds,
            outDim: outDim,
            imageTokenId: ref.imageTokenId,
            videoTokenId: ref.videoTokenId,
            projectionHiddenDim: fc1Mid,
            projectorBias: hasProjBias,
            mergerIntermediateDim: (fc1Mid != merged ? fc1Mid : nil)
        )
    }

    func releaseBacking() {
        lock.withLock {
            for shard in shards.values { shard.adviseDontNeed() }
            shards.removeAll()
            pack?.adviseDontNeed()
            pack = nil
        }
    }

    private func shardLocked(_ file: String) throws -> QWSafetensors {
        if let s = shards[file] { return s }
        let s = try QWSafetensors(path: dir + "/" + file)
        shards[file] = s
        return s
    }

    public func f32(_ key: String, want: [Int]? = nil) throws -> [Float] {
        if let p = lock.withLock({ pack }) {
            let resolvedKey: String
            if let ext = externalKeys[key], (try? p.entry(packPrefix + key)) == nil {
                resolvedKey = ext.fullKey
            } else {
                resolvedKey = packPrefix + key
            }
            let f = try p.rawBf16(resolvedKey)
            if let w = want, f.count != w.reduce(1, *) {
                throw QWVisionError.badLayout("\(key): pack \(f.count) ≠ \(w)")
            }
            return f
        }
        if let file = index[key] {
            let s = try lock.withLock { try shardLocked(file) }
            return try s.f32(prefix + key, want: want)
        }
        if let ext = externalKeys[key] {
            let s = try lock.withLock { try shardLocked(ext.shard) }
            return try s.f32(ext.fullKey, want: want)
        }
        throw QWVisionError.missingTensor(key)
    }

    public func shape(_ key: String) -> [Int]? {
        if let p = lock.withLock({ pack }) {
            if let direct = try? p.entry(packPrefix + key) { return direct.shape }
            if let ext = externalKeys[key], let aliasEntry = try? p.entry(ext.fullKey) { return aliasEntry.shape }
            return nil
        }
        if let file = index[key] {
            return try? lock.withLock { try shardLocked(file) }.shape(prefix + key)
        }
        if let ext = externalKeys[key] {
            return try? lock.withLock { try shardLocked(ext.shard) }.shape(ext.fullKey)
        }
        return nil
    }

    func has(_ key: String) -> Bool { shape(key) != nil }
}
