// QWVision — C18: the vision tower (27-block ViT, hidden 1152, 16 heads,
// head_dim 72, patch 16 / temporal 2 / merge 2, merger → 5120) behind the
// `QWVisionBackend` protocol (spec-18's DRY seam).
//
// This file is the CPU fp32 REFERENCE — the C20-reference-machine role: the
// law authority every accelerated backend (Metal, Core AI/ANE) must match
// against the SAME goldens (vis_index.json) before either may ship. It
import Accelerate
import Foundation
import QWCore
import QWQuant   // Pack mode (release law: the pack is the vision weight source)

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

// MARK: - minimal safetensors reader (the escape hatch: the engine hot path
// never touches this; backends memmap the same files zero-copy C-style).

/// Reads ONE safetensors shard for a key prefix, f16→fp32 on demand.
struct QWSafetensors {
    let data: Data
    var dataStart = 0    // the safetensors offset law: data region begins AFTER the header JSON
    let tensors: [String: (offset: Int, shape: [Int], dtype: String)]

    init(path: String) throws {
        let d = try Data(contentsOf: URL(fileURLWithPath: path), options: .mappedIfSafe)
        self.data = d
        guard d.count > 8 else { throw QWVisionError.badLayout("st too small") }
        var len = 0
        d.withUnsafeBytes { r in for i in 0..<8 { len |= Int(r[i]) << (8 * i) } }
        let hdr = try JSONSerialization.jsonObject(
            with: d.subdata(in: 8..<(8 + len))) as! [String: Any]
        // SAFETENSORS OFFSET LAW (the 93 KB bug, forensics-proven): the
        // header JSON occupies [8, 8+len); the DATA REGION starts at
        // 8+len and data_offsets are relative to THAT. Reading at
        // 8+offset (skipping the header) shifts every tensor by the header
        // size — every bf16 weight silently displaced, no error raised.
        self.dataStart = 8 + len
        var t: [String: (Int, [Int], String)] = [:]
        for (k, v) in hdr where k != "__metadata__" {
            let m = v as! [String: Any]
            t[k] = ((m["data_offsets"] as! [Int])[0],
                    (m["shape"] as! [Int]).map { $0 },
                    (m["dtype"] as! String))
        }
        self.tensors = t
    }

    static func open(dir: String, shardHint: (String) -> String) throws -> [QWSafetensors] {
        // one shard per index entry; callers probe all shards by key.
        []
    }

    /// f16/bf16 → fp32 row-major copy. Shapes asserted against `want`.
    func f32(_ key: String, want: [Int]?) throws -> [Float] {
        guard let t = tensors[key] else { throw QWVisionError.missingTensor(key) }
        if let w = want, t.shape != w {
            throw QWVisionError.shapeMismatch("\(key): \(t.shape) ≠ \(w)")
        }
        let n = t.shape.reduce(1, *)
        var out = [Float](repeating: 0, count: n)
        data.withUnsafeBytes { r in
            out.withUnsafeMutableBufferPointer { ob in
                guard let rawBase = r.baseAddress, let outBase = ob.baseAddress else { return }
                let src = rawBase.advanced(by: dataStart + t.offset).assumingMemoryBound(to: UInt8.self)
                let outU32 = UnsafeMutableRawPointer(outBase).assumingMemoryBound(to: UInt32.self)
                switch t.dtype {
                case "F16":
                    for i in 0..<n {
                        let lo = UInt16(src[i &* 2]), hi = UInt16(src[i &* 2 &+ 1])
                        outBase[i] = Float(Float16(bitPattern: lo | (hi << 8)))
                    }
                case "BF16":
                    for i in 0..<n {
                        let lo = UInt32(src[i &* 2]), hi = UInt32(src[i &* 2 &+ 1])
                        outU32[i] = (lo | (hi << 8)) << 16
                    }
                case "F32":
                    memcpy(outBase, src, n * 4)
                default:
                    fatalError("QWVision: dtype \(t.dtype) not in the law")
                }
            }
        }
        return out
    }

    func shape(_ key: String) -> [Int]? { tensors[key]?.shape }
    func has(_ key: String) -> Bool { tensors[key] != nil }
}

@inline(__always) func qwBF16ToF32(_ bits: UInt16) -> Float {
    Float(bitPattern: UInt32(bits) << 16)
}

/// Multi-shard view over a checkpoint dir: resolves vision_tower.* through
/// the safetensors index (the pack.py escape hatch, Swift side).
public final class QWVisionCheckpoint: @unchecked Sendable {
    let index: [String: String]          // key → shard file (safetensors mode)
    var shards: [String: QWSafetensors] = [:]
    let dir: String
    let prefix: String
    // PACK MODE (release law 2026-10-02): the base checkpoint is REMOVED;
    // the ONLY vision weight source is OUR pack (<pack>.qwp), where every
    // model.visual.* tensor ships as a verbatim-BF16 plane (bit-exact, raw
    var pack: QWQuant.Pack?
    let packPrefix = "model.visual."

    /// Pack-mode init: loads <pack>.qwp once; keys are the tower-relative
    /// names the same as safetensors mode (patch_embed.proj.weight, …).
    public init(pack packPath: String) throws {
        self.dir = packPath
        self.prefix = "vision_tower."
        self.index = [:]
        self.pack = try QWQuant.Pack(url: URL(fileURLWithPath: packPath))
    }

    public init(dir: String) throws {
        self.dir = dir
        self.pack = nil
        let indexPath = dir + "/model.safetensors.index.json"
        if FileManager.default.fileExists(atPath: indexPath) {
            let idxData = try Data(contentsOf: URL(fileURLWithPath: indexPath))
            let obj = try JSONSerialization.jsonObject(with: idxData) as! [String: Any]
            let wm = obj["weight_map"] as! [String: String]
            let detected = wm.keys.contains(where: { $0.hasPrefix("model.visual.") }) ? "model.visual." : "vision_tower."
            self.prefix = detected
            var v: [String: String] = [:]
            for (k, shard) in wm where k.hasPrefix(detected) { v[String(k.dropFirst(detected.count))] = shard }
            self.index = v
        } else {
            let single = try QWSafetensors(path: dir + "/model.safetensors")
            let detected = single.tensors.keys.contains(where: { $0.hasPrefix("model.visual.") }) ? "model.visual." : "vision_tower."
            self.prefix = detected
            var v: [String: String] = [:]
            for k in single.tensors.keys where k.hasPrefix(detected) { v[String(k.dropFirst(detected.count))] = "model.safetensors" }
            self.index = v
            self.shards["model.safetensors"] = single
        }
    }

    func shard(_ file: String) throws -> QWSafetensors {
        if let s = shards[file] { return s }
        let s = try QWSafetensors(path: dir + "/" + file)
        shards[file] = s
        return s
    }

    public func f32(_ key: String, want: [Int]? = nil) throws -> [Float] {
        if let p = pack {
            let f = try p.rawBf16(packPrefix + key)     // verbatim-bf16 → fp32 exact
            if let w = want, f.count != w.reduce(1, *) {
                throw QWVisionError.badLayout("\(key): pack \(f.count) ≠ \(w)")
            }
            return f
        }
        guard let file = index[key] else { throw QWVisionError.missingTensor(key) }
        return try shard(file).f32(prefix + key, want: want)
    }
    public func shape(_ key: String) -> [Int]? {
        if let p = pack { return try? p.entry(packPrefix + key).shape }
        guard let file = index[key] else { return nil }
        return try? shard(file).shape(prefix + key)
    }
    func has(_ key: String) -> Bool { shape(key) != nil }
}

// MARK: - the tower's static shape set (spec-18, measured at load)

public enum QWVisionShape {
    public static let dim = 1152
    public static let heads = 16
    public static let headDim = 72            // 1152 / 16 — FULL rotation
    public static let spatial = 36            // head_dim // 2
    public static let invFreqCount = 18       // spatial / 2
    public static let depth = 27
    public static let mlpInner = 4304         // MEASURED (not 4608 — the
    // merger's width; the MLP inner is the config's intermediate_size)
    public static let merge = 2
    public static let merged = 4608           // 4 × dim (view width)
    public static let out = 5120              // the LLM's hidden width
    public static let posTableSide = 48       // √2304
    public static let ropeTheta: Double = 10_000
    public static let eps: Float = 1e-6
}

// MARK: - LayerNorm (law 4: the torch law, executable)

/// torch LayerNorm: y = (x−μ)·rsqrt(var+eps)·γ + β; var BIASED (÷dim);
/// eps INSIDE the sqrt raw (not eps² — that's a different framework; the
/// golden adjacency 1.9e-3 proves the raw-eps law for THIS checkpoint).
@inline(__always)
func qwLayerNormRow(_ x: UnsafePointer<Float>, _ out: UnsafeMutablePointer<Float>,
                    _ gamma: UnsafePointer<Float>?, _ beta: UnsafePointer<Float>?,
                    dim: Int, eps: Float) {
    let vDim = vDSP_Length(dim)
    var m: Float = 0
    vDSP_meanv(x, 1, &m, vDim)
    var negM = -m
    vDSP_vsadd(x, 1, &negM, out, 1, vDim)
    var s2: Float = 0
    vDSP_svesq(out, 1, &s2, vDim)
    var inv = Float(1.0 / (Double(s2) / Double(dim) + Double(eps)).squareRoot())
    vDSP_vsmul(out, 1, &inv, out, 1, vDim)
    if let g = gamma, let b = beta {
        vDSP_vmul(out, 1, g, 1, out, 1, vDim)
        vDSP_vadd(out, 1, b, 1, out, 1, vDim)
    }
}

func qwLayerNormRows(_ x: UnsafePointer<Float>, _ out: UnsafeMutablePointer<Float>,
                     _ gamma: UnsafePointer<Float>?, _ beta: UnsafePointer<Float>?,
                     rows: Int, dim: Int, eps: Float) {
    nonisolated(unsafe) let xPtr = x, oPtr = out, gPtr = gamma, bPtr = beta
    if rows >= 64 {
        let chunks = min(16, rows), perChunk = (rows + chunks - 1) / chunks
        DispatchQueue.concurrentPerform(iterations: chunks) { c in
            let rStart = c * perChunk, rEnd = min(rows, rStart + perChunk)
            guard rStart < rEnd else { return }
            for r in rStart..<rEnd { qwLayerNormRow(xPtr + r &* dim, oPtr + r &* dim, gPtr, bPtr, dim: dim, eps: eps) }
        }
    } else {
        for r in 0..<rows { qwLayerNormRow(xPtr + r &* dim, oPtr + r &* dim, gPtr, bPtr, dim: dim, eps: eps) }
    }
}

func qwLayerNorm(_ x: [Float], into out: inout [Float], gamma: [Float], beta: [Float], rows: Int, dim: Int) {
    x.withUnsafeBufferPointer { xb in
        out.withUnsafeMutableBufferPointer { ob in
            gamma.withUnsafeBufferPointer { g in
                beta.withUnsafeBufferPointer { b in
                    qwLayerNormRows(xb.baseAddress!, ob.baseAddress!, g.baseAddress!, b.baseAddress!,
                                    rows: rows, dim: dim, eps: QWVisionShape.eps)
                }
            }
        }
    }
}

// MARK: - the GELU split (law 4b: two GELUs, one checkpoint)

/// gelu_pytorch_tanh — blocks' act (hidden_act from config, golden-bound).
@inline(__always) public func qwGeluTanh(_ x: Float) -> Float {
    0.5 * x * (1.0 + tanhf(0.7978845608028654 * (x + 0.044715 * x * x * x)))
}
/// nn.GELU() default = ERF form — the MERGER's act (the split is what the
/// goldens bind; mixing them diffs ~1e-3 on merger outputs, well above tol).
@inline(__always) public func qwGeluErf(_ x: Float) -> Float {
    0.5 * x * (1.0 + erf(x * 0.7071067811865476))
}

@inline(__always)
func qwApplyGeluTanhInPlace(_ ptr: UnsafeMutablePointer<Float>, count: Int, scratch: UnsafeMutablePointer<Float>) {
    let vCount = vDSP_Length(count)
    vDSP_vmul(ptr, 1, ptr, 1, scratch, 1, vCount)
    vDSP_vmul(scratch, 1, ptr, 1, scratch, 1, vCount)
    var c1: Float = 0.044715, c2: Float = 0.7978845608028654, one: Float = 1.0, half: Float = 0.5
    vDSP_vsma(scratch, 1, &c1, ptr, 1, scratch, 1, vCount)
    vDSP_vsmul(scratch, 1, &c2, scratch, 1, vCount)
    var n32 = Int32(count)
    vvtanhf(scratch, scratch, &n32)
    vDSP_vsadd(scratch, 1, &one, scratch, 1, vCount)
    vDSP_vmul(ptr, 1, scratch, 1, ptr, 1, vCount)
    vDSP_vsmul(ptr, 1, &half, ptr, 1, vCount)
}

// MARK: - fp32 GEMM (Accelerate AMX cblas_sgemm over row-major [nOut, nIn] weights)

@_silgen_name("cblas_sgemm$NEWLAPACK")
private func qw_cblas_sgemm(
    _ order: Int32, _ transA: Int32, _ transB: Int32,
    _ m: Int32, _ n: Int32, _ k: Int32, _ alpha: Float,
    _ a: UnsafePointer<Float>, _ lda: Int32,
    _ b: UnsafePointer<Float>, _ ldb: Int32, _ beta: Float,
    _ c: UnsafeMutablePointer<Float>, _ ldc: Int32
)

/// y[r,o] = Σ_k x[r,k]·w[o,k] (+bias[o]) (+add[r·ldAdd+o]); optional act.
func qwGemmRef(y: UnsafeMutablePointer<Float>, ldY: Int, yBase: Int,
               x: [Float], rows: Int, w: [Float], nOut: Int, nIn: Int,
               bias: [Float]?, add: (src: [Float], ld: Int)? = nil,
               act: ((Float) -> Float)? = nil, rowOffset: Int = 0) {
    guard rows > 0, nOut > 0, nIn > 0 else { return }
    x.withUnsafeBufferPointer { xb in
        w.withUnsafeBufferPointer { wb in
            guard let xBasePtr = xb.baseAddress, let wBasePtr = wb.baseAddress else { return }
            let yStart = y + yBase + rowOffset * ldY
            qw_cblas_sgemm(101, 111, 112, Int32(rows), Int32(nOut), Int32(nIn), 1.0,
                           xBasePtr, Int32(nIn), wBasePtr, Int32(nIn), 0.0, yStart, Int32(ldY))
            let bConst = bias?.withUnsafeBufferPointer { $0.baseAddress }
            let aConst = add?.src.withUnsafeBufferPointer { $0.baseAddress }
            let addLd = add?.ld ?? 0
            guard bConst != nil || aConst != nil || act != nil else { return }
            let vOut = vDSP_Length(nOut)
            for r in 0..<rows {
                let yRow = yStart + r * ldY
                if let b = bConst { vDSP_vadd(yRow, 1, b, 1, yRow, 1, vOut) }
                if let a = aConst { vDSP_vadd(yRow, 1, a + (r + rowOffset) * addLd, 1, yRow, 1, vOut) }
                if let f = act { for o in 0..<nOut { yRow[o] = f(yRow[o]) } }
            }
        }
    }
}

// MARK: - weights (loaded once; the checkpoint's native layouts preserved)

public final class QWVisionWeights: @unchecked Sendable {
    public let ck: QWVisionCheckpoint
    public let posEmbed: [Float]          // [2304, 1152] raster row-major
    public let patchW: [Float]            // [1152, 1536] (o, (t,h,w)·c) native
    public let patchB: [Float]            // [1152]
    public let blocks: [QWVisionBlockW]   // 27
    public let mergerGamma: [Float]       // LN(1152) — PRESENCE asserted (affine)
    public let mergerBeta: [Float]
    public let fc1W: [Float]              // [4608, 4608]
    public let fc1B: [Float]
    public let fc2Out: Int                // LLM hidden size (e.g. 5120 for 27B, 2048 for 35B-A3B)
    public let fc2W: [Float]              // [fc2Out, 4608]
    public let fc2B: [Float]              // [fc2Out]
    public let invFreq: [Float]           // [18]

    public convenience init(dir: String) throws {
        try self.init(checkpoint: try QWVisionCheckpoint(dir: dir))
    }

    /// Pack-mode init (release law): weights come from OUR .qwp via the
    /// checkpoint view's pack mode (verbatim-BF16 planes, exact widen).
    public convenience init(pack packPath: String) throws {
        try self.init(checkpoint: try QWVisionCheckpoint(pack: packPath))
    }

    init(checkpoint ck: QWVisionCheckpoint) throws {
        self.ck = ck
        let d = QWVisionShape.dim
        self.posEmbed = try ck.f32("pos_embed.weight", want: [2304, d])
        // LAW 1 (asserted, not assumed): the patch weight is a 5-D conv3d
        // kernel [1152, 2/3, 16, 16, 3/2] — EITHER layout, and the two
        // ASSETS DIFFER (this is the asset-identity lesson of 2026-10-02):
        let pw = ck.shape("patch_embed.proj.weight")
        guard pw == [1152, 3, 2, 16, 16] || pw == [1152, 2, 16, 16, 3] else {
            throw QWVisionError.badLayout("patch_embed.proj.weight \(pw.debugDescription) ≠ [1152,3,2,16,16] (o,c,t,h,w) nor [1152,2,16,16,3] (o,t,h,w,c)")
        }
        // LAW 1b (PAIRING FOLD — adjudicated against the s0 goldens): the
        // C19 slab row is (c-major × temporal-dup × raster): x[k] = pixel
        // [c·512 + t·256 + raster] (x.reshape(n,3,2,256)). The folded target
        // is f[o, c·512 + t·256 + raster].
        //   * BASE (o,c,t,h,w): the flat IS f[o, c·512 + t·256 + raster]
        //     already (c-major, then t, then raster) — the fold is the
        //     IDENTITY (a flat copy).
        //   * LEGACY (o,t,h,w,c): the size-3 axis is LAST, so the flat is
        //     W[o, t·768 + raster·3 + c] and the fold is the permute the
        //     46-permutation probe vs the goldens proved (only pairings
        //     reaching golden-rounding 9.6e-4; every other fold ≥1.8e-2).
        self.patchW = try {
            let isBase = (pw == [1152, 3, 2, 16, 16])
            let native = try ck.f32("patch_embed.proj.weight",
                                    want: isBase ? [1152, 3, 2, 16, 16] : [1152, 2, 16, 16, 3])
            var f = [Float](repeating: 0, count: 1152 * 1536)
            if isBase {
                f = native   // (o,c,t,h,w) == folded layout: identity
            } else {
                for o in 0..<1152 {
                    let ob = o * 1536
                    for t in 0..<2 {
                        for raster in 0..<256 {
                            for c in 0..<3 {
                                // native[ob + t·768 + raster·3 + c] → folded[ob + c·512 + t·256 + raster]
                                f[ob + c * 512 + t * 256 + raster] = native[ob + t * 768 + raster * 3 + c]
                            }
                        }
                    }
                }
            }
            return f
        }()
        self.patchB = try ck.f32("patch_embed.proj.bias", want: [d])
        var bl: [QWVisionBlockW] = []
        var blkIdx = 0
        while ck.has("blocks.\(blkIdx).norm1.weight") {
            bl.append(try QWVisionBlockW(ck, "blocks.\(blkIdx)"))
            blkIdx += 1
        }
        guard !bl.isEmpty else { throw QWVisionError.missingTensor("blocks.0.norm1.weight") }
        self.blocks = bl
        // merger LN is AFFINE (measured: .weight/.bias exist — law 6).
        guard ck.has("merger.norm.weight"), ck.has("merger.norm.bias") else {
            throw QWVisionError.badLayout("merger.norm non-affine law violated (affine weights present in the law; absent here)")
        }
        self.mergerGamma = try ck.f32("merger.norm.weight", want: [d])
        self.mergerBeta = try ck.f32("merger.norm.bias", want: [d])
        self.fc1W = try ck.f32("merger.linear_fc1.weight", want: [4608, 4608])
        self.fc1B = try ck.f32("merger.linear_fc1.bias", want: [4608])
        let outDim = ck.shape("merger.linear_fc2.bias")?[0]
            ?? ck.shape("merger.linear_fc2.weight")?[0]
            ?? QWVisionShape.out
        self.fc2Out = outDim
        self.fc2W = try ck.f32("merger.linear_fc2.weight", want: [outDim, 4608])
        self.fc2B = try ck.f32("merger.linear_fc2.bias", want: [outDim])
        var f = [Float](repeating: 0, count: QWVisionShape.invFreqCount)
        for i in 0..<f.count {
            f[i] = Float(1.0 / pow(QWVisionShape.ropeTheta, Double(2 * i) / Double(QWVisionShape.spatial)))
        }
        self.invFreq = f
        ck.shards.removeAll(); ck.pack?.adviseDontNeed(); ck.pack = nil
    }

    /// The measured MLP inner width (4304 — NOT 4608; the merge width is a
    /// different number; the test binds the confusion out).
    var mlpInnerCheck: Int { QWVisionShape.mlpInner }
}

public final class QWVisionBlockW: @unchecked Sendable {
    let norm1G: [Float], norm1B: [Float], norm2G: [Float], norm2B: [Float]
    let qkvW: [Float], qkvB: [Float]        // [3456, 1152] fused q‖k‖v
    let projW: [Float], projB: [Float]
    let l1W: [Float], l1B: [Float]          // [4304, 1152]
    let l2W: [Float], l2B: [Float]          // [1152, 4304]
    init(_ ck: QWVisionCheckpoint, _ p: String) throws {
        let d = QWVisionShape.dim
        norm1G = try ck.f32("\(p).norm1.weight", want: [d]); norm1B = try ck.f32("\(p).norm1.bias", want: [d])
        norm2G = try ck.f32("\(p).norm2.weight", want: [d]); norm2B = try ck.f32("\(p).norm2.bias", want: [d])
        qkvW = try ck.f32("\(p).attn.qkv.weight", want: [3 * d, d])
        qkvB = try ck.f32("\(p).attn.qkv.bias", want: [3 * d])
        projW = try ck.f32("\(p).attn.proj.weight", want: [d, d])
        projB = try ck.f32("\(p).attn.proj.bias", want: [d])
        l1W = try ck.f32("\(p).mlp.linear_fc1.weight", want: [QWVisionShape.mlpInner, d])
        l1B = try ck.f32("\(p).mlp.linear_fc1.bias", want: [QWVisionShape.mlpInner])
        l2W = try ck.f32("\(p).mlp.linear_fc2.weight", want: [d, QWVisionShape.mlpInner])
        l2B = try ck.f32("\(p).mlp.linear_fc2.bias", want: [d])
    }
}

// MARK: - the forward stages (each mirrors one golden stage, law-numbered)

/// Per-grid geometry the stages consume (the C19 golden bins + coords are
/// RE-READ from disk by the test harness — the reference consumes THE SAME
/// bytes the MSL kernel will gather: one table, three consumers).
public struct QWVisionGridPlan: Sendable {
    public let thw: (t: Int, h: Int, w: Int)
    public let nPatch: Int              // t·h·w
    public let nToken: Int              // nPatch / merge²
    public let frameLen: Int            // h·w (per-FRAME segment length, law 5)
    public init(t: Int, h: Int, w: Int, merge: Int = 2) {
        self.thw = (t, h, w)
        self.nPatch = t * h * w
        self.nToken = t * h * w / (merge * merge)
        self.frameLen = h * w
    }
}

public final class QWVisionRef: @unchecked Sendable {
    public let w: QWVisionWeights
    private let profileStages = ProcessInfo.processInfo.environment["QW_VISION_PROFILE"] == "1"

    public init(weights: QWVisionWeights) { self.w = weights }

    @inline(__always)
    private func profileStart() -> UInt64 {
        profileStages ? DispatchTime.now().uptimeNanoseconds : 0
    }

    @inline(__always)
    private func profileEnd(_ stage: String, since start: UInt64) {
        guard profileStages else { return }
        let milliseconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
        print(String(format: "vision_ref_stage_ms=%.3f stage=%@", milliseconds, stage))
    }

    /// C19-golden bins re-read from disk (the strict-interp law: the
    /// reference and the Metal kernel gather identical tables).
    public struct InterpBins: Sendable {
        public let idx: [Int32]         // [nPatch, 4] i32 (fit asserted in C19)
        public let wgt: [Float]         // [nPatch, 4] f32
        public init(idx: [Int32], wgt: [Float]) {
            precondition(idx.count == Int(idx.count / 4) * 4 && wgt.count == idx.count, "idx and wgt must have matching lengths divisible by 4")
            self.idx = idx
            self.wgt = wgt
        }
        public init(outDir: String, tag: String) throws {
            guard let ib = try? Data(contentsOf: URL(fileURLWithPath: "\(outDir)/mm_interp_idx_\(tag).bin")),
                  let wb = try? Data(contentsOf: URL(fileURLWithPath: "\(outDir)/mm_interp_w_\(tag).bin"))
            else { throw QWVisionError.goldenMissing("mm_interp_\(tag)") }
            self.idx = ib.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
            self.wgt = wb.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            guard idx.count == Int(idx.count / 4) * 4, wgt.count == idx.count else {
                throw QWVisionError.goldenMissing("mm_interp_\(tag) shape")
            }
        }
        public var nPatch: Int { Int(idx.count / 4) }
    }

    /// The full reference forward: patch slab [nPatch, 1536] fp32 → merged
    /// visual tokens [nToken, fc2Out] fp32.
    public func encode(patches x: [Float], plan: QWVisionGridPlan,
                       bins: InterpBins, posIds: [Int32]) -> [Float] {
        let n = plan.nPatch, d = QWVisionShape.dim
        precondition(x.count == n * 1536 && bins.nPatch == n && posIds.count == 2 * n)
        let encodeStart = profileStart()
        var stageStart = profileStart()
        var s = [Float](repeating: 0, count: n * d)
        s.withUnsafeMutableBufferPointer { sb in
            qwGemmRef(y: sb.baseAddress!, ldY: d, yBase: 0,
                      x: x, rows: n, w: w.patchW, nOut: d, nIn: 1536, bias: w.patchB)
        }
        profileEnd("patch_embed", since: stageStart)
        stageStart = profileStart()
        foldPosEmbed(into: &s, bins: bins)
        profileEnd("position_embedding", since: stageStart)
        stageStart = profileStart()
        let (cos, sin) = ropeCosSin(posIds: posIds, n: n)
        profileEnd("rope_tables", since: stageStart)
        var cu = [Int](repeating: 0, count: plan.thw.t + 1)
        for f in 0..<plan.thw.t { cu[f + 1] = cu[f] + plan.frameLen }
        let maxSeg = max(1, plan.frameLen), attnWorkers = QWVisionShape.heads
        var normed = [Float](repeating: 0, count: n * d)
        var qkv = [Float](repeating: 0, count: n * 3456)
        var attnOut = [Float](repeating: 0, count: n * d)
        var mlpH = [Float](repeating: 0, count: n * QWVisionShape.mlpInner)
        var mlpScratch = [Float](repeating: 0, count: n * QWVisionShape.mlpInner)
        var attnQ = [Float](repeating: 0, count: attnWorkers * maxSeg * 72)
        var attnK = [Float](repeating: 0, count: attnWorkers * maxSeg * 72)
        var attnScores = [Float](repeating: 0, count: attnWorkers * maxSeg * maxSeg)
        for (index, blk) in w.blocks.enumerated() {
            stageStart = profileStart()
            block(&s, blk, plan: plan, cu: cu, cos: cos, sin: sin,
                  profileTag: profileStages ? "block\(index)" : nil,
                  normedBuf: &normed, qkvBuf: &qkv, attnBuf: &attnOut,
                  mlpHBuf: &mlpH, mlpScratchBuf: &mlpScratch,
                  attnQBuf: &attnQ, attnKBuf: &attnK, attnScoresBuf: &attnScores, maxSeg: maxSeg)
            profileEnd("block\(index)", since: stageStart)
        }
        stageStart = profileStart()
        merger(&s, plan: plan)
        profileEnd("merger", since: stageStart)
        profileEnd("encode_total", since: encodeStart)
        return s
    }

    public func patchEmbedStage(_ x: [Float], n: Int) -> [Float] {
        var s = [Float](repeating: 0, count: n * 1152)
        s.withUnsafeMutableBufferPointer { sb in
            qwGemmRef(y: sb.baseAddress!, ldY: 1152, yBase: 0,
                      x: x, rows: n, w: w.patchW, nOut: 1152, nIn: 1536, bias: w.patchB)
        }
        return s
    }

    public func foldPosEmbed(into s: inout [Float], bins: InterpBins) {
        let n = bins.nPatch, vDim = vDSP_Length(1152)
        bins.idx.withUnsafeBufferPointer { ib in
            bins.wgt.withUnsafeBufferPointer { wb in
                w.posEmbed.withUnsafeBufferPointer { tb in
                    s.withUnsafeMutableBufferPointer { sb in
                        let sBase = sb.baseAddress!, tBase = tb.baseAddress!
                        let iBase = ib.baseAddress!, wBase = wb.baseAddress!
                        for p in 0..<n {
                            let sRow = sBase + p * 1152
                            for c in 0..<4 {
                                var wc = wBase[p * 4 + c]
                                let tc = tBase + Int(iBase[p * 4 + c]) * 1152
                                vDSP_vsma(tc, 1, &wc, sRow, 1, sRow, 1, vDim)
                            }
                        }
                    }
                }
            }
        }
    }

    /// stage-2 rope: [n,2] coords → cos/sin [n,72].
    public func ropeCosSin(posIds: [Int32], n: Int) -> (cos: [Float], sin: [Float]) {
        var cos = [Float](repeating: 0, count: n * 72)
        var sin = [Float](repeating: 0, count: n * 72)
        w.invFreq.withUnsafeBufferPointer { fb in
            for p in 0..<n {
                let h = Float(posIds[p * 2]), wd = Float(posIds[p * 2 + 1])
                for i in 0..<18 {
                    let ah = h * fb[i], aw = wd * fb[i]
                    let c = cosf(ah), s = sinf(ah), c2 = cosf(aw), s2 = sinf(aw)
                    let o = p * 72 + i
                    cos[o] = c;    cos[o + 18] = c2;  cos[o + 36] = c;   cos[o + 54] = c2
                    sin[o] = s;    sin[o + 18] = s2;  sin[o + 36] = s;   sin[o + 54] = s2
                }
            }
        }
        return (cos, sin)
    }

    public var debugStages: ((String, [Float]) -> Void)?

    public func block(_ s: inout [Float], _ blk: QWVisionBlockW, plan: QWVisionGridPlan,
                      cu: [Int], cos: [Float], sin: [Float], profileTag: String? = nil) {
        let n = plan.nPatch, d = QWVisionShape.dim, maxSeg = max(1, plan.frameLen), heads = QWVisionShape.heads
        var normed = [Float](repeating: 0, count: n * d)
        var qkv = [Float](repeating: 0, count: n * 3456)
        var attnOut = [Float](repeating: 0, count: n * d)
        var mlpH = [Float](repeating: 0, count: n * QWVisionShape.mlpInner)
        var mlpScratch = [Float](repeating: 0, count: n * QWVisionShape.mlpInner)
        var attnQ = [Float](repeating: 0, count: heads * maxSeg * 72)
        var attnK = [Float](repeating: 0, count: heads * maxSeg * 72)
        var attnScores = [Float](repeating: 0, count: heads * maxSeg * maxSeg)
        block(&s, blk, plan: plan, cu: cu, cos: cos, sin: sin, profileTag: profileTag,
              normedBuf: &normed, qkvBuf: &qkv, attnBuf: &attnOut,
              mlpHBuf: &mlpH, mlpScratchBuf: &mlpScratch,
              attnQBuf: &attnQ, attnKBuf: &attnK, attnScoresBuf: &attnScores, maxSeg: maxSeg)
    }

    private func block(_ s: inout [Float], _ blk: QWVisionBlockW, plan: QWVisionGridPlan,
                       cu: [Int], cos: [Float], sin: [Float], profileTag: String?,
                       normedBuf: inout [Float], qkvBuf: inout [Float], attnBuf: inout [Float],
                       mlpHBuf: inout [Float], mlpScratchBuf: inout [Float],
                       attnQBuf: inout [Float], attnKBuf: inout [Float], attnScoresBuf: inout [Float], maxSeg: Int) {
        let n = plan.nPatch, d = 1152
        var stageStart = profileStart()
        qwLayerNorm(s, into: &normedBuf, gamma: blk.norm1G, beta: blk.norm1B, rows: n, dim: d)
        profileEnd("\(profileTag ?? "block").norm1", since: stageStart)
        stageStart = profileStart()
        qkvBuf.withUnsafeMutableBufferPointer { qb in
            qwGemmRef(y: qb.baseAddress!, ldY: 3456, yBase: 0,
                      x: normedBuf, rows: n, w: blk.qkvW, nOut: 3456, nIn: d, bias: blk.qkvB)
        }
        profileEnd("\(profileTag ?? "block").qkv", since: stageStart)
        if let dbg = debugStages { dbg("normed", normedBuf); dbg("qkv", qkvBuf) }
        let scale: Float = Float(1.0 / (72.0).squareRoot())
        let heads = QWVisionShape.heads, hd = QWVisionShape.headDim
        stageStart = profileStart()
        qkvBuf.withUnsafeBufferPointer { qkvB in
            cos.withUnsafeBufferPointer { cosB in
                sin.withUnsafeBufferPointer { sinB in
                    attnBuf.withUnsafeMutableBufferPointer { ab in
                        attnQBuf.withUnsafeMutableBufferPointer { aqb in
                            attnKBuf.withUnsafeMutableBufferPointer { akb in
                                attnScoresBuf.withUnsafeMutableBufferPointer { ascb in
                                    nonisolated(unsafe) let qkvBase = qkvB.baseAddress!, cosBase = cosB.baseAddress!
                                    nonisolated(unsafe) let sinBase = sinB.baseAddress!, attnBase = ab.baseAddress!
                                    nonisolated(unsafe) let qWorkBase = aqb.baseAddress!, kWorkBase = akb.baseAddress!
                                    nonisolated(unsafe) let scWorkBase = ascb.baseAddress!, dbgFn = debugStages
                                    for seg in 0..<(cu.count - 1) {
                                        let lo = cu[seg], hi = cu[seg + 1], m = hi - lo
                                        guard m > 0 else { continue }
                                        let runHead: @Sendable (Int, Int) -> Void = { (head: Int, slot: Int) in
                                            let qPtr = qWorkBase + slot * maxSeg * hd
                                            let kPtr = kWorkBase + slot * maxSeg * hd
                                            let scPtr = scWorkBase + slot * maxSeg * maxSeg
                                            let half = hd / 2
                                            for j in 0..<m {
                                                let p = lo + j
                                                let qSrc = qkvBase + p * 3456 + head * hd, kSrc = qSrc + 1152
                                                let cRow = cosBase + p * 72, sRow = sinBase + p * 72
                                                let qDst = qPtr + j * hd, kDst = kPtr + j * hd
                                                for i in 0..<half {
                                                    let c0 = cRow[i], s0 = sRow[i], c1 = cRow[i + half], s1 = sRow[i + half]
                                                    qDst[i] = qSrc[i] * c0 - qSrc[i + half] * s0
                                                    qDst[i + half] = qSrc[i + half] * c1 + qSrc[i] * s1
                                                    kDst[i] = kSrc[i] * c0 - kSrc[i + half] * s0
                                                    kDst[i + half] = kSrc[i + half] * c1 + kSrc[i] * s1
                                                }
                                            }
                                            if head == 0, let dbg = dbgFn {
                                                var vDbg = [Float](repeating: 0, count: m * hd)
                                                for j in 0..<m {
                                                    memcpy(&vDbg[j * hd], qkvBase + (lo + j) * 3456 + 2304 + head * hd, hd * 4)
                                                }
                                                dbg("q0", Array(UnsafeBufferPointer(start: qPtr, count: m * hd)))
                                                dbg("k0", Array(UnsafeBufferPointer(start: kPtr, count: m * hd)))
                                                dbg("v0", vDbg)
                                            }
                                            qw_cblas_sgemm(101, 111, 112, Int32(m), Int32(m), Int32(hd), scale,
                                                           qPtr, Int32(hd), kPtr, Int32(hd), 0.0, scPtr, Int32(m))
                                            var m32 = Int32(m), vM = vDSP_Length(m)
                                            for qi in 0..<m {
                                                let row = scPtr + qi * m
                                                var mx: Float = -.greatestFiniteMagnitude
                                                vDSP_maxv(row, 1, &mx, vM)
                                                var negMx = -mx
                                                vDSP_vsadd(row, 1, &negMx, row, 1, vM)
                                                vvexpf(row, row, &m32)
                                                var sum: Float = 0
                                                vDSP_sve(row, 1, &sum, vM)
                                                var inv = 1.0 / sum
                                                vDSP_vsmul(row, 1, &inv, row, 1, vM)
                                            }
                                            let vStart = qkvBase + lo * 3456 + 2304 + head * hd
                                            let outStart = attnBase + lo * d + head * hd
                                            qw_cblas_sgemm(101, 111, 111, Int32(m), Int32(hd), Int32(m), 1.0,
                                                           scPtr, Int32(m), vStart, 3456, 0.0, outStart, Int32(d))
                                        }
                                        if debugStages == nil {
                                            DispatchQueue.concurrentPerform(iterations: heads) { head in runHead(head, head) }
                                        } else {
                                            for head in 0..<heads { runHead(head, 0) }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        profileEnd("\(profileTag ?? "block").attention", since: stageStart)
        if let dbg = debugStages { dbg("attnOut", attnBuf) }
        stageStart = profileStart()
        normedBuf.withUnsafeMutableBufferPointer { pb in
            qwGemmRef(y: pb.baseAddress!, ldY: d, yBase: 0,
                      x: attnBuf, rows: n, w: blk.projW, nOut: d, nIn: d, bias: blk.projB)
        }
        s.withUnsafeMutableBufferPointer { sb in
            normedBuf.withUnsafeBufferPointer { pb in
                vDSP_vadd(sb.baseAddress!, 1, pb.baseAddress!, 1, sb.baseAddress!, 1, vDSP_Length(n * d))
            }
        }
        profileEnd("\(profileTag ?? "block").attention_projection", since: stageStart)
        if let dbg = debugStages { dbg("projOut", normedBuf); dbg("afterAttn", s) }
        stageStart = profileStart()
        qwLayerNorm(s, into: &normedBuf, gamma: blk.norm2G, beta: blk.norm2B, rows: n, dim: d)
        profileEnd("\(profileTag ?? "block").norm2", since: stageStart)
        stageStart = profileStart()
        let mlpCount = n * QWVisionShape.mlpInner
        mlpHBuf.withUnsafeMutableBufferPointer { hb in
            qwGemmRef(y: hb.baseAddress!, ldY: QWVisionShape.mlpInner, yBase: 0,
                      x: normedBuf, rows: n, w: blk.l1W, nOut: QWVisionShape.mlpInner, nIn: d, bias: blk.l1B)
            mlpScratchBuf.withUnsafeMutableBufferPointer { scb in
                qwApplyGeluTanhInPlace(hb.baseAddress!, count: mlpCount, scratch: scb.baseAddress!)
            }
        }
        normedBuf.withUnsafeMutableBufferPointer { yb in
            qwGemmRef(y: yb.baseAddress!, ldY: d, yBase: 0,
                      x: mlpHBuf, rows: n, w: blk.l2W, nOut: d, nIn: QWVisionShape.mlpInner, bias: blk.l2B)
        }
        s.withUnsafeMutableBufferPointer { sb in
            normedBuf.withUnsafeBufferPointer { yb in
                vDSP_vadd(sb.baseAddress!, 1, yb.baseAddress!, 1, sb.baseAddress!, 1, vDSP_Length(n * d))
            }
        }
        profileEnd("\(profileTag ?? "block").mlp", since: stageStart)
    }

    /// the merger (law 6): LN per patch row → view-merge 2×2→4608 → fc1 → GELU-ERF → fc2 → [nToken, fc2Out].
    func merger(_ s: inout [Float], plan: QWVisionGridPlan) {
        let n = plan.nPatch, d = 1152, nt = plan.nToken
        var normed = [Float](repeating: 0, count: n * d)
        qwLayerNorm(s, into: &normed, gamma: w.mergerGamma, beta: w.mergerBeta, rows: n, dim: d)
        var h = [Float](repeating: 0, count: nt * 4608)
        var y = [Float](repeating: 0, count: nt * w.fc2Out)
        h.withUnsafeMutableBufferPointer { hb in
            qwGemmRef(y: hb.baseAddress!, ldY: 4608, yBase: 0,
                      x: normed, rows: nt, w: w.fc1W, nOut: 4608, nIn: 4608,
                      bias: w.fc1B, act: qwGeluErf)
        }
        y.withUnsafeMutableBufferPointer { yb in
            qwGemmRef(y: yb.baseAddress!, ldY: w.fc2Out, yBase: 0,
                      x: h, rows: nt, w: w.fc2W, nOut: w.fc2Out, nIn: 4608,
                      bias: w.fc2B)
        }
        s = y
    }
}
