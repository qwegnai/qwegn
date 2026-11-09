// QWMM — C19: multimodal ingestion math (spec-19).
//
// The zero-copy interleave plan + the grid/position laws, ported to bind the
// C19 goldens byte-exactly (Scripts/oracle/gen_mm_fixtures.py — live
// transformers truth, rule 2). Zero kernels here: this module decides WHERE
// every token's embedding lands and WHAT its 3D mRope position is; the
// gather/vision doors write.
import Foundation
import QWCore
import QWHAL
import QWAlloc
import QWToken
import QWVision

// MARK: - A: smart_resize (image_processing_qwen2_vl_fast.smart_resize port) ─

public enum QWGridError: Error, Equatable {
    case aspectRatio(max: Int, min: Int)          // ratio ≥ 200 (golden contract)
    case mergeShape(t: Int, h: Int, w: Int)       // h%merge≠0 or w%merge≠0 (golden contract)
    case placeholderOverflow                      // placeholder without a grid
    case pixelDimensions(expectedH: Int, expectedW: Int, actualH: Int, actualW: Int)
    case bufferLength(expected: Int, actual: Int)
}

/// Python round(x) == banker's rounding (round-half-to-even) — Swift's
/// .rounded() is half-AWAY-from-zero; the difference bites exactly at the
/// factor-half boundary (e.g. h=300, factor=32 → 9.375 no, but h=816 → 25.5
func pyRound(_ x: Double) -> Double {
    let f = x.rounded(.down)
    let frac = x - f
    if frac > 0.5 { return f + 1 }
    if frac < 0.5 { return f }
    // exact .5 → to even
    let i = Int(f)
    return Double(i % 2 == 0 ? i : i + 1)
}

public struct QWVisionGrid: Equatable, Sendable {
    public let t: Int, h: Int, w: Int
    public init(t: Int, h: Int, w: Int) throws {
        guard h % QWMMConfig.merge == 0, w % QWMMConfig.merge == 0 else {
            throw QWGridError.mergeShape(t: t, h: h, w: w)
        }
        self.t = t; self.h = h; self.w = w
    }
    public var patches: Int { t * h * w }
    public var tokens: Int { patches / (QWMMConfig.merge * QWMMConfig.merge) }
    public var plan: QWVisionGridPlan {
        QWVisionGridPlan(t: t, h: h, w: w, merge: QWMMConfig.merge)
    }
    public var interpBins: QWVisionRef.InterpBins {
        let taps = qwInterpTaps(grid: self)
        return QWVisionRef.InterpBins(idx: taps.indices, wgt: taps.weights)
    }
    public var ropePositions: [Int32] {
        qwVisionRopePositions(grid: self)
    }
    public var cuSeqlens: [Int32] {
        qwVisionCuSeqlens(grid: self)
    }
}

public enum QWMMConfig {
    public static let patch = 16
    public static let temporal = 2
    public static let merge = 2
    public static let factor = patch * merge          // 32 — NOT the legacy 28
    public static let minPixels = 65_536
    public static let maxPixels = 16_777_216
    public static let posSide = 48                    // int(2304 ** 0.5)
    public static let imageToken = 248056             //  <|image|>  (vocab-asserted golden-side)
    public static let videoToken = 248057             //  <|video|>
    /// The checkpoint's smart_resize: aspect < 200 (raises at ≥200),
    /// factor-round (banker's), then clamp the pixel budget with floor
    /// (shrink arm) / ceil (grow arm) — the exact operator mix the golden
    public static func smartResize(height: Int, width: Int, maxPixels: Int = QWMMConfig.maxPixels) throws -> (h: Int, w: Int) {
        guard Double(max(height, width)) / Double(min(height, width)) < 200 else {
            throw QWGridError.aspectRatio(max: max(height, width), min: min(height, width))
        }
        var hBar = pyRound(Double(height) / Double(factor)) * Double(factor)
        var wBar = pyRound(Double(width) / Double(factor)) * Double(factor)
        if hBar * wBar > Double(maxPixels) {
            let beta = (Double(height * width) / Double(maxPixels)).squareRoot()
            hBar = max(Double(factor), (Double(height) / beta / Double(factor)).rounded(.down) * Double(factor))
            wBar = max(Double(factor), (Double(width) / beta / Double(factor)).rounded(.down) * Double(factor))
        } else if hBar * wBar < Double(minPixels) {
            let beta = (Double(minPixels) / Double(height * width)).squareRoot()
            hBar = (Double(height) * beta / Double(factor)).rounded(.up) * Double(factor)
            wBar = (Double(width) * beta / Double(factor)).rounded(.up) * Double(factor)
        }
        return (Int(hBar), Int(wBar))
    }
}

// MARK: - D: pos-embed bilinear interpolation (vision_utils port, exact taps) ─

/// One axis's taps + weights for a patch position on an axis of `size`
/// against a `side`-long table (bilinear, align_corners=True, border pad —
/// the ONLY mode the checkpoint uses; operator order preserved verbatim:
func axisTaps(_ index: Int, size: Int) -> (taps: [Int], weights: [Float]) {
    let side = QWMMConfig.posSide
    let denom = max(Float(size - 1), 1)
    let src = Float(index) * Float(side - 1) / denom
    let floorv = src.rounded(.down)
    let base = Int(floorv)                 // src ≥ 0 always (index ≥ 0)
    var taps = [Int](), ws = [Float]()
    for off in [0, 1] {
        taps.append(min(max(base + off, 0), side - 1))
        let d = abs(src - floorv - Float(off))
        ws.append(max(1 - d, 0))
    }
    return (taps, ws)
}

/// The golden-law interpolation table for one grid: for each patch in
/// MERGE-BLOCK ORDER (the slab law — block raster hb-outer, inside (i,j)
/// 2×2), 4 taps (h-major outer product) into the flattened 48×48 pos-embed
public func qwInterpTaps(grid: QWVisionGrid) -> (indices: [Int32], weights: [Float]) {
    let m = QWMMConfig.merge
    let blocksW = grid.w / m
    var indices = [Int32](); var weights = [Float]()
    indices.reserveCapacity(grid.patches * 4); weights.reserveCapacity(grid.patches * 4)
    for _ in 0..<grid.t {                          // `within` repeats across frames
        for within in 0..<(grid.h * grid.w) {
            let inCol = within % m
            let inRow = (within / m) % m
            let blockCol = (within / (m * m)) % blocksW
            let blockRow = within / (m * m * blocksW)
            let row = blockRow * m + inRow
            let col = blockCol * m + inCol
            let (ht, hw) = axisTaps(row, size: grid.h)
            let (wt, ww) = axisTaps(col, size: grid.w)
            for hh in 0..<2 { for wc in 0..<2 {
                indices.append(Int32(ht[hh] * QWMMConfig.posSide + wt[wc]))
                weights.append(hw[hh] * ww[wc])
            }}
        }
    }
    return (indices, weights)
}

// MARK: - C/E: vision rope positions + per-frame segments (vision_utils port) ─

/// 2-axis (h,w) rope coords in merge-block patch order, repeated t times.
public func qwVisionRopePositions(grid: QWVisionGrid) -> [Int32] {
    let m = QWMMConfig.merge
    var hw = [Int32](); hw.reserveCapacity(grid.h * grid.w * 2)
    for hb in 0..<(grid.h / m) { for wb in 0..<(grid.w / m) {
        for i in 0..<m { for j in 0..<m {
            hw.append(Int32(hb * m + i)); hw.append(Int32(wb * m + j))
        }}} }
    var out = [Int32](); out.reserveCapacity(grid.patches * 2)
    for _ in 0..<grid.t { out.append(contentsOf: hw) }
    return out
}

/// cu_seqlens: per-FRAME segments (t+1 entries; the clip is NOT one segment).
public func qwVisionCuSeqlens(grid: QWVisionGrid) -> [Int32] {
    var cu = [Int32](); cu.reserveCapacity(grid.t + 1)
    cu.append(0)
    for _ in 0..<grid.t { cu.append(cu.last! + Int32(grid.h * grid.w)) }
    return cu
}

// MARK: - F: mRope 3D positions (get_rope_index port — the Qwen3.5 law) ──────

public struct QWMMBlock: Equatable, Sendable {
    public enum Kind: Equatable, Sendable { case text, image, video }
    public let kind: Kind
    public let tokens: [Int]          // token ids (placeholders already expanded)
    public let grid: QWVisionGrid?    // nil for text
    public init(kind: Kind, tokens: [Int], grid: QWVisionGrid?) { self.kind = kind; self.tokens = tokens; self.grid = grid }
}

/// Qwen3_5Model.get_rope_index port. NOTE: QWVisionGrid carries the FULL
/// patch grid (pre-merge, like image_grid_thw); the authority divides h/w
/// by spatial_merge_size inside the method (`llm_grid_h/w`) — the port
public func qwMRopePositions(blocks: [QWMMBlock]) -> (pos: [Int32], delta: Int) {
    var S = 0; for b in blocks { S += b.tokens.count }
    var pT = [Int32](), pH = [Int32](), pW = [Int32]()
    pT.reserveCapacity(S); pH.reserveCapacity(S); pW.reserveCapacity(S)
    var cur = 0
    let m = QWMMConfig.merge
    for b in blocks {
        switch b.kind {
        case .text:
            for i in 0..<b.tokens.count {
                let p = Int32(cur + i); pT.append(p); pH.append(p); pW.append(p)
            }
            cur += b.tokens.count
        case .image:
            let g = b.grid!
            emitCube(t: g.t, h: g.h, w: g.w, cur: cur, &pT, &pH, &pW)
            cur += max(g.h, g.w) / m
        case .video:
            let g = b.grid!
            precondition(b.tokens.count == g.tokens, "video block token count ≠ grid tokens")
            for _ in 0..<g.t {
                emitCube(t: 1, h: g.h, w: g.w, cur: cur, &pT, &pH, &pW)
                cur += max(g.h, g.w) / m
            }
        }
    }
    precondition(pT.count == S, "span walk lost tokens")
    var mx = Int32.min
    for i in 0..<S { mx = max(mx, max(pT[i], max(pH[i], pW[i]))) }
    var pos = [Int32](); pos.reserveCapacity(3 * S)
    pos.append(contentsOf: pT); pos.append(contentsOf: pH); pos.append(contentsOf: pW)
    return (pos, Int(mx) + 1 - S)
}

/// One media cube: temporal-major raster of (t × h/m × w/m). `cur` offsets
/// h/w; the t plane carries cur + i (start_position added after arange).
private func emitCube(t lt: Int, h hh: Int, w ww: Int, cur: Int,
                      _ pT: inout [Int32], _ pH: inout [Int32], _ pW: inout [Int32]) {
    let m = QWMMConfig.merge
    let lh = hh / m, lw = ww / m
    for i in 0..<lt {
        let hArange = (0..<lh).map { Int32(cur + $0) }
        let wArange = (0..<lw).map { Int32(cur + $0) }
        for hv in hArange {
            for wv in wArange {
                pT.append(Int32(cur + i)); pH.append(hv); pW.append(wv)
            }
        }
    }
}

// MARK: - B: patch-slab builder (Law B) ─────────────────────────────────────

/// Construct the C19 patch-slab layout from normalized Float RGB pixels [-1, 1].
/// Rows are merge-block ordered (hb outer, wb inner, inside (i,j) 2x2).
/// Each row has 1536 floats: for c in 0..<3: 256 raster floats duplicated twice (temporal dup).
public func qwBuildPatchSlab(normalizedRGB: [Float], height: Int, width: Int, grid: QWVisionGrid) throws -> [Float] {
    let expectedH = grid.h * QWMMConfig.patch
    let expectedW = grid.w * QWMMConfig.patch
    guard height == expectedH && width == expectedW else {
        throw QWGridError.pixelDimensions(expectedH: expectedH, expectedW: expectedW, actualH: height, actualW: width)
    }
    let expectedCount = height * width * 3
    guard normalizedRGB.count == expectedCount else {
        throw QWGridError.bufferLength(expected: expectedCount, actual: normalizedRGB.count)
    }

    let m = QWMMConfig.merge
    let patch = QWMMConfig.patch
    let factor = patch * m
    let blocksW = grid.w / m
    let totalPatches = grid.patches
    var out = [Float](repeating: 0, count: totalPatches * 1536)

    normalizedRGB.withUnsafeBufferPointer { src in
        out.withUnsafeMutableBufferPointer { dst in
            for pi in 0..<totalPatches {
                let hb = pi / (4 * blocksW)
                let rem = pi % (4 * blocksW)
                let wb = rem / 4
                let p = rem % 4
                let i = p / 2
                let j = p % 2
                let tp = hb * factor + i * patch
                let wp = wb * factor + j * patch

                let rowOffset = pi * 1536
                for c in 0..<3 {
                    let channelOffset = rowOffset + c * 512
                    for r in 0..<patch {
                        let srcRowStart = (tp + r) * width * 3 + wp * 3 + c
                        let dstRasterOffset = channelOffset + r * patch
                        for col in 0..<patch {
                            let val = src[srcRowStart + col * 3]
                            dst[dstRasterOffset + col] = val
                            dst[dstRasterOffset + 256 + col] = val
                        }
                    }
                }
            }
        }
    }
    return out
}

/// Construct the C19 patch-slab layout from raw UInt8 RGB pixels [0, 255].
/// Normalizes to [-1, 1] using (val / 255.0 - 0.5) / 0.5 per the processor law.
public func qwBuildPatchSlab(imageRGB: [UInt8], height: Int, width: Int, grid: QWVisionGrid) throws -> [Float] {
    let expectedH = grid.h * QWMMConfig.patch
    let expectedW = grid.w * QWMMConfig.patch
    guard height == expectedH && width == expectedW else {
        throw QWGridError.pixelDimensions(expectedH: expectedH, expectedW: expectedW, actualH: height, actualW: width)
    }
    let expectedCount = height * width * 3
    guard imageRGB.count == expectedCount else {
        throw QWGridError.bufferLength(expected: expectedCount, actual: imageRGB.count)
    }

    let m = QWMMConfig.merge
    let patch = QWMMConfig.patch
    let factor = patch * m
    let blocksW = grid.w / m
    let totalPatches = grid.patches
    var out = [Float](repeating: 0, count: totalPatches * 1536)

    imageRGB.withUnsafeBufferPointer { src in
        out.withUnsafeMutableBufferPointer { dst in
            for pi in 0..<totalPatches {
                let hb = pi / (4 * blocksW)
                let rem = pi % (4 * blocksW)
                let wb = rem / 4
                let p = rem % 4
                let i = p / 2
                let j = p % 2
                let tp = hb * factor + i * patch
                let wp = wb * factor + j * patch

                let rowOffset = pi * 1536
                for c in 0..<3 {
                    let channelOffset = rowOffset + c * 512
                    for r in 0..<patch {
                        let srcRowStart = (tp + r) * width * 3 + wp * 3 + c
                        let dstRasterOffset = channelOffset + r * patch
                        for col in 0..<patch {
                            let raw = src[srcRowStart + col * 3]
                            let norm = (Float(raw) / 255.0 - 0.5) / 0.5
                            dst[dstRasterOffset + col] = norm
                            dst[dstRasterOffset + 256 + col] = norm
                        }
                    }
                }
            }
        }
    }
    return out
}

// MARK: - A+G: placeholder expansion + the X₀ slab plan ─────────────────────

/// One span of the slab plan: either a text-gather span or a vision block
/// the C18 backend must WRITE IN PLACE (protocol law: backends write into
/// the slab, never return a tensor — the zero-copy interleave).
public struct QWX0Span: Equatable, Sendable {
    public enum Content: Equatable, Sendable {
        case text(tokenStart: Int, count: Int)                   // range into the FINAL token stream
        case vision(grid: QWVisionGrid, patches: Int, rows: Int) // rows = merged tokens
    }
    public let content: Content
    public let rowStart: Int                                      // row index in X₀
    public var rowCount: Int {
        switch content { case .text(_, let c): return c; case .vision(_, _, let r): return r }
    }
}

public struct QWX0Plan: Equatable, Sendable {
    public let tokens: [Int]                 // EXPANDED stream (placeholders repeated nTok×)
    public let spans: [QWX0Span]
    public let positions: [Int32]            // [3 × S] mRope, row-major planes
    public let mropeDelta: Int
    public var rows: Int { spans.reduce(0) { $0 + $1.rowCount } }

    /// Build the plan from a RAW token stream (placeholders single-occurrence,
    /// as the chat template emits them): expand each  <|image|> /
    ///  <|video|> to nTok = t·h·w/merge² copies (law G — the template
    public init(tokens: [Int], images: [QWVisionGrid], videos: [QWVisionGrid]) throws {
        var out: [Int] = []
        var spans: [QWX0Span] = []
        var blocks: [QWMMBlock] = []
        var ii = 0, vi = 0, row = 0
        var textRun: [Int] = []
        func flushText() {
            guard !textRun.isEmpty else { return }
            spans.append(QWX0Span(content: .text(tokenStart: out.count, count: textRun.count), rowStart: row))
            blocks.append(QWMMBlock(kind: .text, tokens: textRun, grid: nil))
            out.append(contentsOf: textRun); row += textRun.count; textRun = []
        }
        for t in tokens {
            if t == QWMMConfig.imageToken {
                guard ii < images.count else { throw QWGridError.placeholderOverflow }
                let g = images[ii]; ii += 1
                flushText()
                let exp = g.tokens
                spans.append(QWX0Span(content: .vision(grid: g, patches: g.patches, rows: exp), rowStart: row))
                blocks.append(QWMMBlock(kind: .image, tokens: Array(repeating: t, count: exp), grid: g))
                out.append(contentsOf: Array(repeating: t, count: exp)); row += exp
            } else if t == QWMMConfig.videoToken {
                guard vi < videos.count else { throw QWGridError.placeholderOverflow }
                let g = videos[vi]; vi += 1
                flushText()
                let exp = g.tokens
                spans.append(QWX0Span(content: .vision(grid: g, patches: g.patches, rows: exp), rowStart: row))
                blocks.append(QWMMBlock(kind: .video, tokens: Array(repeating: t, count: exp), grid: g))
                out.append(contentsOf: Array(repeating: t, count: exp)); row += exp
            } else {
                textRun.append(t)
            }
        }
        flushText()
        let rope = qwMRopePositions(blocks: blocks)
        precondition(rope.pos.count == 3 * out.count)
        self.tokens = out; self.spans = spans; self.positions = rope.pos; self.mropeDelta = rope.delta
    }
}

// MARK: - X₀ Slab Interleaving & Vision Pipeline ───────────────────────────

/// Interleave text token embeddings and merged visual tokens into the contiguous X₀ slab.
/// Each row in X₀ has hiddenDim (5120) floats.
/// Text spans write text embeddings at their rowStart.
public func qwInterleaveX0(
    textEmbeddings: [Float],
    visualTokens: [Float],
    plan: QWX0Plan,
    hiddenDim: Int = 5120
) throws -> [Float] {
    let totalRows = plan.rows
    var x0 = [Float](repeating: 0, count: totalRows * hiddenDim)
    try qwInterleaveX0(into: &x0, textEmbeddings: textEmbeddings, visualTokens: visualTokens, plan: plan, hiddenDim: hiddenDim)
    return x0
}

public func qwInterleaveX0(
    into x0: inout [Float],
    textEmbeddings: [Float],
    visualTokens: [Float],
    plan: QWX0Plan,
    hiddenDim: Int = 5120
) throws {
    guard x0.count == plan.rows * hiddenDim else {
        throw QWGridError.bufferLength(expected: plan.rows * hiddenDim, actual: x0.count)
    }
    x0.withUnsafeMutableBufferPointer { x0Buf in
        textEmbeddings.withUnsafeBufferPointer { textBuf in
            visualTokens.withUnsafeBufferPointer { visBuf in
                var textOffset = 0
                var visOffset = 0
                for span in plan.spans {
                    let rowStart = span.rowStart
                    let rowCount = span.rowCount
                    let dstOffset = rowStart * hiddenDim
                    let count = rowCount * hiddenDim
                    switch span.content {
                    case .text:
                        guard textOffset + count <= textBuf.count else { return }
                        for i in 0..<count {
                            x0Buf[dstOffset + i] = textBuf[textOffset + i]
                        }
                        textOffset += count
                    case .vision:
                        guard visOffset + count <= visBuf.count else { return }
                        for i in 0..<count {
                            x0Buf[dstOffset + i] = visBuf[visOffset + i]
                        }
                        visOffset += count
                    }
                }
            }
        }
    }
}

/// End-to-end multimodal image patch encoding through a vision backend.
/// Ingests normalized float RGB pixels, forms the C19 patch slab, evaluates grid
/// geometry, position IDs, and interpolation taps in-memory, and encodes visual tokens.
public func qwEncodeVision(
    normalizedRGB: [Float],
    height: Int,
    width: Int,
    grid: QWVisionGrid,
    backend: any QWVisionBackend
) async throws -> [Float] {
    let patches = try qwBuildPatchSlab(normalizedRGB: normalizedRGB, height: height, width: width, grid: grid)
    return try await backend.encode(patches: patches, grid: grid.plan, bins: grid.interpBins, posIds: grid.ropePositions)
}

/// End-to-end multimodal image patch encoding through a vision backend from UInt8 RGB bytes [0, 255].
public func qwEncodeVision(
    imageRGB: [UInt8],
    height: Int,
    width: Int,
    grid: QWVisionGrid,
    backend: any QWVisionBackend
) async throws -> [Float] {
    let patches = try qwBuildPatchSlab(imageRGB: imageRGB, height: height, width: width, grid: grid)
    return try await backend.encode(patches: patches, grid: grid.plan, bins: grid.interpBins, posIds: grid.ropePositions)
}
