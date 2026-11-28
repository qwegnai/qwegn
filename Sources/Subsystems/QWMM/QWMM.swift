// QWMM.swift — Multimodal preprocessing math, grid geometry, and X0 plan (spec-05).
import Accelerate
import Foundation
import QWCore
import QWHAL
import QWAlloc
import QWToken
import QWVision

/// Config-sourced multimodal tokenization and patch geometry.
public struct QWMMSpec: Sendable, Equatable, Hashable {
    public let patch: Int
    public let temporal: Int
    public let merge: Int
    public let posSide: Int
    public let imageToken: Int
    public let videoToken: Int
    public let minPixels: Int
    public let maxPixels: Int
    public let usesUnfoldedPatchEmbed: Bool
    public let mropeLayout: QWModelDescriptor.MRoPELayout
    public let imageMean: [Float]
    public let imageStd: [Float]

    public var isInkling2: Bool { usesUnfoldedPatchEmbed }
    public var factor: Int { patch * merge }
    public var patchArea: Int { patch * patch }
    public var temporalPatchArea: Int { temporal * patchArea }
    public var patchChannels: Int { 3 * temporalPatchArea }

    public init(
        patch: Int,
        temporal: Int,
        merge: Int,
        posSide: Int,
        imageToken: Int,
        videoToken: Int,
        minPixels: Int = 65_536,
        maxPixels: Int = 16_777_216,
        usesUnfoldedPatchEmbed: Bool = false,
        isInkling2: Bool? = nil,
        mropeLayout: QWModelDescriptor.MRoPELayout = .qwenInterleaved,
        imageMean: [Float] = [0.5, 0.5, 0.5],
        imageStd: [Float] = [0.5, 0.5, 0.5]
    ) {
        self.patch = patch
        self.temporal = temporal
        self.merge = merge
        self.posSide = posSide
        self.imageToken = imageToken
        self.videoToken = videoToken
        self.minPixels = minPixels
        self.maxPixels = maxPixels
        self.usesUnfoldedPatchEmbed = isInkling2 ?? usesUnfoldedPatchEmbed
        self.mropeLayout = mropeLayout
        self.imageMean = imageMean.count == 3 ? imageMean : [0.5, 0.5, 0.5]
        self.imageStd = imageStd.count == 3 ? imageStd : [0.5, 0.5, 0.5]
    }

    public init(
        visionConfig: QWVisionConfig,
        minPixels: Int = 65_536,
        maxPixels: Int = 16_777_216,
        imageMeanOverride: [Float]? = nil,
        imageStdOverride: [Float]? = nil
    ) {
        self.init(
            patch: visionConfig.patch,
            temporal: visionConfig.temporal,
            merge: visionConfig.merge,
            posSide: visionConfig.posSide,
            imageToken: visionConfig.imageTokenId,
            videoToken: visionConfig.videoTokenId,
            minPixels: minPixels,
            maxPixels: maxPixels,
            usesUnfoldedPatchEmbed: visionConfig.architecture == .inkling2,
            mropeLayout: visionConfig.mropeLayout,
            imageMean: imageMeanOverride ?? visionConfig.imageMean,
            imageStd: imageStdOverride ?? visionConfig.imageStd
        )
    }

    public init(
        descriptor: QWModelDescriptor,
        minPixels: Int = 65_536,
        maxPixels: Int = 16_777_216,
        imageMeanOverride: [Float]? = nil,
        imageStdOverride: [Float]? = nil
    ) {
        self.init(
            visionConfig: QWVisionConfig(descriptor: descriptor),
            minPixels: minPixels,
            maxPixels: maxPixels,
            imageMeanOverride: imageMeanOverride,
            imageStdOverride: imageStdOverride
        )
    }

    public func withPixelNormalization(mean: [Float], std: [Float]) -> QWMMSpec {
        QWMMSpec(
            patch: patch, temporal: temporal, merge: merge, posSide: posSide,
            imageToken: imageToken, videoToken: videoToken, minPixels: minPixels,
            maxPixels: maxPixels, usesUnfoldedPatchEmbed: usesUnfoldedPatchEmbed,
            mropeLayout: mropeLayout, imageMean: mean, imageStd: std
        )
    }

    public static let qwen3VLReference = QWMMSpec(visionConfig: .qwen3VLReference)
}

// MARK: - A: smart_resize (image_processing_qwen2_vl_fast.smart_resize port)

public enum QWGridError: Error, Equatable {
    case aspectRatio(max: Int, min: Int)
    case mergeShape(t: Int, h: Int, w: Int)
    case placeholderOverflow
    case pixelDimensions(expectedH: Int, expectedW: Int, actualH: Int, actualW: Int)
    case bufferLength(expected: Int, actual: Int)
}

/// Banker's rounding (round-half-to-even) matching Python's `round(x)`.
func pyRound(_ x: Double) -> Double {
    let f = x.rounded(.down)
    let frac = x - f
    if frac > 0.5 { return f + 1 }
    if frac < 0.5 { return f }
    let i = Int(f)
    return Double(i % 2 == 0 ? i : i + 1)
}

public struct QWVisionGrid: Equatable, Sendable {
    public let t: Int, h: Int, w: Int
    public let spec: QWMMSpec
    public init(t: Int, h: Int, w: Int, spec: QWMMSpec = .qwen3VLReference) throws {
        guard t > 0, h > 0, w > 0, spec.merge > 0, spec.patch > 0, spec.temporal > 0, spec.posSide >= 0 else {
            throw QWGridError.mergeShape(t: t, h: h, w: w)
        }
        guard h % spec.merge == 0, w % spec.merge == 0 else {
            throw QWGridError.mergeShape(t: t, h: h, w: w)
        }
        self.t = t; self.h = h; self.w = w; self.spec = spec
    }
    public var patches: Int { t * h * w }
    public var tokens: Int { patches / (spec.merge * spec.merge) }
    public var plan: QWVisionGridPlan {
        QWVisionGridPlan(t: t, h: h, w: w, merge: spec.merge)
    }
    public var interpBins: QWVisionRef.InterpBins {
        let taps = qwInterpTaps(grid: self)
        return QWVisionRef.InterpBins(idx: taps.indices, wgt: taps.weights)
    }
    public var ropePositions: [Int32] {
        qwVisionRopePositions(grid: self)
    }
}

public enum QWMMConfig {
    public static let patch = QWMMSpec.qwen3VLReference.patch
    public static let temporal = QWMMSpec.qwen3VLReference.temporal
    public static let merge = QWMMSpec.qwen3VLReference.merge
    public static let factor = QWMMSpec.qwen3VLReference.factor
    public static let minPixels = QWMMSpec.qwen3VLReference.minPixels
    public static let maxPixels = QWMMSpec.qwen3VLReference.maxPixels
    public static let posSide = QWMMSpec.qwen3VLReference.posSide
    public static let imageToken = QWMMSpec.qwen3VLReference.imageToken
    public static let videoToken = QWMMSpec.qwen3VLReference.videoToken

    /// Aspect check (< 200), factor-rounding, and pixel-budget clamp.
    public static func smartResize(
        height: Int,
        width: Int,
        maxPixels: Int? = nil,
        spec: QWMMSpec = .qwen3VLReference
    ) throws -> (h: Int, w: Int) {
        guard height > 0, width > 0 else {
            throw QWGridError.pixelDimensions(
                expectedH: max(1, height), expectedW: max(1, width), actualH: height, actualW: width
            )
        }
        guard spec.factor > 0 else {
            throw QWGridError.mergeShape(t: 1, h: spec.patch, w: spec.merge)
        }
        guard Double(max(height, width)) / Double(min(height, width)) < 200 else {
            throw QWGridError.aspectRatio(max: max(height, width), min: min(height, width))
        }
        let factor = Double(spec.factor)
        let budgetMax = maxPixels ?? spec.maxPixels
        let budgetMin = spec.minPixels
        guard budgetMax >= spec.factor * spec.factor, budgetMin >= 0 else {
            throw QWGridError.pixelDimensions(
                expectedH: spec.factor, expectedW: spec.factor, actualH: height, actualW: width
            )
        }
        var hBar = max(factor, pyRound(Double(height) / factor) * factor)
        var wBar = max(factor, pyRound(Double(width) / factor) * factor)
        let area = Double(height) * Double(width)
        if hBar * wBar > Double(budgetMax) {
            let beta = (area / Double(budgetMax)).squareRoot()
            hBar = max(factor, (Double(height) / beta / factor).rounded(.down) * factor)
            wBar = max(factor, (Double(width) / beta / factor).rounded(.down) * factor)
        } else if hBar * wBar < Double(budgetMin) {
            let beta = (Double(budgetMin) / area).squareRoot()
            hBar = max(factor, (Double(height) * beta / factor).rounded(.up) * factor)
            wBar = max(factor, (Double(width) * beta / factor).rounded(.up) * factor)
        }
        while hBar * wBar > Double(budgetMax) {
            if wBar >= hBar && wBar > factor {
                wBar = max(factor, min(wBar - factor, (Double(budgetMax) / hBar / factor).rounded(.down) * factor))
            } else if hBar > factor {
                hBar = max(factor, min(hBar - factor, (Double(budgetMax) / wBar / factor).rounded(.down) * factor))
            } else {
                throw QWGridError.pixelDimensions(
                    expectedH: spec.factor, expectedW: spec.factor, actualH: height, actualW: width
                )
            }
        }
        return (Int(hBar), Int(wBar))
    }
}

// MARK: - D: pos-embed bilinear interpolation (vision_utils port, exact taps)

/// Bilinear align_corners=True axis taps into a `side`-element embedding grid.
func axisTaps(_ index: Int, size: Int, side: Int) -> (taps: [Int], weights: [Float]) {
    guard side > 0 else { return ([0, 0], [1, 0]) }
    let denom = max(Float(size - 1), 1)
    let src = Float(index) * Float(side - 1) / denom
    let floorv = src.rounded(.down)
    let base = Int(floorv)
    var taps = [Int](), ws = [Float]()
    for off in [0, 1] {
        taps.append(min(max(base + off, 0), side - 1))
        let d = abs(src - floorv - Float(off))
        ws.append(max(1 - d, 0))
    }
    return (taps, ws)
}

/// Golden-law interpolation table in merge-block order into `side x side` pos_embed.
public func qwInterpTaps(grid: QWVisionGrid) -> (indices: [Int32], weights: [Float]) {
    if grid.spec.posSide <= 0 { return ([], []) }
    if grid.spec.isInkling2 { return qwInkling2InterpTaps(grid: grid) }
    let m = grid.spec.merge
    let side = grid.spec.posSide
    let blocksW = grid.w / m
    var indices = [Int32](); var weights = [Float]()
    indices.reserveCapacity(grid.patches * 4); weights.reserveCapacity(grid.patches * 4)
    for _ in 0..<grid.t {
        for within in 0..<(grid.h * grid.w) {
            let inCol = within % m
            let inRow = (within / m) % m
            let blockCol = (within / (m * m)) % blocksW
            let blockRow = within / (m * m * blocksW)
            let row = blockRow * m + inRow
            let col = blockCol * m + inCol
            let (ht, hw) = axisTaps(row, size: grid.h, side: side)
            let (wt, ww) = axisTaps(col, size: grid.w, side: side)
            for hh in 0..<2 { for wc in 0..<2 {
                indices.append(Int32(ht[hh] * side + wt[wc]))
                weights.append(hw[hh] * ww[wc])
            }}
        }
    }
    return (indices, weights)
}

// MARK: - C/E: vision rope positions + per-frame segments

/// 2-axis (h,w) rope coords in merge-block patch order, repeated t times.
public func qwVisionRopePositions(grid: QWVisionGrid) -> [Int32] {
    if grid.spec.isInkling2 { return qwInkling2RopePositions(grid: grid) }
    let m = grid.spec.merge
    var hw = [Int32](); hw.reserveCapacity(grid.h * grid.w * 2)
    for hb in 0..<(grid.h / m) { for wb in 0..<(grid.w / m) {
        for i in 0..<m { for j in 0..<m {
            hw.append(Int32(hb * m + i)); hw.append(Int32(wb * m + j))
        }}} }
    var out = [Int32](); out.reserveCapacity(grid.patches * 2)
    for _ in 0..<grid.t { out.append(contentsOf: hw) }
    return out
}

// MARK: - F: mRope 3D positions

public enum QWMMBlock: Equatable, Sendable {
    case text([Int])
    case image(tokens: [Int], grid: QWVisionGrid)
    case video(tokens: [Int], grid: QWVisionGrid)

    public var tokens: [Int] {
        switch self {
        case .text(let tokens), .image(let tokens, _), .video(let tokens, _):
            return tokens
        }
    }
}

/// Builds 3-plane (t, h, w) mRope coordinates over text, image, and video blocks.
public func qwMRopePositions(
    blocks: [QWMMBlock],
    merge: Int? = nil,
    layout: QWModelDescriptor.MRoPELayout = .qwenInterleaved,
    initialCursor: Int = 0
) -> (pos: [Int32], delta: Int) {
    var S = 0; for b in blocks { S += b.tokens.count }
    guard S > 0 else { return ([], 0) }
    var pT = [Int32](), pH = [Int32](), pW = [Int32]()
    pT.reserveCapacity(S); pH.reserveCapacity(S); pW.reserveCapacity(S)
    var cur = initialCursor
    for b in blocks {
        switch b {
        case .text(let tokens):
            for i in 0..<tokens.count {
                let p = Int32(cur + i); pT.append(p); pH.append(p); pW.append(p)
            }
            cur += tokens.count
        case .image(let tokens, let g):
            let m = merge ?? g.spec.merge
            precondition(tokens.count == g.tokens, "image block token count ≠ grid tokens")
            if layout == .bailingVideoRope {
                cur = emitBailingCube(t: g.t, h: g.h, w: g.w, merge: m, cur: cur, &pT, &pH, &pW)
            } else {
                emitCube(t: g.t, h: g.h, w: g.w, merge: m, cur: cur, &pT, &pH, &pW)
                cur += max(g.h, g.w) / m
            }
        case .video(let tokens, let g):
            let m = merge ?? g.spec.merge
            precondition(tokens.count == g.tokens, "video block token count ≠ grid tokens")
            if layout == .bailingVideoRope {
                cur = emitBailingCube(t: g.t, h: g.h, w: g.w, merge: m, cur: cur, &pT, &pH, &pW)
            } else {
                for _ in 0..<g.t {
                    emitCube(t: 1, h: g.h, w: g.w, merge: m, cur: cur, &pT, &pH, &pW)
                    cur += max(g.h, g.w) / m
                }
            }
        }
    }
    precondition(pT.count == S, "span walk lost tokens")
    var mx = Int32.min
    for i in 0..<S { mx = max(mx, max(pT[i], max(pH[i], pW[i]))) }
    var pos = [Int32](); pos.reserveCapacity(3 * S)
    pos.append(contentsOf: pT); pos.append(contentsOf: pH); pos.append(contentsOf: pW)
    let delta = (layout == .bailingVideoRope) ? (cur - (initialCursor + S)) : (Int(mx) + 1 - (initialCursor + S))
    return (pos, delta)
}

private func emitBailingCube(
    t lt: Int, h hh: Int, w ww: Int, merge m: Int, cur: Int,
    _ pT: inout [Int32], _ pH: inout [Int32], _ pW: inout [Int32]
) -> Int {
    guard lt > 0 else { return cur }
    let lh = hh / m, lw = ww / m
    let hCenter = (lh - 1) / 2, wCenter = (lw - 1) / 2
    for ti in 0..<lt {
        let baseT = cur + ti * 2
        for hi in 0..<lh {
            for wi in 0..<lw {
                pT.append(Int32(baseT))
                pH.append(Int32(baseT + hi - hCenter))
                pW.append(Int32(baseT + wi - wCenter))
            }
        }
    }
    return cur + (lt - 1) * 2 + 1
}

private func emitCube(t lt: Int, h hh: Int, w ww: Int, merge m: Int, cur: Int,
                      _ pT: inout [Int32], _ pH: inout [Int32], _ pW: inout [Int32]) {
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

// MARK: - B: patch-slab builder (vDSP-accelerated)

private func validatePatchSlabInputs(
    grid: QWVisionGrid,
    height: Int,
    width: Int,
    sampleCount: Int
) throws {
    guard grid.t == 1 else {
        throw QWGridError.mergeShape(t: grid.t, h: grid.h, w: grid.w)
    }
    let spec = grid.spec
    let expectedH = grid.h * spec.patch
    let expectedW = grid.w * spec.patch
    guard height == expectedH && width == expectedW else {
        throw QWGridError.pixelDimensions(expectedH: expectedH, expectedW: expectedW, actualH: height, actualW: width)
    }
    let expectedCount = height * width * 3
    guard sampleCount == expectedCount else {
        throw QWGridError.bufferLength(expected: expectedCount, actual: sampleCount)
    }
}

private func scatterPlanarPatches(
    planes: [Float],
    grid: QWVisionGrid,
    width: Int,
    pixelCount: Int
) -> [Float] {
    let spec = grid.spec
    let m = spec.merge
    let patch = spec.patch
    let factor = spec.factor
    let patchArea = spec.patchArea
    let temporal = spec.temporal
    let temporalArea = spec.temporalPatchArea
    let patchChannels = spec.patchChannels
    let cellsPerBlock = m * m
    let blocksW = grid.w / m
    let totalPatches = grid.patches
    let patchBytes = patchArea * MemoryLayout<Float>.stride
    var out = [Float](repeating: 0, count: totalPatches * patchChannels)

    planes.withUnsafeBufferPointer { srcPlanes in
        out.withUnsafeMutableBufferPointer { dst in
            guard let planeBase = srcPlanes.baseAddress, let dstBase = dst.baseAddress else { return }
            let vPatch = vDSP_Length(patch)
            let vWidth = vDSP_Length(width)
            for pi in 0..<totalPatches {
                let hb = pi / (cellsPerBlock * blocksW)
                let rem = pi % (cellsPerBlock * blocksW)
                let wb = rem / cellsPerBlock
                let p = rem % cellsPerBlock
                let i = p / m
                let j = p % m
                let tp = hb * factor + i * patch
                let wp = wb * factor + j * patch
                let rowOffset = pi * patchChannels
                for c in 0..<3 {
                    let planeC = planeBase + c * pixelCount
                    let dstPatch0 = dstBase + rowOffset + c * temporalArea
                    vDSP_mmov(planeC + tp * width + wp, dstPatch0, vPatch, vPatch, vWidth, vPatch)
                    for t in 1..<temporal {
                        memcpy(dstPatch0 + t * patchArea, dstPatch0, patchBytes)
                    }
                }
            }
        }
    }
    return out
}

/// Construct the patch-slab layout from normalized Float RGB pixels [-1, 1].
public func qwBuildPatchSlab(normalizedRGB: [Float], height: Int, width: Int, grid: QWVisionGrid) throws -> [Float] {
    if grid.spec.isInkling2 {
        return try qwBuildInkling2PatchSlab(normalizedPaddedRGB: normalizedRGB, paddedH: height, paddedW: width, grid: grid)
    }
    try validatePatchSlabInputs(grid: grid, height: height, width: width, sampleCount: normalizedRGB.count)
    let pixelCount = height * width
    guard pixelCount > 0 else { return [] }
    var planes = [Float](repeating: 0, count: 3 * pixelCount)
    let vPixels = vDSP_Length(pixelCount)
    normalizedRGB.withUnsafeBufferPointer { src in
        planes.withUnsafeMutableBufferPointer { dst in
            guard let sBase = src.baseAddress, let pBase = dst.baseAddress else { return }
            for c in 0..<3 {
                vDSP_mmov(sBase + c, pBase + c * pixelCount, 1, vPixels, 3, 1)
            }
        }
    }
    return scatterPlanarPatches(planes: planes, grid: grid, width: width, pixelCount: pixelCount)
}

/// Construct the patch-slab layout from raw UInt8 RGB pixels [0, 255], applying per-channel mean/std.
public func qwBuildPatchSlab(imageRGB: [UInt8], height: Int, width: Int, grid: QWVisionGrid) throws -> [Float] {
    let mean = grid.spec.imageMean
    let std = grid.spec.imageStd
    if grid.spec.usesUnfoldedPatchEmbed {
        var norm = [Float](repeating: 0, count: imageRGB.count)
        let scale: Float = 1.0 / 255.0
        for idx in 0..<(imageRGB.count / 3) {
            let b = idx * 3
            norm[b] = (Float(imageRGB[b]) * scale - mean[0]) / max(1e-6, std[0])
            norm[b + 1] = (Float(imageRGB[b + 1]) * scale - mean[1]) / max(1e-6, std[1])
            norm[b + 2] = (Float(imageRGB[b + 2]) * scale - mean[2]) / max(1e-6, std[2])
        }
        return try qwBuildInkling2PatchSlab(normalizedPaddedRGB: norm, paddedH: height, paddedW: width, grid: grid)
    }
    try validatePatchSlabInputs(grid: grid, height: height, width: width, sampleCount: imageRGB.count)
    let pixelCount = height * width
    guard pixelCount > 0 else { return [] }
    var planes = [Float](repeating: 0, count: 3 * pixelCount)
    let vPixels = vDSP_Length(pixelCount)
    var div255: Float = 255.0
    imageRGB.withUnsafeBufferPointer { src in
        planes.withUnsafeMutableBufferPointer { dst in
            guard let sBase = src.baseAddress, let pBase = dst.baseAddress else { return }
            for c in 0..<3 {
                var negMean = -mean[c]
                var invStd = 1.0 / max(1e-6, std[c])
                let planeC = pBase + c * pixelCount
                vDSP_vfltu8(sBase + c, 3, planeC, 1, vPixels)
                vDSP_vsdiv(planeC, 1, &div255, planeC, 1, vPixels)
                vDSP_vsadd(planeC, 1, &negMean, planeC, 1, vPixels)
                vDSP_vsmul(planeC, 1, &invStd, planeC, 1, vPixels)
            }
        }
    }
    return scatterPlanarPatches(planes: planes, grid: grid, width: width, pixelCount: pixelCount)
}

// MARK: - A+G: placeholder expansion + the X₀ slab plan

/// One span of the X₀ slab: either a text-gather span or a merged vision block.
public struct QWX0Span: Equatable, Sendable {
    public enum Content: Equatable, Sendable {
        case text(tokenStart: Int, count: Int)
        case vision(grid: QWVisionGrid, patches: Int, rows: Int)
    }
    public let content: Content
    public let rowStart: Int
    public var rowCount: Int {
        switch content { case .text(_, let c): return c; case .vision(_, _, let r): return r }
    }
}

public struct QWX0Plan: Equatable, Sendable {
    public let tokens: [Int]
    public let spans: [QWX0Span]
    public let visionSpanIsVideo: [Bool]
    public let positions: [Int32]
    public let mropeDelta: Int
    public var rows: Int { spans.reduce(0) { $0 + $1.rowCount } }

    /// Expand raw media placeholder tokens to `t*h*w/merge²` rows and compute 3D mRope positions.
    public init(
        tokens: [Int],
        images: [QWVisionGrid],
        videos: [QWVisionGrid],
        spec: QWMMSpec = .qwen3VLReference
    ) throws {
        var out: [Int] = []
        var spans: [QWX0Span] = []
        var visIsVideo: [Bool] = []
        var blocks: [QWMMBlock] = []
        var ii = 0, vi = 0, row = 0
        var textRun: [Int] = []
        func flushText() {
            guard !textRun.isEmpty else { return }
            spans.append(QWX0Span(content: .text(tokenStart: out.count, count: textRun.count), rowStart: row))
            blocks.append(.text(textRun))
            out.append(contentsOf: textRun); row += textRun.count; textRun = []
        }
        for t in tokens {
            if t == spec.imageToken {
                guard ii < images.count else { throw QWGridError.placeholderOverflow }
                let g = images[ii]; ii += 1
                flushText()
                let exp = g.tokens
                spans.append(QWX0Span(content: .vision(grid: g, patches: g.patches, rows: exp), rowStart: row))
                visIsVideo.append(false)
                blocks.append(.image(tokens: Array(repeating: t, count: exp), grid: g))
                out.append(contentsOf: Array(repeating: t, count: exp)); row += exp
            } else if spec.videoToken >= 0 && t == spec.videoToken {
                guard vi < videos.count else { throw QWGridError.placeholderOverflow }
                let g = videos[vi]; vi += 1
                flushText()
                let exp = g.tokens
                spans.append(QWX0Span(content: .vision(grid: g, patches: g.patches, rows: exp), rowStart: row))
                visIsVideo.append(true)
                blocks.append(.video(tokens: Array(repeating: t, count: exp), grid: g))
                out.append(contentsOf: Array(repeating: t, count: exp)); row += exp
            } else {
                textRun.append(t)
            }
        }
        flushText()
        guard ii == images.count, vi == videos.count else {
            throw QWGridError.placeholderOverflow
        }
        let rope = qwMRopePositions(blocks: blocks, merge: spec.merge, layout: spec.mropeLayout)
        precondition(rope.pos.count == 3 * out.count)
        self.tokens = out
        self.spans = spans
        self.visionSpanIsVideo = visIsVideo
        self.positions = rope.pos
        self.mropeDelta = rope.delta
    }
}

// MARK: - X₀ Slab Interleaving & Vision Pipeline

/// Interleave text embeddings and merged visual tokens into the contiguous X₀ slab.
public func qwInterleaveX0(
    textEmbeddings: [Float],
    visualTokens: [Float],
    plan: QWX0Plan,
    hiddenDim: Int
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
    hiddenDim: Int
) throws {
    guard hiddenDim >= 0 else {
        throw QWGridError.bufferLength(expected: 0, actual: x0.count)
    }
    let (expectedTotal, totalOverflow) = plan.rows.multipliedReportingOverflow(by: hiddenDim)
    guard !totalOverflow, x0.count == expectedTotal else {
        throw QWGridError.bufferLength(expected: totalOverflow ? Int.max : expectedTotal, actual: x0.count)
    }
    try x0.withUnsafeMutableBufferPointer { x0Buf in
        try textEmbeddings.withUnsafeBufferPointer { textBuf in
            try visualTokens.withUnsafeBufferPointer { visBuf in
                var textOffset = 0
                var visOffset = 0
                for span in plan.spans {
                    let rowStart = span.rowStart
                    let rowCount = span.rowCount
                    let dstOffset = rowStart * hiddenDim
                    let count = rowCount * hiddenDim
                    switch span.content {
                    case .text:
                        guard textOffset + count <= textBuf.count else {
                            throw QWGridError.bufferLength(expected: textOffset + count, actual: textBuf.count)
                        }
                        for i in 0..<count {
                            x0Buf[dstOffset + i] = textBuf[textOffset + i]
                        }
                        textOffset += count
                    case .vision:
                        guard visOffset + count <= visBuf.count else {
                            throw QWGridError.bufferLength(expected: visOffset + count, actual: visBuf.count)
                        }
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
