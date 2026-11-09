// QWMMTests — C19 golden bindings (Scripts/oracle/gen_mm_fixtures.py goldens).
//
// Families:
// Full note: docs/STYLE_NOTES.md#QWMMTests.swift.
import XCTest
@testable import QWMM
import QWVision
import QWToken

final class QWMMTests: XCTestCase {

    static let out = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("Scripts/spike/out")

    /// The multimodal fixtures come from `gen_mm_fixtures.py`, which needs a
    /// local Qwen checkpoint. Without it, the test skips with that reason.
    static func fixture(_ name: String) throws -> Data {
        let url = out.appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("needs \(name): generate with a local Qwen checkpoint (gen_mm_fixtures.py)")
        }
        return try Data(contentsOf: url)
    }

    static func index() throws -> QWJson {
        try QWJson.parse(fixture("mm_index.json"))
    }

    static func ints(_ name: String) throws -> [Int32] {
        let d = try fixture(name)
        return d.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
    }
    static func floats(_ name: String) throws -> [Float] {
        let d = try fixture(name)
        return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    static func grid(_ j: QWJson) throws -> QWVisionGrid {
        let a = j.arrayValue!
        return try QWVisionGrid(t: a[0].intValue!, h: a[1].intValue!, w: a[2].intValue!)
    }

    // MARK: A — smart_resize table + aspect contract

    func testSmartResizeTable() throws {
        let idx = try Self.index()
        let rows = idx["resize"].arrayValue!
        XCTAssertEqual(rows.count, 14)
        for row in rows {
            let h = row["h"].intValue!, w = row["w"].intValue!
            let got = try QWMMConfig.smartResize(height: h, width: w)
            XCTAssertEqual(got.h, row["H"].intValue!, "H at \(h)x\(w)")
            XCTAssertEqual(got.w, row["W"].intValue!, "W at \(h)x\(w)")
            // grid + placeholder-count law (G) re-derived from the resize
            let thw = row["thw"].arrayValue!
            XCTAssertEqual(thw[1].intValue, got.h / 16); XCTAssertEqual(thw[2].intValue, got.w / 16)
            let g = try QWVisionGrid(t: thw[0].intValue!, h: thw[1].intValue!, w: thw[2].intValue!)
            XCTAssertEqual(g.tokens, row["n_tokens"].intValue!, "n_tokens at \(h)x\(w)")
        }
    }

    func testAspectContractRaises() throws {
        let idx = try Self.index()
        for c in idx["raise_contracts"].arrayValue! where c["kind"].stringValue == "aspect" {
            XCTAssertThrowsError(try QWMMConfig.smartResize(height: c["h"].intValue!, width: c["w"].intValue!)) { e in
                guard case QWGridError.aspectRatio = e as! QWGridError else { return XCTFail("wrong error \(e)") }
            }
        }
    }

    func testMergeShapeContractRaises() throws {
        // the golden's merge_shape contracts ([4,10,7], [3,5,4]) must raise
        // at GRID CONSTRUCTION (the Swift mirror of the authority's reshape).
        let idx = try Self.index()
        let shapes = idx["raise_contracts"].arrayValue!.filter { $0["kind"].stringValue == "merge_shape" }
        XCTAssertEqual(shapes.count, 2)
        for s in shapes {
            let thw = s["thw"].arrayValue!
            XCTAssertThrowsError(try QWVisionGrid(t: thw[0].intValue!, h: thw[1].intValue!, w: thw[2].intValue!)) { e in
                guard case QWGridError.mergeShape = e as! QWGridError else { return XCTFail("wrong error \(e)") }
            }
        }
    }

    // MARK: C/D/E — per-grid bins, byte-identical

    func testPerGridBinsByteIdentical() throws {
        let idx = try Self.index()
        let grids = idx["grids"].objectPairs
        XCTAssertFalse(grids.isEmpty)
        for (_, g) in grids {
            let tag = "\(g["thw"].arrayValue![0].intValue!)x\(g["thw"].arrayValue![1].intValue!)x\(g["thw"].arrayValue![2].intValue!)"
            let grid = try Self.grid(g["thw"])
            XCTAssertEqual(grid.patches, g["n_patch"].intValue!, "patch count law at \(tag)")
            // C: rope positions [n,2] interleaved h,w (i32 bins)
            XCTAssertEqual(try Self.ints("mm_pos_\(tag).bin"), qwVisionRopePositions(grid: grid), "pos bin diverged at \(tag)")
            // D: interp taps — index bins i32, weights f32
            let (idxBins, wBins) = qwInterpTaps(grid: grid)
            XCTAssertEqual(try Self.ints("mm_interp_idx_\(tag).bin"), idxBins, "interp idx bins diverged at \(tag)")
            XCTAssertEqual(try Self.floats("mm_interp_w_\(tag).bin"), wBins, "interp w bins diverged at \(tag)")
            // E: per-frame cu_seqlens
            XCTAssertEqual(try Self.ints("mm_cu_\(tag).bin"), qwVisionCuSeqlens(grid: grid), "cu bins diverged at \(tag)")
        }
    }

    // MARK: F — get_rope_index goldens for the 6 mixed sequences

    func testMRopeGoldenCases() throws {
        let idx = try Self.index()
        let cases = idx["rope"].objectPairs
        XCTAssertEqual(cases.count, 6)
        for (name, c) in cases {
            let S = c["S"].intValue!
            let types = c["types"].arrayValue!.map { $0.intValue! }
            // Rebuild the case's blocks EXACTLY as the generator's authority
            // saw them: type-run grouping over `types`; image grids consumed
            // in order per IMG span; a video CLIP grid [t,h,w] is split by
            var blocks: [QWMMBlock] = []
            var ii = 0, vi = 0, i = 0
            var clip: QWVisionGrid? = nil       // current clip grid
            var clipFramesLeft = 0              // frames left in it
            let imgs = try c["img"].arrayValue!.map { try Self.grid($0) }
            let vids = try c["vid"].arrayValue!.map { try Self.grid($0) }
            while i < S {
                let t = types[i]
                var j = i
                while j < S && types[j] == t { j += 1 }
                switch t {
                case 0:
                    blocks.append(QWMMBlock(kind: .text, tokens: Array(0..<(j - i)), grid: nil))
                case 1:
                    let g = imgs[ii]; ii += 1
                    XCTAssertEqual(j - i, g.tokens, "image span ≠ grid tokens at \(name) idx \(i)")
                    blocks.append(QWMMBlock(kind: .image, tokens: Array(repeating: 248056, count: j - i), grid: g))
                default:
                    if clipFramesLeft == 0 { clip = vids[vi]; vi += 1; clipFramesLeft = clip!.t }
                    let g = clip!
                    XCTAssertEqual(j - i, g.h * g.w / 4, "span ≠ one frame at \(name) idx \(i)")
                    blocks.append(QWMMBlock(kind: .video, tokens: Array(repeating: 248057, count: j - i),
                                           grid: try QWVisionGrid(t: 1, h: g.h, w: g.w)))
                    clipFramesLeft -= 1
                }
                i = j
            }
            let got = qwMRopePositions(blocks: blocks)
            let gold = try Self.ints("mm_rope_pos_\(name).bin")
            XCTAssertEqual(got.pos.count, 3 * S)
            for p in 0..<3 {   // plane-by-plane (t,h,w) compare
                let mine = Array(got.pos[(p * S)..<(p * S + S)])
                let theirs = Array(gold[(p * S)..<(p * S + S)])
                XCTAssertEqual(mine, theirs, "rope plane \(p) diverged at case \(name)")
            }
            let dGold = try Self.ints("mm_rope_delta_\(name).bin")
            XCTAssertEqual(got.delta, Int(dGold[0]), "delta law at case \(name)")
        }
    }

    // MARK: B — slab law (recorded contract; the door lands with C21)

    func testSlabLawRecorded() throws {
        let idx = try Self.index()
        let slab = idx["slab"]
        XCTAssertEqual(slab["rows"].intValue, 256)
        XCTAssertEqual(slab["row_f32"].intValue, 1536)
        XCTAssertTrue(slab["law"].stringValue!.contains("merge-block order"))
        XCTAssertTrue(slab["law"].stringValue!.contains("temporal dup"))
    }

    func testBuildPatchSlabParityWithGolden() throws {
        let goldSlab = try Self.floats("mm_patches_c0.bin")
        XCTAssertEqual(goldSlab.count, 256 * 1536)

        // Reconstruct the 256x256x3 normalized image from the golden slab
        var imageRGB = [Float](repeating: 0, count: 256 * 256 * 3)
        let blocksW = 16 / QWMMConfig.merge
        for pi in 0..<256 {
            let hb = pi / (4 * blocksW)
            let rem = pi % (4 * blocksW)
            let wb = rem / 4
            let p = rem % 4
            let i = p / 2
            let j = p % 2
            let tp = hb * 32 + i * 16
            let wp = wb * 32 + j * 16

            let rowOffset = pi * 1536
            for c in 0..<3 {
                let channelOffset = rowOffset + c * 512
                for r in 0..<16 {
                    for col in 0..<16 {
                        let val = goldSlab[channelOffset + r * 16 + col]
                        imageRGB[(tp + r) * 256 * 3 + (wp + col) * 3 + c] = val
                    }
                }
            }
        }

        let grid = try QWVisionGrid(t: 1, h: 16, w: 16)
        let builtSlab = try qwBuildPatchSlab(normalizedRGB: imageRGB, height: 256, width: 256, grid: grid)
        XCTAssertEqual(builtSlab.count, 256 * 1536)

        // Exact bitwise Float parity on normalized input
        XCTAssertEqual(builtSlab, goldSlab, "builtSlab diverged from golden mm_patches_c0.bin")
    }

    func testBuildPatchSlabUInt8Parity() throws {
        let goldSlab = try Self.floats("mm_patches_c0.bin")
        let blocksW = 16 / QWMMConfig.merge
        var imageUInt8 = [UInt8](repeating: 0, count: 256 * 256 * 3)

        for pi in 0..<256 {
            let hb = pi / (4 * blocksW)
            let rem = pi % (4 * blocksW)
            let wb = rem / 4
            let p = rem % 4
            let i = p / 2
            let j = p % 2
            let tp = hb * 32 + i * 16
            let wp = wb * 32 + j * 16

            let rowOffset = pi * 1536
            for c in 0..<3 {
                let channelOffset = rowOffset + c * 512
                for r in 0..<16 {
                    for col in 0..<16 {
                        let normVal = goldSlab[channelOffset + r * 16 + col]
                        // Inverse norm: x in [-1, 1] -> u8 in [0, 255]
                        let u8 = UInt8(clamping: Int(((normVal * 0.5 + 0.5) * 255.0).rounded()))
                        imageUInt8[(tp + r) * 256 * 3 + (wp + col) * 3 + c] = u8
                    }
                }
            }
        }

        let grid = try QWVisionGrid(t: 1, h: 16, w: 16)
        let builtSlab = try qwBuildPatchSlab(imageRGB: imageUInt8, height: 256, width: 256, grid: grid)
        XCTAssertEqual(builtSlab.count, 256 * 1536)

        // u8 quantization tolerance: <= 1.0 / 127.5 ≈ 0.0079
        var maxDiff: Float = 0
        for i in 0..<builtSlab.count {
            maxDiff = max(maxDiff, abs(builtSlab[i] - goldSlab[i]))
        }
        XCTAssertLessThanOrEqual(maxDiff, 1.0 / 127.5, "UInt8 slab quantization error too large: \(maxDiff)")
    }

    func testBuildPatchSlabErrorContracts() throws {
        let grid = try QWVisionGrid(t: 1, h: 16, w: 16)
        let validNorm = [Float](repeating: 0, count: 256 * 256 * 3)

        // Height mismatch
        XCTAssertThrowsError(try qwBuildPatchSlab(normalizedRGB: validNorm, height: 250, width: 256, grid: grid)) { e in
            guard case QWGridError.pixelDimensions = e as! QWGridError else { return XCTFail("wrong error \(e)") }
        }

        // Width mismatch
        XCTAssertThrowsError(try qwBuildPatchSlab(normalizedRGB: validNorm, height: 256, width: 200, grid: grid)) { e in
            guard case QWGridError.pixelDimensions = e as! QWGridError else { return XCTFail("wrong error \(e)") }
        }

        // Buffer length mismatch
        let shortBuf = [Float](repeating: 0, count: 100)
        XCTAssertThrowsError(try qwBuildPatchSlab(normalizedRGB: shortBuf, height: 256, width: 256, grid: grid)) { e in
            guard case QWGridError.bufferLength = e as! QWGridError else { return XCTFail("wrong error \(e)") }
        }
    }

    // MARK: plan laws on the same shapes

    func testPlanExpansionAndSpans() throws {
        // [10,11] + IMG(6 tokens grid 1×4×6) + [12] + VID frames 2×(1×4×4 spans
        // — plan consumes the FULL clip grid and expands t·h·w/4 = 8) + [13]
        let img = try QWVisionGrid(t: 1, h: 4, w: 6)
        let vid = try QWVisionGrid(t: 2, h: 4, w: 4)
        let plan = try QWX0Plan(tokens: [10, 11, QWMMConfig.imageToken, 12, QWMMConfig.videoToken, 13],
                                images: [img], videos: [vid])
        XCTAssertEqual(plan.tokens.count, 2 + 6 + 1 + 8 + 1)      // expansion law G
        XCTAssertEqual(plan.spans.count, 5)
        XCTAssertEqual(plan.rows, plan.tokens.count)
        // spans in sequence order with exact row starts
        var row = 0
        for s in plan.spans {
            XCTAssertEqual(s.rowStart, row)
            row += s.rowCount
        }
        XCTAssertEqual(row, plan.rows)
        // positions: 3 planes over the expanded stream; text-axis law on text rows
        let S = plan.tokens.count
        XCTAssertEqual(plan.positions.count, 3 * S)
        for (i, t) in plan.tokens.enumerated() where t != QWMMConfig.imageToken && t != QWMMConfig.videoToken {
            XCTAssertEqual(plan.positions[i], plan.positions[S + i])
            XCTAssertEqual(plan.positions[i], plan.positions[2 * S + i])
        }
        // the image span (grid 1×4×6): h plane sweeps [cur, cur+h/merge),
        // w plane sweeps [cur, cur+w/merge) — the authority's per-axis
        // arange law (llm_grid_h=2, llm_grid_w=3 here; NOT max/2 per axis).
        let imgSpan = plan.spans[1]
        let cur = 2   // after the 2-token text run
        var hs = Set<Int32>(), ws = Set<Int32>()
        for r in imgSpan.rowStart..<imgSpan.rowStart + imgSpan.rowCount {
            hs.insert(plan.positions[S + r]); ws.insert(plan.positions[2 * S + r])
        }
        XCTAssertEqual(hs, Set((0..<2).map { Int32(cur + $0) }))   // h/merge = 2
        XCTAssertEqual(ws, Set((0..<3).map { Int32(cur + $0) }))   // w/merge = 3
        // the NEXT text span advances by max(h,w)/merge = 6/2 = 3 → cur+3
        XCTAssertEqual(plan.positions[8], Int32(cur + 3))
        // video block (2 frames of 4×4): each frame sweeps [cur′, cur′+2) on
        // h/w and advances cur′ by 2; frame 2 starts where frame 1 ended.
        let vidSpan = plan.spans[3]
        var hs2 = Set<Int32>()
        for r in vidSpan.rowStart..<vidSpan.rowStart + vidSpan.rowCount { hs2.insert(plan.positions[S + r]) }
        // frames start at cur+4 (image advance 3 + the lone text token 12
        // advance 1) and each frame advances by its own max/2 = 2
        XCTAssertEqual(hs2, Set((0..<2).flatMap { f in (0..<2).map { Int32(cur + 4 + 2 * f + $0) } }))
    }

    func testPlanPlaceholderOverflow() throws {
        XCTAssertThrowsError(try QWX0Plan(tokens: [1, QWMMConfig.imageToken, 2], images: [], videos: [])) { e in
            guard case QWGridError.placeholderOverflow = e as! QWGridError else { return XCTFail("wrong \(e)") }
        }
    }

    // MARK: QWVision integration & In-Memory Bins

    func testInterpBinsInMemParity() throws {
        let idx = try Self.index()
        let grids = idx["grids"].objectPairs
        XCTAssertFalse(grids.isEmpty)
        for (_, g) in grids {
            let tag = "\(g["thw"].arrayValue![0].intValue!)x\(g["thw"].arrayValue![1].intValue!)x\(g["thw"].arrayValue![2].intValue!)"
            let grid = try Self.grid(g["thw"])
            let inMemBins = grid.interpBins
            let goldIdx = try Self.ints("mm_interp_idx_\(tag).bin")
            let goldW = try Self.floats("mm_interp_w_\(tag).bin")
            XCTAssertEqual(inMemBins.idx, goldIdx, "in-mem interp idx diverged at \(tag)")
            XCTAssertEqual(inMemBins.wgt, goldW, "in-mem interp w diverged at \(tag)")
            XCTAssertEqual(grid.ropePositions, try Self.ints("mm_pos_\(tag).bin"), "grid.ropePositions diverged at \(tag)")
            XCTAssertEqual(grid.plan.nPatch, grid.patches)
            XCTAssertEqual(grid.plan.nToken, grid.tokens)
        }
    }

    func testInterleaveX0Slab() throws {
        let hiddenDim = 5120
        let img = try QWVisionGrid(t: 1, h: 4, w: 4) // 4 tokens
        let plan = try QWX0Plan(tokens: [10, 11, QWMMConfig.imageToken, 12], images: [img], videos: [])
        XCTAssertEqual(plan.tokens.count, 2 + 4 + 1) // 7 tokens
        XCTAssertEqual(plan.rows, 7)

        // Synthetic text embeddings: token i -> value Float(i + 1)
        var textEmbeddings = [Float]()
        for i in 0..<3 { // 3 text tokens: 10, 11, 12
            textEmbeddings.append(contentsOf: Array(repeating: Float(i + 1), count: hiddenDim))
        }

        // Synthetic vision tokens: visual token j -> value Float(100 + j)
        var visualTokens = [Float]()
        for j in 0..<4 { // 4 visual tokens
            visualTokens.append(contentsOf: Array(repeating: Float(100 + j), count: hiddenDim))
        }

        let x0 = try qwInterleaveX0(textEmbeddings: textEmbeddings, visualTokens: visualTokens, plan: plan, hiddenDim: hiddenDim)
        XCTAssertEqual(x0.count, 7 * hiddenDim)

        // Verify span 0 (text, 2 tokens): row 0 -> 1.0, row 1 -> 2.0
        XCTAssertEqual(x0[0 * hiddenDim], 1.0)
        XCTAssertEqual(x0[1 * hiddenDim], 2.0)

        // Verify span 1 (vision, 4 tokens): row 2 -> 100.0, row 3 -> 101.0, row 4 -> 102.0, row 5 -> 103.0
        XCTAssertEqual(x0[2 * hiddenDim], 100.0)
        XCTAssertEqual(x0[3 * hiddenDim], 101.0)
        XCTAssertEqual(x0[4 * hiddenDim], 102.0)
        XCTAssertEqual(x0[5 * hiddenDim], 103.0)

        // Verify span 2 (text, 1 token): row 6 -> 3.0
        XCTAssertEqual(x0[6 * hiddenDim], 3.0)
    }

}

extension QWJson {
    var arrayValue: [QWJson]? { if case .array(let a) = self { return a }; return nil }
    var stringValue: String? { if case .string(let s) = self { return s }; return nil }
    var intValue: Int? {
        switch self { case .int(let i): return i; case .double(let d): return Int(d); default: return nil }
    }
    /// ordered (key, value) pairs for an object — golden sections iterate in
    /// json.dump order (determinism law), never Dictionary order.
    var objectPairs: [(String, QWJson)] {
        if case .object(let p) = self { return p }
        return []
    }
}
