import XCTest
@testable import QWFlashNext
import QWCore

// MLX/oMLX tie contract: with every block score equal (all zero: q = 0 after the zero projection), the
// selection is the last k of a stable ascending sort by (score, index), so the HIGHEST blocks win.
final class FNSelectionTieTests: XCTestCase {
    func testZeroScoreTiesPreferHigherBlocks() {
        let hidden = 8, heads = 2, headDim = 8, compress = 4, budget = 8   // blockTopK = 2
        let indexer = FNQSAIndexer(
            heads: heads, kvHeads: 1, headDim: headDim, budget: budget, compress: compress, eps: 1e-6,
            qkProj: [Float](repeating: 0, count: (heads + 1) * headDim * hidden),
            qNorm: [Float](repeating: 0, count: headDim), kNorm: [Float](repeating: 0, count: headDim))
        let tokens = 20
        let x = (0..<(tokens * hidden)).map { Float(($0 * 37) % 11) - 5 }
        let rope = FNRope(rotaryDim: 4, theta: 1e7)
        let sel = indexer.select(x, rows: tokens, hidden: hidden, rope: rope)
        // Query 19: complete blocks 0...4 (tokens 0-19), tail empty. Equal scores -> blocks 4 and 3 (tokens 12-19).
        let row = (0..<tokens).filter { sel[19][$0] }
        XCTAssertEqual(row, Array(12..<20))
        // Query 9: complete blocks 0...1 fit in k=2 (tokens 0-7); the causal tail 8-9 stays visible.
        XCTAssertEqual((0..<tokens).filter { sel[9][$0] }, Array(0..<10))
    }
}
