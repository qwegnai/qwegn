// FNNgramMLXRowsTests.swift — Unit tests for MLX-Serve PLE N-gram on-demand row reader.

import XCTest
import Foundation
@testable import QWFlashNext
@testable import QWQuant

final class FNNgramMLXRowsTests: XCTestCase {
    var tempDir: URL!

    override func setUpWithError() throws {
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("fn-ngram-mlx-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let dir = tempDir {
            try? FileManager.default.removeItem(at: dir)
        }
    }

    /// Creates a synthetic MLX safetensors file with 4 rows, 32 elements/row (wCols = 4, gCols = 2).
    private func createSyntheticMLXSafetensors(url: URL) throws {
        // 4 rows, wCols = 4 (32 elements per row since 4 words * 8 4-bit nibbles = 32 elements).
        // gCols = 2 (2 groups per row, groupSize = 16 elements = 2 words).
        let rows = 4
        let wCols = 4
        let gCols = 2

        let wBytes = rows * wCols * 4 // 64 bytes
        let sBytes = rows * gCols * 2 // 16 bytes
        let bBytes = rows * gCols * 2 // 16 bytes

        let headerObj: [String: Any] = [
            "weight": [
                "dtype": "U32",
                "shape": [rows, wCols],
                "data_offsets": [0, wBytes]
            ],
            "scales": [
                "dtype": "BF16",
                "shape": [rows, gCols],
                "data_offsets": [wBytes, wBytes + sBytes]
            ],
            "biases": [
                "dtype": "BF16",
                "shape": [rows, gCols],
                "data_offsets": [wBytes + sBytes, wBytes + sBytes + bBytes]
            ]
        ]

        let headerJSON = try JSONSerialization.data(withJSONObject: headerObj, options: [.sortedKeys])
        var headerLength = UInt64(headerJSON.count).littleEndian

        var data = Data()
        withUnsafeBytes(of: &headerLength) { data.append(contentsOf: $0) }
        data.append(headerJSON)

        // Write weights: row r, element c has nibble (c % 16).
        var wWords: [UInt32] = []
        for r in 0..<rows {
            for col in 0..<wCols {
                var word: UInt32 = 0
                for nibble in 0..<8 {
                    let elem = col * 8 + nibble
                    let q = UInt32((r + elem) % 15)
                    word |= (q << (nibble * 4))
                }
                wWords.append(word.littleEndian)
            }
        }
        for w in wWords {
            var val = w
            withUnsafeBytes(of: &val) { data.append(contentsOf: $0) }
        }

        // Write scales (BF16 = 2.0 -> 0x4000)
        let scaleBF16 = UInt16(0x4000).littleEndian
        for _ in 0..<(rows * gCols) {
            var val = scaleBF16
            withUnsafeBytes(of: &val) { data.append(contentsOf: $0) }
        }

        // Write biases (BF16 = 1.0 -> 0x3F80)
        let biasBF16 = UInt16(0x3F80).littleEndian
        for _ in 0..<(rows * gCols) {
            var val = biasBF16
            withUnsafeBytes(of: &val) { data.append(contentsOf: $0) }
        }

        try data.write(to: url)
    }

    func testReadRowMatchesAffineDequant() throws {
        let fileURL = tempDir.appendingPathComponent("ngram_table.bin")
        try createSyntheticMLXSafetensors(url: fileURL)

        let mlxRows = try FNNgramMLXRows(url: fileURL)
        XCTAssertEqual(mlxRows.entryCount, 4)
        XCTAssertEqual(mlxRows.headDim, 32)

        for r in 0..<4 {
            let rowVals = mlxRows.row(r)
            XCTAssertEqual(rowVals.count, 32)
            for c in 0..<32 {
                let q = Float((r + c) % 15)
                let expected = q * 2.0 + 1.0
                XCTAssertEqual(rowVals[c], expected, accuracy: 1e-4, "Row \(r) col \(c) mismatch")
            }
        }
    }
}
