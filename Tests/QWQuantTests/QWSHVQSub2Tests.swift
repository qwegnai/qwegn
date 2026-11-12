// QWSHVQSub2Tests.swift — Canonical codebook, dyadic 0-ULP stepMap15,
// QWWeightFormat row sizing, and QWP1 dequant parity for SHVQ1 & SHVQ15.
import Foundation
import XCTest
@testable import QWCore
@testable import QWQuant

final class QWSHVQSub2Tests: XCTestCase {

    func testSub2CodebookAndDyadicStepMap15Invariants() {
        XCTAssertEqual(QWSHVQSub2Tables.lut1Bits.count, 128 * 8)
        XCTAssertEqual(QWSHVQSub2Tables.lut15PackedI8.count, 2048 * 2)
        XCTAssertEqual(QWSHVQSub2Tables.stepMap15Bits.count, 256)
        XCTAssertEqual(QWSHVQSub2Tables.stepMap15Bits[0], 0)

        for bits in QWSHVQSub2Tables.lut1Bits {
            let f = Float16(bitPattern: bits)
            XCTAssertFalse(f.isNaN)
            XCTAssertFalse(f.isInfinite)
        }

        // Every 8D centroid in lut1Bits (128x8) and lut15PackedI8 (2048x8) has bounded Gaussian radius.
        for c in 0..<128 {
            var normSq: Float = 0
            for j in 0..<8 {
                let v = Float(Float16(bitPattern: QWSHVQSub2Tables.lut1Bits[c * 8 + j]))
                normSq += v * v
            }
            let norm = normSq.squareRoot()
            XCTAssertGreaterThan(norm, 0.25, "lut1 centroid \(c) must be non-degenerate")
            XCTAssertLessThanOrEqual(norm, 3.0, "lut1 centroid \(c) norm must stay within 8D spherical bound")
        }
        for c in 0..<2048 {
            var normSq: Float = 0
            for half in 0..<2 {
                let p32 = QWSHVQSub2Tables.lut15PackedI8[c * 2 + half]
                for sub in 0..<4 {
                    let raw = UInt8((p32 >> (sub * 8)) & 0xFF)
                    let v = Float(Float16(bitPattern: QWSHVQSub2Tables.stepMap15Bits[Int(raw)]))
                    normSq += v * v
                }
            }
            let norm = normSq.squareRoot()
            XCTAssertGreaterThan(norm, 0.25, "lut15 centroid \(c) must be non-degenerate")
            XCTAssertLessThanOrEqual(norm, 4.0, "lut15 centroid \(c) norm must stay within 8D spherical bound")
        }

        // Every signed int8 byte pattern (-127...127) in lut15PackedI8 must match
        // Float16(Float(q) * (3.0 / 128.0)) bit-for-bit in stepMap15Bits.
        for q in -127...127 {
            let u8 = UInt8(bitPattern: Int8(q))
            let expected = Float16(Float(q) * (3.0 / 128.0)).bitPattern
            XCTAssertEqual(QWSHVQSub2Tables.stepMap15Bits[Int(u8)], expected,
                           "stepMap15Bits[\(u8)] mismatch for q=\(q)")
        }

        for packed in QWSHVQSub2Tables.lut15PackedI8 {
            for j in 0..<4 {
                let raw = UInt8((packed >> (j * 8)) & 0xFF)
                let q = Int8(bitPattern: raw)
                XCTAssertGreaterThan(q, -128, "lut15 must avoid -128 so Int8 negation never overflows")
            }
        }
    }

    func testQWWeightFormatRowBytesForSHVQ1AndSHVQ15() {
        let fmt1 = QWWeightFormat.hqBlock(bits: 1, groupSize: 64, denseSHVQ: true, expertSHVQ: true)
        let fmt15 = QWWeightFormat.hqBlock(bits: 15, groupSize: 64, denseSHVQ: true, expertSHVQ: true)
        XCTAssertTrue(fmt1.usesDenseAQKernels)
        XCTAssertTrue(fmt15.usesDenseAQKernels)
        // K = 512 => 8 groups of 64. SHVQ1_G64 = 10 B/grp (80 B/row), SHVQ15_G64 = 14 B/grp (112 B/row).
        XCTAssertEqual(fmt1.rowBytes(k: 512), 80)
        XCTAssertEqual(fmt15.rowBytes(k: 512), 112)
        XCTAssertEqual(fmt1.expertRowBytes(k: 512), 80)
        XCTAssertEqual(fmt15.expertRowBytes(k: 512), 112)
        XCTAssertEqual(fmt1.expertDownRowBytes(k: 512), 80)
        XCTAssertEqual(fmt15.expertDownRowBytes(k: 512), 112)
    }

    func testQWP1PackDequantSHVQ1AndSHVQ15() throws {
        for bits in [1, 15] {
            let tmpDir = FileManager.default.temporaryDirectory
                .appendingPathComponent("qwg_shvq_sub2_\(bits)_\(UUID().uuidString)", isDirectory: true)
            try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: tmpDir) }

            let rows = 4, cols = 64, group = 64
            let gBytes = QWSHVQBlock.bytes(bits: bits, group: group)
            var denseSlab = Data()
            var expectedRotRows = [[Float]]()
            for r in 0..<rows {
                var block = [UInt8](repeating: 0, count: gBytes)
                let sc = Float16(0.03125 * Float(r + 1)).bitPattern
                block[0] = UInt8(sc & 0xFF)
                block[1] = UInt8(sc >> 8)
                for b in 2..<gBytes {
                    block[b] = UInt8((r * 37 + b * 23 + 11) & 0xFF)
                }
                var unpacked = [Float]()
                block.withUnsafeBytes { QWSHVQBlock.unpack(group: $0, bits: bits, group: group, into: &unpacked) }
                unpacked.withUnsafeMutableBufferPointer { buf in
                    for off in stride(from: 0, to: cols, by: 32) {
                        QWQuant.fwht32OrthoBlock(buf.baseAddress! + off)
                    }
                }
                expectedRotRows.append(unpacked)
                denseSlab.append(contentsOf: block)
            }

            var fileBytes = Data("QWP1".utf8)
            fileBytes.append(Data(repeating: 0, count: 60))
            let denseOff = fileBytes.count
            fileBytes.append(denseSlab)
            let pad = (64 - (fileBytes.count % 64)) % 64
            if pad > 0 { fileBytes.append(Data(repeating: 0, count: pad)) }
            let expOff = fileBytes.count
            fileBytes.append(denseSlab)
            fileBytes.append(denseSlab)

            let hdrObj: [String: Any] = [
                "provenance": [
                    "geometry_hash": QWQuant.geometryHash,
                    "model_id": "qwen3.8-27b",
                    "geometry_digest": QWModelDescriptor.qwen38_27b.geometryDigestV2Hex,
                ],
                "tensors": [
                    "w_sub2": [
                        "shape": [rows, cols],
                        "hq4_off": denseOff,
                        "hq": ["bits": bits, "group": group, "vq": "shvq1"],
                    ],
                    "w_exp_sub2": [
                        "shape": [2, rows, cols],
                        "hq4_off": expOff,
                        "experts": 2,
                        "e_hq4": denseSlab.count,
                        "hq": ["bits": bits, "group": group, "vq": "shvq1"],
                    ],
                ],
            ]
            let hdrData = try JSONSerialization.data(withJSONObject: hdrObj, options: [.sortedKeys])
            var hdrLenLE = UInt64(hdrData.count).littleEndian
            withUnsafeBytes(of: &hdrLenLE) { raw in
                fileBytes.replaceSubrange(4..<12, with: raw)
            }
            fileBytes.append(hdrData)

            let qwpURL = tmpDir.appendingPathComponent("model.qwp")
            try fileBytes.write(to: qwpURL)
            let pack = try QWQuant.Pack(url: qwpURL, descriptor: .qwen38_27b, skipIdentityCheck: true)
            let entry = try pack.entry("w_sub2")
            XCTAssertTrue(entry.isSHVQ)
            XCTAssertEqual(entry.hqBlockBytes, gBytes)

            let deq = try pack.dequantQ4g("w_sub2")
            let expectedFlat = expectedRotRows.flatMap { $0 }
            XCTAssertEqual(deq.count, expectedFlat.count)
            for i in 0..<deq.count {
                XCTAssertEqual(deq[i], expectedFlat[i], accuracy: 1e-6)
            }
            let deqExp1 = try pack.dequantExpertQ4g("w_exp_sub2", expert: 1)
            for i in 0..<deqExp1.count {
                XCTAssertEqual(deqExp1[i], expectedFlat[i], accuracy: 1e-6)
            }
        }
    }
}
