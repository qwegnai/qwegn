// QWSHVQTests.swift — Size law, canonical codebook invariants, 0-ULP CPU
// unpack/dot parity, and QWP1 container dequantization tests for SHVQ.
import Foundation
import XCTest
@testable import QWCore
@testable import QWQuant

final class QWSHVQTests: XCTestCase {

    func testSupportedPairsAndByteLayouts() {
        XCTAssertEqual(QWSHVQBlock.supported.count, 12)
        let expectedBPW: [Int: [Int: Double]] = [
            2: [32: 2.5, 64: 2.25, 128: 2.125, 256: 2.0625],
            3: [32: 3.5, 64: 3.25, 128: 3.125, 256: 3.0625],
            4: [32: 4.5, 64: 4.25, 128: 4.125, 256: 4.0625],
        ]
        for (bits, group) in QWSHVQBlock.supported {
            XCTAssertTrue(QWSHVQBlock.isSupported(bits: bits, group: group))
            let b = QWSHVQBlock.bytes(bits: bits, group: group)
            XCTAssertEqual(b, 2 + (group * bits) / 8)
            XCTAssertEqual(QWSHVQBlock.bytesIfSupported(bits: bits, group: group), b)
            XCTAssertEqual(QWSHVQBlock.bitsPerWeight(bits: bits, group: group),
                           expectedBPW[bits]![group]!, accuracy: 1e-12)
            XCTAssertEqual(QWSHVQBlock.formatName(bits: bits, group: group),
                           "SHVQ\(bits)_G\(group)")
            // Verify 2-byte savings per group over HQ/AQ (which carry 4-byte scale+bias).
            XCTAssertEqual(QWAQBlock.bytes(bits: bits, group: group) - b, 2)
        }
        XCTAssertFalse(QWSHVQBlock.isSupported(bits: 5, group: 32))
        XCTAssertNil(QWSHVQBlock.bytesIfSupported(bits: 6, group: 32))
    }

    func testCanonicalCodebookInvariants() {
        XCTAssertEqual(QWSHVQTables.lut2Bits.count, 256 * 8)
        XCTAssertEqual(QWSHVQTables.lut3Bits.count, 256 * 4)
        XCTAssertEqual(QWSHVQTables.lut4PackedU8.count, 4096)
        XCTAssertEqual(QWSHVQTables.stepMap4Bits.count, 256)

        for bits in QWSHVQTables.lut2Bits {
            XCTAssertEqual(bits & 0x8000, 0, "lut2 magnitude entries must have zero sign bit")
            XCTAssertFalse(Float16(bitPattern: bits).isNaN)
        }
        for bits in QWSHVQTables.lut3Bits {
            XCTAssertEqual(bits & 0x8000, 0, "lut3 magnitude entries must have zero sign bit")
            XCTAssertFalse(Float16(bitPattern: bits).isNaN)
        }
        for bits in QWSHVQTables.stepMap4Bits {
            XCTAssertEqual(bits & 0x8000, 0, "stepMap4 entries must have zero sign bit")
            XCTAssertFalse(Float16(bitPattern: bits).isNaN)
        }
        for packed in QWSHVQTables.lut4PackedU8 {
            for j in 0..<4 {
                let step = Int((packed >> (j * 8)) & 0xFF)
                XCTAssertLessThan(step, 256, "lut4 packed index must fall in 0..<256")
            }
        }
    }

    func testUnpackSignFlipAndScaleZeroULP() {
        for (bits, group) in QWSHVQBlock.supported {
            let groupBytes = QWSHVQBlock.bytes(bits: bits, group: group)
            var posBlock = [UInt8](repeating: 0, count: groupBytes)
            var negBlock = [UInt8](repeating: 0, count: groupBytes)
            let scBits = Float16(0.0625).bitPattern
            posBlock[0] = UInt8(scBits & 0xFF)
            posBlock[1] = UInt8(scBits >> 8)
            negBlock[0] = posBlock[0]
            negBlock[1] = posBlock[1]

            let nOct = group / 8
            if bits == 2 {
                for oct in 0..<nOct {
                    let code = UInt8((oct * 37 + 19) & 0xFF)
                    posBlock[2 + 2 * oct] = 0x00
                    posBlock[2 + 2 * oct + 1] = code
                    negBlock[2 + 2 * oct] = 0xFF
                    negBlock[2 + 2 * oct + 1] = code
                }
            } else if bits == 3 {
                for oct in 0..<nOct {
                    let c0 = UInt8((oct * 29 + 7) & 0xFF)
                    let c1 = UInt8((oct * 53 + 11) & 0xFF)
                    posBlock[2 + 3 * oct] = 0x00
                    posBlock[2 + 3 * oct + 1] = c0
                    posBlock[2 + 3 * oct + 2] = c1
                    negBlock[2 + 3 * oct] = 0xFF
                    negBlock[2 + 3 * oct + 1] = c0
                    negBlock[2 + 3 * oct + 2] = c1
                }
            } else {
                for sub in 0..<(nOct * 2) {
                    let idx12 = UInt16((sub * 251 + 43) & 0x0FFF)
                    let posU16 = idx12
                    let negU16 = idx12 | 0xF000
                    posBlock[2 + 2 * sub] = UInt8(posU16 & 0xFF)
                    posBlock[2 + 2 * sub + 1] = UInt8(posU16 >> 8)
                    negBlock[2 + 2 * sub] = UInt8(negU16 & 0xFF)
                    negBlock[2 + 2 * sub + 1] = UInt8(negU16 >> 8)
                }
            }

            var posOut = [Float]()
            var negOut = [Float]()
            posBlock.withUnsafeBytes { QWSHVQBlock.unpack(group: $0, bits: bits, group: group, into: &posOut) }
            negBlock.withUnsafeBytes { QWSHVQBlock.unpack(group: $0, bits: bits, group: group, into: &negOut) }
            XCTAssertEqual(posOut.count, group)
            XCTAssertEqual(negOut.count, group)
            for i in 0..<group {
                XCTAssertEqual(negOut[i], -posOut[i],
                               "SHVQ\(bits)_G\(group) lane \(i) must flip sign bit-exact")
            }

            let xs = (0..<group).map { Float(($0 % 17) - 8) * 0.125 }
            let posDot = posBlock.withUnsafeBytes { QWSHVQBlock.dot(group: $0, bits: bits, group: group, x: xs) }
            let negDot = negBlock.withUnsafeBytes { QWSHVQBlock.dot(group: $0, bits: bits, group: group, x: xs) }
            XCTAssertEqual(negDot, -posDot, accuracy: 1e-6)
        }
    }

    func testQWP1PackDequantSHVQ2DAndExpertStack() throws {
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwg_shvq_test_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let rows = 4, cols = 64
        let bits = 3, group = 64
        let gBytes = QWSHVQBlock.bytes(bits: bits, group: group)
        var denseSlab = Data()
        var expectedRotRows = [[Float]]()
        for r in 0..<rows {
            var block = [UInt8](repeating: 0, count: gBytes)
            let sc = Float16(0.03125 * Float(r + 1)).bitPattern
            block[0] = UInt8(sc & 0xFF)
            block[1] = UInt8(sc >> 8)
            for oct in 0..<(group / 8) {
                block[2 + 3 * oct] = UInt8((r * 19 + oct * 41) & 0xFF)
                block[2 + 3 * oct + 1] = UInt8((r * 7 + oct * 13) & 0xFF)
                block[2 + 3 * oct + 2] = UInt8((r * 11 + oct * 23) & 0xFF)
            }
            var unpacked = [Float]()
            block.withUnsafeBytes { QWSHVQBlock.unpack(group: $0, bits: bits, group: group, into: &unpacked) }
            // Inverse 32-point orthonormal FWHT matches QWQuant.Pack.dequantQ4g.
            unpacked.withUnsafeMutableBufferPointer { buf in
                for b in stride(from: 0, to: cols, by: 32) {
                    QWQuant.fwht32OrthoBlock(buf.baseAddress! + b)
                }
            }
            expectedRotRows.append(unpacked)
            denseSlab.append(contentsOf: block)
        }

        // Build minimal QWP1 file with both a 2D tensor and a 2-expert stacked tensor.
        var fileBytes = Data("QWP1".utf8)
        fileBytes.append(Data(repeating: 0, count: 60)) // 8B hdrLen + pad to offset 64
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
                "w_shvq3": [
                    "shape": [rows, cols],
                    "hq4_off": denseOff,
                    "hq": ["bits": bits, "group": group, "vq": "shvq1"],
                ],
                "w_exp_shvq3": [
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
        let entry = try pack.entry("w_shvq3")
        XCTAssertTrue(entry.isSHVQ)
        XCTAssertEqual(entry.hqBlockBytes, gBytes)

        let deq = try pack.dequantQ4g("w_shvq3")
        let expectedFlat = expectedRotRows.flatMap { $0 }
        XCTAssertEqual(deq.count, expectedFlat.count)
        for i in 0..<deq.count {
            XCTAssertEqual(deq[i], expectedFlat[i], accuracy: 1e-6)
        }

        let deqExp1 = try pack.dequantExpertQ4g("w_exp_shvq3", expert: 1)
        for i in 0..<deqExp1.count {
            XCTAssertEqual(deqExp1[i], expectedFlat[i], accuracy: 1e-6)
        }
    }
}
