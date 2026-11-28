// QWLinearWeightTests.swift — Unit tests for QWLinearWeight BF16 and quantized variants.
import Metal
import QWAlloc
import QWCore
@testable import QWGEMM
import QWQuant
import XCTest

final class QWLinearWeightTests: XCTestCase {
    func testBF16Properties() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal device available")
        }
        let n = 16
        let k = 64
        let expectedBytes = n * k * 2
        guard let buf = device.makeBuffer(length: expectedBytes, options: .storageModeShared) else {
            XCTFail("Failed to allocate Metal buffer")
            return
        }
        let slice = QWFPBufferSlice(buffer: buf, offset: 0, bytes: expectedBytes)
        let weight = QWLinearWeight.bf16(QWGemmFP.BF16Bank(weights: slice, n: n, k: k))

        XCTAssertEqual(weight.n, n)
        XCTAssertEqual(weight.k, k)
        XCTAssertFalse(weight.isQuant)
        XCTAssertFalse(weight.needsRotation)
        XCTAssertEqual(weight.residentBytes, expectedBytes)
    }

    func testQuantPropertiesAcrossContainers() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal device available")
        }
        let n = 8
        let k = 128
        let cases: [(kind: QWWeightView.ContainerKind, rotated: Bool, expectedBytes: Int)] = [
            (.incumbentHQ4G32, true, n * (k / 32) * 20),
            (.nativeAffine(bits: 3, group: 64), false, n * (k / 64) * 28),
            (.shvq(bits: 4, group: 32), true, n * (k / 32) * QWSHVQBlock.bytes(bits: 4, group: 32)),
        ]
        for entry in cases {
            let allocBytes = entry.expectedBytes + 256
            guard let buf = device.makeBuffer(length: allocBytes, options: .storageModeShared) else {
                XCTFail("Failed to allocate Metal buffer")
                return
            }
            let region = QWMRegion(buffer: buf, base: 0, length: buf.length)
            let tv = QWTensorValue(region: region, offset: 0, shape: QWTensorShape([allocBytes / 2], .f16))
            let view = try XCTUnwrap(QWWeightView(blocks: tv, n: n, k: k, containerKind: entry.kind))
            let weight = QWLinearWeight.quant(view)

            XCTAssertEqual(weight.n, n)
            XCTAssertEqual(weight.k, k)
            XCTAssertTrue(weight.isQuant)
            XCTAssertEqual(weight.needsRotation, entry.rotated)
            XCTAssertEqual(weight.residentBytes, entry.expectedBytes)
        }
    }
}
