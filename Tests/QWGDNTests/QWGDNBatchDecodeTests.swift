// QWGDNBatchDecodeTests.swift — Synthetic 0-ULP parity & overlap checks for batched GDN/KDA.
import XCTest
import Metal
import Foundation
@testable import QWGDN
import QWCore
import QWHAL
import QWAlloc

final class QWGDNBatchDecodeTests: XCTestCase {
    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    override func setUpWithError() throws {
        let onCI = ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] == "true"
        try XCTSkipUnless(!onCI && Self.hal.caps.isMetal4TensorCapable,
                          "1024-thread GDN recurrent batch kernel requires bare-metal Apple Silicon GPU")
    }

    private func sharedTV(_ dev: any MTLDevice, _ count: Int, _ dt: QWDType, baseBytes: Int = 0) throws -> QWTensorValue {
        let byteLen = Int((Double(count) * dt.bytesPerParam).rounded())
        let buf = try XCTUnwrap(dev.makeBuffer(length: max(64, baseBytes + byteLen), options: .storageModeShared))
        let region = QWMRegion(buffer: buf, base: baseBytes, length: byteLen)
        return QWTensorValue(region: region, offset: 0, shape: QWTensorShape([count], dt))
    }

    private func fillDeterministicF16(_ tv: QWTensorValue, count: Int, seed: Int) {
        let ptr = tv.region.pointer.advanced(by: tv.offset).assumingMemoryBound(to: UInt16.self)
        for i in 0..<count {
            let v = sin(Float(i + seed * 37 + 1) * 0.031) * 0.35
            ptr[i] = Float16(v).bitPattern
        }
    }

    private func fillDeterministicF32(_ tv: QWTensorValue, count: Int, seed: Int, scale: Float = 0.25) {
        let ptr = tv.region.pointer.advanced(by: tv.offset).assumingMemoryBound(to: Float.self)
        for i in 0..<count {
            ptr[i] = cos(Float(i + seed * 53 + 1) * 0.023) * scale
        }
    }

    private func assertBitExactF16(_ a: QWTensorValue, _ b: QWTensorValue, count: Int, label: String) {
        let pa = a.region.pointer.advanced(by: a.offset).assumingMemoryBound(to: UInt16.self)
        let pb = b.region.pointer.advanced(by: b.offset).assumingMemoryBound(to: UInt16.self)
        for i in 0..<count {
            if pa[i] != pb[i] {
                XCTFail("\(label) f16 mismatch at [\(i)]: 0x\(String(pa[i], radix: 16)) vs 0x\(String(pb[i], radix: 16))")
                return
            }
        }
    }

    private func assertBitExactF32(_ a: QWTensorValue, _ b: QWTensorValue, count: Int, label: String) {
        let pa = a.region.pointer.advanced(by: a.offset).assumingMemoryBound(to: UInt32.self)
        let pb = b.region.pointer.advanced(by: b.offset).assumingMemoryBound(to: UInt32.self)
        for i in 0..<count {
            if pa[i] != pb[i] {
                XCTFail("\(label) f32 mismatch at [\(i)]: 0x\(String(pa[i], radix: 16)) vs 0x\(String(pb[i], radix: 16))")
                return
            }
        }
    }

    private func runBatchParity(hal: QWHal, geometry: QWModelGeometry, batchSizes: [Int],
                                xStride: Int, zStride: Int, zInXOffset: Int,
                                betasStride: Int, glogStride: Int, glogInControlsOffset: Int) throws {
        let conv = try QWGDNConv(hal: hal, geometry: geometry)
        let dec = try QWGDNDecode(hal: hal)
        let qkvC = geometry.gdnQKVChannels
        let zC = geometry.gdnZChannels
        let vh = geometry.gdnVHeads
        let hd = geometry.gdnHeadDim
        let gCh = geometry.gdnGlogChannels
        let ringElems = QWGDNConv.stateRows * qkvC
        let recElems = vh * hd * hd

        let convW = try sharedTV(hal.device, qkvC * 4, .f16)
        fillDeterministicF16(convW, count: qkvC * 4, seed: 3)
        let normW = try sharedTV(hal.device, hd, .f16)
        fillDeterministicF16(normW, count: hd, seed: 7)

        for B in batchSizes {
            let xBufTV = try sharedTV(hal.device, B * xStride, .f16, baseBytes: 256)
            fillDeterministicF16(xBufTV, count: B * xStride, seed: 11 + B)
            let zBufTV = (zInXOffset > 0)
                ? QWTensorValue(region: xBufTV.region, offset: zInXOffset * 2,
                                shape: QWTensorShape([(B - 1) * zStride + zC], .f16))
                : try sharedTV(hal.device, B * zStride, .f16, baseBytes: 128)
            if zInXOffset == 0 { fillDeterministicF16(zBufTV, count: B * zStride, seed: 19 + B) }

            let ctrlElems = glogInControlsOffset > 0 ? B * betasStride : B * betasStride
            let betasBufTV = try sharedTV(hal.device, ctrlElems, .f32, baseBytes: 512)
            fillDeterministicF32(betasBufTV, count: ctrlElems, seed: 23 + B, scale: 0.4)
            let glogBufTV: QWTensorValue
            if glogInControlsOffset > 0 {
                glogBufTV = QWTensorValue(region: betasBufTV.region, offset: glogInControlsOffset * 4,
                                          shape: QWTensorShape([(B - 1) * glogStride + gCh], .f32))
            } else {
                let tv = try sharedTV(hal.device, B * glogStride, .f32, baseBytes: 384)
                let ptr = tv.region.pointer.advanced(by: tv.offset).assumingMemoryBound(to: Float.self)
                for i in 0..<(B * glogStride) {
                    let raw = -0.05 - Float(i % 17) * 0.01
                    ptr[i] = geometry.gdnVectorDecay ? exp(raw) : raw
                }
                glogBufTV = tv
            }

            let ringPad = 256
            let ringSpanBytes = ringPad + ringElems * 2
            let sharedRingBuf = try XCTUnwrap(hal.device.makeBuffer(length: B * ringSpanBytes, options: .storageModeShared))
            let recPad = 512
            let recSpanBytes = recPad + recElems * 4
            let sharedRecBuf = try XCTUnwrap(hal.device.makeBuffer(length: B * recSpanBytes, options: .storageModeShared))

            var serRing: [QWTensorValue] = [], batRing: [QWTensorValue] = []
            var serRec: [QWTensorValue] = [], batRec: [QWTensorValue] = []
            for b in 0..<B {
                let sr = try sharedTV(hal.device, ringElems, .f16)
                fillDeterministicF16(sr, count: ringElems, seed: 100 + b)
                let brReg = QWMRegion(buffer: sharedRingBuf, base: b * ringSpanBytes + ringPad, length: ringElems * 2)
                let br = QWTensorValue(region: brReg, offset: 0, shape: QWTensorShape([ringElems], .f16))
                br.region.pointer.copyMemory(from: sr.region.pointer, byteCount: ringElems * 2)
                serRing.append(sr); batRing.append(br)

                let srec = try sharedTV(hal.device, recElems, .f32)
                fillDeterministicF32(srec, count: recElems, seed: 200 + b, scale: 0.05)
                let brecReg = QWMRegion(buffer: sharedRecBuf, base: b * recSpanBytes + recPad, length: recElems * 4)
                let brec = QWTensorValue(region: brecReg, offset: 0, shape: QWTensorShape([recElems], .f32))
                brec.region.pointer.copyMemory(from: srec.region.pointer, byteCount: recElems * 4)
                serRec.append(srec); batRec.append(brec)
            }

            let serConvY = try sharedTV(hal.device, B * qkvC, .f16)
            let batConvY = try sharedTV(hal.device, B * qkvC, .f16)
            let serO = try sharedTV(hal.device, B * zC, .f16)
            let batO = try sharedTV(hal.device, B * zC, .f16)
            let serScratch = try sharedTV(hal.device, zC, .f32)

            let (cbS, encS) = try XCTUnwrap(hal.begin())
            for b in 0..<B {
                let xRow = QWTensorValue(region: xBufTV.region, offset: xBufTV.offset + b * xStride * 2,
                                         shape: QWTensorShape([qkvC], .f16))
                let zRow = QWTensorValue(region: zBufTV.region, offset: zBufTV.offset + b * zStride * 2,
                                         shape: QWTensorShape([zC], .f16))
                let bRow = QWTensorValue(region: betasBufTV.region, offset: betasBufTV.offset + b * betasStride * 4,
                                         shape: QWTensorShape([vh], .f32))
                let gRow = QWTensorValue(region: glogBufTV.region, offset: glogBufTV.offset + b * glogStride * 4,
                                         shape: QWTensorShape([gCh], .f32))
                let cRow = QWTensorValue(region: serConvY.region, offset: b * qkvC * 2,
                                         shape: QWTensorShape([qkvC], .f16))
                let oRow = QWTensorValue(region: serO.region, offset: b * zC * 2,
                                         shape: QWTensorShape([zC], .f16))
                try conv.encodeT1WithHeadnorm(encS, x: xRow, w: convW, bias: nil,
                                              stIn: serRing[b], y: cRow, stOut: serRing[b])
                try dec.encode(encS, y: cRow, betas: bRow, glog: gRow,
                               state: serRec[b], z: zRow, w: normW, o: oRow, oScratch: serScratch)
            }
            hal.end(cbS, encS)
            cbS.waitUntilCompleted()
            XCTAssertEqual(cbS.status, .completed)

            let (cbB, encB) = try XCTUnwrap(hal.begin())
            try conv.encodeBatchT1WithHeadnorm(
                encB, x: xBufTV, xStride: xStride, w: convW, bias: nil,
                y: batConvY, yStride: qkvC, rows: B, stateAt: { batRing[$0] })
            try dec.encodeBatch(
                encB, y: batConvY, yStride: qkvC,
                betas: betasBufTV, betasStride: betasStride,
                glog: glogBufTV, glogStride: glogStride,
                z: zBufTV, zStride: zStride, w: normW,
                o: batO, oStride: zC, rows: B, stateAt: { batRec[$0] })
            hal.end(cbB, encB)
            cbB.waitUntilCompleted()
            XCTAssertEqual(cbB.status, .completed)

            assertBitExactF16(batConvY, serConvY, count: B * qkvC, label: "convY(B=\(B))")
            assertBitExactF16(batO, serO, count: B * zC, label: "o(B=\(B))")
            for b in 0..<B {
                assertBitExactF16(batRing[b], serRing[b], count: ringElems, label: "ring[\(b)](B=\(B))")
                assertBitExactF32(batRec[b], serRec[b], count: recElems, label: "rec[\(b)](B=\(B))")
            }
        }
    }

    func testBatchConvAndRecurrentBitExactGDN() throws {
        let g = QWModelGeometry.qwen38_27b
        try runBatchParity(
            hal: Self.hal, geometry: g, batchSizes: [1, 2, 4, 8],
            xStride: g.gdnTotalN, zStride: g.gdnTotalN, zInXOffset: g.gdnQKVChannels,
            betasStride: 2 * g.gdnVHeads, glogStride: 2 * g.gdnVHeads, glogInControlsOffset: g.gdnVHeads)
    }

    func testBatchConvAndRecurrentBitExactKDA() throws {
        let desc = try QWModelDescriptor.decode(configJSON: QWGDNDecodeTests.kdaConfig)
        let kdaGeo = QWModelGeometry(descriptor: desc)
        let kdaHal = try QWHal(geometry: kdaGeo)
        try runBatchParity(
            hal: kdaHal, geometry: kdaGeo, batchSizes: [2, 4, 8],
            xStride: kdaGeo.gdnQKVChannels, zStride: kdaGeo.gdnZChannels, zInXOffset: 0,
            betasStride: kdaGeo.gdnVHeads, glogStride: kdaGeo.gdnGlogChannels, glogInControlsOffset: 0)
    }

    func testBatchStateOverlapThrows() throws {
        let g = QWModelGeometry.qwen38_27b
        let conv = try QWGDNConv(hal: Self.hal, geometry: g)
        let dec = try QWGDNDecode(hal: Self.hal)
        let qkvC = g.gdnQKVChannels, zC = g.gdnZChannels, vh = g.gdnVHeads, hd = g.gdnHeadDim
        let ringElems = QWGDNConv.stateRows * qkvC, recElems = vh * hd * hd
        let ringBuf = try XCTUnwrap(Self.hal.device.makeBuffer(length: ringElems * 2 + 128, options: .storageModeShared))
        let r0 = QWTensorValue(region: QWMRegion(buffer: ringBuf, base: 0, length: ringElems * 2),
                               offset: 0, shape: QWTensorShape([ringElems], .f16))
        let r1 = QWTensorValue(region: QWMRegion(buffer: ringBuf, base: 64, length: ringElems * 2),
                               offset: 0, shape: QWTensorShape([ringElems], .f16))
        let recBuf = try XCTUnwrap(Self.hal.device.makeBuffer(length: recElems * 4 + 128, options: .storageModeShared))
        let s0 = QWTensorValue(region: QWMRegion(buffer: recBuf, base: 0, length: recElems * 4),
                               offset: 0, shape: QWTensorShape([recElems], .f32))
        let s1 = QWTensorValue(region: QWMRegion(buffer: recBuf, base: 64, length: recElems * 4),
                               offset: 0, shape: QWTensorShape([recElems], .f32))

        let x = try sharedTV(Self.hal.device, 2 * qkvC, .f16)
        let w = try sharedTV(Self.hal.device, qkvC * 4, .f16)
        let y = try sharedTV(Self.hal.device, 2 * qkvC, .f16)
        let betas = try sharedTV(Self.hal.device, 2 * vh, .f32)
        let glog = try sharedTV(Self.hal.device, 2 * g.gdnGlogChannels, .f32)
        let z = try sharedTV(Self.hal.device, 2 * zC, .f16)
        let nw = try sharedTV(Self.hal.device, hd, .f16)
        let o = try sharedTV(Self.hal.device, 2 * zC, .f16)

        let (cb, enc) = try XCTUnwrap(Self.hal.begin())
        XCTAssertThrowsError(try conv.encodeBatchT1WithHeadnorm(
            enc, x: x, xStride: qkvC, w: w, bias: nil, states: [r0, r1], y: y, yStride: qkvC))
        XCTAssertThrowsError(try dec.encodeBatch(
            enc, y: y, yStride: qkvC, betas: betas, betasStride: vh,
            glog: glog, glogStride: g.gdnGlogChannels, states: [s0, s1],
            z: z, zStride: zC, w: nw, o: o, oStride: zC))
        Self.hal.end(cb, enc)
        cb.waitUntilCompleted()
    }
}
