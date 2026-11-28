// QWGDNSigmoidScaleTests.swift — CPU-vs-Metal parity, batch/rows bit-exact parity,
// and chunked prefill vs streamed decode tests for GDN `sigmoid * 2.0` gated RMSNorm.
import Foundation
import Metal
import XCTest
@testable import QWGDN
import QWAlloc
import QWCore
import QWHAL
import QWQuant

final class QWGDNSigmoidScaleTests: XCTestCase {
    static let geo = QWModelGeometry.qwen38_27b
    static let hal: QWHal = { try! QWHal(geometry: geo) }()

    override func setUpWithError() throws {
        let onCI = ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] == "true"
        try XCTSkipUnless(!onCI && Self.hal.caps.isMetal4TensorCapable,
                          "GDN sigmoid gate-scale tests require bare-metal Apple Silicon GPU")
    }

    private func sharedTV(_ count: Int, _ dt: QWDType, baseBytes: Int = 0) throws -> QWTensorValue {
        let byteLen = max(16, Int((Double(count) * dt.bytesPerParam).rounded()))
        let buf = try XCTUnwrap(Self.hal.device.makeBuffer(length: max(64, baseBytes + byteLen), options: .storageModeShared))
        memset(buf.contents(), 0, buf.length)
        return QWTensorValue(region: QWMRegion(buffer: buf, base: baseBytes, length: byteLen),
                             offset: 0, shape: QWTensorShape([count], dt))
    }

    private func fillDetF16(_ tv: QWTensorValue, count: Int, seed: Int, scale: Float = 0.35) {
        let ptr = tv.region.pointer.advanced(by: tv.offset).assumingMemoryBound(to: UInt16.self)
        for i in 0..<count {
            let v = sin(Float(i + seed * 37 + 1) * 0.029) * scale
            ptr[i] = Float16(v).bitPattern
        }
    }

    private func fillDetF32(_ tv: QWTensorValue, count: Int, seed: Int, scale: Float = 0.25, negShift: Float = 0.0) {
        let ptr = tv.region.pointer.advanced(by: tv.offset).assumingMemoryBound(to: Float.self)
        for i in 0..<count {
            ptr[i] = cos(Float(i + seed * 53 + 1) * 0.019) * scale + negShift
        }
    }

    private func readU16(_ tv: QWTensorValue, count: Int) -> [UInt16] {
        let ptr = tv.region.pointer.advanced(by: tv.offset).assumingMemoryBound(to: UInt16.self)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    private func readF32(_ tv: QWTensorValue, count: Int) -> [Float] {
        let ptr = tv.region.pointer.advanced(by: tv.offset).assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    private func cpuGatedNormRow(oRaw: [Float], zBits: [UInt16], wBits: [UInt16],
                                 vHeads: Int, headDim: Int, eps: Float,
                                 sigmoid: Bool, gateScale: Float, rotate: Bool) -> [Float] {
        var out = [Float](repeating: 0, count: vHeads * headDim)
        for h in 0..<vHeads {
            let base = h * headDim
            var sumSq: Float = 0
            for d in 0..<headDim {
                let v = oRaw[base + d]
                sumSq += v * v
            }
            let rms = 1.0 / sqrt(sumSq / Float(headDim) + eps)
            var headRow = [Float](repeating: 0, count: headDim)
            for d in 0..<headDim {
                let zn = Float(Float16(bitPattern: zBits[base + d]))
                var gate: Float = sigmoid ? (1.0 / (1.0 + exp(-zn))) : (zn / (1.0 + exp(-zn)))
                if gateScale != 1.0 { gate *= gateScale }
                let wc = Float(Float16(bitPattern: wBits[d]))
                let hg = Float16((oRaw[base + d] * rms * wc) * gate)
                headRow[d] = Float(hg)
            }
            if rotate {
                headRow.withUnsafeMutableBufferPointer { buf in
                    guard let p = buf.baseAddress else { return }
                    for b in stride(from: 0, to: headDim, by: 32) {
                        QWQuant.fwht32OrthoBlock(p + b)
                    }
                }
            }
            for d in 0..<headDim {
                out[base + d] = Float(Float16(headRow[d]))
            }
        }
        return out
    }

    func testRecurrentDecodeSigmoidGateScale2MatchesCPUReference() throws {
        let vh = Self.geo.gdnVHeads
        let hd = Self.geo.gdnHeadDim
        let qkv = Self.geo.gdnQKVChannels
        let zCh = Self.geo.gdnZChannels
        let gCh = Self.geo.gdnGlogChannels
        let stateElems = vh * hd * hd

        let decSig2 = try QWGDNDecode(hal: Self.hal, normSigmoid: true, gateScale: 2.0)
        let decSig1 = try QWGDNDecode(hal: Self.hal, normSigmoid: true, gateScale: 1.0)

        let y = try sharedTV(qkv, .f16)
        let betas = try sharedTV(vh, .f32)
        let glog = try sharedTV(gCh, .f32)
        let z = try sharedTV(zCh, .f16)
        let w = try sharedTV(hd, .f16)
        fillDetF16(y, count: qkv, seed: 11, scale: 0.4)
        fillDetF32(betas, count: vh, seed: 13, scale: 0.5)
        fillDetF32(glog, count: gCh, seed: 17, scale: 0.05, negShift: -0.08)
        fillDetF16(z, count: zCh, seed: 23, scale: 1.25)
        fillDetF16(w, count: hd, seed: 29, scale: 0.85)

        let st2 = try sharedTV(stateElems, .f32)
        let st1 = try sharedTV(stateElems, .f32)
        fillDetF32(st2, count: stateElems, seed: 31, scale: 0.1)
        memcpy(st1.region.pointer, st2.region.pointer, stateElems * 4)

        let o2 = try sharedTV(zCh, .f16)
        let o1 = try sharedTV(zCh, .f16)
        let scr2 = try sharedTV(zCh, .f32)
        let scr1 = try sharedTV(zCh, .f32)

        guard let (cb, enc) = Self.hal.begin() else { XCTFail("hal.begin"); return }
        try decSig2.encode(enc, y: y, betas: betas, glog: glog, state: st2, z: z, w: w, o: o2, oScratch: scr2)
        try decSig1.encode(enc, y: y, betas: betas, glog: glog, state: st1, z: z, w: w, o: o1, oScratch: scr1)
        Self.hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)

        let raw2 = readF32(scr2, count: zCh)
        let raw1 = readF32(scr1, count: zCh)
        for i in 0..<zCh {
            XCTAssertEqual(raw2[i].bitPattern, raw1[i].bitPattern, "oScratch mismatch at \(i)")
        }

        let rot = Self.hal.weightBasis.rotatesActivations
        let cpuRef2 = cpuGatedNormRow(
            oRaw: raw2, zBits: readU16(z, count: zCh), wBits: readU16(w, count: hd),
            vHeads: vh, headDim: hd, eps: Self.geo.rmsEps,
            sigmoid: true, gateScale: 2.0, rotate: rot
        )
        let got2 = readU16(o2, count: zCh).map { Float(Float16(bitPattern: $0)) }
        let got1 = readU16(o1, count: zCh).map { Float(Float16(bitPattern: $0)) }

        var maxErr: Float = 0
        var maxScaleDiff: Float = 0
        for i in 0..<zCh {
            maxErr = max(maxErr, abs(got2[i] - cpuRef2[i]))
            maxScaleDiff = max(maxScaleDiff, abs(got2[i] - 2.0 * got1[i]))
        }
        XCTAssertLessThanOrEqual(maxErr, 2e-3, "Sigmoid*2.0 GPU vs CPU maxErr \(maxErr)")
        XCTAssertLessThanOrEqual(maxScaleDiff, 4e-3, "gateScale=2.0 vs 2x(gateScale=1.0) diff \(maxScaleDiff)")
    }

    func testBatchDecodeAndMultiRowVerifySigmoidScale2BitExactParity() throws {
        let vh = Self.geo.gdnVHeads
        let hd = Self.geo.gdnHeadDim
        let qkv = Self.geo.gdnQKVChannels
        let zCh = Self.geo.gdnZChannels
        let gCh = Self.geo.gdnGlogChannels
        let stateElems = vh * hd * hd

        let dec = try QWGDNDecode(hal: Self.hal, normSigmoid: true, gateScale: 2.0)
        let w = try sharedTV(hd, .f16)
        fillDetF16(w, count: hd, seed: 41, scale: 0.9)

        let B = 4
        let yBat = try sharedTV(B * qkv, .f16)
        let bBat = try sharedTV(B * vh, .f32)
        let gBat = try sharedTV(B * gCh, .f32)
        let zBat = try sharedTV(B * zCh, .f16)
        fillDetF16(yBat, count: B * qkv, seed: 43, scale: 0.35)
        fillDetF32(bBat, count: B * vh, seed: 47, scale: 0.45)
        fillDetF32(gBat, count: B * gCh, seed: 53, scale: 0.04, negShift: -0.07)
        fillDetF16(zBat, count: B * zCh, seed: 59, scale: 1.1)

        var serStates: [QWTensorValue] = []
        var batStates: [QWTensorValue] = []
        for b in 0..<B {
            let s0 = try sharedTV(stateElems, .f32)
            let s1 = try sharedTV(stateElems, .f32)
            fillDetF32(s0, count: stateElems, seed: 100 + b, scale: 0.08)
            memcpy(s1.region.pointer, s0.region.pointer, stateElems * 4)
            serStates.append(s0)
            batStates.append(s1)
        }
        let oSer = try sharedTV(B * zCh, .f16)
        let oBat = try sharedTV(B * zCh, .f16)
        let scrSer = try sharedTV(zCh, .f32)

        guard let (cb, enc) = Self.hal.begin() else { XCTFail("hal.begin"); return }
        for b in 0..<B {
            try dec.encode(
                enc,
                y: yBat.slice(byteOffset: b * qkv * 2, count: qkv, dtype: .f16),
                betas: bBat.slice(byteOffset: b * vh * 4, count: vh, dtype: .f32),
                glog: gBat.slice(byteOffset: b * gCh * 4, count: gCh, dtype: .f32),
                state: serStates[b],
                z: zBat.slice(byteOffset: b * zCh * 2, count: zCh, dtype: .f16),
                w: w,
                o: oSer.slice(byteOffset: b * zCh * 2, count: zCh, dtype: .f16),
                oScratch: scrSer
            )
        }
        try dec.encodeBatch(
            enc, y: yBat, yStride: qkv, betas: bBat, betasStride: vh,
            glog: gBat, glogStride: gCh, states: batStates,
            z: zBat, zStride: zCh, w: w, o: oBat, oStride: zCh
        )
        Self.hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)

        let uSer = readU16(oSer, count: B * zCh)
        let uBat = readU16(oBat, count: B * zCh)
        XCTAssertEqual(uSer, uBat, "encodeBatch must be 0-ULP bit-exact vs serial encode")
        for b in 0..<B {
            let stS = readF32(serStates[b], count: stateElems).map(\.bitPattern)
            let stB = readF32(batStates[b], count: stateElems).map(\.bitPattern)
            XCTAssertEqual(stS, stB, "Batch slot \(b) state must be 0-ULP bit-exact")
        }

        let R = 4
        let stSeq = try sharedTV(stateElems, .f32)
        let stRows = try sharedTV(stateElems, .f32)
        fillDetF32(stSeq, count: stateElems, seed: 211, scale: 0.08)
        memcpy(stRows.region.pointer, stSeq.region.pointer, stateElems * 4)
        let oSeq = try sharedTV(R * zCh, .f16)
        let oRows = try sharedTV(R * zCh, .f16)

        guard let (cb2, enc2) = Self.hal.begin() else { XCTFail("hal.begin"); return }
        for r in 0..<R {
            try dec.encode(
                enc2,
                y: yBat.slice(byteOffset: r * qkv * 2, count: qkv, dtype: .f16),
                betas: bBat.slice(byteOffset: r * vh * 4, count: vh, dtype: .f32),
                glog: gBat.slice(byteOffset: r * gCh * 4, count: gCh, dtype: .f32),
                state: stSeq,
                z: zBat.slice(byteOffset: r * zCh * 2, count: zCh, dtype: .f16),
                w: w,
                o: oSeq.slice(byteOffset: r * zCh * 2, count: zCh, dtype: .f16),
                oScratch: scrSer
            )
        }
        try dec.encodeRows(
            enc2, y: yBat, betas: bBat, glog: gBat, state: stRows,
            zBase: zBat, w: w, o: oRows, rows: R
        )
        Self.hal.end(cb2, enc2)
        cb2.waitUntilCompleted()
        XCTAssertNil(cb2.error)

        XCTAssertEqual(readU16(oSeq, count: R * zCh), readU16(oRows, count: R * zCh),
                       "encodeRows must be 0-ULP bit-exact vs serial encode")
        XCTAssertEqual(readF32(stSeq, count: stateElems).map(\.bitPattern),
                       readF32(stRows, count: stateElems).map(\.bitPattern))
    }

    func testPrefillSigmoidGateScale2MatchesStreamedDecode() throws {
        let vh = Self.geo.gdnVHeads
        let hd = Self.geo.gdnHeadDim
        let qkv = Self.geo.gdnQKVChannels
        let zCh = Self.geo.gdnZChannels
        let gCh = Self.geo.gdnGlogChannels
        let stateElems = vh * hd * hd
        let T = 66
        let L = 64

        let pre = try QWGDNPrefill(hal: Self.hal, normSigmoid: true, gateScale: 2.0)
        let dec = try QWGDNDecode(hal: Self.hal, normSigmoid: true, gateScale: 2.0)

        let y = try sharedTV(T * qkv, .f16)
        let betas = try sharedTV(T * vh, .f32)
        let glog = try sharedTV(T * gCh, .f32)
        let z = try sharedTV(T * zCh, .f16)
        let w = try sharedTV(hd, .f16)
        fillDetF16(y, count: T * qkv, seed: 71, scale: 0.25)
        fillDetF32(betas, count: T * vh, seed: 73, scale: 0.35)
        fillDetF32(glog, count: T * gCh, seed: 79, scale: 0.03, negShift: -0.06)
        fillDetF16(z, count: T * zCh, seed: 83, scale: 1.0)
        fillDetF16(w, count: hd, seed: 89, scale: 0.9)

        let stPre = try sharedTV(stateElems, .f32)
        let stDec = try sharedTV(stateElems, .f32)
        fillDetF32(stPre, count: stateElems, seed: 97, scale: 0.05)
        memcpy(stDec.region.pointer, stPre.region.pointer, stateElems * 4)

        let uw = QWGDNPrefill.uwabElems(T: T)
        let U = try sharedTV(uw, .f32)
        let W = try sharedTV(uw, .f32)
        let A = try sharedTV(uw, .f32)
        let B = try sharedTV(uw, .f32)
        let oScrPre = try sharedTV(QWGDNPrefill.oScratchElems(T: T), .f32)
        let oPre = try sharedTV(T * zCh, .f16)
        let oDec = try sharedTV(T * zCh, .f16)
        let oScrDec = try sharedTV(zCh, .f32)

        guard let (cb, enc) = Self.hal.begin() else { XCTFail("hal.begin"); return }
        try pre.encode(enc, y: y, betas: betas, glog: glog, z: z, w: w,
                       state: stPre, o: oPre, U: U, W: W, A: A, B: B, oScratch: oScrPre, T: T, L: L)
        for t in 0..<T {
            try dec.encode(
                enc,
                y: y.slice(byteOffset: t * qkv * 2, count: qkv, dtype: .f16),
                betas: betas.slice(byteOffset: t * vh * 4, count: vh, dtype: .f32),
                glog: glog.slice(byteOffset: t * gCh * 4, count: gCh, dtype: .f32),
                state: stDec,
                z: z.slice(byteOffset: t * zCh * 2, count: zCh, dtype: .f16),
                w: w,
                o: oDec.slice(byteOffset: t * zCh * 2, count: zCh, dtype: .f16),
                oScratch: oScrDec
            )
        }
        Self.hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)

        let gotPre = readU16(oPre, count: T * zCh).map { Float(Float16(bitPattern: $0)) }
        let gotDec = readU16(oDec, count: T * zCh).map { Float(Float16(bitPattern: $0)) }
        var maxDiff: Float = 0
        for i in 0..<(T * zCh) {
            maxDiff = max(maxDiff, abs(gotPre[i] - gotDec[i]))
        }
        XCTAssertLessThan(maxDiff, 3e-3, "Chunked prefill vs streamed decode sigmoid*2.0 diff \(maxDiff)")
    }
}
