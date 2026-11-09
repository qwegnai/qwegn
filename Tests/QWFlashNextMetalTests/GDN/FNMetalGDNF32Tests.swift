import XCTest
import Foundation
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWGDN
@testable import QWFlashNext

// f32 GDN kernels vs their fp16 originals on the same random inputs (conv, prologue, recurrence readout).
final class FNMetalGDNF32Tests: XCTestCase {
    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    static func lcg(_ n: Int, seed: UInt64, scale: Float) -> [Float] {
        var s = seed
        return (0..<n).map { _ in
            s = s &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int64(bitPattern: s >> 33) % 2001 - 1000) / 1000 * scale
        }
    }

    func testF32GDNKernelsMatchFp16Originals() throws {
        let T = 2, C = 10240, VH = 48, D = 128
        let arena = try QWArena(hal: Self.hal, weights: 64 << 20, activations: 512 << 20, kv: 0, gdnStates: 0)
        // The context prewarms the doors it is given; this test wants
        // the kernel objects, not a prewarm.
        let ctx = try FNGPUContext(hal: Self.hal, doors: [], geometry: .qwen38_27b)
        let act = arena.activationsRegion()
        func t32(_ a: [Float]) throws -> QWTensorValue {
            let t = try act.allocate(QWTensorShape([a.count], .f32))
            arena.copyIn(t, from: a.withUnsafeBufferPointer { Data(buffer: $0) }); return t
        }
        func t16(_ a: [Float]) throws -> QWTensorValue {
            let t = try act.allocate(QWTensorShape([a.count], .f16))
            arena.copyIn(t, from: a.map { Float16($0) }.withUnsafeBufferPointer { Data(buffer: $0) }); return t
        }
        func read32(_ t: QWTensorValue, _ n: Int) -> [Float] {
            Array(UnsafeBufferPointer(start: arena.cpuPointer(of: t).bindMemory(to: Float.self, capacity: n), count: n))
        }
        func read16(_ t: QWTensorValue, _ n: Int) -> [Float] {
            Array(UnsafeBufferPointer(start: arena.cpuPointer(of: t).bindMemory(to: Float16.self, capacity: n), count: n)).map { Float($0) }
        }
        func rel(_ a: [Float], _ b: [Float]) -> (nan: Int, rel: Double) {
            var n = 0.0, d = 0.0, nanCount = 0
            for i in 0..<a.count {
                if a[i].isNaN { nanCount += 1; continue }
                let e = Double(a[i] - b[i]); n += e * e; d += Double(b[i]) * Double(b[i])
            }
            return (nanCount, (n / max(d, 1e-30)).squareRoot())
        }
        let xIn = Self.lcg(T * C, seed: 1, scale: 0.5)
        let wIn = Self.lcg(C * 4, seed: 2, scale: 0.3)
        let bIn = Self.lcg(T * VH, seed: 3, scale: 0.5)
        let aIn = Self.lcg(T * VH, seed: 4, scale: 0.5)
        let aLog = Self.lcg(VH, seed: 5, scale: 0.2), dtBias = Self.lcg(VH, seed: 6, scale: 0.2)
        let stZero = [Float](repeating: 0, count: 3 * C)

        // conv + headnorm + prep: fp16 conv+headnorm vs f32 fused conv+prep and seq conv+prep
        let xh = try t16(xIn), wh = try t16(wIn), sth = try t16(stZero)
        let yh = try act.allocate(QWTensorShape([T * C], .f16)), sth2 = try act.allocate(QWTensorShape([3 * C], .f16))
        let xf = try t32(xIn), wf = try t32(wIn), stf = try t32(stZero), stfSeq = try t32(stZero)
        let bF = try t32(bIn), aF = try t32(aIn), aLogF = try t32(aLog), dtF = try t32(dtBias)
        let yPf = try act.allocate(QWTensorShape([T * C], .f32)), stf2 = try act.allocate(QWTensorShape([3 * C], .f32))
        let glPf = try act.allocate(QWTensorShape([T * VH], .f32)), btPf = try act.allocate(QWTensorShape([T * VH], .f32))
        let yPSeq = try act.allocate(QWTensorShape([T * C], .f32)), stfSeq2 = try act.allocate(QWTensorShape([3 * C], .f32))
        let glPSeq = try act.allocate(QWTensorShape([T * VH], .f32)), btPSeq = try act.allocate(QWTensorShape([T * VH], .f32))
        guard let (cb, enc) = Self.hal.begin() else { throw XCTSkip("no cb") }
        try ctx.convKernel.encodeRowsWithHeadnorm(enc, x: xh, w: wh, bias: nil, stIn: sth, y: yh, stOut: sth2, rows: T)
        try QWGDNF32.encodeConvPrepF32(enc, hal: Self.hal, x: xf, w: wf, stIn: stf, stOut: stf2,
                                       b: bF, a: aF, aLog: aLogF, dtBias: dtF, y: yPf, glog: glPf, beta: btPf, rows: T)
        try QWGDNF32.encodeConvPrepSeqF32(enc, hal: Self.hal, x: xf, w: wf, stIn: stfSeq, stOut: stfSeq2,
                                          b: bF, a: aF, aLog: aLogF, dtBias: dtF, y: yPSeq, glog: glPSeq, beta: btPSeq, rows: T)
        Self.hal.end(cb, enc); cb.waitUntilCompleted()
        XCTAssertNil(cb.error)
        let yPRef = read16(yh, T * C), yPNew = read32(yPf, T * C), yPSeqOut = read32(yPSeq, T * C)
        print(String(format: "GDN32 conv_prep y vs fp16: nan=%d rel=%.3e", rel(yPNew, yPRef).nan, rel(yPNew, yPRef).rel))
        print(String(format: "GDN32 conv_prep y vs seq: nan=%d rel=%.3e", rel(yPNew, yPSeqOut).nan, rel(yPNew, yPSeqOut).rel))
        print(String(format: "GDN32 prep glog vs seq: nan=%d rel=%.3e", rel(read32(glPf, T * VH), read32(glPSeq, T * VH)).nan, rel(read32(glPf, T * VH), read32(glPSeq, T * VH)).rel))
        print(String(format: "GDN32 prep beta vs seq: nan=%d rel=%.3e", rel(read32(btPf, T * VH), read32(btPSeq, T * VH)).nan, rel(read32(btPf, T * VH), read32(btPSeq, T * VH)).rel))
        XCTAssertEqual(rel(yPNew, yPRef).nan, 0)
        XCTAssertLessThan(rel(yPNew, yPRef).rel, 5e-3)
        XCTAssertLessThan(rel(yPNew, yPSeqOut).rel, 1e-6)

        // recurrence readout + fused gated RMSNorm + H_32: same state, same betas/glog, y from each path
        let stateInit = Self.lcg(VH * D * D, seed: 7, scale: 0.05)
        let zIn = Self.lcg(VH * D, seed: 8, scale: 0.5)
        let nwIn = Self.lcg(D, seed: 9, scale: 1.0)
        let sA = try t32(stateInit), sB = try t32(stateInit)
        let oA = try act.allocate(QWTensorShape([VH * D], .f32)), oB = try act.allocate(QWTensorShape([VH * D], .f32))
        let outA = try act.allocate(QWTensorShape([VH * D], .f16))
        let zHalfRef = try t16(zIn), zF32 = try t32(zIn), nwF32 = try t32(nwIn)
        let yHalfRef = try t16(Array(yPRef[0..<C])), yF32 = try t32(Array(yPNew[0..<C]))
        let gl0 = read32(glPf, T * VH), be0 = read32(btPf, T * VH)
        let glT = try t32(Array(gl0[0..<VH])), beT = try t32(Array(be0[0..<VH]))
        guard let (cb3, enc3) = Self.hal.begin() else { throw XCTSkip("no cb") }
        try ctx.recurrence.encodeRaw(enc3, y: yHalfRef, betas: beT, glog: glT, state: sA, oScratch: oA)
        enc3.memoryBarrier(scope: .buffers)
        try QWGDNSigmoidNorm.encode(enc3, hal: Self.hal, oRaw: oA, z: zHalfRef, weight: nwF32, out: outA)
        try QWGDNF32.encodeRecurrentF32(enc3, hal: Self.hal, y: yF32, betas: beT, glog: glT, state: sB, z: zF32, normWeight: nwF32, out: oB)
        Self.hal.end(cb3, enc3); cb3.waitUntilCompleted()
        XCTAssertNil(cb3.error)
        let oRef = read16(outA, VH * D), oNew = read32(oB, VH * D)
        print(String(format: "GDN32 recurrence out: nan=%d rel=%.3e", rel(oNew, oRef).nan, rel(oNew, oRef).rel))
        let sRef = read32(sA, VH * D * D), sNew = read32(sB, VH * D * D)
        print(String(format: "GDN32 recurrence state: nan=%d rel=%.3e", rel(sNew, sRef).nan, rel(sNew, sRef).rel))
    }
}
