import XCTest
import Metal
import Foundation
import QWCore
import QWHAL
import QWAlloc
import QWGDN

// Bench + parity: QWGDNDecode.encode (gated norm, z/w/o bound) vs encodeRaw (norm tail bound as nil).
// Gate: QW_BENCH_GDN=1. Bit-identity: oScratch and state. Timing: 50 dispatches per buffer, GPU timestamps, median of 20.
final class FNGDNRawBenchTests: XCTestCase {
    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    static func lcg(_ n: Int, seed: UInt64, scale: Float, offset: Float = 0) -> [Float] {
        var s = seed
        return (0..<n).map { _ in
            s = s &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int64(bitPattern: s >> 33) % 2001 - 1000) / 1000 * scale + offset
        }
    }

    func testRawEqualsNormedAndTiming() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["QW_BENCH_GDN"] == "1", "bench gated by QW_BENCH_GDN=1")
        let rec = try QWGDNDecode(hal: Self.hal)
        let VH = QWGDNDecode.vHeads, D = QWGDNDecode.headDim
        let S = VH * D * D
        let arena = try QWArena(hal: Self.hal, weights: 4096, activations: 256 << 20, kv: 0, gdnStates: 0)
        let act = arena.activationsRegion()
        func up32(_ a: [Float]) throws -> QWTensorValue {
            let t = try act.allocate(QWTensorShape([a.count], .f32))
            arena.copyIn(t, from: a.withUnsafeBufferPointer { Data(buffer: $0) }); return t
        }
        func up16(_ a: [Float16]) throws -> QWTensorValue {
            let t = try act.allocate(QWTensorShape([a.count], .f16))
            arena.copyIn(t, from: a.withUnsafeBufferPointer { Data(buffer: $0) }); return t
        }
        let y = try up16(Self.lcg(10240, seed: 1, scale: 0.5).map { Float16($0) })
        let betas = try up32(Self.lcg(VH, seed: 2, scale: 0.5, offset: 0.5))
        let glog = try up32(Self.lcg(VH, seed: 3, scale: 0.05, offset: -0.05))
        let stateInit = Self.lcg(S, seed: 4, scale: 0.01)
        let stateA = try up32(stateInit), stateB = try up32(stateInit)
        let z = try up16(Self.lcg(VH * D, seed: 5, scale: 1).map { Float16($0) })
        let w = try up16(Self.lcg(D, seed: 6, scale: 1).map { Float16($0) })
        let oA = try act.allocate(QWTensorShape([VH * D], .f32)), oB = try act.allocate(QWTensorShape([VH * D], .f32))
        let oOut = try act.allocate(QWTensorShape([VH * D], .f16))

        // parity: one step each from identical state
        guard let (cb, enc) = Self.hal.begin() else { return XCTFail("no command buffer") }
        try rec.encode(enc, y: y, betas: betas, glog: glog, state: stateA, z: z, w: w, o: oOut, oScratch: oA)
        try rec.encodeRaw(enc, y: y, betas: betas, glog: glog, state: stateB, oScratch: oB)
        Self.hal.end(cb, enc); cb.waitUntilCompleted()
        XCTAssertNil(cb.error)
        func bits(_ t: QWTensorValue, _ n: Int) -> [UInt32] {
            Array(UnsafeBufferPointer(start: arena.cpuPointer(of: t).bindMemory(to: UInt32.self, capacity: n), count: n))
        }
        let oMis = zip(bits(oA, VH * D), bits(oB, VH * D)).filter { $0 != $1 }.count
        let sMis = zip(bits(stateA, S), bits(stateB, S)).filter { $0 != $1 }.count
        print("GDNRAW parity: oScratch mismatches=\(oMis)/\(VH * D), state mismatches=\(sMis)/\(S)")

        // timing
        func timeIt(_ body: (any MTLComputeCommandEncoder) throws -> Void) throws -> Double {
            var ts = [Double]()
            for _ in 0..<20 {
                guard let (cb, enc) = Self.hal.begin() else { throw NSError(domain: "bench", code: 1) }
                for _ in 0..<50 { try body(enc) }
                Self.hal.end(cb, enc); cb.waitUntilCompleted()
                ts.append((cb.gpuEndTime - cb.gpuStartTime) * 1e6 / 50.0)
            }
            ts.sort(); return ts[ts.count / 2]
        }
        let normed = try timeIt { e in try rec.encode(e, y: y, betas: betas, glog: glog, state: stateA, z: z, w: w, o: oOut, oScratch: oA) }
        let raw = try timeIt { e in try rec.encodeRaw(e, y: y, betas: betas, glog: glog, state: stateB, oScratch: oB) }
        let normed2 = try timeIt { e in try rec.encode(e, y: y, betas: betas, glog: glog, state: stateA, z: z, w: w, o: oOut, oScratch: oA) }
        let raw2 = try timeIt { e in try rec.encodeRaw(e, y: y, betas: betas, glog: glog, state: stateB, oScratch: oB) }
        print("GDNRAW timing T=1 per step: encode(normed)=\(normed) us, encodeRaw=\(raw) us | repeat: \(normed2) us, \(raw2) us")
        XCTAssertEqual(oMis, 0)
        XCTAssertEqual(sMis, 0)
    }
}
