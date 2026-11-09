import XCTest
import Metal
import Foundation
@testable import QWGDN
import QWCore
import QWHAL
import QWAlloc
import QWQuant

/// C11 conv-kernel gate: `qwg_gdn_conv` vs torch fixtures
/// (gen_gdn_conv_fixtures.py — golden = qw38_oracle GatedDeltaNet: conv1d
/// k=4 depthwise + bias + SiLU, state := last 3 RAW rows). Parity law: 1-ulp
final class QWGDNConvTests: XCTestCase {

    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    static let spikeOut = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Scripts/spike/out")

    static func load(_ name: String) throws -> Data {
        let p = spikeOut.appendingPathComponent(name)
        guard let d = FileManager.default.contents(atPath: p.path) else {
            throw NSError(domain: "fixture", code: 2, userInfo: [NSLocalizedDescriptionKey: "missing \(name)"])
        }
        return d
    }
    static func u16s(_ d: Data) -> [UInt16] {
        d.withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }
    }
    static func buf(_ d: Data) -> MTLBuffer {
        let b = hal.device.makeBuffer(length: max(16, d.count), options: .storageModeShared)!
        d.withUnsafeBytes { _ = memcpy(b.contents(), $0.baseAddress!, $0.count) }
        return b
    }
    static func f16(_ bits: UInt16) -> Float { QWQuant.f16tof32Bits(bits) }
    static func readF(_ b: [UInt16]) -> [Float] { b.map { f16($0) } }

    static func tv(_ b: MTLBuffer, _ n: Int) -> QWTensorValue {
        QWTensorValue(region: QWMRegion(buffer: b, base: 0, length: max(16, n * 2)),
                      offset: 0, shape: QWTensorShape([n], .f16))
    }

    /// 1-ulp adjacency (C06 law) — local copy (test targets are independent).
    static func parityViolations(_ got: [Float], _ ref: [Float], absTol: Float = 1e-3)
        -> (violations: Int, worstAbs: Double, worstIdx: Int) {
        var viol = 0, worst = 0.0, worstIdx = -1
        for i in 0..<got.count {
            let d = abs(Double(got[i]) - Double(ref[i]))
            if d > worst { worst = d; worstIdx = i }
            guard got[i].isFinite && ref[i].isFinite else { viol += 1; continue }
            let spacing = max(Double(abs(got[i])), Double(abs(ref[i]))) * Double.ulpOfOne
            if d > max(Double(absTol), spacing) { viol += 1 }
        }
        return (viol, worst, worstIdx)
    }

    static let C = 10240

    func testMultiBandInplaceStateMatchesDisjointState() throws {
        let hal = Self.hal, conv = try QWGDNConv(hal: Self.hal, geometry: .qwen38_27b)
        let T = 65
        let x = Self.buf(try Self.load("gdn_conv_x_t65_pfx.bin"))
        let w = Self.buf(try Self.load("gdn_conv_w.bin"))
        let bias = Self.buf(try Self.load("gdn_conv_bias.bin"))
        let state = try Self.load("gdn_conv_state_t65_pfx.bin")
        let source = Self.buf(state), inplace = Self.buf(state)
        let destination = try XCTUnwrap(hal.device.makeBuffer(length: state.count, options: .storageModeShared))
        let expected = try XCTUnwrap(hal.device.makeBuffer(length: T * Self.C * 2, options: .storageModeShared))
        let actual = try XCTUnwrap(hal.device.makeBuffer(length: expected.length, options: .storageModeShared))
        for fused in [false, true] {
            for iteration in 0..<64 {
                state.withUnsafeBytes { inplace.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
                let pair = try XCTUnwrap(hal.begin())
                func encode(stateOut: MTLBuffer, stateIn: MTLBuffer, output: MTLBuffer) throws {
                    if fused {
                        try conv.encodePrefillWithHeadnorm(pair.1, x: Self.tv(x, T * Self.C),
                            w: Self.tv(w, Self.C * 4), bias: Self.tv(bias, Self.C),
                            stIn: Self.tv(stateIn, 3 * Self.C), y: Self.tv(output, T * Self.C),
                            stOut: Self.tv(stateOut, 3 * Self.C), T: T)
                    } else {
                        try conv.encode(pair.1, x: Self.tv(x, T * Self.C),
                            w: Self.tv(w, Self.C * 4), bias: Self.tv(bias, Self.C),
                            stIn: Self.tv(stateIn, 3 * Self.C), y: Self.tv(output, T * Self.C),
                            stOut: Self.tv(stateOut, 3 * Self.C), T: T)
                    }
                }
                try encode(stateOut: destination, stateIn: source, output: expected)
                try encode(stateOut: inplace, stateIn: inplace, output: actual)
                pair.1.endEncoding()
                pair.0.commit(); pair.0.waitUntilCompleted()
                XCTAssertEqual(pair.0.status, .completed)
                let yEqual = memcmp(expected.contents(), actual.contents(), expected.length) == 0
                XCTAssertTrue(yEqual, "in-place output differs: fused=\(fused), iteration=\(iteration)")
                XCTAssertEqual(memcmp(destination.contents(), inplace.contents(), state.count), 0)
                if !yEqual { return }
            }
        }
    }

    /// One conv launch → (y bits [T·C], stateOut bits [3C]).
    static func runConv(x: Data, w: Data, bias: Data, state: Data, T: Int) throws -> (y: [UInt16], st: [UInt16]) {
        let enc = try QWGDNConv(hal: Self.hal, geometry: .qwen38_27b)
        let yb = hal.device.makeBuffer(length: T * C * 2, options: .storageModeShared)!
        let stb = hal.device.makeBuffer(length: 3 * C * 2, options: .storageModeShared)!
        guard let (cb, ce) = hal.begin() else { throw NSError(domain: "hal", code: 1) }
        try enc.encode(ce, x: tv(buf(x), T * C), w: tv(buf(w), C * 4),
                       bias: tv(buf(bias), C), stIn: tv(buf(state), 3 * C),
                       y: tv(yb, T * C), stOut: tv(stb, 3 * C), T: T)
        ce.endEncoding()                                          // end-before-wait
        cb.commit(); cb.waitUntilCompleted()
        let p1 = yb.contents().assumingMemoryBound(to: UInt16.self)
        let p2 = stb.contents().assumingMemoryBound(to: UInt16.self)
        return ((0..<(T * C)).map { p1[$0] }, (0..<(3 * C)).map { p2[$0] })
    }

    /// One conv launch with an IN-PLACE state buffer (C12's fused-prologue
    /// aliasing contract): stIn == stOut == same MTLBuffer.
    static func runConvInplace(x: Data, w: Data, bias: Data, stBuf: MTLBuffer, T: Int) throws -> [UInt16] {
        let enc = try QWGDNConv(hal: Self.hal, geometry: .qwen38_27b)
        let yb = hal.device.makeBuffer(length: T * C * 2, options: .storageModeShared)!
        guard let (cb, ce) = hal.begin() else { throw NSError(domain: "hal", code: 1) }
        try enc.encode(ce, x: tv(buf(x), T * C), w: tv(buf(w), C * 4),
                       bias: tv(buf(bias), C), stIn: tv(stBuf, 3 * C),
                       y: tv(yb, T * C), stOut: tv(stBuf, 3 * C), T: T)
        ce.endEncoding()
        cb.commit(); cb.waitUntilCompleted()
        let p1 = yb.contents().assumingMemoryBound(to: UInt16.self)
        return (0..<(T * C)).map { p1[$0] }
    }

    func testConvParity() throws {
        let w = try Self.load("gdn_conv_w.bin"), bias = try Self.load("gdn_conv_bias.bin")
        for T in [1, 2, 3, 4, 5, 6, 7, 8, 63, 64, 65, 66, 127, 128, 129, 130] {
            let x = try Self.load("gdn_conv_x_t\(T).bin")
            let st = try Self.load("gdn_conv_state_t\(T).bin")
            let yRef = Self.readF(Self.u16s(try Self.load("gdn_conv_y_t\(T).bin")))
            let stRef = Self.u16s(try Self.load("gdn_conv_state2_t\(T).bin"))
            let (y, st2) = try Self.runConv(x: x, w: w, bias: bias, state: st, T: T)
            let got = Self.readF(y)
            XCTAssertEqual(got.filter { !$0.isFinite }.count, 0, "T=\(T): non-finite y")
            let v = Self.parityViolations(got, yRef)
            XCTAssertEqual(v.violations, 0, "T=\(T): y parity (worst \(v.worstAbs) @\(v.worstIdx))")
            XCTAssertEqual(st2, stRef, "T=\(T): rotated state must be BYTE equal (row copy)")
        }
    }

    func testConvPrefixState() throws {
        let w = try Self.load("gdn_conv_w.bin"), bias = try Self.load("gdn_conv_bias.bin")
        for T in [1, 2, 65] {
            let x = try Self.load("gdn_conv_x_t\(T)_pfx.bin")
            let st = try Self.load("gdn_conv_state_t\(T)_pfx.bin")
            let yRef = Self.readF(Self.u16s(try Self.load("gdn_conv_y_t\(T)_pfx.bin")))
            let stRef = Self.u16s(try Self.load("gdn_conv_state2_t\(T)_pfx.bin"))
            let (y, st2) = try Self.runConv(x: x, w: w, bias: bias, state: st, T: T)
            let v = Self.parityViolations(Self.readF(y), yRef)
            XCTAssertEqual(v.violations, 0, "T=\(T) pfx: y parity (worst \(v.worstAbs) @\(v.worstIdx))")
            XCTAssertEqual(st2, stRef, "T=\(T) pfx: state byte equal")
        }
    }

    func testStreamedEqualsPrefill() throws {
        // THE spec parity law: decode token-by-token (T=1 × N, state rotated
        // in place) must reproduce the prefill y rows AND the final state.
        let w = try Self.load("gdn_conv_w.bin"), bias = try Self.load("gdn_conv_bias.bin")
        let N = 130
        let xAll = Self.u16s(try Self.load("gdn_conv_x_t\(N).bin"))
        let yRef = Self.u16s(try Self.load("gdn_conv_y_t\(N).bin"))
        let stRef = Self.u16s(try Self.load("gdn_conv_state2_t\(N).bin"))

        let stBuf = Self.hal.device.makeBuffer(length: 3 * Self.C * 2, options: .storageModeShared)!
        var viol = 0, worst = 0.0
        for t in 0..<N {
            let row = Array(xAll[(t * Self.C)..<(t * Self.C + Self.C)])
            let rowBits = Data(bytes: row, count: row.count * 2)
            let y = try Self.runConvInplace(x: rowBits, w: w, bias: bias, stBuf: stBuf, T: 1)
            // per-step row vs the prefill fixture row
            for c in 0..<Self.C {
                let g = Double(Self.f16(y[c])), r = Double(Self.f16(yRef[t * Self.C + c]))
                let d = abs(g - r)
                if d > worst { worst = d }
                let sp = max(abs(g), abs(r)) * Double.ulpOfOne
                if !(g.isFinite && r.isFinite) || d > max(1e-3, sp) { viol += 1 }
            }
        }
        XCTAssertEqual(viol, 0, "streamed≠prefill y (worst \(worst))")
        let p = stBuf.contents().assumingMemoryBound(to: UInt16.self)
        let final = (0..<(3 * Self.C)).map { p[$0] }
        XCTAssertEqual(final, stRef, "streamed final state ≠ prefill state2")
    }

    func testBiasNullSlab() throws {
        // spec-11 said "no bias"; the encoder must bind a zero slab when nil.
        let w = try Self.load("gdn_conv_w.bin")   // bias fixture NOT loaded: this test's point is bias: nil
        let T = 4
        let x = try Self.load("gdn_conv_x_t\(T).bin")
        let st = try Self.load("gdn_conv_state_t\(T).bin")
        let enc = try QWGDNConv(hal: Self.hal, geometry: .qwen38_27b)
        let yb = Self.hal.device.makeBuffer(length: T * Self.C * 2, options: .storageModeShared)!
        let stb = Self.hal.device.makeBuffer(length: 3 * Self.C * 2, options: .storageModeShared)!
        guard let (cb, ce) = Self.hal.begin() else { throw NSError(domain: "hal", code: 1) }
        try enc.encode(ce, x: Self.tv(Self.buf(x), T * Self.C), w: Self.tv(Self.buf(w), Self.C * 4),
                       bias: nil, stIn: Self.tv(Self.buf(st), 3 * Self.C),
                       y: Self.tv(yb, T * Self.C), stOut: Self.tv(stb, 3 * Self.C), T: T)
        ce.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let p = yb.contents().assumingMemoryBound(to: UInt16.self)
        let got = Self.readF((0..<(T * Self.C)).map { p[$0] })
        XCTAssertEqual(got.filter { !$0.isFinite }.count, 0, "bias-nil path must stay finite")
        // sanity: with bias removed the values differ from the biased fixture
        let yRef = Self.readF(Self.u16s(try Self.load("gdn_conv_y_t\(T).bin")))
        XCTAssertNotEqual(got, yRef, "nil-bias path must not silently equal biased output")
    }

    func testShapeGuards() throws {
        let enc = try QWGDNConv(hal: Self.hal, geometry: .qwen38_27b)
        let x = Self.hal.device.makeBuffer(length: 2048, options: .storageModeShared)!
        let w = Self.hal.device.makeBuffer(length: 16, options: .storageModeShared)!
        guard let (cb, ce) = Self.hal.begin() else { throw NSError(domain: "hal", code: 1) }
        XCTAssertThrowsError(try enc.encode(ce, x: Self.tv(x, 2048), w: Self.tv(w, 16),
                                            bias: nil, stIn: Self.tv(x, 2048),
                                            y: Self.tv(x, 2048), stOut: Self.tv(x, 2048), T: 5))
        ce.endEncoding(); cb.commit()          // discarded buffer: end anyway
    }
}
