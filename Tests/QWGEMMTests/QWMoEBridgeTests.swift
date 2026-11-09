// QWMoEBridgeTests.swift — the standard MoE path's f32 bridge: the f16
// normed hidden becomes the two f32 copies the MoE chain reads — the
// rotated one (the expert GEMMs' input) and the un-rotated one (the
// router's input, whose weight is the packer's raw plane).
import XCTest
import Metal
import QWAlloc
import QWCore
import QWGEMM
import QWHAL
import QWQuant

final class QWMoEBridgeTests: XCTestCase {
    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    /// The PACKER's rotation — not a third copy of it. `fwht32OrthoBlock` is
    /// the function the packer itself rotates the weights with, so the test
    /// pins the packer's basis and not a copy of it that could drift.
    static func fwht32(_ v: inout [Float]) {
        v.withUnsafeMutableBufferPointer { buf in
            for b in stride(from: 0, to: buf.count, by: 32) {
                QWQuant.fwht32OrthoBlock(buf.baseAddress! + b)
            }
        }
    }

    /// Deterministic, non-degenerate values: no all-zero row, no constant.
    static func inputs(rows: Int, hidden: Int) -> [Float] {
        var v = [Float](repeating: 0, count: rows * hidden)
        for i in 0..<(rows * hidden) {
            v[i] = Float((i % 23) - 11) * 0.125 + 0.0625
        }
        return v
    }

    /// The bridge's two outputs, read back from the GPU.
    func runBridge(rows: Int, hidden: Int, x: [Float],
                   hal: QWHal = QWMoEBridgeTests.hal) throws -> (rot: [Float], plain: [Float]) {
        // x is f16, the two outputs are f32: 10 bytes per element, plus slack
        // for the arena's 256-byte alignment on each of the three carves.
        let arena = try QWArena(hal: hal, weights: 0,
                                activations: rows * hidden * 10 + 1024,
                                kv: 0, gdnStates: 0)
        let a = arena.activationsRegion()
        let xt = try a.allocate(QWTensorShape([rows * hidden], .f16))
        let rot = try a.allocate(QWTensorShape([rows * hidden], .f32))
        let plain = try a.allocate(QWTensorShape([rows * hidden], .f32))
        var bits = [UInt16](repeating: 0, count: x.count)
        for i in 0..<x.count { bits[i] = QWQuant.f32tof16Bits(x[i]) }
        arena.copyIn(xt, from: Data(bytes: bits, count: bits.count * 2))

        guard let (cb, enc) = hal.begin() else { throw NSError(domain: "hal", code: 1) }
        do {
            try QWGemm.encodeMoEBridgeF16ToF32(enc, hal: hal, x: xt,
                                                xRot32: rot, xPlain32: plain,
                                                rows: rows, hidden: hidden)
        } catch {
            enc.endEncoding()
            throw error
        }
        hal.end(cb, enc)
        cb.waitUntilCompleted()

        let rp = arena.cpuPointer(of: rot).assumingMemoryBound(to: Float.self)
        let pp = arena.cpuPointer(of: plain).assumingMemoryBound(to: Float.self)
        return ((0..<(rows * hidden)).map { rp[$0] }, (0..<(rows * hidden)).map { pp[$0] })
    }

    /// The bridge widens the hidden into the **rotated** copy the expert
    /// GEMMs read, and un-rotates it into the **plain** copy the router
    /// reads — the WHT is its own inverse.
    func testTheBridgeWidensAndUnRotates() throws {
        for (rows, hidden) in [(1, 256), (3, 256), (2, 512), (1, 1024), (4, 2048)] {
            let x = Self.inputs(rows: rows, hidden: hidden)
            let got = try runBridge(rows: rows, hidden: hidden, x: x)

            // The rotated copy is the plain widening of the f16 input.
            var worstRot = 0.0
            for i in 0..<(rows * hidden) {
                let want = QWQuant.f16tof32Bits(QWQuant.f32tof16Bits(x[i]))
                worstRot = max(worstRot, abs(Double(got.rot[i]) - Double(want)))
            }
            XCTAssertEqual(worstRot, 0, "rows \(rows) hidden \(hidden): the rotated copy is the widening")

            // The plain copy is the 32-point WHT of each row.
            for row in 0..<rows {
                var f = [Float](repeating: 0, count: hidden)
                for j in 0..<hidden {
                    f[j] = QWQuant.f16tof32Bits(QWQuant.f32tof16Bits(x[row * hidden + j]))
                }
                Self.fwht32(&f)
                for j in 0..<hidden {
                    let d = abs(Double(got.plain[row * hidden + j]) - Double(f[j]))
                    XCTAssertLessThanOrEqual(d, 1e-5,
                        "rows \(rows) hidden \(hidden): the plain copy at \(row),\(j) is \(got.plain[row * hidden + j]), want \(f[j])")
                }
            }
        }
    }

    /// In the STANDARD basis the hidden that reaches the bridge is already
    /// the plain one, so it must come out the other side unchanged. This
    /// guards the HELPER's gate; the bridge's own branch is a short-circuit.
    func testTheStandardBasisPassesTheHiddenThrough() throws {
        let standardHal = try QWHal(geometry: .qwen38_27b, weightBasis: .standard)
        for (rows, hidden) in [(1, 256), (3, 256), (2, 512)] {
            let x = Self.inputs(rows: rows, hidden: hidden)
            let got = try runBridge(rows: rows, hidden: hidden, x: x, hal: standardHal)

            // Both copies are the plain widening of the f16 input.
            var worst = 0.0
            for i in 0..<(rows * hidden) {
                let want = QWQuant.f16tof32Bits(QWQuant.f32tof16Bits(x[i]))
                worst = max(worst, abs(Double(got.rot[i]) - Double(want)))
                worst = max(worst, abs(Double(got.plain[i]) - Double(want)))
            }
            XCTAssertEqual(worst, 0,
                           "rows \(rows) hidden \(hidden): the standard basis must not un-rotate")
        }
    }

    /// The bridge's butterfly needs its 4-lane groups fully populated and
    /// all 32 lanes iterating alike, so a hidden the norm family cannot
    /// serve is refused here too.
    func testAHiddenTheButterflyCannotServeIsRefused() throws {
        let arena = try QWArena(hal: Self.hal, weights: 0, activations: 8192, kv: 0, gdnStates: 0)
        let a = arena.activationsRegion()
        let x = try a.allocate(QWTensorShape([512], .f16))
        let rot = try a.allocate(QWTensorShape([512], .f32))
        let plain = try a.allocate(QWTensorShape([512], .f32))
        guard let (cb, enc) = Self.hal.begin() else { return }
        defer { enc.endEncoding(); _ = cb }
        // 128 and 384 are multiples of 128 but NOT of 256: they pin the
        // law's boundary, so a weakened law (`% 128`) would fail here.
        for hidden in [64, 128, 320, 384, 511] {
            XCTAssertThrowsError(try QWGemm.encodeMoEBridgeF16ToF32(
                enc, hal: Self.hal, x: x, xRot32: rot, xPlain32: plain, rows: 1, hidden: hidden),
                "hidden \(hidden) must be refused") { e in
                XCTAssertTrue("\(e)".contains("256"), "the refusal names the law: \(e)")
            }
        }
    }
}
