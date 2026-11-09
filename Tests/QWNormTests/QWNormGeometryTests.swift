// QWNormGeometryTests.swift — the norm family must serve any hidden the
// rotation covers, not only the 27B's 5120.
import XCTest
import Metal
@testable import QWNorm
import QWCore
import QWHAL
import QWAlloc
import QWQuant

final class QWNormGeometryTests: XCTestCase {
    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    /// A standard geometry through the same decoder a real checkpoint takes.
    static func standard(hidden: Int) throws -> QWModelGeometry {
        let config: [String: Any] = [
            "model_type": "qwen3",
            "hidden_size": hidden,
            "num_hidden_layers": 1,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "intermediate_size": 256,
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "eos_token_id": 2,
            "vocab_size": 64,
        ]
        let data = try JSONSerialization.data(withJSONObject: config)
        return QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: data))
    }

    static func f16(_ bits: UInt16) -> Float { QWQuant.f16tof32Bits(bits) }

    /// The 32-point normalized WHT, the kernel's butterfly: strides
    /// 1, 2, 4, 8, 16 then one 1/sqrt(32) scale.
    static func fwht32(_ v: inout [Float]) {
        let inv = 1.0 / Float(32).squareRoot()
        var base = 0
        while base < v.count {
            var s = 1
            while s < 32 {
                var i = 0
                while i < 32 {
                    for j in 0..<s {
                        let a = v[base + i + j], b = v[base + i + j + s]
                        v[base + i + j] = a + b
                        v[base + i + j + s] = a - b
                    }
                    i += 2 * s
                }
                s *= 2
            }
            for i in 0..<32 { v[base + i] *= inv }
            base += 32
        }
    }

    struct Case {
        let rows: Int, hidden: Int
        let x: [Float], gamma: [Float], residual: [Float]
    }

    /// Deterministic, non-degenerate inputs: no all-zero row, no constant.
    static func makeCase(rows: Int, hidden: Int) -> Case {
        var x = [Float](repeating: 0, count: rows * hidden)
        var g = [Float](repeating: 0, count: hidden)
        var r = [Float](repeating: 0, count: rows * hidden)
        for i in 0..<(rows * hidden) {
            x[i] = Float((i % 17) - 8) * 0.25 + 0.125
            r[i] = Float((i % 11) - 5) * 0.5 + 0.0625
        }
        for i in 0..<hidden {
            g[i] = Float((i % 13) - 6) * 0.0625
        }
        return Case(rows: rows, hidden: hidden, x: x, gamma: g, residual: r)
    }

    /// The reference: the residual update in f16, then the RMSNorm's
    /// `(1 + gamma)` law, then the rotation into the H_32 basis.
    static func reference(_ c: Case, eps: Float) -> (y: [Float], residual: [Float]) {
        var res = [Float](repeating: 0, count: c.rows * c.hidden)
        for i in 0..<res.count {
            res[i] = QWQuant.f16tof32Bits(QWQuant.f32tof16Bits(c.residual[i] + c.x[i]))
        }
        var y = [Float](repeating: 0, count: res.count)
        for row in 0..<c.rows {
            let base = row * c.hidden
            var sumSq: Float = 0
            for j in 0..<c.hidden { sumSq += res[base + j] * res[base + j] }
            let rstd = 1.0 / (sumSq / Float(c.hidden) + eps).squareRoot()
            var f = [Float](repeating: 0, count: c.hidden)
            for j in 0..<c.hidden {
                f[j] = res[base + j] * rstd * (c.gamma[j] + 1.0)
            }
            fwht32(&f)
            for j in 0..<c.hidden { y[base + j] = f[j] }
        }
        return (y, res)
    }

    /// The f16 spacing at `v` — the house 1-ulp adjacency law.
    static func f16Spacing(_ v: Float) -> Double {
        let e = Int((v.bitPattern >> 23) & 0xFF) - 127 - 10
        return e >= 0 ? Double(1 << e) : 1.0 / Double(1 << -e)
    }

    /// The house parity law (spec 06): a violation is beyond `absTol`
    /// **and** not 1-ulp f16 adjacent. The floor matters: the kernel and
    /// the reference sum the squares differently.
    static func parityViolations(_ got: [Float], _ ref: [Float], absTol: Float = 1e-3)
        -> (violations: Int, worstAbs: Double, worstRatio: Double) {
        var viol = 0, worstAbs = 0.0, worstRatio = 0.0
        for i in 0..<got.count {
            let d = abs(Double(got[i]) - Double(ref[i]))
            worstAbs = max(worstAbs, d)
            guard d > Double(absTol) else { continue }
            let sp = min(f16Spacing(got[i]), f16Spacing(ref[i]))
            let r = d / sp
            worstRatio = max(worstRatio, r)
            if r > 1.0001 { viol += 1 }
        }
        return (viol, worstAbs, worstRatio)
    }

    /// Runs a norm variant and returns the output and the updated
    /// residual, both as f32 values read back from the GPU.
    func runNorm(_ variant: QWNormVariant, _ c: Case, geometry: QWModelGeometry)
        throws -> (y: [Float], residual: [Float], scales: [Float]) {
        let arena = try QWArena(hal: Self.hal, weights: 0,
                                activations: c.rows * c.hidden * 2 * 4 + c.hidden * 2 + 1024,
                                kv: 0, gdnStates: 0)
        let a = arena.activationsRegion()
        let x = try a.allocate(QWTensorShape([c.rows * c.hidden], .f16))
        let g = try a.allocate(QWTensorShape([c.hidden], .f16))
        let y = try a.allocate(QWTensorShape([c.rows * c.hidden], .f16))
        let res = try a.allocate(QWTensorShape([c.rows * c.hidden], .f16))
        let scales = try a.allocate(QWTensorShape([c.rows], .f32))
        func pack(_ v: [Float]) -> Data {
            var bits = [UInt16](repeating: 0, count: v.count)
            for i in 0..<v.count { bits[i] = QWQuant.f32tof16Bits(v[i]) }
            return Data(bytes: bits, count: bits.count * 2)
        }
        arena.copyIn(x, from: pack(c.x))
        arena.copyIn(g, from: pack(c.gamma))
        arena.copyIn(res, from: pack(c.residual))

        guard let (cb, enc) = Self.hal.begin() else { throw NSError(domain: "hal", code: 1) }
        do {
            try QWNorm.encode(enc, hal: Self.hal, variant: variant,
                              x: x, gamma: g, out: y, residual: res,
                              rowScalesOut: scales, geometry: geometry)
        } catch {
            enc.endEncoding()
            throw error
        }
        Self.hal.end(cb, enc)
        cb.waitUntilCompleted()

        func read(_ t: QWTensorValue) -> [Float] {
            let p = arena.cpuPointer(of: t).assumingMemoryBound(to: UInt16.self)
            return (0..<(c.rows * c.hidden)).map { Self.f16(p[$0]) }
        }
        let sp = arena.cpuPointer(of: scales).assumingMemoryBound(to: Float.self)
        return (read(y), read(res), (0..<c.rows).map { sp[$0] })
    }

    /// The fused residual norm must WRITE at a hidden the 27B does not
    /// have — it used to return without writing anywhere else, which left
    /// the residual untouched and the normed output stale.
    func testTheFusedResidualNormServesANon27BHidden() throws {
        for hidden in [1024, 2048, 4096] {
            let geometry = try Self.standard(hidden: hidden)
            let c = Self.makeCase(rows: 2, hidden: hidden)
            let got = try runNorm(.rmsnormResidual, c, geometry: geometry)
            let ref = Self.reference(c, eps: geometry.rmsEps)

            // 1. The residual update is the bit-exact f16 add.
            var worstRes = 0.0
            for i in 0..<c.rows * hidden {
                worstRes = max(worstRes, abs(Double(got.residual[i]) - Double(ref.residual[i])))
            }
            XCTAssertEqual(worstRes, 0, "hidden \(hidden): the residual update must be the bit-exact f16 add")

            // 2. The normed output is the rotated (1 + gamma) RMSNorm.
            let p = Self.parityViolations(got.y, ref.y)
            XCTAssertEqual(p.violations, 0,
                "hidden \(hidden): \(p.violations) norm outputs beyond tol (worst \(p.worstAbs), ratio \(p.worstRatio))")
        }
    }

    /// The prologue-scale sibling carries the same general path, so it
    /// must serve a non-27B hidden too: the residual is updated, the row
    /// scale is written, and the output is the rotation WITHOUT the rstd.
    func testThePrologueScaleResidualNormServesANon27BHidden() throws {
        for hidden in [1024, 4096] {
            let geometry = try Self.standard(hidden: hidden)
            let c = Self.makeCase(rows: 2, hidden: hidden)
            let got = try runNorm(.rmsnormResidualPrologueScale, c, geometry: geometry)
            let ref = Self.reference(c, eps: geometry.rmsEps)

            var worstRes = 0.0
            for i in 0..<c.rows * hidden {
                worstRes = max(worstRes, abs(Double(got.residual[i]) - Double(ref.residual[i])))
            }
            XCTAssertEqual(worstRes, 0, "hidden \(hidden): the residual update must be the bit-exact f16 add")

            // The scale is the row's rstd: the normed row's L2 norm.
            for row in 0..<c.rows {
                let base = row * hidden
                var sumSq: Float = 0
                for j in 0..<hidden { sumSq += ref.residual[base + j] * ref.residual[base + j] }
                let rstd = 1.0 / (sumSq / Float(hidden) + geometry.rmsEps).squareRoot()
                XCTAssertEqual(Double(got.scales[row]), Double(rstd), accuracy: 1e-6,
                    "hidden \(hidden): the row scale must be the row's rstd")
            }

            // The output is the rotation of the unscaled normed row.
            var unscaled = [Float](repeating: 0, count: c.rows * hidden)
            for row in 0..<c.rows {
                let base = row * hidden
                var f = [Float](repeating: 0, count: hidden)
                for j in 0..<hidden { f[j] = ref.residual[base + j] * (c.gamma[j] + 1.0) }
                Self.fwht32(&f)
                for j in 0..<hidden { unscaled[base + j] = f[j] }
            }
            let p = Self.parityViolations(got.y, unscaled)
            XCTAssertEqual(p.violations, 0,
                "hidden \(hidden): \(p.violations) prologue-scale outputs beyond tol (worst \(p.worstAbs))")
        }
    }

    /// A hidden the rotation cannot serve is REFUSED, by name — the
    /// family's law is a multiple of 256: uint4 rows, and a collective
    /// shuffle that needs all 32 lanes to iterate alike.
    func testAHiddenTheRotationCannotServeIsRefused() throws {
        for hidden in [64, 320, 4095] {
            let geometry = try Self.standard(hidden: hidden)
            let c = Self.makeCase(rows: 1, hidden: hidden)
            XCTAssertThrowsError(try runNorm(.rmsnormResidual, c, geometry: geometry),
                "hidden \(hidden) must be refused") { e in
                XCTAssertTrue("\(e)".contains("256"), "the refusal names the law: \(e)")
            }
        }
    }

    /// The 27B's own shape must keep working — the register-resident fast
    /// path is what the golden gate pins.
    func testTheFusedResidualNormStillServesThe27BHidden() throws {
        let geometry = QWModelGeometry.qwen38_27b
        let c = Self.makeCase(rows: 1, hidden: geometry.hidden)
        let got = try runNorm(.rmsnormResidual, c, geometry: geometry)
        let ref = Self.reference(c, eps: geometry.rmsEps)
        var worstRes = 0.0
        for i in 0..<c.rows * geometry.hidden {
            worstRes = max(worstRes, abs(Double(got.residual[i]) - Double(ref.residual[i])))
        }
        XCTAssertEqual(worstRes, 0, "the 27B's residual update must be the bit-exact f16 add")
        let p = Self.parityViolations(got.y, ref.y)
        XCTAssertEqual(p.violations, 0,
            "the 27B's norm output must stay within the house law (worst \(p.worstAbs))")
    }
}
