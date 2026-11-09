import XCTest
import Metal
@testable import QWNorm
import QWCore
import QWHAL
import QWAlloc
import QWQuant

/// C06 gate (specs/06-qwnorm.md): parity vs torch-math fp32 fixtures for all
/// three variants (maxabs ≤ 1e-3 f16 out; row-scales ≤ 5e-4 rel), residual
/// update bit-exact, head-norm rejects with a clear error. Authoritative BW
final class QWNormTests: XCTestCase {

    static let H = QWModelGeometry.qwen38_27b.hidden
    static let fixtureRoot: URL = {
        var d = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        for _ in 0..<8 {
            let c = d.appendingPathComponent("Scripts/spike/out")
            if FileManager.default.fileExists(atPath: c.path) { return c }
            d = d.deletingLastPathComponent()
        }
        fatalError("run Scripts/oracle/gen_norm_fixtures.py")
    }()
    static func fixture(_ n: String) -> Data {
        try! Data(contentsOf: fixtureRoot.appendingPathComponent(n))
    }
    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    struct IO {
        let rows: Int
        let arena: QWArena
        let x, g, y, res, scales: QWTensorValue
    }

    /// f16 bits → f32 exact (delegates to the C04 converter — the same math
    /// the packer reader uses, so the test proves engine↔packer agreement).
    static func f16(_ bits: UInt16) -> Float { QWQuant.f16tof32Bits(bits) }
    static func floats(_ d: Data) -> [Float] { d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) } }
    static func f16Vals(_ d: Data) -> [Float] {
        d.withUnsafeBytes { raw in
            (0..<d.count / 2).map { f16(raw.load(fromByteOffset: $0 * 2, as: UInt16.self)) }
        }
    }

    func makeIO(_ R: Int, residual: Bool, scales: Bool) throws -> IO {
        // budget: x + y + res (all always allocated) + γ + row scales + slack
        let arena = try QWArena(hal: Self.hal, weights: 0,
                                activations: R * Self.H * 2 * 3 + Self.H * 2 + R * 4 + 8 * 256,
                                kv: 0, gdnStates: 0)
        let a = arena.activationsRegion()
        let x = try a.allocate(QWTensorShape([R * Self.H], .f16))
        let g = try a.allocate(QWTensorShape([Self.H], .f16))
        let y = try a.allocate(QWTensorShape([R * Self.H], .f16))
        let res = try a.allocate(QWTensorShape([R * Self.H], .f16))
        let sc = try a.allocate(QWTensorShape([max(R, 1)], .f32))
        arena.copyIn(x, from: Self.fixture("rms_N\(R)_x.bin"))
        arena.copyIn(g, from: Self.fixture("rms_N\(R)_g.bin"))
        if residual { arena.copyIn(res, from: Self.fixture("rms_N\(R)_res_in.bin")) }
        _ = scales
        return IO(rows: R, arena: arena, x: x, g: g, y: y, res: res, scales: sc)
    }

    func run(_ io: IO, _ v: QWNormVariant) throws {
        guard let (cb, enc) = Self.hal.begin() else { throw NSError(domain: "hal", code: 1) }
        do {
            let needsRes = (v == .rmsnormResidual || v == .rmsnormResidualPrologueScale)
            let needsScales = (v == .rmsnormPrologueScale || v == .rmsnormResidualPrologueScale)
            try QWNorm.encode(enc, hal: Self.hal, variant: v, x: io.x, gamma: io.g, out: io.y,
                              residual: needsRes ? io.res : nil,
                              rowScalesOut: needsScales ? io.scales : nil, geometry: .qwen38_27b)
        } catch {
            // encoder MUST end even on the throw path (headNorm's variantNotHere)
            // or Metal aborts on live dealloc; don't commit — nothing was encoded.
            enc.endEncoding()
            throw error
        }
        Self.hal.end(cb, enc)
        cb.waitUntilCompleted()
    }
    func readF16(_ io: IO, _ n: Int) -> [Float] {
        let p = io.arena.cpuPointer(of: io.y).assumingMemoryBound(to: UInt16.self)
        return (0..<n).map { Self.f16(p[$0]) }
    }
    func readResF16(_ io: IO, _ n: Int) -> [Float] {
        let p = io.arena.cpuPointer(of: io.res).assumingMemoryBound(to: UInt16.self)
        return (0..<n).map { Self.f16(p[$0]) }
    }
    /// Count elements of (got, ref) that exceed absTol AND are not 1-ulp f16
    /// adjacent; returns (violations, worstAbsDiff, worstIdx, worstRatio).
    /// f16-out parity law (spec 06): fp32 sumsq ordering drift ~1e-5 flips
    static func parityViolations(_ got: [Float], _ ref: [Float], absTol: Float = 1e-3)
        -> (violations: Int, worstAbs: Double, worstIdx: Int, worstRatio: Double) {
        func f16Spacing(_ v: Float) -> Double {    // spacing of |v|'s f16 binade
            let e = Int((v.bitPattern >> 23) & 0xFF) - 127 - 10   // spacing exponent
            return e >= 0 ? Double(1 << e) : 1.0 / Double(1 << -e)
        }
        var viol = 0, worst = -1, worstAbs = 0.0, worstRatio = 0.0
        for i in 0..<got.count {
            let d = abs(Double(got[i]) - Double(ref[i]))
            if d > worstAbs { worstAbs = d; worst = i }
            guard d > Double(absTol) else { continue }
            // both sides exact f16 values → 1-ulp adjacency ⇔ d ≤ binade spacing
            let sp = min(f16Spacing(got[i]), f16Spacing(ref[i]))
            let r = d / sp
            worstRatio = max(worstRatio, r)
            if r > 1.0001 { viol += 1 }
        }
        return (viol, worstAbs, worst, worstRatio)
    }

    func readScales(_ io: IO, _ n: Int) -> [Float] {
        let p = io.arena.cpuPointer(of: io.scales).assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: p, count: n))
    }

    // MARK: gates

    func testRmsnormParityAllRows() throws {
        for R in [1, 512, 4096] {
            let io = try makeIO(R, residual: false, scales: false)
            try run(io, .rmsnorm)
            let got = readF16(io, R * Self.H)
            let ref = Self.floats(Self.fixture("rms_N\(R)_y.bin"))   // f32 fixture
            let v = Self.parityViolations(got, ref)
            XCTAssertEqual(v.violations, 0,
                "variant1 parity R=\(R): \(v.violations) beyond 1-ulp (worst \(v.worstAbs) @\(v.worstIdx) ratio \(v.worstRatio))")
        }
    }

    func testFusedResidualParityAndUpdate() throws {
        let R = 512
        let io = try makeIO(R, residual: true, scales: false)
        try run(io, .rmsnormResidual)
        let yGot = readF16(io, R * Self.H)
        let yRef = Self.floats(Self.fixture("rms_N\(R)_res_y.bin"))   // f32 fixture
        let resGot = readResF16(io, R * Self.H)
        let resRef = Self.f16Vals(Self.fixture("rms_N\(R)_res_out.bin"))  // f16 fixture
        var mR = 0.0
        for i in 0..<yGot.count {
            mR = max(mR, abs(Double(resGot[i]) - Double(resRef[i])))
        }
        let v = Self.parityViolations(yGot, yRef)
        XCTAssertEqual(v.violations, 0,
            "variant2 y parity: \(v.violations) beyond 1-ulp (worst \(v.worstAbs) @\(v.worstIdx) ratio \(v.worstRatio))")
        XCTAssertEqual(mR, 0, "residual update must be BIT-exact f16 add")
    }

    func testWideFusedResidualParityAndUpdate() throws {
        let io = try makeIO(1, residual: true, scales: false)
        let pipeline = try Self.hal.pipeline("qwg_rmsnorm_residual_wide")
        guard let (cb, enc) = Self.hal.begin() else { return XCTFail("cmdbuffer") }
        enc.setComputePipelineState(pipeline)
        io.x.bind(enc, index: 0)
        io.g.bind(enc, index: 1)
        io.y.bind(enc, index: 2)
        var hiddenU4 = UInt32(Self.H / 8)
        var eps = QWModelGeometry.qwen38_27b.rmsEps
        enc.setBytes(&hiddenU4, length: 4, index: 3)
        enc.setBytes(&eps, length: 4, index: 4)
        io.res.bind(enc, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: Int(hiddenU4), height: 1, depth: 1))
        Self.hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)

        let got = readF16(io, Self.H)
        let reference = Self.floats(Self.fixture("rms_N1_res_y.bin"))
        let parity = Self.parityViolations(got, reference)
        XCTAssertEqual(parity.violations, 0,
                       "wide residual parity: \(parity.violations) violations (worst \(parity.worstAbs))")
        let residual = readResF16(io, Self.H)
        let expected = Self.f16Vals(Self.fixture("rms_N1_res_out.bin"))
        XCTAssertEqual(residual, expected, "residual update must be bit-exact f16 add")
    }

    func testPrologueScaleParity() throws {
        let R = 4096
        let io = try makeIO(R, residual: false, scales: true)
        try run(io, .rmsnormPrologueScale)
        let scGot = readScales(io, R)
        let scRef = Self.floats(Self.fixture("rms_N\(R)_rs.bin"))
        var rel = 0.0
        for i in 0..<R {
            rel = max(rel, abs(Double(scGot[i]) - Double(scRef[i])) / abs(Double(scRef[i])))
        }
        XCTAssertLessThanOrEqual(rel, 5e-4, "row-scale side vector relative parity")
    }

    func testResidualPrologueScaleParity() throws {
        let R = 512
        let io = try makeIO(R, residual: true, scales: true)
        try run(io, .rmsnormResidualPrologueScale)

        // 1. Fused residual update is bit-exact
        let resGot = readResF16(io, R * Self.H)
        let resRef = Self.f16Vals(Self.fixture("rms_N\(R)_res_out.bin"))
        var mR = 0.0
        for i in 0..<resGot.count {
            mR = max(mR, abs(Double(resGot[i]) - Double(resRef[i])))
        }
        XCTAssertEqual(mR, 0, "residual update in variant 4 must be BIT-exact f16 add")

        // 2. Row scales match rstd of the updated residual
        let scGot = readScales(io, R)
        var maxRelScale = 0.0
        for r in 0..<R {
            var sumSq: Float = 0
            for c in 0..<Self.H {
                let v = resRef[r * Self.H + c]
                sumSq += v * v
            }
            let rstdRef = 1.0 / sqrt(sumSq / Float(Self.H) + 1e-6)
            let rel = abs(Double(scGot[r]) - Double(rstdRef)) / Double(rstdRef)
            maxRelScale = max(maxRelScale, rel)
        }
        XCTAssertLessThanOrEqual(maxRelScale, 5e-4, "variant 4 row-scale relative error vs residual rstd")

        // 3. y is fwht32(res · (1 + γ)) unscaled by rstd
        let yGot = readF16(io, R * Self.H)
        let gRef = Self.f16Vals(Self.fixture("rms_N\(R)_g.bin"))
        var yRefUnscaled = [Float](repeating: 0, count: R * Self.H)
        for r in 0..<R {
            for c in 0..<Self.H {
                let idx = r * Self.H + c
                let g1 = gRef[c] + 1.0
                yRefUnscaled[idx] = resRef[idx] * g1
            }
        }
        yRefUnscaled.withUnsafeMutableBufferPointer { buf in
            for b in stride(from: 0, to: buf.count, by: 32) {
                QWQuant.fwht32OrthoBlock(buf.baseAddress! + b)
            }
        }
        for i in 0..<yRefUnscaled.count {
            yRefUnscaled[i] = Float(Float16(yRefUnscaled[i]))
        }
        let v = Self.parityViolations(yGot, yRefUnscaled, absTol: 1e-3)
        XCTAssertEqual(v.violations, 0,
            "variant 4 unscaled y parity: \(v.violations) beyond tol (worst \(v.worstAbs) @\(v.worstIdx) ratio \(v.worstRatio))")
    }

    func testHeadNormNotHere() throws {
        let io = try makeIO(2, residual: false, scales: false)
        XCTAssertThrowsError(try run(io, .headNorm)) { err in
            guard case QWNormError.variantNotHere = err else { return XCTFail("\(err)") }
        }
    }
}
