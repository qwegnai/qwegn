// QWStandardQwen3Tests.swift — the standard qwen3 path: the module that
// serves the family, and the rope its heads take.
//
// The q/k norm is ONE vector shared by every head: upstream's
// `Qwen3RMSNorm(self.head_dim)` stores a [head_dim] weight, and the rope
// kernel reads it un-offset. The general kernel serves this family — the
// 27B's frozen one is compiled for a 256-half head.
import Foundation
import XCTest
import QWAlloc
import QWCore
import QWHAL
import QWQuant
import QWRope
@testable import QWEngine

final class QWStandardQwen3Tests: XCTestCase {
    /// The module serves the family. The registry is the only admission
    /// point, and a family with no module is refused loudly (Law L4).
    func testTheQwen3FamiliesHaveAModule() {
        XCTAssertNotNil(QWModuleRegistry.moduleType(for: .qwen3Dense))
        XCTAssertNotNil(QWModuleRegistry.moduleType(for: .qwen3MoE))
        XCTAssertEqual(Qwen3DenseModule.family, .qwen3Dense)
        XCTAssertEqual(Qwen3MoEModule.family, .qwen3MoE)
        XCTAssertTrue(QWModuleRegistry.supportedFamilies.contains(.qwen3Dense))
        XCTAssertTrue(QWModuleRegistry.supportedFamilies.contains(.qwen3MoE))
    }

    /// The family declares the q/k norm it carries, and a family that does
    /// not carry one does not claim it.
    func testTheFamilyDeclaresItsQKNorm() throws {
        XCTAssertTrue(try QWModelGeometry.stubTinyQwen3Geometry().descriptor.hasQKNorm)
        XCTAssertFalse(try QWModelGeometry.stubTinyLlamaGeometry().descriptor.hasQKNorm)
    }

    /// The rope the family's head takes: the shared q norm, the per-kv-head
    /// k norm and the rotation, against a CPU reference that runs the same
    /// maths. The general kernel had NO test before this one.
    func testTheGeneralRopeAppliesTheSharedQKNormAndTheRotation() throws {
        let geometry = try QWModelGeometry.stubTinyQwen3Geometry()
        let S = 2
        let head = geometry.headDim
        let qSlot = geometry.qProjOut
        let kAll = geometry.kvHeads * geometry.headDim
        let hal = try QWHal(geometry: geometry)
        let arena = try QWArena(hal: hal, weights: 1 << 20, activations: 4 << 20,
                                kv: 0, gdnStates: 0)
        let a = arena.activationsRegion()

        let q = try a.allocate(QWTensorShape([S * qSlot], .f16))
        let k = try a.allocate(QWTensorShape([S * kAll], .f16))
        let wq = try a.allocate(QWTensorShape([head], .f16))
        let wk = try a.allocate(QWTensorShape([kAll], .f16))
        let qo = try a.allocate(QWTensorShape([S * qSlot], .f16))
        let ko = try a.allocate(QWTensorShape([S * kAll], .f16))

        // Deterministic values: a fixed stride, so a failure is
        // reproducible and the numbers are not all ones.
        let qv = Self.values(count: S * qSlot, stride: 0.37)
        let kv = Self.values(count: S * kAll, stride: 0.23)
        let wqv = Self.values(count: head, stride: 0.11)
        let wkv = Self.values(count: kAll, stride: 0.17)
        arena.copyIn(q, from: Self.f16Data(qv))
        arena.copyIn(k, from: Self.f16Data(kv))
        arena.copyIn(wq, from: Self.f16Data(wqv))
        arena.copyIn(wk, from: Self.f16Data(wkv))
        let rows = QWRopeTables(rows: S, yarn: nil, geometry: geometry)

        let rope = try QWRopeApply(tables: rows, hal: hal, weights: arena.weightsRegion(),
                                   arena: arena, geometry: geometry)
        guard let (cb, enc) = hal.begin() else { throw XCTSkip("no command buffer") }
        do {
            try rope.encodeQK(enc, hal: hal, q: q, k: k, wq: wq, wk: wk,
                              qOut: qo, kOut: ko, positions: Array(0..<S))
        } catch {
            enc.endEncoding()
            throw error
        }
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error, "the rope failed: \(String(describing: cb.error))")

        let qGot = Self.f16Values(arena, qo, S * qSlot)
        let kGot = Self.f16Values(arena, ko, S * kAll)
        let qRef = Self.reference(x: qv, w: wqv, rows: rows.cosSin, S: S,
                                  qSlot: qSlot, head: head, ropeDims: geometry.rotaryDims,
                                  eps: geometry.rmsEps, sharedWeight: true)
        let kRef = Self.reference(x: kv, w: wkv, rows: rows.cosSin, S: S,
                                  qSlot: kAll, head: head, ropeDims: geometry.rotaryDims,
                                  eps: geometry.rmsEps, sharedWeight: false)
        for i in 0..<qGot.count {
            XCTAssertLessThan(abs(qGot[i] - qRef[i]), 2e-3 * max(1, abs(qRef[i])),
                              "q element \(i): got \(qGot[i]), want \(qRef[i])")
        }
        for i in 0..<kGot.count {
            XCTAssertLessThan(abs(kGot[i] - kRef[i]), 2e-3 * max(1, abs(kRef[i])),
                              "k element \(i): got \(kGot[i]), want \(kRef[i])")
        }
        // The norm must have run: without it the numbers would be the raw
        // activations, which differ by far more than the f16 tolerance.
        var differs = 0
        for i in 0..<qGot.count where abs(qGot[i] - qv[i]) > 1e-2 { differs += 1 }
        XCTAssertGreaterThan(differs, 0, "the q norm did not change the numbers")
    }

    /// The CPU reference: the shared norm, then the rotation, in the
    /// kernel's own order. The weight is the model's `1 + w`; the q norm is
    /// shared, the k norm per kv head.
    private static func reference(x: [Float], w: [Float], rows: [Float], S: Int,
                                  qSlot: Int, head: Int, ropeDims: Int,
                                  eps: Float, sharedWeight: Bool) -> [Float] {
        var out = [Float](repeating: 0, count: x.count)
        let half = ropeDims / 2
        for s in 0..<S {
            for h in 0..<(qSlot / head) {
                let base = s * qSlot + h * head
                var acc: Float = 0
                for j in 0..<head { acc += x[base + j] * x[base + j] }
                let rstd = 1 / sqrt(acc / Float(head) + eps)
                var xn = [Float](repeating: 0, count: head)
                for j in 0..<head {
                    let g = sharedWeight ? w[j] : w[h * head + j]
                    xn[j] = x[base + j] * rstd * (g + 1)
                }
                for j in 0..<ropeDims {
                    let p = j & (half - 1)
                    let c = rows[s * ropeDims + p]
                    let sn = rows[s * ropeDims + half + p]
                    let rot = j < half ? -xn[j + half] : xn[j - half]
                    out[base + j] = xn[j] * c + rot * sn
                }
                for j in ropeDims..<head { out[base + j] = xn[j] }
            }
        }
        return out
    }

    /// Deterministic values: a fixed stride, offset so the signs mix.
    private static func values(count: Int, stride: Float) -> [Float] {
        (0..<count).map { i in Float((i % 17)) * stride - 0.4 + Float(i % 3) * 0.1 }
    }

    private static func f16Data(_ v: [Float]) -> Data {
        var bytes = Data(capacity: v.count * 2)
        for x in v {
            let bits = QWQuant.f32tof16Bits(x)
            withUnsafeBytes(of: bits.littleEndian) { bytes.append(contentsOf: $0) }
        }
        return bytes
    }

    private static func f16Values(_ arena: QWArena, _ t: QWTensorValue, _ count: Int) -> [Float] {
        let p = arena.cpuPointer(of: t).assumingMemoryBound(to: UInt16.self)
        return (0..<count).map { Float(Float16(bitPattern: p[$0])) }
    }
}
