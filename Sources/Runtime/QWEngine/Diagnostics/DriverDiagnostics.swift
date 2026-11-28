// DriverDiagnostics.swift — QWEngineDriver DriverDiagnostics family (split from QWEngineDriver.swift; same-module extension).
import Foundation
import Darwin
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWNorm
import QWRope
import QWKV
import QWAttn
import QWGDN
import QWMLP
import QWHead
import QWMTP
import QWQuant

extension QWEngineDriver {
    public func logitsStats() -> (nan: Int, argmax: Int, max: Float, min: Float) {
        guard standardState != nil else { return (0, 0, -.infinity, .infinity) }
        let p = sc.logits.region.buffer.contents()
            .advanced(by: sc.logits.region.base + sc.logits.offset)
            .assumingMemoryBound(to: Float.self)
        var nan = 0, am = 0
        var mx: Float = -.infinity, mn: Float = .infinity
        for i in 0..<geometry.vocab {
            let v = p[i]
            if v.isNaN { nan += 1; continue }
            if v > mx { mx = v; am = i }
            if v < mn { mn = v }
        }
        return (nan, am, mx, mn)
    }

    /// Golden-fixture door (P0-4): the LAST step's logits as [Float] —
    /// the f32 bytes exactly as the GPU wrote them (bitPattern-preserved
    /// readback; the engine is deterministic under the pinned-FMA law, so
    public func logitVector() -> [Float] {
        guard standardState != nil else { return [] }
        let p = sc.logits.region.buffer.contents()
            .advanced(by: sc.logits.region.base + sc.logits.offset)
            .assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: p, count: geometry.vocab))
    }

    /// Steady-state allocator counters (the soak gate's data feed).
    public var steadyAllocs: Int {
        trunkExecutor.steadyStateAllocations
    }

    var standardSteadyStateAllocations: Int {
        arena.weightsRegion().steadyStateAllocs + arena.activationsRegion().steadyStateAllocs
            + arena.gdnRegion().steadyStateAllocs
    }

    /// Purges the prefill activation scratchpad from physical RAM using madvise (MADV_DONTNEED).
    /// Safe to call once prompt prefill completes. Single-token decode scratchpad (sc)
    /// is kept intact.
    public func evictPrefillScratch() {
        guard standardState != nil, pscByteLength > 0 else { return }
        let ptr = pscBuffer.contents().advanced(by: pscByteOffset)
        _ = madvise(ptr, pscByteLength, MADV_DONTNEED)
    }

    // MARK: - session lifecycle ──────────────────────────────────────────────

    /// Admit a sequence: KV handle + a pre-carved, ZEROED GDN slot (both
    /// full slabs — 151 MB memset is ~5 ms on the M5's unified bandwidth,
    /// at ADMISSION only; never per step).
}
