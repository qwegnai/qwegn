// QWDecisionTopologyAttnNAXParityTests.swift — Parity gate comparing NAX topological FlashAttention vs scalar reference.
import XCTest
import Metal
import Foundation
@testable import QWAttn
import QWCore
import QWHAL
import QWAlloc

final class QWDecisionTopologyAttnNAXParityTests: XCTestCase {
    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    private struct SplitMix64 {
        var state: UInt64
        mutating func next() -> UInt64 {
            state = state &+ 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            return z ^ (z >> 31)
        }
        mutating func nextSigned(_ scale: Float = 0.75) -> Float {
            let u = Float(next() >> 11) * (1.0 / 9007199254740992.0)
            return (u * 2.0 - 1.0) * scale
        }
    }

    private func makeF16Tensor(values: [Float], shape: [Int]) throws -> QWTensorValue {
        let ts = QWTensorShape(shape, .f16)
        let bytes = max(64, qwAlignUp(ts.totalBytes, 64))
        guard let buf = Self.hal.device.makeBuffer(length: bytes, options: .storageModeShared) else {
            XCTFail("Failed to allocate Metal buffer"); throw QWAttnError.badShape("alloc")
        }
        memset(buf.contents(), 0, bytes)
        let ptr = buf.contents().assumingMemoryBound(to: Float16.self)
        for (i, v) in values.enumerated() { ptr[i] = Float16(v) }
        return QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: bytes), offset: 0, shape: ts)
    }

    private func makeI32Tensor(values: [Int32], shape: [Int]) throws -> QWTensorValue {
        let ts = QWTensorShape(shape, .i32)
        let bytes = max(64, qwAlignUp(ts.totalBytes, 64))
        guard let buf = Self.hal.device.makeBuffer(length: bytes, options: .storageModeShared) else {
            XCTFail("Failed to allocate Metal buffer"); throw QWAttnError.badShape("alloc")
        }
        memset(buf.contents(), 0, bytes)
        let ptr = buf.contents().assumingMemoryBound(to: Int32.self)
        for (i, v) in values.enumerated() { ptr[i] = v }
        return QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: bytes), offset: 0, shape: ts)
    }

    private func runPair(
        op: QWDecisionTopologyAttn,
        seqLen: Int,
        qHeads: Int,
        kvHeads: Int,
        headDim: Int,
        topology: QWDecisionModelConfig.AttentionTopology,
        groupIds: [Int32],
        stateEndPos: Int = 0,
        mediaPrefixLen: Int = 0,
        slidingWindow: Int = 0,
        seed: UInt64
    ) throws -> (maxDiff: Float, naxOut: [Float16]) {
        var rng = SplitMix64(state: seed)
        let qVals = (0..<(seqLen * qHeads * headDim)).map { _ in rng.nextSigned() }
        let kVals = (0..<(seqLen * kvHeads * headDim)).map { _ in rng.nextSigned() }
        let vVals = (0..<(seqLen * kvHeads * headDim)).map { _ in rng.nextSigned() }
        let qT = try makeF16Tensor(values: qVals, shape: [seqLen, qHeads * headDim])
        let kT = try makeF16Tensor(values: kVals, shape: [seqLen, kvHeads * headDim])
        let vT = try makeF16Tensor(values: vVals, shape: [seqLen, kvHeads * headDim])
        let gT = try makeI32Tensor(values: groupIds, shape: [seqLen])
        let outRef = try makeF16Tensor(values: [Float](repeating: 0, count: seqLen * qHeads * headDim), shape: [seqLen, qHeads * headDim])
        let outNax = try makeF16Tensor(values: [Float](repeating: 0, count: seqLen * qHeads * headDim), shape: [seqLen, qHeads * headDim])

        guard let (cb, enc) = Self.hal.begin() else { throw QWAttnError.badShape("cb") }
        try op.encodePrefill(
            enc, q: qT, k: kT, v: vT, groupIds: gT, out: outRef,
            seqLen: seqLen, qHeads: qHeads, kvHeads: kvHeads, headDim: headDim,
            topology: topology, stateEndPos: stateEndPos, mediaPrefixLen: mediaPrefixLen,
            slidingWindow: slidingWindow, forceScalar: true
        )
        try op.encodePrefill(
            enc, q: qT, k: kT, v: vT, groupIds: gT, out: outNax,
            seqLen: seqLen, qHeads: qHeads, kvHeads: kvHeads, headDim: headDim,
            topology: topology, stateEndPos: stateEndPos, mediaPrefixLen: mediaPrefixLen,
            slidingWindow: slidingWindow, forceScalar: false
        )
        Self.hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertEqual(cb.status, .completed)

        let total = seqLen * qHeads * headDim
        let rPtr = outRef.region.pointer.advanced(by: outRef.offset).assumingMemoryBound(to: Float16.self)
        let nPtr = outNax.region.pointer.advanced(by: outNax.offset).assumingMemoryBound(to: Float16.self)
        var maxDiff: Float = 0
        var naxOut = [Float16](repeating: 0, count: total)
        for i in 0..<total {
            naxOut[i] = nPtr[i]
            XCTAssertTrue(nPtr[i].isFinite)
            maxDiff = max(maxDiff, abs(Float(rPtr[i]) - Float(nPtr[i])))
        }
        return (maxDiff, naxOut)
    }

    private func cpuAllowElem(
        qPos: Int,
        kPos: Int,
        qGroup: Int32,
        kGroup: Int32,
        topology: QWDecisionModelConfig.AttentionTopology,
        stateEndPos: Int,
        mediaPrefixLen: Int,
        slidingWindow: Int
    ) -> Bool {
        if qGroup < 0 || kGroup < 0 { return false }
        if slidingWindow > 0 {
            return (kPos <= qPos) && ((qPos - kPos) < slidingWindow)
        }
        switch topology {
        case .causal:
            return kPos <= qPos
        case .prefixBidirectional:
            return (qPos < stateEndPos && kPos < stateEndPos) || (kPos <= qPos)
        case .promptBidirectional:
            return true
        case .packedSet:
            return qGroup == 0 || kGroup == 0 || kGroup == qGroup
        case .bidirectionalEncoder(let isolateMediaPrefix):
            let effMediaLen = isolateMediaPrefix ? max(0, mediaPrefixLen) : 0
            return effMediaLen == 0 || ((qPos < effMediaLen) == (kPos < effMediaLen))
        }
    }

    private func runScalarVsCPU(
        op: QWDecisionTopologyAttn,
        seqLen: Int,
        qHeads: Int,
        kvHeads: Int,
        headDim: Int,
        topology: QWDecisionModelConfig.AttentionTopology,
        groupIds: [Int32],
        stateEndPos: Int = 0,
        mediaPrefixLen: Int = 0,
        slidingWindow: Int = 0,
        seed: UInt64
    ) throws -> (maxDiff: Float, scalarOut: [Float16]) {
        var rng = SplitMix64(state: seed)
        let qRaw = (0..<(seqLen * qHeads * headDim)).map { _ in Float(Float16(rng.nextSigned())) }
        let kRaw = (0..<(seqLen * kvHeads * headDim)).map { _ in Float(Float16(rng.nextSigned())) }
        let vRaw = (0..<(seqLen * kvHeads * headDim)).map { _ in Float(Float16(rng.nextSigned())) }
        let qT = try makeF16Tensor(values: qRaw, shape: [seqLen, qHeads * headDim])
        let kT = try makeF16Tensor(values: kRaw, shape: [seqLen, kvHeads * headDim])
        let vT = try makeF16Tensor(values: vRaw, shape: [seqLen, kvHeads * headDim])
        let gT = try makeI32Tensor(values: groupIds, shape: [seqLen])
        let outScalar = try makeF16Tensor(values: [Float](repeating: 0, count: seqLen * qHeads * headDim), shape: [seqLen, qHeads * headDim])

        guard let (cb, enc) = Self.hal.begin() else { throw QWAttnError.badShape("cb") }
        try op.encodePrefill(
            enc, q: qT, k: kT, v: vT, groupIds: gT, out: outScalar,
            seqLen: seqLen, qHeads: qHeads, kvHeads: kvHeads, headDim: headDim,
            topology: topology, stateEndPos: stateEndPos, mediaPrefixLen: mediaPrefixLen,
            slidingWindow: slidingWindow, forceScalar: true
        )
        Self.hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertEqual(cb.status, .completed)

        let gqaRatio = max(1, qHeads / max(1, kvHeads))
        let scale = 1.0 / sqrtf(Float(headDim))
        let total = seqLen * qHeads * headDim
        let sPtr = outScalar.region.pointer.advanced(by: outScalar.offset).assumingMemoryBound(to: Float16.self)
        var scalarOut = [Float16](repeating: 0, count: total)
        var maxDiff: Float = 0

        for qPos in 0..<seqLen {
            let qGroup = groupIds[qPos]
            for h in 0..<qHeads {
                let kvHead = min(kvHeads - 1, h / gqaRatio)
                let qOff = (qPos * qHeads + h) * headDim
                if qGroup < 0 {
                    for d in 0..<headDim {
                        let got = sPtr[qOff + d]
                        scalarOut[qOff + d] = got
                        XCTAssertEqual(Float(got), 0.0, accuracy: 0.0)
                    }
                    continue
                }
                var m: Float = -Float.infinity
                var scores = [Float](repeating: -Float.infinity, count: seqLen)
                for kPos in 0..<seqLen {
                    let allow = cpuAllowElem(
                        qPos: qPos, kPos: kPos, qGroup: qGroup, kGroup: groupIds[kPos],
                        topology: topology, stateEndPos: stateEndPos,
                        mediaPrefixLen: mediaPrefixLen, slidingWindow: slidingWindow
                    )
                    if !allow { continue }
                    let kOff = (kPos * kvHeads + kvHead) * headDim
                    var dot: Float = 0
                    for d in 0..<headDim { dot += qRaw[qOff + d] * kRaw[kOff + d] }
                    let s = dot * scale
                    scores[kPos] = s
                    if s > m { m = s }
                }
                var l: Float = 0
                var acc = [Float](repeating: 0, count: headDim)
                for kPos in 0..<seqLen where scores[kPos].isFinite {
                    let p = expf(scores[kPos] - m)
                    l += p
                    let vOff = (kPos * kvHeads + kvHead) * headDim
                    for d in 0..<headDim { acc[d] += p * vRaw[vOff + d] }
                }
                let invL = (l > 0) ? (1.0 / l) : 0.0
                for d in 0..<headDim {
                    let got = sPtr[qOff + d]
                    scalarOut[qOff + d] = got
                    XCTAssertTrue(got.isFinite)
                    let ref = Float(Float16(acc[d] * invL))
                    maxDiff = max(maxDiff, abs(Float(got) - ref))
                }
            }
        }
        return (maxDiff, scalarOut)
    }

    func testAllTopologiesGQAAndPaddingParityD64AndD128() throws {
        try XCTSkipUnless(Self.hal.caps.isMetal4TensorCapable, "Decision topology NAX requires Metal 4 tensor hardware")
        let op = try QWDecisionTopologyAttn(hal: Self.hal)

        for headDim in [64, 128] {
            // 1. Causal S=65
            let gAll0_65 = [Int32](repeating: 0, count: 65)
            let c = try runPair(
                op: op, seqLen: 65, qHeads: 8, kvHeads: 2, headDim: headDim,
                topology: .causal(supportsPrefixCache: false), groupIds: gAll0_65, seed: UInt64(10 + headDim)
            )
            XCTAssertLessThan(c.maxDiff, 2e-3, "causal hd=\(headDim)")

            // 2. Prefix bidirectional S=73, stateEndPos=37
            let p = try runPair(
                op: op, seqLen: 73, qHeads: 16, kvHeads: 4, headDim: headDim,
                topology: .prefixBidirectional,
                groupIds: [Int32](repeating: 0, count: 73), stateEndPos: 37, seed: UInt64(20 + headDim)
            )
            XCTAssertLessThan(p.maxDiff, 2e-3, "prefixBidirectional hd=\(headDim)")

            // 3. Prompt bidirectional S=17
            let pb = try runPair(
                op: op, seqLen: 17, qHeads: 8, kvHeads: 2, headDim: headDim,
                topology: .promptBidirectional, groupIds: [Int32](repeating: 0, count: 17), seed: UInt64(30 + headDim)
            )
            XCTAssertLessThan(pb.maxDiff, 2e-3, "promptBidirectional hd=\(headDim)")

            // 4. PackedSet S=131 with shared prefix (0), isolated groups (1, 2, 3), and padded (-1)
            var packedGroups = [Int32](repeating: 0, count: 131)
            for i in 40..<70 { packedGroups[i] = 1 }
            for i in 70..<102 { packedGroups[i] = 2 }
            for i in 102..<120 { packedGroups[i] = 3 }
            for i in 120..<131 { packedGroups[i] = -1 }
            packedGroups[5] = -1
            let ps = try runPair(
                op: op, seqLen: 131, qHeads: 8, kvHeads: 2, headDim: headDim,
                topology: .packedSet(maxSetSize: 8), groupIds: packedGroups, seed: UInt64(40 + headDim)
            )
            XCTAssertLessThan(ps.maxDiff, 2e-3, "packedSet hd=\(headDim)")
            // Padded rows (-1) must output exact zero.
            let rowStride = 8 * headDim
            for paddedRow in [5, 120, 130] {
                for d in 0..<rowStride {
                    XCTAssertEqual(Float(ps.naxOut[paddedRow * rowStride + d]), 0.0, accuracy: 0.0)
                }
            }

            // 5. BidirectionalEncoder with mediaPrefixLen=20
            let be = try runPair(
                op: op, seqLen: 67, qHeads: 8, kvHeads: 2, headDim: headDim,
                topology: .bidirectionalEncoder(isolateMediaPrefix: true),
                groupIds: [Int32](repeating: 0, count: 67), mediaPrefixLen: 20, seed: UInt64(50 + headDim)
            )
            XCTAssertLessThan(be.maxDiff, 2e-3, "bidirectionalEncoder hd=\(headDim)")

            // 6. SlidingWindow=16 at S=65
            let sw = try runPair(
                op: op, seqLen: 65, qHeads: 8, kvHeads: 2, headDim: headDim,
                topology: .causal(supportsPrefixCache: false),
                groupIds: gAll0_65, slidingWindow: 16, seed: UInt64(60 + headDim)
            )
            XCTAssertLessThan(sw.maxDiff, 2e-3, "slidingWindow hd=\(headDim)")

            // 7. Partial sgi 1 Q tile (S=55 -> 55 % 32 == 23) causal
            let c55 = try runPair(
                op: op, seqLen: 55, qHeads: 8, kvHeads: 2, headDim: headDim,
                topology: .causal(supportsPrefixCache: false),
                groupIds: [Int32](repeating: 0, count: 55), seed: UInt64(70 + headDim)
            )
            XCTAssertLessThan(c55.maxDiff, 2e-3, "causal S=55 hd=\(headDim)")

            // 8. Partial sgi 1 Q tile (S=55) packedSet with slidingWindow=16 override + -1 padding
            var g55 = [Int32](repeating: 0, count: 55)
            for i in 16..<30 { g55[i] = 1 }
            for i in 30..<48 { g55[i] = 2 }
            for i in 48..<55 { g55[i] = -1 }
            g55[10] = -1
            let psw55 = try runPair(
                op: op, seqLen: 55, qHeads: 8, kvHeads: 2, headDim: headDim,
                topology: .packedSet(maxSetSize: 8), groupIds: g55, slidingWindow: 16, seed: UInt64(80 + headDim)
            )
            XCTAssertLessThan(psw55.maxDiff, 2e-3, "packedSet+slidingWindow S=55 hd=\(headDim)")
            for paddedRow in [10, 48, 54] {
                for d in 0..<rowStride {
                    XCTAssertEqual(Float(psw55.naxOut[paddedRow * rowStride + d]), 0.0, accuracy: 0.0)
                }
            }
        }
    }

    func testScalarAnalyticalBoundsMatchesCPUReference() throws {
        let op = try QWDecisionTopologyAttn(hal: Self.hal)
        let qHeads = 8
        let kvHeads = 2

        for headDim in [32, 256] {
            var gCausal = [Int32](repeating: 0, count: 41)
            gCausal[0] = -1
            gCausal[19] = -1
            gCausal[40] = -1
            let c = try runScalarVsCPU(
                op: op, seqLen: 41, qHeads: qHeads, kvHeads: kvHeads, headDim: headDim,
                topology: .causal(supportsPrefixCache: false), groupIds: gCausal, seed: UInt64(100 + headDim)
            )
            XCTAssertLessThan(c.maxDiff, 2e-3, "scalar causal hd=\(headDim)")

            let sw1 = try runScalarVsCPU(
                op: op, seqLen: 41, qHeads: qHeads, kvHeads: kvHeads, headDim: headDim,
                topology: .causal(supportsPrefixCache: false), groupIds: gCausal, slidingWindow: 1, seed: UInt64(110 + headDim)
            )
            XCTAssertLessThan(sw1.maxDiff, 2e-3, "scalar slidingWindow=1 hd=\(headDim)")

            let sw16 = try runScalarVsCPU(
                op: op, seqLen: 41, qHeads: qHeads, kvHeads: kvHeads, headDim: headDim,
                topology: .causal(supportsPrefixCache: false), groupIds: gCausal, slidingWindow: 16, seed: UInt64(120 + headDim)
            )
            XCTAssertLessThan(sw16.maxDiff, 2e-3, "scalar slidingWindow=16 hd=\(headDim)")

            for stateEnd in [0, 19, 100] {
                let pb = try runScalarVsCPU(
                    op: op, seqLen: 41, qHeads: qHeads, kvHeads: kvHeads, headDim: headDim,
                    topology: .prefixBidirectional, groupIds: gCausal, stateEndPos: stateEnd, seed: UInt64(130 + stateEnd + headDim)
                )
                XCTAssertLessThan(pb.maxDiff, 2e-3, "scalar prefixBidirectional stateEnd=\(stateEnd) hd=\(headDim)")
            }

            var gPrompt = [Int32](repeating: 0, count: 19)
            gPrompt[3] = -1
            gPrompt[18] = -1
            let pr = try runScalarVsCPU(
                op: op, seqLen: 19, qHeads: qHeads, kvHeads: kvHeads, headDim: headDim,
                topology: .promptBidirectional, groupIds: gPrompt, seed: UInt64(240 + headDim)
            )
            XCTAssertLessThan(pr.maxDiff, 2e-3, "scalar promptBidirectional hd=\(headDim)")

            // Interleaved/non-monotonic packedSet groups + -1 padding
            var gPacked = [Int32](repeating: 0, count: 45)
            for i in 8..<15 { gPacked[i] = 2 }
            for i in 15..<23 { gPacked[i] = 1 }
            for i in 23..<30 { gPacked[i] = 2 }
            for i in 30..<38 { gPacked[i] = 3 }
            for i in 38..<45 { gPacked[i] = -1 }
            gPacked[2] = -1
            let ps = try runScalarVsCPU(
                op: op, seqLen: 45, qHeads: qHeads, kvHeads: kvHeads, headDim: headDim,
                topology: .packedSet(maxSetSize: 8), groupIds: gPacked, seed: UInt64(250 + headDim)
            )
            XCTAssertLessThan(ps.maxDiff, 2e-3, "scalar packedSet hd=\(headDim)")

            let psw = try runScalarVsCPU(
                op: op, seqLen: 45, qHeads: qHeads, kvHeads: kvHeads, headDim: headDim,
                topology: .packedSet(maxSetSize: 8), groupIds: gPacked, slidingWindow: 8, seed: UInt64(260 + headDim)
            )
            XCTAssertLessThan(psw.maxDiff, 2e-3, "scalar packedSet+slidingWindow=8 hd=\(headDim)")

            for mediaLen in [0, 17, 60] {
                let beIso = try runScalarVsCPU(
                    op: op, seqLen: 41, qHeads: qHeads, kvHeads: kvHeads, headDim: headDim,
                    topology: .bidirectionalEncoder(isolateMediaPrefix: true),
                    groupIds: gCausal, mediaPrefixLen: mediaLen, seed: UInt64(300 + mediaLen + headDim)
                )
                XCTAssertLessThan(beIso.maxDiff, 2e-3, "scalar bidirEnc(iso=true,len=\(mediaLen)) hd=\(headDim)")
            }

            let beUniso = try runScalarVsCPU(
                op: op, seqLen: 41, qHeads: qHeads, kvHeads: kvHeads, headDim: headDim,
                topology: .bidirectionalEncoder(isolateMediaPrefix: false),
                groupIds: gCausal, mediaPrefixLen: 17, seed: UInt64(390 + headDim)
            )
            XCTAssertLessThan(beUniso.maxDiff, 2e-3, "scalar bidirEnc(iso=false) hd=\(headDim)")
        }
    }
}
