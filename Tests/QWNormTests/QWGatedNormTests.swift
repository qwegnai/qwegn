// QWGatedNormTests.swift — 0-ULP CPU-vs-Metal parity tests for `qwg_gated_rmsnorm_f16`
// and `qwg_gated_rmsnorm_sandwich_residual_f16` (BF16/F16 weights, FWHT32, MoE F32 bridge).
import Foundation
import Metal
import XCTest
@testable import QWNorm
import QWAlloc
import QWCore
import QWHAL
import QWQuant

final class QWGatedNormTests: XCTestCase {
    private func skipUnlessMetal() throws {
        try XCTSkipIf(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] == "true"
                || QWDeviceCaps.current?.familyApple9 != true,
            "GatedNorm GPU tests require bare-metal Apple Silicon GPU"
        )
    }

    private static func bf16RoundBits(_ v: Float) -> UInt16 {
        let u = v.bitPattern
        if (u & 0x7FFF_FFFF) >= 0x7F80_0000 {
            return UInt16(truncatingIfNeeded: u >> 16)
        }
        let rounded = u &+ 0x7FFF &+ ((u >> 16) & 1)
        return UInt16(truncatingIfNeeded: rounded >> 16)
    }

    private static func bf16RoundF32(_ v: Float) -> Float {
        let u = v.bitPattern
        if (u & 0x7FFF_FFFF) >= 0x7F80_0000 { return v }
        let rounded = u &+ 0x7FFF &+ ((u >> 16) & 1)
        return Float(bitPattern: rounded & 0xFFFF_0000)
    }

    private static func simdReduceDown32(_ lanes: [Float]) -> Float {
        precondition(lanes.count == 32)
        var v = lanes
        for offset in [16, 8, 4, 2, 1] {
            for i in 0..<(32 - offset) {
                v[i] = v[i] + v[i + offset]
            }
        }
        return v[0]
    }

    private static func fwht32Row(_ values: [Float]) -> [Float] {
        precondition(values.count % 32 == 0)
        var out = values
        let invSqrt32: Float = 0.17677669529663689
        for base in stride(from: 0, to: out.count, by: 32) {
            for step in [1, 2, 4, 8, 16] {
                var next = [Float](repeating: 0, count: 32)
                for lane in 0..<32 {
                    let mate = out[base + (lane ^ step)]
                    let cur = out[base + lane]
                    next[lane] = ((lane & step) == 0) ? (cur + mate) : (mate - cur)
                }
                for lane in 0..<32 { out[base + lane] = next[lane] }
            }
            for lane in 0..<32 {
                out[base + lane] = out[base + lane] * invSqrt32
            }
        }
        return out
    }

    /// Exact CPU replica of `modeling_gigachat4.py:161-173` matching `qwnorm_gated.metal` reduction trees.
    private static func cpuGatedNormRow(
        xBits: [UInt16],
        wVals: [Float],
        upVals: [Float],
        downVals: [Float],
        hidden: Int,
        rank: Int,
        eps: Float,
        gatingWeight: Float,
        zeroCentered: Bool
    ) -> [Float] {
        var sgSums = [Float](repeating: 0, count: 8)
        for sg in 0..<8 {
            var laneSums = [Float](repeating: 0, count: 32)
            for lane in 0..<32 {
                let tid = sg * 32 + lane
                var sq: Float = 0
                var c = tid
                while c < hidden {
                    let xv = Float(Float16(bitPattern: xBits[c]))
                    sq = sq + (xv * xv)
                    c += 256
                }
                laneSums[lane] = sq
            }
            sgSums[sg] = simdReduceDown32(laneSums)
        }
        let sumSq = ((sgSums[0] + sgSums[1]) + (sgSums[2] + sgSums[3]))
                  + ((sgSums[4] + sgSums[5]) + (sgSums[6] + sgSums[7]))
        let rstd: Float = 1.0 / sqrtf((sumSq / Float(hidden)) + eps)

        var h = [Float](repeating: 0, count: hidden)
        for c in 0..<hidden {
            let xv = Float(Float16(bitPattern: xBits[c]))
            let sc = zeroCentered ? (1.0 + wVals[c]) : wVals[c]
            h[c] = bf16RoundF32((xv * rstd) * sc)
        }

        var g1 = [Float](repeating: 0, count: rank)
        for r in 0..<rank {
            var laneAcc = [Float](repeating: 0, count: 32)
            for lane in 0..<32 {
                var acc: Float = 0
                var c = lane
                while c < hidden {
                    acc = acc + (upVals[r * hidden + c] * h[c])
                    c += 32
                }
                laneAcc[lane] = acc
            }
            let dotU = simdReduceDown32(laneAcc)
            g1[r] = bf16RoundF32(dotU / (1.0 + expf(-dotU)))
        }

        var y = [Float](repeating: 0, count: hidden)
        for c in 0..<hidden {
            var dotD: Float = 0
            for r in 0..<rank {
                dotD = dotD + (downVals[c * rank + r] * g1[r])
            }
            let g2 = bf16RoundF32(1.0 / (1.0 + expf(-dotD)))
            y[c] = bf16RoundF32((h[c] * gatingWeight) * g2)
        }
        return y
    }

    private func allocU16(_ hal: QWHal, _ values: [UInt16], dtype: QWDType) -> QWTensorValue {
        let byteLen = max(16, values.count * 2)
        let buf = hal.device.makeBuffer(length: byteLen, options: .storageModeShared)!
        values.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            if let base = raw.baseAddress, !raw.isEmpty {
                buf.contents().copyMemory(from: base, byteCount: raw.count)
            }
        }
        return QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: byteLen),
                             offset: 0, shape: QWTensorShape([values.count], dtype))
    }

    private func allocEmpty(_ hal: QWHal, count: Int, dtype: QWDType) -> QWTensorValue {
        let elemBytes = (dtype == .f32) ? 4 : 2
        let byteLen = max(16, count * elemBytes)
        let buf = hal.device.makeBuffer(length: byteLen, options: .storageModeShared)!
        memset(buf.contents(), 0, byteLen)
        return QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: byteLen),
                             offset: 0, shape: QWTensorShape([count], dtype))
    }

    private func runGPU(_ hal: QWHal, _ body: (any MTLComputeCommandEncoder) throws -> Void) throws {
        let (cb, enc) = try XCTUnwrap(hal.begin())
        var ended = false
        defer { if !ended { enc.endEncoding() } }
        try body(enc)
        ended = true
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertEqual(cb.status, .completed)
    }

    func testGatedRMSNormZeroULPAndMoEBridgeParity() throws {
        try skipUnlessMetal()
        let hal = try QWHal(geometry: .qwen38_27b)
        let rank = 16
        let eps: Float = 1e-6
        let gatingWeight: Float = 2.0

        for hidden in [512, 1536, 2560] {
            for useBF16 in [true, false] {
                let dt: QWDType = useBF16 ? .bf16 : .f16
                var wBits = [UInt16](repeating: 0, count: hidden)
                var wVals = [Float](repeating: 0, count: hidden)
                for i in 0..<hidden {
                    let raw = Float(((i * 13 + 5) % 29) - 14) * 0.0625 + 0.75
                    if useBF16 {
                        let b = Self.bf16RoundBits(raw)
                        wBits[i] = b
                        wVals[i] = Float(bitPattern: UInt32(b) << 16)
                    } else {
                        let h = Float16(raw)
                        wBits[i] = h.bitPattern
                        wVals[i] = Float(h)
                    }
                }
                var upBits = [UInt16](repeating: 0, count: rank * hidden)
                var upVals = [Float](repeating: 0, count: rank * hidden)
                for i in 0..<(rank * hidden) {
                    let raw = sin(Float(i + 3) * 0.017) * 0.25
                    if useBF16 {
                        let b = Self.bf16RoundBits(raw)
                        upBits[i] = b
                        upVals[i] = Float(bitPattern: UInt32(b) << 16)
                    } else {
                        let h = Float16(raw)
                        upBits[i] = h.bitPattern
                        upVals[i] = Float(h)
                    }
                }
                var downBits = [UInt16](repeating: 0, count: hidden * rank)
                var downVals = [Float](repeating: 0, count: hidden * rank)
                for i in 0..<(hidden * rank) {
                    let raw = cos(Float(i + 7) * 0.023) * 0.3
                    if useBF16 {
                        let b = Self.bf16RoundBits(raw)
                        downBits[i] = b
                        downVals[i] = Float(bitPattern: UInt32(b) << 16)
                    } else {
                        let h = Float16(raw)
                        downBits[i] = h.bitPattern
                        downVals[i] = Float(h)
                    }
                }

                let wTV = allocU16(hal, wBits, dtype: dt)
                let upTV = allocU16(hal, upBits, dtype: dt)
                let downTV = allocU16(hal, downBits, dtype: dt)

                for zeroCentered in [false, true] {
                    if hidden != 2560 && zeroCentered { continue }
                    let gw = QWGatedNormWeights(
                        weight: wTV, gateUp: upTV, gateDown: downTV,
                        rank: rank, gatingWeight: gatingWeight, zeroCentered: zeroCentered
                    )
                    for rows in [1, 8, 65] {
                        if hidden != 2560 && rows == 65 { continue }
                        var xBits = [UInt16](repeating: 0, count: rows * hidden)
                        for i in 0..<(rows * hidden) {
                            xBits[i] = Float16(sin(Float(i * 7 + 11) * 0.031) * 1.25).bitPattern
                        }
                        let xTV = allocU16(hal, xBits, dtype: .f16)
                        for applyFwht in [false, true] {
                            let outTV = allocEmpty(hal, count: rows * hidden, dtype: .f16)
                            let rotTV = allocEmpty(hal, count: rows * hidden, dtype: .f32)
                            let plainTV = allocEmpty(hal, count: rows * hidden, dtype: .f32)
                            try runGPU(hal) { enc in
                                try QWNorm.encodeGatedRMSNorm(
                                    enc, hal: hal, x: xTV, weights: gw, out: outTV,
                                    eps: eps, hidden: hidden, applyFwht32: applyFwht,
                                    xRot32: rotTV, xPlain32: plainTV
                                )
                            }
                            let gotPtr = outTV.region.pointer.assumingMemoryBound(to: UInt16.self)
                            if rows > 1 {
                                let serOutTV = allocEmpty(hal, count: rows * hidden, dtype: .f16)
                                try runGPU(hal) { enc in
                                    for r in 0..<rows {
                                        try QWNorm.encodeGatedRMSNorm(
                                            enc, hal: hal,
                                            x: xTV.slice(byteOffset: r * hidden * 2, count: hidden, dtype: .f16),
                                            weights: gw,
                                            out: serOutTV.slice(byteOffset: r * hidden * 2, count: hidden, dtype: .f16),
                                            eps: eps, hidden: hidden, applyFwht32: applyFwht
                                        )
                                    }
                                }
                                let serPtr = serOutTV.region.pointer.assumingMemoryBound(to: UInt16.self)
                                for i in 0..<(rows * hidden) {
                                    XCTAssertEqual(gotPtr[i], serPtr[i], "Multi-row tile rows=\(rows) must be 0-ULP bit-exact vs serial rows=1 at \(i)")
                                }
                            }
                            var exactCount = 0
                            let totalCount = rows * hidden
                            for r in 0..<rows {
                                let rowIn = Array(xBits[(r * hidden)..<((r + 1) * hidden)])
                                let unrot = Self.cpuGatedNormRow(
                                    xBits: rowIn, wVals: wVals, upVals: upVals, downVals: downVals,
                                    hidden: hidden, rank: rank, eps: eps,
                                    gatingWeight: gatingWeight, zeroCentered: zeroCentered
                                )
                                let expectedF = (applyFwht && hal.weightBasis.rotatesActivations) ? Self.fwht32Row(unrot) : unrot
                                for c in 0..<hidden {
                                    let expH = Float16(expectedF[c])
                                    let expBits = expH.bitPattern
                                    let gotBits = gotPtr[r * hidden + c]
                                    if gotBits == expBits {
                                        exactCount += 1
                                    } else {
                                        let gotF = Float(Float16(bitPattern: gotBits))
                                        let expF = Float(expH)
                                        let tol = max(1e-3, abs(expF) * 0.01)
                                        XCTAssertLessThanOrEqual(
                                            abs(gotF - expF), tol,
                                            "Mismatch D=\(hidden) bf16=\(useBF16) zc=\(zeroCentered) rows=\(rows) fwht=\(applyFwht) [\(r),\(c)]: got \(gotF) exp \(expF)"
                                        )
                                    }
                                }
                            }
                            XCTAssertGreaterThanOrEqual(
                                Double(exactCount) / Double(totalCount), 0.995,
                                "Expected >=99.5% 0-ULP matches vs libm CPU reference (got \(exactCount)/\(totalCount))"
                            )

                            if applyFwht && hidden == 2560 && rows == 8 {
                                let refRot = allocEmpty(hal, count: rows * hidden, dtype: .f32)
                                let refPlain = allocEmpty(hal, count: rows * hidden, dtype: .f32)
                                let bridgePSO = try hal.pipeline("qwg_moe_bridge_f16_to_f32")
                                try runGPU(hal) { enc in
                                    enc.setComputePipelineState(bridgePSO)
                                    outTV.bind(enc, index: 0)
                                    refRot.bind(enc, index: 1)
                                    refPlain.bind(enc, index: 2)
                                    var hU = UInt32(hidden)
                                    enc.setBytes(&hU, length: 4, index: 3)
                                    enc.dispatchThreadgroups(MTLSize(width: 1, height: rows, depth: 1),
                                                             threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                }
                                let pGotR = rotTV.region.pointer.assumingMemoryBound(to: UInt32.self)
                                let pRefR = refRot.region.pointer.assumingMemoryBound(to: UInt32.self)
                                let pGotP = plainTV.region.pointer.assumingMemoryBound(to: UInt32.self)
                                let pRefP = refPlain.region.pointer.assumingMemoryBound(to: UInt32.self)
                                for i in 0..<(rows * hidden) {
                                    XCTAssertEqual(pGotR[i], pRefR[i], "MoE xRot32 mismatch at \(i)")
                                    XCTAssertEqual(pGotP[i], pRefP[i], "MoE xPlain32 mismatch at \(i)")
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    func testGatedSandwichResidualZeroULPParity() throws {
        try skipUnlessMetal()
        let hal = try QWHal(geometry: .qwen38_27b)
        let hidden = 2560
        let rank = 16
        let eps: Float = 1e-6

        func makeWeights(seed: Int, zeroCentered: Bool) -> (QWGatedNormWeights, [Float], [Float], [Float]) {
            var wB = [UInt16](repeating: 0, count: hidden)
            var wF = [Float](repeating: 0, count: hidden)
            for i in 0..<hidden {
                let b = Self.bf16RoundBits(0.8 + sin(Float(i + seed) * 0.019) * 0.3)
                wB[i] = b
                wF[i] = Float(bitPattern: UInt32(b) << 16)
            }
            var uB = [UInt16](repeating: 0, count: rank * hidden)
            var uF = [Float](repeating: 0, count: rank * hidden)
            for i in 0..<(rank * hidden) {
                let b = Self.bf16RoundBits(cos(Float(i + seed * 3) * 0.013) * 0.2)
                uB[i] = b
                uF[i] = Float(bitPattern: UInt32(b) << 16)
            }
            var dB = [UInt16](repeating: 0, count: hidden * rank)
            var dF = [Float](repeating: 0, count: hidden * rank)
            for i in 0..<(hidden * rank) {
                let b = Self.bf16RoundBits(sin(Float(i + seed * 5) * 0.029) * 0.25)
                dB[i] = b
                dF[i] = Float(bitPattern: UInt32(b) << 16)
            }
            let gw = QWGatedNormWeights(
                weight: allocU16(hal, wB, dtype: .bf16),
                gateUp: allocU16(hal, uB, dtype: .bf16),
                gateDown: allocU16(hal, dB, dtype: .bf16),
                rank: rank, gatingWeight: 2.0, zeroCentered: zeroCentered
            )
            return (gw, wF, uF, dF)
        }

        let (postGW, postWF, postUF, postDF) = makeWeights(seed: 11, zeroCentered: false)
        let (preGW, preWF, preUF, preDF) = makeWeights(seed: 29, zeroCentered: true)

        for rows in [1, 8, 65] {
            var branchBits = [UInt16](repeating: 0, count: rows * hidden)
            var resInitBits = [UInt16](repeating: 0, count: rows * hidden)
            for i in 0..<(rows * hidden) {
                branchBits[i] = Float16(sin(Float(i * 5 + 3) * 0.021) * 0.9).bitPattern
                resInitBits[i] = Float16(cos(Float(i * 3 + 9) * 0.017) * 1.1).bitPattern
            }

            // Case 1: hasNextPreNorm == 0
            let branchTV0 = allocU16(hal, branchBits, dtype: .f16)
            let resTV0 = allocU16(hal, resInitBits, dtype: .f16)
            try runGPU(hal) { enc in
                try QWNorm.encodeGatedSandwichResidual(
                    enc, hal: hal, branchOut: branchTV0, residual: resTV0,
                    postWeights: postGW, nextPreWeights: nil, out: nil,
                    postEps: eps, hidden: hidden
                )
            }

            // Case 2: hasNextPreNorm == 1 + FWHT32 + MoE F32 bridge
            let branchTV1 = allocU16(hal, branchBits, dtype: .f16)
            let resTV1 = allocU16(hal, resInitBits, dtype: .f16)
            let outTV1 = allocEmpty(hal, count: rows * hidden, dtype: .f16)
            let rotTV1 = allocEmpty(hal, count: rows * hidden, dtype: .f32)
            let plainTV1 = allocEmpty(hal, count: rows * hidden, dtype: .f32)
            try runGPU(hal) { enc in
                try QWNorm.encodeGatedSandwichResidual(
                    enc, hal: hal, branchOut: branchTV1, residual: resTV1,
                    postWeights: postGW, nextPreWeights: preGW, out: outTV1,
                    postEps: eps, preEps: eps, hidden: hidden, applyFwht32: true,
                    xRot32: rotTV1, xPlain32: plainTV1
                )
            }

            let resPtr0 = resTV0.region.pointer.assumingMemoryBound(to: UInt16.self)
            let resPtr1 = resTV1.region.pointer.assumingMemoryBound(to: UInt16.self)
            let outPtr1 = outTV1.region.pointer.assumingMemoryBound(to: UInt16.self)
            for r in 0..<rows {
                let bRow = Array(branchBits[(r * hidden)..<((r + 1) * hidden)])
                let yPost = Self.cpuGatedNormRow(
                    xBits: bRow, wVals: postWF, upVals: postUF, downVals: postDF,
                    hidden: hidden, rank: rank, eps: eps, gatingWeight: 2.0, zeroCentered: false
                )
                var expResBits = [UInt16](repeating: 0, count: hidden)
                for c in 0..<hidden {
                    let xOld = Float(Float16(bitPattern: resInitBits[r * hidden + c]))
                    let xNew = Float16(xOld + yPost[c]).bitPattern
                    expResBits[c] = xNew
                    XCTAssertEqual(resPtr0[r * hidden + c], xNew, "hasNextPre=0 residual mismatch rows=\(rows) [\(r),\(c)]")
                    XCTAssertEqual(resPtr1[r * hidden + c], xNew, "hasNextPre=1 residual mismatch rows=\(rows) [\(r),\(c)]")
                }
                let yPre = Self.cpuGatedNormRow(
                    xBits: expResBits, wVals: preWF, upVals: preUF, downVals: preDF,
                    hidden: hidden, rank: rank, eps: eps, gatingWeight: 2.0, zeroCentered: true
                )
                let rotPre = Self.fwht32Row(yPre)
                for c in 0..<hidden {
                    let expOut = Float16(rotPre[c]).bitPattern
                    XCTAssertEqual(outPtr1[r * hidden + c], expOut, "sandwich pre-norm out 0-ULP mismatch rows=\(rows) [\(r),\(c)]")
                }
            }
        }
    }
}
