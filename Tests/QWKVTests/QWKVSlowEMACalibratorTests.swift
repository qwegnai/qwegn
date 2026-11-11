// QWKVSlowEMACalibratorTests.swift — Out-of-sample generalization, regime guard,
// and container-compatible Q4 round-trip tests for slow-EMA KV calibration (H4a).
import Foundation
import XCTest
@testable import QWKV
import QWCore

final class QWKVSlowEMACalibratorTests: XCTestCase {

    private func makeChannelOutlierKVSamples(tokens: Int, kvHeads: Int, headDim: Int, seed: UInt64) -> [Float] {
        var rng = seed
        func nextUniform() -> Float {
            rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27
            let u = (rng &* 0x2545_F491_4F6C_DD1D) >> 40
            return (Float(u) + 0.5) / Float(1 << 24)
        }
        func nextGaussian() -> Float {
            let u1 = max(nextUniform(), 1e-7)
            let u2 = nextUniform()
            return sqrt(-2.0 * log(u1)) * cos(2.0 * Float.pi * u2)
        }

        var out = [Float](repeating: 0, count: tokens * kvHeads * headDim)
        for t in 0..<tokens {
            for h in 0..<kvHeads {
                let scaleH: Float = 0.8 + 0.3 * Float(h)
                let base = (t * kvHeads + h) * headDim
                for d in 0..<headDim {
                    // Persistent RoPE/sink outlier channels (~7.5 sigma).
                    let isOutlier = (d == 3 || d == 67 || d == 131 || d == 199)
                    let g = nextGaussian()
                    out[base + d] = (isOutlier ? g * 7.5 : g) * scaleH + 0.25 * Float(h - 1)
                }
            }
        }
        return out
    }

    private func makeLaplaceHeavyTailedKVSamples(tokens: Int, kvHeads: Int, headDim: Int, seed: UInt64) -> [Float] {
        var rng = seed
        func nextUniform() -> Float {
            rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27
            let u = (rng &* 0x2545_F491_4F6C_DD1D) >> 40
            return (Float(u) + 0.5) / Float(1 << 24)
        }
        var out = [Float](repeating: 0, count: tokens * kvHeads * headDim)
        for i in 0..<out.count {
            // Symmetric Laplace^1.35 heavy-tailed KV distribution (kurtosis > 9).
            let u = nextUniform() - 0.5
            let sgn: Float = u >= 0 ? 1.0 : -1.0
            let mag = pow(-log(max(1.0 - 2.0 * abs(u), 1e-6)), 1.35)
            out[i] = sgn * mag
        }
        return out
    }

    func testRefusesRegimeViolationBelowEightTimesK() {
        let tooFew = [Float](repeating: 0.1, count: 4 * 16)
        XCTAssertThrowsError(
            try QWKVSlowEMACalibrator.fit(samples: tooFew, kvHeads: 4, headDim: 16, bits: 4)
        ) { err in
            XCTAssertTrue("\(err)".contains("regime"), "must refuse n < 8k: \(err)")
        }
    }

    func testCalibratedQ4BeatsUncalibratedOutOfSampleWithStandardDequantizer() throws {
        let kvHeads = 4, headDim = 256
        let rowStride = kvHeads * headDim
        let calibSamples = makeLaplaceHeavyTailedKVSamples(tokens: 32, kvHeads: kvHeads, headDim: headDim, seed: 0x1111_2222)
        let heldOutSamples = makeLaplaceHeavyTailedKVSamples(tokens: 24, kvHeads: kvHeads, headDim: headDim, seed: 0x9999_AAAA)

        let codebook4 = try QWKVSlowEMACalibrator.fit(
            samples: calibSamples, kvHeads: kvHeads, headDim: headDim, bits: 4, epochs: 2.0
        )
        XCTAssertEqual(codebook4.centroids.count, kvHeads)
        XCTAssertTrue(codebook4.centroids.allSatisfy { $0.count == 16 })

        var uncalibErrSq: Double = 0
        var calibErrSq: Double = 0
        var qBuf = [UInt8](repeating: 0, count: rowStride / 2)
        var sBuf = [Float16](repeating: 0, count: kvHeads * 2)
        var mBuf = [Float16](repeating: 0, count: kvHeads * 2)
        var deq = [Float](repeating: 0, count: rowStride)

        for t in 0..<24 {
            let row = Array(heldOutSamples[(t * rowStride)..<((t + 1) * rowStride)])

            try QWKVCompressedLayout.quantizeQ4(
                input: row, qOut: &qBuf, scalesOut: &sBuf, minsOut: &mBuf,
                kvHeads: kvHeads, headDim: headDim
            )
            try QWKVCompressedLayout.dequantizeQ4(
                qIn: qBuf, scalesIn: sBuf, minsIn: mBuf, out: &deq,
                kvHeads: kvHeads, headDim: headDim
            )
            for i in 0..<rowStride {
                let d = Double(row[i] - deq[i])
                uncalibErrSq += d * d
            }

            // Calibrated Q4 writes identical container layout decoded by standard dequantizeQ4!
            try QWKVCompressedLayout.quantizeQ4Calibrated(
                input: row, codebook: codebook4, qOut: &qBuf, scalesOut: &sBuf, minsOut: &mBuf
            )
            try QWKVCompressedLayout.dequantizeQ4(
                qIn: qBuf, scalesIn: sBuf, minsIn: mBuf, out: &deq,
                kvHeads: kvHeads, headDim: headDim
            )
            for i in 0..<rowStride {
                let d = Double(row[i] - deq[i])
                calibErrSq += d * d
            }
        }

        let uncalibMSE = uncalibErrSq / Double(24 * rowStride)
        let calibMSE = calibErrSq / Double(24 * rowStride)
        print(String(format: "[H4a_KV_Q4] held-out MSE: uncalibrated=%.6f vs slow-EMA calibrated=%.6f (%.1f%% reduction)",
                     uncalibMSE, calibMSE, (1.0 - calibMSE / uncalibMSE) * 100.0))
        XCTAssertLessThan(calibMSE, uncalibMSE * 0.86,
                          "slow-EMA calibrated Q4 must cut out-of-sample heavy-tailed KV MSE with zero layout change")
    }

    func testNonUniformSlowEMA3BitAnd4BitCodebooksGeneralizeOutOfSample() throws {
        let kvHeads = 4, headDim = 256
        let rowStride = kvHeads * headDim
        let calibSamples = makeChannelOutlierKVSamples(tokens: 32, kvHeads: kvHeads, headDim: headDim, seed: 0x3333_4444)
        let heldOut = makeChannelOutlierKVSamples(tokens: 16, kvHeads: kvHeads, headDim: headDim, seed: 0x7777_8888)

        let cb3 = try QWKVSlowEMACalibrator.fit(samples: calibSamples, kvHeads: kvHeads, headDim: headDim, bits: 3)
        let cb4 = try QWKVSlowEMACalibrator.fit(samples: calibSamples, kvHeads: kvHeads, headDim: headDim, bits: 4)

        var codes = [UInt8](repeating: 0, count: rowStride)
        var means = [Float16](repeating: 0, count: kvHeads * 2)
        var stds = [Float16](repeating: 0, count: kvHeads * 2)
        var deq3 = [Float](repeating: 0, count: rowStride)
        var deq4 = [Float](repeating: 0, count: rowStride)
        var qBuf = [UInt8](repeating: 0, count: rowStride / 2)
        var sBuf = [Float16](repeating: 0, count: kvHeads * 2)
        var mBuf = [Float16](repeating: 0, count: kvHeads * 2)
        var deqU4 = [Float](repeating: 0, count: rowStride)

        var mse3 = 0.0, mse4 = 0.0, mseUniform4 = 0.0, mseUniform3 = 0.0
        for t in 0..<16 {
            let row = Array(heldOut[(t * rowStride)..<((t + 1) * rowStride)])
            try QWKVSlowEMACalibrator.quantizeNonUniformLUT(
                input: row, codebook: cb3, codesOut: &codes, meansOut: &means, stdsOut: &stds
            )
            try QWKVSlowEMACalibrator.dequantizeNonUniformLUT(
                codesIn: codes, meansIn: means, stdsIn: stds, codebook: cb3, out: &deq3
            )
            try QWKVSlowEMACalibrator.quantizeNonUniformLUT(
                input: row, codebook: cb4, codesOut: &codes, meansOut: &means, stdsOut: &stds
            )
            try QWKVSlowEMACalibrator.dequantizeNonUniformLUT(
                codesIn: codes, meansIn: means, stdsIn: stds, codebook: cb4, out: &deq4
            )
            try QWKVCompressedLayout.quantizeQ4(
                input: row, qOut: &qBuf, scalesOut: &sBuf, minsOut: &mBuf,
                kvHeads: kvHeads, headDim: headDim
            )
            try QWKVCompressedLayout.dequantizeQ4(
                qIn: qBuf, scalesIn: sBuf, minsIn: mBuf, out: &deqU4,
                kvHeads: kvHeads, headDim: headDim
            )
            // Also compute group-128 uniform 3-bit baseline MSE for fair 3-bit comparison.
            for g in 0..<(kvHeads * 2) {
                let gBase = g * 128
                let gSlice = row[gBase..<(gBase + 128)]
                let mn = gSlice.min() ?? 0
                let mx = gSlice.max() ?? 0
                let sc = max((mx - mn) / 7.0, 1e-8)
                for i in 0..<128 {
                    let x = row[gBase + i]
                    let q = max(0.0, min(7.0, round((x - mn) / sc)))
                    let d = (q * sc + mn) - x
                    mseUniform3 += Double(d * d)
                }
            }
            for i in 0..<rowStride {
                mse3 += Double((row[i] - deq3[i]) * (row[i] - deq3[i]))
                mse4 += Double((row[i] - deq4[i]) * (row[i] - deq4[i]))
                mseUniform4 += Double((row[i] - deqU4[i]) * (row[i] - deqU4[i]))
            }
        }
        mse3 /= Double(16 * rowStride)
        mse4 /= Double(16 * rowStride)
        mseUniform4 /= Double(16 * rowStride)
        mseUniform3 /= Double(16 * rowStride)
        print(String(format: "[H4a_KV_LUT] held-out MSE: uniform_3b=%.6f -> slowEMA_3b=%.6f | uniform_Q4=%.6f -> slowEMA_4b=%.6f",
                     mseUniform3, mse3, mseUniform4, mse4))
        XCTAssertLessThan(mse4, mse3, "4-bit slow-EMA LUT must beat 3-bit slow-EMA LUT")
        XCTAssertLessThan(mse4, mseUniform4 * 0.50,
                          "4-bit slow-EMA LUT must halve (>50% reduction) out-of-sample KV MSE vs uniform Q4")
        XCTAssertLessThan(mse3, mseUniform3 * 0.40,
                          "3-bit slow-EMA LUT must cut uniform 3-bit KV MSE by >60% while shrinking 1M KV by 25%")
    }
}
