// QWKVSlowEMACalibrator.swift — Out-of-sample slow-EMA e-means KV calibrator
// (arXiv:2611.09501 §4.2 Table 3). Fits per-channel slow-EMA variance equalizers,
// 3/4-bit non-uniform codebooks, and container-compatible Q4 clipping envelopes.

import Foundation
import QWCore

public struct QWKVSlowEMACodebook: Codable, Sendable, Equatable {
    public let bits: Int
    public let kvHeads: Int
    public let headDim: Int
    /// Per-channel `slow-EMA` RMS scales `[kvHeads, headDim]` equalizing persistent RoPE/sink outlier channels.
    public let channelScales: [[Float]]
    /// Sorted normalized centroids `[kvHeads, 1 << bits]` learned via `slow-EMA` (`S_j / N_j`).
    public let centroids: [[Float]]
    /// Optimal container-compatible affine clipping ratios `[kvHeads]` (`alpha in (0.5, 1.0]`).
    public let affineClipRatios: [Float]

    public init(
        bits: Int,
        kvHeads: Int,
        headDim: Int,
        channelScales: [[Float]],
        centroids: [[Float]],
        affineClipRatios: [Float]
    ) {
        self.bits = bits
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.channelScales = channelScales
        self.centroids = centroids
        self.affineClipRatios = affineClipRatios
    }
}

public enum QWKVSlowEMACalibrator {

    /// Fits a per-head `slow-EMA` (`OUT_OF_SAMPLE_CALIB`) codebook on calibration KV rows.
    /// `samples` holds `tokenCount * kvHeads * headDim` floats; enforces `n >= 8 * k` per head.
    public static func fit(
        samples: [Float],
        kvHeads: Int = 4,
        headDim: Int = 256,
        bits: Int = 4,
        epochs: Double = 2.0,
        seed: UInt64 = 20261110
    ) throws -> QWKVSlowEMACodebook {
        let rowStride = kvHeads * headDim
        guard kvHeads > 0, headDim > 0, (3...4).contains(bits),
              !samples.isEmpty, samples.count % rowStride == 0 else {
            throw QWKVError.badArgument("invalid slow-EMA KV calibration geometry or sample count")
        }
        let k = 1 << bits
        let tokenCount = samples.count / rowStride
        let groupsPerHead = max(1, (headDim + 127) / 128)
        let pointsPerHead = tokenCount * headDim
        guard pointsPerHead >= 8 * k else {
            throw QWKVError.badArgument(
                "e-means regime violation: \(pointsPerHead) scalars/head < 8 * k (\(8 * k))"
            )
        }

        var allChanScales: [[Float]] = []
        var allCentroids: [[Float]] = []
        var clipRatios: [Float] = []
        allChanScales.reserveCapacity(kvHeads)
        allCentroids.reserveCapacity(kvHeads)
        clipRatios.reserveCapacity(kvHeads)

        for h in 0..<kvHeads {
            // 1. Annealed slow-EMA per-channel RMS tracker (equalizes persistent RoPE/sink channels).
            var emaVar = [Float](repeating: 0, count: headDim)
            var emaWeight: Float = 1e-10
            for t in 0..<tokenCount {
                let progress = Double(t) / Double(max(1, tokenCount - 1))
                let alpha = Float(0.1 + 0.9 * 0.5 * (1.0 + cos(Double.pi * progress))) * 0.25
                let gamma = 1.0 - alpha
                emaWeight = gamma * emaWeight + alpha
                let headBase = t * rowStride + h * headDim
                for d in 0..<headDim {
                    let v = samples[headBase + d]
                    emaVar[d] = gamma * emaVar[d] + alpha * (v * v)
                }
            }
            let chanScale = emaVar.map { max(sqrt($0 / max(emaWeight, 1e-12)), 1e-5) }
            allChanScales.append(chanScale)

            // 2. Normalize calibration tokens by channelScales + per-group zero-mean unit-RMS.
            var normVals: [Float] = []
            normVals.reserveCapacity(pointsPerHead)
            var groupSamples: [[Float]] = []
            groupSamples.reserveCapacity(min(256, tokenCount * groupsPerHead))

            for t in 0..<tokenCount {
                let headBase = t * rowStride + h * headDim
                for g in 0..<groupsPerHead {
                    let gStart = g * 128
                    let gLen = min(128, headDim - gStart)
                    let base = headBase + gStart
                    var eq = [Float](repeating: 0, count: gLen)
                    var mean: Float = 0
                    for i in 0..<gLen {
                        let v = samples[base + i] / chanScale[gStart + i]
                        eq[i] = v
                        mean += v
                    }
                    mean /= Float(gLen)
                    var varSum: Float = 0
                    for i in 0..<gLen {
                        let diff = eq[i] - mean
                        varSum += diff * diff
                    }
                    let invStd = 1.0 / sqrt(max(varSum / Float(gLen), 1e-12))
                    for i in 0..<gLen {
                        normVals.append((eq[i] - mean) * invStd)
                    }
                    if groupSamples.count < 256 {
                        groupSamples.append(Array(samples[base..<(base + gLen)]))
                    }
                }
            }

            let c = fitSlowEMA1D(points: normVals, k: k, epochs: epochs, seed: seed &+ UInt64(h * 977))
            allCentroids.append(c)
            clipRatios.append(fitOptimalAffineClipRatio(groups: groupSamples))
        }

        return QWKVSlowEMACodebook(
            bits: bits,
            kvHeads: kvHeads,
            headDim: headDim,
            channelScales: allChanScales,
            centroids: allCentroids,
            affineClipRatios: clipRatios
        )
    }

    /// Quantizes one KV token into standard `QWKVCompressedLayout` Q4 `(qOut, scalesOut, minsOut)`
    /// using calibrated `slow-EMA` shrinkage + multi-start coordinate polish (zero layout change).
    public static func quantizeQ4Calibrated(
        input: [Float],
        codebook: QWKVSlowEMACodebook,
        qOut: inout [UInt8],
        scalesOut: inout [Float16],
        minsOut: inout [Float16]
    ) throws {
        let kvHeads = codebook.kvHeads
        let headDim = codebook.headDim
        let total = kvHeads * headDim
        let groupsPerHead = max(1, (headDim + 127) / 128)
        let totalGroups = kvHeads * groupsPerHead
        guard headDim % 2 == 0,
              input.count == total, qOut.count == total / 2,
              scalesOut.count == totalGroups, minsOut.count == totalGroups else {
            throw QWKVError.badArgument("calibrated q4 buffer shapes mismatch")
        }

        let ratiosBase = codebook.affineClipRatios
        for h in 0..<kvHeads {
            let rCal = ratiosBase[h]
            let trialRatios: [Float] = [min(0.45, rCal * 0.6), rCal * 0.8, rCal, 1.0]
            for g in 0..<groupsPerHead {
                let gIdx = h * groupsPerHead + g
                let gStart = g * 128
                let gLen = min(128, headDim - gStart)
                let inOff = h * headDim + gStart
                let outByteOff = inOff / 2

                var minV = input[inOff]
                var maxV = input[inOff]
                var meanV: Float = 0
                for i in 0..<gLen {
                    let v = input[inOff + i]
                    if v < minV { minV = v }
                    if v > maxV { maxV = v }
                    meanV += v
                }
                meanV /= Float(gLen)

                var bestS16 = Float16(max((maxV - minV) / 15.0, 1e-8))
                var bestM16 = Float16(minV)
                var bestErr = Float.infinity

                for r in trialRatios {
                    var curMin = meanV - (meanV - minV) * r
                    var curScale = max(((meanV + (maxV - meanV) * r) - curMin) / 15.0, 1e-8)
                    for _ in 0..<2 {
                        var sumX: Float = 0, sumQ: Float = 0, sumQQ: Float = 0, sumXQ: Float = 0
                        let inv0 = 1.0 / curScale
                        for i in 0..<gLen {
                            let x = input[inOff + i]
                            let q = max(0.0, min(15.0, round((x - curMin) * inv0)))
                            sumX += x; sumQ += q; sumQQ += q * q; sumXQ += x * q
                        }
                        let nF = Float(gLen)
                        let det = nF * sumQQ - sumQ * sumQ
                        if det > 1e-6 {
                            let fitScale = (nF * sumXQ - sumQ * sumX) / det
                            let fitMin = (sumX - fitScale * sumQ) / nF
                            if fitScale > 1e-8 { curScale = fitScale; curMin = fitMin }
                        }
                    }
                    let candS16 = Float16(max(curScale, 1e-8))
                    let candM16 = Float16(curMin)
                    let sc = max(Float(candS16), 1e-8)
                    let mn = Float(candM16)
                    let inv = 1.0 / sc
                    var err: Float = 0
                    for i in 0..<gLen {
                        let x = input[inOff + i]
                        let q = max(0.0, min(15.0, round((x - mn) * inv)))
                        let d = (q * sc + mn) - x
                        err += d * d
                    }
                    if err < bestErr {
                        bestErr = err
                        bestS16 = candS16
                        bestM16 = candM16
                    }
                }

                scalesOut[gIdx] = bestS16
                minsOut[gIdx] = bestM16

                let invS = 1.0 / max(Float(bestS16), 1e-8)
                let baseM = Float(bestM16)
                for i in 0..<(gLen / 2) {
                    let v0 = input[inOff + 2 * i]
                    let v1 = input[inOff + 2 * i + 1]
                    let q0 = UInt8(max(0.0, min(15.0, round((v0 - baseM) * invS))))
                    let q1 = UInt8(max(0.0, min(15.0, round((v1 - baseM) * invS))))
                    qOut[outByteOff + i] = q0 | (q1 << 4)
                }
            }
        }
    }

    /// Quantizes one KV token using channel-equalized non-uniform `slow-EMA` 3-bit or 4-bit LUTs.
    public static func quantizeNonUniformLUT(
        input: [Float],
        codebook: QWKVSlowEMACodebook,
        codesOut: inout [UInt8],
        meansOut: inout [Float16],
        stdsOut: inout [Float16]
    ) throws {
        let kvHeads = codebook.kvHeads
        let headDim = codebook.headDim
        let groupsPerHead = max(1, (headDim + 127) / 128)
        let totalGroups = kvHeads * groupsPerHead
        guard input.count == kvHeads * headDim,
              codesOut.count == input.count,
              meansOut.count == totalGroups,
              stdsOut.count == totalGroups else {
            throw QWKVError.badArgument("non-uniform LUT KV buffer shapes mismatch")
        }
        for h in 0..<kvHeads {
            let lut = codebook.centroids[h]
            let chanScale = codebook.channelScales[h]
            for g in 0..<groupsPerHead {
                let gIdx = h * groupsPerHead + g
                let gStart = g * 128
                let gLen = min(128, headDim - gStart)
                let base = h * headDim + gStart
                var eq = [Float](repeating: 0, count: gLen)
                var mean: Float = 0
                for i in 0..<gLen {
                    let v = input[base + i] / chanScale[gStart + i]
                    eq[i] = v
                    mean += v
                }
                mean /= Float(gLen)
                var varSum: Float = 0
                for i in 0..<gLen {
                    let d = eq[i] - mean
                    varSum += d * d
                }
                let std = max(sqrt(varSum / Float(gLen)), 1e-6)
                let m16 = Float16(mean)
                let s16 = Float16(std)
                meansOut[gIdx] = m16
                stdsOut[gIdx] = s16
                let mF = Float(m16)
                let invS = 1.0 / max(Float(s16), 1e-6)
                for i in 0..<gLen {
                    let u = (eq[i] - mF) * invS
                    codesOut[base + i] = nearestSortedCode(lut, value: u)
                }
            }
        }
    }

    /// Dequantizes a channel-equalized non-uniform `slow-EMA` LUT-encoded KV token.
    public static func dequantizeNonUniformLUT(
        codesIn: [UInt8],
        meansIn: [Float16],
        stdsIn: [Float16],
        codebook: QWKVSlowEMACodebook,
        out: inout [Float]
    ) throws {
        let kvHeads = codebook.kvHeads
        let headDim = codebook.headDim
        let groupsPerHead = max(1, (headDim + 127) / 128)
        let totalGroups = kvHeads * groupsPerHead
        guard codesIn.count == kvHeads * headDim,
              out.count == codesIn.count,
              meansIn.count == totalGroups,
              stdsIn.count == totalGroups else {
            throw QWKVError.badArgument("non-uniform LUT KV dequant shapes mismatch")
        }
        for h in 0..<kvHeads {
            let lut = codebook.centroids[h]
            let chanScale = codebook.channelScales[h]
            for g in 0..<groupsPerHead {
                let gIdx = h * groupsPerHead + g
                let gStart = g * 128
                let gLen = min(128, headDim - gStart)
                let base = h * headDim + gStart
                let mF = Float(meansIn[gIdx])
                let sF = Float(stdsIn[gIdx])
                for i in 0..<gLen {
                    let c = Int(codesIn[base + i])
                    let eq = mF + lut[min(lut.count - 1, max(0, c))] * sF
                    out[base + i] = eq * chanScale[gStart + i]
                }
            }
        }
    }

    // MARK: - Private `arXiv:2611.09501` `slow-EMA` Trainer

    private static func nearestSortedCode(_ lut: [Float], value: Float) -> UInt8 {
        var lo = 0
        var hi = lut.count - 1
        while lo < hi {
            let mid = (lo + hi) >> 1
            let thresh = 0.5 * (lut[mid] + lut[mid + 1])
            if value > thresh { lo = mid + 1 } else { hi = mid }
        }
        return UInt8(lo)
    }

    private static func fitSlowEMA1D(points: [Float], k: Int, epochs: Double, seed: UInt64) -> [Float] {
        let n = points.count
        let rawB = 0.035 * pow(Double(k), 0.13) * pow(Double(n), 0.88) * pow(max(epochs, 0.25), 0.81)
        let log2B = round(log2(max(Double(k), rawB)))
        let batchSize = min(n, max(k, 1 << Int(max(4.0, min(20.0, log2B)))))
        let alpha0 = Float(min(1.0, 0.824 * pow(Double(k), -0.46) * pow(Double(batchSize), 0.90)
                               * pow(Double(n), -0.45) * pow(max(epochs, 0.25), -0.11)))
        let totalSteps = max(4, Int(ceil(epochs * Double(n) / Double(batchSize))))

        var rng = seed == 0 ? 0x9E37_79B9_7F4A_7C15 : seed
        func nextIndex(_ bound: Int) -> Int {
            rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27
            return Int((rng &* 0x2545_F491_4F6C_DD1D) % UInt64(bound))
        }

        var pseudoN = [Float](repeating: 1e-10, count: k)
        var pseudoS = [Float](repeating: 0, count: k)
        for j in 0..<k {
            pseudoS[j] = points[nextIndex(n)] * 1e-10
        }

        var batchCounts = [Float](repeating: 0, count: k)
        var batchSums = [Float](repeating: 0, count: k)
        var centroids = [Float](repeating: 0, count: k)

        for step in 0..<totalSteps {
            for j in 0..<k { centroids[j] = pseudoS[j] / max(pseudoN[j], 1e-20) }
            let order = (0..<k).sorted { centroids[$0] < centroids[$1] }
            pseudoN = order.map { pseudoN[$0] }
            pseudoS = order.map { pseudoS[$0] }
            for j in 0..<k { centroids[j] = pseudoS[j] / max(pseudoN[j], 1e-20) }

            for j in 0..<k { batchCounts[j] = 0; batchSums[j] = 0 }
            for _ in 0..<batchSize {
                let x = points[nextIndex(n)]
                let idx = Int(nearestSortedCode(centroids, value: x))
                batchCounts[idx] += 1.0
                batchSums[idx] += x
            }

            let progress = Double(step) / Double(max(1, totalSteps - 1))
            let alpha = alpha0 * Float(0.1 + 0.9 * 0.5 * (1.0 + cos(Double.pi * progress)))
            let gamma = 1.0 - alpha
            for j in 0..<k {
                pseudoN[j] = gamma * pseudoN[j] + alpha * batchCounts[j]
                pseudoS[j] = gamma * pseudoS[j] + alpha * batchSums[j]
            }

            if (step + 1) % 5 == 0 && step + 1 < totalSteps {
                let meanN = pseudoN.reduce(0, +) / Float(k)
                for j in 0..<k where pseudoN[j] < 0.1 * meanN {
                    if let donor = (0..<k).max(by: { pseudoN[$0] < pseudoN[$1] }), donor != j {
                        let cDonor = pseudoS[donor] / max(pseudoN[donor], 1e-20)
                        let halfN = 0.5 * pseudoN[donor]
                        pseudoN[donor] = halfN
                        pseudoN[j] = halfN
                        pseudoS[donor] = (cDonor - 1e-4) * halfN
                        pseudoS[j] = (cDonor + 1e-4) * halfN
                    }
                }
            }
        }

        return (0..<k).map { pseudoS[$0] / max(pseudoN[$0], 1e-20) }.sorted()
    }

    private static func fitOptimalAffineClipRatio(groups: [[Float]]) -> Float {
        guard !groups.isEmpty else { return 1.0 }
        let grid: [Float] = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00]
        var bestRatio: Float = 1.0
        var bestMSE = Float.infinity
        for r in grid {
            var errSum: Float = 0
            for grp in groups {
                guard let minV = grp.min(), let maxV = grp.max() else { continue }
                let meanV = grp.reduce(0, +) / Float(grp.count)
                let lo = meanV - (meanV - minV) * r
                let hi = meanV + (maxV - meanV) * r
                let sc = max(Float(Float16(max((hi - lo) / 15.0, 1e-8))), 1e-8)
                let mn = Float(Float16(lo))
                let inv = 1.0 / sc
                for x in grp {
                    let q = max(0.0, min(15.0, round((x - mn) * inv)))
                    let d = (q * sc + mn) - x
                    errSum += d * d
                }
            }
            if errSum < bestMSE {
                bestMSE = errSum
                bestRatio = r
            }
        }
        return bestRatio
    }
}
