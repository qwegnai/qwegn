import Foundation

// Partial rotary embedding, half-split pairing (HF apply_rotary_pos_emb: rotate_half over the first
// `rotaryDim` dims; the remaining dims pass through).

public struct FNRope: Sendable {
    public let rotaryDim: Int
    public let theta: Double
    public let yarnFactor: Float?
    public let originalMaxPosition: Int
    public let betaFast: Float
    public let betaSlow: Float
    public let truncate: Bool

    public var attentionScaling: Float {
        guard let f = yarnFactor, f > 1.0 else { return 1.0 }
        return Float(0.1 * Foundation.log(Double(f)) + 1.0)
    }

    public init(rotaryDim: Int, theta: Double,
                yarnFactor: Float? = nil,
                originalMaxPosition: Int = 262_144,
                betaFast: Float = 32,
                betaSlow: Float = 1,
                truncate: Bool = true) {
        precondition(rotaryDim % 2 == 0)
        self.rotaryDim = rotaryDim
        self.theta = theta
        self.yarnFactor = yarnFactor
        self.originalMaxPosition = originalMaxPosition
        self.betaFast = betaFast
        self.betaSlow = betaSlow
        self.truncate = truncate
    }

    private func inverseFrequencies() -> [Double] {
        let half = rotaryDim / 2
        let rDim = rotaryDim
        guard let factor = yarnFactor, factor > 1.0 else {
            return (0..<half).map { 1 / pow(theta, Double(2 * $0) / Double(rDim)) }
        }
        let dim = Float(rDim)
        let base = Float(theta)
        let posFreqs = (0..<half).map { pow(base, Float($0) * 2.0 / dim) }
        let invEx = posFreqs.map { 1.0 / $0 }
        let invIn = posFreqs.map { 1.0 / (factor * $0) }
        func fcd(_ nr: Double) -> Double {
            (Double(rDim) * Foundation.log(Double(originalMaxPosition) / (nr * 2.0 * Double.pi)))
                / (2.0 * Foundation.log(theta))
        }
        var lo = fcd(Double(betaFast)), hi = fcd(Double(betaSlow))
        if truncate { lo = Foundation.floor(lo); hi = Foundation.ceil(hi) }
        let low = max(lo, 0.0)
        var high = min(hi, Double(rDim - 1))
        if low == high { high += 0.001 }
        return (0..<half).map { i in
            let ramp = min(max((Float(i) - Float(low)) / Float(high - low), 0), 1)
            let eef = 1 - ramp
            return Double(invIn[i] * (1 - eef) + invEx[i] * eef)
        }
    }

    /// cos/sin tables for `0..<count` scaled by `stride` (`pos = p * stride`).
    public func tables(count: Int, stride: Int = 1) -> (cos: [Float], sin: [Float]) {
        let half = rotaryDim / 2
        let rDim = rotaryDim
        let inv = inverseFrequencies()
        let scale = attentionScaling
        var c = [Float](repeating: 0, count: count * rDim)
        var s = [Float](repeating: 0, count: count * rDim)
        guard count > 0 else { return (c, s) }
        struct PtrBox: @unchecked Sendable {
            let cp: UnsafeMutablePointer<Float>
            let sp: UnsafeMutablePointer<Float>
        }
        let box = c.withUnsafeMutableBufferPointer { cb in
            s.withUnsafeMutableBufferPointer { sb in
                PtrBox(cp: cb.baseAddress!, sp: sb.baseAddress!)
            }
        }
        let chunks = max(1, min(count, ProcessInfo.processInfo.activeProcessorCount))
        let per = (count + chunks - 1) / chunks
        DispatchQueue.concurrentPerform(iterations: chunks) { ch in
            let lo = ch * per, hi = min(count, lo + per)
            guard lo < hi else { return }
            for p in lo..<hi {
                let pos = Double(p * stride)
                let base = p * rDim
                for i in 0..<half {
                    let angle = pos * inv[i]
                    let cv = Float(cos(angle)) * scale, sv = Float(sin(angle)) * scale
                    box.cp[base + i] = cv; box.cp[base + i + half] = cv
                    box.sp[base + i] = sv; box.sp[base + i + half] = sv
                }
            }
        }
        return (c, s)
    }

    /// cos/sin tables for absolute positions; each row has `rotaryDim` entries (halves duplicated).
    public func tables(positions: [Int]) -> (cos: [Float], sin: [Float]) {
        let half = rotaryDim / 2
        let inv = inverseFrequencies()
        let scale = attentionScaling
        var c = [Float](repeating: 0, count: positions.count * rotaryDim)
        var s = c
        for (p, pos) in positions.enumerated() {
            for i in 0..<half {
                let angle = Double(pos) * inv[i]
                let cv = Float(cos(angle)) * scale, sv = Float(sin(angle)) * scale
                c[p * rotaryDim + i] = cv; c[p * rotaryDim + i + half] = cv
                s[p * rotaryDim + i] = sv; s[p * rotaryDim + i + half] = sv
            }
        }
        return (c, s)
    }

    /// In-place rotation of one head vector (length `dim`) with the row `table` (position index `p`).
    public func apply(_ v: inout [Float], base: Int, dim: Int, cos c: [Float], sin s: [Float], row p: Int) {
        let half = rotaryDim / 2
        let x = Array(v[base..<(base + dim)])
        for i in 0..<rotaryDim {
            let rot = i < half ? -x[i + half] : x[i - half]
            v[base + i] = x[i] * c[p * rotaryDim + i] + rot * s[p * rotaryDim + i]
        }
    }
}
