import Foundation

// FlashNext CPU primitives (fp32 reference). Shapes are row-major; nn.Linear weights are [out, in].

public enum FNMath {
    @inline(__always) public static func silu(_ x: Float) -> Float { x / (1 + expf(-x)) }
    @inline(__always) public static func sigmoid(_ x: Float) -> Float { 1 / (1 + expf(-x))  }

    /// y[o] = sum_i W[o*in + i] * x[i]  (nn.Linear, bias-free).
    public static func linear(_ x: ArraySlice<Float>, weight: [Float], out: Int, inDim: Int) -> [Float] {
        var y = [Float](repeating: 0, count: out)
        x.withUnsafeBufferPointer { xb in
            weight.withUnsafeBufferPointer { wb in
                for o in 0..<out {
                    var acc: Float = 0
                    let row = o * inDim
                    for i in 0..<inDim { acc += wb[row + i] * xb[i] }
                    y[o] = acc
                }
            }
        }
        return y
    }

    /// Qwen4ExpTextRMSNorm: normalize each `group`-wide slice independently; scale is (1 + weight).
    public static func groupRMSNorm(_ x: [Float], rows: Int, width: Int, group: Int,
                                    weight: [Float], eps: Float) -> [Float] {
        precondition(width % group == 0 && weight.count == width)
        var out = [Float](repeating: 0, count: x.count)
        for r in 0..<rows {
            let base = r * width
            for g in stride(from: 0, to: width, by: group) {
                var ss: Float = 0
                for k in 0..<group { let v = x[base + g + k]; ss += v * v }
                let inv = 1 / (ss / Float(group) + eps).squareRoot()
                for k in 0..<group {
                    out[base + g + k] = x[base + g + k] * inv * (1 + weight[g + k])
                }
            }
        }
        return out
    }
}
