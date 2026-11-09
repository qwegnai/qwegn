import Metal
import QWAlloc
import QWGEMM
import QWHAL

// FlashNext per-layer embedding dispatches (Sources/QWMetal/qwple.metal).
extension QWGemm {
    /// gv f32 [T, W] = sigmoid(sign(g) sqrt(|g|)) * vals, g = <keyNorm, queryNorm> per (token, stream) / sqrt(H).
    public static func encodeFnPLEGate(_ enc: any MTLComputeCommandEncoder, hal: QWHal, keyNorm: QWTensorValue,
                                       queryNorm: QWTensorValue, vals: QWTensorValue, gv: QWTensorValue,
                                       rows: Int, hidden: Int, hc: Int, valStride: Int? = nil) throws {
        if let valStride { precondition(valStride >= hidden) }
        enc.setComputePipelineState(try hal.pipeline(valStride == nil ? "qwg_ple_gate" : "qwg_ple_gate_strided"))
        enc.bindTensors(keyNorm, queryNorm, vals, gv, startIndex: 0)
        enc.setUInt32s(hidden, hc, startIndex: 4)
        if let valStride { enc.setUInt32s(valStride, startIndex: 6) }
        enc.dispatchThreadgroups(MTLSize(width: rows * hc, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    /// out f32 [T, W] = gv + SiLU(causal dilated depthwise conv over gvNorm).
    public static func encodeFnPLEConv(_ enc: any MTLComputeCommandEncoder, hal: QWHal, gv: QWTensorValue,
                                       gvNorm: QWTensorValue, convWeight: QWTensorValue, out: QWTensorValue,
                                       rows: Int, width: Int, kernel: Int, dilation: Int,
                                       hist: QWTensorValue? = nil) throws {
        enc.setComputePipelineState(try hal.pipeline("qwg_ple_conv"))
        enc.bindTensors(gv, gvNorm, convWeight, out, startIndex: 0)
        enc.setUInt32s(width, kernel, dilation, rows, startIndex: 4)
        // No history: bind any buffer; HL = 0 means the kernel never reads it.
        (hist ?? gv).bind(enc, index: 8)
        enc.setUInt32s(hist == nil ? 0 : (kernel - 1) * dilation, startIndex: 9)
        enc.dispatchThreads(MTLSize(width: width, height: rows, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    }

    /// Roll the PLE conv history: newHist = last HL rows of [oldHist | gvNorm]. HL = (K-1)*dilation.
    public static func encodeFnPLEHistory(_ enc: any MTLComputeCommandEncoder, hal: QWHal, oldHist: QWTensorValue,
                                          gvNorm: QWTensorValue, newHist: QWTensorValue, width: Int, historyRows: Int, rows: Int) throws {
        enc.setComputePipelineState(try hal.pipeline("qwg_ple_hist"))
        enc.bindTensors(oldHist, gvNorm, newHist, startIndex: 0)
        enc.setUInt32s(width, historyRows, rows, startIndex: 3)
        enc.dispatchThreads(MTLSize(width: width, height: historyRows, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    }
}

extension QWGemm {
    /// Residual injection into the HC streams: out[t, c, :] = base[t, c, :] + src[t, :] * inj[t, c].
    public static func encodeFnInject(_ enc: any MTLComputeCommandEncoder, hal: QWHal, base: QWTensorValue,
                                      src: QWTensorValue, inj: QWTensorValue, out: QWTensorValue,
                                      rows: Int, hidden: Int, hc: Int) throws {
        enc.setComputePipelineState(try hal.pipeline("qwg_inject"))
        enc.bindTensors(base, src, inj, out, startIndex: 0)
        enc.setUInt32s(hidden, hc, startIndex: 4)
        enc.dispatchThreads(MTLSize(width: hc * hidden, height: rows, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
}
