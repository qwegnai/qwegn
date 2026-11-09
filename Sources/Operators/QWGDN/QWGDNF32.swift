// QWGDNF32.swift — GatedDeltaNet prologue, depthwise causal conv, and single/sequence recurrence encoders in f32.
import Metal
import QWAlloc
import QWGEMM
import QWHAL

public enum QWGDNF32 {
    /// Fused T <= 8 depthwise conv + SiLU + per-head q/k L2 norm + glog/beta in f32.
    public static func encodeConvPrepF32(_ enc: any MTLComputeCommandEncoder, hal: QWHal, x: QWTensorValue,
                                         w: QWTensorValue, stIn: QWTensorValue, stOut: QWTensorValue,
                                         stC: QWTensorValue? = nil, stD: QWTensorValue? = nil,
                                         b: QWTensorValue, a: QWTensorValue, aLog: QWTensorValue,
                                         dtBias: QWTensorValue, y: QWTensorValue, glog: QWTensorValue,
                                         beta: QWTensorValue, rows: Int = 1, xStride: Int = 10240,
                                         baStride: Int = 48,
                                         ringBase: QWTensorValue? = nil, ringParity: Int = 0) throws {
        enc.setComputePipelineState(try hal.pipeline("qwg_gdn_conv_prep_f32"))
        enc.bindTensors(x, w, stIn, stOut, b, a, aLog, dtBias, y, glog, beta, startIndex: 0)
        let tPacked = ringBase != nil ? (0x8000 | ((ringParity & 7) << 8) | rows) : rows
        enc.setUInt32s(tPacked, xStride, baStride, startIndex: 11)
        (ringBase ?? stC ?? stOut).bind(enc, index: 14)
        (stD ?? stIn).bind(enc, index: 15)
        enc.dispatchThreadgroups(MTLSize(width: 64, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    /// Fused multi-token depthwise conv + SiLU + per-head q/k L2 norm + glog/beta in f32.
    public static func encodeConvPrepSeqF32(_ enc: any MTLComputeCommandEncoder, hal: QWHal, x: QWTensorValue,
                                            w: QWTensorValue, stIn: QWTensorValue, stOut: QWTensorValue,
                                            b: QWTensorValue, a: QWTensorValue, aLog: QWTensorValue,
                                            dtBias: QWTensorValue, y: QWTensorValue, glog: QWTensorValue,
                                            beta: QWTensorValue, rows: Int, xStride: Int = 10240) throws {
        enc.setComputePipelineState(try hal.pipeline("qwg_gdn_conv_prep_seq_f32"))
        enc.bindTensors(x, w, stIn, stOut, b, a, aLog, dtBias, y, glog, beta, startIndex: 0)
        enc.setUInt32s(rows, xStride, startIndex: 11)
        enc.dispatchThreadgroups(MTLSize(width: 64, height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    /// Recurrent step on f32 y with fused sigmoid-gated RMSNorm and H_32 rotation.
    public static func encodeRecurrentF32(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                          y: QWTensorValue, betas: QWTensorValue, glog: QWTensorValue,
                                          state: QWTensorValue, z: QWTensorValue,
                                          normWeight: QWTensorValue, out: QWTensorValue) throws {
        enc.setComputePipelineState(try hal.pipeline("qwg_gdn_recurrent_f32"))
        enc.bindTensors(y, betas, glog, state, z, normWeight, out, startIndex: 0)
        enc.dispatchThreadgroups(MTLSize(width: 48, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
    }

    /// Verify rows: R tokens through the T = 1 recurrence in one dispatch (state carried in registers).
    /// Row t reads y[t * 10240], betas/glog[t * 48], z[t * zStride] and writes out[t * 6144].
    public static func encodeRecurrentRowsF32(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                              y: QWTensorValue, betas: QWTensorValue, glog: QWTensorValue,
                                              state: QWTensorValue, stateRow0: QWTensorValue? = nil,
                                              stateRow1: QWTensorValue? = nil, stateRow2: QWTensorValue? = nil,
                                              z: QWTensorValue, zStride: Int,
                                              normWeight: QWTensorValue, out: QWTensorValue, rows: Int,
                                              ringBase: QWTensorValue? = nil, ringParity: Int = 0) throws {
        if rows == 1 {
            try encodeRecurrentF32(enc, hal: hal, y: y, betas: betas, glog: glog, state: state, z: z,
                                   normWeight: normWeight, out: out)
            return
        }
        enc.setComputePipelineState(try hal.pipeline("qwg_gdn_recurrent_rows_f32"))
        enc.bindTensors(y, betas, glog, state, z, normWeight, out, startIndex: 0)
        let rPacked = ringBase != nil ? (0x8000 | ((ringParity & 7) << 8) | rows) : rows
        enc.setUInt32s(rPacked, startIndex: 7)
        enc.setUInt32s(zStride, startIndex: 8)
        if let base = ringBase {
            (stateRow0 ?? state).bind(enc, index: 9)
            base.bind(enc, index: 10)
            (stateRow2 ?? state).bind(enc, index: 11)
        } else if let s1 = stateRow1 {
            (stateRow0 ?? state).bind(enc, index: 9)
            s1.bind(enc, index: 10)
            (stateRow2 ?? state).bind(enc, index: 11)
        } else {
            let finalOut = stateRow0 ?? state
            state.bind(enc, index: 9)
            finalOut.bind(enc, index: 10)
            finalOut.bind(enc, index: 11)
        }
        enc.dispatchThreadgroups(MTLSize(width: 48, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
    }

    /// Register-resident multi-token GDN recurrence (192 TGs x 128 threads, zero barriers) + in-place norm/sigmoid/H_32.
    public static func encodeRecurrentSeqF32(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                             y: QWTensorValue, betas: QWTensorValue, glog: QWTensorValue,
                                             state: QWTensorValue, z: QWTensorValue, normWeight: QWTensorValue,
                                             out: QWTensorValue, tokens: Int, convDim: Int, vHeads: Int,
                                             zStride: Int? = nil, out16: QWTensorValue? = nil) throws {
        enc.setComputePipelineState(try hal.pipeline("qwg_gdn_recurrent_seq_f32"))
        enc.bindTensors(y, betas, glog, state, out, startIndex: 0)
        enc.setUInt32s(tokens, convDim, vHeads, startIndex: 5)
        enc.dispatchThreadgroups(MTLSize(width: 4, height: vHeads, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc.memoryBarrier(scope: .buffers)
        enc.setComputePipelineState(try hal.pipeline("qwg_gdn_norm_sigmoid_f32"))
        enc.bindTensors(out, z, normWeight, startIndex: 0)
        enc.setUInt32s(vHeads, zStride ?? (vHeads * 128), startIndex: 3)
        (out16 ?? out).bind(enc, index: 5)
        enc.setUInt32s(out16 != nil ? 1 : 0, startIndex: 6)
        enc.dispatchThreadgroups(MTLSize(width: vHeads, height: tokens, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }
}
