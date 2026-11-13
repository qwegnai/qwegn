import Metal
import QWAlloc
import QWHAL

// The sparse-MoE router dispatch (Sources/QWMetal/subsystems/qwrouter.metal).
// It lives in the operator target because two engines route: the hybrid one
// and the standard path's sparse layer.
extension QWGemm {
    /// idx i32 [rows, topK] and wts f32 [rows, topK] from logits f32 [rows, experts] (rank order, ties to lower index).
    public static func encodeFnMoeRoute(_ enc: any MTLComputeCommandEncoder, hal: QWHal, logits: QWTensorValue,
                                        idx: QWTensorValue, wts: QWTensorValue, rows: Int, experts: Int,
                                        topK: Int, normTopK: Bool,
                                        guSh: QWTensorValue? = nil, actSh: QWTensorValue? = nil,
                                        sharedIntermediate: Int = 0,
                                        bias: QWTensorValue? = nil,
                                        expertBudget: Int = 0,
                                        splitFeature: Bool = false,
                                        routedScaleParam: QWTensorValue? = nil) throws {
        precondition(experts % 32 == 0 && experts <= 512 && topK <= 16, "route needs E % 32 == 0, E <= 512, K <= 16")
        let hasShared = guSh != nil && actSh != nil && sharedIntermediate > 0
        let normFlags = (normTopK ? 1 : 0)
            | (bias != nil ? 2 : 0)
            | (splitFeature ? 8 : 0)
            | (routedScaleParam != nil ? 16 : 0)
        if rows == 1 || hasShared {
            enc.setComputePipelineState(try hal.pipeline("qwg_moe_route_decode"))
            enc.bindTensors(logits, idx, wts, startIndex: 0)
            enc.setUInt32s(experts, topK, normFlags, startIndex: 3)
            (bias ?? logits).bind(enc, index: 9)
            (routedScaleParam ?? logits).bind(enc, index: 10)
            if let gu = guSh, let act = actSh, sharedIntermediate > 0 {
                enc.bindTensors(gu, act, startIndex: 6)
                enc.setUInt32s(sharedIntermediate, startIndex: 8)
                enc.dispatchThreadgroups(MTLSize(width: 2, height: rows, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
            } else {
                enc.bindTensors(logits, logits, startIndex: 6)
                enc.setUInt32s(0, startIndex: 8)
                enc.dispatchThreadgroups(MTLSize(width: 1, height: rows, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
            }
        } else {
            enc.setComputePipelineState(try hal.pipeline("qwg_moe_route"))
            enc.bindTensors(logits, idx, wts, startIndex: 0)
            enc.setUInt32s(experts, topK, normFlags, startIndex: 3)
            (bias ?? logits).bind(enc, index: 6)
            (routedScaleParam ?? logits).bind(enc, index: 7)
            enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        }
        if expertBudget > 0 && rows > 1 && rows <= 4 && expertBudget < rows * topK && rows * topK <= 64 {
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(try hal.pipeline("qwg_moe_route_decode"))
            enc.bindTensors(logits, idx, wts, startIndex: 0)
            enc.setUInt32s(rows, topK, normFlags | 4, startIndex: 3)
            enc.bindTensors(logits, logits, startIndex: 6)
            enc.setUInt32s(max(topK, expertBudget), startIndex: 8)
            (bias ?? logits).bind(enc, index: 9)
            (routedScaleParam ?? logits).bind(enc, index: 10)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        }
    }

    /// Router logits for T rows: y[t, e] = x[t] . W[e], f32 or bf16 weights and f32 activations, tiled over tokens and experts.
    public static func encodeFnRouterGemm(_ enc: any MTLComputeCommandEncoder, hal: QWHal, weight: QWTensorValue,
                                          x: QWTensorValue, y: QWTensorValue, rows: Int, n: Int, k: Int,
                                          shGateWeight: QWTensorValue? = nil,
                                          shGateOut: QWTensorValue? = nil) throws {
        precondition(k % 32 == 0, "router GEMM needs K % 32 == 0")
        let hasShGate = (shGateWeight != nil && shGateOut != nil) ? 1 : 0
        // NAX path: bf16 W exact, f32 X as three bf16 terms; ~3x the simdgroup kernel at prefill sizes.
        if (weight.shape.dtype == .bf16 || weight.shape.dtype == .f16) && n % 64 == 0 && rows >= 128 {
            enc.setComputePipelineState(try hal.pipeline("qwg_router_nax"))
            enc.bindTensors(weight, x, y, startIndex: 0)
            enc.setUInt32s(rows, n, k, startIndex: 3)
            (shGateWeight ?? weight).bind(enc, index: 7)
            (shGateOut ?? y).bind(enc, index: 8)
            enc.setUInt32s(hasShGate, startIndex: 9)
            enc.dispatchThreadgroups(MTLSize(width: n / 64 + hasShGate, height: (rows + 63) / 64, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            return
        }
        enc.setComputePipelineState(try hal.pipeline("qwg_router_gemm"))
        enc.bindTensors(weight, x, y, startIndex: 0)
        let isBF16 = (weight.shape.dtype == .f16 || weight.shape.dtype == .bf16) ? 1 : 0
        enc.setUInt32s(rows, n, k, isBF16, startIndex: 3)
        (shGateWeight ?? weight).bind(enc, index: 7)
        (shGateOut ?? y).bind(enc, index: 8)
        enc.setUInt32s(hasShGate, startIndex: 9)
        if rows <= 4 {
            let tpg = 64
            let sgs = n + hasShGate
            let sgPerTG = tpg / 32
            enc.dispatchThreadgroups(MTLSize(width: (sgs + sgPerTG - 1) / sgPerTG, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1))
            return
        }
        enc.dispatchThreadgroups(MTLSize(width: (n + 31) / 32, height: (rows + 31) / 32, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
}
