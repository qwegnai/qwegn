// FNGPURouterBF16.swift — The hybrid path's bf16 router dispatch.
//
// The standard path's router reads an f32 plane (Sources/QWMetal/subsystems/
// qwrouter.metal's qwg_router_gemm), so this one is the hybrid path's alone.
import Metal
import QWAlloc
import QWGEMM
import QWHAL

extension QWGemm {
    /// Decode router logits + shared-expert sigmoid gate from BF16 weights [experts + 1, k] (4 rows per TG, 1 SG/row).
    public static func encodeFnRouterBF16Decode(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                                weight: QWTensorValue, x: QWTensorValue,
                                                logits: QWTensorValue, sg: QWTensorValue,
                                                rows: Int, experts: Int, k: Int) throws {
        enc.setComputePipelineState(try hal.pipeline("qwg_moe_router_bf16_f32"))
        enc.bindTensors(weight, x, logits, sg, startIndex: 0)
        enc.setUInt32s(experts, k, startIndex: 4)
        enc.dispatchThreadgroups(MTLSize(width: (experts + 4) / 4, height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }
}
