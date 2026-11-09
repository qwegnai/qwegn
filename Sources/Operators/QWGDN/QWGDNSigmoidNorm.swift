// QWGDNSigmoidNorm.swift — GDN epilogue host dispatch (Sources/QWMetal/qwgdn_sigmoid_norm.metal):
// gated RMSNorm with a sigmoid gate and 32-point Walsh-Hadamard rotation.
import Metal
import QWAlloc
import QWHAL

public enum QWGDNSigmoidNorm {
    public static let vHeads = 48
    public static let headDim = 128

    public static func encode(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                              oRaw: QWTensorValue, z: QWTensorValue, weight: QWTensorValue,
                              out: QWTensorValue) throws {
        let pso = try hal.pipeline("qwg_gdn_norm_sigmoid")
        enc.setComputePipelineState(pso)
        oRaw.bind(enc, index: 0)
        z.bind(enc, index: 1)
        weight.bind(enc, index: 2)
        out.bind(enc, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: vHeads, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: headDim, height: 1, depth: 1))
    }
}
