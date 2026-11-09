import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc

/// Flags for fused 4-branch gated residual kernel (spec-23).
public struct QWGatedResidualFlags: OptionSet, Sendable {
    public let rawValue: UInt32
    public init(rawValue: UInt32) { self.rawValue = rawValue }

    public static let hasGDN         = QWGatedResidualFlags(rawValue: 1 << 0)
    public static let hasAttn        = QWGatedResidualFlags(rawValue: 1 << 1)
    public static let hasMLP         = QWGatedResidualFlags(rawValue: 1 << 2)
    public static let hasPLE         = QWGatedResidualFlags(rawValue: 1 << 3)
    public static let hasGate        = QWGatedResidualFlags(rawValue: 1 << 4)
    public static let pleSigmoid     = QWGatedResidualFlags(rawValue: 1 << 5)
    public static let gammaPlusOne   = QWGatedResidualFlags(rawValue: 1 << 6)
}

public enum QWGatedResidualError: Error, CustomStringConvertible {
    case badShape(String)
    case missingBuffer(String)
    public var description: String {
        switch self {
        case .badShape(let s): return "QWGatedResidual: bad shape: \(s)"
        case .missingBuffer(let s): return "QWGatedResidual: missing buffer: \(s)"
        }
    }
}

/// Fused 4-branch gated residual accumulator & RMSNorm epilogue.
/// Fuses linear recurrent GDN, full attention, SwiGLU MLP,
/// into a single register epilogue adhering strictly to <= 32 KiB static threadgroup memory.
public enum QWGatedResidual {

    /// Encode the fused 4-branch gated residual into command encoder.
    /// If `normOut` and `gamma` are provided, computes fused residual + RMSNorm epilogue.
    /// Otherwise, performs in-place fused accumulation into `residual`.
    public static func encode(
        _ enc: any MTLComputeCommandEncoder,
        hal: QWHal,
        residual: QWTensorValue,
        gdnOut: QWTensorValue? = nil,
        attnOut: QWTensorValue? = nil,
        mlpOut: QWTensorValue? = nil,
        pleEmbed: QWTensorValue? = nil,
        pleGate: QWTensorValue? = nil,
        gamma: QWTensorValue? = nil,
        normOut: QWTensorValue? = nil,
        eps: Float = QWModelGeometry.qwen38_27b.rmsEps,
        flags: QWGatedResidualFlags = [.gammaPlusOne, .pleSigmoid],
        rows: Int = 1
    ) throws {
        let totalElems = residual.shape.dims.reduce(1, *)
        guard totalElems % rows == 0 else {
            throw QWGatedResidualError.badShape("residual elements \(totalElems) not divisible by rows \(rows)")
        }
        let hidden = totalElems / rows
        guard hidden % 8 == 0 else {
            throw QWGatedResidualError.badShape("hidden \(hidden) must be divisible by 8 for uint4 vectorization")
        }
        var hiddenU4 = UInt32(hidden / 8)

        var effectiveFlags = flags
        if gdnOut != nil { effectiveFlags.insert(.hasGDN) }
        if attnOut != nil { effectiveFlags.insert(.hasAttn) }
        if mlpOut != nil { effectiveFlags.insert(.hasMLP) }
        if pleEmbed != nil { effectiveFlags.insert(.hasPLE) }
        if pleGate != nil { effectiveFlags.insert(.hasGate) }

        var rawFlags = effectiveFlags.rawValue

        if let normOut = normOut {
            guard let gamma = gamma else {
                throw QWGatedResidualError.missingBuffer("gamma weight required when normOut is specified")
            }
            let pso = try hal.pipeline("qwg_rmsnorm_gated_residual_4branch")
            enc.setComputePipelineState(pso)

            residual.bind(enc, index: 0)
            if let gdnOut { gdnOut.bind(enc, index: 1) }
            if let attnOut { attnOut.bind(enc, index: 2) }
            if let mlpOut { mlpOut.bind(enc, index: 3) }
            if let pleEmbed { pleEmbed.bind(enc, index: 4) }
            if let pleGate { pleGate.bind(enc, index: 5) }
            gamma.bind(enc, index: 6)
            normOut.bind(enc, index: 7)

            enc.setBytes(&hiddenU4, length: 4, index: 8)
            var epsCopy = eps
            enc.setBytes(&epsCopy, length: 4, index: 9)
            enc.setBytes(&rawFlags, length: 4, index: 10)

            enc.dispatchThreadgroups(
                MTLSize(width: rows, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
            )
        } else {
            let pso = try hal.pipeline("qwg_gated_residual_4branch")
            enc.setComputePipelineState(pso)

            residual.bind(enc, index: 0)
            if let gdnOut { gdnOut.bind(enc, index: 1) }
            if let attnOut { attnOut.bind(enc, index: 2) }
            if let mlpOut { mlpOut.bind(enc, index: 3) }
            if let pleEmbed { pleEmbed.bind(enc, index: 4) }
            if let pleGate { pleGate.bind(enc, index: 5) }

            enc.setBytes(&hiddenU4, length: 4, index: 6)
            enc.setBytes(&rawFlags, length: 4, index: 7)

            enc.dispatchThreadgroups(
                MTLSize(width: rows, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
            )
        }
    }

    /// Pure host Swift reference for calculating fused 4-branch gated residual.
    public static func hostFusedResidual(
        residual: [Float],
        gdnOut: [Float]? = nil,
        attnOut: [Float]? = nil,
        mlpOut: [Float]? = nil,
        pleEmbed: [Float]? = nil,
        pleGate: [Float]? = nil,
        pleSigmoid: Bool = true
    ) -> [Float] {
        let n = residual.count
        var cur = residual.map { Float(Float16($0)) }

        if let gdn = gdnOut {
            precondition(gdn.count == n)
            for i in 0..<n {
                cur[i] = Float(Float16(cur[i] + Float(Float16(gdn[i]))))
            }
        }
        if let attn = attnOut {
            precondition(attn.count == n)
            for i in 0..<n {
                cur[i] = Float(Float16(cur[i] + Float(Float16(attn[i]))))
            }
        }
        if let mlp = mlpOut {
            precondition(mlp.count == n)
            for i in 0..<n {
                cur[i] = Float(Float16(cur[i] + Float(Float16(mlp[i]))))
            }
        }
        if let ple = pleEmbed {
            precondition(ple.count == n)
            for i in 0..<n {
                var g: Float = 1.0
                if let gate = pleGate {
                    let gv = Float(Float16(gate[i]))
                    if pleSigmoid {
                        g = Float(Float16(1.0 / (1.0 + exp(-gv))))
                    } else {
                        g = gv
                    }
                }
                let inj = Float(Float16(Float(Float16(ple[i])) * g))
                cur[i] = Float(Float16(cur[i] + inj))
            }
        }
        return cur
    }
}
