import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc

/// C06 — QWNorm: RMSNorm family + fused epilogues (specs/06-qwnorm.md).
/// One kernel family; grid = rows, 32 threads/row, fp32 sumsq (law), eps from
/// C01. Variant 2 (residual-fused) is the shipping default until the P3
public enum QWNormVariant: String, Sendable {
    case rmsnorm                       // variant 1: correctness anchor
    case rmsnormResidual               // variant 2: residual += x fused (shipping)
    case rmsnormPrologueScale          // variant 3: row-scale side vector (P3 race, standalone)
    case rmsnormResidualPrologueScale  // variant 4: fused residual + prologue-scale (P3 race, combined)
    case rmsnormFromRot                // variant 6: un-rotate rotated input, RMSNorm, re-rotate output (MTP preFcNormHidden)
    case headNorm                      // variant 5: lives in C08/C09 rope prologue
}

public enum QWNormError: Error, CustomStringConvertible {
    case badShape(String)
    case variantNotHere(String)
    case geometryTooWide(String)
    public var description: String {
        switch self {
        case .badShape(let s): return "QWNorm: \(s)"
        case .geometryTooWide(let s): return "QWNorm: \(s)"
        case .variantNotHere(let s): return "QWNorm: \(s) is fused into the rope kernel (C08/C09)"
        }
    }
}

public enum QWNorm {
    private static func kernelName(_ v: QWNormVariant) -> String {
        switch v {
        case .rmsnorm: return "qwg_rmsnorm"
        case .rmsnormResidual: return "qwg_rmsnorm_residual"
        case .rmsnormPrologueScale: return "qwg_rmsnorm_prologue_scale"
        case .rmsnormResidualPrologueScale: return "qwg_rmsnorm_residual_pscale"
        case .rmsnormFromRot: return "qwg_rmsnorm_from_rot"
        case .headNorm: return ""
        }
    }

    /// Encode one norm call. rows inferred from x (1-D [rows·hidden] or 2-D).
    /// `residual` required for .rmsnormResidual; `rowScalesOut` for
    /// .rmsnormPrologueScale. gamma is [hidden] f16.
    public static func encode(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                              variant: QWNormVariant, x: QWTensorValue,
                              gamma: QWTensorValue, out: QWTensorValue,
                              residual: QWTensorValue? = nil,
                              rowScalesOut: QWTensorValue? = nil,
                              geometry: QWModelGeometry) throws {
        guard variant != .headNorm else {
            throw QWNormError.variantNotHere("head-norm")
        }
        let hidden = geometry.hidden
        let elems = x.shape.dims.reduce(1, *)
        guard elems % hidden == 0 else {
            throw QWNormError.badShape("rows·hidden mismatch: \(x.shape.dims) % \(hidden)")
        }
        let rows = elems / hidden
        guard gamma.shape.dims.reduce(1, *) == hidden else {
            throw QWNormError.badShape("gamma must be [\(hidden)]")
        }
        switch variant {
        case .rmsnormResidual:
            guard let r = residual, r.shape.dims.reduce(1, *) == elems else {
                throw QWNormError.badShape("residual required, same shape as x")
            }
        case .rmsnormPrologueScale:
            guard let s = rowScalesOut, s.shape.dims == [rows] else {
                throw QWNormError.badShape("rowScalesOut must be [\(rows)] f32")
            }
        case .rmsnormResidualPrologueScale:
            guard let r = residual, r.shape.dims.reduce(1, *) == elems else {
                throw QWNormError.badShape("residual required, same shape as x")
            }
            guard let s = rowScalesOut, s.shape.dims == [rows] else {
                throw QWNormError.badShape("rowScalesOut must be [\(rows)] f32")
            }
        default: break
        }
        let pso = try hal.pipeline(kernelName(variant))
        enc.setComputePipelineState(pso)
        x.bind(enc, index: 0)
        gamma.bind(enc, index: 1)
        out.bind(enc, index: 2)
        var hiddenU4 = UInt32(hidden / 8)      // uint4 = 8 halves
        // Every variant in this family is uint4-vectorised and rotates
        // into the H_32 basis in groups of 32 halves, so the hidden must
        // be a multiple of 256: the row's uint4 loads must stay 16-byte
        // aligned, every 4-lane butterfly group must be fully populated,
        // and — because the rotation's cross-lane stages are a collective
        // `simd_shuffle_xor` — all 32 lanes must iterate the same number
        // of times. The 27B's 5120 takes the residual kernel's
        // register-resident path; every other hidden takes its general
        // path, which is the same arithmetic in the same order.
        guard hidden % 256 == 0 else {
            throw QWNormError.geometryTooWide("hidden \(hidden) must be a multiple of 256")
        }
        var eps = geometry.rmsEps
        enc.setBytes(&hiddenU4, length: 4, index: 3)
        enc.setBytes(&eps, length: 4, index: 4)
        switch variant {
        case .rmsnormResidual:      residual!.bind(enc, index: 5)
        case .rmsnormPrologueScale: rowScalesOut!.bind(enc, index: 5)
        case .rmsnormResidualPrologueScale:
            residual!.bind(enc, index: 5)
            rowScalesOut!.bind(enc, index: 6)
        default: break
        }
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    public static func encodeMTPPreFcNorm(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                          embeddings: QWTensorValue, hidden: QWTensorValue,
                                          previous: QWTensorValue, gammaEmb: QWTensorValue,
                                          gammaHid: QWTensorValue, out: QWTensorValue,
                                          tStart: Int, pairs: Int,
                                          geometry: QWModelGeometry) throws {
        guard pairs > 0 else { return }
        // The same family law: this kernel runs the same 32-half
        // butterfly, so the hidden must be a multiple of 256.
        guard geometry.hidden % 256 == 0 else {
            throw QWNormError.geometryTooWide("hidden \(geometry.hidden) must be a multiple of 256")
        }
        let pso = try hal.pipeline("qwg_mtp_pre_fc_norm")
        enc.setComputePipelineState(pso)
        embeddings.bind(enc, index: 0)
        hidden.bind(enc, index: 1)
        previous.bind(enc, index: 2)
        gammaEmb.bind(enc, index: 3)
        gammaHid.bind(enc, index: 4)
        out.bind(enc, index: 5)
        var hiddenU4 = UInt32(geometry.hidden / 8)
        var eps = geometry.rmsEps
        var tStartU = UInt32(tStart)
        enc.setBytes(&hiddenU4, length: 4, index: 6)
        enc.setBytes(&eps, length: 4, index: 7)
        enc.setBytes(&tStartU, length: 4, index: 8)
        enc.dispatchThreadgroups(MTLSize(width: pairs, height: 2, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }
}
