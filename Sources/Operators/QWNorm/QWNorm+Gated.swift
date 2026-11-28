// QWNorm+Gated.swift — Fused low-rank GatedNorm and sandwich-residual GatedNorm encoders.
// Supports raw BF16 or FP16 weights, cooperative 4-row prefill tiling, in-register FWHT32,
// and optional fused MoE `(xRot32, xPlain32)` F32 bridge.
import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc

/// Low-rank bottleneck `GatedNorm` parameters (`modeling_gigachat4.py:161-173`).
public struct QWGatedNormWeights: Sendable {
    public let weight: QWTensorValue
    public let gateUp: QWTensorValue
    public let gateDown: QWTensorValue
    public let rank: Int
    public let gatingWeight: Float
    public let zeroCentered: Bool

    public init(
        weight: QWTensorValue,
        gateUp: QWTensorValue,
        gateDown: QWTensorValue,
        rank: Int = 16,
        gatingWeight: Float = 2.0,
        zeroCentered: Bool = false
    ) {
        self.weight = weight
        self.gateUp = gateUp
        self.gateDown = gateDown
        self.rank = rank
        self.gatingWeight = gatingWeight
        self.zeroCentered = zeroCentered
    }
}

extension QWNorm {
    private struct GatedNormParams {
        var rows: UInt32
        var hidden: UInt32
        var rank: UInt32
        var flags: UInt32
        var eps: Float
        var gatingWeight: Float
        var rowTile: UInt32
        var pad0: UInt32 = 0
    }

    private struct GatedSandwichParams {
        var rows: UInt32
        var hidden: UInt32
        var postRank: UInt32
        var preRank: UInt32
        var flags: UInt32
        var rowTile: UInt32
        var postEps: Float
        var postGatingWeight: Float
        var preEps: Float
        var preGatingWeight: Float
    }

    private static func validateGatedWeights(
        _ w: QWGatedNormWeights,
        hidden: Int,
        bitShift: UInt32
    ) throws -> UInt32 {
        guard (1...32).contains(w.rank) else {
            throw QWNormError.badShape("GatedNorm rank \(w.rank) must be in 1...32")
        }
        let validW = (w.weight.shape.dtype == .bf16 || w.weight.shape.dtype == .f16)
            && w.weight.shape.elementCount == hidden
        let validUp = (w.gateUp.shape.dtype == .bf16 || w.gateUp.shape.dtype == .f16)
            && w.gateUp.shape.elementCount == w.rank * hidden
        let validDown = (w.gateDown.shape.dtype == .bf16 || w.gateDown.shape.dtype == .f16)
            && w.gateDown.shape.elementCount == hidden * w.rank
        guard validW && validUp && validDown else {
            throw QWNormError.badShape(
                "GatedNorm weights require bf16|f16 shapes weight[\(hidden)], gateUp[\(w.rank), \(hidden)], gateDown[\(hidden), \(w.rank)]"
            )
        }
        var bits: UInt32 = 0
        if w.weight.shape.dtype == .bf16 { bits |= 1 }
        if w.gateUp.shape.dtype == .bf16 { bits |= 2 }
        if w.gateDown.shape.dtype == .bf16 { bits |= 4 }
        if w.zeroCentered { bits |= 8 }
        return bits << bitShift
    }

    private static func encodeMultiTGGatedPass(
        _ enc: any MTLComputeCommandEncoder,
        hal: QWHal,
        x: QWTensorValue,
        weights: QWGatedNormWeights,
        outOrRes: QWTensorValue,
        rows: Int,
        hidden D: Int,
        eps: Float,
        flags: UInt32,
        xRot32: QWTensorValue?,
        xPlain32: QWTensorValue?
    ) throws {
        let scratch = hal.gatedNormScratch
        let maxHScratchBytes = 4_194_304
        let g1ByteOffset = 4_194_304
        let rawCap = min(512, maxHScratchBytes / max(2, D * 2))
        let maxChunkRows = max(4, (rawCap / 4) * 4)
        let useFusedHG1 = (D <= 4096 && rows <= 32)
        let psoHG1 = useFusedHG1 ? try hal.pipeline("qwg_gated_stage_h_g1_f16") : nil
        let psoH = useFusedHG1 ? nil : try hal.pipeline("qwg_gated_stage_h_f16")
        let psoG1 = useFusedHG1 ? nil : try hal.pipeline("qwg_gated_stage_g1_f16")
        let psoDown = try hal.pipeline("qwg_gated_apply_down_f16")

        var r0 = 0
        while r0 < rows {
            let chunkRows = min(maxChunkRows, rows - r0)
            let rowGroups4 = (chunkRows + 3) / 4
            let useWideG1 = !useFusedHG1 && (D <= 4096) && (chunkRows <= 32)
            var chunkFlags = flags
            if useWideG1 { chunkFlags |= 512 }
            let xSlice = (r0 == 0 && chunkRows == rows)
                ? x : x.slice(byteOffset: r0 * D * 2, count: chunkRows * D, dtype: .f16)
            let outSlice = (r0 == 0 && chunkRows == rows)
                ? outOrRes : outOrRes.slice(byteOffset: r0 * D * 2, count: chunkRows * D, dtype: .f16)
            let rotSlice = xRot32.map {
                (r0 == 0 && chunkRows == rows) ? $0 : $0.slice(byteOffset: r0 * D * 4, count: chunkRows * D, dtype: .f32)
            }
            let plainSlice = xPlain32.map {
                (r0 == 0 && chunkRows == rows) ? $0 : $0.slice(byteOffset: r0 * D * 4, count: chunkRows * D, dtype: .f32)
            }
            var params = GatedNormParams(
                rows: UInt32(chunkRows),
                hidden: UInt32(D),
                rank: UInt32(weights.rank),
                flags: chunkFlags,
                eps: eps,
                gatingWeight: weights.gatingWeight,
                rowTile: 1
            )

            if let psoHG1 {
                enc.setComputePipelineState(psoHG1)
                xSlice.bind(enc, index: 0)
                weights.weight.bind(enc, index: 1)
                weights.gateUp.bind(enc, index: 2)
                enc.setBuffer(scratch, offset: 0, index: 3)
                enc.setBuffer(scratch, offset: g1ByteOffset, index: 4)
                enc.setBytes(&params, length: MemoryLayout<GatedNormParams>.stride, index: 5)
                enc.dispatchThreadgroups(
                    MTLSize(width: weights.rank, height: chunkRows, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
                )
                enc.memoryBarrier(scope: .buffers)
            } else if let psoH, let psoG1 {
                enc.setComputePipelineState(psoH)
                xSlice.bind(enc, index: 0)
                weights.weight.bind(enc, index: 1)
                enc.setBuffer(scratch, offset: 0, index: 2)
                enc.setBytes(&params, length: MemoryLayout<GatedNormParams>.stride, index: 3)
                enc.dispatchThreadgroups(
                    MTLSize(width: chunkRows, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
                )
                enc.memoryBarrier(scope: .buffers)

                enc.setComputePipelineState(psoG1)
                enc.setBuffer(scratch, offset: 0, index: 0)
                weights.gateUp.bind(enc, index: 1)
                enc.setBuffer(scratch, offset: g1ByteOffset, index: 2)
                enc.setBytes(&params, length: MemoryLayout<GatedNormParams>.stride, index: 3)
                enc.dispatchThreadgroups(
                    MTLSize(width: weights.rank, height: rowGroups4, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: useWideG1 ? 256 : 32, height: 1, depth: 1)
                )
                enc.memoryBarrier(scope: .buffers)
            }

            enc.setComputePipelineState(psoDown)
            enc.setBuffer(scratch, offset: 0, index: 0)
            enc.setBuffer(scratch, offset: g1ByteOffset, index: 1)
            weights.gateDown.bind(enc, index: 2)
            outSlice.bind(enc, index: 3)
            enc.setBytes(&params, length: MemoryLayout<GatedNormParams>.stride, index: 4)
            (rotSlice ?? plainSlice ?? outSlice).bind(enc, index: 5)
            (plainSlice ?? rotSlice ?? outSlice).bind(enc, index: 6)
            enc.dispatchThreadgroups(
                MTLSize(width: (D + 255) / 256, height: rowGroups4, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
            )
            r0 += chunkRows
            if r0 < rows {
                enc.memoryBarrier(scope: .buffers)
            }
        }
    }

    /// Encodes standalone / multi-row cooperative `GatedNorm` (`qwg_gated_rmsnorm_f16`).
    public static func encodeGatedRMSNorm(
        _ enc: any MTLComputeCommandEncoder,
        hal: QWHal,
        x: QWTensorValue,
        weights: QWGatedNormWeights,
        out: QWTensorValue,
        eps: Float? = nil,
        hidden: Int? = nil,
        applyFwht32: Bool = true,
        xRot32: QWTensorValue? = nil,
        xPlain32: QWTensorValue? = nil,
        skipRot32: Bool = false
    ) throws {
        let D = hidden ?? hal.geometry.hidden
        guard D > 0, D <= 8192 else {
            throw QWNormError.geometryTooWide("GatedNorm hidden \(D) must be in 1...8192")
        }
        if applyFwht32 && D % 32 != 0 {
            throw QWNormError.geometryTooWide("GatedNorm applyFwht32 requires hidden (\(D)) % 32 == 0")
        }
        let elems = x.shape.elementCount
        guard x.shape.dtype == .f16, out.shape.dtype == .f16,
              elems > 0, elems % D == 0, out.shape.elementCount >= elems else {
            throw QWNormError.badShape("GatedNorm x/out must be f16 multiples of hidden \(D)")
        }
        let rows = elems / D
        var flags = try validateGatedWeights(weights, hidden: D, bitShift: 0)
        if applyFwht32 { flags |= 16 }
        let emitMoE = (xPlain32 != nil || xRot32 != nil)
        if emitMoE {
            guard let plain = xPlain32, plain.shape.dtype == .f32, plain.shape.elementCount >= elems else {
                throw QWNormError.badShape("GatedNorm MoE bridge requires xPlain32[\(elems)] f32")
            }
            if !skipRot32 {
                guard let rot = xRot32, rot.shape.dtype == .f32, rot.shape.elementCount >= elems else {
                    throw QWNormError.badShape("GatedNorm MoE bridge requires xRot32[\(elems)] f32 unless skipRot32")
                }
            }
            flags |= 32
            if skipRot32 { flags |= 64 }
        }

        let resolvedEps = eps ?? hal.geometry.rmsEps
        if rows <= 32 || (D > 3072 && rows <= 256) {
            try encodeMultiTGGatedPass(
                enc, hal: hal, x: x, weights: weights, outOrRes: out,
                rows: rows, hidden: D, eps: resolvedEps, flags: flags,
                xRot32: xRot32, xPlain32: xPlain32
            )
            return
        }

        let rowTile = (rows >= 64 && D <= 3072) ? 4 : 1
        let tgHBytes = ((max(256, rowTile * D * 2) + 15) & ~15)
        var params = GatedNormParams(
            rows: UInt32(rows),
            hidden: UInt32(D),
            rank: UInt32(weights.rank),
            flags: flags,
            eps: resolvedEps,
            gatingWeight: weights.gatingWeight,
            rowTile: UInt32(rowTile)
        )
        let pso = try hal.pipeline("qwg_gated_rmsnorm_f16")
        let tptg = (weights.rank >= 16 && pso.maxTotalThreadsPerThreadgroup >= 512) ? 512 : 256
        enc.setComputePipelineState(pso)
        x.bind(enc, index: 0)
        weights.weight.bind(enc, index: 1)
        weights.gateUp.bind(enc, index: 2)
        weights.gateDown.bind(enc, index: 3)
        out.bind(enc, index: 4)
        enc.setBytes(&params, length: MemoryLayout<GatedNormParams>.stride, index: 5)
        (xRot32 ?? xPlain32 ?? out).bind(enc, index: 6)
        (xPlain32 ?? xRot32 ?? out).bind(enc, index: 7)
        enc.setThreadgroupMemoryLength(tgHBytes, index: 0)
        enc.dispatchThreadgroups(
            MTLSize(width: (rows + rowTile - 1) / rowTile, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tptg, height: 1, depth: 1)
        )
    }

    /// Encodes fused sandwich residual `x_new = half(residual + GatedNorm_post(branchOut))`
    /// and optional immediate `out = optional_fwht32(GatedNorm_pre(x_new))` + MoE F32 bridge.
    public static func encodeGatedSandwichResidual(
        _ enc: any MTLComputeCommandEncoder,
        hal: QWHal,
        branchOut: QWTensorValue,
        residual: QWTensorValue,
        postWeights: QWGatedNormWeights,
        nextPreWeights: QWGatedNormWeights? = nil,
        out: QWTensorValue? = nil,
        postEps: Float? = nil,
        preEps: Float? = nil,
        hidden: Int? = nil,
        applyFwht32: Bool = true,
        xRot32: QWTensorValue? = nil,
        xPlain32: QWTensorValue? = nil,
        skipRot32: Bool = false
    ) throws {
        let D = hidden ?? hal.geometry.hidden
        guard D > 0, D <= 8192 else {
            throw QWNormError.geometryTooWide("GatedNorm sandwich hidden \(D) must be in 1...8192")
        }
        let elems = branchOut.shape.elementCount
        guard branchOut.shape.dtype == .f16, residual.shape.dtype == .f16,
              elems > 0, elems % D == 0, residual.shape.elementCount >= elems else {
            throw QWNormError.badShape("GatedNorm sandwich branchOut/residual must be f16 multiples of hidden \(D)")
        }
        let rows = elems / D
        var flags = try validateGatedWeights(postWeights, hidden: D, bitShift: 0)
        let preW: QWGatedNormWeights
        if let nextPre = nextPreWeights {
            if applyFwht32 && D % 32 != 0 {
                throw QWNormError.geometryTooWide("GatedNorm sandwich applyFwht32 requires hidden (\(D)) % 32 == 0")
            }
            guard let outTensor = out, outTensor.shape.dtype == .f16, outTensor.shape.elementCount >= elems else {
                throw QWNormError.badShape("GatedNorm sandwich with nextPreWeights requires out[\(elems)] f16")
            }
            flags |= 128
            if applyFwht32 { flags |= 16 }
            flags |= try validateGatedWeights(nextPre, hidden: D, bitShift: 8)
            preW = nextPre
        } else {
            preW = postWeights
        }
        let emitMoE = (nextPreWeights != nil) && (xPlain32 != nil || xRot32 != nil)
        if emitMoE {
            guard let plain = xPlain32, plain.shape.dtype == .f32, plain.shape.elementCount >= elems else {
                throw QWNormError.badShape("GatedNorm sandwich MoE bridge requires xPlain32[\(elems)] f32")
            }
            if !skipRot32 {
                guard let rot = xRot32, rot.shape.dtype == .f32, rot.shape.elementCount >= elems else {
                    throw QWNormError.badShape("GatedNorm sandwich MoE bridge requires xRot32[\(elems)] f32 unless skipRot32")
                }
            }
            flags |= 32
            if skipRot32 { flags |= 64 }
        }

        let resolvedPostEps = postEps ?? hal.geometry.rmsEps
        let resolvedPreEps = preEps ?? resolvedPostEps
        if rows <= 32 || (D > 3072 && rows <= 256) {
            let postFlags = (flags & 0x0F) | 128
            try encodeMultiTGGatedPass(
                enc, hal: hal, x: branchOut, weights: postWeights, outOrRes: residual,
                rows: rows, hidden: D, eps: resolvedPostEps, flags: postFlags,
                xRot32: nil, xPlain32: nil
            )
            if let nextPre = nextPreWeights, let outTensor = out {
                enc.memoryBarrier(scope: .buffers)
                let preFlags = ((flags >> 8) & 0x0F) | (flags & (16 | 32 | 64))
                try encodeMultiTGGatedPass(
                    enc, hal: hal, x: residual, weights: nextPre, outOrRes: outTensor,
                    rows: rows, hidden: D, eps: resolvedPreEps, flags: preFlags,
                    xRot32: xRot32, xPlain32: xPlain32
                )
            }
            return
        }

        let rowTile = (rows >= 64 && D <= 3072) ? 4 : 1
        let tgHBytes = ((max(256, rowTile * D * 2) + 15) & ~15)
        var params = GatedSandwichParams(
            rows: UInt32(rows),
            hidden: UInt32(D),
            postRank: UInt32(postWeights.rank),
            preRank: UInt32(preW.rank),
            flags: flags,
            rowTile: UInt32(rowTile),
            postEps: resolvedPostEps,
            postGatingWeight: postWeights.gatingWeight,
            preEps: resolvedPreEps,
            preGatingWeight: preW.gatingWeight
        )
        let outBind = out ?? residual
        let pso = try hal.pipeline("qwg_gated_rmsnorm_sandwich_residual_f16")
        let maxRank = max(postWeights.rank, preW.rank)
        let tptg = (maxRank >= 16 && pso.maxTotalThreadsPerThreadgroup >= 512) ? 512 : 256
        enc.setComputePipelineState(pso)
        branchOut.bind(enc, index: 0)
        residual.bind(enc, index: 1)
        postWeights.weight.bind(enc, index: 2)
        postWeights.gateUp.bind(enc, index: 3)
        postWeights.gateDown.bind(enc, index: 4)
        preW.weight.bind(enc, index: 5)
        preW.gateUp.bind(enc, index: 6)
        preW.gateDown.bind(enc, index: 7)
        outBind.bind(enc, index: 8)
        enc.setBytes(&params, length: MemoryLayout<GatedSandwichParams>.stride, index: 9)
        (xRot32 ?? xPlain32 ?? outBind).bind(enc, index: 10)
        (xPlain32 ?? xRot32 ?? outBind).bind(enc, index: 11)
        enc.setThreadgroupMemoryLength(tgHBytes, index: 0)
        enc.dispatchThreadgroups(
            MTLSize(width: (rows + rowTile - 1) / rowTile, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tptg, height: 1, depth: 1)
        )
    }
}
