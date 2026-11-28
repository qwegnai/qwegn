// QWGemmFP+MoESwigluNax.swift — Fused grouped-NAX MoE gate+up+softcap SwiGLU encoders for NVFP4_G16 & MXFP8_G32.
import Foundation
import Metal
import QWAlloc
import QWCore
import QWHAL

private struct QWGemmFPMoENaxSwigluParams {
    var experts: UInt32
    var n: UInt32
    var k: UInt32
    var wRows: UInt32
    var inDtype: UInt32
    var outDtype: UInt32
    var flags: UInt32
    var gateCap: Float
    var upCap: Float
}

private struct QWGemmFPMoEGatherParams {
    var totalPairs: UInt32
    var k: UInt32
    var inDtype: UInt32
    var outDtype: UInt32
}

extension QWGemmFP {
    /// Gathers token rows `x[sortedTok[s]]` into contiguous `xSorted[s]` before grouped NAX GEMMs.
    public static func encodeMoEGatherX(
        _ enc: any MTLComputeCommandEncoder, hal: QWHal,
        x: QWFPBufferSlice, sortedTok: QWTensorValue, xSorted: QWFPBufferSlice,
        totalPairs: Int, k: Int,
        inDtype: QWFPElemDtype, outDtype: QWFPElemDtype
    ) throws {
        guard totalPairs > 0, k > 0 else { return }
        var params = QWGemmFPMoEGatherParams(
            totalPairs: UInt32(totalPairs), k: UInt32(k),
            inDtype: inDtype.rawValue, outDtype: outDtype.rawValue)
        enc.setComputePipelineState(try hal.pipeline("qwg_moe_gather_x"))
        x.bind(enc, index: 0)
        sortedTok.bind(enc, index: 1)
        xSorted.bind(enc, index: 2)
        enc.setBytes(&params, length: MemoryLayout<QWGemmFPMoEGatherParams>.stride, index: 3)
        let gx = (k + 7) / 8
        enc.dispatchThreads(MTLSize(width: gx, height: totalPairs, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(256, gx), height: 1, depth: 1))
    }

    /// Encodes fused grouped-NAX `gate_up + softcap SwiGLU` prefill directly into `actOut` over `NVFP4_G16`.
    public static func encodeMoENaxNVFP4SwiGLU(
        _ enc: any MTLComputeCommandEncoder, hal: QWHal,
        bank: NVFP4Bank, x: QWFPBufferSlice, actOut: QWFPBufferSlice,
        expertOff: QWTensorValue, tileE: QWTensorValue, tileR: QWTensorValue,
        tileCount: QWTensorValue, tilePlan: QWGemm.MoETilePlan,
        intermediate: Int, inDtype: QWFPElemDtype, outDtype: QWFPElemDtype,
        gateCap: Float = 0, upCap: Float = 0,
        sortedTok: QWTensorValue? = nil
    ) throws {
        let useNAX = hal.caps.isMetal4TensorCapable
        guard hal.caps.supportsSimdgroupMMA else {
            throw QWGemmError.badShape("qwg_moe_nax_gemm_nvfp4_g16_swiglu requires Apple7+ simdgroup or Metal 4 NAX tensor support")
        }
        guard inDtype == .f16 || inDtype == .bf16 else {
            throw QWGemmError.badShape("NVFP4 NAX SwiGLU requires .f16 or .bf16 input activation")
        }
        guard intermediate > 0, bank.rowsPerExpert >= 2 * intermediate, bank.k > 0, bank.k % 16 == 0 else {
            throw QWGemmError.badShape("NVFP4 NAX SwiGLU requires concat gate_up weights and k % 16 == 0")
        }
        guard tileCount.shape.elementCount >= tilePlan.scratchInts else {
            throw QWGemmError.badShape("tileCount buffer too small for MoETilePlan")
        }
        if bank.splitGateUpScales2 {
            guard let s2 = bank.scales2, s2.bytes >= 2 * bank.experts * 4 else {
                throw QWGemmError.badShape("splitGateUpScales2 NVFP4 bank requires 2*experts s2")
            }
        }
        let colTiles = (intermediate + 31) / 32
        guard tilePlan.guColTiles == colTiles else {
            throw QWGemmError.badShape("MoETilePlan guColTiles=\(tilePlan.guColTiles) != BN32 colTiles=\(colTiles) (pass gateUpRows=2*I)")
        }
        let flags: UInt32 = (bank.scales2 != nil ? 1 : 0)
            | (sortedTok != nil ? 2 : 0)
            | (bank.splitGateUpScales2 && bank.scales2 != nil ? 8 : 0)
        var params = QWGemmFPMoENaxSwigluParams(
            experts: UInt32(bank.experts), n: UInt32(intermediate), k: UInt32(bank.k),
            wRows: UInt32(bank.rowsPerExpert),
            inDtype: inDtype.rawValue, outDtype: outDtype.rawValue,
            flags: flags, gateCap: gateCap, upCap: upCap)
        enc.setComputePipelineState(try hal.pipeline(useNAX ? "qwg_moe_nax_gemm_nvfp4_g16_swiglu" : "qwg_moe_sg_gemm_nvfp4_g16_swiglu"))
        bank.weights.bind(enc, index: 0)
        bank.scales.bind(enc, index: 1)
        (bank.scales2 ?? bank.scales).bind(enc, index: 2)
        x.bind(enc, index: 3)
        actOut.bind(enc, index: 4)
        enc.bindTensors(expertOff, tileE, tileR, tileCount, startIndex: 5)
        enc.setBytes(&params, length: MemoryLayout<QWGemmFPMoENaxSwigluParams>.stride, index: 9)
        (sortedTok ?? expertOff).bind(enc, index: 11)
        QWGemm.dispatchTileChunks(
            enc, indirectBuffer: tileCount.region.buffer,
            baseByteOffset: tileCount.region.base + tileCount.offset + 16,
            colTiles: colTiles, maxTiles: tilePlan.maxTiles, flags: 0, flagsIndex: 10)
    }

    /// Encodes fused grouped-NAX `gate_up + softcap SwiGLU` prefill directly into `actOut` over `MXFP8_G32`.
    public static func encodeMoENaxMXFP8SwiGLU(
        _ enc: any MTLComputeCommandEncoder, hal: QWHal,
        bank: MXFP8Bank, x: QWFPBufferSlice, actOut: QWFPBufferSlice,
        expertOff: QWTensorValue, tileE: QWTensorValue, tileR: QWTensorValue,
        tileCount: QWTensorValue, tilePlan: QWGemm.MoETilePlan,
        intermediate: Int, inDtype: QWFPElemDtype, outDtype: QWFPElemDtype,
        gateCap: Float = 0, upCap: Float = 0,
        sortedTok: QWTensorValue? = nil
    ) throws {
        let useNAX = hal.caps.isMetal4TensorCapable
        guard hal.caps.supportsSimdgroupMMA else {
            throw QWGemmError.badShape("qwg_moe_nax_gemm_mxfp8_g32_swiglu requires Apple7+ simdgroup or Metal 4 NAX tensor support")
        }
        guard inDtype == .f16 || inDtype == .bf16 else {
            throw QWGemmError.badShape("MXFP8 NAX SwiGLU requires .f16 or .bf16 input activation")
        }
        guard intermediate > 0, bank.rowsPerExpert >= 2 * intermediate, bank.k > 0, bank.k % 32 == 0 else {
            throw QWGemmError.badShape("MXFP8 NAX SwiGLU requires concat gate_up weights and k % 32 == 0")
        }
        guard tileCount.shape.elementCount >= tilePlan.scratchInts else {
            throw QWGemmError.badShape("tileCount buffer too small for MoETilePlan")
        }
        let colTiles = (intermediate + 31) / 32
        guard tilePlan.guColTiles == colTiles else {
            throw QWGemmError.badShape("MoETilePlan guColTiles=\(tilePlan.guColTiles) != BN32 colTiles=\(colTiles) (pass gateUpRows=2*I)")
        }
        let flags: UInt32 = (sortedTok != nil ? 2 : 0)
        var params = QWGemmFPMoENaxSwigluParams(
            experts: UInt32(bank.experts), n: UInt32(intermediate), k: UInt32(bank.k),
            wRows: UInt32(bank.rowsPerExpert),
            inDtype: inDtype.rawValue, outDtype: outDtype.rawValue,
            flags: flags, gateCap: gateCap, upCap: upCap)
        enc.setComputePipelineState(try hal.pipeline(useNAX ? "qwg_moe_nax_gemm_mxfp8_g32_swiglu" : "qwg_moe_sg_gemm_mxfp8_g32_swiglu"))
        bank.weights.bind(enc, index: 0)
        bank.scales.bind(enc, index: 1)
        x.bind(enc, index: 2)
        actOut.bind(enc, index: 3)
        enc.bindTensors(expertOff, tileE, tileR, tileCount, startIndex: 4)
        enc.setBytes(&params, length: MemoryLayout<QWGemmFPMoENaxSwigluParams>.stride, index: 8)
        (sortedTok ?? expertOff).bind(enc, index: 10)
        QWGemm.dispatchTileChunks(
            enc, indirectBuffer: tileCount.region.buffer,
            baseByteOffset: tileCount.region.base + tileCount.offset + 16,
            colTiles: colTiles, maxTiles: tilePlan.maxTiles, flags: 0, flagsIndex: 9)
    }
}
