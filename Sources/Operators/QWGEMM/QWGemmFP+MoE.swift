// QWGemmFP+MoE.swift — Grouped-NAX, compact active-bucket, SwiGLU,
// and combine encoders for QWGemmFP (BF16 / NVFP4_G16 / MXFP8_G32).
import Foundation
import Metal
import QWAlloc
import QWCore
import QWHAL

extension QWGemmFP {
    /// Encodes grouped-NAX prefill GEMM over `NVFP4_G16` expert bank.
    public static func encodeMoENaxNVFP4(
        _ enc: any MTLComputeCommandEncoder, hal: QWHal,
        bank: NVFP4Bank, x: QWFPBufferSlice, y: QWFPBufferSlice,
        expertOff: QWTensorValue, tileE: QWTensorValue, tileR: QWTensorValue,
        tileCount: QWTensorValue, tilePlan: QWGemm.MoETilePlan,
        inDtype: QWFPElemDtype, outDtype: QWFPElemDtype,
        nOverride: Int? = nil, yStride: Int? = nil, yColOff: Int = 0,
        sortedTok: QWTensorValue? = nil, isDownIndirect: Bool = false
    ) throws {
        let useNAX = hal.caps.isMetal4TensorCapable
        guard hal.caps.supportsSimdgroupMMA else {
            throw QWGemmError.badShape("qwg_moe_nax_gemm_nvfp4_g16 requires Apple7+ simdgroup or Metal 4 NAX tensor support")
        }
        guard inDtype == .f16 || inDtype == .bf16 else {
            throw QWGemmError.badShape("NVFP4 NAX prefill requires .f16 or .bf16 input activation")
        }
        let nProj = nOverride ?? bank.rowsPerExpert
        guard nProj > 0, bank.k > 0, bank.k % 16 == 0 else {
            throw QWGemmError.badShape("NVFP4 NAX prefill requires k % 16 == 0 and n > 0")
        }
        guard tileCount.shape.elementCount >= tilePlan.scratchInts else {
            throw QWGemmError.badShape("tileCount buffer too small for MoETilePlan")
        }
        if bank.splitGateUpScales2 {
            guard nProj == bank.rowsPerExpert, nProj % 2 == 0,
                  let s2 = bank.scales2, s2.bytes >= 2 * bank.experts * 4 else {
                throw QWGemmError.badShape("splitGateUpScales2 NVFP4 bank requires rowsPerExpert=2*I and 2*experts s2")
            }
        }
        let colTiles = (nProj + 63) / 64
        let flags: UInt32 = (bank.scales2 != nil ? 1 : 0)
            | (sortedTok != nil ? 2 : 0)
            | (bank.splitGateUpScales2 && bank.scales2 != nil ? 8 : 0)
        let params = QWGemmFPMoENaxParams(
            experts: UInt32(bank.experts), n: UInt32(nProj), k: UInt32(bank.k),
            yStride: UInt32(yStride ?? nProj), yColOff: UInt32(yColOff),
            inDtype: inDtype.rawValue, outDtype: outDtype.rawValue, flags: flags)
        enc.setComputePipelineState(try hal.pipeline(useNAX ? "qwg_moe_nax_gemm_nvfp4_g16" : "qwg_moe_sg_gemm_nvfp4_g16"))
        bank.weights.bind(enc, index: 0)
        bank.scales.bind(enc, index: 1)
        (bank.scales2 ?? bank.scales).bind(enc, index: 2)
        x.bind(enc, index: 3)
        y.bind(enc, index: 4)
        enc.bindTensors(expertOff, tileE, tileR, tileCount, startIndex: 5)
        enc.setPod(params, index: 9)
        (sortedTok ?? expertOff).bind(enc, index: 11)
        let baseByteOffset = tileCount.region.base + tileCount.offset
            + (isDownIndirect ? tilePlan.downIndirectByteOffset : 16)
        QWGemm.dispatchTileChunks(
            enc, indirectBuffer: tileCount.region.buffer,
            baseByteOffset: baseByteOffset, colTiles: colTiles,
            maxTiles: tilePlan.maxTiles, flags: 0, flagsIndex: 10)
    }

    /// Encodes grouped-NAX prefill GEMM over `MXFP8_G32` expert bank (with native `NAXTile<bfloat>` weights).
    public static func encodeMoENaxMXFP8(
        _ enc: any MTLComputeCommandEncoder, hal: QWHal,
        bank: MXFP8Bank, x: QWFPBufferSlice, y: QWFPBufferSlice,
        expertOff: QWTensorValue, tileE: QWTensorValue, tileR: QWTensorValue,
        tileCount: QWTensorValue, tilePlan: QWGemm.MoETilePlan,
        inDtype: QWFPElemDtype, outDtype: QWFPElemDtype,
        nOverride: Int? = nil, yStride: Int? = nil, yColOff: Int = 0,
        sortedTok: QWTensorValue? = nil, isDownIndirect: Bool = false
    ) throws {
        let useNAX = hal.caps.isMetal4TensorCapable
        guard hal.caps.supportsSimdgroupMMA else {
            throw QWGemmError.badShape("qwg_moe_nax_gemm_mxfp8_g32 requires Apple7+ simdgroup or Metal 4 NAX tensor support")
        }
        guard inDtype == .f16 || inDtype == .bf16 else {
            throw QWGemmError.badShape("MXFP8 NAX prefill requires .f16 or .bf16 input activation")
        }
        let nProj = nOverride ?? bank.rowsPerExpert
        guard nProj > 0, bank.k > 0, bank.k % 32 == 0 else {
            throw QWGemmError.badShape("MXFP8 NAX prefill requires k % 32 == 0 and n > 0")
        }
        guard tileCount.shape.elementCount >= tilePlan.scratchInts else {
            throw QWGemmError.badShape("tileCount buffer too small for MoETilePlan")
        }
        let colTiles = (nProj + 63) / 64
        let flags: UInt32 = (sortedTok != nil ? 2 : 0)
        let params = QWGemmFPMoENaxParams(
            experts: UInt32(bank.experts), n: UInt32(nProj), k: UInt32(bank.k),
            yStride: UInt32(yStride ?? nProj), yColOff: UInt32(yColOff),
            inDtype: inDtype.rawValue, outDtype: outDtype.rawValue, flags: flags)
        enc.setComputePipelineState(try hal.pipeline(useNAX ? "qwg_moe_nax_gemm_mxfp8_g32" : "qwg_moe_sg_gemm_mxfp8_g32"))
        bank.weights.bind(enc, index: 0)
        bank.scales.bind(enc, index: 1)
        x.bind(enc, index: 2)
        y.bind(enc, index: 3)
        enc.bindTensors(expertOff, tileE, tileR, tileCount, startIndex: 4)
        enc.setPod(params, index: 8)
        (sortedTok ?? expertOff).bind(enc, index: 10)
        let baseByteOffset = tileCount.region.base + tileCount.offset
            + (isDownIndirect ? tilePlan.downIndirectByteOffset : 16)
        QWGemm.dispatchTileChunks(
            enc, indirectBuffer: tileCount.region.buffer,
            baseByteOffset: baseByteOffset, colTiles: colTiles,
            maxTiles: tilePlan.maxTiles, flags: 0, flagsIndex: 9)
    }

    /// Compacts active experts from `expertOff` into `activeExperts` + indirect dispatch args (`48 B`).
    public static func encodeMoEBuildActiveBuckets(
        _ enc: any MTLComputeCommandEncoder, hal: QWHal,
        expertOff: QWFPBufferSlice, activeExperts: QWFPBufferSlice,
        indirectArgs: QWFPBufferSlice, plan: MoEActiveBucketPlan
    ) throws {
        if indirectArgs.bytes > 0 && indirectArgs.bytes < MoEActiveBucketPlan.indirectBufferBytes {
            throw QWGemmError.badShape("indirectArgs buffer requires at least 48 bytes")
        }
        if activeExperts.bytes > 0 && activeExperts.bytes < plan.experts * 4 {
            throw QWGemmError.badShape("activeExperts buffer too small for \(plan.experts) experts")
        }
        enc.setComputePipelineState(try hal.pipeline("qwg_moe_build_active_buckets"))
        expertOff.bind(enc, index: 0)
        activeExperts.bind(enc, index: 1)
        indirectArgs.bind(enc, index: 2)
        enc.setUInt32s(plan.experts, plan.gateUpGridX, plan.downGridX, startIndex: 3)
        enc.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Encodes compact active-expert `NVFP4_G16` fused gate+up+SwiGLU (split or concatenated bank).
    public static func encodeMoECompactNVFP4GateUpSwiGLU(
        _ enc: any MTLComputeCommandEncoder, hal: QWHal,
        x: QWFPBufferSlice, gate: NVFP4Bank, up: NVFP4Bank? = nil,
        expertOff: QWFPBufferSlice, activeExperts: QWFPBufferSlice,
        activeArgs: QWFPBufferSlice, sortedTok: QWFPBufferSlice,
        sortedPairIdx: QWFPBufferSlice, actOut: QWFPBufferSlice,
        intermediate: Int, topK: Int,
        inDtype: QWFPElemDtype, outDtype: QWFPElemDtype,
        gateCap: Float = 0, upCap: Float = 0,
        upConcatScales2: QWFPBufferSlice? = nil,
        sortedOrder: Bool = true, useIndirectDispatch: Bool = true,
        maxActiveExperts: Int? = nil, directT1: Bool = false
    ) throws {
        guard intermediate > 0, gate.k > 0, gate.k % 16 == 0 else {
            throw QWGemmError.badShape("NVFP4 compact gate_up requires k % 16 == 0")
        }
        let upBank = up ?? gate
        let isConcat = (up == nil)
        if isConcat && gate.rowsPerExpert < 2 * intermediate {
            throw QWGemmError.badShape("Concat gateUp NVFP4 bank rowsPerExpert \(gate.rowsPerExpert) < \(2 * intermediate)")
        }
        if isConcat && gate.splitGateUpScales2 {
            guard let s2 = gate.scales2, s2.bytes >= 2 * gate.experts * 4 else {
                throw QWGemmError.badShape("splitGateUpScales2 NVFP4 bank requires 2*experts s2")
            }
        }
        let hasS2 = gate.scales2 != nil && (up != nil ? up?.scales2 != nil : true)
        let flags: UInt32 = (hasS2 ? 1 : 0) | (sortedOrder ? 2 : 0) | (directT1 ? 4 : 0)
        let defaultUpS2 = gate.splitGateUpScales2 ? gate.scales2?.slice(byteOffset: gate.experts * 4) : gate.scales2
        let upS2Slice = up != nil ? upBank.scales2 : (upConcatScales2 ?? defaultUpS2)
        let params = QWGemmFPCompactMoEParams(
            n: UInt32(intermediate), k: UInt32(gate.k), topK: UInt32(topK),
            numExperts: UInt32(gate.experts),
            wExpertRows: UInt32(gate.rowsPerExpert),
            upRowBias: UInt32(isConcat ? intermediate : 0),
            inDtype: inDtype.rawValue, outDtype: outDtype.rawValue,
            flags: flags, gateCap: gateCap, upCap: upCap)
        let kName = directT1
            ? "qwg_moe_compact_nvfp4_gate_up_swiglu_t1"
            : "qwg_moe_compact_nvfp4_gate_up_swiglu"
        enc.setComputePipelineState(try hal.pipeline(kName))
        x.bind(enc, index: 0)
        gate.weights.bind(enc, index: 1)
        gate.scales.bind(enc, index: 2)
        (gate.scales2 ?? gate.scales).bind(enc, index: 3)
        upBank.weights.bind(enc, index: 4)
        upBank.scales.bind(enc, index: 5)
        (upS2Slice ?? upBank.scales).bind(enc, index: 6)
        expertOff.bind(enc, index: 7)
        activeExperts.bind(enc, index: 8)
        activeArgs.bind(enc, index: 9)
        sortedTok.bind(enc, index: 10)
        sortedPairIdx.bind(enc, index: 11)
        actOut.bind(enc, index: 12)
        enc.setPod(params, index: 13)
        dispatchCompactGrid(
            enc, activeArgs: activeArgs,
            indirectOffset: MoEActiveBucketPlan.gateUpIndirectByteOffset,
            gridX: (intermediate + 31) / 32,
            maxActive: directT1 ? topK : (maxActiveExperts ?? gate.experts),
            useIndirect: directT1 ? false : useIndirectDispatch)
    }

    /// Encodes compact active-expert `NVFP4_G16` down projection.
    public static func encodeMoECompactNVFP4Down(
        _ enc: any MTLComputeCommandEncoder, hal: QWHal,
        actIn: QWFPBufferSlice, down: NVFP4Bank,
        expertOff: QWFPBufferSlice, activeExperts: QWFPBufferSlice,
        activeArgs: QWFPBufferSlice, sortedPairIdx: QWFPBufferSlice,
        slotOut: QWFPBufferSlice, topK: Int,
        inDtype: QWFPElemDtype, outDtype: QWFPElemDtype,
        sortedOrder: Bool = true, useIndirectDispatch: Bool = true,
        maxActiveExperts: Int? = nil, directT1: Bool = false
    ) throws {
        guard down.rowsPerExpert > 0, down.k > 0, down.k % 16 == 0 else {
            throw QWGemmError.badShape("NVFP4 compact down requires k % 16 == 0")
        }
        let flags: UInt32 = (down.scales2 != nil ? 1 : 0) | (sortedOrder ? 2 : 0) | (directT1 ? 4 : 0)
        let params = QWGemmFPCompactMoEParams(
            n: UInt32(down.rowsPerExpert), k: UInt32(down.k), topK: UInt32(topK),
            numExperts: UInt32(down.experts), wExpertRows: UInt32(down.rowsPerExpert),
            upRowBias: 0, inDtype: inDtype.rawValue, outDtype: outDtype.rawValue,
            flags: flags, gateCap: 0, upCap: 0)
        let kName = directT1 ? "qwg_moe_compact_nvfp4_down_t1" : "qwg_moe_compact_nvfp4_down"
        enc.setComputePipelineState(try hal.pipeline(kName))
        actIn.bind(enc, index: 0)
        down.weights.bind(enc, index: 1)
        down.scales.bind(enc, index: 2)
        (down.scales2 ?? down.scales).bind(enc, index: 3)
        expertOff.bind(enc, index: 4)
        activeExperts.bind(enc, index: 5)
        activeArgs.bind(enc, index: 6)
        sortedPairIdx.bind(enc, index: 7)
        slotOut.bind(enc, index: 8)
        enc.setPod(params, index: 9)
        dispatchCompactGrid(
            enc, activeArgs: activeArgs,
            indirectOffset: MoEActiveBucketPlan.downIndirectByteOffset,
            gridX: (down.rowsPerExpert + 31) / 32,
            maxActive: directT1 ? topK : (maxActiveExperts ?? down.experts),
            useIndirect: directT1 ? false : useIndirectDispatch)
    }

    /// Encodes compact active-expert `MXFP8_G32` fused gate+up+SwiGLU (split or concatenated bank).
    public static func encodeMoECompactMXFP8GateUpSwiGLU(
        _ enc: any MTLComputeCommandEncoder, hal: QWHal,
        x: QWFPBufferSlice, gate: MXFP8Bank, up: MXFP8Bank? = nil,
        expertOff: QWFPBufferSlice, activeExperts: QWFPBufferSlice,
        activeArgs: QWFPBufferSlice, sortedTok: QWFPBufferSlice,
        sortedPairIdx: QWFPBufferSlice, actOut: QWFPBufferSlice,
        intermediate: Int, topK: Int,
        inDtype: QWFPElemDtype, outDtype: QWFPElemDtype,
        gateCap: Float = 0, upCap: Float = 0,
        sortedOrder: Bool = true, useIndirectDispatch: Bool = true,
        maxActiveExperts: Int? = nil, directT1: Bool = false
    ) throws {
        guard intermediate > 0, gate.k > 0, gate.k % 32 == 0 else {
            throw QWGemmError.badShape("MXFP8 compact gate_up requires k % 32 == 0")
        }
        let upBank = up ?? gate
        let isConcat = (up == nil)
        if isConcat && gate.rowsPerExpert < 2 * intermediate {
            throw QWGemmError.badShape("Concat gateUp MXFP8 bank rowsPerExpert \(gate.rowsPerExpert) < \(2 * intermediate)")
        }
        let flags: UInt32 = (sortedOrder ? 2 : 0) | (directT1 ? 4 : 0)
        let params = QWGemmFPCompactMoEParams(
            n: UInt32(intermediate), k: UInt32(gate.k), topK: UInt32(topK),
            numExperts: UInt32(gate.experts), wExpertRows: UInt32(gate.rowsPerExpert),
            upRowBias: UInt32(isConcat ? intermediate : 0),
            inDtype: inDtype.rawValue, outDtype: outDtype.rawValue,
            flags: flags, gateCap: gateCap, upCap: upCap)
        let kName = directT1
            ? "qwg_moe_compact_mxfp8_gate_up_swiglu_t1"
            : "qwg_moe_compact_mxfp8_gate_up_swiglu"
        enc.setComputePipelineState(try hal.pipeline(kName))
        x.bind(enc, index: 0)
        gate.weights.bind(enc, index: 1)
        gate.scales.bind(enc, index: 2)
        upBank.weights.bind(enc, index: 3)
        upBank.scales.bind(enc, index: 4)
        expertOff.bind(enc, index: 5)
        activeExperts.bind(enc, index: 6)
        activeArgs.bind(enc, index: 7)
        sortedTok.bind(enc, index: 8)
        sortedPairIdx.bind(enc, index: 9)
        actOut.bind(enc, index: 10)
        enc.setPod(params, index: 11)
        dispatchCompactGrid(
            enc, activeArgs: activeArgs,
            indirectOffset: MoEActiveBucketPlan.gateUpIndirectByteOffset,
            gridX: (intermediate + 31) / 32,
            maxActive: directT1 ? topK : (maxActiveExperts ?? gate.experts),
            useIndirect: directT1 ? false : useIndirectDispatch)
    }

    /// Encodes compact active-expert `MXFP8_G32` down projection.
    public static func encodeMoECompactMXFP8Down(
        _ enc: any MTLComputeCommandEncoder, hal: QWHal,
        actIn: QWFPBufferSlice, down: MXFP8Bank,
        expertOff: QWFPBufferSlice, activeExperts: QWFPBufferSlice,
        activeArgs: QWFPBufferSlice, sortedPairIdx: QWFPBufferSlice,
        slotOut: QWFPBufferSlice, topK: Int,
        inDtype: QWFPElemDtype, outDtype: QWFPElemDtype,
        sortedOrder: Bool = true, useIndirectDispatch: Bool = true,
        maxActiveExperts: Int? = nil, directT1: Bool = false
    ) throws {
        guard down.rowsPerExpert > 0, down.k > 0, down.k % 32 == 0 else {
            throw QWGemmError.badShape("MXFP8 compact down requires k % 32 == 0")
        }
        let flags: UInt32 = (sortedOrder ? 2 : 0) | (directT1 ? 4 : 0)
        let params = QWGemmFPCompactMoEParams(
            n: UInt32(down.rowsPerExpert), k: UInt32(down.k), topK: UInt32(topK),
            numExperts: UInt32(down.experts), wExpertRows: UInt32(down.rowsPerExpert),
            upRowBias: 0, inDtype: inDtype.rawValue, outDtype: outDtype.rawValue,
            flags: flags, gateCap: 0, upCap: 0)
        let kName = directT1 ? "qwg_moe_compact_mxfp8_down_t1" : "qwg_moe_compact_mxfp8_down"
        enc.setComputePipelineState(try hal.pipeline(kName))
        actIn.bind(enc, index: 0)
        down.weights.bind(enc, index: 1)
        down.scales.bind(enc, index: 2)
        expertOff.bind(enc, index: 3)
        activeExperts.bind(enc, index: 4)
        activeArgs.bind(enc, index: 5)
        sortedPairIdx.bind(enc, index: 6)
        slotOut.bind(enc, index: 7)
        enc.setPod(params, index: 8)
        dispatchCompactGrid(
            enc, activeArgs: activeArgs,
            indirectOffset: MoEActiveBucketPlan.downIndirectByteOffset,
            gridX: (down.rowsPerExpert + 31) / 32,
            maxActive: directT1 ? topK : (maxActiveExperts ?? down.experts),
            useIndirect: directT1 ? false : useIndirectDispatch)
    }

    /// Prefill epilogue SwiGLU supporting standard (`cap == 0`), softcapped (`cap > 0`), `f16`/`bf16`/`f32`, and optional FWHT-32.
    public static func encodeSwiGLUSoftcap(
        _ enc: any MTLComputeCommandEncoder, hal: QWHal,
        gu: QWFPBufferSlice, act: QWFPBufferSlice,
        rows: Int, intermediate: Int,
        gateCap: Float = 0, upCap: Float = 0,
        inDtype: QWFPElemDtype = .f16, outDtype: QWFPElemDtype = .f16,
        rotateFwht32: Bool = false
    ) throws {
        guard rows > 0, intermediate > 0 else { return }
        if rotateFwht32 && (intermediate % 32 != 0) {
            throw QWGemmError.badShape("qwg_swiglu_softcap_f16 rotateFwht32 requires intermediate % 32 == 0")
        }
        let flags: UInt32 = (inDtype == .bf16 ? 1 : 0)
            | (outDtype == .bf16 ? 2 : 0)
            | (inDtype == .f32 ? 4 : 0)
            | (outDtype == .f32 ? 8 : 0)
            | (rotateFwht32 ? 16 : 0)
        let params = QWGemmFPSwiGLUParams(
            m: UInt32(rows), i: UInt32(intermediate),
            gateCap: gateCap, upCap: upCap, flags: flags)
        enc.setComputePipelineState(try hal.pipeline("qwg_swiglu_softcap_f16"))
        gu.bind(enc, index: 0)
        act.bind(enc, index: 1)
        enc.setPod(params, index: 2)
        let nx = max(1, (intermediate + 7) / 8)
        enc.dispatchThreads(
            MTLSize(width: nx, height: rows, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(256, nx), height: 1, depth: 1))
    }

    /// Typed down-projection weighted reduction and optional gated shared-expert combine.
    public static func encodeMoEDownReduceCombineBF16(
        _ enc: any MTLComputeCommandEncoder, hal: QWHal,
        dnAcc: QWFPBufferSlice, invPerm: QWFPBufferSlice?,
        weights: QWFPBufferSlice, shared: QWFPBufferSlice?,
        gate: QWFPBufferSlice?, out: QWFPBufferSlice,
        hidden: Int, topK: Int, tokens: Int,
        inDtype: QWFPElemDtype, outDtype: QWFPElemDtype,
        sharedDtype: QWFPElemDtype = .bf16,
        ids: QWFPBufferSlice? = nil
    ) throws {
        guard hidden > 0, topK > 0, tokens > 0 else { return }
        var flags: UInt32 = 0
        if invPerm == nil { flags |= 1 }
        if shared == nil { flags |= 2 }
        if gate != nil && shared != nil { flags |= 8 }
        if ids != nil { flags |= 16 }
        let params = QWGemmFPCombineParams(
            hidden: UInt32(hidden), topK: UInt32(topK), tokens: UInt32(tokens),
            inDtype: inDtype.rawValue, outDtype: outDtype.rawValue,
            sharedDtype: sharedDtype.rawValue, flags: flags)
        enc.setComputePipelineState(try hal.pipeline("qwg_moe_down_reduce_combine_bf16"))
        dnAcc.bind(enc, index: 0)
        (invPerm ?? dnAcc).bind(enc, index: 1)
        weights.bind(enc, index: 2)
        (shared ?? out).bind(enc, index: 3)
        (gate ?? weights).bind(enc, index: 4)
        out.bind(enc, index: 5)
        enc.setPod(params, index: 6)
        (ids ?? weights).bind(enc, index: 7)
        let vecGroups = (hidden + 3) / 4
        let gridX = (vecGroups + 255) / 256
        enc.dispatchThreadgroups(
            MTLSize(width: gridX, height: tokens, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    private static func dispatchCompactGrid(
        _ enc: any MTLComputeCommandEncoder,
        activeArgs: QWFPBufferSlice, indirectOffset: Int,
        gridX: Int, maxActive: Int, useIndirect: Bool
    ) {
        let tpg = MTLSize(width: 256, height: 1, depth: 1)
        if useIndirect {
            enc.dispatchThreadgroups(
                indirectBuffer: activeArgs.buffer,
                indirectBufferOffset: activeArgs.offset + indirectOffset,
                threadsPerThreadgroup: tpg)
        } else {
            enc.dispatchThreadgroups(
                MTLSize(width: max(1, gridX), height: max(1, maxActive), depth: 1),
                threadsPerThreadgroup: tpg)
        }
    }
}
