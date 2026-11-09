// QWGemmMoE.swift — Expert-sorted grouped MoE, NAX/MPP grouped MoE, per-pair MoE, and shared-expert combine encoders.
import Foundation
import Metal
import QWAlloc
import QWCore
import QWHAL

struct QWNaxGroupedParams {
    var experts: UInt32
    var n: UInt32
    var k: UInt32
    var blocksPerRow: UInt32
    var rowTiles: UInt32
    var colTiles: UInt32
}

extension QWGemm {
    public enum MoETileFilter: UInt32 {
        case all = 0
        case residentOnly = 0x40
        case coldOnly = 0x80
    }

    /// Expert-sorted pair order (counting sort of (token, route) pairs by expert id), with optional fused NAX tile list build.
    public static func encodeMoEGroupPairsF32(_ enc: any MTLComputeCommandEncoder, hal: QWHal, ids: QWTensorValue,
                                              expertOff: QWTensorValue, sortedPairIdx: QWTensorValue,
                                              sortedTok: QWTensorValue, experts: Int, totalPairs: Int,
                                              routes: Int, invPerm: QWTensorValue? = nil,
                                              tileE: QWTensorValue? = nil, tileR: QWTensorValue? = nil,
                                              tileCount: QWTensorValue? = nil) throws {
        precondition(experts <= 512, "the expert-sorted pair order holds 512 experts")
        enc.setComputePipelineState(try hal.pipeline("qwg_moe_group_pairs_f32"))
        enc.bindTensors(ids, expertOff, sortedPairIdx, sortedTok, startIndex: 0)
        enc.setUInt32s(experts, totalPairs, routes, startIndex: 4)
        (invPerm ?? sortedPairIdx).bind(enc, index: 7)
        enc.setUInt32s(invPerm != nil ? 1 : 0, startIndex: 8)
        let hasTiles = tileE != nil && tileR != nil && tileCount != nil
        (tileE ?? sortedPairIdx).bind(enc, index: 9)
        (tileR ?? sortedPairIdx).bind(enc, index: 10)
        (tileCount ?? sortedPairIdx).bind(enc, index: 11)
        enc.setUInt32s(hasTiles ? 1 : 0, startIndex: 12)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Dynamic-residency span map (`qwg_moe_span_ids`): router ids → resident bank slot, or
    /// `residentExperts` + the miss's rank in expert-id order, or -1 for an invalid id.
    /// `slotMap` holds one Int32 per physical expert (bank slot or -1).
    public static func encodeMoESpanIds(_ enc: any MTLComputeCommandEncoder, hal: QWHal, ids: QWTensorValue,
                                        slotMap: any MTLBuffer, out: QWTensorValue, count: Int,
                                        experts: Int, residentExperts: Int) throws {
        precondition(experts <= 512, "the span map holds 512 experts")
        enc.setComputePipelineState(try hal.pipeline("qwg_moe_span_ids"))
        ids.bind(enc, index: 0)
        enc.setBuffer(slotMap, offset: 0, index: 1)
        out.bind(enc, index: 2)
        enc.setUInt32s(count, experts, residentExperts, startIndex: 3)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
    }

    /// Routed gate_up with one output row per simdgroup (4 simdgroups = 128 threads = 4 rows per TG),
    /// or fused 32-channel paired gate+up + SwiGLU + H_32 when `fuseSiluFwht` is true.
    /// Multi-token expert reuse can be disabled when independent rows have better occupancy.
    public static func encodeMoEGateUpF32R1(_ enc: any MTLComputeCommandEncoder, hal: QWHal, x: QWTensorValue,
                                            bank: QWWeightView, ids: QWTensorValue, out: QWTensorValue,
                                            rowsPerExpert: Int, routes: Int, tokens: Int,
                                            fuseSiluFwht: Bool = false,
                                            coldBank: QWWeightView? = nil,
                                            tileFilter: MoETileFilter = .all,
                                            reuseExpertsAcrossTokens: Bool = true) throws {
        let intermediate = rowsPerExpert / 2
        let useFused = fuseSiluFwht && intermediate > 0 && (intermediate % 32 == 0)
        let residentE = coldBank != nil ? (bank.n / max(1, rowsPerExpert)) : 0
        let filterBits = Int(tileFilter.rawValue) << 8
        let dedupToks = (useFused && reuseExpertsAcrossTokens && (2...4).contains(tokens)) ? ((tokens & 0x7) << 10) : 0
        let eBase = (routes & 0x3FF) | dedupToks | filterBits | ((residentE & 0x7FFF) << 16)
        let ePacked = useFused ? (eBase | 0x8000_0000) : eBase
        if bank.isIncumbentContainer {
            enc.setComputePipelineState(try hal.pipeline("qwg_moe_gateup_q4g_f32_r1"))
        } else {
            guard let aqName = QWGemmAQ.moeGateUpF32R1KernelName(bits: bank.bits, group: bank.groupSize) else {
                throw QWGemmError.badWeights("no affine-quantised moe gateup r1 kernel for \(bank.bits)-bit group \(bank.groupSize)")
            }
            enc.setComputePipelineState(try hal.pipeline(aqName))
        }
        enc.bindTensors(bank.blocks, x, ids, out, startIndex: 0)
        enc.setUInt32s(rowsPerExpert, bank.k, ePacked, startIndex: 4)
        (coldBank ?? bank).blocks.bind(enc, index: 7)
        if useFused {
            enc.dispatchThreadgroups(MTLSize(width: intermediate / 32, height: routes, depth: tokens),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        } else {
            enc.dispatchThreadgroups(MTLSize(width: (rowsPerExpert + 3) / 4, height: routes, depth: tokens),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        }
    }

    /// Routed down + shared-expert combine, f32 activations (4 simdgroups = 128 threads = 16 rows per TG).
    /// Multi-token expert reuse can be disabled when independent rows have better occupancy.
    public static func encodeMoEDownSharedCombineF32(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                                     x: QWTensorValue, bank: QWWeightView,
                                                     rowsPerRoute: Int, routes: Int,
                                                     weights: QWTensorValue, shared: QWTensorValue,
                                                     gate: QWTensorValue, out: QWTensorValue,
                                                     ids: QWTensorValue, tokens: Int,
                                                     out16: QWTensorValue? = nil,
                                                     coldBank: QWWeightView? = nil,
                                                     tileFilter: MoETileFilter = .all,
                                                     routeOutput: Bool = false,
                                                     reuseExpertsAcrossTokens: Bool = true) throws {
        if bank.isIncumbentContainer {
            enc.setComputePipelineState(try hal.pipeline("qwg_moe_down_shared_combine_q4g_f32"))
        } else {
            guard let aqName = QWGemmAQ.moeDownSharedCombineF32KernelName(bits: bank.bits, group: bank.groupSize) else {
                throw QWGemmError.badWeights("no affine-quantised moe down combine kernel for \(bank.bits)-bit group \(bank.groupSize)")
            }
            enc.setComputePipelineState(try hal.pipeline(aqName))
        }
        let residentE = coldBank != nil ? (bank.n / max(1, rowsPerRoute)) : 0
        let filterMode: Int
        switch tileFilter {
        case .all:          filterMode = 0
        case .residentOnly: filterMode = 1
        case .coldOnly:     filterMode = 2
        }
        let dedupToks = (reuseExpertsAcrossTokens && (2...4).contains(tokens)) ? ((tokens & 0x7) << 8) : 0
        let ePacked = (routes & 0xFF) | dedupToks | ((residentE & 0x3FFF) << 16) | (filterMode << 30)
        enc.bindTensors(bank.blocks, x, weights, shared, gate, out, startIndex: 0)
        enc.setUInt32s(rowsPerRoute, bank.k, ePacked, startIndex: 6)
        ids.bind(enc, index: 9)
        (out16 ?? out).bind(enc, index: 10)
        enc.setUInt32s((out16 != nil ? 1 : 0) | (routeOutput ? 2 : 0), startIndex: 11)
        (coldBank ?? bank).blocks.bind(enc, index: 12)
        enc.dispatchThreadgroups(MTLSize(width: (rowsPerRoute + 15) / 16, height: dedupToks != 0 ? 1 : tokens, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    public static func encodeMoEGateUpNax(_ enc: any MTLComputeCommandEncoder, hal: QWHal, xe: QWTensorValue,
                                          bank: QWWeightView, expertOff: QWTensorValue, tileE: QWTensorValue,
                                          tileR: QWTensorValue, tileCount: QWTensorValue, gu16: QWTensorValue,
                                          rowsPerExpert: Int, pairs: Int,
                                          sortedTok: QWTensorValue? = nil,
                                          coldBank: QWWeightView? = nil,
                                          tileFilter: MoETileFilter = .all) throws {
        precondition(rowsPerExpert % 64 == 0 && bank.k % 64 == 0, "NAX gate/up needs N and K multiples of 64")
        let residentE = bank.n / rowsPerExpert
        let coldE = (coldBank?.n ?? 0) / rowsPerExpert
        let experts = residentE + coldE
        let packedExperts = coldBank != nil
            ? ((UInt32(residentE & 0xFFFF) << 16) | UInt32(experts & 0xFFFF))
            : UInt32(experts)
        let maxTiles = min(pairs, min(experts, pairs) + (pairs + 63) / 64)  // ≤1 tile per expert ≤32 rows, else ⌈c/64⌉
        let grp = QWNaxGroupedParams(experts: packedExperts, n: UInt32(rowsPerExpert), k: UInt32(bank.k),
                                     blocksPerRow: UInt32(bank.k / max(1, bank.groupSize)), rowTiles: 0, colTiles: UInt32(rowsPerExpert / 64))
        if bank.isIncumbentContainer {
            enc.setComputePipelineState(try hal.pipeline("qwg_moe_nax_gemm"))
        } else {
            guard let aqName = QWGemmAQ.moeNaxGemmKernelName(bits: bank.bits, group: bank.groupSize) else {
                throw QWGemmError.badWeights("no affine-quantised moe nax gemm kernel for \(bank.bits)-bit group \(bank.groupSize)")
            }
            enc.setComputePipelineState(try hal.pipeline(aqName))
        }
        enc.bindTensors(bank.blocks, xe, gu16, expertOff, tileE, tileR, tileCount, startIndex: 0)
        enc.setPod(grp, index: 7)
        enc.bindTensors(sortedTok ?? tileE, startIndex: 9)
        enc.setUInt32s(sortedTok != nil ? 1 : 0, startIndex: 10)
        (coldBank ?? bank).blocks.bind(enc, index: 11)
        dispatchTileChunks(enc, colTiles: rowsPerExpert / 64, maxTiles: maxTiles, flags: tileFilter.rawValue, flagsIndex: 8)
    }

    /// Grouped NAX/MPP GEMM with a hi/lo split activation and an fp32 output in expert-major rows.
    public static func encodeMoENaxSplitF32(_ enc: any MTLComputeCommandEncoder, hal: QWHal, xHi: QWTensorValue,
                                            xLo: QWTensorValue, bank: QWWeightView, expertOff: QWTensorValue,
                                            tileE: QWTensorValue, tileR: QWTensorValue, tileCount: QWTensorValue,
                                            y32: QWTensorValue, rowsPerExpert: Int, pairs: Int,
                                            coldBank: QWWeightView? = nil,
                                            tileFilter: MoETileFilter = .all) throws {
        precondition(rowsPerExpert % 64 == 0 && bank.k % 64 == 0, "NAX split needs N and K multiples of 64")
        let residentE = bank.n / rowsPerExpert
        let coldE = (coldBank?.n ?? 0) / rowsPerExpert
        let experts = residentE + coldE
        let packedExperts = coldBank != nil
            ? ((UInt32(residentE & 0xFFFF) << 16) | UInt32(experts & 0xFFFF))
            : UInt32(experts)
        let maxTiles = min(pairs, min(experts, pairs) + (pairs + 63) / 64)  // ≤1 tile per expert ≤32 rows, else ⌈c/64⌉
        let grp = QWNaxGroupedParams(experts: packedExperts, n: UInt32(rowsPerExpert), k: UInt32(bank.k),
                                     blocksPerRow: UInt32(bank.k / max(1, bank.groupSize)), rowTiles: 0, colTiles: UInt32(rowsPerExpert / 64))
        if bank.isIncumbentContainer {
            enc.setComputePipelineState(try hal.pipeline("qwg_moe_nax_split_f32"))
        } else {
            guard let aqName = QWGemmAQ.moeNaxSplitF32KernelName(bits: bank.bits, group: bank.groupSize) else {
                throw QWGemmError.badWeights("no affine-quantised moe nax split kernel for \(bank.bits)-bit group \(bank.groupSize)")
            }
            enc.setComputePipelineState(try hal.pipeline(aqName))
        }
        enc.bindTensors(bank.blocks, xHi, xLo, y32, expertOff, tileE, tileR, tileCount, startIndex: 0)
        enc.setPod(grp, index: 8)
        (coldBank ?? bank).blocks.bind(enc, index: 10)
        let tbFlags: UInt32 = ((y32.shape.dtype == .f16) ? 0x8000_0000 : 0) | tileFilter.rawValue
        dispatchTileChunks(enc, colTiles: rowsPerExpert / 64, maxTiles: maxTiles, flags: tbFlags, flagsIndex: 9)
    }

    /// Grouped NAX tiles in bounded dispatches: one huge grid costs Flash-Next r8k ~5% GPU time vs
    /// ~2560-threadgroup slices (35B-A3B neutral; fewer TGs per slice underfills its narrow gate/up). The slice's first tile index rides in bits 8..30 of the tileBlock word.
    static func dispatchTileChunks(_ enc: any MTLComputeCommandEncoder, colTiles: Int, maxTiles: Int,
                                   flags: UInt32, flagsIndex: Int) {
        let chunk = max(1, 2560 / colTiles)
        var base = 0
        while base < maxTiles {
            let n = min(chunk, maxTiles - base)
            var word = flags | (UInt32(base) << 8)
            enc.setBytes(&word, length: 4, index: flagsIndex)
            enc.dispatchThreadgroups(MTLSize(width: colTiles, height: n, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            base += n
        }
    }

    /// Fused expert-sorted down reduction + shared expert combine using inverse permutation invPerm[pair] = sortedSlot.
    public static func encodeMoEDownReduceCombineF32(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                                     dnAcc: QWTensorValue, invPerm: QWTensorValue?,
                                                     weights: QWTensorValue, shared: QWTensorValue,
                                                     gate: QWTensorValue, out: QWTensorValue,
                                                     hidden: Int, routes: Int, tokens: Int,
                                                     out16: QWTensorValue? = nil) throws {
        enc.setComputePipelineState(try hal.pipeline("qwg_moe_down_reduce_combine_f32"))
        dnAcc.bind(enc, index: 0)
        (invPerm ?? weights).bind(enc, index: 1)
        enc.bindTensors(weights, shared, gate, out, startIndex: 2)
        enc.setUInt32s(hidden, routes, tokens, startIndex: 6)
        (out16 ?? out).bind(enc, index: 9)
        let flags = (out16 != nil ? 1 : 0) | (dnAcc.shape.dtype == .f16 ? 2 : 0) | (invPerm == nil ? 4 : 0)
        enc.setUInt32s(flags, startIndex: 10)
        let cols4 = (hidden + 3) / 4
        enc.dispatchThreadgroups(MTLSize(width: (cols4 + 255) / 256, height: tokens, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// The standard MoE path's **f32 bridge**: the f16 normed hidden becomes
    /// the rotated f32 copy the expert GEMMs read and the un-rotated f32 copy
    /// the router reads. The hidden obeys the norm family's law.
    public static func encodeMoEBridgeF16ToF32(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                                x: QWTensorValue, xRot32: QWTensorValue,
                                                xPlain32: QWTensorValue,
                                                rows: Int, hidden: Int,
                                                skipRot32: Bool = false) throws {
        // The dense engine's residual stream is f16 and the MoE operator chain
        // is f32-native, so the chain needs a widening the engine does not
        // have. One pass over the f16 normed hidden produces the two f32
        // copies the chain reads: the **rotated** one (the expert GEMMs' input
        // — the packer rotates every weight's input dimension) and the
        // **un-rotated** one (the router's input — the router's weight is the
        // packer's raw plane).
        //
        // `hidden % 256 == 0` is the norm family's law: the bridge's butterfly
        // needs its 4-lane groups fully populated and all 32 lanes iterating
        // alike.
        // The law and the buffers are two different failures: a buffer
        // that is too small must not advertise the 256 law.
        guard rows > 0, hidden > 0 else {
            throw QWGemmError.badGeometry("moe bridge: rows=\(rows) hidden=\(hidden)")
        }
        guard hidden % 256 == 0 else {
            throw QWGemmError.badGeometry(
                "moe bridge: hidden \(hidden) must be a multiple of 256")
        }
        guard x.shape.dtype == .f16, xRot32.shape.dtype == .f32, xPlain32.shape.dtype == .f32,
              x.shape.elementCount >= rows * hidden,
              xRot32.shape.elementCount >= rows * hidden,
              xPlain32.shape.elementCount >= rows * hidden else {
            throw QWGemmError.badShape(
                "moe bridge: rows=\(rows) hidden=\(hidden) do not fit the buffers")
        }
        enc.setComputePipelineState(try hal.pipeline("qwg_moe_bridge_f16_to_f32"))
        enc.bindTensors(x, xRot32, xPlain32, startIndex: 0)
        let hPacked = skipRot32 ? (hidden | 0x8000_0000) : hidden
        enc.setUInt32s(hPacked, startIndex: 3)
        // One row: a 32-lane threadgroup per 256 hidden (lane-aligned, bit-identical) so decode is not latency-bound.
        let blocks = rows == 1 ? hidden / 256 : 1
        enc.dispatchThreadgroups(MTLSize(width: blocks, height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }
}
