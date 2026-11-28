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

    public static func moeMaxTiles(experts: Int, pairs: Int) -> Int {
        min(pairs, min(experts, pairs) + (pairs + 63) / 64)
    }

    public static func moeSliceCount(colTiles: Int, maxTiles: Int) -> Int {
        guard colTiles > 0, maxTiles > 0 else { return 0 }
        let chunk = max(1, 2560 / colTiles)
        return (maxTiles + chunk - 1) / chunk
    }

    /// Single source of truth for grouped-NAX MoE active-tile indirect dispatch layout.
    public struct MoETilePlan: Sendable {
        public let experts: Int
        public let pairs: Int
        public let maxTiles: Int
        public let guColTiles: Int
        public let dnColTiles: Int
        public let guSlices: Int
        public let dnSlices: Int
        public var scratchInts: Int { 4 + (guSlices + dnSlices) * 3 }
        public var downIndirectByteOffset: Int { 16 + guSlices * 12 }
        var packedWriteWord: Int {
            1 | ((guColTiles & 0x7FFF) << 1) | ((dnColTiles & 0xFFFF) << 16)
        }

        public init(experts: Int, pairs: Int, gateUpRows: Int, downRows: Int) {
            self.experts = experts
            self.pairs = pairs
            let m = QWGemm.moeMaxTiles(experts: experts, pairs: pairs)
            let guCol = (gateUpRows + 63) / 64
            let dnCol = (downRows + 63) / 64
            self.maxTiles = m
            self.guColTiles = guCol
            self.dnColTiles = dnCol
            self.guSlices = QWGemm.moeSliceCount(colTiles: guCol, maxTiles: m)
            self.dnSlices = QWGemm.moeSliceCount(colTiles: dnCol, maxTiles: m)
        }
    }

    private static let disableMoETailCompact: Bool =
        ProcessInfo.processInfo.environment["QW_MOE_TAIL_COMPACT"] == "0"

    /// Expert-sorted pair order (counting sort of (token, route) pairs by expert id), with optional fused NAX tile list build.
    public static func encodeMoEGroupPairsF32(_ enc: any MTLComputeCommandEncoder, hal: QWHal, ids: QWTensorValue,
                                              expertOff: QWTensorValue, sortedPairIdx: QWTensorValue,
                                              sortedTok: QWTensorValue, experts: Int, totalPairs: Int,
                                              routes: Int, invPerm: QWTensorValue? = nil,
                                              tileE: QWTensorValue? = nil, tileR: QWTensorValue? = nil,
                                              tileCount: QWTensorValue? = nil,
                                              tilePlan: MoETilePlan? = nil,
                                              directSlotTok: Bool = false) throws {
        precondition(experts <= 1024, "the expert-sorted pair order holds 1024 experts")
        let hasTiles = tileE != nil && tileR != nil && tileCount != nil
        var writeWord = 0
        if hasTiles {
            guard let plan = tilePlan, let tc = tileCount, tc.shape.elementCount >= plan.scratchInts else {
                throw QWGemmError.badShape("moe group pairs: tileCount scratch does not fit MoETilePlan")
            }
            writeWord = plan.packedWriteWord
        }
        enc.setComputePipelineState(try hal.pipeline("qwg_moe_group_pairs_f32"))
        enc.bindTensors(ids, expertOff, sortedPairIdx, sortedTok, startIndex: 0)
        enc.setUInt32s(experts, totalPairs, routes, startIndex: 4)
        (invPerm ?? sortedPairIdx).bind(enc, index: 7)
        let writeWordInv = (invPerm != nil ? 1 : 0)
            | (directSlotTok ? 2 : 0)
            | (disableMoETailCompact ? 4 : 0)
        enc.setUInt32s(writeWordInv, startIndex: 8)
        (tileE ?? sortedPairIdx).bind(enc, index: 9)
        (tileR ?? sortedPairIdx).bind(enc, index: 10)
        (tileCount ?? sortedPairIdx).bind(enc, index: 11)
        enc.setUInt32s(writeWord, startIndex: 12)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Dynamic-residency span map (`qwg_moe_span_ids`): router ids → resident bank slot, or
    /// `residentExperts` + the miss's rank in expert-id order, or -1 for an invalid id.
    /// `slotMap` holds one Int32 per physical expert (bank slot or -1).
    public static func encodeMoESpanIds(_ enc: any MTLComputeCommandEncoder, hal: QWHal, ids: QWTensorValue,
                                        slotMap: any MTLBuffer, out: QWTensorValue, count: Int,
                                        experts: Int, residentExperts: Int) throws {
        precondition(experts <= 1024, "the span map holds 1024 experts")
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
                                            reuseExpertsAcrossTokens: Bool = true,
                                            gateSoftcap: Float = 0, upSoftcap: Float = 0,
                                            swigluClamp: QWSwiGLUClampSpec? = nil) throws {
        let intermediate = rowsPerExpert / 2
        let useFused = fuseSiluFwht && intermediate > 0 && (intermediate % 32 == 0)
        let residentE = coldBank != nil ? (bank.n / max(1, rowsPerExpert)) : 0
        let filterBits = Int(tileFilter.rawValue) << 8
        let dedupToks = (useFused && reuseExpertsAcrossTokens && (2...16).contains(tokens)) ? (((tokens - 1) & 0xF) << 10) : 0
        let eBase = (routes & 0x3FF) | dedupToks | filterBits | ((residentE & 0x7FFF) << 16)
        let ePacked = useFused ? (eBase | 0x8000_0000) : eBase
        let hasSoftcap = (gateSoftcap > 0 || upSoftcap > 0)
        let clamp = (useFused && !hasSoftcap) ? swigluClamp : nil
        if bank.isIncumbentContainer {
            enc.setComputePipelineState(try hal.pipeline(
                hasSoftcap ? "qwg_moe_gateup_q4g_f32_r1_sc" : "qwg_moe_gateup_q4g_f32_r1",
                swigluClamp: clamp))
        } else {
            guard let aqName = QWGemmAQ.moeGateUpF32R1KernelName(bits: bank.bits, group: bank.groupSize,
                                                                 isSHVQ: bank.isSHVQ, hasSoftcap: hasSoftcap) else {
                throw QWGemmError.badWeights("no affine-quantised moe gateup r1 kernel for \(bank.bits)-bit group \(bank.groupSize)")
            }
            enc.setComputePipelineState(try hal.pipeline(aqName, swigluClamp: clamp))
        }
        enc.bindTensors(bank.blocks, x, ids, out, startIndex: 0)
        enc.setUInt32s(rowsPerExpert, bank.k, ePacked, startIndex: 4)
        (coldBank ?? bank).blocks.bind(enc, index: 7)
        if hasSoftcap {
            var softcaps = SIMD2<Float>(gateSoftcap, upSoftcap)
            enc.setBytes(&softcaps, length: MemoryLayout<SIMD2<Float>>.stride, index: 8)
        }
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
                                                     reuseExpertsAcrossTokens: Bool = true,
                                                     skipShared: Bool = false,
                                                     numGroups: Int = 0,
                                                     tokenIndexedX: Bool = false) throws {
        if bank.isIncumbentContainer {
            enc.setComputePipelineState(try hal.pipeline("qwg_moe_down_shared_combine_q4g_f32"))
        } else {
            guard let aqName = QWGemmAQ.moeDownSharedCombineF32KernelName(bits: bank.bits, group: bank.groupSize,
                                                                          isSHVQ: bank.isSHVQ) else {
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
        let dedupToks = (numGroups == 0 && reuseExpertsAcrossTokens && (2...4).contains(tokens)) ? ((tokens & 0x7) << 8) : 0
        let ePacked = (routes & 0xFF) | dedupToks | ((numGroups & 0x1F) << 11) | ((residentE & 0x3FFF) << 16) | (filterMode << 30)
        enc.bindTensors(bank.blocks, x, weights, shared, gate, out, startIndex: 0)
        enc.setUInt32s(rowsPerRoute, bank.k, ePacked, startIndex: 6)
        ids.bind(enc, index: 9)
        (out16 ?? out).bind(enc, index: 10)
        let combineFlags = (out16 != nil ? 1 : 0) | (routeOutput ? 2 : 0) | (skipShared ? 4 : 0)
            | (tokenIndexedX ? 8 : 0) | (out16?.shape.dtype == .bf16 ? 16 : 0)
        enc.setUInt32s(combineFlags, startIndex: 11)
        (coldBank ?? bank).blocks.bind(enc, index: 12)
        let tgRows = numGroups > 0 ? (tokens * numGroups) : (dedupToks != 0 ? 1 : tokens)
        enc.dispatchThreadgroups(MTLSize(width: (rowsPerRoute + 15) / 16, height: tgRows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    public static func encodeMoEGateUpNax(_ enc: any MTLComputeCommandEncoder, hal: QWHal, xe: QWTensorValue,
                                          bank: QWWeightView, expertOff: QWTensorValue, tileE: QWTensorValue,
                                          tileR: QWTensorValue, tileCount: QWTensorValue, gu16: QWTensorValue,
                                          rowsPerExpert: Int, tilePlan: MoETilePlan,
                                          sortedTok: QWTensorValue? = nil,
                                          coldBank: QWWeightView? = nil,
                                          tileFilter: MoETileFilter = .all) throws {
        precondition(rowsPerExpert % 64 == 0 && bank.k % 64 == 0, "NAX gate/up needs N and K multiples of 64")
        guard tileCount.shape.elementCount >= 4 + tilePlan.guSlices * 3 else {
            throw QWGemmError.badShape("moe gateup nax: tileCount buffer too small for indirect slices")
        }
        let residentE = bank.n / rowsPerExpert
        let coldE = (coldBank?.n ?? 0) / rowsPerExpert
        let experts = residentE + coldE
        let packedExperts = coldBank != nil
            ? ((UInt32(residentE & 0xFFFF) << 16) | UInt32(experts & 0xFFFF))
            : UInt32(experts)
        let colTiles = rowsPerExpert / 64
        let grp = QWNaxGroupedParams(experts: packedExperts, n: UInt32(rowsPerExpert), k: UInt32(bank.k),
                                     blocksPerRow: UInt32(bank.k / max(1, bank.groupSize)), rowTiles: 0, colTiles: UInt32(colTiles))
        let useNAX = hal.caps.isMetal4TensorCapable
        if bank.isIncumbentContainer {
            enc.setComputePipelineState(try hal.pipeline(useNAX ? "qwg_moe_nax_gemm" : "qwg_moe_sg_gemm"))
        } else {
            guard let aqName = QWGemmAQ.moeNaxGemmKernelName(bits: bank.bits, group: bank.groupSize,
                                                             isSHVQ: bank.isSHVQ, useNAX: useNAX) else {
                throw QWGemmError.badWeights("no affine-quantised moe nax gemm kernel for \(bank.bits)-bit group \(bank.groupSize)")
            }
            enc.setComputePipelineState(try hal.pipeline(aqName))
        }
        enc.bindTensors(bank.blocks, xe, gu16, expertOff, tileE, tileR, tileCount, startIndex: 0)
        enc.setPod(grp, index: 7)
        enc.bindTensors(sortedTok ?? tileE, startIndex: 9)
        enc.setUInt32s(sortedTok != nil ? 1 : 0, startIndex: 10)
        (coldBank ?? bank).blocks.bind(enc, index: 11)
        dispatchTileChunks(enc, indirectBuffer: tileCount.region.buffer,
                           baseByteOffset: tileCount.region.base + tileCount.offset + 16,
                           colTiles: colTiles, maxTiles: tilePlan.maxTiles,
                           flags: tileFilter.rawValue, flagsIndex: 8)
    }

    /// Grouped NAX/MPP GEMM with a hi/lo split activation and an fp32 output in expert-major rows.
    public static func encodeMoENaxSplitF32(_ enc: any MTLComputeCommandEncoder, hal: QWHal, xHi: QWTensorValue,
                                            xLo: QWTensorValue, bank: QWWeightView, expertOff: QWTensorValue,
                                            tileE: QWTensorValue, tileR: QWTensorValue, tileCount: QWTensorValue,
                                            y32: QWTensorValue, rowsPerExpert: Int, tilePlan: MoETilePlan,
                                            coldBank: QWWeightView? = nil,
                                            tileFilter: MoETileFilter = .all) throws {
        precondition(rowsPerExpert % 64 == 0 && bank.k % 64 == 0, "NAX split needs N and K multiples of 64")
        guard tileCount.shape.elementCount >= tilePlan.scratchInts else {
            throw QWGemmError.badShape("moe down nax: tileCount buffer too small for indirect slices")
        }
        let residentE = bank.n / rowsPerExpert
        let coldE = (coldBank?.n ?? 0) / rowsPerExpert
        let experts = residentE + coldE
        let packedExperts = coldBank != nil
            ? ((UInt32(residentE & 0xFFFF) << 16) | UInt32(experts & 0xFFFF))
            : UInt32(experts)
        let colTiles = rowsPerExpert / 64
        let grp = QWNaxGroupedParams(experts: packedExperts, n: UInt32(rowsPerExpert), k: UInt32(bank.k),
                                     blocksPerRow: UInt32(bank.k / max(1, bank.groupSize)), rowTiles: 0, colTiles: UInt32(colTiles))
        let useNAX = hal.caps.isMetal4TensorCapable
        if bank.isIncumbentContainer {
            enc.setComputePipelineState(try hal.pipeline(useNAX ? "qwg_moe_nax_split_f32" : "qwg_moe_sg_split_f32"))
        } else {
            guard let aqName = QWGemmAQ.moeNaxSplitF32KernelName(bits: bank.bits, group: bank.groupSize,
                                                                 isSHVQ: bank.isSHVQ, useNAX: useNAX) else {
                throw QWGemmError.badWeights("no affine-quantised moe nax split kernel for \(bank.bits)-bit group \(bank.groupSize)")
            }
            enc.setComputePipelineState(try hal.pipeline(aqName))
        }
        enc.bindTensors(bank.blocks, xHi, xLo, y32, expertOff, tileE, tileR, tileCount, startIndex: 0)
        enc.setPod(grp, index: 8)
        (coldBank ?? bank).blocks.bind(enc, index: 10)
        let tbFlags: UInt32 = ((y32.shape.dtype == .f16) ? 0x8000_0000 : 0) | tileFilter.rawValue
        let tcBase = tileCount.region.base + tileCount.offset
        dispatchTileChunks(enc, indirectBuffer: tileCount.region.buffer,
                           baseByteOffset: tcBase + tilePlan.downIndirectByteOffset,
                           colTiles: colTiles, maxTiles: tilePlan.maxTiles,
                           flags: tbFlags, flagsIndex: 9)
    }

    /// Grouped NAX tiles in bounded-TG indirect slices (`0` empty threadgroups).
    static func dispatchTileChunks(_ enc: any MTLComputeCommandEncoder, indirectBuffer: any MTLBuffer,
                                   baseByteOffset: Int, colTiles: Int, maxTiles: Int,
                                   flags: UInt32, flagsIndex: Int) {
        let chunk = max(1, 2560 / colTiles)
        var base = 0
        var slice = 0
        while base < maxTiles {
            var word = flags | (UInt32(base) << 8)
            enc.setBytes(&word, length: 4, index: flagsIndex)
            enc.dispatchThreadgroups(indirectBuffer: indirectBuffer,
                                     indirectBufferOffset: baseByteOffset + slice * 12,
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            base += min(chunk, maxTiles - base)
            slice += 1
        }
    }

    /// Fused expert-sorted down reduction + shared expert combine using inverse permutation invPerm[pair] = sortedSlot.
    public static func encodeMoEDownReduceCombineF32(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                                     dnAcc: QWTensorValue, invPerm: QWTensorValue?,
                                                     weights: QWTensorValue, shared: QWTensorValue,
                                                     gate: QWTensorValue, out: QWTensorValue,
                                                     hidden: Int, routes: Int, tokens: Int,
                                                     out16: QWTensorValue? = nil,
                                                     skipShared: Bool = false,
                                                     unweightedSingle: Bool = false,
                                                     ids: QWTensorValue? = nil,
                                                     numGroups: Int = 0,
                                                     scatterPerRoute: Bool = false,
                                                     skipOut32: Bool = false,
                                                     nofmaShared: Bool = false) throws {
        enc.setComputePipelineState(try hal.pipeline(
            nofmaShared ? "qwg_moe_down_reduce_combine_f32_nofma" : "qwg_moe_down_reduce_combine_f32"))
        dnAcc.bind(enc, index: 0)
        (invPerm ?? weights).bind(enc, index: 1)
        enc.bindTensors(weights, shared, gate, out, startIndex: 2)
        enc.setUInt32s(hidden, routes, tokens, startIndex: 6)
        (out16 ?? out).bind(enc, index: 9)
        let flags = (out16 != nil ? 1 : 0)
            | (dnAcc.shape.dtype == .f16 ? 2 : 0)
            | (invPerm == nil ? 4 : 0)
            | (skipShared ? 8 : 0)
            | (unweightedSingle ? 16 : 0)
            | (ids != nil ? 32 : 0)
            | (scatterPerRoute ? 64 : 0)
            | (skipOut32 ? 128 : 0)
            | ((numGroups & 0xFF) << 8)
            | (out16?.shape.dtype == .bf16 ? 0x10000 : 0)
            | (out16?.shape.dtype == .f32 ? 0x20000 : 0)
        enc.setUInt32s(flags, startIndex: 10)
        (ids ?? weights).bind(enc, index: 11)
        let cols4 = (hidden + 3) / 4
        let tgRows = scatterPerRoute ? (tokens * routes) : (numGroups > 0 ? (tokens * numGroups) : tokens)
        enc.dispatchThreadgroups(MTLSize(width: (cols4 + 255) / 256, height: tgRows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// The standard MoE path's **f32 bridge**: the f16 normed hidden becomes
    /// the rotated f32 copy the expert GEMMs read and the un-rotated f32 copy
    /// the router reads. The hidden obeys the norm family's law.
    public static func encodeMoEBridgeF16ToF32(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                                x: QWTensorValue, xRot32: QWTensorValue,
                                                xPlain32: QWTensorValue,
                                                rows: Int, hidden: Int,
                                                skipRot32: Bool = false,
                                                plainF16: Bool = false) throws {
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
        let plainDtype: QWDType = plainF16 ? .f16 : .f32
        guard x.shape.dtype == .f16, xRot32.shape.dtype == .f32, xPlain32.shape.dtype == plainDtype,
              x.shape.elementCount >= rows * hidden,
              xRot32.shape.elementCount >= rows * hidden,
              xPlain32.shape.elementCount >= rows * hidden else {
            throw QWGemmError.badShape(
                "moe bridge: rows=\(rows) hidden=\(hidden) do not fit the buffers")
        }
        enc.setComputePipelineState(try hal.pipeline("qwg_moe_bridge_f16_to_f32"))
        enc.bindTensors(x, xRot32, xPlain32, startIndex: 0)
        let hPacked = hidden | (skipRot32 ? 0x8000_0000 : 0) | (plainF16 ? 0x4000_0000 : 0)
        enc.setUInt32s(hPacked, startIndex: 3)
        // One row: a 32-lane threadgroup per 256 hidden (lane-aligned, bit-identical) so decode is not latency-bound.
        let blocks = rows == 1 ? hidden / 256 : 1
        enc.dispatchThreadgroups(MTLSize(width: blocks, height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    /// Unified NAX grouped gate+up prefill dispatch across legacy `QWWeightView` and FP expert banks.
    public static func encodeMoEBankGateUpNax(
        _ enc: any MTLComputeCommandEncoder, hal: QWHal,
        xe: QWFPBufferSlice, bank: QWMoEExpertBank, upBank: QWMoEExpertBank? = nil,
        expertOff: QWTensorValue, tileE: QWTensorValue, tileR: QWTensorValue,
        tileCount: QWTensorValue, guOut: QWFPBufferSlice,
        intermediate: Int, tilePlan: MoETilePlan,
        sortedTok: QWTensorValue? = nil, coldBank: QWWeightView? = nil,
        inDtype: QWFPElemDtype = .f16, outDtype: QWFPElemDtype = .f16
    ) throws {
        switch (bank, upBank) {
        case (.weightView(let v), nil):
            let xeT = QWTensorValue(region: QWMRegion(buffer: xe.buffer, base: xe.offset, length: max(xe.bytes, 1)),
                                    offset: 0, shape: QWTensorShape([max(1, xe.bytes / 2)], .f16))
            let guT = QWTensorValue(region: QWMRegion(buffer: guOut.buffer, base: guOut.offset, length: max(guOut.bytes, 1)),
                                    offset: 0, shape: QWTensorShape([max(1, guOut.bytes / 2)], .f16))
            try encodeMoEGateUpNax(
                enc, hal: hal, xe: xeT, bank: v, expertOff: expertOff,
                tileE: tileE, tileR: tileR, tileCount: tileCount, gu16: guT,
                rowsPerExpert: 2 * intermediate, tilePlan: tilePlan,
                sortedTok: sortedTok, coldBank: coldBank)
        case (.nvfp4(let g), nil):
            try QWGemmFP.encodeMoENaxNVFP4(
                enc, hal: hal, bank: g, x: xe, y: guOut,
                expertOff: expertOff, tileE: tileE, tileR: tileR, tileCount: tileCount,
                tilePlan: tilePlan, inDtype: inDtype, outDtype: outDtype,
                nOverride: 2 * intermediate, yStride: 2 * intermediate, yColOff: 0,
                sortedTok: sortedTok)
        case (.nvfp4(let g), .nvfp4(let u)):
            try QWGemmFP.encodeMoENaxNVFP4(
                enc, hal: hal, bank: g, x: xe, y: guOut,
                expertOff: expertOff, tileE: tileE, tileR: tileR, tileCount: tileCount,
                tilePlan: tilePlan, inDtype: inDtype, outDtype: outDtype,
                nOverride: intermediate, yStride: 2 * intermediate, yColOff: 0,
                sortedTok: sortedTok)
            try QWGemmFP.encodeMoENaxNVFP4(
                enc, hal: hal, bank: u, x: xe, y: guOut,
                expertOff: expertOff, tileE: tileE, tileR: tileR, tileCount: tileCount,
                tilePlan: tilePlan, inDtype: inDtype, outDtype: outDtype,
                nOverride: intermediate, yStride: 2 * intermediate, yColOff: intermediate,
                sortedTok: sortedTok)
        case (.mxfp8(let g), nil):
            try QWGemmFP.encodeMoENaxMXFP8(
                enc, hal: hal, bank: g, x: xe, y: guOut,
                expertOff: expertOff, tileE: tileE, tileR: tileR, tileCount: tileCount,
                tilePlan: tilePlan, inDtype: inDtype, outDtype: outDtype,
                nOverride: 2 * intermediate, yStride: 2 * intermediate, yColOff: 0,
                sortedTok: sortedTok)
        case (.mxfp8(let g), .mxfp8(let u)):
            try QWGemmFP.encodeMoENaxMXFP8(
                enc, hal: hal, bank: g, x: xe, y: guOut,
                expertOff: expertOff, tileE: tileE, tileR: tileR, tileCount: tileCount,
                tilePlan: tilePlan, inDtype: inDtype, outDtype: outDtype,
                nOverride: intermediate, yStride: 2 * intermediate, yColOff: 0,
                sortedTok: sortedTok)
            try QWGemmFP.encodeMoENaxMXFP8(
                enc, hal: hal, bank: u, x: xe, y: guOut,
                expertOff: expertOff, tileE: tileE, tileR: tileR, tileCount: tileCount,
                tilePlan: tilePlan, inDtype: inDtype, outDtype: outDtype,
                nOverride: intermediate, yStride: 2 * intermediate, yColOff: intermediate,
                sortedTok: sortedTok)
        default:
            throw QWGemmError.badWeights("mismatched MoE gate/up expert bank container types")
        }
    }

    /// Unified NAX grouped down prefill dispatch across legacy `QWWeightView` and FP expert banks.
    public static func encodeMoEBankDownNax(
        _ enc: any MTLComputeCommandEncoder, hal: QWHal,
        act: QWFPBufferSlice, bank: QWMoEExpertBank,
        expertOff: QWTensorValue, tileE: QWTensorValue, tileR: QWTensorValue,
        tileCount: QWTensorValue, yOut: QWFPBufferSlice,
        hidden: Int, tilePlan: MoETilePlan, coldBank: QWWeightView? = nil,
        inDtype: QWFPElemDtype = .f16, outDtype: QWFPElemDtype = .f16
    ) throws {
        switch bank {
        case .weightView(let v):
            let actT = QWTensorValue(region: QWMRegion(buffer: act.buffer, base: act.offset, length: max(act.bytes, 1)),
                                     offset: 0, shape: QWTensorShape([max(1, act.bytes / 2)], .f16))
            let yDtype: QWDType = (outDtype == .f32) ? .f32 : .f16
            let yT = QWTensorValue(region: QWMRegion(buffer: yOut.buffer, base: yOut.offset, length: max(yOut.bytes, 1)),
                                   offset: 0, shape: QWTensorShape([max(1, yOut.bytes / (yDtype == .f32 ? 4 : 2))], yDtype))
            try encodeMoENaxSplitF32(
                enc, hal: hal, xHi: actT, xLo: actT, bank: v,
                expertOff: expertOff, tileE: tileE, tileR: tileR, tileCount: tileCount,
                y32: yT, rowsPerExpert: hidden, tilePlan: tilePlan, coldBank: coldBank)
        case .nvfp4(let b):
            try QWGemmFP.encodeMoENaxNVFP4(
                enc, hal: hal, bank: b, x: act, y: yOut,
                expertOff: expertOff, tileE: tileE, tileR: tileR, tileCount: tileCount,
                tilePlan: tilePlan, inDtype: inDtype, outDtype: outDtype,
                nOverride: hidden, yStride: hidden, yColOff: 0,
                sortedTok: nil, isDownIndirect: true)
        case .mxfp8(let b):
            try QWGemmFP.encodeMoENaxMXFP8(
                enc, hal: hal, bank: b, x: act, y: yOut,
                expertOff: expertOff, tileE: tileE, tileR: tileR, tileCount: tileCount,
                tilePlan: tilePlan, inDtype: inDtype, outDtype: outDtype,
                nOverride: hidden, yStride: hidden, yColOff: 0,
                sortedTok: nil, isDownIndirect: true)
        }
    }

    /// Dispatches active-bucket compact gate+up+SwiGLU for FP MoE expert banks.
    public static func encodeMoEBankCompactGateUpSwiGLU(
        _ enc: any MTLComputeCommandEncoder, hal: QWHal,
        x: QWFPBufferSlice, gate: QWMoEExpertBank, up: QWMoEExpertBank? = nil,
        expertOff: QWFPBufferSlice, activeExperts: QWFPBufferSlice,
        activeArgs: QWFPBufferSlice, sortedTok: QWFPBufferSlice,
        sortedPairIdx: QWFPBufferSlice, actOut: QWFPBufferSlice,
        intermediate: Int, topK: Int,
        inDtype: QWFPElemDtype, outDtype: QWFPElemDtype,
        gateCap: Float = 0, upCap: Float = 0,
        sortedOrder: Bool = true, useIndirectDispatch: Bool = true,
        maxActiveExperts: Int? = nil, directT1: Bool = false
    ) throws {
        switch (gate, up) {
        case (.nvfp4(let g), nil):
            try QWGemmFP.encodeMoECompactNVFP4GateUpSwiGLU(
                enc, hal: hal, x: x, gate: g, up: nil, expertOff: expertOff,
                activeExperts: activeExperts, activeArgs: activeArgs,
                sortedTok: sortedTok, sortedPairIdx: sortedPairIdx, actOut: actOut,
                intermediate: intermediate, topK: topK,
                inDtype: inDtype, outDtype: outDtype,
                gateCap: gateCap, upCap: upCap, sortedOrder: sortedOrder,
                useIndirectDispatch: useIndirectDispatch, maxActiveExperts: maxActiveExperts,
                directT1: directT1)
        case (.nvfp4(let g), .nvfp4(let u)):
            try QWGemmFP.encodeMoECompactNVFP4GateUpSwiGLU(
                enc, hal: hal, x: x, gate: g, up: u, expertOff: expertOff,
                activeExperts: activeExperts, activeArgs: activeArgs,
                sortedTok: sortedTok, sortedPairIdx: sortedPairIdx, actOut: actOut,
                intermediate: intermediate, topK: topK,
                inDtype: inDtype, outDtype: outDtype,
                gateCap: gateCap, upCap: upCap, sortedOrder: sortedOrder,
                useIndirectDispatch: useIndirectDispatch, maxActiveExperts: maxActiveExperts,
                directT1: directT1)
        case (.mxfp8(let g), nil):
            try QWGemmFP.encodeMoECompactMXFP8GateUpSwiGLU(
                enc, hal: hal, x: x, gate: g, up: nil, expertOff: expertOff,
                activeExperts: activeExperts, activeArgs: activeArgs,
                sortedTok: sortedTok, sortedPairIdx: sortedPairIdx, actOut: actOut,
                intermediate: intermediate, topK: topK,
                inDtype: inDtype, outDtype: outDtype,
                gateCap: gateCap, upCap: upCap, sortedOrder: sortedOrder,
                useIndirectDispatch: useIndirectDispatch, maxActiveExperts: maxActiveExperts,
                directT1: directT1)
        case (.mxfp8(let g), .mxfp8(let u)):
            try QWGemmFP.encodeMoECompactMXFP8GateUpSwiGLU(
                enc, hal: hal, x: x, gate: g, up: u, expertOff: expertOff,
                activeExperts: activeExperts, activeArgs: activeArgs,
                sortedTok: sortedTok, sortedPairIdx: sortedPairIdx, actOut: actOut,
                intermediate: intermediate, topK: topK,
                inDtype: inDtype, outDtype: outDtype,
                gateCap: gateCap, upCap: upCap, sortedOrder: sortedOrder,
                useIndirectDispatch: useIndirectDispatch, maxActiveExperts: maxActiveExperts,
                directT1: directT1)
        default:
            throw QWGemmError.badWeights("compact FP gate_up requires matching NVFP4 or MXFP8 banks")
        }
    }

    /// Dispatches active-bucket compact down projection for FP MoE expert banks.
    public static func encodeMoEBankCompactDown(
        _ enc: any MTLComputeCommandEncoder, hal: QWHal,
        actIn: QWFPBufferSlice, down: QWMoEExpertBank,
        expertOff: QWFPBufferSlice, activeExperts: QWFPBufferSlice,
        activeArgs: QWFPBufferSlice, sortedPairIdx: QWFPBufferSlice,
        slotOut: QWFPBufferSlice, topK: Int,
        inDtype: QWFPElemDtype, outDtype: QWFPElemDtype,
        sortedOrder: Bool = true, useIndirectDispatch: Bool = true,
        maxActiveExperts: Int? = nil, directT1: Bool = false
    ) throws {
        switch down {
        case .nvfp4(let d):
            try QWGemmFP.encodeMoECompactNVFP4Down(
                enc, hal: hal, actIn: actIn, down: d, expertOff: expertOff,
                activeExperts: activeExperts, activeArgs: activeArgs,
                sortedPairIdx: sortedPairIdx, slotOut: slotOut,
                topK: topK, inDtype: inDtype, outDtype: outDtype,
                sortedOrder: sortedOrder, useIndirectDispatch: useIndirectDispatch,
                maxActiveExperts: maxActiveExperts, directT1: directT1)
        case .mxfp8(let d):
            try QWGemmFP.encodeMoECompactMXFP8Down(
                enc, hal: hal, actIn: actIn, down: d, expertOff: expertOff,
                activeExperts: activeExperts, activeArgs: activeArgs,
                sortedPairIdx: sortedPairIdx, slotOut: slotOut,
                topK: topK, inDtype: inDtype, outDtype: outDtype,
                sortedOrder: sortedOrder, useIndirectDispatch: useIndirectDispatch,
                maxActiveExperts: maxActiveExperts, directT1: directT1)
        case .weightView:
            throw QWGemmError.badWeights("compact FP down requires NVFP4 or MXFP8 bank")
        }
    }

    /// Runs grouped-NAX prefill (`T > 8`) or active-bucket compact decode (`T <= 8`) across FP MoE banks, then combines routes + shared output.
    public static func encodeMoEBankRoutedCombine(
        _ enc: inout any MTLComputeCommandEncoder, hal: QWHal,
        x: QWFPBufferSlice, xDtype: QWFPElemDtype, interDtype: QWFPElemDtype,
        gateUp: QWMoEExpertBank, up: QWMoEExpertBank?, down: QWMoEExpertBank,
        idx: QWTensorValue, wts: QWTensorValue,
        shared: QWFPBufferSlice?, sharedDtype: QWFPElemDtype, gate: QWFPBufferSlice?,
        out: QWFPBufferSlice, outDtype: QWFPElemDtype,
        expOff: QWTensorValue, sortedPair: QWTensorValue,
        sortedTok: QWTensorValue, invPerm: QWTensorValue,
        tileE: QWTensorValue?, tileR: QWTensorValue?, tileCount: QWTensorValue?,
        activeExperts: QWTensorValue?, activeIndirect: QWTensorValue?,
        guBuf: QWTensorValue, actBuf: QWTensorValue, dnAccBuf: QWTensorValue,
        hidden H: Int, experts E: Int, topK K: Int, intermediate I: Int, rows T: Int,
        gateSoftcap: Float, upSoftcap: Float,
        supportsGroupedNaxPrefill: Bool, concurrentBarriers: Bool,
        rotateEncoder: ((inout any MTLComputeCommandEncoder) throws -> Void)?,
        minNaxRows: Int = 9
    ) throws {
        @inline(__always) func barrier() {
            if concurrentBarriers { enc.memoryBarrier(scope: .buffers) }
        }
        let useNax = T >= minNaxRows && supportsGroupedNaxPrefill && hal.caps.supportsSimdgroupMMA
            && (xDtype == .f16 || xDtype == .bf16) && tileE != nil && tileR != nil && tileCount != nil
        if useNax, let te = tileE, let tr = tileR, let tc = tileCount {
            guard let rotateEncoder else {
                throw QWGemmError.badShape("FP grouped NAX prefill requires rotateEncoder for indirect slices")
            }
            let downIsRotated: Bool = {
                if case .weightView(let d) = down { return d.isRotated }
                return false
            }()
            let isNVFP4GU: Bool = {
                if case .nvfp4 = gateUp { return true }
                return false
            }()
            let useFusedFPGateUpSwiGLU: Bool = {
                guard up == nil, I % 32 == 0, !downIsRotated else { return false }
                switch gateUp {
                case .nvfp4, .mxfp8: return true
                case .weightView: return false
                }
            }()
            let guProjRows = useFusedFPGateUpSwiGLU ? (2 * I) : ((up != nil) ? I : (2 * I))
            let tilePlan = MoETilePlan(experts: E, pairs: T * K, gateUpRows: guProjRows, downRows: H)
            try encodeMoEGroupPairsF32(
                enc, hal: hal, ids: idx, expertOff: expOff,
                sortedPairIdx: sortedPair, sortedTok: sortedTok,
                experts: E, totalPairs: T * K, routes: K,
                invPerm: invPerm, tileE: te, tileR: tr, tileCount: tc, tilePlan: tilePlan)
            let guSlice = QWFPBufferSlice(guBuf), actSlice = QWFPBufferSlice(actBuf)
            let canPreGather = useFusedFPGateUpSwiGLU && guBuf.bytes >= T * K * H * 2
            let gatherDtype: QWFPElemDtype = isNVFP4GU ? .f16 : xDtype
            let naxInterDtype: QWFPElemDtype = (canPreGather && isNVFP4GU) ? .f16 : interDtype
            if canPreGather {
                barrier()
                try QWGemmFP.encodeMoEGatherX(
                    enc, hal: hal, x: x, sortedTok: sortedTok, xSorted: guSlice,
                    totalPairs: T * K, k: H, inDtype: xDtype, outDtype: gatherDtype)
            }
            try rotateEncoder(&enc)
            let naxX: QWFPBufferSlice = canPreGather ? guSlice : x
            let naxXDtype: QWFPElemDtype = canPreGather ? gatherDtype : xDtype
            let naxSortedTok: QWTensorValue? = canPreGather ? nil : sortedTok
            switch gateUp {
            case .nvfp4(let g) where useFusedFPGateUpSwiGLU:
                try QWGemmFP.encodeMoENaxNVFP4SwiGLU(
                    enc, hal: hal, bank: g, x: naxX, actOut: actSlice,
                    expertOff: expOff, tileE: te, tileR: tr, tileCount: tc,
                    tilePlan: tilePlan, intermediate: I, inDtype: naxXDtype, outDtype: naxInterDtype,
                    gateCap: gateSoftcap, upCap: upSoftcap, sortedTok: naxSortedTok)
            case .mxfp8(let g) where useFusedFPGateUpSwiGLU:
                try QWGemmFP.encodeMoENaxMXFP8SwiGLU(
                    enc, hal: hal, bank: g, x: naxX, actOut: actSlice,
                    expertOff: expOff, tileE: te, tileR: tr, tileCount: tc,
                    tilePlan: tilePlan, intermediate: I, inDtype: naxXDtype, outDtype: naxInterDtype,
                    gateCap: gateSoftcap, upCap: upSoftcap, sortedTok: naxSortedTok)
            default:
                try encodeMoEBankGateUpNax(
                    enc, hal: hal, xe: x, bank: gateUp, upBank: up,
                    expertOff: expOff, tileE: te, tileR: tr, tileCount: tc,
                    guOut: guSlice, intermediate: I, tilePlan: tilePlan,
                    sortedTok: sortedTok, inDtype: xDtype, outDtype: interDtype)
                barrier()
                try QWGemmFP.encodeSwiGLUSoftcap(
                    enc, hal: hal, gu: guSlice, act: actSlice,
                    rows: T * K, intermediate: I, gateCap: gateSoftcap, upCap: upSoftcap,
                    inDtype: interDtype, outDtype: interDtype, rotateFwht32: downIsRotated)
            }
            barrier()
            try encodeMoEBankDownNax(
                enc, hal: hal, act: actSlice, bank: down,
                expertOff: expOff, tileE: te, tileR: tr, tileCount: tc,
                yOut: QWFPBufferSlice(dnAccBuf), hidden: H, tilePlan: tilePlan,
                inDtype: naxInterDtype, outDtype: naxInterDtype)
            barrier()
            try QWGemmFP.encodeMoEDownReduceCombineBF16(
                enc, hal: hal, dnAcc: QWFPBufferSlice(dnAccBuf), invPerm: QWFPBufferSlice(invPerm),
                weights: QWFPBufferSlice(wts), shared: shared, gate: gate, out: out,
                hidden: H, topK: K, tokens: T,
                inDtype: naxInterDtype, outDtype: outDtype, sharedDtype: sharedDtype)
            return
        } else if T == 1 {
            let idxSlice = QWFPBufferSlice(idx)
            try encodeMoEBankCompactGateUpSwiGLU(
                enc, hal: hal, x: x, gate: gateUp, up: up,
                expertOff: QWFPBufferSlice(expOff), activeExperts: idxSlice,
                activeArgs: QWFPBufferSlice(activeIndirect ?? expOff),
                sortedTok: QWFPBufferSlice(sortedTok), sortedPairIdx: QWFPBufferSlice(sortedPair),
                actOut: QWFPBufferSlice(actBuf), intermediate: I, topK: K,
                inDtype: xDtype, outDtype: interDtype, gateCap: gateSoftcap, upCap: upSoftcap,
                sortedOrder: false, useIndirectDispatch: false, maxActiveExperts: K, directT1: true)
            barrier()
            try encodeMoEBankCompactDown(
                enc, hal: hal, actIn: QWFPBufferSlice(actBuf), down: down,
                expertOff: QWFPBufferSlice(expOff), activeExperts: idxSlice,
                activeArgs: QWFPBufferSlice(activeIndirect ?? expOff),
                sortedPairIdx: QWFPBufferSlice(sortedPair), slotOut: QWFPBufferSlice(dnAccBuf),
                topK: K, inDtype: interDtype, outDtype: interDtype,
                sortedOrder: false, useIndirectDispatch: false, maxActiveExperts: K, directT1: true)
            barrier()
            try QWGemmFP.encodeMoEDownReduceCombineBF16(
                enc, hal: hal, dnAcc: QWFPBufferSlice(dnAccBuf), invPerm: nil,
                weights: QWFPBufferSlice(wts), shared: shared, gate: gate, out: out,
                hidden: H, topK: K, tokens: T,
                inDtype: interDtype, outDtype: outDtype, sharedDtype: sharedDtype, ids: idxSlice)
            return
        } else {
            guard let actExp = activeExperts, let actInd = activeIndirect,
                  actInd.bytes >= QWGemmFP.MoEActiveBucketPlan.indirectBufferBytes,
                  actExp.bytes >= E * 4 else {
                throw QWGemmError.badShape("FP compact MoE requires activeExperts (>= E*4 B) and activeIndirect (>= 48 B)")
            }
            let bucketPlan = QWGemmFP.MoEActiveBucketPlan(experts: E, intermediate: I, hidden: H)
            let expOffS = QWFPBufferSlice(expOff), actExpS = QWFPBufferSlice(actExp), actIndS = QWFPBufferSlice(actInd)
            try encodeMoEGroupAndActiveBuckets(
                enc, hal: hal, ids: idx, expertOff: expOff,
                activeExperts: actExp, activeIndirect: actInd,
                sortedPairIdx: sortedPair, sortedTok: sortedTok,
                experts: E, totalPairs: T * K, routes: K,
                plan: bucketPlan, invPerm: invPerm,
                concurrentBarriers: concurrentBarriers)
            let maxAct = min(E, T * K)
            let directBatch = (T <= 16 && maxAct <= 128)
            let indirectDispatch = !directBatch && (rotateEncoder != nil)
            if indirectDispatch, let rot = rotateEncoder { try rot(&enc) } else { barrier() }
            try encodeMoEBankCompactGateUpSwiGLU(
                enc, hal: hal, x: x, gate: gateUp, up: up,
                expertOff: expOffS, activeExperts: actExpS, activeArgs: actIndS,
                sortedTok: QWFPBufferSlice(sortedTok), sortedPairIdx: QWFPBufferSlice(sortedPair),
                actOut: QWFPBufferSlice(actBuf), intermediate: I, topK: K,
                inDtype: xDtype, outDtype: interDtype, gateCap: gateSoftcap, upCap: upSoftcap,
                sortedOrder: true, useIndirectDispatch: indirectDispatch, maxActiveExperts: maxAct)
            barrier()
            try encodeMoEBankCompactDown(
                enc, hal: hal, actIn: QWFPBufferSlice(actBuf), down: down,
                expertOff: expOffS, activeExperts: actExpS, activeArgs: actIndS,
                sortedPairIdx: QWFPBufferSlice(sortedPair), slotOut: QWFPBufferSlice(dnAccBuf),
                topK: K, inDtype: interDtype, outDtype: interDtype,
                sortedOrder: true, useIndirectDispatch: indirectDispatch, maxActiveExperts: maxAct)
            barrier()
        }
        try QWGemmFP.encodeMoEDownReduceCombineBF16(
            enc, hal: hal, dnAcc: QWFPBufferSlice(dnAccBuf), invPerm: QWFPBufferSlice(invPerm),
            weights: QWFPBufferSlice(wts), shared: shared, gate: gate, out: out,
            hidden: H, topK: K, tokens: T,
            inDtype: interDtype, outDtype: outDtype, sharedDtype: sharedDtype)
    }
}
