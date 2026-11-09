// QWAttnDecodeApply.swift — engine-side encoder for the C09 split-K decode.
//
// Contract mirrors qwattn_decode.metal: TWO launches per (token, layer):
//   1. qwg_attn_decode        grid = (24 q-heads × splits) × 32 threads —
//      online fp32 softmax over this split's key range (paged gather via the
//      C10 view) + new-token K/V write fused into the epilogue (split-0,
//      head%6==0 → one writer per kv-head; every split would store identical
import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWKV

public enum QWAttnDecodeError: Error, CustomStringConvertible {
    case badShape(String)
    public var description: String { switch self { case .badShape(let s): return "QWAttnDecodeError badShape: \(s)" } }
}

/// Encoder for `qwg_attn_decode` + `qwg_attn_decode_combine`. Stateless
/// after init; scratch is caller-owned.
public final class QWAttnDecode {
    public static let maxSplits = 64
    public static let maxLongContextSplits = 256
    /// Floats per partial row: O[256] + m + l.
    /// The partial row's stride: the head's floats plus the two softmax
    /// accumulators. The kernel derives it from QW_HEAD_DIM.
    public static func partialRowFloats(geometry: QWModelGeometry) -> Int { geometry.headDim + 2 }

    public let geometry: QWModelGeometry
    /// One simdgroup per q head of a kv head: the GQA ratio x 32 lanes.
    public let threadsPerGroup: Int
    private let hal: QWHal
    private let isGeneral: Bool
    private let psoDecode: any MTLComputePipelineState
    private let psoDecodeSparse: any MTLComputePipelineState
    private let psoCombine: any MTLComputePipelineState
    private let psoSharedPrefixByRows: [Int: any MTLComputePipelineState]
    private let psoSharedCombine: (any MTLComputePipelineState)?
    private let sharedPrefixRows: Int?
    /// 4c: pool addressing is a COMPILE-TIME template parameter of the
    /// kernels (exact-bit law), so the engine's KV mode picks the pipeline
    /// VARIANT at init; encode() asserts any caller's view matches it.
    public let kvMode: QWKVMode

    static func decodeKernelName(_ mode: QWKVMode) -> String {
        switch mode {
        case .f16: return "qwg_attn_decode"
        case .q8:  return "qwg_attn_decode_q8"
        case .q4:  return "qwg_attn_decode_q4"
        }
    }
    static func sparseDecodeKernelName(_ mode: QWKVMode) -> String {
        switch mode {
        case .f16: return "qwg_attn_decode_sparse"
        case .q8:  return "qwg_attn_decode_sparse_q8"
        case .q4:  return "qwg_attn_decode_sparse_q4"
        }
    }
    static func batchedKernelName(_ mode: QWKVMode) -> String {
        switch mode {
        case .f16: return "qwg_attn_decode_batched"
        case .q8:  return "qwg_attn_decode_batched_q8"
        case .q4:  return "qwg_attn_decode_batched_q4"
        }
    }
    static func sharedKernelName(_ mode: QWKVMode, _ b: Int) -> String {
        switch mode {
        case .f16: return "qwg_attn_decode_shared_m\(b)"
        case .q8:  return "qwg_attn_decode_shared_m\(b)_q8"
        case .q4:  return "qwg_attn_decode_shared_m\(b)_q4"
        }
    }
    /// EXP-9 sub-item 3 (opt-in, NOT default): batched-dispatch variants of
    /// the two kernels above. Lazily resolved so callers that never use the
    /// batched path pay no extra pipeline-build cost.
    private lazy var psoDecodeBatched: any MTLComputePipelineState = {
        try! hal.pipeline(Self.batchedKernelName(self.kvMode))
    }()
    private lazy var psoCombineBatched: any MTLComputePipelineState = {
        try! hal.pipeline("qwg_attn_decode_combine_batched")
    }()

    /// `forceGeneral` is the test seam: it selects the general kernels at a
    /// geometry the frozen ones could serve, so the equivalence test can
    /// compare the two on the SAME data. It changes no computation.
    public init(hal: QWHal, sharedPrefixRows: Int? = nil, kvMode: QWKVMode = .f16,
                geometry: QWModelGeometry,
                forceGeneral: Bool = false) throws {
        guard sharedPrefixRows == nil || (2...8).contains(sharedPrefixRows!) else {
            throw QWAttnDecodeError.badShape("shared prefix requires B=2...8")
        }
        self.geometry = geometry
        self.threadsPerGroup = max(1, geometry.qHeads / max(1, geometry.kvHeads)) * 32
        self.hal = hal
        self.kvMode = kvMode
        // The frozen decode serves the 27B's shape and nothing else (C108);
        // every other geometry takes the general pair, whose lanes past the
        // head's data carry zeros (C102).
        let gqa = geometry.gqaRatio
        let general = forceGeneral || !geometry.is27BShape
        self.isGeneral = general
        if general {
            // The general kernel stages 2 × the GQA ratio keys per phase,
            // and its staging holds at most 16.
            guard gqa <= 8 else {
                throw QWAttnDecodeError.badShape(
                    "the general decode stages 2×\(gqa) keys per phase (max 16)")
            }
            guard kvMode == .f16 || kvMode == .q8 || kvMode == .q4 else {
                throw QWAttnDecodeError.badShape(
                    "a \(geometry.headDim)-half head supports f16, q8, or q4 KV only")
            }
        }
        let generalDecodeName: String
        switch kvMode {
        case .f16: generalDecodeName = "qwg_attn_decode_general"
        case .q8:  generalDecodeName = "qwg_attn_decode_general_q8"
        case .q4:  generalDecodeName = "qwg_attn_decode_general_q4"
        }
        self.psoDecode = try hal.pipeline(general ? generalDecodeName
                                                   : Self.decodeKernelName(kvMode))
        self.psoDecodeSparse = try hal.pipeline(Self.sparseDecodeKernelName(kvMode))
        self.psoCombine = try hal.pipeline(general ? "qwg_attn_decode_combine_general"
                                                     : "qwg_attn_decode_combine")
        self.sharedPrefixRows = sharedPrefixRows
        if let sharedPrefixRows {
            var map: [Int: any MTLComputePipelineState] = [:]
            for r in 2...sharedPrefixRows {
                map[r] = try hal.pipeline(Self.sharedKernelName(kvMode, r))
            }
            psoSharedPrefixByRows = map
            psoSharedCombine = try hal.pipeline("qwg_attn_decode_combine_batched")
        } else {
            psoSharedPrefixByRows = [:]
            psoSharedCombine = nil
        }
    }

    /// Partials region bytes for `splits` (all 24 heads × maxSplits slots —
    /// a fixed-size scratch keeps the host plan stateless across tokens;
    /// 24·64·258·4 B = 1.54 MiB per concurrent decode).
    public static func decodePartialsBytes(geometry: QWModelGeometry, maxSplits: Int = maxSplits) -> Int {
        geometry.qHeads * maxSplits * (partialRowFloats(geometry: geometry) * 4)
    }

    /// EXP-9 sub-item 3: partials bytes for a BATCHED call of `T` queries
    /// (same per-query stride as `decodePartialsBytes()`, repeated T times —
    /// keep T small, see `encodeBatched` header; T=32 ≈ 49 MiB).
    public static func decodePartialsBytesBatched(T: Int, geometry: QWModelGeometry, maxSplits: Int = maxSplits) -> Int {
        T * decodePartialsBytes(geometry: geometry, maxSplits: maxSplits)
    }

    /// Split plan (ADR knob): P5 FIX — the old 2048 keys/split left the
    /// grid at 4 threadgroups below 2k context (16 attn layers × 4 TG ×
    /// 192 threads = the GPU mostly idle; kernel header records 311 GB/s at
    public static func splits(forKeys keys: Int, tokensPerSplit: Int = 256) -> Int {
        if keys > 262_144 {
            return min(maxLongContextSplits, max(64, (keys + 4095) / 4096))
        }
        return max(8, min(64, (keys + tokensPerSplit - 1) / tokensPerSplit))
    }

    /// Encode one decode step: qgate [12288] f16 (q̂‖gate interleaved,
    /// rope/norm applied by C06.4/C07 upstream), the C10 (layer, kind) views
    /// for this sequence, the new token's k̂/v̂ [1024] f16 rows, out [6144]
    public func encode(_ enc: any MTLComputeCommandEncoder,
                       qgate: QWTensorValue,
                       kv: (k: QWKVLayerView, v: QWKVLayerView),
                       knew: QWTensorValue, vnew: QWTensorValue,
                       partials: QWMRegion,
                       out: QWTensorValue,
                       C: Int, splits: Int) throws {
        guard C >= 0, splits >= 1, splits <= Self.maxLongContextSplits,
              qgate.shape.elementCount == geometry.qProjOut,
              knew.shape.elementCount == geometry.kvHeads * geometry.headDim,
              vnew.shape.elementCount == geometry.kvHeads * geometry.headDim,
              out.shape.elementCount == geometry.qHeads * geometry.headDim,
              partials.length >= Self.decodePartialsBytes(geometry: geometry, maxSplits: max(Self.maxSplits, splits)),
              kv.k.pageBytes == QWKVCache.pageBytes(mode: kv.k.mode, geometry: geometry),
              kv.k.mode == kv.v.mode,
              kv.k.pool.buffer === kv.v.pool.buffer || true   // pools may share a buffer
        else { throw QWAttnDecodeError.badShape("C=\(C) splits=\(splits)") }
        // page count the kernel will index must exist in the table tensor
        let pages = (C + 1 + 255) / 256
        guard kv.k.table.shape.elementCount >= pages,
              kv.v.table.shape.elementCount >= pages
        else { throw QWAttnDecodeError.badShape("page table too short for C=\(C)") }

        var cU = UInt32(C), sU = UInt32(splits)
        // 4c: the kernel's pool addressing mode comes from the VIEW, never
        // a caller flag. q8 pages address in bytes (payload 262144 + f16
        // scale array 2048); f16 pages address in QW_PGEL halves. Any other
        // mode failed the load guard — reaching here is a driver bug, so it
        // throws (never a silent f16 fallback).
        guard kv.k.mode == kvMode else {
            throw QWAttnDecodeError.badShape("view mode \(kv.k.mode) != pipeline mode \(kvMode)")
        }

        enc.setComputePipelineState(psoDecode)
        qgate.bind(enc, index: 0)
        kv.k.pool.bindTo(enc, index: 1)          // pool extent (buffer, base)
        kv.v.pool.bindTo(enc, index: 2)
        kv.k.table.bind(enc, index: 3)
        kv.v.table.bind(enc, index: 4)
        knew.bind(enc, index: 5)
        vnew.bind(enc, index: 6)
        enc.setBuffer(partials.buffer, offset: partials.base, index: 7)
        enc.setBytes(&cU, length: 4, index: 8)
        enc.setBytes(&sU, length: 4, index: 9)
        if isGeneral {
            out.bind(enc, index: 10)
        }
        // v3 shape: one workgroup = ONE kv-head's six q-heads (192 lanes =
        // 6 simdgroups) — the spec's head fusion. Grid = 4 × splits.
        enc.dispatchThreadgroups(
            MTLSize(width: geometry.kvHeads, height: splits, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))

        if !isGeneral || splits > 1 {
            enc.setComputePipelineState(psoCombine)
            qgate.bind(enc, index: 0)
            out.bind(enc, index: 1)
            enc.setBuffer(partials.buffer, offset: partials.base, index: 2)
            enc.setBytes(&sU, length: 4, index: 3)
            enc.dispatchThreadgroups(
                MTLSize(width: geometry.qHeads, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        }
    }

    /// Encode one LSA sparse decode step over `blockList[0..<blockCount]` + tail `[4*((C+1)/4) .. C]`.
    public func encodeSparse(_ enc: any MTLComputeCommandEncoder,
                             qgate: QWTensorValue,
                             kv: (k: QWKVLayerView, v: QWKVLayerView),
                             knew: QWTensorValue, vnew: QWTensorValue,
                             partials: QWMRegion,
                             out: QWTensorValue,
                             blockList: QWTensorValue,
                             blockCount: QWTensorValue,
                             C: Int, splits: Int, listStride: Int) throws {
        guard C >= 0, splits >= 1, splits <= Self.maxLongContextSplits,
              qgate.shape.elementCount == geometry.qProjOut,
              knew.shape.elementCount == geometry.kvHeads * geometry.headDim,
              vnew.shape.elementCount == geometry.kvHeads * geometry.headDim,
              out.shape.elementCount == geometry.qHeads * geometry.headDim,
              partials.length >= Self.decodePartialsBytes(geometry: geometry, maxSplits: max(Self.maxSplits, splits)),
              kv.k.pageBytes == QWKVCache.pageBytes(mode: kv.k.mode, geometry: geometry),
              kv.k.mode == kv.v.mode,
              kv.k.mode == kvMode
        else { throw QWAttnDecodeError.badShape("sparse C=\(C) splits=\(splits)") }
        let pages = (C + 1 + 255) / 256
        guard kv.k.table.shape.elementCount >= pages,
              kv.v.table.shape.elementCount >= pages
        else { throw QWAttnDecodeError.badShape("page table too short for C=\(C)") }

        var cU = UInt32(C), sU = UInt32(splits), strideU = UInt32(listStride)
        enc.setComputePipelineState(psoDecodeSparse)
        qgate.bind(enc, index: 0)
        kv.k.pool.bindTo(enc, index: 1)
        kv.v.pool.bindTo(enc, index: 2)
        kv.k.table.bind(enc, index: 3)
        kv.v.table.bind(enc, index: 4)
        knew.bind(enc, index: 5)
        knew.shape.elementCount >= 0 ? vnew.bind(enc, index: 6) : ()
        enc.setBuffer(partials.buffer, offset: partials.base, index: 7)
        enc.setBytes(&cU, length: 8 / 2, index: 8)
        enc.setBytes(&sU, length: 4, index: 9)
        out.bind(enc, index: 10)
        blockList.bind(enc, index: 11)
        blockCount.bind(enc, index: 12)
        enc.setBytes(&strideU, length: 4, index: 13)
        enc.dispatchThreadgroups(
            MTLSize(width: geometry.kvHeads, height: splits, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))

        if splits > 1 {
            enc.setComputePipelineState(psoCombine)
            qgate.bind(enc, index: 0)
            out.bind(enc, index: 1)
            enc.setBuffer(partials.buffer, offset: partials.base, index: 2)
            enc.setBytes(&sU, length: 4, index: 3)
            enc.dispatchThreadgroups(
                MTLSize(width: geometry.qHeads, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        }
    }

    /// EXP-9 sub-item 3 (OPT-IN, NOT default): ONE dispatch pair for a
    /// T-query CONTINUATION CHUNK (pos>0), replacing a Swift-level loop of
    /// T `encode(...)` calls. Each query qi (0..<T) attends EXACTLY
    public func encodeBatched(_ enc: any MTLComputeCommandEncoder,
                              qgate: QWTensorValue,
                              kv: (k: QWKVLayerView, v: QWKVLayerView),
                              knew: QWTensorValue, vnew: QWTensorValue,
                              partials: QWMRegion,
                              out: QWTensorValue,
                              cBase: Int, T: Int, splits: Int, sharedPrefix: Bool = false) throws {
        if sharedPrefix {
            func valid(_ value: QWTensorValue, dtype: QWDType = .f16) -> Bool {
                let region = value.region
                let alignment = dtype == .i32 ? 4 : 2
                return value.shape.dtype == dtype && value.offset >= 0 &&
                    value.offset % alignment == 0 && region.base % alignment == 0 &&
                    region.base >= 0 && region.length >= 0 && region.base <= region.buffer.length &&
                    region.length <= region.buffer.length - region.base &&
                    value.offset <= region.length && value.bytes <= region.length - value.offset
            }
            func overlaps(_ buffer: any MTLBuffer, _ start: Int, _ bytes: Int,
                          _ other: any MTLBuffer, _ otherStart: Int, _ otherBytes: Int) -> Bool {
                buffer === other && start < otherStart + otherBytes && otherStart < start + bytes
            }
            guard let maxShared = sharedPrefixRows, (2...maxShared).contains(T), (2...8).contains(T),
                  cBase >= 0, cBase <= Int(UInt32.max) - T - Self.maxLongContextSplits,
                  valid(qgate), valid(knew), valid(vnew), valid(out),
                  valid(kv.k.table, dtype: .i32), valid(kv.v.table, dtype: .i32),
                  partials.base >= 0, partials.base % 4 == 0,
                  partials.base <= partials.buffer.length,
                  partials.length >= 0, partials.length <= partials.buffer.length - partials.base,
                  kv.v.pageBytes == QWKVCache.pageBytes(mode: kv.v.mode, geometry: geometry)
            else { throw QWAttnDecodeError.badShape("shared prefix configuration or tensor extent") }
            for value in [qgate, knew, vnew, out] {
                guard !overlaps(partials.buffer, partials.base, partials.length,
                                value.region.buffer, value.region.base + value.offset, value.bytes)
                else { throw QWAttnDecodeError.badShape("shared prefix partials alias a tensor") }
            }
            for value in [qgate, knew, vnew] {
                guard !overlaps(out.region.buffer, out.region.base + out.offset, out.bytes,
                                value.region.buffer, value.region.base + value.offset, value.bytes)
                else { throw QWAttnDecodeError.badShape("shared prefix output aliases input") }
            }
            for pool in [kv.k.pool, kv.v.pool] {
                guard !overlaps(pool.buffer, pool.base, pool.length, partials.buffer, partials.base, partials.length),
                      !overlaps(pool.buffer, pool.base, pool.length,
                                out.region.buffer, out.region.base + out.offset, out.bytes)
                else { throw QWAttnDecodeError.badShape("shared prefix scratch aliases a KV pool") }
            }
        }
        guard cBase >= 0, T >= 1, splits >= 1, splits <= Self.maxLongContextSplits,
              qgate.shape.elementCount == T * geometry.qProjOut,
              knew.shape.elementCount == T * geometry.kvHeads * geometry.headDim,
              vnew.shape.elementCount == T * geometry.kvHeads * geometry.headDim,
              out.shape.elementCount == T * geometry.qHeads * geometry.headDim,
              partials.length >= Self.decodePartialsBytesBatched(T: T, geometry: geometry, maxSplits: max(Self.maxSplits, splits)),
              kv.k.pageBytes == QWKVCache.pageBytes(mode: kv.k.mode, geometry: geometry),
              kv.k.mode == kv.v.mode
        else { throw QWAttnDecodeError.badShape("cBase=\(cBase) T=\(T) splits=\(splits)") }
        let pages = (cBase + T + 255) / 256   // last query's C = cBase+T-1, nKeys = cBase+T
        guard kv.k.table.shape.elementCount >= pages,
              kv.v.table.shape.elementCount >= pages
        else { throw QWAttnDecodeError.badShape("page table too short for cBase=\(cBase) T=\(T)") }

        // A common split stride preserves the reference's 12-key fold boundaries.
        // At a stride transition, use the scalar reference instead of reordering.
        if sharedPrefix && (cBase + splits) / splits != (cBase + T + splits - 1) / splits {
            func row(_ tensor: QWTensorValue, _ index: Int, _ columns: Int) -> QWTensorValue {
                QWTensorValue(region: tensor.region, offset: tensor.offset + index * columns * 2,
                              shape: QWTensorShape([columns], .f16))
            }
            for query in 0..<T {
                try encode(enc, qgate: row(qgate, query, geometry.qProjOut), kv: kv,
                    knew: row(knew, query, geometry.kvHeads * geometry.headDim),
                    vnew: row(vnew, query, geometry.kvHeads * geometry.headDim),
                    partials: partials, out: row(out, query, geometry.qHeads * geometry.headDim),
                    C: cBase + query, splits: splits)
                enc.memoryBarrier(scope: .buffers)
            }
            return
        }
        var cU = UInt32(cBase), sU = UInt32(splits)
        guard kv.k.mode == kvMode else {
            throw QWAttnDecodeError.badShape("view mode \(kv.k.mode) != pipeline mode \(kvMode)")
        }

        if sharedPrefix {
            guard let psoSharedPrefix = psoSharedPrefixByRows[T] else {
                throw QWAttnDecodeError.badShape("shared prefix pipeline absent")
            }
            enc.setComputePipelineState(psoSharedPrefix)
        } else {
            enc.setComputePipelineState(psoDecodeBatched)
        }
        qgate.bind(enc, index: 0)
        kv.k.pool.bindTo(enc, index: 1)
        kv.v.pool.bindTo(enc, index: 2)
        kv.k.table.bind(enc, index: 3)
        kv.v.table.bind(enc, index: 4)
        knew.bind(enc, index: 5)
        vnew.bind(enc, index: 6)
        enc.setBuffer(partials.buffer, offset: partials.base, index: 7)
        enc.setBytes(&cU, length: 4, index: 8)
        enc.setBytes(&sU, length: 4, index: 9)
        enc.dispatchThreadgroups(
            MTLSize(width: geometry.kvHeads, height: splits, depth: sharedPrefix ? 1 : T),
            threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))

        if sharedPrefix {
            guard let psoSharedCombine else { throw QWAttnDecodeError.badShape("shared combine pipeline absent") }
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(psoSharedCombine)
        } else {
            enc.setComputePipelineState(psoCombineBatched)
        }
        qgate.bind(enc, index: 0)
        out.bind(enc, index: 1)
        enc.setBuffer(partials.buffer, offset: partials.base, index: 2)
        enc.setBytes(&sU, length: 4, index: 3)
        enc.dispatchThreadgroups(
            MTLSize(width: geometry.qHeads, height: T, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }
}

extension QWMRegion {
    /// Bind the WHOLE region extent (pool buffers carry base offsets).
    func bindTo(_ e: any MTLComputeCommandEncoder, index: Int) {
        e.setBuffer(buffer, offset: base, index: index)
    }
}
