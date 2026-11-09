import Foundation
import QWAlloc
import QWCore
import QWGEMM
import QWHAL

/// KV storage mode for Flash-Next full-attention layers (`f32`, `f16` default, `q8`, `q4`).
/// `indexK` and `pooled` always store `f16` (2 bytes/element).
public enum FNKVStorageMode: String, Sendable {
    case f32
    case f16
    case q8
    case q4

    public static func resolve(_ requested: String?) throws -> FNKVStorageMode {
        guard let raw = requested?.lowercased(), !raw.isEmpty, raw != "auto" else {
            return .f16
        }
        guard let mode = FNKVStorageMode(rawValue: raw) else {
            throw FNGPUError.kvModeNotImplemented(requested ?? raw)
        }
        return mode
    }

    public var prepKernelName: String {
        switch self {
        case .f32: return "qwg_attn_prep_f32"
        case .f16: return "qwg_attn_prep_f16"
        case .q8:  return "qwg_attn_prep_q8"
        case .q4:  return "qwg_attn_prep_q4"
        }
    }

    public var sparseGatedKernelName: String {
        switch self {
        case .f32: return "qwg_attn_sparse_gated_f32"
        case .f16: return "qwg_attn_sparse_gated_f16"
        case .q8:  return "qwg_attn_sparse_gated_q8"
        case .q4:  return "qwg_attn_sparse_gated_q4"
        }
    }

    public var sparseGatedGQA3KernelName: String {
        switch self {
        case .f32: return "qwg_attn_sparse_gated_gqa3_f32"
        case .f16: return "qwg_attn_sparse_gated_gqa3_f16"
        case .q8:  return "qwg_attn_sparse_gated_gqa3_q8"
        case .q4:  return "qwg_attn_sparse_gated_gqa3_q4"
        }
    }

    public var sparseGatedNAXKernelName: String {
        switch self {
        case .f32: return "qwg_attn_sparse_gated_nax_f32"
        case .f16: return "qwg_attn_sparse_gated_nax_f16"
        case .q8:  return "qwg_attn_sparse_gated_nax_q8"
        case .q4:  return "qwg_attn_sparse_gated_nax_q4"
        }
    }

    /// Bytes per token across all 12 trunk + 1 MTP full-attention layers (K + V + meta + indexK + pooled/4) plus shared RoPE.
    public var perTokenBytes: Int {
        switch self {
        case .f32: return 65_536
        case .f16: return 31_296
        case .q8:  return 18_096
        case .q4:  return 11_752
        }
    }
}

/// Per-layer decode state of one full-attention layer: the K/V cache (f32, f16, q8, or q4) and the QSA index caches
/// (f16 raw index keys per token, f16 pooled block keys per completed block), absolute-indexed by position.
public final class FNGPUAttnDecodeState {
    public struct LSAClusterState {
        public let centroids: QWTensorValue      // f16 [maxBuckets, indexHeadDim]
        public let bucketMembers: QWTensorValue  // i32 [maxBuckets] (uchar4)
        public let bucketSizes: QWTensorValue    // u8  [maxBuckets]
        public init(centroids: QWTensorValue, bucketMembers: QWTensorValue, bucketSizes: QWTensorValue) {
            self.centroids = centroids
            self.bucketMembers = bucketMembers
            self.bucketSizes = bucketSizes
        }
    }

    public let fullCapacity: Int, hidden: Int, rows: Int
    public private(set) var capacity: Int
    public let kvMode: FNKVStorageMode
    public var lsaClusters: LSAClusterState?
    public private(set) var k: QWTensorValue, v: QWTensorValue
    public private(set) var kMeta: QWTensorValue, vMeta: QWTensorValue     // f16 scales/mins for q8/q4 (1-elem dummy for f32/f16)
    public private(set) var indexK: QWTensorValue                          // f16 [capacity, indexHeadDim]  raw index keys
    public private(set) var pooled: QWTensorValue                          // f16 [capacity / compress, indexHeadDim]
    private let fullK: QWTensorValue, fullV: QWTensorValue
    private let fullKMeta: QWTensorValue, fullVMeta: QWTensorValue
    private let fullIndexK: QWTensorValue, fullPooled: QWTensorValue
    public let cos: QWTensorValue, sin: QWTensorValue         // f32 [capacity, rotaryDim]
    public let bcos: QWTensorValue, bsin: QWTensorValue       // f32 [capacity / compress, rotaryDim]
    public let rot32: QWTensorValue                           // f32 [hidden]  rotated input
    public let src: QWTensorValue                             // f32 [bankRows]
    public let q: QWTensorValue, gateF: QWTensorValue         // f32 [heads, headDim]
    public let idxQ: QWTensorValue                            // f32 [indexHeads, indexHeadDim]
    public let list: QWTensorValue                            // i32 [blockTopK]
    public let qsaScores: QWTensorValue                       // i32/u32 [rows, capacity / compress]
    public let attnOut: QWTensorValue                         // f32 [heads, headDim]

    public init(arena: QWArena, mixer a: FNGPUAttnMixer, hidden: Int, capacity: Int,
                kvMode: FNKVStorageMode = .f16,
                sharedRope: (cos: QWTensorValue, sin: QWTensorValue, bcos: QWTensorValue, bsin: QWTensorValue)? = nil,
                sharedQSAScores: QWTensorValue? = nil) throws {
        precondition(capacity % a.compress == 0, "capacity must be a multiple of the compress block")
        self.fullCapacity = capacity
        self.capacity = capacity
        self.hidden = hidden
        self.rows = 1
        self.kvMode = kvMode
        let act = arena.activationsRegion()
        let kvAlloc = arena.regionSizes.kv > 0 ? arena.kvRegion() : act
        let kvVAlloc = arena.regionSizes.kv > 0 ? (arena.kvVSub ?? kvAlloc) : act
        let blocks = capacity / a.compress
        func f32(_ n: Int) throws -> QWTensorValue { try act.allocate(QWTensorShape([max(1, n)], .f32)) }
        func f16(_ n: Int) throws -> QWTensorValue { try act.allocate(QWTensorShape([max(1, n)], .f16)) }
        func kvF32(_ sub: QWSubAllocator, _ n: Int) throws -> QWTensorValue {
            try sub.allocate(QWTensorShape([max(1, n)], .f32))
        }
        func kvF16(_ sub: QWSubAllocator, _ n: Int) throws -> QWTensorValue {
            try sub.allocate(QWTensorShape([max(1, n)], .f16))
        }
        func up32(_ t: QWTensorValue, _ a: [Float]) {
            arena.write(t, from: a)
        }
        let kvElems = capacity * a.kvHeads * a.headDim
        switch kvMode {
        case .f32:
            k = try kvF32(kvAlloc, kvElems)
            v = try kvF32(kvVAlloc, kvElems)
            kMeta = try kvF16(kvAlloc, 1)
            vMeta = kMeta
        case .f16:
            k = try kvF16(kvAlloc, kvElems)
            v = try kvF16(kvVAlloc, kvElems)
            kMeta = try kvF16(kvAlloc, 1)
            vMeta = kMeta
        case .q8:
            k = try kvF16(kvAlloc, kvElems / 2)
            v = try kvF16(kvVAlloc, kvElems / 2)
            kMeta = try kvF16(kvAlloc, capacity * a.kvHeads)
            vMeta = try kvF16(kvVAlloc, capacity * a.kvHeads)
        case .q4:
            k = try kvF16(kvAlloc, kvElems / 4)
            v = try kvF16(kvVAlloc, kvElems / 4)
            kMeta = try kvF16(kvAlloc, capacity * a.kvHeads * 4)
            vMeta = try kvF16(kvVAlloc, capacity * a.kvHeads * 4)
        }
        indexK = try kvF16(kvAlloc, capacity * a.indexHeadDim)
        pooled = try kvF16(kvVAlloc, blocks * a.indexHeadDim)
        fullK = k; fullV = v; fullKMeta = kMeta; fullVMeta = vMeta
        fullIndexK = indexK; fullPooled = pooled
        if let sr = sharedRope {
            cos = sr.cos; sin = sr.sin; bcos = sr.bcos; bsin = sr.bsin
        } else {
            cos = try f32(capacity * a.rotaryDim)
            sin = try f32(capacity * a.rotaryDim)
            bcos = try f32(blocks * a.rotaryDim)
            bsin = try f32(blocks * a.rotaryDim)
            let (c, s) = a.rope.tables(count: capacity, stride: 1)
            up32(cos, c); up32(sin, s)
            let (bc, bs) = a.rope.tables(count: blocks, stride: a.compress)
            up32(bcos, bc); up32(bsin, bs)
        }
        rot32 = try f32(hidden)
        src = try f32(a.bankRows)
        q = try f32(a.heads * a.headDim)
        gateF = try f32(a.heads * a.headDim)
        idxQ = try f32(a.indexHeads * a.indexHeadDim)
        list = try act.allocate(QWTensorShape([a.blockTopK], .i32))
        arena.write(list, from: (0..<a.blockTopK).map(Int32.init))
        if let sq = sharedQSAScores {
            qsaScores = sq
        } else {
            qsaScores = try act.allocate(QWTensorShape([max(1, blocks)], .i32))
        }
        attnOut = try f32(a.heads * a.headDim)
    }

    /// Row-tiled state for a verify pass of `rows` tokens: shares `base`'s K/V and index caches (absolute-
    /// indexed by position, so each row writes its own slot) and owns `rows`-wide scratch, one row per token.
    public init(sharingCachesOf base: FNGPUAttnDecodeState, arena: QWArena, mixer a: FNGPUAttnMixer, rows: Int,
                sharedQSAScores: QWTensorValue? = nil) throws {
        precondition(rows >= 1, "rows must be >= 1")
        self.fullCapacity = base.fullCapacity
        self.capacity = base.capacity
        self.hidden = base.hidden
        self.rows = rows
        self.kvMode = base.kvMode
        self.lsaClusters = base.lsaClusters
        k = base.k; v = base.v; kMeta = base.kMeta; vMeta = base.vMeta
        indexK = base.indexK; pooled = base.pooled
        fullK = base.fullK; fullV = base.fullV; fullKMeta = base.fullKMeta; fullVMeta = base.fullVMeta
        fullIndexK = base.fullIndexK; fullPooled = base.fullPooled
        cos = base.cos; sin = base.sin; bcos = base.bcos; bsin = base.bsin
        let act = arena.activationsRegion()
        let blocks = base.fullCapacity / a.compress
        func f32(_ n: Int) throws -> QWTensorValue { try act.allocate(QWTensorShape([max(1, n)], .f32)) }
        rot32 = try f32(rows * base.hidden)
        src = try f32(rows * a.bankRows)
        q = try f32(rows * a.heads * a.headDim)
        gateF = try f32(rows * a.heads * a.headDim)
        idxQ = try f32(rows * a.indexHeads * a.indexHeadDim)
        list = try act.allocate(QWTensorShape([rows * a.blockTopK], .i32))
        arena.write(list, from: (0..<rows).flatMap { _ in (0..<a.blockTopK).map(Int32.init) })
        if let sq = sharedQSAScores {
            qsaScores = sq
        } else {
            qsaScores = try act.allocate(QWTensorShape([max(1, rows * blocks)], .i32))
        }
        attnOut = try f32(rows * a.heads * a.headDim)
    }

    /// Selects a contiguous KV/indexK/pooled partition `slot` out of `totalSlots` (or restores the full capacity when `totalSlots <= 1`).
    public func selectPartition(slot: Int, totalSlots: Int, mixer a: FNGPUAttnMixer) {
        if totalSlots <= 1 {
            capacity = fullCapacity
            k = fullK; v = fullV; kMeta = fullKMeta; vMeta = fullVMeta
            indexK = fullIndexK; pooled = fullPooled
            return
        }
        let slotCap = max(a.compress, ((fullCapacity / totalSlots) / a.compress) * a.compress)
        let pos0 = slot * slotCap
        capacity = fullCapacity
        k = fullK; v = fullV; kMeta = fullKMeta; vMeta = fullVMeta
        let kvSlice = kvViews(pos: pos0, count: slotCap, kvHeads: a.kvHeads, headDim: a.headDim)
        capacity = slotCap
        k = kvSlice.k; v = kvSlice.v; kMeta = kvSlice.kMeta; vMeta = kvSlice.vMeta
        indexK = fullIndexK.slice(byteOffset: pos0 * a.indexHeadDim * 2, count: slotCap * a.indexHeadDim, dtype: .f16)
        let block0 = pos0 / a.compress
        let blocks = slotCap / a.compress
        pooled = fullPooled.slice(byteOffset: block0 * a.indexHeadDim * 2, count: blocks * a.indexHeadDim, dtype: .f16)
    }

    /// Slices K, V, kMeta, and vMeta at `pos` for `count` rows.
    public func kvViews(pos: Int, count: Int, kvHeads: Int, headDim: Int)
        -> (k: QWTensorValue, v: QWTensorValue, kMeta: QWTensorValue, vMeta: QWTensorValue) {
        let segs = count * kvHeads
        let seg0 = pos * kvHeads
        switch kvMode {
        case .f32:
            return (k.slice(byteOffset: seg0 * headDim * 4, count: segs * headDim, dtype: .f32),
                    v.slice(byteOffset: seg0 * headDim * 4, count: segs * headDim, dtype: .f32),
                    kMeta, vMeta)
        case .f16:
            return (k.slice(byteOffset: seg0 * headDim * 2, count: segs * headDim, dtype: .f16),
                    v.slice(byteOffset: seg0 * headDim * 2, count: segs * headDim, dtype: .f16),
                    kMeta, vMeta)
        case .q8:
            return (k.slice(byteOffset: seg0 * headDim, count: segs * (headDim / 2), dtype: .f16),
                    v.slice(byteOffset: seg0 * headDim, count: segs * (headDim / 2), dtype: .f16),
                    kMeta.slice(byteOffset: seg0 * 2, count: segs, dtype: .f16),
                    vMeta.slice(byteOffset: seg0 * 2, count: segs, dtype: .f16))
        case .q4:
            return (k.slice(byteOffset: seg0 * (headDim / 2), count: segs * (headDim / 4), dtype: .f16),
                    v.slice(byteOffset: seg0 * (headDim / 2), count: segs * (headDim / 4), dtype: .f16),
                    kMeta.slice(byteOffset: seg0 * 8, count: segs * 4, dtype: .f16),
                    vMeta.slice(byteOffset: seg0 * 8, count: segs * 4, dtype: .f16))
        }
    }

    public static func allocateLSAClusters(
        hal: QWHal,
        kvCap: Int,
        indexHeadDim: Int,
        states: [(FNGPUAttnDecodeState, FNGPUAttnDecodeState?)]
    ) throws -> QWMRegion {
        let maxSuperblocks = max(1, (kvCap + 511) / 512)
        let maxBuckets = maxSuperblocks * 64
        var cursor = 0
        func align256(_ x: Int) -> Int { (x + 255) & ~255 }
        func reserve(_ bytes: Int) -> Int {
            let off = cursor
            cursor = align256(cursor + max(256, bytes))
            return off
        }
        var offsets: [(Int, Int, Int)] = []
        for _ in 0..<states.count {
            let cOff = reserve(maxBuckets * indexHeadDim * 2)
            let mOff = reserve(maxBuckets * 4)
            let sOff = reserve(maxBuckets)
            offsets.append((cOff, mOff, sOff))
        }
        let totalBytes = max(4096, cursor)
        guard let buf = hal.device.makeBuffer(length: totalBytes, options: .storageModeShared) else {
            throw QWAllocError.outOfSpace(region: "fn-lsa-clusters", asked: totalBytes, free: 0)
        }
        buf.label = "fn-lsa-clusters"
        memset(buf.contents(), 0, totalBytes)
        let reg = QWMRegion(buffer: buf, base: 0, length: totalBytes)
        for (i, (st, vSt)) in states.enumerated() {
            let (cOff, mOff, sOff) = offsets[i]
            let cs = LSAClusterState(
                centroids: QWTensorValue(region: reg, offset: cOff, shape: QWTensorShape([maxBuckets * indexHeadDim], .f16)),
                bucketMembers: QWTensorValue(region: reg, offset: mOff, shape: QWTensorShape([maxBuckets], .i32)),
                bucketSizes: QWTensorValue(region: reg, offset: sOff, shape: QWTensorShape([max(1, maxBuckets / 4)], .i32))
            )
            st.lsaClusters = cs
            vSt?.lsaClusters = cs
        }
        return reg
    }
}
