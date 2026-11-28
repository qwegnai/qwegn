// QWQSACacheState.swift — Layer-4 QSA attention K/V and index cache state with coarse LSA clusters.
import Foundation
import QWAlloc
import QWCore
import QWGEMM
import QWHAL
import QWRope

/// Errors raised during QSA cache configuration and block selection.
public enum QWQSAError: Error, Equatable, CustomStringConvertible {
    case selectionLimitExceeded(positions: Int, limit: Int)
    case kvModeNotImplemented(String)

    public var description: String {
        switch self {
        case .selectionLimitExceeded(let positions, let limit):
            return "QWQSAError selectionLimitExceeded: positions=\(positions) limit=\(limit)"
        case .kvModeNotImplemented(let mode):
            return "QWQSAError kvModeNotImplemented: \(mode)"
        }
    }
}

/// Geometry & RoPE contract required to allocate and partition a `QWQSACacheState`.
public protocol QWQSAMixerSpec: Sendable {
    var heads: Int { get }
    var kvHeads: Int { get }
    var headDim: Int { get }
    var indexHeads: Int { get }
    var indexHeadDim: Int { get }
    var rotaryDim: Int { get }
    var compress: Int { get }
    var blockTopK: Int { get }
    var bankRows: Int { get }
    var rope: QWRopeCPU { get }
}

/// KV storage mode for QSA full-attention layers (`f32`, `f16` default, `q8`, `q4`).
/// `indexK` and `pooled` always store `f16` (2 bytes/element).
public enum QWQSAKVStorageMode: String, Sendable {
    case f32
    case f16
    case q8
    case q4

    public static func resolve(_ requested: String?) throws -> QWQSAKVStorageMode {
        guard let raw = requested?.lowercased(), !raw.isEmpty, raw != "auto" else {
            return .f16
        }
        guard let mode = QWQSAKVStorageMode(rawValue: raw) else {
            throw QWQSAError.kvModeNotImplemented(requested ?? raw)
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

    public var sparseGatedNAXSplitKernelName: String {
        switch self {
        case .f32: return "qwg_attn_sparse_gated_nax_split_f32"
        case .f16: return "qwg_attn_sparse_gated_nax_split_f16"
        case .q8:  return "qwg_attn_sparse_gated_nax_split_q8"
        case .q4:  return "qwg_attn_sparse_gated_nax_split_q4"
        }
    }

    public static let sparseGatedNAXCombineKernelName = "qwg_attn_sparse_gated_nax_combine"

    /// Bytes per token across 12 trunk + 1 MTP QSA layers (K + V + meta + indexK + pooled/4) plus shared RoPE.
    public var perTokenBytes: Int {
        switch self {
        case .f32: return 65_536
        case .f16: return 31_296
        case .q8:  return 18_096
        case .q4:  return 11_752
        }
    }
}

public typealias FNKVStorageMode = QWQSAKVStorageMode

/// Per-layer decode state of one QSA full-attention layer: K/V cache (`f32`/`f16`/`q8`/`q4`)
/// and QSA index caches (`f16` raw index keys per token, `f16` pooled block keys per block).
public final class QWQSACacheState: @unchecked Sendable {
    public struct LSAClusterState: Sendable {
        public let centroids: QWTensorValue
        public let bucketMembers: QWTensorValue
        public let bucketSizes: QWTensorValue
        public let emeansScratch: QWCoarsePoolIndex.Scratch?
        public let maxUnits: Int
        public var isEmeans: Bool { emeansScratch != nil }

        public init(centroids: QWTensorValue,
                    bucketMembers: QWTensorValue,
                    bucketSizes: QWTensorValue,
                    emeansScratch: QWCoarsePoolIndex.Scratch? = nil,
                    maxUnits: Int = 0) {
            self.centroids = centroids
            self.bucketMembers = bucketMembers
            self.bucketSizes = bucketSizes
            self.emeansScratch = emeansScratch
            self.maxUnits = maxUnits
        }
    }

    public static let defaultMaxSplits = 32
    public let fullCapacity: Int, hidden: Int, rows: Int
    public let maxSplits: Int
    public private(set) var capacity: Int
    public let kvMode: QWQSAKVStorageMode
    public var lsaClusters: LSAClusterState? {
        didSet {
            if !isPartitioned { fullLSAClusters = lsaClusters }
        }
    }
    private var fullLSAClusters: LSAClusterState?
    private var isPartitioned: Bool = false
    public private(set) var k: QWTensorValue, v: QWTensorValue
    public private(set) var kMeta: QWTensorValue, vMeta: QWTensorValue
    public private(set) var indexK: QWTensorValue
    public private(set) var pooled: QWTensorValue
    private let fullK: QWTensorValue, fullV: QWTensorValue
    private let fullKMeta: QWTensorValue, fullVMeta: QWTensorValue
    private let fullIndexK: QWTensorValue, fullPooled: QWTensorValue
    public let cos: QWTensorValue, sin: QWTensorValue
    public let bcos: QWTensorValue, bsin: QWTensorValue
    public let rot32: QWTensorValue
    public let src: QWTensorValue
    public let q: QWTensorValue, gateF: QWTensorValue
    public let idxQ: QWTensorValue
    public let list: QWTensorValue
    public let qsaScores: QWTensorValue
    public let attnOut: QWTensorValue
    public let splitPartO: QWTensorValue
    public let splitPartML: QWTensorValue

    public init(arena: QWArena, mixer a: any QWQSAMixerSpec, hidden: Int, capacity: Int,
                kvMode: QWQSAKVStorageMode = .f16,
                sharedRope: (cos: QWTensorValue, sin: QWTensorValue, bcos: QWTensorValue, bsin: QWTensorValue)? = nil,
                sharedQSAScores: QWTensorValue? = nil) throws {
        precondition(capacity % a.compress == 0, "capacity must be a multiple of the compress block")
        self.fullCapacity = capacity
        self.capacity = capacity
        self.hidden = hidden
        self.rows = 1
        self.maxSplits = Self.defaultMaxSplits
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
        func up32(_ t: QWTensorValue, _ vals: [Float]) {
            arena.write(t, from: vals)
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
        splitPartO = try act.allocate(QWTensorShape([1, a.heads, maxSplits, a.headDim], .f32))
        splitPartML = try act.allocate(QWTensorShape([1, a.heads, maxSplits, 2], .f32))
    }

    /// Row-tiled state for a verify pass of `rows` tokens sharing `base`'s K/V and index caches.
    public init(sharingCachesOf base: QWQSACacheState, arena: QWArena, mixer a: any QWQSAMixerSpec, rows: Int,
                sharedQSAScores: QWTensorValue? = nil) throws {
        precondition(rows >= 1, "rows must be >= 1")
        self.fullCapacity = base.fullCapacity
        self.capacity = base.capacity
        self.hidden = base.hidden
        self.rows = rows
        self.maxSplits = Self.defaultMaxSplits
        self.kvMode = base.kvMode
        self.fullLSAClusters = base.fullLSAClusters ?? base.lsaClusters
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
        let splitRows = min(rows, 8)
        splitPartO = try act.allocate(QWTensorShape([splitRows, a.heads, maxSplits, a.headDim], .f32))
        splitPartML = try act.allocate(QWTensorShape([splitRows, a.heads, maxSplits, 2], .f32))
    }

    /// Selects a contiguous KV/indexK/pooled partition `slot` out of `totalSlots`, or restores full capacity when `totalSlots <= 1`.
    public func selectPartition(slot: Int, totalSlots: Int, mixer a: any QWQSAMixerSpec) {
        if totalSlots <= 1 {
            let wasPartitioned = isPartitioned
            isPartitioned = false
            capacity = fullCapacity
            k = fullK; v = fullV; kMeta = fullKMeta; vMeta = fullVMeta
            indexK = fullIndexK; pooled = fullPooled
            lsaClusters = fullLSAClusters
            if wasPartitioned, let cs = fullLSAClusters {
                memset(cs.bucketSizes.region.pointer.advanced(by: cs.bucketSizes.offset), 0, cs.bucketSizes.bytes)
                memset(cs.bucketMembers.region.pointer.advanced(by: cs.bucketMembers.offset), 0, cs.bucketMembers.bytes)
            }
            return
        }
        if !isPartitioned {
            fullLSAClusters = lsaClusters
        }
        isPartitioned = true
        lsaClusters = nil
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
        states: [(QWQSACacheState, QWQSACacheState?)],
        clusterBlock: Int = QWLSAConfig.emeansClusterBlock,
        maxBatchRows: Int = 32
    ) throws -> QWMRegion {
        let useEmeans = (clusterBlock >= QWLSAConfig.emeansClusterBlock)
        let maxUnits = useEmeans
            ? QWCoarsePoolIndex.maxMacros(forMaxContext: kvCap)
            : max(1, (kvCap + 511) / 512)
        let maxBuckets = maxUnits * 64
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
            let mOff = reserve(useEmeans ? maxUnits * 2048 : maxBuckets * 4)
            let sOff = reserve(useEmeans ? maxBuckets * 4 : maxBuckets)
            offsets.append((cOff, mOff, sOff))
        }
        let candCap = 14336
        let scSizes = QWCoarsePoolIndex.scratchByteSizes(
            maxBatchRows: maxBatchRows, maxClusters: maxBuckets, candCap: candCap)
        let scOff = useEmeans ? reserve(scSizes.cScore) : 0
        let shOff = useEmeans ? reserve(scSizes.uHist) : 0
        let ssOff = useEmeans ? reserve(scSizes.candState) : 0
        let spOff = useEmeans ? reserve(scSizes.candPools) : 0
        let suOff = useEmeans ? reserve(scSizes.candU) : 0
        let totalBytes = max(4096, cursor)
        guard let buf = hal.device.makeBuffer(length: totalBytes, options: .storageModeShared) else {
            throw QWAllocError.outOfSpace(region: "fn-lsa-clusters", asked: totalBytes, free: 0)
        }
        buf.label = "fn-lsa-clusters"
        memset(buf.contents(), 0, totalBytes)
        let reg = QWMRegion(buffer: buf, base: 0, length: totalBytes)
        let sharedScratch: QWCoarsePoolIndex.Scratch? = useEmeans ? QWCoarsePoolIndex.Scratch(
            cScore: QWTensorValue(region: reg, offset: scOff, shape: QWTensorShape([maxBatchRows, maxBuckets], .i32)),
            uHist: QWTensorValue(region: reg, offset: shOff, shape: QWTensorShape([maxBatchRows, 512], .i32)),
            candState: QWTensorValue(region: reg, offset: ssOff, shape: QWTensorShape([maxBatchRows * 2], .i32)),
            candPools: QWTensorValue(region: reg, offset: spOff, shape: QWTensorShape([maxBatchRows, candCap], .i32)),
            candU: QWTensorValue(region: reg, offset: suOff, shape: QWTensorShape([maxBatchRows, candCap], .i32)),
            maxBatchRows: maxBatchRows,
            clusterStride: maxBuckets,
            candCap: candCap
        ) : nil
        for (i, (st, vSt)) in states.enumerated() {
            let (cOff, mOff, sOff) = offsets[i]
            let cs = LSAClusterState(
                centroids: QWTensorValue(region: reg, offset: cOff, shape: QWTensorShape([maxBuckets * indexHeadDim], .f16)),
                bucketMembers: QWTensorValue(region: reg, offset: mOff,
                                             shape: QWTensorShape([useEmeans ? maxUnits * 512 : maxBuckets], .i32)),
                bucketSizes: QWTensorValue(region: reg, offset: sOff,
                                           shape: QWTensorShape([useEmeans ? maxBuckets : max(1, maxBuckets / 4)], .i32)),
                emeansScratch: sharedScratch,
                maxUnits: maxUnits
            )
            st.lsaClusters = cs
            vSt?.lsaClusters = cs
        }
        return reg
    }
}

/// Transitional Wave 3A alias while `QWFlashNext` migrates to `QWAttn`.
public typealias FNGPUAttnDecodeState = QWQSACacheState
