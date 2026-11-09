// Types.swift — KV value/handle types (split from QWKVCache.swift).
import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc

public enum QWKVError: Error, CustomStringConvertible {
    case outOfPages(asked: Int, free: Int)
    case badArgument(String)
    public var description: String {
        switch self {
        case .outOfPages(let a, let f): return "QWKV: out of pages (ask \(a), free \(f))"
        case .badArgument(let s): return "QWKV: \(s)"
        }
    }
}

public enum QWKVMode: Sendable, CaseIterable {
    case f16, q8, q4

    public var bytesPerElement: Double {
        switch self {
        case .f16: return 2.0
        case .q8: return 1.0
        case .q4: return 0.5
        }
    }
}
public enum QWKVKind: String, Sendable, CaseIterable { case k = "K", v = "V" }

public typealias QWKVSeqID = UInt32

/// Zero-copy kernel-facing view of one (layer, kind) pool + one sequence's
/// device page table for exactly that pair. C09 binds pool.buffer + table.
public struct QWKVLayerView: Sendable {
    public let pool: QWMRegion           // whole pool extent (buffer + base)
    public let table: QWTensorValue      // int32 page ids, pool-local
    public let pageBytes: Int
    /// 4c: the pool's element encoding. Decoders bind their `q8mode`
    /// constant from this — never from a caller's free-form flag (no silent
    /// fallback: a q8 view CANNOT be read as f16; pageBytes is checked
    public let mode: QWKVMode
    public init(pool: QWMRegion, table: QWTensorValue, pageBytes: Int,
                mode: QWKVMode = .f16) {
        self.pool = pool; self.table = table; self.pageBytes = pageBytes
        self.mode = mode
    }
}

/// Refcounted page snapshot (prefix cache, C20): pins pages so grow/rollback
/// cannot recycle them while a consumer reads. Pinned across (layer, kind).
public struct QWKVSnap: Sendable {
    public let seq: QWKVSeqID
    public let tokenCount: Int
    let cache: QWKVCache
    let pins: [(layer: Int, kind: QWKVKind, ids: [Int32])]
}

/// Cold-path immutable bytes. Adoption copies into private destination pages;
/// it does not pin donor pages or require partial-page copy-on-write.
public struct QWKVPrivatePrefix: Sendable {
    public let tokenCount: Int
    public let memoryBytes: Int
    public let mode: QWKVMode
    let cache: QWKVCache
    let planes: [Data]

    init(tokenCount: Int, memoryBytes: Int, mode: QWKVMode, cache: QWKVCache, planes: [Data]) {
        self.tokenCount = tokenCount; self.memoryBytes = memoryBytes; self.mode = mode
        self.cache = cache; self.planes = planes
    }
}

/// Immutable view of written FP16 pages retained by the cache. The cache
/// keeps the physical pages alive until this value is released or deinit.
/// All ownership changes, including deinit on an arbitrary thread, enter the
public final class QWKVSharedPrefix: @unchecked Sendable {
    public let tokenCount: Int
    /// Number of referenced physical pages across all layer/KV-kind pools.
    /// This is logical per-prefix accounting; other prefixes may reference
    /// the same physical pages.
    public let pageCount: Int
    /// Referenced page bytes (`pageCount × bytesPerPage`), not unique bytes
    /// attributable to this prefix when other owners share those pages.
    public let memoryBytes: Int
    public let mode: QWKVMode
    let cache: QWKVCache
    let epoch: UUID
    let pages: [[Int32]]
    var released = false

    init(tokenCount: Int, cache: QWKVCache, epoch: UUID, pages: [[Int32]], mode: QWKVMode) {
        self.tokenCount = tokenCount
        self.cache = cache
        self.epoch = epoch
        self.pages = pages
        self.mode = mode
        self.pageCount = pages.reduce(0) { $0 + $1.count }
        self.memoryBytes = self.pageCount * cache.bytesPerPage
    }

    deinit { cache.releaseSharedPrefix(self) }
}

/// Handle = sequence identity + page-granular capacity. Grows in pages,
/// never moves.
public final class QWKVHandle: @unchecked Sendable {
    public let seq: QWKVSeqID
    public internal(set) var tokenCapacity: Int
    let cache: QWKVCache
    var finished = false
    init(seq: QWKVSeqID, capacity: Int, cache: QWKVCache) {
        self.seq = seq; self.tokenCapacity = capacity; self.cache = cache
    }
}
