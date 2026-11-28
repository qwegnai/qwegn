import Foundation
import Metal
import QWCore
import QWHAL

/// C03 — QWAlloc: one shared-storage arena per engine launch; everything
/// GPU-visible is carved from it. Zero per-tensor MTLBuffers, 256 B tensor
/// alignment, activation rings, paged KV grid, GDN state slots, and hard
public enum QWAllocError: Error, CustomStringConvertible {
    case arenaTooLarge(asked: Int, limit: Int)
    case outOfSpace(region: String, asked: Int, free: Int)
    case misalignment(bytes: Int)

    public var description: String {
        switch self {
        case .arenaTooLarge(let a, let l): return "QWAlloc: asked \(a) B > working set \(l) B"
        case .outOfSpace(let r, let a, let f): return "QWAlloc: \(r) out of space (ask \(a), free \(f))"
        case .misalignment(let b): return "QWAlloc: misaligned request \(b)"
        }
    }
}

@inline(__always)
public func qwAlignUp(_ n: Int, _ a: Int = 256) -> Int { (n + a - 1) & ~(a - 1) }

@inline(__always)
private func qwCheckedAlignUp(_ n: Int, _ a: Int = 256) -> Int? {
    guard n >= 0, a > 0, (a & (a - 1)) == 0 else { return nil }
    let (sum, ov) = n.addingReportingOverflow(a - 1)
    guard !ov else { return nil }
    return sum & ~(a - 1)
}

/// A region of the arena: one MTLBuffer + byte extent. The DRY currency of
/// every launcher: (buffer, offset) pairs, never per-tensor buffers.
public struct QWMRegion: @unchecked Sendable {
    public let buffer: any MTLBuffer
    public let base: Int
    public let length: Int
    public var pointer: UnsafeMutableRawPointer {
        buffer.contents().advanced(by: base)
    }
    public init(buffer: any MTLBuffer, base: Int, length: Int) {
        self.buffer = buffer; self.base = base; self.length = length
    }
}

/// The zero-copy view passed to every kernel launch and Swift stage.
public struct QWTensorValue: Sendable {
    public let region: QWMRegion
    public let offset: Int          // byte offset within region
    public let shape: QWTensorShape
    public var bytes: Int { shape.totalBytes }
    public init(region: QWMRegion, offset: Int, shape: QWTensorShape) {
        self.region = region; self.offset = offset; self.shape = shape
    }
    /// Bind into an encoder at `index` (offset folded into setBuffer).
    public func bind(_ e: any MTLComputeCommandEncoder, index: Int) {
        e.setBuffer(region.buffer, offset: region.base + offset, index: index)
    }

    /// Zero-copy sub-view at `byteOffset` with `count` elements of `dtype`.
    @inline(__always)
    public func slice(byteOffset: Int, count: Int, dtype: QWDType? = nil) -> QWTensorValue {
        QWTensorValue(region: region, offset: offset + byteOffset, shape: QWTensorShape([count], dtype ?? shape.dtype))
    }
}

extension MTLComputeCommandEncoder {
    /// Binds a POD value into `index` using its stride.
    @inline(__always)
    public func setPod<T>(_ value: T, index: Int) {
        var copy = value
        withUnsafeBytes(of: &copy) { (raw: UnsafeRawBufferPointer) in
            guard let base = raw.baseAddress, raw.count > 0 else { return }
            setBytes(base, length: raw.count, index: index)
        }
    }

    /// Binds one or more `UInt32` values into consecutive buffer indices starting at `startIndex`.
    @inline(__always)
    public func setUInt32s(_ values: Int..., startIndex: Int) {
        precondition(startIndex >= 0, "setUInt32s startIndex must be non-negative")
        for (offset, v) in values.enumerated() {
            guard var u = UInt32(exactly: v) else {
                preconditionFailure("setUInt32s value \(v) out of UInt32 range")
            }
            setBytes(&u, length: 4, index: startIndex + offset)
        }
    }

    /// Binds one or more tensors into consecutive buffer indices starting at `startIndex`.
    @inline(__always)
    public func bindTensors(_ tensors: QWTensorValue..., startIndex: Int = 0) {
        for (offset, t) in tensors.enumerated() {
            t.bind(self, index: startIndex + offset)
        }
    }
}

/// Minimal shape record (rank ≤ 4, row-major, strides in elements).
public struct QWTensorShape: Sendable, Hashable {
    public let dims: [Int]
    public let dtype: QWDType
    public var checkedElementCount: Int? {
        var acc = 1
        for d in dims {
            guard d >= 0 else { return nil }
            let (m, ov) = acc.multipliedReportingOverflow(by: d)
            if ov { return nil }
            acc = m
        }
        return acc
    }
    public var checkedTotalBytes: Int? {
        guard let elems = checkedElementCount else { return nil }
        switch dtype {
        case .bf16, .f16:
            let (b, ov) = elems.multipliedReportingOverflow(by: 2)
            return ov ? nil : b
        case .f32, .i32:
            let (b, ov) = elems.multipliedReportingOverflow(by: 4)
            return ov ? nil : b
        case .q8_0, .hq8_g32:
            let groups = elems / 32, rem = elems % 32
            let (gBytes, ov1) = groups.multipliedReportingOverflow(by: 34)
            guard !ov1 else { return nil }
            let (b, ov2) = gBytes.addingReportingOverflow((rem * 34 + 16) / 32)
            return ov2 ? nil : b
        case .q4g:
            let groups = elems / 32, rem = elems % 32
            let (gBytes, ov1) = groups.multipliedReportingOverflow(by: 20)
            guard !ov1 else { return nil }
            let (b, ov2) = gBytes.addingReportingOverflow((rem * 20 + 16) / 32)
            return ov2 ? nil : b
        }
    }
    public var elementCount: Int {
        guard let n = checkedElementCount else { preconditionFailure("QWTensorShape elementCount overflow") }
        return n
    }
    public var totalBytes: Int {
        guard let b = checkedTotalBytes else { preconditionFailure("QWTensorShape totalBytes overflow") }
        return b
    }
    public init(_ dims: [Int], _ dtype: QWDType) {
        precondition((1...4).contains(dims.count), "rank 1…4")
        precondition(dims.allSatisfy { $0 >= 0 }, "QWTensorShape dims must be non-negative")
        self.dims = dims; self.dtype = dtype
    }
}

/// Bump sub-allocator over one or more regions. Free-list-free by design: phases
/// reserve once at start, hot paths never alloc (counters prove it).
public final class QWSubAllocator: @unchecked Sendable {
    public let region: QWMRegion
    public let regions: [QWMRegion]
    private var slabIndex: Int = 0
    private var cursor: Int = 0
    private let lock = NSLock()
    /// Counters the CI gate watches: allocs after `markSteadyState()`.
    private var _steadyStateAllocs: Int = 0
    public var steadyStateAllocs: Int { lock.withLock { _steadyStateAllocs } }
    private var steady = false
    /// High-water bookkeeping for the memory-pressure contract.
    private var _usedBytes: Int = 0
    public var usedBytes: Int { lock.withLock { _usedBytes } }
    public let totalCapacityBytes: Int
    public var freeBytes: Int { lock.withLock { max(0, totalCapacityBytes - _usedBytes) } }

    public init(region: QWMRegion) {
        precondition(region.length >= 0, "QWSubAllocator region length must be non-negative")
        self.region = region
        self.regions = [region]
        self.totalCapacityBytes = region.length
    }

    public init(regions: [QWMRegion]) {
        precondition(!regions.isEmpty, "QWSubAllocator requires at least one region")
        var total = 0
        for r in regions {
            precondition(r.length >= 0, "QWSubAllocator region length must be non-negative")
            let (sum, ov) = total.addingReportingOverflow(r.length)
            precondition(!ov, "QWSubAllocator totalCapacityBytes overflow")
            total = sum
        }
        self.region = regions[0]
        self.regions = regions
        self.totalCapacityBytes = total
    }

    private func allocateLocked(bytes: Int) throws -> (region: QWMRegion, offset: Int) {
        if steady { _steadyStateAllocs += 1 }   // gate fails this if != 0
        guard bytes >= 0 else {
            throw QWAllocError.misalignment(bytes: bytes)
        }
        let freeInCurrent = slabIndex < regions.count ? max(0, regions[slabIndex].length - cursor) : 0
        guard let n = qwCheckedAlignUp(bytes, 256) else {
            throw QWAllocError.outOfSpace(region: "sub", asked: bytes, free: freeInCurrent)
        }
        var candSlab = slabIndex
        var candCursor = cursor
        while candSlab < regions.count {
            let (end, ov) = candCursor.addingReportingOverflow(n)
            if !ov, end <= regions[candSlab].length {
                let reg = regions[candSlab]
                let off = candCursor
                slabIndex = candSlab
                cursor = end
                let (newUsed, usedOv) = _usedBytes.addingReportingOverflow(n)
                _usedBytes = usedOv ? totalCapacityBytes : newUsed
                return (reg, off)
            }
            candSlab += 1
            candCursor = 0
        }
        throw QWAllocError.outOfSpace(region: "sub", asked: n, free: freeInCurrent)
    }

    /// Carve `bytes` (auto-aligned to 256) across slabs. Returns byte offset in the selected slab.
    @discardableResult
    public func allocate(bytes: Int) throws -> Int {
        let (_, off) = try lock.withLock { try allocateLocked(bytes: bytes) }
        return off
    }

    /// View of a carved tensor at the current cursor (advances to next slab if multi-slab and needed).
    public func allocate(_ shape: QWTensorShape) throws -> QWTensorValue {
        guard let bytes = shape.checkedTotalBytes else {
            throw QWAllocError.outOfSpace(region: "sub", asked: Int.max, free: freeBytes)
        }
        let (reg, off) = try lock.withLock { try allocateLocked(bytes: bytes) }
        return QWTensorValue(region: reg, offset: off, shape: shape)
    }

    public func markSteadyState() { lock.withLock { steady = true } }

    /// C31: return the cursor to the region base. Only legal when no slice
    /// carved from this region is still live (single owner per session batch).
    public func reset() {
        lock.withLock {
            slabIndex = 0
            cursor = 0
            _usedBytes = 0
        }
    }
}

/// Ring of fixed-size slabs (decode activation pipeline): O(1) rotate, no
/// locking in the hot path (single-threaded encode per engine contract).
public final class QWRingAllocator: @unchecked Sendable {
    public let region: QWMRegion
    public let slabBytes: Int
    public let depth: Int
    private var next = 0
    public private(set) var steadyStateAcquires: Int = 0
    private var steady = false

    init(region: QWMRegion, slabBytes: Int, depth: Int) {
        precondition(slabBytes > 0 && depth > 0, "QWRingAllocator requires slabBytes > 0 and depth > 0")
        guard let aligned = qwCheckedAlignUp(slabBytes, 256) else {
            preconditionFailure("QWRingAllocator slabBytes overflow")
        }
        let (total, ov) = aligned.multipliedReportingOverflow(by: depth)
        precondition(!ov && total <= region.length, "QWRingAllocator exceeds region length")
        self.region = region
        self.slabBytes = aligned
        self.depth = depth
    }

    /// Next slab view (wraps; reuses oldest — caller guarantees no hazard
    /// beyond `depth` in-flight steps; the engine's encode loop is serial).
    public func acquire(_ shape: QWTensorShape) -> QWTensorValue {
        precondition(shape.totalBytes <= slabBytes, "QWRingAllocator acquire \(shape.totalBytes) > slabBytes \(slabBytes)")
        if steady { steadyStateAcquires += 1 }   // ring acquire is NOT an alloc;
        // counter here just proves the ring shape fits the steady loop size.
        let off = next * slabBytes
        next = (next + 1) % depth
        return QWTensorValue(region: region, offset: off, shape: shape)
    }
    public func markSteadyState() { steady = true }
}

/// Fixed-size page arena for the KV cache (C10): a grid of equal pages,
/// each 128 B-aligned; pages handed out by index, free-list is a stack.
public final class QWPageArena: @unchecked Sendable {
    public let region: QWMRegion
    public let pageSize: Int
    public let pageCount: Int
    private let lock = NSLock()
    private var freeStack: [Int]
    private var isFree: [Bool]

    public init(region: QWMRegion, pageSize: Int, pageCount: Int) {
        precondition(pageSize > 0 && pageCount >= 0, "QWPageArena requires pageSize > 0 and pageCount >= 0")
        guard let aligned = qwCheckedAlignUp(pageSize, 128) else {
            preconditionFailure("QWPageArena pageSize overflow")
        }
        let (total, ov) = aligned.multipliedReportingOverflow(by: pageCount)
        precondition(!ov && total <= region.length, "QWPageArena exceeds region length")
        self.region = region
        self.pageSize = aligned
        self.pageCount = pageCount
        self.freeStack = (0..<pageCount).reversed()
        self.isFree = [Bool](repeating: true, count: pageCount)
    }
    public func allocPage() -> Int? {
        lock.withLock {
            guard let id = freeStack.popLast() else { return nil }
            isFree[id] = false
            return id
        }
    }
    public func freePage(_ i: Int) {
        lock.withLock {
            guard i >= 0, i < pageCount, !isFree[i] else { return }
            isFree[i] = true
            freeStack.append(i)
        }
    }
    public var freePages: Int { lock.withLock { freeStack.count } }
}

/// The arena. One per process: one giant shared buffer (or per-region shared
/// buffers when `totalBytes > hal.device.maxBufferLength`), split at fixed
/// offsets into weights / activation / KV / GDN-state regions.
public final class QWArena: @unchecked Sendable {
    public enum ArenaHazardPolicy: Sendable, Equatable {
        /// Track mutable/shared arenas (`bigBuffer`, `activations`, `kv`, `gdnStates`);
        /// untrack split read-only `weights` slabs when `total > maxBufferLength`.
        case safeDefault
        /// Force `.hazardTrackingModeTracked` on all arena buffers (`QWG_TRACKED_ARENAS=1`).
        case forceTracked
        /// Force `.hazardTrackingModeUntracked` on all arena buffers (`QWG_UNTRACKED_ARENAS=1`).
        case forceUntracked
    }

    public let hal: QWHal
    public let totalBytes: Int
    public let regionSizes: (weights: Int, act: Int, kv: Int, gdn: Int)
    public let hazardPolicy: ArenaHazardPolicy
    private let bigBuffer: any MTLBuffer
    public let allBuffers: [any MTLBuffer]
    public private(set) var committedBytes: Int = 0
    // PERSISTENT sub-allocators (static-analysis C4: freshly-built
    // QWSubAllocators reset their cursor to 0 on every accessor call —
    // steadyAllocs read 0 forever AND any second caller silently aliased
    public let weightsSub: QWSubAllocator
    public let activationsSub: QWSubAllocator
    public let kvSub: QWSubAllocator
    public let kvVSub: QWSubAllocator?
    public let kvAuxSub: QWSubAllocator?
    public let gdnSub: QWSubAllocator

    static func weightSlabCap(maxBufferLength maxBuf: Int) -> Int {
        let alignedMax = max(256, (maxBuf / 256) * 256)
        return alignedMax > (2048 << 20)
            ? max(256, ((alignedMax - (1536 << 20)) / 256) * 256)
            : alignedMax
    }

    public static func resolveHazardPolicy(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> ArenaHazardPolicy {
        if environment["QWG_TRACKED_ARENAS"] == "1" { return .forceTracked }
        if environment["QWG_UNTRACKED_ARENAS"] == "1" { return .forceUntracked }
        return .safeDefault
    }

    static func resourceOptions(
        forWeightSlab: Bool,
        policy: ArenaHazardPolicy
    ) -> MTLResourceOptions {
        switch policy {
        case .forceTracked:
            return [.storageModeShared, .hazardTrackingModeTracked]
        case .forceUntracked:
            return [.storageModeShared, .hazardTrackingModeUntracked]
        case .safeDefault:
            return forWeightSlab
                ? [.storageModeShared, .hazardTrackingModeUntracked]
                : .storageModeShared
        }
    }

    /// Sizes decided once at launch by the caller's allocation plan.
    /// All region capacities and bases are rounded to 256 B.
    public init(
        hal: QWHal,
        weights: Int,
        activations: Int,
        kv: Int,
        gdnStates: Int,
        keepAlive: Bool = true,
        hazardPolicy: ArenaHazardPolicy? = nil
    ) throws {
        guard let w = qwCheckedAlignUp(weights, 256),
              let a = qwCheckedAlignUp(activations, 256),
              let k = qwCheckedAlignUp(kv, 256),
              let g = qwCheckedAlignUp(gdnStates, 256) else {
            throw QWAllocError.misalignment(bytes: min(weights, min(activations, min(kv, gdnStates))))
        }
        let (wa, ov0) = w.addingReportingOverflow(a)
        let (wak, ov1) = wa.addingReportingOverflow(k)
        let (total, ov2) = wak.addingReportingOverflow(g)
        let limit = hal.caps.maxWorkingSetBytes
        guard !ov0, !ov1, !ov2, total <= limit else {
            throw QWAllocError.arenaTooLarge(asked: (ov0 || ov1 || ov2) ? Int.max : total, limit: limit)
        }
        let maxBuf = max(256, (Int(hal.device.maxBufferLength) / 256) * 256)
        let resolvedPolicy = hazardPolicy ?? Self.resolveHazardPolicy()
        self.hal = hal
        self.totalBytes = total
        self.regionSizes = (w, a, k, g)
        self.hazardPolicy = resolvedPolicy
        if total <= maxBuf {
            let bigOpts = Self.resourceOptions(forWeightSlab: false, policy: resolvedPolicy)
            guard let buf = hal.device.makeBuffer(length: max(256, total), options: bigOpts) else {
                throw QWAllocError.outOfSpace(region: "arena", asked: total, free: 0)
            }
            self.committedBytes = total
            self.bigBuffer = buf
            self.allBuffers = [buf]
            self.weightsSub = QWSubAllocator(region: QWMRegion(buffer: buf, base: 0, length: w))
            self.activationsSub = QWSubAllocator(region: QWMRegion(buffer: buf, base: w, length: a))
            self.kvSub = QWSubAllocator(region: QWMRegion(buffer: buf, base: w + a, length: k))
            self.kvVSub = nil
            self.kvAuxSub = nil
            self.gdnSub = QWSubAllocator(region: QWMRegion(buffer: buf, base: w + a + k, length: g))
        } else {
            func allocBuf(_ len: Int, _ name: String, isWeightSlab: Bool = false) throws -> any MTLBuffer {
                guard len <= maxBuf else {
                    throw QWAllocError.outOfSpace(region: name, asked: len, free: maxBuf)
                }
                let opts = Self.resourceOptions(forWeightSlab: isWeightSlab, policy: resolvedPolicy)
                guard let b = hal.device.makeBuffer(length: max(256, len), options: opts) else {
                    throw QWAllocError.outOfSpace(region: name, asked: len, free: 0)
                }
                return b
            }
            var bufs: [any MTLBuffer] = []
            var wRegions: [QWMRegion] = []
            var slackBytes = 0
            if w <= maxBuf {
                let wBuf = try allocBuf(w, "weights", isWeightSlab: true)
                bufs.append(wBuf)
                wRegions.append(QWMRegion(buffer: wBuf, base: 0, length: w))
            } else {
                // Split weights across maxBuf-bounded slabs with up to 1.5 GiB boundary slack
                // so no single contiguous tensor allocation fails at a slab seam.
                let slabCap = Self.weightSlabCap(maxBufferLength: maxBuf)
                slackBytes = (min(1536 << 20, max(0, limit - total)) / 256) * 256
                let (wWithSlack, ovSlack) = w.addingReportingOverflow(slackBytes)
                if ovSlack { slackBytes = 0 }
                var remainingW = ovSlack ? w : wWithSlack
                var slabIdx = 0
                while remainingW > 0 {
                    let chunk = min(slabCap, remainingW)
                    let sb = try allocBuf(chunk, "weights.\(slabIdx)", isWeightSlab: true)
                    bufs.append(sb)
                    wRegions.append(QWMRegion(buffer: sb, base: 0, length: chunk))
                    remainingW -= chunk
                    slabIdx += 1
                }
            }
            self.committedBytes = total + slackBytes
            let aBuf = try allocBuf(a, "activations")
            let gBuf = try allocBuf(g, "gdn")
            bufs.append(contentsOf: [aBuf, gBuf])
            self.bigBuffer = wRegions[0].buffer
            self.weightsSub = QWSubAllocator(regions: wRegions)
            self.activationsSub = QWSubAllocator(region: QWMRegion(buffer: aBuf, base: 0, length: a))
            self.gdnSub = QWSubAllocator(region: QWMRegion(buffer: gBuf, base: 0, length: g))
            if k <= maxBuf {
                let kBuf = try allocBuf(k, "kv")
                bufs.append(kBuf)
                self.kvSub = QWSubAllocator(region: QWMRegion(buffer: kBuf, base: 0, length: k))
                self.kvVSub = nil
                self.kvAuxSub = nil
            } else {
                let halfK = ((k / 2) / 256) * 256
                let kVLen = min(maxBuf, halfK)
                let rem = k - kVLen
                if rem <= maxBuf {
                    let kKBuf = try allocBuf(rem, "kv.k")
                    let kVBuf = try allocBuf(kVLen, "kv.v")
                    bufs.append(contentsOf: [kKBuf, kVBuf])
                    self.kvSub = QWSubAllocator(region: QWMRegion(buffer: kKBuf, base: 0, length: rem))
                    self.kvVSub = QWSubAllocator(region: QWMRegion(buffer: kVBuf, base: 0, length: kVLen))
                    self.kvAuxSub = nil
                } else {
                    let kKL = maxBuf
                    let kVL = maxBuf
                    let auxL = qwAlignUp(k - kKL - kVL)
                    let kKBuf = try allocBuf(kKL, "kv.k")
                    let kVBuf = try allocBuf(kVL, "kv.v")
                    let auxBuf = try allocBuf(auxL, "kv.aux")
                    bufs.append(contentsOf: [kKBuf, kVBuf, auxBuf])
                    self.kvSub = QWSubAllocator(region: QWMRegion(buffer: kKBuf, base: 0, length: kKL))
                    self.kvVSub = QWSubAllocator(region: QWMRegion(buffer: kVBuf, base: 0, length: kVL))
                    self.kvAuxSub = QWSubAllocator(region: QWMRegion(buffer: auxBuf, base: 0, length: auxL))
                }
            }
            self.allBuffers = bufs
        }
        if keepAlive {
            hal.makeResident(allBuffers)
        }
    }

    deinit {
        hal.evictResident(allBuffers)
    }

    /// Faults in and GPU-wires every page now. A fresh multi-GB arena otherwise pays its zero-fill and
    /// wiring (~200 ms per 12 GB) inside the first command buffer that touches it.
    public func prefault() {
        guard let cb = hal.queue.makeCommandBuffer(), let blit = cb.makeBlitCommandEncoder() else { return }
        for buf in allBuffers where buf.length > 0 {
            blit.fill(buffer: buf, range: 0..<buf.length, value: 0)
        }
        blit.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }

    /// Region accessors — fixed extents, no shared cursor between regions.
    /// Return the PERSISTENT sub-allocators (identity-stable: markSteadyState
    /// and steadyStateAllocs survive across calls — the C4 fix).
    public func weightsRegion() -> QWSubAllocator { weightsSub }
    public func activationsRegion() -> QWSubAllocator { activationsSub }
    public func kvRegion() -> QWSubAllocator { kvSub }
    public func gdnRegion() -> QWSubAllocator { gdnSub }

    public func ring(slabBytes: Int, depth: Int) throws -> QWRingAllocator {
        guard slabBytes > 0, depth > 0, let aligned = qwCheckedAlignUp(slabBytes, 256) else {
            throw QWAllocError.misalignment(bytes: slabBytes)
        }
        let (need, ov) = aligned.multipliedReportingOverflow(by: depth)
        guard !ov, need <= activationsSub.region.length else {
            throw QWAllocError.outOfSpace(region: "ring", asked: ov ? Int.max : need, free: activationsSub.region.length)
        }
        return QWRingAllocator(region: activationsSub.region, slabBytes: slabBytes, depth: depth)
    }
    public func pages(pageSize: Int, count: Int) throws -> QWPageArena {
        guard pageSize > 0, count >= 0, let aligned = qwCheckedAlignUp(pageSize, 128) else {
            throw QWAllocError.misalignment(bytes: pageSize)
        }
        let (need, ov) = aligned.multipliedReportingOverflow(by: count)
        guard !ov, need <= kvSub.region.length else {
            throw QWAllocError.outOfSpace(region: "pages", asked: ov ? Int.max : need, free: kvSub.region.length)
        }
        return QWPageArena(region: kvSub.region, pageSize: pageSize, pageCount: count)
    }

    /// Zero-copy CPU pointer into the arena (mapped shared storage).
    public func cpuPointer(in sub: QWSubAllocator, offset: Int) -> UnsafeMutableRawPointer {
        sub.region.buffer.contents().advanced(by: sub.region.base + offset)
    }

    /// CPU pointer of an allocated tensor (shared storage is mapped — this is
    /// the copy-in/copy-out door; no staging buffers, no blits).
    public func cpuPointer(of t: QWTensorValue) -> UnsafeMutableRawPointer {
        t.region.buffer.contents().advanced(by: t.region.base + t.offset)
    }

    /// Copy fixture/load bytes into an allocated arena tensor.
    public func copyIn(_ t: QWTensorValue, from d: Data) {
        let (regEnd, ovReg) = t.offset.addingReportingOverflow(d.count)
        precondition(t.offset >= 0 && d.count <= t.bytes && !ovReg && regEnd <= t.region.length,
                     "copyIn \(d.count) > tensor \(t.bytes)")
        guard !d.isEmpty else { return }
        let dst = cpuPointer(of: t)
        withExtendedLifetime(t.region.buffer) {
            d.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                guard let base = raw.baseAddress else { return }
                dst.copyMemory(from: base, byteCount: raw.count)
            }
        }
    }

    /// Zero-allocation copy from a contiguous array into an allocated arena tensor.
    @inline(__always)
    public func write<Element>(_ t: QWTensorValue, from array: [Element]) {
        let (byteCount, ov) = array.count.multipliedReportingOverflow(by: MemoryLayout<Element>.stride)
        precondition(!ov, "write byteCount overflow")
        let (regEnd, ovReg) = t.offset.addingReportingOverflow(byteCount)
        precondition(t.offset >= 0 && byteCount <= t.bytes && !ovReg && regEnd <= t.region.length,
                     "write \(byteCount) > tensor \(t.bytes)")
        guard byteCount > 0 else { return }
        let dst = cpuPointer(of: t)
        withExtendedLifetime(t.region.buffer) {
            array.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                guard let base = raw.baseAddress else { return }
                dst.copyMemory(from: base, byteCount: byteCount)
            }
        }
    }

    /// Read `count` elements of `Element` from a shared-storage arena tensor.
    @inline(__always)
    public func read<Element>(_ t: QWTensorValue, as type: Element.Type = Element.self, count: Int) -> [Element] {
        precondition(count >= 0, "read count must be non-negative")
        guard count > 0 else { return [] }
        let (byteCount, ov) = count.multipliedReportingOverflow(by: MemoryLayout<Element>.stride)
        precondition(!ov, "read byteCount overflow")
        let (regEnd, ovReg) = t.offset.addingReportingOverflow(byteCount)
        precondition(t.offset >= 0 && byteCount <= t.bytes && !ovReg && regEnd <= t.region.length,
                     "read \(byteCount) > tensor \(t.bytes)")
        let base = UnsafeRawPointer(cpuPointer(of: t))
        return [Element](unsafeUninitializedCapacity: count) { dst, initialized in
            for i in 0..<count {
                dst[i] = base.loadUnaligned(fromByteOffset: i * MemoryLayout<Element>.stride, as: Element.self)
            }
            initialized = count
        }
    }
}
