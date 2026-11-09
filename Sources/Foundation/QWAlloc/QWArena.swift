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

/// Ping-pong buffer pair for recurrent / causal history state that alternates read and write slots each pass.
public struct QWPingPongBuffer: Sendable {
    public let pair: [QWTensorValue]
    public var parity: Int = 0

    public init(_ a: QWTensorValue, _ b: QWTensorValue) {
        self.pair = [a, b]
    }

    public init(pair: [QWTensorValue]) {
        precondition(pair.count == 2, "QWPingPongBuffer requires 2 tensors")
        self.pair = pair
    }

    @inline(__always)
    public var current: QWTensorValue { pair[parity] }

    @inline(__always)
    public var next: QWTensorValue { pair[1 - parity] }

    @inline(__always)
    public mutating func advance() { parity = 1 - parity }

    public subscript(index: Int) -> QWTensorValue { pair[index] }
}

extension MTLComputeCommandEncoder {
    /// Binds a POD value into `index` using its stride.
    @inline(__always)
    public func setPod<T>(_ value: T, index: Int) {
        var copy = value
        withUnsafeBytes(of: &copy) { raw in
            setBytes(raw.baseAddress!, length: raw.count, index: index)
        }
    }

    /// Binds one or more `UInt32` values into consecutive buffer indices starting at `startIndex`.
    @inline(__always)
    public func setUInt32s(_ values: Int..., startIndex: Int) {
        for (offset, v) in values.enumerated() {
            var u = UInt32(v)
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
    public var elementCount: Int { dims.reduce(1, *) }
    public var totalBytes: Int { Int((Double(elementCount) * dtype.bytesPerParam).rounded()) }
    public init(_ dims: [Int], _ dtype: QWDType) {
        precondition((1...4).contains(dims.count), "rank 1…4")
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
    public private(set) var steadyStateAllocs: Int = 0
    private var steady = false
    /// High-water bookkeeping for the memory-pressure contract.
    public private(set) var usedBytes: Int = 0
    public let totalCapacityBytes: Int
    public var freeBytes: Int { max(0, totalCapacityBytes - usedBytes) }

    public init(region: QWMRegion) {
        self.region = region
        self.regions = [region]
        self.totalCapacityBytes = region.length
    }

    public init(regions: [QWMRegion]) {
        precondition(!regions.isEmpty, "QWSubAllocator requires at least one region")
        self.region = regions[0]
        self.regions = regions
        self.totalCapacityBytes = regions.reduce(0) { $0 + $1.length }
    }

    /// Carve `bytes` (auto-aligned to 256) from the current slab. Returns byte offset in `region`.
    @discardableResult
    public func allocate(bytes: Int) throws -> Int {
        lock.lock(); defer { lock.unlock() }
        if steady { steadyStateAllocs += 1 }   // gate fails this if != 0
        let n = qwAlignUp(bytes)
        guard cursor + n <= region.length else {
            throw QWAllocError.outOfSpace(region: "sub", asked: n,
                                          free: region.length - cursor)
        }
        let off = cursor
        cursor += n
        usedBytes += n
        return off
    }

    /// View of a carved tensor at the current cursor (advances to next slab if multi-slab and needed).
    public func allocate(_ shape: QWTensorShape) throws -> QWTensorValue {
        if regions.count == 1 {
            let off = try allocate(bytes: shape.totalBytes)
            return QWTensorValue(region: region, offset: off, shape: shape)
        }
        lock.lock(); defer { lock.unlock() }
        if steady { steadyStateAllocs += 1 }
        let n = qwAlignUp(shape.totalBytes)
        while slabIndex < regions.count && cursor + n > regions[slabIndex].length {
            slabIndex += 1
            cursor = 0
        }
        guard slabIndex < regions.count else {
            let freeInCurrent = slabIndex < regions.count ? (regions[slabIndex].length - cursor) : 0
            throw QWAllocError.outOfSpace(region: "sub", asked: n, free: freeInCurrent)
        }
        let reg = regions[slabIndex]
        let off = cursor
        cursor += n
        usedBytes += n
        return QWTensorValue(region: reg, offset: off, shape: shape)
    }

    public func markSteadyState() { lock.lock(); steady = true; lock.unlock() }

    /// C31: return the cursor to the region base. Only legal when no slice
    /// carved from this region is still live (single owner per session batch).
    public func reset() {
        lock.lock(); defer { lock.unlock() }
        slabIndex = 0
        cursor = 0
        usedBytes = 0
    }
    public func resetSteadyCounters() { lock.lock(); steadyStateAllocs = 0; lock.unlock() }
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
        self.region = region
        self.slabBytes = qwAlignUp(slabBytes)
        self.depth = depth
    }

    /// Next slab view (wraps; reuses oldest — caller guarantees no hazard
    /// beyond `depth` in-flight steps; the engine's encode loop is serial).
    public func acquire(_ shape: QWTensorShape) -> QWTensorValue {
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
    private var freeStack: [Int]

    public init(region: QWMRegion, pageSize: Int, pageCount: Int) {
        self.region = region
        self.pageSize = qwAlignUp(pageSize, 128)
        self.pageCount = pageCount
        self.freeStack = (0..<pageCount).reversed()
    }
    public func allocPage() -> Int? { freeStack.popLast().map { $0 } }
    public func freePage(_ i: Int) { freeStack.append(i) }
    public var freePages: Int { freeStack.count }
    public func pageView(_ i: Int, _ shape: QWTensorShape) -> QWTensorValue {
        QWTensorValue(region: region, offset: i * pageSize, shape: shape)
    }
}

/// The arena. One per process: one giant shared buffer (or per-region shared
/// buffers when `totalBytes > hal.device.maxBufferLength`), split at fixed
/// offsets into weights / activation / KV / GDN-state regions.
public final class QWArena: @unchecked Sendable {
    public let hal: QWHal
    public let totalBytes: Int
    public let regionSizes: (weights: Int, act: Int, kv: Int, gdn: Int)
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

    /// Sizes decided once at launch by the caller's allocation plan.
    /// All region capacities and bases are rounded to 256 B.
    public init(hal: QWHal, weights: Int, activations: Int, kv: Int, gdnStates: Int, keepAlive: Bool = true) throws {
        let w = qwAlignUp(weights), a = qwAlignUp(activations)
        let k = qwAlignUp(kv), g = qwAlignUp(gdnStates)
        let total = w + a + k + g
        let limit = hal.caps.maxWorkingSetBytes
        guard total <= limit else { throw QWAllocError.arenaTooLarge(asked: total, limit: limit) }
        let maxBuf = max(256, (Int(hal.device.maxBufferLength) / 256) * 256)
        self.hal = hal
        self.totalBytes = total
        self.regionSizes = (w, a, k, g)
        self.committedBytes = total
        if total <= maxBuf {
            guard let buf = hal.device.makeBuffer(length: max(256, total), options: .storageModeShared) else {
                throw QWAllocError.outOfSpace(region: "arena", asked: total, free: 0)
            }
            self.bigBuffer = buf
            self.allBuffers = [buf]
            self.weightsSub = QWSubAllocator(region: QWMRegion(buffer: buf, base: 0, length: w))
            self.activationsSub = QWSubAllocator(region: QWMRegion(buffer: buf, base: w, length: a))
            self.kvSub = QWSubAllocator(region: QWMRegion(buffer: buf, base: w + a, length: k))
            self.kvVSub = nil
            self.kvAuxSub = nil
            self.gdnSub = QWSubAllocator(region: QWMRegion(buffer: buf, base: w + a + k, length: g))
        } else {
            func allocBuf(_ len: Int, _ name: String) throws -> any MTLBuffer {
                guard len <= maxBuf else {
                    throw QWAllocError.outOfSpace(region: name, asked: len, free: maxBuf)
                }
                guard let b = hal.device.makeBuffer(length: max(256, len), options: .storageModeShared) else {
                    throw QWAllocError.outOfSpace(region: name, asked: len, free: 0)
                }
                return b
            }
            var bufs: [any MTLBuffer] = []
            var wRegions: [QWMRegion] = []
            if w <= maxBuf {
                let wBuf = try allocBuf(w, "weights")
                bufs.append(wBuf)
                wRegions.append(QWMRegion(buffer: wBuf, base: 0, length: w))
            } else {
                // Split weights across maxBuf-bounded slabs with 1.5 GiB boundary slack
                // so no single contiguous tensor allocation fails at a slab seam.
                let slabCap = max(256, ((maxBuf - (1536 << 20)) / 256) * 256)
                let boundarySlack = min(1536 << 20, max(0, limit - total))
                var remainingW = w + boundarySlack
                var slabIdx = 0
                while remainingW > 0 {
                    let chunk = min(slabCap, remainingW)
                    let sb = try allocBuf(chunk, "weights.\(slabIdx)")
                    bufs.append(sb)
                    wRegions.append(QWMRegion(buffer: sb, base: 0, length: chunk))
                    remainingW -= chunk
                    slabIdx += 1
                }
            }
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
        let need = qwAlignUp(slabBytes) * depth
        guard need <= regionSizes.act else {
            throw QWAllocError.outOfSpace(region: "ring", asked: need, free: regionSizes.act)
        }
        return QWRingAllocator(region: activationsSub.region, slabBytes: slabBytes, depth: depth)
    }
    public func pages(pageSize: Int, count: Int) throws -> QWPageArena {
        let need = qwAlignUp(pageSize, 128) * count
        guard need <= regionSizes.kv else {
            throw QWAllocError.outOfSpace(region: "pages", asked: need, free: regionSizes.kv)
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
        precondition(d.count <= t.bytes, "copyIn \(d.count) > tensor \(t.bytes)")
        memcpy(cpuPointer(of: t), (d as NSData).bytes, d.count)
    }

    /// Zero-allocation copy from a contiguous array into an allocated arena tensor.
    @inline(__always)
    public func write<Element>(_ t: QWTensorValue, from array: [Element]) {
        let byteCount = array.count * MemoryLayout<Element>.stride
        precondition(byteCount <= t.bytes, "write \(byteCount) > tensor \(t.bytes)")
        guard byteCount > 0 else { return }
        _ = array.withUnsafeBytes { raw in
            memcpy(cpuPointer(of: t), raw.baseAddress!, byteCount)
        }
    }

    /// Read `count` elements of `Element` from a shared-storage arena tensor.
    @inline(__always)
    public func read<Element>(_ t: QWTensorValue, as type: Element.Type = Element.self, count: Int) -> [Element] {
        let ptr = cpuPointer(of: t).bindMemory(to: Element.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }
}
