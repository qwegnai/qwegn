// PlaneIO.swift — the one pack→memory mover. Verbatim planes land in the
// destination through an F_NOCACHE pread (single kernel→unified copy, no
// heap staging, no page-cache pollution); the mmap fallback is a single
// memcpy. Zero-copy mapping enforces the page-alignment law here.
import Foundation
import Darwin
import Metal
import QWHAL
import QWQuant

public enum PlaneIOError: Error, CustomStringConvertible {
    case layoutUnavailable(name: String, layout: String)
    case readFailed(name: String, message: String)

    public var description: String {
        switch self {
        case .layoutUnavailable(let name, let layout):
            return "PlaneIO: \(name) has no \(layout) plane"
        case .readFailed(let name, let message):
            return "PlaneIO: read failed for \(name): \(message)"
        }
    }
}

public final class PlaneIO: @unchecked Sendable {
    public let pack: QWQuant.Pack
    public let policy: LoadPolicy
    private let streamer: QWWeightStreamer?

    public init(pack: QWQuant.Pack, policy: LoadPolicy) {
        self.pack = pack
        self.policy = policy
        streamer = policy.directIO ? try? QWWeightStreamer(url: pack.url, unbuffered: true) : nil
    }

    public static var pageSize: Int { Int(sysconf(_SC_PAGESIZE)) }

    public func load(_ name: String, layout: PlaneLayout,
                     into dst: UnsafeMutableRawPointer) throws {
        let (offset, count) = try region(name, layout)
        try read(name, offset: offset, count: count, into: dst)
    }

    /// Parallel batch plane loader using concurrent `pread` across `(name, dst)` pairs.
    public func loadBatch(_ items: [(name: String, dst: UnsafeMutableRawPointer)],
                          layout: PlaneLayout) throws {
        try loadBatchTwoPhase(firstPhase: items, secondPhase: [], layout: layout, onFirstPhaseComplete: nil)
    }

    /// Single-wave two-phase parallel batch loader: dispatches `firstPhase` and `secondPhase`
    /// in one `DispatchQueue.concurrentPerform` wave and fires `onFirstPhaseComplete` after `firstPhase`.
    public func loadBatchTwoPhase(
        firstPhase: [(name: String, dst: UnsafeMutableRawPointer)],
        secondPhase: [(name: String, dst: UnsafeMutableRawPointer)],
        layout: PlaneLayout,
        onFirstPhaseComplete: (@Sendable () throws -> Void)? = nil
    ) throws {
        if firstPhase.isEmpty && secondPhase.isEmpty { return }
        if secondPhase.isEmpty && onFirstPhaseComplete == nil && firstPhase.count == 1 {
            try load(firstPhase[0].name, layout: layout, into: firstPhase[0].dst)
            return
        }
        struct Job: @unchecked Sendable {
            let name: String
            let offset: Int
            let count: Int
            let dst: UnsafeMutableRawPointer
        }
        final class PhaseBox: @unchecked Sendable {
            let lock = NSLock()
            var first: Error?
            var remainingFirst: Int
            init(remainingFirst: Int) {
                self.remainingFirst = remainingFirst
            }
            func record(_ e: Error) {
                lock.lock()
                if first == nil { first = e }
                lock.unlock()
            }
            func completeFirstJob() -> Bool {
                lock.lock()
                remainingFirst -= 1
                let fire = (remainingFirst == 0 && first == nil)
                lock.unlock()
                return fire
            }
        }
        func coalescePhase(_ items: [(name: String, dst: UnsafeMutableRawPointer)]) throws -> (jobs: [Job], bytes: Int) {
            guard !items.isEmpty else { return ([], 0) }
            var raw: [Job] = []
            raw.reserveCapacity(items.count)
            for item in items {
                let (off, cnt) = try region(item.name, layout)
                if let last = raw.last,
                   last.offset + last.count == off,
                   last.dst.advanced(by: last.count) == item.dst {
                    raw[raw.count - 1] = Job(name: last.name, offset: last.offset, count: last.count + cnt, dst: last.dst)
                } else {
                    raw.append(Job(name: item.name, offset: off, count: cnt, dst: item.dst))
                }
            }
            raw.sort { $0.offset < $1.offset }
            var coalesced: [Job] = []
            coalesced.reserveCapacity(raw.count)
            var bytes = 0
            for j in raw {
                bytes += j.count
                if let last = coalesced.last,
                   last.offset + last.count == j.offset,
                   last.dst.advanced(by: last.count) == j.dst {
                    coalesced[coalesced.count - 1] = Job(name: last.name, offset: last.offset, count: last.count + j.count, dst: last.dst)
                } else {
                    coalesced.append(j)
                }
            }
            return (coalesced, bytes)
        }
        let (c1, b1) = try coalescePhase(firstPhase)
        let (c2, b2) = try coalescePhase(secondPhase)
        let totalBytes = b1 + b2
        let page = max(16_384, Self.pageSize)
        let sliceBytes = totalBytes <= (32 << 20) ? (512 << 10) : (1 << 20)
        func sliceJobs(_ coalesced: [Job], bytes: Int) -> [Job] {
            guard !coalesced.isEmpty else { return [] }
            var out: [Job] = []
            out.reserveCapacity(max(coalesced.count, bytes / sliceBytes + 2))
            for j in coalesced {
                if j.count <= sliceBytes {
                    out.append(j)
                    continue
                }
                var curOff = j.offset
                var curDst = j.dst
                var rem = j.count
                while rem > 0 {
                    let rawEnd = curOff + sliceBytes
                    let alignedEnd = (rawEnd / page) * page
                    let step = (alignedEnd > curOff) ? min(rem, alignedEnd - curOff) : min(rem, sliceBytes)
                    out.append(Job(name: j.name, offset: curOff, count: step, dst: curDst))
                    curOff += step
                    curDst = curDst.advanced(by: step)
                    rem -= step
                }
            }
            return out
        }
        let jobs1 = sliceJobs(c1, bytes: b1)
        let jobs2 = sliceJobs(c2, bytes: b2)
        let firstCount = jobs1.count
        let jobsSnap = jobs1 + jobs2
        if jobsSnap.count == 1 {
            let j = jobsSnap[0]
            try read(j.name, offset: j.offset, count: j.count, into: j.dst)
            if firstCount == 1 {
                try onFirstPhaseComplete?()
            }
            return
        }
        let box = PhaseBox(remainingFirst: firstCount)
        if firstCount == 0, let onFirstPhaseComplete {
            try onFirstPhaseComplete()
        }
        DispatchQueue.concurrentPerform(iterations: jobsSnap.count) { idx in
            let j = jobsSnap[idx]
            do {
                try self.read(j.name, offset: j.offset, count: j.count, into: j.dst)
            } catch {
                box.record(error)
            }
            if let onFirstPhaseComplete, idx < firstCount, box.completeFirstJob() {
                do {
                    try onFirstPhaseComplete()
                } catch {
                    box.record(error)
                }
            }
        }
        if let err = box.first { throw err }
    }

    public func load(_ name: String, layout: PlaneLayout) throws -> Data {
        let (offset, count) = try region(name, layout)
        var out = Data(count: count)
        try out.withUnsafeMutableBytes { raw in
            try read(name, offset: offset, count: count, into: raw.baseAddress!)
        }
        return out
    }

    /// bf16 `raw_off` slice by element offset (chunked uploads / GPU converts).
    public func loadRaw(_ name: String, elemOffset: Int, elems: Int,
                        into dst: UnsafeMutableRawPointer) throws {
        let e = try pack.entry(name)
        guard let base = e.rawOff else { throw PlaneIOError.layoutUnavailable(name: name, layout: "raw") }
        try read(name, offset: base + elemOffset * 2, count: elems * 2, into: dst)
    }

    private func region(_ name: String, _ layout: PlaneLayout) throws -> (Int, Int) {
        let e = try pack.entry(name)
        guard let offset = layout.offset(e),
              let count = layout.byteCount(e) else {
            throw PlaneIOError.layoutUnavailable(name: name, layout: String(describing: layout))
        }
        return (offset, count)
    }

    private func read(_ name: String, offset: Int, count: Int,
                      into dst: UnsafeMutableRawPointer) throws {
        if let streamer {
            do {
                try streamer.streamTensor(offset: offset, byteCount: count, destination: dst)
                return
            } catch {
                try mmapCopy(name, offset: offset, count: count, into: dst)
                return
            }
        }
        try mmapCopy(name, offset: offset, count: count, into: dst)
    }

    private func mmapCopy(_ name: String, offset: Int, count: Int,
                          into dst: UnsafeMutableRawPointer) throws {
        do {
            try pack.withRawPointer(at: offset, byteCount: count) { src in
                dst.copyMemory(from: src, byteCount: count)
            }
        } catch {
            throw PlaneIOError.readFailed(name: name, message: String(describing: error))
        }
    }

    /// Page-aligned `bytesNoCopy` Metal view over the pack mmap covering `[offset, offset + byteCount)`,
    /// even when `offset` sits inside a page; returns `(buffer, subOffset, mappedLength)` or `nil`.
    public static func mapZeroCopyRange(pack: QWQuant.Pack, device: any MTLDevice,
                                        offset: Int, byteCount: Int) -> (buffer: any MTLBuffer, subOffset: Int, mappedLength: Int)? {
        let page = pageSize
        guard page > 0, byteCount > 0, offset >= 0 else { return nil }
        let start = offset - offset % page
        let subOffset = offset - start
        let (spanBytes, ov0) = subOffset.addingReportingOverflow(byteCount)
        let (padded, ov1) = spanBytes.addingReportingOverflow(page - 1)
        guard !ov0, !ov1 else { return nil }
        let length = (padded / page) * page
        let mapped: (any MTLBuffer)? = (try? pack.withRawPointer(at: start, byteCount: length) { raw -> (any MTLBuffer)? in
            guard UInt(bitPattern: raw) % UInt(page) == 0 else { return nil }
            return device.makeBuffer(
                bytesNoCopy: UnsafeMutableRawPointer(mutating: raw),
                length: length, options: .storageModeShared, deallocator: nil)
        }) ?? nil
        guard let buf = mapped else { return nil }
        QWResidencyRegistry.note(buf)
        return (buf, subOffset, length)
    }
}
