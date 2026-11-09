// QWResidency.swift — Keeps long-lived GPU allocations wired.
// The iogpu collector unwires allocations no command buffer referenced recently once the process wires
// a large share of RAM (MoE slabs: ~90 GiB), so idle layers re-wire at ~190 GB/s on the critical path
// (hundreds of ms in early decode and prefill). Referencing every long-lived allocation from each encoder
// keeps them warm (a queue-attached MTLResidencySet does not: the set is reclaimed whole). Between requests
// an idle heartbeat commits an empty encoder referencing the same set; otherwise the first request after
// a few idle seconds re-wires it all (~650 ms for ~90 GiB) before its first kernel.

import Foundation
import Metal

/// Zero-copy (mmap-backed) buffers created by loaders that have no HAL at
/// hand. The owning engine adopts them via `QWHal.adoptPendingResidency()`.
public enum QWResidencyRegistry {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var pending = NSHashTable<AnyObject>.weakObjects()

    public static func note(_ buffer: any MTLBuffer) {
        lock.lock(); defer { lock.unlock() }
        pending.add(buffer)
    }

    static func drain(device: any MTLDevice) -> [any MTLBuffer] {
        lock.lock(); defer { lock.unlock() }
        var out: [any MTLBuffer] = []
        var keep: [AnyObject] = []
        for obj in pending.allObjects {
            guard let b = obj as? any MTLBuffer else { continue }
            if b.device === device { out.append(b) } else { keep.append(obj) }
        }
        pending.removeAllObjects()
        for k in keep { pending.add(k) }
        return out
    }
}

/// Env `QWG_KEEPALIVE=0` disables keep-alive (A/B and diagnostics).
let keepAliveEnabled = ProcessInfo.processInfo.environment["QWG_KEEPALIVE"] != "0"
/// Idle heartbeat period; `QWG_KEEPALIVE_IDLE_MS=0` disables the heartbeat.
let keepAliveIdleMs: Int = Int(ProcessInfo.processInfo.environment["QWG_KEEPALIVE_IDLE_MS"] ?? "") ?? 250
/// After this long without real GPU work the heartbeat lets the collector reclaim the memory.
let keepAliveIdleMaxSec: Double = Double(ProcessInfo.processInfo.environment["QWG_KEEPALIVE_IDLE_MAX_S"] ?? "") ?? 60

extension QWHal {
    /// Registers `buffers` as long-lived; every encoder begun by this HAL references them.
    public func makeResident(_ buffers: [any MTLBuffer]) {
        guard !buffers.isEmpty else { return }
        residencyLock.lock(); defer { residencyLock.unlock() }
        for b in buffers where keepAliveIDs.insert(ObjectIdentifier(b)).inserted { keepAlive.append(b) }
        keepAliveLastTouch = CFAbsoluteTimeGetCurrent()
        startKeepAliveHeartbeatLocked()
    }

    private func startKeepAliveHeartbeatLocked() {
        guard keepAliveEnabled, keepAliveIdleMs > 0, keepAliveHeartbeat == nil else { return }
        // An encoder with no dispatch is dropped before it reaches the GPU, so the heartbeat runs a no-op kernel.
        guard let lib = try? device.makeLibrary(source: "kernel void qw_keepalive_nop() {}", options: nil),
              let fn = lib.makeFunction(name: "qw_keepalive_nop"),
              let pso = try? device.makeComputePipelineState(function: fn) else { return }
        keepAliveNop = pso
        let period = DispatchTimeInterval.milliseconds(keepAliveIdleMs)
        let timer = DispatchSource.makeTimerSource(queue: DispatchQueue.global(qos: .utility))
        timer.schedule(deadline: .now() + period, repeating: period, leeway: .milliseconds(keepAliveIdleMs / 4))
        timer.setEventHandler { [weak self] in self?.keepAliveIdleTick() }
        keepAliveHeartbeat = timer
        timer.resume()
    }

    private func keepAliveIdleTick() {
        residencyLock.lock(); defer { residencyLock.unlock() }
        let idleSec = CFAbsoluteTimeGetCurrent() - keepAliveLastTouch
        guard idleSec * 1000 >= Double(keepAliveIdleMs), idleSec < keepAliveIdleMaxSec, !keepAlive.isEmpty,
              let pso = keepAliveNop,
              let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return }
        enc.useResources(keepAlive, usage: .read)
        enc.setComputePipelineState(pso)
        enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
    }

    /// Unregisters `buffers` so their memory can be released.
    public func evictResident(_ buffers: [any MTLBuffer]) {
        guard !buffers.isEmpty else { return }
        let gone = Set(buffers.map { ObjectIdentifier($0) })
        residencyLock.lock(); defer { residencyLock.unlock() }
        guard !keepAliveIDs.isDisjoint(with: gone) else { return }
        keepAliveIDs.subtract(gone)
        keepAlive.removeAll { gone.contains(ObjectIdentifier($0)) }
    }

    /// Adopts every zero-copy loader buffer created on this device since the last call.
    public func adoptPendingResidency() {
        makeResident(QWResidencyRegistry.drain(device: device))
    }

    public var keepAliveCount: Int {
        residencyLock.lock(); defer { residencyLock.unlock() }
        return keepAlive.count
    }

    /// References every long-lived allocation from `enc` (cheap: one call, no GPU work).
    @inline(__always)
    public func touchKeepAlive(_ enc: any MTLComputeCommandEncoder) {
        guard keepAliveEnabled else { return }
        residencyLock.lock(); defer { residencyLock.unlock() }
        keepAliveLastTouch = CFAbsoluteTimeGetCurrent()
        if !keepAlive.isEmpty { enc.useResources(keepAlive, usage: .read) }
    }
}
