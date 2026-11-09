// DriverColdEvents.swift — event-pipelined cold-expert staging for 1-row decode.
// Default on (tuning.coldEvents; QW_COLD_EVENTS=0 disables): each MoE route CB signals an MTLSharedEvent and commits without
// a host wait; a worker remaps ids, loads misses, writes promotions and signals the expert CB's wait value.
// A tuned delay CB (QW_COLD_EVENT_BRIDGE=0 disables) lets that signal land before the GPU waits; resident
// experts are mapped to bank slots on the GPU and run inside it, overlapping the remap and SSD reads
// (tuning.coldEventOverlap; QW_COLD_EVENT_OVERLAP=0 disables).
import Foundation
import QuartzCore
import Metal
import QWAlloc
import QWHAL
import QWGEMM

final class ColdEventPipeline: @unchecked Sendable {
    let event: any MTLSharedEvent
    let promoList: any MTLBuffer
    let promoCapacity: Int
    let bridgeTick: any MTLBuffer
    /// GPU-mapped resident ids ([rows · K] Int32, -1 for a miss) for the overlapped resident pass.
    let residentIds: any MTLBuffer
    /// Delay length in dependent FMAs; tuned by `tuneBridge` from the step's timings.
    private(set) var bridgeIters: UInt32 = 3000
    /// Host signal time per job; nil when the job read from SSD (late by design).
    private var signalTimes: [CFTimeInterval?] = []
    /// This step's bridge CBs, in route order (encode thread only).
    var bridgeCBs: [any MTLCommandBuffer] = []
    static let bridgeEnabled = ProcessInfo.processInfo.environment["QW_COLD_EVENT_BRIDGE"] != "0"
    private var nextValue: UInt64 = 0
    private let cond = NSCondition()
    private var jobs: [(wait: UInt64, run: () throws -> Bool)] = []
    private var isStopped = false
    private var workerExited = false
    private var failure: (any Error)?
    private var ioUs: Double = 0
    private var waitUs: Double = 0
    private var staged = 0
    static let spinTimeout: TimeInterval = 30

    init?(device: any MTLDevice, promoCapacity: Int = 64) {
        guard let event = device.makeSharedEvent(),
              let list = device.makeBuffer(length: 4 * (1 + 2 * promoCapacity), options: .storageModeShared),
              let tick = device.makeBuffer(length: 16, options: .storageModePrivate),
              let resIds = device.makeBuffer(length: 4 * promoCapacity, options: .storageModePrivate)
        else { return nil }
        self.residentIds = resIds
        self.bridgeTick = tick
        self.event = event
        self.promoList = list
        self.promoCapacity = promoCapacity
        let t = Thread { [unowned self] in self.workerLoop() }
        t.qualityOfService = .userInteractive
        t.name = "qwg.cold-events"
        t.start()
    }

    /// Stops the background cold-event worker thread and waits for it to exit cleanly.
    func shutdown() {
        cond.lock()
        if !isStopped {
            isStopped = true
            cond.broadcast()
        }
        while !workerExited {
            cond.wait()
        }
        cond.unlock()
    }

    deinit {
        shutdown()
    }

    /// Reserves (ready, done): the GPU signals `ready`, the worker signals `done`.
    func reserve() -> (ready: UInt64, done: UInt64) {
        nextValue += 2
        return (nextValue - 1, nextValue)
    }

    /// `run` returns whether it read from SSD.
    func enqueue(ready: UInt64, _ run: @escaping () throws -> Bool) {
        cond.lock()
        jobs.append((ready, run))
        cond.signal()
        cond.unlock()
    }

    func addStats(ioUs: Double, staged: Int) {
        cond.lock(); self.ioUs += ioUs; self.staged += staged; cond.unlock()
    }

    /// Drains failure and counters; call only after the step's last CB completed.
    func takeStepResult() -> (failure: (any Error)?, ioUs: Double, waitUs: Double, staged: Int) {
        cond.lock(); defer { cond.unlock() }
        let r = (failure, ioUs, waitUs, staged)
        failure = nil; ioUs = 0; waitUs = 0; staged = 0
        return r
    }

    /// Retunes the delay from the step's bridge CBs and the host signal times: grow while the
    /// host signals after the bridge ends, shrink on spare slack. Call after the step completed.
    func tuneBridge() {
        cond.lock()
        let signals = signalTimes
        signalTimes.removeAll(keepingCapacity: true)
        cond.unlock()
        let bridges = bridgeCBs
        bridgeCBs.removeAll(keepingCapacity: true)
        guard Self.bridgeEnabled, signals.count == bridges.count, !signals.isEmpty else { return }
        var late = 0
        var slack: [Double] = []
        for (cb, signal) in zip(bridges, signals) {
            guard let t = signal else { continue }
            let end = cb.gpuEndTime
            guard end > 0 else { return }
            if t > end { late += 1 } else { slack.append(end - t) }
        }
        // A late layer costs a ~115 us wake, slack only its own length: shrink only when none was late.
        let iters = Double(bridgeIters)
        if late > 1 {
            bridgeIters = UInt32(min(iters * 1.2 + 100, 12_000))
        } else if late == 0, let minSlack = slack.min(), minSlack > 8e-6 {
            bridgeIters = UInt32(max(iters * 0.97, 500))
        }
    }

    private func workerLoop() {
        while true {
            cond.lock()
            while jobs.isEmpty && !isStopped { cond.wait() }
            if isStopped && jobs.isEmpty {
                workerExited = true
                cond.broadcast()
                cond.unlock()
                return
            }
            let job = jobs.removeFirst()
            let failed = failure != nil
            cond.unlock()
            let t0 = Date()
            var ready = event.signaledValue >= job.wait
            while !ready, Date().timeIntervalSince(t0) < Self.spinTimeout {
                ready = event.signaledValue >= job.wait
            }
            let waited = Date().timeIntervalSince(t0) * 1e6
            var err: (any Error)?
            var didIO = false
            if !ready {
                err = QWEngineDriverError.state("cold event \(job.wait) never signalled")
            } else if !failed {
                do { didIO = try job.run() } catch { err = error }
            }
            if err != nil || failed { promoList.contents().storeBytes(of: UInt32(0), as: UInt32.self) }
            cond.lock()
            waitUs += waited
            if let err, failure == nil { failure = err }
            cond.unlock()
            // Always signal, so the GPU never waits forever on a failed job.
            event.signaledValue = job.wait + 1
            let signalled = CACurrentMediaTime()
            cond.lock(); signalTimes.append(didIO ? nil : signalled); cond.unlock()
        }
    }
}

extension QWEngineDriver {
    func coldEventsUsable(_ moe: MoEW, rows: Int) -> Bool {
        let totalRoutes = rows * moe.expertsPerTok
        guard tuning.coldEvents, !Self.decodeProfileEnabled, !QWMoEChain.coldTimingProxy,
              (1...4).contains(rows), stepCBs != nil,
              let st = moe.coldStreamer, st.isDynamic, st.stagingSlots >= totalRoutes,
              moe.coldGateUp != nil, moe.coldDown != nil,
              st.guExpertBytes % 16 == 0, st.downExpertBytes % 16 == 0 else { return false }
        if coldEvents == nil { coldEvents = ColdEventPipeline(device: hal.device) }
        return coldEvents != nil && totalRoutes <= coldEvents!.promoCapacity
    }

    /// Ends the route CB with a signal, hands the id rewrite + IO to the worker,
    /// and opens the expert CB behind the worker's completion value. With `resident`, the route CB
    /// also maps the ids to resident bank slots on the GPU and the resident experts run in the
    /// bridge CB, overlapping the host remap and SSD reads; the expert CB then needs only the misses.
    /// Returns whether `resident` was encoded.
    @discardableResult
    func stageColdExpertsEvent(moe: MoEW, s: MoEScratch, rows: Int = 1,
                               resident: ((any MTLComputeCommandEncoder, QWTensorValue) throws -> Void)? = nil) throws -> Bool {
        guard let pipe = coldEvents, let streamer = moe.coldStreamer,
              let curCB = activeCB, let curEnc = activeEnc else {
            throw QWEngineDriverError.state("cold event staging without pipeline or active CB")
        }
        let routes = moe.expertsPerTok
        let count = rows * routes
        var residentIds: QWTensorValue?
        if resident != nil, tuning.coldEventOverlap, count <= pipe.promoCapacity,
           let map = streamer.attachGPUSlotMap(device: hal.device) {
            let ids = QWTensorValue(region: QWMRegion(buffer: pipe.residentIds, base: 0, length: 4 * pipe.promoCapacity),
                                    offset: 0, shape: QWTensorShape([count], .i32))
            curEnc.setComputePipelineState(try hal.pipeline("qwg_moe_resident_ids"))
            s.idx.bind(curEnc, index: 0)
            curEnc.setBuffer(map, offset: 0, index: 1)
            ids.bind(curEnc, index: 2)
            var cnt = UInt32(count), experts = UInt32(streamer.totalExperts)
            curEnc.setBytes(&cnt, length: 4, index: 3)
            curEnc.setBytes(&experts, length: 4, index: 4)
            curEnc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: count, height: 1, depth: 1))
            residentIds = ids
        }
        let v = pipe.reserve()
        curEnc.endEncoding()
        curCB.encodeSignalEvent(pipe.event, value: v.ready)
        curCB.commit()
        stepCBs?.append(curCB)
        // The signal fires only when its CB completes, so the delay is a CB of its own.
        if let residentIds, let resident {
            guard let (resCB, resEnc) = hal.begin(.serial) else {
                throw QWEngineDriverError.state("cold event resident cmdbuffer")
            }
            try resident(resEnc, residentIds)
            if ColdEventPipeline.bridgeEnabled {
                resEnc.setComputePipelineState(try hal.pipeline("qwg_cold_event_bridge"))
                resEnc.setBuffer(pipe.bridgeTick, offset: 0, index: 0)
                var iters = pipe.bridgeIters
                resEnc.setBytes(&iters, length: 4, index: 1)
                resEnc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                hal.end(resCB, resEnc)
                pipe.bridgeCBs.append(resCB)
            } else {
                hal.end(resCB, resEnc)
            }
        } else if ColdEventPipeline.bridgeEnabled, let (bridgeCB, bridge) = hal.begin(.serial) {
            bridge.setComputePipelineState(try hal.pipeline("qwg_cold_event_bridge"))
            bridge.setBuffer(pipe.bridgeTick, offset: 0, index: 0)
            var iters = pipe.bridgeIters
            bridge.setBytes(&iters, length: 4, index: 1)
            bridge.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            hal.end(bridgeCB, bridge)
            pipe.bridgeCBs.append(bridgeCB)
        }
        let idxPtr = UnsafeMutableRawPointer(s.idx.region.buffer.contents()
            .advanced(by: s.idx.region.base + s.idx.offset))
        let list = pipe.promoList.contents().assumingMemoryBound(to: UInt32.self)
        let cap = pipe.promoCapacity
        pipe.enqueue(ready: v.ready) {
            let ids = idxPtr.assumingMemoryBound(to: UInt32.self)
            let prep = try streamer.prepareSpan(idxPtr: ids, startRow: 0, totalRows: rows, routes: routes)
            defer { streamer.pendingPromotions.removeAll(keepingCapacity: true) }
            guard prep.nextRow == rows, streamer.pendingPromotions.count <= cap else {
                throw QWEngineDriverError.state("cold event span did not fit staging")
            }
            for (i, p) in streamer.pendingPromotions.enumerated() {
                list[1 + 2 * i] = UInt32(p.staging)
                list[2 + 2 * i] = UInt32(p.bank)
            }
            list[0] = UInt32(streamer.pendingPromotions.count)
            let t0 = Date()
            if prep.stagedCount > 0 { try streamer.executeBatch(prep.guBatch + prep.downBatch) }
            pipe.addStats(ioUs: Date().timeIntervalSince(t0) * 1e6, staged: prep.stagedCount)
            return prep.stagedCount > 0
        }
        guard let next = hal.begin(.serial, waitingFor: pipe.event, value: v.done) else {
            throw QWEngineDriverError.state("cold event cmdbuffer")
        }
        activeCB = next.0
        activeEnc = next.1
        return residentIds != nil
    }

    /// Staging→bank copies for the promotion list the worker wrote (count may be 0).
    func encodeEventPromotions(moe: MoEW) throws {
        guard let pipe = coldEvents, let enc = activeEnc, let streamer = moe.coldStreamer,
              let coldGU = moe.coldGateUp, let coldDown = moe.coldDown else {
            throw QWEngineDriverError.state("cold event promotion without encoder or cold bank")
        }
        enc.setComputePipelineState(try hal.pipeline("qwg_moe_promote_copy"))
        for (src, dst, bytes) in [(coldGU.blocks, moe.gateUp.blocks, streamer.guExpertBytes),
                                  (coldDown.blocks, moe.down.blocks, streamer.downExpertBytes)] {
            src.bind(enc, index: 0)
            dst.bind(enc, index: 1)
            enc.setBuffer(pipe.promoList, offset: 0, index: 2)
            var vec = UInt32(bytes / 16)
            enc.setBytes(&vec, length: 4, index: 3)
            enc.dispatchThreads(MTLSize(width: min(Int(vec), 65_536), height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        enc.memoryBarrier(scope: .buffers)
    }
}
