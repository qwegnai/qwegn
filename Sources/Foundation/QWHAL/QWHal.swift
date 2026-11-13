// QWHal.swift — Metal 4.0 hardware abstraction layer for Qwen3.8 engines.
// Owns MTLDevice, command queues, GPU family capability probes, metallib
// pipeline caching (with post-init lock-free freeze), and Flash-Next door
// command-buffer batching (single-drain decode orchestration).
// Nothing above QWHAL calls MTLCreateSystemDefaultDevice directly.

import Foundation
import Metal
import QuartzCore
import QWCore

public struct QWDeviceCaps: Sendable {
    public let name: String
    public let familyApple9: Bool
    public let familyApple10: Bool
    public let familyApple11: Bool
    public let unifiedMemory: Bool
    public let maxWorkingSetBytes: Int
    public let threadsPerTG: Int
    /// Measured DRAM read bandwidth, GB/s (S3 fair run; 585 on this machine).
    public let measuredReadBWBps: Double
    /// Measured L2-resident read bandwidth, GB/s (~950 at 32 MB).
    public let measuredL2BWBps: Double

    public var isMetal4TensorCapable: Bool { familyApple10 }
}

extension QWDeviceCaps {
    /// The device's capabilities, read from the device itself.
    public static func probe(_ dev: MTLDevice) -> QWDeviceCaps {
        var apple9 = false, apple10 = false, apple11 = false
        if #available(macOS 14.0, *) { apple9 = dev.supportsFamily(.apple9) }
        if #available(macOS 15.0, *) { apple10 = dev.supportsFamily(.apple10) }
        if #available(macOS 26.0, *), let f11 = MTLGPUFamily(rawValue: 1011) {
            apple11 = dev.supportsFamily(f11)
        }
        return QWDeviceCaps(
            name: dev.name,
            familyApple9: apple9,
            familyApple10: apple10,
            familyApple11: apple11,
            unifiedMemory: dev.hasUnifiedMemory,
            maxWorkingSetBytes: Int(dev.recommendedMaxWorkingSetSize),
            threadsPerTG: 1024,
            measuredReadBWBps: 585e9,
            measuredL2BWBps: 950e9
        )
    }

    /// The capabilities of this machine, found once. They do not change
    /// while the process runs, and finding them means creating a Metal
    /// device: a request must not pay for that every time.
    public static let current: QWDeviceCaps? = {
        guard let dev = MTLCreateSystemDefaultDevice() else { return nil }
        return probe(dev)
    }()
}

public enum QWHalError: Error, CustomStringConvertible {
    case noDevice
    case metallibNotFound(String)
    case metallibLoadFailed(String)
    case metallibBasisMismatch(path: String, expected: QWWeightBasis, found: QWWeightBasis)
    case kernelMissing(String)

    public var description: String {
        switch self {
        case .noDevice: return "QWHAL: no Metal device"
        case .metallibNotFound(let p): return "QWHAL: metallib not found at \(p)"
        case .metallibLoadFailed(let p): return "QWHAL: metallib load failed: \(p)"
        case .metallibBasisMismatch(let p, let expected, let found):
            return "QWHAL: metallib at \(p) is in the \(found) basis, but the model needs \(expected)"
        case .kernelMissing(let n): return "QWHAL: kernel '\(n)' missing from metallib"
        }
    }
}

/// One HAL instance per process. Sendable through immutable fields; the
/// queue is internally synchronized by Metal itself.
public final class QWHal: @unchecked Sendable {
    public enum ComputeDispatchMode: Sendable {
        case serial
        case concurrent
    }

    public let device: any MTLDevice
    public let caps: QWDeviceCaps
    public let queue: any MTLCommandQueue
    /// Long-lived allocations every encoder references (see QWResidency.swift).
    let residencyLock = NSLock()
    var keepAlive: [any MTLResource] = []
    var keepAliveIDs = Set<ObjectIdentifier>()
    var keepAliveLastTouch: CFAbsoluteTime = 0
    var keepAliveHeartbeat: (any DispatchSourceTimer)?
    var keepAliveNop: (any MTLComputePipelineState)?
    public let metallib: any MTLLibrary
    public let geometry: QWModelGeometry
    private let functionConstants: MTLFunctionConstantValues

    /// Pipeline cache: kernel name to state (created once, reused forever).
    /// After freezePipelines() the map is read-only and lock-free on hits.
    private final class PSOStore: @unchecked Sendable {
        var map: [String: any MTLComputePipelineState] = [:]
        var frozen = false
        /// Lock acquisitions observed after freeze — must stay flat in a step.
        var lockCount = 0
    }
    private let psoStore = PSOStore()
    private let psoLock = NSLock()

    /// Freeze the pipeline cache so post-freeze hits take no lock. Idempotent.
    public func freezePipelines() {
        psoLock.lock(); defer { psoLock.unlock() }
        psoStore.frozen = true
    }

    /// The constants the standard path added, as values. The builder is
    /// write-only, so this is the readable seam over the same numbers.
    public static func standardConstants(geometry: QWModelGeometry)
        -> (hasQGate: Bool, qStride: UInt32, attnScale: Float) {
        (geometry.attnOutputGate,
         geometry.attnOutputGate ? 2 : 1,
         Float(1.0 / Double(geometry.headDim).squareRoot()))
    }

    public static func makeFunctionConstants(geometry: QWModelGeometry) -> MTLFunctionConstantValues {
        let fc = MTLFunctionConstantValues()
        func setU32(_ v: Int, at idx: Int) {
            var u = UInt32(v)
            fc.setConstantValue(&u, type: .uint, index: idx)
        }
        setU32(geometry.hidden, at: 0)
        setU32(geometry.vocab, at: 1)
        setU32(geometry.intermediate, at: 2)
        setU32(geometry.numLayers, at: 3)
        setU32(geometry.fullAttentionInterval, at: 4)
        setU32(geometry.qHeads, at: 5)
        setU32(geometry.kvHeads, at: 6)
        setU32(geometry.headDim, at: 7)
        setU32(geometry.qProjOut, at: 8)
        setU32(geometry.rotaryDims, at: 9)
        var eps = Float(geometry.rmsEps)
        fc.setConstantValue(&eps, type: .float, index: 10)
        setU32(geometry.maxPositionEmbeddings, at: 11)
        setU32(geometry.gdnQKHeads, at: 12)
        setU32(geometry.gdnVHeads, at: 13)
        setU32(geometry.gdnHeadDim, at: 14)
        setU32(geometry.gdnQKVChannels, at: 15)
        setU32(geometry.gdnZChannels, at: 16)
        setU32(geometry.gdnConvKernel, at: 17)
        setU32(geometry.qHeads / max(1, geometry.kvHeads), at: 18)
        setU32(geometry.gdnVHeads / max(1, geometry.gdnQKHeads), at: 19)
        let kvRow = geometry.kvHeads * geometry.headDim
        let q4GroupsPerHead = max(1, (geometry.headDim + 127) / 128)
        setU32(kvRow, at: 20)
        setU32(256 * kvRow, at: 21)
        setU32(kvRow / 2, at: 22)
        setU32(geometry.kvHeads * 2, at: 23)
        setU32(geometry.kvHeads * q4GroupsPerHead * 4, at: 24)
        // The attention output gate: a gated model packs Q‖gate in one q row.
        let standard = Self.standardConstants(geometry: geometry)
        var hasGate = standard.hasQGate
        fc.setConstantValue(&hasGate, type: .bool, index: 25)
        setU32(Int(standard.qStride), at: 26)
        var attnScale = standard.attnScale
        fc.setConstantValue(&attnScale, type: .float, index: 27)
        // MLX's gather kernel reads three function constants (200, 201, 202)
        // out of the vendored `quantized_nax.h`: each says that one extent is
        // a whole number of tiles. Only that kernel reads them.
        var alignM = false, alignN = true, alignK = true
        fc.setConstantValue(&alignM, type: .bool, index: 200)
        fc.setConstantValue(&alignN, type: .bool, index: 201)
        fc.setConstantValue(&alignK, type: .bool, index: 202)
        // Generalized architecture function constants (220..227):
        setU32(geometry.mhcConfig?.streams ?? 1, at: 220)
        setU32(geometry.mhcConfig?.iterations ?? 20, at: 221)
        var gdnVecDecay = geometry.gdnVectorDecay
        fc.setConstantValue(&gdnVecDecay, type: .bool, index: 222)
        var gdnGateMin = Float(geometry.gdnGateLowerBound)
        fc.setConstantValue(&gdnGateMin, type: .float, index: 223)
        var swigluLim = Float(geometry.swigluLimit ?? 0.0)
        fc.setConstantValue(&swigluLim, type: .float, index: 224)
        var routedScale = geometry.descriptor.routedScalingFactorLearnable ? Float(1.0) : Float(geometry.routedScalingFactor)
        fc.setConstantValue(&routedScale, type: .float, index: 225)
        setU32(geometry.kvLoraRank, at: 226)
        var indexKPool = geometry.indexKPoolCompress
        fc.setConstantValue(&indexKPool, type: .bool, index: 227)
        var situBeta: Float = 0.0
        var situLinBeta: Float = 0.0
        if case let .situ(beta, linearBeta) = geometry.descriptor.activationKind {
            situBeta = beta
            situLinBeta = linearBeta
        }
        fc.setConstantValue(&situBeta, type: .float, index: 228)
        fc.setConstantValue(&situLinBeta, type: .float, index: 229)
        var mhcAdditiveEps = geometry.mhcConfig?.hasExitHead == true
        fc.setConstantValue(&mhcAdditiveEps, type: .bool, index: 230)
        var mhcHcEps = geometry.mhcConfig?.hcEps ?? (geometry.mhcConfig?.streams == 4 ? Float(1e-6) : Float(0.0))
        fc.setConstantValue(&mhcHcEps, type: .float, index: 231)
        // The activation basis is NOT a function constant. It is a build-time
        // property of the metallib — see `QWWeightBasis.metallibFileName` —
        // because the rotation is a chain of `simd_shuffle_xor` calls and a
        // runtime branch in front of them puts the shuffle in a divergent
        // region, which cost 14.5% of prefill throughput when it was tried.
        return fc
    }

    /// The basis this HAL's kernels assume the activations are in.
    public let weightBasis: QWWeightBasis

    /// The marker kernels the two libraries carry. Only the names are used:
    /// the HAL asks the library which basis it is rather than trusting the
    /// path it was given (the standard name is spelled in two pieces).
    static let standardBasisMarker = "qwg_basis_marker" + "_standard"
    static let rotatedBasisMarker = "qwg_basis_marker"

    /// The basis a metallib was compiled for. A library with no marker at
    /// all predates the marker, and that is the incumbent's — the rotated
    /// basis.
    static func basis(of library: any MTLLibrary) -> QWWeightBasis {
        let names = library.functionNames
        if names.contains(standardBasisMarker) { return .standard }
        if names.contains(rotatedBasisMarker) { return .rotated }
        // Neither marker: a library built before the marker existed. That is
        // the incumbent's own, and the incumbent's basis is the rotated one.
        return .rotated
    }

    /// - Parameter weightBasis: the basis the model's weights are quantised
    ///   in. It picks the metallib and so decides whether the kernels rotate
    ///   the activations; it must be the basis the loader actually produced.
    deinit {
        keepAliveHeartbeat?.cancel()
    }

    public init(metallibPath: String? = nil, geometry: QWModelGeometry,
                weightBasis: QWWeightBasis = .rotated) throws {
        self.geometry = geometry
        self.weightBasis = weightBasis
        self.functionConstants = Self.makeFunctionConstants(geometry: geometry)
        guard let dev = MTLCreateSystemDefaultDevice() else { throw QWHalError.noDevice }
        device = dev
        // Cold-expert decode commits ~2 CBs per MoE layer; the default cap of 64 would block encode.
        guard let q = dev.makeCommandQueue(maxCommandBufferCount: 256) else { throw QWHalError.noDevice }
        queue = q
        doorStack.reserveCapacity(16)

        let mtlPath: String
        if let p = metallibPath {
            mtlPath = p
        } else {
            mtlPath = QWHal.defaultMetallibPath(basis: weightBasis)
        }
        guard FileManager.default.fileExists(atPath: mtlPath) else {
            throw QWHalError.metallibNotFound(mtlPath)
        }
        do {
            metallib = try dev.makeLibrary(URL: URL(fileURLWithPath: mtlPath))
        } catch {
            throw QWHalError.metallibLoadFailed(mtlPath)
        }
        // The library says which basis it is. If it is not the basis the
        // model's weights are in, the run would produce numbers that mean
        // nothing and no error anywhere — so it is refused here instead.
        let libBasis = Self.basis(of: metallib)
        guard libBasis == weightBasis else {
            throw QWHalError.metallibBasisMismatch(path: mtlPath,
                                                    expected: weightBasis,
                                                    found: libBasis)
        }

        caps = QWDeviceCaps.probe(dev)
    }

    /// Bundled metallib lookup: env override -> bundle -> build dir walk-up -> installed Qwegn.app.
    /// The basis picks the file: two builds of the same sources, one per basis.
    public static func defaultMetallibPath(basis: QWWeightBasis,
                                           appBundleURL: URL? = nil,
                                           searchDirs: [URL]? = nil) -> String {
        if let env = ProcessInfo.processInfo.environment["QW_METALLIB"] { return env }
        // A bundle that ships one library per basis names them after the
        // basis; the incumbent's generic "default" is the last resort, and a
        // bundle holding the wrong one is refused by the basis check below.
        let resource = basis.metallibFileName.replacingOccurrences(of: ".metallib", with: "")
        if searchDirs == nil, appBundleURL == nil {
            if let b = Bundle.main.url(forResource: resource, withExtension: "metallib") {
                return b.path
            }
            if let b = Bundle.main.url(forResource: "default", withExtension: "metallib") {
                return b.path
            }
        }
        let name = basis.metallibFileName
        let execDir = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath().deletingLastPathComponent()
        let startDirs = searchDirs ?? [execDir, URL(fileURLWithPath: FileManager.default.currentDirectoryPath)]
        for rel in ["Sources/QWMetal/\(name)",
                    ".build/release/\(name)",
                    ".build/debug/\(name)",
                    name] {
            for startDir in startDirs {
                var dir = startDir
                for _ in 0..<6 {
                    let cand = dir.appendingPathComponent(rel)
                    if FileManager.default.fileExists(atPath: cand.path) { return cand.path }
                    dir = dir.deletingLastPathComponent()
                }
            }
        }
        var bundleDirs: [URL] = []
        if let explicitApp = appBundleURL {
            bundleDirs.append(explicitApp.appendingPathComponent("Contents/Resources", isDirectory: true))
        } else if let envApp = ProcessInfo.processInfo.environment["QWEGN_APP_BUNDLE"], !envApp.isEmpty {
            bundleDirs.append(URL(fileURLWithPath: envApp.expandingTildePath).appendingPathComponent("Contents/Resources", isDirectory: true))
        }
        bundleDirs.append(contentsOf: [
            URL(fileURLWithPath: "/Applications/Qwegn.app/Contents/Resources", isDirectory: true),
            FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Applications/Qwegn.app/Contents/Resources", isDirectory: true),
            execDir.deletingLastPathComponent().appendingPathComponent("share/qwegn", isDirectory: true),
            URL(fileURLWithPath: "/usr/local/share/qwegn", isDirectory: true),
        ])
        for dir in bundleDirs {
            let cand = dir.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: cand.path) { return cand.path }
        }
        return name
    }

    /// Cached pipeline state. First use pays creation; post-freeze hits take no lock.
    public func pipeline(_ name: String) throws -> any MTLComputePipelineState {
        QWDispatchProbe.lookups += 1   // TEMP-PROBE
        if psoStore.frozen, let hit = psoStore.map[name] { return hit }
        psoLock.lock(); defer { psoLock.unlock() }
        psoStore.lockCount += 1
        if let p = psoStore.map[name] { return p }
        let fn: any MTLFunction
        do {
            fn = try metallib.makeFunction(name: name, constantValues: functionConstants)
        } catch {
            throw QWHalError.kernelMissing(name)
        }
        let psoT0 = CFAbsoluteTimeGetCurrent()
        let p = try device.makeComputePipelineState(function: fn)
        QWHalPSOStats.add(CFAbsoluteTimeGetCurrent() - psoT0)
        psoStore.map[name] = p
        return p
    }

    /// Begin a command buffer + encoder as a pair.
    public func begin() -> (any MTLCommandBuffer, any MTLComputeCommandEncoder)? {
        guard let cb = queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return nil }
        touchKeepAlive(enc)
        return (cb, enc)
    }

    /// Begin with explicit compute dispatch type.
    public func begin(_ mode: ComputeDispatchMode) -> (any MTLCommandBuffer, any MTLComputeCommandEncoder)? {
        guard let cb = queue.makeCommandBuffer() else { return nil }
        switch mode {
        case .serial:
            guard let enc = cb.makeComputeCommandEncoder() else { return nil }
            touchKeepAlive(enc)
            return (cb, enc)
        case .concurrent:
            guard let enc = cb.makeComputeCommandEncoder(dispatchType: .concurrent) else { return nil }
            touchKeepAlive(enc)
            return (cb, enc)
        }
    }

    public func end(_ cb: any MTLCommandBuffer, _ enc: any MTLComputeCommandEncoder) {
        enc.endEncoding()
        cb.commit()
    }

    /// Ends `enc` and opens a fresh compute command encoder on `cb` with matching `dispatchType`
    /// so GPU-written indirect dispatch counts are visible to the AGX command processor.
    public func rotateEncoder(_ cb: any MTLCommandBuffer,
                              _ enc: inout any MTLComputeCommandEncoder) throws {
        let mode: ComputeDispatchMode = (enc.dispatchType == .concurrent) ? .concurrent : .serial
        enc.endEncoding()
        let next: (any MTLComputeCommandEncoder)?
        switch mode {
        case .serial: next = cb.makeComputeCommandEncoder()
        case .concurrent: next = cb.makeComputeCommandEncoder(dispatchType: .concurrent)
        }
        guard let next else { throw QWHalError.noDevice }
        touchKeepAlive(next)
        enc = next
    }

    /// Like `begin(_:)`, but the command buffer first waits on the GPU for `event` to reach `value`,
    /// so host-side staging can finish while later command buffers are already committed.
    public func begin(_ mode: ComputeDispatchMode, waitingFor event: any MTLSharedEvent,
                      value: UInt64) -> (any MTLCommandBuffer, any MTLComputeCommandEncoder)? {
        guard let cb = queue.makeCommandBuffer() else { return nil }
        cb.encodeWaitForEvent(event, value: value)
        let enc: (any MTLComputeCommandEncoder)?
        switch mode {
        case .serial: enc = cb.makeComputeCommandEncoder()
        case .concurrent: enc = cb.makeComputeCommandEncoder(dispatchType: .concurrent)
        }
        guard let enc else { return nil }
        touchKeepAlive(enc)
        return (cb, enc)
    }

    // MARK: - Door batching (attaches door encoders to one in-flight command buffer)
    public private(set) var doorsOpen = 0
    /// C29 certification: total doorBegin calls (one per encode scope).
    public private(set) var doorOpens = 0
    private var doorCB: (any MTLCommandBuffer)?
    private var doorEncoder: (any MTLComputeCommandEncoder)?
    private var doorEncoderDispatched = false
    private struct DoorSubmission {
        let encoder: any MTLComputeCommandEncoder
        let commandBuffer: any MTLCommandBuffer
        let batched: Bool
        let t0: UInt64
        let region: String
        /// True when this door opened the batch's open encoder (a sampled top-level door) and must end it.
        let ownsEncoder: Bool
    }
    private let doorLock = NSLock()
    /// Preallocated LIFO submission stack — nested doors resolve without hashing.
    private var doorStack: [DoorSubmission] = []
    /// Diagnostics for the C02 host-edge gate (cheap, one increment per door).
    public private(set) var doorDrainCount = 0
    public private(set) var doorSubmissionCount = 0

    /// Called by the host-edge test before a step; not for production paths.
    public func resetDoorDiagnostics() {
        doorLock.lock()
        doorDrainCount = 0
        doorSubmissionCount = 0
        doorLock.unlock()
    }

    /// Pipeline-cache lock acquisitions since init (0 after freeze in a step).
    public var pipelineLockAcquisitions: Int { psoStore.lockCount }

    /// Open a batch scope (nesting-safe; QW_DOOR_BATCH=0|walk|step|all).
    public static let doorBatchScope = ProcessInfo.processInfo.environment["QW_DOOR_BATCH"]
        ?? ProcessInfo.processInfo.environment["QW_FN_DOOR_BATCH"]
        ?? "all"
    /// C34: when held, walk door batches stay OPEN across prefill chunks — one
    /// command buffer for the whole prefill, committed once at the head readback.
    /// Safe: nothing executes until commit, so host writes cannot race the GPU.
    public var doorBatchHold = false
    public static var doorBatchWalk: Bool { doorBatchScope == "all" || doorBatchScope == "walk" }
    public static var doorBatchStep: Bool { doorBatchScope == "all" || doorBatchScope == "step" }
    static let doorBatchEnabled = doorBatchScope != "0"

    public func beginDoors() {
        guard Self.doorBatchEnabled else { return }
        doorLock.lock()
        doorsOpen += 1
        doorLock.unlock()
    }

    private func endOpenDoorEncoderLocked() {
        if let enc = doorEncoder {
            enc.endEncoding()
            doorEncoder = nil
            doorEncoderDispatched = false
        }
    }

    /// Commit and wait the in-flight batch command buffer, if any.
    /// Must precede any host read of door outputs or host write to live door inputs.
    public func flushDoors(site: String = "") {
        doorLock.lock()
        endOpenDoorEncoderLocked()
        guard let cb = doorCB else {
            doorLock.unlock()
            return
        }
        doorDrainCount += 1
        let w0 = QWPhaseProbe.shared.isEnabled ? mach_absolute_time() : 0
        if QWPhaseProbe.shared.isEnabled { gpuLedgerAttach(cb, "flush:" + site) }
        QWPhaseProbe.shared.noteSubmission()
        QWHalQueueTimer.attach(cb, site: site)
        cb.commit()
        QWPhaseProbe.shared.noteWait()
        cb.waitUntilCompleted()
        resolveStageSamplesLocked(cb)
        QWPhaseProbe.shared.addFrom("cpu:flushWait", t0: w0)
        QWPhaseProbe.shared.addFrom("cpu:flushWait:" + (site.isEmpty ? "structural" : site), t0: w0)
        doorCB = nil
        doorLock.unlock()
    }

    /// A1: QW_GPU_STAGES=1 timestamps every door encoder (stage-boundary GPU
    /// counters, no host drains) so per-kernel GPU ms become attributable.
    public static let stageSamplingDefault = ProcessInfo.processInfo.environment["QW_GPU_STAGES"] == "1"
    public var stageSamplingEnabled = QWHal.stageSamplingDefault
    private var stageProfiler: QWGPUStageProfiler<String>?
    /// Region -> accumulated GPU microseconds since the last take.
    public private(set) var stageMicros: [String: Double] = [:]
    private var stageSamplingFailures = 0

    public func takeStageMicros() -> [String: Double] {
        doorLock.lock(); defer { doorLock.unlock() }
        let out = stageMicros
        stageMicros = [:]
        return out
    }

    /// Caller holds doorLock.
    private func makeStageProfilerLocked() {
        guard QWGPUStageProfiler<String>.isSupported(on: device) else {
            stageSamplingEnabled = false
            print("QWHAL: QW_GPU_STAGES requested but stage sampling is unsupported; disabled")
            return
        }
        do { stageProfiler = try QWGPUStageProfiler<String>(device: device, segmentCapacity: 16_384) }
        catch {
            stageSamplingEnabled = false
            print("QWHAL: stage profiler init failed (\(error)); sampling disabled")
        }
    }

    /// Caller holds doorLock; call only after the command buffer completed.
    private func resolveStageSamplesLocked(_ cb: any MTLCommandBuffer) {
        guard let p = stageProfiler, p.segmentCount > 0 else { return }
        defer { p.reset() }
        guard let pairs = try? p.resolveByStage() else {
            stageSamplingEnabled = false
            print("QWHAL: stage sample resolution failed; sampling disabled")
            return
        }
        for (region, micros) in pairs {
            stageMicros[region, default: 0] += micros
            QWPhaseProbe.shared.addGpu("gpu:stage:" + region, ms: micros / 1000)
        }
    }

    func gpuLedgerAttach(_ cb: any MTLCommandBuffer, _ tag: String) {
        cb.addCompletedHandler { buf in
            let a = buf.gpuStartTime, b = buf.gpuEndTime
            if b > a { QWPhaseProbe.shared.addGpu("gpu:" + tag, ms: (b - a) * 1000) }
        }
    }

    /// Close a batch scope (drains any pending door CB first).
    public func closeDoors() {
        doorLock.lock()
        endOpenDoorEncoderLocked()
        let cb = doorCB
        if cb != nil { doorDrainCount += 1 }
        if cb != nil { QWPhaseProbe.shared.noteSubmission() }
        if let cb, QWPhaseProbe.shared.isEnabled { gpuLedgerAttach(cb, "flush:doorClose") }
        if let cb { QWHalQueueTimer.attach(cb, site: "close") }
        cb?.commit()
        if cb != nil { QWPhaseProbe.shared.noteWait() }
        cb?.waitUntilCompleted()
        if let cb { resolveStageSamplesLocked(cb) }
        doorCB = nil
        if doorsOpen > 0 { doorsOpen -= 1 }
        doorLock.unlock()
    }

    /// Open a door encoder. Inside a batch (beginDoors) all doors share the one open encoder, so nested doors
    /// reuse it. Under stage sampling a top-level door opens its own sampled encoder and closes it at doorEnd.
    /// Outside a batch the door owns a standalone CB and commits at doorEnd.
    public func doorBegin(region: String = "") -> (any MTLComputeCommandEncoder)? {
        doorLock.lock()
        defer { doorLock.unlock() }
        doorOpens &+= 1
        let batched = doorsOpen > 0
        let cb: any MTLCommandBuffer
        let enc: any MTLComputeCommandEncoder
        var ownsEncoder = false
        if batched {
            guard let batchCB = doorCB ?? queue.makeCommandBuffer() else { return nil }
            doorCB = batchCB
            cb = batchCB
            if let open = doorEncoder {
                enc = open
            } else {
                if stageSamplingEnabled, stageProfiler == nil { makeStageProfilerLocked() }
                if stageSamplingEnabled, let p = stageProfiler, p.hasSegmentCapacity,
                   let sampled = try? p.makeEncoder(commandBuffer: batchCB, stage: region) {
                    enc = sampled
                    ownsEncoder = true
                } else {
                    guard let shared = batchCB.makeComputeCommandEncoder() else { return nil }
                    enc = shared
                }
                touchKeepAlive(enc)
                doorEncoder = enc
                doorEncoderDispatched = true
            }
        } else {
            guard let standaloneCB = queue.makeCommandBuffer() else { return nil }
            cb = standaloneCB
            if stageSamplingEnabled, stageProfiler == nil { makeStageProfilerLocked() }
            if stageSamplingEnabled, let p = stageProfiler, p.hasSegmentCapacity,
               let sampled = try? p.makeEncoder(commandBuffer: standaloneCB, stage: region) {
                enc = sampled
            } else {
                guard let plain = standaloneCB.makeComputeCommandEncoder() else { return nil }
                enc = plain
            }
            touchKeepAlive(enc)
        }
        let t0 = QWPhaseProbe.shared.isEnabled ? mach_absolute_time() : 0
        doorSubmissionCount += 1
        doorStack.append(DoorSubmission(encoder: enc, commandBuffer: cb, batched: batched, t0: t0,
                                        region: region, ownsEncoder: ownsEncoder))
        return enc
    }

    /// C29/C21: encode a contiguous kernel sequence under ONE door. Callers pass
    /// the returned encoder to their helpers via `into:`; the scope is closed by
    /// exactly one `doorEnd` + one `flushDoors` (law L1) at a single exit (L2).
    public func encoder(into outer: (any MTLComputeCommandEncoder)?,
                        region: String) throws -> any MTLComputeCommandEncoder {
        if let outer { return outer }
        guard let e = doorBegin(region: region) else {
            throw NSError(domain: "QWHAL", code: 1, userInfo: [NSLocalizedDescriptionKey: "QWHAL: could not open door in \(region)"])
        }
        return e
    }

    /// Close a door encoder. Batched: the door that owns the open encoder ends it (commit happens at flush).
    /// Standalone: end, commit and wait immediately.
    public func doorEnd(_ enc: any MTLComputeCommandEncoder) {
        doorLock.lock()
        guard let slot = doorStack.lastIndex(where: { ($0.encoder as AnyObject) === (enc as AnyObject) }) else {
            doorLock.unlock()
            preconditionFailure("QWHAL: doorEnd received an encoder not created by doorBegin")
        }
        let submission = doorStack.remove(at: slot)
        if submission.batched {
            if submission.ownsEncoder, let open = doorEncoder, (open as AnyObject) === (enc as AnyObject) {
                enc.endEncoding()
                doorEncoder = nil
                doorEncoderDispatched = false
            }
            QWPhaseProbe.shared.addFrom("cpu:doorEncode", t0: submission.t0)
        } else {
            doorDrainCount += 1
            enc.endEncoding()
            QWPhaseProbe.shared.addFrom("cpu:doorEncode", t0: submission.t0)
            let w0 = QWPhaseProbe.shared.isEnabled ? mach_absolute_time() : 0
            QWPhaseProbe.shared.noteSubmission()
            if QWPhaseProbe.shared.isEnabled { gpuLedgerAttach(submission.commandBuffer, "flush:solo") }
            submission.commandBuffer.commit()
            QWPhaseProbe.shared.noteWait()
            submission.commandBuffer.waitUntilCompleted()
            resolveStageSamplesLocked(submission.commandBuffer)
            QWPhaseProbe.shared.addFrom("cpu:doorWait", t0: w0)
        }
        doorLock.unlock()
    }
}

/// Diagnostic (QW_PREFILL_TIMER=1 prints it): total time spent creating compute pipelines (cold start).
public enum QWHalPSOStats {
    nonisolated(unsafe) public static var seconds: Double = 0
    nonisolated(unsafe) public static var count: Int = 0
    static func add(_ s: Double) { seconds += s; count += 1 }
}

/// Diagnostic (QW_QUEUE_TIMER=1): per command buffer, commit -> GPU start (queue latency), GPU busy,
/// GPU end -> host wake-up. Explains where flush-wait time goes.
public enum QWHalQueueTimer {
    public static let enabled: Bool = ProcessInfo.processInfo.environment["QW_QUEUE_TIMER"] == "1"
    static func attach(_ cb: any MTLCommandBuffer, site: String) {
        guard enabled else { return }
        let tCommit = CACurrentMediaTime()
        cb.addCompletedHandler { buf in
            let now = CACurrentMediaTime()
            print(String(format: "queue[%@]: commit->gpuStart %.1f ms, gpu busy %.1f ms, gpuEnd->handler %.1f ms, status %ld %@",
                         site, (buf.gpuStartTime - tCommit) * 1000, (buf.gpuEndTime - buf.gpuStartTime) * 1000,
                         (now - buf.gpuEndTime) * 1000, buf.status.rawValue, buf.error?.localizedDescription ?? "ok"))
        }
    }
}

// TEMP-PROBE: dispatch count per encoded step (remove after measurement).
public enum QWDispatchProbe { public nonisolated(unsafe) static var lookups = 0 }
