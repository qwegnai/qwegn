// QWHal.swift — Metal 4.0 hardware abstraction layer for Qwen3.8 engines.
// Owns MTLDevice, command queues, GPU family capability probes, and metallib
// pipeline caching (with post-init lock-free freeze).
// Nothing above QWHAL calls MTLCreateSystemDefaultDevice directly.

import Foundation
import IOKit
import Metal
import os
import QuartzCore
import QWCore
import Synchronization

public struct QWDeviceCaps: Sendable {
    public let name: String
    public let familyApple7: Bool
    public let familyApple8: Bool
    public let familyApple9: Bool
    public let familyApple10: Bool
    public let familyApple11: Bool
    public let unifiedMemory: Bool
    public let maxWorkingSetBytes: Int
    public let maxBufferLength: Int
    public let gpuCores: Int
    public let threadsPerTG: Int
    /// Measured DRAM read bandwidth, B/s (only populated on verified reference hardware).
    public let measuredReadBWBps: Double?
    /// Measured L2-resident read bandwidth, B/s (only populated on verified reference hardware).
    public let measuredL2BWBps: Double?
    public let isSimulated: Bool

    public var isMetal4TensorCapable: Bool { familyApple10 || familyApple11 }
    public var supportsMetal4Library: Bool { familyApple9 || familyApple10 || familyApple11 }
    public var supportsSimdgroupMMA: Bool {
        familyApple7 || familyApple8 || familyApple9 || familyApple10 || familyApple11
    }
    public var hasDynamicCaching: Bool { familyApple9 || familyApple10 || familyApple11 }

    public var gpuFamilyLabel: String {
        if familyApple11 { return "Apple11" }
        if familyApple10 { return "Apple10" }
        if familyApple9 { return "Apple9" }
        if familyApple8 { return "Apple8" }
        if familyApple7 { return "Apple7" }
        return "Standard"
    }

    public init(
        name: String, familyApple7: Bool, familyApple8: Bool, familyApple9: Bool,
        familyApple10: Bool, familyApple11: Bool, unifiedMemory: Bool,
        maxWorkingSetBytes: Int, maxBufferLength: Int, gpuCores: Int, threadsPerTG: Int,
        measuredReadBWBps: Double?, measuredL2BWBps: Double?, isSimulated: Bool
    ) {
        self.name = name
        self.familyApple7 = familyApple7
        self.familyApple8 = familyApple8
        self.familyApple9 = familyApple9
        self.familyApple10 = familyApple10
        self.familyApple11 = familyApple11
        self.unifiedMemory = unifiedMemory
        self.maxWorkingSetBytes = maxWorkingSetBytes
        self.maxBufferLength = maxBufferLength
        self.gpuCores = gpuCores
        self.threadsPerTG = threadsPerTG
        self.measuredReadBWBps = measuredReadBWBps
        self.measuredL2BWBps = measuredL2BWBps
        self.isSimulated = isSimulated
    }
}

extension QWDeviceCaps {
    /// The device's capabilities, read from the device itself.
    public static func probe(_ dev: MTLDevice) -> QWDeviceCaps {
        var apple7 = dev.supportsFamily(.apple7)
        var apple8 = dev.supportsFamily(.apple8)
        var apple9 = false, apple10 = false, apple11 = false
        if #available(macOS 14.0, *) { apple9 = dev.supportsFamily(.apple9) }
        if #available(macOS 15.0, *) { apple10 = dev.supportsFamily(.apple10) }
        if #available(macOS 26.0, *), let f11 = MTLGPUFamily(rawValue: 1011) {
            apple11 = dev.supportsFamily(f11)
        }
        var simulated = false
        let env = ProcessInfo.processInfo.environment
        switch env["QW_SIMULATE_GPU_FAMILY"]?.lowercased() {
        case "apple7", "m1":
            apple7 = true; apple8 = false; apple9 = false; apple10 = false; apple11 = false
            simulated = true
        case "apple8", "m2":
            apple7 = true; apple8 = true; apple9 = false; apple10 = false; apple11 = false
            simulated = true
        case "apple9", "m3", "m4":
            apple7 = true; apple8 = true; apple9 = true; apple10 = false; apple11 = false
            simulated = true
        case "apple10", "m5":
            apple7 = true; apple8 = true; apple9 = true
            apple10 = dev.supportsFamily(.apple10); apple11 = false
            simulated = true
        case "apple11", "m6":
            apple7 = true; apple8 = true; apple9 = true
            let hasTensor = dev.supportsFamily(.apple10)
            apple10 = hasTensor; apple11 = hasTensor
            simulated = true
        default:
            break
        }
        var cores = probeGPUCores()
        if let rawCores = env["QW_SIMULATE_GPU_CORES"], let parsed = Int(rawCores), parsed > 0 {
            cores = parsed
            simulated = true
        }
        let nativeWorkingSet = Int(dev.recommendedMaxWorkingSetSize)
        var maxWorkingSet = nativeWorkingSet
        if let rawWS = env["QW_SIMULATE_WORKING_SET_GB"], let wsGB = Double(rawWS), wsGB > 0 {
            let simBytes = Int(wsGB * 1024.0 * 1024.0 * 1024.0)
            maxWorkingSet = min(nativeWorkingSet, max(1, simBytes))
            simulated = true
        }
        let isRefM5Max = apple10 && !apple11 && !simulated && dev.name == "Apple M5 Max"
        return QWDeviceCaps(
            name: dev.name,
            familyApple7: apple7,
            familyApple8: apple8,
            familyApple9: apple9,
            familyApple10: apple10,
            familyApple11: apple11,
            unifiedMemory: dev.hasUnifiedMemory,
            maxWorkingSetBytes: maxWorkingSet,
            maxBufferLength: min(dev.maxBufferLength, Int.max),
            gpuCores: cores,
            threadsPerTG: 1024,
            measuredReadBWBps: isRefM5Max ? 585e9 : nil,
            measuredL2BWBps: isRefM5Max ? 950e9 : nil,
            isSimulated: simulated
        )
    }

    private static func probeGPUCores() -> Int {
        var iterator = io_iterator_t()
        let match = IOServiceMatching("IOAccelerator")
        if IOServiceGetMatchingServices(kIOMainPortDefault, match, &iterator) == KERN_SUCCESS {
            defer { IOObjectRelease(iterator) }
            var entry = IOIteratorNext(iterator)
            while entry != 0 {
                defer { IOObjectRelease(entry) }
                var props: Unmanaged<CFMutableDictionary>?
                if IORegistryEntryCreateCFProperties(entry, &props, kCFAllocatorDefault, 0) == KERN_SUCCESS,
                   let p = props?.takeRetainedValue() as? [String: Any],
                   let c = decodeGPUCoreCount(p["gpu-core-count"]), c > 0 {
                    return c
                }
                entry = IOIteratorNext(iterator)
            }
        }
        return 16
    }

    private static func decodeGPUCoreCount(_ rawValue: Any?) -> Int? {
        if let count = rawValue as? Int { return count > 0 ? count : nil }
        if let num = rawValue as? NSNumber {
            let count = num.intValue
            return count > 0 ? count : nil
        }
        guard let data = rawValue as? Data else { return nil }
        if data.count == MemoryLayout<UInt32>.size {
            let v = data.withUnsafeBytes { $0.load(as: UInt32.self) }
            let count = Int(UInt32(littleEndian: v))
            return count > 0 ? count : nil
        }
        if data.count == MemoryLayout<UInt64>.size {
            let v = data.withUnsafeBytes { $0.load(as: UInt64.self) }
            guard let count = Int(exactly: UInt64(littleEndian: v)), count > 0 else { return nil }
            return count
        }
        return nil
    }

    /// The capabilities of this machine, found once. They do not change
    /// while the process runs, and finding them means creating a Metal
    /// device: a request must not pay for that every time.
    public static let current: QWDeviceCaps? = {
        guard let dev = MTLCreateSystemDefaultDevice() else { return nil }
        return probe(dev)
    }()

    /// Current Metal device allocated bytes, exposed from HAL so higher layers need no Metal import.
    public static func systemDeviceAllocatedBytes() -> UInt64 {
        guard let dev = MTLCreateSystemDefaultDevice() else { return 0 }
        return UInt64(dev.currentAllocatedSize)
    }
}

public enum QWHalError: Error, CustomStringConvertible {
    case noDevice
    case metallibNotFound(String)
    case metallibLoadFailed(String)
    case metallibBasisMismatch(path: String, expected: QWWeightBasis, found: QWWeightBasis)
    case kernelMissing(String)
    case naxKernelDisallowed(String)

    public var description: String {
        switch self {
        case .noDevice: return "QWHAL: no Metal device"
        case .metallibNotFound(let p): return "QWHAL: metallib not found at \(p)"
        case .metallibLoadFailed(let p): return "QWHAL: metallib load failed: \(p)"
        case .metallibBasisMismatch(let p, let expected, let found):
            return "QWHAL: metallib at \(p) is in the \(found) basis, but the model needs \(expected)"
        case .kernelMissing(let n): return "QWHAL: kernel '\(n)' missing from metallib"
        case .naxKernelDisallowed(let n):
            return "QWHAL: kernel '\(n)' requires Apple10+ NAX tensor hardware (disallowed on pre-Apple10 or simulated non-NAX device)"
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
    /// Fixed 4.06 MiB (4,259,840 B) private scratch buffer for multi-TG GatedNorm staging (`hScratch` + `g1Scratch`).
    public let gatedNormScratch: any MTLBuffer
    /// Lock-free snapshot residency state (see QWResidency.swift).
    final class FrozenResidencySnapshot: @unchecked Sendable {
        let resources: [any MTLResource]
        init(_ resources: [any MTLResource]) { self.resources = resources }
    }

    struct MutableResidencyState: @unchecked Sendable {
        var resources: [any MTLResource] = []
        var ids = Set<ObjectIdentifier>()
        var retiredSnapshots: [FrozenResidencySnapshot] = []
        var heartbeat: (any DispatchSourceTimer)?
        var nopPSO: (any MTLComputePipelineState)?
    }

    let residencyState = OSAllocatedUnfairLock(initialState: MutableResidencyState())
    let frozenResidencyPtr = Atomic<UnsafeMutableRawPointer?>(nil)
    let residencyActiveReaders = Atomic<Int>(0)
    let keepAliveLastTouchNs = Atomic<UInt64>(0)
    let prefersRetainedCommandBuffers: Bool
    public let metallib: any MTLLibrary
    public let geometry: QWModelGeometry
    public let gdnNormSigmoidOverride: Bool?
    public let gdnGateScaleOverride: Float?
    private let functionConstants: MTLFunctionConstantValues
    private let aq4FunctionConstants: MTLFunctionConstantValues

    /// Immutable snapshot published atomically so post-freeze hits take no lock.
    final class FrozenPSOSnapshot: @unchecked Sendable {
        let map: [String: any MTLComputePipelineState]
        init(_ map: [String: any MTLComputePipelineState]) { self.map = map }
    }

    struct MutablePSOState: @unchecked Sendable {
        var map: [String: any MTLComputePipelineState] = [:]
        var frozen = false
        var lockCount = 0
        var retiredSnapshots: [FrozenPSOSnapshot] = []
    }

    let psoState = OSAllocatedUnfairLock(initialState: MutablePSOState())
    private let frozenSnapshotPtr = Atomic<UnsafeMutableRawPointer?>(nil)

    private func publishFrozenSnapshotLocked(_ state: inout MutablePSOState) {
        let ptr = Unmanaged.passRetained(FrozenPSOSnapshot(state.map)).toOpaque()
        if let prev = frozenSnapshotPtr.exchange(ptr, ordering: .releasing) {
            state.retiredSnapshots.append(
                Unmanaged<FrozenPSOSnapshot>.fromOpaque(prev).takeRetainedValue()
            )
        }
    }

    /// Freeze the pipeline cache so post-freeze hits take no lock. Idempotent.
    public func freezePipelines() {
        psoState.withLock { state in
            state.frozen = true
            publishFrozenSnapshotLocked(&state)
        }
    }

    /// The constants the standard path added, as values. The builder is
    /// write-only, so this is the readable seam over the same numbers.
    public static func standardConstants(geometry: QWModelGeometry)
        -> (hasQGate: Bool, qStride: UInt32, attnScale: Float) {
        (geometry.attnOutputGate,
         geometry.attnOutputGate ? 2 : 1,
         Float(1.0 / Double(geometry.headDim).squareRoot()))
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

    deinit {
        let heartbeat = residencyState.withLock { state -> (any DispatchSourceTimer)? in
            let timer = state.heartbeat
            state.heartbeat = nil
            state.retiredSnapshots.removeAll()
            return timer
        }
        heartbeat?.cancel()
        if let raw = frozenResidencyPtr.exchange(nil, ordering: .acquiring) {
            Unmanaged<FrozenResidencySnapshot>.fromOpaque(raw).release()
        }
        if let raw = frozenSnapshotPtr.exchange(nil, ordering: .acquiring) {
            Unmanaged<FrozenPSOSnapshot>.fromOpaque(raw).release()
        }
    }

    /// - Parameter weightBasis: the basis the model's weights are quantised
    ///   in. It picks the metallib and so decides whether the kernels rotate
    ///   the activations; it must be the basis the loader actually produced.
    public init(metallibPath: String? = nil, geometry: QWModelGeometry,
                weightBasis: QWWeightBasis = .rotated,
                gdnNormSigmoid: Bool? = nil,
                gdnGateScale: Float? = nil) throws {
        self.geometry = geometry
        self.weightBasis = weightBasis
        self.gdnNormSigmoidOverride = gdnNormSigmoid
        self.gdnGateScaleOverride = gdnGateScale
        self.prefersRetainedCommandBuffers =
            ProcessInfo.processInfo.environment["QWG_RETAINED_CB"] == "1"
        self.functionConstants = Self.makeFunctionConstants(
            geometry: geometry, isAQ4: false, weightBasis: weightBasis,
            gdnNormSigmoid: gdnNormSigmoid, gdnGateScale: gdnGateScale
        )
        self.aq4FunctionConstants = Self.makeFunctionConstants(
            geometry: geometry, isAQ4: true, weightBasis: weightBasis,
            gdnNormSigmoid: gdnNormSigmoid, gdnGateScale: gdnGateScale
        )
        guard let dev = MTLCreateSystemDefaultDevice() else { throw QWHalError.noDevice }
        device = dev
        let probedCaps = QWDeviceCaps.probe(dev)
        // Cold-expert decode commits ~2 CBs per MoE layer; the default cap of 64 would block encode.
        guard let q = dev.makeCommandQueue(maxCommandBufferCount: 256),
              let gnScratch = dev.makeBuffer(length: 4_259_840, options: .storageModePrivate) else {
            throw QWHalError.noDevice
        }
        gnScratch.label = "qwg-gated-norm-scratch"
        queue = q
        gatedNormScratch = gnScratch

        let mtlPath: String
        if let p = metallibPath {
            mtlPath = p
        } else {
            mtlPath = QWHal.defaultMetallibPath(basis: weightBasis, metal4: probedCaps.supportsMetal4Library)
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

        caps = probedCaps
    }

    /// Bundled metallib lookup: env override -> bundle -> build dir walk-up -> installed Qwegn.app.
    /// The basis picks the file: two builds of the same sources, one per basis.
    public static func defaultMetallibPath(basis: QWWeightBasis,
                                           metal4: Bool = true,
                                           appBundleURL: URL? = nil,
                                           searchDirs: [URL]? = nil) -> String {
        let env = ProcessInfo.processInfo.environment
        if !metal4 {
            if basis == .standard, let m3StdEnv = env["QW_METALLIB_M3_STD"] { return m3StdEnv }
            if let m3Env = env["QW_METALLIB_M3"] { return m3Env }
        }
        if basis == .standard, let stdEnv = env["QW_METALLIB_STD"] { return stdEnv }
        if let rotEnv = env["QW_METALLIB"] { return rotEnv }
        // A bundle that ships one library per basis names them after the
        // basis; the incumbent's generic "default" is the last resort, and a
        // bundle holding the wrong one is refused by the basis check below.
        let name = basis.metallibFileName(metal4: metal4)
        let resource = name.replacingOccurrences(of: ".metallib", with: "")
        if searchDirs == nil, appBundleURL == nil {
            if let b = Bundle.main.url(forResource: resource, withExtension: "metallib") {
                return b.path
            }
            if metal4, let b = Bundle.main.url(forResource: "default", withExtension: "metallib") {
                return b.path
            }
        }
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

    /// Cached pipeline state specialized for HQ4/AQ4 and optional per-layer SwiGLU clamp.
    /// Post-freeze hits take no lock.
    public func pipeline(
        _ name: String,
        isAQ4: Bool = false,
        swigluClamp: QWSwiGLUClampSpec? = nil
    ) throws -> any MTLComputePipelineState {
        if !caps.isMetal4TensorCapable && QWNaxKernelRegistry.names.contains(name) {
            throw QWHalError.naxKernelDisallowed(name)
        }
        let baseKey = isAQ4 ? "\(name)#aq4" : name
        let defaultLim = Float(geometry.swigluLimit ?? 0.0)
        let key: String
        let customClamp: QWSwiGLUClampSpec?
        if let clamp = swigluClamp, clamp.limit > 0 {
            if clamp.limit == defaultLim && !clamp.postSilu {
                key = baseKey
                customClamp = nil
            } else {
                key = "\(baseKey)#sw=\(clamp.limit.bitPattern)\(clamp.postSilu ? "p" : "r")"
                customClamp = clamp
            }
        } else if swigluClamp != nil, defaultLim > 0 {
            key = "\(baseKey)#sw=0"
            customClamp = QWSwiGLUClampSpec(limit: 0, postSilu: false)
        } else {
            key = baseKey
            customClamp = nil
        }
        if let raw = frozenSnapshotPtr.load(ordering: .acquiring),
           let hit = Unmanaged<FrozenPSOSnapshot>.fromOpaque(raw).takeUnretainedValue().map[key] {
            return hit
        }
        return try psoState.withLock { state in
            state.lockCount += 1
            if let p = state.map[key] {
                if state.frozen { publishFrozenSnapshotLocked(&state) }
                return p
            }
            let fn: any MTLFunction
            do {
                let constants: MTLFunctionConstantValues
                if let customClamp {
                    constants = Self.makeFunctionConstants(
                        geometry: geometry, isAQ4: isAQ4, swigluClamp: customClamp, weightBasis: weightBasis,
                        gdnNormSigmoid: gdnNormSigmoidOverride, gdnGateScale: gdnGateScaleOverride
                    )
                } else {
                    constants = isAQ4 ? aq4FunctionConstants : functionConstants
                }
                fn = try metallib.makeFunction(name: name, constantValues: constants)
            } catch {
                throw QWHalError.kernelMissing(name)
            }
            let psoT0 = CFAbsoluteTimeGetCurrent()
            let p = try device.makeComputePipelineState(function: fn)
            QWHalPSOStats.add(CFAbsoluteTimeGetCurrent() - psoT0)
            state.map[key] = p
            if state.frozen { publishFrozenSnapshotLocked(&state) }
            return p
        }
    }

    /// Cached pipeline state specialized for independent DFlash drafter and target-head rotation bases.
    public func pipeline(
        _ name: String,
        fcDrafterRotated: Bool,
        fcHeadRotated: Bool
    ) throws -> any MTLComputePipelineState {
        if !caps.isMetal4TensorCapable && QWNaxKernelRegistry.names.contains(name) {
            throw QWHalError.naxKernelDisallowed(name)
        }
        let defRot = weightBasis.rotatesActivations
        if fcDrafterRotated == defRot && fcHeadRotated == defRot {
            return try pipeline(name)
        }
        let key = "\(name)#dr=\(fcDrafterRotated ? 1 : 0)#hr=\(fcHeadRotated ? 1 : 0)"
        if let raw = frozenSnapshotPtr.load(ordering: .acquiring),
           let hit = Unmanaged<FrozenPSOSnapshot>.fromOpaque(raw).takeUnretainedValue().map[key] {
            return hit
        }
        return try psoState.withLock { state in
            state.lockCount += 1
            if let p = state.map[key] {
                if state.frozen { publishFrozenSnapshotLocked(&state) }
                return p
            }
            let constants = Self.makeFunctionConstants(
                geometry: geometry, isAQ4: false, weightBasis: weightBasis,
                fcDrafterRotated: fcDrafterRotated, fcHeadRotated: fcHeadRotated,
                gdnNormSigmoid: gdnNormSigmoidOverride, gdnGateScale: gdnGateScaleOverride
            )
            let fn: any MTLFunction
            do {
                fn = try metallib.makeFunction(name: name, constantValues: constants)
            } catch {
                throw QWHalError.kernelMissing(name)
            }
            let psoT0 = CFAbsoluteTimeGetCurrent()
            let p = try device.makeComputePipelineState(function: fn)
            QWHalPSOStats.add(CFAbsoluteTimeGetCurrent() - psoT0)
            state.map[key] = p
            if state.frozen { publishFrozenSnapshotLocked(&state) }
            return p
        }
    }

    /// Cached pipeline state specialized for GDN output-norm gating (`QW_FC_GDN_NORM_SIGMOID` + `QW_FC_GDN_GATE_SCALE`).
    public func pipeline(
        _ name: String,
        gdnNormSigmoid: Bool?,
        gdnGateScale: Float?
    ) throws -> any MTLComputePipelineState {
        let effSig = gdnNormSigmoid ?? gdnNormSigmoidOverride
        let effScale = gdnGateScale ?? gdnGateScaleOverride
        if effSig == gdnNormSigmoidOverride && effScale == gdnGateScaleOverride {
            return try pipeline(name)
        }
        let sigTag = effSig.map { $0 ? "1" : "0" } ?? "d"
        let scaleTag = effScale.map { String($0.bitPattern, radix: 16) } ?? "d"
        let key = "\(name)#gsig=\(sigTag)#gsc=\(scaleTag)"
        if let raw = frozenSnapshotPtr.load(ordering: .acquiring),
           let hit = Unmanaged<FrozenPSOSnapshot>.fromOpaque(raw).takeUnretainedValue().map[key] {
            return hit
        }
        return try psoState.withLock { state in
            state.lockCount += 1
            if let p = state.map[key] {
                if state.frozen { publishFrozenSnapshotLocked(&state) }
                return p
            }
            let constants = Self.makeFunctionConstants(
                geometry: geometry, isAQ4: false, weightBasis: weightBasis,
                gdnNormSigmoid: effSig, gdnGateScale: effScale
            )
            let fn: any MTLFunction
            do {
                fn = try metallib.makeFunction(name: name, constantValues: constants)
            } catch {
                throw QWHalError.kernelMissing(name)
            }
            let psoT0 = CFAbsoluteTimeGetCurrent()
            let p = try device.makeComputePipelineState(function: fn)
            QWHalPSOStats.add(CFAbsoluteTimeGetCurrent() - psoT0)
            state.map[key] = p
            if state.frozen { publishFrozenSnapshotLocked(&state) }
            return p
        }
    }

    /// Begin a command buffer + encoder as a pair.
    public func begin() -> (any MTLCommandBuffer, any MTLComputeCommandEncoder)? {
        begin(.serial)
    }

    /// Begin with explicit compute dispatch type.
    public func begin(_ mode: ComputeDispatchMode) -> (any MTLCommandBuffer, any MTLComputeCommandEncoder)? {
        guard let cb = queue.makeCommandBuffer() else { return nil }
        return openComputeEncoder(on: cb, mode: mode)
    }

    /// Allocates an unretained-reference command buffer (unless `QWG_RETAINED_CB=1`).
    /// Callers must guarantee every referenced Metal resource outlives GPU completion.
    public func makeUnretainedCommandBuffer() -> (any MTLCommandBuffer)? {
        prefersRetainedCommandBuffers
            ? queue.makeCommandBuffer()
            : queue.makeCommandBufferWithUnretainedReferences()
    }

    /// Begin a hot-path compute encoder on an unretained-reference command buffer.
    public func beginUnretained(
        _ mode: ComputeDispatchMode = .serial
    ) -> (any MTLCommandBuffer, any MTLComputeCommandEncoder)? {
        guard let cb = makeUnretainedCommandBuffer() else { return nil }
        return openComputeEncoder(on: cb, mode: mode)
    }

    /// Begin an unretained-reference compute encoder waiting on `event == value`.
    public func beginUnretained(
        _ mode: ComputeDispatchMode,
        waitingFor event: any MTLSharedEvent,
        value: UInt64
    ) -> (any MTLCommandBuffer, any MTLComputeCommandEncoder)? {
        guard let cb = makeUnretainedCommandBuffer() else { return nil }
        cb.encodeWaitForEvent(event, value: value)
        return openComputeEncoder(on: cb, mode: mode)
    }

    private func openComputeEncoder(
        on cb: any MTLCommandBuffer,
        mode: ComputeDispatchMode
    ) -> (any MTLCommandBuffer, any MTLComputeCommandEncoder)? {
        let enc: (any MTLComputeCommandEncoder)? = switch mode {
        case .serial: cb.makeComputeCommandEncoder()
        case .concurrent: cb.makeComputeCommandEncoder(dispatchType: .concurrent)
        }
        guard let enc else { return nil }
        touchKeepAlive(enc)
        return (cb, enc)
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
        guard let (_, next) = openComputeEncoder(on: cb, mode: mode) else { throw QWHalError.noDevice }
        enc = next
    }

    /// Like `begin(_:)`, but the command buffer first waits on the GPU for `event` to reach `value`,
    /// so host-side staging can finish while later command buffers are already committed.
    public func begin(_ mode: ComputeDispatchMode, waitingFor event: any MTLSharedEvent,
                      value: UInt64) -> (any MTLCommandBuffer, any MTLComputeCommandEncoder)? {
        guard let cb = queue.makeCommandBuffer() else { return nil }
        cb.encodeWaitForEvent(event, value: value)
        return openComputeEncoder(on: cb, mode: mode)
    }
}

/// Diagnostic (QW_PREFILL_TIMER=1 prints it): total time spent creating compute pipelines (cold start).
public enum QWHalPSOStats {
    private static let state = OSAllocatedUnfairLock<Double>(initialState: 0)
    public static var seconds: Double {
        get { state.withLock { $0 } }
        set { state.withLock { $0 = newValue } }
    }
    static func add(_ s: Double) { state.withLock { $0 += s } }
}
