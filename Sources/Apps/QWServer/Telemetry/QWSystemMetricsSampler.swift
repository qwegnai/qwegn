// QWSystemMetricsSampler.swift — The host's own readings.
// A reading is measured or unavailable. It is never estimated from the
// engine's speed, and never carried forward as if it were current.

import Foundation
import MachO
import Metal
import IOKit
import QWCore

/// Snapshot of system hardware utilization across CPU, GPU, and Memory.
public struct QWSystemHardwareUtilization: Sendable, Codable, Equatable {
    /// The host's CPU load. Nil means unavailable — never a zero that reads
    /// like an idle CPU.
    public var cpuUsagePercent: Double?
    /// Why the CPU reading is unavailable, when it is.
    public var cpuUsageUnavailableReason: String?
    /// The GPU's own device utilization, when the host can be read.
    /// Nil means unavailable — never a zero that reads like an idle GPU.
    public var gpuUsagePercent: Double?
    /// Why the GPU reading is unavailable, when it is.
    public var gpuUsageUnavailableReason: String?
    public var chipIdentifier: String           // e.g. "M5 Max 40GPU"
    /// The memory in use, when the host can be read. Nil means unavailable.
    public var memoryUsedBytes: UInt64?
    public var memoryTotalBytes: UInt64
    public var memoryUsedGB: Double?
    public var memoryTotalGB: Double
    public var memoryUsagePercent: Double?      // 0.0 - 100.0%
    /// Why the memory reading is unavailable, when it is.
    public var memoryUnavailableReason: String?
    /// The host's memory pressure level, when it can be read. Nil means
    /// unavailable — never "Normal", which reads like a healthy host.
    public var memoryPressureLevel: String?
    /// Why the pressure reading is unavailable, when it is.
    public var memoryPressureUnavailableReason: String?
    /// The measured memory pressure percentage. macOS reports a level, not a
    /// percentage, so this is nil unless a host reading supplied one.
    public var memoryPressurePercent: Double?
    public var metalAllocatedBytes: UInt64
    public var metalMaxWorkingSetBytes: UInt64
    /// When the reading was taken. A cached reading says how old it is.
    public var sampledAt: Date

    public init(
        cpuUsagePercent: Double? = nil,
        cpuUsageUnavailableReason: String? = nil,
        gpuUsagePercent: Double? = nil,
        gpuUsageUnavailableReason: String? = nil,
        chipIdentifier: String = "Apple Silicon",
        memoryUsedBytes: UInt64? = nil,
        memoryTotalBytes: UInt64 = 0,
        memoryUsedGB: Double? = nil,
        memoryTotalGB: Double = 0,
        memoryUsagePercent: Double? = nil,
        memoryUnavailableReason: String? = nil,
        memoryPressureLevel: String? = nil,
        memoryPressureUnavailableReason: String? = nil,
        memoryPressurePercent: Double? = nil,
        metalAllocatedBytes: UInt64 = 0,
        metalMaxWorkingSetBytes: UInt64 = 0,
        sampledAt: Date = Date()
    ) {
        self.cpuUsagePercent = cpuUsagePercent
        self.cpuUsageUnavailableReason = cpuUsageUnavailableReason
        self.gpuUsagePercent = gpuUsagePercent
        self.gpuUsageUnavailableReason = gpuUsageUnavailableReason
        self.chipIdentifier = chipIdentifier
        self.memoryUsedBytes = memoryUsedBytes
        self.memoryTotalBytes = memoryTotalBytes
        self.memoryUsedGB = memoryUsedGB
        self.memoryTotalGB = memoryTotalGB
        self.memoryUsagePercent = memoryUsagePercent
        self.memoryUnavailableReason = memoryUnavailableReason
        self.memoryPressureLevel = memoryPressureLevel
        self.memoryPressureUnavailableReason = memoryPressureUnavailableReason
        self.memoryPressurePercent = memoryPressurePercent
        self.metalAllocatedBytes = metalAllocatedBytes
        self.metalMaxWorkingSetBytes = metalMaxWorkingSetBytes
        self.sampledAt = sampledAt
    }

    /// Decodes a reading from an older payload: a field that is not there is
    /// unavailable, not zero.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            cpuUsagePercent: try c.decodeIfPresent(Double.self, forKey: .cpuUsagePercent),
            cpuUsageUnavailableReason: try c.decodeIfPresent(String.self, forKey: .cpuUsageUnavailableReason),
            gpuUsagePercent: try c.decodeIfPresent(Double.self, forKey: .gpuUsagePercent),
            gpuUsageUnavailableReason: try c.decodeIfPresent(String.self, forKey: .gpuUsageUnavailableReason),
            chipIdentifier: try c.decodeIfPresent(String.self, forKey: .chipIdentifier) ?? "Apple Silicon",
            memoryUsedBytes: try c.decodeIfPresent(UInt64.self, forKey: .memoryUsedBytes),
            memoryTotalBytes: try c.decodeIfPresent(UInt64.self, forKey: .memoryTotalBytes) ?? 0,
            memoryUsedGB: try c.decodeIfPresent(Double.self, forKey: .memoryUsedGB),
            memoryTotalGB: try c.decodeIfPresent(Double.self, forKey: .memoryTotalGB) ?? 0,
            memoryUsagePercent: try c.decodeIfPresent(Double.self, forKey: .memoryUsagePercent),
            memoryUnavailableReason: try c.decodeIfPresent(String.self, forKey: .memoryUnavailableReason),
            memoryPressureLevel: try c.decodeIfPresent(String.self, forKey: .memoryPressureLevel),
            memoryPressureUnavailableReason: try c.decodeIfPresent(String.self, forKey: .memoryPressureUnavailableReason),
            memoryPressurePercent: try c.decodeIfPresent(Double.self, forKey: .memoryPressurePercent),
            metalAllocatedBytes: try c.decodeIfPresent(UInt64.self, forKey: .metalAllocatedBytes) ?? 0,
            metalMaxWorkingSetBytes: try c.decodeIfPresent(UInt64.self, forKey: .metalMaxWorkingSetBytes) ?? 0,
            sampledAt: try c.decodeIfPresent(Date.self, forKey: .sampledAt) ?? Date())
    }
}

/// Samples the host's hardware. The readings are the host's own; the engine's
/// speed never becomes a hardware number.
public final class QWSystemMetricsSampler: @unchecked Sendable {
    public static let shared = QWSystemMetricsSampler()

    /// How long a cached reading is served before it is sampled again.
    /// Half a second, not two: a reading two seconds old is the
    /// machine's load a moment ago, not now.
    public static let refreshInterval: TimeInterval = 0.5

    private let lock = NSLock()
    private var lastCpuTicks: (user: UInt32, sys: UInt32, idle: UInt32, nice: UInt32)?
    private var metalDevice: MTLDevice?
    private var cached: QWSystemHardwareUtilization?
    public let chipIdentifier: String

    public init() {
        self.metalDevice = MTLCreateSystemDefaultDevice()
        self.chipIdentifier = Self.detectChipIdentifier(device: self.metalDevice)
        // Prime initial CPU sample
        _ = sampleCPULoad()
    }

    /// The last reading, sampled again when it is older than the interval.
    /// The reading carries its own timestamp, so a consumer can see its age.
    public func current() -> QWSystemHardwareUtilization {
        lock.lock()
        let cached = self.cached
        let fresh = cached.map { Date().timeIntervalSince($0.sampledAt) < Self.refreshInterval } ?? false
        lock.unlock()
        if let cached, fresh { return cached }
        return sample()
    }

    /// Samples the host now: CPU load, the GPU's own utilization when the
    /// host can be read, and unified memory.
    public func sample() -> QWSystemHardwareUtilization {
        lock.lock()
        defer { lock.unlock() }

        let cpu = sampleCPULoad()
        let mem = sampleMemoryAndPressure()
        let gpu = sampleGPU(memTotal: mem.totalBytes)
        let reading = QWSystemHardwareUtilization(
            cpuUsagePercent: cpu.percent,
            cpuUsageUnavailableReason: cpu.reason,
            gpuUsagePercent: gpu.usagePercent,
            gpuUsageUnavailableReason: gpu.unavailableReason,
            chipIdentifier: chipIdentifier,
            memoryUsedBytes: mem.usedBytes,
            memoryTotalBytes: mem.totalBytes,
            memoryUsedGB: mem.usedBytes.map { Double($0) / (1024.0 * 1024.0 * 1024.0) },
            memoryTotalGB: Double(mem.totalBytes) / (1024.0 * 1024.0 * 1024.0),
            memoryUsagePercent: mem.usagePercent,
            memoryUnavailableReason: mem.unavailableReason,
            memoryPressureLevel: mem.pressureLevel,
            memoryPressureUnavailableReason: mem.pressureUnavailableReason,
            memoryPressurePercent: mem.pressurePercent,
            metalAllocatedBytes: gpu.allocatedBytes,
            metalMaxWorkingSetBytes: gpu.maxWorkingSetBytes,
            sampledAt: Date())
        cached = reading
        return reading
    }

    // MARK: - Detect Chip Identifier (e.g. "M5 Max 40GPU")

    private static func detectChipIdentifier(device: MTLDevice?) -> String {
        let devName = device?.name ?? "Apple Silicon"
        let cleanName = devName.replacingOccurrences(of: "Apple ", with: "")

        var gpuCores = 0
        var iterator = io_iterator_t()
        let match = IOServiceMatching("IOAccelerator")
        if IOServiceGetMatchingServices(kIOMainPortDefault, match, &iterator) == KERN_SUCCESS {
            var entry = IOIteratorNext(iterator)
            while entry != 0 {
                var props: Unmanaged<CFMutableDictionary>?
                if IORegistryEntryCreateCFProperties(entry, &props, kCFAllocatorDefault, 0) == KERN_SUCCESS,
                   let p = props?.takeRetainedValue() as? [String: Any] {
                    if let c = p["gpu-core-count"] as? Int {
                        gpuCores = c
                    } else if let d = p["gpu-core-count"] as? Data {
                        gpuCores = d.withUnsafeBytes { $0.load(as: Int.self) }
                    }
                }
                IOObjectRelease(entry)
                entry = IOIteratorNext(iterator)
            }
            IOObjectRelease(iterator)
        }

        if gpuCores > 0 {
            return "\(cleanName) \(gpuCores)GPU"
        } else {
            return cleanName
        }
    }

    // MARK: - Private Sampling Helpers

    private func sampleCPULoad() -> (percent: Double?, reason: String?) {
        var loadInfo = host_cpu_load_info()
        var count = mach_msg_type_number_t(MemoryLayout<host_cpu_load_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &loadInfo) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO, $0, &count)
            }
        }

        guard kr == KERN_SUCCESS else {
            // The host would not tell us. That is a gap, not a number: the
            // previous reading is not this reading.
            return (nil, "the host did not report CPU load")
        }

        let user = loadInfo.cpu_ticks.0
        let sys = loadInfo.cpu_ticks.1
        let idle = loadInfo.cpu_ticks.2
        let nice = loadInfo.cpu_ticks.3

        guard let prev = lastCpuTicks else {
            // The first sample has no interval to compare against. A load
            // needs two samples; until then it is unknown, not zero.
            lastCpuTicks = (user, sys, idle, nice)
            return (nil, "the first CPU sample has no interval to compare against")
        }

        let deltaUser = Double(user >= prev.user ? user - prev.user : 0)
        let deltaSys = Double(sys >= prev.sys ? sys - prev.sys : 0)
        let deltaIdle = Double(idle >= prev.idle ? idle - prev.idle : 0)
        let deltaNice = Double(nice >= prev.nice ? nice - prev.nice : 0)
        let total = deltaUser + deltaSys + deltaIdle + deltaNice

        lastCpuTicks = (user, sys, idle, nice)
        guard total > 0 else {
            return (nil, "the host reported no CPU ticks to compare")
        }
        let active = deltaUser + deltaSys + deltaNice
        let percent = min(100.0, max(0.0, (active / total) * 100.0))
        return (percent, nil)
    }

    private func sampleMemoryAndPressure() -> (
        usedBytes: UInt64?,
        totalBytes: UInt64,
        usagePercent: Double?,
        unavailableReason: String?,
        pressureLevel: String?,
        pressureUnavailableReason: String?,
        pressurePercent: Double?
    ) {
        var vmStats = vm_statistics64()
        var vCount = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &vmStats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(vCount)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &vCount)
            }
        }

        let totalBytes = ProcessInfo.processInfo.physicalMemory
        var usedBytes: UInt64? = nil
        var unavailableReason: String? = nil
        if kr == KERN_SUCCESS {
            let pageSize = UInt64(getpagesize())
            let active = UInt64(vmStats.active_count)
            let wire = UInt64(vmStats.wire_count)
            let compressed = UInt64(vmStats.compressor_page_count)
            usedBytes = (active + wire + compressed) * pageSize
        } else {
            // The host would not tell us. That is a gap, not a number.
            unavailableReason = "the host did not report virtual memory statistics"
        }

        let usagePct = usedBytes.flatMap { used in
            totalBytes > 0 ? min(100.0, (Double(used) / Double(totalBytes)) * 100.0) : nil
        }

        var pressureVal: Int32 = 0
        var pSize = MemoryLayout<Int32>.size
        let pressureRead = sysctlbyname("kern.memorystatus_vm_pressure_level", &pressureVal, &pSize, nil, 0)

        // A failed read is a gap. "Normal" would be a healthy-looking guess at
        // a measurement the host refused to make.
        let pressureLevel: String?
        let pressureReason: String?
        if pressureRead == 0 {
            switch pressureVal {
            case 2: pressureLevel = "Warning"
            case 4: pressureLevel = "Critical"
            default: pressureLevel = "Normal"
            }
            pressureReason = nil
        } else {
            pressureLevel = nil
            pressureReason = "the host did not report its memory pressure level"
        }

        // macOS reports a pressure level, not a percentage. There is no
        // measured percentage to report, so there is none.
        return (usedBytes, totalBytes, usagePct, unavailableReason, pressureLevel, pressureReason, nil)
    }

    /// The GPU's own device utilization, straight from IOAccelerator.
    /// When the host does not publish it, the reading is unavailable: the
    /// engine's speed says nothing about how busy the GPU is.
    private func sampleGPU(memTotal: UInt64) -> (
        usagePercent: Double?,
        unavailableReason: String?,
        allocatedBytes: UInt64,
        maxWorkingSetBytes: UInt64
    ) {
        let dev = metalDevice ?? MTLCreateSystemDefaultDevice()
        let allocated = UInt64(dev?.currentAllocatedSize ?? 0)
        let maxWorkingSet = UInt64(dev?.recommendedMaxWorkingSetSize ?? memTotal)

        var ioKitGpuPercent: Double? = nil
        var iterator = io_iterator_t()
        let match = IOServiceMatching("IOAccelerator")
        if IOServiceGetMatchingServices(kIOMainPortDefault, match, &iterator) == KERN_SUCCESS {
            var entry = IOIteratorNext(iterator)
            while entry != 0 {
                var props: Unmanaged<CFMutableDictionary>?
                if IORegistryEntryCreateCFProperties(entry, &props, kCFAllocatorDefault, 0) == KERN_SUCCESS,
                   let p = props?.takeRetainedValue() as? [String: Any],
                   let perf = p["PerformanceStatistics"] as? [String: Any] {
                    if let devUtil = perf["Device Utilization %"] as? Int {
                        ioKitGpuPercent = Double(devUtil)
                        IOObjectRelease(entry)
                        break
                    }
                }
                IOObjectRelease(entry)
                entry = IOIteratorNext(iterator)
            }
            IOObjectRelease(iterator)
        }

        return (
            usagePercent: ioKitGpuPercent,
            unavailableReason: ioKitGpuPercent == nil
                ? "this host does not publish the GPU's device utilization" : nil,
            allocatedBytes: allocated,
            maxWorkingSetBytes: maxWorkingSet
        )
    }
}
