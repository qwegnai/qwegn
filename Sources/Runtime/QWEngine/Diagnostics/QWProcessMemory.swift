// QWProcessMemory.swift — What this process is using, as the OS reports it.
//
// The measurement belongs to the process, not to one engine: a caller that
// wants the footprint asks the process, never a concrete driver.

import Foundation
import Darwin

/// The process's own memory footprint.
public enum QWProcessMemory {
    /// Process physical memory footprint in bytes, as reported by Darwin
    /// TASK_VM_INFO.
    public static func physicalFootprintBytes() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kerr = withUnsafeMutablePointer(to: &info) { infoPtr in
            infoPtr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), intPtr, &count)
            }
        }
        guard kerr == KERN_SUCCESS else { return 0 }
        return UInt64(info.phys_footprint)
    }
}
