// QWAdmission.swift — Dense working-set admission planner (Qwen3.8-27B, qwen3_5).
// Computes explicit working-set budgets from QWModelDescriptor fields
// (model resident bytes, KV, GDN state, and scratch) with deterministic safe failure.

import Foundation
#if canImport(Darwin)
import Darwin
#endif

/// HQ4_G32_M5 plane geometry shared by admission and the pack reader (QWQuant aliases it).
public enum QWHQ4Layout {
    public static let group = 32
    public static let blockBytes = 20
}

/// KV storage dtype widths (`f16` = 2 B, `q8` = 1 B, `q4` = 0.5 B).
public enum QWAdmissionDtype: Sendable {
    case f16, q8, q4
    public var bytesPerParam: Int { switch self { case .f16: 2; case .q8, .q4: 1 } }
    public func kvBytes(elements: Int) -> Int {
        switch self { case .f16: elements * 2; case .q8: elements; case .q4: elements / 2 }
    }
}

public enum QWAdmissionError: Error, CustomStringConvertible, Equatable {
    case bad(String)
    public var description: String {
        switch self { case .bad(let s): return "QWAdmission: \(s)" }
    }
}

/// Deterministic working-set admission plan for one or more sequences.
public struct QWAdmissionPlan: Sendable, Equatable {
    public struct Components: Sendable, Equatable, CustomStringConvertible {
        public let modelResidentBytes: Int
        public let kvBytes: Int
        public let gdnStateBytes: Int
        public let scratchBytes: Int
        public var total: Int { modelResidentBytes + kvBytes + gdnStateBytes + scratchBytes }
        public var description: String {
            "model \(modelResidentBytes) kv \(kvBytes) gdn \(gdnStateBytes) scratch \(scratchBytes) → TOTAL \(total)"
        }
    }

    public let descriptor: QWModelDescriptor
    public let components: Components
    public let workingSetCapBytes: Int
    public let contextTokens: Int
    public let kvDtype: QWAdmissionDtype
    public let scratchPerSeq: Int

    public var headroomBytes: Int { workingSetCapBytes - components.total }

    /// Binary admission for the current configuration: fit iff headroom >= 0.
    public func admit() -> QWAdmissionVerdict {
        headroomBytes >= 0 ? .admitted(headroomBytes: headroomBytes)
                           : .refused(shortByBytes: -headroomBytes)
    }

    /// Marginal admission: checks whether one additional sequence fits in headroom.
    public func admitAnotherSequence() -> QWAdmissionVerdict {
        let s = Self.perSequenceStateBytes(descriptor: descriptor,
                                           kvDtype: kvDtype,
                                           contextTokens: contextTokens,
                                           scratchPerSeq: scratchPerSeq)
        let per = s.kv + s.gdn + s.scratch
        return per <= headroomBytes ? .admitted(headroomBytes: headroomBytes - per)
                                    : .refused(shortByBytes: per - headroomBytes)
    }

    /// Build the admission plan from descriptor-derived component formulas.
    public init(descriptor: QWModelDescriptor,
                contextTokens: Int,
                kvDtype: QWAdmissionDtype = .q4,
                scratchPerSeq: Int,
                workingSetCapBytes: Int) throws {
        guard contextTokens > 0, scratchPerSeq >= 0, workingSetCapBytes > 0 else {
            throw QWAdmissionError.bad("admission: contextTokens/scratch/cap must be positive (scratch ≥ 0)")
        }
        let model = try Self.modelResidentBytes(descriptor: descriptor)
        let perSeq = Self.perSequenceStateBytes(descriptor: descriptor,
                                                kvDtype: kvDtype,
                                                contextTokens: contextTokens,
                                                scratchPerSeq: scratchPerSeq)
        self.descriptor = descriptor
        self.contextTokens = contextTokens
        self.kvDtype = kvDtype
        self.scratchPerSeq = scratchPerSeq
        self.workingSetCapBytes = workingSetCapBytes
        self.components = Components(
            modelResidentBytes: model,
            kvBytes: perSeq.kv, gdnStateBytes: perSeq.gdn, scratchBytes: perSeq.scratch)
    }

    /// Stable report dictionary for CLI and ledger comparison.
    public func report() -> [String: Any] {
        [
            "model": descriptor.family.rawValue,
            "components": [
                "model_resident_bytes": components.modelResidentBytes,
                "kv_bytes": components.kvBytes,
                "gdn_state_bytes": components.gdnStateBytes,
                "scratch_bytes": components.scratchBytes,
                "total_bytes": components.total,
            ],
            "working_set_cap_bytes": workingSetCapBytes,
            "headroom_bytes": headroomBytes,
            "admit": admit() == .admitted(headroomBytes: headroomBytes),
        ]
    }

    // MARK: - Component Formulas

    /// Resident weights of the model as stored in an HQ4_G32_M5 pack: 20 bytes per
    /// 32-weight group (0.625 B/param). Covers embeddings, attention/GDN projections, and dense or MoE MLPs.
    public static func modelResidentBytes(descriptor d: QWModelDescriptor) throws -> Int {
        let hq4 = { (elems: Int) in elems / QWHQ4Layout.group * QWHQ4Layout.blockBytes }
        var t = 0
        t += hq4(d.vocab * d.hidden) * 2
        let attnLayers = (0..<d.numLayers).filter { d.isFullAttention($0) }
        let gdnLayers = d.numLayers - attnLayers.count
        t += hq4(attnLayers.count * (d.qProjOut * d.hidden
                                     + 2 * d.kvHeads * d.headDim * d.hidden
                                     + d.qProjOut * d.hidden))
        t += hq4(gdnLayers * (d.gdnQKVChannels * d.hidden
                              + d.gdnConvKernel * d.gdnQKVChannels
                              + d.hidden * d.gdnVHeads * d.gdnHeadDim))
        let mlpPerLayerElems: Int
        if let numE = d.numExperts, let moeI = d.moeIntermediate {
            let denseLayers = min(d.numLayers, max(0, d.firstKDenseReplace))
            let moeLayers = d.numLayers - denseLayers
            let denseI = d.denseIntermediate ?? moeI
            let sharedI = d.sharedExpertIntermediate ?? 0
            t += hq4(denseLayers * 3 * (denseI * d.hidden))
            t += hq4(moeLayers * 3 * ((numE * moeI + sharedI) * d.hidden))
            mlpPerLayerElems = 3 * ((numE * moeI + sharedI) * d.hidden)
        } else if let mlpI = d.denseIntermediate {
            t += hq4(d.numLayers * 3 * (mlpI * d.hidden))
            mlpPerLayerElems = 3 * (mlpI * d.hidden)
        } else {
            throw QWAdmissionError.bad("admission: descriptor has neither dense nor MoE MLP intermediate size")
        }
        // MTP draft head: each MTP layer is one full-attention block + MLP, plus its fc (2h → h).
        let mtpBlock = d.qProjOut * d.hidden + 2 * d.kvHeads * d.headDim * d.hidden
            + d.qProjOut * d.hidden + mlpPerLayerElems + 2 * d.hidden * d.hidden
        t += hq4(d.mtpLayers * mtpBlock)
        return t
    }

    /// Per-sequence state bytes: KV cache, GDN recurrent + conv state, and scratch.
    public static func perSequenceStateBytes(descriptor d: QWModelDescriptor,
                                             kvDtype: QWAdmissionDtype,
                                             contextTokens: Int,
                                             scratchPerSeq: Int)
        -> (kv: Int, gdn: Int, scratch: Int) {
        let attnLayers = (0..<d.numLayers).filter { d.isFullAttention($0) }.count
        let kvElems = d.mlaUseNoPE
            ? attnLayers * contextTokens * ((d.kvLoraRank ?? 512) + (d.qkRopeHeadDim ?? 64))
            : 2 * attnLayers * contextTokens * d.kvHeads * d.headDim
        let kv = kvDtype.kvBytes(elements: kvElems)
        let gdnLayers = d.numLayers - attnLayers
        let rec = d.gdnVHeads * d.gdnHeadDim * d.gdnHeadDim
        let ring = (d.gdnConvKernel - 1) * d.gdnQKVChannels
        let gdn = gdnLayers * (rec + ring) * 4
        return (kv, gdn, scratchPerSeq)
    }

    /// Legacy byte-width overload retained for callers passing explicit `kvBytesPerParam`.
    public static func perSequenceStateBytes(descriptor d: QWModelDescriptor,
                                             kvBytesPerParam: Int,
                                             contextTokens: Int,
                                             scratchPerSeq: Int)
        -> (kv: Int, gdn: Int, scratch: Int) {
        let dtype: QWAdmissionDtype = kvBytesPerParam >= 2 ? .f16 : .q8
        var res = perSequenceStateBytes(descriptor: d, kvDtype: dtype, contextTokens: contextTokens, scratchPerSeq: scratchPerSeq)
        if kvBytesPerParam > 0 && kvBytesPerParam != dtype.bytesPerParam {
            let attnLayers = (0..<d.numLayers).filter { d.isFullAttention($0) }.count
            res.kv = 2 * attnLayers * contextTokens * d.kvHeads * d.headDim * kvBytesPerParam
        }
        return res
    }
}

public enum QWAdmissionVerdict: Equatable, Sendable {
    case admitted(headroomBytes: Int)
    case refused(shortByBytes: Int)
}

/// Physical RAM query via `sysctlbyname("hw.memsize")` and live reclaimable RAM via `host_statistics64`.
public enum QWAdmission {
    public static func physicalMemoryBytes() -> Int? {
        #if canImport(Darwin)
        var size: Int64 = 0
        var len = MemoryLayout<Int64>.size
        guard sysctlbyname("hw.memsize", &size, &len, nil, 0) == 0 else { return nil }
        return Int(size)
        #else
        return nil
        #endif
    }

    /// Returns currently free + file-cache reclaimable + purgeable physical memory in bytes.
    /// Excludes wired memory, VM compressor pages, and active anonymous allocations held by running processes.
    public static func reclaimableMemoryBytes() -> Int? {
        #if canImport(Darwin)
        var vmStat = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &vmStat) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return nil }
        let pageSize = UInt64(sysconf(_SC_PAGESIZE))
        let reclaimablePages = UInt64(vmStat.free_count)
            + UInt64(vmStat.external_page_count)
            + UInt64(vmStat.purgeable_count)
        return Int(reclaimablePages * pageSize)
        #else
        return nil
        #endif
    }
}
