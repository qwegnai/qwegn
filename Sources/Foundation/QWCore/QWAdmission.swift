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

    /// Resident weights of the model: HQ4_G32_M5 for standard dense/MoE models, and
    /// config-derived mixed-precision (`HQ8_G32` trunk + `SHVQ` routed experts) for hybrid HCA / Latent-MoE models.
    public static func modelResidentBytes(descriptor d: QWModelDescriptor) throws -> Int {
        let hcaLayers = d.compressedAttentionLayers.count
        if hcaLayers > 0 || d.moeLatentSize != nil || d.hybridMLAConfig != nil {
            return try hybridModelResidentBytes(descriptor: d)
        }
        let hq4 = { (elems: Int) in elems / QWHQ4Layout.group * QWHQ4Layout.blockBytes }
        var t = 0
        t += hq4(d.vocab * d.hidden) * 2
        let attnLayers = (0..<d.numLayers).filter { d.isFullAttention($0) }
        let gdnLayers = d.linearAttentionLayers.count
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
        let mtpBlock = d.qProjOut * d.hidden + 2 * d.kvHeads * d.headDim * d.hidden
            + d.qProjOut * d.hidden + mlpPerLayerElems + 2 * d.hidden * d.hidden
        t += hq4(d.mtpLayers * mtpBlock)
        return t
    }

    private static func hybridModelResidentBytes(descriptor d: QWModelDescriptor) throws -> Int {
        let hq8 = { (elems: Int) in (elems / 32) * 34 }
        let shvqBlockBytes = { (tag: String?, fallback: Int) -> Int in
            switch tag?.lowercased() {
            case "shvq1_g64", "shvq1g64": return 10
            case "shvq15_g64", "shvq15g64": return 14
            case "shvq2_g64", "shvq2g64": return 18
            case "shvq3_g64", "shvq3g64": return 26
            case "shvq4_g64", "shvq4g64": return 34
            default: return fallback
            }
        }
        let guBlock = shvqBlockBytes(d.quantization?.expertGateUp, 14)
        let dnBlock = shvqBlockBytes(d.quantization?.expertDown, 26)

        var bytes = d.vocab * d.hidden * 2 + hq8(d.vocab * d.hidden)
        var trunkElems = 0
        var rawBytes = 0

        if let hca = d.hcaConfig {
            let hcaCount = d.compressedAttentionLayers.count
            let perHCA = d.hidden * hca.qLoraRank
                + hca.qLoraRank * hca.qHeads * hca.headDim
                + (hca.gatedAttnUseQLora ? hca.qLoraRank : d.hidden) * (hca.oGroups * hca.oLoraRank)
                + d.hidden * (hca.kvHeads * hca.headDim)
                + 2 * d.hidden * hca.headDim
                + hca.oGroups * hca.oLoraRank * ((hca.qHeads / max(1, hca.oGroups)) * hca.vHeadDim)
                + d.hidden * (hca.oGroups * hca.oLoraRank)
            trunkElems += hcaCount * perHCA
            rawBytes += hcaCount * (hca.slidingWindow * hca.headDim * 4 + hca.qHeads * hca.qHeads * 4)
        }

        let mlaCount = d.fullAttentionLayers.count
        if let mla = d.hybridMLAConfig, mlaCount > 0 {
            let idxH = d.indexerHeads ?? 0
            let idxD = d.indexerHeadDim ?? 0
            let perMLA = d.hidden * mla.qLoraRank
                + mla.qLoraRank * mla.qHeads * (mla.qkNopeHeadDim + mla.qkRopeHeadDim)
                + mla.qLoraRank * mla.qHeads * mla.vHeadDim
                + d.hidden * (mla.kvLoraRank + mla.qkRopeHeadDim)
                + mla.kvLoraRank * mla.qHeads * (mla.qkNopeHeadDim + mla.vHeadDim)
                + d.hidden * (mla.qHeads * mla.vHeadDim)
                + mla.qLoraRank * idxH * idxD + d.hidden * idxD + d.hidden * idxH
            trunkElems += mlaCount * perMLA
            rawBytes += mlaCount * (mla.qHeads * mla.qHeads * 4)
        }

        let gdnCount = d.linearAttentionLayers.count
        if gdnCount > 0 {
            trunkElems += gdnCount * (d.gdnQKVChannels * d.hidden
                + d.gdnConvKernel * d.gdnQKVChannels
                + d.hidden * d.gdnVHeads * d.gdnHeadDim)
        }

        if let numE = d.numExperts, let moeI = d.moeIntermediate {
            let denseCount = min(d.numLayers, max(0, d.firstKDenseReplace))
            let moeCount = d.numLayers - denseCount
            let denseI = d.denseIntermediate ?? moeI
            let sharedI = d.sharedExpertIntermediate ?? 0
            let routedDim = d.moeLatentSize ?? d.hidden
            trunkElems += denseCount * 3 * denseI * d.hidden
            if routedDim != d.hidden {
                trunkElems += moeCount * 2 * d.hidden * routedDim
            }
            trunkElems += moeCount * 3 * sharedI * d.hidden
            let gateUpElems = moeCount * numE * 2 * moeI * routedDim
            let downElems = moeCount * numE * moeI * routedDim
            bytes += (gateUpElems / 64) * guBlock + (downElems / 64) * dnBlock
            let routerSplits = d.moeSplitFeatureRouting ? 2 : 1
            rawBytes += moeCount * routerSplits * numE * d.hidden * 4
        } else if let mlpI = d.denseIntermediate {
            trunkElems += d.numLayers * 3 * mlpI * d.hidden
        } else {
            throw QWAdmissionError.bad("admission: descriptor has neither dense nor MoE MLP intermediate size")
        }

        if let mhc = d.mhcConfig {
            let mixRows = (2 + mhc.streams) * mhc.streams
            rawBytes += (mhc.end - mhc.start) * 2 * mixRows * (mhc.streams * d.hidden) * 4
        }
        if d.mtpLayers > 0 {
            trunkElems += d.mtpLayers * 2 * d.hidden * d.hidden
        }
        bytes += hq8(trunkElems) + rawBytes
        return bytes
    }

    /// Per-sequence state bytes: KV cache, GDN recurrent + conv state, and scratch.
    public static func perSequenceStateBytes(descriptor d: QWModelDescriptor,
                                             kvDtype: QWAdmissionDtype,
                                             contextTokens: Int,
                                             scratchPerSeq: Int)
        -> (kv: Int, gdn: Int, scratch: Int) {
        let attnLayers = (0..<d.numLayers).filter { d.isFullAttention($0) }.count
        let hcaLayers = d.compressedAttentionLayers.count
        let kv: Int
        if hcaLayers > 0 || d.hybridMLAConfig != nil {
            var totalKV = 0
            if let hca = d.hcaConfig, hcaLayers > 0 {
                let ratio = max(1, hca.defaultCompressRatio)
                let compTokens = (contextTokens + ratio - 1) / ratio
                let ringTokens = min(contextTokens, hca.slidingWindow)
                totalKV += hcaLayers * (compTokens + ringTokens) * hca.kvHeads * hca.headDim * 2
            }
            if let mla = d.hybridMLAConfig, attnLayers > 0 {
                let ckvBytes = kvDtype.kvBytes(elements: attnLayers * contextTokens * mla.kvLoraRank)
                let ropeBytes = attnLayers * contextTokens * mla.qkRopeHeadDim * 2
                let idxComp = max(1, d.indexerCompressRatio ?? 1)
                let idxDim = d.indexerHeadDim ?? 0
                let idxBytes = idxDim > 0 ? attnLayers * ((contextTokens + idxComp - 1) / idxComp) * idxDim * 2 : 0
                totalKV += ckvBytes + ropeBytes + idxBytes
            }
            kv = totalKV
        } else {
            let kvElems = d.mlaUseNoPE
                ? attnLayers * contextTokens * ((d.kvLoraRank ?? 512) + (d.qkRopeHeadDim ?? 64))
                : 2 * attnLayers * contextTokens * d.kvHeads * d.headDim
            kv = kvDtype.kvBytes(elements: kvElems)
        }
        let gdnLayers = d.linearAttentionLayers.count
        let rec = d.gdnVHeads * d.gdnHeadDim * d.gdnHeadDim
        let ring = max(0, d.gdnConvKernel - 1) * d.gdnQKVChannels
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
