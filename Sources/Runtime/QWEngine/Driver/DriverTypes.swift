// DriverTypes.swift — QWEngineDriver nested value types
// (split from QWEngineDriver.swift; same-module extension).
import Foundation
import Darwin
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWNorm
import QWRope
import QWKV
import QWAttn
import QWGDN
import QWMLP
import QWHead
import QWMTP
import QWQuant
import QWLoad
import QWState

extension QWEngineDriver {
    struct GdnW {
        let inProj: QWWeightView
        let qkv: QWWeightView, z: QWWeightView, a: QWWeightView, b: QWWeightView
        let conv: QWTensorValue, normW: QWTensorValue, aLog: QWTensorValue, dtBias: QWTensorValue
        let out: QWWeightView
        let fgbProj: QWWeightView?
        let fB: QWTensorValue?
        let gB: QWTensorValue?

        init(inProj: QWWeightView, qkv: QWWeightView, z: QWWeightView, a: QWWeightView, b: QWWeightView,
             conv: QWTensorValue, normW: QWTensorValue, aLog: QWTensorValue, dtBias: QWTensorValue,
             out: QWWeightView, fgbProj: QWWeightView? = nil, fB: QWTensorValue? = nil, gB: QWTensorValue? = nil) {
            self.inProj = inProj; self.qkv = qkv; self.z = z; self.a = a; self.b = b
            self.conv = conv; self.normW = normW; self.aLog = aLog; self.dtBias = dtBias
            self.out = out; self.fgbProj = fgbProj; self.fB = fB; self.gB = gB
        }
    }
    struct AttnW {
        let qkvProj: QWWeightView
        let q: QWWeightView, k: QWWeightView, v: QWWeightView, o: QWWeightView
        /// The per-head q/k norms, when the model has them. A standard family
        /// (llama, mistral, qwen2) has neither: its q and k are projected and
        /// roped as they are.
        let wq: QWTensorValue?, wkTiled: QWTensorValue?
        let mla: QWMLAAttnOperator.LayerWeights?

        init(qkvProj: QWWeightView, q: QWWeightView, k: QWWeightView, v: QWWeightView, o: QWWeightView,
             wq: QWTensorValue?, wkTiled: QWTensorValue?, mla: QWMLAAttnOperator.LayerWeights? = nil) {
            self.qkvProj = qkvProj; self.q = q; self.k = k; self.v = v; self.o = o
            self.wq = wq; self.wkTiled = wkTiled; self.mla = mla
        }
    }
    enum QWLayerPass: Equatable, Sendable { case fullAttention, gdn, moe, gdnMoe }
    /// On-demand F_NOCACHE SSD streamer for cold routed experts (e >= residentExperts).
    /// Streams cold expert weights into staging slots inside `gateUp`/`down` and zeroes them after use.
    final class ColdExpertStreamer: @unchecked Sendable {
        let io: PlaneIO
        let layerPrefix: String
        let residentExperts: Int
        let totalExperts: Int
        let stagingSlots: Int
        let intermediate: Int
        let hidden: Int
        let guRowBytes: Int
        let downRowBytes: Int
        let guBasePtr: UnsafeMutableRawPointer
        let downBasePtr: UnsafeMutableRawPointer
        let expertPermutation: [Int]?

        init(io: PlaneIO, layerPrefix: String, residentExperts: Int, totalExperts: Int,
             stagingSlots: Int, intermediate: Int, hidden: Int,
             guRowBytes: Int, downRowBytes: Int,
             guBasePtr: UnsafeMutableRawPointer, downBasePtr: UnsafeMutableRawPointer,
             expertPermutation: [Int]? = nil) {
            self.io = io
            self.layerPrefix = layerPrefix
            self.residentExperts = residentExperts
            self.totalExperts = totalExperts
            self.stagingSlots = stagingSlots
            self.intermediate = intermediate
            self.hidden = hidden
            self.guRowBytes = guRowBytes
            self.downRowBytes = downRowBytes
            self.guBasePtr = guBasePtr
            self.downBasePtr = downBasePtr
            self.expertPermutation = expertPermutation
        }

        // ── Dynamic residency (trunk layers only) ──────────────────────────
        // The K resident bank slots form a per-layer LRU cache over all
        // experts instead of a fixed calibration hot set. A miss is staged as
        // before (idx = K + stagingSlot) and then promoted into the LRU victim
        // slot by a GPU copy after the layer's cold compute, so the next token
        // that routes to it hits. Router ids stay physical (0..<totalExperts);
        // only the CPU rewrite maps them to bank slots.
        private(set) var isDynamic = false
        private var slotOfExpert: [Int32] = []
        private var expertInSlot: [Int32] = []
        private var slotStamp: [UInt64] = []
        private var clock: UInt64 = 0
        /// (staging slot, bank slot) copies to encode after the cold compute.
        var pendingPromotions: [(staging: Int, bank: Int)] = []
        static let verifyPromotions = ProcessInfo.processInfo.environment["QW_EXPERT_CACHE_VERIFY"] == "1"
        /// Optional MoE-Spec (arXiv:2602.16052) cap on unique cold experts staged per verify pass (`totalRows` in 2...4).
        static let draftColdExpertBudget: Int? = ProcessInfo.processInfo.environment["QW_MOE_SPEC_COLD_BUDGET"].flatMap(Int.init)
        var promotedForVerify: [(bank: Int, expert: Int)] = []
        private(set) var dynamicHits = 0
        private(set) var dynamicMisses = 0

        /// Host-written mirror of `slotOfExpert` (Int32 per physical expert) for GPU-side id mapping.
        private(set) var gpuSlotMap: (any MTLBuffer)?

        func attachGPUSlotMap(device: any MTLDevice) -> (any MTLBuffer)? {
            guard isDynamic else { return nil }
            if gpuSlotMap == nil {
                gpuSlotMap = device.makeBuffer(length: 4 * totalExperts, options: .storageModeShared)
                syncGPUSlotMap()
            }
            return gpuSlotMap
        }

        private func syncGPUSlotMap() {
            guard let map = gpuSlotMap else { return }
            slotOfExpert.withUnsafeBytes { map.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        }

        func enableDynamicResidency() {
            guard !isDynamic, stagingSlots > 0, residentExperts > 0 else { return }
            slotOfExpert = [Int32](repeating: -1, count: totalExperts)
            expertInSlot = [Int32](repeating: -1, count: residentExperts)
            slotStamp = [UInt64](repeating: 0, count: residentExperts)
            for s in 0..<residentExperts {
                slotOfExpert[s] = Int32(s)
                expertInSlot[s] = Int32(s)
                slotStamp[s] = UInt64(residentExperts - s)
            }
            clock = UInt64(residentExperts)
            isDynamic = true
        }

        var guExpertBytes: Int { 2 * intermediate * guRowBytes }

        /// Debug: byte-compare promoted bank slots against the pack (QW_EXPERT_CACHE_VERIFY=1).
        /// Call only when no GPU work is in flight.
        func verifyPromoted(guBank: UnsafeMutableRawPointer, downBank: UnsafeMutableRawPointer) throws -> Int {
            defer { promotedForVerify.removeAll(keepingCapacity: true) }
            var bad = 0
            let gb = guExpertBytes, db = downExpertBytes, single = intermediate * guRowBytes
            var tmp: UnsafeMutableRawPointer?
            posix_memalign(&tmp, 16_384, gb + db)
            guard let tmp else { return 0 }
            defer { free(tmp) }
            for (bank, p) in promotedForVerify {
                let e = (expertPermutation != nil && p < expertPermutation!.count) ? expertPermutation![p] : p
                try io.loadBatch([
                    (name: layerPrefix + "mlp.experts.\(e).gate_proj.weight", dst: tmp),
                    (name: layerPrefix + "mlp.experts.\(e).up_proj.weight", dst: tmp.advanced(by: single)),
                    (name: layerPrefix + "mlp.experts.\(e).down_proj.weight", dst: tmp.advanced(by: gb)),
                ], layout: .hq4)
                if memcmp(tmp, guBank.advanced(by: bank * gb), gb) != 0
                    || memcmp(tmp.advanced(by: gb), downBank.advanced(by: bank * db), db) != 0 {
                    bad += 1
                }
            }
            return bad
        }
        var downExpertBytes: Int { hidden * downRowBytes }

        private func prepareSpanDynamic(
            idxPtr: UnsafeMutablePointer<UInt32>,
            startRow: Int, totalRows: Int, routes: Int
        ) throws -> (
            nextRow: Int,
            stagedCount: Int,
            guBatch: [(name: String, dst: UnsafeMutableRawPointer)],
            downBatch: [(name: String, dst: UnsafeMutableRawPointer)]
        ) {
            guard startRow < totalRows else { return (totalRows, 0, [], []) }
            clock &+= 1
            let now = clock
            let resU32 = UInt32(residentExperts)
            let isVerifyWindow = totalRows >= 2 && totalRows <= 4
            let effectiveColdSlots = (isVerifyWindow && Self.draftColdExpertBudget != nil)
                ? max(1, min(stagingSlots, Self.draftColdExpertBudget!))
                : stagingSlots
            var stagedFor: [Int: UInt32] = [:]
            var stagedItems: [(expert: Int, slot: Int)] = []
            var r = startRow
            while r < totalRows {
                let rowBase = r * routes
                var need = 0
                for k in 0..<routes {
                    let raw = idxPtr[rowBase + k]
                    if Int32(bitPattern: raw) < 0 { continue }
                    let p = Int(raw)
                    guard p < totalExperts, slotOfExpert[p] < 0, stagedFor[p] == nil else { continue }
                    var seen = false
                    for prev in 0..<k where Int(idxPtr[rowBase + prev]) == p { seen = true; break }
                    if !seen { need += 1 }
                }
                if r > startRow && stagedItems.count + need > effectiveColdSlots && !(isVerifyWindow && Self.draftColdExpertBudget != nil) { break }
                for k in 0..<routes {
                    let i = rowBase + k
                    let raw = idxPtr[i]
                    if Int32(bitPattern: raw) < 0 { continue }
                    let p = Int(raw)
                    guard p < totalExperts else { continue }
                    let s = Int(slotOfExpert[p])
                    if s >= 0 {
                        idxPtr[i] = UInt32(s)
                        slotStamp[s] = now
                        dynamicHits += 1
                        continue
                    }
                    if let x = stagedFor[p] { idxPtr[i] = x; continue }
                    dynamicMisses += 1
                    let slotCap = (r == 0) ? stagingSlots : effectiveColdSlots
                    if stagedItems.count < slotCap {
                        let slot = stagedItems.count
                        let x = resU32 + UInt32(slot)
                        stagedFor[p] = x
                        stagedItems.append((expert: p, slot: slot))
                        idxPtr[i] = x
                    } else {
                        idxPtr[i] = 0xFFFF_FFFF
                    }
                }
                r += 1
            }
            let (gu, dn) = try promoteAndBatch(stagedItems, now: now)
            return (r, stagedItems.count, gu, dn)
        }

        /// Whether a span of `routes` ids can always be mapped by `qwg_moe_span_ids`: every
        /// non-resident expert fits a staging slot, so the span never splits.
        func canMapWholeSpan() -> Bool {
            isDynamic && totalExperts <= 512 && stagingSlots >= totalExperts - residentExperts
                && Self.draftColdExpertBudget == nil
        }

        /// Dynamic-residency bookkeeping for a span mapped on the GPU by `qwg_moe_span_ids` (`mapped[i]` = bank
        /// slot, `residentExperts` + staging slot, or -1) from the physical router `ids`: stamps hits, stages
        /// each miss into its GPU-chosen slot, then promotes and batches exactly like `prepareSpan`.
        func prepareSpanMapped(
            ids: UnsafePointer<UInt32>, mapped: UnsafePointer<Int32>, count: Int
        ) throws -> (
            stagedCount: Int,
            guBatch: [(name: String, dst: UnsafeMutableRawPointer)],
            downBatch: [(name: String, dst: UnsafeMutableRawPointer)]
        ) {
            precondition(canMapWholeSpan(), "span map needs a dynamic streamer whose staging holds every miss")
            clock &+= 1
            let now = clock
            let res = Int32(residentExperts)
            var expertOfSlot = [Int](repeating: -1, count: stagingSlots)
            var firstSeen = [Int](repeating: 0, count: stagingSlots)
            var staged = 0
            for i in 0..<count {
                let m = mapped[i]
                if m < 0 { continue }
                if m < res {
                    guard Int(ids[i]) < totalExperts, slotOfExpert[Int(ids[i])] == m else {
                        throw QWEngineDriverError.state("span map slot \(m) disagrees with host residency")
                    }
                    slotStamp[Int(m)] = now
                    dynamicHits += 1
                    continue
                }
                let slot = Int(m - res)
                guard slot < stagingSlots else {
                    throw QWEngineDriverError.state("span map staging slot \(slot) out of \(stagingSlots)")
                }
                if expertOfSlot[slot] < 0 {
                    expertOfSlot[slot] = Int(ids[i])
                    firstSeen[slot] = i
                    staged += 1
                    dynamicMisses += 1
                } else if expertOfSlot[slot] != Int(ids[i]) {
                    throw QWEngineDriverError.state("span map gives staging slot \(slot) two experts")
                }
            }
            var stagedItems: [(expert: Int, slot: Int)] = []
            stagedItems.reserveCapacity(staged)
            for slot in 0..<staged {
                guard expertOfSlot[slot] >= 0 else {
                    throw QWEngineDriverError.state("span map staging slots are not dense at \(slot)")
                }
                stagedItems.append((expert: expertOfSlot[slot], slot: slot))
            }
            // Promote in route first-appearance order, as `prepareSpanDynamic` does, so the
            // post-span residency (and thus decode) is identical to the host-mapped path.
            stagedItems.sort { firstSeen[$0.slot] < firstSeen[$1.slot] }
            let (gu, dn) = try promoteAndBatch(stagedItems, now: now)
            return (staged, gu, dn)
        }

        /// Promotes every staged miss into the least-recently-used bank slot the current span does not
        /// read (stamp < now), mirrors the residency to the GPU, and builds the SSD load batches.
        /// Initial stamps are reverse-ordered by calibration rank so cold tail slots evict first.
        private func promoteAndBatch(_ stagedItems: [(expert: Int, slot: Int)], now: UInt64) throws -> (
            [(name: String, dst: UnsafeMutableRawPointer)],
            [(name: String, dst: UnsafeMutableRawPointer)]
        ) {
            for item in stagedItems {
                var victim = -1
                var best = UInt64.max
                for s in 0..<residentExperts where slotStamp[s] < now && slotStamp[s] < best {
                    best = slotStamp[s]
                    victim = s
                }
                guard victim >= 0 else { break }
                let old = Int(expertInSlot[victim])
                if old >= 0 { slotOfExpert[old] = -1 }
                expertInSlot[victim] = Int32(item.expert)
                slotOfExpert[item.expert] = Int32(victim)
                slotStamp[victim] = now
                pendingPromotions.append((staging: item.slot, bank: victim))
                if Self.verifyPromotions { promotedForVerify.append((bank: victim, expert: item.expert)) }
            }
            if !stagedItems.isEmpty { syncGPUSlotMap() }
            return try makeBatches(stagedItems)
        }

        private func makeBatches(_ stagedItems: [(expert: Int, slot: Int)]) throws -> (
            [(name: String, dst: UnsafeMutableRawPointer)],
            [(name: String, dst: UnsafeMutableRawPointer)]
        ) {
            var guBatch: [(name: String, dst: UnsafeMutableRawPointer)] = []
            var downBatch: [(name: String, dst: UnsafeMutableRawPointer)] = []
            guard !stagedItems.isEmpty else { return (guBatch, downBatch) }
            let guExpertBytes = self.guExpertBytes
            let singleGuBytes = intermediate * guRowBytes
            let downExpertBytes = self.downExpertBytes
            let allowMissingWeights = io.pack.tensors.count < totalExperts
            guBatch.reserveCapacity(stagedItems.count * 2)
            downBatch.reserveCapacity(stagedItems.count)
            for item in stagedItems {
                let logicalE = item.expert
                let e = (expertPermutation != nil && logicalE < expertPermutation!.count)
                    ? expertPermutation![logicalE]
                    : logicalE
                let coldSlot = item.slot
                let guDst = guBasePtr.advanced(by: coldSlot * guExpertBytes)
                let upDst = guDst.advanced(by: singleGuBytes)
                let downDst = downBasePtr.advanced(by: coldSlot * downExpertBytes)
                let gName = layerPrefix + "mlp.experts.\(e).gate_proj.weight"
                let uName = layerPrefix + "mlp.experts.\(e).up_proj.weight"
                let dName = layerPrefix + "mlp.experts.\(e).down_proj.weight"
                if (try? io.pack.entry(gName)) != nil {
                    guBatch.append((name: gName, dst: guDst))
                    guBatch.append((name: uName, dst: upDst))
                    downBatch.append((name: dName, dst: downDst))
                } else if allowMissingWeights {
                    memset(guDst, 0, guExpertBytes)
                    memset(downDst, 0, downExpertBytes)
                } else {
                    throw QWEngineDriverError.weightMissing(gName, "cold expert missing from QWP pack")
                }
            }
            return (guBatch, downBatch)
        }

        /// Finds the longest prefix `startRow ..< nextRow` (at least 1 row) whose distinct
        /// cold experts (`e >= residentExperts`) fit in `stagingSlots`, rewrites `idxPtr` in place,
        /// and prepares the `PlaneIO` gate/up and down load batches without executing the synchronous SSD read yet.
        func prepareSpan(
            idxPtr: UnsafeMutablePointer<UInt32>,
            startRow: Int, totalRows: Int, routes: Int
        ) throws -> (
            nextRow: Int,
            stagedCount: Int,
            guBatch: [(name: String, dst: UnsafeMutableRawPointer)],
            downBatch: [(name: String, dst: UnsafeMutableRawPointer)]
        ) {
            guard stagingSlots > 0, startRow < totalRows else {
                return (totalRows, 0, [], [])
            }
            if isDynamic {
                return try prepareSpanDynamic(idxPtr: idxPtr, startRow: startRow,
                                              totalRows: totalRows, routes: routes)
            }
            let resU32 = UInt32(residentExperts)
            let isVerifyWindow = totalRows >= 2 && totalRows <= 4
            let effectiveColdSlots = (isVerifyWindow && Self.draftColdExpertBudget != nil)
                ? max(1, min(stagingSlots, Self.draftColdExpertBudget!))
                : stagingSlots
            var firstColdRow = totalRows
            outerScan: for row in startRow..<totalRows {
                let base = row * routes
                for k in 0..<routes {
                    let raw = idxPtr[base + k]
                    if Int32(bitPattern: raw) >= 0 && raw >= resU32 {
                        firstColdRow = row
                        break outerScan
                    }
                }
            }
            if firstColdRow == totalRows {
                return (totalRows, 0, [], [])
            }
            var slotForExpert: [Int: Int] = [:]
            var stagedCount = 0
            let guExpertBytes = 2 * intermediate * guRowBytes
            let singleGuBytes = intermediate * guRowBytes
            let downExpertBytes = hidden * downRowBytes
            let allowMissingWeights = io.pack.tensors.count < totalExperts

            var r = firstColdRow
            var stagedItems: [(expert: Int, slot: Int)] = []
            while r < totalRows {
                let rowBase = r * routes
                var neededForRow = 0
                for k in 0..<routes {
                    let raw = idxPtr[rowBase + k]
                    if Int32(bitPattern: raw) < 0 { continue }
                    let e = Int(raw)
                    guard e >= residentExperts, slotForExpert[e] == nil else { continue }
                    var seenEarlierInRow = false
                    for prev in 0..<k where Int(idxPtr[rowBase + prev]) == e {
                        seenEarlierInRow = true
                        break
                    }
                    if !seenEarlierInRow {
                        neededForRow += 1
                    }
                }
                if r > startRow && stagedCount + neededForRow > effectiveColdSlots && !(isVerifyWindow && Self.draftColdExpertBudget != nil) {
                    break
                }
                for k in 0..<routes {
                    let i = rowBase + k
                    let raw = idxPtr[i]
                    if Int32(bitPattern: raw) < 0 { continue }
                    let e = Int(raw)
                    guard e >= residentExperts else { continue }
                    if let existingSlot = slotForExpert[e] {
                        idxPtr[i] = UInt32(existingSlot)
                        continue
                    }
                    let slotCap = (r == 0) ? stagingSlots : effectiveColdSlots
                    guard stagedCount < slotCap else {
                        idxPtr[i] = 0xFFFF_FFFF
                        continue
                    }
                    let coldSlot = stagedCount
                    let rewrittenIdx = residentExperts + coldSlot
                    stagedCount += 1
                    slotForExpert[e] = rewrittenIdx
                    stagedItems.append((expert: e, slot: coldSlot))
                    idxPtr[i] = UInt32(rewrittenIdx)
                }
                r += 1
            }
            var guBatch: [(name: String, dst: UnsafeMutableRawPointer)] = []
            var downBatch: [(name: String, dst: UnsafeMutableRawPointer)] = []
            if !stagedItems.isEmpty {
                guBatch.reserveCapacity(stagedItems.count * 2)
                downBatch.reserveCapacity(stagedItems.count)
                for item in stagedItems {
                    let logicalE = item.expert
                    let e = (expertPermutation != nil && logicalE < expertPermutation!.count)
                        ? expertPermutation![logicalE]
                        : logicalE
                    let coldSlot = item.slot
                    let guDst = guBasePtr.advanced(by: coldSlot * guExpertBytes)
                    let upDst = guDst.advanced(by: singleGuBytes)
                    let downDst = downBasePtr.advanced(by: coldSlot * downExpertBytes)
                    let gName = layerPrefix + "mlp.experts.\(e).gate_proj.weight"
                    let uName = layerPrefix + "mlp.experts.\(e).up_proj.weight"
                    let dName = layerPrefix + "mlp.experts.\(e).down_proj.weight"
                    if (try? io.pack.entry(gName)) != nil {
                        guBatch.append((name: gName, dst: guDst))
                        guBatch.append((name: uName, dst: upDst))
                        downBatch.append((name: dName, dst: downDst))
                    } else if allowMissingWeights {
                        memset(guDst, 0, guExpertBytes)
                        memset(downDst, 0, downExpertBytes)
                    } else {
                        throw QWEngineDriverError.weightMissing(gName, "cold expert missing from QWP pack")
                    }
                }
            }
            return (r, stagedCount, guBatch, downBatch)
        }

        func executeBatch(_ batch: [(name: String, dst: UnsafeMutableRawPointer)]) throws {
            guard !batch.isEmpty else { return }
            try io.loadBatch(batch, layout: .hq4)
        }

        func executeTwoPhaseBatch(
            guBatch: [(name: String, dst: UnsafeMutableRawPointer)],
            downBatch: [(name: String, dst: UnsafeMutableRawPointer)],
            onGateUpLoaded: (@Sendable () throws -> Void)? = nil
        ) throws {
            guard !guBatch.isEmpty || !downBatch.isEmpty else { return }
            try io.loadBatchTwoPhase(
                firstPhase: guBatch,
                secondPhase: downBatch,
                layout: .hq4,
                onFirstPhaseComplete: onGateUpLoaded
            )
        }

        func stageSpan(idxPtr: UnsafeMutablePointer<UInt32>,
                       startRow: Int, totalRows: Int, routes: Int) throws -> (nextRow: Int, stagedCount: Int) {
            let prep = try prepareSpan(idxPtr: idxPtr, startRow: startRow, totalRows: totalRows, routes: routes)
            try executeTwoPhaseBatch(guBatch: prep.guBatch, downBatch: prep.downBatch)
            return (prep.nextRow, prep.stagedCount)
        }

        /// Immediately zeroes/unloads all `stagedCount` cold expert staging slots after the MoE down-projection.
        func unload(stagedCount: Int) {
            guard stagedCount > 0, !isDynamic else { return }
            let guExpertBytes = 2 * intermediate * guRowBytes
            let downExpertBytes = hidden * downRowBytes
            memset(guBasePtr, 0, stagedCount * guExpertBytes)
            memset(downBasePtr, 0, stagedCount * downExpertBytes)
        }
    }

    struct MoEW {
        let router: QWTensorValue
        let routerBias: QWTensorValue?
        let gateUp: QWWeightView
        let down: QWWeightView
        let coldGateUp: QWWeightView?
        let coldDown: QWWeightView?
        let numExperts: Int
        let expertsPerTok: Int
        let sharedGateUp: QWWeightView?
        let sharedDown: QWWeightView?
        let sharedGateScalar: QWTensorValue?
        let coldStreamer: ColdExpertStreamer?

        init(router: QWTensorValue, gateUp: QWWeightView, down: QWWeightView,
             numExperts: Int, expertsPerTok: Int,
             sharedGateUp: QWWeightView? = nil, sharedDown: QWWeightView? = nil,
             sharedGateScalar: QWTensorValue? = nil,
             routerBias: QWTensorValue? = nil,
             coldGateUp: QWWeightView? = nil, coldDown: QWWeightView? = nil,
             coldStreamer: ColdExpertStreamer? = nil) {
            self.router = router
            self.routerBias = routerBias
            self.gateUp = gateUp
            self.down = down
            self.coldGateUp = coldGateUp
            self.coldDown = coldDown
            self.numExperts = numExperts
            self.expertsPerTok = expertsPerTok
            self.sharedGateUp = sharedGateUp
            self.sharedDown = sharedDown
            self.sharedGateScalar = sharedGateScalar
            self.coldStreamer = coldStreamer
        }
    }
    struct MHCLayerWeights {
        let attnW: QWTensorValue
        let attnB: QWTensorValue
        let ffnW: QWTensorValue
        let ffnB: QWTensorValue
    }
    struct LayerW {
        let ln1: QWTensorValue, ln2: QWTensorValue
        let gu: QWMLPWeight, down: QWWeightView
        let kind: QWLayerPass
        let gdn: GdnW?, attn: AttnW?
        /// The sparse MoE weights — nil for a dense layer. A MoE layer's
        /// `gu`/`down` hold the FIRST expert's views, so they stay valid
        /// views; the banks live here.
        let moe: MoEW?
        let mhc: MHCLayerWeights?

        init(ln1: QWTensorValue, ln2: QWTensorValue, gu: QWMLPWeight, down: QWWeightView,
             kind: QWLayerPass, gdn: GdnW?, attn: AttnW?, moe: MoEW?, mhc: MHCLayerWeights? = nil) {
            self.ln1 = ln1
            self.ln2 = ln2
            self.gu = gu
            self.down = down
            self.kind = kind
            self.gdn = gdn
            self.attn = attn
            self.moe = moe
            self.mhc = mhc
        }
    }
    /// The sparse-MoE chain's scratch. One set of shapes serves a decode step
    /// (one row) and a prefill chunk (`rows` rows), so the chain is one body.
    struct MoEScratch {
        let xRot32: QWTensorValue      // [rows · H] f32 — the rotated hidden
        let xPlain32: QWTensorValue    // [rows · H] f32 — the un-rotated hidden
        let logits: QWTensorValue      // [rows · E] f32 — the router's scores
        let idx: QWTensorValue         // [rows · K] i32 — the chosen experts
        let wts: QWTensorValue         // [rows · K] f32 — the router's weights
        let gu32: QWTensorValue        // [rows · K · 2I] f32 — the gate/up
        let act32: QWTensorValue       // [rows · K · I] f32 — the SwiGLU
        let guSh32: QWTensorValue?     // [rows · 2 · ISh] f32 — shared expert gate/up
        let actSh32: QWTensorValue?    // [rows · ISh] f32 — shared expert SwiGLU
        let shared: QWTensorValue      // [rows · H] f32 — shared expert down (or zeroed)
        let gate: QWTensorValue        // [rows] f32 — shared expert sigmoid gate (or zeroed)
        let expOff: QWTensorValue?     // [E + 1] i32 — expert offset table
        let sortedPair: QWTensorValue? // [rows · K] i32 — sorted pair index
        let sortedTok: QWTensorValue?  // [rows · K] i32 — sorted token index
        let invPerm: QWTensorValue?    // [rows · K] i32 — inverse permutation
        let tileE: QWTensorValue?      // [rows · K] i32 — tile expert index
        let tileR: QWTensorValue?      // [rows · K] i32 — tile row index
        let tileCount: QWTensorValue?  // [1] i32 — active tile count
        let gu16: QWTensorValue?       // [rows · K · 2I] f16 — NAX gate/up out
        let dnHi: QWTensorValue?       // [rows · K · I] f16 — intermediate SwiGLU
        let dnAcc: QWTensorValue       // [rows · K · H] f32 — route-ordered down contributions
        let naxX16: QWTensorValue?     // [rows · H] f16 — rotated hidden for NAX
    }
    struct Scratch {
        let x: QWTensorValue                             // residual stream [H] f16 (in-place)
        let ln1: QWTensorValue                           // input_layernorm out [H] f16
        let ln2: QWTensorValue                           // post_attn_layernorm out [H] f16
        let gs: QWTensorValue                            // group sums K=5120 [40] f32
        let gs48: QWTensorValue                          // group sums K=6144 [48] f32
        let gsI: QWTensorValue                           // group sums K=17408 [136] f32
        let gdnY: QWTensorValue                          // [16480] f32 (qkv ‖ z ‖ a ‖ b)
        let gdnY16: QWTensorValue                        // [16480] f16 (qkv ‖ z ‖ a ‖ b)
        let qkv32: QWTensorValue, qkv16: QWTensorValue
        let z32: QWTensorValue, z16: QWTensorValue
        let a32: QWTensorValue, b32: QWTensorValue
        let betas: QWTensorValue, glog: QWTensorValue
        let convY: QWTensorValue
        let oScr: QWTensorValue, oOut: QWTensorValue
        let attnY: QWTensorValue                         // [14336] f32 (q ‖ k ‖ v)
        let attnY16: QWTensorValue                       // [14336] f16 (q ‖ k ‖ v)
        let q32: QWTensorValue, k32: QWTensorValue, v32: QWTensorValue
        let q16: QWTensorValue, k16: QWTensorValue, v16: QWTensorValue
        let qR: QWTensorValue, kR: QWTensorValue
        let attOut: QWTensorValue
        let o32: QWTensorValue, o16: QWTensorValue       // attn/attn-out [H]
        let gu32: QWTensorValue, gu16: QWTensorValue
        let dn32: QWTensorValue, dn16: QWTensorValue
        let logits: QWTensorValue
        let keysHi: QWTensorValue, keysLo: QWTensorValue, token: QWTensorValue
        let partials: QWMRegion
        /// The sparse-MoE chain's scratch — nil when the model has no MoE layer.
        let moe: MoEScratch?
        let mhcStream: QWTensorValue?
        let mhcControls: QWTensorValue?
        let fgb16: QWTensorValue?

        init(x: QWTensorValue, ln1: QWTensorValue, ln2: QWTensorValue,
             gs: QWTensorValue, gs48: QWTensorValue, gsI: QWTensorValue,
             gdnY: QWTensorValue, gdnY16: QWTensorValue,
             qkv32: QWTensorValue, qkv16: QWTensorValue,
             z32: QWTensorValue, z16: QWTensorValue,
             a32: QWTensorValue, b32: QWTensorValue,
             betas: QWTensorValue, glog: QWTensorValue,
             convY: QWTensorValue, oScr: QWTensorValue, oOut: QWTensorValue,
             attnY: QWTensorValue, attnY16: QWTensorValue,
             q32: QWTensorValue, k32: QWTensorValue, v32: QWTensorValue,
             q16: QWTensorValue, k16: QWTensorValue, v16: QWTensorValue,
             qR: QWTensorValue, kR: QWTensorValue,
             attOut: QWTensorValue,
             o32: QWTensorValue, o16: QWTensorValue,
             gu32: QWTensorValue, gu16: QWTensorValue,
             dn32: QWTensorValue, dn16: QWTensorValue,
             logits: QWTensorValue,
             keysHi: QWTensorValue, keysLo: QWTensorValue, token: QWTensorValue,
             partials: QWMRegion, moe: MoEScratch?,
             mhcStream: QWTensorValue? = nil, mhcControls: QWTensorValue? = nil,
             fgb16: QWTensorValue? = nil) {
            self.x = x; self.ln1 = ln1; self.ln2 = ln2
            self.gs = gs; self.gs48 = gs48; self.gsI = gsI
            self.gdnY = gdnY; self.gdnY16 = gdnY16
            self.qkv32 = qkv32; self.qkv16 = qkv16
            self.z32 = z32; self.z16 = z16
            self.a32 = a32; self.b32 = b32
            self.betas = betas; self.glog = glog
            self.convY = convY; self.oScr = oScr; self.oOut = oOut
            self.attnY = attnY; self.attnY16 = attnY16
            self.q32 = q32; self.k32 = k32; self.v32 = v32
            self.q16 = q16; self.k16 = k16; self.v16 = v16
            self.qR = qR; self.kR = kR
            self.attOut = attOut
            self.o32 = o32; self.o16 = o16
            self.gu32 = gu32; self.gu16 = gu16
            self.dn32 = dn32; self.dn16 = dn16
            self.logits = logits
            self.keysHi = keysHi; self.keysLo = keysLo; self.token = token
            self.partials = partials; self.moe = moe
            self.mhcStream = mhcStream; self.mhcControls = mhcControls
            self.fgb16 = fgb16
        }
    }
    struct PrefillScratch {
        let maxChunk: Int
        let x: QWTensorValue                             // [maxChunk · H] f16
        let ln1: QWTensorValue                           // [maxChunk · H] f16
        let ln2: QWTensorValue                           // [maxChunk · H] f16
        let gs: QWTensorValue                            // group sums K=5120: maxChunk · 256 B f32
        let gs48: QWTensorValue                          // group sums K=6144: maxChunk · 256 B f32
        let gsI: QWTensorValue                           // group sums K=17408: maxChunk · 768 B f32
        let gdnY: QWTensorValue                          // [64 · 16480] f32
        let gdnY16: QWTensorValue                        // [64 · 16480] f16
        let attnY: QWTensorValue                         // [64 · 14336] f32
        let attnY16: QWTensorValue                       // [64 · 14336] f16
        let qkv32: QWTensorValue, qkv16: QWTensorValue   // [64 · 10240] f32 / f16
        let z32: QWTensorValue, z16: QWTensorValue       // [64 · 6144] f32 / f16
        let a32: QWTensorValue, b32: QWTensorValue       // [64 · 48] f32 (exact contiguous stride 48 f32)
        let betas: QWTensorValue, glog: QWTensorValue    // [64 · 48] f32 (exact contiguous stride 48 f32)
        let convY: QWTensorValue                         // [64 · 10240] f16
        let oScr: QWTensorValue                          // [64 · 6144] f32
        let oOut: QWTensorValue                          // [64 · 6144] f16
        let attOut: QWTensorValue                        // [64 · 6144] f16
        let attV16: QWTensorValue                        // [(maxChunk + 32) · 1024] f16
        let attQR16: QWTensorValue                       // [(maxChunk + 32) · 12288] f16
        let attKR16: QWTensorValue                       // [(maxChunk + 32) · 1024] f16
        let U: QWTensorValue, W: QWTensorValue           // [64 · 6144] f32
        let A: QWTensorValue, B: QWTensorValue           // [64 · 6144] f32
        let o32: QWTensorValue, o16: QWTensorValue       // [64 · H] f32 / f16
        let guRaw16: QWTensorValue                       // [64 · 34816] f16
        let gu32: QWTensorValue, gu16: QWTensorValue     // [64 · I] f32 / f16
        let dn32: QWTensorValue, dn16: QWTensorValue     // [64 · H] f32 / f16
        /// The sparse-MoE chain's scratch — nil when the model has no MoE layer.
        let moe: MoEScratch?
        let mhcStream: QWTensorValue?
        let mhcControls: QWTensorValue?
        let fgb16: QWTensorValue?

        init(maxChunk: Int, x: QWTensorValue, ln1: QWTensorValue, ln2: QWTensorValue,
             gs: QWTensorValue, gs48: QWTensorValue, gsI: QWTensorValue,
             gdnY: QWTensorValue, gdnY16: QWTensorValue,
             attnY: QWTensorValue, attnY16: QWTensorValue,
             qkv32: QWTensorValue, qkv16: QWTensorValue,
             z32: QWTensorValue, z16: QWTensorValue,
             a32: QWTensorValue, b32: QWTensorValue,
             betas: QWTensorValue, glog: QWTensorValue,
             convY: QWTensorValue, oScr: QWTensorValue, oOut: QWTensorValue,
             attOut: QWTensorValue, attV16: QWTensorValue, attQR16: QWTensorValue, attKR16: QWTensorValue,
             U: QWTensorValue, W: QWTensorValue, A: QWTensorValue, B: QWTensorValue,
             o32: QWTensorValue, o16: QWTensorValue,
             guRaw16: QWTensorValue, gu32: QWTensorValue, gu16: QWTensorValue,
             dn32: QWTensorValue, dn16: QWTensorValue,
             moe: MoEScratch?,
             mhcStream: QWTensorValue? = nil, mhcControls: QWTensorValue? = nil,
             fgb16: QWTensorValue? = nil) {
            self.maxChunk = maxChunk; self.x = x; self.ln1 = ln1; self.ln2 = ln2
            self.gs = gs; self.gs48 = gs48; self.gsI = gsI
            self.gdnY = gdnY; self.gdnY16 = gdnY16
            self.attnY = attnY; self.attnY16 = attnY16
            self.qkv32 = qkv32; self.qkv16 = qkv16
            self.z32 = z32; self.z16 = z16
            self.a32 = a32; self.b32 = b32
            self.betas = betas; self.glog = glog
            self.convY = convY; self.oScr = oScr; self.oOut = oOut
            self.attOut = attOut; self.attV16 = attV16; self.attQR16 = attQR16; self.attKR16 = attKR16
            self.U = U; self.W = W; self.A = A; self.B = B
            self.o32 = o32; self.o16 = o16
            self.guRaw16 = guRaw16; self.gu32 = gu32; self.gu16 = gu16
            self.dn32 = dn32; self.dn16 = dn16
            self.moe = moe
            self.mhcStream = mhcStream; self.mhcControls = mhcControls
            self.fgb16 = fgb16
        }
    }
    struct GdnSlot {
        let recSlab: QWTensorValue
        let ringSlab: QWTensorValue
        let rec: [QWTensorValue]      // per-GDN-layer views into recSlab
        let ring: [QWTensorValue]     // per-GDN-layer views into ringSlab
        let trunkHidden: QWTensorValue?
        var dflashSlot: QWDFlashSessionSlot? = nil
    }
}

/// Per-step decode telemetry: host time inside cold-expert staging (wait + route readback + IO)
/// and the QW_DECODE_PROFILE=1 stage-boundary GPU profiler state.
struct QWDecodeTelemetry {
    var coldStageUs: Double = 0
    var coldWaitUs: Double = 0
    var coldIOUs: Double = 0
    var coldStaged = 0
    var profiler: QWGPUStageProfiler<String>?
    var stageTotals: [String: (us: Double, n: Int)] = [:]
    var profiledSteps = 0
    /// QW_PREFILL_PROFILE=1: cold-staging counters and wall clock at the chunk start.
    var prefillChunkStart: (stageUs: Double, waitUs: Double, ioUs: Double, staged: Int, t0: Date)?
}
