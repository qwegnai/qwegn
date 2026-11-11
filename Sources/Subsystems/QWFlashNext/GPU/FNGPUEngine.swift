// FNGPUEngine.swift — High-level GPU inference engine for Qwen3.8-Flash-Next (48 layers, MoE + GDN + QSA + PLE).
// Integrates FNGPUDecodeStack, FNGPUPrefillStack, and FNGPUHead for CLI (`qwg run`/`repl`) and HTTP server (`qwg serve`).

import Foundation
import QWAlloc
import QWCore
import QWGEMM
import QWHAL
import QWHead
import QWQuant

public final class FNGPUEngine: @unchecked Sendable {
    struct LayerRecord {
        let weights: FNGPULayerWeights
        let attnState: FNGPUAttnDecodeState?
        let ple: FNNgramEmbedding?
        let gdn: FNGPUGDNMixer?
        let attn: FNGPUAttnMixer?
        let lowrank: Int
    }

    public let packPath: String
    public let descriptor: QWModelDescriptor
    public let hal: QWHal
    public let arena: QWArena
    public let ctx: FNGPUContext
    public let capacity: Int
    /// Largest chunk `generate` accepts; sizes the prefill arena and the MTP prefill buffers.
    public let maxPrefillRows: Int
    /// Maximum number of concurrent session slots tracked by the prefix cache warm store (`1...16`).
    public let maxConcurrentSessions: Int
    public var prefixCacheDirectory: URL?
    /// The warm tier: the prefixes the RAM holds. Absent when the
    /// settings gave the engine no room for one.
    var prefixCacheWarmStore: FNPrefixWarmStore?
    /// The bytes the cold tier may hold. Zero means the disk tier is off.
    var prefixCacheSSDByteLimit: Int
    /// Prefix cache retention callback: (tokens, chargedBytes, storeTotalBytes, warmLimitBytes).
    public var onCacheRetained: ((Int, Int, Int, Int) -> Void)?
    /// Prefix cache oversize skip callback: (tokens, charge, warmLimitBytes).
    public var onCacheSkippedOversize: ((Int, Int, Int) -> Void)?
    /// Prefix cache eviction callback: (tokens, freedBytes).
    public var onCacheEvicted: ((Int, Int) -> Void)?

    /// The prefix cache as it stands on disk: how many saved prefixes
    /// it holds and how much room they take. A count that was not made
    /// is absent, never zero.
    public var prefixCacheDiskState: (entries: Int, bytes: Int)? {
        guard let dir = prefixCacheDirectory, prefixCacheSSDByteLimit > 0 else { return nil }
        return FNGPUEngine.diskState(in: dir)
    }

    /// The prefix cache as it stands in the RAM: how many prefixes the
    /// warm tier holds and how many bytes they take. Absent when there
    /// is no warm tier, never zero.
    public var prefixCacheWarmState: (entries: Int, bytes: Int)? {
        guard let warm = prefixCacheWarmStore, warm.isEnabled else { return nil }
        return warm.state
    }
    public var prefixCacheWarmLimitBytes: Int? { prefixCacheWarmStore?.byteLimit }

    /// Whether a cache tier is configured at all. A directory with no
    /// room behind it is no cache.
    public var hasPrefixCacheTier: Bool {
        prefixCacheWarmStore != nil || (prefixCacheDirectory != nil && prefixCacheSSDByteLimit > 0)
    }

    let pack: QWQuant.Pack?
    let gpuPLETable: FNGPUPLETable?
    let embedTable: FNGPUEmbedTable
    let decodeStack: FNGPUDecodeStack
    let verifyStack: FNGPUDecodeStack?
    let mtp: FNGPUMTP?
    let head: FNGPUHead
    let layerRecords: [LayerRecord]
    let maxPLEDim: Int
    let hidden: Int
    let hc: Int
    let width: Int
    let vocab: Int
    var prefillArena: QWArena?
    let extraWeightBytes: Int
    var cachedPrefill: (rows: Int, stack: FNGPUPrefillStack)?
    var lastPrefillStackRows: Int?
    var highWaterPos: Int = 0
    var residentKVWatermark = QWResidentKVWatermarkTracker<Int64>()
    var currentPartitionSlot: Int { get { residentKVWatermark.currentPartitionSlot } set { residentKVWatermark.currentPartitionSlot = newValue } }
    var currentPartitionTotalSlots: Int { get { residentKVWatermark.currentPartitionTotalSlots } set { residentKVWatermark.currentPartitionTotalSlots = newValue } }
    var residentPartitionTotalSlots: Int { get { residentKVWatermark.residentPartitionTotalSlots } set { residentKVWatermark.residentPartitionTotalSlots = newValue } }
    var residentKVHistoryBySlot: [Int: [Int64]] { get { residentKVWatermark.historyBySlot } set { residentKVWatermark.historyBySlot = newValue } }
    var residentKVHasMTPBySlot: [Int: Bool] { get { residentKVWatermark.hasMTPBySlot } set { residentKVWatermark.hasMTPBySlot = newValue } }
    var residentKVHistory: [Int64] { get { residentKVWatermark.currentHistory } set { residentKVWatermark.currentHistory = newValue } }
    var residentKVHasMTP: Bool { get { residentKVWatermark.currentHasMTP } set { residentKVWatermark.currentHasMTP = newValue } }
    var activeCapacity: Int {
        layerRecords.compactMap(\.attnState).first?.capacity ?? capacity
    }
    private var cachedSeenHistory: [Int64] = [], cachedSeenSet: Set<Int> = []
    public lazy var batchScheduler = FNGPUBatchScheduler(engine: self)

    public var arenaBytes: Int { arena.totalBytes + extraWeightBytes }
    public var weightUsedBytes: Int { arena.weightsSub.usedBytes + extraWeightBytes }
    public var weightReservedBytes: Int { arena.regionSizes.weights + extraWeightBytes }
    /// The storage format of the LM Head weight (`AQ{b}_G{g}` when native, `HQ{b}_G{g}` when rotated).
    public var headWeightFormatLabel: String {
        head.weight.formatName
    }
    /// The prefill stack is built for the shape of the prompt: a new
    /// shape builds a new one — a cold start, not a leak — in the
    /// prefill arena, which the steady-state counter does not cover.
    public var prefillStackNote: String {
        guard let rows = cachedPrefill?.rows ?? lastPrefillStackRows else { return "" }
        return "prefill stack:            \(rows) rows (built for this prompt's shape)\n"
    }

    /// The steady-state allocations this engine has made. The counters
    /// are armed when the init is done, so they count the steady state
    /// and not the loading.
    public var steadyStateAllocations: Int {
        arena.weightsRegion().steadyStateAllocs + arena.activationsRegion().steadyStateAllocs
            + arena.gdnRegion().steadyStateAllocs
    }
    public var supportsSpeculativeDecode: Bool { verifyStack != nil && mtp != nil }

    /// Default KV layout Flash-Next runs when no KV mode is requested.
    public static let implementedKVLayout = "f16"
    /// The L1 cache in front of the PLE n-gram table, in bytes.
    public static let ngramRowCacheBytes = 64 << 20
    public let kvMode: FNKVStorageMode
    /// The frozen numeric plan of the loaded model.
    public let plan: QWEnginePlan
    public let capacityPlan: QWCapacityPlan
    public let capacityLogMessage: String
    public let evictPrefillScratch: Bool
    /// Latent Sparse Attention configuration for this engine instance.
    public let lsaConfig: QWLSAConfig
    public let lsaClusterRegion: QWMRegion?

    /// - Parameter kvMode: nil = auto (run `f16`). Supports `f32`, `f16`, `q8`, and `q4`.
    /// - Parameter prefixCacheWarmByteLimit: the bytes the RAM may hold; zero
    ///   means the warm tier is off.
    public init(packPath: String, maxTokens: Int = 4096, kvMode: String? = nil, osReserveGiB: Double? = nil,
                prefillChunkSize: Int = QWPrefillLimits.chunkRows,
                prefixCacheDirectory: URL? = nil,
                prefixCacheWarmByteLimit: Int = 0,
                prefixCacheSSDByteLimit: Int = 0,
                maxConcurrentSessions: Int = FNPrefixWarmStore.defaultMaxSlots,
                evictPrefillScratch: Bool = Double(ProcessInfo.processInfo.physicalMemory) / 1_000_000_000.0 <= 36.5,
                prewarmDoors: [String] = [],
                lsaConfig: QWLSAConfig = QWLSAConfig(),
                streamNgrams: Bool = false) throws {
        self.evictPrefillScratch = evictPrefillScratch
        self.lsaConfig = lsaConfig
        self.prefixCacheDirectory = prefixCacheDirectory
        let resolvedMaxSessions = max(1, min(16, maxConcurrentSessions))
        let resolvedKVMode = try FNKVStorageMode.resolve(kvMode)
        self.kvMode = resolvedKVMode
        let expanded = packPath.expandingTildePath
        self.packPath = expanded
        let resolvedChunk = QWPrefillLimits.clampedRows(prefillChunkSize)
        self.maxPrefillRows = resolvedChunk
        let location = try QWModelLocation(url: URL(fileURLWithPath: expanded))
        let configURL = location.assets.appendingPathComponent("config.json")
        let desc = try QWModelDescriptor.decode(configJSON: Data(contentsOf: configURL))
        self.descriptor = desc

        let loader: FNPackLoader
        if location.kind == .qwp1 {
            let p = try QWQuant.Pack(url: location.weights, descriptor: desc)
            self.pack = p
            loader = FNPackLoader(pack: p, descriptor: desc)
        } else {
            self.pack = nil
            let st = try QWSafeTensorsContainer(url: location.weights)
            loader = FNPackLoader(st: st, descriptor: desc)
        }
        let useStandardBasis = loader.st.map(FNGPULoader.hasNativeAffineWeights) ?? false
        let hal = try QWHal(geometry: QWModelGeometry(descriptor: desc),
                            weightBasis: useStandardBasis ? .standard : .rotated)
        self.hal = hal
        let perTokBytes = resolvedKVMode.perTokenBytes
        let fnWeightsEstimate = Self.estimateWeightBytes(loader: loader, useStandardBasis: useStandardBasis)
        let packDiskBytes = (location.kind == .qwp1)
            ? ((try? FileManager.default.attributesOfItem(atPath: location.weights.path)[.size] as? Int) ?? 0) : 0
        let fnFixedScratchEstimate = (1408 << 20) + (836 << 20)
        let retainedPrefillBytes = (!evictPrefillScratch && resolvedChunk > 1024)
            ? Self.prefillActivationBytes(maxRows: resolvedChunk, capacity: min(maxTokens, desc.maxPositionEmbeddings)) : 0
        let availBytes = QWCapacityBudget.availableBytes(osReserveGiB: osReserveGiB)
        let minKVBytes = min(maxTokens, desc.maxPositionEmbeddings) * perTokBytes
        let resolvedPLE = try FNGPUPLETable.resolveBacking(
            assetsURL: location.assets, weightsURL: location.weights, hasPLE: !desc.pleModuleIndices.isEmpty,
            streamNgrams: streamNgrams, hal: hal, weightsEstimate: fnWeightsEstimate,
            fixedScratchBytes: fnFixedScratchEstimate + retainedPrefillBytes,
            minKVBytes: minKVBytes, availableBytes: availBytes)
        let gpuPLE = resolvedPLE.gpuTable
        self.gpuPLETable = gpuPLE
        let ngramCacheBytes = (desc.pleModuleIndices.isEmpty || resolvedPLE.residentBytes > 0) ? 0 : FNGPUEngine.ngramRowCacheBytes
        let rows = FNNgramCachedRows(backing: resolvedPLE.rows, capacityBytes: ngramCacheBytes)
        let admission = QWResidencyAdmission.planFlashNext(
            weightsEstimate: fnWeightsEstimate, packDiskBytes: packDiskBytes, gpuPLEBytes: resolvedPLE.residentBytes,
            ngramRowCacheBytes: ngramCacheBytes, baseScratchBytes: fnFixedScratchEstimate,
            retainedPrefillBytes: retainedPrefillBytes, lsaClusterBytes: 0, minKVBytes: minKVBytes,
            requestedWarmCacheBytes: prefixCacheWarmByteLimit, requestedSessions: resolvedMaxSessions,
            osReserveGiB: osReserveGiB,
            snapshot: QWLiveDeviceMemorySnapshotSource(maxWorkingSetBytes: hal.caps.maxWorkingSetBytes).capture())
        self.maxConcurrentSessions = admission.warmCache.maxConcurrentSessions
        let tiers = FNPrefixCacheTiers(warmByteLimit: admission.warmCache.admittedWarmBytes, ssdByteLimit: prefixCacheSSDByteLimit)
        self.prefixCacheSSDByteLimit = tiers.ssdByteLimit
        self.prefixCacheWarmStore = tiers.isEnabled
            ? FNPrefixWarmStore(byteLimit: tiers.warmByteLimit, maxSlots: self.maxConcurrentSessions) : nil
        let fnFixedBytes = admission.totalFixedBytes
        let plan = try QWCapacityPlan.plan(
            nativeWindow: desc.maxPositionEmbeddings, requested: maxTokens, perTokenBytes: perTokBytes,
            fixedBytes: fnFixedBytes, availableBytes: admission.availableBytes, yarnAllowed: true)
        guard plan.fits else { throw FNGPUError.contextDoesNotFit(requested: maxTokens, maxFitting: plan.maxFittingContext) }
        let alignedCap = QWCapacityPlan.steppedCapacity(plan.context)
        let kvCap = alignedCap >= 32_768 ? (alignedCap + 1024) : alignedCap
        let yarnFactor: Float? = alignedCap > desc.maxPositionEmbeddings ? Float(alignedCap) / Float(desc.maxPositionEmbeddings) : nil
        self.capacity = kvCap
        // The plan freezes what this engine actually runs: its geometry, its
        // KV mode and its prefill chunk. The digest binds all of it.
        self.plan = try QWEngineBootstrap.makePlan(
            geometry: QWModelGeometry(descriptor: desc), packPath: expanded,
            prefillChunkRows: resolvedChunk, kvStorageMode: resolvedKVMode.rawValue)
        self.capacityPlan = plan
        self.capacityLogMessage = try QWEngineBootstrap.admit(
            weightsBytes: admission.weightsBytes + resolvedPLE.residentBytes,
            fixedScratchBytes: admission.fixedScratchBytes + ngramCacheBytes,
            kvMode: resolvedKVMode.rawValue, perTokenBytes: perTokBytes,
            maxTokens: alignedCap, totalPlannedBytes: fnFixedBytes + alignedCap * perTokBytes,
            availableBytes: admission.availableBytes, osReserveGiB: osReserveGiB)

        let H = desc.hidden
        let C = desc.hcCount ?? 4
        let W = C * H
        let lowrank = desc.hcLowrank ?? 320
        self.hidden = H
        self.hc = C
        self.width = W
        self.vocab = desc.vocab

        let prefix = "model.language_model."
        defer { FNGPULoader.releaseRepackStageBuffer() }
        let hasHiBitDense = self.pack?.tensors.values.contains(where: { $0.shape.count == 2 && $0.hqBits > 4 }) ?? false
        let weightsArenaBytes = (self.pack != nil) ? ((hasHiBitDense ? 3584 : 2560) << 20) : (6 << 30)
        let baseScratchBytes = 1344 << 20
        let combinedActBytes = kvCap * perTokBytes + baseScratchBytes
        let maxBuf = Int(hal.device.maxBufferLength)
        let arena: QWArena
        if weightsArenaBytes + combinedActBytes <= maxBuf {
            arena = try QWArena(hal: hal, weights: weightsArenaBytes, activations: combinedActBytes, kv: 0, gdnStates: 0)
        } else {
            let sharedScratchBytes = baseScratchBytes + kvCap * 704
            let kvBytes = kvCap * perTokBytes
            arena = try QWArena(hal: hal, weights: weightsArenaBytes, activations: sharedScratchBytes, kv: kvBytes, gdnStates: 0)
        }
        self.arena = arena
        self.prefillArena = nil
        self.embedTable = try FNGPUEmbedTable.load(arena: arena, loader: loader, name: prefix + "embed_tokens.weight",
                                                   vocab: desc.vocab, hidden: H, hc: C)
        let ctx = try FNGPUContext(hal: hal, doors: prewarmDoors,
                                  geometry: QWModelGeometry(descriptor: desc))
        self.ctx = ctx

        let staging = (loader.st != nil && !useStandardBasis)
            ? hal.device.makeBuffer(length: 128 * 1024 * 1024, options: .storageModeShared)
            : nil
        var slots: [FNGPUDecodeStack.Slot] = []
        var records: [LayerRecord] = []
        var pleDimMax = 0
        var sharedRope: (cos: QWTensorValue, sin: QWTensorValue, bcos: QWTensorValue, bsin: QWTensorValue)?
        var sharedQSAScores1: QWTensorValue?
        for i in 0..<desc.numLayers {
            try autoreleasepool {
                let gw = try FNGPULoader.layer(i, loader: loader, arena: arena, staging: staging, yarnFactor: yarnFactor)
                let pleEmb = loader.pleEmbedding(i, pleRows: rows)
                var gdnM: FNGPUGDNMixer?
                var attnM: FNGPUAttnMixer?
                var state: FNGPUAttnDecodeState?
                switch gw.mixer {
                case .gdn(let g):
                    gdnM = g
                case .attention(let a):
                    attnM = a
                    let st = try FNGPUAttnDecodeState(arena: arena, mixer: a, hidden: H, capacity: kvCap,
                                                      kvMode: resolvedKVMode,
                                                      sharedRope: sharedRope, sharedQSAScores: sharedQSAScores1)
                    if sharedRope == nil {
                        sharedRope = (cos: st.cos, sin: st.sin, bcos: st.bcos, bsin: st.bsin)
                    }
                    if sharedQSAScores1 == nil {
                        sharedQSAScores1 = st.qsaScores
                    }
                    state = st
                }
                records.append(LayerRecord(weights: gw, attnState: state, ple: pleEmb,
                                           gdn: gdnM, attn: attnM, lowrank: gw.attnHC.lowrank))
                if let pl = gw.ple { pleDimMax = max(pleDimMax, pl.embedDim) }
                if (i & 3) == 3 { loader.st?.adviseDontNeed() }
            }
        }
        let sharedGDN = records.compactMap(\.gdn).first
        let sharedAttn = records.compactMap(\.attn).first
        let sharedDecodeScr: FNGPUScratch? = try records.first.map { firstRec in
            try FNGPUScratch(arena: arena, T: 1, hidden: H, hc: C, lowrank: firstRec.lowrank,
                             moe: firstRec.weights.moe, ple: nil, gdn: sharedGDN, attn: sharedAttn,
                             extraEmbedDim: pleDimMax, maxCapacity: kvCap)
        }
        for rec in records {
            let scr: FNGPUScratch
            if rec.weights.ple != nil || sharedDecodeScr == nil {
                scr = try FNGPUScratch(arena: arena, T: 1, hidden: H, hc: C, lowrank: rec.lowrank,
                                       moe: rec.weights.moe, ple: rec.weights.ple, gdn: rec.gdn ?? sharedGDN,
                                       attn: rec.attn ?? sharedAttn, maxCapacity: kvCap)
            } else {
                scr = sharedDecodeScr!
            }
            slots.append(.init(weights: rec.weights, scratch: scr, attn: rec.attnState, ple: rec.ple))
        }
        self.layerRecords = records
        self.maxPLEDim = pleDimMax
        self.decodeStack = try FNGPUDecodeStack(arena: arena, ctx: ctx, slots: slots, hidden: H, hc: C,
                                                gpuPLETable: gpuPLE)
        let mixerPrefix = prefix + "hyper_connection_mixer."
        let headBank = try FNGPULoader.hq4Bank(arena, loader, "lm_head.weight", n: desc.vocab, k: H, staging: staging)
        let mixerDown = try FNGPULoader.hq4Bank(arena, loader, mixerPrefix + "input_mix_weight_down.weight", n: lowrank, k: W, staging: staging)
        let mixerUp = try FNGPULoader.hq4Bank(arena, loader, mixerPrefix + "input_mix_weight_up.weight", n: W, k: lowrank, staging: staging)
        let mixerNorm = try FNGPULoader.f32Buffer(arena, try loader.tensor(mixerPrefix + "hc_norm.weight"))
        self.head = try FNGPUHead(arena: arena, hal: hal, head: headBank, mixerDown: mixerDown, mixerUp: mixerUp,
                                  normWeight: mixerNorm, eps: desc.rmsEps, hidden: H, hc: C, lowrank: lowrank, maxRows: 8)

        if FNGPUMTP.isAvailable(in: loader) {
            var vSlots: [FNGPUDecodeStack.Slot] = []
            var sharedQSAScores2: QWTensorValue?
            let sharedVerifyScr: FNGPUScratch? = try records.first.map { firstRec in
                try FNGPUScratch(arena: arena, T: 8, hidden: H, hc: C, lowrank: firstRec.lowrank,
                                 moe: firstRec.weights.moe, ple: nil, gdn: sharedGDN, attn: sharedAttn,
                                 extraEmbedDim: pleDimMax, maxCapacity: kvCap)
            }
            for rec in records {
                let st2 = try rec.attnState.flatMap { st in
                    try rec.attn.map { a -> FNGPUAttnDecodeState in
                        let s2 = try FNGPUAttnDecodeState(sharingCachesOf: st, arena: arena, mixer: a, rows: 8,
                                                          sharedQSAScores: sharedQSAScores2)
                        if sharedQSAScores2 == nil { sharedQSAScores2 = s2.qsaScores }
                        return s2
                    }
                }
                let scr2: FNGPUScratch
                if rec.weights.ple != nil || sharedVerifyScr == nil {
                    scr2 = try FNGPUScratch(arena: arena, T: 8, hidden: H, hc: C, lowrank: rec.lowrank,
                                            moe: rec.weights.moe, ple: rec.weights.ple, gdn: rec.gdn ?? sharedGDN,
                                            attn: rec.attn ?? sharedAttn, maxCapacity: kvCap)
                } else {
                    scr2 = sharedVerifyScr!
                }
                vSlots.append(.init(weights: rec.weights, scratch: scr2, attn: st2, ple: rec.ple))
            }
            self.verifyStack = try FNGPUDecodeStack(arena: arena, ctx: ctx, slots: vSlots, hidden: H, hc: C, rows: 8,
                                                    gpuPLETable: gpuPLE)
            self.mtp = try FNGPUMTP(arena: arena, ctx: ctx, loader: loader, embedTable: embedTable,
                                    headWeight: headBank, capacity: kvCap,
                                    maxPrefillRows: max(resolvedChunk, QWPrefillLimits.maxRows),
                                    kvMode: resolvedKVMode,
                                    sharedRope: sharedRope, sharedQSAScores1: sharedQSAScores1,
                                    sharedQSAScores2: sharedQSAScores2, staging: staging,
                                    yarnFactor: yarnFactor)
        } else {
            self.verifyStack = nil
            self.mtp = nil
        }
        // Native flat QSA wins <= 262K; above 262K, auto mode engages e-means-4096 coarse clustering.
        let useQSACluster = lsaConfig.mode == .qsaCluster
            || (lsaConfig.mode == .auto && kvCap > QWLSAConfig.qsaAutoClusterThresholdTokens)
        if useQSACluster, let attnM = sharedAttn {
            var allStates: [(FNGPUAttnDecodeState, FNGPUAttnDecodeState?)] = []
            for (slotIdx, rec) in records.enumerated() {
                if let st = rec.attnState { allStates.append((st, self.verifyStack?.slots[slotIdx].attn)) }
            }
            if let mtp = self.mtp { allStates.append((mtp.attnState1, mtp.attnState2)) }
            let fnClusterBlock = (lsaConfig.clusterBlock == QWLSAConfig.defaultClusterBlock)
                ? QWLSAConfig.emeansClusterBlock
                : lsaConfig.alignedClusterBlock
            self.lsaClusterRegion = try FNGPUAttnDecodeState.allocateLSAClusters(
                hal: hal, kvCap: kvCap, indexHeadDim: attnM.indexHeadDim,
                states: allStates, clusterBlock: fnClusterBlock)
        } else {
            self.lsaClusterRegion = nil
        }
        let mainWBuf = arena.weightsRegion().region.buffer
        var standaloneBytes = 0
        if self.pack == nil {
            for rec in records {
                if rec.weights.moe.gateUp.blocks.region.buffer !== mainWBuf { standaloneBytes += rec.weights.moe.gateUp.blocks.region.length }
                if rec.weights.moe.down.blocks.region.buffer !== mainWBuf { standaloneBytes += rec.weights.moe.down.blocks.region.length }
            }
            if let mtp = self.mtp {
                if mtp.layer.moe.gateUp.blocks.region.buffer !== mainWBuf { standaloneBytes += mtp.layer.moe.gateUp.blocks.region.length }
                if mtp.layer.moe.down.blocks.region.buffer !== mainWBuf { standaloneBytes += mtp.layer.moe.down.blocks.region.length }
            }
        }
        self.extraWeightBytes = standaloneBytes
        loader.st?.adviseDontNeed()
        hal.adoptPendingResidency()
        if let vStack = self.verifyStack {
            _ = try? vStack.forwardRows(history: [0, 0], embedTable: embedTable, head: head,
                                        readStream: false, verifyAllRows: true, mtp: self.mtp, activeRows: 2)
            highWaterPos = 2
            wipeSessionState()
        }
        if !evictPrefillScratch && dynamicMaxPrefillRows > 1024 {
            let bytes = Self.prefillActivationBytes(maxRows: dynamicMaxPrefillRows, capacity: capacity)
            if let pre = try? QWArena(hal: hal, weights: 0, activations: bytes, kv: 0, gdnStates: 0, keepAlive: false) {
                pre.prefault(); prefillArena = pre
            }
        }
        arena.weightsRegion().markSteadyState(); arena.activationsRegion().markSteadyState()
        arena.kvRegion().markSteadyState(); arena.kvVSub?.markSteadyState()
        arena.kvAuxSub?.markSteadyState(); arena.gdnRegion().markSteadyState()
        prefixCacheWarmStore?.onEvict = { [weak self] tokCount, freedBytes in self?.onCacheEvicted?(tokCount, freedBytes) }
    }

    deinit {
        if let b = gpuPLETable?.buffer { hal.evictResident([b]) }
    }

    /// Prefill arena budget: ~1.45 MB per prefill row plus tiled QSA score scratch.
    static func prefillActivationBytes(maxRows: Int, capacity: Int = 4096) -> Int {
        let qsaScoreBytes = max(min(maxRows, 64) * max(1, capacity / 4), min(maxRows, 4096) * 2048) * 4
        return maxRows * 1_450_000 + qsaScoreBytes + (16 << 20)
    }

    /// Computes actual resident GPU weight bytes from the checkpoint rather than a static placeholder.
    static func estimateWeightBytes(loader: FNPackLoader, useStandardBasis: Bool) -> Int {
        let hasHiBitDense = loader.pack?.tensors.values.contains(where: { $0.shape.count == 2 && $0.hqBits > 4 }) ?? false
        let minEstimate = (hasHiBitDense ? 3_408 : 2_384) << 20
        if let st = loader.st {
            var total = 0
            for t in st.tensors.values where !t.name.contains("ple_embed_tokens") {
                total += (!useStandardBasis && t.dtype == .bf16 && t.shape.count >= 2) ? (t.byteLength * 20) / 64 : t.byteLength
            }
            return max(minEstimate, total)
        }
        return minEstimate
    }

    private func buildPrefillStack(arena: QWArena, rows: Int) throws -> FNGPUPrefillStack {
        let firstRec = layerRecords[0]
        let sharedAttn = layerRecords.compactMap(\.attn).first
        let sharedGDN = layerRecords.compactMap(\.gdn).first
        let pscr = try FNGPUScratch(arena: arena, T: rows, hidden: hidden, hc: hc, lowrank: firstRec.lowrank,
                                    moe: firstRec.weights.moe, ple: firstRec.weights.ple,
                                    gdn: sharedGDN, attn: sharedAttn, extraEmbedDim: maxPLEDim, maxCapacity: capacity)
        var prefillSlots: [FNGPUPrefillStack.Slot] = []
        for rec in layerRecords {
            let pleBuf = try rec.weights.ple.map { try arena.activationsRegion().allocate(QWTensorShape([rows * $0.embedDim], .f32)) }
            let pleBufAlt = try (gpuPLETable == nil ? rec.weights.ple : nil).map {
                try arena.activationsRegion().allocate(QWTensorShape([rows * $0.embedDim], .f32))
            }
            prefillSlots.append(.init(weights: rec.weights, attn: rec.attnState, ple: rec.ple,
                                      pleEmbed: pleBuf, pleEmbedAlt: pleBufAlt, scratch: pscr))
        }
        return try FNGPUPrefillStack(arena: arena, ctx: ctx, slots: prefillSlots, hidden: hidden, hc: hc,
                                     gpuPLETable: gpuPLETable)
    }

    /// Dynamic prefill ceiling: respects configured `maxPrefillRows`.
    private var dynamicMaxPrefillRows: Int { max(32, maxPrefillRows) }

    private func ensurePrefillStack(rows: Int, allowShrink: Bool = true) throws -> FNGPUPrefillStack {
        let cap = dynamicMaxPrefillRows
        let allocRows = max(32, min(cap, ((rows + 31) / 32) * 32))
        if let cached = cachedPrefill, rows <= cached.stack.maxRows,
           !(evictPrefillScratch && allowShrink && cached.rows > 1024 && allocRows <= 512) {
            cached.stack.setActiveRows(rows)
            return cached.stack
        }
        let neededBytes = Self.prefillActivationBytes(maxRows: allocRows, capacity: capacity)
        let pArena: QWArena
        if let existing = prefillArena, existing.regionSizes.act >= neededBytes,
           !(evictPrefillScratch && allowShrink
             && existing.regionSizes.act > Self.prefillActivationBytes(maxRows: 1024, capacity: capacity)
             && allocRows <= 512) {
            existing.activationsSub.reset()
            pArena = existing
        } else {
            cachedPrefill = nil; prefillArena = nil
            let arenaRows = (!evictPrefillScratch && allocRows > 1024) ? cap : allocRows
            let arenaBytes = Self.prefillActivationBytes(maxRows: arenaRows, capacity: capacity)
            let created = try QWArena(hal: hal, weights: 0, activations: arenaBytes, kv: 0, gdnStates: 0, keepAlive: false)
            created.prefault()
            prefillArena = created; pArena = created
        }
        let stk = try buildPrefillStack(arena: pArena, rows: allocRows)
        stk.setActiveRows(rows)
        cachedPrefill = (rows: allocRows, stack: stk)
        lastPrefillStackRows = allocRows
        return stk
    }

    /// Releases the temporary prefill scratch arena once prefill has completed when scratch eviction is enabled.
    public func evictPrefillScratchIfNeeded() {
        guard evictPrefillScratch else { return }
        cachedPrefill = nil; prefillArena = nil
    }

    /// Runs ONE prefill chunk and advances the session's cursor. The head
    /// runs on the last chunk, and its token is the first generated one.
    private func prefillChunk(_ session: inout FNGPUSessionData, history: [Int64],
                               cacheHistory: [Int64],
                               chunkSize: Int, targetEnd: Int? = nil,
                               limitOverride: Int? = nil, gpuSec: inout Double) throws {
        let pos = session.prefillPos
        let limitPos = min(history.count, targetEnd ?? limitOverride ?? history.count)
        let subchunk = FNGPUPrefillStack.resolveSubchunk(
            pos: pos, limitPos: limitPos, chunkSize: chunkSize,
            hidden: hidden, hc: hc, visualSpans: session.visualSpans
        )
        let rows = subchunk.rows
        let isLast = (pos + rows == history.count)
        let isCheckpointEnd = (targetEnd != nil && pos + rows == limitPos && limitPos < history.count)
        let runHeadNow = isLast || isCheckpointEnd
        let runMTP = (session.useSpec || hasPrefixCacheTier) && mtp != nil
        session.lastRows = rows
        if rows == 1 {
            // Same single-row staging: the id goes into head.token, the gather runs on-device.
            arena.write(head.token, from: [Int32(history[pos])])
            session.headToken = Int(history[pos])
            let prefix = Array(history.prefix(pos + 1))
            if let streamIn = subchunk.visualStreamIn {
                session.lastArgmax = try decodeStack.forwardToken(
                    history: prefix, streamIn: streamIn, head: head, readStream: false
                ).token
            } else {
                session.lastArgmax = try decodeStack.forwardToken(
                    history: prefix, embedTable: embedTable, head: head, readStream: false
                ).token
            }
            session.headToken = session.lastArgmax
            gpuSec += decodeStack.lastGPUMillis / 1000.0
            if runMTP, let mtp {
                let nextTok = isLast ? Int32(session.lastArgmax) : Int32(history[pos + 1])
                let preInj = decodeStack.slots.last.map { ($0.scratch.h2, $0.scratch.moeOut, $0.scratch.injB) }
                session.mtpDraft = try mtp.forwardPrefill(streamBuf: decodeStack.lastStreamTensor, preInject: preInj,
                                                          nextTokens: [nextTok], startPos: pos,
                                                          prefillScratch: mtp.scratch1, projectLastRow: runHeadNow)
                gpuSec += mtp.lastGPUMillis / 1000.0
            }
            session.prefillPos += 1
            session.plePrecomputed = false
            if residentKVHistory.count == pos {
                residentKVHistory.append(cacheHistory[pos])
            }
            highWaterPos = max(highWaterPos, session.prefillPos)
            return
        }
        let rebuiltStack = (cachedPrefill == nil || rows > cachedPrefill!.stack.maxRows)
        let prefillStack = try ensurePrefillStack(rows: rows, allowShrink: pos == 0 && session.visualSpans.isEmpty)
        if rebuiltStack || prefillStack.maxRows != lastPrefillStackRows {
            session.plePrecomputed = false
            session.pleParity = 0
        }
        let nextRows = FNGPUPrefillStack.resolveSubchunk(
            pos: pos + rows, limitPos: limitPos, chunkSize: chunkSize,
            hidden: hidden, hc: hc, visualSpans: session.visualSpans
        ).rows
        let nextChunkPos: Int? = (!isCheckpointEnd && nextRows > 1 && nextRows <= prefillStack.maxRows) ? (pos + rows) : nil
        let fusedMTP = runMTP ? mtp : nil
        if let streamIn = subchunk.visualStreamIn {
            session.lastArgmax = try prefillStack.forward(history: history, streamIn: streamIn, head: head,
                                                          startPos: pos, readStream: false, runHead: runHeadNow,
                                                          mtp: fusedMTP, projectMTP: runHeadNow,
                                                          pleParity: session.pleParity,
                                                          plePrecomputed: session.plePrecomputed,
                                                          nextChunkStartPos: nextChunkPos,
                                                          nextChunkCount: nextRows).token
        } else {
            session.lastArgmax = try prefillStack.forward(history: history, embedTable: embedTable, head: head,
                                                          startPos: pos, readStream: false, runHead: runHeadNow,
                                                          mtp: fusedMTP, projectMTP: runHeadNow,
                                                          pleParity: session.pleParity,
                                                          plePrecomputed: session.plePrecomputed,
                                                          nextChunkStartPos: nextChunkPos,
                                                          nextChunkCount: nextRows).token
        }
        // The head wrote its argmax into `head.token`; the next step gathers
        // from it, so the session tracks what is on the device.
        if runHeadNow { session.headToken = session.lastArgmax }
        gpuSec += prefillStack.lastGPUMillis / 1000.0
        if fusedMTP != nil && runHeadNow {
            session.mtpDraft = prefillStack.lastMTPDraft
        }
        if nextChunkPos != nil {
            session.pleParity = 1 - session.pleParity
            session.plePrecomputed = true
        } else {
            session.plePrecomputed = false
        }
        session.prefillPos += rows
        if residentKVHistory.count == pos {
            residentKVHistory.append(contentsOf: cacheHistory[pos..<(pos + rows)])
        }
        highWaterPos = max(highWaterPos, session.prefillPos)
    }

    // MARK: - The module seam
    //
    // The engine owns the numerics. The generation loop lives in the core
    // (`QWEngineCore`), which drives these entry points one step at a time.

    /// The head's current logit row, for the shared sampler.
    public func currentLogits() -> [Float] { head.readLogits() }

    /// Computes the context-adaptive, 64-row-aligned prefill chunk size for a prompt of `totalTokens` tokens.
    /// Scales up to 8K (`QWPrefillLimits.maxRows = 8192`) on large contexts and balances chunk sizes evenly.
    public func dynamicPrefillChunkRows(forTotalTokens totalTokens: Int) -> Int {
        QWPrefillLimits.balancedChunkRows(forTotalTokens: totalTokens, maxChunkRows: dynamicMaxPrefillRows)
    }

    /// Prepares the prefill stack for a prompt of `count` tokens, so the
    /// allocation lands before the measured prefill and not inside it.
    public func preparePrefill(forTokenCount count: Int) throws {
        guard count > 1 else { return }
        let chunkRows = dynamicPrefillChunkRows(forTotalTokens: count)
        _ = try ensurePrefillStack(rows: min(chunkRows, count))
    }

    /// Starts a session: sizes the arena and prepares repetition-penalty tracking.
    public func beginSession(_ session: inout FNGPUSessionData, maxNew: Int) {
        _ = maxNew
        guard session.sampler.config.repetitionPenalty != 1.0 else {
            session.seenTokens = []
            return
        }
        let n = vocab
        let hist = session.history
        var seen: Set<Int>
        let startIdx: Int
        if !cachedSeenHistory.isEmpty, hist.count >= cachedSeenHistory.count,
           hist.prefix(cachedSeenHistory.count) == cachedSeenHistory[...] {
            seen = cachedSeenSet; startIdx = cachedSeenHistory.count
        } else {
            seen = Set<Int>(minimumCapacity: min(hist.count, 4096)); startIdx = 0
        }
        for i in startIdx..<hist.count {
            let id = Int(hist[i])
            if id >= 0 && id < n { seen.insert(id) }
        }
        cachedSeenHistory = hist; cachedSeenSet = seen; session.seenTokens = seen
    }

    /// Flushes a deferred prefix cache save after the first token has been emitted to the caller.
    public func flushPendingPrefixCache(_ session: inout FNGPUSessionData) {
        guard session.pendingPrefixSave else { evictPrefillScratchIfNeeded(); return }
        session.pendingPrefixSave = false
        if hasPrefixCacheTier {
            let cacheHistory = QWVisualSpan.saltedCacheHistory(history: session.history, visualSpans: session.visualSpans)
            savePrefixCache(history: cacheHistory, lastChunkRows: session.lastRows, useSpec: session.useSpec,
                            lastArgmax: session.lastArgmax, mtpDraft: session.mtpDraft, cacheDir: prefixCacheDirectory)
        }
        evictPrefillScratchIfNeeded()
    }

    /// Runs the prefill up to the core's chunk and returns the first generated token.
    public func encodePrefillChunk(_ session: inout FNGPUSessionData,
                                    chunk: Range<Int>, runsHead: Bool) throws -> Int {
        let history = session.history
        let cacheHistory = QWVisualSpan.saltedCacheHistory(history: history, visualSpans: session.visualSpans)
        let activeCap = activeCapacity
        guard history.count < activeCap else {
            throw FNGPUError.contextDoesNotFit(requested: history.count + 1, maxFitting: activeCap - 1)
        }
        var gpuSec = session.prefillGPUSec
        let canUsePrefixCache = hasPrefixCacheTier
        if !session.restoreDone {
            session.restoreDone = true
            let prevWater = highWaterPos
            if hasPrefixCacheTier, !cacheHistory.isEmpty,
               let restored = try lookupAndRestorePrefixCache(history: cacheHistory, rawHistory: history,
                                                              useSpec: session.useSpec,
                                                              cacheDir: prefixCacheDirectory, gpuSec: &gpuSec,
                                                              allowExact: true) {
                session.prefixCacheHit = true; session.prefixCacheMatchedTokens = restored.matched
                session.prefixCacheTier = restored.tier; session.prefillPos = restored.matched
                session.lastArgmax = restored.lastArgmax; session.headToken = restored.lastArgmax
                session.mtpDraft = restored.mtpDraft
            } else {
                highWaterPos = max(highWaterPos, prevWater)
                wipeSessionState(overwriteTokens: history.count)
                residentKVHasMTP = (session.useSpec || canUsePrefixCache) && mtp != nil
            }
            highWaterPos = max(highWaterPos, min(activeCap, history.count + 8))
        }
        let limit = runsHead ? history.count : min(chunk.upperBound, history.count)
        let autoSharedRetain = (prefixCacheWarmStore?.hasSharedBase == true && runsHead && history.count > session.prefillPos + 4)
            ? (history.count - 4) : nil
        let effectiveRetainLen = session.retainPrefixLength ?? autoSharedRetain ?? prefixCacheWarmStore?.sharedBranchPrefixLength(for: cacheHistory)
        if canUsePrefixCache, let retainLen = effectiveRetainLen, retainLen >= 128,
           retainLen < history.count, session.prefillPos < retainLen {
            let hasMTP = (session.useSpec || canUsePrefixCache) && mtp != nil
            let baseSlice = cacheHistory.prefix(retainLen)
            let baseHash = Self.fnPrefixHash(tokens: baseSlice, kvMode: kvMode, hasMTP: hasMTP)
            let baseName = String(format: "fn_entry_%016llx_%d.qwcache", baseHash, retainLen)
            if prefixCacheWarmStore?.contains(name: baseName) != true {
                let baseLimit = min(limit, retainLen)
                let baseStepChunk = dynamicPrefillChunkRows(forTotalTokens: max(1, retainLen - session.prefillPos))
                while session.prefillPos < baseLimit {
                    let nextRows = min(baseStepChunk, retainLen - session.prefillPos)
                    if !runsHead && session.prefillPos + nextRows > baseLimit { break }
                    try prefillChunk(&session, history: history, cacheHistory: cacheHistory,
                                     chunkSize: baseStepChunk, targetEnd: retainLen, gpuSec: &gpuSec)
                }
                if session.prefillPos == retainLen {
                    savePrefixCache(history: Array(baseSlice), lastChunkRows: session.lastRows, useSpec: session.useSpec,
                                    lastArgmax: session.lastArgmax, mtpDraft: session.mtpDraft,
                                    cacheDir: prefixCacheDirectory, isSharedBase: prefixCacheWarmStore?.hasSharedBase != true)
                }
            }
        }
        let uncachedTotal = max(1, history.count - max(session.prefillPos, session.prefixCacheHit ? session.prefixCacheMatchedTokens : 0))
        let stepChunk = dynamicPrefillChunkRows(forTotalTokens: uncachedTotal)
        while session.prefillPos < limit {
            let nextRows = FNGPUPrefillStack.resolveSubchunk(
                pos: session.prefillPos, limitPos: limit, chunkSize: stepChunk,
                hidden: hidden, hc: hc, visualSpans: session.visualSpans
            ).rows
            if !runsHead && session.prefillPos + nextRows > limit { break }
            try prefillChunk(&session, history: history, cacheHistory: cacheHistory,
                             chunkSize: stepChunk, limitOverride: limit, gpuSec: &gpuSec)
        }
        if runsHead && session.prefillPos >= history.count && !session.hasFirstToken {
            session.prefillGPUSec = gpuSec
            let firstTok = try drawFirstToken(&session)
            if canUsePrefixCache { session.pendingPrefixSave = true } else { evictPrefillScratchIfNeeded() }
            return firstTok
        }
        session.prefillGPUSec = gpuSec
        return session.hasFirstToken ? session.firstToken : session.lastArgmax
    }

    /// Runs one decode step and returns the token the model produced (using `sampler: QWSampler`).
    public func encodeDecodeToken(_ session: inout FNGPUSessionData, token: Int,
                                   position: Int) throws -> Int {
        if session.pendingPrefixSave {
            flushPendingPrefixCache(&session)
        }
        // A speculative round that committed two tokens already fed both to
        // the model. The second one is handed back now, without a GPU round.
        if !session.pendingTokens.isEmpty {
            if !session.roundTokenCounts.isEmpty {
                session.roundTokenCounts[session.roundTokenCounts.count - 1] += 1
            }
            return session.pendingTokens.removeFirst()
        }
        let activeCap = activeCapacity
        guard session.history.count + 1 < activeCap else {
            // The context is full: the request cannot be served in full, and
            // a truncated answer is not an answer the caller asked for.
            throw FNGPUError.contextDoesNotFit(requested: session.history.count + 1,
                                                maxFitting: activeCap - 1)
        }
        if session.useSpec, let vStack = verifyStack, let mtp,
           session.history.count + 8 < activeCap {
            return try speculativeStep(&session, token: token, vStack: vStack, mtp: mtp)
        }
        // The gather reads the token from the device, so the token to feed is
        // staged before the step runs.
        if session.headToken != token {
            arena.write(head.token, from: [Int32(token)])
            session.headToken = token
        }
        let prevCount = session.history.count
        session.history.append(Int64(token))
        if residentKVHistory.count == prevCount {
            residentKVHistory.append(Int64(token))
        }
        highWaterPos = max(highWaterPos, session.history.count)
        if session.sampler.config.repetitionPenalty != 1.0 && token >= 0 && token < vocab {
            session.seenTokens.insert(token)
        }
        let stepT0 = Date()
        let r = try decodeStack.forwardToken(history: session.history, embedTable: embedTable,
                                              head: head, readStream: false)
        session.stepMs = Date().timeIntervalSince(stepT0) * 1000.0
        session.lastStepGPUMillis = decodeStack.lastGPUMillis
        session.headToken = r.token
        let next = try sampleFromHead(argmaxToken: r.token, seenTokens: session.seenTokens,
                                      sampler: session.sampler, constraint: session.constraint)
        if session.collectLogprobs {
            recordTokenLogprob(&session, token: next, row: 0)
        }
        if next != r.token {
            arena.write(head.token, from: [Int32(next)])
            session.headToken = next
        }
        return next
    }
}
