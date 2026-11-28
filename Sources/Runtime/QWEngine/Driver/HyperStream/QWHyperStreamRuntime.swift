// QWHyperStreamRuntime.swift — High-level GPU inference engine for Qwen3.8-Flash-Next (48 layers, MoE + GDN + QSA + PLE).
// Integrates FNGPUDecodeStack, FNGPUPrefillStack, and FNGPUHead for CLI (`qwg run`/`repl`) and HTTP server (`qwg serve`).

import Foundation
import QWAlloc
import QWAttn
import QWCore
import QWDecision
import QWGEMM
import QWHAL
import QWHead
import QWHyperPLE
import QWMTP
import QWQuant
import QWVision

public final class QWHyperStreamRuntime: @unchecked Sendable {
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
    public let geometry: QWModelGeometry
    public let hasVisionWeights: Bool
    let layerKernels: [any QWLayerKernel]
    public let hal: QWHal
    let stateLock = NSLock()
    var _onCacheEvent: (@Sendable (QWPrefixCacheEvent) -> Void)?
    var _gpuTurnArbitrator: QWGPUTurnArbitrator?
    var _prefixCacheHit: Bool = false
    var _prefixCacheMatchedTokens: Int = 0
    var _prefixCacheTier: FNPrefixCacheTier?
    var _naxLogMessage: String = ""
    var _lastResult: QWGenerationResult?
    var _lastSpecProposals: Int = 0
    var _lastSpecAccepted: Int = 0
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
        return QWHyperStreamRuntime.diskState(in: dir)
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
    public let dflash: QWDFlashDrafter?
    let dflashPrefixMarker: String?
    let dflashHeadStorage: QWHeadStorage?
    public let dflashMaxDrafts: Int
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
    var residentKVHistory: [Int64] { get { residentKVWatermark.currentHistory } set { residentKVWatermark.currentHistory = newValue } }
    var residentKVHasMTP: Bool { get { residentKVWatermark.currentHasMTP } set { residentKVWatermark.currentHasMTP = newValue } }
    public var activeCapacity: Int {
        layerRecords.compactMap(\.attnState).first?.capacity ?? capacity
    }

    /// Resolves KV capacity, adding 1,024-token decode headroom for `alignedCap >= 4096` when RAM permits.
    public static func resolveKVCapacity(
        alignedCap: Int,
        fixedBytes: Int,
        perTokenBytes: Int,
        availableBytes: Int
    ) -> Int {
        QWCapacityBudget.resolveKVCapacity(
            alignedCap: alignedCap,
            fixedBytes: fixedBytes,
            perTokenBytes: perTokenBytes,
            availableBytes: availableBytes
        )
    }
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
    public var supportsSpeculativeDecode: Bool { verifyStack != nil && (mtp != nil || dflash != nil) }
    public var dflashReportLine: String {
        guard let dflash else { return "" }
        let kind = dflash.config.isDSpine ? "dspine" : "dflash2"
        return "dflash:                   \(kind) layers=\(dflash.config.numLayers) win=\(dflash.config.slidingWindow) block<=\(dflashMaxDrafts)\n"
    }

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
                streamNgrams: Bool = false,
                ropeYarnFactor: Double? = nil,
                mtpBackend: String? = nil,
                mtpDrafts: Int = 0,
                dflashDraftPackPath: String? = nil,
                dflashSelectorTopK: Int? = nil,
                mtpDraftStrategy: String = "hybrid") throws {
        _ = mtpDraftStrategy
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
        let useDFlash = FNGPUDFlash.isDFlashName(mtpBackend) && mtpDrafts > 0
        let perTokBytes = FNGPUDFlash.adjustedPerTokenKVBytes(resolvedKVMode.perTokenBytes, usesDFlash: useDFlash)
        let dflashFootprint = FNGPUDFlash.estimateFootprint(
            targetDescriptor: desc, targetPackPath: expanded,
            dflashPackPath: dflashDraftPackPath, isDFlash: useDFlash)
        let rawWeightsEstimate = Self.estimateWeightBytes(loader: loader, useStandardBasis: useStandardBasis)
        let fnWeightsEstimate = useDFlash
            ? max(512 << 20, rawWeightsEstimate - dflashFootprint.nativeMTPWeightBytes + dflashFootprint.dflashWeightBytes)
            : rawWeightsEstimate
        let rawPackDiskBytes = (location.kind == .qwp1)
            ? ((try? FileManager.default.attributesOfItem(atPath: location.weights.path)[.size] as? Int) ?? 0) : 0
        let packDiskBytes = (useDFlash && rawPackDiskBytes > 0)
            ? max(512 << 20, rawPackDiskBytes - dflashFootprint.nativeMTPWeightBytes + dflashFootprint.dflashWeightBytes)
            : rawPackDiskBytes
        let fnFixedScratchEstimate = (1408 << 20) + (836 << 20)
            + (useDFlash ? max(0, dflashFootprint.dflashScratchBytes - (112 << 20)) : 0)
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
        let ngramCacheBytes = (desc.pleModuleIndices.isEmpty || resolvedPLE.residentBytes > 0) ? 0 : QWHyperStreamRuntime.ngramRowCacheBytes
        let rows = FNNgramCachedRows(backing: resolvedPLE.rows, capacityBytes: ngramCacheBytes)
        let admission = QWResidencyAdmission.planHyperStream(
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
        let kvCap = Self.resolveKVCapacity(
            alignedCap: alignedCap,
            fixedBytes: fnFixedBytes,
            perTokenBytes: perTokBytes,
            availableBytes: admission.availableBytes
        )
        let resolvedYarn = ProcessInfo.processInfo.environment["QW_ROPE_YARN_FACTOR"].flatMap(Double.init) ?? ropeYarnFactor
        let yarnFactor: Float? = resolvedYarn.map { Float($0) }
            ?? (alignedCap > desc.maxPositionEmbeddings ? Float(alignedCap) / Float(desc.maxPositionEmbeddings) : nil)
        self.capacity = kvCap
        // The plan freezes what this engine actually runs: its geometry, its
        // KV mode and its prefill chunk. The digest binds all of it.
        let modelGeometry = QWModelGeometry(descriptor: desc)
        self.geometry = modelGeometry
        self.plan = try QWEngineBootstrap.makePlan(
            geometry: modelGeometry, packPath: expanded,
            prefillChunkRows: resolvedChunk, kvStorageMode: resolvedKVMode.rawValue)
        self.layerKernels = try QWLayerKernelRegistry.resolve(plan: self.plan)
        self.hasVisionWeights = desc.capabilities.contains(.vision)
            && desc.visDepth > 0
            && QWVisionOnDemandBackend.hasVisionWeights(modelPath: expanded)
        self.capacityPlan = plan
        self.capacityLogMessage = try QWEngineBootstrap.admit(
            weightsBytes: admission.weightsBytes + resolvedPLE.residentBytes,
            fixedScratchBytes: admission.fixedScratchBytes + ngramCacheBytes,
            kvMode: resolvedKVMode.rawValue, perTokenBytes: perTokBytes,
            maxTokens: alignedCap, totalPlannedBytes: fnFixedBytes + kvCap * perTokBytes,
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
        let weightsArenaBytes = (self.pack != nil)
            ? ((useDFlash ? (hasHiBitDense ? 3328 : 2304) : (hasHiBitDense ? 3584 : 2560)) << 20)
            : (6 << 30)
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

        let loadVerifyStack = useDFlash || FNGPUMTP.isAvailable(in: loader)
        if loadVerifyStack {
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
            if useDFlash {
                let resolvedDFlashPack = FNGPUDFlash.resolveDrafterPackPath(
                    targetDescriptor: desc, explicitPath: dflashDraftPackPath)
                let drafter = try QWDFlashDrafter(
                    geometry: QWModelGeometry(descriptor: desc), hal: hal,
                    packPath: resolvedDFlashPack,
                    drafts: max(1, min(FNGPUDFlash.maxDrafts, mtpDrafts)),
                    selectorTopK: dflashSelectorTopK)
                self.dflash = drafter
                self.dflashPrefixMarker = FNGPUDFlash.dflashGeometryMarker(config: drafter.config)
                self.dflashMaxDrafts = min(FNGPUDFlash.maxDrafts, max(1, drafter.config.blockSize - 1))
                self.dflashHeadStorage = QWHeadWeight(
                    blocks: headBank.blocks, n: headBank.n, k: headBank.k,
                    bits: headBank.bits, groupSize: headBank.groupSize,
                    isRotated: headBank.isRotated, isIncumbentContainer: headBank.isIncumbentContainer,
                    isSHVQ: headBank.isSHVQ
                ).map { .q4g($0) }
                self.mtp = nil
            } else {
                self.dflash = nil
                self.dflashPrefixMarker = nil
                self.dflashMaxDrafts = 0
                self.dflashHeadStorage = nil
                self.mtp = try FNGPUMTP(arena: arena, ctx: ctx, loader: loader, embedTable: embedTable,
                                        headWeight: headBank, capacity: kvCap,
                                        maxPrefillRows: max(resolvedChunk, QWPrefillLimits.maxRows),
                                        kvMode: resolvedKVMode,
                                        sharedRope: sharedRope, sharedQSAScores1: sharedQSAScores1,
                                        sharedQSAScores2: sharedQSAScores2, staging: staging,
                                        yarnFactor: yarnFactor)
            }
        } else {
            self.verifyStack = nil
            self.mtp = nil
            self.dflash = nil
            self.dflashPrefixMarker = nil
            self.dflashMaxDrafts = 0
            self.dflashHeadStorage = nil
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
        let dflashBytes = self.dflash.map { FNGPUDFlash.drafterWeightBytes(packPath: $0.packPath, isDSpine: $0.config.isDSpine) } ?? 0
        self.extraWeightBytes = Self.standaloneWeightBytes(
            arenaBuffers: arena.allBuffers,
            layers: self.layerRecords.map(\.weights),
            head: self.head,
            embedTable: self.embedTable,
            gpuPLETable: self.gpuPLETable,
            mtp: self.mtp
        ) + dflashBytes
        loader.st?.adviseDontNeed()
        hal.adoptPendingResidency()
        hal.primeResidency()
        if let vStack = self.verifyStack {
            _ = try? vStack.forwardRows(history: [0, 0], embedTable: embedTable, head: head,
                                        readStream: false, verifyAllRows: true, mtp: self.mtp,
                                        activeRows: 2, dflash: self.dflash)
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
        wireRuntimeCallbacks()
    }

    deinit {
        if let b = gpuPLETable?.buffer { hal.evictResident([b]) }
    }

    /// Evaluates log-probabilities for prompt tokens `[1..<count]` in a single incremental pass.
    public func computeEchoPromptLogprobs(
        promptTokens: [Int],
        sampling: QWSamplerConfig,
        topLogprobsCount: Int,
        shouldStop: (@Sendable () -> Bool)? = nil
    ) throws -> [QWTokenLogprob] {
        guard promptTokens.count >= 2 else { return [] }
        return try batchScheduler.runExclusive {
            let activeCap = activeCapacity
            guard promptTokens.count < activeCap else {
                throw FNGPUError.contextDoesNotFit(
                    requested: promptTokens.count + 1,
                    maxFitting: FNGPUDFlash.maxFittingContext(activeCapacity: activeCap)
                )
            }
            wipeSessionState()
            defer { wipeSessionState() }
            var session = FNGPUSessionData(
                history: [Int64(promptTokens[0])],
                sampler: QWSampler(config: sampling),
                useSpec: false,
                collectLogprobs: false,
                topLogprobsCount: max(0, min(20, topLogprobsCount)),
                usePrefixCache: false
            )
            beginSession(&session, maxNew: 0)
            _ = try encodePrefillChunk(&session, chunk: 0..<1, runsHead: true)
            var out: [QWTokenLogprob] = []
            out.reserveCapacity(promptTokens.count - 1)
            recordTokenLogprob(&session, token: promptTokens[1], row: 0)
            if let lp = session.tokenLogprobs.last { out.append(lp) }
            for i in 1..<(promptTokens.count - 1) {
                if shouldStop?() == true { break }
                _ = try encodeDecodeToken(&session, token: promptTokens[i], position: i)
                recordTokenLogprob(&session, token: promptTokens[i + 1], row: 0)
                if let lp = session.tokenLogprobs.last { out.append(lp) }
            }
            return out
        }
    }

    /// Validates decision evaluation inputs and returns an early output when no GPU work is needed.
    static func prepareDecisionEvaluation(
        tokens: [Int],
        positions: [Int],
        candidateTokenIds: [Int],
        vocab: Int,
        activeCapacity: Int
    ) throws -> [QWDecisionSlotOutput]? {
        guard !tokens.isEmpty else {
            throw QWDecisionError.invalidQuestion("decision sequence must contain at least one token")
        }
        if positions.isEmpty { return [] }
        let maxTokensLimit = FNGPUDFlash.maxFittingContext(activeCapacity: activeCapacity)
        guard tokens.count <= maxTokensLimit else {
            throw QWDecisionError.contextOverflow(tokens: tokens.count, limit: maxTokensLimit)
        }
        for tok in tokens where tok < 0 || tok >= vocab {
            throw QWEngineDriverError.badToken(tok)
        }
        for pos in positions where pos < 0 || pos >= tokens.count {
            throw QWDecisionError.invalidQuestion("slot position \(pos) outside sequence 0..<\(tokens.count)")
        }
        if !candidateTokenIds.isEmpty,
           !candidateTokenIds.contains(where: { (0..<vocab).contains($0) }) {
            return positions.map { _ in QWDecisionSlotOutput(tokenLogits: [:]) }
        }
        return nil
    }

    /// Evaluates single-pass decision logits at `positions` inside `tokens` using an ephemeral session.
    public func evaluateDecisionLogits(
        tokens: [Int],
        positions: [Int],
        candidateTokenIds: [Int],
        visualSpans: [QWVisualSpan] = []
    ) throws -> [QWDecisionSlotOutput] {
        try batchScheduler.runExclusive {
            if let early = try Self.prepareDecisionEvaluation(
                tokens: tokens,
                positions: positions,
                candidateTokenIds: candidateTokenIds,
                vocab: vocab,
                activeCapacity: activeCapacity
            ) {
                return early
            }
            wipeSessionState()
            defer {
                wipeSessionState()
                evictPrefillScratchIfNeeded()
            }

            let sortedPositions = Array(Set(positions)).sorted()
            let prefixTokens = tokens.map(Int64.init)
            var session = FNGPUSessionData(
                history: [],
                sampler: QWSampler(config: .greedy),
                useSpec: false,
                collectLogprobs: false,
                visualSpans: visualSpans,
                usePrefixCache: false
            )
            beginSession(&session, maxNew: 0)
            try preparePrefill(forTokenCount: tokens.count)

            var capturedByPosition: [Int: QWDecisionSlotOutput] = [:]
            capturedByPosition.reserveCapacity(sortedPositions.count)

            for targetPos in sortedPositions {
                let targetEnd = targetPos + 1
                session.history.append(contentsOf: prefixTokens[session.history.count..<targetEnd])
                session.prefillStepChunk = 0
                var cur = session.prefillPos
                let chunkCap = dynamicPrefillChunkRows(forTotalTokens: targetEnd - cur)
                while cur < targetEnd {
                    let end = min(targetEnd, cur + chunkCap)
                    _ = try encodePrefillChunk(
                        &session,
                        chunk: cur..<end,
                        runsHead: end == targetEnd
                    )
                    cur = end
                }
                let logitsPtr = arena.cpuPointer(of: head.logits).assumingMemoryBound(to: Float.self)
                let slotOut = withExtendedLifetime(arena) {
                    QWTrunkRuntimeSupport.extractDecisionSlotOutput(
                        logitsRow: logitsPtr,
                        vocab: vocab,
                        candidateTokenIds: candidateTokenIds
                    )
                }
                capturedByPosition[targetPos] = slotOut
            }
            return positions.map { capturedByPosition[$0] ?? QWDecisionSlotOutput() }
        }
    }

}
