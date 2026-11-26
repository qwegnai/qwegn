// QWDecisionPackEngine.swift — Capability-dispatched native Metal 4 backend for System-1 decision packs.

import Dispatch
import Foundation
import ImageIO
import QWAlloc
import QWAttn
import QWCore
import QWDecision
import QWGDN
import QWHAL
import QWHead
import QWMM
import QWQuant
import QWVision

/// Executes single-pass System-1 decisions and embeddings natively on Metal 4 across all head kinds.
public final class QWDecisionPackEngine: QWDecisionExecutionBackend, QWEmbeddingExecutionBackend, @unchecked Sendable {
    public let descriptor: QWModelDescriptor
    public let decisionConfig: QWDecisionModelConfig
    public let maxContextLength: Int
    public var embeddingDimension: Int { descriptor.hidden }
    public let gpuTurnArbitrator: QWGPUTurnArbitrator?
    public let isSidecar: Bool
    public let visionBackend: (any QWVisionBackend)?
    public let loadVision: Bool

    let hal: QWHal
    let weights: QWDecisionPackWeights
    let shortConv: QWShortConv
    let topoAttn: QWDecisionTopologyAttn
    let headOp: QWDecisionHeadOperator
    let scratchArena: QWDecisionScratchArena

    private static let evalQueue = DispatchQueue(label: "qwegn.decision.pack.metal", qos: .userInteractive)
    private static let maxImageSourceBytes = 20 * 1024 * 1024
    private static let maxImageDimension = 16_384
    private static let maxImageArea: Int64 = 67_108_864

    /// Returns true when `descriptor` and `root` are runnable by the generalized Metal 4 decision engine.
    public static func canLoad(descriptor: QWModelDescriptor, root: QWModelRoot) -> Bool {
        guard descriptor.decisionConfig != nil else { return false }
        guard root.kind == .qwp1 || root.kind == .safetensorsSingle || root.kind == .safetensorsSharded else { return false }
        guard descriptor.numLayers > 0, descriptor.layerTypes.count == descriptor.numLayers else { return false }
        guard (descriptor.numExperts ?? 0) == 0, (descriptor.denseIntermediate ?? 0) > 0 else {
            return false
        }
        switch descriptor.activationKind {
        case .silu, .gelu, .geluTanh:
            break
        case .situ:
            return false
        }
        let supported: Set<String> = [
            "conv", "shortconv", "centered_conv",
            "full_attention", "sliding_attention", "attention",
            "encoder_attention", "bidirectional_attention",
        ]
        return descriptor.layerTypes.allSatisfy { supported.contains($0) }
    }

    public init(
        packPath: String,
        descriptor: QWModelDescriptor,
        gpuTurnArbitrator: QWGPUTurnArbitrator? = nil,
        isSidecar: Bool = true,
        visionBackend: (any QWVisionBackend)? = nil,
        loadVision: Bool = true
    ) throws {
        guard let dc = descriptor.decisionConfig else {
            throw QWDecisionError.backendFailed("missing decisionConfig for native Metal decision pack")
        }
        let root = QWModelRoot(url: URL(fileURLWithPath: packPath))
        let geometry = QWModelGeometry(descriptor: descriptor)
        let usesRotated = try QWDecisionWeightLoader.probeRotatedBasis(root: root, descriptor: descriptor)
        let hal = try QWHal(geometry: geometry, weightBasis: usesRotated ? .rotated : .standard)
        self.hal = hal
        self.descriptor = descriptor
        self.decisionConfig = dc
        self.maxContextLength = max(1, descriptor.maxPositionEmbeddings)
        self.gpuTurnArbitrator = gpuTurnArbitrator
        self.isSidecar = isSidecar
        self.visionBackend = visionBackend
        self.loadVision = loadVision
        self.weights = try QWDecisionWeightLoader.load(root: root, descriptor: descriptor, hal: hal)
        self.shortConv = try QWShortConv(hal: hal)
        self.topoAttn = try QWDecisionTopologyAttn(hal: hal)
        self.headOp = try QWDecisionHeadOperator(hal: hal)
        self.scratchArena = QWDecisionScratchArena(hal: hal)
    }

    public func evaluate(_ request: QWDecisionBackendBatchRequest) async throws -> QWDecisionBackendOutput {
        let t0 = ContinuousClock.now
        let grouped = request.slots.enumerated().reduce(into: [Int: [(Int, QWDecisionSlotTarget)]]()) { acc, pair in
            acc[pair.element.sequenceIndex, default: []].append((pair.offset, pair.element))
        }
        let sortedSeqs = grouped.keys.sorted()
        var encodedCache: [Data: (grid: QWVisionGrid, features: [Float])] = [:]
        var prepared: [PreparedDecisionSeq] = []
        prepared.reserveCapacity(sortedSeqs.count)

        for seqIdx in sortedSeqs {
            guard seqIdx >= 0 && seqIdx < request.sequences.count else {
                throw QWDecisionError.invalidQuestion("sequence index \(seqIdx) out of bounds")
            }
            let seq = request.sequences[seqIdx]
            let entries = grouped[seqIdx] ?? []
            prepared.append(try await prepareSequence(seq: seq, entries: entries, cache: &encodedCache))
        }

        let candIds = request.uniqueCandidateTokenIds.isEmpty
            ? Array(Set(request.slots.flatMap { $0.candidateTokenGroups.flatMap { $0 } })).sorted()
            : request.uniqueCandidateTokenIds
        let evalClosure: @Sendable () async throws -> [QWDecisionSlotOutput] = { [self, prepared, candIds] in
            try await withCheckedThrowingContinuation { continuation in
                Self.evalQueue.async {
                    do {
                        var out = [QWDecisionSlotOutput](repeating: QWDecisionSlotOutput(), count: request.slots.count)
                        let seqRes = try self.executeBatchOnGPU(prepared, candidateIds: candIds)
                        for (slotIdx, val) in seqRes {
                            if slotIdx < out.count { out[slotIdx] = val }
                        }
                        continuation.resume(returning: out)
                    } catch {
                        continuation.resume(throwing: error)
                    }
                }
            }
        }

        let slotOutputs: [QWDecisionSlotOutput]
        if isSidecar, let arb = gpuTurnArbitrator {
            slotOutputs = try await arb.runHighPrioritySidecar { try await evalClosure() }
        } else {
            slotOutputs = try await evalClosure()
        }

        let elapsed = t0.duration(to: ContinuousClock.now)
        let latencyMs = Double(elapsed.components.seconds) * 1000.0 + Double(elapsed.components.attoseconds) / 1e15
        let promptTokens = prepared.reduce(0) { $0 + $1.tokens.count }
        return QWDecisionBackendOutput(
            slotOutputs: slotOutputs, promptTokens: promptTokens, latencyMs: (latencyMs * 100.0).rounded() / 100.0
        )
    }

    /// Computes pooled, Matryoshka-truncated, L2-normalized embeddings in coalesced Metal 4 batches.
    public func embedBatch(
        tokenBatches: [[Int]],
        pooling: QWDecisionModelConfig.PoolingKind,
        normalize: Bool,
        truncateDimensions: Int?
    ) async throws -> (embeddings: [[Float]], promptTokens: Int, latencyMs: Double) {
        guard !tokenBatches.isEmpty else { return ([], 0, 0) }
        let t0 = ContinuousClock.now
        let prepared: [PreparedDecisionSeq] = tokenBatches.map { toks in
            let clamped = Array(toks.prefix(maxContextLength))
            return PreparedDecisionSeq(
                tokens: clamped, visualSpans: [], encodedImages: [],
                topology: decisionConfig.topology, stateEndPos: 0, questionSpans: [],
                markerPositions: [], chain: "default",
                expandedStateLen: 0, origToExpandedPtr: [], origPtrLen: 0, slotTargets: []
            )
        }
        let promptTokens = prepared.reduce(0) { $0 + $1.tokens.count }
        let runClosure: @Sendable () async throws -> [[Float]] = { [self, prepared, pooling, normalize, truncateDimensions] in
            try await withCheckedThrowingContinuation { continuation in
                Self.evalQueue.async {
                    do {
                        let maxTokensPerCB = 16_384
                        var out: [[Float]] = []
                        out.reserveCapacity(prepared.count)
                        var start = 0
                        while start < prepared.count {
                            var end = start
                            var tokSum = 0
                            while end < prepared.count {
                                let len = prepared[end].tokens.count
                                if end > start && tokSum + len > maxTokensPerCB { break }
                                tokSum += len
                                end += 1
                            }
                            let chunk = Array(prepared[start..<end])
                            out.append(contentsOf: try self.executeEmbedChunkOnGPU(
                                chunk, pooling: pooling, normalize: normalize, truncateDimensions: truncateDimensions
                            ))
                            start = end
                        }
                        continuation.resume(returning: out)
                    } catch {
                        continuation.resume(throwing: error)
                    }
                }
            }
        }
        let vectors: [[Float]]
        if isSidecar, let arb = gpuTurnArbitrator {
            vectors = try await arb.runHighPrioritySidecar { try await runClosure() }
        } else {
            vectors = try await runClosure()
        }
        let elapsed = t0.duration(to: ContinuousClock.now)
        let latencyMs = Double(elapsed.components.seconds) * 1000.0 + Double(elapsed.components.attoseconds) / 1e15
        return (vectors, promptTokens, (latencyMs * 100.0).rounded() / 100.0)
    }

    private func prepareSequence(
        seq: QWDecisionSequenceInput,
        entries: [(Int, QWDecisionSlotTarget)],
        cache: inout [Data: (grid: QWVisionGrid, features: [Float])]
    ) async throws -> PreparedDecisionSeq {
        let rawPtrLen = seq.pointerValidMask.isEmpty ? max(1, seq.stateEndPosition) : seq.pointerValidMask.count
        if seq.images.isEmpty {
            if let vb = visionBackend, vb.isLoaded { try vb.prepareForPrompt(hasImages: false) }
            let slotTargets = entries.map { (slotIdx: $0.0, pos: $0.1.tokenPosition, endPos: $0.1.pointerEndPosition, isPtr: $0.1.isPointerSlot) }
            return PreparedDecisionSeq(
                tokens: seq.tokenIds, visualSpans: [], encodedImages: [],
                topology: seq.topology, stateEndPos: seq.stateEndPosition, questionSpans: seq.questionSpans,
                markerPositions: seq.markerPositions, chain: seq.chain,
                expandedStateLen: min(rawPtrLen, seq.tokenIds.count), origToExpandedPtr: [], origPtrLen: rawPtrLen,
                slotTargets: slotTargets
            )
        }
        guard loadVision, let vb = visionBackend, vb.isVisionEnabled else {
            throw QWDecisionError.backendFailed("Vision tower is disabled (--no-load-vision) or unavailable for this model")
        }
        try vb.prepareForPrompt(hasImages: true)
        let visCap = decisionConfig.visionMaxPixels ?? 1_638_400
        let spec = QWMMSpec(visionConfig: vb.visionConfig, minPixels: 65_536, maxPixels: 16_777_216)
        var encodedImages: [(grid: QWVisionGrid, features: [Float])] = []
        encodedImages.reserveCapacity(seq.images.count)
        for imgData in seq.images {
            if let cached = cache[imgData] {
                encodedImages.append(cached)
            } else {
                let encoded = try await Self.encodeImage(imgData, capPixelsBudget: visCap, spec: spec, backend: vb)
                cache[imgData] = encoded
                encodedImages.append(encoded)
            }
        }
        return try Self.expandMultimodalSequence(
            seq: seq, entries: entries, encodedImages: encodedImages, spec: spec, hidden: descriptor.hidden, rawPtrLen: rawPtrLen
        )
    }

    private static func expandMultimodalSequence(
        seq: QWDecisionSequenceInput,
        entries: [(Int, QWDecisionSlotTarget)],
        encodedImages: [(grid: QWVisionGrid, features: [Float])],
        spec: QWMMSpec,
        hidden: Int,
        rawPtrLen: Int
    ) throws -> PreparedDecisionSeq {
        let placeholders = seq.tokenIds.indices.filter { seq.tokenIds[$0] == spec.imageToken }
        guard placeholders.count == encodedImages.count else {
            throw QWDecisionError.invalidQuestion("expected \(encodedImages.count) image placeholders, found \(placeholders.count)")
        }
        let phSet = Dictionary(uniqueKeysWithValues: placeholders.enumerated().map { ($1, $0) })
        var expandedTokens: [Int] = []
        var visualSpans: [QWVisualSpan] = []
        var origToExp = [Int?](repeating: nil, count: seq.tokenIds.count + 1)

        for i in 0..<seq.tokenIds.count {
            origToExp[i] = expandedTokens.count
            if let imgIdx = phSet[i] {
                let startTok = expandedTokens.count
                let g = encodedImages[imgIdx].grid
                let nTok = g.tokens
                for _ in 0..<nTok { expandedTokens.append(spec.imageToken) }
                visualSpans.append(QWVisualSpan(
                    tokenOffset: startTok, tokenCount: nTok, embeddings: encodedImages[imgIdx].features,
                    gridT: g.t, gridH: g.h, gridW: g.w, spatialMergeSize: spec.merge
                ))
            } else {
                expandedTokens.append(seq.tokenIds[i])
            }
        }
        origToExp[seq.tokenIds.count] = expandedTokens.count
        _ = hidden

        func mapBoundary(_ origPos: Int) -> Int {
            let clamped = max(0, min(seq.tokenIds.count, origPos))
            return origToExp[clamped] ?? expandedTokens.count
        }
        let shiftedStateEnd = mapBoundary(seq.stateEndPosition)
        let shiftedQSpans = seq.questionSpans.map { mapBoundary($0.lowerBound)..<mapBoundary($0.upperBound) }
        let shiftedMarkers = seq.markerPositions.map { mapBoundary($0) }
        let shiftedSlots = entries.map { (idx, target) -> (Int, Int, Int?, Bool) in
            let p = mapBoundary(target.tokenPosition)
            let pe = target.pointerEndPosition.map { mapBoundary($0) }
            return (idx, p, pe, target.isPointerSlot)
        }
        let clampedPtrLen = min(rawPtrLen, seq.tokenIds.count)
        let expPtrLen = mapBoundary(clampedPtrLen)
        let ptrMap: [Int?] = (0..<clampedPtrLen).map { origIdx in
            if !seq.pointerValidMask.isEmpty && !seq.pointerValidMask[origIdx] { return nil }
            return origToExp[origIdx]
        }
        return PreparedDecisionSeq(
            tokens: expandedTokens, visualSpans: visualSpans, encodedImages: encodedImages,
            topology: seq.topology, stateEndPos: shiftedStateEnd, questionSpans: shiftedQSpans,
            markerPositions: shiftedMarkers, chain: seq.chain,
            expandedStateLen: expPtrLen, origToExpandedPtr: ptrMap, origPtrLen: clampedPtrLen,
            slotTargets: shiftedSlots
        )
    }

    private static func encodeImage(
        _ data: Data, capPixelsBudget: Int, spec: QWMMSpec, backend: any QWVisionBackend
    ) async throws -> (grid: QWVisionGrid, features: [Float]) {
        let header: QWImageHeaderInfo
        let source: CGImageSource
        do {
            (source, header) = try qwInspectImageHeader(
                data: data,
                maxSourceBytes: maxImageSourceBytes,
                maxDimension: maxImageDimension,
                maxArea: maxImageArea
            )
        } catch let err as QWOrientedImageError {
            throw QWDecisionError.invalidQuestion(err.description)
        }
        // Apply d1 `cap_pixels` before `smartResize(..., maxPixels: 16_777_216)` matching HF processor.
        var capW = header.width
        var capH = header.height
        if Double(header.width) * Double(header.height) > Double(capPixelsBudget) {
            let scale = (Double(capPixelsBudget) / (Double(header.width) * Double(header.height))).squareRoot()
            capW = max(1, Int(Double(header.width) * scale))
            capH = max(1, Int(Double(header.height) * scale))
        }
        let resized = try QWMMConfig.smartResize(height: capH, width: capW, maxPixels: spec.maxPixels, spec: spec)
        let grid = try QWVisionGrid(t: 1, h: resized.h / spec.patch, w: resized.w / spec.patch, spec: spec)
        guard let full = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw QWDecisionError.invalidQuestion("Failed to decode CGImage")
        }
        let oriented: QWOrientedRGBImage
        do {
            oriented = try qwRenderOrientedRGBSlab(
                cgImage: full, targetWidth: resized.w, targetHeight: resized.h, orientation: header.orientation
            )
        } catch let err as QWOrientedImageError {
            throw QWDecisionError.backendFailed(err.description)
        }
        let patches = try qwBuildPatchSlab(imageRGB: oriented.rgb, height: resized.h, width: resized.w, grid: grid)
        let features = try await backend.encode(
            patches: patches, grid: grid.plan, bins: grid.interpBins, posIds: grid.ropePositions
        )
        return (grid, features)
    }
}
