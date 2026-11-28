// QWDFlashFootprint.swift — Shared DFlash 2 / DSpine pack resolution,
// weight/scratch footprint estimation, and KV admission sizing helpers.
import Foundation
import Metal
import QWAlloc
import QWCore
import QWGEMM
import QWHAL
import QWHead
import QWLoad
import QWQuant

public enum QWDFlashFootprint {
    public static let defaultMTPPackSavingsBytes = 1_440 << 20
    public static let defaultDFlashWeightBytes = 576 << 20
    public static let defaultDFlashScratchBytes = 112 << 20

    public static func isDFlashName(_ raw: String?) -> Bool {
        guard let lower = raw?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() else { return false }
        return lower == "dflash2" || lower == "dflash" || lower == "dspine"
    }

    public static func adjustedPerTokenKVBytes(_ perTokenBytes: Int, usesDFlash: Bool) -> Int {
        usesDFlash ? (512 + (((max(0, perTokenBytes - 512) * 12) + 12) / 13)) : perTokenBytes
    }

    public static func kvBytesPerTokenPerLayer(config: QWDFlashConfig) -> Int {
        if config.isMLA {
            return (config.kvLoraRank + config.qkRopeHeadDim) * 2
        }
        return 2 * config.numKeyValueHeads * config.headDim * 2
    }

    public static func kvBytesPerToken(config: QWDFlashConfig) -> Int {
        config.numLayers * kvBytesPerTokenPerLayer(config: config)
    }

    public static func maxFittingContext(activeCapacity: Int) -> Int {
        max(0, activeCapacity - 1)
    }

    public static func resolveDrafterPackPath(
        targetDescriptor: QWModelDescriptor?,
        explicitPath: String?,
        targetPackPath: String? = nil,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> String {
        let trimmed = (explicitPath ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty { return trimmed.expandingTildePath }
        let env = (environment["QW_DFLASH_DRAFT_PACK"] ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if !env.isEmpty { return env.expandingTildePath }
        return QWDFlashConfig.defaultPackPath(for: targetDescriptor, targetPackPath: targetPackPath)
    }

    public static func drafterWeightBytes(packPath: String?, isDSpine: Bool) -> Int {
        guard let raw = packPath?.trimmingCharacters(in: .whitespacesAndNewlines),
              !raw.isEmpty else {
            return defaultDFlashWeightBytes
        }
        let url = raw.expandingTildePath.canonicalFileURL
        guard FileManager.default.fileExists(atPath: url.path) else {
            return defaultDFlashWeightBytes
        }
        let root = QWModelRoot(url: url)
        if root.kind == .safetensorsSingle || root.kind == .safetensorsSharded,
           let st = try? QWSafeTensorsContainer(url: root.weights) {
            return QWDFlashDrafter.computeSafeTensorsWeightCapacity(st: st)
        }
        if let cfgURL = root.configURL,
           let data = try? Data(contentsOf: cfgURL),
           let desc = try? QWModelDescriptor.decodeDrafter(configJSON: data),
           let pack = try? QWQuant.Pack(url: url, descriptor: desc, skipIdentityCheck: true) {
            return QWDFlashDrafter.computeWeightCapacity(pack: pack, isDSpine: isDSpine)
        }
        let fileBytes = (try? FileManager.default.attributesOfItem(atPath: root.weights.path)[.size] as? Int) ?? 0
        return max(64 << 20, fileBytes + (32 << 20))
    }

    public static func estimateScratchBytes(config: QWDFlashConfig, sessionSlotCount: Int = 2) -> Int {
        func aligned(_ bytes: Int) -> Int { ((bytes + 4095) / 4096) * 4096 }
        let h = config.hiddenSize, v = config.vocabSize, i = config.intermediateSize
        let maxCtx = config.maxCacheSize, maxB = max(8, config.blockSize), maxD = maxB - 1
        let hasConv = config.convKernelSize > 0
        let convProj = hasConv ? (2 * config.convKernelSize * (h / max(1, config.convGroupSize))) : 0
        let slots = max(1, sessionSlotCount)
        var total = aligned(maxCtx * config.targetLayerIds.count * h * 2) + aligned(maxCtx * h * 2)
        total += aligned(maxB * h * 4) * 2 + aligned(maxB * h * 2) * 3
        total += aligned(maxB * 4) * 2 + (hasConv ? aligned(maxB * convProj * 4) : 0)
        total += aligned(maxB * i * 2) + aligned(max(maxD * v * 4, maxB * 2 * i * 2))
        total += aligned(maxD * 256 * 16 * 4) * 2 + aligned(maxD * 16 * 4) * 2
        total += aligned(maxCtx * kvBytesPerTokenPerLayer(config: config))
        let slotKVFloor = maxCtx * kvBytesPerToken(config: config)
        if config.isMLA {
            let qHeads = config.numAttentionHeads
            let qLora = config.qLoraRank
            let kvLora = config.kvLoraRank
            let nopeDim = config.qkNopeHeadDim
            let ropeDim = config.qkRopeHeadDim
            let vDim = config.vHeadDim
            let gateDim = config.gatedAttention ? (qHeads * vDim) : 0
            total += aligned(maxB * (qLora + kvLora + ropeDim + gateDim) * 2)
            total += aligned(maxB * qHeads * vDim * 2)
            total += aligned(maxB * qLora * 2)
            total += aligned(maxB * qHeads * (nopeDim + ropeDim) * 2)
            total += aligned(maxB * kvLora * 2) + aligned(maxB * ropeDim * 2)
            total += aligned(maxB * qHeads * kvLora * 2) * 2
            let alignedSlotKV = aligned(config.numLayers * maxCtx * kvLora * 2)
                + aligned(config.numLayers * maxCtx * ropeDim * 2)
            total += slots * max(slotKVFloor, alignedSlotKV)
        } else {
            let qDim = config.numAttentionHeads * config.headDim
            let kvDim = config.numKeyValueHeads * config.headDim
            total += aligned(maxB * (qDim + 2 * kvDim) * 2) + aligned(maxB * qDim * 2)
            let alignedSlotKV = 2 * aligned(config.numLayers * maxCtx * kvDim * 2)
            total += slots * max(slotKVFloor, alignedSlotKV)
        }
        if config.isDSpine {
            let bc = config.bridgeCodeSize, br = config.bridgeRank
            total += aligned(maxB * 4) + aligned(maxD * bc * 4) + aligned(maxB * bc * 2)
            total += aligned(maxB * 3 * br * 4) + aligned(maxD * br * 2) + aligned(maxD * h * 4)
            total += aligned(maxD * 16 * br * 2) + aligned(maxD * 16 * h * 4)
            total += aligned(maxD * 16 * h * 2) * 2 + aligned(maxD * 16 * 16 * 4)
        } else {
            total += aligned(maxD * config.selectorRank * 4)
        }
        return max(32 << 20, total + (16 << 20))
    }

    public static func estimateFootprint(
        targetDescriptor: QWModelDescriptor,
        targetPackPath: String?,
        dflashPackPath: String?,
        isDFlash: Bool,
        sessionSlotCount: Int = 2
    ) -> (nativeMTPWeightBytes: Int, dflashWeightBytes: Int, dflashScratchBytes: Int) {
        guard isDFlash else { return (0, 0, 0) }
        var nativeMTP = defaultMTPPackSavingsBytes
        if let tPath = targetPackPath?.trimmingCharacters(in: .whitespacesAndNewlines), !tPath.isEmpty {
            let tURL = tPath.expandingTildePath.canonicalFileURL
            if FileManager.default.fileExists(atPath: tURL.path),
               let pack = try? QWQuant.Pack(url: tURL, descriptor: targetDescriptor, skipIdentityCheck: true) {
                let summed = pack.tensors.reduce(0) { acc, kv in
                    guard kv.key.hasPrefix("mtp.") else { return acc }
                    return acc + (PlaneLayout.resolve(kv.value)?.byteCount(kv.value)
                        ?? ((kv.value.hq8PayloadBytes ?? 0) + (kv.value.hq8ScaleBytes ?? 0)))
                }
                if summed > 0 { nativeMTP = summed }
            }
        }
        let resolved = resolveDrafterPackPath(
            targetDescriptor: targetDescriptor, explicitPath: dflashPackPath, targetPackPath: targetPackPath)
        guard !resolved.isEmpty, FileManager.default.fileExists(atPath: resolved.canonicalFileURL.path) else {
            return (nativeMTP, defaultDFlashWeightBytes, defaultDFlashScratchBytes)
        }
        let cfg = QWDFlashConfig.load(from: resolved.canonicalFileURL, target: targetDescriptor)
        return (
            nativeMTP,
            drafterWeightBytes(packPath: resolved, isDSpine: cfg.isDSpine),
            estimateScratchBytes(config: cfg, sessionSlotCount: sessionSlotCount)
        )
    }
}

extension QWDFlashDrafter {
    public static func encodeSlicedHeadLogits(
        _ enc: any MTLComputeCommandEncoder,
        hal: QWHal,
        headStorage: QWHeadStorage,
        geometry: QWModelGeometry,
        xRot16: QWTensorValue,
        xNorm16: QWTensorValue,
        xRot32: QWTensorValue? = nil,
        yLogits32: QWTensorValue,
        rows: Int,
        vocabRows: Int
    ) throws {
        guard vocabRows > 0 else {
            throw QWDFlashError.state("QWDFlashDrafter: invalid vocabRows=\(vocabRows)")
        }
        switch headStorage {
        case .q4g(let hw):
            let w: QWWeightView
            if vocabRows < hw.view.n {
                let sliced = hw.view.blocks.slice(
                    byteOffset: 0, count: (vocabRows * hw.view.rowBytes) / 2, dtype: .f16
                )
                guard let sub = hw.view.subview(blocks: sliced, n: vocabRows, k: hw.view.k) else {
                    throw QWDFlashError.state("QWDFlashDrafter: failed to slice lm_head to vocabRows=\(vocabRows)")
                }
                w = sub
            } else {
                w = hw.view
            }
            if rows >= 2 {
                let x16 = w.isRotated ? xRot16 : xNorm16
                try QWHead.encodeBatched(enc, hal: hal, view: w, x: x16, y: yLogits32, m: rows, preferNAX: true)
            } else if w.isRotated, let xRot32 {
                try QWGemm.encodeHQ4F32V4(enc, hal: hal, weight: w, x: xRot32, y: yLogits32, rows: rows)
            } else {
                guard let subHead = QWHeadWeight(
                    blocks: w.blocks, n: w.n, k: w.k, bits: w.bits, groupSize: w.groupSize,
                    isRotated: w.isRotated, isIncumbentContainer: w.isIncumbentContainer, isSHVQ: w.isSHVQ
                ) else {
                    throw QWDFlashError.state("QWDFlashDrafter: invalid head weight for vocabRows=\(w.n)")
                }
                try QWHead.encodeQ4g(enc, hal: hal, weight: subHead, x: w.isRotated ? xRot16 : xNorm16, y: yLogits32)
            }
        case .f16(let w):
            let slicedW = vocabRows < geometry.vocab
                ? w.slice(byteOffset: 0, count: vocabRows * geometry.hidden, dtype: .f16)
                : w
            try QWHead.encodeF16(
                enc, hal: hal, w: slicedW, x: xNorm16, y: yLogits32,
                geometry: geometry, M: rows, N: vocabRows
            )
        case .bf16(let w):
            try QWHead.encodeBF16(
                enc, hal: hal, w: w, x: xNorm16, y: yLogits32,
                geometry: geometry, M: rows, N: vocabRows
            )
        }
    }
}
