// QWDFlashFootprintTests.swift — Unit tests for unified QWDFlashFootprint helpers.
import Foundation
import XCTest
@testable import QWMTP
import QWCore
import QWQuant

final class QWDFlashFootprintTests: XCTestCase {
    func testIsDFlashNameAndKVSizingHelpers() {
        XCTAssertTrue(QWDFlashFootprint.isDFlashName("dflash2"))
        XCTAssertTrue(QWDFlashFootprint.isDFlashName(" dflash "))
        XCTAssertTrue(QWDFlashFootprint.isDFlashName("DSPINE\n"))
        XCTAssertFalse(QWDFlashFootprint.isDFlashName("variational"))
        XCTAssertFalse(QWDFlashFootprint.isDFlashName(nil))

        XCTAssertEqual(QWDFlashFootprint.adjustedPerTokenKVBytes(31_296, usesDFlash: false), 31_296)
        XCTAssertEqual(QWDFlashFootprint.adjustedPerTokenKVBytes(31_296, usesDFlash: true), 28_928)
        XCTAssertEqual(QWDFlashFootprint.adjustedPerTokenKVBytes(18_096, usesDFlash: true), 16_744)
        XCTAssertEqual(QWDFlashFootprint.maxFittingContext(activeCapacity: 0), 0)
        XCTAssertEqual(QWDFlashFootprint.maxFittingContext(activeCapacity: 4096), 4095)
    }

    func testResolveDrafterPackPathPrecedenceAndWeightEstimation() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-footprint-st-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let envPath = "/tmp/env-dflash-\(UUID().uuidString)"
        let resolvedExplicit = QWDFlashFootprint.resolveDrafterPackPath(
            targetDescriptor: nil,
            explicitPath: "  \(dir.path)  ",
            environment: ["QW_DFLASH_DRAFT_PACK": envPath]
        )
        XCTAssertEqual(resolvedExplicit, dir.path)

        let resolvedEnv = QWDFlashFootprint.resolveDrafterPackPath(
            targetDescriptor: nil,
            explicitPath: "  ",
            environment: ["QW_DFLASH_DRAFT_PACK": " \(envPath) "]
        )
        XCTAssertEqual(resolvedEnv, envPath)

        XCTAssertEqual(
            QWDFlashFootprint.drafterWeightBytes(packPath: "/tmp/missing-\(UUID().uuidString)", isDSpine: false),
            QWDFlashFootprint.defaultDFlashWeightBytes
        )

        let tensorBytes = 64 * 64 * 2
        let hdrObj: [String: Any] = [
            "fc.weight": ["dtype": "BF16", "shape": [64, 64], "data_offsets": [0, tensorBytes]]
        ]
        let hdrData = try JSONSerialization.data(withJSONObject: hdrObj, options: [.sortedKeys])
        var stData = Data()
        var hdrLen = UInt64(hdrData.count).littleEndian
        withUnsafeBytes(of: &hdrLen) { (raw: UnsafeRawBufferPointer) in
            stData.append(contentsOf: raw)
        }
        stData.append(hdrData)
        stData.append(Data(repeating: 0, count: tensorBytes))
        try stData.write(to: dir.appendingPathComponent("model.safetensors"))
        let cfgObj: [String: Any] = [
            "model_type": "qwen3",
            "hidden_size": 64,
            "num_hidden_layers": 1,
            "vocab_size": 1024,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 128,
            "intermediate_size": 128,
            "dflash_config": ["block_size": 8, "target_layer_ids": [0]]
        ]
        try JSONSerialization.data(withJSONObject: cfgObj).write(to: dir.appendingPathComponent("config.json"))

        let st = try QWSafeTensorsContainer(url: dir.appendingPathComponent("model.safetensors"))
        let expected = QWDFlashDrafter.computeSafeTensorsWeightCapacity(st: st)
        XCTAssertEqual(QWDFlashFootprint.drafterWeightBytes(packPath: dir.path, isDSpine: false), expected)

        let dflashCfg = QWDFlashConfig(
            hiddenSize: 2048, numLayers: 4, numAttentionHeads: 16, numKeyValueHeads: 8,
            headDim: 128, intermediateSize: 4096, vocabSize: 151_936,
            slidingWindow: 2048, blockSize: 8, targetLayerIds: [0, 1, 2, 3]
        )
        let dspineCfg = QWDFlashConfig(
            hiddenSize: 2048, numLayers: 4, numAttentionHeads: 16, numKeyValueHeads: 8,
            headDim: 128, intermediateSize: 4096, vocabSize: 151_936,
            slidingWindow: 4096, blockSize: 16, targetLayerIds: [0, 1, 2, 3], isDSpine: true
        )
        XCTAssertGreaterThan(QWDFlashFootprint.estimateScratchBytes(config: dflashCfg), 32 << 20)
        XCTAssertGreaterThan(
            QWDFlashFootprint.estimateScratchBytes(config: dspineCfg, sessionSlotCount: 4),
            QWDFlashFootprint.estimateScratchBytes(config: dspineCfg, sessionSlotCount: 1)
        )
    }

    func testEstimateFootprintHandlesOffAndMissingPaths() {
        let missing = "/tmp/nonexistent-dflash-\(UUID().uuidString)"
        let off = QWDFlashFootprint.estimateFootprint(
            targetDescriptor: .qwen38_27b, targetPackPath: nil, dflashPackPath: nil, isDFlash: false
        )
        XCTAssertEqual(off.nativeMTPWeightBytes, 0)
        XCTAssertEqual(off.dflashWeightBytes, 0)
        XCTAssertEqual(off.dflashScratchBytes, 0)

        let on = QWDFlashFootprint.estimateFootprint(
            targetDescriptor: .qwen38_27b, targetPackPath: missing, dflashPackPath: missing, isDFlash: true
        )
        XCTAssertEqual(on.nativeMTPWeightBytes, QWDFlashFootprint.defaultMTPPackSavingsBytes)
        XCTAssertEqual(on.dflashWeightBytes, QWDFlashFootprint.defaultDFlashWeightBytes)
        XCTAssertEqual(on.dflashScratchBytes, QWDFlashFootprint.defaultDFlashScratchBytes)
    }

    func testSelectHybridPLDDraftsIntSequenceParity() {
        let recycler = QWTokenRecycler()
        let periodic = [10, 11, 12, 13, 14, 10, 11, 12, 13, 14, 10, 11, 12, 13, 14]
        XCTAssertEqual(
            QWContextCopyDrafter.selectHybridPLDDrafts(
                history: periodic, extraToken: 10, previousToken: 14, pendingToken: 10,
                mode: .mtp, maxDrafts: 7, recycler: recycler),
            []
        )
        XCTAssertEqual(
            QWContextCopyDrafter.selectHybridPLDDrafts(
                history: periodic, extraToken: 10, previousToken: 14, pendingToken: 10,
                mode: .hybrid, maxDrafts: 7, recycler: recycler),
            [11, 12, 13, 14, 10, 11, 12]
        )

        recycler.ingestSequence([500, 501, 502, 503, 504, 505])
        XCTAssertEqual(
            QWContextCopyDrafter.selectHybridPLDDrafts(
                history: [1, 2, 500, 501], extraToken: nil, previousToken: 500, pendingToken: 501,
                mode: .ngram, maxDrafts: 4, recycler: recycler),
            [502, 503, 504, 505]
        )
    }
}
