// QWEngineSeamsTests.swift — Unit tests for QWTrunkExecutor seams, DFlash sizing, and draft-head slicing.
import Foundation
import Metal
import XCTest
import QWAlloc
import QWCore
import QWHAL
import QWHead
import QWMTP
import QWQuant
@testable import QWEngine

private final class StubTrunkExecutor: QWTrunkExecutor, @unchecked Sendable {
    var hasPrefixCacheTier: Bool = true
    var arenaBytes: Int = 0
    var weightReservedBytes: Int = 0
    var weightUsedBytes: Int = 0
    var activeCapacity: Int = 1024
    var steadyStateAllocations: Int = 0
    var dflashDrafter: QWDFlashDrafter? = nil
    var prefixCacheState: QWPrefixCacheState = QWPrefixCacheState(isConfigured: false)
    var lastResult: QWGenerationResult? = nil
    var onCacheEvent: (@Sendable (QWPrefixCacheEvent) -> Void)? = nil
    var gpuTurnArbitrator: QWGPUTurnArbitrator? = nil
    var cachedEntries: [(tokens: [Int], salt: String?)] = []

    func hasCachedPrefix(tokens: [Int], cacheSalt: String?) -> Bool {
        cachedEntries.contains { $0.tokens == tokens && $0.salt == cacheSalt }
    }

    func buildCapabilities() -> Set<QWKnobCapability> { [] }

    func buildReport() -> QWEngineReport {
        QWEngineReport(
            capacityLog: "", arenaBytes: 0, weightUsedBytes: 0, weightReservedBytes: 0,
            steadyAllocations: 0, pipelineSeconds: 0, detailLines: "", weightFormat: "HQ4_G32"
        )
    }

    func runGeneration(
        request _: QWGenerationRequest,
        onToken _: ((Int) -> Bool)?,
        onStopToken _: ((Int) -> Void)?
    ) throws -> QWGenerationResult {
        QWGenerationResult(tokens: [], stepMs: [], stepTokenCounts: [])
    }

    func configurePrefixCache(_: QWCacheConfig) throws {}

    func buildStandalonePrefixCache(
        tokens _: [Int], outputURL _: URL, pinAsSharedBase _: Bool,
        onPrefillProgress _: (@Sendable (Int, Int) -> Void)?
    ) throws -> Int { 0 }

    func mountStandalonePrefixCache(from _: URL) throws -> (tokens: [Int], byteCount: Int) { ([], 0) }
    func unmountSharedBasePrefixes() {}

    func computeEchoPromptLogprobs(
        promptTokens _: [Int], sampling _: QWSamplerConfig,
        topLogprobsCount _: Int, shouldStop _: (@Sendable () -> Bool)?
    ) throws -> [QWTokenLogprob] { [] }
}

final class QWEngineSeamsTests: XCTestCase {
    private static var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    func testTrunkExecutorProtocolForwardsArbitratorAndPrefixLookup() {
        let stub = StubTrunkExecutor()
        XCTAssertEqual(stub.maxBatchSlots, 1)
        XCTAssertNil(stub.gpuTurnArbitrator)

        let arb = QWGPUTurnArbitrator()
        let trunk: any QWTrunkExecutor = stub
        trunk.gpuTurnArbitrator = arb
        XCTAssertTrue(stub.gpuTurnArbitrator === arb)

        stub.cachedEntries.append((tokens: [1, 2, 3], salt: "tenant-a"))
        XCTAssertTrue(trunk.hasCachedPrefix(tokens: [1, 2, 3], cacheSalt: "tenant-a"))
        XCTAssertFalse(trunk.hasCachedPrefix(tokens: [1, 2, 3], cacheSalt: nil))
        XCTAssertFalse(trunk.hasCachedPrefix(tokens: [1, 2], cacheSalt: "tenant-a"))
    }

    func testDriverAndRuntimeSourceSeamsWireArbitratorPrefixCacheAndHygiene() throws {
        let driverSrc = try String(
            contentsOf: Self.repoRoot.appendingPathComponent("Sources/Runtime/QWEngine/Driver/QWEngineDriver.swift"),
            encoding: .utf8
        )
        XCTAssertTrue(driverSrc.contains("var trunkExecutor: any QWTrunkExecutor"))
        XCTAssertTrue(driverSrc.contains("trunkExecutor.gpuTurnArbitrator"))

        let confSrc = try String(
            contentsOf: Self.repoRoot.appendingPathComponent(
                "Sources/Runtime/QWEngine/Generation/QWEngineDriverConformance.swift"),
            encoding: .utf8
        )
        XCTAssertTrue(confSrc.contains("trunkExecutor.hasCachedPrefix(tokens: tokens, cacheSalt: cacheSalt)"))
        XCTAssertFalse(confSrc.contains("if let trunk = trunkExecutor"))

        let schedSrc = try String(
            contentsOf: Self.repoRoot.appendingPathComponent("Sources/Runtime/QWEngine/Core/QWBatchScheduler.swift"),
            encoding: .utf8
        )
        XCTAssertFalse(schedSrc.contains("BatchScratchBox"))
        XCTAssertFalse(schedSrc.contains("ensureHeadScratch"))

        let decSrc = try String(
            contentsOf: Self.repoRoot.appendingPathComponent("Sources/Runtime/QWEngine/Driver/DriverDecisionEval.swift"),
            encoding: .utf8
        )
        XCTAssertFalse(decSrc.contains("if let trunk = trunkExecutor"))

        let cacheSrc = try String(
            contentsOf: Self.repoRoot.appendingPathComponent("Sources/Runtime/QWEngine/Cache/PrefixCache.swift"),
            encoding: .utf8
        )
        XCTAssertFalse(cacheSrc.contains("if let trunk = trunkExecutor"))

        let oldPath = Self.repoRoot.appendingPathComponent(
            "Sources/Runtime/QWEngine/Driver/DriverInitPhases+Ernie55.swift")
        let newPath = Self.repoRoot.appendingPathComponent(
            "Sources/Runtime/QWEngine/Driver/DriverInitPhases+ResidencyAndLSA.swift")
        XCTAssertFalse(FileManager.default.fileExists(atPath: oldPath.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: newPath.path))

        let inkWeights = try String(
            contentsOf: Self.repoRoot.appendingPathComponent(
                "Sources/Runtime/QWEngine/Driver/Inkling2/QWInkling2Weights.swift"),
            encoding: .utf8
        )
        let inkQWP = try String(
            contentsOf: Self.repoRoot.appendingPathComponent(
                "Sources/Runtime/QWEngine/Driver/Inkling2/QWInkling2Weights+QWP.swift"),
            encoding: .utf8
        )
        XCTAssertTrue(inkWeights.contains("visual."))
        XCTAssertTrue(inkQWP.contains("visual."))
    }

    func testFNGPUDFlashSizesSafeTensorsDrafterWeights() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-dflash-st-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let tensorBytes = 64 * 64 * 2
        let hdrObj: [String: Any] = [
            "fc.weight": [
                "dtype": "BF16",
                "shape": [64, 64],
                "data_offsets": [0, tensorBytes],
            ],
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
        try Data("{\"vocab_size\": 1024}".utf8).write(to: dir.appendingPathComponent("config.json"))

        let st = try QWSafeTensorsContainer(url: dir.appendingPathComponent("model.safetensors"))
        let expected = QWDFlashDrafter.computeSafeTensorsWeightCapacity(st: st)
        let actual = FNGPUDFlash.drafterWeightBytes(packPath: dir.path, isDSpine: false)
        XCTAssertEqual(actual, expected)
    }

    func testMTPTDraftHeadSupportsVocabRowsSlicingForSingleAndBatched() throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "Metal device required")
        let hal = try QWHal(geometry: .qwen38_27b)
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-draft-head-slice-\(UUID().uuidString)").path
        defer { try? FileManager.default.removeItem(atPath: path) }

        let n = QWModelGeometry.qwen38_27b.vocab
        let k = QWModelGeometry.qwen38_27b.hidden
        let blockBytes = n * (k / 32) * 20
        let hq4Off = 256
        let payloadEnd = ((hq4Off + blockBytes + 255) & ~255)
        let header = Data("""
        {"version":2,"provenance":{"quant":"hq4_g32_m5"},"tensors":{"\(QWMTPTDraftHead.headName)":\
        {"shape":[\(n),\(k)],"dtype":"hq4_g32_m5","hq4_off":\(hq4Off),"hq4_len":\(blockBytes)}}}
        """.utf8)
        FileManager.default.createFile(atPath: path, contents: nil)
        let fh = try XCTUnwrap(FileHandle(forWritingAtPath: path))
        defer { try? fh.close() }
        fh.seek(toFileOffset: 0)
        fh.write(Data("QWP1".utf8))
        var lenLE = UInt64(header.count).littleEndian
        withUnsafeBytes(of: &lenLE) { (raw: UnsafeRawBufferPointer) in fh.write(Data(raw)) }
        fh.seek(toFileOffset: UInt64(payloadEnd))
        fh.write(header)
        fh.truncateFile(atOffset: UInt64(payloadEnd + header.count))

        let pack = try QWQuant.Pack(url: URL(fileURLWithPath: path), descriptor: .qwen38_27b, skipIdentityCheck: true)
        let head = try QWMTPTDraftHead(geometry: QWTestGeometry.g27, hal: hal, sidecar: pack)

        let vocabRows = 98_304
        let m = 2
        let arena = try QWArena(hal: hal, weights: 1 << 16, activations: 8 << 20, kv: 1 << 12, gdnStates: 1 << 12)
        let x1 = try arena.activationsRegion().allocate(QWTensorShape([k], .f16))
        let y1 = try arena.activationsRegion().allocate(QWTensorShape([vocabRows], .f32))
        let x2 = try arena.activationsRegion().allocate(QWTensorShape([m * k], .f16))
        let y2 = try arena.activationsRegion().allocate(QWTensorShape([m * vocabRows], .f32))

        let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
        XCTAssertThrowsError(try head.encode(enc, hal: hal, x: x1, y: y1, vocabRows: 0))
        XCTAssertThrowsError(try head.encodeBatched(enc, hal: hal, x: x2, y: y2, m: m, vocabRows: 0))
        try head.encode(enc, hal: hal, x: x1, y: y1, vocabRows: vocabRows)
        try head.encodeBatched(enc, hal: hal, x: x2, y: y2, m: m, vocabRows: vocabRows)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        XCTAssertEqual(cb.status, .completed)
    }
}
