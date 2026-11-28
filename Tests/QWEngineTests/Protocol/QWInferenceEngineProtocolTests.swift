// QWInferenceEngineProtocolTests.swift — The public engine protocol
// names no concrete engine, and the server dispatches on capability.
//
// The protocol is the engine-neutral surface: a consumer that holds
// `any QWInferenceEngine` holds an engine, not a family. The server
// asks for a generation and reads the result; it never decides which
// path the engine takes, and it never downcasts to one engine.

import XCTest
import QWCore
@testable import QWEngine

/// A compile-time conformance proof: this call site does not compile
/// unless the type satisfies the protocol.
private func requireConformance<T: QWInferenceEngine>(_ type: T.Type) {}

final class QWInferenceEngineProtocolTests: XCTestCase {
    /// The protocol names no concrete engine: a consumer that holds
    /// `any QWInferenceEngine` holds an engine, not a family. The guard
    /// is the bare type name, so a leak in any signature is caught.
    func testTheProtocolNamesNoConcreteEngine() throws {
        let source = try Self.read("Sources/Runtime/QWEngine/Generation/QWInferenceEngine.swift")
        for concrete in ["QWEngineDriver", "QWHyperStreamRuntime", "FNGPUEngine"] {
            XCTAssertFalse(source.contains(concrete),
                           "the public protocol must not name \(concrete)")
        }
    }

    /// The server names no dense engine: the module builds and
    /// configures the one it serves, so the server never says which
    /// engine it got.
    func testTheServerNamesNoDenseEngine() throws {
        let root = try Self.packageRoot()
        for path in try Self.swiftFiles(under: root.appendingPathComponent("Sources/Apps/QWServer")) {
            let source = try String(contentsOf: path, encoding: .utf8)
            XCTAssertFalse(source.contains("QWEngineDriver"),
                           "\(path.lastPathComponent) names the dense engine")
            XCTAssertFalse(source.contains("QWHyperStreamRuntime"),
                           "\(path.lastPathComponent) names the HyperStream runtime")
        }
    }

    /// Unified engine and module conform: the protocol is the engine-neutral surface
    /// every family offers. The proof is compile-time — the call site
    /// does not compile unless the conformance exists.
    func testBothModulesConformToTheProtocol() throws {
        requireConformance(QWEngineDriver.self)
        requireConformance(DenseModule.self)
        let dense: Any.Type = QWEngineDriver.self
        let module: Any.Type = DenseModule.self
        XCTAssertTrue(dense is any QWInferenceEngine.Type)
        XCTAssertTrue(module is any QWInferenceEngine.Type)
        let root = try Self.packageRoot()
        let packageSwift = try Self.read("Package.swift")
        XCTAssertFalse(packageSwift.contains("\"QWFlashNext\""),
                       "QWFlashNext target must be retired from Package.swift")
        for path in try Self.swiftFiles(under: root.appendingPathComponent("Sources")) {
            let source = try String(contentsOf: path, encoding: .utf8)
            XCTAssertFalse(source.contains("FlashNextModule"),
                           "\(path.lastPathComponent) must not retain retired FlashNextModule")
            XCTAssertFalse(source.contains("FNGPUEngine"),
                           "\(path.lastPathComponent) must not retain retired FNGPUEngine")
        }
        let batchSched = try Self.read("Sources/Runtime/QWEngine/Core/QWBatchScheduler.swift")
        XCTAssertTrue(batchSched.contains("public final class QWBatchScheduler"),
                      "single unified QWBatchScheduler must live in Core/QWBatchScheduler.swift")
        XCTAssertFalse(batchSched.contains("import QWFlashNext"))
        for unifiedFile in [
            "Sources/Runtime/QWEngine/Driver/QWEngineDriver.swift",
            "Sources/Runtime/QWEngine/Generation/QWEngineDriverConformance.swift",
            "Sources/Runtime/QWEngine/Core/QWBatchScheduler.swift",
            "Sources/Runtime/QWEngine/Cache/PrefixCache.swift",
            "Sources/Runtime/QWEngine/Session/DriverSessions.swift",
            "Sources/Runtime/QWEngine/Driver/DriverBatchDecode.swift",
            "Sources/Runtime/QWEngine/Driver/DriverDecisionEval.swift",
            "Sources/Runtime/QWEngine/Diagnostics/DriverDiagnostics.swift",
        ] {
            let content = try Self.read(unifiedFile)
            XCTAssertFalse(content.contains("hyperRuntime"), "\(unifiedFile) must not branch on hyperRuntime")
            XCTAssertFalse(content.contains("inkling2Runtime"), "\(unifiedFile) must not branch on inkling2Runtime")
        }
    }

    func testModulePolicyIsDispatchedThroughTheProtocol() throws {
        let module: any QWInferenceModule.Type = ToyModule.self
        let resolved = try module.resolveSpeculation(
            QWSpeculativeParameters(mtpDrafts: 5), descriptor: .qwen38_27b,
            environment: [:], modelSwitch: false)
        XCTAssertEqual(resolved.engineDrafts, 2)
        XCTAssertEqual(resolved.reportedBackend, "toy")
        XCTAssertEqual(resolved.reportedDrafts, 2)
    }

    /// The consumer layer reaches into no engine's API and reports with
    /// no engine's error type: the spec's gate is
    /// `rg "QWEngineDriver\.|QWEngineDriverError" Sources/Apps` → 0.
    func testTheConsumerLayerReachesIntoNoEngineAPI() throws {
        let root = try Self.packageRoot()
        for path in try Self.swiftFiles(under: root.appendingPathComponent("Sources/Apps")) {
            let source = try String(contentsOf: path, encoding: .utf8)
            XCTAssertFalse(source.contains("QWEngineDriver."),
                           "\(path.lastPathComponent) reaches into the dense engine's API")
            XCTAssertFalse(source.contains("QWEngineDriverError"),
                           "\(path.lastPathComponent) reports with the driver's error type")
        }
    }

    /// The types the protocol speaks live at module level, not inside a
    /// concrete engine: no file in the tree names them as the driver's,
    /// and the module-level declarations exist.
    func testTheEngineTypesAreNotNestedInTheConcreteEngine() throws {
        let root = try Self.packageRoot()
        for path in try Self.swiftFiles(under: root.appendingPathComponent("Sources")) {
            let source = try String(contentsOf: path, encoding: .utf8)
            for nested in ["QWEngineDriver.PrefixCacheStats", "QWEngineDriver.GenerationStats",
                           "QWEngineDriver.ExperimentalGenerationStats",
                           "QWEngineDriver.PrefixCacheEvent"] {
                XCTAssertFalse(source.contains(nested), "\(nested) lives at module level now")
            }
        }
        let types = try Self.read("Sources/Runtime/QWEngine/Generation/QWEngineTypes.swift")
        for declared in ["public struct QWPrefixCacheStats", "public struct QWGenerationStats",
                         "public struct QWExperimentalGenerationStats",
                         "public struct QWPrefixCacheState", "public struct QWGenerationRequest"] {
            XCTAssertTrue(types.contains(declared), "\(declared) is declared at module level")
        }
    }

    /// Apps import only `QWEngine` (never `QWFlashNext`), the server has one
    /// unified backend over `any QWInferenceEngine`, and both family loaders
    /// share `QWHQ4Quantizer` in `QWLoad`.
    func testAppsAndDriversUseUnifiedEngineAndLoaderDoors() throws {
        let root = try Self.packageRoot()
        for path in try Self.swiftFiles(under: root.appendingPathComponent("Sources/Apps")) {
            let source = try String(contentsOf: path, encoding: .utf8)
            XCTAssertFalse(source.contains("import QWFlashNext"),
                           "\(path.lastPathComponent) must import only QWEngine, not QWFlashNext")
        }
        let serverBackend = try Self.read("Sources/Apps/QWServer/Core/QWModelBackend.swift")
        XCTAssertTrue(serverBackend.contains("public struct QWUnifiedModelBackend: QWModelBackend"),
                      "server backend must use a single unified QWUnifiedModelBackend")
        XCTAssertFalse(serverBackend.contains("struct QWDenseModelBackend"),
                       "per-family server backend adapter must be unified")
        XCTAssertFalse(serverBackend.contains("struct QWFlashNextModelBackend"),
                       "per-family server backend adapter must be unified")

        let denseLoader = try Self.read("Sources/Runtime/QWEngine/Driver/DriverWeightLoading.swift")
        let flashLoader = try Self.read("Sources/Runtime/QWEngine/Driver/HyperStream/FNGPUWeights.swift")
        XCTAssertTrue(denseLoader.contains("QWHQ4Quantizer."),
                      "QWEngineDriver must delegate online HQ4 quantization to QWHQ4Quantizer")
        XCTAssertTrue(flashLoader.contains("QWHQ4Quantizer."),
                      "FNGPULoader must delegate online HQ4 quantization to QWHQ4Quantizer")
    }

    /// HyperStream-internal helpers (`FNGPUDFlash`, `FNGPUBatchScheduler`, `FNPrefixWarmStore`)
    /// must not leak into driver residency, module, speculative, app, or lower-layer sources.
    func testHyperStreamInternalTypesDoNotLeakOutsideHyperStream() throws {
        let root = try Self.packageRoot()
        let sourcesRoot = root.appendingPathComponent("Sources")
        let hyperDir = sourcesRoot.appendingPathComponent("Runtime/QWEngine/Driver/HyperStream").path
        for path in try Self.swiftFiles(under: sourcesRoot) where !path.path.hasPrefix(hyperDir + "/") {
            let source = try String(contentsOf: path, encoding: .utf8)
            for forbidden in ["FNGPUDFlash", "FNGPUBatchScheduler", "FNPrefixWarmStore"] {
                XCTAssertFalse(source.contains(forbidden),
                               "\(path.path) must not reference \(forbidden) outside HyperStream/")
            }
        }
    }

    func testResolveSpeculationTrimsBackendReportsZeroWithoutMTPAndPreservesZeroDrafts() throws {
        // Standard path trims mtpBackend whitespace before backend resolution and reporting.
        let std = try DenseModule.resolveSpeculation(
            QWSpeculativeParameters(mtpDrafts: 4, mtpBackend: "  dflash2 \n"),
            descriptor: .qwen38_27b, environment: [:], modelSwitch: false
        )
        XCTAssertEqual(std.engineBackend, .dflash2)
        XCTAssertEqual(std.engineDrafts, 4)
        XCTAssertEqual(std.reportedBackend, "dflash2")

        // HyperStream non-DFlash without .mtp capability reports 0 drafts.
        let hyperNoMTP = QWModelDescriptor(
            family: .qwen38FlashNext, modelType: "qwen4_exp", capabilities: [],
            hidden: 256, vocab: 512, denseIntermediate: nil, numLayers: 4,
            layerTypes: ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            fullAttentionInterval: 4, rmsEps: 1e-6, maxPositionEmbeddings: 4096,
            qHeads: 4, kvHeads: 2, headDim: 64, attnOutputGate: false,
            partialRotaryFactor: 0.25, ropeTheta: 1e7,
            indexerHeads: nil, indexerKvHeads: nil, indexerHeadDim: nil,
            indexerBudget: nil, indexerCompressRatio: nil,
            numExperts: 4, expertsPerTok: 2, moeIntermediate: 128,
            sharedExpertIntermediate: 128, expertGateType: nil,
            hcCount: 4, hcLowrank: 64, gdnQKHeads: 2, gdnVHeads: 4, gdnHeadDim: 64,
            gdnQKVChannels: 512, gdnZChannels: 256, gdnConvKernel: 4,
            pleLayerIds: nil, pleEmbedDim: nil, pleConvKernel: nil,
            ngramSize: nil, headsPerNgram: nil, ngramVocabSizeBase: nil,
            makeNgramVocabDivisibleBy: nil, splitNgramParts: nil,
            mtpLayers: 0, mtpUseDedicatedEmbeddings: false, eosTokenId: 255
        )
        let noMtpRes = try DenseModule.resolveSpeculation(
            QWSpeculativeParameters(mtpDrafts: 3, mtpBackend: "variational"),
            descriptor: hyperNoMTP, environment: [:], modelSwitch: false
        )
        XCTAssertEqual(noMtpRes.engineDrafts, 0)
        XCTAssertEqual(noMtpRes.reportedDrafts, 0)

        // HyperStream modelSwitch compatible DFlash branch preserves explicit mtpDrafts == 0.
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-spec-zero-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir.appendingPathComponent("assets"), withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let cfg: [String: Any] = [
            "model_type": "qwen3", "hidden_size": 256, "num_hidden_layers": 1, "vocab_size": 512,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 128,
            "dflash_config": ["block_size": 8, "target_layer_ids": [0]]
        ]
        try JSONSerialization.data(withJSONObject: cfg).write(
            to: dir.appendingPathComponent("assets/config.json"))
        let hdrData = try JSONSerialization.data(withJSONObject: [
            "provenance": ["source": "unit-test"],
            "tensors": [
                "fc.weight": ["shape": [256, 256], "hq4_off": 64],
                "hidden_norm.weight": ["shape": [256], "raw_off": 64],
                "norm.weight": ["shape": [256], "raw_off": 64],
                "layers.0.self_attn.q_proj.weight": ["shape": [256, 256], "hq4_off": 64],
            ]
        ])
        var qwp = Data("QWP1".utf8)
        var len = UInt64(hdrData.count)
        withUnsafeBytes(of: &len) { (raw: UnsafeRawBufferPointer) in qwp.append(contentsOf: raw) }
        qwp.append(Data(count: 64))
        qwp.append(hdrData)
        try qwp.write(to: dir.appendingPathComponent("model.qwp"))

        var zeroReq = QWSpeculativeParameters(mtpDrafts: 0, mtpBackend: "dflash2")
        zeroReq.dflashDraftPackPath = dir.path
        let zeroRes = try DenseModule.resolveSpeculation(
            zeroReq, descriptor: hyperNoMTP, environment: [:], modelSwitch: true
        )
        XCTAssertEqual(zeroRes.parameters.mtpDrafts, 0)
        XCTAssertEqual(zeroRes.engineDrafts, 0)
        XCTAssertEqual(zeroRes.reportedDrafts, 0)
        XCTAssertNil(zeroRes.engineBackend)
        XCTAssertEqual(zeroRes.reportedBackend, "dflash2")
    }

    /// Reads a file relative to the package root: the directory that
    /// holds `Package.swift`.
    static func read(_ relativePath: String) throws -> String {
        let root = try packageRoot()
        return try String(contentsOf: root.appendingPathComponent(relativePath), encoding: .utf8)
    }

    /// The package root: the directory that holds `Package.swift`.
    static func packageRoot() throws -> URL {
        var root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        while root.pathComponents.count > 1 {
            if FileManager.default.fileExists(atPath: root.appendingPathComponent("Package.swift").path) {
                return root
            }
            root.deleteLastPathComponent()
        }
        throw NSError(domain: "QWInferenceEngineProtocolTests", code: 1, userInfo: [
            NSLocalizedDescriptionKey: "the package root was not found above \(#filePath)"
        ])
    }

    /// Every Swift file under a directory, in a stable order.
    static func swiftFiles(under root: URL) throws -> [URL] {
        guard let walker = FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil) else {
            return []
        }
        var files: [URL] = []
        for case let url as URL in walker where url.pathExtension == "swift" {
            files.append(url)
        }
        return files.sorted { $0.path < $1.path }
    }
}
