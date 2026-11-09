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
        for concrete in ["QWEngineDriver", "FNGPUEngine"] {
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
        }
    }

    /// Both modules conform: the protocol is the engine-neutral surface
    /// every family offers. The proof is compile-time — the call site
    /// does not compile unless the conformance exists.
    func testBothModulesConformToTheProtocol() throws {
        requireConformance(QWEngineDriver.self)
        requireConformance(FlashNextModule.self)
        let dense: Any.Type = QWEngineDriver.self
        let flash: Any.Type = FlashNextModule.self
        XCTAssertTrue(dense is any QWInferenceEngine.Type)
        XCTAssertTrue(flash is any QWInferenceEngine.Type)
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
        let flashLoader = try Self.read("Sources/Subsystems/QWFlashNext/GPU/FNGPUWeights.swift")
        XCTAssertTrue(denseLoader.contains("QWHQ4Quantizer."),
                      "QWEngineDriver must delegate online HQ4 quantization to QWHQ4Quantizer")
        XCTAssertTrue(flashLoader.contains("QWHQ4Quantizer."),
                      "FNGPULoader must delegate online HQ4 quantization to QWHQ4Quantizer")
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
