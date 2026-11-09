import XCTest
import Foundation
import QWCore
@testable import QWQuant

/// Flat-root resolution: a model folder may hold `model.qwp`, `*.safetensors`,
/// `config.json` and `ngram.bin` side by side, with no `assets/` subfolder.
final class QWFlatRootLocationTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("qwg-flat-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    private func write(_ name: String, _ text: String = "{}") throws {
        try Data(text.utf8).write(to: root.appendingPathComponent(name))
    }

    // MARK: - The weight container

    func testAutoPrefersThePackedQWPWhenBothContainersExist() throws {
        try write("model.qwp", "QWP1")
        try write("model.safetensors")
        try write("config.json")
        let root = QWModelRoot(url: self.root, preference: .auto)
        XCTAssertEqual(root.kind, .qwp1)
        XCTAssertEqual(root.weights.lastPathComponent, "model.qwp")
    }

    func testAutoFallsBackToSingleFileSafetensors() throws {
        try write("model.safetensors")
        try write("config.json")
        let root = QWModelRoot(url: self.root, preference: .auto)
        XCTAssertEqual(root.kind, .safetensorsSingle)
        XCTAssertEqual(root.weights.lastPathComponent, "model.safetensors")
    }

    func testAutoFallsBackToShardedSafetensors() throws {
        try write("model.safetensors.index.json")
        try write("model-00001-of-00002.safetensors")
        try write("config.json")
        let root = QWModelRoot(url: self.root, preference: .auto)
        XCTAssertEqual(root.kind, .safetensorsSharded)
    }

    func testSafetensorsPreferenceIgnoresThePackedQWP() throws {
        try write("model.qwp", "QWP1")
        try write("model.safetensors")
        try write("config.json")
        let root = QWModelRoot(url: self.root, preference: .safetensors)
        XCTAssertEqual(root.kind, .safetensorsSingle)
    }

    func testQWPPreferenceReportsNoContainerWhenOnlySafetensorsExist() throws {
        try write("model.safetensors")
        try write("config.json")
        let root = QWModelRoot(url: self.root, preference: .qwp)
        XCTAssertNil(root.kind)
    }

    func testAFileArgumentIsTheWeightsItself() throws {
        try write("model.safetensors")
        let file = root.appendingPathComponent("model.safetensors")
        let resolved = QWModelRoot(url: file, preference: .auto)
        XCTAssertEqual(resolved.kind, .safetensorsSingle)
        XCTAssertEqual(resolved.weights.lastPathComponent, "model.safetensors")
    }

    // MARK: - The assets directory

    func testFlatRootPutsAssetsAtTheRootWhenNgramBinIsBesideIt() throws {
        try write("model.safetensors")
        try write("config.json")
        try write("ngram.bin")
        let root = QWModelRoot(url: self.root, preference: .auto)
        XCTAssertEqual(root.assets.standardizedFileURL.path, self.root.standardizedFileURL.path)
    }

    /// A flat root is marked by its config sitting beside the weights.
    func testFlatRootPutsAssetsAtTheRootWhenTheConfigIsBesideIt() throws {
        try write("model.safetensors")
        try write("config.json")
        let root = QWModelRoot(url: self.root, preference: .auto)
        XCTAssertEqual(root.assets.standardizedFileURL.path, self.root.standardizedFileURL.path)
    }

    func testLegacyRootStillResolvesTheAssetsSubfolder() throws {
        try write("model.qwp", "QWP1")
        let assets = root.appendingPathComponent("assets", isDirectory: true)
        try FileManager.default.createDirectory(at: assets, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: assets.appendingPathComponent("config.json"))
        let root = QWModelRoot(url: self.root, preference: .auto)
        XCTAssertEqual(root.assets.lastPathComponent, "assets")
        XCTAssertEqual(root.configURL?.lastPathComponent, "config.json")
    }

    /// The legacy contract: a directory root's assets is `<dir>/assets`,
    /// whether or not that folder exists.
    func testADirectoryRootWithoutAnAssetsFolderStillPointsAtAssets() throws {
        try write("model.qwp", "QWP1")
        let root = QWModelRoot(url: self.root, preference: .auto)
        XCTAssertEqual(root.assets.lastPathComponent, "assets")
        XCTAssertEqual(root.assets.deletingLastPathComponent().standardizedFileURL.path,
                       self.root.standardizedFileURL.path)
    }

    // MARK: - The config

    func testAFlatRootReadsTheConfigBesideTheWeights() throws {
        try write("model.safetensors")
        try write("config.json", #"{"model_type":"llama"}"#)
        let root = QWModelRoot(url: self.root, preference: .auto)
        XCTAssertEqual(root.configURL?.lastPathComponent, "config.json")
        XCTAssertEqual(root.configURL?.deletingLastPathComponent().standardizedFileURL.path,
                       self.root.standardizedFileURL.path)
    }

    /// The config is the descriptor's business, not the root's: a `.qwp`
    /// pack with no `config.json` beside it is still a valid root.
    func testARootWithoutAConfigHasNoConfigURL() {
        try? write("model.safetensors")
        let resolved = QWModelRoot(url: self.root, preference: .auto)
        XCTAssertNil(resolved.configURL)
    }

    /// A folder with only a config is a valid root: the model catalog scans
    /// such folders and must be able to classify them.
    func testAConfigOnlyFolderIsARootWithNoWeightContainer() throws {
        try write("config.json", #"{"model_type":"llama"}"#)
        let root = QWModelRoot(url: self.root, preference: .auto)
        XCTAssertNil(root.kind)
        XCTAssertEqual(root.configURL?.lastPathComponent, "config.json")
    }

    // MARK: - QWModelLocation

    func testModelLocationCarriesTheResolvedKind() throws {
        try write("model.safetensors")
        try write("config.json")
        let loc = try QWModelLocation(url: root)
        XCTAssertEqual(loc.kind, .safetensorsSingle)
    }

    func testModelLocationThrowsWhenThereIsNoContainer() throws {
        try write("config.json")
        XCTAssertThrowsError(try QWModelLocation(url: root)) { error in
            guard case QWQuantError.modelFileMissing = error else {
                return XCTFail("expected modelFileMissing, got \(error)")
            }
        }
    }

    func testModelLocationHonoursTheAssetsOverride() throws {
        try write("model.safetensors")
        try write("config.json")
        let override = root.appendingPathComponent("elsewhere")
        let loc = try QWModelLocation(url: root, assetsOverride: override)
        XCTAssertEqual(loc.assets, override)
    }
}
