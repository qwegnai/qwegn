import XCTest
@testable import QWCore

/// The Hugging Face hub cache written by the `hf` CLI is a model source: a runnable repo
/// is listed by its repo name, and a repo that is not a served model is not listed.
final class QWHfHubCatalogTests: XCTestCase {
    private func tempDir() throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("qwhub-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    func testRunnableHubRepoIsListedByRepoNameAndNonModelReposAreSkipped() throws {
        let hub = try tempDir()
        // A supported model in the hf layout: models--<org>--<repo>/snapshots/<rev>/...
        let repo = hub.appendingPathComponent("models--acme--Synthetic-27B")
        let snap = repo.appendingPathComponent("snapshots/abc123/assets", isDirectory: true)
        try FileManager.default.createDirectory(at: snap, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: repo.appendingPathComponent("refs"), withIntermediateDirectories: true)
        try Data("abc123".utf8).write(to: repo.appendingPathComponent("refs/main"))
        try QWStubModelConfig.dense27BJSON().write(to: snap.appendingPathComponent("config.json"))

        // A hub repo with blobs but no snapshot, and one with a snapshot that is not a model.
        try FileManager.default.createDirectory(at: hub.appendingPathComponent("models--acme--blobs-only/blobs"),
                                                withIntermediateDirectories: true)
        let other = hub.appendingPathComponent("models--other--not-a-model/snapshots/x")
        try FileManager.default.createDirectory(at: other, withIntermediateDirectories: true)
        try Data(#"{"model_type":"not_a_real_family"}"#.utf8).write(to: other.appendingPathComponent("config.json"))

        let found = QWModelCatalog.scanHuggingFaceHub(directory: hub.path)
        XCTAssertEqual(found.map(\.name), ["Synthetic-27B"])
        XCTAssertEqual(found.first?.supported, true)
        XCTAssertEqual(found.first?.family, .qwen38Dense)
    }

    func testMissingHubDirectoryYieldsNoModels() {
        let missing = FileManager.default.temporaryDirectory.appendingPathComponent("absent-\(UUID().uuidString)")
        XCTAssertTrue(QWModelCatalog.scanHuggingFaceHub(directory: missing.path).isEmpty)
    }
}
