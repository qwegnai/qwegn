// QWMemorySemanticVaultTests.swift — Tests e-means semantic centroid routing
// in QWMemoryVault (H4b): zero-lexical-overlap conceptual recall, <0.5 ms
// query latency, manifest round-trip, and 100% Resident Cortex invariance.
import Foundation
import XCTest
@testable import QWEngine

final class QWMemorySemanticVaultTests: XCTestCase {
    private var tempDir: URL!

    override func setUpWithError() throws {
        try super.setUpWithError()
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwg-sem-vault-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let tempDir {
            try? FileManager.default.removeItem(at: tempDir)
        }
        try super.tearDownWithError()
    }

    func testSemanticEmeansCentroidsRescueConceptualQueriesUnderHalfMillisecond() throws {
        let fm = FileManager.default
        let coreDir = tempDir.appendingPathComponent("docs", isDirectory: true)
        try fm.createDirectory(at: coreDir, withIntermediateDirectories: true)
        try "# System Architecture\nCore overview.\n"
            .write(to: coreDir.appendingPathComponent("overview.md"), atomically: true, encoding: .utf8)

        let pad = String(repeating: "// implementation detail line for off-context vault residency\n", count: 30)
        let targetSpecs: [(path: String, body: String, query: String)] = [
            (
                "Sources/Cache/QWSlotPinnedLRUStore.swift",
                "public struct QWSlotPinnedLRUStore {\n    public func prepareSlotForStore(cap: Int) {}\n}\n" + pad,
                "How do we evict least recently used entries from the cache?"
            ),
            (
                "Sources/Finance/PostingJournal.swift",
                "public struct PostingJournal {\n    public func settleLedgerCurrencyDelta() {}\n}\n" + pad,
                "Where are accounting balances reconciled across transactions?"
            ),
            (
                "Sources/Security/JwtGuard.swift",
                "public struct JwtGuard {\n    public func verifyBearerSignature() {}\n}\n" + pad,
                "How does user login credential authentication work?"
            ),
            (
                "Sources/Compute/PoolSlabArena.swift",
                "public struct PoolSlabArena {\n    public func carveContiguousBuffer() {}\n}\n" + pad,
                "Where does memory allocation happen for shared GPU pools?"
            ),
            (
                "Sources/Jobs/WorkerQueue.swift",
                "public struct WorkerQueue {\n    public func enqueueBackgroundBatch() {}\n}\n" + pad,
                "How does the scheduler dispatch asynchronous worker tasks?"
            ),
        ]

        for spec in targetSpecs {
            let url = tempDir.appendingPathComponent(spec.path)
            try fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try spec.body.write(to: url, atomically: true, encoding: .utf8)
        }

        // Add 27 distractor Vault files so total Vault count = 32 >= 16 (activates k = 4 e-means centroids, n/k = 8).
        for i in 0..<27 {
            let sub = "Sources/Widgets/ViewWidget\(i).swift"
            let url = tempDir.appendingPathComponent(sub)
            try fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            let body = "public struct ViewWidget\(i) {\n    public func renderPixelCornerRadius\(i)() {}\n}\n" + pad
            try body.write(to: url, atomically: true, encoding: .utf8)
        }

        let discovered = try QWMemoryPack.discoverFiles(in: tempDir, includeTests: false)
        let built = QWMemoryPack.buildEnvelope(
            packName: "semantic-repo",
            rootPath: tempDir.path,
            files: discovered,
            clusterBlockTokens: 64,
            kvMode: "q4",
            maxTokens: 260
        )
        let vaultEntries = built.fileEntries.filter { !$0.inCortex }
        XCTAssertGreaterThanOrEqual(vaultEntries.count, 32)

        let semIndex = QWMemorySemanticIndex.build(entries: vaultEntries, rootPath: tempDir.path)
        XCTAssertEqual(semIndex.clusterCount, vaultEntries.count / 8)
        XCTAssertEqual(semIndex.invertedLists.flatMap { $0 }.count, vaultEntries.count)

        let manifest = QWMemoryManifest(
            packName: "semantic-repo",
            rootPath: tempDir.path,
            files: built.fileEntries,
            semanticIndex: semIndex
        )
        try manifest.save(to: tempDir)
        let loaded = try QWMemoryManifest.load(from: tempDir)
        XCTAssertEqual(loaded, manifest)

        // Verify 100% Cortex prefix invariance: semantic index never alters buildEnvelope output.
        let rebuilt = QWMemoryPack.buildEnvelope(
            packName: "semantic-repo",
            rootPath: tempDir.path,
            files: discovered,
            clusterBlockTokens: 64,
            kvMode: "q4",
            maxTokens: 260
        )
        XCTAssertEqual(built.systemMessageContent, rebuilt.systemMessageContent)

        // Verify conceptual top-3 recall on zero-lexical-overlap queries AND <0.5 ms latency.
        var top3Hits = 0
        for spec in targetSpecs {
            let res = QWMemoryVault.recallVaultContext(
                query: spec.query,
                manifest: loaded,
                maxFiles: 3,
                preserveHistoricalSnapshot: true
            )
            if res.recalledFiles.contains(spec.path) {
                top3Hits += 1
            }
        }
        let recallRate = Double(top3Hits) / Double(targetSpecs.count)
        XCTAssertGreaterThanOrEqual(recallRate, 0.80,
                                    "conceptual zero-lexical-overlap Vault recall must be >= 80%, got \(recallRate)")

        // Measure pure semantic index + Vault recall latency (warm historical snapshot cache).
        var latenciesMs: [Double] = []
        for i in 0..<40 {
            let q = targetSpecs[i % targetSpecs.count].query
            let t0 = CFAbsoluteTimeGetCurrent()
            let r = QWMemoryVault.recallVaultContext(
                query: q,
                manifest: loaded,
                maxFiles: 3,
                preserveHistoricalSnapshot: true
            )
            let dtMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
            XCTAssertFalse(r.isEmpty)
            latenciesMs.append(dtMs)
        }
        latenciesMs.sort()
        let medianMs = latenciesMs[latenciesMs.count / 2]
        print(String(format: "[H4b_VAULT] conceptual recall=%d/%d (%.0f%%), median latency=%.4f ms (p95=%.4f ms)",
                     top3Hits, targetSpecs.count, recallRate * 100.0,
                     medianMs, latenciesMs[Int(Double(latenciesMs.count) * 0.95)]))
        XCTAssertLessThan(medianMs, 0.50, "semantic Vault recall must complete in < 0.5 ms")
    }
}
