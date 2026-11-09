// QWMemoryPackTests.swift — Unit tests for QWMemoryPack and QWMemoryManifest.
// Verifies 3-tier file discovery, 512-token superblock packing, live working-tree
// <|memory_delta|> overlays, manifest persistence, and repo auto-discovery.

import Foundation
import XCTest
@testable import QWEngine

final class QWMemoryPackTests: XCTestCase {
    private var tempDir: URL!

    override func setUpWithError() throws {
        try super.setUpWithError()
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwg-memory-pack-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let tempDir {
            try? FileManager.default.removeItem(at: tempDir)
        }
        try super.tearDownWithError()
    }

    func testThreeTierDiscoveryFiltersBinariesAndOrdersTiers() throws {
        let fm = FileManager.default
        let schemaDir = tempDir.appendingPathComponent("schemas", isDirectory: true)
        let srcDir = tempDir.appendingPathComponent("services/core", isDirectory: true)
        let testsDir = tempDir.appendingPathComponent("tests/unit", isDirectory: true)
        try fm.createDirectory(at: schemaDir, withIntermediateDirectories: true)
        try fm.createDirectory(at: srcDir, withIntermediateDirectories: true)
        try fm.createDirectory(at: testsDir, withIntermediateDirectories: true)

        try "# Workspace Overview\n".write(to: tempDir.appendingPathComponent("overview.md"), atomically: true, encoding: .utf8)
        try "CREATE TABLE accounts (id INT);\n".write(to: schemaDir.appendingPathComponent("01-accounts.sql"), atomically: true, encoding: .utf8)
        try "def run_engine(): pass\n".write(to: srcDir.appendingPathComponent("engine.py"), atomically: true, encoding: .utf8)
        try "def test_engine(): pass\n".write(to: testsDir.appendingPathComponent("test_engine.py"), atomically: true, encoding: .utf8)
        try "ignored.txt\n".write(to: tempDir.appendingPathComponent(".qwgignore"), atomically: true, encoding: .utf8)
        try "should not be indexed\n".write(to: tempDir.appendingPathComponent("ignored.txt"), atomically: true, encoding: .utf8)
        try Data([0x00, 0x01, 0x02, 0xFF]).write(to: srcDir.appendingPathComponent("blob.dat"))

        let defaultFiles = try QWMemoryPack.discoverFiles(in: tempDir, includeTests: false)
        let defaultPaths = defaultFiles.map(\.relativePath)
        XCTAssertEqual(defaultPaths, ["overview.md", "schemas/01-accounts.sql", "services/core/engine.py"])
        // Tier reflects universal directory nesting depth (0 = root, 1 = depth 1, 2 = depth 2).
        XCTAssertEqual(defaultFiles.map(\.tier), [0, 1, 2])

        let withTests = try QWMemoryPack.discoverFiles(in: tempDir, includeTests: true)
        XCTAssertEqual(
            withTests.map(\.relativePath),
            ["overview.md", "schemas/01-accounts.sql", "services/core/engine.py", "tests/unit/test_engine.py"]
        )
        XCTAssertEqual(withTests.last?.tier, 2)
    }

    func testEnvelopeBuilderAndLiveDeltaOverlay() throws {
        let srcDir = tempDir.appendingPathComponent("Sources", isDirectory: true)
        try FileManager.default.createDirectory(at: srcDir, withIntermediateDirectories: true)
        let fileA = srcDir.appendingPathComponent("A.swift")
        let fileB = srcDir.appendingPathComponent("B.swift")
        try "let a = 1\n".write(to: fileA, atomically: true, encoding: .utf8)
        try "let b = 2\n".write(to: fileB, atomically: true, encoding: .utf8)

        let discovered = try QWMemoryPack.discoverFiles(in: tempDir, includeTests: false)
        let built = QWMemoryPack.buildEnvelope(
            packName: "demo-repo",
            rootPath: tempDir.path,
            files: discovered,
            clusterBlockTokens: 512,
            kvMode: "q4"
        )
        XCTAssertTrue(built.systemMessageContent.hasPrefix("Memory<|memory|>\n"))
        XCTAssertTrue(built.systemMessageContent.contains("[FILE: Sources/A.swift]"))
        XCTAssertTrue(built.systemMessageContent.contains("</|memory|>Memory\n[QWG Memory Notice:"))
        XCTAssertEqual(built.fileEntries.count, 2)

        let artifact = QWMemoryManifest.CacheArtifact(
            modelName: "Qwen3.8-27B",
            geometryDigest: "deadbeef",
            kvMode: "q4",
            lsaMode: "sidecar",
            tokenCount: 1024,
            cacheFileName: "demo-repo_Qwen3.8-27B_q4.qwcache",
            cacheFileBytes: 4096
        )
        let manifest = QWMemoryManifest(
            packName: "demo-repo",
            rootPath: tempDir.path,
            gitHeadSHA: "1234567",
            clusterBlockTokens: 512,
            includeTests: false,
            files: built.fileEntries,
            caches: [artifact]
        )
        try manifest.save(to: tempDir)

        let loaded = try QWMemoryManifest.load(from: tempDir)
        XCTAssertEqual(loaded, manifest)
        XCTAssertEqual(loaded.matchingCache(modelName: "Qwen3.8-27B", geometryDigest: "deadbeef")?.kvMode, "q4")

        // Unchanged working tree produces empty delta.
        let cleanDelta = QWMemoryPack.computeDelta(for: loaded)
        XCTAssertFalse(cleanDelta.hasChanges)
        XCTAssertTrue(cleanDelta.deltaBlock.isEmpty)

        // Modifying A.swift and deleting B.swift produces a <memory_delta> overlay.
        try "let a = 42 // updated\n".write(to: fileA, atomically: true, encoding: .utf8)
        try FileManager.default.removeItem(at: fileB)

        let dirtyDelta = QWMemoryPack.computeDelta(for: loaded)
        XCTAssertTrue(dirtyDelta.hasChanges)
        XCTAssertEqual(dirtyDelta.modifiedFiles, ["Sources/A.swift"])
        XCTAssertEqual(dirtyDelta.deletedFiles, ["Sources/B.swift"])
        XCTAssertTrue(dirtyDelta.deltaBlock.contains("<|memory_delta|>"))
        XCTAssertTrue(dirtyDelta.deltaBlock.contains("[UPDATED FILE: Sources/A.swift]"))
        XCTAssertTrue(dirtyDelta.deltaBlock.contains("let a = 42 // updated"))
        XCTAssertTrue(dirtyDelta.deltaBlock.contains("[DELETED FILE: Sources/B.swift]"))
    }

    func testAutoDiscoveryFromSubdirectoryAndSystemPromptAndGitExclude() throws {
        let nestedDir = tempDir.appendingPathComponent("Sources/Submodule/Deep", isDirectory: true)
        let gitInfoDir = tempDir.appendingPathComponent(".git/info", isDirectory: true)
        try FileManager.default.createDirectory(at: nestedDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: gitInfoDir, withIntermediateDirectories: true)

        let manifest = QWMemoryManifest(packName: "test-repo", rootPath: tempDir.path)
        try manifest.save(to: tempDir)
        QWMemoryPack.ensureGitExclude(in: tempDir)

        let excludeText = try String(contentsOf: gitInfoDir.appendingPathComponent("exclude"), encoding: .utf8)
        XCTAssertTrue(excludeText.contains(".qwg/"))

        let foundFromSubdir = QWMemoryPack.discoverManifestRoot(walkingUpFrom: nestedDir)
        XCTAssertEqual(foundFromSubdir?.standardizedFileURL.path, tempDir.standardizedFileURL.path)

        let systemPrompt = """
        You are an AI coding assistant.
        Active Workspaces:
        - \(nestedDir.path)
        Command Working Directory: \(tempDir.path)
        """
        let foundFromPrompt = QWMemoryPack.discoverManifestRoot(inSystemPrompt: systemPrompt)
        XCTAssertEqual(foundFromPrompt?.standardizedFileURL.path, tempDir.standardizedFileURL.path)
    }

    func testGitignoreGitExcludeAndQwgignoreGlobsAndNegation() throws {
        let fm = FileManager.default
        let docsDir = tempDir.appendingPathComponent("docs", isDirectory: true)
        let vendorDir = tempDir.appendingPathComponent("Sources/Vendor/mlx", isDirectory: true)
        let genDir = tempDir.appendingPathComponent("Sources/App/Generated", isDirectory: true)
        let appDir = tempDir.appendingPathComponent("Sources/App", isDirectory: true)
        let scriptSpikeDir = tempDir.appendingPathComponent("Scripts/spike", isDirectory: true)
        let gitInfoDir = tempDir.appendingPathComponent(".git/info", isDirectory: true)
        try fm.createDirectory(at: docsDir, withIntermediateDirectories: true)
        try fm.createDirectory(at: vendorDir, withIntermediateDirectories: true)
        try fm.createDirectory(at: genDir, withIntermediateDirectories: true)
        try fm.createDirectory(at: appDir, withIntermediateDirectories: true)
        try fm.createDirectory(at: scriptSpikeDir, withIntermediateDirectories: true)
        try fm.createDirectory(at: gitInfoDir, withIntermediateDirectories: true)

        try "private doc\n".write(to: docsDir.appendingPathComponent("secret.md"), atomically: true, encoding: .utf8)
        try "public doc\n".write(to: docsDir.appendingPathComponent("lsa.md"), atomically: true, encoding: .utf8)
        try "vendored header\n".write(to: vendorDir.appendingPathComponent("nax.h"), atomically: true, encoding: .utf8)
        try "generated opts\n".write(to: genDir.appendingPathComponent("Opts.swift"), atomically: true, encoding: .utf8)
        try "real app\n".write(to: appDir.appendingPathComponent("Main.swift"), atomically: true, encoding: .utf8)
        try "excluded by git info\n".write(to: appDir.appendingPathComponent("Scratch.swift"), atomically: true, encoding: .utf8)
        try "spike readme in scripts\n".write(to: scriptSpikeDir.appendingPathComponent("README.md"), atomically: true, encoding: .utf8)

        let gitignore = """
        /docs/*
        !/docs/lsa.md
        """
        try gitignore.write(to: tempDir.appendingPathComponent(".gitignore"), atomically: true, encoding: .utf8)
        try "Sources/App/Scratch.swift\n".write(to: gitInfoDir.appendingPathComponent("exclude"), atomically: true, encoding: .utf8)

        let qwgignore = """
        Sources/Vendor/mlx/
        **/Generated/
        Scripts/spike/
        """
        try qwgignore.write(to: tempDir.appendingPathComponent(".qwgignore"), atomically: true, encoding: .utf8)

        let defaultFiles = try QWMemoryPack.discoverFiles(in: tempDir, includeTests: false)
        XCTAssertEqual(defaultFiles.map(\.relativePath), ["docs/lsa.md", "Sources/App/Main.swift"])
        XCTAssertEqual(defaultFiles.map(\.tier), [1, 2])
    }

    func testUniversalMultiLanguageAndStructuredDataSymbolExtraction() {
        let rustSyms = QWMemoryVault.extractSymbols(
            from: "pub struct KvClusterIndex {}\npub fn dispatch_radix_topk() {}\n",
            relativePath: "crates/attn/src/lib.rs"
        )
        XCTAssertTrue(rustSyms.contains("KvClusterIndex"))
        XCTAssertTrue(rustSyms.contains("dispatch_radix_topk"))

        let pySyms = QWMemoryVault.extractSymbols(
            from: "class EmbeddingPipeline:\n    pass\nasync def encode_document_batch(items):\n    pass\n",
            relativePath: "pipeline/embed.py"
        )
        XCTAssertTrue(pySyms.contains("EmbeddingPipeline"))
        XCTAssertTrue(pySyms.contains("encode_document_batch"))

        let sqlSyms = QWMemoryVault.extractSymbols(
            from: "CREATE TABLE IF NOT EXISTS ledger_postings (id BIGINT);\nCREATE VIEW active_balances AS SELECT 1;\n",
            relativePath: "migrations/001_ledger.sql"
        )
        XCTAssertTrue(sqlSyms.contains("ledger_postings"))
        XCTAssertTrue(sqlSyms.contains("active_balances"))

        let tomlSyms = QWMemoryVault.extractSymbols(
            from: "[telemetry.exporter]\nsample_rate_hz = 100\n",
            relativePath: "config/runtime.toml"
        )
        XCTAssertTrue(tomlSyms.contains("telemetry.exporter"))
        XCTAssertTrue(tomlSyms.contains("sample_rate_hz"))

        let csvSyms = QWMemoryVault.extractSymbols(
            from: "customer_id,invoice_total_usd,settled_timestamp\n101,49.99,2026-11-08\n",
            relativePath: "datasets/invoices.csv"
        )
        XCTAssertEqual(csvSyms, ["customer_id", "invoice_total_usd", "settled_timestamp"])

        let texSyms = QWMemoryVault.extractSymbols(
            from: "\\section{Hierarchical Ward Clustering}\n\\subsection{Low-Rank Projection}\n",
            relativePath: "paper/method.tex"
        )
        XCTAssertEqual(texSyms, ["Hierarchical Ward Clustering", "Low-Rank Projection"])
    }

    func testSliceBeforeSentinelUserTurnIgnoresEmbeddedUserMarkerInSourceCode() {
        let sentinel = "__QWG_MEMORY_USER_SENTINEL__"
        let fullWire = "<|im_start|>system\nMemory<|memory|>\n[FILE: Sources/Server.swift]\n"
            + "if let r = fullWire.range(of: \"<|im_start|>user\") { return }\n"
            + "</|memory|>Memory<|im_end|>\n<|im_start|>user\n\(sentinel)<|im_end|>\n"
        let sliced = QWMemoryPack.sliceBeforeSentinelUserTurn(fullWire: fullWire, sentinel: sentinel)
        XCTAssertTrue(sliced.contains("fullWire.range(of: \"<|im_start|>user\")"))
        XCTAssertTrue(sliced.hasSuffix("</|memory|>Memory<|im_end|>\n"))
        XCTAssertFalse(sliced.contains(sentinel))
    }

    func testTwoTierCortexArchitectureMapOffContextVaultRecallAndElasticBudgetGuard() throws {
        let fm = FileManager.default
        let docsDir = tempDir.appendingPathComponent("docs", isDirectory: true)
        let srcDir = tempDir.appendingPathComponent("Sources/Runtime", isDirectory: true)
        try fm.createDirectory(at: docsDir, withIntermediateDirectories: true)
        try fm.createDirectory(at: srcDir, withIntermediateDirectories: true)

        let docBody = "# Architecture Overview\n## Memory Subsystem\nCore design rules.\n"
        let vaultBody = """
        public struct QWSlotPinnedLRUStore {
            public func prepareSlotForStore(tokens: [Int], pinAsSharedBase: Bool) -> Int {
                return tokens.count
            }
        }
        """ + String(repeating: "// padding line for vault overflow test\n", count: 40)

        try docBody.write(to: docsDir.appendingPathComponent("lsa.md"), atomically: true, encoding: .utf8)
        try vaultBody.write(to: srcDir.appendingPathComponent("QWSlotPinnedLRUStore.swift"), atomically: true, encoding: .utf8)

        let discovered = try QWMemoryPack.discoverFiles(in: tempDir, includeTests: false)
        XCTAssertEqual(discovered.count, 2)

        // Cap token budget so Tier-0 docs/lsa.md fits in Cortex and Tier-1 Swift file overflows to Vault.
        let built = QWMemoryPack.buildEnvelope(
            packName: "tiered-repo",
            rootPath: tempDir.path,
            files: discovered,
            clusterBlockTokens: 64,
            kvMode: "q4",
            maxTokens: 240
        )
        XCTAssertEqual(built.fileEntries.count, 2)
        XCTAssertEqual(built.cortexFileCount, 1)
        XCTAssertEqual(built.vaultFileCount, 1)
        XCTAssertTrue(built.fileEntries[0].inCortex)
        XCTAssertFalse(built.fileEntries[1].inCortex)
        XCTAssertTrue(built.fileEntries[1].symbols.contains("QWSlotPinnedLRUStore"))
        XCTAssertTrue(built.fileEntries[1].symbols.contains("prepareSlotForStore"))
        XCTAssertTrue(built.systemMessageContent.contains("[QWG REPOSITORY CORTEX MAP: tiered-repo"))
        XCTAssertTrue(built.systemMessageContent.contains("[VAULT] Sources/Runtime/QWSlotPinnedLRUStore.swift"))
        XCTAssertTrue(built.systemMessageContent.contains("[FILE: docs/lsa.md]"))
        XCTAssertFalse(built.systemMessageContent.contains("[FILE: Sources/Runtime/QWSlotPinnedLRUStore.swift]"))

        let manifest = QWMemoryManifest(
            packName: "tiered-repo",
            rootPath: tempDir.path,
            files: built.fileEntries
        )

        // Querying a Cortex-resident file returns empty Vault recall (already in Cortex .qwcache).
        let cortexQuery = QWMemoryVault.recallVaultContext(query: "What is in docs/lsa.md?", manifest: manifest)
        XCTAssertTrue(cortexQuery.isEmpty)

        // Querying a Vault symbol or filename pages in the verbatim Vault file on demand.
        let vaultQuery = QWMemoryVault.recallVaultContext(
            query: "How does prepareSlotForStore in QWSlotPinnedLRUStore work?",
            manifest: manifest
        )
        XCTAssertFalse(vaultQuery.isEmpty)
        XCTAssertEqual(vaultQuery.recalledFiles, ["Sources/Runtime/QWSlotPinnedLRUStore.swift"])
        XCTAssertTrue(vaultQuery.recallBlock.contains("<|memory_vault_recall|>"))
        XCTAssertTrue(vaultQuery.recallBlock.contains("public struct QWSlotPinnedLRUStore"))

        // Elastic budget guard preserves 32K Cortex .qwcache alongside 170K agent prompt.
        let cortexTokens = Array(repeating: 1, count: 32_768)
        let normalGuard = QWMemoryVault.applyElasticBudgetGuard(
            mountedMemoryTokens: cortexTokens,
            deltaTokens: [10, 11],
            agentPromptTokens: 170_388,
            maxNewTokens: 16_384,
            maxContext: 1_048_576
        )
        XCTAssertFalse(normalGuard.bypassedOversizedCache)
        XCTAssertEqual(normalGuard.prefixTokens.count, 32_770)

        // Elastic budget guard bypasses oversized legacy 1.01M .qwcache when agent sends 170K prompt (1.18M > 1.05M).
        let oversizedTokens = Array(repeating: 1, count: 1_011_738)
        let rescueGuard = QWMemoryVault.applyElasticBudgetGuard(
            mountedMemoryTokens: oversizedTokens,
            deltaTokens: [],
            agentPromptTokens: 154_004,
            maxNewTokens: 16_384,
            maxContext: 1_049_600
        )
        XCTAssertTrue(rescueGuard.bypassedOversizedCache)
        XCTAssertTrue(rescueGuard.prefixTokens.isEmpty)
        XCTAssertGreaterThan(rescueGuard.availableVaultTokenBudget, 0)
    }
}
