import XCTest
@testable import QWFlashNext

/// The Flash-Next prefix cache's wiring: the RAM answers while it has
/// room, the disk answers the rest, and neither tier is advertised
/// without the plumbing that feeds it.
final class FNPrefixCacheWiringTests: XCTestCase {
    private static let cacheSource = "Sources/Subsystems/QWFlashNext/GPU/FNPrefixCache.swift"
    private static let engineSource = "Sources/Subsystems/QWFlashNext/GPU/FNGPUEngine.swift"
    private static let moduleSource = "Sources/Runtime/QWEngine/Module/FlashNextModule.swift"
    private static let generationSource = "Sources/Runtime/QWEngine/Module/FlashNextGeneration.swift"

    /// The source of a file, read from the package root.
    private static func read(_ relativePath: String) throws -> String {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // QWFlashNextTests
            .deletingLastPathComponent()   // Tests
            .deletingLastPathComponent()   // package root
        return try String(contentsOf: root.appendingPathComponent(relativePath), encoding: .utf8)
    }

    /// The RAM answers before the disk does. The order is the policy: a
    /// hit in the RAM costs no disk at all.
    func testTheRamTierIsAskedBeforeTheDisk() throws {
        let source = try Self.read(Self.cacheSource)
        let warm = try XCTUnwrap(source.range(of: "prefixCacheWarmStore, let blob = warm.blob(name: exactName)"),
                                  "the lookup no longer asks the RAM for the exact prefix")
        let cold = try XCTUnwrap(source.range(of: "readPrefixFile(url: exactURL"),
                                "the lookup no longer reads the disk for the exact prefix")
        XCTAssertLessThan(warm.lowerBound, cold.lowerBound,
                           "the RAM must answer before the disk does")
    }

    /// The disk tier is read only when it is on. With no room on the
    /// disk there is nothing to read.
    func testTheDiskIsOnlyReadWhenTheColdTierIsOn() throws {
        let source = try Self.read(Self.cacheSource)
        XCTAssertTrue(source.contains("guard prefixCacheSSDByteLimit > 0, let cacheDir else { return nil }"),
                      "the disk tier must be gated on the cold limit")
    }

    /// With no tier to hold it, the prefix is not built at all: the
    /// blob is a hundred megabytes of copy.
    func testNothingIsBuiltWhenNoTierCanHoldIt() throws {
        let source = try Self.read(Self.cacheSource)
        XCTAssertTrue(source.contains("guard prefixCacheSSDByteLimit > 0 || prefixCacheWarmStore != nil else { return }"),
                      "the save path must not build a prefix no tier can hold")
    }

    /// A prefix the cold tier cannot hold is not written to it: the
    /// tier stays inside the limit the settings gave it.
    func testTheColdTierNeverGrowsPastItsLimit() throws {
        let source = try Self.read(Self.cacheSource)
        XCTAssertTrue(source.contains("if out.count <= prefixCacheSSDByteLimit {"),
                      "a prefix bigger than the cold limit must not be written to it")
    }

    /// The tier is the one that served the prefix. The name is not
    /// written down where the engine can be.
    func testTheTierIsNotAWrittenDownName() throws {
        for path in [Self.moduleSource, Self.generationSource] {
            let source = try Self.read(path)
            XCTAssertFalse(source.contains("\"cold_ssd\""),
                           "\(path) writes the tier down instead of reporting the one that served it")
        }
    }

    /// The engine runs its cache paths when a tier is on, not when a
    /// directory happens to be there.
    func testTheEngineRunsItsCacheWhenATierIsOn() throws {
        let source = try Self.read(Self.engineSource)
        XCTAssertTrue(source.contains("if hasPrefixCacheTier,"),
                      "the lookup must run when a tier is on")
        XCTAssertFalse(source.contains("if let cacheDir = prefixCacheDirectory,\n               let restored"),
                       "the lookup must not wait for a cache directory")
    }

    func testPromptsAbove4096TokensAreNotRejectedBySavePrefixCache() throws {
        let source = try Self.read(Self.cacheSource)
        XCTAssertFalse(source.contains("N <= 4096"),
                       "savePrefixCache must not cap cached prefixes at 4096 tokens")
        XCTAssertTrue(source.contains("guard N >= 128 else { return }"),
                      "savePrefixCache must retain prefixes >= 128 tokens")
    }

    func testColdSSDSavesEvenWhenWarmStoreIsPresent() throws {
        let source = try Self.read(Self.cacheSource)
        XCTAssertFalse(source.contains("prefixCacheWarmStore == nil"),
                       "Cold SSD write must not be skipped when Warm RAM store is enabled")
    }

    func testWarmStoreEvictsOldestEntriesAndNotifiesCallback() {
        let store = FNPrefixWarmStore(byteLimit: 100)
        var evictedRecords: [(Int, Int)] = []
        store.onEvict = { tokCount, freedBytes in
            evictedRecords.append((tokCount, freedBytes))
        }
        store.store(name: "fn_entry_0000000000000001_1024.qwcache", blob: Data(count: 60))
        XCTAssertEqual(store.state.entries, 1)
        XCTAssertEqual(store.state.bytes, 60)

        // Storing a second 60-byte blob exceeds the 100-byte limit and must evict the first entry
        store.store(name: "fn_entry_0000000000000002_2048.qwcache", blob: Data(count: 60))
        XCTAssertEqual(store.state.entries, 1)
        XCTAssertEqual(store.state.bytes, 60)
        XCTAssertFalse(store.contains(name: "fn_entry_0000000000000001_1024.qwcache"))
        XCTAssertTrue(store.contains(name: "fn_entry_0000000000000002_2048.qwcache"))
        XCTAssertEqual(evictedRecords.count, 1)
        XCTAssertEqual(evictedRecords.first?.0, 1024)
        XCTAssertEqual(evictedRecords.first?.1, 60)
    }
}
