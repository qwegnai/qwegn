// FNPrefixCacheStateTests.swift — The Flash-Next cache is the cold
// tier: a directory of saved prefixes on disk. Its size can be counted,
// so it is counted — and a directory that cannot be read is absent,
// never zero.

import XCTest
import Foundation
@testable import QWFlashNext

final class FNPrefixCacheStateTests: XCTestCase {
    private func makeDirectory() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("fncache-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    func testTheCacheCountsItsOwnEntriesAndBytes() throws {
        let dir = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }

        // A saved prefix, and a file that is not one.
        try Data(repeating: 7, count: 1024)
            .write(to: dir.appendingPathComponent("fn_entry_abcdef0123456789_16.qwcache"))
        try Data(repeating: 1, count: 512)
            .write(to: dir.appendingPathComponent("notes.txt"))
        try Data(repeating: 2, count: 256)
            .write(to: dir.appendingPathComponent("fn_entry_0000000000000000_4.qwcache"))

        let state = try XCTUnwrap(FNGPUEngine.diskState(in: dir))
        XCTAssertEqual(state.entries, 2, "only the saved prefixes are cache entries")
        XCTAssertEqual(state.bytes, 1024 + 256, "the bytes are the prefixes' own")
    }

    func testADirectoryThatCannotBeReadIsAbsentNotZero() {
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent("fncache-missing-\(UUID().uuidString)")
        XCTAssertNil(FNGPUEngine.diskState(in: missing),
                     "a cache that was not measured is absent, not an empty one")
    }
}
