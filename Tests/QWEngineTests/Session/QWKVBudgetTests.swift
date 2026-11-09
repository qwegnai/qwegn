import XCTest
import QWCore
import QWKV
@testable import QWEngine

/// KV budget law (P4-AD-58). Pure arithmetic; loads no model, no Metal.
/// The 32K frozen-history continuation run failed with
/// "QWKV: out of pages (ask 1, free 0)" at QWKVCache.prepareWrites: a
final class QWKVBudgetTests: XCTestCase {

    /// Mirrors QWKVCache.init's pool arithmetic for one cache.
    private func pagesPerPool(regionBytes: Int, layers: Int) -> Int {
        let tables = QWKVCache.tableSlotBytes * 2 * layers * QWKVCache.maxSeqs
        let poolBytes = (regionBytes - tables) / (2 * layers)
        return poolBytes / QWKVCache.pageBytes(mode: .f16)
    }

    /// The failed run geometry: maxTokens 33024, one MTP cache, one donor
    /// with 32767 written pair rows, one adopter with 32800 capacity.
    func testAdopterCopyOnWriteNeedsOnePageBeyondTheSession() throws {
        let maxTokens = 33024
        let sessionPages = (maxTokens + QWKVCache.pageTokens - 1) / QWKVCache.pageTokens
        XCTAssertEqual(QWEngineDriver.kvPagesPerPool(maxTokens: maxTokens), sessionPages + 1)
        let donorRows = 32767            // ceil → 128 pages, last one partially written
        let adopterCapacity = 32800
        let adopterPages = (adopterCapacity + QWKVCache.pageTokens - 1) / QWKVCache.pageTokens
        XCTAssertEqual(adopterPages, sessionPages)
        XCTAssertNotEqual(donorRows % QWKVCache.pageTokens, 0)
        let donorPages = (donorRows + QWKVCache.pageTokens - 1) / QWKVCache.pageTokens
        // The adopter shares the donor's 128 pages, allocates one private
        // page, and the first append into the shared tail page must
        // copy-on-write: donor + adopter-private + COW replacement.
        let requiredPerPool = donorPages + 1 + 1
        XCTAssertEqual(QWEngineDriver.kvPagesPerPool(maxTokens: maxTokens), requiredPerPool)

        // The NEW budget delivers exactly that many pages per pool.
        let mtpRegion = QWEngineDriver.kvBudgetBytes(maxTokens: maxTokens, mtpDrafts: 1, geometry: .qwen38_27b)
            - QWEngineDriver.kvBudgetBytes(maxTokens: maxTokens, mtpDrafts: 0, geometry: .qwen38_27b)
        XCTAssertEqual(pagesPerPool(regionBytes: mtpRegion, layers: 1),
                       QWEngineDriver.kvPagesPerPool(maxTokens: maxTokens))

        // The OLD budget (what the failing run used) is reproduced and
        // proven to be the defect: no free page for the COW replacement.
        let oldMtpRegion = QWKVCache.kvBytes(tokens: maxTokens, layers: 1, mode: .f16)
            + QWKVCache.tableSlotBytes * 2 * QWKVCache.maxSeqs
        XCTAssertEqual(pagesPerPool(regionBytes: oldMtpRegion, layers: 1), sessionPages)
        XCTAssertLessThan(pagesPerPool(regionBytes: oldMtpRegion, layers: 1), requiredPerPool)
    }

    func testBudgetIsExactAndDisabledPathKeepsTrunkOnly() throws {
        let bytes0 = QWEngineDriver.kvBudgetBytes(maxTokens: 4096, mtpDrafts: 0, geometry: .qwen38_27b)
        let bytes1 = QWEngineDriver.kvBudgetBytes(maxTokens: 4096, mtpDrafts: 1, geometry: .qwen38_27b)
        let bytes2 = QWEngineDriver.kvBudgetBytes(maxTokens: 4096, mtpDrafts: 2, geometry: .qwen38_27b)
        XCTAssertEqual(bytes1, bytes2)
        let mtp = bytes1 - bytes0
        XCTAssertEqual(pagesPerPool(regionBytes: bytes0, layers: 16), 16 + 1)
        XCTAssertEqual(pagesPerPool(regionBytes: mtp, layers: 1), 16 + 1)
        // Exact layout: the budget leaves no slack beyond whole pages and
        // the MTP region is exactly one extra-page-per-pool cache.
        XCTAssertEqual((bytes0 - QWKVCache.tableSlotBytes * 2 * 16 * QWKVCache.maxSeqs)
                       % QWKVCache.pageBytes(mode: .f16), 0)
        XCTAssertEqual((mtp - QWKVCache.tableSlotBytes * 2 * QWKVCache.maxSeqs)
                       % QWKVCache.pageBytes(mode: .f16), 0)
        XCTAssertEqual(mtp,
                       QWKVCache.kvBytes(tokens: 4096 + QWKVCache.pageTokens, layers: 1, mode: .f16)
                       + QWKVCache.tableSlotBytes * 2 * QWKVCache.maxSeqs)
    }
}
