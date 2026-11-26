// QWPrewarmTests.swift — Gate optional pipelines before engine prewarm.

import XCTest
@testable import QWEngine

final class QWPrewarmTests: XCTestCase {
    func testLSASelectorsArePrewarmedOnlyWithAnLSAOperator() {
        let doors = [
            "qwg_lsa_ward_cluster_512",
            "qwg_lsa_select_blocks",
            "qwg_lsa_select_score",
            "qwg_lsa_select_hist2",
            "qwg_lsa_select_expand",
            "qwg_lsa_select_finalize",
            "qwg_mla_indexer_select",
        ]

        XCTAssertEqual(QWEngineDriver.prewarmDoors(doors, hasLSAOperator: true), doors)
        XCTAssertEqual(
            QWEngineDriver.prewarmDoors(doors, hasLSAOperator: false),
            ["qwg_lsa_ward_cluster_512", "qwg_mla_indexer_select"]
        )
    }
}
