import XCTest
@testable import QWCore

/// The knob registry is the single declaration of every user-facing control.
final class QWKnobRegistryTests: XCTestCase {
    func testEveryKnobHasExactlyOneDefault() {
        for descriptor in QWKnobRegistry.all {
            XCTAssertNoThrow(try descriptor.validate(descriptor.defaultValue),
                             "knob \(descriptor.id) has an invalid default")
        }
        XCTAssertEqual(QWKnobRegistry.all.count, Set(QWKnobRegistry.all.map(\.id)).count,
                       "duplicate knob ids")
    }

    func testEveryKnobIdIsNonEmptyAndStable() {
        for descriptor in QWKnobRegistry.all {
            XCTAssertFalse(descriptor.id.isEmpty)
            XCTAssertFalse(descriptor.surfaces.isEmpty, "knob \(descriptor.id) has no surface")
            XCTAssertNotNil(QWKnobRegistry.descriptor(descriptor.id))
        }
        XCTAssertNil(QWKnobRegistry.descriptor("noSuchKnob"))
    }

    func testPrefillChunkRowsIsNormalTierAndLoadScope() throws {
        let knob = try XCTUnwrap(QWKnobRegistry.descriptor("prefillChunkRows"))
        XCTAssertEqual(knob.tier, .normal)
        XCTAssertEqual(knob.scope, .load)
        XCTAssertEqual(knob.defaultValue.intValue, 512)
        XCTAssertTrue(knob.surfaces.contains(.cli))
        XCTAssertTrue(knob.surfaces.contains(.dashboard))
    }

    func testKVModeIsExpertTierAndRequiresQuantCapability() throws {
        let knob = try XCTUnwrap(QWKnobRegistry.descriptor("kvMode"))
        XCTAssertEqual(knob.tier, .expert)
        XCTAssertNotNil(knob.requiredCapability)
        XCTAssertEqual(knob.kind, .string(allowed: ["f16", "q8", "q4"]))
        XCTAssertFalse(knob.surfaces.contains(.menuBar),
                       "expert knobs are not exposed on the normal MenuBar surface")
    }

    func testGdnPrefillRequiresGdnCapability() throws {
        let knob = try XCTUnwrap(QWKnobRegistry.descriptor("gdnPrefill"))
        XCTAssertEqual(knob.requiredCapability, .gdnPrefill)
    }

    func testExpertKnobRejectsOutOfRangeValue() throws {
        let knob = try XCTUnwrap(QWKnobRegistry.descriptor("prefillChunkRows"))
        XCTAssertThrowsError(try knob.validate(.int(7)))
        XCTAssertThrowsError(try knob.validate(.int(9000)))
        XCTAssertNoThrow(try knob.validate(.int(8192)))
    }

    func testWrongValueTypeIsRejected() throws {
        let knob = try XCTUnwrap(QWKnobRegistry.descriptor("prefillChunkRows"))
        XCTAssertThrowsError(try knob.validate(.string("512"))) { error in
            guard case QWKnobError.invalidValue = error else {
                return XCTFail("expected invalidValue, got \(error)")
            }
        }
    }

    func testUnrestrictedStringAcceptsAnyValue() throws {
        let knob = try XCTUnwrap(QWKnobRegistry.descriptor("systemPrompt"))
        XCTAssertNoThrow(try knob.validate(.string("")))
        XCTAssertNoThrow(try knob.validate(.string("be brief")))
    }

    func testCapabilityAbsenceIsUnavailableNotZero() throws {
        let knob = try XCTUnwrap(QWKnobRegistry.descriptor("gdnPrefill"))
        XCTAssertFalse(knob.isAvailable(under: []))
        XCTAssertTrue(knob.isAvailable(under: [.gdnPrefill]))
        let without = QWKnobRegistry.descriptors(availableUnder: [])
        XCTAssertFalse(without.contains { $0.id == "gdnPrefill" })
    }

    func testValueCapabilityIsEnforcedSeparatelyFromKnobAvailability() throws {
        let knob = try XCTUnwrap(QWKnobRegistry.descriptor("kvMode"))
        // The knob needs q8 to be offered; q4 needs q4 as well.
        XCTAssertFalse(knob.isAvailable(under: []))
        XCTAssertTrue(knob.isAvailable(under: [.kvQuantQ8]))
        XCTAssertNoThrow(try knob.validate(.string("q4"), capabilities: [.kvQuantQ8, .kvQuantQ4],
                                           model: "m"))
        XCTAssertThrowsError(try knob.validate(.string("q4"), capabilities: [.kvQuantQ8],
                                               model: "m")) { error in
            guard case QWKnobError.capabilityUnavailable = error else {
                return XCTFail("expected capabilityUnavailable, got \(error)")
            }
        }
    }

    func testDescriptorsAreFilterableBySurfaceAndCapability() {
        let cli = QWKnobRegistry.descriptors(for: .cli)
        XCTAssertTrue(cli.contains { $0.id == "prefillChunkRows" })
        let cache = QWKnobRegistry.descriptors(requireing: .prefixCache)
        XCTAssertTrue(cache.contains { $0.id == "warmCacheLimitGiB" })
        XCTAssertFalse(cache.contains { $0.id == "prefillChunkRows" })
    }

    func testTierPartitionIsDisjoint() {
        let normal = QWKnobRegistry.all.filter { $0.tier == .normal }.map(\.id)
        let expert = QWKnobRegistry.all.filter { $0.tier == .expert }.map(\.id)
        XCTAssertTrue(Set(normal).isDisjoint(with: Set(expert)))
        XCTAssertFalse(normal.isEmpty)
        XCTAssertFalse(expert.isEmpty)
    }

    func testBalancedPrefillChunkRowsAcrossContextRegimes() {
        XCTAssertEqual(QWPrefillLimits.maxRows, 8192)
        XCTAssertEqual(QWPrefillLimits.balancedChunkRows(forTotalTokens: 1, maxChunkRows: 8192), 1)
        XCTAssertEqual(QWPrefillLimits.balancedChunkRows(forTotalTokens: 65, maxChunkRows: 8192), 65)
        XCTAssertEqual(QWPrefillLimits.balancedChunkRows(forTotalTokens: 512, maxChunkRows: 8192), 512)
        XCTAssertEqual(QWPrefillLimits.balancedChunkRows(forTotalTokens: 4243, maxChunkRows: 8192), 4243)
        // 8,240 tokens splits into 2 balanced 64-row-aligned chunks: 4160 + 4080 (instead of 8192 + 48).
        XCTAssertEqual(QWPrefillLimits.balancedChunkRows(forTotalTokens: 8240, maxChunkRows: 8192), 4160)
        // 10,407 tokens splits into 2 balanced 64-row-aligned chunks: 5248 + 5159 (instead of 8192 + 2215).
        XCTAssertEqual(QWPrefillLimits.balancedChunkRows(forTotalTokens: 10407, maxChunkRows: 8192), 5248)
        // 16,311 tokens splits into 2 balanced chunks: 8192 + 8119.
        XCTAssertEqual(QWPrefillLimits.balancedChunkRows(forTotalTokens: 16311, maxChunkRows: 8192), 8192)
        // 32,768 tokens splits into 4 full 8192-row chunks.
        XCTAssertEqual(QWPrefillLimits.balancedChunkRows(forTotalTokens: 32768, maxChunkRows: 8192), 8192)
    }

    func testLSAClusterBlockAcceptsEmeansMacroBlock() throws {
        let knob = try XCTUnwrap(QWKnobRegistry.descriptor("lsaClusterBlock"))
        XCTAssertNoThrow(try knob.validate(.int(256), capabilities: [.lsa], model: "m"))
        XCTAssertNoThrow(try knob.validate(.int(512), capabilities: [.lsa], model: "m"))
        XCTAssertNoThrow(try knob.validate(.int(1024), capabilities: [.lsa], model: "m"))
        XCTAssertNoThrow(try knob.validate(.int(QWLSAConfig.emeansClusterBlock), capabilities: [.lsa], model: "m"))
        XCTAssertThrowsError(try knob.validate(.int(128), capabilities: [.lsa], model: "m"))
        XCTAssertThrowsError(try knob.validate(.int(8192), capabilities: [.lsa], model: "m"))
    }

    func testRecommendedDefaultsAndPromotedCLITuningHandles() throws {
        XCTAssertEqual(QWKnobRegistry.descriptor("lsaMode")?.defaultValue, .string("auto"))
        XCTAssertEqual(QWKnobRegistry.descriptor("kvMode")?.defaultValue, .string("q4"))
        for id in ["memoryPolicy", "residentExpertsPerLayer", "moeSpecBudget"] {
            let d = try XCTUnwrap(QWKnobRegistry.descriptor(id))
            XCTAssertTrue(d.surfaces.contains(.cli), "\(id) must be exposed on the CLI")
        }
        let tuning = QWEngineTuning(values: [:], environment: ["QW_MOE_SPEC_BUDGET": "4"])
        XCTAssertEqual(tuning.moeSpecBudget, 4)
    }
}
