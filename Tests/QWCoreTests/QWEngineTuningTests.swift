// QWEngineTuningTests.swift — Engine tunables resolve environment > values > default,
// and an invalid value at any layer falls through to the next.

import XCTest
@testable import QWCore

final class QWEngineTuningTests: XCTestCase {
    func testDefaultsMatchTheRegistry() throws {
        let tuning = QWEngineTuning.default
        for id in QWEngineTuning.knobIds {
            let descriptor = try XCTUnwrap(QWKnobRegistry.descriptor(id), "'\(id)' is not in the registry")
            if let value = tuning.knobValues[id] {
                XCTAssertEqual(value, descriptor.defaultValue, "'\(id)' default disagrees with the registry")
            }
            XCTAssertNotNil(QWEngineTuning.environmentNames[id], "'\(id)' has no environment override")
        }
        XCTAssertNil(tuning.residentExpertsPerLayer)
        XCTAssertNil(tuning.osReserveGiB)
    }

    func testValuesOverrideDefaults() {
        let tuning = QWEngineTuning(values: [
            "memoryPolicy": .string("safe"), "residentExpertsPerLayer": .int(200),
            "osReserveGiB": .double(12), "coldEvents": .bool(false),
        ])
        XCTAssertEqual(tuning.memoryPolicy, .safe)
        XCTAssertEqual(tuning.residentExpertsPerLayer, 200)
        XCTAssertEqual(tuning.osReserveGiB, 12)
        XCTAssertFalse(tuning.coldEvents)
        XCTAssertTrue(tuning.expertCache)
    }

    func testEnvironmentOverridesValues() {
        let tuning = QWEngineTuning(values: ["memoryPolicy": .string("safe"), "prefillSpanMap": .bool(true)],
                                    environment: ["QW_MEMORY_POLICY": "maxResident", "QW_PREFILL_SPAN_MAP": "0",
                                                  "QW_RESIDENT_EXPERTS_PER_LAYER": "64"])
        XCTAssertEqual(tuning.memoryPolicy, .maxResident)
        XCTAssertFalse(tuning.prefillSpanMap)
        XCTAssertEqual(tuning.residentExpertsPerLayer, 64)
    }

    func testInvalidValuesFallThrough() {
        let tuning = QWEngineTuning(values: ["memoryPolicy": .string("bogus"), "residentExpertsPerLayer": .int(-3),
                                             "expertCache": .string("yes")],
                                    environment: ["QW_OS_RESERVE_GIB": "lots", "QW_MEMORY_POLICY": ""])
        XCTAssertEqual(tuning, .default)
    }

    func testZeroResidentExpertsMeansAuto() {
        XCTAssertNil(QWEngineTuning(values: ["residentExpertsPerLayer": .int(0)]).residentExpertsPerLayer)
    }

    func testOverlayKeepsValuesTheEnvironmentDoesNotName() {
        let base = QWEngineTuning(values: ["coldEventOverlap": .bool(false), "osReserveGiB": .double(10)])
        let overlaid = base.overlaid(environment: ["QW_EXPERT_CACHE": "off"])
        XCTAssertFalse(overlaid.coldEventOverlap)
        XCTAssertFalse(overlaid.expertCache)
        XCTAssertEqual(overlaid.osReserveGiB, 10)
    }
}
