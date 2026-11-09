// QWServerSettingsKnobMappingTests.swift — The legacy settings, read as
// knob values. Every knob the mapping names must exist in the registry, and
// every value must be one the registry accepts.

import XCTest
import QWCore
@testable import QWServer

final class QWServerSettingsKnobMappingTests: XCTestCase {
    func testEveryMappedKnobExistsInTheRegistry() {
        let values = QWServerSettings().knobValues
        XCTAssertFalse(values.isEmpty)
        for id in values.keys {
            XCTAssertNotNil(QWKnobRegistry.descriptor(id),
                            "'\(id)' is mapped from the settings but the registry does not declare it")
        }
    }

    func testEveryMappedValueIsAcceptedByItsKnob() throws {
        let values = QWServerSettings().knobValues
        for (id, value) in values {
            let descriptor = try XCTUnwrap(QWKnobRegistry.descriptor(id))
            XCTAssertNoThrow(try descriptor.validate(value),
                             "the default settings carry a value '\(id)' the registry refuses")
        }
    }

    /// The mapping must follow the settings it reads: a changed setting is
    /// a changed knob value.
    func testTheMappingFollowsTheSettings() {
        var settings = QWServerSettings()
        settings.temperature = 0.35
        settings.topK = 7
        settings.prefillChunkSize = 256
        settings.speculativeDecodingEnabled = false
        let values = settings.knobValues
        XCTAssertEqual(values["temperature"], .double(0.35))
        XCTAssertEqual(values["topK"], .int(7))
        XCTAssertEqual(values["prefillChunkRows"], .int(256))
        XCTAssertEqual(values["speculativeEnabled"], .bool(false))
    }

    /// A scaling factor with no scaling is a setting the user turned off.
    /// Carrying it would migrate a scaling nobody asked for.
    func testAScalingFactorWithNoScalingIsNotCarried() {
        var settings = QWServerSettings()
        settings.ropeScalingType = "none"
        settings.ropeScalingFactor = 2.0
        XCTAssertNil(settings.knobValues["ropeScalingFactor"])
        settings.ropeScalingType = "yarn"
        XCTAssertEqual(settings.knobValues["ropeScalingFactor"], .double(2.0))
    }

    /// Every knob the registry declares must be accounted for: mapped from
    /// a setting, or named here as having no legacy counterpart.
    func testEveryRegistryKnobIsAccountedFor() {
        // The scaling factor is carried only when the scaling is on, so the
        // settings that carry everything are the ones that turn it on.
        var settings = QWServerSettings()
        settings.ropeScalingType = "yarn"
        let withoutLegacyCounterpart: Set<String> = [
            "prefixCacheEnabled",   // the old settings had no switch for it
            "cacheEntryLimit",      // the old settings had no limit for it
            "splitKTokensPerSplit", // the old settings had no split for it
        ]
        let mapped = Set(settings.knobValues.keys)
        for descriptor in QWKnobRegistry.all {
            let accounted = mapped.contains(descriptor.id)
                || withoutLegacyCounterpart.contains(descriptor.id)
            XCTAssertTrue(accounted,
                          "knob '\(descriptor.id)' is neither mapped nor listed as having no legacy counterpart")
        }
        for id in withoutLegacyCounterpart {
            XCTAssertFalse(mapped.contains(id),
                           "'\(id)' is listed as having no legacy counterpart but is mapped")
        }
    }

    /// Engine tuning rides in one dictionary: a default is never stored, so
    /// a profile holds only what the user changed.
    func testEngineTuningStoresOnlyNonDefaults() {
        var settings = QWServerSettings()
        settings.apply(knobValues: ["memoryPolicy": .string("safe"), "expertCache": .bool(true),
                                    "residentExpertsPerLayer": .int(200), "coldEvents": .string("nope")])
        XCTAssertEqual(settings.engineTuning, ["memoryPolicy": .string("safe"),
                                               "residentExpertsPerLayer": .int(200)])
        XCTAssertEqual(settings.knobValues["memoryPolicy"], .string("safe"))
        XCTAssertEqual(settings.knobValues["expertCache"], .bool(true))
        XCTAssertEqual(settings.resolvedEngineTuning.memoryPolicy, .safe)
        XCTAssertEqual(settings.resolvedEngineTuning.residentExpertsPerLayer, 200)
        settings.setEngineTuning("memoryPolicy", .string("maxResident"))
        XCTAssertNil(settings.engineTuning["memoryPolicy"])
    }

    func testEngineTuningRoundTripsThroughJSONAndResets() throws {
        var settings = QWServerSettings()
        settings.setEngineTuning("prefillSpanMap", .bool(false))
        settings.setEngineTuning("osReserveGiB", .double(12))
        let decoded = try JSONDecoder().decode(QWServerSettings.self, from: JSONEncoder().encode(settings))
        XCTAssertEqual(decoded.engineTuning, settings.engineTuning)
        XCTAssertFalse(decoded.resolvedEngineTuning.prefillSpanMap)
        XCTAssertEqual(QWServerSettings.settingKeys(forKnobIds: ["prefillSpanMap", "topK"]), ["engineTuning", "topK"])
        XCTAssertTrue(QWServerSettings.modelScopedSettingKeys.contains("engineTuning"))
        var reset = decoded
        reset.applyModelConfigDefaults(packPath: "")
        XCTAssertTrue(reset.engineTuning.isEmpty)
    }
}
