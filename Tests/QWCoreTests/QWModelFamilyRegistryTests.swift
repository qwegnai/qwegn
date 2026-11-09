import XCTest
import Foundation
@testable import QWCore

/// The registry is where a `model_type` becomes a family. The standard
/// families are registered so their configs decode, and they are NOT
/// officially supported until a module serves them.
final class QWModelFamilyRegistryTests: XCTestCase {
    func testStandardModelTypesAreRegistered() {
        for type in ["qwen3", "qwen3_moe", "qwen2", "llama", "mistral"] {
            XCTAssertNotNil(QWModelFamilyRegistry.shared.spec(forModelType: type),
                            "expected \(type) to be registered")
        }
    }

    func testHybridModelTypesAreUnchanged() {
        XCTAssertEqual(QWModelFamilyRegistry.shared.spec(forModelType: "qwen3_5")?.family, .qwen38Dense)
        XCTAssertEqual(QWModelFamilyRegistry.shared.spec(forModelType: "qwen4_exp")?.family, .qwen38FlashNext)
    }

    func testNewFamiliesAreRegisteredButDistinctFromTheHybridOnes() {
        let qwen3 = QWModelFamilyRegistry.shared.spec(forModelType: "qwen3")!
        XCTAssertNotEqual(qwen3.family, .qwen38Dense)
        XCTAssertNotEqual(qwen3.family, .qwen38FlashNext)
    }

    func testHybridDigestIsUnchangedByTheNewFamilies() {
        let spec = QWModelFamilyRegistry.shared.spec(forModelType: "qwen3_5")!
        XCTAssertEqual(spec.family, .qwen38Dense)
        XCTAssertEqual(spec.defaultAttnOutputGate, true)
    }

    func testTheStandardDenseFamiliesAreOfficiallySupported() {
        // The standard path runs its own writer, decode and paged prefill
        // (4.3b, 4.3c, 4.3d), so the dense standard families are served.
        for type in ["qwen2", "llama"] {
            XCTAssertTrue(QWModelFamilyRegistry.shared.isSupported(modelType: type), type)
        }
        XCTAssertTrue(QWModelFamilyRegistry.shared.isSupported(modelType: "qwen3_5"))
        XCTAssertTrue(QWModelFamilyRegistry.shared.isSupported(modelType: "qwen4_exp"))
    }

    func testMistralWaitsForTheSlidingWindow() {
        // Every published Mistral config declares `sliding_window: 4096`,
        // and the engine has no window — so the admission gate must not
        // claim the family until the window lands (C116).
        XCTAssertFalse(QWModelFamilyRegistry.shared.isSupported(modelType: "mistral"))
    }

    func testTheQwen3FamiliesAreOfficiallySupported() {
        // Task 4.4 served the MoE path, Task 4.5a the dense one, and
        // Task 4.5b checked every published checkpoint of both families
        // against the decoder's refusals before the gate opened.
        for type in ["qwen3", "qwen3_moe"] {
            XCTAssertTrue(QWModelFamilyRegistry.shared.isSupported(modelType: type), type)
        }
    }

    func testNoStandardFamilyPacksAnAttentionOutputGate() {
        for type in ["qwen3", "qwen3_moe", "qwen2", "llama", "mistral"] {
            let spec = QWModelFamilyRegistry.shared.spec(forModelType: type)!
            XCTAssertFalse(spec.defaultAttnOutputGate, type)
        }
    }
}
