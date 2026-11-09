// QWWeightFormatDecisionTests.swift — the container decision: which bytes a
// model's weights live in, and every way that decision says no.
import XCTest
@testable import QWEngine
@testable import QWCore

final class QWWeightFormatDecisionTests: XCTestCase {

    // MARK: - fixtures

    /// A root that holds the given container: a real directory with the
    /// file the resolver looks for, so the kind is the one the engine sees.
    private func root(_ kind: QWWeightSourceKind = .safetensorsSingle) throws -> QWModelRoot {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("qwg-format-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        let name = kind == .qwp1 ? "model.qwp" : "model.safetensors"
        try Data().write(to: dir.appendingPathComponent(name))
        return QWModelRoot(url: dir)
    }

    /// A hybrid geometry: linear attention on most of its layers, as a
    /// Next-style model has. The GDN's in-projection HAS a twin in the
    /// family, so a hybrid model keeps its own bytes like any other.
    private func hybridGeometry() -> QWModelGeometry {
        QWModelGeometry.qwen38_27b
    }

    private func decide(_ quantization: QWQuantizationSpec?,
                        geometry: QWModelGeometry,
                        mtpDrafts: Int = 0,
                        kind: QWWeightSourceKind = .safetensorsSingle) throws -> QWWeightFormat {
        try QWEngineDriver.weightFormat(for: try root(kind), quantization: quantization,
                                        geometry: geometry, mtpDrafts: mtpDrafts)
    }

    // MARK: - the family

    /// An MLX affine checkpoint is held in its OWN bytes: the decision keeps
    /// the family's container, and the basis that goes with it.
    func testAMLXAffineCheckpointKeepsItsOwnBytes() throws {
        let geometry = try QWModelGeometry.stubPureAttention()
        let format = try decide(QWQuantizationSpec(bits: 3, groupSize: 64), geometry: geometry)
        XCTAssertEqual(format, .nativeAffine(bits: 3, groupSize: 64))
        XCTAssertTrue(format.isNativeAffine)
        XCTAssertEqual(format.name, "AQ3_G64")
        XCTAssertEqual(format.basis, .standard)
    }

    /// The container's name is what the engine reports, so it must name the
    /// pair and not the family.
    func testTheContainerNamesItsOwnPair() throws {
        let geometry = try QWModelGeometry.stubPureAttention()
        let two = try decide(QWQuantizationSpec(bits: 2, groupSize: 64), geometry: geometry)
        XCTAssertEqual(two.name, "AQ2_G64")
        XCTAssertEqual(two.bits, 2)
        XCTAssertEqual(two.groupSize, 64)
    }

    // MARK: - the four ways to say no

    /// A scheme the engine does not read is REFUSED BY NAME: its bytes would
    /// be read as a scheme they are not.
    func testASchemeTheEngineDoesNotReadIsRefusedByName() throws {
        let geometry = try QWModelGeometry.stubPureAttention()
        XCTAssertThrowsError(try decide(QWQuantizationSpec(method: "gptq", bits: 3, groupSize: 64),
                                         geometry: geometry)) { error in
            let message = "\(error)"
            XCTAssertTrue(message.contains("gptq"), message)
            XCTAssertTrue(message.contains("affine"), message)
        }
        XCTAssertThrowsError(try decide(QWQuantizationSpec(bits: 3, groupSize: 64, mode: "nf4"),
                                         geometry: geometry)) { error in
            let message = "\(error)"
            XCTAssertTrue(message.contains("nf4"), message)
        }
    }

    /// A pair the family does not hold is not a refusal: the incumbent's
    /// container holds any width, and the values are the same either way.
    func testAPairTheFamilyDoesNotHoldKeepsTheIncumbent() throws {
        let geometry = try QWModelGeometry.stubPureAttention()
        let format = try decide(QWQuantizationSpec(bits: 5, groupSize: 64), geometry: geometry)
        XCTAssertEqual(format, .hq4G32)
        XCTAssertEqual(format.name, "HQ4_G32")
        XCTAssertEqual(format.basis, .rotated)
    }

    /// (4, 32) is kept in its own native affine container without re-quantizing.
    func testATieIsNotAWin() throws {
        let geometry = try QWModelGeometry.stubPureAttention()
        let format = try decide(QWQuantizationSpec(bits: 4, groupSize: 32), geometry: geometry)
        XCTAssertEqual(format, .nativeAffine(bits: 4, groupSize: 32))
    }

    /// Wider supported pairs (6-bit, 8-bit) also keep their own native affine bytes
    /// without being down-quantized to HQ4_G32.
    func testAWiderPairKeepsTheIncumbent() throws {
        let geometry = try QWModelGeometry.stubPureAttention()
        for pair in [(bits: 6, group: 64), (bits: 8, group: 64)] {
            let format = try decide(QWQuantizationSpec(bits: pair.bits, groupSize: pair.group),
                                    geometry: geometry)
            XCTAssertEqual(format, .nativeAffine(bits: pair.bits, groupSize: pair.group), "\(pair.bits)-bit G\(pair.group)")
        }
    }

    /// A path without a twin keeps the incumbent's container. MTP drafts read
    /// weights the family has no kernel for; experts and hybrid GDN layers have
    /// twins, so they keep their native bytes.
    func testAPathWithoutATwinKeepsTheIncumbent() throws {
        let dense = try QWModelGeometry.stubPureAttention()
        let moe = try QWModelGeometry.stubPureMoE()
        let hybrid = hybridGeometry()
        XCTAssertTrue(QWEngineDriver.familyServes(geometry: dense, mtpDrafts: 0))
        XCTAssertTrue(QWEngineDriver.familyServes(geometry: moe, mtpDrafts: 0))
        XCTAssertTrue(QWEngineDriver.familyServes(geometry: hybrid, mtpDrafts: 0))
        XCTAssertFalse(QWEngineDriver.familyServes(geometry: dense, mtpDrafts: 1))
        let quantization = QWQuantizationSpec(bits: 3, groupSize: 64)
        XCTAssertEqual(try decide(quantization, geometry: moe), .nativeAffine(bits: 3, groupSize: 64))
        XCTAssertEqual(try decide(quantization, geometry: hybrid), .nativeAffine(bits: 3, groupSize: 64))
        XCTAssertEqual(try decide(quantization, geometry: dense, mtpDrafts: 1), .hq4G32)
    }

    // MARK: - the containers that never reach the family

    /// The pack holds the engine's own container, so it keeps it.
    func testThePackKeepsTheIncumbent() throws {
        let geometry = try QWModelGeometry.stubPureAttention()
        let format = try decide(QWQuantizationSpec(bits: 3, groupSize: 64),
                                geometry: geometry, kind: .qwp1)
        XCTAssertEqual(format, .hq4G32)
    }

    /// A checkpoint that says nothing about its quantisation keeps the
    /// incumbent: there is no pair to keep in its own bytes.
    func testACheckpointWithNoQuantisationKeepsTheIncumbent() throws {
        let geometry = try QWModelGeometry.stubPureAttention()
        XCTAssertEqual(try decide(nil, geometry: geometry), .hq4G32)
        XCTAssertEqual(try decide(QWQuantizationSpec(bits: 3), geometry: geometry), .hq4G32)
    }

    /// The environment can force the incumbent, so both routes are measurable
    /// in one build. It cannot force a container the checkpoint cannot hold.
    func testTheEnvironmentCanForceTheIncumbent() throws {
        let geometry = try QWModelGeometry.stubPureAttention()
        let quantization = QWQuantizationSpec(bits: 3, groupSize: 64)
        setenv("QW_WEIGHT_FORMAT", "incumbent", 1)
        defer { unsetenv("QW_WEIGHT_FORMAT") }
        XCTAssertEqual(try decide(quantization, geometry: geometry), .hq4G32)
        XCTAssertEqual(QWEngineDriver.forcedWeightFormat(), .hq4G32)
    }

    // MARK: - the reason the family was declined

    /// Every way the family is declined NAMES its cause. A model in the
    /// incumbent's container says why: the user cannot otherwise tell an
    /// unsupported pair from a bigger one from a path with no twin.
    func testEveryDeclineNamesItsCause() throws {
        let dense = try QWModelGeometry.stubPureAttention()
        let moe = try QWModelGeometry.stubPureMoE()
        let hybrid = hybridGeometry()
        let quantization = QWQuantizationSpec(bits: 3, groupSize: 64)
        func reason(_ q: QWQuantizationSpec?, _ g: QWModelGeometry,
                    drafts: Int = 0, pack: Bool = false) throws -> String? {
            try QWEngineDriver.weightFormatChoice(
                for: try root(pack ? .qwp1 : .safetensorsSingle), quantization: q,
                geometry: g, mtpDrafts: drafts).declineReason
        }
        // The family was chosen: there is nothing to explain.
        XCTAssertNil(try reason(quantization, dense))
        // A checkpoint that says nothing, or a pair the family does not hold.
        XCTAssertTrue(try reason(nil, dense)!.contains("quantisation"))
        XCTAssertTrue(try reason(QWQuantizationSpec(bits: 5, groupSize: 64), dense)!.contains("5-bit"))
        XCTAssertNil(try reason(QWQuantizationSpec(bits: 4, groupSize: 32), dense))
        // Both MoE and hybrid GDN have twins, so neither is declined.
        XCTAssertNil(try reason(quantization, moe))
        XCTAssertNil(try reason(quantization, hybrid))
        XCTAssertTrue(try reason(quantization, dense, drafts: 1)!.contains("MTP"))
        // The environment, and the engine's own pack.
        setenv("QW_WEIGHT_FORMAT", "incumbent", 1)
        defer { unsetenv("QW_WEIGHT_FORMAT") }
        XCTAssertTrue(try reason(quantization, dense)!.contains("QW_WEIGHT_FORMAT"))
        unsetenv("QW_WEIGHT_FORMAT")
        XCTAssertTrue(try reason(quantization, dense, pack: true)!.contains("pack"))
    }
}
