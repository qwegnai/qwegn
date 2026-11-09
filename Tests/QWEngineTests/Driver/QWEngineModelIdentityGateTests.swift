import XCTest
@testable import QWEngine
import QWQuant

/// §3 identity gate (plan gate): "a Flash-Next pack reaches an explicit
/// unsupported-feature error instead of the 27B loader". The FN-identity
/// pack is synthesized from the 27B selfpack fixture exactly like
final class QWEngineModelIdentityGateTests: XCTestCase {

    private static var selfpack: URL {
        var dir = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        for _ in 0..<8 {
            let cand = dir.appendingPathComponent("Tests/QWQuantTests/fixtures/selfpack.qwp")
            if FileManager.default.fileExists(atPath: cand.path) {
                return cand
            }
            dir = dir.deletingLastPathComponent()
        }
        fatalError("selfpack.qwp fixture missing — run: python3 Scripts/quantize/pack.py --selfpack Tests/QWQuantTests/fixtures/selfpack.qwp")
    }

    private static func fnIdentityPack() throws -> URL {
        let data = try Data(contentsOf: selfpack)
        guard let prov = data.range(of: Data("\"provenance\": {".utf8))
            ?? data.range(of: Data("\"provenance\":{".utf8)) else {
            throw NSError(domain: "test", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "provenance object not in header"])
        }
        let insertAt = prov.upperBound
        let extra = " \"model_id\": \"qwen3.8-flash-next\","
        var out = Data()
        out.append(data.subdata(in: 0..<insertAt))
        out.append(Data(extra.utf8))
        out.append(data.subdata(in: insertAt..<data.count))
        var len = out.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 4, as: UInt64.self) }
        len += UInt64(extra.utf8.count)
        withUnsafeBytes(of: len) { out.replaceSubrange(4..<12, with: $0) }
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("fn-gate.qwp")
        try out.write(to: tmp)
        return tmp
    }

    func testFNPackRejectedBeforeEngineInit() throws {
        let pack = try Self.fnIdentityPack()
        do {
            _ = try QWEngineDriver(packPath: pack.path, prewarmDoors: DenseModule.requiredDoors)
            XCTFail("FN pack must not reach the 27B engine")
        } catch let error as QWEngineDriverError {
            guard case QWEngineDriverError.unsupportedModel(let m) = error else {
                return XCTFail("expected unsupportedModel, got \(error)")
            }
            XCTAssertEqual(m, "qwen3.8-flash-next")
        }
        // The error message is operator-actionable and names the mechanism.
        let msg = QWEngineDriverError.unsupportedModel("qwen3.8-flash-next").description
        XCTAssertTrue(msg.contains("no engine adapter"), msg)
    }

    func testLegacyControlPackStillAdmittedToInit() throws {
        // The selfpack fixture has no model_id → the identity gate must be
        // silent for it (it proceeds into HAL/pack open; failure there is a
        // different, pre-existing behavior — we only assert the identity
        // gate did NOT reject it with unsupportedModel).
        do {
            _ = try QWEngineDriver(packPath: Self.selfpack.path, prewarmDoors: DenseModule.requiredDoors)
        } catch let error as QWEngineDriverError {
            if case QWEngineDriverError.unsupportedModel = error {
                XCTFail("legacy control pack rejected by identity gate: \(error)")
            }
        } catch {
            // non-identity errors are outside this gate's scope
        }
    }
}
