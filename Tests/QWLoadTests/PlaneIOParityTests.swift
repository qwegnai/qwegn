// PlaneIOParityTests.swift — SPEC_QWLOAD parity law: the mover's bytes are
// identical to the frozen legacy loader bytes. Selfpack fixture always;
// the local FN pack when installed (skipped otherwise).
import XCTest
import Foundation
@testable import QWLoad
import QWQuant
import QWHAL
import QWCore
import QWEngine

final class PlaneIOParityTests: XCTestCase {

    private static func find(_ rel: [String], under file: String = #filePath) -> URL? {
        var d = URL(fileURLWithPath: file).deletingLastPathComponent()
        for _ in 0..<8 {
            let c = d.appendingPathComponent("Tests/QWQuantTests/fixtures/selfpack.qwp")
            if FileManager.default.fileExists(atPath: c.path) { return c }
            d = d.deletingLastPathComponent()
        }
        _ = rel
        return nil
    }

    func testLayoutByteMathMatchesLegacyFormulas() throws {
        let e = QWQTensorEntry(shape: [640, 10240], hq4Off: 4096)
        XCTAssertEqual(PlaneLayout.hq4.byteCount(e), 640 * (10240 / 32) * 20)
        XCTAssertEqual(PlaneLayout.hq4.offset(e), 4096)
        XCTAssertEqual(PlaneLayout.resolve(e), .hq4)
        let r = QWQTensorEntry(shape: [4, 10240], rawOff: 256)
        XCTAssertEqual(PlaneLayout.raw.byteCount(r), 4 * 10240 * 2)
        XCTAssertEqual(PlaneLayout.resolve(r), .raw)
        let ragged = QWQTensorEntry(shape: [4, 100], hq4Off: 0)
        XCTAssertNil(PlaneLayout.hq4.byteCount(ragged))
    }

    private func legacyBytes(_ pack: QWQuant.Pack, _ name: String,
                             _ layout: PlaneLayout) throws -> Data {
        let e = try pack.entry(name)
        guard let count = layout.byteCount(e) else { throw NSError(domain: "t", code: 1) }
        var d = Data(count: count)
        try d.withUnsafeMutableBytes { raw in
            let p = raw.baseAddress!
            switch layout {
            case .raw: try pack.copyRawBytes(name, to: p, at: 0, count: e.shape.reduce(1, *))
            case .hq4: try pack.copyPlaneBytes(name, key: \.hq4Off, to: p, count: count)
            }
        }
        return d
    }

    func testSelfpackMoverParityBothArms() throws {
        guard let url = Self.find([]) else { throw XCTSkip("selfpack fixture missing") }
        let pack = try QWQuant.Pack(url: url)
        for arm in [["QW_PLANE_BUFFERED": "1"], [:]] {
            let io = PlaneIO(pack: pack, policy: LoadPolicy(descriptor: pack.descriptor, env: arm))
            var checked = 0
            for name in pack.tensorNames {
                let e = try pack.entry(name)
                guard let layout = PlaneLayout.resolve(e) else { continue }
                let want = try legacyBytes(pack, name, layout)
                let got = try io.load(name, layout: layout)
                XCTAssertEqual(got, want, "\(name) [\(#function)] arm=\(arm)")
                checked += 1
            }
            XCTAssertGreaterThan(checked, 0, "selfpack must contain verbatim planes")
        }
    }


    func testHeadroomProbeSane() throws {
        let h = PackWiring.headroomBytes()
        let total = Int(ProcessInfo.processInfo.physicalMemory)
        XCTAssertGreaterThan(h, 0)
        XCTAssertLessThan(h, total)
    }

    func testCheckpointTensorLoaderResolveNameAndSpan() throws {
        let existing: Set<String> = ["model.layers.0.self_attn.q_proj.weight", "model.embed_tokens.weight"]
        let resolved = QWCheckpointTensorLoader.resolveName(
            "model.language_model.layers.0.self_attn.q_proj.weight",
            rawExists: { existing.contains($0) }
        )
        XCTAssertEqual(resolved, "model.layers.0.self_attn.q_proj.weight")

        let tied = QWCheckpointTensorLoader.resolveName(
            "lm_head.weight",
            defaultPrefix: "model.language_model.",
            tieWordEmbeddings: true,
            rawExists: { existing.contains($0) }
        )
        XCTAssertEqual(tied, "model.embed_tokens.weight")

        guard let url = Self.find([]) else { throw XCTSkip("selfpack fixture missing") }
        let pack = try QWQuant.Pack(url: url)
        let span = try QWCheckpointTensorLoader.span(pack, "w_test")
        XCTAssertEqual(span.rows, 256)
        XCTAssertEqual(span.cols, 1024)
        XCTAssertEqual(span.bytes, 256 * (1024 / 32) * 20)
    }
}
