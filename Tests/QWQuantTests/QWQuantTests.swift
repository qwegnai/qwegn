import XCTest
@testable import QWQuant
import QWCore

final class QWQuantTests: XCTestCase {

    func testModelDirectoryAndLegacyPaths() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-model-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let weights = root.appendingPathComponent("model.qwp")
        try FileManager.default.copyItem(at: Self.fixtures, to: weights)
        let location = try QWModelLocation(url: root)
        XCTAssertEqual(location.weights, weights)
        XCTAssertEqual(location.assets, root.appendingPathComponent("assets", isDirectory: true))
        let pack = try QWQuant.Pack(url: root)
        XCTAssertEqual(pack.url, weights)
        XCTAssertEqual(try pack.dequantQ4g("w_test"),
                       try QWQuant.Pack(url: weights).dequantQ4g("w_test"))
        let legacy = try QWModelLocation(url: weights)
        XCTAssertEqual(legacy.assets.path, weights.path + ".assets")
        let override = root.appendingPathComponent("external-assets")
        XCTAssertEqual(try QWModelLocation(url: root, assetsOverride: override).assets, override)
        try FileManager.default.removeItem(at: weights)
        XCTAssertThrowsError(try QWModelLocation(url: root))
        XCTAssertThrowsError(try QWQuant.Pack(url: root))
    }

    static var fixtures: URL {
        var d = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        for _ in 0..<8 {
            let c = d.appendingPathComponent("Tests/QWQuantTests/fixtures/selfpack.qwp")
            if FileManager.default.fileExists(atPath: c.path) { return c }
            d = d.deletingLastPathComponent()
        }
        fatalError("selfpack.qwp fixture missing — run: python3 Scripts/quantize/pack.py --selfpack Tests/QWQuantTests/fixtures/selfpack.qwp")
    }

    static func readNpyF32(_ url: URL) throws -> [Float] {
        let d = try Data(contentsOf: url)
        let magic: [UInt8] = [0x93] + Array("NUMPY".utf8)
        precondition(d.count > 128 && Array(d[d.startIndex..<d.startIndex + 6]) == magic, "bad npy")
        let major = d[d.startIndex + 6]
        precondition(major == 1, "npy v2 unsupported")
        let hdrLen = Int(d[d.startIndex + 8]) | (Int(d[d.startIndex + 9]) << 8)
        let hdr = String(decoding: d[(d.startIndex + 10)..<(d.startIndex + 10 + hdrLen)], as: UTF8.self)
        precondition(hdr.contains("<f4"), "npy dtype not f32")
        let base = d.startIndex + 10 + hdrLen
        let n = (d.count - (10 + hdrLen)) / 4
        var out = [Float](repeating: 0, count: n)
        d.withUnsafeBytes { raw in
            let p = raw.baseAddress!.advanced(by: base).assumingMemoryBound(to: Float.self)
            for i in 0..<n { out[i] = p[i] }
        }
        return out
    }

    func testSwiftReaderMatchesNumpyDequant() throws {
        let pack = try QWQuant.Pack(url: Self.fixtures)
        let dq4 = try pack.dequantQ4g("w_test")
        let ref4 = try Self.readNpyF32(URL(fileURLWithPath: Self.fixtures.path + ".dq4.npy"))
        XCTAssertEqual(dq4.count, ref4.count)
        var bad4 = 0, max4 = 0.0
        var firstBad4 = -1
        for i in 0..<dq4.count where dq4[i].bitPattern != ref4[i].bitPattern {
            bad4 += 1; max4 = max(max4, abs(Double(dq4[i]) - Double(ref4[i])))
            if firstBad4 < 0 { firstBad4 = i }
        }
        XCTAssertEqual(bad4, 0, "hq4_g32_m5 dequant must be BIT-identical to numpy fp32")
        XCTAssertNil(try pack.entry("w_test").q8Off, "new packs must not carry q8_off")
        XCTAssertThrowsError(try pack.dequantQ8("w_test")) { err in
            guard case QWQuantError.notQuantPlane = err else { return XCTFail("\(err)") }
        }
    }

    func testRawBf16Plane() throws {
        let pack = try QWQuant.Pack(url: Self.fixtures)
        let e = try pack.entry("norm_test")
        XCTAssertTrue(e.isRawBf16, "norm_test must be a raw plane")
        XCTAssertFalse(e.quantized)
        let got = try pack.rawBf16("norm_test")
        let ref = try Self.readNpyF32(URL(fileURLWithPath: Self.fixtures.path + ".raw.npy"))
        XCTAssertEqual(got.count, ref.count)
        var bad = 0
        for i in 0..<got.count where got[i].bitPattern != ref[i].bitPattern {
            bad += 1
        }
        XCTAssertEqual(bad, 0, "verbatim-bf16 plane must be BIT-identical to numpy")
        XCTAssertThrowsError(try pack.dequantQ4g("norm_test")) { err in
            guard case QWQuantError.notQuantPlane = err else { return XCTFail("\(err)") }
        }
        XCTAssertThrowsError(try pack.rawBf16("w_test")) { err in
            guard case QWQuantError.notRawPlane = err else { return XCTFail("\(err)") }
        }
    }

    func testGeometryGuard() throws {
        let data = try Data(contentsOf: Self.fixtures)
        guard let range = data.range(of: Data("\"geometry_hash\": \"".utf8))
            ?? data.range(of: Data("\"geometry_hash\":\"".utf8)) else {
            return XCTFail("geometry_hash not in header (pack.py regression)")
        }
        var bad = data
        let at = range.upperBound
        bad[at] = bad[at] == UInt8(ascii: "0") ? UInt8(ascii: "1") : UInt8(ascii: "0")
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("bad.qwp")
        try bad.write(to: tmp)
        XCTAssertThrowsError(try QWQuant.Pack(url: tmp)) { err in
            guard case QWQuantError.geometryMismatch = err else { return XCTFail("\(err)") }
        }
    }

    func testF16ConversionParity() {
        let cases: [(UInt16, Double)] = [
            (0x3C00, 1.0), (0xBC00, -1.0), (0x3C01, 1.0009765625),
            (0x0001, 5.960464477539063e-8), (0x7BFF, 65504.0),
            (0x33C0, 0.2421875), (0x3555, 0.333251953125),
        ]
        for (bits, want) in cases {
            XCTAssertEqual(QWQuant.f16bits(bits), want, accuracy: abs(want) * 1e-12 + 1e-21,
                           "f16 bits 0x\(String(bits, radix: 16))")
        }
        XCTAssertTrue(QWQuant.f16bits(0x7C00) == .infinity)
        XCTAssertTrue(QWQuant.f16bits(0x7C01).isNaN)
    }

    func testGeometryHashCrossLanguage() throws {
        let data = try Data(contentsOf: Self.fixtures)
        let tail = String(decoding: data.suffix(2048), as: UTF8.self)
        XCTAssertTrue(tail.contains(QWQuant.geometryHash),
                      "python geometry_hash digest ≠ Swift digest — formulas drifted")
    }


    private static var expertFixtures: URL {
        var d = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        for _ in 0..<8 {
            let c = d.appendingPathComponent("Tests/QWQuantTests/fixtures/selfpack_expert.qwp")
            if FileManager.default.fileExists(atPath: c.path) { return c }
            d = d.deletingLastPathComponent()
        }
        fatalError("selfpack_expert.qwp missing — run: python3 Scripts/quantize/pack.py --selfpack-expert Tests/QWQuantTests/fixtures/selfpack_expert.qwp")
    }



    private func QWVisionGoldenConfigURL() -> URL? {
        var dir = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        for _ in 0..<8 {
            let cand = dir.appendingPathComponent("Scripts/oracle/config_flashnext_golden.json")
            if FileManager.default.fileExists(atPath: cand.path) { return cand }
            dir = dir.deletingLastPathComponent()
        }
        return nil
    }
}
