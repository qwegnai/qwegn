// QWMMOrientedImageTests.swift — Tests for shared EXIF-oriented vImage RGB888 slab decoder (G2).

import CoreGraphics
import Foundation
import ImageIO
import QWMM
import UniformTypeIdentifiers
import XCTest

final class QWMMOrientedImageTests: XCTestCase {
    private func makeAsymmetricPNG(orientation: UInt32) throws -> Data {
        // 4x2 image: top-left pixel is red (255,0,0), rest are white (255,255,255)
        let width = 4
        let height = 2
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        rgba[0] = 255
        rgba[1] = 0
        rgba[2] = 0
        rgba[3] = 255
        guard let cs = CGColorSpace(name: CGColorSpace.sRGB),
              let ctx = CGContext(
                  data: &rgba, width: width, height: height, bitsPerComponent: 8,
                  bytesPerRow: width * 4, space: cs,
                  bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
              ),
              let cg = ctx.makeImage() else {
            throw XCTSkip("CoreGraphics sRGB image creation unavailable")
        }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil) else {
            throw XCTSkip("CGImageDestination unavailable")
        }
        let props: [CFString: Any] = [kCGImagePropertyOrientation: orientation]
        CGImageDestinationAddImage(dest, cg, props as CFDictionary)
        XCTAssertTrue(CGImageDestinationFinalize(dest))
        return out as Data
    }

    func testInspectHeaderAndOrientationsSwapAxesCorrectly() throws {
        for orient in UInt32(1)...UInt32(8) {
            let png = try makeAsymmetricPNG(orientation: orient)
            let (_, header) = try qwInspectImageHeader(data: png)
            let swaps = (5...8).contains(orient)
            XCTAssertEqual(header.width, swaps ? 2 : 4, "orientation \(orient)")
            XCTAssertEqual(header.height, swaps ? 4 : 2, "orientation \(orient)")
            XCTAssertEqual(header.orientation, orient)

            let decoded = try qwDecodeOrientedRGBSlab(data: png, useThumbnailDownsample: false)
            XCTAssertEqual(decoded.width, header.width)
            XCTAssertEqual(decoded.height, header.height)
            XCTAssertEqual(decoded.rgb.count, header.width * header.height * 3)
            // Verify red marker pixel lands in a distinct corner (never lost or alpha-corrupted).
            var redPixels = 0
            for i in 0..<(decoded.width * decoded.height) {
                let r = decoded.rgb[i * 3]
                let g = decoded.rgb[i * 3 + 1]
                let b = decoded.rgb[i * 3 + 2]
                if r > 200 && g < 50 && b < 50 { redPixels += 1 }
            }
            XCTAssertEqual(redPixels, 1, "orientation \(orient) lost red corner")
        }
    }

    func testDecodeRejectsOversizedBoundsAndEmptyBuffer() throws {
        XCTAssertThrowsError(try qwInspectImageHeader(data: Data())) { err in
            XCTAssertEqual(err as? QWOrientedImageError, .invalidSource("Empty image buffer"))
        }
        let png = try makeAsymmetricPNG(orientation: 6)
        XCTAssertThrowsError(try qwInspectImageHeader(data: png, maxDimension: 3)) { err in
            guard case .tooLarge = err as? QWOrientedImageError else {
                return XCTFail("Expected tooLarge, got \(err)")
            }
        }
        XCTAssertThrowsError(try qwInspectImageHeader(data: png, maxArea: 4)) { err in
            guard case .tooLarge = err as? QWOrientedImageError else {
                return XCTFail("Expected tooLarge, got \(err)")
            }
        }
    }

    func testThumbnailDownsampleResizesCorrectly() throws {
        let png = try makeAsymmetricPNG(orientation: 1)
        let decoded = try qwDecodeOrientedRGBSlab(data: png, targetWidth: 2, targetHeight: 1, useThumbnailDownsample: true)
        XCTAssertEqual(decoded.width, 2)
        XCTAssertEqual(decoded.height, 1)
        XCTAssertEqual(decoded.rgb.count, 6)
    }
}
