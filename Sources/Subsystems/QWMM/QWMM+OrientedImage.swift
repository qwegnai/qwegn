// QWMM+OrientedImage.swift — Shared EXIF-oriented ImageIO + vImage RGB888 slab decoder.

import Accelerate
import CoreGraphics
import Foundation
import ImageIO

/// Oriented RGB888 pixel buffer produced by `QWMM`.
public struct QWOrientedRGBImage: Sendable, Equatable {
    public let rgb: [UInt8]
    public let width: Int
    public let height: Int

    public init(rgb: [UInt8], width: Int, height: Int) {
        self.rgb = rgb
        self.width = width
        self.height = height
    }
}

/// Validated image header geometry after EXIF axis orientation.
public struct QWImageHeaderInfo: Sendable, Equatable {
    public let width: Int
    public let height: Int
    public let orientation: UInt32

    public init(width: Int, height: Int, orientation: UInt32) {
        self.width = width
        self.height = height
        self.orientation = orientation
    }
}

/// Errors emitted by shared `QWMM` image inspection and RGB slab decoding.
public enum QWOrientedImageError: Error, CustomStringConvertible, Equatable {
    case emptyImage
    case invalidSource(String)
    case tooLarge(String)
    case renderFailed(String)

    public var description: String {
        switch self {
        case .emptyImage:
            return "QWMM: image has 0 width or height"
        case .invalidSource(let reason):
            return "QWMM: \(reason)"
        case .tooLarge(let reason):
            return "QWMM: \(reason)"
        case .renderFailed(let reason):
            return "QWMM: \(reason)"
        }
    }
}

/// Inspects an encoded image buffer and returns its `CGImageSource` and EXIF-oriented bounds.
public func qwInspectImageHeader(
    data: Data,
    maxSourceBytes: Int = 20 * 1024 * 1024,
    maxDimension: Int = 16_384,
    maxArea: Int64 = 67_108_864
) throws -> (source: CGImageSource, header: QWImageHeaderInfo) {
    guard !data.isEmpty else {
        throw QWOrientedImageError.invalidSource("Empty image buffer")
    }
    guard data.count <= maxSourceBytes else {
        throw QWOrientedImageError.tooLarge("Image buffer exceeds \(maxSourceBytes) bytes")
    }
    guard let source = CGImageSourceCreateWithData(data as CFData, nil) else {
        throw QWOrientedImageError.invalidSource("CGImageSource creation failed")
    }
    guard let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
          let w = (props[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue,
          let h = (props[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue else {
        throw QWOrientedImageError.invalidSource("Missing pixel dimensions in image header")
    }
    guard w > 0, h > 0 else {
        throw QWOrientedImageError.emptyImage
    }
    let rawOrient = (props[kCGImagePropertyOrientation] as? NSNumber)?.uint32Value ?? 1
    let orientation: UInt32 = (1...8).contains(rawOrient) ? rawOrient : 1
    let swapsAxes = (5...8).contains(orientation)
    let orientedW = swapsAxes ? h : w
    let orientedH = swapsAxes ? w : h
    guard orientedW <= maxDimension && orientedH <= maxDimension else {
        throw QWOrientedImageError.tooLarge("Image dimensions \(orientedW)x\(orientedH) exceed max \(maxDimension)")
    }
    let (area, overflow) = Int64(orientedW).multipliedReportingOverflow(by: Int64(orientedH))
    guard !overflow, area <= maxArea else {
        throw QWOrientedImageError.tooLarge("Image pixel area \(orientedW)x\(orientedH) exceeds max \(maxArea)")
    }
    return (source, QWImageHeaderInfo(width: orientedW, height: orientedH, orientation: orientation))
}

/// Renders a `CGImage` into an sRGB `RGB888` byte slab applying EXIF orientation via `vImage`.
public func qwRenderOrientedRGBSlab(
    cgImage: CGImage,
    targetWidth: Int,
    targetHeight: Int,
    orientation: UInt32 = 1
) throws -> QWOrientedRGBImage {
    guard targetWidth > 0, targetHeight > 0 else {
        throw QWOrientedImageError.emptyImage
    }
    guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
        throw QWOrientedImageError.renderFailed("Failed to create sRGB colorspace")
    }
    var rgba = [UInt8](repeating: 255, count: targetWidth * targetHeight * 4)
    let bitmapInfo = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
    let drew = rgba.withUnsafeMutableBytes { (ptr: UnsafeMutableRawBufferPointer) -> Bool in
        guard let context = CGContext(
            data: ptr.baseAddress,
            width: targetWidth,
            height: targetHeight,
            bitsPerComponent: 8,
            bytesPerRow: targetWidth * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ) else {
            return false
        }
        context.setFillColor(CGColor(srgbRed: 1.0, green: 1.0, blue: 1.0, alpha: 1.0))
        context.fill(CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight))
        let swapsAxes = (5...8).contains(orientation)
        let drawW = swapsAxes ? targetHeight : targetWidth
        let drawH = swapsAxes ? targetWidth : targetHeight
        let w = CGFloat(targetWidth)
        let h = CGFloat(targetHeight)
        let transform: CGAffineTransform = switch orientation {
        case 2: CGAffineTransform(a: -1, b: 0, c: 0, d: 1, tx: w, ty: 0)
        case 3: CGAffineTransform(a: -1, b: 0, c: 0, d: -1, tx: w, ty: h)
        case 4: CGAffineTransform(a: 1, b: 0, c: 0, d: -1, tx: 0, ty: h)
        case 5: CGAffineTransform(a: 0, b: -1, c: -1, d: 0, tx: w, ty: h)
        case 6: CGAffineTransform(a: 0, b: -1, c: 1, d: 0, tx: 0, ty: h)
        case 7: CGAffineTransform(a: 0, b: 1, c: 1, d: 0, tx: 0, ty: 0)
        case 8: CGAffineTransform(a: 0, b: 1, c: -1, d: 0, tx: w, ty: 0)
        default: .identity
        }
        context.concatenate(transform)
        context.interpolationQuality = .high
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: drawW, height: drawH))
        return true
    }
    guard drew else {
        throw QWOrientedImageError.renderFailed("CGContext creation failed")
    }
    var rgb = [UInt8](repeating: 0, count: targetWidth * targetHeight * 3)
    rgba.withUnsafeBytes { (srcRaw: UnsafeRawBufferPointer) in
        rgb.withUnsafeMutableBytes { (dstRaw: UnsafeMutableRawBufferPointer) in
            guard let sBase = srcRaw.baseAddress, let dBase = dstRaw.baseAddress else { return }
            var srcBuf = vImage_Buffer(
                data: UnsafeMutableRawPointer(mutating: sBase),
                height: vImagePixelCount(targetHeight),
                width: vImagePixelCount(targetWidth),
                rowBytes: targetWidth * 4
            )
            var dstBuf = vImage_Buffer(
                data: dBase,
                height: vImagePixelCount(targetHeight),
                width: vImagePixelCount(targetWidth),
                rowBytes: targetWidth * 3
            )
            if vImageConvert_RGBA8888toRGB888(&srcBuf, &dstBuf, vImage_Flags(kvImageNoFlags)) != kvImageNoError {
                let sPtr = sBase.assumingMemoryBound(to: UInt8.self)
                let dPtr = dBase.assumingMemoryBound(to: UInt8.self)
                for i in 0..<(targetWidth * targetHeight) {
                    dPtr[i * 3 + 0] = sPtr[i * 4 + 0]
                    dPtr[i * 3 + 1] = sPtr[i * 4 + 1]
                    dPtr[i * 3 + 2] = sPtr[i * 4 + 2]
                }
            }
        }
    }
    return QWOrientedRGBImage(rgb: rgb, width: targetWidth, height: targetHeight)
}

/// Decodes an encoded image buffer into an EXIF-oriented `RGB888` slab, using thumbnail downsampling when shrinking.
public func qwDecodeOrientedRGBSlab(
    data: Data,
    targetWidth: Int? = nil,
    targetHeight: Int? = nil,
    maxSourceBytes: Int = 20 * 1024 * 1024,
    maxDimension: Int = 16_384,
    maxArea: Int64 = 67_108_864,
    useThumbnailDownsample: Bool = true
) throws -> QWOrientedRGBImage {
    let (source, header) = try qwInspectImageHeader(
        data: data,
        maxSourceBytes: maxSourceBytes,
        maxDimension: maxDimension,
        maxArea: maxArea
    )
    let outW = targetWidth ?? header.width
    let outH = targetHeight ?? header.height
    guard outW > 0, outH > 0 else {
        throw QWOrientedImageError.emptyImage
    }
    let cgImage: CGImage
    let renderOrientation: UInt32
    if useThumbnailDownsample && (outW < header.width || outH < header.height) {
        let thumbOpts: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceShouldCacheImmediately: false,
            kCGImageSourceThumbnailMaxPixelSize: max(outW, outH),
        ]
        guard let thumb = CGImageSourceCreateThumbnailAtIndex(source, 0, thumbOpts as CFDictionary) else {
            throw QWOrientedImageError.renderFailed("CGImageSourceCreateThumbnailAtIndex failed")
        }
        cgImage = thumb
        renderOrientation = 1
    } else {
        guard let full = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw QWOrientedImageError.renderFailed("CGImageSourceCreateImageAtIndex failed")
        }
        cgImage = full
        renderOrientation = header.orientation
    }
    return try qwRenderOrientedRGBSlab(
        cgImage: cgImage,
        targetWidth: outW,
        targetHeight: outH,
        orientation: renderOrientation
    )
}
