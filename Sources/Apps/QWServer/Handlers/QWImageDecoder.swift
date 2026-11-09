import Foundation
import CoreGraphics
import CryptoKit
import ImageIO
import QWCore
import QWEngine
import QWMM
import QWVision

public enum QWImageDecoderError: Error, CustomStringConvertible, Equatable {
    case invalidDataUrl(String)
    case base64DecodeFailed
    case fileNotFound(String)
    case imageDecodeFailed(String)
    case emptyImage

    public var description: String {
        switch self {
        case .invalidDataUrl(let str):
            return "QWImageDecoder: invalid data URI scheme in '\(str.prefix(60))...'"
        case .base64DecodeFailed:
            return "QWImageDecoder: failed to decode base64 image data"
        case .fileNotFound(let path):
            return "QWImageDecoder: image file not found at '\(path)'"
        case .imageDecodeFailed(let reason):
            return "QWImageDecoder: failed to decode image: \(reason)"
        case .emptyImage:
            return "QWImageDecoder: image has 0 width or height"
        }
    }
}

public enum QWImageDecoder {
    public struct DecodedImage: Sendable {
        public let rgb: [UInt8]
        public let width: Int
        public let height: Int

        public init(rgb: [UInt8], width: Int, height: Int) {
            self.rgb = rgb
            self.width = width
            self.height = height
        }
    }

    /// Decodes an image from an OpenAI image_url string (data URI, file path, or file:// URL)
    /// into RGB8 pixel bytes [0, 255] with extracted width and height.
    public static func decode(urlOrDataUri: String) throws -> DecodedImage {
        let trimmed = urlOrDataUri.trimmingCharacters(in: .whitespacesAndNewlines)
        let data: Data

        if trimmed.starts(with: "data:") {
            guard let commaIdx = trimmed.firstIndex(of: ",") else {
                throw QWImageDecoderError.invalidDataUrl(trimmed)
            }
            let base64Part = trimmed[trimmed.index(after: commaIdx)...]
            guard let decoded = Data(base64Encoded: String(base64Part),
                                     options: .ignoreUnknownCharacters) else {
                throw QWImageDecoderError.base64DecodeFailed
            }
            data = decoded
        } else if trimmed.starts(with: "file://") {
            guard let url = URL(string: trimmed) else {
                throw QWImageDecoderError.fileNotFound(trimmed)
            }
            guard let fileData = try? Data(contentsOf: url) else {
                throw QWImageDecoderError.fileNotFound(url.path)
            }
            data = fileData
        } else if trimmed.starts(with: "/") || trimmed.starts(with: "~") {
            let path = trimmed.expandingTildePath
            guard FileManager.default.fileExists(atPath: path),
                  let fileData = try? Data(contentsOf: URL(fileURLWithPath: path)) else {
                throw QWImageDecoderError.fileNotFound(path)
            }
            data = fileData
        } else if trimmed.starts(with: "http://") || trimmed.starts(with: "https://") {
            guard let url = URL(string: trimmed),
                  let urlData = try? Data(contentsOf: url) else {
                throw QWImageDecoderError.imageDecodeFailed("Failed to fetch remote image URL '\(trimmed)'")
            }
            data = urlData
        } else if FileManager.default.fileExists(atPath: trimmed.expandingTildePath),
                  let fileData = try? Data(contentsOf: URL(fileURLWithPath: trimmed.expandingTildePath)) {
            data = fileData
        } else {
            // Try base64 directly
            if let decoded = Data(base64Encoded: trimmed, options: .ignoreUnknownCharacters) {
                data = decoded
            } else {
                throw QWImageDecoderError.invalidDataUrl(trimmed)
            }
        }

        return try decode(data: data)
    }

    /// Default live pixel budget (1024×1024 = 1024 patches = 256 visual tokens); overridable via QW_VISION_MAX_PIXELS.
    public static var liveMaxPixels: Int {
        if let raw = ProcessInfo.processInfo.environment["QW_VISION_MAX_PIXELS"],
           let parsed = Int(raw), parsed >= QWMMConfig.minPixels {
            return min(parsed, QWMMConfig.maxPixels)
        }
        return 1_048_576
    }

    /// Decodes an image and resamples it to `QWMMConfig.smartResize(height:width:maxPixels:)` when needed.
    public static func decodeSmartResized(
        urlOrDataUri: String,
        maxPixels: Int = QWImageDecoder.liveMaxPixels
    ) throws -> (image: DecodedImage, grid: QWVisionGrid) {
        let raw = try decode(urlOrDataUri: urlOrDataUri)
        let resized = try QWMMConfig.smartResize(height: raw.height, width: raw.width, maxPixels: maxPixels)
        let grid = try QWVisionGrid(t: 1, h: resized.h / QWMMConfig.patch, w: resized.w / QWMMConfig.patch)
        if raw.width == resized.w && raw.height == resized.h {
            return (raw, grid)
        }
        return (try resampleRGB(raw, targetWidth: resized.w, targetHeight: resized.h), grid)
    }

    private static func resampleRGB(_ src: DecodedImage, targetWidth: Int, targetHeight: Int) throws -> DecodedImage {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var srcRGBA = [UInt8](repeating: 255, count: src.width * src.height * 4)
        for i in 0..<(src.width * src.height) {
            srcRGBA[i * 4 + 0] = src.rgb[i * 3 + 0]
            srcRGBA[i * 4 + 1] = src.rgb[i * 3 + 1]
            srcRGBA[i * 4 + 2] = src.rgb[i * 3 + 2]
        }
        let provider = CGDataProvider(data: Data(srcRGBA) as CFData)!
        guard let cgImage = CGImage(
            width: src.width, height: src.height,
            bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: src.width * 4,
            space: colorSpace,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent
        ) else {
            throw QWImageDecoderError.imageDecodeFailed("CGImage creation for resize failed")
        }
        var dstRGBA = [UInt8](repeating: 0, count: targetWidth * targetHeight * 4)
        dstRGBA.withUnsafeMutableBytes { ptr in
            let ctx = CGContext(
                data: ptr.baseAddress, width: targetWidth, height: targetHeight,
                bitsPerComponent: 8, bytesPerRow: targetWidth * 4, space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            )
            ctx?.interpolationQuality = .high
            ctx?.draw(cgImage, in: CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight))
        }
        var rgb = [UInt8](repeating: 0, count: targetWidth * targetHeight * 3)
        for i in 0..<(targetWidth * targetHeight) {
            rgb[i * 3 + 0] = dstRGBA[i * 4 + 0]
            rgb[i * 3 + 1] = dstRGBA[i * 4 + 1]
            rgb[i * 3 + 2] = dstRGBA[i * 4 + 2]
        }
        return DecodedImage(rgb: rgb, width: targetWidth, height: targetHeight)
    }

    private struct EncodedCacheKey: Hashable, Sendable {
        let digest: String
        let hiddenDim: Int
        let maxPixels: Int
    }

    private final class EncodedImageLRUCache: @unchecked Sendable {
        private let lock = NSLock()
        private let maxEntries = 8
        private var order: [EncodedCacheKey] = []
        private var entries: [EncodedCacheKey: (grid: QWVisionGrid, features: [Float])] = [:]

        func get(_ key: EncodedCacheKey) -> (grid: QWVisionGrid, features: [Float])? {
            lock.lock()
            defer { lock.unlock() }
            guard let val = entries[key] else { return nil }
            if let idx = order.firstIndex(of: key) {
                order.remove(at: idx)
                order.append(key)
            }
            return val
        }

        func put(_ key: EncodedCacheKey, value: (grid: QWVisionGrid, features: [Float])) {
            lock.lock()
            defer { lock.unlock() }
            if entries[key] != nil, let idx = order.firstIndex(of: key) {
                order.remove(at: idx)
            } else if order.count >= maxEntries, let oldest = order.first {
                order.removeFirst()
                entries.removeValue(forKey: oldest)
            }
            order.append(key)
            entries[key] = value
        }

        func clear() {
            lock.lock()
            defer { lock.unlock() }
            order.removeAll()
            entries.removeAll()
        }
    }

    private static let encodedCache = EncodedImageLRUCache()

    public static func clearEncodedImageCache() {
        encodedCache.clear()
    }

    private static func cacheKey(urlOrDataUri: String, hiddenDim: Int, maxPixels: Int) -> EncodedCacheKey {
        let trimmed = urlOrDataUri.trimmingCharacters(in: .whitespacesAndNewlines)
        var hasher = SHA256()
        hasher.update(data: Data(trimmed.utf8))
        if (trimmed.starts(with: "/") || trimmed.starts(with: "~")) && trimmed.count < 1024 {
            let path = trimmed.expandingTildePath
            if let attrs = try? FileManager.default.attributesOfItem(atPath: path) {
                if let size = attrs[.size] as? NSNumber {
                    hasher.update(data: Data("s:\(size.uint64Value)".utf8))
                }
                if let mtime = attrs[.modificationDate] as? Date {
                    hasher.update(data: Data("m:\(mtime.timeIntervalSince1970)".utf8))
                }
            }
        }
        let digest = hasher.finalize().map { String(format: "%02x", $0) }.joined()
        return EncodedCacheKey(digest: digest, hiddenDim: hiddenDim, maxPixels: maxPixels)
    }

    fileprivate static func cachedEncodedImage(
        urlOrDataUri: String, hiddenDim: Int, maxPixels: Int = QWImageDecoder.liveMaxPixels
    ) -> (grid: QWVisionGrid, features: [Float])? {
        encodedCache.get(cacheKey(urlOrDataUri: urlOrDataUri, hiddenDim: hiddenDim, maxPixels: maxPixels))
    }

    fileprivate static func storeEncodedImage(
        urlOrDataUri: String, hiddenDim: Int, maxPixels: Int = QWImageDecoder.liveMaxPixels,
        grid: QWVisionGrid, features: [Float]
    ) {
        encodedCache.put(
            cacheKey(urlOrDataUri: urlOrDataUri, hiddenDim: hiddenDim, maxPixels: maxPixels),
            value: (grid: grid, features: features)
        )
    }

    /// Loads vision on demand when `imageUrls` has uncached images, encodes all images into `QWVisualSpan`s,
    /// and unloads the vision head when `imageUrls` is empty to free RAM.
    public static func prepareMultimodalTurn(
        promptTokens: [Int],
        imageUrls: [String],
        hiddenDim: Int,
        visionBackend: (any QWVisionBackend)?,
        onImageEncoded: ((Int, Int, Int, Int, Double) -> Void)? = nil
    ) async throws -> (tokens: [Int], visualSpans: [QWVisualSpan], loadedOnDemand: Bool, unloadedOnText: Bool) {
        if imageUrls.isEmpty {
            if let visionBackend, visionBackend.isLoaded {
                try visionBackend.prepareForPrompt(hasImages: false)
                return (promptTokens, [], false, true)
            }
            return (promptTokens, [], false, false)
        }
        guard let visionBackend else {
            throw QWMultimodalError.badArgument("Model pack does not have an active vision tower")
        }
        let maxPixels = liveMaxPixels
        let wasLoaded = visionBackend.isLoaded
        var loadedNow = false
        var images: [(grid: QWVisionGrid, features: [Float])] = []
        let clock = ContinuousClock()
        for (idx, urlStr) in imageUrls.enumerated() {
            if let cached = cachedEncodedImage(urlOrDataUri: urlStr, hiddenDim: hiddenDim, maxPixels: maxPixels) {
                onImageEncoded?(idx + 1, imageUrls.count, cached.grid.h, cached.grid.w, 0.0)
                images.append(cached)
                continue
            }
            if !visionBackend.isLoaded {
                try visionBackend.prepareForPrompt(hasImages: true)
                loadedNow = !wasLoaded && visionBackend.isLoaded
            }
            let (decoded, grid) = try decodeSmartResized(urlOrDataUri: urlStr, maxPixels: maxPixels)
            let t0 = clock.now
            let visualTokens = try await qwEncodeVision(
                imageRGB: decoded.rgb, height: decoded.height, width: decoded.width,
                grid: grid, backend: visionBackend
            )
            let d = t0.duration(to: clock.now)
            let ms = Double(d.components.attoseconds) / 1e15 + Double(d.components.seconds) * 1e3
            storeEncodedImage(urlOrDataUri: urlStr, hiddenDim: hiddenDim, maxPixels: maxPixels,
                              grid: grid, features: visualTokens)
            onImageEncoded?(idx + 1, imageUrls.count, grid.h, grid.w, ms)
            images.append((grid: grid, features: visualTokens))
        }
        let mmPrompt = try QWMultimodalPrompt(
            promptTokens: promptTokens, images: images, hiddenDim: hiddenDim
        )
        return (mmPrompt.expandedTokens, mmPrompt.visualSpans, loadedNow, false)
    }

    /// Synchronous CLI helper: loads vision on demand for `imagePath`, encodes into `QWVisualSpan`s, and unloads immediately.
    public static func prepareOneShotMultimodalTurn(
        promptTokens: [Int],
        imagePath: String,
        modelPath: String,
        hiddenDim: Int
    ) throws -> (tokens: [Int], visualSpans: [QWVisualSpan]) {
        let controller = try QWCLIVisionController(modelPath: modelPath)
        let out = try controller.prepareTurn(
            promptTokens: promptTokens, imagePaths: [imagePath],
            currentTurnHasImage: true, hiddenDim: hiddenDim
        )
        controller.unload()
        return out
    }

    /// Decodes raw image data (JPEG, PNG, WebP, TIFF) via ImageIO into RGB8 bytes.
    public static func decode(data: Data) throws -> DecodedImage {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else {
            throw QWImageDecoderError.imageDecodeFailed("CGImageSource creation failed")
        }
        guard let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw QWImageDecoderError.imageDecodeFailed("CGImageSourceCreateImageAtIndex failed")
        }

        let width = cgImage.width
        let height = cgImage.height
        guard width > 0, height > 0 else {
            throw QWImageDecoderError.emptyImage
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        rgba.withUnsafeMutableBytes { ptr in
            let context = CGContext(
                data: ptr.baseAddress,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width * 4,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            )
            context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        }

        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for i in 0..<(width * height) {
            rgb[i * 3 + 0] = rgba[i * 4 + 0]
            rgb[i * 3 + 1] = rgba[i * 4 + 1]
            rgb[i * 3 + 2] = rgba[i * 4 + 2]
        }

        return DecodedImage(rgb: rgb, width: width, height: height)
    }
}

/// Prompt-scoped vision residency controller for interactive CLI (`qwg repl`) and one-shot CLI (`qwg run`).
public final class QWCLIVisionController: @unchecked Sendable {
    private let backend: QWVisionOnDemandBackend

    public init(modelPath: String) throws {
        self.backend = try QWVisionOnDemandBackend(modelPath: modelPath)
    }

    public var isLoaded: Bool { backend.isLoaded }

    public func unload() {
        backend.unload()
    }

    /// Encodes `imagePaths` when present, loading vision on demand and unloading when `currentTurnHasImage == false`.
    public func prepareTurn(
        promptTokens: [Int],
        imagePaths: [String],
        currentTurnHasImage: Bool,
        hiddenDim: Int
    ) throws -> (tokens: [Int], visualSpans: [QWVisualSpan]) {
        if imagePaths.isEmpty {
            try backend.prepareForPrompt(hasImages: false)
            return (promptTokens, [])
        }
        let maxPixels = QWImageDecoder.liveMaxPixels
        var images: [(grid: QWVisionGrid, features: [Float])] = []
        for path in imagePaths {
            if let cached = QWImageDecoder.cachedEncodedImage(
                urlOrDataUri: path, hiddenDim: hiddenDim, maxPixels: maxPixels
            ) {
                images.append(cached)
                continue
            }
            if !backend.isLoaded {
                try backend.prepareForPrompt(hasImages: true)
            }
            let (decoded, grid) = try QWImageDecoder.decodeSmartResized(urlOrDataUri: path, maxPixels: maxPixels)
            let patches = try qwBuildPatchSlab(
                imageRGB: decoded.rgb, height: decoded.height, width: decoded.width, grid: grid
            )
            let visualTokens = try backend.encodeSync(
                patches: patches, grid: grid.plan, bins: grid.interpBins, posIds: grid.ropePositions
            )
            QWImageDecoder.storeEncodedImage(
                urlOrDataUri: path, hiddenDim: hiddenDim, maxPixels: maxPixels,
                grid: grid, features: visualTokens
            )
            images.append((grid: grid, features: visualTokens))
        }
        if !currentTurnHasImage {
            try backend.prepareForPrompt(hasImages: false)
        }
        let mmPrompt = try QWMultimodalPrompt(
            promptTokens: promptTokens, images: images, hiddenDim: hiddenDim
        )
        return (mmPrompt.expandedTokens, mmPrompt.visualSpans)
    }
}
