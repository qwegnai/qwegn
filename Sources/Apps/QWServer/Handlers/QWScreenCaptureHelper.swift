// QWScreenCaptureHelper.swift — Privacy-gated macOS ScreenCaptureKit display snapshot
// helper for local CLI (`--screen`, `/screen`), `Qwegn.app`, and loopback WebUI chat.

import CoreGraphics
import Foundation
import ImageIO
import ScreenCaptureKit
import UniformTypeIdentifiers

public enum QWScreenCaptureError: Error, CustomStringConvertible, Sendable, Equatable {
    case permissionDenied(String)
    case noDisplayAvailable
    case captureFailed(String)

    public var description: String {
        switch self {
        case .permissionDenied(let guidance):
            return guidance
        case .noDisplayAvailable:
            return "No active macOS display available for screen capture."
        case .captureFailed(let reason):
            return "Screen capture failed: \(reason)"
        }
    }
}

public struct QWScreenSnapshot: Sendable {
    public let pngData: Data
    public let width: Int
    public let height: Int
    public let displayID: UInt32

    public var dataURL: String {
        "data:image/png;base64," + pngData.base64EncodedString()
    }
}

public enum QWScreenCaptureHelper {
    public static let permissionGuidance =
        "Screen Recording permission is not granted. Enable Qwegn (or your terminal) in System Settings -> Privacy & Security -> Screen & System Audio Recording."

    private final class SyncBox<T: Sendable>: @unchecked Sendable {
        private let lock = NSLock()
        private var result: Result<T, Error>?
        func store(_ value: Result<T, Error>) { lock.withLock { result = value } }
        func take() throws -> T {
            try lock.withLock {
                guard let result else {
                    throw QWScreenCaptureError.captureFailed("Screen capture task did not complete")
                }
                return try result.get()
            }
        }
    }

    /// Captures the primary active display via `ScreenCaptureKit` when authorized by macOS TCC.
    public static func captureActiveDisplay(
        promptForPermissionIfNeeded: Bool = false,
        maxDimension: Int = 1920
    ) async throws -> QWScreenSnapshot {
        if !CGPreflightScreenCaptureAccess() {
            if promptForPermissionIfNeeded {
                _ = CGRequestScreenCaptureAccess()
            }
            guard CGPreflightScreenCaptureAccess() else {
                throw QWScreenCaptureError.permissionDenied(Self.permissionGuidance)
            }
        }
        let shareable: SCShareableContent
        do {
            shareable = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
        } catch {
            throw QWScreenCaptureError.permissionDenied(Self.permissionGuidance)
        }
        let mainID = CGMainDisplayID()
        guard let display = shareable.displays.first(where: { $0.displayID == mainID }) ?? shareable.displays.first else {
            throw QWScreenCaptureError.noDisplayAvailable
        }
        let filter = SCContentFilter(display: display, excludingWindows: [])
        let config = SCStreamConfiguration()
        let rawW = max(1, display.width)
        let rawH = max(1, display.height)
        let cap = max(256, maxDimension)
        if max(rawW, rawH) > cap {
            let scale = Double(cap) / Double(max(rawW, rawH))
            config.width = max(1, Int((Double(rawW) * scale).rounded()))
            config.height = max(1, Int((Double(rawH) * scale).rounded()))
        } else {
            config.width = rawW
            config.height = rawH
        }
        config.showsCursor = true
        let cgImage: CGImage
        do {
            cgImage = try await SCScreenshotManager.captureImage(contentFilter: filter, configuration: config)
        } catch {
            throw QWScreenCaptureError.captureFailed(error.localizedDescription)
        }
        let mutableData = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(
            mutableData as CFMutableData,
            UTType.png.identifier as CFString,
            1,
            nil
        ) else {
            throw QWScreenCaptureError.captureFailed("Failed to initialize PNG encoder")
        }
        CGImageDestinationAddImage(dest, cgImage, nil)
        guard CGImageDestinationFinalize(dest) else {
            throw QWScreenCaptureError.captureFailed("Failed to finalize PNG image")
        }
        return QWScreenSnapshot(
            pngData: mutableData as Data,
            width: cgImage.width,
            height: cgImage.height,
            displayID: display.displayID
        )
    }

    /// Synchronous CLI helper that writes the captured display PNG to an ephemeral `0o600` file.
    public static func captureActiveDisplayToTempFileSync(
        promptForPermissionIfNeeded: Bool = true,
        maxDimension: Int = 1920
    ) throws -> (fileURL: URL, snapshot: QWScreenSnapshot) {
        let sem = DispatchSemaphore(value: 0)
        let box = SyncBox<QWScreenSnapshot>()
        Task.detached(priority: .userInitiated) {
            do {
                let snap = try await captureActiveDisplay(
                    promptForPermissionIfNeeded: promptForPermissionIfNeeded,
                    maxDimension: maxDimension
                )
                box.store(.success(snap))
            } catch {
                box.store(.failure(error))
            }
            sem.signal()
        }
        sem.wait()
        let snap = try box.take()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwg-screen-\(UUID().uuidString).png")
        try snap.pngData.write(to: url, options: .atomic)
        _ = url.path.withCString { chmod($0, S_IRUSR | S_IWUSR) }
        return (url, snap)
    }
}
