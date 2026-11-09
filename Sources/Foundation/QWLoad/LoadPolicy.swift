// LoadPolicy.swift — one place turning the model descriptor (config.json)
// plus env kill-switches into load behavior. Defaults are config-derived;
// env vars may only DISABLE (ship law), never invent a default.
import Foundation
import QWCore

public struct LoadPolicy: Sendable {
    public var directIO: Bool
    public var zeroCopyMaps: Bool
    public var wireByDefault: Bool
    public var wireForced: Bool
    public var wireSkipped: Bool
    public var wireHeadroomMarginBytes: Int

    public init(descriptor: QWModelDescriptor,
                env: [String: String] = ProcessInfo.processInfo.environment) {
        directIO = env["QW_PLANE_BUFFERED"] != "1"
        zeroCopyMaps = env["QW_PLANE_MAP_DISABLE"] != "1"
        wireByDefault = false
        wireForced = env["QW_WIRE_FORCE"] == "1" || env["QW_FN_WIRE_FORCE"] == "1"
        wireSkipped = env["QW_WIRE_SKIP"] == "1" || env["QW_FN_WIRE_SKIP"] == "1"
        wireHeadroomMarginBytes = 6 << 30
    }
}
