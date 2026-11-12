import Foundation

/// Canonical version definition for the QWEGN engine and all companion tools (CLI, server, menu bar).
public enum QWVersion {
    /// Semantic release version string
    public static let current: String = "0.17.1"

    /// Formatted display version string (e.g., "v0.4.0-phase4")
    public static var displayString: String {
        "v\(current)"
    }
}
