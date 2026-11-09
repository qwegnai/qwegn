import SwiftUI
import AppKit

/// The popover's canvas: macOS Liquid Glass, drawn by the system.
/// The old canvas was an `NSVisualEffectView` — pre-Liquid-Glass.
public struct SystemVibrancyView: View {
    /// Reduce Transparency asks for no glass. The canvas becomes the
    /// surface it was standing in for.
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency

    public init() {}

    public var body: some View {
        ZStack {
            Color.clear
            if reduceTransparency {
                Rectangle()
                    .fill(TelemetryTheme.surface)
            } else {
                // The system glass: the blur, the saturation and the
                // tint are the system's business, not ours.
                Rectangle()
                    .fill(.clear)
                    .background(.ultraThinMaterial)
                    .glassEffect(.regular, in: .rect(cornerRadius: 0))
            }
        }
    }
}
