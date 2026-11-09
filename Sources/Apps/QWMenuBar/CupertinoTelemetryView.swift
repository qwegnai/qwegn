import SwiftUI
import AppKit
import QWServer

/// The Menu Bar popover: Liquid Glass chrome, telemetry content.
///
/// The chrome is glass, the cards are content, and the body scrolls.
public struct CupertinoTelemetryView: View {
    @ObservedObject public var telemetry: TelemetryObservable
    public var onOpenWebUI: () -> Void
    public var onOpenLogs: () -> Void
    public var onOpenSettings: () -> Void
    public var onQuit: () -> Void

    @State var modelRowWidth: CGFloat = 408
    @State var modelTitleWidth: CGFloat = 110

    /// The system's accessibility settings. L-M7: the popover honours
    /// all three. The values are read here so the cards can use them
    /// too — they are extensions of this view.
    @Environment(\.accessibilityReduceTransparency) internal var reduceTransparency: Bool
    @Environment(\.accessibilityReduceMotion) internal var reduceMotion: Bool
    @Environment(\.colorSchemeContrast) internal var contrast: ColorSchemeContrast

    public init(
        telemetry: TelemetryObservable,
        onOpenWebUI: (() -> Void)? = nil,
        onOpenLogs: @escaping () -> Void = {},
        onOpenSettings: @escaping () -> Void = {},
        onQuit: @escaping () -> Void = { NSApplication.shared.terminate(nil) }
    ) {
        self.telemetry = telemetry
        self.onOpenWebUI = onOpenWebUI ?? { telemetry.openWebDashboard() }
        self.onOpenLogs = onOpenLogs
        self.onOpenSettings = onOpenSettings
        self.onQuit = onQuit
    }

    public var body: some View {
        // One container for the glass: the canvas, the header and the
        // footer read as one surface, not three.
        //
        // The canvas is NOT inside the glass container. A container
        // composites its glass members as one layer, and that layer is
        // drawn above the container's plain content: with the canvas
        // inside it, the canvas covered the whole panel and the content
        // underneath it — every card, every label — was never seen.
        // The canvas is the glass behind the content; the content's own
        // glass (the buttons) is grouped in its own container.
        ZStack {
            SystemVibrancyView()
                .edgesIgnoringSafeArea(.all)

            GlassEffectContainer(spacing: 10) {
                // The panel shows everything it has: it is a panel, not
                // a document, so there is nothing below the fold and
                // nothing to scroll to. The tiles take the room the
                // panel has, so the panel is full and every tile is the
                // same size.
                VStack(spacing: 10) {
                    headerSection
                    failureBanner
                    loadedWeightsCard
                    metricGrid
                    traceStrip
                    footerDock
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 12)
                .frame(maxWidth: .infinity)
                // The glass is the chrome; the content must not inherit
                // its tint.
                .tint(.clear)
            }
        }
        .frame(width: 440)
        .frame(minHeight: 530, maxHeight: 580)
    }

    /// A failure the user can see. A rejected load is a rejected load.
    @ViewBuilder
    var failureBanner: some View {
        if let error = telemetry.displayedError {
            HStack(spacing: 6) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.system(size: 10))
                    .foregroundColor(TelemetryTheme.failure)
                Text(error)
                    .font(TelemetryTheme.ui(10))
                    .foregroundColor(TelemetryTheme.primary(increasedContrast: contrast == .increased))
                    .lineLimit(2)
                Spacer(minLength: 0)
            }
            .padding(8)
            .background(TelemetryTheme.failure.opacity(0.12))
            .cornerRadius(TelemetryTheme.Radius.large)
            .overlay(RoundedRectangle(cornerRadius: TelemetryTheme.Radius.large)
                .stroke(TelemetryTheme.failure.opacity(0.35), lineWidth: 0.5))
        }
    }

    /// The status dot: the connection first, then the engine's phase.
    var statusIndicatorColor: Color {
        switch telemetry.connection {
        case .unknown:
            return TelemetryTheme.unknown
        case .disconnected:
            return TelemetryTheme.failure
        case .connected:
            switch telemetry.enginePhase {
            case .some(.loaded): return TelemetryTheme.active
            case .some(.loading), .some(.unloading): return TelemetryTheme.warning
            case .some(.failed): return TelemetryTheme.failure
            // A server that is up with no model is on standby, which is
            // not the same as a state nobody has observed.
            case .some(.unloaded): return TelemetryTheme.warning
            case .none: return TelemetryTheme.unknown
            }
        }
    }

    /// Whether the engine is working right now. The halo pulses while it
    /// does, and only while it does.
    var isGenerating: Bool {
        guard let phase = telemetry.dashboard?.generation else { return false }
        return phase != .idle
    }

    // MARK: - 1. Header Section
}
