// TelemetryCardPrimitives.swift — Reusable metric cells, buttons, and AppKit menu bridge

import AppKit
import QWCore
import QWServer
import SwiftUI

extension CupertinoTelemetryView {
    var horizontalHairline: some View {
        Rectangle()
            .fill(TelemetryTheme.separator(increasedContrast: contrast == .increased))
            .frame(height: 0.5)
    }

    var verticalHairline: some View {
        Rectangle()
            .fill(TelemetryTheme.separator(increasedContrast: contrast == .increased))
            .frame(width: 0.5)
    }

    func heroMetricColumn(
        title: String,
        card: QWDashboardCard?,
        tint: Color,
        valueTemplate: String = "999.9"
    ) -> some View {
        let modeBadge = extractRateModeBadge(from: card?.title)
        let isCached = card?.value == "from cache"
        return VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Text(title)
                    .font(TelemetryTheme.ui(11, .medium))
                    .foregroundStyle(TelemetryTheme.secondary(increasedContrast: contrast == .increased))
                    .lineLimit(1)
                Spacer(minLength: 0)
                if let modeBadge, card?.value != nil {
                    Text(modeBadge)
                        .font(TelemetryTheme.ui(9.5, .medium))
                        .foregroundStyle(TelemetryTheme.quiet(increasedContrast: contrast == .increased))
                        .frame(minWidth: 28)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 1)
                        .background(TelemetryTheme.rowSubstrate)
                        .clipShape(.rect(cornerRadius: 4))
                }
            }
            .frame(minHeight: 15)

            ZStack(alignment: .leading) {
                HStack(alignment: .firstTextBaseline, spacing: 4) {
                    Text(valueTemplate)
                        .font(TelemetryTheme.numeric(20, .bold))
                        .lineLimit(1)
                    Text(card?.unit ?? "tok/s")
                        .font(TelemetryTheme.ui(11, .medium))
                        .lineLimit(1)
                }
                .hidden()
                .accessibilityHidden(true)

                if isCached {
                    Text("From cache")
                        .font(TelemetryTheme.ui(16, .semibold))
                        .foregroundStyle(TelemetryTheme.cool)
                        .lineLimit(1)
                } else {
                    HStack(alignment: .firstTextBaseline, spacing: 4) {
                        ZStack(alignment: .trailing) {
                            Text(valueTemplate)
                                .font(TelemetryTheme.numeric(20, .bold))
                                .lineLimit(1)
                                .hidden()
                                .accessibilityHidden(true)
                            Text(card?.value ?? "—")
                                .font(TelemetryTheme.numeric(20, .bold))
                                .foregroundStyle(card?.value == nil ? TelemetryTheme.quiet(increasedContrast: contrast == .increased) : tint)
                                .lineLimit(1)
                                .minimumScaleFactor(0.8)
                        }
                        Text(card?.unit ?? "tok/s")
                            .font(TelemetryTheme.ui(11, .medium))
                            .foregroundStyle(
                                card?.value == nil
                                    ? TelemetryTheme.quiet(increasedContrast: contrast == .increased)
                                    : TelemetryTheme.secondary(increasedContrast: contrast == .increased)
                            )
                            .lineLimit(1)
                    }
                }
            }
        }
        .padding(.horizontal, 12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .help(card?.reason ?? "")
    }

    func inlineMetricCell(
        label: String,
        valueText: String,
        isAvailable: Bool,
        tint: Color,
        help: String?
    ) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(TelemetryTheme.ui(10, .medium))
                .foregroundStyle(TelemetryTheme.quiet(increasedContrast: contrast == .increased))
                .lineLimit(1)
            Text(valueText)
                .font(TelemetryTheme.numeric(12, .semibold))
                .foregroundStyle(isAvailable ? tint : TelemetryTheme.quiet(increasedContrast: contrast == .increased))
                .lineLimit(1)
                .minimumScaleFactor(0.78)
        }
        .padding(.horizontal, 11)
        .frame(maxWidth: .infinity, alignment: .leading)
        .help(help ?? "")
    }

    func siliconGaugeCell(
        title: String,
        card: QWDashboardCard?,
        tint: Color,
        helpSuffix: String? = nil
    ) -> some View {
        let pct = (card?.unit == "%") ? card?.number : nil
        let helpText = [card?.reason, helpSuffix.map { "Pressure: \($0)" }]
            .compactMap { $0 }
            .filter { !$0.isEmpty }
            .joined(separator: " · ")

        return VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 4) {
                Text(title)
                    .font(TelemetryTheme.ui(10, .medium))
                    .foregroundStyle(TelemetryTheme.quiet(increasedContrast: contrast == .increased))
                    .lineLimit(1)
                Spacer(minLength: 2)
                Text(formattedCardValue(card))
                    .font(TelemetryTheme.numeric(11.5, .semibold))
                    .foregroundStyle(card?.value == nil ? TelemetryTheme.quiet(increasedContrast: contrast == .increased) : tint)
                    .lineLimit(1)
                    .minimumScaleFactor(0.85)
            }
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule()
                        .fill(TelemetryTheme.hairline(increasedContrast: contrast == .increased))
                    if let pct {
                        Capsule()
                            .fill(tint)
                            .frame(width: max(2, geo.size.width * min(1.0, max(0.0, pct / 100.0))))
                    }
                }
            }
            .frame(height: 3)
        }
        .padding(.horizontal, 11)
        .frame(maxWidth: .infinity, alignment: .leading)
        .help(helpText)
    }

    func extractRateModeBadge(from title: String?) -> String? {
        guard let title,
              let open = title.firstIndex(of: "("),
              let close = title[open...].firstIndex(of: ")") else {
            return nil
        }
        let inner = title[title.index(after: open)..<close].lowercased()
        switch inner {
        case "average": return "avg"
        case "live": return "live"
        case "last": return "last"
        default: return String(inner)
        }
    }

    func formattedContextSummary(_ card: QWDashboardCard?) -> String {
        let activeText: String? = {
            guard let card, card.value != nil else { return nil }
            if let num = card.number {
                let n = Int(num.rounded())
                return n >= 10_000
                    ? QWMenuBarStatusFormatter.formatFixedActiveTokens(n)
                    : QWTokenCountFormatter.format(n)
            }
            return card.value
        }()
        let budgetText = telemetry.contextBudget.map { QWMenuBarStatusFormatter.formatCompactBudget($0) }
        func padActive(_ raw: String) -> String {
            let width = QWMenuBarStatusFormatter.activeContextFieldWidth
            return raw.count >= width ? raw : String(repeating: " ", count: width - raw.count) + raw
        }
        switch (activeText, budgetText) {
        case let (.some(active), .some(budget)):
            return "\(padActive(active)) / \(budget)"
        case let (.some(active), .none):
            return "\(padActive(active)) tok"
        case let (.none, .some(budget)):
            return "\(padActive("—")) / \(budget)"
        case (.none, .none):
            return "—"
        }
    }

    func engineMemoryHelp(arena: QWDashboardCard?, resident: QWDashboardCard?) -> String {
        var notes: [String] = []
        if let val = arena?.value, let unit = arena?.unit {
            notes.append("Metal arena: \(val) \(unit)")
        }
        if let val = resident?.value, let unit = resident?.unit {
            notes.append("Resident weights: \(val) \(unit)")
        }
        if notes.isEmpty {
            return arena?.reason ?? resident?.reason ?? ""
        }
        return notes.joined(separator: " · ")
    }

    func memoryBarTint(pressureCard: QWDashboardCard?) -> Color {
        switch pressureCard?.value?.lowercased() {
        case "critical": return TelemetryTheme.failure
        case "warn", "warning": return TelemetryTheme.warning
        default: return TelemetryTheme.active
        }
    }

    func formattedCardValue(_ card: QWDashboardCard?, minValueWidth: Int = 0) -> String {
        guard let card, let val = card.value else { return "—" }
        let padded = (minValueWidth > 0 && val.count < minValueWidth)
            ? String(repeating: " ", count: minValueWidth - val.count) + val
            : val
        if let unit = card.unit, !unit.isEmpty {
            return "\(padded) \(unit)"
        }
        return padded
    }

    func factPill(_ text: String, tint: Color) -> some View {
        Text(text)
            .font(TelemetryTheme.ui(10.5, .medium))
            .foregroundStyle(tint)
            .lineLimit(1)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(tint.opacity(0.12))
            .clipShape(.rect(cornerRadius: TelemetryTheme.Radius.small))
    }

    func marqueeFactPill(_ text: String, tint: Color, help: String? = nil) -> some View {
        MarqueeText(
            text: text,
            fontSize: 10.5,
            weight: .medium,
            nsWeight: .medium,
            color: tint,
            shrinksToFit: true
        )
        .padding(.horizontal, 6)
        .padding(.vertical, 2)
        .background(tint.opacity(0.12))
        .clipShape(.rect(cornerRadius: TelemetryTheme.Radius.small))
        .help(help ?? text)
    }

    func legendChip(_ title: String, color: Color) -> some View {
        HStack(spacing: 4) {
            Circle().fill(color).frame(width: 6, height: 6)
            Text(title)
                .font(TelemetryTheme.ui(10.5))
                .foregroundStyle(TelemetryTheme.secondary(increasedContrast: contrast == .increased))
        }
    }

    func headerActionButton(
        title: String,
        icon: String,
        tint: Color,
        busy: Bool,
        disabled: Bool,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            HStack(spacing: 4) {
                if busy {
                    ProgressView().controlSize(.mini)
                } else {
                    Image(systemName: icon)
                        .font(.system(size: 10.5, weight: .semibold))
                }
                Text(title)
                    .font(TelemetryTheme.ui(11, .semibold))
                    .lineLimit(1)
            }
            .foregroundStyle(tint)
            .padding(.horizontal, 9)
            .frame(height: 26)
            .background(tint.opacity(0.12))
            .clipShape(.rect(cornerRadius: TelemetryTheme.Radius.medium))
            .overlay(
                RoundedRectangle(cornerRadius: TelemetryTheme.Radius.medium)
                    .stroke(tint.opacity(0.32), lineWidth: 0.75)
            )
        }
        .buttonStyle(.plain)
        .disabled(disabled)
        .opacity(disabled ? 0.5 : 1.0)
        .accessibilityLabel(title)
    }

    func lifecycleButton(
        title: String,
        icon: String,
        tint: Color,
        busy: Bool,
        disabled: Bool,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            HStack(spacing: 4) {
                if busy {
                    ProgressView().controlSize(.mini)
                } else {
                    Image(systemName: icon)
                        .font(.system(size: 10.5))
                }
                Text(title)
                    .font(TelemetryTheme.ui(11, .semibold))
                    .lineLimit(1)
            }
            .foregroundStyle(tint)
            .padding(.horizontal, 9)
            .frame(height: 25)
            .background(tint.opacity(0.12))
            .clipShape(.rect(cornerRadius: TelemetryTheme.Radius.small))
            .overlay(
                RoundedRectangle(cornerRadius: TelemetryTheme.Radius.small)
                    .stroke(tint.opacity(0.30), lineWidth: 0.75)
            )
        }
        .buttonStyle(.plain)
        .disabled(disabled)
        .opacity(disabled ? 0.45 : 1.0)
    }

    func dockButton(
        title: String,
        icon: String,
        shortcut: String,
        isDestructive: Bool = false,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            HStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.system(size: 10.5))
                Text(title)
                    .font(TelemetryTheme.ui(11, .medium))
                    .lineLimit(1)
                Text(shortcut)
                    .font(TelemetryTheme.numeric(9, .medium))
                    .foregroundStyle(isDestructive ? TelemetryTheme.failure.opacity(0.75) : TelemetryTheme.quiet(increasedContrast: contrast == .increased))
                    .padding(.horizontal, 3.5)
                    .padding(.vertical, 1)
                    .background(TelemetryTheme.rowSubstrate)
                    .clipShape(.rect(cornerRadius: 4))
            }
            .foregroundStyle(isDestructive ? TelemetryTheme.failure : TelemetryTheme.primary(increasedContrast: contrast == .increased))
            .frame(maxWidth: .infinity)
            .frame(height: 28)
            .background(TelemetryTheme.substrate)
            .clipShape(.rect(cornerRadius: TelemetryTheme.Radius.medium))
            .overlay(
                RoundedRectangle(cornerRadius: TelemetryTheme.Radius.medium)
                    .stroke(TelemetryTheme.hairline(increasedContrast: contrast == .increased), lineWidth: 0.5)
            )
        }
        .buttonStyle(.plain)
        .help("\(title) (\(shortcut))")
    }
}

@MainActor var menuTargetsAssocKey: UInt8 = 0

@MainActor
final class MenuClosureTarget: NSObject {
    private let handler: () -> Void
    init(_ handler: @escaping () -> Void) { self.handler = handler }
    @objc func invoke(_ sender: Any?) { handler() }
}

struct MenuAnchorBridge: NSViewRepresentable {
    var onResolve: (NSView) -> Void
    func makeNSView(context: Context) -> NSView { NSView() }
    func updateNSView(_ nsView: NSView, context: Context) {
        DispatchQueue.main.async { onResolve(nsView) }
    }
}
