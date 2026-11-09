// TelemetryTheme.swift — The design tokens the MenuBar draws with.
// One place for the palette, the type and the metrics, so a card in the
// popover and a card in the web dashboard read as the same instrument.

import SwiftUI
import AppKit

/// The popover's design tokens.
public enum TelemetryTheme {
    // MARK: - Colour

    /// The canvas behind the popover, when the glass is turned off.
    public static let surface = Color(hex: "#111318")
    /// The accent: controls, focus, the decode trace.
    public static let accent = Color(hex: "#0A84FF")
    /// The prefill trace.
    public static let prefill = Color(hex: "#FFD60A")
    /// A cool accent: the cache, the CPU.
    public static let cool = Color(hex: "#64D2FF")
    /// The quiet grey: memory.
    public static let neutral = Color(hex: "#98989D")
    /// A lighter blue: the selected row.
    public static let highlight = Color(hex: "#AAC7FF")
    /// The ink of the emblem.
    public static let logoInk = Color(hex: "#F3F4F6")
    /// An engine that is running.
    public static let active = Color(hex: "#30D158")
    /// A warning: memory pressure, a warm-up.
    public static let warning = Color(hex: "#FF9F0A")
    /// A failure.
    public static let failure = Color(hex: "#FF453A")
    /// A state nobody has observed.
    public static let unknown = Color(hex: "#8B91A0")

    // MARK: - Type

    /// The type for the interface's own words.
    public static let label = Font.system(size: 10, weight: .medium)
    public static let body = Font.system(size: 11, weight: .regular)
    public static let headline = Font.system(size: 13, weight: .semibold)
    /// The type for numbers: monospaced, so a digit does not move.
    public static let metric = Font.system(size: 11, weight: .medium, design: .monospaced)

    /// The interface face. DESIGN.md names Inter; a machine without it
    /// uses the system face, so the popover is readable everywhere.
    public static func ui(_ size: CGFloat, _ weight: Font.Weight = .regular) -> Font {
        guard let face = Face.inter else {
            return .system(size: size, weight: weight)
        }
        return .custom(face, size: size).weight(weight)
    }

    /// The numeric face. DESIGN.md names JetBrains Mono; a machine
    /// without it uses the system's own monospace. Either way the digits
    /// are tabular, so a value that streams does not make the row jump.
    public static func numeric(_ size: CGFloat, _ weight: Font.Weight = .regular) -> Font {
        guard let face = Face.jetbrainsMono else {
            return .system(size: size, weight: weight, design: .monospaced).monospacedDigit()
        }
        return .custom(face, size: size).weight(weight).monospacedDigit()
    }

    /// The faces DESIGN.md names, and whether this machine has them.
    /// A missing face is a fact the theme knows, not a silent
    /// substitution the reader cannot see.
    enum Face {
        static let inter = firstAvailable(["Inter", "Inter-Regular", "Inter Display"])
        static let jetbrainsMono = firstAvailable(["JetBrains Mono", "JetBrainsMono Nerd Font",
                                                    "JetBrainsMonoNL Nerd Font"])

        private static func firstAvailable(_ names: [String]) -> String? {
            names.first { NSFont(name: $0, size: 1) != nil }
        }
    }

    // MARK: - Metrics

    public enum Radius {
        public static let small: CGFloat = 4
        public static let medium: CGFloat = 6
        public static let large: CGFloat = 8
    }

    public enum Space {
        public static let xSmall: CGFloat = 2
        public static let small: CGFloat = 4
        public static let medium: CGFloat = 8
        public static let large: CGFloat = 12
    }

    /// The hairline the cards are drawn with.
    public static let hairline = Color.white.opacity(0.08)
    /// A card's substrate on the glass.
    public static let substrate = Color.white.opacity(0.05)
    /// A row's substrate inside a card.
    public static let rowSubstrate = Color.white.opacity(0.03)

    // MARK: - Accessibility

    /// The hairline, honouring Increased Contrast. A stronger edge is
    /// what the user asked for when they ask for more contrast.
    public static func hairline(increasedContrast: Bool) -> Color {
        increasedContrast ? Color.white.opacity(0.28) : hairline
    }

    /// The quieter text, honouring Increased Contrast.
    public static func secondary(increasedContrast: Bool) -> Color {
        increasedContrast ? Color.white.opacity(0.88) : Color.white.opacity(0.45)
    }

    /// The quietest text — the ages and the reasons, honouring Increased
    /// Contrast.
    public static func quiet(increasedContrast: Bool) -> Color {
        increasedContrast ? Color.white.opacity(0.75) : Color.white.opacity(0.35)
    }

    /// A separator between two values, honouring Increased Contrast.
    public static func separator(increasedContrast: Bool) -> Color {
        increasedContrast ? Color.white.opacity(0.45) : Color.white.opacity(0.20)
    }

    /// The text on the canvas, honouring Increased Contrast.
    public static func primary(increasedContrast: Bool) -> Color {
        increasedContrast ? Color.white : Color.white.opacity(0.95)
    }
}
