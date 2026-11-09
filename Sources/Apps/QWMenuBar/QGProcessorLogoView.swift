// QGProcessorLogoView.swift — logo mark and Color hex helper
// (split from CupertinoTelemetryView.swift).
import SwiftUI
import AppKit
import QWServer

public struct QGProcessorLogoView: View {
    public init() {}

    /// Draws the canonical 600×600 QG Processor microchip vector mark into a top-left-origin CoreGraphics context.
    public static func draw(in ctx: CGContext, size: CGSize, strokeColor: CGColor) {
        ctx.saveGState()
        let scale = min(size.width, size.height) / 600.0
        ctx.scaleBy(x: scale, y: scale)

        ctx.setStrokeColor(strokeColor)
        ctx.setLineWidth(24)
        ctx.setLineCap(.round)
        ctx.setLineJoin(.round)

        // Bus Pins
        for x: CGFloat in [220, 260, 300, 340, 380] {
            ctx.move(to: CGPoint(x: x, y: 62))
            ctx.addLine(to: CGPoint(x: x, y: 108))
            ctx.move(to: CGPoint(x: x, y: 492))
            ctx.addLine(to: CGPoint(x: x, y: 538))
        }
        ctx.move(to: CGPoint(x: 62, y: 220)); ctx.addLine(to: CGPoint(x: 108, y: 220))
        ctx.move(to: CGPoint(x: 62, y: 260)); ctx.addLine(to: CGPoint(x: 108, y: 260))
        ctx.move(to: CGPoint(x: 62, y: 300)); ctx.addLine(to: CGPoint(x: 108, y: 300))
        ctx.move(to: CGPoint(x: 62, y: 340)); ctx.addLine(to: CGPoint(x: 340, y: 340))
        ctx.move(to: CGPoint(x: 62, y: 380)); ctx.addLine(to: CGPoint(x: 108, y: 380))
        for y: CGFloat in [220, 260, 300, 340, 380] {
            ctx.move(to: CGPoint(x: 492, y: y))
            ctx.addLine(to: CGPoint(x: 538, y: y))
        }
        ctx.strokePath()

        // Die Brackets
        ctx.move(to: CGPoint(x: 276, y: 135))
        ctx.addLine(to: CGPoint(x: 180, y: 135))
        ctx.addQuadCurve(to: CGPoint(x: 135, y: 180), control: CGPoint(x: 135, y: 135))
        ctx.addLine(to: CGPoint(x: 135, y: 276))

        ctx.move(to: CGPoint(x: 324, y: 135))
        ctx.addLine(to: CGPoint(x: 420, y: 135))
        ctx.addQuadCurve(to: CGPoint(x: 465, y: 180), control: CGPoint(x: 465, y: 135))
        ctx.addLine(to: CGPoint(x: 465, y: 276))

        ctx.move(to: CGPoint(x: 135, y: 324))
        ctx.addLine(to: CGPoint(x: 135, y: 420))
        ctx.addQuadCurve(to: CGPoint(x: 180, y: 465), control: CGPoint(x: 135, y: 465))
        ctx.addLine(to: CGPoint(x: 276, y: 465))

        ctx.move(to: CGPoint(x: 465, y: 324))
        ctx.addLine(to: CGPoint(x: 465, y: 420))
        ctx.addQuadCurve(to: CGPoint(x: 420, y: 465), control: CGPoint(x: 465, y: 465))
        ctx.addLine(to: CGPoint(x: 324, y: 465))
        ctx.strokePath()

        // Letter Q
        let qRect = CGRect(x: 180, y: 175, width: 112, height: 250)
        ctx.addPath(CGPath(roundedRect: qRect, cornerWidth: 36, cornerHeight: 36, transform: nil))
        ctx.move(to: CGPoint(x: 246, y: 336))
        ctx.addLine(to: CGPoint(x: 346, y: 436))
        ctx.strokePath()

        // Letter G
        ctx.move(to: CGPoint(x: 324, y: 175))
        ctx.addLine(to: CGPoint(x: 380, y: 175))
        ctx.addQuadCurve(to: CGPoint(x: 412, y: 207), control: CGPoint(x: 412, y: 175))
        ctx.addLine(to: CGPoint(x: 412, y: 268))

        ctx.move(to: CGPoint(x: 324, y: 326))
        ctx.addLine(to: CGPoint(x: 412, y: 326))
        ctx.addLine(to: CGPoint(x: 412, y: 393))
        ctx.addQuadCurve(to: CGPoint(x: 380, y: 425), control: CGPoint(x: 412, y: 425))
        ctx.addLine(to: CGPoint(x: 324, y: 425))
        ctx.strokePath()

        ctx.restoreGState()
    }

    public var body: some View {
        GeometryReader { geo in
            let w = geo.size.width
            let h = geo.size.height
            let scale = min(w, h) / 600.0

            Canvas { ctx, _ in
                ctx.scaleBy(x: scale, y: scale)

                // Bus Pins
                let pinPath = Path { p in
                    // Top
                    p.move(to: CGPoint(x: 220, y: 62)); p.addLine(to: CGPoint(x: 220, y: 108))
                    p.move(to: CGPoint(x: 260, y: 62)); p.addLine(to: CGPoint(x: 260, y: 108))
                    p.move(to: CGPoint(x: 300, y: 62)); p.addLine(to: CGPoint(x: 300, y: 108))
                    p.move(to: CGPoint(x: 340, y: 62)); p.addLine(to: CGPoint(x: 340, y: 108))
                    p.move(to: CGPoint(x: 380, y: 62)); p.addLine(to: CGPoint(x: 380, y: 108))
                    // Bottom
                    p.move(to: CGPoint(x: 220, y: 492)); p.addLine(to: CGPoint(x: 220, y: 538))
                    p.move(to: CGPoint(x: 260, y: 492)); p.addLine(to: CGPoint(x: 260, y: 538))
                    p.move(to: CGPoint(x: 300, y: 492)); p.addLine(to: CGPoint(x: 300, y: 538))
                    p.move(to: CGPoint(x: 340, y: 492)); p.addLine(to: CGPoint(x: 340, y: 538))
                    p.move(to: CGPoint(x: 380, y: 492)); p.addLine(to: CGPoint(x: 380, y: 538))
                    // Left
                    p.move(to: CGPoint(x: 62, y: 220)); p.addLine(to: CGPoint(x: 108, y: 220))
                    p.move(to: CGPoint(x: 62, y: 260)); p.addLine(to: CGPoint(x: 108, y: 260))
                    p.move(to: CGPoint(x: 62, y: 300)); p.addLine(to: CGPoint(x: 108, y: 300))
                    p.move(to: CGPoint(x: 62, y: 340)); p.addLine(to: CGPoint(x: 340, y: 340))
                    p.move(to: CGPoint(x: 62, y: 380)); p.addLine(to: CGPoint(x: 108, y: 380))
                    // Right
                    p.move(to: CGPoint(x: 492, y: 220)); p.addLine(to: CGPoint(x: 538, y: 220))
                    p.move(to: CGPoint(x: 492, y: 260)); p.addLine(to: CGPoint(x: 538, y: 260))
                    p.move(to: CGPoint(x: 492, y: 300)); p.addLine(to: CGPoint(x: 538, y: 300))
                    p.move(to: CGPoint(x: 492, y: 340)); p.addLine(to: CGPoint(x: 538, y: 340))
                    p.move(to: CGPoint(x: 492, y: 380)); p.addLine(to: CGPoint(x: 538, y: 380))
                }
                ctx.stroke(pinPath, with: .color(TelemetryTheme.logoInk), style: StrokeStyle(lineWidth: 24, lineCap: .round))

                // Die Brackets
                let bracketPath = Path { p in
                    p.move(to: CGPoint(x: 276, y: 135))
                    p.addLine(to: CGPoint(x: 180, y: 135))
                    p.addQuadCurve(to: CGPoint(x: 135, y: 180), control: CGPoint(x: 135, y: 135))
                    p.addLine(to: CGPoint(x: 135, y: 276))

                    p.move(to: CGPoint(x: 324, y: 135))
                    p.addLine(to: CGPoint(x: 420, y: 135))
                    p.addQuadCurve(to: CGPoint(x: 465, y: 180), control: CGPoint(x: 465, y: 135))
                    p.addLine(to: CGPoint(x: 465, y: 276))

                    p.move(to: CGPoint(x: 135, y: 324))
                    p.addLine(to: CGPoint(x: 135, y: 420))
                    p.addQuadCurve(to: CGPoint(x: 180, y: 465), control: CGPoint(x: 135, y: 465))
                    p.addLine(to: CGPoint(x: 276, y: 465))

                    p.move(to: CGPoint(x: 465, y: 324))
                    p.addLine(to: CGPoint(x: 465, y: 420))
                    p.addQuadCurve(to: CGPoint(x: 420, y: 465), control: CGPoint(x: 465, y: 465))
                    p.addLine(to: CGPoint(x: 324, y: 465))
                }
                ctx.stroke(bracketPath, with: .color(TelemetryTheme.logoInk), style: StrokeStyle(lineWidth: 24, lineCap: .round, lineJoin: .round))

                // Letter Q
                let qPath = Path { p in
                    p.addRoundedRect(in: CGRect(x: 180, y: 175, width: 112, height: 250), cornerSize: CGSize(width: 36, height: 36))
                    p.move(to: CGPoint(x: 246, y: 336))
                    p.addLine(to: CGPoint(x: 346, y: 436))
                }
                ctx.stroke(qPath, with: .color(TelemetryTheme.logoInk), style: StrokeStyle(lineWidth: 24, lineCap: .round, lineJoin: .round))

                // Letter G
                let gPath = Path { p in
                    p.move(to: CGPoint(x: 324, y: 175))
                    p.addLine(to: CGPoint(x: 380, y: 175))
                    p.addQuadCurve(to: CGPoint(x: 412, y: 207), control: CGPoint(x: 412, y: 175))
                    p.addLine(to: CGPoint(x: 412, y: 268))

                    p.move(to: CGPoint(x: 324, y: 326))
                    p.addLine(to: CGPoint(x: 412, y: 326))
                    p.addLine(to: CGPoint(x: 412, y: 393))
                    p.addQuadCurve(to: CGPoint(x: 380, y: 425), control: CGPoint(x: 412, y: 425))
                    p.addLine(to: CGPoint(x: 324, y: 425))
                }
                ctx.stroke(gPath, with: .color(TelemetryTheme.logoInk), style: StrokeStyle(lineWidth: 24, lineCap: .round, lineJoin: .round))
            }
        }
    }
}

// MARK: - Color Extension Helper
extension Color {
    public init(hex: String) {
        let clean = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var int: UInt64 = 0
        Scanner(string: clean).scanHexInt64(&int)
        let r, g, b: UInt64
        switch clean.count {
        case 3:
            (r, g, b) = ((int >> 8) * 17, (int >> 4 & 0xF) * 17, (int & 0xF) * 17)
        case 6:
            (r, g, b) = (int >> 16, int >> 8 & 0xFF, int & 0xFF)
        default:
            (r, g, b) = (1, 1, 1)
        }
        self.init(
            .sRGB,
            red: Double(r) / 255,
            green: Double(g) / 255,
            blue: Double(b) / 255,
            opacity: 1
        )
    }
}
