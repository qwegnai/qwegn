// DualSparklineView.swift — prefill/decode sparkline card
// (split from CupertinoTelemetryView.swift).
// A sample that was not measured is a gap in the line, never a zero on
// the floor: a flat line at zero would claim the engine did nothing.
import SwiftUI
import AppKit
import QWServer

public struct DualSparklineView: View {
    public var prefillData: [Double?]
    public var decodeData: [Double?]

    public init(prefillData: [Double?], decodeData: [Double?]) {
        self.prefillData = prefillData
        self.decodeData = decodeData
    }

    public var body: some View {
        GeometryReader { geo in
            let w = geo.size.width
            let h = geo.size.height
            let maxP = prefillData.compactMap { $0 }.max() ?? 0.0
            let maxD = decodeData.compactMap { $0 }.max() ?? 0.0
            let maxVal = max(40.0, max(maxP, maxD) * 1.15)

            let pRuns = runs(data: prefillData, width: w, height: h, maxVal: maxVal)
            let dRuns = runs(data: decodeData, width: w, height: h, maxVal: maxVal)

            ZStack {
                ForEach(Array(pRuns.enumerated()), id: \.offset) { _, run in
                    areaPath(run, height: h)
                        .fill(LinearGradient(colors: [TelemetryTheme.prefill.opacity(0.25), Color.clear],
                                             startPoint: .top, endPoint: .bottom))
                    linePath(run)
                        .stroke(TelemetryTheme.prefill, lineWidth: 1.2)
                }

                ForEach(Array(dRuns.enumerated()), id: \.offset) { _, run in
                    areaPath(run, height: h)
                        .fill(LinearGradient(colors: [TelemetryTheme.accent.opacity(0.25), Color.clear],
                                             startPoint: .top, endPoint: .bottom))
                    linePath(run)
                        .stroke(TelemetryTheme.accent, lineWidth: 1.2)
                }
            }
        }
    }

    private func areaPath(_ points: [CGPoint], height: CGFloat) -> Path {
        Path { p in
            guard let first = points.first else { return }
            p.move(to: CGPoint(x: first.x, y: height))
            p.addLine(to: first)
            for pt in points.dropFirst() { p.addLine(to: pt) }
            if let last = points.last { p.addLine(to: CGPoint(x: last.x, y: height)) }
            p.closeSubpath()
        }
    }

    private func linePath(_ points: [CGPoint]) -> Path {
        Path { p in
            guard let first = points.first else { return }
            p.move(to: first)
            for pt in points.dropFirst() { p.addLine(to: pt) }
        }
    }

    /// The points of the samples that exist, split into the runs between
    /// gaps. A missing sample breaks the line instead of filling it.
    private func runs(data: [Double?], width: CGFloat, height: CGFloat, maxVal: Double) -> [[CGPoint]] {
        guard data.count > 1 else { return [] }
        let step = width / CGFloat(data.count - 1)
        var out: [[CGPoint]] = []
        var current: [CGPoint] = []
        for (index, val) in data.enumerated() {
            guard let val else {
                if current.count > 1 { out.append(current) }
                current = []
                continue
            }
            let x = CGFloat(index) * step
            let norm = CGFloat(min(1.0, max(0.0, val / maxVal)))
            let y = height - (norm * (height - 4)) - 2
            current.append(CGPoint(x: x, y: y))
        }
        if current.count > 1 { out.append(current) }
        return out
    }
}

// MARK: - QG Processor Microchip Vector Logo
