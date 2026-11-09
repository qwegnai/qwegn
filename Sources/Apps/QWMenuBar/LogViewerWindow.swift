import SwiftUI
import AppKit
import QWServer

/// Dedicated streaming Log Viewer Window with JetBrains Mono typography,
/// real-time log ingestion, search filtering, and an explicit "Reveal in Finder" button.
public struct LogViewerWindow: View {
    @ObservedObject public var telemetry: TelemetryObservable
    /// Increased Contrast asks for stronger text. L-M7: it is honoured.
    @Environment(\.colorSchemeContrast) internal var contrast: ColorSchemeContrast
    @State private var filterLevel: String = "ALL"
    @State private var searchText: String = ""
    @State private var autoScroll: Bool = true
    public var onClose: () -> Void

    public init(telemetry: TelemetryObservable, onClose: @escaping () -> Void = {}) {
        self.telemetry = telemetry
        self.onClose = onClose
    }

    private var filteredLogs: [String] {
        telemetry.recentLogs.filter { line in
            let matchesLevel: Bool
            if filterLevel == "ALL" {
                matchesLevel = true
            } else {
                matchesLevel = line.contains(" \(filterLevel) ")
            }

            let matchesSearch: Bool
            if searchText.isEmpty {
                matchesSearch = true
            } else {
                matchesSearch = line.localizedCaseInsensitiveContains(searchText)
            }

            return matchesLevel && matchesSearch
        }
    }

    public var body: some View {
        ZStack {
            // The glass lives inside a container: the system blends a
            // group of glass together, and glass outside one is alone.
            GlassEffectContainer(spacing: 8) {
                SystemVibrancyView()
                    .edgesIgnoringSafeArea(.all)
            }

            VStack(spacing: 0) {
                // Subtle Unified Toolbar
                HStack(spacing: 8) {
                    // Left: App / Log Identifier
                    HStack(spacing: 6) {
                        Image(systemName: "terminal")
                            .font(.system(size: 12, weight: .medium))
                            .foregroundColor(TelemetryTheme.accent)
                        Text("Engine Logs")
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundColor(TelemetryTheme.primary(increasedContrast: contrast == .increased))
                        Text("\(filteredLogs.count)")
                            .font(.system(size: 10, weight: .medium, design: .monospaced))
                            .foregroundColor(TelemetryTheme.secondary(increasedContrast: contrast == .increased))
                            .padding(.horizontal, 5)
                            .padding(.vertical, 1)
                            .background(Color.white.opacity(0.08))
                            .cornerRadius(4)
                    }

                    Spacer(minLength: 8)

                    // Center: Subtle Search Field
                    HStack(spacing: 5) {
                        Image(systemName: "magnifyingglass")
                            .font(.system(size: 10))
                            .foregroundColor(TelemetryTheme.quiet(increasedContrast: contrast == .increased))
                        TextField("Filter logs...", text: $searchText)
                            .textFieldStyle(.plain)
                            .font(.system(size: 11))
                        if !searchText.isEmpty {
                            Button(action: { searchText = "" }) {
                                Image(systemName: "xmark.circle.fill")
                                    .font(.system(size: 10))
                                    .foregroundColor(TelemetryTheme.quiet(increasedContrast: contrast == .increased))
                            }
                            .buttonStyle(.plain)
                        }
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .frame(minWidth: 140, maxWidth: 200)
                    .background(Color.black.opacity(0.35))
                    .cornerRadius(6)
                    .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.white.opacity(0.10), lineWidth: 0.5))

                    // Filter Level Picker (Compact Menu Style - no overlap)
                    Picker("", selection: $filterLevel) {
                        Text("All Levels").tag("ALL")
                        Text("Info").tag("INFO")
                        Text("Warn").tag("WARN")
                        Text("Error").tag("ERROR")
                        Text("Debug").tag("DEBUG")
                    }
                    .pickerStyle(.menu)
                    .frame(width: 105)
                    .font(.system(size: 11))

                    // Auto-Scroll Toggle Button
                    Button(action: { autoScroll.toggle() }) {
                        HStack(spacing: 3) {
                            Image(systemName: autoScroll ? "arrow.down.to.line" : "pause")
                                .font(.system(size: 10))
                            Text("Scroll")
                                .font(.system(size: 10))
                        }
                        .foregroundColor(autoScroll ? TelemetryTheme.active : .white.opacity(0.40))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 4)
                        .background(Color.white.opacity(autoScroll ? 0.10 : 0.04))
                        .cornerRadius(5)
                        .overlay(RoundedRectangle(cornerRadius: 5).stroke(Color.white.opacity(0.08), lineWidth: 0.5))
                    }
                    .buttonStyle(.plain)
                    .help(autoScroll ? "Auto-scroll enabled (click to pause)" : "Auto-scroll paused (click to follow)")

                    Divider()
                        .frame(height: 14)
                        .background(Color.white.opacity(0.12))

                    // Action Icons: Finder, Copy, Trash, Close
                    HStack(spacing: 4) {
                        Button(action: revealInFinder) {
                            HStack(spacing: 3) {
                                Image(systemName: "folder")
                                    .font(.system(size: 10))
                                Text("Finder")
                                    .font(.system(size: 10))
                            }
                            .foregroundColor(.white.opacity(0.75))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 4)
                            .background(Color.white.opacity(0.06))
                            .cornerRadius(5)
                        }
                        .buttonStyle(.plain)
                        .help("Reveal log folder in macOS Finder")

                        Button(action: copyLogs) {
                            Image(systemName: "doc.on.doc")
                                .font(.system(size: 11))
                                .foregroundColor(.white.opacity(0.60))
                                .frame(width: 22, height: 22)
                                .background(Color.white.opacity(0.04))
                                .cornerRadius(5)
                        }
                        .buttonStyle(.plain)
                        .help("Copy visible logs to clipboard")

                        Button(action: { telemetry.recentLogs.removeAll() }) {
                            Image(systemName: "trash")
                                .font(.system(size: 11))
                                .foregroundColor(.white.opacity(0.60))
                                .frame(width: 22, height: 22)
                                .background(Color.white.opacity(0.04))
                                .cornerRadius(5)
                        }
                        .buttonStyle(.plain)
                        .help("Clear log viewer buffer")

                        Button(action: onClose) {
                            Image(systemName: "xmark")
                                .font(.system(size: 11, weight: .semibold))
                                .foregroundColor(TelemetryTheme.quiet(increasedContrast: contrast == .increased))
                                .frame(width: 22, height: 22)
                        }
                        .buttonStyle(.plain)
                        .help("Close window")
                    }
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(Color.black.opacity(0.25))

                Divider().background(Color.white.opacity(0.08))

                // Terminal Scroll Area
                ScrollViewReader { proxy in
                    ScrollView(.vertical, showsIndicators: true) {
                        LazyVStack(alignment: .leading, spacing: 2) {
                            ForEach(Array(filteredLogs.enumerated()), id: \.offset) { idx, line in
                                logLineView(line)
                                    .id(idx)
                            }
                        }
                        .padding(10)
                    }
                    .background(Color(red: 12/255, green: 14/255, blue: 18/255).opacity(0.85))
                    .onChange(of: filteredLogs.count) { _, newCount in
                        if autoScroll && newCount > 0 {
                            withAnimation(.easeOut(duration: 0.1)) {
                                proxy.scrollTo(newCount - 1, anchor: .bottom)
                            }
                        }
                    }
                }
            }
        }
        .frame(width: 680, height: 460)
    }

    private func logLineView(_ line: String) -> some View {
        let trimmed = line.trimmingCharacters(in: .newlines)
        let color: Color
        if trimmed.contains(" ERROR ") {
            color = TelemetryTheme.failure
        } else if trimmed.contains(" WARN ") {
            color = TelemetryTheme.warning
        } else if trimmed.contains(" DEBUG ") {
            color = TelemetryTheme.unknown
        } else if trimmed.contains(" -> 200 OK ") {
            color = TelemetryTheme.active
        } else {
            color = TelemetryTheme.highlight
        }

        return Text(trimmed)
            .font(.system(size: 11, design: .monospaced))
            .foregroundColor(color)
            .textSelection(.enabled)
    }

    private func revealInFinder() {
        let path = telemetry.settings.expandedLogFilePath
        let url = URL(fileURLWithPath: path)
        let dir = url.deletingLastPathComponent()

        // Ensure file exists so Finder can select it
        if !FileManager.default.fileExists(atPath: url.path) {
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            FileManager.default.createFile(atPath: url.path, contents: Data())
        }

        NSWorkspace.shared.activateFileViewerSelecting([url])
    }

    private func copyLogs() {
        let allText = filteredLogs.joined()
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(allText, forType: .string)
    }
}
