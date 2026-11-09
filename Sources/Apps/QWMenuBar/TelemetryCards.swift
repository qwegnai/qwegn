// TelemetryCards.swift — CupertinoTelemetryView card subviews
// (split from CupertinoTelemetryView.swift; same-module extension).
import SwiftUI
import AppKit
import QWServer

extension CupertinoTelemetryView {
    // MARK: - 1. Command Header Bar
    var headerSection: some View {
        HStack(spacing: 10) {
            // Emblem with live status halo badge
            ZStack(alignment: .bottomTrailing) {
                ZStack {
                    RoundedRectangle(cornerRadius: 8)
                        .fill(Color.white.opacity(0.07))
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(TelemetryTheme.hairline(increasedContrast: contrast == .increased), lineWidth: 0.5)
                        )
                    QGProcessorLogoView()
                        .frame(width: 20, height: 20)
                }
                .frame(width: 32, height: 32)

                // Live status dot + pulse when actively generating
                ZStack {
                    if telemetry.isServing {
                        Circle()
                            .fill(statusIndicatorColor)
                            .frame(width: 12, height: 12)
                            .opacity(isGenerating && !reduceMotion ? 0.40 : 0.15)
                            .scaleEffect(isGenerating && !reduceMotion ? 1.35 : 1.0)
                            .animation(isGenerating && !reduceMotion
                                       ? .easeInOut(duration: 0.9).repeatForever(autoreverses: true)
                                       : .default, value: isGenerating)
                    }
                    Circle()
                        .fill(Color(hex: "#111318"))
                        .frame(width: 10, height: 10)
                    Circle()
                        .fill(statusIndicatorColor)
                        .frame(width: 7, height: 7)
                        .shadow(color: statusIndicatorColor.opacity(0.7), radius: 2)
                }
                .offset(x: 2, y: 2)
            }

            // Title, Version & Status Subtitle
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text("Qwegn Engine")
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundColor(TelemetryTheme.primary(increasedContrast: contrast == .increased))
                        .lineLimit(1)

                    Text(telemetry.engineVersion.hasPrefix("v") ? telemetry.engineVersion : "v\(telemetry.engineVersion)")
                        .font(TelemetryTheme.numeric(9, .medium))
                        .foregroundColor(.white.opacity(0.70))
                        .padding(.horizontal, 5)
                        .padding(.vertical, 1.5)
                        .background(Color.white.opacity(0.08))
                        .cornerRadius(TelemetryTheme.Radius.small)
                }

                HStack(spacing: 5) {
                    Text(telemetry.serverStatusText)
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundColor(statusIndicatorColor)
                        .lineLimit(1)

                    Text("•")
                        .foregroundColor(.white.opacity(0.28))
                        .font(TelemetryTheme.ui(8))

                    Text("\(telemetry.settings.host):\(telemetry.serverPortString)")
                        .font(TelemetryTheme.numeric(10, .medium))
                        .foregroundColor(.white.opacity(0.65))
                        .lineLimit(1)
                }
            }

            Spacer(minLength: 4)

            // Primary Lifecycle Controls: Start/Stop & Restart
            HStack(spacing: 6) {
                let actionTitle = telemetry.primaryActionTitle
                let actionColor = (actionTitle == "Stop") ? TelemetryTheme.failure : TelemetryTheme.active
                Button(action: { telemetry.performPrimaryAction() }) {
                    HStack(spacing: 4) {
                        Image(systemName: telemetry.primaryActionIcon)
                            .font(.system(size: 9, weight: .bold))
                        Text(actionTitle)
                            .font(.system(size: 10, weight: .semibold))
                    }
                    .foregroundColor(actionColor)
                    .padding(.horizontal, 2)
                }
                .buttonStyle(.glass)

                Button(action: { telemetry.restartEngine() }) {
                    Image(systemName: "arrow.clockwise")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundColor(.white.opacity(0.75))
                }
                .buttonStyle(.glass)
                .disabled(!telemetry.isConnected)
                .accessibilityLabel("Restart model")
                .help("Restart the model inside the running server (⌘R)")
            }
        }
        .padding(.horizontal, 2)
    }

    // MARK: - 2. Loaded Model Weights Card
    var loadedWeightsCard: some View {
        // The model the server says is engaged, not the name the user
        // configured: the server can swap models at runtime.
        let modelName = telemetry.dashboard?.lifecycle.model
        // What the server says about the resident model. Nothing is
        // assumed: a fact the server did not report is not shown.
        let facts = telemetry.dashboard?.facts
        let quant = facts?.quantization
        let metal = facts?.metalFamily

        let alias = telemetry.settings.modelAlias.trimmingCharacters(in: .whitespacesAndNewlines)
        let showAlias = !alias.isEmpty && alias.lowercased() != (modelName ?? "").lowercased()
        let badgeParts = [quant, metal].compactMap { $0 }
        let badge = badgeParts.isEmpty ? nil : badgeParts.joined(separator: " • ")

        return VStack(alignment: .leading, spacing: 6) {
            // Row 1: Left-aligned interactive Model Selector pill + flush-right Load/Unload button
            HStack(spacing: 6) {
                Button(action: { showModelSelectorMenu() }) {
                    HStack(spacing: 6) {
                        Image(systemName: "cpu.fill")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundColor(TelemetryTheme.accent)

                        // No model named: either the engine has none, or the
                        // server did not answer. Only the first is a fact.
                        Text(modelName ?? (telemetry.connection == .connected ? "No model loaded" : "—"))
                            .font(.system(size: 11.5, weight: .semibold, design: .monospaced))
                            .foregroundColor(modelName == nil ? .white.opacity(0.55) : .white.opacity(0.96))
                            .lineLimit(1)
                            .minimumScaleFactor(0.75)

                        Spacer(minLength: 4)

                        Image(systemName: "chevron.up.chevron.down")
                            .font(.system(size: 8.5, weight: .bold))
                            .foregroundColor(.white.opacity(0.60))
                    }
                    .padding(.horizontal, 8)
                    .frame(maxWidth: .infinity, minHeight: 25, maxHeight: 25, alignment: .leading)
                    .background(Color.black.opacity(0.28))
                    .cornerRadius(TelemetryTheme.Radius.small)
                    .overlay(
                        RoundedRectangle(cornerRadius: TelemetryTheme.Radius.small)
                            .stroke(Color.white.opacity(0.10), lineWidth: 0.5)
                    )
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help(modelName.map { "Active model: \($0) — click to switch model" } ?? "Click to select and load a model")

                modelActions
            }

            // Row 2: Full Quantization + Metal family details & optional Alias (flush left-to-right)
            if badge != nil || showAlias {
                HStack(spacing: 6) {
                    if let badge {
                        MarqueeBadgeView(text: badge, maxAvailableWidth: .infinity)
                    }
                    if showAlias {
                        factPill(label: "Alias", value: alias, valueColor: TelemetryTheme.accent.opacity(0.95), expand: true)
                    }
                }
            }

            Divider()
                .overlay(TelemetryTheme.hairline(increasedContrast: contrast == .increased))

            // Row 3: Hardware & Engine Configuration Facts (evenly spanning full card width)
            modelFacts
                .font(TelemetryTheme.metric)
                .foregroundColor(TelemetryTheme.secondary(increasedContrast: contrast == .increased))
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity)
        .background(TelemetryTheme.substrate)
        .cornerRadius(TelemetryTheme.Radius.large)
        .overlay(
            RoundedRectangle(cornerRadius: TelemetryTheme.Radius.large)
                .stroke(TelemetryTheme.hairline(increasedContrast: contrast == .increased), lineWidth: 0.5)
        )
    }

    @MainActor
    func showModelSelectorMenu() {
        telemetry.refreshDiscoveredModels()
        let menu = NSMenu(title: "Select Model")
        let currentModel = telemetry.dashboard?.lifecycle.model

        if telemetry.discoveredModels.isEmpty {
            let emptyItem = NSMenuItem(title: "No models found in ~/.qwg/models", action: nil, keyEquivalent: "")
            emptyItem.isEnabled = false
            menu.addItem(emptyItem)
        } else {
            for item in telemetry.discoveredModels {
                let menuItem = NSMenuItem(
                    title: item.name,
                    action: #selector(ModelSelectorMenuTarget.selectModel(_:)),
                    keyEquivalent: ""
                )
                menuItem.target = ModelSelectorMenuTarget.shared
                menuItem.representedObject = ModelSelectorPayload(telemetry: telemetry, model: item)
                menuItem.state = (item.name == currentModel) ? .on : .off
                menu.addItem(menuItem)
            }
        }

        menu.addItem(.separator())

        let refreshItem = NSMenuItem(
            title: "Refresh Models",
            action: #selector(ModelSelectorMenuTarget.refreshModels(_:)),
            keyEquivalent: ""
        )
        refreshItem.target = ModelSelectorMenuTarget.shared
        refreshItem.representedObject = telemetry
        menu.addItem(refreshItem)

        let finderItem = NSMenuItem(
            title: "Show ~/.qwg/models in Finder…",
            action: #selector(ModelSelectorMenuTarget.revealFinder(_:)),
            keyEquivalent: ""
        )
        finderItem.target = ModelSelectorMenuTarget.shared
        finderItem.representedObject = telemetry
        menu.addItem(finderItem)

        menu.popUp(positioning: nil, at: NSEvent.mouseLocation, in: nil)
    }

    /// The facts the server reported about the resident model. Each is
    /// shown only when the server reported it, and each says what it is:
    /// the host's cores are not the engine's threads.
    var modelFacts: some View {
        HStack(spacing: 5) {
            if let cores = telemetry.dashboard?.facts?.hostLogicalCores {
                factPill(label: "Cores", value: "\(cores)", valueColor: .white.opacity(0.92), expand: true)
            }
            if let gdn = telemetry.dashboard?.facts?.gdnPrefillMode, !gdn.isEmpty {
                factPill(label: "GDN", value: gdn, valueColor: TelemetryTheme.active, expand: true)
            }
            if let ctx = telemetry.contextBudget {
                factPill(label: "Context", value: "\(ctx / 1024)k", valueColor: .white.opacity(0.92), expand: true)
            }
            if let kv = telemetry.dashboard?.knobs["kvMode"]?.stringValue, !kv.isEmpty {
                factPill(label: "KV", value: kv.uppercased(), valueColor: TelemetryTheme.accent, expand: true)
            }
            if let drafts = telemetry.dashboard?.knobs["mtpDrafts"]?.intValue {
                factPill(label: "MTP", value: drafts > 0 ? "\(drafts)d" : "off",
                         valueColor: drafts > 0 ? TelemetryTheme.active : .white.opacity(0.65), expand: true)
            }
        }
        .frame(maxWidth: .infinity)
    }

    func factPill(label: String, value: String, valueColor: Color, expand: Bool = false) -> some View {
        HStack(spacing: 3) {
            Text("\(label):")
                .font(TelemetryTheme.ui(9, .medium))
                .foregroundColor(TelemetryTheme.secondary(increasedContrast: contrast == .increased))
            Text(value)
                .font(TelemetryTheme.numeric(9, .semibold))
                .foregroundColor(valueColor)
        }
        .lineLimit(1)
        .fixedSize(horizontal: true, vertical: true)
        .padding(.horizontal, 6)
        .frame(maxWidth: expand ? .infinity : nil, minHeight: 22, maxHeight: 22)
        .background(Color.black.opacity(0.26))
        .cornerRadius(TelemetryTheme.Radius.small)
        .overlay(
            RoundedRectangle(cornerRadius: TelemetryTheme.Radius.small)
                .stroke(Color.white.opacity(0.06), lineWidth: 0.5)
        )
    }

    /// The actions the server says this model can perform. The renderer
    /// does not decide: it draws the published list.
    var modelActions: some View {
        HStack(spacing: 6) {
            if let load = action("load") {
                Button(action: { telemetry.reloadModel() }) {
                    HStack(spacing: 4) {
                        Image(systemName: "arrow.down.circle")
                            .font(.system(size: 9, weight: .semibold))
                        Text(load.title)
                            .font(TelemetryTheme.ui(9, .semibold))
                    }
                    .foregroundColor(TelemetryTheme.accent)
                    .padding(.horizontal, 8)
                    .frame(minHeight: 25, maxHeight: 25)
                    .background(TelemetryTheme.accent.opacity(0.14))
                    .cornerRadius(TelemetryTheme.Radius.small)
                    .overlay(
                        RoundedRectangle(cornerRadius: TelemetryTheme.Radius.small)
                            .stroke(TelemetryTheme.accent.opacity(0.30), lineWidth: 0.5)
                    )
                }
                .buttonStyle(.plain)
                .lineLimit(1)
                .fixedSize(horizontal: true, vertical: true)
                .disabled(!load.isEnabled)
                .help(load.reason ?? "Load the model")
            }
            if let unload = action("unload") {
                Button(action: { telemetry.unloadModel() }) {
                    HStack(spacing: 4) {
                        Image(systemName: "eject.fill")
                            .font(.system(size: 8, weight: .semibold))
                        Text(unload.title)
                            .font(TelemetryTheme.ui(9, .medium))
                    }
                    .foregroundColor(.white.opacity(0.85))
                    .padding(.horizontal, 8)
                    .frame(minHeight: 25, maxHeight: 25)
                    .background(Color.white.opacity(0.08))
                    .cornerRadius(TelemetryTheme.Radius.small)
                    .overlay(
                        RoundedRectangle(cornerRadius: TelemetryTheme.Radius.small)
                            .stroke(Color.white.opacity(0.10), lineWidth: 0.5)
                    )
                }
                .buttonStyle(.plain)
                .lineLimit(1)
                .fixedSize(horizontal: true, vertical: true)
                .disabled(!unload.isEnabled)
                .help(unload.reason ?? "Unload the model")
            }
        }
    }

    // MARK: - Compact Quantization Badge View
    // Displays the quantization & Metal family pill without ever expanding parent layout width.
    public struct MarqueeBadgeView: View {
        public let text: String
        public let maxAvailableWidth: CGFloat?

        let badgeFont = Font.system(size: 9, weight: .semibold, design: .monospaced)
        let badgeColor = TelemetryTheme.active

        public init(text: String, maxAvailableWidth: CGFloat? = nil) {
            self.text = text
            self.maxAvailableWidth = maxAvailableWidth
        }

        public var body: some View {
            Text(text)
                .font(badgeFont)
                .foregroundColor(badgeColor)
                .lineLimit(1)
                .minimumScaleFactor(0.80)
                .truncationMode(.tail)
                .fixedSize(horizontal: true, vertical: true)
                .padding(.horizontal, 8)
                .frame(maxWidth: maxAvailableWidth, minHeight: 22, maxHeight: 22, alignment: .center)
                .background(badgeColor.opacity(0.14))
                .cornerRadius(TelemetryTheme.Radius.small)
                .overlay(
                    RoundedRectangle(cornerRadius: TelemetryTheme.Radius.small)
                        .stroke(badgeColor.opacity(0.30), lineWidth: 0.5)
                )
                .help(text)
        }
    }

    // MARK: - 3. The Measurements
    /// Every measurement the server published, in one grid. One shape
    /// for every tile and one column for every row: the panel is read
    /// by looking across it, and an uneven row breaks the reading.
    var metricGrid: some View {
        VStack(spacing: 6) {
            HStack(spacing: 6) {
                metricTile("Prefill", card("throughput.display_prefill_tokens_per_sec"),
                           accent: TelemetryTheme.prefill)
                metricTile("Decode", card("throughput.display_decode_tokens_per_sec"),
                           accent: TelemetryTheme.accent)
                metricTile("TTFT", card("latency.ttft_ms"),
                           accent: TelemetryTheme.cool)
                metricTile("Engine GPU", card("gpu.busy_ratio"),
                           accent: TelemetryTheme.active)
            }
            .frame(maxHeight: .infinity)
            HStack(spacing: 6) {
                metricTile("Host GPU", card("gpu.device"),
                           accent: TelemetryTheme.cool)
                metricTile("Host CPU", card("hardware.cpu_percent"),
                           accent: TelemetryTheme.cool)
                metricTile("Memory", card("hardware.memory_percent"),
                           accent: TelemetryTheme.neutral)
                metricTile("Memory pressure", card("hardware.memory_pressure_level"),
                           accent: TelemetryTheme.warning)
            }
            .frame(maxHeight: .infinity)
            HStack(spacing: 6) {
                metricTile("Warm cache", card("cache.warm_bytes"),
                           accent: TelemetryTheme.active)
                metricTile("SSD cache", card("cache.ssd_bytes"),
                           accent: TelemetryTheme.cool)
                metricTile("Reused", card("cache.tokens_reused_total"),
                           accent: TelemetryTheme.warning)
                metricTile("Context", card("context.active_tokens"),
                           accent: TelemetryTheme.accent)
            }
            .frame(maxHeight: .infinity)
        }
        .frame(maxHeight: .infinity)
    }

    // MARK: - 4. Live Trace Card
    /// The published trace: the samples the server recorded, gaps and
    /// all (Amber Prefill + Blue Decode).
    var traceStrip: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack(spacing: 8) {
                Text("Live trace")
                    .font(TelemetryTheme.ui(9, .semibold))
                    .foregroundColor(TelemetryTheme.secondary(increasedContrast: contrast == .increased))

                HStack(spacing: 6) {
                    HStack(spacing: 3) {
                        Circle()
                            .fill(TelemetryTheme.prefill)
                            .frame(width: 5, height: 5)
                        Text("Prefill")
                            .font(TelemetryTheme.ui(8, .medium))
                            .foregroundColor(TelemetryTheme.quiet(increasedContrast: contrast == .increased))
                    }
                    HStack(spacing: 3) {
                        Circle()
                            .fill(TelemetryTheme.accent)
                            .frame(width: 5, height: 5)
                        Text("Decode")
                            .font(TelemetryTheme.ui(8, .medium))
                            .foregroundColor(TelemetryTheme.quiet(increasedContrast: contrast == .increased))
                    }
                }

                Spacer()

                Text("last 60 turns")
                    .font(TelemetryTheme.ui(8))
                    .foregroundColor(TelemetryTheme.quiet(increasedContrast: contrast == .increased))
            }

            DualSparklineView(prefillData: card("throughput.display_prefill_tokens_per_sec")?.series ?? [],
                              decodeData: card("throughput.display_decode_tokens_per_sec")?.series ?? [])
                .frame(height: 28)
                .padding(.horizontal, 6)
                .padding(.vertical, 3)
                .background(Color.black.opacity(0.28))
                .cornerRadius(TelemetryTheme.Radius.medium)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .background(TelemetryTheme.substrate)
        .cornerRadius(TelemetryTheme.Radius.large)
        .overlay(
            RoundedRectangle(cornerRadius: TelemetryTheme.Radius.large)
                .stroke(TelemetryTheme.hairline(increasedContrast: contrast == .increased), lineWidth: 0.5)
        )
    }

    // MARK: - 5. Footer Command Dock
    var footerDock: some View {
        HStack(spacing: 8) {
            Button(action: onOpenWebUI) {
                HStack(spacing: 4) {
                    Image(systemName: "globe")
                        .font(TelemetryTheme.label)
                    Text("Web UI")
                        .font(TelemetryTheme.label)
                }
                .foregroundColor(TelemetryTheme.cool)
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(.glass)
            .help("Open Web Dashboard in Browser (⌘D)")

            Button(action: onOpenLogs) {
                HStack(spacing: 4) {
                    Image(systemName: "terminal")
                        .font(TelemetryTheme.label)
                    Text("Logs")
                        .font(TelemetryTheme.label)
                }
                .foregroundColor(.white.opacity(0.80))
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(.glass)
            .help("Open Live Logs Window (⌘L)")

            Button(action: onOpenSettings) {
                HStack(spacing: 4) {
                    Image(systemName: "gearshape")
                        .font(TelemetryTheme.label)
                    Text("Settings")
                        .font(TelemetryTheme.label)
                }
                .foregroundColor(.white.opacity(0.80))
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(.glass)
            .help("Open Engine Settings (⌘,)")

            Button(action: onQuit) {
                HStack(spacing: 4) {
                    Image(systemName: "power")
                        .font(TelemetryTheme.label)
                    Text("Quit")
                        .font(TelemetryTheme.label)
                }
                .foregroundColor(TelemetryTheme.failure.opacity(0.90))
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(.glass)
            .help("Quit QWEGN Engine (⌘Q)")
        }
        .padding(.horizontal, 2)
    }
}

extension CupertinoTelemetryView {
    /// The published card with this id. The MenuBar draws what the
    /// server published; it computes nothing of its own.
    func card(_ id: String) -> QWDashboardCard? {
        telemetry.dashboard?.cards.first { $0.id == id }
    }

    /// The published action with this id. An action the model cannot
    /// perform is not published, so there is nothing to draw.
    func action(_ id: String) -> QWDashboardAction? {
        telemetry.dashboard?.actions.first { $0.id == id }
    }

    /// Formats large token integers cleanly (e.g., 2019842 -> 2.02M, 127793 -> 127.8k)
    /// so numbers never crowd or clip their unit labels.
    func displayValue(for card: QWDashboardCard?) -> String {
        guard let raw = card?.value else { return "—" }
        if card?.unit == "tok", let n = Double(raw), n >= 10_000 {
            if n >= 1_000_000 {
                return String(format: "%.2fM", n / 1_000_000.0)
            } else {
                return String(format: "%.1fk", n / 1_000.0)
            }
        }
        return raw
    }

    /// One measurement, in the panel's one tile shape. The title is the
    /// panel's short name; the value is the server's. A reading the
    /// server has not taken is a dash, and its reason is a tooltip.
    func metricTile(_ title: String, _ card: QWDashboardCard?, accent: Color) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title)
                .font(TelemetryTheme.ui(9, .medium))
                .foregroundColor(TelemetryTheme.secondary(increasedContrast: contrast == .increased))
                .lineLimit(1)
                .minimumScaleFactor(0.8)

            HStack(alignment: .firstTextBaseline, spacing: 2) {
                Text(displayValue(for: card))
                    .font(TelemetryTheme.numeric(13, .semibold))
                    .foregroundColor(card?.value == nil ? .white.opacity(0.40) : accent)
                    .lineLimit(1)
                    .minimumScaleFactor(0.65)
                if let unit = card?.unit {
                    Text(unit)
                        .font(TelemetryTheme.ui(8))
                        .foregroundColor(TelemetryTheme.secondary(increasedContrast: contrast == .increased))
                        .lineLimit(1)
                }
            }

            Spacer(minLength: 0)

            // A bar for a percentage, the age of a reading that has gone
            // stale. One or the other: the tile has room for one, and
            // every tile has the same room.
            if card?.isStale == true {
                Text(ageText(card))
                    .font(TelemetryTheme.numeric(8))
                    .foregroundColor(TelemetryTheme.quiet(increasedContrast: contrast == .increased))
                    .lineLimit(1)
            } else if let number = card?.number, card?.unit == "%" {
                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        Capsule().fill(Color.black.opacity(0.38))
                        Capsule().fill(accent)
                            .frame(width: max(0, min(geo.size.width,
                                                     geo.size.width * CGFloat(number / 100.0))))
                    }
                }
                .frame(height: 3)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .frame(minHeight: 48, maxHeight: .infinity, alignment: .topLeading)
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .background(Color.black.opacity(0.24))
        .cornerRadius(TelemetryTheme.Radius.medium)
        .overlay(
            RoundedRectangle(cornerRadius: TelemetryTheme.Radius.medium)
                .stroke(TelemetryTheme.hairline(increasedContrast: contrast == .increased), lineWidth: 0.5)
        )
        .help(card?.reason ?? (card?.value.map { "\(title): \($0) \(card?.unit ?? "")" } ?? ""))
    }

    /// How old a reading is, in words.
    func ageText(_ card: QWDashboardCard?) -> String {
        guard let age = card?.age else { return "" }
        if age < 60 { return "\(Int(age)) s ago" }
        return "\(Int(age / 60)) min ago"
    }
}

private final class ModelSelectorPayload: NSObject {
    let telemetry: TelemetryObservable
    let model: DiscoveredModel

    init(telemetry: TelemetryObservable, model: DiscoveredModel) {
        self.telemetry = telemetry
        self.model = model
    }
}

@MainActor
private final class ModelSelectorMenuTarget: NSObject {
    static let shared = ModelSelectorMenuTarget()

    @objc func selectModel(_ sender: NSMenuItem) {
        guard let payload = sender.representedObject as? ModelSelectorPayload else { return }
        payload.telemetry.switchModel(to: payload.model)
    }

    @objc func refreshModels(_ sender: NSMenuItem) {
        guard let telemetry = sender.representedObject as? TelemetryObservable else { return }
        telemetry.refreshDiscoveredModels()
    }

    @objc func revealFinder(_ sender: NSMenuItem) {
        guard let telemetry = sender.representedObject as? TelemetryObservable else { return }
        telemetry.revealModelsFolder()
    }
}
