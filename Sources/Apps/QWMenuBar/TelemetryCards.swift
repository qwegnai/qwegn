// TelemetryCards.swift — Popover header, model lifecycle, unified metrics, and dock

import AppKit
import QWCore
import QWServer
import SwiftUI

extension CupertinoTelemetryView {
    // MARK: - Header (Daemon Controls Separated from Model Lifecycle)

    var headerSection: some View {
        let daemonBusy = telemetry.engineState.isTransitioning || telemetry.isDaemonTransitioning
        let modelBusy = telemetry.isModelTransitioning
            || telemetry.enginePhase == .loading
            || telemetry.enginePhase == .unloading
        let isRunning = telemetry.isAnyServerRunning
        let powerTint = isRunning ? TelemetryTheme.failure : TelemetryTheme.active

        return HStack(spacing: 10) {
            ZStack {
                Circle()
                    .fill(statusDotColor.opacity(isActivelyGenerating && !reduceMotion ? 0.16 : 0.08))
                    .frame(width: 32, height: 32)
                QGProcessorLogoView()
                    .frame(width: 20, height: 20)
                    .foregroundStyle(TelemetryTheme.logoInk)
            }

            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 5) {
                    Text("Qwegn")
                        .font(TelemetryTheme.ui(13, .semibold))
                        .foregroundStyle(TelemetryTheme.primary(increasedContrast: contrast == .increased))
                    Button {
                        onOpenUpdates(false)
                    } label: {
                        Text("v\(telemetry.engineVersion)")
                            .font(TelemetryTheme.numeric(11, .medium))
                            .foregroundStyle(TelemetryTheme.secondary(increasedContrast: contrast == .increased))
                    }
                    .buttonStyle(.plain)
                    .help("Check for updates")
                    Text(":\(telemetry.serverPortString)")
                        .font(TelemetryTheme.numeric(11))
                        .foregroundStyle(TelemetryTheme.quiet(increasedContrast: contrast == .increased))
                    if telemetry.updater.shouldShowBadge {
                        Button {
                            onOpenUpdates(false)
                        } label: {
                            Text(telemetry.updater.badgeText)
                                .font(TelemetryTheme.numeric(9.5, .semibold))
                                .foregroundStyle(TelemetryTheme.accent)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 1.5)
                                .background(TelemetryTheme.accent.opacity(0.14))
                                .clipShape(Capsule())
                                .overlay(
                                    Capsule()
                                        .stroke(TelemetryTheme.accent.opacity(0.35), lineWidth: 0.6)
                                )
                                .fixedSize(horizontal: true, vertical: false)
                        }
                        .buttonStyle(.plain)
                        .help("Open Software Update window")
                    }
                }

                HStack(spacing: 5) {
                    Circle()
                        .fill(statusDotColor)
                        .frame(width: 7, height: 7)
                    Text(telemetry.serverStatusText)
                        .font(TelemetryTheme.ui(11, .medium))
                        .foregroundStyle(TelemetryTheme.secondary(increasedContrast: contrast == .increased))
                        .lineLimit(1)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            HStack(spacing: 6) {
                if isRunning {
                    headerActionButton(
                        title: "Restart",
                        icon: "arrow.clockwise",
                        tint: TelemetryTheme.accent,
                        busy: false,
                        disabled: daemonBusy || modelBusy
                    ) {
                        telemetry.restartServerProcess()
                    }
                    .help("Restart qwg server (⌘R)")

                    headerActionButton(
                        title: "Stop",
                        icon: "power",
                        tint: powerTint,
                        busy: daemonBusy,
                        disabled: daemonBusy
                    ) {
                        telemetry.toggleDaemon()
                    }
                    .help("Stop qwg server")
                } else {
                    headerActionButton(
                        title: telemetry.daemonActionTitle,
                        icon: telemetry.daemonActionIcon,
                        tint: powerTint,
                        busy: daemonBusy,
                        disabled: daemonBusy
                    ) {
                        telemetry.toggleDaemon()
                    }
                }
            }
            .fixedSize(horizontal: true, vertical: false)
            .layoutPriority(1)
        }
        .padding(.leading, 2)
    }

    // MARK: - Resident Model Card

    var loadedWeightsCard: some View {
        let modelName = telemetry.dashboard?.lifecycle.model
        let modelBusy = telemetry.isModelTransitioning
            || telemetry.enginePhase == .loading
            || telemetry.enginePhase == .unloading
        let traits = residentTraitPills

        return VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .center, spacing: 8) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("RESIDENT MODEL")
                        .font(TelemetryTheme.ui(10, .semibold))
                        .tracking(0.4)
                        .foregroundStyle(TelemetryTheme.quiet(increasedContrast: contrast == .increased))
                    MarqueeText(
                        text: modelName ?? (telemetry.isConnected ? "No model loaded" : "—"),
                        fontSize: 13,
                        weight: .semibold,
                        nsWeight: .semibold,
                        color: TelemetryTheme.primary(increasedContrast: contrast == .increased)
                    )
                }
                .frame(maxWidth: .infinity, alignment: .leading)

                HStack(spacing: 6) {
                    modelSwitcherControl

                    if let unloadAction = action("unload") {
                        lifecycleButton(
                            title: modelBusy ? "Working…" : "Unload",
                            icon: "eject.fill",
                            tint: TelemetryTheme.warning,
                            busy: modelBusy,
                            disabled: modelBusy || !unloadAction.isEnabled || !telemetry.isConnected
                        ) {
                            telemetry.unloadModel()
                        }
                        .help("Unload model weights from memory")
                    } else if let loadAction = action("load") {
                        lifecycleButton(
                            title: modelBusy ? "Loading…" : "Load",
                            icon: "arrow.down.circle.fill",
                            tint: TelemetryTheme.active,
                            busy: modelBusy,
                            disabled: modelBusy || !loadAction.isEnabled || !telemetry.isConnected
                        ) {
                            telemetry.reloadModel()
                        }
                        .help(loadAction.reason ?? "Load active model into memory")
                    }
                }
                .fixedSize(horizontal: true, vertical: false)
                .layoutPriority(1)
            }

            if telemetry.isModelLoaded, !traits.isEmpty {
                HStack(spacing: 5) {
                    ForEach(traits, id: \.text) { trait in
                        if trait.allowsCompression {
                            marqueeFactPill(trait.text, tint: trait.tint, help: trait.help)
                                .layoutPriority(1)
                        } else {
                            factPill(trait.text, tint: trait.tint)
                                .fixedSize(horizontal: true, vertical: false)
                                .layoutPriority(2)
                                .help(trait.help ?? trait.text)
                        }
                    }
                    Spacer(minLength: 0)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(TelemetryTheme.substrate)
        .clipShape(.rect(cornerRadius: TelemetryTheme.Radius.large))
        .overlay(
            RoundedRectangle(cornerRadius: TelemetryTheme.Radius.large)
                .stroke(TelemetryTheme.hairline(increasedContrast: contrast == .increased), lineWidth: 0.75)
        )
    }

    @ViewBuilder
    var decisionSidecarCard: some View {
        if let facts = telemetry.dashboard?.facts, facts.decisionSidecarLoaded == true {
            let sidecarName = facts.decisionSidecarModel ?? "—"
            let backendLabel: String? = {
                let raw = (facts.decisionSidecarBackend ?? facts.decisionSidecarComputeUnit ?? "").lowercased()
                if raw.contains("ane") { return "ANE" }
                if raw.contains("metal") { return "Metal 4" }
                return facts.decisionSidecarBackend
            }()
            let latText = facts.decisionSidecarMeanLatencyMs.map { String(format: "%.1f ms", $0) } ?? "— ms"
            let memText = facts.decisionSidecarBytes.flatMap { bytes -> String? in
                guard bytes > 0 else { return nil }
                return ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .memory)
            }
            let archLabel: String? = facts.decisionSidecarArchetype.flatMap { raw -> String? in
                switch raw {
                case "verbalizer", "causal_verbalizer": return "Verbalizer"
                case "optionMarkers": return "Markers"
                case "optionMarkers+CoreAI": return "Markers ANE"
                case "verbalizer+CoreAI": return "Verbalizer ANE"
                case "sequencePair", "pooled_classifier", "encoder_pointer": return "CrossEncoder"
                default: return raw.isEmpty ? nil : raw
                }
            }
            let visionLabel: String? = facts.decisionSidecarVision.map { $0 ? "Vision" : "Text-Only" }
            let tokensText: String? = facts.decisionSidecarPromptTokens.flatMap { $0 > 0 ? "\($0) tok" : nil }

            VStack(alignment: .leading, spacing: 6) {
                HStack(alignment: .center, spacing: 8) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("DECISION SIDECAR")
                            .font(TelemetryTheme.ui(10, .semibold))
                            .tracking(0.4)
                            .foregroundStyle(TelemetryTheme.quiet(increasedContrast: contrast == .increased))
                        MarqueeText(
                            text: sidecarName,
                            fontSize: 12.5,
                            weight: .semibold,
                            nsWeight: .semibold,
                            color: TelemetryTheme.primary(increasedContrast: contrast == .increased)
                        )
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)

                    lifecycleButton(
                        title: "Unload",
                        icon: "eject.fill",
                        tint: TelemetryTheme.warning,
                        busy: false,
                        disabled: !telemetry.isConnected
                    ) {
                        telemetry.unloadDecisionSidecar()
                    }
                    .fixedSize(horizontal: true, vertical: false)
                    .layoutPriority(1)
                    .help("Unload System-1 decision sidecar")
                }

                HStack(spacing: 5) {
                    if let backendLabel, !backendLabel.isEmpty {
                        factPill(backendLabel, tint: backendLabel == "ANE" ? TelemetryTheme.active : TelemetryTheme.accent)
                            .fixedSize(horizontal: true, vertical: false)
                    }
                    factPill(latText, tint: TelemetryTheme.neutral)
                        .fixedSize(horizontal: true, vertical: false)
                    if let memText {
                        factPill(memText, tint: TelemetryTheme.cool)
                            .fixedSize(horizontal: true, vertical: false)
                    }
                    if let reqs = facts.decisionSidecarTotalRequests, reqs > 0 {
                        factPill("\(reqs) req", tint: TelemetryTheme.neutral)
                            .fixedSize(horizontal: true, vertical: false)
                    }
                    Spacer(minLength: 0)
                }
                .frame(maxWidth: .infinity, alignment: .leading)

                if visionLabel != nil || archLabel != nil || tokensText != nil {
                    HStack(spacing: 5) {
                        if let visionLabel {
                            factPill(
                                visionLabel,
                                tint: facts.decisionSidecarVision == true ? TelemetryTheme.cool : TelemetryTheme.neutral
                            )
                            .fixedSize(horizontal: true, vertical: false)
                        }
                        if let archLabel {
                            factPill(archLabel, tint: TelemetryTheme.neutral)
                                .fixedSize(horizontal: true, vertical: false)
                        }
                        if let tokensText {
                            factPill(tokensText, tint: TelemetryTheme.neutral)
                                .fixedSize(horizontal: true, vertical: false)
                        }
                        Spacer(minLength: 0)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 9)
            .background(TelemetryTheme.substrate)
            .clipShape(.rect(cornerRadius: TelemetryTheme.Radius.large))
            .overlay(
                RoundedRectangle(cornerRadius: TelemetryTheme.Radius.large)
                    .stroke(TelemetryTheme.hairline(increasedContrast: contrast == .increased), lineWidth: 0.75)
            )
        }
    }

    private struct TraitPill: Equatable {
        let text: String
        let tint: Color
        let help: String?
        let allowsCompression: Bool
    }

    private var residentTraitPills: [TraitPill] {
        guard let dashboard = telemetry.dashboard else { return [] }
        var pills: [TraitPill] = []

        if let q = dashboard.facts?.quantization, !q.isEmpty {
            let cleanQ = q.replacingOccurrences(of: " (online GPU)", with: "")
            pills.append(TraitPill(text: cleanQ, tint: TelemetryTheme.accent, help: q, allowsCompression: true))
        }
        if let kv = dashboard.knobs["kvMode"]?.stringValue, !kv.isEmpty {
            pills.append(TraitPill(text: "KV \(kv.uppercased())", tint: TelemetryTheme.neutral, help: "KV cache format", allowsCompression: false))
        }
        if let budget = telemetry.contextBudget, budget > 0 {
            let compact = QWMenuBarStatusFormatter.formatCompactBudget(budget)
            pills.append(TraitPill(text: "\(compact) ctx", tint: TelemetryTheme.neutral, help: "Context budget", allowsCompression: false))
        }
        let specEnabled = dashboard.knobs["speculativeEnabled"]?.boolValue ?? true
        if specEnabled, let drafts = dashboard.knobs["mtpDrafts"]?.intValue, drafts > 0 {
            pills.append(TraitPill(text: "Spec ×\(drafts)", tint: TelemetryTheme.prefill, help: "Speculative draft tokens", allowsCompression: false))
        }
        if let memPack = dashboard.facts?.mountedMemoryPack, !memPack.isEmpty {
            pills.append(TraitPill(text: "MemPack", tint: TelemetryTheme.cool, help: memPack, allowsCompression: false))
        }
        return pills
    }

    private var modelSwitcherControl: some View {
        Button(action: presentModelSwitcherMenu) {
            HStack(spacing: 4) {
                Text("Switch")
                    .font(TelemetryTheme.ui(11, .medium))
                Image(systemName: "chevron.up.chevron.down")
                    .font(.system(size: 9.5, weight: .medium))
            }
            .foregroundStyle(TelemetryTheme.secondary(increasedContrast: contrast == .increased))
            .padding(.horizontal, 8)
            .frame(height: 25)
            .background(TelemetryTheme.rowSubstrate)
            .clipShape(.rect(cornerRadius: TelemetryTheme.Radius.small))
            .overlay(
                RoundedRectangle(cornerRadius: TelemetryTheme.Radius.small)
                    .stroke(TelemetryTheme.hairline(increasedContrast: contrast == .increased), lineWidth: 0.5)
            )
            .background(MenuAnchorBridge { switcherAnchor = $0 })
        }
        .buttonStyle(.plain)
        .help("Switch active model pack")
        .onAppear { telemetry.refreshDiscoveredModels() }
    }

    private func presentModelSwitcherMenu() {
        telemetry.refreshDiscoveredModels()
        let menu = NSMenu()
        let currentModel = telemetry.dashboard?.lifecycle.model
        let targets = NSMutableArray()

        if telemetry.discoveredModels.isEmpty {
            let empty = NSMenuItem(title: "No models in ~/.qwg/models", action: nil, keyEquivalent: "")
            empty.isEnabled = false
            menu.addItem(empty)
        } else {
            for model in telemetry.discoveredModels {
                let target = MenuClosureTarget { telemetry.switchModel(to: model) }
                targets.add(target)
                let item = NSMenuItem(title: model.name, action: #selector(MenuClosureTarget.invoke), keyEquivalent: "")
                item.target = target
                if model.name == currentModel { item.state = .on }
                menu.addItem(item)
            }
        }

        menu.addItem(.separator())
        let refreshTarget = MenuClosureTarget { telemetry.refreshDiscoveredModels() }
        targets.add(refreshTarget)
        let refreshItem = NSMenuItem(title: "Refresh Models List", action: #selector(MenuClosureTarget.invoke), keyEquivalent: "")
        refreshItem.target = refreshTarget
        menu.addItem(refreshItem)

        let finderTarget = MenuClosureTarget { telemetry.revealModelsFolder() }
        targets.add(finderTarget)
        let finderItem = NSMenuItem(title: "Open ~/.qwg/models in Finder", action: #selector(MenuClosureTarget.invoke), keyEquivalent: "")
        finderItem.target = finderTarget
        menu.addItem(finderItem)

        objc_setAssociatedObject(menu, &menuTargetsAssocKey, targets, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
        guard let anchor = switcherAnchor ?? NSApp.keyWindow?.contentView else { return }
        menu.popUp(positioning: nil, at: NSPoint(x: 0, y: anchor.bounds.height + 4), in: anchor)
    }

    // MARK: - Unified Inference Performance Card

    var inferenceTelemetryCard: some View {
        let decodeCard = card("throughput.display_decode_tokens_per_sec")
        let prefillCard = card("throughput.display_prefill_tokens_per_sec")
        let prefillProgCard = card("prefill.progress_tokens")
        let prefillTotalCard = card("prefill.total_tokens")
        let ttftCard = card("latency.ttft_ms")
        let contextCard = card("context.active_tokens")
        let engGpuCard = card("gpu.busy_ratio")

        let prefillSeries = prefillCard?.series ?? []
        let decodeSeries = decodeCard?.series ?? []
        let hasPrefillTrace = prefillSeries.contains { $0 != nil }
        let hasDecodeTrace = decodeSeries.contains { $0 != nil }
        let prefillProgNum = prefillProgCard?.number
        let prefillTotalNum = prefillTotalCard?.number
        let isPrefilling = telemetry.dashboard?.generation == .prefill
            && prefillProgNum != nil
            && (prefillTotalNum ?? 0) > 0

        return VStack(spacing: 0) {
            HStack(spacing: 0) {
                heroMetricColumn(title: "Decode", card: decodeCard, tint: TelemetryTheme.accent, valueTemplate: "999.9")
                verticalHairline
                heroMetricColumn(title: "Prefill", card: prefillCard, tint: TelemetryTheme.prefill, valueTemplate: "9999.9")
            }
            .padding(.vertical, 10)

            horizontalHairline
            prefillProgressBarRow(
                isActive: isPrefilling,
                prog: prefillProgNum,
                total: prefillTotalNum,
                progValue: prefillProgCard?.value,
                totalText: formattedCardValue(prefillTotalCard),
                reason: prefillProgCard?.reason ?? prefillTotalCard?.reason
            )

            horizontalHairline

            HStack(spacing: 0) {
                inlineMetricCell(
                    label: "TTFT",
                    valueText: formattedCardValue(ttftCard, minValueWidth: 4),
                    isAvailable: ttftCard?.value != nil,
                    tint: TelemetryTheme.cool,
                    help: ttftCard?.reason
                )
                verticalHairline
                inlineMetricCell(
                    label: "Context",
                    valueText: formattedContextSummary(contextCard),
                    isAvailable: contextCard?.value != nil || telemetry.contextBudget != nil,
                    tint: TelemetryTheme.primary(increasedContrast: contrast == .increased),
                    help: contextCard?.reason ?? "Active context usage against configured budget"
                )
                verticalHairline
                inlineMetricCell(
                    label: "Eng GPU",
                    valueText: formattedCardValue(engGpuCard, minValueWidth: 5),
                    isAvailable: engGpuCard?.value != nil,
                    tint: TelemetryTheme.accent,
                    help: engGpuCard?.reason
                )
            }
            .padding(.vertical, 8)

            if hasPrefillTrace || hasDecodeTrace {
                horizontalHairline
                VStack(alignment: .leading, spacing: 5) {
                    HStack(spacing: 8) {
                        Text("THROUGHPUT TRACE")
                            .font(TelemetryTheme.ui(10, .semibold))
                            .tracking(0.4)
                            .foregroundStyle(TelemetryTheme.quiet(increasedContrast: contrast == .increased))
                        Spacer()
                        if hasPrefillTrace {
                            legendChip("Prefill", color: TelemetryTheme.prefill)
                        }
                        if hasDecodeTrace {
                            legendChip("Decode", color: TelemetryTheme.accent)
                        }
                    }
                    DualSparklineView(prefillData: prefillSeries, decodeData: decodeSeries)
                        .frame(height: 50)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 9)
            }
        }
        .background(TelemetryTheme.substrate)
        .clipShape(.rect(cornerRadius: TelemetryTheme.Radius.large))
        .overlay(
            RoundedRectangle(cornerRadius: TelemetryTheme.Radius.large)
                .stroke(TelemetryTheme.hairline(increasedContrast: contrast == .increased), lineWidth: 0.75)
        )
    }

    // MARK: - Unified Memory, Cache & System Card

    var memoryAndSystemCard: some View {
        let arenaCard = card("memory.arena_reserved_bytes")
        let residentCard = card("memory.resident_bytes")
        let engineMemCard = (arenaCard?.value != nil ? arenaCard : residentCard)
        let hasPrefixCache = telemetry.dashboard?.capabilities.contains("prefixCache") == true
        let warmCard = card("cache.warm_bytes")
        let ssdCard = card("cache.ssd_bytes")
        let reusedCard = card("cache.tokens_reused_total")
        let memCard = card("hardware.memory_percent")
        let pressureCard = card("hardware.memory_pressure_level")
        let gpuTempCard = card("hardware.gpu_temp_celsius")
        let fanCard = card("hardware.fan_rpm")
        let powerCard = card("hardware.system_power_watts")
        let thermalCard = card("hardware.thermal_state")

        return VStack(spacing: 0) {
            HStack(spacing: 0) {
                inlineMetricCell(
                    label: "Engine RAM",
                    valueText: formattedCardValue(engineMemCard, minValueWidth: 6),
                    isAvailable: engineMemCard?.value != nil,
                    tint: TelemetryTheme.highlight,
                    help: engineMemoryHelp(arena: arenaCard, resident: residentCard)
                )

                if hasPrefixCache {
                    verticalHairline
                    inlineMetricCell(
                        label: "Warm RAM",
                        valueText: formattedCardValue(warmCard, minValueWidth: 5),
                        isAvailable: warmCard?.value != nil,
                        tint: TelemetryTheme.cool,
                        help: warmCard?.reason
                    )
                    verticalHairline
                    inlineMetricCell(
                        label: "SSD Cache",
                        valueText: formattedCardValue(ssdCard, minValueWidth: 5),
                        isAvailable: ssdCard?.value != nil,
                        tint: TelemetryTheme.highlight,
                        help: ssdCard?.reason
                    )
                    verticalHairline
                    inlineMetricCell(
                        label: "Reused",
                        valueText: formattedCardValue(reusedCard),
                        isAvailable: reusedCard?.value != nil,
                        tint: TelemetryTheme.active,
                        help: reusedCard?.reason
                    )
                } else {
                    verticalHairline
                    inlineMetricCell(
                        label: "Weights",
                        valueText: formattedCardValue(residentCard, minValueWidth: 6),
                        isAvailable: residentCard?.value != nil,
                        tint: TelemetryTheme.cool,
                        help: residentCard?.reason
                    )
                }
            }
            .padding(.vertical, 8)

            horizontalHairline

            HStack(spacing: 0) {
                siliconGaugeCell(title: "Sys GPU", card: card("gpu.device"), tint: TelemetryTheme.accent)
                verticalHairline
                siliconGaugeCell(title: "CPU", card: card("hardware.cpu_percent"), tint: TelemetryTheme.warning)
                verticalHairline
                siliconGaugeCell(
                    title: "Memory",
                    card: memCard,
                    tint: memoryBarTint(pressureCard: pressureCard),
                    helpSuffix: pressureCard?.value
                )
            }
            .padding(.vertical, 8)

            horizontalHairline

            HStack(spacing: 0) {
                inlineMetricCell(
                    label: "GPU Temp",
                    valueText: formattedCardValue(gpuTempCard, minValueWidth: 5),
                    isAvailable: gpuTempCard?.value != nil,
                    tint: tempTint(gpuTempCard),
                    help: gpuTempCard?.reason ?? "AppleSMC GPU hotspot temperature"
                )
                verticalHairline
                inlineMetricCell(
                    label: "Fans",
                    valueText: formattedCardValue(fanCard, minValueWidth: 4),
                    isAvailable: fanCard?.value != nil,
                    tint: TelemetryTheme.cool,
                    help: fanCard?.reason ?? telemetry.dashboard?.facts?.fanControlStatus ?? "Cooling fan speed"
                )
                verticalHairline
                inlineMetricCell(
                    label: "Power",
                    valueText: formattedCardValue(powerCard, minValueWidth: 5),
                    isAvailable: powerCard?.value != nil,
                    tint: TelemetryTheme.secondary(increasedContrast: contrast == .increased),
                    help: powerCard?.reason ?? "AppleSMC system power draw"
                )
                verticalHairline
                inlineMetricCell(
                    label: "Thermal",
                    valueText: formattedCardValue(thermalCard),
                    isAvailable: thermalCard?.value != nil,
                    tint: thermalStateTint(thermalCard),
                    help: thermalCard?.reason ?? "macOS thermal pressure state"
                )
            }
            .padding(.vertical, 8)
        }
        .background(TelemetryTheme.substrate)
        .clipShape(.rect(cornerRadius: TelemetryTheme.Radius.large))
        .overlay(
            RoundedRectangle(cornerRadius: TelemetryTheme.Radius.large)
                .stroke(TelemetryTheme.hairline(increasedContrast: contrast == .increased), lineWidth: 0.75)
        )
    }

    // MARK: - Symmetric Footer Dock

    var footerDock: some View {
        HStack(spacing: 8) {
            dockButton(title: "Web UI", icon: "globe", shortcut: "⌘D", action: onOpenWebUI)
            dockButton(title: "Logs", icon: "doc.text.magnifyingglass", shortcut: "⌘L", action: onOpenLogs)
            dockButton(title: "Settings", icon: "slider.horizontal.3", shortcut: "⌘,", action: onOpenSettings)
            dockButton(title: "Quit", icon: "power", shortcut: "⌘Q", isDestructive: true, action: onQuit)
        }
    }

    // MARK: - Card & Control Builders

    private func card(_ id: String) -> QWDashboardCard? {
        telemetry.dashboard?.cards.first { $0.id == id }
    }

    private func action(_ id: String) -> QWDashboardAction? {
        telemetry.dashboard?.actions.first { $0.id == id }
    }

    private func tempTint(_ tempCard: QWDashboardCard?) -> Color {
        guard let c = tempCard?.number else { return TelemetryTheme.secondary(increasedContrast: contrast == .increased) }
        if c >= 85 { return TelemetryTheme.failure }
        if c >= 72 { return TelemetryTheme.warning }
        return TelemetryTheme.prefill
    }

    private func thermalStateTint(_ stateCard: QWDashboardCard?) -> Color {
        switch stateCard?.value?.lowercased() {
        case "critical", "serious": return TelemetryTheme.failure
        case "fair": return TelemetryTheme.warning
        default: return TelemetryTheme.active
        }
    }

    private func prefillProgressBarRow(
        isActive: Bool,
        prog: Double?,
        total: Double?,
        progValue: String?,
        totalText: String,
        reason: String?
    ) -> some View {
        let fraction: Double = if isActive, let prog, let total, total > 0 {
            min(1.0, max(0.0, prog / total))
        } else {
            0
        }
        let pct = Int((fraction * 100.0).rounded())
        let statusText = isActive
            ? "\(progValue ?? "—") / \(totalText) · \(String(format: "%3d%%", pct))"
            : "—"
        let secondaryInk = TelemetryTheme.secondary(increasedContrast: contrast == .increased)
        let quietInk = TelemetryTheme.quiet(increasedContrast: contrast == .increased)

        return VStack(alignment: .leading, spacing: 5) {
            HStack(spacing: 6) {
                Text("PREFILL PROGRESS")
                    .font(TelemetryTheme.ui(9.5, .semibold))
                    .tracking(0.4)
                    .foregroundStyle(isActive ? TelemetryTheme.prefill : quietInk)
                Spacer(minLength: 4)
                Text(statusText)
                    .font(TelemetryTheme.numeric(11, .medium))
                    .foregroundStyle(isActive ? secondaryInk : quietInk)
                    .lineLimit(1)
                    .minimumScaleFactor(0.85)
            }
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 2.5)
                        .fill(TelemetryTheme.rowSubstrate)
                        .frame(height: 4.5)
                    RoundedRectangle(cornerRadius: 2.5)
                        .fill(TelemetryTheme.prefill)
                        .frame(width: isActive ? max(fraction > 0 ? 3 : 0, geo.size.width * fraction) : 0, height: 4.5)
                }
            }
            .frame(height: 4.5)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .help(isActive ? "" : (reason ?? ""))
    }
}
