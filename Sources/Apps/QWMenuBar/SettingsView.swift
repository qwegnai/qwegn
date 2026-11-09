import SwiftUI
import AppKit
import ServiceManagement
import QWCore
import QWServer

/// Comprehensive, perfectly aligned settings window for all server, kernel, memory, and logging parameters.
public struct SettingsView: View {
    @ObservedObject public var telemetry: TelemetryObservable
    /// Increased Contrast asks for stronger text. L-M7: it is honoured.
    @Environment(\.colorSchemeContrast) internal var contrast: ColorSchemeContrast
    @State var draftSettings: QWServerSettings
    @State var activeTab: SettingsTab = .general
    @State var saveConfirmationMessage: String?
    @State var validationErrorMessage: String?
    @State var discoveredModels: [DiscoveredModel] = []
    @State var discoveredDrafterModels: [DiscoveredModel] = []
    public var onClose: () -> Void

    public enum SettingsTab: String, CaseIterable, Identifiable {
        case general = "General"
        case model = "Model"
        case sampling = "Sampling"
        case speculative = "Speculative"
        case performance = "Performance"
        case cache = "KV Cache"
        case logging = "Logging"

        public var id: String { rawValue }

        public var icon: String {
            switch self {
            case .general: return "gearshape"
            case .model: return "brain"
            case .sampling: return "slider.horizontal.below.rectangle"
            case .speculative: return "bolt.badge.clock"
            case .performance: return "gauge.with.dots.needle.67percent"
            case .cache: return "internaldrive"
            case .logging: return "doc.text"
            }
        }
    }

    public init(telemetry: TelemetryObservable, onClose: @escaping () -> Void = {}) {
        self.telemetry = telemetry
        self._draftSettings = State(initialValue: telemetry.settings)
        self.onClose = onClose
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
                // Header Bar
                HStack(spacing: 8) {
                    Image(systemName: "gearshape.fill")
                        .font(.system(size: 13))
                        .foregroundColor(TelemetryTheme.accent)
                    Text("QWEGN Engine Settings")
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundColor(.white)
                    Spacer()
                    Text("~/.qwg/settings.json")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundColor(TelemetryTheme.quiet(increasedContrast: contrast == .increased))
                    Button(action: onClose) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 14))
                            .foregroundColor(TelemetryTheme.quiet(increasedContrast: contrast == .increased))
                    }
                    .buttonStyle(.plain)
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 10)
                .background(Color.white.opacity(0.04))

                Divider().background(Color.white.opacity(0.08))

                // Tab Switcher
                HStack(spacing: 4) {
                    ForEach(SettingsTab.allCases) { tab in
                        Button(action: { activeTab = tab }) {
                            HStack(spacing: 5) {
                                Image(systemName: tab.icon)
                                    .font(.system(size: 11))
                                Text(tab.rawValue)
                                    .font(.system(size: 11, weight: activeTab == tab ? .semibold : .regular))
                            }
                            .foregroundColor(activeTab == tab ? .white : .white.opacity(0.60))
                            .padding(.horizontal, 8)
                            .padding(.vertical, 5)
                            .background(activeTab == tab ? TelemetryTheme.accent.opacity(0.35) : TelemetryTheme.rowSubstrate)
                            .cornerRadius(6)
                            .overlay(RoundedRectangle(cornerRadius: 6).stroke(activeTab == tab ? TelemetryTheme.accent.opacity(0.60) : Color.white.opacity(0.06), lineWidth: 0.5))
                        }
                        .buttonStyle(.plain)
                    }
                    Spacer()
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .background(Color.black.opacity(0.20))

                Divider().background(Color.white.opacity(0.08))

                // Main Tab Content
                ScrollView(.vertical, showsIndicators: true) {
                    VStack(alignment: .leading, spacing: 14) {
                        pendingRestartPanel
                        switch activeTab {
                        case .general:
                            generalSection
                        case .model:
                            modelSection
                        case .sampling:
                            samplingSection
                        case .speculative:
                            speculativeSection
                        case .performance:
                            performanceSection
                        case .cache:
                            cacheSection
                        case .logging:
                            loggingSection
                        }
                    }
                    .padding(16)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)

                Divider().background(Color.white.opacity(0.08))

                // Bottom Action Bar
                HStack(spacing: 10) {
                    Button("Reset to Defaults") {
                        resetToTopPerformanceDefaults()
                    }
                    .buttonStyle(.plain)
                    .font(.system(size: 11))
                    .foregroundColor(.white.opacity(0.60))
                    .help("Reinstate the engine's top-performance default parameters for the selected model")

                    Text("•")
                        .font(.system(size: 10))
                        .foregroundColor(TelemetryTheme.separator(increasedContrast: contrast == .increased))

                    Button(action: revealSettingsInFinder) {
                        HStack(spacing: 3) {
                            Image(systemName: "folder")
                                .font(.system(size: 10))
                            Text("Reveal in Finder")
                        }
                    }
                    .buttonStyle(.plain)
                    .font(.system(size: 11))
                    .foregroundColor(.white.opacity(0.60))

                    Spacer()

                    if let err = validationErrorMessage {
                        Text(err)
                            .font(.system(size: 11, weight: .medium))
                            .foregroundColor(TelemetryTheme.failure)
                            .lineLimit(2)
                    } else if let msg = saveConfirmationMessage {
                        Text(msg)
                            .font(.system(size: 11, weight: .medium))
                            .foregroundColor(TelemetryTheme.active)
                    }

                    Button("Cancel") {
                        draftSettings = telemetry.settings
                        onClose()
                    }
                    .buttonStyle(.plain)
                    .font(.system(size: 11))
                    .foregroundColor(.white.opacity(0.70))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .background(Color.white.opacity(0.08))
                    .cornerRadius(6)

                    Button("Save & Apply") {
                        applyAndSave()
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.regular)
                    .disabled(draftSettings == telemetry.settings || validationErrorMessage != nil)
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 10)
                .background(Color.black.opacity(0.30))
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onAppear {
            refreshDiscoveredModels()
        }
    }

    // MARK: - 1. General & System Section
    private var generalSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            settingsGroup(title: "System Integration", icon: "macwindow") {
                Toggle("Run at startup", isOn: $draftSettings.runAtStartup)
                    .toggleStyle(.switch)
                    .onChange(of: draftSettings.runAtStartup) { _, newValue in
                        updateRunAtStartup(newValue)
                    }

                Toggle("Auto-start inference server on launch", isOn: $draftSettings.autoStartServer)
                    .toggleStyle(.switch)
            }

            settingsGroup(title: "Server & App Lifecycle", icon: "server.rack") {
                HStack(spacing: 6) {
                    Toggle("Keep the server running after the app quits", isOn: $draftSettings.keepServerRunningAfterQuit)
                        .toggleStyle(.switch)
                    SettingsRestartBadge(key: "keepServerRunningAfterQuit")
                }
                Text(QWSettingsCoherence.effectNote(for: "keepServerRunningAfterQuit") ?? "")
                    .font(.system(size: 9))
                    .foregroundColor(TelemetryTheme.quiet(increasedContrast: contrast == .increased))
            }

            settingsGroup(title: "Network & Endpoints", icon: "network") {
                formRow(label: "Listen Host:", key: "host") {
                    TextField("127.0.0.1", text: $draftSettings.host)
                        .textFieldStyle(.roundedBorder)
                }
                formRow(label: "Listen Port:", key: "port") {
                    TextField("8080", value: $draftSettings.port, formatter: plainNumberFormatter)
                        .textFieldStyle(.roundedBorder)
                }
                formRow(label: "Poll Interval (ms):", key: "telemetryPollIntervalMs") {
                    TextField("500", value: $draftSettings.telemetryPollIntervalMs, formatter: plainNumberFormatter)
                        .textFieldStyle(.roundedBorder)
                }
            }
        }
    }

    // MARK: - 2. Model & Weights Section
    private var modelSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            settingsGroup(title: "Model Selection (~/.qwg/models)", icon: "brain") {
                formRow(label: "Installed Model:", key: "modelName") {
                    HStack(spacing: 8) {
                        Picker("", selection: $draftSettings.modelName) {
                            ForEach(discoveredModels.filter(\.supported)) { m in
                                Text(m.name).tag(m.name)
                            }
                        }
                        .pickerStyle(.menu)
                        .onChange(of: draftSettings.modelName) { _, newName in
                            if let match = discoveredModels.first(where: { $0.name == newName }), !match.packPath.isEmpty {
                                draftSettings.modelPackPath = match.packPath
                            }
                            let (effective, _) = QWSettingsManager.shared.effectiveSettings(
                                forModel: newName,
                                packPath: draftSettings.modelPackPath,
                                base: draftSettings
                            )
                            draftSettings = effective
                            validateModelAndDrafterOnSelection()
                        }

                        Button(action: refreshDiscoveredModels) {
                            Image(systemName: "arrow.clockwise")
                                .font(.system(size: 11))
                        }
                        .buttonStyle(.plain)
                        .help("Refresh models in ~/.qwg/models")

                        Button(action: revealModelsFolderInFinder) {
                            Image(systemName: "folder")
                                .font(.system(size: 11))
                        }
                        .buttonStyle(.plain)
                        .help("Open ~/.qwg/models in Finder")
                    }
                }

                formRow(label: "Model Alias:", key: "modelAlias") {
                    VStack(alignment: .leading, spacing: 2) {
                        TextField("e.g. qwen-27b (optional)", text: Binding(
                            get: { draftSettings.modelAlias },
                            set: { draftSettings.setAlias($0, for: draftSettings.modelName) }
                        ))
                            .textFieldStyle(.roundedBorder)
                        Text("Alias is persisted per model in ~/.qwg/settings.json, returned in /v1/models, and used for API responses & inference")
                            .font(.system(size: 9))
                            .foregroundColor(TelemetryTheme.quiet(increasedContrast: contrast == .increased))
                    }
                }

                VStack(alignment: .leading, spacing: 4) {
                    SettingsRowLabel(text: "Model Pack (.qwp path):", key: "modelPackPath")
                    HStack(spacing: 8) {
                        TextField("/path/to/model.qwp", text: $draftSettings.modelPackPath)
                            .textFieldStyle(.roundedBorder)
                        Button("Browse...") {
                            selectModelPackFile()
                        }
                    }
                }

                HStack(spacing: 6) {
                    Toggle("Lazy Load Model Weights on First Request", isOn: $draftSettings.lazyLoad)
                        .toggleStyle(.switch)
                    SettingsRestartBadge(key: "lazyLoad")
                }

                VStack(alignment: .leading, spacing: 2) {
                    Toggle("Load Vision Tower (Multimodal Image Input)", isOn: $draftSettings.loadVision)
                        .toggleStyle(.switch)
                        .disabled(!selectedModelSupportsVision)
                    Text(selectedModelSupportsVision
                        ? (QWSettingsCoherence.effectNote(for: "loadVision") ?? "Loads or unloads the vision tower in place without restarting the engine or resetting the KV cache.")
                        : "Selected model does not include a vision tower.")
                        .font(.system(size: 9))
                        .foregroundColor(TelemetryTheme.quiet(increasedContrast: contrast == .increased))
                }
            }

            settingsGroup(title: "Context & KV Format", icon: "slider.horizontal.3") {
                ContextLimitSliderRow(
                    maxContext: $draftSettings.maxContext,
                    estimate: selectedModelContextMemoryEstimate,
                    modelName: draftSettings.modelName,
                    kvMode: mountedMemoryPack?.enforcedKVMode ?? draftSettings.kvMode,
                    contrast: contrast
                )

                formRow(label: "KV Cache Mode:", key: "kvMode") {
                    VStack(alignment: .leading, spacing: 4) {
                        Picker("", selection: Binding(
                            get: { mountedMemoryPack?.enforcedKVMode ?? draftSettings.kvMode },
                            set: { newVal in
                                if !isKVModeLockedByMemoryPack {
                                    draftSettings.kvMode = newVal
                                }
                            }
                        )) {
                            Text("FP16").tag("f16")
                            Text("Q8_0 (8-bit)").tag("q8")
                            Text("Q4_0 (4-bit)").tag("q4")
                        }
                        .pickerStyle(.segmented)
                        .disabled(isKVModeLockedByMemoryPack)

                        if let mounted = mountedMemoryPack, mounted.enforcesKVMode {
                            HStack(spacing: 6) {
                                Image(systemName: "lock.fill")
                                    .font(.system(size: 9))
                                    .foregroundColor(TelemetryTheme.warning)
                                Text("Locked by Memory Pack (\(mounted.packName) · \(mounted.enforcedKVMode.uppercased()))")
                                    .font(.system(size: 9, weight: .medium))
                                    .foregroundColor(TelemetryTheme.warning)
                                Spacer()
                                Button("Unmount") {
                                    unmountActiveMemoryPack()
                                }
                                .buttonStyle(.bordered)
                                .controlSize(.mini)
                            }
                        }
                    }
                }
            }
        }
    }

    // MARK: - 3. Sampling & Reasoning Section
    private var samplingSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            settingsGroup(title: "Sampling & Generation Hyperparameters", icon: "slider.horizontal.3") {
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        SettingsRowLabel(text: "Temperature:", key: "temperature")
                        Spacer()
                        Text(draftSettings.temperature == 0 ? "0.00 (Greedy / Argmax)" : String(format: "%.2f", draftSettings.temperature))
                            .font(.system(size: 11, weight: .semibold, design: .monospaced))
                            .foregroundColor(draftSettings.temperature == 0 ? TelemetryTheme.warning : TelemetryTheme.accent)
                    }
                    Slider(value: $draftSettings.temperature, in: 0.0...2.0, step: 0.05)
                }

                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        SettingsRowLabel(text: "Top-P (Nucleus):", key: "topP")
                        Spacer()
                        Text(String(format: "%.2f", draftSettings.topP))
                            .font(.system(size: 11, weight: .semibold, design: .monospaced))
                            .foregroundColor(TelemetryTheme.accent)
                    }
                    Slider(value: $draftSettings.topP, in: 0.05...1.0, step: 0.05)
                }

                intSliderRow(label: "Top-K Filtering:", key: "topK", range: 0...100, step: 1,
                             readout: { $0 <= 0 ? "0 (Disabled)" : ($0 == 1 ? "1 (Greedy / Argmax)" : "\($0)") },
                             tint: { $0 <= 0 ? .white.opacity(0.40) : ($0 == 1 ? TelemetryTheme.warning : TelemetryTheme.accent) },
                             value: $draftSettings.topK)

                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        SettingsRowLabel(text: "Repetition Penalty:", key: "repetitionPenalty")
                        Spacer()
                        Text(String(format: "%.2f", draftSettings.repetitionPenalty))
                            .font(.system(size: 11, weight: .semibold, design: .monospaced))
                            .foregroundColor(TelemetryTheme.accent)
                    }
                    Slider(value: $draftSettings.repetitionPenalty, in: 1.0...1.5, step: 0.01)
                }

                formRow(label: "Max Tokens:", key: "maxTokens") {
                    Picker("", selection: $draftSettings.maxTokens) {
                        Text("256 tokens").tag(256)
                        Text("512 tokens (Default)").tag(512)
                        Text("1,024 tokens").tag(1024)
                        Text("2,048 tokens").tag(2048)
                        Text("4,096 tokens").tag(4096)
                        Text("8,192 tokens").tag(8192)
                        Text("16,384 tokens").tag(16384)
                    }
                    .pickerStyle(.menu)
                }

                formRow(label: "Random Seed:", key: "seed") {
                    VStack(alignment: .leading, spacing: 2) {
                        TextField("-1 (Random)", value: $draftSettings.seed, formatter: plainNumberFormatter)
                            .textFieldStyle(.roundedBorder)
                        Text("Set -1 for non-deterministic pseudorandom sampling")
                            .font(.system(size: 9))
                            .foregroundColor(TelemetryTheme.quiet(increasedContrast: contrast == .increased))
                    }
                }
            }

            settingsGroup(title: "Reasoning & Template Conditioning", icon: "sparkles") {
                HStack(spacing: 6) {
                    Toggle("Enable Thinking Mode (<think>...</think>)", isOn: $draftSettings.enableThinking)
                        .toggleStyle(.switch)
                    SettingsRestartBadge(key: "enableThinking")
                }

                formRow(label: "Reasoning Effort:", key: "reasoningEffort") {
                    Picker("", selection: $draftSettings.reasoningEffort) {
                        Text("Low").tag("low")
                        Text("Medium").tag("medium")
                        Text("Extra High (Default)").tag("xhigh")
                    }
                    .pickerStyle(.segmented)
                }

                VStack(alignment: .leading, spacing: 4) {
                    SettingsRowLabel(text: "Default System Prompt:", key: "systemPrompt")
                    TextField("e.g. You are a helpful AI assistant...", text: $draftSettings.systemPrompt)
                        .textFieldStyle(.roundedBorder)
                    Text("Prepended if incoming request does not provide a system role message")
                        .font(.system(size: 9))
                        .foregroundColor(TelemetryTheme.quiet(increasedContrast: contrast == .increased))
                }
            }
        }
    }

    // MARK: - 4. Speculative Decoding Section
    private var speculativeSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            settingsGroup(title: "Speculative Multi-Token Prediction (MTP)", icon: "bolt.badge.clock") {
                HStack(spacing: 6) {
                    Toggle("Enable Speculative Decoding", isOn: $draftSettings.speculativeDecodingEnabled)
                        .toggleStyle(.switch)
                    SettingsRestartBadge(key: "speculativeDecodingEnabled")
                    Spacer()
                    Button("Reset to Defaults") {
                        resetToTopPerformanceDefaults()
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .help("Reinstate top-performance defaults (all tabs) for the selected model")
                }

                formRow(label: "MTP Backend:", key: "mtpBackend") {
                    Picker("", selection: $draftSettings.mtpBackend) {
                        Text("Standard").tag("standard")
                        Text("Variational (PLM)").tag("variational")
                        if selectedModelSupportsDFlash2 {
                            Text("DFlash 2").tag("dflash2")
                        } else {
                            Text("DFlash 2 (Dense Only)").tag("dflash2")
                        }
                    }
                    .pickerStyle(.segmented)
                    .onChange(of: draftSettings.mtpBackend) { _, newBackend in
                        let maxD = selectedModelMaxDrafts
                        if newBackend != "dflash2" && draftSettings.mtpDrafts > maxD {
                            draftSettings.mtpDrafts = maxD
                            validationErrorMessage = nil
                        } else if newBackend == "dflash2" {
                            if draftSettings.mtpDrafts == 1 {
                                draftSettings.mtpDrafts = 7
                            }
                            if draftSettings.dflashDraftPackPath.isEmpty {
                                draftSettings.dflashDraftPackPath = discoveredDrafterModels.first?.packPath ?? ""
                            }
                            do {
                                try draftSettings.validateDFlashConfiguration()
                                validationErrorMessage = nil
                            } catch {
                                validationErrorMessage = "DFlash 2 fail-fast: \(error)"
                                draftSettings.mtpBackend = "variational"
                                if draftSettings.mtpDrafts > maxD { draftSettings.mtpDrafts = min(3, maxD) }
                            }
                        }
                    }
                }

                intSliderRow(label: "MTP Draft Tokens:", key: "mtpDrafts",
                             range: 0...Double(selectedModelMaxDrafts), step: 1,
                             readout: { draftSettings.mtpBackend == "dflash2" ? "\($0) drafts (B=\($0 + 1))" : "\($0) drafts" },
                             value: $draftSettings.mtpDrafts)

                if draftSettings.mtpBackend != "dflash2" {
                    formRow(label: "Draft Strategy:", key: "mtpDraftStrategy") {
                        Picker("", selection: $draftSettings.mtpDraftStrategy) {
                            Text("Hybrid (MTP + N-Gram)").tag("hybrid")
                            Text("MTP Only").tag("mtp")
                            Text("N-Gram Only").tag("ngram")
                        }
                        .pickerStyle(.segmented)
                    }
                }

                if draftSettings.mtpBackend == "variational" {
                    Divider().background(Color.white.opacity(0.08))

                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            SettingsRowLabel(text: "Relaxed Margin (ε):", key: "mtpRelaxedMargin")
                            Spacer()
                            Text(draftSettings.mtpRelaxedMargin == 0
                                 ? "0.00 (Strict Argmax)"
                                 : String(format: "%.2f nats", draftSettings.mtpRelaxedMargin))
                                .font(.system(size: 11, weight: .semibold, design: .monospaced))
                                .foregroundColor(draftSettings.mtpRelaxedMargin == 0 ? TelemetryTheme.warning : TelemetryTheme.accent)
                        }
                        Slider(value: $draftSettings.mtpRelaxedMargin, in: 0.0...5.0, step: 0.1)
                        Text("Accepts drafted tokens whose verifier logit is within ε of the target argmax (arXiv:2610.26693)")
                            .font(.system(size: 9))
                            .foregroundColor(TelemetryTheme.quiet(increasedContrast: contrast == .increased))
                    }

                    VariationalFloorRow(value: $draftSettings.mtpRelaxedMinProb, contrast: contrast)

                    HStack(spacing: 6) {
                        Toggle("Quadratic Prefix Seeding (Batch Verify Seed)", isOn: $draftSettings.mtpQuadraticSeeding)
                            .toggleStyle(.switch)
                        SettingsRestartBadge(key: "mtpQuadraticSeeding")
                    }
                }

                Divider().background(Color.white.opacity(0.08))

                HStack(spacing: 6) {
                    Toggle("Adaptive Draft Depth (Entropy Stop)", isOn: $draftSettings.mtpAdaptiveDepth)
                        .toggleStyle(.switch)
                    SettingsRestartBadge(key: "mtpAdaptiveDepth")
                }

                if draftSettings.mtpAdaptiveDepth {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            SettingsRowLabel(text: "Entropy Threshold:", key: "mtpEntropyThreshold")
                            Spacer()
                            Text(String(format: "%.2f", draftSettings.mtpEntropyThreshold))
                                .font(.system(size: 11, weight: .semibold, design: .monospaced))
                                .foregroundColor(TelemetryTheme.accent)
                        }
                        Slider(value: $draftSettings.mtpEntropyThreshold, in: 0.05...1.0, step: 0.05)
                    }
                }

                if draftSettings.mtpBackend == "dflash2" {
                    Divider().background(Color.white.opacity(0.08))

                    formRow(label: "Drafter Model:", key: "dflashDraftPackPath") {
                        HStack(spacing: 8) {
                            Picker("", selection: $draftSettings.dflashDraftPackPath) {
                                ForEach(discoveredDrafterModels) { m in
                                    Text(m.name).tag(m.packPath)
                                }
                            }
                            .pickerStyle(.menu)
                            .onChange(of: draftSettings.dflashDraftPackPath) { _, _ in
                                validateModelAndDrafterOnSelection()
                            }

                            Button(action: refreshDiscoveredModels) {
                                Image(systemName: "arrow.clockwise")
                                    .font(.system(size: 11))
                            }
                            .buttonStyle(.plain)
                            .help("Refresh drafter models in ~/.qwg/models")
                        }
                    }

                    VStack(alignment: .leading, spacing: 4) {
                        SettingsRowLabel(text: "DFlash 2 Drafter Pack Path:", key: "dflashDraftPackPath")
                        HStack(spacing: 8) {
                            TextField("~/.qwg/models/Qwen3.8-27B-DFlash2-hq4g32", text: $draftSettings.dflashDraftPackPath)
                                .textFieldStyle(.roundedBorder)
                            Button("Browse...") {
                                selectDrafterPackFile()
                            }
                        }
                    }

                    formRow(label: "Selector Top-K:", key: "dflashSelectorTopK") {
                        Picker("", selection: $draftSettings.dflashSelectorTopK) {
                            Text("1 (Argmax)").tag(1)
                            Text("4").tag(4)
                            Text("8").tag(8)
                            Text("16 (Default)").tag(16)
                        }
                        .pickerStyle(.segmented)
                    }
                }
            }
        }
    }

    // MARK: - 6. KV Cache & Storage Section
    private var cacheSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            settingsGroup(title: "Codebase & Knowledge Memory Pack (.qwg)", icon: "memorychip") {
                if let mounted = mountedMemoryPack {
                    HStack(spacing: 8) {
                        Image(systemName: "lock.shield.fill")
                            .font(.system(size: 12))
                            .foregroundColor(TelemetryTheme.active)
                        VStack(alignment: .leading, spacing: 2) {
                            Text("\(mounted.packName) — \(mounted.fileCount) files · \(mounted.tokenCount) tokens · KV \(mounted.enforcedKVMode.uppercased()) (enforced)")
                                .font(.system(size: 11, weight: .semibold))
                                .foregroundColor(.white)
                            Text("\(mounted.rootPath) · LSA \(mounted.lsaMode) · Live delta: \(mounted.modifiedFileCount) file(s)")
                                .font(.system(size: 9, design: .monospaced))
                                .foregroundColor(TelemetryTheme.quiet(increasedContrast: contrast == .increased))
                        }
                        Spacer()
                        Button("Mount...") {
                            selectAndMountMemoryPackDirectory()
                        }
                        .controlSize(.small)
                        Button("Unmount") {
                            unmountActiveMemoryPack()
                        }
                        .controlSize(.small)
                    }
                } else {
                    HStack(spacing: 8) {
                        Text("Auto-discovers <repo>/.qwg/manifest.json from coding agent system prompts, or mount manually:")
                            .font(.system(size: 10))
                            .foregroundColor(TelemetryTheme.quiet(increasedContrast: contrast == .increased))
                        Spacer()
                        Button("Mount Repository...") {
                            selectAndMountMemoryPackDirectory()
                        }
                        .controlSize(.small)
                    }
                }
            }

            settingsGroup(title: "Two-Tier Hierarchical Prefix Cache", icon: "internaldrive") {
                formRow(label: "Warm RAM Limit (GB):", key: "warmRAMLimitGB") {
                    TextField("4.0", value: $draftSettings.warmRAMLimitGB, formatter: plainNumberFormatter)
                        .textFieldStyle(.roundedBorder)
                }

                intSliderRow(label: "Concurrent Session Slots:", key: "maxConcurrentSessions",
                             range: 1...16, step: 1,
                             readout: { "\($0) slots" },
                             value: $draftSettings.maxConcurrentSessions)
                Text("Independent session slots preserved simultaneously in Warm RAM (e.g. main agent + subagents sharing a system prompt).")
                    .font(.system(size: 9))
                    .foregroundColor(TelemetryTheme.quiet(increasedContrast: contrast == .increased))

                formRow(label: "Persist Cache to SSD:", key: "ssdCacheEnabled") {
                    Toggle("Enable Cold SSD Persistence", isOn: $draftSettings.ssdCacheEnabled)
                        .toggleStyle(.switch)
                }

                formRow(label: "Cold SSD Limit (GB):", key: "ssdCacheLimitGB") {
                    TextField("10.0", value: $draftSettings.ssdCacheLimitGB, formatter: plainNumberFormatter)
                        .textFieldStyle(.roundedBorder)
                        .disabled(!draftSettings.ssdCacheEnabled)
                }

                VStack(alignment: .leading, spacing: 4) {
                    SettingsRowLabel(text: "Cache Directory:", key: "cacheDir")
                    HStack(spacing: 8) {
                        TextField("~/.qwg/cache", text: $draftSettings.cacheDir)
                            .textFieldStyle(.roundedBorder)
                            .disabled(!draftSettings.ssdCacheEnabled)
                        Button("Browse...") {
                            selectCacheDirectory()
                        }
                        .disabled(!draftSettings.ssdCacheEnabled)
                    }
                }
            }
        }
    }

    // MARK: - 7. Logging & Diagnostics Section
    private var loggingSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            settingsGroup(title: "Logging Configuration", icon: "doc.text") {
                formRow(label: "Log Level:", key: "logLevel") {
                    Picker("", selection: $draftSettings.logLevel) {
                        Text("Debug").tag("debug")
                        Text("Info").tag("info")
                        Text("Warn").tag("warn")
                        Text("Error").tag("error")
                    }
                    .pickerStyle(.segmented)
                }

                formRow(label: "Log File Name:", key: "logFileName") {
                    TextField("qwg.log", text: $draftSettings.logFileName)
                        .textFieldStyle(.roundedBorder)
                }

                VStack(alignment: .leading, spacing: 4) {
                    SettingsRowLabel(text: "Log Directory:", key: "logDir")
                    HStack(spacing: 8) {
                        TextField("~/.qwg/logs", text: $draftSettings.logDir)
                            .textFieldStyle(.roundedBorder)
                        Button("Reveal in Finder") {
                            revealLogFolderInFinder()
                        }
                    }
                }
            }
        }
    }

    // UI Building Blocks and Actions live in SettingsView+Actions.swift.
}
