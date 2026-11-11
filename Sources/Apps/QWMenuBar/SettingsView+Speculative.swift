// SettingsView+Speculative.swift — The Speculative tab: Standard, Variational,
// DFlash 2, and DSpine block-drafter controls with config.json auto-detection.

import SwiftUI
import QWCore
import QWServer

extension SettingsView {
    // MARK: - 4. Speculative Decoding Section
    var speculativeSection: some View {
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
                        Text("Variational").tag("variational")
                        if selectedModelSupportsDFlash2 {
                            Text("DFlash 2").tag("dflash2")
                            Text("DSpine").tag("dspine")
                        } else {
                            Text("DFlash 2 (Dense)").tag("dflash2")
                            Text("DSpine (Dense)").tag("dspine")
                        }
                    }
                    .pickerStyle(.segmented)
                    .onChange(of: draftSettings.mtpBackend) { _, newBackend in
                        handleMTPBackendSelectionChange(newBackend)
                    }
                }

                intSliderRow(label: "MTP Draft Tokens:", key: "mtpDrafts",
                             range: 0...Double(selectedModelMaxDrafts), step: 1,
                             readout: { isExternalDrafterSelected ? "\($0) drafts (B=\($0 + 1))" : "\($0) drafts" },
                             value: $draftSettings.mtpDrafts)

                if !isExternalDrafterSelected {
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

                if isExternalDrafterSelected {
                    Divider().background(Color.white.opacity(0.08))

                    formRow(label: "Drafter Model:", key: "dflashDraftPackPath") {
                        HStack(spacing: 8) {
                            Picker("", selection: $draftSettings.dflashDraftPackPath) {
                                ForEach(discoveredDrafterModels) { m in
                                    Text(m.name).tag(m.packPath)
                                }
                            }
                            .pickerStyle(.menu)
                            .onChange(of: draftSettings.dflashDraftPackPath) { _, newPath in
                                syncDrafterBackendFromPack(newPath)
                                validateModelAndDrafterOnSelection()
                            }

                            Button(action: refreshDiscoveredModels) {
                                Image(systemName: "arrow.clockwise")
                                    .font(.system(size: 11))
                            }
                            .buttonStyle(.plain)
                            .help("Refresh DFlash 2 / DSpine drafter models in ~/.qwg/models")
                        }
                    }

                    VStack(alignment: .leading, spacing: 4) {
                        SettingsRowLabel(text: "Drafter Pack Path:", key: "dflashDraftPackPath")
                        HStack(spacing: 8) {
                            TextField("~/.qwg/models/<drafter-pack>", text: $draftSettings.dflashDraftPackPath)
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

    private func handleMTPBackendSelectionChange(_ newBackend: String) {
        let lower = newBackend.lowercased()
        let isExt = (lower == "dflash2" || lower == "dspine")
        if !isExt {
            let maxD = selectedModelMaxDrafts
            if draftSettings.mtpDrafts > maxD {
                draftSettings.mtpDrafts = maxD
            }
            validationErrorMessage = nil
            return
        }
        if lower == "dspine",
           let match = discoveredDrafterModels.first(where: {
               QWServerSettings.inferDrafterBackendAndMaxDrafts(packPath: $0.packPath).backend == "dspine"
           }) {
            draftSettings.dflashDraftPackPath = match.packPath
        } else if lower == "dflash2",
                  let match = discoveredDrafterModels.first(where: {
                      QWServerSettings.inferDrafterBackendAndMaxDrafts(packPath: $0.packPath).backend == "dflash2"
                  }) {
            draftSettings.dflashDraftPackPath = match.packPath
        } else if draftSettings.dflashDraftPackPath.isEmpty {
            draftSettings.dflashDraftPackPath = discoveredDrafterModels.first?.packPath ?? ""
        }
        let inferred = QWServerSettings.inferDrafterBackendAndMaxDrafts(packPath: draftSettings.dflashDraftPackPath)
        draftSettings.mtpDrafts = inferred.maxDrafts
        do {
            try draftSettings.validateDFlashConfiguration()
            validationErrorMessage = nil
        } catch {
            validationErrorMessage = "\(newBackend.uppercased()) fail-fast: \(error)"
            draftSettings.mtpBackend = "variational"
            let fallbackMax = selectedModelMaxDrafts
            if draftSettings.mtpDrafts > fallbackMax { draftSettings.mtpDrafts = min(3, fallbackMax) }
        }
    }

    private func syncDrafterBackendFromPack(_ path: String) {
        guard !path.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        let inferred = QWServerSettings.inferDrafterBackendAndMaxDrafts(packPath: path)
        if isExternalDrafterSelected {
            if draftSettings.mtpBackend != inferred.backend {
                draftSettings.mtpBackend = inferred.backend
            }
            draftSettings.mtpDrafts = inferred.maxDrafts
        }
    }
}
