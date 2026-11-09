// SettingsView+Performance.swift — The Performance tab: memory policy,
// expert residency and streaming, prefill shape, and sparse attention.
// Every value is stored per model profile; an absent value is the default.

import SwiftUI
import QWCore
import QWServer

extension SettingsView {
    // MARK: - 5. Performance & Memory Section
    var performanceSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            settingsGroup(title: "Memory & Residency", icon: "memorychip") {
                HStack(spacing: 6) {
                    Text("Stored per model · QW_* environment variables override")
                        .font(.system(size: 9))
                        .foregroundColor(TelemetryTheme.quiet(increasedContrast: contrast == .increased))
                    Spacer()
                    Button("Reset Performance Defaults") { resetPerformanceDefaults() }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        .help("Reinstate the measured-best performance settings for the selected model")
                }

                formRow(label: "Memory Policy:", key: "engineTuning") {
                    Picker("", selection: tuningString("memoryPolicy")) {
                        Text("Max Resident (Fastest)").tag(QWMemoryPolicy.maxResident.rawValue)
                        Text("Safe (No Swap)").tag(QWMemoryPolicy.safe.rawValue)
                    }
                    .pickerStyle(.segmented)
                }
                Text(tuningString("memoryPolicy").wrappedValue == QWMemoryPolicy.maxResident.rawValue
                     ? "Sizes resident weights to physical RAM less the OS reserve. Fastest; other apps' memory may be swapped out."
                     : "Sizes resident weights to memory that is free or reclaimable at load, less 6 GiB headroom. Never swaps.")
                    .font(.system(size: 9))
                    .foregroundColor(TelemetryTheme.quiet(increasedContrast: contrast == .increased))

                doubleSliderRow(label: "OS Reserve:", id: "osReserveGiB", range: 2...32, step: 1,
                                readout: { String(format: "%.0f GiB", $0) })

                formRow(label: "Scratch Eviction:", key: "evictPrefillScratch") {
                    Picker("", selection: $draftSettings.evictPrefillScratch) {
                        Text("Auto").tag("auto")
                        Text("Always").tag("always")
                        Text("Never").tag("never")
                    }
                    .pickerStyle(.segmented)
                }
            }

            if selectedModelStreamsExperts {
                settingsGroup(title: "MoE Expert Streaming", icon: "square.stack.3d.down.right") {
                    intSliderRow(label: "Resident Experts / Layer:", key: "engineTuning",
                                 range: 0...Double(selectedModelExpertCount), step: 1,
                                 readout: { $0 == 0 ? "Auto (memory policy)" : "\($0) of \(selectedModelExpertCount)" },
                                 value: tuningInt("residentExpertsPerLayer"))
                    tuningToggle("Cross-Token Expert Cache", id: "expertCache")
                    tuningToggle("Event-Pipelined Cold Expert Staging (Decode)", id: "coldEvents")
                    tuningToggle("Overlap Resident Experts with Cold Staging", id: "coldEventOverlap")
                        .disabled(!tuningBool("coldEvents").wrappedValue)
                    tuningToggle("Mapped Expert Spans per Prefill Chunk", id: "prefillSpanMap")
                    Text("Experts that do not fit in memory stream from SSD. These keep the GPU busy while they load.")
                        .font(.system(size: 9))
                        .foregroundColor(TelemetryTheme.quiet(increasedContrast: contrast == .increased))
                }
            }

            settingsGroup(title: "Prefill & Concurrency", icon: "cpu") {
                // Bounds are the resolver's clamp bounds (QWPrefillLimits).
                intSliderRow(label: "Prefill Chunk Rows:", key: "prefillChunkSize",
                             range: Double(QWPrefillLimits.minRows)...Double(QWPrefillLimits.maxRows),
                             step: Double(QWPrefillLimits.rowStep), readout: { "\($0) rows" },
                             value: $draftSettings.prefillChunkSize)

                intSliderRow(label: "Concurrent Sessions:", key: "maxConcurrentSessions",
                             range: 1...16, step: 1,
                             readout: { $0 == 1 ? "1 session (Single-stream)" : "\($0) sessions" },
                             value: $draftSettings.maxConcurrentSessions)
                Text("Protects up to \(draftSettings.maxConcurrentSessions) concurrent agent/subagent KV cache slots in Warm RAM without mutual eviction, and enables parallel GPU batching when requests fit within per-slot context.")
                    .font(.system(size: 9))
                    .foregroundColor(TelemetryTheme.quiet(increasedContrast: contrast == .increased))
            }

            settingsGroup(title: "Latent Sparse Attention (LSA — arXiv:2610.42147)", icon: "point.3.connected.trianglepath.dotted") {
                formRow(label: "LSA Mode:", key: "lsaMode") {
                    Picker("", selection: $draftSettings.lsaMode) {
                        Text("Off").tag("off")
                        Text("Auto").tag("auto")
                        Text("Sidecar (.qlsa)").tag("sidecar")
                        Text("Weight-SVD").tag("svd")
                        Text("QSA-Cluster").tag("qsa-cluster")
                    }
                    .pickerStyle(.segmented)
                    .onChange(of: draftSettings.lsaMode) { _, _ in
                        validateModelAndDrafterOnSelection()
                    }
                }

                HStack(spacing: 8) {
                    Circle()
                        .fill(selectedModelHasLSASidecar ? TelemetryTheme.active : Color.white.opacity(0.30))
                        .frame(width: 7, height: 7)
                    Text(selectedModelLSASidecarStatusText)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundColor(selectedModelHasLSASidecar ? TelemetryTheme.active : TelemetryTheme.quiet(increasedContrast: contrast == .increased))
                    Spacer()
                    Button("Build .qlsa (SVD)") {
                        generateLSASidecarForSelectedModel()
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.mini)
                    .help("Synthesize a model.qlsa projection sidecar for the selected model using Weight-SVD")
                }

                if draftSettings.lsaMode != "off" {
                    Divider().background(Color.white.opacity(0.08))

                    formRow(label: "Key Budget (k):", key: "lsaBudget") {
                        Picker("", selection: $draftSettings.lsaBudget) {
                            Text("512 tokens").tag(512)
                            Text("1,024 tokens").tag(1024)
                            Text("2,048 tokens (Recommended)").tag(2048)
                            Text("4,096 tokens").tag(4096)
                            Text("8,192 tokens").tag(8192)
                        }
                        .pickerStyle(.menu)
                    }

                    formRow(label: "Cluster Block (L):", key: "lsaClusterBlock") {
                        Picker("", selection: $draftSettings.lsaClusterBlock) {
                            Text("256 tokens").tag(256)
                            Text("512 tokens (Default)").tag(512)
                            Text("1,024 tokens").tag(1024)
                        }
                        .pickerStyle(.segmented)
                    }

                    VStack(alignment: .leading, spacing: 4) {
                        SettingsRowLabel(text: "Custom .qlsa Sidecar Path (Optional):", key: "lsaSidecarPath")
                        HStack(spacing: 8) {
                            TextField("Auto-detected from <model_dir>/model.qlsa", text: $draftSettings.lsaSidecarPath)
                                .textFieldStyle(.roundedBorder)
                            Button("Browse...") {
                                selectLSASidecarFile()
                            }
                        }
                    }

                    Text("Subquadratic dynamic Ward clustering (M=64 buckets/512-tok block, max 16 keys/bucket) with 4 sink + 512 sliding-window tokens. Contexts <= k+516 automatically run exact dense attention.")
                        .font(.system(size: 9))
                        .foregroundColor(TelemetryTheme.quiet(increasedContrast: contrast == .increased))
                }
            }
        }
    }

    /// The selected model has routed experts the engine can stream (MoE outside FlashNext's own engine).
    var selectedModelStreamsExperts: Bool {
        if let desc = try? draftSettings.validatePrimaryModel() {
            return (desc.numExperts ?? 0) > 0 && desc.family != .qwen38FlashNext
        }
        if let disc = discoveredModels.first(where: { $0.name == draftSettings.modelName }) {
            return disc.capabilities.contains("moe") && disc.family != QWModelDescriptor.Family.qwen38FlashNext.rawValue
        }
        return false
    }

    var selectedModelExpertCount: Int {
        max(1, (try? draftSettings.validatePrimaryModel())?.numExperts ?? 256)
    }

    func resetPerformanceDefaults() {
        let rawPath = draftSettings.modelPackPath.trimmingCharacters(in: .whitespacesAndNewlines)
        var baseline = draftSettings
        baseline.applyModelConfigDefaults(packPath: draftSettings.resolveModelReference(
            rawPath.isEmpty ? draftSettings.modelName : rawPath).packPath)
        draftSettings.engineTuning = baseline.engineTuning
        draftSettings.evictPrefillScratch = baseline.evictPrefillScratch
        draftSettings.prefillChunkSize = baseline.prefillChunkSize
        draftSettings.maxConcurrentSessions = baseline.maxConcurrentSessions
        draftSettings.gdnPrefill = baseline.gdnPrefill
    }

    private func tuningValue(_ id: String) -> QWKnobValue? {
        draftSettings.engineTuning[id] ?? QWKnobRegistry.descriptor(id)?.defaultValue
    }

    func tuningBool(_ id: String) -> Binding<Bool> {
        Binding(get: { tuningValue(id)?.boolValue ?? true },
                set: { draftSettings.setEngineTuning(id, .bool($0)) })
    }

    func tuningInt(_ id: String) -> Binding<Int> {
        Binding(get: { tuningValue(id)?.intValue ?? 0 },
                set: { draftSettings.setEngineTuning(id, .int($0)) })
    }

    func tuningString(_ id: String) -> Binding<String> {
        Binding(get: { tuningValue(id)?.stringValue ?? "" },
                set: { draftSettings.setEngineTuning(id, .string($0)) })
    }

    private func tuningToggle(_ title: String, id: String) -> some View {
        HStack(spacing: 6) {
            Toggle(title, isOn: tuningBool(id)).toggleStyle(.switch)
            SettingsRestartBadge(key: "engineTuning")
        }
    }

    private func doubleSliderRow(label: String, id: String, range: ClosedRange<Double>, step: Double,
                                 readout: @escaping (Double) -> String) -> some View {
        let value = Binding<Double>(get: { tuningValue(id)?.doubleValue ?? range.lowerBound },
                                    set: { draftSettings.setEngineTuning(id, .double($0)) })
        return VStack(alignment: .leading, spacing: 6) {
            HStack {
                SettingsRowLabel(text: label, key: "engineTuning")
                Spacer()
                Text(readout(value.wrappedValue))
                    .font(.system(size: 11, weight: .semibold, design: .monospaced))
                    .foregroundColor(TelemetryTheme.accent)
            }
            Slider(value: value, in: range, step: step)
        }
    }
}
