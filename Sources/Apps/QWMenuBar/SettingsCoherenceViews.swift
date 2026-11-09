// SettingsCoherenceViews.swift — The marks that say when a setting takes
// effect. A setting the server reads only when it starts is marked, so a
// change that waits for a restart is never silent.

import SwiftUI
import QWCore
import QWServer

/// The marker on a setting the server reads when it starts.
struct SettingsRestartBadge: View {
    let key: String

    var body: some View {
        if QWSettingsCoherence.needsServerRestart(key) {
            Image(systemName: "arrow.clockwise.circle")
                .font(.system(size: 9))
                .foregroundColor(TelemetryTheme.warning)
                .help("The server reads this setting when it starts. A change to it takes effect after the server restarts.")
        }
    }
}

/// A settings label with the marker that says when the value takes
/// effect.
struct SettingsRowLabel: View {
    let text: String
    let key: String

    var body: some View {
        HStack(spacing: 4) {
            Text(text)
                .font(.system(size: 11, weight: .medium))
                .foregroundColor(.white.opacity(0.70))
            SettingsRestartBadge(key: key)
        }
    }
}

/// The Variational MTP fidelity floor (τ): a relaxed draft must also hold `p >= τ`.
struct VariationalFloorRow: View {
    @Binding var value: Double
    let contrast: ColorSchemeContrast

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                SettingsRowLabel(text: "Fidelity Floor (τ):", key: "mtpRelaxedMinProb")
                Spacer()
                Text(value == 0 ? "0.00 (Off)" : String(format: "p ≥ %.2f", value))
                    .font(.system(size: 11, weight: .semibold, design: .monospaced))
                    .foregroundColor(value == 0 ? TelemetryTheme.warning : TelemetryTheme.accent)
            }
            Slider(value: $value, in: 0.0...0.5, step: 0.05)
            Text("A relaxed draft must also hold at least τ of the verifier's probability mass")
                .font(.system(size: 9))
                .foregroundColor(TelemetryTheme.quiet(increasedContrast: contrast == .increased))
        }
    }
}

/// Power-of-2 Context Limit slider (64 to 1,048,576 tokens) with live model memory prediction and over-budget alert.
struct ContextLimitSliderRow: View {
    @Binding var maxContext: Int
    let estimate: QWContextMemoryEstimate?
    let modelName: String
    let kvMode: String
    let contrast: ColorSchemeContrast
    @State private var isDraggingSlider = false
    @State private var showOverBudgetAlert = false

    private var isOverBudget: Bool {
        guard let estimate else { return false }
        return !estimate.fits
    }

    private var maxFittingStep: Int {
        guard let estimate else { return 64 }
        let fitting = QWSettingsCoherence.contextSliderSteps.filter { $0 <= estimate.maxFittingContext }
        return fitting.last ?? 64
    }

    private var stepBinding: Binding<Double> {
        Binding(
            get: { Double(QWSettingsCoherence.nearestContextStepIndex(for: maxContext)) },
            set: { newIndex in
                maxContext = QWSettingsCoherence.contextTokens(forStepIndex: Int(newIndex.rounded()))
            }
        )
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                SettingsRowLabel(text: "Context Limit:", key: "maxContext")
                Spacer()
                Text(QWSettingsCoherence.formatContextTokens(maxContext))
                    .font(.system(size: 11, weight: .semibold, design: .monospaced))
                    .foregroundColor(isOverBudget ? TelemetryTheme.failure : TelemetryTheme.accent)
            }
            Slider(
                value: stepBinding,
                in: 0...Double(QWSettingsCoherence.contextSliderSteps.count - 1),
                step: 1,
                onEditingChanged: { editing in
                    isDraggingSlider = editing
                    if !editing && isOverBudget {
                        showOverBudgetAlert = true
                    }
                }
            )
            HStack {
                Text("64")
                Spacer()
                Text("1k")
                Spacer()
                Text("16k")
                Spacer()
                Text("256k")
                Spacer()
                Text("1M")
            }
            .font(.system(size: 9, design: .monospaced))
            .foregroundColor(TelemetryTheme.quiet(increasedContrast: contrast == .increased))

            if let est = estimate {
                let gib = 1_073_741_824.0
                let reqGiB = Double(est.requiredBytes) / gib
                let wGiB = Double(est.weightsBytes) / gib
                let kvGiB = Double(est.kvBytes) / gib
                let scrGiB = Double(est.scratchAndStateBytes) / gib
                let availGiB = Double(est.availableBytes) / gib
                Text(String(
                    format: "Predicted Memory: %.1f GiB (Weights %.1f GiB · KV %.1f GiB · Scratch %.1f GiB) / %.1f GiB budget",
                    reqGiB, wGiB, kvGiB, scrGiB, availGiB
                ))
                .font(.system(size: 9, design: .monospaced))
                .foregroundColor(est.fits
                    ? TelemetryTheme.quiet(increasedContrast: contrast == .increased)
                    : TelemetryTheme.failure)

                if !est.fits {
                    let overGiB = Double(est.requiredBytes - est.availableBytes) / gib
                    HStack(spacing: 5) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .font(.system(size: 9))
                            .foregroundColor(TelemetryTheme.failure)
                        Text(String(
                            format: "Exceeds available memory budget by %.1f GiB (max fitting context: %@).",
                            overGiB, QWSettingsCoherence.formatContextTokens(maxFittingStep)
                        ))
                        .font(.system(size: 9, weight: .medium))
                        .foregroundColor(TelemetryTheme.failure)
                    }
                }
            }
        }
        .onChange(of: maxContext) { _, _ in
            if !isDraggingSlider && isOverBudget {
                showOverBudgetAlert = true
            }
        }
        .onChange(of: kvMode) { _, _ in
            if isOverBudget {
                showOverBudgetAlert = true
            }
        }
        .alert("Context Limit Exceeds Available Memory", isPresented: $showOverBudgetAlert) {
            Button("Clamp to \(QWSettingsCoherence.formatContextTokens(maxFittingStep))") {
                maxContext = maxFittingStep
            }
            Button("Keep Anyway", role: .cancel) {}
        } message: {
            if let est = estimate {
                let gib = 1_073_741_824.0
                Text(String(
                    format: "%@ at %@ KV requires %.1f GiB (%.1f GiB weights + %.1f GiB KV + %.1f GiB scratch), which exceeds the %.1f GiB available memory budget for %@. Reduce Context Limit to %@ or switch KV Cache Mode to a lower bit-width.",
                    QWSettingsCoherence.formatContextTokens(maxContext),
                    kvMode.uppercased(),
                    Double(est.requiredBytes) / gib,
                    Double(est.weightsBytes) / gib,
                    Double(est.kvBytes) / gib,
                    Double(est.scratchAndStateBytes) / gib,
                    Double(est.availableBytes) / gib,
                    modelName.isEmpty ? "the selected model" : modelName,
                    QWSettingsCoherence.formatContextTokens(maxFittingStep)
                ))
            }
        }
    }
}
