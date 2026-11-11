// SettingsView+Actions.swift — the SettingsView actions.
import AppKit
import ServiceManagement
import SwiftUI
import QWCore
import QWQuant
import QWServer

extension SettingsView {

    // MARK: - Actions

    func refreshDiscoveredModels() {
        let scanned = QWServerSettings.scanModelsDirectory()
        discoveredModels = scanned
        if draftSettings.pruneRemovedModels(discovered: scanned) {
            try? QWSettingsManager.shared.saveSettings(draftSettings)
            telemetry.settings = draftSettings
        }
        // The configured pack is the model the engine runs. The name the
        // server serves it under can be an id of the user's choosing, not
        // a directory name, so a name that matches no installed pack does
        // not mean the model is gone.
        if !discoveredModels.contains(where: { $0.name == draftSettings.modelName }) {
            let expandedPack = draftSettings.modelPackPath.expandingTildePath
            let inModelsFolder = expandedPack.isEmpty || expandedPack.hasPrefix(QWPaths.modelsDirectoryPath)
            if let configured = discoveredModels.first(where: { $0.packPath == expandedPack }) {
                // The pack is installed: keep the id the server serves,
                // and let the picker name it.
                draftSettings.modelAlias = draftSettings.alias(for: draftSettings.modelName)
                discoveredModels.insert(DiscoveredModel(name: draftSettings.modelName, packPath: configured.packPath), at: 0)
            } else if inModelsFolder, let first = discoveredModels.first(where: \.supported) {
                draftSettings.modelName = first.name
                draftSettings.modelPackPath = first.packPath
                draftSettings.modelAlias = draftSettings.alias(for: first.name)
            } else if !draftSettings.modelName.isEmpty {
                discoveredModels.insert(DiscoveredModel(name: draftSettings.modelName, packPath: draftSettings.modelPackPath), at: 0)
            }
        } else {
            draftSettings.modelAlias = draftSettings.alias(for: draftSettings.modelName)
        }
        discoveredDrafterModels = QWServerSettings.scanDrafterModelsDirectory()
        if !draftSettings.dflashDraftPackPath.isEmpty && !discoveredDrafterModels.contains(where: { $0.packPath == draftSettings.dflashDraftPackPath }) {
            let name = QWModelRoot.canonicalModelName(for: URL(fileURLWithPath: draftSettings.dflashDraftPackPath))
            discoveredDrafterModels.insert(DiscoveredModel(name: name, packPath: draftSettings.dflashDraftPackPath), at: 0)
        } else if draftSettings.dflashDraftPackPath.isEmpty, let first = discoveredDrafterModels.first {
            draftSettings.dflashDraftPackPath = first.packPath
        }
    }

    func selectDrafterPackFile() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url {
            draftSettings.dflashDraftPackPath = url.path
            refreshDiscoveredModels()
            validateModelAndDrafterOnSelection()
        }
    }

    func revealModelsFolderInFinder() {
        let url = QWPaths.modelsDirectoryURL
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: url.path)
    }

    func selectModelPackFile() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url {
            let baseName = QWModelRoot.canonicalModelName(for: url)
            let entry = QWModelCatalog.classify(name: baseName, path: url.path)
            guard entry.supported else {
                validationErrorMessage = "Cannot select '\(baseName)': \(entry.reason ?? "not runnable")"
                return
            }
            draftSettings.modelPackPath = url.path
            if !baseName.isEmpty {
                draftSettings.modelName = baseName
            }
            refreshDiscoveredModels()
            validateModelAndDrafterOnSelection()
        }
    }

    var selectedModelSupportsVision: Bool {
        if let disc = discoveredModels.first(where: { $0.name == draftSettings.modelName }),
           !disc.capabilities.isEmpty {
            return disc.capabilities.contains("vision")
        }
        if let desc = try? draftSettings.validatePrimaryModel() {
            return desc.visDepth > 0 && desc.capabilities.contains(.vision)
        }
        return true
    }

    var selectedModelSupportsSpeculative: Bool {
        if let disc = discoveredModels.first(where: { $0.name == draftSettings.modelName }) {
            if !disc.capabilities.isEmpty {
                return disc.capabilities.contains("mtp") || disc.maxSpeculativeDrafts > 0
            }
        }
        if let desc = try? draftSettings.validatePrimaryModel() {
            return desc.supportsNativeMTP || (desc.hcCount == nil && desc.numExperts == nil)
        }
        return true
    }

    var selectedModelSupportsDFlash2: Bool {
        if let desc = try? draftSettings.validatePrimaryModel() {
            return desc.hcCount == nil && desc.numExperts == nil
        }
        if let disc = discoveredModels.first(where: { $0.name == draftSettings.modelName }) {
            return !disc.capabilities.contains("moe") && !disc.capabilities.contains("hyperConnection")
        }
        return true
    }

    var isExternalDrafterSelected: Bool {
        let lower = draftSettings.mtpBackend.lowercased()
        return lower == "dflash2" || lower == "dspine"
    }

    var selectedModelMaxDrafts: Int {
        if isExternalDrafterSelected {
            let inferred = QWServerSettings.inferDrafterBackendAndMaxDrafts(packPath: draftSettings.dflashDraftPackPath)
            return draftSettings.mtpBackend.lowercased() == "dspine" ? max(15, inferred.maxDrafts) : inferred.maxDrafts
        }
        if let desc = try? draftSettings.validatePrimaryModel(), desc.maxSpeculativeDrafts > 0 {
            return desc.maxSpeculativeDrafts
        }
        if let disc = discoveredModels.first(where: { $0.name == draftSettings.modelName }), disc.maxSpeculativeDrafts > 0 {
            return disc.maxSpeculativeDrafts
        }
        return 5
    }

    var selectedModelHasLSASidecar: Bool {
        let rawPath = draftSettings.modelPackPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let ref = draftSettings.resolveModelReference(rawPath.isEmpty ? draftSettings.modelName : rawPath, discovered: discoveredModels)
        let explicit = draftSettings.lsaSidecarPath.trimmingCharacters(in: .whitespacesAndNewlines)
        return QWLSAConfig.resolveSidecarURL(packPath: ref.packPath, explicitSidecarPath: explicit.isEmpty ? nil : explicit) != nil
    }

    var selectedModelLSASidecarStatusText: String {
        let rawPath = draftSettings.modelPackPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let ref = draftSettings.resolveModelReference(rawPath.isEmpty ? draftSettings.modelName : rawPath, discovered: discoveredModels)
        let explicit = draftSettings.lsaSidecarPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let url = QWLSAConfig.resolveSidecarURL(packPath: ref.packPath, explicitSidecarPath: explicit.isEmpty ? nil : explicit) else {
            return "No .qlsa sidecar detected (Auto uses in-memory Weight-SVD)"
        }
        if let desc = try? draftSettings.validatePrimaryModel(),
           let info = try? QWLSASidecar.peekInfo(url: url, descriptor: desc) {
            let matchNote = info.geometryMatches ? "verified" : "digest mismatch"
            return "Detected \(url.lastPathComponent) [\(info.method), d=\(info.indexDim), \(matchNote)]"
        }
        return "Detected \(url.lastPathComponent)"
    }

    func selectLSASidecarFile() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url {
            draftSettings.lsaSidecarPath = url.path
            validateModelAndDrafterOnSelection()
        }
    }

    func generateLSASidecarForSelectedModel() {
        do {
            let rawPath = draftSettings.modelPackPath.trimmingCharacters(in: .whitespacesAndNewlines)
            let ref = draftSettings.resolveModelReference(rawPath.isEmpty ? draftSettings.modelName : rawPath, discovered: discoveredModels)
            let desc = try draftSettings.validatePrimaryModel()
            let indexDim = QWLSAConfig.defaultIndexDim
            let tensors = try QWLSASidecar.synthesizeFromModel(packPath: ref.packPath, descriptor: desc, indexDim: indexDim)
            let root = QWModelRoot(url: URL(fileURLWithPath: ref.packPath))
            let outURL = root.directory.appendingPathComponent("model.qlsa")
            let prov = QWLSASidecar.Header.Provenance(
                modelId: root.directory.lastPathComponent,
                geometryDigest: desc.geometryDigestV2Hex,
                method: "weight_svd",
                indexDim: indexDim,
                poolSize: QWLSAConfig.poolSize,
                clusterBlockSize: draftSettings.lsaClusterBlock,
                bucketsPerBlock: QWLSAConfig.bucketsPerBlock,
                maxBucketKeys: QWLSAConfig.maxBucketKeys,
                recommendedBudget: draftSettings.lsaBudget
            )
            let shape = [indexDim, desc.lsaInputDim]
            try QWLSASidecar.write(to: outURL, provenance: prov, layerTensors: tensors, wqShape: shape, wkShape: shape)
            refreshDiscoveredModels()
            validationErrorMessage = nil
            saveConfirmationMessage = "Built \(outURL.lastPathComponent) (\(tensors.count) layers)"
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.5) {
                saveConfirmationMessage = nil
            }
        } catch {
            validationErrorMessage = "LSA sidecar build failed: \(error)"
        }
    }

    func resetToTopPerformanceDefaults() {
        let rawPath = draftSettings.modelPackPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let ref = draftSettings.resolveModelReference(rawPath.isEmpty ? draftSettings.modelName : rawPath)
        draftSettings.applyModelConfigDefaults(packPath: ref.packPath)
        QWSettingsManager.shared.removeProfile(forModel: draftSettings.modelName)
        validationErrorMessage = nil
        saveConfirmationMessage = "Reset to model & engine defaults"
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.5) {
            saveConfirmationMessage = nil
        }
    }

    func validateModelAndDrafterOnSelection() {
        do {
            let desc = try draftSettings.validatePrimaryModel()
            if !(desc.visDepth > 0 && desc.capabilities.contains(.vision)) {
                draftSettings.loadVision = false
            }
            let supportsDFlash = desc.hcCount == nil && desc.numExperts == nil
            if !supportsDFlash && isExternalDrafterSelected {
                draftSettings.mtpBackend = "variational"
            }
            let maxD = isExternalDrafterSelected
                ? QWServerSettings.inferDrafterBackendAndMaxDrafts(packPath: draftSettings.dflashDraftPackPath).maxDrafts
                : max(1, desc.maxSpeculativeDrafts)
            if draftSettings.mtpDrafts > maxD {
                draftSettings.mtpDrafts = maxD
            }
            if isExternalDrafterSelected {
                try draftSettings.validateDFlashConfiguration()
            }
            if draftSettings.lsaMode.lowercased() == "sidecar" && !selectedModelHasLSASidecar {
                validationErrorMessage = "LSA mode 'sidecar' requires a model.qlsa file (click 'Build .qlsa (SVD)' or choose 'Auto')"
                return
            }
            validationErrorMessage = nil
        } catch {
            validationErrorMessage = "\(error)"
        }
    }

    func selectCacheDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url {
            draftSettings.cacheDir = url.path
        }
    }

    var mountedMemoryPack: QWEngineInfoResponse.MountedMemoryPackInfo? {
        telemetry.engineInfo?.mountedMemoryPack
    }

    var isKVModeLockedByMemoryPack: Bool {
        mountedMemoryPack?.enforcesKVMode == true
    }

    var selectedModelContextMemoryEstimate: QWContextMemoryEstimate? {
        var effective = draftSettings
        if let enforced = mountedMemoryPack?.enforcedKVMode {
            effective.kvMode = enforced
        }
        return effective.estimateContextMemory()
    }

    func selectAndMountMemoryPackDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.message = "Select a repository or knowledge directory containing .qwg/manifest.json"
        if panel.runModal() == .OK, let url = panel.url {
            telemetry.mountMemoryPack(path: url.path) { result in
                switch result {
                case .success(let info):
                    draftSettings.kvMode = info.enforcedKVMode
                    validationErrorMessage = nil
                    saveConfirmationMessage = "Mounted '\(info.packName)' (KV \(info.enforcedKVMode.uppercased()) locked)"
                    DispatchQueue.main.asyncAfter(deadline: .now() + 2.5) {
                        saveConfirmationMessage = nil
                    }
                case .failure(let error):
                    validationErrorMessage = "Memory pack mount failed: \(error.localizedDescription)"
                }
            }
        }
    }

    func unmountActiveMemoryPack() {
        telemetry.unmountMemoryPack {
            validationErrorMessage = nil
            saveConfirmationMessage = "Unmounted memory pack (KV Cache Mode unlocked)"
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.5) {
                saveConfirmationMessage = nil
            }
        }
    }

    func revealLogFolderInFinder() {
        let url = draftSettings.logDir.canonicalFileURL
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: url.path)
    }

    func revealSettingsInFinder() {
        let url = QWPaths.settingsFileURL
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }

    func updateRunAtStartup(_ enabled: Bool) {
        if #available(macOS 13.0, *) {
            do {
                if enabled {
                    if SMAppService.mainApp.status != .enabled {
                        try SMAppService.mainApp.register()
                    }
                } else {
                    if SMAppService.mainApp.status == .enabled {
                        try SMAppService.mainApp.unregister()
                    }
                }
            } catch {
                print("Failed to update SMAppService runAtStartup: \(error)")
            }
        }
    }

    func applyAndSave() {
        do {
            try draftSettings.validateBeforeLaunch()
            try telemetry.saveSettings(draftSettings)
            validationErrorMessage = nil
            saveConfirmationMessage = "Settings saved successfully!"
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.5) {
                saveConfirmationMessage = nil
            }
        } catch {
            validationErrorMessage = "Validation failed: \(error)"
        }
    }

    // MARK: - Pending Server Restart & UI Building Blocks

    var pendingRestartPanel: some View {
        let pending = telemetry.pendingServerRestartKeys(for: draftSettings)
        return Group {
            if pending.isEmpty {
                EmptyView()
            } else {
                VStack(alignment: .leading, spacing: 6) {
                    Text("\(pending.count) changed setting\(pending.count == 1 ? "" : "s") need a server restart")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundColor(TelemetryTheme.warning)
                    Text(pending.map { QWSettingsCoherence.displayName(for: $0) }.joined(separator: ", "))
                        .font(.system(size: 10))
                        .foregroundColor(.white.opacity(0.70))
                    HStack(spacing: 8) {
                        Button("Restart Server") { telemetry.restartServerProcess() }
                            .buttonStyle(.borderedProminent)
                            .controlSize(.small)
                            .disabled(validationErrorMessage != nil)
                        Text("Stops the server process and starts it again. The engine restart does not.")
                            .font(.system(size: 9))
                            .foregroundColor(TelemetryTheme.quiet(increasedContrast: contrast == .increased))
                    }
                }
                .padding(10)
                .background(Color.white.opacity(0.05))
                .cornerRadius(8)
            }
        }
    }

    func formRow<Content: View>(label: String, key: String = "",
                                @ViewBuilder content: () -> Content) -> some View {
        HStack(alignment: .center, spacing: 12) {
            SettingsRowLabel(text: label, key: key)
                .frame(width: 130, alignment: .leading)
            content()
        }
    }

    func intSliderRow(label: String, key: String, range: ClosedRange<Double>, step: Double,
                      readout: @escaping (Int) -> String = { "\($0)" },
                      tint: @escaping (Int) -> Color = { _ in TelemetryTheme.accent },
                      value: Binding<Int>) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                SettingsRowLabel(text: label, key: key)
                Spacer()
                Text(readout(value.wrappedValue))
                    .font(.system(size: 11, weight: .semibold, design: .monospaced))
                    .foregroundColor(tint(value.wrappedValue))
            }
            Slider(value: Binding(get: { Double(value.wrappedValue) },
                                  set: { value.wrappedValue = Int($0) }),
                   in: range, step: step)
        }
    }

    func settingsGroup<Content: View>(title: String, icon: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 6) {
                Image(systemName: icon)
                    .font(.system(size: 11))
                    .foregroundColor(TelemetryTheme.accent)
                Text(title)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(TelemetryTheme.primary(increasedContrast: contrast == .increased))
            }
            VStack(alignment: .leading, spacing: 10) {
                content()
            }
            .padding(12)
            .background(Color.white.opacity(0.04))
            .cornerRadius(8)
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.white.opacity(0.08), lineWidth: 0.5))
        }
    }

    var plainNumberFormatter: NumberFormatter {
        let f = NumberFormatter()
        f.numberStyle = .none
        f.usesGroupingSeparator = false
        return f
    }
}

extension TelemetryObservable {
    public func mountMemoryPack(
        path: String,
        onComplete: (@MainActor (Result<QWEngineInfoResponse.MountedMemoryPackInfo, Error>) -> Void)? = nil
    ) {
        guard let url = URL(string: "http://\(settings.host):\(settings.port)/v1/memory/mount") else { return }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try? JSONSerialization.data(withJSONObject: ["path": path])
        req.timeoutInterval = 120.0
        Task {
            do {
                let (data, res) = try await URLSession.shared.data(for: req)
                guard let http = res as? HTTPURLResponse, http.statusCode == 200 else {
                    let body = String(data: data, encoding: .utf8) ?? "mount failed"
                    throw NSError(domain: "QWMemoryPack", code: 1, userInfo: [NSLocalizedDescriptionKey: body])
                }
                let info = try JSONDecoder().decode(QWEngineInfoResponse.MountedMemoryPackInfo.self, from: data)
                self.settings.kvMode = info.enforcedKVMode
                self.fetchHttpEngineInfo()
                self.fetchDashboard()
                onComplete?(.success(info))
            } catch {
                onComplete?(.failure(error))
            }
        }
    }

    public func unmountMemoryPack(onComplete: (@MainActor () -> Void)? = nil) {
        guard let url = URL(string: "http://\(settings.host):\(settings.port)/v1/memory/unmount") else { return }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.timeoutInterval = 30.0
        Task {
            _ = try? await URLSession.shared.data(for: req)
            self.fetchHttpEngineInfo()
            self.fetchDashboard()
            onComplete?()
        }
    }
}
