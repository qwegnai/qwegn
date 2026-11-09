// QWCli.swift — Pure CLI library logic for pack inspection, admission planning, and I/O.
// Implements `inspectReport`, `admissionPlanReport`, `dequantToNpy`,
// `quantizePlan`, `downloadPlan`, NumPy `.npy` serialization, and terminal
// ANSI color styling for the `qwg` command-line executable.

import Foundation
import QWCore
import QWKV
import QWQuant

/// Pure CLI logic library backing the `qwg` executable.
public enum QWCli {

    // MARK: - CLI Options & Validation

    public enum CLIError: Error, CustomStringConvertible, Equatable {
        case invalidKVMode(String)
        case unpriceableKVMode(String)
        case invalidContextBudget(Int, max: Int)
        case inputFileNotFound(String)

        public var description: String {
            switch self {
            case .invalidKVMode(let mode):
                return "Invalid KV cache mode '\(mode)' (supported: f16, q8, q4)"
            case .unpriceableKVMode(let mode):
                return "Invalid KV cache mode '\(mode)' for the admission plan (it prices f16 and q8)"
            case .invalidContextBudget(let tokens, let max):
                return "Invalid context budget \(tokens) tokens (must be positive and <= \(max))"
            case .inputFileNotFound(let path):
                return "Multimodal input file not found: \(path)"
            }
        }
    }

    /// Resolves string representation of KV cache mode ("f16", "q8", "q4") to `QWKVMode`.
    /// There is no "auto": a knob value is either one the registry declares or
    /// it is not a value at all.
    public static func resolveKVMode(_ string: String) throws -> QWKVMode {
        switch string.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "f16":
            return .f16
        case "q8":
            return .q8
        case "q4":
            return .q4
        default:
            throw CLIError.invalidKVMode(string)
        }
    }

    /// Validates context budget within valid positive range up to 1M tokens.
    public static func validateContextBudget(tokens: Int, maxTokens: Int = QWKVCache.maxTokensPerSeq) throws {
        guard tokens > 0 && tokens <= maxTokens else {
            throw CLIError.invalidContextBudget(tokens, max: maxTokens)
        }
    }

    /// Multimodal command options for image and video inputs.
    public struct MultimodalInputOptions: Equatable, Sendable {
        public var imagePath: String?
        public var videoPath: String?

        public init(imagePath: String? = nil, videoPath: String? = nil) {
            self.imagePath = imagePath
            self.videoPath = videoPath
        }

        public var hasInputs: Bool {
            (imagePath != nil && !imagePath!.isEmpty) || (videoPath != nil && !videoPath!.isEmpty)
        }

        public func validate(fileManager: FileManager = .default) throws {
            if let img = imagePath, !img.isEmpty {
                guard fileManager.fileExists(atPath: img) else {
                    throw CLIError.inputFileNotFound(img)
                }
            }
            if let vid = videoPath, !vid.isEmpty {
                guard fileManager.fileExists(atPath: vid) else {
                    throw CLIError.inputFileNotFound(vid)
                }
            }
        }
    }

    // MARK: - Inspect & Admission Plan

    /// Generates the `qwg inspect <pack>` JSON report with provenance, geometry guard, and tensor table.
    public static func inspectReport(packURL: URL) throws -> String {
        let peek = try QWQuant.Pack.peekProvenance(url: packURL)
        var obj: [String: Any] = [
            "pack": packURL.lastPathComponent,
        ]
        if let mid = peek.modelId {
            obj["modelId"] = mid
            obj["geometryDigest"] = peek.geometryDigest ?? "MISSING"
        }
        let declared = peek.modelId
        let isControl = declared == nil || declared == QWModelDescriptor.Family.qwen38Dense.rawValue
        if !isControl {
            obj["modelStatus"] = "declared '\(declared ?? "?")' — engine supports only '\(QWModelDescriptor.Family.qwen38Dense.rawValue)' for tensor inspection"
            obj["tensors"] = peek.tensorCount
            obj["sourceModel"] = peek.source ?? "unknown"
            obj["sourceSha256"] = peek.sha256Source ?? "unknown"
            let data = try JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted, .sortedKeys])
            return String(decoding: data, as: UTF8.self)
        }
        let pack = try QWQuant.Pack(url: packURL)
        let prov = pack.provenance
        obj["tensors"] = pack.tensorNames.count
        obj["geometryHash"] = prov.geometryHash ?? "MISSING"
        obj["geometryGuard"] = prov.geometryHash == QWQuant.geometryHash ? "ok" : "MISMATCH"
        obj["sourceModel"] = prov.source ?? "unknown"
        obj["sourceSha256"] = prov.sha256Source ?? "unknown"
        var rows: [[String: Any]] = []
        for name in pack.tensorNames {
            let e = try pack.entry(name)
            var row: [String: Any] = ["name": name, "shape": e.shape]
            if e.hq8Quantized {
                row["kind"] = "hq8_g32_m5"
                row["rows"] = e.nRows ?? 0
                row["cols"] = e.kCols ?? 0
                row["groups"] = e.groups ?? 0
            } else if e.quantized {
                row["kind"] = "q4g"
                row["rows"] = e.nRows ?? 0
                row["cols"] = e.kCols ?? 0
                row["groups"] = e.groups ?? 0
            } else {
                row["kind"] = "bf16"
            }
            rows.append(row)
        }
        obj["table"] = rows
        let data = try JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted, .sortedKeys])
        return String(decoding: data, as: UTF8.self)
    }

    /// Generates the `qwg plan` JSON admission report pricing every resident component against the cap.
    public static func admissionPlanReport(configURL: URL,
                                           contextTokens: Int,
                                           kvMode: String = "f16",
                                           scratchMB: Int,
                                           capGB: Double?) throws -> String {
        let kv: QWAdmissionDtype
        switch kvMode {
        case "f16": kv = .f16
        case "q8":  kv = .q8
        default: throw CLIError.unpriceableKVMode(kvMode)
        }
        let descriptor = try QWModelDescriptor.decode(configJSON: try Data(contentsOf: configURL))
        let ram = QWAdmission.physicalMemoryBytes()
        let cap = capGB.map { Int64(($0 * 1_073_741_824).rounded()) } ?? Int64(ram ?? 0)
        guard cap > 0 else {
            throw CLIError.inputFileNotFound("no working-set cap (hw.memsize unreadable; pass --cap-gb)")
        }
        let plan = try QWAdmissionPlan(
            descriptor: descriptor,
            contextTokens: contextTokens,
            kvDtype: kv,
            scratchPerSeq: scratchMB * 1_048_576,
            workingSetCapBytes: Int(cap))
        var obj = plan.report()
        obj["ram_bytes"] = ram ?? NSNull()
        switch plan.admit() {
        case .admitted(let headroom):
            obj["verdict"] = "ADMIT"
            obj["headroom_bytes"] = headroom
        case .refused(let short):
            obj["verdict"] = "REFUSE"
            obj["short_by_bytes"] = short
        }
        if case .admitted = plan.admitAnotherSequence() {
            obj["marginal_verdict"] = "one_more_seq_fits"
        } else {
            obj["marginal_verdict"] = "one_more_seq_refused"
        }
        let data = try JSONSerialization.data(withJSONObject: obj,
                                              options: [.prettyPrinted, .sortedKeys])
        return String(decoding: data, as: UTF8.self)
    }

    /// Dequantizes a stored pack tensor to a little-endian float32 `.npy` file.
    public static func dequantToNpy(packURL: URL, tensor: String, out: URL) throws {
        let pack = try QWQuant.Pack(url: packURL)
        let f32: [Float]
        if try pack.entry(tensor).q8Off == nil {
            f32 = try pack.dequantQ4g(tensor)
        } else {
            f32 = try pack.dequantQ8(tensor)
        }
        try writeNpy(f32, shape: try pack.entry(tensor).shape, to: out)
    }

    // MARK: - Quantize & Pull

    /// Resolved command and provenance summary for `qwg quantize`.
    public struct QuantizePlan: Equatable {
        public var command: [String]
        public var provenanceLine: String
        public init(command: [String], provenanceLine: String) {
            self.command = command; self.provenanceLine = provenanceLine
        }
    }

    public static func quantizePlan(repo: String, outPack: URL,
                                    quant: String, scriptsDir: URL) -> QuantizePlan {
        let py = scriptsDir.deletingLastPathComponent().appendingPathComponent(".venv-oracle/bin/python")
        let packPy = scriptsDir.appendingPathComponent("quantize/pack.py")
        return QuantizePlan(
            command: [py.path, packPy.path, "--model", repo, "--out", outPack.path, "--quant", quant],
            provenanceLine: "source=\(repo) quant=\(quant) → \(outPack.lastPathComponent) (packer embeds source sha256)")
    }

    /// Splits a file byte range into `parts` chunks (at most 1 chunk per 4 MiB).
    public static func downloadPlan(size: Int, parts: Int) -> [(start: Int, end: Int)] {
        guard size > 0, parts > 0 else { return [] }
        let p = min(parts, max(1, size / (4 << 20)))
        var out: [(Int, Int)] = []
        let step = (size + p - 1) / p
        var s = 0
        while s < size {
            out.append((s, min(size, s + step) - 1))
            s += step
        }
        return out
    }

    /// Returns whether a HuggingFace repository file path is required for local model inference.
    public static func shouldDownloadRepoFile(_ path: String) -> Bool {
        let name = (path as NSString).lastPathComponent
        if name.hasPrefix(".") { return false }
        // The root README is the hub's model card; a native pack's assets/ keeps the upstream
        // README and LICENSE next to the weights, so those are pulled.
        let upper = name.uppercased()
        if upper.hasPrefix("README") || upper.hasPrefix("LICENSE") {
            return path.contains("/")
        }
        let ext = (name as NSString).pathExtension.lowercased()
        let allowedExts: Set<String> = ["json", "safetensors", "qwp", "qlsa", "sha256", "txt", "jinja",
                                        "model", "tiktoken", "bin"]
        return allowedExts.contains(ext)
    }

    /// Downloads a checkpoint from the HuggingFace Hub using parallel HTTP Range requests.
    /// Every file gets its own progress row (bar, bytes, rate) that updates while it downloads.
    @discardableResult
    public static func pullCheckpoint(repo: String, folder: URL, parts: Int = 4) throws -> URL {
        let trimmed = repo.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.contains("/"), !trimmed.contains("..") else {
            throw NSError(domain: "QWCli.pull", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Invalid HF repo id '\(repo)' (expected org/model)"])
        }
        let repoLeaf = trimmed.split(separator: "/").last.map(String.init) ?? trimmed
        let targetDir = folder.lastPathComponent == repoLeaf
            ? folder
            : folder.appendingPathComponent(repoLeaf, isDirectory: true)
        let fm = FileManager.default
        try fm.createDirectory(at: targetDir, withIntermediateDirectories: true)

        let hfBase = (ProcessInfo.processInfo.environment["HF_ENDPOINT"] ?? "https://huggingface.co")
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        let token = ProcessInfo.processInfo.environment["HF_TOKEN"]
        guard let treeURL = URL(string: "\(hfBase)/api/models/\(trimmed)/tree/main?recursive=true") else {
            throw NSError(domain: "QWCli.pull", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "Invalid tree URL for '\(trimmed)'"])
        }

        let treeData = try fetchSyncData(url: treeURL, token: token, range: nil)
        guard let entries = try JSONSerialization.jsonObject(with: treeData) as? [[String: Any]] else {
            throw NSError(domain: "QWCli.pull", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "Unexpected HF tree response for '\(trimmed)'"])
        }

        var files: [(path: String, size: Int)] = []
        for entry in entries {
            guard (entry["type"] as? String) == "file",
                  let path = entry["path"] as? String,
                  shouldDownloadRepoFile(path) else { continue }
            let lfsSize = (entry["lfs"] as? [String: Any])?["size"] as? Int
            let directSize = entry["size"] as? Int
            files.append((path: path, size: lfsSize ?? directSize ?? 0))
        }
        guard !files.isEmpty else {
            throw NSError(domain: "QWCli.pull", code: 4,
                          userInfo: [NSLocalizedDescriptionKey: "No model files found in '\(trimmed)'"])
        }

        let totalBytes = files.reduce(0) { $0 + $1.size }
        print(String(format: "pulling %@ -> %@ (%d files, %.2f GiB)",
                     trimmed, targetDir.path, files.count, Double(totalBytes) / 1_073_741_824.0))

        // One progress row per file, in repo order. Files already complete are shown as cached.
        let progress = PullProgress(files: files.map { PullProgress.Row(path: $0.path, total: $0.size) },
                                    color: .auto())
        progress.start()
        defer { progress.finish() }

        for (index, file) in files.enumerated() {
            let dstURL = targetDir.appendingPathComponent(file.path)
            try fm.createDirectory(at: dstURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            if file.size > 0,
               let attrs = try? fm.attributesOfItem(atPath: dstURL.path),
               let existingSize = attrs[.size] as? Int,
               existingSize == file.size {
                progress.markCached(index)
                continue
            }
            let encodedPath = file.path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? file.path
            guard let fileURL = URL(string: "\(hfBase)/\(trimmed)/resolve/main/\(encodedPath)") else {
                throw NSError(domain: "QWCli.pull", code: 5,
                              userInfo: [NSLocalizedDescriptionKey: "Invalid file URL for '\(file.path)'"])
            }
            try downloadFileRanged(url: fileURL, dstURL: dstURL, size: file.size, parts: max(1, parts),
                                   token: token) { received in progress.update(index, received: received) }
            progress.markDone(index)
        }

        let stFiles = files.map(\.path).filter { $0.hasSuffix(".safetensors") }
        let modelST = targetDir.appendingPathComponent("model.safetensors")
        let indexST = targetDir.appendingPathComponent("model.safetensors.index.json")
        if stFiles.count == 1, stFiles[0] != "model.safetensors",
           !fm.fileExists(atPath: modelST.path), !fm.fileExists(atPath: indexST.path) {
            try? fm.createSymbolicLink(atPath: modelST.path, withDestinationPath: stFiles[0])
        }
        return targetDir
    }

    private final class PullBox<T>: @unchecked Sendable {
        let lock = NSLock()
        var value: T
        init(_ value: T) { self.value = value }
    }

    private static func fetchSyncData(url: URL, token: String?, range: (start: Int, end: Int)?) throws -> Data {
        var req = URLRequest(url: url)
        if let token, !token.isEmpty {
            req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        if let range {
            req.setValue("bytes=\(range.start)-\(range.end)", forHTTPHeaderField: "Range")
        }
        let sem = DispatchSemaphore(value: 0)
        let box = PullBox<Result<Data, Error>>(.failure(NSError(domain: "QWCli.pull", code: -1)))
        URLSession.shared.dataTask(with: req) { data, resp, err in
            defer { sem.signal() }
            if let err {
                box.lock.lock(); box.value = .failure(err); box.lock.unlock()
                return
            }
            if let http = resp as? HTTPURLResponse, !(200...299).contains(http.statusCode) {
                let msg = "HTTP \(http.statusCode) for \(url.absoluteString)"
                box.lock.lock()
                box.value = .failure(NSError(domain: "QWCli.pull", code: http.statusCode,
                                             userInfo: [NSLocalizedDescriptionKey: msg]))
                box.lock.unlock()
                return
            }
            box.lock.lock(); box.value = .success(data ?? Data()); box.lock.unlock()
        }.resume()
        sem.wait()
        return try box.value.get()
    }

    /// Streams one ranged chunk per request into a shared `.part` file, reporting every received
    /// block, so a file's progress moves while its bytes arrive rather than when a chunk completes.
    private final class RangedFileDownload: NSObject, URLSessionDataDelegate, @unchecked Sendable {
        private struct Task { let start: Int; let expected: Int; var received: Int }
        private let lock = NSLock()
        private let handle: FileHandle
        private var tasks: [Int: Task] = [:]
        private var totalReceived = 0
        private var firstError: Error?
        private let group = DispatchGroup()
        private let progress: (Int) -> Void

        init(handle: FileHandle, progress: @escaping (Int) -> Void) {
            self.handle = handle
            self.progress = progress
        }

        func add(_ task: URLSessionDataTask, start: Int, expected: Int) {
            lock.withLock { tasks[task.taskIdentifier] = Task(start: start, expected: expected, received: 0) }
            group.enter()
        }

        /// Blocks until every chunk finished, then reports the first failure, if any.
        func wait() throws {
            group.wait()
            if let err = lock.withLock({ firstError }) { throw err }
        }

        private func fail(_ error: Error) {
            if firstError == nil { firstError = error }
        }

        func urlSession(_ session: URLSession, dataTask: URLSessionDataTask,
                        didReceive response: URLResponse, completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
            if let http = response as? HTTPURLResponse, !(200...299).contains(http.statusCode) {
                lock.withLock {
                    fail(NSError(domain: "QWCli.pull", code: http.statusCode,
                                 userInfo: [NSLocalizedDescriptionKey: "HTTP \(http.statusCode) for \(response.url?.path ?? "file")"]))
                }
                completionHandler(.cancel)
                return
            }
            completionHandler(.allow)
        }

        func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
            lock.withLock {
                guard var t = tasks[dataTask.taskIdentifier] else { return }
                let room = t.expected - t.received
                let n = min(room, data.count)
                guard n > 0 else { return }
                do {
                    try handle.seek(toOffset: UInt64(t.start + t.received))
                    try handle.write(contentsOf: n == data.count ? data : data.prefix(n))
                } catch {
                    fail(error)
                    dataTask.cancel()
                    return
                }
                t.received += n
                tasks[dataTask.taskIdentifier] = t
                totalReceived += n
                progress(totalReceived)
            }
        }

        func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
            lock.withLock {
                if let error { fail(error) }
                else if let t = tasks[task.taskIdentifier], t.received != t.expected {
                    fail(NSError(domain: "QWCli.pull", code: 6,
                                 userInfo: [NSLocalizedDescriptionKey:
                                    "short read at offset \(t.start): \(t.received) of \(t.expected) bytes"]))
                }
            }
            group.leave()
        }
    }

    /// Downloads one file as parallel ranges into `dstURL`. `progress` receives the running
    /// byte total of this file as data arrives.
    private static func downloadFileRanged(url: URL, dstURL: URL, size: Int, parts: Int, token: String?,
                                           progress: @escaping (Int) -> Void) throws {
        let plan = downloadPlan(size: size, parts: parts)
        let fm = FileManager.default
        if plan.count <= 1 {
            let data = try fetchSyncData(url: url, token: token, range: nil)
            try data.write(to: dstURL, options: .atomic)
            progress(data.count)
            return
        }
        let tmpURL = dstURL.appendingPathExtension("part")
        _ = fm.createFile(atPath: tmpURL.path, contents: nil)
        let initHandle = try FileHandle(forWritingTo: tmpURL)
        try initHandle.truncate(atOffset: UInt64(size))
        try initHandle.close()

        let writer = try FileHandle(forWritingTo: tmpURL)
        defer { try? writer.close() }
        let job = RangedFileDownload(handle: writer, progress: progress)
        let session = URLSession(configuration: .default, delegate: job, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        for chunk in plan {
            var req = URLRequest(url: url)
            if let token, !token.isEmpty {
                req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
            }
            req.setValue("bytes=\(chunk.start)-\(chunk.end)", forHTTPHeaderField: "Range")
            let task = session.dataTask(with: req)
            job.add(task, start: chunk.start, expected: chunk.end - chunk.start + 1)
            task.resume()
        }
        do {
            try job.wait()
        } catch {
            try? fm.removeItem(at: tmpURL)
            throw error
        }
        if fm.fileExists(atPath: dstURL.path) {
            try fm.removeItem(at: dstURL)
        }
        try fm.moveItem(at: tmpURL, to: dstURL)
    }

    // MARK: - NumPy Writer

    /// Writes `f32` in little-endian C-order `.npy` format byte-identical to `numpy.save`.
    static func writeNpy(_ f32: [Float], shape: [Int], to url: URL) throws {
        var desc = "{'descr': '<f4', 'fortran_order': False, 'shape': ("
        for (i, d) in shape.enumerated() { desc += "\(i == 0 ? "" : ", ")\(d)" }
        if shape.count == 1 { desc += "," }
        desc += "), }"
        var hdr = Data([0x93] + Array("NUMPY".utf8) + [1, 0])
        let L = 11 + desc.utf8.count
        let pad = (64 - L % 64) % 64
        let lenField = desc.utf8.count + pad + 1
        hdr.append(UInt8(lenField & 0xff)); hdr.append(UInt8(lenField >> 8))
        hdr.append(contentsOf: desc.utf8)
        hdr.append(contentsOf: [UInt8](repeating: 0x20, count: pad))
        hdr.append(0x0a)
        var body = Data(capacity: f32.count * 4)
        f32.withUnsafeBytes { body.append(contentsOf: $0) }
        try (hdr + body).write(to: url)
    }

    // MARK: - Terminal Styling

    /// Terminal styling with automatic TTY and `NO_COLOR` environment detection.
    public struct Color: Sendable {
        public let enabled: Bool

        public init(enabled: Bool) {
            self.enabled = enabled
        }

        public static func auto(force: Bool? = nil) -> Color {
            if let f = force { return Color(enabled: f) }
            if ProcessInfo.processInfo.environment["NO_COLOR"] != nil { return Color(enabled: false) }
            if let term = ProcessInfo.processInfo.environment["TERM"], term == "dumb" { return Color(enabled: false) }
            return Color(enabled: isatty(FileHandle.standardOutput.fileDescriptor) != 0)
        }

        public var thinking: String { enabled ? "\u{001B}[2;90m" : "" }
        public var reset: String { enabled ? "\u{001B}[0m" : "" }
        public var prompt: String { enabled ? "\u{001B}[1;36m" : "" }
        public var systemNotice: String { enabled ? "\u{001B}[33m" : "" }
        public var dim: String { enabled ? "\u{001B}[2m" : "" }
        public var bold: String { enabled ? "\u{001B}[1m" : "" }
    }
}
