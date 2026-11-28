import Darwin
import Foundation
import Metal
import QWCore
import QWHAL
import QWQuant

// BF16 PLE N-gram rows read from the source safetensors shards. The pack's `assets/ple_ngram_locations.json`
// gives, per shard, the row layout and the absolute byte offset of row 0. Two paths read it: the safetensors
// header path (the offset is taken from the parsed header and cross-checked against the sidecar) and the
// legacy `pread` path. Reference-only: reads are on demand, one row at a time.

public final class FNNgramShardRows: FNNgramRows, @unchecked Sendable {
    /// Where a shard's row bytes come from.
    enum Source {
        /// The legacy path: a `pread` at an absolute file offset.
        case file(fd: Int32, baseOffset: Int)
        /// The container path: the row bytes are read out of the mapping.
        case tensor(QWSafeTensorsReader)
    }

    struct Shard {
        /// The file the rows come from.
        let file: String
        let firstRow: Int
        let rows: Int
        let rowBytes: Int
        let source: Source
    }

    public let headDim: Int
    /// The number of rows the table holds.
    public let rowCount: Int
    let shards: [Shard]          // sorted by firstRow
    let uniqueFds: [Int32]

    public init(locations: URL, snapshot: URL) throws {
        let root = try Self.sidecar(at: locations)
        let list = try Self.shards(in: root)
        var fileToFd: [String: Int32] = [:]
        var openedFds: [Int32] = []
        var transferredFds = false
        defer {
            if !transferredFds {
                for fd in openedFds where fd >= 0 { close(fd) }
            }
        }
        var built: [Shard] = []
        for s in list {
            let file = try Self.string(s, "file", in: locations)
            let shape = try Self.shape(s, in: locations)
            let rowBytes = try Self.int(s, "row_bytes", in: locations)
            guard rowBytes == shape[1] * 2 else {
                throw QWSafeTensorsError.badHeader(
                    "the PLE shard \(file) has \(shape[1]) BF16 lanes but the locations JSON says \(rowBytes) bytes per row")
            }
            let fd: Int32
            if let existing = fileToFd[file] {
                fd = existing
            } else {
                let path = snapshot.appendingPathComponent(file).path
                let opened = open(path, O_RDONLY)
                guard opened >= 0 else {
                    throw NSError(domain: "FNNgramShardRows", code: Int(errno),
                                  userInfo: [NSLocalizedDescriptionKey: "Failed to open PLE shard \(path)"])
                }
                fileToFd[file] = opened
                openedFds.append(opened)
                fd = opened
            }
            built.append(Shard(file: file, firstRow: try Self.int(s, "first_row", in: locations),
                               rows: shape[0], rowBytes: rowBytes,
                               source: .file(fd: fd, baseOffset: try Self.int(s, "base_off", in: locations))))
        }
        let sorted = built.sorted { $0.firstRow < $1.firstRow }
        let table = try Self.validate(shards: sorted, totalRows: root["total_rows"] as? Int)
        shards = sorted
        headDim = table.headDim
        rowCount = table.rowCount
        uniqueFds = openedFds
        transferredFds = true
    }

    /// Reads the rows straight out of a safetensors payload. The **header**
    /// is the authority on the byte offsets and the sidecar on the row
    /// layout, and the two are checked against each other.
    public init(container: QWSafeTensorsContainer, locations: URL) throws {
        let root = try Self.sidecar(at: locations)
        let list = try Self.shards(in: root)
        var built: [Shard] = []
        var seenKeys: Set<String> = []
        for s in list {
            let key = try Self.string(s, "key", in: locations)
            guard !key.isEmpty else {
                throw QWSafeTensorsError.badHeader("a PLE shard entry has no tensor name")
            }
            // Two shards naming one tensor would serve both row ranges from
            // the same payload, in bounds and wrong.
            guard seenKeys.insert(key).inserted else {
                throw QWSafeTensorsError.badHeader("the PLE sidecar names \(key) twice")
            }
            let reader = try container.reader(of: key)
            let info = reader.tensorInfo
            guard info.dtype == .bf16 else {
                throw QWSafeTensorsError.unsupportedDType(info.dtype.rawValue)
            }
            if let sidecarDType = s["dtype"] as? String, sidecarDType != info.dtype.rawValue {
                throw QWSafeTensorsError.badHeader(
                    "PLE tensor \(key) is \(info.dtype.rawValue) in the header, \(sidecarDType) in the locations JSON")
            }
            let shape = try Self.shape(s, in: locations)
            guard info.shape == shape else {
                throw QWSafeTensorsError.badHeader(
                    "PLE tensor \(key) is \(info.shape) in the header, \(shape) in the locations JSON")
            }
            let rowBytes = try Self.int(s, "row_bytes", in: locations)
            guard rowBytes == shape[1] * 2 else {
                throw QWSafeTensorsError.badHeader(
                    "PLE tensor \(key) has \(shape[1]) BF16 lanes but the locations JSON says \(rowBytes) bytes per row")
            }
            // The sidecar's absolute offset must agree with the header's, and
            // it must be there: without it there is nothing to agree with.
            let base = try Self.int(s, "base_off", in: locations)
            guard let range = container.byteRange(of: key) else {
                throw QWSafeTensorsError.missingTensor(key)
            }
            guard range.lowerBound == base else {
                throw QWSafeTensorsError.badHeader(
                    "PLE tensor \(key) starts at \(range.lowerBound) in the file, the locations JSON says \(base)")
            }
            built.append(Shard(file: reader.shardURL.lastPathComponent,
                               firstRow: try Self.int(s, "first_row", in: locations),
                               rows: shape[0], rowBytes: rowBytes, source: .tensor(reader)))
        }
        let sorted = built.sorted { $0.firstRow < $1.firstRow }
        let table = try Self.validate(shards: sorted, totalRows: root["total_rows"] as? Int)
        shards = sorted
        headDim = table.headDim
        rowCount = table.rowCount
        uniqueFds = []
    }

    /// The sidecar's parsed root.
    private static func sidecar(at locations: URL) throws -> [String: Any] {
        let raw = try Data(contentsOf: locations)
        guard let root = try JSONSerialization.jsonObject(with: raw) as? [String: Any] else {
            throw QWSafeTensorsError.badHeader("the PLE locations sidecar is not an object")
        }
        return root
    }

    /// The sidecar's shard list. A malformed sidecar is refused, not fatal.
    private static func shards(in root: [String: Any]) throws -> [[String: Any]] {
        guard let list = root["shards"] as? [[String: Any]] else {
            throw QWSafeTensorsError.badHeader("the PLE locations sidecar has no shard list")
        }
        return list
    }

    /// A string field of a shard entry.
    private static func string(_ shard: [String: Any], _ key: String,
                               in locations: URL) throws -> String {
        guard let value = shard[key] as? String else {
            throw QWSafeTensorsError.badHeader(
                "a PLE shard entry in \(locations.lastPathComponent) has no '\(key)'")
        }
        return value
    }

    /// An integer field of a shard entry.
    private static func int(_ shard: [String: Any], _ key: String,
                            in locations: URL) throws -> Int {
        guard let value = shard[key] as? Int else {
            throw QWSafeTensorsError.badHeader(
                "a PLE shard entry in \(locations.lastPathComponent) has no '\(key)'")
        }
        return value
    }

    /// A shard's shape.
    private static func shape(_ shard: [String: Any], in locations: URL) throws -> [Int] {
        guard let shape = shard["shape"] as? [Int], shape.count == 2 else {
            throw QWSafeTensorsError.badHeader(
                "a PLE shard entry in \(locations.lastPathComponent) has no two-dimensional 'shape'")
        }
        return shape
    }

    /// The tensor names the sidecar names. A reader that needs only these
    /// should not map the rest of the checkpoint.
    public static func tensorNames(in locations: URL) throws -> Set<String> {
        let list = try shards(in: sidecar(at: locations))
        var names: Set<String> = []
        for s in list {
            let key = try string(s, "key", in: locations)
            guard !key.isEmpty else {
                throw QWSafeTensorsError.badHeader("a PLE shard entry has no tensor name")
            }
            names.insert(key)
        }
        return names
    }

    /// Checks that the shard list agrees with itself and with the sidecar, and
    /// returns the table's head dim and row count. A shard that disagrees
    /// about its row size would read off the end of its own tensor.
    private static func validate(shards: [Shard], totalRows: Int?) throws -> (headDim: Int, rowCount: Int) {
        guard let first = shards.first else {
            throw QWSafeTensorsError.badHeader("the PLE locations sidecar lists no shards")
        }
        guard first.rowBytes % 2 == 0 else {
            throw QWSafeTensorsError.badHeader(
                "a PLE row of \(first.rowBytes) bytes is not a whole number of BF16 lanes")
        }
        let headDim = first.rowBytes / 2
        var expectedFirstRow = 0
        var rowCount = 0
        for s in shards {
            guard s.rowBytes == first.rowBytes else {
                throw QWSafeTensorsError.badHeader(
                    "the PLE shard at row \(s.firstRow) has \(s.rowBytes) bytes per row, the table has \(first.rowBytes)")
            }
            guard s.firstRow == expectedFirstRow else {
                throw QWSafeTensorsError.badHeader(
                    "the PLE shard starts at row \(s.firstRow), the table expects row \(expectedFirstRow)")
            }
            expectedFirstRow += s.rows
            rowCount += s.rows
        }
        if let totalRows, totalRows != rowCount {
            throw QWSafeTensorsError.badHeader(
                "the PLE sidecar says \(totalRows) rows, its shards hold \(rowCount)")
        }
        return (headDim, rowCount)
    }

    deinit {
        for fd in uniqueFds where fd >= 0 { close(fd) }
    }

    public func copyRow(_ id: Int, to dst: UnsafeMutablePointer<Float>) throws {
        var lo = 0, hi = shards.count - 1
        while lo < hi {
            let mid = (lo + hi + 1) / 2
            if shards[mid].firstRow <= id { lo = mid } else { hi = mid - 1 }
        }
        let s = shards[lo]
        guard id >= s.firstRow && id < s.firstRow + s.rows else {
            throw FNNgramRowsError.rowOutOfBounds(id: id, count: rowCount)
        }
        let rowInShard = id - s.firstRow
        let n = headDim
        switch s.source {
        case .file(let fd, let baseOffset):
            let off = Int64(baseOffset + rowInShard * s.rowBytes)
            let u16 = UnsafeMutableRawPointer(dst).assumingMemoryBound(to: UInt16.self).advanced(by: n)
            try QWPOSIX.preadExact(fd: fd, into: UnsafeMutableRawPointer(u16), count: s.rowBytes, offset: off)
            for i in 0..<n {
                dst[i] = Float(bitPattern: UInt32(u16[i]) << 16)
            }
        case .tensor(let reader):
            reader.withBytes { raw in
                let base = rowInShard * s.rowBytes
                QWNgramFormat.decodeRow(dtype: .bf16, from: raw.baseAddress!.advanced(by: base),
                                        headDim: n, into: dst)
            }
        }
    }

    public func row(_ id: Int) -> [Float] {
        let n = headDim
        return [Float](unsafeUninitializedCapacity: n) { out, count in
            qwCopyNgramRowBestEffort(self, id: id, to: out.baseAddress!, headDim: n)
            count = n
        }
    }
}

/// Resolves the PLE N-gram row table from a model's `assets/` directory, preferring self-contained
/// `assets/ngram.bin`, then `ngram_table.bin`, then `assets/ple_ngram_locations.json` + the HF snapshot shards.
public enum FNNgramRowsLoader {
    public static func open(assetsURL: URL, mapped: Bool = false) throws -> any FNNgramRows {
        let ngramBin = assetsURL.appendingPathComponent("ngram.bin")
        if FileManager.default.fileExists(atPath: ngramBin.path) {
            return try FNNgramBinRows(url: ngramBin, mapped: mapped)
        }
        let ngramTableBin = assetsURL.appendingPathComponent("ngram_table.bin")
        if FileManager.default.fileExists(atPath: ngramTableBin.path) {
            return try FNNgramMLXRows(url: ngramTableBin, mapped: mapped)
        }
        let locations = assetsURL.appendingPathComponent("ple_ngram_locations.json")
        if FileManager.default.fileExists(atPath: locations.path),
           let snapshot = snapshotDirectory(assetsURL: assetsURL, locations: locations) {
            // Prefer the container: the row offsets come from the parsed
            // safetensors header, and the payload stays mapped, so a row
            // read is a memory read and not a `pread`.
            if let index = indexJSON(in: snapshot) {
                // Only the shards the sidecar names are mapped: the sidecar's
                // 33 files, not the checkpoint's 131.
                let wanted = try FNNgramShardRows.tensorNames(in: locations)
                let container = try QWSafeTensorsContainer(url: index, onlyTensors: wanted)
                return try FNNgramShardRows(container: container, locations: locations)
            }
            return try FNNgramShardRows(locations: locations, snapshot: snapshot)
        }
        throw NSError(domain: "FNNgramRowsLoader", code: 1,
                      userInfo: [NSLocalizedDescriptionKey: "Missing assets/ngram.bin in \(assetsURL.path)"])
    }

    /// Snapshot directory for PLE shards: env overrides, pack-local, then HF cache matching `locations`.
    static func snapshotDirectory(assetsURL: URL? = nil, locations: URL? = nil) -> URL? {
        if let envSnap = ProcessInfo.processInfo.environment["FN_SNAPSHOT_DIR"], !envSnap.isEmpty {
            return URL(fileURLWithPath: envSnap)
        }
        if let envAssets = ProcessInfo.processInfo.environment["QW_PLE_ASSETS_DIR"], !envAssets.isEmpty {
            return URL(fileURLWithPath: envAssets)
        }
        let requiredFiles = shardFilenames(in: locations)
        let fm = FileManager.default
        if let assetsURL {
            for candidate in [assetsURL.deletingLastPathComponent(), assetsURL] {
                if snapshotMatches(candidate, requiredFiles: requiredFiles, fm: fm) {
                    return candidate
                }
            }
        }
        let hubRoot = fm.homeDirectoryForCurrentUser.appendingPathComponent(".cache/huggingface/hub")
        guard let models = try? fm.contentsOfDirectory(atPath: hubRoot.path) else { return nil }
        for m in models.sorted() where m.hasPrefix("models--") {
            let snapsRoot = hubRoot.appendingPathComponent(m).appendingPathComponent("snapshots")
            guard let items = try? fm.contentsOfDirectory(atPath: snapsRoot.path) else { continue }
            for rev in items.sorted() where !rev.hasPrefix(".") {
                let candidate = snapsRoot.appendingPathComponent(rev)
                if snapshotMatches(candidate, requiredFiles: requiredFiles, fm: fm) {
                    return candidate
                }
            }
        }
        return nil
    }

    private static func shardFilenames(in locations: URL?) -> [String] {
        guard let locations,
              let data = try? Data(contentsOf: locations),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let shards = root["shards"] as? [[String: Any]] else {
            return []
        }
        var seen = Set<String>()
        var files: [String] = []
        for s in shards {
            if let f = s["file"] as? String, !f.isEmpty, seen.insert(f).inserted {
                files.append(f)
            }
        }
        return files
    }

    private static func snapshotMatches(_ dir: URL, requiredFiles: [String], fm: FileManager) -> Bool {
        guard !requiredFiles.isEmpty else {
            return indexJSON(in: dir) != nil
        }
        return requiredFiles.allSatisfy { fm.fileExists(atPath: dir.appendingPathComponent($0).path) }
    }

    /// The checkpoint's tensor index, if the snapshot has one.
    static func indexJSON(in snapshot: URL) -> URL? {
        let items = (try? FileManager.default.contentsOfDirectory(atPath: snapshot.path)) ?? []
        return items.sorted().first(where: { $0.hasSuffix(".index.json") })
            .map { snapshot.appendingPathComponent($0) }
    }
}
