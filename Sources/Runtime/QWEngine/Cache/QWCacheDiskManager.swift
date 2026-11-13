import Foundation
import QWCore
import QWHAL
import QWKV

/// Manages ephemeral cold Tier-3 prefix cache storage on NVMe SSD.
/// Uses unbuffered direct I/O via QWDispatchIO for high-throughput streaming.
public final class QWCacheDiskManager: @unchecked Sendable {
    enum ReadError: Error, CustomStringConvertible {
        case invalidHeader
        case metadataMismatch
        case truncated(offset: Int64, expected: Int, actual: Int)

        var description: String {
            switch self {
            case .invalidHeader: return "invalid cache metadata header"
            case .metadataMismatch: return "cache metadata differs from its index"
            case .truncated(let offset, let expected, let actual):
                return "short cache read at \(offset): expected \(expected) bytes, got \(actual)"
            }
        }
    }

    static func readExactly(_ io: QWDispatchIO, offset: Int64,
                            into destination: UnsafeMutableRawPointer, length: Int) throws {
        let actual = try io.readSync(offset: offset, into: destination, length: length)
        guard actual == length else {
            throw ReadError.truncated(offset: offset, expected: length, actual: actual)
        }
    }

    public struct StoredMetadata: Codable, Sendable, Equatable {
        public let tokens: [Int]
        public let mode: String
        public let nextToken: Int
        public let position: Int
        public let generated: Int
        public let roundRecord: QWRoundRecord
        public let recBytes: Int
        public let ringBytes: Int
        public let trunkHiddenBytes: Int
        public let trunkHiddenValid: Bool
        public let logitsBytes: Int
        public let trunkTokenCount: Int
        public let trunkPlaneSizes: [Int]
        public let hasMtp: Bool
        public let mtpTokenCount: Int
        public let mtpPlaneSizes: [Int]
        public let allowsSuffix: Bool
        public var formatVersion: Int? = nil
        public var mlaIndexerBytes: Int? = nil
    }

    public struct DiskHit: Sendable {
        public let metadata: StoredMetadata
        public let fileURL: URL
        public let matchedTokens: Int
    }

    public let config: QWCacheConfig
    private let lock = NSLock()
    private var entries: [String: StoredMetadata] = [:]
    private var entryFiles: [String: URL] = [:]
    private var entryBytes: [String: Int] = [:]
    private var lru: [String] = []
    public private(set) var chargedBytes: Int = 0
    public var entryCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return entries.count
    }

    public init(config: QWCacheConfig) {
        self.config = config
        prepareDirectory()
        if config.wipeOnStart {
            purgeAll()
        } else {
            loadDiskIndex()
        }
    }

    deinit {
        if config.wipeOnStart {
            purgeAll()
        }
    }

    public func prepareDirectory() {
        try? FileManager.default.createDirectory(at: config.cacheDirectory,
                                                 withIntermediateDirectories: true,
                                                 attributes: [.posixPermissions: 0o700])
    }

    /// Rebuilds the in-memory index from existing `.qwcache` files on SSD.
    public func loadDiskIndex() {
        lock.lock()
        defer { lock.unlock() }
        let fm = FileManager.default
        guard let files = try? fm.contentsOfDirectory(
            at: config.cacheDirectory,
            includingPropertiesForKeys: [.fileSizeKey, .contentModificationDateKey]
        ) else { return }

        let decoder = JSONDecoder()
        var discovered: [(key: String, meta: StoredMetadata, url: URL, bytes: Int, date: Date)] = []

        for file in files where file.pathExtension == "qwcache" && file.lastPathComponent.hasPrefix("entry_") {
            guard let io = try? QWDispatchIO(path: file.path, mode: .readOnly, directIO: false) else { continue }
            defer { io.close() }
            var metaLen: UInt64 = 0
            let readLen = withUnsafeMutableBytes(of: &metaLen) { raw in
                (try? io.readSync(offset: 0, into: raw.baseAddress!, length: MemoryLayout<UInt64>.size)) ?? 0
            }
            guard readLen == MemoryLayout<UInt64>.size, metaLen > 0, metaLen < 16 * 1024 * 1024 else { continue }
            var metaData = Data(count: Int(metaLen))
            let actualMeta = metaData.withUnsafeMutableBytes { raw in
                (try? io.readSync(offset: Int64(MemoryLayout<UInt64>.size), into: raw.baseAddress!, length: Int(metaLen))) ?? 0
            }
            guard actualMeta == Int(metaLen),
                  let meta = try? decoder.decode(StoredMetadata.self, from: metaData) else { continue }

            let key = Self.cacheKey(tokens: meta.tokens, mode: meta.mode)
            let resourceValues = try? file.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey])
            let fileSize = resourceValues?.fileSize ?? (
                MemoryLayout<UInt64>.size + Int(metaLen) + meta.recBytes + meta.ringBytes +
                meta.trunkHiddenBytes + meta.logitsBytes +
                meta.trunkPlaneSizes.reduce(0, +) + meta.mtpPlaneSizes.reduce(0, +)
            )
            let modDate = resourceValues?.contentModificationDate ?? Date.distantPast
            discovered.append((key, meta, file, fileSize, modDate))
        }

        discovered.sort { $0.date < $1.date }
        for item in discovered {
            guard item.bytes <= config.ssdByteLimit else { continue }
            while chargedBytes + item.bytes > config.ssdByteLimit, !lru.isEmpty {
                evictOldest()
            }
            entries[item.key] = item.meta
            entryFiles[item.key] = item.url
            entryBytes[item.key] = item.bytes
            chargedBytes += item.bytes
            touch(item.key)
        }
    }

    /// Purges all cache files and resets index.
    public func purgeAll() {
        lock.lock()
        defer { lock.unlock() }
        let fm = FileManager.default
        if let files = try? fm.contentsOfDirectory(at: config.cacheDirectory, includingPropertiesForKeys: nil) {
            for file in files where file.pathExtension == "qwcache" || file.lastPathComponent.hasPrefix("entry_") {
                try? fm.removeItem(at: file)
            }
        }
        entries.removeAll()
        entryFiles.removeAll()
        entryBytes.removeAll()
        lru.removeAll()
        chargedBytes = 0
    }

    /// Deterministic FNV-1a-64 hash across process restarts.
    public static func cacheKey(tokens: [Int], mode: String) -> String {
        let hash = QWPrefixHash.fnv1a64(tokens: tokens, mode: mode)
        return String(format: "%016llx_%d", hash, tokens.count)
    }

    private static func alignUp(_ value: Int, to alignment: Int = QWDispatchIO.pageSize) -> Int {
        let mask = alignment - 1
        return (value + mask) & ~mask
    }

    private static func readHeader(_ io: QWDispatchIO, expected: StoredMetadata? = nil) throws -> (meta: StoredMetadata, nextOffset: Int64) {
        var length: UInt64 = 0
        try withUnsafeMutableBytes(of: &length) {
            try readExactly(io, offset: 0, into: $0.baseAddress!, length: MemoryLayout<UInt64>.size)
        }
        guard length > 0, length < 16 * 1024 * 1024 else { throw ReadError.invalidHeader }
        var data = Data(count: Int(length))
        try data.withUnsafeMutableBytes {
            try readExactly(io, offset: Int64(MemoryLayout<UInt64>.size),
                            into: $0.baseAddress!, length: $0.count)
        }
        let decoded = try JSONDecoder().decode(StoredMetadata.self, from: data)
        if let expected, decoded != expected {
            throw ReadError.metadataMismatch
        }
        return (decoded, Int64(MemoryLayout<UInt64>.size) + Int64(length))
    }

    /// Look up a matching prefix on SSD (exact or longest prefix if allowed).
    public func lookup(tokens: [Int], mode: String) -> DiskHit? {
        lock.lock()
        defer { lock.unlock() }
        let exact = Self.cacheKey(tokens: tokens, mode: mode)
        if let meta = entries[exact], let url = entryFiles[exact] {
            touch(exact)
            return DiskHit(metadata: meta, fileURL: url, matchedTokens: tokens.count)
        }
        var bestKey: String?
        var bestCount = 0
        for (key, meta) in entries where meta.allowsSuffix && meta.mode == mode &&
            meta.tokens.count < tokens.count && meta.tokens.count > bestCount {
            if tokens.starts(with: meta.tokens) {
                bestKey = key
                bestCount = meta.tokens.count
            }
        }
        guard let bestKey, let meta = entries[bestKey], let url = entryFiles[bestKey] else {
            return nil
        }
        touch(bestKey)
        return DiskHit(metadata: meta, fileURL: url, matchedTokens: bestCount)
    }

    /// Streams shared prefix directly from unified memory / KV pools to an arbitrary `.qwcache` file URL.
    static func writeStandaloneFile(
        to fileURL: URL,
        tokens: [Int],
        mode: String,
        nextToken: Int,
        state: QWEnginePrefixState,
        logits: Data,
        trunk: QWKVSharedPrefix,
        kvCache: QWKVCache,
        mtp: QWKVSharedPrefix?,
        mtpCache: QWKVCache?,
        allowsSuffix: Bool,
        maxBytes: Int? = nil,
        onBeforeWrite: ((Int) -> Void)? = nil
    ) throws -> (metadata: StoredMetadata, totalBytes: Int)? {
        var trunkPlaneSizes: [Int] = []
        var trunkPageCounts: [Int: Int] = [:]
        try kvCache.withSharedPrefixPagePointers(trunk) { planeIdx, _, _, byteCount in
            trunkPageCounts[planeIdx, default: 0] += byteCount
        }
        for i in 0..<(trunkPageCounts.count) {
            trunkPlaneSizes.append(trunkPageCounts[i] ?? 0)
        }

        var mtpPlaneSizes: [Int] = []
        if let mtp, let mtpCache {
            var mtpPageCounts: [Int: Int] = [:]
            try mtpCache.withSharedPrefixPagePointers(mtp) { planeIdx, _, _, byteCount in
                mtpPageCounts[planeIdx, default: 0] += byteCount
            }
            for i in 0..<(mtpPageCounts.count) {
                mtpPlaneSizes.append(mtpPageCounts[i] ?? 0)
            }
        }

        let meta = StoredMetadata(
            tokens: tokens,
            mode: mode,
            nextToken: nextToken,
            position: state.position,
            generated: state.generated,
            roundRecord: state.roundRecord,
            recBytes: state.recData.count,
            ringBytes: state.ringData.count,
            trunkHiddenBytes: state.trunkHiddenData?.count ?? 0,
            trunkHiddenValid: state.trunkHiddenValid,
            logitsBytes: logits.count,
            trunkTokenCount: state.position,
            trunkPlaneSizes: trunkPlaneSizes,
            hasMtp: !mtpPlaneSizes.isEmpty,
            mtpTokenCount: state.roundRecord.mtpPhysicalRows,
            mtpPlaneSizes: mtpPlaneSizes,
            allowsSuffix: allowsSuffix,
            formatVersion: 2,
            mlaIndexerBytes: state.mlaIndexerData?.count
        )

        let metaData = try JSONEncoder().encode(meta)
        var metaLen = UInt64(metaData.count)
        let headerRawLen = MemoryLayout<UInt64>.size + metaData.count +
            state.recData.count + state.ringData.count + (state.trunkHiddenData?.count ?? 0) +
            (state.mlaIndexerData?.count ?? 0) + logits.count
        let alignedHeaderLen = Self.alignUp(headerRawLen)
        let kvPayloadBytes = trunkPlaneSizes.reduce(0, +) + mtpPlaneSizes.reduce(0, +)
        let totalPayloadBytes = alignedHeaderLen + kvPayloadBytes

        if let maxBytes, totalPayloadBytes > maxBytes { return nil }
        onBeforeWrite?(totalPayloadBytes)

        try? FileManager.default.createDirectory(
            at: fileURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let io = try QWDispatchIO(path: fileURL.path, mode: .writeOnly, directIO: true)
        defer { io.close() }

        var headerBuf = Data(capacity: alignedHeaderLen)
        withUnsafeBytes(of: &metaLen) { headerBuf.append(contentsOf: $0) }
        headerBuf.append(metaData)
        headerBuf.append(state.recData)
        headerBuf.append(state.ringData)
        if let hidden = state.trunkHiddenData {
            headerBuf.append(hidden)
        }
        if let mla = state.mlaIndexerData {
            headerBuf.append(mla)
        }
        headerBuf.append(logits)
        if headerBuf.count < alignedHeaderLen {
            headerBuf.append(Data(count: alignedHeaderLen - headerBuf.count))
        }

        try headerBuf.withUnsafeBytes { raw in
            try io.writeSync(offset: 0, buffer: raw.baseAddress!, length: raw.count)
        }

        var writeOffset = Int64(alignedHeaderLen)
        try kvCache.withSharedPrefixPagePointers(trunk) { _, _, ptr, byteCount in
            try io.writeSync(offset: writeOffset, buffer: ptr, length: byteCount)
            writeOffset += Int64(byteCount)
        }
        if let mtp, let mtpCache {
            try mtpCache.withSharedPrefixPagePointers(mtp) { _, _, ptr, byteCount in
                try io.writeSync(offset: writeOffset, buffer: ptr, length: byteCount)
                writeOffset += Int64(byteCount)
            }
        }
        return (meta, totalPayloadBytes)
    }

    /// Streams shared prefix directly from unified memory / KV pools to SSD with F_NOCACHE direct I/O.
    func writeDirect(
        tokens: [Int],
        mode: String,
        nextToken: Int,
        state: QWEnginePrefixState,
        logits: Data,
        trunk: QWKVSharedPrefix,
        kvCache: QWKVCache,
        mtp: QWKVSharedPrefix?,
        mtpCache: QWKVCache?,
        allowsSuffix: Bool
    ) throws -> URL? {
        lock.lock()
        defer { lock.unlock() }

        let key = Self.cacheKey(tokens: tokens, mode: mode)
        let filename = "entry_\(key).qwcache"
        let fileURL = config.cacheDirectory.appendingPathComponent(filename)

        guard let written = try Self.writeStandaloneFile(
            to: fileURL,
            tokens: tokens,
            mode: mode,
            nextToken: nextToken,
            state: state,
            logits: logits,
            trunk: trunk,
            kvCache: kvCache,
            mtp: mtp,
            mtpCache: mtpCache,
            allowsSuffix: allowsSuffix,
            maxBytes: config.ssdByteLimit,
            onBeforeWrite: { [weak self] totalPayloadBytes in
                guard let self else { return }
                while self.chargedBytes + totalPayloadBytes > self.config.ssdByteLimit, !self.lru.isEmpty {
                    self.evictOldest()
                }
            }
        ) else { return nil }

        if let prevBytes = entryBytes[key] {
            chargedBytes -= prevBytes
        }
        entries[key] = written.metadata
        entryFiles[key] = fileURL
        entryBytes[key] = written.totalBytes
        chargedBytes += written.totalBytes
        touch(key)
        return fileURL
    }

    /// Reads an arbitrary `.qwcache` file directly into KV pool pages via F_NOCACHE DispatchIO.
    static func readStandaloneFile(
        at fileURL: URL,
        expectedMetadata: StoredMetadata? = nil,
        kvCache: QWKVCache,
        mtpCache: QWKVCache?
    ) throws -> (
        metadata: StoredMetadata,
        state: QWEnginePrefixState,
        nextToken: Int,
        logits: Data,
        trunk: QWKVSharedPrefix,
        mtp: QWKVSharedPrefix?
    ) {
        let io = try QWDispatchIO(path: fileURL.path, mode: .readOnly, directIO: true)
        defer { io.close() }

        let header = try Self.readHeader(io, expected: expectedMetadata)
        let meta = header.meta
        var offset = header.nextOffset

        var recData = Data(count: meta.recBytes)
        if meta.recBytes > 0 {
            try recData.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
                if let base = raw.baseAddress {
                    try Self.readExactly(io, offset: offset, into: base, length: meta.recBytes)
                }
            }
        }
        offset += Int64(meta.recBytes)

        var ringData = Data(count: meta.ringBytes)
        if meta.ringBytes > 0 {
            try ringData.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
                if let base = raw.baseAddress {
                    try Self.readExactly(io, offset: offset, into: base, length: meta.ringBytes)
                }
            }
        }
        offset += Int64(meta.ringBytes)

        var trunkHiddenData: Data? = nil
        if meta.trunkHiddenBytes > 0 {
            var hidden = Data(count: meta.trunkHiddenBytes)
            try hidden.withUnsafeMutableBytes {
                try Self.readExactly(io, offset: offset, into: $0.baseAddress!, length: meta.trunkHiddenBytes)
            }
            trunkHiddenData = hidden
            offset += Int64(meta.trunkHiddenBytes)
        }

        var mlaIndexerData: Data? = nil
        if let mlaBytes = meta.mlaIndexerBytes, mlaBytes > 0 {
            var mla = Data(count: mlaBytes)
            try mla.withUnsafeMutableBytes {
                try Self.readExactly(io, offset: offset, into: $0.baseAddress!, length: mlaBytes)
            }
            mlaIndexerData = mla
            offset += Int64(mlaBytes)
        }

        var logits = Data(count: meta.logitsBytes)
        try logits.withUnsafeMutableBytes {
            if let base = $0.baseAddress {
                try Self.readExactly(io, offset: offset, into: base, length: meta.logitsBytes)
            }
        }
        offset += Int64(meta.logitsBytes)

        if (meta.formatVersion ?? 1) >= 2 {
            offset = Int64(Self.alignUp(Int(offset)))
        }

        let tokenCount = meta.position
        let tempH = try kvCache.allocate(promptTokens: tokenCount)
        defer { kvCache.finish(tempH) }
        try kvCache.populatePrivatePagesDirectly(into: tempH, tokenCount: tokenCount) { _, _, ptr, byteCount in
            try Self.readExactly(io, offset: offset, into: ptr, length: byteCount)
            offset += Int64(byteCount)
        }
        let trunkShared = try kvCache.captureSharedPrefix(tempH)

        var mtpShared: QWKVSharedPrefix? = nil
        if let mtpCache, meta.hasMtp, !meta.mtpPlaneSizes.isEmpty {
            let mtpCount = meta.roundRecord.mtpPhysicalRows
            let tempM = try mtpCache.allocate(promptTokens: mtpCount)
            defer { mtpCache.finish(tempM) }
            try mtpCache.populatePrivatePagesDirectly(into: tempM, tokenCount: mtpCount) { _, _, ptr, byteCount in
                try Self.readExactly(io, offset: offset, into: ptr, length: byteCount)
                offset += Int64(byteCount)
            }
            mtpShared = try mtpCache.captureSharedPrefix(tempM)
        }

        let prefixState = QWEnginePrefixState(
            position: meta.position,
            generated: meta.generated,
            roundRecord: meta.roundRecord,
            recData: recData,
            ringData: ringData,
            trunkHiddenData: trunkHiddenData,
            trunkHiddenValid: meta.trunkHiddenValid,
            stochastic: nil,
            mlaIndexerData: mlaIndexerData
        )

        return (meta, prefixState, meta.nextToken, logits, trunkShared, mtpShared)
    }

    /// Reads SSD cache directly into KV pool pages via F_NOCACHE DispatchIO with zero intermediate KV heap copies.
    func readDirectIntoCache(
        hit: DiskHit,
        kvCache: QWKVCache,
        mtpCache: QWKVCache?
    ) throws -> (
        state: QWEnginePrefixState,
        nextToken: Int,
        logits: Data,
        trunk: QWKVSharedPrefix,
        mtp: QWKVSharedPrefix?
    ) {
        lock.lock()
        defer { lock.unlock() }
        let read = try Self.readStandaloneFile(
            at: hit.fileURL,
            expectedMetadata: hit.metadata,
            kvCache: kvCache,
            mtpCache: mtpCache
        )
        return (read.state, read.nextToken, read.logits, read.trunk, read.mtp)
    }

    /// Writes prefix state to SSD via QWDispatchIO.
    func write(tokens: [Int], mode: String, nextToken: Int,
               state: QWEnginePrefixState, logits: Data,
               trunkPlanes: [Data], mtpPlanes: [Data],
               allowsSuffix: Bool) throws -> URL? {
        lock.lock()
        defer { lock.unlock() }

        let key = Self.cacheKey(tokens: tokens, mode: mode)
        let filename = "entry_\(key).qwcache"
        let fileURL = config.cacheDirectory.appendingPathComponent(filename)

        let meta = StoredMetadata(
            tokens: tokens,
            mode: mode,
            nextToken: nextToken,
            position: state.position,
            generated: state.generated,
            roundRecord: state.roundRecord,
            recBytes: state.recData.count,
            ringBytes: state.ringData.count,
            trunkHiddenBytes: state.trunkHiddenData?.count ?? 0,
            trunkHiddenValid: state.trunkHiddenValid,
            logitsBytes: logits.count,
            trunkTokenCount: state.position,
            trunkPlaneSizes: trunkPlanes.map { $0.count },
            hasMtp: !mtpPlanes.isEmpty,
            mtpTokenCount: state.roundRecord.mtpPhysicalRows,
            mtpPlaneSizes: mtpPlanes.map { $0.count },
            allowsSuffix: allowsSuffix,
            formatVersion: 2,
            mlaIndexerBytes: state.mlaIndexerData?.count
        )

        let encoder = JSONEncoder()
        let metaData = try encoder.encode(meta)
        var metaLen = UInt64(metaData.count)

        let headerRawLen = MemoryLayout<UInt64>.size + metaData.count +
            state.recData.count + state.ringData.count + (state.trunkHiddenData?.count ?? 0) +
            (state.mlaIndexerData?.count ?? 0) + logits.count
        let alignedHeaderLen = Self.alignUp(headerRawLen)
        let kvBytes = trunkPlanes.reduce(0, { $0 + $1.count }) + mtpPlanes.reduce(0, { $0 + $1.count })
        let totalPayloadBytes = alignedHeaderLen + kvBytes

        // Check against SSD capacity limit
        guard totalPayloadBytes <= config.ssdByteLimit else { return nil }

        // Evict oldest until within SSD capacity
        while chargedBytes + totalPayloadBytes > config.ssdByteLimit, !lru.isEmpty {
            evictOldest()
        }

        // Assemble buffer
        var buffer = Data(capacity: totalPayloadBytes)
        withUnsafeBytes(of: &metaLen) { buffer.append(contentsOf: $0) }
        buffer.append(metaData)
        buffer.append(state.recData)
        buffer.append(state.ringData)
        if let hidden = state.trunkHiddenData {
            buffer.append(hidden)
        }
        if let mla = state.mlaIndexerData {
            buffer.append(mla)
        }
        buffer.append(logits)
        if buffer.count < alignedHeaderLen {
            buffer.append(Data(count: alignedHeaderLen - buffer.count))
        }
        for plane in trunkPlanes { buffer.append(plane) }
        for plane in mtpPlanes { buffer.append(plane) }

        // Write using QWDispatchIO with F_NOCACHE directIO
        let io = try QWDispatchIO(path: fileURL.path, mode: .writeOnly, directIO: true)
        defer { io.close() }
        try buffer.withUnsafeBytes { raw in
            try io.writeSync(offset: 0, buffer: raw.baseAddress!, length: raw.count)
        }

        if let prevBytes = entryBytes[key] {
            chargedBytes -= prevBytes
        }
        entries[key] = meta
        entryFiles[key] = fileURL
        entryBytes[key] = totalPayloadBytes
        chargedBytes += totalPayloadBytes
        touch(key)
        return fileURL
    }

    /// Reads deserialized prefix payload from SSD via QWDispatchIO.
    func read(hit: DiskHit) throws -> (
        state: QWEnginePrefixState,
        nextToken: Int,
        logits: Data,
        trunkPlanes: [Data],
        mtpPlanes: [Data]
    ) {
        lock.lock()
        defer { lock.unlock() }

        let meta = hit.metadata
        let fileURL = hit.fileURL
        let io = try QWDispatchIO(path: fileURL.path, mode: .readOnly, directIO: true)
        defer { io.close() }

        var offset = try Self.readHeader(io, expected: meta).nextOffset

        // Read recData
        var recData = Data(count: meta.recBytes)
        if meta.recBytes > 0 {
            try recData.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
                if let base = raw.baseAddress {
                    _ = try io.readSync(offset: offset, into: base, length: meta.recBytes)
                }
            }
        }
        offset += Int64(meta.recBytes)

        // Read ringData
        var ringData = Data(count: meta.ringBytes)
        if meta.ringBytes > 0 {
            try ringData.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
                if let base = raw.baseAddress {
                    _ = try io.readSync(offset: offset, into: base, length: meta.ringBytes)
                }
            }
        }
        offset += Int64(meta.ringBytes)

        // Read trunkHiddenData
        var trunkHiddenData: Data? = nil
        if meta.trunkHiddenBytes > 0 {
            var hidden = Data(count: meta.trunkHiddenBytes)
            try hidden.withUnsafeMutableBytes {
                _ = try io.readSync(offset: offset, into: $0.baseAddress!, length: meta.trunkHiddenBytes)
            }
            trunkHiddenData = hidden
            offset += Int64(meta.trunkHiddenBytes)
        }

        var mlaIndexerData: Data? = nil
        if let mlaBytes = meta.mlaIndexerBytes, mlaBytes > 0 {
            var mla = Data(count: mlaBytes)
            try mla.withUnsafeMutableBytes {
                try Self.readExactly(io, offset: offset, into: $0.baseAddress!, length: mlaBytes)
            }
            mlaIndexerData = mla
            offset += Int64(mlaBytes)
        }

        // Read logits
        var logits = Data(count: meta.logitsBytes)
        try logits.withUnsafeMutableBytes {
            _ = try io.readSync(offset: offset, into: $0.baseAddress!, length: meta.logitsBytes)
        }
        offset += Int64(meta.logitsBytes)

        if (meta.formatVersion ?? 1) >= 2 {
            offset = Int64(Self.alignUp(Int(offset)))
        }

        // Read trunkPlanes
        var trunkPlanes: [Data] = []
        trunkPlanes.reserveCapacity(meta.trunkPlaneSizes.count)
        for size in meta.trunkPlaneSizes {
            var plane = Data(count: size)
            try plane.withUnsafeMutableBytes {
                _ = try io.readSync(offset: offset, into: $0.baseAddress!, length: size)
            }
            trunkPlanes.append(plane)
            offset += Int64(size)
        }

        // Read mtpPlanes
        var mtpPlanes: [Data] = []
        if meta.hasMtp {
            mtpPlanes.reserveCapacity(meta.mtpPlaneSizes.count)
            for size in meta.mtpPlaneSizes {
                var plane = Data(count: size)
                try plane.withUnsafeMutableBytes {
                    _ = try io.readSync(offset: offset, into: $0.baseAddress!, length: size)
                }
                mtpPlanes.append(plane)
                offset += Int64(size)
            }
        }

        let prefixState = QWEnginePrefixState(
            position: meta.position,
            generated: meta.generated,
            roundRecord: meta.roundRecord,
            recData: recData,
            ringData: ringData,
            trunkHiddenData: trunkHiddenData,
            trunkHiddenValid: meta.trunkHiddenValid,
            stochastic: nil,
            mlaIndexerData: mlaIndexerData
        )

        return (prefixState, meta.nextToken, logits, trunkPlanes, mtpPlanes)
    }

    private func touch(_ key: String) {
        lru.removeAll { $0 == key }
        lru.append(key)
    }

    @discardableResult
    private func evictOldest() -> Bool {
        guard let oldest = lru.first else { return false }
        lru.removeFirst()
        if let file = entryFiles.removeValue(forKey: oldest) {
            try? FileManager.default.removeItem(at: file)
        }
        entries.removeValue(forKey: oldest)
        if let bytes = entryBytes.removeValue(forKey: oldest) {
            chargedBytes -= bytes
        }
        return true
    }
}
