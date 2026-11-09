// Zero-copy safetensors reader: maps a shard read-only and indexes its
// tensors. A single-file checkpoint and a sharded one (index.json + shards)
// are both a QWSafeTensorsContainer. No tensor is ever copied into the heap.

import Darwin
import Foundation

public enum QWSafeTensorsError: Error, Equatable, CustomStringConvertible {
    case badHeader(String)
    case missingTensor(String)
    case unsupportedDType(String)

    public var description: String {
        switch self {
        case .badHeader(let detail): return "bad safetensors header: \(detail)"
        case .missingTensor(let name): return "safetensors has no tensor '\(name)'"
        case .unsupportedDType(let raw): return "safetensors dtype '\(raw)' is not supported"
        }
    }
}

/// The safetensors dtypes, in the spec's order. `bitSize` is in BITS: `F4`
/// and `F6_*` are sub-byte, so a byte-per-element model cannot express them.
public enum QWSafeTensorsDType: String, Sendable {
    case bool = "BOOL"
    case f4 = "F4"
    case f6E2M3 = "F6_E2M3"
    case f6E3M2 = "F6_E3M2"
    case u8 = "U8"
    case i8 = "I8"
    case f8E5M2 = "F8_E5M2"
    case f8E4M3 = "F8_E4M3"
    case f8E8M0 = "F8_E8M0"
    case f8E4M3FNUZ = "F8_E4M3FNUZ"
    case f8E5M2FNUZ = "F8_E5M2FNUZ"
    case i16 = "I16"
    case u16 = "U16"
    case f16 = "F16"
    case bf16 = "BF16"
    case i32 = "I32"
    case u32 = "U32"
    case f32 = "F32"
    case c64 = "C64"
    case f64 = "F64"
    case i64 = "I64"
    case u64 = "U64"

    /// The size of one element, in BITS.
    public var bitSize: Int {
        switch self {
        case .f4: return 4
        case .f6E2M3, .f6E3M2: return 6
        case .bool, .u8, .i8, .f8E5M2, .f8E4M3, .f8E8M0, .f8E4M3FNUZ, .f8E5M2FNUZ:
            return 8
        case .i16, .u16, .f16, .bf16: return 16
        case .i32, .u32, .f32: return 32
        case .c64, .f64, .i64, .u64: return 64
        }
    }
}

public struct QWSafeTensorsTensor: Sendable, Hashable {
    public let name: String
    public let dtype: QWSafeTensorsDType
    public let shape: [Int]
    /// Offset within the shard's **data** region.
    public let byteOffset: Int
    public let byteLength: Int

    /// Elements per row for a 2-D tensor; `nil` for other ranks.
    public var rows: Int? { shape.count == 2 ? shape[0] : nil }
    public var cols: Int? { shape.count == 2 ? shape[1] : nil }

    /// The number of elements, or `nil` if the shape overflows.
    public var elementCount: Int? {
        shape.reduce(1) { acc, dim -> Int? in
            guard let acc else { return nil }
            let (m, over) = acc.multipliedReportingOverflow(by: dim)
            return over ? nil : m
        }
    }
}

/// One shard's header plus its mapped bytes.
public final class QWSafeTensorsShard: @unchecked Sendable {
    public let url: URL
    /// Offset of the data region within the file (`8 + header length`).
    public let dataOffset: Int
    public let tensors: [String: QWSafeTensorsTensor]
    let data: Data          // holds the mmap alive

    init(url: URL, data: Data, dataOffset: Int, tensors: [String: QWSafeTensorsTensor]) {
        self.url = url
        self.data = data
        self.dataOffset = dataOffset
        self.tensors = tensors
    }

    public func adviseDontNeed() {
        data.withUnsafeBytes { raw in
            guard let base = raw.baseAddress, raw.count > 0 else { return }
            let pageSize = Int(sysconf(_SC_PAGESIZE))
            let addr = Int(bitPattern: base)
            let alignedAddr = addr & ~(pageSize - 1)
            let alignedLen = raw.count + (addr - alignedAddr)
            if let ptr = UnsafeMutableRawPointer(bitPattern: alignedAddr) {
                _ = posix_madvise(ptr, alignedLen, POSIX_MADV_DONTNEED)
            }
        }
    }
}

/// A read-only view over one safetensors file or a sharded checkpoint.
///
/// The payload is mapped read-only and the header is parsed once.
public final class QWSafeTensorsContainer: @unchecked Sendable {
    public let url: URL
    public let shards: [URL: QWSafeTensorsShard]
    /// The file's own tensor names -> (shard, tensor).
    private let index: [String: (QWSafeTensorsShard, QWSafeTensorsTensor)]
    /// Canonical names the file does not use -> the tensor they name.
    private let aliases: [String: (QWSafeTensorsShard, QWSafeTensorsTensor)]
    /// The file's tensors. The canonical aliases are deliberately NOT here:
    /// a consumer that loads every weight must not load one twice.
    public let tensors: [String: QWSafeTensorsTensor]

    public convenience init(url: URL) throws {
        try self.init(url: url, onlyTensors: nil)
    }

    /// A container over only the named tensors. A reader that needs 33 of a
    /// checkpoint's 131 shards should not map the other 98, so a partially
    /// downloaded checkpoint stays usable.
    public init(url: URL, onlyTensors wanted: Set<String>?) throws {
        self.url = url
        var index: [String: (QWSafeTensorsShard, QWSafeTensorsTensor)] = [:]
        var shards: [URL: QWSafeTensorsShard] = [:]

        if url.lastPathComponent.hasSuffix(".index.json") {
            let base = url.deletingLastPathComponent()
            let raw = try Data(contentsOf: url)
            let obj = try JSONSerialization.jsonObject(with: raw) as? [String: Any] ?? [:]
            guard let weightMap = obj["weight_map"] as? [String: String] else {
                throw QWSafeTensorsError.badHeader("index.json has no weight_map")
            }
            for (tensorName, fileName) in weightMap {
                if let wanted, !wanted.contains(tensorName) { continue }
                let shardURL = base.appendingPathComponent(fileName).standardizedFileURL
                let shard: QWSafeTensorsShard
                if let existing = shards[shardURL] {
                    shard = existing
                } else {
                    shard = try QWSafeTensorsShard.open(shardURL)
                    shards[shardURL] = shard
                }
                guard let t = shard.tensors[tensorName] else {
                    throw QWSafeTensorsError.badHeader(
                        "index.json names '\(tensorName)' but \(fileName) does not contain it")
                }
                index[tensorName] = (shard, t)
            }
            if let wanted {
                for name in wanted where index[name] == nil {
                    throw QWSafeTensorsError.missingTensor(name)
                }
            }
        } else {
            let shard = try QWSafeTensorsShard.open(url)
            shards[url.standardizedFileURL] = shard
            for (name, t) in shard.tensors { index[name] = (shard, t) }
        }

        self.shards = shards
        self.index = index
        self.aliases = QWSafeTensorsContainer.buildAliases(index)
        self.tensors = index.mapValues { $0.1 }
    }

    /// Bridges the `model.language_model.` / `language_model.model.` / `model.` prefixes:
    /// HF and MLX write different prefix spellings for the same tensor across checkpoints.
    /// A canonical name claimed by more than one source is NOT aliased.
    private static func buildAliases(
        _ index: [String: (QWSafeTensorsShard, QWSafeTensorsTensor)]
    ) -> [String: (QWSafeTensorsShard, QWSafeTensorsTensor)] {
        let prefixes = ["model.language_model.", "language_model.model.", "model."]
        var candidates: [String: Set<String>] = [:]
        for name in index.keys {
            var matched = false
            for prefix in prefixes where name.hasPrefix(prefix) {
                matched = true
                let suffix = String(name.dropFirst(prefix.count))
                for targetPrefix in prefixes {
                    let canonical = targetPrefix + suffix
                    guard canonical != name else { continue }
                    candidates[canonical, default: []].insert(name)
                }
                break
            }
            if !matched && name.hasPrefix("language_model.") {
                let canonical = String(name.dropFirst("language_model.".count))
                if canonical != name {
                    candidates[canonical, default: []].insert(name)
                }
            }
        }
        var aliases: [String: (QWSafeTensorsShard, QWSafeTensorsTensor)] = [:]
        for (canonical, sources) in candidates {
            guard index[canonical] == nil, sources.count == 1, let source = sources.first else { continue }
            if let value = index[source] { aliases[canonical] = value }
        }
        return aliases
    }

    /// The tensor under `name`, whether the file spells it that way or the
    /// canonical bridge does. Its `name` is always the file's own name.
    public func tensor(_ name: String) -> QWSafeTensorsTensor? {
        locate(name)?.1
    }

    /// The shard a tensor lives in — the byte range below is within it.
    public func shardURL(of name: String) -> URL? {
        locate(name)?.0.url
    }

    private func locate(_ name: String) -> (QWSafeTensorsShard, QWSafeTensorsTensor)? {
        if let hit = index[name] { return hit }
        return aliases[name]
    }

    /// The tensor's byte range within its shard **file** (see `shardURL(of:)`).
    public func byteRange(of name: String) -> Range<Int>? {
        guard let (shard, t) = locate(name) else { return nil }
        let start = shard.dataOffset + t.byteOffset
        return start..<(start + t.byteLength)
    }

    /// Reads a tensor's bytes without copying them into the heap.
    ///
    /// The pointer is valid only inside `body`; it must not outlive the call.
    public func withBytes<T>(of name: String, _ body: (UnsafeRawBufferPointer) throws -> T) throws -> T {
        guard let (shard, t) = locate(name) else { throw QWSafeTensorsError.missingTensor(name) }
        return try shard.data.withUnsafeBytes { raw in
            let start = shard.dataOffset + t.byteOffset
            guard start >= 0, start + t.byteLength <= raw.count else {
                throw QWSafeTensorsError.badHeader(
                    "tensor \(name) runs past the end of \(shard.url.lastPathComponent)")
            }
            return try body(UnsafeRawBufferPointer(rebasing: raw[start..<start + t.byteLength]))
        }
    }

    /// Passes the whole mapped shard buffer and the tensor's `[start..<end]` byte range
    /// inside it so callers can page-align `bytesNoCopy` Metal mappings without copying.
    public func withShardRange<T>(of name: String,
                                  _ body: (QWSafeTensorsShard, UnsafeRawBufferPointer, Range<Int>) throws -> T) throws -> T {
        guard let (shard, t) = locate(name) else { throw QWSafeTensorsError.missingTensor(name) }
        return try shard.data.withUnsafeBytes { raw in
            let start = shard.dataOffset + t.byteOffset
            guard start >= 0, start + t.byteLength <= raw.count else {
                throw QWSafeTensorsError.badHeader(
                    "tensor \(name) runs past the end of \(shard.url.lastPathComponent)")
            }
            return try body(shard, raw, start..<(start + t.byteLength))
        }
    }

    /// A reader bound to one tensor's bytes: the shard and the range are
    /// resolved once, so a hot path pays no lookup and no range check per
    /// read. The pointer handed to `body` is valid only inside it.
    public func reader(of name: String) throws -> QWSafeTensorsReader {
        guard let (shard, t) = locate(name) else { throw QWSafeTensorsError.missingTensor(name) }
        let start = shard.dataOffset + t.byteOffset
        guard start >= 0, start + t.byteLength <= shard.data.count else {
            throw QWSafeTensorsError.badHeader(
                "tensor \(name) runs past the end of \(shard.url.lastPathComponent)")
        }
        return QWSafeTensorsReader(shard: shard, tensor: t)
    }

    public func adviseDontNeed() {
        for shard in shards.values {
            shard.adviseDontNeed()
        }
    }

    /// Advises the kernel that the interior full pages of `name` will not be read again,
    /// without touching boundary pages shared with adjacent tensors in the same shard.
    public func adviseDontNeed(of name: String) {
        guard let (shard, t) = locate(name) else { return }
        shard.data.withUnsafeBytes { raw in
            guard let base = raw.baseAddress, raw.count > 0 else { return }
            let pageSize = Int(sysconf(_SC_PAGESIZE))
            guard pageSize > 0, Int(bitPattern: base) % pageSize == 0 else { return }
            let start = shard.dataOffset + t.byteOffset
            let end = min(raw.count, start + t.byteLength)
            let alignedStart = ((start + pageSize - 1) / pageSize) * pageSize
            let alignedEnd = (end / pageSize) * pageSize
            guard alignedEnd > alignedStart else { return }
            let ptr = UnsafeMutableRawPointer(mutating: base.advanced(by: alignedStart))
            _ = posix_madvise(ptr, alignedEnd - alignedStart, POSIX_MADV_DONTNEED)
        }
    }

    /// True when the checkpoint stores MLX affine-quantized (`u32` + `.scales` + `.biases`) weights.
    public var isMLXCheckpoint: Bool {
        tensor("model.language_model.embed_tokens.scales") != nil
            || tensor("embed_tokens.scales") != nil
    }

    /// Reads and dequantizes a tensor (`bf16`, `f16`, `f32`, or MLX affine `u32` with `.scales`/`.biases`) into `[Float]`.
    public func readFloats(_ name: String, expectedElementCount: Int? = nil) throws -> [Float] {
        let r = try reader(of: name)
        let info = r.tensorInfo
        let count = info.elementCount ?? info.shape.reduce(1, *)
        if info.dtype == .u32 {
            let basePrefix = name.hasSuffix(".weight") ? String(name.dropLast(".weight".count)) : name
            let sReader = try reader(of: basePrefix + ".scales")
            let bReader = try reader(of: basePrefix + ".biases")
            let sInfo = sReader.tensorInfo
            let n = info.shape.first ?? 1
            let wCols = info.shape.count == 2 ? info.shape[1] : count / max(1, n)
            let sElems = sInfo.elementCount ?? sInfo.shape.reduce(1, *)
            let gCols = sInfo.shape.count == 2 ? sInfo.shape[1] : sElems / max(1, n)
            guard n > 0, wCols > 0, gCols > 0 else {
                throw QWSafeTensorsError.unsupportedDType("u32 invalid shape \(name)")
            }
            let bits: Int
            let groupSize: Int
            let k: Int
            if let expected = expectedElementCount {
                k = expected / n
                guard n * k == expected, k % gCols == 0 else {
                    throw QWSafeTensorsError.badHeader("invalid MLX affine layout for \(name)")
                }
                groupSize = k / gCols
                bits = (wCols * 32) / k
            } else {
                bits = (wCols * 32) % (gCols * 64) == 0 ? (wCols * 32) / (gCols * 64) : (wCols * 32) / (gCols * 32)
                groupSize = (wCols * 32) / max(1, gCols * bits)
                k = gCols * groupSize
            }
            guard groupSize > 0, k == gCols * groupSize, QWBF16.isSupportedMLXAffineBits(bits) else {
                throw QWSafeTensorsError.unsupportedDType("u32 unsupported MLX affine bits=\(bits) \(name)")
            }
            if sInfo.dtype == .f16 || sInfo.dtype == .bf16 {
                return r.withBytes { wRaw in
                    sReader.withBytes { sRaw in
                        bReader.withBytes { bRaw in
                            let wWords = wRaw.bindMemory(to: UInt32.self)
                            guard let wPtr = wWords.baseAddress else { return [] }
                            let sU16 = sRaw.bindMemory(to: UInt16.self)
                            let bU16 = bRaw.bindMemory(to: UInt16.self)
                            let isF16 = sInfo.dtype == .f16
                            return [Float](unsafeUninitializedCapacity: n * k) { dst, initialized in
                                nonisolated(unsafe) let dPtr = dst.baseAddress!
                                nonisolated(unsafe) let wBase = wPtr
                                nonisolated(unsafe) let sBase = sU16.baseAddress!
                                nonisolated(unsafe) let bBase = bU16.baseAddress!
                                let workers = n >= 64 ? min(16, n) : 1
                                let perWorker = (n + workers - 1) / workers
                                DispatchQueue.concurrentPerform(iterations: workers) { w in
                                    let startR = w * perWorker
                                    let endR = min(n, startR + perWorker)
                                    guard startR < endR else { return }
                                    for row in startR..<endR {
                                        let rowWords = wBase.advanced(by: row * wCols)
                                        let sbRowBase = row * gCols
                                        let dstRowBase = row * k
                                        for g in 0..<gCols {
                                            let sBits = UInt16(littleEndian: sBase[sbRowBase + g])
                                            let bBits = UInt16(littleEndian: bBase[sbRowBase + g])
                                            let scale = isF16 ? Float(Float16(bitPattern: sBits)) : Float(bitPattern: UInt32(sBits) << 16)
                                            let bias = isF16 ? Float(Float16(bitPattern: bBits)) : Float(bitPattern: UInt32(bBits) << 16)
                                            let dstGroupBase = dstRowBase + g * groupSize
                                            if bits == 4 && groupSize % 8 == 0 {
                                                let wordBase = (g * groupSize) / 8
                                                let wordsInGroup = groupSize / 8
                                                for wIdx in 0..<wordsInGroup {
                                                    var word = UInt32(littleEndian: rowWords[wordBase + wIdx])
                                                    let baseOut = dstGroupBase + wIdx * 8
                                                    for nib in 0..<8 {
                                                        let q = Float(word & 0xF)
                                                        word >>= 4
                                                        dPtr[baseOut + nib] = q * scale + bias
                                                    }
                                                }
                                            } else if bits == 8 && groupSize % 4 == 0 {
                                                let wordBase = (g * groupSize) / 4
                                                let wordsInGroup = groupSize / 4
                                                for wIdx in 0..<wordsInGroup {
                                                    var word = UInt32(littleEndian: rowWords[wordBase + wIdx])
                                                    let baseOut = dstGroupBase + wIdx * 4
                                                    for byteIdx in 0..<4 {
                                                        let q = Float(word & 0xFF)
                                                        word >>= 8
                                                        dPtr[baseOut + byteIdx] = q * scale + bias
                                                    }
                                                }
                                            } else {
                                                for j in 0..<groupSize {
                                                    let col = g * groupSize + j
                                                    let q = QWBF16.unpackMLXAffineQ(rowWords: rowWords, col: col, bits: bits)
                                                    dPtr[dstGroupBase + j] = q * scale + bias
                                                }
                                            }
                                        }
                                    }
                                }
                                initialized = n * k
                            }
                        }
                    }
                }
            }
            let scales = try readFloats(basePrefix + ".scales", expectedElementCount: sElems)
            let biases = try readFloats(basePrefix + ".biases", expectedElementCount: sElems)
            return r.withBytes { wRaw in
                let wWords = wRaw.bindMemory(to: UInt32.self)
                guard let wPtr = wWords.baseAddress else { return [] }
                var out = [Float](repeating: 0, count: n * k)
                for row in 0..<n {
                    let rowWords = wPtr.advanced(by: row * wCols)
                    let sbRowBase = row * gCols
                    let dstRowBase = row * k
                    for c in 0..<k {
                        let g = c / groupSize
                        let q = QWBF16.unpackMLXAffineQ(rowWords: rowWords, col: c, bits: bits)
                        out[dstRowBase + c] = q * scales[sbRowBase + g] + biases[sbRowBase + g]
                    }
                }
                return out
            }
        }
        if let expected = expectedElementCount, count != expected {
            throw QWSafeTensorsError.badHeader("shape \(count) != \(expected) for \(name)")
        }
        return try r.withBytes { raw in
            var out = [Float](repeating: 0, count: count)
            switch info.dtype {
            case .bf16:
                let u16 = raw.bindMemory(to: UInt16.self)
                for i in 0..<count { out[i] = Float(bitPattern: UInt32(UInt16(littleEndian: u16[i])) << 16) }
            case .f16:
                let u16 = raw.bindMemory(to: UInt16.self)
                for i in 0..<count { out[i] = Float(Float16(bitPattern: UInt16(littleEndian: u16[i]))) }
            case .f32:
                let u32 = raw.bindMemory(to: UInt32.self)
                for i in 0..<count { out[i] = Float(bitPattern: UInt32(littleEndian: u32[i])) }
            default:
                throw QWSafeTensorsError.unsupportedDType(info.dtype.rawValue)
            }
            return out
        }
    }
}

/// Reads one tensor's bytes without copying them into the heap. The shard
/// and the byte range are resolved once, and the payload stays mapped: a
/// read is a memory read, not a syscall.
public final class QWSafeTensorsReader: @unchecked Sendable {
    private let shard: QWSafeTensorsShard
    private let tensor: QWSafeTensorsTensor

    init(shard: QWSafeTensorsShard, tensor: QWSafeTensorsTensor) {
        self.shard = shard
        self.tensor = tensor
    }

    /// The tensor this reader is bound to.
    public var tensorInfo: QWSafeTensorsTensor { tensor }

    /// The file the tensor lives in.
    public var shardURL: URL { shard.url }

    /// The tensor's bytes. The pointer is valid only inside `body`; it must
    /// not outlive the call.
    public func withBytes<T>(_ body: (UnsafeRawBufferPointer) throws -> T) rethrows -> T {
        try shard.data.withUnsafeBytes { raw in
            let start = shard.dataOffset + tensor.byteOffset
            return try body(UnsafeRawBufferPointer(rebasing: raw[start..<start + tensor.byteLength]))
        }
    }
}

extension QWSafeTensorsShard {
    /// The spec's own limit on a header, in bytes.
    static let maxHeaderBytes = 100_000_000

    struct Entry {
        let name: String
        let dtype: QWSafeTensorsDType
        let shape: [Int]
        let start: Int
        let end: Int
    }

    static func open(_ url: URL) throws -> QWSafeTensorsShard {
        let fd = Darwin.open(url.path, O_RDONLY)
        guard fd >= 0 else {
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno),
                          userInfo: [NSFilePathErrorKey: url.path])
        }
        defer { Darwin.close(fd) }
        var st = Darwin.stat()
        guard Darwin.fstat(fd, &st) == 0 else {
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno),
                          userInfo: [NSFilePathErrorKey: url.path])
        }
        let fileSize = Int(st.st_size)
        guard fileSize >= 8 else { throw QWSafeTensorsError.badHeader("file too short") }
        let rawMap = Darwin.mmap(nil, fileSize, PROT_READ, MAP_SHARED, fd, 0)
        guard let mappedPtr = rawMap, mappedPtr != MAP_FAILED else {
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno),
                          userInfo: [NSFilePathErrorKey: url.path])
        }
        let data = Data(bytesNoCopy: mappedPtr, count: fileSize, deallocator: .custom { ptr, len in
            if len > 0 { _ = Darwin.munmap(ptr, len) }
        })
        let headerLength = data.prefix(8).withUnsafeBytes { raw -> UInt64 in
            UInt64(littleEndian: raw.loadUnaligned(as: UInt64.self))
        }
        // The header length comes from the file, so it is untrusted: every
        // conversion of it must be checked, or a corrupt file traps the
        // process instead of throwing.
        guard headerLength <= UInt64(maxHeaderBytes) else {
            throw QWSafeTensorsError.badHeader("header is larger than the \(maxHeaderBytes)-byte limit")
        }
        guard let headerLength = Int(exactly: headerLength) else {
            throw QWSafeTensorsError.badHeader("header length does not fit")
        }
        let headerEnd = 8 + headerLength
        guard headerEnd <= data.count else { throw QWSafeTensorsError.badHeader("header length past EOF") }

        let headerData = data.subdata(in: 8..<headerEnd)
        let parsed: Any
        do {
            parsed = try JSONSerialization.jsonObject(with: headerData)
        } catch {
            throw QWSafeTensorsError.badHeader("header is not valid JSON")
        }
        guard let header = parsed as? [String: Any] else {
            throw QWSafeTensorsError.badHeader("header is not a JSON object")
        }

        var entries: [Entry] = []
        for (name, value) in header {
            guard name != "__metadata__" else { continue }
            guard let v = value as? [String: Any] else {
                throw QWSafeTensorsError.badHeader("malformed tensor entry \(name)")
            }
            guard let dtypeRaw = v["dtype"] as? String else {
                throw QWSafeTensorsError.badHeader("tensor \(name) has no dtype")
            }
            guard let dtype = QWSafeTensorsDType(rawValue: dtypeRaw) else {
                throw QWSafeTensorsError.unsupportedDType(dtypeRaw)
            }
            guard let shape = v["shape"] as? [Int] else {
                throw QWSafeTensorsError.badHeader("tensor \(name) has no shape")
            }
            guard let offsets = v["data_offsets"] as? [Int], offsets.count == 2 else {
                throw QWSafeTensorsError.badHeader("tensor \(name) has no data_offsets")
            }
            entries.append(Entry(name: name, dtype: dtype, shape: shape,
                                 start: offsets[0], end: offsets[1]))
        }

        // The tensors must tile the data region: ordered, contiguous, from
        // zero, and ending exactly at the end of the file. A header that
        // overlaps or leaves a hole is a silent misread waiting to happen.
        entries.sort { ($0.start, $0.name) < ($1.start, $1.name) }
        var running = 0
        for e in entries {
            guard e.start >= 0, e.end >= e.start else {
                throw QWSafeTensorsError.badHeader("tensor \(e.name) has a negative or inverted range")
            }
            guard e.start == running else {
                throw QWSafeTensorsError.badHeader(
                    "tensor \(e.name) starts at \(e.start) but the previous tensor ends at \(running)")
            }
            var elements = 1
            for dim in e.shape {
                guard dim >= 0 else {
                    throw QWSafeTensorsError.badHeader("tensor \(e.name) has a negative dimension")
                }
                let (m, over) = elements.multipliedReportingOverflow(by: dim)
                guard !over else {
                    throw QWSafeTensorsError.badHeader("tensor \(e.name) has a shape that overflows")
                }
                elements = m
            }
            let (bits, over) = elements.multipliedReportingOverflow(by: e.dtype.bitSize)
            guard !over else {
                throw QWSafeTensorsError.badHeader("tensor \(e.name) overflows its byte size")
            }
            guard bits % 8 == 0 else {
                throw QWSafeTensorsError.badHeader("tensor \(e.name) is not byte-aligned")
            }
            let size = bits / 8
            guard e.end - e.start == size else {
                throw QWSafeTensorsError.badHeader(
                    "tensor \(e.name) holds \(e.end - e.start) bytes but its shape and dtype need \(size)")
            }
            running = e.end
        }
        let dataRegion = data.count - headerEnd
        guard running == dataRegion else {
            throw QWSafeTensorsError.badHeader(
                "the tensors cover \(running) bytes but the data region holds \(dataRegion)")
        }

        var tensors: [String: QWSafeTensorsTensor] = [:]
        for e in entries {
            tensors[e.name] = QWSafeTensorsTensor(name: e.name, dtype: e.dtype, shape: e.shape,
                                                   byteOffset: e.start, byteLength: e.end - e.start)
        }
        return QWSafeTensorsShard(url: url, data: data, dataOffset: headerEnd, tensors: tensors)
    }
}
