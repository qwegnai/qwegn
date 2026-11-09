import Darwin
import Foundation
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

    public func copyRow(_ id: Int, to dst: UnsafeMutablePointer<Float>) {
        var lo = 0, hi = shards.count - 1
        while lo < hi {
            let mid = (lo + hi + 1) / 2
            if shards[mid].firstRow <= id { lo = mid } else { hi = mid - 1 }
        }
        let s = shards[lo]
        precondition(id >= s.firstRow && id < s.firstRow + s.rows, "PLE row \(id) outside the shard table")
        let rowInShard = id - s.firstRow
        let n = headDim
        switch s.source {
        case .file(let fd, let baseOffset):
            let off = off_t(baseOffset + rowInShard * s.rowBytes)
            let u16 = UnsafeMutableRawPointer(dst).assumingMemoryBound(to: UInt16.self).advanced(by: n)
            let got = pread(fd, u16, s.rowBytes, off)
            precondition(got == s.rowBytes, "PLE short read on \(s.file)")
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
            copyRow(id, to: out.baseAddress!)
            count = n
        }
    }
}

/// PLE N-gram rows read directly from the packed `assets/ngram.bin` file (QNGRAM v1 BF16, v2 Q8_G32
/// or v2 Q4_G32; the rows are sector-packed, 12, 24 or 40 rows to a 4096-byte sector).
public final class FNNgramBinRows: FNNgramRows, @unchecked Sendable {
    public let headDim: Int
    public let entryCount: Int
    public let rowBytes: Int
    public let rowsPerSector: Int
    public let dataOffset: Int
    /// The row format this file carries.
    public let dtype: QWNgramDType
    /// The number of hot rows the writer recorded (0 when unknown).
    public let l1Hot: Int
    private let fd: Int32
    private let mappedPtr: UnsafeRawPointer?
    private let mappedBytes: Int

    public init(url: URL, mapped: Bool = false) throws {
        let opened = open(url.path, O_RDONLY)
        guard opened >= 0 else {
            throw NSError(domain: "FNNgramBinRows", code: Int(errno),
                          userInfo: [NSLocalizedDescriptionKey: "Failed to open ngram.bin at \(url.path)"])
        }
        var header = [UInt8](repeating: 0, count: QWNgramFormat.headerBytes)
        let got = header.withUnsafeMutableBytes {
            pread(opened, $0.baseAddress!, QWNgramFormat.headerBytes, 0)
        }
        guard got == QWNgramFormat.headerBytes else {
            close(opened)
            throw NSError(domain: "FNNgramBinRows", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "Short header in ngram.bin at \(url.path)"])
        }
        let hdr: QWNgramHeader
        let rps: Int
        do {
            hdr = try QWNgramFormat.parse(header)
            rps = try QWNgramFormat.rowsPerSector(rowBytes: hdr.rowBytes)
        } catch {
            close(opened)
            throw error
        }
        // The file must hold the rows the header claims. The Python auditor
        // enforces the same law; a reader and an auditor for one format
        // must not disagree about how much of the file must be there.
        var st = stat()
        guard fstat(opened, &st) == 0 else {
            close(opened)
            throw NSError(domain: "FNNgramBinRows", code: 5,
                          userInfo: [NSLocalizedDescriptionKey:
                            "Cannot stat ngram.bin at \(url.path)"])
        }
        let size = Int(st.st_size)
        let whole = (hdr.entryCount - 1) / rps
        let tailRows = hdr.entryCount - whole * rps
        let minSize = hdr.dataOffset + whole * QWNgramFormat.sectorSize
            + tailRows * hdr.rowBytes
        let maxSize = hdr.dataOffset + (whole + 1) * QWNgramFormat.sectorSize
        guard size >= minSize, size <= maxSize else {
            close(opened)
            throw NSError(domain: "FNNgramBinRows", code: 6,
                          userInfo: [NSLocalizedDescriptionKey:
                            "ngram.bin size \(size) does not hold its \(hdr.entryCount) rows "
                            + "(expected [\(minSize), \(maxSize)]) at \(url.path)"])
        }
        let ptr: UnsafeRawPointer?
        if mapped {
            let result = Darwin.mmap(nil, size, PROT_READ, MAP_PRIVATE, opened, 0)
            guard let result, result != MAP_FAILED else {
                let error = errno
                close(opened)
                throw NSError(domain: "FNNgramBinRows", code: Int(error),
                              userInfo: [NSLocalizedDescriptionKey: "Failed to map ngram.bin at \(url.path)"])
            }
            ptr = UnsafeRawPointer(result)
        } else {
            ptr = nil
        }
        self.dtype = hdr.dtype
        self.l1Hot = hdr.l1Hot
        self.entryCount = hdr.entryCount
        self.headDim = hdr.headDim
        self.rowBytes = hdr.rowBytes
        self.rowsPerSector = rps
        self.dataOffset = hdr.dataOffset
        self.fd = opened
        self.mappedPtr = ptr
        self.mappedBytes = size
    }

    deinit {
        if let mappedPtr {
            Darwin.munmap(UnsafeMutableRawPointer(mutating: mappedPtr), mappedBytes)
        }
        if fd >= 0 { close(fd) }
    }

    public func copyRow(_ id: Int, to dst: UnsafeMutablePointer<Float>) {
        precondition(id >= 0 && id < entryCount, "PLE row \(id) outside ngram.bin [0..<\(entryCount)]")
        let off = off_t(dataOffset + (id / rowsPerSector) * QWNgramFormat.sectorSize
                        + (id % rowsPerSector) * rowBytes)
        if let mappedPtr {
            QWNgramFormat.decodeRow(dtype: dtype, from: mappedPtr.advanced(by: Int(off)),
                                    headDim: headDim, into: dst)
            return
        }
        if dtype == .bf16 {
            let n = headDim
            let u16 = UnsafeMutableRawPointer(dst).assumingMemoryBound(to: UInt16.self).advanced(by: n)
            let got = pread(fd, u16, rowBytes, off)
            precondition(got == rowBytes, "PLE short read on ngram.bin at row \(id)")
            for i in 0..<n {
                dst[i] = Float(bitPattern: UInt32(u16[i]) << 16)
            }
            return
        }
        // The row bytes land in a scratch buffer of their own: the
        // caller's buffer holds the row's values and nothing else.
        withUnsafeTemporaryAllocation(of: UInt8.self, capacity: rowBytes) { scratch in
            let got = pread(fd, scratch.baseAddress!, rowBytes, off)
            precondition(got == rowBytes, "PLE short read on ngram.bin at row \(id)")
            QWNgramFormat.decodeRow(dtype: dtype, from: scratch.baseAddress!,
                                    headDim: headDim, into: dst)
        }
    }

    public func row(_ id: Int) -> [Float] {
        let n = headDim
        return [Float](unsafeUninitializedCapacity: n) { out, count in
            copyRow(id, to: out.baseAddress!)
            count = n
        }
    }
}

/// PLE N-gram rows read on demand from an MLX-Serve `ngram_table.bin` safetensors file
/// (`weight` U32 `[rows, headDim/8]`, `scales` BF16/F16 `[rows, groups]`, `biases` BF16/F16 `[rows, groups]`),
/// using read-only `mmap` (`POSIX_MADV_RANDOM`, `pread` fallback) and SIMD8 4-bit nibble unpacking.
public final class FNNgramMLXRows: FNNgramRows, @unchecked Sendable {
    public let headDim: Int
    public let entryCount: Int
    private let wCols: Int
    private let gCols: Int
    private let groupSize: Int
    private let isF16: Bool
    private let fd: Int32
    private let fileSize: Int
    private let mappedPtr: UnsafeRawPointer?
    private let wOffset: Int
    private let sOffset: Int
    private let bOffset: Int

    public init(url: URL) throws {
        let opened = Darwin.open(url.path, O_RDONLY)
        guard opened >= 0 else {
            throw QWSafeTensorsError.badHeader("Cannot open \(url.path)")
        }
        var st = Darwin.stat()
        guard Darwin.fstat(opened, &st) == 0, st.st_size >= 8 else {
            Darwin.close(opened)
            throw QWSafeTensorsError.badHeader("Cannot stat or short file \(url.path)")
        }
        let fileSize = Int(st.st_size)
        var hdrLenLE: UInt64 = 0
        let gotLen = withUnsafeMutableBytes(of: &hdrLenLE) { raw in
            Darwin.pread(opened, raw.baseAddress!, 8, 0)
        }
        let hdrLen = Int(UInt64(littleEndian: hdrLenLE))
        guard gotLen == 8, hdrLen > 0, hdrLen <= 100_000_000, 8 + hdrLen <= fileSize else {
            Darwin.close(opened)
            throw QWSafeTensorsError.badHeader("Invalid safetensors header length in \(url.lastPathComponent)")
        }
        var headerData = Data(count: hdrLen)
        let gotHdr = headerData.withUnsafeMutableBytes { raw in
            Darwin.pread(opened, raw.baseAddress!, hdrLen, 8)
        }
        guard gotHdr == hdrLen,
              let header = (try? JSONSerialization.jsonObject(with: headerData)) as? [String: Any] else {
            Darwin.close(opened)
            throw QWSafeTensorsError.badHeader("Invalid safetensors JSON header in \(url.lastPathComponent)")
        }
        func parseTensor(_ name: String) throws -> (dtype: QWSafeTensorsDType, shape: [Int], start: Int, end: Int) {
            guard let v = header[name] as? [String: Any],
                  let dtypeRaw = v["dtype"] as? String,
                  let dtype = QWSafeTensorsDType(rawValue: dtypeRaw),
                  let shape = v["shape"] as? [Int],
                  let offsets = v["data_offsets"] as? [Int], offsets.count == 2 else {
                throw QWSafeTensorsError.badHeader("Missing or malformed tensor \(name) in \(url.lastPathComponent)")
            }
            return (dtype, shape, offsets[0], offsets[1])
        }
        let wInfo: (dtype: QWSafeTensorsDType, shape: [Int], start: Int, end: Int)
        let sInfo: (dtype: QWSafeTensorsDType, shape: [Int], start: Int, end: Int)
        let bInfo: (dtype: QWSafeTensorsDType, shape: [Int], start: Int, end: Int)
        do {
            wInfo = try parseTensor("weight")
            sInfo = try parseTensor("scales")
            bInfo = try parseTensor("biases")
        } catch {
            Darwin.close(opened)
            throw error
        }
        let dataBase = 8 + hdrLen
        guard wInfo.dtype == .u32, wInfo.shape.count == 2,
              (sInfo.dtype == .bf16 || sInfo.dtype == .f16), sInfo.shape.count == 2,
              (bInfo.dtype == .bf16 || bInfo.dtype == .f16), bInfo.shape.count == 2,
              wInfo.shape[0] == sInfo.shape[0], wInfo.shape[0] == bInfo.shape[0],
              sInfo.shape[1] == bInfo.shape[1],
              dataBase + wInfo.end <= fileSize,
              dataBase + sInfo.end <= fileSize,
              dataBase + bInfo.end <= fileSize else {
            Darwin.close(opened)
            throw QWSafeTensorsError.badHeader("Unexpected tensor shapes/dtypes in \(url.lastPathComponent)")
        }
        self.entryCount = wInfo.shape[0]
        self.wCols = wInfo.shape[1]
        self.gCols = sInfo.shape[1]
        self.headDim = wCols * 8
        self.groupSize = (wCols * 8) / gCols
        self.isF16 = (sInfo.dtype == .f16)
        self.fd = opened
        self.fileSize = fileSize
        self.wOffset = dataBase + wInfo.start
        self.sOffset = dataBase + sInfo.start
        self.bOffset = dataBase + bInfo.start
        let mapped = Darwin.mmap(nil, fileSize, PROT_READ, MAP_PRIVATE, opened, 0)
        if let mapped, mapped != MAP_FAILED {
            _ = Darwin.posix_madvise(mapped, fileSize, POSIX_MADV_RANDOM)
            self.mappedPtr = UnsafeRawPointer(mapped)
        } else {
            self.mappedPtr = nil
        }
    }

    deinit {
        if let mappedPtr {
            Darwin.munmap(UnsafeMutableRawPointer(mutating: mappedPtr), fileSize)
        }
        if fd >= 0 { Darwin.close(fd) }
    }

    @inline(__always)
    private static func decodeGroups(wPtr: UnsafeRawPointer, sPtr: UnsafeRawPointer, bPtr: UnsafeRawPointer,
                                     gCols: Int, wordsPerGroup: Int, groupSize: Int, isF16: Bool,
                                     into dst: UnsafeMutablePointer<Float>) {
        let shifts = SIMD8<UInt32>(0, 4, 8, 12, 16, 20, 24, 28)
        let mask = SIMD8<UInt32>(repeating: 0xF)
        for g in 0..<gCols {
            let sBits = sPtr.loadUnaligned(fromByteOffset: g * 2, as: UInt16.self).littleEndian
            let bBits = bPtr.loadUnaligned(fromByteOffset: g * 2, as: UInt16.self).littleEndian
            let scale = isF16 ? Float(Float16(bitPattern: sBits)) : Float(bitPattern: UInt32(sBits) << 16)
            let bias = isF16 ? Float(Float16(bitPattern: bBits)) : Float(bitPattern: UInt32(bBits) << 16)
            let scaleVec = SIMD8<Float>(repeating: scale)
            let biasVec = SIMD8<Float>(repeating: bias)
            let groupWordBase = g * wordsPerGroup
            let groupDstBase = g * groupSize
            for u in 0..<wordsPerGroup {
                let word = wPtr.loadUnaligned(fromByteOffset: (groupWordBase + u) * 4, as: UInt32.self).littleEndian
                let nibbles = (SIMD8<UInt32>(repeating: word) &>> shifts) & mask
                let outVec = SIMD8<Float>(nibbles) * scaleVec + biasVec
                let baseIdx = groupDstBase + u * 8
                dst[baseIdx + 0] = outVec[0]
                dst[baseIdx + 1] = outVec[1]
                dst[baseIdx + 2] = outVec[2]
                dst[baseIdx + 3] = outVec[3]
                dst[baseIdx + 4] = outVec[4]
                dst[baseIdx + 5] = outVec[5]
                dst[baseIdx + 6] = outVec[6]
                dst[baseIdx + 7] = outVec[7]
            }
        }
    }

    public func copyRow(_ id: Int, to dst: UnsafeMutablePointer<Float>) {
        precondition(id >= 0 && id < entryCount, "PLE row \(id) outside ngram_table.bin [0..<\(entryCount)]")
        let wRowBytes = wCols * 4
        let sbRowBytes = gCols * 2
        let wByteOff = wOffset + id * wRowBytes
        let sByteOff = sOffset + id * sbRowBytes
        let bByteOff = bOffset + id * sbRowBytes
        let isF16 = self.isF16
        let gCols = self.gCols
        let groupSize = self.groupSize
        let wordsPerGroup = groupSize / 8
        if let mappedPtr {
            Self.decodeGroups(wPtr: mappedPtr.advanced(by: wByteOff),
                              sPtr: mappedPtr.advanced(by: sByteOff),
                              bPtr: mappedPtr.advanced(by: bByteOff),
                              gCols: gCols, wordsPerGroup: wordsPerGroup, groupSize: groupSize, isF16: isF16,
                              into: dst)
            return
        }
        let wOff = off_t(wByteOff)
        let sOff = off_t(sByteOff)
        let bOff = off_t(bByteOff)
        withUnsafeTemporaryAllocation(of: UInt8.self, capacity: wRowBytes + 2 * sbRowBytes) { scratch in
            let wPtr = UnsafeRawPointer(scratch.baseAddress!)
            let sPtr = UnsafeRawPointer(scratch.baseAddress!.advanced(by: wRowBytes))
            let bPtr = UnsafeRawPointer(scratch.baseAddress!.advanced(by: wRowBytes + sbRowBytes))
            let gotW = Darwin.pread(fd, UnsafeMutableRawPointer(mutating: wPtr), wRowBytes, wOff)
            let gotS = Darwin.pread(fd, UnsafeMutableRawPointer(mutating: sPtr), sbRowBytes, sOff)
            let gotB = Darwin.pread(fd, UnsafeMutableRawPointer(mutating: bPtr), sbRowBytes, bOff)
            precondition(gotW == wRowBytes && gotS == sbRowBytes && gotB == sbRowBytes,
                         "PLE short pread on ngram_table.bin at row \(id)")
            Self.decodeGroups(wPtr: wPtr, sPtr: sPtr, bPtr: bPtr,
                              gCols: gCols, wordsPerGroup: wordsPerGroup, groupSize: groupSize, isF16: isF16,
                              into: dst)
        }
    }

    public func row(_ id: Int) -> [Float] {
        let n = headDim
        return [Float](unsafeUninitializedCapacity: n) { out, count in
            copyRow(id, to: out.baseAddress!)
            count = n
        }
    }
}

/// Resolves the PLE N-gram row table from a model's `assets/` directory, preferring self-contained
/// `assets/ngram.bin`, then `ngram_table.bin`, then `assets/ple_ngram_locations.json` + the HF snapshot shards.
public enum FNNgramRowsLoader {
    public static func open(assetsURL: URL) throws -> any FNNgramRows {
        let ngramBin = assetsURL.appendingPathComponent("ngram.bin")
        if FileManager.default.fileExists(atPath: ngramBin.path) {
            return try FNNgramBinRows(url: ngramBin)
        }
        let ngramTableBin = assetsURL.appendingPathComponent("ngram_table.bin")
        if FileManager.default.fileExists(atPath: ngramTableBin.path) {
            return try FNNgramMLXRows(url: ngramTableBin)
        }
        let locations = assetsURL.appendingPathComponent("ple_ngram_locations.json")
        let snapshot = snapshotDirectory()
        if let snapshot {
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

    /// The snapshot the PLE shards live in: the override first, then the HF
    /// cache.
    static func snapshotDirectory() -> URL? {
        if let envSnap = ProcessInfo.processInfo.environment["FN_SNAPSHOT_DIR"], !envSnap.isEmpty {
            return URL(fileURLWithPath: envSnap)
        }
        if let envAssets = ProcessInfo.processInfo.environment["QW_PLE_ASSETS_DIR"], !envAssets.isEmpty {
            return URL(fileURLWithPath: envAssets)
        }
        let hubRoot = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
        guard let models = try? FileManager.default.contentsOfDirectory(atPath: hubRoot.path) else { return nil }
        for m in models where m.localizedCaseInsensitiveContains("Qwen3.8-Flash-Next") {
            let snapsRoot = hubRoot.appendingPathComponent(m).appendingPathComponent("snapshots")
            if let items = try? FileManager.default.contentsOfDirectory(atPath: snapsRoot.path),
               let first = items.sorted().first(where: { !$0.hasPrefix(".") }) {
                return snapsRoot.appendingPathComponent(first)
            }
        }
        return nil
    }

    /// The checkpoint's tensor index, if the snapshot has one.
    static func indexJSON(in snapshot: URL) -> URL? {
        let items = (try? FileManager.default.contentsOfDirectory(atPath: snapshot.path)) ?? []
        return items.sorted().first(where: { $0.hasSuffix(".index.json") })
            .map { snapshot.appendingPathComponent($0) }
    }
}
