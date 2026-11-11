// QWCheckpointTensorLoader.swift — Unified checkpoint-to-GPU tensor loader
// and device embedding table in QWLoad. Owns prefix-agnostic tensor lookup,
// zero-copy PlaneIO mapping, MLX affine repacking, and HQ4 bank materialization.

import Darwin
import Foundation
import Metal
import QWAlloc
import QWCore
import QWGEMM
import QWHAL
import QWQuant

public enum QWCheckpointLoaderError: Error, Equatable, Sendable {
    case notHQ4(String)
}

/// Device embedding table backed by native MLX affine blocks or a raw BF16 plane.
public struct QWDeviceEmbedTable: Sendable {
    public enum Storage: Sendable {
        case nativeAffine(blocks: QWTensorValue, bits: Int, group: Int)
        case bf16(QWTensorValue)
    }

    public let storage: Storage
    public let vocab: Int, hidden: Int, hc: Int
    public let containerNote: String

    public init(storage: Storage, vocab: Int, hidden: Int, hc: Int, containerNote: String) {
        self.storage = storage
        self.vocab = vocab
        self.hidden = hidden
        self.hc = hc
        self.containerNote = containerNote
    }

    public static func load(arena: QWArena, pack: QWQuant.Pack, name: String,
                            vocab: Int, hidden: Int, hc: Int) throws -> QWDeviceEmbedTable {
        let e = try pack.entry(name)
        guard e.isRawBf16, let off = e.rawOff else {
            throw QWCheckpointLoaderError.notHQ4(name)
        }
        let bytes = vocab * hidden * 2
        // Copy raw bf16 bytes into the wired weights region to avoid unwired-page faults.
        let t = try arena.weightsRegion().allocate(QWTensorShape([vocab * hidden], .f16))
        _ = try pack.withRawPointer(at: off, byteCount: bytes) { src in
            memcpy(arena.cpuPointer(of: t), src, bytes)
        }
        return QWDeviceEmbedTable(storage: .bf16(t), vocab: vocab, hidden: hidden, hc: hc,
                                  containerNote: "the pack's raw bf16 plane")
    }

    public static func load(arena: QWArena, container: QWSafeTensorsContainer, name: String,
                            vocab: Int, hidden: Int, hc: Int) throws -> QWDeviceEmbedTable {
        let layout = try? QWAQRepacker.layout(container: container, name: name, rows: vocab, cols: hidden)
        if let layout, QWAQBlock.isSupported(bits: layout.bits, group: layout.group) {
            return try nativeAffine(arena: arena, st: container, name: name, vocab: vocab,
                                    hidden: hidden, hc: hc, layout: layout)
        }
        let t = try arena.weightsRegion().allocate(QWTensorShape([vocab * hidden], .f16))
        let dst = arena.cpuPointer(of: t).assumingMemoryBound(to: UInt16.self)
        try QWHQ4Quantizer.copySafeTensorsRowsAsBF16(
            st: container, name: name, rowOffset: 0, rowCount: vocab, cols: hidden, into: dst
        )
        let why = layout.map { "the checkpoint's \($0.bits)-bit group \($0.group) is not a family pair" }
            ?? "the checkpoint's embedding is not an MLX affine tensor"
        return QWDeviceEmbedTable(storage: .bf16(t), vocab: vocab, hidden: hidden, hc: hc,
                                  containerNote: "the incumbent's bf16 plane: " + why)
    }

    private static func nativeAffine(arena: QWArena, st: QWSafeTensorsContainer, name: String,
                                     vocab: Int, hidden: Int, hc: Int,
                                     layout: QWAQRepacker.Layout) throws -> QWDeviceEmbedTable {
        let rowBytes = QWAQBlock.rowBytes(bits: layout.bits, group: layout.group, k: hidden)
        let t = try arena.weightsRegion().allocate(QWTensorShape([(vocab * rowBytes) / 2], .f16))
        try QWAQRepacker.repackMLXAffine(hal: arena.hal, container: st, name: name,
                                         rows: vocab, cols: hidden,
                                         bits: layout.bits, group: layout.group,
                                         into: t.region.buffer, dstByteOffset: t.region.base + t.offset)
        let format = QWAQBlock.formatName(bits: layout.bits, group: layout.group)
        return QWDeviceEmbedTable(storage: .nativeAffine(blocks: t, bits: layout.bits, group: layout.group),
                                  vocab: vocab, hidden: hidden, hc: hc,
                                  containerNote: "the family's \(format) blocks")
    }

    /// Gathers the rows of `token` into `out` on the device.
    public func gather(enc: any MTLComputeCommandEncoder, hal: QWHal,
                       token: QWTensorValue, out: QWTensorValue, rows: Int, hc: Int) throws {
        switch storage {
        case .bf16(let table):
            try QWGemm.encodeEmbedBf16F32(enc, hal: hal, table: table, token: token, out: out,
                                          hidden: hidden, hc: hc, rows: rows)
        case .nativeAffine(let blocks, let bits, let group):
            try QWGemmAQ.encodeEmbedAQ(enc, hal: hal, blocks: blocks, token: token, out: out,
                                       hidden: hidden, hc: hc, bits: bits, group: group, rows: rows)
        }
    }
}

/// Deep module for checkpoint-to-GPU weight bank materialization.
public enum QWCheckpointTensorLoader {
    /// Standard checkpoint prefixes across VLM and text-only HF checkpoints.
    public static let prefixAlternates = ["model.language_model.", "language_model.model.", "model."]

    /// Resolves `name` under whichever prefix the checkpoint used.
    public static func resolveName(
        _ name: String,
        defaultPrefix: String = "model.language_model.",
        tieWordEmbeddings: Bool = false,
        rawExists: (String) -> Bool
    ) -> String {
        if rawExists(name) { return name }
        var matchedPrefix = false
        for prefix in prefixAlternates {
            guard name.hasPrefix(prefix) else { continue }
            matchedPrefix = true
            let rest = String(name.dropFirst(prefix.count))
            for alt in prefixAlternates + [""] {
                let candidate = alt + rest
                if rawExists(candidate) { return candidate }
            }
        }
        if !matchedPrefix {
            for alt in prefixAlternates {
                let candidate = alt + name
                if rawExists(candidate) { return candidate }
            }
        }
        if name == "lm_head.weight", tieWordEmbeddings {
            let tied = resolveName(defaultPrefix + "embed_tokens.weight",
                                   defaultPrefix: defaultPrefix,
                                   tieWordEmbeddings: false,
                                   rawExists: rawExists)
            if rawExists(tied) { return tied }
        }
        return name
    }

    /// True when the SafeTensors checkpoint carries MLX affine quantized weights and
    /// `QW_WEIGHT_FORMAT` has not forced the incumbent rotated `HQ4_G32` path.
    public static func hasNativeAffineWeights(_ st: QWSafeTensorsContainer) -> Bool {
        let forced = ProcessInfo.processInfo.environment["QW_WEIGHT_FORMAT"]?.lowercased()
        if forced == "hq4g32" || forced == "incumbent" {
            return false
        }
        return st.tensors.keys.contains(where: { $0.hasSuffix(".scales") })
    }

    /// Byte span of a 2-D or stacked HQ/SHVQ tensor inside a `.qwp` pack.
    public struct PackSpan: Sendable {
        public let off: Int, bytes: Int, rows: Int, cols: Int
        public let bits: Int, group: Int, isIncumbent: Bool, isSHVQ: Bool

        public init(off: Int, bytes: Int, rows: Int, cols: Int, bits: Int, group: Int,
                    isIncumbent: Bool, isSHVQ: Bool = false) {
            self.off = off
            self.bytes = bytes
            self.rows = rows
            self.cols = cols
            self.bits = bits
            self.group = group
            self.isIncumbent = isIncumbent
            self.isSHVQ = isSHVQ
        }
    }

    public static func span(_ pack: QWQuant.Pack, _ name: String) throws -> PackSpan {
        let e = try pack.entry(name)
        guard e.quantized, let off = e.hq4Off, let rows = e.nRows, let cols = e.kCols else {
            throw QWCheckpointLoaderError.notHQ4(name)
        }
        let grp = e.hqGroup, blk = e.hqBlockBytes
        guard cols % grp == 0 else { throw QWCheckpointLoaderError.notHQ4(name) }
        return PackSpan(off: off, bytes: rows * (cols / grp) * blk, rows: rows, cols: cols,
                        bits: e.hqBits, group: grp, isIncumbent: e.isIncumbentHQ4, isSHVQ: e.isSHVQ)
    }

    public static func stackedSpan(_ pack: QWQuant.Pack, _ name: String) throws -> (span: PackSpan, n: Int, k: Int) {
        let e = try pack.entry(name)
        guard e.quantized, e.shape.count == 3, let off = e.hq4Off, let count = e.experts, let per = e.eHq4 else {
            throw QWCheckpointLoaderError.notHQ4(name)
        }
        let s = PackSpan(off: off, bytes: count * per, rows: count * e.shape[1], cols: e.shape[2],
                         bits: e.hqBits, group: e.hqGroup, isIncumbent: e.isIncumbentHQ4, isSHVQ: e.isSHVQ)
        return (s, count * e.shape[1], e.shape[2])
    }

    public static func mappedBank(_ arena: QWArena, _ pack: QWQuant.Pack, _ s: PackSpan, n: Int, k: Int) -> QWWeightView? {
        guard let res = PlaneIO.mapZeroCopyRange(pack: pack, device: arena.hal.device,
                                                 offset: s.off, byteCount: s.bytes) else { return nil }
        let t = QWTensorValue(region: QWMRegion(buffer: res.buffer, base: 0, length: res.mappedLength),
                              offset: res.subOffset, shape: QWTensorShape([s.bytes / 2], .f16))
        let rot = arena.hal.weightBasis.rotatesActivations
        return QWWeightView(blocks: t, n: n, k: k,
                            bits: s.bits, groupSize: s.group,
                            isRotated: rot, isIncumbentContainer: rot && s.isIncumbent,
                            isSHVQ: rot && s.isSHVQ)
    }

    public static func bankFrom(_ arena: QWArena, _ pack: QWQuant.Pack, _ s: PackSpan, n: Int, k: Int) throws -> QWWeightView {
        if let v = mappedBank(arena, pack, s, n: n, k: k) { return v }
        return try concatBank(arena, pack, [s], n: n, k: k)
    }

    public static func concatBank(_ arena: QWArena, _ pack: QWQuant.Pack, _ spans: [PackSpan], n: Int, k: Int) throws -> QWWeightView {
        guard let first = spans.first,
              spans.allSatisfy({
                  $0.bits == first.bits && $0.group == first.group
                      && $0.isIncumbent == first.isIncumbent && $0.isSHVQ == first.isSHVQ
              })
        else {
            throw QWCheckpointLoaderError.notHQ4("concatBank mismatched HQ formats")
        }
        let totalBytes = spans.reduce(0) { $0 + $1.bytes }
        let t = try arena.weightsRegion().allocate(QWTensorShape([totalBytes / 2], .f16))
        let dst = arena.cpuPointer(of: t)
        var off = 0
        for s in spans {
            _ = try pack.withRawPointer(at: s.off, byteCount: s.bytes) { src in memcpy(dst.advanced(by: off), src, s.bytes) }
            off += s.bytes
        }
        let rot = arena.hal.weightBasis.rotatesActivations
        return QWWeightView(blocks: t, n: n, k: k,
                            bits: first.bits, groupSize: first.group,
                            isRotated: rot, isIncumbentContainer: rot && first.isIncumbent,
                            isSHVQ: rot && first.isSHVQ)!
    }

    public static func hqBank(arena: QWArena, pack: QWQuant.Pack?, st: QWSafeTensorsContainer?,
                              name: String, n: Int, k: Int, staging: (any MTLBuffer)? = nil) throws -> QWWeightView {
        if let pack {
            return try bankFrom(arena, pack, try span(pack, name), n: n, k: k)
        }
        guard let st else { throw QWCheckpointLoaderError.notHQ4(name) }
        if !arena.hal.weightBasis.rotatesActivations,
           let layout = try? QWAQRepacker.layout(container: st, name: name, rows: n, cols: k),
           QWAQBlock.isSupported(bits: layout.bits, group: layout.group) {
            let rowBytes = QWAQBlock.rowBytes(bits: layout.bits, group: layout.group, k: k)
            let totalBytes = n * rowBytes
            let t = try arena.weightsRegion().allocate(QWTensorShape([totalBytes / 2], .f16))
            try QWAQRepacker.repackMLXAffine(
                hal: arena.hal, container: st, name: name, rows: n, cols: k,
                bits: layout.bits, group: layout.group,
                into: t.region.buffer, dstByteOffset: t.region.base + t.offset
            )
            return QWWeightView(blocks: t, n: n, k: k,
                                bits: layout.bits, groupSize: layout.group,
                                isRotated: false)!
        }
        let totalBytes = n * (k / 32) * 20
        let t = try arena.weightsRegion().allocate(QWTensorShape([totalBytes / 2], .f16))
        try QWHQ4Quantizer.quantizeSafeTensorsInto(
            hal: arena.hal, st: st, name: name, srcRowOffset: 0, rowCount: n, k: k,
            dstBuffer: t.region.buffer, dstByteOffset: t.region.base + t.offset, staging: staging
        )
        return QWWeightView(blocks: t, n: n, k: k,
                             isRotated: arena.hal.weightBasis.rotatesActivations)!
    }

    public static func concatHQBank(arena: QWArena, pack: QWQuant.Pack?, st: QWSafeTensorsContainer?,
                                    names: [String], k: Int,
                                    staging: (any MTLBuffer)? = nil) throws -> (view: QWWeightView, rows: Int) {
        if let pack {
            let spans = try names.map { try span(pack, $0) }
            let rows = spans.reduce(0) { $0 + $1.rows }
            return (try concatBank(arena, pack, spans, n: rows, k: k), rows)
        }
        guard let st else { throw QWCheckpointLoaderError.notHQ4(names.first ?? "") }
        let counts = try names.map { try st.reader(of: $0).tensorInfo.shape[0] }
        let totalRows = counts.reduce(0, +)
        if !arena.hal.weightBasis.rotatesActivations {
            let layouts = zip(names, counts).compactMap { (name, rows) in
                try? QWAQRepacker.layout(container: st, name: name, rows: rows, cols: k)
            }
            if layouts.count == names.count, let first = layouts.first,
               QWAQBlock.isSupported(bits: first.bits, group: first.group),
               layouts.allSatisfy({ $0.bits == first.bits && $0.group == first.group }) {
                let rowBytes = QWAQBlock.rowBytes(bits: first.bits, group: first.group, k: k)
                let totalBytes = totalRows * rowBytes
                let t = try arena.weightsRegion().allocate(QWTensorShape([totalBytes / 2], .f16))
                var rowOff = 0
                for (name, rows) in zip(names, counts) {
                    try QWAQRepacker.repackMLXAffine(
                        hal: arena.hal, container: st, name: name, rows: rows, cols: k,
                        bits: first.bits, group: first.group,
                        into: t.region.buffer,
                        dstByteOffset: t.region.base + t.offset + rowOff * rowBytes
                    )
                    rowOff += rows
                }
                return (QWWeightView(blocks: t, n: totalRows, k: k,
                                     bits: first.bits, groupSize: first.group,
                                     isRotated: false)!, totalRows)
            }
        }
        let rowBytes = (k / 32) * 20
        let totalBytes = totalRows * rowBytes
        let t = try arena.weightsRegion().allocate(QWTensorShape([totalBytes / 2], .f16))
        var rowOff = 0
        for (name, rows) in zip(names, counts) {
            try QWHQ4Quantizer.quantizeSafeTensorsInto(
                hal: arena.hal, st: st, name: name, srcRowOffset: 0, rowCount: rows, k: k,
                dstBuffer: t.region.buffer,
                dstByteOffset: t.region.base + t.offset + rowOff * rowBytes,
                staging: staging
            )
            rowOff += rows
        }
        return (QWWeightView(blocks: t, n: totalRows, k: k,
                              isRotated: arena.hal.weightBasis.rotatesActivations)!, totalRows)
    }

    private static let stageLock = NSLock()
    nonisolated(unsafe) private static var sharedRepackStage: (any MTLBuffer)?

    private static func acquireRepackStage(device: any MTLDevice, minBytes: Int) -> (any MTLBuffer)? {
        stageLock.lock()
        defer { stageLock.unlock() }
        if let existing = sharedRepackStage, existing.length >= minBytes {
            return existing
        }
        sharedRepackStage = nil
        let buf = device.makeBuffer(length: minBytes, options: .storageModeShared)
        sharedRepackStage = buf
        return buf
    }

    public static func releaseRepackStageBuffer() {
        stageLock.lock()
        sharedRepackStage = nil
        stageLock.unlock()
    }

    private static func moeSlabCacheDirectory(for st: QWSafeTensorsContainer, isRotated: Bool) -> URL? {
        if ProcessInfo.processInfo.environment["QW_DISABLE_MOE_SLAB_CACHE"] == "1" {
            return nil
        }
        let fm = FileManager.default
        let baseDir: URL
        if let custom = ProcessInfo.processInfo.environment["QW_MOE_CACHE_DIR"], !custom.isEmpty {
            baseDir = URL(fileURLWithPath: custom)
        } else {
            baseDir = fm.homeDirectoryForCurrentUser.appendingPathComponent(".qwg/cache/moe_slabs")
        }
        let stPath = st.url.standardizedFileURL.path
        let attrs = try? fm.attributesOfItem(atPath: stPath)
        let size = (attrs?[.size] as? NSNumber)?.uint64Value ?? 0
        let mtime = (attrs?[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
        let keyStr = "\(stPath)|\(size)|\(Int64(mtime))|\(st.tensors.count)|\(isRotated ? "rot" : "std")"
        var hash: UInt64 = 0xcbf29ce484222325
        for b in keyStr.utf8 {
            hash ^= UInt64(b)
            hash = hash &* 0x100000001b3
        }
        let dir = baseDir.appendingPathComponent(String(format: "st_%016llx", hash))
        do {
            try fm.createDirectory(at: dir, withIntermediateDirectories: true)
            return dir
        } catch {
            return nil
        }
    }

    private static func mapCachedMoESlab(at url: URL, alignedGU: Int, alignedDN: Int,
                                         device: any MTLDevice) -> (guBuf: any MTLBuffer, dnBuf: any MTLBuffer)? {
        let totalBytes = alignedGU + alignedDN
        let fd = Darwin.open(url.path, O_RDONLY)
        guard fd >= 0 else { return nil }
        defer { Darwin.close(fd) }
        var st = Darwin.stat()
        guard Darwin.fstat(fd, &st) == 0, Int(st.st_size) == totalBytes else { return nil }
        let raw = Darwin.mmap(nil, totalBytes, PROT_READ, MAP_SHARED, fd, 0)
        guard let basePtr = raw, basePtr != MAP_FAILED else { return nil }
        _ = Darwin.posix_madvise(basePtr, totalBytes, POSIX_MADV_RANDOM)
        guard let guBuf = device.makeBuffer(
            bytesNoCopy: basePtr, length: alignedGU, options: .storageModeShared,
            deallocator: { ptr, len in _ = Darwin.munmap(ptr, len) }
        ) else {
            _ = Darwin.munmap(basePtr, totalBytes)
            return nil
        }
        let dnPtr = basePtr.advanced(by: alignedGU)
        guard let dnBuf = device.makeBuffer(
            bytesNoCopy: dnPtr, length: alignedDN, options: .storageModeShared,
            deallocator: { ptr, len in _ = Darwin.munmap(ptr, len) }
        ) else {
            _ = Darwin.munmap(dnPtr, alignedDN)
            return nil
        }
        QWResidencyRegistry.note(guBuf)
        QWResidencyRegistry.note(dnBuf)
        return (guBuf, dnBuf)
    }

    private static func pwriteAllNoCache(fd: Int32, from src: UnsafeRawPointer, count: Int, offset: Int) -> Bool {
        let chunkLimit = 64 << 20
        var written = 0
        while written < count {
            let chunk = min(chunkLimit, count - written)
            let w = Darwin.pwrite(fd, src.advanced(by: written), chunk, off_t(offset + written))
            if w <= 0 { return false }
            written += w
        }
        return true
    }

    public static func stackedExpertBanks(
        arena: QWArena, pack: QWQuant.Pack?, st: QWSafeTensorsContainer?, mp: String,
        experts: Int, inter: Int, hidden: Int, staging: (any MTLBuffer)? = nil
    ) throws -> (gateUp: QWWeightView, down: QWWeightView) {
        if let pack {
            let gu = try stackedSpan(pack, mp + "experts.gate_up_proj")
            let dn = try stackedSpan(pack, mp + "experts.down_proj")
            return (try bankFrom(arena, pack, gu.span, n: gu.n, k: gu.k),
                    try bankFrom(arena, pack, dn.span, n: dn.n, k: dn.k))
        }
        guard let st else { throw QWCheckpointLoaderError.notHQ4(mp) }
        let guRows = experts * 2 * inter
        let dnRows = experts * hidden
        let downName = st.tensor(mp + "experts.down_proj") != nil
            ? (mp + "experts.down_proj") : (mp + "switch_mlp.down_proj.weight")
        let pageSize = max(4096, Int(sysconf(_SC_PAGESIZE)))

        if !arena.hal.weightBasis.rotatesActivations {
            let guLayout: QWAQRepacker.Layout? = st.tensor(mp + "experts.gate_up_proj") != nil
                ? (try? QWAQRepacker.layout(container: st, name: mp + "experts.gate_up_proj", rows: guRows, cols: hidden))
                : (try? QWAQRepacker.layout(container: st, name: mp + "switch_mlp.gate_proj.weight", rows: experts * inter, cols: hidden))
            let dnLayout = try? QWAQRepacker.layout(container: st, name: downName, rows: dnRows, cols: inter)
            if let guLayout, let dnLayout,
               QWAQBlock.isSupported(bits: guLayout.bits, group: guLayout.group),
               QWAQBlock.isSupported(bits: dnLayout.bits, group: dnLayout.group) {
                let guRowBytes = QWAQBlock.rowBytes(bits: guLayout.bits, group: guLayout.group, k: hidden)
                let guBytes = guRows * guRowBytes
                let dnRowBytes = QWAQBlock.rowBytes(bits: dnLayout.bits, group: dnLayout.group, k: inter)
                let dnBytes = dnRows * dnRowBytes
                let alignedGU = ((guBytes + pageSize - 1) / pageSize) * pageSize
                let alignedDN = ((dnBytes + pageSize - 1) / pageSize) * pageSize

                if guBytes + dnBytes >= (16 << 20),
                   let cacheDir = moeSlabCacheDirectory(for: st, isRotated: false) {
                    let sanitizedMP = mp.replacingOccurrences(of: ".", with: "_")
                    let slabName = "\(sanitizedMP)gu\(guLayout.bits)g\(guLayout.group)_dn\(dnLayout.bits)g\(dnLayout.group)_\(guRows)x\(hidden)_\(dnRows)x\(inter).slab"
                    let slabURL = cacheDir.appendingPathComponent(slabName)
                    if let mapped = mapCachedMoESlab(at: slabURL, alignedGU: alignedGU, alignedDN: alignedDN, device: arena.hal.device) {
                        let guTensor = QWTensorValue(region: QWMRegion(buffer: mapped.guBuf, base: 0, length: guBytes),
                                                     offset: 0, shape: QWTensorShape([guBytes / 2], .f16))
                        let dnTensor = QWTensorValue(region: QWMRegion(buffer: mapped.dnBuf, base: 0, length: dnBytes),
                                                     offset: 0, shape: QWTensorShape([dnBytes / 2], .f16))
                        return (QWWeightView(blocks: guTensor, n: guRows, k: hidden,
                                             bits: guLayout.bits, groupSize: guLayout.group, isRotated: false)!,
                                QWWeightView(blocks: dnTensor, n: dnRows, k: inter,
                                             bits: dnLayout.bits, groupSize: dnLayout.group, isRotated: false)!)
                    }
                    if let stageBuf = acquireRepackStage(device: arena.hal.device, minBytes: max(alignedGU, alignedDN)) {
                        let tmpURL = slabURL.appendingPathExtension("tmp.\(getpid())")
                        let tmpFD = Darwin.open(tmpURL.path, O_RDWR | O_CREAT | O_TRUNC, 0o600)
                        if tmpFD >= 0 {
                            _ = Darwin.fcntl(tmpFD, F_NOCACHE, 1)
                            if Darwin.ftruncate(tmpFD, off_t(alignedGU + alignedDN)) == 0 {
                                if st.tensor(mp + "experts.gate_up_proj") != nil {
                                    try QWAQRepacker.repackMLXAffine(
                                        hal: arena.hal, container: st, name: mp + "experts.gate_up_proj",
                                        rows: guRows, cols: hidden, bits: guLayout.bits, group: guLayout.group,
                                        into: stageBuf, dstByteOffset: 0)
                                } else {
                                    try QWAQRepacker.repackMLXAffineSwitchGateUp(
                                        hal: arena.hal, container: st,
                                        gateName: mp + "switch_mlp.gate_proj.weight",
                                        upName: mp + "switch_mlp.up_proj.weight",
                                        experts: experts, inter: inter, cols: hidden,
                                        bits: guLayout.bits, group: guLayout.group,
                                        into: stageBuf, dstByteOffset: 0)
                                }
                                let okGU = pwriteAllNoCache(fd: tmpFD, from: stageBuf.contents(), count: guBytes, offset: 0)
                                try QWAQRepacker.repackMLXAffine(
                                    hal: arena.hal, container: st, name: downName,
                                    rows: dnRows, cols: inter, bits: dnLayout.bits, group: dnLayout.group,
                                    into: stageBuf, dstByteOffset: 0)
                                let okDN = okGU && pwriteAllNoCache(fd: tmpFD, from: stageBuf.contents(), count: dnBytes, offset: alignedGU)
                                Darwin.close(tmpFD)
                                if okDN, Darwin.rename(tmpURL.path, slabURL.path) == 0,
                                   let mapped = mapCachedMoESlab(at: slabURL, alignedGU: alignedGU, alignedDN: alignedDN, device: arena.hal.device) {
                                    let guTensor = QWTensorValue(region: QWMRegion(buffer: mapped.guBuf, base: 0, length: guBytes),
                                                                 offset: 0, shape: QWTensorShape([guBytes / 2], .f16))
                                    let dnTensor = QWTensorValue(region: QWMRegion(buffer: mapped.dnBuf, base: 0, length: dnBytes),
                                                                 offset: 0, shape: QWTensorShape([dnBytes / 2], .f16))
                                    return (QWWeightView(blocks: guTensor, n: guRows, k: hidden,
                                                         bits: guLayout.bits, groupSize: guLayout.group, isRotated: false)!,
                                            QWWeightView(blocks: dnTensor, n: dnRows, k: inter,
                                                         bits: dnLayout.bits, groupSize: dnLayout.group, isRotated: false)!)
                                }
                            } else {
                                Darwin.close(tmpFD)
                            }
                            _ = Darwin.unlink(tmpURL.path)
                        }
                    }
                }

                guard let guBuf = arena.hal.device.makeBuffer(length: guBytes, options: .storageModeShared),
                      let dnBuf = arena.hal.device.makeBuffer(length: dnBytes, options: .storageModeShared) else {
                    throw QWCheckpointLoaderError.notHQ4(mp + "switch_mlp")
                }
                arena.hal.makeResident([guBuf, dnBuf])
                let guTensor = QWTensorValue(region: QWMRegion(buffer: guBuf, base: 0, length: guBytes),
                                             offset: 0, shape: QWTensorShape([guBytes / 2], .f16))
                let dnTensor = QWTensorValue(region: QWMRegion(buffer: dnBuf, base: 0, length: dnBytes),
                                             offset: 0, shape: QWTensorShape([dnBytes / 2], .f16))
                if st.tensor(mp + "experts.gate_up_proj") != nil {
                    try QWAQRepacker.repackMLXAffine(
                        hal: arena.hal, container: st, name: mp + "experts.gate_up_proj",
                        rows: guRows, cols: hidden, bits: guLayout.bits, group: guLayout.group,
                        into: guBuf, dstByteOffset: 0)
                } else {
                    try QWAQRepacker.repackMLXAffineSwitchGateUp(
                        hal: arena.hal, container: st,
                        gateName: mp + "switch_mlp.gate_proj.weight",
                        upName: mp + "switch_mlp.up_proj.weight",
                        experts: experts, inter: inter, cols: hidden,
                        bits: guLayout.bits, group: guLayout.group,
                        into: guBuf, dstByteOffset: 0)
                }
                try QWAQRepacker.repackMLXAffine(
                    hal: arena.hal, container: st, name: downName,
                    rows: dnRows, cols: inter, bits: dnLayout.bits, group: dnLayout.group,
                    into: dnBuf, dstByteOffset: 0)
                return (QWWeightView(blocks: guTensor, n: guRows, k: hidden,
                                     bits: guLayout.bits, groupSize: guLayout.group, isRotated: false)!,
                        QWWeightView(blocks: dnTensor, n: dnRows, k: inter,
                                     bits: dnLayout.bits, groupSize: dnLayout.group, isRotated: false)!)
            }
        }

        let guRowBytes = (hidden / 32) * 20
        let guBytes = guRows * guRowBytes
        let dnRowBytes = (inter / 32) * 20
        let dnBytes = dnRows * dnRowBytes
        guard let guBuf = arena.hal.device.makeBuffer(length: guBytes, options: .storageModeShared),
              let dnBuf = arena.hal.device.makeBuffer(length: dnBytes, options: .storageModeShared) else {
            throw QWCheckpointLoaderError.notHQ4(mp + "switch_mlp")
        }
        arena.hal.makeResident([guBuf, dnBuf])
        let guTensor = QWTensorValue(region: QWMRegion(buffer: guBuf, base: 0, length: guBytes),
                                     offset: 0, shape: QWTensorShape([guBytes / 2], .f16))
        let dnTensor = QWTensorValue(region: QWMRegion(buffer: dnBuf, base: 0, length: dnBytes),
                                     offset: 0, shape: QWTensorShape([dnBytes / 2], .f16))
        let stageBytes = 128 * 1024 * 1024
        guard let stageBuf = staging ?? arena.hal.device.makeBuffer(length: stageBytes, options: .storageModeShared) else {
            throw QWCheckpointLoaderError.notHQ4(mp)
        }
        if st.tensor(mp + "experts.gate_up_proj") != nil {
            try QWHQ4Quantizer.quantizeSafeTensorsInto(
                hal: arena.hal, st: st, name: mp + "experts.gate_up_proj",
                srcRowOffset: 0, rowCount: guRows, k: hidden,
                dstBuffer: guBuf, dstByteOffset: 0, staging: stageBuf)
        } else {
            try QWHQ4Quantizer.quantizeSwitchGateUpInto(
                hal: arena.hal, st: st, gateName: mp + "switch_mlp.gate_proj.weight",
                upName: mp + "switch_mlp.up_proj.weight", experts: experts, inter: inter, hidden: hidden,
                dstBuffer: guBuf, dstByteOffset: 0, staging: stageBuf)
        }
        try QWHQ4Quantizer.quantizeSafeTensorsInto(
            hal: arena.hal, st: st, name: downName,
            srcRowOffset: 0, rowCount: dnRows, k: inter,
            dstBuffer: dnBuf, dstByteOffset: 0, staging: stageBuf)
        let rot = arena.hal.weightBasis.rotatesActivations
        return (QWWeightView(blocks: guTensor, n: guRows, k: hidden, isRotated: rot)!,
                QWWeightView(blocks: dnTensor, n: dnRows, k: inter, isRotated: rot)!)
    }
}
