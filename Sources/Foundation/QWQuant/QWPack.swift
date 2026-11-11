// Host-side QWP1 pack container reader and reference HQ4_G32_M5 dequantizer.
// Supports 2D HQ4_G32_M5 planes and verbatim BF16 planes for Qwen3.8-27B.

import CryptoKit
import Darwin
import Foundation
import QWCore

public enum QWQuantError: Error, CustomStringConvertible {
    case badMagic
    case geometryMismatch(pack: String, expected: String)
    case modelMismatch(pack: String, expected: String)
    case geometryMissing(String)
    case tensorMissing(String)
    case truncated
    case notQuantPlane(String)
    case hq8SwitchOff(String)
    case notRawPlane(String)
    case modelFileMissing(String)

    public var description: String {
        switch self {
        case .badMagic: return "QWQuant: bad magic (want QWP1)"
        case .geometryMismatch(let p, let e): return "QWQuant: pack geometry \(p) ≠ engine \(e)"
        case .modelMismatch(let p, let e):
            return "QWQuant: pack model '\(p)' ≠ engine model '\(e)' — refuse before allocation"
        case .geometryMissing(let m): return "QWQuant: pack for model '\(m)' carries no geometry_digest"
        case .tensorMissing(let n): return "QWQuant: tensor '\(n)' not in pack"
        case .truncated: return "QWQuant: truncated pack"
        case .notQuantPlane(let n): return "QWQuant: '\(n)' is a verbatim-bf16 tensor (no hq4/q8 planes)"
        case .hq8SwitchOff(let n):
            return "QWQuant: '\(n)' is HQ8-packed but QW_DENSE_W8 is off — set QW_DENSE_W8=1"
        case .notRawPlane(let n): return "QWQuant: '\(n)' is a quantized plane (no raw_off)"
        case .modelFileMissing(let p): return "QWQuant: model weights not found: \(p)"
        }
    }
}

public struct QWModelLocation: Sendable {
    public let weights: URL
    public let assets: URL
    /// Which weight container the root resolved to.
    public let kind: QWWeightSourceKind
    /// The preference that produced `kind`.
    public let preference: QWWeightSourcePreference

    public init(url: URL,
                assetsOverride: URL? = nil,
                preference: QWWeightSourcePreference = .auto) throws {
        let root = QWModelRoot(url: url, preference: preference)
        guard let kind = root.kind else {
            throw QWQuantError.modelFileMissing(root.weights.path)
        }
        self.weights = root.weights
        self.kind = kind
        self.preference = preference
        self.assets = assetsOverride ?? root.assets
    }
}

public struct QWHQFormat: Decodable, Sendable, Equatable, Hashable {
    public let bits: Int
    public let group: Int
    public let vq: String?

    public init(bits: Int, group: Int, vq: String? = nil) {
        self.bits = bits
        self.group = group
        self.vq = vq
    }
}

public struct QWQTensorEntry: Decodable, Sendable {
    public let shape: [Int]
    public let hq4Off: Int?
    public var q4gOff: Int? { hq4Off }
    public let q8Off: Int?
    public let q8SOff: Int?
    public let hq8Off: Int?
    public let hq8SOff: Int?
    public let rawOff: Int?
    public let kind: String?
    /// Expert stacks: expert count and bytes per expert (`hq4_off + e*e_hq4`).
    public let experts: Int?
    public let eHq4: Int?
    public let hq: QWHQFormat?

    public init(shape: [Int], hq4Off: Int? = nil, q8Off: Int? = nil, q8SOff: Int? = nil,
                hq8Off: Int? = nil, hq8SOff: Int? = nil,
                rawOff: Int? = nil, kind: String? = nil, experts: Int? = nil, eHq4: Int? = nil,
                hq: QWHQFormat? = nil) {
        self.shape = shape
        self.hq4Off = hq4Off
        self.q8Off = q8Off
        self.q8SOff = q8SOff
        self.hq8Off = hq8Off
        self.hq8SOff = hq8SOff
        self.rawOff = rawOff
        self.kind = kind
        self.experts = experts
        self.eHq4 = eHq4
        self.hq = hq
    }

    enum CodingKeys: String, CodingKey {
        case shape
        case hq4Off = "hq4_off"
        case q8Off = "q8_off"
        case q8SOff = "q8_s_off"
        case hq8Off = "hq8_off"
        case hq8SOff = "hq8_s_off"
        case rawOff = "raw_off"
        case kind
        case experts
        case eHq4 = "e_hq4"
        case hq
    }

    public var hqBits: Int { hq?.bits ?? 4 }
    public var hqGroup: Int { hq?.group ?? QWQuant.group }
    public var isIncumbentHQ4: Bool { hq == nil }
    public var isSHVQ: Bool { hq?.vq == "shvq1" }
    public var hqBlockBytes: Int {
        if let hq {
            if hq.vq == "shvq1" {
                return QWSHVQBlock.bytes(bits: hq.bits, group: hq.group)
            }
            return QWAQBlock.bytes(bits: hq.bits, group: hq.group)
        }
        return QWQuant.blockBytes
    }
    public var nRows: Int? { shape.count == 2 ? shape[0] : nil }
    public var kCols: Int? { shape.count == 2 ? shape[1] : nil }
    public var groups: Int? { shape.count == 2 ? shape[1] / hqGroup : nil }
    public var isRawBf16: Bool { rawOff != nil }
    public var quantized: Bool { rawOff == nil && hq4Off != nil }
    public var hq8Quantized: Bool { rawOff == nil && hq8Off != nil && hq4Off == nil }

    /// Validated HQ8 plane extents `(payloadBytes, scaleBytes)` for a 2-D
    /// `hq8_g32_m5` entry (payload `N*K` int8, scale `N*(K/32)*2` f16).
    public var hq8Extents: (payload: Int, scale: Int)? {
        guard let n = nRows, let k = kCols, k % QWQuant.group == 0,
              hq8Off != nil, hq8SOff != nil else { return nil }
        let payload = n * k, scale = n * (k / QWQuant.group) * 2
        guard payload > 0, scale > 0 else { return nil }
        return (payload, scale)
    }
    public var hq8PayloadBytes: Int? { hq8Extents?.payload }
    public var hq8ScaleBytes: Int? { hq8Extents?.scale }


}

public enum QWQuant {
    public static let magic = "QWP1"
    public static let group = QWHQ4Layout.group
    public static let blockBytes = QWHQ4Layout.blockBytes
    public static let invSqrt32: Float = Float(bitPattern: 0x3E3504F3)

    /// Stable v1 geometry digest of the C01 numbers a 27B control pack must match.
    public static var geometryHash: String {
        sha256Hex(geometryHashString)
    }

    public static var geometryHashString: String {
        QWModelDescriptor.qwen38_27b.legacyGeoV1String
    }

    struct Header: Decodable {
        struct Prov: Decodable {
            let geometry_hash: String?
            let source: String?
            let sha256_source: String?
            let model_id: String?
            let geometry_digest: String?
        }
        let provenance: Prov
        let tensors: [String: QWQTensorEntry]
    }

    /// Mmap-backed QWP1 pack reader.
    public final class Pack {
        public let url: URL
        let header: Header
        let rawBase: UnsafeMutableRawPointer
        let rawCount: Int
        public let descriptor: QWModelDescriptor
        public private(set) var wiredBytes = 0

        public convenience init(url: URL) throws {
            let loc = try QWModelLocation(url: url)
            let cfgURL = loc.assets.appendingPathComponent("config.json")
            let desc: QWModelDescriptor
            if FileManager.default.fileExists(atPath: cfgURL.path) {
                desc = try QWModelDescriptor.decode(configJSON: Data(contentsOf: cfgURL))
            } else {
                desc = QWModelDescriptor.qwen38_27b
            }
            try self.init(url: url, descriptor: desc)
        }

        private static func mapFileShared(url: URL) throws -> (UnsafeMutableRawPointer, Int) {
            let fd = open(url.path, O_RDONLY)
            guard fd >= 0 else { throw QWQuantError.modelFileMissing(url.path) }
            defer { close(fd) }
            var st = stat()
            guard fstat(fd, &st) == 0 else { throw QWQuantError.modelFileMissing(url.path) }
            let size = Int(st.st_size)
            guard size > 12 else { throw QWQuantError.truncated }
            let raw = mmap(nil, size, PROT_READ, MAP_SHARED, fd, 0)
            guard let ptr = raw, ptr != MAP_FAILED else {
                throw QWQuantError.modelFileMissing(url.path)
            }
            return (ptr, size)
        }

        deinit {
            if rawCount > 0 {
                _ = munmap(rawBase, rawCount)
            }
        }

        /// Opens a pack and validates its v1/v2 geometry identity against `descriptor`.
        public init(url: URL, descriptor: QWModelDescriptor, skipIdentityCheck: Bool = false) throws {
            let url = try QWModelLocation(url: url).weights
            self.url = url
            let (base, count) = try Self.mapFileShared(url: url)
            self.rawBase = base
            self.rawCount = count
            let bytes = base.assumingMemoryBound(to: UInt8.self)
            let magicSlice = UnsafeBufferPointer(start: bytes, count: 4)
            guard String(decoding: magicSlice, as: UTF8.self) == QWQuant.magic else {
                throw QWQuantError.badMagic
            }
            var hdrLen: UInt64 = 0
            for i in 0..<8 {
                hdrLen |= UInt64(bytes[4 + i]) << (8 * i)
            }
            let hdrStart = count - Int(hdrLen)
            guard hdrStart >= 12 else { throw QWQuantError.truncated }
            let hdrData = Data(bytes: base.advanced(by: hdrStart), count: Int(hdrLen))
            let header = try JSONDecoder().decode(Header.self, from: hdrData)
            if !skipIdentityCheck {
                if let mid = header.provenance.model_id {
                    let midFamily = QWModelDescriptor.Family(rawValue: mid)
                        ?? QWModelFamilyRegistry.shared.spec(forModelType: mid)?.family
                    let familyMatches = midFamily == descriptor.family
                        || (midFamily == .qwen3MoE && descriptor.family == .qwen38Dense && (descriptor.numExperts ?? 0) > 0)
                    guard familyMatches else {
                        throw QWQuantError.modelMismatch(pack: mid, expected: descriptor.family.rawValue)
                    }
                    guard let dg = header.provenance.geometry_digest, !dg.isEmpty else {
                        throw QWQuantError.geometryMissing(mid)
                    }
                    guard dg == descriptor.geometryDigestV2Hex else {
                        throw QWQuantError.geometryMismatch(pack: dg, expected: descriptor.geometryDigestV2Hex)
                    }
                } else {
                    guard descriptor.family == .qwen38Dense else {
                        throw QWQuantError.modelMismatch(pack: "legacy-v1", expected: descriptor.family.rawValue)
                    }
                    let expectedLegacy = sha256Hex(descriptor.legacyGeoV1String)
                    if let g = header.provenance.geometry_hash, g != expectedLegacy {
                        throw QWQuantError.geometryMismatch(pack: g, expected: expectedLegacy)
                    }
                }
            }
            self.header = header
            self.descriptor = descriptor
        }

        public func entry(_ name: String) throws -> QWQTensorEntry {
            guard let e = header.tensors[name] else { throw QWQuantError.tensorMissing(name) }
            return e
        }

        public var tensorNames: [String] { header.tensors.keys.sorted() }
        public var tensors: [String: QWQTensorEntry] { header.tensors }
        public var packedByteCount: Int { rawCount }

        public var provenance: (geometryHash: String?, source: String?, sha256Source: String?) {
            (header.provenance.geometry_hash, header.provenance.source, header.provenance.sha256_source)
        }

        public var modelIdentity: (modelId: String?, geometryDigest: String?) {
            (header.provenance.model_id, header.provenance.geometry_digest)
        }

        /// Reads only the trailer provenance without binding a model descriptor.
        public static func peekProvenance(url: URL) throws -> (
            modelId: String?, geometryHash: String?, geometryDigest: String?,
            source: String?, sha256Source: String?, tensorCount: Int
        ) {
            let url = try QWModelLocation(url: url).weights
            let fh = try FileHandle(forReadingFrom: url)
            defer { try? fh.close() }
            guard let lenData = try fh.read(upToCount: 12), lenData.count == 12 else {
                throw QWQuantError.truncated
            }
            guard String(decoding: lenData[0..<4], as: UTF8.self) == magic else {
                throw QWQuantError.badMagic
            }
            var hdrLen: UInt64 = 0
            for i in 0..<8 {
                hdrLen |= UInt64(lenData[4 + i]) << (8 * i)
            }
            let fileSize = fh.seekToEndOfFile()
            guard fileSize >= 12 + hdrLen else { throw QWQuantError.truncated }
            try fh.seek(toOffset: fileSize - hdrLen)
            guard let head = try fh.read(upToCount: Int(hdrLen)) else { throw QWQuantError.truncated }
            let h = try JSONDecoder().decode(Header.self, from: head)
            return (h.provenance.model_id, h.provenance.geometry_hash,
                    h.provenance.geometry_digest, h.provenance.source, h.provenance.sha256_source,
                    h.tensors.count)
        }

        @inline(__always)
        private static func dequantHQ4Row(
            base: UnsafeRawPointer, dstBase: UnsafeMutablePointer<Float>,
            row r: Int, groups g: Int, kCols: Int
        ) {
            for gg in 0..<g {
                let blk = base.advanced(by: (r * g + gg) * QWQuant.blockBytes)
                let dst = dstBase.advanced(by: r * kCols + gg * 32)
                QWQuant.dequantHQ4Group(blk, into: dst)
                QWQuant.fwht32OrthoBlock(dst)
            }
        }

        @inline(__always)
        public static func dequantHQ4Slab(
            base: UnsafeRawPointer, nRows: Int, kCols: Int
        ) -> [Float] {
            let g = kCols / QWQuant.group
            var out = [Float](repeating: 0, count: nRows * kCols)
            out.withUnsafeMutableBufferPointer { outBuf in
                let dstBase = outBuf.baseAddress!
                if nRows >= 64 {
                    let rawAddr = Int(bitPattern: base)
                    let dstAddr = Int(bitPattern: dstBase)
                    let workers = min(nRows, 16)
                    let per = (nRows + workers - 1) / workers
                    DispatchQueue.concurrentPerform(iterations: workers) { w in
                        let start = w * per
                        let end = min(nRows, start + per)
                        guard start < end else { return }
                        let wBase = UnsafeRawPointer(bitPattern: rawAddr)!
                        let wDst = UnsafeMutablePointer<Float>(bitPattern: dstAddr)!
                        for r in start..<end {
                            dequantHQ4Row(base: wBase, dstBase: wDst, row: r, groups: g, kCols: kCols)
                        }
                    }
                } else {
                    for r in 0..<nRows {
                        dequantHQ4Row(base: base, dstBase: dstBase, row: r, groups: g, kCols: kCols)
                    }
                }
            }
            return out
        }

        /// Dequantizes a 2D `HQ{2,3,4,6,8}_G{32,64,128,256}` or `SHVQ` slab in the rotated layout.
        public static func dequantHQBlockSlab(
            base: UnsafeRawPointer, nRows: Int, kCols: Int, bits: Int, group: Int,
            isSHVQ: Bool = false
        ) -> [Float] {
            let g = kCols / group
            let blkBytes = isSHVQ
                ? QWSHVQBlock.bytes(bits: bits, group: group)
                : QWAQBlock.bytes(bits: bits, group: group)
            var out = [Float](repeating: 0, count: nRows * kCols)
            out.withUnsafeMutableBufferPointer { outBuf in
                let dstBase = outBuf.baseAddress!
                var groupBuf = [Float]()
                groupBuf.reserveCapacity(group)
                for r in 0..<nRows {
                    for gg in 0..<g {
                        let blkPtr = base.advanced(by: (r * g + gg) * blkBytes)
                        let rawBuf = UnsafeRawBufferPointer(start: blkPtr, count: blkBytes)
                        if isSHVQ {
                            QWSHVQBlock.unpack(group: rawBuf, bits: bits, group: group, into: &groupBuf)
                        } else {
                            QWAQBlock.unpack(group: rawBuf, bits: bits, group: group, into: &groupBuf)
                        }
                        let rowDst = dstBase.advanced(by: r * kCols + gg * group)
                        for i in 0..<group {
                            rowDst[i] = groupBuf[i]
                        }
                        var off = 0
                        while off < group {
                            QWQuant.fwht32OrthoBlock(rowDst.advanced(by: off))
                            off += 32
                        }
                    }
                }
            }
            return out
        }

        /// Dequantizes a 2D `HQ4_G32_M5`, `HQ`, or `SHVQ` tensor into fp32 floats.
        public func dequantQ4g(_ name: String) throws -> [Float] {
            let e = try entry(name)
            guard let hq4 = e.hq4Off else { throw QWQuantError.notQuantPlane(name) }
            guard let n = e.nRows, let g = e.groups, let K = e.kCols else {
                throw QWQuantError.notQuantPlane(name)
            }
            let blkBytes = e.hqBlockBytes
            guard K % e.hqGroup == 0, hq4 + n * g * blkBytes <= rawCount else {
                throw QWQuantError.truncated
            }
            let ptr = UnsafeRawPointer(rawBase).advanced(by: hq4)
            if e.isIncumbentHQ4 {
                return Self.dequantHQ4Slab(base: ptr, nRows: n, kCols: K)
            }
            return Self.dequantHQBlockSlab(base: ptr, nRows: n, kCols: K,
                                           bits: e.hqBits, group: e.hqGroup, isSHVQ: e.isSHVQ)
        }


        /// One expert of a 3-D HQ/SHVQ expert stack `[E, R, C]`; expert `e` starts at `hq4_off + e*e_hq4`.
        public func dequantExpertQ4g(_ name: String, expert: Int) throws -> [Float] {
            let e = try entry(name)
            guard let hq4 = e.hq4Off, let count = e.experts, let per = e.eHq4, e.shape.count == 3 else {
                throw QWQuantError.notQuantPlane(name)
            }
            let rows = e.shape[1], cols = e.shape[2]
            let grp = e.hqGroup, blkBytes = e.hqBlockBytes
            guard (0..<count).contains(expert), cols % grp == 0,
                  per == rows * (cols / grp) * blkBytes else {
                throw QWQuantError.notQuantPlane(name)
            }
            let start = hq4 + expert * per
            guard start + per <= rawCount else { throw QWQuantError.truncated }
            let ptr = UnsafeRawPointer(rawBase).advanced(by: start)
            if e.isIncumbentHQ4 {
                return Self.dequantHQ4Slab(base: ptr, nRows: rows, kCols: cols)
            }
            return Self.dequantHQBlockSlab(base: ptr, nRows: rows, kCols: cols,
                                           bits: e.hqBits, group: grp, isSHVQ: e.isSHVQ)
        }

        /// HQ8_G32_M5 rot-domain dequant: `float(q) * float16(scale)` per
        /// group-32, WITHOUT the inverse FWHT (kernel-comparison domain).
        public func dequantHQ8Rot(_ name: String) throws -> [Float] {
            let e = try entry(name)
            guard let p = e.hq8Off, let s = e.hq8SOff,
                  let ext = e.hq8Extents, let n = e.nRows, let g = e.groups,
                  let K = e.kCols else { throw QWQuantError.notQuantPlane(name) }
            guard p + ext.payload <= rawCount,
                  s + ext.scale <= rawCount else { throw QWQuantError.truncated }
            var out = [Float](repeating: 0, count: n * K)
            let q = UnsafeRawPointer(rawBase).advanced(by: p).assumingMemoryBound(to: Int8.self)
            let sc = UnsafeRawPointer(rawBase).advanced(by: s).assumingMemoryBound(to: UInt16.self)
            for r in 0..<n {
                for gg in 0..<g {
                    let scale = QWQuant.f16tof32Bits(sc[r * g + gg])
                    let row = r * K + gg * QWQuant.group
                    for j in 0..<QWQuant.group {
                        out[row + j] = Float(q[row + j]) * scale
                    }
                }
            }
            return out
        }

        /// HQ8_G32_M5 standard-domain dequant with inverse 32-point orthonormal FWHT.
        public func dequantHQ8(_ name: String) throws -> [Float] {
            var out = try dequantHQ8Rot(name)
            let count = out.count
            out.withUnsafeMutableBufferPointer { buf in
                guard let base = buf.baseAddress else { return }
                var off = 0
                while off + 32 <= count {
                    QWQuant.fwht32OrthoBlock(base.advanced(by: off))
                    off += 32
                }
            }
            return out
        }

        /// Legacy `q8_0` escape hatch (pre-Q8-removal packs only).
        public func dequantQ8(_ name: String) throws -> [Float] {
            let e = try entry(name)
            guard let q8 = e.q8Off, let q8s = e.q8SOff else { throw QWQuantError.notQuantPlane(name) }
            guard let n = e.nRows, let g = e.groups, let K = e.kCols else {
                throw QWQuantError.notQuantPlane(name)
            }
            guard q8s + n * g * 2 <= rawCount else { throw QWQuantError.truncated }
            var out = [Float](repeating: 0, count: n * K)
            let q = UnsafeRawPointer(rawBase).advanced(by: q8).assumingMemoryBound(to: Int8.self)
            let sc = UnsafeRawPointer(rawBase).advanced(by: q8s).assumingMemoryBound(to: UInt16.self)
            for r in 0..<n {
                for gg in 0..<g {
                    let s = QWQuant.f16tof32Bits(sc[r * g + gg])
                    let row = r * K + gg * 128
                    for j in 0..<128 { out[row + j] = Float(q[row + j]) * s }
                }
            }
            return out
        }

        /// Widens a verbatim BF16 plane into `[Float]`.
        public func rawBf16(_ name: String) throws -> [Float] {
            let e = try entry(name)
            guard let off = e.rawOff else { throw QWQuantError.notRawPlane(name) }
            let count = e.shape.reduce(1, *)
            guard off + count * 2 <= rawCount else { throw QWQuantError.truncated }
            var out = [Float](repeating: 0, count: count)
            let u = UnsafeRawPointer(rawBase).advanced(by: off).assumingMemoryBound(to: UInt16.self)
            for i in 0..<count {
                out[i] = Float(bitPattern: UInt32(u[i]) << 16)
            }
            return out
        }

        public func copyPlaneBytes(_ name: String, key: KeyPath<QWQTensorEntry, Int?>,
                                   to dst: UnsafeMutableRawPointer, count: Int) throws {
            let e = try entry(name)
            guard let base = e[keyPath: key] else { throw QWQuantError.tensorMissing(name) }
            guard base + count <= rawCount else { throw QWQuantError.truncated }
            dst.copyMemory(from: UnsafeRawPointer(rawBase).advanced(by: base), byteCount: count)
        }

        public func copyRawBytes(_ name: String, to dst: UnsafeMutableRawPointer,
                                 at elemOffset: Int, count elems: Int) throws {
            let e = try entry(name)
            guard let off = e.rawOff else { throw QWQuantError.notRawPlane(name) }
            let base = off + elemOffset * 2
            guard base + elems * 2 <= rawCount else { throw QWQuantError.truncated }
            dst.copyMemory(from: UnsafeRawPointer(rawBase).advanced(by: base), byteCount: elems * 2)
        }

        @inline(__always)
        public func withRawPointer<R>(at byteOffset: Int, byteCount: Int,
                                      _ body: (UnsafeRawPointer) throws -> R) throws -> R {
            guard byteOffset >= 0, byteCount >= 0, byteOffset + byteCount <= rawCount else {
                throw QWQuantError.truncated
            }
            return try body(UnsafeRawPointer(rawBase).advanced(by: byteOffset))
        }

        public func adviseDontNeed() {
            if rawCount > 0 {
                _ = madvise(rawBase, rawCount, MADV_DONTNEED)
            }
        }

        /// C30-T6: asynchronous kernel readahead over the pack mmap. Advances in
        /// Pins up to `limitBytes` of the pack mmap into RAM in 1 GiB chunks via `mlock`.
        @discardableResult
        public func wire(limitBytes: Int = .max) -> Int {
            guard wiredBytes == 0 else { return wiredBytes }
            var cap = Swift.max(0, limitBytes)
            var rl = rlimit()
            if getrlimit(RLIMIT_MEMLOCK, &rl) == 0, rl.rlim_cur < rlim_t(cap) {
                cap = Int(rl.rlim_cur)
            }
            let gran = 1 << 30
            var wired = 0
            let want = Swift.min(cap, rawCount)
            if want > 0 {
                _ = madvise(rawBase, want, MADV_WILLNEED)
            }
            while wired < want {
                let chunk = Swift.min(gran, want - wired)
                if mlock(UnsafeRawPointer(rawBase).advanced(by: wired), chunk) != 0 { break }
                wired += chunk
            }
            wiredBytes = wired
            return wired
        }
    }

    @inline(__always)
    public static func f16bits(_ bits: UInt16) -> Double {
        let sign: Double = (bits >> 15) & 1 == 1 ? -1 : 1
        let e = Int((bits >> 10) & 0x1F), m = Int(bits & 0x3FF)
        if e == 0 { return sign * Double(m) * pow(2.0, -24) }
        if e == 31 { return m != 0 ? .nan : sign * .infinity }
        return sign * (1.0 + Double(m) / 1024.0) * pow(2.0, Double(e - 15))
    }

    @inline(__always)
    public static func f16tof32Bits(_ h: UInt16) -> Float {
        let sign = UInt32(h & 0x8000) << 16
        let e = UInt32((h >> 10) & 0x1F)
        let m = UInt32(h & 0x3FF)
        if e == 0 {
            if m == 0 { return Float(bitPattern: sign) }
            var mb = m, mbits = 0
            while (mb & 0x400) == 0 { mb <<= 1; mbits += 1 }
            let exp = UInt32(113 - mbits)
            return Float(bitPattern: sign | (exp << 23) | ((mb << 13) & 0x7F_FFFF))
        }
        if e == 31 { return Float(bitPattern: sign | 0x7F80_0000 | (m << 13)) }
        return Float(bitPattern: sign | ((e &+ 112) << 23) | (m << 13))
    }

    @inline(__always)
    public static func f32tof16Bits(_ x: Float) -> UInt16 {
        Float16(x).bitPattern
    }

    @inline(__always)
    public static func bf16ToF16Array(
        src: UnsafePointer<UInt16>, dst: UnsafeMutablePointer<UInt16>, count: Int
    ) {
        var i = 0
        while i + 8 <= count {
            dst[i]     = Float16(Float(bitPattern: UInt32(src[i]) << 16)).bitPattern
            dst[i + 1] = Float16(Float(bitPattern: UInt32(src[i + 1]) << 16)).bitPattern
            dst[i + 2] = Float16(Float(bitPattern: UInt32(src[i + 2]) << 16)).bitPattern
            dst[i + 3] = Float16(Float(bitPattern: UInt32(src[i + 3]) << 16)).bitPattern
            dst[i + 4] = Float16(Float(bitPattern: UInt32(src[i + 4]) << 16)).bitPattern
            dst[i + 5] = Float16(Float(bitPattern: UInt32(src[i + 5]) << 16)).bitPattern
            dst[i + 6] = Float16(Float(bitPattern: UInt32(src[i + 6]) << 16)).bitPattern
            dst[i + 7] = Float16(Float(bitPattern: UInt32(src[i + 7]) << 16)).bitPattern
            i += 8
        }
        while i < count {
            dst[i] = Float16(Float(bitPattern: UInt32(src[i]) << 16)).bitPattern
            i += 1
        }
    }

    @inline(__always)
    public static func bf16tof32Bits(_ b: UInt16) -> Float {
        Float(bitPattern: UInt32(b) << 16)
    }

    @inline(never)
    public static func mulRn(_ a: Float, _ b: Float) -> Float { a * b }

    @inline(never)
    public static func addRn(_ a: Float, _ b: Float) -> Float { a + b }

    @inline(never)
    public static func subRn(_ a: Float, _ b: Float) -> Float { a - b }

    /// Two bytes, little-endian, as the producer wrote them.
    @inline(__always)
    public static func u16le(_ bytes: UnsafeRawPointer, _ offset: Int) -> UInt16 {
        UInt16(littleEndian: bytes.loadUnaligned(fromByteOffset: offset, as: UInt16.self))
    }

    /// Four bytes, little-endian, as the producer wrote them.
    @inline(__always)
    public static func u32le(_ bytes: UnsafeRawPointer, _ offset: Int) -> UInt32 {
        UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: offset, as: UInt32.self))
    }

    /// Decodes one `Block_HQ4_G32` (20 B) into 32 values, in the block's
    /// own domain: the magic-bias form and **no** Walsh-Hadamard step.
    /// The transform belongs to the weight form, which is multiplied; a
    /// row that is added to the residual stream is stored as it is.
    @inline(__always)
    public static func dequantHQ4Group(_ blk: UnsafeRawPointer,
                                       into dst: UnsafeMutablePointer<Float>) {
        let s = QWQuant.f16tof32Bits(QWQuant.u16le(blk, 0))
        let bias = QWQuant.f16tof32Bits(QWQuant.u16le(blk, 2))
        for j in 0..<4 {
            let qs = QWQuant.u32le(blk, 4 + 4 * j)
            let b = 8 * j
            dst[b + 0] = QWQuant.addRn(QWQuant.mulRn(Float((qs >> 0) & 0xF) + 1024.0, s), bias)
            dst[b + 1] = QWQuant.addRn(QWQuant.mulRn(Float((qs >> 16) & 0xF) + 1024.0, s), bias)
            dst[b + 2] = QWQuant.addRn(QWQuant.mulRn(Float((qs >> 4) & 0xF) + 1024.0, s), bias)
            dst[b + 3] = QWQuant.addRn(QWQuant.mulRn(Float((qs >> 20) & 0xF) + 1024.0, s), bias)
            dst[b + 4] = QWQuant.addRn(QWQuant.mulRn(Float((qs >> 8) & 0xF) + 1024.0, s), bias)
            dst[b + 5] = QWQuant.addRn(QWQuant.mulRn(Float((qs >> 24) & 0xF) + 1024.0, s), bias)
            dst[b + 6] = QWQuant.addRn(QWQuant.mulRn(Float((qs >> 12) & 0xF) + 1024.0, s), bias)
            dst[b + 7] = QWQuant.addRn(QWQuant.mulRn(Float((qs >> 28) & 0xF) + 1024.0, s), bias)
        }
    }

    public static func fwht32OrthoBlock(_ p: UnsafeMutablePointer<Float>) {
        var h = 1
        while h < 32 {
            var i0 = 0
            while i0 < 32 {
                for j in 0..<h {
                    let u = p[i0 + j]
                    let v = p[i0 + j + h]
                    p[i0 + j] = addRn(u, v)
                    p[i0 + j + h] = subRn(u, v)
                }
                i0 += 2 * h
            }
            h *= 2
        }
        let scale = invSqrt32
        for i in 0..<32 {
            p[i] = mulRn(p[i], scale)
        }
    }

    static func sha256Hex(_ s: String) -> String {
        SHA256.hash(data: Data(s.utf8)).map { String(format: "%02x", $0) }.joined()
    }
}
