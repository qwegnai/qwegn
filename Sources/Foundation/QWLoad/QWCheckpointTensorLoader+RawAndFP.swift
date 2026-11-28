// QWCheckpointTensorLoader+RawAndFP.swift — Repack staging and stacked MoE bank materialization.
import Darwin
import Foundation
import Metal
import QWAlloc
import QWCore
import QWHAL
import QWQuant

extension QWCheckpointTensorLoader {
    private static let stageLock = NSLock()
    nonisolated(unsafe) private static var sharedRepackStage: (any MTLBuffer)?

    private static func withRepackStage<R>(
        device: any MTLDevice,
        minBytes: Int,
        _ body: (any MTLBuffer) throws -> R
    ) throws -> R? {
        guard minBytes > 0 else { return nil }
        return try stageLock.withLock {
            if let existing = sharedRepackStage, existing.device === device, existing.length >= minBytes {
                return try body(existing)
            }
            sharedRepackStage = nil
            guard let buf = device.makeBuffer(length: minBytes, options: .storageModeShared) else {
                return nil
            }
            sharedRepackStage = buf
            return try body(buf)
        }
    }

    public static func releaseRepackStageBuffer() {
        stageLock.withLock {
            sharedRepackStage = nil
        }
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
        var keyStr = "\(stPath)|\(size)|\(Int64(mtime))|\(st.tensors.count)|\(isRotated ? "rot" : "std")"
        if st.shards.count > 1 {
            let shardMeta = st.shards.keys.sorted { $0.path < $1.path }.map { url -> String in
                let a = try? fm.attributesOfItem(atPath: url.path)
                let s = (a?[.size] as? NSNumber)?.uint64Value ?? 0
                let m = Int64((a?[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0)
                return "\(url.lastPathComponent):\(s):\(m)"
            }.joined(separator: ",")
            keyStr += "|\(shardMeta)"
        }
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

    private static func mapCachedMoESlab(
        at url: URL, alignedGU: Int, alignedDN: Int, device: any MTLDevice
    ) -> (guBuf: any MTLBuffer, dnBuf: any MTLBuffer)? {
        let pageSize = Int(sysconf(_SC_PAGESIZE))
        let (totalBytes, ov) = alignedGU.addingReportingOverflow(alignedDN)
        guard pageSize > 0, alignedGU > 0, alignedDN > 0, !ov,
              alignedGU % pageSize == 0, alignedDN % pageSize == 0 else { return nil }
        let fd = Darwin.open(url.path, O_RDONLY | O_CLOEXEC)
        guard fd >= 0 else { return nil }
        defer { Darwin.close(fd) }
        var st = Darwin.stat()
        guard Darwin.fstat(fd, &st) == 0, let stSize = Int(exactly: st.st_size), stSize == totalBytes else { return nil }
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
        guard fd >= 0, count >= 0, offset >= 0 else { return false }
        let (_, ov) = offset.addingReportingOverflow(count)
        guard !ov else { return false }
        let chunkLimit = 64 << 20
        var written = 0
        while written < count {
            let chunk = min(chunkLimit, count - written)
            let w = Darwin.pwrite(fd, src.advanced(by: written), chunk, off_t(offset + written))
            if w < 0 {
                if errno == EINTR { continue }
                return false
            }
            if w == 0 { return false }
            written += w
        }
        return true
    }

    private static func fsyncFD(_ fd: Int32) -> Bool {
        while true {
            if Darwin.fsync(fd) == 0 { return true }
            if errno == EINTR { continue }
            return false
        }
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
        guard let st, experts > 0, inter > 0, hidden > 0 else {
            throw QWCheckpointLoaderError.notHQ4(mp)
        }
        let (expInter, ovEI) = experts.multipliedReportingOverflow(by: inter)
        let (guRows, ovGU) = expInter.multipliedReportingOverflow(by: 2)
        let (dnRows, ovDN) = experts.multipliedReportingOverflow(by: hidden)
        guard !ovEI, !ovGU, !ovDN else { throw QWCheckpointLoaderError.notHQ4(mp) }
        let downName = st.tensor(mp + "experts.down_proj") != nil
            ? (mp + "experts.down_proj") : (mp + "switch_mlp.down_proj.weight")
        let pageSize = max(4096, Int(sysconf(_SC_PAGESIZE)))

        if !arena.hal.weightBasis.rotatesActivations {
            let isFusedGateUp = st.tensor(mp + "experts.gate_up_proj") != nil
            let guLayout: QWAQRepacker.Layout? = isFusedGateUp
                ? (try? QWAQRepacker.layout(container: st, name: mp + "experts.gate_up_proj", rows: guRows, cols: hidden))
                : (try? QWAQRepacker.layout(container: st, name: mp + "switch_mlp.gate_proj.weight", rows: expInter, cols: hidden))
            let upLayout: QWAQRepacker.Layout? = isFusedGateUp
                ? guLayout
                : (try? QWAQRepacker.layout(container: st, name: mp + "switch_mlp.up_proj.weight", rows: expInter, cols: hidden))
            let dnLayout = try? QWAQRepacker.layout(container: st, name: downName, rows: dnRows, cols: inter)
            if guLayout != nil && (upLayout != guLayout || dnLayout == nil) {
                throw QWCheckpointLoaderError.notHQ4(mp + "switch_mlp")
            }
            if let guLayout, let dnLayout,
               QWAQBlock.isSupported(bits: guLayout.bits, group: guLayout.group),
               QWAQBlock.isSupported(bits: dnLayout.bits, group: dnLayout.group) {
                let guRowBytes = QWAQBlock.rowBytes(bits: guLayout.bits, group: guLayout.group, k: hidden)
                let (guBytes, ovGUB) = guRows.multipliedReportingOverflow(by: guRowBytes)
                let dnRowBytes = QWAQBlock.rowBytes(bits: dnLayout.bits, group: dnLayout.group, k: inter)
                let (dnBytes, ovDNB) = dnRows.multipliedReportingOverflow(by: dnRowBytes)
                let (guPad, ovGUP) = guBytes.addingReportingOverflow(pageSize - 1)
                let (dnPad, ovDNP) = dnBytes.addingReportingOverflow(pageSize - 1)
                let (totalAQBytes, ovTot) = guBytes.addingReportingOverflow(dnBytes)
                guard !ovGUB, !ovDNB, !ovGUP, !ovDNP, !ovTot, guBytes > 0, dnBytes > 0,
                      guBytes % 2 == 0, dnBytes % 2 == 0 else {
                    throw QWCheckpointLoaderError.notHQ4(mp + "switch_mlp")
                }
                let alignedGU = (guPad / pageSize) * pageSize
                let alignedDN = (dnPad / pageSize) * pageSize

                if totalAQBytes >= (16 << 20),
                   let cacheDir = moeSlabCacheDirectory(for: st, isRotated: false) {
                    let sanitizedMP = mp.replacingOccurrences(of: ".", with: "_").replacingOccurrences(of: "/", with: "_")
                    let slabName = "\(sanitizedMP)gu\(guLayout.bits)g\(guLayout.group)_dn\(dnLayout.bits)g\(dnLayout.group)_\(guRows)x\(hidden)_\(dnRows)x\(inter).slab"
                    let slabURL = cacheDir.appendingPathComponent(slabName)
                    if let mapped = mapCachedMoESlab(at: slabURL, alignedGU: alignedGU, alignedDN: alignedDN, device: arena.hal.device) {
                        let guTensor = QWTensorValue(region: QWMRegion(buffer: mapped.guBuf, base: 0, length: guBytes),
                                                     offset: 0, shape: QWTensorShape([guBytes / 2], .f16))
                        let dnTensor = QWTensorValue(region: QWMRegion(buffer: mapped.dnBuf, base: 0, length: dnBytes),
                                                     offset: 0, shape: QWTensorShape([dnBytes / 2], .f16))
                        return (try makeWeightView(blocks: guTensor, n: guRows, k: hidden,
                                                   bits: guLayout.bits, groupSize: guLayout.group,
                                                   isRotated: false, name: mp + "experts.gate_up_proj"),
                                try makeWeightView(blocks: dnTensor, n: dnRows, k: inter,
                                                   bits: dnLayout.bits, groupSize: dnLayout.group,
                                                   isRotated: false, name: downName))
                    }
                    let tmpURL = slabURL.appendingPathExtension("tmp.\(getpid()).\(UUID().uuidString)")
                    var tmpFD = Darwin.open(tmpURL.path, O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0o600)
                    if tmpFD >= 0 {
                        var renameDone = false
                        defer {
                            if tmpFD >= 0 {
                                _ = Darwin.close(tmpFD)
                                tmpFD = -1
                            }
                            if !renameDone {
                                _ = Darwin.unlink(tmpURL.path)
                            }
                        }
                        _ = Darwin.fcntl(tmpFD, F_NOCACHE, 1)
                        let (totalAligned, ovAligned) = alignedGU.addingReportingOverflow(alignedDN)
                        if !ovAligned, Darwin.ftruncate(tmpFD, off_t(totalAligned)) == 0 {
                            let writeFD = tmpFD
                            let wroteAll: Bool = try withRepackStage(
                                device: arena.hal.device,
                                minBytes: max(alignedGU, alignedDN)
                            ) { stageBuf in
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
                                let stagePtr = stageBuf.contents()
                                let guOk = withExtendedLifetime(stageBuf) {
                                    pwriteAllNoCache(fd: writeFD, from: stagePtr, count: guBytes, offset: 0)
                                }
                                guard guOk else { return false }
                                try QWAQRepacker.repackMLXAffine(
                                    hal: arena.hal, container: st, name: downName,
                                    rows: dnRows, cols: inter, bits: dnLayout.bits, group: dnLayout.group,
                                    into: stageBuf, dstByteOffset: 0)
                                return withExtendedLifetime(stageBuf) {
                                    pwriteAllNoCache(fd: writeFD, from: stagePtr, count: dnBytes, offset: alignedGU)
                                }
                            } ?? false
                            if wroteAll, fsyncFD(tmpFD) {
                                _ = Darwin.close(tmpFD)
                                tmpFD = -1
                                if Darwin.rename(tmpURL.path, slabURL.path) == 0 {
                                    renameDone = true
                                    if let mapped = mapCachedMoESlab(at: slabURL, alignedGU: alignedGU, alignedDN: alignedDN, device: arena.hal.device) {
                                        let guTensor = QWTensorValue(region: QWMRegion(buffer: mapped.guBuf, base: 0, length: guBytes),
                                                                     offset: 0, shape: QWTensorShape([guBytes / 2], .f16))
                                        let dnTensor = QWTensorValue(region: QWMRegion(buffer: mapped.dnBuf, base: 0, length: dnBytes),
                                                                     offset: 0, shape: QWTensorShape([dnBytes / 2], .f16))
                                        return (try makeWeightView(blocks: guTensor, n: guRows, k: hidden,
                                                                   bits: guLayout.bits, groupSize: guLayout.group,
                                                                   isRotated: false, name: mp + "experts.gate_up_proj"),
                                                try makeWeightView(blocks: dnTensor, n: dnRows, k: inter,
                                                                   bits: dnLayout.bits, groupSize: dnLayout.group,
                                                                   isRotated: false, name: downName))
                                    }
                                }
                            }
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
                return (try makeWeightView(blocks: guTensor, n: guRows, k: hidden,
                                           bits: guLayout.bits, groupSize: guLayout.group,
                                           isRotated: false, name: mp + "experts.gate_up_proj"),
                        try makeWeightView(blocks: dnTensor, n: dnRows, k: inter,
                                           bits: dnLayout.bits, groupSize: dnLayout.group,
                                           isRotated: false, name: downName))
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
        return (try makeWeightView(blocks: guTensor, n: guRows, k: hidden, isRotated: rot, name: mp + "experts.gate_up_proj"),
                try makeWeightView(blocks: dnTensor, n: dnRows, k: inter, isRotated: rot, name: downName))
    }
}
