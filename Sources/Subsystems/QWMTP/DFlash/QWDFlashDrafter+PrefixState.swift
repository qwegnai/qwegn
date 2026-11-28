// QWDFlashDrafter+PrefixState.swift — Shared binary prefix KV state serialization
// and draft-vocab resolution for DFlash 2 / DSpine drafters.
import Foundation
import Metal
import QWAlloc
import QWCore

public struct QWDFlashPrefixState: Sendable, Equatable {
    public static let magic: UInt64 = 0x3156_4b46_4457_5151

    public var magic: UInt64
    public var numLayers: UInt32
    public var activeRows: UInt32
    public var geometryHash: UInt64
    public var cacheOffset: UInt64
    public var rowBytes: UInt32
    public var reserved: UInt32

    public init(
        magic: UInt64 = Self.magic,
        numLayers: UInt32,
        activeRows: UInt32,
        geometryHash: UInt64,
        cacheOffset: UInt64,
        rowBytes: UInt32,
        reserved: UInt32 = 0
    ) {
        self.magic = magic
        self.numLayers = numLayers
        self.activeRows = activeRows
        self.geometryHash = geometryHash
        self.cacheOffset = cacheOffset
        self.rowBytes = rowBytes
        self.reserved = reserved
    }
}

extension QWDFlashDrafter {
    /// Draft vocab rows for DFlash proposals (`QWG_DRAFT_VOCAB` override, or `defaultCap` when `< fullVocab`).
    public static func resolveDraftVocab(fullVocab: Int, defaultCap: Int? = nil) -> Int? {
        if let raw = ProcessInfo.processInfo.environment["QWG_DRAFT_VOCAB"]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           let value = Int(raw) {
            return (value > 0 && value < fullVocab) ? value : nil
        }
        if let defaultCap, defaultCap > 0, defaultCap < fullVocab {
            return defaultCap
        }
        return nil
    }

    private static func slotRowBytes(config: QWDFlashConfig) -> (k: Int, v: Int, headerRowBytes: Int) {
        if config.isMLA {
            let kBytes = config.kvLoraRank * 2
            let vBytes = config.qkRopeHeadDim * 2
            return (kBytes, vBytes, kBytes + vBytes)
        }
        let rb = config.numKeyValueHeads * config.headDim * 2
        return (rb, rb, rb)
    }

    public static func prefixGeometryMarker(config: QWDFlashConfig) -> String {
        let layers = config.targetLayerIds.map(String.init).joined(separator: ",")
        let base = "df:\(config.hiddenSize):\(config.numLayers):\(config.numAttentionHeads):"
            + "\(config.numKeyValueHeads):\(config.headDim):\(config.maxCacheSize):"
            + "\(config.isDSpine ? 1 : 0):\(layers)"
        if config.isMLA {
            return base + ":mla:\(config.kvLoraRank):\(config.qkRopeHeadDim)"
        }
        return base
    }

    public func prefixGeometryMarker() -> String {
        Self.prefixGeometryMarker(config: config)
    }

    public static func prefixGeometryHash(_ marker: String) -> UInt64 {
        marker.utf8.reduce(0xcbf2_9ce4_8422_2325) { ($0 ^ UInt64($1)) &* 0x1000_0000_01b3 }
    }

    public static func prefixPayloadBytes(config: QWDFlashConfig, tokenCount: Int) -> Int {
        let activeRows = min(max(0, tokenCount), config.maxCacheSize)
        let rb = slotRowBytes(config: config)
        return MemoryLayout<QWDFlashPrefixState>.size + config.numLayers * activeRows * (rb.k + rb.v)
    }

    private static func readPrefixHeader(from src: UnsafeRawPointer) -> QWDFlashPrefixState {
        var hdr = QWDFlashPrefixState(
            magic: 0, numLayers: 0, activeRows: 0,
            geometryHash: 0, cacheOffset: 0, rowBytes: 0, reserved: 0
        )
        withUnsafeMutableBytes(of: &hdr) { (dst: UnsafeMutableRawBufferPointer) in
            if let dstBase = dst.baseAddress {
                dstBase.copyMemory(from: src, byteCount: MemoryLayout<QWDFlashPrefixState>.size)
            }
        }
        return hdr
    }

    private static func writePrefixHeader(
        to dstBase: UnsafeMutableRawPointer,
        at writeOffset: inout Int,
        config: QWDFlashConfig,
        marker: String,
        tokenCount: Int
    ) {
        let clampedTokens = max(0, tokenCount)
        let activeRows = min(clampedTokens, config.maxCacheSize)
        let rb = slotRowBytes(config: config)
        var hdr = QWDFlashPrefixState(
            numLayers: UInt32(config.numLayers),
            activeRows: UInt32(activeRows),
            geometryHash: prefixGeometryHash(marker),
            cacheOffset: UInt64(clampedTokens),
            rowBytes: UInt32(rb.headerRowBytes)
        )
        withUnsafeBytes(of: &hdr) { (buf: UnsafeRawBufferPointer) in
            if let src = buf.baseAddress {
                dstBase.advanced(by: writeOffset).copyMemory(from: src, byteCount: buf.count)
                writeOffset += buf.count
            }
        }
    }

    private static func hoistSourcePointers(
        slot: QWDFlashSessionSlot,
        numLayers: Int,
        kBytes: Int,
        vBytes: Int
    ) -> [(UnsafeRawPointer, any MTLBuffer, Int)] {
        guard slot.kLayers.count >= numLayers, slot.vLayers.count >= numLayers else { return [] }
        return (0..<numLayers).flatMap { l -> [(UnsafeRawPointer, any MTLBuffer, Int)] in
            let kt = slot.kLayers[l]
            let vt = slot.vLayers[l]
            return [
                (UnsafeRawPointer(kt.region.buffer.contents().advanced(by: kt.region.base + kt.offset)), kt.region.buffer, kBytes),
                (UnsafeRawPointer(vt.region.buffer.contents().advanced(by: vt.region.base + vt.offset)), vt.region.buffer, vBytes)
            ]
        }
    }

    private static func hoistDestinationPointers(
        slot: QWDFlashSessionSlot,
        numLayers: Int,
        kBytes: Int,
        vBytes: Int
    ) -> [(UnsafeMutableRawPointer, any MTLBuffer, Int)] {
        guard slot.kLayers.count >= numLayers, slot.vLayers.count >= numLayers else { return [] }
        return (0..<numLayers).flatMap { l -> [(UnsafeMutableRawPointer, any MTLBuffer, Int)] in
            let kt = slot.kLayers[l]
            let vt = slot.vLayers[l]
            return [
                (kt.region.buffer.contents().advanced(by: kt.region.base + kt.offset), kt.region.buffer, kBytes),
                (vt.region.buffer.contents().advanced(by: vt.region.base + vt.offset), vt.region.buffer, vBytes)
            ]
        }
    }

    private static func restoreFromHoisted(
        base: UnsafeRawPointer,
        dataCount: Int,
        offset: inout Int,
        dstLayers: [(UnsafeMutableRawPointer, any MTLBuffer, Int)],
        config: QWDFlashConfig,
        expectedMarker: String,
        expectedTokens: Int
    ) -> Int? {
        let hdrSize = MemoryLayout<QWDFlashPrefixState>.size
        guard expectedTokens >= 0, offset >= 0, offset + hdrSize <= dataCount else { return nil }
        let hdr = readPrefixHeader(from: base.advanced(by: offset))
        offset += hdrSize
        let active = min(expectedTokens, config.maxCacheSize)
        let rb = slotRowBytes(config: config)
        let payloadBytes = config.numLayers * active * (rb.k + rb.v)
        guard hdr.magic == QWDFlashPrefixState.magic,
              hdr.geometryHash == prefixGeometryHash(expectedMarker),
              Int(hdr.numLayers) == config.numLayers,
              Int(hdr.cacheOffset) == expectedTokens,
              Int(hdr.activeRows) == active,
              Int(hdr.rowBytes) == rb.headerRowBytes,
              offset + payloadBytes <= dataCount else {
            return nil
        }
        if payloadBytes > 0 {
            guard dstLayers.count == 2 * config.numLayers else { return nil }
            for (dst, buf, copyBytes) in dstLayers where copyBytes > 0 {
                withExtendedLifetime(buf) {
                    dst.copyMemory(from: base.advanced(by: offset), byteCount: copyBytes)
                }
                offset += copyBytes
            }
        }
        return Int(hdr.cacheOffset)
    }

    public static func appendPrefixKV(
        to dstBase: UnsafeMutableRawPointer,
        at writeOffset: inout Int,
        slot: QWDFlashSessionSlot?,
        config: QWDFlashConfig,
        marker: String,
        tokenCount: Int
    ) {
        let activeRows = min(max(0, tokenCount), config.maxCacheSize)
        let rb = slotRowBytes(config: config)
        let kLayerBytes = activeRows * rb.k
        let vLayerBytes = activeRows * rb.v
        writePrefixHeader(to: dstBase, at: &writeOffset, config: config, marker: marker, tokenCount: tokenCount)
        guard kLayerBytes + vLayerBytes > 0, let slot else { return }
        let srcLayers = hoistSourcePointers(slot: slot, numLayers: config.numLayers, kBytes: kLayerBytes, vBytes: vLayerBytes)
        guard srcLayers.count == 2 * config.numLayers else { return }
        withExtendedLifetime(slot) {
            for (src, buf, copyBytes) in srcLayers where copyBytes > 0 {
                withExtendedLifetime(buf) {
                    dstBase.advanced(by: writeOffset).copyMemory(from: src, byteCount: copyBytes)
                }
                writeOffset += copyBytes
            }
        }
    }

    public static func restorePrefixKV(
        from base: UnsafeRawPointer,
        dataCount: Int,
        offset: inout Int,
        slot: QWDFlashSessionSlot?,
        config: QWDFlashConfig,
        expectedMarker: String,
        expectedTokens: Int
    ) -> Int? {
        guard let slot else { return nil }
        let activeRows = min(max(0, expectedTokens), config.maxCacheSize)
        let rb = slotRowBytes(config: config)
        let dstLayers = hoistDestinationPointers(
            slot: slot, numLayers: config.numLayers, kBytes: activeRows * rb.k, vBytes: activeRows * rb.v
        )
        return withExtendedLifetime(slot) {
            restoreFromHoisted(
                base: base,
                dataCount: dataCount,
                offset: &offset,
                dstLayers: dstLayers,
                config: config,
                expectedMarker: expectedMarker,
                expectedTokens: expectedTokens
            )
        }
    }

    public static func validatePrefixBlob(
        data: Data,
        expectedMarker: String,
        expectedTokens: Int,
        config: QWDFlashConfig
    ) -> Bool {
        guard expectedTokens >= 0, config.numLayers > 0 else { return false }
        let start = data.count - prefixPayloadBytes(config: config, tokenCount: expectedTokens)
        guard start >= 0 else { return false }
        let active = min(expectedTokens, config.maxCacheSize)
        let rb = slotRowBytes(config: config)
        return data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> Bool in
            guard let base = raw.baseAddress else { return false }
            let hdr = readPrefixHeader(from: base.advanced(by: start))
            return hdr.magic == QWDFlashPrefixState.magic
                && hdr.geometryHash == prefixGeometryHash(expectedMarker)
                && Int(hdr.numLayers) == config.numLayers
                && Int(hdr.cacheOffset) == expectedTokens
                && Int(hdr.activeRows) == active
                && Int(hdr.rowBytes) == rb.headerRowBytes
        }
    }

    public func capturePrefixBlob(slot: QWDFlashSessionSlot, tokenCount: Int) -> Data {
        let totalBytes = Self.prefixPayloadBytes(config: config, tokenCount: tokenCount)
        let activeRows = min(max(0, tokenCount), config.maxCacheSize)
        let rb = Self.slotRowBytes(config: config)
        let kLayerBytes = activeRows * rb.k
        let vLayerBytes = activeRows * rb.v
        let srcLayers = (kLayerBytes + vLayerBytes > 0)
            ? Self.hoistSourcePointers(slot: slot, numLayers: config.numLayers, kBytes: kLayerBytes, vBytes: vLayerBytes)
            : []
        let marker = prefixGeometryMarker()
        var data = Data(count: totalBytes)
        var writeOffset = 0
        withExtendedLifetime(slot) {
            data.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
                guard let dstBase = raw.baseAddress else { return }
                Self.writePrefixHeader(
                    to: dstBase,
                    at: &writeOffset,
                    config: config,
                    marker: marker,
                    tokenCount: tokenCount
                )
                guard kLayerBytes + vLayerBytes > 0, srcLayers.count == 2 * config.numLayers else { return }
                for (src, buf, copyBytes) in srcLayers where copyBytes > 0 {
                    withExtendedLifetime(buf) {
                        dstBase.advanced(by: writeOffset).copyMemory(from: src, byteCount: copyBytes)
                    }
                    writeOffset += copyBytes
                }
            }
        }
        return data
    }

    public func restorePrefixBlob(_ data: Data, into slot: QWDFlashSessionSlot, expectedTokens: Int) -> Int? {
        guard expectedTokens >= 0,
              data.count == Self.prefixPayloadBytes(config: config, tokenCount: expectedTokens) else {
            return nil
        }
        let activeRows = min(expectedTokens, config.maxCacheSize)
        let rb = Self.slotRowBytes(config: config)
        let dstLayers = Self.hoistDestinationPointers(
            slot: slot, numLayers: config.numLayers, kBytes: activeRows * rb.k, vBytes: activeRows * rb.v
        )
        let marker = prefixGeometryMarker()
        return withExtendedLifetime(slot) {
            data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> Int? in
                guard let base = raw.baseAddress else { return nil }
                var offset = 0
                guard let restored = Self.restoreFromHoisted(
                    base: base,
                    dataCount: raw.count,
                    offset: &offset,
                    dstLayers: dstLayers,
                    config: config,
                    expectedMarker: marker,
                    expectedTokens: expectedTokens
                ), offset == raw.count else {
                    return nil
                }
                return restored
            }
        }
    }
}
