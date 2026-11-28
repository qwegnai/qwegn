// AQRepack.swift — the one MLX-affine → family repacker.
//
// The checkpoint's payload is copied VERBATIM and the group's own scale and
// bias become the family's f16 header. A header f16 cannot hold exactly is
// REFUSED by name, and the first group is checked against the CPU's own view
// of the checkpoint. Both engines use this door; there is no second copy.

import Foundation
import Darwin
import Metal
import QWHAL
import QWQuant

/// The size-preserving family's repack of an MLX affine checkpoint.
public enum QWAQRepacker {

    /// The pair an MLX affine tensor was written with, and the dtype of its
    /// scale and bias headers.
    public struct Layout: Sendable, Equatable {
        public let bits: Int
        public let group: Int
        public let groups: Int          // groups per row
        public let scaleDType: UInt32   // 0 = bf16, 1 = f16
    }

    /// What the repacker refuses, by name.
    public enum RepackError: Error, CustomStringConvertible {
        case notAffine(name: String, dtype: String)
        case missingHeader(name: String, kind: String)
        case headerMismatch(name: String)
        case badLayout(name: String, reason: String)
        case pairMismatch(name: String, checkpoint: String, wanted: String)
        case headersNotExact(name: String, count: Int)
        case deviceUnavailable(reason: String)
        case repackFailed(name: String, message: String)

        public var description: String {
            switch self {
            case let .notAffine(name, dtype):
                return "the family's container needs an MLX affine weight, not \(dtype) (\(name))"
            case let .missingHeader(name, kind):
                return "the MLX \(kind) tensor is missing (\(name))"
            case let .headerMismatch(name):
                return "the scales and the biases differ in dtype (\(name))"
            case let .badLayout(name, reason):
                return "invalid MLX group layout (\(name)): \(reason)"
            case let .pairMismatch(name, checkpoint, wanted):
                return "the checkpoint's \(checkpoint) is not the wanted \(wanted) (\(name))"
            case let .headersNotExact(name, count):
                return "\(count) group header(s) do not survive the f16 conversion exactly (\(name))"
            case let .deviceUnavailable(reason):
                return "the repack cannot reach the device: \(reason)"
            case let .repackFailed(name, message):
                return "repack failed for \(name): \(message)"
            }
        }
    }

    /// The affine layout of an MLX checkpoint tensor: the pair it was written
    /// with, and the dtype of its scale and bias headers.
    public static func layout(container: QWSafeTensorsContainer, name: String,
                              rows: Int, cols: Int) throws -> Layout {
        guard let info = container.tensor(name) else {
            throw RepackError.notAffine(name: name, dtype: "missing")
        }
        guard info.dtype == .u32, name.hasSuffix(".weight") else {
            throw RepackError.notAffine(name: name, dtype: info.dtype.rawValue)
        }
        let stem = String(name.dropLast(".weight".count))
        guard let sInfo = container.tensor(stem + ".scales"), let sElems = sInfo.elementCount else {
            throw RepackError.missingHeader(name: stem + ".scales", kind: "scales")
        }
        guard let bInfo = container.tensor(stem + ".biases"), bInfo.elementCount == sElems else {
            throw RepackError.missingHeader(name: stem + ".biases", kind: "biases")
        }
        guard sInfo.dtype == bInfo.dtype else {
            throw RepackError.headerMismatch(name: name)
        }
        guard rows > 0, cols > 0, sElems % rows == 0 else {
            throw RepackError.badLayout(name: name, reason: "the scales do not divide into \(rows) rows")
        }
        let groups = sElems / rows
        guard groups > 0, cols % groups == 0 else {
            throw RepackError.badLayout(name: name, reason: "invalid MLX group layout")
        }
        let groupSize = cols / groups
        let packedCols = info.shape.count >= 2 ? info.shape[info.shape.count - 1] : (info.elementCount ?? 0) / rows
        let (packedElems, ovElems) = rows.multipliedReportingOverflow(by: packedCols)
        let (packedBits, ovBits) = packedCols.multipliedReportingOverflow(by: 32)
        guard packedCols > 0, !ovElems, info.elementCount == packedElems,
              !ovBits, packedBits % cols == 0 else {
            throw RepackError.badLayout(name: name, reason: "the packed columns do not divide into \(cols)")
        }
        let bits = packedBits / cols
        let scaleDType: UInt32
        switch sInfo.dtype {
        case .bf16: scaleDType = 0
        case .f16: scaleDType = 1
        default:
            throw RepackError.badLayout(name: name, reason: "the scales are \(sInfo.dtype.rawValue)")
        }
        return Layout(bits: bits, group: groupSize, groups: groups, scaleDType: scaleDType)
    }

    /// Repacks one MLX affine tensor into the family's container, on the GPU.
    /// The payload is the checkpoint's own bytes, COPIED VERBATIM; a header
    /// that does not survive the f16 conversion is refused.
    public static func repackMLXAffine(hal: QWHal, container: QWSafeTensorsContainer,
                                       name: String, rows: Int, cols: Int,
                                       bits: Int, group: Int,
                                       into dstBuffer: any MTLBuffer, dstByteOffset: Int) throws {
        try autoreleasepool {
            let layout = try layout(container: container, name: name, rows: rows, cols: cols)
            guard layout.bits == bits, layout.group == group else {
                throw RepackError.pairMismatch(name: name,
                                               checkpoint: "\(layout.bits)-bit group \(layout.group)",
                                               wanted: "\(bits)-bit group \(group)")
            }
            let stem = String(name.dropLast(".weight".count))
            let (payload, payloadOff) = try deviceBufferSlice(container: container, name: name, device: hal.device)
            let (scales, scalesOff) = try deviceBufferSlice(container: container, name: stem + ".scales", device: hal.device)
            let (biases, biasesOff) = try deviceBufferSlice(container: container, name: stem + ".biases", device: hal.device)
            let pso = try hal.pipeline("qwg_repack_mlx_affine_to_aq_m5")
            guard let flags = hal.device.makeBuffer(length: 4, options: .storageModeShared) else {
                throw RepackError.deviceUnavailable(reason: "cannot allocate the repack's flag word")
            }
            memset(flags.contents(), 0, 4)
            guard rows > 0, cols > 0, bits > 0, group > 0, dstByteOffset >= 0,
                  var rowsU = UInt32(exactly: rows),
                  var colsU = UInt32(exactly: cols),
                  var bitsU = UInt32(exactly: bits),
                  var groupU = UInt32(exactly: group) else {
                throw RepackError.badLayout(name: name, reason: "dimensions out of UInt32 range")
            }
            var dtypeU = layout.scaleDType
            guard let (cb, enc) = hal.begin() else {
                throw RepackError.deviceUnavailable(reason: "no command buffer")
            }
            enc.setComputePipelineState(pso)
            enc.setBuffer(payload, offset: payloadOff, index: 0)
            enc.setBuffer(scales, offset: scalesOff, index: 1)
            enc.setBuffer(biases, offset: biasesOff, index: 2)
            enc.setBuffer(dstBuffer, offset: dstByteOffset, index: 3)
            enc.setBytes(&rowsU, length: 4, index: 4)
            enc.setBytes(&colsU, length: 4, index: 5)
            enc.setBytes(&bitsU, length: 4, index: 6)
            enc.setBytes(&groupU, length: 4, index: 7)
            enc.setBytes(&dtypeU, length: 4, index: 8)
            enc.setBuffer(flags, offset: 0, index: 9)
            let groupsTotal = rows * layout.groups
            let tgs = min(1024, max(1, (groupsTotal + 255) / 256))
            enc.dispatchThreadgroups(MTLSize(width: tgs, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            hal.end(cb, enc)
            cb.waitUntilCompleted()
            if let e = cb.error {
                throw RepackError.repackFailed(name: name, message: e.localizedDescription)
            }
            let refused = flags.contents().load(as: UInt32.self)
            guard refused == 0 else {
                throw RepackError.headersNotExact(name: name, count: Int(refused))
            }
            try verifyFirstGroup(name: name, bits: bits, group: group, scaleDType: layout.scaleDType,
                                 container: container, into: dstBuffer, dstByteOffset: dstByteOffset)
        }
        let stem = String(name.dropLast(".weight".count))
        container.adviseDontNeed(of: name)
        container.adviseDontNeed(of: stem + ".scales")
        container.adviseDontNeed(of: stem + ".biases")
    }

    /// Repacks 3D MLX affine `switch_mlp.gate_proj` `[experts, inter, cols]` and
    /// `switch_mlp.up_proj` `[experts, inter, cols]` directly into `[experts, 2 * inter, cols]`
    /// interleaved AQ blocks in a single GPU dispatch.
    public static func repackMLXAffineSwitchGateUp(
        hal: QWHal, container: QWSafeTensorsContainer,
        gateName: String, upName: String,
        experts: Int, inter: Int, cols: Int,
        bits: Int, group: Int,
        into dstBuffer: any MTLBuffer, dstByteOffset: Int
    ) throws {
        try autoreleasepool {
            let (rows, ovRows) = experts.multipliedReportingOverflow(by: inter)
            guard experts > 0, inter > 0, cols > 0, bits > 0, group > 0, dstByteOffset >= 0, !ovRows else {
                throw RepackError.badLayout(name: gateName, reason: "invalid switch dimensions")
            }
            let gLayout = try layout(container: container, name: gateName, rows: rows, cols: cols)
            let uLayout = try layout(container: container, name: upName, rows: rows, cols: cols)
            guard gLayout.bits == bits, gLayout.group == group else {
                throw RepackError.pairMismatch(name: gateName,
                                               checkpoint: "\(gLayout.bits)-bit group \(gLayout.group)",
                                               wanted: "\(bits)-bit group \(group)")
            }
            guard uLayout == gLayout else {
                throw RepackError.pairMismatch(name: upName,
                                               checkpoint: "\(uLayout.bits)-bit group \(uLayout.group)",
                                               wanted: "\(gLayout.bits)-bit group \(gLayout.group)")
            }
            let gStem = String(gateName.dropLast(".weight".count))
            let uStem = String(upName.dropLast(".weight".count))
            let (gPay, gPayOff) = try deviceBufferSlice(container: container, name: gateName, device: hal.device)
            let (gSc, gScOff) = try deviceBufferSlice(container: container, name: gStem + ".scales", device: hal.device)
            let (gBi, gBiOff) = try deviceBufferSlice(container: container, name: gStem + ".biases", device: hal.device)
            let (uPay, uPayOff) = try deviceBufferSlice(container: container, name: upName, device: hal.device)
            let (uSc, uScOff) = try deviceBufferSlice(container: container, name: uStem + ".scales", device: hal.device)
            let (uBi, uBiOff) = try deviceBufferSlice(container: container, name: uStem + ".biases", device: hal.device)

            let pso = try hal.pipeline("qwg_repack_mlx_affine_switch_gate_up_to_aq_m5")
            guard let flags = hal.device.makeBuffer(length: 4, options: .storageModeShared) else {
                throw RepackError.deviceUnavailable(reason: "cannot allocate the repack's flag word")
            }
            memset(flags.contents(), 0, 4)
            guard var expertsU = UInt32(exactly: experts),
                  var interU = UInt32(exactly: inter),
                  var colsU = UInt32(exactly: cols),
                  var bitsU = UInt32(exactly: bits),
                  var groupU = UInt32(exactly: group) else {
                throw RepackError.badLayout(name: gateName, reason: "dimensions out of UInt32 range")
            }
            var dtypeU = gLayout.scaleDType
            guard let (cb, enc) = hal.begin() else {
                throw RepackError.deviceUnavailable(reason: "no command buffer")
            }
            enc.setComputePipelineState(pso)
            enc.setBuffer(gPay, offset: gPayOff, index: 0)
            enc.setBuffer(gSc, offset: gScOff, index: 1)
            enc.setBuffer(gBi, offset: gBiOff, index: 2)
            enc.setBuffer(uPay, offset: uPayOff, index: 3)
            enc.setBuffer(uSc, offset: uScOff, index: 4)
            enc.setBuffer(uBi, offset: uBiOff, index: 5)
            enc.setBuffer(dstBuffer, offset: dstByteOffset, index: 6)
            enc.setBytes(&expertsU, length: 4, index: 7)
            enc.setBytes(&interU, length: 4, index: 8)
            enc.setBytes(&colsU, length: 4, index: 9)
            enc.setBytes(&bitsU, length: 4, index: 10)
            enc.setBytes(&groupU, length: 4, index: 11)
            enc.setBytes(&dtypeU, length: 4, index: 12)
            enc.setBuffer(flags, offset: 0, index: 13)
            let groupsTotal = 2 * rows * gLayout.groups
            let tgs = min(1024, max(1, (groupsTotal + 255) / 256))
            enc.dispatchThreadgroups(MTLSize(width: tgs, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            hal.end(cb, enc)
            cb.waitUntilCompleted()
            if let e = cb.error {
                throw RepackError.repackFailed(name: gateName, message: e.localizedDescription)
            }
            let refused = flags.contents().load(as: UInt32.self)
            guard refused == 0 else {
                throw RepackError.headersNotExact(name: gateName, count: Int(refused))
            }
            try verifyFirstGroup(name: gateName, bits: bits, group: group, scaleDType: gLayout.scaleDType,
                                 container: container, into: dstBuffer, dstByteOffset: dstByteOffset)
        }
        let gStem = String(gateName.dropLast(".weight".count))
        let uStem = String(upName.dropLast(".weight".count))
        container.adviseDontNeed(of: gateName)
        container.adviseDontNeed(of: gStem + ".scales")
        container.adviseDontNeed(of: gStem + ".biases")
        container.adviseDontNeed(of: upName)
        container.adviseDontNeed(of: uStem + ".scales")
        container.adviseDontNeed(of: uStem + ".biases")
    }

    /// Page-rounded `bytesNoCopy` Metal buffer + sub-page byte offset over a tensor in `container`
    /// when its enclosing shard covers the page-aligned span; falls back to `deviceBuffer` at the tail.
    static func deviceBufferSlice(container: QWSafeTensorsContainer,
                                  name: String,
                                  device: any MTLDevice) throws -> (any MTLBuffer, Int) {
        try container.withShardRange(of: name) { shard, shardRaw, range in
            guard let shardBase = shardRaw.baseAddress, range.count > 0 else {
                throw RepackError.deviceUnavailable(reason: "empty checkpoint tensor")
            }
            let pageSize = Int(sysconf(_SC_PAGESIZE))
            if pageSize > 0, Int(bitPattern: shardBase) % pageSize == 0 {
                let alignedStart = (range.lowerBound / pageSize) * pageSize
                let subOffset = range.lowerBound - alignedStart
                let span = subOffset + range.count
                let alignedLen = ((span + pageSize - 1) / pageSize) * pageSize
                if alignedStart + alignedLen <= shardRaw.count {
                    let regionPtr = shardBase.advanced(by: alignedStart)
                    if let buf = device.makeBuffer(
                        bytesNoCopy: UnsafeMutableRawPointer(mutating: regionPtr),
                        length: alignedLen, options: .storageModeShared,
                        deallocator: { _, _ in withExtendedLifetime(shard) {} }) {
                        QWResidencyRegistry.note(buf)
                        return (buf, subOffset)
                    }
                }
            }
            let slice = UnsafeRawBufferPointer(rebasing: shardRaw[range])
            return (try deviceBuffer(over: slice, keepingAlive: container, device: device), 0)
        }
    }

    /// A device buffer over the checkpoint's own bytes, with no copy when the
    /// platform will map the file and a copy when it will not. The deallocator
    /// keeps the owner alive, so the buffer's life bounds the mapping's.
    static func deviceBuffer(over raw: UnsafeRawBufferPointer,
                             keepingAlive owner: QWSafeTensorsContainer,
                             device: any MTLDevice) throws -> any MTLBuffer {
        guard let base = raw.baseAddress, raw.count > 0 else {
            throw RepackError.deviceUnavailable(reason: "empty checkpoint tensor")
        }
        let pageSize = Int(sysconf(_SC_PAGESIZE))
        if pageSize > 0,
           Int(bitPattern: base) % pageSize == 0,
           raw.count % pageSize == 0,
           let buf = device.makeBuffer(
               bytesNoCopy: UnsafeMutableRawPointer(mutating: base),
               length: raw.count, options: .storageModeShared,
               deallocator: { _, _ in withExtendedLifetime(owner) {} }) {
            QWResidencyRegistry.note(buf)
            return buf
        }
        guard let buf = device.makeBuffer(length: raw.count, options: .storageModeShared) else {
            throw RepackError.deviceUnavailable(reason: "cannot allocate a device buffer")
        }
        let dstPtr = buf.contents()
        withExtendedLifetime(buf) {
            dstPtr.copyMemory(from: base, byteCount: raw.count)
        }
        return buf
    }

    /// The first group of the first row, against the CPU's own view of the
    /// checkpoint. A mapping the device did not take would show up here.
    private static func verifyFirstGroup(name: String, bits: Int, group: Int, scaleDType: UInt32,
                                         container: QWSafeTensorsContainer,
                                         into dstBuffer: any MTLBuffer, dstByteOffset: Int) throws {
        let payloadLen = group * bits / 8
        let stem = String(name.dropLast(".weight".count))
        let header = try container.withBytes(of: stem + ".scales") { raw -> UInt16 in
            guard raw.count >= 2 else { throw RepackError.badLayout(name: name, reason: "empty scales") }
            return UInt16(littleEndian: raw.loadUnaligned(as: UInt16.self))
        }
        let bias = try container.withBytes(of: stem + ".biases") { raw -> UInt16 in
            guard raw.count >= 2 else { throw RepackError.badLayout(name: name, reason: "empty biases") }
            return UInt16(littleEndian: raw.loadUnaligned(as: UInt16.self))
        }
        let scaleF: Float = scaleDType == 0
            ? Float(bitPattern: UInt32(header) << 16)
            : Float(Float16(bitPattern: header))
        let biasF: Float = scaleDType == 0
            ? Float(bitPattern: UInt32(bias) << 16)
            : Float(Float16(bitPattern: bias))
        let scaleH = Float16(scaleF)
        let biasH = Float16(biasF)
        let kF16NormalMin: Float = 6.103515625e-5
        let scaleOk = (Float(scaleH) == scaleF) || (scaleF.isFinite && scaleH.isFinite && abs(scaleF) < kF16NormalMin)
        let biasOk = (Float(biasH) == biasF) || (biasF.isFinite && biasH.isFinite && abs(biasF) < kF16NormalMin)
        guard scaleOk && biasOk else {
            throw RepackError.headersNotExact(name: name, count: 1)
        }
        let payload = try container.withBytes(of: name) { raw -> [UInt8] in
            guard raw.count >= payloadLen else {
                throw RepackError.badLayout(name: name, reason: "short payload")
            }
            return Array(raw.prefix(payloadLen))
        }
        let (endOff, ovEnd) = dstByteOffset.addingReportingOverflow(4 + payloadLen)
        guard dstByteOffset >= 0, !ovEnd, endOff <= dstBuffer.length else {
            throw RepackError.repackFailed(name: name, message: "destination buffer smaller than first group")
        }
        let dstRaw = UnsafeRawPointer(dstBuffer.contents().advanced(by: dstByteOffset))
        let gotScale = Float16(bitPattern: UInt16(littleEndian: dstRaw.loadUnaligned(fromByteOffset: 0, as: UInt16.self)))
        let gotBias = Float16(bitPattern: UInt16(littleEndian: dstRaw.loadUnaligned(fromByteOffset: 2, as: UInt16.self)))
        guard gotScale == scaleH, gotBias == biasH else {
            throw RepackError.repackFailed(name: name, message: "the repacked header does not match")
        }
        for (i, byte) in payload.enumerated() where dstRaw.loadUnaligned(fromByteOffset: 4 + i, as: UInt8.self) != byte {
            throw RepackError.repackFailed(name: name, message: "the repacked payload does not match")
        }
    }
}
