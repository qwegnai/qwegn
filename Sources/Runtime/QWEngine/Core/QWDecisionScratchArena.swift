// QWDecisionScratchArena.swift — Reusable Metal scratch buffers and cached RoPE tables for decision inference.

import Foundation
import Metal
import QWAlloc
import QWCore
import QWDecision
import QWHAL
import QWRope

final class QWDecisionScratchArena: @unchecked Sendable {
    private final class BufferSlot {
        var buffer: (any MTLBuffer)?
        var capacity: Int = 0

        func ensure(hal: QWHal, bytes: Int) throws -> any MTLBuffer {
            let aligned = max(64, qwAlignUp(bytes, 4096))
            if let existing = buffer, capacity >= aligned {
                return existing
            }
            guard let fresh = hal.device.makeBuffer(length: aligned, options: .storageModeShared) else {
                throw QWDecisionError.backendFailed("failed to allocate Metal scratch buffer of \(aligned) bytes")
            }
            buffer = fresh
            capacity = aligned
            return fresh
        }
    }

    private let hal: QWHal
    private var slots: [String: BufferSlot] = [:]
    private var ropeCapacitySeqLen: Int = 0
    private var cachedRopeTables: [String: (table: QWTensorValue, rotaryDims: Int, zeroTailArg: Bool)] = [:]
    private var retainedRopeBuffers: [any MTLBuffer] = []

    init(hal: QWHal) {
        self.hal = hal
    }

    func tensor(key: String, shape: [Int], dtype: QWDType, zeroFill: Bool = false) throws -> QWTensorValue {
        let tShape = QWTensorShape(shape, dtype)
        let slot = slots[key] ?? {
            let created = BufferSlot()
            slots[key] = created
            return created
        }()
        let buf = try slot.ensure(hal: hal, bytes: tShape.totalBytes)
        if zeroFill && tShape.totalBytes > 0 {
            memset(buf.contents(), 0, tShape.totalBytes)
        }
        return QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: buf.length), offset: 0, shape: tShape)
    }

    func tensorI32(key: String, values: [Int32], shape: [Int]) throws -> QWTensorValue {
        let tv = try tensor(key: key, shape: shape, dtype: .i32)
        let byteLen = values.count * MemoryLayout<Int32>.stride
        if byteLen > 0 {
            let dst = tv.region.buffer.contents().advanced(by: tv.region.base + tv.offset)
            withExtendedLifetime(tv.region.buffer) {
                values.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                    if let base = raw.baseAddress {
                        dst.copyMemory(from: base, byteCount: byteLen)
                    }
                }
            }
        }
        return tv
    }

    func ropeTables(
        for descriptor: QWModelDescriptor,
        maxSeqLen: Int
    ) throws -> [String: (table: QWTensorValue, rotaryDims: Int, zeroTailArg: Bool)] {
        let targetRows = max(1, maxSeqLen)
        if targetRows <= ropeCapacitySeqLen && !cachedRopeTables.isEmpty {
            return cachedRopeTables
        }
        let allocRows = max(targetRows, 256)
        let hd = descriptor.headDim
        var out: [String: (table: QWTensorValue, rotaryDims: Int, zeroTailArg: Bool)] = [:]
        var bufs: [any MTLBuffer] = []

        func makeTableBuffer(_ values: [Float], shape: [Int]) throws -> QWTensorValue {
            let tShape = QWTensorShape(shape, .f32)
            let byteLen = values.count * MemoryLayout<Float>.stride
            let aligned = max(64, qwAlignUp(byteLen, 64))
            guard let buf = hal.device.makeBuffer(length: aligned, options: .storageModeShared) else {
                throw QWDecisionError.backendFailed("failed to allocate RoPE Metal buffer")
            }
            let dst = buf.contents()
            withExtendedLifetime(buf) {
                if aligned > byteLen { memset(dst, 0, aligned) }
                if byteLen > 0 {
                    values.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                        if let base = raw.baseAddress {
                            dst.copyMemory(from: base, byteCount: byteLen)
                        }
                    }
                }
            }
            bufs.append(buf)
            return QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: aligned), offset: 0, shape: tShape)
        }

        if let perKind = descriptor.perKindRoPEConfig, !perKind.byLayerKind.isEmpty {
            let zeroTail = perKind.zeroTailPartialRotation
            for (kind, entry) in perKind.byLayerKind {
                let built = try QWRopeTables.makePerKindTable(
                    rows: allocRows, headDim: hd, entry: entry, zeroTailFreqs: zeroTail
                )
                let t = try makeTableBuffer(built.cosSin, shape: [allocRows, built.rotaryDims])
                out[kind] = (t, built.rotaryDims, false)
            }
        } else {
            let entry = PerKindRoPEEntry(ropeTheta: descriptor.ropeTheta, partialRotaryFactor: 1.0, ropeType: "default")
            let built = try QWRopeTables.makePerKindTable(
                rows: allocRows, headDim: hd, entry: entry, zeroTailFreqs: false
            )
            let t = try makeTableBuffer(built.cosSin, shape: [allocRows, built.rotaryDims])
            out["default"] = (t, built.rotaryDims, false)
        }

        retainedRopeBuffers = bufs
        cachedRopeTables = out
        ropeCapacitySeqLen = allocRows
        return out
    }
}
