// QWKVCompressedLayout.swift — Layout geometry and quantization arithmetic
// for compressed KV cache (Q8 and Q4 modes).
//
// Layout Law:
// - .f16: 256 tokens × 4 kv-heads × 256 head_dim × 2 bytes = 512 KiB per page.
// - .q8:  256 tokens × 4 kv-heads × 256 head_dim × 1 byte = 256 KiB data +
//         256 tokens × 4 kv-heads × 2 bytes (fp16 scale) = 2048 B scales.

import Foundation
import QWCore

public struct QWKVCompressedLayout: Sendable {
    public let mode: QWKVMode
    public let dataBytesPerPage: Int
    public let scaleBytesPerPage: Int
    public let totalBytesPerPage: Int

    public init(mode: QWKVMode) {
        self.init(mode: mode, geometry: .qwen38_27b)
    }

    /// The page's layout, from the model's own geometry: a page is 256
    /// tokens of `kvHeads × headDim` elements. The **compressed** modes are
    /// the 27B's layout only — their readers hard-code 4 heads × 256 (C103).
    public init(mode: QWKVMode, geometry: QWModelGeometry) {
        self.mode = mode
        let pageTokens = QWKVCache.pageTokens
        let kvHeads = geometry.usesNoPEMLA ? 1 : geometry.kvHeads
        let headDim = geometry.usesNoPEMLA ? geometry.kvChannels : geometry.headDim
        let totalElements = pageTokens * geometry.kvChannels

        switch mode {
        case .f16:
            self.dataBytesPerPage = totalElements * 2
            self.scaleBytesPerPage = 0
            self.totalBytesPerPage = self.dataBytesPerPage
        case .q8:
            // 1 byte per element + 1 Float16 scale per (token, head)
            self.dataBytesPerPage = totalElements * 1
            self.scaleBytesPerPage = pageTokens * kvHeads * MemoryLayout<Float16>.stride
            self.totalBytesPerPage = self.dataBytesPerPage + self.scaleBytesPerPage
        case .q4:
            // 0.5 bytes per element (nibble) + 1 Float16 scale and 1 Float16 min per group of <=128 within each head
            self.dataBytesPerPage = totalElements / 2
            let groupsPerHead = max(1, (headDim + 127) / 128)
            let groupsPerPage = pageTokens * kvHeads * groupsPerHead
            self.scaleBytesPerPage = groupsPerPage * (MemoryLayout<Float16>.stride * 2)
            self.totalBytesPerPage = self.dataBytesPerPage + self.scaleBytesPerPage
        }
    }

    /// Symmetric per-(token, head) Q8 quantization of `kvHeads × headDim` elements.
    public static func quantizeQ8(
        input: [Float],
        qOut: inout [Int8],
        scalesOut: inout [Float16],
        kvHeads: Int = 4,
        headDim: Int = 256
    ) throws {
        let total = kvHeads * headDim
        guard kvHeads > 0, headDim > 0,
              input.count == total, qOut.count == total, scalesOut.count == kvHeads else {
            throw QWKVError.badArgument(
                "q8 quantization expects \(total) elements and \(kvHeads) scales (got \(input.count), \(qOut.count), \(scalesOut.count))")
        }
        for h in 0..<kvHeads {
            let offset = h * headDim
            var maxAbs: Float = 0.0
            for d in 0..<headDim {
                let v = abs(input[offset + d])
                if v > maxAbs { maxAbs = v }
            }
            let scale = max(maxAbs / 127.0, 1e-8)
            let s16 = Float16(scale)
            scalesOut[h] = s16
            let invScale = 1.0 / Float(s16)
            for d in 0..<headDim {
                let scaled = round(input[offset + d] * invScale)
                let clamped = max(-127.0, min(127.0, scaled))
                qOut[offset + d] = Int8(clamped)
            }
        }
    }

    /// Symmetric per-(token, head) Q8 dequantization of `kvHeads × headDim` elements.
    public static func dequantizeQ8(
        qIn: [Int8],
        scalesIn: [Float16],
        out: inout [Float],
        kvHeads: Int = 4,
        headDim: Int = 256
    ) throws {
        let total = kvHeads * headDim
        guard kvHeads > 0, headDim > 0,
              qIn.count == total, scalesIn.count == kvHeads, out.count == total else {
            throw QWKVError.badArgument(
                "q8 dequantization expects \(total) elements and \(kvHeads) scales (got \(qIn.count), \(scalesIn.count), \(out.count))")
        }
        for h in 0..<kvHeads {
            let offset = h * headDim
            let sc = Float(scalesIn[h])
            for d in 0..<headDim {
                out[offset + d] = Float(qIn[offset + d]) * sc
            }
        }
    }

    /// Asymmetric group-128 Q4 quantization of `kvHeads × headDim` elements.
    public static func quantizeQ4(
        input: [Float],
        qOut: inout [UInt8],
        scalesOut: inout [Float16],
        minsOut: inout [Float16],
        kvHeads: Int = 4,
        headDim: Int = 256
    ) throws {
        let total = kvHeads * headDim
        let groupsPerHead = max(1, (headDim + 127) / 128)
        let totalGroups = kvHeads * groupsPerHead
        guard kvHeads > 0, headDim > 0, headDim % 2 == 0,
              input.count == total, qOut.count == total / 2,
              scalesOut.count == totalGroups, minsOut.count == totalGroups else {
            throw QWKVError.badArgument(
                "q4 quantization expects \(total) elements, \(total / 2) bytes, \(totalGroups) scales and mins (got \(input.count), \(qOut.count), \(scalesOut.count), \(minsOut.count))")
        }
        for h in 0..<kvHeads {
            for g in 0..<groupsPerHead {
                let gIdx = h * groupsPerHead + g
                let groupStart = g * 128
                let groupLen = min(128, headDim - groupStart)
                let inOffset = h * headDim + groupStart
                let outByteOffset = inOffset / 2
                var minVal = input[inOffset]
                var maxVal = input[inOffset]
                for i in 1..<groupLen {
                    let v = input[inOffset + i]
                    if v < minVal { minVal = v }
                    if v > maxVal { maxVal = v }
                }
                let scale = max((maxVal - minVal) / 15.0, 1e-8)
                let s16 = Float16(scale)
                let m16 = Float16(minVal)
                scalesOut[gIdx] = s16
                minsOut[gIdx] = m16

                let invScale = 1.0 / Float(s16)
                let baseMin = Float(m16)

                for i in 0..<(groupLen / 2) {
                    let val0 = input[inOffset + 2 * i]
                    let val1 = input[inOffset + 2 * i + 1]
                    let q0 = UInt8(max(0.0, min(15.0, round((val0 - baseMin) * invScale))))
                    let q1 = UInt8(max(0.0, min(15.0, round((val1 - baseMin) * invScale))))
                    qOut[outByteOffset + i] = q0 | (q1 << 4)
                }
            }
        }
    }

    /// Asymmetric group-128 Q4 dequantization of `kvHeads × headDim` elements.
    public static func dequantizeQ4(
        qIn: [UInt8],
        scalesIn: [Float16],
        minsIn: [Float16],
        out: inout [Float],
        kvHeads: Int = 4,
        headDim: Int = 256
    ) throws {
        let total = kvHeads * headDim
        let groupsPerHead = max(1, (headDim + 127) / 128)
        let totalGroups = kvHeads * groupsPerHead
        guard kvHeads > 0, headDim > 0, headDim % 2 == 0,
              qIn.count == total / 2, scalesIn.count == totalGroups,
              minsIn.count == totalGroups, out.count == total else {
            throw QWKVError.badArgument(
                "q4 dequantization expects \(total / 2) bytes, \(totalGroups) scales and mins, \(total) elements (got \(qIn.count), \(scalesIn.count), \(minsIn.count), \(out.count))")
        }
        for h in 0..<kvHeads {
            for g in 0..<groupsPerHead {
                let gIdx = h * groupsPerHead + g
                let groupStart = g * 128
                let groupLen = min(128, headDim - groupStart)
                let outOffset = h * headDim + groupStart
                let inByteOffset = outOffset / 2
                let sc = Float(scalesIn[gIdx])
                let mn = Float(minsIn[gIdx])
                for i in 0..<(groupLen / 2) {
                    let byte = qIn[inByteOffset + i]
                    let q0 = Float(byte & 0x0F)
                    let q1 = Float(byte >> 4)
                    out[outOffset + 2 * i] = q0 * sc + mn
                    out[outOffset + 2 * i + 1] = q1 * sc + mn
                }
            }
        }
    }
}
