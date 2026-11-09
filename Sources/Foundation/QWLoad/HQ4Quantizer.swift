// HQ4Quantizer.swift — the one safetensors → HQ4_G32_M5 online GPU quantizer.
//
// Unpacks .safetensors rows (.u32 MLX affine, .bf16, .f16, .f32) into a shared
// BF16 staging buffer and dispatches `qwg_quantize_bf16_to_hq4_g32_m5` into the
// destination Metal buffer. Both `QWEngineDriver` and `FNGPULoader` use this
// door; there is no second copy.

import Foundation
import Darwin
import Metal
import QWHAL
import QWQuant

/// Shared online GPU quantizer from `.safetensors` into `HQ4_G32_M5` blocks.
public enum QWHQ4Quantizer {

    /// Why an online `.safetensors` → `HQ4_G32_M5` quantization failed.
    public enum QuantizeError: Error, CustomStringConvertible, Sendable {
        case missingTensor(name: String, reason: String)
        case invalidLayout(name: String, reason: String)
        case unsupportedDType(name: String, dtype: String)
        case deviceFailure(name: String, reason: String)

        public var description: String {
            switch self {
            case let .missingTensor(name, reason):
                return "\(name): \(reason)"
            case let .invalidLayout(name, reason):
                return "\(name): \(reason)"
            case let .unsupportedDType(name, dtype):
                return "\(name): unsupported safetensors dtype \(dtype)"
            case let .deviceFailure(name, reason):
                return "\(name): \(reason)"
            }
        }
    }

    /// Unpacks `rowCount` rows (starting at `rowOffset`, each of width `cols`)
    /// from `st` into `dst` as `BF16` (`UInt16` bit patterns).
    public static func copySafeTensorsRowsAsBF16(
        st: QWSafeTensorsContainer,
        name: String,
        rowOffset: Int,
        rowCount: Int,
        cols: Int,
        into dst: UnsafeMutablePointer<UInt16>,
        parallel: Bool = true
    ) throws {
        guard st.tensor(name) != nil else {
            throw QuantizeError.missingTensor(name: name, reason: "missing in safetensors")
        }
        let r = try st.reader(of: name)
        let info = r.tensorInfo

        if info.dtype == .u32 {
            let basePrefix = name.hasSuffix(".weight") ? String(name.dropLast(".weight".count)) : name
            let scalesName = basePrefix + ".scales"
            let biasesName = basePrefix + ".biases"
            guard st.tensor(scalesName) != nil else {
                throw QuantizeError.missingTensor(name: scalesName, reason: "missing MLX scales")
            }
            guard st.tensor(biasesName) != nil else {
                throw QuantizeError.missingTensor(name: biasesName, reason: "missing MLX biases")
            }
            let sReader = try st.reader(of: scalesName)
            let bReader = try st.reader(of: biasesName)
            let sInfo = sReader.tensorInfo
            let bInfo = bReader.tensorInfo
            guard (sInfo.dtype == .bf16 || sInfo.dtype == .f16), bInfo.dtype == sInfo.dtype else {
                throw QuantizeError.invalidLayout(name: scalesName, reason: "unsupported MLX scale/bias dtype")
            }
            guard let wCols = info.shape.last, let gCols = sInfo.shape.last, gCols > 0, cols % gCols == 0 else {
                throw QuantizeError.invalidLayout(name: name, reason: "invalid MLX group layout")
            }
            let bits = (wCols * 32) == cols * 4 ? 4 : (wCols * 32) / cols
            guard QWBF16.isSupportedMLXAffineBits(bits) else {
                throw QuantizeError.invalidLayout(name: name, reason: "unsupported MLX affine bit-width \(bits)")
            }
            let groupSize = cols / gCols
            let isF16 = (sInfo.dtype == .f16)
            r.withBytes { wRaw in
                sReader.withBytes { sRaw in
                    bReader.withBytes { bRaw in
                        let wWords = wRaw.bindMemory(to: UInt32.self)
                        let sU16 = sRaw.bindMemory(to: UInt16.self)
                        let bU16 = bRaw.bindMemory(to: UInt16.self)
                        nonisolated(unsafe) let wPtr = wWords.baseAddress!
                        nonisolated(unsafe) let sPtr = sU16.baseAddress!
                        nonisolated(unsafe) let bPtr = bU16.baseAddress!
                        nonisolated(unsafe) let dPtr = dst
                        let workers = (parallel && rowCount >= 64) ? min(16, rowCount) : 1
                        let perWorker = (rowCount + workers - 1) / workers
                        DispatchQueue.concurrentPerform(iterations: workers) { w in
                            let startR = w * perWorker
                            let endR = min(rowCount, startR + perWorker)
                            guard startR < endR else { return }
                            for rIdx in startR..<endR {
                                let srcRow = rowOffset + rIdx
                                let rowWords = wPtr.advanced(by: srcRow * wCols)
                                let sbRowBase = srcRow * gCols
                                let dstRowBase = rIdx * cols
                                for g in 0..<gCols {
                                    let sBits = UInt16(littleEndian: sPtr[sbRowBase + g])
                                    let bBits = UInt16(littleEndian: bPtr[sbRowBase + g])
                                    let scale = isF16
                                        ? Float(Float16(bitPattern: sBits))
                                        : Float(bitPattern: UInt32(sBits) << 16)
                                    let bias = isF16
                                        ? Float(Float16(bitPattern: bBits))
                                        : Float(bitPattern: UInt32(bBits) << 16)
                                    let dstGroupBase = dstRowBase + g * groupSize
                                    let fastFinite = scale.isFinite && bias.isFinite && abs(scale) < 1e30 && abs(bias) < 1e30
                                    if bits == 4 && groupSize % 8 == 0 {
                                        let wordBase = (g * groupSize) / 8
                                        let wordsInGroup = groupSize / 8
                                        if fastFinite {
                                            let scaleVec = SIMD8<Float>(repeating: scale)
                                            let biasVec = SIMD8<Float>(repeating: bias)
                                            for wIdx in 0..<wordsInGroup {
                                                let word = UInt32(littleEndian: rowWords[wordBase + wIdx])
                                                let baseOut = dstGroupBase + wIdx * 8
                                                let qVec = SIMD8<Float>(
                                                    Float(word & 0xF),
                                                    Float((word >> 4) & 0xF),
                                                    Float((word >> 8) & 0xF),
                                                    Float((word >> 12) & 0xF),
                                                    Float((word >> 16) & 0xF),
                                                    Float((word >> 20) & 0xF),
                                                    Float((word >> 24) & 0xF),
                                                    Float(word >> 28)
                                                )
                                                let fVec = biasVec.addingProduct(qVec, scaleVec)
                                                let bVec = unsafeBitCast(fVec, to: SIMD8<UInt32>.self)
                                                let rounded = (bVec &+ 0x7FFF &+ ((bVec &>> 16) & 1)) &>> 16
                                                let u16Vec = SIMD8<UInt16>(truncatingIfNeeded: rounded)
                                                UnsafeMutableRawPointer(dPtr.advanced(by: baseOut))
                                                    .storeBytes(of: u16Vec, as: SIMD8<UInt16>.self)
                                            }
                                        } else {
                                            for wIdx in 0..<wordsInGroup {
                                                var word = UInt32(littleEndian: rowWords[wordBase + wIdx])
                                                let baseOut = dstGroupBase + wIdx * 8
                                                for nib in 0..<8 {
                                                    let q = Float(word & 0xF)
                                                    word >>= 4
                                                    dPtr[baseOut + nib] = QWBF16.fromFloatRNE(q * scale + bias)
                                                }
                                            }
                                        }
                                    } else if bits == 8 && groupSize % 8 == 0 && fastFinite {
                                        let wordBase = (g * groupSize) / 4
                                        let pairsInGroup = groupSize / 8
                                        let scaleVec = SIMD8<Float>(repeating: scale)
                                        let biasVec = SIMD8<Float>(repeating: bias)
                                        for pIdx in 0..<pairsInGroup {
                                            let w0 = UInt32(littleEndian: rowWords[wordBase + pIdx * 2])
                                            let w1 = UInt32(littleEndian: rowWords[wordBase + pIdx * 2 + 1])
                                            let baseOut = dstGroupBase + pIdx * 8
                                            let qVec = SIMD8<Float>(
                                                Float(w0 & 0xFF),
                                                Float((w0 >> 8) & 0xFF),
                                                Float((w0 >> 16) & 0xFF),
                                                Float(w0 >> 24),
                                                Float(w1 & 0xFF),
                                                Float((w1 >> 8) & 0xFF),
                                                Float((w1 >> 16) & 0xFF),
                                                Float(w1 >> 24)
                                            )
                                            let fVec = biasVec.addingProduct(qVec, scaleVec)
                                            let bVec = unsafeBitCast(fVec, to: SIMD8<UInt32>.self)
                                            let rounded = (bVec &+ 0x7FFF &+ ((bVec &>> 16) & 1)) &>> 16
                                            let u16Vec = SIMD8<UInt16>(truncatingIfNeeded: rounded)
                                            UnsafeMutableRawPointer(dPtr.advanced(by: baseOut))
                                                .storeBytes(of: u16Vec, as: SIMD8<UInt16>.self)
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
                                                dPtr[baseOut + byteIdx] = QWBF16.fromFloatRNE(q * scale + bias)
                                            }
                                        }
                                    } else {
                                        for j in 0..<groupSize {
                                            let col = g * groupSize + j
                                            let q = QWBF16.unpackMLXAffineQ(rowWords: rowWords, col: col, bits: bits)
                                            dPtr[dstGroupBase + j] = QWBF16.fromFloatRNE(q * scale + bias)
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            return
        }

        let elemStart = rowOffset * cols
        let elemCount = rowCount * cols
        if let totalElems = info.elementCount, elemStart + elemCount > totalElems {
            throw QuantizeError.invalidLayout(
                name: name,
                reason: "shape \(info.shape) smaller than requested rows \(rowOffset + rowCount) × \(cols)"
            )
        }
        try r.withBytes { raw in
            switch info.dtype {
            case .bf16:
                memcpy(dst, raw.baseAddress!.advanced(by: elemStart * 2), elemCount * 2)
            case .f16:
                let src = raw.bindMemory(to: UInt16.self)
                for i in 0..<elemCount {
                    let f = Float(Float16(bitPattern: UInt16(littleEndian: src[elemStart + i])))
                    dst[i] = QWBF16.fromFloatRNE(f)
                }
            case .f32:
                let src = raw.bindMemory(to: UInt32.self)
                for i in 0..<elemCount {
                    let bits = UInt32(littleEndian: src[elemStart + i])
                    dst[i] = QWBF16.fromFloatRNE(Float(bitPattern: bits))
                }
            default:
                throw QuantizeError.unsupportedDType(name: name, dtype: info.dtype.rawValue)
            }
        }
    }

    /// Dispatches `qwg_quantize_bf16_to_hq4_g32_m5` for `chunkRows` rows of width `k`
    /// from `stageBuf` into `dstBuffer` at `dstByteOffset`.
    private static func dispatchQuantBF16(
        hal: QWHal,
        pso: any MTLComputePipelineState,
        stageBuf: any MTLBuffer,
        chunkRows: Int,
        k: Int,
        dstBuffer: any MTLBuffer,
        dstByteOffset: Int,
        name: String
    ) throws {
        guard let (cb, enc) = hal.begin(.serial) else {
            throw QuantizeError.deviceFailure(name: name, reason: "cmdbuffer unavailable")
        }
        var nU = UInt32(chunkRows)
        var kU = UInt32(k)
        enc.setComputePipelineState(pso)
        enc.setBuffer(stageBuf, offset: 0, index: 0)
        enc.setBuffer(dstBuffer, offset: dstByteOffset, index: 1)
        enc.setBytes(&nU, length: 4, index: 4)
        enc.setBytes(&kU, length: 4, index: 5)
        let blocks = chunkRows * (k / 32)
        let tgs = min(1024, max(1, (blocks + 255) / 256))
        enc.dispatchThreadgroups(
            MTLSize(width: tgs, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        if let err = cb.error {
            throw QuantizeError.deviceFailure(name: name, reason: "online quant cb: \(err.localizedDescription)")
        }
    }

    /// Quantizes `rowCount` rows of `name` (starting at `srcRowOffset`, width `k`)
    /// from `st` into `dstBuffer` at `dstByteOffset` as `HQ4_G32_M5`.
    public static func quantizeSafeTensorsInto(
        hal: QWHal,
        st: QWSafeTensorsContainer,
        name: String,
        srcRowOffset: Int = 0,
        rowCount: Int,
        k: Int,
        dstBuffer: any MTLBuffer,
        dstByteOffset: Int,
        staging: (any MTLBuffer)? = nil,
        quantPSO: (any MTLComputePipelineState)? = nil
    ) throws {
        let pso = try quantPSO ?? hal.pipeline("qwg_quantize_bf16_to_hq4_g32_m5")
        let stageBytes = 128 * 1024 * 1024
        guard let stageBuf = staging ?? hal.device.makeBuffer(length: stageBytes, options: .storageModeShared) else {
            throw QuantizeError.deviceFailure(name: name, reason: "safetensors online quantizer staging unavailable")
        }
        if srcRowOffset == 0, let info = st.tensor(name), info.dtype != .u32 {
            guard info.elementCount == rowCount * k else {
                throw QuantizeError.invalidLayout(name: name, reason: "shape \(info.shape) ≠ [\(rowCount), \(k)]")
            }
        }
        let stagePtr = stageBuf.contents().assumingMemoryBound(to: UInt16.self)
        let maxChunkRows = max(1, stageBuf.length / (k * 2))
        let hq4RowBytes = (k / 32) * 20
        var doneRows = 0
        while doneRows < rowCount {
            let chunkRows = min(maxChunkRows, rowCount - doneRows)
            try copySafeTensorsRowsAsBF16(
                st: st,
                name: name,
                rowOffset: srcRowOffset + doneRows,
                rowCount: chunkRows,
                cols: k,
                into: stagePtr
            )
            try dispatchQuantBF16(
                hal: hal,
                pso: pso,
                stageBuf: stageBuf,
                chunkRows: chunkRows,
                k: k,
                dstBuffer: dstBuffer,
                dstByteOffset: dstByteOffset + doneRows * hq4RowBytes,
                name: name
            )
            doneRows += chunkRows
        }
    }

    /// Interleaves per-expert `gate` (`[experts, inter, hidden]`) and `up`
    /// (`[experts, inter, hidden]`) from `st` into `[experts, 2 * inter, hidden]`
    /// `HQ4_G32_M5` blocks in `dstBuffer` at `dstByteOffset`.
    public static func quantizeSwitchGateUpInto(
        hal: QWHal,
        st: QWSafeTensorsContainer,
        gateName: String,
        upName: String,
        experts: Int,
        inter: Int,
        hidden: Int,
        dstBuffer: any MTLBuffer,
        dstByteOffset: Int,
        staging: (any MTLBuffer)? = nil,
        quantPSO: (any MTLComputePipelineState)? = nil
    ) throws {
        guard st.tensor(gateName) != nil, st.tensor(upName) != nil else {
            throw QuantizeError.missingTensor(name: gateName, reason: "missing switch_mlp gate/up in safetensors")
        }
        let pso = try quantPSO ?? hal.pipeline("qwg_quantize_bf16_to_hq4_g32_m5")
        let stageBytes = 128 * 1024 * 1024
        guard let stageBuf = staging ?? hal.device.makeBuffer(length: stageBytes, options: .storageModeShared) else {
            throw QuantizeError.deviceFailure(name: gateName, reason: "safetensors online quantizer staging unavailable")
        }
        let stagePtr = stageBuf.contents().assumingMemoryBound(to: UInt16.self)
        let expertElems = 2 * inter * hidden
        let batchExperts = max(1, min(experts, stageBuf.length / (expertElems * 2)))
        let guRowBytes = (hidden / 32) * 20
        var eStart = 0
        while eStart < experts {
            let eCount = min(batchExperts, experts - eStart)
            if eCount >= 2 {
                let baseExpert = eStart
                let stBox = st
                nonisolated(unsafe) let stageBase = stagePtr
                nonisolated(unsafe) var capturedError: (any Error)?
                let errLock = NSLock()
                let workers = min(16, eCount)
                let perWorker = (eCount + workers - 1) / workers
                DispatchQueue.concurrentPerform(iterations: workers) { w in
                    let startI = w * perWorker
                    let endI = min(eCount, startI + perWorker)
                    guard startI < endI else { return }
                    do {
                        for i in startI..<endI {
                            let e = baseExpert + i
                            let basePtr = stageBase.advanced(by: i * expertElems)
                            try copySafeTensorsRowsAsBF16(
                                st: stBox,
                                name: gateName,
                                rowOffset: e * inter,
                                rowCount: inter,
                                cols: hidden,
                                into: basePtr,
                                parallel: false
                            )
                            try copySafeTensorsRowsAsBF16(
                                st: stBox,
                                name: upName,
                                rowOffset: e * inter,
                                rowCount: inter,
                                cols: hidden,
                                into: basePtr.advanced(by: inter * hidden),
                                parallel: false
                            )
                        }
                    } catch {
                        errLock.lock()
                        if capturedError == nil { capturedError = error }
                        errLock.unlock()
                    }
                }
                if let err = capturedError { throw err }
            } else {
                for i in 0..<eCount {
                    let e = eStart + i
                    let basePtr = stagePtr.advanced(by: i * expertElems)
                    try copySafeTensorsRowsAsBF16(
                        st: st,
                        name: gateName,
                        rowOffset: e * inter,
                        rowCount: inter,
                        cols: hidden,
                        into: basePtr
                    )
                    try copySafeTensorsRowsAsBF16(
                        st: st,
                        name: upName,
                        rowOffset: e * inter,
                        rowCount: inter,
                        cols: hidden,
                        into: basePtr.advanced(by: inter * hidden)
                    )
                }
            }
            let chunkRows = eCount * 2 * inter
            try dispatchQuantBF16(
                hal: hal,
                pso: pso,
                stageBuf: stageBuf,
                chunkRows: chunkRows,
                k: hidden,
                dstBuffer: dstBuffer,
                dstByteOffset: dstByteOffset + eStart * 2 * inter * guRowBytes,
                name: gateName
            )
            eStart += eCount
        }
    }
}
