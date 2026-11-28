// EncodeKernels+HeadAndEmbed.swift — mHC, target head, and CPU embedding staging on QWEngineDriver.

import Foundation
import Darwin
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWNorm
import QWRope
import QWKV
import QWAttn
import QWGDN
import QWMLP
import QWHead
import QWMTP
import QWQuant

extension QWEngineDriver {
    func encodeMHCEnter(_ enc: any MTLComputeCommandEncoder,
                        x: QWTensorValue, stream: QWTensorValue, rows: Int = 1) throws {
        guard let pso = psoMHCEnter else {
            throw QWEngineDriverError.state("qwg_mhc_enter_f16 pipeline is absent")
        }
        let hidden = geometry.hidden
        enc.setComputePipelineState(pso)
        enc.bindTensors(x, stream, startIndex: 0)
        enc.setUInt32s(hidden, startIndex: 2)
        enc.dispatchThreadgroups(MTLSize(width: (hidden + 255) / 256, height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Rows per threadgroup in qwmhc_rows.metal (8 for C=4, 4 for C=2).
    static func mhcRouterRowsPerGroup(streams: Int) -> Int {
        streams == 4 ? 8 : 4
    }

    func encodeMHCRouterNorm(_ enc: any MTLComputeCommandEncoder,
                             stream: QWTensorValue, w: QWTensorValue, b: QWTensorValue,
                             gamma: QWTensorValue, controls: QWTensorValue,
                             out: QWTensorValue, rows: Int = 1,
                             mergeBranchOut: QWTensorValue? = nil,
                             moeBridge: (xRot32: QWTensorValue, xPlain32: QWTensorValue, skipRot32: Bool)? = nil) throws {
        try checkNormGeometry()
        guard let pso = psoMHCRouterNorm else {
            throw QWEngineDriverError.state("qwg_mhc_router_norm_f16 pipeline is absent")
        }
        let hidden = geometry.hidden
        let preScale = geometry.mhcConfig?.preScale ?? 1.0
        var iterations = UInt32(geometry.mhcConfig?.iterations ?? 20)
        if mergeBranchOut != nil {
            iterations |= 0x8000_0000
        }
        enc.setComputePipelineState(pso)
        if let split = mhcSplit, rows <= QWMHCSplitRouter.maxRows {
            let partialPSO = split.partial
            let finalizePSO = moeBridge != nil ? split.finalizeMoE : split.finalize
            let partials = split.partials
            let parts = (hidden + 63) / 64
            enc.setComputePipelineState(partialPSO)
            enc.bindTensors(stream, w, controls, startIndex: 0)
            (mergeBranchOut ?? out).bind(enc, index: 3)
            enc.setBuffer(partials, offset: 0, index: 4)
            enc.setUInt32s(hidden, startIndex: 5)
            enc.setBytes(&iterations, length: 4, index: 6)
            enc.dispatchThreadgroups(MTLSize(width: parts, height: rows, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(finalizePSO)
            stream.bind(enc, index: 0)
            enc.setBuffer(partials, offset: 0, index: 1)
            enc.bindTensors(b, gamma, out, controls, startIndex: 2)
            var packedHidden = UInt32(hidden)
            if let moeBridge, moeBridge.skipRot32 { packedHidden |= 0x8000_0000 }
            enc.setBytes(&packedHidden, length: 4, index: 6)
            enc.setPod(geometry.rmsEps, index: 7)
            enc.setPod(preScale, index: 8)
            enc.setBytes(&iterations, length: 4, index: 9)
            if let moeBridge {
                moeBridge.xRot32.bind(enc, index: 10)
                moeBridge.xPlain32.bind(enc, index: 11)
            }
            enc.dispatchThreadgroups(MTLSize(width: (hidden + 255) / 256, height: rows, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            return
        }
        let rowsPSO = rows >= 16 ? psoMHCRouterNormRows : nil
        if let rowsPSO { enc.setComputePipelineState(rowsPSO) }
        enc.bindTensors(stream, w, b, gamma, out, controls, startIndex: 0)
        enc.setUInt32s(hidden, startIndex: 6)
        enc.setPod(geometry.rmsEps, index: 7)
        enc.setPod(preScale, index: 8)
        enc.setBytes(&iterations, length: 4, index: 9)
        (mergeBranchOut ?? out).bind(enc, index: 10)
        if rowsPSO != nil {
            let rpg = Self.mhcRouterRowsPerGroup(streams: geometry.mhcConfig?.streams ?? 4)
            enc.setUInt32s(rows, startIndex: 11)
            enc.dispatchThreadgroups(MTLSize(width: (rows + rpg - 1) / rpg, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            return
        }
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    func encodeMHCMerge(_ enc: any MTLComputeCommandEncoder,
                        stream: QWTensorValue, branchOut: QWTensorValue,
                        controls: QWTensorValue, xOut: QWTensorValue,
                        nextGamma: QWTensorValue, nextNormOut: QWTensorValue,
                        rows: Int = 1, isLeave: Bool, sinkhornIters: Int,
                        hasNextNorm: Bool) throws {
        try checkNormGeometry()
        guard let pso = psoMHCMerge else {
            throw QWEngineDriverError.state("qwg_mhc_merge_f16 pipeline is absent")
        }
        let hidden = geometry.hidden
        var flags: UInt32 = UInt32(sinkhornIters & 0xFF) << 8
        if isLeave { flags |= 1 }
        if hasNextNorm { flags |= 2 }
        enc.setComputePipelineState(pso)
        enc.bindTensors(stream, branchOut, controls, xOut, nextGamma, nextNormOut, startIndex: 0)
        enc.setUInt32s(hidden, startIndex: 6)
        enc.setPod(geometry.rmsEps, index: 7)
        enc.setUInt32s(Int(flags), startIndex: 8)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    func encodeMHCExitHead(_ enc: any MTLComputeCommandEncoder,
                           stream: QWTensorValue, headW: QWTensorValue, headB: QWTensorValue,
                           normGamma: QWTensorValue, outNorm: QWTensorValue,
                           outRot32: QWTensorValue? = nil,
                           rows: Int = 1, applyNorm: Bool = true) throws {
        try checkNormGeometry()
        guard let pso = psoMHCExitHead else {
            throw QWEngineDriverError.state("qwg_mhc_exit_head_f16 pipeline is absent")
        }
        let preScale = geometry.mhcConfig?.preScale ?? 1.0
        let flags = (applyNorm ? 1 : 0) | (outRot32 != nil ? 2 : 0)
        enc.setComputePipelineState(pso)
        enc.bindTensors(stream, headW, headB, normGamma, outNorm, startIndex: 0)
        (outRot32 ?? outNorm).bind(enc, index: 5)
        enc.setUInt32s(geometry.hidden, startIndex: 6)
        enc.setPod(geometry.rmsEps, index: 7)
        enc.setPod(preScale, index: 8)
        enc.setUInt32s(flags, startIndex: 9)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    func encodeTargetHead(_ enc: any MTLComputeCommandEncoder, x: QWTensorValue) throws {
        let softcap = geometry.finalLogitSoftcapping ?? 0.0
        switch headStorage {
        case .f16(let weight):
            encodeFwht32Fast(enc, x: x)
            enc.memoryBarrier(scope: .buffers)
            try QWHead.encodeF16(enc, hal: hal, w: weight, x: x, y: sc.logits,
                                 geometry: geometry, M: 1, N: geometry.vocab, softcap: softcap)
            enc.memoryBarrier(scope: .buffers)
            encodeFwht32Fast(enc, x: x)
        case .bf16(let weight):
            encodeFwht32Fast(enc, x: x)
            enc.memoryBarrier(scope: .buffers)
            try QWHead.encodeBF16(enc, hal: hal, w: weight, x: x, y: sc.logits,
                                  geometry: geometry, M: 1, N: geometry.vocab, softcap: softcap)
            enc.memoryBarrier(scope: .buffers)
            encodeFwht32Fast(enc, x: x)
        case .q4g:
            try head.encode(enc, storage: headStorage, x: x, gs: sc.gs, y: sc.logits, M: 1, softcap: softcap)
        }
        if let mm = multimaxParams {
            enc.memoryBarrier(scope: .buffers)
            try head.encodeMultiMax(enc, logits: sc.logits, params: mm, count: geometry.vocab)
        }
    }

    func encodeTargetHeadGreedy(_ enc: any MTLComputeCommandEncoder, x: QWTensorValue,
                                writeLogits: Bool = true) throws {
        let softcap = geometry.finalLogitSoftcapping ?? 0.0
        if multimaxParams == nil, case .q4g(let weight) = headStorage,
           head.splits == 1 || !weight.view.isIncumbentContainer {
            try head.encodeGreedyQ4g(enc, x: x, w: weight, y: sc.logits,
                                     keysHi: sc.keysHi, keysLo: sc.keysLo, token: sc.token,
                                     writeLogits: writeLogits, softcap: softcap)
        } else {
            try encodeTargetHead(enc, x: x)
            enc.memoryBarrier(scope: .buffers)
            try head.encodeArgmax(enc, y: sc.logits, keysHi: sc.keysHi, keysLo: sc.keysLo,
                                  token: sc.token, N: geometry.vocab)
        }
    }

    // MARK: - small CPU-side copies (engine-private; steady allocs stay 0) ──
    /// C5 FIX: token ids arrive from the tokenizer and from public API
    /// callers — an id outside 0…vocab−1 would read the embed slab out of
    func copyEmbedRow(_ token: Int, to dst: QWTensorValue) throws {
        guard token >= 0, token < geometry.vocab else { throw QWEngineDriverError.badToken(token) }
        let H = geometry.hidden
        let dstBase = dst.region.buffer.contents().advanced(by: dst.region.base + dst.offset)
            .assumingMemoryBound(to: UInt16.self)
        if let pack {
            let byteOffset = embedRawOff + token * H * 2
            try pack.withRawPointer(at: byteOffset, byteCount: H * 2) { raw in
                let src = raw.assumingMemoryBound(to: UInt16.self)
                QWQuant.bf16ToF16Array(src: src, dst: dstBase, count: H)
            }
            return
        }
        guard let embedReader else {
            throw QWEngineDriverError.weightMissing(embedPlane, "no embedding reader")
        }
        if embedReader.tensorInfo.dtype == .u32,
           let sReader = embedScalesReader,
           let bReader = embedBiasesReader {
            let packedCols = embedReader.tensorInfo.shape.count == 2
                ? embedReader.tensorInfo.shape[1]
                : (embedReader.tensorInfo.elementCount ?? 0) / geometry.vocab
            let groups = sReader.tensorInfo.shape.count == 2
                ? sReader.tensorInfo.shape[1]
                : (sReader.tensorInfo.elementCount ?? 0) / geometry.vocab
            let groupSize = H / max(1, groups)
            let bits = (packedCols * 32) / H
            let sIsF16 = (sReader.tensorInfo.dtype == .f16)
            embedReader.withBytes { wRaw in
                sReader.withBytes { sRaw in
                    bReader.withBytes { bRaw in
                        let w32 = wRaw.bindMemory(to: UInt32.self)
                        guard let wPtr = w32.baseAddress else { return }
                        let s16 = sRaw.bindMemory(to: UInt16.self)
                        let b16 = bRaw.bindMemory(to: UInt16.self)
                        Self.decodeQuantizedEmbedRow(
                            tok: token, H: H, packedCols: packedCols, groups: groups,
                            groupSize: groupSize, bits: bits, sIsF16: sIsF16,
                            wPtr: wPtr, s16: s16, b16: b16, out: dstBase
                        )
                    }
                }
            }
            return
        }
        let isF16 = (embedReader.tensorInfo.dtype == .f16)
        withExtendedLifetime(dst.region.buffer) {
            embedReader.withBytes { rawBuf in
                guard let base = rawBuf.baseAddress else { return }
                let src = base.advanced(by: token * H * 2).assumingMemoryBound(to: UInt16.self)
                if isF16 {
                    UnsafeMutableRawPointer(dstBase).copyMemory(from: UnsafeRawPointer(src), byteCount: H * 2)
                } else {
                    QWQuant.bf16ToF16Array(src: src, dst: dstBase, count: H)
                }
            }
        }
    }

    private static func decodeQuantizedEmbedRow(
        tok: Int, H: Int, packedCols: Int, groups: Int,
        groupSize: Int, bits: Int, sIsF16: Bool,
        wPtr: UnsafePointer<UInt32>,
        s16: UnsafeBufferPointer<UInt16>,
        b16: UnsafeBufferPointer<UInt16>,
        out: UnsafeMutablePointer<UInt16>
    ) {
        let rowWords = wPtr.advanced(by: tok * packedCols)
        let sBase = tok * groups
        if bits == 4 && groupSize % 8 == 0 {
            let wordsPerGroup = groupSize / 8
            for g in 0..<groups {
                let sc: Float = sIsF16
                    ? Float(Float16(bitPattern: s16[sBase + g]))
                    : Float(bitPattern: UInt32(s16[sBase + g]) << 16)
                let bi: Float = sIsF16
                    ? Float(Float16(bitPattern: b16[sBase + g]))
                    : Float(bitPattern: UInt32(b16[sBase + g]) << 16)
                let wGroup = rowWords.advanced(by: g * wordsPerGroup)
                let outGroup = out.advanced(by: g * groupSize)
                for wIdx in 0..<wordsPerGroup {
                    var word = wGroup[wIdx]
                    let cBase = wIdx * 8
                    for k in 0..<8 {
                        let q = Float(word & 0xF)
                        word >>= 4
                        outGroup[cBase + k] = QWQuant.f32tof16Bits(q * sc + bi)
                    }
                }
            }
        } else if bits == 2 && groupSize % 16 == 0 {
            let wordsPerGroup = groupSize / 16
            for g in 0..<groups {
                let sc: Float = sIsF16
                    ? Float(Float16(bitPattern: s16[sBase + g]))
                    : Float(bitPattern: UInt32(s16[sBase + g]) << 16)
                let bi: Float = sIsF16
                    ? Float(Float16(bitPattern: b16[sBase + g]))
                    : Float(bitPattern: UInt32(b16[sBase + g]) << 16)
                let wGroup = rowWords.advanced(by: g * wordsPerGroup)
                let outGroup = out.advanced(by: g * groupSize)
                for wIdx in 0..<wordsPerGroup {
                    var word = wGroup[wIdx]
                    let cBase = wIdx * 16
                    for k in 0..<16 {
                        let q = Float(word & 0x3)
                        word >>= 2
                        outGroup[cBase + k] = QWQuant.f32tof16Bits(q * sc + bi)
                    }
                }
            }
        } else {
            for g in 0..<groups {
                let sc: Float = sIsF16
                    ? Float(Float16(bitPattern: s16[sBase + g]))
                    : Float(bitPattern: UInt32(s16[sBase + g]) << 16)
                let bi: Float = sIsF16
                    ? Float(Float16(bitPattern: b16[sBase + g]))
                    : Float(bitPattern: UInt32(b16[sBase + g]) << 16)
                let cStart = g * groupSize
                let cEnd = min(H, cStart + groupSize)
                for c in cStart..<cEnd {
                    let q = QWBF16.unpackMLXAffineQ(rowWords: rowWords, col: c, bits: bits)
                    out[c] = QWQuant.f32tof16Bits(q * sc + bi)
                }
            }
        }
    }

    func copyEmbedRows(_ tokens: [Int], start: Int, count: Int, to dst: QWTensorValue) throws {
        guard count > 0 else { return }
        if count == 1 {
            try copyEmbedRow(tokens[start], to: dst)
            return
        }
        for t in 0..<count {
            let tok = tokens[start + t]
            guard tok >= 0, tok < geometry.vocab else { throw QWEngineDriverError.badToken(tok) }
        }
        let H = geometry.hidden
        let dstBase = dst.region.buffer.contents().advanced(by: dst.region.base + dst.offset)
            .assumingMemoryBound(to: UInt16.self)
        if let pack {
            try pack.withRawPointer(at: embedRawOff, byteCount: geometry.vocab * H * 2) { rawBase in
                let rawAddr = Int(bitPattern: rawBase)
                let dstAddr = Int(bitPattern: dstBase)
                if count >= 64 {
                    DispatchQueue.concurrentPerform(iterations: count) { t in
                        let tok = tokens[start + t]
                        let src = UnsafeRawPointer(bitPattern: rawAddr)!.advanced(by: tok * H * 2)
                            .assumingMemoryBound(to: UInt16.self)
                        let out = UnsafeMutablePointer<UInt16>(bitPattern: dstAddr)!.advanced(by: t * H)
                        QWQuant.bf16ToF16Array(src: src, dst: out, count: H)
                    }
                } else {
                    for t in 0..<count {
                        let tok = tokens[start + t]
                        let src = rawBase.advanced(by: tok * H * 2).assumingMemoryBound(to: UInt16.self)
                        QWQuant.bf16ToF16Array(src: src, dst: dstBase.advanced(by: t * H), count: H)
                    }
                }
            }
            return
        }
        guard let embedReader else {
            throw QWEngineDriverError.weightMissing(embedPlane, "no embedding reader")
        }
        if embedReader.tensorInfo.dtype == .u32,
           let sReader = embedScalesReader,
           let bReader = embedBiasesReader {
            let packedCols = embedReader.tensorInfo.shape.count == 2
                ? embedReader.tensorInfo.shape[1]
                : (embedReader.tensorInfo.elementCount ?? 0) / geometry.vocab
            let groups = sReader.tensorInfo.shape.count == 2
                ? sReader.tensorInfo.shape[1]
                : (sReader.tensorInfo.elementCount ?? 0) / geometry.vocab
            let groupSize = H / max(1, groups)
            let bits = (packedCols * 32) / H
            let sIsF16 = (sReader.tensorInfo.dtype == .f16)
            embedReader.withBytes { wRaw in
                sReader.withBytes { sRaw in
                    bReader.withBytes { bRaw in
                        let w32 = wRaw.bindMemory(to: UInt32.self)
                        guard let wPtr = w32.baseAddress else { return }
                        let s16 = sRaw.bindMemory(to: UInt16.self)
                        let b16 = bRaw.bindMemory(to: UInt16.self)
                        let wAddr = Int(bitPattern: wPtr)
                        let sAddr = Int(bitPattern: s16.baseAddress!)
                        let bAddr = Int(bitPattern: b16.baseAddress!)
                        let dstAddr = Int(bitPattern: dstBase)
                        let sCount = s16.count
                        let bCount = b16.count
                        if count >= 16 {
                            DispatchQueue.concurrentPerform(iterations: count) { t in
                                let wP = UnsafePointer<UInt32>(bitPattern: wAddr)!
                                let sBuf = UnsafeBufferPointer(start: UnsafePointer<UInt16>(bitPattern: sAddr), count: sCount)
                                let bBuf = UnsafeBufferPointer(start: UnsafePointer<UInt16>(bitPattern: bAddr), count: bCount)
                                let outP = UnsafeMutablePointer<UInt16>(bitPattern: dstAddr)!.advanced(by: t * H)
                                Self.decodeQuantizedEmbedRow(
                                    tok: tokens[start + t], H: H, packedCols: packedCols, groups: groups,
                                    groupSize: groupSize, bits: bits, sIsF16: sIsF16,
                                    wPtr: wP, s16: sBuf, b16: bBuf, out: outP
                                )
                            }
                        } else {
                            for t in 0..<count {
                                Self.decodeQuantizedEmbedRow(
                                    tok: tokens[start + t], H: H, packedCols: packedCols, groups: groups,
                                    groupSize: groupSize, bits: bits, sIsF16: sIsF16,
                                    wPtr: wPtr, s16: s16, b16: b16, out: dstBase.advanced(by: t * H)
                                )
                            }
                        }
                    }
                }
            }
            return
        }
        let isF16 = (embedReader.tensorInfo.dtype == .f16)
        withExtendedLifetime(dst.region.buffer) {
            embedReader.withBytes { rawBuf in
                guard let base = rawBuf.baseAddress else { return }
                let baseAddr = Int(bitPattern: base)
                let dstAddr = Int(bitPattern: dstBase)
                if count >= 64 && !isF16 {
                    DispatchQueue.concurrentPerform(iterations: count) { t in
                        let tok = tokens[start + t]
                        let src = UnsafeRawPointer(bitPattern: baseAddr)!.advanced(by: tok * H * 2).assumingMemoryBound(to: UInt16.self)
                        let out = UnsafeMutablePointer<UInt16>(bitPattern: dstAddr)!.advanced(by: t * H)
                        QWQuant.bf16ToF16Array(src: src, dst: out, count: H)
                    }
                } else {
                    for t in 0..<count {
                        let tok = tokens[start + t]
                        let src = base.advanced(by: tok * H * 2).assumingMemoryBound(to: UInt16.self)
                        let out = dstBase.advanced(by: t * H)
                        if isF16 {
                            UnsafeMutableRawPointer(out).copyMemory(from: UnsafeRawPointer(src), byteCount: H * 2)
                        } else {
                            QWQuant.bf16ToF16Array(src: src, dst: out, count: H)
                        }
                    }
                }
            }
        }
    }

    func applyVisualSpans(start: Int, count: Int, to dst: QWTensorValue) {
        guard !activeVisualSpans.isEmpty, count > 0 else { return }
        let H = geometry.hidden
        let dstBase = dst.region.buffer.contents().advanced(by: dst.region.base + dst.offset)
            .assumingMemoryBound(to: UInt16.self)
        for span in activeVisualSpans {
            let lo = max(start, span.tokenOffset)
            let hi = min(start + count, span.tokenOffset + span.tokenCount)
            guard lo < hi else { continue }
            span.embeddings.withUnsafeBufferPointer { embBuf in
                guard let embBase = embBuf.baseAddress else { return }
                for absPos in lo..<hi {
                    let row = absPos - start
                    let visRow = absPos - span.tokenOffset
                    guard (visRow + 1) * H <= embBuf.count else { continue }
                    let src = embBase.advanced(by: visRow * H)
                    let out = dstBase.advanced(by: row * H)
                    for c in 0..<H {
                        out[c] = Float16(src[c]).bitPattern
                    }
                }
            }
        }
    }
}
