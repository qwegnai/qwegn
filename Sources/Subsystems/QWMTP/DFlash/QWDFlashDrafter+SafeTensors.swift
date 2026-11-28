// QWDFlashDrafter+SafeTensors.swift — Zero-copy BF16 SafeTensors loader for
// drop-in DFlash 2 drafters (loads linear banks natively in BF16, norms/convs in FP16).
import Foundation
import Metal
import QWAlloc
import QWCore
import QWGEMM
import QWHAL
import QWQuant

extension QWDFlashDrafter {
    static func loadSafeTensorsF16(
        st: QWSafeTensorsContainer,
        wSub: QWSubAllocator,
        hal: QWHal,
        psoBf16F16: any MTLComputePipelineState,
        cvtTmp: any MTLBuffer,
        name: String,
        elems: Int
    ) throws -> QWTensorValue {
        guard let meta = st.tensor(name) else {
            throw QWDFlashError.weightMissing(name, "missing in SafeTensors")
        }
        guard meta.dtype == .bf16, meta.elementCount == elems else {
            throw QWDFlashError.weightMissing(name, "SafeTensors dtype=\(meta.dtype) elems=\(meta.elementCount ?? -1) != \(elems)")
        }
        let t = try wSub.allocate(QWTensorShape([elems], .f16))
        if elems >= 4096 {
            let chunk = 8 * 1024 * 1024
            let cvtBase = cvtTmp.contents()
            try withExtendedLifetime(cvtTmp) {
                try st.withBytes(of: name) { raw in
                    guard raw.count == elems * 2, let srcBase = raw.baseAddress else {
                        throw QWDFlashError.weightMissing(name, "SafeTensors byte count mismatch")
                    }
                    var done = 0
                    while done < elems {
                        let n = min(chunk, elems - done)
                        cvtBase.copyMemory(from: srcBase.advanced(by: done * 2), byteCount: n * 2)
                        guard let (cb, enc) = hal.begin() else {
                            throw QWDFlashError.state("DFlash SafeTensors bf16->f16 cmdbuffer")
                        }
                        var start = UInt32(0), nn = UInt32(n)
                        enc.setComputePipelineState(psoBf16F16)
                        enc.setBuffer(cvtTmp, offset: 0, index: 0)
                        enc.setBuffer(t.region.buffer, offset: t.region.base + t.offset + done * 2, index: 1)
                        enc.setBytes(&start, length: 4, index: 2)
                        enc.setBytes(&nn, length: 4, index: 3)
                        enc.dispatchThreadgroups(MTLSize(width: (n + 255) / 256, height: 1, depth: 1),
                                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                        hal.end(cb, enc)
                        cb.waitUntilCompleted()
                        done += n
                    }
                }
            }
            return t
        }
        let dstBase = t.region.buffer.contents()
        let dst = dstBase.advanced(by: t.region.base + t.offset).assumingMemoryBound(to: UInt16.self)
        try withExtendedLifetime(t.region.buffer) {
            try st.withBytes(of: name) { raw in
                guard raw.count == elems * 2, let srcBase = raw.baseAddress else {
                    throw QWDFlashError.weightMissing(name, "SafeTensors byte count mismatch")
                }
                let src = srcBase.assumingMemoryBound(to: UInt16.self)
                for i in 0..<elems {
                    let f = Float(bitPattern: UInt32(src[i]) << 16)
                    dst[i] = QWQuant.f32tof16Bits(f)
                }
            }
        }
        return t
    }

    static func loadSafeTensorsRawBF16(
        st: QWSafeTensorsContainer,
        wSub: QWSubAllocator,
        name: String,
        elems: Int
    ) throws -> QWTensorValue {
        guard let meta = st.tensor(name) else {
            throw QWDFlashError.weightMissing(name, "missing in SafeTensors")
        }
        guard meta.dtype == .bf16, meta.elementCount == elems else {
            throw QWDFlashError.weightMissing(name, "SafeTensors dtype=\(meta.dtype) elems=\(meta.elementCount ?? -1) != \(elems)")
        }
        let t = try wSub.allocate(QWTensorShape([elems], .bf16))
        let dstBase = t.region.buffer.contents()
        let dst = dstBase.advanced(by: t.region.base + t.offset)
        try withExtendedLifetime(t.region.buffer) {
            try st.withBytes(of: name) { raw in
                guard raw.count == elems * 2, let srcBase = raw.baseAddress else {
                    throw QWDFlashError.weightMissing(name, "SafeTensors byte count mismatch")
                }
                dst.copyMemory(from: srcBase, byteCount: raw.count)
            }
        }
        return t
    }

    static func loadSafeTensorsBF16Bank(
        st: QWSafeTensorsContainer,
        wSub: QWSubAllocator,
        name: String,
        n: Int,
        k: Int
    ) throws -> DFlashLinearWeight {
        guard let meta = st.tensor(name) else {
            throw QWDFlashError.weightMissing(name, "missing in SafeTensors")
        }
        guard meta.dtype == .bf16, meta.shape == [n, k] else {
            throw QWDFlashError.weightMissing(name, "SafeTensors shape \(meta.shape) != [\(n), \(k)]")
        }
        let t = try loadSafeTensorsRawBF16(st: st, wSub: wSub, name: name, elems: n * k)
        return .bf16(QWGemmFP.BF16Bank(weights: t, n: n, k: k))
    }

    static func concatSafeTensorsBF16Banks(
        st: QWSafeTensorsContainer,
        wSub: QWSubAllocator,
        specs: [(name: String, n: Int)],
        k: Int
    ) throws -> DFlashLinearWeight {
        for spec in specs {
            guard let meta = st.tensor(spec.name) else {
                throw QWDFlashError.weightMissing(spec.name, "missing in SafeTensors")
            }
            guard meta.dtype == .bf16, meta.shape == [spec.n, k] else {
                throw QWDFlashError.weightMissing(spec.name, "SafeTensors concat shape \(meta.shape) != [\(spec.n), \(k)]")
            }
        }
        let totalN = specs.reduce(0) { $0 + $1.n }
        let totalElems = totalN * k
        let blocks = try wSub.allocate(QWTensorShape([totalElems], .bf16))
        let dstBase = blocks.region.buffer.contents()
        var byteOff = 0
        try withExtendedLifetime(blocks.region.buffer) {
            for spec in specs {
                let expectedBytes = spec.n * k * 2
                let dst = dstBase.advanced(by: blocks.region.base + blocks.offset + byteOff)
                try st.withBytes(of: spec.name) { raw in
                    guard raw.count == expectedBytes, let srcBase = raw.baseAddress else {
                        throw QWDFlashError.weightMissing(spec.name, "SafeTensors concat byte mismatch")
                    }
                    dst.copyMemory(from: srcBase, byteCount: expectedBytes)
                }
                byteOff += expectedBytes
            }
        }
        return .bf16(QWGemmFP.BF16Bank(weights: blocks, n: totalN, k: k))
    }

    static func loadDeinterleavedSafeTensorsQBProj(
        st: QWSafeTensorsContainer,
        wSub: QWSubAllocator,
        prefix: String,
        qHeads: Int,
        nopeDim: Int,
        ropeDim: Int,
        qLoraRank: Int
    ) throws -> DFlashLinearWeight {
        let nopeName = prefix + "self_attn.q_b_nope_proj.weight"
        let ropeName = prefix + "self_attn.q_b_rope_proj.weight"
        if st.tensor(nopeName) != nil && st.tensor(ropeName) != nil {
            return try concatSafeTensorsBF16Banks(
                st: st, wSub: wSub,
                specs: [(nopeName, qHeads * nopeDim), (ropeName, qHeads * ropeDim)],
                k: qLoraRank
            )
        }
        let fusedName = prefix + "self_attn.q_b_proj.weight"
        let totalHeadDim = nopeDim + ropeDim
        let totalRows = qHeads * totalHeadDim
        guard let meta = st.tensor(fusedName) else {
            throw QWDFlashError.weightMissing(fusedName, "missing in SafeTensors")
        }
        guard meta.dtype == .bf16, meta.shape == [totalRows, qLoraRank] else {
            throw QWDFlashError.weightMissing(
                fusedName, "SafeTensors shape \(meta.shape) != [\(totalRows), \(qLoraRank)]"
            )
        }
        let blocks = try wSub.allocate(QWTensorShape([totalRows * qLoraRank], .bf16))
        let dstBase = blocks.region.buffer.contents()
            .advanced(by: blocks.region.base + blocks.offset)
        let nopeChunkBytes = nopeDim * qLoraRank * 2
        let ropeChunkBytes = ropeDim * qLoraRank * 2
        let srcHeadBytes = totalHeadDim * qLoraRank * 2
        let dstRopeBaseOff = qHeads * nopeChunkBytes
        try withExtendedLifetime(blocks.region.buffer) {
            try st.withBytes(of: fusedName) { raw in
                let expectedBytes = totalRows * qLoraRank * 2
                guard raw.count == expectedBytes, let srcBase = raw.baseAddress else {
                    throw QWDFlashError.weightMissing(fusedName, "SafeTensors byte count mismatch")
                }
                for h in 0..<qHeads {
                    let srcHead = srcBase.advanced(by: h * srcHeadBytes)
                    dstBase.advanced(by: h * nopeChunkBytes)
                        .copyMemory(from: srcHead, byteCount: nopeChunkBytes)
                    if ropeChunkBytes > 0 {
                        dstBase.advanced(by: dstRopeBaseOff + h * ropeChunkBytes)
                            .copyMemory(from: srcHead.advanced(by: nopeChunkBytes), byteCount: ropeChunkBytes)
                    }
                }
            }
        }
        return .bf16(QWGemmFP.BF16Bank(weights: blocks, n: totalRows, k: qLoraRank))
    }
}
