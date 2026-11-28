// QWAttnSinkRel+Sparse.swift — Sparse LSA / e-means-4096 prefill (NAX) and Split-KV GQA decode
// encoders for QWAttnSinkRel (`f16`, `q8`, `q4`).
import Foundation
import Metal
import QWAlloc
import QWCore
import QWHAL
import QWKV

extension QWAttnSinkRel {
    private struct SparseMeta {
        var listStride: UInt32
        var rowIndexed: UInt32
        var pad0: UInt32 = 0
        var pad1: UInt32 = 0
    }

    private func prefillSparsePipeline(_ mode: QWKVMode) throws -> any MTLComputePipelineState {
        let pso: (any MTLComputePipelineState)? = switch mode {
        case .f16: psoPrefillSparseF16
        case .q8:  psoPrefillSparseQ8
        case .q4:  psoPrefillSparseQ4
        }
        guard let pso else {
            throw QWAttnSinkRelError.unsupportedHardware(
                "NAX sparse sink/rel prefill requires Apple10 Metal 4 hardware"
            )
        }
        return pso
    }

    private func decodeSparsePipeline(_ mode: QWKVMode) -> any MTLComputePipelineState {
        switch mode {
        case .f16: return psoDecodeSparseF16
        case .q8:  return psoDecodeSparseQ8
        case .q4:  return psoDecodeSparseQ4
        }
    }

    private func validateIntTensor(_ t: QWTensorValue, minElements: Int, name: String) throws {
        guard t.shape.dtype == .i32 else {
            throw QWAttnSinkRelError.badShape("\(name) must be .i32, got \(t.shape.dtype)")
        }
        let reg = t.region
        guard reg.base >= 0, reg.length >= 0,
              reg.base <= reg.buffer.length, reg.length <= reg.buffer.length - reg.base,
              t.offset >= 0, t.offset <= reg.length, t.bytes <= reg.length - t.offset,
              (reg.base + t.offset) % 4 == 0,
              t.shape.elementCount >= minElements
        else {
            throw QWAttnSinkRelError.badShape(
                "\(name) invalid bounds/alignment: elems=\(t.shape.elementCount) need=\(minElements)"
            )
        }
    }

    /// Encode sparse NAX FlashAttention prefill over LSA / e-means-4096 selected 4-token blocks plus causal tail.
    public func encodePrefillSparse(
        _ enc: any MTLComputeCommandEncoder,
        config: QWSinkRelConfig,
        q: QWTensorValue,
        r: QWTensorValue? = nil,
        relProj: QWTensorValue? = nil,
        sink: QWTensorValue? = nil,
        gate: QWTensorValue? = nil,
        kv: QWSinkRelKVCache,
        out: QWTensorValue,
        blockList: QWTensorValue,
        blockCount: QWTensorValue,
        listStride: Int,
        T: Int,
        posOffset: Int = 0,
        kvLen: Int? = nil,
        perTileBlockList: Bool? = nil
    ) throws {
        let resolvedKVLen = kvLen ?? (posOffset + T)
        try validateCommon(
            config: config, q: q, r: r, relProj: relProj, sink: sink, gate: gate,
            kv: kv, out: out, T: T, posOffset: posOffset, kvLen: resolvedKVLen, isPrefill: true
        )
        guard listStride >= 0 else {
            throw QWAttnSinkRelError.badShape("listStride \(listStride) must be >= 0")
        }
        let numTiles = (T + 31) / 32
        let isCompactTile = perTileBlockList
            ?? (blockCount.shape.elementCount < T || blockList.shape.elementCount < T * listStride)
        let reqRows = isCompactTile ? numTiles : T
        try validateIntTensor(blockCount, minElements: reqRows, name: "blockCount")
        try validateIntTensor(blockList, minElements: reqRows * listStride, name: "blockList")

        let pso = try prefillSparsePipeline(config.kvMode)
        var params = makeMetalParams(
            config: config, T: T, posOffset: posOffset, kvLen: resolvedKVLen,
            ringCap: kv.ringCap, splits: 1
        )
        var sp = SparseMeta(listStride: UInt32(listStride), rowIndexed: isCompactTile ? 0 : 1)

        enc.setComputePipelineState(pso)
        q.bind(enc, index: 0)
        bindOptionalTensor(r, active: config.hasRelBias, enc: enc, index: 1)
        bindOptionalTensor(relProj, active: config.hasRelBias, enc: enc, index: 2)
        bindOptionalTensor(sink, active: config.hasSink, enc: enc, index: 3)
        bindOptionalTensor(gate, active: config.hasOutGate, enc: enc, index: 4)
        enc.setBuffer(kv.kCache.buffer, offset: kv.kCache.base, index: 5)
        enc.setBuffer(kv.vCache.buffer, offset: kv.vCache.base, index: 6)
        bindMetaRegion(kv.kMeta, enc: enc, index: 7)
        bindMetaRegion(kv.vMeta, enc: enc, index: 8)
        out.bind(enc, index: 9)
        enc.setBytes(&params, length: MemoryLayout<MetalParams>.stride, index: 10)
        blockList.bind(enc, index: 11)
        blockCount.bind(enc, index: 12)
        enc.setBytes(&sp, length: MemoryLayout<SparseMeta>.stride, index: 13)

        let gqa = config.nH / config.nKV
        enc.dispatchThreadgroups(
            MTLSize(width: numTiles, height: config.nKV, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64 * gqa, height: 1, depth: 1)
        )
    }

    /// Encode sparse Split-KV GQA decode & verify (`T = 1...16`, optional tree mask) over selected blocks + tail.
    public func encodeDecodeSparse(
        _ enc: any MTLComputeCommandEncoder,
        config: QWSinkRelConfig,
        q: QWTensorValue,
        r: QWTensorValue? = nil,
        relProj: QWTensorValue? = nil,
        sink: QWTensorValue? = nil,
        gate: QWTensorValue? = nil,
        kv: QWSinkRelKVCache,
        partials: QWSinkRelPartials? = nil,
        out: QWTensorValue,
        blockList: QWTensorValue,
        blockCount: QWTensorValue,
        listStride: Int,
        T: Int,
        posOffset: Int,
        kvLen: Int? = nil,
        splits: Int? = nil,
        treeMeta: TreeMeta? = nil
    ) throws {
        let resolvedKVLen = kvLen ?? (posOffset + T)
        try validateCommon(
            config: config, q: q, r: r, relProj: relProj, sink: sink, gate: gate,
            kv: kv, out: out, T: T, posOffset: posOffset, kvLen: resolvedKVLen, isPrefill: false
        )
        guard listStride >= 0 else {
            throw QWAttnSinkRelError.badShape("listStride \(listStride) must be >= 0")
        }
        try validateIntTensor(blockCount, minElements: 1, name: "blockCount")
        try validateIntTensor(blockList, minElements: listStride, name: "blockList")

        let gqa = config.nH / config.nKV
        let approxKeys = min(resolvedKVLen, listStride * 4 + (posOffset & 3) + T)
        let resolvedSplits = splits ?? Self.splits(
            forKeys: approxKeys, kvHeads: config.nKV, batchTokens: T,
            slidingWindow: config.slidingWindow, gpuCores: hal.caps.gpuCores, gqa: gqa
        )
        guard (1...Self.maxSplits).contains(resolvedSplits) else {
            throw QWAttnSinkRelError.badShape("splits \(resolvedSplits) out of range 1...\(Self.maxSplits)")
        }

        let partO: QWMRegion
        let partML: QWMRegion
        if resolvedSplits > 1 {
            guard let partials else {
                throw QWAttnSinkRelError.badShape("splits=\(resolvedSplits) > 1 requires QWSinkRelPartials")
            }
            let need = Self.partialsBytes(
                tokens: T, qHeads: config.nH, headDim: config.headDim, splits: resolvedSplits
            )
            try validateRegion(partials.partO, minBytes: need.partOBytes, alignBytes: 4, name: "partO")
            try validateRegion(partials.partML, minBytes: need.partMLBytes, alignBytes: 8, name: "partML")
            partO = partials.partO
            partML = partials.partML
        } else {
            partO = partials?.partO ?? dummyRegion
            partML = partials?.partML ?? dummyRegion
        }

        var params = makeMetalParams(
            config: config, T: T, posOffset: posOffset, kvLen: resolvedKVLen,
            ringCap: kv.ringCap, splits: resolvedSplits
        )
        var tm = Self.packTreeMeta(treeMeta)
        enc.setComputePipelineState(decodeSparsePipeline(config.kvMode))
        q.bind(enc, index: 0)
        bindOptionalTensor(r, active: config.hasRelBias, enc: enc, index: 1)
        bindOptionalTensor(relProj, active: config.hasRelBias, enc: enc, index: 2)
        bindOptionalTensor(sink, active: config.hasSink, enc: enc, index: 3)
        bindOptionalTensor(gate, active: config.hasOutGate, enc: enc, index: 4)
        enc.setBuffer(kv.kCache.buffer, offset: kv.kCache.base, index: 5)
        enc.setBuffer(kv.vCache.buffer, offset: kv.vCache.base, index: 6)
        bindMetaRegion(kv.kMeta, enc: enc, index: 7)
        bindMetaRegion(kv.vMeta, enc: enc, index: 8)
        out.bind(enc, index: 9)
        enc.setBuffer(partO.buffer, offset: partO.base, index: 10)
        enc.setBuffer(partML.buffer, offset: partML.base, index: 11)
        enc.setBytes(&params, length: MemoryLayout<MetalParams>.stride, index: 12)
        enc.setBytes(&tm, length: MemoryLayout<MetalTreeMeta>.stride, index: 13)
        blockList.bind(enc, index: 14)
        blockCount.bind(enc, index: 15)

        let gChunks = (gqa + 3) / 4
        enc.dispatchThreadgroups(
            MTLSize(width: resolvedSplits, height: config.nKV * gChunks, depth: (T + 15) / 16),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )

        if resolvedSplits > 1 {
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(psoCombine)
            enc.setBuffer(partO.buffer, offset: partO.base, index: 0)
            enc.setBuffer(partML.buffer, offset: partML.base, index: 1)
            bindOptionalTensor(gate, active: config.hasOutGate, enc: enc, index: 2)
            out.bind(enc, index: 3)
            enc.setBytes(&params, length: MemoryLayout<MetalParams>.stride, index: 4)
            enc.dispatchThreadgroups(
                MTLSize(width: config.nH, height: T, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
            )
        }
    }
}
