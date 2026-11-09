import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWKV

public enum QWAttnPagedPrefillError: Error, CustomStringConvertible {
    case badShape(String)
    public var description: String { switch self { case .badShape(let s): return "QWAttnPagedPrefillError badShape: \(s)" } }
}

/// Metal encoder for paged prefill FlashAttention (`qwg_attn_prefill_paged_nax_pair`).
/// Attends a chunk of T queries starting at `pos` against all keys in [0, pos+T)
/// residing in the paged KV cache pools.
public final class QWAttnPagedPrefill: Sendable {
    public let geometry: QWModelGeometry
    private let hal: QWHal
    private let psoF16: any MTLComputePipelineState
    private let psoQ8: any MTLComputePipelineState
    private let psoQ4: any MTLComputePipelineState
    private let psoSparseF16: any MTLComputePipelineState
    private let psoSparseQ8: any MTLComputePipelineState
    private let psoSparseQ4: any MTLComputePipelineState

    public init(hal: QWHal, geometry: QWModelGeometry) throws {
        self.geometry = geometry
        self.hal = hal
        self.psoF16 = try hal.pipeline("qwg_attn_prefill_paged_nax_pair")
        self.psoQ8 = try hal.pipeline("qwg_attn_prefill_paged_nax_pair_q8")
        self.psoQ4 = try hal.pipeline("qwg_attn_prefill_paged_nax_pair_q4")
        self.psoSparseF16 = try hal.pipeline("qwg_attn_prefill_paged_nax_sparse_pair")
        self.psoSparseQ8 = try hal.pipeline("qwg_attn_prefill_paged_nax_sparse_pair_q8")
        self.psoSparseQ4 = try hal.pipeline("qwg_attn_prefill_paged_nax_sparse_pair_q4")
    }

    /// Encode paged FlashAttention over the KV cache.
    /// - Parameters:
    ///   - enc: active compute command encoder
    public func encode(_ enc: any MTLComputeCommandEncoder,
                       qgate: QWTensorValue,
                       kv: (k: QWKVLayerView, v: QWKVLayerView),
                       out: QWTensorValue,
                       pos: Int, T: Int) throws {
        let sp = (T + 31) / 32 * 32
        let wantQGate = sp * geometry.qProjOut
        let wantOut = T * geometry.qHeads * geometry.headDim
        guard pos >= 0, T > 0,
              qgate.shape.dtype == .f16, out.shape.dtype == .f16,
              qgate.shape.elementCount >= wantQGate,
              out.shape.elementCount >= wantOut,
              kv.k.mode == kv.v.mode,
              kv.k.pageBytes == QWKVCache.pageBytes(mode: kv.k.mode, geometry: geometry),
              kv.v.pageBytes == QWKVCache.pageBytes(mode: kv.v.mode, geometry: geometry)
        else {
            throw QWAttnPagedPrefillError.badShape("pos=\(pos) T=\(T) qgate=\(qgate.shape.elementCount) wantQ=\(wantQGate) out=\(out.shape.elementCount) wantOut=\(wantOut)")
        }

        let neededPages = (pos + T + 255) / 256
        guard kv.k.table.shape.elementCount >= neededPages,
              kv.v.table.shape.elementCount >= neededPages
        else {
            throw QWAttnPagedPrefillError.badShape("page table too short: needed=\(neededPages) k=\(kv.k.table.shape.elementCount) v=\(kv.v.table.shape.elementCount)")
        }

        let pso: any MTLComputePipelineState
        switch kv.k.mode {
        case .f16: pso = psoF16
        case .q8:  pso = psoQ8
        case .q4:  pso = psoQ4
        }

        enc.setComputePipelineState(pso)
        qgate.bind(enc, index: 0)
        enc.setBuffer(kv.k.pool.buffer, offset: kv.k.pool.base, index: 1)
        enc.setBuffer(kv.v.pool.buffer, offset: kv.v.pool.base, index: 2)
        kv.k.table.bind(enc, index: 3)
        kv.v.table.bind(enc, index: 4)
        out.bind(enc, index: 5)
        var posU = UInt32(pos)
        var tU = UInt32(T)
        enc.setBytes(&posU, length: 4, index: 6)
        enc.setBytes(&tU, length: 4, index: 7)

        let numTiles = (T + 31) / 32
        let gqaRatio = max(1, geometry.qHeads / max(1, geometry.kvHeads))
        let totalThreadgroups = numTiles * geometry.kvHeads
        enc.dispatchThreadgroups(MTLSize(width: totalThreadgroups, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 64 * gqaRatio, height: 1, depth: 1))
    }

    /// Encode LSA sparse paged FlashAttention over `blockList` + local chunk tail.
    public func encodeSparse(_ enc: any MTLComputeCommandEncoder,
                             qgate: QWTensorValue,
                             kv: (k: QWKVLayerView, v: QWKVLayerView),
                             out: QWTensorValue,
                             blockList: QWTensorValue,
                             blockCount: QWTensorValue,
                             pos: Int, T: Int, listStride: Int) throws {
        let sp = (T + 31) / 32 * 32
        let wantQGate = sp * geometry.qProjOut
        let wantOut = T * geometry.qHeads * geometry.headDim
        guard pos >= 0, T > 0,
              qgate.shape.dtype == .f16, out.shape.dtype == .f16,
              qgate.shape.elementCount >= wantQGate,
              out.shape.elementCount >= wantOut,
              kv.k.mode == kv.v.mode,
              kv.k.pageBytes == QWKVCache.pageBytes(mode: kv.k.mode, geometry: geometry),
              kv.v.pageBytes == QWKVCache.pageBytes(mode: kv.v.mode, geometry: geometry)
        else {
            throw QWAttnPagedPrefillError.badShape("sparse pos=\(pos) T=\(T)")
        }
        let neededPages = (pos + T + 255) / 256
        guard kv.k.table.shape.elementCount >= neededPages,
              kv.v.table.shape.elementCount >= neededPages
        else {
            throw QWAttnPagedPrefillError.badShape("page table too short: needed=\(neededPages)")
        }

        let pso: any MTLComputePipelineState
        switch kv.k.mode {
        case .f16: pso = psoSparseF16
        case .q8:  pso = psoSparseQ8
        case .q4:  pso = psoSparseQ4
        }

        enc.setComputePipelineState(pso)
        qgate.bind(enc, index: 0)
        enc.setBuffer(kv.k.pool.buffer, offset: kv.k.pool.base, index: 1)
        enc.setBuffer(kv.v.pool.buffer, offset: kv.v.pool.base, index: 2)
        kv.k.table.bind(enc, index: 3)
        kv.v.table.bind(enc, index: 4)
        out.bind(enc, index: 5)
        var posU = UInt32(pos)
        var tU = UInt32(T)
        var strideU = UInt32(listStride)
        enc.setBytes(&posU, length: 4, index: 6)
        enc.setBytes(&tU, length: 4, index: 7)
        blockList.bind(enc, index: 8)
        blockCount.bind(enc, index: 9)
        enc.setBytes(&strideU, length: 4, index: 10)

        let numTiles = (T + 31) / 32
        let gqaRatio = max(1, geometry.qHeads / max(1, geometry.kvHeads))
        let totalThreadgroups = numTiles * geometry.kvHeads
        enc.dispatchThreadgroups(MTLSize(width: totalThreadgroups, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 64 * gqaRatio, height: 1, depth: 1))
    }
}
