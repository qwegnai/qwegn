// QWAttnPagedPrefillGeneralApply.swift — the standard path's paged
// prefill: one scalar kernel that attends a chunk of T queries at `pos`
// against all the keys in [0, pos+T), reading the K/V out of the paged KV
// pool through the page table. The 27B keeps its NAX pair.
import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWKV

public enum QWAttnPagedPrefillGeneralError: Error, CustomStringConvertible {
    case badShape(String)
    public var description: String {
        switch self {
        case .badShape(let s): return "QWAttnPagedPrefillGeneralError badShape: \(s)"
        }
    }
}

/// Encoder for `qwg_attn_prefill_paged_general`. One simdgroup per
/// (query, head); the head window masks the lanes past the head's own
/// width, and the scale is the geometry's.
public final class QWAttnPagedPrefillGeneral: Sendable {
    public let geometry: QWModelGeometry
    private let hal: QWHal
    private let pso: any MTLComputePipelineState
    private let psoSparse: any MTLComputePipelineState

    public init(hal: QWHal, geometry: QWModelGeometry) throws {
        self.geometry = geometry
        self.hal = hal
        self.pso = try hal.pipeline("qwg_attn_prefill_paged_general")
        self.psoSparse = try hal.pipeline("qwg_attn_prefill_paged_general_sparse")
    }

    /// Encode the chunk's attention over the pool. The chunk's own K/V must
    /// already be in the pool. The q row is `QW_QPROJ_OUT` wide and the
    /// output row `qHeads × headDim`: both must be **contiguous** (spec 08).
    public func encode(_ enc: any MTLComputeCommandEncoder,
                       qgate: QWTensorValue,
                       kv: (k: QWKVLayerView, v: QWKVLayerView),
                       out: QWTensorValue,
                       pos: Int, T: Int) throws {
        let wantQGate = T * geometry.qProjOut
        let wantOut = T * geometry.qHeads * geometry.headDim
        guard pos >= 0, T > 0, T <= 65535,
              qgate.shape.dtype == .f16, out.shape.dtype == .f16,
              qgate.shape.elementCount >= wantQGate,
              out.shape.elementCount >= wantOut,
              kv.k.mode == kv.v.mode,
              (kv.k.mode == .f16 || kv.k.mode == .q8 || kv.k.mode == .q4),
              kv.k.pageBytes == QWKVCache.pageBytes(mode: kv.k.mode, geometry: geometry),
              kv.v.pageBytes == QWKVCache.pageBytes(mode: kv.v.mode, geometry: geometry)
        else {
            throw QWAttnPagedPrefillGeneralError.badShape(
                "pos=\(pos) T=\(T) qgate=\(qgate.shape.elementCount) wantQ=\(wantQGate) "
                + "out=\(out.shape.elementCount) wantOut=\(wantOut) mode=\(kv.k.mode)")
        }
        let neededPages = (pos + T + 255) / 256
        guard kv.k.table.shape.elementCount >= neededPages,
              kv.v.table.shape.elementCount >= neededPages
        else {
            throw QWAttnPagedPrefillGeneralError.badShape(
                "page table too short: needed=\(neededPages) k=\(kv.k.table.shape.elementCount) "
                + "v=\(kv.v.table.shape.elementCount)")
        }

        enc.setComputePipelineState(pso)
        qgate.bind(enc, index: 0)
        enc.setBuffer(kv.k.pool.buffer, offset: kv.k.pool.base, index: 1)
        enc.setBuffer(kv.v.pool.buffer, offset: kv.v.pool.base, index: 2)
        kv.k.table.bind(enc, index: 3)
        kv.v.table.bind(enc, index: 4)
        out.bind(enc, index: 5)
        var posU = UInt32(pos)
        var q8U: UInt32
        switch kv.k.mode {
        case .f16: q8U = 0
        case .q8:  q8U = 1
        case .q4:  q8U = 2
        }
        enc.setBytes(&posU, length: 4, index: 6)
        enc.setBytes(&q8U, length: 4, index: 7)

        // One threadgroup per (head, query); 32 lanes onto the head.
        enc.dispatchThreadgroups(MTLSize(width: geometry.qHeads, height: T, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    /// Encode LSA sparse paged general prefill over `blockList` + unaligned tail.
    public func encodeSparse(_ enc: any MTLComputeCommandEncoder,
                             qgate: QWTensorValue,
                             kv: (k: QWKVLayerView, v: QWKVLayerView),
                             out: QWTensorValue,
                             blockList: QWTensorValue,
                             blockCount: QWTensorValue,
                             pos: Int, T: Int, listStride: Int) throws {
        let wantQGate = T * geometry.qProjOut
        let wantOut = T * geometry.qHeads * geometry.headDim
        guard pos >= 0, T > 0, T <= 65535,
              qgate.shape.dtype == .f16, out.shape.dtype == .f16,
              qgate.shape.elementCount >= wantQGate,
              out.shape.elementCount >= wantOut,
              kv.k.mode == kv.v.mode,
              (kv.k.mode == .f16 || kv.k.mode == .q8 || kv.k.mode == .q4),
              kv.k.pageBytes == QWKVCache.pageBytes(mode: kv.k.mode, geometry: geometry),
              kv.v.pageBytes == QWKVCache.pageBytes(mode: kv.v.mode, geometry: geometry)
        else {
            throw QWAttnPagedPrefillGeneralError.badShape("sparse pos=\(pos) T=\(T)")
        }
        let neededPages = (pos + T + 255) / 256
        guard kv.k.table.shape.elementCount >= neededPages,
              kv.v.table.shape.elementCount >= neededPages
        else {
            throw QWAttnPagedPrefillGeneralError.badShape("page table too short: needed=\(neededPages)")
        }

        enc.setComputePipelineState(psoSparse)
        qgate.bind(enc, index: 0)
        enc.setBuffer(kv.k.pool.buffer, offset: kv.k.pool.base, index: 1)
        enc.setBuffer(kv.v.pool.buffer, offset: kv.v.pool.base, index: 2)
        kv.k.table.bind(enc, index: 3)
        kv.v.table.bind(enc, index: 4)
        out.bind(enc, index: 5)
        var posU = UInt32(pos)
        var q8U: UInt32
        switch kv.k.mode {
        case .f16: q8U = 0
        case .q8:  q8U = 1
        case .q4:  q8U = 2
        }
        var strideU = UInt32(listStride)
        enc.setBytes(&posU, length: 4, index: 6)
        enc.setBytes(&q8U, length: 4, index: 7)
        blockList.bind(enc, index: 8)
        blockCount.bind(enc, index: 9)
        enc.setBytes(&strideU, length: 4, index: 10)

        enc.dispatchThreadgroups(MTLSize(width: geometry.qHeads, height: T, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }
}
