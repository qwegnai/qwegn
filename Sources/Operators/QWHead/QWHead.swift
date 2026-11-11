// QWHead.swift — C15 encoder: split-K LM-head GEMV (raced S) + greedy
// argmax (two-stage deterministic — no 64-bit atomics exist on this
// toolchain, measured; see qwg_head_argmax's law). The distribution path
// (T>0) is QWSampler on the CPU-mapped logits (spec 15 race, DECISIONS).
//
// ENCODER NEVER ALLOCATES (C09 steady-state law): y and the S>1 partials
// scratch are caller-owned QWTensorValues; sizes via yElems/partialElems.
import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWGEMM

public enum QWHeadError: Error, CustomStringConvertible {
    case badShape(String)
    /// A path that reads the incumbent's container was handed the family's.
    /// Refused BY NAME: the two layouts differ, and the incumbent's kernel
    /// would read the family's blocks as if they were its own.
    case unsupportedContainer(String)
    public var description: String {
        switch self {
        case .badShape(let s): return "QWHeadError badShape: \(s)"
        case .unsupportedContainer(let s): return "QWHeadError unsupportedContainer: \(s)"
        }
    }
}

public struct QWHeadWeight: Sendable {
    public let view: QWWeightView
    public init?(blocks: QWTensorValue, n: Int, k: Int, bits: Int = 4, groupSize: Int = 32,
                 isRotated: Bool, isIncumbentContainer: Bool? = nil, isSHVQ: Bool = false) {
        guard let v = QWWeightView(blocks: blocks, n: n, k: k, bits: bits,
                                   groupSize: groupSize, isRotated: isRotated,
                                   isIncumbentContainer: isIncumbentContainer, isSHVQ: isSHVQ) else { return nil }
        self.view = v
    }
    public init?(nibbles: QWTensorValue, scales: QWTensorValue? = nil, mins: QWTensorValue? = nil,
                 n: Int, k: Int, isRotated: Bool, isIncumbentContainer: Bool? = nil, isSHVQ: Bool = false) {
        self.init(blocks: nibbles, n: n, k: k, isRotated: isRotated,
                  isIncumbentContainer: isIncumbentContainer, isSHVQ: isSHVQ)
    }
}

public enum QWHeadStorage: Sendable {
    case f16(QWTensorValue)
    case q4g(QWHeadWeight)
}

public final class QWHead {
    public var splits: Int = 1
    public static let rowsPerTG = 8

    public let geometry: QWModelGeometry
    private let hal: QWHal
    private let psoGemv: any MTLComputePipelineState
    private let psoGemvArgmax: any MTLComputePipelineState
    private let psoCombine: any MTLComputePipelineState
    private let psoArgmax: any MTLComputePipelineState
    private let psoArgmaxFin: any MTLComputePipelineState
    private let f2plan: QWGemmPlan

    public init(hal: QWHal, geometry: QWModelGeometry) throws {
        self.geometry = geometry
        self.hal = hal
        self.psoGemv = try hal.pipeline("qwg_head_gemv_split")
        self.psoGemvArgmax = try hal.pipeline("qwg_head_gemv_q4g_argmax")
        self.psoCombine = try hal.pipeline("qwg_head_combine")
        self.psoArgmax = try hal.pipeline("qwg_head_argmax")
        self.psoArgmaxFin = try hal.pipeline("qwg_head_argmax_fin")
        self.f2plan = QWGemm.plan(QWGemmShape(m: 1, n: geometry.vocab, k: geometry.hidden),
                                  dtype: .q4g, epi: .none)
    }

    public static func yElems(_ N: Int) -> Int { N }                       // f32
    public static func partialElems(_ N: Int, _ S: Int) -> Int { N &* S }  // f32 (S>1)
    public static func argmaxSlots(_ N: Int) -> Int { (N + 255) / 256 }
    public static func gemvArgmaxSlots(_ N: Int) -> Int { bandFor(N) }
    public static func bandFor(_ N: Int) -> Int { (N + rowsPerTG - 1) / rowsPerTG }

    /// Small verify windows share each FP16 weight load across all input rows.
    /// The widths come from the model's **own** geometry — the 27B's used to
    /// stand in for them here.
    public static func encodeF16(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                 w: QWTensorValue, x: QWTensorValue, y: QWTensorValue,
                                 geometry: QWModelGeometry,
                                 M: Int = 1, N: Int? = nil) throws {
        let hidden = geometry.hidden
        let N = N ?? geometry.vocab
        guard (1...16).contains(M), N > 0, N <= geometry.vocab,
              w.shape.dtype == .f16, w.shape.elementCount == N * hidden,
              x.shape.dtype == .f16, x.shape.elementCount == M * hidden,
              y.shape.dtype == .f32, y.shape.elementCount == M * N else {
            throw QWHeadError.badShape("FP16 head requires M=1...16, W[N,H], X[M,H], Y[M,N]")
        }
        func validate(_ tensor: QWTensorValue, alignment: Int) throws {
            guard tensor.offset >= 0, tensor.region.base >= 0,
                  tensor.offset <= tensor.region.length,
                  tensor.bytes <= tensor.region.length - tensor.offset,
                  tensor.region.base <= tensor.region.buffer.length,
                  tensor.region.length <= tensor.region.buffer.length - tensor.region.base,
                  (tensor.region.base + tensor.offset) % alignment == 0 else {
                throw QWHeadError.badShape("FP16 head buffer extent or alignment is invalid")
            }
        }
        try validate(w, alignment: 16)
        try validate(x, alignment: 16)
        try validate(y, alignment: 4)
        if M > 8 {
            let c0 = M / 2
            let c1 = M - c0
            let x0 = QWTensorValue(region: x.region, offset: x.offset, shape: QWTensorShape([c0 * hidden], .f16))
            let y0 = QWTensorValue(region: y.region, offset: y.offset, shape: QWTensorShape([c0 * N], .f32))
            let x1 = QWTensorValue(region: x.region, offset: x.offset + c0 * hidden * 2, shape: QWTensorShape([c1 * hidden], .f16))
            let y1 = QWTensorValue(region: y.region, offset: y.offset + c0 * N * 4, shape: QWTensorShape([c1 * N], .f32))
            try encodeF16(enc, hal: hal, w: w, x: x0, y: y0, geometry: geometry, M: c0, N: N)
            try encodeF16(enc, hal: hal, w: w, x: x1, y: y1, geometry: geometry, M: c1, N: N)
            return
        }
        let kernel = M == 1 ? "qwg_gemv_f16_v4"
            : "qwg_head_f16_m\(M)"
        let pipeline = try hal.pipeline(kernel)
        enc.setComputePipelineState(pipeline)
        w.bind(enc, index: 0)
        x.bind(enc, index: 1)
        y.bind(enc, index: 3)
        var n = UInt32(N), k = UInt32(hidden)
        enc.setBytes(&n, length: 4, index: 4)
        enc.setBytes(&k, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: N, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    public func encode(_ enc: any MTLComputeCommandEncoder, storage: QWHeadStorage,
                       x: QWTensorValue, gs: QWTensorValue? = nil, y: QWTensorValue,
                       M: Int = 1) throws {
        switch storage {
        case .f16(let weight):
            try Self.encodeF16(enc, hal: hal, w: weight, x: x, y: y, geometry: geometry, M: M)
        case .q4g(let weight):
            if M == 1 {
                try encode(enc, x: x, gs: gs, w: weight, y: y)
            } else {
                try Self.encodeBatched(enc, hal: hal, view: weight.view,
                                       x: x, gs: gs, y: y, m: M)
            }
        }
    }

    public static func encodeBatched(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                     view: QWWeightView, x: QWTensorValue,
                                     gs: QWTensorValue? = nil, y: QWTensorValue,
                                     m: Int) throws {
        _ = gs
        guard (2...16).contains(m) else {
            throw QWHeadError.badShape("batched HQ4_G32 head requires m=2…16 (got \(m))")
        }
        guard x.shape.dtype == .f16, x.shape.elementCount == m * view.k,
              y.shape.dtype == .f32, y.shape.elementCount >= m * view.n else {
            throw QWHeadError.badShape("batched HQ4_G32 head x/y shape-dtype mismatch")
        }
        if !view.isIncumbentContainer && (view.isSHVQ || !(view.bits == 4 && view.groupSize == 32)) {
            var r0 = 0
            while r0 < m {
                let chunk = min(4, m - r0)
                let xSlice = QWTensorValue(region: x.region,
                                           offset: x.offset + r0 * view.k * 2,
                                           shape: QWTensorShape([chunk, view.k], .f16))
                let ySlice = QWTensorValue(region: y.region,
                                           offset: y.offset + r0 * view.n * 4,
                                           shape: QWTensorShape([chunk, view.n], .f32))
                try QWGemmAQ.encode(enc, hal: hal, blocks: view.blocks, x: xSlice, y: ySlice,
                                    n: view.n, k: view.k, bits: view.bits, group: view.groupSize,
                                    rows: chunk, isSHVQ: view.isSHVQ)
                r0 += chunk
            }
            return
        }
        try encodeBatchedQ4GKernel(enc, hal: hal, view: view, x: x, y: y, m: m)
    }

    /// Dispatches the one-pass HQ4_G32/AQ4_G32 batched head; at m >= 4 each x load is
    /// shared across two rows (bit-identical per row).
    public static func encodeBatchedQ4GKernel(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                              view: QWWeightView, x: QWTensorValue,
                                              y: QWTensorValue, m: Int) throws {
        if m > 8 {
            let c0 = m / 2
            let c1 = m - c0
            let x0 = QWTensorValue(region: x.region, offset: x.offset, shape: QWTensorShape([c0 * view.k], .f16))
            let y0 = QWTensorValue(region: y.region, offset: y.offset, shape: QWTensorShape([c0 * view.n], .f32))
            let x1 = QWTensorValue(region: x.region, offset: x.offset + c0 * view.k * 2, shape: QWTensorShape([c1 * view.k], .f16))
            let y1 = QWTensorValue(region: y.region, offset: y.offset + c0 * view.n * 4, shape: QWTensorShape([c1 * view.n], .f32))
            try encodeBatchedQ4GKernel(enc, hal: hal, view: view, x: x0, y: y0, m: c0)
            try encodeBatchedQ4GKernel(enc, hal: hal, view: view, x: x1, y: y1, m: c1)
            return
        }
        let rowPaired = (view.isIncumbentContainer || (view.bits == 4 && view.groupSize == 32)) && m >= 4
        let pso = try hal.pipeline(rowPaired ? "qwg_head_q4g_rp_m\(m)" : "qwg_head_q4g_m\(m)")
        enc.setComputePipelineState(pso)
        view.blocks.bind(enc, index: 0)
        x.bind(enc, index: 1)
        y.bind(enc, index: 3)
        var n = UInt32(view.n) | (view.isIncumbentContainer ? 0 : 0x8000_0000), k = UInt32(view.k)
        enc.setBytes(&n, length: 4, index: 4)
        enc.setBytes(&k, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: (view.n + 7) / 8, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: rowPaired ? 128 : 256, height: 1, depth: 1))
    }

    public static func encodeQ4g(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                 weight: QWHeadWeight, x: QWTensorValue,
                                 gs: QWTensorValue? = nil, y: QWTensorValue) throws {
        let view = weight.view
        if !view.isIncumbentContainer {
            try QWGemmAQ.encode(enc, hal: hal, blocks: view.blocks, x: x, y: y,
                                n: view.n, k: view.k, bits: view.bits, group: view.groupSize,
                                isSHVQ: view.isSHVQ)
            return
        }
        let plan = QWGemm.plan(QWGemmShape(m: 1, n: view.n, k: view.k),
                               dtype: .q4g, epi: .none)
        try QWGemm.encode(enc, hal: hal, plan: plan, x: x,
                          xGroupSum: gs, wQ4: view, y: y)
    }

    /// GEMV: x [K] f16 → y [N] f32. One dispatch (S=1) or GEMV+combine (S>1).
    public func encode(_ enc: any MTLComputeCommandEncoder,
                       x: QWTensorValue, gs: QWTensorValue? = nil, w: QWHeadWeight,
                       y: QWTensorValue, partials: QWTensorValue? = nil) throws {
        let K = w.view.k, N = w.view.n
        guard K == geometry.hidden else { throw QWHeadError.badShape("head K=\(K) ≠ hidden") }
        guard x.shape.elementCount == K, x.shape.dtype == .f16,
              y.shape.elementCount >= Self.yElems(N), y.shape.dtype == .f32 else {
            throw QWHeadError.badShape("x/y shape-dtype mismatch")
        }
        if !w.view.isIncumbentContainer {
            try QWGemmAQ.encode(enc, hal: hal, blocks: w.view.blocks, x: x, y: y,
                                n: N, k: K, bits: w.view.bits, group: w.view.groupSize,
                                isSHVQ: w.view.isSHVQ)
            return
        }
        let S = splits
        let nBand = Self.bandFor(N)
        if S == 1 {
            try QWGemm.encode(enc, hal: hal, plan: f2plan, x: x,
                              xGroupSum: gs, wQ4: w.view, y: y)
            return
        }
        if S > 1 {
            guard let pp = partials,
                  pp.shape.elementCount >= Self.partialElems(N, S), pp.shape.dtype == .f32 else {
                throw QWHeadError.badShape("partials required for S>1")
            }
        }
        var nU = UInt32(N), kU = UInt32(K), sU = UInt32(S), nbU = UInt32(nBand)
        enc.setComputePipelineState(psoGemv)
        w.view.blocks.bind(enc, index: 0)
        x.bind(enc, index: 1)
        y.bind(enc, index: 3)
        enc.setBytes(&nU, length: 4, index: 4)
        enc.setBytes(&kU, length: 4, index: 5)
        enc.setBytes(&sU, length: 4, index: 8)
        enc.setBytes(&nbU, length: 4, index: 9)
        partials?.bind(enc, index: 10)
        enc.dispatchThreadgroups(MTLSize(width: nBand &* S, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        if S > 1 {
            enc.setComputePipelineState(psoCombine)
            partials!.bind(enc, index: 0)
            y.bind(enc, index: 1)
            enc.setBytes(&nU, length: 4, index: 2)
            enc.setBytes(&sU, length: 4, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: (N + 255) / 256, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
    }

    /// Greedy argmax (T=0): y [N] f32 → token u32. Two dispatches; the
    /// key-slot buffers (u32 ×2 arrays) are caller-owned. No 64-bit atomics.
    public static func encodeArgmax(_ enc: any MTLComputeCommandEncoder,
                                    psoArgmax: any MTLComputePipelineState,
                                    psoArgmaxFin: any MTLComputePipelineState,
                                    y: QWTensorValue, keysHi: QWTensorValue, keysLo: QWTensorValue,
                                    token: QWTensorValue, N: Int) throws {
        let nTG = argmaxSlots(N)
        guard y.shape.elementCount >= N, y.shape.dtype == .f32,
              keysHi.shape.elementCount >= nTG, keysHi.shape.dtype == .i32,
              keysLo.shape.elementCount >= nTG, keysLo.shape.dtype == .i32,
              token.shape.elementCount >= 1, token.shape.dtype == .i32 else {
            throw QWHeadError.badShape("argmax buffers mismatch (N=\(N))")
        }
        var nU = UInt32(N)
        enc.setComputePipelineState(psoArgmax)
        y.bind(enc, index: 0)
        keysHi.bind(enc, index: 1)
        keysLo.bind(enc, index: 2)
        enc.setBytes(&nU, length: 4, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: nTG, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.memoryBarrier(scope: .buffers)
        var nTG32 = UInt32(nTG)
        enc.setComputePipelineState(psoArgmaxFin)
        keysHi.bind(enc, index: 0)
        keysLo.bind(enc, index: 1)
        token.bind(enc, index: 2)
        enc.setBytes(&nTG32, length: 4, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
    }

    public static func encodeArgmax(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                    y: QWTensorValue, keysHi: QWTensorValue, keysLo: QWTensorValue,
                                    token: QWTensorValue, N: Int) throws {
        try encodeArgmax(enc,
                         psoArgmax: try hal.pipeline("qwg_head_argmax"),
                         psoArgmaxFin: try hal.pipeline("qwg_head_argmax_fin"),
                         y: y, keysHi: keysHi, keysLo: keysLo, token: token, N: N)
    }

    public func encodeArgmax(_ enc: any MTLComputeCommandEncoder,
                             y: QWTensorValue, keysHi: QWTensorValue, keysLo: QWTensorValue,
                             token: QWTensorValue, N: Int) throws {
        try Self.encodeArgmax(enc, psoArgmax: psoArgmax, psoArgmaxFin: psoArgmaxFin,
                              y: y, keysHi: keysHi, keysLo: keysLo, token: token, N: N)
    }

    /// Fused LM-head GEMV + block argmax epilogue followed by final key
    /// reduction. The bank's own basis says which kernel reads it: the
    /// incumbent's over `Block_HQ4_G32`, the family's twin over its blocks.
    public func encodeGreedyQ4g(_ enc: any MTLComputeCommandEncoder,
                                x: QWTensorValue, w: QWHeadWeight,
                                y: QWTensorValue, keysHi: QWTensorValue, keysLo: QWTensorValue,
                                token: QWTensorValue,
                                writeLogits: Bool = true) throws {
        let K = w.view.k, N = w.view.n
        let nTG = (N + 31) / 32
        guard K == geometry.hidden,
              x.shape.elementCount == K, x.shape.dtype == .f16,
              y.shape.elementCount >= Self.yElems(N), y.shape.dtype == .f32,
              keysHi.shape.elementCount >= nTG, keysHi.shape.dtype == .i32,
              keysLo.shape.elementCount >= nTG, keysLo.shape.dtype == .i32,
              token.shape.elementCount >= 1, token.shape.dtype == .i32 else {
            throw QWHeadError.badShape("encodeGreedyQ4g buffers mismatch (N=\(N), K=\(K))")
        }
        var nU = UInt32(N) | 0x8000_0000 | (writeLogits ? 0 : 0x4000_0000), kU = UInt32(K), nTG32 = UInt32(nTG)
        if w.view.isIncumbentContainer {
            enc.setComputePipelineState(psoGemvArgmax)
        } else {
            guard let name = QWGemmAQ.headArgmaxKernelName(bits: w.view.bits,
                                                           group: w.view.groupSize,
                                                           isSHVQ: w.view.isSHVQ) else {
                throw QWHeadError.badShape("no affine-quantised head kernel for "
                                           + "\(w.view.bits)-bit group \(w.view.groupSize)")
            }
            enc.setComputePipelineState(try hal.pipeline(name))
        }
        w.view.blocks.bind(enc, index: 0)
        x.bind(enc, index: 1)
        y.bind(enc, index: 3)
        enc.setBytes(&nU, length: 4, index: 4)
        enc.setBytes(&kU, length: 4, index: 5)
        keysHi.bind(enc, index: 6)
        keysLo.bind(enc, index: 7)
        enc.dispatchThreadgroups(MTLSize(width: nTG, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.memoryBarrier(scope: .buffers)
        enc.setComputePipelineState(psoArgmaxFin)
        keysHi.bind(enc, index: 0)
        keysLo.bind(enc, index: 1)
        token.bind(enc, index: 2)
        enc.setBytes(&nTG32, length: 4, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
    }
}
