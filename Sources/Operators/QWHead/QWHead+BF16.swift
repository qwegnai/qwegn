// QWHead+BF16.swift — Batched BF16 LM-head GEMV and fused 2-stage
// deterministic GPU greedy argmax (BF16/F16/HQ4/AQ4/SHVQ) for M=1..8.
import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWGEMM

extension QWHead {
    /// Maximum Stage-1 key slots per batch row across all batched argmax kernels.
    public static func batchedArgmaxSlots(_ N: Int) -> Int {
        (N + 7) / 8
    }

    /// Batched BF16 LM-head GEMV (`W [N, K] bf16` * `x [M, K] bf16|f16` -> `y [M, N] f32`)
    /// with optional `tanh(x / softcap) * softcap` logit softcapping.
    public static func encodeBF16(
        _ enc: any MTLComputeCommandEncoder,
        hal: QWHal,
        w: QWTensorValue,
        x: QWTensorValue,
        y: QWTensorValue,
        geometry: QWModelGeometry,
        M: Int = 1,
        N: Int? = nil,
        softcap: Float = 0.0
    ) throws {
        let hidden = geometry.hidden
        let vocabN = N ?? geometry.vocab
        guard (1...16).contains(M), vocabN > 0, vocabN <= geometry.vocab, vocabN < (1 << 30),
              w.shape.dtype == .bf16, w.shape.elementCount >= vocabN * hidden,
              x.shape.dtype == .bf16 || x.shape.dtype == .f16, x.shape.elementCount == M * hidden,
              y.shape.dtype == .f32, y.shape.elementCount >= M * vocabN else {
            throw QWHeadError.badShape("BF16 head requires M=1...16, W[N,H] bf16, X[M,H] bf16|f16, Y[M,N] f32")
        }
        let elemAlign = (hidden % 8 == 0) ? 16 : 2
        try validateTensor(w, alignment: elemAlign, label: "BF16 head weight")
        try validateTensor(x, alignment: elemAlign, label: "BF16 head input")
        try validateTensor(y, alignment: 4, label: "BF16 head output")

        if M > 8 && vocabN >= (1 << 24) {
            let c0 = M / 2
            let c1 = M - c0
            let xDt = x.shape.dtype
            let x0 = QWTensorValue(region: x.region, offset: x.offset, shape: QWTensorShape([c0 * hidden], xDt))
            let y0 = QWTensorValue(region: y.region, offset: y.offset, shape: QWTensorShape([c0 * vocabN], .f32))
            let x1 = QWTensorValue(region: x.region, offset: x.offset + c0 * hidden * 2, shape: QWTensorShape([c1 * hidden], xDt))
            let y1 = QWTensorValue(region: y.region, offset: y.offset + c0 * vocabN * 4, shape: QWTensorShape([c1 * vocabN], .f32))
            try encodeBF16(enc, hal: hal, w: w, x: x0, y: y0, geometry: geometry, M: c0, N: vocabN, softcap: softcap)
            try encodeBF16(enc, hal: hal, w: w, x: x1, y: y1, geometry: geometry, M: c1, N: vocabN, softcap: softcap)
            return
        }

        let rowsPerTG = (M <= 4) ? 16 : 8
        let tgCount = (vocabN + rowsPerTG - 1) / rowsPerTG
        let psoName = (M > 8) ? "qwg_head_bf16_m16" : "qwg_head_bf16_m\(M)"
        let pso = try hal.pipeline(psoName)
        enc.setComputePipelineState(pso)
        w.bind(enc, index: 0)
        x.bind(enc, index: 1)
        y.bind(enc, index: 3)
        var nU = UInt32(vocabN) | (M > 8 ? (UInt32(M) << 24) : 0) | (x.shape.dtype == .f16 ? 0x8000_0000 : 0)
        var kU = UInt32(hidden)
        var cap = softcap
        enc.setBytes(&nU, length: 4, index: 4)
        enc.setBytes(&kU, length: 4, index: 5)
        enc.setBytes(&cap, length: 4, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: tgCount, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    public func encodeBF16(
        _ enc: any MTLComputeCommandEncoder,
        w: QWTensorValue,
        x: QWTensorValue,
        y: QWTensorValue,
        M: Int = 1,
        N: Int? = nil,
        softcap: Float = 0.0
    ) throws {
        try Self.encodeBF16(enc, hal: hal, w: w, x: x, y: y,
                            geometry: geometry, M: M, N: N, softcap: softcap)
    }

    private static func encodeBatchedArgmaxFin(
        _ enc: any MTLComputeCommandEncoder,
        hal: QWHal,
        keysHi: QWTensorValue,
        keysLo: QWTensorValue,
        tokens: QWTensorValue,
        slots: Int,
        rows: Int,
        softcap: Float = 0.0
    ) throws {
        let psoName = (softcap > 0) ? "qwg_head_argmax_batched_fin_softcap" : "qwg_head_argmax_batched_fin"
        let pso = try hal.pipeline(psoName)
        enc.setComputePipelineState(pso)
        keysHi.bind(enc, index: 0)
        keysLo.bind(enc, index: 1)
        tokens.bind(enc, index: 2)
        var nTGU = UInt32(slots)
        var mU = UInt32(rows)
        enc.setBytes(&nTGU, length: 4, index: 3)
        enc.setBytes(&mU, length: 4, index: 4)
        if softcap > 0 {
            var cap = softcap
            enc.setBytes(&cap, length: 4, index: 5)
        }
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
    }

    /// Fused 2-stage deterministic GPU batched greedy argmax writing `Int32[M]` directly
    /// without materializing `M * N * 4` FP32 logits in DRAM.
    public static func encodeBatchedGreedyArgmax(
        _ enc: any MTLComputeCommandEncoder,
        hal: QWHal,
        storage: QWHeadStorage,
        x: QWTensorValue,
        keysHi: QWTensorValue,
        keysLo: QWTensorValue,
        tokens: QWTensorValue,
        geometry: QWModelGeometry,
        M: Int = 1,
        N: Int? = nil,
        softcap: Float = 0.0,
        allowSinglePassM16: Bool = false
    ) throws {
        let hidden = geometry.hidden
        let vocabN = N ?? geometry.vocab
        guard (1...16).contains(M), vocabN > 0, vocabN <= geometry.vocab, vocabN < (1 << 30),
              keysHi.shape.dtype == .i32, keysLo.shape.dtype == .i32,
              tokens.shape.dtype == .i32, tokens.shape.elementCount >= M else {
            throw QWHeadError.badShape("batched greedy argmax requires M=1...16, valid N, and i32 key/token buffers")
        }
        try validateTensor(keysHi, alignment: 4, label: "argmax keysHi")
        try validateTensor(keysLo, alignment: 4, label: "argmax keysLo")
        try validateTensor(tokens, alignment: 4, label: "argmax tokens")

        let canSinglePassM16: Bool = {
            guard M > 8, allowSinglePassM16 else { return M <= 8 }
            guard vocabN < (1 << 24) else { return false }
            if case .q4g(let weight) = storage, !weight.view.isSHVQ {
                return weight.view.isIncumbentContainer || (weight.view.bits == 4 && weight.view.groupSize == 32)
            }
            return true
        }()
        if M > 8 && !canSinglePassM16 {
            let c0 = M / 2
            let c1 = M - c0
            let xDt = x.shape.dtype
            guard x.shape.elementCount == M * hidden else {
                throw QWHeadError.badShape("batched greedy argmax input elementCount mismatch")
            }
            let x0 = QWTensorValue(region: x.region, offset: x.offset, shape: QWTensorShape([c0 * hidden], xDt))
            let t0 = QWTensorValue(region: tokens.region, offset: tokens.offset, shape: QWTensorShape([c0], .i32))
            let x1 = QWTensorValue(region: x.region, offset: x.offset + c0 * hidden * 2, shape: QWTensorShape([c1 * hidden], xDt))
            let t1 = QWTensorValue(region: tokens.region, offset: tokens.offset + c0 * 4, shape: QWTensorShape([c1], .i32))
            try encodeBatchedGreedyArgmax(enc, hal: hal, storage: storage, x: x0, keysHi: keysHi, keysLo: keysLo,
                                          tokens: t0, geometry: geometry, M: c0, N: vocabN, softcap: softcap,
                                          allowSinglePassM16: allowSinglePassM16)
            enc.memoryBarrier(scope: .buffers)
            try encodeBatchedGreedyArgmax(enc, hal: hal, storage: storage, x: x1, keysHi: keysHi, keysLo: keysLo,
                                          tokens: t1, geometry: geometry, M: c1, N: vocabN, softcap: softcap,
                                          allowSinglePassM16: allowSinglePassM16)
            return
        }

        let stage1Slots: Int
        var finSoftcap: Float = 0.0
        switch storage {
        case .bf16(let w):
            guard w.shape.dtype == .bf16, w.shape.elementCount >= vocabN * hidden,
                  x.shape.dtype == .bf16 || x.shape.dtype == .f16, x.shape.elementCount == M * hidden else {
                throw QWHeadError.badShape("BF16 batched argmax weight/input mismatch")
            }
            let elemAlign = (hidden % 8 == 0) ? 16 : 2
            try validateTensor(w, alignment: elemAlign, label: "BF16 argmax weight")
            try validateTensor(x, alignment: elemAlign, label: "BF16 argmax input")
            let rowsPerTG = (M <= 4) ? 16 : 8
            stage1Slots = (vocabN + rowsPerTG - 1) / rowsPerTG
            guard keysHi.shape.elementCount >= M * stage1Slots,
                  keysLo.shape.elementCount >= M * stage1Slots else {
                throw QWHeadError.badShape("argmax keysHi/keysLo require >= \(M * stage1Slots) elements")
            }
            let pso = try hal.pipeline(M > 8 ? "qwg_head_bf16_argmax_m16" : "qwg_head_bf16_argmax_m\(M)")
            enc.setComputePipelineState(pso)
            w.bind(enc, index: 0)
            x.bind(enc, index: 1)
            keysHi.bind(enc, index: 2)
            keysLo.bind(enc, index: 3)
            var nU = UInt32(vocabN) | (M > 8 ? (UInt32(M) << 24) : 0) | (x.shape.dtype == .f16 ? 0x8000_0000 : 0)
            var kU = UInt32(hidden)
            var cap = softcap
            enc.setBytes(&nU, length: 4, index: 4)
            enc.setBytes(&kU, length: 4, index: 5)
            enc.setBytes(&cap, length: 4, index: 6)
            enc.dispatchThreadgroups(MTLSize(width: stage1Slots, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))

        case .f16(let w):
            guard w.shape.dtype == .f16, w.shape.elementCount >= vocabN * hidden,
                  x.shape.dtype == .f16, x.shape.elementCount == M * hidden else {
                throw QWHeadError.badShape("FP16 batched argmax weight/input mismatch")
            }
            let elemAlign = (hidden % 8 == 0) ? 16 : 2
            try validateTensor(w, alignment: elemAlign, label: "FP16 argmax weight")
            try validateTensor(x, alignment: elemAlign, label: "FP16 argmax input")
            let rowsPerTG = (M <= 4) ? 16 : 8
            stage1Slots = (vocabN + rowsPerTG - 1) / rowsPerTG
            guard keysHi.shape.elementCount >= M * stage1Slots,
                  keysLo.shape.elementCount >= M * stage1Slots else {
                throw QWHeadError.badShape("argmax keysHi/keysLo require >= \(M * stage1Slots) elements")
            }
            let pso = try hal.pipeline(M > 8 ? "qwg_head_f16_argmax_m16" : "qwg_head_f16_argmax_m\(M)")
            enc.setComputePipelineState(pso)
            w.bind(enc, index: 0)
            x.bind(enc, index: 1)
            keysHi.bind(enc, index: 2)
            keysLo.bind(enc, index: 3)
            var nU = UInt32(vocabN) | (M > 8 ? (UInt32(M) << 24) : 0)
            var kU = UInt32(hidden)
            var cap = softcap
            enc.setBytes(&nU, length: 4, index: 4)
            enc.setBytes(&kU, length: 4, index: 5)
            enc.setBytes(&cap, length: 4, index: 6)
            enc.dispatchThreadgroups(MTLSize(width: stage1Slots, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))

        case .q4g(let weight):
            let view = weight.view
            guard view.k == hidden, vocabN <= view.n,
                  x.shape.dtype == .f16, x.shape.elementCount == M * hidden else {
                throw QWHeadError.badShape("Q4G batched argmax weight/input mismatch")
            }
            try validateTensor(view.blocks, alignment: view.isSHVQ ? 2 : 4, label: "Q4G argmax blocks")
            try validateTensor(x, alignment: 16, label: "Q4G argmax input")
            if !view.isSHVQ && (view.isIncumbentContainer || (view.bits == 4 && view.groupSize == 32)) {
                stage1Slots = (vocabN + 7) / 8
                guard keysHi.shape.elementCount >= M * stage1Slots,
                      keysLo.shape.elementCount >= M * stage1Slots else {
                    throw QWHeadError.badShape("argmax keysHi/keysLo require >= \(M * stage1Slots) elements")
                }
                let isAQ4 = !view.isIncumbentContainer
                let pso = try hal.pipeline(M > 8 ? "qwg_head_q4g_argmax_m16" : "qwg_head_q4g_argmax_m\(M)", isAQ4: isAQ4)
                enc.setComputePipelineState(pso)
                view.blocks.bind(enc, index: 0)
                x.bind(enc, index: 1)
                keysHi.bind(enc, index: 2)
                keysLo.bind(enc, index: 3)
                var nU = UInt32(vocabN) | (M > 8 ? (UInt32(M) << 24) : 0)
                var kU = UInt32(hidden)
                var cap = softcap
                enc.setBytes(&nU, length: 4, index: 4)
                enc.setBytes(&kU, length: 4, index: 5)
                enc.setBytes(&cap, length: 4, index: 6)
                enc.dispatchThreadgroups(MTLSize(width: stage1Slots, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: M > 8 ? 256 : 128, height: 1, depth: 1))
            } else if view.isSHVQ && M > 8,
                      let m16Name = QWGemmAQ.headArgmaxM16KernelName(bits: view.bits, group: view.groupSize, isSHVQ: true) {
                stage1Slots = (vocabN + 15) / 16
                finSoftcap = softcap
                guard keysHi.shape.elementCount >= M * stage1Slots,
                      keysLo.shape.elementCount >= M * stage1Slots else {
                    throw QWHeadError.badShape("argmax keysHi/keysLo require >= \(M * stage1Slots) elements")
                }
                enc.setComputePipelineState(try hal.pipeline(m16Name))
                view.blocks.bind(enc, index: 0)
                x.bind(enc, index: 1)
                tokens.bind(enc, index: 3)
                keysHi.bind(enc, index: 6)
                keysLo.bind(enc, index: 7)
                var nU = UInt32(vocabN) | 0xC000_0000 | (UInt32(M) << 24)
                var kU = UInt32(hidden)
                enc.setBytes(&nU, length: 4, index: 4)
                enc.setBytes(&kU, length: 4, index: 5)
                enc.dispatchThreadgroups(MTLSize(width: stage1Slots, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            } else {
                guard let name = QWGemmAQ.headArgmaxKernelName(bits: view.bits, group: view.groupSize,
                                                               isSHVQ: view.isSHVQ) else {
                    throw QWHeadError.badShape("no argmax head kernel for \(view.bits)-bit group \(view.groupSize)")
                }
                stage1Slots = (vocabN + 31) / 32
                guard keysHi.shape.elementCount >= M * stage1Slots,
                      keysLo.shape.elementCount >= M * stage1Slots else {
                    throw QWHeadError.badShape("argmax keysHi/keysLo require >= \(M * stage1Slots) elements")
                }
                let pso = try hal.pipeline(name)
                enc.setComputePipelineState(pso)
                view.blocks.bind(enc, index: 0)
                tokens.bind(enc, index: 3)
                var kU = UInt32(hidden)
                enc.setBytes(&kU, length: 4, index: 5)
                finSoftcap = softcap
                if view.isSHVQ && M >= 2 && vocabN < (1 << 24) {
                    let psoM4 = QWGemmAQ.headArgmaxM4KernelName(bits: view.bits, group: view.groupSize, isSHVQ: true)
                        .flatMap { try? hal.pipeline($0) }
                    var activeIsM4 = false
                    var rowOffset = 0
                    while rowOffset < M {
                        let rem = M - rowOffset
                        let chunk = (psoM4 != nil)
                            ? ((rem > 4 && rem - 4 == 1) ? 3 : min(4, rem))
                            : min(8, rem)
                        let useM4 = (chunk >= 2 && chunk <= 4) && (psoM4 != nil)
                        if useM4 != activeIsM4 {
                            if let m4 = psoM4, useM4 {
                                enc.setComputePipelineState(m4)
                            } else {
                                enc.setComputePipelineState(pso)
                            }
                            activeIsM4 = useM4
                        }
                        let xChunk = (rowOffset == 0 && chunk == M)
                            ? x
                            : x.slice(byteOffset: rowOffset * hidden * 2, count: chunk * hidden, dtype: .f16)
                        let khChunk = keysHi.slice(byteOffset: rowOffset * stage1Slots * 4,
                                                   count: chunk * stage1Slots, dtype: .i32)
                        let klChunk = keysLo.slice(byteOffset: rowOffset * stage1Slots * 4,
                                                   count: chunk * stage1Slots, dtype: .i32)
                        var nU = UInt32(vocabN) | 0xC000_0000 | (chunk >= 2 ? (UInt32(chunk) << 24) : 0)
                        enc.setBytes(&nU, length: 4, index: 4)
                        xChunk.bind(enc, index: 1)
                        khChunk.bind(enc, index: 6)
                        klChunk.bind(enc, index: 7)
                        enc.dispatchThreadgroups(MTLSize(width: stage1Slots, height: 1, depth: 1),
                                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                        rowOffset += chunk
                    }
                } else {
                    var nU = UInt32(vocabN) | 0xC000_0000
                    enc.setBytes(&nU, length: 4, index: 4)
                    for m in 0..<M {
                        x.slice(byteOffset: m * hidden * 2, count: hidden, dtype: .f16).bind(enc, index: 1)
                        keysHi.slice(byteOffset: m * stage1Slots * 4, count: stage1Slots, dtype: .i32).bind(enc, index: 6)
                        keysLo.slice(byteOffset: m * stage1Slots * 4, count: stage1Slots, dtype: .i32).bind(enc, index: 7)
                        enc.dispatchThreadgroups(MTLSize(width: stage1Slots, height: 1, depth: 1),
                                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    }
                }
            }
        }

        enc.memoryBarrier(scope: .buffers)
        try encodeBatchedArgmaxFin(enc, hal: hal, keysHi: keysHi, keysLo: keysLo,
                                   tokens: tokens, slots: stage1Slots, rows: M,
                                   softcap: finSoftcap)
    }

    public func encodeBatchedGreedyArgmax(
        _ enc: any MTLComputeCommandEncoder,
        storage: QWHeadStorage,
        x: QWTensorValue,
        keysHi: QWTensorValue,
        keysLo: QWTensorValue,
        tokens: QWTensorValue,
        M: Int = 1,
        N: Int? = nil,
        softcap: Float = 0.0,
        allowSinglePassM16: Bool = false
    ) throws {
        try Self.encodeBatchedGreedyArgmax(
            enc, hal: hal, storage: storage, x: x,
            keysHi: keysHi, keysLo: keysLo, tokens: tokens,
            geometry: geometry, M: M, N: N, softcap: softcap,
            allowSinglePassM16: allowSinglePassM16
        )
    }
}
