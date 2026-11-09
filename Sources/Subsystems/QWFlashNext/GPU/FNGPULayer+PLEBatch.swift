import Foundation
import Metal
import QWAlloc
import QWCore
import QWGEMM
import QWHAL

extension FNGPULayerPasses {
    static func pleBatchKeyGate(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                p: FNGPUPLE, s: FNGPUScratch, rows: Int, strided: Bool = true) throws {
        let W = s.width, H = s.hidden, C = s.hc, kvRow = W + H
        if strided {
            try QWGemm.encodeFnHCNorm(enc, hal: hal, x: s.pleKV, weight: p.normKey, xn: s.pleKeyN,
                                      rows: rows, width: W, hidden: H, eps: p.eps, xStride: kvRow)
            barrier(enc)
            let values = s.pleKV.slice(byteOffset: W * 4, count: (rows - 1) * kvRow + H, dtype: .f32)
            try QWGemm.encodeFnPLEGate(enc, hal: hal, keyNorm: s.pleKeyN, queryNorm: s.pleQueryN,
                                       vals: values, gv: s.pleGV, rows: rows, hidden: H, hc: C, valStride: kvRow)
        } else {
            for b in 0..<rows {
                let key = s.pleKV.slice(byteOffset: b * kvRow * 4, count: W, dtype: .f32)
                let keyNorm = s.pleKeyN.slice(byteOffset: b * W * 4, count: W, dtype: .f32)
                try QWGemm.encodeFnHCNorm(enc, hal: hal, x: key, weight: p.normKey, xn: keyNorm,
                                          rows: 1, width: W, hidden: H, eps: p.eps)
            }
            barrier(enc)
            for b in 0..<rows {
                let values = s.pleKV.slice(byteOffset: (b * kvRow + W) * 4, count: H, dtype: .f32)
                let keyNorm = s.pleKeyN.slice(byteOffset: b * W * 4, count: W, dtype: .f32)
                let queryNorm = s.pleQueryN.slice(byteOffset: b * W * 4, count: W, dtype: .f32)
                let gv = s.pleGV.slice(byteOffset: b * W * 4, count: W, dtype: .f32)
                try QWGemm.encodeFnPLEGate(enc, hal: hal, keyNorm: keyNorm, queryNorm: queryNorm,
                                           vals: values, gv: gv, rows: 1, hidden: H, hc: C)
            }
        }
    }
}
