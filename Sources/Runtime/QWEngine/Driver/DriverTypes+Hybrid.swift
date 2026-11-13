// DriverTypes+Hybrid.swift — Hybrid mHC exit-head and shared-last-layer ECHO draft
// weight descriptors for QWEngineDriver.

import Foundation
import Metal
import QWAlloc
import QWCore
import QWGEMM
import QWQuant

extension QWEngineDriver {
    struct MHCExitHeadWeights: Sendable {
        let headW: QWTensorValue
        let headB: QWTensorValue
    }

    struct EchoDraftWeights: @unchecked Sendable {
        let eNorm: QWTensorValue
        let hNorm: QWTensorValue
        let eProj: QWWeightView
        let hProj: QWWeightView
        let hcHeadW: QWTensorValue
        let hcHeadB: QWTensorValue
        let sharedHeadNorm: QWTensorValue
        let routerBias: QWTensorValue?

        var residentBuffers: [any MTLBuffer] {
            var out: [any MTLBuffer] = [
                eNorm.region.buffer,
                hNorm.region.buffer,
                eProj.blocks.region.buffer,
                hProj.blocks.region.buffer,
                hcHeadW.region.buffer,
                hcHeadB.region.buffer,
                sharedHeadNorm.region.buffer
            ]
            if let b = routerBias {
                out.append(b.region.buffer)
            }
            return out
        }
    }
}
