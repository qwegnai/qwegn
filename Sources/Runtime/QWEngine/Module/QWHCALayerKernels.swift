// QWHCALayerKernels.swift — Hybrid Compressed Attention (HCA) layer kernels
// for dense (`HCALayerKernel`) and sparse-MoE (`HCAMoELayerKernel`) blocks.

import Foundation
import Metal
import QWAlloc
import QWAttn
import QWCore
import QWHAL

final class HCALayerKernel: DenseLayerKernel, QWLayerKernel, @unchecked Sendable {
    static let key = QWEnginePlan.LayerKernelKey(attention: .hcaCompressed, mlp: .dense, ple: false)

    static func make() -> any QWLayerKernel { HCALayerKernel() }

    override func encodeAttention(_ context: QWLayerDecodeContext) throws {
        let driver = context.driver
        var enc = context.enc
        func mark(_ stage: String) {
            driver.profileStage(stage)
            if let a = driver.activeEnc { enc = a }
        }
        guard let hca = driver.hcaOperator,
              let hcaW = context.weights.hca,
              let wFusedIn = hcaW.wFusedIn,
              let wQUpGate = hcaW.wQUpGate,
              let woB = hcaW.woB else {
            throw QWEngineDriverError.state("HCA layer \(context.layer) missing operator or weights")
        }
        let sc = driver.sc
        let H = driver.geometry.hidden
        let pos = context.position
        let hcaIdx = driver.hcaIndexMap[context.layer]
        let fusedInRow = driver.prefixRows(hca.fusedInScratch, rows: 1, cols: hca.inStride, dt: .f16)
        let qrRow = driver.prefixRows(hca.qrScratch, rows: 1, cols: hca.qLoraRank, dt: .f16)
        let qUpRow = driver.prefixRows(hca.qUpGateScratch, rows: 1, cols: hca.qUpStride, dt: .f16)
        let zOutRow = driver.prefixRows(hca.zGatedScratch, rows: 1, cols: hca.zOutStride, dt: .f16)

        mark("hca.in_proj")
        try driver.gemvH16T4(enc, n: wFusedIn.n, k: H, x: sc.ln1, gs: sc.gs, wQ4: wFusedIn, y16: fusedInRow)
        mark("hca.in_prep")
        hca.encodeInPrepAndCompress(enc, hcaW: hcaW, hcaLayer: hcaIdx, pos0: pos, rows: 1)
        mark("hca.q_up_proj")
        try driver.gemvH16T4(enc, n: wQUpGate.n, k: hca.qLoraRank, x: qrRow, gs: sc.gs, wQ4: wQUpGate, y16: qUpRow)
        mark("hca.attend")
        hca.encodeQPrepAttendVHAAndWoA(
            enc, hcaW: hcaW, zGatedOut: zOutRow,
            hcaLayer: hcaIdx, pos0: pos, rows: 1,
            stage: QWEngineDriver.decodeProfileEnabled
                ? { name in driver.profileStage(name); return driver.activeEnc } : nil)
        enc = driver.activeEnc ?? enc
        mark("hca.o_proj")
        try driver.gemvH16T4(enc, n: H, k: hca.zOutStride, x: zOutRow, gs: sc.gs48, wQ4: woB, y16: sc.o16)
    }

    override func encodeAttentionPrefill(_ context: QWLayerPrefillContext) throws {
        let driver = context.driver
        var enc = context.enc
        func mark(_ stage: String) -> (any MTLComputeCommandEncoder)? {
            driver.profileStage(stage)
            if let a = driver.activeEnc { enc = a }
            return driver.activeEnc
        }
        guard let hca = driver.hcaOperator,
              let hcaW = context.weights.hca,
              let wFusedIn = hcaW.wFusedIn,
              let wQUpGate = hcaW.wQUpGate,
              let woB = hcaW.woB else {
            throw QWEngineDriverError.state("HCA layer \(context.layer) missing operator or weights")
        }
        let psc = driver.psc
        let H = driver.geometry.hidden
        let T = context.rows
        let pos = context.position
        let hcaIdx = driver.hcaIndexMap[context.layer]
        let activeLn1 = driver.prefixRows(psc.ln1, rows: T, cols: H, dt: .f16)
        let fusedInRows = driver.prefixRows(hca.fusedInScratch, rows: T, cols: hca.inStride, dt: .f16)
        let qrRows = driver.prefixRows(hca.qrScratch, rows: T, cols: hca.qLoraRank, dt: .f16)
        let qUpRows = driver.prefixRows(hca.qUpGateScratch, rows: T, cols: hca.qUpStride, dt: .f16)
        let zOutRows = driver.prefixRows(hca.zGatedScratch, rows: T, cols: hca.zOutStride, dt: .f16)

        _ = mark("hca.in_proj")
        driver.gemmBatchedH16(enc, m: T, n: wFusedIn.n, k: H, x: activeLn1, wQ4: wFusedIn, y16: fusedInRows)
        _ = mark("hca.in_prep")
        hca.encodeInPrepAndCompress(enc, hcaW: hcaW, hcaLayer: hcaIdx, pos0: pos, rows: T)
        _ = mark("hca.q_up_proj")
        driver.gemmBatchedH16(enc, m: T, n: wQUpGate.n, k: hca.qLoraRank, x: qrRows, wQ4: wQUpGate, y16: qUpRows)
        hca.encodeQPrepAttendVHAAndWoA(
            enc, hcaW: hcaW, zGatedOut: zOutRows,
            hcaLayer: hcaIdx, pos0: pos, rows: T,
            stage: QWEngineDriver.prefillProfileEnabled ? mark : nil)
        _ = mark("hca.o_proj")
        enc = driver.activeEnc ?? enc
        driver.gemmBatchedH16(
            enc, m: T, n: H, k: hca.zOutStride,
            x: zOutRows, wQ4: woB,
            y16: driver.prefixRows(psc.o16, rows: T, cols: H, dt: .f16))
    }
}

final class HCAMoELayerKernel: DenseLayerKernel, QWLayerKernel, @unchecked Sendable {
    static let key = QWEnginePlan.LayerKernelKey(attention: .hcaCompressed, mlp: .moe, ple: false)

    static func make() -> any QWLayerKernel { HCAMoELayerKernel() }

    private let attention = HCALayerKernel()

    override func encodeAttention(_ context: QWLayerDecodeContext) throws {
        try attention.encodeAttention(context)
    }

    override func encodeAttentionPrefill(_ context: QWLayerPrefillContext) throws {
        try attention.encodeAttentionPrefill(context)
    }

    override func encodeMLP(_ context: QWLayerDecodeContext) throws {
        let driver = context.driver
        let sc = driver.sc
        guard let moe = context.weights.moe else {
            throw QWEngineDriverError.state("an HCA MoE layer has no MoE weights")
        }
        guard let s = sc.moe else {
            throw QWEngineDriverError.state("the decode scratch has no MoE tensors")
        }
        var active = driver.activeEnc ?? context.enc
        try QWMoEChain.encode(enc: &active, hal: driver.hal, moe: moe, s: s,
                              x16: sc.ln2, dn32: sc.dn32, dn16: sc.dn16,
                              rows: 1, geometry: driver.geometry, driver: driver)
    }

    override func encodeMLPPrefill(_ context: QWLayerPrefillContext) throws {
        let driver = context.driver
        let psc = driver.psc
        let geometry = driver.geometry
        let H = geometry.hidden
        let T = context.rows
        guard let moe = context.weights.moe else {
            throw QWEngineDriverError.state("an HCA MoE layer has no MoE weights")
        }
        guard let s = psc.moe else {
            throw QWEngineDriverError.state("the prefill scratch has no MoE tensors")
        }
        var active = driver.activeEnc ?? context.enc
        try QWMoEChain.encode(enc: &active, hal: driver.hal, moe: moe, s: s,
                              x16: driver.prefixRows(psc.ln2, rows: T, cols: H, dt: .f16),
                              dn32: psc.dn32, dn16: psc.dn16,
                              rows: T, geometry: geometry, driver: driver)
    }
}
