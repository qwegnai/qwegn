// DriverWeightLoading+EchoDraft.swift — Dedicated mHC-MTP draft layer weight loader.

import QWAlloc
import QWCore
import QWHAL

extension QWEngineDriver.WeightLoader {
    mutating func loadDedicatedEchoDraftWeights() throws -> QWEngineDriver.EchoDraftWeights {
        let h = geometry.hidden
        let streams = geometry.mhcConfig?.streams ?? 4
        let mtpPfx = hasTensor("mtp_layers.0.enorm.weight")
            ? "mtp_layers.0."
            : (QWEngineDriver.P + "mtp_layers.0.")
        let dblk = mtpPfx + "decoder_layer."

        let eNorm = try f16wNorm(mtpPfx + "enorm.weight", h)
        let hNorm = try f16wNorm(mtpPfx + "hnorm.weight", h)
        let eProj = try q4(mtpPfx + "e_proj.weight", n: h, k: h)
        let hProj = try q4(mtpPfx + "h_proj.weight", n: h, k: h)

        let ln1 = try f16wNorm(dblk + "input_layernorm.weight", h)
        let ln2 = try f16wNorm(dblk + "post_attention_layernorm.weight", h)
        let (gu, down, moe) = try loadLayerMLP(
            dblk, isMoE: geometry.numExperts > 0, geometry: geometry, layer: geometry.numLayers)
        let attn = try loadGQAAttn(dblk, geometry: geometry, includeMBSA: false)
        let mhcW: MHCLayerWeights?
        if let mhc = geometry.mhcConfig {
            mhcW = try loadMHCOrContractedHC(dblk, mhc: mhc)
        } else {
            mhcW = nil
        }
        let dedicatedLayer = LayerW(
            ln1: ln1, ln2: ln2, gu: gu, down: down,
            kind: moe == nil ? .fullAttention : .moe,
            gdn: nil, attn: attn, moe: moe, mhc: mhcW)

        let hcPrefix = hasTensor(mtpPfx + "mhc_contract.weight")
            ? (mtpPfx + "mhc_contract")
            : (mtpPfx + "hc_head")
        let hcPair = try loadExitHeadPair(prefix: hcPrefix, streams: streams)
        let sharedHeadNorm = try f16wNorm(mtpPfx + "final_layernorm.weight", h)

        return QWEngineDriver.EchoDraftWeights(
            eNorm: eNorm, hNorm: hNorm, eProj: eProj, hProj: hProj,
            hcHeadW: hcPair.w, hcHeadB: hcPair.b,
            sharedHeadNorm: sharedHeadNorm, routerBias: nil,
            dedicatedLayer: dedicatedLayer)
    }
}
