// QWSigLIP2Vision.swift — SigLIP2 NaFlex vision tower weights and CPU reference forward.

import Accelerate
import Foundation
import QWCore

public final class QWSigLIP2LayerW: @unchecked Sendable {
    let ln1G: [Float], ln1B: [Float]
    let ln2G: [Float], ln2B: [Float]
    let qW: [Float], qB: [Float]?
    let kW: [Float], kB: [Float]?
    let vW: [Float], vB: [Float]?
    let outW: [Float], outB: [Float]?
    let fc1W: [Float], fc1B: [Float]?
    let fc2W: [Float], fc2B: [Float]?

    init(_ ck: QWVisionCheckpoint, _ p: String, dim: Int, mlpInner: Int) throws {
        ln1G = try ck.f32("\(p).layer_norm1.weight", want: [dim])
        ln1B = try loadOptionalTensor(ck, keys: ["\(p).layer_norm1.bias"], want: [dim]) ?? [Float](repeating: 0, count: dim)
        ln2G = try ck.f32("\(p).layer_norm2.weight", want: [dim])
        ln2B = try loadOptionalTensor(ck, keys: ["\(p).layer_norm2.bias"], want: [dim]) ?? [Float](repeating: 0, count: dim)
        qW = try ck.f32("\(p).self_attn.q_proj.weight", want: [dim, dim])
        qB = try loadOptionalTensor(ck, keys: ["\(p).self_attn.q_proj.bias"], want: [dim])
        kW = try ck.f32("\(p).self_attn.k_proj.weight", want: [dim, dim])
        kB = try loadOptionalTensor(ck, keys: ["\(p).self_attn.k_proj.bias"], want: [dim])
        vW = try ck.f32("\(p).self_attn.v_proj.weight", want: [dim, dim])
        vB = try loadOptionalTensor(ck, keys: ["\(p).self_attn.v_proj.bias"], want: [dim])
        outW = try ck.f32("\(p).self_attn.out_proj.weight", want: [dim, dim])
        outB = try loadOptionalTensor(ck, keys: ["\(p).self_attn.out_proj.bias"], want: [dim])
        fc1W = try ck.f32("\(p).mlp.fc1.weight", want: [mlpInner, dim])
        fc1B = try loadOptionalTensor(ck, keys: ["\(p).mlp.fc1.bias"], want: [mlpInner])
        fc2W = try ck.f32("\(p).mlp.fc2.weight", want: [dim, mlpInner])
        fc2B = try loadOptionalTensor(ck, keys: ["\(p).mlp.fc2.bias"], want: [dim])
    }

    var residentBytes: Int {
        var count = ln1G.count + ln1B.count + ln2G.count + ln2B.count
        count += qW.count + (qB?.count ?? 0) + kW.count + (kB?.count ?? 0)
        count += vW.count + (vB?.count ?? 0) + outW.count + (outB?.count ?? 0)
        count += fc1W.count + (fc1B?.count ?? 0) + fc2W.count + (fc2B?.count ?? 0)
        return count * MemoryLayout<Float>.stride
    }
}

/// Weights for SigLIP2 / NaFlex fixed-patch + 2D bilinear interpolation vision tower.
public final class QWSigLIP2VisionWeights: @unchecked Sendable {
    public let config: QWVisionConfig
    public let patchW: [Float], patchB: [Float]?
    public let posEmbed: [Float]
    public let layers: [QWSigLIP2LayerW]
    public let postNormG: [Float]?, postNormB: [Float]?
    public let projNormG: [Float]?, projNormB: [Float]?
    public let projNormPreMerge: Bool
    public let fc1W: [Float]?, fc1B: [Float]?
    public let fc2W: [Float]?, fc2B: [Float]?
    public let projMid: Int
    public let outDim: Int

    public var residentBytes: Int {
        var top = patchW.count + (patchB?.count ?? 0) + posEmbed.count
        top += (postNormG?.count ?? 0) + (postNormB?.count ?? 0)
        top += (projNormG?.count ?? 0) + (projNormB?.count ?? 0)
        top += (fc1W?.count ?? 0) + (fc1B?.count ?? 0) + (fc2W?.count ?? 0) + (fc2B?.count ?? 0)
        return top * MemoryLayout<Float>.stride + layers.reduce(0) { $0 + $1.residentBytes }
    }

    public init(checkpoint ck: QWVisionCheckpoint) throws {
        let cfg = ck.config
        let d = cfg.dim
        self.config = cfg
        let sigPeWant = (ck.shape("embeddings.patch_embedding.weight")?.count == 4)
            ? [d, 3, cfg.patch, cfg.patch]
            : [d, cfg.patchChannels]
        self.patchW = try ck.f32("embeddings.patch_embedding.weight", want: sigPeWant)
        self.patchB = try loadOptionalTensor(ck, keys: ["embeddings.patch_embedding.bias"], want: [d])
        self.posEmbed = try loadFirstTensor(
            ck,
            keys: ["embeddings.position_embedding.weight", "embeddings.position_embedding"],
            want: [cfg.numPosEmbeds, d]
        )

        let pfx = ck.has("encoder.layers.0.layer_norm1.weight") ? "encoder.layers" : "layers"
        var depth = 0
        while ck.has("\(pfx).\(depth).layer_norm1.weight") { depth += 1 }
        guard depth > 0 else { throw QWVisionError.missingTensor("\(pfx).0.layer_norm1.weight") }
        self.layers = try (0..<depth).map { try QWSigLIP2LayerW(ck, "\(pfx).\($0)", dim: d, mlpInner: cfg.mlpInner) }

        if ck.has("post_layernorm.weight") {
            self.postNormG = try ck.f32("post_layernorm.weight", want: [d])
            self.postNormB = try loadOptionalTensor(ck, keys: ["post_layernorm.bias"], want: [d]) ?? [Float](repeating: 0, count: d)
        } else {
            self.postNormG = nil
            self.postNormB = nil
        }

        let merged = cfg.merged
        if let ns = ck.shape("merger.norm.weight") {
            let normDim = ns.first ?? merged
            self.projNormPreMerge = (normDim == d && normDim != merged)
            self.projNormG = try ck.f32("merger.norm.weight", want: [normDim])
            self.projNormB = try loadOptionalTensor(ck, keys: ["merger.norm.bias"], want: [normDim]) ?? [Float](repeating: 0, count: normDim)
        } else {
            self.projNormPreMerge = false
            self.projNormG = nil
            self.projNormB = nil
        }
        if let fc1S = ck.shape("merger.linear_fc1.weight"), let fc2S = ck.shape("merger.linear_fc2.weight") {
            self.projMid = fc1S[0]
            self.outDim = fc2S[0]
            self.fc1W = try ck.f32("merger.linear_fc1.weight", want: [projMid, merged])
            self.fc1B = try loadOptionalTensor(ck, keys: ["merger.linear_fc1.bias"], want: [projMid])
            self.fc2W = try ck.f32("merger.linear_fc2.weight", want: [outDim, projMid])
            self.fc2B = try loadOptionalTensor(ck, keys: ["merger.linear_fc2.bias"], want: [outDim])
        } else {
            self.projMid = merged
            self.outDim = merged
            self.fc1W = nil; self.fc1B = nil; self.fc2W = nil; self.fc2B = nil
        }
        ck.releaseBacking()
    }
}

/// CPU FP32 reference evaluator for SigLIP2 NaFlex vision tower.
public final class QWSigLIP2VisionRef: @unchecked Sendable {
    public let w: QWSigLIP2VisionWeights
    public init(weights: QWSigLIP2VisionWeights) { self.w = weights }

    public func encode(patches: [Float], plan: QWVisionGridPlan, bins: QWVisionRef.InterpBins) -> [Float] {
        let c = w.config
        let d = c.dim
        let nPatch = plan.nPatch
        let nToken = plan.nToken
        guard nPatch > 0, nToken > 0 else { return [] }

        var s = [Float](repeating: 0, count: nPatch * d)
        runLinearProj(out: &s, x: patches, rows: nPatch, w: w.patchW, bias: w.patchB, nOut: d, nIn: c.patchChannels)
        foldPosEmbed(into: &s, bins: bins, dim: d)

        var normed = [Float](repeating: 0, count: nPatch * d)
        var qBuf = [Float](repeating: 0, count: nPatch * d)
        var kBuf = [Float](repeating: 0, count: nPatch * d)
        var vBuf = [Float](repeating: 0, count: nPatch * d)
        var attnOut = [Float](repeating: 0, count: nPatch * d)
        var branch = [Float](repeating: 0, count: nPatch * d)
        var mlpH = [Float](repeating: 0, count: nPatch * c.mlpInner)
        var geluScratch = [Float](repeating: 0, count: nPatch * c.mlpInner)

        for layer in w.layers {
            qwLayerNorm(s, into: &normed, gamma: layer.ln1G, beta: layer.ln1B, rows: nPatch, dim: d, eps: c.eps)
            runLinearProj(out: &qBuf, x: normed, rows: nPatch, w: layer.qW, bias: layer.qB, nOut: d, nIn: d)
            runLinearProj(out: &kBuf, x: normed, rows: nPatch, w: layer.kW, bias: layer.kB, nOut: d, nIn: d)
            runLinearProj(out: &vBuf, x: normed, rows: nPatch, w: layer.vW, bias: layer.vB, nOut: d, nIn: d)
            runBidirectionalGQAAttn(
                qBuf: qBuf, kBuf: kBuf, vBuf: vBuf, out: &attnOut,
                nSeq: nPatch, heads: c.heads, kvHeads: c.heads, headDim: c.headDim, ropeCosSin: nil
            )
            runLinearProj(out: &branch, x: attnOut, rows: nPatch, w: layer.outW, bias: layer.outB, nOut: d, nIn: d)
            addRowInto(&s, src: branch)

            qwLayerNorm(s, into: &normed, gamma: layer.ln2G, beta: layer.ln2B, rows: nPatch, dim: d, eps: c.eps)
            runLinearProj(out: &mlpH, x: normed, rows: nPatch, w: layer.fc1W, bias: layer.fc1B, nOut: c.mlpInner, nIn: d)
            mlpH.withUnsafeMutableBufferPointer { mb in
                geluScratch.withUnsafeMutableBufferPointer { sb in
                    guard let mp = mb.baseAddress, let sp = sb.baseAddress else { return }
                    qwApplyGeluTanhInPlace(mp, count: nPatch * c.mlpInner, scratch: sp)
                }
            }
            runLinearProj(out: &branch, x: mlpH, rows: nPatch, w: layer.fc2W, bias: layer.fc2B, nOut: d, nIn: c.mlpInner)
            addRowInto(&s, src: branch)
        }

        if let pg = w.postNormG, let pb = w.postNormB {
            qwLayerNorm(s, into: &normed, gamma: pg, beta: pb, rows: nPatch, dim: d, eps: c.eps)
            s = normed
        }
        if let ng = w.projNormG, let nb = w.projNormB {
            if w.projNormPreMerge {
                qwLayerNorm(s, into: &normed, gamma: ng, beta: nb, rows: nPatch, dim: d, eps: c.eps)
                s = normed
            } else {
                var mNormed = [Float](repeating: 0, count: nToken * c.merged)
                qwLayerNorm(s, into: &mNormed, gamma: ng, beta: nb, rows: nToken, dim: c.merged, eps: c.eps)
                s = mNormed
            }
        }
        guard let fc1W = w.fc1W, let fc2W = w.fc2W else { return s }
        var mid = [Float](repeating: 0, count: nToken * w.projMid)
        runLinearProj(out: &mid, x: s, rows: nToken, w: fc1W, bias: w.fc1B, nOut: w.projMid, nIn: c.merged, act: qwGeluErf)
        var out = [Float](repeating: 0, count: nToken * w.outDim)
        runLinearProj(out: &out, x: mid, rows: nToken, w: fc2W, bias: w.fc2B, nOut: w.outDim, nIn: w.projMid)
        return out
    }

    private func foldPosEmbed(into s: inout [Float], bins: QWVisionRef.InterpBins, dim: Int) {
        let numPos = w.posEmbed.count / dim
        guard numPos > 0, bins.nPatch > 0 else { return }
        let vDim = vDSP_Length(dim)
        bins.idx.withUnsafeBufferPointer { ib in
            bins.wgt.withUnsafeBufferPointer { wb in
                w.posEmbed.withUnsafeBufferPointer { tb in
                    s.withUnsafeMutableBufferPointer { sb in
                        guard let sBase = sb.baseAddress, let tBase = tb.baseAddress,
                              let iBase = ib.baseAddress, let wBase = wb.baseAddress else { return }
                        for p in 0..<bins.nPatch {
                            let sRow = sBase + p * dim
                            for c in 0..<4 {
                                let idx = Int(iBase[p * 4 + c])
                                guard idx >= 0, idx < numPos else { continue }
                                var wc = wBase[p * 4 + c]
                                vDSP_vsma(tBase + idx * dim, 1, &wc, sRow, 1, sRow, 1, vDim)
                            }
                        }
                    }
                }
            }
        }
    }
}
