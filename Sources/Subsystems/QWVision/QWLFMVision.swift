// QWLFMVision.swift — Hybrid LFM3-VL (VisionConv2d + 2D Axial-RoPE GQA) and SigLIP2 NaFlex ViT.

import Accelerate
import Foundation
import QWCore

@_silgen_name("cblas_sgemm$NEWLAPACK")
private func qw_lfm_sgemm(
    _ order: Int32, _ transA: Int32, _ transB: Int32,
    _ m: Int32, _ n: Int32, _ k: Int32, _ alpha: Float,
    _ a: UnsafePointer<Float>, _ lda: Int32,
    _ b: UnsafePointer<Float>, _ ldb: Int32, _ beta: Float,
    _ c: UnsafeMutablePointer<Float>, _ ldc: Int32
)

internal func loadOptionalTensor(_ ck: QWVisionCheckpoint, keys: [String], want: [Int]? = nil) throws -> [Float]? {
    for key in keys where ck.shape(key) != nil {
        return try ck.f32(key, want: want)
    }
    return nil
}

internal func loadFirstTensor(_ ck: QWVisionCheckpoint, keys: [String], want: [Int]? = nil) throws -> [Float] {
    if let hit = try loadOptionalTensor(ck, keys: keys, want: want) {
        return hit
    }
    throw QWVisionError.missingTensor(keys.joined(separator: " | "))
}

internal func qwRMSNorm(_ x: [Float], into out: inout [Float], gamma: [Float], rows: Int, dim: Int, eps: Float) {
    guard rows > 0, dim > 0, x.count >= rows * dim, out.count >= rows * dim, gamma.count >= dim else { return }
    let vDim = vDSP_Length(dim)
    x.withUnsafeBufferPointer { xb in
        out.withUnsafeMutableBufferPointer { ob in
            gamma.withUnsafeBufferPointer { gb in
                guard let xp = xb.baseAddress, let op = ob.baseAddress, let gp = gb.baseAddress else { return }
                for r in 0..<rows {
                    let xr = xp + r * dim
                    let yr = op + r * dim
                    var sq: Float = 0
                    vDSP_svesq(xr, 1, &sq, vDim)
                    var scale = Float(1.0 / (Double(sq) / Double(dim) + Double(eps)).squareRoot())
                    vDSP_vsmul(xr, 1, &scale, yr, 1, vDim)
                    vDSP_vmul(yr, 1, gp, 1, yr, 1, vDim)
                }
            }
        }
    }
}

internal func addVectors(_ dst: UnsafeMutablePointer<Float>, _ addend: UnsafePointer<Float>, count: Int) {
    vDSP_vadd(dst, 1, addend, 1, dst, 1, vDSP_Length(count))
}

internal func addRowInto(_ dst: inout [Float], src: [Float]) {
    let count = min(dst.count, src.count)
    dst.withUnsafeMutableBufferPointer { db in
        src.withUnsafeBufferPointer { sb in
            guard let d = db.baseAddress, let s = sb.baseAddress else { return }
            addVectors(d, s, count: count)
        }
    }
}

internal func runLinearProj(
    out: inout [Float],
    x: [Float],
    rows: Int,
    w: [Float],
    bias: [Float]?,
    nOut: Int,
    nIn: Int,
    act: ((Float) -> Float)? = nil
) {
    out.withUnsafeMutableBufferPointer { ob in
        guard let op = ob.baseAddress else { return }
        qwGemmRef(y: op, ldY: nOut, yBase: 0, x: x, rows: rows, w: w, nOut: nOut, nIn: nIn, bias: bias, act: act)
    }
}

// Lfm3VlVisionRotaryEmbedding: [freq_h, freq_w] per half-head, rotated via rotate_half pairs (j, j + headDim/2).
private func build2DAxialRope(posIds: [Int32], nPatch: Int, headDim: Int, ropeTheta: Double) -> (cos: [Float], sin: [Float]) {
    let halfPairs = headDim / 2
    let invCount = max(1, halfPairs / 2)
    let spatialDim = max(2, halfPairs)
    var invFreq = [Float](repeating: 0, count: invCount)
    for i in 0..<invCount {
        invFreq[i] = Float(1.0 / pow(ropeTheta, Double(2 * i) / Double(spatialDim)))
    }
    var cos = [Float](repeating: 1, count: nPatch * halfPairs)
    var sin = [Float](repeating: 0, count: nPatch * halfPairs)
    for p in 0..<nPatch {
        let py = Float(posIds[p * 2])
        let px = Float(posIds[p * 2 + 1])
        let off = p * halfPairs
        for i in 0..<invCount {
            let ay = py * invFreq[i]
            let ax = px * invFreq[i]
            cos[off + i] = cosf(ay)
            sin[off + i] = sinf(ay)
            if invCount + i < halfPairs {
                cos[off + invCount + i] = cosf(ax)
                sin[off + invCount + i] = sinf(ax)
            }
        }
    }
    return (cos, sin)
}

internal func runBidirectionalGQAAttn(
    qBuf: [Float],
    kBuf: [Float],
    vBuf: [Float],
    out: inout [Float],
    nSeq: Int,
    heads: Int,
    kvHeads: Int,
    headDim: Int,
    ropeCosSin: (cos: [Float], sin: [Float])?
) {
    let qDim = heads * headDim
    let kvDim = kvHeads * headDim
    let kvRepeat = max(1, heads / max(1, kvHeads))
    let scale = Float(1.0 / Double(headDim).squareRoot())

    var qHead = [Float](repeating: 0, count: heads * nSeq * headDim)
    var kHead = [Float](repeating: 0, count: kvHeads * nSeq * headDim)
    var vHead = [Float](repeating: 0, count: kvHeads * nSeq * headDim)
    var scores = [Float](repeating: 0, count: heads * nSeq * nSeq)

    let halfPairs = headDim / 2
    for row in 0..<nSeq {
        let qOff = row * qDim
        let kvOff = row * kvDim
        let rOff = row * halfPairs
        for h in 0..<heads {
            let dstBase = (h * nSeq + row) * headDim
            let srcBase = qOff + h * headDim
            if let cs = ropeCosSin {
                for j in 0..<halfPairs {
                    let j0 = j, j1 = j + halfPairs
                    let cc = cs.cos[rOff + j], ss = cs.sin[rOff + j]
                    qHead[dstBase + j0] = qBuf[srcBase + j0] * cc - qBuf[srcBase + j1] * ss
                    qHead[dstBase + j1] = qBuf[srcBase + j1] * cc + qBuf[srcBase + j0] * ss
                }
            } else {
                for d in 0..<headDim { qHead[dstBase + d] = qBuf[srcBase + d] }
            }
        }
        for kh in 0..<kvHeads {
            let dstBase = (kh * nSeq + row) * headDim
            let srcBase = kvOff + kh * headDim
            if let cs = ropeCosSin {
                for j in 0..<halfPairs {
                    let j0 = j, j1 = j + halfPairs
                    let cc = cs.cos[rOff + j], ss = cs.sin[rOff + j]
                    kHead[dstBase + j0] = kBuf[srcBase + j0] * cc - kBuf[srcBase + j1] * ss
                    kHead[dstBase + j1] = kBuf[srcBase + j1] * cc + kBuf[srcBase + j0] * ss
                }
            } else {
                for d in 0..<headDim { kHead[dstBase + d] = kBuf[srcBase + d] }
            }
            for d in 0..<headDim { vHead[dstBase + d] = vBuf[srcBase + d] }
        }
    }

    qHead.withUnsafeBufferPointer { qhb in
        kHead.withUnsafeBufferPointer { khb in
            vHead.withUnsafeBufferPointer { vhb in
                scores.withUnsafeMutableBufferPointer { scb in
                    out.withUnsafeMutableBufferPointer { ob in
                        guard let q0 = qhb.baseAddress, let k0 = khb.baseAddress, let v0 = vhb.baseAddress,
                              let sc0 = scb.baseAddress, let o0 = ob.baseAddress else { return }
                        nonisolated(unsafe) let qu = q0, ku = k0, vu = v0, scu = sc0, ou = o0
                        DispatchQueue.concurrentPerform(iterations: heads) { head in
                            let qPtr = qu + head * nSeq * headDim
                            let kh = min(kvHeads - 1, head / kvRepeat)
                            let kPtr = ku + kh * nSeq * headDim
                            let vPtr = vu + kh * nSeq * headDim
                            let scPtr = scu + head * nSeq * nSeq
                            qw_lfm_sgemm(101, 111, 112, Int32(nSeq), Int32(nSeq), Int32(headDim),
                                         scale, qPtr, Int32(headDim), kPtr, Int32(headDim), 0, scPtr, Int32(nSeq))
                            var n32 = Int32(nSeq), vN = vDSP_Length(nSeq)
                            for qi in 0..<nSeq {
                                let row = scPtr + qi * nSeq
                                var mx: Float = -.greatestFiniteMagnitude
                                vDSP_maxv(row, 1, &mx, vN)
                                var negMx = -mx
                                vDSP_vsadd(row, 1, &negMx, row, 1, vN)
                                vvexpf(row, row, &n32)
                                var sum: Float = 0
                                vDSP_sve(row, 1, &sum, vN)
                                var inv = sum > 0 ? (1.0 / sum) : 0
                                vDSP_vsmul(row, 1, &inv, row, 1, vN)
                            }
                            qw_lfm_sgemm(101, 111, 111, Int32(nSeq), Int32(headDim), Int32(nSeq),
                                         1.0, scPtr, Int32(nSeq), vPtr, Int32(headDim), 0, ou + head * headDim, Int32(qDim))
                        }
                    }
                }
            }
        }
    }
}

public final class QWLFM3VisionBlockW: @unchecked Sendable {
    public enum Kind: Sendable {
        case conv(inProjW: [Float]?, inProjB: [Float]?, convW: [Float], convB: [Float]?, outProjW: [Float], outProjB: [Float]?, kernelSize: Int)
        case attention(qW: [Float], qB: [Float]?, kW: [Float], kB: [Float]?, vW: [Float], vB: [Float]?, outW: [Float], outB: [Float]?, qNormW: [Float]?, kNormW: [Float]?)
    }

    public enum MLP: Sendable {
        case gelu(fc1W: [Float], fc1B: [Float]?, fc2W: [Float], fc2B: [Float]?)
        case swiglu(w1: [Float], w2: [Float], w3: [Float])
    }

    let norm1G: [Float], norm1B: [Float]?
    let norm2G: [Float], norm2B: [Float]?
    let kind: Kind
    let mlp: MLP

    init(_ ck: QWVisionCheckpoint, _ p: String, dim: Int, kvDim: Int, headDim: Int, mlpInner: Int, isConv: Bool, kernelSize: Int) throws {
        self.norm1G = try loadFirstTensor(ck, keys: ["\(p).operator_norm.weight", "\(p).norm1.weight", "\(p).layer_norm1.weight", "\(p).input_layernorm.weight"], want: [dim])
        self.norm1B = try loadOptionalTensor(ck, keys: ["\(p).norm1.bias", "\(p).layer_norm1.bias", "\(p).input_layernorm.bias"], want: [dim])
        self.norm2G = try loadFirstTensor(ck, keys: ["\(p).ffn_norm.weight", "\(p).norm2.weight", "\(p).layer_norm2.weight", "\(p).post_attention_layernorm.weight"], want: [dim])
        self.norm2B = try loadOptionalTensor(ck, keys: ["\(p).norm2.bias", "\(p).layer_norm2.bias", "\(p).post_attention_layernorm.bias"], want: [dim])
        if isConv {
            let inProjW = try loadOptionalTensor(ck, keys: ["\(p).conv.in_proj.weight"], want: [3 * dim, dim])
            let inProjB = try loadOptionalTensor(ck, keys: ["\(p).conv.in_proj.bias"], want: [3 * dim])
            let convKey = ck.has("\(p).conv.spatial_conv.weight") ? "\(p).conv.spatial_conv.weight" : "\(p).conv2d.weight"
            let convWant = (ck.shape(convKey)?.count == 4)
                ? [dim, 1, kernelSize, kernelSize]
                : [dim, kernelSize, kernelSize]
            let cw = try ck.f32(convKey, want: convWant)
            let cb = try loadOptionalTensor(ck, keys: ["\(p).conv.spatial_conv.bias", "\(p).conv2d.bias"], want: [dim])
            let ow = try loadFirstTensor(ck, keys: ["\(p).conv.out_proj.weight", "\(p).out_proj.weight", "\(p).conv_proj.weight", "\(p).proj.weight"], want: [dim, dim])
            let ob = try loadOptionalTensor(ck, keys: ["\(p).conv.out_proj.bias", "\(p).out_proj.bias", "\(p).conv_proj.bias", "\(p).proj.bias"], want: [dim])
            self.kind = .conv(inProjW: inProjW, inProjB: inProjB, convW: cw, convB: cb, outProjW: ow, outProjB: ob, kernelSize: kernelSize)
        } else {
            let qW = try loadFirstTensor(ck, keys: ["\(p).self_attn.q_proj.weight", "\(p).attn.q_proj.weight"], want: [dim, dim])
            let qB = try loadOptionalTensor(ck, keys: ["\(p).self_attn.q_proj.bias", "\(p).attn.q_proj.bias"], want: [dim])
            let kW = try loadFirstTensor(ck, keys: ["\(p).self_attn.k_proj.weight", "\(p).attn.k_proj.weight"], want: [kvDim, dim])
            let kB = try loadOptionalTensor(ck, keys: ["\(p).self_attn.k_proj.bias", "\(p).attn.k_proj.bias"], want: [kvDim])
            let vW = try loadFirstTensor(ck, keys: ["\(p).self_attn.v_proj.weight", "\(p).attn.v_proj.weight"], want: [kvDim, dim])
            let vB = try loadOptionalTensor(ck, keys: ["\(p).self_attn.v_proj.bias", "\(p).attn.v_proj.bias"], want: [kvDim])
            let oW = try loadFirstTensor(ck, keys: ["\(p).self_attn.out_proj.weight", "\(p).attn.out_proj.weight", "\(p).attn.proj.weight"], want: [dim, dim])
            let oB = try loadOptionalTensor(ck, keys: ["\(p).self_attn.out_proj.bias", "\(p).attn.out_proj.bias", "\(p).attn.proj.bias"], want: [dim])
            let qNW = try loadOptionalTensor(ck, keys: ["\(p).self_attn.q_layernorm.weight"], want: [headDim])
            let kNW = try loadOptionalTensor(ck, keys: ["\(p).self_attn.k_layernorm.weight"], want: [headDim])
            self.kind = .attention(qW: qW, qB: qB, kW: kW, kB: kB, vW: vW, vB: vB, outW: oW, outB: oB, qNormW: qNW, kNormW: kNW)
        }
        if ck.has("\(p).feed_forward.w1.weight") {
            let w1 = try ck.f32("\(p).feed_forward.w1.weight", want: [mlpInner, dim])
            let w2 = try ck.f32("\(p).feed_forward.w2.weight", want: [dim, mlpInner])
            let w3 = try ck.f32("\(p).feed_forward.w3.weight", want: [mlpInner, dim])
            self.mlp = .swiglu(w1: w1, w2: w2, w3: w3)
        } else {
            let fc1W = try loadFirstTensor(ck, keys: ["\(p).mlp.fc1.weight", "\(p).mlp.linear_fc1.weight"], want: [mlpInner, dim])
            let fc1B = try loadOptionalTensor(ck, keys: ["\(p).mlp.fc1.bias", "\(p).mlp.linear_fc1.bias"], want: [mlpInner])
            let fc2W = try loadFirstTensor(ck, keys: ["\(p).mlp.fc2.weight", "\(p).mlp.linear_fc2.weight"], want: [dim, mlpInner])
            let fc2B = try loadOptionalTensor(ck, keys: ["\(p).mlp.fc2.bias", "\(p).mlp.linear_fc2.bias"], want: [dim])
            self.mlp = .gelu(fc1W: fc1W, fc1B: fc1B, fc2W: fc2W, fc2B: fc2B)
        }
    }

    var residentBytes: Int {
        var n = norm1G.count + (norm1B?.count ?? 0) + norm2G.count + (norm2B?.count ?? 0)
        switch mlp {
        case .gelu(let fc1W, let fc1B, let fc2W, let fc2B):
            n += fc1W.count + (fc1B?.count ?? 0) + fc2W.count + (fc2B?.count ?? 0)
        case .swiglu(let w1, let w2, let w3):
            n += w1.count + w2.count + w3.count
        }
        switch kind {
        case .conv(let inW, let inB, let cw, let cb, let ow, let ob, _):
            n += (inW?.count ?? 0) + (inB?.count ?? 0) + cw.count + (cb?.count ?? 0) + ow.count + (ob?.count ?? 0)
        case .attention(let qw, let qb, let kw, let kb, let vw, let vb, let ow, let ob, let qn, let kn):
            n += qw.count + (qb?.count ?? 0) + kw.count + (kb?.count ?? 0)
            n += vw.count + (vb?.count ?? 0) + ow.count + (ob?.count ?? 0)
            n += (qn?.count ?? 0) + (kn?.count ?? 0)
        }
        return n * MemoryLayout<Float>.stride
    }
}

/// Weights for `Lfm3VlVisionModel` hybrid conv + 2D Axial-RoPE GQA + downsample projector.
public final class QWLFM3VisionWeights: @unchecked Sendable {
    public let config: QWVisionConfig
    public let patchW: [Float]
    public let patchB: [Float]?
    public let embeddingNormW: [Float]?
    public let blocks: [QWLFM3VisionBlockW]
    public let projNormG: [Float]?
    public let projNormB: [Float]?
    public let projNormPreMerge: Bool
    public let fc1W: [Float], fc1B: [Float]?
    public let fc2W: [Float], fc2B: [Float]?
    public let projMid: Int
    public let outDim: Int

    public var residentBytes: Int {
        let top = patchW.count + (patchB?.count ?? 0) + (embeddingNormW?.count ?? 0)
            + (projNormG?.count ?? 0) + (projNormB?.count ?? 0)
            + fc1W.count + (fc1B?.count ?? 0) + fc2W.count + (fc2B?.count ?? 0)
        return top * MemoryLayout<Float>.stride + blocks.reduce(0) { $0 + $1.residentBytes }
    }

    public init(checkpoint ck: QWVisionCheckpoint) throws {
        let cfg = ck.config
        let d = cfg.dim
        let kvDim = cfg.kvHeads * cfg.headDim
        self.config = cfg
        let peKey = ["patchifier.weight", "patch_embed.proj.weight", "embeddings.patch_embedding.weight"]
            .first(where: { ck.shape($0) != nil }) ?? "patchifier.weight"
        let peWant = (ck.shape(peKey)?.count == 4) ? [d, 3, cfg.patch, cfg.patch] : [d, cfg.patchChannels]
        self.patchW = try ck.f32(peKey, want: peWant)
        self.patchB = try loadOptionalTensor(ck, keys: ["patchifier.bias", "patch_embed.proj.bias", "embeddings.patch_embedding.bias"], want: [d])
        self.embeddingNormW = try loadOptionalTensor(ck, keys: ["embedding_norm.weight"], want: [d])

        let pfx = (ck.has("blocks.0.norm1.weight") || ck.has("blocks.0.layer_norm1.weight") || ck.has("blocks.0.operator_norm.weight"))
            ? "blocks" : "layers"
        var depth = 0
        while ck.has("\(pfx).\(depth).norm1.weight")
                || ck.has("\(pfx).\(depth).layer_norm1.weight")
                || ck.has("\(pfx).\(depth).operator_norm.weight") { depth += 1 }
        guard depth > 0 else { throw QWVisionError.missingTensor("\(pfx).0.operator_norm.weight") }

        self.blocks = try (0..<depth).map { idx in
            let isConv = idx < cfg.visionLayerTypes.count
                ? (cfg.visionLayerTypes[idx] == "conv")
                : (ck.has("\(pfx).\(idx).conv.spatial_conv.weight") || ck.has("\(pfx).\(idx).conv2d.weight"))
            return try QWLFM3VisionBlockW(ck, "\(pfx).\(idx)", dim: d, kvDim: kvDim, headDim: cfg.headDim, mlpInner: cfg.mlpInner, isConv: isConv, kernelSize: cfg.convKernelSize)
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
        let fc1S = ck.shape("merger.linear_fc1.weight")
        let fc2S = ck.shape("merger.linear_fc2.weight")
        self.projMid = fc1S?.first ?? cfg.effectiveMergerIntermediateDim
        self.outDim = fc2S?.first ?? cfg.outDim
        self.fc1W = try ck.f32("merger.linear_fc1.weight", want: [projMid, merged])
        self.fc1B = try loadOptionalTensor(ck, keys: ["merger.linear_fc1.bias"], want: [projMid])
        self.fc2W = try ck.f32("merger.linear_fc2.weight", want: [outDim, projMid])
        self.fc2B = try loadOptionalTensor(ck, keys: ["merger.linear_fc2.bias"], want: [outDim])
        ck.releaseBacking()
    }
}

/// CPU FP32 reference evaluator for `Lfm3VlVisionModel`.
public final class QWLFM3VisionRef: @unchecked Sendable {
    public let w: QWLFM3VisionWeights
    public init(weights: QWLFM3VisionWeights) { self.w = weights }

    public func encode(patches: [Float], plan: QWVisionGridPlan, posIds: [Int32]) -> [Float] {
        let c = w.config
        let d = c.dim
        let nPatch = plan.nPatch
        let nToken = plan.nToken
        guard nPatch > 0, nToken > 0 else { return [] }

        var s = [Float](repeating: 0, count: nPatch * d)
        runLinearProj(out: &s, x: patches, rows: nPatch, w: w.patchW, bias: w.patchB, nOut: d, nIn: c.patchChannels)
        var normed = [Float](repeating: 0, count: nPatch * d)
        if let embNW = w.embeddingNormW {
            qwRMSNorm(s, into: &normed, gamma: embNW, rows: nPatch, dim: d, eps: c.eps)
            s = normed
        }

        let rope = build2DAxialRope(posIds: posIds, nPatch: nPatch, headDim: c.headDim, ropeTheta: c.ropeTheta)
        let kvDim = c.kvHeads * c.headDim
        var branch = [Float](repeating: 0, count: nPatch * d)
        var bcx = [Float](repeating: 0, count: nPatch * 3 * d)
        var convGated = [Float](repeating: 0, count: nPatch * d)
        var qBuf = [Float](repeating: 0, count: nPatch * d)
        var qNormed = [Float](repeating: 0, count: nPatch * d)
        var kBuf = [Float](repeating: 0, count: nPatch * kvDim)
        var kNormed = [Float](repeating: 0, count: nPatch * kvDim)
        var vBuf = [Float](repeating: 0, count: nPatch * kvDim)
        var attnOut = [Float](repeating: 0, count: nPatch * d)
        var mlpH = [Float](repeating: 0, count: nPatch * c.mlpInner)
        var mlpUp = [Float](repeating: 0, count: nPatch * c.mlpInner)

        for blk in w.blocks {
            if let b1 = blk.norm1B {
                qwLayerNorm(s, into: &normed, gamma: blk.norm1G, beta: b1, rows: nPatch, dim: d, eps: c.eps)
            } else {
                qwRMSNorm(s, into: &normed, gamma: blk.norm1G, rows: nPatch, dim: d, eps: c.eps)
            }
            switch blk.kind {
            case .conv(let inProjW, let inProjB, let convW, let convB, let outProjW, let outProjB, let kSize):
                if let inW = inProjW {
                    runLinearProj(out: &bcx, x: normed, rows: nPatch, w: inW, bias: inProjB, nOut: 3 * d, nIn: d)
                    for row in 0..<nPatch {
                        let base = row * 3 * d
                        let dst = row * d
                        for ch in 0..<d {
                            convGated[dst + ch] = bcx[base + ch] * bcx[base + 2 * d + ch]
                        }
                    }
                    runVisionConv2d(
                        x: convGated, convW: convW, convB: convB, out: &attnOut,
                        plan: plan, posIds: posIds, dim: d, kSize: kSize
                    )
                    for row in 0..<nPatch {
                        let cBase = row * 3 * d + d
                        let dst = row * d
                        for ch in 0..<d {
                            attnOut[dst + ch] *= bcx[cBase + ch]
                        }
                    }
                } else {
                    runVisionConv2d(
                        x: normed, convW: convW, convB: convB, out: &attnOut,
                        plan: plan, posIds: posIds, dim: d, kSize: kSize
                    )
                }
                runLinearProj(out: &branch, x: attnOut, rows: nPatch, w: outProjW, bias: outProjB, nOut: d, nIn: d)
            case .attention(let qW, let qB, let kW, let kB, let vW, let vB, let oW, let oB, let qNW, let kNW):
                runLinearProj(out: &qBuf, x: normed, rows: nPatch, w: qW, bias: qB, nOut: d, nIn: d)
                runLinearProj(out: &kBuf, x: normed, rows: nPatch, w: kW, bias: kB, nOut: kvDim, nIn: d)
                runLinearProj(out: &vBuf, x: normed, rows: nPatch, w: vW, bias: vB, nOut: kvDim, nIn: d)
                if let qNW {
                    qwRMSNorm(qBuf, into: &qNormed, gamma: qNW, rows: nPatch * c.heads, dim: c.headDim, eps: c.eps)
                }
                if let kNW {
                    qwRMSNorm(kBuf, into: &kNormed, gamma: kNW, rows: nPatch * c.kvHeads, dim: c.headDim, eps: c.eps)
                }
                runBidirectionalGQAAttn(
                    qBuf: qNW != nil ? qNormed : qBuf,
                    kBuf: kNW != nil ? kNormed : kBuf,
                    vBuf: vBuf, out: &attnOut,
                    nSeq: nPatch, heads: c.heads, kvHeads: c.kvHeads, headDim: c.headDim, ropeCosSin: rope
                )
                runLinearProj(out: &branch, x: attnOut, rows: nPatch, w: oW, bias: oB, nOut: d, nIn: d)
            }
            addRowInto(&s, src: branch)

            if let b2 = blk.norm2B {
                qwLayerNorm(s, into: &normed, gamma: blk.norm2G, beta: b2, rows: nPatch, dim: d, eps: c.eps)
            } else {
                qwRMSNorm(s, into: &normed, gamma: blk.norm2G, rows: nPatch, dim: d, eps: c.eps)
            }
            switch blk.mlp {
            case .gelu(let fc1W, let fc1B, let fc2W, let fc2B):
                runLinearProj(out: &mlpH, x: normed, rows: nPatch, w: fc1W, bias: fc1B, nOut: c.mlpInner, nIn: d, act: qwGeluErf)
                runLinearProj(out: &branch, x: mlpH, rows: nPatch, w: fc2W, bias: fc2B, nOut: d, nIn: c.mlpInner)
            case .swiglu(let w1, let w2, let w3):
                runLinearProj(out: &mlpH, x: normed, rows: nPatch, w: w1, bias: nil, nOut: c.mlpInner, nIn: d)
                runLinearProj(out: &mlpUp, x: normed, rows: nPatch, w: w3, bias: nil, nOut: c.mlpInner, nIn: d)
                let count = nPatch * c.mlpInner
                for i in 0..<count {
                    let g = mlpH[i]
                    mlpH[i] = (g / (1.0 + expf(-g))) * mlpUp[i]
                }
                runLinearProj(out: &branch, x: mlpH, rows: nPatch, w: w2, bias: nil, nOut: d, nIn: c.mlpInner)
            }
            addRowInto(&s, src: branch)
        }

        if let ng = w.projNormG, let nb = w.projNormB, w.projNormPreMerge {
            qwLayerNorm(s, into: &normed, gamma: ng, beta: nb, rows: nPatch, dim: d, eps: c.projectorNormEps)
            s = normed
        }
        if let ng = w.projNormG, let nb = w.projNormB, !w.projNormPreMerge {
            var mergedNormed = [Float](repeating: 0, count: nToken * c.merged)
            qwLayerNorm(s, into: &mergedNormed, gamma: ng, beta: nb, rows: nToken, dim: c.merged, eps: c.projectorNormEps)
            s = mergedNormed
        }
        var mid = [Float](repeating: 0, count: nToken * w.projMid)
        runLinearProj(out: &mid, x: s, rows: nToken, w: w.fc1W, bias: w.fc1B, nOut: w.projMid, nIn: c.merged, act: qwGeluErf)
        var out = [Float](repeating: 0, count: nToken * w.outDim)
        runLinearProj(out: &out, x: mid, rows: nToken, w: w.fc2W, bias: w.fc2B, nOut: w.outDim, nIn: w.projMid)
        return out
    }

    private func runVisionConv2d(
        x: [Float],
        convW: [Float],
        convB: [Float]?,
        out: inout [Float],
        plan: QWVisionGridPlan,
        posIds: [Int32],
        dim: Int,
        kSize: Int
    ) {
        let tCount = max(1, plan.thw.t)
        let gh = plan.thw.h
        let gw = plan.thw.w
        let frameLen = gh * gw
        let pad = kSize / 2
        var rasterIn = [Float](repeating: 0, count: frameLen * dim)
        var rasterOut = [Float](repeating: 0, count: frameLen * dim)

        for frame in 0..<tCount {
            let pBase = frame * frameLen
            for i in 0..<frameLen {
                let p = pBase + i
                let ry = Int(posIds[2 * p])
                let rx = Int(posIds[2 * p + 1])
                let dst = (ry * gw + rx) * dim
                let src = p * dim
                for d in 0..<dim { rasterIn[dst + d] = x[src + d] }
            }
            for ry in 0..<gh {
                for rx in 0..<gw {
                    let dst = (ry * gw + rx) * dim
                    for d in 0..<dim {
                        var acc: Float = convB?[d] ?? 0
                        let wBase = d * kSize * kSize
                        for ky in 0..<kSize {
                            let sy = ry + ky - pad
                            if sy < 0 || sy >= gh { continue }
                            for kx in 0..<kSize {
                                let sx = rx + kx - pad
                                if sx < 0 || sx >= gw { continue }
                                acc += rasterIn[(sy * gw + sx) * dim + d] * convW[wBase + ky * kSize + kx]
                            }
                        }
                        rasterOut[dst + d] = acc
                    }
                }
            }
            for i in 0..<frameLen {
                let p = pBase + i
                let ry = Int(posIds[2 * p])
                let rx = Int(posIds[2 * p + 1])
                let src = (ry * gw + rx) * dim
                let dst = p * dim
                for d in 0..<dim { out[dst + d] = rasterOut[src + d] }
            }
        }
    }
}
