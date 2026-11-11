// DriverWeightLoading+GLM5.swift — NoPE MLA + Key-Pool Indexer and KDA linear
// attention weight loaders for QWEngineDriver.WeightLoader.
import Foundation
import Metal
import QWAlloc
import QWAttn
import QWCore
import QWGEMM
import QWHAL
import QWMLP
import QWQuant

extension QWEngineDriver.WeightLoader {

    /// Orthonormal 32-point Fast Walsh-Hadamard Transform in place over each
    /// contiguous 32-element slice of `values`.
    private static func fwht32OrthoInPlace(_ values: inout [Float]) {
        let scale: Float = 1.0 / sqrt(32.0)
        var base = 0
        while base + 32 <= values.count {
            var h = 1
            while h < 32 {
                var i = 0
                while i < 32 {
                    for j in i..<(i + h) {
                        let a = values[base + j]
                        let b = values[base + j + h]
                        values[base + j] = a + b
                        values[base + j + h] = a - b
                    }
                    i += h * 2
                }
                h *= 2
            }
            for k in 0..<32 {
                values[base + k] *= scale
            }
            base += 32
        }
    }

    mutating func loadMLAAttnLayer(
        blk: String, layer: Int,
        ln1: QWTensorValue, ln2: QWTensorValue,
        gu: QWMLPWeight, down: QWWeightView,
        moe: QWEngineDriver.MoEW?,
        geometry: QWModelGeometry
    ) throws -> QWEngineDriver.LayerW {
        let qRank = geometry.qLoraRank
        let kvRank = geometry.kvLoraRank
        let qTotal = geometry.qHeads * geometry.headDim
        let vTotal = geometry.qHeads * geometry.vHeadDim
        let idxH = geometry.indexerHeads
        let idxD = geometry.indexerHeadDim
        let h = geometry.hidden

        let (qkvProj, qkvSub) = try concatQ4([
            (blk + "self_attn.q_a_proj.weight", qRank),
            (blk + "self_attn.kv_a_proj_with_mqa.weight", kvRank),
            (blk + "self_attn.indexer.wk.weight", idxD)
        ], k: h)
        let qANormW = try f16w(blk + "self_attn.q_a_layernorm.weight", qRank)
        let kvANormW = try f16w(blk + "self_attn.kv_a_layernorm.weight", kvRank)
        let kNormW = try f16w(blk + "self_attn.indexer.k_norm.weight", idxD)
        let kNormB = try f16w(blk + "self_attn.indexer.k_norm.bias", idxD)

        let gateFloats = try rawFloats(blk + "self_attn.indexer.index_kpool_compress_gate", idxD * h)
        var wProjFloats = try rawFloats(blk + "self_attn.indexer.weights_proj.weight", idxH * h)
        let headScale: Float = 1.0 / sqrt(Float(max(1, idxH)))
        for i in 0..<wProjFloats.count {
            wProjFloats[i] *= headScale
        }
        var combined = gateFloats + wProjFloats
        if hl.weightBasis.rotatesActivations {
            Self.fwht32OrthoInPlace(&combined)
        }
        let wGateWeights = try allocateWeight(QWTensorShape([(idxD + idxH) * h], .f16))
        var bits = [UInt16](repeating: 0, count: combined.count)
        for i in 0..<combined.count {
            bits[i] = QWQuant.f32tof16Bits(combined[i])
        }
        bits.withUnsafeBytes {
            wGateWeights.region.buffer.contents()
                .advanced(by: wGateWeights.region.base + wGateWeights.offset)
                .copyMemory(from: $0.baseAddress!, byteCount: combined.count * 2)
        }

        let ape = try f16w(blk + "self_attn.indexer.index_kpool_compress_ape", 4 * idxD)
        let (qBProj, _) = try concatQ4([
            (blk + "self_attn.q_b_proj.weight", qTotal),
            (blk + "self_attn.indexer.wq_b.weight", idxH * idxD)
        ], k: qRank)
        let kvBProj = try q4(blk + "self_attn.kv_b_proj.weight", n: qTotal + vTotal, k: kvRank)
        let o = try q4(blk + "self_attn.o_proj.weight", n: h, k: vTotal)

        let mlaW = QWMLAAttnOperator.LayerWeights(
            qANormW: qANormW,
            kvANormW: kvANormW,
            kNormW: kNormW,
            kNormB: kNormB,
            wGateWeights: wGateWeights,
            ape: ape,
            qBProj: qBProj,
            kvBProj: kvBProj
        )
        let mhcW: QWEngineDriver.MHCLayerWeights?
        if let mhc = geometry.mhcConfig, layer >= mhc.start, layer < mhc.end {
            mhcW = try loadMHC(blk, mhc: mhc)
        } else {
            mhcW = nil
        }
        return QWEngineDriver.LayerW(
            ln1: ln1, ln2: ln2, gu: gu, down: down,
            kind: moe == nil ? .fullAttention : .moe,
            gdn: nil,
            attn: QWEngineDriver.AttnW(
                qkvProj: qkvProj, q: qkvSub[0], k: qkvSub[1], v: qkvSub[2],
                o: o, wq: nil, wkTiled: nil, mla: mlaW),
            moe: moe, mhc: mhcW
        )
    }

    mutating func loadKDALayer(
        blk: String, layer: Int,
        ln1: QWTensorValue, ln2: QWTensorValue,
        gu: QWMLPWeight, down: QWWeightView,
        moe: QWEngineDriver.MoEW?,
        geometry: QWModelGeometry
    ) throws -> QWEngineDriver.LayerW {
        let qkvC = geometry.gdnQKVChannels
        let zC = geometry.gdnZChannels
        let qkCh = geometry.gdnQKHeads * geometry.gdnHeadDim
        let lr = geometry.gdnLowRankDim
        let vh = geometry.gdnVHeads
        let h = geometry.hidden
        let kConv = geometry.gdnConvKernel

        let (inProj, qkvSub) = try concatQ4([
            (blk + "self_attn.q_proj.weight", qkCh),
            (blk + "self_attn.k_proj.weight", qkCh),
            (blk + "self_attn.v_proj.weight", zC)
        ], k: h)
        let (fgbProj, fgbSub) = try concatQ4([
            (blk + "self_attn.f_a_proj.weight", lr),
            (blk + "self_attn.g_a_proj.weight", lr),
            (blk + "self_attn.b_proj.weight", vh)
        ], k: h)
        let fB = try f16w(blk + "self_attn.f_b_proj.weight", zC * lr)
        let gB = try f16w(blk + "self_attn.g_b_proj.weight", zC * lr)

        let conv: QWTensorValue
        if hasTensor(blk + "self_attn.q_conv1d.weight") {
            let qC = try rawFloats(blk + "self_attn.q_conv1d.weight", qkCh * kConv)
            let kC = try rawFloats(blk + "self_attn.k_conv1d.weight", qkCh * kConv)
            let vC = try rawFloats(blk + "self_attn.v_conv1d.weight", zC * kConv)
            let allConv = qC + kC + vC
            let t = try allocateWeight(QWTensorShape([qkvC * kConv], .f16))
            var bits = [UInt16](repeating: 0, count: allConv.count)
            for i in 0..<allConv.count {
                bits[i] = QWQuant.f32tof16Bits(allConv[i])
            }
            bits.withUnsafeBytes {
                t.region.buffer.contents()
                    .advanced(by: t.region.base + t.offset)
                    .copyMemory(from: $0.baseAddress!, byteCount: allConv.count * 2)
            }
            conv = t
        } else {
            conv = try f16w(blk + "self_attn.conv1d.weight", qkvC * kConv)
        }

        let gnw = try f16w(blk + "self_attn.o_norm.weight", geometry.gdnHeadDim)
        let aLog = try f32w(blk + "self_attn.A_log", vh)
        let dtB = try f32w(blk + "self_attn.dt_bias", zC)
        let out = try q4(blk + "self_attn.o_proj.weight", n: h, k: zC)

        let mhcW: QWEngineDriver.MHCLayerWeights?
        if let mhc = geometry.mhcConfig, layer >= mhc.start, layer < mhc.end {
            mhcW = try loadMHC(blk, mhc: mhc)
        } else {
            mhcW = nil
        }
        return QWEngineDriver.LayerW(
            ln1: ln1, ln2: ln2, gu: gu, down: down,
            kind: moe == nil ? .gdn : .gdnMoe,
            gdn: QWEngineDriver.GdnW(
                inProj: inProj, qkv: qkvSub[0], z: qkvSub[2], a: fgbSub[0], b: fgbSub[2],
                conv: conv, normW: gnw, aLog: aLog, dtBias: dtB, out: out,
                fgbProj: fgbProj, fB: fB, gB: gB),
            attn: nil, moe: moe, mhc: mhcW
        )
    }

    /// Ephemeral NextN MTP layer weights (`model.layers.<numLayers>.*`) loaded on demand from SSD.
    struct NextNMTPLayerWeights {
        let eNorm: QWTensorValue
        let hNorm: QWTensorValue
        let ehProj: QWWeightView
        let sharedHeadNorm: QWTensorValue
        let layer: QWEngineDriver.LayerW
    }

    mutating func loadNextNMTPLayer(geometry: QWModelGeometry) throws -> NextNMTPLayerWeights? {
        guard geometry.mtpLayers > 0 else { return nil }
        let mtpIdx = geometry.numLayers
        let blk = QWEngineDriver.P + "layers.\(mtpIdx)."
        guard hasTensor(blk + "enorm.weight"), hasTensor(blk + "eh_proj.weight") else {
            return nil
        }
        let h = geometry.hidden
        let eNorm = try f16wNorm(blk + "enorm.weight", h)
        let hNorm = try f16wNorm(blk + "hnorm.weight", h)
        let ehProj = try q4(blk + "eh_proj.weight", n: h, k: 2 * h)
        let ln1 = try f16wNorm(blk + "input_layernorm.weight", h)
        let attnOnly = try loadMLAAttnWeights(blk: blk, geometry: geometry)
        let ln2 = try f16wNorm(blk + "post_attention_layernorm.weight", h)
        let moe = try packedMoE(blk, experts: geometry.numExperts, topK: geometry.expertsPerTok)
        let guView = try firstExpert(moe.gateUp, rows: 2 * geometry.moeIntermediate)
        guard let gu = QWMLPWeight(blocks: guView.blocks, n: 2 * geometry.moeIntermediate, k: h,
                                   bits: guView.bits, groupSize: guView.groupSize,
                                   isRotated: guView.isRotated,
                                   isIncumbentContainer: guView.isIncumbentContainer,
                                   isSHVQ: guView.isSHVQ) else {
            throw QWEngineDriverError.weightMissing(blk + "mlp.experts", "mtp gu view invalid")
        }
        let down = try firstExpert(moe.down, rows: h)
        let sharedHeadNorm = try f16wNorm(blk + "shared_head.norm.weight", h)
        let layer = QWEngineDriver.LayerW(
            ln1: ln1, ln2: ln2, gu: gu, down: down,
            kind: .moe, gdn: nil, attn: attnOnly, moe: moe, mhc: nil
        )
        return NextNMTPLayerWeights(
            eNorm: eNorm, hNorm: hNorm, ehProj: ehProj,
            sharedHeadNorm: sharedHeadNorm, layer: layer
        )
    }

    mutating func loadMLAAttnWeights(
        blk: String,
        geometry: QWModelGeometry
    ) throws -> QWEngineDriver.AttnW {
        let qRank = geometry.qLoraRank
        let kvRank = geometry.kvLoraRank
        let qTotal = geometry.qHeads * geometry.headDim
        let vTotal = geometry.qHeads * geometry.vHeadDim
        let idxH = geometry.indexerHeads
        let idxD = geometry.indexerHeadDim
        let h = geometry.hidden

        let (qkvProj, qkvSub) = try concatQ4([
            (blk + "self_attn.q_a_proj.weight", qRank),
            (blk + "self_attn.kv_a_proj_with_mqa.weight", kvRank),
            (blk + "self_attn.indexer.wk.weight", idxD)
        ], k: h)
        let qANormW = try f16w(blk + "self_attn.q_a_layernorm.weight", qRank)
        let kvANormW = try f16w(blk + "self_attn.kv_a_layernorm.weight", kvRank)
        let kNormW = try f16w(blk + "self_attn.indexer.k_norm.weight", idxD)
        let kNormB = try f16w(blk + "self_attn.indexer.k_norm.bias", idxD)

        let gateFloats = try rawFloats(blk + "self_attn.indexer.index_kpool_compress_gate", idxD * h)
        var wProjFloats = try rawFloats(blk + "self_attn.indexer.weights_proj.weight", idxH * h)
        let headScale: Float = 1.0 / sqrt(Float(max(1, idxH)))
        for i in 0..<wProjFloats.count {
            wProjFloats[i] *= headScale
        }
        var combined = gateFloats + wProjFloats
        if hl.weightBasis.rotatesActivations {
            Self.fwht32OrthoInPlace(&combined)
        }
        let wGateWeights = try allocateWeight(QWTensorShape([(idxD + idxH) * h], .f16))
        var bits = [UInt16](repeating: 0, count: combined.count)
        for i in 0..<combined.count {
            bits[i] = QWQuant.f32tof16Bits(combined[i])
        }
        bits.withUnsafeBytes {
            wGateWeights.region.buffer.contents()
                .advanced(by: wGateWeights.region.base + wGateWeights.offset)
                .copyMemory(from: $0.baseAddress!, byteCount: combined.count * 2)
        }

        let ape = try f16w(blk + "self_attn.indexer.index_kpool_compress_ape", 4 * idxD)
        let (qBProj, _) = try concatQ4([
            (blk + "self_attn.q_b_proj.weight", qTotal),
            (blk + "self_attn.indexer.wq_b.weight", idxH * idxD)
        ], k: qRank)
        let kvBProj = try q4(blk + "self_attn.kv_b_proj.weight", n: qTotal + vTotal, k: kvRank)
        let o = try q4(blk + "self_attn.o_proj.weight", n: h, k: vTotal)

        let mlaW = QWMLAAttnOperator.LayerWeights(
            qANormW: qANormW,
            kvANormW: kvANormW,
            kNormW: kNormW,
            kNormB: kNormB,
            wGateWeights: wGateWeights,
            ape: ape,
            qBProj: qBProj,
            kvBProj: kvBProj
        )
        return QWEngineDriver.AttnW(
            qkvProj: qkvProj, q: qkvSub[0], k: qkvSub[1], v: qkvSub[2],
            o: o, wq: nil, wkTiled: nil, mla: mlaW)
    }

    mutating func ensureSharedColdBanks(coldE: Int, i: Int, h: Int) throws -> (gu: QWWeightView, down: QWWeightView) {
        if let existing = sharedColdBanks { return existing }
        let guElems = (coldE * 2 * i * format.expertRowBytes(k: h)) / 2
        let downElems = (coldE * h * format.expertDownRowBytes(k: i)) / 2
        let guBlocks = try allocateWeight(QWTensorShape([guElems], .f16))
        let downBlocks = try allocateWeight(QWTensorShape([downElems], .f16))
        memset(guBlocks.region.buffer.contents().advanced(by: guBlocks.region.base + guBlocks.offset), 0, guElems * 2)
        memset(downBlocks.region.buffer.contents().advanced(by: downBlocks.region.base + downBlocks.offset), 0, downElems * 2)
        guard let guView = QWWeightView(blocks: guBlocks, n: coldE * 2 * i, k: h,
                                        bits: format.expertBits, groupSize: format.expertGroupSize,
                                        isRotated: hl.weightBasis.rotatesActivations,
                                        isIncumbentContainer: format.expertIsIncumbentContainer,
                                        isSHVQ: format.expertIsSHVQ),
              let downView = QWWeightView(blocks: downBlocks, n: coldE * h, k: i,
                                          bits: format.expertDownBits, groupSize: format.expertDownGroupSize,
                                          isRotated: hl.weightBasis.rotatesActivations,
                                          isIncumbentContainer: format.expertDownIsIncumbentContainer,
                                          isSHVQ: format.expertDownIsSHVQ) else {
            throw QWEngineDriverError.weightMissing("sharedColdBanks", "cold expert staging bank view invalid")
        }
        let pair = (gu: guView, down: downView)
        sharedColdBanks = pair
        return pair
    }

    func firstExpert(_ bank: QWWeightView, rows: Int) throws -> QWWeightView {
        let rowBytes = bank.rowBytes
        let sub = QWTensorValue(region: bank.blocks.region,
                                offset: bank.blocks.offset,
                                shape: QWTensorShape([rows * rowBytes / 2], .f16))
        guard let v = bank.subview(blocks: sub, n: rows, k: bank.k) else {
            throw QWEngineDriverError.state("the first expert's view is invalid")
        }
        return v
    }

    mutating func loadDecoderWeights(geometry: QWModelGeometry) throws -> (layers: [ QWEngineDriver.LayerW], finalNorm: QWTensorValue) {
        let layers = try loadLayers(geometry: geometry)
        let norm = try finalNorm(geometry.hidden)
        st?.adviseDontNeed()
        return (layers, norm)
    }

    mutating func loadMHCBranch(_ blk: String, prefix: String, mhc: QWModelDescriptor.MHCConfig) throws -> (w: QWTensorValue, b: QWTensorValue) {
        let c = mhc.streams
        let routerRows = 2 * c + (mhc.fixedIdentity ? 0 : c * c)
        let width = c * geometry.hidden
        let wElems = routerRows * width
        if hasTensor(blk + prefix + ".weight") {
            return (
                try f32wOrDequant(blk + prefix + ".weight", wElems),
                try f32wOrDequant(blk + prefix + ".bias", routerRows)
            )
        }
        let hcPrefix = prefix.replacingOccurrences(of: "mhc_", with: "hc_")
        var fnFloats = try rawFloats(blk + hcPrefix + "_fn", wElems)
        if hasTensor(blk + hcPrefix + "_scale") {
            let sc = try rawFloats(blk + hcPrefix + "_scale", 3)
            for r in 0..<routerRows {
                let s: Float = (r < c) ? sc[0] : ((r < 2 * c) ? sc[1] : sc[2])
                let rowStart = r * width
                for k in 0..<width { fnFloats[rowStart + k] *= s }
            }
        }
        let wTensor = try allocateWeight(QWTensorShape([wElems], .f32))
        fnFloats.withUnsafeBytes {
            wTensor.region.buffer.contents()
                .advanced(by: wTensor.region.base + wTensor.offset)
                .copyMemory(from: $0.baseAddress!, byteCount: wElems * 4)
        }
        let bTensor = try f32wOrDequant(blk + hcPrefix + "_base", routerRows)
        return (wTensor, bTensor)
    }

    mutating func loadMHC(_ blk: String, mhc: QWModelDescriptor.MHCConfig) throws -> QWEngineDriver.MHCLayerWeights {
        let attn = try loadMHCBranch(blk, prefix: "mhc_attn", mhc: mhc)
        let ffn = try loadMHCBranch(blk, prefix: "mhc_ffn", mhc: mhc)
        return QWEngineDriver.MHCLayerWeights(attnW: attn.w, attnB: attn.b, ffnW: ffn.w, ffnB: ffn.b)
    }
}
