// DriverWeightLoading+GigaChat4.swift — Raw BF16/F16 low-rank GatedNorm loaders
// for trunk sandwich norms, MLA latent norms, and NextN MTP norms.
import Foundation
import Metal
import QWAlloc
import QWAttn
import QWCore
import QWHAL
import QWLoad
import QWNorm
import QWQuant

extension QWEngineDriver.WeightLoader {
    private static func f32ToBF16Bits(_ value: Float) -> UInt16 {
        let u = value.bitPattern
        if (u & 0x7FFF_FFFF) > 0x7F80_0000 {
            return UInt16((u >> 16) | 0x0040)
        }
        let lsb = (u >> 16) & 1
        return UInt16(truncatingIfNeeded: (u &+ 0x7FFF &+ lsb) >> 16)
    }

    mutating func rawGatedNormTensor(_ name: String, _ elems: Int) throws -> QWTensorValue {
        let resolved = resolveName(name)
        if let pk, let io {
            let e = try pk.entry(resolved)
            guard PlaneLayout.resolve(e) == .raw, e.shape.reduce(1, *) == elems else {
                throw QWEngineDriverError.weightMissing(name, "expected raw norm plane [\(elems)]")
            }
            if e.kind == nil || e.kind == "bf16" {
                let t = try allocateWeight(QWTensorShape([elems], .bf16))
                let dstBuf = t.region.buffer
                let dstPtr = dstBuf.contents().advanced(by: t.region.base + t.offset)
                try withExtendedLifetime(dstBuf) {
                    try io.load(resolved, layout: .raw, into: dstPtr)
                }
                return t
            }
            let floats = try pk.rawBf16(resolved)
            guard floats.count == elems else {
                throw QWEngineDriverError.weightMissing(name, "shape \(floats.count) ≠ \(elems)")
            }
            let t = try allocateWeight(QWTensorShape([elems], .bf16))
            let dstPtr = t.region.buffer.contents()
                .advanced(by: t.region.base + t.offset)
                .assumingMemoryBound(to: UInt16.self)
            for idx in 0..<elems {
                dstPtr[idx] = Self.f32ToBF16Bits(floats[idx])
            }
            return t
        }
        guard let st, let info = st.tensor(resolved) else {
            throw QWEngineDriverError.weightMissing(name, "missing gated norm tensor")
        }
        let total = info.elementCount ?? info.shape.reduce(1, *)
        guard total == elems else {
            throw QWEngineDriverError.weightMissing(name, "shape \(total) ≠ \(elems)")
        }
        if info.dtype == .bf16 || info.dtype == .f16 {
            let outDtype: QWDType = info.dtype == .f16 ? .f16 : .bf16
            if outDtype == .bf16 {
                let t = try allocateWeight(QWTensorShape([elems], .bf16))
                let reader = try st.reader(of: resolved)
                let dstBuf = t.region.buffer
                let dstPtr = dstBuf.contents().advanced(by: t.region.base + t.offset)
                try withExtendedLifetime(dstBuf) {
                    try reader.withBytes { raw in
                        guard raw.count >= elems * 2, let base = raw.baseAddress else {
                            throw QWEngineDriverError.weightMissing(name, "bf16 norm byte length mismatch")
                        }
                        dstPtr.copyMemory(from: base, byteCount: elems * 2)
                    }
                }
                return t
            }
        }
        let floats = try st.readFloats(resolved, expectedElementCount: elems)
        let t = try allocateWeight(QWTensorShape([elems], .bf16))
        let dstPtr = t.region.buffer.contents()
            .advanced(by: t.region.base + t.offset)
            .assumingMemoryBound(to: UInt16.self)
        for idx in 0..<elems {
            dstPtr[idx] = Self.f32ToBF16Bits(floats[idx])
        }
        return t
    }

    mutating func loadGatedNorm(
        _ prefix: String,
        dim: Int,
        config: QWModelDescriptor.GatedNormConfig
    ) throws -> QWGatedNormWeights {
        let pfx = prefix.hasSuffix(".") ? prefix : (prefix + ".")
        let w = try rawGatedNormTensor(pfx + "weight", dim)
        let gu = try rawGatedNormTensor(pfx + "gate_up_projection.weight", config.rank * dim)
        let gd = try rawGatedNormTensor(pfx + "gate_down_projection.weight", dim * config.rank)
        return QWGatedNormWeights(
            weight: w,
            gateUp: gu,
            gateDown: gd,
            rank: config.rank,
            gatingWeight: config.gatingWeight,
            zeroCentered: geometry.normGammaIsPlusOne && !isMLXCheckpoint
        )
    }

    mutating func loadGatedLatentNorm(
        _ prefix: String,
        dim: Int,
        config: QWModelDescriptor.GatedNormConfig
    ) throws -> QWMLAAttnOperator.GatedLatentNorm {
        let gn = try loadGatedNorm(prefix, dim: dim, config: config)
        return QWMLAAttnOperator.GatedLatentNorm(
            weight: gn.weight,
            gateUp: gn.gateUp,
            gateDown: gn.gateDown,
            rank: gn.rank,
            gatingWeight: gn.gatingWeight,
            zeroCentered: gn.zeroCentered
        )
    }

    mutating func loadLayerNorms(
        _ blk: String,
        geometry: QWModelGeometry
    ) throws -> (ln1: QWTensorValue, ln2: QWTensorValue, gated: QWEngineDriver.GatedLayerNorms?) {
        let h = geometry.hidden
        guard let gn = geometry.gatedNormConfig else {
            let ln1 = try f16wNorm(blk + "input_layernorm.weight", h)
            let ln2 = try f16wNorm(blk + "post_attention_layernorm.weight", h)
            return (ln1, ln2, nil)
        }
        let input = try loadGatedNorm(blk + "input_layernorm.", dim: h, config: gn)
        let postSelfAttn = geometry.useSandwichNorm
            ? try loadGatedNorm(blk + "post_self_attn_layernorm.", dim: h, config: gn)
            : input
        let postAttn = try loadGatedNorm(blk + "post_attention_layernorm.", dim: h, config: gn)
        let postFFN = geometry.useSandwichNorm
            ? try loadGatedNorm(blk + "post_feedforward_layernorm.", dim: h, config: gn)
            : postAttn
        let bundle = QWEngineDriver.GatedLayerNorms(
            input: input,
            postSelfAttn: postSelfAttn,
            postAttn: postAttn,
            postFFN: postFFN
        )
        return (input.weight, postAttn.weight, bundle)
    }

    mutating func loadFinalNormBundle(
        geometry: QWModelGeometry
    ) throws -> (norm: QWTensorValue, gated: QWGatedNormWeights?) {
        let h = geometry.hidden
        guard let gn = geometry.gatedNormConfig else {
            return (try finalNorm(h), nil)
        }
        let fg = try loadGatedNorm(QWEngineDriver.P + "norm.", dim: h, config: gn)
        return (fg.weight, fg)
    }
}
