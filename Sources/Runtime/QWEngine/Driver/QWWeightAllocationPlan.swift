import QWAlloc
import QWCore

public enum QWWeightArenaSizing: String, Sendable {
    case planned
    case legacyFixed = "legacy-fixed"
}

/// Mirrors loader allocation order, not pack storage: concatenations share
/// subviews, embeddings stay mapped, and RoPE occupies the weight region.
struct QWWeightAllocationPlan {
    let shapes: [QWTensorShape]
    let bytes: Int

    init(geometry: QWModelGeometry, ropeRows: Int, mtpDrafts: Int = 0, headQ4g: Bool = false,
         maxMTPDrafts: Int = 5, format: QWWeightFormat = .hq4G32) throws {
        guard ropeRows > 0, ropeRows <= Int.max / 256 - 1 else {
            throw QWEngineDriverError.state("invalid RoPE row count: \(ropeRows)")
        }
        guard mtpDrafts >= 0 && mtpDrafts <= maxMTPDrafts else {
            throw QWEngineDriverError.state("invalid MTP drafts: \(mtpDrafts) (supported: 0...\(maxMTPDrafts))")
        }
        var requests: [QWTensorShape] = []
        func tensor(_ elements: Int, _ dtype: QWDType) {
            requests.append(QWTensorShape([elements], dtype))
        }
        // The bank's size is the CONTAINER's size: the incumbent's twenty
        // bytes per group of thirty-two, or the family's own native bytes.
        func q4(_ n: Int, _ k: Int) {
            tensor(n * (format.rowBytes(k: k) / 2), .f16)
        }
        func q4Expert(_ n: Int, _ k: Int) {
            tensor(n * (format.expertRowBytes(k: k) / 2), .f16)
        }
        func q4ExpertDown(_ n: Int, _ k: Int) {
            tensor(n * (format.expertDownRowBytes(k: k) / 2), .f16)
        }
        let h = geometry.hidden, i = geometry.intermediate
        let kv = geometry.kvHeads * geometry.headDim
        if headQ4g {
            q4(geometry.vocab, h)
        } else {
            tensor(geometry.vocab * h, .f16)
        }
        var allocatedSharedCold = false
        for layer in 0..<geometry.numLayers {
            tensor(h, .f16)
            tensor(h, .f16)
            if geometry.isMoELayer(layer) {
                // A sparse MoE layer: the router (the pack's raw plane,
                // widened to f32) and the two banks — ONE contiguous view
                // each, the resident experts stacked in order (C119).
                let resE = geometry.effectiveResidentExperts
                let coldE = max(0, geometry.numExperts - resE)
                let outDim = geometry.moeRoutedDim
                let routerRows = geometry.moeSplitFeatureRouting ? (2 * geometry.numExperts) : geometry.numExperts
                tensor(routerRows * h, .f32)
                if geometry.hasBiasCorrectedRouter {
                    tensor(geometry.numExperts, .f32)
                }
                if geometry.routedScalingFactorLearnable {
                    tensor(geometry.numExperts, .f32)
                }
                q4Expert(resE * 2 * geometry.moeIntermediate, outDim)
                q4ExpertDown(resE * outDim, geometry.moeIntermediate)
                if coldE > 0 && !allocatedSharedCold {
                    allocatedSharedCold = true
                    q4Expert(coldE * 2 * geometry.moeIntermediate, outDim)
                    q4ExpertDown(coldE * outDim, geometry.moeIntermediate)
                }
                if let latL = geometry.moeLatentSize, latL > 0 {
                    q4(latL, h)
                    tensor(latL, .f16)
                    q4(h, latL)
                }
                if geometry.sharedExpertIntermediate > 0 {
                    q4(2 * geometry.sharedExpertIntermediate, h)
                    q4(h, geometry.sharedExpertIntermediate)
                    if geometry.sharedExpertHasGate {
                        tensor(h, .f32)
                    }
                }
            } else {
                q4(2 * i, h)
                q4(h, i)
            }
            if geometry.isCompressedAttention(layer), let hca = geometry.hcaConfig {
                let qHeads = hca.qHeads
                let headDim = hca.headDim
                let qRank = hca.qLoraRank
                let oGroups = max(1, hca.oGroups)
                let oLoraRank = hca.oLoraRank
                let groupIn = (qHeads / oGroups) * headDim
                let compRatio = max(1, hca.defaultCompressRatio)
                q4(qRank + 3 * headDim, h)
                tensor(qRank, .f16)
                tensor(headDim, .f16)
                tensor(headDim, .f16)
                tensor(compRatio * headDim, .f32)
                q4(qHeads * headDim + oGroups * oLoraRank, qRank)
                q4(oGroups * oLoraRank, groupIn)
                q4(h, oGroups * oLoraRank)
                if hca.useVHA {
                    tensor(qHeads * qHeads, .f16)
                }
                if hca.hasAttnSinkBias {
                    tensor(qHeads, .f32)
                }
            } else if geometry.isFullAttention(layer) {
                if geometry.usesNoPEMLA {
                    let qRank = geometry.qLoraRank
                    let kvRank = geometry.kvLoraRank
                    let qTotal = geometry.qHeads * geometry.headDim
                    let vTotal = geometry.qHeads * geometry.vHeadDim
                    let idxH = geometry.indexerHeads
                    let idxD = geometry.indexerHeadDim
                    q4(qRank + kvRank + idxD, h)
                    tensor(qRank, .f16)
                    tensor(kvRank, .f16)
                    tensor(idxD, .f16)
                    tensor(idxD, .f16)
                    tensor((idxD + idxH) * h, .f16)
                    tensor(4 * idxD, .f16)
                    q4(qTotal + idxH * idxD, qRank)
                    q4(qTotal + vTotal, kvRank)
                    q4(h, vTotal)
                } else if geometry.usesDecoupledRopeMLA, let hmla = geometry.hybridMLAConfig {
                    let qHeads = hmla.qHeads
                    let qRank = hmla.qLoraRank
                    let kvRank = hmla.kvLoraRank
                    let nopeDim = hmla.qkNopeHeadDim
                    let ropeDim = hmla.qkRopeHeadDim
                    let vDim = hmla.vHeadDim
                    let idxH = max(1, geometry.indexerHeads)
                    let idxD = max(32, geometry.indexerHeadDim)
                    let qCh = qHeads * (nopeDim + ropeDim + (geometry.attnOutputGate ? vDim : 0))
                    q4(qRank + kvRank + ropeDim + idxD, h)
                    tensor(qRank, .f16)
                    tensor(kvRank, .f16)
                    tensor(idxD, .f16)
                    tensor(idxD, .f16)
                    tensor(idxH * h, .f16)
                    q4(qCh + idxH * idxD, qRank)
                    q4(qHeads * (nopeDim + vDim), kvRank)
                    q4(h, qHeads * vDim)
                    if hmla.useVHA {
                        tensor(qHeads * qHeads, .f16)
                    }
                    if hmla.hasAttnSinkBias {
                        tensor(qHeads, .f32)
                    }
                } else {
                    q4(geometry.qProjOut + 2 * kv, h)
                    q4(h, geometry.qHeads * geometry.headDim)
                    // The q/k norms only exist when the model has them.
                    if geometry.descriptor.hasQKNorm {
                        tensor(geometry.headDim, .f16)
                        tensor(kv, .f16)
                    }
                }
            } else {
                if geometry.gdnVectorDecay {
                    let lr = geometry.gdnLowRankDim
                    q4(geometry.gdnQKVChannels, h)
                    q4(2 * lr + geometry.gdnVHeads, h)
                    tensor(geometry.gdnZChannels * lr, .f16)
                    tensor(geometry.gdnZChannels * lr, .f16)
                    tensor(geometry.gdnQKVChannels * geometry.gdnConvKernel, .f16)
                    tensor(geometry.gdnHeadDim, .f16)
                    tensor(geometry.gdnVHeads, .f32)
                    tensor(geometry.gdnZChannels, .f32)
                    q4(h, geometry.gdnZChannels)
                } else {
                    q4(geometry.gdnQKVChannels + geometry.gdnZChannels
                        + 2 * geometry.gdnVHeads, h)
                    tensor(geometry.gdnQKVChannels * geometry.gdnConvKernel, .f16)
                    tensor(geometry.gdnHeadDim, .f16)
                    tensor(geometry.gdnVHeads, .f32)
                    tensor(geometry.gdnVHeads, .f32)
                    q4(h, geometry.gdnZChannels)
                }
            }
            if let mhc = geometry.mhcConfig, layer >= mhc.start, layer < mhc.end {
                let routerRows = 2 * mhc.streams + (mhc.fixedIdentity ? 0 : mhc.streams * mhc.streams)
                tensor(routerRows * mhc.streams * h, .f32)
                tensor(routerRows, .f32)
                tensor(routerRows * mhc.streams * h, .f32)
                tensor(routerRows, .f32)
            }
        }
        tensor(h, .f16)
        if let mhc = geometry.mhcConfig, mhc.hasExitHead {
            tensor(mhc.streams * mhc.streams * h, .f32)
            tensor(mhc.streams, .f32)
        }
        if geometry.multimaxHead {
            tensor(8, .f32)
        }
        let tableRopeRows = geometry.usesLatentOrCompressedAttention ? 1 : ropeRows
        requests.append(QWTensorShape([tableRopeRows, geometry.rotaryDims], .f32))
        if mtpDrafts > 0 && geometry.mtpSharedLastLayer {
            let c = geometry.mhcConfig?.streams ?? 4
            tensor(h, .f16)
            tensor(h, .f16)
            q4(h, h)
            q4(h, h)
            tensor(c * c * h, .f32)
            tensor(c, .f32)
            tensor(h, .f16)
            if geometry.hasBiasCorrectedRouter && geometry.numExperts > 0 {
                tensor(geometry.numExperts, .f32)
            }
        } else if mtpDrafts > 0 {
            // MTP Layer weights (Qwen3.5 mtp.* or NextN layers.<numLayers>.*):
            // 1. pre_fc_norm_embedding / enorm
            tensor(h, .f16)
            // 2. pre_fc_norm_hidden / hnorm
            tensor(h, .f16)
            // 3. fc / eh_proj [H, 2*H]
            q4(h, 2 * h)
            // 4. input_layernorm
            tensor(h, .f16)
            // 5-8. self_attn (NoPE MLA or GQA)
            if geometry.usesNoPEMLA {
                let qRank = geometry.qLoraRank
                let kvRank = geometry.kvLoraRank
                let qTotal = geometry.qHeads * geometry.headDim
                let vTotal = geometry.qHeads * geometry.vHeadDim
                let idxH = geometry.indexerHeads
                let idxD = geometry.indexerHeadDim
                q4(qRank + kvRank + idxD, h)
                tensor(qRank, .f16)
                tensor(kvRank, .f16)
                tensor(idxD, .f16)
                tensor(idxD, .f16)
                tensor((idxD + idxH) * h, .f16)
                tensor(4 * idxD, .f16)
                q4(qTotal + idxH * idxD, qRank)
                q4(qTotal + vTotal, kvRank)
                q4(h, vTotal)
            } else {
                q4(geometry.qProjOut + 2 * kv, h)
                q4(h, geometry.qHeads * geometry.headDim)
                tensor(geometry.headDim, .f16)
                tensor(kv, .f16)
            }
            // 9. post_attention_layernorm
            tensor(h, .f16)
            // 10. mlp gate_up / down_proj (or sparse MoE bank + shared expert)
            if geometry.numExperts > 0 {
                let resE = geometry.effectiveResidentExperts
                let coldE = max(0, geometry.numExperts - resE)
                tensor(geometry.numExperts * h, .f32)
                if geometry.hasBiasCorrectedRouter {
                    tensor(geometry.numExperts, .f32)
                }
                q4Expert(resE * 2 * geometry.moeIntermediate, h)
                q4ExpertDown(resE * h, geometry.moeIntermediate)
                if coldE > 0 && !allocatedSharedCold {
                    allocatedSharedCold = true
                    q4Expert(coldE * 2 * geometry.moeIntermediate, h)
                    q4ExpertDown(coldE * h, geometry.moeIntermediate)
                }
                if geometry.sharedExpertIntermediate > 0 {
                    q4(2 * geometry.sharedExpertIntermediate, h)
                    q4(h, geometry.sharedExpertIntermediate)
                    if geometry.sharedExpertHasGate {
                        tensor(h, .f32)
                    }
                }
            } else {
                q4(2 * i, h)
                q4(h, i)
            }
            // 12. norm / shared_head.norm
            tensor(h, .f16)
        }
        var total = 0
        for shape in requests {
            let (rawBytes, productOverflow) = shape.elementCount.multipliedReportingOverflow(
                by: shape.dtype == .f16 ? 2 : 4)
            let (padded, paddingOverflow) = rawBytes.addingReportingOverflow(255)
            let (next, totalOverflow) = total.addingReportingOverflow(padded & ~255)
            guard !productOverflow, !paddingOverflow, !totalOverflow else {
                throw QWEngineDriverError.state("weight allocation plan exceeds address space")
            }
            total = next
        }
        shapes = requests
        bytes = total
    }

    func check(_ shape: QWTensorShape, at index: Int) throws {
        guard shapes.indices.contains(index), shapes[index] == shape else {
            throw QWEngineDriverError.state("weight loader/plan mismatch at allocation \(index)")
        }
    }

    func finish(requests: Int, usedBytes: Int) throws {
        guard requests == shapes.count, usedBytes == bytes else {
            throw QWEngineDriverError.state(
                "weight loader/plan mismatch: \(requests)/\(shapes.count) requests, \(usedBytes)/\(bytes) bytes")
        }
    }
}
