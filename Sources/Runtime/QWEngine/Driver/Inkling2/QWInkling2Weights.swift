// QWInkling2Weights.swift — Zero-conversion safetensors GPU loader for Inkling-2.
import Foundation
import Metal
import QWAlloc
import QWCore
import QWGEMM
import QWHead
import QWQuant

struct Ink2Slice: @unchecked Sendable {
    let buffer: any MTLBuffer
    let offset: Int
    let length: Int

    func tv(count: Int, dtype: QWDType) -> QWTensorValue {
        QWTensorValue(region: QWMRegion(buffer: buffer, base: 0, length: buffer.length),
                      offset: offset, shape: QWTensorShape([count], dtype))
    }
}

typealias Ink2LinearW = QWLinearWeight

extension QWSubAllocator {
    static func makeWeightAllocator(device: any MTLDevice, capacity: Int) throws -> QWSubAllocator {
        let aligned = max(4096, ((capacity + 4095) / 4096) * 4096)
        guard let buf = device.makeBuffer(length: aligned, options: .storageModeShared) else {
            throw QWInkling2Error.allocationFailed("weights buffer \(aligned) bytes")
        }
        return QWSubAllocator(region: QWMRegion(buffer: buf, base: 0, length: aligned))
    }

    var slabBuffer: any MTLBuffer { region.buffer }

    func allocateBytes(_ byteCount: Int, label: String) throws -> Ink2Slice {
        do {
            let off = try allocate(bytes: byteCount)
            return Ink2Slice(buffer: region.buffer, offset: off, length: byteCount)
        } catch {
            throw QWInkling2Error.allocationFailed("overflow storing \(label)")
        }
    }

    func copy(_ name: String, from st: QWSafeTensorsContainer) throws -> Ink2Slice {
        guard let info = st.tensor(name) else {
            throw QWSafeTensorsError.missingTensor(name)
        }
        let slice = try allocateBytes(info.byteLength, label: name)
        let dst = slice.buffer.contents().advanced(by: slice.offset)
        try withExtendedLifetime(slice.buffer) {
            try st.withBytes(of: name) { raw in
                guard raw.count == slice.length else {
                    throw QWInkling2Error.allocationFailed("size mismatch storing \(name)")
                }
                if let src = raw.baseAddress, raw.count > 0 {
                    dst.copyMemory(from: src, byteCount: raw.count)
                }
            }
        }
        return slice
    }
}

enum QWInkling2Error: Error, CustomStringConvertible {
    case allocationFailed(String)
    case missingConfig(String)
    case badState(String)

    var description: String {
        switch self {
        case .allocationFailed(let m): return "QWInkling2: allocation failed — \(m)"
        case .missingConfig(let m): return "QWInkling2: missing config — \(m)"
        case .badState(let m): return "QWInkling2: \(m)"
        }
    }
}

final class QWInkling2Weights: @unchecked Sendable {
    enum AttnNorm: @unchecked Sendable {
        case sliding(inputNorm: Ink2Slice)
        case global(qNorm: Ink2Slice, kNorm: Ink2Slice, vNorm: Ink2Slice)
    }

    enum KVSConv: @unchecked Sendable {
        case staticConv(k: Ink2Slice, v: Ink2Slice)
        case dynamicConv(
            kStatic: Ink2Slice, kProjW: Ink2LinearW,
            vStatic: Ink2Slice, vProjW: Ink2LinearW
        )
    }

    struct ControllerW: @unchecked Sendable {
        let down: Ink2Slice?
        let forget: Ink2Slice?
        let gain: Ink2Slice?
        let downW: Ink2LinearW
        let forgetGainW: Ink2LinearW
    }

    struct LayerControllers: @unchecked Sendable {
        let attn: ControllerW
        let k: ControllerW
        let v: ControllerW
        let mlp: ControllerW
    }

    enum ExpertsW: @unchecked Sendable {
        case nvfp4(gateUp: QWGemmFP.NVFP4Bank, down: QWGemmFP.NVFP4Bank)
        case mxfp8(gateUp: QWGemmFP.MXFP8Bank, down: QWGemmFP.MXFP8Bank)
        case qwp(gateUp: QWWeightView, down: QWWeightView)

        var gateUpBank: QWMoEExpertBank {
            switch self {
            case .nvfp4(let gateUp, _): return .nvfp4(gateUp)
            case .mxfp8(let gateUp, _): return .mxfp8(gateUp)
            case .qwp(let gateUp, _): return .weightView(gateUp)
            }
        }

        var upBank: QWMoEExpertBank? { nil }

        var downBank: QWMoEExpertBank {
            switch self {
            case .nvfp4(_, let down): return .nvfp4(down)
            case .mxfp8(_, let down): return .mxfp8(down)
            case .qwp(_, let down): return .weightView(down)
            }
        }
    }

    struct LayerWeights: @unchecked Sendable {
        let layerIdx: Int
        let isSliding: Bool
        let attnInputQuery: Ink2Slice?
        let mlpInputQuery: Ink2Slice
        let attnNorm: AttnNorm
        let postAttnNorm: Ink2Slice
        let attnSConv: Ink2Slice
        let mlpSConv: Ink2Slice
        let projLayout: AttnProjSlabLayout
        let qProjW: Ink2LinearW
        let kProjW: Ink2LinearW
        let vProjW: Ink2LinearW
        let oProjW: Ink2LinearW
        let gateProjW: Ink2LinearW
        let rProjW: Ink2LinearW
        let sinkProjW: Ink2LinearW
        let qgrsBank: QWGemmFP.BF16Bank?
        let qgrskvBank: QWGemmFP.BF16Bank?
        let qgrsView: QWWeightView?
        let qgrskvView: QWWeightView?
        let qScale: Ink2Slice
        let relScale: Ink2Slice
        let relProj: Ink2Slice
        let relProjTV: QWTensorValue
        let kvSConv: KVSConv
        let controllers: LayerControllers?
        let routerGate: Ink2Slice
        let routerGateBank: QWGemmFP.BF16Bank
        let routerBias: Ink2Slice
        let experts: ExpertsW
    }

    let embedTokens: Ink2Slice
    let embedNorm: Ink2Slice
    let lmHeadSlice: Ink2Slice?
    let lmHeadTV: QWTensorValue?
    let lmHeadBank: QWGemmFP.BF16Bank?
    let lmHeadStorage: QWHeadStorage
    let finalNorm: Ink2Slice
    let finalInputQuery: Ink2Slice
    let messageQueries: Ink2Slice
    let layers: [LayerWeights]
    let residentBuffers: [any MTLBuffer]
    let totalBytes: Int
    let hasVisionWeights: Bool
    let weightBasis: QWWeightBasis
    let isQuantizedQWP: Bool

    init(
        embedTokens: Ink2Slice,
        embedNorm: Ink2Slice,
        lmHeadSlice: Ink2Slice?,
        lmHeadTV: QWTensorValue?,
        lmHeadBank: QWGemmFP.BF16Bank?,
        lmHeadStorage: QWHeadStorage,
        finalNorm: Ink2Slice,
        finalInputQuery: Ink2Slice,
        messageQueries: Ink2Slice,
        layers: [LayerWeights],
        residentBuffers: [any MTLBuffer],
        hasVisionWeights: Bool,
        weightBasis: QWWeightBasis,
        isQuantizedQWP: Bool
    ) {
        self.embedTokens = embedTokens
        self.embedNorm = embedNorm
        self.lmHeadSlice = lmHeadSlice
        self.lmHeadTV = lmHeadTV
        self.lmHeadBank = lmHeadBank
        self.lmHeadStorage = lmHeadStorage
        self.finalNorm = finalNorm
        self.finalInputQuery = finalInputQuery
        self.messageQueries = messageQueries
        self.layers = layers
        self.residentBuffers = residentBuffers
        self.totalBytes = residentBuffers.reduce(0) { $0 + $1.length }
        self.hasVisionWeights = hasVisionWeights
        self.weightBasis = weightBasis
        self.isQuantizedQWP = isQuantizedQWP
    }

    convenience init(
        device: any MTLDevice,
        st: QWSafeTensorsContainer,
        descriptor: QWModelDescriptor,
        ink2: QWModelDescriptor.Inkling2Config
    ) throws {
        func sumBytes(forPrefix prefix: String?, excludePrefix: String?) -> Int {
            var total = 0
            for (name, t) in st.tensors {
                if let prefix, !name.hasPrefix(prefix) { continue }
                if let excludePrefix, name.hasPrefix(excludePrefix) { continue }
                if name.contains("vision_tower.") || name.hasPrefix("visual.") || name.contains(".visual.") { continue }
                let aligned = ((t.byteLength + 255) / 256) * 256
                let isSlabDoubled = ((descriptor.hidden * 2) % 16 != 0 && name.contains(".self_attn.")
                    && (name.hasSuffix("q_proj.weight") || name.hasSuffix("gate_proj.weight")
                        || name.hasSuffix("r_proj.weight") || name.hasSuffix("sink_proj.weight")))
                    || (name.hasSuffix("forget_weight") || name.hasSuffix("gain_weight"))
                total += isSlabDoubled ? (aligned * 2) : aligned
            }
            return total + 65536
        }

        let pfx = st.tensor("language_model.model.embed_tokens.weight") != nil ? "language_model." : ""
        let mdl = "\(pfx)model."
        let globalNames = [
            "\(mdl)embed_tokens.weight", "\(mdl)embed_norm.weight",
            "\(pfx)lm_head.weight", "\(mdl)norm.weight",
            "\(mdl)final_input.query", "\(mdl)message_queries"
        ]
        var gCap = 65536
        for name in globalNames {
            guard let t = st.tensor(name) else {
                throw QWSafeTensorsError.missingTensor(name)
            }
            gCap += ((t.byteLength + 255) / 256) * 256
        }
        let gBump = try QWSubAllocator.makeWeightAllocator(device: device, capacity: gCap)
        let embedTokens = try gBump.copy(globalNames[0], from: st)
        let embedNorm = try gBump.copy(globalNames[1], from: st)
        let lmHeadSlice = try gBump.copy(globalNames[2], from: st)
        let headVocab = ink2.unpaddedVocabSize
        let lmHeadTV = lmHeadSlice.tv(count: headVocab * descriptor.hidden, dtype: .bf16)
        let lmHeadBank = lmHeadSlice.bf16Bank(n: headVocab, k: descriptor.hidden)
        let finalNorm = try gBump.copy(globalNames[3], from: st)
        let finalInputQuery = try gBump.copy(globalNames[4], from: st)
        let messageQueries = try gBump.copy(globalNames[5], from: st)

        let H = descriptor.hidden
        let nH = descriptor.qHeads
        let nKV = descriptor.kvHeads
        let headDim = descriptor.headDim
        let qDim = nH * headDim
        let kvDim = nKV * headDim
        let rDim = nH * ink2.dRel
        let moeI = descriptor.moeIntermediate ?? 1152
        let nExp = descriptor.numExperts ?? 128

        var buffers: [any MTLBuffer] = [gBump.slabBuffer]
        var builtLayers: [LayerWeights] = []
        builtLayers.reserveCapacity(descriptor.numLayers)

        for l in 0..<descriptor.numLayers {
            let lPfx = "\(mdl)layers.\(l)."
            let lBump = try QWSubAllocator.makeWeightAllocator(device: device, capacity: sumBytes(forPrefix: lPfx, excludePrefix: nil))
            let isSliding = ink2.isSlidingLayer(l)
            let attnQuery = (l > 0) ? try lBump.copy("\(lPfx)attention_input.query", from: st) : nil
            let mlpQuery = try lBump.copy("\(lPfx)mlp_input.query", from: st)
            let attnNorm: AttnNorm = isSliding
                ? .sliding(inputNorm: try lBump.copy("\(lPfx)input_layernorm.weight", from: st))
                : .global(
                    qNorm: try lBump.copy("\(lPfx)q_input_layernorm.weight", from: st),
                    kNorm: try lBump.copy("\(lPfx)k_input_layernorm.weight", from: st),
                    vNorm: try lBump.copy("\(lPfx)v_input_layernorm.weight", from: st)
                )
            let postAttnNorm = try lBump.copy("\(lPfx)post_attention_layernorm.weight", from: st)
            let attnSConv = try lBump.copy("\(lPfx)attn_sconv.conv.weight", from: st)
            let mlpSConv = try lBump.copy("\(lPfx)mlp_sconv.conv.weight", from: st)
            let sa = "\(lPfx)self_attn."
            let projLoad = try lBump.buildAttnProjSlab(
                sa: sa, isSliding: isSliding, qDim: qDim, rDim: rDim,
                sinkDim: nH, kvDim: kvDim, hidden: H, from: st
            )
            let oProj = try lBump.copy("\(sa)o_proj.weight", from: st)
            let qScale = try lBump.copy("\(sa)q_scale", from: st)
            let relScale = try lBump.copy("\(sa)rel_scale", from: st)
            let relProj = try lBump.copy("\(sa)rel_proj", from: st)
            let relProjTV = QWTensorValue(
                region: QWMRegion(buffer: relProj.buffer, base: 0, length: relProj.buffer.length),
                offset: relProj.offset,
                shape: QWTensorShape([relProj.length / 2], .bf16)
            )
            let kvSConv: KVSConv
            if isSliding {
                kvSConv = .staticConv(
                    k: try lBump.copy("\(sa)k_sconv.conv.weight", from: st),
                    v: try lBump.copy("\(sa)v_sconv.conv.weight", from: st)
                )
            } else {
                let kp = try lBump.copy("\(sa)k_sconv.weight_proj.weight", from: st)
                let vp = try lBump.copy("\(sa)v_sconv.weight_proj.weight", from: st)
                kvSConv = .dynamicConv(
                    kStatic: try lBump.copy("\(sa)k_sconv.static_weight", from: st),
                    kProjW: .bf16(kp.bf16Bank(n: nKV * 4, k: H)),
                    vStatic: try lBump.copy("\(sa)v_sconv.static_weight", from: st),
                    vProjW: .bf16(vp.bf16Bank(n: nKV * 4, k: H))
                )
            }

            let ctrl: LayerControllers?
            if l >= ink2.loopCoreStart && l < ink2.loopCoreEnd {
                func loadCtrl(_ name: String) throws -> ControllerW {
                    try lBump.buildControllerWeights(
                        cp: "\(lPfx)controllers.\(name).", hidden: H, rank: ink2.loopGateRank, from: st
                    )
                }
                ctrl = LayerControllers(
                    attn: try loadCtrl("attn"),
                    k: try loadCtrl("k"),
                    v: try loadCtrl("v"),
                    mlp: try loadCtrl("mlp")
                )
            } else {
                ctrl = nil
            }

            let routerGate = try lBump.copy("\(lPfx)mlp.gate_weight", from: st)
            let routerGateBank = routerGate.bf16Bank(n: nExp, k: H)
            let routerBias = try lBump.copy("\(lPfx)mlp.e_score_correction_bias", from: st)
            let sw = "\(lPfx)mlp.switch_mlp."
            let mode = ink2.expertProjSpec(layer: l, projection: "gate_proj").mode
            let experts: ExpertsW
            switch mode {
            case .nvfp4:
                experts = try lBump.buildNVFP4Experts(
                    sw: sw, experts: nExp, moeI: moeI, hidden: H, from: st
                )
            case .mxfp8:
                experts = try lBump.buildMXFP8Experts(
                    sw: sw, experts: nExp, moeI: moeI, hidden: H, from: st
                )
            case .bf16:
                throw QWInkling2Error.badState("BF16 switch_mlp not supported in nativeMixedFP loader")
            }

            buffers.append(lBump.slabBuffer)
            builtLayers.append(LayerWeights(
                layerIdx: l, isSliding: isSliding,
                attnInputQuery: attnQuery, mlpInputQuery: mlpQuery,
                attnNorm: attnNorm, postAttnNorm: postAttnNorm,
                attnSConv: attnSConv, mlpSConv: mlpSConv,
                projLayout: projLoad.layout,
                qProjW: .bf16(projLoad.qProj.bf16Bank(n: qDim, k: H)),
                kProjW: .bf16(projLoad.kProj.bf16Bank(n: kvDim, k: H)),
                vProjW: .bf16(projLoad.vProj.bf16Bank(n: kvDim, k: H)),
                oProjW: .bf16(oProj.bf16Bank(n: H, k: qDim)),
                gateProjW: .bf16(projLoad.gateProj.bf16Bank(n: qDim, k: H)),
                rProjW: .bf16(projLoad.rProj.bf16Bank(n: rDim, k: H)),
                sinkProjW: .bf16(projLoad.sinkProj.bf16Bank(n: nH, k: H)),
                qgrsBank: projLoad.qgrsBank, qgrskvBank: projLoad.qgrskvBank,
                qgrsView: nil, qgrskvView: nil,
                qScale: qScale, relScale: relScale, relProj: relProj, relProjTV: relProjTV,
                kvSConv: kvSConv, controllers: ctrl,
                routerGate: routerGate, routerGateBank: routerGateBank,
                routerBias: routerBias, experts: experts
            ))
        }

        let hasVis = st.tensor("vision_tower.embeddings.patch_embedding.weight") != nil
            || st.tensor("model.visual.embeddings.patch_embedding.weight") != nil
        self.init(
            embedTokens: embedTokens,
            embedNorm: embedNorm,
            lmHeadSlice: lmHeadSlice,
            lmHeadTV: lmHeadTV,
            lmHeadBank: lmHeadBank,
            lmHeadStorage: .bf16(lmHeadTV),
            finalNorm: finalNorm,
            finalInputQuery: finalInputQuery,
            messageQueries: messageQueries,
            layers: builtLayers,
            residentBuffers: buffers,
            hasVisionWeights: hasVis,
            weightBasis: .standard,
            isQuantizedQWP: false
        )
        st.adviseDontNeed()
    }
}
