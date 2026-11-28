// QWInkling2TreeSpecParityTests.swift — SHVQ4 DSpark QWP1 and 16-node tree verify/compact parity across KV modes.
import Foundation
import Metal
import XCTest
import QWAlloc
@testable import QWCore
@testable import QWEngine
import QWGEMM
import QWHAL
import QWHead
import QWKV
import QWQuant

final class QWInkling2TreeSpecParityTests: XCTestCase {
    private func requireApple9Metal() throws -> QWHal {
        try XCTSkipIf(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] == "true"
                || QWDeviceCaps.current?.supportsSimdgroupMMA != true,
            "requires bare-metal Apple Silicon GPU"
        )
        return try QWHal(geometry: .qwen38_27b)
    }

    private func makeSyntheticDescriptor(vocab: Int = 128) -> (QWModelDescriptor, QWModelDescriptor.Inkling2Config, QWModelDescriptor.Inkling2DSparkConfig) {
        let obj: [String: Any] = [
            "architectures": ["Inkling2LoopForConditionalGeneration"],
            "eos_token_id": vocab - 1,
            "model_type": "inkling2_loop",
            "generation_config": ["eos_token_id": vocab - 1, "pad_token_id": 0],
            "mtp_config": [
                "hidden_size": 64, "num_hidden_layers": 3, "num_attention_heads": 4,
                "num_key_value_heads": 2, "head_dim": 32, "intermediate_size": 64,
                "max_position_embeddings": 256, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
                "q_bias": false, "o_bias": false, "local_layer_ids": [0, 2],
                "sliding_window_size": 6, "block_size": 8, "markov_rank": 32,
                "markov_head_type": "rnn_hidden", "target_layer_ids": [0, 1, 2, 4]
            ],
            "text_config": [
                "model_type": "inkling2_loop_text",
                "model_max_length": 256,
                "hidden_size": 64,
                "num_hidden_layers": 3,
                "vocab_size": vocab,
                "unpadded_vocab_size": vocab,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 32,
                "d_rel": 8,
                "rel_extent": 32,
                "q_bias": false,
                "o_bias": false,
                "log_scaling_n_floor": 8,
                "log_scaling_alpha": 0.25,
                "rms_norm_eps": 1e-6,
                "use_value_norm_no_gamma": true,
                "qk_l2_clip_threshold": 22.6274,
                "rel_l2_clip_threshold": 8.0,
                "local_layer_ids": [0, 1],
                "sconv_kernel_size": 4,
                "final_logit_softcapping": 30.0,
                "swa_head_dim": 32,
                "swa_num_attention_heads": 4,
                "swa_num_key_value_heads": 2,
                "sliding_window_size": 6,
                "n_layers_per_attn_res_block": 1,
                "attn_res_normalize_values": true,
                "n_routed_experts": 4,
                "num_experts_per_tok": 2,
                "n_shared_experts": 0,
                "intermediate_size": 64,
                "mlp_gate_softcap": 4.0,
                "mlp_up_softcap": 25.0,
                "route_scale": 4.0,
                "use_gate_bias": true,
                "loop_config": [
                    "core_start": 1, "core_end": 2, "num_passes": 2,
                    "gate_rank": 16, "is_varloop": false
                ]
            ]
        ]
        let data = try! JSONSerialization.data(withJSONObject: obj)
        let desc = try! QWModelDescriptor.decode(configJSON: data)
        return (desc, desc.inkling2Config!, desc.inkling2Config!.dspark!)
    }

    private func makeSHVQ4Data(n: Int, k: Int, seed: UInt64) -> Data {
        let blkBytes = QWSHVQBlock.bytes(bits: 4, group: 32)
        let g = k / 32
        var out = Data(count: n * g * blkBytes)
        var rng = seed
        out.withUnsafeMutableBytes { raw in
            let p = raw.bindMemory(to: UInt8.self).baseAddress!
            for r in 0..<n {
                for gg in 0..<g {
                    let off = (r * g + gg) * blkBytes
                    rng = rng &* 6364136223846793005 &+ 1442695040888963407
                    let sc = Float16(0.02 + Double((rng >> 20) & 0x3F) * 0.001).bitPattern
                    p[off] = UInt8(sc & 0xFF)
                    p[off + 1] = UInt8(sc >> 8)
                    for b in 2..<blkBytes {
                        rng = rng &* 6364136223846793005 &+ 1442695040888963407
                        p[off + b] = UInt8(truncatingIfNeeded: (rng >> 32) ^ UInt64(b * 37))
                    }
                }
            }
        }
        return out
    }

    private func makeBF16Data(count: Int, scale: Double = 0.05, seed: UInt64) -> Data {
        var vals = [UInt16](repeating: 0, count: count)
        var rng = seed
        for i in 0..<count {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let v = (Double(Int64(bitPattern: rng >> 16) % 2001) / 1000.0) * scale
            vals[i] = QWInk2TestRef.bf16Bits(Float(v))
        }
        return vals.withUnsafeBytes { Data($0) }
    }

    private func makeBF16OnesData(count: Int, val: Float = 1.0) -> Data {
        let bits = QWInk2TestRef.bf16Bits(val)
        return [UInt16](repeating: bits, count: count).withUnsafeBytes { Data($0) }
    }

    private func writeSyntheticDSparkQWP(
        to url: URL,
        vocab V: Int,
        cfg: QWModelDescriptor.Inkling2DSparkConfig
    ) throws {
        let H = cfg.hiddenSize, R = cfg.markovRank
        let nH = cfg.qHeads, nKV = cfg.kvHeads, hd = cfg.headDim
        let qDim = nH * hd, kvDim = nKV * hd, mlpI = cfg.intermediateSize

        var specs: [(name: String, shape: [Int], quant: Bool, data: Data)] = []
        specs.append(("fuse_weight", [4, 4], false, makeBF16OnesData(count: 16, val: 0.25)))
        specs.append(("mask_embed", [H], false, makeBF16Data(count: H, seed: 101)))
        specs.append(("norm.weight", [H], false, makeBF16OnesData(count: H)))
        specs.append(("markov_head.markov_w1.weight", [V, R], false, makeBF16Data(count: V * R, seed: 102)))
        specs.append(("markov_head.rnn_proj.weight", [3 * R, 2 * R + 2 * H], true,
                      makeSHVQ4Data(n: 3 * R, k: 2 * R + 2 * H, seed: 103)))
        specs.append(("markov_head.rnn_proj.bias", [3 * R], false, makeBF16Data(count: 3 * R, scale: 0.01, seed: 104)))
        specs.append(("markov_head.hidden_out.weight", [H, R], true, makeSHVQ4Data(n: H, k: R, seed: 105)))
        specs.append(("markov_head.hidden_out.bias", [H], false, makeBF16Data(count: H, scale: 0.01, seed: 106)))
        specs.append(("markov_head.dflare_norm.weight", [H], false, makeBF16OnesData(count: H)))

        for l in 0..<cfg.numLayers {
            let p = "layers.\(l)."
            let s = UInt64(200 + l * 30)
            specs.append((p + "dflare_norm.weight", [H], false, makeBF16OnesData(count: H)))
            specs.append((p + "input_layernorm.weight", [H], false, makeBF16OnesData(count: H)))
            specs.append((p + "post_attention_layernorm.weight", [H], false, makeBF16OnesData(count: H)))
            specs.append((p + "self_attn.q_norm.weight", [hd], false, makeBF16OnesData(count: hd)))
            specs.append((p + "self_attn.k_norm.weight", [hd], false, makeBF16OnesData(count: hd)))
            specs.append((p + "self_attn.k_norm_ctx.weight", [hd], false, makeBF16OnesData(count: hd)))
            specs.append((p + "self_attn.q_proj.weight", [qDim, H], true, makeSHVQ4Data(n: qDim, k: H, seed: s + 1)))
            specs.append((p + "self_attn.k_proj.weight", [kvDim, H], true, makeSHVQ4Data(n: kvDim, k: H, seed: s + 2)))
            specs.append((p + "self_attn.v_proj.weight", [kvDim, H], true, makeSHVQ4Data(n: kvDim, k: H, seed: s + 3)))
            specs.append((p + "self_attn.o_proj.weight", [H, qDim], true, makeSHVQ4Data(n: H, k: qDim, seed: s + 4)))
            specs.append((p + "self_attn.k_proj_ctx.weight", [kvDim, H], true, makeSHVQ4Data(n: kvDim, k: H, seed: s + 5)))
            specs.append((p + "self_attn.v_proj_ctx.weight", [kvDim, H], true, makeSHVQ4Data(n: kvDim, k: H, seed: s + 6)))
            specs.append((p + "mlp.gate_proj.weight", [mlpI, H], true, makeSHVQ4Data(n: mlpI, k: H, seed: s + 7)))
            specs.append((p + "mlp.up_proj.weight", [mlpI, H], true, makeSHVQ4Data(n: mlpI, k: H, seed: s + 8)))
            specs.append((p + "mlp.down_proj.weight", [H, mlpI], true, makeSHVQ4Data(n: H, k: mlpI, seed: s + 9)))
        }

        var blob = Data("QWP1".utf8)
        blob.append(Data(count: 60))
        var tensorJSON: [String: Any] = [:]
        for item in specs {
            while (blob.count % 64) != 0 { blob.append(0) }
            let off = blob.count
            blob.append(item.data)
            if item.quant {
                tensorJSON[item.name] = [
                    "shape": item.shape,
                    "hq4_off": off,
                    "hq": ["bits": 4, "group": 32, "vq": "shvq4"]
                ]
            } else {
                tensorJSON[item.name] = ["shape": item.shape, "kind": "bf16", "raw_off": off]
            }
        }
        let hdrObj: [String: Any] = ["provenance": ["source": "unit-test"], "tensors": tensorJSON]
        let hdrData = try JSONSerialization.data(withJSONObject: hdrObj)
        var hdrLen = UInt64(hdrData.count).littleEndian
        blob.replaceSubrange(4..<12, with: Data(bytes: &hdrLen, count: 8))
        blob.append(hdrData)
        try blob.write(to: url)
    }

    private func buildSyntheticWeights(hal: QWHal, desc: QWModelDescriptor, ink2: QWModelDescriptor.Inkling2Config) throws -> QWInkling2Weights {
        let H = desc.hidden, V = desc.vocab, nH = desc.qHeads, nKV = desc.kvHeads, hd = desc.headDim
        let qDim = nH * hd, kvDim = nKV * hd, rDim = nH * ink2.dRel
        let nExp = desc.numExperts!, moeI = desc.moeIntermediate!
        let bump = try QWSubAllocator.makeWeightAllocator(device: hal.device, capacity: 8 << 20)

        func putData(_ d: Data, _ label: String) throws -> Ink2Slice {
            let sl = try bump.allocateBytes(d.count, label: label)
            let dst = sl.buffer.contents().advanced(by: sl.offset)
            withExtendedLifetime(sl.buffer) {
                d.withUnsafeBytes { raw in
                    if let p = raw.baseAddress, !d.isEmpty { dst.copyMemory(from: p, byteCount: d.count) }
                }
            }
            return sl
        }
        func shvqView(n: Int, k: Int, seed: UInt64, _ label: String) throws -> QWWeightView {
            let sl = try putData(makeSHVQ4Data(n: n, k: k, seed: seed), label)
            return QWWeightView(blocks: sl.tv(count: sl.length / 2, dtype: .f16), n: n, k: k, containerKind: .shvq(bits: 4, group: 32))!
        }

        let embedTokens = try putData(makeBF16Data(count: V * H, scale: 0.2, seed: 1), "embed")
        let embedNorm = try putData(makeBF16OnesData(count: H), "embedNorm")
        let lmHeadSl = try putData(makeSHVQ4Data(n: V, k: H, seed: 2), "lmHead")
        let lmHeadTV = lmHeadSl.tv(count: lmHeadSl.length / 2, dtype: .f16)
        let lmHeadW = QWHeadWeight(blocks: lmHeadTV, n: V, k: H, bits: 4, groupSize: 32, isRotated: true, isIncumbentContainer: false, isSHVQ: true)!
        let finalNorm = try putData(makeBF16OnesData(count: H), "finalNorm")
        let finalQ = try putData(makeBF16Data(count: 8 * 8, scale: 0.1, seed: 3), "finalQ")
        let msgQ = try putData(makeBF16Data(count: 4 * 4 * H, scale: 0.05, seed: 4), "msgQ")

        var layers: [QWInkling2Weights.LayerWeights] = []
        for l in 0..<desc.numLayers {
            let s = UInt64(50 + l * 40)
            let isSl = ink2.isSlidingLayer(l)
            let layout = AttnProjSlabLayout(qDim: qDim, rDim: rDim, sinkDim: nH, kvDim: kvDim, isSliding: isSl)
            let qW: Ink2LinearW = .quant(try shvqView(n: qDim, k: H, seed: s + 1, "q"))
            let gW: Ink2LinearW = .quant(try shvqView(n: qDim, k: H, seed: s + 2, "g"))
            let rW: Ink2LinearW = .quant(try shvqView(n: rDim, k: H, seed: s + 3, "r"))
            let sW: Ink2LinearW = .quant(try shvqView(n: nH, k: H, seed: s + 4, "s"))
            let kW: Ink2LinearW = .quant(try shvqView(n: kvDim, k: H, seed: s + 5, "k"))
            let vW: Ink2LinearW = .quant(try shvqView(n: kvDim, k: H, seed: s + 6, "v"))
            let oW: Ink2LinearW = .quant(try shvqView(n: H, k: qDim, seed: s + 7, "o"))
            let relProj = try putData(makeBF16Data(count: ink2.dRel * ink2.relExtent, scale: 0.05, seed: s + 8), "relProj")
            let kvSConv: QWInkling2Weights.KVSConv = isSl
                ? .staticConv(k: try putData(makeBF16Data(count: kvDim * 4, scale: 0.1, seed: s + 9), "kc"),
                              v: try putData(makeBF16Data(count: kvDim * 4, scale: 0.1, seed: s + 10), "vc"))
                : .dynamicConv(
                    kStatic: try putData(makeBF16Data(count: kvDim * 4, scale: 0.1, seed: s + 11), "ks"),
                    kProjW: .bf16((try putData(makeBF16Data(count: nKV * 4 * H, scale: 0.05, seed: s + 12), "kp")).bf16Bank(n: nKV * 4, k: H)),
                    vStatic: try putData(makeBF16Data(count: kvDim * 4, scale: 0.1, seed: s + 13), "vs"),
                    vProjW: .bf16((try putData(makeBF16Data(count: nKV * 4 * H, scale: 0.05, seed: s + 14), "vp")).bf16Bank(n: nKV * 4, k: H))
                )
            func mkCtrl(_ seedBase: UInt64) throws -> QWInkling2Weights.ControllerW {
                let rank = ink2.loopGateRank
                let dw = try putData(makeBF16Data(count: rank * 2 * H, scale: 0.04, seed: seedBase), "cd")
                let fg = try putData(makeBF16Data(count: 2 * H * rank, scale: 0.04, seed: seedBase + 1), "cfg")
                let fSl = Ink2Slice(buffer: fg.buffer, offset: fg.offset, length: H * rank * 2)
                let gSl = Ink2Slice(buffer: fg.buffer, offset: fg.offset + H * rank * 2, length: H * rank * 2)
                return QWInkling2Weights.ControllerW(
                    down: dw, forget: fSl, gain: gSl,
                    downW: .bf16(dw.bf16Bank(n: rank, k: 2 * H)),
                    forgetGainW: .bf16(fg.bf16Bank(n: 2 * H, k: rank))
                )
            }
            let ctrls: QWInkling2Weights.LayerControllers? = (l >= ink2.loopCoreStart && l < ink2.loopCoreEnd)
                ? QWInkling2Weights.LayerControllers(
                    attn: try mkCtrl(s + 20), k: try mkCtrl(s + 22),
                    v: try mkCtrl(s + 24), mlp: try mkCtrl(s + 26)
                )
                : nil
            let rGate = try putData(makeBF16Data(count: nExp * H, scale: 0.1, seed: s + 15), "rg")
            let rBias = try putData(Data(count: nExp * 4), "rb")
            let guView = try shvqView(n: nExp * 2 * moeI, k: H, seed: s + 16, "mgu")
            let dnView = try shvqView(n: nExp * H, k: moeI, seed: s + 17, "mdn")
            let attnNorm: QWInkling2Weights.AttnNorm = isSl
                ? .sliding(inputNorm: try putData(makeBF16OnesData(count: H), "inNorm"))
                : .global(qNorm: try putData(makeBF16OnesData(count: H), "qn"),
                          kNorm: try putData(makeBF16OnesData(count: H), "kn"),
                          vNorm: try putData(makeBF16OnesData(count: H), "vn"))
            layers.append(QWInkling2Weights.LayerWeights(
                layerIdx: l, isSliding: isSl,
                attnInputQuery: l > 0 ? try putData(makeBF16Data(count: 8 * 8, scale: 0.1, seed: s + 18), "aq") : nil,
                mlpInputQuery: try putData(makeBF16Data(count: 8 * 8, scale: 0.1, seed: s + 19), "mq"),
                attnNorm: attnNorm, postAttnNorm: try putData(makeBF16OnesData(count: H), "pan"),
                attnSConv: try putData(makeBF16Data(count: H * 4, scale: 0.1, seed: s + 28), "asc"),
                mlpSConv: try putData(makeBF16Data(count: H * 4, scale: 0.1, seed: s + 29), "msc"),
                projLayout: layout, qProjW: qW, kProjW: kW, vProjW: vW, oProjW: oW,
                gateProjW: gW, rProjW: rW, sinkProjW: sW,
                qgrsBank: nil, qgrskvBank: nil, qgrsView: nil, qgrskvView: nil,
                qScale: try putData(makeBF16OnesData(count: qDim), "qsc"),
                relScale: try putData(makeBF16OnesData(count: rDim), "rsc"),
                relProj: relProj, relProjTV: relProj.tv(count: relProj.length / 2, dtype: .bf16),
                kvSConv: kvSConv, controllers: ctrls,
                routerGate: rGate, routerGateBank: rGate.bf16Bank(n: nExp, k: H),
                routerBias: rBias, experts: .qwp(gateUp: guView, down: dnView)
            ))
        }
        return QWInkling2Weights(
            embedTokens: embedTokens, embedNorm: embedNorm,
            lmHeadSlice: lmHeadSl, lmHeadTV: lmHeadTV, lmHeadBank: nil, lmHeadStorage: .q4g(lmHeadW),
            finalNorm: finalNorm, finalInputQuery: finalQ, messageQueries: msgQ,
            layers: layers, residentBuffers: [bump.slabBuffer], hasVisionWeights: false,
            weightBasis: .rotated, isQuantizedQWP: true
        )
    }

    func testTreePlanAndTemplateInvariants() {
        let allModes: [(DSparkTreeMode, Int)] = [
            (.tree4_d2, 4), (.tree5_d3, 5), (.tree6_d4, 6), (.tree7_d5, 7),
            (.tree8_d5, 8), (.tree8_d6, 8), (.tree16_d5, 16), (.tree16_d6, 16), (.tree16_d7, 16)
        ]
        for (mode, expectedNodes) in allModes {
            let tpl = try! XCTUnwrap(DSparkTreeTemplate.make(mode: mode, maxNodes: 16))
            XCTAssertEqual(tpl.plan.nodesCount, expectedNodes)
            XCTAssertEqual(tpl.plan.parents[0], -1)
            XCTAssertEqual(tpl.plan.sconvLags[0], SIMD4<Int32>(-1, -2, -3, 0))
            for u in 1..<expectedNodes {
                let p = tpl.plan.parents[u]
                XCTAssertTrue((0..<u).contains(p))
                XCTAssertEqual(tpl.plan.depths[u], tpl.plan.depths[p] + 1)
                XCTAssertEqual(tpl.plan.ancMask[u], tpl.plan.ancMask[p] | (1 << UInt32(u)))
            }
            for maxN in [4, 6, 10, 14] {
                let sub = try! XCTUnwrap(DSparkTreeTemplate.make(mode: mode, maxNodes: maxN))
                XCTAssertGreaterThanOrEqual(sub.plan.nodesCount, 4)
                XCTAssertLessThanOrEqual(sub.plan.nodesCount, min(expectedNodes, maxN))
                for level in sub.levels {
                    XCTAssertLessThanOrEqual(level.branches.count, 4)
                    for b in level.branches {
                        XCTAssertLessThan(b.prevTokNode, sub.plan.nodesCount)
                        XCTAssertLessThanOrEqual(b.outNodeBase + b.keepK, sub.plan.nodesCount)
                    }
                }
            }
        }
        XCTAssertNil(DSparkTreeTemplate.make(mode: .linear, maxNodes: 16))
    }

    func testSHVQ4DSparkQWPLoadIngestDraftAndTreeTop1Parity() throws {
        let hal = try requireApple9Metal()
        let (desc, ink2, dsCfg) = makeSyntheticDescriptor(vocab: 128)
        let weights = try buildSyntheticWeights(hal: hal, desc: desc, ink2: ink2)
        let session = try QWInkling2Session(
            device: hal.device, descriptor: desc, ink2: ink2, capacity: 128,
            kvModeRaw: 0, enableSpeculative: true, enableQWPScratch: true
        )
        let tmpDir = FileManager.default.temporaryDirectory.appendingPathComponent("qwg_ds_qwp_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }
        let qwpURL = tmpDir.appendingPathComponent("dspark.qwp")
        try writeSyntheticDSparkQWP(to: qwpURL, vocab: desc.vocab, cfg: dsCfg)

        let ds = try QWInkling2DSpark(
            hal: hal, mtpURL: qwpURL, descriptor: desc, ink2: ink2,
            config: dsCfg, weights: weights, capacity: 128
        )
        let fwd = try QWInkling2Forward(hal: hal, weights: weights, session: session)
        try fwd.forward(tokens: [3, 7, 11], posOffset: 0, runHeadRows: 1, greedyHead: true,
                        captureTapBuf: ds.tapBuf, tapVisits: dsCfg.targetLayerIds)
        try ds.ingestContextTaps(count: 3, posOffset: 0, rowStride: 3)

        let lin = try ds.draftBlock(anchor: 15, count: 7)
        XCTAssertEqual(lin.count, 7)

        for mode in [DSparkTreeMode.tree4_d2, .tree5_d3, .tree6_d4, .tree7_d5, .tree8_d5, .tree8_d6, .tree16_d5, .tree16_d6, .tree16_d7] {
            let tpl = try XCTUnwrap(DSparkTreeTemplate.make(mode: mode, maxNodes: 16))
            let treeToks = try ds.draftTree(anchor: 15, template: tpl)
            XCTAssertEqual(treeToks.count, tpl.plan.nodesCount)
            XCTAssertEqual(treeToks[0], 15)
            XCTAssertEqual(treeToks[1], lin[0], "tree top-1 child at root must match linear draftBlock[0] for \(mode.rawValue)")
            for cList in tpl.plan.children where cList.count > 1 {
                let ids = cList.map { treeToks[$0] }
                XCTAssertEqual(Set(ids).count, ids.count, "sibling children must be distinct")
            }
        }
    }

    func testEndToEndInkling2TreeVerifyKVCompactConvCommitParityAcrossKVModes() throws {
        let hal = try requireApple9Metal()
        let (desc, ink2, dsCfg) = makeSyntheticDescriptor(vocab: 128)
        let weights = try buildSyntheticWeights(hal: hal, desc: desc, ink2: ink2)
        let tpl = try XCTUnwrap(DSparkTreeTemplate.make(mode: .tree16_d6, maxNodes: 16))
        let tmpDir = FileManager.default.temporaryDirectory.appendingPathComponent("qwg_ds_e2e_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }
        let qwpURL = tmpDir.appendingPathComponent("dspark.qwp")
        try writeSyntheticDSparkQWP(to: qwpURL, vocab: desc.vocab, cfg: dsCfg)

        // Non-first-child path through `.tree16_d6`: parents [-1, 0,0,0, 1,1,2,3, 4,4,5,6, 8,10, 12, 14]
        let keptNodes = [0, 2, 6, 11]
        var treeTokens = (0..<16).map { 10 + $0 }
        treeTokens[0] = 19; treeTokens[2] = 43; treeTokens[6] = 71; treeTokens[11] = 97
        let prefix = Array(1...10)
        let seqTokens = [19, 43, 71, 97]
        let followTok = 63

        for kvMode in [UInt32(0), UInt32(1), UInt32(2)] {
            let sessRef = try QWInkling2Session(
                device: hal.device, descriptor: desc, ink2: ink2, capacity: 128,
                kvModeRaw: kvMode, enableSpeculative: true, enableQWPScratch: true
            )
            let fwdRef = try QWInkling2Forward(hal: hal, weights: weights, session: sessRef)
            try fwdRef.forward(tokens: prefix, posOffset: 0, runHeadRows: 0, greedyHead: false, updateState: true)

            let refTokPtr = sessRef.headTokensBuf.contents().assumingMemoryBound(to: Int32.self)
            var refVerifyToks: [Int32] = []
            for (idx, tok) in seqTokens.enumerated() {
                try fwdRef.forward(tokens: [tok], posOffset: prefix.count + idx,
                                   runHeadRows: 1, greedyHead: true, updateState: true)
                refVerifyToks.append(refTokPtr[0])
            }
            try fwdRef.forward(tokens: [followTok], posOffset: prefix.count + seqTokens.count,
                               runHeadRows: 1, greedyHead: false, updateState: false)
            let refFollowLogits = QWInk2TestRef.floatsFromF32Buffer(sessRef.logitsF32Buf, count: desc.vocab)
            try fwdRef.forward(tokens: [followTok], posOffset: prefix.count + seqTokens.count,
                               runHeadRows: 1, greedyHead: true, updateState: true)
            let refFollowTok = refTokPtr[0]

            let sessTree = try QWInkling2Session(
                device: hal.device, descriptor: desc, ink2: ink2, capacity: 128,
                kvModeRaw: kvMode, enableSpeculative: true, enableQWPScratch: true
            )
            let ds = try QWInkling2DSpark(
                hal: hal, mtpURL: qwpURL, descriptor: desc, ink2: ink2,
                config: dsCfg, weights: weights, capacity: 128
            )
            let fwdTree = try QWInkling2Forward(hal: hal, weights: weights, session: sessTree)
            try fwdTree.forward(tokens: prefix, posOffset: 0, runHeadRows: 0, greedyHead: false, updateState: true,
                                captureTapBuf: ds.tapBuf, tapVisits: dsCfg.targetLayerIds)
            try ds.ingestContextTaps(count: prefix.count, posOffset: 0, rowStride: prefix.count)

            try fwdTree.forward(tokens: treeTokens, posOffset: prefix.count, runHeadRows: 16, greedyHead: true,
                                updateState: true, captureTapBuf: ds.tapBuf, tapVisits: dsCfg.targetLayerIds,
                                recordVerifyConv: true, treePlan: tpl.plan)
            let treeTokPtr = sessTree.headTokensBuf.contents().assumingMemoryBound(to: Int32.self)
            for (step, node) in keptNodes.enumerated() {
                XCTAssertEqual(treeTokPtr[node], refVerifyToks[step],
                               "tree verify node \(node) mismatch kvMode=\(kvMode)")
            }

            _ = try ds.finishUnifiedSpeculativeRound(
                session: sessTree, oldPos: prefix.count, rowStride: 16, keptNodes: keptNodes,
                ingestTaps: false, nextAnchor: 0, nextDraft: .none
            )

            try fwdTree.forward(tokens: [followTok], posOffset: prefix.count + keptNodes.count,
                                runHeadRows: 1, greedyHead: false, updateState: false)
            let treeFollowLogits = QWInk2TestRef.floatsFromF32Buffer(sessTree.logitsF32Buf, count: desc.vocab)
            try fwdTree.forward(tokens: [followTok], posOffset: prefix.count + keptNodes.count,
                                runHeadRows: 1, greedyHead: true, updateState: true)
            XCTAssertEqual(treeTokPtr[0], refFollowTok, "post-compaction follow token mismatch kvMode=\(kvMode)")
            for v in 0..<desc.vocab {
                XCTAssertEqual(treeFollowLogits[v], refFollowLogits[v], accuracy: 1e-4,
                               "post-compaction logit mismatch v=\(v) kvMode=\(kvMode)")
            }
        }
    }

    func testRepetitionPenaltyWindowExcludesPromptAndTracksLoopTokens() {
        var recent: [Int] = []
        var active: [Int] = []
        let promptIDs = [200_005, 17, 42, 99, 200_005]
        for tok in promptIDs {
            QWInkling2Runtime.recordGeneratedTokenForPenalty(
                tok, vocab: 256_000, recentGenerated: &recent, activeLoopTokens: &active
            )
            XCTAssertEqual(active, [])
        }
        recent.removeAll()
        active.removeAll()
        for _ in 0..<4 {
            QWInkling2Runtime.recordGeneratedTokenForPenalty(
                7, vocab: 256_000, recentGenerated: &recent, activeLoopTokens: &active
            )
        }
        XCTAssertEqual(active, [7])

        recent.removeAll()
        active.removeAll()
        var refWindow: [Int] = []
        let totalSteps = QWInkling2Runtime.repetitionPenaltyWindow + 64
        for step in 0..<totalSteps {
            let tok = (step % 5 == 0) ? 42 : ((step % 2 == 0) ? 7 : 8)
            QWInkling2Runtime.recordGeneratedTokenForPenalty(
                tok, vocab: 256_000, recentGenerated: &recent, activeLoopTokens: &active
            )
            refWindow.append(tok)
            if refWindow.count > QWInkling2Runtime.repetitionPenaltyWindow {
                refWindow.removeFirst(refWindow.count - QWInkling2Runtime.repetitionPenaltyWindow)
            }
            XCTAssertLessThanOrEqual(recent.count, QWInkling2Runtime.repetitionPenaltyWindow)
            XCTAssertEqual(recent, refWindow)
            XCTAssertEqual(active, QWSampler.loopRepetitionTokens(in: refWindow))
        }
    }
}
