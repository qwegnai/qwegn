// QWDFlashInkling2UnitTests.swift — Synthetic GPU & config parity tests for
// Inkling-2 DFlash2 (decoupled layers, softcap modes, SafeTensors, BF16 tap packing).
import Foundation
import Metal
import XCTest
@testable import QWMTP
import QWAlloc
import QWCore
import QWGEMM
import QWHAL
import QWQuant

final class QWDFlashInkling2UnitTests: XCTestCase {
    private static func bf16Bits(_ value: Float) -> UInt16 {
        UInt16((value.bitPattern &+ 0x8000) >> 16)
    }

    private static func bf16ToF32(_ bits: UInt16) -> Float {
        Float(bitPattern: UInt32(bits) << 16)
    }

    private static func writeSafeTensorsFile(
        to url: URL,
        tensors: [(name: String, shape: [Int], data: [UInt16])]
    ) throws {
        var header: [String: Any] = [:]
        var body = Data()
        for t in tensors {
            let start = body.count
            let byteCount = t.data.count * 2
            t.data.withUnsafeBytes { raw in
                body.append(contentsOf: raw)
            }
            header[t.name] = [
                "dtype": "BF16",
                "shape": t.shape,
                "data_offsets": [start, start + byteCount],
            ]
        }
        var headerBytes = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        while (8 + headerBytes.count) % 8 != 0 {
            headerBytes.append(0x20)
        }
        var out = Data()
        var hdrLen = UInt64(headerBytes.count).littleEndian
        withUnsafeBytes(of: &hdrLen) { out.append(contentsOf: $0) }
        out.append(headerBytes)
        out.append(body)
        try out.write(to: url)
    }

    private static func makeSyntheticInkling2Target(
        vocab: Int,
        hidden: Int,
        numLayers: Int
    ) throws -> QWModelDescriptor {
        let cfg: [String: Any] = [
            "model_type": "inkling2_loop",
            "eos_token_id": 2,
            "text_config": [
                "model_type": "inkling2_loop_text",
                "hidden_size": hidden,
                "vocab_size": vocab,
                "intermediate_size": hidden * 2,
                "num_hidden_layers": numLayers,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 128,
                "model_max_length": 1024,
                "rms_norm_eps": 1e-6,
                "final_logit_softcapping": 30.0,
                "unpadded_vocab_size": vocab - 10,
                "local_layer_ids": [0, 1],
                "sliding_window_size": 256,
                "sconv_kernel_size": 4,
                "n_layers_per_attn_res_block": 2,
                "n_routed_experts": 4,
                "num_experts_per_tok": 2,
                "loop_config": [
                    "core_start": 1,
                    "core_end": 3,
                    "num_passes": 2,
                    "gate_rank": 16,
                    "is_varloop": false,
                ],
            ],
        ]
        return try QWModelDescriptor.decode(configJSON: JSONSerialization.data(withJSONObject: cfg))
    }

    func testConfigDecoupledTargetLayersSoftcapAndSafeTensorsValidation() throws {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwg_dflash_ink2_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }

        let H = 128, I = 256, V = 300, R = 256, convProj = 2 * 2 * (H / 16)
        let cfgJSON: [String: Any] = [
            "architectures": ["DFlash2DraftModel"],
            "model_type": "qwen3",
            "hidden_size": H,
            "intermediate_size": I,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 128,
            "vocab_size": V,
            "rms_norm_eps": 1e-6,
            "rope_parameters": ["rope_theta": 10000],
            "dflash_config": [
                "block_size": 8,
                "mask_token_id": 270,
                "target_layer_ids": [1, 3, 6],
                "conv_kernel_size": 2,
                "conv_group_size": 16,
                "selector_rank": R,
                "selector_top_k": 16,
                "final_logit_softcapping": 30.0,
                "selector_softcap_mode": "after_correction",
            ],
        ]
        try JSONSerialization.data(withJSONObject: cfgJSON).write(to: tmp.appendingPathComponent("config.json"))
        let ones128 = [UInt16](repeating: Self.bf16Bits(1.0), count: H)
        let zeros = { (n: Int) in [UInt16](repeating: Self.bf16Bits(0.02), count: n) }
        try Self.writeSafeTensorsFile(
            to: tmp.appendingPathComponent("model.safetensors"),
            tensors: [
                ("fc.weight", [H, 3 * H], zeros(H * 3 * H)),
                ("hidden_norm.weight", [H], ones128),
                ("norm.weight", [H], ones128),
                ("candidate_selector.hidden_projection.weight", [R, H], zeros(R * H)),
                ("candidate_selector.predecessor_codebook", [V, R], zeros(V * R)),
                ("candidate_selector.successor_codebook", [V, R], zeros(V * R)),
                ("layers.0.input_layernorm.weight", [H], ones128),
                ("layers.0.post_attention_layernorm.weight", [H], ones128),
                ("layers.0.self_attn.q_norm.weight", [128], ones128),
                ("layers.0.self_attn.k_norm.weight", [128], ones128),
                ("layers.0.attention_conv.base_kernel", [2, 2, H], zeros(2 * 2 * H)),
                ("layers.0.attention_conv.kernel_projection.weight", [convProj, H], zeros(convProj * H)),
                ("layers.0.mlp_conv.base_kernel", [2, 2, H], zeros(2 * 2 * H)),
                ("layers.0.mlp_conv.kernel_projection.weight", [convProj, H], zeros(convProj * H)),
                ("layers.0.self_attn.q_proj.weight", [128, H], zeros(128 * H)),
                ("layers.0.self_attn.k_proj.weight", [128, H], zeros(128 * H)),
                ("layers.0.self_attn.v_proj.weight", [128, H], zeros(128 * H)),
                ("layers.0.self_attn.o_proj.weight", [H, 128], zeros(H * 128)),
                ("layers.0.mlp.gate_proj.weight", [I, H], zeros(I * H)),
                ("layers.0.mlp.up_proj.weight", [I, H], zeros(I * H)),
                ("layers.0.mlp.down_proj.weight", [H, I], zeros(H * I)),
            ]
        )

        let target = try Self.makeSyntheticInkling2Target(vocab: V, hidden: H, numLayers: 4)
        let loaded = QWDFlashConfig.load(from: tmp, target: target)
        XCTAssertEqual(loaded.targetLayerIds, [1, 3, 6])
        XCTAssertEqual(loaded.ropeTheta, 10_000.0, accuracy: 0.1)
        XCTAssertEqual(loaded.finalLogitSoftcapping, 30.0, accuracy: 0.001)
        XCTAssertEqual(loaded.selectorSoftcapMode, .afterCorrection)
        try QWDFlashConfig.validateDrafter(targetDescriptor: target, packPath: tmp.path, drafts: 7)

        let hal = try QWHal(geometry: QWModelGeometry(descriptor: target), weightBasis: .rotated)
        let drafter = try QWDFlashDrafter(
            geometry: QWModelGeometry(descriptor: target), hal: hal,
            packPath: tmp.path, drafts: 7, sessionSlotCount: 1, headRotated: false
        )
        XCTAssertFalse(drafter.isRotatedDrafter)
        XCTAssertFalse(drafter.isRotatedHead)

        var seenVocabRows: Int? = nil
        let propRes = try drafter.propose(
            hal: hal, slot: drafter.sessionSlots[0], cacheOffset: 0,
            pendingToken: 5, draftCount: 2, draftVocab: 277,
            encodeHeadLogits: { _, _, _, _, yLogits32, rows, vocabRows in
                seenVocabRows = vocabRows
                let ptr = yLogits32.region.buffer.contents()
                    .advanced(by: yLogits32.region.base + yLogits32.offset)
                    .assumingMemoryBound(to: Float.self)
                for r in 0..<rows {
                    for v in 0..<vocabRows { ptr[r * vocabRows + v] = Float(v == 7 ? 5.0 : -1.0) }
                }
            },
            encodeAnchorMaskEmbed: { _, propX32, _, _, rows in
                let ptr = propX32.region.buffer.contents()
                    .advanced(by: propX32.region.base + propX32.offset)
                    .assumingMemoryBound(to: Float.self)
                for i in 0..<(rows * H) { ptr[i] = 0.01 }
            }
        )
        XCTAssertEqual(seenVocabRows, 277)
        XCTAssertEqual(propRes.drafts.count, 2)
    }

    func testSelectorGreedySoftcapModes() throws {
        let hal = try QWHal(geometry: .qwen38_27b, weightBasis: .rotated)
        let dev = hal.device
        let psoSel = try hal.pipeline("qwg_dflash_selector_greedy")

        let vocab = 4, topK = 2, D = 1
        var unary = [Float](repeating: 0, count: 16)
        unary[0] = 100.0
        unary[1] = 20.0
        var cands = [Int32](repeating: -1, count: 16)
        cands[0] = 1
        cands[1] = 2

        let hProjBuf = try XCTUnwrap(dev.makeBuffer(length: 256 * 4, options: .storageModeShared))
        let hPtr = hProjBuf.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<256 { hPtr[i] = 1.0 }

        var pred = [UInt16](repeating: 0, count: vocab * 256)
        var succ = [UInt16](repeating: 0, count: vocab * 256)
        for j in 0..<256 {
            pred[j] = Self.bf16Bits(1.0)
            succ[1 * 256 + j] = Self.bf16Bits(-70.0 / 256.0)
            succ[2 * 256 + j] = Self.bf16Bits(4.0 / 256.0)
        }
        let uBuf = try XCTUnwrap(dev.makeBuffer(bytes: unary, length: unary.count * 4, options: .storageModeShared))
        let cBuf = try XCTUnwrap(dev.makeBuffer(bytes: cands, length: cands.count * 4, options: .storageModeShared))
        let pBuf = try XCTUnwrap(dev.makeBuffer(bytes: pred, length: pred.count * 2, options: .storageModeShared))
        let sBuf = try XCTUnwrap(dev.makeBuffer(bytes: succ, length: succ.count * 2, options: .storageModeShared))
        let outBuf = try XCTUnwrap(dev.makeBuffer(length: 4, options: .storageModeShared))

        func run(mode: UInt32, cap: Float) throws -> Int32 {
            let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
            let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
            var anchor: Int32 = 0, dU = UInt32(D), tkU = UInt32(topK), vU = UInt32(vocab)
            var capF = cap, modeU = mode, rankU = UInt32(256)
            enc.setComputePipelineState(psoSel)
            enc.setBuffer(hProjBuf, offset: 0, index: 0)
            enc.setBuffer(uBuf, offset: 0, index: 1)
            enc.setBuffer(cBuf, offset: 0, index: 2)
            enc.setBuffer(pBuf, offset: 0, index: 3)
            enc.setBuffer(sBuf, offset: 0, index: 4)
            enc.setBuffer(outBuf, offset: 0, index: 5)
            enc.setBytes(&anchor, length: 4, index: 6)
            enc.setBytes(&dU, length: 4, index: 7)
            enc.setBytes(&tkU, length: 4, index: 8)
            enc.setBytes(&vU, length: 4, index: 9)
            enc.setBytes(&capF, length: 4, index: 10)
            enc.setBytes(&modeU, length: 4, index: 11)
            enc.setBytes(&rankU, length: 4, index: 12)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            return outBuf.contents().assumingMemoryBound(to: Int32.self)[0]
        }

        XCTAssertEqual(try run(mode: 1, cap: 30.0), 2)
        XCTAssertEqual(try run(mode: 2, cap: 30.0), 1)
    }

    func testPackBF16TapsUnrotatedAndRowMapParity() throws {
        let hal = try QWHal(geometry: .qwen38_27b, weightBasis: .rotated)
        let dev = hal.device
        let psoUnrot = try hal.pipeline("qwg_dflash_pack_bf16_taps", fcDrafterRotated: false, fcHeadRotated: false)

        let totalSlots = 2, srcRows = 4, H = 64
        var src = [UInt16](repeating: 0, count: totalSlots * srcRows * H)
        for s in 0..<totalSlots {
            for r in 0..<srcRows {
                for c in 0..<H {
                    src[(s * srcRows + r) * H + c] = Self.bf16Bits(Float(s * 10 + r) + Float(c) * 0.01)
                }
            }
        }
        let sBuf = try XCTUnwrap(dev.makeBuffer(bytes: src, length: src.count * 2, options: .storageModeShared))
        let outRows = 2
        let dBuf = try XCTUnwrap(dev.makeBuffer(length: outRows * totalSlots * H * 2, options: .storageModeShared))

        struct TapP {
            var srcPlaneStrideU4, srcRowOffset, dstRowOffset, hiddenU4, totalSlots, useRowMap: UInt32
            var rowMap: (
                UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32
            )
        }
        var p = TapP(
            srcPlaneStrideU4: UInt32((srcRows * H) / 8), srcRowOffset: 0, dstRowOffset: 0,
            hiddenU4: UInt32(H / 8), totalSlots: UInt32(totalSlots), useRowMap: 1,
            rowMap: (1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        )
        let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
        enc.setComputePipelineState(psoUnrot)
        enc.setBuffer(sBuf, offset: 0, index: 0)
        enc.setBuffer(dBuf, offset: 0, index: 1)
        enc.setBytes(&p, length: MemoryLayout<TapP>.stride, index: 2)
        enc.dispatchThreadgroups(MTLSize(width: outRows, height: totalSlots, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let dPtr = dBuf.contents().assumingMemoryBound(to: Float16.self)
        let picked = [1, 3]
        for (dstR, srcR) in picked.enumerated() {
            for s in 0..<totalSlots {
                for c in 0..<H {
                    let expected = Float16(Self.bf16ToF32(src[(s * srcRows + srcR) * H + c]))
                    let actual = dPtr[(dstR * totalSlots + s) * H + c]
                    XCTAssertEqual(actual, expected)
                }
            }
        }
    }

    func testBF16F16GEMVAndNAXParity() throws {
        let hal = try QWHal(geometry: .qwen38_27b, weightBasis: .rotated)
        let dev = hal.device
        let n = 128, k = 256
        var wArr = [UInt16](repeating: 0, count: n * k)
        for i in 0..<(n * k) { wArr[i] = Self.bf16Bits(Float((i % 29) - 14) * 0.05) }
        let wBuf = try XCTUnwrap(dev.makeBuffer(bytes: wArr, length: wArr.count * 2, options: .storageModeShared))
        let bank = QWGemmFP.BF16Bank(weights: QWFPBufferSlice(buffer: wBuf, offset: 0, bytes: wBuf.length), n: n, k: k)

        for m in [1, 8, 47] {
            var xArr = [Float16](repeating: 0, count: m * k)
            for i in 0..<(m * k) { xArr[i] = Float16(Float((i % 19) - 9) * 0.1) }
            let xBuf = try XCTUnwrap(dev.makeBuffer(bytes: xArr, length: xArr.count * 2, options: .storageModeShared))
            let yBuf = try XCTUnwrap(dev.makeBuffer(length: m * n * 2, options: .storageModeShared))

            let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
            let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
            try QWGemmFP.encodeBF16(
                enc, hal: hal, weight: bank,
                x: QWFPBufferSlice(buffer: xBuf, offset: 0, bytes: xBuf.length),
                y: QWFPBufferSlice(buffer: yBuf, offset: 0, bytes: yBuf.length),
                m: m, inDtype: .f16, outDtype: .f16
            )
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()

            let yPtr = yBuf.contents().assumingMemoryBound(to: Float16.self)
            var maxErr: Float = 0
            for r in 0..<m {
                for c in 0..<n {
                    var acc: Float = 0
                    for j in 0..<k {
                        acc += Self.bf16ToF32(wArr[c * k + j]) * Float(xArr[r * k + j])
                    }
                    let diff = abs(Float(yPtr[r * n + c]) - acc)
                    if diff > maxErr { maxErr = diff }
                }
            }
            XCTAssertLessThan(maxErr, 0.05, "m=\(m) maxErr=\(maxErr)")
        }
    }

    func testDFlashSiluMulRotatedVsUnrotatedParity() throws {
        let hal = try QWHal(geometry: .qwen38_27b, weightBasis: .rotated)
        let dev = hal.device
        let psoUnrot = try hal.pipeline("qwg_dflash_silu_mul_f16", fcDrafterRotated: false, fcHeadRotated: false)
        let psoRot = try hal.pipeline("qwg_dflash_silu_mul_f16", fcDrafterRotated: true, fcHeadRotated: true)

        let M = 3, I = 64
        var gu = [Float16](repeating: 0, count: M * 2 * I)
        for m in 0..<M {
            for c in 0..<I {
                let g = Float((m * 17 + c) % 23 - 11) * 0.125
                let u = Float((m * 13 + c) % 19 - 9) * 0.1
                gu[m * 2 * I + c] = Float16(g)
                gu[m * 2 * I + I + c] = Float16(u)
            }
        }
        let guBuf = try XCTUnwrap(dev.makeBuffer(bytes: gu, length: gu.count * 2, options: .storageModeShared))
        let outBuf = try XCTUnwrap(dev.makeBuffer(length: M * I * 2, options: .storageModeShared))

        func run(_ pso: any MTLComputePipelineState) throws -> [Float] {
            let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
            let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
            var mU = UInt32(M), iU = UInt32(I)
            enc.setComputePipelineState(pso)
            enc.setBuffer(guBuf, offset: 0, index: 0)
            enc.setBuffer(outBuf, offset: 0, index: 1)
            enc.setBytes(&mU, length: 4, index: 2)
            enc.setBytes(&iU, length: 4, index: 3)
            enc.dispatchThreadgroups(
                MTLSize(width: ((I + 7) / 8 + 127) / 128, height: M, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
            )
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            let ptr = outBuf.contents().assumingMemoryBound(to: Float16.self)
            return (0..<(M * I)).map { Float(ptr[$0]) }
        }

        let unrotActual = try run(psoUnrot)
        let rotActual = try run(psoRot)

        var unrotRef = [Float](repeating: 0, count: M * I)
        for m in 0..<M {
            for c in 0..<I {
                let g = Float(gu[m * 2 * I + c])
                let u = Float(gu[m * 2 * I + I + c])
                let v = (g / (1.0 + expf(-g))) * u
                unrotRef[m * I + c] = v
                XCTAssertEqual(unrotActual[m * I + c], Float(Float16(v)), accuracy: 0.01)
            }
        }

        let invSqrt32 = 1.0 / sqrtf(32.0)
        for m in 0..<M {
            for blk in stride(from: 0, to: I, by: 32) {
                var f = Array(unrotRef[(m * I + blk)..<(m * I + blk + 32)])
                var step = 1
                while step < 32 {
                    for base in stride(from: 0, to: 32, by: step * 2) {
                        for k in 0..<step {
                            let a = f[base + k], b = f[base + k + step]
                            f[base + k] = a + b
                            f[base + k + step] = a - b
                        }
                    }
                    step *= 2
                }
                for k in 0..<32 {
                    XCTAssertEqual(rotActual[m * I + blk + k], Float(Float16(f[k] * invSqrt32)), accuracy: 0.02)
                }
            }
        }
    }
}
