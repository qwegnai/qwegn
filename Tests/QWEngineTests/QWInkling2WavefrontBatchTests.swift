// QWInkling2WavefrontBatchTests.swift — B=2..4 continuous wavefront batch tests for Inkling-2.
// Covers admission/1M preservation, 0-ULP solo==batch parity, scheduler coalescing,
// sampling/cancellation/pollPending, warm prefix cache hits, and LSA gating.
import CryptoKit
import Foundation
import Metal
import XCTest
@testable import QWCore
@testable import QWEngine
@testable import QWHAL
import QWHead
import QWKV
import QWQuant

final class QWInkling2WavefrontBatchTests: XCTestCase {
    private enum PlaneSpec {
        case bf16(shape: [Int], values: [Float])
        case f32(shape: [Int], values: [Float])
        case hq4(shape: [Int], n: Int, k: Int, seed: UInt32, scale: Float, experts: Int? = nil, eHq4: Int? = nil)
    }

    private struct PackConfig {
        let vocab: Int = 128
        let hidden: Int = 128
        let numLayers: Int = 3
        let qHeads: Int = 4
        let kvHeads: Int = 2
        let headDim: Int = 128
        let dRel: Int = 8
        let relExtent: Int = 32
        let slidingWindow: Int = 16
        let localLayerIds: Set<Int> = [0]
        let loopCoreStart: Int = 1
        let loopCoreEnd: Int = 3
        let numPasses: Int = 2
        let gateRank: Int = 16
        let numExperts: Int = 4
        let topK: Int = 2
        let moeInter: Int = 64
    }

    private func requireMetal4Hardware() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.isMetal4TensorCapable == true,
            "Inkling-2 wavefront batch tests require bare-metal Apple Silicon Metal 4 hardware"
        )
    }

    private func makeTempDir(prefix: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(prefix)-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        return dir
    }

    private static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func detVec(_ count: Int, seedInit: UInt64, amp: Float, offset: Float = 0) -> [Float] {
        var rng = seedInit &+ 0x9E37_79B9_7F4A_7C15
        var out = [Float](repeating: 0, count: count)
        for i in 0..<count {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let u = Float((rng >> 33) & 0xFFFF) / 32768.0 - 1.0
            out[i] = offset + u * amp
        }
        return out
    }

    private static func makeIncumbentHQ4Data(n: Int, k: Int, seedInit: UInt32, scale: Float) -> Data {
        let groups = k / 32
        var seed = seedInit
        var out = Data(count: n * groups * 20)
        out.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            guard let ptr = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            for r in 0..<n {
                for g in 0..<groups {
                    let base = (r * groups + g) * 20
                    let scVal = Float(Float16(scale * (1.0 + 0.1 * Float((r + g) % 5))))
                    let sc = Float16(scVal).bitPattern
                    let bi = Float16(-1031.5 * scVal).bitPattern
                    ptr[base] = UInt8(sc & 0xFF)
                    ptr[base + 1] = UInt8(sc >> 8)
                    ptr[base + 2] = UInt8(bi & 0xFF)
                    ptr[base + 3] = UInt8(bi >> 8)
                    for i in 0..<16 {
                        seed = seed &* 1664525 &+ 1013904223
                        ptr[base + 4 + i] = UInt8((seed >> 24) & 0xFF)
                    }
                }
            }
        }
        return out
    }

    private static func writeQWP1(
        at rootDir: URL,
        provenance: [String: Any],
        planes: [(String, PlaneSpec)]
    ) throws {
        var file = Data("QWP1".utf8) + Data(repeating: 0, count: 60)
        var entries: [String: [String: Any]] = [:]
        func padTo(_ align: Int) {
            let rem = file.count % align
            if rem != 0 { file.append(Data(repeating: 0, count: align - rem)) }
        }
        for (name, spec) in planes {
            switch spec {
            case .bf16(let shape, let vals):
                padTo(256)
                let off = file.count
                var raw = Data(capacity: vals.count * 2)
                for bf in QWInk2TestRef.bf16Array(vals) {
                    var le = bf.littleEndian
                    withUnsafeBytes(of: &le) { (buf: UnsafeRawBufferPointer) in raw.append(contentsOf: buf) }
                }
                file.append(raw)
                entries[name] = ["shape": shape, "kind": "bf16", "raw_off": off]
            case .f32(let shape, let vals):
                padTo(256)
                let off = file.count
                var raw = Data(capacity: vals.count * 4)
                for val in vals {
                    var le = val.bitPattern.littleEndian
                    withUnsafeBytes(of: &le) { (buf: UnsafeRawBufferPointer) in raw.append(contentsOf: buf) }
                }
                file.append(raw)
                entries[name] = ["shape": shape, "kind": "f32", "raw_off": off]
            case .hq4(let shape, let n, let k, let seed, let scale, let experts, let eHq4):
                padTo(16_384)
                let off = file.count
                file.append(makeIncumbentHQ4Data(n: n, k: k, seedInit: seed, scale: scale))
                var meta: [String: Any] = ["shape": shape, "hq4_off": off]
                if let experts { meta["experts"] = experts }
                if let eHq4 { meta["e_hq4"] = eHq4 }
                entries[name] = meta
            }
        }
        let hdrData = try JSONSerialization.data(
            withJSONObject: ["version": 2, "provenance": provenance, "tensors": entries],
            options: [.sortedKeys]
        )
        var hdrLen = UInt64(hdrData.count).littleEndian
        withUnsafeBytes(of: &hdrLen) { (buf: UnsafeRawBufferPointer) in file.replaceSubrange(4..<12, with: buf) }
        file.append(hdrData)
        try file.write(to: rootDir.appendingPathComponent("model.qwp"))
    }

    private func buildSyntheticInkling2Pack(at rootDir: URL, maxPos: Int = 2048) throws {
        let c = PackConfig()
        let qDim = c.qHeads * c.headDim
        let kvDim = c.kvHeads * c.headDim
        let rDim = c.qHeads * c.dRel
        let msgLayers = c.loopCoreEnd - c.loopCoreStart
        let cfgDict: [String: Any] = [
            "architectures": ["Inkling2LoopForConditionalGeneration"],
            "model_type": "inkling2_loop",
            "eos_token_id": c.vocab - 1,
            "generation_config": ["eos_token_id": c.vocab - 1, "pad_token_id": 0],
            "text_config": [
                "model_type": "inkling2_loop_text",
                "vocab_size": c.vocab,
                "unpadded_vocab_size": c.vocab,
                "hidden_size": c.hidden,
                "num_hidden_layers": c.numLayers,
                "num_attention_heads": c.qHeads,
                "num_key_value_heads": c.kvHeads,
                "head_dim": c.headDim,
                "d_rel": c.dRel,
                "rel_extent": c.relExtent,
                "q_bias": false,
                "o_bias": false,
                "log_scaling_n_floor": 8,
                "log_scaling_alpha": 0.25,
                "rms_norm_eps": 1e-6,
                "use_value_norm_no_gamma": true,
                "qk_l2_clip_threshold": 22.627416997969522,
                "rel_l2_clip_threshold": 8.0,
                "local_layer_ids": Array(c.localLayerIds).sorted(),
                "sconv_kernel_size": 4,
                "final_logit_softcapping": 30.0,
                "swa_head_dim": c.headDim,
                "swa_num_attention_heads": c.qHeads,
                "swa_num_key_value_heads": c.kvHeads,
                "sliding_window_size": c.slidingWindow,
                "n_layers_per_attn_res_block": 1,
                "attn_res_normalize_values": true,
                "n_routed_experts": c.numExperts,
                "num_experts_per_tok": c.topK,
                "n_shared_experts": 0,
                "intermediate_size": c.moeInter,
                "moe_intermediate_size": c.moeInter,
                "mlp_gate_softcap": 4.0,
                "mlp_up_softcap": 25.0,
                "route_scale": 4.0,
                "use_gate_bias": true,
                "max_position_embeddings": maxPos,
                "rope_theta": 1_000_000.0,
                "loop_config": [
                    "core_start": c.loopCoreStart,
                    "core_end": c.loopCoreEnd,
                    "num_passes": c.numPasses,
                    "gate_rank": c.gateRank,
                    "is_varloop": false,
                ],
            ],
        ]
        let cfgJSON = try JSONSerialization.data(withJSONObject: cfgDict, options: [.sortedKeys])
        let descriptor = try QWModelDescriptor.decode(configJSON: cfgJSON)

        let assetsDir = rootDir.appendingPathComponent("assets")
        try FileManager.default.createDirectory(at: assetsDir, withIntermediateDirectories: true)
        let cfgURL = assetsDir.appendingPathComponent("config.json")
        let tokURL = assetsDir.appendingPathComponent("tokenizer.json")
        let tplURL = assetsDir.appendingPathComponent("chat_template.jinja")
        try cfgJSON.write(to: cfgURL)
        try Data("{\"version\":\"1.0\"}".utf8).write(to: tokURL)
        try Data("bos".utf8).write(to: tplURL)
        var manifest: [String: [String: Any]] = [:]
        for u in [cfgURL, tokURL, tplURL] {
            let d = try Data(contentsOf: u)
            manifest[u.lastPathComponent] = ["sha256": Self.sha256Hex(d), "bytes": d.count]
        }
        let mfData = try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
        try mfData.write(to: assetsDir.appendingPathComponent("manifest.json"))

        var planes: [(String, PlaneSpec)] = [
            ("model.embed_tokens.weight", .bf16(shape: [c.vocab, c.hidden], values: Self.detVec(c.vocab * c.hidden, seedInit: 11, amp: 0.25))),
            ("model.embed_norm.weight", .bf16(shape: [c.hidden], values: Self.detVec(c.hidden, seedInit: 12, amp: 0.05, offset: 1.0))),
            ("lm_head.weight", .hq4(shape: [c.vocab, c.hidden], n: c.vocab, k: c.hidden, seed: 13, scale: 0.02)),
            ("model.norm.weight", .bf16(shape: [c.hidden], values: Self.detVec(c.hidden, seedInit: 14, amp: 0.05, offset: 1.0))),
            ("model.final_input.query", .f32(shape: [1, c.hidden], values: Self.detVec(c.hidden, seedInit: 15, amp: 0.1))),
            ("model.message_queries", .f32(shape: [msgLayers, 4, c.hidden], values: Self.detVec(msgLayers * 4 * c.hidden, seedInit: 16, amp: 0.1))),
        ]

        let guPerExpBytes = (c.hidden / 32) * (2 * c.moeInter) * 20
        let downPerExpBytes = (c.moeInter / 32) * c.hidden * 20
        for l in 0..<c.numLayers {
            let lp = "model.layers.\(l)."
            let s = UInt64((l + 1) * 1000)
            let su = UInt32((l + 1) * 1000)
            let isSliding = c.localLayerIds.contains(l)
            if l > 0 {
                let qRows = isSliding ? 1 : 3
                planes.append(("\(lp)attention_input.query", .f32(shape: [qRows, c.hidden], values: Self.detVec(qRows * c.hidden, seedInit: s + 1, amp: 0.1))))
            }
            planes.append(("\(lp)mlp_input.query", .f32(shape: [1, c.hidden], values: Self.detVec(c.hidden, seedInit: s + 2, amp: 0.1))))
            if isSliding {
                planes.append(("\(lp)input_layernorm.weight", .bf16(shape: [c.hidden], values: Self.detVec(c.hidden, seedInit: s + 3, amp: 0.05, offset: 1.0))))
            } else {
                planes.append(("\(lp)q_input_layernorm.weight", .bf16(shape: [c.hidden], values: Self.detVec(c.hidden, seedInit: s + 4, amp: 0.05, offset: 1.0))))
                planes.append(("\(lp)k_input_layernorm.weight", .bf16(shape: [c.hidden], values: Self.detVec(c.hidden, seedInit: s + 5, amp: 0.05, offset: 1.0))))
                planes.append(("\(lp)v_input_layernorm.weight", .bf16(shape: [c.hidden], values: Self.detVec(c.hidden, seedInit: s + 6, amp: 0.05, offset: 1.0))))
            }
            planes.append(("\(lp)post_attention_layernorm.weight", .bf16(shape: [c.hidden], values: Self.detVec(c.hidden, seedInit: s + 7, amp: 0.05, offset: 1.0))))
            planes.append(("\(lp)attn_sconv.conv.weight", .bf16(shape: [c.hidden, 4], values: Self.detVec(c.hidden * 4, seedInit: s + 8, amp: 0.1))))
            planes.append(("\(lp)mlp_sconv.conv.weight", .bf16(shape: [c.hidden, 4], values: Self.detVec(c.hidden * 4, seedInit: s + 9, amp: 0.1))))

            planes.append(("\(lp)self_attn.q_proj.weight", .hq4(shape: [qDim, c.hidden], n: qDim, k: c.hidden, seed: su + 10, scale: 0.008)))
            planes.append(("\(lp)self_attn.gate_proj.weight", .hq4(shape: [qDim, c.hidden], n: qDim, k: c.hidden, seed: su + 11, scale: 0.008)))
            planes.append(("\(lp)self_attn.r_proj.weight", .hq4(shape: [rDim, c.hidden], n: rDim, k: c.hidden, seed: su + 12, scale: 0.008)))
            planes.append(("\(lp)self_attn.sink_proj.weight", .hq4(shape: [c.qHeads, c.hidden], n: c.qHeads, k: c.hidden, seed: su + 13, scale: 0.008)))
            planes.append(("\(lp)self_attn.k_proj.weight", .hq4(shape: [kvDim, c.hidden], n: kvDim, k: c.hidden, seed: su + 14, scale: 0.008)))
            planes.append(("\(lp)self_attn.v_proj.weight", .hq4(shape: [kvDim, c.hidden], n: kvDim, k: c.hidden, seed: su + 15, scale: 0.008)))
            planes.append(("\(lp)self_attn.o_proj.weight", .hq4(shape: [c.hidden, qDim], n: c.hidden, k: qDim, seed: su + 16, scale: 0.008)))
            planes.append(("\(lp)self_attn.q_scale", .bf16(shape: [qDim], values: Self.detVec(qDim, seedInit: s + 17, amp: 0.05, offset: 1.0))))
            planes.append(("\(lp)self_attn.rel_scale", .bf16(shape: [rDim], values: Self.detVec(rDim, seedInit: s + 18, amp: 0.05, offset: 1.0))))
            planes.append(("\(lp)self_attn.rel_proj", .bf16(shape: [c.dRel, c.relExtent], values: Self.detVec(c.dRel * c.relExtent, seedInit: s + 19, amp: 0.05))))

            if isSliding {
                planes.append(("\(lp)self_attn.k_sconv.conv.weight", .bf16(shape: [kvDim, 4], values: Self.detVec(kvDim * 4, seedInit: s + 20, amp: 0.1))))
                planes.append(("\(lp)self_attn.v_sconv.conv.weight", .bf16(shape: [kvDim, 4], values: Self.detVec(kvDim * 4, seedInit: s + 21, amp: 0.1))))
            } else {
                planes.append(("\(lp)self_attn.k_sconv.static_weight", .bf16(shape: [kvDim, 4], values: Self.detVec(kvDim * 4, seedInit: s + 22, amp: 0.1))))
                planes.append(("\(lp)self_attn.v_sconv.static_weight", .bf16(shape: [kvDim, 4], values: Self.detVec(kvDim * 4, seedInit: s + 23, amp: 0.1))))
                planes.append(("\(lp)self_attn.k_sconv.weight_proj.weight", .bf16(shape: [c.kvHeads * 4, c.hidden], values: Self.detVec(c.kvHeads * 4 * c.hidden, seedInit: s + 24, amp: 0.08))))
                planes.append(("\(lp)self_attn.v_sconv.weight_proj.weight", .bf16(shape: [c.kvHeads * 4, c.hidden], values: Self.detVec(c.kvHeads * 4 * c.hidden, seedInit: s + 25, amp: 0.08))))
            }

            if l >= c.loopCoreStart && l < c.loopCoreEnd {
                for branch in ["attn", "k", "v", "mlp"] {
                    let bSeed = s &+ UInt64(branch.utf8.reduce(0, { $0 + Int($1) }) * 17)
                    planes.append(("\(lp)controllers.\(branch).down_weight", .bf16(shape: [c.gateRank, 2 * c.hidden], values: Self.detVec(c.gateRank * 2 * c.hidden, seedInit: bSeed + 1, amp: 0.05))))
                    planes.append(("\(lp)controllers.\(branch).forget_weight", .bf16(shape: [c.hidden, c.gateRank], values: Self.detVec(c.hidden * c.gateRank, seedInit: bSeed + 2, amp: 0.05))))
                    planes.append(("\(lp)controllers.\(branch).gain_weight", .bf16(shape: [c.hidden, c.gateRank], values: Self.detVec(c.hidden * c.gateRank, seedInit: bSeed + 3, amp: 0.05))))
                }
            }

            planes.append(("\(lp)mlp.gate_weight", .bf16(shape: [c.numExperts, c.hidden], values: Self.detVec(c.numExperts * c.hidden, seedInit: s + 30, amp: 0.1))))
            planes.append(("\(lp)mlp.e_score_correction_bias", .f32(shape: [c.numExperts], values: Self.detVec(c.numExperts, seedInit: s + 31, amp: 0.02))))
            planes.append(("\(lp)mlp.experts.gate_up_proj", .hq4(shape: [c.numExperts, 2 * c.moeInter, c.hidden], n: c.numExperts * 2 * c.moeInter, k: c.hidden, seed: su + 32, scale: 0.008, experts: c.numExperts, eHq4: guPerExpBytes)))
            planes.append(("\(lp)mlp.experts.down_proj", .hq4(shape: [c.numExperts, c.hidden, c.moeInter], n: c.numExperts * c.hidden, k: c.moeInter, seed: su + 33, scale: 0.008, experts: c.numExperts, eHq4: downPerExpBytes)))
        }

        try Self.writeQWP1(
            at: rootDir,
            provenance: [
                "model_id": "inkling2_loop",
                "geometry_digest": descriptor.geometryDigestV2Hex,
                "source": "synthetic-wavefront-test",
                "sha256_source": "synthetic",
                "quant_recorded": "q4g",
                "assets_dir": "assets",
                "assets_manifest_sha256": Self.sha256Hex(mfData),
            ],
            planes: planes
        )
    }

    private final class LockedBox<T>: @unchecked Sendable {
        private let lock = NSLock()
        private var value: T
        init(_ initial: T) { self.value = initial }
        func withValue<R>(_ body: (inout T) -> R) -> R {
            lock.withLock { body(&value) }
        }
    }

    private func runConcurrentBurst(
        scheduler: QWBatchScheduler,
        requests: [QWGenerationRequest]
    ) throws -> [QWGenerationResult] {
        let count = requests.count
        let box = LockedBox([Result<QWGenerationResult, any Error>?](repeating: nil, count: count))
        let readySem = DispatchSemaphore(value: 0)
        let goGroup = DispatchGroup()
        goGroup.enter()
        let queue = DispatchQueue(label: "test.ink2.wavefront.burst", attributes: .concurrent)
        let workGroup = DispatchGroup()

        for i in 0..<count {
            let req = requests[i]
            QWBatchAdmissionTracker.beginPreparing()
            workGroup.enter()
            queue.async {
                readySem.signal()
                goGroup.wait()
                let outcome: Result<QWGenerationResult, any Error>
                do {
                    QWBatchAdmissionTracker.endPreparing()
                    outcome = .success(try scheduler.generate(request: req, onToken: nil))
                } catch {
                    outcome = .failure(error)
                }
                box.withValue { $0[i] = outcome }
                workGroup.leave()
            }
        }
        for _ in 0..<count { readySem.wait() }
        goGroup.leave()
        workGroup.wait()

        return try (0..<count).map { idx in
            guard let res = box.withValue({ $0[idx] }) else {
                throw QWEngineDriverError.state("missing result for burst slot \(idx)")
            }
            return try res.get()
        }
    }

    func testWaveSlotPlanningPreserves1MAdmissionAndBudget() throws {
        let basePlan = try QWCapacityPlan.plan(
            nativeWindow: 1_048_576,
            requested: 1_048_576,
            perTokenBytes: 8192,
            fixedBytes: 11_630_067_712,
            availableBytes: 96 << 30,
            yarnAllowed: false
        )
        let baseSizing = QWInkling2Runtime.AdmissionSizing(
            payRowBytes: 256,
            metaRowBytes: 64,
            kvRowTotal: 640,
            slidingVisits: 8,
            globalVisits: 12,
            slidingCap: 1088,
            convBytes: 64 << 20,
            weightsEst: 11 << 30,
            fixedScratchEst: 700 << 20,
            perTokBytes: 7680,
            availableBytes: 96 << 30,
            capacityPlan: basePlan,
            alignedCap: 1_048_576,
            finalCap: 1_048_576,
            totalPlannedBytes: 20_220_000_000
        )
        let plan1M = QWInkling2Runtime.planCompanionWaveSlots(sizing: baseSizing, maxConcurrentSessions: 4)
        XCTAssertEqual(plan1M.targetSlots, 1)
        XCTAssertEqual(plan1M.extraSlotBytes, 0)
        XCTAssertEqual(plan1M.waveSlotCap, 1_048_576)

        let normPlan = try QWCapacityPlan.plan(
            nativeWindow: 65_536,
            requested: 65_536,
            perTokenBytes: 7680,
            fixedBytes: 10 << 30,
            availableBytes: 64 << 30,
            yarnAllowed: false
        )
        let normalSizing = QWInkling2Runtime.AdmissionSizing(
            payRowBytes: 256,
            metaRowBytes: 64,
            kvRowTotal: 640,
            slidingVisits: 8,
            globalVisits: 12,
            slidingCap: 1088,
            convBytes: 64 << 20,
            weightsEst: 9 << 30,
            fixedScratchEst: 700 << 20,
            perTokBytes: 7680,
            availableBytes: 64 << 30,
            capacityPlan: normPlan,
            alignedCap: 65_536,
            finalCap: 65_536,
            totalPlannedBytes: 10 << 30
        )
        let planSolo = QWInkling2Runtime.planCompanionWaveSlots(sizing: normalSizing, maxConcurrentSessions: 1)
        XCTAssertEqual(planSolo.targetSlots, 1)
        XCTAssertEqual(planSolo.extraSlotBytes, 0)

        let planFull = QWInkling2Runtime.planCompanionWaveSlots(sizing: normalSizing, maxConcurrentSessions: 4)
        XCTAssertEqual(planFull.targetSlots, 4)
        XCTAssertEqual(planFull.waveSlotCap, 32_768)
        XCTAssertGreaterThan(planFull.extraSlotBytes, 0)

        let tightAvailable = normalSizing.totalPlannedBytes + planFull.extraSlotBytes + (4 << 30) + 1024
        let tightSizing = QWInkling2Runtime.AdmissionSizing(
            payRowBytes: normalSizing.payRowBytes,
            metaRowBytes: normalSizing.metaRowBytes,
            kvRowTotal: normalSizing.kvRowTotal,
            slidingVisits: normalSizing.slidingVisits,
            globalVisits: normalSizing.globalVisits,
            slidingCap: normalSizing.slidingCap,
            convBytes: normalSizing.convBytes,
            weightsEst: normalSizing.weightsEst,
            fixedScratchEst: normalSizing.fixedScratchEst,
            perTokBytes: normalSizing.perTokBytes,
            availableBytes: tightAvailable,
            capacityPlan: normPlan,
            alignedCap: normalSizing.alignedCap,
            finalCap: normalSizing.finalCap,
            totalPlannedBytes: normalSizing.totalPlannedBytes
        )
        let planTight = QWInkling2Runtime.planCompanionWaveSlots(sizing: tightSizing, maxConcurrentSessions: 4)
        XCTAssertEqual(planTight.targetSlots, 2)
        XCTAssertLessThanOrEqual(
            tightSizing.totalPlannedBytes + (planTight.targetSlots - 1) * planTight.extraSlotBytes + (4 << 30),
            tightSizing.availableBytes
        )
    }

    func testWavefrontParityAcrossKVModesAndBatchSizes() throws {
        try requireMetal4Hardware()
        let packDir = try makeTempDir(prefix: "qw-ink2-wave-pack")
        try buildSyntheticInkling2Pack(at: packDir)

        let prompts: [[Int]] = [
            [1, 5, 11, 23, 37, 41],
            (0..<17).map { 2 + ($0 * 7) % 120 },
            (0..<33).map { 3 + ($0 * 11) % 120 },
            (0..<50).map { 4 + ($0 * 13) % 120 },
        ]
        let requests = prompts.map {
            QWGenerationRequest(prompt: $0, maxNew: 12, sampling: .greedy)
        }

        for kvMode in [QWKVMode.f16, .q8, .q4] {
            let driver = try QWEngineDriver(
                packPath: packDir.path,
                maxTokens: 256,
                kvMode: kvMode,
                mtpDrafts: 0,
                prefillChunkSize: 64,
                maxConcurrentSessions: 4
            )
            let ink = try XCTUnwrap(driver.trunkExecutor as? QWInkling2Runtime)
            XCTAssertEqual(ink.maxBatchSlots, 4, "kvMode=\(kvMode)")
            XCTAssertEqual(ink.waveSlots.count, 4, "kvMode=\(kvMode)")
            XCTAssertEqual(driver.maxBatchSlots, 4, "kvMode=\(kvMode)")

            let scheduler = QWBatchScheduler(engine: driver)
            var soloExpected: [[Int]] = []
            for req in requests {
                let res = try scheduler.generate(request: req, onToken: nil)
                XCTAssertEqual(res.tokens.count, 12)
                soloExpected.append(res.tokens)
            }
            XCTAssertEqual(ink.steadyStateAllocations, 0, "solo steadyStateAllocations != 0 for \(kvMode)")

            for waveSize in [2, 3, 4] {
                let sliceReqs = Array(requests.prefix(waveSize))
                let burstResults = try runConcurrentBurst(scheduler: scheduler, requests: sliceReqs)
                for i in 0..<waveSize {
                    XCTAssertEqual(
                        burstResults[i].tokens,
                        soloExpected[i],
                        "0-ULP mismatch for kvMode=\(kvMode) waveSize=\(waveSize) idx=\(i)"
                    )
                }
                XCTAssertEqual(ink.steadyStateAllocations, 0, "batch steadyStateAllocations != 0 for \(kvMode)")
                if waveSize == 4 {
                    XCTAssertGreaterThanOrEqual(
                        scheduler.peakCoalescedWaveSize,
                        3,
                        "expected concurrent B=4 burst to coalesce >= 3 for \(kvMode)"
                    )
                }
            }
        }
    }

    func testWavefrontSamplingCancellationPrefixCacheAndLSAGating() throws {
        try requireMetal4Hardware()
        let packDir = try makeTempDir(prefix: "qw-ink2-wave-feat")
        let cacheDir = try makeTempDir(prefix: "qw-ink2-wave-cache")
        try buildSyntheticInkling2Pack(at: packDir, maxPos: 1536)

        let cacheCfg = QWCacheConfig(
            warmByteLimit: 64 << 20,
            ssdByteLimit: 0,
            entryLimit: 32,
            maxConcurrentSessions: 4,
            cacheDirectory: cacheDir,
            wipeOnStart: true
        )
        let driver = try QWEngineDriver(
            packPath: packDir.path,
            maxTokens: 256,
            kvMode: .f16,
            mtpDrafts: 0,
            prefillChunkSize: 64,
            prefixCacheConfig: cacheCfg,
            prefixCacheDirectory: cacheDir,
            maxConcurrentSessions: 4
        )
        let ink = try XCTUnwrap(driver.trunkExecutor as? QWInkling2Runtime)
        let scheduler = QWBatchScheduler(engine: driver)

        // 1. Seeded stochastic + repetitionPenalty 0-ULP solo vs B=3 parity.
        let stochSampler = QWSamplerConfig(
            temperature: 0.85,
            topK: 16,
            topP: 0.9,
            seed: 4242,
            uniformVersion: .upper24,
            repetitionPenalty: 1.25,
            hasExplicitSeed: true
        )
        let stochPrompts: [[Int]] = [
            [1, 9, 17, 25, 33, 41, 49, 57],
            (0..<19).map { 5 + ($0 * 5) % 120 },
            (0..<27).map { 7 + ($0 * 9) % 120 },
        ]
        let stochReqs = stochPrompts.map {
            QWGenerationRequest(prompt: $0, maxNew: 10, sampling: stochSampler)
        }
        let stochSolo = try stochReqs.map { try scheduler.generate(request: $0, onToken: nil).tokens }
        let stochBatch = try runConcurrentBurst(scheduler: scheduler, requests: stochReqs)
        for i in 0..<stochReqs.count {
            XCTAssertEqual(stochBatch[i].tokens, stochSolo[i], "stochastic+penalty mismatch at slot \(i)")
        }

        // 2. stopIDs + onToken cancellation + mid-wave pollPending admission.
        let baseStopPrompt = [2, 14, 26, 38, 50, 62]
        let greedyBase = try scheduler.generate(
            request: QWGenerationRequest(prompt: baseStopPrompt, maxNew: 8, sampling: .greedy),
            onToken: nil
        ).tokens
        let earlyStopID = greedyBase[2]
        let cancelPrompt = [3, 15, 27, 39, 51, 63, 75]
        let cancelExpected = try scheduler.generate(
            request: QWGenerationRequest(prompt: cancelPrompt, maxNew: 8, sampling: .greedy),
            onToken: nil
        ).tokens
        let thirdPrompt = [4, 16, 28, 40, 52, 64, 76, 88]
        let thirdExpected = try scheduler.generate(
            request: QWGenerationRequest(prompt: thirdPrompt, maxNew: 8, sampling: .greedy),
            onToken: nil
        ).tokens

        let cancelEmitted = LockedBox(0)
        let pollCalls = LockedBox(0)
        let thirdJob = QWBatchScheduledJob(
            request: QWGenerationRequest(prompt: thirdPrompt, maxNew: 8, sampling: .greedy),
            onToken: nil
        )
        let pendingMidWave = LockedBox([thirdJob])
        let stopReq = QWGenerationRequest(
            prompt: baseStopPrompt, maxNew: 8, stopIDs: [earlyStopID], sampling: .greedy
        )
        let soloStopRes = try scheduler.generate(request: stopReq, onToken: nil)
        XCTAssertEqual(soloStopRes.tokens, Array(greedyBase.prefix(2)))
        XCTAssertEqual(soloStopRes.stopTokenID, earlyStopID)
        let stopJob = QWBatchScheduledJob(request: stopReq, onToken: nil)
        let cancelJob = QWBatchScheduledJob(
            request: QWGenerationRequest(prompt: cancelPrompt, maxNew: 8, sampling: .greedy),
            onToken: { _ in
                cancelEmitted.withValue { count in
                    count += 1
                    return count < 2
                }
            }
        )
        ink.runScheduledBatch(
            initialJobs: [stopJob, cancelJob],
            pollPending: { maxAdmit in
                pollCalls.withValue { $0 += 1 }
                return pendingMidWave.withValue { queue in
                    var picked: [QWBatchScheduledJob] = []
                    while picked.count < maxAdmit, !queue.isEmpty {
                        picked.append(queue.removeFirst())
                    }
                    return picked
                }
            },
            requeueDelayed: { _ in XCTFail("unexpected requeue") }
        )
        let batchStopRes = try stopJob.wait()
        XCTAssertEqual(batchStopRes.tokens, soloStopRes.tokens)
        XCTAssertEqual(batchStopRes.stopTokenID, earlyStopID)
        XCTAssertEqual(try cancelJob.wait().tokens, Array(cancelExpected.prefix(2)))
        XCTAssertEqual(try thirdJob.wait().tokens, thirdExpected)
        XCTAssertGreaterThan(pollCalls.withValue { $0 }, 0)

        // 3. Warm prefix cache retention and hit across companion wavefront slots (`solo cached == batch cached`).
        let sharedPrefix = (0..<24).map { 10 + $0 }
        let suffixPrompts = [
            sharedPrefix + [101, 102, 103, 104, 105],
            sharedPrefix + [106, 107, 108, 109, 110],
            sharedPrefix + [111, 112, 113, 114, 115],
        ]
        _ = try scheduler.generate(
            request: QWGenerationRequest(
                prompt: sharedPrefix + [99],
                maxNew: 2,
                sampling: .greedy,
                usePrefixCache: true,
                retainPrefixLength: sharedPrefix.count,
                cacheSalt: "solo-warm"
            ),
            onToken: nil
        )
        let soloCached = try suffixPrompts.map {
            try scheduler.generate(
                request: QWGenerationRequest(
                    prompt: $0,
                    maxNew: 8,
                    sampling: .greedy,
                    usePrefixCache: true,
                    retainPrefixLength: sharedPrefix.count,
                    cacheSalt: "solo-warm"
                ),
                onToken: nil
            )
        }
        for i in 0..<suffixPrompts.count {
            XCTAssertEqual(soloCached[i].cache?.hit, true, "expected solo cache hit at \(i)")
            XCTAssertEqual(soloCached[i].cache?.matchedTokens, 24, "expected solo matchedTokens=24 at \(i)")
        }

        _ = try scheduler.generate(
            request: QWGenerationRequest(
                prompt: sharedPrefix + [99],
                maxNew: 2,
                sampling: .greedy,
                usePrefixCache: true,
                retainPrefixLength: sharedPrefix.count,
                cacheSalt: "batch-warm"
            ),
            onToken: nil
        )
        let cacheHits = LockedBox(0)
        driver.onCacheEvent = { ev in
            if case .hit(_, let matched, _) = ev, matched >= 24 {
                cacheHits.withValue { $0 += 1 }
            }
        }
        let cachedBurst = try runConcurrentBurst(
            scheduler: scheduler,
            requests: suffixPrompts.map {
                QWGenerationRequest(
                    prompt: $0,
                    maxNew: 8,
                    sampling: .greedy,
                    usePrefixCache: true,
                    cacheSalt: "batch-warm"
                )
            }
        )
        for i in 0..<suffixPrompts.count {
            XCTAssertEqual(cachedBurst[i].tokens, soloCached[i].tokens, "cached solo vs batch mismatch at \(i)")
            XCTAssertEqual(cachedBurst[i].cache?.hit, true, "expected cache hit on slot \(i)")
            XCTAssertEqual(cachedBurst[i].cache?.matchedTokens, 24, "expected matchedTokens=24 on slot \(i)")
        }
        XCTAssertGreaterThanOrEqual(cacheHits.withValue { $0 }, 3)
        XCTAssertEqual(ink.steadyStateAllocations, 0)

        // 4. Single-concurrency config disables companion slots and preserves solo-only budget.
        let soloDriver = try QWEngineDriver(
            packPath: packDir.path,
            maxTokens: 256,
            kvMode: .f16,
            mtpDrafts: 0,
            prefillChunkSize: 64,
            maxConcurrentSessions: 1
        )
        XCTAssertEqual(soloDriver.trunkExecutor.maxBatchSlots, 1)
        XCTAssertEqual(soloDriver.maxBatchSlots, 1)

        // 5. LSA eligibility gate: requests beyond denseBypassThreshold are requeued/bypassed from wavefront.
        let lsaCfg = QWLSAConfig(mode: .svd, budget: 512)
        let lsaDriver = try QWEngineDriver(
            packPath: packDir.path,
            maxTokens: 1536,
            kvMode: .f16,
            mtpDrafts: 0,
            prefillChunkSize: 64,
            lsaConfig: lsaCfg,
            maxConcurrentSessions: 3
        )
        let lsaInk = try XCTUnwrap(lsaDriver.trunkExecutor as? QWInkling2Runtime)
        let threshold = lsaCfg.denseBypassThreshold
        let shortReq1 = QWGenerationRequest(prompt: [1, 2, 3, 4, 5, 6], maxNew: 4, sampling: .greedy)
        let shortReq2 = QWGenerationRequest(prompt: [2, 4, 6, 8, 10, 12], maxNew: 4, sampling: .greedy)
        let overReq = QWGenerationRequest(
            prompt: Array(repeating: 7, count: threshold),
            maxNew: 4,
            sampling: .greedy
        )
        let shortExpected1 = try lsaDriver.generate(request: shortReq1, onToken: nil).tokens
        let shortExpected2 = try lsaDriver.generate(request: shortReq2, onToken: nil).tokens
        let job1 = QWBatchScheduledJob(request: shortReq1, onToken: nil)
        let job2 = QWBatchScheduledJob(request: shortReq2, onToken: nil)
        let overJob = QWBatchScheduledJob(request: overReq, onToken: nil)
        let pendingOver = LockedBox([overJob])
        let requeuedCount = LockedBox(0)
        lsaInk.runScheduledBatch(
            initialJobs: [job1, job2],
            pollPending: { maxAdmit in
                pendingOver.withValue { q in
                    var out: [QWBatchScheduledJob] = []
                    while out.count < maxAdmit, !q.isEmpty { out.append(q.removeFirst()) }
                    return out
                }
            },
            requeueDelayed: { delayed in
                requeuedCount.withValue { $0 += delayed.count }
            }
        )
        XCTAssertEqual(requeuedCount.withValue { $0 }, 1)
        XCTAssertEqual(try job1.wait().tokens, shortExpected1)
        XCTAssertEqual(try job2.wait().tokens, shortExpected2)
    }
}
