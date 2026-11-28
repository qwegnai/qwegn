// QWKDAAndMLAGeneralizationTests.swift — GPU parity for generalized KDA, Direct-Q MLA, HCA, MBSA, and Vision.
import Foundation
import Metal
import XCTest
@testable import QWAlloc
@testable import QWAttn
@testable import QWCore
@testable import QWGDN
@testable import QWGEMM
@testable import QWHAL
@testable import QWKV
@testable import QWMM
@testable import QWQuant
@testable import QWVision

final class QWKDAAndMLAGeneralizationTests: XCTestCase {
    private struct LCG {
        var state: UInt64
        mutating func nextUniform() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let u = Double((state >> 11) & 0x1FFFFF) / Double(1 << 21)
            return Float(u * 2.0 - 1.0)
        }
    }

    private func requireGPU() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.supportsSimdgroupMMA == true,
            "Requires bare-metal Apple Silicon Metal GPU"
        )
    }

    private func makeF16Buf(_ device: any MTLDevice, _ vals: [Float]) -> any MTLBuffer {
        let bits = vals.map { Float16($0).bitPattern }
        let buf = device.makeBuffer(length: max(256, bits.count * 2), options: .storageModeShared)!
        bits.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            if let src = raw.baseAddress, !raw.isEmpty { buf.contents().copyMemory(from: src, byteCount: raw.count) }
        }
        return buf
    }

    private func makeF32Buf(_ device: any MTLDevice, _ vals: [Float]) -> any MTLBuffer {
        let buf = device.makeBuffer(length: max(256, vals.count * 4), options: .storageModeShared)!
        vals.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            if let src = raw.baseAddress, !raw.isEmpty { buf.contents().copyMemory(from: src, byteCount: raw.count) }
        }
        return buf
    }

    private func tv(_ buf: any MTLBuffer, _ count: Int, _ dtype: QWDType) -> QWTensorValue {
        QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: buf.length),
                      offset: 0, shape: QWTensorShape([count], dtype))
    }

    private func readF16(_ tv: QWTensorValue, count: Int) -> [Float] {
        let ptr = tv.region.pointer.advanced(by: tv.offset).assumingMemoryBound(to: Float16.self)
        return (0..<count).map { Float(ptr[$0]) }
    }

    private func readF32(_ tv: QWTensorValue, count: Int) -> [Float] {
        let ptr = tv.region.pointer.advanced(by: tv.offset).assumingMemoryBound(to: Float.self)
        return (0..<count).map { ptr[$0] }
    }

    private func makeSHVQBank(bits: Int, group: Int, n: Int, k: Int, device: any MTLDevice) -> any MTLBuffer {
        let groups = k / group
        let gBytes = QWSHVQBlock.bytes(bits: bits, group: group)
        var seed: UInt32 = 0x5A17_9B31
        var bank = Data(count: max(256, n * groups * gBytes))
        bank.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            guard let dst = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            for r in 0..<n {
                for g in 0..<groups {
                    let off = (r * groups + g) * gBytes
                    let sc = Float16(0.03125 * (1.0 + 0.1 * Float((r + g) % 5))).bitPattern
                    dst[off] = UInt8(sc & 0xFF)
                    dst[off + 1] = UInt8(sc >> 8)
                    for b in 2..<gBytes {
                        seed = seed &* 1664525 &+ 1013904223
                        dst[off + b] = UInt8((seed >> 16) & 0xFF)
                    }
                }
            }
        }
        let buf = device.makeBuffer(length: bank.count, options: .storageModeShared)!
        bank.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            if let src = raw.baseAddress { buf.contents().copyMemory(from: src, byteCount: bank.count) }
        }
        return buf
    }

    // MARK: - 1. KDA: .fullRank, .lowRank(64), and o_norm .sigmoid vs .silu

    func testKDAGatesFullRankDirectLowRank64AndNormActivation() throws {
        try requireGPU()
        let fullCfg: [String: Any] = [
            "model_type": "bailing_moe_v3", "hidden_size": 256, "intermediate_size": 512,
            "vocab_size": 1024, "num_hidden_layers": 2,
            "layer_types": ["linear_attention", "full_attention"], "full_attention_interval": 2,
            "num_attention_heads": 4, "num_key_value_heads": 4, "head_dim": 128,
            "linear_num_key_heads": 4, "linear_num_value_heads": 4, "linear_key_head_dim": 128,
            "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
            "no_kda_lora": true, "kda_lower_bound": -5.0, "partial_rotary_factor": 0.5,
            "rms_norm_eps": 1e-6, "max_position_embeddings": 2048, "eos_token_id": 2
        ]
        let fullGeom = QWModelGeometry(descriptor: try QWModelDescriptor.decode(
            configJSON: JSONSerialization.data(withJSONObject: fullCfg)))
        let halFull = try QWHal(geometry: fullGeom)
        let decFull = try QWGDNDecode(hal: halFull)

        let vh = 4, hd = 128, zCh = vh * hd, H = 256, rowsDirect = 6
        var rng = LCG(state: 0x1001)
        let fgbDirect = (0..<(rowsDirect * 2 * zCh)).map { _ in Float(Float16(rng.nextUniform() * 0.4)) }
        let ln1Direct = (0..<(rowsDirect * H)).map { _ in Float(Float16(rng.nextUniform() * 0.3)) }
        let bProjDirect = (0..<(vh * H)).map { _ in Float(Float16(rng.nextUniform() * 0.25)) }
        let aLogVals = (0..<vh).map { Float($0) * 0.15 - 0.6 }
        let dtBiasVals = (0..<zCh).map { sin(Float($0) * 0.03) * 0.2 }

        let bBuf = halFull.device.makeBuffer(length: max(256, rowsDirect * vh * 4), options: .storageModeShared)!
        let gBuf = halFull.device.makeBuffer(length: max(256, rowsDirect * zCh * 4), options: .storageModeShared)!
        let zBuf = halFull.device.makeBuffer(length: max(256, rowsDirect * zCh * 2), options: .storageModeShared)!
        guard let (cb1, enc1) = halFull.begin() else { XCTFail("hal.begin"); return }
        try decFull.encodeKDAGatesDirect(
            enc1,
            fgb: tv(makeF16Buf(halFull.device, fgbDirect), rowsDirect * 2 * zCh, .f16),
            ln1: tv(makeF16Buf(halFull.device, ln1Direct), rowsDirect * H, .f16),
            bProjF16: tv(makeF16Buf(halFull.device, bProjDirect), vh * H, .f16),
            aLog: tv(makeF32Buf(halFull.device, aLogVals), vh, .f32),
            dtBias: tv(makeF32Buf(halFull.device, dtBiasVals), zCh, .f32),
            betas: tv(bBuf, rowsDirect * vh, .f32),
            glog: tv(gBuf, rowsDirect * zCh, .f32),
            zOut: tv(zBuf, rowsDirect * zCh, .f16),
            rows: rowsDirect)
        halFull.end(cb1, enc1)
        cb1.waitUntilCompleted()
        XCTAssertNil(cb1.error)

        let gotBetas = readF32(tv(bBuf, rowsDirect * vh, .f32), count: rowsDirect * vh)
        let gotGlog = readF32(tv(gBuf, rowsDirect * zCh, .f32), count: rowsDirect * zCh)
        let gotZ = readF16(tv(zBuf, rowsDirect * zCh, .f16), count: rowsDirect * zCh)
        for t in 0..<rowsDirect {
            for h in 0..<vh {
                var dot: Float = 0
                for k in 0..<H { dot += ln1Direct[t * H + k] * bProjDirect[h * H + k] }
                XCTAssertEqual(gotBetas[t * vh + h], 1.0 / (1.0 + exp(-dot)), accuracy: 5e-3)
            }
            for c in 0..<zCh {
                let rate = exp(aLogVals[c / hd])
                let gate = -5.0 / (1.0 + exp(-rate * (fgbDirect[t * 2 * zCh + c] + dtBiasVals[c])))
                XCTAssertEqual(gotGlog[t * zCh + c], exp(gate), accuracy: 1e-4)
                XCTAssertEqual(gotZ[t * zCh + c], fgbDirect[t * 2 * zCh + zCh + c], accuracy: 1e-3)
            }
        }

        // Low-rank KDA (lr=64) scalar (rows=4) and MMA (rows=36)
        let lr = 64, lrRows = 36, fgbStride = 2 * lr + vh
        let fgbLR = (0..<(lrRows * fgbStride)).map { _ in Float(Float16(rng.nextUniform() * 0.3)) }
        let fBLR = (0..<(zCh * lr)).map { _ in Float(Float16(rng.nextUniform() * 0.15)) }
        let gBLR = (0..<(zCh * lr)).map { _ in Float(Float16(rng.nextUniform() * 0.15)) }
        for (rows, mma) in [(4, false), (36, true)] {
            decFull.kdaGatesMMAEnabled = mma
            let bO = halFull.device.makeBuffer(length: max(256, rows * vh * 4), options: .storageModeShared)!
            let gO = halFull.device.makeBuffer(length: max(256, rows * zCh * 4), options: .storageModeShared)!
            let zO = halFull.device.makeBuffer(length: max(256, rows * zCh * 2), options: .storageModeShared)!
            guard let (cb, enc) = halFull.begin() else { XCTFail("hal.begin"); return }
            try decFull.encodeKDAGates(
                enc,
                fgb: tv(makeF16Buf(halFull.device, Array(fgbLR.prefix(rows * fgbStride))), rows * fgbStride, .f16),
                fB: tv(makeF16Buf(halFull.device, fBLR), zCh * lr, .f16),
                gB: tv(makeF16Buf(halFull.device, gBLR), zCh * lr, .f16),
                aLog: tv(makeF32Buf(halFull.device, aLogVals), vh, .f32),
                dtBias: tv(makeF32Buf(halFull.device, dtBiasVals), zCh, .f32),
                betas: tv(bO, rows * vh, .f32),
                glog: tv(gO, rows * zCh, .f32),
                zOut: tv(zO, rows * zCh, .f16),
                rows: rows, lowRankDim: lr)
            halFull.end(cb, enc)
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error)
            let gOut = readF32(tv(gO, rows * zCh, .f32), count: rows * zCh)
            for t in [0, rows - 1] {
                for c in 0..<zCh {
                    let h = c / hd, colInH = c % hd
                    var pf: Float = 0
                    for k in 0..<lr { pf += fgbLR[t * fgbStride + k] * fBLR[(h * hd + colInH) * lr + k] }
                    let refG = exp(-5.0 / (1.0 + exp(-exp(aLogVals[h]) * (pf + dtBiasVals[c]))))
                    XCTAssertEqual(gOut[t * zCh + c], refG, accuracy: 8e-3, "mma=\(mma) t=\(t) c=\(c)")
                }
            }
        }

        // o_norm activation check: sigmoid (KDA) vs silu (standard GDN without KDA)
        var gdnCfg = fullCfg
        gdnCfg.removeValue(forKey: "no_kda_lora")
        gdnCfg["kda_lower_bound"] = nil
        gdnCfg["model_type"] = "qwen3_5"
        gdnCfg["rope_theta"] = 10000.0
        gdnCfg["num_hidden_layers"] = 2
        gdnCfg["full_attention_interval"] = 2
        gdnCfg["layer_types"] = ["linear_attention", "full_attention"]
        let gdnGeom = QWModelGeometry(descriptor: try QWModelDescriptor.decode(
            configJSON: JSONSerialization.data(withJSONObject: gdnCfg)))
        let halSilu = try QWHal(geometry: gdnGeom)
        let decSilu = try QWGDNDecode(hal: halSilu)
        let yIn = (0..<(3 * zCh)).map { _ in Float(Float16(rng.nextUniform() * 0.2)) }
        let zIn = (0..<zCh).map { _ in Float(Float16(1.2)) }
        let wIn = [Float](repeating: 1.0, count: hd)
        func runNormAct(_ dec: QWGDNDecode, _ hal: QWHal) throws -> [Float] {
            let st = hal.device.makeBuffer(length: vh * hd * hd * 4, options: .storageModeShared)!
            memset(st.contents(), 0, st.length)
            let oBuf = hal.device.makeBuffer(length: max(256, zCh * 2), options: .storageModeShared)!
            let oScr = hal.device.makeBuffer(length: max(256, zCh * 4), options: .storageModeShared)!
            let gCh = hal.geometry.gdnGlogChannels
            guard let (cb, enc) = hal.begin() else { throw NSError(domain: "hal", code: 1) }
            do {
                try dec.encode(
                    enc, y: tv(makeF16Buf(hal.device, yIn), 3 * zCh, .f16),
                    betas: tv(makeF32Buf(hal.device, [Float](repeating: 0.8, count: vh)), vh, .f32),
                    glog: tv(makeF32Buf(hal.device, [Float](repeating: 0.9, count: gCh)), gCh, .f32),
                    state: tv(st, vh * hd * hd, .f32), z: tv(makeF16Buf(hal.device, zIn), zCh, .f16),
                    w: tv(makeF16Buf(hal.device, wIn), hd, .f16), o: tv(oBuf, zCh, .f16),
                    oScratch: tv(oScr, zCh, .f32))
            } catch {
                enc.endEncoding()
                throw error
            }
            hal.end(cb, enc)
            cb.waitUntilCompleted()
            return readF16(tv(oBuf, zCh, .f16), count: zCh)
        }
        let sigO = try runNormAct(decFull, halFull)
        let silO = try runNormAct(decSilu, halSilu)
        // With z = 1.2, silu(z)/sigmoid(z) == z == 1.2
        var maxRatioErr: Float = 0
        for i in 0..<zCh where abs(sigO[i]) > 0.05 {
            maxRatioErr = max(maxRatioErr, abs((silO[i] / sigO[i]) - 1.2))
        }
        XCTAssertLessThan(maxRatioErr, 0.05)
    }

    // MARK: - 2. Direct-Q Dense-Streaming MLA (>2560), 3D MRoPE, and Head-Wise o_gate

    func testDirectQDenseStreamMLA3072AndHeadWiseGateAndMRoPE() throws {
        try requireGPU()
        let cfgObj: [String: Any] = [
            "model_type": "bailing_moe_v3", "hidden_size": 256, "intermediate_size": 512,
            "vocab_size": 1024, "num_hidden_layers": 1, "layer_types": ["full_attention"],
            "num_attention_heads": 16, "num_key_value_heads": 16, "head_dim": 128,
            "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
            "kv_lora_rank": 512, "gated_attention_proj_granularity_type": "head_wise",
            "rope_interleave": true, "mrope_section": [8, 12, 12],
            "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "max_position_embeddings": 4096,
            "eos_token_id": 2
        ]
        let geo = QWModelGeometry(descriptor: try QWModelDescriptor.decode(
            configJSON: JSONSerialization.data(withJSONObject: cfgObj)))
        let hal = try QWHal(geometry: geo)

        let kvRows = geo.qHeads * (geo.qkNopeHeadDim + geo.vHeadDim)
        let bankBuf = makeSHVQBank(bits: 4, group: 32, n: kvRows, k: geo.kvLoraRank, device: hal.device)
        let bankView = try XCTUnwrap(QWWeightView(
            blocks: tv(bankBuf, bankBuf.length / 2, .f16), n: kvRows, k: geo.kvLoraRank,
            bits: 4, groupSize: 32, isRotated: true, isSHVQ: true))
        var rng = LCG(state: 0xBEEF)
        let ln1Vals = (0..<(32 * geo.hidden)).map { _ in Float(Float16(rng.nextUniform() * 0.25)) }
        let ln1T = tv(makeF16Buf(hal.device, ln1Vals), 32 * geo.hidden, .f16)
        let gProjVals = (0..<(geo.qHeads * geo.hidden)).map { _ in Float(Float16(rng.nextUniform() * 0.15)) }
        let mlaW = QWMLAAttnOperator.LayerWeights(
            qANormW: tv(makeF16Buf(hal.device, [1.0]), 1, .f16),
            kvANormW: tv(makeF16Buf(hal.device, [Float](repeating: 1.0, count: geo.kvLoraRank)), geo.kvLoraRank, .f16),
            kNormW: tv(makeF16Buf(hal.device, [1.0]), 1, .f16),
            kNormB: tv(makeF16Buf(hal.device, [0.0]), 1, .f16),
            wGateWeights: tv(makeF16Buf(hal.device, [0.0]), 1, .f16),
            ape: nil, qBProj: bankView, kvBProj: bankView,
            oGateF16: tv(makeF16Buf(hal.device, gProjVals), geo.qHeads * geo.hidden, .f16))

        func seedInputs(_ op: QWMLAAttnOperator, rows: Int, salt: Int) {
            let latPtr = op.region.pointer.advanced(by: op.mlaLatentScratch.offset).assumingMemoryBound(to: UInt16.self)
            for i in 0..<(rows * op.latentStride) {
                latPtr[i] = Float16(sin(Float((i + salt) % 241) * 0.04) * 0.25).bitPattern
            }
            let qPtr = op.region.pointer.advanced(by: op.mlaQFullScratch.offset).assumingMemoryBound(to: UInt16.self)
            for i in 0..<(rows * op.qFullStride) {
                qPtr[i] = Float16(cos(Float((i + salt) % 239) * 0.04) * 0.2).bitPattern
            }
        }

        func runMode(_ mode: QWKVMode, mma: Bool, hShift: Int32 = 0) throws -> ([Float], [Float], [Float]) {
            let pageBytes = QWKVCache.pageBytes(mode: mode, geometry: geo)
            let pagesPerPool = (4096 / 256) + 4
            let tables = QWKVCache.tableSlotBytes * 2 * 1 * QWKVCache.maxSeqs
            let arena = try QWArena(hal: hal, weights: 0, activations: 0,
                                    kv: pageBytes * pagesPerPool * 2 + tables, gdnStates: 4096)
            let cache = try QWKVCache(hal: hal, arena: arena, layers: [0], mode: mode, geometry: geo)
            let seq = try cache.allocate(promptTokens: 3200)
            let kvViews = try cache.layerViews(seq, layer: 0)
            let op = try QWMLAAttnOperator(hal: hal, geometry: geo, numAttnLayers: 1, maxContext: 4096, maxPrefillChunk: 32)
            XCTAssertFalse(op.hasIndexer)
            XCTAssertTrue(op.isDirectQ)
            XCTAssertEqual(op.topKPools, 0)
            XCTAssertEqual(op.outputGateMode, .headWiseFromHidden)
            XCTAssertEqual(op.mropeLayout, .bailingVideoRope)
            op.sparseAttnMMAEnabled = mma

            if let (cbSeed, encSeed) = hal.begin() {
                for p0 in stride(from: 0, to: 3040, by: 32) {
                    seedInputs(op, rows: 32, salt: p0)
                    op.encodeLatentPrepAndPool(encSeed, ln1: ln1T, mlaW: mlaW, kv: kvViews,
                                               fullAttnLayer: 0, pos0: p0, rows: 32)
                }
                hal.end(cbSeed, encSeed)
                cbSeed.waitUntilCompleted()
                XCTAssertNil(cbSeed.error)
            }

            let mrope32: [Int32] = (0..<32).map { Int32(3040 + $0) }
                + (0..<32).map { Int32(3040 + $0) + hShift }
                + (0..<32).map { Int32(3040 + $0) }
            let prefillOut = tv(makeF16Buf(hal.device, [Float](repeating: 0, count: 32 * geo.qHeads * geo.vHeadDim)),
                                32 * geo.qHeads * geo.vHeadDim, .f16)
            seedInputs(op, rows: 32, salt: 3040)
            let (cbP, encP) = try XCTUnwrap(hal.begin())
            op.encodeLatentPrepAndPool(encP, ln1: ln1T, mlaW: mlaW, kv: kvViews,
                                       fullAttnLayer: 0, pos0: 3040, rows: 32, mropePositions: mrope32)
            op.encodeScoreSelectAbsorbAttendAndProjUV(encP, mlaW: mlaW, kv: kvViews, attOut: prefillOut,
                                                      fullAttnLayer: 0, pos0: 3040, rows: 32,
                                                      ln1: ln1T, mropePositions: mrope32)
            hal.end(cbP, encP)
            cbP.waitUntilCompleted()
            XCTAssertNil(cbP.error)

            let kPeTail = op.layers[0].kPeCache!
            let kPeSnap = readF16(kPeTail.slice(byteOffset: 3040 * op.ropeDim * 2, count: op.ropeDim, dtype: .f16),
                                  count: op.ropeDim)

            let decodeOut = tv(makeF16Buf(hal.device, [Float](repeating: 0, count: geo.qHeads * geo.vHeadDim)),
                               geo.qHeads * geo.vHeadDim, .f16)
            seedInputs(op, rows: 1, salt: 3072)
            let (cbD, encD) = try XCTUnwrap(hal.begin())
            op.encodeLatentPrepAndPool(encD, ln1: ln1T, mlaW: mlaW, kv: kvViews, fullAttnLayer: 0, pos0: 3072, rows: 1)
            op.encodeScoreSelectAbsorbAttendAndProjUV(encD, mlaW: mlaW, kv: kvViews, attOut: decodeOut,
                                                      fullAttnLayer: 0, pos0: 3072, rows: 1, ln1: ln1T)
            hal.end(cbD, encD)
            cbD.waitUntilCompleted()
            XCTAssertNil(cbD.error)
            return (
                readF16(prefillOut, count: 32 * geo.qHeads * geo.vHeadDim),
                readF16(decodeOut, count: geo.qHeads * geo.vHeadDim),
                kPeSnap
            )
        }

        for mode in [QWKVMode.f16, .q8, .q4] {
            let (pMMA, dSplit, kPe0) = try runMode(mode, mma: true)
            let (pSc, dSc, _) = try runMode(mode, mma: false)
            let maxPErr = zip(pMMA, pSc).map { abs($0 - $1) }.max() ?? 999
            let maxDErr = zip(dSplit, dSc).map { abs($0 - $1) }.max() ?? 999
            XCTAssertLessThan(maxPErr, 2e-2, "prefill dense MLA mismatch mode=\(mode)")
            XCTAssertLessThan(maxDErr, 1e-2, "decode split-K MLA mismatch mode=\(mode)")
            XCTAssertGreaterThan(pSc.map { abs($0) }.max() ?? 0, 1e-4)
            if mode == .f16 {
                let (pShift, _, kPeShift) = try runMode(mode, mma: false, hShift: -3)
                let peDiff = zip(kPe0, kPeShift).map { abs($0 - $1) }.max() ?? 0
                let outDiff = zip(pSc, pShift).map { abs($0 - $1) }.max() ?? 0
                XCTAssertGreaterThan(peDiff, 1e-2, "bailingVideoRope 3D MRoPE must rotate negative centered H axis in kPe")
                XCTAssertGreaterThan(outDiff, 1e-4, "bailingVideoRope 3D MRoPE must alter attention output")
                let bGrid = try QWVisionGrid(t: 1, h: 8, w: 8, spec: .qwen3VLReference)
                let (bPos, bDelta) = qwMRopePositions(
                    blocks: [.image(tokens: [Int](repeating: 0, count: bGrid.tokens), grid: bGrid)],
                    merge: 2, layout: .bailingVideoRope)
                XCTAssertTrue(bPos.contains(where: { $0 < 0 }))
                XCTAssertEqual(bDelta, 1 - bGrid.tokens)
            }
        }
    }

    // MARK: - 3. Non-Square External linear_proj and Bias-Free QWVision Merger

    func testNonSquareExternalAndBiasFreeVisionMergerParity() throws {
        try requireGPU()
        for withBias in [true, false] {
            let dir = FileManager.default.temporaryDirectory
                .appendingPathComponent("qwg-vis-nonsquare-\(withBias)-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: dir) }

            let d = 128, heads = 2, mlp = 256, depth = 1, patch = 4, temporal = 2, merge = 2
            let merged = merge * merge * d // 512
            let fc1Mid = 256
            let outD = 192
            let qkvD = 3 * d
            var specs: [(String, [Int], Float)] = [
                ("model.visual.pos_embed.weight", [16, d], 0.05),
                ("model.visual.patch_embed.proj.weight", [d, 3, temporal, patch, patch], 0.05),
                ("model.visual.patch_embed.proj.bias", [d], 0.02),
                ("model.visual.merger.norm.weight", [d], 1.0),
                ("model.visual.merger.norm.bias", [d], 0.02),
                ("model.visual.blocks.0.norm1.weight", [d], 1.0),
                ("model.visual.blocks.0.norm1.bias", [d], 0.02),
                ("model.visual.blocks.0.norm2.weight", [d], 1.0),
                ("model.visual.blocks.0.norm2.bias", [d], 0.02),
                ("model.visual.blocks.0.attn.qkv.weight", [qkvD, d], 0.05),
                ("model.visual.blocks.0.attn.qkv.bias", [qkvD], 0.02),
                ("model.visual.blocks.0.attn.proj.weight", [d, d], 0.05),
                ("model.visual.blocks.0.attn.proj.bias", [d], 0.02),
                ("model.visual.blocks.0.mlp.linear_fc1.weight", [mlp, d], 0.04),
                ("model.visual.blocks.0.mlp.linear_fc1.bias", [mlp], 0.02),
                ("model.visual.blocks.0.mlp.linear_fc2.weight", [d, mlp], 0.04),
                ("model.visual.blocks.0.mlp.linear_fc2.bias", [d], 0.02),
                (withBias ? "linear_proj.0.weight" : "model.visual.merger.linear_fc1.weight", [fc1Mid, merged], 0.04),
                (withBias ? "linear_proj.2.weight" : "model.visual.merger.linear_fc2.weight", [outD, fc1Mid], 0.04),
            ]
            if withBias {
                specs.append(("linear_proj.0.bias", [fc1Mid], 0.02))
                specs.append(("linear_proj.2.bias", [outD], 0.02))
            }
            specs.sort { $0.0 < $1.0 }
            var rng = LCG(state: withBias ? 7001 : 7002)
            var hdr: [String: Any] = [:]
            var payload = Data()
            for (name, shape, scale) in specs {
                let count = shape.reduce(1, *)
                let start = payload.count
                var vals = [UInt16](repeating: 0, count: count)
                let isGamma = name.hasSuffix("norm1.weight") || name.hasSuffix("norm2.weight") || name.hasSuffix("merger.norm.weight")
                for i in 0..<count {
                    vals[i] = Float16(isGamma ? (1.0 + 0.1 * rng.nextUniform()) : (scale * rng.nextUniform())).bitPattern
                }
                vals.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in payload.append(raw.bindMemory(to: UInt8.self)) }
                hdr[name] = ["dtype": "F16", "shape": shape, "data_offsets": [start, payload.count]]
            }
            let hdrData = try JSONSerialization.data(withJSONObject: hdr)
            var fileData = Data(count: 8 + hdrData.count + payload.count)
            var hdrLen = UInt64(hdrData.count).littleEndian
            withUnsafeBytes(of: &hdrLen) { (raw: UnsafeRawBufferPointer) in fileData.replaceSubrange(0..<8, with: raw) }
            fileData.replaceSubrange(8..<(8 + hdrData.count), with: hdrData)
            fileData.replaceSubrange((8 + hdrData.count)..<fileData.count, with: payload)
            try fileData.write(to: dir.appendingPathComponent("model.safetensors"))

            let weights = try QWVisionWeights(dir: dir.path)
            XCTAssertEqual(weights.config.dim, d)
            XCTAssertEqual(weights.config.heads, heads)
            XCTAssertEqual(weights.config.depth, depth)
            XCTAssertEqual(weights.fc1Mid, fc1Mid)
            XCTAssertEqual(weights.fc2Out, outD)
            XCTAssertEqual(weights.config.projectorBias, withBias)

            let spec = QWMMSpec(visionConfig: weights.config)
            let grid = try QWVisionGrid(t: 1, h: 4, w: 4, spec: spec)
            let patches = (0..<(grid.patches * weights.config.patchChannels)).map { _ in 0.4 * rng.nextUniform() }
            let backend = try QWVisionOnDemandBackend(modelPath: dir.path, keepResident: true, allowOnDemand: true)
            let metalOut = try backend.encodeSync(patches: patches, grid: grid.plan, bins: grid.interpBins, posIds: grid.ropePositions)
            let refOut = QWVisionRef(weights: weights).encode(
                patches: patches, plan: grid.plan, bins: grid.interpBins, posIds: grid.ropePositions)
            XCTAssertEqual(metalOut.count, grid.tokens * outD)
            let maxAbs = zip(metalOut, refOut).map { abs($0 - $1) }.max() ?? 999
            XCTAssertLessThan(maxAbs, 5e-3, "vision non-square merger mismatch withBias=\(withBias)")
        }
    }

    // MARK: - 4. HCA headDim=256 and MBSA idxHeads=2, idxDim=64

    func testHCADim256AndMBSADecoupledIndexerParity() throws {
        try requireGPU()
        let hal = try QWHal(geometry: .qwen38_27b)

        // Part A: HCA headDim=256 unsplit vs Split-K combine vs CPU reference
        let hcaPso = try hal.pipeline("qwg_hca_attn_mma")
        let hcaComb = try hal.pipeline("qwg_hca_attn_combine")
        let T = 4, pos0 = 260, qHeads = 16, hd = 256, compressRatio = 128, sw = 128, ringCap = 256
        let nSplit = 4
        let qUpStride = qHeads * hd
        var rng = LCG(state: 0x2560)
        let qUpVals = (0..<(T * qUpStride)).map { _ in Float(Float16(rng.nextUniform() * 0.3)) }
        let compVals = (0..<(4 * hd)).map { _ in Float(Float16(rng.nextUniform() * 0.25)) }
        let swaVals = (0..<(ringCap * hd)).map { _ in Float(Float16(rng.nextUniform() * 0.25)) }
        let sinkVals = (0..<qHeads).map { Float($0 % 4) * 0.2 - 0.3 }
        let dummyI32 = hal.device.makeBuffer(length: 256, options: .storageModeShared)!
        let qUpT = tv(makeF16Buf(hal.device, qUpVals), T * qUpStride, .f16)
        let compT = tv(makeF16Buf(hal.device, compVals), 4 * hd, .f16)
        let swaT = tv(makeF16Buf(hal.device, swaVals), ringCap * hd, .f16)
        let sinkT = tv(makeF32Buf(hal.device, sinkVals), qHeads, .f32)
        let outUnsplit = tv(hal.device.makeBuffer(length: T * qHeads * hd * 2, options: .storageModeShared)!, T * qHeads * hd, .f16)
        let outSplit = tv(hal.device.makeBuffer(length: T * qHeads * hd * 2, options: .storageModeShared)!, T * qHeads * hd, .f16)
        let partO = tv(hal.device.makeBuffer(length: T * nSplit * qHeads * hd * 2, options: .storageModeShared)!, T * nSplit * qHeads * hd, .f16)
        let partML = tv(hal.device.makeBuffer(length: T * nSplit * qHeads * 8, options: .storageModeShared)!, T * nSplit * qHeads * 2, .f32)
        var params = QWHCAAttnOperator.AttnParams(
            T: UInt32(T), pos0: UInt32(pos0), qHeads: UInt32(qHeads), headDim: UInt32(hd),
            qUpStride: UInt32(qUpStride), compressRatio: UInt32(compressRatio), slidingWindow: UInt32(sw),
            ringMask: UInt32(ringCap - 1), compBudget: 128, listStride: 128, useBlockList: 0, hasSink: 1)
        for splitCount in [1, nSplit] {
            let (cb, enc) = try XCTUnwrap(hal.begin())
            enc.setComputePipelineState(hcaPso)
            qUpT.bind(enc, index: 0); compT.bind(enc, index: 1); swaT.bind(enc, index: 2)
            enc.setBuffer(dummyI32, offset: 0, index: 3); enc.setBuffer(dummyI32, offset: 0, index: 4)
            sinkT.bind(enc, index: 5); (splitCount > 1 ? partO : outUnsplit).bind(enc, index: 6)
            partML.bind(enc, index: 7); enc.setBytes(&params, length: MemoryLayout<QWHCAAttnOperator.AttnParams>.stride, index: 8)
            enc.dispatchThreadgroups(MTLSize(width: T, height: qHeads / 16, depth: splitCount),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            if splitCount > 1 {
                enc.memoryBarrier(scope: .buffers)
                var nSU = UInt32(splitCount)
                enc.setComputePipelineState(hcaComb)
                partO.bind(enc, index: 0); partML.bind(enc, index: 1); sinkT.bind(enc, index: 2)
                outSplit.bind(enc, index: 3)
                enc.setBytes(&params, length: MemoryLayout<QWHCAAttnOperator.AttnParams>.stride, index: 4)
                enc.setBytes(&nSU, length: 4, index: 5)
                enc.dispatchThreadgroups(MTLSize(width: T * qHeads, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            }
            hal.end(cb, enc)
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error)
        }
        let uOut = readF16(outUnsplit, count: T * qHeads * hd)
        let sOut = readF16(outSplit, count: T * qHeads * hd)
        XCTAssertLessThan(zip(uOut, sOut).map { abs($0 - $1) }.max() ?? 999, 3e-3)
        // Verify first row/head against CPU MQA reference
        let pos = pos0
        let totalComp = (pos + 1) / compressRatio
        let swaStart = pos + 1 - sw
        var keys = [[Float]]()
        for c in 0..<totalComp { keys.append(Array(compVals[(c * hd)..<((c + 1) * hd)])) }
        for p in swaStart...pos { let r = p & (ringCap - 1); keys.append(Array(swaVals[(r * hd)..<((r + 1) * hd)])) }
        let sc = 1.0 / sqrt(Float(hd))
        var maxScore = sinkVals[0]
        let scores = keys.map { k -> Float in
            let dot = (0..<hd).reduce(Float(0)) { $0 + qUpVals[$1] * k[$1] } * sc
            maxScore = max(maxScore, dot)
            return dot
        }
        var wSum = exp(sinkVals[0] - maxScore)
        var acc = [Float](repeating: 0, count: hd)
        for (i, s) in scores.enumerated() {
            let e = exp(s - maxScore); wSum += e
            for d in 0..<hd { acc[d] += e * keys[i][d] }
        }
        let refErr = (0..<hd).map { abs(sOut[$0] - acc[$0] / wSum) }.max() ?? 999
        XCTAssertLessThan(refErr, 3e-3)

        // Part B: MBSA idxHeads=2, idxDim=64 != kvHeads=4, headDim=128
        let mbsaCfg: [String: Any] = [
            "model_type": "maxwell_moe", "hidden_size": 512, "intermediate_size": 1024, "vocab_size": 1024,
            "num_hidden_layers": 1, "num_experts": 0, "num_attention_heads": 16, "num_key_value_heads": 4,
            "attention_head_size": 128, "qk_layernorm": true, "attention_variant": "block_sparse_gqa",
            "block_sparse_gqa_indexer_n_heads": 2, "block_sparse_gqa_indexer_head_dim": 64,
            "block_sparse_gqa_block_size": 32, "block_sparse_gqa_prefix_window_size": 32,
            "block_sparse_gqa_local_window_size": 64, "block_sparse_gqa_num_remote_blocks": 2,
            "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "max_position_embeddings": 1024, "eos_token_id": 2
        ]
        let mbsaGeo = QWModelGeometry(descriptor: try QWModelDescriptor.decode(
            configJSON: JSONSerialization.data(withJSONObject: mbsaCfg)))
        let mbsaHal = try QWHal(geometry: mbsaGeo, weightBasis: .standard)
        let mbsa = try XCTUnwrap(try QWMBSAAttnOperator.makeIfEnabled(
            hal: mbsaHal, geometry: mbsaGeo, attnLayers: [0], kvMode: .f16, maxContext: 512, maxPrefillChunk: 64))
        XCTAssertEqual(mbsa.idxHeads, 2)
        XCTAssertEqual(mbsa.idxDim, 64)
        XCTAssertEqual(mbsa.supportsNAXPrefill, mbsaHal.caps.isMetal4TensorCapable)

        let idxD = mbsa.idxDim, idxH = mbsa.idxHeads
        let mbsaW = QWMBSAAttnOperator.LayerWeights(
            wqIdx: tv(makeF16Buf(mbsaHal.device, [Float](repeating: 0.2, count: idxD)), idxD, .f16),
            wkIdx: tv(makeF16Buf(mbsaHal.device, [Float](repeating: 0.2, count: idxD)), idxD, .f16))
        let ropeTbl = tv(makeF32Buf(mbsaHal.device, [Float](repeating: 1.0, count: 512 * mbsaGeo.rotaryDims)),
                         512 * mbsaGeo.rotaryDims, .f32)
        let pageBytes = QWKVCache.pageBytes(mode: .f16, geometry: mbsaGeo)
        let tables = QWKVCache.tableSlotBytes * 2 * QWKVCache.maxSeqs
        let kvArena = try QWArena(hal: mbsaHal, weights: 0, activations: 0, kv: pageBytes * 4 + tables, gdnStates: 4096)
        let kvCache = try QWKVCache(hal: mbsaHal, arena: kvArena, layers: [0], mode: .f16, geometry: mbsaGeo)
        let seq = try kvCache.allocate(promptTokens: 320)
        let kvViews = try kvCache.layerViews(seq, layer: 0)

        for (tag, rows, p0) in [("decode", 4, 240), ("nax_prefill", 40, 244)] {
            let qIdxSrc = (0..<(rows * (idxH + 1) * idxD)).map { _ in Float(Float16(rng.nextUniform() * 0.2)) }
            let dst = mbsa.region.pointer.advanced(by: mbsa.idxQKVScratch.offset).assumingMemoryBound(to: UInt16.self)
            for i in 0..<qIdxSrc.count { dst[i] = Float16(qIdxSrc[i]).bitPattern }
            let qgateT = tv(makeF16Buf(mbsaHal.device, [Float](repeating: 0.1, count: rows * mbsaGeo.qProjOut)),
                            rows * mbsaGeo.qProjOut, .f16)
            let outT = tv(makeF16Buf(mbsaHal.device, [Float](repeating: 0, count: rows * mbsaGeo.qHeads * mbsaGeo.headDim)),
                          rows * mbsaGeo.qHeads * mbsaGeo.headDim, .f16)
            let (cb, enc) = try XCTUnwrap(mbsaHal.begin())
            mbsa.encodeIndexPrep(enc, weights: mbsaW, ropeTable: ropeTbl, layerIndex: 0, pos0: p0, rows: rows)
            enc.memoryBarrier(scope: .buffers)
            mbsa.encodeSelectAndAttend(enc, qgate: qgateT, kv: kvViews, out: outT, layerIndex: 0, pos0: p0, rows: rows)
            mbsaHal.end(cb, enc)
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error, "MBSA \(tag) failed")
            let vals = readF16(outT, count: rows * mbsaGeo.qHeads * mbsaGeo.headDim)
            XCTAssertTrue(vals.allSatisfy { $0.isFinite }, "MBSA \(tag) produced non-finite values")
        }
    }
}
