import XCTest
import Metal
@testable import QWCore
@testable import QWHAL
@testable import QWEngine
import QWAlloc
import QWGEMM
import QWQuant

/// Split mHC router (partial + finalize) and the multi-row prefill router must match the fused
/// qwg_mhc_router_norm_f16 kernel.
final class QWMHCSplitRouterTests: XCTestCase {
    private let c = 4
    private let hidden = 4096

    private func buffer(_ hal: QWHal, _ values: [Float]) throws -> any MTLBuffer {
        try XCTUnwrap(hal.device.makeBuffer(bytes: values, length: values.count * 4, options: .storageModeShared))
    }

    private func buffer(_ hal: QWHal, f16 values: [Float]) throws -> any MTLBuffer {
        let h = values.map { Float16($0) }
        return try XCTUnwrap(hal.device.makeBuffer(bytes: h, length: h.count * 2, options: .storageModeShared))
    }

    private func floats(_ b: any MTLBuffer, _ n: Int) -> [Float] {
        Array(UnsafeBufferPointer(start: b.contents().bindMemory(to: Float.self, capacity: n), count: n))
    }

    private func halves(_ b: any MTLBuffer, _ n: Int) -> [Float] {
        let p = b.contents().bindMemory(to: UInt16.self, capacity: n)
        return [Float](unsafeUninitializedCapacity: n) { dst, initCnt in
            guard let dp = dst.baseAddress else { initCnt = 0; return }
            var i = 0; while i < n { dp[i] = Float(Float16(bitPattern: p[i])); i += 1 }
            initCnt = n
        }
    }

    private struct Outputs {
        var stream: [Float]
        var norm: [Float]
        var ctrl: [Float]
    }

    private enum Kernel { case fused, split, rows }

    private func run(_ kernel: Kernel, merge: Bool, rows: Int = 3,
                     streams: Int? = nil, hiddenOverride: Int? = nil,
                     geometry: QWModelGeometry? = nil) throws -> Outputs {
        let activeC = streams ?? c
        let activeHidden = hiddenOverride ?? hidden
        let geo = try geometry ?? QWWeightAllocationPlanTests.glm53FlashGeometry()
        let hal = try QWHal(geometry: geo)
        let numRows = 2 * activeC + activeC * activeC
        let numCtrl = activeC + activeC * activeC
        let width = activeC * activeHidden
        let streamArr = (0..<(rows * width)).map { Float(($0 % 997) - 498) * 0.0019 }
        let wF32 = (0..<(numRows * width)).map { cos(Float($0 % 2048) * 0.013) * 0.05 }
        let stream = try buffer(hal, streamArr)
        let w = try buffer(hal, wF32)
        let b = try buffer(hal, (0..<numRows).map { sin(Float($0) * 0.3) * 0.2 })
        let gamma = try buffer(hal, f16: (0..<activeHidden).map { sin(Float($0) * 0.01) * 0.1 })
        let out = try buffer(hal, f16: [Float](repeating: 0, count: rows * activeHidden))
        var prev = [Float](repeating: 0, count: rows * numCtrl)
        for t in 0..<rows {
            for i in 0..<activeC { prev[t * numCtrl + i] = 0.5 + 0.1 * Float(i) }
            for m in 0..<(activeC * activeC) { prev[t * numCtrl + activeC + m] = (m % (activeC + 1) == 0 ? 0.7 : 0.1) }
        }
        let ctrl = try buffer(hal, prev)
        let branch = try buffer(hal, f16: (0..<(rows * activeHidden)).map { Float((($0 * 7) % 503) - 251) * 0.002 })
        var uHidden = UInt32(activeHidden)
        var eps: Float = 1e-6
        var preScale: Float = 1.0
        var iters: UInt32 = 20 | (merge ? 0x8000_0000 : 0)

        let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let enc = try XCTUnwrap(cb.makeComputeCommandEncoder(dispatchType: .serial))
        if kernel == .split {
            let parts = (activeHidden + 63) / 64
            let partials = try XCTUnwrap(hal.device.makeBuffer(length: rows * parts * (numRows + 1 + activeC * (activeC + 1) / 2) * 4))
            enc.setComputePipelineState(try hal.pipeline("qwg_mhc_router_partial_f32"))
            enc.setBuffer(stream, offset: 0, index: 0)
            enc.setBuffer(w, offset: 0, index: 1)
            enc.setBuffer(ctrl, offset: 0, index: 2)
            enc.setBuffer(merge ? branch : out, offset: 0, index: 3)
            enc.setBuffer(partials, offset: 0, index: 4)
            enc.setBytes(&uHidden, length: 4, index: 5)
            enc.setBytes(&iters, length: 4, index: 6)
            enc.dispatchThreadgroups(MTLSize(width: parts, height: rows, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(try hal.pipeline("qwg_mhc_router_finalize_f16"))
            enc.setBuffer(stream, offset: 0, index: 0)
            enc.setBuffer(partials, offset: 0, index: 1)
            enc.setBuffer(b, offset: 0, index: 2)
            enc.setBuffer(gamma, offset: 0, index: 3)
            enc.setBuffer(out, offset: 0, index: 4)
            enc.setBuffer(ctrl, offset: 0, index: 5)
            enc.setBytes(&uHidden, length: 4, index: 6)
            enc.setBytes(&eps, length: 4, index: 7)
            enc.setBytes(&preScale, length: 4, index: 8)
            enc.setBytes(&iters, length: 4, index: 9)
            enc.dispatchThreadgroups(MTLSize(width: (activeHidden + 255) / 256, height: rows, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        } else {
            enc.setComputePipelineState(try hal.pipeline(kernel == .rows ? "qwg_mhc_router_norm_rows_f16"
                                                                         : "qwg_mhc_router_norm_f16"))
            for (i, buf) in [stream, w, b, gamma, out, ctrl].enumerated() {
                enc.setBuffer(buf, offset: 0, index: i)
            }
            enc.setBytes(&uHidden, length: 4, index: 6)
            enc.setBytes(&eps, length: 4, index: 7)
            enc.setBytes(&preScale, length: 4, index: 8)
            enc.setBytes(&iters, length: 4, index: 9)
            enc.setBuffer(merge ? branch : out, offset: 0, index: 10)
            var uRows = UInt32(rows)
            enc.setBytes(&uRows, length: 4, index: 11)
            let rpg = QWEngineDriver.mhcRouterRowsPerGroup(streams: activeC)
            let groups = kernel == .rows ? (rows + rpg - 1) / rpg : rows
            enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)
        return Outputs(stream: floats(stream, rows * width), norm: halves(out, rows * activeHidden),
                       ctrl: floats(ctrl, rows * numCtrl))
    }

    private func assertZeroULP(_ a: [Float], _ b: [Float], _ what: String) {
        XCTAssertEqual(a.count, b.count)
        guard a.count == b.count else { return }
        a.withUnsafeBufferPointer { aB in
            b.withUnsafeBufferPointer { bB in
                guard let ap = aB.baseAddress, let bp = bB.baseAddress else { return }
                if memcmp(ap, bp, aB.count * 4) == 0 { return }
                var i = 0
                while i < aB.count {
                    if ap[i].bitPattern != bp[i].bitPattern {
                        XCTAssertEqual(ap[i].bitPattern, bp[i].bitPattern, "\(what) 0-ULP mismatch at \(i): \(ap[i]) vs \(bp[i])")
                        return
                    }
                    i += 1
                }
            }
        }
    }

    private func c2Geometry() throws -> QWModelGeometry {
        let json = """
        {
          "architectures": ["SenseNovaLoopForConditionalGeneration"],
          "model_type": "sensenova_loop",
          "loop_settings": { "start": 0, "end": 1, "eval_count": 1, "counts": [1] },
          "mhc": { "start": 0, "end": 1, "streams": 2, "iterations": 20, "fixed_identity": false, "pre_scale": 1.0 },
          "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "vocab_size": 1024,
            "num_hidden_layers": 2,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "layer_types": ["linear_attention", "full_attention"],
            "max_position_embeddings": 8192,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
            "eos_token_id": 2
          }
        }
        """.data(using: .utf8)!
        return QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: json))
    }

    private func assertClose(_ a: [Float], _ b: [Float], tol: Float, _ what: String) {
        XCTAssertEqual(a.count, b.count)
        let worst = a.withUnsafeBufferPointer { aB in
            b.withUnsafeBufferPointer { bB -> Float in
                guard let ap = aB.baseAddress, let bp = bB.baseAddress else { return 0 }
                var mx: Float = 0, i = 0
                while i < aB.count {
                    let r = abs(ap[i] - bp[i]) / max(1, abs(bp[i]))
                    if r > mx { mx = r }
                    i += 1
                }
                return mx
            }
        }
        XCTAssertLessThanOrEqual(worst, tol, "\(what) max rel err \(worst)")
    }

    func testSplitMatchesFusedWithoutMerge() throws {
        let fused = try run(.fused, merge: false)
        let split = try run(.split, merge: false)
        assertClose(split.stream, fused.stream, tol: 0, "stream")
        assertClose(split.ctrl, fused.ctrl, tol: 1e-4, "ctrl")
        assertClose(split.norm, fused.norm, tol: 1e-3, "norm")
    }

    func testSplitMatchesFusedWithMerge() throws {
        let fused = try run(.fused, merge: true)
        let split = try run(.split, merge: true)
        assertClose(split.stream, fused.stream, tol: 1e-6, "stream")
        assertClose(split.ctrl, fused.ctrl, tol: 1e-4, "ctrl")
        assertClose(split.norm, fused.norm, tol: 1e-3, "norm")
    }

    func testRowsKernelMatchesFused() throws {
        for merge in [false, true] {
            let fused = try run(.fused, merge: merge, rows: 7)
            let multi = try run(.rows, merge: merge, rows: 7)
            assertClose(multi.stream, fused.stream, tol: 0, "stream merge=\(merge)")
            assertClose(multi.ctrl, fused.ctrl, tol: 1e-5, "ctrl merge=\(merge)")
            assertClose(multi.norm, fused.norm, tol: 1e-3, "norm merge=\(merge)")
        }
    }

    func testC2SplitAndRowsMatchFusedZeroULP() throws {
        let geo = try c2Geometry()
        for merge in [false, true] {
            let fused7 = try run(.fused, merge: merge, rows: 7, streams: 2, geometry: geo)
            let multi7 = try run(.rows, merge: merge, rows: 7, streams: 2, geometry: geo)
            assertZeroULP(multi7.stream, fused7.stream, "c2 rows stream merge=\(merge)")
            assertZeroULP(multi7.ctrl, fused7.ctrl, "c2 rows ctrl merge=\(merge)")
            assertZeroULP(multi7.norm, fused7.norm, "c2 rows norm merge=\(merge)")

            let fused3 = try run(.fused, merge: merge, rows: 3, streams: 2, geometry: geo)
            let split3 = try run(.split, merge: merge, rows: 3, streams: 2, geometry: geo)
            assertClose(split3.stream, fused3.stream, tol: 1e-6, "c2 split stream merge=\(merge)")
            assertClose(split3.ctrl, fused3.ctrl, tol: 1e-4, "c2 split ctrl merge=\(merge)")
            assertClose(split3.norm, fused3.norm, tol: 1e-3, "c2 split norm merge=\(merge)")
        }
    }

    private func ernieAdditiveEpsGeometry() throws -> QWModelGeometry {
        let json = """
        {
          "architectures": ["Ernie5_5_MoeForCausalLM"],
          "model_type": "ernie5_5_moe",
          "vocab_size": 1024,
          "hidden_size": 4096,
          "intermediate_size": 256,
          "num_hidden_layers": 2,
          "max_position_embeddings": 8192,
          "rms_norm_eps": 1e-6,
          "num_attention_heads": 64,
          "num_key_value_heads": 1,
          "head_dim": 128,
          "partial_rotary_factor": 1.0,
          "layer_types": ["full_attention", "full_attention"],
          "enable_hyper_connections": true,
          "hc_mult": 4,
          "hc_sinkhorn_iters": 20,
          "hc_eps": 1e-6,
          "eos_token_id": 2
        }
        """.data(using: .utf8)!
        return QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: json))
    }

    func testRowsKernelLargeAndSmallParity() throws {
        let geoErnie = try ernieAdditiveEpsGeometry()
        for merge in [false, true] {
            let fused65 = try run(.fused, merge: merge, rows: 65, geometry: geoErnie)
            let multi65 = try run(.rows, merge: merge, rows: 65, geometry: geoErnie)
            assertZeroULP(multi65.stream, fused65.stream, "ernie rows stream T=65 merge=\(merge)")
            assertClose(multi65.ctrl, fused65.ctrl, tol: 1e-5, "ernie rows ctrl T=65 merge=\(merge)")
            assertClose(multi65.norm, fused65.norm, tol: 1e-3, "ernie rows norm T=65 merge=\(merge)")
        }
        let fusedSmall = try run(.fused, merge: true, rows: 17, hiddenOverride: 128, geometry: geoErnie)
        let multiSmall = try run(.rows, merge: true, rows: 17, hiddenOverride: 128, geometry: geoErnie)
        assertZeroULP(multiSmall.stream, fusedSmall.stream, "h128 rows stream")
        assertClose(multiSmall.ctrl, fusedSmall.ctrl, tol: 1e-5, "h128 rows ctrl")
        assertClose(multiSmall.norm, fusedSmall.norm, tol: 1e-3, "h128 rows norm")
    }

    func testRowsKernelInformationalLatencyVsFusedAt4087() throws {
        let geo = try ernieAdditiveEpsGeometry()
        let hal = try QWHal(geometry: geo)

        let rows = 4087
        let activeC = 4
        let activeHidden = 4096
        let width = activeC * activeHidden
        let numRows = 2 * activeC + activeC * activeC
        let numCtrl = activeC + activeC * activeC
        let streamCount = rows * width
        let ctrlCount = rows * numCtrl

        let streamRef = try XCTUnwrap(hal.device.makeBuffer(length: streamCount * 4, options: .storageModeShared))
        let streamRun = try XCTUnwrap(hal.device.makeBuffer(length: streamCount * 4, options: .storageModeShared))
        let streamPtr = streamRef.contents().bindMemory(to: Float.self, capacity: streamCount)
        let sLut = (0..<997).map { Float($0 - 498) * 0.0019 }
        sLut.withUnsafeBufferPointer { lB in
            guard let lp = lB.baseAddress else { return }
            var i = 0
            while i + 997 <= streamCount { memcpy(streamPtr + i, lp, 997 * 4); i += 997 }
            while i < streamCount { streamPtr[i] = lp[i % 997]; i += 1 }
        }

        let ctrlRef = try XCTUnwrap(hal.device.makeBuffer(length: ctrlCount * 4, options: .storageModeShared))
        let ctrlRun = try XCTUnwrap(hal.device.makeBuffer(length: ctrlCount * 4, options: .storageModeShared))
        let ctrlPtr = ctrlRef.contents().bindMemory(to: Float.self, capacity: ctrlCount)
        var ctrlRow = [Float](repeating: 0, count: numCtrl)
        for i in 0..<activeC { ctrlRow[i] = 0.5 + 0.1 * Float(i) }
        for m in 0..<(activeC * activeC) { ctrlRow[activeC + m] = (m % (activeC + 1) == 0 ? 0.7 : 0.1) }
        ctrlRow.withUnsafeBufferPointer { rB in
            guard let rp = rB.baseAddress else { return }
            var t = 0
            while t < rows { memcpy(ctrlPtr + t * numCtrl, rp, numCtrl * 4); t += 1 }
        }

        let w = try XCTUnwrap(hal.device.makeBuffer(length: numRows * width * 4, options: .storageModeShared))
        let cosLut = (0..<2048).map { cos(Float($0) * 0.013) * 0.05 }
        cosLut.withUnsafeBufferPointer { lB in
            guard let lp = lB.baseAddress else { return }
            let wp = w.contents().bindMemory(to: Float.self, capacity: numRows * width)
            var i = 0, lim = numRows * width
            while i + 2048 <= lim { memcpy(wp + i, lp, 2048 * 4); i += 2048 }
            while i < lim { wp[i] = lp[i & 2047]; i += 1 }
        }
        let b = try buffer(hal, (0..<numRows).map { sin(Float($0) * 0.3) * 0.2 })
        let gamma = try buffer(hal, f16: (0..<activeHidden).map { sin(Float($0) * 0.01) * 0.1 })
        let branch = try XCTUnwrap(hal.device.makeBuffer(length: rows * activeHidden * 2, options: .storageModeShared))
        let brLut = (0..<503).map { Float16(Float((($0 * 7) % 503) - 251) * 0.002) }
        brLut.withUnsafeBufferPointer { lB in
            guard let lp = lB.baseAddress else { return }
            let bp = branch.contents().bindMemory(to: Float16.self, capacity: rows * activeHidden)
            var i = 0, lim = rows * activeHidden
            while i + 503 <= lim { memcpy(bp + i, lp, 503 * 2); i += 503 }
            while i < lim { bp[i] = lp[i % 503]; i += 1 }
        }
        let outFused = try XCTUnwrap(hal.device.makeBuffer(length: rows * activeHidden * 2, options: .storageModeShared))
        let outRows = try XCTUnwrap(hal.device.makeBuffer(length: rows * activeHidden * 2, options: .storageModeShared))

        let psoFused = try hal.pipeline("qwg_mhc_router_norm_f16")
        let psoRows = try hal.pipeline("qwg_mhc_router_norm_rows_f16")

        func resetInputs() throws {
            let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
            let blit = try XCTUnwrap(cb.makeBlitCommandEncoder())
            blit.copy(from: streamRef, sourceOffset: 0, to: streamRun, destinationOffset: 0, size: streamCount * 4)
            blit.copy(from: ctrlRef, sourceOffset: 0, to: ctrlRun, destinationOffset: 0, size: ctrlCount * 4)
            blit.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error)
        }

        func runOnce(isRows: Bool, outBuf: any MTLBuffer) throws -> Double {
            try resetInputs()
            var uHidden = UInt32(activeHidden)
            var eps: Float = 1e-6
            var preScale: Float = 1.0
            var iters: UInt32 = 20 | 0x8000_0000
            var uRows = UInt32(rows)
            let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
            let enc = try XCTUnwrap(cb.makeComputeCommandEncoder(dispatchType: .serial))
            enc.setComputePipelineState(isRows ? psoRows : psoFused)
            for (i, buf) in [streamRun, w, b, gamma, outBuf, ctrlRun].enumerated() {
                enc.setBuffer(buf, offset: 0, index: i)
            }
            enc.setBytes(&uHidden, length: 4, index: 6)
            enc.setBytes(&eps, length: 4, index: 7)
            enc.setBytes(&preScale, length: 4, index: 8)
            enc.setBytes(&iters, length: 4, index: 9)
            enc.setBuffer(branch, offset: 0, index: 10)
            let groups: Int
            if isRows {
                enc.setBytes(&uRows, length: 4, index: 11)
                let rpg = QWEngineDriver.mhcRouterRowsPerGroup(streams: activeC)
                groups = (rows + rpg - 1) / rpg
            } else {
                groups = rows
            }
            enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error)
            return (cb.gpuEndTime - cb.gpuStartTime) * 1000.0
        }

        _ = try runOnce(isRows: false, outBuf: outFused)
        let fusedStream = floats(streamRun, streamCount)
        let fusedCtrl = floats(ctrlRun, ctrlCount)
        _ = try runOnce(isRows: true, outBuf: outRows)
        assertZeroULP(floats(streamRun, streamCount), fusedStream, "T=4087 stream fused vs rows")
        assertClose(floats(ctrlRun, ctrlCount), fusedCtrl, tol: 1e-5, "T=4087 ctrl fused vs rows")
        assertClose(halves(outRows, rows * activeHidden), halves(outFused, rows * activeHidden),
                    tol: 1e-3, "T=4087 norm fused vs rows")

        var fusedBest = Double.greatestFiniteMagnitude
        var rowsBest = Double.greatestFiniteMagnitude
        for iter in 0..<7 {
            let fMs = try runOnce(isRows: false, outBuf: outFused)
            let rMs = try runOnce(isRows: true, outBuf: outRows)
            if iter >= 2 {
                fusedBest = min(fusedBest, fMs)
                rowsBest = min(rowsBest, rMs)
            }
        }
        print(String(format: "[QWMHCSplitRouterTests] T=%d H=%d C=%d merge=true minGPU: fused=%.3f ms vs rows=%.3f ms (%.2fx)",
                     rows, activeHidden, activeC, fusedBest, rowsBest, fusedBest / max(rowsBest, 1e-6)))
    }

    private func runRouterKernel(hal: QWHal, pso: any MTLComputePipelineState, isNax: Bool,
                                 wBuf: any MTLBuffer, xBuf: any MTLBuffer, outBuf: any MTLBuffer,
                                 rows: Int, n: Int, k: Int,
                                 weightMode: UInt32 = 0, flags: UInt32 = 0) throws -> Double {
        var uM = UInt32(rows)
        var uN = UInt32(n)
        var uK = UInt32(k)
        var wm = weightMode
        var fl = flags
        let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let enc = try XCTUnwrap(cb.makeComputeCommandEncoder(dispatchType: .serial))
        enc.setComputePipelineState(pso)
        enc.setBuffer(wBuf, offset: 0, index: 0)
        enc.setBuffer(xBuf, offset: 0, index: 1)
        enc.setBuffer(outBuf, offset: 0, index: 2)
        enc.setBytes(&uM, length: 4, index: 3)
        enc.setBytes(&uN, length: 4, index: 4)
        enc.setBytes(&uK, length: 4, index: 5)
        if !isNax {
            enc.setBytes(&wm, length: 4, index: 6)
        }
        enc.setBuffer(wBuf, offset: 0, index: 7)
        enc.setBuffer(outBuf, offset: 0, index: 8)
        enc.setBytes(&fl, length: 4, index: 9)
        if isNax {
            enc.dispatchThreadgroups(MTLSize(width: n / 64, height: (rows + 63) / 64, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        } else {
            enc.dispatchThreadgroups(MTLSize(width: (n + 31) / 32, height: (rows + 31) / 32, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)
        return (cb.gpuEndTime - cb.gpuStartTime) * 1000.0
    }

    func testRouterNaxF32InformationalLatencyVsGemmAt4087() throws {
        let geo = try ernieAdditiveEpsGeometry()
        let hal = try QWHal(geometry: geo)
        try XCTSkipUnless(hal.caps.isMetal4TensorCapable, "qwg_router_nax_f32 requires Metal 4")

        let rows = 4087
        let n = 1024
        let k = 4096
        let wBuf = try XCTUnwrap(hal.device.makeBuffer(length: n * k * 4, options: .storageModeShared))
        let xBuf = try XCTUnwrap(hal.device.makeBuffer(length: rows * k * 4, options: .storageModeShared))
        let outNax = try XCTUnwrap(hal.device.makeBuffer(length: rows * n * 4, options: .storageModeShared))
        let outGemm = try XCTUnwrap(hal.device.makeBuffer(length: rows * n * 4, options: .storageModeShared))

        let wPtr = wBuf.contents().bindMemory(to: Float.self, capacity: n * k)
        for i in 0..<(n * k) { wPtr[i] = Float(((i * 13) % 997) - 498) * 0.0002 }
        let xPtr = xBuf.contents().bindMemory(to: Float.self, capacity: rows * k)
        for i in 0..<(rows * k) { xPtr[i] = Float(((i * 7) % 991) - 495) * 0.0005 }

        let psoNax = try hal.pipeline("qwg_router_nax_f32")
        let psoGemm = try hal.pipeline("qwg_router_gemm")

        _ = try runRouterKernel(hal: hal, pso: psoGemm, isNax: false,
                                wBuf: wBuf, xBuf: xBuf, outBuf: outGemm, rows: rows, n: n, k: k)
        _ = try runRouterKernel(hal: hal, pso: psoNax, isNax: true,
                                wBuf: wBuf, xBuf: xBuf, outBuf: outNax, rows: rows, n: n, k: k)
        assertClose(floats(outNax, rows * n), floats(outGemm, rows * n),
                    tol: 2e-3, "T=4087 N=1024 K=4096 router_nax_f32 vs gemm")

        var gemmBest = Double.greatestFiniteMagnitude
        var naxBest = Double.greatestFiniteMagnitude
        for iter in 0..<7 {
            let gMs = try runRouterKernel(hal: hal, pso: psoGemm, isNax: false,
                                          wBuf: wBuf, xBuf: xBuf, outBuf: outGemm, rows: rows, n: n, k: k)
            let nMs = try runRouterKernel(hal: hal, pso: psoNax, isNax: true,
                                          wBuf: wBuf, xBuf: xBuf, outBuf: outNax, rows: rows, n: n, k: k)
            if iter >= 2 {
                gemmBest = min(gemmBest, gMs)
                naxBest = min(naxBest, nMs)
            }
        }
        print(String(format: "[QWMHCSplitRouterTests] T=%d N=%d K=%d minGPU: router_gemm=%.3f ms vs router_nax_f32=%.3f ms (%.2fx)",
                     rows, n, k, gemmBest, naxBest, gemmBest / max(naxBest, 1e-6)))
    }

    func testRouterNaxF16InputParityAndLatencyAt4087() throws {
        let geo = try ernieAdditiveEpsGeometry()
        let hal = try QWHal(geometry: geo)
        try XCTSkipUnless(hal.caps.isMetal4TensorCapable, "qwg_router_nax requires Metal 4")

        let rows = 4087
        let n = 1024
        let k = 4096
        let wF32Buf = try XCTUnwrap(hal.device.makeBuffer(length: n * k * 4, options: .storageModeShared))
        let wBF16Buf = try XCTUnwrap(hal.device.makeBuffer(length: n * k * 2, options: .storageModeShared))
        let xF32Buf = try XCTUnwrap(hal.device.makeBuffer(length: rows * k * 4, options: .storageModeShared))
        let xF16Buf = try XCTUnwrap(hal.device.makeBuffer(length: rows * k * 2, options: .storageModeShared))
        let outRefF32 = try XCTUnwrap(hal.device.makeBuffer(length: rows * n * 4, options: .storageModeShared))
        let outRefBF16 = try XCTUnwrap(hal.device.makeBuffer(length: rows * n * 4, options: .storageModeShared))
        let outNaxF32 = try XCTUnwrap(hal.device.makeBuffer(length: rows * n * 4, options: .storageModeShared))
        let outNaxBF16 = try XCTUnwrap(hal.device.makeBuffer(length: rows * n * 4, options: .storageModeShared))

        let wF32Ptr = wF32Buf.contents().bindMemory(to: Float.self, capacity: n * k)
        let wBF16Ptr = wBF16Buf.contents().bindMemory(to: UInt16.self, capacity: n * k)
        let wLutF32 = (0..<997).map { Float((($0 * 13) % 997) - 498) * 0.0002 }
        let wLutBF16 = wLutF32.map { QWBF16.fromFloatRNE($0) }
        wLutF32.withUnsafeBufferPointer { lf in
            wLutBF16.withUnsafeBufferPointer { lb in
                guard let lfp = lf.baseAddress, let lbp = lb.baseAddress else { return }
                var i = 0, m = 0, lim = n * k
                while i < lim {
                    wF32Ptr[i] = lfp[m]; wBF16Ptr[i] = lbp[m]
                    m += 1; if m == 997 { m = 0 }
                    i += 1
                }
            }
        }
        let xF32Ptr = xF32Buf.contents().bindMemory(to: Float.self, capacity: rows * k)
        let xF16Ptr = xF16Buf.contents().bindMemory(to: Float16.self, capacity: rows * k)
        let xLutF16 = (0..<991).map { Float16(Float((($0 * 7) % 991) - 495) * 0.0005) }
        let xLutF32 = xLutF16.map { Float($0) }
        xLutF16.withUnsafeBufferPointer { lh in
            xLutF32.withUnsafeBufferPointer { lf in
                guard let lhp = lh.baseAddress, let lfp = lf.baseAddress else { return }
                var i = 0, m = 0, lim = rows * k
                while i < lim {
                    xF16Ptr[i] = lhp[m]; xF32Ptr[i] = lfp[m]
                    m += 1; if m == 991 { m = 0 }
                    i += 1
                }
            }
        }

        let psoGemm = try hal.pipeline("qwg_router_gemm")
        let psoNaxF32 = try hal.pipeline("qwg_router_nax_f32")
        let psoNaxBF16 = try hal.pipeline("qwg_router_nax")

        _ = try runRouterKernel(hal: hal, pso: psoGemm, isNax: false,
                                wBuf: wF32Buf, xBuf: xF32Buf, outBuf: outRefF32, rows: rows, n: n, k: k, weightMode: 0)
        _ = try runRouterKernel(hal: hal, pso: psoGemm, isNax: false,
                                wBuf: wBF16Buf, xBuf: xF32Buf, outBuf: outRefBF16, rows: rows, n: n, k: k, weightMode: 1)
        _ = try runRouterKernel(hal: hal, pso: psoNaxF32, isNax: true,
                                wBuf: wF32Buf, xBuf: xF16Buf, outBuf: outNaxF32, rows: rows, n: n, k: k, flags: 2)
        _ = try runRouterKernel(hal: hal, pso: psoNaxBF16, isNax: true,
                                wBuf: wBF16Buf, xBuf: xF16Buf, outBuf: outNaxBF16, rows: rows, n: n, k: k, flags: 2)

        assertClose(floats(outNaxF32, rows * n), floats(outRefF32, rows * n),
                    tol: 5e-4, "T=4087 N=1024 K=4096 router_nax_f32(f16 X) vs gemm")
        assertClose(floats(outNaxBF16, rows * n), floats(outRefBF16, rows * n),
                    tol: 2e-3, "T=4087 N=1024 K=4096 router_nax(bf16 W, f16 X) vs gemm")

        var f32Best = Double.greatestFiniteMagnitude
        var bf16Best = Double.greatestFiniteMagnitude
        for iter in 0..<7 {
            let fMs = try runRouterKernel(hal: hal, pso: psoNaxF32, isNax: true,
                                          wBuf: wF32Buf, xBuf: xF16Buf, outBuf: outNaxF32, rows: rows, n: n, k: k, flags: 2)
            let bMs = try runRouterKernel(hal: hal, pso: psoNaxBF16, isNax: true,
                                          wBuf: wBF16Buf, xBuf: xF16Buf, outBuf: outNaxBF16, rows: rows, n: n, k: k, flags: 2)
            if iter >= 2 {
                f32Best = min(f32Best, fMs)
                bf16Best = min(bf16Best, bMs)
            }
        }
        print(String(format: "[QWMHCSplitRouterTests] T=%d N=%d K=%d F16-X minGPU: router_nax_f32=%.3f ms vs router_nax_bf16=%.3f ms",
                     rows, n, k, f32Best, bf16Best))
    }

    func testMoEBridgePlainF16MatchesF32AndFeedsRouterNAX() throws {
        let geo = try ernieAdditiveEpsGeometry()
        let rows = 64, n = 128, k = 4096
        for basis in [QWWeightBasis.rotated, QWWeightBasis.standard] {
            let hal = try QWHal(geometry: geo, weightBasis: basis)
            let xBuf = try buffer(hal, f16: (0..<(rows * k)).map { Float((($0 * 7) % 991) - 495) * 0.0005 })
            let rotBuf = try buffer(hal, [Float](repeating: 0, count: rows * k))
            let p32Buf = try buffer(hal, [Float](repeating: 0, count: rows * k))
            let p16Buf = try buffer(hal, [Float](repeating: 0, count: (rows * k + 1) / 2))
            let wF32Buf = try buffer(hal, (0..<(n * k)).map { Float((($0 * 13) % 997) - 498) * 0.0002 })
            let wBF16Buf = try XCTUnwrap(hal.device.makeBuffer(length: n * k * 2, options: .storageModeShared))
            let bfPtr = wBF16Buf.contents().bindMemory(to: UInt16.self, capacity: n * k)
            let wfPtr = wF32Buf.contents().bindMemory(to: Float.self, capacity: n * k)
            for i in 0..<(n * k) { bfPtr[i] = QWBF16.fromFloatRNE(wfPtr[i]) }
            let yRefF32 = try buffer(hal, [Float](repeating: 0, count: rows * n))
            let yRefBF16 = try buffer(hal, [Float](repeating: 0, count: rows * n))
            let y16F32 = try buffer(hal, [Float](repeating: 0, count: rows * n))
            let y16BF16 = try buffer(hal, [Float](repeating: 0, count: rows * n))

            let xT = QWTensorValue(region: QWMRegion(buffer: xBuf, base: 0, length: rows * k * 2),
                                   offset: 0, shape: QWTensorShape([rows * k], .f16))
            let rotT = QWTensorValue(region: QWMRegion(buffer: rotBuf, base: 0, length: rows * k * 4),
                                     offset: 0, shape: QWTensorShape([rows * k], .f32))
            let p32T = QWTensorValue(region: QWMRegion(buffer: p32Buf, base: 0, length: rows * k * 4),
                                     offset: 0, shape: QWTensorShape([rows * k], .f32))
            let p16T = QWTensorValue(region: QWMRegion(buffer: p16Buf, base: 0, length: rows * k * 2),
                                     offset: 0, shape: QWTensorShape([rows * k], .f16))

            let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
            let enc = try XCTUnwrap(cb.makeComputeCommandEncoder(dispatchType: .serial))
            try QWGemm.encodeMoEBridgeF16ToF32(enc, hal: hal, x: xT, xRot32: rotT, xPlain32: p32T,
                                               rows: rows, hidden: k, skipRot32: true, plainF16: false)
            try QWGemm.encodeMoEBridgeF16ToF32(enc, hal: hal, x: xT, xRot32: rotT, xPlain32: p16T,
                                               rows: rows, hidden: k, skipRot32: true, plainF16: true)
            enc.memoryBarrier(scope: .buffers)
            if hal.caps.isMetal4TensorCapable {
                let wfT = QWTensorValue(region: QWMRegion(buffer: wF32Buf, base: 0, length: n * k * 4),
                                        offset: 0, shape: QWTensorShape([n * k], .f32))
                let wbT = QWTensorValue(region: QWMRegion(buffer: wBF16Buf, base: 0, length: n * k * 2),
                                        offset: 0, shape: QWTensorShape([n * k], .bf16))
                let yfT = QWTensorValue(region: QWMRegion(buffer: y16F32, base: 0, length: rows * n * 4),
                                        offset: 0, shape: QWTensorShape([rows * n], .f32))
                let ybT = QWTensorValue(region: QWMRegion(buffer: y16BF16, base: 0, length: rows * n * 4),
                                        offset: 0, shape: QWTensorShape([rows * n], .f32))
                try QWGemm.encodeFnRouterGemm(enc, hal: hal, weight: wfT, x: p16T, y: yfT, rows: rows, n: n, k: k)
                try QWGemm.encodeFnRouterGemm(enc, hal: hal, weight: wbT, x: p16T, y: ybT, rows: rows, n: n, k: k)
            }
            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted(); XCTAssertNil(cb.error)
            let p32AsF16 = floats(p32Buf, rows * k).map { Float(Float16($0)) }
            assertClose(halves(p16Buf, rows * k), p32AsF16, tol: 1e-3, "plainF16 bridge basis=\(basis)")
            if hal.caps.isMetal4TensorCapable {
                let psoGemm = try hal.pipeline("qwg_router_gemm")
                _ = try runRouterKernel(hal: hal, pso: psoGemm, isNax: false,
                                        wBuf: wF32Buf, xBuf: p32Buf, outBuf: yRefF32, rows: rows, n: n, k: k, weightMode: 0)
                _ = try runRouterKernel(hal: hal, pso: psoGemm, isNax: false,
                                        wBuf: wBF16Buf, xBuf: p32Buf, outBuf: yRefBF16, rows: rows, n: n, k: k, weightMode: 1)
                assertClose(floats(y16F32, rows * n), floats(yRefF32, rows * n), tol: 5e-4, "plain16 F32 router basis=\(basis)")
                assertClose(floats(y16BF16, rows * n), floats(yRefBF16, rows * n), tol: 2e-3, "plain16 BF16 router basis=\(basis)")
            }
        }
    }
}
