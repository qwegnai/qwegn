import XCTest
import Metal
@testable import QWGEMM
import QWCore
import QWHAL
import QWAlloc

final class QWBatchedGemmTests: XCTestCase {
    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    override func setUpWithError() throws {
        try XCTSkipUnless(Self.hal.caps.isMetal4TensorCapable, "Batched NAX GEMM requires Metal 4 tensor hardware")
    }

    static func makeSyntheticBlocks(n: Int, k: Int, seedAdd: Int = 3) -> Data {
        let g32 = k / 32
        let totalBlocks = n * g32
        var data = Data(count: totalBlocks * 20)
        let s16 = Float16(0.01)
        let b16 = Float16(-0.08 - 1024.0 * Float(s16))
        data.withUnsafeMutableBytes { raw in
            let u16 = raw.bindMemory(to: UInt16.self)
            for b in 0..<totalBlocks {
                let base = b * 10
                u16[base + 0] = s16.bitPattern
                u16[base + 1] = b16.bitPattern
                for w in 2..<10 {
                    let lo = UInt16(((b * 8 + w) * 17 + seedAdd) & 0xFF)
                    let hi = UInt16(((b * 8 + w) * 29 + seedAdd + 7) & 0xFF)
                    u16[base + w] = lo | (hi << 8)
                }
            }
        }
        return data
    }

    func testBatchedGemmNumericalParityVsF1() throws {
        let K = 5120
        let N = 256
        let M = 8

        let G32 = K / 32
        let G128 = K / 128
        let blockBytes = N * G32 * 20

        let arena = try QWArena(hal: Self.hal,
                                weights: blockBytes + 1024,
                                activations: 1024 * 1024,
                                kv: 0, gdnStates: 0)

        let wreg = arena.weightsRegion()
        let act = arena.activationsRegion()

        let blkV = try wreg.allocate(QWTensorShape([N * G32 * 10], .f16))
        arena.copyIn(blkV, from: Self.makeSyntheticBlocks(n: N, k: K))

        // Fill X [M, K]
        let xV = try act.allocate(QWTensorShape([M * K], .f16))
        var xData = Data(count: M * K * 2)
        xData.withUnsafeMutableBytes { buf in
            let ptr = buf.bindMemory(to: UInt16.self)
            for i in 0..<(M * K) {
                let v = Float((i % 13) - 6) * 0.05
                ptr[i] = Float16(v).bitPattern
            }
        }
        arena.copyIn(xV, from: xData)

        // Output Y [M, N]
        let yBatched16 = try act.allocate(QWTensorShape([M * N], .f16))
        let yBatched32 = try act.allocate(QWTensorShape([M * N], .f32))

        // Run batched kernel
        let psoBatched = try Self.hal.pipeline("qwg_gemm_q4g_h16")
        guard let (cb, enc) = Self.hal.begin() else { XCTFail("cb begin"); return }

        enc.setComputePipelineState(psoBatched)
        blkV.bind(enc, index: 0)
        xV.bind(enc, index: 1)
        yBatched16.bind(enc, index: 2)
        yBatched32.bind(enc, index: 3)
        var mU = UInt32(M), nU = UInt32(N), kU = UInt32(K)
        enc.setBytes(&mU, length: 4, index: 4)
        enc.setBytes(&nU, length: 4, index: 5)
        enc.setBytes(&kU, length: 4, index: 6)
        var xStride = UInt32(K), yStride = UInt32(N)
        enc.setBytes(&xStride, length: 4, index: 7)
        enc.setBytes(&yStride, length: 4, index: 8)

        enc.dispatchThreadgroups(MTLSize(width: (N + 63) / 64, height: (M + 63) / 64, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        Self.hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error, "batched kernel error: \(String(describing: cb.error))")

        // Compute F1 reference for each row
        let yF1_16 = try act.allocate(QWTensorShape([M * N], .f16))
        let yF1_32 = try act.allocate(QWTensorShape([M * N], .f32))
        let gsV = try act.allocate(QWTensorShape([G128], .f32))

        let psoF1 = try Self.hal.pipeline("qwg_gemv_q4g_v4_h16_dual")

        for m in 0..<M {
            guard let (cbF1, encF1) = Self.hal.begin() else { XCTFail("cbF1"); return }
            let xmRow = QWTensorValue(region: xV.region, offset: xV.offset + m * K * 2, shape: QWTensorShape([K], .f16))
            let y16Row = QWTensorValue(region: yF1_16.region, offset: yF1_16.offset + m * N * 2, shape: QWTensorShape([N], .f16))
            let y32Row = QWTensorValue(region: yF1_32.region, offset: yF1_32.offset + m * N * 4, shape: QWTensorShape([N], .f32))

            encF1.setComputePipelineState(psoF1)
            blkV.bind(encF1, index: 0)
            xmRow.bind(encF1, index: 1)
            gsV.bind(encF1, index: 2)
            y32Row.bind(encF1, index: 3)
            var nU = UInt32(N), kU = UInt32(K)
            encF1.setBytes(&nU, length: 4, index: 4)
            encF1.setBytes(&kU, length: 4, index: 5)
            y16Row.bind(encF1, index: 8)
            encF1.dispatchThreadgroups(MTLSize(width: N, height: 1, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            Self.hal.end(cbF1, encF1)
            cbF1.waitUntilCompleted()
        }

        let b16Ptr = arena.cpuPointer(of: yBatched16).bindMemory(to: Float16.self, capacity: M * N)
        let b32Ptr = arena.cpuPointer(of: yBatched32).bindMemory(to: Float.self, capacity: M * N)

        let f1_16Ptr = arena.cpuPointer(of: yF1_16).bindMemory(to: Float16.self, capacity: M * N)
        let f1_32Ptr = arena.cpuPointer(of: yF1_32).bindMemory(to: Float.self, capacity: M * N)

        var maxDiff32: Float = 0
        var maxDiff16: Float = 0
        for i in 0..<(M * N) {
            let diff32 = abs(b32Ptr[i] - f1_32Ptr[i])
            let diff16 = abs(Float(b16Ptr[i]) - Float(f1_16Ptr[i]))
            if diff32 > maxDiff32 { maxDiff32 = diff32 }
            if diff16 > maxDiff16 { maxDiff16 = diff16 }
        }
        print("[testBatchedGemm] maxDiff32=\(maxDiff32), maxDiff16=\(maxDiff16)")
        XCTAssertLessThan(maxDiff16, 0.05, "maxDiff16 exceeds tolerance")
    }

    func testBatchedGemmSupportsEveryRowCountFromTwoThroughSix() throws {
        let M = 6, N = 64, K = 128
        let G32 = K / 32
        let blockBytes = N * G32 * 20
        let arena = try QWArena(hal: Self.hal, weights: blockBytes + 1024,
                                activations: 16 << 10,
                                kv: 0, gdnStates: 0)
        let weights = arena.weightsRegion(), activations = arena.activationsRegion()
        let blocks = try weights.allocate(QWTensorShape([N * G32 * 10], .f16))
        arena.copyIn(blocks, from: Self.makeSyntheticBlocks(n: N, k: K))
        let input = try activations.allocate(QWTensorShape([M * K], .f16))
        var inputData = [UInt16](repeating: 0, count: M * K)
        for i in inputData.indices { inputData[i] = Float16(Float(i % 19 - 9) / 23).bitPattern }
        inputData.withUnsafeBytes { arena.copyIn(input, from: Data($0)) }
        let partial = try activations.allocate(QWTensorShape([M * N], .f16))
        let reference = try activations.allocate(partial.shape)
        let dummy32 = try activations.allocate(QWTensorShape([M * N], .f32))
        let baseline = try Self.hal.pipeline("qwg_gemm_q4g_h16")
        do {
            let cb = try XCTUnwrap(Self.hal.queue.makeCommandBuffer())
            let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
            enc.setComputePipelineState(baseline)
            blocks.bind(enc, index: 0)
            input.bind(enc, index: 1)
            reference.bind(enc, index: 2)
            dummy32.bind(enc, index: 3)
            var m = UInt32(M), n = UInt32(N), k = UInt32(K)
            var xStride = UInt32(K), yStride = UInt32(N)
            enc.setBytes(&m, length: 4, index: 4)
            enc.setBytes(&n, length: 4, index: 5)
            enc.setBytes(&k, length: 4, index: 6)
            enc.setBytes(&xStride, length: 4, index: 7)
            enc.setBytes(&yStride, length: 4, index: 8)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
        }
        for rows in 2...6 {
            let pointer = partial.region.pointer.advanced(by: partial.offset).assumingMemoryBound(to: UInt16.self)
            for i in 0..<(M * N) { pointer[i] = 0x3555 }
            let cb = try XCTUnwrap(Self.hal.queue.makeCommandBuffer())
            let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
            enc.setComputePipelineState(baseline)
            blocks.bind(enc, index: 0)
            input.bind(enc, index: 1)
            partial.bind(enc, index: 2)
            dummy32.bind(enc, index: 3)
            var m = UInt32(rows), n = UInt32(N), k = UInt32(K)
            var xStride = UInt32(K), yStride = UInt32(N)
            enc.setBytes(&m, length: 4, index: 4)
            enc.setBytes(&n, length: 4, index: 5)
            enc.setBytes(&k, length: 4, index: 6)
            enc.setBytes(&xStride, length: 4, index: 7)
            enc.setBytes(&yStride, length: 4, index: 8)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            XCTAssertEqual(cb.status, .completed, "\(String(describing: cb.error))")
            let partialValues = partial.region.pointer.advanced(by: partial.offset).assumingMemoryBound(to: UInt16.self)
            let referenceValues = reference.region.pointer.advanced(by: reference.offset)
                .assumingMemoryBound(to: UInt16.self)
            for row in 0..<M {
                for column in 0..<N {
                    let index = row * N + column
                    if row < rows {
                        XCTAssertEqual(partialValues[index], referenceValues[index], "M=\(rows), row=\(row)")
                    } else {
                        XCTAssertEqual(partialValues[index], 0x3555, "Batched GEMM wrote beyond M=\(rows)")
                    }
                }
            }
        }
    }

    func testBatchedGemmTileVariantsMatchBM64BN64() throws {
        let M = 128, N = 128, K = 128
        let G32 = K / 32
        let blockBytes = N * G32 * 20
        let arena = try QWArena(hal: Self.hal,
                                weights: blockBytes + 1024,
                                activations: M * (K + 4 * N) * 2 + 1024,
                                kv: 0, gdnStates: 0)
        let wreg = arena.weightsRegion()
        let act = arena.activationsRegion()
        let blocks = try wreg.allocate(QWTensorShape([N * G32 * 10], .f16))
        arena.copyIn(blocks, from: Self.makeSyntheticBlocks(n: N, k: K, seedAdd: 11))

        let x = try act.allocate(QWTensorShape([M * K], .f16))
        let y64 = try act.allocate(QWTensorShape([M * N], .f16))
        let y128 = try act.allocate(QWTensorShape([M * N], .f16))
        var xData = Data(count: M * K * 2)
        xData.withUnsafeMutableBytes { raw in
            let values = raw.bindMemory(to: UInt16.self)
            for i in 0..<(M * K) {
                values[i] = Float16(Float((i % 19) - 9) * 0.025).bitPattern
            }
        }
        arena.copyIn(x, from: xData)

        func run(_ kernel: String, _ bm: Int, _ bn: Int, _ output: QWTensorValue) throws {
            let pso = try Self.hal.pipeline(kernel)
            guard let (cb, enc) = Self.hal.begin() else { XCTFail("cb begin"); return }
            blocks.bind(enc, index: 0)
            x.bind(enc, index: 1)
            output.bind(enc, index: 2)
            output.bind(enc, index: 3)
            var m = UInt32(M), n = UInt32(N), k = UInt32(K)
            enc.setBytes(&m, length: 4, index: 4)
            enc.setBytes(&n, length: 4, index: 5)
            enc.setBytes(&k, length: 4, index: 6)
            var xs = UInt32(K), ys = UInt32(N)
            enc.setBytes(&xs, length: 4, index: 7)
            enc.setBytes(&ys, length: 4, index: 8)
            enc.setComputePipelineState(pso)
            enc.dispatchThreadgroups(
                MTLSize(width: (N + bn - 1) / bn, height: (M + bm - 1) / bm, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            Self.hal.end(cb, enc)
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error, "\(kernel) command-buffer error")
        }

        try run("qwg_gemm_q4g_h16", 64, 64, y64)
        try run("qwg_gemm_q4g_h16_bm128", 128, 64, y128)
        let a = arena.cpuPointer(of: y64).bindMemory(to: Float16.self, capacity: M * N)
        let b = arena.cpuPointer(of: y128).bindMemory(to: Float16.self, capacity: M * N)
        var maxDiffBM: Float = 0
        for i in 0..<(M * N) {
            maxDiffBM = max(maxDiffBM, abs(Float(a[i]) - Float(b[i])))
        }
        XCTAssertLessThan(maxDiffBM, 0.05, "BM128 differs from BM64")
    }

    func testBatchedGemmPerformanceM64() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["QW_PERF_GATE"] == "1",
                          "perf gate — run via `make perf`, not the unit path")
        let M = 512
        let shapes: [(String, Int, Int)] = [
            ("inProj", 16480, 5120),
            ("qkvProj", 14336, 5120),
            ("gu (MLP)", 34816, 5120),
            ("out / o", 5120, 6144),
            ("down (MLP)", 5120, 17408),
        ]

        let psoBatched = try Self.hal.pipeline("qwg_gemm_q4g_h16")

        for (name, N, K) in shapes {
            let G32 = K / 32
            let blockBytes = N * G32 * 20

            let arena = try QWArena(hal: Self.hal,
                                    weights: blockBytes + 1024,
                                    activations: 128 * 1024 * 1024,
                                    kv: 0, gdnStates: 0)

            let wreg = arena.weightsRegion()
            let act = arena.activationsRegion()

            let blkV = try wreg.allocate(QWTensorShape([N * G32 * 10], .f16))
            let xV   = try act.allocate(QWTensorShape([M * K], .f16))
            let yV   = try act.allocate(QWTensorShape([M * N], .f16))

            // Warmup
            for _ in 0..<3 {
                guard let (cb, enc) = Self.hal.begin() else { return }
                enc.setComputePipelineState(psoBatched)
                blkV.bind(enc, index: 0)
                xV.bind(enc, index: 1)
                yV.bind(enc, index: 2)
                yV.bind(enc, index: 3) // dummy
                var mU = UInt32(M), nU = UInt32(N), kU = UInt32(K)
                enc.setBytes(&mU, length: 4, index: 4)
                enc.setBytes(&nU, length: 4, index: 5)
                enc.setBytes(&kU, length: 4, index: 6)
                var xStride = UInt32(K), yStride = UInt32(N)
                enc.setBytes(&xStride, length: 4, index: 7)
                enc.setBytes(&yStride, length: 4, index: 8)
                enc.dispatchThreadgroups(MTLSize(width: (N + 63) / 64, height: (M + 63) / 64, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                Self.hal.end(cb, enc)
                cb.waitUntilCompleted()
            }

            // Benchmark 10 iterations
            var timesUs: [Double] = []
            for _ in 0..<10 {
                guard let (cb, enc) = Self.hal.begin() else { return }
                enc.setComputePipelineState(psoBatched)
                blkV.bind(enc, index: 0)
                xV.bind(enc, index: 1)
                yV.bind(enc, index: 2)
                yV.bind(enc, index: 3) // dummy
                var mU = UInt32(M), nU = UInt32(N), kU = UInt32(K)
                enc.setBytes(&mU, length: 4, index: 4)
                enc.setBytes(&nU, length: 4, index: 5)
                enc.setBytes(&kU, length: 4, index: 6)
                var xStride = UInt32(K), yStride = UInt32(N)
                enc.setBytes(&xStride, length: 4, index: 7)
                enc.setBytes(&yStride, length: 4, index: 8)
                enc.dispatchThreadgroups(MTLSize(width: (N + 63) / 64, height: (M + 63) / 64, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                Self.hal.end(cb, enc)
                cb.waitUntilCompleted()
                let us = (cb.gpuEndTime - cb.gpuStartTime) * 1_000_000
                timesUs.append(us)
            }
            timesUs.sort()
            let p50 = timesUs[timesUs.count / 2]
            let tflops = Double(2 * M * N * K) / (p50 * 1e-6) / 1e12
            let bwGbps = Double(blockBytes) / (p50 * 1e-6) / 1e9
            print(String(format: "[bench M=512] %-12@ (N=%5d, K=%5d): %6.1f us | %5.2f TFLOP/s | %5.1f GB/s",
                         name, N, K, p50, tflops, bwGbps))
        }
    }
}
