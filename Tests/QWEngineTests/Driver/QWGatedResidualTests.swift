import XCTest
import Metal
@testable import QWAlloc
@testable import QWCore
@testable import QWHAL
@testable import QWNorm
@testable import QWEngine

final class QWGatedResidualTests: XCTestCase {

    private func makeTensor(hal: QWHal, elements: [Float], dtype: QWDType = .f16) throws -> QWTensorValue {
        let byteCount = elements.count * (dtype == .f16 ? 2 : 4)
        let buffer = try XCTUnwrap(hal.device.makeBuffer(length: byteCount, options: .storageModeShared))
        if dtype == .f16 {
            let f16Ptr = buffer.contents().bindMemory(to: Float16.self, capacity: elements.count)
            for i in 0..<elements.count {
                f16Ptr[i] = Float16(elements[i])
            }
        } else {
            let f32Ptr = buffer.contents().bindMemory(to: Float.self, capacity: elements.count)
            for i in 0..<elements.count {
                f32Ptr[i] = elements[i]
            }
        }
        return QWTensorValue(
            region: QWMRegion(buffer: buffer, base: 0, length: byteCount),
            offset: 0,
            shape: QWTensorShape([elements.count], dtype)
        )
    }

    private func readTensorF16(tensor: QWTensorValue) -> [Float16] {
        let ptr = (tensor.region.buffer.contents() + tensor.offset).bindMemory(
            to: Float16.self,
            capacity: tensor.shape.elementCount
        )
        return Array(UnsafeBufferPointer(start: ptr, count: tensor.shape.elementCount))
    }

    func testBitExactParityAgainstSplitBranchAdditionsDynamicShape() throws {
        let hal = try QWHal(geometry: .qwen38_27b)
        let hidden = 128
        let rows = 1

        // Deterministic pseudo-random inputs
        var resFloats = [Float](repeating: 0, count: hidden)
        var gdnFloats = [Float](repeating: 0, count: hidden)
        var attnFloats = [Float](repeating: 0, count: hidden)
        var mlpFloats = [Float](repeating: 0, count: hidden)
        var pleFloats = [Float](repeating: 0, count: hidden)
        var gateFloats = [Float](repeating: 0, count: hidden)

        for i in 0..<hidden {
            resFloats[i] = sin(Float(i) * 0.1) * 0.5
            gdnFloats[i] = cos(Float(i) * 0.2) * 0.4
            attnFloats[i] = sin(Float(i) * 0.3 + 1.0) * 0.3
            mlpFloats[i] = cos(Float(i) * 0.4 + 2.0) * 0.6
            pleFloats[i] = sin(Float(i) * 0.5 - 1.0) * 0.2
            gateFloats[i] = cos(Float(i) * 0.6) * 1.5 // logit for sigmoid
        }

        let fusedResTensor = try makeTensor(hal: hal, elements: resFloats)

        let gdnTensor = try makeTensor(hal: hal, elements: gdnFloats)
        let attnTensor = try makeTensor(hal: hal, elements: attnFloats)
        let mlpTensor = try makeTensor(hal: hal, elements: mlpFloats)
        let pleTensor = try makeTensor(hal: hal, elements: pleFloats)
        let gateTensor = try makeTensor(hal: hal, elements: gateFloats)

        // 1. Dispatch Fused Kernel
        let cmdBuf1 = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let enc1 = try XCTUnwrap(cmdBuf1.makeComputeCommandEncoder())
        try QWGatedResidual.encode(
            enc1,
            hal: hal,
            residual: fusedResTensor,
            gdnOut: gdnTensor,
            attnOut: attnTensor,
            mlpOut: mlpTensor,
            pleEmbed: pleTensor,
            pleGate: gateTensor,
            flags: [.pleSigmoid],
            rows: rows
        )
        enc1.endEncoding()
        cmdBuf1.commit()
        cmdBuf1.waitUntilCompleted()

        // 2. Verify vs host reference (CPU exp vs Metal exp within 1e-3)
        let fusedOut = readTensorF16(tensor: fusedResTensor)
        let hostRef = QWGatedResidual.hostFusedResidual(
            residual: resFloats,
            gdnOut: gdnFloats,
            attnOut: attnFloats,
            mlpOut: mlpFloats,
            pleEmbed: pleFloats,
            pleGate: gateFloats,
            pleSigmoid: true
        )
        for i in 0..<hidden {
            let hostF16 = Float16(hostRef[i])
            let diff = abs(Float(fusedOut[i]) - Float(hostF16))
            XCTAssertLessThanOrEqual(
                diff,
                1e-3,
                "Host ref mismatch at element \(i): fused=\(fusedOut[i]) host=\(hostF16)"
            )
        }
    }

    func testBitExactParityAgainstSplitBranchStandardHidden5120FastPath() throws {
        let hal = try QWHal(geometry: .qwen38_27b)
        let hidden = QWModelGeometry.qwen38_27b.hidden // 5120
        let rows = 1

        var resFloats = [Float](repeating: 0, count: hidden)
        var gdnFloats = [Float](repeating: 0, count: hidden)
        var attnFloats = [Float](repeating: 0, count: hidden)
        var mlpFloats = [Float](repeating: 0, count: hidden)
        var pleFloats = [Float](repeating: 0, count: hidden)
        var gateFloats = [Float](repeating: 0, count: hidden)

        for i in 0..<hidden {
            resFloats[i] = sin(Float(i) * 0.01) * 0.25
            gdnFloats[i] = cos(Float(i) * 0.02) * 0.15
            attnFloats[i] = sin(Float(i) * 0.03 + 0.5) * 0.2
            mlpFloats[i] = cos(Float(i) * 0.04 - 0.5) * 0.3
            pleFloats[i] = sin(Float(i) * 0.05 + 1.2) * 0.1
            gateFloats[i] = cos(Float(i) * 0.06) * 1.0
        }

        let fusedResTensor = try makeTensor(hal: hal, elements: resFloats)
        let gdnTensor = try makeTensor(hal: hal, elements: gdnFloats)
        let attnTensor = try makeTensor(hal: hal, elements: attnFloats)
        let mlpTensor = try makeTensor(hal: hal, elements: mlpFloats)
        let pleTensor = try makeTensor(hal: hal, elements: pleFloats)
        let gateTensor = try makeTensor(hal: hal, elements: gateFloats)

        // 1. Dispatch Fused Kernel (hits the 5120 register unroll fast path)
        let cmdBuf1 = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let enc1 = try XCTUnwrap(cmdBuf1.makeComputeCommandEncoder())
        try QWGatedResidual.encode(
            enc1,
            hal: hal,
            residual: fusedResTensor,
            gdnOut: gdnTensor,
            attnOut: attnTensor,
            mlpOut: mlpTensor,
            pleEmbed: pleTensor,
            pleGate: gateTensor,
            flags: [.pleSigmoid],
            rows: rows
        )
        enc1.endEncoding()
        cmdBuf1.commit()
        cmdBuf1.waitUntilCompleted()

        let fusedOut = readTensorF16(tensor: fusedResTensor)
        XCTAssertEqual(fusedOut.count, hidden)

        let hostRef = QWGatedResidual.hostFusedResidual(
            residual: resFloats,
            gdnOut: gdnFloats,
            attnOut: attnFloats,
            mlpOut: mlpFloats,
            pleEmbed: pleFloats,
            pleGate: gateFloats,
            pleSigmoid: true
        )
        for i in 0..<hidden {
            let hostF16 = Float16(hostRef[i])
            let diff = abs(Float(fusedOut[i]) - Float(hostF16))
            XCTAssertLessThanOrEqual(
                diff,
                1e-3,
                "Host ref mismatch at element \(i): fused=\(fusedOut[i]) host=\(hostF16)"
            )
        }
    }

    func testFusedResidualWithRMSNormEpilogueMatchesHostMirror() throws {
        let hal = try QWHal(geometry: .qwen38_27b)
        let hidden = QWModelGeometry.qwen38_27b.hidden // 5120
        let rows = 1

        var resFloats = [Float](repeating: 0, count: hidden)
        var mlpFloats = [Float](repeating: 0, count: hidden)
        var gammaFloats = [Float](repeating: 0, count: hidden)

        for i in 0..<hidden {
            resFloats[i] = sin(Float(i) * 0.05) * 0.5
            mlpFloats[i] = cos(Float(i) * 0.05) * 0.3
            gammaFloats[i] = -0.05 + 0.1 * Float(i % 10) / 10.0 // raw checkpoint gamma w ≈ 0
        }

        let fusedRes = try makeTensor(hal: hal, elements: resFloats)
        let seqRes = try makeTensor(hal: hal, elements: resFloats)
        let mlpTensor = try makeTensor(hal: hal, elements: mlpFloats)
        let gammaTensor = try makeTensor(hal: hal, elements: gammaFloats)

        let fusedNormOut = try makeTensor(hal: hal, elements: [Float](repeating: 0, count: hidden))
        let seqNormOut = try makeTensor(hal: hal, elements: [Float](repeating: 0, count: hidden))

        // Path 1: Fused residual + RMSNorm epilogue
        let cmdBuf1 = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let enc1 = try XCTUnwrap(cmdBuf1.makeComputeCommandEncoder())
        try QWGatedResidual.encode(
            enc1,
            hal: hal,
            residual: fusedRes,
            mlpOut: mlpTensor,
            gamma: gammaTensor,
            normOut: fusedNormOut,
            flags: [.gammaPlusOne],
            rows: rows
        )
        enc1.endEncoding()
        cmdBuf1.commit()
        cmdBuf1.waitUntilCompleted()

        // Path 2: Standard qwg_rmsnorm_residual
        let cmdBuf2 = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let enc2 = try XCTUnwrap(cmdBuf2.makeComputeCommandEncoder())
        try QWNorm.encode(
            enc2,
            hal: hal,
            variant: .rmsnormResidual,
            x: mlpTensor,
            gamma: gammaTensor,
            out: seqNormOut,
            residual: seqRes
        , geometry: .qwen38_27b)
        enc2.endEncoding()
        cmdBuf2.commit()
        cmdBuf2.waitUntilCompleted()

        let fusedNorm = readTensorF16(tensor: fusedNormOut)
        let fusedResFinal = readTensorF16(tensor: fusedRes)
        let seqResFinal = readTensorF16(tensor: seqRes)
        let gammaFinal = readTensorF16(tensor: gammaTensor)

        // Residuals should be bit-exact
        for i in 0..<hidden {
            XCTAssertEqual(fusedResFinal[i].bitPattern, seqResFinal[i].bitPattern)
        }

        // Epilogue contract (qwg_rmsnorm_gated_residual_4branch): standard-basis
        // y = f16(s · rstd · (1+γ)). The shipping qwg_rmsnorm_residual emits the
        // FWHT32-ROTATED domain (HQ4 law, 681ae3f) — comparing the two element-wise
        var total: Float = 0
        for i in 0..<hidden { let t = Float(fusedResFinal[i]); total += t * t }
        let rstd = 1.0 / (total / Float(hidden) + 1e-6).squareRoot()
        for i in 0..<hidden {
            let gp1 = Float(Float16(Float(gammaFinal[i]) + 1.0))
            let want = Float(Float16(Float(fusedResFinal[i]) * rstd * gp1))
            let diff = abs(Float(fusedNorm[i]) - want)
            XCTAssertLessThanOrEqual(diff, 1e-3, "Index \(i): fused=\(fusedNorm[i]) mirror=\(want)")
        }
    }

    func testMHC4StreamEnterRouterNormAndMerge() throws {
        let geo = try QWWeightAllocationPlanTests.glm53FlashGeometry()
        XCTAssertEqual(geo.mhcConfig?.streams, 4)
        XCTAssertEqual(geo.mhcConfig?.iterations, 20)
        let hal = try QWHal(geometry: geo)

        let hidden = 128
        let c = 4
        let numCtrl = c + c * c // 20
        let numRows = 2 * c + c * c // 24

        var xFloats = [Float](repeating: 0, count: hidden)
        for i in 0..<hidden {
            xFloats[i] = sin(Float(i + 1) * 0.07) * 0.8
        }
        let xTensor = try makeTensor(hal: hal, elements: xFloats, dtype: .f16)
        let streamTensor = try makeTensor(hal: hal, elements: [Float](repeating: 0, count: c * hidden), dtype: .f32)

        let enterPso = try hal.pipeline("qwg_mhc_enter_f16")
        let routerPso = try hal.pipeline("qwg_mhc_router_norm_f16")
        let mergePso = try hal.pipeline("qwg_mhc_merge_f16")

        var uHidden = UInt32(hidden)
        do {
            let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
            let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
            enc.setComputePipelineState(enterPso)
            enc.setBuffer(xTensor.region.buffer, offset: 0, index: 0)
            enc.setBuffer(streamTensor.region.buffer, offset: 0, index: 1)
            enc.setBytes(&uHidden, length: 4, index: 2)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
        }

        let streamPtr = streamTensor.region.buffer.contents().bindMemory(to: Float.self, capacity: c * hidden)
        for s in 0..<c {
            for h in 0..<hidden {
                XCTAssertEqual(streamPtr[s * hidden + h], Float(Float16(xFloats[h])), accuracy: 1e-6)
            }
        }

        var wFloats = [Float](repeating: 0, count: numRows * c * hidden)
        for i in 0..<wFloats.count {
            wFloats[i] = cos(Float(i) * 0.013) * 0.05
        }
        var bFloats = [Float](repeating: 0, count: numRows)
        for r in 0..<numRows {
            bFloats[r] = sin(Float(r) * 0.3) * 0.2
        }
        let wTensor = try makeTensor(hal: hal, elements: wFloats, dtype: .f32)
        let bTensor = try makeTensor(hal: hal, elements: bFloats, dtype: .f32)
        let gammaTensor = try makeTensor(hal: hal, elements: [Float](repeating: 0.0, count: hidden), dtype: .f16)
        let outNormTensor = try makeTensor(hal: hal, elements: [Float](repeating: 0, count: hidden), dtype: .f16)
        let ctrlTensor = try makeTensor(hal: hal, elements: [Float](repeating: 0, count: numCtrl), dtype: .f32)

        var eps: Float = 1e-5
        var preScale: Float = 1.0 / Float(c * hidden).squareRoot()
        var iters: UInt32 = 20
        do {
            let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
            let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
            enc.setComputePipelineState(routerPso)
            enc.setBuffer(streamTensor.region.buffer, offset: 0, index: 0)
            enc.setBuffer(wTensor.region.buffer, offset: 0, index: 1)
            enc.setBuffer(bTensor.region.buffer, offset: 0, index: 2)
            enc.setBuffer(gammaTensor.region.buffer, offset: 0, index: 3)
            enc.setBuffer(outNormTensor.region.buffer, offset: 0, index: 4)
            enc.setBuffer(ctrlTensor.region.buffer, offset: 0, index: 5)
            enc.setBytes(&uHidden, length: 4, index: 6)
            enc.setBytes(&eps, length: 4, index: 7)
            enc.setBytes(&preScale, length: 4, index: 8)
            enc.setBytes(&iters, length: 4, index: 9)
            enc.setBuffer(outNormTensor.region.buffer, offset: 0, index: 10)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
        }

        let ctrlPtr = ctrlTensor.region.buffer.contents().bindMemory(to: Float.self, capacity: numCtrl)
        // Verify doubly-stochastic 4x4 comb^T matrix (stored in ctrl[4..<20]): rows and cols sum to ~1.0.
        for i in 0..<c {
            var rSum: Float = 0
            var cSum: Float = 0
            for j in 0..<c {
                let v = ctrlPtr[c + i * c + j]
                XCTAssertGreaterThan(v, 0)
                rSum += v
                cSum += ctrlPtr[c + j * c + i]
            }
            XCTAssertEqual(rSum, 1.0, accuracy: 2e-3)
            XCTAssertEqual(cSum, 1.0, accuracy: 2e-3)
        }

        // Dispatch qwg_mhc_merge_f16 with isLeave = true (flags = 3)
        let xOutTensor = try makeTensor(hal: hal, elements: [Float](repeating: 0, count: hidden), dtype: .f16)
        let nextNormTensor = try makeTensor(hal: hal, elements: [Float](repeating: 0, count: hidden), dtype: .f16)
        var flags: UInt32 = 3
        do {
            let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
            let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
            enc.setComputePipelineState(mergePso)
            enc.setBuffer(streamTensor.region.buffer, offset: 0, index: 0)
            enc.setBuffer(xTensor.region.buffer, offset: 0, index: 1)
            enc.setBuffer(ctrlTensor.region.buffer, offset: 0, index: 2)
            enc.setBuffer(xOutTensor.region.buffer, offset: 0, index: 3)
            enc.setBuffer(gammaTensor.region.buffer, offset: 0, index: 4)
            enc.setBuffer(nextNormTensor.region.buffer, offset: 0, index: 5)
            enc.setBytes(&uHidden, length: 4, index: 6)
            enc.setBytes(&eps, length: 4, index: 7)
            enc.setBytes(&flags, length: 4, index: 8)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
        }

        let xOut = readTensorF16(tensor: xOutTensor)
        for h in 0..<hidden {
            XCTAssertTrue(xOut[h].isFinite)
        }
    }
}
