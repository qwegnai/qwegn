import Foundation
import Metal
import XCTest
import QWAlloc
import QWCore
import QWGEMM
import QWHAL
@testable import QWEngine

final class QWMTPBatchedMLPTests: XCTestCase {
    func testUnifiedBatchPipelinesSupportRowsTwoThroughSix() throws {
        let hal = try QWHal(geometry: .qwen38_27b)
        for rows in 2...16 {
            let arena = try QWArena(hal: hal, weights: 256, activations: 4 << 20,
                                    kv: 256, gdnStates: 256)
            let allocator = arena.activationsRegion()
            _ = try QWMTPBatchedMLP(geometry: QWTestGeometry.g27, hal: hal, allocator: allocator, rows: rows)
            _ = try QWMTPBatchedTrunk(geometry: QWTestGeometry.g27, hal: hal, allocator: allocator, rows: rows)
            XCTAssertEqual(allocator.usedBytes,
                           QWMTPBatchedMLP.allocationBytes(geometry: QWTestGeometry.g27, rows: rows) +
                           QWMTPBatchedTrunk.allocationBytes(geometry: QWTestGeometry.g27, rows: rows))
        }
    }

    func testExactShippingShapesAndRefusalBeforeDispatch() throws {
        let hal = try QWHal(geometry: .qwen38_27b)
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Scripts/spike/out")
        func tensor(_ data: Data, dtype: QWDType = .f16) throws -> QWTensorValue {
            let buffer = try XCTUnwrap(hal.device.makeBuffer(length: data.count, options: .storageModeShared))
            data.withUnsafeBytes { buffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
            return QWTensorValue(region: QWMRegion(buffer: buffer, base: 0, length: data.count),
                                 offset: 0, shape: QWTensorShape([data.count / (dtype == .f16 ? 2 : 4)], dtype))
        }
        func weight(_ name: String, n: Int, k: Int) throws -> QWWeightView {
            func load(_ suffix: String) throws -> QWTensorValue {
                try tensor(Data(contentsOf: root.appendingPathComponent(name + suffix)))
            }
            return try XCTUnwrap(QWWeightView(blocks: try load(".bin"), n: n, k: k, isRotated: true))
        }
        let gu = try weight("mlp_gu", n: 2 * QWModelGeometry.qwen38_27b.intermediate, k: QWModelGeometry.qwen38_27b.hidden)
        let dn = try weight("mlp_down", n: QWModelGeometry.qwen38_27b.hidden, k: QWModelGeometry.qwen38_27b.intermediate)
        func slice(_ value: QWTensorValue, row: Int, columns: Int) -> QWTensorValue {
            QWTensorValue(region: value.region, offset: value.offset + row * columns * 2,
                          shape: QWTensorShape([columns], .f16))
        }
        func reference(_ enc: any MTLComputeCommandEncoder, input: QWTensorValue,
                       sums: QWTensorValue, weight: QWWeightView, output: QWTensorValue, paired: Bool) throws {
            enc.setComputePipelineState(try hal.pipeline(
                paired ? "qwg_gemv_q4g_silu_pair_dx_h16" : "qwg_gemv_q4g_dx_h16_t4"))
            weight.blocks.bind(enc, index: 0); input.bind(enc, index: 1); sums.bind(enc, index: 2)
            output.bind(enc, index: 3)
            var n = UInt32(weight.n), k = UInt32(weight.k)
            enc.setBytes(&n, length: 4, index: 4); enc.setBytes(&k, length: 4, index: 5)
            output.bind(enc, index: 8)
            enc.dispatchThreadgroups(MTLSize(width: paired ? weight.n / 2 : weight.n, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: paired ? 64 : 32, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)
        }
        for rows in [2, 3, 4, 5, 6, 12, 16] {
            let arena = try QWArena(hal: hal, weights: 256, activations: 4 << 20, kv: 256, gdnStates: 256)
            let allocator = arena.activationsRegion()
            let batch = try QWMTPBatchedMLP(geometry: QWTestGeometry.g27, hal: hal, allocator: allocator, rows: rows)
            XCTAssertEqual(allocator.usedBytes, QWMTPBatchedMLP.allocationBytes(geometry: QWTestGeometry.g27, rows: rows))
            let beforeTrunk = allocator.usedBytes
            let trunk = try QWMTPBatchedTrunk(geometry: QWTestGeometry.g27, hal: hal, allocator: allocator, rows: rows)
            XCTAssertEqual(allocator.usedBytes - beforeTrunk, QWMTPBatchedTrunk.allocationBytes(geometry: QWTestGeometry.g27, rows: rows))
            let input: QWTensorValue
            if rows == 2 || rows == 3 {
                input = try tensor(Data(contentsOf: root.appendingPathComponent(
                    rows == 2 ? "mlp_x_M2.bin" : "f5gu_x_M3.bin")))
            } else {
                let values = (0..<(rows * QWModelGeometry.qwen38_27b.hidden)).map {
                    Float16(Float(($0 * 17 + 11) % 61 - 30) / 32)
                }
                input = try tensor(values.withUnsafeBytes { Data($0) })
            }
            let h = try allocator.allocate(QWTensorShape([rows * QWModelGeometry.qwen38_27b.intermediate], .f16))
            if ProcessInfo.processInfo.environment["QW_MTP_MLP_BENCH"] == "1" {
                var samples: [Double] = []
                for iteration in 0..<7 {
                    let buffer = try XCTUnwrap(hal.queue.makeCommandBuffer())
                    let encoder = try XCTUnwrap(buffer.makeComputeCommandEncoder())
                    do {
                        for _ in 0..<8 {
                            try batch.encodeGateUp(encoder, input: input, weight: gu, output: h)
                        }
                    } catch {
                        encoder.endEncoding()
                        throw error
                    }
                    encoder.endEncoding()
                    buffer.commit()
                    buffer.waitUntilCompleted()
                    XCTAssertEqual(buffer.status, .completed, "\(String(describing: buffer.error))")
                    let ms = try QWMTPStageTiming.gpuDurationMs(buffer) / 8
                    if iteration >= 2 { samples.append(ms) }
                }
                print("MTP_GATE_UP_MICRO B\(rows) medianMs=\(samples.sorted()[samples.count / 2]) samples=\(samples)")
            }
            let hRef = try allocator.allocate(h.shape)
            let yRef = try allocator.allocate(QWTensorShape([rows * QWModelGeometry.qwen38_27b.hidden], .f16))
            let gs = try allocator.allocate(QWTensorShape([QWModelGeometry.qwen38_27b.hidden / 128], .f32))
            let hs = try allocator.allocate(QWTensorShape([QWModelGeometry.qwen38_27b.intermediate / 128], .f32))
            let projectionRef = try allocator.allocate(trunk.gdnProjection.shape)
            let parametersRef = try allocator.allocate(trunk.controls.shape)
            let aLog = try allocator.allocate(QWTensorShape([48], .f32))
            let bias = try allocator.allocate(aLog.shape)
            let y32 = try allocator.allocate(QWTensorShape([QWModelGeometry.qwen38_27b.gdnTotalN], .f32))
            let a = aLog.region.pointer.advanced(by: aLog.offset).assumingMemoryBound(to: Float.self)
            let b = bias.region.pointer.advanced(by: bias.offset).assumingMemoryBound(to: Float.self)
            for i in 0..<48 { a[i] = Float(i % 7) / 3 - 2; b[i] = Float(i % 5) / 10 - 0.2 }
            allocator.markSteadyState()
            for scenario in 0..<3 {
                if scenario > 0 {
                    memset(input.region.pointer, 0, input.bytes)
                    if scenario == 2 {
                        let values = input.region.pointer.assumingMemoryBound(to: Float16.self)
                        for row in 0..<rows {
                            values[row * QWModelGeometry.qwen38_27b.hidden] = Float16(row + 1) * 16
                            values[(row + 1) * QWModelGeometry.qwen38_27b.hidden - 1] = -16
                            values[row * QWModelGeometry.qwen38_27b.hidden + 1] = .leastNonzeroMagnitude
                        }
                    }
                }
                let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
                let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
                do {
                    try batch.encodeGateUp(enc, input: input, weight: gu, output: h)
                    try batch.encodeDown(enc, input: h, weight: dn)
                    for row in 0..<rows {
                        try reference(enc, input: slice(input, row: row, columns: QWModelGeometry.qwen38_27b.hidden),
                            sums: gs, weight: gu, output: slice(hRef, row: row, columns: QWModelGeometry.qwen38_27b.intermediate),
                            paired: true)
                        try reference(enc, input: slice(hRef, row: row, columns: QWModelGeometry.qwen38_27b.intermediate),
                            sums: hs, weight: dn, output: slice(yRef, row: row, columns: QWModelGeometry.qwen38_27b.hidden),
                            paired: false)
                    }
                } catch {
                    enc.endEncoding()
                    throw error
                }
                enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                XCTAssertEqual(cb.status, .completed, "\(String(describing: cb.error))")
                let gotH = Data(bytes: h.region.pointer.advanced(by: h.offset), count: h.bytes)
                let expectedH = Data(bytes: hRef.region.pointer.advanced(by: hRef.offset), count: hRef.bytes)
                let gotY = Data(bytes: batch.downRows.region.pointer.advanced(by: batch.downRows.offset),
                                count: batch.downRows.bytes)
                let expectedY = Data(bytes: yRef.region.pointer.advanced(by: yRef.offset), count: yRef.bytes)
                XCTAssertEqual(gotH, expectedH, "B\(rows) gate/up scenario \(scenario)")
                XCTAssertEqual(gotY, expectedY, "B\(rows) T4 down scenario \(scenario)")
                for gdn in [true, false] {
                    let columns = gdn ? QWModelGeometry.qwen38_27b.gdnTotalN : QWModelGeometry.qwen38_27b.attnTotalN
                    let weight = try XCTUnwrap(QWWeightView(blocks: gu.blocks,
                                                            n: columns, k: QWModelGeometry.qwen38_27b.hidden, isRotated: true))
                    let output = gdn ? trunk.gdnProjection : trunk.attentionProjection
                    let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
                    let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
                    do {
                        if gdn {
                            try batch.encodeGDNProjection(enc, input: input, weight: weight, output: output,
                                controls: QWMTPBatchedMLP.GDNControls(aLog: aLog, bias: bias, output: trunk.controls),
                                trunk: trunk)
                        } else {
                            try batch.encodeProjection(enc, input: input, weight: weight, output: output)
                        }
                        for row in 0..<rows {
                            let x = slice(input, row: row, columns: QWModelGeometry.qwen38_27b.hidden)
                            let y = slice(projectionRef, row: row, columns: columns)
                            if !gdn {
                                try reference(enc, input: x, sums: gs, weight: weight, output: y, paired: false)
                                continue
                            }
                            enc.setComputePipelineState(try hal.pipeline("qwg_gemv_q4g_dx_h16_gdn_inproj_t4"))
                            weight.blocks.bind(enc, index: 0); x.bind(enc, index: 1); gs.bind(enc, index: 2)
                            y32.bind(enc, index: 3)
                            var n = UInt32(columns), k = UInt32(QWModelGeometry.qwen38_27b.hidden)
                            enc.setBytes(&n, length: 4, index: 4); enc.setBytes(&k, length: 4, index: 5)
                            y.bind(enc, index: 8); aLog.bind(enc, index: 9); bias.bind(enc, index: 10)
                            enc.setBuffer(parametersRef.region.buffer,
                                offset: parametersRef.region.base + parametersRef.offset + row * 96 * 4, index: 11)
                            enc.setBuffer(parametersRef.region.buffer,
                                offset: parametersRef.region.base + parametersRef.offset + (row * 96 + 48) * 4, index: 12)
                            enc.dispatchThreadgroups(MTLSize(width: columns, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                            enc.memoryBarrier(scope: .buffers)
                        }
                    } catch {
                        enc.endEncoding()
                        throw error
                    }
                    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                    XCTAssertEqual(cb.status, .completed, "\(String(describing: cb.error))")
                    XCTAssertEqual(Data(bytes: output.region.pointer.advanced(by: output.offset), count: output.bytes),
                        Data(bytes: projectionRef.region.pointer.advanced(by: projectionRef.offset), count: output.bytes),
                        "B\(rows) \(gdn ? "GDN" : "attention") projection scenario \(scenario)")
                    if gdn {
                        XCTAssertEqual(
                            Data(bytes: trunk.controls.region.pointer.advanced(by: trunk.controls.offset),
                                 count: trunk.controls.bytes),
                            Data(bytes: parametersRef.region.pointer.advanced(by: parametersRef.offset),
                                 count: parametersRef.bytes), "B\(rows) FP32 beta/decay scenario \(scenario)")
                    }
                }
            }
            let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
            let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
            XCTAssertThrowsError(try batch.encodeGateUp(enc, input: h, weight: gu, output: h))
            XCTAssertThrowsError(try batch.encodeDown(enc, input: input, weight: dn))
            let alias = QWTensorValue(region: h.region, offset: h.offset, shape: input.shape)
            XCTAssertThrowsError(try batch.encodeGateUp(enc, input: alias, weight: gu, output: h))
            let short = QWTensorValue(region: QWMRegion(buffer: input.region.buffer, base: 0, length: 2),
                                       offset: 0, shape: input.shape)
            XCTAssertThrowsError(try batch.encodeGateUp(enc, input: short, weight: gu, output: h))
            let extreme = QWTensorValue(region: trunk.controls.region, offset: Int.max - 8, shape: input.shape)
            let gdnWeight = try XCTUnwrap(QWWeightView(blocks: gu.blocks,
                n: QWModelGeometry.qwen38_27b.gdnTotalN, k: QWModelGeometry.qwen38_27b.hidden, isRotated: true))
            let controls = QWMTPBatchedMLP.GDNControls(aLog: aLog, bias: bias, output: trunk.controls)
            XCTAssertThrowsError(try batch.encodeGDNProjection(enc, input: extreme, weight: gdnWeight,
                output: trunk.gdnProjection, controls: controls, trunk: trunk))
            let extremeOutput = QWTensorValue(region: trunk.controls.region, offset: Int.max - 8,
                                              shape: trunk.gdnProjection.shape)
            XCTAssertThrowsError(try batch.encodeGDNProjection(enc, input: input, weight: gdnWeight,
                output: extremeOutput, controls: controls, trunk: trunk))
            let hugeShape = QWTensorValue(region: input.region, offset: 0,
                                          shape: QWTensorShape([Int.max, Int.max], .f16))
            XCTAssertThrowsError(try batch.encodeGDNProjection(enc, input: hugeShape, weight: gdnWeight,
                output: trunk.gdnProjection, controls: .init(aLog: aLog, bias: bias, output: trunk.controls),
                trunk: trunk))
            XCTAssertThrowsError(try batch.downRow(-1))
            XCTAssertThrowsError(try batch.downRow(rows))
            XCTAssertThrowsError(try batch.outputSumRow(-1))
            XCTAssertThrowsError(try trunk.gdnRow(rows))
            XCTAssertThrowsError(try trunk.attentionRow(-1))
            XCTAssertThrowsError(try trunk.outputRow(rows))
            enc.endEncoding()
            XCTAssertEqual(allocator.steadyStateAllocs, 0)
        }
    }

    func testNonIncumbentHQ4AndAQ4RowPairedAndFwht32BitExactParity() throws {
        let hal = try QWHal(geometry: .qwen38_27b)
        let geom = QWModelGeometry.qwen38_27b
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Scripts/spike/out")
        func tensor(_ data: Data, dtype: QWDType = .f16) throws -> QWTensorValue {
            let buffer = try XCTUnwrap(hal.device.makeBuffer(length: data.count, options: .storageModeShared))
            data.withUnsafeBytes { buffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
            return QWTensorValue(region: QWMRegion(buffer: buffer, base: 0, length: data.count),
                                 offset: 0, shape: QWTensorShape([data.count / (dtype == .f16 ? 2 : 4)], dtype))
        }
        let guBlocks = try tensor(Data(contentsOf: root.appendingPathComponent("mlp_gu.bin")))
        let dnBlocks = try tensor(Data(contentsOf: root.appendingPathComponent("mlp_down.bin")))

        for isRotated in [true, false] {
            let gu = try XCTUnwrap(QWWeightView(blocks: guBlocks, n: 2 * geom.intermediate, k: geom.hidden,
                                                bits: 4, groupSize: 32, isRotated: isRotated,
                                                isIncumbentContainer: false))
            let dn = try XCTUnwrap(QWWeightView(blocks: dnBlocks, n: geom.hidden, k: geom.intermediate,
                                                bits: 4, groupSize: 32, isRotated: isRotated,
                                                isIncumbentContainer: false))
            XCTAssertFalse(gu.isIncumbentContainer)
            XCTAssertFalse(QWMTPBatchedMLP.usesRowPairs(gu, activeRows: 2))
            XCTAssertTrue(QWMTPBatchedMLP.usesRowPairs(gu, activeRows: 4))

            let arena = try QWArena(hal: hal, weights: 256, activations: 8 << 20, kv: 256, gdnStates: 256)
            let alloc = arena.activationsRegion()
            let batch = try QWMTPBatchedMLP(geometry: QWTestGeometry.g27, hal: hal, allocator: alloc, rows: 5)
            let vals = (0..<(5 * geom.hidden)).map { Float16(Float(($0 * 17 + 11) % 61 - 30) / 32) }
            let input5 = try tensor(vals.withUnsafeBytes { Data($0) })
            let input2 = QWTensorValue(region: input5.region, offset: 0, shape: QWTensorShape([2 * geom.hidden], .f16))
            let h5 = try alloc.allocate(QWTensorShape([5 * geom.intermediate], .f16))
            let h2 = try alloc.allocate(QWTensorShape([2 * geom.intermediate], .f16))
            let hFwht5 = try alloc.allocate(QWTensorShape([5 * geom.intermediate], .f16))
            let down2Ref = try alloc.allocate(QWTensorShape([2 * geom.hidden], .f16))

            let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
            let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
            // Non-row-paired M=2 vs row-paired M=5 must be bit-identical on the first 2 rows.
            try batch.encodeGateUp(enc, input: input2, weight: gu, output: h2, activeRows: 2, fuseFwht32: false)
            try batch.encodeGateUp(enc, input: input5, weight: gu, output: h5, activeRows: 5, fuseFwht32: false)
            try batch.encodeDown(enc, input: h2, weight: dn, activeRows: 2)
            // Fused fwht32 on M=5 must equal unfused M=5 followed by qwg_fwht32_f16.
            try batch.encodeGateUp(enc, input: input5, weight: gu, output: hFwht5, activeRows: 5, fuseFwht32: true)
            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            XCTAssertEqual(cb.status, .completed)

            let gotH2 = Data(bytes: h2.region.pointer.advanced(by: h2.offset), count: h2.bytes)
            let gotH5First2 = Data(bytes: h5.region.pointer.advanced(by: h5.offset), count: h2.bytes)
            XCTAssertEqual(gotH2, gotH5First2, "row-paired gateUp M=5 vs M=2 parity failed for isRotated=\(isRotated)")

            // Save M=2 down output before running M=5 down on unmutated h5.
            memcpy(down2Ref.region.pointer.advanced(by: down2Ref.offset),
                   batch.downRows.region.pointer.advanced(by: batch.downRows.offset), down2Ref.bytes)
            let cb2 = try XCTUnwrap(hal.queue.makeCommandBuffer())
            let enc2 = try XCTUnwrap(cb2.makeComputeCommandEncoder())
            try batch.encodeDown(enc2, input: h5, weight: dn, activeRows: 5)
            let copyPso = try hal.pipeline("qwg_fwht32_f16")
            enc2.setComputePipelineState(copyPso)
            h5.bind(enc2, index: 0)
            var countU = UInt32(5 * geom.intermediate)
            enc2.setBytes(&countU, length: 4, index: 1)
            enc2.dispatchThreadgroups(MTLSize(width: Int(countU) / 32, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            enc2.endEncoding(); cb2.commit(); cb2.waitUntilCompleted()

            let gotDown2 = Data(bytes: down2Ref.region.pointer.advanced(by: down2Ref.offset), count: down2Ref.bytes)
            let gotDown5First2 = Data(bytes: batch.downRows.region.pointer.advanced(by: batch.downRows.offset),
                                      count: down2Ref.bytes)
            XCTAssertEqual(gotDown2, gotDown5First2, "row-paired down M=5 vs M=2 parity failed for isRotated=\(isRotated)")

            let gotFwht5 = Data(bytes: hFwht5.region.pointer.advanced(by: hFwht5.offset), count: hFwht5.bytes)
            let refFwht5 = Data(bytes: h5.region.pointer.advanced(by: h5.offset), count: h5.bytes)
            XCTAssertEqual(gotFwht5, refFwht5, "4-row gateUpFwht32 parity failed for isRotated=\(isRotated)")
        }
    }

    func testMultiRowAQAllBitWidthsAndGroupsParity() throws {
        let hal = try QWHal(geometry: .qwen38_27b)
        let geom = QWModelGeometry.qwen38_27b
        func tensor(_ data: Data, dtype: QWDType = .f16) throws -> QWTensorValue {
            let buffer = try XCTUnwrap(hal.device.makeBuffer(length: data.count, options: .storageModeShared))
            data.withUnsafeBytes { buffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
            return QWTensorValue(region: QWMRegion(buffer: buffer, base: 0, length: data.count),
                                 offset: 0, shape: QWTensorShape([data.count / (dtype == .f16 ? 2 : 4)], dtype))
        }
        // 8-bit G32 needs 36B per 32 elements; max N = 2 * geom.intermediate (34,816), K = geom.hidden (5,120).
        // Every 16-bit word lies in 0x2800...0x2FFF (finite normal FP16 in [0.03125, 0.125]) so both
        // (scale, bias) headers at any group boundary and packed bit payloads are well-conditioned.
        let maxBlocksBytes = 2 * geom.intermediate * (geom.hidden / 32) * 36
        var words = [UInt16](repeating: 0, count: maxBlocksBytes / 2)
        for i in 0..<words.count {
            words[i] = 0x2800 | UInt16((i &* 131 &+ 17) & 0x07FF)
        }
        let fullData = words.withUnsafeBytes { Data($0) }
        let rawBlocks = try tensor(fullData)
        let arena = try QWArena(hal: hal, weights: 256, activations: 16 << 20, kv: 256, gdnStates: 256)
        let alloc = arena.activationsRegion()
        let batch = try QWMTPBatchedMLP(geometry: QWTestGeometry.g27, hal: hal, allocator: alloc, rows: 4)
        let trunk = try QWMTPBatchedTrunk(geometry: QWTestGeometry.g27, hal: hal, allocator: alloc, rows: 4)
        let vals = (0..<(4 * geom.hidden)).map { Float16(Float(($0 * 17 + 11) % 61 - 30) / 32) }
        let input4 = try tensor(vals.withUnsafeBytes { Data($0) })
        let aLog = try alloc.allocate(QWTensorShape([48], .f32))
        let bias = try alloc.allocate(aLog.shape)
        let aPtr = aLog.region.pointer.advanced(by: aLog.offset).assumingMemoryBound(to: Float.self)
        let bPtr = bias.region.pointer.advanced(by: bias.offset).assumingMemoryBound(to: Float.self)
        for i in 0..<48 { aPtr[i] = Float(i % 7) / 3 - 2; bPtr[i] = Float(i % 5) / 10 - 0.2 }

        let outProjBuf = try alloc.allocate(QWTensorShape([4 * geom.attnTotalN], .f16))
        let refProjBuf = try alloc.allocate(QWTensorShape([4 * geom.attnTotalN], .f16))
        let outGuBuf = try alloc.allocate(QWTensorShape([4 * geom.intermediate], .f16))
        let refGuBuf = try alloc.allocate(QWTensorShape([4 * geom.intermediate], .f16))
        let outGdnBuf = try alloc.allocate(QWTensorShape([4 * geom.gdnTotalN], .f16))
        let refGdnBuf = try alloc.allocate(QWTensorShape([4 * geom.gdnTotalN], .f16))
        let outCtrlBuf = try alloc.allocate(QWTensorShape([4 * 96], .f32))
        let refCtrlBuf = try alloc.allocate(QWTensorShape([4 * 96], .f32))

        let gs = try alloc.allocate(QWTensorShape([geom.hidden / 128], .f32))
        let y32 = try alloc.allocate(QWTensorShape([geom.gdnTotalN], .f32))

        for (bits, group) in [(2, 32), (3, 64), (6, 32), (8, 128)] {
            let projW = try XCTUnwrap(QWWeightView(blocks: rawBlocks, n: geom.attnTotalN, k: geom.hidden,
                                                   bits: bits, groupSize: group, isRotated: true,
                                                   isIncumbentContainer: false))
            let gdnW = try XCTUnwrap(QWWeightView(blocks: rawBlocks, n: geom.gdnTotalN, k: geom.hidden,
                                                  bits: bits, groupSize: group, isRotated: true,
                                                  isIncumbentContainer: false))
            let guW = try XCTUnwrap(QWWeightView(blocks: rawBlocks, n: 2 * geom.intermediate, k: geom.hidden,
                                                 bits: bits, groupSize: group, isRotated: true,
                                                 isIncumbentContainer: false))
            let psoProj1 = try hal.pipeline("qwg_gemv_aq\(bits)_g\(group)_dx_h16_t4")
            let psoGu1 = try hal.pipeline("qwg_gemv_aq\(bits)_g\(group)_silu_pair_dx_h16")
            let psoGdn1 = try hal.pipeline("qwg_gemv_aq\(bits)_g\(group)_dx_h16_gdn_inproj_t4")
            let psoFwht = try hal.pipeline("qwg_fwht32_f16")
            for m in [2, 3, 4] {
                let inM = QWTensorValue(region: input4.region, offset: 0, shape: QWTensorShape([m * geom.hidden], .f16))
                let outProjM = QWTensorValue(region: outProjBuf.region, offset: outProjBuf.offset, shape: QWTensorShape([m * geom.attnTotalN], .f16))
                let refProjM = QWTensorValue(region: refProjBuf.region, offset: refProjBuf.offset, shape: QWTensorShape([m * geom.attnTotalN], .f16))
                let outGuM = QWTensorValue(region: outGuBuf.region, offset: outGuBuf.offset, shape: QWTensorShape([m * geom.intermediate], .f16))
                let refGuM = QWTensorValue(region: refGuBuf.region, offset: refGuBuf.offset, shape: QWTensorShape([m * geom.intermediate], .f16))
                let outGdnM = QWTensorValue(region: outGdnBuf.region, offset: outGdnBuf.offset, shape: QWTensorShape([m * geom.gdnTotalN], .f16))
                let refGdnM = QWTensorValue(region: refGdnBuf.region, offset: refGdnBuf.offset, shape: QWTensorShape([m * geom.gdnTotalN], .f16))
                let outCtrl = QWTensorValue(region: outCtrlBuf.region, offset: outCtrlBuf.offset, shape: QWTensorShape([m * 96], .f32))
                let refCtrl = QWTensorValue(region: refCtrlBuf.region, offset: refCtrlBuf.offset, shape: QWTensorShape([m * 96], .f32))

                let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
                let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
                do {
                    try batch.encodeProjection(enc, input: inM, weight: projW, output: outProjM, activeRows: m)
                    try batch.encodeGateUp(enc, input: inM, weight: guW, output: outGuM, activeRows: m, fuseFwht32: true)
                    try batch.encodeGDNProjection(
                        enc, input: inM, weight: gdnW, output: outGdnM,
                        controls: .init(aLog: aLog, bias: bias, output: outCtrl),
                        trunk: trunk, activeRows: m)

                    // Compare against m separate single-row launches (M = 1).
                    for r in 0..<m {
                        let in1 = QWTensorValue(region: input4.region, offset: r * geom.hidden * 2,
                                                shape: QWTensorShape([geom.hidden], .f16))
                        let refP1 = QWTensorValue(region: refProjM.region, offset: refProjM.offset + r * geom.attnTotalN * 2,
                                                  shape: QWTensorShape([geom.attnTotalN], .f16))
                        let refGu1 = QWTensorValue(region: refGuM.region, offset: refGuM.offset + r * geom.intermediate * 2,
                                                   shape: QWTensorShape([geom.intermediate], .f16))
                        let refGdn1 = QWTensorValue(region: refGdnM.region, offset: refGdnM.offset + r * geom.gdnTotalN * 2,
                                                    shape: QWTensorShape([geom.gdnTotalN], .f16))
                        var nProj = UInt32(projW.n), kHidden = UInt32(geom.hidden), mOne: UInt32 = 1
                        enc.setComputePipelineState(psoProj1)
                        projW.blocks.bind(enc, index: 0); in1.bind(enc, index: 1); gs.bind(enc, index: 2)
                        refP1.bind(enc, index: 3)
                        enc.setBytes(&nProj, length: 4, index: 4); enc.setBytes(&kHidden, length: 4, index: 5)
                        refP1.bind(enc, index: 8)
                        enc.setBytes(&mOne, length: 4, index: 9)
                        enc.dispatchThreadgroups(MTLSize(width: projW.n, height: 1, depth: 1),
                                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

                        var nGu = UInt32(guW.n), fuseZero: UInt32 = 0
                        enc.setComputePipelineState(psoGu1)
                        guW.blocks.bind(enc, index: 0); in1.bind(enc, index: 1); gs.bind(enc, index: 2)
                        refGu1.bind(enc, index: 3)
                        enc.setBytes(&nGu, length: 4, index: 4); enc.setBytes(&kHidden, length: 4, index: 5)
                        refGu1.bind(enc, index: 8)
                        enc.setBytes(&mOne, length: 4, index: 9); enc.setBytes(&fuseZero, length: 4, index: 10)
                        enc.dispatchThreadgroups(MTLSize(width: guW.n / 2, height: 1, depth: 1),
                                                 threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                        enc.memoryBarrier(scope: .buffers)
                        var countGu = UInt32(geom.intermediate)
                        enc.setComputePipelineState(psoFwht)
                        refGu1.bind(enc, index: 0); enc.setBytes(&countGu, length: 4, index: 1)
                        enc.dispatchThreadgroups(MTLSize(width: geom.intermediate / 32, height: 1, depth: 1),
                                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

                        var nGdn = UInt32(gdnW.n), zRow0 = UInt32(geom.gdnZRow0), aRow0 = UInt32(geom.gdnARow0), bRow0 = UInt32(geom.gdnBRow0), ctrlStride: UInt32 = 96
                        enc.setComputePipelineState(psoGdn1)
                        gdnW.blocks.bind(enc, index: 0); in1.bind(enc, index: 1); gs.bind(enc, index: 2)
                        y32.bind(enc, index: 3)
                        enc.setBytes(&nGdn, length: 4, index: 4); enc.setBytes(&kHidden, length: 4, index: 5)
                        refGdn1.bind(enc, index: 8); aLog.bind(enc, index: 9); bias.bind(enc, index: 10)
                        enc.setBuffer(refCtrl.region.buffer, offset: refCtrl.region.base + refCtrl.offset + r * 96 * 4, index: 11)
                        enc.setBuffer(refCtrl.region.buffer, offset: refCtrl.region.base + refCtrl.offset + (r * 96 + 48) * 4, index: 12)
                        enc.setBytes(&zRow0, length: 4, index: 13); enc.setBytes(&aRow0, length: 4, index: 14); enc.setBytes(&bRow0, length: 4, index: 15)
                        enc.setBytes(&mOne, length: 4, index: 16); enc.setBytes(&ctrlStride, length: 4, index: 17)
                        enc.dispatchThreadgroups(MTLSize(width: gdnW.n, height: 1, depth: 1),
                                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                        enc.memoryBarrier(scope: .buffers)
                    }
                } catch {
                    enc.endEncoding()
                    throw error
                }
                enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                XCTAssertEqual(cb.status, .completed)

                XCTAssertEqual(
                    Data(bytes: outProjM.region.pointer.advanced(by: outProjM.offset), count: outProjM.bytes),
                    Data(bytes: refProjM.region.pointer.advanced(by: refProjM.offset), count: refProjM.bytes),
                    "AQ(\(bits), G\(group)) M=\(m) projection parity failed")
                XCTAssertEqual(
                    Data(bytes: outGuM.region.pointer.advanced(by: outGuM.offset), count: outGuM.bytes),
                    Data(bytes: refGuM.region.pointer.advanced(by: refGuM.offset), count: refGuM.bytes),
                    "AQ(\(bits), G\(group)) M=\(m) gateUp+fwht32 parity failed")
                XCTAssertEqual(
                    Data(bytes: outGdnM.region.pointer.advanced(by: outGdnM.offset), count: outGdnM.bytes),
                    Data(bytes: refGdnM.region.pointer.advanced(by: refGdnM.offset), count: refGdnM.bytes),
                    "AQ(\(bits), G\(group)) M=\(m) GDN inProj parity failed")
                XCTAssertEqual(
                    Data(bytes: outCtrl.region.pointer.advanced(by: outCtrl.offset), count: outCtrl.bytes),
                    Data(bytes: refCtrl.region.pointer.advanced(by: refCtrl.offset), count: refCtrl.bytes),
                    "AQ(\(bits), G\(group)) M=\(m) GDN controls parity failed")
            }
        }
    }
}
