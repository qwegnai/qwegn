// QWAttnDecodeGeneralQuantKVTests.swift — Numerical parity tests for general Q8/Q4 decode and KV writers.

import XCTest
import Metal
import Foundation
@testable import QWAttn
import QWCore
import QWHAL
import QWAlloc
import QWKV
import QWQuant

final class QWAttnDecodeGeneralQuantKVTests: XCTestCase {
    static func makeGeometry(qHeads: Int = 16, kvHeads: Int = 2, headDim: Int = 256) throws -> QWModelGeometry {
        let config: [String: Any] = [
            "model_type": "qwen3",
            "hidden_size": qHeads * headDim,
            "num_hidden_layers": 1,
            "num_attention_heads": qHeads,
            "num_key_value_heads": kvHeads,
            "intermediate_size": 512,
            "max_position_embeddings": 256,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "eos_token_id": 2,
            "vocab_size": 128,
        ]
        let data = try JSONSerialization.data(withJSONObject: config)
        return QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: data))
    }

    func testQ8AndQ4GeneralWriterAndDecodeParity() throws {
        for (qHeads, kvHeads, headDim) in [(8, 2, 128), (16, 2, 256)] {
            for count in [32, 260] {
                try checkParity(geometry: Self.makeGeometry(qHeads: qHeads, kvHeads: kvHeads, headDim: headDim),
                                count: count)
            }
        }
    }

    private func checkParity(geometry: QWModelGeometry, count: Int) throws {
        let hal = try QWHal(geometry: geometry)
        let C = count
        let rowElems = geometry.kvHeads * geometry.headDim
        let pages = (C + 255) / 256

        for mode in [QWKVMode.q8, QWKVMode.q4] {
            let layout = QWKVCompressedLayout(mode: mode, geometry: geometry)
            let pageBytes = layout.totalBytesPerPage
            let kpool = hal.device.makeBuffer(length: pageBytes * pages, options: .storageModeShared)!
            let vpool = hal.device.makeBuffer(length: pageBytes * pages, options: .storageModeShared)!
            let table = hal.device.makeBuffer(length: 16, options: .storageModeShared)!
            let pageTable = table.contents().bindMemory(to: Int32.self, capacity: 4)
            for page in 0..<pages { pageTable[page] = Int32(pages - page - 1) }

            // Generate deterministic input K/V rows
            var kData = [UInt16](repeating: 0, count: C * rowElems)
            var vData = [UInt16](repeating: 0, count: C * rowElems)
            for t in 0..<C {
                for e in 0..<rowElems {
                    let kf = sin(Float(t * 31 + e)) * 0.5
                    let vf = cos(Float(t * 17 + e)) * 0.5
                    kData[t * rowElems + e] = Float16(kf).bitPattern
                    vData[t * rowElems + e] = Float16(vf).bitPattern
                }
            }
            let kIn = hal.device.makeBuffer(length: kData.count * 2, options: .storageModeShared)!
            let vIn = hal.device.makeBuffer(length: vData.count * 2, options: .storageModeShared)!
            memcpy(kIn.contents(), kData, kData.count * 2)
            memcpy(vIn.contents(), vData, vData.count * 2)

            // 1. Run qwg_kv_paged_write_general
            let psoWriter = try hal.pipeline("qwg_kv_paged_write_general")
            guard let (cbWrite, encWrite) = hal.begin() else { throw NSError(domain: "hal", code: 1) }
            encWrite.setComputePipelineState(psoWriter)
            encWrite.setBuffer(kpool, offset: 0, index: 0)
            encWrite.setBuffer(vpool, offset: 0, index: 1)
            encWrite.setBuffer(table, offset: 0, index: 2)
            encWrite.setBuffer(table, offset: 0, index: 3)
            encWrite.setBuffer(kIn, offset: 0, index: 4)
            encWrite.setBuffer(vIn, offset: 0, index: 5)
            encWrite.setUInt32s(0, C, startIndex: 6)
            var q8ModeVal: UInt32 = (mode == .q8 ? 1 : 2)
            encWrite.setBytes(&q8ModeVal, length: 4, index: 8)
            encWrite.dispatchThreadgroups(MTLSize(width: geometry.kvHeads, height: C, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            encWrite.endEncoding()
            cbWrite.commit()
            cbWrite.waitUntilCompleted()
            XCTAssertEqual(cbWrite.status, .completed)

            // 2. Decode pipeline for general Q8/Q4
            let dec = try QWAttnDecode(hal: hal, kvMode: mode, geometry: geometry)
            let outHalves = geometry.qHeads * geometry.headDim
            let out = hal.device.makeBuffer(length: outHalves * 2, options: .storageModeShared)!
            let pBytes = QWAttnDecode.decodePartialsBytes(geometry: geometry)
            let partBuf = hal.device.makeBuffer(length: pBytes, options: .storageModeShared)!
            let partialsRegion = QWMRegion(buffer: partBuf, base: 0, length: pBytes)

            var qData = [UInt16](repeating: 0, count: geometry.qProjOut)
            for i in 0..<geometry.qProjOut {
                qData[i] = Float16(cos(Float(i * 13)) * 0.4).bitPattern
            }
            let qBuf = hal.device.makeBuffer(length: qData.count * 2, options: .storageModeShared)!
            memcpy(qBuf.contents(), qData, qData.count * 2)

            let kNewBuf = hal.device.makeBuffer(length: rowElems * 2, options: .storageModeShared)!
            let vNewBuf = hal.device.makeBuffer(length: rowElems * 2, options: .storageModeShared)!
            memcpy(kNewBuf.contents(), kData, rowElems * 2)
            memcpy(vNewBuf.contents(), vData, rowElems * 2)

            let tableTensor = QWTensorValue(region: QWMRegion(buffer: table, base: 0, length: 16),
                                            offset: 0, shape: QWTensorShape([4], .i32))
            let kView = QWKVLayerView(pool: QWMRegion(buffer: kpool, base: 0, length: pageBytes * pages),
                                      table: tableTensor, pageBytes: pageBytes, mode: mode)
            let vView = QWKVLayerView(pool: QWMRegion(buffer: vpool, base: 0, length: pageBytes * pages),
                                      table: tableTensor, pageBytes: pageBytes, mode: mode)
            let kvViews = (k: kView, v: vView)

            guard let (cbDec, encDec) = hal.begin() else { throw NSError(domain: "hal", code: 2) }
            try dec.encode(encDec,
                           qgate: QWTensorValue(region: QWMRegion(buffer: qBuf, base: 0, length: qBuf.length),
                                                offset: 0, shape: QWTensorShape([geometry.qProjOut], .f16)),
                           kv: kvViews,
                           knew: QWTensorValue(region: QWMRegion(buffer: kNewBuf, base: 0, length: kNewBuf.length),
                                                offset: 0, shape: QWTensorShape([rowElems], .f16)),
                           vnew: QWTensorValue(region: QWMRegion(buffer: vNewBuf, base: 0, length: vNewBuf.length),
                                                offset: 0, shape: QWTensorShape([rowElems], .f16)),
                           partials: partialsRegion,
                           out: QWTensorValue(region: QWMRegion(buffer: out, base: 0, length: out.length),
                                              offset: 0, shape: QWTensorShape([outHalves], .f16)),
                           C: C - 1, splits: 1)
            encDec.endEncoding()
            cbDec.commit()
            cbDec.waitUntilCompleted()
            XCTAssertEqual(cbDec.status, .completed)

            let kRows = (0..<C).map { row in
                row == C - 1 ? Self.floatRow(kData, row: 0, width: rowElems)
                    : Self.quantizedRow(kData, row: row, geometry: geometry, mode: mode)
            }
            let vRows = (0..<C).map { row in
                row == C - 1 ? Self.floatRow(vData, row: 0, width: rowElems)
                    : Self.quantizedRow(vData, row: row, geometry: geometry, mode: mode)
            }
            let outPtr = out.contents().assumingMemoryBound(to: UInt16.self)
            let finiteCount = (0..<outHalves).filter {
                let value = Float(Float16(bitPattern: outPtr[$0]))
                return value.isFinite && !value.isZero
            }.count
            XCTAssertGreaterThan(finiteCount, outHalves / 2, "non-trivial output for mode \(mode)")
            for head in 0..<geometry.qHeads {
                let q = (0..<geometry.headDim).map {
                    Float(Float16(bitPattern: qData[head * geometry.headDim + $0]))
                }
                let ref = QWAttnDecodeGeneralTests.reference(
                    q: q, k: kRows, v: vRows, kvh: head / geometry.gqaRatio,
                    headDim: geometry.headDim)
                let expected = QWAttnDecodeGeneralTests.combined(ref.o, l: ref.l, headDim: geometry.headDim)
                for dim in 0..<geometry.headDim {
                    let actual = Float(Float16(bitPattern: outPtr[head * geometry.headDim + dim]))
                    XCTAssertEqual(actual, expected[dim], accuracy: 0.003,
                                   "mode=\(mode) C=\(C) head=\(head) dim=\(dim)")
                }
            }
        }
    }

    private static func floatRow(_ data: [UInt16], row: Int, width: Int) -> [Float] {
        (0..<width).map { Float(Float16(bitPattern: data[row * width + $0])) }
    }

    private static func quantizedRow(_ data: [UInt16], row: Int, geometry: QWModelGeometry,
                                     mode: QWKVMode) -> [Float] {
        let width = geometry.kvHeads * geometry.headDim
        let input = floatRow(data, row: row, width: width)
        var result = [Float](repeating: 0, count: width)
        let groupSize = mode == .q8 ? geometry.headDim : 128
        for start in stride(from: 0, to: width, by: groupSize) {
            let end = min(start + groupSize, width)
            let values = input[start..<end]
            let minimum = mode == .q8 ? Float(0) : Float(Float16(values.min()!))
            let maximum = mode == .q8 ? values.map(abs).max()! : values.max()!
            let rawScale = mode == .q8 ? maximum / 127 : (maximum - values.min()!) / 15
            let scale = Float(Float16(max(rawScale, 1e-8)))
            for index in start..<end {
                let rounded = ((input[index] - minimum) / scale).rounded()
                let code = mode == .q8 ? max(-127, min(127, rounded)) : max(0, min(15, rounded))
                result[index] = Float(Float16(code * scale + minimum))
            }
        }
        return result
    }
}
