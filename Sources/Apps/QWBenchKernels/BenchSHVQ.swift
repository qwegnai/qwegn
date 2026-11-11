// BenchSHVQ.swift — Cold-streaming SHVQ vs AQ live kernel suites.
// Races SHVQ{2,3,4} against AQ{2,3,4} on cold 256 MiB L2/SLC-flushed banks
// across dense decode GEMV, routed MoE f32_r1, MoE down+combine, and NAX M=64.
import Foundation
import Metal
import QWCore
import QWHAL
import QWQuant

enum SHVQSuites {
    struct Row {
        let suite: String
        let shape: String
        let bits: Int
        let group: Int
        let shvqKernel: String
        let aqKernel: String
        let shvqMs: Double
        let aqMs: Double
        let shvqRange: (Double, Double)
        let aqRange: (Double, Double)
        let shvqErr: Double
        let aqErr: Double
    }

    static func run(hal: QWHal, samples: Int, repeats: Int) throws -> [Row] {
        var out: [Row] = []
        for (bits, group) in [(2, 64), (3, 64), (4, 32), (4, 64)] {
            out.append(try runDecodeT4(hal: hal, bits: bits, group: group,
                                       rows: 12288, cols: 4096, samples: samples, repeats: repeats))
        }
        for (bits, group) in [(2, 64), (3, 64), (4, 64)] {
            out.append(try runMoEGateUpR1(hal: hal, label: "moe_gateup_glm", bits: bits, group: group,
                                          experts: 12, routes: 8, rowsPerExpert: 4096, cols: 2048,
                                          samples: samples, repeats: repeats))
            out.append(try runMoEGateUpR1(hal: hal, label: "moe_gateup_fn", bits: bits, group: group,
                                          experts: 12, routes: 10, rowsPerExpert: 2048, cols: 4096,
                                          samples: samples, repeats: repeats))
            out.append(try runMoEDownCombine(hal: hal, bits: bits, group: group,
                                             experts: 12, routes: 8, rowsPerRoute: 2048, cols: 2048,
                                             samples: samples, repeats: repeats))
        }
        for (bits, group) in [(2, 64), (3, 64), (4, 32), (4, 64)] {
            out.append(try runPrefillNaxF32(hal: hal, bits: bits, group: group,
                                            m: 64, rows: 4096, cols: 4096, samples: samples, repeats: repeats))
        }
        return out
    }

    private static func makeSHVQBank(bits: Int, group: Int, rows: Int, cols: Int) -> Data {
        let groups = cols / group
        let gBytes = QWSHVQBlock.bytes(bits: bits, group: group)
        var bytes = [UInt8](repeating: 0, count: rows * groups * gBytes)
        var seed: UInt64 = 0x9E37_79B9_7F4A_7C15 ^ UInt64(bits * 131 + group)
        for r in 0..<rows {
            for g in 0..<groups {
                let off = (r * groups + g) * gBytes
                let sc = Float16(LogicalWeights.scale(row: r, group: g)).bitPattern
                bytes[off] = UInt8(sc & 0xFF)
                bytes[off + 1] = UInt8(sc >> 8)
                for b in 2..<gBytes {
                    seed = seed &* 6364136223846793005 &+ 1442695040888963407
                    bytes[off + b] = UInt8((seed >> 32) & 0xFF)
                }
            }
        }
        return Data(bytes)
    }

    private static func shvqRowRef(bits: Int, group: Int, row: Int, cols: Int,
                                   bank: Data, x: [Float], roundWToF16: Bool = false) -> BenchReference {
        let groups = cols / group
        let gBytes = QWSHVQBlock.bytes(bits: bits, group: group)
        var val: Float = 0, mag: Float = 0
        var tmp = [Float]()
        bank.withUnsafeBytes { raw in
            let base = raw.baseAddress!.assumingMemoryBound(to: UInt8.self)
            for g in 0..<groups {
                let off = (row * groups + g) * gBytes
                let slice = UnsafeRawBufferPointer(start: base.advanced(by: off), count: gBytes)
                QWSHVQBlock.unpack(group: slice, bits: bits, group: group, into: &tmp)
                for j in 0..<group {
                    let w = roundWToF16 ? Float(Float16(tmp[j])) : tmp[j]
                    let p = w * x[g * group + j]
                    val += p
                    mag += abs(p)
                }
            }
        }
        return BenchReference(value: val, magnitude: mag)
    }

    private static func swigluMul(_ gate: Float, _ up: Float) -> Float {
        (gate / (1.0 + exp(-gate))) * up
    }

    private static func pairAlternating(
        samples: Int,
        shvqRun: () throws -> Double,
        aqRun: () throws -> Double
    ) throws -> (shvq: [Double], aq: [Double]) {
        var sTimes = [Double](), aTimes = [Double]()
        for i in 0..<samples {
            if i % 2 == 0 {
                sTimes.append(try shvqRun())
                aTimes.append(try aqRun())
            } else {
                aTimes.append(try aqRun())
                sTimes.append(try shvqRun())
            }
        }
        return (sTimes.sorted(), aTimes.sorted())
    }

    private static func runDecodeT4(hal: QWHal, bits: Int, group: Int,
                                    rows: Int, cols: Int, samples: Int, repeats: Int) throws -> Row {
        let device = hal.device
        let sName = "qwg_gemv_shvq\(bits)_g\(group)_dx_h16_t4"
        let aName = "qwg_gemv_aq\(bits)_g\(group)_dx_h16_t4"
        let sBank = makeSHVQBank(bits: bits, group: group, rows: rows, cols: cols)
        let aBank = AQBanks.make(bits: bits, group: group, rows: rows, cols: cols)
        let sW = try BenchRunner.buffer(device, bytes: sBank.count)
        let aW = try BenchRunner.buffer(device, bytes: aBank.count)
        sBank.withUnsafeBytes { sW.contents().copyMemory(from: $0.baseAddress!, byteCount: sBank.count) }
        aBank.withUnsafeBytes { aW.contents().copyMemory(from: $0.baseAddress!, byteCount: aBank.count) }
        let xs = BenchRunner.makeActivations(cols: cols)
        let xBuf = try BenchRunner.buffer(device, bytes: cols * 2)
        BenchRunner.fillF16(xBuf, xs)
        let yBuf = try BenchRunner.buffer(device, bytes: rows * 2)
        let sPso = try hal.pipeline(sName)
        let aPso = try hal.pipeline(aName)

        func encodeOne(_ enc: any MTLComputeCommandEncoder, pso: any MTLComputePipelineState, w: any MTLBuffer) {
            enc.setComputePipelineState(pso)
            enc.setBuffer(w, offset: 0, index: 0)
            enc.setBuffer(xBuf, offset: 0, index: 1)
            enc.setBuffer(yBuf, offset: 0, index: 3)
            var n32 = UInt32(rows) | 0x4000_0000, k32 = UInt32(cols)
            enc.setBytes(&n32, length: 4, index: 4)
            enc.setBytes(&k32, length: 4, index: 5)
            enc.setBuffer(yBuf, offset: 0, index: 8)
            enc.dispatchThreadgroups(MTLSize(width: (rows + 15) / 16, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        }

        _ = try BenchRunner.time(hal: hal, name: sName, warm: { encodeOne($0, pso: sPso, w: sW) },
                                 timed: { encodeOne($0, pso: sPso, w: sW) }, repeats: 1)
        let gotS = yBuf.contents().bindMemory(to: UInt16.self, capacity: rows)
        var sErr = 0.0
        for r in stride(from: 0, to: rows, by: 64) {
            let ref = shvqRowRef(bits: bits, group: group, row: r, cols: cols, bank: sBank, x: xs)
            sErr = max(sErr, BenchRunner.relativeError(got: Float(Float16(bitPattern: gotS[r])), reference: ref))
        }
        guard sErr <= 2e-3 else { throw BenchError.verificationFailed(sName, sErr) }

        _ = try BenchRunner.time(hal: hal, name: aName, warm: { encodeOne($0, pso: aPso, w: aW) },
                                 timed: { encodeOne($0, pso: aPso, w: aW) }, repeats: 1)
        let gotA = yBuf.contents().bindMemory(to: UInt16.self, capacity: rows)
        var aErr = 0.0
        for r in stride(from: 0, to: rows, by: 64) {
            let ref = AQBanks.referenceRow(bits: bits, group: group, row: r, cols: cols, bank: aBank, x: xs)
            aErr = max(aErr, BenchRunner.relativeError(got: Float(Float16(bitPattern: gotA[r])), reference: ref))
        }
        guard aErr <= 2e-3 else { throw BenchError.verificationFailed(aName, aErr) }

        let (sT, aT) = try pairAlternating(
            samples: samples,
            shvqRun: { try BenchRunner.time(hal: hal, name: sName, warm: { encodeOne($0, pso: sPso, w: sW) },
                                            timed: { encodeOne($0, pso: sPso, w: sW) }, repeats: repeats) },
            aqRun: { try BenchRunner.time(hal: hal, name: aName, warm: { encodeOne($0, pso: aPso, w: aW) },
                                          timed: { encodeOne($0, pso: aPso, w: aW) }, repeats: repeats) }
        )
        return Row(suite: "decode_dx_t4", shape: "\(rows)x\(cols)", bits: bits, group: group,
                   shvqKernel: sName, aqKernel: aName,
                   shvqMs: sT[sT.count / 2], aqMs: aT[aT.count / 2],
                   shvqRange: (sT.first!, sT.last!), aqRange: (aT.first!, aT.last!),
                   shvqErr: sErr, aqErr: aErr)
    }

    private static func runMoEGateUpR1(hal: QWHal, label: String, bits: Int, group: Int,
                                       experts: Int, routes: Int, rowsPerExpert: Int, cols: Int,
                                       samples: Int, repeats: Int) throws -> Row {
        let device = hal.device
        let sName = "qwg_moe_gateup_shvq\(bits)_g\(group)_f32_r1"
        let aName = "qwg_moe_gateup_aq\(bits)_g\(group)_f32_r1"
        let totalRows = experts * rowsPerExpert
        let inter = rowsPerExpert / 2
        let sBank = makeSHVQBank(bits: bits, group: group, rows: totalRows, cols: cols)
        let aBank = AQBanks.make(bits: bits, group: group, rows: totalRows, cols: cols)
        let sW = try BenchRunner.buffer(device, bytes: sBank.count)
        let aW = try BenchRunner.buffer(device, bytes: aBank.count)
        sBank.withUnsafeBytes { sW.contents().copyMemory(from: $0.baseAddress!, byteCount: sBank.count) }
        aBank.withUnsafeBytes { aW.contents().copyMemory(from: $0.baseAddress!, byteCount: aBank.count) }
        let xs = BenchRunner.makeActivations(cols: cols)
        let xBuf = try BenchRunner.buffer(device, bytes: cols * 4)
        BenchRunner.fillF32(xBuf, xs)
        let chosen = (0..<routes).map { Int32(($0 * 3 + 1) % experts) }
        let idsBuf = try BenchRunner.buffer(device, bytes: routes * 4)
        idsBuf.contents().copyMemory(from: chosen, byteCount: routes * 4)
        let outBuf = try BenchRunner.buffer(device, bytes: routes * inter * 4)
        let sPso = try hal.pipeline(sName)
        let aPso = try hal.pipeline(aName)

        func encodeOne(_ enc: any MTLComputeCommandEncoder, pso: any MTLComputePipelineState, w: any MTLBuffer) {
            enc.setComputePipelineState(pso)
            enc.setBuffer(w, offset: 0, index: 0)
            enc.setBuffer(xBuf, offset: 0, index: 1)
            enc.setBuffer(idsBuf, offset: 0, index: 2)
            enc.setBuffer(outBuf, offset: 0, index: 3)
            var n32 = UInt32(rowsPerExpert), k32 = UInt32(cols)
            var e32 = UInt32(routes) | 0x8000_0000
            enc.setBytes(&n32, length: 4, index: 4)
            enc.setBytes(&k32, length: 4, index: 5)
            enc.setBytes(&e32, length: 4, index: 6)
            enc.setBuffer(w, offset: 0, index: 7)
            enc.dispatchThreadgroups(MTLSize(width: inter / 32, height: routes, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }

        _ = try BenchRunner.time(hal: hal, name: sName, warm: { encodeOne($0, pso: sPso, w: sW) },
                                 timed: { encodeOne($0, pso: sPso, w: sW) }, repeats: 1)
        let gotS = outBuf.contents().bindMemory(to: Float.self, capacity: routes * inter)
        var exp0Block = [Float](repeating: 0, count: 32)
        var blockMag: Float = 0
        let e0 = Int(chosen[0])
        for c in 0..<32 {
            let gRef = shvqRowRef(bits: bits, group: group, row: e0 * rowsPerExpert + c, cols: cols, bank: sBank, x: xs)
            let uRef = shvqRowRef(bits: bits, group: group, row: e0 * rowsPerExpert + inter + c, cols: cols, bank: sBank, x: xs)
            exp0Block[c] = swigluMul(gRef.value, uRef.value)
            blockMag = max(blockMag, abs(exp0Block[c]))
        }
        BenchRunner.fwht32(&exp0Block)
        var sErr = 0.0
        for c in 0..<32 {
            let ref = BenchReference(value: exp0Block[c], magnitude: max(blockMag, 1e-4))
            sErr = max(sErr, BenchRunner.relativeError(got: gotS[c], reference: ref))
        }
        guard sErr <= 2e-3 else { throw BenchError.verificationFailed(sName, sErr) }

        _ = try BenchRunner.time(hal: hal, name: aName, warm: { encodeOne($0, pso: aPso, w: aW) },
                                 timed: { encodeOne($0, pso: aPso, w: aW) }, repeats: 1)
        let gotA = outBuf.contents().bindMemory(to: Float.self, capacity: routes * inter)
        var aBlock = [Float](repeating: 0, count: 32)
        var aMag: Float = 0
        for c in 0..<32 {
            let gRef = AQBanks.referenceRow(bits: bits, group: group, row: e0 * rowsPerExpert + c, cols: cols, bank: aBank, x: xs)
            let uRef = AQBanks.referenceRow(bits: bits, group: group, row: e0 * rowsPerExpert + inter + c, cols: cols, bank: aBank, x: xs)
            aBlock[c] = swigluMul(gRef.value, uRef.value)
            aMag = max(aMag, abs(aBlock[c]))
        }
        BenchRunner.fwht32(&aBlock)
        var aErr = 0.0
        for c in 0..<32 {
            let ref = BenchReference(value: aBlock[c], magnitude: max(aMag, 1e-4))
            aErr = max(aErr, BenchRunner.relativeError(got: gotA[c], reference: ref))
        }
        guard aErr <= 2e-3 else { throw BenchError.verificationFailed(aName, aErr) }

        let (sT, aT) = try pairAlternating(
            samples: samples,
            shvqRun: { try BenchRunner.time(hal: hal, name: sName, warm: { encodeOne($0, pso: sPso, w: sW) },
                                            timed: { encodeOne($0, pso: sPso, w: sW) }, repeats: repeats) },
            aqRun: { try BenchRunner.time(hal: hal, name: aName, warm: { encodeOne($0, pso: aPso, w: aW) },
                                          timed: { encodeOne($0, pso: aPso, w: aW) }, repeats: repeats) }
        )
        return Row(suite: label, shape: "k=\(routes) \(rowsPerExpert)x\(cols)", bits: bits, group: group,
                   shvqKernel: sName, aqKernel: aName,
                   shvqMs: sT[sT.count / 2], aqMs: aT[aT.count / 2],
                   shvqRange: (sT.first!, sT.last!), aqRange: (aT.first!, aT.last!),
                   shvqErr: sErr, aqErr: aErr)
    }

    private static func runMoEDownCombine(hal: QWHal, bits: Int, group: Int,
                                          experts: Int, routes: Int, rowsPerRoute: Int, cols: Int,
                                          samples: Int, repeats: Int) throws -> Row {
        let device = hal.device
        let sName = "qwg_moe_down_shared_combine_shvq\(bits)_g\(group)_f32"
        let aName = "qwg_moe_down_shared_combine_aq\(bits)_g\(group)_f32"
        let totalRows = experts * rowsPerRoute
        let sBank = makeSHVQBank(bits: bits, group: group, rows: totalRows, cols: cols)
        let aBank = AQBanks.make(bits: bits, group: group, rows: totalRows, cols: cols)
        let sW = try BenchRunner.buffer(device, bytes: sBank.count)
        let aW = try BenchRunner.buffer(device, bytes: aBank.count)
        sBank.withUnsafeBytes { sW.contents().copyMemory(from: $0.baseAddress!, byteCount: sBank.count) }
        aBank.withUnsafeBytes { aW.contents().copyMemory(from: $0.baseAddress!, byteCount: aBank.count) }
        let xs = BenchRunner.makeActivations(cols: routes * cols)
        let xBuf = try BenchRunner.buffer(device, bytes: routes * cols * 4)
        BenchRunner.fillF32(xBuf, xs)
        let rWeights = (0..<routes).map { Float(1.0 / Double(routes)) + 0.01 * Float($0) }
        let rwBuf = try BenchRunner.buffer(device, bytes: routes * 4)
        BenchRunner.fillF32(rwBuf, rWeights)
        let shared = BenchRunner.makeActivations(cols: rowsPerRoute)
        let shBuf = try BenchRunner.buffer(device, bytes: rowsPerRoute * 4)
        BenchRunner.fillF32(shBuf, shared)
        let gateVal: [Float] = [0.5]
        let gBuf = try BenchRunner.buffer(device, bytes: 4)
        BenchRunner.fillF32(gBuf, gateVal)
        let chosen = (0..<routes).map { Int32(($0 * 2 + 1) % experts) }
        let idsBuf = try BenchRunner.buffer(device, bytes: routes * 4)
        idsBuf.contents().copyMemory(from: chosen, byteCount: routes * 4)
        let outBuf = try BenchRunner.buffer(device, bytes: rowsPerRoute * 4)
        let sPso = try hal.pipeline(sName)
        let aPso = try hal.pipeline(aName)

        func encodeOne(_ enc: any MTLComputeCommandEncoder, pso: any MTLComputePipelineState, w: any MTLBuffer) {
            enc.setComputePipelineState(pso)
            enc.setBuffer(w, offset: 0, index: 0)
            enc.setBuffer(xBuf, offset: 0, index: 1)
            enc.setBuffer(rwBuf, offset: 0, index: 2)
            enc.setBuffer(shBuf, offset: 0, index: 3)
            enc.setBuffer(gBuf, offset: 0, index: 4)
            enc.setBuffer(outBuf, offset: 0, index: 5)
            var n32 = UInt32(rowsPerRoute), k32 = UInt32(cols), e32 = UInt32(routes)
            var hasOut16: UInt32 = 0
            enc.setBytes(&n32, length: 4, index: 6)
            enc.setBytes(&k32, length: 4, index: 7)
            enc.setBytes(&e32, length: 4, index: 8)
            enc.setBuffer(idsBuf, offset: 0, index: 9)
            enc.setBuffer(outBuf, offset: 0, index: 10)
            enc.setBytes(&hasOut16, length: 4, index: 11)
            enc.setBuffer(w, offset: 0, index: 12)
            enc.dispatchThreadgroups(MTLSize(width: (rowsPerRoute + 15) / 16, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        }

        _ = try BenchRunner.time(hal: hal, name: sName, warm: { encodeOne($0, pso: sPso, w: sW) },
                                 timed: { encodeOne($0, pso: sPso, w: sW) }, repeats: 1)
        let gotS = outBuf.contents().bindMemory(to: Float.self, capacity: rowsPerRoute)
        var sErr = 0.0
        for h in stride(from: 0, to: rowsPerRoute, by: 64) {
            var want = 0.5 * shared[h], mag = abs(want)
            for rt in 0..<routes {
                let xSlice = Array(xs[(rt * cols)..<((rt + 1) * cols)])
                let rRef = shvqRowRef(bits: bits, group: group, row: Int(chosen[rt]) * rowsPerRoute + h, cols: cols, bank: sBank, x: xSlice)
                want += rWeights[rt] * rRef.value
                mag += rWeights[rt] * rRef.magnitude
            }
            sErr = max(sErr, BenchRunner.relativeError(got: gotS[h], reference: BenchReference(value: want, magnitude: mag)))
        }
        guard sErr <= 2e-3 else { throw BenchError.verificationFailed(sName, sErr) }

        _ = try BenchRunner.time(hal: hal, name: aName, warm: { encodeOne($0, pso: aPso, w: aW) },
                                 timed: { encodeOne($0, pso: aPso, w: aW) }, repeats: 1)
        let gotA = outBuf.contents().bindMemory(to: Float.self, capacity: rowsPerRoute)
        var aErr = 0.0
        for h in stride(from: 0, to: rowsPerRoute, by: 64) {
            var want = 0.5 * shared[h], mag = abs(want)
            for rt in 0..<routes {
                let xSlice = Array(xs[(rt * cols)..<((rt + 1) * cols)])
                let rRef = AQBanks.referenceRow(bits: bits, group: group, row: Int(chosen[rt]) * rowsPerRoute + h, cols: cols, bank: aBank, x: xSlice)
                want += rWeights[rt] * rRef.value
                mag += rWeights[rt] * rRef.magnitude
            }
            aErr = max(aErr, BenchRunner.relativeError(got: gotA[h], reference: BenchReference(value: want, magnitude: mag)))
        }
        guard aErr <= 2e-3 else { throw BenchError.verificationFailed(aName, aErr) }

        let (sT, aT) = try pairAlternating(
            samples: samples,
            shvqRun: { try BenchRunner.time(hal: hal, name: sName, warm: { encodeOne($0, pso: sPso, w: sW) },
                                            timed: { encodeOne($0, pso: sPso, w: sW) }, repeats: repeats) },
            aqRun: { try BenchRunner.time(hal: hal, name: aName, warm: { encodeOne($0, pso: aPso, w: aW) },
                                          timed: { encodeOne($0, pso: aPso, w: aW) }, repeats: repeats) }
        )
        return Row(suite: "moe_down_combine", shape: "k=\(routes) \(rowsPerRoute)x\(cols)", bits: bits, group: group,
                   shvqKernel: sName, aqKernel: aName,
                   shvqMs: sT[sT.count / 2], aqMs: aT[aT.count / 2],
                   shvqRange: (sT.first!, sT.last!), aqRange: (aT.first!, aT.last!),
                   shvqErr: sErr, aqErr: aErr)
    }

    private static func runPrefillNaxF32(hal: QWHal, bits: Int, group: Int,
                                         m: Int, rows: Int, cols: Int, samples: Int, repeats: Int) throws -> Row {
        let device = hal.device
        let sName = "qwg_gemm_shvq\(bits)_g\(group)_nax_f32"
        let aName = "qwg_gemm_aq\(bits)_g\(group)_nax_f32"
        let sBank = makeSHVQBank(bits: bits, group: group, rows: rows, cols: cols)
        let aBank = AQBanks.make(bits: bits, group: group, rows: rows, cols: cols)
        let sW = try BenchRunner.buffer(device, bytes: sBank.count)
        let aW = try BenchRunner.buffer(device, bytes: aBank.count)
        sBank.withUnsafeBytes { sW.contents().copyMemory(from: $0.baseAddress!, byteCount: sBank.count) }
        aBank.withUnsafeBytes { aW.contents().copyMemory(from: $0.baseAddress!, byteCount: aBank.count) }
        let xs = BenchRunner.makeActivations(cols: m * cols)
        let xhBuf = try BenchRunner.buffer(device, bytes: m * cols * 2)
        BenchRunner.fillF16(xhBuf, xs)
        let y32Buf = try BenchRunner.buffer(device, bytes: m * rows * 4)
        let sPso = try hal.pipeline(sName)
        let aPso = try hal.pipeline(aName)

        func encodeOne(_ enc: any MTLComputeCommandEncoder, pso: any MTLComputePipelineState, w: any MTLBuffer) {
            enc.setComputePipelineState(pso)
            enc.setBuffer(w, offset: 0, index: 0)
            enc.setBuffer(xhBuf, offset: 0, index: 1)
            enc.setBuffer(xhBuf, offset: 0, index: 2)
            enc.setBuffer(y32Buf, offset: 0, index: 3)
            var m32 = UInt32(m), n32 = UInt32(rows), k32 = UInt32(cols)
            enc.setBytes(&m32, length: 4, index: 4)
            enc.setBytes(&n32, length: 4, index: 5)
            enc.setBytes(&k32, length: 4, index: 6)
            enc.dispatchThreadgroups(MTLSize(width: (rows + 63) / 64, height: (m + 63) / 64, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        }

        _ = try BenchRunner.time(hal: hal, name: sName, warm: { encodeOne($0, pso: sPso, w: sW) },
                                 timed: { encodeOne($0, pso: sPso, w: sW) }, repeats: 1)
        let gotS = y32Buf.contents().bindMemory(to: Float.self, capacity: m * rows)
        var sErr = 0.0
        for tok in [0, m - 1] {
            let xSlice = Array(xs[(tok * cols)..<((tok + 1) * cols)])
            for r in [0, rows / 2, rows - 1] {
                let ref = shvqRowRef(bits: bits, group: group, row: r, cols: cols, bank: sBank, x: xSlice, roundWToF16: true)
                sErr = max(sErr, BenchRunner.relativeError(got: gotS[tok * rows + r], reference: ref))
            }
        }
        guard sErr <= 2e-3 else { throw BenchError.verificationFailed(sName, sErr) }

        _ = try BenchRunner.time(hal: hal, name: aName, warm: { encodeOne($0, pso: aPso, w: aW) },
                                 timed: { encodeOne($0, pso: aPso, w: aW) }, repeats: 1)
        let gotA = y32Buf.contents().bindMemory(to: Float.self, capacity: m * rows)
        var aErr = 0.0
        for tok in [0, m - 1] {
            let xSlice = Array(xs[(tok * cols)..<((tok + 1) * cols)])
            for r in [0, rows / 2, rows - 1] {
                let ref = AQBanks.referenceRow(bits: bits, group: group, row: r, cols: cols, bank: aBank, x: xSlice)
                aErr = max(aErr, BenchRunner.relativeError(got: gotA[tok * rows + r], reference: ref))
            }
        }
        guard aErr <= 2e-3 else { throw BenchError.verificationFailed(aName, aErr) }

        let (sT, aT) = try pairAlternating(
            samples: samples,
            shvqRun: { try BenchRunner.time(hal: hal, name: sName, warm: { encodeOne($0, pso: sPso, w: sW) },
                                            timed: { encodeOne($0, pso: sPso, w: sW) }, repeats: repeats) },
            aqRun: { try BenchRunner.time(hal: hal, name: aName, warm: { encodeOne($0, pso: aPso, w: aW) },
                                          timed: { encodeOne($0, pso: aPso, w: aW) }, repeats: repeats) }
        )
        return Row(suite: "prefill_nax_f32", shape: "m=\(m) \(rows)x\(cols)", bits: bits, group: group,
                   shvqKernel: sName, aqKernel: aName,
                   shvqMs: sT[sT.count / 2], aqMs: aT[aT.count / 2],
                   shvqRange: (sT.first!, sT.last!), aqRange: (aT.first!, aT.last!),
                   shvqErr: sErr, aqErr: aErr)
    }
}
