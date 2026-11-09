import XCTest
import Foundation
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWQuant

final class QWMoEReuseTests: XCTestCase {
    private enum Format: CaseIterable {
        case incumbent
        case affine

        var group: Int { self == .incumbent ? 32 : 64 }
        var basis: QWWeightBasis { self == .incumbent ? .rotated : .standard }
    }

    private func code(row: Int, column: Int) -> UInt32 {
        UInt32((row * 7 + column * 3 + 5) % 16)
    }

    private func weight(row: Int, column: Int) -> Double {
        (Double(code(row: row, column: column)) - 8) / 16
    }

    private func bank(rows: Int, columns: Int, format: Format) throws -> Data {
        var data = Data()
        for row in 0..<rows {
            for start in stride(from: 0, to: columns, by: format.group) {
                let values = (0..<format.group).map { code(row: row, column: start + $0) }
                if format == .incumbent {
                    data.append(QWAQProducer.hq4G32Block(codes: values, scale: 1.0 / 16))
                } else {
                    data.append(try QWAQBlock.pack(bits: 4, group: format.group,
                                                   values: values, scale: 1.0 / 16, bias: -0.5))
                }
            }
        }
        return data
    }

    func testExpertReuseAndIndependentRowsMatchCPU() throws {
        let experts = 15, routes = 3, hidden = 64, intermediate = 64
        let patterns: [[Int32]] = [
            [0, 1, 2, 1, 3, 2, 0, 3, -1, 3, 1, 0, 4, 5, 6],
            Array(0..<15).map(Int32.init),
        ]
        for format in Format.allCases {
            let hal = try QWHal(geometry: .qwen38_27b, weightBasis: format.basis)
            let gateBank = try bank(rows: experts * 2 * intermediate, columns: hidden, format: format)
            let downBank = try bank(rows: experts * hidden, columns: intermediate, format: format)
            for tokens in 1...5 {
                for pattern in patterns {
                    let ids = Array(pattern.prefix(tokens * routes))
                    let arena = try QWArena(hal: hal, weights: gateBank.count + downBank.count + (1 << 20),
                                            activations: 4 << 20, kv: 0, gdnStates: 0)
                    let weightsRegion = arena.weightsRegion()
                    let gateBlocks = try weightsRegion.allocate(QWTensorShape([gateBank.count / 2], .f16))
                    let downBlocks = try weightsRegion.allocate(QWTensorShape([downBank.count / 2], .f16))
                    arena.copyIn(gateBlocks, from: gateBank)
                    arena.copyIn(downBlocks, from: downBank)
                    let gateView = try XCTUnwrap(QWWeightView(blocks: gateBlocks, n: experts * 2 * intermediate,
                                                              k: hidden, bits: 4, groupSize: format.group,
                                                              isRotated: format == .incumbent))
                    let downView = try XCTUnwrap(QWWeightView(blocks: downBlocks, n: experts * hidden,
                                                              k: intermediate, bits: 4, groupSize: format.group,
                                                              isRotated: format == .incumbent))
                    let act = arena.activationsRegion()
                    let x = try act.allocate(QWTensorShape([tokens * hidden], .f32))
                    let idTensor = try act.allocate(QWTensorShape([ids.count], .i32))
                    let routeWeights = try act.allocate(QWTensorShape([ids.count], .f32))
                    let shared = try act.allocate(QWTensorShape([tokens * hidden], .f32))
                    let sharedGate = try act.allocate(QWTensorShape([tokens], .f32))
                    let xs = (0..<(tokens * hidden)).map { Float(($0 * 3 % 17) - 8) / 64 }
                    let ws = (0..<ids.count).map { Float(($0 % routes) + 1) / 8 }
                    let sharedValues = (0..<(tokens * hidden)).map { Float(($0 % 11) - 5) / 32 }
                    let gates = (0..<tokens).map { Float($0 + 1) / 8 }
                    arena.write(x, from: xs)
                    arena.write(idTensor, from: ids)
                    arena.write(routeWeights, from: ws)
                    arena.write(shared, from: sharedValues)
                    arena.write(sharedGate, from: gates)

                    var expectedAct = [Float](repeating: 0, count: ids.count * intermediate)
                    for slot in ids.indices where ids[slot] >= 0 {
                        let expert = Int(ids[slot]), token = slot / routes
                        for row in 0..<intermediate {
                            var gate = 0.0, up = 0.0
                            for column in 0..<hidden {
                                let input = Double(xs[token * hidden + column])
                                gate += weight(row: expert * 2 * intermediate + row, column: column) * input
                                up += weight(row: expert * 2 * intermediate + intermediate + row, column: column) * input
                            }
                            expectedAct[slot * intermediate + row] = Float(gate / (1 + exp(-gate)) * up)
                        }
                    }
                    if format == .incumbent {
                        expectedAct.withUnsafeMutableBufferPointer { ptr in
                            for slot in ids.indices where ids[slot] >= 0 {
                                for offset in stride(from: 0, to: intermediate, by: 32) {
                                    QWQuant.fwht32OrthoBlock(ptr.baseAddress! + slot * intermediate + offset)
                                }
                            }
                        }
                    }
                    var expectedRoutes = [Double](repeating: 0, count: ids.count * hidden)
                    var expectedOut = [Double](repeating: 0, count: tokens * hidden)
                    for token in 0..<tokens {
                        for row in 0..<hidden {
                            var sum = Double(gates[token] * sharedValues[token * hidden + row])
                            for route in 0..<routes {
                                let slot = token * routes + route
                                guard ids[slot] >= 0 else { continue }
                                var dot = 0.0
                                for column in 0..<intermediate {
                                    dot += weight(row: Int(ids[slot]) * hidden + row, column: column)
                                        * Double(expectedAct[slot * intermediate + column])
                                }
                                expectedRoutes[slot * hidden + row] = dot
                                sum += Double(ws[slot]) * dot
                            }
                            expectedOut[token * hidden + row] = sum
                        }
                    }
                    let modes: [(Bool?, Bool?)] = [
                        (false, false), (false, true), (true, false), (true, true), (nil, nil),
                    ]
                    for (reuseGate, reuseDown) in modes {
                        let outputAct = try act.allocate(QWTensorShape([ids.count * intermediate], .f32))
                        let output = try act.allocate(QWTensorShape([tokens * hidden], .f32))
                        let routeOutput = try act.allocate(QWTensorShape([ids.count * hidden], .f32))
                        let (cb, enc) = try XCTUnwrap(hal.begin())
                        if let reuseGate {
                            try QWGemm.encodeMoEGateUpF32R1(enc, hal: hal, x: x, bank: gateView, ids: idTensor,
                                                            out: outputAct, rowsPerExpert: 2 * intermediate,
                                                            routes: routes, tokens: tokens, fuseSiluFwht: true,
                                                            reuseExpertsAcrossTokens: reuseGate)
                        } else {
                            try QWGemm.encodeMoEGateUpF32R1(enc, hal: hal, x: x, bank: gateView, ids: idTensor,
                                                            out: outputAct, rowsPerExpert: 2 * intermediate,
                                                            routes: routes, tokens: tokens, fuseSiluFwht: true)
                        }
                        enc.memoryBarrier(scope: .buffers)
                        for (result, isRouteOutput) in [(output, false), (routeOutput, true)] {
                            if let reuseDown {
                                try QWGemm.encodeMoEDownSharedCombineF32(enc, hal: hal, x: outputAct, bank: downView,
                                                                         rowsPerRoute: hidden, routes: routes,
                                                                         weights: routeWeights, shared: shared,
                                                                         gate: sharedGate, out: result, ids: idTensor,
                                                                         tokens: tokens, routeOutput: isRouteOutput,
                                                                         reuseExpertsAcrossTokens: reuseDown)
                            } else {
                                try QWGemm.encodeMoEDownSharedCombineF32(enc, hal: hal, x: outputAct, bank: downView,
                                                                         rowsPerRoute: hidden, routes: routes,
                                                                         weights: routeWeights, shared: shared,
                                                                         gate: sharedGate, out: result, ids: idTensor,
                                                                         tokens: tokens, routeOutput: isRouteOutput)
                            }
                        }
                        hal.end(cb, enc)
                        cb.waitUntilCompleted()
                        XCTAssertNil(cb.error)
                        let actualAct = arena.read(outputAct, as: Float.self, count: expectedAct.count)
                        for slot in ids.indices where ids[slot] >= 0 {
                            for column in 0..<intermediate {
                                let i = slot * intermediate + column
                                XCTAssertEqual(actualAct[i], expectedAct[i], accuracy: 1e-4,
                                               "\(format), \(tokens) rows, gate reuse \(String(describing: reuseGate)), slot \(slot)")
                            }
                        }
                        let actualOut = arena.read(output, as: Float.self, count: expectedOut.count)
                        for i in actualOut.indices {
                            XCTAssertEqual(Double(actualOut[i]), expectedOut[i], accuracy: 1e-4,
                                           "\(format), \(tokens) rows, down reuse \(String(describing: reuseDown)), output \(i)")
                        }
                        let actualRoutes = arena.read(routeOutput, as: Float.self, count: expectedRoutes.count)
                        for slot in ids.indices where ids[slot] >= 0 {
                            for row in 0..<hidden {
                                let i = slot * hidden + row
                                XCTAssertEqual(Double(actualRoutes[i]), expectedRoutes[i], accuracy: 1e-4,
                                               "\(format), \(tokens) rows, down reuse \(String(describing: reuseDown)), route \(slot)")
                            }
                        }
                    }
                }
            }
        }
    }
}
