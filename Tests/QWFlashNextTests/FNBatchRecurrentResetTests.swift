// FNBatchRecurrentResetTests.swift — Tests unconditional GDN/conv/PLE buffer reset at highWaterPos == 0.
import Foundation
import Metal
import XCTest
import QWAlloc
import QWCore
@testable import QWEngine
import QWGEMM
import QWHAL
import QWHyperPLE

final class FNBatchRecurrentResetTests: XCTestCase {
    private static func fillBytes(_ ptr: UnsafeMutableRawPointer, count: Int, byte: UInt8) {
        memset(ptr, Int32(byte), count)
    }

    private static func allZero(_ ptr: UnsafeMutableRawPointer, count: Int) -> Bool {
        let buf = UnsafeRawBufferPointer(start: ptr, count: count)
        return buf.allSatisfy { $0 == 0 }
    }

    private static func readSource(_ relativePath: String) throws -> String {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        return try String(contentsOf: root.appendingPathComponent(relativePath), encoding: .utf8)
    }

    func testWipeRecurrentLayerAndExtraSlotResetAtZeroHighWaterPos() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.supportsSimdgroupMMA == true,
            "Metal buffer zeroing check requires bare-metal Apple Silicon GPU"
        )
        let hal = try QWHal(geometry: .qwen38_27b)
        let arena = try QWArena(hal: hal, weights: 8 << 10, activations: 64 << 10, kv: 0, gdnStates: 0)
        let act = arena.activationsRegion()
        let wBuf = try arena.weightsRegion().allocate(QWTensorShape([32, 32], .f16))
        let weightView = try XCTUnwrap(QWWeightView(blocks: wBuf, n: 32, k: 32, bits: 4, groupSize: 32, isRotated: true))
        let vec = try act.allocate(QWTensorShape([8], .f32))

        let gState = try act.allocate(QWTensorShape([16], .f32))
        let gConv = try act.allocate(QWTensorShape([16], .f32))
        let pHist = try act.allocate(QWTensorShape([16], .f32))
        Self.fillBytes(arena.cpuPointer(of: gState), count: gState.bytes, byte: 0x5A)
        Self.fillBytes(arena.cpuPointer(of: gConv), count: gConv.bytes, byte: 0x5A)
        Self.fillBytes(arena.cpuPointer(of: pHist), count: pHist.bytes, byte: 0x5A)

        let gdn = FNGPUGDNMixer(
            qkv: weightView, z: weightView, qkvZ: nil, out: weightView, inProjBA: weightView,
            conv: vec, normWeight: vec, aLog: vec, dtBias: vec, inProjB: vec, inProjA: vec,
            states: [gState], convState: [gConv], vHeads: 1, kHeads: 1, kDim: 4, vDim: 4, convDim: 4
        )
        gdn.stateParity = 3
        gdn.convParity = 5
        let ple = FNGPUPLE(
            keyValueBank: weightView, keyBank: weightView, valueBank: weightView,
            normKey: vec, normQuery: vec, normConv: vec, convW: vec,
            embedDim: 4, kernel: 4, dilation: 1, eps: 1e-6, history: [pHist]
        )
        ple.parity = 1

        QWHyperStreamRuntime.wipeRecurrentLayer(gdn: gdn, ple: ple, arena: arena)
        XCTAssertTrue(Self.allZero(arena.cpuPointer(of: gState), count: gState.bytes))
        XCTAssertTrue(Self.allZero(arena.cpuPointer(of: gConv), count: gConv.bytes))
        XCTAssertTrue(Self.allZero(arena.cpuPointer(of: pHist), count: pHist.bytes))
        XCTAssertEqual(gdn.stateParity, 0)
        XCTAssertEqual(gdn.convParity, 0)
        XCTAssertEqual(ple.parity, 0)

        var slotStates: [FNBatchSlotState] = []
        for s in 0..<4 {
            let sBuf = try act.allocate(QWTensorShape([16], .f32))
            let cBuf = try act.allocate(QWTensorShape([16], .f32))
            let hBuf = try act.allocate(QWTensorShape([16], .f32))
            Self.fillBytes(arena.cpuPointer(of: sBuf), count: sBuf.bytes, byte: 0x7C)
            Self.fillBytes(arena.cpuPointer(of: cBuf), count: cBuf.bytes, byte: 0x7C)
            Self.fillBytes(arena.cpuPointer(of: hBuf), count: hBuf.bytes, byte: 0x7C)
            slotStates.append(FNBatchSlotState(
                gdnStates: [[sBuf]], gdnStateParity: [s + 1],
                gdnConvState: [[cBuf]], gdnConvParity: [s + 2],
                pleHistory: [[hBuf]], pleParity: [1],
                highWaterPos: 0
            ))
        }

        FNGPUBatchScheduler.resetExtraSlotRecurrentStates(
            slotStates: &slotStates, gdnFlags: [true], pleFlags: [true], arena: arena
        )
        XCTAssertEqual(slotStates[0].gdnStateParity[0], 1)
        XCTAssertEqual(slotStates[0].gdnConvParity[0], 2)
        XCTAssertEqual(slotStates[0].pleParity[0], 1)
        XCTAssertFalse(Self.allZero(arena.cpuPointer(of: slotStates[0].gdnStates[0][0]), count: 64))

        for s in 1..<4 {
            XCTAssertEqual(slotStates[s].gdnStateParity[0], 0)
            XCTAssertEqual(slotStates[s].gdnConvParity[0], 0)
            XCTAssertEqual(slotStates[s].pleParity[0], 0)
            XCTAssertTrue(Self.allZero(arena.cpuPointer(of: slotStates[s].gdnStates[0][0]), count: 64))
            XCTAssertTrue(Self.allZero(arena.cpuPointer(of: slotStates[s].gdnConvState[0][0]), count: 64))
            XCTAssertTrue(Self.allZero(arena.cpuPointer(of: slotStates[s].pleHistory[0][0]), count: 64))
        }
    }

    func testSessionAndSchedulerWiringDoNotGateRecurrentResetOnHighWaterPos() throws {
        let sessionSrc = try Self.readSource("Sources/Runtime/QWEngine/Driver/HyperStream/FNGPUSession.swift")
        XCTAssertFalse(sessionSrc.contains("guard used > 0 else"))
        XCTAssertTrue(sessionSrc.contains("Self.wipeRecurrentLayer(gdn: rec.gdn, ple: rec.weights.ple, arena: arena)"))

        let schedSrc = try Self.readSource("Sources/Runtime/QWEngine/Driver/HyperStream/FNGPUBatchScheduler.swift")
        XCTAssertFalse(schedSrc.contains("for s in 1..<slotStates.count where slotStates[s].highWaterPos > 0"))
        XCTAssertTrue(schedSrc.contains("Self.resetExtraSlotRecurrentStates("))

        let driveSrc = try Self.readSource("Sources/Runtime/QWEngine/Driver/HyperStream/FNGPUBatchScheduler+Drive.swift")
        XCTAssertTrue(driveSrc.contains("slotStates[syncSlot].highWaterPos = engine.highWaterPos"))
    }
}
