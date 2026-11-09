import XCTest
import Metal
import QWAlloc
import QWCore
import QWHAL
import QWHead
import QWState
import QWMTP
@testable import QWEngine

final class QWVerifyWindowTests: XCTestCase {
    func testSharedGreedyTokenValidation() throws {
        let cases: [([Float], Int)] = [([-3, 2, 2], 1), ([-5, -4, -6], 1), ([0, -0, 0], 0)]
        for (scores, expected) in cases {
            try scores.withUnsafeBufferPointer {
                XCTAssertEqual(try QWVerifyWindow.greedyToken($0), expected)
            }
        }
        let invalid: [[Float]] = [[], [.nan, 1], [1, .infinity], [-.infinity, 1]]
        for scores in invalid {
            try scores.withUnsafeBufferPointer {
                XCTAssertThrowsError(try QWVerifyWindow.greedyToken($0))
            }
        }
    }

    func testLayerMajorScratchIsConditionalAndExactlyBudgeted() throws {
        let hal = try QWHal(geometry: .qwen38_27b)
        for rows in [2, 3] {
            let expected = QWVerifyWindow.allocationBytes(geometry: QWTestGeometry.g27, rows: rows) +
                rows * (QWModelGeometry.qwen38_27b.hidden + QWModelGeometry.qwen38_27b.intermediate) * 2
            XCTAssertEqual(QWVerifyWindow.allocationBytes(geometry: QWTestGeometry.g27, rows: rows, layerMajor: true), expected)
            let arena = try QWArena(hal: hal, weights: 0, activations: 4 << 20, kv: 0, gdnStates: 0)
            let allocator = arena.activationsRegion()
            let window = try QWVerifyWindow(geometry: QWTestGeometry.g27, allocator: allocator, rows: rows, layerMajor: true)
            XCTAssertEqual(allocator.usedBytes, expected)
            for row in 0..<rows {
                XCTAssertEqual(try window.normalizedRow(row).bytes, QWModelGeometry.qwen38_27b.hidden * 2)
                XCTAssertEqual(try window.mlpRow(row).bytes, QWModelGeometry.qwen38_27b.intermediate * 2)
            }
            XCTAssertThrowsError(try window.normalizedRow(-1))
            XCTAssertThrowsError(try window.mlpRow(rows))
        }
        let arena = try QWArena(hal: hal, weights: 0, activations: 4 << 20, kv: 0, gdnStates: 0)
        let disabled = try QWVerifyWindow(geometry: QWTestGeometry.g27, allocator: arena.activationsRegion(), rows: 2)
        XCTAssertNil(disabled.normalized)
        XCTAssertNil(disabled.mlpRows)
        XCTAssertNil(disabled.targetSelector)
        XCTAssertNil(disabled.targetTokens)
        XCTAssertThrowsError(try disabled.normalizedRow(0))
        XCTAssertThrowsError(try disabled.mlpRow(0))
    }

    func testGumbelComparisonsEncodeSharedPhiloxWinners() throws {
        let hal = try QWHal(geometry: .qwen38_27b)
        let arena = try QWArena(hal: hal, weights: 0, activations: 128 << 10, kv: 0, gdnStates: 0)
        let allocator = arena.activationsRegion()
        let window = try QWVerifyWindow(geometry: QWTestGeometry.g27, allocator: allocator, rows: 2, vocabulary: 4)
        let draft = try allocator.allocate(QWTensorShape([4], .f32))
        let winners = try allocator.allocate(QWTensorShape([4], .i32))
        let draftValues: [Float] = [0.25, 1.5, -0.5, -2]
        let targetValues: [Float] = [1.25, 0.75, -0.5, -2]
        draftValues.withUnsafeBufferPointer {
            draft.region.pointer.advanced(by: draft.offset).copyMemory(
                from: $0.baseAddress!, byteCount: draft.bytes)
        }
        let target = window.logits.region.pointer.advanced(by: window.logits.offset)
            .assumingMemoryBound(to: Float.self)
        let targetRows = [targetValues, targetValues].flatMap { $0 }
        targetRows.withUnsafeBufferPointer { target.update(from: $0.baseAddress!, count: $0.count) }
        let seed: UInt64 = 137, position: UInt64 = 3
        let temperature: Float = 0.75
        let aliasedWinners = QWTensorValue(region: window.logits.region,
            offset: window.logits.offset + 4 * MemoryLayout<Float>.size,
            shape: QWTensorShape([4], .i32))
        let invalidCB = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let invalidEnc = try XCTUnwrap(invalidCB.makeComputeCommandEncoder())
        XCTAssertThrowsError(try window.encodeGumbelComparisons(invalidEnc, hal: hal,
            draftLogits: [draft], seed: seed, startingPosition: position,
            temperature: temperature, winners: aliasedWinners))
        invalidEnc.endEncoding()

        let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
        try window.encodeGumbelComparisons(enc, hal: hal, draftLogits: [draft], seed: seed,
                                           startingPosition: position, temperature: temperature,
                                           winners: winners)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        XCTAssertEqual(cb.status, .completed, "\(String(describing: cb.error))")

        func winner(_ logits: [Float], row: Int) -> Int32 {
            let base = position * 4 + UInt64(row * 4)
            return Int32(logits.indices.max { lhs, rhs in
                let lu = max(Double(QWPhilox.u(base + UInt64(lhs), seed: seed, version: .upper24)),
                             1.0 / 16_777_216.0)
                let ru = max(Double(QWPhilox.u(base + UInt64(rhs), seed: seed, version: .upper24)),
                             1.0 / 16_777_216.0)
                let ls = Double(logits[lhs]) / Double(temperature) - log(-log(lu))
                let rs = Double(logits[rhs]) / Double(temperature) - log(-log(ru))
                return ls == rs ? lhs > rhs : ls < rs
            }!)
        }
        let actual = winners.region.pointer.advanced(by: winners.offset).assumingMemoryBound(to: Int32.self)
        XCTAssertEqual(Array(UnsafeBufferPointer(start: actual, count: 4)),
                       [winner(draftValues, row: 0), winner(targetValues, row: 0),
                        winner(targetValues, row: 1), winner(targetValues, row: 1)])
        let expectedDraft = Int(winner(draftValues, row: 0))
        let targetAtDraft = Int(winner(targetValues, row: 0))
        let decision = try window.completedGumbelDecision(winners: winners)
        XCTAssertEqual(decision.accepted, expectedDraft == targetAtDraft ? 1 : 0)
        XCTAssertEqual(decision.finalToken, expectedDraft == targetAtDraft
            ? Int(winner(targetValues, row: 1)) : targetAtDraft)
        XCTAssertEqual(decision.draftTokens, [expectedDraft])
    }

    func testGPUTargetJournalRejectsAnyBadRowAndResetsAfterCompletion() throws {
        let hal = try QWHal(geometry: .qwen38_27b), h = QWModelGeometry.qwen38_27b.hidden, vocabulary = 33
        let arena = try QWArena(hal: hal, weights: h * vocabulary * 2 + 256,
                                activations: 256 << 10, kv: 0, gdnStates: 0)
        let weight = try arena.weightsRegion().allocate(QWTensorShape([h * vocabulary], .f16))
        let head = try QWHead(hal: hal, geometry: .qwen38_27b)
        memset(weight.region.pointer.advanced(by: weight.offset), 0, weight.bytes)
        XCTAssertEqual(QWVerifyWindow.allocationBytes(geometry: QWTestGeometry.g27, rows: 3, gpuTargets: true) -
            QWVerifyWindow.allocationBytes(geometry: QWTestGeometry.g27, rows: 3), 8704)
        for rows in [2, 3] {
            let before = arena.activationsRegion().usedBytes
            let window = try QWVerifyWindow(geometry: QWTestGeometry.g27, allocator: arena.activationsRegion(), rows: rows,
                                            vocabulary: vocabulary, gpuTargets: true)
            XCTAssertEqual(arena.activationsRegion().usedBytes - before,
                QWVerifyWindow.allocationBytes(geometry: QWTestGeometry.g27, rows: rows, vocabulary: vocabulary, gpuTargets: true))
            for badRow in 0..<rows {
                memset(window.hidden.region.pointer.advanced(by: window.hidden.offset), 0, window.hidden.bytes)
                let bad = try window.hiddenRow(badRow)
                bad.region.pointer.advanced(by: bad.offset).assumingMemoryBound(to: UInt16.self).pointee = 0x7e00
                func submit() throws {
                    let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
                    let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
                    do { try window.encodeHead(enc, hal: hal, head: head, storage: .f16(weight)) }
                    catch { enc.endEncoding(); throw error }
                    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                    XCTAssertEqual(cb.status, .completed)
                }
                try submit()
                XCTAssertThrowsError(try window.completedTargets()) { error in
                    XCTAssertTrue(String(describing: error).contains("GPU target row \(badRow)"))
                }
                let journal = try XCTUnwrap(window.targetTokens)
                let ids = journal.region.pointer.advanced(by: journal.offset).assumingMemoryBound(to: Int32.self)
                for row in 0..<rows { XCTAssertEqual(ids[row], row == badRow ? -1 : 0) }
                bad.region.pointer.advanced(by: bad.offset).assumingMemoryBound(to: UInt16.self).pointee = 0
                try submit()
                XCTAssertEqual(try window.completedTargets(), [Int](repeating: 0, count: rows))
            }
        }
    }

    func testQ4gHeadComputesTargetsForEveryVerificationRow() throws {
        let hal = try QWHal(geometry: .qwen38_27b), hidden = QWModelGeometry.qwen38_27b.hidden, vocabulary = 33
        let blocks = vocabulary * (hidden / 32)
        let arena = try QWArena(hal: hal,
            weights: (blocks * 20) + 4096, activations: 1 << 20, kv: 0, gdnStates: 0)
        let weightArena = arena.weightsRegion()
        let blk = try weightArena.allocate(QWTensorShape([blocks * 10], .f16))
        memset(blk.region.pointer.advanced(by: blk.offset), 0, blk.bytes)
        let weight = try XCTUnwrap(QWHeadWeight(blocks: blk, n: vocabulary, k: hidden, isRotated: true))
        let window = try QWVerifyWindow(geometry: QWTestGeometry.g27, allocator: arena.activationsRegion(), rows: 2,
                                        vocabulary: vocabulary, gpuTargets: true)
        memset(window.hidden.region.pointer.advanced(by: window.hidden.offset), 0, window.hidden.bytes)
        let head = try QWHead(hal: hal, geometry: .qwen38_27b)
        let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
        do {
            try window.encodeHead(enc, hal: hal, head: head, storage: .q4g(weight))
        } catch {
            enc.endEncoding()
            throw error
        }
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        XCTAssertEqual(cb.status, .completed, "\(String(describing: cb.error))")
        XCTAssertEqual(try window.completedTargets(), [0, 0])
        let scores = window.logits.region.pointer.advanced(by: window.logits.offset)
            .assumingMemoryBound(to: Float.self)
        for index in 0..<(2 * vocabulary) { XCTAssertEqual(scores[index], 0) }
    }

    func testOrderedRowsRetainAllHiddenAndFullHeadLogits() throws {
        let hal = try QWHal(geometry: .qwen38_27b)
        let h = QWModelGeometry.qwen38_27b.hidden, v = 257
        XCTAssertEqual(QWVerifyWindow.allocationBytes(geometry: QWTestGeometry.g27, rows: 2), 2_027_776)
        XCTAssertEqual(QWVerifyWindow.allocationBytes(geometry: QWTestGeometry.g27, rows: 3), 3_041_536)
        for b in [2, 3] {
            let arena = try QWArena(hal: hal, weights: v * h * 2 + 256,
                                    activations: 1 << 20, kv: 256, gdnStates: 8 << 20)
            let a = arena.activationsRegion(), g = arena.gdnRegion()
            let window = try QWVerifyWindow(geometry: QWTestGeometry.g27, allocator: a, rows: b, vocabulary: v)
            XCTAssertEqual(a.usedBytes, QWVerifyWindow.allocationBytes(geometry: QWTestGeometry.g27, rows: b, vocabulary: v))
            let checkpoint = try QWGDNReplayCheckpoint(geometry: QWTestGeometry.g27, allocator: g, layers: 1, rows: b)
            let rec = try g.allocate(QWTensorShape([QWSlotGeometry.recElems(geometry: .qwen38_27b)], .f32))
            let ring = try g.allocate(QWTensorShape([QWSlotGeometry.ringElems(geometry: .qwen38_27b)], .f16))
            let staging = try a.allocate(QWTensorShape([h], .f16))
            let raw = try a.allocate(QWTensorShape([10240], .f16))
            let beta = try a.allocate(QWTensorShape([48], .f32))
            let decay = try a.allocate(QWTensorShape([48], .f32))
            memset(raw.region.pointer.advanced(by: raw.offset), 0, raw.bytes)
            memset(beta.region.pointer.advanced(by: beta.offset), 0, beta.bytes)
            memset(decay.region.pointer.advanced(by: decay.offset), 0, decay.bytes)
            let reference = try a.allocate(QWTensorShape([b * v], .f32))
            let w = try arena.weightsRegion().allocate(QWTensorShape([v * h], .f16))
            let wp = w.region.pointer.advanced(by: w.offset).assumingMemoryBound(to: Float16.self)
            let ep = window.embeddings.region.pointer.advanced(by: window.embeddings.offset).assumingMemoryBound(to: Float16.self)
            for i in 0..<(v * h) { wp[i] = Float16(Float(i % 97 - 48) / 137) }
            for i in 0..<(b * h) { ep[i] = Float16(Float((i * 7 + i / h) % 113 - 56) / 53) }
            memset(rec.region.pointer.advanced(by: rec.offset), 0x37, rec.bytes)
            memset(ring.region.pointer.advanced(by: ring.offset), 0x38, ring.bytes)
            func bytes(_ t: QWTensorValue) -> Data {
                Data(bytes: t.region.pointer.advanced(by: t.offset), count: t.bytes)
            }

            let originalRec = bytes(rec), originalRing = bytes(ring)
            _ = try hal.pipeline("qwg_verify_copy")
            let head = try QWHead(hal: hal, geometry: .qwen38_27b)
            a.markSteadyState()
            g.markSteadyState()
            let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
            var order: [Int] = []
            try window.encode(cb, hal: hal, head: head, checkpoint: checkpoint,
                              rec: rec, ring: ring, headStorage: .f16(w)) { row in
                order.append(row)
                let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
                defer { enc.endEncoding() }
                // A small stand-in producer; the production driver supplies its
                // shipping 64-layer row encoder at this exact orchestration seam.
                try QWVerifyWindow.encodeCopy(enc, hal: hal, source: window.embeddingRow(row), destination: staging)
                try checkpoint.encodeCapture(enc, hal: hal, layer: 0, row: row,
                                             rawInput: raw, postConv: raw, beta: beta, decay: decay)
                try QWVerifyWindow.encodeCopy(enc, hal: hal, source: staging, destination: window.hiddenRow(row))
                let out = QWTensorValue(region: reference.region, offset: reference.offset + row * v * 4,
                                         shape: QWTensorShape([v], .f32))
                try QWHead.encodeF16(enc, hal: hal, w: w, x: staging, y: out, geometry: .qwen38_27b, N: v)
            }
            cb.commit()
            cb.waitUntilCompleted()
            XCTAssertEqual(cb.status, .completed, "\(String(describing: cb.error))")
            XCTAssertEqual(order, Array(0..<b))
            XCTAssertEqual(bytes(window.hidden), bytes(window.embeddings), "all hidden rows survive staging reuse")
            XCTAssertEqual(bytes(window.logits), bytes(reference), "all FP32 head rows match repeated shipping GEMV")
            let lp = window.logits.region.pointer.advanced(by: window.logits.offset).assumingMemoryBound(to: Float.self)
            for i in 0..<(b * v) { lp[i] = -1 }
            for row in 0..<b { lp[row * v + row + 1] = 2; lp[row * v + row + 20] = 2 }
            let decision = try window.greedyDecision(drafts: Array(1..<b), maxNew: b)
            XCTAssertEqual(decision.tokens, Array(1...b), "argmax ties choose the lowest vocabulary ID")
            lp[0] = .nan
            XCTAssertThrowsError(try window.greedyDecision(drafts: Array(1..<b), maxNew: b))
            XCTAssertEqual(bytes(checkpoint.recurrent), originalRec)
            XCTAssertEqual(bytes(checkpoint.rings), originalRing)
            XCTAssertEqual(a.steadyStateAllocs, 0)
            XCTAssertEqual(g.steadyStateAllocs, 0)
            XCTAssertThrowsError(try window.hiddenRow(b))
            XCTAssertThrowsError(try window.embeddingRow(-1))
            let invalidCB = try XCTUnwrap(hal.queue.makeCommandBuffer())
            let invalidEnc = try XCTUnwrap(invalidCB.makeComputeCommandEncoder())
            XCTAssertThrowsError(try QWVerifyWindow.encodeCopy(invalidEnc, hal: hal, source: staging, destination: staging))
            let truncated = QWTensorValue(region: QWMRegion(buffer: staging.region.buffer,
                                                             base: staging.region.base, length: 16),
                                           offset: 0, shape: staging.shape)
            XCTAssertThrowsError(try QWVerifyWindow.encodeCopy(invalidEnc, hal: hal, source: truncated, destination: staging))
            invalidEnc.endEncoding()
            let incompleteCB = try XCTUnwrap(hal.queue.makeCommandBuffer())
            XCTAssertThrowsError(try window.encode(incompleteCB, hal: hal, head: head,
                                                    checkpoint: checkpoint, rec: rec, ring: ring,
                                                    headStorage: .f16(w)) { _ in },
                                 "verification cannot succeed without every layer/row journal capture")
            XCTAssertEqual(bytes(window.hidden), bytes(window.embeddings))
        }
    }
}
