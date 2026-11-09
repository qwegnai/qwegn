import XCTest
import Metal
import QWAlloc
import QWAttn
import QWCore
import QWGEMM
import QWHAL
import QWHead
import QWKV
import QWMLP
import QWQuant
import QWRope
@testable import QWEngine

final class QWMTPDraftEncoderTests: XCTestCase {
    private let fixtures = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Scripts/spike/out")

    private func load(_ name: String) throws -> Data {
        try Data(contentsOf: fixtures.appendingPathComponent(name))
    }

    private func tensor(_ data: Data, type: QWDType, hal: QWHal) throws -> QWTensorValue {
        let buffer = try XCTUnwrap(hal.device.makeBuffer(length: max(16, data.count), options: .storageModeShared))
        data.withUnsafeBytes { buffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        return QWTensorValue(region: QWMRegion(buffer: buffer, base: 0, length: buffer.length),
                             offset: 0, shape: QWTensorShape([data.count / (type == .f16 ? 2 : 4)], type))
    }

    func testDraftBlockRungsReplayAndFP16Head() throws {
        let hal = try QWHal(geometry: .qwen38_27b)
        let h = QWModelGeometry.qwen38_27b.hidden, intermediate = 1024, vocabulary = 33
        let arena = try QWArena(hal: hal, weights: vocabulary * h * 2 + 8192,
                                activations: 2 << 20, kv: 6 << 20,
                                gdnStates: QWAttnDecode.decodePartialsBytes(geometry: .qwen38_27b) + 256)
        func weight(_ names: [(String, Int)], k: Int) throws -> QWWeightView {
            var blk = Data()
            var n = 0
            for (name, rows) in names {
                blk.append(try load("\(name).bin"))
                n += rows
            }
            return try XCTUnwrap(QWWeightView(blocks: try tensor(blk, type: .f16, hal: hal), n: n, k: k, isRotated: true))
        }
        func norm(_ name: String) throws -> QWTensorValue {
            try tensor(load("mtpc_\(name).bin"), type: .f16, hal: hal)
        }
        let qkv = try weight([("mtpc_q", 12288), ("mtpc_k", 1024), ("mtpc_v", 1024)], k: h)
        func slice(_ first: Int, _ count: Int) throws -> QWWeightView {
            let rowElems = (h / 32) * 10
            let b = QWTensorValue(region: qkv.blocks.region, offset: qkv.blocks.offset + first * rowElems * 2,
                                  shape: QWTensorShape([count * rowElems], .f16))
            return try XCTUnwrap(QWWeightView(blocks: b, n: count, k: h, isRotated: true))
        }
        let gu = try weight([("mtpc_gu", 2 * intermediate)], k: h)
        let weights = QWMTPDraftWeights(
            preFcNormEmbedding: try norm("enorm_w"), preFcNormHidden: try norm("hnorm_w"),
            fc: try weight([("mtpc_fc", h)], k: 2 * h), inputNorm: try norm("ln1"),
            qkvProj: qkv, q: try slice(0, 12288), k: try slice(12288, 1024), v: try slice(13312, 1024),
            o: try weight([("mtpc_o", h)], k: 6144), wq: try norm("wq"), wkTiled: try norm("wk"),
            postAttnNorm: try norm("ln2"),
            gu: try XCTUnwrap(QWMLPWeight(blocks: gu.blocks, n: gu.n, k: gu.k, isRotated: true)),
            down: try weight([("mtpc_dn", h)], k: intermediate), norm: try norm("postnorm"))
        let g = arena.gdnRegion()
        let partialOffset = try g.allocate(bytes: QWAttnDecode.decodePartialsBytes(geometry: .qwen38_27b))
        let scratch = try QWMTPDraftScratch(geometry: QWTestGeometry.g27, 
            allocator: arena.activationsRegion(),
            partials: QWMRegion(buffer: g.region.buffer, base: g.region.base + partialOffset,
                               length: QWAttnDecode.decodePartialsBytes(geometry: .qwen38_27b)),
            intermediate: intermediate, vocabulary: vocabulary)
        let rope = try QWRopeApply(tables: QWRopeTables(rows: 20, yarn: nil, geometry: .qwen38_27b),
                                   hal: hal, weights: arena.weightsRegion(), arena: arena)
        let encoder = try QWMTPDraftEncoder(geometry: QWTestGeometry.g27, hal: hal, weights: weights, scratch: scratch, rope: rope, ropeRows: 20)
        let cache = try QWKVCache(hal: hal, arena: arena, layers: [0], mode: .f16, geometry: .qwen38_27b)
        let handle = try cache.allocate(promptTokens: 256)
        defer { cache.finish(handle) }
        let seedRows = 15, kvElems = 1024
        for (kind, name) in [(QWKVKind.k, "mtpc_seedkq_k.bin"), (.v, "mtpc_seedkq_v.bin")] {
            let data = try load(name)
            let rows = data.withUnsafeBytes { p -> [UInt16] in
                let bits = p.bindMemory(to: UInt16.self)
                var output = [UInt16](repeating: 0, count: seedRows * kvElems)
                for t in 0..<seedRows {
                    for head in 0..<4 {
                        for d in 0..<256 { output[t * kvElems + head * 256 + d] = bits[(head * seedRows + t) * 256 + d] }
                    }
                }
                return output
            }
            try cache.append(handle, positions: 0..<seedRows, layer: 0, kind: kind, src: rows)
        }
        let embedding = try norm("e"), hidden = try norm("cast_f16")
        let headWeight = try arena.weightsRegion().allocate(QWTensorShape([vocabulary * h], .f16))
        let wp = headWeight.region.pointer.advanced(by: headWeight.offset).assumingMemoryBound(to: Float16.self)
        for i in 0..<(vocabulary * h) { wp[i] = 0 }
        for row in 0..<vocabulary { wp[row * h + (row * 73) % h] = Float16(row % 2 == 0 ? 1 : -1) }
        arena.activationsRegion().markSteadyState()
        g.markSteadyState()

        func run(position: Int, hiddenInput: QWTensorValue? = nil, rows: Int = 15) throws {
            let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
            let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
            do {
                try encoder.encode(enc, embedding: embedding, hidden: hiddenInput ?? hidden,
                                   kv: cache.layerViews(handle, layer: 0),
                                   cacheRows: rows, absolutePosition: position)
                try encoder.encodeHead(enc, weight: headWeight)
            } catch {
                enc.endEncoding()
                throw error
            }
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            guard cb.status == .completed else {
                throw QWEngineDriverError.state("test draft GPU failure: \(String(describing: cb.error))")
            }
            cache.markWritten(handle, upTo: rows + 1)
        }
        func bytes(_ t: QWTensorValue) -> Data {
            Data(bytes: t.region.pointer.advanced(by: t.offset), count: t.bytes)
        }
        func pairBytes(start: Int = 0, count: Int = 2) throws -> Data {
            let views = try cache.layerViews(handle, layer: 0)
            var data = Data()
            for view in [views.k, views.v] {
                let table = view.table.region.pointer.advanced(by: view.table.offset).assumingMemoryBound(to: Int32.self)
                for row in start..<(start + count) {
                    let pageID = table[row / 256]
                    data.append(Data(bytes: view.pool.pointer.advanced(by: Int(pageID) * view.pageBytes + row % 256 * 1024 * 2),
                                     count: 1024 * 2))
                }
            }
            return data
        }
        func parity(_ t: QWTensorValue, reference: String, maxUlp: Double = 64) throws {
            let ref = try load(reference)
            XCTAssertEqual(t.bytes, ref.count, reference)
            let got = t.region.pointer.advanced(by: t.offset).assumingMemoryBound(to: Float16.self)
            ref.withUnsafeBytes {
                let expected = $0.bindMemory(to: UInt16.self)
                var violations = 0, worst = 0.0
                for i in 0..<expected.count {
                    let a = Double(Float(got[i])), b = Double(QWQuant.f16tof32Bits(expected[i]))
                    let error = abs(a - b)
                    let spacing = min(Double(got[i].ulp), Double(Float16(bitPattern: expected[i]).ulp))
                    if !a.isFinite || !b.isFinite || (error > 1e-2 && error > maxUlp * spacing) { violations += 1 }
                    worst = max(worst, error)
                }
                XCTAssertEqual(violations, 0, "\(reference): worst \(worst)")
            }
        }

        try run(position: 16)
        try parity(scratch.enorm, reference: "mtpc_enorm_ref.bin", maxUlp: 1)
        try parity(scratch.hnorm, reference: "mtpc_hnorm_ref.bin", maxUlp: 1)
        try parity(scratch.ln1, reference: "mtpc_x1h.bin")
        try parity(scratch.qRope, reference: "mtpc_qdoor_ref.bin")
        try parity(scratch.kRope, reference: "mtpc_knew_ref.bin")
        try parity(scratch.attOut, reference: "mtpc_att_out.bin")
        try parity(scratch.ln2, reference: "mtpc_ln2a.bin")
        try parity(scratch.mlpIntermediate, reference: "mtpc_h.bin", maxUlp: 96)
        try parity(scratch.postNorm, reference: "mtpc_z_step.bin")
        let z = scratch.postNorm.region.pointer.advanced(by: scratch.postNorm.offset).assumingMemoryBound(to: Float16.self)
        let logits = scratch.draftLogits.region.pointer.advanced(by: scratch.draftLogits.offset).assumingMemoryBound(to: Float.self)
        for row in 0..<vocabulary {
            let expected = Float(z[(row * 73) % h]) * (row % 2 == 0 ? 1 : -1)
            XCTAssertEqual(logits[row], expected, "full FP16 head row \(row)")
        }
        let firstHidden = bytes(scratch.postNorm), firstRope = bytes(scratch.qRope)
        try cache.rollback(handle, by: 1)
        try run(position: 16)
        XCTAssertEqual(bytes(scratch.postNorm), firstHidden, "restored pair cache must reproduce the draft")
        try cache.rollback(handle, by: 1)
        try run(position: 18)
        XCTAssertNotEqual(bytes(scratch.qRope), firstRope, "RoPE must use absolute position, not physical pair count")
        try cache.rollback(handle, by: 1)
        try run(position: 16)
        let firstToken = try QWVerifyWindow.greedyToken(UnsafeBufferPointer(start: logits, count: vocabulary))
        let copiedHidden = try tensor(firstHidden, type: .f16, hal: hal)
        try run(position: 17, hiddenInput: copiedHidden, rows: 16)
        let expectedChain = bytes(scratch.postNorm)
        let secondToken = try QWVerifyWindow.greedyToken(UnsafeBufferPointer(start: logits, count: vocabulary))
        let referenceDraftPairs = [try pairBytes(start: seedRows, count: 1),
                                   try pairBytes(start: seedRows, count: 2)]
        try cache.rollback(handle, by: 2)
        let chainCB = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let chainEnc = try XCTUnwrap(chainCB.makeComputeCommandEncoder())
        do {
            let views = try cache.layerViews(handle, layer: 0)
            try encoder.encode(chainEnc, embedding: embedding, hidden: hidden,
                               kv: views, cacheRows: 15, absolutePosition: 16)
            try encoder.encode(chainEnc, embedding: embedding, hidden: scratch.postNorm,
                               kv: views, cacheRows: 16, absolutePosition: 17)
        } catch {
            chainEnc.endEncoding()
            throw error
        }
        chainEnc.endEncoding()
        XCTAssertEqual(cache.written(handle), 15, "encoding must not publish pair-cache cursors")
        chainCB.commit()
        chainCB.waitUntilCompleted()
        XCTAssertEqual(chainCB.status, .completed, "\(String(describing: chainCB.error))")
        XCTAssertEqual(bytes(scratch.postNorm), expectedChain, "GPU hidden chaining must match separate reference steps")
        cache.markWritten(handle, upTo: 17)
        XCTAssertEqual(arena.activationsRegion().steadyStateAllocs, 0)
        XCTAssertEqual(g.steadyStateAllocs, 0)

        for selection in [QWMTPDraftChain.Selection.cpuReference, .gpuChecked] {
            for count in 1...2 {
                try cache.rollback(handle, by: cache.written(handle) - seedRows)
                var embeddingTokens: [Int] = []
                let tokens = try QWMTPDraftChain.run(
                    hal: hal, draft: encoder, headWeight: headWeight, hidden: hidden,
                    kv: cache.layerViews(handle, layer: 0), pendingToken: 0, drafts: count,
                    position: 16, cacheRows: seedRows, selection: selection,
                    embedding: { token, destination in
                        embeddingTokens.append(token)
                        destination.region.pointer.advanced(by: destination.offset)
                            .copyMemory(from: embedding.region.pointer.advanced(by: embedding.offset),
                                        byteCount: embedding.bytes)
                    },
                    gpuFailure: { XCTFail("draft chain command buffer failed") })
                XCTAssertEqual(tokens, Array([firstToken, secondToken].prefix(count)))
                XCTAssertEqual(embeddingTokens, Array([0, firstToken].prefix(count)),
                               "the next pair must embed the sampled draft token")
                XCTAssertEqual(bytes(scratch.postNorm), count == 1 ? firstHidden : expectedChain)
                XCTAssertEqual(try pairBytes(start: seedRows, count: count), referenceDraftPairs[count - 1],
                               "draft cache rows must preserve chained MTP hidden, not verified trunk hidden")
                XCTAssertEqual(cache.written(handle), seedRows, "chain must leave publication to the round owner")
                cache.markWritten(handle, upTo: seedRows + count)
                XCTAssertEqual(arena.activationsRegion().steadyStateAllocs, 0)
                XCTAssertEqual(g.steadyStateAllocs, 0)
            }
        }
        let mappedPath = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-draft-embedding-\(UUID().uuidString)").path
        defer {
            do { try FileManager.default.removeItem(atPath: mappedPath) }
            catch { XCTFail("draft mapping cleanup: \(error)") }
        }
        let source = embedding.region.pointer.advanced(by: embedding.offset).assumingMemoryBound(to: UInt16.self)
        let bf16 = (0..<h).map { UInt16(QWQuant.f16tof32Bits(source[$0]).bitPattern >> 16) }
        var tableData = Data()
        for _ in 0..<vocabulary { bf16.withUnsafeBytes { tableData.append(contentsOf: $0) } }
        try tableData.write(to: URL(fileURLWithPath: mappedPath))
        let table = try QWMTPEmbeddingTable(hal: hal, path: mappedPath, offset: 0,
                                          vocabulary: vocabulary, hidden: h)
        for count in 1...2 {
            try cache.rollback(handle, by: cache.written(handle) - seedRows)
            let cpuTokens = try QWMTPDraftChain.run(
                hal: hal, draft: encoder, headWeight: headWeight, hidden: hidden,
                kv: cache.layerViews(handle, layer: 0), pendingToken: 0, drafts: count,
                position: 16, cacheRows: seedRows, selection: .gpuChecked,
                embedding: { _, destination in
                    let out = destination.region.pointer.advanced(by: destination.offset).assumingMemoryBound(to: UInt16.self)
                    for i in 0..<h { out[i] = QWQuant.f32tof16Bits(QWQuant.bf16tof32Bits(bf16[i])) }
                }, gpuFailure: { XCTFail("CPU-embedding chain GPU failure") })
            let expectedHidden = bytes(scratch.postNorm)
            let expectedPairs = try pairBytes(start: seedRows, count: count)
            let gpuTokens = try QWMTPDraftChain.run(
                hal: hal, draft: encoder, headWeight: headWeight, hidden: hidden,
                kv: cache.layerViews(handle, layer: 0), pendingToken: 0, drafts: count,
                position: 16, cacheRows: seedRows, selection: .gpuChecked, deviceEmbedding: table,
                embedding: { _, _ in XCTFail("device chain must not gather on CPU") },
                gpuFailure: { XCTFail("device chain GPU failure") })
            XCTAssertEqual(gpuTokens, cpuTokens)
            XCTAssertEqual(bytes(scratch.postNorm), expectedHidden)
            XCTAssertEqual(try pairBytes(start: seedRows, count: count), expectedPairs)
            XCTAssertEqual(cache.written(handle), seedRows)
        }
        let invalidWords = [UInt16](repeating: 0x7e00, count: vocabulary * h)
        let invalidData = invalidWords.withUnsafeBytes { Data($0) }
        let invalidHead = try tensor(invalidData, type: .f16, hal: hal)
        XCTAssertThrowsError(try QWMTPDraftChain.run(
            hal: hal, draft: encoder, headWeight: invalidHead, hidden: hidden,
            kv: cache.layerViews(handle, layer: 0), pendingToken: 0, drafts: 2,
            position: 16, cacheRows: seedRows, selection: .gpuChecked, deviceEmbedding: table,
            embedding: { _, _ in XCTFail("invalid device chain must not gather on CPU") },
            gpuFailure: { XCTFail("invalid scores must drain without a command-buffer failure") })) { error in
                XCTAssertTrue(String(describing: error).contains("invalid token -1 at row 0"))
            }
        let invalidIDs = table.tokens.contents().assumingMemoryBound(to: Int32.self)
        XCTAssertEqual(invalidIDs[0], -1, "each invalid selection must survive later dispatches")
        XCTAssertEqual(invalidIDs[1], -1)
        XCTAssertEqual(cache.written(handle), seedRows, "invalid feedback must not publish cache cursors")
        var invalidEmbeddingCalled = false
        XCTAssertThrowsError(try QWMTPDraftChain.run(
            hal: hal, draft: encoder, headWeight: headWeight, hidden: hidden,
            kv: cache.layerViews(handle, layer: 0), pendingToken: 0, drafts: 0,
            position: 16, cacheRows: seedRows,
            embedding: { _, _ in invalidEmbeddingCalled = true },
            gpuFailure: { XCTFail("invalid chain must not submit GPU work") }))
        XCTAssertFalse(invalidEmbeddingCalled)

        // Three independent trunk hidden rows, deliberately not MTP outputs.
        var embeddingRows = Data(), hiddenRows = Data()
        embeddingRows.append(try load("mtpc_e.bin"))
        embeddingRows.append(try load("mtpc_cast_f16.bin"))
        embeddingRows.append(try load("mtpc_e.bin"))
        hiddenRows.append(try load("mtpc_cast_f16.bin"))
        hiddenRows.append(firstHidden)
        hiddenRows.append(expectedChain)
        let promptEmbedding = try tensor(embeddingRows, type: .f16, hal: hal)
        let promptHidden = try tensor(hiddenRows, type: .f16, hal: hal)
        let carriedHidden = try tensor(try load("mtpc_cast_f16.bin"), type: .f16, hal: hal)
        func rowsView(_ t: QWTensorValue, start: Int, count: Int) -> QWTensorValue {
            QWTensorValue(region: t.region, offset: t.offset + start * h * 2,
                          shape: QWTensorShape([count * h], .f16))
        }
        func resetPairs() throws {
            try cache.rollback(handle, by: cache.written(handle))
            try cache.grow(handle, to: 256)
        }
        func seed(_ chunks: [Int], physicalStart: Int = 0) throws {
            var position = 0, pairRows = physicalStart
            let kv = try cache.layerViews(handle, layer: 0)
            for count in chunks {
                let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
                try QWMTPPromptSeeder.encodeChunk(cb, draft: encoder,
                                                  embeddings: rowsView(promptEmbedding, start: position, count: count),
                                                  trunkHidden: rowsView(promptHidden, start: position, count: count),
                                                  previousHidden: carriedHidden,
                                                  position: position, count: count, cacheRows: pairRows, kv: kv)
                XCTAssertEqual(cache.written(handle), pairRows, "encoding must not publish pair cursors")
                cb.commit()
                cb.waitUntilCompleted()
                XCTAssertEqual(cb.status, .completed, "\(String(describing: cb.error))")
                pairRows += QWMTPPromptSeeder.pairCount(position: position, count: count)
                position += count
                cache.markWritten(handle, upTo: pairRows)
                XCTAssertEqual(bytes(carriedHidden), bytes(rowsView(promptHidden, start: position - 1, count: 1)),
                               "carry last trunk hidden, including P=1")
            }
        }
        try resetPairs()
        let referenceCB = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let referenceEnc = try XCTUnwrap(referenceCB.makeComputeCommandEncoder())
        do {
            let kv = try cache.layerViews(handle, layer: 0)
            for t in 1..<3 {
                try encoder.encode(referenceEnc,
                                   embedding: rowsView(promptEmbedding, start: t, count: 1),
                                   hidden: rowsView(promptHidden, start: t - 1, count: 1),
                                   kv: kv, cacheRows: t - 1, absolutePosition: t)
            }
        } catch {
            referenceEnc.endEncoding()
            throw error
        }
        referenceEnc.endEncoding()
        referenceCB.commit()
        referenceCB.waitUntilCompleted()
        XCTAssertEqual(referenceCB.status, .completed)
        cache.markWritten(handle, upTo: 2)
        let referencePairs = try pairBytes()
        for chunks in [[3], [1, 2], [2, 1], [1, 1, 1]] {
            try resetPairs()
            try seed(chunks)
            XCTAssertEqual(try pairBytes(), referencePairs, "shifted prompt pair parity: chunks \(chunks)")
        }
        try resetPairs()
        try cache.grow(handle, to: 512)
        cache.markWritten(handle, upTo: 255)
        try seed([3], physicalStart: 255)
        XCTAssertEqual(try pairBytes(start: 255), referencePairs, "seed scatter must honor page tables at 255/256")
        XCTAssertEqual(QWMTPPromptSeeder.pairCount(position: 0, count: 1), 0)
        XCTAssertEqual(QWMTPPromptSeeder.pairCount(position: 3, count: 1), 1)

        let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
        defer { enc.endEncoding() }
        let before = bytes(scratch.postNorm)
        XCTAssertThrowsError(try encoder.encode(enc, embedding: embedding, hidden: hidden,
                                               kv: cache.layerViews(handle, layer: 0),
                                               cacheRows: 16, absolutePosition: 20))
        XCTAssertEqual(bytes(scratch.postNorm), before)
        XCTAssertThrowsError(try QWMTPPromptSeeder.encode(enc, draft: encoder, embeddings: promptEmbedding,
                                                         trunkHidden: promptHidden, previousHidden: hidden,
                                                         position: 19, count: 3, cacheRows: 2,
                                                         kv: cache.layerViews(handle, layer: 0)))
        XCTAssertThrowsError(try QWMTPPromptSeeder.encode(enc, draft: encoder, embeddings: promptEmbedding,
                                                         trunkHidden: promptHidden, previousHidden: hidden,
                                                         position: 0, count: 3, cacheRows: 511,
                                                         kv: cache.layerViews(handle, layer: 0)))
        XCTAssertThrowsError(try QWMTPPromptSeeder.encode(enc, draft: encoder, embeddings: promptEmbedding,
                                                         trunkHidden: promptHidden,
                                                         previousHidden: rowsView(promptHidden, start: 0, count: 1),
                                                         position: 0, count: 3, cacheRows: 2,
                                                         kv: cache.layerViews(handle, layer: 0)))
        XCTAssertEqual(bytes(scratch.postNorm), before)
    }
}
