import XCTest
import Metal
import Foundation
@testable import QWAttn
import QWCore
import QWHAL
import QWAlloc
import QWKV
import QWQuant

final class QWLSATests: XCTestCase {
    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()
    static let geometry = QWModelGeometry.qwen38_27b

    static func makeOperator(maxContext: Int = 8192, budget: Int = 512) throws -> QWLSAOperator {
        let cfg = QWLSAConfig(mode: .svd, budget: budget, clusterBlock: 512)
        let inDim = geometry.lsaInputDim
        let basis = QWLSASidecar.synthesizeDeterministicOrthogonalBasis(inDim: inDim, indexDim: 64, seed: 101)
        let lw = QWLSAOperator.LayerWeights(physicalLayer: 3, wqIdxF16: basis, wkIdxF16: basis)
        return try QWLSAOperator(
            hal: hal,
            geometry: geometry,
            config: cfg,
            indexDim: 64,
            layerWeights: [lw],
            maxContext: maxContext,
            maxPrefillChunk: 512
        )
    }

    func testWardClusteringInvariantsAndPartitionOf128Blocks() throws {
        let op = try Self.makeOperator(maxContext: 4096, budget: 512)
        let T = 512
        let qgate = QWAttnPagedPrefillTests.synthQGate(n: T, seed: 0x1234)
        let kIn = QWAttnPagedPrefillTests.synthKV(n: T, seed: 0x5678)

        let qBuf = Self.hal.device.makeBuffer(bytes: qgate, length: qgate.count * 2, options: .storageModeShared)!
        let kBuf = Self.hal.device.makeBuffer(bytes: kIn, length: kIn.count * 2, options: .storageModeShared)!
        let qTV = QWTensorValue(region: QWMRegion(buffer: qBuf, base: 0, length: qBuf.length),
                                offset: 0, shape: QWTensorShape([T * Self.geometry.qProjOut], .f16))
        let kTV = QWTensorValue(region: QWMRegion(buffer: kBuf, base: 0, length: kBuf.length),
                                offset: 0, shape: QWTensorShape([T * Self.geometry.kvChannels], .f16))

        guard let (cb, enc) = Self.hal.begin() else { XCTFail("hal.begin failed"); return }
        op.encodeIndexAndCluster(enc, qgate: qgate.isEmpty ? qTV : qTV, kIn: kTV, fullAttnLayer: 0, pos0: 0, rows: T)
        Self.hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)

        let ls = op.layers[0]
        let basePtr = op.region.pointer

        // 1. Check bucket sizes sum to 512 and each size is in {4, 8, 12, 16}
        let sizesPtr = basePtr.advanced(by: ls.bucketSizes.offset).assumingMemoryBound(to: UInt8.self)
        var totalTokens = 0
        for b in 0..<64 {
            let sz = Int(sizesPtr[b])
            XCTAssertTrue([4, 8, 12, 16].contains(sz), "bucket \(b) has invalid size \(sz)")
            totalTokens += sz
        }
        XCTAssertEqual(totalTokens, 512, "sum of bucket token counts must equal 512")

        // 2. Check bucket members form an exact partition of 0..<128
        let membersPtr = basePtr.advanced(by: ls.bucketMembers.offset).assumingMemoryBound(to: UInt8.self)
        var seenBlocks = Set<Int>()
        for b in 0..<64 {
            let nBlocks = Int(sizesPtr[b]) / 4
            for m in 0..<nBlocks {
                let blk = Int(membersPtr[b * 4 + m])
                XCTAssertTrue((0..<128).contains(blk), "invalid member block \(blk) in bucket \(b)")
                XCTAssertFalse(seenBlocks.contains(blk), "duplicate member block \(blk) in bucket \(b)")
                seenBlocks.insert(blk)
            }
        }
        XCTAssertEqual(seenBlocks.count, 128, "all 128 pooled blocks must be partitioned across the 64 Ward buckets")

        // 3. Check all 64 centroids have unit L2 norm
        let centPtr = basePtr.advanced(by: ls.centroids.offset).assumingMemoryBound(to: UInt16.self)
        for b in 0..<64 {
            var normSq: Float = 0
            for d in 0..<op.indexDim {
                let v = Float(Float16(bitPattern: centPtr[b * op.indexDim + d]))
                normSq += v * v
            }
            XCTAssertEqual(sqrt(normSq), 1.0, accuracy: 2e-2, "centroid \(b) is not unit-normalized")
        }
    }

    func testBlockSelectorShortAndLongContextInvariants() throws {
        let op = try Self.makeOperator(maxContext: 8192, budget: 512) // budgetBlocks = 128, windowBlocks = 128
        let chunk = 512
        let qgate = QWAttnPagedPrefillTests.synthQGate(n: chunk, seed: 0xABCD)
        let kIn = QWAttnPagedPrefillTests.synthKV(n: chunk, seed: 0xEF01)
        let qBuf = Self.hal.device.makeBuffer(bytes: qgate, length: qgate.count * 2, options: .storageModeShared)!
        let kBuf = Self.hal.device.makeBuffer(bytes: kIn, length: kIn.count * 2, options: .storageModeShared)!
        let qTV = QWTensorValue(region: QWMRegion(buffer: qBuf, base: 0, length: qBuf.length),
                                offset: 0, shape: QWTensorShape([chunk * Self.geometry.qProjOut], .f16))
        let kTV = QWTensorValue(region: QWMRegion(buffer: kBuf, base: 0, length: kBuf.length),
                                offset: 0, shape: QWTensorShape([chunk * Self.geometry.kvChannels], .f16))

        // Populate 8 superblocks = 4096 tokens
        for sb in 0..<8 {
            guard let (cb, enc) = Self.hal.begin() else { XCTFail("hal.begin"); return }
            op.encodeIndexAndCluster(enc, qgate: qTV, kIn: kTV, fullAttnLayer: 0, pos0: sb * chunk, rows: chunk)
            Self.hal.end(cb, enc)
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error)
        }

        // Now run block selection at pos0 = 4095 (1024 completed 4-token blocks > 1 + 128 + 128 = 257)
        guard let (cb, enc) = Self.hal.begin() else { XCTFail("hal.begin"); return }
        op.encodeIndexAndCluster(enc, qgate: qTV.slice(byteOffset: 0, count: Self.geometry.qProjOut),
                                 kIn: kTV.slice(byteOffset: 0, count: Self.geometry.kvChannels),
                                 fullAttnLayer: 0, pos0: 4095, rows: 1)
        op.encodeSelectBlocks(enc, fullAttnLayer: 0, pos0: 4095, rows: 1)
        Self.hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)

        let basePtr = op.region.pointer
        let count = Int(basePtr.advanced(by: op.blockCountScratch.offset).assumingMemoryBound(to: UInt32.self)[0])
        XCTAssertEqual(count, 1 + 128 + 128) // sink(1) + budgetBlocks(128) + windowBlocks(128) = 257

        let listPtr = basePtr.advanced(by: op.blockListScratch.offset).assumingMemoryBound(to: Int32.self)
        let selected = (0..<count).map { Int(listPtr[$0]) }
        // Must start with sink block 0
        XCTAssertEqual(selected.first, 0)
        // Must be strictly ascending
        for i in 1..<selected.count {
            XCTAssertLessThan(selected[i - 1], selected[i], "selected block list not strictly ascending at \(i)")
        }
        // Last 128 blocks must be the local window [1024 - 128 ..< 1024] = [896 ..< 1024]
        let expectedWindow = Array(896..<1024)
        XCTAssertEqual(Array(selected.suffix(128)), expectedWindow)
    }

    func testSparseDecodeMatchesDenseWhenAllBlocksSelected() throws {
        let C = 300 // 75 four-token blocks + 1 tail token (at pos=300)
        let (cache, _, partials) = try QWAttnDecodeTests.makeEnv(tokensCeiling: C + 256)
        let seq = try cache.allocate(promptTokens: C + 1)

        let kAll = QWAttnPagedPrefillTests.synthKV(n: C + 1, seed: 0x1111)
        let vAll = QWAttnPagedPrefillTests.synthV(n: C + 1, seed: 0x2222)
        let qgate = QWAttnPagedPrefillTests.synthQGate(n: 1, seed: 0x3333)
        let kvRow = Self.geometry.kvChannels
        try cache.append(seq, positions: 0..<C, layer: 3, kind: .k, src: Array(kAll[0..<(C * kvRow)]))
        try cache.append(seq, positions: 0..<C, layer: 3, kind: .v, src: Array(vAll[0..<(C * kvRow)]))
        let views = try cache.layerViews(seq, layer: 3)

        let knew = Array(kAll[(C * kvRow)..<((C + 1) * kvRow)])
        let vnew = Array(vAll[(C * kvRow)..<((C + 1) * kvRow)])
        let qTV = QWAttnDecodeTests.tv(QWAttnPagedPrefillTests.buf(qgate), qgate.count)
        let knTV = QWAttnDecodeTests.tv(QWAttnPagedPrefillTests.buf(knew), knew.count)
        let vnTV = QWAttnDecodeTests.tv(QWAttnPagedPrefillTests.buf(vnew), vnew.count)

        let outDenseBuf = Self.hal.device.makeBuffer(length: Self.geometry.qChannels * 2, options: .storageModeShared)!
        let outSparseBuf = Self.hal.device.makeBuffer(length: Self.geometry.qChannels * 2, options: .storageModeShared)!
        let outDenseTV = QWAttnDecodeTests.tv(outDenseBuf, Self.geometry.qChannels)
        let outSparseTV = QWAttnDecodeTests.tv(outSparseBuf, Self.geometry.qChannels)

        let totalBlocks = (C + 1) / 4
        let blockListArr = (0..<totalBlocks).map { Int32($0) }
        var blockCountVal = UInt32(totalBlocks)
        let listBuf = Self.hal.device.makeBuffer(bytes: blockListArr, length: max(16, blockListArr.count * 4), options: .storageModeShared)!
        let countBuf = Self.hal.device.makeBuffer(bytes: &blockCountVal, length: 16, options: .storageModeShared)!
        let listTV = QWTensorValue(region: QWMRegion(buffer: listBuf, base: 0, length: listBuf.length),
                                   offset: 0, shape: QWTensorShape([totalBlocks], .i32))
        let countTV = QWTensorValue(region: QWMRegion(buffer: countBuf, base: 0, length: 4),
                                    offset: 0, shape: QWTensorShape([1], .i32))

        let dec = try QWAttnDecode(hal: Self.hal, kvMode: .f16, geometry: Self.geometry)
        let splits = QWAttnDecode.splits(forKeys: C + 1)
        guard let (cb, enc) = Self.hal.begin() else { XCTFail("hal.begin"); return }
        try dec.encode(enc, qgate: qTV, kv: views, knew: knTV, vnew: vnTV,
                       partials: partials, out: outDenseTV, C: C, splits: splits)
        enc.memoryBarrier(scope: .buffers)
        try dec.encodeSparse(enc, qgate: qTV, kv: views, knew: knTV, vnew: vnTV,
                             partials: partials, out: outSparseTV,
                             blockList: listTV, blockCount: countTV,
                             C: C, splits: splits, listStride: totalBlocks)
        Self.hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)

        let denseF = QWAttnDecodeTests.readF(QWAttnDecodeTests.u16row(outDenseBuf, byteOffset: 0, Self.geometry.qChannels))
        let sparseF = QWAttnDecodeTests.readF(QWAttnDecodeTests.u16row(outSparseBuf, byteOffset: 0, Self.geometry.qChannels))
        var maxDiff: Float = 0
        for i in 0..<denseF.count {
            maxDiff = max(maxDiff, abs(denseF[i] - sparseF[i]))
        }
        XCTAssertLessThan(maxDiff, 1e-3, "sparse decode with full block list must match dense decode")
    }

    func testSparsePagedPrefillMatchesDenseWhenAllBlocksSelected() throws {
        try XCTSkipUnless(
            Self.hal.caps.supportsSimdgroupMMA,
            "Sparse paged MMA/NAX prefill requires Apple7+ hardware"
        )
        let pos = 64
        let T = 8
        let sp = 32
        let ceiling = pos + T + 256
        let (cache, _, _) = try QWAttnDecodeTests.makeEnv(tokensCeiling: ceiling)
        let h = try cache.allocate(promptTokens: ceiling)

        let kAll = QWAttnPagedPrefillTests.synthKV(n: pos + T, seed: 0x4444)
        let vAll = QWAttnPagedPrefillTests.synthV(n: pos + T, seed: 0x5555)
        try cache.append(h, positions: 0..<(pos + T), layer: 3, kind: .k, src: kAll)
        try cache.append(h, positions: 0..<(pos + T), layer: 3, kind: .v, src: vAll)
        let views = try cache.layerViews(h, layer: 3)

        let qgateBits = QWAttnPagedPrefillTests.synthQGate(n: sp, seed: 0x6666)
        let qgateBuf = QWAttnPagedPrefillTests.buf(qgateBits)
        let qgateTV = QWAttnPagedPrefillTests.tv(qgateBuf, sp * Self.geometry.qProjOut)

        let outDenseBuf = Self.hal.device.makeBuffer(length: T * Self.geometry.qChannels * 2, options: .storageModeShared)!
        let outSparseBuf = Self.hal.device.makeBuffer(length: T * Self.geometry.qChannels * 2, options: .storageModeShared)!
        let outDenseTV = QWAttnPagedPrefillTests.tv(outDenseBuf, T * Self.geometry.qChannels)
        let outSparseTV = QWAttnPagedPrefillTests.tv(outSparseBuf, T * Self.geometry.qChannels)

        let listStride = (pos + T) / 4 + 4
        var blockListArr = [Int32](repeating: 0, count: T * listStride)
        var blockCountArr = [UInt32](repeating: 0, count: T)
        for r in 0..<T {
            let nBlocks = (pos + r + 1) / 4
            blockCountArr[r] = UInt32(nBlocks)
            for b in 0..<nBlocks {
                blockListArr[r * listStride + b] = Int32(b)
            }
        }
        let listBuf = Self.hal.device.makeBuffer(bytes: blockListArr, length: max(16, blockListArr.count * 4), options: .storageModeShared)!
        let countBuf = Self.hal.device.makeBuffer(bytes: blockCountArr, length: max(16, blockCountArr.count * 4), options: .storageModeShared)!
        let listTV = QWTensorValue(region: QWMRegion(buffer: listBuf, base: 0, length: listBuf.length),
                                   offset: 0, shape: QWTensorShape([T * listStride], .i32))
        let countTV = QWTensorValue(region: QWMRegion(buffer: countBuf, base: 0, length: countBuf.length),
                                    offset: 0, shape: QWTensorShape([T], .i32))

        let pagedPrefill = try QWAttnPagedPrefill(hal: Self.hal, geometry: Self.geometry)
        guard let (cb, enc) = Self.hal.begin() else { XCTFail("hal.begin"); return }
        try pagedPrefill.encode(enc, qgate: qgateTV, kv: views, out: outDenseTV, pos: pos, T: T)
        enc.memoryBarrier(scope: .buffers)
        try pagedPrefill.encodeSparse(enc, qgate: qgateTV, kv: views, out: outSparseTV,
                                      blockList: listTV, blockCount: countTV,
                                      pos: pos, T: T, listStride: listStride)
        Self.hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)

        let denseF = QWAttnPagedPrefillTests.readOut(outDenseBuf, T * Self.geometry.qChannels)
        let sparseF = QWAttnPagedPrefillTests.readOut(outSparseBuf, T * Self.geometry.qChannels)
        let maxDiff = denseF.indices.reduce(Float(0)) { max($0, abs(denseF[$1] - sparseF[$1])) }
        XCTAssertLessThan(maxDiff, 1e-3, "sparse paged prefill with full block list must match dense paged prefill")
    }

    func testSparsePagedPrefillQuantKVAndCausalBoundaryParity() throws {
        try XCTSkipUnless(
            Self.hal.caps.supportsSimdgroupMMA,
            "Sparse paged MMA/NAX prefill requires Apple7+ hardware"
        )
        let pos = 256, T = 40, sp = 64, ceiling = pos + T + 256
        let kAll = QWAttnPagedPrefillTests.synthKV(n: pos + T, seed: 0x7711)
        let vAll = QWAttnPagedPrefillTests.synthV(n: pos + T, seed: 0x8822)
        let qgateBits = QWAttnPagedPrefillTests.synthQGate(n: sp, seed: 0x9933)
        let qgateTV = QWAttnPagedPrefillTests.tv(QWAttnPagedPrefillTests.buf(qgateBits), sp * Self.geometry.qProjOut)
        let prefixBlocks: [Int32] = [0, 3, 9, 17, 31, 48, 60, 63]
        let listStride = prefixBlocks.count + 16
        var blockList = [Int32](repeating: 0, count: T * listStride)
        var blockCount = [UInt32](repeating: 0, count: T)
        for r in 0..<T {
            let cur = ((pos + r + 1) >> 2) - 64
            blockCount[r] = UInt32(prefixBlocks.count + cur)
            for (i, b) in prefixBlocks.enumerated() { blockList[r * listStride + i] = b }
            for i in 0..<cur { blockList[r * listStride + prefixBlocks.count + i] = Int32(64 + i) }
        }
        let lBuf = Self.hal.device.makeBuffer(bytes: blockList, length: blockList.count * 4, options: .storageModeShared)!
        let cBuf = Self.hal.device.makeBuffer(bytes: blockCount, length: blockCount.count * 4, options: .storageModeShared)!
        let lTV = QWTensorValue(region: QWMRegion(buffer: lBuf, base: 0, length: lBuf.length),
                                offset: 0, shape: QWTensorShape([T * listStride], .i32))
        let cTV = QWTensorValue(region: QWMRegion(buffer: cBuf, base: 0, length: cBuf.length),
                                offset: 0, shape: QWTensorShape([T], .i32))
        let paged = try QWAttnPagedPrefill(hal: Self.hal, geometry: Self.geometry)
        let gen = try QWAttnPagedPrefillGeneral(hal: Self.hal, geometry: Self.geometry)
        for mode in [QWKVMode.f16, .q8, .q4] {
            let needPages = (ceiling + 255) / 256
            let arena = try QWArena(hal: Self.hal, weights: 0, activations: 0,
                                    kv: QWKVCache.pageBytes(mode: mode, geometry: Self.geometry) * needPages * 2
                                        + QWKVCache.tableSlotBytes * 2 * QWKVCache.maxSeqs, gdnStates: 0)
            let cache = try QWKVCache(hal: Self.hal, arena: arena, layers: [3], mode: mode, geometry: Self.geometry)
            let h = try cache.allocate(promptTokens: ceiling)
            try cache.append(h, positions: 0..<(pos + T), layer: 3, kind: .k, src: kAll)
            try cache.append(h, positions: 0..<(pos + T), layer: 3, kind: .v, src: vAll)
            let views = try cache.layerViews(h, layer: 3)
            let oNaxBuf = Self.hal.device.makeBuffer(length: T * Self.geometry.qChannels * 2, options: .storageModeShared)!
            let oGenBuf = Self.hal.device.makeBuffer(length: T * Self.geometry.qChannels * 2, options: .storageModeShared)!
            let oNaxTV = QWAttnPagedPrefillTests.tv(oNaxBuf, T * Self.geometry.qChannels)
            let oGenTV = QWAttnPagedPrefillTests.tv(oGenBuf, T * Self.geometry.qChannels)
            guard let (cb, enc) = Self.hal.begin() else { XCTFail("hal.begin"); return }
            try paged.encodeSparse(enc, qgate: qgateTV, kv: views, out: oNaxTV, blockList: lTV, blockCount: cTV, pos: pos, T: T, listStride: listStride)
            try gen.encodeSparse(enc, qgate: qgateTV, kv: views, out: oGenTV, blockList: lTV, blockCount: cTV, pos: pos, T: T, listStride: listStride)
            Self.hal.end(cb, enc); cb.waitUntilCompleted(); XCTAssertNil(cb.error)
            let naxF = QWAttnPagedPrefillTests.readOut(oNaxBuf, T * Self.geometry.qChannels)
            let genF = QWAttnPagedPrefillTests.readOut(oGenBuf, T * Self.geometry.qChannels)
            var worst: Float = 0
            for i in 0..<naxF.count {
                XCTAssertTrue(naxF[i].isFinite, "non-finite sparse NAX output at \(i) mode=\(mode)")
                worst = max(worst, abs(naxF[i] - genF[i]))
            }
            XCTAssertLessThan(worst, 2e-3, "hd256 sparse NAX vs general parity failed for \(mode): maxDiff=\(worst)")
            cache.finish(h)
        }
    }

    func testLSAIndexerStateCaptureAndRestoreRoundTripPreservesBitExactBlockSelection() throws {
        let op1 = try Self.makeOperator(maxContext: 8192, budget: 512)
        let op2 = try Self.makeOperator(maxContext: 8192, budget: 512)
        let chunk = 512
        let qgate = QWAttnPagedPrefillTests.synthQGate(n: chunk, seed: 0x3141)
        let kIn = QWAttnPagedPrefillTests.synthKV(n: chunk, seed: 0x5926)
        let qBuf = Self.hal.device.makeBuffer(bytes: qgate, length: qgate.count * 2, options: .storageModeShared)!
        let kBuf = Self.hal.device.makeBuffer(bytes: kIn, length: kIn.count * 2, options: .storageModeShared)!
        let qTV = QWTensorValue(region: QWMRegion(buffer: qBuf, base: 0, length: qBuf.length),
                                offset: 0, shape: QWTensorShape([chunk * Self.geometry.qProjOut], .f16))
        let kTV = QWTensorValue(region: QWMRegion(buffer: kBuf, base: 0, length: kBuf.length),
                                offset: 0, shape: QWTensorShape([chunk * Self.geometry.kvChannels], .f16))

        // Populate 8 superblocks (4096 tokens) + 3 open ring tokens -> position = 4099
        for sb in 0..<8 {
            guard let (cb, enc) = Self.hal.begin() else { XCTFail("hal.begin"); return }
            op1.encodeIndexAndCluster(enc, qgate: qTV, kIn: kTV, fullAttnLayer: 0, pos0: sb * chunk, rows: chunk)
            Self.hal.end(cb, enc)
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error)
        }
        guard let (cbTail, encTail) = Self.hal.begin() else { XCTFail("hal.begin"); return }
        op1.encodeIndexAndCluster(
            encTail,
            qgate: qTV.slice(byteOffset: 0, count: 3 * Self.geometry.qProjOut),
            kIn: kTV.slice(byteOffset: 0, count: 3 * Self.geometry.kvChannels),
            fullAttnLayer: 0, pos0: 4096, rows: 3)
        Self.hal.end(cbTail, encTail)
        cbTail.waitUntilCompleted()
        XCTAssertNil(cbTail.error)

        let pos = 4099
        let captured = op1.captureIndexerState(position: pos)
        XCTAssertEqual(captured.count, op1.indexerStateBytes(position: pos))
        XCTAssertFalse(op2.restoreIndexerState(captured, position: pos - 1))
        XCTAssertTrue(op2.restoreIndexerState(captured, position: pos))

        // Feed the 4th token (pos0 = 4099, rows = 1) to complete block 1024 and select blocks on both operators
        let qStep = qTV.slice(byteOffset: 3 * Self.geometry.qProjOut * 2, count: Self.geometry.qProjOut)
        let kStep = kTV.slice(byteOffset: 3 * Self.geometry.kvChannels * 2, count: Self.geometry.kvChannels)
        for op in [op1, op2] {
            guard let (cb, enc) = Self.hal.begin() else { XCTFail("hal.begin"); return }
            op.encodeIndexAndCluster(enc, qgate: qStep, kIn: kStep, fullAttnLayer: 0, pos0: pos, rows: 1)
            op.encodeSelectBlocks(enc, fullAttnLayer: 0, pos0: pos, rows: 1)
            Self.hal.end(cb, enc)
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error)
        }

        let count1 = Int(op1.region.pointer.advanced(by: op1.blockCountScratch.offset).assumingMemoryBound(to: UInt32.self)[0])
        let count2 = Int(op2.region.pointer.advanced(by: op2.blockCountScratch.offset).assumingMemoryBound(to: UInt32.self)[0])
        XCTAssertEqual(count1, count2)
        XCTAssertGreaterThan(count1, 200)

        let list1 = op1.region.pointer.advanced(by: op1.blockListScratch.offset).assumingMemoryBound(to: Int32.self)
        let list2 = op2.region.pointer.advanced(by: op2.blockListScratch.offset).assumingMemoryBound(to: Int32.self)
        let blocks1 = (0..<count1).map { Int(list1[$0]) }
        let blocks2 = (0..<count2).map { Int(list2[$0]) }
        XCTAssertEqual(blocks1, blocks2, "Restored QWLSAOperator must produce bit-identical selected blocks")
    }

    func testLSAPinnedBaseDeltaCaptureAndRestorePreservesBitExactBlockSelection() throws {
        let op1 = try Self.makeOperator(maxContext: 8192, budget: 512)
        let op2 = try Self.makeOperator(maxContext: 8192, budget: 512)
        let chunk = 512
        let qgate = QWAttnPagedPrefillTests.synthQGate(n: chunk, seed: 0x2718)
        let kIn = QWAttnPagedPrefillTests.synthKV(n: chunk, seed: 0x2818)
        let qBuf = Self.hal.device.makeBuffer(bytes: qgate, length: qgate.count * 2, options: .storageModeShared)!
        let kBuf = Self.hal.device.makeBuffer(bytes: kIn, length: kIn.count * 2, options: .storageModeShared)!
        let qTV = QWTensorValue(region: QWMRegion(buffer: qBuf, base: 0, length: qBuf.length),
                                offset: 0, shape: QWTensorShape([chunk * Self.geometry.qProjOut], .f16))
        let kTV = QWTensorValue(region: QWMRegion(buffer: kBuf, base: 0, length: kBuf.length),
                                offset: 0, shape: QWTensorShape([chunk * Self.geometry.kvChannels], .f16))

        for sb in 0..<8 {
            guard let (cb, enc) = Self.hal.begin() else { XCTFail("hal.begin"); return }
            op1.encodeIndexAndCluster(enc, qgate: qTV, kIn: kTV, fullAttnLayer: 0, pos0: sb * chunk, rows: chunk)
            Self.hal.end(cb, enc)
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error)
        }
        let basePos = 4096
        let fullBase = op1.captureIndexerState(position: basePos, forceFull: true)
        op1.setPinnedBaseState(fullBase, position: basePos)
        op2.setPinnedBaseState(fullBase, position: basePos)
        XCTAssertTrue(op2.restoreIndexerState(fullBase, position: basePos))

        guard let (cbTail, encTail) = Self.hal.begin() else { XCTFail("hal.begin"); return }
        op1.encodeIndexAndCluster(
            encTail,
            qgate: qTV.slice(byteOffset: 0, count: 19 * Self.geometry.qProjOut),
            kIn: kTV.slice(byteOffset: 0, count: 19 * Self.geometry.kvChannels),
            fullAttnLayer: 0, pos0: basePos, rows: 19)
        Self.hal.end(cbTail, encTail)
        cbTail.waitUntilCompleted()
        XCTAssertNil(cbTail.error)

        let turn1Pos = basePos + 19
        let deltaSnapshot = op1.captureIndexerState(position: turn1Pos)
        XCTAssertLessThan(deltaSnapshot.count, fullBase.count / 10,
                          "Pinned-base delta snapshot must only store blocks/superblocks above pinnedBasePos512")
        XCTAssertTrue(op2.restoreIndexerState(deltaSnapshot, position: turn1Pos))
        XCTAssertEqual(op1.captureIndexerState(position: turn1Pos, forceFull: true),
                       op2.captureIndexerState(position: turn1Pos, forceFull: true),
                       "Delta restore on resident base must match full indexer state bit-for-bit")
    }

    func testSparseBatchedDecodeBitExactMatchesSerialSparseAcrossModes() throws {
        let layer = 3
        let kvRow = Self.geometry.kvChannels
        let qProj = Self.geometry.qProjOut
        let qChan = Self.geometry.qChannels
        for mode in [QWKVMode.f16, .q8, .q4] {
            for T in [4, 8] {
                for mod4 in 0...3 {
                    let cBase = 300 + mod4
                    let ceiling = cBase + T + 256
                    let pageBytes = QWKVCache.pageBytes(mode: mode, geometry: Self.geometry)
                    let pagesPerSeq = (ceiling + 255) / 256
                    let poolBytes = pageBytes * pagesPerSeq * 3 * 2
                    let tables = QWKVCache.tableSlotBytes * 2 * QWKVCache.maxSeqs
                    let pBytes = QWAttnDecode.decodePartialsBytesBatched(T: T, geometry: Self.geometry)
                    let arena = try QWArena(hal: Self.hal, weights: 0, activations: 0,
                                            kv: poolBytes + tables, gdnStates: pBytes)
                    let cache = try QWKVCache(hal: Self.hal, arena: arena, layers: [layer],
                                              mode: mode, geometry: Self.geometry)
                    let g = arena.gdnRegion()
                    let pOff = try g.allocate(bytes: pBytes)
                    let partials = QWMRegion(buffer: g.region.buffer, base: g.region.base + pOff, length: pBytes)

                    let seqSerial = try cache.allocate(promptTokens: ceiling)
                    let seqBatched = try cache.allocate(promptTokens: ceiling)
                    let seqPre = try cache.allocate(promptTokens: ceiling)
                    let kAll = QWAttnPagedPrefillTests.synthKV(n: cBase + T, seed: 0x3100 + UInt64(mod4))
                    let vAll = QWAttnPagedPrefillTests.synthV(n: cBase + T, seed: 0x4200 + UInt64(mod4))
                    let qAll = QWAttnPagedPrefillTests.synthQGate(n: T, seed: 0x5300 + UInt64(mod4))
                    let baseRange = 0..<cBase
                    let preRange = 0..<(cBase + T)
                    try cache.append(seqSerial, positions: baseRange, layer: layer, kind: .k, src: Array(kAll[0..<(cBase * kvRow)]))
                    try cache.append(seqSerial, positions: baseRange, layer: layer, kind: .v, src: Array(vAll[0..<(cBase * kvRow)]))
                    try cache.append(seqBatched, positions: baseRange, layer: layer, kind: .k, src: Array(kAll[0..<(cBase * kvRow)]))
                    try cache.append(seqBatched, positions: baseRange, layer: layer, kind: .v, src: Array(vAll[0..<(cBase * kvRow)]))
                    try cache.append(seqPre, positions: preRange, layer: layer, kind: .k, src: kAll)
                    try cache.append(seqPre, positions: preRange, layer: layer, kind: .v, src: vAll)

                    let kNewAll = Array(kAll[(cBase * kvRow)..<((cBase + T) * kvRow)])
                    let vNewAll = Array(vAll[(cBase * kvRow)..<((cBase + T) * kvRow)])
                    let qBuf = QWAttnPagedPrefillTests.buf(qAll)
                    let kNewBuf = QWAttnPagedPrefillTests.buf(kNewAll)
                    let vNewBuf = QWAttnPagedPrefillTests.buf(vNewAll)
                    let outSerialBuf = Self.hal.device.makeBuffer(length: T * qChan * 2, options: .storageModeShared)!
                    let outBatchBuf = Self.hal.device.makeBuffer(length: T * qChan * 2, options: .storageModeShared)!
                    let outPreBuf = Self.hal.device.makeBuffer(length: T * qChan * 2, options: .storageModeShared)!

                    let listStride = (cBase + T) / 4 + 4
                    var bListArr = [Int32](repeating: 0, count: T * listStride)
                    var bCountArr = [UInt32](repeating: 0, count: T)
                    for r in 0..<T {
                        let blocks = (cBase + r + 1) / 4
                        bCountArr[r] = UInt32(blocks)
                        for b in 0..<blocks { bListArr[r * listStride + b] = Int32(b) }
                    }
                    let listBuf = Self.hal.device.makeBuffer(bytes: bListArr, length: bListArr.count * 4, options: .storageModeShared)!
                    let countBuf = Self.hal.device.makeBuffer(bytes: bCountArr, length: bCountArr.count * 4, options: .storageModeShared)!
                    let listRegion = QWMRegion(buffer: listBuf, base: 0, length: listBuf.length)
                    let countRegion = QWMRegion(buffer: countBuf, base: 0, length: countBuf.length)

                    let dec = try QWAttnDecode(hal: Self.hal, kvMode: mode, geometry: Self.geometry)
                    let splits = QWAttnDecode.splits(forKeys: cBase + T)
                    let serialViews = try cache.layerViews(seqSerial, layer: layer)
                    let batchViews = try cache.layerViews(seqBatched, layer: layer)
                    let preViews = try cache.layerViews(seqPre, layer: layer)

                    func rowTV(_ buf: any MTLBuffer, _ row: Int, _ cols: Int) -> QWTensorValue {
                        QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: buf.length),
                                      offset: row * cols * 2, shape: QWTensorShape([cols], .f16))
                    }
                    guard let (cb, enc) = Self.hal.begin() else { XCTFail("hal.begin"); return }
                    for r in 0..<T {
                        let qRow = rowTV(qBuf, r, qProj)
                        let kRow = rowTV(kNewBuf, r, kvRow)
                        let vRow = rowTV(vNewBuf, r, kvRow)
                        let oRow = rowTV(outSerialBuf, r, qChan)
                        let bl = QWTensorValue(region: listRegion, offset: r * listStride * 4, shape: QWTensorShape([listStride], .i32))
                        let bc = QWTensorValue(region: countRegion, offset: r * 4, shape: QWTensorShape([1], .i32))
                        try dec.encodeSparse(enc, qgate: qRow, kv: serialViews, knew: kRow, vnew: vRow,
                                             partials: partials, out: oRow, blockList: bl, blockCount: bc,
                                             C: cBase + r, splits: splits, listStride: listStride)
                        enc.memoryBarrier(scope: .buffers)
                    }
                    try dec.encodeSparseBatched(
                        enc,
                        qgate: QWAttnPagedPrefillTests.tv(qBuf, T * qProj),
                        kv: batchViews,
                        knew: QWAttnPagedPrefillTests.tv(kNewBuf, T * kvRow),
                        vnew: QWAttnPagedPrefillTests.tv(vNewBuf, T * kvRow),
                        partials: partials,
                        out: QWAttnPagedPrefillTests.tv(outBatchBuf, T * qChan),
                        blockList: QWTensorValue(region: listRegion, offset: 0, shape: QWTensorShape([T * listStride], .i32)),
                        blockCount: QWTensorValue(region: countRegion, offset: 0, shape: QWTensorShape([T], .i32)),
                        cBase: cBase, T: T, splits: splits, listStride: listStride, writeKV: true)
                    enc.memoryBarrier(scope: .buffers)
                    try dec.encodeSparseBatched(
                        enc,
                        qgate: QWAttnPagedPrefillTests.tv(qBuf, T * qProj),
                        kv: preViews,
                        knew: QWAttnPagedPrefillTests.tv(kNewBuf, T * kvRow),
                        vnew: QWAttnPagedPrefillTests.tv(vNewBuf, T * kvRow),
                        partials: partials,
                        out: QWAttnPagedPrefillTests.tv(outPreBuf, T * qChan),
                        blockList: QWTensorValue(region: listRegion, offset: 0, shape: QWTensorShape([T * listStride], .i32)),
                        blockCount: QWTensorValue(region: countRegion, offset: 0, shape: QWTensorShape([T], .i32)),
                        cBase: cBase, T: T, splits: splits, listStride: listStride, writeKV: false)
                    Self.hal.end(cb, enc)
                    cb.waitUntilCompleted()
                    XCTAssertNil(cb.error)

                    let sPtr = outSerialBuf.contents().assumingMemoryBound(to: UInt16.self)
                    let bPtr = outBatchBuf.contents().assumingMemoryBound(to: UInt16.self)
                    let pPtr = outPreBuf.contents().assumingMemoryBound(to: UInt16.self)
                    for idx in 0..<(T * qChan) {
                        XCTAssertEqual(bPtr[idx], sPtr[idx], "0-ULP mismatch (writeKV=true) mode=\(mode) T=\(T) mod4=\(mod4) idx=\(idx)")
                        XCTAssertEqual(pPtr[idx], sPtr[idx], "0-ULP mismatch (writeKV=false) mode=\(mode) T=\(T) mod4=\(mod4) idx=\(idx)")
                    }
                    cache.finish(seqSerial); cache.finish(seqBatched); cache.finish(seqPre)
                }
            }
        }
    }

    func testIndexPrepQModesAndSparsePagedNAXHd128Parity() throws {
        // Part 1: Verify .none / .tiled32 Q-modes match .allRows on pooledKeys, centroids, and block selections.
        let chunk = 512
        let qTV512 = QWAttnPagedPrefillTests.tv(QWAttnPagedPrefillTests.buf(QWAttnPagedPrefillTests.synthQGate(n: chunk, seed: 0x3141)), chunk * Self.geometry.qProjOut)
        let kTV512 = QWAttnPagedPrefillTests.tv(QWAttnPagedPrefillTests.buf(QWAttnPagedPrefillTests.synthKV(n: chunk, seed: 0x5926)), chunk * Self.geometry.kvChannels)
        let opRef = try Self.makeOperator(maxContext: 8192, budget: 512)
        let opOpt = try Self.makeOperator(maxContext: 8192, budget: 512)
        for sb in 0..<8 {
            guard let (c1, e1) = Self.hal.begin(), let (c2, e2) = Self.hal.begin() else { XCTFail("hal.begin"); return }
            opRef.encodeIndexAndCluster(e1, qgate: qTV512, kIn: kTV512, fullAttnLayer: 0, pos0: sb * chunk, rows: chunk, qMode: .allRows)
            opOpt.encodeIndexAndCluster(e2, qgate: qTV512, kIn: kTV512, fullAttnLayer: 0, pos0: sb * chunk, rows: chunk, qMode: .none)
            Self.hal.end(c1, e1); Self.hal.end(c2, e2); c1.waitUntilCompleted(); c2.waitUntilCompleted()
        }
        let pkRef = opRef.region.pointer.advanced(by: opRef.layers[0].pooledKeys.offset).assumingMemoryBound(to: UInt16.self)
        let pkOpt = opOpt.region.pointer.advanced(by: opOpt.layers[0].pooledKeys.offset).assumingMemoryBound(to: UInt16.self)
        for i in 0..<(1024 * opRef.indexDim) { XCTAssertEqual(pkOpt[i], pkRef[i], "pooledKeys mismatch at \(i)") }

        for T in [50, 64] {
            guard let (c1, e1) = Self.hal.begin(), let (c2, e2) = Self.hal.begin() else { XCTFail("hal.begin"); return }
            let qSub = qTV512.slice(byteOffset: 0, count: T * Self.geometry.qProjOut)
            let kSub = kTV512.slice(byteOffset: 0, count: T * Self.geometry.kvChannels)
            opRef.encodeIndexAndCluster(e1, qgate: qSub, kIn: kSub, fullAttnLayer: 0, pos0: 4096, rows: T, qMode: .allRows)
            e1.memoryBarrier(scope: .buffers)
            opRef.encodeSelectBlocks(e1, fullAttnLayer: 0, pos0: 4096, rows: T, tiled32: true)
            opOpt.encodeIndexAndCluster(e2, qgate: qSub, kIn: kSub, fullAttnLayer: 0, pos0: 4096, rows: T, qMode: .tiled32)
            e2.memoryBarrier(scope: .buffers)
            opOpt.encodeSelectBlocks(e2, fullAttnLayer: 0, pos0: 4096, rows: T, tiled32: true)
            Self.hal.end(c1, e1); Self.hal.end(c2, e2); c1.waitUntilCompleted(); c2.waitUntilCompleted()
            let rCnt = opRef.region.pointer.advanced(by: opRef.blockCountScratch.offset).assumingMemoryBound(to: UInt32.self)
            let oCnt = opOpt.region.pointer.advanced(by: opOpt.blockCountScratch.offset).assumingMemoryBound(to: UInt32.self)
            let rLst = opRef.region.pointer.advanced(by: opRef.blockListScratch.offset).assumingMemoryBound(to: Int32.self)
            let oLst = opOpt.region.pointer.advanced(by: opOpt.blockListScratch.offset).assumingMemoryBound(to: Int32.self)
            for tile in 0..<((T + 31) / 32) {
                let refRow = min(tile * 32 + 31, T - 1)
                XCTAssertEqual(oCnt[refRow], rCnt[refRow])
                for b in 0..<Int(rCnt[refRow]) {
                    XCTAssertEqual(oLst[refRow * opRef.listStride + b], rLst[refRow * opRef.listStride + b])
                }
            }
        }

        // Part 2: headDim = 128 (_hd128) NAX sparse paged prefill parity vs scalar general kernel across f16/q8/q4.
        try XCTSkipUnless(
            Self.hal.caps.supportsSimdgroupMMA,
            "Paged MMA/NAX _hd128 test requires Apple7+ hardware"
        )
        let geo = try QWAttnPagedPrefillGeneralTests.tinyGeometry(gated: true)
        XCTAssertEqual(geo.headDim, 128)
        let hal128 = try QWHal(geometry: geo)
        let nax = try QWAttnPagedPrefill(hal: hal128, geometry: geo)
        let gen = try QWAttnPagedPrefillGeneral(hal: hal128, geometry: geo)
        let pos = 256, T = 48, sp = 64, ceiling = pos + T + 256, qChan = geo.qChannels
        let fix = QWAttnPagedPrefillGeneralTests.fixture(geometry: geo, keys: pos + T, queries: T)
        let qPad = fix.qRows + [UInt16](repeating: 0, count: (sp - T) * geo.qProjOut)
        let qBuf = hal128.device.makeBuffer(bytes: qPad, length: qPad.count * 2, options: .storageModeShared)!
        let qTV = QWTensorValue(region: QWMRegion(buffer: qBuf, base: 0, length: qBuf.length),
                                offset: 0, shape: QWTensorShape([sp * geo.qProjOut], .f16))
        let prefixBlocks: [Int32] = [0, 5, 11, 20, 40, 62, 63]
        let listStride = prefixBlocks.count + 16
        var bList = [Int32](repeating: 0, count: T * listStride)
        var bCount = [UInt32](repeating: 0, count: T)
        for r in 0..<T {
            let curBlocks = ((pos + r + 1) >> 2) - 64
            bCount[r] = UInt32(prefixBlocks.count + curBlocks)
            for (idx, blk) in prefixBlocks.enumerated() { bList[r * listStride + idx] = blk }
            for idx in 0..<12 { bList[r * listStride + prefixBlocks.count + idx] = Int32(64 + idx) }
        }
        let lBuf = hal128.device.makeBuffer(bytes: bList, length: bList.count * 4, options: .storageModeShared)!
        let cBuf = hal128.device.makeBuffer(bytes: bCount, length: bCount.count * 4, options: .storageModeShared)!
        let lTV = QWTensorValue(region: QWMRegion(buffer: lBuf, base: 0, length: lBuf.length), offset: 0, shape: QWTensorShape([T * listStride], .i32))
        let cTV = QWTensorValue(region: QWMRegion(buffer: cBuf, base: 0, length: cBuf.length), offset: 0, shape: QWTensorShape([T], .i32))

        for mode in [QWKVMode.f16, .q8, .q4] {
            let needPages = (ceiling + 255) / 256
            let arena = try QWArena(hal: hal128, weights: 0, activations: 0,
                                    kv: QWKVCache.pageBytes(mode: mode, geometry: geo) * needPages * 2
                                        + QWKVCache.tableSlotBytes * 2 * QWKVCache.maxSeqs, gdnStates: 0)
            let cache = try QWKVCache(hal: hal128, arena: arena, layers: [0], mode: mode, geometry: geo)
            let seq = try cache.allocate(promptTokens: ceiling)
            try cache.append(seq, positions: 0..<(pos + T), layer: 0, kind: .k, src: fix.kRows.flatMap { $0 })
            try cache.append(seq, positions: 0..<(pos + T), layer: 0, kind: .v, src: fix.vRows.flatMap { $0 })
            let views = try cache.layerViews(seq, layer: 0)
            let oNaxBuf = hal128.device.makeBuffer(length: T * qChan * 2, options: .storageModeShared)!
            let oGenBuf = hal128.device.makeBuffer(length: T * qChan * 2, options: .storageModeShared)!
            let oNaxTV = QWTensorValue(region: QWMRegion(buffer: oNaxBuf, base: 0, length: oNaxBuf.length), offset: 0, shape: QWTensorShape([T * qChan], .f16))
            let oGenTV = QWTensorValue(region: QWMRegion(buffer: oGenBuf, base: 0, length: oGenBuf.length), offset: 0, shape: QWTensorShape([T * qChan], .f16))
            guard let (cb, enc) = hal128.begin() else { XCTFail("hal128.begin"); return }
            try nax.encodeSparse(enc, qgate: qTV, kv: views, out: oNaxTV, blockList: lTV, blockCount: cTV, pos: pos, T: T, listStride: listStride)
            try gen.encodeSparse(enc, qgate: qTV, kv: views, out: oGenTV, blockList: lTV, blockCount: cTV, pos: pos, T: T, listStride: listStride)
            hal128.end(cb, enc); cb.waitUntilCompleted(); XCTAssertNil(cb.error)
            let nPtr = oNaxBuf.contents().assumingMemoryBound(to: UInt16.self)
            let gPtr = oGenBuf.contents().assumingMemoryBound(to: UInt16.self)
            var worst: Float = 0
            for i in 0..<(T * qChan) {
                worst = max(worst, abs(QWQuant.f16tof32Bits(nPtr[i]) - QWQuant.f16tof32Bits(gPtr[i])))
            }
            XCTAssertLessThan(worst, 2e-3, "_hd128 sparse NAX vs general failed for \(mode), maxDiff=\(worst)")
            cache.finish(seq)
        }
    }

    func testMillionTokenLayoutStridesAndLiveGPUPoolSelect() throws {
        let context1M = 1_048_576
        for clusterBlk in [QWLSAConfig.defaultClusterBlock, QWLSAConfig.emeansClusterBlock] {
            let cfg = QWLSAConfig(mode: .svd, budget: 2048, clusterBlock: clusterBlk)
            let emeans = (cfg.alignedClusterBlock == QWLSAConfig.emeansClusterBlock)
            let maxBlocks = (context1M + 3) / 4
            let maxSB = emeans
                ? ((context1M + QWLSAConfig.emeansClusterBlock - 1) / QWLSAConfig.emeansClusterBlock)
                : ((context1M + 511) / 512)
            let reusePlan = QWLSALoopReusePlan.build(physicalLayers: [0, 1], enabled: false)
            let plan = QWLSAOperator.buildLayoutPlan(
                inDim: Self.geometry.lsaInputDim, indexDim: 64, ringCap: 16_384,
                maxBlocks: maxBlocks, maxSuperblocks: maxSB, maxBuckets: maxSB * 64,
                listStride: max(512, cfg.maxTotalSelectedBlocks), maxPrefillChunk: 8192, emeans: emeans,
                physicalLayers: [0, 1], reusePlan: reusePlan
            )
            XCTAssertGreaterThan(plan.totalBytes, 0)
            for off in plan.layerOffsets {
                XCTAssertEqual(off.pooledOff & 255, 0)
                XCTAssertLessThan(off.pooledOff + maxBlocks * 64 * 2, plan.totalBytes)
                XCTAssertNotEqual(off.wqOff, off.wkOff)
            }
            let est = QWLSAOperator.estimateRegionBytes(
                geometry: Self.geometry, lsaConfig: cfg, attnLayers: [0, 1],
                maxContext: context1M, maxPrefillChunk: 8192
            )
            XCTAssertGreaterThanOrEqual(est, plan.totalBytes)
        }

        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "Metal device required")
        let inDim = Self.geometry.lsaInputDim
        let wq = QWLSASidecar.synthesizeDeterministicOrthogonalBasis(inDim: inDim, indexDim: 64, seed: 301)
        let wk = QWLSASidecar.synthesizeDeterministicOrthogonalBasis(inDim: inDim, indexDim: 64, seed: 302)
        let op = try QWLSAOperator(
            hal: Self.hal,
            geometry: Self.geometry,
            config: QWLSAConfig(mode: .svd, budget: 512, clusterBlock: 512),
            layerWeights: [QWLSAOperator.LayerWeights(physicalLayer: 0, wqIdxF16: wq, wkIdxF16: wk)],
            maxContext: context1M,
            maxPrefillChunk: 32
        )
        XCTAssertEqual(op.maxBlocks, 262_144)
        XCTAssertEqual(op.maxSuperblocks, 2048)
        XCTAssertEqual(op.maxBuckets, 131_072)
        XCTAssertNotEqual(op.layers[0].wqIdx.offset, op.layers[0].wkIdx.offset)
        XCTAssertGreaterThan(op.fullIndexerStateBytes(position: context1M), 0)
        XCTAssertGreaterThan(op.deltaIndexerStateBytes(basePos512: context1M - 512, position: context1M), 0)

        let ls = op.layers[0]
        // Seed middle-zone block 261_888 (superblock 2046) so top-k must index beyond 32-bit half offsets.
        let targetBlk = 261_888
        let targetBucket = 2046 * 64
        let unitDim0 = Float16(1.0).bitPattern
        let pooledPtr = ls.pooledKeys.region.pointer.advanced(by: ls.pooledKeys.offset).assumingMemoryBound(to: UInt16.self)
        let centPtr = ls.centroids.region.pointer.advanced(by: ls.centroids.offset).assumingMemoryBound(to: UInt16.self)
        let sizePtr = ls.bucketSizes.region.pointer.advanced(by: ls.bucketSizes.offset).assumingMemoryBound(to: UInt8.self)
        pooledPtr[targetBlk * 64] = unitDim0
        centPtr[targetBucket * 64] = unitDim0
        sizePtr[targetBucket] = 1

        let qElems = 4 * Self.geometry.qProjOut
        let kElems = 4 * Self.geometry.kvChannels
        var qArr = [UInt16](repeating: 0, count: qElems)
        var kArr = [UInt16](repeating: 0, count: kElems)
        for t in 0..<4 {
            let kBase = t * Self.geometry.kvChannels + Self.geometry.lsaHeadSliceStart
            for kv in 0..<Self.geometry.kvHeads {
                kArr[kBase + kv * Self.geometry.headDim] = Float16(0.5).bitPattern
            }
            for h in 0..<Self.geometry.qHeads {
                let qOff = t * Self.geometry.qProjOut + h * 2 * Self.geometry.headDim + Self.geometry.lsaHeadSliceStart
                qArr[qOff] = Float16(0.5).bitPattern
            }
        }
        let qBuf = Self.hal.device.makeBuffer(bytes: qArr, length: qArr.count * 2, options: .storageModeShared)!
        let kBuf = Self.hal.device.makeBuffer(bytes: kArr, length: kArr.count * 2, options: .storageModeShared)!
        let qTV = QWTensorValue(region: QWMRegion(buffer: qBuf, base: 0, length: qBuf.length),
                                offset: 0, shape: QWTensorShape([4, Self.geometry.qProjOut], .f16))
        let kTV = QWTensorValue(region: QWMRegion(buffer: kBuf, base: 0, length: kBuf.length),
                                offset: 0, shape: QWTensorShape([4, Self.geometry.kvChannels], .f16))

        let (cb, enc) = try XCTUnwrap(Self.hal.begin())
        op.encodeIndexAndCluster(enc, qgate: qTV, kIn: kTV, fullAttnLayer: 0, pos0: context1M - 4, rows: 4, qMode: .none)
        let qIdxPtr = op.qIdxScratch.region.pointer.advanced(by: op.qIdxScratch.offset).assumingMemoryBound(to: UInt16.self)
        qIdxPtr[0] = unitDim0
        op.encodeSelectBlocks(enc, fullAttnLayer: 0, pos0: context1M - 1, rows: 1)
        Self.hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)

        var tailNorm = 0.0
        for d in 0..<64 {
            let v = Double(QWQuant.f16tof32Bits(pooledPtr[262_143 * 64 + d]))
            tailNorm += v * v
        }
        XCTAssertEqual(tailNorm, 1.0, accuracy: 2e-2)
        let cnt = op.blockCountScratch.region.pointer.advanced(by: op.blockCountScratch.offset)
            .assumingMemoryBound(to: Int32.self)[0]
        XCTAssertEqual(cnt, 257) // 1 sink + 128 budgetBlocks + 128 windowBlocks
        let listPtr = op.blockListScratch.region.pointer.advanced(by: op.blockListScratch.offset)
            .assumingMemoryBound(to: Int32.self)
        let picked = (0..<Int(cnt)).map { Int(listPtr[$0]) }
        XCTAssertTrue(picked.contains(targetBlk), "1M selector dropped high block \(targetBlk)")
        XCTAssertTrue(picked.contains(262_143), "1M selector dropped final tail block 262143")
    }
}
