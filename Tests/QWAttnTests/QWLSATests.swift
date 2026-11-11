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
        try XCTSkipUnless(Self.hal.caps.isMetal4TensorCapable, "Sparse paged NAX prefill requires Metal 4 tensor hardware")
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
        var maxDiff: Float = 0
        for i in 0..<denseF.count {
            maxDiff = max(maxDiff, abs(denseF[i] - sparseF[i]))
        }
        XCTAssertLessThan(maxDiff, 1e-3, "sparse paged prefill with full block list must match dense paged prefill")
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
}
