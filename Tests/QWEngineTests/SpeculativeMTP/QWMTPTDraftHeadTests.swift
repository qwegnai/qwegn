import Foundation
import Metal
import XCTest
import QWAlloc
import QWCore
import QWGEMM
import QWHAL
import QWQuant
@testable import QWEngine

/// L1 (P4-AD-60): the draft-only q4g LM head. The sidecar is a real QWP1
/// pack; the full-vocab planes (636 MB) are written as a SPARSE file in
/// these tests — the mapping, the wire guards, and the kernel's
final class QWMTPTDraftHeadTests: XCTestCase {
    /// The QWP1 wire for a q4g-only [n, k] tensor, written as a sparse file
    /// (zero-fill writes are holes; only the header and the tail bytes
    /// touch the disk). Offsets are FILE offsets, 256 B aligned — the
    private static func makeSparseQ4gSidecar(at path: String, n: Int, k: Int) throws {
        let blocks = n * (k / 32)
        let blockBytes = blocks * 20
        func align(_ b: Int) -> Int { (b + 255) & ~255 }
        let prefix = 12                       // magic + u64 headerLen
        let hq4Off = align(prefix)
        let payloadEnd = hq4Off + align(blockBytes)
        let header = Data("""
        {"version":2,"provenance":{"quant":"hq4_g32_m5"},"tensors":{"\(QWMTPTDraftHead.headName)":\
        {"shape":[\(n),\(k)],"dtype":"hq4_g32_m5","hq4_off":\(hq4Off),"hq4_len":\(blockBytes)}}}
        """.utf8)
        FileManager.default.createFile(atPath: path, contents: nil)
        let file = FileHandle(forWritingAtPath: path)!
        defer { try? file.close() }
        file.seek(toFileOffset: 0)
        file.write(Data("QWP1".utf8))
        withUnsafeBytes(of: UInt64(header.count).littleEndian) { file.write(Data($0)) }
        file.seek(toFileOffset: UInt64(payloadEnd))
        file.write(header)
        file.truncateFile(atOffset: UInt64(payloadEnd + header.count))
    }

    func testDraftHeadMapsAndRejectsBadGuards() throws {
        let hal = try QWHal(geometry: .qwen38_27b)
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-draft-head-\(UUID().uuidString)").path
        defer { try? FileManager.default.removeItem(atPath: path) }
        try Self.makeSparseQ4gSidecar(at: path, n: QWModelGeometry.qwen38_27b.vocab, k: QWModelGeometry.qwen38_27b.hidden)
        let pack = try QWQuant.Pack(url: URL(fileURLWithPath: path))
        let head = try QWMTPTDraftHead(geometry: QWTestGeometry.g27, hal: hal, sidecar: pack)
        XCTAssertEqual(head.n, QWModelGeometry.qwen38_27b.vocab)
        XCTAssertEqual(head.k, QWModelGeometry.qwen38_27b.hidden)
        XCTAssertEqual(head.mappedBuffers.count, 1)

        let arena = try QWArena(hal: hal, weights: 1 << 16, activations: 4 << 20,
                                kv: 1 << 12, gdnStates: 1 << 12)
        let x = try arena.activationsRegion().allocate(QWTensorShape([QWModelGeometry.qwen38_27b.hidden], .f16))
        let gs = try arena.activationsRegion().allocate(QWTensorShape([QWModelGeometry.qwen38_27b.hidden / 128], .f32))
        let y = try arena.activationsRegion().allocate(QWTensorShape([QWModelGeometry.qwen38_27b.vocab], .f32))
        let alias = QWTensorValue(region: QWMRegion(buffer: head.mappedBuffers[0], base: 0,
                                                    length: head.mappedBuffers[0].length),
                                  offset: 0, shape: QWTensorShape([QWModelGeometry.qwen38_27b.vocab], .f32))
        guard let cb = hal.queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else {
            return XCTFail("command buffer")
        }
        // Anti-aliasing law: the output must not be a mapped plane.
        XCTAssertThrowsError(try head.encode(enc, hal: hal, x: x, gs: gs, y: alias),
                             "output must not alias a mapped plane")
        // Shape guards: wrong dtype / wrong length.
        let badX = QWTensorValue(region: x.region, offset: x.offset,
                                 shape: QWTensorShape([QWModelGeometry.qwen38_27b.hidden], .f32))
        XCTAssertThrowsError(try head.encode(enc, hal: hal, x: badX, gs: gs, y: y))
        let badY = QWTensorValue(region: y.region, offset: y.offset,
                                 shape: QWTensorShape([QWModelGeometry.qwen38_27b.vocab / 2], .f32))
        XCTAssertThrowsError(try head.encode(enc, hal: hal, x: x, gs: gs, y: badY))
        enc.endEncoding()
        cb.commit()

        // A zero-plane GEMV on a zero x is all zeros: the mapping + the
        // shipped F2 route complete (the q4g kernel itself is the C05
        // ADR-001 winner; this test pins the mapping and the guards).
        let good = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let goodEnc = try XCTUnwrap(good.makeComputeCommandEncoder())
        try head.encode(goodEnc, hal: hal, x: x, gs: gs, y: y)
        goodEnc.endEncoding()
        good.commit()
        good.waitUntilCompleted()
        XCTAssertEqual(good.status, .completed)
        let bits = y.region.pointer.advanced(by: y.offset).assumingMemoryBound(to: Float.self)
        for i in stride(from: 0, to: QWModelGeometry.qwen38_27b.vocab, by: 4096) {
            XCTAssertEqual(bits[i], 0, "zero planes must produce zero logits at \(i)")
        }

        let nTG = (QWModelGeometry.qwen38_27b.vocab + 7) / 8
        let keysHi = try arena.activationsRegion().allocate(QWTensorShape([nTG], .i32))
        let keysLo = try arena.activationsRegion().allocate(QWTensorShape([nTG], .i32))
        let token = try arena.activationsRegion().allocate(QWTensorShape([1], .i32))
        bits[0] = 42.0
        for writeFlag in [UInt32(0), UInt32(1)] {
            let fcb = try XCTUnwrap(hal.queue.makeCommandBuffer())
            let fenc = try XCTUnwrap(fcb.makeComputeCommandEncoder())
            var nU = UInt32(QWModelGeometry.qwen38_27b.vocab), kU = UInt32(QWModelGeometry.qwen38_27b.hidden)
            var wf = writeFlag, nTG32 = UInt32(nTG)
            fenc.setComputePipelineState(try hal.pipeline("qwg_mtp_head_gemv_q4g_argmax"))
            head.view.blocks.bind(fenc, index: 0)
            x.bind(fenc, index: 1)
            y.bind(fenc, index: 3)
            fenc.setBytes(&nU, length: 4, index: 4)
            fenc.setBytes(&kU, length: 4, index: 5)
            keysHi.bind(fenc, index: 6)
            keysLo.bind(fenc, index: 7)
            fenc.setBytes(&wf, length: 4, index: 8)
            fenc.dispatchThreadgroups(MTLSize(width: nTG, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            fenc.memoryBarrier(scope: .buffers)
            fenc.setComputePipelineState(try hal.pipeline("qwg_head_argmax_fin"))
            keysHi.bind(fenc, index: 0)
            keysLo.bind(fenc, index: 1)
            token.bind(fenc, index: 2)
            fenc.setBytes(&nTG32, length: 4, index: 3)
            fenc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
            fenc.endEncoding()
            fcb.commit()
            fcb.waitUntilCompleted()
            XCTAssertEqual(fcb.status, .completed)
            let tok = token.region.pointer.advanced(by: token.offset).assumingMemoryBound(to: Int32.self).pointee
            XCTAssertEqual(tok, 0)
            XCTAssertEqual(bits[0], writeFlag == 0 ? 42.0 : 0.0)
        }
    }

    /// The flag contract: resolveMTPFlag is inert (false) when drafts == 0;
    /// the loud failure lives in the driver init — QW_MTP_DRAFT_HEAD=1 with
    /// a sidecar missing never falls back silently to the bf16 head.
    func testFlagContract() throws {
        // The flag is inert when drafts == 0 (returns false, never throws);
        // the loud failure lives in the driver init — QW_MTP_DRAFT_HEAD=1
        // with drafts > 0 and no sidecar path throws.
        XCTAssertEqual(try QWEngineDriver.resolveMTPFlag(name: "QW_MTP_DRAFT_HEAD",
                                                         value: "1", drafts: 0), false)
        XCTAssertEqual(try QWEngineDriver.resolveMTPFlag(name: "QW_MTP_DRAFT_HEAD",
                                                         value: "1", drafts: 1), true)
        XCTAssertEqual(try QWEngineDriver.resolveMTPFlag(name: "QW_MTP_DRAFT_HEAD",
                                                         value: "0", drafts: 2), false)
        XCTAssertThrowsError(try QWEngineDriver.resolveMTPFlag(name: "QW_MTP_DRAFT_HEAD",
                                                               value: "2", drafts: 1))
    }
}
