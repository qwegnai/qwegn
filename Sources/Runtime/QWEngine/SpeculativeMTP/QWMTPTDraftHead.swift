import Darwin
import Foundation
import Metal
import QWAlloc
import QWCore
import QWGEMM
import QWHAL
import QWHead
import QWQuant

/// L1 (P4-AD-60): the MTP draft LM head in q4g, file-backed via mmap.
///
/// The draft head is a DRAFT-ONLY artifact: the target verification always
final class QWMTPTDraftHead {
    let geometry: QWModelGeometry
    static let headName = "lm_head.weight"

    let n: Int            // vocab rows (248320)
    let k: Int            // hidden columns (5120)
    let view: QWWeightView
    /// The mapped planes' own buffers — anti-aliasing law: the GEMV output
    /// must never live inside one of them (C23 clobber class).
    let mappedBuffers: [any MTLBuffer]
    private let f2plan: QWGemmPlan

    init(geometry: QWModelGeometry, hal: QWHal, sidecar: QWQuant.Pack) throws {
        self.geometry = geometry
        let entry = try sidecar.entry(Self.headName)
        guard let n = entry.nRows, let k = entry.kCols, entry.quantized else {
            throw QWEngineDriverError.state("draft head '\(Self.headName)' is not a quantized 2-D plane")
        }
        guard n == geometry.vocab, k == geometry.hidden else {
            throw QWEngineDriverError.state("draft head requires [\(geometry.vocab), \(geometry.hidden)] (got [\(n), \(k)])")
        }
        guard let hq4Off = entry.hq4Off else {
            throw QWEngineDriverError.state("draft head missing hq4_off plane offset")
        }
        let path = sidecar.url.path

        func mapped(_ offset: Int, _ bytes: Int, _ elems: Int) throws -> QWTensorValue {
            let descriptor = open(path, O_RDONLY)
            guard descriptor >= 0 else {
                throw QWEngineDriverError.state("draft head file open: \(String(cString: strerror(errno)))")
            }
            defer { close(descriptor) }
            var info = stat()
            guard fstat(descriptor, &info) == 0 else {
                throw QWEngineDriverError.state("draft head file stat: \(String(cString: strerror(errno)))")
            }
            guard info.st_size >= 0, let fileBytes = Int(exactly: info.st_size),
                  offset >= 0, bytes > 0, offset % 2 == 0,
                  bytes <= fileBytes - offset else {
                throw QWEngineDriverError.state("draft head plane \(offset)+\(bytes) exceeds file")
            }
            let page = Int(getpagesize()), alignedOffset = offset / page * page
            let byteOffset = offset - alignedOffset
            guard bytes <= Int.max - byteOffset - page else {
                throw QWEngineDriverError.state("draft head plane length overflow")
            }
            let length = (bytes + byteOffset + page - 1) / page * page
            guard length <= hal.device.maxBufferLength else {
                throw QWEngineDriverError.state("draft head plane exceeds Metal buffer limit")
            }
            guard let address = mmap(nil, length, PROT_READ, MAP_SHARED,
                                     descriptor, off_t(alignedOffset)),
                  address != MAP_FAILED else {
                throw QWEngineDriverError.state("draft head mmap: \(String(cString: strerror(errno)))")
            }
            guard let mapped = hal.device.makeBuffer(bytesNoCopy: address, length: length,
                options: .storageModeShared,
                deallocator: { pointer, size in munmap(pointer, size) }) else {
                munmap(address, length)
                throw QWEngineDriverError.state("Metal cannot wrap draft head mapping")
            }
            hal.makeResident([mapped])
            return QWTensorValue(region: QWMRegion(buffer: mapped, base: 0, length: length),
                                 offset: byteOffset, shape: QWTensorShape([elems], .f16))
        }

        self.n = n
        self.k = k
        let bits = entry.hqBits
        let groupSize = entry.hqGroup
        let maybeBytes = entry.isSHVQ
            ? QWSHVQBlock.bytesIfSupported(bits: bits, group: groupSize)
            : QWAQBlock.bytesIfBlockSupported(bits: bits, group: groupSize)
        guard k % groupSize == 0,
              let groupBytes = maybeBytes else {
            throw QWEngineDriverError.state("draft head unsupported quant bits=\(bits) group=\(groupSize)")
        }
        let totalGroups = n * (k / groupSize)
        let totalBytes = totalGroups * groupBytes
        let blocks = try mapped(hq4Off, totalBytes, totalBytes / 2)
        self.mappedBuffers = [blocks.region.buffer]
        let rot = hal.weightBasis.rotatesActivations
        guard let v = QWWeightView(blocks: blocks, n: n, k: k,
                                   bits: bits, groupSize: groupSize,
                                   isRotated: rot,
                                   isIncumbentContainer: rot && entry.isIncumbentHQ4,
                                   isSHVQ: rot && entry.isSHVQ) else {
            throw QWEngineDriverError.state("draft head QWWeightView layout invalid")
        }
        self.view = v
        self.f2plan = QWGemm.plan(QWGemmShape(m: 1, n: n, k: k), dtype: .q4g, epi: .none)
    }

    /// Encode the draft head GEMV (M=1) on the caller's encoder:
    /// x [K] f16 → y [N] f32.
    func encode(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                x: QWTensorValue, gs: QWTensorValue? = nil, y: QWTensorValue) throws {
        guard x.shape.dtype == .f16, x.shape.elementCount == k,
              y.shape.dtype == .f32, y.shape.elementCount >= n else {
            throw QWEngineDriverError.state("draft head x/y shape mismatch")
        }
        for buf in mappedBuffers {
            guard y.region.buffer !== buf else {
                throw QWEngineDriverError.state("draft head output aliases a mapped plane")
            }
        }
        if !view.isIncumbentContainer {
            try QWGemmAQ.encode(enc, hal: hal, blocks: view.blocks, x: x, y: y,
                                n: n, k: k, bits: view.bits, group: view.groupSize,
                                isSHVQ: view.isSHVQ)
            return
        }
        try QWGemm.encode(enc, hal: hal, plan: f2plan, x: x, xGroupSum: gs, wQ4: view, y: y)
    }

    func encodeBatched(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                       x: QWTensorValue, gs: QWTensorValue? = nil, y: QWTensorValue,
                       m: Int) throws {
        _ = gs
        guard (2...16).contains(m) else {
            throw QWEngineDriverError.state("batched draft head m must be 2…16 (got \(m))")
        }
        guard x.shape.dtype == .f16, x.shape.elementCount == m * k,
              y.shape.dtype == .f32, y.shape.elementCount >= m * n else {
            throw QWEngineDriverError.state("batched draft head x/y shape mismatch")
        }
        for buf in mappedBuffers {
            guard y.region.buffer !== buf else {
                throw QWEngineDriverError.state("draft head output aliases a mapped plane")
            }
        }
        if !view.isIncumbentContainer && (view.isSHVQ || !(view.bits == 4 && view.groupSize == 32)) {
            for r in 0..<m {
                let xRow = QWTensorValue(region: x.region, offset: x.offset + r * k * 2,
                                         shape: QWTensorShape([k], .f16))
                let yRow = QWTensorValue(region: y.region, offset: y.offset + r * n * 4,
                                         shape: QWTensorShape([n], .f32))
                try QWGemmAQ.encode(enc, hal: hal, blocks: view.blocks, x: xRow, y: yRow,
                                    n: n, k: k, bits: view.bits, group: view.groupSize,
                                    isSHVQ: view.isSHVQ)
            }
            return
        }
        try QWHead.encodeBatchedQ4GKernel(enc, hal: hal, view: view, x: x, y: y, m: m)
    }
}
