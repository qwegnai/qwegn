import Darwin
import Foundation
import Metal
import QWAlloc
import QWCore
import QWHAL

/// A file-backed private mapping, not a second converted embedding allocation.
/// Metal owns the mapping lifetime; the file descriptor closes after mapping.
final class QWMTPEmbeddingTable {
    let buffer: any MTLBuffer
    let byteOffset: Int
    let vocabulary: Int
    let hidden: Int
    let tokens: any MTLBuffer
    private let pipeline: any MTLComputePipelineState

    init(hal: QWHal, path: String, offset: Int, vocabulary: Int, hidden: Int) throws {
        guard offset >= 0, offset % 2 == 0, vocabulary > 0, hidden > 0,
              vocabulary <= Int(Int32.max), hidden <= Int(UInt32.max),
              vocabulary <= Int.max / hidden / 2 else {
            throw QWEngineDriverError.state("mapped embedding geometry invalid")
        }
        pipeline = try hal.pipeline("qwg_mtp_embed_bf16")
        self.vocabulary = vocabulary
        self.hidden = hidden
        let descriptor = open(path, O_RDONLY)
        guard descriptor >= 0 else {
            throw QWEngineDriverError.state("embedding file open: \(String(cString: strerror(errno)))")
        }
        defer { close(descriptor) }
        var info = stat()
        guard fstat(descriptor, &info) == 0 else {
            throw QWEngineDriverError.state("embedding file stat: \(String(cString: strerror(errno)))")
        }
        let bytes = vocabulary * hidden * 2
        guard info.st_size >= 0, let fileBytes = Int(exactly: info.st_size),
              offset <= fileBytes, bytes <= fileBytes - offset else {
            throw QWEngineDriverError.state("mapped embedding plane exceeds file")
        }
        let page = Int(getpagesize()), alignedOffset = offset / page * page
        byteOffset = offset - alignedOffset
        guard bytes <= Int.max - byteOffset - page else {
            throw QWEngineDriverError.state("mapped embedding length overflow")
        }
        let length = (bytes + byteOffset + page - 1) / page * page
        guard length <= hal.device.maxBufferLength else {
            throw QWEngineDriverError.state("mapped embedding exceeds Metal buffer limit")
        }
        // Read-only shared mapping avoids IOGPU Copy-On-Write page-fault storms on first GPU bind.
        guard let address = mmap(nil, length, PROT_READ, MAP_SHARED, descriptor, off_t(alignedOffset)),
              address != MAP_FAILED else {
            throw QWEngineDriverError.state("embedding mmap: \(String(cString: strerror(errno)))")
        }
        guard let mapped = hal.device.makeBuffer(bytesNoCopy: address, length: length,
            options: .storageModeShared, deallocator: { pointer, size in munmap(pointer, size) }) else {
            munmap(address, length)
            throw QWEngineDriverError.state("Metal cannot wrap embedding mapping")
        }
        buffer = mapped
        hal.makeResident([mapped])
        guard let journal = hal.device.makeBuffer(length: 256, options: .storageModeShared) else {
            throw QWEngineDriverError.state("embedding draft token journal allocation")
        }
        tokens = journal
    }

    func encode(_ enc: any MTLComputeCommandEncoder, token: QWTensorValue,
                destination: QWTensorValue) throws {
        guard token.shape.dtype == .i32, token.shape.elementCount == 1,
              destination.shape.dtype == .f16, destination.shape.elementCount == hidden,
              validExtent(token, alignment: 4), validExtent(destination, alignment: 2),
              destination.region.buffer !== buffer,
              !overlap(token, destination) else {
            throw QWEngineDriverError.state("device embedding token/destination invalid")
        }
        enc.memoryBarrier(scope: .buffers)
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(buffer, offset: byteOffset, index: 0)
        token.bind(enc, index: 1)
        destination.bind(enc, index: 2)
        var n = UInt32(vocabulary), h = UInt32(hidden)
        enc.setBytes(&n, length: 4, index: 3)
        enc.setBytes(&h, length: 4, index: 4)
        enc.dispatchThreads(MTLSize(width: hidden, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.memoryBarrier(scope: .buffers)
    }

    private func validExtent(_ value: QWTensorValue, alignment: Int) -> Bool {
        let region = value.region
        guard region.base >= 0, region.base <= region.buffer.length,
              region.length >= 0, region.length <= region.buffer.length - region.base,
              value.offset >= 0, value.offset <= region.length,
              value.bytes >= 0, value.bytes <= region.length - value.offset else { return false }
        return (region.base + value.offset) % alignment == 0
    }

    private func overlap(_ lhs: QWTensorValue, _ rhs: QWTensorValue) -> Bool {
        guard lhs.region.buffer === rhs.region.buffer else { return false }
        let start = lhs.region.base + lhs.offset, end = rhs.region.base + rhs.offset
        return start < end + rhs.bytes && end < start + lhs.bytes
    }
}
