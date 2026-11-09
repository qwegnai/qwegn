import XCTest
import Foundation
@testable import QWHAL

final class QWDispatchIOTests: XCTestCase {

    private func createTempFile(size: Int) throws -> URL {
        let tempDir = FileManager.default.temporaryDirectory
        let fileURL = tempDir.appendingPathComponent("qw_dispatch_io_\(UUID().uuidString).bin")
        var buffer = [UInt8](repeating: 0, count: size)
        for i in 0..<size {
            buffer[i] = UInt8(i & 0xFF)
        }
        let data = Data(buffer)
        try data.write(to: fileURL)
        return fileURL
    }

    func testDispatchIOFileReadWriteRoundtrip() async throws {
        let size = 64 * 1024 // 64 KB
        let fileURL = try createTempFile(size: size)
        defer { try? FileManager.default.removeItem(at: fileURL) }

        let io = try QWDispatchIO(path: fileURL.path, mode: .readWrite, directIO: false)
        defer { io.close() }

        // Read first 1024 bytes
        let data = try await io.read(offset: 0, length: 1024)
        XCTAssertEqual(data.count, 1024)
        data.withUnsafeBytes { raw in
            for i in 0..<1024 {
                XCTAssertEqual(raw[i], UInt8(i & 0xFF), "Mismatch at byte \(i)")
            }
        }

        // Write a modified 512-byte pattern at offset 2048
        let writeBuf = [UInt8](repeating: 0xAA, count: 512)
        try await io.write(offset: 2048, bytes: writeBuf)

        // Read back offset 2048 and verify 0xAA
        let readBack = try await io.readData(offset: 2048, length: 512)
        XCTAssertEqual(readBack.count, 512)
        for i in 0..<512 {
            XCTAssertEqual(readBack[i], 0xAA, "Mismatch at overwritten byte \(i)")
        }
    }

    func testDispatchIODirectIOPageAligned() async throws {
        let pageSize = QWDispatchIO.pageSize // 16 KB on Apple Silicon
        let totalSize = pageSize * 4
        let fileURL = try createTempFile(size: totalSize)
        defer { try? FileManager.default.removeItem(at: fileURL) }

        // Enable direct I/O (F_NOCACHE)
        let io = try QWDispatchIO(path: fileURL.path, mode: .readOnly, directIO: true)
        defer { io.close() }

        // Allocate page-aligned memory
        var rawPtr: UnsafeMutableRawPointer? = nil
        let status = posix_memalign(&rawPtr, pageSize, pageSize)
        XCTAssertEqual(status, 0)
        guard let dst = rawPtr else { return XCTFail("posix_memalign failed") }
        defer { free(dst) }

        // Read page 1 (offset = pageSize)
        let bytesRead = try await io.read(offset: Int64(pageSize), length: pageSize, into: dst)
        XCTAssertEqual(bytesRead, pageSize)

        let typed = dst.assumingMemoryBound(to: UInt8.self)
        for i in 0..<pageSize {
            let expected = UInt8((pageSize + i) & 0xFF)
            XCTAssertEqual(typed[i], expected, "Direct I/O byte mismatch at \(i)")
        }
    }

    func testDispatchIOStreamingChunks() async throws {
        let chunkSize = 8192
        let chunkCount = 8
        let totalSize = chunkSize * chunkCount
        let fileURL = try createTempFile(size: totalSize)
        defer { try? FileManager.default.removeItem(at: fileURL) }

        let io = try QWDispatchIO(path: fileURL.path, mode: .readOnly, directIO: false)
        defer { io.close() }

        var receivedOffsets: [Int64] = []
        var totalBytesReceived = 0

        try await io.readChunks(offset: 0, length: totalSize, chunkSize: chunkSize) { offset, chunkData in
            receivedOffsets.append(offset)
            totalBytesReceived += chunkData.count
        }

        XCTAssertEqual(receivedOffsets.count, chunkCount)
        XCTAssertEqual(totalBytesReceived, totalSize)
        for (idx, off) in receivedOffsets.enumerated() {
            XCTAssertEqual(off, Int64(idx * chunkSize))
        }
    }

    func testDispatchIOSyncFallback() throws {
        let size = 16384
        let fileURL = try createTempFile(size: size)
        defer { try? FileManager.default.removeItem(at: fileURL) }

        let io = try QWDispatchIO(path: fileURL.path, mode: .readWrite, directIO: false)
        defer { io.close() }

        var buf = [UInt8](repeating: 0, count: 512)
        let readBytes = try buf.withUnsafeMutableBytes { raw in
            try io.readSync(offset: 1024, into: raw.baseAddress!, length: 512)
        }
        XCTAssertEqual(readBytes, 512)
        for i in 0..<512 {
            XCTAssertEqual(buf[i], UInt8((1024 + i) & 0xFF))
        }

        // Test sync write
        let newBuf = [UInt8](repeating: 0x55, count: 256)
        try newBuf.withUnsafeBytes { raw in
            try io.writeSync(offset: 4096, buffer: raw.baseAddress!, length: 256)
        }

        var verifyBuf = [UInt8](repeating: 0, count: 256)
        _ = try verifyBuf.withUnsafeMutableBytes { raw in
            try io.readSync(offset: 4096, into: raw.baseAddress!, length: 256)
        }
        for i in 0..<256 {
            XCTAssertEqual(verifyBuf[i], 0x55)
        }
    }

    func testDispatchIOErrorContracts() throws {
        let nonExistentPath = "/tmp/qw_non_existent_\(UUID().uuidString).bin"
        XCTAssertThrowsError(try QWDispatchIO(path: nonExistentPath, mode: .readOnly)) { error in
            guard case QWDispatchIOError.openFailed = error else {
                return XCTFail("Expected openFailed, got \(error)")
            }
        }

        let tempFile = try createTempFile(size: 1024)
        defer { try? FileManager.default.removeItem(at: tempFile) }

        let io = try QWDispatchIO(path: tempFile.path, mode: .readOnly)
        io.close()

        // Calling read on closed channel
        let expectation = expectation(description: "Throws on closed channel")
        Task {
            do {
                _ = try await io.read(offset: 0, length: 128)
                XCTFail("Should have thrown channelClosed")
            } catch {
                guard case QWDispatchIOError.channelClosed = error else {
                    XCTFail("Expected channelClosed, got \(error)")
                    return
                }
                expectation.fulfill()
            }
        }
        wait(for: [expectation], timeout: 2.0)
    }
}
