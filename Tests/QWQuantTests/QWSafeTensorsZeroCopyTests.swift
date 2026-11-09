import XCTest
import Foundation
@testable import QWQuant

/// The container's whole reason to exist is that it does NOT copy a tensor.
/// This measures that: reading one small tensor out of a large mapped file
/// must not raise the process's peak resident memory.
final class QWSafeTensorsZeroCopyTests: XCTestCase {
    private func peakResidentBytes() -> Int {
        var usage = rusage()
        getrusage(RUSAGE_SELF, &usage)
        return Int(usage.ru_maxrss)          // bytes on Darwin
    }

    func testReadingATensorDoesNotCopyTheFile() throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("zc-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("big.safetensors")

        // One big tensor and one small one. The file is written in chunks so
        // the writer's own footprint does not pollute the measurement.
        let bigBytes = 64 * 1024 * 1024
        let smallBytes = 4096
        let header: [String: Any] = [
            "big": ["dtype": "I8", "shape": [bigBytes], "data_offsets": [0, bigBytes]],
            "small": ["dtype": "I8", "shape": [smallBytes],
                      "data_offsets": [bigBytes, bigBytes + smallBytes]],
        ]
        let headerJSON = try JSONSerialization.data(withJSONObject: header)
        FileManager.default.createFile(atPath: url.path, contents: nil)
        let handle = try FileHandle(forWritingTo: url)
        try handle.write(contentsOf: withUnsafeBytes(of: UInt64(headerJSON.count).littleEndian) { Data($0) })
        try handle.write(contentsOf: headerJSON)
        let chunk = Data(repeating: 0xAB, count: 1024 * 1024)
        var written = 0
        while written < bigBytes {
            try handle.write(contentsOf: chunk)
            written += chunk.count
        }
        try handle.write(contentsOf: Data(repeating: 0xCD, count: smallBytes))
        try handle.close()

        let before = peakResidentBytes()
        let c = try QWSafeTensorsContainer(url: url)
        let sum = try c.withBytes(of: "small") { raw -> Int in
            raw.reduce(into: 0) { $0 += Int($1) }
        }
        let afterMapped = peakResidentBytes()
        XCTAssertGreaterThan(sum, 0)

        // The same file read the ordinary way: the whole thing lands in the heap.
        let copy = try Data(contentsOf: url)
        let afterCopy = peakResidentBytes()
        XCTAssertEqual(copy.count, 8 + headerJSON.count + bigBytes + smallBytes)

        let mappedDelta = afterMapped - before
        let copyDelta = afterCopy - afterMapped
        XCTAssertLessThan(mappedDelta, copyDelta / 4,
                          "the mapped read raised peak RSS by \(mappedDelta) B; a copy raises \(copyDelta) B")
    }
}
