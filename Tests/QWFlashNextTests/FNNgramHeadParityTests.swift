import Foundation
import XCTest
@testable import QWFlashNext

private struct SyntheticNgramRows: FNNgramRows {
    let headDim: Int

    init(headDim: Int = 5) { self.headDim = headDim }

    func row(_ id: Int) -> [Float] {
        (0..<headDim).map { Float(id * headDim + $0) }
    }
}

final class FNNgramHeadParityTests: XCTestCase {
    func testParallelHeadsMatchSerialBatchRows() {
        let histories: [[Int64]] = [
            [3],
            [9, 8, 7, 6],
            [4, 99, 5, 6, 7],
            [10, 11, 12, 13, 14, 15, 16],
        ]
        for ngram in 3...5 {
            let embedding = FNNgramEmbedding(vocab: 128, ngram: ngram, headsPerNgram: 4,
                                             baseVocab: 64, seed: 123, eos: 99,
                                             pleLayerIndex: 0, rows: SyntheticNgramRows())
            let heads = embedding.heads.sizes.count
            let rowWidth = heads * embedding.rows.headDim
            for batchSize in [1, 2, 4] {
                var serial = [Float](repeating: 0, count: batchSize * rowWidth)
                serial.withUnsafeMutableBufferPointer { output in
                    for row in 0..<batchSize {
                        embedding.embedToken(into: output.baseAddress!, tokens: histories[row],
                                             startPos: histories[row].count - 1 - row, t: row)
                    }
                }
                for parallelHeads in [false, true] {
                    var staged = [Float](repeating: 0, count: batchSize * rowWidth)
                    staged.withUnsafeMutableBufferPointer { output in
                        let address = Int(bitPattern: output.baseAddress!)
                        let batchHistory = Array(histories.prefix(batchSize))
                        FNGPUDecodeStack.stageBatchPLE(
                            targets: [(embedding, address)], histories: batchHistory,
                            positions: batchHistory.map { $0.count - 1 }, parallelHeads: parallelHeads
                        )
                    }
                    XCTAssertEqual(staged, serial, "ngram \(ngram), batch \(batchSize), heads \(parallelHeads)")
                }
            }
        }
    }

    func testMultipleTargetsAndHeterogeneousHeadFallback() {
        let histories: [[Int64]] = [[1, 2, 3], [4, 99, 5, 6], [7, 8], [9, 10, 11]]
        for secondNgram in [3, 5] {
            let first = FNNgramEmbedding(vocab: 128, ngram: 3, headsPerNgram: 4, baseVocab: 64,
                                         seed: 1, eos: 99, pleLayerIndex: 0, rows: SyntheticNgramRows())
            let second = FNNgramEmbedding(vocab: 128, ngram: secondNgram, headsPerNgram: 4, baseVocab: 64,
                                          seed: 2, eos: 99, pleLayerIndex: 1, rows: SyntheticNgramRows())
            func serial(_ embedding: FNNgramEmbedding) -> [Float] {
                let width = embedding.heads.sizes.count * embedding.rows.headDim
                var result = [Float](repeating: 0, count: histories.count * width)
                result.withUnsafeMutableBufferPointer { output in
                    for row in histories.indices {
                        embedding.embedToken(into: output.baseAddress!, tokens: histories[row],
                                             startPos: histories[row].count - 1 - row, t: row)
                    }
                }
                return result
            }
            let expectedFirst = serial(first), expectedSecond = serial(second)
            for parallelHeads in [false, true] {
                var firstOutput = [Float](repeating: 0, count: expectedFirst.count)
                var secondOutput = [Float](repeating: 0, count: expectedSecond.count)
                firstOutput.withUnsafeMutableBufferPointer { firstBuffer in
                    secondOutput.withUnsafeMutableBufferPointer { secondBuffer in
                        FNGPUDecodeStack.stageBatchPLE(
                            targets: [(first, Int(bitPattern: firstBuffer.baseAddress!)),
                                      (second, Int(bitPattern: secondBuffer.baseAddress!))],
                            histories: histories, positions: histories.map { $0.count - 1 },
                            parallelHeads: parallelHeads
                        )
                    }
                }
                XCTAssertEqual(firstOutput, expectedFirst)
                XCTAssertEqual(secondOutput, expectedSecond)
            }
        }
    }

    func testSyntheticBatchStagingMicrobenchmark() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["QW_PERF_GATE"] == "1",
                          "synthetic CPU benchmark is opt-in")
        let embedding = FNNgramEmbedding(vocab: 128, ngram: 3, headsPerNgram: 8, baseVocab: 64,
                                         seed: 1, eos: 99, pleLayerIndex: 0,
                                         rows: SyntheticNgramRows(headDim: 160))
        let histories: [[Int64]] = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        let positions = histories.map { $0.count - 1 }
        let count = histories.count * embedding.heads.sizes.count * embedding.rows.headDim
        var buffer = [Float](repeating: 0, count: count)
        func measure(_ parallelHeads: Bool) -> Double {
            let start = DispatchTime.now().uptimeNanoseconds
            buffer.withUnsafeMutableBufferPointer { output in
                let targets = [(ple: embedding, addr: Int(bitPattern: output.baseAddress!))]
                for _ in 0..<20 {
                    FNGPUDecodeStack.stageBatchPLE(targets: targets, histories: histories,
                                                   positions: positions, parallelHeads: parallelHeads)
                }
            }
            return Double(DispatchTime.now().uptimeNanoseconds - start) / 20_000_000
        }
        _ = measure(false)
        _ = measure(true)
        var serial: [Double] = [], parallel: [Double] = []
        for sample in 0..<5 {
            if sample.isMultiple(of: 2) {
                serial.append(measure(false))
                parallel.append(measure(true))
            } else {
                parallel.append(measure(true))
                serial.append(measure(false))
            }
        }
        print("Synthetic CPU PLE ms/step: per-row \(serial), per-head \(parallel)")
    }

    func testRealBatchStagingMicrobenchmark() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["QW_REAL_PLE_BENCH"] == "1",
                          "real PLE table benchmark is opt-in")
        let url = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".qwg/models/Qwen3.8-Flash-Next-hq4g32/assets/ngram.bin")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: url.path), "native PLE table is absent")
        let rows = try FNNgramBinRows(url: url)
        let mappedRows = try FNNgramBinRows(url: url, mapped: true)
        let embedding = FNNgramEmbedding(vocab: 248_320, ngram: 3, headsPerNgram: 8,
                                         baseVocab: 20_000_000, seed: 1234, eos: 248_044,
                                         pleLayerIndex: 0, rows: rows)
        let mappedEmbedding = FNNgramEmbedding(vocab: 248_320, ngram: 3, headsPerNgram: 8,
                                               baseVocab: 20_000_000, seed: 1234, eos: 248_044,
                                               pleLayerIndex: 0, rows: mappedRows)
        let histories: [[Int64]] = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        let positions = histories.map { $0.count - 1 }
        let count = histories.count * embedding.heads.sizes.count * rows.headDim
        var buffer = [Float](repeating: 0, count: count)
        func measure(_ source: FNNgramEmbedding, _ parallelHeads: Bool) -> Double {
            let start = DispatchTime.now().uptimeNanoseconds
            buffer.withUnsafeMutableBufferPointer { output in
                let targets = [(ple: source, addr: Int(bitPattern: output.baseAddress!))]
                for _ in 0..<10 {
                    FNGPUDecodeStack.stageBatchPLE(targets: targets, histories: histories,
                                                   positions: positions, parallelHeads: parallelHeads)
                }
            }
            return Double(DispatchTime.now().uptimeNanoseconds - start) / 10_000_000
        }
        _ = measure(embedding, false)
        let expected = buffer
        _ = measure(embedding, true)
        XCTAssertEqual(buffer, expected)
        _ = measure(mappedEmbedding, false)
        XCTAssertEqual(buffer, expected)
        _ = measure(mappedEmbedding, true)
        XCTAssertEqual(buffer, expected)
        let variants = [(embedding, false), (embedding, true), (mappedEmbedding, false), (mappedEmbedding, true)]
        var samples = [[Double]](repeating: [], count: variants.count)
        for sample in 0..<5 {
            let order = sample.isMultiple(of: 2) ? Array(variants.indices) : Array(variants.indices.reversed())
            for index in order {
                samples[index].append(measure(variants[index].0, variants[index].1))
            }
        }
        print("Real native PLE warm CPU ms/step: pread-row \(samples[0]), pread-head \(samples[1]), "
              + "mmap-row \(samples[2]), mmap-head \(samples[3])")
    }
}
