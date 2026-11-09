import XCTest
@testable import QWEngine

final class QWPreparedGreedyRoundTests: XCTestCase {
    private func boundary(position: Int = 256, generated: Int = 7,
                          record: QWRoundRecord? = nil, hiddenValid: Bool = true,
                          trunkRewind: UInt64 = 3, pairRewind: UInt64 = 5) -> QWGreedyRoundBoundary {
        QWGreedyRoundBoundary(
            position: position, generated: generated,
            record: record ?? QWRoundRecord(pendingToken: 10, consumedCount: 256, outputCount: 7,
                                           mtpPhysicalRows: 255, absolutePosition: 256, samplerDrawIndex: 9),
            hiddenValid: hiddenValid, trunkRewind: trunkRewind, pairRewind: pairRewind)
    }

    func testOnlyPreparedBudgetAndStopPrefixesAreAuthorized() throws {
        let base = boundary()
        for drafts in [[11], [11, 12]] {
            for accepted in 0...drafts.count {
                var targets = drafts + [13]
                if accepted < drafts.count { targets[accepted] = 99 }
                for budget in 0...3 {
                    let stopSets: [Set<Int>] = [[], [11], [12], [13], [99]]
                    for stops in stopSets {
                        let decision = try QWGreedyRoundDecision.make(geometry: QWTestGeometry.g27, 
                            drafts: drafts, targetTokens: targets, maxNew: budget, stopIDs: stops)
                        let lease = QWGreedyRoundLease(boundary: base, decision: decision)
                        for retained in 0...decision.tokens.count {
                            let round = try lease.prepared.truncating(to: retained)
                            XCTAssertNoThrow(try lease.validate(round, current: base))
                            XCTAssertEqual(round.decision.tokens, Array(decision.tokens.prefix(retained)))
                            XCTAssertEqual(round.decision.stopped, decision.stopped)
                            XCTAssertThrowsError(try round.truncating(to: retained + 1),
                                                 "a shortened authorization cannot be expanded")
                        }
                        XCTAssertThrowsError(try lease.prepared.truncating(to: decision.tokens.count + 1))
                        XCTAssertThrowsError(try lease.prepared.truncating(to: -1))
                    }
                }
            }
        }
    }

    func testIdenticalOutputCannotReplayAnotherPreparation() throws {
        let base = boundary()
        let decision = try QWGreedyRoundDecision.make(geometry: QWTestGeometry.g27, drafts: [11, 12], targetTokens: [11, 12, 13], maxNew: 3)
        let first = QWGreedyRoundLease(boundary: base, decision: decision)
        let second = QWGreedyRoundLease(boundary: base, decision: decision)
        XCTAssertEqual(first.prepared.decision, second.prepared.decision)
        XCTAssertThrowsError(try second.validate(first.prepared, current: base))
        XCTAssertThrowsError(try second.validate(first.prepared.truncating(to: 0), current: base))
        XCTAssertNoThrow(try second.validate(second.prepared, current: base))
    }

    func testMetadataAndRewindChangesRefuseResolution() throws {
        let base = boundary()
        let decision = try QWGreedyRoundDecision.make(geometry: QWTestGeometry.g27, drafts: [11], targetTokens: [11, 12], maxNew: 2)
        let lease = QWGreedyRoundLease(boundary: base, decision: decision)
        var altered: [QWGreedyRoundBoundary] = [
            boundary(position: 257), boundary(generated: 8), boundary(hiddenValid: false),
            boundary(trunkRewind: 4), boundary(pairRewind: 6)
        ]
        var records = [QWRoundRecord]()
        var record = base.record
        record.pendingToken = 99; records.append(record)
        record = base.record; record.consumedCount += 1; records.append(record)
        record = base.record; record.outputCount += 1; records.append(record)
        record = base.record; record.mtpPhysicalRows += 1; records.append(record)
        record = base.record; record.absolutePosition += 1; records.append(record)
        record = base.record; record.samplerDrawIndex += 1; records.append(record)
        record = base.record; record.trunkHiddenSeed = [1]; records.append(record)
        altered.append(contentsOf: records.map { boundary(record: $0) })
        for current in altered {
            XCTAssertThrowsError(try lease.validateBoundary(current), "discard must reject stale baseline")
            XCTAssertThrowsError(try lease.validate(lease.prepared, current: current))
            XCTAssertThrowsError(try lease.validate(lease.prepared.truncating(to: 0), current: current))
        }
        XCTAssertNoThrow(try lease.validate(lease.prepared, current: base))
    }
}
