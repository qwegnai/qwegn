import XCTest
import Metal
@testable import QWAlloc
@testable import QWCore
@testable import QWHAL
@testable import QWGEMM
@testable import QWEngine
@testable import QWQuant

/// A path that reads the INCUMBENT's container must refuse the family's, by
/// name: the two layouts differ, and the incumbent's kernel would read the
/// family's blocks as if they were its own.
final class QWContainerRefusalTests: XCTestCase {

    private static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    private let n = 4
    private let k = 256

    /// A view over the given container's bytes.
    private func makeView(bits: Int, group: Int, isRotated: Bool) throws -> QWWeightView {
        let arena = try QWArena(hal: Self.hal, weights: 1 << 20, activations: 1 << 20,
                                kv: 0, gdnStates: 0)
        let bytes = QWAQBlock.rowBytes(bits: bits, group: group, k: k) * n
        let t = try arena.weightsRegion().allocate(QWTensorShape([bytes / 2], .f16))
        return try XCTUnwrap(QWWeightView(blocks: t, n: n, k: k, bits: bits,
                                          groupSize: group, isRotated: isRotated))
    }

    /// The dual-store decode GEMV has no family twin — the family's decode
    /// GEMV writes f16 only — so it must refuse the family's blocks.
    func testTheDualStorePathRefusesEveryFamilyContainer() throws {
        for (bits, group) in QWAQBlock.supported {
            let view = try makeView(bits: bits, group: group, isRotated: false)
            XCTAssertThrowsError(
                try QWEngineDriver.requireIncumbentContainer(view, "the dual-store decode GEMV"),
                "\(bits)-bit group \(group) is the family's, not the incumbent's"
            ) { error in
                let message = "\(error)"
                XCTAssertTrue(message.contains("incumbent"), message)
                XCTAssertTrue(message.contains("family"), message)
            }
        }
    }

    /// The family's `(4, 32)` pair is the trap: the SAME pair and the SAME
    /// size as the incumbent's, and a different layout. The basis is what
    /// tells them apart, so the refusal must not be decided by the pair.
    func testTheFourBitG32TrapIsRefusedByTheBasisNotThePair() throws {
        let family = try makeView(bits: 4, group: 32, isRotated: false)
        let incumbent = try makeView(bits: 4, group: 32, isRotated: true)
        XCTAssertEqual(family.rowBytes, incumbent.rowBytes)
        XCTAssertFalse(family.isIncumbentContainer)
        XCTAssertTrue(incumbent.isIncumbentContainer)
        XCTAssertThrowsError(
            try QWEngineDriver.requireIncumbentContainer(family, "the dual-store decode GEMV")
        )
        XCTAssertNoThrow(
            try QWEngineDriver.requireIncumbentContainer(incumbent, "the dual-store decode GEMV")
        )
    }
}
