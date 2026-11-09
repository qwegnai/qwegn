import XCTest
import Metal
@testable import QWGEMM
import QWCore
import QWHAL
import QWAlloc
import QWQuant

/// `QWWeightView` names the FORMAT as well as the shape, so a bank can be
/// sized and dispatched at its native bytes instead of the incumbent's
/// 5.00 bpw.
final class QWWeightViewFormatTests: XCTestCase {
    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    /// The size law, in the plan's own terms: the bytes one group takes and
    /// the bits per weight that follows. The `(4, 32)` row is the incumbent.
    func testTheSizeLawHoldsEveryPair() {
        let expected: [(bits: Int, group: Int, bytes: Int, bpw: Double)] = [
            (2, 32, 12, 3.00), (2, 64, 20, 2.50), (2, 128, 36, 2.25), (2, 256, 68, 2.125),
            (3, 32, 16, 4.00), (3, 64, 28, 3.50), (3, 128, 52, 3.25), (3, 256, 100, 3.125),
            (4, 32, 20, 5.00), (4, 64, 36, 4.50), (4, 128, 68, 4.25), (4, 256, 132, 4.125),
            (6, 32, 28, 7.00), (6, 64, 52, 6.50), (6, 128, 100, 6.25), (6, 256, 196, 6.125),
            (8, 32, 36, 9.00), (8, 64, 68, 8.50), (8, 128, 132, 8.25), (8, 256, 260, 8.125),
        ]
        XCTAssertEqual(expected.count, QWAQBlock.supported.count)
        for e in expected {
            XCTAssertEqual(QWAQBlock.bytesIfSupported(bits: e.bits, group: e.group), e.bytes)
            XCTAssertEqual(QWAQBlock.bytes(bits: e.bits, group: e.group), e.bytes,
                           "\(e.bits)-bit group \(e.group)")
            XCTAssertEqual(QWAQBlock.bitsPerWeight(bits: e.bits, group: e.group), e.bpw,
                           accuracy: 1e-12, "\(e.bits)-bit group \(e.group)")
            XCTAssertEqual(QWAQBlock.rowBytes(bits: e.bits, group: e.group, k: 4096),
                           (4096 / e.group) * e.bytes)
        }
    }

    /// A pair outside the family is a refusal, never a trap: the caller
    /// reads it from a file.
    func testAnOutsidePairIsRefusedNotCrashed() {
        XCTAssertNil(QWAQBlock.bytesIfSupported(bits: 0, group: 64))
        XCTAssertNil(QWAQBlock.bytesIfSupported(bits: 9, group: 64))
        XCTAssertNil(QWAQBlock.bytesIfSupported(bits: 4, group: 48))
        XCTAssertNil(QWAQBlock.bytesIfSupported(bits: 4, group: 0))
        XCTAssertNotNil(QWAQBlock.bytesIfSupported(bits: 4, group: 64))
    }

    /// The default view is the incumbent: 4-bit, group 32, rotated, and
    /// sized at 20 bytes per group.
    func testTheDefaultViewIsTheIncumbent() throws {
        let (view, _) = try makeView(n: 4, k: 128, bits: 4, groupSize: 32, isRotated: true)
        XCTAssertEqual(view.bits, 4)
        XCTAssertEqual(view.groupSize, 32)
        XCTAssertTrue(view.isRotated)
        XCTAssertEqual(view.rowBytes, 4 * 20)
        XCTAssertEqual(view.formatName, "HQ4_G32")
    }

    /// A 3-bit group-64 bank is sized at 28 bytes per group — 3.50 bpw,
    /// not the incumbent's 5.00.
    func testAnAffineBankIsSizedAtItsNativeBytes() throws {
        let (view, _) = try makeView(n: 4, k: 128, bits: 3, groupSize: 64, isRotated: false)
        XCTAssertEqual(view.bits, 3)
        XCTAssertEqual(view.groupSize, 64)
        XCTAssertFalse(view.isRotated)
        XCTAssertEqual(view.rowBytes, 2 * 28)
        XCTAssertEqual(view.requiredBytes, 8 * 28)
        XCTAssertEqual(view.formatName, "AQ3_G64")
    }

    /// A view that does not hold its own rows is refused.
    func testAViewThatDoesNotFitIsRefused() throws {
        let hal = Self.hal
        let arena = try QWArena(hal: hal, weights: 4096, activations: 4096, kv: 0, gdnStates: 0)
        let t = try arena.weightsRegion().allocate(QWTensorShape([512], .f16))  // 1024 B
        // 4 rows x 2 groups x 28 B = 224 B fits.
        XCTAssertNotNil(QWWeightView(blocks: t, n: 4, k: 128, bits: 3, groupSize: 64, isRotated: false))
        // 16 rows x 4 groups x 28 B = 1792 B does not.
        XCTAssertNil(QWWeightView(blocks: t, n: 16, k: 256, bits: 3, groupSize: 64, isRotated: false))
        // k must be a multiple of the group.
        XCTAssertNil(QWWeightView(blocks: t, n: 4, k: 100, bits: 3, groupSize: 64, isRotated: false))
    }

    /// Rotated low-bit views use the bit-stream container, not the incumbent.
    func testRotatedViewsPreserveTheContainerBasis() throws {
        let (view, _) = try makeView(n: 4, k: 128, bits: 4, groupSize: 32, isRotated: true)
        XCTAssertEqual(view.formatName, "HQ4_G32")
        for bits in [2, 3] {
            let lowBit = try XCTUnwrap(QWWeightView(blocks: view.blocks, n: 4, k: 128, bits: bits,
                                                    groupSize: 64, isRotated: true))
            XCTAssertTrue(lowBit.isRotated)
            XCTAssertFalse(lowBit.isIncumbentContainer)
            XCTAssertEqual(lowBit.formatName, "HQ\(bits)_G64")
        }
    }

    /// The `(4, 32)` pair has TWO containers: the same size, and not the
    /// same bytes. The rotated one is the incumbent's; the unrotated one is
    /// the family's, and the dispatcher reads the basis to tell them apart.
    func testTheFourBitG32PairHasTwoContainers() throws {
        let (rotated, _) = try makeView(n: 2, k: 64, bits: 4, groupSize: 32, isRotated: true)
        let (plain, _) = try makeView(n: 2, k: 64, bits: 4, groupSize: 32, isRotated: false)
        XCTAssertEqual(rotated.rowBytes, plain.rowBytes)
        XCTAssertEqual(rotated.formatName, "HQ4_G32")
        XCTAssertEqual(plain.formatName, "AQ4_G32")
        XCTAssertNotEqual(rotated.isRotated, plain.isRotated)
        XCTAssertNotEqual(rotated, plain)
        XCTAssertEqual(rotated, rotated)
    }

    /// Two views over the same bytes are the same tensor.
    func testEqualityFollowsTheBytes() throws {
        let arena = try QWArena(hal: Self.hal, weights: 1 << 20, activations: 1 << 20,
                                kv: 0, gdnStates: 0)
        // One buffer that holds both a 3-bit G64 bank and a 4-bit G32 bank.
        let t = try arena.weightsRegion().allocate(QWTensorShape([1024], .f16))  // 2048 B
        let a = try XCTUnwrap(QWWeightView(blocks: t, n: 4, k: 128, bits: 3,
                                           groupSize: 64, isRotated: false))
        let same = QWWeightView(blocks: t, n: 4, k: 128, bits: 3, groupSize: 64,
                                 isRotated: false)
        XCTAssertEqual(a, same)
        // The same bytes read as the incumbent's format are a DIFFERENT
        // tensor: the format is part of the identity.
        let other = QWWeightView(blocks: t, n: 4, k: 128, bits: 4, groupSize: 32,
                                  isRotated: true)
        XCTAssertNotNil(other)
        XCTAssertNotEqual(a, other!)
    }

    /// A view of a `(bits, group)` pair the family does not support is
    /// refused, so a bad file cannot reach a kernel.
    func testAnUnsupportedPairIsRefused() throws {
        let (blocks, _) = try makeView(n: 2, k: 64, bits: 4, groupSize: 32, isRotated: true)
        XCTAssertNil(QWWeightView(blocks: blocks.blocks, n: 2, k: 64, bits: 5, groupSize: 64, isRotated: true))
        XCTAssertNil(QWWeightView(blocks: blocks.blocks, n: 2, k: 64, bits: 3, groupSize: 48, isRotated: true))
    }

    /// Rotated HQ{2,3,4,6,8}_G{32,64,128,256} continuous bit-stream blocks
    /// (`isRotated: true, isIncumbentContainer: false`) are supported for all
    /// 20 HQ pairs and report `HQ<bits>_G<group>` format names.
    func testRotatedHQBitStreamViewsSupportAllPairs() throws {
        let arena = try QWArena(hal: Self.hal, weights: 1 << 20, activations: 1 << 20, kv: 0, gdnStates: 0)
        let t = try arena.weightsRegion().allocate(QWTensorShape([4096], .f16))
        XCTAssertEqual(QWAQBlock.hqSupported.count, 20)
        let expected = Set([2, 3, 4, 6, 8].flatMap { bits in
            [32, 64, 128, 256].map { "\(bits):\($0)" }
        })
        XCTAssertEqual(Set(QWAQBlock.hqSupported.map { "\($0.bits):\($0.group)" }), expected)
        for pair in QWAQBlock.hqSupported {
            let view = try XCTUnwrap(QWWeightView(blocks: t, n: 2, k: 256,
                                                  bits: pair.bits, groupSize: pair.group,
                                                  isRotated: true, isIncumbentContainer: false))
            XCTAssertTrue(view.isRotated)
            XCTAssertFalse(view.isIncumbentContainer)
            XCTAssertEqual(view.formatName, "HQ\(pair.bits)_G\(pair.group)")
            XCTAssertEqual(view.rowBytes, (256 / pair.group) * QWAQBlock.bytes(bits: pair.bits, group: pair.group))
            XCTAssertNotNil(QWGemmAQ.f32BMKernelName(bits: pair.bits, group: pair.group))
            XCTAssertNotNil(QWGemmAQ.f32V4KernelName(bits: pair.bits, group: pair.group))
            XCTAssertNotNil(QWGemmAQ.moeGateUpF32R1KernelName(bits: pair.bits, group: pair.group))
            XCTAssertNotNil(QWGemmAQ.moeDownSharedCombineF32KernelName(bits: pair.bits, group: pair.group))
        }
    }

    // MARK: - Helpers

    private func makeView(n: Int, k: Int, bits: Int, groupSize: Int, isRotated: Bool)
        throws -> (QWWeightView, QWArena) {
        let arena = try QWArena(hal: Self.hal, weights: 1 << 20, activations: 1 << 20,
                                kv: 0, gdnStates: 0)
        let bytes = QWAQBlock.rowBytes(bits: bits, group: groupSize, k: k) * n
        let t = try arena.weightsRegion().allocate(QWTensorShape([bytes / 2], .f16))
        let view = QWWeightView(blocks: t, n: n, k: k, bits: bits,
                                groupSize: groupSize, isRotated: isRotated)
        return (try XCTUnwrap(view), arena)
    }
}
