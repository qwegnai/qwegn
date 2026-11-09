import XCTest
import QWKV
@testable import QWEngine

final class QWMTPConfigurationTests: XCTestCase {
    func testVerifyHeadQ4RequiresDraftHeadPlanes() throws {
        XCTAssertNoThrow(try QWEngineDriver.validateMTPVerifyHeadQ4(enabled: false, draftHead: false))
        XCTAssertNoThrow(try QWEngineDriver.validateMTPVerifyHeadQ4(enabled: false, draftHead: true))
        XCTAssertNoThrow(try QWEngineDriver.validateMTPVerifyHeadQ4(enabled: true, draftHead: true))
        XCTAssertThrowsError(try QWEngineDriver.validateMTPVerifyHeadQ4(enabled: true, draftHead: false)) { error in
            XCTAssertTrue(String(describing: error).contains("QW_MTP_VERIFY_HEAD_Q4"))
        }
    }

    func testStrictFlagsAndDisabledPathWithoutEnvironmentMutation() throws {
        for name in ["QW_MTP_DEVICE_EMBEDDING", "QW_MTP_LAYER_MAJOR", "QW_MTP_GPU_TARGETS",
                     "QW_MTP_PROFILE", "QW_MTP_BATCHED_MLP", "QW_MTP_BATCHED_TRUNK",
                     "QW_MTP_SHARED_PREFIX_ATTN",
                     "QW_MTP_DRAFT_HEAD", "QW_MTP_VERIFY_HEAD_Q4", "QW_MTP_ADAPTIVE_DEPTH"] {
            for drafts in 0...5 {
                XCTAssertFalse(try QWEngineDriver.resolveMTPFlag(name: name, value: nil, drafts: drafts))
                XCTAssertFalse(try QWEngineDriver.resolveMTPFlag(name: name, value: "0", drafts: drafts))
                XCTAssertEqual(try QWEngineDriver.resolveMTPFlag(name: name, value: "1", drafts: drafts), drafts > 0)
                for bad in ["", "true", "01", "-1", "2", " 1", "1 "] {
                    XCTAssertThrowsError(try QWEngineDriver.resolveMTPFlag(name: name, value: bad, drafts: drafts)) { error in
                        XCTAssertTrue(String(describing: error).contains(name))
                    }
                }
            }
            for drafts in [-1, 6, Int.max] {
                XCTAssertThrowsError(try QWEngineDriver.resolveMTPFlag(name: name, value: "1", drafts: drafts))
            }
        }
    }

    func testResolveMTPFlagDefaultValue() throws {
        for name in ["QW_MTP_DEVICE_EMBEDDING", "QW_MTP_LAYER_MAJOR", "QW_MTP_BATCHED_MLP",
                     "QW_MTP_BATCHED_TRUNK", "QW_MTP_SHARED_PREFIX_ATTN", "QW_MTP_DRAFT_HEAD"] {
            for drafts in 0...5 {
                let expectedUnset = (drafts > 0)
                XCTAssertEqual(try QWEngineDriver.resolveMTPFlag(name: name, value: nil, drafts: drafts, defaultValue: true),
                               expectedUnset)
                XCTAssertFalse(try QWEngineDriver.resolveMTPFlag(name: name, value: "0", drafts: drafts, defaultValue: true))
                XCTAssertEqual(try QWEngineDriver.resolveMTPFlag(name: name, value: "1", drafts: drafts, defaultValue: true),
                               drafts > 0)
            }
        }
    }

    func testYaRNFactorResolution() throws {
        // Unset = native window (frozen tables). ≤ 1 = explicit no-op.
        XCTAssertNil(try QWEngineDriver.resolveYaRNFactor(nil))
        XCTAssertEqual(try QWEngineDriver.resolveYaRNFactor("1"), nil)
        XCTAssertEqual(try QWEngineDriver.resolveYaRNFactor("0.5"), nil)
        XCTAssertEqual(try QWEngineDriver.resolveYaRNFactor("4"), 4)
        XCTAssertEqual(try QWEngineDriver.resolveYaRNFactor("4.0"), 4)
        // Fail-loud: every invalid form throws before load (no silent fallback).
        for bad in ["", "true", " ", " 1", "1 ", "-4", "0", "inf", "nan", "65", "1e300"] {
            XCTAssertThrowsError(try QWEngineDriver.resolveYaRNFactor(bad)) { error in
                XCTAssertTrue(String(describing: error).contains("QW_ROPE_YARN_FACTOR"),
                              "error must name the env var: \(error)")
            }
        }
    }
    func testKVModeGuardRefusesUnshippedKernels() throws {
        // 4c shipped: f16 (original), q8 and q4 all have a decode-kernel
        // data path (runtime q8mode constant in qwattn_decode/_batched/
        // _shared, quantise-on-write + dequant-on-read, byte-identity +
        // golden-band gates in QWAttnDecodeQuantKVTests). The guard set
        // must stay EXACTLY this triple — a new QWKVMode without a kernel
        // lane must be refused at load (no silent wrong-bytes decode).
        XCTAssertNoThrow(try QWEngineDriver.validateKVMode(mode: .f16))
        XCTAssertNoThrow(try QWEngineDriver.validateKVMode(mode: .q8))
        XCTAssertNoThrow(try QWEngineDriver.validateKVMode(mode: .q4))
        XCTAssertEqual(QWEngineDriver.kernelSupportedKVModes, [.f16, .q8, .q4])
    }
}
