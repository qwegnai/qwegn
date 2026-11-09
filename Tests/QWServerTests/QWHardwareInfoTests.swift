import XCTest
@testable import QWServer
import QWCore
import QWKV

final class QWHardwareInfoTests: XCTestCase {
    func testCurrentHardwareInfoExtraction() {
        let hw = QWMachineHardwareInfo.current()
        XCTAssertFalse(hw.modelIdentifier.isEmpty, "Model identifier should not be empty")
        XCTAssertFalse(hw.cpuBrand.isEmpty, "CPU brand should not be empty")
        XCTAssertGreaterThan(hw.physicalCores, 0, "Physical cores must be positive")
        XCTAssertGreaterThan(hw.logicalCores, 0, "Logical cores must be positive")
        XCTAssertGreaterThan(hw.physicalMemoryBytes, 0, "Physical memory must be positive")
        XCTAssertGreaterThan(hw.physicalMemoryGB, 0.0, "Physical memory GB must be positive")
        XCTAssertFalse(hw.osProductVersion.isEmpty, "OS version should not be empty")
        XCTAssertFalse(hw.kernelRelease.isEmpty, "Kernel release should not be empty")
    }

    func testHardwareAuditBannerRendering() {
        let hw = QWMachineHardwareInfo(
            modelIdentifier: "Mac17,6",
            cpuBrand: "Apple M5 Max",
            physicalCores: 18,
            logicalCores: 18,
            physicalMemoryBytes: 137_438_953_472,
            osProductVersion: "27.2",
            kernelRelease: "27.2.0"
        )
        let params = QWEngineParametersInfo(
            modelPackPath: "/models/Qwen3.8-27B-q4g/model.qwp",
            modelName: "qwen3.8-27b",
            modelDescriptor: .qwen38_27b,
            tensorCount: 1199,
            headStorage: "Q4g",
            visionStatus: "Active (vision tower loaded)",
            contextBudget: 4096,
            kvMode: .f16,
            mtpDrafts: 0,
            mtpBackend: "standard",
            gdnPrefillMode: "chunked",
            prefillScratchEviction: "auto"
        )

        let banner = QWHardwareBanner.renderBanner(
            hardware: hw,
            caps: nil,
            params: params,
            host: "127.0.0.1",
            port: 8080
        )

        XCTAssertTrue(banner.contains("HARDWARE & ENGINE AUDIT"))
        XCTAssertTrue(banner.contains("Apple M5 Max"))
        XCTAssertTrue(banner.contains("128.00 GB"))
        XCTAssertTrue(banner.contains("18 physical / 18 logical"))
        // The architecture lines are the loaded model's own numbers,
        // read off its descriptor — not literals from the banner file.
        XCTAssertTrue(banner.contains("qwen3_5 (model_type qwen3_5)"))
        XCTAssertTrue(banner.contains("hidden 5120"))
        XCTAssertTrue(banner.contains("64 layers"))
        XCTAssertTrue(banner.contains("Q 24 / KV 4"))
        XCTAssertTrue(banner.contains("48 linear_attention"))
        XCTAssertTrue(banner.contains("Gated DeltaNet, 48 layers"))
        XCTAssertTrue(banner.contains("Multimodal Vision  : Active"))
        XCTAssertTrue(banner.contains("POST /v1/chat/completions"))
        XCTAssertTrue(banner.contains("POST /v1/models/load"))
        XCTAssertTrue(banner.contains("POST /v1/models/unload"))
        XCTAssertTrue(banner.contains("POST /api/load"))
        XCTAssertTrue(banner.contains("POST /api/unload"))
        XCTAssertTrue(banner.contains("Prefill Chunk Size : \(QWPrefillLimits.chunkRows) tokens"))
        XCTAssertTrue(banner.contains("http://127.0.0.1:8080"))
        // The old hardcoded strings must be gone entirely.
        XCTAssertFalse(banner.contains("D=5120"))
        XCTAssertFalse(banner.contains("40 Q-heads"))
        XCTAssertFalse(banner.contains("slot capacity: 1M tokens"))
        XCTAssertFalse(banner.contains("28-layer ViT"))
    }

    /// A backend that reports no descriptor gets an honest unknown —
    /// never invented numbers.
    func testBannerWithoutDescriptorReportsUnknownInsteadOfLiterals() {
        let hw = QWMachineHardwareInfo(
            modelIdentifier: "Mac17,6", cpuBrand: "Apple M5 Max",
            physicalCores: 18, logicalCores: 18,
            physicalMemoryBytes: 137_438_953_472,
            osProductVersion: "27.2", kernelRelease: "27.2.0"
        )
        let params = QWEngineParametersInfo(
            modelPackPath: "/models/whatever",
            modelName: "whatever"
        )
        let banner = QWHardwareBanner.renderBanner(
            hardware: hw, caps: nil, params: params, host: "127.0.0.1", port: 8080
        )
        XCTAssertTrue(banner.contains("(unknown"),
                    "missing descriptor must say unknown, not fabricate")
        XCTAssertFalse(banner.contains("D=5120"))
        XCTAssertFalse(banner.contains("Qwen3.8-27B (D=5120"))
    }
}
