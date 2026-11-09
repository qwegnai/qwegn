// FnConfigFixture — the canonical Flash-Next config (the oracle golden the
// packer ships), read from the source tree. Real geometry, zero weights:
// geometry and budget laws must be checkable without a model pack.
import Foundation
import QWCore

enum FnConfigFixture {
    /// Repo fixture URL, located by walking up from the caller's file path.
    static func url(from file: String = #filePath) -> URL {
        var dir = URL(fileURLWithPath: file).deletingLastPathComponent()
        for _ in 0..<8 {
            let u = dir.appendingPathComponent("Scripts/oracle/config_flashnext_golden.json")
            if FileManager.default.fileExists(atPath: u.path) { return u }
            dir = dir.deletingLastPathComponent()
        }
        fatalError("repo fixture unreachable from \(file)")
    }

    /// Decoded production descriptor from the checked-in golden config.
    static func descriptor(from file: String = #filePath) throws -> QWModelDescriptor {
        try QWModelDescriptor.decode(configJSON: Data(contentsOf: url(from: file)))
    }
}
