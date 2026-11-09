// SoakRecorder.swift — The one writer of the measured windows.
//
// Every window that reaches the gate goes through here. The schema is
// owned by QWCli.SoakSample; the recorder only owns the file.

import Foundation

public extension QWCli {

    /// Writes measured windows into a JSONL file, one window per line.
    /// The file is the telemetry export: `Scripts/gen_soak_sample.py`
    /// turns it into the `m1_soak_*.jsonl` artifact the gate verifies.
    struct SoakRecorder {
        /// Where the windows go.
        public let url: URL
        /// How many windows were written.
        public private(set) var written: Int = 0

        public init(url: URL) {
            self.url = url
        }

        /// The export path: `m1_export_<stamp>.jsonl` in a directory.
        public static func exportURL(in directory: URL, stamp: Date = Date()) -> URL {
            directory.appendingPathComponent("m1_export_\(stampStamp(stamp)).jsonl")
        }

        /// The stamp every soak file carries: seconds since the epoch.
        private static func stampStamp(_ date: Date) -> Int {
            Int(date.timeIntervalSince1970)
        }

        /// Writes one window.
        public mutating func record(_ window: SoakSample) throws {
            try window.append(to: url)
            written += 1
        }

        /// Writes a run of windows.
        public mutating func record(_ windows: [SoakSample]) throws {
            for window in windows { try record(window) }
        }
    }
}
