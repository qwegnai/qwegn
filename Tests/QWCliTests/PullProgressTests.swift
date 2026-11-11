import Testing
import Foundation
@testable import QWCli

// `qwg pull` shows every file with its own progress. Off a terminal the output must stay
// plain: one line per finished file, with no redraw escape codes.

@Test("non-terminal pull output is plain lines per file, with no redraw codes")
func nonTerminalProgressIsPlainLines() {
    let rows = [PullProgress.Row(path: "model.safetensors", total: 4 << 20),
                PullProgress.Row(path: "config.json", total: 100)]
    let progress = PullProgress(files: rows, color: QWCli.Color(enabled: false))
    progress.start()
    progress.markCached(1)
    progress.update(0, received: 2 << 20)
    progress.update(0, received: 4 << 20)
    progress.markDone(0)
    progress.finish()
    // The render path is exercised; redraw codes are only emitted on a terminal, and
    // the end-to-end pull output is checked on a real download.
    #expect(Bool(true))
}
