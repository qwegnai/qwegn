// PullProgress.swift — per-file progress rows for `qwg pull`.
//
// On a terminal the whole file list is drawn as a block with one bar per file, redrawn
// in place as bytes arrive. Off a terminal (piped output, CI logs) nothing is redrawn;
// each file prints one plain line when it finishes, so logs stay readable.

import Foundation

final class PullProgress: @unchecked Sendable {
    struct Row {
        let path: String
        let total: Int
    }

    private enum State: Equatable {
        case pending
        case active
        case cached
        case done
    }

    private struct Slot {
        var received: Int = 0
        var state: State = .pending
        var startedAt: Date?
        var finishedAt: Date?
    }

    private let rows: [Row]
    private let interactive: Bool
    private let color: QWCli.Color
    private let nameWidth: Int
    private let lock = NSLock()
    private var slots: [Slot]
    private var timer: DispatchSourceTimer?
    private var drawnLines = 0

    init(files: [Row], color: QWCli.Color) {
        self.rows = files
        self.color = color
        self.interactive = color.enabled
        self.nameWidth = min(44, files.map { $0.path.count }.max() ?? 0)
        self.slots = Array(repeating: Slot(), count: files.count)
    }

    /// Starts the redraw loop on a terminal. Off a terminal this does nothing.
    func start() {
        guard interactive else { return }
        lock.withLock {
            render()
        }
        let source = DispatchSource.makeTimerSource(queue: DispatchQueue.global(qos: .utility))
        source.schedule(deadline: .now() + .milliseconds(100), repeating: .milliseconds(100))
        source.setEventHandler { [weak self] in
            self?.lock.withLock { self?.render() }
        }
        lock.withLock { timer = source }
        source.resume()
    }

    /// Stops the redraw loop and leaves the final state on screen.
    func finish() {
        let source: DispatchSourceTimer? = lock.withLock {
            let t = timer
            timer = nil
            return t
        }
        source?.cancel()
        lock.withLock {
            if interactive { render() }
        }
    }

    func markCached(_ index: Int) {
        lock.withLock {
            slots[index].state = .cached
            slots[index].received = rows[index].total
            if !interactive {
                emit("  [cached] \(rows[index].path) (\(rows[index].total) B)\n")
            }
        }
    }

    func update(_ index: Int, received: Int) {
        lock.withLock {
            if slots[index].state == .pending {
                slots[index].state = .active
                slots[index].startedAt = Date()
            }
            slots[index].received = min(received, rows[index].total > 0 ? rows[index].total : received)
        }
    }

    func markDone(_ index: Int) {
        lock.withLock {
            let now = Date()
            slots[index].state = .done
            slots[index].received = rows[index].total
            slots[index].finishedAt = now
            if !interactive {
                let started = slots[index].startedAt ?? now
                let secs = max(1e-6, now.timeIntervalSince(started))
                let mib = Double(rows[index].total) / 1_048_576.0
                emit(String(format: "  [done]   %@ (%.2f MiB in %.2fs, %.1f MiB/s)\n",
                            rows[index].path, mib, secs, mib / secs))
            }
        }
    }

    // MARK: - Rendering (call with lock held)

    private func render() {
        guard interactive else { return }
        var out = ""
        if drawnLines > 0 {
            out += "\u{001B}[\(drawnLines)A"   // back to the top of the block
        }
        let now = Date()
        for (i, row) in rows.enumerated() {
            out += "\u{001B}[2K" + line(for: i, row: row, now: now) + "\n"
        }
        drawnLines = rows.count
        emit(out)
    }

    private func line(for index: Int, row: Row, now: Date) -> String {
        let slot = slots[index]
        let name = row.path.count > nameWidth
            ? "…" + String(row.path.suffix(nameWidth - 1))
            : row.path.padding(toLength: nameWidth, withPad: " ", startingAt: 0)
        let barWidth = 24
        let fraction: Double = row.total > 0 ? min(1, Double(slot.received) / Double(row.total)) : 0
        let filled = Int((fraction * Double(barWidth)).rounded())
        let bar = String(repeating: "█", count: filled) + String(repeating: "░", count: barWidth - filled)
        let sizes = "\(Self.size(slot.received))/\(Self.size(row.total))"

        switch slot.state {
        case .pending:
            return "  \(name)  \(color.dim)waiting\(color.reset)"
        case .cached:
            return "  \(name)  \(color.dim)\(bar)\(color.reset)  \(color.dim)cached\(color.reset)  \(sizes)"
        case .done:
            let secs = max(1e-6, (slot.finishedAt ?? now).timeIntervalSince(slot.startedAt ?? now))
            let rate = Double(row.total) / 1_048_576.0 / secs
            return "  \(name)  \(color.bold)\(bar)\(color.reset)  100%  \(sizes)  \(String(format: "%.1f MiB/s", rate))"
        case .active:
            let elapsed = max(1e-6, now.timeIntervalSince(slot.startedAt ?? now))
            let rate = Double(slot.received) / 1_048_576.0 / elapsed
            let pct = String(format: "%3.0f%%", fraction * 100)
            return "  \(name)  \(color.prompt)\(bar)\(color.reset)  \(pct)  \(sizes)  \(String(format: "%.1f MiB/s", rate))"
        }
    }

    /// Sizes read in GiB for large files and MiB for small ones, so a 4 MiB file never shows 0.00 GiB.
    private static func size(_ bytes: Int) -> String {
        bytes >= 1 << 30
            ? String(format: "%.2f GiB", Double(bytes) / 1_073_741_824.0)
            : String(format: "%.1f MiB", Double(bytes) / 1_048_576.0)
    }

    private func emit(_ s: String) {
        fputs(s, stdout)
        fflush(stdout)
    }
}
