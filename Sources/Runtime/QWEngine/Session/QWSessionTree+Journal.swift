// QWSessionTree+Journal.swift — Crash-safe 0o600 JSONL append-only journal & replay for QWSessionTree (G4).

import Darwin
import Foundation

extension QWSessionTree {
    enum JournalEntry: Codable {
        case checkpoint(node: QWSessionNode)
        case prune(nodeIds: [String])
    }

    func appendJournalRecordLocked(_ entry: JournalEntry) throws {
        guard let url = journalURL else { return }
        let fm = FileManager.default
        let dir = url.deletingLastPathComponent()
        if !fm.fileExists(atPath: dir.path) {
            try fm.createDirectory(
                at: dir,
                withIntermediateDirectories: true,
                attributes: [.posixPermissions: NSNumber(value: Int16(0o700))]
            )
        }
        chmod(dir.path, mode_t(0o700))

        let encoder = JSONEncoder()
        var data = try encoder.encode(entry)
        data.append(0x0A)
        do {
            if fm.fileExists(atPath: url.path) {
                let handle = try FileHandle(forWritingTo: url)
                defer { try? handle.close() }
                try handle.seekToEnd()
                try handle.write(contentsOf: data)
            } else {
                try data.write(to: url, options: .atomic)
            }
            chmod(url.path, mode_t(0o600))
        } catch {
            throw QWSessionTreeError.journalFailed(error.localizedDescription)
        }
    }

    func replayJournalLocked(from url: URL) {
        guard let data = try? Data(contentsOf: url), !data.isEmpty else { return }
        let text = String(decoding: data, as: UTF8.self)
        let lines = text.split(separator: "\n", omittingEmptySubsequences: true)
        let decoder = JSONDecoder()
        for (idx, lineSub) in lines.enumerated() {
            let lineData = Data(lineSub.utf8)
            guard let entry = try? decoder.decode(JournalEntry.self, from: lineData) else {
                // Ignore a truncated trailing line after an unclean shutdown.
                if idx == lines.count - 1 { break }
                continue
            }
            switch entry {
            case .checkpoint(let node):
                var clean = node
                clean.childIds = []
                insertNodeLocked(clean)
            case .prune(let nodeIds):
                for rid in nodeIds {
                    if let target = nodes.removeValue(forKey: rid) {
                        if let pid = target.parentId, var parent = nodes[pid] {
                            parent.childIds.removeAll { $0 == rid }
                            nodes[pid] = parent
                        }
                        decrementPinLocked(for: target)
                    }
                }
                let pruned = Set(nodeIds)
                insertionOrder.removeAll { pruned.contains($0) }
            }
        }
    }
}
