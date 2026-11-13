// QWPrefixHealingConstraint.swift — Token healing constraint over byte-prefix trie paths.

import Foundation
import QWHead

final class QWPrefixHealingConstraint: QWTokenConstraint, @unchecked Sendable {
    private let prefixBytes: [UInt8]
    private let table: QWJSONTokenTable
    private let completable: [Bool]
    private var consumed: Int = 0

    var hasSupport: Bool { !prefixBytes.isEmpty && completable[0] }
    var allowsSpeculation: Bool { true }
    var isUnconstrainedNow: Bool { consumed >= prefixBytes.count }
    var speculationBoundaryToken: Int? { nil }

    init(prefixBytes: [UInt8], table: QWJSONTokenTable) {
        self.prefixBytes = prefixBytes
        self.table = table
        let total = prefixBytes.count
        var dp = [Bool](repeating: false, count: total + 1)
        dp[total] = true
        let trie = table.fullTrie
        for i in stride(from: total - 1, through: 0, by: -1) {
            var node: Int32 = 0
            for step in 0..<(total - i) {
                guard let child = Self.childNode(in: trie, node: node, byte: prefixBytes[i + step]) else { break }
                let len = step + 1
                if len < total - i {
                    if trie.tokenCount[Int(child)] > 0 && dp[i + len] {
                        dp[i] = true
                        break
                    }
                } else if Self.subtreeHasTokens(trie, node: child) {
                    dp[i] = true
                    break
                }
                node = child
            }
        }
        self.completable = dp
    }

    func allowedTokens() throws -> [Int]? {
        guard consumed < prefixBytes.count else { return nil }
        let trie = table.fullTrie
        let remCount = prefixBytes.count - consumed
        var out: [Int] = []
        var node: Int32 = 0
        for step in 0..<remCount {
            guard let child = Self.childNode(in: trie, node: node, byte: prefixBytes[consumed + step]) else {
                break
            }
            let len = step + 1
            if len < remCount {
                if completable[consumed + len] {
                    Self.appendNodeTokens(trie, node: child, into: &out)
                }
            } else {
                Self.collectSubtreeTokens(trie, root: child, into: &out)
            }
            node = child
        }
        out.sort()
        guard !out.isEmpty else { throw QWTokenConstraintError.emptySupport }
        return out
    }

    func accept(_ token: Int) throws {
        guard consumed < prefixBytes.count else { return }
        guard token >= 0 && token < table.vocabulary,
              let bytes = table.bytes[token], !bytes.isEmpty else {
            throw QWTokenConstraintError.rejectedToken(token)
        }
        let remCount = prefixBytes.count - consumed
        if bytes.count < remCount {
            guard prefixBytes[consumed..<(consumed + bytes.count)].elementsEqual(bytes),
                  completable[consumed + bytes.count] else {
                throw QWTokenConstraintError.rejectedToken(token)
            }
            consumed += bytes.count
        } else {
            guard bytes.prefix(remCount).elementsEqual(prefixBytes[consumed...]) else {
                throw QWTokenConstraintError.rejectedToken(token)
            }
            consumed = prefixBytes.count
        }
    }

    private static func childNode(in trie: QWJSONTokenTable.Trie, node: Int32, byte: UInt8) -> Int32? {
        var lo = Int(trie.childStart[Int(node)])
        var hi = lo + Int(trie.childCount[Int(node)]) - 1
        while lo <= hi {
            let mid = lo + ((hi - lo) >> 1)
            let b = trie.childByte[mid]
            if b == byte { return trie.childNode[mid] }
            if b < byte { lo = mid + 1 } else { hi = mid - 1 }
        }
        return nil
    }

    private static func appendNodeTokens(_ trie: QWJSONTokenTable.Trie, node: Int32, into out: inout [Int]) {
        let start = Int(trie.tokenStart[Int(node)])
        let count = Int(trie.tokenCount[Int(node)])
        for i in start..<(start + count) { out.append(Int(trie.tokens[i])) }
    }

    private static func subtreeHasTokens(_ trie: QWJSONTokenTable.Trie, node: Int32) -> Bool {
        var stack: [Int32] = [node]
        while let cur = stack.popLast() {
            if trie.tokenCount[Int(cur)] > 0 { return true }
            let start = Int(trie.childStart[Int(cur)])
            let count = Int(trie.childCount[Int(cur)])
            for i in start..<(start + count) { stack.append(trie.childNode[i]) }
        }
        return false
    }

    private static func collectSubtreeTokens(_ trie: QWJSONTokenTable.Trie, root: Int32, into out: inout [Int]) {
        var stack: [Int32] = [root]
        while let cur = stack.popLast() {
            appendNodeTokens(trie, node: cur, into: &out)
            let start = Int(trie.childStart[Int(cur)])
            let count = Int(trie.childCount[Int(cur)])
            for i in start..<(start + count) { stack.append(trie.childNode[i]) }
        }
    }
}
