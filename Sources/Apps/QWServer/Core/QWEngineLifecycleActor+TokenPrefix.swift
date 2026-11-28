// QWEngineLifecycleActor+TokenPrefix.swift — Logprobs, stop suffixes, and incremental token prefix encoding.

import Foundation
import QWCli
import QWCore
import QWEngine
import QWHAL
import QWHead
import QWKV
import QWMM
import QWQuant
import QWToken
import QWVision

extension QWEngineLifecycleActor {
    nonisolated static func makeLogprobEntry(_ lp: QWTokenLogprob, tokenizer: QWTokenizer) -> ChatLogprobTokenEntry {
        let tokStr = (try? tokenizer.decode([lp.tokenId], skipSpecialTokens: false)) ?? ""
        let tokBytes = Array(tokStr.utf8).map { Int($0) }
        let tops: [ChatLogprobTokenEntry.TopLogprob] = lp.topCandidates.map { cand in
            let cStr = (try? tokenizer.decode([cand.tokenId], skipSpecialTokens: false)) ?? ""
            let cBytes = Array(cStr.utf8).map { Int($0) }
            return ChatLogprobTokenEntry.TopLogprob(token: cStr, logprob: cand.logprob, bytes: cBytes)
        }
        return ChatLogprobTokenEntry(token: tokStr, logprob: lp.logprob, bytes: tokBytes, topLogprobs: tops)
    }

    struct IncrementalStopScanner: @unchecked Sendable {
        struct Entry {
            let stop: String
            let bytes: [UInt8]
            let properPrefixes: [[UInt8]]
        }
        let entries: [Entry]
        let isEmpty: Bool

        init(stops: [String]) {
            var built: [Entry] = []
            for stop in stops where !stop.isEmpty {
                var prefixes: [[UInt8]] = []
                var idx = stop.endIndex
                while idx > stop.startIndex {
                    idx = stop.index(before: idx)
                    if idx > stop.startIndex {
                        prefixes.append(Array(stop[..<idx].utf8))
                    }
                }
                built.append(Entry(stop: stop, bytes: Array(stop.utf8), properPrefixes: prefixes))
            }
            self.entries = built
            self.isEmpty = built.isEmpty
        }

        func firstMatch(in utf8: [UInt8], previousCount: Int) -> (byteOffset: Int, stop: String)? {
            var earliest: (byteOffset: Int, stop: String)? = nil
            for entry in entries {
                let m = entry.bytes.count
                guard m > 0, utf8.count >= m else { continue }
                let scanStart = max(0, previousCount - (m - 1))
                let scanEnd = utf8.count - m
                guard scanStart <= scanEnd else { continue }
                let firstByte = entry.bytes[0]
                for i in scanStart...scanEnd where utf8[i] == firstByte {
                    if let current = earliest, i >= current.byteOffset { break }
                    var matched = true
                    for j in 1..<m where utf8[i + j] != entry.bytes[j] {
                        matched = false
                        break
                    }
                    if matched {
                        earliest = (i, entry.stop)
                        break
                    }
                }
            }
            return earliest
        }

        func heldSuffixUTF8Count(in utf8: [UInt8]) -> Int {
            var best = 0
            for entry in entries {
                for prefix in entry.properPrefixes {
                    let len = prefix.count
                    guard len > best, utf8.count >= len else { continue }
                    if utf8[(utf8.count - len)..<utf8.count].elementsEqual(prefix) {
                        best = len
                        break
                    }
                }
            }
            return best
        }
    }

    struct IncrementalJSONObjectTracker: Sendable {
        private var started = false
        private var depth = 0
        private var inString = false
        private var escaped = false
        private var scannedCount = 0
        private(set) var completedByteOffset: Int? = nil

        mutating func advance(utf8: [UInt8]) {
            guard completedByteOffset == nil, scannedCount < utf8.count else { return }
            for idx in scannedCount..<utf8.count {
                let b = utf8[idx]
                if inString {
                    if escaped {
                        escaped = false
                    } else if b == 0x5C {
                        escaped = true
                    } else if b == 0x22 {
                        inString = false
                    }
                    continue
                }
                if b == 0x22 {
                    inString = true
                } else if b == 0x7B {
                    depth += 1
                    started = true
                } else if b == 0x7D {
                    depth -= 1
                    if started && depth == 0 {
                        scannedCount = idx + 1
                        completedByteOffset = idx + 1
                        return
                    }
                }
            }
            scannedCount = utf8.count
        }
    }

    nonisolated static func firstCompleteJsonObjectEnd(in text: String) -> String.Index? {
        var depth = 0
        var inString = false
        var escaped = false
        var started = false
        for idx in text.indices {
            let ch = text[idx]
            if inString {
                if escaped {
                    escaped = false
                } else if ch == "\\" {
                    escaped = true
                } else if ch == "\"" {
                    inString = false
                }
                continue
            }
            if ch == "\"" {
                inString = true
            } else if ch == "{" {
                depth += 1
                started = true
            } else if ch == "}" {
                depth -= 1
                if started && depth == 0 {
                    return text.index(after: idx)
                }
            }
        }
        return nil
    }

    /// Incrementally encodes a rendered chat wire string on caller executor using a lock-protected
    /// token prefix cache up to `<|im_start|>` turn boundaries across concurrent sessions.
    nonisolated func encodeWireIncrementallyWithSystemPrefix(
        wire: String,
        tokenizer tok: QWTokenizer,
        packPath: String,
        cachePrompt: Bool = true,
        cacheSalt: String? = nil
    ) throws -> (tokens: [Int], systemPrefixTokenCount: Int?) {
        guard cachePrompt else {
            return (try tok.encode(wire), nil)
        }
        let saltKey = cacheSalt?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let marker: String
        let roleBeforeMarker: Bool
        if tok.tokenId(for: "<|im_start|>") != nil {
            marker = "<|im_start|>"
            roleBeforeMarker = false
        } else if tok.tokenId(for: "<|maxwell_start|>") != nil {
            marker = "<|maxwell_start|>"
            roleBeforeMarker = false
        } else if tok.tokenId(for: "<|role_sep|>\n") != nil {
            marker = "<|role_sep|>\n"
            roleBeforeMarker = true
        } else if tok.tokenId(for: "<|role_sep|>") != nil {
            marker = "<|role_sep|>"
            roleBeforeMarker = true
        } else {
            return (try tok.encode(wire), nil)
        }
        guard let lastRange = wire.range(of: marker, options: .backwards),
              lastRange.lowerBound > wire.startIndex else {
            return (try tok.encode(wire), nil)
        }
        let maxEntries = max(12, getConfig().maxConcurrentSessions * 2 + 4)

        var systemWire: String? = nil
        if roleBeforeMarker {
            if (wire.hasPrefix("<s>system\(marker)") || wire.hasPrefix("system\(marker)")
                || wire.hasPrefix("<s>developer\(marker)") || wire.hasPrefix("developer\(marker)")),
               let firstRange = wire.range(of: marker),
               let secondRange = wire.range(of: marker, range: firstRange.upperBound..<wire.endIndex),
               secondRange.lowerBound <= lastRange.lowerBound {
                systemWire = String(wire[..<secondRange.lowerBound])
            }
        } else if wire.hasPrefix("\(marker)system") || wire.hasPrefix("\(marker)developer") {
            let afterFirst = wire.index(wire.startIndex, offsetBy: marker.count)
            if let secondRange = wire.range(of: marker, range: afterFirst..<wire.endIndex),
               secondRange.lowerBound <= lastRange.lowerBound {
                systemWire = String(wire[..<secondRange.lowerBound])
            }
        }

        let newPrefixWire = String(wire[..<lastRange.lowerBound])
        let tailWire = String(wire[lastRange.lowerBound...])
        let newUTF8Count = newPrefixWire.utf8.count
        let markerUTF8 = Array(marker.utf8)

        if let matched = tokenPrefixCache.lookupBestPrefix(
            newPrefixWire: newPrefixWire,
            packPath: packPath,
            saltKey: saltKey,
            markerUTF8: markerUTF8
        ) {
            let deltaStart = newPrefixWire.utf8.index(newPrefixWire.utf8.startIndex, offsetBy: matched.prefixWire.utf8.count)
            let deltaWire = String(newPrefixWire[deltaStart...])
            if deltaWire.isEmpty {
                let tailTokens = try tok.encode(tailWire)
                return (matched.prefixTokens + tailTokens, matched.systemTokenCount)
            } else if deltaWire.utf8.starts(with: markerUTF8) {
                let deltaTokens = try tok.encode(deltaWire)
                let updatedPrefixTokens = matched.prefixTokens + deltaTokens
                let sysCount = matched.systemTokenCount
                    ?? (systemWire != nil && matched.prefixWire == systemWire ? matched.prefixTokens.count : nil)
                tokenPrefixCache.store(
                    CachedTokenPrefixEntry(
                        packPath: packPath, cacheSalt: saltKey, prefixWire: newPrefixWire,
                        prefixTokens: updatedPrefixTokens, systemTokenCount: sysCount
                    ),
                    maxEntries: maxEntries
                )
                let tailTokens = try tok.encode(tailWire)
                return (updatedPrefixTokens + tailTokens, sysCount)
            }
        }

        var resolvedSystemTokenCount: Int? = nil
        let prefixTokens: [Int]
        if let sysW = systemWire, sysW.utf8.count < newUTF8Count, newPrefixWire.utf8.starts(with: sysW.utf8) {
            let sysTokens = try tok.encode(sysW)
            resolvedSystemTokenCount = sysTokens.count
            tokenPrefixCache.store(
                CachedTokenPrefixEntry(
                    packPath: packPath, cacheSalt: saltKey, prefixWire: sysW, prefixTokens: sysTokens,
                    systemTokenCount: sysTokens.count
                ),
                maxEntries: maxEntries
            )
            let remStart = newPrefixWire.utf8.index(newPrefixWire.utf8.startIndex, offsetBy: sysW.utf8.count)
            let remTokens = try tok.encode(String(newPrefixWire[remStart...]))
            prefixTokens = sysTokens + remTokens
        } else {
            prefixTokens = try tok.encode(newPrefixWire)
            if let sysW = systemWire, sysW.utf8.count == newUTF8Count {
                resolvedSystemTokenCount = prefixTokens.count
            }
        }
        tokenPrefixCache.store(
            CachedTokenPrefixEntry(
                packPath: packPath, cacheSalt: saltKey, prefixWire: newPrefixWire,
                prefixTokens: prefixTokens, systemTokenCount: resolvedSystemTokenCount
            ),
            maxEntries: maxEntries
        )
        let tailTokens = try tok.encode(tailWire)
        return (prefixTokens + tailTokens, resolvedSystemTokenCount)
    }

    nonisolated public func validateRequestCapabilities(_ request: ChatCompletionRequest) throws {
        guard !request.extractImageUrls().isEmpty || !request.extractVideoInputs().isEmpty else { return }
        let cfg = statusBox.getConfig()
        guard cfg.loadVision else {
            throw QWMultimodalError.badArgument(
                "Vision tower is disabled (--no-load-vision). Enable --load-vision to process multimodal inputs."
            )
        }
        let state = statusBox.engineState()
        if state.isLoaded {
            let visDepth = statusBox.getLoadedParamsInfo()?.modelDescriptor?.visDepth ?? 0
            if !state.capabilities.contains(.vision) || visDepth <= 0
                || !QWVisionOnDemandBackend.hasVisionWeights(modelPath: cfg.packPath) {
                throw QWMultimodalError.badArgument("Loaded model does not support vision inputs.")
            }
        } else if !QWVisionOnDemandBackend.hasVisionWeights(modelPath: cfg.packPath) {
            throw QWMultimodalError.badArgument("Model pack does not contain vision weights.")
        }
    }

    nonisolated static func prepareTokenHeal(
        prepared: QWPreparedChatPrompt,
        request: ChatCompletionRequest,
        promptTokens: [Int],
        systemPrefixTokenCount: Int?,
        stops: Set<Int>,
        tokenizer: QWTokenizer,
        vocabulary: Int,
        supportsTokenConstraints: Bool,
        hasImages: Bool
    ) -> (promptTokens: [Int], constraint: (any QWTokenConstraint)?, stripUTF8Count: Int, didPeel: Bool) {
        guard prepared.isAssistantPrefill,
              prepared.activeConstraint == nil,
              supportsTokenConstraints,
              !hasImages,
              promptTokens.count >= 2 else {
            return (promptTokens, prepared.activeConstraint, 0, false)
        }
        let prefillText = request.messages.last?.extractTextContent() ?? ""
        let prefillUTF8 = Array(prefillText.utf8)
        let lastTok = promptTokens[promptTokens.count - 1]
        guard !prefillUTF8.isEmpty,
              !stops.contains(lastTok),
              (promptTokens.count - 1) >= (systemPrefixTokenCount ?? 0) else {
            return (promptTokens, prepared.activeConstraint, 0, false)
        }
        let table = QWStructuredOutput.tokenTable(for: tokenizer, vocabulary: vocabulary)
        guard lastTok >= 0 && lastTok < table.vocabulary,
              let tokBytes = table.bytes[lastTok], !tokBytes.isEmpty,
              tokBytes.count <= prefillUTF8.count,
              prefillUTF8.suffix(tokBytes.count).elementsEqual(tokBytes) else {
            return (promptTokens, prepared.activeConstraint, 0, false)
        }
        var probe = QWStreamDecoder(tokenizer)
        guard let healedStr = try? probe.decode(lastTok, skipSpecialTokens: true),
              !healedStr.isEmpty, probe.flush().isEmpty else {
            return (promptTokens, prepared.activeConstraint, 0, false)
        }
        let constraint = QWPrefixHealingConstraint(prefixBytes: tokBytes, table: table)
        guard constraint.hasSupport else {
            return (promptTokens, prepared.activeConstraint, 0, false)
        }
        return (Array(promptTokens.dropLast()), constraint, healedStr.utf8.count, true)
    }
}
