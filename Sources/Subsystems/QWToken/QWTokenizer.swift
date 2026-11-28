// QWTokenizer — C17: byte-level BPE loading the checkpoint's OWN
// tokenizer.json at runtime (no HF runtime; spec-17). Parity port: every
// rule below is bound by the 10k-case goldens (gen_token_fixtures.py).
//
// Engine contract (read FROM the file, never hardcoded — spec-17):
//   normalizer    NFC   → Foundation (ICU) NFC
//   pre_tokenizer Sequence[Split(Isolated regex), ByteLevel(use_regex=false)]
import Foundation
import QWCore

public final class QWTokenizer: @unchecked Sendable {
    // All state is value-type `let` set in init; the only non-Sendable
    // member is NSRegularExpression (internally immutable post-compile and
    // thread-safe for matching — @unchecked above is that fact, not a lie).

    enum PreTokenStep {
        case digits(individualDigits: Bool, regex: NSRegularExpression)
        case splitIsolated(regex: NSRegularExpression)
        case splitRemovedInverted(regex: NSRegularExpression)
    }

    enum Mode {
        case byteLevel
        case metaspaceByteFallback(replacement: String, prependScheme: String, splitWords: Bool)
    }

    struct AddedPattern: Sendable {
        let pat: [UInt8]
        let ord: Int
        let id: Int
        let singleWord: Bool
        let lstrip: Bool
        let rstrip: Bool
        let normalized: Bool
    }

    struct MergeRule: Sendable {
        let rank: Int
        let merged: UInt32
    }

    let mode: Mode
    let usesNFC: Bool
    let stripLeadingSpace: Bool
    let ignoreMerges: Bool
    /// byte → ByteLevel symbol scalar (derived at load; law above).
    let byteMap: [Unicode.Scalar]
    /// vocab symbol (ByteLevel string) → token id.
    let vocab: [String: Int]
    /// pair key → merged rank + symbol interner id; ABSENT when unmergeable.
    let mergeRules: [UInt64: MergeRule]
    /// interner: vocab symbol string → dense id (BPE closure stays in vocab)
    let symId: [String: UInt32]
    let syms: [String]
    /// interner id → token id (-1 when absent)
    let symTok: [Int]
    /// token id → raw vocab piece (ByteLevel string)
    let idToPiece: [Int: String]
    /// ByteLevel symbol scalar → original byte (inverse of byteMap)
    let byteUnmap: [Unicode.Scalar: UInt8]
    /// ByteFallback token id (<0x00>..<0xFF>) → original byte
    let byteFallbackUnmap: [Int: UInt8]
    /// added tokens (file order preserved; special flag from the file)
    let added: [(s: String, id: Int)]
    let specialIds: Set<Int>
    /// chained pre-tokenizer stages (from the file, compiled at load)
    let preSteps: [PreTokenStep]
    /// byte -> symbol interner id (direct O(1) table, zero allocations)
    let byteToSymId: [UInt32]
    /// byte -> token id (direct O(1) table for 1-byte spans, -1 if unresolved)
    let byteToTokId: [Int]
    /// pre-encoded added token patterns sorted longest-first (tie-break file order)
    let rawAddedPats: [AddedPattern]
    let rawAddedFirstByte: [Bool]
    let normAddedPats: [AddedPattern]
    let normAddedFirstByte: [Bool]
    let postProcessorPrefixIds: [Int]
    let postProcessorSuffixIds: [Int]
    let storedTokenizerConfigJSON: Data?
    let storedSpecialTokensMapJSON: Data?
    let storedGenerationConfigJSON: Data?

    public enum Error: Swift.Error, CustomStringConvertible {
        case badJSON(String), unresolvedByte(String), regex(String)
        public var description: String {
            switch self {
            case .badJSON(let m): return "QWTokenizer: tokenizer.json — \(m)"
            case .unresolvedByte(let s): return "QWTokenizer: byte '\(s)' not in vocab — no-fallback BPE must NOT drop it (risk-1)"
            case .regex(let m): return "QWTokenizer: pre_tokenizer regex — \(m)"
            }
        }
    }

    // MARK: load
    public init(
        json: Data,
        tokenizerConfigJSON: Data? = nil,
        specialTokensMapJSON: Data? = nil,
        generationConfigJSON: Data? = nil
    ) throws {
        self.storedTokenizerConfigJSON = tokenizerConfigJSON
        self.storedSpecialTokensMapJSON = specialTokensMapJSON
        self.storedGenerationConfigJSON = generationConfigJSON
        guard let root = try JSONSerialization.jsonObject(with: json) as? [String: Any] else {
            throw Error.badJSON("not an object")
        }
        guard let model = root["model"] as? [String: Any],
              model["type"] as? String == "BPE",
              let vocabObj = model["vocab"] as? [String: Int],
              let mergesArr = model["merges"] as? [Any] else {
            throw Error.badJSON("model.type/vocab/merges shape unexpected")
        }
        let pre = root["pre_tokenizer"] as? [String: Any]
        let preType = pre?["type"] as? String
        let subs = (pre?["pretokenizers"] as? [[String: Any]]) ?? (pre.map { [$0] } ?? [])
        let normObj = root["normalizer"] as? [String: Any]
        let normType = normObj?["type"] as? String
        let detectedMode: Mode
        var compiledSteps: [PreTokenStep] = []
        if preType == "Sequence",
           subs.contains(where: { ($0["type"] as? String) == "ByteLevel" && ($0["use_regex"] as? Bool) == false }) {
            guard normType == nil || normType == "NFC" else {
                throw Error.badJSON("normalizer ≠ NFC")
            }
            for sub in subs {
                let stype = sub["type"] as? String
                if stype == "ByteLevel" {
                    guard (sub["use_regex"] as? Bool) == false else {
                        throw Error.badJSON("ByteLevel use_regex=true unsupported")
                    }
                } else if stype == "Digits" {
                    let indiv = (sub["individual_digits"] as? Bool) ?? false
                    let dPat = indiv ? "\\p{Nd}" : "\\p{Nd}+"
                    guard let re = try? NSRegularExpression(pattern: dPat) else {
                        throw Error.regex("ICU refused Digits regex")
                    }
                    compiledSteps.append(.digits(individualDigits: indiv, regex: re))
                } else if stype == "Split" {
                    let behavior = sub["behavior"] as? String
                    let invert = (sub["invert"] as? Bool) ?? false
                    guard let rPat = (sub["pattern"] as? [String: Any])?["Regex"] as? String,
                          (behavior == "Isolated" && !invert) || (behavior == "Removed" && invert) else {
                        throw Error.badJSON("unsupported Split pre_tokenizer configuration")
                    }
                    let icuPat = Self.normalizeOnigStackedExactRepeat(rPat)
                    guard let re = try? NSRegularExpression(pattern: icuPat) else {
                        throw Error.regex("ICU refused \(rPat.prefix(40))…")
                    }
                    compiledSteps.append(invert ? .splitRemovedInverted(regex: re) : .splitIsolated(regex: re))
                } else {
                    throw Error.badJSON("unsupported Sequence pre_tokenizer stage '\(stype ?? "nil")'")
                }
            }
            guard !compiledSteps.isEmpty else {
                throw Error.badJSON("pre_tokenizer Sequence has no Split/Digits stages before ByteLevel")
            }
            detectedMode = .byteLevel
        } else if let metaSub = subs.first(where: { ($0["type"] as? String) == "Metaspace" }),
                  (model["byte_fallback"] as? Bool) == true {
            let rep = (metaSub["replacement"] as? String) ?? "\u{2581}"
            let scheme = (metaSub["prepend_scheme"] as? String) ?? "always"
            let sp = (metaSub["split"] as? Bool) ?? false
            detectedMode = .metaspaceByteFallback(replacement: rep, prependScheme: scheme, splitWords: sp)
        } else {
            throw Error.badJSON("pre_tokenizer Sequence[Split(Isolated),ByteLevel(no regex)] not found")
        }
        self.mode = detectedMode
        self.usesNFC = (normType == "NFC")
        self.ignoreMerges = (model["ignore_merges"] as? Bool) ?? false
        let decSubs = ((root["decoder"] as? [String: Any])?["decoders"] as? [[String: Any]]) ?? []
        self.stripLeadingSpace = decSubs.contains(where: {
            ($0["type"] as? String) == "Strip" && ($0["content"] as? String) == " " && (($0["start"] as? Int) ?? 0) > 0
        })
        self.preSteps = compiledSteps
        self.vocab = vocabObj

        struct AddedEntry {
            let content: String
            let id: Int
            let singleWord: Bool
            let lstrip: Bool
            let rstrip: Bool
            let normalized: Bool
        }
        var addedList: [(String, Int)] = []
        var addedEntries: [AddedEntry] = []
        var spIds: Set<Int> = []
        if let ats = root["added_tokens"] as? [[String: Any]] {
            for t in ats {
                guard let c = t["content"] as? String, let id = t["id"] as? Int else {
                    throw Error.badJSON("added_tokens entry")
                }
                let isSpecial = (t["special"] as? Bool) ?? false
                let normalized = (t["normalized"] as? Bool) ?? !isSpecial
                let singleWord = (t["single_word"] as? Bool) ?? false
                let lstrip = (t["lstrip"] as? Bool) ?? false
                let rstrip = (t["rstrip"] as? Bool) ?? false
                addedList.append((c, id))
                addedEntries.append(AddedEntry(
                    content: c, id: id, singleWord: singleWord,
                    lstrip: lstrip, rstrip: rstrip, normalized: normalized
                ))
                if isSpecial { spIds.insert(id) }
            }
        }
        self.added = addedList
        self.specialIds = spIds

        var map = [Unicode.Scalar](repeating: " ", count: 256)
        var unmap = [Unicode.Scalar: UInt8](minimumCapacity: 256)
        var bfUnmap = [Int: UInt8]()
        switch detectedMode {
        case .byteLevel:
            var n = 0
            for b in 0..<256 {
                let isDirect = (b >= 0x21 && b <= 0x7E)
                    || (b >= 0xA1 && b <= 0xAC)
                    || (b >= 0xAE && b <= 0xFF)
                let sc: Unicode.Scalar
                if isDirect {
                    sc = Unicode.Scalar(UInt8(b))
                } else {
                    guard let shifted = Unicode.Scalar(0x100 + n) else {
                        throw Error.badJSON("byte fold overflow at \(b)")
                    }
                    sc = shifted
                    n += 1
                }
                let isImpossibleUTF8 = (b == 0xC0 || b == 0xC1 || b >= 0xF5)
                if vocabObj[String(sc)] == nil && !isImpossibleUTF8 {
                    throw Error.badJSON("no vocab token for byte \(b) (\(String(sc).debugDescription))")
                }
                map[b] = sc
            }
            for (b, sc) in map.enumerated() { unmap[sc] = UInt8(b) }
        case .metaspaceByteFallback:
            for b in 0..<256 {
                let hexKey = String(format: "<0x%02X>", b)
                guard let tid = vocabObj[hexKey] else {
                    throw Error.badJSON("missing byte_fallback token \(hexKey)")
                }
                bfUnmap[tid] = UInt8(b)
            }
        }
        self.byteMap = map
        self.byteUnmap = unmap
        self.byteFallbackUnmap = bfUnmap

        // interner + eager merge-closure tables (load once, encode hot)
        guard vocabObj.count < Int(UInt32.max) else {
            throw Error.badJSON("vocab exceeds UInt32 symbol capacity")
        }
        var sId = [String: UInt32](minimumCapacity: vocabObj.count)
        var sList: [String] = []
        sList.reserveCapacity(vocabObj.count)
        for (s, _) in vocabObj { sId[s] = UInt32(sList.count); sList.append(s) }
        var mr = [UInt64: MergeRule](minimumCapacity: mergesArr.count)
        for entry in mergesArr {
            let part0: String
            let part1: String
            if let a = entry as? [String], a.count == 2 {
                part0 = a[0]
                part1 = a[1]
            } else if let s = entry as? String {
                let utf8 = Array(s.utf8)
                guard let spaceIdx = utf8.firstIndex(of: 0x20) else {
                    throw Error.badJSON("merge refs non-vocab symbol: \(s)")
                }
                part0 = String(decoding: utf8[0..<spaceIdx], as: UTF8.self)
                part1 = String(decoding: utf8[(spaceIdx + 1)..<utf8.count], as: UTF8.self)
            } else {
                throw Error.badJSON("merge entry type")
            }
            guard let a = sId[part0], let b = sId[part1] else {
                if case .metaspaceByteFallback = detectedMode { continue }
                throw Error.badJSON("merge refs non-vocab symbol: \(part0) \(part1)")
            }
            let k = UInt64(a) << 32 | UInt64(b)
            guard let m = sId[sList[Int(a)] + sList[Int(b)]] else { continue }  // unmergeable pair (see law above)
            if mr[k] == nil { mr[k] = MergeRule(rank: mr.count, merged: m) }   // first rank wins (list order)
        }
        self.mergeRules = mr
        self.symId = sId
        self.syms = sList
        var st = [Int](repeating: -1, count: sList.count)
        for (s, id) in sId {
            if let tid = vocabObj[s] { st[Int(id)] = tid }
        }
        self.symTok = st
        var ip = [Int: String](minimumCapacity: vocabObj.count + addedList.count)
        for (s, id) in vocabObj { ip[id] = s }
        for (s, id) in addedList { ip[id] = s }
        self.idToPiece = ip

        var b2s = [UInt32](repeating: 0, count: 256)
        switch detectedMode {
        case .byteLevel:
            for (b, sc) in map.enumerated() {
                let key = String(String.UnicodeScalarView([sc]))
                if let sid = sId[key] {
                    b2s[b] = sid
                } else if b == 0xC0 || b == 0xC1 || b >= 0xF5 {
                    b2s[b] = UInt32.max
                } else {
                    throw Error.badJSON("byte symbol \(key) not in symbol map")
                }
            }
        case .metaspaceByteFallback:
            for b in 0..<256 {
                let hexKey = String(format: "<0x%02X>", b)
                guard let sid = sId[hexKey] else {
                    throw Error.badJSON("byte_fallback symbol \(hexKey) not in symbol map")
                }
                b2s[b] = sid
            }
        }
        self.byteToSymId = b2s
        var b2t = [Int](repeating: -1, count: 256)
        for b in 0..<256 {
            let sid = b2s[b]
            if sid != UInt32.max, Int(sid) < st.count {
                b2t[b] = st[Int(sid)]
            }
        }
        self.byteToTokId = b2t

        var rawPats: [AddedPattern] = []
        var rawFirst = [Bool](repeating: false, count: 256)
        var normPats: [AddedPattern] = []
        var normFirst = [Bool](repeating: false, count: 256)
        let nfcEnabled = (normType == "NFC")
        for (ord, entry) in addedEntries.enumerated() {
            if entry.normalized {
                let patStr = nfcEnabled ? entry.content.precomposedStringWithCanonicalMapping : entry.content
                let p = Array(patStr.utf8)
                if !p.isEmpty {
                    normPats.append(AddedPattern(
                        pat: p, ord: ord, id: entry.id, singleWord: entry.singleWord,
                        lstrip: entry.lstrip, rstrip: entry.rstrip, normalized: true
                    ))
                    normFirst[Int(p[0])] = true
                }
            } else {
                let p = Array(entry.content.utf8)
                if !p.isEmpty {
                    rawPats.append(AddedPattern(
                        pat: p, ord: ord, id: entry.id, singleWord: entry.singleWord,
                        lstrip: entry.lstrip, rstrip: entry.rstrip, normalized: false
                    ))
                    rawFirst[Int(p[0])] = true
                }
            }
        }
        let sortLongestFirst: (AddedPattern, AddedPattern) -> Bool = {
            $0.pat.count != $1.pat.count ? $0.pat.count > $1.pat.count : $0.ord < $1.ord
        }
        self.rawAddedPats = rawPats.sorted(by: sortLongestFirst)
        self.rawAddedFirstByte = rawFirst
        self.normAddedPats = normPats.sorted(by: sortLongestFirst)
        self.normAddedFirstByte = normFirst
        let (ppPrefix, ppSuffix) = Self.parseTemplateProcessingSingle(
            root["post_processor"] as? [String: Any],
            vocab: vocabObj,
            added: addedList
        )
        self.postProcessorPrefixIds = ppPrefix
        self.postProcessorSuffixIds = ppSuffix
    }

    private static func parseTemplateProcessingSingle(
        _ postObj: [String: Any]?,
        vocab: [String: Int],
        added: [(s: String, id: Int)]
    ) -> (prefix: [Int], suffix: [Int]) {
        guard let postObj else { return ([], []) }
        let tplObj: [String: Any]?
        if (postObj["type"] as? String) == "TemplateProcessing" {
            tplObj = postObj
        } else if (postObj["type"] as? String) == "Sequence",
                  let procs = postObj["processors"] as? [[String: Any]] {
            tplObj = procs.first(where: { ($0["type"] as? String) == "TemplateProcessing" })
        } else {
            tplObj = nil
        }
        guard let tpl = tplObj, let single = tpl["single"] as? [[String: Any]] else {
            return ([], [])
        }
        let spMap = tpl["special_tokens"] as? [String: [String: Any]]
        func resolveSpecialIds(_ name: String) -> [Int] {
            if let entry = spMap?[name], let rawIds = entry["ids"] as? [Any] {
                let ids = rawIds.compactMap { ($0 as? NSNumber)?.intValue }
                if !ids.isEmpty { return ids }
            }
            if let tid = vocab[name] ?? added.first(where: { $0.s == name })?.id {
                return [tid]
            }
            return []
        }
        var prefix: [Int] = []
        var suffix: [Int] = []
        var sawSequenceA = false
        for item in single {
            if item["Sequence"] != nil {
                sawSequenceA = true
            } else if let sp = item["SpecialToken"] as? [String: Any],
                      let spId = sp["id"] as? String {
                let resolved = resolveSpecialIds(spId)
                if sawSequenceA {
                    suffix.append(contentsOf: resolved)
                } else {
                    prefix.append(contentsOf: resolved)
                }
            }
        }
        return sawSequenceA ? (prefix, suffix) : ([], [])
    }

    /// Onig/fancy-regex (HF `tokenizers`) parses `<atom>{n}+` as `(?:<atom>{n})+`,
    /// whereas ICU parses `{n}+` as a possessive exact-`n` quantifier.
    private static func normalizeOnigStackedExactRepeat(_ pattern: String) -> String {
        guard let meta = try? NSRegularExpression(
            pattern: #"(\\[pP]\{[^}]+\}|\\.|\[[^\]]+\])\{(\d+)\}\+"#
        ) else {
            return pattern
        }
        let ns = pattern as NSString
        return meta.stringByReplacingMatches(
            in: pattern,
            range: NSRange(location: 0, length: ns.length),
            withTemplate: "(?:$1{$2})+"
        )
    }
}

/// Thread-local bounded per-`encode` BPE word cache (max 4096 keys, zero per-lookup allocs).
struct QWWordMergeCache {
    private struct Entry {
        let hash: UInt64
        let keyStart: UInt32
        let keyLen: UInt16
        let outStart: UInt32
        let outLen: UInt16
    }

    static let maxEntries = 4096
    static let maxKeyBytes = 256

    private var slots: [Int32] = []
    private var entries: [Entry] = []
    private var keyBytes: [UInt8] = []
    private var outTokens: [Int] = []

    @inline(__always)
    static func hashBytes(_ ptr: UnsafePointer<UInt8>, count: Int) -> UInt64 {
        var h: UInt64 = 0xCBF2_9CE4_8422_2325
        var i = 0
        while i < count {
            h ^= UInt64(ptr[i])
            h &*= 0x0000_0100_0000_01B3
            i += 1
        }
        h ^= h >> 33
        h &*= 0xFF51_AFD7_ED55_8CCD
        h ^= h >> 33
        return h
    }

    @inline(__always)
    func lookup(
        _ ptr: UnsafePointer<UInt8>,
        count: Int,
        hash: UInt64,
        into ids: inout [Int]
    ) -> Bool {
        let slotCount = slots.count
        guard slotCount > 0 else { return false }
        let mask = slotCount - 1
        var idx = Int(hash & UInt64(mask))
        while true {
            let slotVal = slots[idx]
            if slotVal < 0 { return false }
            let e = entries[Int(slotVal)]
            if e.hash == hash && Int(e.keyLen) == count {
                let kBase = Int(e.keyStart)
                let match = keyBytes.withUnsafeBufferPointer { kb -> Bool in
                    guard let kAddr = kb.baseAddress else { return false }
                    return memcmp(kAddr.advanced(by: kBase), ptr, count) == 0
                }
                if match {
                    let oStart = Int(e.outStart)
                    let oEnd = oStart + Int(e.outLen)
                    ids.append(contentsOf: outTokens[oStart..<oEnd])
                    return true
                }
            }
            idx = (idx + 1) & mask
        }
    }

    mutating func insert(
        _ ptr: UnsafePointer<UInt8>,
        count: Int,
        hash: UInt64,
        tokens: [Int]
    ) {
        guard entries.count < Self.maxEntries,
              count > 0, count <= Self.maxKeyBytes,
              tokens.count <= Int(UInt16.max),
              keyBytes.count <= Int(UInt32.max) - count,
              outTokens.count <= Int(UInt32.max) - tokens.count else { return }
        if slots.isEmpty {
            slots = [Int32](repeating: -1, count: 256)
            entries.reserveCapacity(128)
            keyBytes.reserveCapacity(1024)
            outTokens.reserveCapacity(256)
        } else if (entries.count + 1) * 2 > slots.count {
            rehash(to: slots.count * 2)
        }
        let keyStart = UInt32(keyBytes.count)
        keyBytes.append(contentsOf: UnsafeBufferPointer(start: ptr, count: count))
        let outStart = UInt32(outTokens.count)
        outTokens.append(contentsOf: tokens)
        let entryIdx = Int32(entries.count)
        entries.append(Entry(
            hash: hash,
            keyStart: keyStart,
            keyLen: UInt16(count),
            outStart: outStart,
            outLen: UInt16(tokens.count)
        ))
        let mask = slots.count - 1
        var idx = Int(hash & UInt64(mask))
        while slots[idx] >= 0 {
            idx = (idx + 1) & mask
        }
        slots[idx] = entryIdx
    }

    private mutating func rehash(to newCap: Int) {
        var next = [Int32](repeating: -1, count: newCap)
        let mask = newCap - 1
        for (i, e) in entries.enumerated() {
            var idx = Int(e.hash & UInt64(mask))
            while next[idx] >= 0 {
                idx = (idx + 1) & mask
            }
            next[idx] = Int32(i)
        }
        slots = next
    }
}
