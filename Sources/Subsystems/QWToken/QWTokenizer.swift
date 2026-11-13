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
    }

    enum Mode {
        case byteLevel
        case metaspaceByteFallback(replacement: String, prependScheme: String, splitWords: Bool)
    }

    let mode: Mode
    let usesNFC: Bool
    let stripLeadingSpace: Bool
    /// byte → ByteLevel symbol scalar (derived at load; law above).
    let byteMap: [Unicode.Scalar]
    /// vocab symbol (ByteLevel string) → token id.
    let vocab: [String: Int]
    /// merged-pair rank (merges file order). Key = a<<20 | b over interner
    /// ids (sym count 248k « 2^20 cap).
    let ranks: [UInt64: Int]
    /// interner: vocab symbol string → dense id (BPE closure stays in vocab)
    let symId: [String: UInt32]
    let syms: [String]
    /// pair key → merged symbol interner id; ABSENT when the merged string
    /// is not a vocab entry (that pair cannot merge — no fatalError path).
    let pairMerged: [UInt64: UInt32]
    /// interner id → token id
    let symTok: [UInt32: Int]
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
    /// pre-encoded added token patterns for fast scanning
    let addedPats: [(pat: [UInt8], ord: Int, id: Int)]
    /// 256-entry lookup table marking valid first UTF-8 bytes of `addedPats`
    let addedFirstByte: [Bool]

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
    public init(json: Data) throws {
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
                    guard (sub["behavior"] as? String) == "Isolated",
                          ((sub["invert"] as? Bool) ?? false) == false,
                          let rPat = (sub["pattern"] as? [String: Any])?["Regex"] as? String else {
                        throw Error.badJSON("unsupported Split pre_tokenizer configuration")
                    }
                    let icuPat = Self.normalizeOnigStackedExactRepeat(rPat)
                    guard let re = try? NSRegularExpression(pattern: icuPat) else {
                        throw Error.regex("ICU refused \(rPat.prefix(40))…")
                    }
                    compiledSteps.append(.splitIsolated(regex: re))
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
        let decSubs = ((root["decoder"] as? [String: Any])?["decoders"] as? [[String: Any]]) ?? []
        self.stripLeadingSpace = decSubs.contains(where: {
            ($0["type"] as? String) == "Strip" && ($0["content"] as? String) == " " && (($0["start"] as? Int) ?? 0) > 0
        })
        self.preSteps = compiledSteps
        self.vocab = vocabObj

        var addedList: [(String, Int)] = []
        var spIds: Set<Int> = []
        if let ats = root["added_tokens"] as? [[String: Any]] {
            for t in ats {
                guard let c = t["content"] as? String, let id = t["id"] as? Int else {
                    throw Error.badJSON("added_tokens entry")
                }
                addedList.append((c, id))
                if (t["special"] as? Bool) == true { spIds.insert(id) }
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
        var sId = [String: UInt32](minimumCapacity: vocabObj.count)
        var sList: [String] = []
        sList.reserveCapacity(vocabObj.count)
        for (s, _) in vocabObj { sId[s] = UInt32(sList.count); sList.append(s) }
        var rk = [UInt64: Int](minimumCapacity: mergesArr.count)
        var pm = [UInt64: UInt32](minimumCapacity: mergesArr.count)
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
            let k = UInt64(a) << 20 | UInt64(b)
            guard let m = sId[sList[Int(a)] + sList[Int(b)]] else { continue }  // unmergeable pair (see law above)
            if rk[k] == nil { rk[k] = rk.count }   // first rank wins (list order)
            pm[k] = m
        }
        self.ranks = rk
        self.pairMerged = pm
        self.symId = sId
        self.syms = sList
        var st = [UInt32: Int](minimumCapacity: sList.count)
        for (s, id) in sId { st[id] = vocabObj[s] }
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

        var pats: [(pat: [UInt8], ord: Int, id: Int)] = []
        var firstByteTable = [Bool](repeating: false, count: 256)
        for (ord, a) in addedList.enumerated() {
            let p = Array(a.0.utf8)
            if !p.isEmpty {
                pats.append((p, ord, a.1))
                firstByteTable[Int(p[0])] = true
            }
        }
        self.addedPats = pats
        self.addedFirstByte = firstByteTable
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
