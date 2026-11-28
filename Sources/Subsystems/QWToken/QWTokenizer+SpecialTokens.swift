// QWTokenizer+SpecialTokens.swift — Piece lookup and config-driven special/stop token resolution.
import CoreFoundation
import Foundation
import QWCore

extension QWTokenizer {
    public func piece(_ id: Int) -> String? {
        idToPiece[id] ?? added.first(where: { $0.id == id })?.s
    }

    public func tokenId(for token: String) -> Int? {
        vocab[token] ?? added.first(where: { $0.s == token })?.id
    }

    private static func tokenString(_ raw: Any?) -> String? {
        if let s = raw as? String, !s.isEmpty { return s }
        if let d = raw as? [String: Any], let c = d["content"] as? String, !c.isEmpty { return c }
        return nil
    }

    private static func intArray(_ raw: Any?) -> [Int] {
        if let num = raw as? NSNumber, CFGetTypeID(num) != CFBooleanGetTypeID() {
            return [num.intValue]
        }
        if let arr = raw as? [Any] {
            return arr.compactMap { elem in
                guard let n = elem as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID() else { return nil }
                return n.intValue
            }
        }
        return []
    }

    public func resolveSpecialTokens(
        fallback: QWModelSpecialTokens,
        tokenizerConfigJSON: Data? = nil,
        specialTokensMapJSON: Data? = nil,
        generationConfigJSON: Data? = nil
    ) -> QWModelSpecialTokens {
        let tcObj = (tokenizerConfigJSON ?? storedTokenizerConfigJSON)
            .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
        let stmObj = (specialTokensMapJSON ?? storedSpecialTokensMapJSON)
            .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
        let genObj = (generationConfigJSON ?? storedGenerationConfigJSON)
            .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }

        let maxKnownVocab = max(vocab.values.max() ?? -1, added.map(\.id).max() ?? -1) + 1
        let upperBound = max(fallback.eosTokenId + 1, maxKnownVocab)
        func valid(_ id: Int?) -> Int? {
            guard let id, id >= 0, id < upperBound else { return nil }
            return id
        }

        let tcEosStr = Self.tokenString(tcObj?["eos_token"]) ?? Self.tokenString(stmObj?["eos_token"])
        let tcEosId = valid(tcEosStr.flatMap { tokenId(for: $0) })
        let genEosIds = Self.intArray(genObj?["eos_token_id"]).compactMap(valid)

        let imStart = valid(
            tokenId(for: "<|im_start|>")
            ?? tokenId(for: "<|ernie:im_start|>")
            ?? tokenId(for: "<|maxwell_start|>")
            ?? tokenId(for: "<role>")
            ?? tokenId(for: "<|role_sep|>\n")
        ) ?? fallback.imStartTokenId
        let imEnd = valid(
            tokenId(for: "<|im_end|>")
            ?? tokenId(for: "<|ernie:im_end|>")
            ?? tokenId(for: "<|maxwell_end|>")
            ?? tokenId(for: "<|role_end|>")
            ?? tokenId(for: "<|message_sep|>\n\n")
            ?? tcEosId
        ) ?? fallback.imEndTokenId
        let think = valid(
            tokenId(for: "<think>")
            ?? tokenId(for: "<ernie:think>")
        ) ?? fallback.thinkTokenId
        let endThink = valid(
            tokenId(for: "</think>")
            ?? tokenId(for: "</ernie:think>")
        ) ?? fallback.endThinkTokenId

        var extra = fallback.extraStopTokenIds
        func appendStop(_ id: Int?) {
            guard let id = valid(id), !extra.contains(id) else { return }
            extra.append(id)
        }
        for s in ["<|im_end|>", "<|ernie:im_end|>", "<|maxwell_end|>", "<|role_end|>",
                  "<|message_sep|>\n\n", "<|ERNIE_endofprompt|>", "<|content_model_end_sampling|>",
                  "<|endoftext|>", "<|eot_id|>", "</s>", "<|end_of_text|>"] {
            appendStop(tokenId(for: s))
        }
        appendStop(tcEosId)
        for gid in genEosIds { appendStop(gid) }

        let resolvedEos = genEosIds.first ?? tcEosId ?? fallback.eosTokenId
        let hasChatIds = fallback.usesChatTemplateIds
            || tokenId(for: "<|im_end|>") != nil
            || tokenId(for: "<|ernie:im_end|>") != nil
            || tokenId(for: "<|maxwell_end|>") != nil
            || tokenId(for: "<|message_sep|>\n\n") != nil
        return QWModelSpecialTokens(
            eosTokenId: resolvedEos,
            imStartTokenId: imStart,
            imEndTokenId: imEnd,
            thinkTokenId: think,
            endThinkTokenId: endThink,
            extraStopTokenIds: extra,
            usesChatTemplateIds: hasChatIds
        )
    }
}
