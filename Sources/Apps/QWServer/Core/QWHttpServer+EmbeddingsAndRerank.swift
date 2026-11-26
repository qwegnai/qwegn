// QWHttpServer+EmbeddingsAndRerank.swift — OpenAI POST /v1/embeddings and Cohere/Jina POST /v1/rerank handlers.

import Foundation
import Network
import QWCore
import QWDecision
import QWEngine
import QWToken

private struct QWEmbeddingsHTTPRequest: Decodable {
    let model: String?
    let input: QWJson
    let encodingFormat: String?
    let dimensions: Int?
    let pooling: String?
    let normalize: Bool?

    enum CodingKeys: String, CodingKey {
        case model, input, dimensions, pooling, normalize
        case encodingFormat = "encoding_format"
    }
}

private struct QWRerankHTTPRequest: Decodable {
    let model: String?
    let query: String
    let documents: [QWJson]
    let topN: Int?
    let returnDocuments: Bool?

    enum CodingKeys: String, CodingKey {
        case model, query, documents
        case topN = "top_n"
        case returnDocuments = "return_documents"
    }
}

extension QWHttpServer {
    func handleEmbeddingsRoute(connection: NWConnection, body: Data) async {
        do {
            let req = try QWServerRequestDecoder.decode(QWEmbeddingsHTTPRequest.self, from: body)
            let texts = try parseEmbeddingInputs(req.input)
            let format = (req.encodingFormat ?? "float").trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            guard format == "float" || format == "base64" else {
                throw QWDecisionError.invalidQuestion("encoding_format must be 'float' or 'base64'")
            }
            if let dims = req.dimensions, dims <= 0 {
                throw QWDecisionError.invalidQuestion("dimensions must be greater than 0")
            }
            let pooling: QWDecisionModelConfig.PoolingKind?
            if let rawPool = req.pooling?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(), !rawPool.isEmpty {
                switch rawPool {
                case "mean", "mean_non_pad": pooling = .meanNonPad
                case "last", "last_token": pooling = .lastToken
                case "first", "cls", "first_token": pooling = .firstToken
                default: throw QWDecisionError.invalidQuestion("unsupported pooling '\(rawPool)'")
                }
            } else {
                pooling = nil
            }

            let res = try await actor.embedTexts(
                texts,
                preferredModel: req.model,
                pooling: pooling,
                normalize: req.normalize ?? true,
                dimensions: req.dimensions
            )
            let dataItems: [[String: Any]] = res.embeddings.enumerated().map { idx, vec in
                let embVal: Any = (format == "base64")
                    ? QWEmbeddingMath.encodeBase64LittleEndian(vec)
                    : vec.map { Double($0) }
                return [
                    "object": "embedding",
                    "index": idx,
                    "embedding": embVal,
                ]
            }
            let payload: [String: Any] = [
                "object": "list",
                "data": dataItems,
                "model": res.modelName,
                "usage": [
                    "prompt_tokens": res.promptTokens,
                    "total_tokens": res.promptTokens,
                ],
            ]
            let data = try JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
            sendRawData(connection, status: 200, contentType: "application/json", data: data)
        } catch {
            sendDecisionError(connection, error: error)
        }
    }

    func handleRerankRoute(connection: NWConnection, body: Data) async {
        do {
            let req = try QWServerRequestDecoder.decode(QWRerankHTTPRequest.self, from: body)
            let query = req.query.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !query.isEmpty else {
                throw QWDecisionError.invalidQuestion("rerank requires a non-empty 'query'")
            }
            guard !req.documents.isEmpty else {
                throw QWDecisionError.invalidQuestion("rerank requires at least one document")
            }
            if let topN = req.topN, topN <= 0 {
                throw QWDecisionError.invalidQuestion("top_n must be greater than 0")
            }
            let docs = try req.documents.enumerated().map { idx, item -> String in
                if case .string(let s) = item {
                    let trimmed = s.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !trimmed.isEmpty else {
                        throw QWDecisionError.invalidQuestion("document at index \(idx) must not be empty")
                    }
                    return s
                }
                if case .object = item, case .string(let s) = item.get("text") {
                    let trimmed = s.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !trimmed.isEmpty else {
                        throw QWDecisionError.invalidQuestion("document at index \(idx) must not be empty")
                    }
                    return s
                }
                throw QWDecisionError.invalidQuestion("document at index \(idx) must be a string or {text} object")
            }

            let decReqs = docs.map { doc in
                QWDecisionRequest(
                    state: .text(doc),
                    questions: [
                        QWDecisionQuestion(
                            name: "relevant",
                            kind: .noul,
                            instructions: "Is this document relevant to the query: \(query)"
                        ),
                    ],
                    truncation: .autoState
                )
            }
            let envelopes = try await actor.decideBatch(decReqs, preferredModel: req.model)
            let totalPromptTokens = envelopes.reduce(0) { $0 + $1.response.usage.promptTokens }
            let modelName = envelopes.first?.modelName ?? req.model ?? actor.getModelName()
            let returnDocs = req.returnDocuments ?? false

            struct ScoredDoc {
                let index: Int
                let score: Double
                let text: String
            }
            var scored: [ScoredDoc] = []
            scored.reserveCapacity(docs.count)
            for (idx, env) in envelopes.enumerated() {
                let firstAns = env.response.answers.first
                let probs = firstAns?.probs ?? []
                let fallbackScore: Double = (firstAns?.noul == true) ? (firstAns?.confidence ?? 0.5) : 0.0
                let pYes = probs.first(where: { $0.key == "yes" })?.value ?? fallbackScore
                scored.append(ScoredDoc(index: idx, score: pYes, text: docs[idx]))
            }
            scored.sort {
                if $0.score != $1.score { return $0.score > $1.score }
                return $0.index < $1.index
            }
            let capped = req.topN.map { Array(scored.prefix($0)) } ?? scored
            let results: [[String: Any]] = capped.map { item in
                var entry: [String: Any] = [
                    "index": item.index,
                    "relevance_score": item.score,
                ]
                if returnDocs {
                    entry["document"] = ["text": item.text]
                }
                return entry
            }
            let payload: [String: Any] = [
                "id": "rerank_" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(16),
                "model": modelName,
                "results": results,
                "meta": [
                    "billed_units": ["search_units": 1],
                    "tokens": ["input_tokens": totalPromptTokens],
                ],
            ]
            let data = try JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
            sendRawData(connection, status: 200, contentType: "application/json", data: data)
        } catch {
            sendDecisionError(connection, error: error)
        }
    }

    private func parseEmbeddingInputs(_ input: QWJson) throws -> [String] {
        switch input {
        case .string(let s):
            guard !s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw QWDecisionError.invalidQuestion("embedding input must not be empty")
            }
            return [s]
        case .array(let arr):
            guard !arr.isEmpty else {
                throw QWDecisionError.invalidQuestion("embedding input array must not be empty")
            }
            return try arr.enumerated().map { idx, el in
                guard case .string(let s) = el, !s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                    throw QWDecisionError.invalidQuestion("embedding input[\(idx)] must be a non-empty string")
                }
                return s
            }
        default:
            throw QWDecisionError.invalidQuestion("embedding 'input' must be a string or array of strings")
        }
    }
}
