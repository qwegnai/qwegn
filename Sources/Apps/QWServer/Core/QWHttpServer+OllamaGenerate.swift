import Foundation
import Network

/// The Ollama text-generation answer: the fields Ollama clients read.
/// Durations are in nanoseconds, as Ollama reports them. A field the
/// engine did not measure is left out, never filled in.
struct OllamaGenerateResponse: Codable {
    let model: String
    let createdAt: String
    let response: String
    let done: Bool
    let doneReason: String?
    let promptEvalCount: Int
    let evalCount: Int
    let totalDuration: Int64
    let promptEvalDuration: Int64
    let evalDuration: Int64

    enum CodingKeys: String, CodingKey {
        case model
        case createdAt = "created_at"
        case response
        case done
        case doneReason = "done_reason"
        case promptEvalCount = "prompt_eval_count"
        case evalCount = "eval_count"
        case totalDuration = "total_duration"
        case promptEvalDuration = "prompt_eval_duration"
        case evalDuration = "eval_duration"
    }
}

extension QWHttpServer {

    /// POST /api/generate — the Ollama completion endpoint. The prompt
    /// goes through the same engine path as the chat routes, as one
    /// user message, and the answer carries the engine's real counts.
    func handleOllamaGenerate(connection: NWConnection, body: Data) async throws {
        guard let parsed = (try? JSONSerialization.jsonObject(with: body)) as? [String: Any] else {
            sendJsonResponse(connection, status: 400, object: ["error": "invalid request: malformed JSON body"])
            return
        }
        guard let prompt = parsed["prompt"] as? String else {
            sendJsonResponse(connection, status: 400, object: ["error": "prompt: required field is missing"])
            return
        }

        let requested = (parsed["model"] as? String ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard let replyModel = resolveReplyModelName(requested: requested, strictNotFound: true) else {
            sendJsonResponse(connection, status: 404, object: ["error": "model \"\(requested)\" not found, try pulling it first"])
            return
        }

        // Ollama's options carry the generation limit and temperature.
        let options = parsed["options"] as? [String: Any] ?? [:]
        var payload: [String: Any] = [
            "messages": [["role": "user", "content": prompt]],
            "stream": false,
        ]
        if let n = (options["num_predict"] as? Int) ?? (options["num_predict"] as? NSNumber)?.intValue, n > 0 {
            payload["max_tokens"] = n
        }
        if let t = options["temperature"] as? Double { payload["temperature"] = t }
        if let p = options["top_p"] as? Double { payload["top_p"] = p }
        if let k = options["top_k"] as? Int { payload["top_k"] = k }
        if let s = options["seed"] as? Int { payload["seed"] = s }

        let req: ChatCompletionRequest
        do {
            req = try JSONDecoder().decode(ChatCompletionRequest.self,
                                           from: JSONSerialization.data(withJSONObject: payload))
        } catch {
            sendJsonResponse(connection, status: 400, object: ["error": "invalid request: \(error.localizedDescription)"])
            return
        }

        let started = Date()
        let result = try await actor.handleChatCompletion(request: req, onToken: nil)
        let elapsedNs = Int64(Date().timeIntervalSince(started) * 1e9)

        // The engine's own prefill and decode timings. A timing the
        // engine did not measure is reported as 0, never estimated.
        let resp = OllamaGenerateResponse(
            model: replyModel,
            createdAt: OllamaChatResponse.now(),
            response: result.content,
            done: true,
            doneReason: result.truncated ? "length" : "stop",
            promptEvalCount: result.promptTokens,
            evalCount: result.completionTokens,
            totalDuration: elapsedNs,
            promptEvalDuration: Int64((result.prefillSec ?? 0) * 1e9),
            evalDuration: Int64((result.decodeSec ?? 0) * 1e9)
        )
        sendJsonResponse(connection, status: 200, object: resp)
    }
}
