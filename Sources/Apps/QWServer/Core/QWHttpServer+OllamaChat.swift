import Foundation
import Network

/// The facts a model carries on disk: the bytes it occupies and when
/// those bytes changed. Everything the compatible endpoints publish
/// about a model is read from the file system, never invented.
public enum QWModelDiskStats {
    /// The bytes the model occupies on disk: the pack file's size, or
    /// the sum of a model directory's files. nil when the path does
    /// not exist — no path, no size.
    public static func sizeBytes(of path: String?) -> Int64? {
        guard let path, !path.isEmpty else { return nil }
        let fm = FileManager.default
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: path, isDirectory: &isDir) else { return nil }
        if isDir.boolValue {
            var total: Int64 = 0
            if let en = fm.enumerator(atPath: path) {
                for case let rel as String in en {
                    let attrs = try? fm.attributesOfItem(atPath: (path as NSString).appendingPathComponent(rel))
                    if let n = attrs?[.size] as? NSNumber { total += n.int64Value }
                }
            }
            return total
        }
        let attrs = try? fm.attributesOfItem(atPath: path)
        return (attrs?[.size] as? NSNumber)?.int64Value
    }

    /// When the model's bytes last changed, as an RFC 3339 UTC instant.
    /// nil when there is no path to measure.
    public static func modifiedRFC3339(of path: String?) -> String? {
        guard let path, !path.isEmpty else { return nil }
        let attrs = try? FileManager.default.attributesOfItem(atPath: path)
        guard let date = attrs?[.modificationDate] as? Date else { return nil }
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = TimeZone(identifier: "UTC")
        f.dateFormat = "yyyy-MM-dd'T'HH:mm:ssZ"
        return f.string(from: date)
    }

    /// When the model's bytes last changed, as unix seconds. nil when
    /// there is no path to measure.
    public static func modifiedUnix(of path: String?) -> Int? {
        guard let path, !path.isEmpty else { return nil }
        let attrs = try? FileManager.default.attributesOfItem(atPath: path)
        guard let date = attrs?[.modificationDate] as? Date else { return nil }
        return Int(date.timeIntervalSince1970)
    }
}

/// The answer POST /api/chat gives: the Ollama shape, with the counts
/// the engine actually reported.
struct OllamaChatResponse: Codable {
    let model: String
    let createdAt: String
    let done: Bool
    let doneReason: String?
    let promptEvalCount: Int
    let evalCount: Int
    let message: OllamaChatMessage

    enum CodingKeys: String, CodingKey {
        case model
        case createdAt = "created_at"
        case done
        case doneReason = "done_reason"
        case promptEvalCount = "prompt_eval_count"
        case evalCount = "eval_count"
        case message
    }

    static func now() -> String {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f.string(from: Date())
    }
}

struct OllamaChatMessage: Codable {
    let role: String
    let content: String
}

extension QWHttpServer {

    /// POST /api/chat — the Ollama chat endpoint. The engine answers
    /// the same way it answers the OpenAI-compatible route; only the
    /// envelope differs.
    func handleOllamaChat(connection: NWConnection, body: Data) async throws {
        // Ollama clients carry the sampler settings in "options";
        // map them onto the request fields the engine reads.
        let req: ChatCompletionRequest
        do {
            var payload: [String: Any] = (try JSONSerialization.jsonObject(with: body)) as? [String: Any] ?? [:]
            if let opts = payload["options"] as? [String: Any] {
                for key in ["temperature", "top_p", "top_k", "seed"] {
                    if let v = opts[key] { payload[key] = v }
                }
                if let n = (opts["num_predict"] as? Int) ?? (opts["num_predict"] as? NSNumber)?.intValue, n > 0 {
                    payload["max_tokens"] = n
                }
            }
            req = try JSONDecoder().decode(ChatCompletionRequest.self,
                                           from: JSONSerialization.data(withJSONObject: payload))
        } catch {
            // Ollama's error shape is {"error": "..."}.
            sendJsonResponse(connection, status: 400, object: ["error": "invalid request: malformed JSON body"])
            return
        }

        let requested = (req.model ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard let replyModel = resolveReplyModelName(requested: requested, strictNotFound: true) else {
            sendJsonResponse(connection, status: 404, object: ["error": "model \"\(requested)\" not found, try pulling it first"])
            return
        }

        let created = OllamaChatResponse.now()

        if req.isStreaming {
            sendNDJSONHeaders(connection)

            let result = try await actor.handleChatCompletion(request: req) { piece in
                let chunk = OllamaChatResponse(model: replyModel, createdAt: created, done: false, doneReason: nil,
                                               promptEvalCount: 0, evalCount: 0,
                                               message: OllamaChatMessage(role: "assistant", content: piece))
                if let data = try? JSONEncoder().encode(chunk) {
                    connection.send(content: data + Data("\n".utf8), isComplete: false, completion: .idempotent)
                }
            }

            let done = OllamaChatResponse(model: replyModel, createdAt: created, done: true,
                                          doneReason: result.truncated ? "length" : "stop",
                                          promptEvalCount: result.promptTokens, evalCount: result.completionTokens,
                                          message: OllamaChatMessage(role: "assistant", content: ""))
            if let data = try? JSONEncoder().encode(done) {
                connection.send(content: data + Data("\n".utf8), isComplete: true, completion: .contentProcessed({ _ in
                    connection.cancel()
                }))
            }
        } else {
            let result = try await actor.handleChatCompletion(request: req, onToken: nil)
            let resp = OllamaChatResponse(model: replyModel, createdAt: created, done: true,
                                          doneReason: result.truncated ? "length" : "stop",
                                          promptEvalCount: result.promptTokens, evalCount: result.completionTokens,
                                          message: OllamaChatMessage(role: "assistant", content: result.content))
            sendJsonResponse(connection, status: 200, object: resp)
        }
    }
}
