import Foundation

/// The Ollama-compatible payloads. They live apart from the OpenAI-
/// compatible schemas so each file stays under the repository's
/// file-size rule.
public struct OllamaModelTag: Codable, Sendable {
    public let name: String
    public let model: String
    public let modifiedAt: String?
    public let size: Int64?

    enum CodingKeys: String, CodingKey {
        case name
        case model
        case modifiedAt = "modified_at"
        case size
    }

    /// The measured facts of one model on disk. A model that has no
    /// path on disk carries no size: an absent field is honest, a
    /// number the server made up is not.
    public init(name: String, model: String, modifiedAt: String?, size: Int64?) {
        self.name = name
        self.model = model
        self.modifiedAt = modifiedAt
        self.size = size
    }
}

public struct OllamaTagsResponse: Codable, Sendable {
    public let models: [OllamaModelTag]

    public init(models: [OllamaModelTag]) {
        self.models = models
    }
}

public struct OllamaPsResponse: Codable, Sendable {
    public struct RunningModel: Codable, Sendable {
        public let name: String
        public let model: String
        public let size: Int64?
        public let expiresAt: String

        enum CodingKeys: String, CodingKey {
            case name
            case model
            case size
            case expiresAt = "expires_at"
        }

        public init(name: String, model: String, size: Int64?, expiresAt: String) {
            self.name = name
            self.model = model
            self.size = size
            self.expiresAt = expiresAt
        }
    }
    public let models: [RunningModel]

    public init(models: [RunningModel]) {
        self.models = models
    }
}
