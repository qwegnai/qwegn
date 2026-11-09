import Foundation
import CoreFoundation
import QWCore
import QWQuant
import QWToken

public struct QWModelGenerationDefaults: Sendable, Equatable {
    public let temperature: Double?
    public let doSample: Bool?
    public let topP: Double?
    public let topK: Int?
    public let repetitionPenalty: Double?
    public let maxNewTokens: Int?
    public let maxLength: Int?

    public var effectiveTemperature: Double? {
        if doSample == false { return 0 }
        return temperature
    }
}

public struct QWModelRuntimeAssets: Sendable {
    public let assetsURL: URL
    public let descriptor: QWModelDescriptor
    public let tokenizerJSON: Data
    public let chatTemplate: QWChatTemplate
    public let generationDefaults: QWModelGenerationDefaults

    public enum Error: Swift.Error, CustomStringConvertible {
        case badJSON(String)
        case badGenerationValue(String)
        case unsupportedConfig(String)
        case templateMissing(String)
        case ambiguousTemplate(String)
        case tokenizerMissing(String)

        public var description: String {
            switch self {
            case .badJSON(let path): return "model assets: invalid JSON at '\(path)'"
            case .badGenerationValue(let field):
                return "model assets: unsupported generation_config.json value for '\(field)'"
            case .unsupportedConfig(let reason): return "model assets: \(reason)"
            case .templateMissing(let path):
                return "model assets: no chat template found in '\(path)' or chat_template.jinja"
            case .ambiguousTemplate(let path):
                return "model assets: multiple named chat templates in '\(path)' have no 'default' template"
            case .tokenizerMissing(let path): return "model assets: tokenizer.json missing at '\(path)'"
            }
        }
    }

    public static func load(packPath: String,
                            assetsOverride: URL? = nil) throws -> QWModelRuntimeAssets {
        let location = try QWModelLocation(url: URL(fileURLWithPath: packPath),
                                           assetsOverride: assetsOverride)
        let assets = location.assets
        let tokenizerURL = assets.appendingPathComponent("tokenizer.json")
        guard let tokenizerJSON = FileManager.default.contents(atPath: tokenizerURL.path) else {
            throw Error.tokenizerMissing(tokenizerURL.path)
        }

        let descriptor = try validateModelConfig(at: assets.appendingPathComponent("config.json"))
        let (templateSource, tokenizerContext) = try loadChatTemplate(assets: assets)
        let chatTemplate = try QWChatTemplate(source: templateSource, tokenizerContext: tokenizerContext)

        let generationURL = assets.appendingPathComponent("generation_config.json")
        let generationDefaults: QWModelGenerationDefaults
        if FileManager.default.fileExists(atPath: generationURL.path) {
            let object = try jsonObject(at: generationURL)
            generationDefaults = try parseGenerationDefaults(object, path: generationURL.path)
        } else {
            generationDefaults = QWModelGenerationDefaults(
                temperature: nil, doSample: nil, topP: nil, topK: nil,
                repetitionPenalty: nil, maxNewTokens: nil, maxLength: nil)
        }

        return QWModelRuntimeAssets(assetsURL: assets, descriptor: descriptor,
                                   tokenizerJSON: tokenizerJSON,
                                   chatTemplate: chatTemplate,
                                   generationDefaults: generationDefaults)
    }

    private static func validateModelConfig(at url: URL) throws -> QWModelDescriptor {
        guard FileManager.default.fileExists(atPath: url.path) else {
            return .qwen38_27b
        }
        let descriptor = try QWModelDescriptor.decode(configJSON: Data(contentsOf: url))
        // Model-type admission only. Geometry identity stays bound by the
        // pack trailer guard (QWPack); digesting an HF envelope config against
        // frozen constants breaks on text_config.model_type aliasing.
        guard QWModelFamilyRegistry.shared.isSupported(modelType: descriptor.modelType) else {
            throw Error.unsupportedConfig("config.json model_type '\(descriptor.modelType)' is not supported")
        }
        return descriptor
    }

    private static func loadChatTemplate(assets: URL) throws -> (String, [String: QWJson]) {
        let tokenizerConfigURL = assets.appendingPathComponent("tokenizer_config.json")
        let templateURL = assets.appendingPathComponent("chat_template.jinja")
        let tokenizerTemplate: String?
        var tokenizerContext: [String: QWJson] = [:]
        if FileManager.default.fileExists(atPath: tokenizerConfigURL.path) {
            let object = try jsonObject(at: tokenizerConfigURL)
            tokenizerTemplate = try chatTemplate(from: object["chat_template"],
                                                 path: tokenizerConfigURL.path)
            for name in ["bos_token", "eos_token", "unk_token", "pad_token", "sep_token",
                         "cls_token", "mask_token"] {
                if let token = tokenizerToken(from: object[name]) {
                    tokenizerContext[name] = .string(token)
                }
            }
        } else {
            tokenizerTemplate = nil
        }

        let bundledTemplate: String?
        if FileManager.default.fileExists(atPath: templateURL.path) {
            do {
                bundledTemplate = try String(contentsOf: templateURL, encoding: .utf8)
            } catch {
                throw Error.unsupportedConfig("chat template at '\(templateURL.path)' is not valid UTF-8")
            }
        } else {
            bundledTemplate = nil
        }

        if let tokenizerTemplate, let bundledTemplate, tokenizerTemplate != bundledTemplate {
            throw Error.unsupportedConfig(
                "chat_template in '\(tokenizerConfigURL.path)' disagrees with '\(templateURL.path)'")
        }
        guard let source = tokenizerTemplate ?? bundledTemplate else {
            throw Error.templateMissing(tokenizerConfigURL.path)
        }
        return (source, tokenizerContext)
    }

    private static func tokenizerToken(from raw: Any?) -> String? {
        if let token = raw as? String { return token }
        if let token = raw as? [String: Any] { return token["content"] as? String }
        return nil
    }

    private static func chatTemplate(from raw: Any?, path: String) throws -> String? {
        guard let raw, !(raw is NSNull) else { return nil }
        if let value = raw as? String { return value }
        guard let entries = raw as? [[String: Any]], !entries.isEmpty else {
            throw Error.badJSON(path)
        }
        let selected: [String: Any]
        if let defaultEntry = entries.first(where: { ($0["name"] as? String) == "default" }) {
            selected = defaultEntry
        } else if entries.count == 1 {
            selected = entries[0]
        } else {
            throw Error.ambiguousTemplate(path)
        }
        guard let value = selected["template"] as? String else {
            throw Error.badJSON(path)
        }
        return value
    }

    private static func jsonObject(at url: URL) throws -> [String: Any] {
        do {
            guard let object = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
                    as? [String: Any] else {
                throw Error.badJSON(url.path)
            }
            return object
        } catch let error as Error {
            throw error
        } catch {
            throw Error.badJSON(url.path)
        }
    }

    private static func parseGenerationDefaults(_ object: [String: Any],
                                                path: String) throws -> QWModelGenerationDefaults {
        func number<T: LosslessStringConvertible>(_ key: String, as: T.Type) throws -> T? {
            guard let raw = object[key], !(raw is NSNull) else { return nil }
            guard let number = raw as? NSNumber,
                  CFGetTypeID(number) != CFBooleanGetTypeID(),
                  let value = T(String(describing: number)) else {
                throw Error.badGenerationValue("\(path):\(key)")
            }
            return value
        }
        func bool(_ key: String) throws -> Bool? {
            guard let raw = object[key], !(raw is NSNull) else { return nil }
            guard let value = raw as? NSNumber,
                  CFGetTypeID(value) == CFBooleanGetTypeID() else {
                throw Error.badGenerationValue("\(path):\(key)")
            }
            return value.boolValue
        }

        let temperature = try number("temperature", as: Double.self)
        let topP = try number("top_p", as: Double.self)
        let topK = try number("top_k", as: Int.self)
        let repetitionPenalty = try number("repetition_penalty", as: Double.self)
        let maxNewTokens = try number("max_new_tokens", as: Int.self)
        let maxLength = try number("max_length", as: Int.self)
        let doSample = try bool("do_sample")

        if let temperature, !temperature.isFinite || temperature < 0 {
            throw Error.badGenerationValue("\(path):temperature")
        }
        if let topP, !topP.isFinite || topP <= 0 || topP > 1 {
            throw Error.badGenerationValue("\(path):top_p")
        }
        if let topK, topK < 0 { throw Error.badGenerationValue("\(path):top_k") }
        if let repetitionPenalty,
           !repetitionPenalty.isFinite || repetitionPenalty <= 0 {
            throw Error.badGenerationValue("\(path):repetition_penalty")
        }
        if let maxNewTokens, maxNewTokens < 0 {
            throw Error.badGenerationValue("\(path):max_new_tokens")
        }
        if let maxLength, maxLength <= 0 {
            throw Error.badGenerationValue("\(path):max_length")
        }
        return QWModelGenerationDefaults(
            temperature: temperature, doSample: doSample, topP: topP, topK: topK,
            repetitionPenalty: repetitionPenalty, maxNewTokens: maxNewTokens,
            maxLength: maxLength)
    }
}
