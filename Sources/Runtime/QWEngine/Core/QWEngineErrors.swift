// QWEngineErrors.swift — The engine's one error vocabulary.
//
// The dense driver, the Flash-Next module and the shared core all report
// with this type. A consumer that catches it holds an engine's error, not
// a family's: the error no longer says which engine served the request.

import Foundation

/// What the engine could not do. One type for every family.
public enum QWEngineError: Error, Equatable, CustomStringConvertible {
    case packMissing(String)
    case weightMissing(String, String)
    case posBeyondTable(Int)
    case badToken(Int)
    case busy
    case unloading
    case state(String)
    case badMTPDrafts(Int)
    case badMTPBackend(String)
    case badYaRN(String)
    case unsupportedKVMode(String)
    case unsupportedModel(String)
    case badConfig(String)

    public var description: String {
        switch self {
        case .packMissing(let p): return "QWEngine: pack missing: \(p)"
        case .weightMissing(let n, let w): return "QWEngine: \(n): \(w)"
        case .posBeyondTable(let p): return "QWEngine: position \(p) beyond rope table"
        case .badToken(let t): return "QWEngine: token id \(t) outside the model vocabulary"
        case .busy: return "QWEngine: single-flight engine — another sequence is computing"
        case .unloading: return "QWEngine: the model is being unloaded — the request was refused"
        case .state(let m): return "QWEngine: \(m)"
        case .badMTPDrafts(let d): return "QWEngine: invalid MTP drafts count \(d) (supported: 0...5, or 0...7 for dflash2)"
        case .badMTPBackend(let v): return "QWEngine: invalid MTP backend '\(v)' (supported: standard, variational, dflash2)"
        case .badYaRN(let m): return "QWEngine: QW_ROPE_YARN_FACTOR — \(m)"
        case .unsupportedKVMode(let m): return "QWEngine: unsupported KV mode '\(m)' (supported: f16, q8, q4)."
        case .unsupportedModel(let m): return "QWEngine: model '\(m)' has no engine adapter for this engine: a pack must not run under another family's geometry"
        case .badConfig(let m): return "QWEngine: \(m)"
        }
    }
}
