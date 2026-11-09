import Foundation

/// The MTP execution backend (LighteningMTP plan, stage L1). `standard` is
/// the shipping path and the default whenever drafts are requested;
/// `dflash2` is the block-diffusion drafter: its own pack, up to seven.
public enum QWMTPBackend: String, CaseIterable, Sendable, Codable {
    case standard
    case variational
    case dflash2

    /// Parse a CLI/API name. Case-sensitive: only the exact lowercase
    /// product names are accepted. Unknown names throw — never silently
    /// fall back to Standard (that would mislabel the run).
    public static func resolve(name: String) throws -> QWMTPBackend {
        switch name {
        case "standard", "mtp", "hybrid", "ngram": return .standard
        case "variational", "relaxed", "plm": return .variational
        case "dflash2", "dflash", "dspine": return .dflash2
        default: throw QWEngineError.badMTPBackend(name)
        }
    }

    /// The effective kernel route for this backend at the current stage.
    public var route: String {
        switch self {
        case .standard: return "standard"
        case .variational: return "variational:relaxed-argmax-quadratic-v1"
        case .dflash2: return "dflash2:block-diffusion-5L-w2048-conv2-top16"
        }
    }

    /// The qualification profile recorded in verbose output and provenance.
    public var qualificationProfile: String {
        switch self {
        case .standard: return "standard-shipping"
        case .variational: return "variational-relaxed-acceptance"
        case .dflash2: return "dflash2-block-speculative"
        }
    }

    /// The backend/drafts contract (LighteningMTP L1). `nil` means "no
    /// explicit backend selection": with drafts 0 this is N0, with positive
    /// drafts it resolves to Standard.
    public static func resolveSelection(_ backend: QWMTPBackend?, drafts: Int) throws -> QWMTPBackend {
        let maxDrafts = backend == .dflash2 ? 15 : 5
        guard (0...maxDrafts).contains(drafts) else { throw QWEngineError.badMTPDrafts(drafts) }
        if let backend {
            guard drafts > 0 else {
                throw QWEngineError.state(
                    "explicit MTP backend '\(backend.rawValue)' requires positive --mtp-drafts (1...\(maxDrafts))")
            }
            return backend
        }
        return .standard
    }
}
