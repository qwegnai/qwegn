// PlaneLayout.swift — single byte-math authority for verbatim pack planes.
// Engines classify (which layout) and transform (dtypes); sizes and offsets
// for the mover come from here only.
import QWQuant

public enum PlaneLayout: Sendable, Hashable {
    case raw
    case hq4

    /// Trailer kind of a verbatim plane; nil when absent.
    public static func resolve(_ e: QWQTensorEntry) -> PlaneLayout? {
        if e.rawOff != nil { return .raw }
        if e.hq4Off != nil { return .hq4 }
        return nil
    }

    public func offset(_ e: QWQTensorEntry) -> Int? {
        switch self {
        case .raw: return e.rawOff
        case .hq4: return e.hq4Off
        }
    }

    public func byteCount(_ e: QWQTensorEntry) -> Int? {
        guard !e.shape.isEmpty else { return nil }
        switch self {
        case .raw:
            return e.shape.reduce(1, *) * 2
        case .hq4:
            if let experts = e.experts, let per = e.eHq4, e.shape.count == 3 {
                return experts * per
            }
            guard e.shape.count >= 2 else { return nil }
            let n = e.shape[0]
            let k = e.shape.dropFirst().reduce(1, *)
            let grp = e.hqGroup
            guard k % grp == 0 else { return nil }
            return n * (k / grp) * e.hqBlockBytes
        }
    }
}
