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
            var elems = 1
            for dim in e.shape {
                guard dim >= 0 else { return nil }
                let (m, ov) = elems.multipliedReportingOverflow(by: dim)
                if ov { return nil }
                elems = m
            }
            let (bytes, ov) = elems.multipliedReportingOverflow(by: e.rawElemBytes)
            return ov ? nil : bytes
        case .hq4:
            if let experts = e.experts, let per = e.eHq4, e.shape.count == 3 {
                guard experts >= 0, per >= 0 else { return nil }
                let (bytes, ov) = experts.multipliedReportingOverflow(by: per)
                return ov ? nil : bytes
            }
            guard e.shape.count >= 2 else { return nil }
            let n = e.shape[0]
            guard n >= 0 else { return nil }
            var k = 1
            for dim in e.shape.dropFirst() {
                guard dim >= 0 else { return nil }
                let (m, ov) = k.multipliedReportingOverflow(by: dim)
                if ov { return nil }
                k = m
            }
            let grp = e.hqGroup
            guard grp > 0, k % grp == 0 else { return nil }
            if !e.isIncumbentHQ4 {
                let valid = e.isSHVQ
                    ? QWSHVQBlock.isSupported(bits: e.hqBits, group: grp)
                    : QWAQBlock.isBlockSupported(bits: e.hqBits, group: grp)
                guard valid else { return nil }
            }
            let (ng, ov1) = n.multipliedReportingOverflow(by: k / grp)
            guard !ov1 else { return nil }
            let (bytes, ov2) = ng.multipliedReportingOverflow(by: e.hqBlockBytes)
            return ov2 ? nil : bytes
        }
    }
}
