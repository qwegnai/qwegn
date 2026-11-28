// QWLinearWeight.swift — Unified linear weight handle for BF16 banks and quantized views.
import QWAlloc

/// Linear projection weight stored either as a dense BF16 bank or a quantized block view.
public enum QWLinearWeight: @unchecked Sendable {
    case bf16(QWGemmFP.BF16Bank)
    case quant(QWWeightView)

    public var n: Int {
        switch self {
        case .bf16(let b): return b.n
        case .quant(let v): return v.n
        }
    }

    public var k: Int {
        switch self {
        case .bf16(let b): return b.k
        case .quant(let v): return v.k
        }
    }

    public var isQuant: Bool {
        if case .quant = self { return true }
        return false
    }

    public var needsRotation: Bool {
        if case .quant(let v) = self { return v.containerKind.isRotated }
        return false
    }

    public var residentBytes: Int {
        switch self {
        case .bf16(let b): return max(0, b.n) * max(0, b.k) * 2
        case .quant(let v): return v.requiredBytes
        }
    }
}
