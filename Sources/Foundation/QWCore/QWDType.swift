// QWDType.swift — Tensor element dtypes with size and group semantics shared
// by the packer, allocator views, and kernel launchers.
import Foundation

public enum QWDType: String, Sendable, Hashable {
    case bf16, f16, f32, i32
    case q8_0
    case q4g          // HQ4_G32_M5 group-32 Hadamard + M5 magic-bias: 20 B / 32 params = 0.625 B/param
    case hq8_g32      // HQ8_G32_M5 symmetric Hadamard INT8 planar: 34 B / 32 params = 1.0625 B/param

    public static let hq4_g32: QWDType = .q4g

    public var bytesPerParam: Double {
        switch self {
        case .bf16, .f16: return 2
        case .f32, .i32: return 4
        case .q8_0: return 1 + 2.0 / 32.0         // byte + f16 scale per 32
        case .q4g: return 20.0 / 32.0             // 20 B Block_HQ4_G32 per 32 params = 0.625 B/param
        case .hq8_g32: return 34.0 / 32.0         // 32 B int8 + 2 B f16 scale per 32 params
        }
    }
    public var groupSize: Int {
        switch self { case .q8_0, .q4g, .hq8_g32: return 32; default: return 0 }
    }
}
