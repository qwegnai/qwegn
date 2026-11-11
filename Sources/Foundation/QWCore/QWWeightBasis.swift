// QWWeightBasis.swift — the basis a model's weights are quantised in.
//
// The engine's own weights are quantised ROTATED; an MLX affine
// checkpoint's planes are not. The activations must be in the same basis as
// the weights they meet, so the basis is a MODEL property, never a
// per-tensor one. The design, and why the basis is a build-time switch, are
// in specs/02-qwhal.md.

public enum QWWeightBasis: String, Sendable, Hashable, Codable {
    /// `Block_HQ4_G32`: the weights carry the rotation, and the activations
    /// are rotated to match.
    case rotated
    /// The size-preserving affine formats (`Block_AQ{b}_G{G}`): the weights
    /// are quantised as they are, and the activations stay in the standard
    /// basis.
    case standard

    /// Whether this basis rotates the activations. The Metal side reads the
    /// same fact from the metallib it was built with, and the two must never
    /// disagree: one says what the weights are, the other what the kernels do.
    public var rotatesActivations: Bool { self == .rotated }

    /// The metallib that carries this basis: the basis is a build-time
    /// property of the library, not a runtime switch (specs/02-qwhal.md).
    public var metallibFileName: String {
        switch self {
        case .rotated: return "kernel.metallib"
        case .standard: return "kernel_std.metallib"
        }
    }
}

/// The container the engine holds a model's weights in. The two differ in
/// more than their size: they hold their weights in different BASES, and
/// the activations have to be in the same basis as the weights they meet.
public enum QWWeightFormat: Sendable, Hashable {
    /// The engine's own container: `Block_HQ4_G32`, twenty bytes per group
    /// of thirty-two — five bytes per weight — with the weights in the
    /// ROTATED basis. A checkpoint the engine inflates lands here: the
    /// inflation re-quantises the dequantised weights, and the quantiser
    /// rotates them.
    case hq4G32
    /// Rotated `Block_AQ{b}_G{G}` or `SHVQ{b}_G{G}` container in a `.qwp` pack when the dense or
    /// expert `(bits, group)` pair or container layout differs from the incumbent `(4, 32)`.
    case hqBlock(bits: Int, groupSize: Int, expertBits: Int? = nil, expertGroupSize: Int? = nil,
                 denseIncumbent: Bool = false, expertIncumbent: Bool = false,
                 expertDownBits: Int? = nil, expertDownGroupSize: Int? = nil,
                 expertDownIncumbent: Bool? = nil,
                 denseSHVQ: Bool = false, expertSHVQ: Bool = false,
                 expertDownSHVQ: Bool? = nil)
    /// The size-preserving affine container: `Block_AQ{b}_G{G}`, the
    /// checkpoint's own bytes, with the weights in the STANDARD basis. An
    /// MLX affine checkpoint's planes carry no rotation, so a checkpoint
    /// kept in its own bytes is a model whose activations must NOT be
    /// rotated to match.
    case nativeAffine(bits: Int, groupSize: Int)

    /// The basis the weights in this container are quantised in, and so the
    /// basis the activations must be in.
    public var basis: QWWeightBasis {
        switch self {
        case .hq4G32, .hqBlock: return .rotated
        case .nativeAffine: return .standard
        }
    }

    /// The bits one weight takes, header included.
    public var bits: Int {
        switch self {
        case .hq4G32: return 4
        case .hqBlock(let bits, _, _, _, _, _, _, _, _, _, _, _): return bits
        case .nativeAffine(let bits, _): return bits
        }
    }

    /// The group one scale and one bias cover.
    public var groupSize: Int {
        switch self {
        case .hq4G32: return 32
        case .hqBlock(_, let groupSize, _, _, _, _, _, _, _, _, _, _): return groupSize
        case .nativeAffine(_, let groupSize): return groupSize
        }
    }

    /// The bits one routed expert gate/up weight takes.
    public var expertBits: Int {
        switch self {
        case .hqBlock(let bits, _, let eb, _, _, _, _, _, _, _, _, _): return eb ?? bits
        default: return bits
        }
    }

    /// The group one scale and one bias cover in routed expert gate/up weights.
    public var expertGroupSize: Int {
        switch self {
        case .hqBlock(_, let groupSize, _, let eg, _, _, _, _, _, _, _, _): return eg ?? groupSize
        default: return groupSize
        }
    }

    /// The bits one routed expert down_proj weight takes.
    public var expertDownBits: Int {
        switch self {
        case .hqBlock(let bits, _, let eb, _, _, _, let edb, _, _, _, _, _): return edb ?? eb ?? bits
        default: return expertBits
        }
    }

    /// The group one scale and one bias cover in routed expert down_proj weights.
    public var expertDownGroupSize: Int {
        switch self {
        case .hqBlock(_, let groupSize, _, let eg, _, _, _, let edg, _, _, _, _): return edg ?? eg ?? groupSize
        default: return expertGroupSize
        }
    }

    /// Whether dense 2D weights use the incumbent `Block_HQ4_G32` container.
    public var denseIsIncumbentContainer: Bool {
        switch self {
        case .hq4G32: return true
        case .hqBlock(_, _, _, _, let denseIncumbent, _, _, _, _, _, _, _): return denseIncumbent
        case .nativeAffine: return false
        }
    }

    /// Whether routed expert gate/up weights use the incumbent `Block_HQ4_G32` container.
    public var expertIsIncumbentContainer: Bool {
        switch self {
        case .hq4G32: return true
        case .hqBlock(_, _, _, _, _, let expertIncumbent, _, _, _, _, _, _): return expertIncumbent
        case .nativeAffine: return false
        }
    }

    /// Whether routed expert down_proj weights use the incumbent `Block_HQ4_G32` container.
    public var expertDownIsIncumbentContainer: Bool {
        switch self {
        case .hq4G32: return true
        case .hqBlock(_, _, _, _, _, let expertIncumbent, _, _, let edInc, _, _, _): return edInc ?? expertIncumbent
        case .nativeAffine: return false
        }
    }

    /// Whether dense 2D weights use `SHVQ{2,3,4}_G{32,64,128,256}` blocks.
    public var denseIsSHVQ: Bool {
        if case .hqBlock(_, _, _, _, _, _, _, _, _, let ds, _, _) = self { return ds }
        return false
    }

    /// Whether routed expert gate/up weights use `SHVQ{2,3,4}_G{32,64,128,256}` blocks.
    public var expertIsSHVQ: Bool {
        if case .hqBlock(_, _, _, _, _, _, _, _, _, _, let es, _) = self { return es }
        return false
    }

    /// Whether routed expert down_proj weights use `SHVQ{2,3,4}_G{32,64,128,256}` blocks.
    public var expertDownIsSHVQ: Bool {
        if case .hqBlock(_, _, _, _, _, _, _, _, _, _, let es, let eds) = self { return eds ?? es }
        return false
    }

    /// Whether this container is the size-preserving family.
    public var isNativeAffine: Bool {
        if case .nativeAffine = self { return true }
        return false
    }

    /// Whether dense 2D weights use `qwg_*_aq*` / `qwg_*_shvq*` kernels instead of `qwg_*_q4g_*`.
    public var usesDenseAQKernels: Bool {
        !denseIsIncumbentContainer
    }

    /// The container's name: the incumbent's `HQ4_G32`, or the family's.
    public var name: String {
        switch self {
        case .hq4G32: return "HQ4_G32"
        case .hqBlock(let bits, let groupSize, let eb, let eg, _, _, let edb, let edg, _,
                      let ds, let es, let eds):
            let expB = eb ?? bits, expG = eg ?? groupSize
            let downB = edb ?? expB, downG = edg ?? expG
            let downS = eds ?? es
            let dTag = ds ? "SHVQ\(bits)_G\(groupSize)" : "HQ\(bits)_G\(groupSize)"
            let eTag = es ? "SHVQ\(expB)_G\(expG)" : "HQ\(expB)_G\(expG)"
            let dnTag = downS ? "SHVQ\(downB)_G\(downG)" : "HQ\(downB)_G\(downG)"
            if dnTag != eTag {
                return "\(dTag)+Exp\(eTag)/\(dnTag)"
            }
            if eTag != dTag {
                return "\(dTag)+Exp\(eTag)"
            }
            return dTag
        case .nativeAffine(let bits, let groupSize): return "AQ\(bits)_G\(groupSize)"
        }
    }

    /// The bytes one row of `k` weights takes in this container.
    public func rowBytes(k: Int) -> Int {
        switch self {
        case .hq4G32:
            return (k / 32) * 20
        case .hqBlock(let bits, let groupSize, _, _, _, _, _, _, _, let ds, _, _):
            let hdr = ds ? 2 : 4
            return (k / groupSize) * (groupSize * bits / 8 + hdr)
        case .nativeAffine(let bits, let groupSize):
            return (k / groupSize) * (groupSize * bits / 8 + 4)
        }
    }

    /// The bytes one row of `k` routed expert gate/up weights takes in this container.
    public func expertRowBytes(k: Int) -> Int {
        let b = expertBits
        let g = expertGroupSize
        let hdr = expertIsSHVQ ? 2 : 4
        return (k / g) * (g * b / 8 + hdr)
    }

    /// The bytes one row of `k` routed expert down_proj weights takes in this container.
    public func expertDownRowBytes(k: Int) -> Int {
        let b = expertDownBits
        let g = expertDownGroupSize
        let hdr = expertDownIsSHVQ ? 2 : 4
        return (k / g) * (g * b / 8 + hdr)
    }
}
