// QWMLP.swift — C14 encoder: SwiGLU FFN, four launches (2 group-sums + 2
// GEMVs), zero intermediate DRAM round-trips (h stays L2-hot inside one
// command buffer; spec-14 fusion doctrine). See qwg_mlp.metal for the
// kernel laws (batch geometry, dequant-once-dot-M, h-f16 chain contract).
//
// ENCODER NEVER ALLOCATES (C09 steady-state law): h (the [M][I] f16
// intermediate), y, and the two group-sum scratch tensors gs/hs are all
import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWGEMM

public enum QWMLPError: Error, CustomStringConvertible {
    case badShape(String)
    public var description: String { switch self { case .badShape(let s): return "QWMLPError badShape: \(s)" } }
}

/// Packed HQ4_G32_M5 weight plane for one projection (contiguous 20-byte Block_HQ4_G32),
/// `n` rows × `k` inner — layout-validated at init by QWWeightView rules.
public struct QWMLPWeight: Sendable {
    public let view: QWWeightView
    public init?(blocks: QWTensorValue, n: Int, k: Int, bits: Int = 4, groupSize: Int = 32,
                 isRotated: Bool, isIncumbentContainer: Bool? = nil, isSHVQ: Bool = false) {
        guard let v = QWWeightView(blocks: blocks, n: n, k: k, bits: bits,
                                   groupSize: groupSize, isRotated: isRotated,
                                   isIncumbentContainer: isIncumbentContainer, isSHVQ: isSHVQ) else { return nil }
        self.view = v
    }
    public init?(nibbles: QWTensorValue, scales: QWTensorValue? = nil, mins: QWTensorValue? = nil,
                 n: Int, k: Int, isRotated: Bool, isIncumbentContainer: Bool? = nil, isSHVQ: Bool = false) {
        self.init(blocks: nibbles, n: n, k: k, isRotated: isRotated,
                  isIncumbentContainer: isIncumbentContainer, isSHVQ: isSHVQ)
    }
}

public struct QWMLPPruningPlan: Sendable {
    public let activeChannels: QWTensorValue // [I_active] i32/u32
    public let activeGroups: QWTensorValue   // [G_active] i32/u32
    public let iActive: Int
    public let gActive: Int

    public init(activeChannels: QWTensorValue, activeGroups: QWTensorValue, iActive: Int, gActive: Int) {
        self.activeChannels = activeChannels
        self.activeGroups = activeGroups
        self.iActive = iActive
        self.gActive = gActive
    }

    /// Builds a pruning plan from an array of active group indices (each group is 128 intermediate channels, i.e. 4 HQ4_G32 blocks).
    /// Intermediate channels are 17,408 (136 super-groups of 128).
    public static func make(device: any MTLDevice, activeGroups: [UInt32]) -> QWMLPPruningPlan {
        let sortedGroups = Array(Set(activeGroups)).sorted()
        let gCount = sortedGroups.count

        // Active groups buffer [G_active] UInt32
        let gBuf = device.makeBuffer(length: max(16, gCount * MemoryLayout<UInt32>.stride), options: .storageModeShared)!
        gBuf.contents().copyMemory(from: sortedGroups, byteCount: gCount * MemoryLayout<UInt32>.stride)
        let gTensor = QWTensorValue(
            region: QWMRegion(buffer: gBuf, base: 0, length: gCount * MemoryLayout<UInt32>.stride),
            offset: 0,
            shape: QWTensorShape([gCount], .i32)
        )

        // Active channels buffer [G_active * 128] UInt32
        let iCount = gCount * 128
        var channels = [UInt32]()
        channels.reserveCapacity(iCount)
        for g in sortedGroups {
            let base = g * 128
            for c in 0..<128 {
                channels.append(base + UInt32(c))
            }
        }
        let cBuf = device.makeBuffer(length: max(16, iCount * MemoryLayout<UInt32>.stride), options: .storageModeShared)!
        cBuf.contents().copyMemory(from: channels, byteCount: iCount * MemoryLayout<UInt32>.stride)
        let cTensor = QWTensorValue(
            region: QWMRegion(buffer: cBuf, base: 0, length: iCount * MemoryLayout<UInt32>.stride),
            offset: 0,
            shape: QWTensorShape([iCount], .i32)
        )

        return QWMLPPruningPlan(activeChannels: cTensor, activeGroups: gTensor, iActive: iCount, gActive: gCount)
    }

    /// Identity plan where all 136 super-groups (17,408 channels) are active (100% density).
    public static func allActive(device: any MTLDevice, geometry: QWModelGeometry) -> QWMLPPruningPlan {
        let totalGroups = geometry.intermediate / 128
        return make(device: device, activeGroups: (0..<UInt32(totalGroups)).map { $0 })
    }
}

public final class QWMLP {
    public static let maxBatch = 8                           // M > 8 is F3 prefill territory

    public let geometry: QWModelGeometry
    private let hal: QWHal
    private let psoGu: any MTLComputePipelineState
    private let psoDown: any MTLComputePipelineState
    private let psoGuPruned: any MTLComputePipelineState
    private let psoDownPruned: any MTLComputePipelineState
    /// C14 addendum: M=1 down door, one h-block across 4 rows/TG. Certified
    /// win on ITS door (432 -> 586-648 GB/s) but the 27B decode A/B read
    /// +0.0% because dense decode uses the C05 v4 door instead; opt-in only.
    private var psoDownRows4: (any MTLComputePipelineState)?
    private var rows4Override: Bool?
    public static let downRows4Enabled =
        ProcessInfo.processInfo.environment["QW_MLP_DOWN_ROWS4"] == "1"

    public init(hal: QWHal, geometry: QWModelGeometry) throws {
        self.geometry = geometry
        self.hal = hal
        self.psoGu = try hal.pipeline("qwg_mlp_gu")
        self.psoDown = try hal.pipeline("qwg_mlp_down")
        self.psoGuPruned = try hal.pipeline("qwg_mlp_gu_pruned")
        self.psoDownPruned = try hal.pipeline("qwg_mlp_down_pruned")
        self.psoDownRows4 = Self.downRows4Enabled
            ? try hal.pipeline("qwg_mlp_down_m1_rows4") : nil
    }

    /// Test seam: force the M=1 rows4 door on/off regardless of the env switch.
    public convenience init(hal: QWHal, downRows4: Bool, geometry: QWModelGeometry) throws {
        try self.init(hal: hal, geometry: geometry)
        self.rows4Override = downRows4
        if downRows4, psoDownRows4 == nil {
            self.psoDownRows4 = try hal.pipeline("qwg_mlp_down_m1_rows4")
        }
    }

    func hElems(M: Int) -> Int { M * geometry.intermediate }        // f16
    func yElems(M: Int) -> Int { M * geometry.hidden }              // f16

    /// Full FFN forward: x [M][K] f16 → y [M][H] f16. Two dispatches (gu + down)
    /// under HQ4_G32_M5 (zero xgroupsum passes).
    public func encode(_ enc: any MTLComputeCommandEncoder,
                       x: QWTensorValue, wGu: QWMLPWeight, wDown: QWMLPWeight,
                       h: QWTensorValue, y: QWTensorValue,
                       gs: QWTensorValue? = nil, hs: QWTensorValue? = nil, M: Int,
                       pruning: QWMLPPruningPlan? = nil) throws {
        if let pruning = pruning {
            try encodePruned(enc, x: x, wGu: wGu, wDown: wDown, h: h, y: y, gs: gs, hs: hs, M: M, pruning: pruning)
            return
        }
        let K = wGu.view.k
        try validate(wGu: wGu, wDown: wDown, M: M)
        guard x.shape.elementCount == M * K, x.shape.dtype == .f16,
              h.shape.elementCount == hElems(M: M), h.shape.dtype == .f16,
              y.shape.elementCount == yElems(M: M), y.shape.dtype == .f16 else {
            throw QWMLPError.badShape("x/h/y shape-dtype mismatch (M=\(M))")
        }
        try encodeGu(enc, x: x, gs: gs, wGu: wGu, h: h, M: M)            // 0: gu
        try encodeDown(enc, wDown: wDown, h: h, hs: hs, y: y, M: M)      // 1: down
    }

    /// Pruned SwiGLU forward: evaluates only the salient intermediate groups/channels.
    public func encodePruned(_ enc: any MTLComputeCommandEncoder,
                             x: QWTensorValue, wGu: QWMLPWeight, wDown: QWMLPWeight,
                             h: QWTensorValue, y: QWTensorValue,
                             gs: QWTensorValue? = nil, hs: QWTensorValue? = nil, M: Int,
                             pruning: QWMLPPruningPlan) throws {
        let K = wGu.view.k
        try validate(wGu: wGu, wDown: wDown, M: M)
        guard x.shape.elementCount == M * K, x.shape.dtype == .f16,
              h.shape.elementCount == hElems(M: M), h.shape.dtype == .f16,
              y.shape.elementCount == yElems(M: M), y.shape.dtype == .f16 else {
            throw QWMLPError.badShape("x/h/y shape-dtype mismatch (M=\(M))")
        }
        guard pruning.iActive <= geometry.intermediate, pruning.gActive <= (geometry.intermediate / 128) else {
            throw QWMLPError.badShape("pruning active counts exceed bounds: iActive=\(pruning.iActive), gActive=\(pruning.gActive)")
        }
        try encodeGuPruned(enc, x: x, gs: gs, wGu: wGu, h: h, M: M,
                           activeChannels: pruning.activeChannels, iActive: pruning.iActive)
        try encodeDownPruned(enc, wDown: wDown, h: h, hs: hs, y: y, M: M,
                             activeGroups: pruning.activeGroups, gActive: pruning.gActive)
    }

    private func validate(wGu: QWMLPWeight, wDown: QWMLPWeight, M: Int) throws {
        let I = geometry.intermediate, H = geometry.hidden
        guard M >= 1, M <= Self.maxBatch else { throw QWMLPError.badShape("M=\(M) outside 1...\(Self.maxBatch)") }
        guard wGu.view.k % 32 == 0, wGu.view.n == 2 * I else {
            throw QWMLPError.badShape("gu weight must be [2I][K] k%32==0 (got [\(wGu.view.n)][\(wGu.view.k)])")
        }
        guard wDown.view.n == H, wDown.view.k == I else {
            throw QWMLPError.badShape("down weight must be [H][I] (got [\(wDown.view.n)][\(wDown.view.k)])")
        }
    }

    /// Stage 1 GEMV only. h [M][I] f16 out (with fused fwht32 epilogue).
    public func encodeGu(_ enc: any MTLComputeCommandEncoder,
                         x: QWTensorValue, gs: QWTensorValue? = nil, wGu: QWMLPWeight,
                         h: QWTensorValue, M: Int) throws {
        _ = gs
        let I = geometry.intermediate, K = wGu.view.k
        guard M >= 1, M <= Self.maxBatch else { throw QWMLPError.badShape("M=\(M)") }
        guard wGu.view.n == 2 * I else { throw QWMLPError.badShape("gu [2I][K]") }
        guard x.shape.elementCount == M * K, x.shape.dtype == .f16,
              h.shape.elementCount == hElems(M: M), h.shape.dtype == .f16 else {
            throw QWMLPError.badShape("gu x/h mismatch (M=\(M))")
        }
        var iU = UInt32(I), kU = UInt32(K), mU = UInt32(M)
        enc.setComputePipelineState(psoGu)
        wGu.view.blocks.bind(enc, index: 0)
        x.bind(enc, index: 3)
        h.bind(enc, index: 5)
        enc.setBytes(&iU, length: 4, index: 6)
        enc.setBytes(&kU, length: 4, index: 7)
        enc.setBytes(&mU, length: 4, index: 8)
        enc.dispatchThreadgroups(MTLSize(width: I, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 64 * M, height: 1, depth: 1))
    }

    /// Stage 2 GEMV only. h [M][I] f16 in, y [M][H] f16 out.
    public func encodeDown(_ enc: any MTLComputeCommandEncoder,
                           wDown: QWMLPWeight, h: QWTensorValue, hs: QWTensorValue? = nil,
                           y: QWTensorValue, M: Int) throws {
        _ = hs
        let I = geometry.intermediate, H = geometry.hidden
        guard M >= 1, M <= Self.maxBatch else { throw QWMLPError.badShape("M=\(M)") }
        guard wDown.view.n == H, wDown.view.k == I else { throw QWMLPError.badShape("down [H][I]") }
        guard h.shape.elementCount == hElems(M: M), h.shape.dtype == .f16,
              y.shape.elementCount == yElems(M: M), y.shape.dtype == .f16 else {
            throw QWMLPError.badShape("down h/y mismatch (M=\(M))")
        }
        var nU = UInt32(H), iU = UInt32(I), mU = UInt32(M)
        if M == 1, rows4Override != false, let pso = psoDownRows4 {
            // C14 addendum: 16 rows/TG, one h-block load serves four rows.
            enc.setComputePipelineState(pso)
            wDown.view.blocks.bind(enc, index: 0)
            h.bind(enc, index: 3)
            y.bind(enc, index: 5)
            enc.setBytes(&nU, length: 4, index: 6)
            enc.setBytes(&iU, length: 4, index: 7)
            enc.setBytes(&mU, length: 4, index: 8)
            enc.dispatchThreadgroups(MTLSize(width: (H + 15) / 16, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            return
        }
        enc.setComputePipelineState(psoDown)
        wDown.view.blocks.bind(enc, index: 0)
        h.bind(enc, index: 3)
        y.bind(enc, index: 5)
        enc.setBytes(&nU, length: 4, index: 6)
        enc.setBytes(&iU, length: 4, index: 7)
        enc.setBytes(&mU, length: 4, index: 8)
        enc.dispatchThreadgroups(MTLSize(width: H, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    /// Pruned Stage 1 GEMV only: evaluates only `iActive` intermediate channels.
    public func encodeGuPruned(_ enc: any MTLComputeCommandEncoder,
                               x: QWTensorValue, gs: QWTensorValue? = nil, wGu: QWMLPWeight,
                               h: QWTensorValue, M: Int,
                               activeChannels: QWTensorValue, iActive: Int) throws {
        _ = gs
        let I = geometry.intermediate, K = wGu.view.k
        guard M >= 1, M <= Self.maxBatch else { throw QWMLPError.badShape("M=\(M)") }
        guard wGu.view.n == 2 * I else { throw QWMLPError.badShape("gu [2I][K]") }
        guard x.shape.elementCount == M * K, x.shape.dtype == .f16,
              h.shape.elementCount == hElems(M: M), h.shape.dtype == .f16 else {
            throw QWMLPError.badShape("gu x/h mismatch (M=\(M))")
        }
        guard iActive >= 1, iActive <= I, activeChannels.shape.elementCount >= iActive else {
            throw QWMLPError.badShape("iActive=\(iActive) invalid (exceeds intermediate \(I) or activeChannels length)")
        }
        var iU = UInt32(I), kU = UInt32(K), mU = UInt32(M), actU = UInt32(iActive)
        enc.setComputePipelineState(psoGuPruned)
        wGu.view.blocks.bind(enc, index: 0)
        x.bind(enc, index: 3)
        h.bind(enc, index: 5)
        enc.setBytes(&iU, length: 4, index: 6)
        enc.setBytes(&kU, length: 4, index: 7)
        enc.setBytes(&mU, length: 4, index: 8)
        activeChannels.bind(enc, index: 9)
        enc.setBytes(&actU, length: 4, index: 10)
        enc.dispatchThreadgroups(MTLSize(width: iActive, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 64 * M, height: 1, depth: 1))
    }

    /// Pruned Stage 2 GEMV only: evaluates only `gActive` super-groups of intermediate channels.
    public func encodeDownPruned(_ enc: any MTLComputeCommandEncoder,
                                 wDown: QWMLPWeight, h: QWTensorValue, hs: QWTensorValue? = nil,
                                 y: QWTensorValue, M: Int,
                                 activeGroups: QWTensorValue, gActive: Int) throws {
        _ = hs
        let I = geometry.intermediate, H = geometry.hidden
        guard M >= 1, M <= Self.maxBatch else { throw QWMLPError.badShape("M=\(M)") }
        guard wDown.view.n == H, wDown.view.k == I else { throw QWMLPError.badShape("down [H][I]") }
        guard h.shape.elementCount == hElems(M: M), h.shape.dtype == .f16,
              y.shape.elementCount == yElems(M: M), y.shape.dtype == .f16 else {
            throw QWMLPError.badShape("down h/y mismatch (M=\(M))")
        }
        let totalG = I / 128
        guard gActive >= 1, gActive <= totalG, activeGroups.shape.elementCount >= gActive else {
            throw QWMLPError.badShape("gActive=\(gActive) invalid (exceeds groups \(totalG) or activeGroups length)")
        }
        var nU = UInt32(H), iU = UInt32(I), mU = UInt32(M), gActU = UInt32(gActive)
        enc.setComputePipelineState(psoDownPruned)
        wDown.view.blocks.bind(enc, index: 0)
        h.bind(enc, index: 3)
        y.bind(enc, index: 5)
        enc.setBytes(&nU, length: 4, index: 6)
        enc.setBytes(&iU, length: 4, index: 7)
        enc.setBytes(&mU, length: 4, index: 8)
        activeGroups.bind(enc, index: 9)
        enc.setBytes(&gActU, length: 4, index: 10)
        enc.dispatchThreadgroups(MTLSize(width: H, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }
}
