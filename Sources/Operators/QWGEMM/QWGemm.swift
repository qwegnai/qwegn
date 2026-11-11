import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWQuant

/// C05 — QWGEMM: the single matmul door of the engine (spec specs/05-qwgemm.md).
///
/// Dispatch table = ADR-001 (docs/DECISIONS.md), fair-machine raced. `plan`
public enum QWGemmFamily: String, Sendable {
    case f1DecodeGemv      // M=1, K=5120, N∈{1024,6144,10240,12288,17408}
    case f2LmHeadGemv      // M=1, K=5120, N=248320
    case f3PrefillGemm     // M=512..4096 (EXP-1 resolved: mpsGraphBmmF16)
    case f4AttentionTiles
    case f5SkinnyGemm      // M∈{2,3} const-space batch (F5 v7 — DECISIONS)
}

public enum QWEpilogue: String, Sendable {
    case none
    case siluMulPair       // packed gate‖up rows → silu(gate)·up in-register
}

public struct QWGemmShape: Sendable, Hashable {
    public let m: Int, n: Int, k: Int
    public init(m: Int, n: Int, k: Int) { self.m = m; self.n = n; self.k = k }
    public func family(vocab: Int = QWModelGeometry.qwen38_27b.vocab) -> QWGemmFamily {
        if m == 1 && n == vocab { return .f2LmHeadGemv }
        if m == 1 { return .f1DecodeGemv }
        if m >= 512 { return .f3PrefillGemm }
        if m <= 8 { return .f5SkinnyGemm }
        return .f4AttentionTiles
    }
}

public struct QWGemmPlan: Sendable, Hashable {
    public enum Backend: String, Sendable {
        case handQ4gV4          // F1/F2 winner (q4-g128 planes, 32-thr rows)
        case handQ4gV4H16       // F1/F2 with direct f16 store (P3a)
        case handQ4gSiluPair    // F1 + .siluMulPair epilogue fused
        case handQ4gSiluPairH16 // F1 + .siluMulPair with direct f16 store (P3a)
        case handQ4gSkinny      // F5 v6 M=2/3 TG-staged batch (C16 race)
        case handF16V4          // f16 escape-hatch path (f16 GEMV, 32-thr rows)
        case mpsF16             // library opponent where it measurably won
        // EXP-1 F3 race winner (bench_f3, 2026-10-05, ALL 15 shipping cells,
        // both clock windows: C wins x13–x72 over F1×M; B = hand simdgroup_
        // matrix q4g MMA ran 1.55–1.68 TF/s (A-vs-B ordering is frequency-
        // dependent, both lose to C by an order of magnitude). MPSGraph bmm
        // runs 33–61 TF/s cold — M5's tensor path is reachable only through
        // MPS, not hand MMA (the 4.08 TF/s "ceiling" is the issue-path one).
        // C carries the f16-weight-rounding parity class (≤2e-2 vs fp32 CPU,
        // S1-q4g form) and costs the dequant-to-f16 pass every call (timed
        // in). The door is queue-based (MPSGraph runAsync + completionHandler
        // — the only completion protocol canary-proven on this OS), NOT an
        // MTLComputeCommandEncoder kernel: encode() refuses with the pointer.
        case mpsGraphBmmF16     // F3 winner: dequant→f16 scratch + MPSGraph bmm
        case blocked            // F4: ladder pending (F3 resolved by EXP-1)
    }
    public let backend: Backend
    public let kernel: String
    public let threadsPerTG: Int
    public let family: QWGemmFamily
    public let epilogue: QWEpilogue
}

public enum QWGemmError: Error, CustomStringConvertible {
    case f3Blocked(String)
    case badWeights(String)
    case badShape(String)
    case badGeometry(String)

    public var description: String {
        switch self {
        case .f3Blocked(let s): return "QWGEMM: F3 prefill path blocked — \(s) (S4 locked matmul2d; race = simdgroup_matrix → MPSGraph)"
        case .badWeights(let s): return "QWGEMM: weight view invalid — \(s)"
        case .badShape(let s): return "QWGEMM: shape invalid — \(s)"
        case .badGeometry(let s): return "QWGEMM: geometry invalid — \(s)"
        }
    }
}

/// Read-side view of one quantized weight tensor: it names the FORMAT as well
/// as the shape, so the arena can be sized at the pair's NATIVE bytes. It also
/// names the BASIS, and every construction states it (specs/04-qwquant.md).
public struct QWWeightView: Sendable {
    public let blocks: QWTensorValue
    public var nibbles: QWTensorValue { blocks }
    public var scales: QWTensorValue { blocks }
    public var mins: QWTensorValue { blocks }
    public let n: Int          // output rows
    public let k: Int          // inner dim (must be % groupSize)
    public let bits: Int       // 2, 3, 4, 6, 8
    public let groupSize: Int  // 32, 64, 128, 256
    public let isRotated: Bool // true: the FWHT-rotated basis (HQ4_G32 or HQ{2,3,4,6,8}_G{32,64,128,256})
    /// Whether the INCUMBENT's container (`Block_HQ4_G32`, folded bias, word-interleaved nibbles)
    /// holds this view. Rotated `HQ{2,3,4,6,8}_G{32,64,128,256}` bit-stream blocks set `isRotated = true`
    /// and `isIncumbentContainer = false`.
    public let isIncumbentContainer: Bool
    /// Whether this view holds rotated `SHVQ{2,3,4}_G{32,64,128,256}` vector-quantized blocks.
    public let isSHVQ: Bool

    /// The bytes this bank's rows take at the pair's native size.
    public var rowBytes: Int {
        isSHVQ
            ? QWSHVQBlock.rowBytes(bits: bits, group: groupSize, k: k)
            : QWAQBlock.rowBytes(bits: bits, group: groupSize, k: k)
    }
    /// The bytes the whole bank takes at the pair's native size.
    public var requiredBytes: Int { rowBytes * n }
    /// The container's name — `HQ4_G32` for the incumbent's container, `SHVQ{b}_G{G}` for
    /// rotated SHVQ blocks, `HQ{b}_G{G}` for rotated bit-stream blocks, and `AQ{b}_G{G}` for affine blocks.
    public var formatName: String {
        if isIncumbentContainer { return "HQ4_G32" }
        if isSHVQ { return QWSHVQBlock.formatName(bits: bits, group: groupSize) }
        if isRotated { return "HQ\(bits)_G\(groupSize)" }
        return QWAQBlock.formatName(bits: bits, group: groupSize)
    }

    /// `isRotated` has no default ON PURPOSE: a view that guessed the basis
    /// could carry rotated weights into a standard-basis run. Stating it
    /// everywhere means the compiler finds the sites when the basis changes.
    public init?(blocks: QWTensorValue, n: Int, k: Int,
                 bits: Int = 4, groupSize: Int = 32, isRotated: Bool,
                 isIncumbentContainer: Bool? = nil, isSHVQ: Bool = false) {
        let incumbent = isIncumbentContainer ?? (!isSHVQ && isRotated && bits == 4 && groupSize == 32)
        if isSHVQ {
            guard isRotated && !incumbent && QWSHVQBlock.isSupported(bits: bits, group: groupSize) else { return nil }
        } else if incumbent {
            guard isRotated && bits == 4 && groupSize == 32 else { return nil }
        } else if isRotated {
            guard QWAQBlock.isHQSupported(bits: bits, group: groupSize) else { return nil }
        } else {
            guard QWAQBlock.isSupported(bits: bits, group: groupSize) else { return nil }
        }
        let maybeGroupBytes = isSHVQ
            ? QWSHVQBlock.bytesIfSupported(bits: bits, group: groupSize)
            : QWAQBlock.bytesIfBlockSupported(bits: bits, group: groupSize)
        guard n > 0, k > 0, k % groupSize == 0,
              let groupBytes = maybeGroupBytes,
              blocks.bytes >= n * (k / groupSize) * groupBytes
        else { return nil }
        self.blocks = blocks
        self.n = n; self.k = k
        self.bits = bits; self.groupSize = groupSize
        self.isRotated = isRotated
        self.isIncumbentContainer = incumbent
        self.isSHVQ = isSHVQ
    }
    public init?(nibbles: QWTensorValue, scales: QWTensorValue? = nil, mins: QWTensorValue? = nil,
                 n: Int, k: Int,
                 bits: Int = 4, groupSize: Int = 32, isRotated: Bool,
                 isIncumbentContainer: Bool? = nil, isSHVQ: Bool = false) {
        self.init(blocks: nibbles, n: n, k: k,
                  bits: bits, groupSize: groupSize, isRotated: isRotated,
                  isIncumbentContainer: isIncumbentContainer, isSHVQ: isSHVQ)
    }

    /// A view over other bytes of the same bank — a slice of a concatenated
    /// bank, an expert's rows out of a stack. It is the SAME weights in the
    /// SAME basis, so it inherits the basis rather than being told again.
    public func subview(blocks: QWTensorValue, n: Int, k: Int) -> QWWeightView? {
        QWWeightView(blocks: blocks, n: n, k: k,
                     bits: bits, groupSize: groupSize, isRotated: isRotated,
                     isIncumbentContainer: isIncumbentContainer, isSHVQ: isSHVQ)
    }
}

extension QWWeightView: Equatable {
    /// Two views are the same tensor when they name the same format and
    /// shape over the same bytes of the same buffer.
    public static func == (a: QWWeightView, b: QWWeightView) -> Bool {
        a.n == b.n && a.k == b.k
            && a.bits == b.bits && a.groupSize == b.groupSize
            && a.isRotated == b.isRotated
            && a.isIncumbentContainer == b.isIncumbentContainer
            && a.isSHVQ == b.isSHVQ
            && a.blocks.bytes == b.blocks.bytes
            && a.blocks.offset == b.blocks.offset
            && a.blocks.region.base == b.blocks.region.base
            && a.blocks.region.buffer === b.blocks.region.buffer
    }
}

public enum QWGemm {
    /// Constant-BINDING cap for the v7 skinny window: 256 KB — a MEASURED
    /// hardware law (C14 GPU-quiet gate: bindings above 256 KB throw at
    /// encode on this hardware even though the compiler accepts them —
    public static let maxConstantBytes: Int = 256 * 1024

    /// Pure table lookup — ADR-001 winners. No side effects.
    public static func plan(_ shape: QWGemmShape, dtype: QWDType, epi: QWEpilogue, outDtype: QWDType = .f32) -> QWGemmPlan {
        let fam = shape.family()
        switch (fam, dtype, epi, outDtype) {
        case (_, .q4g, .siluMulPair, .f16):
            return QWGemmPlan(backend: .handQ4gSiluPairH16, kernel: "qwg_gemv_q4g_silu_pair_h16",
                              threadsPerTG: 64, family: fam, epilogue: epi)
        case (_, .q4g, .siluMulPair, _):
            // 64 threads = two 32-lane folds (gate rows ‖ up rows) in parallel
            return QWGemmPlan(backend: .handQ4gSiluPair, kernel: "qwg_gemv_q4g_silu_pair",
                              threadsPerTG: 64, family: fam, epilogue: epi)
        case (.f1DecodeGemv, .q4g, .none, .f16), (.f2LmHeadGemv, .q4g, .none, .f16):
            return QWGemmPlan(backend: .handQ4gV4H16, kernel: "qwg_gemv_q4g_v4_h16",
                              threadsPerTG: 32, family: fam, epilogue: epi)
        case (.f1DecodeGemv, .q4g, .none, _), (.f2LmHeadGemv, .q4g, .none, _):
            return QWGemmPlan(backend: .handQ4gV4, kernel: "qwg_gemv_q4g_v4",
                              threadsPerTG: 32, family: fam, epilogue: epi)
        case (.f5SkinnyGemm, .q4g, .none, _):
            if shape.m == 2 || shape.m == 3 {
                let xBytes = shape.m * shape.k * 2
                if xBytes <= maxConstantBytes {
                    return QWGemmPlan(backend: .handQ4gSkinny,
                                      kernel: shape.m == 2 ? "qwg_skinny_m2" : "qwg_skinny_m3",
                                      threadsPerTG: 32, family: fam, epilogue: epi)
                }
            }
            return QWGemmPlan(backend: .blocked, kernel: "", threadsPerTG: 0,
                              family: fam, epilogue: epi)
        case (.f1DecodeGemv, .f16, .none, _), (.f2LmHeadGemv, .f16, .none, _):
            return QWGemmPlan(backend: .handF16V4, kernel: "qwg_gemv_f16_v4",
                              threadsPerTG: 32, family: fam, epilogue: epi)
        case (.f3PrefillGemm, .q4g, .none, _):
            if shape.k % 32 == 0 {
                return QWGemmPlan(backend: .mpsGraphBmmF16, kernel: "mpsgraph_bmm_f16",
                                  threadsPerTG: 0, family: fam, epilogue: epi)
            }
            return QWGemmPlan(backend: .blocked, kernel: "", threadsPerTG: 0,
                              family: fam, epilogue: epi)
        case (_, _, _, _):
            return QWGemmPlan(backend: .blocked, kernel: "", threadsPerTG: 0,
                              family: fam, epilogue: epi)
        }
    }

    /// Encode a decode GEMV directly into f16 storage (P3a).
    /// If y32 is provided, uses dual-store (writes both y32 and y16).
    public static func encodeH16(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                 plan: QWGemmPlan, x: QWTensorValue,
                                 xGroupSum: QWTensorValue? = nil, wQ4: QWWeightView,
                                 y16: QWTensorValue, y32: QWTensorValue? = nil) throws {
        guard plan.backend == .handQ4gV4H16 || plan.backend == .handQ4gSiluPairH16 else {
            throw QWGemmError.badWeights("encodeH16 requires handQ4gV4H16 or handQ4gSiluPairH16")
        }
        guard y16.shape.dtype == .f16 else {
            throw QWGemmError.badWeights("y16 must have dtype .f16")
        }
        guard plan.epilogue != .siluMulPair || wQ4.n % 2 == 0 else {
            throw QWGemmError.badWeights("silu_pair needs even N")
        }
        if !wQ4.isIncumbentContainer {
            guard y32 == nil else {
                throw QWGemmError.badWeights("dual-store encodeH16 requires incumbent container")
            }
            if plan.epilogue == .siluMulPair {
                guard let name = QWGemmAQ.siluPairKernelName(bits: wQ4.bits, group: wQ4.groupSize,
                                                             isSHVQ: wQ4.isSHVQ) else {
                    throw QWGemmError.badWeights("no affine-quantised silu_pair kernel for \(wQ4.bits)-bit group \(wQ4.groupSize)")
                }
                let pso = try hal.pipeline(name)
                enc.setComputePipelineState(pso)
                wQ4.blocks.bind(enc, index: 0)
                x.bind(enc, index: 1)
                y16.bind(enc, index: 3)
                var nPacked = UInt32(wQ4.n) | 0x8000_0000
                var kk = UInt32(wQ4.k)
                enc.setBytes(&nPacked, length: 4, index: 4)
                enc.setBytes(&kk, length: 4, index: 5)
                y16.bind(enc, index: 8)
                let intermediate = wQ4.n / 2
                enc.dispatchThreadgroups(MTLSize(width: (intermediate + 7) / 8, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                return
            } else {
                try QWGemmAQ.encodeDecodeT4(enc, hal: hal, blocks: wQ4.blocks, x: x, y16: y16,
                                            n: wQ4.n, k: wQ4.k, bits: wQ4.bits, group: wQ4.groupSize,
                                            isSHVQ: wQ4.isSHVQ)
                return
            }
        }
        let kernelName: String
        if plan.backend == .handQ4gV4H16 {
            kernelName = (y32 != nil) ? "qwg_gemv_q4g_v4_h16_dual" : "qwg_gemv_q4g_v4_h16"
        } else {
            kernelName = plan.kernel
        }
        let pso = try hal.pipeline(kernelName)
        enc.setComputePipelineState(pso)
        wQ4.blocks.bind(enc, index: 0)
        x.bind(enc, index: 1)
        if let y32 = y32 {
            y32.bind(enc, index: 3)
        } else {
            y16.bind(enc, index: 3)
        }
        var n = UInt32(wQ4.n), kk = UInt32(wQ4.k)
        enc.setBytes(&n, length: 4, index: 4)
        enc.setBytes(&kk, length: 4, index: 5)
        y16.bind(enc, index: 8)
        let rows = plan.epilogue == .siluMulPair ? wQ4.n / 2 : wQ4.n
        if plan.backend == .handQ4gV4H16 {
            enc.dispatchThreadgroups(MTLSize(width: (rows + 3) / 4, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        } else if plan.backend == .handQ4gSiluPairH16 {
            enc.dispatchThreadgroups(MTLSize(width: (rows + 3) / 4, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        } else {
            enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: plan.threadsPerTG, height: 1, depth: 1))
        }
    }


    /// Encode a decode/f2 GEMV (the engine's hot door). y layout:
    /// .none → y[n]; .siluMulPair → y[n/2]. Encode path allocates nothing.
    public static func encode(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                              plan: QWGemmPlan, x: QWTensorValue,
                              xGroupSum: QWTensorValue? = nil, wQ4: QWWeightView,
                              y: QWTensorValue) throws {
        guard plan.backend != .blocked else {
            if plan.family == .f5SkinnyGemm {
                throw QWGemmError.badWeights("F5 window is M∈{2,3} (M≥4: F1×M is the C14-measured winner; K≤cap) — run F1 per row")
            }
            throw QWGemmError.f3Blocked("no winner raced for \(plan.family.rawValue)")
        }
        if plan.backend == .mpsGraphBmmF16 {
            throw QWGemmError.f3Blocked("plan resolved to mpsGraphBmmF16 — use QWEngineDriver NAX prefill door")
        }
        guard plan.epilogue != .siluMulPair || wQ4.n % 2 == 0 else {
            throw QWGemmError.badWeights("silu_pair needs even N")
        }
        if !wQ4.isIncumbentContainer && plan.epilogue == .none {
            try QWGemmAQ.encode(enc, hal: hal, blocks: wQ4.blocks, x: x, y: y,
                                n: wQ4.n, k: wQ4.k, bits: wQ4.bits, group: wQ4.groupSize,
                                isSHVQ: wQ4.isSHVQ)
            return
        }
        let pso = try hal.pipeline(plan.kernel)
        enc.setComputePipelineState(pso)
        wQ4.blocks.bind(enc, index: 0)
        x.bind(enc, index: 1)
        y.bind(enc, index: 3)
        var n = UInt32(wQ4.n), kk = UInt32(wQ4.k)
        enc.setBytes(&n, length: 4, index: 4)
        enc.setBytes(&kk, length: 4, index: 5)
        let rows = plan.epilogue == .siluMulPair ? wQ4.n / 2 : wQ4.n
        if plan.backend == .handQ4gV4 {
            enc.dispatchThreadgroups(MTLSize(width: (rows + 3) / 4, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        } else {
            enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: plan.threadsPerTG, height: 1, depth: 1))
        }
    }

    /// In-place 32-point normalized Fast Walsh-Hadamard Transform (H_32 / sqrt(32))
    /// over `count ?? x.shape.elementCount` f16 elements. Since H_32 is symmetric
    /// and orthogonal (H_32^2 = I), this both rotates and un-rotates activations.
    public static func encodeFwht32(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                    x: QWTensorValue, count: Int? = nil) throws {
        let n = count ?? x.shape.elementCount
        guard x.shape.dtype == .f16, n > 0, n % 32 == 0 else {
            throw QWGemmError.badWeights("fwht32 f16: n=\(n) must be > 0 and divisible by 32")
        }
        let pso = try hal.pipeline("qwg_fwht32_f16")
        enc.setComputePipelineState(pso)
        x.bind(enc, index: 0)
        let tgSize = (n % 256 == 0) ? 256 : 32
        enc.dispatchThreadgroups(MTLSize(width: n / tgSize, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
    }

    /// SEAM LAW (C16): GEMV doors emit f32 (accumulate law); every
    /// downstream door (norm/rope/attn/MLP) takes f16 engine storage. This
    /// is the explicit cast seam — 256-thread TGs, 8-elem vectors.
    public static func encodeCast(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                  x32: QWTensorValue, y16: QWTensorValue, count: Int? = nil) throws {
        let n = count ?? x32.shape.elementCount
        guard x32.shape.dtype == .f32, y16.shape.dtype == .f16,
              n > 0, x32.shape.elementCount >= n, y16.shape.elementCount >= n, n % 8 == 0 else {
            throw QWGemmError.badWeights("cast f32→f16: n=\(n) must fit buffers and divide 8")
        }
        let pso = try hal.pipeline("qwg_cast_f32_to_f16")
        enc.setComputePipelineState(pso)
        x32.bind(enc, index: 0)
        y16.bind(enc, index: 1)
        var nF8 = UInt32(n / 8)
        enc.setBytes(&nF8, length: 4, index: 2)
        enc.dispatchThreadgroups(MTLSize(width: (Int(n) / 8 + 255) / 256, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
}
