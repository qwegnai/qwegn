import Foundation
import Metal
import QWCore
import QWQuant
import QWHAL
import QWAlloc
import QWGEMM
import QWGDN

/// Activation buffers for one FlashNext layer pass over T tokens (prefill / verify shape). Sized from the layer dims.
public final class FNGPUScratch {
    public var T: Int
    public let maxT: Int, hidden: Int, hc: Int, width: Int
    // PLE and stream
    public let pleEmbed: QWTensorValue, pleKV: QWTensorValue, pleK: QWTensorValue, pleV: QWTensorValue
    public let pleKeyN: QWTensorValue, pleQueryN: QWTensorValue, pleGV: QWTensorValue, pleGVN: QWTensorValue, pleOut: QWTensorValue
    public let h: QWTensorValue, h2: QWTensorValue
    // HC
    public let hcXN: QWTensorValue, hcLow: QWTensorValue, hcGate: QWTensorValue
    public let hcRotW: QWTensorValue
    public let mixedA: QWTensorValue, mixedB: QWTensorValue, injA: QWTensorValue, injB: QWTensorValue
    public let rot16: QWTensorValue
    public let q32: QWTensorValue, gateF32: QWTensorValue, attnOut32: QWTensorValue, rot32: QWTensorValue
    public let oG: QWTensorValue, moeOut: QWTensorValue
    // GatedDeltaNet
    public let qkvZ32: QWTensorValue, qkv32: QWTensorValue, z32: QWTensorValue
    public let gdnBA: QWTensorValue, gdnB: QWTensorValue, gdnA: QWTensorValue
    public let yDense32: QWTensorValue, glog: QWTensorValue, beta: QWTensorValue
    public let outNorm32: QWTensorValue
    // Full attention
    public let src: QWTensorValue, q: QWTensorValue, k: QWTensorValue, v: QWTensorValue, gateF: QWTensorValue
    public let idxQ: QWTensorValue, idxK: QWTensorValue, pooled: QWTensorValue, list: QWTensorValue, qsaScores: QWTensorValue, attnOut: QWTensorValue
    public let cosT: QWTensorValue, sinT: QWTensorValue, bcos: QWTensorValue, bsin: QWTensorValue
    public let blocks: Int, topK: Int, bankRows: Int
    // MoE
    public let logits: QWTensorValue, idx: QWTensorValue, wRoute: QWTensorValue
    public let gu: QWTensorValue, gu32: QWTensorValue, act32: QWTensorValue, zerosRes: QWTensorValue, onesInj: QWTensorValue, routed: QWTensorValue
    public let xe16: QWTensorValue, tileE: QWTensorValue, tileR: QWTensorValue, tileCount: QWTensorValue
    public let dnHi: QWTensorValue, dnLo: QWTensorValue, dnAcc: QWTensorValue
    public let naxX16: QWTensorValue, naxY16: QWTensorValue
    public let expOff: QWTensorValue, sortedPair: QWTensorValue, sortedTok: QWTensorValue, invPerm: QWTensorValue, downPair: QWTensorValue
    public let guSh: QWTensorValue, actSh: QWTensorValue, sh: QWTensorValue, sg: QWTensorValue
    public let guSh32: QWTensorValue, actSh32: QWTensorValue
    public let groupIds: QWTensorValue?, onesRoute: QWTensorValue?
    public let xLatent32: QWTensorValue?, xLatent16: QWTensorValue?, zLatent32: QWTensorValue?

    /// Allocate for T tokens. `gdn`/`attn` dims come from the first layers of each kind (nil when the model has none).
    public init(arena: QWArena, T: Int, hidden H: Int, hc C: Int, lowrank R: Int, moe: FNGPUMoE, ple: FNGPUPLE?,
                gdn: FNGPUGDNMixer?, attn: FNGPUAttnMixer?, extraEmbedDim: Int = 0, maxCapacity: Int = 0) throws {
        self.T = T; self.maxT = T; self.hidden = H; self.hc = C; self.width = C * H
        let W = C * H, NE = moe.experts, TK = moe.topK, I = moe.intermediate, IS = moe.sharedIntermediate
        let act = arena.activationsRegion()
        func f32(_ n: Int) throws -> QWTensorValue { try act.allocate(QWTensorShape([max(n, 1)], .f32)) }
        func f16(_ n: Int) throws -> QWTensorValue { try act.allocate(QWTensorShape([max(n, 1)], .f16)) }
        func i32(_ n: Int) throws -> QWTensorValue { try act.allocate(QWTensorShape([max(n, 1)], .i32)) }
        func zeros32(_ n: Int) throws -> QWTensorValue {
            let t = try f32(n); arena.copyIn(t, from: Data(count: n * 4)); return t
        }
        let isPrefill = T > 8
        let E = max(ple?.embedDim ?? 0, extraEmbedDim, 1)
        pleEmbed = try f32(T * E)
        pleKV = try f32(T * (W + H))
        pleK = pleKV.slice(byteOffset: 0, count: T * W, dtype: .f32)
        pleV = pleKV.slice(byteOffset: T * W * 4, count: T * H, dtype: .f32)
        pleKeyN = try f32(T * W); pleQueryN = try f32(T * W); pleGV = try f32(T * W); pleGVN = try f32(T * W); pleOut = try f32(T * W)
        h = try f32(T * W); h2 = try f32(T * W)
        hcXN = try f32(T * W); hcLow = try f32(T * R); hcGate = try f32(isPrefill ? T * W : 1)
        hcRotW = try f32(T * W)
        mixedA = try f32(T * H); mixedB = try f32(T * H); injA = try f32(T * C); injB = try f32(T * C)
        rot16 = try f16(isPrefill ? T * H : 1)
        let aNH = attn?.heads ?? 1, aDA = attn?.headDim ?? 1
        let attnP = isPrefill ? T * aNH * aDA : 1
        q32 = try f32(attnP); gateF32 = try f32(attnP); attnOut32 = try f32(attnP)
        rot32 = try f32(T * max(H, E))
        oG = try f32(T * H); moeOut = try f32(T * H)

        let VH = gdn?.vHeads ?? 1, VD = gdn?.vDim ?? 1, VALUE = VH * VD, CONV = gdn?.convDim ?? 1
        qkvZ32 = try f32(T * (CONV + VALUE))
        qkv32 = qkvZ32.slice(byteOffset: 0, count: T * CONV, dtype: .f32)
        z32 = qkvZ32.slice(byteOffset: T * CONV * 4, count: T * VALUE, dtype: .f32)
        gdnBA = try f32(T * 2 * VH)
        gdnB = gdnBA.slice(byteOffset: 0, count: T * VH, dtype: .f32)
        gdnA = gdnBA.slice(byteOffset: T * VH * 4, count: T * VH, dtype: .f32)
        yDense32 = try f32(T * CONV); glog = try f32(T * VH); beta = try f32(T * VH)
        outNorm32 = try f32(T * VALUE)

        let IH = attn?.indexHeads ?? 1, ID = attn?.indexHeadDim ?? 1
        // Full attention
        bankRows = attn?.bankRows ?? 1
        topK = attn?.blockTopK ?? 1
        blocks = attn.map { T / $0.compress } ?? 1
        let maxBlocks = attn.map { max(T, maxCapacity) / max(1, $0.compress) } ?? 1
        let scoreElems = isPrefill ? max(min(T, 64) * maxBlocks, min(T, 4096) * 2048) : 1
        src = try f32(isPrefill ? T * bankRows : 1)
        q = try f16(1); k = try f16(1); v = try f16(1); gateF = try f32(1)
        idxQ = try f32(isPrefill ? T * IH * ID : 1); idxK = try f32(1); pooled = try f32(1)
        list = try i32(isPrefill ? T * topK : 1); qsaScores = try i32(scoreElems); attnOut = try f16(1)
        cosT = try f32(1); sinT = try f32(1); bcos = try f32(1); bsin = try f32(1)

        let NG = moe.numLatentGroups, L = moe.latentDim
        let maxRoutes = max(TK, NG)
        logits = try f32(T * NE); idx = try i32(T * TK); wRoute = try f32(T * TK)
        let pairP = isPrefill ? T * maxRoutes : 1
        expOff = try i32(isPrefill ? NE + 1 : 1); sortedPair = try i32(pairP); sortedTok = try i32(pairP); invPerm = try i32(pairP)
        gu = try f16(isPrefill ? T * TK * 2 * I : 1)
        let plan23 = isPrefill ? QWGemm.MoETilePlan(
            experts: NE, pairs: T * TK, gateUpRows: 2 * I, downRows: L > 0 ? L : H).scratchInts : 1
        let plan1 = (isPrefill && NG > 0 && L > 0) ? QWGemm.MoETilePlan(
            experts: NG, pairs: T * NG, gateUpRows: L, downRows: 64).scratchInts : 1
        let plan4 = (isPrefill && NG > 0) ? QWGemm.MoETilePlan(
            experts: NG, pairs: T * NG, gateUpRows: H, downRows: 64).scratchInts : 1
        let tcInts = max(plan23, max(plan1, plan4))
        xe16 = try f16(1); tileE = try i32(pairP); tileR = try i32(pairP); tileCount = try i32(tcInts)
        dnHi = try f16(isPrefill ? T * TK * I : 1); dnLo = dnHi; dnAcc = try f16(isPrefill ? T * maxRoutes * H : 1)
        downPair = dnAcc
        naxX16 = try f16(isPrefill ? T * W : 1); naxY16 = naxX16
        let r1Rows = min(T, 8) * TK
        gu32 = try f32(r1Rows * 2 * I); act32 = try f32(r1Rows * I)
        zerosRes = try zeros32(1)
        let ones = try f32(T); arena.write(ones, from: [Float](repeating: 1, count: T))
        onesInj = ones
        routed = try f32(1)
        guSh = try f16(1); actSh = try f16(1); sh = try f32(T * H); sg = try f32(T)
        guSh32 = try f32(T * 2 * IS); actSh32 = try f32(T * IS)
        if NG > 0 && L > 0 {
            groupIds = try i32(T * NG)
            let onesRCount = max(T, 8) * NG
            let onesR = try f32(onesRCount)
            arena.write(onesR, from: [Float](repeating: 1, count: onesRCount))
            onesRoute = onesR
            xLatent32 = try f32(T * NG * L)
            xLatent16 = try f16(isPrefill ? T * NG * L : 1)
            zLatent32 = try f32((isPrefill ? T * TK : max(T, 8) * NG) * L)
        } else {
            groupIds = nil; onesRoute = nil; xLatent32 = nil; xLatent16 = nil; zLatent32 = nil
        }
    }
}

/// Kernel objects shared by every layer pass.
public final class FNGPUContext {
    public let hal: QWHal
    let recurrence: QWGDNDecode
    let convKernel: QWGDNConv

    /// Creates the shared kernel objects and the doors they dispatch.
    /// The doors come from the module that owns this engine: there is
    /// no second list of them here.
    public init(hal: QWHal, doors: [String], geometry: QWModelGeometry) throws {
        self.hal = hal
        self.recurrence = try QWGDNDecode(hal: hal)
        self.convKernel = try QWGDNConv(hal: hal, geometry: geometry)
        // A door that does not compile is a door this build cannot
        // dispatch: the engine says so now, not later in a turn.
        for name in doors { _ = try hal.pipeline(name) }
        hal.freezePipelines()
    }
}
