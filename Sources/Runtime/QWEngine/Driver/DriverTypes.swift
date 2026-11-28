// DriverTypes.swift — QWEngineDriver nested value types
// (split from QWEngineDriver.swift; same-module extension).
import Foundation
import Darwin
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWNorm
import QWRope
import QWKV
import QWAttn
import QWGDN
import QWMLP
import QWHead
import QWMTP
import QWQuant
import QWLoad

extension QWEngineDriver {
    final class GdnW: @unchecked Sendable {
        let inProj: QWWeightView
        let qkv: QWWeightView, z: QWWeightView, a: QWWeightView, b: QWWeightView
        let conv: QWTensorValue, normW: QWTensorValue, aLog: QWTensorValue, dtBias: QWTensorValue
        let out: QWWeightView
        let fgbProj: QWWeightView?
        let bProjF16: QWTensorValue?
        let fB: QWTensorValue?
        let gB: QWTensorValue?

        init(inProj: QWWeightView, qkv: QWWeightView, z: QWWeightView, a: QWWeightView, b: QWWeightView,
             conv: QWTensorValue, normW: QWTensorValue, aLog: QWTensorValue, dtBias: QWTensorValue,
             out: QWWeightView, fgbProj: QWWeightView? = nil, bProjF16: QWTensorValue? = nil,
             fB: QWTensorValue? = nil, gB: QWTensorValue? = nil) {
            self.inProj = inProj; self.qkv = qkv; self.z = z; self.a = a; self.b = b
            self.conv = conv; self.normW = normW; self.aLog = aLog; self.dtBias = dtBias
            self.out = out; self.fgbProj = fgbProj; self.bProjF16 = bProjF16; self.fB = fB; self.gB = gB
        }
    }
    final class AttnW: @unchecked Sendable {
        let qkvProj: QWWeightView
        let q: QWWeightView, k: QWWeightView, v: QWWeightView, o: QWWeightView
        /// The per-head q/k norms, when the model has them. A standard family
        /// (llama, mistral, qwen2) has neither: its q and k are projected and
        /// roped as they are.
        let wq: QWTensorValue?, wkTiled: QWTensorValue?
        let mla: QWMLAAttnOperator.LayerWeights?
        let mbsa: QWMBSAAttnOperator.LayerWeights?
        let qkvBias: QWTensorValue?
        let oBias: QWTensorValue?

        init(qkvProj: QWWeightView, q: QWWeightView, k: QWWeightView, v: QWWeightView, o: QWWeightView,
             wq: QWTensorValue?, wkTiled: QWTensorValue?,
             mla: QWMLAAttnOperator.LayerWeights? = nil,
             mbsa: QWMBSAAttnOperator.LayerWeights? = nil,
             qkvBias: QWTensorValue? = nil,
             oBias: QWTensorValue? = nil) {
            self.qkvProj = qkvProj; self.q = q; self.k = k; self.v = v; self.o = o
            self.wq = wq; self.wkTiled = wkTiled; self.mla = mla; self.mbsa = mbsa
            self.qkvBias = qkvBias; self.oBias = oBias
        }
    }
    enum QWLayerPass: Equatable, Sendable { case fullAttention, gdn, hca, moe, gdnMoe, hcaMoe }
    /// On-demand F_NOCACHE SSD streamer for cold routed experts (e >= residentExperts).
    /// Streams cold expert weights into staging slots inside `gateUp`/`down` and zeroes them after use.


    final class MoEW: @unchecked Sendable {
        let router: QWTensorValue
        let routerBias: QWTensorValue?
        let routedScaleParam: QWTensorValue?
        let splitFeatureRouting: Bool
        let latent: QWMoEOperator.LatentBottleneck?
        let gateUp: QWWeightView
        let down: QWWeightView
        let coldGateUp: QWWeightView?
        let coldDown: QWWeightView?
        let numExperts: Int
        let expertsPerTok: Int
        let sharedGateUp: QWWeightView?
        let sharedDown: QWWeightView?
        let sharedGateScalar: QWTensorValue?
        let coldStreamer: ColdExpertStreamer?
        let useSigmoidScore: Bool
        let nGroup: Int
        let topkGroup: Int
        let groupTopM: Int
        let routedSwigluClamp: QWSwiGLUClampSpec?
        let sharedSwigluClamp: QWSwiGLUClampSpec?

        init(router: QWTensorValue, gateUp: QWWeightView, down: QWWeightView,
             numExperts: Int, expertsPerTok: Int,
             sharedGateUp: QWWeightView? = nil, sharedDown: QWWeightView? = nil,
             sharedGateScalar: QWTensorValue? = nil,
             routerBias: QWTensorValue? = nil,
             routedScaleParam: QWTensorValue? = nil,
             splitFeatureRouting: Bool = false,
             latent: QWMoEOperator.LatentBottleneck? = nil,
             coldGateUp: QWWeightView? = nil, coldDown: QWWeightView? = nil,
             coldStreamer: ColdExpertStreamer? = nil,
             useSigmoidScore: Bool = false,
             nGroup: Int = 1, topkGroup: Int = 1, groupTopM: Int = 1,
             routedSwigluClamp: QWSwiGLUClampSpec? = nil,
             sharedSwigluClamp: QWSwiGLUClampSpec? = nil) {
            self.router = router
            self.routerBias = routerBias
            self.routedScaleParam = routedScaleParam
            self.splitFeatureRouting = splitFeatureRouting
            self.latent = latent
            self.gateUp = gateUp
            self.down = down
            self.coldGateUp = coldGateUp
            self.coldDown = coldDown
            self.numExperts = numExperts
            self.expertsPerTok = expertsPerTok
            self.sharedGateUp = sharedGateUp
            self.sharedDown = sharedDown
            self.sharedGateScalar = sharedGateScalar
            self.coldStreamer = coldStreamer
            self.useSigmoidScore = useSigmoidScore
            self.nGroup = nGroup
            self.topkGroup = topkGroup
            self.groupTopM = groupTopM
            self.routedSwigluClamp = routedSwigluClamp
            self.sharedSwigluClamp = sharedSwigluClamp
        }
    }
    struct MHCLayerWeights {
        let attnW: QWTensorValue
        let attnB: QWTensorValue
        let ffnW: QWTensorValue
        let ffnB: QWTensorValue
    }
    struct GatedLayerNorms: @unchecked Sendable {
        let input: QWGatedNormWeights
        let postSelfAttn: QWGatedNormWeights
        let postAttn: QWGatedNormWeights
        let postFFN: QWGatedNormWeights
    }
    struct GatedMTPNorms: @unchecked Sendable {
        let enorm: QWGatedNormWeights
        let hnorm: QWGatedNormWeights
        let input: QWGatedNormWeights
        let postSelfAttn: QWGatedNormWeights
        let postAttn: QWGatedNormWeights
        let postFFN: QWGatedNormWeights
        let sharedHead: QWGatedNormWeights
    }
    final class LayerW: @unchecked Sendable {
        let ln1: QWTensorValue, ln2: QWTensorValue
        let gu: QWMLPWeight, down: QWWeightView
        let kind: QWLayerPass
        let gdn: GdnW?, attn: AttnW?, hca: QWHCAAttnOperator.LayerWeights?
        /// The sparse MoE weights — nil for a dense layer. A MoE layer's
        /// `gu`/`down` hold the FIRST expert's views, so they stay valid
        /// views; the banks live here.
        let moe: MoEW?
        let mhc: MHCLayerWeights?
        let gatedNorms: GatedLayerNorms?

        init(ln1: QWTensorValue, ln2: QWTensorValue, gu: QWMLPWeight, down: QWWeightView,
             kind: QWLayerPass, gdn: GdnW?, attn: AttnW?, moe: MoEW?,
             mhc: MHCLayerWeights? = nil, hca: QWHCAAttnOperator.LayerWeights? = nil,
             gatedNorms: GatedLayerNorms? = nil) {
            self.ln1 = ln1
            self.ln2 = ln2
            self.gu = gu
            self.down = down
            self.kind = kind
            self.gdn = gdn
            self.attn = attn
            self.hca = hca
            self.moe = moe
            self.mhc = mhc
            self.gatedNorms = gatedNorms
        }
    }
    /// The sparse-MoE chain's scratch. One set of shapes serves a decode step
    /// (one row) and a prefill chunk (`rows` rows), so the chain is one body.
    struct MoEScratch {
        let xRot32: QWTensorValue      // [rows · H] f32 — the rotated hidden
        let xPlain32: QWTensorValue    // [rows · H] f32 — the un-rotated hidden
        let logits: QWTensorValue      // [rows · E] f32 — the router's scores
        let idx: QWTensorValue         // [rows · K] i32 — the chosen experts
        let wts: QWTensorValue         // [rows · K] f32 — the router's weights
        let gu32: QWTensorValue        // [rows · K · 2I] f32 — the gate/up
        let act32: QWTensorValue       // [rows · K · I] f32 — the SwiGLU
        let guSh32: QWTensorValue?     // [rows · 2 · ISh] f32 — shared expert gate/up
        let actSh32: QWTensorValue?    // [rows · ISh] f32 — shared expert SwiGLU
        let shared: QWTensorValue      // [rows · H] f32 — shared expert down (or zeroed)
        let gate: QWTensorValue        // [rows] f32 — shared expert sigmoid gate (or zeroed)
        let expOff: QWTensorValue?     // [E + 1] i32 — expert offset table
        let sortedPair: QWTensorValue? // [rows · K] i32 — sorted pair index
        let sortedTok: QWTensorValue?  // [rows · K] i32 — sorted token index
        let invPerm: QWTensorValue?    // [rows · K] i32 — inverse permutation
        let tileE: QWTensorValue?      // [rows · K] i32 — tile expert index
        let tileR: QWTensorValue?      // [rows · K] i32 — tile row index
        let tileCount: QWTensorValue?  // [1] i32 — active tile count
        let gu16: QWTensorValue?       // [rows · K · 2I] f16 — NAX gate/up out
        let dnHi: QWTensorValue?       // [rows · K · I] f16 — intermediate SwiGLU
        let dnAcc: QWTensorValue       // [rows · K · outDim] f32 — route-ordered down contributions
        let naxX16: QWTensorValue?     // [rows · H] f16 — rotated hidden for NAX
        let latent: QWMoEOperator.LatentScratch?
        let activeExperts: QWTensorValue?
        let activeIndirect: QWTensorValue?

        init(xRot32: QWTensorValue, xPlain32: QWTensorValue,
             logits: QWTensorValue, idx: QWTensorValue, wts: QWTensorValue,
             gu32: QWTensorValue, act32: QWTensorValue,
             guSh32: QWTensorValue?, actSh32: QWTensorValue?,
             shared: QWTensorValue, gate: QWTensorValue,
             expOff: QWTensorValue?, sortedPair: QWTensorValue?,
             sortedTok: QWTensorValue?, invPerm: QWTensorValue?,
             tileE: QWTensorValue?, tileR: QWTensorValue?, tileCount: QWTensorValue?,
             gu16: QWTensorValue?, dnHi: QWTensorValue?, dnAcc: QWTensorValue,
             naxX16: QWTensorValue?, latent: QWMoEOperator.LatentScratch?,
             activeExperts: QWTensorValue? = nil, activeIndirect: QWTensorValue? = nil) {
            self.xRot32 = xRot32; self.xPlain32 = xPlain32
            self.logits = logits; self.idx = idx; self.wts = wts
            self.gu32 = gu32; self.act32 = act32
            self.guSh32 = guSh32; self.actSh32 = actSh32
            self.shared = shared; self.gate = gate
            self.expOff = expOff; self.sortedPair = sortedPair
            self.sortedTok = sortedTok; self.invPerm = invPerm
            self.tileE = tileE; self.tileR = tileR; self.tileCount = tileCount
            self.gu16 = gu16; self.dnHi = dnHi; self.dnAcc = dnAcc
            self.naxX16 = naxX16; self.latent = latent
            self.activeExperts = activeExperts; self.activeIndirect = activeIndirect
        }
    }
    struct Scratch {
        let x: QWTensorValue                             // residual stream [H] f16 (in-place)
        let ln1: QWTensorValue                           // input_layernorm out [H] f16
        let ln2: QWTensorValue                           // post_attn_layernorm out [H] f16
        let gs: QWTensorValue                            // group sums K=5120 [40] f32
        let gs48: QWTensorValue                          // group sums K=6144 [48] f32
        let gsI: QWTensorValue                           // group sums K=17408 [136] f32
        let gdnY: QWTensorValue                          // [16480] f32 (qkv ‖ z ‖ a ‖ b)
        let gdnY16: QWTensorValue                        // [16480] f16 (qkv ‖ z ‖ a ‖ b)
        let qkv32: QWTensorValue, qkv16: QWTensorValue
        let z32: QWTensorValue, z16: QWTensorValue
        let a32: QWTensorValue, b32: QWTensorValue
        let betas: QWTensorValue, glog: QWTensorValue
        let convY: QWTensorValue
        let oScr: QWTensorValue, oOut: QWTensorValue
        let attnY: QWTensorValue                         // [14336] f32 (q ‖ k ‖ v)
        let attnY16: QWTensorValue                       // [14336] f16 (q ‖ k ‖ v)
        let q32: QWTensorValue, k32: QWTensorValue, v32: QWTensorValue
        let q16: QWTensorValue, k16: QWTensorValue, v16: QWTensorValue
        let qR: QWTensorValue, kR: QWTensorValue
        let attOut: QWTensorValue
        let o32: QWTensorValue, o16: QWTensorValue       // attn/attn-out [H]
        let gu32: QWTensorValue, gu16: QWTensorValue
        let dn32: QWTensorValue, dn16: QWTensorValue
        let logits: QWTensorValue
        let keysHi: QWTensorValue, keysLo: QWTensorValue, token: QWTensorValue
        let partials: QWMRegion
        /// The sparse-MoE chain's scratch — nil when the model has no MoE layer.
        let moe: MoEScratch?
        let mhcStream: QWTensorValue?
        let mhcControls: QWTensorValue?
        let fgb16: QWTensorValue?

        init(x: QWTensorValue, ln1: QWTensorValue, ln2: QWTensorValue,
             gs: QWTensorValue, gs48: QWTensorValue, gsI: QWTensorValue,
             gdnY: QWTensorValue, gdnY16: QWTensorValue,
             qkv32: QWTensorValue, qkv16: QWTensorValue,
             z32: QWTensorValue, z16: QWTensorValue,
             a32: QWTensorValue, b32: QWTensorValue,
             betas: QWTensorValue, glog: QWTensorValue,
             convY: QWTensorValue, oScr: QWTensorValue, oOut: QWTensorValue,
             attnY: QWTensorValue, attnY16: QWTensorValue,
             q32: QWTensorValue, k32: QWTensorValue, v32: QWTensorValue,
             q16: QWTensorValue, k16: QWTensorValue, v16: QWTensorValue,
             qR: QWTensorValue, kR: QWTensorValue,
             attOut: QWTensorValue,
             o32: QWTensorValue, o16: QWTensorValue,
             gu32: QWTensorValue, gu16: QWTensorValue,
             dn32: QWTensorValue, dn16: QWTensorValue,
             logits: QWTensorValue,
             keysHi: QWTensorValue, keysLo: QWTensorValue, token: QWTensorValue,
             partials: QWMRegion, moe: MoEScratch?,
             mhcStream: QWTensorValue? = nil, mhcControls: QWTensorValue? = nil,
             fgb16: QWTensorValue? = nil) {
            self.x = x; self.ln1 = ln1; self.ln2 = ln2
            self.gs = gs; self.gs48 = gs48; self.gsI = gsI
            self.gdnY = gdnY; self.gdnY16 = gdnY16
            self.qkv32 = qkv32; self.qkv16 = qkv16
            self.z32 = z32; self.z16 = z16
            self.a32 = a32; self.b32 = b32
            self.betas = betas; self.glog = glog
            self.convY = convY; self.oScr = oScr; self.oOut = oOut
            self.attnY = attnY; self.attnY16 = attnY16
            self.q32 = q32; self.k32 = k32; self.v32 = v32
            self.q16 = q16; self.k16 = k16; self.v16 = v16
            self.qR = qR; self.kR = kR
            self.attOut = attOut
            self.o32 = o32; self.o16 = o16
            self.gu32 = gu32; self.gu16 = gu16
            self.dn32 = dn32; self.dn16 = dn16
            self.logits = logits
            self.keysHi = keysHi; self.keysLo = keysLo; self.token = token
            self.partials = partials; self.moe = moe
            self.mhcStream = mhcStream; self.mhcControls = mhcControls
            self.fgb16 = fgb16
        }
    }
    struct PrefillScratch {
        let maxChunk: Int
        let x: QWTensorValue                             // [maxChunk · H] f16
        let ln1: QWTensorValue                           // [maxChunk · H] f16
        let ln2: QWTensorValue                           // [maxChunk · H] f16
        let gdnY: QWTensorValue                          // [maxChunk · gdnTotalN] f32 (non-KDA)
        let attnY16: QWTensorValue                       // [maxChunk · attnTotalN] f16 (non-MLA)
        let qkv16: QWTensorValue                         // [maxChunk · qkvC] f16
        let z16: QWTensorValue                           // [maxChunk · zC] f16
        let betas: QWTensorValue, glog: QWTensorValue    // [maxChunk · vh / glogC] f32
        let convY: QWTensorValue                         // [maxChunk · qkvC] f16
        let oScr: QWTensorValue                          // [maxChunk · zC] f32
        let oOut: QWTensorValue                          // [maxChunk · zC] f16
        let attOut: QWTensorValue                        // [maxChunk · qC] f16
        let attV16: QWTensorValue                        // [(maxChunk + 32) · kvE] f16 (non-MLA)
        let attQR16: QWTensorValue                       // [(maxChunk + 32) · qOut] f16 (non-MLA)
        let attKR16: QWTensorValue                       // [(maxChunk + 32) · kvE] f16 (non-MLA)
        let U: QWTensorValue, W: QWTensorValue           // [maxChunk · zC] f32
        let A: QWTensorValue, B: QWTensorValue           // [maxChunk · zC] f32
        let o16: QWTensorValue                           // [maxChunk · H] f16
        let guRaw16: QWTensorValue                       // [maxChunk · max(2I, 2H)] f16
        let gu16: QWTensorValue                          // [maxChunk · I] f16
        let dn32: QWTensorValue, dn16: QWTensorValue     // [maxChunk · H] f32 / f16
        /// The sparse-MoE chain's scratch — nil when the model has no MoE layer.
        let moe: MoEScratch?
        let mhcStream: QWTensorValue?
        let mhcControls: QWTensorValue?
        let fgb16: QWTensorValue?

        init(maxChunk: Int, x: QWTensorValue, ln1: QWTensorValue, ln2: QWTensorValue,
             gdnY: QWTensorValue, attnY16: QWTensorValue,
             qkv16: QWTensorValue, z16: QWTensorValue,
             betas: QWTensorValue, glog: QWTensorValue,
             convY: QWTensorValue, oScr: QWTensorValue, oOut: QWTensorValue,
             attOut: QWTensorValue, attV16: QWTensorValue, attQR16: QWTensorValue, attKR16: QWTensorValue,
             U: QWTensorValue, W: QWTensorValue, A: QWTensorValue, B: QWTensorValue,
             o16: QWTensorValue, guRaw16: QWTensorValue, gu16: QWTensorValue,
             dn32: QWTensorValue, dn16: QWTensorValue,
             moe: MoEScratch?,
             mhcStream: QWTensorValue? = nil, mhcControls: QWTensorValue? = nil,
             fgb16: QWTensorValue? = nil) {
            self.maxChunk = maxChunk; self.x = x; self.ln1 = ln1; self.ln2 = ln2
            self.gdnY = gdnY; self.attnY16 = attnY16
            self.qkv16 = qkv16; self.z16 = z16
            self.betas = betas; self.glog = glog
            self.convY = convY; self.oScr = oScr; self.oOut = oOut
            self.attOut = attOut; self.attV16 = attV16; self.attQR16 = attQR16; self.attKR16 = attKR16
            self.U = U; self.W = W; self.A = A; self.B = B
            self.o16 = o16; self.guRaw16 = guRaw16; self.gu16 = gu16
            self.dn32 = dn32; self.dn16 = dn16
            self.moe = moe
            self.mhcStream = mhcStream; self.mhcControls = mhcControls
            self.fgb16 = fgb16
        }
    }
    struct GdnSlot {
        let recSlab: QWTensorValue
        let ringSlab: QWTensorValue
        let rec: [QWTensorValue]      // per-GDN-layer views into recSlab
        let ring: [QWTensorValue]     // per-GDN-layer views into ringSlab
        let trunkHidden: QWTensorValue?
        var dflashSlot: QWDFlashSessionSlot? = nil
        var dsparkSlot: QWDSparkDrafter.SessionSlot? = nil
    }
}

/// Per-step decode telemetry: host time inside cold-expert staging (wait + route readback + IO)
/// and the QW_DECODE_PROFILE=1 stage-boundary GPU profiler state.
struct QWDecodeTelemetry {
    var coldStageUs: Double = 0
    var coldWaitUs: Double = 0
    var coldIOUs: Double = 0
    var coldStaged = 0
    var profiler: QWGPUStageProfiler<String>?
    var stageTotals: [String: (us: Double, n: Int)] = [:]
    var profiledSteps = 0
    /// QW_PREFILL_PROFILE=1: cold-staging counters and wall clock at the chunk start.
    var prefillChunkStart: (stageUs: Double, waitUs: Double, ioUs: Double, staged: Int, t0: Date)?
}
