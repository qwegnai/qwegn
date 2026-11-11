import Metal
import QWAlloc
import QWCore
import QWGEMM
import QWHAL

// FlashNext hyper-connection mixer dispatches (Sources/QWMetal/subsystems/qwhc*.metal), f32 throughout.
extension QWGemm {
    /// xn = groupRMSNorm(x, groups of `hidden`, (1 + w)) for `rows` tokens of `width` = groups * hidden.
    public static func encodeFnHCNorm(_ enc: any MTLComputeCommandEncoder, hal: QWHal, x: QWTensorValue,
                                      weight: QWTensorValue, xn: QWTensorValue, rows: Int, width: Int,
                                      hidden: Int, eps: Float, xStride: Int? = nil) throws {
        if let xStride { precondition(xStride >= width) }
        enc.setComputePipelineState(try hal.pipeline(xStride == nil ? "qwg_hc_norm" : "qwg_hc_norm_strided"))
        enc.bindTensors(x, weight, xn, startIndex: 0)
        enc.setUInt32s(hidden, startIndex: 3)
        enc.setPod(eps, index: 4)
        enc.setUInt32s(width, startIndex: 5)
        if let xStride { enc.setUInt32s(xStride, startIndex: 6) }
        enc.dispatchThreadgroups(MTLSize(width: rows * (width / hidden), height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Fused hc_mix + 32-point WHT (mixed -> rotMixed) + optional fused hc.inject BF16 dot.
    public static func encodeFnHCMixRot(_ enc: any MTLComputeCommandEncoder, hal: QWHal, gate: QWTensorValue,
                                        xn: QWTensorValue, mixed: QWTensorValue, rotMixed: QWTensorValue,
                                        rows: Int, hidden: Int, hc: Int,
                                        rotMixed16: QWTensorValue? = nil,
                                        applySigmoid: Bool = false,
                                        skipRot32: Bool = false,
                                        injWeight: QWTensorValue? = nil,
                                        injOut: QWTensorValue? = nil,
                                        xnRecompute: (stream: QWTensorValue, normWeight: QWTensorValue)? = nil) throws {
        enc.setComputePipelineState(try hal.pipeline("qwg_hc_mix_h32"))
        enc.bindTensors(gate, xn, mixed, rotMixed, startIndex: 0)
        enc.setUInt32s(hidden, hc, startIndex: 4)
        (rotMixed16 ?? rotMixed).bind(enc, index: 6)
        let fuseInj = injWeight != nil && injOut != nil && hc <= 4
        let flags = (rotMixed16 != nil ? 1 : 0)
                  | (applySigmoid ? 2 : 0)
                  | (gate.shape.dtype == .f16 ? 4 : 0)
                  | (skipRot32 ? 8 : 0)
                  | (fuseInj ? 16 : 0)
                  | (xnRecompute != nil ? 32 : 0)
        enc.setUInt32s(flags, startIndex: 7)
        (injWeight ?? gate).bind(enc, index: 8)
        (injOut ?? mixed).bind(enc, index: 9)
        (xnRecompute?.stream ?? xn).bind(enc, index: 10)
        (xnRecompute?.normWeight ?? xn).bind(enc, index: 11)
        let groups32 = (hidden + 31) / 32
        let widthTG = fuseInj ? 1 : (groups32 + 7) / 8
        enc.dispatchThreadgroups(MTLSize(width: widthTG, height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Fused H_32(hcLow) + hc.up HQ4 [hc*hidden, lowrank] + sigmoid + 4-stream hc_mix + H_32(mixed -> rotMixed).
    public static func encodeFnHCUpMixRot(_ enc: any MTLComputeCommandEncoder, hal: QWHal, upWeight: QWWeightView,
                                          hcLow: QWTensorValue, xn: QWTensorValue, mixed: QWTensorValue,
                                          rotMixed: QWTensorValue, rows: Int, hidden: Int, hc: Int) throws {
        precondition(hidden % 32 == 0 && upWeight.k % 32 == 0 && upWeight.k <= 512, "HC up+mix+rot needs R <= 512")
        let name: String
        if upWeight.isIncumbentContainer {
            name = "qwg_hc_up_mix_h32_q4g_f32"
        } else {
            guard let aqName = QWGemmAQ.hcUpMixH32F32KernelName(bits: upWeight.bits, group: upWeight.groupSize,
                                                                isSHVQ: upWeight.isSHVQ) else {
                throw QWGemmError.badWeights("no affine-quantised hc up mix kernel for \(upWeight.bits)-bit group \(upWeight.groupSize)")
            }
            name = aqName
        }
        enc.setComputePipelineState(try hal.pipeline(name))
        enc.bindTensors(upWeight.blocks, hcLow, xn, mixed, rotMixed, startIndex: 0)
        enc.setUInt32s(hidden, hc, upWeight.k, startIndex: 5)
        enc.dispatchThreadgroups(MTLSize(width: hidden / 32, height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
}

extension QWGemm {
    /// HC stream norm and its 32-point rotation in one dispatch: xn = norm(x) * (1 + w), rot = WHT32(xn).
    public static func encodeFnHCNormRot(_ enc: any MTLComputeCommandEncoder, hal: QWHal, x: QWTensorValue, weight: QWTensorValue,
                                         xn: QWTensorValue, rot: QWTensorValue, rows: Int, width: Int, hidden: Int,
                                         eps: Float, rot16: QWTensorValue? = nil, rinvOnly: Bool = false) throws {
        precondition(hidden % 32 == 0 && width % hidden == 0, "fused HC norm+rotation needs hidden % 32 == 0")
        enc.setComputePipelineState(try hal.pipeline("qwg_hc_norm_h32"))
        enc.bindTensors(x, weight, xn, rot, startIndex: 0)
        enc.setUInt32s(hidden, startIndex: 4)
        enc.setPod(eps, index: 5)
        enc.setUInt32s(width, startIndex: 6)
        (rot16 ?? rot).bind(enc, index: 7)
        enc.setUInt32s((rot16 != nil ? 1 : 0) | (rinvOnly ? 2 : 0), startIndex: 8)
        enc.dispatchThreadgroups(MTLSize(width: rows * (width / hidden), height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Fused residual injection (xOut = base + src * inj) + HC stream norm + 32-point rotation in one dispatch.
    public static func encodeFnHCInjectNormRot(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                               base: QWTensorValue, src: QWTensorValue, inj: QWTensorValue,
                                               xOut: QWTensorValue, weight: QWTensorValue,
                                               xn: QWTensorValue, rot: QWTensorValue,
                                               rows: Int, width: Int, hidden: Int, eps: Float,
                                               rot16: QWTensorValue? = nil, rinvOnly: Bool = false) throws {
        precondition(hidden % 32 == 0 && width % hidden == 0, "fused HC inject+norm+rotation needs hidden % 32 == 0")
        enc.setComputePipelineState(try hal.pipeline("qwg_hc_inject_norm_h32"))
        enc.bindTensors(base, src, inj, xOut, weight, xn, rot, startIndex: 0)
        enc.setUInt32s(hidden, startIndex: 7)
        enc.setPod(eps, index: 8)
        enc.setUInt32s(width, startIndex: 9)
        (rot16 ?? rot).bind(enc, index: 10)
        enc.setUInt32s((rot16 != nil ? 1 : 0) | (rinvOnly ? 2 : 0), startIndex: 11)
        enc.dispatchThreadgroups(MTLSize(width: rows * (width / hidden), height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Fused silu(y / scale) + 32-point WHT over [rows, lowrank] low-rank HC activations.
    public static func encodeFnHCLowEpilogue(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                             yLow: QWTensorValue, rows: Int, lowrank: Int, scale: Float,
                                             out16: QWTensorValue? = nil) throws {
        precondition(lowrank % 32 == 0, "HC lowrank epilogue needs lowrank % 32 == 0")
        enc.setComputePipelineState(try hal.pipeline("qwg_hc_low_epilogue_f32"))
        yLow.bind(enc, index: 0)
        (out16 ?? yLow).bind(enc, index: 1)
        enc.setUInt32s(rows, lowrank, startIndex: 2)
        enc.setPod(scale, index: 4)
        enc.setUInt32s(out16 != nil ? 1 : 0, startIndex: 5)
        enc.dispatchThreadgroups(MTLSize(width: lowrank / 32, height: (rows + 7) / 8, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Single-row HC inject+norm+rotation selecting the longest accepted draft prefix in `0..<totalRows`.
    public static func encodeFnHCInjectNormRotSel(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                                  base: QWTensorValue, src: QWTensorValue, inj: QWTensorValue,
                                                  xOut: QWTensorValue, weight: QWTensorValue,
                                                  xn: QWTensorValue, rot: QWTensorValue,
                                                  verifiedTok: QWTensorValue, draftTok: QWTensorValue,
                                                  width: Int, hidden: Int, eps: Float,
                                                  totalRows: Int = 2) throws {
        precondition(hidden % 32 == 0 && width % hidden == 0, "fused HC inject+norm+rotation needs hidden % 32 == 0")
        enc.setComputePipelineState(try hal.pipeline("qwg_hc_inject_norm_h32_sel"))
        enc.bindTensors(base, src, inj, xOut, weight, xn, rot, startIndex: 0)
        enc.setUInt32s(hidden, startIndex: 7)
        enc.setPod(eps, index: 8)
        enc.setUInt32s(width, startIndex: 9)
        enc.bindTensors(verifiedTok, draftTok, startIndex: 10)
        enc.setUInt32s(totalRows, startIndex: 12)
        enc.dispatchThreadgroups(MTLSize(width: width / hidden, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Fused decode (T=1) hc.down HQ4 [lowrank, width] + hc.inject BF16 [hc, width] in one 128-thread dispatch.
    public static func encodeFnHCDownInjectDecode(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                                  downWeight: QWWeightView, xRot: QWTensorValue, yLow: QWTensorValue,
                                                  injWeight: QWTensorValue, xXn: QWTensorValue, yInj: QWTensorValue,
                                                  hc: Int, scale: Float, rows: Int = 1) throws {
        let nDownTG = (downWeight.n + 3) / 4
        let name: String
        if downWeight.isIncumbentContainer {
            name = "qwg_hc_down_inject_f32"
        } else {
            guard let aqName = QWGemmAQ.hcDownInjectF32KernelName(bits: downWeight.bits, group: downWeight.groupSize,
                                                                  isSHVQ: downWeight.isSHVQ) else {
                throw QWGemmError.badWeights("no affine-quantised hc down inject kernel for \(downWeight.bits)-bit group \(downWeight.groupSize)")
            }
            name = aqName
        }
        enc.setComputePipelineState(try hal.pipeline(name))
        enc.bindTensors(downWeight.blocks, xRot, yLow, injWeight, xXn, yInj, startIndex: 0)
        enc.setUInt32s(downWeight.n, downWeight.k, startIndex: 6)
        enc.setPod(scale, index: 8)
        enc.setUInt32s(hc, startIndex: 9)
        enc.dispatchThreadgroups(MTLSize(width: nDownTG + hc, height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    /// Decode (T <= 8) HC norm (+ optional `xOut = base + src * inj`) + rotation + down + inject in ONE dispatch.
    /// Returns false (encoding nothing) when the down bank has no fused kernel or `xOut` aliases `x`.
    public static func encodeFnHCNormDownInjectDecode(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                                      downWeight: QWWeightView, x: QWTensorValue,
                                                      src: QWTensorValue?, inj: QWTensorValue?, xOut: QWTensorValue?,
                                                      normWeight: QWTensorValue, xn: QWTensorValue, yLow: QWTensorValue,
                                                      injWeight: QWTensorValue, yInj: QWTensorValue,
                                                      width: Int, hidden: Int, eps: Float, hc: Int, scale: Float,
                                                      rows: Int) throws -> Bool {
        guard !downWeight.isIncumbentContainer, hc <= 16, hidden % 32 == 0, width % hidden == 0,
              width / hidden <= 4, downWeight.k == width,
              let name = QWGemmAQ.hcNormDownInjectF32KernelName(bits: downWeight.bits, group: downWeight.groupSize,
                                                                isSHVQ: downWeight.isSHVQ)
        else { return false }
        let injecting = src != nil && inj != nil && xOut != nil
        if injecting, let o = xOut, o.region.buffer === x.region.buffer {
            let a0 = o.region.base + o.offset, b0 = x.region.base + x.offset
            let len = rows * width * 4
            if a0 < b0 + len && b0 < a0 + len { return false }
        }
        enc.setComputePipelineState(try hal.pipeline(name))
        enc.bindTensors(downWeight.blocks, x, src ?? x, inj ?? x, xOut ?? xn, normWeight, xn, yLow, injWeight, yInj,
                        startIndex: 0)
        enc.setUInt32s(downWeight.n, width, hidden, startIndex: 10)
        enc.setPod(eps, index: 13)
        enc.setPod(scale, index: 14)
        enc.setUInt32s(hc, injecting ? 1 : 0, startIndex: 15)
        enc.dispatchThreadgroups(MTLSize(width: (downWeight.n + 3) / 4 + hc, height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        return true
    }
}
