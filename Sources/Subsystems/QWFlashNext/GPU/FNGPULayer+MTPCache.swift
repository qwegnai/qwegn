import Metal
import QWAlloc

extension FNGPULayerPasses {
    static func mtpPrefillCache(_ enc: any MTLComputeCommandEncoder, ctx: FNGPUContext,
                                layer: FNGPULayerWeights, s: FNGPUScratch, stream: QWTensorValue,
                                state: FNGPUAttnDecodeState, pos: Int,
                                inInject: (base: QWTensorValue, src: QWTensorValue, scale: QWTensorValue),
                                lastRowScratch: FNGPUScratch? = nil) throws {
        guard layer.ple == nil, case .attention(let attention) = layer.mixer else {
            throw FNGPUError.notHQ4("MTP cache-only prefill requires an attention layer without PLE")
        }
        try hcMix(enc, hal: ctx.hal, hc: layer.attnHC, s: s, stream: stream,
                  mixed: s.mixedA, inj: s.injA,
                  injectBase: inInject.base, injectSrc: inInject.src, injectScale: inInject.scale)
        try attentionPrefill(enc, ctx: ctx, a: attention, c: state, s: s, pos: pos, out: s.oG,
                             cacheOnly: lastRowScratch == nil, lastRowOnly: lastRowScratch != nil)
        if let tail = lastRowScratch {
            let r = s.T - 1
            let streamLast = view(stream, byteOffset: r * s.width * 4, count: s.width, dtype: .f32)
            let oGLast = view(s.oG, byteOffset: r * s.hidden * 4, count: s.hidden, dtype: .f32)
            let injALast = view(s.injA, byteOffset: r * s.hc * 4, count: s.hc, dtype: .f32)
            try hcMix(enc, hal: ctx.hal, hc: layer.mlpHC, s: tail, stream: tail.h2,
                      mixed: tail.mixedB, inj: tail.injB,
                      injectBase: streamLast, injectSrc: oGLast, injectScale: injALast, rows: 1)
            try moe(enc, ctx: ctx, m: layer.moe, s: tail, mixed: tail.mixedB, out: tail.moeOut, rows: 1)
        }
    }
}
