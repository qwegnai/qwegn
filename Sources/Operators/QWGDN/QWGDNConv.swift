// QWGDNConv.swift — engine-side encoder for the C11 causal depthwise conv1d.
//
// Mirrors qwgdn_conv.metal exactly: grid = (5 ch-bands × ⌈T/32⌉ token-bands)
// × 256 threads (dispatchThreadgroups — dispatchThreads would dispatch
// THREADS and leave bands inactive, the C08 law). Buffers are f16 engine
// storage; the conv state is the RAW pre-SiLU rows in age order (row 0 =
// oldest = padded[0]); stIn and stOut may be the SAME buffer only if the
import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc

public enum QWGDNConvError: Error, CustomStringConvertible {
    case badShape(String)
    public var description: String { switch self { case .badShape(let s): return "QWGDNConvError badShape: \(s)" } }
}

/// Encoder for `qwg_gdn_conv`. Stateless after init.
public final class QWGDNConv {
    public static let channels = QWModelGeometry.qwen38_27b.gdnQKVChannels      // 10240
    public static let stateRows = 3
    static let chPerBand = 256           // 32 lanes x 8 channels, 40 bands (all GPU cores busy at small T)
    static let tokBand = 32

    public let geometry: QWModelGeometry
    private let hal: QWHal
    private let pso: any MTLComputePipelineState
    private let psoT1Headnorm: any MTLComputePipelineState
    private let psoRowsHeadnorm: any MTLComputePipelineState
    private let psoHeadnormPrefill: any MTLComputePipelineState
    private let psoRotate: any MTLComputePipelineState
    private let zeroSlab: MTLBuffer                    // optional-bias & zero-ring stand-in
    public let zeroRing: QWTensorValue

    public init(hal: QWHal, geometry: QWModelGeometry) throws {
        self.geometry = geometry
        self.hal = hal
        self.pso = try hal.pipeline("qwg_gdn_conv")
        self.psoT1Headnorm = try hal.pipeline("qwg_gdn_conv_headnorm_t1")
        self.psoRowsHeadnorm = try hal.pipeline("qwg_gdn_conv_headnorm_rows")
        self.psoHeadnormPrefill = try hal.pipeline("qwg_gdn_conv_headnorm")
        self.psoRotate = try hal.pipeline("qwg_gdn_conv_rotate")
        // Metal does not zero-initialise new buffers: the nil-bias & pos==0 ring paths read this slab, so it must be cleared.
        let ch = max(1, geometry.gdnQKVChannels)
        let ringBytes = QWGDNConv.stateRows * ch * 2
        let slab = hal.device.makeBuffer(length: ringBytes, options: .storageModeShared)!
        memset(slab.contents(), 0, ringBytes)
        self.zeroSlab = slab
        self.zeroRing = QWTensorValue(region: QWMRegion(buffer: slab, base: 0, length: ringBytes),
                                      offset: 0,
                                      shape: QWTensorShape([QWGDNConv.stateRows * ch], .f16))
    }

    /// Bytes per tensor role (f16 units): x/y = T·channels, state = 3·channels.
    public static func sizes(T: Int, channels: Int = QWGDNConv.channels) -> (xy: Int, state: Int, w: Int, bias: Int) {
        (T * channels, stateRows * channels, channels * 4, channels)
    }

    private var instChannels: Int { max(1, geometry.gdnQKVChannels) }

    private func validateStateAliasing(_ input: QWTensorValue, _ output: QWTensorValue) throws -> Bool {
        guard input.region.buffer === output.region.buffer else { return false }
        let a = input.region.base + input.offset, b = output.region.base + output.offset
        guard a == b || a + input.bytes <= b || b + output.bytes <= a else {
            throw QWGDNConvError.badShape("partially overlapping state views")
        }
        return a == b
    }

    private func rotate(_ enc: any MTLComputeCommandEncoder, x: QWTensorValue,
                        input: QWTensorValue, output: QWTensorValue, T: Int) {
        enc.memoryBarrier(scope: .buffers)
        enc.setComputePipelineState(psoRotate)
        x.bind(enc, index: 0)
        input.bind(enc, index: 1)
        output.bind(enc, index: 2)
        var count = UInt32(T)
        enc.setBytes(&count, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: instChannels, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.memoryBarrier(scope: .buffers)
    }

    /// Encode one conv+silu+rotate. `bias` may be nil → zero slab.
    public func encode(_ enc: any MTLComputeCommandEncoder,
                       x: QWTensorValue, w: QWTensorValue, bias: QWTensorValue?,
                       stIn: QWTensorValue,
                       y: QWTensorValue, stOut: QWTensorValue,
                       T: Int) throws {
        let ch = instChannels
        let s = Self.sizes(T: T, channels: ch)
        guard T > 0,
              x.shape.elementCount == s.xy, y.shape.elementCount == s.xy,
              stIn.shape.elementCount == s.state, stOut.shape.elementCount == s.state,
              w.shape.elementCount == s.w,
              x.shape.dtype == .f16, y.shape.dtype == .f16,
              stIn.shape.dtype == .f16, stOut.shape.dtype == .f16
        else { throw QWGDNConvError.badShape("T=\(T): x/y \(x.shape.elementCount)/\(y.shape.elementCount) need \(s.xy)") }
        let inplace = try validateStateAliasing(stIn, stOut)

        enc.setComputePipelineState(pso)
        x.bind(enc, index: 0)
        w.bind(enc, index: 1)
        if let b = bias { b.bind(enc, index: 2) }
        else { enc.setBuffer(zeroSlab, offset: 0, index: 2) }
        stIn.bind(enc, index: 3)
        y.bind(enc, index: 4)
        stOut.bind(enc, index: 5)
        var tU = UInt32(T)
        enc.setBytes(&tU, length: 4, index: 6)
        let bands = (T + Self.tokBand - 1) / Self.tokBand
        enc.dispatchThreadgroups(
            MTLSize(width: ch / Self.chPerBand, height: bands, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        if inplace && T > Self.tokBand { rotate(enc, x: x, input: stIn, output: stOut, T: T) }
    }

    /// Encode T=1 causal conv1d with fused HeadNorm for Q and K heads.
    /// Eliminates standalone psoHeadnorm dispatch and avoids writing unnormalized conv bytes to DRAM.
    public func encodeT1WithHeadnorm(_ enc: any MTLComputeCommandEncoder,
                                     x: QWTensorValue, w: QWTensorValue, bias: QWTensorValue?,
                                     stIn: QWTensorValue,
                                     y: QWTensorValue, stOut: QWTensorValue) throws {
        let ch = instChannels
        let s = Self.sizes(T: 1, channels: ch)
        guard x.shape.elementCount == s.xy, y.shape.elementCount == s.xy,
              stIn.shape.elementCount == s.state, stOut.shape.elementCount == s.state,
              w.shape.elementCount == s.w,
              x.shape.dtype == .f16, y.shape.dtype == .f16,
              stIn.shape.dtype == .f16, stOut.shape.dtype == .f16
        else { throw QWGDNConvError.badShape("T=1: x/y \(x.shape.elementCount)/\(y.shape.elementCount) need \(s.xy)") }
        _ = try validateStateAliasing(stIn, stOut)

        enc.setComputePipelineState(psoT1Headnorm)
        x.bind(enc, index: 0)
        w.bind(enc, index: 1)
        if let b = bias { b.bind(enc, index: 2) }
        else { enc.setBuffer(zeroSlab, offset: 0, index: 2) }
        stIn.bind(enc, index: 3)
        y.bind(enc, index: 4)
        stOut.bind(enc, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: ch / 128, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    /// Encode R verification rows of causal conv1d with fused HeadNorm + replay capture.
    public func encodeRowsWithHeadnorm(_ enc: any MTLComputeCommandEncoder,
                                       x: QWTensorValue, w: QWTensorValue, bias: QWTensorValue?,
                                       stIn: QWTensorValue,
                                       y: QWTensorValue, stOut: QWTensorValue,
                                       rows: Int, xStride: Int? = nil,
                                       rawCapture: QWTensorValue? = nil,
                                       controls: QWTensorValue? = nil,
                                       betasCapture: QWTensorValue? = nil,
                                       glogCapture: QWTensorValue? = nil) throws {
        let ch = instChannels
        let effectiveStride = xStride ?? ch
        let s = Self.sizes(T: rows, channels: ch)
        guard rows > 0, effectiveStride >= ch,
              x.shape.elementCount >= (rows - 1) * effectiveStride + ch,
              y.shape.elementCount == s.xy,
              stIn.shape.elementCount == s.state, stOut.shape.elementCount == s.state,
              w.shape.elementCount == s.w,
              x.shape.dtype == .f16, y.shape.dtype == .f16,
              stIn.shape.dtype == .f16, stOut.shape.dtype == .f16
        else { throw QWGDNConvError.badShape("rows=\(rows): x/y \(x.shape.elementCount)/\(y.shape.elementCount)") }
        _ = try validateStateAliasing(stIn, stOut)

        enc.setComputePipelineState(psoRowsHeadnorm)
        x.bind(enc, index: 0)
        w.bind(enc, index: 1)
        if let b = bias { b.bind(enc, index: 2) }
        else { enc.setBuffer(zeroSlab, offset: 0, index: 2) }
        stIn.bind(enc, index: 3)
        y.bind(enc, index: 4)
        stOut.bind(enc, index: 5)
        var rU = UInt32(rows), strideU = UInt32(effectiveStride)
        enc.setBytes(&rU, length: 4, index: 6)
        enc.setBytes(&strideU, length: 4, index: 7)
        if let rawCapture { rawCapture.bind(enc, index: 8) }
        else { enc.setBuffer(nil, offset: 0, index: 8) }
        if let controls, let betasCapture, let glogCapture {
            controls.bind(enc, index: 9)
            betasCapture.bind(enc, index: 10)
            glogCapture.bind(enc, index: 11)
        } else {
            enc.setBuffer(nil, offset: 0, index: 9)
        }
        enc.dispatchThreadgroups(MTLSize(width: ch / 128, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    /// Encode prefill causal conv1d with fused HeadNorm for Q and K heads.
    /// Eliminates standalone psoHeadnorm dispatch and avoids writing unnormalized conv bytes to DRAM.
    public func encodePrefillWithHeadnorm(_ enc: any MTLComputeCommandEncoder,
                                          x: QWTensorValue, w: QWTensorValue, bias: QWTensorValue?,
                                          stIn: QWTensorValue,
                                          y: QWTensorValue, stOut: QWTensorValue,
                                          T: Int,
                                          gdnY32: QWTensorValue? = nil,
                                          aLog: QWTensorValue? = nil,
                                          dtBias: QWTensorValue? = nil,
                                          betas: QWTensorValue? = nil,
                                          glog: QWTensorValue? = nil) throws {
        let ch = instChannels
        let s = Self.sizes(T: T, channels: ch)
        guard T > 0,
              x.shape.elementCount == s.xy, y.shape.elementCount == s.xy,
              stIn.shape.elementCount == s.state, stOut.shape.elementCount == s.state,
              w.shape.elementCount == s.w,
              x.shape.dtype == .f16, y.shape.dtype == .f16,
              stIn.shape.dtype == .f16, stOut.shape.dtype == .f16
        else { throw QWGDNConvError.badShape("T=\(T): x/y \(x.shape.elementCount)/\(y.shape.elementCount) need \(s.xy)") }
        let inplace = try validateStateAliasing(stIn, stOut)

        enc.setComputePipelineState(psoHeadnormPrefill)
        x.bind(enc, index: 0)
        w.bind(enc, index: 1)
        if let b = bias { b.bind(enc, index: 2) }
        else { enc.setBuffer(zeroSlab, offset: 0, index: 2) }
        stIn.bind(enc, index: 3)
        y.bind(enc, index: 4)
        stOut.bind(enc, index: 5)
        var tU = UInt32(T)
        enc.setBytes(&tU, length: 4, index: 6)
        if let gdnY32, let aLog, let dtBias, let betas, let glog {
            gdnY32.bind(enc, index: 7)
            aLog.bind(enc, index: 8)
            dtBias.bind(enc, index: 9)
            betas.bind(enc, index: 10)
            glog.bind(enc, index: 11)
        } else {
            enc.setBuffer(nil, offset: 0, index: 7)
        }
        let bands = (T + Self.tokBand - 1) / Self.tokBand
        enc.dispatchThreadgroups(
            MTLSize(width: ch / Self.chPerBand, height: bands, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        if inplace && T > Self.tokBand { rotate(enc, x: x, input: stIn, output: stOut, T: T) }
    }
}
