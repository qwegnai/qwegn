// QWDecisionMetalTypes.swift — Layer specs, weight containers, and tensor builder for Metal decision models.

import Foundation
import Metal
import QWAlloc
import QWCore
import QWDecision
import QWHAL
import QWHead

public enum QWDecisionNormKind: Sendable, Hashable {
    case none
    case layerNorm(eps: Float)
    case rmsNorm(eps: Float, unitOffset: Bool)
}

public enum QWDecisionBlockKind: Sendable, Hashable {
    case shortConvCausal(channels: Int, kernelSize: Int, hasBias: Bool, hasInProj: Bool)
    case shortConvCentered(channels: Int, kernelSize: Int, hasBias: Bool, hasInProj: Bool)
    case attention(
        topology: QWDecisionModelConfig.AttentionTopology,
        qHeads: Int,
        kvHeads: Int,
        headDim: Int,
        rope: PerKindRoPEEntry?,
        zeroTail: Bool,
        hasSigmoidGate: Bool,
        scale: Float?
    )
}

public struct QWDecisionLayerSpec: Sendable, Hashable {
    public let hidden: Int
    public let ffnDim: Int
    public let block: QWDecisionBlockKind
    public let preBlockNorm: QWDecisionNormKind
    public let postBlockNorm: QWDecisionNormKind
    public let preFFNNorm: QWDecisionNormKind
    public let postFFNNorm: QWDecisionNormKind
    public let ffnActivation: QWDecisionHeadActivation

    public init(
        hidden: Int,
        ffnDim: Int,
        block: QWDecisionBlockKind,
        preBlockNorm: QWDecisionNormKind,
        postBlockNorm: QWDecisionNormKind = .none,
        preFFNNorm: QWDecisionNormKind,
        postFFNNorm: QWDecisionNormKind = .none,
        ffnActivation: QWDecisionHeadActivation = .gelu
    ) {
        self.hidden = hidden
        self.ffnDim = ffnDim
        self.block = block
        self.preBlockNorm = preBlockNorm
        self.postBlockNorm = postBlockNorm
        self.preFFNNorm = preFFNNorm
        self.postFFNNorm = postFFNNorm
        self.ffnActivation = ffnActivation
    }
}

public struct QWDecisionNormWeights: Sendable {
    public let weight: QWTensorValue
    public let bias: QWTensorValue?

    public init(weight: QWTensorValue, bias: QWTensorValue? = nil) {
        self.weight = weight
        self.bias = bias
    }
}

public struct QWDecisionShortConvWeights: Sendable {
    public let inProjW: QWTensorValue?
    public let inProjB: QWTensorValue?
    public let convW: QWTensorValue
    public let convB: QWTensorValue?
    public let outProjW: QWTensorValue
    public let outProjB: QWTensorValue?

    public init(
        inProjW: QWTensorValue? = nil,
        inProjB: QWTensorValue? = nil,
        convW: QWTensorValue,
        convB: QWTensorValue? = nil,
        outProjW: QWTensorValue,
        outProjB: QWTensorValue? = nil
    ) {
        self.inProjW = inProjW
        self.inProjB = inProjB
        self.convW = convW
        self.convB = convB
        self.outProjW = outProjW
        self.outProjB = outProjB
    }
}

public struct QWDecisionAttentionWeights: Sendable {
    public let qW: QWTensorValue
    public let qB: QWTensorValue?
    public let kW: QWTensorValue
    public let kB: QWTensorValue?
    public let vW: QWTensorValue
    public let vB: QWTensorValue?
    public let oW: QWTensorValue
    public let oB: QWTensorValue?
    public let gateW: QWTensorValue?
    public let gateB: QWTensorValue?

    public init(
        qW: QWTensorValue, qB: QWTensorValue? = nil,
        kW: QWTensorValue, kB: QWTensorValue? = nil,
        vW: QWTensorValue, vB: QWTensorValue? = nil,
        oW: QWTensorValue, oB: QWTensorValue? = nil,
        gateW: QWTensorValue? = nil, gateB: QWTensorValue? = nil
    ) {
        self.qW = qW; self.qB = qB
        self.kW = kW; self.kB = kB
        self.vW = vW; self.vB = vB
        self.oW = oW; self.oB = oB
        self.gateW = gateW; self.gateB = gateB
    }
}

public struct QWDecisionLayerWeights: Sendable {
    public let spec: QWDecisionLayerSpec
    public let preBlockNorm: QWDecisionNormWeights?
    public let postBlockNorm: QWDecisionNormWeights?
    public let preFFNNorm: QWDecisionNormWeights?
    public let postFFNNorm: QWDecisionNormWeights?
    public let shortConv: QWDecisionShortConvWeights?
    public let attention: QWDecisionAttentionWeights?
    public let ffnUpW: QWTensorValue
    public let ffnUpB: QWTensorValue?
    public let ffnDownW: QWTensorValue
    public let ffnDownB: QWTensorValue?

    public init(
        spec: QWDecisionLayerSpec,
        preBlockNorm: QWDecisionNormWeights?,
        postBlockNorm: QWDecisionNormWeights? = nil,
        preFFNNorm: QWDecisionNormWeights?,
        postFFNNorm: QWDecisionNormWeights? = nil,
        shortConv: QWDecisionShortConvWeights? = nil,
        attention: QWDecisionAttentionWeights? = nil,
        ffnUpW: QWTensorValue,
        ffnUpB: QWTensorValue? = nil,
        ffnDownW: QWTensorValue,
        ffnDownB: QWTensorValue? = nil
    ) {
        self.spec = spec
        self.preBlockNorm = preBlockNorm
        self.postBlockNorm = postBlockNorm
        self.preFFNNorm = preFFNNorm
        self.postFFNNorm = postFFNNorm
        self.shortConv = shortConv
        self.attention = attention
        self.ffnUpW = ffnUpW
        self.ffnUpB = ffnUpB
        self.ffnDownW = ffnDownW
        self.ffnDownB = ffnDownB
    }
}

/// Shared-storage GPU tensor builder for decision models & parity harnesses.
public final class QWDecisionTensorBuilder: @unchecked Sendable {
    private let hal: QWHal
    private var retainedBuffers: [any MTLBuffer] = []

    public init(hal: QWHal) {
        self.hal = hal
    }

    public func makeF16FromFloats(_ values: [Float], shape: [Int]) throws -> QWTensorValue {
        try makeRawTensor(values.map { Float16($0).bitPattern }, shape: shape, dtype: .f16)
    }

    public func makeF16FromU16(_ bits: [UInt16], shape: [Int]) throws -> QWTensorValue {
        try makeRawTensor(bits, shape: shape, dtype: .f16)
    }

    public func makeBF16FromU16(_ bits: [UInt16], shape: [Int]) throws -> QWTensorValue {
        try makeRawTensor(bits, shape: shape, dtype: .bf16)
    }

    public func makeF32(_ values: [Float], shape: [Int]) throws -> QWTensorValue {
        try makeRawTensor(values, shape: shape, dtype: .f32)
    }

    public func makeI32(_ values: [Int32], shape: [Int]) throws -> QWTensorValue {
        try makeRawTensor(values, shape: shape, dtype: .i32)
    }

    public func makeEmptyTensor(shape: [Int], dtype: QWDType) throws -> QWTensorValue {
        let tShape = QWTensorShape(shape, dtype)
        let bytes = max(64, qwAlignUp(tShape.totalBytes, 64))
        guard let buf = hal.device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw QWDecisionError.backendFailed("failed to allocate Metal tensor of \(bytes) bytes")
        }
        memset(buf.contents(), 0, bytes)
        retainedBuffers.append(buf)
        return QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: bytes), offset: 0, shape: tShape)
    }

    private func makeRawTensor<T>(_ items: [T], shape: [Int], dtype: QWDType) throws -> QWTensorValue {
        let tShape = QWTensorShape(shape, dtype)
        guard items.count == tShape.elementCount else {
            throw QWDecisionError.invalidQuestion("tensor count \(items.count) != shape \(shape)")
        }
        let byteLen = items.count * MemoryLayout<T>.stride
        let allocLen = max(64, qwAlignUp(byteLen, 64))
        guard let buf = hal.device.makeBuffer(length: allocLen, options: .storageModeShared) else {
            throw QWDecisionError.backendFailed("failed to allocate shared Metal buffer")
        }
        memset(buf.contents(), 0, allocLen)
        items.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            if let base = raw.baseAddress, byteLen > 0 {
                memcpy(buf.contents(), base, byteLen)
            }
        }
        retainedBuffers.append(buf)
        return QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: allocLen), offset: 0, shape: tShape)
    }
}
