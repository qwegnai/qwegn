// DriverWeightLoadingAQ.swift — the size-preserving family's loader route.
//
// A checkpoint kept in its own bytes is READ in its own bytes: the payload
// is copied verbatim and the headers are written from the checkpoint's own
// scale and bias. Nothing is inflated and no weight is converted to a float
// on the way in. The design is in specs/04-qwquant.md.
import Foundation
import Darwin
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWLoad
import QWQuant

extension QWEngineDriver.WeightLoader {

    // MARK: - the planes

    /// The plane for one weight: the incumbent's container, or the family's.
    mutating func planeView(_ name: String, n: Int, k: Int,
                             key: KeyPath<QWQTensorEntry, Int?>) throws -> QWTensorValue {
        if format.isNativeAffine {
            return try aqPlaneView(name, n: n, k: k)
        }
        let name = resolveName(name)
        let elems: Int
        switch key {
        case \QWQTensorEntry.hq4Off, \QWQTensorEntry.q4gOff:
            if let pk {
                let entry = try pk.entry(name)
                if let expCount = entry.experts, let per = entry.eHq4 {
                    elems = (expCount * per) / 2
                } else {
                    elems = (n * (k / entry.hqGroup) * entry.hqBlockBytes) / 2
                }
            } else {
                elems = (n * format.rowBytes(k: k)) / 2
            }
        default:
            throw QWEngineDriverError.weightMissing(name, "plane key unsupported")
        }
        let t = try allocateWeight(QWTensorShape([elems], .f16))
        if let io {
            let dst = t.region.buffer.contents().advanced(by: t.region.base + t.offset)
            try io.load(name, layout: .hq4, into: dst)
        } else {
            try quantizeSafetensorsToHQ4(
                name, n: n, k: k,
                into: t.region.buffer,
                dstByteOffset: t.region.base + t.offset
            )
        }
        return t
    }

    /// The bank's view, carrying the container's own pair and the model's basis.
    mutating func q4(_ name: String, n: Int, k: Int) throws -> QWWeightView {
        let blocks = try planeView(name, n: n, k: k, key: \QWQTensorEntry.hq4Off)
        let resolved = resolveName(name)
        let entry = try? pk?.entry(resolved)
        let bits = entry?.hqBits ?? format.bits
        let groupSize = entry?.hqGroup ?? format.groupSize
        let isIncumbent = entry?.isIncumbentHQ4 ?? format.denseIsIncumbentContainer
        let isSHVQ = entry?.isSHVQ ?? format.denseIsSHVQ
        guard let v = QWWeightView(blocks: blocks, n: n, k: k,
                                    bits: bits, groupSize: groupSize,
                                    isRotated: hl.weightBasis.rotatesActivations,
                                    isIncumbentContainer: isIncumbent,
                                    isSHVQ: isSHVQ) else {
            throw QWEngineDriverError.weightMissing(name, "QWWeightView layout invalid")
        }
        // The bank's basis must be the MODEL's basis: the bank and the
        // activations meet in one GEMV, and a rotated weight read by an
        // un-rotated activation is a different number, not a rounding error.
        guard v.isRotated == hl.weightBasis.rotatesActivations else {
            throw QWEngineDriverError.badConfig(
                "weight '\(name)' is in the \(v.isRotated ? "rotated" : "standard") "
                + "basis but the model is \(hl.weightBasis)")
        }
        return v
    }

    // MARK: - the family's repack

    /// The family's plane: the checkpoint's own bytes, repacked on the GPU.
    mutating func aqPlaneView(_ name: String, n: Int, k: Int) throws -> QWTensorValue {
        let elems = (n * format.rowBytes(k: k)) / 2
        let t = try allocateWeight(QWTensorShape([elems], .f16))
        try repackSafetensorsToAQ(name, n: n, k: k,
                                  into: t.region.buffer, dstByteOffset: t.region.base + t.offset)
        return t
    }

    /// The family's concatenated plane: each tensor repacked into its own row
    /// slice. The family's rows are uniform, so the slices are the same bytes
    /// the incumbent's are — only the row size differs.
    mutating func aqConcatPlane(_ tensors: [(name: String, n: Int)], k: Int) throws -> QWTensorValue {
        let rowBytes = format.rowBytes(k: k)
        let totalN = tensors.reduce(0) { $0 + $1.n }
        let t = try allocateWeight(QWTensorShape([(totalN * rowBytes) / 2], .f16))
        var row0 = 0
        for (name, n) in tensors {
            try repackSafetensorsToAQ(name, n: n, k: k, into: t.region.buffer,
                                      dstByteOffset: t.region.base + t.offset + row0 * rowBytes)
            row0 += n
        }
        return t
    }

    /// Repacks one MLX affine tensor into the family's container, on the GPU.
    /// The repack itself is QWLoad's door — the one both engines use — and
    /// this is the driver's name resolution and the model's own pair.
    func repackSafetensorsToAQ(_ name: String, n: Int, k: Int,
                                into dstBuffer: any MTLBuffer, dstByteOffset: Int) throws {
        guard let st else {
            throw QWEngineDriverError.weightMissing(name, "the family is a safetensors route only")
        }
        let resolved = resolveName(name)
        do {
            try QWAQRepacker.repackMLXAffine(hal: hl, container: st, name: resolved,
                                             rows: n, cols: k,
                                             bits: format.bits, group: format.groupSize,
                                             into: dstBuffer, dstByteOffset: dstByteOffset)
        } catch let e as QWAQRepacker.RepackError {
            throw QWEngineDriverError.weightMissing(name, e.description)
        }
    }

    /// Repacks 3D `[experts, i, h]` `switch_mlp` gate and up projections into
    /// interleaved `[experts, 2 * i, h]` `AQ` blocks on the GPU.
    func repackSafetensorsSwitchGateUpToAQ(
        gateName: String, upName: String,
        experts: Int, i: Int, h: Int,
        into dstBuffer: any MTLBuffer, dstByteOffset: Int
    ) throws {
        guard let st else {
            throw QWEngineDriverError.weightMissing(gateName, "the family is a safetensors route only")
        }
        let resolvedGate = resolveName(gateName)
        let resolvedUp = resolveName(upName)
        do {
            try QWAQRepacker.repackMLXAffineSwitchGateUp(
                hal: hl, container: st,
                gateName: resolvedGate, upName: resolvedUp,
                experts: experts, inter: i, cols: h,
                bits: format.bits, group: format.groupSize,
                into: dstBuffer, dstByteOffset: dstByteOffset
            )
        } catch let e as QWAQRepacker.RepackError {
            throw QWEngineDriverError.weightMissing(gateName, e.description)
        }
    }
}
