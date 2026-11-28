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
                             key: KeyPath<QWQTensorEntry, Int?>,
                             residentExperts: Int? = nil,
                             formatOverride: QWWeightFormat? = nil) throws -> QWTensorValue {
        let effectiveFormat = formatOverride ?? format
        if effectiveFormat.isNativeAffine {
            return try aqPlaneView(name, n: n, k: k, formatOverride: effectiveFormat)
        }
        let name = resolveName(name)
        let elems: Int
        switch key {
        case \QWQTensorEntry.hq4Off, \QWQTensorEntry.q4gOff:
            if let pk {
                let entry = try pk.entry(name)
                guard let layout = PlaneLayout.resolve(entry), layout == .hq4 else {
                    throw QWEngineDriverError.weightMissing(name, "needs hq4_off plane")
                }
                elems = try packPlaneHalfCount(name, entry: entry, layout: layout,
                                               n: n, k: k, residentExperts: residentExperts)
            } else {
                let rowBytes = residentExperts != nil
                    ? effectiveFormat.expertDownRowBytes(k: k) : effectiveFormat.rowBytes(k: k)
                elems = (n * rowBytes) / 2
            }
        default:
            throw QWEngineDriverError.weightMissing(name, "plane key unsupported")
        }
        let t = try allocateWeight(QWTensorShape([elems], .f16))
        if let io {
            let dst = t.region.buffer.contents().advanced(by: t.region.base + t.offset)
            try io.loadSlice(name, layout: .hq4, byteOffset: 0, byteCount: elems * 2, into: dst)
        } else {
            try quantizeSafetensorsToHQ4(
                name, n: n, k: k,
                into: t.region.buffer,
                dstByteOffset: t.region.base + t.offset
            )
        }
        return t
    }

    private func packPlaneHalfCount(
        _ name: String, entry: QWQTensorEntry, layout: PlaneLayout,
        n: Int, k: Int, residentExperts: Int?
    ) throws -> Int {
        guard layout == .hq4, let totalBytes = layout.byteCount(entry) else {
            throw QWEngineDriverError.weightMissing(name, "plane layout byteCount invalid")
        }
        let isFused3D = entry.shape.count == 3 || (entry.experts != nil && entry.eHq4 != nil)
        if isFused3D {
            guard let expCount = entry.experts ?? (entry.shape.count == 3 ? entry.shape[0] : nil),
                  expCount > 0 else {
                throw QWEngineDriverError.weightMissing(name, "3D expert tensor missing expert count")
            }
            let rowsPerExpert: Int
            if entry.shape.count == 3 {
                rowsPerExpert = entry.shape[1]
                guard entry.shape[2] == k else {
                    throw QWEngineDriverError.weightMissing(
                        name, "3D expert tensor k=\(entry.shape[2]) != expected \(k)")
                }
            } else if entry.shape.count == 2, entry.shape[0] % expCount == 0, entry.shape[1] == k {
                rowsPerExpert = entry.shape[0] / expCount
            } else {
                throw QWEngineDriverError.weightMissing(name, "3D expert tensor shape invalid: \(entry.shape)")
            }
            guard rowsPerExpert > 0, totalBytes % expCount == 0 else {
                throw QWEngineDriverError.weightMissing(name, "3D expert tensor has invalid per-expert bytes")
            }
            let per = totalBytes / expCount
            if let explicitPer = entry.eHq4 {
                guard per == explicitPer else {
                    throw QWEngineDriverError.weightMissing(name, "3D expert e_hq4 \(explicitPer) != expected \(per)")
                }
            }
            let sliceExperts = residentExperts ?? (n / rowsPerExpert)
            let (expectedRows, ovRows) = sliceExperts.multipliedReportingOverflow(by: rowsPerExpert)
            guard sliceExperts > 0, sliceExperts <= expCount, !ovRows, expectedRows == n else {
                throw QWEngineDriverError.weightMissing(
                    name, "3D expert slice \(sliceExperts) invalid for n=\(n), experts=\(expCount)")
            }
            let (sliceBytes, ovBytes) = sliceExperts.multipliedReportingOverflow(by: per)
            guard !ovBytes, sliceBytes % 2 == 0 else {
                throw QWEngineDriverError.weightMissing(name, "3D expert slice byte count invalid")
            }
            return sliceBytes / 2
        }
        guard entry.shape.count == 2, entry.shape[0] > 0, entry.shape[1] == k,
              n >= 0, n <= entry.shape[0], totalBytes % entry.shape[0] == 0 else {
            throw QWEngineDriverError.weightMissing(
                name, "2D plane shape \(entry.shape) incompatible with [\(n), \(k)]")
        }
        let rowBytes = totalBytes / entry.shape[0]
        let (byteCount, ovBytes) = n.multipliedReportingOverflow(by: rowBytes)
        guard !ovBytes, byteCount % 2 == 0 else {
            throw QWEngineDriverError.weightMissing(name, "2D plane byte count invalid")
        }
        return byteCount / 2
    }

    /// The bank's view, carrying the container's own pair and the model's basis.
    mutating func q4(_ name: String, n: Int, k: Int,
                     formatOverride: QWWeightFormat? = nil) throws -> QWWeightView {
        let effectiveFormat = formatOverride ?? format
        let blocks = try planeView(name, n: n, k: k, key: \QWQTensorEntry.hq4Off,
                                   formatOverride: formatOverride)
        let resolved = resolveName(name)
        let entry = try? pk?.entry(resolved)
        let bits = entry?.hqBits ?? effectiveFormat.bits
        let groupSize = entry?.hqGroup ?? effectiveFormat.groupSize
        let isIncumbent = entry?.isIncumbentHQ4 ?? effectiveFormat.denseIsIncumbentContainer
        let isSHVQ = entry?.isSHVQ ?? effectiveFormat.denseIsSHVQ
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
    mutating func aqPlaneView(_ name: String, n: Int, k: Int,
                              formatOverride: QWWeightFormat? = nil) throws -> QWTensorValue {
        let effectiveFormat = formatOverride ?? format
        let elems = (n * effectiveFormat.rowBytes(k: k)) / 2
        let t = try allocateWeight(QWTensorShape([elems], .f16))
        try repackSafetensorsToAQ(name, n: n, k: k,
                                  bits: effectiveFormat.bits, group: effectiveFormat.groupSize,
                                  into: t.region.buffer, dstByteOffset: t.region.base + t.offset)
        return t
    }

    /// The family's concatenated plane: each tensor repacked into its own row
    /// slice. The family's rows are uniform, so the slices are the same bytes
    /// the incumbent's are — only the row size differs.
    mutating func aqConcatPlane(_ tensors: [(name: String, n: Int)], k: Int, extraZeroRows: Int = 0) throws -> QWTensorValue {
        let rowBytes = format.rowBytes(k: k)
        let dataRows = tensors.reduce(0) { $0 + $1.n }
        let totalN = dataRows + extraZeroRows
        let t = try allocateWeight(QWTensorShape([(totalN * rowBytes) / 2], .f16))
        var row0 = 0
        for (name, n) in tensors {
            try repackSafetensorsToAQ(name, n: n, k: k, into: t.region.buffer,
                                      dstByteOffset: t.region.base + t.offset + row0 * rowBytes)
            row0 += n
        }
        if extraZeroRows > 0 {
            let tail = t.region.buffer.contents().advanced(by: t.region.base + t.offset + dataRows * rowBytes)
            memset(tail, 0, extraZeroRows * rowBytes)
        }
        return t
    }

    /// Repacks one MLX affine tensor into the family's container, on the GPU.
    /// The repack itself is QWLoad's door — the one both engines use — and
    /// this is the driver's name resolution and the model's own pair.
    func repackSafetensorsToAQ(_ name: String, n: Int, k: Int,
                                bits: Int? = nil, group: Int? = nil,
                                into dstBuffer: any MTLBuffer, dstByteOffset: Int) throws {
        guard let st else {
            throw QWEngineDriverError.weightMissing(name, "the family is a safetensors route only")
        }
        let resolved = resolveName(name)
        do {
            try QWAQRepacker.repackMLXAffine(hal: hl, container: st, name: resolved,
                                             rows: n, cols: k,
                                             bits: bits ?? format.bits, group: group ?? format.groupSize,
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
