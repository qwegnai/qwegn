// QWInkling2Weights+QWP.swift — QWP1 GPU loader for Inkling-2 dense, controller, MoE, and head weights.
import Foundation
import Metal
import QWAlloc
import QWCore
import QWGEMM
import QWHead
import QWLoad
import QWQuant

private extension QWQTensorEntry {
    var qwpContainerKind: QWWeightView.ContainerKind? {
        guard PlaneLayout.resolve(self) == .hq4 else { return nil }
        let rotated = hq?.vq != "aq"
        if isSHVQ {
            guard rotated else { return nil }
            return .shvq(bits: hqBits, group: hqGroup)
        }
        if isIncumbentHQ4 {
            return .incumbentHQ4G32
        }
        return rotated ? .rotatedHQ(bits: hqBits, group: hqGroup) : .nativeAffine(bits: hqBits, group: hqGroup)
    }
}

private extension QWSubAllocator {
    func copyQWP(_ name: String, from pack: QWQuant.Pack) throws -> (slice: Ink2Slice, entry: QWQTensorEntry) {
        let entry = try pack.entry(name)
        guard let layout = PlaneLayout.resolve(entry),
              let off = layout.offset(entry),
              let bytes = layout.byteCount(entry) else {
            throw QWInkling2Error.badState("invalid QWP1 plane layout for \(name)")
        }
        let slice = try allocateBytes(bytes, label: name)
        if bytes > 0 {
            let dst = slice.buffer.contents().advanced(by: slice.offset)
            try withExtendedLifetime(slice.buffer) {
                try pack.withRawPointer(at: off, byteCount: bytes) { ptr in
                    dst.copyMemory(from: ptr, byteCount: bytes)
                }
            }
        }
        return (slice, entry)
    }

    func makeWeightView(slice: Ink2Slice, n: Int, k: Int, entry: QWQTensorEntry, name: String) throws -> QWWeightView {
        guard let kind = entry.qwpContainerKind else {
            throw QWInkling2Error.badState("unsupported QWP1 container for \(name)")
        }
        guard (slice.length % 2) == 0 else {
            throw QWInkling2Error.badState("unaligned QWP1 block byte count for \(name): \(slice.length)")
        }
        let tv = slice.tv(count: slice.length / 2, dtype: .f16)
        guard let view = QWWeightView(blocks: tv, n: n, k: k, containerKind: kind) else {
            throw QWInkling2Error.badState("invalid QWWeightView shape (\(n)x\(k)) for \(name)")
        }
        return view
    }

    func loadQWPLinearW(
        _ name: String,
        n: Int,
        k: Int,
        from pack: QWQuant.Pack
    ) throws -> (linear: Ink2LinearW, slice: Ink2Slice, entry: QWQTensorEntry) {
        let (slice, entry) = try copyQWP(name, from: pack)
        if entry.rawOff != nil {
            let needBytes = n * k * 2
            guard entry.kind != "f32", slice.length >= needBytes else {
                throw QWInkling2Error.badState("expected BF16 weights [\(n), \(k)] for \(name)")
            }
            let sliced = (slice.length == needBytes)
                ? slice
                : Ink2Slice(buffer: slice.buffer, offset: slice.offset, length: needBytes)
            return (.bf16(sliced.bf16Bank(n: n, k: k)), sliced, entry)
        }
        let view = try makeWeightView(slice: slice, n: n, k: k, entry: entry, name: name)
        return (.quant(view), slice, entry)
    }

    func dequantQWPEntryToBF16(
        entry: QWQTensorEntry,
        nRows: Int,
        kCols: Int,
        name: String,
        from pack: QWQuant.Pack,
        into dstRaw: UnsafeMutableRawPointer
    ) throws {
        guard let layout = PlaneLayout.resolve(entry),
              let off = layout.offset(entry),
              let bytes = layout.byteCount(entry) else {
            throw QWInkling2Error.badState("invalid QWP1 plane layout for \(name)")
        }
        let count = nRows * kCols
        if entry.rawOff != nil {
            if entry.kind == "f32" {
                guard bytes == count * 4 else {
                    throw QWInkling2Error.badState("expected F32 weights [\(nRows), \(kCols)] for \(name)")
                }
                let bp = dstRaw.assumingMemoryBound(to: UInt16.self)
                try pack.withRawPointer(at: off, byteCount: bytes) { ptr in
                    let fp = ptr.assumingMemoryBound(to: Float.self)
                    for i in 0..<count { bp[i] = QWInkling2DSpark.f32ToBF16(fp[i]) }
                }
                return
            }
            guard bytes == count * 2 else {
                throw QWInkling2Error.badState("expected BF16 weights [\(nRows), \(kCols)] for \(name)")
            }
            try pack.withRawPointer(at: off, byteCount: bytes) { ptr in
                dstRaw.copyMemory(from: ptr, byteCount: bytes)
            }
            return
        }
        let f32: [Float] = try pack.withRawPointer(at: off, byteCount: bytes) { ptr in
            if entry.isIncumbentHQ4 {
                return QWQuant.Pack.dequantHQ4Slab(base: ptr, nRows: nRows, kCols: kCols)
            }
            if entry.hq?.vq != "aq" {
                return QWQuant.Pack.dequantHQBlockSlab(
                    base: ptr, nRows: nRows, kCols: kCols,
                    bits: entry.hqBits, group: entry.hqGroup, isSHVQ: entry.isSHVQ
                )
            }
            let gCount = kCols / entry.hqGroup
            let blkBytes = QWAQBlock.bytes(bits: entry.hqBits, group: entry.hqGroup)
            var out = [Float](repeating: 0, count: count)
            var groupBuf = [Float]()
            groupBuf.reserveCapacity(entry.hqGroup)
            for r in 0..<nRows {
                for gg in 0..<gCount {
                    let blkPtr = ptr.advanced(by: (r * gCount + gg) * blkBytes)
                    QWAQBlock.unpack(
                        group: UnsafeRawBufferPointer(start: blkPtr, count: blkBytes),
                        bits: entry.hqBits, group: entry.hqGroup, into: &groupBuf
                    )
                    let rowDst = r * kCols + gg * entry.hqGroup
                    for i in 0..<entry.hqGroup { out[rowDst + i] = groupBuf[i] }
                }
            }
            return out
        }
        guard f32.count == count else {
            throw QWInkling2Error.badState("dequantized element count mismatch for \(name)")
        }
        let bp = dstRaw.assumingMemoryBound(to: UInt16.self)
        for i in 0..<count { bp[i] = QWInkling2DSpark.f32ToBF16(f32[i]) }
    }

    func interleaveQWPExpertPair(
        gateName: String,
        upName: String,
        experts: Int,
        bytesPerExpert: Int,
        label: String,
        from pack: QWQuant.Pack
    ) throws -> Ink2Slice {
        let gEntry = try pack.entry(gateName)
        let uEntry = try pack.entry(upName)
        guard let gLayout = PlaneLayout.resolve(gEntry), let gOff = gLayout.offset(gEntry),
              let gBytes = gLayout.byteCount(gEntry),
              let uLayout = PlaneLayout.resolve(uEntry), let uOff = uLayout.offset(uEntry),
              let uBytes = uLayout.byteCount(uEntry),
              gBytes == experts * bytesPerExpert, uBytes == experts * bytesPerExpert else {
            throw QWInkling2Error.badState("\(label) QWP1 expert byte mismatch")
        }
        let slab = try allocateBytes(experts * 2 * bytesPerExpert, label: label)
        let dst = slab.buffer.contents().advanced(by: slab.offset)
        try withExtendedLifetime(slab.buffer) {
            try pack.withRawPointer(at: gOff, byteCount: gBytes) { gPtr in
                try pack.withRawPointer(at: uOff, byteCount: uBytes) { uPtr in
                    for e in 0..<experts {
                        let srcOff = e * bytesPerExpert
                        let dstOff = e * 2 * bytesPerExpert
                        dst.advanced(by: dstOff).copyMemory(from: gPtr.advanced(by: srcOff), byteCount: bytesPerExpert)
                        dst.advanced(by: dstOff + bytesPerExpert).copyMemory(from: uPtr.advanced(by: srcOff), byteCount: bytesPerExpert)
                    }
                }
            }
        }
        return slab
    }
}

private struct Ink2QWPAttnProjLoad {
    let layout: AttnProjSlabLayout
    let qProjW: Ink2LinearW
    let gateProjW: Ink2LinearW
    let rProjW: Ink2LinearW
    let sinkProjW: Ink2LinearW
    let kProjW: Ink2LinearW
    let vProjW: Ink2LinearW
    let qgrsBank: QWGemmFP.BF16Bank?
    let qgrskvBank: QWGemmFP.BF16Bank?
    let qgrsView: QWWeightView?
    let qgrskvView: QWWeightView?
}

private func buildQWPAttnProj(
    bump: QWSubAllocator,
    sa: String,
    isSliding: Bool,
    qDim: Int,
    rDim: Int,
    sinkDim: Int,
    kvDim: Int,
    hidden: Int,
    from pack: QWQuant.Pack
) throws -> Ink2QWPAttnProjLoad {
    let layout = AttnProjSlabLayout(
        qDim: qDim, rDim: rDim, sinkDim: sinkDim, kvDim: kvDim, isSliding: isSliding
    )
    let specs: [(name: String, rows: Int)] = [
        ("\(sa)q_proj.weight", qDim),
        ("\(sa)gate_proj.weight", qDim),
        ("\(sa)r_proj.weight", rDim),
        ("\(sa)sink_proj.weight", sinkDim),
        ("\(sa)k_proj.weight", kvDim),
        ("\(sa)v_proj.weight", kvDim)
    ]
    let entries = try specs.map { try pack.entry($0.name) }
    let qgrsAllRawBF16 = entries[0..<4].allSatisfy { $0.rawOff != nil && $0.kind != "f32" } && ((hidden * 2) % 16) == 0
    let kvAllRawBF16 = entries[4..<6].allSatisfy { $0.rawOff != nil && $0.kind != "f32" }
    let qgrsKinds = entries[0..<4].compactMap { $0.qwpContainerKind }
    let qgrsSameQuant = qgrsKinds.count == 4 && qgrsKinds.allSatisfy { $0 == qgrsKinds[0] }
    let kvSameQuant = entries[4].qwpContainerKind != nil
        && entries[4].qwpContainerKind == entries[0].qwpContainerKind
        && entries[5].qwpContainerKind == entries[0].qwpContainerKind

    func copyConcatSlab(_ parts: [(name: String, rows: Int)], totalBytes: Int, label: String) throws -> (Ink2Slice, [Ink2Slice]) {
        let slab = try bump.allocateBytes(totalBytes, label: label)
        let dst = slab.buffer.contents().advanced(by: slab.offset)
        var off = 0
        var subSlices: [Ink2Slice] = []
        subSlices.reserveCapacity(parts.count)
        try withExtendedLifetime(slab.buffer) {
            for (name, _) in parts {
                let e = try pack.entry(name)
                guard let l = PlaneLayout.resolve(e), let pOff = l.offset(e), let bytes = l.byteCount(e) else {
                    throw QWInkling2Error.badState("invalid QWP1 layout for \(name)")
                }
                guard off + bytes <= totalBytes else {
                    throw QWInkling2Error.badState("copyConcatSlab overflow for \(name): \(off + bytes) > \(totalBytes)")
                }
                try pack.withRawPointer(at: pOff, byteCount: bytes) { ptr in
                    dst.advanced(by: off).copyMemory(from: ptr, byteCount: bytes)
                }
                subSlices.append(Ink2Slice(buffer: slab.buffer, offset: slab.offset + off, length: bytes))
                off += bytes
            }
        }
        return (slab, subSlices)
    }

    if qgrsAllRawBF16 {
        let packKV = isSliding && kvAllRawBF16
        let count = packKV ? 6 : 4
        let totalRows = packKV ? layout.totalRows : layout.qGateRSinkRows
        let (slab, subs) = try copyConcatSlab(Array(specs.prefix(count)), totalBytes: totalRows * hidden * 2, label: "\(sa)qgrsBF16")
        let qgrsSlice = Ink2Slice(buffer: slab.buffer, offset: slab.offset, length: layout.qGateRSinkRows * hidden * 2)
        let kW: Ink2LinearW = packKV ? .bf16(subs[4].bf16Bank(n: kvDim, k: hidden))
            : try bump.loadQWPLinearW(specs[4].name, n: kvDim, k: hidden, from: pack).linear
        let vW: Ink2LinearW = packKV ? .bf16(subs[5].bf16Bank(n: kvDim, k: hidden))
            : try bump.loadQWPLinearW(specs[5].name, n: kvDim, k: hidden, from: pack).linear
        return Ink2QWPAttnProjLoad(
            layout: layout,
            qProjW: .bf16(subs[0].bf16Bank(n: qDim, k: hidden)),
            gateProjW: .bf16(subs[1].bf16Bank(n: qDim, k: hidden)),
            rProjW: .bf16(subs[2].bf16Bank(n: rDim, k: hidden)),
            sinkProjW: .bf16(subs[3].bf16Bank(n: sinkDim, k: hidden)),
            kProjW: kW, vProjW: vW,
            qgrsBank: qgrsSlice.bf16Bank(n: layout.qGateRSinkRows, k: hidden),
            qgrskvBank: packKV ? slab.bf16Bank(n: layout.totalRows, k: hidden) : nil,
            qgrsView: nil, qgrskvView: nil
        )
    }

    if qgrsSameQuant, let kind = qgrsKinds.first {
        let packKV = isSliding && kvSameQuant
        let count = packKV ? 6 : 4
        let rowBytes = kind.rowBytes(k: hidden)
        let totalRows = packKV ? layout.totalRows : layout.qGateRSinkRows
        let (slab, subs) = try copyConcatSlab(Array(specs.prefix(count)), totalBytes: totalRows * rowBytes, label: "\(sa)qgrsQuant")
        let qgrsTV = Ink2Slice(buffer: slab.buffer, offset: slab.offset, length: layout.qGateRSinkRows * rowBytes)
            .tv(count: (layout.qGateRSinkRows * rowBytes) / 2, dtype: .f16)
        guard let qgrsView = QWWeightView(blocks: qgrsTV, n: layout.qGateRSinkRows, k: hidden, containerKind: kind) else {
            throw QWInkling2Error.badState("invalid fused QWP1 qgrsView for \(sa)")
        }
        let qgrskvView: QWWeightView?
        if packKV {
            let fullTV = slab.tv(count: slab.length / 2, dtype: .f16)
            qgrskvView = QWWeightView(blocks: fullTV, n: layout.totalRows, k: hidden, containerKind: kind)
        } else {
            qgrskvView = nil
        }
        let subViews: [QWWeightView] = try zip(specs.prefix(count), subs).map { spec, sub in
            let tv = sub.tv(count: sub.length / 2, dtype: .f16)
            guard let sv = qgrsView.subview(blocks: tv, n: spec.rows, k: hidden) else {
                throw QWInkling2Error.badState("invalid QWP1 subview for \(spec.name)")
            }
            return sv
        }
        let kW: Ink2LinearW = packKV ? .quant(subViews[4])
            : try bump.loadQWPLinearW(specs[4].name, n: kvDim, k: hidden, from: pack).linear
        let vW: Ink2LinearW = packKV ? .quant(subViews[5])
            : try bump.loadQWPLinearW(specs[5].name, n: kvDim, k: hidden, from: pack).linear
        return Ink2QWPAttnProjLoad(
            layout: layout,
            qProjW: .quant(subViews[0]), gateProjW: .quant(subViews[1]),
            rProjW: .quant(subViews[2]), sinkProjW: .quant(subViews[3]),
            kProjW: kW, vProjW: vW,
            qgrsBank: nil, qgrskvBank: nil, qgrsView: qgrsView, qgrskvView: qgrskvView
        )
    }

    let qW = try bump.loadQWPLinearW(specs[0].name, n: qDim, k: hidden, from: pack).linear
    let gW = try bump.loadQWPLinearW(specs[1].name, n: qDim, k: hidden, from: pack).linear
    let rW = try bump.loadQWPLinearW(specs[2].name, n: rDim, k: hidden, from: pack).linear
    let sW = try bump.loadQWPLinearW(specs[3].name, n: sinkDim, k: hidden, from: pack).linear
    let kW = try bump.loadQWPLinearW(specs[4].name, n: kvDim, k: hidden, from: pack).linear
    let vW = try bump.loadQWPLinearW(specs[5].name, n: kvDim, k: hidden, from: pack).linear
    return Ink2QWPAttnProjLoad(
        layout: layout, qProjW: qW, gateProjW: gW, rProjW: rW, sinkProjW: sW,
        kProjW: kW, vProjW: vW, qgrsBank: nil, qgrskvBank: nil, qgrsView: nil, qgrskvView: nil
    )
}

private func buildQWPController(
    bump: QWSubAllocator,
    cp: String,
    hidden: Int,
    rank: Int,
    from pack: QWQuant.Pack
) throws -> QWInkling2Weights.ControllerW {
    let downEntry = try pack.entry("\(cp)down_weight")
    let fEntry = try pack.entry("\(cp)forget_weight")
    let gEntry = try pack.entry("\(cp)gain_weight")
    let downBytes = rank * 2 * hidden * 2
    let partBytes = hidden * rank * 2

    let downSlice = try bump.allocateBytes(downBytes, label: "\(cp)down_weight")
    let downDst = downSlice.buffer.contents().advanced(by: downSlice.offset)
    try withExtendedLifetime(downSlice.buffer) {
        try bump.dequantQWPEntryToBF16(
            entry: downEntry, nRows: rank, kCols: 2 * hidden, name: "\(cp)down_weight",
            from: pack, into: downDst
        )
    }
    let downW: Ink2LinearW = .bf16(downSlice.bf16Bank(n: rank, k: 2 * hidden))

    let fgSlab = try bump.allocateBytes(2 * partBytes, label: "\(cp)fgSlab")
    let fgDst = fgSlab.buffer.contents().advanced(by: fgSlab.offset)
    try withExtendedLifetime(fgSlab.buffer) {
        try bump.dequantQWPEntryToBF16(
            entry: fEntry, nRows: hidden, kCols: rank, name: "\(cp)forget_weight",
            from: pack, into: fgDst
        )
        try bump.dequantQWPEntryToBF16(
            entry: gEntry, nRows: hidden, kCols: rank, name: "\(cp)gain_weight",
            from: pack, into: fgDst.advanced(by: partBytes)
        )
    }
    let forgetSlice = Ink2Slice(buffer: fgSlab.buffer, offset: fgSlab.offset, length: partBytes)
    let gainSlice = Ink2Slice(buffer: fgSlab.buffer, offset: fgSlab.offset + partBytes, length: partBytes)
    return QWInkling2Weights.ControllerW(
        down: downSlice, forget: forgetSlice, gain: gainSlice,
        downW: downW, forgetGainW: .bf16(fgSlab.bf16Bank(n: 2 * hidden, k: rank))
    )
}

private func buildQWPExperts(
    bump: QWSubAllocator,
    lp: String,
    experts: Int,
    moeI: Int,
    hidden: Int,
    from pack: QWQuant.Pack
) throws -> QWInkling2Weights.ExpertsW {
    let sw = "\(lp)mlp.switch_mlp."
    if (try? pack.entry("\(sw)gate_proj.weight")) != nil {
        let expWBytes = moeI * (hidden / 2)
        let expSBytes = moeI * (hidden / 16)
        let s2Bytes = experts * 4
        let guW = try bump.interleaveQWPExpertPair(
            gateName: "\(sw)gate_proj.weight", upName: "\(sw)up_proj.weight",
            experts: experts, bytesPerExpert: expWBytes, label: "\(sw)gate_up.weight", from: pack
        )
        let guS = try bump.interleaveQWPExpertPair(
            gateName: "\(sw)gate_proj.scales", upName: "\(sw)up_proj.scales",
            experts: experts, bytesPerExpert: expSBytes, label: "\(sw)gate_up.scales", from: pack
        )
        let guS2 = try bump.allocateBytes(2 * s2Bytes, label: "\(sw)gate_up.scales2")
        let gs2 = try bump.copyQWP("\(sw)gate_proj.weight_scale_2", from: pack).slice
        let us2 = try bump.copyQWP("\(sw)up_proj.weight_scale_2", from: pack).slice
        guard gs2.length == s2Bytes, us2.length == s2Bytes else {
            throw QWInkling2Error.badState("\(sw) scales2 size mismatch")
        }
        guS2.copyConcatenated([gs2, us2])

        let (dw, _) = try bump.copyQWP("\(sw)down_proj.weight", from: pack)
        let (ds, _) = try bump.copyQWP("\(sw)down_proj.scales", from: pack)
        let (ds2, _) = try bump.copyQWP("\(sw)down_proj.weight_scale_2", from: pack)
        return .nvfp4(
            gateUp: QWGemmFP.NVFP4Bank(
                weights: guW.fpSlice, scales: guS.fpSlice, scales2: guS2.fpSlice,
                experts: experts, rowsPerExpert: 2 * moeI, k: hidden, splitGateUpScales2: true
            ),
            down: QWGemmFP.NVFP4Bank(
                weights: dw.fpSlice, scales: ds.fpSlice, scales2: ds2.fpSlice,
                experts: experts, rowsPerExpert: hidden, k: moeI
            )
        )
    }

    let guName = "\(lp)mlp.experts.gate_up_proj"
    let dnName = "\(lp)mlp.experts.down_proj"
    let (guSlice, guEntry) = try bump.copyQWP(guName, from: pack)
    let (dnSlice, dnEntry) = try bump.copyQWP(dnName, from: pack)
    let guView = try bump.makeWeightView(slice: guSlice, n: experts * 2 * moeI, k: hidden, entry: guEntry, name: guName)
    let dnView = try bump.makeWeightView(slice: dnSlice, n: experts * hidden, k: moeI, entry: dnEntry, name: dnName)
    return .qwp(gateUp: guView, down: dnView)
}

extension QWInkling2Weights {
    convenience init(
        qwp pack: QWQuant.Pack,
        device: any MTLDevice,
        descriptor: QWModelDescriptor,
        ink2: QWModelDescriptor.Inkling2Config
    ) throws {
        func sumQWPBytes(forPrefix prefix: String?) throws -> Int {
            var total = 0
            for (name, entry) in pack.tensors {
                if let prefix, !name.hasPrefix(prefix) { continue }
                if name.contains("vision_tower.") || name.hasPrefix("visual.") || name.contains(".visual.") { continue }
                guard let layout = PlaneLayout.resolve(entry),
                      let bytes = layout.byteCount(entry) else { continue }
                let effBytes = (entry.rawOff == nil && name.contains(".controllers.") && entry.shape.count == 2)
                    ? max(bytes, entry.shape[0] * entry.shape[1] * 2)
                    : bytes
                total += ((effBytes + 255) / 256) * 256
            }
            return total + 65536
        }

        let pfx = (try? pack.entry("language_model.model.embed_tokens.weight")) != nil ? "language_model." : ""
        let mdl = "\(pfx)model."
        let globalNames = [
            "\(mdl)embed_tokens.weight", "\(mdl)embed_norm.weight",
            "\(pfx)lm_head.weight", "\(mdl)norm.weight",
            "\(mdl)final_input.query", "\(mdl)message_queries"
        ]
        var gCap = 65536
        for name in globalNames {
            let e = try pack.entry(name)
            guard let layout = PlaneLayout.resolve(e), let bytes = layout.byteCount(e) else {
                throw QWInkling2Error.badState("invalid global QWP1 entry \(name)")
            }
            gCap += ((bytes + 255) / 256) * 256
        }
        let gBump = try QWSubAllocator.makeWeightAllocator(device: device, capacity: gCap)
        let embedTokens = try gBump.copyQWP(globalNames[0], from: pack).slice
        let embedNorm = try gBump.copyQWP(globalNames[1], from: pack).slice
        let H = descriptor.hidden
        let headVocab = ink2.unpaddedVocabSize
        let headLoaded = try gBump.loadQWPLinearW(globalNames[2], n: headVocab, k: H, from: pack)
        let lmHeadStorage: QWHeadStorage
        let lmHeadSlice: Ink2Slice?
        let lmHeadTV: QWTensorValue?
        let lmHeadBank: QWGemmFP.BF16Bank?
        var anyRotated = false
        switch headLoaded.linear {
        case .bf16(let bank):
            let tv = headLoaded.slice.tv(count: headVocab * H, dtype: .bf16)
            lmHeadStorage = .bf16(tv)
            lmHeadSlice = headLoaded.slice
            lmHeadTV = tv
            lmHeadBank = bank
        case .quant(let view):
            if view.isRotated { anyRotated = true }
            guard let hw = QWHeadWeight(
                blocks: view.blocks, n: view.n, k: view.k,
                bits: view.bits, groupSize: view.groupSize,
                isRotated: view.isRotated,
                isIncumbentContainer: view.isIncumbentContainer,
                isSHVQ: view.isSHVQ
            ) else {
                throw QWInkling2Error.badState("invalid QWP1 QWHeadWeight for lm_head.weight")
            }
            lmHeadStorage = .q4g(hw)
            lmHeadSlice = nil
            lmHeadTV = nil
            lmHeadBank = nil
        }

        let finalNorm = try gBump.copyQWP(globalNames[3], from: pack).slice
        let finalInputQuery = try gBump.copyQWP(globalNames[4], from: pack).slice
        let messageQueries = try gBump.copyQWP(globalNames[5], from: pack).slice

        let nH = descriptor.qHeads
        let nKV = descriptor.kvHeads
        let headDim = descriptor.headDim
        let qDim = nH * headDim
        let kvDim = nKV * headDim
        let rDim = nH * ink2.dRel
        let moeI = descriptor.moeIntermediate ?? 1152
        let nExp = descriptor.numExperts ?? 128

        var buffers: [any MTLBuffer] = [gBump.slabBuffer]
        var builtLayers: [LayerWeights] = []
        builtLayers.reserveCapacity(descriptor.numLayers)

        func trackRotated(_ w: Ink2LinearW) {
            if case .quant(let v) = w, v.isRotated { anyRotated = true }
        }

        for l in 0..<descriptor.numLayers {
            let lPfx = "\(mdl)layers.\(l)."
            // Reserve 2x for possible NVFP4 gate_proj+up_proj + scale_2 staging copies before building experts.
            let layerCap = try sumQWPBytes(forPrefix: lPfx) + (nExp * 4 * 4)
            let lBump = try QWSubAllocator.makeWeightAllocator(device: device, capacity: layerCap)
            let isSliding = ink2.isSlidingLayer(l)
            let attnQuery = (l > 0) ? try lBump.copyQWP("\(lPfx)attention_input.query", from: pack).slice : nil
            let mlpQuery = try lBump.copyQWP("\(lPfx)mlp_input.query", from: pack).slice
            let attnNorm: AttnNorm = isSliding
                ? .sliding(inputNorm: try lBump.copyQWP("\(lPfx)input_layernorm.weight", from: pack).slice)
                : .global(
                    qNorm: try lBump.copyQWP("\(lPfx)q_input_layernorm.weight", from: pack).slice,
                    kNorm: try lBump.copyQWP("\(lPfx)k_input_layernorm.weight", from: pack).slice,
                    vNorm: try lBump.copyQWP("\(lPfx)v_input_layernorm.weight", from: pack).slice
                )
            let postAttnNorm = try lBump.copyQWP("\(lPfx)post_attention_layernorm.weight", from: pack).slice
            let attnSConv = try lBump.copyQWP("\(lPfx)attn_sconv.conv.weight", from: pack).slice
            let mlpSConv = try lBump.copyQWP("\(lPfx)mlp_sconv.conv.weight", from: pack).slice
            let sa = "\(lPfx)self_attn."
            let proj = try buildQWPAttnProj(
                bump: lBump, sa: sa, isSliding: isSliding,
                qDim: qDim, rDim: rDim, sinkDim: nH, kvDim: kvDim, hidden: H, from: pack
            )
            let oLoaded = try lBump.loadQWPLinearW("\(sa)o_proj.weight", n: H, k: qDim, from: pack)
            let oProjW = oLoaded.linear
            [proj.qProjW, proj.kProjW, proj.vProjW, oProjW, proj.gateProjW, proj.rProjW, proj.sinkProjW]
                .forEach(trackRotated)

            let qScale = try lBump.copyQWP("\(sa)q_scale", from: pack).slice
            let relScale = try lBump.copyQWP("\(sa)rel_scale", from: pack).slice
            let relProj = try lBump.copyQWP("\(sa)rel_proj", from: pack).slice
            let relProjTV = relProj.tv(count: relProj.length / 2, dtype: .bf16)

            let kvSConv: KVSConv
            if isSliding {
                kvSConv = .staticConv(
                    k: try lBump.copyQWP("\(sa)k_sconv.conv.weight", from: pack).slice,
                    v: try lBump.copyQWP("\(sa)v_sconv.conv.weight", from: pack).slice
                )
            } else {
                let kp = try lBump.loadQWPLinearW("\(sa)k_sconv.weight_proj.weight", n: nKV * 4, k: H, from: pack).linear
                let vp = try lBump.loadQWPLinearW("\(sa)v_sconv.weight_proj.weight", n: nKV * 4, k: H, from: pack).linear
                trackRotated(kp); trackRotated(vp)
                kvSConv = .dynamicConv(
                    kStatic: try lBump.copyQWP("\(sa)k_sconv.static_weight", from: pack).slice,
                    kProjW: kp,
                    vStatic: try lBump.copyQWP("\(sa)v_sconv.static_weight", from: pack).slice,
                    vProjW: vp
                )
            }

            let ctrl: LayerControllers?
            if l >= ink2.loopCoreStart && l < ink2.loopCoreEnd {
                func lc(_ name: String) throws -> ControllerW {
                    let c = try buildQWPController(
                        bump: lBump, cp: "\(lPfx)controllers.\(name).", hidden: H, rank: ink2.loopGateRank, from: pack
                    )
                    trackRotated(c.downW); trackRotated(c.forgetGainW)
                    return c
                }
                ctrl = LayerControllers(attn: try lc("attn"), k: try lc("k"), v: try lc("v"), mlp: try lc("mlp"))
            } else {
                ctrl = nil
            }

            let routerGate = try lBump.copyQWP("\(lPfx)mlp.gate_weight", from: pack).slice
            let routerBias = try lBump.copyQWP("\(lPfx)mlp.e_score_correction_bias", from: pack).slice
            let experts = try buildQWPExperts(
                bump: lBump, lp: lPfx, experts: nExp, moeI: moeI, hidden: H, from: pack
            )
            if case .qwp(let gu, let dn) = experts, gu.isRotated || dn.isRotated {
                anyRotated = true
            }

            buffers.append(lBump.slabBuffer)
            builtLayers.append(LayerWeights(
                layerIdx: l, isSliding: isSliding,
                attnInputQuery: attnQuery, mlpInputQuery: mlpQuery,
                attnNorm: attnNorm, postAttnNorm: postAttnNorm,
                attnSConv: attnSConv, mlpSConv: mlpSConv,
                projLayout: proj.layout,
                qProjW: proj.qProjW, kProjW: proj.kProjW, vProjW: proj.vProjW, oProjW: oProjW,
                gateProjW: proj.gateProjW, rProjW: proj.rProjW, sinkProjW: proj.sinkProjW,
                qgrsBank: proj.qgrsBank, qgrskvBank: proj.qgrskvBank,
                qgrsView: proj.qgrsView, qgrskvView: proj.qgrskvView,
                qScale: qScale, relScale: relScale, relProj: relProj, relProjTV: relProjTV,
                kvSConv: kvSConv, controllers: ctrl,
                routerGate: routerGate, routerGateBank: routerGate.bf16Bank(n: nExp, k: H),
                routerBias: routerBias, experts: experts
            ))
        }

        let hasVis = pack.tensors.keys.contains { $0.contains("vision_tower.") || $0.contains("visual.") }
        self.init(
            embedTokens: embedTokens,
            embedNorm: embedNorm,
            lmHeadSlice: lmHeadSlice,
            lmHeadTV: lmHeadTV,
            lmHeadBank: lmHeadBank,
            lmHeadStorage: lmHeadStorage,
            finalNorm: finalNorm,
            finalInputQuery: finalInputQuery,
            messageQueries: messageQueries,
            layers: builtLayers,
            residentBuffers: buffers,
            hasVisionWeights: hasVis,
            weightBasis: anyRotated ? .rotated : .standard,
            isQuantizedQWP: true
        )
        pack.adviseDontNeed()
    }
}
