// QWInkling2Weights+Slabs.swift — Contiguous BF16 attention projection slabs and MoE FP bank builders.
import Foundation
import Metal
import QWAlloc
import QWCore
import QWGEMM
import QWQuant

struct AttnProjSlabLayout: Sendable {
    let qDim: Int
    let rDim: Int
    let sinkDim: Int
    let kvDim: Int
    let isSliding: Bool

    var qGateRSinkRows: Int { qDim + qDim + rDim + sinkDim }
    var totalRows: Int { isSliding ? (qGateRSinkRows + 2 * kvDim) : qGateRSinkRows }

    var qOutByteOffset: Int { 0 }
    var gateOutByteOffset: Int { qDim * 2 }
    var rOutByteOffset: Int { 2 * qDim * 2 }
    var sinkOutByteOffset: Int { (2 * qDim + rDim) * 2 }
    var kOutByteOffset: Int { qGateRSinkRows * 2 }
    var vOutByteOffset: Int { (qGateRSinkRows + kvDim) * 2 }
    var fusedOutBytes: Int { totalRows * 2 }
}

struct Ink2AttnProjLoad {
    let layout: AttnProjSlabLayout
    let qProj: Ink2Slice
    let gateProj: Ink2Slice
    let rProj: Ink2Slice
    let sinkProj: Ink2Slice
    let kProj: Ink2Slice
    let vProj: Ink2Slice
    let qgrsBank: QWGemmFP.BF16Bank
    let qgrskvBank: QWGemmFP.BF16Bank?
}

extension Ink2Slice {
    var fpSlice: QWFPBufferSlice {
        QWFPBufferSlice(buffer: buffer, offset: offset, bytes: length)
    }

    func bf16Bank(n: Int, k: Int) -> QWGemmFP.BF16Bank {
        QWGemmFP.BF16Bank(weights: fpSlice, n: n, k: k)
    }
}

extension QWSubAllocator {
    func buildAttnProjSlab(
        sa: String,
        isSliding: Bool,
        qDim: Int,
        rDim: Int,
        sinkDim: Int,
        kvDim: Int,
        hidden: Int,
        from st: QWSafeTensorsContainer
    ) throws -> Ink2AttnProjLoad {
        let layout = AttnProjSlabLayout(
            qDim: qDim, rDim: rDim, sinkDim: sinkDim, kvDim: kvDim, isSliding: isSliding
        )
        let qName = "\(sa)q_proj.weight"
        let gateName = "\(sa)gate_proj.weight"
        let rName = "\(sa)r_proj.weight"
        let sinkName = "\(sa)sink_proj.weight"
        let kName = "\(sa)k_proj.weight"
        let vName = "\(sa)v_proj.weight"

        let rowBytes = hidden * 2
        // When every row is 16-byte aligned (H % 8 == 0), tight packing preserves alignment for every sub-slice.
        let canTightPack = (rowBytes % 16) == 0
        if !canTightPack {
            let qS = try copy(qName, from: st)
            let gS = try copy(gateName, from: st)
            let rS = try copy(rName, from: st)
            let sS = try copy(sinkName, from: st)
            let kS = try copy(kName, from: st)
            let vS = try copy(vName, from: st)
            let slab = try allocateBytes(layout.qGateRSinkRows * rowBytes, label: "\(sa)qgrsSlab")
            let dst = slab.buffer.contents().advanced(by: slab.offset)
            let parts = [qS, gS, rS, sS]
            var off = 0
            withExtendedLifetime(slab.buffer) {
                for p in parts {
                    dst.advanced(by: off).copyMemory(from: p.buffer.contents().advanced(by: p.offset), byteCount: p.length)
                    off += p.length
                }
            }
            return Ink2AttnProjLoad(
                layout: layout, qProj: qS, gateProj: gS, rProj: rS, sinkProj: sS, kProj: kS, vProj: vS,
                qgrsBank: slab.bf16Bank(n: layout.qGateRSinkRows, k: hidden), qgrskvBank: nil
            )
        }

        var parts: [(name: String, expectedBytes: Int)] = [
            (qName, qDim * rowBytes),
            (gateName, qDim * rowBytes),
            (rName, rDim * rowBytes),
            (sinkName, sinkDim * rowBytes)
        ]
        if isSliding {
            parts.append((kName, kvDim * rowBytes))
            parts.append((vName, kvDim * rowBytes))
        }
        let totalSlabBytes = parts.reduce(0) { $0 + $1.expectedBytes }
        let slab = try allocateBytes(totalSlabBytes, label: "\(sa)qgrsSlab")
        let dstBase = slab.buffer.contents().advanced(by: slab.offset)
        var subSlices: [Ink2Slice] = []
        subSlices.reserveCapacity(parts.count)
        var cursor = 0
        try withExtendedLifetime(slab.buffer) {
            for part in parts {
                try st.withBytes(of: part.name) { raw in
                    guard raw.count == part.expectedBytes else {
                        throw QWInkling2Error.badState("\(part.name) size \(raw.count) != expected \(part.expectedBytes)")
                    }
                    if let src = raw.baseAddress, raw.count > 0 {
                        dstBase.advanced(by: cursor).copyMemory(from: src, byteCount: raw.count)
                    }
                    subSlices.append(Ink2Slice(buffer: slab.buffer, offset: slab.offset + cursor, length: raw.count))
                    cursor += raw.count
                }
            }
        }
        let kS = isSliding ? subSlices[4] : try copy(kName, from: st)
        let vS = isSliding ? subSlices[5] : try copy(vName, from: st)
        let qgrsSlice = Ink2Slice(
            buffer: slab.buffer, offset: slab.offset, length: layout.qGateRSinkRows * rowBytes
        )
        let qgrsBank = qgrsSlice.bf16Bank(n: layout.qGateRSinkRows, k: hidden)
        let qgrskvBank = isSliding ? slab.bf16Bank(n: layout.totalRows, k: hidden) : nil
        return Ink2AttnProjLoad(
            layout: layout,
            qProj: subSlices[0], gateProj: subSlices[1], rProj: subSlices[2], sinkProj: subSlices[3],
            kProj: kS, vProj: vS,
            qgrsBank: qgrsBank, qgrskvBank: qgrskvBank
        )
    }

    private func interleaveExpertPair(
        gateName: String,
        upName: String,
        experts: Int,
        bytesPerExpert: Int,
        label: String,
        from st: QWSafeTensorsContainer
    ) throws -> Ink2Slice {
        let slab = try allocateBytes(experts * 2 * bytesPerExpert, label: label)
        let dst = slab.buffer.contents().advanced(by: slab.offset)
        try withExtendedLifetime(slab.buffer) {
            try st.withBytes(of: gateName) { gRaw in
                try st.withBytes(of: upName) { uRaw in
                    guard gRaw.count == experts * bytesPerExpert, uRaw.count == experts * bytesPerExpert else {
                        throw QWInkling2Error.badState("\(label) byte mismatch")
                    }
                    guard let gPtr = gRaw.baseAddress, let uPtr = uRaw.baseAddress else {
                        throw QWInkling2Error.badState("\(label) pointer missing")
                    }
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

    func buildNVFP4Experts(
        sw: String,
        experts: Int,
        moeI: Int,
        hidden: Int,
        from st: QWSafeTensorsContainer
    ) throws -> QWInkling2Weights.ExpertsW {
        let expWBytes = moeI * (hidden / 2)
        let expSBytes = moeI * (hidden / 16)
        let s2Bytes = experts * 4
        let guW = try interleaveExpertPair(
            gateName: "\(sw)gate_proj.weight", upName: "\(sw)up_proj.weight",
            experts: experts, bytesPerExpert: expWBytes, label: "\(sw)gate_up.weight", from: st
        )
        let guS = try interleaveExpertPair(
            gateName: "\(sw)gate_proj.scales", upName: "\(sw)up_proj.scales",
            experts: experts, bytesPerExpert: expSBytes, label: "\(sw)gate_up.scales", from: st
        )
        let guS2 = try allocateBytes(2 * s2Bytes, label: "\(sw)gate_up.scales2")
        let guS2Dst = guS2.buffer.contents().advanced(by: guS2.offset)
        try withExtendedLifetime(guS2.buffer) {
            try st.withBytes(of: "\(sw)gate_proj.weight_scale_2") { gRaw in
                guard gRaw.count == s2Bytes, let gPtr = gRaw.baseAddress else {
                    throw QWInkling2Error.badState("\(sw)gate scales2 mismatch")
                }
                guS2Dst.copyMemory(from: gPtr, byteCount: s2Bytes)
            }
            try st.withBytes(of: "\(sw)up_proj.weight_scale_2") { uRaw in
                guard uRaw.count == s2Bytes, let uPtr = uRaw.baseAddress else {
                    throw QWInkling2Error.badState("\(sw)up scales2 mismatch")
                }
                guS2Dst.advanced(by: s2Bytes).copyMemory(from: uPtr, byteCount: s2Bytes)
            }
        }
        let dw = try copy("\(sw)down_proj.weight", from: st)
        let ds = try copy("\(sw)down_proj.scales", from: st)
        let ds2 = try copy("\(sw)down_proj.weight_scale_2", from: st)

        let gateUpBank = QWGemmFP.NVFP4Bank(
            weights: guW.fpSlice, scales: guS.fpSlice, scales2: guS2.fpSlice,
            experts: experts, rowsPerExpert: 2 * moeI, k: hidden, splitGateUpScales2: true
        )
        let downBank = QWGemmFP.NVFP4Bank(
            weights: dw.fpSlice, scales: ds.fpSlice, scales2: ds2.fpSlice,
            experts: experts, rowsPerExpert: hidden, k: moeI
        )
        return .nvfp4(gateUp: gateUpBank, down: downBank)
    }

    func buildMXFP8Experts(
        sw: String,
        experts: Int,
        moeI: Int,
        hidden: Int,
        from st: QWSafeTensorsContainer
    ) throws -> QWInkling2Weights.ExpertsW {
        let expWBytes = moeI * hidden
        let expSBytes = moeI * (hidden / 32)
        let guW = try interleaveExpertPair(
            gateName: "\(sw)gate_proj.weight", upName: "\(sw)up_proj.weight",
            experts: experts, bytesPerExpert: expWBytes, label: "\(sw)gate_up.weight", from: st
        )
        let guS = try interleaveExpertPair(
            gateName: "\(sw)gate_proj.scales", upName: "\(sw)up_proj.scales",
            experts: experts, bytesPerExpert: expSBytes, label: "\(sw)gate_up.scales", from: st
        )
        let dw = try copy("\(sw)down_proj.weight", from: st)
        let ds = try copy("\(sw)down_proj.scales", from: st)
        let gateUpBank = QWGemmFP.MXFP8Bank(
            weights: guW.fpSlice, scales: guS.fpSlice,
            experts: experts, rowsPerExpert: 2 * moeI, k: hidden
        )
        let downBank = QWGemmFP.MXFP8Bank(
            weights: dw.fpSlice, scales: ds.fpSlice,
            experts: experts, rowsPerExpert: hidden, k: moeI
        )
        return .mxfp8(gateUp: gateUpBank, down: downBank)
    }

    func buildControllerWeights(
        cp: String,
        hidden: Int,
        rank: Int,
        from st: QWSafeTensorsContainer
    ) throws -> QWInkling2Weights.ControllerW {
        let down = try copy("\(cp)down_weight", from: st)
        let downBank = down.bf16Bank(n: rank, k: 2 * hidden)
        let rowBytes = rank * 2
        let partBytes = hidden * rowBytes
        // Pack [forget; gain] tightly whenever sub-slice byte offset preserves 16-byte vector alignment.
        if (partBytes % 16) == 0 {
            let fgSlab = try allocateBytes(2 * partBytes, label: "\(cp)fgSlab")
            let dst = fgSlab.buffer.contents().advanced(by: fgSlab.offset)
            try withExtendedLifetime(fgSlab.buffer) {
                for (idx, suffix) in ["forget_weight", "gain_weight"].enumerated() {
                    try st.withBytes(of: "\(cp)\(suffix)") { raw in
                        guard raw.count == partBytes, let src = raw.baseAddress else {
                            throw QWInkling2Error.badState("\(cp)\(suffix) byte mismatch \(raw.count) != \(partBytes)")
                        }
                        dst.advanced(by: idx * partBytes).copyMemory(from: src, byteCount: partBytes)
                    }
                }
            }
            let forget = Ink2Slice(buffer: fgSlab.buffer, offset: fgSlab.offset, length: partBytes)
            let gain = Ink2Slice(buffer: fgSlab.buffer, offset: fgSlab.offset + partBytes, length: partBytes)
            let fgBank = fgSlab.bf16Bank(n: 2 * hidden, k: rank)
            return QWInkling2Weights.ControllerW(
                down: down, forget: forget, gain: gain,
                downW: .bf16(downBank), forgetGainW: .bf16(fgBank)
            )
        }
        let forget = try copy("\(cp)forget_weight", from: st)
        let gain = try copy("\(cp)gain_weight", from: st)
        let fgSlab = try allocateBytes(2 * partBytes, label: "\(cp)fgSlab")
        let dst = fgSlab.buffer.contents().advanced(by: fgSlab.offset)
        withExtendedLifetime(fgSlab.buffer) {
            dst.copyMemory(from: forget.buffer.contents().advanced(by: forget.offset), byteCount: partBytes)
            dst.advanced(by: partBytes).copyMemory(from: gain.buffer.contents().advanced(by: gain.offset), byteCount: partBytes)
        }
        return QWInkling2Weights.ControllerW(
            down: down, forget: forget, gain: gain,
            downW: .bf16(downBank), forgetGainW: .bf16(fgSlab.bf16Bank(n: 2 * hidden, k: rank))
        )
    }
}
