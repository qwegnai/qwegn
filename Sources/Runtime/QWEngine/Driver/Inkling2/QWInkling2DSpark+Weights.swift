// QWInkling2DSpark+Weights.swift — Source loading, QWP1 SHVQ/HQ/AQ dequantization & weight folding for DSpark.
import Accelerate
import Foundation
import Metal
import QWAlloc
import QWCore
import QWGEMM
import QWHead
import QWLoad
import QWQuant

@_silgen_name("cblas_dgemm$NEWLAPACK")
private func qwInk2Dgemm(
    _ order: Int32, _ transA: Int32, _ transB: Int32,
    _ m: Int32, _ n: Int32, _ k: Int32,
    _ alpha: Double, _ a: UnsafePointer<Double>, _ lda: Int32,
    _ b: UnsafePointer<Double>, _ ldb: Int32,
    _ beta: Double, _ c: UnsafeMutablePointer<Double>, _ ldc: Int32
)

enum DSparkSource {
    case safetensors(QWSafeTensorsContainer)
    case qwp(QWQuant.Pack)

    func entry(of name: String) throws -> QWQTensorEntry? {
        guard case .qwp(let pack) = self else { return nil }
        return try pack.entry(name)
    }

    func containerKind(of name: String) throws -> QWWeightView.ContainerKind? {
        guard let e = try entry(of: name), PlaneLayout.resolve(e) == .hq4 else { return nil }
        let rotated = e.hq?.vq != "aq"
        if e.isSHVQ {
            guard rotated else { return nil }
            return .shvq(bits: e.hqBits, group: e.hqGroup)
        }
        if e.isIncumbentHQ4 { return .incumbentHQ4G32 }
        return rotated ? .rotatedHQ(bits: e.hqBits, group: e.hqGroup) : .nativeAffine(bits: e.hqBits, group: e.hqGroup)
    }

    func isRawBF16(of name: String) throws -> Bool {
        switch self {
        case .safetensors: return true
        case .qwp(let pack):
            let e = try pack.entry(name)
            return PlaneLayout.resolve(e) == .raw && e.kind != "f32"
        }
    }

    func byteLength(of name: String) throws -> Int {
        switch self {
        case .safetensors(let st):
            guard let t = st.tensor(name) else {
                throw QWInkling2Error.badState("DSpark tensor missing: \(name)")
            }
            return t.byteLength
        case .qwp(let pack):
            let e = try pack.entry(name)
            guard let layout = PlaneLayout.resolve(e), let bytes = layout.byteCount(e) else {
                throw QWInkling2Error.badState("DSpark QWP1 tensor invalid: \(name)")
            }
            return bytes
        }
    }

    func withBytes<R>(of name: String, _ body: (UnsafeRawBufferPointer) throws -> R) throws -> R {
        switch self {
        case .safetensors(let st):
            return try st.withBytes(of: name, body)
        case .qwp(let pack):
            let e = try pack.entry(name)
            guard let layout = PlaneLayout.resolve(e),
                  let off = layout.offset(e), let bytes = layout.byteCount(e) else {
                throw QWInkling2Error.badState("DSpark QWP1 tensor invalid: \(name)")
            }
            return try pack.withRawPointer(at: off, byteCount: bytes) { ptr in
                try body(UnsafeRawBufferPointer(start: ptr, count: bytes))
            }
        }
    }

    func adviseDontNeed() {
        switch self {
        case .safetensors(let st): st.adviseDontNeed()
        case .qwp(let pack): pack.adviseDontNeed()
        }
    }
}

extension QWInkling2DSpark {
    @inline(__always)
    static func bf16ToF32(_ b: UInt16) -> Float { Float(bitPattern: UInt32(b) << 16) }

    @inline(__always)
    static func f32ToBF16(_ v: Float) -> UInt16 {
        var bits = v.bitPattern
        if (bits & 0x7F80_0000) == 0x7F80_0000 { return UInt16(truncatingIfNeeded: bits >> 16) }
        bits = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
        return UInt16(truncatingIfNeeded: bits >> 16)
    }

    static func dequantizeMatrixToF32(src: DSparkSource, name: String, nRows: Int, kCols: Int) throws -> [Float] {
        let e = try src.entry(of: name)
        return try src.withBytes(of: name) { raw in
            guard let base = raw.baseAddress else {
                throw QWInkling2Error.badState("empty bytes for \(name)")
            }
            if let e, PlaneLayout.resolve(e) == .hq4 {
                if e.isIncumbentHQ4 {
                    return QWQuant.Pack.dequantHQ4Slab(base: base, nRows: nRows, kCols: kCols)
                }
                if e.hq?.vq != "aq" {
                    return QWQuant.Pack.dequantHQBlockSlab(
                        base: base, nRows: nRows, kCols: kCols,
                        bits: e.hqBits, group: e.hqGroup, isSHVQ: e.isSHVQ
                    )
                }
                let g = kCols / e.hqGroup
                let blkBytes = QWAQBlock.bytes(bits: e.hqBits, group: e.hqGroup)
                var out = [Float](repeating: 0, count: nRows * kCols)
                var groupBuf = [Float]()
                groupBuf.reserveCapacity(e.hqGroup)
                for r in 0..<nRows {
                    for gg in 0..<g {
                        let ptr = base.advanced(by: (r * g + gg) * blkBytes)
                        QWAQBlock.unpack(group: UnsafeRawBufferPointer(start: ptr, count: blkBytes),
                                         bits: e.hqBits, group: e.hqGroup, into: &groupBuf)
                        let dst = r * kCols + gg * e.hqGroup
                        for i in 0..<e.hqGroup { out[dst + i] = groupBuf[i] }
                    }
                }
                return out
            }
            let count = nRows * kCols
            if e?.kind == "f32" || raw.count == count * 4 {
                let fp = raw.bindMemory(to: Float.self)
                return (0..<count).map { fp[$0] }
            }
            let bp = raw.bindMemory(to: UInt16.self)
            return (0..<count).map { bf16ToF32(bp[$0]) }
        }
    }

    static func computeAlphas(src: DSparkSource, rows: Int) throws -> [(Float, Float, Float, Float)] {
        guard rows == 4 else { throw QWInkling2Error.badState("expected 4 DSpark fuse rows") }
        return try src.withBytes(of: "fuse_weight") { raw in
            let p = raw.bindMemory(to: UInt16.self)
            var out: [(Float, Float, Float, Float)] = []
            for r in 0..<4 {
                let row = (0..<4).map { bf16ToF32(p[r * 4 + $0]) }
                let mx = row.max() ?? 0
                let ex = row.map { expf($0 - mx) }
                let sm = ex.reduce(0, +)
                let a = ex.map { bf16ToF32(f32ToBF16($0 / sm)) }
                out.append((a[0], a[1], a[2], a[3]))
            }
            return out
        }
    }

    static func unpackQuantHeadTile(
        basePtr: UnsafeRawPointer,
        view: QWWeightView,
        rowStart: Int,
        rowCount: Int,
        hidden H: Int,
        into headTile: inout [Double]
    ) {
        let tilePtr = basePtr.advanced(by: rowStart * view.rowBytes)
        if view.isIncumbentContainer {
            let f32 = QWQuant.Pack.dequantHQ4Slab(base: tilePtr, nRows: rowCount, kCols: H)
            for i in 0..<(rowCount * H) { headTile[i] = Double(f32[i]) }
            return
        }
        if view.isRotated {
            let f32 = QWQuant.Pack.dequantHQBlockSlab(
                base: tilePtr, nRows: rowCount, kCols: H,
                bits: view.bits, group: view.groupSize, isSHVQ: view.isSHVQ
            )
            for i in 0..<(rowCount * H) { headTile[i] = Double(f32[i]) }
            return
        }
        let gCount = H / view.groupSize
        let blkBytes = QWAQBlock.bytes(bits: view.bits, group: view.groupSize)
        var groupBuf = [Float]()
        groupBuf.reserveCapacity(view.groupSize)
        for r in 0..<rowCount {
            for gg in 0..<gCount {
                let blkPtr = tilePtr.advanced(by: (r * gCount + gg) * blkBytes)
                let rawBuf = UnsafeRawBufferPointer(start: blkPtr, count: blkBytes)
                QWAQBlock.unpack(group: rawBuf, bits: view.bits, group: view.groupSize, into: &groupBuf)
                let dstOff = r * H + gg * view.groupSize
                for i in 0..<view.groupSize { headTile[dstOff + i] = Double(groupBuf[i]) }
            }
        }
    }

    static func foldMarkovHead(
        src: DSparkSource, lmHeadStorage: QWHeadStorage,
        V: Int, H: Int, R: Int, foldWIsAQ8: Bool,
        dstW: any MTLBuffer, dstB: any MTLBuffer
    ) throws {
        let hiddenOutF32 = try dequantizeMatrixToF32(
            src: src, name: "markov_head.hidden_out.weight", nRows: H, kCols: R
        )
        var projF64 = [Double](repeating: 0, count: H * R)
        for i in 0..<(H * R) { projF64[i] = Double(hiddenOutF32[i]) }

        var biasF64 = [Double](repeating: 0, count: H)
        try src.withBytes(of: "markov_head.hidden_out.bias") { raw in
            let p = raw.bindMemory(to: UInt16.self)
            for i in 0..<H { biasF64[i] = Double(bf16ToF32(p[i])) }
        }

        let headRawPtr: UnsafeRawPointer
        let quantHeadView: QWWeightView?
        switch lmHeadStorage {
        case .bf16(let tv):
            headRawPtr = UnsafeRawPointer(tv.region.buffer.contents().advanced(by: tv.region.base + tv.offset))
            quantHeadView = nil
        case .q4g(let hw):
            let v = hw.view
            headRawPtr = UnsafeRawPointer(v.blocks.region.buffer.contents().advanced(by: v.blocks.region.base + v.blocks.offset))
            quantHeadView = v
        case .f16:
            throw QWInkling2Error.badState("unsupported .f16 lmHeadStorage in foldMarkovHead")
        }
        let outWPtr = dstW.contents()
        let outBPtr = dstB.contents().assumingMemoryBound(to: UInt16.self)
        let tileRows = 4096
        let numTiles = (V + tileRows - 1) / tileRows
        nonisolated(unsafe) let hUnsafe = headRawPtr, wUnsafe = outWPtr, bUnsafe = outBPtr

        projF64.withUnsafeBufferPointer { projBuf in
            biasF64.withUnsafeBufferPointer { biasBuf in
                guard let projBase = projBuf.baseAddress, let biasBase = biasBuf.baseAddress else { return }
                nonisolated(unsafe) let pU = projBase, biU = biasBase
                DispatchQueue.concurrentPerform(iterations: numTiles) { tile in
                    let start = tile * tileRows
                    let m = min(tileRows, V - start)
                    guard m > 0 else { return }
                    var headTile = [Double](repeating: 0, count: m * H)
                    var outWTile = [Double](repeating: 0, count: m * R)
                    var outBTile = [Double](repeating: 0, count: m)
                    if let qv = quantHeadView {
                        unpackQuantHeadTile(
                            basePtr: hUnsafe, view: qv, rowStart: start, rowCount: m, hidden: H, into: &headTile
                        )
                    } else {
                        let bfPtr = hUnsafe.assumingMemoryBound(to: UInt16.self).advanced(by: start * H)
                        for i in 0..<(m * H) { headTile[i] = Double(bf16ToF32(bfPtr[i])) }
                    }
                    headTile.withUnsafeBufferPointer { hB in
                        outWTile.withUnsafeMutableBufferPointer { wB in
                            outBTile.withUnsafeMutableBufferPointer { bB in
                                guard let hb = hB.baseAddress, let wb = wB.baseAddress, let bb = bB.baseAddress else { return }
                                qwInk2Dgemm(101, 111, 111, Int32(m), Int32(R), Int32(H), 1.0, hb, Int32(H), pU, Int32(R), 0.0, wb, Int32(R))
                                qwInk2Dgemm(101, 111, 111, Int32(m), 1, Int32(H), 1.0, hb, Int32(H), biU, 1, 0.0, bb, 1)
                            }
                        }
                    }
                    if foldWIsAQ8 {
                        let nGroups = R / 32
                        let rowBytes = nGroups * 36
                        let dstTile = wUnsafe.advanced(by: start * rowBytes).assumingMemoryBound(to: UInt8.self)
                        for row in 0..<m {
                            let rSrc = row * R
                            let rDst = dstTile.advanced(by: row * rowBytes)
                            for g in 0..<nGroups {
                                let gOff = rSrc + g * 32
                                var mn = outWTile[gOff], mx = mn
                                for j in 1..<32 {
                                    let v = outWTile[gOff + j]
                                    if v < mn { mn = v }
                                    if v > mx { mx = v }
                                }
                                let minF16 = Float16(mn)
                                let scaleF16 = Float16(max(0.0, mx - Double(minF16)) / 255.0)
                                let minD = Double(minF16), scaleD = Double(scaleF16)
                                let gDst = rDst.advanced(by: g * 36)
                                var sb = (scaleF16.bitPattern, minF16.bitPattern)
                                memcpy(gDst, &sb, 4)
                                let qDst = gDst.advanced(by: 4)
                                let inv = scaleD > 0 ? (1.0 / scaleD) : 0.0
                                for j in 0..<32 {
                                    let q = scaleD > 0
                                        ? Int(((outWTile[gOff + j] - minD) * inv).rounded(.toNearestOrEven))
                                        : 0
                                    qDst[j] = UInt8(clamping: q)
                                }
                            }
                        }
                    } else {
                        let dstWTile = wUnsafe.assumingMemoryBound(to: UInt16.self).advanced(by: start * R)
                        for i in 0..<(m * R) { dstWTile[i] = f32ToBF16(Float(outWTile[i])) }
                    }
                    let dstBTile = bUnsafe.advanced(by: start)
                    for i in 0..<m { dstBTile[i] = f32ToBF16(Float(outBTile[i])) }
                }
            }
        }
    }

    static func linearWeight(
        buffer: any MTLBuffer, offset: Int, byteLength: Int,
        n: Int, k: Int, kind: QWWeightView.ContainerKind?, name: String
    ) throws -> Ink2LinearW {
        if let kind {
            guard (byteLength % 2) == 0 else {
                throw QWInkling2Error.badState("unaligned DSpark QWP1 block bytes for \(name)")
            }
            let tv = QWTensorValue(region: QWMRegion(buffer: buffer, base: 0, length: buffer.length),
                                   offset: offset, shape: QWTensorShape([byteLength / 2], .f16))
            guard let view = QWWeightView(blocks: tv, n: n, k: k, containerKind: kind) else {
                throw QWInkling2Error.badState("invalid DSpark QWWeightView (\(n)x\(k)) for \(name)")
            }
            return .quant(view)
        }
        return .bf16(QWGemmFP.BF16Bank(
            weights: QWFPBufferSlice(buffer: buffer, offset: offset, bytes: max(0, buffer.length - offset)),
            n: n, k: k
        ))
    }
}
