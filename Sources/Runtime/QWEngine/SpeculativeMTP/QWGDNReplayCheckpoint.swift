import Metal
import QWAlloc
import QWCore
import QWGDN
import QWState
import QWHAL

/// Single-flight GPU workspace. The round owner must finish capture before
/// scratch reuse and check GPU completion before publishing restored metadata.
final class QWGDNReplayCheckpoint {
    let geometry: QWModelGeometry
    let layers: Int
    let rows: Int
    let recurrent: QWTensorValue
    let rings: QWTensorValue
    private let raw: QWTensorValue
    private let normalized: QWTensorValue
    private let betas: QWTensorValue
    private let glog: QWTensorValue
    private let innovations: QWTensorValue
    private var sourceRec: QWTensorValue?
    private var sourceRing: QWTensorValue?
    private var captured: [UInt32]
    private let fusedCapture: Bool

    static func allocationBytes(geometry: QWModelGeometry, layers: Int, rows: Int) -> Int {
        precondition((0...128).contains(layers) && (1...16).contains(rows))
        if layers == 0 { return 7 * 256 }
        func aligned(_ bytes: Int) -> Int { (bytes + 255) / 256 * 256 }
        let qkvC = geometry.gdnQKVChannels, gdnVH = geometry.gdnVHeads, zC = geometry.gdnZChannels
        let glogC = geometry.gdnGlogChannels
        return aligned(layers * QWSlotGeometry.recElems(geometry: geometry) * 4)
            + aligned(layers * QWSlotGeometry.ringElems(geometry: geometry) * 2)
            + 2 * aligned(layers * rows * qkvC * 2)
            + aligned(layers * rows * gdnVH * 4)
            + aligned(layers * rows * glogC * 4)
            + aligned(layers * rows * zC * 4)
    }

    init(geometry: QWModelGeometry, allocator: QWSubAllocator, layers: Int = 48, rows: Int, fusedCapture: Bool = false) throws {
        self.geometry = geometry
        guard layers >= 0, layers <= 128, (1...16).contains(rows) else {
            throw QWEngineDriverError.state("GPU checkpoint requires 0...128 layers and 1...16 rows")
        }
        self.layers = layers
        self.rows = rows
        self.fusedCapture = fusedCapture
        captured = [UInt32](repeating: 0, count: layers)
        let qkvC = geometry.gdnQKVChannels, gdnVH = geometry.gdnVHeads, zC = geometry.gdnZChannels
        let glogC = geometry.gdnGlogChannels
        let recElems = max(1, layers * QWSlotGeometry.recElems(geometry: geometry))
        let ringElems = max(1, layers * QWSlotGeometry.ringElems(geometry: geometry))
        let rowQKV = max(1, layers * rows * qkvC)
        let rowVH = max(1, layers * rows * gdnVH)
        let rowGlog = max(1, layers * rows * glogC)
        let rowZ = max(1, layers * rows * zC)
        recurrent = try allocator.allocate(QWTensorShape([recElems], .f32))
        rings = try allocator.allocate(QWTensorShape([ringElems], .f16))
        raw = try allocator.allocate(QWTensorShape([rowQKV], .f16))
        normalized = try allocator.allocate(QWTensorShape([rowQKV], .f16))
        betas = try allocator.allocate(QWTensorShape([rowVH], .f32))
        glog = try allocator.allocate(QWTensorShape([rowGlog], .f32))
        innovations = try allocator.allocate(QWTensorShape([rowZ], .f32))
    }

    private func view(_ tensor: QWTensorValue, index: Int, count: Int) -> QWTensorValue {
        let width = tensor.shape.dtype == .f32 ? 4 : 2
        return QWTensorValue(region: tensor.region, offset: tensor.offset + index * count * width,
                             shape: QWTensorShape([count], tensor.shape.dtype))
    }

    private func slice(_ tensor: QWTensorValue, startRow: Int, rowCount: Int, rowWidth: Int) -> QWTensorValue {
        let width = tensor.shape.dtype == .f32 ? 4 : 2
        return QWTensorValue(region: tensor.region, offset: tensor.offset + startRow * rowWidth * width,
                             shape: QWTensorShape([rowCount * rowWidth], tensor.shape.dtype))
    }

    private func validate(_ t: QWTensorValue, like other: QWTensorValue) throws {
        guard t.shape.dtype == other.shape.dtype, t.bytes == other.bytes,
              t.offset >= 0, t.region.base >= 0, t.offset <= t.region.length,
              t.bytes <= t.region.length - t.offset,
              t.region.base <= t.region.buffer.length,
              t.region.length <= t.region.buffer.length - t.region.base,
              (t.region.base + t.offset) % 4 == 0 else {
            throw QWEngineDriverError.state("GPU checkpoint tensor shape or extent is invalid")
        }
        if t.region.buffer === recurrent.region.buffer {
            let start = t.region.base + t.offset
            let end = start + t.bytes
            let workspaceStart = recurrent.region.base + recurrent.offset
            let workspaceEnd = innovations.region.base + innovations.offset + innovations.bytes
            guard end <= workspaceStart || workspaceEnd <= start else {
                throw QWEngineDriverError.state("GPU checkpoint source aliases workspace")
            }
        }
    }

    private func sameStorage(_ a: QWTensorValue, _ b: QWTensorValue?) -> Bool {
        guard let b else { return false }
        return a.region.buffer === b.region.buffer
            && a.region.base + a.offset == b.region.base + b.offset && a.bytes == b.bytes
    }

    private func validateRetained(_ count: Int) throws {
        guard sourceRec != nil, sourceRing != nil, (0...rows).contains(count) else {
            throw QWEngineDriverError.state("GPU replay has no checkpoint or retained row count is invalid")
        }
        let required = UInt32((1 << count) - 1)
        for layer in 0..<layers where captured[layer] & required != required {
            throw QWEngineDriverError.state("GPU replay journal is incomplete")
        }
    }

    func validateCompleteCapture(activeRows: Int? = nil) throws {
        try validateRetained(activeRows ?? rows)
    }

    func validateResolution(rec: QWTensorValue, ring: QWTensorValue,
                            activeRows: Int? = nil) throws {
        if layers == 0 {
            try validateCompleteCapture(activeRows: activeRows)
            return
        }
        try validate(rec, like: recurrent)
        try validate(ring, like: rings)
        try validateCompleteCapture(activeRows: activeRows)
        guard sameStorage(rec, sourceRec), sameStorage(ring, sourceRing) else {
            throw QWEngineDriverError.state("GPU resolution requires original checkpoint state buffers")
        }
    }

    private func copy(_ enc: any MTLBlitCommandEncoder, _ source: QWTensorValue,
                      _ destination: QWTensorValue) {
        enc.copy(from: source.region.buffer, sourceOffset: source.region.base + source.offset,
                 to: destination.region.buffer, destinationOffset: destination.region.base + destination.offset,
                 size: source.bytes)
    }

    func encodeBackup(_ enc: any MTLBlitCommandEncoder,
                      rec: QWTensorValue, ring: QWTensorValue) throws {
        if layers == 0 {
            sourceRec = rec
            sourceRing = ring
            return
        }
        try validate(rec, like: recurrent)
        try validate(ring, like: rings)
        if rec.region.buffer === ring.region.buffer {
            let recStart = rec.region.base + rec.offset, ringStart = ring.region.base + ring.offset
            guard recStart + rec.bytes <= ringStart || ringStart + ring.bytes <= recStart else {
                throw QWEngineDriverError.state("GPU checkpoint recurrent and ring slabs alias")
            }
        }
        copy(enc, rec, recurrent)
        copy(enc, ring, rings)
        sourceRec = rec
        sourceRing = ring
        for layer in 0..<layers { captured[layer] = 0 }
    }

    func encodeRestore(_ enc: any MTLBlitCommandEncoder,
                       rec: QWTensorValue, ring: QWTensorValue) throws {
        if layers == 0 {
            guard sourceRec != nil, sourceRing != nil else {
                throw QWEngineDriverError.state("GPU restore requires the checkpoint's original state buffers")
            }
            return
        }
        try validate(rec, like: recurrent)
        try validate(ring, like: rings)
        guard sameStorage(rec, sourceRec), sameStorage(ring, sourceRing) else {
            throw QWEngineDriverError.state("GPU restore requires the checkpoint's original state buffers")
        }
        copy(enc, recurrent, rec)
        copy(enc, rings, ring)
    }

    func encodeCapture(_ enc: any MTLBlitCommandEncoder, layer: Int, row: Int,
                       rawInput: QWTensorValue, postConv: QWTensorValue,
                       beta: QWTensorValue, decay: QWTensorValue) throws {
        guard sourceRec != nil, sourceRing != nil,
              (0..<layers).contains(layer), (0..<rows).contains(row) else {
            throw QWEngineDriverError.state("GPU replay capture index is invalid")
        }
        let index = layer * rows + row
        let qkvC = geometry.gdnQKVChannels, gdnVH = geometry.gdnVHeads, glogC = geometry.gdnGlogChannels
        let r = view(raw, index: index, count: qkvC)
        let n = view(normalized, index: index, count: qkvC)
        let b = view(betas, index: index, count: gdnVH)
        let g = view(glog, index: index, count: glogC)
        try validate(rawInput, like: r)
        try validate(postConv, like: n)
        try validate(beta, like: b)
        try validate(decay, like: g)
        copy(enc, rawInput, r)
        copy(enc, postConv, n)
        copy(enc, beta, b)
        copy(enc, decay, g)
        captured[layer] |= UInt32(1 << row)
    }

    /// Serial compute capture keeps the shipping row encoder open. A concurrent
    /// caller must fence producers before capture and capture before scratch reuse.
    func encodeCapture(_ enc: any MTLComputeCommandEncoder, hal: QWHal, layer: Int, row: Int,
                       rawInput: QWTensorValue, postConv: QWTensorValue,
                       beta: QWTensorValue, decay: QWTensorValue) throws {
        guard sourceRec != nil, sourceRing != nil,
              (0..<layers).contains(layer), (0..<rows).contains(row) else {
            throw QWEngineDriverError.state("GPU replay capture index is invalid")
        }
        let index = layer * rows + row
        let qkvC = geometry.gdnQKVChannels, gdnVH = geometry.gdnVHeads, glogC = geometry.gdnGlogChannels
        let r = view(raw, index: index, count: qkvC)
        let n = view(normalized, index: index, count: qkvC)
        let b = view(betas, index: index, count: gdnVH)
        let g = view(glog, index: index, count: glogC)
        try validate(rawInput, like: r)
        try validate(postConv, like: n)
        try validate(beta, like: b)
        try validate(decay, like: g)
        if fusedCapture {
            enc.setComputePipelineState(try hal.pipeline("qwg_gdn_capture_replay"))
            rawInput.bind(enc, index: 0)
            postConv.bind(enc, index: 1)
            beta.bind(enc, index: 2)
            decay.bind(enc, index: 3)
            r.bind(enc, index: 4)
            n.bind(enc, index: 5)
            b.bind(enc, index: 6)
            g.bind(enc, index: 7)
            let qkvWords = max(qkvC / 2, glogC)
            let tgCount = (qkvWords + 255) / 256
            enc.dispatchThreadgroups(MTLSize(width: tgCount, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        } else {
            try QWVerifyWindow.encodeCopy(enc, hal: hal, source: rawInput, destination: r)
            try QWVerifyWindow.encodeCopy(enc, hal: hal, source: postConv, destination: n)
            try QWVerifyWindow.encodeCopy(enc, hal: hal, source: beta, destination: b)
            try QWVerifyWindow.encodeCopy(enc, hal: hal, source: decay, destination: g)
        }
        captured[layer] |= UInt32(1 << row)
    }

    func innovationRow(layer: Int, row: Int) throws -> QWTensorValue {
        guard (0..<layers).contains(layer), (0..<rows).contains(row) else {
            throw QWEngineDriverError.state("GPU innovation tape index is invalid")
        }
        return view(innovations, index: layer * rows + row, count: geometry.gdnZChannels)
    }

    func encodeBatchedKDA(_ enc: any MTLComputeCommandEncoder,
                          conv: QWGDNConv, decoder: QWGDNDecode,
                          layer: Int, activeRows: Int,
                          gdnProjection: QWTensorValue, fgbProjection: QWTensorValue,
                          fBWeight: QWTensorValue, gBWeight: QWTensorValue,
                          aLog: QWTensorValue, dtBias: QWTensorValue,
                          convWeight: QWTensorValue, normWeight: QWTensorValue,
                          zScratch: QWTensorValue,
                          ring: QWTensorValue, state: QWTensorValue,
                          output: QWTensorValue) throws {
        guard sourceRec != nil, sourceRing != nil,
              (0..<layers).contains(layer), (2...rows).contains(activeRows) else {
            throw QWEngineDriverError.state("GPU batched KDA capture index or active row count is invalid")
        }
        let startRow = layer * rows
        let qkvC = geometry.gdnQKVChannels, gdnVH = geometry.gdnVHeads, zC = geometry.gdnZChannels
        let glogC = geometry.gdnGlogChannels
        let r = slice(raw, startRow: startRow, rowCount: activeRows, rowWidth: qkvC)
        let n = slice(normalized, startRow: startRow, rowCount: activeRows, rowWidth: qkvC)
        let b = slice(betas, startRow: startRow, rowCount: activeRows, rowWidth: gdnVH)
        let g = slice(glog, startRow: startRow, rowCount: activeRows, rowWidth: glogC)
        let inv = slice(innovations, startRow: startRow, rowCount: activeRows, rowWidth: zC)
        let xView = QWTensorValue(region: gdnProjection.region, offset: gdnProjection.offset,
                                  shape: QWTensorShape([activeRows * qkvC], .f16))
        let zView = QWTensorValue(region: zScratch.region, offset: zScratch.offset,
                                  shape: QWTensorShape([activeRows * zC], .f16))
        try decoder.encodeKDAGates(
            enc, fgb: fgbProjection, fB: fBWeight, gB: gBWeight,
            aLog: aLog, dtBias: dtBias, betas: b, glog: g, zOut: zView, rows: activeRows)
        try conv.encodeRowsWithHeadnorm(
            enc, x: xView, w: convWeight, bias: nil,
            stIn: ring, y: n, stOut: ring,
            rows: activeRows, xStride: qkvC, rawCapture: r)
        enc.memoryBarrier(scope: .buffers)
        try decoder.encodeRows(
            enc, y: n, betas: b, glog: g, state: state,
            zBase: zView, w: normWeight, o: output, innovation: inv,
            rows: activeRows, zStride: zC)
        enc.memoryBarrier(scope: .buffers)
        captured[layer] |= UInt32((1 << activeRows) - 1)
    }

    func encodeBatchedGDN(_ enc: any MTLComputeCommandEncoder,
                          conv: QWGDNConv, decoder: QWGDNDecode,
                          layer: Int, activeRows: Int,
                          gdnProjection: QWTensorValue, controls: QWTensorValue,
                          convWeight: QWTensorValue, normWeight: QWTensorValue,
                          ring: QWTensorValue, state: QWTensorValue,
                          output: QWTensorValue) throws {
        guard sourceRec != nil, sourceRing != nil,
              (0..<layers).contains(layer), (2...rows).contains(activeRows) else {
            throw QWEngineDriverError.state("GPU batched GDN capture index or active row count is invalid")
        }
        let startRow = layer * rows
        let qkvC = geometry.gdnQKVChannels, gdnVH = geometry.gdnVHeads, zC = geometry.gdnZChannels
        let r = slice(raw, startRow: startRow, rowCount: activeRows, rowWidth: qkvC)
        let n = slice(normalized, startRow: startRow, rowCount: activeRows, rowWidth: qkvC)
        let b = slice(betas, startRow: startRow, rowCount: activeRows, rowWidth: gdnVH)
        let g = slice(glog, startRow: startRow, rowCount: activeRows, rowWidth: gdnVH)
        let inv = slice(innovations, startRow: startRow, rowCount: activeRows, rowWidth: zC)
        let xView = QWTensorValue(region: gdnProjection.region, offset: gdnProjection.offset,
                                  shape: QWTensorShape([activeRows * geometry.gdnTotalN], .f16))
        let cView = QWTensorValue(region: controls.region, offset: controls.offset,
                                  shape: QWTensorShape([activeRows * (2 * gdnVH)], .f32))
        try conv.encodeRowsWithHeadnorm(
            enc, x: xView, w: convWeight, bias: nil,
            stIn: ring, y: n, stOut: ring,
            rows: activeRows, xStride: geometry.gdnTotalN,
            rawCapture: r, controls: cView, betasCapture: b, glogCapture: g)
        enc.memoryBarrier(scope: .buffers)
        let zSpan = (activeRows - 1) * geometry.gdnTotalN + zC
        let zBase = QWTensorValue(
            region: gdnProjection.region,
            offset: gdnProjection.offset + qkvC * 2,
            shape: QWTensorShape([zSpan], .f16))
        try decoder.encodeRows(
            enc, y: n, betas: b, glog: g, state: state,
            zBase: zBase, w: normWeight, o: output, innovation: inv,
            rows: activeRows, zStride: geometry.gdnTotalN)
        enc.memoryBarrier(scope: .buffers)
        captured[layer] |= UInt32((1 << activeRows) - 1)
    }

    func encodeReplay(_ enc: any MTLComputeCommandEncoder, decoder: QWGDNDecode,
                      rec: QWTensorValue, retaining count: Int) throws {
        if layers == 0 {
            try validateRetained(count)
            return
        }
        try validate(rec, like: recurrent)
        try validateRetained(count)
        guard sameStorage(rec, sourceRec) else {
            throw QWEngineDriverError.state("GPU replay requires the checkpoint's original recurrent slab")
        }
        let qkvC = geometry.gdnQKVChannels, gdnVH = geometry.gdnVHeads, glogC = geometry.gdnGlogChannels
        for layer in 0..<layers {
            let state = view(rec, index: layer, count: QWSlotGeometry.recElems(geometry: geometry))
            for row in 0..<count {
                let index = layer * rows + row
                try decoder.encodeStateReplay(enc, y: view(normalized, index: index, count: qkvC),
                                               betas: view(betas, index: index, count: gdnVH),
                                               glog: view(glog, index: index, count: glogC), state: state)
            }
        }
    }

    /// Returns false for rows whose inverse would amplify rounding enough to
    /// make snapshot-and-replay the safer path.
    func rollbackTapeIsStable(retaining count: Int, activeRows: Int? = nil) throws -> Bool {
        let verifiedRows = activeRows ?? rows
        guard (2...rows).contains(verifiedRows), (0...verifiedRows).contains(count) else {
            throw QWEngineDriverError.state("GPU rollback active row count is invalid")
        }
        try validateRetained(count)
        guard layers > 0, count < verifiedRows else { return true }
        let decayValues = glog.region.pointer.advanced(by: glog.offset).assumingMemoryBound(to: Float.self)
        let innovationValues = innovations.region.pointer.advanced(by: innovations.offset)
            .assumingMemoryBound(to: Float.self)
        let normalizedValues = normalized.region.pointer.advanced(by: normalized.offset)
            .assumingMemoryBound(to: UInt16.self)
        let qkvC = geometry.gdnQKVChannels, gdnVH = geometry.gdnVHeads, gdnHD = geometry.gdnHeadDim
        let zC = geometry.gdnZChannels, kOffset = geometry.gdnQKHeads * gdnHD
        let glogC = geometry.gdnGlogChannels
        let vPerQK = gdnVH / geometry.gdnQKHeads
        for layer in 0..<layers {
            for row in count..<verifiedRows {
                let rowIndex = layer * rows + row
                for head in 0..<gdnVH {
                    if geometry.gdnVectorDecay {
                        let decayBase = rowIndex * glogC + head * gdnHD
                        for channel in 0..<gdnHD {
                            let decay = decayValues[decayBase + channel]
                            guard decay >= 1.0e-3 && decay <= 1.0e38 else { return false }
                        }
                    } else {
                        let decay = decayValues[rowIndex * glogC + head]
                        guard decay >= -6.907755 && decay <= 88.0 else { return false }
                    }
                    let innovationStart = rowIndex * zC + head * gdnHD
                    for channel in 0..<gdnHD where !innovationValues[innovationStart + channel].isFinite {
                        return false
                    }
                    let keyStart = rowIndex * qkvC + kOffset + (head / vPerQK) * gdnHD
                    for channel in 0..<gdnHD
                    where !Float16(bitPattern: normalizedValues[keyStart + channel]).isFinite {
                        return false
                    }
                }
            }
        }
        return true
    }

    func encodeRollbackTape(_ enc: any MTLComputeCommandEncoder, decoder: QWGDNDecode,
                            rec: QWTensorValue, activeRows: Int? = nil,
                            retaining count: Int) throws {
        let verifiedRows = activeRows ?? rows
        if layers == 0 {
            guard (2...rows).contains(verifiedRows), (0...verifiedRows).contains(count) else {
                throw QWEngineDriverError.state("GPU rollback active row count is invalid")
            }
            try validateRetained(count)
            return
        }
        guard (2...rows).contains(verifiedRows), (0...verifiedRows).contains(count) else {
            throw QWEngineDriverError.state("GPU rollback active row count is invalid")
        }
        try validate(rec, like: recurrent)
        try validateRetained(count)
        guard sameStorage(rec, sourceRec) else {
            throw QWEngineDriverError.state("GPU rollback requires the original recurrent slab")
        }
        guard try rollbackTapeIsStable(retaining: count, activeRows: verifiedRows) else {
            throw QWEngineDriverError.state("GPU innovation tape is ill-conditioned; restore and replay instead")
        }
        let rollbackRows = verifiedRows - count
        if rollbackRows > 0 {
            let qkvC = geometry.gdnQKVChannels, zC = geometry.gdnZChannels, glogC = geometry.gdnGlogChannels
            for layer in 0..<layers {
                let state = view(rec, index: layer, count: QWSlotGeometry.recElems(geometry: geometry))
                let startRow = layer * rows + count
                try decoder.encodeStateRollbackRows(
                    enc,
                    y: slice(normalized, startRow: startRow, rowCount: rollbackRows, rowWidth: qkvC),
                    innovation: slice(innovations, startRow: startRow, rowCount: rollbackRows, rowWidth: zC),
                    glog: slice(glog, startRow: startRow, rowCount: rollbackRows, rowWidth: glogC),
                    state: state,
                    rows: rollbackRows)
            }
            enc.memoryBarrier(scope: .buffers)
        }
    }

    /// A 3-row raw-input ring needs no convolution to reconstruct its history.
    func encodeReplayRings(_ enc: any MTLBlitCommandEncoder,
                           ring: QWTensorValue, retaining count: Int) throws {
        if layers == 0 {
            try validateRetained(count)
            return
        }
        try validate(ring, like: rings)
        try validateRetained(count)
        guard sameStorage(ring, sourceRing) else {
            throw QWEngineDriverError.state("GPU replay requires the checkpoint's original ring slab")
        }
        let qkvC = geometry.gdnQKVChannels
        for layer in 0..<layers {
            for age in 0..<3 {
                let destination = view(ring, index: layer * 3 + age, count: qkvC)
                let source = age + count < 3
                    ? view(rings, index: layer * 3 + age + count, count: qkvC)
                    : view(raw, index: layer * rows + age + count - 3, count: qkvC)
                copy(enc, source, destination)
            }
        }
    }
}
