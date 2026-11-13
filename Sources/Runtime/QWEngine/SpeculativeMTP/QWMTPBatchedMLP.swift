import Metal
import QWAlloc
import QWCore
import QWGEMM
import QWHAL

struct QWMTPBatchedMLP {
    let geometry: QWModelGeometry
    let hal: QWHal
    let rows: Int
    let sums: QWTensorValue
    let downRows: QWTensorValue
    private let downViews: [QWTensorValue]
    private let sumViews: [QWTensorValue]
    private let gateUp: [Int: any MTLComputePipelineState]
    private let gateUpFwht32: [Int: any MTLComputePipelineState]
    private let down: [Int: any MTLComputePipelineState]
    private let gateUpRowPaired: [Int: any MTLComputePipelineState]
    private let downRowPaired: [Int: any MTLComputePipelineState]
    /// From this many rows the 4-bit G32 kernels compute two weight rows per simdgroup,
    /// so each activation load serves both; below it the one-row kernels are faster.
    static let rowPairedMinimumRows = 4

    static func usesQ4GMKernel(_ weight: QWWeightView) -> Bool {
        weight.isIncumbentContainer || (!weight.isSHVQ && weight.bits == 4 && weight.groupSize == 32)
    }

    static func usesRowPairs(_ weight: QWWeightView, activeRows: Int) -> Bool {
        usesQ4GMKernel(weight) && activeRows >= rowPairedMinimumRows
    }

    static func sumStride(geometry: QWModelGeometry) -> Int {
        let hcaOut = geometry.hcaConfig.map { max(1, $0.oGroups) * $0.oLoraRank } ?? 0
        let maxOut = max(geometry.intermediate,
                         max(geometry.gdnZChannels,
                             max(geometry.qHeads * geometry.headDim,
                                 max(geometry.qHeads * geometry.vHeadDim, hcaOut))))
        return max(1, maxOut / 128)
    }

    static func allocationBytes(geometry: QWModelGeometry, rows: Int) -> Int {
        precondition((2...16).contains(rows))
        func aligned(_ bytes: Int) -> Int { (bytes + 255) / 256 * 256 }
        return aligned(rows * sumStride(geometry: geometry) * 4) + aligned(rows * geometry.hidden * 2)
    }

    init(geometry: QWModelGeometry, hal: QWHal, allocator: QWSubAllocator, rows: Int) throws {
        self.geometry = geometry
        self.hal = hal
        guard (2...16).contains(rows) else { throw QWEngineDriverError.state("batched MLP requires B=2...16") }
        self.rows = rows
        let maxKernelRows = min(8, rows)
        var gateUpPipelines: [Int: any MTLComputePipelineState] = [:]
        var gateUpFwht32Pipelines: [Int: any MTLComputePipelineState] = [:]
        var downPipelines: [Int: any MTLComputePipelineState] = [:]
        for count in 2...maxKernelRows {
            gateUpPipelines[count] = try hal.pipeline("qwm_mlp_gate_up_m\(count)")
            gateUpFwht32Pipelines[count] = try hal.pipeline("qwm_mlp_gate_up_fwht32_m\(count)")
            downPipelines[count] = try hal.pipeline("qwm_mlp_down_m\(count)")
        }
        var gateUpRowPairedPipelines: [Int: any MTLComputePipelineState] = [:]
        var downRowPairedPipelines: [Int: any MTLComputePipelineState] = [:]
        for count in 2...maxKernelRows where count >= Self.rowPairedMinimumRows {
            gateUpRowPairedPipelines[count] = try hal.pipeline("qwm_mlp_gate_up_rp_m\(count)")
            downRowPairedPipelines[count] = try hal.pipeline("qwm_mlp_down_rp_m\(count)")
        }
        gateUpRowPaired = gateUpRowPairedPipelines
        downRowPaired = downRowPairedPipelines
        gateUp = gateUpPipelines
        gateUpFwht32 = gateUpFwht32Pipelines
        down = downPipelines
        sums = try allocator.allocate(QWTensorShape([rows * Self.sumStride(geometry: geometry)], .f32))
        let sumStorage = sums
        let hcaOutCols = geometry.hcaConfig.map { max(1, $0.oGroups) * $0.oLoraRank } ?? 0
        let outChannels = max(geometry.gdnZChannels,
                              max(geometry.qHeads * geometry.headDim,
                                  max(geometry.qHeads * geometry.vHeadDim, hcaOutCols)))
        sumViews = (0..<rows).map { row in
            QWTensorValue(region: sumStorage.region, offset: sumStorage.offset + row * Self.sumStride(geometry: geometry) * 4,
                          shape: QWTensorShape([max(1, outChannels / 128)], .f32))
        }
        let storage = try allocator.allocate(QWTensorShape([rows * geometry.hidden], .f16))
        downRows = storage
        downViews = (0..<rows).map { row in
            QWTensorValue(region: storage.region, offset: storage.offset + row * geometry.hidden * 2,
                          shape: QWTensorShape([geometry.hidden], .f16))
        }
    }

    func downRow(_ row: Int) throws -> QWTensorValue {
        guard (0..<rows).contains(row) else { throw QWEngineDriverError.state("invalid batched MLP down row") }
        return downViews[row]
    }

    func outputSumRow(_ row: Int) throws -> QWTensorValue {
        guard (0..<rows).contains(row) else { throw QWEngineDriverError.state("invalid batched output sum row") }
        return sumViews[row]
    }

    func encodeProjection(_ enc: any MTLComputeCommandEncoder, input: QWTensorValue,
                          weight: QWWeightView, output: QWTensorValue, activeRows: Int? = nil,
                          preparedSums: Bool = false) throws {
        let count = activeRows ?? rows
        let attnOutCols = geometry.qHeads * geometry.headDim
        let vOutCols = geometry.qHeads * geometry.vHeadDim
        let hcaOutCols = geometry.hcaConfig.map { max(1, $0.oGroups) * $0.oLoraRank } ?? 0
        let hcaInCols = geometry.hcaConfig.map { $0.qLoraRank + 3 * $0.headDim } ?? 0
        let hcaQUpCols = geometry.hcaConfig.map { $0.qHeads * $0.headDim + max(1, $0.oGroups) * $0.oLoraRank } ?? 0
        let mlaLatentCols = geometry.qLoraRank + geometry.kvLoraRank + geometry.indexerHeadDim
        let mlaQBCols = attnOutCols + geometry.indexerHeads * geometry.indexerHeadDim
        let dMlaLatCols = geometry.hybridMLAConfig.map {
            $0.qLoraRank + $0.kvLoraRank + $0.qkRopeHeadDim + max(32, geometry.indexerHeadDim)
        } ?? 0
        let dMlaQBCols = geometry.hybridMLAConfig.map {
            $0.qHeads * ($0.qkNopeHeadDim + $0.qkRopeHeadDim + (geometry.attnOutputGate ? $0.vHeadDim : 0))
                + max(1, geometry.indexerHeads) * max(32, geometry.indexerHeadDim)
        } ?? 0
        let dMlaQRank = geometry.hybridMLAConfig?.qLoraRank ?? geometry.qLoraRank
        let kdaFgbCols = 2 * geometry.gdnLowRankDim + geometry.gdnVHeads
        let validProj = (weight.n == geometry.hidden
                && (weight.k == geometry.gdnZChannels || weight.k == attnOutCols
                    || weight.k == vOutCols || weight.k == hcaOutCols))
            || (weight.n == geometry.attnTotalN && weight.k == geometry.hidden)
            || (weight.n == geometry.gdnQKVChannels && weight.k == geometry.hidden)
            || (weight.n == kdaFgbCols && weight.k == geometry.hidden)
            || (weight.n == mlaLatentCols && weight.k == geometry.hidden)
            || (weight.n == mlaQBCols && weight.k == geometry.qLoraRank)
            || (weight.n == dMlaLatCols && weight.k == geometry.hidden)
            || (weight.n == dMlaQBCols && weight.k == dMlaQRank)
            || (weight.n == hcaInCols && weight.k == geometry.hidden)
            || (weight.n == hcaQUpCols && weight.k == (geometry.hcaConfig?.qLoraRank ?? -1))
        guard validProj else {
            throw QWEngineDriverError.state("unsupported batched trunk projection")
        }
        guard Self.usesQ4GMKernel(weight) else {
            try encodeAQGemvRows(enc, input: input, weight: weight, output: output,
                                 n: weight.n, k: weight.k, activeRows: count)
            return
        }
        if count > 8 && count <= rows,
           input.shape.dtype == .f16, input.shape.dims == [count * weight.k],
           output.shape.dtype == .f16, output.shape.dims.count == 1, output.shape.dims[0] >= count * weight.n,
           valid(input, alignment: 8), valid(output, alignment: 2), !overlaps(output, input), !overlaps(sums, input) {
            let c0 = count / 2
            let c1 = count - c0
            let in0 = input.slice(byteOffset: 0, count: c0 * weight.k)
            let out0 = output.slice(byteOffset: 0, count: c0 * weight.n)
            let in1 = input.slice(byteOffset: c0 * weight.k * 2, count: c1 * weight.k)
            let out1 = output.slice(byteOffset: c0 * weight.n * 2, count: c1 * weight.n)
            try encodeProjection(enc, input: in0, weight: weight, output: out0, activeRows: c0, preparedSums: preparedSums)
            try encodeProjection(enc, input: in1, weight: weight, output: out1, activeRows: c1, preparedSums: preparedSums)
            return
        }
        let rowPaired = Self.usesRowPairs(weight, activeRows: count)
        guard let pipeline = (rowPaired ? downRowPaired : down)[count] else {
            throw QWEngineDriverError.state("unsupported batched MLP active row count")
        }
        try encode(enc, input: input, weight: weight, output: output, pipeline: pipeline,
                   n: weight.n, k: weight.k, paired: false, activeRows: count,
                   preparedSums: preparedSums, rowPaired: rowPaired)
    }

    struct GDNControls {
        let aLog: QWTensorValue
        let bias: QWTensorValue
        let output: QWTensorValue
    }

    func encodeGDNProjection(_ enc: any MTLComputeCommandEncoder, input: QWTensorValue,
                             weight: QWWeightView, output: QWTensorValue,
                             controls: GDNControls, trunk: QWMTPBatchedTrunk,
                             activeRows: Int? = nil) throws {
        let count = activeRows ?? rows
        guard weight.n == geometry.gdnTotalN, weight.k == geometry.hidden,
              input.shape.dtype == .f16, input.shape.dims.count == 1,
              input.shape.dims[0] == count * geometry.hidden,
              output.shape.dtype == .f16, output.shape.dims.count == 1,
              output.shape.dims[0] >= count * geometry.gdnTotalN,
              valid(input, alignment: 8), valid(output, alignment: 2), valid(sums, alignment: 16),
              valid(weight.blocks, alignment: 4),
              controls.aLog.shape.dtype == .f32, controls.aLog.shape.dims.count == 1,
              controls.aLog.shape.dims[0] == geometry.gdnVHeads,
              controls.bias.shape.dtype == .f32, controls.bias.shape.dims.count == 1,
              controls.bias.shape.dims[0] == geometry.gdnVHeads,
              controls.output.shape.dtype == .f32, controls.output.shape.dims.count == 1,
              controls.output.shape.dims[0] >= count * (2 * geometry.gdnVHeads),
              valid(controls.aLog, alignment: 4), valid(controls.bias, alignment: 4),
              valid(controls.output, alignment: 4),
              !overlaps(controls.output, input), !overlaps(controls.output, output),
              !overlaps(controls.output, sums), !overlaps(controls.output, weight.blocks),
              !overlaps(controls.output, controls.aLog), !overlaps(controls.output, controls.bias),
              !overlaps(sums, controls.aLog), !overlaps(sums, controls.bias),
              !overlaps(output, controls.aLog), !overlaps(output, controls.bias) else {
            throw QWEngineDriverError.state("invalid batched GDN controls")
        }
        guard Self.usesQ4GMKernel(weight) else {
            try encodeAQGDNRows(enc, input: input, weight: weight, output: output,
                                controls: controls, n: geometry.gdnTotalN, k: geometry.hidden,
                                activeRows: count)
            return
        }
        if count > 8 && count <= rows {
            let c0 = count / 2
            let c1 = count - c0
            let ctrlStride = 2 * geometry.gdnVHeads
            let in0 = input.slice(byteOffset: 0, count: c0 * geometry.hidden)
            let out0 = output.slice(byteOffset: 0, count: c0 * geometry.gdnTotalN)
            let ctrl0 = GDNControls(aLog: controls.aLog, bias: controls.bias,
                                    output: controls.output.slice(byteOffset: 0, count: c0 * ctrlStride))
            let in1 = input.slice(byteOffset: c0 * geometry.hidden * 2, count: c1 * geometry.hidden)
            let out1 = output.slice(byteOffset: c0 * geometry.gdnTotalN * 2, count: c1 * geometry.gdnTotalN)
            let ctrl1 = GDNControls(aLog: controls.aLog, bias: controls.bias,
                                    output: controls.output.slice(byteOffset: c0 * ctrlStride * 4, count: c1 * ctrlStride))
            try encodeGDNProjection(enc, input: in0, weight: weight, output: out0, controls: ctrl0, trunk: trunk, activeRows: c0)
            try encodeGDNProjection(enc, input: in1, weight: weight, output: out1, controls: ctrl1, trunk: trunk, activeRows: c1)
            return
        }
        let rowPaired = Self.usesRowPairs(weight, activeRows: count)
        let pipeline = try trunk.gdnPipeline(forActiveRows: count, rowPaired: rowPaired)
        try encode(enc, input: input, weight: weight, output: output, pipeline: pipeline,
                   n: geometry.gdnTotalN, k: geometry.hidden, paired: false,
                   activeRows: count, controls: controls, rowPaired: rowPaired)
    }

    func encodeGateUp(_ enc: any MTLComputeCommandEncoder, input: QWTensorValue,
                      weight: QWWeightView, output: QWTensorValue,
                      activeRows: Int? = nil, fuseFwht32: Bool = false) throws {
        let count = activeRows ?? rows
        let useFwht = fuseFwht32 && geometry.intermediate > 0 && (geometry.intermediate % 32 == 0)
        guard Self.usesQ4GMKernel(weight) else {
            try encodeAQGateUpRows(enc, input: input, weight: weight, output: output,
                                   n: 2 * geometry.intermediate, k: geometry.hidden,
                                   activeRows: count, fuseFwht32: useFwht)
            return
        }
        if count > 8 && count <= rows,
           input.shape.dtype == .f16, input.shape.dims == [count * geometry.hidden],
           output.shape.dtype == .f16, output.shape.dims.count == 1,
           output.shape.dims[0] >= count * geometry.intermediate,
           valid(input, alignment: 8), valid(output, alignment: 2), !overlaps(output, input), !overlaps(sums, input) {
            let c0 = count / 2
            let c1 = count - c0
            let in0 = input.slice(byteOffset: 0, count: c0 * geometry.hidden)
            let out0 = output.slice(byteOffset: 0, count: c0 * geometry.intermediate)
            let in1 = input.slice(byteOffset: c0 * geometry.hidden * 2, count: c1 * geometry.hidden)
            let out1 = output.slice(byteOffset: c0 * geometry.intermediate * 2, count: c1 * geometry.intermediate)
            try encodeGateUp(enc, input: in0, weight: weight, output: out0, activeRows: c0, fuseFwht32: fuseFwht32)
            try encodeGateUp(enc, input: in1, weight: weight, output: out1, activeRows: c1, fuseFwht32: fuseFwht32)
            return
        }
        let rowPaired = !useFwht && Self.usesRowPairs(weight, activeRows: count)
        guard let pipeline = (useFwht ? gateUpFwht32[count] : (rowPaired ? gateUpRowPaired : gateUp)[count]) else {
            throw QWEngineDriverError.state("unsupported batched MLP active row count")
        }
        try encode(enc, input: input, weight: weight, output: output, pipeline: pipeline,
                   n: 2 * geometry.intermediate, k: geometry.hidden, paired: true,
                   fwht32: useFwht, activeRows: count, rowPaired: rowPaired)
    }

    func encodeDown(_ enc: any MTLComputeCommandEncoder, input: QWTensorValue,
                    weight: QWWeightView, activeRows: Int? = nil) throws {
        let count = activeRows ?? rows
        guard Self.usesQ4GMKernel(weight) else {
            try encodeAQGemvRows(enc, input: input, weight: weight, output: downRows,
                                 n: geometry.hidden, k: geometry.intermediate, activeRows: count)
            return
        }
        if count > 8 && count <= rows,
           input.shape.dtype == .f16, input.shape.dims == [count * geometry.intermediate],
           valid(input, alignment: 8), !overlaps(downRows, input), !overlaps(sums, input) {
            let c0 = count / 2
            let c1 = count - c0
            let in0 = input.slice(byteOffset: 0, count: c0 * geometry.intermediate)
            let out0 = downRows.slice(byteOffset: 0, count: c0 * geometry.hidden)
            let in1 = input.slice(byteOffset: c0 * geometry.intermediate * 2, count: c1 * geometry.intermediate)
            let out1 = downRows.slice(byteOffset: c0 * geometry.hidden * 2, count: c1 * geometry.hidden)
            let rp0 = Self.usesRowPairs(weight, activeRows: c0)
            let rp1 = Self.usesRowPairs(weight, activeRows: c1)
            guard let p0 = (rp0 ? downRowPaired : down)[c0],
                  let p1 = (rp1 ? downRowPaired : down)[c1] else {
                throw QWEngineDriverError.state("unsupported batched MLP active row count")
            }
            try encode(enc, input: in0, weight: weight, output: out0, pipeline: p0,
                       n: geometry.hidden, k: geometry.intermediate, paired: false,
                       activeRows: c0, rowPaired: rp0)
            try encode(enc, input: in1, weight: weight, output: out1, pipeline: p1,
                       n: geometry.hidden, k: geometry.intermediate, paired: false,
                       activeRows: c1, rowPaired: rp1)
            return
        }
        let rowPaired = Self.usesRowPairs(weight, activeRows: count)
        guard let pipeline = (rowPaired ? downRowPaired : down)[count] else {
            throw QWEngineDriverError.state("unsupported batched MLP active row count")
        }
        try encode(enc, input: input, weight: weight, output: downRows, pipeline: pipeline,
                   n: geometry.hidden, k: geometry.intermediate, paired: false,
                   activeRows: count, rowPaired: rowPaired)
    }

    private func encodeAQGemvRows(_ enc: any MTLComputeCommandEncoder, input: QWTensorValue,
                                  weight: QWWeightView, output: QWTensorValue,
                                  n: Int, k: Int, activeRows: Int) throws {
        guard (2...rows).contains(activeRows) else {
            throw QWEngineDriverError.state("batched MLP active row count is invalid")
        }
        let activeOutput = QWTensorValue(region: output.region, offset: output.offset,
            shape: QWTensorShape([activeRows * n], .f16))
        guard weight.n == n, weight.k == k,
              input.shape.dtype == .f16, input.shape.dims.count == 1, input.shape.dims[0] == activeRows * k,
              valid(input, alignment: 8), valid(activeOutput, alignment: 2),
              valid(weight.blocks, alignment: 4),
              !overlaps(activeOutput, input), !overlaps(activeOutput, weight.blocks) else {
            throw QWEngineDriverError.state("batched AQ GEMV geometry mismatch")
        }
        guard let kernelName = QWGemmAQ.decodeKernelName(bits: weight.bits, group: weight.groupSize,
                                                         isSHVQ: weight.isSHVQ) else {
            throw QWEngineDriverError.state("no AQ decode kernel for \(weight.bits)-bit group \(weight.groupSize)")
        }
        let pso = try hal.pipeline(kernelName)
        enc.memoryBarrier(scope: .buffers)
        enc.setComputePipelineState(pso)
        weight.blocks.bind(enc, index: 0)
        var kU = UInt32(k)
        enc.setBytes(&kU, length: 4, index: 5)
        var r0 = 0
        while r0 < activeRows {
            let rem = activeRows - r0
            let chunkRows = (rem > 4 && rem - 4 == 1) ? 3 : min(4, rem)
            let xChunk = input.slice(byteOffset: r0 * k * 2, count: chunkRows * k)
            let yChunk = activeOutput.slice(byteOffset: r0 * n * 2, count: chunkRows * n)
            xChunk.bind(enc, index: 1)
            yChunk.bind(enc, index: 3)
            yChunk.bind(enc, index: 8)
            var nU = UInt32(n) | 0x4000_0000 | (UInt32(chunkRows) << 24)
            enc.setBytes(&nU, length: 4, index: 4)
            enc.dispatchThreadgroups(MTLSize(width: (n + 15) / 16, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            r0 += chunkRows
        }
        enc.memoryBarrier(scope: .buffers)
    }

    private func encodeAQGateUpRows(_ enc: any MTLComputeCommandEncoder, input: QWTensorValue,
                                    weight: QWWeightView, output: QWTensorValue,
                                    n: Int, k: Int, activeRows: Int, fuseFwht32: Bool) throws {
        guard (2...rows).contains(activeRows) else {
            throw QWEngineDriverError.state("batched MLP active row count is invalid")
        }
        let intermediate = n / 2
        let activeOutput = QWTensorValue(region: output.region, offset: output.offset,
            shape: QWTensorShape([activeRows * intermediate], .f16))
        guard weight.n == n, weight.k == k,
              input.shape.dtype == .f16, input.shape.dims.count == 1, input.shape.dims[0] == activeRows * k,
              valid(input, alignment: 8), valid(activeOutput, alignment: 2),
              valid(weight.blocks, alignment: 4),
              !overlaps(activeOutput, input), !overlaps(activeOutput, weight.blocks) else {
            throw QWEngineDriverError.state("batched AQ gate/up geometry mismatch")
        }
        guard let kernelName = QWGemmAQ.siluPairKernelName(bits: weight.bits, group: weight.groupSize,
                                                           isSHVQ: weight.isSHVQ) else {
            throw QWEngineDriverError.state("no AQ silu_pair kernel for \(weight.bits)-bit group \(weight.groupSize)")
        }
        let pso = try hal.pipeline(kernelName)
        enc.memoryBarrier(scope: .buffers)
        enc.setComputePipelineState(pso)
        weight.blocks.bind(enc, index: 0)
        var kU = UInt32(k)
        enc.setBytes(&kU, length: 4, index: 5)
        let tgWidth = (intermediate + 31) / 32
        var r0 = 0
        while r0 < activeRows {
            let rem = activeRows - r0
            let chunkRows = (rem > 4 && rem - 4 == 1) ? 3 : min(4, rem)
            let xChunk = input.slice(byteOffset: r0 * k * 2, count: chunkRows * k)
            let yChunk = activeOutput.slice(byteOffset: r0 * intermediate * 2, count: chunkRows * intermediate)
            xChunk.bind(enc, index: 1)
            yChunk.bind(enc, index: 3)
            yChunk.bind(enc, index: 8)
            var nU = UInt32(n) | 0x4000_0000 | (UInt32(chunkRows) << 24) | (fuseFwht32 ? 0x8000_0000 : 0)
            enc.setBytes(&nU, length: 4, index: 4)
            enc.dispatchThreadgroups(MTLSize(width: tgWidth, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            r0 += chunkRows
        }
        enc.memoryBarrier(scope: .buffers)
    }

    private func encodeAQGDNRows(_ enc: any MTLComputeCommandEncoder, input: QWTensorValue,
                                 weight: QWWeightView, output: QWTensorValue,
                                 controls: GDNControls, n: Int, k: Int, activeRows: Int) throws {
        guard let kernelName = QWGemmAQ.gdnInProjDecodeKernelName(bits: weight.bits, group: weight.groupSize,
                                                                  isSHVQ: weight.isSHVQ) else {
            throw QWEngineDriverError.state("no AQ GDN decode kernel for \(weight.bits)-bit group \(weight.groupSize)")
        }
        let pso = try hal.pipeline(kernelName)
        let ctrlStride = 2 * geometry.gdnVHeads
        enc.memoryBarrier(scope: .buffers)
        enc.setComputePipelineState(pso)
        weight.blocks.bind(enc, index: 0)
        controls.aLog.bind(enc, index: 9)
        controls.bias.bind(enc, index: 10)
        var kU = UInt32(k)
        enc.setBytes(&kU, length: 4, index: 5)
        var r0 = 0
        while r0 < activeRows {
            let rem = activeRows - r0
            let chunkRows = (rem > 4 && rem - 4 == 1) ? 3 : min(4, rem)
            let xChunk = input.slice(byteOffset: r0 * k * 2, count: chunkRows * k)
            let yChunk = output.slice(byteOffset: r0 * n * 2, count: chunkRows * n)
            let cChunk = controls.output.slice(byteOffset: r0 * ctrlStride * 4, count: chunkRows * ctrlStride)
            xChunk.bind(enc, index: 1)
            yChunk.bind(enc, index: 3)
            yChunk.bind(enc, index: 8)
            cChunk.bind(enc, index: 11)
            cChunk.bind(enc, index: 12)
            var nU = UInt32(n) | 0x4000_0000 | (UInt32(chunkRows) << 24)
            enc.setBytes(&nU, length: 4, index: 4)
            enc.dispatchThreadgroups(MTLSize(width: (n + 15) / 16, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            r0 += chunkRows
        }
        enc.memoryBarrier(scope: .buffers)
    }

    private func encode(_ enc: any MTLComputeCommandEncoder, input: QWTensorValue,
                        weight: QWWeightView, output: QWTensorValue,
                        pipeline: any MTLComputePipelineState, n: Int, k: Int, paired: Bool,
                        fwht32: Bool = false, activeRows: Int, preparedSums: Bool = false,
                        controls: GDNControls? = nil, rowPaired: Bool = false) throws {
        guard (2...rows).contains(activeRows) else {
            throw QWEngineDriverError.state("batched MLP active row count is invalid")
        }
        let activeOutput = QWTensorValue(region: output.region, offset: output.offset,
            shape: QWTensorShape([activeRows * (paired ? n / 2 : n)], .f16))
        guard weight.n == n, weight.k == k,
              input.shape.dtype == .f16, input.shape.dims.count == 1, input.shape.dims[0] == activeRows * k,
              valid(input, alignment: 8), valid(activeOutput, alignment: 2), valid(sums, alignment: 16),
              valid(weight.blocks, alignment: 4),
              !overlaps(activeOutput, input), !overlaps(sums, input), !overlaps(activeOutput, sums),
              !overlaps(activeOutput, weight.blocks),
              !overlaps(sums, weight.blocks) else {
            throw QWEngineDriverError.state("batched MLP geometry mismatch")
        }
        _ = preparedSums
        enc.memoryBarrier(scope: .buffers)
        enc.setComputePipelineState(pipeline)
        weight.blocks.bind(enc, index: 0)
        input.bind(enc, index: 1)
        activeOutput.bind(enc, index: 3)
        var nU = UInt32(n) | (weight.isIncumbentContainer ? 0 : 0x8000_0000), kU = UInt32(k)
        enc.setBytes(&nU, length: 4, index: 4)
        enc.setBytes(&kU, length: 4, index: 5)
        if let controls {
            controls.aLog.bind(enc, index: 9)
            controls.bias.bind(enc, index: 10)
            controls.output.bind(enc, index: 11)
        }
        // Row-paired kernels run 64 threads: one gate/up pair, or two rows, per simdgroup.
        let tgWidth = paired ? (fwht32 ? (n / 2 + 31) / 32 : (n / 2 + (rowPaired ? 1 : 3)) / (rowPaired ? 2 : 4))
            : (n + 3) / 4
        let tptg = rowPaired ? 64 : (paired ? 256 : 128)
        enc.dispatchThreadgroups(MTLSize(width: tgWidth, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tptg, height: 1, depth: 1))
        enc.memoryBarrier(scope: .buffers)
    }

    private func valid(_ value: QWTensorValue, alignment: Int) -> Bool {
        let region = value.region
        return region.base >= 0 && region.length >= 0 && region.base <= region.buffer.length &&
            region.length <= region.buffer.length - region.base && value.offset >= 0 &&
            value.offset <= region.length && value.bytes <= region.length - value.offset &&
            (region.base + value.offset) % alignment == 0
    }

    private func overlaps(_ a: QWTensorValue, _ b: QWTensorValue) -> Bool {
        guard a.region.buffer === b.region.buffer else { return false }
        let startA = a.region.base + a.offset, startB = b.region.base + b.offset
        return startA < startB + b.bytes && startB < startA + a.bytes
    }
}
