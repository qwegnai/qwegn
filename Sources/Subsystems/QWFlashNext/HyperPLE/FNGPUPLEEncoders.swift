import Foundation
import Metal
import QWAlloc
import QWCore
import QWGEMM
import QWHAL

// FlashNext per-layer embedding dispatches (Sources/QWMetal/qwple.metal).
extension QWGemm {
    /// gv f32 [T, W] = sigmoid(sign(g) sqrt(|g|)) * vals, g = <keyNorm, queryNorm> per (token, stream) / sqrt(H).
    public static func encodeFnPLEGate(_ enc: any MTLComputeCommandEncoder, hal: QWHal, keyNorm: QWTensorValue,
                                       queryNorm: QWTensorValue, vals: QWTensorValue, gv: QWTensorValue,
                                       rows: Int, hidden: Int, hc: Int, valStride: Int? = nil) throws {
        if let valStride { precondition(valStride >= hidden) }
        enc.setComputePipelineState(try hal.pipeline(valStride == nil ? "qwg_ple_gate" : "qwg_ple_gate_strided"))
        enc.bindTensors(keyNorm, queryNorm, vals, gv, startIndex: 0)
        enc.setUInt32s(hidden, hc, startIndex: 4)
        if let valStride { enc.setUInt32s(valStride, startIndex: 6) }
        enc.dispatchThreadgroups(MTLSize(width: rows * hc, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    /// out f32 [T, W] = gv + SiLU(causal dilated depthwise conv over gvNorm).
    public static func encodeFnPLEConv(_ enc: any MTLComputeCommandEncoder, hal: QWHal, gv: QWTensorValue,
                                       gvNorm: QWTensorValue, convWeight: QWTensorValue, out: QWTensorValue,
                                       rows: Int, width: Int, kernel: Int, dilation: Int,
                                       hist: QWTensorValue? = nil) throws {
        enc.setComputePipelineState(try hal.pipeline("qwg_ple_conv"))
        enc.bindTensors(gv, gvNorm, convWeight, out, startIndex: 0)
        enc.setUInt32s(width, kernel, dilation, rows, startIndex: 4)
        // No history: bind any buffer; HL = 0 means the kernel never reads it.
        (hist ?? gv).bind(enc, index: 8)
        enc.setUInt32s(hist == nil ? 0 : (kernel - 1) * dilation, startIndex: 9)
        enc.dispatchThreads(MTLSize(width: width, height: rows, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    }

    /// Roll the PLE conv history: newHist = last HL rows of [oldHist | gvNorm]. HL = (K-1)*dilation.
    public static func encodeFnPLEHistory(_ enc: any MTLComputeCommandEncoder, hal: QWHal, oldHist: QWTensorValue,
                                          gvNorm: QWTensorValue, newHist: QWTensorValue, width: Int, historyRows: Int, rows: Int) throws {
        enc.setComputePipelineState(try hal.pipeline("qwg_ple_hist"))
        enc.bindTensors(oldHist, gvNorm, newHist, startIndex: 0)
        enc.setUInt32s(width, historyRows, rows, startIndex: 3)
        enc.dispatchThreads(MTLSize(width: width, height: historyRows, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    }
}

/// Resident Metal view over a QNGRAM v3 `vq2g32` file for zero-`pread` GPU PLE gather.
public struct FNGPUPLETable: @unchecked Sendable {
    public let buffer: any MTLBuffer
    public let headDim: Int
    public let entryCount: Int
    public let dataOffset: Int
    public let rowsPerSector: Int
    public let rowBytes: Int

    public init(buffer: any MTLBuffer, headDim: Int, entryCount: Int,
                dataOffset: Int, rowsPerSector: Int, rowBytes: Int) {
        self.buffer = buffer
        self.headDim = headDim
        self.entryCount = entryCount
        self.dataOffset = dataOffset
        self.rowsPerSector = rowsPerSector
        self.rowBytes = rowBytes
    }

    public init?(binRows: FNNgramBinRows, device: any MTLDevice) {
        guard let buf = binRows.makeResidentGPUBuffer(device: device) else { return nil }
        self.buffer = buf
        self.headDim = binRows.headDim
        self.entryCount = binRows.entryCount
        self.dataOffset = binRows.dataOffset
        self.rowsPerSector = binRows.rowsPerSector
        self.rowBytes = binRows.rowBytes
    }

    public static func isResidentMappingAdmissible(binRows: FNNgramBinRows, weightsURL: URL,
                                                   hasPLE: Bool, streamNgrams: Bool, hal: QWHal,
                                                   weightsEstimate: Int, fixedScratchBytes: Int,
                                                   minKVBytes: Int, availableBytes: Int) -> Bool {
        guard hasPLE, !streamNgrams else { return false }
        let packDiskBytes = (try? FileManager.default.attributesOfItem(atPath: weightsURL.path)[.size] as? Int) ?? 0
        let snap = QWDeviceMemorySnapshot(
            physicalBytes: availableBytes > 0 ? (availableBytes + QWCapacityBudget.osReserveBytes) : 0,
            reclaimableBytes: QWAdmission.reclaimableMemoryBytes(),
            maxWorkingSetBytes: hal.caps.maxWorkingSetBytes
        )
        return QWResidencyAdmission.isPLEAdmissible(
            weightsEstimate: weightsEstimate, packDiskBytes: packDiskBytes,
            fixedScratchBytes: fixedScratchBytes, pleFileBytes: binRows.fileBytes,
            minKVBytes: minKVBytes, snapshot: snap)
    }

    public static func openIfAdmissible(backing: any FNNgramRows, weightsURL: URL,
                                        hasPLE: Bool, streamNgrams: Bool = false, hal: QWHal,
                                        weightsEstimate: Int, fixedScratchBytes: Int,
                                        minKVBytes: Int, availableBytes: Int) -> FNGPUPLETable? {
        guard ProcessInfo.processInfo.environment["QWG_PLE_GPU_GATHER"] != "0",
              let binRows = backing as? FNNgramBinRows,
              binRows.dtype == .vq2g32,
              isResidentMappingAdmissible(
                  binRows: binRows, weightsURL: weightsURL, hasPLE: hasPLE, streamNgrams: streamNgrams,
                  hal: hal, weightsEstimate: weightsEstimate, fixedScratchBytes: fixedScratchBytes,
                  minKVBytes: minKVBytes, availableBytes: availableBytes) else { return nil }
        return FNGPUPLETable(binRows: binRows, device: hal.device)
    }

    public static func resolveBacking(assetsURL: URL, weightsURL: URL,
                                      hasPLE: Bool, streamNgrams: Bool, hal: QWHal,
                                      weightsEstimate: Int, fixedScratchBytes: Int,
                                      minKVBytes: Int, availableBytes: Int) throws -> (rows: any FNNgramRows, gpuTable: FNGPUPLETable?, residentBytes: Int) {
        let baseRows = try FNNgramRowsLoader.open(assetsURL: assetsURL, mapped: false)
        let gpuPLE = openIfAdmissible(
            backing: baseRows, weightsURL: weightsURL, hasPLE: hasPLE, streamNgrams: streamNgrams,
            hal: hal, weightsEstimate: weightsEstimate, fixedScratchBytes: fixedScratchBytes,
            minKVBytes: minKVBytes, availableBytes: availableBytes)
        if let gpuPLE {
            return (baseRows, gpuPLE, gpuPLE.buffer.length)
        }
        if let binRows = baseRows as? FNNgramBinRows,
           isResidentMappingAdmissible(
               binRows: binRows, weightsURL: weightsURL, hasPLE: hasPLE, streamNgrams: streamNgrams,
               hal: hal, weightsEstimate: weightsEstimate, fixedScratchBytes: fixedScratchBytes,
               minKVBytes: minKVBytes, availableBytes: availableBytes) {
            let mappedRows = try FNNgramRowsLoader.open(assetsURL: assetsURL, mapped: true)
            return (mappedRows, nil, binRows.fileBytes)
        }
        return (baseRows, nil, 0)
    }
}

extension QWGemm {
    /// Gathers `numTargets * rowsPerTarget` VQ2_G32 rows directly from resident `table.buffer` into `outRegion`.
    public static func encodeFnPLEVQ2Gather(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                            table: FNGPUPLETable, rowIds: QWTensorValue,
                                            targetFloatOffsets: QWTensorValue, outRegion: QWMRegion,
                                            rowsPerTarget: Int, numTargets: Int) throws {
        guard rowsPerTarget > 0, numTargets > 0 else { return }
        precondition(table.headDim % 32 == 0, "VQ2 PLE gather requires headDim % 32 == 0")
        enc.setComputePipelineState(try hal.pipeline("qwg_ple_vq2_gather"))
        enc.setBuffer(table.buffer, offset: 0, index: 0)
        rowIds.bind(enc, index: 1)
        targetFloatOffsets.bind(enc, index: 2)
        enc.setBuffer(outRegion.buffer, offset: outRegion.base, index: 3)
        enc.setUInt32s(table.headDim, rowsPerTarget, numTargets,
                       table.dataOffset, table.rowsPerSector, table.rowBytes, startIndex: 4)
        let groupsPerRow = table.headDim / 32
        let totalTeams = numTargets * rowsPerTarget * groupsPerRow
        let tgs = max(1, min(1024, (totalTeams + 15) / 16))
        enc.dispatchThreadgroups(MTLSize(width: tgs, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Residual injection into the HC streams: out[t, c, :] = base[t, c, :] + src[t, :] * inj[t, c].
    public static func encodeFnInject(_ enc: any MTLComputeCommandEncoder, hal: QWHal, base: QWTensorValue,
                                      src: QWTensorValue, inj: QWTensorValue, out: QWTensorValue,
                                      rows: Int, hidden: Int, hc: Int) throws {
        enc.setComputePipelineState(try hal.pipeline("qwg_inject"))
        enc.bindTensors(base, src, inj, out, startIndex: 0)
        enc.setUInt32s(hidden, hc, startIndex: 4)
        enc.dispatchThreads(MTLSize(width: hc * hidden, height: rows, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
}
