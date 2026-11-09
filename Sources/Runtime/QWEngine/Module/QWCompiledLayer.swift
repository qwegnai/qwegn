// QWCompiledLayer.swift — Unified 4-Seam Compiled Layer & Stack Schedule.
// Compiles any QWEnginePlan into a flat array of concrete 4-seam layer descriptors
// (StreamCarrier, SequenceMixer, ChannelMixer, optional PLE) and command-buffer cuts.

import Foundation
import QWCore
import QWFlashNext

/// Stream carrier operator seam across model architectures.
public enum QWStreamCarrierKind: String, Sendable, Equatable, Codable {
    /// Standard 1-stream RMSNorm + residual (`x ∈ R^H`).
    case singleStream
    /// Multi-stream Hyper-Connections with Sinkhorn-Knopp doubly-stochastic mixing (`mHC`).
    case mhc
    /// Multi-lane (`hc * H`) low-rank gated Hyper-Connection carrier (`QWHCOperator`).
    case hyperConnection
}

/// Sequence mixer operator seam across model architectures.
public enum QWSequenceMixerKind: String, Sendable, Equatable, Codable {
    /// Linear attention / Gated DeltaNet (`QWGDN`).
    case gdn
    /// Paged GQA full attention (`QWAttn`).
    case fullAttention
    /// Query-Sparse Attention with block-pooled indexer & LSA (`QWQSAOperator`).
    case qsaSparse
    /// Multi-head Latent Attention (`QWMLAOperator`).
    case mla
}

/// Channel mixer (FFN) operator seam across model architectures.
public enum QWChannelMixerKind: String, Sendable, Equatable, Codable {
    /// Dense SwiGLU MLP (`QWMLP`).
    case dense
    /// Routed + optional shared Sparse-MoE (`QWMoEOperator`).
    case moe
}

/// One compiled decoder layer resolved at model load time across the 4 operator seams.
public struct QWCompiledLayer: Sendable, Equatable {
    public let index: Int
    public let streamCarrier: QWStreamCarrierKind
    public let sequenceMixer: QWSequenceMixerKind
    public let channelMixer: QWChannelMixerKind
    public let hasPLE: Bool
    public let kernelKey: QWEnginePlan.LayerKernelKey

    public init(
        index: Int,
        streamCarrier: QWStreamCarrierKind,
        sequenceMixer: QWSequenceMixerKind,
        channelMixer: QWChannelMixerKind,
        hasPLE: Bool,
        kernelKey: QWEnginePlan.LayerKernelKey
    ) {
        self.index = index
        self.streamCarrier = streamCarrier
        self.sequenceMixer = sequenceMixer
        self.channelMixer = channelMixer
        self.hasPLE = hasPLE
        self.kernelKey = kernelKey
    }
}

/// Unified compiled layer stack schedule across single-chunk (no PLE) and multi-chunk (async PLE) models.
public struct QWCompiledLayerStack: Sendable, Equatable {
    public let layers: [QWCompiledLayer]
    public let cuts: [Int]
    public let steps: [[QWLayerStackSchedule.Step]]
    public let firstPLELayerIndex: Int?

    public init(plan: QWEnginePlan, descriptor: QWModelDescriptor? = nil) {
        let usesHC = plan.modelCapabilities.contains(.hyperConnections) || (descriptor?.hcCount ?? 1) > 1
        let mhcRange: Range<Int>? = descriptor?.mhcConfig.map { $0.start..<$0.end }
        let usesMLA = (descriptor?.kvLoraRank ?? 0) > 0

        var compiled: [QWCompiledLayer] = []
        compiled.reserveCapacity(plan.layers.count)
        for layer in plan.layers {
            let carrier: QWStreamCarrierKind
            if usesHC {
                carrier = .hyperConnection
            } else if let mhcRange, mhcRange.contains(layer.index) {
                carrier = .mhc
            } else {
                carrier = .singleStream
            }

            let seqMixer: QWSequenceMixerKind
            switch layer.attention {
            case .gdn:
                seqMixer = .gdn
            case .qsaSparse:
                seqMixer = .qsaSparse
            case .fullAttention:
                seqMixer = usesMLA ? .mla : .fullAttention
            }

            let chanMixer: QWChannelMixerKind = (layer.mlp == .moe) ? .moe : .dense
            compiled.append(
                QWCompiledLayer(
                    index: layer.index,
                    streamCarrier: carrier,
                    sequenceMixer: seqMixer,
                    channelMixer: chanMixer,
                    hasPLE: layer.hasPLE,
                    kernelKey: layer.kernelKey
                )
            )
        }
        self.layers = compiled
        let anyPLE = compiled.contains(where: \.hasPLE)
        self.firstPLELayerIndex = compiled.firstIndex(where: \.hasPLE)
        let resolvedCuts = QWLayerStackSchedule.decodeCuts(slotCount: compiled.count, hasPLE: anyPLE)
        self.cuts = resolvedCuts
        var chunkHasPLE: [Bool] = []
        for idx in 0..<max(0, resolvedCuts.count - 1) {
            let lo = resolvedCuts[idx], hi = resolvedCuts[idx + 1]
            let hasPLE = (lo..<hi).contains { compiled[$0].hasPLE }
            chunkHasPLE.append(hasPLE)
        }
        self.steps = QWLayerStackSchedule.stepOrder(chunkHasPLE: chunkHasPLE)
    }

    public func prefillChunkEnd(start: Int, layersPerCB: Int = 12, pleStaged: Bool = false) -> Int {
        QWLayerStackSchedule.prefillChunkEnd(
            start: start,
            count: layers.count,
            layersPerCB: layersPerCB,
            firstPLE: pleStaged ? nil : firstPLELayerIndex
        )
    }

    public static func supportsRollbackWindow(_ rows: Int) -> Bool {
        QWFourSlotRollbackRing.supportsWindow(rows)
    }
}
