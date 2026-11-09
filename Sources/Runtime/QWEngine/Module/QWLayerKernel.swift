// QWLayerKernel.swift — One layer kind's kernels.
//
// The plan resolves the dispatch. A layer's kind is read once, when the
// model is loaded, and the table it names is built once — the hot loop looks
// a kernel up instead of switching on the kind.

import Foundation
import Metal
import QWCore

/// One layer kind's kernels.
public protocol QWLayerKernel: AnyObject, Sendable {
    /// The plan key this kernel serves.
    static var key: QWEnginePlan.LayerKernelKey { get }
    /// A kernel of this kind.
    static func make() -> any QWLayerKernel
    /// Encodes one layer of one decode step.
    func encodeDecode(_ context: QWLayerDecodeContext) throws
    /// Encodes one layer of one prefill chunk.
    func encodePrefillChunk(_ context: QWLayerPrefillContext) throws
}

/// The kernels a plan can name.
public enum QWLayerKernelRegistry {
    /// The kernel that serves a plan key, or nil when the plan names a kind
    /// this engine does not implement. A plan that names a kind nobody
    /// implements is a load-time error, never a silent fallback.
    public static func kernel(for key: QWEnginePlan.LayerKernelKey) -> (any QWLayerKernel.Type)? {
        switch key {
        case QWEnginePlan.LayerKernelKey(attention: .gdn, mlp: .dense, ple: false):
            return GDNLayerKernel.self
        case QWEnginePlan.LayerKernelKey(attention: .gdn, mlp: .moe, ple: false):
            return GDNMoELayerKernel.self
        case QWEnginePlan.LayerKernelKey(attention: .fullAttention, mlp: .dense, ple: false),
             QWEnginePlan.LayerKernelKey(attention: .qsaSparse, mlp: .dense, ple: false):
            return FullAttentionLayerKernel.self
        case QWEnginePlan.LayerKernelKey(attention: .fullAttention, mlp: .moe, ple: false),
             QWEnginePlan.LayerKernelKey(attention: .qsaSparse, mlp: .moe, ple: false):
            return MoELayerKernel.self
        default:
            return nil
        }
    }

    /// The kernel each layer of a plan runs.
    /// The dispatch is resolved here, once, when the model is loaded. A plan
    /// that names a kind this engine does not implement is a load-time error.
    public static func resolve(plan: QWEnginePlan) throws -> [any QWLayerKernel] {
        try plan.layers.map { layer in
            let key = layer.kernelKey
            guard let kind = kernel(for: key) else {
                throw QWModuleError.notImplemented(
                    "no layer kernel for \(key.attention.rawValue)/\(key.mlp.rawValue)")
            }
            return kind.make()
        }
    }

    /// The kernel each layer of a loaded model runs.
    /// The plan and the weights must agree about the kind: one says linear
    /// attention, the other says full attention, and the load fails loud.
    static func resolve(layers: [QWEngineDriver.LayerW],
                        plan: QWEnginePlan) throws -> [any QWLayerKernel] {
        guard plan.layers.count == layers.count else {
            throw QWEngineDriverError.state(
                "the plan names \(plan.layers.count) layers, the weights carry \(layers.count)")
        }
        let kernels = try resolve(plan: plan)
        let fullAttnKey = plan.modelCapabilities.contains(.qsaIndexer) ? "qsaSparse" : "fullAttention"
        for (index, weights) in layers.enumerated() {
            let key = plan.layers[index].kernelKey
            // The plan and the weights must agree about the kind — the
            // attention mixer AND the MLP family: a plan that says `.moe`
            // against dense weights is a different model.
            let runtime: (attention: String, mlp: String) = switch weights.kind {
            case .gdn: ("gdn", "dense")
            case .gdnMoe: ("gdn", "moe")
            case .fullAttention: (fullAttnKey, "dense")
            case .moe: (fullAttnKey, "moe")
            }
            guard runtime.attention == key.attention.rawValue,
                  runtime.mlp == key.mlp.rawValue else {
                throw QWEngineDriverError.state(
                    "layer \(index) is \(runtime.attention)/\(runtime.mlp) in the weights and "
                    + "\(key.attention.rawValue)/\(key.mlp.rawValue) in the plan")
            }
        }
        return kernels
    }
}

/// What a layer kernel needs to encode one decode step.
/// A value, not a reference: the step allocates nothing, so a kernel call
/// may not put one of these on the heap.
public struct QWLayerDecodeContext {
    /// The engine the layer belongs to.
    let driver: QWEngineDriver
    private let _enc: any MTLComputeCommandEncoder
    /// The encoder the step writes into (tracks `driver.activeEnc` when cold experts rotate the encoder).
    var enc: any MTLComputeCommandEncoder { driver.activeEnc ?? _enc }
    /// The session the step belongs to.
    let session: QWEngineSession
    /// The absolute position the step encodes.
    let position: Int
    /// The layer being encoded.
    let layer: Int
    /// The layer's weights.
    let weights: QWEngineDriver.LayerW

    init(driver: QWEngineDriver, enc: any MTLComputeCommandEncoder,
         session: QWEngineSession, position: Int, layer: Int, weights: QWEngineDriver.LayerW) {
        self.driver = driver
        self._enc = enc
        self.session = session
        self.position = position
        self.layer = layer
        self.weights = weights
    }
}

/// What a layer kernel needs to encode one prefill chunk.
/// A value, for the same reason as the decode context.
public struct QWLayerPrefillContext {
    /// The engine the layer belongs to.
    let driver: QWEngineDriver
    private let _enc: any MTLComputeCommandEncoder
    /// The encoder the chunk writes into (tracks `driver.activeEnc` when cold experts rotate the encoder).
    var enc: any MTLComputeCommandEncoder { driver.activeEnc ?? _enc }
    /// The session the chunk belongs to.
    let session: QWEngineSession
    /// The absolute position the chunk starts at.
    let position: Int
    /// The layer being encoded.
    let layer: Int
    /// The layer's weights.
    let weights: QWEngineDriver.LayerW
    /// The rows in this chunk.
    let rows: Int
    /// Which prefill path the recurrent layers take.
    let mode: GDNPrefillMode

    init(driver: QWEngineDriver, enc: any MTLComputeCommandEncoder,
         session: QWEngineSession, position: Int, layer: Int, weights: QWEngineDriver.LayerW,
         rows: Int, mode: GDNPrefillMode) {
        self.driver = driver
        self._enc = enc
        self.session = session
        self.position = position
        self.layer = layer
        self.weights = weights
        self.rows = rows
        self.mode = mode
    }
}
