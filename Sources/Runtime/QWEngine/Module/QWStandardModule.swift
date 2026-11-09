// QWStandardModule.swift — The standard families' module.
//
// The dense engine's request lifecycle with the standard model's contract:
// no attention output gate, no GDN, no PLE, no MTP, no vision, a dense
// SwiGLU MLP. The FWHT32 is the packer's weight domain (C84), so this
// module adds no numerics. One module type per family. The qwen3 family
// adds a q/k RMSNorm — ONE vector shared by every head (C142).

import Foundation
import QWCore

/// The shared body of the standard families' module.
open class StandardModule: DenseModule, @unchecked Sendable {
    /// The body names no family: the seam names one per module type, so
    /// each standard family has its own type over this one. Reading it here
    /// is a misuse, and says so.
    open override class var family: QWModelDescriptor.Family {
        fatalError("StandardModule is a body: each standard family has its own module type")
    }
    /// The standard families support Q8 and Q4 KV cache quantization and LSA; they ship none of
    /// the 27B's hybrid-only extras (no DFlash, no MTP, no vision, no GDN prefill).
    open override class var capabilities: Set<QWKnobCapability> { [.kvQuantQ8, .kvQuantQ4, .lsa] }
    open override class var supportedKVStorageModes: Set<String> { ["f16", "q8", "q4"] }

    /// The doors shared by the standard dense model families.
    open override class var requiredDoors: [String] { QWKernelInventory.standard }

    /// The standard path's serving law: a head's width the rotation covers,
    /// and f16 KV off the 27B's shape. The driver's predicate states it (C110).
    static func checkServed(_ plan: QWEnginePlan) throws {
        try QWEngineDriver.validateServing(headDim: plan.headDim, kvHeads: plan.kvHeads,
                                            qHeads: plan.qHeads, kvMode: plan.kvStorageMode)
    }

    open override class func make(plan: QWEnginePlan,
                                  engine: (any QWLoadedEngine)?) throws -> any QWInferenceModule {
        try checkServed(plan)
        return try super.make(plan: plan, engine: engine)
    }
}

/// The llama family.
public final class LlamaModule: StandardModule, @unchecked Sendable {
    public override class var family: QWModelDescriptor.Family { .llamaDense }
}

/// The mistral family.
public final class MistralModule: StandardModule, @unchecked Sendable {
    public override class var family: QWModelDescriptor.Family { .mistralDense }
}

/// The qwen2 family.
public final class Qwen2DenseModule: StandardModule, @unchecked Sendable {
    public override class var family: QWModelDescriptor.Family { .qwen2Dense }
}

/// The qwen3 family. Its q and k carry a shared RMSNorm — ONE vector for
/// every head, not one per head (C142) — so the rope kernel reads it
/// un-offset.
public final class Qwen3DenseModule: StandardModule, @unchecked Sendable {
    public override class var family: QWModelDescriptor.Family { .qwen3Dense }
}

/// The qwen3 sparse-MoE family. Its layer is the dense attention followed by
/// the routed experts, so its doors are the standard ones plus the
/// per-pair chain's six.
public final class Qwen3MoEModule: StandardModule, @unchecked Sendable {
    public override class var family: QWModelDescriptor.Family { .qwen3MoE }

    /// The sparse-MoE doors the standard path's routed layer dispatches: the
    /// per-pair chain and the router's own GEMM, which only a routed layer
    /// runs.
    public static let moeDoors = QWKernelInventory.moe_only

    public override class var requiredDoors: [String] {
        QWKernelInventory.moe
    }
}
