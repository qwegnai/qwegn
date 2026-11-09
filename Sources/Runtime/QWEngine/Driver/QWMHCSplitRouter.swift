import Metal
import QWCore
import QWHAL

/// Split mHC router-norm used when rows <= maxRows (decode, MTP verify): wide partial dots
/// across ceil(hidden/64) threadgroups per row (plus stream Gram terms), then a finalize
/// sliced over ceil(hidden/256) threadgroups per row. QW_MHC_SPLIT=0 disables.
struct QWMHCSplitRouter {
    static let maxRows = 16
    let partial: any MTLComputePipelineState
    let finalize: any MTLComputePipelineState
    let partials: any MTLBuffer

    init?(hal: QWHal, geometry: QWModelGeometry) throws {
        guard let cfg = geometry.mhcConfig,
              ProcessInfo.processInfo.environment["QW_MHC_SPLIT"] != "0" else { return nil }
        partial = try hal.pipeline("qwg_mhc_router_partial_f32")
        finalize = try hal.pipeline("qwg_mhc_router_finalize_f16")
        let parts = (geometry.hidden + 63) / 64
        let outs = 2 * cfg.streams + cfg.streams * cfg.streams + 1 + cfg.streams * (cfg.streams + 1) / 2
        guard let buf = hal.device.makeBuffer(length: Self.maxRows * parts * outs * 4,
                                              options: .storageModePrivate) else { return nil }
        partials = buf
    }
}
