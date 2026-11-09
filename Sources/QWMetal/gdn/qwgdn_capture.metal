#include <metal_stdlib>
#include "include/qw_geometry_constants.metalh"
using namespace metal;

// Word copies preserve FP16/FP32 bits, including non-finite payloads.
kernel void qwg_gdn_capture_replay(
        const device uint* raw [[buffer(0)]],
        const device uint* normalized [[buffer(1)]],
        const device uint* beta [[buffer(2)]],
        const device uint* decay [[buffer(3)]],
        device uint* savedRaw [[buffer(4)]],
        device uint* savedNormalized [[buffer(5)]],
        device uint* savedBeta [[buffer(6)]],
        device uint* savedDecay [[buffer(7)]],
        uint index [[thread_position_in_grid]]) {
    const uint qkvWords = QW_GDN_QKV_CH / 2u;
    if (index < qkvWords) {
        savedRaw[index] = raw[index];
        savedNormalized[index] = normalized[index];
    }
    if (index < QW_GDN_V_HEADS) {
        savedBeta[index] = beta[index];
    }
    const uint glogWords = QW_GDN_VECTOR_DECAY ? QW_GDN_Z_CH : QW_GDN_V_HEADS;
    if (index < glogWords) {
        savedDecay[index] = decay[index];
    }
}
