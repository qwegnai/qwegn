// DriverInitPhases+Ernie55.swift — HCA Weight-SVD LSA sidecar loader and unified
// Metal residency set / resource-binding helpers for QWEngineDriver.

import Foundation
import Metal
import QWAlloc
import QWAttn
import QWCore
import QWHAL
import QWQuant

extension QWEngineDriver {

    static func loadHCALSASidecar(
        hal: QWHal,
        geometry: QWModelGeometry,
        lsaConfig: QWLSAConfig,
        packPath: String,
        pack: QWQuant.Pack?,
        safetensors: QWSafeTensorsContainer?,
        hcaLayers: [Int]
    ) -> (buffer: any MTLBuffer, indexDim: Int, byLayer: [Int: (wq: QWTensorValue, wk: QWTensorValue)])? {
        guard lsaConfig.isEnabled, let hca = geometry.hcaConfig, !hcaLayers.isEmpty else {
            return nil
        }
        let inDim = hca.qkNopeHeadDim
        guard inDim > 0 else { return nil }
        var indexDim = QWLSAConfig.defaultIndexDim
        let sidecarURL = QWLSAConfig.resolveSidecarURL(
            packPath: packPath, explicitSidecarPath: lsaConfig.sidecarPath)

        var rawPairs: [(physLayer: Int, wq: [UInt16], wk: [UInt16])] = []
        if lsaConfig.mode != .svd, let url = sidecarURL,
           let loaded = try? QWLSASidecar(url: url, descriptor: geometry.descriptor) {
            indexDim = loaded.header.provenance.indexDim
            let shape = [indexDim, inDim]
            for vLayer in hcaLayers {
                let phys = geometry.executionPlan[vLayer]
                let wq = (try? loaded.readF16Tensor("layers.\(phys).wq_idx", expectedShape: shape))
                    ?? QWLSASidecar.synthesizeDeterministicOrthogonalBasis(
                        inDim: inDim, indexDim: indexDim, seed: UInt64(phys &* 2 &+ 1))
                let wk = (try? loaded.readF16Tensor("layers.\(phys).wk_idx", expectedShape: shape)) ?? wq
                rawPairs.append((phys, wq, wk))
            }
        } else {
            let synth: [Int: (wq: [UInt16], wk: [UInt16])]?
            if let pk = pack {
                synth = try? QWLSASidecar.synthesizeFromPack(
                    pack: pk, descriptor: geometry.descriptor, indexDim: indexDim)
            } else if let st = safetensors {
                synth = try? QWLSASidecar.synthesizeFromSafeTensors(
                    container: st, descriptor: geometry.descriptor, indexDim: indexDim)
            } else {
                synth = nil
            }
            for vLayer in hcaLayers {
                let phys = geometry.executionPlan[vLayer]
                if let p = synth?[phys] {
                    rawPairs.append((phys, p.wq, p.wk))
                } else {
                    let b = QWLSASidecar.synthesizeDeterministicOrthogonalBasis(
                        inDim: inDim, indexDim: indexDim, seed: UInt64(phys &* 2 &+ 1))
                    rawPairs.append((phys, b, b))
                }
            }
        }

        let elemsPerMat = indexDim * inDim
        let strideBytes = ((elemsPerMat * 2) + 255) & ~255
        let totalBytes = max(4096, rawPairs.count * 2 * strideBytes)
        guard let buf = hal.device.makeBuffer(length: totalBytes, options: .storageModeShared) else {
            return nil
        }
        buf.label = "hca-lsa-sidecar"
        memset(buf.contents(), 0, totalBytes)
        let reg = QWMRegion(buffer: buf, base: 0, length: totalBytes)
        var out: [Int: (wq: QWTensorValue, wk: QWTensorValue)] = [:]
        var cursor = 0
        for item in rawPairs {
            let wqOff = cursor
            cursor += strideBytes
            let wkOff = cursor
            cursor += strideBytes
            item.wq.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                if let base = raw.baseAddress, !raw.isEmpty {
                    buf.contents().advanced(by: wqOff).copyMemory(from: base, byteCount: raw.count)
                }
            }
            item.wk.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                if let base = raw.baseAddress, !raw.isEmpty {
                    buf.contents().advanced(by: wkOff).copyMemory(from: base, byteCount: raw.count)
                }
            }
            let wqT = QWTensorValue(region: reg, offset: wqOff, shape: QWTensorShape([elemsPerMat], .f16))
            let wkT = QWTensorValue(region: reg, offset: wkOff, shape: QWTensorShape([elemsPerMat], .f16))
            out[item.physLayer] = (wqT, wkT)
        }
        return (buf, indexDim, out)
    }

    func allResidentBuffers() -> [any MTLBuffer] {
        var resident: [any MTLBuffer] = arena.allBuffers
        if let lsa = lsaOperator { resident.append(lsa.region.buffer) }
        if let mla = mlaOperator { resident.append(mla.region.buffer) }
        if let hca = hcaOperator { resident.append(hca.region.buffer) }
        if let hcaLsa = hcaLsaBuffer { resident.append(hcaLsa) }
        if let echo = echoDrafter { resident.append(echo.region.buffer) }
        if let ew = echoWeights { resident += ew.residentBuffers }
        switch headStorage {
        case .f16(let t): resident.append(t.region.buffer)
        case .q4g(let w): resident.append(w.view.blocks.region.buffer)
        }
        if let emb = mtpDeviceEmbedding { resident += [emb.buffer, emb.tokens] }
        if let dh = mtpDraftHead { resident += dh.mappedBuffers }
        for lw in layersW {
            resident += [lw.gu.view.blocks.region.buffer, lw.down.blocks.region.buffer]
            if let g = lw.gdn { resident += [g.inProj.blocks.region.buffer, g.out.blocks.region.buffer] }
            if let a = lw.attn { resident += [a.qkvProj.blocks.region.buffer, a.o.blocks.region.buffer] }
            if let hc = lw.hca {
                if let f = hc.wFusedIn { resident.append(f.blocks.region.buffer) }
                if let q = hc.wQUpGate { resident.append(q.blocks.region.buffer) }
                if let oa = hc.woA { resident.append(oa.blocks.region.buffer) }
                if let ob = hc.woB { resident.append(ob.blocks.region.buffer) }
            }
            if let m = lw.moe, let lat = m.latent {
                resident += [lat.fc1.blocks.region.buffer, lat.fc2.blocks.region.buffer]
            }
        }
        if let mw = mtpWeights {
            resident += [mw.fc.blocks.region.buffer, mw.qkvProj.blocks.region.buffer, mw.o.blocks.region.buffer,
                         mw.gu.view.blocks.region.buffer, mw.down.blocks.region.buffer]
        }
        return resident
    }

    func bindResidentResources(_ enc: any MTLComputeCommandEncoder) {
        for buf in arena.allBuffers { enc.useResource(buf, usage: [.read, .write]) }
        if let lsa = lsaOperator { enc.useResource(lsa.region.buffer, usage: [.read, .write]) }
        if let mla = mlaOperator { enc.useResource(mla.region.buffer, usage: [.read, .write]) }
        if let hca = hcaOperator { enc.useResource(hca.region.buffer, usage: [.read, .write]) }
        if let hcaLsa = hcaLsaBuffer { enc.useResource(hcaLsa, usage: .read) }
        if let echo = echoDrafter { enc.useResource(echo.region.buffer, usage: [.read, .write]) }
        if let ew = echoWeights {
            for b in ew.residentBuffers { enc.useResource(b, usage: .read) }
        }
        switch headStorage {
        case .f16(let t): enc.useResource(t.region.buffer, usage: .read)
        case .q4g(let w): enc.useResource(w.view.blocks.region.buffer, usage: .read)
        }
        if let emb = mtpDeviceEmbedding {
            enc.useResource(emb.buffer, usage: .read)
            enc.useResource(emb.tokens, usage: [.read, .write])
        }
        if let dh = mtpDraftHead {
            for buf in dh.mappedBuffers { enc.useResource(buf, usage: .read) }
        }
        for lw in layersW {
            enc.useResource(lw.gu.view.blocks.region.buffer, usage: .read)
            enc.useResource(lw.down.blocks.region.buffer, usage: .read)
            if let g = lw.gdn {
                enc.useResource(g.inProj.blocks.region.buffer, usage: .read)
                enc.useResource(g.out.blocks.region.buffer, usage: .read)
            }
            if let a = lw.attn {
                enc.useResource(a.qkvProj.blocks.region.buffer, usage: .read)
                enc.useResource(a.o.blocks.region.buffer, usage: .read)
            }
            if let hc = lw.hca {
                if let f = hc.wFusedIn { enc.useResource(f.blocks.region.buffer, usage: .read) }
                if let q = hc.wQUpGate { enc.useResource(q.blocks.region.buffer, usage: .read) }
                if let oa = hc.woA { enc.useResource(oa.blocks.region.buffer, usage: .read) }
                if let ob = hc.woB { enc.useResource(ob.blocks.region.buffer, usage: .read) }
            }
            if let m = lw.moe, let lat = m.latent {
                enc.useResource(lat.fc1.blocks.region.buffer, usage: .read)
                enc.useResource(lat.fc2.blocks.region.buffer, usage: .read)
            }
        }
        if let mw = mtpWeights {
            for b in [mw.fc.blocks.region.buffer, mw.qkvProj.blocks.region.buffer, mw.o.blocks.region.buffer,
                      mw.gu.view.blocks.region.buffer, mw.down.blocks.region.buffer] {
                enc.useResource(b, usage: .read)
            }
        }
    }
}
