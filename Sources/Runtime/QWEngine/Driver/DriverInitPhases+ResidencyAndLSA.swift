// DriverInitPhases+ResidencyAndLSA.swift — HCA Weight-SVD LSA sidecar loader and unified
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
        let bufPtr = buf.contents()
        memset(bufPtr, 0, totalBytes)
        let reg = QWMRegion(buffer: buf, base: 0, length: totalBytes)
        var out: [Int: (wq: QWTensorValue, wk: QWTensorValue)] = [:]
        var cursor = 0
        for item in rawPairs {
            let wqOff = cursor
            cursor += strideBytes
            let wkOff = cursor
            cursor += strideBytes
            let wqDst = bufPtr.advanced(by: wqOff)
            item.wq.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                if let base = raw.baseAddress, !raw.isEmpty {
                    wqDst.copyMemory(from: base, byteCount: raw.count)
                }
            }
            let wkDst = bufPtr.advanced(by: wkOff)
            item.wk.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                if let base = raw.baseAddress, !raw.isEmpty {
                    wkDst.copyMemory(from: base, byteCount: raw.count)
                }
            }
            let wqT = QWTensorValue(region: reg, offset: wqOff, shape: QWTensorShape([elemsPerMat], .f16))
            let wkT = QWTensorValue(region: reg, offset: wkOff, shape: QWTensorShape([elemsPerMat], .f16))
            out[item.physLayer] = (wqT, wkT)
        }
        return (buf, indexDim, out)
    }

    func allResidentBuffers() -> [any MTLBuffer] {
        var resident: [any MTLBuffer] = []
        var seen: Set<ObjectIdentifier> = []
        func appendUnique(_ buf: any MTLBuffer) {
            if seen.insert(ObjectIdentifier(buf)).inserted {
                resident.append(buf)
            }
        }
        appendUnique(hal.gatedNormScratch)
        for buf in arena.allBuffers { appendUnique(buf) }
        if let lsa = lsaOperator { appendUnique(lsa.region.buffer) }
        if let mla = mlaOperator { appendUnique(mla.region.buffer) }
        if let mbsa = mbsaOperator { appendUnique(mbsa.region.buffer) }
        if let hca = hcaOperator { appendUnique(hca.region.buffer) }
        if let hcaLsa = hcaLsaBuffer { appendUnique(hcaLsa) }
        if let mhc = mhcSplit { appendUnique(mhc.partials) }
        if let ce = coldEvents {
            appendUnique(ce.promoList)
            appendUnique(ce.bridgeTick)
            appendUnique(ce.residentIds)
        }
        if let sm = spanMapIds { appendUnique(sm) }
        if geometry.hcaConfig == nil, geometry.blockSparseGQAConfig == nil,
           let bmlp = try? sharedBatchMLP() {
            appendUnique(bmlp.sums.region.buffer)
            appendUnique(bmlp.downRows.region.buffer)
        }
        if let df = dflashDrafter {
            for b in df.residentBuffers { appendUnique(b) }
            appendUnique(df.targetHidden.region.buffer)
            for slot in df.sessionSlots {
                appendUnique(slot.kSlab.region.buffer)
                appendUnique(slot.vSlab.region.buffer)
            }
        }
        if let ds = dsparkDrafter {
            for b in ds.activeBuffers { appendUnique(b) }
        }
        if let echo = echoDrafter { appendUnique(echo.region.buffer) }
        if let ew = echoWeights {
            for b in ew.residentBuffers { appendUnique(b) }
        }
        switch headStorage {
        case .f16(let t), .bf16(let t): appendUnique(t.region.buffer)
        case .q4g(let w): appendUnique(w.view.blocks.region.buffer)
        }
        if let emb = mtpDeviceEmbedding {
            appendUnique(emb.buffer)
            appendUnique(emb.tokens)
        }
        if let dh = mtpDraftHead {
            for buf in dh.mappedBuffers { appendUnique(buf) }
        }
        for lw in layersW {
            appendUnique(lw.gu.view.blocks.region.buffer)
            appendUnique(lw.down.blocks.region.buffer)
            if let g = lw.gdn {
                appendUnique(g.inProj.blocks.region.buffer)
                appendUnique(g.out.blocks.region.buffer)
            }
            if let a = lw.attn {
                appendUnique(a.qkvProj.blocks.region.buffer)
                appendUnique(a.o.blocks.region.buffer)
                if let qb = a.qkvBias { appendUnique(qb.region.buffer) }
                if let ob = a.oBias { appendUnique(ob.region.buffer) }
                if let q = a.mbsa?.qkvIdxProj { appendUnique(q.blocks.region.buffer) }
                if let fg = a.mla?.flashGateW { appendUnique(fg.region.buffer) }
                if let fb = a.mla?.flashBias { appendUnique(fb.region.buffer) }
            }
            if let hc = lw.hca {
                if let f = hc.wFusedIn { appendUnique(f.blocks.region.buffer) }
                if let q = hc.wQUpGate { appendUnique(q.blocks.region.buffer) }
                if let oa = hc.woA { appendUnique(oa.blocks.region.buffer) }
                if let ob = hc.woB { appendUnique(ob.blocks.region.buffer) }
            }
            if let m = lw.moe {
                if let lat = m.latent {
                    appendUnique(lat.fc1.blocks.region.buffer)
                    appendUnique(lat.fc2.blocks.region.buffer)
                }
                if let gm = m.coldStreamer?.gpuSlotMap { appendUnique(gm) }
            }
        }
        if let mw = mtpWeights {
            for b in [mw.fc.blocks.region.buffer, mw.qkvProj.blocks.region.buffer, mw.o.blocks.region.buffer,
                      mw.gu.view.blocks.region.buffer, mw.down.blocks.region.buffer] {
                appendUnique(b)
            }
            if let qb = mw.qkvBias { appendUnique(qb.region.buffer) }
            if let ob = mw.oBias { appendUnique(ob.region.buffer) }
        }
        return resident
    }

    func bindResidentResources(_ enc: any MTLComputeCommandEncoder) {
        var seen: Set<ObjectIdentifier> = []
        func useUnique(_ buf: any MTLBuffer, usage: MTLResourceUsage) {
            if seen.insert(ObjectIdentifier(buf)).inserted {
                enc.useResource(buf, usage: usage)
            }
        }
        useUnique(hal.gatedNormScratch, usage: [.read, .write])
        for buf in arena.allBuffers { useUnique(buf, usage: [.read, .write]) }
        if let lsa = lsaOperator { useUnique(lsa.region.buffer, usage: [.read, .write]) }
        if let mla = mlaOperator { useUnique(mla.region.buffer, usage: [.read, .write]) }
        if let mbsa = mbsaOperator { useUnique(mbsa.region.buffer, usage: [.read, .write]) }
        if let hca = hcaOperator { useUnique(hca.region.buffer, usage: [.read, .write]) }
        if let hcaLsa = hcaLsaBuffer { useUnique(hcaLsa, usage: .read) }
        if let mhc = mhcSplit { useUnique(mhc.partials, usage: [.read, .write]) }
        if let ce = coldEvents {
            useUnique(ce.promoList, usage: [.read, .write])
            useUnique(ce.bridgeTick, usage: [.read, .write])
            useUnique(ce.residentIds, usage: [.read, .write])
        }
        if let sm = spanMapIds { useUnique(sm, usage: [.read, .write]) }
        if geometry.hcaConfig == nil, geometry.blockSparseGQAConfig == nil,
           let bmlp = try? sharedBatchMLP() {
            useUnique(bmlp.sums.region.buffer, usage: [.read, .write])
            useUnique(bmlp.downRows.region.buffer, usage: [.read, .write])
        }
        if let df = dflashDrafter {
            for b in df.residentBuffers { useUnique(b, usage: [.read, .write]) }
            useUnique(df.targetHidden.region.buffer, usage: [.read, .write])
            for slot in df.sessionSlots {
                useUnique(slot.kSlab.region.buffer, usage: [.read, .write])
                useUnique(slot.vSlab.region.buffer, usage: [.read, .write])
            }
        }
        if let ds = dsparkDrafter {
            for b in ds.activeBuffers { useUnique(b, usage: [.read, .write]) }
        }
        if let echo = echoDrafter { useUnique(echo.region.buffer, usage: [.read, .write]) }
        if let ew = echoWeights {
            for b in ew.residentBuffers { useUnique(b, usage: .read) }
        }
        switch headStorage {
        case .f16(let t), .bf16(let t): useUnique(t.region.buffer, usage: .read)
        case .q4g(let w): useUnique(w.view.blocks.region.buffer, usage: .read)
        }
        if let emb = mtpDeviceEmbedding {
            useUnique(emb.buffer, usage: .read)
            useUnique(emb.tokens, usage: [.read, .write])
        }
        if let dh = mtpDraftHead {
            for buf in dh.mappedBuffers { useUnique(buf, usage: .read) }
        }
        for lw in layersW {
            useUnique(lw.gu.view.blocks.region.buffer, usage: .read)
            useUnique(lw.down.blocks.region.buffer, usage: .read)
            if let g = lw.gdn {
                useUnique(g.inProj.blocks.region.buffer, usage: .read)
                useUnique(g.out.blocks.region.buffer, usage: .read)
            }
            if let a = lw.attn {
                useUnique(a.qkvProj.blocks.region.buffer, usage: .read)
                useUnique(a.o.blocks.region.buffer, usage: .read)
                if let qb = a.qkvBias { useUnique(qb.region.buffer, usage: .read) }
                if let ob = a.oBias { useUnique(ob.region.buffer, usage: .read) }
                if let q = a.mbsa?.qkvIdxProj { useUnique(q.blocks.region.buffer, usage: .read) }
                if let fg = a.mla?.flashGateW { useUnique(fg.region.buffer, usage: .read) }
                if let fb = a.mla?.flashBias { useUnique(fb.region.buffer, usage: .read) }
            }
            if let hc = lw.hca {
                if let f = hc.wFusedIn { useUnique(f.blocks.region.buffer, usage: .read) }
                if let q = hc.wQUpGate { useUnique(q.blocks.region.buffer, usage: .read) }
                if let oa = hc.woA { useUnique(oa.blocks.region.buffer, usage: .read) }
                if let ob = hc.woB { useUnique(ob.blocks.region.buffer, usage: .read) }
            }
            if let m = lw.moe {
                if let lat = m.latent {
                    useUnique(lat.fc1.blocks.region.buffer, usage: .read)
                    useUnique(lat.fc2.blocks.region.buffer, usage: .read)
                }
                if let gm = m.coldStreamer?.gpuSlotMap { useUnique(gm, usage: [.read, .write]) }
            }
        }
        if let mw = mtpWeights {
            for b in [mw.fc.blocks.region.buffer, mw.qkvProj.blocks.region.buffer, mw.o.blocks.region.buffer,
                      mw.gu.view.blocks.region.buffer, mw.down.blocks.region.buffer] {
                useUnique(b, usage: .read)
            }
            if let qb = mw.qkvBias { useUnique(qb.region.buffer, usage: .read) }
            if let ob = mw.oBias { useUnique(ob.region.buffer, usage: .read) }
        }
    }
}
