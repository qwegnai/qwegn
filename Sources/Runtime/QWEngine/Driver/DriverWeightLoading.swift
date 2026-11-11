// DriverWeightLoading.swift — pack-to-GPU weight loading helpers used by
// QWEngineDriver.init (split from QWEngineDriver.swift; same allocation
// order as the original init-local functions).
import Foundation
import Darwin
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWNorm
import QWRope
import QWKV
import QWAttn
import QWGDN
import QWLoad
import QWMLP
import QWHead
import QWMTP
import QWQuant
import QWState

extension QWEngineDriver {
    struct WeightLoader {
        let pk: QWQuant.Pack?
        let st: QWSafeTensorsContainer?
        let hl: QWHal
        let psoCvt: any MTLComputePipelineState
        let psoQuantBF16: (any MTLComputePipelineState)?
        let quantStaging: (any MTLBuffer)?
        let wsub: QWSubAllocator
        let plan: QWWeightAllocationPlan
        let geometry: QWModelGeometry
        let format: QWWeightFormat
        var weightRequest = 0
        var sharedColdBanks: (gu: QWWeightView, down: QWWeightView)?
        // Shared mover (QWLoad): F_NOCACHE pread straight into the destination,
        // single-memcpy mmap fallback; loading does not duplicate the pack in RAM.
        let io: PlaneIO?

        init(pk: QWQuant.Pack, hl: QWHal, psoCvt: any MTLComputePipelineState,
             wsub: QWSubAllocator, plan: QWWeightAllocationPlan, geometry: QWModelGeometry,
             format: QWWeightFormat = .hq4G32) {
            self.pk = pk; self.st = nil; self.hl = hl; self.psoCvt = psoCvt
            self.psoQuantBF16 = nil; self.quantStaging = nil
            self.wsub = wsub; self.plan = plan; self.geometry = geometry
            self.format = format
            self.io = PlaneIO(pack: pk, policy: LoadPolicy(descriptor: pk.descriptor))
        }

        init(st: QWSafeTensorsContainer, hl: QWHal, psoCvt: any MTLComputePipelineState,
             psoQuantBF16: any MTLComputePipelineState,
             wsub: QWSubAllocator, plan: QWWeightAllocationPlan, geometry: QWModelGeometry,
             format: QWWeightFormat = .hq4G32) {
            self.pk = nil; self.st = st; self.hl = hl; self.psoCvt = psoCvt
            self.psoQuantBF16 = psoQuantBF16
            self.quantStaging = format.isNativeAffine
                ? nil
                : hl.device.makeBuffer(length: 64 * 1024 * 1024, options: .storageModeShared)
            self.wsub = wsub; self.plan = plan; self.geometry = geometry
            self.format = format
            self.io = nil
        }

        static var prefixAlternates: [String] { QWCheckpointTensorLoader.prefixAlternates }

        private func rawExists(_ candidate: String) -> Bool {
            if let pk { return (try? pk.entry(candidate)) != nil }
            if let st { return st.tensor(candidate) != nil }
            return false
        }

        /// The pack's name for `name`, under whichever prefix the checkpoint
        /// used. The canonical name is returned when none of them is present,
        /// so the caller's error names what the engine asked for.
        func resolveName(_ name: String) -> String {
            QWCheckpointTensorLoader.resolveName(
                name,
                defaultPrefix: QWEngineDriver.P,
                tieWordEmbeddings: geometry.descriptor.tieWordEmbeddings,
                rawExists: rawExists
            )
        }

        /// True when the pack carries `name` (under any accepted prefix).
        func hasTensor(_ name: String) -> Bool {
            rawExists(resolveName(name))
        }

        var isMLXCheckpoint: Bool {
            st?.isMLXCheckpoint == true
        }

        mutating func allocateWeight(_ shape: QWTensorShape) throws -> QWTensorValue {
            try plan.check(shape, at: weightRequest)
            let tensor = try wsub.allocate(shape)
            weightRequest += 1
            return tensor
        }
        // ── W2: embed + lm_head bf16 → f16 (device gather, chunked upload) ──
        mutating func convertBF16(_ name: String) throws -> QWTensorValue {
            let name = resolveName(name)
            guard let pk, let io else {
                throw QWEngineDriverError.weightMissing(name, "convertBF16 requires a QWP pack")
            }
            let e = try pk.entry(name)
            guard e.rawOff != nil else {
                throw QWEngineDriverError.weightMissing(name, "not a bf16 raw plane")
            }
            let total = e.shape.reduce(1, *)
            let dst = try allocateWeight(QWTensorShape([total], .f16))
            // THE GRID LAW: threadgroups-per-dimension ≤ 65535 (Metal cap;
            // an over-cap dispatch silently no-ops — half a matrix would
            // upload as zeros). 65535·256 elems = 32 MB bf16 per chunk.
            let chunk = 65535 * 256
            let tmp = hl.device.makeBuffer(length: chunk * 2, options: .storageModeShared)!
            var done = 0
            while done < total {
                let n = min(chunk, total - done)
                try io.loadRaw(name, elemOffset: done, elems: n, into: tmp.contents())
                guard let (cb, enc) = hl.begin() else { throw QWEngineDriverError.state("cmdbuffer") }
                var start = UInt32(0), nn = UInt32(n)
                enc.setComputePipelineState(psoCvt)
                enc.setBuffer(tmp, offset: 0, index: 0)          // bf16 bytes (ushort)
                enc.setBuffer(dst.region.buffer,
                              offset: dst.region.base + dst.offset + done * 2, index: 1)
                enc.setBytes(&start, length: 4, index: 2)
                enc.setBytes(&nn, length: 4, index: 3)
                enc.dispatchThreadgroups(MTLSize(width: (n + 255) / 256, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                hl.end(cb, enc); cb.waitUntilCompleted()
                // H3 FIX (load path): a failed conversion chunk would
                // silently upload zeros into embed/head — throw.
                if let e = cb.error { throw QWEngineDriverError.state("bf16→f16 cb: \(e.localizedDescription)") }
                done += n
            }
            return dst
        }

        func quantizeSafetensorsToHQ4(_ name: String, n: Int, k: Int,
                                      into dstBuffer: any MTLBuffer,
                                      dstByteOffset: Int) throws {
            let resolved = resolveName(name)
            guard let st, let staging = quantStaging else {
                throw QWEngineDriverError.weightMissing(name, "safetensors online quantizer not configured")
            }
            do {
                try QWHQ4Quantizer.quantizeSafeTensorsInto(
                    hal: hl,
                    st: st,
                    name: resolved,
                    srcRowOffset: 0,
                    rowCount: n,
                    k: k,
                    dstBuffer: dstBuffer,
                    dstByteOffset: dstByteOffset,
                    staging: staging,
                    quantPSO: psoQuantBF16
                )
            } catch let e as QWHQ4Quantizer.QuantizeError {
                throw QWEngineDriverError.weightMissing(name, e.description)
            }
        }

        private func quantizeSafetensorsSwitchGateUp(
            gateName: String, upName: String, experts: Int, i: Int, h: Int,
            into dstBuffer: any MTLBuffer, dstByteOffset: Int
        ) throws {
            let gResolved = resolveName(gateName)
            let uResolved = resolveName(upName)
            guard let st, let staging = quantStaging else {
                throw QWEngineDriverError.weightMissing(gateName, "safetensors online quantizer not configured")
            }
            do {
                try QWHQ4Quantizer.quantizeSwitchGateUpInto(
                    hal: hl,
                    st: st,
                    gateName: gResolved,
                    upName: uResolved,
                    experts: experts,
                    inter: i,
                    hidden: h,
                    dstBuffer: dstBuffer,
                    dstByteOffset: dstByteOffset,
                    staging: staging,
                    quantPSO: psoQuantBF16
                )
            } catch let e as QWHQ4Quantizer.QuantizeError {
                throw QWEngineDriverError.weightMissing(gateName, e.description)
            }
        }

        // ── helpers: planes verbatim (W1); bf16→f16 / →f32 CPU (W4) ────────
        // The plane views themselves live in DriverWeightLoadingAQ.swift: the
        // family's container is sized and filled differently.

        func rawFloats(_ name: String, _ elems: Int) throws -> [Float] {
            let resolved = resolveName(name)
            if let pk {
                let src = try pk.rawBf16(resolved)
                guard src.count == elems else {
                    throw QWEngineDriverError.weightMissing(name, "shape \(src.count) ≠ \(elems)")
                }
                return src
            }
            guard let st else {
                throw QWEngineDriverError.weightMissing(name, "missing in safetensors")
            }
            do {
                let out = try st.readFloats(resolved, expectedElementCount: elems)
                guard out.count == elems else {
                    throw QWEngineDriverError.weightMissing(name, "shape \(out.count) ≠ \(elems)")
                }
                return out
            } catch let err as QWEngineDriverError {
                throw err
            } catch {
                throw QWEngineDriverError.weightMissing(name, "\(error)")
            }
        }

        /// A bf16 plane, verbatim. Private on purpose: a norm gamma must
        /// not come through here, so the engine cannot load one the wrong
        /// way. The norm law is `f16wNorm`, and the raw plane is for the
        /// weights that have no law (the GDN's conv and gated norm).
        mutating func f16w(_ name: String, _ elems: Int) throws -> QWTensorValue {
            let src = try rawFloats(name, elems)
            let t = try allocateWeight(QWTensorShape([elems], .f16))
            var bits = [UInt16](repeating: 0, count: elems)
            for i in 0..<elems { bits[i] = QWQuant.f32tof16Bits(src[i]) }
            bits.withUnsafeBytes { t.region.buffer.contents()
                .advanced(by: t.region.base + t.offset)
                .copyMemory(from: $0.baseAddress!, byteCount: elems * 2) }
            return t
        }
        /// A norm gamma's bits, in the law the engine's `1 + w` norms read:
        /// the 27B's checkpoint stores `w = γ − 1` under that law, a standard
        /// HF checkpoint stores γ itself, so it becomes `γ − 1` here.
        func normGammaBits(_ name: String, _ elems: Int) throws -> [UInt16] {
            let src = try rawFloats(name, elems)
            let shift: Float = (geometry.normGammaIsPlusOne && !isMLXCheckpoint) ? 0.0 : 1.0
            return src.map { QWQuant.f32tof16Bits($0 - shift) }
        }

        /// A norm gamma, stored the way the engine's norm law reads it.
        mutating func f16wNorm(_ name: String, _ elems: Int) throws -> QWTensorValue {
            guard !geometry.normGammaIsPlusOne || isMLXCheckpoint else { return try f16w(name, elems) }
            let bits = try normGammaBits(name, elems)
            let t = try allocateWeight(QWTensorShape([elems], .f16))
            bits.withUnsafeBytes { t.region.buffer.contents()
                .advanced(by: t.region.base + t.offset)
                .copyMemory(from: $0.baseAddress!, byteCount: elems * 2) }
            return t
        }

        /// The model's final RMSNorm gamma. It is a `1 + w` norm gamma like
        /// every other, so it is converted like every other.
        mutating func finalNorm(_ elems: Int) throws -> QWTensorValue {
            try f16wNorm(QWEngineDriver.P + "norm.weight", elems)
        }
        mutating func f32w(_ name: String, _ elems: Int) throws -> QWTensorValue {
            let src = try rawFloats(name, elems)
            let t = try allocateWeight(QWTensorShape([elems], .f32))
            src.withUnsafeBytes { t.region.buffer.contents()
                .advanced(by: t.region.base + t.offset)
                .copyMemory(from: $0.baseAddress!, byteCount: elems * 4) }
            return t
        }
        mutating func f32wOrDequant(_ name: String, _ elems: Int) throws -> QWTensorValue {
            let resolved = resolveName(name)
            let src: [Float]
            if let pk {
                let e = try pk.entry(resolved)
                if e.rawOff != nil {
                    src = try pk.rawBf16(resolved)
                } else {
                    src = try pk.dequantQ4g(resolved)
                }
                guard src.count == elems else {
                    throw QWEngineDriverError.weightMissing(name, "shape \(src.count) ≠ \(elems)")
                }
            } else {
                src = try rawFloats(name, elems)
            }
            let t = try allocateWeight(QWTensorShape([elems], .f32))
            src.withUnsafeBytes { t.region.buffer.contents()
                .advanced(by: t.region.base + t.offset)
                .copyMemory(from: $0.baseAddress!, byteCount: elems * 4) }
            return t
        }
        // ── W3: plane concatenation helper (pure plumbing, zero math change) ──
        private func resolvedPlaneLayout(_ firstName: String?, k: Int) -> (bits: Int, group: Int, incumbent: Bool, shvq: Bool, rowBytes: Int) {
            if let firstName, let pk, let entry = try? pk.entry(resolveName(firstName)), entry.hq4Off != nil {
                return (entry.hqBits, entry.hqGroup, entry.isIncumbentHQ4, entry.isSHVQ,
                        (k / entry.hqGroup) * entry.hqBlockBytes)
            }
            if let firstName, firstName.contains(".experts.") && !firstName.contains("shared_expert") {
                if firstName.contains("down_proj") {
                    return (format.expertDownBits, format.expertDownGroupSize,
                            format.expertDownIsIncumbentContainer, format.expertDownIsSHVQ,
                            format.expertDownRowBytes(k: k))
                }
                return (format.expertBits, format.expertGroupSize,
                        format.expertIsIncumbentContainer, format.expertIsSHVQ,
                        format.expertRowBytes(k: k))
            }
            return (format.bits, format.groupSize, format.denseIsIncumbentContainer,
                    format.denseIsSHVQ, format.rowBytes(k: k))
        }

        mutating func concatPlane(_ tensors: [(name: String, n: Int)], k: Int,
                         key: KeyPath<QWQTensorEntry, Int?>, extraZeroRows: Int = 0) throws -> QWTensorValue {
            if format.isNativeAffine {
                return try aqConcatPlane(tensors, k: k)
            }
            let rowBytes = resolvedPlaneLayout(tensors.first?.name, k: k).rowBytes
            let totalRows = tensors.reduce(0) { $0 + $1.n } + extraZeroRows
            let totalElems = (totalRows * rowBytes) / 2
            let t = try allocateWeight(QWTensorShape([totalElems], .f16))
            let p = t.region.buffer.contents().advanced(by: t.region.base + t.offset)
            var curByteOffset = 0
            if let pk, let io {
                var batch: [(name: String, dst: UnsafeMutableRawPointer)] = []
                batch.reserveCapacity(tensors.count)
                for (name, n) in tensors {
                    let resolved = resolveName(name)
                    let entry = try pk.entry(resolved)
                    guard entry.rawOff == nil, entry[keyPath: key] != nil else {
                        throw QWEngineDriverError.weightMissing(name, "needs hq4_off plane")
                    }
                    batch.append((name: resolved, dst: p.advanced(by: curByteOffset)))
                    curByteOffset += n * rowBytes
                }
                try io.loadBatch(batch, layout: .hq4)
            } else {
                for (name, n) in tensors {
                    let resolved = resolveName(name)
                    try quantizeSafetensorsToHQ4(
                        resolved, n: n, k: k,
                        into: t.region.buffer,
                        dstByteOffset: t.region.base + t.offset + curByteOffset
                    )
                    curByteOffset += n * rowBytes
                }
            }
            if extraZeroRows > 0 {
                memset(p.advanced(by: curByteOffset), 0, extraZeroRows * rowBytes)
            }
            return t
        }
        mutating func concatQ4(_ tensors: [(name: String, n: Int)], k: Int,
                               extraZeroRows: Int = 0) throws -> (QWWeightView, [QWWeightView]) {
            let layout = resolvedPlaneLayout(tensors.first?.name, k: k)
            let rowBytes = layout.rowBytes
            let totalN = tensors.reduce(0) { $0 + $1.n } + extraZeroRows
            let blocks = try concatPlane(tensors, k: k, key: \QWQTensorEntry.hq4Off, extraZeroRows: extraZeroRows)
            guard let bigView = QWWeightView(blocks: blocks, n: totalN, k: k,
                                                 bits: layout.bits, groupSize: layout.group,
                                                 isRotated: hl.weightBasis.rotatesActivations,
                                                 isIncumbentContainer: layout.incumbent,
                                                 isSHVQ: layout.shvq) else {
                throw QWEngineDriverError.weightMissing(tensors.first?.name ?? "concat", "concat view invalid")
            }
            var subViews: [QWWeightView] = []
            var row0 = 0
            for (name, n) in tensors {
                let subBlocks = QWTensorValue(
                    region: blocks.region,
                    offset: blocks.offset + row0 * rowBytes,
                    shape: QWTensorShape([n * rowBytes / 2], .f16)
                )
                guard let sv = bigView.subview(blocks: subBlocks, n: n, k: k) else {
                    throw QWEngineDriverError.weightMissing(name, "subview invalid")
                }
                subViews.append(sv)
                row0 += n
            }
            return (bigView, subViews)
        }
        mutating func packedGU(_ blk: String) throws -> QWMLPWeight {
            let gate = blk + "mlp.gate_proj.weight", up = blk + "mlp.up_proj.weight"
            let (bigView, _) = try concatQ4([(gate, geometry.intermediate), (up, geometry.intermediate)], k: geometry.hidden)
            guard let v = QWMLPWeight(blocks: bigView.blocks, n: 2 * geometry.intermediate, k: geometry.hidden,
                                    bits: bigView.bits, groupSize: bigView.groupSize,
                                    isRotated: bigView.isRotated,
                                    isIncumbentContainer: bigView.isIncumbentContainer,
                                    isSHVQ: bigView.isSHVQ) else {
                throw QWEngineDriverError.weightMissing(gate, "gu view invalid")
            }
            return v
        }

        /// A sparse MoE layer's weights: the router plane and the two banks.
        /// The router is the pack's RAW bf16 plane (the packer copies it
        /// verbatim — not quantised, not rotated), widened to f32; the banks
        /// are ONE contiguous view each, the experts stacked in order (C119).
        mutating func packedMoE(_ blk: String, experts: Int, topK: Int) throws -> MoEW {
            let I = geometry.moeIntermediate
            let H = geometry.hidden
            let resE = geometry.effectiveResidentExperts
            let coldE = max(0, experts - resE)
            let canPermute = coldE > 0
                && geometry.hasBiasCorrectedRouter
                && hasTensor(blk + "mlp.experts.\(experts - 1).gate_proj.weight")
            let expertPerm: [Int]?
            let router: QWTensorValue
            let routerBias: QWTensorValue?
            if canPermute {
                let rawR = try rawFloats(blk + "mlp.gate.weight", experts * H)
                let rawB = try rawFloats(blk + "mlp.gate.e_score_correction_bias", experts)
                let rawLn2 = (try? rawFloats(blk + "post_attention_layernorm.weight", H))
                    ?? [Float](repeating: 1.0, count: H)
                var wNorm = [Float](repeating: 0, count: experts)
                for e in 0..<experts {
                    let base = e * H
                    var sumSq: Float = 0
                    for c in 0..<H {
                        let v = rawR[base + c] * rawLn2[c]
                        sumSq += v * v
                    }
                    wNorm[e] = sqrt(sumSq)
                }
                let perm = (0..<experts).sorted { a, b in
                    if rawB[a] != rawB[b] { return rawB[a] > rawB[b] }
                    if wNorm[a] != wNorm[b] { return wNorm[a] > wNorm[b] }
                    return a < b
                }
                var permR = [Float](repeating: 0, count: experts * H)
                var permB = [Float](repeating: 0, count: experts)
                for dstIdx in 0..<experts {
                    let srcIdx = perm[dstIdx]
                    permB[dstIdx] = rawB[srcIdx]
                    let srcBase = srcIdx * H
                    let dstBase = dstIdx * H
                    for c in 0..<H {
                        permR[dstBase + c] = rawR[srcBase + c]
                    }
                }
                let rT = try allocateWeight(QWTensorShape([experts * H], .f32))
                permR.withUnsafeBytes {
                    rT.region.buffer.contents()
                        .advanced(by: rT.region.base + rT.offset)
                        .copyMemory(from: $0.baseAddress!, byteCount: experts * H * 4)
                }
                let bT = try allocateWeight(QWTensorShape([experts], .f32))
                permB.withUnsafeBytes {
                    bT.region.buffer.contents()
                        .advanced(by: bT.region.base + bT.offset)
                        .copyMemory(from: $0.baseAddress!, byteCount: experts * 4)
                }
                router = rT
                routerBias = bT
                expertPerm = perm
            } else {
                // The router: [E, H] f32, from the raw plane.
                router = try f32w(blk + "mlp.gate.weight", experts * H)
                routerBias = geometry.hasBiasCorrectedRouter
                    ? try f32w(blk + "mlp.gate.e_score_correction_bias", experts)
                    : nil
                expertPerm = nil
            }
            let gu: QWWeightView
            let down: QWWeightView
            if st != nil, hasTensor(blk + "mlp.switch_mlp.gate_proj.weight") {
                let guBlocks: QWTensorValue
                if format.isNativeAffine {
                    let guElems = (resE * 2 * I * format.rowBytes(k: H)) / 2
                    guBlocks = try allocateWeight(QWTensorShape([guElems], .f16))
                    try repackSafetensorsSwitchGateUpToAQ(
                        gateName: blk + "mlp.switch_mlp.gate_proj.weight",
                        upName: blk + "mlp.switch_mlp.up_proj.weight",
                        experts: resE, i: I, h: H,
                        into: guBlocks.region.buffer,
                        dstByteOffset: guBlocks.region.base + guBlocks.offset
                    )
                } else {
                    let guElems = resE * 2 * I * (H / 32) * 10
                    guBlocks = try allocateWeight(QWTensorShape([guElems], .f16))
                    try quantizeSafetensorsSwitchGateUp(
                        gateName: blk + "mlp.switch_mlp.gate_proj.weight",
                        upName: blk + "mlp.switch_mlp.up_proj.weight",
                        experts: resE, i: I, h: H,
                        into: guBlocks.region.buffer,
                        dstByteOffset: guBlocks.region.base + guBlocks.offset
                    )
                }
                let downBlocks = try planeView(
                    blk + "mlp.switch_mlp.down_proj.weight",
                    n: resE * H, k: I, key: \QWQTensorEntry.hq4Off
                )
                guard let guView = QWWeightView(blocks: guBlocks, n: resE * 2 * I, k: H,
                                             bits: format.bits, groupSize: format.groupSize,
                                             isRotated: hl.weightBasis.rotatesActivations),
                      let downView = QWWeightView(blocks: downBlocks, n: resE * H, k: I,
                                                 bits: format.bits, groupSize: format.groupSize,
                                                 isRotated: hl.weightBasis.rotatesActivations) else {
                    throw QWEngineDriverError.weightMissing(blk + "mlp.switch_mlp", "moe bank view invalid")
                }
                gu = guView
                down = downView
            } else if pk != nil, hasTensor(blk + "mlp.experts.gate_up_proj") {
                let guBlocks = try planeView(
                    blk + "mlp.experts.gate_up_proj",
                    n: resE * 2 * I, k: H, key: \QWQTensorEntry.hq4Off
                )
                let downBlocks = try planeView(
                    blk + "mlp.experts.down_proj",
                    n: resE * H, k: I, key: \QWQTensorEntry.hq4Off
                )
                guard let guView = QWWeightView(blocks: guBlocks, n: resE * 2 * I, k: H,
                                             bits: format.expertBits, groupSize: format.expertGroupSize,
                                             isRotated: hl.weightBasis.rotatesActivations,
                                             isIncumbentContainer: format.expertIsIncumbentContainer,
                                             isSHVQ: format.expertIsSHVQ),
                      let downView = QWWeightView(blocks: downBlocks, n: resE * H, k: I,
                                                 bits: format.expertDownBits, groupSize: format.expertDownGroupSize,
                                                 isRotated: hl.weightBasis.rotatesActivations,
                                                 isIncumbentContainer: format.expertDownIsIncumbentContainer,
                                                 isSHVQ: format.expertDownIsSHVQ) else {
                    throw QWEngineDriverError.weightMissing(blk + "mlp.experts.gate_up_proj", "moe bank view invalid")
                }
                gu = guView
                down = downView
            } else {
                // The gate/up bank: every resident expert's gate rows, then its up rows.
                var guSpecs: [(name: String, n: Int)] = []
                for idx in 0..<resE {
                    let e = expertPerm?[idx] ?? idx
                    guSpecs.append((blk + "mlp.experts.\(e).gate_proj.weight", I))
                    guSpecs.append((blk + "mlp.experts.\(e).up_proj.weight", I))
                }
                let (guBank, _) = try concatQ4(guSpecs, k: H)
                // The down bank: every resident expert's [H, I] slab.
                var downSpecs: [(name: String, n: Int)] = []
                for idx in 0..<resE {
                    let e = expertPerm?[idx] ?? idx
                    downSpecs.append((blk + "mlp.experts.\(e).down_proj.weight", H))
                }
                let (downBank, _) = try concatQ4(downSpecs, k: I)
                guard let guView = QWWeightView(blocks: guBank.blocks, n: resE * 2 * I, k: H,
                                             bits: format.expertBits, groupSize: format.expertGroupSize,
                                             isRotated: hl.weightBasis.rotatesActivations,
                                             isIncumbentContainer: format.expertIsIncumbentContainer,
                                             isSHVQ: format.expertIsSHVQ),
                      let downView = QWWeightView(blocks: downBank.blocks, n: resE * H, k: I,
                                                 bits: format.expertDownBits, groupSize: format.expertDownGroupSize,
                                                 isRotated: hl.weightBasis.rotatesActivations,
                                                 isIncumbentContainer: format.expertDownIsIncumbentContainer,
                                                 isSHVQ: format.expertDownIsSHVQ) else {
                    throw QWEngineDriverError.weightMissing(blk + "mlp.experts", "moe bank view invalid")
                }
                gu = guView
                down = downView
            }
            let coldBanks = coldE > 0 ? try ensureSharedColdBanks(coldE: coldE, i: I, h: H) : nil
            let coldStreamer: ColdExpertStreamer?
            if let cb = coldBanks, let io = self.io {
                let guPtr = cb.gu.blocks.region.buffer.contents().advanced(by: cb.gu.blocks.region.base + cb.gu.blocks.offset)
                let downPtr = cb.down.blocks.region.buffer.contents().advanced(by: cb.down.blocks.region.base + cb.down.blocks.offset)
                let probeSuffix = "mlp.gate.weight"
                let resolvedProbe = resolveName(blk + probeSuffix)
                let resolvedLayerPrefix = resolvedProbe.hasSuffix(probeSuffix)
                    ? String(resolvedProbe.dropLast(probeSuffix.count))
                    : resolveName(blk)
                coldStreamer = ColdExpertStreamer(
                    io: io,
                    layerPrefix: resolvedLayerPrefix,
                    residentExperts: resE,
                    totalExperts: experts,
                    stagingSlots: coldE,
                    intermediate: I,
                    hidden: H,
                    guRowBytes: gu.rowBytes,
                    downRowBytes: down.rowBytes,
                    guBasePtr: guPtr,
                    downBasePtr: downPtr,
                    expertPermutation: expertPerm
                )
            } else {
                coldStreamer = nil
            }
            let ISh = geometry.sharedExpertIntermediate
            let sharedGateUp: QWWeightView?
            let sharedDown: QWWeightView?
            let sharedGateScalar: QWTensorValue?
            if ISh > 0 {
                let shPrefix = hasTensor(blk + "mlp.shared_experts.gate_proj.weight")
                    ? (blk + "mlp.shared_experts.")
                    : (blk + "mlp.shared_expert.")
                let (shGU, _) = try concatQ4([
                    (shPrefix + "gate_proj.weight", ISh),
                    (shPrefix + "up_proj.weight", ISh)
                ], k: H)
                sharedGateUp = shGU
                sharedDown = try q4(shPrefix + "down_proj.weight", n: H, k: ISh)
                sharedGateScalar = geometry.sharedExpertHasGate
                    ? try f32w(blk + "mlp.shared_expert_gate.weight", H)
                    : nil
            } else {
                sharedGateUp = nil
                sharedDown = nil
                sharedGateScalar = nil
            }
            return MoEW(router: router, gateUp: gu, down: down,
                        numExperts: experts, expertsPerTok: topK,
                        sharedGateUp: sharedGateUp, sharedDown: sharedDown,
                        sharedGateScalar: sharedGateScalar,
                        routerBias: routerBias,
                        coldGateUp: coldBanks?.gu, coldDown: coldBanks?.down,
                        coldStreamer: coldStreamer)
        }

        /// Every layer's weights, in the plan's order.
        mutating func loadLayers(geometry: QWModelGeometry) throws -> [LayerW] {
            var ws: [LayerW] = []
            for l in 0..<geometry.numLayers {
                let blk = "\(QWEngineDriver.P)layers.\(l)."
                let ln1 = try f16wNorm(blk + "input_layernorm.weight", geometry.hidden)
                let ln2 = try f16wNorm(blk + "post_attention_layernorm.weight", geometry.hidden)
                // A sparse MoE layer has no dense MLP: its weights are the
                // router and the two banks (Task 4.4a). Its attention is
                // the same full attention as a dense layer's.
                let experts = geometry.isMoELayer(l) ? geometry.numExperts : 0
                let moe: MoEW? = experts > 0
                    ? try packedMoE(blk, experts: experts, topK: geometry.expertsPerTok)
                    : nil
                let gu: QWMLPWeight
                let down: QWWeightView
                if let moe {
                    guard let v = QWMLPWeight(blocks: moe.gateUp.blocks,
                                             n: 2 * geometry.moeIntermediate,
                                             k: geometry.hidden,
                                             bits: moe.gateUp.bits,
                                             groupSize: moe.gateUp.groupSize,
                                             isRotated: moe.gateUp.isRotated,
                                             isIncumbentContainer: moe.gateUp.isIncumbentContainer,
                                             isSHVQ: moe.gateUp.isSHVQ) else {
                        throw QWEngineDriverError.weightMissing(blk + "mlp.experts", "moe gu view invalid")
                    }
                    gu = v
                    down = try firstExpert(moe.down, rows: geometry.hidden)
                } else {
                    gu = try packedGU(blk)
                    down = try q4(blk + "mlp.down_proj.weight", n: geometry.hidden, k: geometry.intermediate)
                }
                if geometry.isFullAttention(l) {
                    if geometry.usesNoPEMLA {
                        ws.append(try loadMLAAttnLayer(
                            blk: blk, layer: l, ln1: ln1, ln2: ln2,
                            gu: gu, down: down, moe: moe, geometry: geometry))
                        continue
                    }
                    let attnSpecs = [
                        (blk + "self_attn.q_proj.weight", geometry.qProjOut),
                        (blk + "self_attn.k_proj.weight", (geometry.kvHeads * geometry.headDim)),
                        (blk + "self_attn.v_proj.weight", (geometry.kvHeads * geometry.headDim))
                    ]
                    let (qkvProj, attnSub) = try concatQ4(attnSpecs, k: geometry.hidden)
                    let q = attnSub[0], k = attnSub[1], v = attnSub[2]
                    let o = try q4(blk + "self_attn.o_proj.weight",
                                   n: geometry.hidden, k: geometry.qHeads * geometry.headDim)
                    // q_norm / k_norm are optional: a standard family (llama,
                    // mistral, qwen2) has no RMSNorm on q/k, and its q and k are
                    // projected and roped as they are.
                    let wq: QWTensorValue? = hasTensor(blk + "self_attn.q_norm.weight")
                        ? try f16wNorm(blk + "self_attn.q_norm.weight", geometry.headDim) : nil
                    // wk contract (C16 law): [kvHeads·headDim] — tile k_norm ×4
                    // (a raw [256] dump makes the rope door read OOB).
                    var wkT: QWTensorValue?
                    if hasTensor(blk + "self_attn.k_norm.weight") {
                        let knBits = try normGammaBits(blk + "self_attn.k_norm.weight", geometry.headDim)
                        let t = try allocateWeight(QWTensorShape([(geometry.kvHeads * geometry.headDim)], .f16))
                        let p = t.region.buffer.contents().advanced(by: t.region.base + t.offset)
                        for h in 0..<geometry.kvHeads {
                            knBits.withUnsafeBytes {
                                p.advanced(by: h * geometry.headDim * 2)
                                 .copyMemory(from: $0.baseAddress!, byteCount: geometry.headDim * 2)
                            }
                        }
                        wkT = t
                    }
                    let mhcW: MHCLayerWeights?
                    if let mhc = geometry.mhcConfig, l >= mhc.start, l < mhc.end {
                        mhcW = try loadMHC(blk, mhc: mhc)
                    } else {
                        mhcW = nil
                    }
                    ws.append(LayerW(ln1: ln1, ln2: ln2, gu: gu, down: down,
                                     kind: moe == nil ? .fullAttention : .moe,
                                     gdn: nil,
                                     attn: AttnW(qkvProj: qkvProj, q: q, k: k, v: v, o: o, wq: wq, wkTiled: wkT),
                                     moe: moe, mhc: mhcW))
                } else {
                    if geometry.gdnVectorDecay {
                        ws.append(try loadKDALayer(
                            blk: blk, layer: l, ln1: ln1, ln2: ln2,
                            gu: gu, down: down, moe: moe, geometry: geometry))
                        continue
                    }
                    let gdnSpecs = [
                        (blk + "linear_attn.in_proj_qkv.weight", geometry.gdnQKVChannels),
                        (blk + "linear_attn.in_proj_z.weight", geometry.gdnZChannels),
                        (blk + "linear_attn.in_proj_a.weight", geometry.gdnVHeads),
                        (blk + "linear_attn.in_proj_b.weight", geometry.gdnVHeads)
                    ]
                    let (inProj, gdnSub) = try concatQ4(gdnSpecs, k: geometry.hidden)
                    let qkv = gdnSub[0], z = gdnSub[1], a = gdnSub[2], b = gdnSub[3]
                    let conv = try f16w(blk + "linear_attn.conv1d.weight", geometry.gdnQKVChannels * geometry.gdnConvKernel)
                    // The GDN's gated norm multiplies by `w` outright — it is
                    // not a `1 + w` norm, so its gamma is loaded verbatim.
                    let gnw = try f16w(blk + "linear_attn.norm.weight", geometry.gdnHeadDim)
                    let aLog = try f32w(blk + "linear_attn.A_log", geometry.gdnVHeads)
                    let dtB = try f32w(blk + "linear_attn.dt_bias", geometry.gdnVHeads)
                    let out = try q4(blk + "linear_attn.out_proj.weight", n: geometry.hidden, k: geometry.gdnZChannels)
                    let mhcW: MHCLayerWeights?
                    if let mhc = geometry.mhcConfig, l >= mhc.start, l < mhc.end {
                        mhcW = try loadMHC(blk, mhc: mhc)
                    } else {
                        mhcW = nil
                    }
                    ws.append(LayerW(ln1: ln1, ln2: ln2, gu: gu, down: down,
                                     kind: moe == nil ? .gdn : .gdnMoe,
                                     gdn: GdnW(inProj: inProj, qkv: qkv, z: z, a: a, b: b, conv: conv, normW: gnw,
                                               aLog: aLog, dtBias: dtB, out: out),
                                     attn: nil, moe: moe, mhc: mhcW))
                }
            }
            return ws
        }
    }
}
