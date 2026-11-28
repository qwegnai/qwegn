// QWInkling2DSpark+Tree.swift — Multi-branch DSpark tree templates, GPU top-K Markov rollout & KV compaction.
import Foundation
import Metal
import QWAttn
import QWCore
import QWHAL

enum DSparkTreeMode: String, Sendable {
    case linear
    case tree4_d2
    case tree5_d3
    case tree6_d4
    case tree7_d5
    case tree8_d5
    case tree8_d6
    case tree16_d5
    case tree16_d6
    case tree16_d7

    static func resolveFromEnvironment() -> DSparkTreeMode {
        guard let raw = ProcessInfo.processInfo.environment["QWG_DSPARK_TREE"]?.lowercased() else {
            return .linear
        }
        return DSparkTreeMode(rawValue: raw) ?? .linear
    }
}

struct Ink2TreeVerifyPlan: Sendable, Equatable {
    let nodesCount: Int
    let parents: [Int]
    let children: [[Int]]
    let depths: [UInt32]
    let ancMask: [UInt32]
    let sconvLags: [SIMD4<Int32>]
    let attnTreeMeta: QWAttnSinkRel.TreeMeta

    init(parents: [Int]) {
        let count = parents.count
        self.nodesCount = count
        self.parents = parents
        var ch = [[Int]](repeating: [], count: count)
        var dep = [UInt32](repeating: 0, count: count)
        var mask = [UInt32](repeating: 0, count: count)
        for u in 0..<count {
            let p = parents[u]
            if p >= 0 {
                ch[p].append(u)
                dep[u] = dep[p] + 1
                mask[u] = mask[p] | (UInt32(1) << UInt32(u))
            } else {
                mask[u] = 1
            }
        }
        var lags = [SIMD4<Int32>](repeating: SIMD4<Int32>(-1, -2, -3, 0), count: 16)
        for u in 0..<count {
            var hist = [Int32]()
            var cur = parents[u]
            var pastSteps = 0
            for _ in 0..<3 {
                if cur >= 0 {
                    hist.append(Int32(cur))
                    cur = parents[cur]
                } else {
                    pastSteps += 1
                    hist.append(Int32(-pastSteps))
                }
            }
            lags[u] = SIMD4<Int32>(hist[0], hist[1], hist[2], Int32(dep[u]))
        }
        self.children = ch
        self.depths = dep
        self.ancMask = mask
        self.sconvLags = lags
        self.attnTreeMeta = QWAttnSinkRel.TreeMeta(ancMask: mask, depths: dep)
    }
}

struct DSparkTreeBranch: Sendable {
    let prevTokNode: Int
    let inStateSlot: Int
    let keepK: Int
    let outNodeBase: Int
}

struct DSparkTreeDepthLevel: Sendable {
    let branches: [DSparkTreeBranch]
}

struct DSparkTreeTemplate: Sendable {
    let mode: DSparkTreeMode
    let plan: Ink2TreeVerifyPlan
    let levels: [DSparkTreeDepthLevel]

    static func make(mode: DSparkTreeMode, maxNodes: Int = 16, maxDepth: Int = 7) -> DSparkTreeTemplate? {
        guard mode != .linear, maxNodes >= 4, maxDepth >= 2 else { return nil }
        let fullParents: [Int]
        let fullLevels: [DSparkTreeDepthLevel]
        switch mode {
        case .linear:
            return nil
        case .tree4_d2:
            fullParents = [-1, 0, 0, 1]
            fullLevels = [
                DSparkTreeDepthLevel(branches: [
                    DSparkTreeBranch(prevTokNode: 0, inStateSlot: 0, keepK: 2, outNodeBase: 1)
                ]),
                DSparkTreeDepthLevel(branches: [
                    DSparkTreeBranch(prevTokNode: 1, inStateSlot: 0, keepK: 1, outNodeBase: 3)
                ])
            ]
        case .tree5_d3:
            fullParents = [-1, 0, 0, 1, 3]
            fullLevels = [
                DSparkTreeDepthLevel(branches: [
                    DSparkTreeBranch(prevTokNode: 0, inStateSlot: 0, keepK: 2, outNodeBase: 1)
                ]),
                DSparkTreeDepthLevel(branches: [
                    DSparkTreeBranch(prevTokNode: 1, inStateSlot: 0, keepK: 1, outNodeBase: 3)
                ]),
                DSparkTreeDepthLevel(branches: [
                    DSparkTreeBranch(prevTokNode: 3, inStateSlot: 0, keepK: 1, outNodeBase: 4)
                ])
            ]
        case .tree6_d4:
            fullParents = [-1, 0, 0, 1, 3, 4]
            fullLevels = [
                DSparkTreeDepthLevel(branches: [
                    DSparkTreeBranch(prevTokNode: 0, inStateSlot: 0, keepK: 2, outNodeBase: 1)
                ]),
                DSparkTreeDepthLevel(branches: [
                    DSparkTreeBranch(prevTokNode: 1, inStateSlot: 0, keepK: 1, outNodeBase: 3)
                ]),
                DSparkTreeDepthLevel(branches: [
                    DSparkTreeBranch(prevTokNode: 3, inStateSlot: 0, keepK: 1, outNodeBase: 4)
                ]),
                DSparkTreeDepthLevel(branches: [
                    DSparkTreeBranch(prevTokNode: 4, inStateSlot: 0, keepK: 1, outNodeBase: 5)
                ])
            ]
        case .tree7_d5:
            fullParents = [-1, 0, 0, 1, 3, 4, 5]
            fullLevels = [
                DSparkTreeDepthLevel(branches: [
                    DSparkTreeBranch(prevTokNode: 0, inStateSlot: 0, keepK: 2, outNodeBase: 1)
                ]),
                DSparkTreeDepthLevel(branches: [
                    DSparkTreeBranch(prevTokNode: 1, inStateSlot: 0, keepK: 1, outNodeBase: 3)
                ]),
                DSparkTreeDepthLevel(branches: [
                    DSparkTreeBranch(prevTokNode: 3, inStateSlot: 0, keepK: 1, outNodeBase: 4)
                ]),
                DSparkTreeDepthLevel(branches: [
                    DSparkTreeBranch(prevTokNode: 4, inStateSlot: 0, keepK: 1, outNodeBase: 5)
                ]),
                DSparkTreeDepthLevel(branches: [
                    DSparkTreeBranch(prevTokNode: 5, inStateSlot: 0, keepK: 1, outNodeBase: 6)
                ])
            ]
        case .tree8_d5:
            fullParents = [-1, 0, 0, 1, 2, 3, 5, 6]
            fullLevels = [
                DSparkTreeDepthLevel(branches: [
                    DSparkTreeBranch(prevTokNode: 0, inStateSlot: 0, keepK: 2, outNodeBase: 1)
                ]),
                DSparkTreeDepthLevel(branches: [
                    DSparkTreeBranch(prevTokNode: 1, inStateSlot: 0, keepK: 1, outNodeBase: 3),
                    DSparkTreeBranch(prevTokNode: 2, inStateSlot: 0, keepK: 1, outNodeBase: 4)
                ]),
                DSparkTreeDepthLevel(branches: [
                    DSparkTreeBranch(prevTokNode: 3, inStateSlot: 0, keepK: 1, outNodeBase: 5)
                ]),
                DSparkTreeDepthLevel(branches: [
                    DSparkTreeBranch(prevTokNode: 5, inStateSlot: 0, keepK: 1, outNodeBase: 6)
                ]),
                DSparkTreeDepthLevel(branches: [
                    DSparkTreeBranch(prevTokNode: 6, inStateSlot: 0, keepK: 1, outNodeBase: 7)
                ])
            ]
        case .tree8_d6:
            fullParents = [-1, 0, 0, 1, 3, 4, 5, 6]
            fullLevels = [
                DSparkTreeDepthLevel(branches: [
                    DSparkTreeBranch(prevTokNode: 0, inStateSlot: 0, keepK: 2, outNodeBase: 1)
                ]),
                DSparkTreeDepthLevel(branches: [
                    DSparkTreeBranch(prevTokNode: 1, inStateSlot: 0, keepK: 1, outNodeBase: 3)
                ]),
                DSparkTreeDepthLevel(branches: [
                    DSparkTreeBranch(prevTokNode: 3, inStateSlot: 0, keepK: 1, outNodeBase: 4)
                ]),
                DSparkTreeDepthLevel(branches: [
                    DSparkTreeBranch(prevTokNode: 4, inStateSlot: 0, keepK: 1, outNodeBase: 5)
                ]),
                DSparkTreeDepthLevel(branches: [
                    DSparkTreeBranch(prevTokNode: 5, inStateSlot: 0, keepK: 1, outNodeBase: 6)
                ]),
                DSparkTreeDepthLevel(branches: [
                    DSparkTreeBranch(prevTokNode: 6, inStateSlot: 0, keepK: 1, outNodeBase: 7)
                ])
            ]
        case .tree16_d5:
            fullParents = [-1, 0, 0, 0, 1, 1, 2, 2, 3, 4, 5, 6, 8, 9, 11, 13]
            fullLevels = [
                DSparkTreeDepthLevel(branches: [
                    DSparkTreeBranch(prevTokNode: 0, inStateSlot: 0, keepK: 3, outNodeBase: 1)
                ]),
                DSparkTreeDepthLevel(branches: [
                    DSparkTreeBranch(prevTokNode: 1, inStateSlot: 0, keepK: 2, outNodeBase: 4),
                    DSparkTreeBranch(prevTokNode: 2, inStateSlot: 0, keepK: 2, outNodeBase: 6),
                    DSparkTreeBranch(prevTokNode: 3, inStateSlot: 0, keepK: 1, outNodeBase: 8)
                ]),
                DSparkTreeDepthLevel(branches: [
                    DSparkTreeBranch(prevTokNode: 4, inStateSlot: 0, keepK: 1, outNodeBase: 9),
                    DSparkTreeBranch(prevTokNode: 5, inStateSlot: 0, keepK: 1, outNodeBase: 10),
                    DSparkTreeBranch(prevTokNode: 6, inStateSlot: 1, keepK: 1, outNodeBase: 11),
                    DSparkTreeBranch(prevTokNode: 8, inStateSlot: 2, keepK: 1, outNodeBase: 12)
                ]),
                DSparkTreeDepthLevel(branches: [
                    DSparkTreeBranch(prevTokNode: 9, inStateSlot: 0, keepK: 1, outNodeBase: 13),
                    DSparkTreeBranch(prevTokNode: 11, inStateSlot: 2, keepK: 1, outNodeBase: 14)
                ]),
                DSparkTreeDepthLevel(branches: [
                    DSparkTreeBranch(prevTokNode: 13, inStateSlot: 0, keepK: 1, outNodeBase: 15)
                ])
            ]
        case .tree16_d6:
            fullParents = [-1, 0, 0, 0, 1, 1, 2, 3, 4, 4, 5, 6, 8, 10, 12, 14]
            fullLevels = [
                DSparkTreeDepthLevel(branches: [
                    DSparkTreeBranch(prevTokNode: 0, inStateSlot: 0, keepK: 3, outNodeBase: 1)
                ]),
                DSparkTreeDepthLevel(branches: [
                    DSparkTreeBranch(prevTokNode: 1, inStateSlot: 0, keepK: 2, outNodeBase: 4),
                    DSparkTreeBranch(prevTokNode: 2, inStateSlot: 0, keepK: 1, outNodeBase: 6),
                    DSparkTreeBranch(prevTokNode: 3, inStateSlot: 0, keepK: 1, outNodeBase: 7)
                ]),
                DSparkTreeDepthLevel(branches: [
                    DSparkTreeBranch(prevTokNode: 4, inStateSlot: 0, keepK: 2, outNodeBase: 8),
                    DSparkTreeBranch(prevTokNode: 5, inStateSlot: 0, keepK: 1, outNodeBase: 10),
                    DSparkTreeBranch(prevTokNode: 6, inStateSlot: 1, keepK: 1, outNodeBase: 11)
                ]),
                DSparkTreeDepthLevel(branches: [
                    DSparkTreeBranch(prevTokNode: 8, inStateSlot: 0, keepK: 1, outNodeBase: 12),
                    DSparkTreeBranch(prevTokNode: 10, inStateSlot: 1, keepK: 1, outNodeBase: 13)
                ]),
                DSparkTreeDepthLevel(branches: [
                    DSparkTreeBranch(prevTokNode: 12, inStateSlot: 0, keepK: 1, outNodeBase: 14)
                ]),
                DSparkTreeDepthLevel(branches: [
                    DSparkTreeBranch(prevTokNode: 14, inStateSlot: 0, keepK: 1, outNodeBase: 15)
                ])
            ]
        case .tree16_d7:
            fullParents = [-1, 0, 0, 0, 1, 1, 2, 4, 4, 5, 7, 8, 10, 11, 12, 14]
            fullLevels = [
                DSparkTreeDepthLevel(branches: [
                    DSparkTreeBranch(prevTokNode: 0, inStateSlot: 0, keepK: 3, outNodeBase: 1)
                ]),
                DSparkTreeDepthLevel(branches: [
                    DSparkTreeBranch(prevTokNode: 1, inStateSlot: 0, keepK: 2, outNodeBase: 4),
                    DSparkTreeBranch(prevTokNode: 2, inStateSlot: 0, keepK: 1, outNodeBase: 6)
                ]),
                DSparkTreeDepthLevel(branches: [
                    DSparkTreeBranch(prevTokNode: 4, inStateSlot: 0, keepK: 2, outNodeBase: 7),
                    DSparkTreeBranch(prevTokNode: 5, inStateSlot: 0, keepK: 1, outNodeBase: 9)
                ]),
                DSparkTreeDepthLevel(branches: [
                    DSparkTreeBranch(prevTokNode: 7, inStateSlot: 0, keepK: 1, outNodeBase: 10),
                    DSparkTreeBranch(prevTokNode: 8, inStateSlot: 0, keepK: 1, outNodeBase: 11)
                ]),
                DSparkTreeDepthLevel(branches: [
                    DSparkTreeBranch(prevTokNode: 10, inStateSlot: 0, keepK: 1, outNodeBase: 12),
                    DSparkTreeBranch(prevTokNode: 11, inStateSlot: 1, keepK: 1, outNodeBase: 13)
                ]),
                DSparkTreeDepthLevel(branches: [
                    DSparkTreeBranch(prevTokNode: 12, inStateSlot: 0, keepK: 1, outNodeBase: 14)
                ]),
                DSparkTreeDepthLevel(branches: [
                    DSparkTreeBranch(prevTokNode: 14, inStateSlot: 0, keepK: 1, outNodeBase: 15)
                ])
            ]
        }

        let clampedMax = min(fullParents.count, min(16, maxNodes))
        if clampedMax == fullParents.count && fullLevels.count <= maxDepth {
            return DSparkTreeTemplate(mode: mode, plan: Ink2TreeVerifyPlan(parents: fullParents), levels: fullLevels)
        }
        var trimmedLevels: [DSparkTreeDepthLevel] = []
        var keptNodes = 1
        for level in fullLevels.prefix(maxDepth) {
            let kept = level.branches.compactMap { b -> DSparkTreeBranch? in
                guard b.prevTokNode < clampedMax, b.outNodeBase < clampedMax else { return nil }
                let k = min(b.keepK, clampedMax - b.outNodeBase)
                if k > 0 { keptNodes = max(keptNodes, b.outNodeBase + k) }
                return k > 0 ? DSparkTreeBranch(
                    prevTokNode: b.prevTokNode, inStateSlot: b.inStateSlot, keepK: k, outNodeBase: b.outNodeBase
                ) : nil
            }
            if !kept.isEmpty { trimmedLevels.append(DSparkTreeDepthLevel(branches: kept)) }
        }
        return DSparkTreeTemplate(
            mode: mode,
            plan: Ink2TreeVerifyPlan(parents: Array(fullParents.prefix(keptNodes))),
            levels: trimmedLevels
        )
    }
}

extension QWInkling2DSpark {
    private static func packU32_16(_ arr: [Int]) -> (
        UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
        UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32
    ) {
        func g(_ i: Int) -> UInt32 { i < arr.count ? UInt32(arr[i]) : 0 }
        return (g(0), g(1), g(2), g(3), g(4), g(5), g(6), g(7),
                g(8), g(9), g(10), g(11), g(12), g(13), g(14), g(15))
    }

    func encodeSConvVerifyCommitTree(
        _ enc: any MTLComputeCommandEncoder,
        session: QWInkling2Session,
        keptNodes: [Int]
    ) {
        guard !keptNodes.isEmpty, keptNodes.count <= 16, let vInBuf = session.convVerifyInputsBuf else { return }
        if keptNodes.enumerated().allSatisfy({ $0.offset == $0.element }) {
            encodeSConvVerifyCommit(enc, session: session, kept: keptNodes.count)
            return
        }
        let d = session.descriptor
        let H = d.hidden
        let kvDim = d.kvHeads * d.headDim
        let numVisits = session.executionPlan.count
        var p = SConvVerifyCommitTreeP(
            numVisits: UInt32(numVisits), kvDim: UInt32(kvDim), H: UInt32(H), kept: UInt32(keptNodes.count),
            stateKvStride: UInt32(session.convStateKvStrideElems),
            stateHStride: UInt32(session.convStateHStrideElems),
            verifyKvStride: UInt32(session.convVerifyKvStrideElems),
            verifyHStride: UInt32(session.convVerifyHStrideElems),
            keptNodes: Self.packU32_16(keptNodes)
        )
        let totalCh = 2 * kvDim + 2 * H
        enc.setComputePipelineState(pSConvVerifyCommitTree)
        enc.setBuffer(session.convStatesSlab, offset: 0, index: 0)
        enc.setBuffer(vInBuf, offset: 0, index: 1)
        enc.setBytes(&p, length: MemoryLayout<SConvVerifyCommitTreeP>.stride, index: 2)
        enc.dispatchThreads(MTLSize(width: totalCh, height: numVisits, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(256, totalCh), height: 1, depth: 1))
    }

    func encodeTreeKVCompact(
        _ enc: any MTLComputeCommandEncoder,
        session: QWInkling2Session,
        oldPos: Int,
        keptNodes: [Int]
    ) {
        guard keptNodes.count > 1,
              keptNodes.enumerated().contains(where: { $0.offset != $0.element }) else { return }
        let nKV = session.descriptor.kvHeads
        let headDim = session.descriptor.headDim
        let packed = Self.packU32_16(keptNodes)
        enc.setComputePipelineState(pTreeKVCompact)
        for kv in session.visitKVs {
            var kp = TreeKVCompactP(
                oldPos: UInt32(oldPos), kept: UInt32(keptNodes.count),
                nKV: UInt32(nKV), headDim: UInt32(headDim),
                ringCap: UInt32(kv.ringCap), kvMode: session.kvModeRaw,
                isSliding: kv.isSliding ? 1 : 0, pad: 0,
                keptNodes: packed
            )
            enc.setBuffer(kv.kCache, offset: 0, index: 0)
            enc.setBuffer(kv.vCache, offset: 0, index: 1)
            enc.setBuffer(kv.kMeta, offset: 0, index: 2)
            enc.setBuffer(kv.vMeta, offset: 0, index: 3)
            enc.setBytes(&kp, length: MemoryLayout<TreeKVCompactP>.stride, index: 4)
            enc.dispatchThreadgroups(MTLSize(width: nKV, height: 2, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        }
    }

    func encodeDraftTree(
        anchor: Int,
        template: DSparkTreeTemplate,
        on cb: any MTLCommandBuffer,
        encoder encOpt: inout (any MTLComputeCommandEncoder)?,
        profiler: QWInkling2Profiler?
    ) throws {
        let depthCount = template.levels.count
        guard depthCount > 0, depthCount < config.blockSize else { return }
        try encodeDraftTransformerAndBaseLogits(
            anchor: anchor, draftCount: depthCount, tokOutBuf: treeTokBuf,
            on: cb, encoder: &encOpt, profiler: profiler
        )
        let H = config.hiddenSize
        let R = config.markovRank
        let V = ink2.unpaddedVocabSize
        let mEnc = try beginStage(.dsparkMarkov, on: cb, encoder: &encOpt, profiler: profiler)
        let partCount = min(512, (V + 7) / 8)
        let topkPSO = foldWIsAQ8 ? pTopKAQ8Stage1 : pTopKStage1

        for (stepIdx, level) in template.levels.enumerated() {
            let P = level.branches.count
            guard P >= 1, P <= 4 else { continue }
            let inStBuf = (stepIdx % 2 == 0) ? markovStateBatchBufA : markovStateBatchBufB
            let outStBuf = (stepIdx % 2 == 0) ? markovStateBatchBufB : markovStateBatchBufA
            func b4(_ fn: (DSparkTreeBranch) -> Int) -> (UInt32, UInt32, UInt32, UInt32) {
                (P > 0 ? UInt32(fn(level.branches[0])) : 0,
                 P > 1 ? UInt32(fn(level.branches[1])) : 0,
                 P > 2 ? UInt32(fn(level.branches[2])) : 0,
                 P > 3 ? UInt32(fn(level.branches[3])) : 0)
            }
            var msp = MarkovStepBatchP(
                P: UInt32(P), H: UInt32(H), projRows: UInt32(3 * R), rank: UInt32(R),
                ctxStepIdx: UInt32(stepIdx), resetState: stepIdx == 0 ? 1 : 0,
                parentStateSlots: b4 { $0.inStateSlot },
                prevTokNodes: b4 { $0.prevTokNode }
            )
            mEnc.setComputePipelineState(pMarkovStepBatch)
            mEnc.setBuffer(treeTokBuf, offset: 0, index: 0)
            mEnc.setBuffer(markovW1.buffer, offset: markovW1.offset, index: 1)
            mEnc.setBuffer(rnnProjW.buffer, offset: rnnProjW.offset, index: 2)
            mEnc.setBuffer(markovCtxF32Buf, offset: 0, index: 3)
            mEnc.setBuffer(inStBuf, offset: 0, index: 4)
            mEnc.setBuffer(outStBuf, offset: 0, index: 5)
            mEnc.setBuffer(markovLatentBatchBuf, offset: 0, index: 6)
            mEnc.setBytes(&msp, length: MemoryLayout<MarkovStepBatchP>.stride, index: 7)
            mEnc.dispatchThreadgroups(MTLSize(width: P, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

            var tp = TopKP(
                stepIdx: UInt32(stepIdx), V: UInt32(V), rank: UInt32(R),
                partCount: UInt32(partCount), numParents: UInt32(P),
                softcap: ink2.finalLogitSoftcapping,
                keepK: b4 { $0.keepK },
                outNodeBase: b4 { $0.outNodeBase }
            )
            mEnc.setComputePipelineState(topkPSO)
            mEnc.setBuffer(baseLogitsBF16Buf, offset: 0, index: 0)
            mEnc.setBuffer(markovLatentBatchBuf, offset: 0, index: 1)
            mEnc.setBuffer(foldWBuf, offset: 0, index: 2)
            mEnc.setBuffer(foldBBuf, offset: 0, index: 3)
            mEnc.setBuffer(topkPartValBuf, offset: 0, index: 4)
            mEnc.setBuffer(topkPartIdxBuf, offset: 0, index: 5)
            mEnc.setBytes(&tp, length: MemoryLayout<TopKP>.stride, index: 6)
            mEnc.dispatchThreadgroups(MTLSize(width: partCount, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

            mEnc.setComputePipelineState(pTopKFinal)
            mEnc.setBuffer(topkPartValBuf, offset: 0, index: 0)
            mEnc.setBuffer(topkPartIdxBuf, offset: 0, index: 1)
            mEnc.setBuffer(treeTokBuf, offset: 0, index: 2)
            mEnc.setBytes(&tp, length: MemoryLayout<TopKP>.stride, index: 3)
            mEnc.dispatchThreadgroups(MTLSize(width: P, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
    }

    func draftTree(
        anchor: Int, template: DSparkTreeTemplate, profiler: QWInkling2Profiler? = nil
    ) throws -> [Int] {
        let hostStartNs = profiler?.beginHostTimer() ?? 0
        let cbOpt = (profiler == nil) ? (hal.queue.makeCommandBufferWithUnretainedReferences() ?? hal.queue.makeCommandBuffer())
                                      : hal.queue.makeCommandBuffer()
        guard let cb = cbOpt else {
            throw QWInkling2Error.badState("failed to create DSpark tree draft CB")
        }
        var encOpt: (any MTLComputeCommandEncoder)? = nil
        try encodeDraftTree(anchor: anchor, template: template, on: cb, encoder: &encOpt, profiler: profiler)
        profiler?.endActiveStage(encoder: &encOpt)
        encOpt?.endEncoding()
        profiler?.endHostTimer(hostStartNs)
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error {
            throw QWInkling2Error.badState("DSpark tree draft failed: \(err.localizedDescription)")
        }
        profiler?.finishCommandBuffer(cb, tokens: 0)
        let ptr = treeTokBuf.contents().assumingMemoryBound(to: Int32.self)
        return (0..<template.plan.nodesCount).map { Int(ptr[$0]) }
    }

    enum NextDraft: Equatable {
        case none
        case linear(Int)
        case tree(DSparkTreeTemplate)

        static func == (lhs: NextDraft, rhs: NextDraft) -> Bool {
            switch (lhs, rhs) {
            case (.none, .none): return true
            case (.linear(let a), .linear(let b)): return a == b
            case (.tree(let a), .tree(let b)): return a.mode == b.mode && a.plan.nodesCount == b.plan.nodesCount
            default: return false
            }
        }
    }

    private func readNextDraftTokens(_ nextDraft: NextDraft) -> [Int] {
        switch nextDraft {
        case .none:
            return []
        case .linear(let cnt):
            guard cnt > 0, cnt < config.blockSize else { return [] }
            let ptr = draftTokBuf.contents().assumingMemoryBound(to: Int32.self)
            return (1...cnt).map { Int(ptr[$0]) }
        case .tree(let tpl):
            let ptr = treeTokBuf.contents().assumingMemoryBound(to: Int32.self)
            return (0..<tpl.plan.nodesCount).map { Int(ptr[$0]) }
        }
    }

    private func encodeNextDraft(
        anchor: Int,
        nextDraft: NextDraft,
        on cb: any MTLCommandBuffer,
        encoder encOpt: inout (any MTLComputeCommandEncoder)?,
        profiler: QWInkling2Profiler?
    ) throws {
        switch nextDraft {
        case .none:
            break
        case .linear(let cnt):
            if cnt > 0, cnt < config.blockSize {
                try encodeDraftBlock(anchor: anchor, count: cnt, on: cb, encoder: &encOpt, profiler: profiler)
            }
        case .tree(let tpl):
            try encodeDraftTree(anchor: anchor, template: tpl, on: cb, encoder: &encOpt, profiler: profiler)
        }
    }

    /// Unified 1-CB commit (tree/linear conv + KV compaction) + tap ingest + next draft (linear or tree).
    func finishUnifiedSpeculativeRound(
        session: QWInkling2Session,
        oldPos: Int,
        rowStride: Int,
        keptNodes: [Int],
        ingestTaps: Bool,
        nextAnchor: Int,
        nextDraft: NextDraft,
        profiler: QWInkling2Profiler? = nil,
        allowAsyncNoDraft: Bool = false
    ) throws -> (drafts: [Int], pendingCB: (any MTLCommandBuffer)?) {
        let hostStartNs = profiler?.beginHostTimer() ?? 0
        guard let cb = makeDSparkCB(profiler: profiler) else {
            throw QWInkling2Error.badState("failed to create unified round CB")
        }
        var encOpt: (any MTLComputeCommandEncoder)? = nil
        let commitEnc = try beginStage(.sconv, on: cb, encoder: &encOpt, profiler: profiler)
        encodeSConvVerifyCommitTree(commitEnc, session: session, keptNodes: keptNodes)
        encodeTreeKVCompact(commitEnc, session: session, oldPos: oldPos, keptNodes: keptNodes)
        if ingestTaps {
            let isContiguous = keptNodes.enumerated().allSatisfy { $0.offset == $0.element }
            try encodeIngestContextTaps(
                count: keptNodes.count, posOffset: oldPos, rowStride: rowStride,
                rowMap: isContiguous ? nil : keptNodes, on: cb, encoder: &encOpt, profiler: profiler
            )
            try encodeNextDraft(
                anchor: nextAnchor, nextDraft: nextDraft,
                on: cb, encoder: &encOpt, profiler: profiler
            )
        }
        profiler?.endActiveStage(encoder: &encOpt)
        encOpt?.endEncoding()
        profiler?.endHostTimer(hostStartNs)
        cb.commit()
        if allowAsyncNoDraft && nextDraft == .none && profiler == nil {
            return ([], cb)
        }
        cb.waitUntilCompleted()
        if let err = cb.error {
            throw QWInkling2Error.badState("DSpark unified round CB failed: \(err.localizedDescription)")
        }
        profiler?.finishCommandBuffer(cb, tokens: ingestTaps ? keptNodes.count : 0)
        guard ingestTaps else { return ([], nil) }
        return (readNextDraftTokens(nextDraft), nil)
    }

    /// Fuses context tap ingest and next linear/tree GPU draft into a single command buffer.
    func ingestAndDraft(
        count: Int,
        posOffset: Int,
        rowStride: Int,
        anchor: Int,
        nextDraft: NextDraft,
        profiler: QWInkling2Profiler? = nil
    ) throws -> [Int] {
        let hostStartNs = profiler?.beginHostTimer() ?? 0
        guard let cb = makeDSparkCB(profiler: profiler) else {
            throw QWInkling2Error.badState("failed to create DSpark ingest+draft CB")
        }
        var encOpt: (any MTLComputeCommandEncoder)? = nil
        if count > 0 {
            try encodeIngestContextTaps(
                count: count, posOffset: posOffset, rowStride: rowStride,
                on: cb, encoder: &encOpt, profiler: profiler
            )
        }
        try encodeNextDraft(
            anchor: anchor, nextDraft: nextDraft,
            on: cb, encoder: &encOpt, profiler: profiler
        )
        profiler?.endActiveStage(encoder: &encOpt)
        encOpt?.endEncoding()
        profiler?.endHostTimer(hostStartNs)
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error {
            throw QWInkling2Error.badState("DSpark ingest+draft CB failed: \(err.localizedDescription)")
        }
        profiler?.finishCommandBuffer(cb, tokens: count)
        return readNextDraftTokens(nextDraft)
    }
}
