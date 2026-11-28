// DriverSessions.swift — QWEngineDriver DriverSessions family (split from QWEngineDriver.swift; same-module extension).
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
import QWMLP
import QWHead
import QWMTP
import QWQuant

extension QWEngineDriver {
    public func makeSession(promptTokens: Int) throws -> QWEngineSession {
        if standardState == nil {
            throw QWEngineDriverError.state("runtime manages sessions via generate(request:)")
        }
        guard flight.try() else { throw QWEngineDriverError.busy }
        defer { flight.unlock() }
        try requestGate.validate()
        return try admitSession(promptTokens: promptTokens)
    }

    public func makeStochasticSession(promptTokens: Int, config: QWSamplerConfig) throws -> QWEngineSession {
        if standardState == nil {
            throw QWEngineDriverError.state("runtime manages sessions via generate(request:)")
        }
        let owner = try QWStochasticSession(geometry: geometry, config: config)
        guard flight.try() else { throw QWEngineDriverError.busy }
        defer { flight.unlock() }
        try requestGate.validate()
        let session = try admitSession(promptTokens: promptTokens)
        session.stochastic = owner
        return session
    }

    func makeSession(adopting prefix: QWEngineSharedPrefix, promptTokens: Int) throws -> QWEngineSession {
        guard flight.try() else { throw QWEngineDriverError.busy }
        defer { flight.unlock() }
        try requestGate.validate()
        guard prefix.position == prefix.trunk.tokenCount, promptTokens >= prefix.position,
              (prefix.mtp != nil) == (mtpCache != nil),
              prefix.mtp?.tokenCount == (mtpCache == nil ? nil : prefix.roundRecord.mtpPhysicalRows) else {
            throw QWEngineDriverError.state("shared prefix admission boundary or cache roles invalid")
        }
        return try admitSession(promptTokens: promptTokens, prefix: prefix)
    }

    func admitSession(promptTokens: Int, prefix: QWEngineSharedPrefix? = nil) throws -> QWEngineSession {
        let slot = try admission.acquireSlot(dflashPreferred: true)
        if prefix == nil {
            memset(slot.recSlab.region.buffer.contents()
                   .advanced(by: slot.recSlab.region.base + slot.recSlab.offset),
               0, slot.recSlab.bytes)
            memset(slot.ringSlab.region.buffer.contents()
                   .advanced(by: slot.ringSlab.region.base + slot.ringSlab.offset),
               0, slot.ringSlab.bytes)
            if let hidden = slot.trunkHidden {
                memset(hidden.region.pointer.advanced(by: hidden.offset), 0, hidden.bytes)
            }
            if let dSlot = slot.dflashSlot {
                memset(dSlot.kSlab.region.pointer.advanced(by: dSlot.kSlab.offset), 0, dSlot.kSlab.bytes)
                memset(dSlot.vSlab.region.pointer.advanced(by: dSlot.vSlab.offset), 0, dSlot.vSlab.bytes)
            }
            if let dsSlot = slot.dsparkSlot {
                memset(dsSlot.ringBuffer.contents(), 0, dsSlot.ringBytes)
            }
        }
        var trunkHandle: QWKVHandle?, mtpHandle: QWKVHandle?
        do {
            let stochasticOwner = try prefix.flatMap { try $0.state.makeStochasticOwner(geometry: geometry) }
            let h = try prefix.map { try kvCache.allocate(adopting: $0.trunk, promptTokens: promptTokens) }
                ?? kvCache.allocate(promptTokens: promptTokens)
            trunkHandle = h
            let mtpH: QWKVHandle?
            if let prefix, let shared = prefix.mtp, let mtpCache {
                mtpH = try mtpCache.allocate(adopting: shared, promptTokens: promptTokens)
            } else {
                mtpH = try mtpCache?.allocate(promptTokens: promptTokens)
            }
            mtpHandle = mtpH
            let s = QWEngineSession(kv: h, mtpKV: mtpH, rec: slot.rec, ring: slot.ring,
                                    recSlab: slot.recSlab, ringSlab: slot.ringSlab,
                                    trunkHidden: slot.trunkHidden, dflashSlot: slot.dflashSlot,
                                    dsparkSlot: slot.dsparkSlot,
                                    isSparseMoE: geometry.numExperts > 1)
            s.attnKVViews = try kvCache.layerViewsAll(h, layers: attnLayers)
            if let mtpCache, let mtpH {
                s.mtpKVViews = [try mtpCache.layerViews(mtpH, layer: 0)]
            }
            if let prefix {
                try prefix.state.validateDestination(s, requiresRoundMetadata: mtpCache != nil)
                try validateMLAIndexerState(prefix.state.mlaIndexerData, position: prefix.position)
                prefix.state.publish(into: s, stochastic: stochasticOwner)
                restoreMLAIndexerState(prefix.state.mlaIndexerData, position: prefix.position, into: s)
                if isWeightFreeSpeculation {
                    s.roundRecord.consumedCount = s.position
                    s.roundRecord.absolutePosition = s.position
                    s.roundRecord.outputCount = s.generated
                    s.trunkHiddenValid = s.trunkHidden != nil && s.position > 0
                }
            } else {
                s.dflashCacheOffset = 0
                s.dsparkCacheOffset = 0
            }
            return s
        } catch {
            if let mtpHandle { mtpCache?.finish(mtpHandle) }
            if let trunkHandle { kvCache.finish(trunkHandle) }
            admission.releaseSlot(slot)
            throw error
        }
    }

    /// Release the session's KV pages (every page returns to its pool,
    /// seq id recycled — the soak leak law) and GDN slot. Order matters:
    /// KV finish first (frees pages while the slot is still live), then
    public func finishSession(_ session: QWEngineSession) {
        flight.lock()
        defer { flight.unlock() }
        guard !session.sessionFinished else { return }
        session.sessionFinished = true
        session.mlaIndexerStash = nil
        session.mlaLiveStash = nil
        session.mlaBatchState = nil
        if let cb = session.lastCommitCB {
            cb.waitUntilCompleted()
            session.lastCommitCB = nil
            _ = syncCommitFailure(for: session, cb: cb)
        }
        if hcaOperator?.stateOwner === session { hcaOperator?.stateOwner = nil }
        if mlaOperator?.stateOwner === session { mlaOperator?.stateOwner = nil }
        if lsaOperator?.stateOwner === session { lsaOperator?.stateOwner = nil }
        if mbsaOperator?.stateOwner === session { mbsaOperator?.stateOwner = nil }
        session.stochastic?.stream.cancelActiveRound()
        if let owner = session.stochastic { session.roundRecord.samplerDrawIndex = owner.stream.drawsConsumed }
        if verifiedSession === session {
            verifiedSession = nil
            verifiedInputs.removeAll(keepingCapacity: true)
            preparedGreedyLease = nil
            preparedStochasticWitness = nil
            pendingMTPProfile = nil
            pendingMTPAdaptiveEntropy = nil
            lastRoundUsedPLD = false
            lastDraftedPairs = 0
            session.verificationPending = false
        }
        if let mtpCache, let mtpH = session.mtpKV {
            mtpCache.finish(mtpH)
        }
        kvCache.finish(session.kv)
        admission.releaseSlot(GdnSlot(recSlab: session.recSlab,
                                    ringSlab: session.ringSlab,
                                    rec: session.rec, ring: session.ring,
                                    trunkHidden: session.trunkHidden,
                                    dflashSlot: session.dflashSlot,
                                    dsparkSlot: session.dsparkSlot))
    }

    /// Take an exact snapshot of session state (GDN rec + ring slabs, KV cursors, position).
    public func snapshotSession(_ session: QWEngineSession) throws -> QWEngineSessionSnapshot {
        flight.lock()
        defer { flight.unlock() }
        try requestGate.validate()
        let cb = session.lastCommitCB
        cb?.waitUntilCompleted()
        session.lastCommitCB = nil
        guard !syncCommitFailure(for: session, cb: cb) else {
            throw QWEngineDriverError.state("commit GPU failure before snapshotSession")
        }
        return try QWEngineSessionState.snapshot(
            session, cache: kvCache, mtpCache: mtpCache,
            sparseStateData: captureMLAIndexerState(session))
    }

    /// Roll back session state to an exact snapshot (destructive rewind).
    public func rollbackSession(_ session: QWEngineSession, to snapshot: QWEngineSessionSnapshot) throws {
        flight.lock()
        defer { flight.unlock() }

        try requestGate.validate()
        if let cb = session.lastCommitCB {
            cb.waitUntilCompleted()
            session.lastCommitCB = nil
            _ = syncCommitFailure(for: session, cb: cb)
        }
        try claimMLAIndexerState(session)
        try QWEngineSessionState.restore(session, to: snapshot, cache: kvCache,
                                        mtpCache: mtpCache, attnLayers: attnLayers)
        if hcaOperator != nil || mlaOperator != nil || lsaOperator != nil || mbsaOperator != nil {
            restoreMLAIndexerState(snapshot.sparseStateData, position: snapshot.position, into: session)
            mlaOperator?.truncatePosition(snapshot.position)
            hcaOperator?.truncatePosition(snapshot.position)
        }
        session.tokenRecycler.reset()
        if !session.tokenHistory.isEmpty {
            session.tokenRecycler.ingestSequence(session.tokenHistory)
        }
    }

    func capturePrivatePrefix(_ session: QWEngineSession, maxBytes: Int) throws -> QWEnginePrivatePrefix {
        guard flight.try() else { throw QWEngineDriverError.busy }
        defer { flight.unlock() }
        try requestGate.validate()
        guard verifiedSession == nil else { throw QWEngineDriverError.busy }
        let cb = session.lastCommitCB
        cb?.waitUntilCompleted()
        session.lastCommitCB = nil
        guard !syncCommitFailure(for: session, cb: cb) else {
            throw QWEngineDriverError.state("commit GPU failure before capturePrivatePrefix")
        }
        return try QWEngineSessionState.capturePrivatePrefix(session, cache: kvCache,
                                                            mtpCache: mtpCache, maxBytes: maxBytes,
                                                            mlaIndexerData: captureMLAIndexerState(session))
    }

    func captureSharedPrefix(_ session: QWEngineSession, maxBytes: Int,
                             permit: QWGenerationPermit? = nil) throws -> QWEngineSharedPrefix {
        guard flight.try() else { throw QWEngineDriverError.busy }
        defer { flight.unlock() }
        try requestGate.validate(permit)
        guard verifiedSession == nil else { throw QWEngineDriverError.busy }
        let cb = session.lastCommitCB
        cb?.waitUntilCompleted()
        session.lastCommitCB = nil
        guard !syncCommitFailure(for: session, cb: cb) else {
            throw QWEngineDriverError.state("commit GPU failure before captureSharedPrefix")
        }
        return try QWEngineSessionState.captureSharedPrefix(session, cache: kvCache,
                                                           mtpCache: mtpCache, maxBytes: maxBytes,
                                                           mlaIndexerData: captureMLAIndexerState(session))
    }

    func adoptPrivatePrefix(_ prefix: QWEnginePrivatePrefix, into session: QWEngineSession) throws {
        guard flight.try() else { throw QWEngineDriverError.busy }
        defer { flight.unlock() }
        try requestGate.validate()
        guard verifiedSession == nil else { throw QWEngineDriverError.busy }
        session.lastCommitCB?.waitUntilCompleted()
        session.lastCommitCB = nil
        try validateMLAIndexerState(prefix.mlaIndexerData, position: prefix.position)
        try QWEngineSessionState.adoptPrivatePrefix(geometry: geometry, prefix, into: session, cache: kvCache,
                                                    mtpCache: mtpCache, attnLayers: attnLayers)
        restoreMLAIndexerState(prefix.mlaIndexerData, position: prefix.position, into: session)
    }

    private static let qhmsMagic: UInt32 = 0x534D_4851 // "QHMS"
    private static let qhmsHeaderBytes = 16

    private static func packCompositeSparseState(sub: Data, hca: Data) -> Data {
        var out = Data(capacity: qhmsHeaderBytes + sub.count + hca.count)
        let hdr: [UInt32] = [qhmsMagic, 1, UInt32(sub.count), UInt32(hca.count)]
        hdr.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            out.append(contentsOf: raw)
        }
        out.append(sub)
        out.append(hca)
        return out
    }

    private static func unpackCompositeSparseState(_ data: Data) -> (sub: Data, hca: Data)? {
        guard data.count >= qhmsHeaderBytes else { return nil }
        var hdr = [UInt32](repeating: 0, count: 4)
        _ = hdr.withUnsafeMutableBytes { (dst: UnsafeMutableRawBufferPointer) in
            data.copyBytes(to: dst, from: 0..<qhmsHeaderBytes)
        }
        guard hdr[0] == qhmsMagic, hdr[1] == 1 else { return nil }
        let subLen = Int(hdr[2])
        let hcaLen = Int(hdr[3])
        guard qhmsHeaderBytes + subLen + hcaLen == data.count else { return nil }
        let s0 = qhmsHeaderBytes
        let s1 = s0 + subLen
        return (Data(data[s0..<s1]), Data(data[s1..<(s1 + hcaLen)]))
    }

    private func captureLiveCompositeState(position: Int) -> Data {
        let sub = mlaOperator?.captureIndexerState(position: position)
            ?? lsaOperator?.captureIndexerState(position: position)
            ?? Data()
        let hcaData = hcaOperator?.captureState(position: position) ?? Data()
        return Self.packCompositeSparseState(sub: sub, hca: hcaData)
    }

    func activeSparseIndexerStateBytes(position: Int) -> Int {
        if let hca = hcaOperator {
            let sub = mlaOperator?.indexerStateBytes(position: position)
                ?? lsaOperator?.indexerStateBytes(position: position)
                ?? 0
            return Self.qhmsHeaderBytes + sub + hca.stateBytes(position: position)
        }
        if let mla = mlaOperator { return mla.indexerStateBytes(position: position) }
        if let lsa = lsaOperator { return lsa.indexerStateBytes(position: position) }
        if let mbsa = mbsaOperator { return mbsa.indexerStateBytes(position: position) }
        return 0
    }

    /// MLA/LSA/HCA/MBSA pools live in the operator, so prefix capture copies them
    /// out and adoption writes them back; the session's GPU work must be done.
    func captureMLAIndexerState(_ session: QWEngineSession) -> Data? {
        if let hca = hcaOperator {
            if hca.stateOwner !== session,
               let stash = session.mlaIndexerStash, stash.position == session.position {
                return stash.data
            }
            return captureLiveCompositeState(position: session.position)
        }
        if let mla = mlaOperator {
            if let batchState = session.mlaBatchState {
                if mla.isDecoupledDSA {
                    return mla.captureLiveSessionStash(
                        from: batchState, trunkPosition: session.position,
                        mtpLayerIndex: mlaMTPLayerIndex ?? -1, mtpPosition: 0)
                }
                try? claimMLAIndexerState(session)
                return mla.captureIndexerState(position: session.position)
            }
            if mla.stateOwner !== session, let stash = session.mlaIndexerStash, stash.position == session.position {
                return stash.data
            }
            return mla.captureIndexerState(position: session.position)
        }
        if let lsa = lsaOperator {
            if lsa.stateOwner !== session, let stash = session.mlaIndexerStash, stash.position == session.position {
                return stash.data
            }
            return lsa.captureIndexerState(position: session.position)
        }
        if let mbsa = mbsaOperator {
            if mbsa.stateOwner !== session, let stash = session.mlaIndexerStash, stash.position == session.position {
                return stash.data
            }
            return mbsa.captureIndexerState(position: session.position)
        }
        return nil
    }

    func validateMLAIndexerState(_ data: Data?, position: Int) throws {
        if let hca = hcaOperator, position > 0 {
            guard let data, let parts = Self.unpackCompositeSparseState(data),
                  parts.hca.count == hca.stateBytes(position: position) else {
                throw QWEngineDriverError.state("prefix lacks HCA compressed/ring state for its position")
            }
            if let mla = mlaOperator {
                guard parts.sub.count == mla.indexerStateBytes(position: position) else {
                    throw QWEngineDriverError.state("prefix lacks MLA indexer state inside composite frame")
                }
            } else if let lsa = lsaOperator {
                guard lsa.isValidIndexerState(parts.sub, position: position) else {
                    throw QWEngineDriverError.state("prefix lacks LSA indexer state inside composite frame")
                }
            }
            return
        }
        if let mla = mlaOperator, position > 0 {
            guard data?.count == mla.indexerStateBytes(position: position) else {
                throw QWEngineDriverError.state("prefix lacks MLA indexer state for its position")
            }
        } else if let lsa = lsaOperator, position > 0 {
            guard lsa.isValidIndexerState(data, position: position) else {
                throw QWEngineDriverError.state("prefix lacks LSA indexer state for its position")
            }
        } else if let mbsa = mbsaOperator, position > 0 {
            guard data?.count == mbsa.indexerStateBytes(position: position) else {
                throw QWEngineDriverError.state("prefix lacks MBSA indexer state for its position")
            }
        }
    }

    func restoreMLAIndexerState(_ data: Data?, position: Int, into session: QWEngineSession) {
        if let hca = hcaOperator {
            try? claimMLAIndexerState(session, restoring: false)
            if position <= 0 {
                mlaOperator?.truncatePosition(0)
                hca.truncatePosition(0)
                return
            }
            guard let data, let parts = Self.unpackCompositeSparseState(data) else { return }
            if let mla = mlaOperator { _ = mla.restoreIndexerState(parts.sub, position: position) }
            if let lsa = lsaOperator { _ = lsa.restoreIndexerState(parts.sub, position: position) }
            _ = hca.restoreState(parts.hca, position: position)
            return
        }
        if let mla = mlaOperator {
            try? claimMLAIndexerState(session, restoring: false)
            if position <= 0 { mla.truncatePosition(0); return }
            guard let data else { return }
            _ = mla.restoreIndexerState(data, position: position)
        } else if let lsa = lsaOperator {
            try? claimMLAIndexerState(session, restoring: false)
            guard position > 0, let data else { return }
            _ = lsa.restoreIndexerState(data, position: position)
        } else if let mbsa = mbsaOperator {
            try? claimMLAIndexerState(session, restoring: false)
            guard position > 0, let data else { return }
            _ = mbsa.restoreIndexerState(data, position: position)
        }
    }

    /// The operator holds one session's indexer/HCA pools: switching sessions
    /// stashes the owner's pools and restores the claimant's. Call under the
    /// flight lock, before encoding any of the session's MLA/LSA/HCA work.
    func claimMLAIndexerState(_ session: QWEngineSession, restoring: Bool = true) throws {
        if let hca = hcaOperator {
            let needsClaim = (hca.stateOwner !== session)
                || (mlaOperator != nil && mlaOperator!.stateOwner !== session)
                || (lsaOperator != nil && lsaOperator!.stateOwner !== session)
            guard needsClaim else { return }
            if let prev = (hca.stateOwner ?? mlaOperator?.stateOwner ?? lsaOperator?.stateOwner) as? QWEngineSession,
               !prev.sessionFinished, prev.position > 0 {
                let prevCB = prev.lastCommitCB
                prevCB?.waitUntilCompleted()
                prev.lastCommitCB = nil
                if !syncCommitFailure(for: prev, cb: prevCB) {
                    prev.mlaIndexerStash = (captureLiveCompositeState(position: prev.position), prev.position)
                } else {
                    prev.mlaIndexerStash = nil
                }
            }
            hca.stateOwner = session
            mlaOperator?.stateOwner = session
            lsaOperator?.stateOwner = session
            defer { session.mlaIndexerStash = nil }
            guard restoring, session.position > 0 else {
                mlaOperator?.truncatePosition(session.position)
                hca.truncatePosition(session.position)
                return
            }
            guard let stash = session.mlaIndexerStash, stash.position == session.position,
                  let parts = Self.unpackCompositeSparseState(stash.data),
                  hca.restoreState(parts.hca, position: stash.position) else {
                session.stateFailed = true
                throw QWEngineDriverError.state("HCA state was not stashed at the session position")
            }
            if let mla = mlaOperator, !mla.restoreIndexerState(parts.sub, position: stash.position) {
                session.stateFailed = true
                throw QWEngineDriverError.state("MLA indexer state was not stashed inside composite frame")
            }
            if let lsa = lsaOperator, !lsa.restoreIndexerState(parts.sub, position: stash.position) {
                session.stateFailed = true
                throw QWEngineDriverError.state("LSA indexer state was not stashed inside composite frame")
            }
            return
        }
        if let mla = mlaOperator {
            guard mla.stateOwner !== session else { return }
            let mtpLayer = (mtpDraftEncoder != nil || echoDrafter != nil) && attnLayers.count < mla.layers.count
                ? attnLayers.count : nil
            if let prev = mla.stateOwner as? QWEngineSession, !prev.sessionFinished, prev.position > 0 {
                let prevCB = prev.lastCommitCB
                prevCB?.waitUntilCompleted()
                prev.lastCommitCB = nil
                if !syncCommitFailure(for: prev, cb: prevCB) {
                    let mtpPos = prev.roundRecord.mtpPhysicalRows
                    prev.mlaLiveStash = (
                        mla.captureLiveSessionStash(trunkPosition: prev.position, mtpLayerIndex: mtpLayer, mtpPosition: mtpPos),
                        prev.position, mtpPos
                    )
                } else {
                    prev.mlaLiveStash = nil
                }
            }
            mla.stateOwner = session
            defer {
                session.mlaIndexerStash = nil
                session.mlaLiveStash = nil
            }
            guard restoring, session.position > 0 else {
                session.mlaBatchState = nil
                mla.truncatePosition(session.position)
                return
            }
            if let batchState = session.mlaBatchState {
                let mtpPos = session.roundRecord.mtpPhysicalRows
                let liveData = mla.captureLiveSessionStash(
                    from: batchState, trunkPosition: session.position,
                    mtpLayerIndex: mtpLayer, mtpPosition: mtpPos)
                session.mlaBatchState = nil
                guard mla.restoreLiveSessionStash(
                    liveData, trunkPosition: session.position,
                    mtpLayerIndex: mtpLayer, mtpPosition: mtpPos) else {
                    session.stateFailed = true
                    throw QWEngineDriverError.state("failed to reclaim MLA batch session state")
                }
                return
            }
            if let live = session.mlaLiveStash,
               live.trunkPosition == session.position,
               live.mtpPosition == session.roundRecord.mtpPhysicalRows,
               mla.restoreLiveSessionStash(live.data, trunkPosition: live.trunkPosition,
                                           mtpLayerIndex: mtpLayer, mtpPosition: live.mtpPosition) {
                return
            }
            guard let stash = session.mlaIndexerStash, stash.position == session.position,
                  mla.restoreIndexerState(stash.data, position: stash.position) else {
                session.stateFailed = true
                throw QWEngineDriverError.state("MLA indexer state was not stashed at the session position")
            }
        } else if let lsa = lsaOperator {
            guard lsa.stateOwner !== session else { return }
            if let prev = lsa.stateOwner as? QWEngineSession,
               !prev.sessionFinished, !prev.batchedDecodeTouched, prev.position > 0 {
                let prevCB = prev.lastCommitCB
                prevCB?.waitUntilCompleted()
                prev.lastCommitCB = nil
                if !syncCommitFailure(for: prev, cb: prevCB) {
                    prev.mlaIndexerStash = (lsa.captureIndexerState(position: prev.position), prev.position)
                } else {
                    prev.mlaIndexerStash = nil
                }
            }
            lsa.stateOwner = session
            defer { session.mlaIndexerStash = nil }
            if session.batchedDecodeTouched && session.position < lsaConfig.denseBypassThreshold {
                lsa.stateOwner = nil
                return
            }
            guard restoring, session.position > 0 else { return }
            guard let stash = session.mlaIndexerStash, stash.position == session.position,
                  lsa.restoreIndexerState(stash.data, position: stash.position) else {
                session.stateFailed = true
                throw QWEngineDriverError.state("LSA indexer state was not stashed at the session position")
            }
        } else if let mbsa = mbsaOperator {
            guard mbsa.stateOwner !== session else { return }
            if let prev = mbsa.stateOwner as? QWEngineSession, !prev.sessionFinished, prev.position > 0 {
                let prevCB = prev.lastCommitCB
                prevCB?.waitUntilCompleted()
                prev.lastCommitCB = nil
                if !syncCommitFailure(for: prev, cb: prevCB) {
                    prev.mlaIndexerStash = (mbsa.captureIndexerState(position: prev.position), prev.position)
                } else {
                    prev.mlaIndexerStash = nil
                }
            }
            mbsa.stateOwner = session
            defer { session.mlaIndexerStash = nil }
            guard restoring, session.position > 0 else { return }
            guard let stash = session.mlaIndexerStash, stash.position == session.position,
                  mbsa.restoreIndexerState(stash.data, position: stash.position) else {
                session.stateFailed = true
                throw QWEngineDriverError.state("MBSA indexer state was not stashed at the session position")
            }
        }
    }
}
