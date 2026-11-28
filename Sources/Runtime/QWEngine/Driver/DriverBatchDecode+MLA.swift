// DriverBatchDecode+MLA.swift — Per-session MLA indexer and k_pe pool state
// management across B=2..8 batched decode waves and single-slot transitions.
import Foundation
import Metal
import QWAttn
import QWCore

extension QWEngineDriver {
    var mlaMTPLayerIndex: Int? {
        guard let mla = mlaOperator else { return nil }
        return (mtpDraftEncoder != nil || echoDrafter != nil) && attnLayers.count < mla.layers.count
            ? attnLayers.count : nil
    }

    func prepareMLASessionsForBatchStep(_ sessions: [QWEngineSession]) throws {
        guard let mla = mlaOperator else { return }
        let mtpLayer = mlaMTPLayerIndex
        if let prev = mla.stateOwner as? QWEngineSession {
            let prevCB = prev.lastCommitCB
            prevCB?.waitUntilCompleted()
            prev.lastCommitCB = nil
            if !prev.sessionFinished, prev.position > 0, prev.mlaBatchState == nil {
                if !syncCommitFailure(for: prev, cb: prevCB) {
                    let mtpPos = prev.roundRecord.mtpPhysicalRows
                    prev.mlaLiveStash = (
                        mla.captureLiveSessionStash(
                            trunkPosition: prev.position, mtpLayerIndex: mtpLayer, mtpPosition: mtpPos),
                        prev.position, mtpPos
                    )
                } else {
                    prev.mlaLiveStash = nil
                }
            }
            mla.stateOwner = nil
        }
        for sess in sessions {
            let neededPos = sess.position + 1
            if let existing = sess.mlaBatchState {
                let limit = mla.isDecoupledDSA ? existing.maxBlocks : (existing.maxBlocks * 4)
                if neededPos <= limit {
                    continue
                }
                let mtpPos = sess.roundRecord.mtpPhysicalRows
                let saved = mla.captureLiveSessionStash(
                    from: existing, trunkPosition: sess.position,
                    mtpLayerIndex: mtpLayer, mtpPosition: mtpPos)
                let newCap = min(mla.maxContext, max(neededPos + 1024, existing.maxContext * 2))
                let grown = try mla.makeSessionState(maxContext: newCap)
                guard mla.restoreLiveSessionStash(
                    saved, into: grown, trunkPosition: sess.position,
                    mtpLayerIndex: mtpLayer, mtpPosition: mtpPos) else {
                    sess.stateFailed = true
                    throw QWEngineDriverError.state("failed to grow MLA batch session state")
                }
                sess.mlaBatchState = grown
                continue
            }
            let initCap = min(mla.maxContext, max(4096, sess.kv.tokenCapacity + 1024, neededPos * 2))
            let created = try mla.makeSessionState(maxContext: initCap)
            if sess.position > 0 {
                if let live = sess.mlaLiveStash,
                   live.trunkPosition == sess.position,
                   mla.restoreLiveSessionStash(
                       live.data, into: created, trunkPosition: live.trunkPosition,
                       mtpLayerIndex: mtpLayer, mtpPosition: live.mtpPosition) {
                    sess.mlaLiveStash = nil
                    sess.mlaIndexerStash = nil
                } else if let stash = sess.mlaIndexerStash,
                          stash.position == sess.position,
                          mla.restoreIndexerState(stash.data, position: stash.position) {
                    let liveData = mla.captureLiveSessionStash(
                        trunkPosition: sess.position, mtpLayerIndex: mtpLayer, mtpPosition: 0)
                    guard mla.restoreLiveSessionStash(
                        liveData, into: created, trunkPosition: sess.position,
                        mtpLayerIndex: mtpLayer, mtpPosition: 0) else {
                        sess.stateFailed = true
                        throw QWEngineDriverError.state("failed to restore MLA indexer stash into batch state")
                    }
                    sess.mlaIndexerStash = nil
                    sess.mlaLiveStash = nil
                } else {
                    sess.stateFailed = true
                    throw QWEngineDriverError.state("MLA session state missing before batched decode")
                }
            }
            sess.mlaBatchState = created
        }
    }
}
