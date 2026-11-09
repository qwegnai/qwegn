import Foundation
import QWCore
import QWKV

/// Diagnostic cold-path prefix. All bytes are immutable and all adopters use
/// private pages. No lookup key, eviction policy or zero-copy sharing is implied.
struct QWEnginePrivatePrefix: Sendable {
    let position: Int
    let generated: Int
    let roundRecord: QWRoundRecord
    let recData: Data
    let ringData: Data
    let trunkHiddenData: Data?
    let trunkHiddenValid: Bool
    let dflashKData: Data?
    let dflashVData: Data?
    let dflashCacheOffset: Int
    let stochastic: QWStochasticState?
    let trunk: QWKVPrivatePrefix
    let mtp: QWKVPrivatePrefix?
    let mlaIndexerData: Data?

    init(position: Int, generated: Int, roundRecord: QWRoundRecord,
         recData: Data, ringData: Data, trunkHiddenData: Data?, trunkHiddenValid: Bool,
         dflashKData: Data? = nil, dflashVData: Data? = nil, dflashCacheOffset: Int = 0,
         stochastic: QWStochasticState? = nil,
         trunk: QWKVPrivatePrefix, mtp: QWKVPrivatePrefix?, mlaIndexerData: Data? = nil) {
        self.position = position
        self.generated = generated
        self.roundRecord = roundRecord
        self.recData = recData
        self.ringData = ringData
        self.trunkHiddenData = trunkHiddenData
        self.trunkHiddenValid = trunkHiddenValid
        self.dflashKData = dflashKData
        self.dflashVData = dflashVData
        self.dflashCacheOffset = dflashCacheOffset
        self.stochastic = stochastic
        self.trunk = trunk
        self.mtp = mtp
        self.mlaIndexerData = mlaIndexerData
    }

    var memoryBytes: Int {
        recData.count + ringData.count + (trunkHiddenData?.count ?? 0) + (mlaIndexerData?.count ?? 0) +
            trunk.memoryBytes + (mtp?.memoryBytes ?? 0)
    }
}

extension QWEngineSessionState {
    static func capturePrivatePrefix(_ session: QWEngineSession, cache: QWKVCache,
                                     mtpCache: QWKVCache?, maxBytes: Int,
                                     mlaIndexerData: Data? = nil) throws -> QWEnginePrivatePrefix {
        try validate(session, cache: cache, mtpCache: mtpCache)
        let stateBytes = session.recSlab.bytes + session.ringSlab.bytes + (session.trunkHidden?.bytes ?? 0) +
            (mlaIndexerData?.count ?? 0)
        guard maxBytes >= stateBytes, session.generated >= 0,
              session.position == cache.written(session.kv),
              session.roundRecord.consumedCount == session.position,
              session.roundRecord.absolutePosition == session.position,
              session.roundRecord.outputCount == session.generated,
              session.mtpKV.flatMap({ mtpCache?.written($0) }) == (mtpCache == nil ? nil : session.roundRecord.mtpPhysicalRows) else {
            throw QWEngineDriverError.state("private prefix budget or session boundary invalid")
        }
        let trunk = try cache.capturePrivatePrefix(session.kv, maxBytes: maxBytes - stateBytes)
        let mtp = try session.mtpKV.flatMap { handle in
            try mtpCache?.capturePrivatePrefix(handle, maxBytes: maxBytes - stateBytes - trunk.memoryBytes)
        }
        let state = try snapshot(session, cache: cache, mtpCache: mtpCache)
        return QWEnginePrivatePrefix(position: state.position, generated: state.generated,
            roundRecord: state.roundRecord, recData: state.recData, ringData: state.ringData,
            trunkHiddenData: state.trunkHiddenData, trunkHiddenValid: state.trunkHiddenValid,
            dflashKData: state.dflashKData, dflashVData: state.dflashVData,
            dflashCacheOffset: state.dflashCacheOffset,
            stochastic: state.stochastic,
            trunk: trunk, mtp: mtp, mlaIndexerData: mlaIndexerData)
    }

    static func adoptPrivatePrefix(geometry: QWModelGeometry, _ prefix: QWEnginePrivatePrefix, into session: QWEngineSession,
                                   cache: QWKVCache, mtpCache: QWKVCache?, attnLayers: [Int]) throws {
        try validate(session, cache: cache, mtpCache: mtpCache)
        guard session.position == 0, session.generated == 0, session.stochastic == nil, prefix.generated >= 0,
              prefix.position == prefix.trunk.tokenCount,
              prefix.roundRecord.consumedCount == prefix.position,
              prefix.roundRecord.absolutePosition == prefix.position,
              prefix.roundRecord.outputCount == prefix.generated,
              prefix.recData.count == session.recSlab.bytes,
              prefix.ringData.count == session.ringSlab.bytes,
              prefix.trunkHiddenData?.count == session.trunkHidden?.bytes,
              !prefix.trunkHiddenValid || prefix.trunkHiddenData != nil,
              (prefix.mtp != nil) == (session.mtpKV != nil),
              prefix.mtp?.tokenCount == (mtpCache == nil ? nil : prefix.roundRecord.mtpPhysicalRows) else {
            throw QWEngineDriverError.state("private prefix state or destination boundary invalid")
        }
        try cache.validatePrivateAdoption(prefix.trunk, into: session.kv)
        if let mtp = prefix.mtp, let handle = session.mtpKV, let mtpCache {
            try mtpCache.validatePrivateAdoption(mtp, into: handle)
        }
        let views = try cache.layerViewsAll(session.kv, layers: attnLayers)
        let mtpViews = try session.mtpKV.flatMap { handle in try mtpCache?.layerViews(handle, layer: 0) }
        let state = QWEnginePrefixState(position: prefix.position, generated: prefix.generated,
            roundRecord: prefix.roundRecord, recData: prefix.recData, ringData: prefix.ringData,
            trunkHiddenData: prefix.trunkHiddenData, trunkHiddenValid: prefix.trunkHiddenValid,
            stochastic: prefix.stochastic,
            dflashKData: prefix.dflashKData, dflashVData: prefix.dflashVData,
            dflashCacheOffset: prefix.dflashCacheOffset,
            mlaIndexerData: prefix.mlaIndexerData)
        let owner = try state.makeStochasticOwner(geometry: geometry)
        do {
            try cache.adoptPrivatePrefix(prefix.trunk, into: session.kv)
            if let mtp = prefix.mtp, let handle = session.mtpKV, let mtpCache {
                try mtpCache.adoptPrivatePrefix(mtp, into: handle)
            }
        } catch {
            session.stateFailed = true
            throw error
        }
        state.publish(into: session, stochastic: owner)
        session.attnKVViews = views
        session.mtpKVViews = mtpViews.map { [$0] } ?? []
    }
}
