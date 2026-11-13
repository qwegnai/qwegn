import Foundation
import QWKV

/// Diagnostic snapshot. Speculative rounds require preallocated GPU checkpoints,
/// not these allocating CPU copies. Any KV rewind invalidates earlier snapshots;
/// take a fresh snapshot after restoring. KV prefix bytes must not be overwritten.
public struct QWEngineSessionSnapshot: Sendable {
    let source: QWEngineSession
    let kvGeneration: UInt64
    let mtpKVGeneration: UInt64?
    public let position: Int
    public let generated: Int
    public let roundRecord: QWRoundRecord
    public let kvWritten: Int
    public let mtpKVWritten: Int?
    public let recData: Data
    public let ringData: Data
    public let trunkHiddenData: Data?
    public let trunkHiddenValid: Bool
    let dflashKData: Data?
    let dflashVData: Data?
    let dflashCacheOffset: Int
    let sparseStateData: Data?
    let stochastic: QWStochasticState?

    init(source: QWEngineSession, kvGeneration: UInt64, mtpKVGeneration: UInt64?,
         position: Int, generated: Int, roundRecord: QWRoundRecord,
         kvWritten: Int, mtpKVWritten: Int?, recData: Data, ringData: Data,
         trunkHiddenData: Data?, trunkHiddenValid: Bool,
         dflashKData: Data? = nil, dflashVData: Data? = nil, dflashCacheOffset: Int = 0,
         sparseStateData: Data? = nil,
         stochastic: QWStochasticState? = nil) {
        self.source = source
        self.kvGeneration = kvGeneration
        self.mtpKVGeneration = mtpKVGeneration
        self.position = position
        self.generated = generated
        self.roundRecord = roundRecord
        self.kvWritten = kvWritten
        self.mtpKVWritten = mtpKVWritten
        self.recData = recData
        self.ringData = ringData
        self.trunkHiddenData = trunkHiddenData
        self.trunkHiddenValid = trunkHiddenValid
        self.dflashKData = dflashKData
        self.dflashVData = dflashVData
        self.dflashCacheOffset = dflashCacheOffset
        self.sparseStateData = sparseStateData
        self.stochastic = stochastic
    }
}

/// The caller holds the driver's flight lock and has completed all GPU writes.
enum QWEngineSessionState {
    static func snapshot(_ session: QWEngineSession, cache: QWKVCache,
                         mtpCache: QWKVCache?, sparseStateData: Data? = nil) throws -> QWEngineSessionSnapshot {
        try validate(session, cache: cache, mtpCache: mtpCache)
        return QWEngineSessionSnapshot(
            source: session, kvGeneration: try cache.rollbackGeneration(session.kv),
            mtpKVGeneration: try session.mtpKV.flatMap { try mtpCache?.rollbackGeneration($0) },
            position: session.position, generated: session.generated,
            roundRecord: session.roundRecord, kvWritten: cache.written(session.kv),
            mtpKVWritten: session.mtpKV.flatMap { mtpCache?.written($0) },
            recData: Data(bytes: session.recSlab.region.pointer.advanced(by: session.recSlab.offset),
                          count: session.recSlab.bytes),
            ringData: Data(bytes: session.ringSlab.region.pointer.advanced(by: session.ringSlab.offset),
                           count: session.ringSlab.bytes),
            trunkHiddenData: session.trunkHidden.map {
                Data(bytes: $0.region.pointer.advanced(by: $0.offset), count: $0.bytes)
            }, trunkHiddenValid: session.trunkHiddenValid,
            dflashKData: session.dflashSlot.map {
                Data(bytes: $0.kSlab.region.pointer.advanced(by: $0.kSlab.offset), count: $0.kSlab.bytes)
            },
            dflashVData: session.dflashSlot.map {
                Data(bytes: $0.vSlab.region.pointer.advanced(by: $0.vSlab.offset), count: $0.vSlab.bytes)
            },
            dflashCacheOffset: session.dflashCacheOffset,
            sparseStateData: sparseStateData,
            stochastic: try session.stochastic?.frozenState())
    }

    static func restore(_ session: QWEngineSession, to snapshot: QWEngineSessionSnapshot,
                        cache: QWKVCache, mtpCache: QWKVCache?, attnLayers: [Int]) throws {
        try validate(session, cache: cache, mtpCache: mtpCache)
        guard snapshot.kvGeneration == (try cache.rollbackGeneration(session.kv)),
              snapshot.mtpKVGeneration == (try session.mtpKV.flatMap { try mtpCache?.rollbackGeneration($0) }) else {
            throw QWEngineDriverError.state("snapshot invalidated by a KV rewind")
        }
        if let state = snapshot.stochastic {
            guard let owner = session.stochastic, state.config == owner.config,
                  state.stream.seed == owner.stream.seed,
                  state.stream.drawsConsumed <= owner.stream.drawsConsumed else {
                throw QWEngineDriverError.state("snapshot RNG identity or counter is invalid")
            }
        } else if session.stochastic != nil {
            throw QWEngineDriverError.state("cannot restore an unconfigured RNG into a sampled session")
        }
        guard snapshot.source === session,
              snapshot.recData.count == session.recSlab.bytes,
              snapshot.ringData.count == session.ringSlab.bytes,
              snapshot.trunkHiddenData?.count == session.trunkHidden?.bytes,
              !snapshot.trunkHiddenValid || snapshot.trunkHiddenData != nil,
              snapshot.position >= 0, snapshot.generated >= 0,
              snapshot.kvWritten >= 0, snapshot.kvWritten <= cache.written(session.kv),
              (snapshot.mtpKVWritten != nil) == (session.mtpKV != nil) else {
            throw QWEngineDriverError.state("snapshot identity, shape, or KV boundary is invalid")
        }
        if let mtpKV = session.mtpKV, let mtpCache, let written = snapshot.mtpKVWritten {
            guard written >= 0, written <= mtpCache.written(mtpKV) else {
                throw QWEngineDriverError.state("snapshot MTP boundary is ahead of live KV")
            }
            _ = try mtpCache.layerViews(mtpKV, layer: 0)
        }
        // Validate every view before either cache or state bytes change.
        for layer in attnLayers { _ = try cache.layerViews(session.kv, layer: layer) }
        try cache.rollback(session.kv, by: cache.written(session.kv) - snapshot.kvWritten)
        session.attnKVViews = try cache.layerViewsAll(session.kv, layers: attnLayers)
        if let mtpKV = session.mtpKV, let mtpCache, let written = snapshot.mtpKVWritten {
            try mtpCache.rollback(mtpKV, by: mtpCache.written(mtpKV) - written)
            session.mtpKVViews = [try mtpCache.layerViews(mtpKV, layer: 0)]
        }
        snapshot.recData.withUnsafeBytes {
            session.recSlab.region.pointer.advanced(by: session.recSlab.offset)
                .copyMemory(from: $0.baseAddress!, byteCount: $0.count)
        }
        snapshot.ringData.withUnsafeBytes {
            session.ringSlab.region.pointer.advanced(by: session.ringSlab.offset)
                .copyMemory(from: $0.baseAddress!, byteCount: $0.count)
        }
        session.position = snapshot.position
        session.generated = snapshot.generated
        session.roundRecord = snapshot.roundRecord
        if let owner = session.stochastic { session.roundRecord.samplerDrawIndex = owner.stream.drawsConsumed }
        if let hidden = session.trunkHidden, let data = snapshot.trunkHiddenData {
            data.withUnsafeBytes {
                hidden.region.pointer.advanced(by: hidden.offset).copyMemory(from: $0.baseAddress!, byteCount: $0.count)
            }
        }
        if let dSlot = session.dflashSlot, let kData = snapshot.dflashKData, let vData = snapshot.dflashVData,
           kData.count == dSlot.kSlab.bytes, vData.count == dSlot.vSlab.bytes {
            kData.withUnsafeBytes {
                dSlot.kSlab.region.pointer.advanced(by: dSlot.kSlab.offset).copyMemory(from: $0.baseAddress!, byteCount: $0.count)
            }
            vData.withUnsafeBytes {
                dSlot.vSlab.region.pointer.advanced(by: dSlot.vSlab.offset).copyMemory(from: $0.baseAddress!, byteCount: $0.count)
            }
        }
        session.dflashCacheOffset = snapshot.dflashCacheOffset
        session.trunkHiddenValid = snapshot.trunkHiddenValid
    }

    static func validate(_ session: QWEngineSession, cache: QWKVCache,
                                 mtpCache: QWKVCache?) throws {
        guard !session.sessionFinished, !session.stateFailed, !session.verificationPending,
              (session.mtpKV != nil) == (mtpCache != nil),
              session.recSlab.bytes > 0, session.ringSlab.bytes > 0 else {
            throw QWEngineDriverError.state("session is finished or has inconsistent state storage")
        }
        try cache.validate(session.kv)
        if let mtpKV = session.mtpKV, let mtpCache { try mtpCache.validate(mtpKV) }
        if let owner = session.stochastic {
            guard !owner.stream.isRoundActive, session.roundRecord.samplerDrawIndex == owner.stream.drawsConsumed else {
                throw QWEngineDriverError.state("session RNG boundary does not match its owned stream")
            }
        }
    }
}
