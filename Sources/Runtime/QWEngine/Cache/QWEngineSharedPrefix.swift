import QWCore
import Foundation
import QWKV

struct QWEnginePrefixState: Sendable {
    let position: Int
    let generated: Int
    let roundRecord: QWRoundRecord
    let recData: Data
    let ringData: Data
    let trunkHiddenData: Data?
    let trunkHiddenValid: Bool
    let stochastic: QWStochasticState?
    let dflashKData: Data?
    let dflashVData: Data?
    let dflashCacheOffset: Int
    /// MLA indexer pools and open-block ring rows (operator-owned, not in the slabs).
    let mlaIndexerData: Data?

    init(position: Int, generated: Int, roundRecord: QWRoundRecord, recData: Data, ringData: Data,
         trunkHiddenData: Data?, trunkHiddenValid: Bool, stochastic: QWStochasticState? = nil,
         dflashKData: Data? = nil, dflashVData: Data? = nil, dflashCacheOffset: Int = 0,
         mlaIndexerData: Data? = nil) {
        self.position = position
        self.generated = generated
        self.roundRecord = roundRecord
        self.recData = recData
        self.ringData = ringData
        self.trunkHiddenData = trunkHiddenData
        self.trunkHiddenValid = trunkHiddenValid
        self.stochastic = stochastic
        self.dflashKData = dflashKData
        self.dflashVData = dflashVData
        self.dflashCacheOffset = dflashCacheOffset
        self.mlaIndexerData = mlaIndexerData
    }

    init(_ snapshot: QWEngineSessionSnapshot, mlaIndexerData: Data? = nil) {
        position = snapshot.position
        generated = snapshot.generated
        roundRecord = snapshot.roundRecord
        recData = snapshot.recData
        ringData = snapshot.ringData
        trunkHiddenData = snapshot.trunkHiddenData
        trunkHiddenValid = snapshot.trunkHiddenValid
        stochastic = snapshot.stochastic
        dflashKData = snapshot.dflashKData
        dflashVData = snapshot.dflashVData
        dflashCacheOffset = snapshot.dflashCacheOffset
        self.mlaIndexerData = mlaIndexerData
    }

    var memoryBytes: Int {
        recData.count + ringData.count + (trunkHiddenData?.count ?? 0) + (mlaIndexerData?.count ?? 0)
    }

    func validateDestination(_ session: QWEngineSession, requiresRoundMetadata: Bool) throws {
        guard session.position == 0, session.generated == 0, session.stochastic == nil,
              position >= 0, generated >= 0,
              recData.count == session.recSlab.bytes, ringData.count == session.ringSlab.bytes,
              trunkHiddenData?.count == session.trunkHidden?.bytes,
              !trunkHiddenValid || trunkHiddenData != nil,
              !requiresRoundMetadata || (roundRecord.consumedCount == position &&
                  roundRecord.absolutePosition == position && roundRecord.outputCount == generated) else {
            throw QWEngineDriverError.state("prefix state or destination geometry is invalid")
        }
    }

    func makeStochasticOwner(geometry: QWModelGeometry) throws -> QWStochasticSession? {
        try stochastic.map { state in
            guard state.stream.drawsConsumed == roundRecord.samplerDrawIndex else {
                throw QWEngineDriverError.state("prefix RNG counter does not match its round record")
            }
            return try QWStochasticSession(geometry: geometry, config: state.config, cloning: state.stream)
        }
    }

    /// All geometry and cache operations must succeed before publication.
    func publish(into session: QWEngineSession, stochastic owner: QWStochasticSession? = nil) {
        recData.withUnsafeBytes {
            session.recSlab.region.pointer.advanced(by: session.recSlab.offset)
                .copyMemory(from: $0.baseAddress!, byteCount: $0.count)
        }
        ringData.withUnsafeBytes {
            session.ringSlab.region.pointer.advanced(by: session.ringSlab.offset)
                .copyMemory(from: $0.baseAddress!, byteCount: $0.count)
        }
        if let hidden = session.trunkHidden, let data = trunkHiddenData {
            data.withUnsafeBytes {
                hidden.region.pointer.advanced(by: hidden.offset)
                    .copyMemory(from: $0.baseAddress!, byteCount: $0.count)
            }
        }
        if let dSlot = session.dflashSlot, let kData = dflashKData, let vData = dflashVData,
           kData.count == dSlot.kSlab.bytes, vData.count == dSlot.vSlab.bytes {
            kData.withUnsafeBytes {
                dSlot.kSlab.region.pointer.advanced(by: dSlot.kSlab.offset)
                    .copyMemory(from: $0.baseAddress!, byteCount: $0.count)
            }
            vData.withUnsafeBytes {
                dSlot.vSlab.region.pointer.advanced(by: dSlot.vSlab.offset)
                    .copyMemory(from: $0.baseAddress!, byteCount: $0.count)
            }
        }
        session.position = position
        session.generated = generated
        session.roundRecord = roundRecord
        session.trunkHiddenValid = trunkHiddenValid
        session.dflashCacheOffset = dflashCacheOffset > 0 ? dflashCacheOffset : position
        session.stochastic = owner
    }
}

/// Same-driver immutable state and shared KV pages. Byte accounting charges
/// whole referenced pages, even when another prefix also owns those pages.
struct QWEngineSharedPrefix: Sendable {
    let state: QWEnginePrefixState
    let trunk: QWKVSharedPrefix
    let mtp: QWKVSharedPrefix?
    var position: Int { state.position }
    var roundRecord: QWRoundRecord { state.roundRecord }
    var memoryBytes: Int { state.memoryBytes + trunk.memoryBytes + (mtp?.memoryBytes ?? 0) }
}

extension QWEngineSessionState {
    static func captureSharedPrefix(_ session: QWEngineSession, cache: QWKVCache,
                                    mtpCache: QWKVCache?, maxBytes: Int,
                                    mlaIndexerData: Data? = nil) throws -> QWEngineSharedPrefix {
        try validate(session, cache: cache, mtpCache: mtpCache)
        let stateBytes = session.recSlab.bytes + session.ringSlab.bytes + (session.trunkHidden?.bytes ?? 0) +
            (mlaIndexerData?.count ?? 0)
        guard maxBytes >= stateBytes, session.position == cache.written(session.kv),
              mtpCache == nil || (session.roundRecord.consumedCount == session.position &&
                  session.roundRecord.absolutePosition == session.position &&
                  session.roundRecord.outputCount == session.generated),
              session.mtpKV.flatMap({ mtpCache?.written($0) }) ==
                  (mtpCache == nil ? nil : session.roundRecord.mtpPhysicalRows) else {
            throw QWEngineDriverError.state("shared prefix budget or consumed boundary is invalid")
        }
        let trunk = try cache.captureSharedPrefix(session.kv)
        guard trunk.memoryBytes <= maxBytes - stateBytes else {
            throw QWEngineDriverError.state("shared trunk prefix exceeds byte budget")
        }
        let mtp = try session.mtpKV.flatMap { handle in try mtpCache?.captureSharedPrefix(handle) }
        guard (mtp?.memoryBytes ?? 0) <= maxBytes - stateBytes - trunk.memoryBytes else {
            throw QWEngineDriverError.state("shared MTP prefix exceeds byte budget")
        }
        return QWEngineSharedPrefix(state: QWEnginePrefixState(
            try snapshot(session, cache: cache, mtpCache: mtpCache), mlaIndexerData: mlaIndexerData),
            trunk: trunk, mtp: mtp)
    }
}
