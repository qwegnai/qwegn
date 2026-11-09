// EngineSession.swift — per-sequence session state (pre-carved slots,
// zero steady-state allocations; split from QWEngineDriver.swift).
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
import QWState

/// THE PER-LAYER STATE LAW (C23 fix — the single-slot version conflated
public final class QWEngineSession: @unchecked Sendable {
    public internal(set) var position: Int = 0        // next token's absolute position
    public internal(set) var generated: Int = 0
    public internal(set) var roundRecord: QWRoundRecord
    public var tokenHistory: [Int] = []
    let kv: QWKVHandle
    let mtpKV: QWKVHandle?
    var rec: [QWTensorValue]                  // [gdnLayer] → [48·128·128] f32 (zeroed)
    let ring: [QWTensorValue]                 // [gdnLayer] → [3·10240] f16 (zeroed)
    let recSlab: QWTensorValue                // whole slab (memset at admission)
    let ringSlab: QWTensorValue
    let trunkHidden: QWTensorValue?
    let dflashSlot: QWDFlashSessionSlot?
    public internal(set) var dflashCacheOffset: Int = 0
    var trunkHiddenValid = false
    /// finishSession idempotence witness (static-analysis H1): the driver
    /// checks-and-sets it under the flight lock, so a double finish can
    /// never push one GDN slot into the free list twice.
    var sessionFinished = false
    /// MLA indexer pools captured while another session owned the operator.
    var mlaIndexerStash: (data: Data, position: Int)?
    var stateFailed = false
    var verificationPending = false
    var lastCommitCB: (any MTLCommandBuffer)?
    var stochastic: QWStochasticSession?
    var attnKVViews: [(k: QWKVLayerView, v: QWKVLayerView)] = []
    var mtpKVViews: [(k: QWKVLayerView, v: QWKVLayerView)] = []
    public var recStates: [QWTensorValue] { rec }
    public var ringStates: [QWTensorValue] { ring }
    public var mtpPhysicalRows: Int { roundRecord.mtpPhysicalRows }

    init(kv: QWKVHandle, mtpKV: QWKVHandle? = nil, rec: [QWTensorValue],
         ring: [QWTensorValue], recSlab: QWTensorValue, ringSlab: QWTensorValue,
         trunkHidden: QWTensorValue? = nil, dflashSlot: QWDFlashSessionSlot? = nil) {
        self.kv = kv; self.mtpKV = mtpKV; self.rec = rec; self.ring = ring
        self.recSlab = recSlab; self.ringSlab = ringSlab
        self.trunkHidden = trunkHidden
        self.dflashSlot = dflashSlot
        self.roundRecord = QWRoundRecord()
    }

    public func snapshot(using driver: QWEngineDriver) throws -> QWEngineSessionSnapshot {
        try driver.snapshotSession(self)
    }

    public func rollback(using driver: QWEngineDriver, to snapshot: QWEngineSessionSnapshot) throws {
        try driver.rollbackSession(self, to: snapshot)
    }
}
