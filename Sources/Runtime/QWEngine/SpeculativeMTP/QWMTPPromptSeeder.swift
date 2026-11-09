import Metal
import QWAlloc
import QWCore
import QWKV

/// Seeds shifted trunk-hidden/token-embedding pairs, never draft-hidden pairs.
enum QWMTPPromptSeeder {
    static func pairCount(position: Int, count: Int) -> Int {
        position == 0 ? max(0, count - 1) : count
    }

    /// The trunk producer has ended its encoder. The previous-hidden buffer
    /// is read by the seed pass before its last-row replacement is blitted.
    static func encodeChunk(_ cb: any MTLCommandBuffer, draft: QWMTPDraftEncoder,
                            embeddings: QWTensorValue, trunkHidden: QWTensorValue,
                            previousHidden: QWTensorValue, position: Int, count: Int,
                            cacheRows: Int, kv: (k: QWKVLayerView, v: QWKVLayerView)) throws {
        guard let enc = cb.makeComputeCommandEncoder() else {
            throw QWEngineDriverError.state("MTP seed encoder")
        }
        do {
            try encode(enc, draft: draft, embeddings: embeddings, trunkHidden: trunkHidden,
                       previousHidden: previousHidden, position: position, count: count, cacheRows: cacheRows, kv: kv)
        } catch {
            enc.endEncoding()
            throw error
        }
        enc.endEncoding()
        guard let blit = cb.makeBlitCommandEncoder() else {
            throw QWEngineDriverError.state("MTP hidden retention encoder")
        }
        blit.copy(from: trunkHidden.region.buffer,
                  sourceOffset: trunkHidden.region.base + trunkHidden.offset + (count - 1) * draft.geometry.hidden * 2,
                  to: previousHidden.region.buffer, destinationOffset: previousHidden.region.base + previousHidden.offset,
                  size: previousHidden.bytes)
        blit.endEncoding()
    }

    static func encode(_ enc: any MTLComputeCommandEncoder, draft: QWMTPDraftEncoder,
                       embeddings: QWTensorValue, trunkHidden: QWTensorValue,
                       previousHidden: QWTensorValue, position: Int, count: Int,
                       cacheRows: Int, kv: (k: QWKVLayerView, v: QWKVLayerView)) throws {
        let h = draft.geometry.hidden
        func valid(_ t: QWTensorValue, elements: Int) -> Bool {
            t.shape.dtype == .f16 && t.shape.elementCount == elements
                && t.offset >= 0 && t.region.base >= 0
                && t.offset <= t.region.length && t.bytes <= t.region.length - t.offset
                && t.region.base <= t.region.buffer.length
                && t.region.length <= t.region.buffer.length - t.region.base
                && (t.region.base + t.offset) % 16 == 0
        }
        guard position >= 0, count > 0, count <= 4096, cacheRows >= 0,
              position <= Int.max - count,
              position + count <= draft.positionCapacity,
              valid(embeddings, elements: count * h), valid(trunkHidden, elements: count * h),
              valid(previousHidden, elements: h) else {
            throw QWEngineDriverError.state("MTP prompt seeding input is invalid")
        }
        let pairs = pairCount(position: position, count: count)
        func disjoint(_ t: QWTensorValue) -> Bool {
            guard t.region.buffer === previousHidden.region.buffer else { return true }
            let start = t.region.base + t.offset
            let previousStart = previousHidden.region.base + previousHidden.offset
            return start + t.bytes <= previousStart || previousStart + previousHidden.bytes <= start
        }
        guard disjoint(embeddings), disjoint(trunkHidden) else {
            throw QWEngineDriverError.state("MTP previous hidden must not alias chunk inputs")
        }
        guard cacheRows <= QWKVCache.maxTokensPerSeq - pairs else {
            throw QWEngineDriverError.state("MTP prompt pair capacity exceeded")
        }
        if pairs > 0 {
            let pages = (cacheRows + pairs + QWKVCache.pageTokens - 1) / QWKVCache.pageTokens
            guard kv.k.table.shape.elementCount >= pages, kv.v.table.shape.elementCount >= pages else {
                throw QWEngineDriverError.state("MTP prompt cache must grow before encoding")
            }
        }
        func row(_ t: QWTensorValue, _ index: Int) -> QWTensorValue {
            QWTensorValue(region: t.region, offset: t.offset + index * h * 2,
                          shape: QWTensorShape([h], .f16))
        }
        var physicalRow = cacheRows
        for t in (position == 0 ? 1 : 0)..<count {
            try draft.encodePairCache(enc, embedding: row(embeddings, t),
                                      hidden: t == 0 ? previousHidden : row(trunkHidden, t - 1),
                                      kv: kv, cacheRows: physicalRow, absolutePosition: position + t)
            physicalRow += 1
        }
    }
}
