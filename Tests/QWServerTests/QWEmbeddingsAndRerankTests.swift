// QWEmbeddingsAndRerankTests.swift — Tests for POST /v1/embeddings and POST /v1/rerank (G1).

import Foundation
import QWCore
import QWDecision
import QWToken
import XCTest
@testable import QWServer

private final class MockEmbedAndRerankBackend: QWDecisionExecutionBackend, QWEmbeddingExecutionBackend, @unchecked Sendable {
    let decisionConfig: QWDecisionModelConfig
    let maxContextLength: Int = 4096
    let embeddingDimension: Int = 4

    init() {
        self.decisionConfig = QWDecisionModelConfig(
            framing: .sequencePair(maxLength: 4096),
            topology: .bidirectionalEncoder(isolateMediaPrefix: false),
            head: .pooledClassifier(pooling: .meanNonPad, numLabels: 2, id2label: [0: "no", 1: "yes"])
        )
    }

    func evaluate(_ request: QWDecisionBackendBatchRequest) async throws -> QWDecisionBackendOutput {
        let outs: [QWDecisionSlotOutput] = request.slots.map { slot in
            let seq = request.sequences[slot.sequenceIndex]
            let isHit = seq.tokenIds.contains(90010)
            let isYesPair = (seq.pairIndex == 1)
            let score: Float = (isHit == isYesPair) ? 3.0 : -3.0
            return QWDecisionSlotOutput(pooledLogits: [0.0, score])
        }
        return QWDecisionBackendOutput(slotOutputs: outs, promptTokens: request.sequences.reduce(0) { $0 + $1.tokenIds.count }, latencyMs: 1.5)
    }

    func embedBatch(
        tokenBatches: [[Int]],
        pooling: QWDecisionModelConfig.PoolingKind,
        normalize: Bool,
        truncateDimensions: Int?
    ) async throws -> (embeddings: [[Float]], promptTokens: Int, latencyMs: Double) {
        _ = pooling
        let vecs = tokenBatches.map { toks -> [Float] in
            let raw: [Float] = [Float(toks.count), 1.0, 2.0, 3.0]
            return QWEmbeddingMath.postProcess(raw, normalize: normalize, truncateDimensions: truncateDimensions)
        }
        return (vecs, tokenBatches.reduce(0) { $0 + $1.count }, 1.2)
    }
}

final class QWEmbeddingsAndRerankTests: XCTestCase {
    private func makeTestTokenizer() throws -> QWTokenizer {
        var vocab: [String: Int] = [:]
        var nextId = 0
        var n = 0
        for b in 0..<256 {
            let printable = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)
            let sc = printable ? Unicode.Scalar(UInt8(b)) : Unicode.Scalar(0x100 + n)!
            if !printable { n += 1 }
            if b == 0xC0 || b == 0xC1 || b >= 0xF5 { continue }
            vocab[String(sc)] = nextId
            nextId += 1
        }
        vocab["swift"] = 100
        let preRegex = "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+|\\p{N}{1,3}| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"
        let pre: [[String: Any]] = [
            ["type": "Split", "pattern": ["Regex": preRegex], "behavior": "Isolated", "invert": false],
            ["type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false],
        ]
        let root: [String: Any] = [
            "added_tokens": [
                ["id": 90000, "content": "[CLS]", "special": true, "normalized": false],
                ["id": 90001, "content": "[SEP]", "special": true, "normalized": false],
                ["id": 90010, "content": "targethit", "special": false, "normalized": false],
            ],
            "model": ["type": "BPE", "vocab": vocab, "merges": [[String]]()],
            "pre_tokenizer": ["type": "Sequence", "pretokenizers": pre],
            "normalizer": ["type": "NFC"],
        ]
        return try QWTokenizer(json: try JSONSerialization.data(withJSONObject: root))
    }

    private func makeServer(withSidecar: Bool) throws -> (QWHttpServer, Int) {
        let tok = try makeTestTokenizer()
        let actor = QWEngineLifecycleActor(
            config: QWEngineLifecycleActor.Config(packPath: "/tmp/unused", modelName: "primary"),
            logger: QWServerLogger(level: .none),
            decisionLoader: { path, _, _, _, _ in
                let backend = MockEmbedAndRerankBackend()
                let runner = QWDecisionRunner(tokenizer: tok, backend: backend)
                return QWLoadedDecisionSidecar(
                    runner: runner, modelName: "bge-reranker-mock", packPath: path,
                    backendKind: "metal4", computeUnit: "metal", estimatedBytes: 64 << 20
                )
            }
        )
        let server = QWHttpServer(actor: actor, host: "127.0.0.1", port: 0, logger: QWServerLogger(level: .none))
        if withSidecar {
            let sem = DispatchSemaphore(value: 0)
            Task {
                _ = try? await actor.loadDecisionSidecar(packPath: "/tmp/mock-embed")
                sem.signal()
            }
            sem.wait()
        }
        try server.start()
        let port = Int(server.listener?.port?.rawValue ?? 0)
        return (server, port)
    }

    private func postJSON(port: Int, path: String, body: [String: Any]) async throws -> (Int, [String: Any]) {
        let url = URL(string: "http://127.0.0.1:\(port)\(path)")!
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, resp) = try await URLSession.shared.data(for: req)
        let status = (resp as? HTTPURLResponse)?.statusCode ?? 0
        let json = (try? JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
        return (status, json)
    }

    func testEmbeddingsFloatBase64AndMatryoshkaDimensions() async throws {
        let (server, port) = try makeServer(withSidecar: true)
        defer { server.stop() }

        let (s1, j1) = try await postJSON(port: port, path: "/v1/embeddings", body: [
            "input": ["hello", "swift"],
            "dimensions": 2,
        ])
        XCTAssertEqual(s1, 200)
        let dataArr = try XCTUnwrap(j1["data"] as? [[String: Any]])
        XCTAssertEqual(dataArr.count, 2)
        let emb0 = try XCTUnwrap(dataArr[0]["embedding"] as? [Double])
        XCTAssertEqual(emb0.count, 2)
        let norm0 = sqrt(emb0[0] * emb0[0] + emb0[1] * emb0[1])
        XCTAssertEqual(norm0, 1.0, accuracy: 1e-4)

        let (s2, j2) = try await postJSON(port: port, path: "/v1/embeddings", body: [
            "input": "hello",
            "encoding_format": "base64",
        ])
        XCTAssertEqual(s2, 200)
        let b64Arr = try XCTUnwrap(j2["data"] as? [[String: Any]])
        let b64Str = try XCTUnwrap(b64Arr.first?["embedding"] as? String)
        let rawBytes = try XCTUnwrap(Data(base64Encoded: b64Str))
        XCTAssertEqual(rawBytes.count, 16)

        let (sBad, _) = try await postJSON(port: port, path: "/v1/embeddings", body: ["input": ""])
        XCTAssertEqual(sBad, 400)
    }

    func testRerankOrderingTopNAndReturnDocuments() async throws {
        let (server, port) = try makeServer(withSidecar: true)
        defer { server.stop() }

        let (status, json) = try await postJSON(port: port, path: "/v1/rerank", body: [
            "query": "swift metal",
            "documents": ["unrelated doc", ["text": "targethit kernel"], "another doc"],
            "top_n": 2,
            "return_documents": true,
        ])
        XCTAssertEqual(status, 200)
        let results = try XCTUnwrap(json["results"] as? [[String: Any]])
        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results[0]["index"] as? Int, 1)
        let docObj = try XCTUnwrap(results[0]["document"] as? [String: Any])
        XCTAssertEqual(docObj["text"] as? String, "targethit kernel")
        let s0 = try XCTUnwrap(results[0]["relevance_score"] as? Double)
        let s1 = try XCTUnwrap(results[1]["relevance_score"] as? Double)
        XCTAssertGreaterThan(s0, s1)
    }

    func testEmbeddingsReturns503WhenSidecarUnloaded() async throws {
        let (server, port) = try makeServer(withSidecar: false)
        defer { server.stop() }

        let (status, _) = try await postJSON(port: port, path: "/v1/embeddings", body: ["input": "hello"])
        XCTAssertEqual(status, 503)
    }
}
