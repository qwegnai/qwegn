// QWDecisionServerTests.swift — Integration tests for dual-slot sidecar, REST dialects, /mcp, and arbitration.

import Foundation
import XCTest
import QWCore
import QWDecision
import QWEngine
import QWToken
@testable import QWServer

private final class MockHTTPDecisionBackend: QWDecisionExecutionBackend, @unchecked Sendable {
    let decisionConfig: QWDecisionModelConfig
    let maxContextLength: Int
    let handler: @Sendable (QWDecisionBackendBatchRequest) throws -> QWDecisionBackendOutput

    init(
        decisionConfig: QWDecisionModelConfig,
        maxContextLength: Int = 4096,
        handler: @escaping @Sendable (QWDecisionBackendBatchRequest) throws -> QWDecisionBackendOutput
    ) {
        self.decisionConfig = decisionConfig
        self.maxContextLength = maxContextLength
        self.handler = handler
    }

    func evaluate(_ request: QWDecisionBackendBatchRequest) async throws -> QWDecisionBackendOutput {
        try handler(request)
    }
}

final class QWDecisionServerTests: XCTestCase {
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
        for w in ["yes", "no", "ye", "A", "B", "C", "ĠA", "ĠB", "ĠC", "Ġno", "Ġyes", "QUESTION:", "ANSWER:", "SPAN:", "ESTIMATE:", "BOX:"] where vocab[w] == nil {
            vocab[w] = nextId
            nextId += 1
        }
        let merges: [[String]] = [
            ["n", "o"], ["y", "e"], ["ye", "s"],
            ["Ġ", "A"], ["Ġ", "B"], ["Ġ", "C"], ["Ġ", "no"], ["Ġ", "yes"],
        ]
        var addedTokens: [[String: Any]] = [
            ["id": 90000, "content": "<|im_start|>", "special": true, "normalized": false],
            ["id": 90001, "content": "<|im_end|>", "special": true, "normalized": false],
            ["id": 90002, "content": "<|mask|>", "special": true, "normalized": false],
            ["id": 90003, "content": "[CLS]", "special": true, "normalized": false],
            ["id": 90004, "content": "[SEP]", "special": true, "normalized": false],
            ["id": 90005, "content": "[MASK]", "special": true, "normalized": false],
        ]
        for i in 0..<128 {
            addedTokens.append(["id": 90100 + i, "content": "c\(i)", "special": false, "normalized": false])
        }
        let preRegex = "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+|\\p{N}{1,3}| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"
        let pre: [[String: Any]] = [
            ["type": "Split", "pattern": ["Regex": preRegex], "behavior": "Isolated", "invert": false],
            ["type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false],
        ]
        let root: [String: Any] = [
            "added_tokens": addedTokens,
            "model": ["type": "BPE", "vocab": vocab, "merges": merges],
            "pre_tokenizer": ["type": "Sequence", "pretokenizers": pre],
            "normalizer": ["type": "NFC"],
        ]
        return try QWTokenizer(json: try JSONSerialization.data(withJSONObject: root))
    }

    private func makeMockDecisionLoader() throws -> QWEngineLifecycleActor.DecisionRunnerLoader {
        let tok = try makeTestTokenizer()
        let idNo = try XCTUnwrap(tok.singleTokenId(forText: "no"))
        let idYes = try XCTUnwrap(tok.singleTokenId(forText: "yes"))
        let idA = try XCTUnwrap(tok.singleTokenId(forText: "A"))
        let idB = try XCTUnwrap(tok.singleTokenId(forText: "B"))
        let idC = try XCTUnwrap(tok.singleTokenId(forText: "C"))
        let id0 = try XCTUnwrap(tok.singleTokenId(forText: "0"))
        let id1 = try XCTUnwrap(tok.singleTokenId(forText: "1"))
        let id2 = try XCTUnwrap(tok.singleTokenId(forText: "2"))
        let grid0 = (0..<128).compactMap { tok.singleTokenId(forText: "c\($0)") }

        return { packPath, computeUnit, _, _, _ in
            if packPath.hasSuffix("/classifier") {
                let labels = [0: "allow", 1: "confirm", 2: "deny"]
                let cfg = QWDecisionModelConfig(
                    framing: .sequencePair(maxLength: 512),
                    topology: .bidirectionalEncoder(isolateMediaPrefix: false),
                    head: .pooledClassifier(pooling: .firstToken, numLabels: 3, id2label: labels),
                    defaultScoreReadout: "mean"
                )
                let backend = MockHTTPDecisionBackend(decisionConfig: cfg) { req in
                    let slots = req.slots.map { _ in QWDecisionSlotOutput(pooledLogits: [0.1, 2.4, 0.3]) }
                    return QWDecisionBackendOutput(slotOutputs: slots, promptTokens: 24, latencyMs: 1.5)
                }
                return QWLoadedDecisionSidecar(
                    runner: QWDecisionRunner(tokenizer: tok, backend: backend),
                    modelName: "mock-classifier",
                    packPath: packPath,
                    backendKind: "metal4",
                    computeUnit: computeUnit.rawValue,
                    estimatedBytes: 128 << 20,
                    allowedLabels: ["allow", "confirm", "deny"]
                )
            }

            let cfg = QWDecisionModelConfig(
                framing: .turnVerbalizer(codesPerOption: 1, numbers: false, packSet: true, maxSetSize: 16, colonSlots: true, stateSeparator: "\n\n"),
                topology: .packedSet(maxSetSize: 16),
                head: .vocabVerbalizer(vocabSize: 120_000, tieEmbeddings: true, sparseGatherThreshold: 256),
                hasStatePointer: true,
                hasGridReadout: true,
                defaultScoreReadout: "mean"
            )
            let backend = MockHTTPDecisionBackend(decisionConfig: cfg) { req in
                var outputs: [QWDecisionSlotOutput] = []
                outputs.reserveCapacity(req.slots.count)
                var boxEdgeCounter = 0
                for slot in req.slots {
                    if slot.isPointerSlot {
                        let seqLen = max(4, req.sequences[slot.sequenceIndex].tokenIds.count)
                        var sLogits = [Float](repeating: -10.0, count: seqLen)
                        var eLogits = [Float](repeating: -10.0, count: seqLen)
                        sLogits[0] = -4.0
                        if seqLen > 4 {
                            sLogits[1] = 3.0
                            eLogits[3] = 3.0
                        } else {
                            sLogits[1] = 2.0
                            eLogits[1] = 2.0
                        }
                        outputs.append(QWDecisionSlotOutput(pointerStartLogits: sLogits, pointerEndLogits: eLogits))
                        continue
                    }
                    let count = slot.candidateTokenGroups.count
                    if count == 2 {
                        outputs.append(QWDecisionSlotOutput(tokenLogits: [idNo: 0.2, idYes: 2.5]))
                    } else if count == 3 {
                        // Ordinal levels [0, 1, 2] and 3-way choice [A, B, C]:
                        // Score softmax([0.1, 1.0, 2.0]) gives mean ~1.58 vs argmax 2.0 vs median.
                        outputs.append(QWDecisionSlotOutput(tokenLogits: [
                            idA: 0.2, idB: 2.8, idC: 0.4,
                            id0: 0.1, id1: 1.0, id2: 2.0,
                        ]))
                    } else if count == 128 {
                        let center = [25, 25, 76, 76][boxEdgeCounter % 4]
                        boxEdgeCounter += 1
                        var map: [Int: Float] = [:]
                        for (idx, tid) in grid0.enumerated() {
                            map[tid] = (idx == center) ? 6.0 : -4.0
                        }
                        outputs.append(QWDecisionSlotOutput(tokenLogits: map))
                    } else {
                        var map: [Int: Float] = [:]
                        for (idx, grp) in slot.candidateTokenGroups.enumerated() {
                            if let tid = grp.first {
                                map[tid] = Float(idx == 1 ? 2.5 : 0.2)
                            }
                        }
                        outputs.append(QWDecisionSlotOutput(tokenLogits: map))
                    }
                }
                let totalToks = req.sequences.reduce(0) { $0 + $1.tokenIds.count }
                return QWDecisionBackendOutput(slotOutputs: outputs, promptTokens: totalToks, latencyMs: 3.2)
            }
            return QWLoadedDecisionSidecar(
                runner: QWDecisionRunner(tokenizer: tok, backend: backend),
                modelName: "mock-d1-2b",
                packPath: packPath,
                backendKind: computeUnit == .ane ? "coreai_ane" : "metal4",
                computeUnit: computeUnit.rawValue,
                estimatedBytes: 256 << 20
            )
        }
    }

    func testRestDialectsMcpTelemetryHonestyAndAutoReasoning() async throws {
        QWRequestRecordStore.shared.clear()
        let port = 19140
        let loader = try makeMockDecisionLoader()
        let actor = QWEngineLifecycleActor(
            config: QWEngineLifecycleActor.Config(packPath: "/dummy/primary.qwp", modelName: "test-primary", host: "127.0.0.1", port: port),
            logger: QWServerLogger(level: .none),
            decisionLoader: loader
        )
        let server = QWHttpServer(actor: actor, host: "127.0.0.1", port: port, logger: QWServerLogger(level: .none))
        try server.start()
        defer { server.stop() }
        try await Task.sleep(nanoseconds: 100_000_000)

        let baseURL = URL(string: "http://127.0.0.1:\(port)")!
        func post(_ path: String, json obj: Any) async throws -> (Int, [String: Any], Data) {
            var req = URLRequest(url: baseURL.appendingPathComponent(path))
            req.httpMethod = "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: obj)
            let (data, resp) = try await URLSession.shared.data(for: req)
            let code = try XCTUnwrap(resp as? HTTPURLResponse).statusCode
            let parsed = (try? JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
            return (code, parsed, data)
        }

        // 1. Load decision sidecar cleanly with unregistered test path flag.
        setenv("QWG_ALLOW_UNREGISTERED_MODEL_PATHS", "1", 1)
        defer { unsetenv("QWG_ALLOW_UNREGISTERED_MODEL_PATHS") }

        let (loadCode, loadJson, _) = try await post("/v1/decisions/model/load", json: [
            "pack_path": "/tmp/mock-d1-2b",
            "compute_unit": "metal",
        ])
        XCTAssertEqual(loadCode, 200)
        XCTAssertEqual(loadJson["loaded"] as? Bool, true)
        XCTAssertEqual(loadJson["model_name"] as? String, "mock-d1-2b")

        // 2. Multi-kind POST /v1/decisions: binary, choice, score(mean vs argmax), read, box.
        let (decCode, decJson, _) = try await post("/v1/decisions", json: [
            "state": "Refund requested by VIP user after duplicate charge on 2026-11-20.",
            "questions": [
                ["name": "approve", "kind": "binary", "instructions": "Approve refund?"],
                ["name": "route", "kind": "choice", "instructions": "Route ticket?", "options": ["manual", "auto", "escalate"]],
                ["name": "severity_default", "kind": "score", "instructions": "Urgency?", "options": ["low", "med", "high"]],
                ["name": "severity_argmax", "kind": "score", "instructions": "Urgency argmax?", "options": ["low", "med", "high"], "score_readout": "argmax"],
                ["name": "excerpt", "kind": "read", "instructions": "Extract key phrase"],
                ["name": "bbox", "kind": "box", "instructions": "Locate signature"],
            ],
        ])
        XCTAssertEqual(decCode, 200)
        let answers = try XCTUnwrap(decJson["answers"] as? [[String: Any]])
        XCTAssertEqual(answers.count, 6)
        XCTAssertEqual(answers[0]["noul"] as? Bool, true)
        XCTAssertEqual(answers[1]["choice"] as? String, "auto")
        let meanScore = try XCTUnwrap(answers[2]["score"] as? Double)
        let argmaxScore = try XCTUnwrap(answers[3]["score"] as? Double)
        XCTAssertTrue(answers[2]["spreadReported"] as? Bool == true)
        XCTAssertLessThan(meanScore, 1.99)
        XCTAssertEqual(argmaxScore, 2.0, accuracy: 1e-6)
        XCTAssertEqual(answers[4]["found"] as? Bool, true)
        let boxes = try XCTUnwrap(answers[5]["boxes"] as? [[String: Any]])
        XCTAssertEqual(boxes.count, 1)

        // 3. Batch + SystemOne aliases.
        let (batchCode, batchJson, _) = try await post("/v1/decisions/batch", json: [
            "items": [
                ["state": "Ping", "questions": [["name": "ok", "kind": "noul", "instructions": "Valid?"]]],
                ["state": "Pong", "questions": [["name": "ok", "kind": "noul", "instructions": "Valid?"]]],
            ],
        ])
        XCTAssertEqual(batchCode, 200)
        XCTAssertEqual((batchJson["results"] as? [[String: Any]])?.count, 2)

        let (sysCode, sysJson, _) = try await post("/v1/systemone", json: [
            "model": "mock-d1-2b",
            "state": ["messages": [["role": "user", "content": [["type": "text", "text": "Allow read-only git status"]]]]],
            "questions": [
                "safe": ["type": "noul", "instructions": "Is command safe?"],
                "severity": ["type": "score", "instructions": "Urgency?", "criteria": ["Can wait", "Today", "Immediate"]],
                "bbox": ["type": "box", "instructions": "Locate badge"],
            ],
        ])
        XCTAssertEqual(sysCode, 200)
        let sysAnswers = try XCTUnwrap(sysJson["answers"] as? [String: Any])
        let safeAns = try XCTUnwrap(sysAnswers["safe"] as? [String: Any])
        XCTAssertEqual(safeAns["type"] as? String, "noul")
        XCTAssertGreaterThan(try XCTUnwrap(safeAns["noul"] as? Double), 0.5)
        let scoreAns = try XCTUnwrap(sysAnswers["severity"] as? [String: Any])
        XCTAssertEqual(scoreAns["type"] as? String, "score")
        let legend = try XCTUnwrap(scoreAns["legend"] as? [String: Any])
        XCTAssertEqual(legend["0"] as? String, "Can wait")
        let boxAns = try XCTUnwrap(sysAnswers["bbox"] as? [String: Any])
        XCTAssertEqual(boxAns["type"] as? String, "box")
        XCTAssertNotNil(boxAns["box"] as? [String: Any])
        let sysUsage = try XCTUnwrap(sysJson["usage"] as? [String: Any])
        XCTAssertNotNil(sysUsage["input_tokens"])

        // 3b. TypeSafe GET /v1/models metadata + multipart /v1/systemone + OpenAI /v1/decisions dialect.
        let (modelsData, modelsResp) = try await URLSession.shared.data(from: baseURL.appendingPathComponent("/v1/models"))
        XCTAssertEqual((modelsResp as? HTTPURLResponse)?.statusCode, 200)
        let modelsJson = try XCTUnwrap(JSONSerialization.jsonObject(with: modelsData) as? [String: Any])
        let tsModels = try XCTUnwrap(modelsJson["models"] as? [[String: Any]])
        XCTAssertTrue(tsModels.contains { ($0["name"] as? String) == "mock-d1-2b" && ($0["release_date"] as? String) != nil })

        let boundary = "qwg-test-boundary-123"
        let assetRef = "asset-uuid-1"
        let multiReqObj: [String: Any] = [
            "model": "mock-d1-2b",
            "state": assetRef,
            "assets": [assetRef: ["type": "image", "part": "asset-0"]],
            "questions": ["has_badge": ["type": "noul", "instructions": "Inspect image"]],
        ]
        let multiReqData = try JSONSerialization.data(withJSONObject: multiReqObj)
        let fakePNG = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x01])
        var multiBody = Data("--\(boundary)\r\nContent-Disposition: form-data; name=\"request\"\r\nContent-Type: application/json\r\n\r\n".utf8)
        multiBody.append(multiReqData)
        multiBody.append(Data("\r\n--\(boundary)\r\nContent-Disposition: form-data; name=\"asset-0\"; filename=\"asset-0\"\r\nContent-Type: image/png\r\n\r\n".utf8))
        multiBody.append(fakePNG)
        multiBody.append(Data("\r\n--\(boundary)--\r\n".utf8))
        var multiURLReq = URLRequest(url: baseURL.appendingPathComponent("/v1/systemone"))
        multiURLReq.httpMethod = "POST"
        multiURLReq.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        multiURLReq.httpBody = multiBody
        let (multiRespData, multiResp) = try await URLSession.shared.data(for: multiURLReq)
        XCTAssertEqual((multiResp as? HTTPURLResponse)?.statusCode, 200)
        let multiJson = try XCTUnwrap(JSONSerialization.jsonObject(with: multiRespData) as? [String: Any])
        XCTAssertEqual(((multiJson["answers"] as? [String: Any])?["has_badge"] as? [String: Any])?["type"] as? String, "noul")

        let (oaiCode, oaiJson, _) = try await post("/v1/decisions", json: [
            "model": "mock-d1-2b",
            "input": [["role": "user", "type": "message", "content": [["type": "input_text", "text": "Refund approved."]]]],
            "questions": [
                ["name": "valid", "type": "predicate", "instructions": "Valid?"],
                ["name": "route", "type": "choice", "instructions": "Route?", "choices": [["value": "A", "description": "a"], ["value": "B", "description": "b"], ["value": "C", "description": "c"]]],
            ],
        ])
        XCTAssertEqual(oaiCode, 200)
        let oaiAnswers = try XCTUnwrap(oaiJson["answers"] as? [[String: Any]])
        XCTAssertEqual(oaiAnswers.count, 2)
        XCTAssertEqual(oaiAnswers[0]["type"] as? String, "predicate")
        XCTAssertNotNil(oaiAnswers[0]["probability"] as? Double)
        XCTAssertEqual(oaiAnswers[1]["type"] as? String, "choice")
        XCTAssertEqual((oaiAnswers[1]["probabilities"] as? [[String: Any]])?.count, 3)

        let (healthData, healthResp) = try await URLSession.shared.data(from: baseURL.appendingPathComponent("/health"))
        XCTAssertEqual((healthResp as? HTTPURLResponse)?.statusCode, 200)
        let healthJson = try XCTUnwrap(JSONSerialization.jsonObject(with: healthData) as? [String: Any])
        XCTAssertEqual(healthJson["loaded"] as? Bool, true)
        XCTAssertEqual(healthJson["status"] as? String, "loaded")
        XCTAssertEqual(healthJson["model"] as? String, "mock-d1-2b")

        let (sysBatchCode, sysBatchJson, _) = try await post("/v1/systemone/batch", json: [
            "items": [
                ["state": "A", "questions": [["name": "b", "kind": "binary", "instructions": "Yes?"]]],
            ],
        ])
        XCTAssertEqual(sysBatchCode, 200)
        XCTAssertEqual((sysBatchJson["results"] as? [[String: Any]])?.count, 1)

        // 4. POST /v1/qwegn + 60,000-character rail.
        let (askCode, askJson, _) = try await post("/v1/qwegn", json: [
            "context": "git status",
            "guard": true,
            "question": "Is this safe?",
        ])
        XCTAssertEqual(askCode, 200)
        XCTAssertEqual(askJson["decision"] as? String, "confirm")

        let oversized = String(repeating: "x", count: 60_001)
        let (railCode, _, _) = try await post("/v1/qwegn", json: [
            "context": oversized,
            "question": "Safe?",
        ])
        XCTAssertEqual(railCode, 400)

        // 5. Stateless Streamable-HTTP POST /mcp.
        let (initCode, initJson, _) = try await post("/mcp", json: [
            "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": [:],
        ])
        XCTAssertEqual(initCode, 200)
        XCTAssertNotNil(initJson["result"])

        let (notifCode, _, _) = try await post("/mcp", json: [
            "jsonrpc": "2.0", "method": "notifications/initialized",
        ])
        XCTAssertEqual(notifCode, 202)

        let (listCode, listJson, _) = try await post("/mcp", json: [
            "jsonrpc": "2.0", "id": 2, "method": "tools/list",
        ])
        XCTAssertEqual(listCode, 200)
        let toolList = ((listJson["result"] as? [String: Any])?["tools"] as? [[String: Any]]) ?? []
        let toolNames = Set(toolList.compactMap { $0["name"] as? String })
        XCTAssertTrue(Set(["qwg_decide", "qwg_ask", "qwg_guard", "qwg_status", "qwg_rerank_screen", "qwg_vault_query"]).isSubset(of: toolNames))
        XCTAssertTrue(toolNames.allSatisfy { !$0.lowercased().contains("l" + "aya") })

        for (idx, call) in [
            ("qwg_decide", ["state": "hi", "questions": [["name": "q", "kind": "noul", "instructions": "Ok?"]]] as [String: Any]),
            ("qwg_ask", ["context": "hi", "question": "Ok?"] as [String: Any]),
            ("qwg_guard", ["command": "rm -rf /tmp/foo"] as [String: Any]),
            ("qwg_status", [:] as [String: Any]),
        ].enumerated() {
            let (callCode, callJson, _) = try await post("/mcp", json: [
                "jsonrpc": "2.0",
                "id": 10 + idx,
                "method": "tools/call",
                "params": ["name": call.0, "arguments": call.1],
            ])
            XCTAssertEqual(callCode, 200)
            let res = try XCTUnwrap(callJson["result"] as? [String: Any])
            XCTAssertEqual(res["isError"] as? Bool, false)
        }

        // 6. Verify C >= 3 fixed classifier allowed_labels 400 error.
        let (clsLoadCode, _, _) = try await post("/v1/decisions/model/load", json: [
            "pack_path": "/tmp/mock/classifier",
            "compute_unit": "metal",
        ])
        XCTAssertEqual(clsLoadCode, 200)
        let (subsetCode, subsetJson, _) = try await post("/v1/decisions", json: [
            "state": "Delete temp files",
            "questions": [
                ["name": "act", "kind": "choice", "instructions": "Pick", "options": ["allow", "nonexistent_label"]],
            ],
        ])
        XCTAssertEqual(subsetCode, 400)
        XCTAssertEqual(subsetJson["allowed_labels"] as? [String], ["allow", "confirm", "deny"])

        // 7. GET /v1/decisions/status & POST /v1/decisions/model/unload.
        let (statData, statResp) = try await URLSession.shared.data(from: baseURL.appendingPathComponent("/v1/decisions/status"))
        XCTAssertEqual(try XCTUnwrap(statResp as? HTTPURLResponse).statusCode, 200)
        let statJson = try XCTUnwrap(JSONSerialization.jsonObject(with: statData) as? [String: Any])
        XCTAssertEqual(statJson["loaded"] as? Bool, true)

        let (unloadCode, unloadJson, _) = try await post("/v1/decisions/model/unload", json: [:])
        XCTAssertEqual(unloadCode, 200)
        XCTAssertEqual(unloadJson["loaded"] as? Bool, false)

        // 8. Telemetry honesty: 0-token decisions must NEVER inject 0.0 tok/s into QWRequestRecordStore.
        XCTAssertTrue(QWRequestRecordStore.shared.recent(64).isEmpty)

        // 9. ADR-010 reasoning.effort = "auto" 3-tier rubric with 0.7 fail-open rail.
        XCTAssertEqual(QWChatPolicyResolver.resolveAutoReasoningEffort(tier: "trivial", confidence: 0.85), "off")
        XCTAssertEqual(QWChatPolicyResolver.resolveAutoReasoningEffort(tier: "moderate", confidence: 0.72), "low")
        XCTAssertEqual(QWChatPolicyResolver.resolveAutoReasoningEffort(tier: "deep", confidence: 0.91), "xhigh")
        XCTAssertNil(QWChatPolicyResolver.resolveAutoReasoningEffort(tier: "trivial", confidence: 0.69))
    }

    func testGPUTurnArbitratorSyncPrimaryPreemption() async throws {
        let arb = QWGPUTurnArbitrator()
        let (insideStream, insideCont) = AsyncStream<Void>.makeStream()
        let (firstDoneStream, firstDoneCont) = AsyncStream<Void>.makeStream()
        let (secondDoneStream, secondDoneCont) = AsyncStream<Void>.makeStream()

        DispatchQueue.global(qos: .userInitiated).async {
            arb.withPrimaryStepSync {
                insideCont.yield(())
                insideCont.finish()
                Thread.sleep(forTimeInterval: 0.03)
            }
            firstDoneCont.yield(())
            firstDoneCont.finish()
        }

        for await _ in insideStream { break }
        let sidecarRan = await arb.runHighPrioritySidecar {
            // Queue a second sync primary while sidecar actively holds GPU to exercise syncCondition wake.
            DispatchQueue.global(qos: .userInitiated).async {
                arb.withPrimaryStepSync {
                    Thread.sleep(forTimeInterval: 0.005)
                }
                secondDoneCont.yield(())
                secondDoneCont.finish()
            }
            try? await Task.sleep(nanoseconds: 15_000_000)
            return true
        }
        XCTAssertTrue(sidecarRan)
        for await _ in firstDoneStream { break }
        for await _ in secondDoneStream { break }
        XCTAssertGreaterThanOrEqual(arb.totalPreemptionsServed, 1)
    }

    func testDecisionAndQwegnAskHTTPImagePayloadDecodingAndSSRFRejection() throws {
        let rawBytes = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x01, 0x02, 0x03])
        let b64 = rawBytes.base64EncodedString()

        let decPayload: [String: Any] = [
            "state": "Inspect image",
            "images": ["data:image/png;base64,\(b64)"],
            "questions": [["name": "has_cat", "kind": "noul", "instructions": "Cat visible?"]],
        ]
        let decReq = try JSONDecoder().decode(
            QWDecisionHTTPRequest.self,
            from: JSONSerialization.data(withJSONObject: decPayload)
        )
        let decDomain = try decReq.toDecisionRequest()
        XCTAssertEqual(decDomain.images, [rawBytes])

        let askPayload: [String: Any] = [
            "question": "Is there a defect?",
            "kind": "noul",
            "images": [b64],
        ]
        let askReq = try JSONDecoder().decode(
            QWQwegnAskHTTPRequest.self,
            from: JSONSerialization.data(withJSONObject: askPayload)
        )
        let askDomain = try askReq.toDecisionRequest()
        XCTAssertEqual(askDomain.images, [rawBytes])

        for forbidden in ["http://127.0.0.1/img.png", "https://example.com/a.png", "/etc/passwd"] {
            let badPayload: [String: Any] = [
                "images": [forbidden],
                "questions": [["name": "q", "kind": "noul", "instructions": "Ok?"]],
            ]
            let badReq = try JSONDecoder().decode(
                QWDecisionHTTPRequest.self,
                from: JSONSerialization.data(withJSONObject: badPayload)
            )
            XCTAssertThrowsError(try badReq.toDecisionRequest()) { err in
                XCTAssertTrue(err is QWDecisionError, "expected QWDecisionError for '\(forbidden)'")
            }
        }
    }

    func testEstimateDecisionSidecarBytesDoesNotCrawlParentDirectoryForMissingPath() throws {
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwg-sidecar-est-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let siblingURL = tmpDir.appendingPathComponent("sibling.bin")
        try Data(repeating: 0xAA, count: 1 << 20).write(to: siblingURL)

        let missingPath = tmpDir.appendingPathComponent("missing-pack").path
        XCTAssertEqual(QWEngineLifecycleActor.estimateDecisionSidecarBytes(at: missingPath), 256 << 20)
        XCTAssertEqual(QWEngineLifecycleActor.estimateDecisionSidecarBytes(at: siblingURL.path), 1 << 20)
    }

    func testEstimateDecisionSidecarBytesVisionToggleDifference() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("qwg-st-uma-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let weightBytes = Data(count: 1 << 20)
        let visBytes = Data(count: 2 << 20)
        var blob = Data()
        let header: [String: Any] = [
            "model.embed_tokens.weight": ["dtype": "BF16", "shape": [1024, 512], "data_offsets": [0, weightBytes.count]],
            "model.vision_tower.patch.weight": ["dtype": "BF16", "shape": [1024, 1024], "data_offsets": [weightBytes.count, weightBytes.count + visBytes.count]],
        ]
        let hdrData = try JSONSerialization.data(withJSONObject: header)
        var hdrLen = UInt64(hdrData.count).littleEndian
        blob.append(Data(bytes: &hdrLen, count: 8))
        blob.append(hdrData)
        blob.append(weightBytes)
        blob.append(visBytes)
        try blob.write(to: dir.appendingPathComponent("model.safetensors"))

        let withVis = QWEngineLifecycleActor.estimateDecisionSidecarBytes(at: dir.path, loadVision: true)
        let noVis = QWEngineLifecycleActor.estimateDecisionSidecarBytes(at: dir.path, loadVision: false)
        XCTAssertGreaterThan(withVis, noVis)
        XCTAssertEqual(withVis - noVis, 2 << 20, accuracy: 4096)
    }
}
