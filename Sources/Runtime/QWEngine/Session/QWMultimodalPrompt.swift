// QWMultimodalPrompt.swift — Multimodal prompt preparation and X0 activation interleaving.
import Foundation
import QWCore
import QWMM
import QWVision

public enum QWMultimodalError: Error, CustomStringConvertible {
    case featureDimensionMismatch(expected: Int, actual: Int)
    case gridCountMismatch(expected: Int, actual: Int)
    case badArgument(String)

    public var description: String {
        switch self {
        case .featureDimensionMismatch(let exp, let act):
            return "QWMultimodal: visual feature count mismatch (expected \(exp), got \(act))"
        case .gridCountMismatch(let exp, let act):
            return "QWMultimodal: grid count mismatch (expected \(exp), got \(act))"
        case .badArgument(let msg):
            return "QWMultimodal: \(msg)"
        }
    }
}

public struct QWMultimodalPrompt: Sendable {
    public let originalTokens: [Int]
    public let expandedTokens: [Int]
    public let plan: QWX0Plan
    public let visualTokens: [Float]
    public let hiddenDim: Int

    public init(
        promptTokens: [Int],
        images: [(grid: QWVisionGrid, features: [Float])],
        videos: [(grid: QWVisionGrid, features: [Float])] = [],
        hiddenDim: Int = QWModelGeometry.qwen38_27b.hidden
    ) throws {
        self.originalTokens = promptTokens
        self.hiddenDim = hiddenDim

        var allVisTokens: [Float] = []
        let imageGrids = images.map { $0.grid }
        let videoGrids = videos.map { $0.grid }

        for (grid, feat) in images {
            let expectedCount = grid.tokens * hiddenDim
            guard feat.count == expectedCount else {
                throw QWMultimodalError.featureDimensionMismatch(expected: expectedCount, actual: feat.count)
            }
            allVisTokens.append(contentsOf: feat)
        }

        for (grid, feat) in videos {
            let expectedCount = grid.tokens * hiddenDim
            guard feat.count == expectedCount else {
                throw QWMultimodalError.featureDimensionMismatch(expected: expectedCount, actual: feat.count)
            }
            allVisTokens.append(contentsOf: feat)
        }

        self.visualTokens = allVisTokens
        self.plan = try QWX0Plan(tokens: promptTokens, images: imageGrids, videos: videoGrids)
        self.expandedTokens = plan.tokens
    }

    /// Extracts the contiguous visual feature spans (`[tokenCount * hiddenDim]`) in expanded-prompt token coordinates.
    public var visualSpans: [QWVisualSpan] {
        var out: [QWVisualSpan] = []
        var visOffset = 0
        for span in plan.spans {
            if case .vision(_, _, let rows) = span.content {
                let count = rows * hiddenDim
                let slice = Array(visualTokens[visOffset..<(visOffset + count)])
                out.append(QWVisualSpan(tokenOffset: span.rowStart, tokenCount: rows, embeddings: slice))
                visOffset += count
            }
        }
        return out
    }

    /// Resolves the full X0 slab with text embeddings and visual features interleaved.
    public func resolveX0(textEmbedder: (Int) throws -> [Float]) throws -> [Float] {
        var textEmbeddings: [Float] = []
        for span in plan.spans {
            if case let .text(tokenStart, count) = span.content {
                for i in 0..<count {
                    let tok = expandedTokens[tokenStart + i]
                    let emb = try textEmbedder(tok)
                    guard emb.count == hiddenDim else {
                        throw QWMultimodalError.featureDimensionMismatch(expected: hiddenDim, actual: emb.count)
                    }
                    textEmbeddings.append(contentsOf: emb)
                }
            }
        }
        return try qwInterleaveX0(
            textEmbeddings: textEmbeddings,
            visualTokens: visualTokens,
            plan: plan,
            hiddenDim: hiddenDim
        )
    }
}
