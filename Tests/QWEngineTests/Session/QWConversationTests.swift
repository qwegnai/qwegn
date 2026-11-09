import XCTest
@testable import QWEngine

final class QWConversationTests: XCTestCase {
    func testExactPrefixOnly() {
        let consumed = [1, 2, 3]
        XCTAssertEqual(QWConversation.reusableCount(consumed: consumed, prompt: [1, 2, 3, 4], position: 3), 3)
        for prompt in [[1, 2, 3], [1, 2], [1, 9, 3, 4], [9, 1, 2, 3, 4]] {
            XCTAssertEqual(QWConversation.reusableCount(consumed: consumed, prompt: prompt, position: 3), 0)
        }
        XCTAssertEqual(QWConversation.reusableCount(consumed: consumed, prompt: [1, 2, 3, 4], position: 2), 0)
        XCTAssertEqual(QWConversation.reusableCount(consumed: [], prompt: [1], position: 0), 0)
    }

    func testConsumedNotPredictedBoundary() throws {
        let prompt = [1, 2, 3], output = [4, 5, 6]
        XCTAssertEqual(try QWConversation.committedTokens(prompt: prompt, output: output, position: 5), [1, 2, 3, 4, 5])
        XCTAssertEqual(try QWConversation.committedTokens(prompt: prompt, output: output, position: 6), [1, 2, 3, 4, 5, 6])
        XCTAssertEqual(try QWConversation.committedTokens(prompt: prompt, output: [], position: 3), prompt)
        XCTAssertEqual(try QWConversation.committedTokens(prompt: prompt, output: [4], position: 3), prompt)
        XCTAssertThrowsError(try QWConversation.committedTokens(prompt: prompt, output: output, position: 2))
        XCTAssertThrowsError(try QWConversation.committedTokens(prompt: prompt, output: output, position: 7))
    }
}
