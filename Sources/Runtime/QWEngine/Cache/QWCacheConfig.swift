import Foundation
import QWCore

/// Configuration for the two-tier prefix caching hierarchy (Warm RAM + Cold SSD).
public struct QWCacheConfig: Sendable, Equatable {
    /// Maximum bytes allowed in warm memory (RAM). Default: 4 GiB.
    public var warmByteLimit: Int
    /// Maximum bytes allowed in cold storage on SSD. Default: 10 GiB.
    public var ssdByteLimit: Int
    /// Maximum number of distinct prefix entries in warm memory.
    public var entryLimit: Int
    /// Maximum number of concurrent session slots protected in warm memory and batch scheduling (1...16). Default: 8.
    public var maxConcurrentSessions: Int
    /// Directory used for cold SSD caching. Default: ~/.qwg/kv-cache.
    public var cacheDirectory: URL
    /// Whether to wipe stale cache files on engine startup. Default: true.
    public var wipeOnStart: Bool

    public static let defaultWarmLimit: Int = 4 << 30   // 4 GiB
    public static let defaultSSDLimit: Int = 10 << 30   // 10 GiB
    public static let defaultEntryLimit: Int = 32
    public static let defaultMaxConcurrentSessions: Int = 8

    /// Returns ~/.qwg/kv-cache, expanding tilde and environment variables.
    public static var defaultCacheDirectory: URL {
        QWPaths.defaultKVCacheDirectoryURL
    }

    public init(
        warmByteLimit: Int = defaultWarmLimit,
        ssdByteLimit: Int = defaultSSDLimit,
        entryLimit: Int = defaultEntryLimit,
        maxConcurrentSessions: Int = defaultMaxConcurrentSessions,
        cacheDirectory: URL = defaultCacheDirectory,
        wipeOnStart: Bool = true
    ) {
        self.warmByteLimit = warmByteLimit
        self.ssdByteLimit = ssdByteLimit
        self.entryLimit = entryLimit
        self.maxConcurrentSessions = max(1, min(16, maxConcurrentSessions))
        self.cacheDirectory = cacheDirectory
        self.wipeOnStart = wipeOnStart
    }
}
