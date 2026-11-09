// QWKnobProfile.swift — The knob values one model was configured with.
// A profile records the geometry it was written against: the knobs a model
// honours depend on it, so a profile from another geometry is stale rather
// than applicable.

import Foundation
import QWCore

/// The knob values one model was configured with.
public struct QWKnobProfile: Sendable, Codable, Equatable {
    /// The shape of this file. A profile written against another schema is
    /// migrated, not guessed at.
    public var schemaVersion: Int
    /// The model these values belong to.
    public var modelId: String
    /// The geometry the values were written against — the model's shape,
    /// never its knobs. See `QWProfileStore` for the whole contract.
    public var geometryDigest: String
    /// When the profile was last written. The store stamps this: the
    /// moment the file reached the disk is the one it names.
    public var updatedAt: Date
    /// The values, by knob id. A knob that is not here was never set.
    public var values: [String: QWKnobValue]

    /// The schema version this build writes.
    public static let currentSchemaVersion = 1

    public init(schemaVersion: Int = QWKnobProfile.currentSchemaVersion,
                modelId: String,
                geometryDigest: String = "",
                updatedAt: Date = Date(),
                values: [String: QWKnobValue] = [:]) {
        self.schemaVersion = schemaVersion
        self.modelId = modelId
        self.geometryDigest = geometryDigest
        self.updatedAt = updatedAt
        self.values = values
    }
}
