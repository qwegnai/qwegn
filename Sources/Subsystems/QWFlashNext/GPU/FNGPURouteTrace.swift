import Foundation
import QWAlloc

enum FNGPURouteTrace {
    static let enabled = ProcessInfo.processInfo.environment["QW_FLASH_ROUTE_TRACE"] == "1"

    struct Report: Encodable {
        let schema = 1
        let phase: String
        let position: Int
        let rows: Int
        let layer: Int
        let experts: Int
        let routes: Int
        let ids: [Int32]

        init(phase: String, position: Int, rows: Int, layer: Int,
             experts: Int, routes: Int, ids: [Int32]) throws {
            guard rows > 0, experts > 0, routes > 0,
                  ids.count == rows * routes,
                  ids.allSatisfy({ $0 >= -1 && $0 < experts }) else {
                throw TraceError.invalidRoutes
            }
            self.phase = phase; self.position = position; self.rows = rows
            self.layer = layer; self.experts = experts; self.routes = routes; self.ids = ids
        }
    }

    enum TraceError: Error, CustomStringConvertible {
        case invalidRoutes
        var description: String { "Flash-Next route trace: invalid route dimensions or expert IDs" }
    }

    static func emit(arena: QWArena, slots: [FNGPUDecodeStack.Slot],
                     phase: String, position: Int, rows: Int) throws {
        // Other layers reuse scratch; only dedicated PLE scratch and the last
        // layer's scratch still contain their own routes after stack completion.
        for (index, slot) in slots.enumerated() where slot.weights.ple != nil || index == slots.count - 1 {
            let m = slot.weights.moe
            let report = try Report(phase: phase, position: position, rows: rows,
                                    layer: slot.weights.index, experts: m.experts, routes: m.topK,
                                    ids: arena.read(slot.scratch.idx, as: Int32.self, count: rows * m.topK))
            var data = Data("[flash-routes] ".utf8)
            data.append(try JSONEncoder().encode(report))
            data.append(0x0A)
            try FileHandle.standardError.write(contentsOf: data)
        }
    }
}
