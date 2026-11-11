// KnobBindingTests.swift — The CLI bindings are generated from the registry.
// The rules this suite binds:
//  1. every knob the registry exposes on the `.cli` surface has a binding,
//  2. the checked-in bindings are what the generator produces today,
//  3. the generator reads the registry as the registry declares it,
//  4. a value the registry refuses is refused at the command line.

import Foundation
import Testing
@testable import qwg
import QWCli
import QWCore

// MARK: - The registry, as the generator reads it

/// The generator's own reading of the registry. The suite holds it against
/// the live registry, so a parser that drifts from the source it reads
/// fails here instead of quietly emitting the wrong bindings.
private struct RegistryDump: Decodable {
    let knobs: [Knob]

    struct Knob: Decodable {
        let id: String
        let tier: String
        let scope: String
        let kind: Kind
        let `default`: Value
        let requiredCapability: String?
        let valueCapabilities: [String: String]
        let surfaces: [String]
    }

    struct Kind: Decodable {
        let kind: String
        let min: Double?
        let max: Double?
        let allowed: [String]?

        private enum CodingKeys: String, CodingKey { case kind, min, max, allowed }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            kind = try c.decode(String.self, forKey: .kind)
            min = try c.decodeIfPresent(Double.self, forKey: .min)
            max = try c.decodeIfPresent(Double.self, forKey: .max)
            allowed = try c.decodeIfPresent([String].self, forKey: .allowed)
        }
    }

    struct Value: Decodable {
        let kind: String
        let anyValue: Any

        private enum CodingKeys: String, CodingKey { case kind, value }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            kind = try c.decode(String.self, forKey: .kind)
            switch kind {
            case "int": anyValue = try c.decode(Int.self, forKey: .value)
            case "double": anyValue = try c.decode(Double.self, forKey: .value)
            case "bool": anyValue = try c.decode(Bool.self, forKey: .value)
            default: anyValue = try c.decode(String.self, forKey: .value)
            }
        }
    }
}

private func kindMatches(_ json: RegistryDump.Kind, _ kind: QWKnobKind) -> Bool {
    switch (json.kind, kind) {
    case ("int", .int(let range)):
        return json.min == Double(range.lowerBound) && json.max == Double(range.upperBound)
    case ("double", .double(let range)):
        return json.min == range.lowerBound && json.max == range.upperBound
    case ("bool", .bool):
        return true
    case ("string", .string(let allowed)):
        return Set(json.allowed ?? []) == allowed
    default:
        return false
    }
}

private func defaultMatches(_ json: RegistryDump.Value, _ value: QWKnobValue) -> Bool {
    switch (json.kind, value) {
    case ("int", .int(let v)): return (json.anyValue as? Int) == v
    case ("double", .double(let v)): return (json.anyValue as? Double) == v
    case ("bool", .bool(let v)): return (json.anyValue as? Bool) == v
    case ("string", .string(let v)): return (json.anyValue as? String) == v
    default: return false
    }
}

// MARK: - Helpers

private func repoRoot() -> URL {
    var d = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
    for _ in 0..<6 {
        if FileManager.default.fileExists(atPath: d.appendingPathComponent("Package.swift").path) {
            return d
        }
        d = d.deletingLastPathComponent()
    }
    fatalError("the repository root is missing")
}

private func generatedPath(_ root: URL) -> URL {
    root.appendingPathComponent("Sources/Apps/qwg/Generated/KnobOptions.swift")
}

private func sourcePath(_ root: URL, _ relative: String) -> URL {
    root.appendingPathComponent(relative)
}

/// Runs the generator and returns what it printed.
private func runGenerator(_ arguments: [String]) throws -> String {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    process.arguments = ["python3", "Scripts/gen_knob_cli.py"] + arguments
    process.currentDirectoryURL = repoRoot()
    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = FileHandle.nullDevice
    try process.run()
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    guard process.terminationStatus == 0 else {
        throw NSError(domain: "gen_knob_cli", code: Int(process.terminationStatus), userInfo: [
            NSLocalizedDescriptionKey: "Scripts/gen_knob_cli.py \(arguments.joined(separator: " ")) failed",
        ])
    }
    return String(decoding: data, as: UTF8.self)
}

// MARK: - The bindings

@Test("every knob the registry exposes on the CLI has a generated binding")
func generatedBindingsMatchTheRegistry() throws {
    let root = repoRoot()
    let generated = try String(contentsOf: generatedPath(root), encoding: .utf8)
    for descriptor in QWKnobRegistry.all where descriptor.surfaces.contains(.cli) {
        #expect(generated.contains("\"\(descriptor.id)\""),
                "knob \(descriptor.id) has no generated CLI binding")
    }
}

@Test("the checked-in CLI bindings are what the generator produces today")
func generatorOutputIsUpToDate() throws {
    let root = repoRoot()
    let checkedIn = try String(contentsOf: generatedPath(root), encoding: .utf8)
    let regenerated = try runGenerator(["--stdout"])
    #expect(regenerated == checkedIn,
            "the generated CLI bindings are stale — re-run Scripts/gen_knob_cli.py")
}

@Test("the generator reads the registry as the registry declares it")
func generatorReadsTheRegistryFaithfully() throws {
    let dump = try JSONDecoder().decode(RegistryDump.self,
                                        from: Data(try runGenerator(["--json"]).utf8))
    let declared = Dictionary(uniqueKeysWithValues: QWKnobRegistry.all.map { ($0.id, $0) })
    #expect(dump.knobs.count == declared.count)
    for knob in dump.knobs {
        guard let descriptor = declared[knob.id] else {
            Issue.record("the generator read a knob the registry does not declare: \(knob.id)")
            continue
        }
        #expect(knob.tier == descriptor.tier.rawValue)
        #expect(knob.scope == descriptor.scope.rawValue)
        #expect(kindMatches(knob.kind, descriptor.kind))
        #expect(defaultMatches(knob.default, descriptor.defaultValue))
        #expect(knob.requiredCapability == descriptor.requiredCapability?.rawValue)
        #expect(knob.valueCapabilities == descriptor.valueCapabilities.mapValues { $0.rawValue })
        #expect(Set(knob.surfaces) == Set(descriptor.surfaces.map { $0.rawValue }))
    }
}

@Test("every binding names a property the generated file declares")
func bindingsNameDeclaredProperties() throws {
    let generated = try String(contentsOf: generatedPath(repoRoot()), encoding: .utf8)
    for binding in QWGKnobOptions.bindings {
        #expect(generated.contains("public var \(binding.property):"),
                "\(binding.id) binds to '\(binding.property)', which the file does not declare")
    }
    let cliKnobs = QWKnobRegistry.all.filter { $0.surfaces.contains(.cli) }
    #expect(QWGKnobOptions.bindings.count == cliKnobs.count)
}

@Test("the run, repl and serve surfaces consume the generated bindings")
func surfacesConsumeTheGeneratedBindings() throws {
    let root = repoRoot()
    for surface in ["Sources/Apps/qwg/qwg.swift", "Sources/Apps/qwg/QwgReplAndServe.swift"] {
        let source = try String(contentsOf: sourcePath(root, surface), encoding: .utf8)
        #expect(source.contains("QWGKnobOptions"), "\(surface) must consume the generated bindings")
        #expect(source.contains("prefillChunkSize"), "\(surface) must consume the generated bindings")
        #expect(!source.contains("dflash-block-size"), "\(surface) must not name the retired flag")
    }
}

@Test("the CLI no longer offers an automatic KV mode")
func theAutomaticKVModeIsGone() throws {
    let root = repoRoot()
    let library = try String(contentsOf: sourcePath(root, "Sources/Apps/QWCli/QWCli.swift"),
                            encoding: .utf8)
    #expect(!library.contains("resolveKVRequest"))
    #expect(throws: QWCli.CLIError.self) { try QWCli.resolveKVMode("auto") }
    #expect(throws: QWCli.CLIError.self) { try QWCli.resolveKVMode("AUTO") }
    #expect(try QWCli.resolveKVMode("f16") == .f16)
    #expect(try QWCli.resolveKVMode("q8") == .q8)
    #expect(try QWCli.resolveKVMode("q4") == .q4)
}

// MARK: - The values

@Test("from keeps a value that fits its knob and refuses one that does not")
func fromValidatesAgainstTheRegistry() throws {
    let kept = try QWGKnobOptions.from(["temperature": .double(0.7), "seed": .int(42)])
    #expect(kept.count == 2)
    #expect(kept["temperature"] == .double(0.7))
    #expect(kept["seed"] == .int(42))

    do {
        _ = try QWGKnobOptions.from(["noSuchKnob": .int(1)])
        Issue.record("a name the registry does not know must not pass")
    } catch let error as QWKnobError {
        #expect(error == .unknownKnob("noSuchKnob"))
    }

    do {
        _ = try QWGKnobOptions.from(["temperature": .double(5)])
        Issue.record("a value the registry refuses must not pass")
    } catch let error as QWKnobError {
        guard case .invalidValue(let knob, _, _) = error else {
            Issue.record("expected an invalidValue, got \(error)")
            return
        }
        #expect(knob == "temperature")
    }
}

@Test("explicitValues carries only the knobs the user set")
func explicitValuesCarryOnlyWhatWasSet() throws {
    let unset = try QWGKnobOptions.parse([])
    #expect(unset.explicitValues.isEmpty)
    let set = try QWGKnobOptions.parse(["--temperature", "0.5", "--seed", "7"])
    #expect(set.explicitValues == ["temperature": .double(0.5), "seed": .int(7)])
}

@Test("the bindings refuse a value the registry rejects")
func theBindingsRefuseWhatTheRegistryRejects() throws {
    #expect(throws: (any Error).self) { try QWGKnobOptions.parse(["--temperature", "5"]) }
    #expect(throws: (any Error).self) { try QWGKnobOptions.parse(["--kv-mode", "auto"]) }
    #expect(throws: (any Error).self) { try QWGKnobOptions.parse(["--reasoning-effort", "high"]) }
    #expect(throws: (any Error).self) { try QWGKnobOptions.parse(["--max-context", "0"]) }
}

@Test("a value the registry accepts passes the bindings' own check")
func theBindingsAcceptWhatTheRegistryAccepts() throws {
    let options = try QWGKnobOptions.parse(["--temperature", "0.5", "--kv-mode", "q8"])
    #expect(throws: Never.self) { try options.validate() }
}

@Test("the thinking knob reads the same whichever spelling the user used")
func theThinkingKnobReadsTheSameEitherWay() throws {
    #expect(try QWGKnobOptions.parse(["--enable-thinking", "true"]).explicitValues["enableThinking"] == .bool(true))
    #expect(try QWGKnobOptions.parse(["--enable-thinking", "false"]).explicitValues["enableThinking"] == .bool(false))
    #expect(try QWGKnobOptions.parse(["--thinking"]).explicitValues["enableThinking"] == .bool(true))
    #expect(try QWGKnobOptions.parse(["--no-thinking"]).explicitValues["enableThinking"] == .bool(false))
    #expect(try QWGKnobOptions.parse([]).explicitValues["enableThinking"] == nil)
}

@Test("an enumeration is read without regard to case; a free-form string is not")
func anEnumerationIsReadWithoutRegardToCase() throws {
    #expect(try QWGKnobOptions.parse(["--kv-mode", "F16"]).explicitValues["kvMode"] == .string("f16"))
    #expect(try QWGKnobOptions.parse(["--system", "Hello THERE"]).explicitValues["systemPrompt"] == .string("Hello THERE"))
    #expect(throws: (any Error).self) { try QWGKnobOptions.parse(["--reasoning-effort", "HIGH"]) }
}

// MARK: - The values

/// The registry's default, as the command line would spell it.
private func argument(for value: QWKnobValue) -> String? {
    switch value {
    case .int(let v): return String(v)
    case .double(let v): return String(v)
    case .bool(let v): return v ? "true" : "false"
    case .string(let v): return v
    }
}

@Test("every knob the CLI exposes parses to the kind the registry declares")
func everyCLIKnobParsesToItsKind() throws {
    for descriptor in QWKnobRegistry.all where descriptor.surfaces.contains(.cli) {
        let binding = try #require(QWGKnobOptions.bindings.first { $0.id == descriptor.id },
                                   "\(descriptor.id) has no binding")
        let argOpt = argument(for: descriptor.defaultValue)
        let argument = try #require(argOpt)
        let parsed: QWGKnobOptions
        if case .bool = descriptor.kind {
            // A bool knob may be a flag, which carries no value.
            if let valued = try? QWGKnobOptions.parse(["--\(binding.flag)", argument]) {
                parsed = valued
            } else {
                parsed = try QWGKnobOptions.parse(["--\(binding.flag)"])
            }
            #expect(parsed.explicitValues[descriptor.id] == .bool(true),
                    "\(descriptor.id) did not read back as a bool")
        } else {
            // A value that starts with a dash has to ride the flag: the
            // command line would otherwise read it as the next option.
            let arguments = argument.hasPrefix("-")
                ? ["--\(binding.flag)=\(argument)"]
                : ["--\(binding.flag)", argument]
            parsed = try QWGKnobOptions.parse(arguments)
            #expect(parsed.explicitValues[descriptor.id] == descriptor.defaultValue,
                    "\(descriptor.id) did not read back as \(descriptor.defaultValue)")
        }
        #expect(throws: Never.self) { try parsed.validate() }
    }
}

@Test("every knob the command line exposes is honoured by some command")
func everyCLIKnobIsHonouredBySomeCommand() throws {
    let honoured = Set(QWGKnobOptions.honouredBy.keys)
    for descriptor in QWKnobRegistry.all where descriptor.surfaces.contains(.cli) {
        #expect(honoured.contains(descriptor.id),
                "\(descriptor.id) is on the CLI but no command honours it")
    }
}

/// The source of one command, from its declaration to the next declaration.
private func commandSource(root: URL, file: String, command: String) throws -> String {
    let source = try String(contentsOf: sourcePath(root, file), encoding: .utf8)
    let chunks = source.components(separatedBy: "\nstruct ")
    var chunk = try #require(chunks.first { $0.hasPrefix("\(command):") },
                             "no \(command) in \(file)")
    if chunk.contains("QWResolvedCliEngineConfig.") {
        let resolver = try String(
            contentsOf: sourcePath(root, "Sources/Apps/qwg/Shared/QWCliEngineConfigResolver.swift"),
            encoding: .utf8
        )
        chunk += "\n" + resolver
    }
    return chunk
}

/// The knobs a command's body reads, as the registry's ids.
private func knobsRead(in source: String) -> Set<String> {
    let properties = Set(QWGKnobOptions.bindings.map { $0.property })
    let idsByProperty = Dictionary(uniqueKeysWithValues: QWGKnobOptions.bindings.map { ($0.property, $0.id) })
    var found: Set<String> = []
    var rest = Substring(source)
    while let range = rest.range(of: "knobs.") {
        rest = rest[range.upperBound...]
        let name = rest.prefix(while: { $0.isLetter || $0.isNumber || $0 == "_" })
        guard properties.contains(String(name)), let id = idsByProperty[String(name)] else { continue }
        found.insert(id)
    }
    return found
}

@Test("each command honours exactly the knobs its body reads")
func eachCommandHonoursExactlyWhatItReads() throws {
    let root = repoRoot()
    let surfaces: [(String, String, String)] = [
        ("qwg run", "Sources/Apps/qwg/qwg.swift", "Run"),
        ("qwg plan", "Sources/Apps/qwg/qwg.swift", "Plan"),
        ("qwg repl", "Sources/Apps/qwg/QwgReplAndServe.swift", "Repl"),
        ("qwg serve", "Sources/Apps/qwg/QwgReplAndServe.swift", "Serve"),
    ]
    for (command, file, structName) in surfaces {
        let source = try commandSource(root: root, file: file, command: structName)
        let read = knobsRead(in: source)
        let honoured = try #require(QWGKnobOptions.honoured[command], "no honoured set for \(command)")
        #expect(read == honoured,
                "\(command) reads \(read.sorted()) but honours \(honoured.sorted())")
    }
}
