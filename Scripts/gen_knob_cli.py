#!/usr/bin/env python3
"""Generate the CLI knob bindings from the knob registry.

The registry (Sources/Foundation/QWCore/QWKnobRegistry.swift) is the single
declaration of every user-facing knob. This script reads it and writes the
command-line bindings, so a knob added to the registry reaches the CLI
without a hand-written option. Deterministic: the same registry always
yields the same file.

Usage:
  python3 Scripts/gen_knob_cli.py            # write the generated file
  python3 Scripts/gen_knob_cli.py --stdout   # print it instead
  python3 Scripts/gen_knob_cli.py --json     # print the registry as JSON
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY = os.path.join(ROOT, "Sources/Foundation/QWCore/QWKnobRegistry.swift")
OUTPUT = os.path.join(ROOT, "Sources/Apps/qwg/Generated/KnobOptions.swift")

# The command-line spelling of each knob: the flag the user types and the
# older names that still answer to it. A knob not named here keeps the flag
# its id spells out.
CLI_FLAGS = {
    "contextTokens": ("context-tokens", ["max-context"]),
    "maxTokens": ("max-tokens", []),
    "systemPrompt": ("system-prompt", ["system"]),
    "enableThinking": ("enable-thinking", ["thinking"]),
    "prefillChunkRows": ("prefill-chunk-size", []),
    "prefixCacheEnabled": ("prefix-cache", []),
    "warmCacheLimitGiB": ("warm-cache-limit", []),
    "ssdCacheLimitGiB": ("ssd-cache-limit", []),
    "osReserveGiB": ("os-reserve", []),
    "lazyLoad": ("lazy", []),
    "dflashDraftPackPath": ("dflash-draft-pack", []),
    "dflashSelectorTopK": ("dflash-top-k", []),
}

# How a bool knob is spelled. A flag is the CLI idiom, but the thinking
# knob has always been spelled both ways, and a user who learned one of
# them has to keep it: the value form and the flag form are the same knob.
CLI_BOOL_FORMS = {
    "enableThinking": "option+flag",
}

# The knobs each command honours. The command line speaks one vocabulary,
# but no command applies every knob: `qwg serve` loads the model lazily
# and `qwg run` never does. A knob the user set that the command cannot
# honour is refused, not ignored — a flag that is accepted and then
# dropped is a lie.
CLI_COMMANDS = {
    "qwg run": [
        "contextTokens", "maxTokens", "kvMode", "enableThinking", "reasoningEffort",
        "systemPrompt", "gdnPrefill", "evictPrefillScratch",
        "lsaMode", "lsaBudget", "lsaClusterBlock", "lsaSidecarPath",
        "mtpDrafts", "mtpBackend",
        "mtpRelaxedMargin", "mtpRelaxedMinProb", "mtpDraftStrategy", "mtpQuadraticSeeding", "mtpAdaptiveDepth",
        "mtpEntropyThreshold",
        "dflashDraftPackPath", "dflashSelectorTopK", "memoryPolicy", "osReserveGiB",
        "residentExpertsPerLayer", "moeSpecBudget", "prefillChunkRows",
        "temperature", "topP", "topK", "repetitionPenalty", "seed", "prefixCacheEnabled",
        "cacheDir", "loadVision", "streamNgrams",
    ],
    "qwg repl": [
        "contextTokens", "maxTokens", "kvMode", "enableThinking", "reasoningEffort",
        "systemPrompt", "warmCacheLimitGiB", "ssdCacheLimitGiB", "cacheDir",
        "lsaMode", "lsaBudget", "lsaClusterBlock", "lsaSidecarPath",
        "mtpDrafts",
        "mtpBackend", "mtpRelaxedMargin", "mtpRelaxedMinProb", "mtpDraftStrategy", "mtpQuadraticSeeding",
        "mtpAdaptiveDepth", "mtpEntropyThreshold",
        "dflashDraftPackPath", "dflashSelectorTopK", "memoryPolicy", "osReserveGiB",
        "residentExpertsPerLayer", "moeSpecBudget",
        "prefillChunkRows", "temperature", "topP", "topK", "repetitionPenalty", "seed",
        "loadVision", "streamNgrams",
    ],
    "qwg serve": [
        "contextTokens", "kvMode",
        "lsaMode", "lsaBudget", "lsaClusterBlock", "lsaSidecarPath",
        "mtpDrafts", "mtpBackend",
        "mtpRelaxedMargin", "mtpRelaxedMinProb", "mtpDraftStrategy", "mtpQuadraticSeeding", "mtpAdaptiveDepth",
        "mtpEntropyThreshold",
        "dflashDraftPackPath",
        "dflashSelectorTopK", "memoryPolicy", "osReserveGiB", "residentExpertsPerLayer", "moeSpecBudget",
        "gdnPrefill", "evictPrefillScratch", "prefillChunkRows",
        "maxConcurrentSessions",
        "warmCacheLimitGiB", "ssdCacheLimitGiB", "cacheDir", "logLevel", "temperature",
        "topP", "topK", "maxTokens", "repetitionPenalty", "seed", "enableThinking",
        "reasoningEffort", "systemPrompt", "lazyLoad", "loadVision", "streamNgrams",
    ],
    "qwg plan": ["contextTokens", "kvMode"],
}

# What the flag means, in one line. The help a user reads is written here,
# once, rather than in every surface that exposes the knob.
CLI_HELP = {
    "contextTokens": "Context budget in tokens for the KV cache.",
    "maxTokens": "Max generated tokens per turn.",
    "temperature": "Softmax sampling temperature (0 is greedy).",
    "topP": "Nucleus sampling probability threshold.",
    "topK": "Top-k sampling cutoff (0 disables the cutoff).",
    "repetitionPenalty": "Repetition penalty factor.",
    "seed": "Deterministic RNG seed (-1 for random, written --seed=-1).",
    "enableThinking": "Enable or disable model thinking.",
    "reasoningEffort": "Reasoning effort: low, medium or xhigh.",
    "systemPrompt": "System prompt text.",
    "speculativeEnabled": "Enable speculative decoding.",
    "mtpDrafts": "MTP speculation depth (0 is serial decode).",
    "mtpBackend": "MTP execution backend: standard, variational, dflash2 or dspine.",
    "mtpRelaxedMargin": "Variational MTP relaxed acceptance log-logit margin delta (arXiv:2610.26693).",
    "mtpRelaxedMinProb": "Variational MTP fidelity floor: a relaxed draft also needs target probability >= this (0 disables).",
    "mtpDraftStrategy": "MTP draft proposal strategy: hybrid, neural or ngram.",
    "mtpQuadraticSeeding": "Seed MTP KV cache from verified hidden states on all-accept rounds.",
    "prefillChunkRows": "Prefill chunk size in tokens (32...8192, a multiple of 32).",
    "maxConcurrentSessions": "Maximum concurrent sessions for KV prefix cache slots and batch scheduling (1...16, default 8).",
    "prefixCacheEnabled": "Enable the persistent disk-backed prefix cache.",
    "warmCacheLimitGiB": "Warm RAM prefix cache capacity in GiB (0 disables it).",
    "ssdCacheLimitGiB": "Cold SSD prefix cache capacity in GiB (0 disables it).",
    "cacheDir": "Directory for the SSD prefix cache.",
    "cacheEntryLimit": "How many prefix-cache entries to keep.",
    "memoryPolicy": "Unified-memory residency policy: maxResident or safe.",
    "osReserveGiB": "OS RAM reserve in GiB, subtracted from physical memory.",
    "residentExpertsPerLayer": "Resident hot experts per MoE layer in unified memory (0 sizes automatically from memory policy).",
    "moeSpecBudget": "MoE-Spec routed expert budget per verify layer (0 is lossless, 1...8 caps cold SSD reads).",
    "evictPrefillScratch": "Evict the prefill scratchpad from RAM: auto, always or never.",
    "lsaMode": "Latent Sparse Attention mode: off, auto, sidecar, svd or qsa-cluster (arXiv:2610.42147).",
    "lsaBudget": "LSA active key budget k in tokens (512...8192, default 2048).",
    "lsaClusterBlock": "LSA cluster block size L in tokens (256...4096, default 512; 4096 enables e-means).",
    "lsaSidecarPath": "Explicit path to a .qlsa (QLSA0001) projection sidecar file.",
    "lazyLoad": "Load the model on the first request instead of at startup.",
    "loadVision": "Load the vision tower into memory for multimodal processing.",
    "streamNgrams": "Stream PLE N-gram table from SSD (~64 MB cache) instead of loading into RAM.",
    "logLevel": "Log level: debug, info, warn, error or none.",
    "kvMode": "KV cache quantization: f16, q8 or q4.",
    "gdnPrefill": "GDN prefill algorithm: chunked, regres or naive.",
    "mtpAdaptiveDepth": "Let the MTP depth follow the draft's own confidence.",
    "mtpEntropyThreshold": "Entropy threshold for an adaptive MTP depth.",
    "ropeScalingFactor": "RoPE scaling factor (YaRN).",
    "dflashDraftPackPath": "Path to the DFlash 2 / DSpine drafter model pack.",
    "dflashSelectorTopK": "DFlash 2 / DSpine candidate selector top-K.",
    "splitKTokensPerSplit": "Tokens per split in a split-K GEMV.",
}


def strip_comments(text):
    """The source without its comments; string literals are left alone."""
    out = []
    i = 0
    in_string = False
    while i < len(text):
        c = text[i]
        if in_string:
            out.append(c)
            if c == "\\" and i + 1 < len(text):
                out.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                in_string = False
            i += 1
            continue
        if c == '"':
            in_string = True
            out.append(c)
            i += 1
            continue
        if c == "/" and i + 1 < len(text) and text[i + 1] == "/":
            while i < len(text) and text[i] != "\n":
                i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def balanced(text, start):
    """The text inside the parentheses that open at `start`."""
    depth = 0
    i = start
    in_string = False
    while i < len(text):
        c = text[i]
        if in_string:
            if c == "\\":
                i += 2
                continue
            if c == '"':
                in_string = False
            i += 1
            continue
        if c == '"':
            in_string = True
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
        i += 1
    raise ValueError("unbalanced parentheses in the registry")


def split_arguments(text):
    """The comma-separated arguments of a call, at the top level only."""
    args = []
    current = []
    depth = 0
    i = 0
    in_string = False
    while i < len(text):
        c = text[i]
        if in_string:
            current.append(c)
            if c == "\\" and i + 1 < len(text):
                current.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                in_string = False
            i += 1
            continue
        if c == '"':
            in_string = True
            current.append(c)
        elif c in "([{":
            depth += 1
            current.append(c)
        elif c in ")]}":
            depth -= 1
            current.append(c)
        elif c == "," and depth == 0:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(c)
        i += 1
    tail = "".join(current).strip()
    if tail:
        args.append(tail)
    return args


def parse_number(text):
    """A Swift numeric literal, underscores and all."""
    cleaned = text.replace("_", "").strip()
    if re.fullmatch(r"-?\d+", cleaned):
        return int(cleaned)
    if re.fullmatch(r"-?\d*\.\d+", cleaned):
        return float(cleaned)
    # A bound written as the width of the type it names.
    fixed = {"Int(Int32.max)": 2147483647, "Int(Int32.min)": -2147483648,
             "Int(Int64.max)": 9223372036854775807, "Int(UInt32.max)": 4294967295}
    if cleaned in fixed:
        return fixed[cleaned]
    raise ValueError("not a number: %r" % text)


def parse_string(text):
    """A Swift string literal."""
    text = text.strip()
    if not (text.startswith('"') and text.endswith('"')):
        raise ValueError("not a string: %r" % text)
    return text[1:-1]


def parse_kind(text):
    """The value shape a knob accepts."""
    text = text.strip()
    if text == ".bool":
        return {"kind": "bool"}
    match = re.fullmatch(r"\.int\((.*)\)", text, re.S)
    if match:
        low, high = split_range(match.group(1))
        return {"kind": "int", "min": low, "max": high}
    match = re.fullmatch(r"\.double\((.*)\)", text, re.S)
    if match:
        low, high = split_range(match.group(1))
        return {"kind": "double", "min": low, "max": high}
    match = re.fullmatch(r"\.string\(allowed:\s*\[(.*)\]\)", text, re.S)
    if match:
        allowed = [parse_string(a) for a in split_arguments(match.group(1)) if a.strip()]
        return {"kind": "string", "allowed": sorted(allowed)}
    raise ValueError("not a knob kind: %r" % text)


def split_range(text):
    """The two ends of a Swift range."""
    parts = text.split("...")
    if len(parts) != 2:
        raise ValueError("not a range: %r" % text)
    return parse_number(parts[0]), parse_number(parts[1])


def parse_value(text):
    """A knob's default, tagged with the kind of value it is."""
    text = text.strip()
    for kind, parse in (("int", parse_number), ("double", parse_number),
                        ("bool", lambda t: t.strip() == "true"),
                        ("string", parse_string)):
        match = re.fullmatch(r"\.%s\((.*)\)" % kind, text, re.S)
        if match:
            return {"kind": kind, "value": parse(match.group(1))}
    raise ValueError("not a knob value: %r" % text)


def parse_surfaces(text, named):
    """The surfaces a knob is reachable from."""
    text = text.strip()
    if text.startswith("["):
        return sorted(a.strip().lstrip(".") for a in split_arguments(text[1:-1]) if a.strip())
    if text in named:
        return sorted(named[text])
    raise ValueError("not a surface set: %r" % text)


def parse_capabilities(text):
    """The extra capability a string value needs, per value."""
    text = text.strip()
    if not (text.startswith("[") and text.endswith("]")):
        raise ValueError("not a capability map: %r" % text)
    out = {}
    for entry in split_arguments(text[1:-1]):
        if not entry.strip():
            continue
        key, _, value = entry.partition(":")
        out[parse_string(key)] = value.strip().lstrip(".")
    return out


def find_surface_sets(source):
    """The named surface sets the registry declares."""
    named = {}
    pattern = re.compile(r"let\s+(\w+)\s*:\s*Set<QWKnobSurface>\s*=\s*\[(.*?)\]", re.S)
    for match in pattern.finditer(source):
        members = sorted(a.strip().lstrip(".") for a in split_arguments(match.group(2)) if a.strip())
        named[match.group(1)] = members
    return named


def read_registry(path):
    """Every knob the registry declares, in declaration order."""
    with open(path, encoding="utf-8") as handle:
        source = strip_comments(handle.read())
    named = find_surface_sets(source)
    knobs = []
    for match in re.finditer(r"QWKnobDescriptor\(", source):
        body, _ = balanced(source, match.end() - 1)
        fields = {}
        for argument in split_arguments(body):
            key, _, value = argument.partition(":")
            fields[key.strip()] = value.strip()
        knob = {
            "id": parse_string(fields["id"]),
            "tier": fields["tier"].strip().lstrip("."),
            "scope": fields["scope"].strip().lstrip("."),
            "kind": parse_kind(fields["kind"]),
            "default": parse_value(fields["defaultValue"]),
            "requiredCapability": (fields["requiredCapability"].strip().lstrip(".")
                                    if "requiredCapability" in fields else None),
            "valueCapabilities": (parse_capabilities(fields["valueCapabilities"])
                                  if "valueCapabilities" in fields else {}),
            "surfaces": parse_surfaces(fields["surfaces"], named),
        }
        knobs.append(knob)
    return knobs


def cli_flag(knob_id):
    """The flag a knob answers to, and the older names that still do."""
    if knob_id in CLI_FLAGS:
        return CLI_FLAGS[knob_id]
    words = re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])", knob_id)
    kebab = "-".join(w.lower() for w in words)
    return (kebab, [])


def property_name(flag):
    """The Swift property a flag reads."""
    parts = flag.split("-")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


def swift_type(kind):
    """The Swift type a knob's value is carried in."""
    return {"int": "Int", "double": "Double", "bool": "Bool", "string": "String"}[kind["kind"]]


def swift_literal(value):
    """A knob value as a Swift literal."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return '"%s"' % value.replace("\\", "\\\\").replace('"', '\\"')
    if isinstance(value, float):
        return repr(value)
    return str(value)


def emit(knobs):
    """The generated Swift file."""
    cli = [k for k in knobs if "cli" in k["surfaces"]]
    lines = []
    lines.append("// KnobOptions.swift — GENERATED by Scripts/gen_knob_cli.py; do not edit.")
    lines.append("// The source of truth is the knob registry in")
    lines.append("// Sources/Foundation/QWCore/QWKnobRegistry.swift. Regenerate with")
    lines.append("// `python3 Scripts/gen_knob_cli.py`.")
    lines.append("")
    lines.append("import ArgumentParser")
    lines.append("import Foundation")
    lines.append("import QWCore")
    lines.append("")
    lines.append("/// One knob, bound to the command line.")
    lines.append("public struct QWGKnobBinding: Sendable {")
    lines.append("    /// The registry's id for the knob.")
    lines.append("    public let id: String")
    lines.append("    /// The Swift property the binding reads.")
    lines.append("    public let property: String")
    lines.append("    /// The flag the binding answers to.")
    lines.append("    public let flag: String")
    lines.append("}")
    lines.append("")
    lines.append("/// The command-line bindings for every knob the registry exposes on the")
    lines.append("/// `.cli` surface. A knob the user did not set is absent, never a default")
    lines.append("/// wearing the user's clothes.")
    lines.append("public struct QWGKnobOptions: ParsableArguments {")
    for knob in cli:
        flag, aliases = cli_flag(knob["id"])
        name = property_name(flag)
        names = ", ".join(".customLong(%s)" % json.dumps(a) for a in [flag] + aliases)
        kind = knob["kind"]["kind"]
        help_text = CLI_HELP.get(knob["id"], knob["id"])
        lines.append("    /// %s" % help_text)
        if kind == "bool" and CLI_BOOL_FORMS.get(knob["id"]) == "option+flag":
            # The value form keeps `--enable-thinking true`; the flag form
            # keeps `--thinking` and `--no-thinking`. They are one knob, so
            # the value the user set is one value, whichever spelling it
            # came in.
            lines.append("    @Option(name: .customLong(%s), help: %s)"
                         % (json.dumps(flag), swift_literal(help_text)))
            lines.append("    public var %sOption: Bool?" % name)
            lines.append("")
            lines.append("    /// %s (the flag spelling of the same knob)." % help_text.rstrip("."))
            lines.append("    @Flag(name: [%s], inversion: .prefixedNo, help: %s)"
                         % (", ".join(".customLong(%s)" % json.dumps(a) for a in aliases),
                            swift_literal(help_text)))
            lines.append("    public var %sFlag: Bool?" % name)
            lines.append("")
            lines.append("    /// The knob's value, whichever spelling the user used.")
            lines.append("    public var %s: Bool? { %sOption ?? %sFlag }" % (name, name, name))
            lines.append("")
        elif kind == "bool":
            lines.append("    @Flag(name: [%s], inversion: .prefixedNo, help: %s)"
                         % (names, swift_literal(help_text)))
            lines.append("    public var %s: Bool?" % name)
            lines.append("")
        else:
            # A string knob with a fixed vocabulary is an enumeration, and an
            # enumeration has always been read without regard to case. A
            # free-form string — a path, a prompt — is left exactly as typed.
            transform = ""
            allowed = knob["kind"].get("allowed") or []
            if kind == "string" and allowed:
                if all(v == v.lower() for v in allowed):
                    transform = ", transform: { $0.lowercased() }"
                else:
                    vocab = ", ".join(swift_literal(v) for v in allowed)
                    transform = ", transform: { s in [%s].first { $0.lowercased() == s.lowercased() } ?? s }" % vocab
            lines.append("    @Option(name: [%s], help: %s%s)" % (names, swift_literal(help_text), transform))
            lines.append("    public var %s: %s?" % (name, swift_type(knob["kind"])))
            lines.append("")
    lines.append("    public init() {}")
    lines.append("")
    lines.append("    /// Every binding, in registry order.")
    lines.append("    public static let bindings: [QWGKnobBinding] = [")
    for knob in cli:
        flag, _ = cli_flag(knob["id"])
        lines.append("        QWGKnobBinding(id: %s, property: %s, flag: %s),"
                     % (swift_literal(knob["id"]), swift_literal(property_name(flag)),
                        swift_literal(flag)))
    lines.append("    ]")
    lines.append("")
    lines.append("    /// The values the user set, keyed by the registry's knob id. A knob")
    lines.append("    /// the user did not set is absent from the map.")
    lines.append("    public var explicitValues: [String: QWKnobValue] {")
    lines.append("        var out: [String: QWKnobValue] = [:]")
    for knob in cli:
        flag, _ = cli_flag(knob["id"])
        name = property_name(flag)
        lines.append("        if let v = %s { out[%s] = .%s(v) }"
                     % (name, swift_literal(knob["id"]), knob["kind"]["kind"]))
    lines.append("        return out")
    lines.append("    }")
    lines.append("")
    lines.append("    /// Checks the user's values against the registry. A value the")
    lines.append("    /// registry refuses is an error here, where the error can name the")
    lines.append("    /// flag, instead of a surprise inside the engine.")
    lines.append("    public func validate() throws {")
    lines.append("        _ = try QWGKnobOptions.from(explicitValues)")
    lines.append("    }")
    lines.append("")
    lines.append("    /// The values among `values` that name a knob and fit it. A name the")
    lines.append("    /// registry does not know, or a value it refuses, is an error.")
    lines.append("    public static func from(_ values: [String: QWKnobValue]) throws -> [String: QWKnobValue] {")
    lines.append("        var out: [String: QWKnobValue] = [:]")
    lines.append("        for (id, value) in values {")
    lines.append("            guard let descriptor = QWKnobRegistry.descriptor(id) else {")
    lines.append("                throw QWKnobError.unknownKnob(id)")
    lines.append("            }")
    lines.append("            try descriptor.validate(value)")
    lines.append("            out[id] = value")
    lines.append("        }")
    lines.append("        return out")
    lines.append("    }")
    lines.append("")
    lines.append("    /// The knobs each command honours. A knob the user set that the")
    lines.append("    /// command cannot honour is refused, not ignored.")
    lines.append("    public static let honoured: [String: Set<String>] = [")
    for command, ids in CLI_COMMANDS.items():
        lines.append("        %s: [%s]," % (swift_literal(command),
                                           ", ".join(swift_literal(i) for i in ids)))
    lines.append("    ]")
    lines.append("")
    lines.append("    /// The commands that honour a knob.")
    lines.append("    public static let honouredBy: [String: [String]] = [")
    by_knob = {}
    for command, ids in CLI_COMMANDS.items():
        for knob_id in ids:
            by_knob.setdefault(knob_id, []).append(command)
    for knob_id in sorted(by_knob):
        lines.append("        %s: [%s]," % (swift_literal(knob_id),
                                            ", ".join(swift_literal(c) for c in by_knob[knob_id])))
    lines.append("    ]")
    lines.append("")
    lines.append("    /// Refuses a knob this command cannot honour. The flag the user")
    lines.append("    /// typed is named, and the commands that do honour it are named too.")
    lines.append("    public func requireHonoured(_ command: String) throws {")
    lines.append("        guard let allowed = QWGKnobOptions.honoured[command] else { return }")
    lines.append("        for id in explicitValues.keys.sorted() where !allowed.contains(id) {")
    lines.append("            let flag = QWGKnobOptions.bindings.first { $0.id == id }?.flag ?? id")
    lines.append("            let elsewhere = (QWGKnobOptions.honouredBy[id] ?? []).filter { $0 != command }")
    lines.append("            throw QWGKnobNotHonoured(command: command, flag: flag, elsewhere: elsewhere)")
    lines.append("        }")
    lines.append("    }")
    lines.append("}")
    lines.append("")
    lines.append("/// A knob the user set that the command cannot honour.")
    lines.append("public struct QWGKnobNotHonoured: Error, CustomStringConvertible {")
    lines.append("    /// The command that was asked.")
    lines.append("    public let command: String")
    lines.append("    /// The flag the user typed.")
    lines.append("    public let flag: String")
    lines.append("    /// The commands that do honour the knob.")
    lines.append("    public let elsewhere: [String]")
    lines.append("")
    lines.append("    public init(command: String, flag: String, elsewhere: [String]) {")
    lines.append("        self.command = command")
    lines.append("        self.flag = flag")
    lines.append("        self.elsewhere = elsewhere")
    lines.append("    }")
    lines.append("")
    lines.append("    public var description: String {")
    lines.append("        let hint = elsewhere.isEmpty")
    lines.append("            ? \"\"")
    lines.append("            : \"; \\(elsewhere.joined(separator: \", \")) honour\\(elsewhere.count == 1 ? \"s\" : \"\") it\"")
    lines.append("        return \"--\\(flag) is not honoured by \\(command)\\(hint)\"")
    lines.append("    }")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def registry_json(knobs):
    """The registry as JSON, so a test can hold it against the live one."""
    return json.dumps({"knobs": knobs}, indent=2, sort_keys=True) + "\n"


def main(argv):
    knobs = read_registry(REGISTRY)
    if "--json" in argv:
        sys.stdout.write(registry_json(knobs))
        return 0
    generated = emit(knobs)
    if "--stdout" in argv:
        sys.stdout.write(generated)
        return 0
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as handle:
        handle.write(generated)
    print("wrote %s (%d knobs)" % (os.path.relpath(OUTPUT, ROOT),
                                   len([k for k in knobs if "cli" in k["surfaces"]])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
