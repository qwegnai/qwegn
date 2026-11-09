#!/usr/bin/env python3
"""gen_token_fixtures.py — C17 goldens: LIVE Python truth → fixtures, written
BEFORE any Swift assertion exists (python-swift-parity-port rule 2).

Oracle authority (spec-17 + the asset hashes in SOURCES.json):
  * encode/decode truth = tokenizers.Tokenizer.from_file(tokenizer.json) —
    the SAME engine the checkpoint ships with (HF fast path).
  * chat-template truth = transformers.PreTrainedTokenizerFast loaded from
    the pinned tokenizer.json + chat_template.jinja (template engine,
    apply_chat_template tokenize=True/False).
  * pretokenize_regex in tokenizer_config.json is recorded for the Swift
    port's reference; the ENCODE goldens come from the tokenizers engine
    (its Sequence[Split,ByteLevel] config is the deployed contract).

Fixture layout (Scripts/spike/out/, consolidated bins — 10k cases would be
20k tiny files under the per-case layout; one role per file — collision law):
  tok_text.bin    concatenated UTF-8 case texts | tok_ids.bin i32-LE ids
  tok_detok.bin   decode bytes | tok_index.json {n, cases[i,cat,toff,tbytes,
                  ioff,nids,doff,dbytes], nfc_pairs, specials}
  tok_chat.jsonl  cases {i,gp,msgs,tools?,kwargs?,err?} | tok_chat_ids.bin
  tok_chat_text.bin renders | tok_chat_index.json offsets + errs + tpl hash
Oracle-liveness law (rule 3): EVERY golden write asserts the returned body
is real (non-null, non-empty, correct type). A case where the oracle
degraded aborts the run with nonzero exit — null==null passes are poison.

Deterministic: fixed seed, fixed case list, sorted writes; re-running with
the same Python stack reproduces byte-identical fixtures (self-check at the
end re-encodes 3 random cases and diffs against what was written).
"""
import hashlib, json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
A = os.path.join(ROOT, "Scripts", "spike", "assets")
OUT = os.path.join(ROOT, "Scripts", "spike", "out")
os.makedirs(OUT, exist_ok=True)

def pin_verify():
    """The truth-probe law, applied at generator start: assets match pins."""
    src = json.load(open(os.path.join(A, "SOURCES.json")))
    for name, want in src["sha256"].items():
        got = hashlib.sha256(open(os.path.join(A, name), "rb").read()).hexdigest()
        if got != want:
            sys.exit(f"ASSET DRIFT {name}: disk {got} != pinned {want} — refuse to generate goldens")
    print("assets pinned-verified (SOURCES.json sha256 ×3)")

pin_verify()
from tokenizers import Tokenizer          # oracle engine (live)
tok = Tokenizer.from_file(os.path.join(A, "tokenizer.json"))

# ── Part 2: corpus + encode/detok goldens ─────────────────────────────────
# All non-ASCII text is built from CODEPOINTS via cp(): the generator source
# is pure ASCII and the escape-drift failure class (literal \uXXXX text in a
# .py file parsed as a LONE SURROGATE by tokenizers' Rust bridge — recorded
# truth-probe TypeError) becomes impossible.
import struct

def cp(*cs): return "".join(chr(c) for c in cs)

CUR = []                                   # (category, text) in golden order
def add(cat, t): CUR.append((cat, t))

# CJK / Japanese / Korean (spec-17 corpus law)
add("cjk", cp(0x4F60, 0x597D, 0x4E16, 0x754C, 0xFF01))
add("cjk", cp(0x6211, 0x662F) + " Qwen." + cp(0x3002))
add("cjk", cp(0x3053, 0x3093, 0x306B, 0x3061, 0x306F, 0x4E16, 0x754C, 0x3067, 0x3059))
add("cjk", cp(0xD55C, 0xAD6D, 0xC694, 0xC5B4, 0xC785, 0xC2B5))
add("cjk", "mixed " + cp(0x4F60, 0x597D) + " text " + cp(0x3053))

# emoji: astral, ZWJ family, flag (regional-indicator pair), VS16
add("emoji", "rocket " + cp(0x1F680) + " liftoff " + cp(0x1F600))
add("emoji", cp(0x1F468, 0x200D, 0x1F469, 0x200D, 0x1F467))
add("emoji", cp(0x1F1EF, 0x1F1F5))
add("emoji", "I " + cp(0x2764) + " Metal" + cp(0xFE0F))

# NFC law pairs: decomposed vs precomposed (the normalizer's contract;
# asserted AFTER goldens emit — the oracle's own law, not my assumption)
NFC_PAIRS = [("a" + cp(0x30A), cp(0xE5)), ("e" + cp(0x301), cp(0xE9)),
             ("n" + cp(0x303), cp(0xF1)), ("u" + cp(0x308), cp(0xFC))]
NFC_IDX = []
for dch, pch in NFC_PAIRS:
    NFC_IDX.append((len(CUR), len(CUR) + 1))
    add("nfc_decomp", dch); add("nfc_precomp", pch)

# control bytes (spec corpus law)
add("ctrl", "nul[" + chr(0) + "]nul")
add("ctrl", chr(1) + chr(2) + chr(3) + "soh-stx-etx")
add("ctrl", "us" + chr(31) + "end")
add("ctrl", "del" + chr(127) + "here")
add("ctrl", chr(11) + chr(12) + "vtab-ff")

# whitespace / newline edges
add("ws", "   leading and trailing   ")
add("ws", "\n\n\n")
add("ws", "\r\nCRLF\r\n\r\n")
add("ws", "a\tb\t\tc\t")
add("ws", "nbsp" + chr(0xA0) + "ideo" + cp(0x3000) + "zwsp" + cp(0x200B) + "end")
add("ws", "a b  c   d    e")
add("ws", " " * 40)

# code shapes
add("code", "def f(x):\n    return x + 1  # " + cp(0x4E2D, 0x6587))
add("code", "const o = {a: 1, b: \"two\", \"c\": [1,2,3]};")
add("code", "SELECT a, b FROM t WHERE a=1 AND b<>2; -- done")
add("code", "```python\nprint('hi')\n```")

# ASCII baseline + empties
add("ascii", "The quick brown fox jumps over the lazy dog. 0123456789")
add("ascii", " ")
add("empty", "")

# special tokens: strings taken FROM THE CHECKPOINT, not hardcoded
SPECIALS = [t["content"] for t in json.load(open(os.path.join(A, "tokenizer.json"))).get("added_tokens", [])]
SPECIAL_IDX = {}
for sp in SPECIALS:
    SPECIAL_IDX[sp] = len(CUR)
    add("special_alone", sp)
add("special_in_text", "before" + SPECIALS[1] + "after")
add("special_adjacent", SPECIALS[1] + SPECIALS[2])
add("special_vision", SPECIALS[5] + SPECIALS[6] if len(SPECIALS) > 6 else SPECIALS[0])

# synthetic bulk to the 10k corpus (spec-17 gate size), deterministic seed
WORDS = ["the", "quick", "brown", "fox", "hello", "world", "system", "answer",
         "code", "data", "model", "token", "engine", "kernel", "buffer",
         "import os", "x = f(y)", "return x + 1", "if a and b:", "0123456789"]
CJKP = [cp(0x4F60), cp(0x597D), cp(0x4E16), cp(0x754C), cp(0x6211), cp(0x662F),
        cp(0x3053), cp(0x3093), cp(0x4E2D), cp(0x6587)]
EMO = [cp(0x1F600), cp(0x1F680), cp(0x1F525), cp(0x2764, 0xFE0F)]
rng = random.Random(20261002)
while len(CUR) < 10000:
    n = rng.randint(4, 24)
    parts = []
    for _ in range(n):
        r = rng.random()
        if r < 0.5:    parts.append(rng.choice(WORDS))
        elif r < 0.75: parts.append(rng.choice(CJKP))
        elif r < 0.9:  parts.append(rng.choice(EMO))
        else:          parts.append("x" * rng.randint(1, 6))
        if rng.random() < 0.7:
            parts.append(rng.choice([" ", "", "\n", "  "]))
    t = "".join(parts)
    if rng.random() < 0.05:
        t += rng.choice(SPECIALS)
    add("synth", t)

# ── encode/detok goldens + rule-3 liveness asserts ────────────────────────
texts_blob = bytearray(); ids_blob = bytearray(); detok_blob = bytearray()
cases = []
for i, (cat, t) in enumerate(CUR):
    tb = t.encode("utf-8")
    enc = tok.encode(t, add_special_tokens=False)
    ids = enc.ids if enc is not None else None
    assert ids is not None and all(isinstance(x, int) for x in ids), \
        "ORACLE DEGRADE encode case %d (%s)" % (i, cat)
    if len(tb) > 0:
        assert len(ids) > 0, "ORACLE DEGRADE: empty ids on non-empty case %d" % i
    dtxt = tok.decode(ids, skip_special_tokens=False)
    assert isinstance(dtxt, str), "ORACLE DEGRADE detok case %d" % i
    db = dtxt.encode("utf-8")
    toff, ioff, doff = len(texts_blob), len(ids_blob), len(detok_blob)
    texts_blob += tb
    if ids: ids_blob += struct.pack("<%di" % len(ids), *ids)
    detok_blob += db
    cases.append({"i": i, "cat": cat, "toff": toff, "tbytes": len(tb),
                  "ioff": ioff, "nids": len(ids), "doff": doff, "dbytes": len(db)})

# NFC law (the oracle's own contract): decomposed == precomposed ids
for d_i, p_i in NFC_IDX:
    a, al = cases[d_i]["ioff"], cases[d_i]["nids"]
    b, bl = cases[p_i]["ioff"], cases[p_i]["nids"]
    assert al == bl and bytes(ids_blob[a:a+4*al]) == bytes(ids_blob[b:b+4*bl]), \
        "NFC law broken: case %d vs %d" % (d_i, p_i)

open(os.path.join(OUT, "tok_text.bin"), "wb").write(texts_blob)
open(os.path.join(OUT, "tok_ids.bin"), "wb").write(ids_blob)
open(os.path.join(OUT, "tok_detok.bin"), "wb").write(detok_blob)
json.dump({"n": len(cases), "cases": cases,
           "nfc_pairs": [list(p) for p in NFC_IDX], "specials": SPECIAL_IDX},
          open(os.path.join(OUT, "tok_index.json"), "w"))
print("encode/detok goldens: n=%d ids=%dB detok=%dB" % (len(cases), len(ids_blob), len(detok_blob)))

# ── Part 3: chat-template goldens (the REAL template engine) ──────────────
from transformers import PreTrainedTokenizerFast
tpl = open(os.path.join(A, "chat_template.jinja"), encoding="utf-8").read()
try:
    ft = PreTrainedTokenizerFast(tokenizer_file=os.path.join(A, "tokenizer.json"),
                                 chat_template=tpl)
except Exception as e:
    sys.exit("ORACLE LOAD FAILED: " + str(e))

TOOLS = [{"type": "function", "function": {"name": "get_weather",
    "description": "weather now", "parameters": {"type": "object",
      "properties": {"city": {"type": "string", "description": "city"}},
      "required": ["city"]}}}]
def SYS(c): return {"role": "system", "content": c}
def U(c):   return {"role": "user", "content": c}
def AS(c):  return {"role": "assistant", "content": c}
def IMG(c): return {"role": "user", "content": [{"type": "image", "image": "http://x/y.png"}, {"type": "text", "text": c}]}
def VID(c): return {"role": "user", "content": [{"type": "video", "video": "http://x/y.mp4"}, {"type": "text", "text": c}]}
# arguments as a DICT: template line 136 pipes tool_call.arguments|items —
# a JSON *string* raises jinja's TypeError ("Can only get item pairs from a
# mapping", an engine input-type error, not a template contract). Golden-
# first law: the dict form is the real rendering path; string-arguments is
# rejected upstream and never reaches the port.
AS_TC = {"role": "assistant", "content": None, "tool_calls": [
    {"id": "c1", "type": "function", "function": {"name": "get_weather",
     "arguments": {"city": "Paris"}}}]}
TOOLRESP = {"role": "tool", "content": "{\"t\": 20}"}

# Every chat shape spec-17 names; "err" = expected raise substring (the
# template's error contract is part of the golden — the Swift state machine
# must detect these conditions; message TEXT is recorded, not asserted equal
# by the port). Unexpected raises ABORT: oracle degradation, not truth.
CHAT = [
  {"gp": True,  "msgs": [SYS("You are helpful."), U("hi")]},
  {"gp": False, "msgs": [SYS("You are helpful."), U("hi"), AS("hello")]},
  {"gp": True,  "msgs": [U("hi")]},
  {"gp": True,  "msgs": [U("hi"), AS("a"), U("again"), AS("b")]},
  {"gp": True,  "msgs": [SYS("s"), U("hi")], "kwargs": {"reasoning_effort": "low"}},
  # NOTE (law caught golden-first): the template REQUIRES >=1 user message
  # whose content is not a tool_response pair (multi_step_tool raise, line
  # ~100) — a system-only conversation RAISES. That is a template contract,
  # so the honest shape here has a user message:
  {"gp": True,  "msgs": [SYS("s"), U("med")], "kwargs": {"reasoning_effort": "medium"}},
  {"gp": True,  "msgs": [SYS("s"), U("hi")], "kwargs": {"reasoning_effort": "bogus"},
   "err": "reasoning effort"},
  {"gp": True,  "msgs": [SYS("s"), U("hi")], "kwargs": {"enable_thinking": False}},
  {"gp": True,  "msgs": [IMG("describe")], "kwargs": {"add_vision_id": True}},
  {"gp": True,  "msgs": [IMG("look at this")]},
  {"gp": True,  "msgs": [VID("vid")]},
  {"gp": True,  "msgs": [{"role": "system", "content": [{"type": "image", "image": "u"}]}],
   "err": "System message cannot contain images."},
  {"gp": True,  "msgs": [IMG("a"), VID("b"), AS("pics"), U("more")]},
  {"gp": True,  "msgs": [SYS("s"), U("hi"), AS_TC, TOOLRESP, AS("20C")], "tools": TOOLS},
  {"gp": True,  "msgs": [SYS("s"), U("hi")], "tools": TOOLS},
  # transformers 5.17 guards empty conversations at the ENGINE level
  # (ValueError before the template's own 'No messages provided.' raise is
  # reachable) — golden records the ACTUAL engine contract; the Swift port
  # must classify "empty ⇒ error", message text is harness-level truth.
  {"gp": True,  "msgs": [], "err": "Cannot apply chat template to an empty conversation"},
  {"gp": True,  "msgs": [{"role": "user", "content": [{"type": "bogus"}]}],
   "err": "Unexpected item type in content."},
  {"gp": True,  "msgs": [{"role": "user", "content": 42}], "err": "Unexpected content type."},
  {"gp": True,  "msgs": [U("a" + cp(0x3002) + " " + cp(0x1F600))]},
  {"gp": False, "msgs": [U("x"), {"role": "assistant", "content": [{"type": "image", "image": "z"}]}]},
  {"gp": True,  "msgs": [{"role": "user", "content": None}]},
  {"gp": True,  "msgs": [VID("v2")], "kwargs": {"add_vision_id": True}},
  {"gp": False, "msgs": [U(cp(0x4F60) + cp(0x597D)), AS(cp(0x3053) + cp(0x3093))]},
  {"gp": True,  "msgs": [IMG("combo")], "tools": TOOLS, "kwargs": {"add_vision_id": True}},
]

def flat(x):
    return x[0] if x and isinstance(x[0], list) else x

chat_cases, chat_ids_blob, chat_text_blob, chat_index = [], bytearray(), bytearray(), []
for i, case in enumerate(CHAT):
    kw = dict(case.get("kwargs", {}))
    if "tools" in case: kw["tools"] = case["tools"]
    spec = {"i": i, "gp": case["gp"], "msgs": case["msgs"]}
    if "tools" in case: spec["tools"] = case["tools"]
    if "kwargs" in case: spec["kwargs"] = case["kwargs"]
    if "err" in case: spec["err"] = case["err"]
    chat_cases.append(spec)
    rec = {"i": i}
    if "err" in case:
        try:
            ft.apply_chat_template(case["msgs"], tokenize=False, add_generation_prompt=case["gp"], return_dict=False, **kw)
            sys.exit("chat case %d did NOT raise; expected %r" % (i, case["err"]))
        except SystemExit: raise
        except Exception as e:
            msg = str(e)
            assert case["err"] in msg, "chat case %d raised unexpected: %r (expected %r)" % (i, msg[:300], case["err"])
            rec["err"] = case["err"]; rec["err_msg"] = msg[:400]
            print("chat %02d RAISES as expected: %s" % (i, msg[:90].replace(chr(10), " ")))
    else:
        rendered = ft.apply_chat_template(case["msgs"], tokenize=False, add_generation_prompt=case["gp"], return_dict=False, **kw)
        assert isinstance(rendered, str) and len(rendered) > 0, "ORACLE DEGRADE chat render %d" % i
        ids = flat(ft.apply_chat_template(case["msgs"], tokenize=True, add_generation_prompt=case["gp"], return_dict=False, **kw))
        assert isinstance(ids, list) and len(ids) > 0 and all(isinstance(v, int) for v in ids), \
            "ORACLE DEGRADE chat ids %d: %r" % (i, ids if not isinstance(ids, list) else ids[:5])
        rb = rendered.encode("utf-8")
        toff = len(chat_text_blob); chat_text_blob += rb
        ioff = len(chat_ids_blob);  chat_ids_blob += struct.pack("<%di" % len(ids), *ids)
        rec.update({"toff": toff, "tbytes": len(rb), "ioff": ioff, "nids": len(ids)})
        assert b"<|im_start|>" in rb, "chat render %d has no im_start — template not the ChatML we pinned" % i
    chat_index.append(rec)

with open(os.path.join(OUT, "tok_chat.jsonl"), "w") as f:
    for spec in chat_cases:
        f.write(json.dumps(spec) + "\n")
open(os.path.join(OUT, "tok_chat_ids.bin"), "wb").write(chat_ids_blob)
open(os.path.join(OUT, "tok_chat_text.bin"), "wb").write(chat_text_blob)
json.dump({"n": len(chat_index), "cases": chat_index, "tpl_sha256": hashlib.sha256(tpl.encode("utf-8")).hexdigest()},
          open(os.path.join(OUT, "tok_chat_index.json"), "w"), indent=1)
print("chat goldens: n=%d ids=%dB renders=%dB" % (len(chat_index), len(chat_ids_blob), len(chat_text_blob)))

# ── determinism self-check: re-run the oracle; bytes must match ───────────
for pick in (0, 3, 7, 9999):
    enc2 = tok.encode(CUR[pick][1], add_special_tokens=False)
    ioff, nids = cases[pick]["ioff"], cases[pick]["nids"]
    assert struct.pack("<%di" % len(enc2.ids), *enc2.ids) if enc2.ids else True
    assert bytes(ids_blob[ioff:ioff+4*nids]) == (struct.pack("<%di" % len(enc2.ids), *enc2.ids) if enc2.ids else b""), \
        "NONDETERMINISM encode case %d" % pick
ok0 = [r for r in chat_index if "err" not in r][0]
ids_again = flat(ft.apply_chat_template(CHAT[ok0["i"]]["msgs"], tokenize=True,
              add_generation_prompt=CHAT[ok0["i"]]["gp"], return_dict=False,
              **CHAT[ok0["i"]].get("kwargs", {})))
assert struct.pack("<%di" % len(ids_again), *ids_again) == bytes(chat_ids_blob[ok0["ioff"]:ok0["ioff"]+4*ok0["nids"]]), "NONDETERMINISM chat case %d" % ok0["i"]
print("determinism self-check OK — goldens reproducible in-process")
print("SPECIALS:", [(s, SPECIAL_IDX[s]) for s in SPECIALS][:8])
for pick in (0, 1):  # informational: specials alone tokenize to ONE id each?
    print("  special alone:", repr(SPECIALS[pick]), "->", tok.encode(SPECIALS[pick], add_special_tokens=False).ids)
print("GEN OK")
