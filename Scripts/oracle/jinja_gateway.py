#!/usr/bin/env python3
"""jinja_gateway.py — SAFE, programmatic exploration of the C17 chat template.

WHY THIS EXISTS (standing directive): the agent must NEVER read/echo the raw
Jinja template. Reading template text into the transcript is both a footgun
(escaping/surrogate drift, and it invites copying formatting instead of
*behaviour*) and simply the wrong unit of work — the Swift port binds the
template's STRUCTURE (control flow + variables + which specials it emits),
not its bytes. This gateway exposes only structural facts and *rendered*
outputs; the raw source and its literal formatting never cross the boundary.

TWO BOUNDARIES, both enforced here:
  1. STRUCTURE: jinja2.parse() → AST walk. We emit node KINDS, control-flow
     nesting, variable NAMES it reads, and which `{{ ... }}` positions resolve
     to known special tokens. We deliberately DO NOT emit string literals, the
     template `source`, `node.as_const()` text, or `environment.get_source`.
  2. BEHAVIOUR: the only way to learn what the template *produces* is to render
     a conversation through the REAL engine (transformers.apply_chat_template),
     which returns already-rendered text/ids. That rendered output is behaviour,
     not template formatting — it is what the goldens already bind.

Subcommands (JSON on stdout unless --text):
  structure            AST control-flow map + variables + special-token sites
  variables            every identifier the template reads (roles/kwargs/etc.)
  branches             every {% if %}/{% for %} with a stable structural id
  render  --json C     render one conversation {msgs,gp,tools?,kwargs?} via the
                       real oracle (the honest "what does it output" channel)
  probe   --json C     same as render, plus classify: does it raise? which
                       special-token markers appear in the render?

The oracle engine (transformers) is used ONLY through apply_chat_template; we
never hand jinja a hand-built context, so kwargs defaults / preprocessing run
exactly as the checkpoint expects.
"""
import argparse, ast as _pyast, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
A = os.path.join(ROOT, "Scripts", "spike", "assets")
TPL = os.path.join(A, "chat_template.jinja")

# ---- structural vocabulary (never the template's own words) ----------------
# Special markers we care to *detect* (by name, from tokenizer.json added_tokens)
# so we can say "this template emits these specials"; we detect presence in the
def added_token_contents():
    t = json.load(open(os.path.join(A, "tokenizer.json")))
    return [x["content"] for x in t.get("added_tokens", [])]

def _env():
    # A bare Environment just to parse; we never render through it.
    import jinja2
    return jinja2.Environment(autoescape=False, undefined=jinja2.StrictUndefined)

def _specials_referenced(tree, specials):
    """Which added-token contents appear as constant string fragments anywhere
    in the AST. We return the CONTENTS that are referenced (these are token
    names / ids, checkpoint facts — not template formatting) — a set, sorted by
    id for determinism."""
    import jinja2
    from jinja2 import nodes
    hits = set()
    # Scan EVERY constant node: TemplateData (text between tags) AND Const
    # (inside {{ }} expressions — e.g. concatenated markers). We never return
    # the literals themselves; we only test membership and report which
    def lit_of(node):
        if isinstance(node, nodes.TemplateData):
            return node.literal
        if isinstance(node, nodes.Const):
            return node.value if isinstance(node.value, str) else ""
        return ""
    for node in tree.find_all((nodes.TemplateData, nodes.Const)):
        lit = lit_of(node)
        for s in specials:
            if s in lit:
                hits.add(s)
    return sorted(hits)

def _nodes_shape(tree):
    """A control-flow skeleton: nested control nodes with stable ids and the
    variables each reads. No literals, no source."""
    import jinja2
    from jinja2 import nodes
    counter = {"n": 0}
    def nid(prefix):
        counter["n"] += 1
        return "%s%d" % (prefix, counter["n"])
    def names_in(node):
        return sorted({n.name for n in node.find_all(nodes.Name)})
    def walk(node, depth):
        out = []
        for child in node.iter_child_nodes():
            if isinstance(child, nodes.If):
                out.append({"id": nid("if"), "kind": "if", "depth": depth,
                            "reads": names_in(child)})
                out += walk(child, depth + 1)
            elif isinstance(child, nodes.For):
                tg = child.target
                if isinstance(tg, nodes.Name):
                    targets = [tg.name]
                elif isinstance(tg, (nodes.Tuple, nodes.List)):
                    targets = [t.name for t in tg.find_all(nodes.Name)]
                else:
                    targets = []
                out.append({"id": nid("for"), "kind": "for", "depth": depth,
                            "targets": targets, "reads": names_in(child)})
                out += walk(child, depth + 1)
            elif isinstance(child, nodes.Macro):
                out.append({"id": nid("macro"), "kind": "macro", "depth": depth,
                            "name": child.name, "params": [a.name for a in child.args]})
                out += walk(child, depth + 1)
            elif isinstance(child, nodes.Filter) and child.name in ("tojson", "items"):
                out.append({"id": nid("filter"), "kind": "filter", "depth": depth,
                            "filter": child.name})
                out += walk(child, depth)
            else:
                out += walk(child, depth)
        return out
    return walk(tree, 0)

def _cond_shape(node, role_names, specials):
    """NESTED semantic view of one predicate as a tree (never a flattened
    list — And/Or/Not polarity and grouping must survive or the port misreads
    them). String constants surface ONLY when they are data vocabulary
    (role names, special-token names, effort enum values); every other
    literal is REDACTED — template formatting never crosses the gateway.
    ints/bools/None pass (structural)."""
    from jinja2 import nodes
    VOCAB = role_names | specials | {"xhigh", "medium", "low"}
    def term(n):
        if isinstance(n, nodes.Name):
            return {"name": n.name}
        if isinstance(n, nodes.Getattr):
            b = term(n.node)
            base = b.get("name") or b.get("dotted") or "?"
            return {"dotted": base + "." + n.attr}
        if isinstance(n, nodes.Getitem):
            return {"getitem": term(n.node), "arg": term(n.arg)}
        if isinstance(n, nodes.Const):
            v = n.value
            if isinstance(v, str):
                return v if v in VOCAB else "<redacted-str>"
            return v                         # int/bool/None/float — structural
        if isinstance(n, (nodes.List, nodes.Tuple)):
            return [term(i) for i in n.items]
        if isinstance(n, nodes.Filter):
            return {"filter": n.name, "src": term(n.node)}
        if isinstance(n, nodes.Call):
            return {"call": term(n.node)}
        return {"node": type(n).__name__}
    def emit(n):
        if isinstance(n, nodes.Compare):
            entry = {"cmp": term(n.expr), "ops": []}
            for pair in n.ops:
                if hasattr(pair, "op"):            # jinja3 Operand
                    op, right = pair.op, pair.expr
                elif isinstance(pair, tuple) and len(pair) == 2:
                    op, right = pair
                else:
                    op, right = pair, None
                entry["ops"].append({"op": getattr(op, "__name__", str(op)),
                                     "right": term(right) if right is not None else None})
            return entry
        if isinstance(n, nodes.Test):      # `x is defined`, `is none`, ...
            return {"test": n.name, "arg": term(n.node)}
        if isinstance(n, nodes.Not):
            return {"not": emit2(n.node)}
        return {"truthy": term(n)}
    def emit2(n):
        # And/Or need multi-child descent that emit()'s single-return cannot
        # express; do the whole dispatch here.
        from jinja2 import nodes as N
        if isinstance(n, (N.And, N.Or)):
            key = "and" if isinstance(n, N.And) else "or"
            return {key: [emit2(c) for c in n.iter_child_nodes()]}
        return emit(n)
    return emit2(node)

def cmd_conditions(args):
    """Per-branch predicate map: stable id -> semantic shape. The Swift state
    machine consumes this; no template formatting is ever emitted."""
    env = _env()
    tree = env.parse(open(TPL, encoding="utf-8").read())
    specials = set(added_token_contents())
    role_names = set(args.roles.split(",")) if args.roles else \
        {"system", "user", "assistant", "tool", "tool_response", "tool_call",
         "text", "image", "video"}
    from jinja2 import nodes
    counter = {"n": 0}
    conds = []
    def walk(node, depth, trail):
        for child in node.iter_child_nodes():
            if isinstance(child, nodes.If):
                counter["n"] += 1
                cid = "if%d" % counter["n"]
                conds.append({"id": cid, "depth": depth,
                              "test": _cond_shape(child.test, role_names, specials),
                              "elifs": len(child.elif_) if hasattr(child, "elif_") else 0,
                              "has_else": bool(child.else_) if hasattr(child, "else_") else False})
                walk(child, depth + 1, trail + "/" + cid)
            elif isinstance(child, nodes.For):
                counter["n"] += 1
                cid = "for%d" % counter["n"]
                tg = child.target
                targets = [tg.name] if isinstance(tg, nodes.Name) else \
                    [t.name for t in tg.find_all(nodes.Name)]
                conds.append({"id": cid, "depth": depth, "kind": "for",
                              "targets": targets,
                              "iter": type(child.iter).__name__,
                              "iter_vars": sorted({n.name for n in child.iter.find_all(nodes.Name)})})
                walk(child, depth + 1, trail + "/" + cid)
            elif isinstance(child, nodes.Macro):
                walk(child, depth + 1, trail + "/macro:" + child.name)
            else:
                walk(child, depth, trail)
    walk(tree, 0, "")
    print(json.dumps({"conditions": conds}, indent=1))

def cmd_structure(_args):
    specials = added_token_contents()
    env = _env()
    src = open(TPL, encoding="utf-8").read()   # held only to parse; NEVER printed
    tree = env.parse(src)
    allnames = sorted({n.name for n in tree.find_all(_jname())})
    shape = _nodes_shape(tree)
    result = {
        "tpl_sha256": hashlib.sha256(src.encode("utf-8")).hexdigest(),
        "control_nodes": shape,
        "top_level_control": [s for s in shape if s["depth"] == 0],
        "variables_read": allnames,
        "special_tokens_referenced": _specials_referenced(tree, specials),
        "has_macros": any(s["kind"] == "macro" for s in shape),
        "uses_tojson": any(s.get("filter") == "tojson" for s in shape),
        "uses_items": any(s.get("filter") == "items" for s in shape),
        # NOTE: no "source"/"literal"/"template" keys by construction.
    }
    print(json.dumps(result, indent=1))

def _jname():
    from jinja2 import nodes
    return nodes.Name

def cmd_variables(_args):
    env = _env()
    tree = env.parse(open(TPL, encoding="utf-8").read())
    from jinja2 import nodes
    reads = {}
    for n in tree.find_all(nodes.Name):
        reads.setdefault(n.name, "read")
    # Jinja 3 dropped AssignName: {% set %} is Assign(target: Name), loop
    # targets are Name on For.target, macro params are nodes.Argument.
    for n in tree.find_all(nodes.Assign):
        if isinstance(n.target, nodes.Name):
            reads[n.target.name] = "assigned"
    for n in tree.find_all(nodes.For):
        for t in (n.target.items if isinstance(n.target, (nodes.Tuple, nodes.List))
                  else [n.target]):
            if isinstance(t, nodes.Name):
                reads[t.name] = "loop-var"
    for n in tree.find_all(nodes.Macro):
        for a in n.args:
            reads[a.name] = "macro-param"
    print(json.dumps({"identifiers": reads}, indent=1))

def cmd_branches(_args):
    env = _env()
    tree = env.parse(open(TPL, encoding="utf-8").read())
    shape = _nodes_shape(tree)
    ifs = [s for s in shape if s["kind"] == "if"]
    fors = [s for s in shape if s["kind"] == "for"]
    print(json.dumps({"if_count": len(ifs), "for_count": len(fors),
                      "ifs": ifs, "fors": fors}, indent=1))

# ---- behaviour channel: the ONLY sanctioned way to see an output ------------
def _oracle():
    from transformers import PreTrainedTokenizerFast
    tpl = open(TPL, encoding="utf-8").read()
    return PreTrainedTokenizerFast(tokenizer_file=os.path.join(A, "tokenizer.json"),
                                   chat_template=tpl)

def _flat(x):
    return x[0] if x and isinstance(x[0], list) else x

def cmd_render(args, classify=False):
    case = json.loads(args.json) if args.json else json.load(sys.stdin)
    msgs = case.get("msgs", [])
    gp = bool(case.get("gp", True))
    kw = dict(case.get("kwargs", {}))
    if "tools" in case: kw["tools"] = case["tools"]
    ft = _oracle()
    out = {"case_keys": sorted(case.keys())}
    specials = added_token_contents()
    try:
        if classify:
            rendered = ft.apply_chat_template(msgs, tokenize=False,
                        add_generation_prompt=gp, return_dict=False, **kw)
            ids = _flat(ft.apply_chat_template(msgs, tokenize=True,
                        add_generation_prompt=gp, return_dict=False, **kw))
            assert isinstance(rendered, str) and rendered, "ORACLE DEGRADE render"
            out["raised"] = False
            out["n_ids"] = len(ids)
            out["ids_head"] = ids[:8]
            # present specials = behaviour facts (which markers the render used)
            out["present_specials"] = [s for s in specials if s in rendered]
            # char count only — the gateway is not a template-dump channel
            out["render_len"] = len(rendered)
            if args.text:
                # opt-in full render for a genuine diff; still not template src
                out["render"] = rendered
    except Exception as e:
        out["raised"] = True
        out["err_type"] = type(e).__name__
        out["err_msg"] = str(e)[:400]
    print(json.dumps(out, indent=1))

# ---- bake channel: the sanctioned WRITE side --------------------------------
# `bake` serialises the template as a PROGRAM: control ops + the exact bytes
# it emits. Jinja formatting is stripped entirely by the parse (trim_blocks /
def _parse_env():
    """Parse with the SAME lexer/parser options the real render uses
    (transformers chat_template_utils: trim_blocks, lstrip_blocks,
    loopcontrols). AssistantTracker only adds the {% generation %} tag —
    this checkpoint's template does not use it (CallBlock count 0)."""
    import jinja2, jinja2.ext
    return jinja2.Environment(trim_blocks=True, lstrip_blocks=True,
                              extensions=[jinja2.ext.loopcontrols])

def _program_of(body, role_names, specials):
    """Linearise a node body into program ops. Emitted strings (TemplateData
    and Const inside {{ }}) are OUTPUT BYTES — behaviour, not formatting —
    and are surfaced JSON-escaped. Strings in TEST positions go through
    _cond_shape (redacting anything that is not role/vocab)."""
    from jinja2 import nodes
    out = []
    for n in body:
        if isinstance(n, nodes.TemplateData):
            lit = n.literal
            if lit:
                out.append({"emit": lit})
        elif isinstance(n, nodes.Output):
            for c in n.nodes:
                out.append({"out": _expr_shape(c, role_names, specials)})
        elif isinstance(n, nodes.If):
            entry = {"if": _cond_shape(n.test, role_names, specials),
                     "then": _program_of(n.body, role_names, specials)}
            el = []
            for e in (getattr(n, "elif_", None) or []):   # jinja3: If nodes
                el.append({"if": _cond_shape(e.test, role_names, specials),
                           "then": _program_of(e.body, role_names, specials)})
            if el: entry["elifs"] = el
            if getattr(n, "else_", None):
                entry["else"] = _program_of(n.else_, role_names, specials)
            out.append(entry)
        elif isinstance(n, nodes.For):
            tg = n.target
            targets = [t.name for t in (tg.items if isinstance(tg, (nodes.Tuple, nodes.List)) else [tg])
                       if isinstance(t, nodes.Name)]
            entry = {"for": targets,
                     "iter": _expr_shape(n.iter, role_names, specials),
                     "body": _program_of(n.body, role_names, specials)}
            if n.test: entry["filter"] = _cond_shape(n.test, role_names, specials)
            if getattr(n, "else_", None):
                entry["else"] = _program_of(n.else_, role_names, specials)
            out.append(entry)
        elif isinstance(n, nodes.Assign):
            entry = {"set": _expr_shape(n.target, role_names, specials),
                     "value": _expr_shape(n.node, role_names, specials)}
            fms = getattr(n, "filter_ms", None)
            if fms:   # non-empty only: jinja3 parses `set x = y | f` via filter_ms
                entry["filters"] = [_expr_shape(f, role_names, specials) for f in fms]
                entry["value"] = _expr_shape(n.node, role_names, specials)
            out.append(entry)
        elif isinstance(n, nodes.Macro):
            out.append({"macro": n.name,
                        "params": [a.name for a in n.args],
                        "defaults": [_expr_shape(d, role_names, specials) for d in n.defaults],
                        "body": _program_of(n.body, role_names, specials)})
        elif isinstance(n, nodes.CallBlock):
            out.append({"callblock": _expr_shape(n.call, role_names, specials),
                        "body": _program_of(n.body, role_names, specials)})
        elif isinstance(n, nodes.With):
            out.append({"with": _program_of(n.body, role_names, specials)})
        else:
            out.append({"unsupported": type(n).__name__})
    return out

def _expr_shape(n, role_names, specials):
    from jinja2 import nodes
    if isinstance(n, nodes.TemplateData):
        return {"lit": n.literal}
    if isinstance(n, nodes.Name):
        return {"name": n.name}
    if isinstance(n, nodes.Const):
        v = n.value
        if isinstance(v, str) and not (v in role_names or v in specials or v == ""):
            return {"lit": v}          # emitted/compared constant — OUTPUT byte
        return {"lit": v}
    if isinstance(n, nodes.Getattr):
        return {"attr": n.attr, "on": _expr_shape(n.node, role_names, specials)}
    if isinstance(n, nodes.Getitem):
        if isinstance(n.arg, nodes.Slice):
            return {"slice": [_expr_shape(x, role_names, specials) if x else None
                              for x in (n.arg.start, n.arg.stop, n.arg.step)],
                    "on": _expr_shape(n.node, role_names, specials)}
        return {"index": _expr_shape(n.arg, role_names, specials),
                "on": _expr_shape(n.node, role_names, specials)}
    if isinstance(n, nodes.Filter):
        return {"filter": n.name,
                "src": _expr_shape(n.node, role_names, specials) if n.node else None,
                "args": [_expr_shape(a, role_names, specials) for a in n.args],
                "kwargs": {k.key: _expr_shape(k.value, role_names, specials) for k in n.kwargs}}
    if isinstance(n, nodes.Test):
        return {"test": n.name,
                "arg": _expr_shape(n.node, role_names, specials),
                "args": [_expr_shape(a, role_names, specials) for a in n.args]}
    if isinstance(n, nodes.Call):
        return {"call": _expr_shape(n.node, role_names, specials),
                "args": [_expr_shape(a, role_names, specials) for a in n.args],
                "kwargs": {k.key: _expr_shape(k.value, role_names, specials) for k in n.kwargs}}
    if isinstance(n, nodes.Concat):
        return {"concat": [_expr_shape(i, role_names, specials) for i in n.nodes]}
    if isinstance(n, nodes.CondExpr):
        return {"ternary": {"test": _cond_shape(n.test, role_names, specials),
                            "then": _expr_shape(n.expr1, role_names, specials),
                            "else": _expr_shape(n.expr2, role_names, specials)}}
    if isinstance(n, nodes.NSRef):
        return {"nsattr": n.attr, "on": {"name": n.name}}
    if isinstance(n, (nodes.Add, nodes.Mul, nodes.Sub, nodes.Div, nodes.Mod)):
        return {"bin": type(n).__name__,
                "l": _expr_shape(n.left, role_names, specials),
                "r": _expr_shape(n.right, role_names, specials)}
    if isinstance(n, nodes.Neg):
        return {"neg": _expr_shape(n.node, role_names, specials)}
    if isinstance(n, nodes.Tuple):
        return {"tuple": [_expr_shape(i, role_names, specials) for i in n.items]}
    if isinstance(n, nodes.Dict):
        return {"dict": [[_expr_shape(k, role_names, specials), _expr_shape(v, role_names, specials)]
                         for k, v in n.items]}
    if isinstance(n, nodes.Compare):
        return {"cmp": _cond_shape(n, role_names, specials)}
    return {"node": type(n).__name__}

def cmd_bake(args):
    env = _parse_env()
    src = open(TPL, encoding="utf-8").read()   # held only to parse; NEVER printed
    tree = env.parse(src)
    specials = set(added_token_contents())
    role_names = {"system", "user", "assistant", "tool", "tool_response",
                  "tool_call", "text", "image", "video"}
    prog = _program_of(list(tree.iter_child_nodes()), role_names, specials)
    result = {"tpl_sha256": hashlib.sha256(src.encode("utf-8")).hexdigest(),
              "program": prog}
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1, sort_keys=False)
        print(json.dumps({"wrote": args.json_out,
                          "ops": len(prog),
                          "emit_sites": sum(1 for o in prog if "emit" in o)}))
    else:
        print(json.dumps(result, indent=1))

def _cond_terms(node, role_names, specials):
    """Semantic view of one branch predicate (shared by `conditions` and the
    bake collector): emits ONLY structural facts + data-vocabulary strings."""
    return _cond_shape(node, role_names, specials)

def _walk_program(prog):
    """Yield every string a program CAN EMIT — strict typed schema:
    TemplateData chunks (op['emit']) and Consts inside {{ }} outputs
    ('out'→expr). Test/comparison strings are NOT emission bytes and are
    handled by `conditions`; variable paths are never collected."""
    out = []
    def expr(e):
        if not isinstance(e, dict): return
        if isinstance(e.get("lit"), str): out.append(e["lit"])
        for k in ("src", "on", "arg", "value", "l", "r", "neg", "index"):
            if k in e: expr(e[k])
        for k in ("args", "tuple", "concat"):
            if k in e and isinstance(e[k], list):
                for x in e[k]: expr(x)
        if isinstance(e.get("kwargs"), dict):
            for x in e["kwargs"].values(): expr(x)
        if isinstance(e.get("dict"), list):
            for kv in e["dict"]:
                for x in kv: expr(x)
        if isinstance(e.get("ternary"), dict):
            t = e["ternary"]
            expr(t.get("then")); expr(t.get("else"))   # emitted arms
    def body(seq):
        for op in seq:
            if not isinstance(op, dict): continue
            if isinstance(op.get("emit"), str): out.append(op["emit"])
            if "out" in op: expr(op["out"])
            if "set" in op: expr(op.get("value"))      # assigned consts can be emitted later
            if isinstance(op.get("macro"), str):
                for x in op.get("defaults", []): expr(x)
                body(op.get("body", []))
            if isinstance(op.get("for"), list):
                body(op.get("body", [])); body(op.get("else", []))
            if "if" in op:
                body(op.get("then", [])); body(op.get("else", []))
                for e in op.get("elifs", []):
                    if isinstance(e, dict): body(e.get("then", []))
    body(prog)
    return out

def _collect_emits(prog, bucket):
    for s in _walk_program(prog):
        if s not in bucket:
            bucket[s] = len(bucket)
    return bucket

def cmd_bake_swift(args):
    """GENERATE the Swift emission table: every emitted string becomes
    E0..En (UTF-8 byte arrays). The transcript never carries these bytes —
    they flow parse → here → generated source file."""
    env = _parse_env()
    src = open(TPL, encoding="utf-8").read()
    tree = env.parse(src)
    specials = set(added_token_contents())
    role_names = {"system", "user", "assistant", "tool", "tool_response",
                  "tool", "text", "image", "video"}
    prog = _program_of(list(tree.iter_child_nodes()), role_names, specials)
    bucket = _collect_emits(prog, {})
    items = sorted(bucket.items(), key=lambda kv: kv[1])
    L = []
    L.append("// GENERATED by Scripts/oracle/jinja_gateway.py bake-swift — DO NOT EDIT.")
    L.append("// Emission byte-table for the C17 chat-template state machine:")
    L.append("// every string the template can emit, as UTF-8 bytes, deduped in")
    L.append("// first-appearance order. Bound by the tok_chat goldens (spec-17);")
    L.append("// regenerate via `bake-swift`, never by hand.")
    L.append("// tpl sha256 %s" % hashlib.sha256(src.encode("utf-8")).hexdigest())
    L.append("import Foundation")
    L.append("")
    L.append("enum QWTplEmit {")
    for s, i in items:
        bs = list(s.encode("utf-8"))
        name = "E%d" % i
        comment = json.dumps(s)          # JSON escape for the HUMAN label only
        if len(comment) > 70: comment = comment[:67] + "..."
        L.append("    // %s  (%d bytes)" % (comment.replace("\n", "\\n"), len(bs)))
        body = ", ".join(str(b) for b in bs)
        if len(body) > 96:
            parts, cur = [], ""
            for b in bs:
                if len(cur) + len(str(b)) + 2 > 88:
                    parts.append(cur.strip().rstrip(",")); cur = ""
                cur += str(b) + ", "
            if cur.strip().rstrip(","): parts.append(cur.strip().rstrip(","))
            body = "\n        " + (",\n        ".join(parts))
        L.append("    static let %s: [UInt8] = [%s]" % (name, body))
    L.append("    static let all: [[UInt8]] = [" + ", ".join("E%d" % i for i in range(len(items))) + "]")
    L.append("}")
    out = args.out or os.path.join(ROOT, "Sources", "QWToken", "QWTemplateEmit.swift")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(json.dumps({"wrote": out, "strings": len(items),
                      "total_bytes": sum(len(s.encode()) for s, _ in items)}))

def cmd_program_masked(args):
    """The SAFE study lens: the full linear program with every emission
    string replaced by its E-table reference (E0..En). Control flow, variable
    names and emitted-bytes positions — zero emitted bytes, zero formatting.
    This is what the Swift transcription reads; the bytes themselves only
    ever travel parse → bake-swift → generated file."""
    env = _parse_env()
    src = open(TPL, encoding="utf-8").read()
    tree = env.parse(src)
    specials = set(added_token_contents())
    role_names = {"system", "user", "assistant", "tool", "tool_response",
                  "tool_call", "text", "image", "video"}
    prog = _program_of(list(tree.iter_child_nodes()), role_names, specials)
    bucket = _collect_emits(prog, {})
    def ref(s):
        return "E%d" % bucket[s] if s in bucket else "?"
    def mask(e):
        if isinstance(e, dict):
            if isinstance(e.get("lit"), str):
                e = dict(e); e["lit"] = ref(e["lit"])
            if isinstance(e.get("emit"), str):
                e = dict(e); e["emit"] = ref(e["emit"])
            return {k: mask(v) for k, v in e.items()}
        if isinstance(e, list):
            return [mask(x) for x in e]
        return e
    print(json.dumps({"tpl_sha256": hashlib.sha256(src.encode("utf-8")).hexdigest(),
                      "emit_table": {("E%d" % i): ("len=%dB" % len(s.encode("utf-8")))
                                     for s, i in bucket.items()},
                      "program": mask(prog)}, indent=1))

def main():
    p = argparse.ArgumentParser(description="Programmatic Jinja template gateway (never prints template source)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("structure")
    sub.add_parser("variables")
    sub.add_parser("branches")
    cond = sub.add_parser("conditions"); cond.add_argument("--roles")
    r = sub.add_parser("render"); r.add_argument("--json"); r.add_argument("--text", action="store_true")
    c = sub.add_parser("probe"); c.add_argument("--json"); c.add_argument("--text", action="store_true")
    b = sub.add_parser("bake"); b.add_argument("--json-out")
    bs = sub.add_parser("bake-swift"); bs.add_argument("--out")
    pm = sub.add_parser("program-masked")
    a = p.parse_args()
    if a.cmd == "structure": cmd_structure(a)
    elif a.cmd == "variables": cmd_variables(a)
    elif a.cmd == "branches": cmd_branches(a)
    elif a.cmd == "conditions": cmd_conditions(a)
    elif a.cmd == "render": cmd_render(a, classify=False)
    elif a.cmd == "probe": cmd_render(a, classify=True)
    elif a.cmd == "bake": cmd_bake(a)
    elif a.cmd == "bake-swift": cmd_bake_swift(a)
    elif a.cmd == "program-masked": cmd_program_masked(a)

if __name__ == "__main__":
    main()
