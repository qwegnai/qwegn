#!/usr/bin/env python3
"""Replay the exact Flash-Next JSON failures against a user-managed loopback server."""
import argparse
import hashlib
import http.client
import json
import time

from pathlib import Path
from paired_http_perf import endpoint


MARKER = ("<think>kept</think> </think:opensource> <|channel>thought <|channel|>final "
          "<|content_thinking|> to=self<|message|> <tool_call>literal</tool_call>")
MARKER_SCHEMA = {
    "type": "object", "properties": {"note": {"type": "string", "enum": [MARKER]}},
    "required": ["note"], "additionalProperties": False,
}
NESTED_SCHEMA = {
    "type": "object", "required": ["user"], "additionalProperties": False,
    "properties": {"user": {
        "type": "object", "required": ["name", "pets"], "additionalProperties": False,
        "properties": {
            "name": {"type": "string"},
            "pets": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
        },
    }},
}


def request_for(case, model, seed, top_k=40, top_p=0.9):
    marker = case == "marker"
    request = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": ("Return the JSON object required by the schema." if marker
                        else 'Return ONLY raw JSON: {"user": {"name": string, '
                             '"pets": [string, string]}}. No fences, no prose.'),
        }],
        "temperature": 0 if marker else 0.7,
        "top_k": top_k,
        "top_p": top_p,
        "repetition_penalty": 1.0,
        "max_tokens": 256 if marker else 192,
        "enable_thinking": False,
        "seed": seed,
    }
    if case != "nested-prompt":
        request["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": case, "strict": True,
                            "schema": MARKER_SCHEMA if marker else NESTED_SCHEMA},
        }
    return request


def grade(case, text):
    try:
        value = json.loads(text)
    except (ValueError, TypeError) as error:
        return f"not raw valid JSON: {error}"
    if not isinstance(value, dict):
        return "root is not an object"
    if case == "marker":
        return None if value == {"note": MARKER} else "marker enum differs from required bytes"
    user = value.get("user")
    if not isinstance(user, dict) or not isinstance(user.get("name"), str):
        return "missing string user.name"
    pets = user.get("pets")
    if (not isinstance(pets, list) or len(pets) != 2
            or any(not isinstance(pet, str) for pet in pets)):
        return "user.pets must contain exactly two strings"
    if case == "nested-schema" and (set(value) != {"user"} or set(user) != {"name", "pets"}):
        return "schema forbids additional properties"
    return None


def send(target, payload, timeout):
    host, port, path = target
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    start = time.monotonic()
    try:
        connection.request("POST", path, json.dumps(payload), {"Content-Type": "application/json"})
        response = connection.getresponse()
        body = response.read()
        if response.status != 200:
            raise ValueError(f"HTTP {response.status}: {body.decode('utf-8', errors='replace')[:300]}")
        result = json.loads(body)
        choice = result["choices"][0]
        text = choice["message"]["content"]
        if not isinstance(text, str):
            raise ValueError("response content is not a string")
        return {
            "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "finish_reason": choice["finish_reason"], "usage": result.get("usage"),
            "duration_seconds": time.monotonic() - start,
        }
    finally:
        connection.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--case", choices=["marker", "nested-prompt", "nested-schema"], required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.trials <= 0 or not 0 < args.timeout < float("inf"):
        parser.error("trials and timeout must be positive and timeout finite")
    if args.top_k < 0 or not 0 < args.top_p <= 1:
        parser.error("top-k must be nonnegative and top-p must be in (0, 1]")
    try:
        target = endpoint(args.endpoint)
        if args.output.exists():
            raise ValueError(f"output already exists: {args.output}")
        trials = []
        for index in range(args.trials):
            seed = args.seed if args.case == "marker" else args.seed + index
            request = request_for(args.case, args.model, seed, args.top_k, args.top_p)
            result = send(target, request, args.timeout)
            result["request"] = request
            result["failure"] = grade(args.case, result["text"])
            trials.append(result)
        passed = sum(row["failure"] is None for row in trials)
        report = {
            "version": 1, "case": args.case, "endpoint": args.endpoint,
            "passed": passed, "total": args.trials, "trials": trials,
            "raw_token_ids": None,
            "capture_limit": "Chat API exposes text, not raw generated IDs. "
                             "No logprobs requested: preserve the speculative execution path.",
        }
        encoded = json.dumps(report, indent=2, allow_nan=False)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(encoded + "\n")
        print(f"{args.case}: {passed}/{args.trials} -> {args.output}")
        for index, row in enumerate(trials, 1):
            if row["failure"]:
                print(f"  trial {index}: {row['failure']}; finish={row['finish_reason']}")
        return 0 if passed == args.trials else 1
    except (OSError, http.client.HTTPException, ValueError, KeyError, TypeError, IndexError) as error:
        parser.exit(2, f"probe_quality: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
