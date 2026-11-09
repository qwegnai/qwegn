"""Sweep the published Qwen3 configs and record what the engine does with them.

The published-config audit behind the support flip is a **runnability**
claim: it says how many of the checkpoints a family publishes the engine
can serve. The claim is only worth anything if the list behind it is in
the repo, so this script writes it down:

  - every model the **Qwen organisation** publishes (the `author=Qwen`
    listing on the Hugging Face Hub),
  - the `config.json` of each one that ships one,
  - the subset whose `model_type` is `qwen3` or `qwen3_moe`.

The verdicts are **not** computed here: the engine's own decoder decides
them, and `Tests/QWCoreTests/QWStandardPublishedConfigTests.swift` runs
the recorded configs through `QWStandardConfigDecoder` and asserts the
counts the plan and the spec quote. This script only records the inputs.

    python3 Scripts/audit_published_configs.py \
        --out Tests/QWCoreTests/Fixtures/qwen3_published_configs.json

Re-running it against a live Hub may find more checkpoints than the
recorded set; the counts in the plan and the spec then have to be
re-checked against the new record.
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://huggingface.co/api/models"
HUB = "https://huggingface.co"
# The families this audit covers. The engine serves them from the standard
# path, so a published checkpoint of one of them is a runnability claim.
FAMILIES = ("qwen3", "qwen3_moe")
# The organisation whose models are "published" for this audit. A community
# fine-tune is not a released checkpoint, so it is not in the set.
AUTHOR = "Qwen"
UA = "qwegine-published-config-audit/1"


def fetch_json(url: str, timeout: int = 30):
    """One JSON GET, or None when the hub answers with an error."""
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError,
            TimeoutError):
        return None


def listing() -> list:
    """Every model the organisation publishes, following the hub's pages."""
    out = []
    url = f"{API}?author={AUTHOR}&limit=1000"
    seen = set()
    while url and len(out) < 5000:
        request = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(request, timeout=60) as response:
            page = json.load(response)
            link = response.headers.get("Link", "")
        out.extend(page)
        for m in page:
            seen.add(m.get("id"))
        if not page:
            break
        nxt = None
        for part in link.split(","):
            if 'rel="next"' in part:
                nxt = part.split(";")[0].strip().strip("<>")
        if not nxt:
            break
        url = nxt
    return [m for m in out if m.get("id") not in seen or True]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="seconds to wait between config fetches")
    args = ap.parse_args()

    models = listing()
    print(f"{AUTHOR} publishes {len(models)} models")
    configs = {}
    shipped = 0
    for m in models:
        model_id = m.get("id")
        if not model_id:
            continue
        url = f"{HUB}/{model_id}/resolve/main/config.json"
        config = fetch_json(url)
        if not isinstance(config, dict):
            continue
        shipped += 1
        family = config.get("model_type")
        if family not in FAMILIES:
            continue
        configs[model_id] = config
        if args.sleep:
            time.sleep(args.sleep)
    print(f"{shipped} ship a config.json; {len(configs)} declare {FAMILIES}")
    counts = {}
    for family in FAMILIES:
        counts[family] = sum(1 for c in configs.values()
                              if c.get("model_type") == family)
    print("by family:", counts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(configs, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out} ({args.out.stat().st_size} B)")


if __name__ == "__main__":
    main()
