"""Attempt counts of spread runs. Offline, no model call. Reads each run's calls.jsonl and counts, by role and step type
(the tag's prefix), the physical attempts: records of a request that was sent (they carry `dispatch`: first,
transport_retry, schema_retry or recovery). Records of a request not sent and the bookkeeping records (cut recovery:
`recovery` a string, an attempt's record may carry the event itself; how a 429 wait ended: `rate_limit_wait`,
llm.Caller.wait_429) are counted apart, outside the denominator:

    .venv-agents/Scripts/python.exe -s -E -m spreading.attempts RUN [RUN ...]

A physical attempt's outcome: transport (request failed), context (refused by the server as over num_ctx), cut
(done_reason length), empty (no visible answer), bad_json, schema (parsed, failing the recorded schema by
llm.schema_errors), valid, or other (any other error, e.g. an engine-count violation; its text is kept). Also summed:
the engine's counts and durations."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from .llm import schema_errors
from .runrec import new_run, write

USAGE = ("total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration")


def outcome(rec):
    """The outcome of one physical attempt, the schema checked again rather than read from the error text."""
    e = rec.get("error") or ""
    if e.startswith("request failed"):
        return "transport"
    if e.startswith("context exceeded"):
        return "context"
    if rec.get("done_reason") == "length":
        return "cut"
    raw = rec.get("raw")
    if not isinstance(raw, str) or not raw.strip():
        return "empty"
    try:
        answer = json.loads(raw)
    except json.JSONDecodeError:
        return "bad_json"
    if "schema" in rec and schema_errors(answer, rec["schema"]):
        return "schema"
    return "valid" if not e else "other"


def count(run):
    """{role/step: {"attempts": n, outcome: n, ..., "not_sent": n, "bookkeeping": n, "other_errors": [...],
    "usage": {field: sum}, "loads": n}} for one run folder; "attempts" counts physical attempts only."""
    out = defaultdict(lambda: {"attempts": 0, "not_sent": 0, "bookkeeping": 0, "other_errors": [],
                               "usage": dict.fromkeys(USAGE, 0), "loads": 0})
    for ln in (Path(run) / "calls.jsonl").read_text(encoding="utf8").splitlines():
        rec = json.loads(ln)
        c = out[f"{rec.get('role')}/{str(rec.get('tag', '')).split(':')[0]}"]
        # cut, outcome, not_sent, sent (the "cut" event copies the dispatch); or how a 429 wait ended, no request
        if isinstance(rec.get("recovery"), str) or "rate_limit_wait" in rec:
            c["bookkeeping"] += 1
            continue
        if "dispatch" not in rec:
            c["not_sent"] += 1
            continue
        c["attempts"] += 1
        k = outcome(rec)
        c[k] = c.get(k, 0) + 1
        if k == "other":
            c["other_errors"].append(rec.get("error"))
        u = rec.get("engine_usage") or {}
        for f in USAGE:
            c["usage"][f] += u.get(f) or 0
        c["loads"] += bool((u.get("load_duration") or 0) > 1e9)     # calls with load_duration over 1 s
    return dict(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args(argv)
    counts = {r: count(r) for r in a.runs}
    d = new_run("attempts", {"runs": a.runs, "label": "per physical attempt; retries included"})
    write(d, "counts.json", counts)
    print(json.dumps(counts, indent=1))
    print(d)
    return d


if __name__ == "__main__":
    main()
