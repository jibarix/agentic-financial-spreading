"""Dated run records, never overwritten."""
import json
import os
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def runs_dir():
    """runs/ in the project. SPREADING_RUNS_DIR moves it, read at each call so a subprocess inherits it: the offline
    tests set it, so a test never adds a record to the real runs/."""
    return Path(os.environ.get("SPREADING_RUNS_DIR") or ROOT / "runs")


def new_run(kind, meta, tries=5):
    """A new run folder named by the second. When that name is taken (two records in one second), it waits for the
    next second and takes a new stamp, at most `tries` stamps; mkdir stays the atomic guard."""
    for n in range(tries):
        stamp = time.strftime("%Y-%m-%dT%H%M%S")
        run_id = stamp + "_" + kind
        d = runs_dir() / run_id
        try:
            d.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            if n == tries - 1:
                raise
            while time.strftime("%Y-%m-%dT%H%M%S") == stamp:
                time.sleep(0.05)
    (d / "run.json").write_text(json.dumps({"run_id": run_id, "kind": kind, **meta}, indent=1), encoding="utf8")
    return d


def write(d, name, obj):
    """obj as JSON, a str as text, bytes as they are (no newline translation)."""
    p = Path(d) / name
    if p.exists():
        raise FileExistsError(p)
    if isinstance(obj, bytes):
        p.write_bytes(obj)
        return
    p.write_text(json.dumps(obj, indent=1) if not isinstance(obj, str) else obj, encoding="utf8")


class StopFile(threading.Event):
    """The stop switch as a file: set once STOP exists in the run folder."""

    def __init__(self, path):
        super().__init__()
        self.path = path

    def is_set(self):
        return super().is_set() or self.path.exists()


def search_use(rec):
    """How a line used search: searches, passages read (in a request that went out), passages its final answer cites
    (as evidence or as support), refusals for citing an unread passage, and each passage it met by how it was found
    x read x cited. A passage reached only as a read's neighbour is found "neighbour". Retrieval and citation
    activity counts, not evidence."""
    obs = rec.get("observations") or {}
    disclosed = set(rec.get("disclosed") or [])
    ans = rec.get("answer") or {}
    cited = [i for i in ans.get("evidence_ids", []) if i.startswith("P")]
    cited += [s["evidence_id"] for al in ans.get("allocations", []) for s in al.get("support", [])
              if s["evidence_id"].startswith("P")]
    # v16: no evidence IDs; a passage is "cited" when code found a quote in it (the answer's or a piece's)
    srcs = [ans.get("_quote_source")] + [x.get("quote_source") for x in ans.get("_pieces") or ()]
    cited += [q["block"] for q in srcs if isinstance(q, dict) and str(q.get("block", "")).startswith("P")]
    found = {}
    for o in obs.values():
        for pid, how in ((o.get("retrieval") or {}).get("provenance") or {}).items():
            found.setdefault(pid, how)
    read = {o["read"] for oid, o in obs.items()
            if o["tool"] == "read_passage" and o.get("read") and not o.get("error") and oid in disclosed}
    table = {}
    for pid in sorted(set(found) | read | set(cited), key=lambda x: int(x[1:])):
        key = f"{found.get(pid, 'neighbour')}|{'read' if pid in read else 'unread'}|"               f"{'cited' if pid in cited else 'uncited'}"
        table[key] = table.get(key, 0) + 1
    return {"searches": sum(o["tool"] == "search_notes" for o in obs.values()),
            "reads": len(read), "passages_cited": sorted(set(cited)),
            "unread_citation": (rec.get("counts") or {}).get("unread_citation", 0), "found_read_cited": table}
