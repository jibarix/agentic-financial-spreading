"""Which resources may reach a model, by identity.

The source rule: vendor guides are reading on how others spread, never accounting context.
A model request may carry only approved resources: the template (its rows carry their own definitions since
template format v5), the template skeleton, a pinned FASB edition and the filing. The separate template
rules were retired with template v5. The template and the skeleton are
bound to their real path and the sha256 pinned in PINNED; a new template version is pinned here when it becomes the
standard one. Model-facing code opens these through `load_json`,
which refuses anything else and records the file and its sha256 (`used()`, kept in the run record). `refuse_vendor`
is the extra literal guard, run by llm.Caller on every request before it is sent.

The register of model-written definitions, drafts and review findings was retired with that process.
"""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RULEBOOK = ROOT / "rulebook/draft"
SKELETON = RULEBOOK / "A1_skeleton.json"           # the rows before template v2; earlier drafts name it as parent
TEMPLATE_V6 = RULEBOOK / "A1_template_v6.json"     # kept for runs that used v6
# v7 (v6 with the cash flow memo rows) became the standard template after v6; v6 runs keep v6
TEMPLATE_V7 = RULEBOOK / "A1_template_v7.json"
# v8 (v7 with every row text a model reads worded without tags) is the standard template (tags hidden is the
# default); v7 runs keep v7
TEMPLATE_V8 = RULEBOOK / "A1_template_v8.json"
# v9 (from v8) and v10 (from v6) add the income statement's title fallback (format v3);
# neither is the standard
TEMPLATE_V9 = RULEBOOK / "A1_template_v9.json"
TEMPLATE_V10 = RULEBOOK / "A1_template_v10.json"
# v11 (v10 in format v4: BS.NCL's capital-side reconstruction, BS.CAP's temporary-equity rows, CF.SUP's heading)
# is not the standard
TEMPLATE_V11 = RULEBOOK / "A1_template_v11.json"
# v12, the engine template (format v5): earlier job settings pin it as the UI's
# default; the standard for the command line stays v8
TEMPLATE_V12 = RULEBOOK / "A1_template_v12.json"
# v13 (v12 with the engine's G&A, EBIT and pretax including equity-method income): earlier job settings
# pin it as the UI's default; v12 stays for runs that used it
TEMPLATE_V13 = RULEBOOK / "A1_template_v13.json"
# v14 (v13 with the definitions and report-check alternatives found by
# root-cause readings, format v6): earlier job settings pin it as the UI's default; v13 stays for its runs and the answer keys
TEMPLATE_V14 = RULEBOOK / "A1_template_v14.json"
# v15 (v14 with statement checks, format v7): earlier job settings pin it
# as the UI's default; v14 stays for its runs
TEMPLATE_V15 = RULEBOOK / "A1_template_v15.json"
# v16 (v15 with the D&A memo row taking every D&A add-back the cash flow statement prints): the current job
# settings pin it as the UI's default; v15 stays for its runs
TEMPLATE_V16 = RULEBOOK / "A1_template_v16.json"
TEMPLATE = TEMPLATE_V8                             # the standard template; v2 to v7 stay
PINNED = {SKELETON: "d0379f0e82f435ae76bca41e292eed704eac564e29ab0cc8f87c2c5a01204174",
          TEMPLATE_V6: "7a72a620b55df1ebe240bcc5738edeae35996abc42bac9c08233485b26a3e6e3",
          TEMPLATE_V7: "f7cef945254b7d04ee788c414f26dcf15e1c954afd8f6540d4688e29cd17e8ea",
          TEMPLATE_V8: "8663b7f2e7e1567a558d12725eff0e9cde48e1c804a41198a88661bfe4fcdfbe",
          TEMPLATE_V9: "83e1ca3ca8fbe4db0279284392855a9dae2b86d67e0869d9ae7a5af09a60dd9c",
          TEMPLATE_V10: "eacee7212c4c4c493019c2384d261da6c9fa945a7524d1895e2da8b358894bd8",
          TEMPLATE_V11: "ba3e8fa712f4d7d44bbd42410f43b843fa2bce0a9e7111239eeb4d9adad5449d",
          TEMPLATE_V12: "b45d9a2026cf9ca1eed1e83a6692605f38bcec408a015f23391440550a954486",
          TEMPLATE_V13: "6e1ddf973ae3021bc030a1f63a565f71e0c9ca11a1cc2babda7bcd90c0996331",
          TEMPLATE_V14: "fc3f9f7b36b916ddec6dadb32714c10f3e81126a34dd7632310a2d1237cc9773",
          TEMPLATE_V15: "cfb1a06b476d33a052b81093131abb3ae67299ae1b5d3f4a8b3fbf5aee247aa8",
          TEMPLATE_V16: "cabdde75e2e665d5d3c3424a1f468a31c0b80fa80129a987a578f32dde54bd41"}
VENDOR = re.compile(r"worldscope", re.I)


def templates_dir():
    """Where the templates an analyst saves live (data/templates).
    SPREADING_TEMPLATES_DIR moves it, read at each call so a subprocess inherits it, as the tests do."""
    import os
    return Path(os.environ.get("SPREADING_TEMPLATES_DIR") or ROOT / "data/templates")


def registry():
    """The saved templates' register: one line {path, sha256, saved, from} per save, appended, never rewritten."""
    return templates_dir() / "registered.jsonl"


def registered():
    """Every saved template's path and pinned sha256, from the register (none when it is absent)."""
    if not registry().is_file():
        return {}
    out = {}
    for line in registry().read_text(encoding="utf8").splitlines():
        if line.strip():
            x = json.loads(line)
            out[Path(x["path"])] = x["sha256"]
    return out


def register(path, source):
    """Add a saved template to the register and to PINNED (path and sha256). A path is registered once."""
    p = Path(path).resolve()
    if any(q.resolve() == p for q in PINNED):
        raise Unapproved(f"{p} is already a pinned template; a changed template is a new file")
    entry = {"path": p.as_posix(), "sha256": sha256(p), "saved": __import__("datetime").datetime.now()
             .isoformat(timespec="seconds"), "from": source}
    registry().parent.mkdir(parents=True, exist_ok=True)
    with registry().open("a", encoding="utf8", newline="\n") as f:
        f.write(json.dumps(entry) + "\n")
    PINNED[p] = entry["sha256"]
    return entry


def load_registered():
    """Pin every saved template in the register (called at import, and again after the folder moves in a test)."""
    for p, digest in registered().items():
        PINNED.setdefault(p, digest)

_used = {}
_expected = None     # the run's frozen inventory (begin), or None outside a run


class Unapproved(RuntimeError):
    pass


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def approved(path):
    """Is this file an approved resource, at its real path and its pinned hash? FASB editions and filings are opened
    by their own readers (taxonomy.edition, filing.Filing), which read only from their fixed places."""
    p = Path(path).resolve()
    return p.is_file() and any(p == q.resolve() and sha256(p) == digest for q, digest in PINNED.items())


def pin(path):
    """Make a pinned template (PINNED, not the skeleton; `path` under the project) the one this process spreads with:
    a job's or a queue's `template` setting, so a keyed v6 run is not spread on the standard v7."""
    global TEMPLATE
    p = next((q for q in PINNED if q != SKELETON and q.resolve() == (ROOT / path).resolve()), None)
    if p is None or not approved(p):
        raise Unapproved(f"template {path}: not a pinned template at its path and hash")
    TEMPLATE = p


def load_json(path):
    p = Path(path)
    if not approved(p):
        raise Unapproved(f"{p} is not an approved resource at this path and hash for a model request (see "
                         "provenance.approved)")
    key, digest = str(p.resolve()), sha256(p)
    if _expected is not None and _expected.get(key) != digest:
        raise Unapproved(f"{p} is not in this run's frozen inventory at this hash")
    _used[key] = digest
    return json.loads(p.read_text(encoding="utf8"))


def used():
    """Every approved resource opened since the run began (begin), or in this process outside a run, with its
    sha256: the run-scoped audit, kept apart from the freeze."""
    return dict(_used)


def inventory():
    """The approved resources a spread is set to open: the template, which carries the rows' definitions, by real
    path, with its sha256 (None if absent). The same in any process, whatever it opened before; the
    freeze records this, never `used()`."""
    return {str(Path(q).resolve()): sha256(q) if Path(q).is_file() else None for q in (TEMPLATE,)}


def begin(frozen):
    """Start a run's scope: from here on `load_json` opens only files of the frozen inventory at their frozen hash,
    and `used()` lists what this run opened."""
    global _expected
    _expected = dict(frozen)
    _used.clear()


def end():
    """Leave the run's scope (tests, and a process that runs more than one spread)."""
    global _expected
    _expected = None


def refuse_vendor(*texts):
    """The literal guard: the reason a request is refused, or None."""
    for t in texts:
        if t and VENDOR.search(t):
            return "the request names a vendor guide (Worldscope); vendor guides never reach a model"
    return None
load_registered()      # the saved templates (data/templates/registered.jsonl) are pinned in every process
