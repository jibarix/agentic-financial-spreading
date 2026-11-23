"""The pages' views of one job, read from the job's own records: the line progress of its spread children, the model calls of every child, the tiebreaker lines and the
repair tree. Read only; what the records say, nothing inferred. A view the job's records do not have is None, which
the page shows as "not exercised in this job". The live pages (ui/server.py) use it; so does the trace of a stored
spread's sources to its records (trace).

Records are read as complete lines only (ui/jobs.py complete_lines): a line still being written waits for the next
read. A line counts as processed once its event is written, whatever its outcome; correctness never enters the
progress.
"""
import copy
import json
import re
import threading
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import urlencode

from jobs import CHILD, VERSION, complete_lines, read_json

# Model text sometimes names a line by its tag; the page shows the line's printed label instead (display only).
TEXT = {"reason", "why", "answer", "question", "refused", "tool_error"}
RECENT = 12              # the latest line events shown per statement


def load(p):
    return json.loads(Path(p).read_text(encoding="utf8"))


def runs_of(d, kind):
    """The child run folders of one kind (spread, repair, grade), in the job's order."""
    return sorted(Path(d).glob(f"children/*/*_{kind}"))


# ---- line progress and calls (the jobs page) ----------------------------------------------------------------------

_seen = {}               # path -> (bytes read, lines read, state): calls.jsonl grows to megabytes; each poll reads on
_lock = threading.Lock()  # the server answers polls on many threads: one fold at a time, so no line counts twice


def fold(p, step, state):
    """`state` folded over the complete JSON lines of `p` not read before (kept per path between polls); a copy, so
    the kept state goes on folding."""
    p = Path(p)
    with _lock:
        at, n, state = _seen.get(p, (0, 0, state))
        try:
            with open(p, "rb") as fh:
                fh.seek(at)
                data = fh.read()
        except OSError:
            return copy.deepcopy(state)
        end = data.rfind(b"\n") + 1          # after the last newline: a line still being written waits
        for raw in data[:end].split(b"\n")[:-1]:
            n += 1
            try:
                step(state, json.loads(raw), n)
            except ValueError:
                step(state, {"unreadable": raw.decode("utf8", "replace")[:200]}, n)
        _seen[p] = (at + end, n, state)
        return copy.deepcopy(state)


def kind(r):
    """A calls.jsonl record as spreading/attempts.py count() classes it: "bookkeeping" (a cut, outcome or 429-wait
    record; no request), "not_sent", or "sent" (one physical request)."""
    if isinstance(r.get("recovery"), str) or "rate_limit_wait" in r:
        return "bookkeeping"
    return "sent" if "dispatch" in r else "not_sent"


def add_call(s, r, n):
    k = "unreadable" if "unreadable" in r else kind(r)
    if k != "sent":
        s[k] += 1
        return
    s["calls"] += 1
    s["cost_usd"] = round(s["cost_usd"] + float(r.get("cost_usd") or 0), 6)
    s["tokens"] += int(r.get("prompt_tokens") or 0) + int(r.get("completion_tokens") or 0)
    s["roles"][r.get("role") or "?"] = s["roles"].get(r.get("role") or "?", 0) + 1
    s["errors"] += bool(r.get("error"))
    m = s["models"].setdefault(r.get("model") or "?", {"calls": 0, "tokens": 0, "seconds": 0.0, "cost_usd": 0.0})
    m["calls"] += 1
    m["tokens"] += int(r.get("prompt_tokens") or 0) + int(r.get("completion_tokens") or 0)
    m["seconds"] = round(m["seconds"] + float(r.get("seconds") or 0), 2)
    m["cost_usd"] = round(m["cost_usd"] + float(r.get("cost_usd") or 0), 6)
    s["last"] = {k: r.get(k) for k in ("time", "role", "model", "tag", "seconds")}


def add_embedding(s, e, n):
    """Embedding requests from the child's ledger: each send of a reservation made for the embedding role."""
    if e.get("event") == "reserve" and e.get("role") == "embedding":
        s["ids"][e.get("id")] = 0
    elif e.get("event") == "send" and e.get("id") in s["ids"]:
        s["ids"][e["id"]] += 1
        s["embeddings"] += 1


# A child's call log: a spread or repair child writes calls.jsonl; the corrections child (named totals) writes
# named_totals_calls.jsonl in the same record form (spreading/named_totals.py). A child has one of them.
CALL_LOGS = ("calls.jsonl", "named_totals_calls.jsonl")


def calls(c):
    """A child's chat requests (physical ones only; records that sent nothing are counted apart) and its embedding
    requests."""
    log = next((c / n for n in CALL_LOGS if (c / n).exists()), c / CALL_LOGS[0])
    s = fold(log, add_call, {"calls": 0, "cost_usd": 0.0, "tokens": 0, "roles": {}, "errors": 0,
                                          "models": {}, "last": None, "not_sent": 0, "bookkeeping": 0,
                                          "unreadable": 0})
    s["embeddings"] = fold(c / "ledger/events.jsonl", add_embedding, {"ids": {}, "embeddings": 0})["embeddings"]
    return s


def add_event(s, e, n):
    s.append({**e, "n": n})


def progress(d):
    """Each child's progress in the job's order: a spread child's planned lines, the lines processed so far and their
    latest events (stable IDs <job>#<child>#<line number>), and every child's model calls from its calls.jsonl."""
    d, out = Path(d), []
    for c in sorted(d.glob("children/*/*")):
        if not c.is_dir():
            continue
        child = {"child": c.parent.name, "run": c.name, "kind": c.name.rsplit("_", 1)[-1], "calls": calls(c)}
        if child["kind"] == "spread":
            ev = list(fold(c / "line_events.jsonl", add_event, []))
            planned = next((e for e in ev if e.get("event") == "planned"), None)
            lines = {}
            for e in ev:
                if e.get("event") == "line":
                    lines[e["concept"]] = {**e, "id": f"{d.name}#{c.parent.name}#{e['n']}"}
            want = set(planned["lines"]) if planned else set()
            done = len(want & set(lines))
            child.update(statement=(planned or {}).get("statement"), sections=(planned or {}).get("sections"),
                         planned=len(want) if planned else None, processed=done,
                         percent=round(100 * done / len(want)) if want else None,
                         recent=sorted(lines.values(), key=lambda e: e["n"])[-RECENT:])
        out.append(child)
    return out


def summary(d):
    """The job's line progress overall: lines processed of lines planned in the statements started so far, and the
    statements its settings list that have not started."""
    kids = progress(d)
    spreads = [k for k in kids if k["kind"] == "spread" and k["planned"] is not None]
    planned, done = sum(k["planned"] for k in spreads), sum(k["processed"] for k in spreads)
    listed = [s["statement"] for s in ((read_json(Path(d) / "run.json") or {}).get("settings_content") or {})
              .get("statements") or []]
    started = {k["statement"] for k in spreads}
    return {"children": kids, "planned": planned, "processed": done,
            "percent": round(100 * done / planned) if planned else None,
            "not_started": [s for s in listed if s not in started]}


def operations(d):
    """The evals page's operations view of one job: calls, tokens, seconds and dollars per model over all its children,
    and the company caps against what was used, from job.json once the job has ended (None before)."""
    models = {}
    for k in progress(d):
        for name, m in k["calls"]["models"].items():
            t = models.setdefault(name, {"calls": 0, "tokens": 0, "seconds": 0.0, "cost_usd": 0.0})
            for x in t:
                t[x] = round(t[x] + m[x], 6)
    company = (read_json(Path(d) / "job.json") or {}).get("company") or {}
    caps = [{"cap": x, "limit": v, "used": (company.get("used") or {}).get(x)}
            for x, v in (company.get("limits") or {}).items()] or None
    return {"job_id": Path(d).name, "models": models, "caps": caps, "grade": grade_counts(d)}


def parts_of(d):
    """The parts the job ran with, as job.json records them ({"tiebreaker": "on"|"off", "tree": ...}); None for a
    job whose record does not name them (every job before the Run page's switches)."""
    return (read_json(Path(d) / "job.json") or {}).get("parts")


def grade_counts(d):
    """The job's recorded grade against the AI reference (its grade child): the counts overall and per statement,
    the reference's file name, the grade's own label and the job's parts; None when the job was not graded."""
    g = next(iter(runs_of(d, "grade")), None)
    counts = read_json(g / "counts.json") if g else None
    if not isinstance(counts, dict):
        return None
    run = read_json(g / "run.json") or {}
    return {"child": g.name, "reference": Path(str(run.get("reference") or "")).name, "label": run.get("label"),
            "counts": {k: v for k, v in counts.items() if k in ("all", "IS", "BS", "CF")}, "parts": parts_of(d)}


# ---- tiebreaker lines and the tree (their pages) -------------------------------------------------------------------

def proposed(s):
    """The row (or piece rows) a step's reply proposed, read from the reply as recorded; None when it proposed none."""
    try:
        r = json.loads(s.get("raw") or "")
    except ValueError:
        return None
    a = r.get("step") if isinstance(r.get("step"), dict) else r
    return a.get("row") or [x.get("row") for x in a.get("pieces") or [] if isinstance(x, dict)] or None


def steps(loop):
    """One model's loop on one line: each step's tool or answer, its stated reason, how the step ended, and what code refused."""
    out = []
    for s in (loop or {}).get("steps") or []:
        tokens = (s.get("prompt_eval_count") or 0) + (s.get("eval_count") or 0)
        out.append({"step": s.get("step"), "action": s.get("action") or s.get("tool"), "args": s.get("args"),
                    "why": s.get("why") or "", "seconds": s.get("wall_s"), "ended": s.get("ended"), "tokens": tokens or None,
                    "row": proposed(s) if s.get("action") == "answer" else None,
                    "refused": [r.get("text") for r in s.get("refusals") or []],
                    "call_error": s.get("call_error") or s.get("error"), "tool_error": s.get("tool_error")})
    return out


def pieces(ans):
    """The pieces an answer split the line into, each with its row and amount as code recorded them, and the
    remainder left on the answer's own row; [] when the answer placed the whole line."""
    out = [{"row": x.get("row"), "amount": x.get("amount"), "label": x.get("concept")}
           for x in ans.get("_pieces") or [] if isinstance(x, dict)]
    if out and ans.get("_remainder") not in (None, "0", 0):
        out.append({"row": ans.get("row"), "amount": ans.get("_remainder"), "label": "remainder"})
    return out


def opinion(loop):
    if not loop:
        return None
    ans = loop.get("answer") if isinstance(loop.get("answer"), dict) else {}
    return {"model": loop.get("model"), "row": loop.get("row") or ans.get("row"), "pieces": pieces(ans),
            "amount": loop.get("amount"), "decision": loop.get("decision"),
            "termination": loop.get("termination"), "why": loop.get("why") if loop.get("termination") != "finished" else None,
            "reason": ans.get("reason") or "", "quote": ans.get("quote") or "", "steps": steps(loop)}


def grades(d, reference=None):
    """The job's recorded grade per line (its grade child), with the AI reference's row(s) and the alternatives it
    accepts; {} when the job was not graded."""
    g = next(iter(runs_of(d, "grade")), None)
    if g is None or not (g / "lines.json").exists():
        return {}
    accepted = {}
    ref = load(reference) if reference else []
    for ln in ref["lines"] if isinstance(ref, dict) else ref:
        a = (ln.get("grading") or {}).get("accepted_rows") or []
        a += [r for p in (ln.get("placement") or {}).get("pieces") or [] for r in (p.get("grading") or {}).get("accepted_rows") or []]
        accepted[ln["line_id"]] = a
    out = {}
    for x in load(g / "lines.json"):
        e = x["expected"]
        out[x["concept"]] = {"correctness": x.get("correctness"), "rows": [e["row"]] if e.get("row") else [p["row"] for p in e.get("pieces") or []],
                             "accepted": accepted.get(x["line_id"], [])}
    return out


def where(c, model):
    """Where the child's tiebreaker ran, from its ledger reservations for the role ("local" or "online, paid"); by
    the model's name only when the ledger has none."""
    flags = {e.get("local") for e in complete_lines(c / "ledger/events.jsonl")
             if e.get("event") == "reserve" and e.get("role") == "haiku"}
    if flags in ({True}, {False}):
        return "local" if True in flags else "online, paid"
    return "online, paid" if "/" in (model or "") else "local"


def still_disputed(v):
    """A line the blind comparison found disputed and the second look left unresolved."""
    return ((v.get("consistency") or {}).get("result") == "disputed"
            and ((v.get("feedback") or {}).get("after") or {}).get("result") != "agree")


def revised(v):
    """Whether the second look changed the analyst's placement: its kept retry's decomposition (row and pieces, from
    the record's own comparison after the retry) differs from the blind one. A kept retry can repeat the placement;
    without both decompositions recorded, the line is not called revised."""
    fb, first = v.get("feedback") or {}, (v.get("consistency") or {}).get("analyst")
    after = fb.get("after") or {}
    kept = after.get("analyst") or after.get("decomposition")
    return fb.get("kept") == "retry" and first is not None and kept is not None and kept != first


def tiebreaks(d, graded, root):
    """The spread lines with a tiebreaker record or a vote, from each finished spread child, and with the tiebreaker
    switched off for the job, its lines still disputed (no vote record is made up); None when there are none. The
    tiebreaker's name and where it ran come from the records ("haiku" is only the role's name). When a second look's
    retry was kept, `analyst` is the retry and `analyst_first` the blind answer; `revised` says whether the retry
    changed the placement."""
    out, off = [], (parts_of(d) or {}).get("tiebreaker") == "off"
    for c in runs_of(d, "spread"):
        if not (c / "c4.json").exists():
            continue
        run, c4, claims, lines = load(c / "run.json"), load(c / "c4.json"), load(c / "claims.json"), load(c / "lines.json")
        line = {ln["concept"]: ln for ln in lines}
        third = next((v["haiku"].get("model") for v in c4.values() if v.get("haiku")), None)
        tb = {"model": third, "where": where(c, third)} if third else None
        cost = {}
        for r in complete_lines(c / "calls.jsonl"):
            if r.get("role") == "haiku":
                k = (r.get("tag") or "")[3:].rsplit(":", 1)[0]
                cost[k] = cost.get(k, 0) + float(r.get("cost_usd") or 0)
        for concept, v in c4.items():
            if not (v.get("haiku") or v.get("vote")
                    or off and still_disputed(v)):
                continue
            ln, vote = line.get(concept, {}), v.get("vote") or {}
            out.append({"statement": run["section"][0].split(".")[0], "section": v.get("section"), "concept": concept,
                        "label": ln.get("label", concept), "amount": ln.get("amount"), "unit": ln.get("unit"),
                        "analyst": opinion(v.get("c4")), "checker": opinion(v.get("review")), "haiku": opinion(v.get("haiku")),
                        "analyst_first": opinion(v["c4_first"]) if v.get("c4_first") else None, "revised": revised(v),
                        "second_look": {"ran": bool((v.get("feedback") or {}).get("ran")),
                                        "kept": (v.get("feedback") or {}).get("kept")} if v.get("feedback") else None,
                        "comparison": (v.get("consistency") or {}).get("result"),
                        "tiebreaker": tb, "vote": {"result": vote.get("result"), "why": vote.get("why")},
                        "used": sorted({x["row"] for x in claims if x["source"] == concept}),
                        "committed": [{"row": x["row"], "amount": x.get("amount")} for x in claims if x["source"] == concept],
                        "grade": graded.get(concept),
                        "cost_usd": round(cost.get(concept, 0), 6), "folder": rel(c, root)})
    return out or None


def rel(p, root):
    try:
        return str(Path(p).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return Path(p).name


def spread_lines(d):
    out = {}
    for c in runs_of(d, "spread"):
        if (c / "lines.json").exists():
            out.update({ln["concept"]: ln for ln in load(c / "lines.json")})
    return out


def whole(row, parts, rest):
    """A whole decomposition as recorded: every piece with its row and amount, and what is left on `row`."""
    return {"row": row, "pieces": [{"row": p.get("row"), "amount": p.get("amount"), "label": p.get("concept")}
                                   for p in parts or () if isinstance(p, dict)], "remainder": rest}


def recheck(d, c8, lines, label, graded, req=None):
    """One line the checker reviewed after the repair: the spread's first placement of it, the chosen branch's whole
    decomposition (the review request's), and the checker's recheck with its whole answer, when it finished, and the
    review's recorded status and reason."""
    first = {}
    for s in runs_of(d, "spread"):
        if c8.get("line") and (s / "c4.json").exists():
            first = load(s / "c4.json").get(c8["line"]) or first
    ck = c8.get("checker") or {}
    ans = (first.get("c4") or {}).get("answer")
    mine = (req or {}).get("decomposition")
    theirs = ck.get("answer") if isinstance(ck.get("answer"), dict) else {}
    amount = (lines.get(c8.get("line")) or {}).get("amount")
    return {"line": label.get(c8.get("line"), c8.get("line")), "line_amount": amount,
            "first": {"row": (first.get("c4") or {}).get("row") or first.get("row"), "class": first.get("class"),
                      "pieces": pieces(ans if isinstance(ans, dict) else {}), "why": first.get("why")},
            "candidate": whole((mine["remainder"] or {}).get("row"), mine["pieces"], (mine["remainder"] or {}).get("amount"))
                         if mine else None,
            "recheck": {"model": ck.get("model"), "row": ck.get("row"), "status": c8.get("status"), "why": c8.get("why"),
                        "termination": ck.get("termination"),
                        "answer": whole(ck.get("row"), theirs.get("_pieces"),
                                        theirs.get("_remainder") if theirs.get("_remainder") is not None else amount)
                                  if ck.get("termination") in ("finished", "finished_at_last_step") else None,
                        "reason": theirs.get("reason") or "", "steps": steps(ck),
                        "candidate_row": ((c8.get("comparison") or {}).get("analyst") or {}).get("row")
                                         or ((mine or {}).get("remainder") or {}).get("row")},
            "grade": graded.get(c8.get("line"))}


def tree(d, c, graded, root):
    """One finished repair child: its nodes, the controller's picks (each with every attempt the controller made at
    that level and task), the checker's recheck of every reviewed line and the outcome."""
    t, out, reviews = load(c / "c7_tree.json"), read_json(c / "outcome.json") or {}, load(c / "c8_review.json")["reviews"] or []
    run = load(c / "run.json")
    lines = spread_lines(d)
    label = {k: v["label"] for k, v in lines.items()}
    reqs = {x.get("line"): x.get("request") for x in t["calls"].get("c8") or []}
    rechecks = [recheck(d, r, lines, label, graded, reqs.get(r.get("line"))) for r in reviews]
    depth_pruned = {x["id"]: x["why"] for x in t["pruning"].get("depth") or []}

    def failure(f):
        return {"check": f["check"], "lines": [label.get(x, x) for x in f.get("lines") or []],
                "filed": f.get("filed"), "built": f.get("built"), "gap": f.get("gap")}
    nodes = []
    for n in t["nodes"]:
        th = n.get("thought") or {}
        nodes.append({"id": n["id"], "parent": n["parent"], "depth": n["depth"], "status": n["status"],
                      "action": th.get("action"), "line": label.get(th.get("line"), th.get("line")), "row": th.get("row"),
                      "reason": th.get("reason") or "", "quote": th.get("quote") or "",
                      "rejected": [{"rule": t["settings"]["invariants"].get(w["invariant"], w["invariant"]), "why": w["why"]}
                                   for w in n.get("why") or []],
                      "failures": [failure(f) for f in n.get("failures") or []],
                      "change": n.get("failures_change"), "complete": n.get("complete"), "eligible": n.get("eligible"),
                      "critic": n.get("critic"), "expanded": n.get("expanded"), "skipped": n.get("skipped"),
                      "depth_limit": depth_pruned.get(n["id"])})
    ctl = {}             # (level, task) -> every controller attempt there, in attempt order
    for x in t["calls"].get("controller") or []:
        ctl.setdefault((x.get("level"), (x.get("request") or {}).get("task")), []).append(
            {"attempt": x.get("attempt"), "errors": x.get("errors") or [], "reason": (x.get("reply") or {}).get("reason")})
    for v in ctl.values():
        v.sort(key=lambda a: a["attempt"] or 0)

    def pick(p):
        tried = ctl.get((p["level"], "select" if p["decision"] == "select" else "rank"), [])
        return {"level": p["level"], "decision": p["decision"], "shown": p["shown"], "picked": p["picked"],
                "attempts": tried, "reason": tried[-1]["reason"] if tried else None}
    meaning = [{"question": m.get("question"), "answer": (m.get("reply") or {}).get("answer")} for m in t["calls"].get("meaning") or []]
    section = run.get("section") or ""
    one = rechecks[0] if rechecks else recheck(d, {}, lines, label, graded)
    return {"statement": section.split(".")[0], "section": section, "folder": rel(c, root), "growing": False,
            "settings": {"models": t["settings"]["models"], "beam": t["settings"]["beam"],
                         "children": t["settings"]["children"], "depth": t["settings"]["depth"],
                         "feedback": t["settings"].get("feedback")},      # None: recorded before the critic fed back
            **{k: one[k] for k in ("line", "line_amount", "first", "recheck", "grade")}, "rechecks": rechecks,
            "root_failures": [failure(f) for f in next(n for n in t["nodes"] if n["id"] == "0")["failures"]],
            "nodes": nodes, "picks": [pick(p) for p in t["picks"]], "meaning": meaning, "chosen": t.get("chosen"),
            "termination": t.get("termination"), "why": t.get("why"),
            "feedback": t.get("feedback"),      # which critic verdicts reached a proposer; None: not recorded
            "outcome": out.get("outcome"), "calls": {k: len(v) for k, v in t["calls"].items()}}


def first_turn(r):
    turns = r.get("turns") or []
    text = turns[0].get("content") if turns and isinstance(turns[0], dict) else r.get("user")
    try:             # the JSON code sent, before any note code appended after it ("[step 1 of at most 4]")
        sent = json.JSONDecoder().raw_decode((text or "").lstrip())[0]
    except ValueError:
        return {}
    return sent if isinstance(sent, dict) else {}


def growing(d, c, root):
    """A repair child still running, as its calls.jsonl shows it (complete lines only): each node the generator was
    sent (its ID and depth, as code sent them) with the changes the generator proposed, the critic's verdicts and the
    controller's choices, in the order recorded. Which proposals code kept, and the node IDs it gave them, are in
    c7_tree.json once the tree finishes; nothing here guesses them."""
    label = {k: v["label"] for k, v in spread_lines(d).items()}
    run = read_json(c / "run.json") or {}
    nodes, critic, picks, rechecked = {}, [], [], 0
    for r in complete_lines(c / "calls.jsonl"):
        role, ans, sent = r.get("role"), r.get("answer") if isinstance(r.get("answer"), dict) else {}, first_turn(r)
        if role == "generator" and "node" in sent:
            n = nodes.setdefault(sent["node"], {"id": sent["node"], "depth": sent.get("depth"),
                                                "parent": sent["node"].rpartition(".")[0] or None, "proposed": [],
                                                "asked": 0})
            step = ans.get("step") or {}
            if step.get("action") == "propose":
                n["proposed"] = [{"action": th.get("action"), "line": label.get(th.get("line"), th.get("line")),
                                  "row": th.get("row"), "reason": th.get("reason") or ""}
                                 for th in step.get("thoughts") or [] if isinstance(th, dict)]
            elif step.get("action") == "ask":
                n["asked"] += 1
        elif role == "critic":
            critic.append({"line": label.get(sent.get("line"), sent.get("line")), "verdict": ans.get("verdict"),
                           "reason": ans.get("reason") or ""})
        elif role == "controller":
            picks.append({"level": sent.get("level"), "order": ans.get("order"), "select": ans.get("select"),
                          "choice": ans.get("choice"), "reason": ans.get("reason") or (ans.get("choice") or {}).get("reason")})
        elif role == "checker":
            rechecked += 1
    section = run.get("section") or ""
    return {"statement": section.split(".")[0], "section": section, "folder": rel(c, root), "growing": True,
            "provisional": "proposals only: which ones code kept is recorded when the tree finishes",
            "nodes": list(nodes.values()), "critic": critic, "picks": picks, "recheck_calls": rechecked}


def trees(d, graded, root):
    """Every repair child: finished ones from c7_tree.json, running ones as they grow; None when the job has none."""
    out = [tree(d, c, graded, root) if (c / "c7_tree.json").exists() else growing(d, c, root)
           for c in runs_of(d, "repair")]
    return out or None


def plain(x, labels, key=None):
    if isinstance(x, dict):
        return {k: plain(v, labels, k) for k, v in x.items()}
    if isinstance(x, list):
        return [plain(v, labels, key) for v in x]
    if isinstance(x, str) and key in TEXT:
        for c, lab in labels.items():
            x = x.replace(c, "the line “" + lab + "”")
    return x


def job_views(d, reference=None, root=None):
    """The tiebreaker lines and the trees of one job, model text with line tags shown as labels."""
    root = Path(root or Path(d).parents[1])
    graded = grades(d, reference)
    labels = {k: v["label"] for k, v in spread_lines(d).items()}
    return {"job_id": Path(d).name, "tiebreaks": plain(tiebreaks(d, graded, root), labels),
            "trees": plain(trees(d, graded, root), labels)}


# ---- the trace of a stored spread's sources to the job's own records (Results) ------------------------------------

QUOTE_KEYS = ("tier", "kind", "block", "label")     # where code found the quote; the matching policy stays in the record


def record_href(job, **q):
    """The address of one record (or one entry of it) through /api/job/record: by job ID and fixed names only."""
    return "/api/job/record?" + urlencode({"id": job, **{k: v for k, v in q.items() if v is not None}})


def repairs(d, version):
    """The concepts a committed repair changed on the way to `version`, each with its repair child (children/<step>/
    <run>), by the cumulative parent chain the versions recorded; the repair run is named by its folder, never its
    recorded absolute path."""
    out, seen = {}, set()
    while version and version not in seen and VERSION.fullmatch(version):
        seen.add(version)
        cv = read_json(Path(d) / "cumulative" / version / "cumulative.json") or {}
        run = Path(str(cv.get("repair_run") or "").replace("\\", "/"))
        if run.name and len(run.parts) >= 2:
            child = f"{run.parts[-2]}/{run.name}"
            for concept in cv.get("changed") or []:
                out.setdefault(concept, {"child": child, "version": version})
        version = cv.get("parent")
    return out


def node_of(tree, concept):
    """The node on the committed (else chosen) path of a c7 tree whose thought changed `concept`, or None."""
    nodes = {n.get("id"): n for n in tree.get("nodes") or [] if isinstance(n, dict)}
    at = tree.get("committed") or tree.get("chosen")
    while at in nodes:
        n = nodes[at]
        if ((n.get("thought") or {}).get("line")) == concept:
            return n
        at = n.get("parent")
    return None


def trace(d, head, sources):
    """Each stored source traced to the records of the version it was imported from: its filing (accession), its
    fact as the recorded line names it (concept, period, unit, amount; the internal fact ID stays in the record), where
    code found its quote, and record addresses for the recorded line, the answer, the section's checks, the version's
    run.json, the vote for a disputed line and the tree node for a repaired one. What is not on this computer, or not
    in the records, is said as unavailable; nothing is inferred. `d` is the job folder (None when absent)."""
    job, folder = head.get("job_id"), str(head.get("run_folder") or "")
    parts = folder.replace("\\", "/").split("/")
    version = parts[3] if len(parts) == 4 and parts[0] == "runs" and parts[1] == job and parts[2] == "cumulative"         and VERSION.fullmatch(parts[3]) else None
    out = {"job_id": job, "version": version, "run_folder": folder, "accession": head.get("accession"),
           "pages": ["jobs", "tiebreak", "tree"] if d is not None else [], "unavailable": [], "sources": []}
    vd = Path(d) / "cumulative" / version if d is not None and version else None
    if d is None:
        out["unavailable"].append(f"the job folder {job} is not on this computer: its records cannot be opened")
    elif version is None:
        out["unavailable"].append(f"the stored run folder {folder!r} is not a version folder of job {job}")
    elif not vd.is_dir():
        out["unavailable"].append(f"the version folder {folder} is not on this computer")
        vd = None
    if vd is not None:
        out["run"] = record_href(job, version=version, name="run.json")
        out["cumulative"] = record_href(job, version=version, name="cumulative.json")
    lines = {ln.get("fact"): ln for ln in (read_json(vd / "lines.json") if vd else None) or [] if isinstance(ln, dict)}
    c4 = (read_json(vd / "c4.json") if vd else None) or {}
    fixed = repairs(d, version) if vd else {}
    trees = {}
    for s in sources:
        concept, section = s.get("line"), s.get("section")
        try:
            qs = json.loads(s.get("quote_source") or "null") or {}
        except ValueError:
            qs = {}
        try:
            facts = json.loads(s.get("facts") or "null") or []
        except ValueError:
            facts = []
        ln = lines.get(s.get("fact"))
        t = {"accession": head.get("accession"), "unavailable": [],
             "fact": {k: ln.get(k) for k in ("concept", "label", "period", "unit", "amount")} if ln else None,
             "other_facts": max(0, len([f for f in facts if isinstance(f, list) and f and f[0] != s.get("fact")])),
             "quote_source": {k: qs.get(k) for k in QUOTE_KEYS if qs.get(k) is not None} or None}
        if vd is None:
            t["unavailable"].append("the records of this spread are not on this computer")
        else:
            if ln is None:
                t["unavailable"].append("the source's fact is not among the version's recorded lines")
            t["line"] = record_href(job, version=version, name="lines.json", line=concept)
            t["checks"] = record_href(job, version=version, name="checks.json", section=section)
            entry = c4.get(concept) if isinstance(c4, dict) else None
            if isinstance(entry, dict):
                t["answer"] = record_href(job, version=version, name="c4.json", line=concept, field="c4")
                if entry.get("vote") or entry.get("haiku"):
                    t["vote"] = {"href": record_href(job, version=version, name="c4.json", line=concept, field="vote"),
                                 "statement": head.get("statement"), "concept": concept}
            else:
                t["unavailable"].append("the version's c4.json has no answer for this line")
            fx = fixed.get(concept)
            if fx:
                if fx["child"] not in trees:
                    trees[fx["child"]] = read_json(Path(d) / "children" / fx["child"] / "c7_tree.json")
                n = node_of(trees[fx["child"]] or {}, concept)
                t["tree"] = {"child": fx["child"], "folder": fx["child"].split("/")[-1], "version": fx["version"],
                             "node": n.get("id") if n else None,
                             "href": record_href(job, child=fx["child"], name="c7_tree.json",
                                                 node=n.get("id") if n else None)}
                if n is None:
                    t["unavailable"].append(f"the repair {fx['child']} changed this line, but its tree records no node "
                                            "on the committed path that names it")
        out["sources"].append(t)
    return out


# ---- replay (the Replay page) -------------------------------------------------------------------------------------

LINE_CALL = re.compile(r"C4:(.+):s\d+")      # a call on one line: C4:<concept>:s<step>
CALL_FIELDS = ("time", "role", "model", "tag", "attempt", "seconds", "prompt_tokens", "completion_tokens", "cost_usd",
               "done_reason", "error", "system", "user", "schema", "raw", "answer", "reasoning")


def replay(d):
    """The job's spreads as recorded, in the job's order, for the Replay page: per statement the printed lines (the
    lines the run read, in printed order when the coverage record gives it: replay_printed), the lines the organizer
    planned and the order the job recorded them in, each line's recorded event
    (row, pieces, the checker's review, a vote) with the model calls made on it (time, role, model, seconds, dollars;
    their text by `call`), and the statement's code checks, its totals compared with the filing and its report checks.
    What the records say, nothing inferred; a statement whose plan is not recorded yet is left out."""
    out, (gate, own, totals) = [], replay_gate(d)
    defs = replay_report_defs(d) if gate is not False else ()
    for c in runs_of(d, "spread"):
        ev = complete_lines(c / "line_events.jsonl")
        plan = next((e for e in ev if e.get("event") == "planned"), None)
        if plan is None:
            continue
        on_line, other = {}, []
        for n, r in enumerate(complete_lines(c / "calls.jsonl")):
            if not r.get("role"):
                continue
            item = {"n": n, "time": r.get("time"), "role": r["role"], "model": r.get("model"),
                    "seconds": r.get("seconds"), "cost_usd": r.get("cost_usd"), "error": bool(r.get("error"))}
            m = LINE_CALL.fullmatch(r.get("tag") or "")
            (on_line.setdefault(m.group(1), []) if m else other).append(item)
        dg = read_json(c / "diagnostics.json") or {}
        outs = dg.get("queued_outcomes") or {}
        c4 = read_json(c / "c4.json") or {}
        read = read_json(c / "lines.json") or []
        claims = [x for x in read_json(c / "claims.json") or [] if isinstance(x, dict)]
        dropped = Counter(x.get("concept") for x in read_json(c / "exceptions.json") or [] if isinstance(x, dict)
                          and x.get("step") == "C5" and "second claim dropped" in str(x.get("why")))
        # a printed total the template compares, or one the run marks as a control, is a total, whatever else is
        # recorded for it (result_workbook: "total: compared in Checks", before placed or outside)
        tot = totals | {x.get("concept") for x in read if isinstance(x, dict) and x.get("control")}
        ctl = read_json(c / "controls.json") or {}
        cks = [x for x in read_json(c / "checks.json") or [] if isinstance(x, dict)]
        # the comparisons this statement's records ran on each total, by exact concept (a concept the template
        # compares elsewhere, Net income on the cash flow statement, is a total here but not compared)
        rcs = [x for x in read_json(c / "report_checks.json") or [] if isinstance(x, dict)]
        compared, verified = statement_totals({"checks": cks, "controls": ctl, "report_checks": rcs, "lines": read,
                                               "diagnostics": dg}, defs)
        seen, lines = set(), []
        for e in ev:
            if e.get("event") != "line":
                continue
            k = e.get("concept")
            o = outs.get(k) if e.get("queued") and isinstance(outs.get(k), dict) else None
            lines.append({**{f: e.get(f) for f in ("time", "section", "concept", "label", "from", "row",
                                                   "class", "review", "vote", "not_run")},
                          "placed": replay_signed(e.get("placed"), k, claims),
                          "queued": bool(e.get("queued")), "why": (o or {}).get("why") or replay_c4_why(e, c4),
                          "total": k in tot, "compared": k in tot and bool(compared.get(k)),
                          "total_said": total_said(k, compared, verified) if k in tot else None,
                          "outside": k not in tot and replay_outside(e, o),
                          "calls": [] if k in seen else on_line.get(k, [])})
            seen.add(k)
        replay_dropped(lines, dropped)
        # a "rows add to" check over a section the template controls is counted with the controls, as the workbook
        # counts it (result_workbook, Checks sheet: tied or failed, a statement check when its section is gated)
        sc = [x for x in cks if str(x.get("check")).startswith("rows add to ") and (own is None or x.get("section") in own)]
        out.append({"child": c.relative_to(Path(d) / "children").as_posix(), "statement": plan.get("statement"),
                    "planned_at": plan.get("time"), "sections": plan.get("sections") or [],
                    "planned": plan.get("lines") or [], "printed": replay_printed(read, dg, tot, compared, verified),
                    "printed_order": bool((dg.get("coverage") or {}).get("occurrences")),
                    # whose checks decide the outcome: those the template names ("named"), every recorded check
                    # but the report checks ("none": no statement checks named), or not known ("unknown")
                    "gate": "unknown" if gate is False else "named" if gate else "none",
                    "lines": lines, "other_calls": other,
                    "checks": [{"check": x.get("check"), "ok": x.get("ok"), "section": x.get("section"),
                                "control": any(x is y for y in sc)} for x in cks],
                    # the totals compared with the filing and the template's report checks, each with the state
                    # the workbook counts it under and whether it is a statement check (decides the outcome)
                    "controls": [{"section": x.get("section"), "kind": x.get("kind"),
                                  "recorded": x.get("state") or bucket, "state": control_state(x.get("state") or bucket),
                                  "decides": None if gate is False else not gate or x.get("section") in gate["sections"]}
                                 for bucket, xs in (ctl.items() if isinstance(ctl, dict) else ())
                                 for x in xs or () if isinstance(x, dict)]
                    + [{"section": x.get("section"), "kind": "section", "name": x.get("check"),
                        "recorded": "tied" if x.get("ok") else "failed", "state": "tied" if x.get("ok") else "failed",
                        "decides": None if gate is False else not gate or x.get("section") in gate["sections"]}
                       for x in sc],
                    "reports": [{**{f: x.get(f) for f in ("id", "name", "why")}, "recorded": x.get("state"),
                                 "state": control_state(x.get("state")),
                                 "decides": None if gate is False else bool(gate) and x.get("id") in gate["report_checks"]}
                                for x in rcs],
                    "outcome": (read_json(c / "outcome.json") or {}).get("outcome")})
    return {"job_id": Path(d).name, "statements": out}


def replay_signed(placed, concept, claims):
    """A line event's pieces, each with what it adds to its row: "signed", the signed amount of its claim in the run's
    claims.json (the amount the workbook's Lines sheet and the run's rows.json add up). A claim matches a piece by the
    line it comes from (its "source", the event's concept), the piece's own concept (a split piece records it; the
    line's remainder is the line's concept), its row and its amount; each claim is used once. The line event's amount
    carries no row sign, so adding it read wrong (treasury stock was once added
    to common equity instead of taken off). A piece with no claim gets no "signed": its contribution is
    not recorded, unless replay_dropped finds the run dropped its claim."""
    if not isinstance(placed, list):
        return placed
    pool = [cl for cl in claims if (cl.get("source") or cl.get("concept")) == concept]
    out = []
    for p in placed:
        q = dict(p) if isinstance(p, dict) else p
        if isinstance(q, dict):
            own = q.get("concept") or concept
            m = next((cl for cl in pool if cl.get("concept") == own and cl.get("row") == q.get("row")
                      and same_amount(cl.get("amount"), q.get("amount")) and cl.get("signed") is not None), None)
            if m is not None:
                pool.remove(m)
                q["signed"] = str(m["signed"])
        out.append(q)
    return out


def replay_dropped(lines, dropped):
    """Marks "dropped" the piece whose claim the run's C5 record says it dropped ("second claim dropped"; it adds
    nothing), only where the records name it alone: one such exception on the concept and one piece of that concept in
    the statement without a claim. The exception names the concept only, so with two candidates neither is said
    dropped; their contribution stays not recorded."""
    for concept, n in dropped.items():
        cand = [p for l in lines for p in (l.get("placed") if isinstance(l.get("placed"), list) else ())
                if isinstance(p, dict) and "signed" not in p and (p.get("concept") or l.get("concept")) == concept]
        if n == 1 and len(cand) == 1:
            cand[0]["dropped"] = True


def same_amount(a, b):
    try:
        return Decimal(str(a)) == Decimal(str(b))
    except (InvalidOperation, TypeError, ValueError):
        return False


def replay_gate(d):
    """(gate, controlled sections, total concepts) of the template the job ran with, read as the workbook reads it
    (result_workbook.opened_template, total_concepts). The gate is its statement checks (format v7 outcome_checks:
    {sections, report_checks}); None when the template names none (every check decides the outcome, the report checks
    excepted: spread.counted, result_workbook); False, with no
    sections and no totals, when the template cannot be opened, so no check is said to decide."""
    try:
        from spreading.result_workbook import opened_template, total_concepts
        t = opened_template(d)[1]
        return t.outcome_checks, set(t.controls), total_concepts(t)
    except (ValueError, OSError, ImportError, KeyError, TypeError):
        return False, None, set()


def control_state(s):
    """The state a recorded check is counted under, as the workbook counts it (result_workbook.control_state)."""
    s = str(s or "")
    return "tied" if s == "tied" else "failed" if s in ("differs", "failed") or s.startswith("failed") else "unavailable"


def replay_c4_why(e, c4):
    """A routed line put outside the template: the analyst's recorded reason (c4.json), as the workbook gives it."""
    if not str(e.get("class") or "").startswith("outside") or e.get("queued"):
        return None
    a = ((c4.get(e.get("concept")) or {}).get("c4") or {}) if isinstance(c4, dict) else {}
    r = (a.get("answer") or {}).get("reason") if isinstance(a.get("answer"), dict) else None
    return f"the analyst: {str(r)[:300]}" if r else None


# a total's status is the workbook's rule, shared (spreading/total_status.py, standard library only, so Replay still
# opens where the workbook's Excel support is not installed)
def replay_compared(cks, ctl, reports=(), report_defs=(), facts=None):
    """The comparisons a statement's records ran, by exact concept (total_status.compared_concepts)."""
    from spreading.total_status import compared_concepts
    return compared_concepts(cks, ctl, reports, report_defs, facts)


def statement_totals(r, report_defs=()):
    """The comparisons and the verified decompositions of one statement (total_status.statement_totals)."""
    from spreading.total_status import statement_totals as st
    return st(r, report_defs)


def total_said(k, compared, verified):
    """total_status.total_said, as Replay says it (ui/live.js rpWhere): "compared", "parts" or "none". The rule
    is the workbook's; the words are Replay's."""
    from spreading import total_status as ts
    return {ts.TOTAL: "compared", ts.TOTAL_PARTS: "parts", ts.TOTAL_NOT: "none"}[ts.total_said(k, compared, verified)]


def replay_report_defs(d):
    """The template's report checks (which sides name one filed concept), read as the workbook reads them; none when
    the template cannot be opened (replay_gate's fallback)."""
    try:
        from spreading.result_workbook import opened_template
        return opened_template(d)[1].report_checks
    except (ValueError, OSError, ImportError, KeyError, TypeError):
        return ()


def replay_printed(read, dg, totals=(), compared=None, verified=()):
    """Replay's filing panel: every line the run read, in printed order. lines.json holds only the lines routed to a
    section (a cash flow statement read for its memo row holds one), so the queued lines (diagnostics.json "queued",
    each with its line record) are added, as the workbook's Lines sheet adds them (result_workbook.printed_lines).
    The order is the coverage record's printed order when there is one; a line it does not list follows. Headings
    and printed lines with no line record (a subtotal the run only verified, an excluded line) are not shown. A line
    is a total when its record marks it a control or its concept is in `totals` (the workbook's rule, as replay's);
    a total carries the comparisons its statement's records ran on it (`compared`, replay_compared), each passed or not,
    and its status as the workbook gives it (total_said, with `verified`, the totals whose printed parts add up)."""
    own = {}
    for x in list(read) + [(q or {}).get("line") or {} for q in dg.get("queued") or ()]:
        if isinstance(x, dict) and x.get("concept") and x["concept"] not in own:
            own[x["concept"]] = x
    order = [o.get("concept") for o in (dg.get("coverage") or {}).get("occurrences") or () if isinstance(o, dict)]
    return [{"concept": k, "label": own[k].get("label"), "amount": own[k].get("amount"),
             "total": (t := bool(own[k].get("control")) or k in totals),
             "checked": list((compared or {}).get(k) or []) if t else [],
             "total_said": total_said(k, compared or {}, verified) if t else None} for k in dict.fromkeys([k for k in order if k in own] + list(own))]


def replay_outside(e, o):
    """Whether a line with no row went outside the template, by the workbook's rule (result_workbook.line_status), so
    Replay and Results say the same: a routed line whose class is "outside" (the analyst's OUTSIDE, c4.json); a queued
    line whose outcome state is "outside...", or in which the analyst answered OUTSIDE, unless the outcome is recorded
    as not settled (resolved false with an unresolved, unreviewed or disputed state). `o` is the line's queued outcome
    (diagnostics.json queued_outcomes), None when not recorded."""
    if e.get("placed") or e.get("row"):
        return False
    if not e.get("queued"):
        return str(e.get("class") or "").startswith("outside")
    o = o or {}
    state = str(o.get("state") or e.get("class") or "")
    if o.get("resolved") is False and (state in ("unresolved", "unreviewed") or state.startswith("disputed")):
        return False
    return state.startswith("outside") or (o.get("answers") or {}).get("analyst") == "OUTSIDE"


# A child whose model calls may be opened: a spread or repair child (jobs.CHILD) or the corrections child.
CALL_CHILD = re.compile(rf"(?:{CHILD.pattern})|\d\d_corrections/corrections")


def call_log(d, child):
    """The call log of a child named by CALL_CHILD (its calls.jsonl, or the corrections child's
    named_totals_calls.jsonl); nothing else is read."""
    if not isinstance(child, str) or not CALL_CHILD.fullmatch(child):
        raise ValueError(f"not a child run: {str(child)[:80]!r}")
    c = Path(d) / "children" / child
    return next((c / n for n in CALL_LOGS if (c / n).exists()), c / CALL_LOGS[0])


def call_list(d, child):
    """Every model call a child recorded, in order, without its text: number, time, role, model, tag, attempt,
    seconds, tokens, dollars and error. Each opens with `call`."""
    out = []
    for n, r in enumerate(complete_lines(call_log(d, child))):
        if r.get("role"):
            out.append({"n": n, **{f: r.get(f) for f in ("time", "role", "model", "tag", "attempt", "seconds",
                                                          "prompt_tokens", "completion_tokens", "cost_usd")},
                        "error": bool(r.get("error"))})
    return {"child": child, "calls": out}


def call(d, child, n):
    """One model call of a child (CALL_CHILD), by its line number in the child's call log: what it was sent
    (instructions, message, answer format), what came back (reply, parsed answer, reasoning) and its bookkeeping."""
    log = call_log(d, child)
    if not str(n).isdigit():
        raise ValueError("the call number must be a whole number")
    got = complete_lines(log, int(n))
    if not got or not got[0].get("role"):
        raise ValueError(f"no model call {n} in {child}")
    r = got[0]
    return {"child": child, "n": int(n), **{f: r.get(f) for f in CALL_FIELDS}}
