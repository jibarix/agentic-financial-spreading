"""Grader: one filing's spread runs against its sealed AI reference. Offline, no model call. Two commands, each writing a run folder that is
never overwritten:

    .venv-agents/Scripts/python.exe -s -E -m spreading.grade frame REFERENCE.json [--filing FOLDER] [--statement CF]
    .venv-agents/Scripts/python.exe -s -E -m spreading.grade grade REFERENCE.json FRAME_RUN RUN [RUN ...]

`frame` freezes, before any call, the lines spread.prepare sends to models for each statement (routed and queued only),
with the template's hash. `grade` reads that frame and one spread run folder per statement. The frame is the
denominator: a frame line a run lost or never reached stays in it as "no answer".

Facts compare by (concept, context, unit, value): two fact IDs with the same key are one fact (an exact alias); the
same concept, context and unit with different values is refused. A frame line joins the reference line in its
statement with the same fact. A frame line with no reference line, several, or one the reference does not place
(not_applicable, exception) is "unresolved", as is a run line outside the frame (one the organizer sorted is newly
covered, listed apart); any unresolved line blocks a comparison of runs. A statement with no run (refused) is
graded "no answer" on every frame line. Reference lines the frame never sends are listed by kind, the same for every
arm.

A line gets three separate fields: correctness (correct, wrong, no answer), completion (committed, OUTSIDE or another
answer, unfinished, not run, no run) and review (agree, disputed, unreviewed). Correct means the committed
contributions, row by row, with their signs, equal one of the reference's accepted readings; each claim's signed
amount is first checked against its facts' values. An OUTSIDE answer is correct against outside_template. A reference
with open lines or unscored statuses is refused (not yet supported). "Correct" is agreement with the AI reference, not
accuracy."""
import argparse
import hashlib
import itertools
import json
import re
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .answer_keys import consistent
from .keycheck import load_instance
from .runrec import new_run, write

ROOT = Path(__file__).resolve().parent.parent
FINISHED = ("finished", "finished_at_last_step")
STATUSES = ("exact", "alternatives")
SIGN = {"add": 1, "subtract": -1}
PLACED = ("row", "pieces", "outside_template", "filing_does_not_tell")     # the reference kinds a frame line grades on
INTERPRETATIONS = ROOT / "data/references/grading_interpretations_v3.json"   # optional: readings that replace named reference lines


class Refused(ValueError):
    """The reference, the frame or a run holds something the grader does not know; it stops rather than guess."""


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_seal(path):
    """The reference's sha256, refused unless its .sha256 file ("hash  name" or "hash *name" lines) lists it."""
    path = Path(path)
    seal = {}
    for ln in path.with_suffix(".sha256").read_text(encoding="utf8").splitlines():
        m = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", ln.strip())
        if m:
            seal[m.group(2)] = m.group(1)
    digest = sha(path)
    if seal.get(path.name) != digest:
        raise Refused(f"{path.name}: the seal does not match")
    return digest


def fact_keys(facts):
    """fk for the filing's instance: a fact dict (reference) or ID (run) -> (concept, context, unit, value, decimals).
    One key is one fact, as in the spread's own alias rule (steps.Eligibility: same concept, context, unit, value and
    decimals). Facts of one concept, context and unit that differ only in precision (9,160,000 at -3 and 9,200,000 at
    -5) stay two facts; a group that is not consistent (answer_keys.consistent: XBRL's interval rule over the whole
    group, and one value per decimals) is refused when asked for."""
    def value(v):
        try:
            return Decimal(v)
        except (InvalidOperation, TypeError):
            return v
    group = defaultdict(set)
    for x in facts.values():
        group[(x["concept"], x["context"], x["unit"])].add((value(x["value"]), x.get("decimals")))

    def conflict(vs):
        if not all(isinstance(v, Decimal) for v, _ in vs):
            return len({v for v, _ in vs}) > 1
        return not consistent([(v, "INF" if d in (None, "INF") else int(d)) for v, d in vs])

    def fk(i):
        x = facts[i if isinstance(i, str) else i["id"]]
        k = (x["concept"], x["context"], x["unit"])
        if conflict(list(group[k])):
            raise Refused(f"{k[0]} in {k[1]}: duplicate facts that disagree {sorted(map(str, group[k]))}")
        return (*k, value(x["value"]), x.get("decimals"))
    return fk


def contributions(parts):
    """{row: {fact key: coefficient}} summed, zeros dropped: a piece left in its line's own row cancels out."""
    out = defaultdict(lambda: defaultdict(Decimal))
    for row, key, coef in parts:
        out[row][key] += Decimal(coef)
    return {r: {k: v for k, v in fs.items() if v} for r, fs in out.items() if any(fs.values())}


def readings(ln, fk):
    """Every accepted reading of a placed reference line: a list of contributions, or "OUTSIDE" / "NOT_TOLD". Line-level
    accepted_rows replace the row (or the remainder row); piece-level ones replace that piece's row."""
    p, d = ln["placement"], SIGN.get(ln["direction"])
    if p["kind"] == "outside_template":
        return ["OUTSIDE"]
    if p["kind"] == "filing_does_not_tell":
        return ["NOT_TOLD"]
    if p["kind"] not in ("row", "pieces") or d is None:
        raise Refused(f"{ln['line_id']}: placement kind {p['kind']} with direction {ln['direction']} is not graded")
    head = p["row"] if p["kind"] == "row" else p["remainder_row"]
    out = []
    for h in [head] + ln["grading"]["accepted_rows"]:
        for rows in itertools.product(*[[x["row"]] + x["grading"]["accepted_rows"] for x in p["pieces"]]):
            parts = [(h, fk(ln["fact"]), d)]
            for x, r in zip(p["pieces"], rows):
                s = SIGN[x["direction"]]
                parts += [(r, fk(x["fact"]), s), (h, fk(x["fact"]), -s)]
            out.append(contributions(parts))
    return out


def interpreted(ref, spec, fk, contexts, frame):
    """{line_id: the reference line as graded} for the totals the grading interpretations name (spec:
    {line_id: {"reading", "row", "members"}}). Every condition is checked here; one that
    fails refuses the reference rather than grade on it."""
    by = {ln["line_id"]: ln for ln in ref["lines"]}
    out = {}
    for lid, s in spec.items():
        ln = by[lid]
        st, t = ln["statement"], fk(ln["fact"])
        checks = {"a total the reference does not place": ln["placement"]["kind"] == "not_applicable"
                  and ln["record_type"] == "total"}
        if s["reading"] == "outside":
            checks["no total row"] = ln["total_row"] is None
            placement = {"kind": "outside_template", "row": None, "pieces": [], "remainder_row": None}
            direction = None
        else:
            ms = [by[m] for m in s["members"]]
            keys = [fk(m["fact"]) for m in ms]
            on_row = {x["line_id"] for x in ref["lines"] if x["statement"] == st and x["placement"]["row"] == s["row"]}
            sent = {fk(x["fact"]) for x in frame[st]["lines"]}
            checks.update({
                "its total row": ln["total_row"] == s["row"],
                "every line on the row": on_row == set(s["members"]),
                "exact rows without pieces": all(m["placement"]["kind"] == "row" and not m["placement"]["pieces"]
                                                 and m["grading"] == {**m["grading"], "status": "exact",
                                                                      "accepted_rows": []} for m in ms),
                "one shared direction": len({m["direction"] for m in ms}) == 1 and ms[0]["direction"] in SIGN,
                "the total's unit and period": all(k[2] == t[2] and contexts[k[1]][0] == contexts[t[1]][0]
                                                   for k in keys),
                "sums exactly to the total": sum(k[3] for k in keys) == t[3],
                "no member in the frame": not sent & set(keys)})
            placement = {"kind": "row", "row": s["row"], "pieces": [], "remainder_row": None}
            direction = ms[0]["direction"]
        failed = [k for k, ok in checks.items() if not ok]
        if failed:
            raise Refused(f"{lid}: the grading interpretation does not hold: {failed}")
        out[lid] = {**ln, "placement": placement, "direction": direction, "interpretation": s["reading"]}
    return out


def check_reference(ref):
    """Open lines and unscored statuses are not supported yet."""
    if ref["open_lines"]:
        raise Refused(f"the reference has open lines {ref['open_lines']}; not supported")
    for ln in ref["lines"]:
        for g in [ln["grading"]] + [x["grading"] for x in ln["placement"]["pieces"]]:
            if g["status"] not in STATUSES:
                raise Refused(f"{ln['line_id']}: grading status {g['status']} not supported")


def frame_of(f, statements=None):
    """{statement: {"sections": [...], "lines": [{concept, fact, how}]}}, for each statement of the template, or those
    named (a reference that covers some statements only): what spread.prepare sends to
    models, without A2, over the sections a run of the statement takes (template.run_sections: memo sections
    included). Side-effect free: no model, no search, no embedding."""
    from . import spread, template
    from .taxonomy import Taxonomy, edition
    t = template.active()
    tax = Taxonomy(edition(spread.taxonomy_year(f)))
    out = {}
    for st in statements or t.statements:
        secs = t.run_sections(st)
        p = spread.prepare(f, tax, st, secs)
        lines = [{"concept": l["concept"], "fact": l["fact"], "how": "routed"} for s in p["sections"]
                 for l in p["st"][s]["queue"]]
        lines += [{"concept": q["line"]["concept"], "fact": q["line"]["fact"], "how": "queued"} for q in p["queued"]]
        out[st] = {"sections": p["sections"], "lines": lines}
    return out


def load_run(d):
    """(statement, run.json, {concept: what the spread did}) for one spread run folder. A committed answer is its
    claims: (row, [(fact ID, coefficient)], signed amount). A total the job's corrections record retagged (the run's
    "retag" exception) is keyed by the filing's own tag, as the reference's frame reads it; its fact IDs are the
    filing's own facts. A retag onto a tag the run also prints is refused."""
    from . import template
    d = Path(d)
    rd = lambda n: json.loads((d / n).read_text(encoding="utf8"))
    meta, c4, alloc, claims = rd("run.json"), rd("c4.json"), rd("placements.json"), rd("claims.json")
    exc, diag = rd("exceptions.json"), rd("diagnostics.json")
    facts = {l["concept"]: l["fact"] for l in rd("lines.json")}
    secs = meta["section"] if isinstance(meta["section"], list) else [meta["section"]]
    t = template.active()
    statement = {t.statement_of(s) for s in secs}
    if len(statement) != 1:
        raise Refused(f"{d.name}: sections of {len(statement)} statements")
    queued = diag.get("queued_outcomes") or {}
    out = {}
    for c in set(c4) | set(queued) | {x["concept"] for x in exc if x.get("class") == "not run" and x.get("concept")}:
        e = c4.get(c) or {}
        e = e.get("transfer") or e
        rec, q = e.get("c4") or {}, queued.get(c) or {}
        review = (e.get("consistency") or {}).get("result") or q.get("state")
        if c in alloc:
            done = "committed"
            answer = [(x["row"], [(f, Decimal(str(coef))) for f, coef, _ in x["facts"]], Decimal(str(x["signed"])))
                      for x in claims if x["source"] == c]
        elif rec.get("termination") in FINISHED:
            done = answer = rec["decision"] if rec["decision"] in ("OUTSIDE", "COMBINED", "NOT_TOLD") else (
                f"{rec['decision']} {rec.get('target') or rec.get('row')}")
        elif rec.get("termination"):
            done, answer = f"unfinished: {rec['termination']}", None
        else:
            done, answer = q.get("state") or "not run", None
        out[c] = {"completion": done, "answer": answer, "review": review if c in alloc or c in queued else None,
                  "queued": c in queued, "run": d.name, "fact": facts.get(c)}
    back = {x["concept"]: x["original_concept"] for x in exc if x.get("step") == "corrections"}
    clash = sorted(o for n, o in back.items() if o in out and n in out)
    if clash:
        raise Refused(f"{d.name}: a retagged total's original tag is also a line of the run: {clash}")
    return statement.pop(), meta, {back.get(c, c): v for c, v in out.items()}


def committed(claims, fk):
    """The contributions of a committed answer, after checking each claim's signed amount against its facts."""
    parts = []
    for row, fs, signed in claims:
        if sum(coef * fk(f)[3] for f, coef in fs) != signed:
            raise Refused(f"claim in {row}: signed amount {signed} differs from its facts {fs}")
        parts += [(row, fk(f), coef) for f, coef in fs]
    return contributions(parts)


def grade_line(ln, got, fk):
    """One placed reference line against what the spread did with its frame line (None: no run reached it)."""
    if got is None:
        return {"completion": "no run", "review": None, "answer": None, "correctness": "no answer"}
    r = {"completion": got["completion"], "review": got["review"], "run": got["run"]}
    ans = got["answer"]
    if isinstance(ans, list):
        ans = committed(ans, fk)
        r["answer"] = {row: {"/".join(map(str, k)): str(v) for k, v in fs.items()} for row, fs in ans.items()}
    else:
        r["answer"] = ans
    if ans in (None, "COMBINED") or (isinstance(ans, str) and ans not in ("OUTSIDE", "NOT_TOLD")):
        return {**r, "correctness": "no answer"}
    return {**r, "correctness": "correct" if ans in readings(ln, fk) else "wrong"}


def grade(ref, frame, runs, fk, interp=None, organized=()):
    """(lines, counts, exceptions) for one reference, its frozen frame {statement: {"lines": [...]}} and its runs
    {statement: {concept: ...}}. `fk` maps a fact ID or dict to its (concept, context, unit, value, decimals);
    `interp` replaces the reference lines the grading interpretations name (interpreted). `organized`: the statements
    whose run the organizer sorted; a run line of theirs outside the frame is newly covered, reported apart, and does
    not block (the frame stays fixed). One whose fact is one
    placed reference line (a split piece) is graded against it, counted under "newly covered" only, never in the
    frame's counts (the denominator does not grow); a reference line is
    credited once, never one the frame already joins, and a piece that matches none or several says so."""
    check_reference(ref)
    ref = {**ref, "lines": [(interp or {}).get(ln["line_id"], ln) for ln in ref["lines"]]}
    exc, lines, joined = [], [], set()
    for st, fr in frame.items():
        concepts = Counter(x["concept"] for x in fr["lines"])
        if any(n > 1 for n in concepts.values()):
            raise Refused(f"{st}: a concept on two frame lines {[c for c, n in concepts.items() if n > 1]}")
        run = runs.get(st)
        framed = {ln["line_id"] for x in fr["lines"] for ln in ref["lines"]
                  if ln["statement"] == st and ln["fact"] and fk(ln["fact"]) == fk(x["fact"])}
        for c in sorted(set(run or {}) - set(concepts)):
            new = st in organized
            exc.append({"statement": st, "concept": c, "blocks": not new, "why": "newly covered, outside the frozen "
                        "frame" if new else "a run line outside the frame"})
            got = run[c]
            refs = [ln for ln in ref["lines"] if new and got["fact"] and ln["statement"] == st and ln["fact"]
                    and fk(ln["fact"]) == fk(got["fact"]) and ln["placement"]["kind"] in PLACED]
            if new and got["fact"] and len(refs) != 1:
                exc.append({"statement": st, "concept": c, "blocks": False, "why": f"newly covered: {len(refs)} "
                            "placed reference lines with its fact, not one; not graded"})
            elif refs and refs[0]["line_id"] in framed | joined:
                exc.append({"statement": st, "concept": c, "blocks": False, "why": "newly covered: reference line "
                            f"{refs[0]['line_id']} is already graded; not graded twice"})
            elif refs:
                ln = refs[0]
                joined.add(ln["line_id"])
                lines.append({"statement": st, "concept": c, "fact": got["fact"], "how": "newly covered",
                              "run_fact": got["fact"], "line_id": ln["line_id"], "label": ln["label"],
                              "expected": ln["placement"], "interpretation": ln.get("interpretation"),
                              **grade_line(ln, got, fk)})
        for x in fr["lines"]:
            c, key = x["concept"], fk(x["fact"])
            refs = [ln for ln in ref["lines"] if ln["statement"] == st and ln["fact"] and fk(ln["fact"]) == key]
            joined |= {ln["line_id"] for ln in refs}
            got = None if run is None else run.get(c)
            if got and got["fact"] and fk(got["fact"]) != key:
                raise Refused(f"{st} {c}: the run's fact differs from the frame's")
            g = {"statement": st, "concept": c, "fact": x["fact"], "how": x["how"], "run_fact": got and got["fact"]}
            if got is None and run is not None:
                got = {"completion": "not in the run", "answer": None, "review": None, "run": None}
            if len(refs) != 1 or refs[0]["placement"]["kind"] not in PLACED:
                why = (f"{len(refs)} reference lines with this fact" if len(refs) != 1 else
                       f"the reference does not place it ({refs[0]['line_id']}: {refs[0]['placement']['kind']})")
                exc.append({"statement": st, "concept": c, "why": why, "blocks": True})
                lines.append({**g, "line_id": [ln["line_id"] for ln in refs], "correctness": "unresolved",
                              "why": why, "completion": got and got["completion"]})
                continue
            ln = refs[0]
            lines.append({**g, "line_id": ln["line_id"], "label": ln["label"], "expected": ln["placement"],
                          "interpretation": ln.get("interpretation"), **grade_line(ln, got, fk)})
    outside = [ln for ln in ref["lines"] if ln["statement"] in frame and ln["line_id"] not in joined]
    counts = defaultdict(Counter)
    for g in lines:
        if g["how"] == "newly covered":
            counts["newly covered"][g["correctness"]] += 1
            continue
        for by in ("all", g["statement"], f"{g['statement']} {g['how']}"):
            counts[by][g["correctness"]] += 1
            counts[by]["completed"] += g.get("completion") in ("committed", "OUTSIDE")
        counts["review"][f"{g.get('review')}: {g['correctness']}"] += 1
    for ln in outside:
        counts["reference lines outside the frame"][f"{ln['statement']} {ln['placement']['kind']}"] += 1
    blocks = [x for x in exc if x.get("blocks")]
    counts = {k: dict(v) for k, v in counts.items()}
    counts["decision"] = {"blocked": bool(blocks), "unresolved": len(blocks)}
    return lines, counts, exc + [{"line_id": ln["line_id"], "statement": ln["statement"], "kind":
                                  ln["placement"]["kind"], "why": "a reference line the frame never sends"}
                                 for ln in outside]


def filing_of(ref, folder=None):
    # a recorded folder from another copy of the project (absolute path) is looked up here instead
    fil = Path(folder) if folder and Path(folder).exists() else \
        next(ROOT.glob(f"data/filings*/{ref['cik']}/{ref['accession']}"), None)
    if fil is None:
        raise Refused(f"the reference's filing {ref['cik']}/{ref['accession']} is not under data/filings")
    if (fil.parent.name, fil.name) != (ref["cik"], ref["accession"]):
        raise Refused(f"{fil}: not the reference's filing {ref['cik']}/{ref['accession']}")
    return fil


def template_of(ref):
    """The template file the reference was written on; refused unless the active template is that file, unchanged."""
    from . import provenance
    p = Path(provenance.TEMPLATE).resolve()
    if p != (ROOT / ref["template"]["path"]).resolve() or sha(p) != ref["template"]["sha256"]:
        raise Refused(f"the active template {p} is not the reference's {ref['template']}")
    return str(p), ref["template"]["sha256"]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m spreading.grade")
    sub = ap.add_subparsers(dest="cmd", required=True)
    fr = sub.add_parser("frame", help="freeze the frame of one reference's filing, before any call")
    fr.add_argument("reference")
    fr.add_argument("--filing", help="default: data/filings*/<cik>/<accession>")
    fr.add_argument("--statement", nargs="+", help="the statements graded (default: every statement of the template); "
                    "each must have reference lines")
    gr = sub.add_parser("grade", help="grade one spread run per statement against the reference, on the frame")
    gr.add_argument("reference")
    gr.add_argument("frame", help="a frame run folder")
    gr.add_argument("runs", nargs="*", help="a statement with no run (refused) is graded blank")
    a = ap.parse_args(argv)
    ref_path = Path(a.reference)
    digest = check_seal(ref_path)
    ref = json.loads(ref_path.read_text(encoding="utf8"))
    tpl = template_of(ref)
    if a.cmd == "frame":
        from .filing import Filing
        fil = filing_of(ref, a.filing)
        bare = [st for st in a.statement or () if not any(ln["statement"] == st for ln in ref["lines"])]
        if bare:
            raise Refused(f"the reference has no lines for {bare}")
        frame = frame_of(Filing(fil), a.statement)
        d = new_run("frame", {"reference": str(ref_path), "reference_sha256": digest, "filing": str(fil),
                              "template": {"path": tpl[0], "sha256": tpl[1]}, "a2": None,
                              "statements": sorted(frame),
                              "counts": {st: len(x["lines"]) for st, x in frame.items()}})
        write(d, "frame.json", frame)
        print(d)
        return d
    fmeta = json.loads((Path(a.frame) / "run.json").read_text(encoding="utf8"))
    frame = json.loads((Path(a.frame) / "frame.json").read_text(encoding="utf8"))
    fil = filing_of(ref, fmeta["filing"])
    if fmeta["reference_sha256"] != digest or fmeta["template"]["sha256"] != tpl[1]:
        raise Refused(f"{a.frame}: frozen on another reference or template")
    runs, organized = {}, set()
    for d in a.runs:
        st, meta, got = load_run(d)
        filing_of(ref, meta["filing"])
        if tpl[1] not in (meta["hashes"].get("approved_resources") or {}).values():
            raise Refused(f"{d}: run on another template")
        if not meta.get("no_a2") or st not in frame or meta["section"] not in (frame[st]["sections"],
                                                                                *frame[st]["sections"]):
            raise Refused(f"{d}: scope differs from the frame (sections, statement or A2)")
        if st in runs:
            raise Refused(f"two runs for {st}")
        runs[st] = got
        if "organizer" in meta:
            organized.add(st)
    facts, _, contexts, _ = load_instance(fil)
    fk = fact_keys(facts)
    spec = json.loads(INTERPRETATIONS.read_text(encoding="utf8"))["references"].get(digest, {"lines": {}})["lines"]
    lines, counts, exc = grade(ref, frame, runs, fk, interpreted(ref, spec, fk, contexts, frame), organized)
    d = new_run("grade", {"reference": str(ref_path), "reference_sha256": digest, "frame": str(a.frame),
                          "filing": str(fil), "runs": a.runs, "label": "agreement with the AI reference, not accuracy",
                          "interpretations": {"path": str(INTERPRETATIONS.relative_to(ROOT)),
                                              "sha256": sha(INTERPRETATIONS), "lines": sorted(spec)}})
    write(d, "lines.json", lines)
    write(d, "counts.json", counts)
    write(d, "exceptions.json", exc)
    print(json.dumps(counts, indent=1))
    print(d)
    return d


if __name__ == "__main__":
    main()
