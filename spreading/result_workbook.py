"""The workflow's result as an Excel workbook, labelled "Workflow result - unreviewed", read from one job's
own records: never shown as approved, and edits made in Excel do not return to the system.

Sheets:
  About      - the job, filing, template file and fingerprint, the status, how to correct, the counts (summary());
  Lines      - every printed line in filing order (a split line one row per part): amount, period, unit, fact,
               status, the proposed row with its definition and short evidence, the checker's row where it differed,
               and correction columns ("Your row", "Your signed amount") that keep the original prediction beside them;
  Template   - the template's rows; each item row recalculates from the Lines sheet's corrections, each subtotal from
               its rows by the template's recipe, beside the workflow's own amount;
  Checks     - each check as the workflow recorded it (tied / failed / unavailable; code's line checks passed /
               failed), and the template's controls and report checks recalculated from the Template sheet; the
               Assessment column marks a recorded failure over rows with no line placed, without changing it;
  Review     - the lines and checks a person should look at first: unresolved, failed and outside lines, checker
               disagreements, catch-all placements, failed or unavailable checks, open overlap claims;
  Not spread - the template's not_spread list (format v5).

Every workbook is read back against the snapshot it was written from; the record (runs/<stamp>_result_workbook) keeps
the workbook, the snapshot and the read-back with the workbook's frozen identity (name and sha256).

    python -m spreading.result_workbook JOB_FOLDER
"""
import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.utils import get_column_letter

from spreading import runrec
from spreading.organizer import NAMES     # the statements' names, kept with the organizer
from spreading.template import Template
# a total's status: one rule for the Lines sheet and Replay (ui/views.py), kept where Replay can import it
from spreading.total_status import TOTAL, TOTAL_NOT, TOTAL_PARTS, TOTALS, statement_totals, total_said

ROOT = Path(__file__).resolve().parents[1]
TITLE = "Workflow result — unreviewed"
UNRESOLVED = "unresolved: not established from the allowed statements"
PLACED, OUTSIDE = "placed", "outside the template"
# TOTAL ("total: compared in Checks"), TOTAL_PARTS, TOTAL_NOT: spreading/total_status.py
# ("compared in Checks" only when a comparison ran on that statement, for exactly that concept)
STATES = (PLACED, OUTSIDE, UNRESOLVED, "failed")
EXCLUDED, NOT_AMOUNT, NOT_ROUTED = "excluded from the spread", "not a currency amount", "not routed to a section"
CONTEXT = "printed subtotal: shown for context, not placed"   # a line record marked context (spread.py)
# a printed line the run had nothing to place for: not a failure (not "failed (coverage
# state ... with no line record)"; for example a line the statement prints for one year only)
NO_AMOUNT = "no amount for this period"
GIVEN = {"statement control": TOTAL, "verified subtotal": TOTAL, "excluded": EXCLUDED,
         "not a currency amount": NOT_AMOUNT, "no amount for the period": NO_AMOUNT,
         "second occurrence": "printed again: read at its first occurrence",
         "period type differs": "not spread: its period type differs from the statement's",
         "printed by member only": "not spread: printed by member only, with no total amount"}

LINES = ["Statement", "Section", "Printed line", "Line concept", "Part (tagged concept)", "Fact", "Period", "Unit",
         "Amount as filed", "Signed amount in the row", "Status", "Proposed row", "Proposed row name", "Row definition",
         "Evidence (quote)", "Reason", "Checker's row (where it differed)", "Checker's evidence", "Your row",
         "Your signed amount", "Row used", "Amount used", "Source"]
COL = {h: get_column_letter(i) for i, h in enumerate(LINES, 1)}
TEMPLATE = ["Statement", "Section", "Row", "Row name", "Type", "Policy", "Workflow amount", "Recalculated amount",
            "Definition"]
CHECKS = ["Statement", "Kind", "Section", "Check", "Filed concept", "Filed amount", "Workflow built", "Workflow gap",
          "Tolerance", "Workflow state", "Why", "Recalculated built", "Recalculated gap", "Recalculated state",
          "Assessment"]
# the Assessment column, beside the workflow's recorded state (kept as recorded): a failed control over rows with no
# line placed compares nothing the workflow placed (the recorded failure is not rewritten)
NO_ROWS = ("no line placed in any row it compares: the gap is the whole printed total, so the failure says nothing "
           "about the placements")
REVIEW = ["Statement", "Why listed", "Printed line or check", "Line concept", "Amount as filed", "Proposed row",
          "Checker's row", "Detail"]
NOT_SPREAD = ["Input", "Why it is not spread"]
NO_ROW = ("OUTSIDE", "NOT_TOLD", "COMBINED")
UNKNOWN = "unknown"          # a row with no line placed in it: not shown separately, never zero
FRAMED = STATES + (NOT_ROUTED,)  # the states an eligible line can have; a line routed to no section counts against it


class Formula(str):
    """A cell written as an Excel formula (every other text is written as text)."""


def read(p):
    return json.loads(Path(p).read_text(encoding="utf8"))


def dec(x):
    """A recorded amount as Decimal; None for no amount or for text that is not a number (a concept name)."""
    if x is None or isinstance(x, bool):
        return None
    try:
        return Decimal(str(x))
    except ArithmeticError:
        return None


def short(x, n=300):
    x = " ".join(str(x or "").split())
    if not x:
        return None
    return x if len(x) <= n else x[:n - 1] + "…"


# An Excel cell holds 32,767 characters. A Review detail an exception gives is written whole up to that; its numbers
# (the built and filed amounts of the check it names) and its disposition come last, and a cut at 400 characters lost
# them (an S1 text lost its "filed <amount>").
CELL = 32767
TAIL = 2000


def whole(x):
    """The text whitespace-normalized and whole when it fits in a cell; longer, its start and its last TAIL characters
    (the amounts and the disposition) with the omission marked."""
    x = " ".join(str(x or "").split())
    if not x:
        return None
    if len(x) <= CELL:
        return x
    head = CELL - TAIL - len(f" … [{len(x):,} characters left out] … ")      # the mark at its longest
    return x[:head] + f" … [{len(x) - head - TAIL:,} characters left out] … " + x[-TAIL:]


def exception_line(r, e):
    """The printed line an exception names: the line with its concept and fact, as lines.json records them; with no
    fact named, the only line with its concept; otherwise {}."""
    got = [ln for ln in printed_lines(r) if ln.get("concept") == e.get("concept")]
    if e.get("fact"):
        return next((ln for ln in got if ln.get("fact") == e.get("fact")), {})
    return got[0] if len(got) == 1 else {}


def statement_dirs(job):
    """statement -> the folder of its latest cumulative version (cumulative/<ST>_v<N>, the highest N)."""
    out = {}
    for d in sorted((Path(job) / "cumulative").glob("*_v*")):
        st, _, n = d.name.rpartition("_v")
        if d.is_dir() and n.isdigit() and (st not in out or int(n) > out[st][0]):
            out[st] = (int(n), d)
    return {st: d for st, (_, d) in out.items()}


def opened_template(job):
    """The template the job ran with (its manifest's path and sha256), checked against the file on disk."""
    m = read(Path(job) / "ui_manifest.json") if (Path(job) / "ui_manifest.json").is_file() else None
    t = (m or {}).get("template")
    if not isinstance(t, dict) or not t.get("path") or not t.get("sha256"):
        raise ValueError(f"{Path(job).name}: the job recorded no template file and fingerprint (ui_manifest.json), "
                         "so its workbook cannot be checked against the template it ran with")
    path = ROOT / t["path"]
    import hashlib
    if hashlib.sha256(path.read_bytes()).hexdigest() != t["sha256"]:
        raise ValueError(f"the template file {t['path']} no longer has the fingerprint the job ran with")
    return m, Template(read(path), path)


def line_status(c, claims, c4, queued, control):
    """A line's status from the run's records: placed (it has a claim on a row), outside the template, unresolved
    (no row established from the allowed statements), failed (the loop ended without a valid answer), or a total the
    checks compare. Returns (status, why)."""
    if control:
        return TOTAL, None
    if claims:
        return PLACED, None
    q = queued.get(c) or {}
    a = (c4.get(c) or {}).get("c4") or {}
    decision = a.get("decision")
    state = str(q.get("state") or "")
    if q.get("resolved") is False and (state in ("unresolved", "unreviewed") or state.startswith("disputed")):
        # J4: an unsettled queued line is unresolved, whatever the analyst answered (an analyst OUTSIDE included)
        return UNRESOLVED, q.get("why") or short((a.get("answer") or {}).get("reason"))
    if state.startswith("outside") or decision == "OUTSIDE" or (q.get("answers") or {}).get("analyst") == "OUTSIDE":
        return OUTSIDE, q.get("why") or short((a.get("answer") or {}).get("reason"))
    if decision in ("NOT_TOLD", "COMBINED") or state == "unresolved":
        return UNRESOLVED, q.get("why") or short((a.get("answer") or {}).get("reason"))
    why = q.get("why") or a.get("why") or (c4.get(c) or {}).get("why") or "no answer recorded for this line"
    return "failed", short(why)


def printed_lines(r):
    """Every printed line of a statement result: the lines the run read (lines.json), the queued lines, and the
    printed lines the coverage record lists with a state of their own (a total, a line excluded from the spread, a
    number that is not a currency amount); a sectioned run without a coverage record adds its unrouted lines. A line
    not sent to the spread carries `given` (status, why)."""
    dg = r["diagnostics"] or {}
    lines = [dict(x) for x in r["lines"] or ()]
    have = {x["concept"] for x in lines}
    for q in dg.get("queued") or ():
        if q["line"]["concept"] not in have:
            lines.append(dict(q["line"]))
            have.add(q["line"]["concept"])
    occ = (dg.get("coverage") or {}).get("occurrences")
    if occ is None:
        return lines + [{"concept": c, "given": (NOT_ROUTED, "the run routed it to no section")}
                        for c in dg.get("unrouted_lines") or () if c not in have]
    for o in occ:
        if o["state"] == "structure" or o["concept"] in have:
            continue            # a heading, or a line already shown from the run's own line record
        given = (GIVEN[o["state"]], o.get("reason")) if o["state"] in GIVEN else \
            ("failed", f"coverage state {o['state']!r} with no line record")
        lines.append({"concept": o["concept"], "label": o.get("label"), "given": given})
    return lines


def checker_of(entry, rows):
    """The checker's row and evidence when it differed from the placement `rows`; (None, None) otherwise."""
    r = (entry or {}).get("review") or {}
    if not r:
        return None, None
    ans = r.get("answer") or {}
    got = {ans.get("row") or r.get("row")} | {p.get("row") for p in ans.get("pieces") or () if isinstance(p, dict)}
    got.discard(None)
    if got == set(rows):
        return None, None
    return ", ".join(sorted(got)) or r.get("decision"), short(ans.get("quote"), 200)


def source_link(filing):
    """The filing's folder on EDGAR, from its local path data/filings/<cik>/<accession>."""
    cik, acc = Path(filing).parts[-2:]
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/"


def total_concepts(t):
    """The concepts the template compares as printed totals: each control's filed concept, the concepts of its
    reconstruction and alternatives, and both sides of a report check that compares two concepts."""
    out = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("filed", "concept") and isinstance(v, str):
                    out.add(v)
                else:
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    for c in t.controls.values():
        walk({k: c.get(k) for k in ("filed", "reconstructed", "alternatives")})
    for c in t.report_checks:
        if all("concept" in (c.get(k) or {}) for k in ("left", "right")):
            walk([c["left"], c["right"]])
            out.update(a for v in (c.get("alternatives") or {}).values() for a in v)   # format v6
    return out


def framed_statements(t):
    """The statements the coverage frame counts: those with a section the template spreads (a statement read only
    for a memo row is reported apart)."""
    return [st for st in t.statements if any(s["statement"] == st and s.get("kind") == "spread"
                                             for s in t.data["sections"])]


def in_frame(st, status, concept, amount, framed, totals):
    """An eligible line of the coverage frame: a printed line of a spread statement that is not a total the
    template compares, not excluded, not a non-currency number and not a zero amount. A line routed to no section or
    failed counts even without a recorded amount (it was never read)."""
    if st not in framed or status not in FRAMED or concept in totals:
        return False
    a = dec(amount)
    return (a is not None and a != 0) or (a is None and status in (NOT_ROUTED, "failed"))


def engine_rows(t):
    """The rows whose routing the Review sheet lists: the template's `engine_inputs` (row ids) when it names them,
    otherwise every item and memo row of the template."""
    named = t.data.get("engine_inputs")
    if named:
        return [r if isinstance(r, str) else r["row"] for r in named]
    return [rid for rid, r in t.rows.items() if r["line_type"] in ("item", "supplementary")]


def snapshot(job):
    """The workbook's content from the job's records: [(sheet, headers, rows)], amounts as Decimal, formulas as
    Formula; and the summary (counts per statement)."""
    job = Path(job)
    m, t = opened_template(job)
    dirs = statement_dirs(job)
    if not dirs:
        raise ValueError(f"{job.name}: no statement result recorded (cumulative/ is empty)")
    link = source_link(m["filing"])
    lines, tpl, checks, review, summary = [], [], [], [], {}
    order = [st for st in t.statements if st in dirs]
    framed, totals = framed_statements(t), total_concepts(t)
    job_rec = read(job / "job.json") if (job / "job.json").is_file() else {}     # each statement's outcome, the skips
    recs = {}
    for st in order:
        d = dirs[st]
        r = {n: read(d / f"{n}.json") if (d / f"{n}.json").exists() else None
             for n in ("lines", "claims", "c4", "diagnostics", "rows", "computed_totals", "controls", "report_checks",
                       "checks", "exceptions")}
        recs[st] = r
        by_line = {}
        for cl in r["claims"] or ():
            by_line.setdefault(cl["concept"], []).append(cl)
        queued = (r["diagnostics"] or {}).get("queued_outcomes") or {}
        c4 = r["c4"] or {}
        compared, verified = statement_totals(r, t.report_checks)
        frame = {s: [0, Decimal(0)] for s in FRAMED}
        for ln in printed_lines(r):
            c = ln["concept"]
            # a printed total the template compares is a total even where the run's line record does not mark it
            # (the balance sheet's grand total, read by the run-wide BS.TOT, is not "failed")
            # a printed subtotal the spread shows for context and never asks to place (such as "Net cash provided by
            # operating activities" under the cash flow memo heading) is not "failed" either
            status, why = ln.get("given") or ((CONTEXT, ln.get("reason")) if ln.get("context") and not by_line.get(c)
                                              and not (ln.get("control") or c in totals) else None) or                 line_status(c, by_line.get(c), c4, queued, ln.get("control") or c in totals)
            if status == TOTAL:
                status = total_said(c, compared, verified)
            entry = c4.get(c) or {}
            ans = (entry.get("c4") or {}).get("answer") or {}
            parts = by_line.get(c) or [None]
            chk_row, chk_ev = checker_of(entry, [p["row"] for p in parts if p]) if status == PLACED else \
                checker_of(entry, []) if status in STATES else (None, None)
            period = ln.get("period") or []
            when = (f"at {period[1]}" if period and period[0] == "instant" else " to ".join(period[1:])) or None
            if in_frame(st, status, c, ln.get("amount"), framed, totals):
                frame[status][0] += 1
                frame[status][1] += abs(dec(ln.get("amount")) or 0)
            for p in parts:
                row = p["row"] if p else None
                rd = t.rows.get(row) or {}
                n = len(lines) + 2
                lines.append([st, ln.get("section") or entry.get("section"), ln.get("label"), c,
                              p["source"] if p and p.get("source") != c else None, (p or ln).get("fact"), when,
                              ln.get("unit"), dec(p["amount"] if p else ln.get("amount")),
                              dec(p["signed"]) if p else None,
                              status if not why or status == PLACED else f"{status} ({short(why, 300)})", row, rd.get("name"), short(rd.get("definition"), 400),
                              short(ans.get("quote"), 250) if p is parts[0] else None,
                              short(ans.get("reason"), 300) if p is parts[0] else None, chk_row, chk_ev, None, None,
                              used_formula(COL["Your row"], COL["Proposed row"], n),
                              used_formula(COL["Your signed amount"], COL["Signed amount in the row"], n), link])
                if status == PLACED and chk_row:
                    review.append([st, "checker disagrees", ln.get("label"), c, dec(ln.get("amount")), row, chk_row,
                                   chk_ev])
                if status == PLACED and rd.get("policy") == "catch_all":
                    review.append([st, "catch-all placement", ln.get("label"), c, dec(ln.get("amount")), row, None,
                                   short(rd.get("definition"), 200)])
            if status in (OUTSIDE, UNRESOLVED, "failed", NOT_ROUTED):
                review.append([st, status, ln.get("label"), c, dec(ln.get("amount")), None, chk_row, short(why)])
        for e in r["exceptions"] or ():
            if "claim" in str(e.get("class")) or "overlap" in str(e.get("why")):
                review.append([st, "open overlap claim", None, e.get("concept"), None, None, None,
                               short(f"{e.get('class')}: {e.get('why')}")])
            elif e.get("class") == "named total disagreement":     # J2: shown for review, never counted
                review.append([st, "named total disagreement", None, e.get("concept"), None, None, None,
                               whole(e.get("why"))])
            elif e.get("class") == "filed sign":     # S1: shown for review, never counted; the filed value is kept
                ln = exception_line(r, e)
                review.append([st, "filed sign", ln.get("label"), e.get("concept"), dec(ln.get("amount")), None, None,
                               whole(e.get("why"))])
        summary[st] = {"framed": st in framed, "frame": {"lines": {k: v[0] for k, v in frame.items()},
                                 "amount": {k: str(v[1]) for k, v in frame.items()}}}

    # Template sheet: one row per template row of the statements the job ran
    where = {}
    for st in order:
        for s in t.data["sections"]:
            if s["statement"] != st:
                continue
            for row in t.section_rows(s["id"]):
                where[row["row_id"]] = len(tpl) + 2
                tpl.append([st, s["id"], row["row_id"], row["name"], row["line_type"], row.get("policy"), None, None,
                            short(row.get("definition"), 400)])
    G = get_column_letter(TEMPLATE.index("Recalculated amount") + 1)
    n_lines = len(lines) + 1
    claimed = {(st, cl["row"]) for st in order for cl in recs[st]["claims"] or ()}
    used = f'Lines!${COL["Row used"]}$2:${COL["Row used"]}${n_lines}'
    for i, r in enumerate(tpl):
        st, rid, kind = r[0], r[2], r[4]
        rows, tot = recs[st]["rows"] or {}, recs[st]["computed_totals"] or {}
        rec = t.recipes.get(rid)
        if kind in ("item", "supplementary") or not rec:
            # the run records an empty row as 0; with no line placed in it the row is unknown, not zero
            r[6] = dec(rows.get(rid)) if (st, rid) in claimed else UNKNOWN
            r[7] = Formula(f'=IF(COUNTIF({used},C{i + 2})=0,"{UNKNOWN}",'
                           f'SUMIFS(Lines!${COL["Amount used"]}$2:${COL["Amount used"]}${n_lines},{used},C{i + 2}))')
        else:
            v = dec(rows.get(rid)) if rid in rows else dec((tot.get(rid) or {}).get("value"))
            r[6] = UNKNOWN if v is not None and all_unknown(t, rid, st, claimed) else v
            r[7] = formula(rec, where, G)
    # filled rows: item rows of the run sections with at least one placed line
    placed_rows = {ln[LINES.index("Proposed row")] for ln in lines if ln[LINES.index("Status")] == PLACED}
    for st in order:
        items = [x for x in tpl if x[0] == st and x[4] in ("item", "supplementary")]
        summary[st]["rows_filled"] = [sum(x[2] in placed_rows for x in items), len(items)]

    # Checks sheet. Under statement checks (template format v7) each check says whether it decides
    # the outcome ("statement check") or is shown only; a job without them reads as before
    from spreading.spread import subtotal_check
    gate = t.outcome_checks

    def decides(c):
        return not gate or not subtotal_check(c) or c.get("section") in gate["sections"]

    def tag(yes):
        return "" if not gate else ", statement check" if yes else ", shown only (decides nothing)"

    for st in order:
        rec = recs[st]
        filed_amount = {ln["concept"]: dec(ln.get("amount")) for ln in rec["lines"] or () if ln.get("control")}
        n_checks = {"tied": 0, "failed": 0, "unavailable": 0}
        n_statement = {"tied": 0, "failed": 0, "unavailable": 0}     # the statement checks, under a gate
        shown_failed = []       # failed checks shown only (under a gate), each with its section
        no_rows = 0             # recorded failures over rows with no line placed (the Assessment column)
        line_failed = []        # the failed line checks, each with its section and reason
        for bucket, items in (rec["controls"] or {}).items():
            for x in items:
                state = control_state(x.get("state") or bucket)
                sec = x.get("section")
                ctl = t.controls.get(sec)
                assess = NO_ROWS if state == "failed" and rows_unknown(t, ctl, st, claimed) else None
                no_rows += bool(assess)
                n_checks[state] += 1
                yes = not gate or sec in gate["sections"]
                if gate and yes:
                    n_statement[state] += 1
                built = coef_formula(t.coefficients(ctl["compare"]), where, G) if ctl else None
                filed = dec(x.get("filed_value")) if x.get("filed_value") is not None else filed_amount.get(x.get("filed"))
                n = len(checks) + 2
                tol = dec(x.get("tolerance"))
                # a control with no filed total compares its rows with a total rebuilt from printed ones (rather
                # than "rows add to None")
                total = x.get("filed") or (f"the reconstructed total ({rebuilt(ctl)})" if ctl and ctl.get("reconstructed")
                                           else "a total not named")
                checks.append([st, f"control ({x.get('kind') or 'local'}){tag(yes)}", sec, f"rows add to {total}",
                               x.get("filed"), filed, dec(x.get("built")), dec(x.get("gap")), tol, state,
                               short(x.get("state") if state != "tied" else None) or why_of(x),
                               built, Formula(f"=L{n}-F{n}") if built and filed is not None else None,
                               state_formula(n)
                               if built and filed is not None and tol is not None else None, assess])
                if state == "failed" and not yes:
                    shown_failed.append(f"{section_said(t, sec)}: rows add to {total}")
                if state != "tied":
                    review.append([st, f"check {state}{tag(yes)}", f"rows add to {total}", x.get("filed"), filed,
                                   None, None, "; ".join(filter(None, (why_of(x), assess))) or None])
        for x in rec["report_checks"] or ():
            state = control_state(x.get("state"))
            n_checks[state] += 1
            yes = bool(gate) and x.get("id") in gate["report_checks"]
            if yes:
                n_statement[state] += 1
            label = (f"report check {x.get('id')}" if not gate else f"statement check {x.get('id')}" if yes
                     else f"report check {x.get('id')}, shown only (decides nothing)")
            left, right = (x.get("left") or {}), (x.get("right") or {})
            spec = next((c for c in t.report_checks if c["id"] == x.get("id")), {})
            cells = [where.get((spec.get(k) or {}).get("row")) for k in ("left", "right")]
            n = len(checks) + 2
            tol = dec(x.get("tolerance"))
            checks.append([st, label, None, x.get("name"), None,
                           dec(right.get("value")), dec(left.get("value")), dec(x.get("gap")), tol, state,
                           short("; ".join(filter(None, (
                               *(f"{k} read as {c}, the named total not being in the filing"
                                 for k, c in sorted((x.get("read_as") or {}).items())),
                               x.get("state") if state != "tied" else x.get("note"))))),
                           Formula(f"=Template!{G}{cells[0]}") if all(cells) else None,
                           Formula(f'=IF(OR(ISTEXT(Template!{G}{cells[0]}),ISTEXT(Template!{G}{cells[1]})),'
                                   f'"{UNKNOWN}",Template!{G}{cells[0]}-Template!{G}{cells[1]})') if all(cells) else None,
                           state_formula(n) if all(cells) and tol is not None
                           else None, None])
            if state == "failed":
                review.append([st, f"check failed{tag(yes) if gate else ''}", x.get("name"), None, None, None, None,
                               short(x.get("state"))])
        for x in rec["checks"] or ():
            filed = dec(x.get("filed_value")) if x.get("filed_value") is not None else dec(x.get("filed"))
            why = None if x.get("ok") else line_check_why(x)
            n, tol = len(checks) + 2, dec(x.get("tolerance"))
            ctl = t.controls.get(x.get("section")) if str(x.get("check")).startswith("rows add to") else None
            built = coef_formula(t.coefficients(ctl["compare"]), where, G) if ctl and filed is not None else None
            assess = None
            yes = decides(x)
            if ctl:
                state = "tied" if x.get("ok") else "failed"
                assess = NO_ROWS if state == "failed" and rows_unknown(t, ctl, st, claimed) else None
                no_rows += bool(assess)
            else:
                state = "passed" if x.get("ok") else "failed"
                if state == "failed":       # the statement's failed checks count too, not only its controls
                    (line_failed if yes else shown_failed).append(
                        f"{section_said(t, x.get('section'))}: {x.get('check')}"
                        + (f" ({short(why, 160)})" if why else ""))
            if ctl:
                n_checks[state] += 1
                if gate and yes:
                    n_statement["tied" if state == "tied" else "failed"] += 1
                if state == "failed" and not yes:
                    shown_failed.append(f"{section_said(t, x.get('section'))}: {x.get('check')}")
            checks.append([st, ("section control" if ctl else "line check") + (tag(yes) if subtotal_check(x) else ""), x.get("section"), x.get("check"),
                           x.get("filed") if dec(x.get("filed")) is None else None, filed,
                           dec(x.get("parts") if x.get("parts") is not None else x.get("built")), dec(x.get("gap")),
                           tol, state,
                           why, built, Formula(f"=L{n}-F{n}") if built else None,
                           state_formula(n) if built and tol is not None else None, assess])
            if state in ("failed", "unavailable"):
                review.append([st, f"{'check' if ctl else 'line check'} {state}{tag(yes) if subtotal_check(x) else ''}", x.get("check"), None, None, None,
                               None, "; ".join(filter(None, (why, assess))) or None])
        summary[st]["checks"] = n_checks
        if gate:
            summary[st]["statement_checks"] = n_statement
            summary[st]["shown_only_failed"] = shown_failed
        summary[st]["failed_over_empty_rows"] = no_rows
        summary[st]["line_checks_failed"] = line_failed
        js = (job_rec.get("statements") or {}).get(st) or {}
        summary[st]["outcome"] = js.get("outcome")
        summary[st]["repair_skipped"] = [f"{section_said(t, k.get('section'))}: "
                                         f"{str(k.get('why') or '').removeprefix(str(k.get('section')) + ': ')}"
                                         for k in job_rec.get("skipped") or () if k.get("statement") == st]

    # engine-input routing: for each engine-input row, the lines placed in it, or that it is unknown
    for st in order:
        for rid in engine_rows(t):
            rd = t.rows.get(rid)
            if not rd or t.statement_of(rd["section"]) != st:
                continue
            got = [ln for ln in lines if ln[0] == st and ln[LINES.index("Proposed row")] == rid
                   and ln[LINES.index("Status")] == PLACED]
            for ln in got:
                review.append([st, "engine input: routed here", ln[2], ln[3], ln[LINES.index("Amount as filed")],
                               rid, ln[LINES.index("Checker's row (where it differed)")],
                               f"{rd['name']}: signed amount {ln[LINES.index('Signed amount in the row')]}"])
            if not got:
                review.append([st, f"engine input: {UNKNOWN}", None, None, None, rid, None,
                               f"{rd['name']}: no line placed; not shown separately on the allowed statements, so "
                               "unknown, not zero"])

    summary["all"] = frame_summary(summary, framed, t)
    summary["all"]["memo"] = [memo_of(t, rid, recs) for rid, r in t.rows.items()
                              if r["line_type"] == "supplementary" and t.statement_of(r["section"]) in order
                              and t.statement_of(r["section"]) not in framed]
    about = [["Status", f"{TITLE}: made by the workflow's models and code; no person has reviewed or approved it"],
             ["Company", m.get("company")], ["SEC number (CIK)", m.get("cik")], ["Form", m.get("form")],
             ["Filing (accession)", m.get("accession")], ["Period", m.get("period") if isinstance(m.get("period"), str)
                                                         else json.dumps(m.get("period")) if m.get("period") else None],
             ["Job", job.name], ["Template", m["template"].get("name")], ["Template file", m["template"]["path"]],
             ["Template fingerprint (SHA-256)", m["template"]["sha256"]], ["Source", link],
             ["Headline", headline(summary)],
             *[[f"{t.statements[st]['name'][0].upper()}{t.statements[st]['name'][1:]}", said(summary[st])]
               for st in order],
             ["How to correct", "On the Lines sheet, type a row ID in 'Your row' and, when the sign or amount changes, "
                                "a signed amount in 'Your signed amount'. The proposed row stays beside it; the "
                                "Template and Checks sheets recalculate from 'Row used' and 'Amount used'."],
             ["Unknown", "A row with no line placed in it shows 'unknown', not zero: the allowed statements do "
                         "not show it separately. A subtotal adds the lines placed in its rows and "
                         "is unknown only when no line is placed in any of them."],
             ["Limitation", "Corrections made in Excel do not return to the system."],
             ["Workflow checks and recalculated checks", "The workflow state is what the run recorded; the "
                                                          "recalculated columns follow your corrections."]]
    sheets = [("About", ["Item", "Value"], about), ("Lines", LINES, lines), ("Template", TEMPLATE, tpl),
              ("Checks", CHECKS, checks), ("Review", REVIEW, review),
              ("Not spread", NOT_SPREAD, [[x["input"], x["why"]] for x in t.not_spread()])]
    return sheets, summary


def rebuilt(ctl):
    """A reconstructed total in words: its printed concepts, the first one plus or less the others."""
    terms = ctl["reconstructed"]["terms"]
    out = terms[0]["concept"].split(":")[-1]
    for x in terms[1:]:
        out += f" {'less' if x['coef'] < 0 else 'plus'} {x['concept'].split(':')[-1]}"
    return out


def section_said(t, sid):
    """A section in words: the template's name, or the run itself for a run-level check."""
    if sid in (None, "run"):
        return "the statement as a whole"
    return (t.sections.get(sid) or {}).get("name") or sid


def line_check_why(x):
    """Why a code check failed: its recorded reason, else the lines it names."""
    parts = [str(x["why"])] if x.get("why") else []
    for k in ("lines", "lines_outside_the_total", "queued_unresolved", "claimed_twice", "unresolved_signs",
              "incomplete_subtotals", "extraction_refused", "printed_by_member_only"):
        if x.get(k):
            parts.append(f"{k.replace('_', ' ')}: {', '.join(map(str, x[k]))}")
    return short("; ".join(parts)) or None


def why_of(x):
    inc = x.get("incomplete")
    if inc:
        return short("; ".join(f"{k}: {', '.join(map(str, v))}" for k, v in inc.items() if v))
    return short(x.get("why")) or None


def control_state(s):
    s = str(s or "")
    if s == "tied":
        return "tied"
    if s in ("differs", "failed") or s.startswith("failed"):
        return "failed"
    return "unavailable"


def rows_unknown(t, ctl, st, claimed):
    """True when a control compares rows and no line is placed in any of them (all_unknown of each): its gap is the
    whole printed total, which says nothing about the placements. The recorded state is kept; the Checks sheet's
    Assessment column says so (NO_ROWS)."""
    rows = list(t.coefficients(ctl["compare"])) if ctl else []
    return bool(rows) and all(all_unknown(t, r, st, claimed) for r in rows)


def all_unknown(t, rid, st, claimed, seen=()):
    """True when no line is placed in the row or, for a subtotal, in any row its recipe adds (a concept term is a
    printed amount, so known)."""
    rec = t.recipes.get(rid)
    if not rec or rid in seen:
        return (st, rid) not in claimed
    return all("row" in x and all_unknown(t, x["row"], st, claimed, seen + (rid,)) for x in rec["terms"])


def used_formula(yours, proposed, n):
    """What a Lines row uses: the analyst's entry, else the proposal, else blank (never 0 for an unplaced line)."""
    return Formula(f'=IF({yours}{n}="",IF({proposed}{n}="","",{proposed}{n}),{yours}{n})')


def state_formula(n):
    """A recalculated check's state: tied or failed by its gap, unavailable when the gap is unknown."""
    return Formula(f'=IF(ISNUMBER(M{n}),IF(ABS(M{n})<=I{n},"tied","failed"),"unavailable")')


def coef_formula(k, where, col):
    """=coef*N(cell)+... over the Template sheet's recalculated cells (an unknown row adds nothing: a control
    compares the lines placed with the printed total); None when a row is not on the sheet."""
    if not k or any(r not in where for r in k):
        return None
    return Formula("=" + "+".join(f"({c})*N(Template!{col}{where[r]})" for r, c in sorted(k.items())))


def formula(rec, where, col):
    """A recipe over the Template sheet's cells (rows only; a concept term leaves the cell to the workflow's amount,
    said by an empty formula). A subtotal adds the lines placed in its rows and is unknown only when every row is."""
    terms = rec["terms"]
    if any("row" not in x or x["row"] not in where for x in terms):
        return None
    cells = [f"{col}{where[x['row']]}" for x in terms]
    # the coefficient carries the sign (a difference is +1, -1, ...: template.check), as in Template.evaluate
    out = [f"({Decimal(str(x['coef']))})*N({c})" for x, c in zip(terms, cells)]
    test = "AND(" + ",".join(f"ISTEXT({c})" for c in cells) + ")"
    return Formula(f'=IF({test},"{UNKNOWN}",{"+".join(out)})')


def frame_summary(summary, framed, t):
    """The coverage frame over the eligible lines of the spread statements (in_frame), a line routed to no section
    counted in the denominator; amounts are absolute line amounts."""
    st = [x for x in framed if x in summary]
    n = {k: sum(summary[x]["frame"]["lines"][k] for x in st) for k in FRAMED}
    a = {k: sum(Decimal(summary[x]["frame"]["amount"][k]) for x in st) for k in FRAMED}
    eligible, weight = sum(n.values()), sum(a.values())
    return {"frame": [t.statements[x]["name"] for x in st], "lines": n, "eligible": eligible,
            "not_routed": n[NOT_ROUTED], "amount": {k: str(v) for k, v in a.items()},
            "placed_share": float(n[PLACED] / eligible) if eligible else None,
            "placed_amount_share": float(a[PLACED] / weight) if weight else None}


def memo_of(t, rid, recs):
    """A memo row read outside the spread statements (its own statement is not spread): its signed amount and facts,
    or unknown when no line was placed in it."""
    st = t.statement_of(t.rows[rid]["section"])
    got = [cl for cl in (recs.get(st) or {}).get("claims") or () if cl["row"] == rid]
    if not got:
        return {"row": rid, "name": t.rows[rid]["name"], "statement": st, "amount": UNKNOWN, "facts": []}
    return {"row": rid, "name": t.rows[rid]["name"], "statement": st,
            "amount": str(sum(dec(cl["signed"]) or 0 for cl in got)), "facts": [cl.get("fact") for cl in got]}


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def said_list(xs):
    return " and ".join(xs) if len(xs) < 3 else ", ".join(xs[:-1]) + " and " + xs[-1]


def headline(summary):
    a = summary["all"]
    # a memo amount with its thousands separated and its fact named ("76230000 from Tc_..." read
    # as a code)
    memo = "".join(f" {x['name'][0].upper()}{x['name'][1:]}: "
                   + (x["amount"] if x["amount"] == UNKNOWN else f"{Decimal(str(x['amount'])):,}")
                   + (" (" if "memo" in x["name"].lower() else
                      f" (a memo read from the {NAMES.get(x['statement'], x['statement'])}, ")
                   + f"outside the percentage{'' if x['amount'] == UNKNOWN or not x['facts'] else '; ' + ('fact ' if len(x['facts']) == 1 else 'facts ') + ', '.join(map(str, x['facts']))})."
                   for x in a.get("memo") or ())
    return (f"We were able to classify {pct(a['placed_share'])} of the statement lines into your template "
            f"({a['lines'][PLACED]} of {a['eligible']} eligible lines of the {said_list(a['frame'])}; "
            f"{pct(a['placed_amount_share'])} by amount). {a['lines'][UNRESOLVED]} unresolved, "
            f"{a['lines']['failed']} failed, {a['lines'][OUTSIDE]} outside the template"
            + (f", {a['not_routed']} routed to no section (listed on the Review sheet)" if a["not_routed"] else "")
            + "." + memo)


def said(s):
    """One statement's counts, over the coverage frame the headline adds up (in_frame), so the two reconcile."""
    k, r = s["checks"], s["rows_filled"]
    empty = s.get("failed_over_empty_rows")
    checks = (f"controls: {k['tied']} tied, {k['failed']} failed"
              + (f" ({empty} over rows with no line placed)" if empty else "")
              + f", {k['unavailable']} unavailable; template rows filled {r[0]} of {r[1]}")
    if "frame" not in s:            # a summary saved before the frame (earlier exports): its all-lines counts
        n = s["lines"]
        return (f"lines: {n[PLACED]} placed, {n[UNRESOLVED]} unresolved, {n['failed']} failed, {n[OUTSIDE]} outside; "
                + checks)
    # the statement's outcome, with what blocked it and why no repair ran, after the counts
    if s.get("statement_checks"):       # template format v7: what decides the outcome, then what is shown only
        k2 = s["statement_checks"]
        checks += (f"; statement checks (they decide the outcome): {k2['tied']} tied, {k2['failed']} failed, "
                   f"{k2['unavailable']} unavailable")
        if s.get("shown_only_failed"):
            checks += f"; failed but shown only: {'; '.join(s['shown_only_failed'])}"
    if s.get("outcome"):
        checks += f". Outcome: {s['outcome']}"
        if s.get("line_checks_failed"):
            checks += f"; failed checks: {'; '.join(s['line_checks_failed'])}"
        if s.get("repair_skipped"):
            checks += f"; repair not attempted: {'; '.join(s['repair_skipped'])}"
    if not s.get("framed", True):
        return "outside the percentage (read for its memo rows only); " + checks
    n = s["frame"]["lines"]
    return (f"eligible lines: {n[PLACED]} placed, {n[UNRESOLVED]} unresolved, {n['failed']} failed, "
            f"{n[OUTSIDE]} outside" + (f", {n[NOT_ROUTED]} routed to no section" if n[NOT_ROUTED] else "")
            + "; " + checks)


def cell(x):
    if isinstance(x, Decimal):
        return int(x) if x == x.to_integral_value() and abs(x) < 10 ** 15 else float(x)
    return x


def write(sheets, out):
    wb = Workbook()
    wb.remove(wb.active)
    for name, headers, rows in sheets:
        ws = wb.create_sheet(name)
        for i, r in enumerate([headers] + rows, 1):
            for j, x in enumerate(r, 1):
                x = cell(x)
                if x is None:
                    continue
                c = ws.cell(i, j, x)
                if isinstance(x, str) and not isinstance(x, Formula):
                    c.data_type = "s"
        ws.freeze_panes = "A2"
    wb.save(out)


def read_back(sheets, path):
    """The workbook read back against the snapshot it was written from: a list of differences, empty when equal."""
    wb = load_workbook(path)
    diffs = [] if wb.sheetnames == [s[0] for s in sheets] else [f"sheets {wb.sheetnames}"]
    for name, headers, rows in sheets:
        if name not in wb.sheetnames:
            continue
        got = [[c.value for c in r] for r in wb[name].iter_rows()]
        want = [headers] + rows
        if len(got) != len(want):
            diffs.append(f"{name}: {len(got)} rows read back, {len(want)} written")
            continue
        for i, (g, w) in enumerate(zip(got, want)):
            g = list(g) + [None] * (len(w) - len(g))
            for h, a, b in zip(headers, g, w):
                b = cell(b)
                ok = (a == b) if not isinstance(b, float) else isinstance(a, (int, float)) and float(a) == b
                if not ok:
                    diffs.append(f"{name} row {i + 1} {h}: read {a!r}, wrote {b!r}")
    wb.close()
    return diffs


def coverage(sheets, job):
    """Every printed line of every statement result appears on the Lines sheet with a status (none dropped)."""
    shown = {(r[0], r[3]) for r in sheets[1][2] if r[LINES.index("Status")]}
    want = set()
    for st, d in statement_dirs(job).items():
        r = {n: read(d / f"{n}.json") if (d / f"{n}.json").exists() else None for n in ("lines", "diagnostics")}
        want |= {(st, ln["concept"]) for ln in printed_lines(r)}
    return [f"{st} {c} not on the Lines sheet" for st, c in sorted(want - shown)]


def export(job, label=None):
    """A dated record (runs/<stamp>_result_workbook) holding the workbook, its snapshot, the summary and the
    read-back with the workbook's frozen identity; a failure is recorded in the same record before it is raised."""
    import hashlib
    job = Path(job)
    rec = runrec.new_run("result_workbook", {"job": job.name, "label": label})
    try:
        sheets, summary = snapshot(job)
        m = read(job / "ui_manifest.json")
        # the job's start in the name, so two jobs on one filing download apart
        out = rec / f"workflow_result_{m.get('cik')}_{m.get('accession')}_{job.name[:17]}.xlsx"
        write(sheets, out)
        diffs = read_back(sheets, out) + coverage(sheets, job)
        runrec.write(rec, "snapshot.json", json.dumps([{"sheet": n, "headers": h, "rows": r} for n, h, r in sheets],
                                                      indent=1, default=str))
        runrec.write(rec, "summary.json", summary)
        runrec.write(rec, "read_back.json", {"matches": not diffs, "differences": diffs, "workbook": out.name,
                                             "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                                             "status": TITLE})
    except Exception as e:
        runrec.write(rec, "exceptions.json", {"failed": repr(e)})
        raise
    return out, diffs, summary


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m spreading.result_workbook")
    ap.add_argument("job", help="the job's folder under runs/")
    a = ap.parse_args(argv)
    out, diffs, summary = export(a.job, "command line")
    print(json.dumps({"workbook": str(out), "matches": not diffs, "differences": diffs[:20],
                      "headline": headline(summary)}, indent=1))
    return 0 if not diffs else 1


if __name__ == "__main__":
    sys.exit(main())
