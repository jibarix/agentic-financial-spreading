"""The one reader of a template file (format: FORMATS below). It loads the file, checks it, and answers what
the rest of the code needs: sections and their rows, headings, networks, controls and the arithmetic of totals. No
other module reads a template file or infers anything from the shape of an ID.

    .venv-agents/Scripts/python.exe -s -E -m spreading.template check rulebook/draft/A1_template_v8.json --edition 2025
"""
import argparse
import json
import re
from decimal import Decimal
from pathlib import Path

from . import runrec

STANDARD = runrec.ROOT / "rulebook/draft/A1_template_v8.json"     # the standard template
# format -> the fields it adds to FIELDS: v2 gives each statement its period and adds the report checks (C2, C6b)
# and v3 a statement's title_fallback
FORMATS = {"template_format_v1": {}, "template_format_v2": {"top": {"report_checks"}, "statement": {"period"}},
           "template_format_v3": {"top": {"report_checks"}, "statement": {"period", "title_fallback"}},
           # v4 adds a control's further reconstructions (`alternatives`) and the rows it leaves out of its comparison
           # when another control verifies them (`outside_rows`)
           "template_format_v4": {"top": {"report_checks"}, "statement": {"period", "title_fallback"},
                                  "control": {"alternatives", "outside_rows"}},
           # v5 adds the inputs the template does not spread (`not_spread`: {input, why}), shown as "not spread yet"
           "template_format_v5": {"top": {"report_checks", "not_spread"},
                                  "statement": {"period", "title_fallback"},
                                  "control": {"alternatives", "outside_rows"}},
           # v6 adds a report check's other printed totals (`alternatives`: {concept: [concept, ...]}), read in order
           # when the concept a side names is not in the filing (the company prints the same total under another
           # tag: cash as Cash, equity as StockholdersEquity)
           "template_format_v6": {"top": {"report_checks", "not_spread"},
                                  "statement": {"period", "title_fallback"},
                                  "control": {"alternatives", "outside_rows"},
                                  "report_check": {"alternatives"}},
           # v7 adds the statement checks (`outcome_checks`: {sections, report_checks}): the only section controls
           # and report checks that decide a statement's outcome; every other section's subtotal check is reported
           # beside the spread
           "template_format_v7": {"top": {"report_checks", "not_spread", "outcome_checks"},
                                  "statement": {"period", "title_fallback"},
                                  "control": {"alternatives", "outside_rows"},
                                  "report_check": {"alternatives"}}}
ALTERNATIVE = {"id", "plus", "minus", "capital"}
CAPITAL = {"inclusive", "parent", "nci_row", "temporary_rows"}
OUTSIDE = {"rows", "verified_by"}
FALLBACK = {"words", "excludes", "requires_any"}
FIELDS = {
    "top": {"format", "name", "version", "status", "what", "built_from", "statements", "sections", "rows",
            "relationships", "controls"},
    "statement": {"name", "title_words", "title_excludes", "variants"},
    "variant": {"id", "fasb_network", "routes_to"},
    "section": {"id", "statement", "name", "kind", "variants", "fasb_headings", "output_total_row", "normal_balance"},
    "row": {"row_id", "section", "name", "line_type", "policy", "extra_evidence", "definition", "belongs",
            "not_belongs"},
    "relationship": {"row_id", "op", "terms"},
    "control": {"section", "compare", "filed", "reconstructed", "tolerance"},
    "report_check": {"id", "name", "kind", "left", "right", "period", "tolerance", "deferred", "note"},
}
STATEMENTS = ("BS", "IS", "CF")
KINDS = ("spread", "totals", "supplementary")
LINE_TYPES = ("item", "subtotal", "control", "supplementary")
POLICIES = (None, "catch_all")
EXTRA = (None, "asc")
BALANCES = (None, "debit", "credit")
OPS = ("sum", "difference")
TOLERANCES = ("half_unit_sum",)
PERIODS = ("instant", "duration")
CHECK_KINDS = ("spread", "filed")


class TemplateError(ValueError):
    def __init__(self, reasons):
        super().__init__("; ".join(reasons))
        self.reasons = reasons


class Template:
    def __init__(self, data, path=None):
        self.data, self.path = data, path
        self.statements = data["statements"]
        self.sections = {s["id"]: s for s in data["sections"]}
        self.rows = {r["row_id"]: r for r in data["rows"]}
        self.recipes = {r["row_id"]: r for r in data["relationships"] if r["op"]}
        self.controls = {c["section"]: c for c in data["controls"]}
        self.report_checks = data.get("report_checks") or []
        self.outcome_checks = data.get("outcome_checks")    # format v7; None: every check decides the outcome

    # ---- what the pipeline asks -------------------------------------------------------------------------------------
    def section_rows(self, section):
        return [r for r in self.data["rows"] if r["section"] == section]

    def statement_of(self, section):
        return self.sections[section]["statement"]

    def section_of(self, row_id):
        """The section of a template row; None for an ID the template does not have."""
        r = self.rows.get(row_id)
        return r["section"] if r else None

    def heading_to_section(self):
        return {h: s["id"] for s in self.data["sections"] for h in s["fasb_headings"]}

    def memo(self, section):
        """A memo section: kind supplementary with headings of its own. Its lines are routed and placed like a spread
        section's; its rows sit outside every total, control and report check."""
        s = self.sections[section]
        return s["kind"] == "supplementary" and bool(s["fasb_headings"])

    def run_sections(self, statement, variant=None):
        """The sections a run of the statement takes, in template order: every spread section and every memo
        section of the variant."""
        return [sid for sid, s in self.sections.items() if s["statement"] == statement
                and (s["kind"] == "spread" or self.memo(sid)) and (not variant or variant in s["variants"])]

    def network_of(self, section):
        """The FASB network of the section's first variant."""
        s = self.sections[section]
        return next(v["fasb_network"] for v in self.statements[s["statement"]]["variants"] if v["id"] in s["variants"])

    def period(self, statement):
        """"instant" or "duration": the period a run of the statement reads its amounts for (C1). Format v1 has no
        such field; its spread read the balance sheet only, at the period end."""
        return self.statements[statement].get("period") or ("instant" if statement == "BS" else "duration")

    def indirect_start(self, statement):
        """(operating totals, starting items) for a statement with an indirect variant, else None (C7): this MVP spreads
        indirect filers only, recognised by a net-income starting item under the operating total, the filed total the
        other variant's section also ties to. The two items are fixed here, not template data. Their absence does not
        show the direct method."""
        vs = self.statements[statement]["variants"]
        totals = {self.controls[v["routes_to"]]["filed"] for v in vs if v["routes_to"] in self.controls}
        if not any(v["id"] == "indirect" for v in vs) or not totals or None in totals:
            return None
        return totals, ("us-gaap:NetIncomeLoss", "us-gaap:ProfitLoss")

    def statement_controls(self, statement):
        """The concepts that are the statement's own controls: each control's filed concept and the concept terms of
        its reconstructed expression (C4a's state 4). Never a placement line."""
        out = set()
        for sid, c in self.controls.items():
            if self.statement_of(sid) == statement:
                out |= {c["filed"]} if c["filed"] else set()
                out |= {t["concept"] for t in (c["reconstructed"] or {}).get("terms", []) if "concept" in t}
        return out

    def filed_total(self, section):
        c = self.controls.get(section)
        return c["filed"] if c else None

    def alternatives(self, section):
        """The control's further reconstructions (format v4), in the template's order; none before v4."""
        return list((self.controls.get(section) or {}).get("alternatives") or ())

    def outside_rows(self, section):
        """{rows, verified_by}: the rows of the section its comparison leaves out when the control `verified_by`
        ties and counts each of their facts once (format v4), or None."""
        return (self.controls.get(section) or {}).get("outside_rows")

    def item_rows(self, expr):
        """The item rows an expression adds up, through the recipes; None when it reaches a concept, a row with no
        recipe that is not an item, or nothing."""
        if "row" in expr:
            r = self.rows.get(expr["row"])
            if r and r["line_type"] == "item":
                return {expr["row"]}
            return self.item_rows(self.recipes[expr["row"]]) if expr["row"] in self.recipes else None
        if "concept" in expr:
            return None
        out = set()
        for t in expr["terms"]:
            got = self.item_rows({k: t[k] for k in ("row", "concept") if k in t})
            if got is None:
                return None
            out |= got
        return out or None

    def coefficients(self, expr, k=Decimal(1), out=None):
        """row -> the coefficient of each row an expression reads, through the recipes as `evaluate` reads them (C6's
        tolerance); a concept term adds no row."""
        out = {} if out is None else out
        if "row" in expr:
            r = self.rows.get(expr["row"])
            if expr["row"] in self.recipes and not (r and r["line_type"] == "item"):
                return self.coefficients(self.recipes[expr["row"]], k, out)
            out[expr["row"]] = out.get(expr["row"], Decimal(0)) + k
        for t in expr.get("terms") or ():
            self.coefficients({x: t[x] for x in ("row", "concept") if x in t}, k * Decimal(str(t["coef"])), out)
        return out

    def not_spread(self):
        """The inputs the template names but does not spread ({input, why}, format v5); none before v5."""
        return list(self.data.get("not_spread") or ())

    def unread_items(self):
        """The item rows of a section with a control that no control and no formula of the template reads: printed
        lines outside every total (a cash flow's beginning and ending cash). Read from the template's arithmetic, so
        no row is named in code; the inventory and the sorting leave them out."""
        read = set()
        for c in self.data["controls"]:
            read |= set(self.coefficients(c["compare"]))
        for r in self.data["relationships"]:
            read |= set(self.coefficients({"row": r["row_id"]}))
        return {r["row_id"] for r in self.data["rows"]
                if r["line_type"] == "item" and r["section"] in self.controls and r["row_id"] not in read}

    def local(self, section):
        """Whether the section's control adds up only the section's own item rows."""
        c = self.controls.get(section)
        rows = self.item_rows(c["compare"]) if c else None
        return bool(rows) and all(self.rows[r]["section"] == section for r in rows)

    def anchors(self, section):
        """Whether the section's control may route lines: it is local, and the section is not a variant's routes_to
        target (those lines arrive by the variant). A control over other sections' rows only compares: as an anchor
        it would make their lines ambiguous."""
        return self.local(section) and not any(v["routes_to"] == section for st in self.statements.values()
                                               for v in st["variants"])

    def section_totals(self):
        """section -> the filed concept that checks it, for every control with one that may route lines."""
        return {s: c["filed"] for s, c in self.controls.items() if c["filed"] and self.anchors(s)}

    def derived_totals(self):
        """section -> {plus, minus}: a reconstructed control that is one filed total less another, the form the
        spread's reconstruction reads (coefficients +1 and -1), for a control that may route lines. A control of
        another form is not offered to it."""
        out = {}
        for s, c in self.controls.items():
            if not self.anchors(s):
                continue
            r = c["reconstructed"]
            if (r and r["op"] == "difference" and len(r["terms"]) == 2 and all("concept" in t for t in r["terms"])
                    and [t["coef"] for t in r["terms"]] == [1, -1]):
                out[s] = {"plus": r["terms"][0]["concept"], "minus": r["terms"][1]["concept"]}
        return out

    def title_matches(self, statement, title):
        return titled(self.statements[statement]["title_words"], self.statements[statement]["title_excludes"], title)

    # ---- arithmetic: a missing input stays unavailable, never zero ---------------------------------------------------
    def evaluate(self, expr, rows, concepts=None, _seen=()):
        """Value of {row} or {op, terms} from rows (row_id -> Decimal or None) and concepts (concept -> Decimal or
        None). Returns (value, missing): value is None when any input is unavailable, and missing names them."""
        if "row" in expr:
            rid = expr["row"]
            if rid in self.recipes and rid not in rows:
                if rid in _seen:
                    raise TemplateError([f"{rid}: cycle"])
                return self.evaluate(self.recipes[rid], rows, concepts, _seen + (rid,))
            v = rows.get(rid)
            return (v, []) if v is not None else (None, [rid])
        if "concept" in expr:
            v = (concepts or {}).get(expr["concept"])
            return (v, []) if v is not None else (None, [expr["concept"]])
        total, missing = Decimal(0), []
        for t in expr["terms"]:
            v, m = self.evaluate({k: t[k] for k in ("row", "concept") if k in t}, rows, concepts, _seen)
            missing += m
            if v is not None:
                total += Decimal(str(t["coef"])) * v
        return (None, missing) if missing else (total, [])

    # ---- readiness, per operation ------------------------------------------------------------------------------------
    def readiness(self, tax):
        targets = {v["routes_to"] for st in self.statements.values() for v in st["variants"] if v["routes_to"]}
        out = {}
        for sid, s in self.sections.items():
            if s["kind"] != "spread":
                out[sid] = {"kind": s["kind"]}
                continue
            items = [r for r in self.section_rows(sid) if r["line_type"] == "item"]
            try:
                tax.network(self.network_of(sid), "pre")
                network = True
            except FileNotFoundError:
                network = False
            c = self.controls.get(sid)
            # configuration, implementation and variant applicability kept apart: an income or cash flow line under no
            # heading or anchor reaches its section through the statement queue to the models (C7), which is not built,
            # so that route is "pending", not ready
            route = (True if s["fasb_headings"] or sid in targets or sid in self.section_totals() else
                     "pending: queue to models not built" if self.period(s["statement"]) == "duration" else False)
            tie = bool(c and (c["filed"] or c["reconstructed"]))
            # C6 runs a filed control locally or run-wide; a control with no filed total only in the form the
            # spread reconstructs (derived_totals)
            runs = bool(c and (c["filed"] or sid in self.derived_totals()))
            ops = {"route lines": route, "tie the section": tie, "tie runs in today's spread": runs}
            why = [k for k, v in (("no item rows", bool(items)), ("no FASB network", network),
                                  ("no heading or calculation anchor", route), ("no control", tie),
                                  ("today's spread cannot run its control", runs)) if not v]
            pending = [f"{k}: {v.split(': ', 1)[1]}" for k, v in ops.items() if isinstance(v, str)]
            # a row without a definition is shown to the model as such; it does not stop the spread. Which variant
            # applies to a filing is decided per run (spread.prepare), never here
            defined = sum(bool(r["definition"]) for r in items)
            out[sid] = {"kind": "spread", **ops, "configured": not why, "implemented": not pending,
                        "spread": not why and not pending, "variants": list(s["variants"]),
                        "definitions": f"{defined} of {len(items)} rows", "missing": why, "pending": pending}
        return out


# ---- checks on load -----------------------------------------------------------------------------------------------

def text(x):
    return isinstance(x, str) and bool(x.strip())


def titled(words, excludes, title):
    t = title.upper()
    return any(w in t for w in words) and not any(w in t for w in excludes)


def find_statement(st, reports, prints):
    """The one choice of a statement's report, shared by the pipeline (Filing.role) and the answer-key checker
    (keycheck.statement_pages). st: the template's statement entry; reports: [(key, title)] of FilingSummary's
    Statements reports, in its order; prints(key): the concepts that report prints. (hits, how, why): every report the
    title words match and the exclusions do not, how "title"; only when there is none, the report the fallback's words
    match and its exclusions do not and that prints one of its requires_any concepts, how "fallback", refused (no hits)
    when two or more qualify. why says what was found."""
    hits = [k for k, t in reports if titled(st["title_words"], st["title_excludes"], t)]
    fb = st.get("title_fallback")
    if hits or not fb:
        return hits, "title", f"{len(hits)} report(s) by the title words"
    named = [k for k, t in reports if titled(fb["words"], fb["excludes"], t)]
    told = [k for k in named if set(prints(k)) & set(fb["requires_any"])]
    if len(told) == 1:
        return told, "fallback", (f"no report by the title words; {told[0]} is the one report by the fallback words "
                                  f"that prints one of {fb['requires_any']}")
    return [], "fallback", (f"no report by the title words; {len(named)} by the fallback words {named}, {len(told)} "
                            f"of them printing one of {fb['requires_any']}: refused unless exactly one")


def check(data, tax=None):
    """Every reason the template cannot be used; [] when none. tax: the FASB edition the run uses, or None."""
    bad = []

    if not isinstance(data, dict) or data.get("format") not in FORMATS:
        return [f"template: format {data.get('format') if isinstance(data, dict) else None!r}, expected one of "
                f"{sorted(FORMATS)}"]
    added = FORMATS[data["format"]]

    def fields(kind, obj, where):
        want = FIELDS[kind] | added.get(kind, set())
        if not isinstance(obj, dict) or set(obj) != want:
            got = set(obj) if isinstance(obj, dict) else set()
            bad.append(f"{where}: fields missing {sorted(want - got)}, unknown {sorted(got - want)}")
            return False
        return True

    if not fields("top", data, "template"):
        return bad
    variants = {}
    for k, st in data["statements"].items():
        if k not in STATEMENTS:
            bad.append(f"statement {k}: unknown")
        if fields("statement", st, f"statement {k}"):
            if "period" in added.get("statement", ()) and st["period"] not in PERIODS:
                bad.append(f"statement {k}: period {st['period']!r}")
            fb = st.get("title_fallback")
            if fb is not None and (not isinstance(fb, dict) or set(fb) != FALLBACK
                                   or not all(isinstance(fb[x], list) and all(isinstance(w, str) and w for w in fb[x])
                                              for x in FALLBACK) or not fb["words"] or not fb["requires_any"]):
                bad.append(f"statement {k}: title_fallback is null or {sorted(FALLBACK)}, lists of words, words and "
                           f"requires_any not empty")
            for v in st["variants"]:
                if fields("variant", v, f"statement {k} variant"):
                    variants.setdefault(k, set()).add(v["id"])
    sections, rows = {}, {}
    for s in data["sections"]:
        if not fields("section", s, f"section {s.get('id')}"):
            continue
        sid = s["id"]
        if sid in sections:
            bad.append(f"section {sid}: duplicate")
        sections[sid] = s
        if s["statement"] not in variants:
            bad.append(f"section {sid}: statement {s['statement']} not in the template")
        elif not s["variants"] or not set(s["variants"]) <= variants[s["statement"]]:
            bad.append(f"section {sid}: variants {s['variants']} not all of {s['statement']}'s")
        if s["kind"] not in KINDS:
            bad.append(f"section {sid}: kind {s['kind']!r}")
        if s["normal_balance"] not in BALANCES:
            bad.append(f"section {sid}: normal balance {s['normal_balance']!r}")
    for r in data["rows"]:
        if not fields("row", r, f"row {r.get('row_id')}"):
            continue
        rid = r["row_id"]
        if rid in rows:
            bad.append(f"row {rid}: duplicate")
        rows[rid] = r
        if r["section"] not in sections:
            bad.append(f"row {rid}: section {r['section']} not in the template")
        for f, allowed in (("line_type", LINE_TYPES), ("policy", POLICIES), ("extra_evidence", EXTRA)):
            if r[f] not in allowed:
                bad.append(f"row {rid}: {f} {r[f]!r}")
    for rid, r in rows.items():      # the row's meaning (format v3): each field null until it is written
        if r["definition"] is not None and not text(r["definition"]):
            bad.append(f"row {rid}: definition must be text or null")
        if r["belongs"] is not None and not (isinstance(r["belongs"], list) and all(map(text, r["belongs"]))):
            bad.append(f"row {rid}: belongs must be a list of text items or null")
        if r["not_belongs"] is not None and not isinstance(r["not_belongs"], list):
            bad.append(f"row {rid}: not_belongs must be a list or null")
            continue
        for i, x in enumerate(r["not_belongs"] or ()):
            if not (isinstance(x, dict) and set(x) == {"item", "goes_to"} and text(x["item"])):
                bad.append(f"row {rid}: not_belongs {i} must be {{item: text, goes_to: a row_id or null}}")
            elif x["goes_to"] is None:
                continue
            elif not isinstance(x["goes_to"], str) or x["goes_to"] not in rows:
                bad.append(f"row {rid}: not_belongs {i} goes to {x['goes_to']!r}, not a row of the template")
            elif x["goes_to"] == rid or rows[x["goes_to"]]["line_type"] in ("subtotal", "control"):
                bad.append(f"row {rid}: not_belongs {i} goes to {x['goes_to']}, which cannot take it (the row "
                           "itself, or a computed row)")
    for k, st in data["statements"].items():
        for v in st["variants"]:
            if v.get("routes_to") is not None and v["routes_to"] not in sections:
                bad.append(f"statement {k} variant {v['id']}: routes to unknown section {v['routes_to']}")
    owner = {}
    for sid, s in sections.items():
        for h in s["fasb_headings"]:
            if owner.setdefault(h, sid) != sid:
                bad.append(f"heading {h}: in two sections, {owner[h]} and {sid}")
        o = s["output_total_row"]
        if o is not None and (o not in rows or rows[o]["line_type"] != "subtotal"
                              or sections.get(rows[o]["section"], {}).get("statement") != s["statement"]):
            bad.append(f"section {sid}: output total row {o} is not a subtotal of its statement")
    concepts = set()

    def expr(e, where, recipe=False):
        if not isinstance(e, dict):
            bad.append(f"{where}: not a row or an expression")
            return []
        if "row" in e:
            if e["row"] not in rows:
                bad.append(f"{where}: row {e['row']} not in the template")
            return [e["row"]]
        if e.get("op") not in OPS or not isinstance(e.get("terms"), list) or not e["terms"]:
            bad.append(f"{where}: op must be one of {OPS} with terms")
            return []
        refs = []
        for i, t in enumerate(e["terms"]):
            coef = t.get("coef")
            if isinstance(coef, bool) or not isinstance(coef, (int, float)):
                bad.append(f"{where}: term {i} coefficient {coef!r} is not a number")
                continue
            if e["op"] == "sum" and coef <= 0 or e["op"] == "difference" and (coef > 0) != (i == 0):
                bad.append(f"{where}: term {i} coefficient {coef} does not fit {e['op']}")
            if ("row" in t) == ("concept" in t):
                bad.append(f"{where}: term {i} names neither or both a row and a concept")
            elif "row" in t:
                if t["row"] not in rows:
                    bad.append(f"{where}: row {t['row']} not in the template")
                elif rows[t["row"]]["line_type"] == "supplementary":
                    bad.append(f"{where}: supplementary row {t['row']} is a term")
                refs.append(t["row"])
            else:
                concepts.add(t["concept"])
        return refs

    graph, related = {}, set()
    for rel in data["relationships"]:
        if not fields("relationship", rel, f"relationship {rel.get('row_id')}"):
            continue
        rid = rel["row_id"]
        if rid in related:
            bad.append(f"relationship {rid}: duplicate")
        related.add(rid)
        if rows.get(rid, {}).get("line_type") not in ("subtotal", "control"):
            bad.append(f"relationship {rid}: not a subtotal or control row")
        if rel["op"] is None and rel["terms"] is None:
            continue
        graph[rid] = expr({"op": rel["op"], "terms": rel["terms"]}, f"relationship {rid}")
    seen, done = set(), set()

    def visit(n, path):
        if n in path:
            bad.append(f"relationships: cycle {' -> '.join(path[path.index(n):] + [n])}")
            return
        if n in done:
            return
        for m in graph.get(n, []):
            visit(m, path + [n])
        done.add(n)

    for n in graph:
        visit(n, [])
    for rid, refs in graph.items():
        sub = {m for m in refs if m in graph}
        for m in sub:
            if set(graph[m]) & set(refs):
                bad.append(f"relationship {rid}: {m} and its own terms are both terms")
    def computed(refs, where):
        # a subtotal or control row is only ever computed: as a term it needs its recipe
        for m in refs:
            if rows.get(m, {}).get("line_type") in ("subtotal", "control") and m not in graph:
                bad.append(f"{where}: row {m} has no recipe")

    def in_totals(refs, where, own=None):
        # a check compares rows inside the totals (never a memo row); a section's check, rows of its own statement
        # (a report check may span statements: cash on the balance sheet against the cash flow's ending cash)
        for m in refs:
            if m not in rows:
                continue
            if rows[m]["line_type"] == "supplementary":
                bad.append(f"{where}: supplementary row {m} is outside every total")
            if own and sections.get(rows[m]["section"], {}).get("statement") != own:
                bad.append(f"{where}: row {m} is a row of another statement")

    for rid, refs in graph.items():
        computed(refs, f"relationship {rid}")
        # a total is computed in the run of its own statement, so its terms are rows of that statement
        own = sections.get(rows.get(rid, {}).get("section"), {}).get("statement")
        for m in refs:
            if m in rows and sections.get(rows[m]["section"], {}).get("statement") != own:
                bad.append(f"relationship {rid}: term {m} is a row of another statement")
    controlled = set()
    for c in data["controls"]:
        if not fields("control", c, f"control {c.get('section')}"):
            continue
        if c["section"] in controlled:
            bad.append(f"control {c['section']}: duplicate")
        controlled.add(c["section"])
        if c["section"] not in sections:
            bad.append(f"control {c['section']}: section not in the template")
        refs = expr(c["compare"], f"control {c['section']} compare")
        computed(refs, f"control {c['section']} compare")
        in_totals(refs, f"control {c['section']} compare", sections.get(c["section"], {}).get("statement"))
        if c["tolerance"] not in TOLERANCES:
            bad.append(f"control {c['section']}: tolerance {c['tolerance']!r} is not a named policy")
        if c["filed"] is None and c["reconstructed"] is None:
            bad.append(f"control {c['section']}: neither filed nor reconstructed")
        if c["filed"]:
            concepts.add(c["filed"])
        if c["reconstructed"] is not None:
            computed(expr(c["reconstructed"], f"control {c['section']} reconstructed"),
                     f"control {c['section']} reconstructed")
        where = f"control {c['section']}"
        alts = c.get("alternatives", [])
        if not isinstance(alts, list):
            bad.append(f"{where}: alternatives must be a list")
            alts = []
        if alts and c["reconstructed"] is None:
            bad.append(f"{where}: alternatives need a first reconstruction")
        for a in alts:
            if not isinstance(a, dict) or set(a) != ALTERNATIVE or not isinstance(a.get("capital"), dict) \
                    or set(a["capital"]) != CAPITAL:
                bad.append(f"{where}: an alternative has the fields {sorted(ALTERNATIVE)}, its capital "
                           f"{sorted(CAPITAL)}")
                continue
            cap = a["capital"]
            for x in (a["plus"], a["minus"], cap["inclusive"], cap["parent"]):
                if text(x):
                    concepts.add(x)
                else:
                    bad.append(f"{where} alternative {a['id']}: concept missing")
            for r in [cap["nci_row"], *cap["temporary_rows"]]:
                if rows.get(r, {}).get("line_type") != "item":
                    bad.append(f"{where} alternative {a['id']}: capital row {r} is not an item row")
        out = c.get("outside_rows")
        if out is not None:
            if not isinstance(out, dict) or set(out) != OUTSIDE:
                bad.append(f"{where}: outside_rows has the fields {sorted(OUTSIDE)}")
            else:
                if out["verified_by"] not in {x.get("section") for x in data["controls"]}:
                    bad.append(f"{where}: outside_rows verified by {out['verified_by']}, which has no control")
                for r in out["rows"]:
                    if rows.get(r, {}).get("section") != c["section"]:
                        bad.append(f"{where}: outside row {r} is not a row of the section")
    ns = data.get("not_spread", [])
    if not isinstance(ns, list) or not all(isinstance(x, dict) and set(x) == {"input", "why"} and text(x["input"])
                                           and text(x["why"]) for x in ns):
        bad.append("not_spread: a list of {input: text, why: text}")
    ids = set()
    for i, rc in enumerate(data.get("report_checks") or ()):     # format v2 (C6b): reported, never a check
        where = f"report check {rc.get('id', i) if isinstance(rc, dict) else i}"
        if not fields("report_check", rc, where):
            continue
        if not text(rc["id"]) or rc["id"] in ids:
            bad.append(f"{where}: id missing or duplicate")
        ids.add(rc["id"])
        for f, allowed in (("kind", CHECK_KINDS), ("period", PERIODS), ("tolerance", TOLERANCES)):
            if rc[f] not in allowed:
                bad.append(f"{where}: {f} {rc[f]!r}")
        if not (text(rc["name"]) and text(rc["note"])) or rc["deferred"] is not None and not text(rc["deferred"]):
            bad.append(f"{where}: name and note must be text, deferred text or null")
        both = []
        for side in ("left", "right"):
            e = rc[side]
            if isinstance(e, dict) and set(e) == {"concept"} and text(e["concept"]):
                concepts.add(e["concept"])
            else:
                refs = expr(e, f"{where} {side}")
                computed(refs, f"{where} {side}")
                both += refs
        in_totals(both, where)
        if "alternatives" in rc:
            alt, named = rc["alternatives"], {rc[x].get("concept") for x in ("left", "right") if isinstance(rc[x], dict)}
            if not isinstance(alt, dict) or not all(
                    k in named and isinstance(v, list) and v and all(text(c) and c != k for c in v)
                    for k, v in alt.items()):
                bad.append(f"{where}: alternatives map a concept a side names to a list of other concepts")
            else:
                concepts.update(c for v in alt.values() for c in v)
        if rc["kind"] == "filed" and not all(isinstance(rc[x], dict) and set(rc[x]) == {"concept"}
                                             for x in ("left", "right")):
            bad.append(f"{where}: a filed check compares filed concepts only")
    if "outcome_checks" in data:        # format v7: the statement checks
        oc = data["outcome_checks"]
        have = {c.get("section") for c in data["controls"] if isinstance(c, dict)}
        if (not isinstance(oc, dict) or set(oc) != {"sections", "report_checks"}
                or not all(isinstance(oc[k], list) and oc[k] for k in oc)):
            bad.append("outcome_checks: {sections: [sections with a control], report_checks: [report check ids]}")
        else:
            bad += [f"outcome_checks: section {s} has no control" for s in oc["sections"] if s not in have]
            bad += [f"outcome_checks: report check {r} is not in the template" for r in oc["report_checks"]
                    if r not in ids]
    if tax is not None:
        for k, st in data["statements"].items():
            for v in st["variants"]:
                try:
                    tax.network(v["fasb_network"], "pre")
                except FileNotFoundError:
                    bad.append(f"statement {k} variant {v['id']}: network {v['fasb_network']} not in us-gaap "
                               f"{tax.year}")
        for h, sid in owner.items():
            c = tax.concepts.get(h)
            if c is None or not c["abstract"]:
                bad.append(f"section {sid}: heading {h} {'not in' if c is None else 'not abstract in'} "
                           f"us-gaap {tax.year}")
        for concept in sorted(concepts):
            if not tax.is_amount(concept):
                bad.append(f"concept {concept}: not an amount in us-gaap {tax.year}")
        for c in data["controls"]:
            nb = sections.get(c.get("section"), {}).get("normal_balance")
            fb = (tax.concepts.get(c.get("filed")) or {}).get("balance")
            if nb and fb and nb != fb:
                bad.append(f"section {c['section']}: normal balance {nb}, but us-gaap {tax.year} gives its total "
                           f"{c['filed']} a {fb} balance")
    return bad


def load(path=STANDARD, tax=None):
    """The checked template, or TemplateError with every reason."""
    data = json.loads(Path(path).read_text(encoding="utf8"))
    bad = check(data, tax)
    if bad:
        raise TemplateError(bad)
    return Template(data, Path(path))


_active = None


def active():
    """The template this process spreads with: the standard one, opened on every call as an approved resource (so a
    run's frozen inventory and record see it, as they saw the skeleton) and checked when its content changes. Callers
    reach it as template.active(), so a test can put another template in its place."""
    global _active
    from . import provenance
    data = provenance.load_json(provenance.TEMPLATE)
    if _active is None or _active.data != data:
        bad = check(data)
        if bad:
            raise TemplateError(bad)
        _active = Template(data, provenance.TEMPLATE)
    return _active


def main(argv=None):
    from .taxonomy import Taxonomy, edition
    ap = argparse.ArgumentParser(prog="spreading.template")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="load checks and readiness per section, with a run record")
    c.add_argument("template", nargs="?", default=str(STANDARD))
    c.add_argument("--edition", required=True)
    a = ap.parse_args(argv)
    path = Path(a.template).resolve()
    tax = Taxonomy(edition(a.edition))
    data = json.loads(path.read_text(encoding="utf8"))
    d = runrec.new_run("template_check", {"command": f"python -s -E -m spreading.template check {a.template} "
                                                     f"--edition {a.edition}", "template": str(path),
                                          "edition": a.edition})
    bad = check(data, tax)
    ready = Template(data, path).readiness(tax) if not bad else {}
    runrec.write(d, "result.json", {"exceptions": bad, "readiness": ready})
    print(d.name)
    for reason in bad:
        print("  FAULT", reason)
    for sid, r in ready.items():
        print(f"  {sid:7} " + (r["kind"] if r["kind"] != "spread" else
                               ", ".join(f"{k}: {r[k] if isinstance(r[k], str) else 'yes' if r[k] else 'no'}"
                                         for k in ("route lines", "tie the section", "tie runs in today's spread",
                                                   "configured", "implemented", "spread"))
                               + f", definitions: {r['definitions']}"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

def format_at_least(n):
    """True when the active template's format is v`n` or later. From v4 a run records the "every line allocated" pass
    (spread.section_unknown); from v5 (v12, v13) an excluded cash flow line carries its reason from the filing's labels,
    its amount and its fact (spread.excluded). Earlier formats keep the records they had, so saved runs replay."""
    m = re.fullmatch(r"template_format_v(\d+)", active().data.get("format") or "")
    return bool(m) and int(m.group(1)) >= n
