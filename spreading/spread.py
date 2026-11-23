"""C2-C10 for one filing and one or more sections of one statement.

C2 reads the printed lines of each section by code, one entry per occurrence, and records every line it cannot
route. C3 makes a line a control when the company calculation decomposes it into printed lines that add up, sets
and signs; every other line goes to C4. C4 is the analyst's capped tool loop (analyst.run_c4, the six-rule
prompts), started from the line pack built here (line_pack, calc_notes); code checks its pieces for rules 5 and 6.
C8, the checker, runs the same loop blind on each line C4 placed in a row of the section, and code compares the
whole canonical decomposition (run, compare); an optional third opinion votes on disputed lines, and a line referred
to another section of the run is handed off there. An answer that is not a row of the section (COMBINED, OUTSIDE,
NOT_TOLD, another section's row) is an exception outcome, never a claim. The automatic A3 splits are off. C5 builds
the rows with a claim registry keyed by concept, dims, date and unit. C6 checks each control against its own parts,
each line's allocation against its amount, and the rows against the section total. C7 is off: a failed
reconciliation only lists the lines whose amounts match the gap, as a diagnostic. C9 sets complete / qualified /
blocked; C10 writes the dated record, which is opened before the run so every run ends with an outcome. The quote
helpers (norm, clean, quote_parts, quote_found) are shared with loop.py, steps.py and repair.py.

    python -s -E -m spreading.spread data/filings/47111/0001628280-26-008586 --section BS.CL \
        --max-calls 80 --max-tokens <n>

Each row's definition, belongs and not_belongs are read from the template (template format v5).
"""
import argparse
import hashlib
import itertools
import json
import random
import re
import sys
import time
import traceback
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from xml.etree import ElementTree as ET

from . import calc_network as cn
from .answer_keys import consistent
from .edgar import filing_inputs
from .filing import in_period, line_ident, period_length, period_words, tag_of
from .ledger import sent_by_role
from . import llm
from .llm import Caller
from . import provenance
from .packs import ROOT, load_rows, rows_block, section_rows, statement_rows
from .runrec import new_run, write
from . import tagless, template
from .taxonomy import Taxonomy, edition, editions

OPTIONS = ["COMBINED", "OUTSIDE", "NOT_TOLD"]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def freeze_hashes(prompts, models=None, num_ctx=4096, answer_cap=None, identity=True, hosted=None,
                  third=None):
    """Everything a company's result depends on: code, taxonomy, resources, prompts, model settings. The batch
    records it once and checks it before every company. `identity`: ask Ollama for the models' digests (off only in
    offline tests). `hosted`: the OpenRouter profile (HOSTED) when the roles run there; its endpoint pins, prices and
    thinking setting are frozen instead of Ollama digests, which hosted weights do not have. `third`: Haiku's
    settings (THIRD) when it runs on OpenRouter beside the local roles, frozen under "third"."""
    from . import llm
    if hosted:
        models = {r: v["model"] for r, v in hosted.items()}
    return {"code": {p.name: sha(p) for p in sorted((ROOT / "spreading").glob("*.py"))},
            "taxonomy": {str(t): sha(t) for t in editions().values()},   # every edition on disk
            "a2": None, "a3_contracts": sha(ROOT / "rulebook/draft/A3_contracts.json"),
            # the old skeleton's sha256, kept for old records; the active template is checked in approved_resources
            "template_rows": sha(ROOT / "rulebook/draft/A1_skeleton.json"),
            # the approved rulebook resources the run is set to open (the template carries the rows' definitions), with
            # their sha256; never what this process happened to open
            "approved_resources": provenance.inventory(),
            "prompts": {p: sha(ROOT / "prompts" / f"{p}.txt") for p in prompts},
            "mcp_pin": sha(ROOT / "mcp_pin.json"), "requirements_lock": sha(ROOT / "requirements.lock"),
            # two environments: the agents' (agents_env.json, both locks) and the server's (the pin's venv)
            "agents_env": sha(ROOT / "agents_env.json"),
            "langgraph_additions_lock": sha(ROOT / "requirements-langgraph-additions.lock"),
            "interpreters": {"client": sys.executable, "client_version": sys.version.split()[0],
                             "server": str(ROOT / json.load(open(ROOT / "mcp_pin.json"))["venv"]
                                           / "Scripts" / "python.exe")},
            "models": {**(models or LOOP_MODELS), "seed": 7, "num_ctx": num_ctx,
                       "num_predict": answer_cap or llm.NUM_PREDICT, "temperature": 0,
                       # each local model's pinned num_batch, or "batch not pinned"
                       **({} if hosted else {"num_batch": llm.batch_pins((models or LOOP_MODELS).values())}),
                       # hosted: the embedder is hosted too (every role online), its settings
                       **({"engine": "openrouter",
                           "reasoning": ({r: {"setting": v["reasoning"], "allowance": v["allowance"]}
                                          for r, v in hosted.items()} if per_role(hosted) else {"enabled": False}),
                           "endpoints": {r: {"providers": v["providers"], "price_usd_per_token": list(v["price"]),
                                             **({k: v[k] for k in ("provider_names", "quantization") if k in v}
                                                if per_role(hosted) else {})}
                                         for r, v in hosted.items()},
                           "embedder": llm.HOSTED_EMBED}
                          if hosted
                          else llm.model_identity(models or LOOP_MODELS) if identity else {}),
                       **({"third": {r: {**{k: x for k, x in v.items() if k != "price"},
                                         "price_usd_per_token": list(v["price"])} for r, v in third.items()}}
                          if third else {})}}


# ---- C2: the printed lines of one section ---------------------------------------------------------

def calc_routes(f, role):
    """Where the template's filed section totals reach in the selected role's calculation network, for lines printed
    under no listed heading (C1). Only an anchor printed in the role counts. Returns concept -> [(section, anchor,
    arcs from the anchor down)], one entry per path, and the parent-child pairs not followed because an arc is
    prohibited or repeated. Calculation arcs say what adds up, not what the statement prints: they route only lines
    already printed, never add one.

    A section with a derived total (template.derived_totals) whose own total is not printed is reached from its
    `plus` total without passing through its `minus` total (non-current liabilities: total liabilities outside
    current liabilities). The `plus` total itself is not routed to the section: it is not one of its lines.

    The walk runs on each calculation network of the role separately (calc_network), the reaches are pooled, and
    each step names its arcrole; a line reached twice, in one network or in two, is not uniquely routed. `blocked`
    holds (section, parent, child, reason) for each unusable pair met below a reached node."""
    t = template.active()
    totals, derived = t.section_totals(), t.derived_totals()
    printed = {r["concept"] for r in f.printed(role)}
    n = cn.networks(f, role)
    reach, blocked = defaultdict(list), set()

    def walk(section, root, minus=None):
        for ar, x in cn.reach_down(n, root, lambda c: c == minus).items():
            for c, chains in x["reach"].items():
                if c != minus:
                    reach[c] += [(section, root, [{"parent": s["parent"], "child": s["child"],
                                                   "weight": str(s["weight"]), "arcrole": ar} for s in ch])
                                 for ch in chains]
            nodes = {root} | {c for c in x["reach"] if c != minus}
            blocked.update((section, p, c, why) for (p, c), why in n["networks"][ar]["bad"].items() if p in nodes)
    for section, anchor in totals.items():
        if anchor in printed:
            reach[anchor].append((section, anchor, []))
            walk(section, anchor)
    for section, d in derived.items():
        if totals.get(section) not in printed and d["plus"] in printed and d["minus"] in printed:
            walk(section, d["plus"], d["minus"])
    return reach, blocked


# written verbatim in each memo line's route
MEMO_NOTE = ("no usable calculation path found; missing or unusable relationships can conceal statement membership")


def route_of(r, headings, reach, memo=()):
    """(section or None, route, calculation reaches) of one printed occurrence: the nearest listed heading above it,
    else exactly one calculation path from one printed section total (calc_routes). Shared by section_lines and the
    coverage states (C4a). `memo`: the memo sections (template.memo). A memo section's heading takes a line only when
    no calculation path reaches it; otherwise the heading is set aside (recorded) and the line keeps the ordinary
    route, as if the memo headings were not listed."""
    heads = [p for p in reversed(r["path"]) if p in headings]
    calc = reach.get(r["concept"], [])
    aside = heads[0] if calc and heads and headings[heads[0]] in memo else None
    if aside:
        heads = [h for h in heads if headings[h] not in memo]
    head = heads[0] if heads else None
    if head is not None:
        sec, route = headings[head], {"basis": "heading", "heading": head}
        if sec in memo:
            route["memo"] = MEMO_NOTE
        if aside:
            route["memo_heading_set_aside"] = aside
        if calc and all(s != sec for s, _, _ in calc):
            # heading and calculation disagree: recorded and shown to the model, which judges; code never acts on it
            route["calculation"] = sorted({(s, a) for s, a, _ in calc})
        return sec, route, calc
    if len(calc) == 1:
        sec, anchor, arcs = calc[0]
        route = {"basis": "calculation", "anchor": anchor, "arcs": arcs}
        if aside:
            route["memo_heading_set_aside"] = aside
        return sec, route, calc
    return None, None, calc


def organized(r, sec, code, route, calc, headings, memo):
    """(section, route) of an item the organizer sorted: code's own route when both name the same section; else basis
    "organizer" with code's section beside it, the CALCULATION observation recomputed against the organizer's
    section, and in a memo section the memo heading it is printed under, if any. (None, None) for a blank."""
    if sec is None or sec == code:
        return sec, route if sec else None
    out = {"basis": "organizer", "code_section": code}
    if calc and all(s != sec for s, _, _ in calc):
        out["calculation"] = sorted({(s, a) for s, a, _ in calc})
    head = next((p for p in reversed(r["path"]) if headings.get(p) == sec), None) if sec in memo else None
    if head:
        out.update(heading=head, memo=MEMO_NOTE)
    return sec, out


def member_only(f, concept, key, role=None):
    """Whether the concept has facts of the filer's own entity at the key only with dimensions (read gives none):
    printed by member only. Its members are never summed. Under the printed rows rule (a template whose outcome checks
    decide, format v7) and with the statement's `role`, only when
    the statement's own page prints the concept under a member block: a member fact from a note (a lawsuit, an equity
    investee's own figures) does not mark a statement line, which then has no amount for the period. Without a page
    for the role, as before."""
    ctx = getattr(f, "contexts", {})
    if not any(in_period(ctx[x["context"]], key) and ctx[x["context"]]["dims"] and not x.get("piece")
               and ctx[x["context"]]["entity"] == f.entity and x["value"] != ""
               for x in getattr(f, "by_concept", {}).get(concept, [])):
        return False
    if role is None or template.active().outcome_checks is None:
        return True
    shown = page_members(f, role)
    return shown is None or concept in shown


def page_members(f, role):
    """The concepts the role's R page prints a row for under a member block (`<tr class="rh">`), as prefix:name;
    None when FilingSummary names no one page for the role, or the summary or the page cannot be read (then the line
    is marked as before). Read once per filing and role."""
    cache = f.__dict__.setdefault("_page_members", {})
    if role not in cache:
        try:
            cache[role] = read_page_members(f, role)
        except (ET.ParseError, OSError, UnicodeDecodeError):
            cache[role] = None
    return cache[role]


def read_page_members(f, role):
    """page_members, read from the files."""
    d = getattr(f, "dir", None)
    fs = Path(d) / "FilingSummary.xml" if d else None
    names = ([r.findtext("HtmlFileName") for r in ET.parse(fs).getroot().iter("Report")
              if (r.findtext("Role") or "").strip() == role] if fs and fs.is_file() else [])
    page = Path(d) / names[0] if len(names) == 1 and names[0] else None
    if page is None or not page.is_file():
        return None
    shown, inside = set(), False
    for cls, tr in re.findall(r'<tr class="(r[a-z]*)">(.*?)</tr>', page.read_text(encoding="utf8"), re.S):
        if cls == "rh":
            inside = True
            continue
        m = re.search(r"defref_([^_'\"]+)_([^'\"]+)", tr)
        if inside and m:
            shown.add(f"{m.group(1)}:{m.group(2)}")
    return shown


def ident(x):
    """(tag, dims) of the amount a line, claim or piece names: a split piece's original tag and its member dims
    (corrections.apply), else its concept and dims. Ownership and overlap compare these, never the key, so one raw
    amount cannot be claimed through a piece and through another line."""
    return (x.get("tag") or x["concept"],
            tuple(sorted(tuple(p) for p in ((x.get("dims") or {}).items() if isinstance(x.get("dims"), dict)
                                             else x.get("dims") or ()))))


def raw(f, concept):
    """A line's provenance fields beyond its concept: a split piece's tag and member dims ([[axis, member]]), {}
    for any other line, so the records of unsplit lines are unchanged."""
    p = getattr(f, "pieces", {}).get(concept)
    return {"tag": p["tag"], "dims": sorted([a, m] for a, m in p["dims"].items())} if p else {}


def read(f, concept, key):
    """The concept's facts at the run's period key (C1): Filing.instant_facts for an instant; for a duration,
    Filing.facts_for, a plain currency only."""
    if key[0] == "instant":
        return f.instant_facts(concept, key[1])
    return [x for x in f.facts_for(concept, key) if x["currency"]]


def accuracy(x):
    return "INF" if x["decimals"] in (None, "INF") else int(x["decimals"])


def disagree(facts):
    """Duplicate facts at the key in two currencies, or not consistent (answer_keys.consistent: XBRL's interval rule
    over the whole group, and one value per decimals): none is chosen and the amount stays unresolved."""
    return len({x["currency"] for x in facts}) > 1 or not consistent([(Decimal(x["value"]), accuracy(x))
                                                                      for x in facts])


def chosen(facts):
    """The most precise of consistent duplicates (XBRL duplicates guidance, section 9); the first among equals."""
    return max(facts, key=lambda x: float("inf") if accuracy(x) == "INF" else accuracy(x))


def section_lines(f, role, section, exc, unrouted, key=None, member=None, refused=None, org=None):
    """Lines of the section, in printed order, read at the run's period key (C1; default the balance-sheet instant).
    A line is routed by the nearest listed heading above it; a line under no listed heading is routed by calculation
    (calc_routes) when exactly one path from one printed section total reaches it. Each line records its route.
    Lines routed neither way go to `unrouted` (a statement-wide diagnostic); a current-section line lost that way
    breaks the section total's decomposition, which C3 and C6 report. A line routed here that is printed by member
    only (member_only) is not read; it goes to `member` with a C2 exception, so the run marks the section's coverage
    incomplete. A line routed here whose duplicate facts disagree (disagree) is not read either; it goes to
    `refused` with a C2 exception, and the run marks the section incomplete the same way. `org`: the organizer's
    frozen assignment by concept (organizer.by_concept); an item in it, and its second occurrence, take the
    organizer's section (organized), a blank none."""
    lines, seen = [], set()
    key = key or ("instant", f.period_end)
    t = template.active()
    headings = t.heading_to_section()
    memo = {s for s in set(headings.values()) if t.memo(s)}
    reach, blocked = calc_routes(f, role)
    printed = f.printed(role)
    labels = {r["concept"]: r["label"] for r in printed}
    for sec, p, c, why in sorted(blocked):
        if sec == section:
            exc.append({"step": "C2", "class": "extraction", "concept": c,
                        "why": f"calculation arc {p} -> {c} not usable ({why}); not used for routing"})
    for r in printed:
        facts = read(f, r["concept"], key)
        sec, route, calc = route_of(r, headings, reach, memo)
        if org is not None and r["concept"] in org:
            sec, route = organized(r, org[r["concept"]], sec, route, calc, headings, memo)
        if route and "memo" in route:       # the printed heading, shown to the model (line_pack)
            route["heading_label"] = labels.get(route["heading"]) or tagless.show(route["heading"])
        if not facts:
            if sec == section and member is not None and r["concept"] not in member and member_only(f, r["concept"],
                                                                                                  key, role):
                member.append(r["concept"])
                exc.append({"step": "C2", "class": "extraction", "concept": r["concept"],
                            "why": f"printed by member only: {r['concept']} has facts at the key only with "
                                   "dimensions; not read in this MVP"})
            continue
        if route is not None and route["basis"] == "heading":
            if any(s == section != sec for s, _, _ in calc):
                exc.append({"step": "C2", "class": "extraction", "concept": r["concept"],
                            "why": f"heading routes it to {sec} but the {section} total's calculation reaches "
                                   f"it; heading kept"})
        elif route is None:
            if any(s == section for s, _, _ in calc) and r["concept"] not in (org or ()):
                exc.append({"step": "C2", "class": "extraction", "concept": r["concept"],
                            "why": "under no listed heading and reached by more than one calculation path "
                                   f"({', '.join(sorted({a for _, a, _ in calc}))}); not routed"})
            unrouted.append(r["concept"])
            continue
        if sec != section:
            continue
        if r["concept"] in seen:
            exc.append({"step": "C2", "class": "extraction", "concept": r["concept"],
                        "why": "concept printed twice in the section; second occurrence not used"})
            continue
        seen.add(r["concept"])
        if disagree(facts):
            exc.append({"step": "C2", "class": "extraction", "concept": r["concept"],
                        "why": "duplicate facts not consistent or in two currencies"})
            if refused is not None:
                refused.append(r["concept"])
            continue
        x = chosen(facts)
        lines.append({"concept": r["concept"], "label": r["label"], "fact": x["id"], "unit": x["currency"],
                      "decimals": x["decimals"], "amount": Decimal(x["value"]), "path": r["path"],
                      "route": route, "period": key or ("instant", f.period_end), **raw(f, r["concept"])})
    return lines


def half_unit(decimals):
    """Rounding interval of one filed amount: half of its last reported digit."""
    return Decimal(0) if decimals in (None, "INF") else Decimal(10) ** (-int(decimals)) / 2


# Absent children: with the period key, a calculation child printed on the statement with no applicable fact, or
# only explicitly nil facts, is left out of the total's sum (XBRL 2.1 5.2.5.2; Calculations 1.1 5.1 and 5.6),
# recorded with its reason; no zero is written to any row. Mixed nil and valued facts, or a member-only value, set
# the total aside.
OMIT_ABSENT = True


def own_facts(f, concept, key):
    """The concept's facts of the filer's own entity in the period, without dimensions."""
    ctx = f.contexts
    return [x for x in f.by_concept.get(concept, []) if in_period(ctx[x["context"]], key)
            and (not ctx[x["context"]]["dims"] or x.get("piece")) and ctx[x["context"]]["entity"] == f.entity]


def mixed_nil(f, concept, key):
    """Whether the concept has both explicitly nil and other facts (own entity, period, no dimensions)."""
    here = own_facts(f, concept, key)
    return any(x.get("nil") for x in here) and not all(x.get("nil") for x in here)


def period_absence(f, role, concept, key):
    """Why a calculation child not among the section's lines has no amount for the period, or None when that is not
    established: "no fact for the period" (printed on the statement, no fact of the filer's own entity in the period
    without dimensions, and not printed by member only) or "nil for the period" (every such fact explicitly nil, and
    not printed by member only). An empty value that is not nil, mixed nil and valued facts, a valued fact, a child
    printed by member only (with or without a nil beside it) or not printed on the statement: None."""
    if key is None or concept not in {r["concept"] for r in f.printed(role)} or member_only(f, concept, key, role):
        return None
    here = own_facts(f, concept, key)
    if not here:
        return "no fact for the period"
    if all(x.get("nil") and x["value"] == "" for x in here):
        return "nil for the period"
    return None


def mark_controls(f, role, lines, exc, canonical=None, controls=(), key=None):
    """A control's parts are printed lines of the section that add up to it within rounding, by the two-sided
    tolerance with the calculation's own weights (`tolerance`).

    Asked of each calculation network of the role (calc_network). A network with any unusable relationship out of
    the line cannot verify it (its other children are never summed alone); a supported network needs every child
    printed in the section and a tie. One supported network, the others absent: "verified". Two or more supported:
    verified only when they give the same (child, weight) set. Otherwise "unavailable" with `reason` and
    `alternatives`; "none" when no network has children for it. Each line carries `control_candidate` (a filed
    statement control in `controls`, or calculation children in some network) and `decomposition`; `control` is
    decomposition == "verified".

    With OMIT_ABSENT and the period key, a child that is not a line of the section but has no amount for the period
    (period_absence) is left out of the sum: the total is checked on the remaining children, at least one, with the
    same weights and tolerance; `parts` holds those, and `omitted_parts` the left-out children with weight and
    reason. A child with both nil and valued facts for the period (mixed_nil) sets the network aside, whether or not
    it is a line of the section."""
    by = {l["concept"]: l for l in lines}
    n = cn.networks(f, role)
    for l in lines:
        answers = {ar: x for ar, x in cn.children(n, l["concept"]).items() if x["answer"] != "absent"}
        l["control"], l["parts"] = False, []
        l["control_candidate"] = bool(answers) or l["concept"] in controls
        l["decomposition"] = "unavailable" if answers else "none"
        if not answers:
            continue
        alts, why = {}, None
        for ar, x in sorted(answers.items()):
            if x["answer"] == "unusable":
                p, c, reason = x["bad"][0]
                why = why or f"unusable relationship {p} -> {c}: {reason}"
                alts[ar] = {"unusable": [list(b) for b in x["bad"]]}
                continue
            kids, gone = x["parts"], []
            mixed = [c for c, _ in kids if mixed_nil(f, c, key)] if OMIT_ABSENT and key is not None else []
            if mixed:               # present among the lines or not: a nil beside a value is never summed
                why = why or "a calculation child has nil and valued facts for the period; not used as a control"
                alts[ar] = {"children": [[c, str(w)] for c, w in kids], "mixed_nil": mixed}
                continue
            if OMIT_ABSENT and key is not None:
                gone = [(c, w, period_absence(f, role, c, key)) for c, w in kids if c not in by]
                gone = [g for g in gone if g[2]]
            left = [(c, w) for c, w in kids if c not in {g[0] for g in gone}]
            omitted = {"omitted": [[c, str(w), r] for c, w, r in gone]} if gone else {}
            if not all(c in by for c, _ in left):
                why = why or "calculation children not all printed in the section; not used as a control"
                alts[ar] = {"children": [[c, str(w)] for c, w in kids],
                            "unprinted": [c for c, _ in left if c not in by], **omitted}
                continue
            if not left:
                why = why or "no calculation child has an amount for the period; not used as a control"
                alts[ar] = {"children": [[c, str(w)] for c, w in kids], **omitted}
                continue
            x["kept"], x["gone"] = left, gone
            total = sum(w * by[c]["amount"] for c, w in left)
            tol = tolerance([(by[c]["fact"], w, by[c]["decimals"]) for c, w in left],
                            [(l["fact"], 1, l["decimals"])], canonical or {})
            alts[ar] = {"children": [[c, str(w)] for c, w in kids], "sum": str(total), "tolerance": str(tol),
                        **omitted}
            if abs(total - l["amount"]) > tol:
                why = why or f"calculation does not add: parts {total}, filed {l['amount']}"
        sets = {frozenset((c, w) for c, w in x["parts"]) for x in answers.values() if x["answer"] == "supported"}
        if why is None and len(sets) > 1:
            why = "the role's calculation networks give different decompositions"
        if len(answers) > 1 or why:
            l["alternatives"] = alts
        if why:
            l["reason"] = why
            exc.append({"step": "C3", "class": "extraction", "concept": l["concept"], "why": why})
            continue
        x = next(iter(answers.values()))
        l["control"], l["parts"], l["decomposition"] = True, list(x.get("kept", x["parts"])), "verified"
        if x.get("gone"):
            l["omitted_parts"] = [[c, w, r] for c, w, r in x["gone"]]


def leaves(by, concept):
    """The non-control lines a control decomposes into, with their combined weight; a part the section does not own
    (moved to another section by a hand-off) is left out, and the caller reports it."""
    out = []
    for c, w in by[concept]["parts"]:
        if c not in by:
            continue
        if by[c]["control"]:
            out += [(x, w * v) for x, v in leaves(by, c)]
        else:
            out.append((c, w))
    return out


def absent(by, parts):
    """The parts, at any depth below retained controls, that the section does not own: a missing part fails the
    total's checks even when the missing amounts cancel."""
    out = []
    for c, _ in parts:
        if c not in by:
            out.append(c)
        elif by[c]["control"]:
            out += absent(by, by[c]["parts"])
    return out


# ---- C3: signs and the A2 lookup ------------------------------------------------------------------

def balance_of(tax, f, concept):
    concept = tag_of(concept)
    if concept in tax.concepts:
        return tax.concepts[concept]["balance"]
    return f.extension.get(concept, {}).get("balance")


def sign(balance, section_balance):
    if not balance or not section_balance:
        return None
    return 1 if balance == section_balance else -1


def under(calc, totals, items):
    """Whether one of `items` is a calculation descendant of one of `totals` in `calc` (parent -> [(child, weight)]).
    On the run path (C7) it is asked of each calculation network's usable relationships separately; a caller may
    also pass the pooled calculation."""
    seen, todo = set(), list(totals)
    while todo:
        for c, _ in calc.get(todo.pop(), []):
            if c in items:
                return True
            if c not in seen:
                seen.add(c)
                todo.append(c)
    return False


def path_signs(tax, f, role, key, statement, lines):
    """C3a: each line's direction d on an income or cash flow run, before any model call. Code follows the role's
    calculation arcs up from the line to an anchor, a control's filed concept of the statement printed in the role;
    an arc counts only when its parent has a fact for the run's period in the line's unit. d = a x the weights on
    the path, where a is -1 when the anchor's balance is opposite to the statement's normal balance and +1 otherwise
    (no balance included). Without a path, the line's balance against the normal balance, with why. A weight other
    than +-1, paths that disagree, or neither a path nor a balance leave d None with the reason: never forced to +1.
    Returns concept -> {d, basis, anchor, path, weights, why_fallback[, why]}."""
    t = template.active()
    normal = {s["normal_balance"] for sid, s in t.sections.items() if t.statement_of(sid) == statement
              and s["kind"] == "spread"}
    if not normal:      # a statement read for its memo sections only (template v12's cash flow): their balance
        normal = {s["normal_balance"] for sid, s in t.sections.items() if t.statement_of(sid) == statement
                  and t.memo(sid)}
    if len(normal) != 1:
        raise RuntimeError(f"C3: the {statement} sections do not share one normal balance: {sorted(map(str, normal))}")
    normal = normal.pop()
    printed = {r["concept"] for r in f.printed(role)}
    anchors = {c["filed"] for s, c in t.controls.items() if t.statement_of(s) == statement and c["filed"]} & printed
    n = cn.networks(f, role)

    def walk(c, unit):
        """Paths up within each network (calc_network.paths_up), each with its arcrole; a parent counts only with a
        fact for the period in the line's unit (the first one without is the gap)."""
        gaps = []

        def keep(p):
            xs = read(f, p, key)
            if not xs or xs[0]["currency"] != unit:
                gaps.append(p)
                return False
            return True
        got = cn.paths_up(n, c, anchors, keep)
        paths = [(a, ch, ws, ar) for ar, x in sorted(got.items()) if x["answer"] == "supported"
                 for a, ch, ws in x["paths"]]
        bad = sorted({(p, q, why) for x in got.values() for *_, step in x.get("unusable_paths", [])
                      for p, q, why in step})
        return paths, (gaps[0] if gaps else None), bad

    def direction(anchor, ws):
        d = sign(balance_of(tax, f, anchor), normal) or 1
        for w in ws:
            d *= w
        return d
    out = {}
    for l in lines:
        c = l["concept"]
        if c in anchors:
            out[c] = {"d": direction(c, []), "basis": "anchor", "anchor": c, "path": [], "weights": [],
                      "why_fallback": None}
            continue
        paths, gap, bad = walk(c, l["unit"])
        odd = sorted({w for _, _, ws, _ in paths for w in ws if abs(w) != 1})
        per = defaultdict(set)
        for a, _, ws, ar in paths:
            per[ar].add(direction(a, ws))
        ds = set().union(*per.values())
        if paths:
            a, ch, ws, ar = paths[0]
            # opposing directions within one network: "sign conflict"; between networks: "networks disagree on
            # direction"; never settled by the balance
            conflict = ("sign conflict" if any(len(v) > 1 for v in per.values()) else "networks disagree on direction")
            out[c] = {"d": None if odd or len(ds) > 1 else int(ds.pop()), "basis": "path", "anchor": a, "path": ch,
                      "weights": [str(w) for w in ws], "arcrole": ar, "why_fallback": None,
                      **({"why": f"weight {odd[0]} on path"} if odd else
                         {"why": conflict, "paths": [[a, ch, [str(w) for w in ws], ar] for a, ch, ws, ar in paths]}
                         if len(ds) > 1 else {})}
            continue
        d = sign(balance_of(tax, f, c), normal)
        why = (f"the calculation path passes through {gap}, which has no fact for this period in {l['unit']}" if gap
               else "no calculation path to a statement control")
        if bad:
            why += "; only through unusable relationships " + ", ".join(f"{p} -> {q} ({r})" for p, q, r in bad)
        out[c] = {"d": d, "basis": "balance", "anchor": None, "path": [], "weights": [], "why_fallback": why,
                  **({} if d else {"why": "sign unknown: no calculation path and no balance"})}
    return out


def taxonomy_year(f):
    ns = f.nsmap.get("us-gaap", "")
    return ns.rstrip("/").rsplit("/", 1)[-1]


# ---- the line pack (analyst.py) and the quote helpers (loop.py, steps.py, repair.py) -------------------------------

def calc_notes(line, section):
    """The observation for a line whose heading and calculation disagree (route "calculation", from section_lines):
    which total the company's calculation adds it to. A filing fact shown to the model; code does not move the line,
    and nothing says where another section's model put it."""
    t, out = template.active(), []
    for s, a in line.get("route", {}).get("calculation", []):
        d = t.derived_totals().get(s)
        out.append(f"the company's calculation adds this line to {tagless.ref(a)} without passing through "
                   f"{tagless.ref(d['minus'])}" if d and a == d["plus"] else
                   f"the company's calculation adds this line to {tagless.ref(a)}, not to "
                   f"{tagless.ref(t.filed_total(section))}")
    return out


def line_pack(f, line, section, ids):
    """The start's two blocks: a ROWS SHOWN line before the rows and the catch-all with its section named. A queued
    line (section None) has no SECTION line, and its rows are every offered row of the statement under section
    headings. A memo section's line (template.memo) also shows the heading it is printed under; the section's name,
    in ROWS SHOWN, says its rows sit outside the totals."""
    t = template.active()
    # the line's ID only, so the model can name it in a tool call; no tag, definition or balance (tagless.py)
    head = [f"LINE ID     {tagless.show(line['concept'])}"]
    lines = [f"PRINTED AS  {line['label']}", f"AMOUNT      {line['amount']:,} {line['unit']} "
                                                f"{period_words(line.get('period') or ('instant', f.period_end))}"]
    memo = (line.get("route") or {}).get("heading_label") if section and t.memo(section) else None
    lines += head + ([f"SECTION     {section}"] + [f"CALCULATION {n}" for n in calc_notes(line, section)]
                     + ([f"PRINTED UNDER {memo}"] if memo else []) if section else []) + [""]
    shown = [f"ROWS SHOWN: the rows of this line's section, {section} ({t.sections[section]['name']})." if section
             else f"ROWS SHOWN: every row of the {t.statements[t.statement_of(t.section_of(ids[0]))]['name']}, "
                  "grouped by section."]
    # two blocks, so code can tell where a quote came from; joined they are the whole message
    return {"line": "\n".join(lines),
            "rows": "\n".join(shown + rows_block(ids, named=True, headed=not section)
                              + ["  " + " | ".join(OPTIONS)])}


# the pack's own field names are formatting, not evidence: a quote may run across them
PACK_LABELS = ["PRINTED AS", "AMOUNT", "LINE ID", "CONCEPT", "LABEL", "DEFINITION", "TYPES", "SECTION", "CALCULATION",
               "ROWS YOU MAY CHOOSE", "PRINTED UNDER"]


def norm(s):
    return " ".join(s.replace(",", "").replace("$", "").casefold().split())


def clean(text):
    for k in PACK_LABELS:
        text = text.replace(k, " ")
    return norm(text)


def quote_parts(quote):
    parts = [p.strip(" .") for p in norm(quote).replace("\u2026", "...").replace(";", "...").split("...")]
    return [p for p in parts if p]


def quote_found(quote, text):
    """True if every part of the quote (split at ellipses and semicolons) occurs in the text, ignoring case, spacing,
    thousands separators and dollar signs."""
    parts = quote_parts(quote)
    return bool(parts) and all(p in clean(text) for p in parts)


# ---- C5 build and C6 checks -----------------------------------------------------------------------

def rival_amount_of(f, key, lines, pieces, k):
    """An open claim's decision card: the amount the run holds for the rival key (tag, dims) in its period: a printed line's,
    else a committed piece's, else the one filed value of an undimensioned concept; None otherwise."""
    for l in lines:
        if line_ident(f, l["concept"]) == k:
            return l["amount"]
    for p in pieces:
        if ident(p) == k:
            return p["amount"]
    values = {Decimal(str(y["value"])) for y in (read(f, k[0], key) if not k[1] else [])}
    return values.pop() if len(values) == 1 else None


def build(f, lines, alloc, pieces, moved, signs, exc):
    """Claim registry keyed by (concept, dims, period, unit): one claim per amount; a conflict is an exception. A
    committed piece carries its dims, so a breakdown member is a different amount from the line's total."""
    claims, owner = [], {}
    by = {l["concept"]: l for l in lines}
    cands = [{"fact": l["fact"], "concept": l["concept"], "row": alloc[l["concept"]]["row"],
              "by": alloc[l["concept"]]["by"], "amount": l["amount"] - moved.get(l["concept"], 0),
              "source": l["concept"], **{k: l[k] for k in ("tag", "dims") if k in l}}
             for l in lines if l["concept"] in alloc]
    for i, c in enumerate(cands + pieces):
        src = by[c["source"]]
        key = ident(c) + (tuple(src.get("period") or ("instant", f.period_end)), src["unit"])
        if key in owner:
            exc.append({"step": "C5", "class": "mapping", "concept": c["concept"],
                        "why": f"amount already claimed by {owner[key]}; second claim dropped"})
            continue
        owner[key] = c["source"]
        d = signs[c["source"]]
        c["signed"] = c["amount"] * d
        # C6: the original facts behind the claim, with coefficients: a whole line d, a piece d*r, a rest d for the
        # line's fact and -d*r for each piece's fact
        c["facts"] = ([(c["fact"], d * c.get("r", 1), c.get("decimals"))] if i >= len(cands) else
                      [(c["fact"], d, src["decimals"])] + [(p["fact"], -d * p.get("r", 1), p.get("decimals"))
                                                           for p in pieces if p["source"] == c["source"]])
        claims.append(c)
    return claims


def resolve_control(f, role, section, key=None, canonical=None):
    """The reconstructed total of a section with a derived total (template.derived_totals), or None for a section
    with no derived total. Both filed totals must be printed in the role and have facts of the filer's own entity at
    the balance-sheet date, with no dimensions (Filing.instant_facts), one currency, and no conflicting duplicates;
    `minus` must be a +1 calculation child of `plus`. The result keeps the formula, coefficients, every fact ID
    (duplicates included, for the independence check) and the half-unit allowance of both totals, and is labelled
    reconstructed. When the section's own total is printed, it stays the control and the reconstructed value is
    only compared with it: a difference beyond both allowances is a conflict, and C6 then fails the section. A total
    that cannot be reconstructed is returned refused, with why."""
    t = template.active()
    d = t.derived_totals().get(section)
    if d is None:
        return None
    key = key or ("instant", f.period_end)
    name = t.filed_total(section)
    printed = {r["concept"] for r in f.printed(role)}
    out = {"concept": f"{section} total (reconstructed)", "control": True, "parts": [],
           "label": "reconstructed, not a filed subtotal", "filed_total_printed": name in printed}
    got = {}
    for k in ("plus", "minus"):
        c = d[k]
        xs = read(f, c, key) if c in printed else []
        if c not in printed or not xs:
            return {**out, "refused": f"{c} {'not printed' if c not in printed else 'has no fact'} "
                                      f"{period_words(key)} (own entity, no dimensions)"}
        if disagree(xs):
            return {**out, "refused": f"duplicate facts of {c} not consistent or in two currencies"}
        got[k] = xs
    p, m = chosen(got["plus"]), chosen(got["minus"])
    if p["currency"] != m["currency"]:
        return {**out, "refused": f"{d['plus']} is in {p['currency']}, {d['minus']} in {m['currency']}"}
    # asked of each calculation network not absent for `plus`: every one must hold plus -> minus at +1 and the same
    # other parts, never whichever network permits the reconstruction
    answers = {ar: x for ar, x in cn.children(cn.networks(f, role), d["plus"]).items() if x["answer"] != "absent"}
    sets = {ar: frozenset(x.get("parts", [])) for ar, x in answers.items()}
    lacking = sorted(ar for ar, x in answers.items() if x["answer"] != "supported"
                     or (d["minus"], Decimal(1)) not in x["parts"])
    if not answers or (len(answers) == 1 and lacking):
        return {**out, "refused": f"{d['minus']} is not a +1 calculation child of {d['plus']}"}
    if lacking or len(set(sets.values())) > 1:
        return {**out, "refused": f"the role's calculation networks disagree on {d['plus']}",
                "networks": {ar: {"answer": x["answer"], "children": [[c, str(w)] for c, w in x.get("parts", [])],
                                  **({"unusable": [list(b) for b in x["bad"]]} if "bad" in x else {})}
                             for ar, x in sorted(answers.items())}}
    kids = next(iter(answers.values()))["parts"]
    out.update(amount=Decimal(p["value"]) - Decimal(m["value"]), unit=p["currency"], decimals=None,
               allowance=half_unit(p["decimals"]) + half_unit(m["decimals"]),
               parts=[(c, Decimal(w)) for c, w in kids if c != d["minus"]],
               reconstructed={"formula": f"{d['plus']} - {d['minus']}", "coefficients": {d["plus"]: 1, d["minus"]: -1},
                              "facts": {d["plus"]: p["id"], d["minus"]: m["id"]},
                              "fact_ids": sorted(x["id"] for k in got for x in got[k]),
                              "values": {d["plus"]: p["value"], d["minus"]: m["value"]},
                              "decimals": {d["plus"]: p["decimals"], d["minus"]: m["decimals"]}})
    if name in printed:
        filed = read(f, name, key)
        if filed:
            # the canonical two-sided tolerance; the allowance stays as provenance
            gap = Decimal(filed[0]["value"]) - out["amount"]
            tol = tolerance([(p["id"], 1, p["decimals"]), (m["id"], -1, m["decimals"])],
                            [(filed[0]["id"], 1, filed[0]["decimals"])], canonical or {})
            out["filed_comparison"] = {"filed": filed[0]["value"], "filed_fact": filed[0]["id"],
                                       "reconstructed": out["amount"], "gap": gap, "tolerance": tol,
                                       "conflict": abs(gap) > tol}
    return out


def descendants(n, concept):
    """Every concept below `concept` in any of the role's calculation networks (supported relationships only)."""
    out, todo = set(), [concept]
    while todo:
        for x in cn.children(n, todo.pop()).values():
            for c, _ in x.get("parts", []):
                if c not in out:
                    out.add(c)
                    todo.append(c)
    return out


def capital_side(f, role, section, alt, key, canonical, st, owner):
    """A further reconstruction of the section's total from the capital side (template format v4, `alternatives`):
    plus - minus - E - every capital line, where E is the equity including
    noncontrolling interest if printed, else parent equity, and the capital lines are the lines routed to the capital
    rows' section outside E's calculation tree (temporary equity; noncontrolling interest when E is parent equity),
    each fact counted once. Unavailable, with why, unless the partition is complete and disjoint: every input printed
    with facts of the filer's own entity at the period, no dimensions, consistent duplicates and one currency (missing
    is never zero; a printed zero is a value); no face line between `minus` and `plus` that no section owns; `minus`,
    E and every capital line a +1 calculation child of `plus` in every network, the networks agreeing. The section's
    parts are the other children; one explicitly nil for the period is omitted with its reason, one with no fact makes
    the reconstruction unavailable (it may be capital); a part the section does not own fails C6. Whether each
    capital line is placed, and only on a capital row, is checked after placement (post_checks); agreement with the other reconstructions certifies no row's classification."""
    t, cap = template.active(), alt["capital"]
    head = {"id": alt["id"]}
    printed = f.printed(role)
    names = [r["concept"] for r in printed]

    def refuse(why):
        return {**head, "refused": why}

    for c in (alt["plus"], alt["minus"]):
        if c not in names or not read(f, c, key):
            return refuse(f"{c} {'not printed' if c not in names else 'has no fact'} {period_words(key)} "
                          f"(own entity, no dimensions)")
    inclusive = cap["inclusive"] in names
    e = cap["inclusive"] if inclusive else cap["parent"]
    if e not in names or not read(f, e, key):
        return refuse(f"{e} {'not printed' if e not in names else 'has no fact'} {period_words(key)} "
                      f"(own entity, no dimensions)")
    n = cn.networks(f, role)
    inside = descendants(n, e)
    cap_section = t.section_of(cap["nci_row"])
    # a split piece is named in the calculation by its own key once corrections apply, else by its tag
    lines = [l for l in st.get(cap_section, {}).get("lines", []) if tag_of(l["concept"]) != e
             and l["concept"] not in inside and tag_of(l["concept"]) not in inside and not l.get("context")]
    if any("#" in l["concept"] for l in lines):
        return refuse("a capital line outside the equity total is printed by member; not a plain fact")
    caps = [l["concept"] for l in lines]
    i, j = names.index(alt["minus"]), len(names) - 1 - names[::-1].index(alt["plus"])
    loose = [c for c in names[i + 1:j] if c not in owner and read(f, c, key)]
    if i >= j or loose:
        return refuse("face lines between the two totals in no section: " + ", ".join(loose) if loose else
                      f"{alt['minus']} is not printed above {alt['plus']}")
    inputs = [alt["plus"], alt["minus"], e, *caps]
    got = {}
    for c in inputs:
        xs = read(f, c, key)
        if not xs:          # a capital line with no fact: missing is never zero
            return refuse(f"{c} has no fact {period_words(key)} (own entity, no dimensions)")
        if disagree(xs):
            return refuse(f"duplicate facts of {c} not consistent or in two currencies")
        got[c] = xs
    pick = {c: chosen(xs) for c, xs in got.items()}
    if len({x["currency"] for x in pick.values()}) != 1:
        return refuse("the inputs are not in one currency")
    answers = {ar: x for ar, x in cn.children(n, alt["plus"]).items() if x["answer"] != "absent"}
    need = [alt["minus"], e, *caps]
    lacking = [ar for ar, x in answers.items() if x["answer"] != "supported"
               or any((c, Decimal(1)) not in x["parts"] for c in need)]
    if not answers or lacking or len({frozenset(x["parts"]) for x in answers.values()}) > 1:
        return refuse(f"{', '.join(need)} not each a +1 calculation child of {alt['plus']} in every calculation "
                      "network, the networks agreeing")
    kids = [(c, Decimal(w)) for c, w in next(iter(answers.values()))["parts"] if c not in need]
    mixed = [c for c, _ in kids if mixed_nil(f, c, key)]
    if mixed:
        return refuse(f"a calculation part has nil and valued facts for the period: {', '.join(mixed)}")
    gone = [(c, w, period_absence(f, role, c, key)) for c, w in kids if c not in owner]
    # missing is never zero here: only an explicitly nil part is left out; one with no fact may be capital
    unknown = [c for c, _, why in gone if why == "no fact for the period"]
    if unknown:
        return refuse(f"a calculation part of {alt['plus']} has no fact for the period, so the capital side cannot "
                      f"be shown complete: {', '.join(unknown)}")
    gone = [g for g in gone if g[2]]
    coef = {alt["plus"]: 1, alt["minus"]: -1, e: -1, **{c: -1 for c in caps}}
    amount = sum((Decimal(pick[c]["value"]) * k for c, k in coef.items()), Decimal(0))
    return {**head, "amount": amount, "unit": pick[alt["plus"]]["currency"],
            "allowance": sum((half_unit(pick[c]["decimals"]) for c in coef), Decimal(0)),
            "parts": [(c, w) for c, w in kids if c not in {g[0] for g in gone}],
            **({"omitted_parts": [[c, w, r] for c, w, r in gone]} if gone else {}),
            "equity": e, "equity_kind": "including noncontrolling interest" if inclusive else "parent",
            "capital_lines": caps,
            "capital_rows": list(cap["temporary_rows"]) + ([] if inclusive else [cap["nci_row"]]),
            "reconstructed": {"formula": " - ".join([alt["plus"], alt["minus"], e, *caps]), "coefficients": coef,
                              "facts": {c: pick[c]["id"] for c in coef},
                              "fact_ids": sorted(x["id"] for c in coef for x in got[c]),
                              "values": {c: pick[c]["value"] for c in coef},
                              "decimals": {c: pick[c]["decimals"] for c in coef}}}


def alternative_control(f, role, section, key, canonical, first, st, owner):
    """The section's control when its template lists further reconstructions (format v4): the first one available,
    in the template's order (the original reconstruction first); every reconstruction kept under `alternatives`,
    with why when unavailable. Two or more available are compared with each other (`agreement`, the two-sided
    tolerance over their canonical facts): a gap beyond it leaves the control disputed. A filed total printed is
    compared with the chosen one, as before. None when the template lists none."""
    t = template.active()
    alts = t.alternatives(section)
    if not alts or first is None:
        return first
    r1 = {"id": "R1", **({"refused": first["refused"]} if first.get("refused") else
                         {k: first[k] for k in ("amount", "unit", "allowance", "reconstructed")})}
    tried = [r1] + [capital_side(f, role, section, a, key, canonical, st, owner) for a in alts]
    ok = [a for a in tried if not a.get("refused")]
    base = {k: first[k] for k in ("concept", "control", "label", "filed_total_printed")}
    if not ok:
        return {**base, "parts": [], "alternatives": tried,
                "refused": "; ".join(f"{a['id']}: {a['refused']}" for a in tried)}
    pick = ok[0]
    out = dict(first) if pick["id"] == "R1" else {
        **base, "amount": pick["amount"], "unit": pick["unit"], "decimals": None, "allowance": pick["allowance"],
        "parts": pick["parts"], "reconstructed": pick["reconstructed"],
        **({"omitted_parts": pick["omitted_parts"]} if pick.get("omitted_parts") else {})}
    out["chosen"], out["alternatives"] = pick["id"], tried
    if len(ok) > 1:
        facts = lambda a: [(a["reconstructed"]["facts"][c], k, a["reconstructed"]["decimals"][c])
                           for c, k in a["reconstructed"]["coefficients"].items()]
        out["agreement"] = [{"between": [pick["id"], a["id"]], "gap": a["amount"] - pick["amount"],
                             "tolerance": tolerance(facts(pick), facts(a), canonical or {}),
                             "conflict": abs(a["amount"] - pick["amount"]) > tolerance(facts(pick), facts(a),
                                                                                       canonical or {})}
                            for a in ok[1:]]
    name = t.filed_total(section)
    if pick["id"] != "R1" and first["filed_total_printed"]:
        filed = read(f, name, key)
        if filed:
            gap = Decimal(filed[0]["value"]) - out["amount"]
            tol = tolerance([(x, k, d) for x, k, d in
                             ((pick["reconstructed"]["facts"][c], k, pick["reconstructed"]["decimals"][c])
                              for c, k in pick["reconstructed"]["coefficients"].items())],
                            [(filed[0]["id"], 1, filed[0]["decimals"])], canonical or {})
            out["filed_comparison"] = {"filed": filed[0]["value"], "filed_fact": filed[0]["id"],
                                       "reconstructed": out["amount"], "gap": gap, "tolerance": tol,
                                       "conflict": abs(gap) > tol}
    return out


POST_CHECKS = ("temporary equity counted once in the overall total",
               "capital-side reconstruction's capital lines on capital rows")


def post_checks(checks, claims, controls, canonical):
    """The checks after every section and the run-wide control ties (template format v4),
    in the spread run's tail and in full_pipeline.assemble alike. (1) A section whose comparison left lines
    out on its `outside_rows` passes only when the verifying control's tie ran and passed and its comparison counts
    each such line's canonical fact exactly once; else "temporary equity not verified in the overall total". (2) A
    capital-side reconstruction available for a section passes only when each of its capital lines is placed, and
    only on its capital rows. `controls`: section -> its control (resolve_control, alternative_control)."""
    t, out = template.active(), []
    canonical = canonical or {}
    for s in t.controls:
        o = t.outside_rows(s)
        rec = next((c for c in checks if c.get("section") == s and c.get("outside_the_total_on_allowed_rows")), None)
        if not o or rec is None:
            continue
        v = o["verified_by"]
        ties = next((c for c in checks if c.get("section") == v and c["check"] == f"{v} control ties"), None)
        net = defaultdict(Decimal)
        for x, k, _ in side(claims, t.controls[v]["compare"]):
            net[canonical.get(x, x)] += Decimal(k)
        counts = {}
        for line in rec["outside_the_total_on_allowed_rows"]:
            mine = {canonical.get(x, x) for c in claims if c["source"] == line
                    for x, _, _ in (c.get("facts") or [(c["fact"], 1, None)])}
            counts[line] = {x: str(net[x]) for x in sorted(mine)}
        once = all(v2 and all(abs(Decimal(k)) == 1 for k in v2.values()) for v2 in counts.values())
        ok = bool(ties and ties["ok"]) and once
        out.append({"check": POST_CHECKS[0], "section": s, "ok": ok, "verified_by": v,
                    "verifying_tie": None if ties is None else ties["ok"], "times_counted": counts,
                    **({} if ok else {"why": "temporary equity not verified in the overall total" +
                                      (f": {v} control ties did not run" if ties is None else
                                       f": {v} does not tie" if not ties["ok"] else
                                       f": not counted exactly once in {v}")})})
    for s, ctl in controls.items():
        for a in (ctl or {}).get("alternatives") or ():
            if a.get("refused") or a.get("capital_lines") is None:
                continue
            placed = {l: sorted({c["row"] for c in claims if c["source"] == l}) for l in a["capital_lines"]}
            bad = {l: r for l, r in placed.items() if not r or not set(r) <= set(a["capital_rows"])}
            out.append({"check": POST_CHECKS[1], "section": s, "ok": not bad, "alternative": a["id"],
                        "capital_rows": a["capital_rows"], "placed": placed,
                        **({"why": "a capital line of the reconstruction is not placed, or not only on its capital "
                                   "rows", "lines": sorted(bad)} if bad else {})})
    return out


def side(claims, expr, signs=None, decimals=None):
    """The original facts an expression over rows is built from: [(fact, coefficient, decimals)], the template's
    coefficient of each row times each claim's (build). A claim made without its facts (repair's drafts, test
    fixtures) counts as its own fact times the line's sign."""
    k = template.active().coefficients(expr)
    return [(x, k[c["row"]] * coef, dec) for c in claims if c["row"] in k
            for x, coef, dec in c.get("facts") or [(c["fact"], (signs or {}).get(c["source"], 1),
                                                    c.get("decimals", (decimals or {}).get(c["fact"])))]]


def tolerance(built, filed, canonical):
    """Two-sided rounding (C6, the local controls too): each original fact, aliases merged into the canonical fact
    first, gets one net coefficient, the built side's minus the filed side's; T = sum of |coefficient| x half a unit
    of the fact's decimals. A fact on both sides with the same coefficient adds nothing."""
    net, dec = defaultdict(Decimal), {}
    for facts, s in ((built, 1), (filed, -1)):
        for x, coef, d in facts:
            x = canonical.get(x, x)
            net[x] += s * Decimal(coef)
            dec.setdefault(x, d)
    return sum((abs(v) * half_unit(dec[x]) for x, v in net.items()), Decimal(0))


def c6(lines, claims, ids, unresolved, signs, section, control=None, canonical=None, filed_absent=False,
       runwide=False):
    """The section's checks. `control` (resolve_control): a reconstructed total, used when the section's own total
    is not printed; its parts must be lines the section owns, the rows must add to it within the allowances of its
    leaves and of both filed totals, and none of its facts (canonical, duplicates included) may be a fact the rows
    are built from. A tie shows that the rows add up; it says nothing about whether each line is on the right row.
    A filed total that disagrees with the reconstructed one leaves the control disputed: the section fails.
    The tolerance is the two-sided rule (`tolerance`). A statement control line (C3; IS and CF) is never placed, so it
    is neither unallocated nor outside the total. `filed_absent` (IS and CF): the filed total has no fact for the
    period, so the control is reported unavailable by the run instead of failing here. `runwide` (IS and CF): the
    section's control is not a local filed total (it reads other sections' rows, or is reconstructed), so the run
    computes it after every section (control_states) and the section has no total of its own to find here.
    Negative rows are no check (C5)."""
    by = {l["concept"]: l for l in lines}
    dec = {l["fact"]: l["decimals"] for l in lines}
    rows = {r: sum((c["signed"] for c in claims if c["row"] == r), Decimal(0)) for r in ids}
    checks = []
    units = sorted({l["unit"] for l in lines})
    # a section with no line of its own (a run-wide IS or CF section; a balance sheet section the company does not
    # print, such as noncurrent assets on an unclassified balance sheet) shows no currency
    # conflict, not one currency
    checks.append({"check": "no currency conflict" if not units else "one currency in the section",
                   "ok": len(units) <= 1, "currencies": units})
    for l in lines:
        if l["control"]:
            away = [c for c, _ in l["parts"] if c not in by]
            parts = None if away else sum(w * by[c]["amount"] for c, w in l["parts"])
            checks.append({"check": f"{l['concept']} equals its printed parts", "ok": True,   # verified in mark_controls
                           "filed": l["amount"], "parts": parts,
                           **({"omitted_parts": l["omitted_parts"]} if l.get("omitted_parts") else {}),
                           **({"parts_in_another_section": away, "why": "the filing's arithmetic was verified before "
                               "the hand-off; those parts are now owned by another section"} if away else {})})
    for l in lines:     # a memo line answered without a row (run, MEMO_ANSWERED) has nothing to allocate
        if (not l["control"] and not l.get("statement_control") and not l.get("context")
                and l["concept"] not in unresolved and not l.get("memo_answered")):
            got = sum((c["amount"] for c in claims if c["source"] == l["concept"]), Decimal(0))
            if got != l["amount"]:
                checks.append({"check": f"{l['concept']} fully allocated", "ok": False,
                               "filed": l["amount"], "allocated": got})
    checks.append({"check": "every line allocated", "ok": not unresolved, "lines": sorted(unresolved)})
    controls = [l for l in lines if l["control"]]
    inner = {c for l in controls for c, _ in l["parts"]}
    tops = [l for l in controls if l["concept"] not in inner]
    t = template.active()
    name = t.filed_total(section)
    compare = t.controls[section]["compare"] if section in t.controls else None
    top = by.get(name)
    cmp = (control or {}).get("filed_comparison")
    if cmp:
        checks.append({"check": "filed and reconstructed totals agree", "ok": not cmp["conflict"], **cmp,
                       "why": "a disagreement leaves the control disputed" if cmp["conflict"] else None})
    for a in (control or {}).get("agreement") or ():     # format v4: two reconstructions available
        checks.append({"check": "reconstructions agree", "ok": not a["conflict"], **a,
                       "why": "a disagreement leaves the control disputed" if a["conflict"] else None})
    if top is None and control is not None and not control["filed_total_printed"]:
        checks += reconstructed_checks(by, rows, claims, signs, control, canonical or {}, compare)
    elif runwide or (top is None and filed_absent):
        pass
    elif top is None or not top["control"] or top not in tops:
        why = ("no section total listed for the section" if name is None else
               f"{name} not printed in the section" if top is None else
               f"{name} is not a control whose printed parts add up" if not top["control"] else
               f"{name} sits inside another control")
        checks.append({"check": "section total found", "ok": False, "total": name, "why": why,
                       "roots": [l["concept"] for l in tops]})
    else:
        lv = leaves(by, top["concept"])
        missing = sorted(set(absent(by, top["parts"])))
        outside = [l["concept"] for l in lines if not l["control"] and not l.get("statement_control")
                   and not l.get("context") and l["concept"] not in {c for c, _ in lv}]
        o, allowed = t.outside_rows(section), []
        if o:       # format v4: verified after the run-wide control ties (post_checks)
            allowed = [c for c in outside if {x["row"] for x in claims if x["source"] == c} and
                       {x["row"] for x in claims if x["source"] == c} <= set(o["rows"])]
            outside = [c for c in outside if c not in allowed]
        d = signs[top["concept"]]
        tol = tolerance(side(claims, compare, signs, dec), [(top["fact"], d or 0, top["decimals"])], canonical or {})
        built, unavailable = t.evaluate(compare, rows)
        target = None if d is None else top["amount"] * d
        gap = None if built is None or target is None else target - built
        checks.append({"check": f"rows add to {top['concept']}",
                       "ok": gap is not None and abs(gap) <= tol and not outside and not missing,
                       "filed": target, "built": built, "gap": gap, "tolerance": tol,
                       "lines_outside_the_total": outside,
                       **({"outside_the_total_on_allowed_rows": allowed, "verified_by": o["verified_by"]}
                          if allowed else {}),
                       **({"unavailable": unavailable, "why": "the template's comparison names rows the section "
                           "does not offer"} if unavailable else {}),
                       **({"missing_parts": missing, "why": "parts of the filed total owned by another section"}
                          if missing else {})})
        # a weight of -1 inside the total must match a sign of -1 on the line, or the rows cannot add
        bad = [] if d is None else [c for c, w in lv if w * d != signs[c]]
        if bad:
            checks.append({"check": "calculation weights agree with signs", "ok": False, "lines": bad})
    return rows, checks


def reconstructed_checks(by, rows, claims, signs, control, canonical, compare):
    """C6 against a reconstructed total (see c6). `compare`: the template's expression the rows are compared by."""
    if control.get("refused"):
        return [{"check": "section total found", "ok": False, "total": control["concept"],
                 "why": f"no filed total and none reconstructed: {control['refused']}"}]
    missing = sorted(set(absent(by, control["parts"])))
    checks = [{"check": "section total found", "ok": not missing, "total": control["concept"],
               "label": control["label"], "reconstructed": control["reconstructed"], "amount": control["amount"],
               **({"why": "calculation parts of the total not owned by the section", "missing_parts": missing}
                  if missing else {})}]
    lv = []
    for c, w in control["parts"]:
        if c in by:
            lv += [(x, w * v) for x, v in leaves(by, c)] if by[c]["control"] else [(c, w)]
    inside = {c for c, _ in lv}
    outside = [c for c, l in by.items() if not l["control"] and not l.get("context") and c not in inside]
    r = control["reconstructed"]
    tol = tolerance(side(claims, compare, signs, {l["fact"]: l["decimals"] for l in by.values()}),
                    [(r["facts"][c], signs[control["concept"]] * k, r["decimals"][c]) for c, k in
                     r["coefficients"].items()], canonical)
    built, unavailable = template.active().evaluate(compare, rows)
    target = control["amount"] * signs[control["concept"]]
    gap = None if built is None else target - built
    checks.append({"check": f"rows add to {control['concept']}", "ok": gap is not None and abs(gap) <= tol
                   and not outside and not missing, "filed": target, "built": built, "gap": gap, "tolerance": tol,
                   "lines_outside_the_total": outside, "label": "a tie shows the rows add up, not that each line "
                   "is on the right row",
                   **({"unavailable": unavailable, "why": "the template's comparison names rows the section "
                       "does not offer"} if unavailable else {})})
    bad = [c for c, w in lv if w * signs[control["concept"]] != signs[c]]
    if bad:
        checks.append({"check": "calculation weights agree with signs", "ok": False, "lines": bad})
    used = {canonical.get(c["fact"], c["fact"]) for c in claims}
    mine = {canonical.get(x, x) for x in control["reconstructed"]["fact_ids"]}
    checks.append({"check": "reconstructed total independent of the rows' facts", "ok": not used & mine,
                   "shared": sorted(used & mine)})
    return checks


def computed_totals(rows, incomplete):
    """The template's subtotals, computed from the rows by its recipes and kept apart from the claims and the controls.
    A row the run did not build, or any row of a section whose lines are not all allocated (`incomplete`: section ->
    what is open), is unavailable, never zero, and so is every total over it. A row of a fully allocated section
    that no line reached is a real zero. Only the totals of the statements the run built are written."""
    t = template.active()
    known = {r: (None if t.section_of(r) in incomplete else v) for r, v in rows.items()}
    statements = {t.statement_of(t.section_of(r)) for r in rows}
    out = {}
    for rid, recipe in t.recipes.items():
        if t.statement_of(t.section_of(rid)) not in statements:
            continue
        value, missing = t.evaluate(recipe, known)
        out[rid] = {"value": value, "unavailable": missing,
                    "incomplete_sections": {s: incomplete[s] for s in sorted({t.section_of(m) for m in missing})
                                            if s in incomplete},
                    "label": "computed from the rows by the template's recipe"}
    return out


def section_unknown(incomplete, offered, unknown, positive=False):
    """C6 propagation: an unresolved item whose section is unknown (`unknown`: concept -> why) makes every offered
    section incomplete, so no control or total over it ties; the coverage check fails. Returns that check when it
    fails. When nothing is unknown it returns the check passing only for a new execution: an analyst version
    (`positive`) or a run on a template of format v4 or later, so a closure can see it run and pass. A historical record keeps no passing record and reproduces byte for byte; there, an absent
    check stays absent and is never read as passed."""
    for c in sorted(unknown):
        for s in offered:
            incomplete.setdefault(s, {"unallocated": [], "partly_allocated": []}).setdefault("section_unknown", []).append(
                f"unresolved {c}; section unknown, so it may belong here")
    if not unknown:
        positive = positive or template.format_at_least(4)
        return [{"check": "every line allocated", "ok": True, "section": "run", "lines": []}] if positive else []
    return [{"check": "every line allocated", "ok": False, "section": "run", "lines": sorted(unknown),
             "why": "unresolved lines whose section is unknown"}]


def filed_side(tax, f, key, terms, normal):
    """(value, [(fact, coefficient, decimals)], why): each filed concept read for the period (C1) times its
    coefficient and its direction (its balance against the normal balance, else +1); never a substitute concept."""
    value, facts = Decimal(0), []
    for x in terms:
        xs = read(f, x["concept"], key)
        if not xs:
            return None, [], f"filed total not in the filing ({x['concept']})"
        if disagree(xs):
            return None, [], f"filed facts of {x['concept']} not consistent"
        y = chosen(xs)
        a = ((sign(balance_of(tax, f, x["concept"]), normal) if normal else None) or 1) * Decimal(str(x["coef"]))
        facts.append((y["id"], a, y["decimals"]))
        value += a * Decimal(y["value"])
    return value, facts, None


def control_states(f, tax, key, statement, sections, rows, claims, checks, incomplete, canonical):
    """C6a: each control of the statement after every section of the run: tied, differs (with the gap) or
    unavailable (with the reason); never passed without a computed tie. The income and cash flow statements' local
    controls come from c6's check. The controls over other sections' rows (BS.TOT, IS.NOP's reconstruction, IS.TAX,
    IS.NI, IS.CMN, CF.REC) are computed here by the template's expression once the run holds every section it reads
    ("not evaluated" otherwise). A row of an incomplete section is unknown, and so is every total over it. The
    balance sheet's local controls stay c6's checks only.

    Scope of controls.json: it lists these states, so on a balance-sheet run it holds the run-wide controls only
    (BS.TOT); each balance-sheet section's own total is read in checks.json ("rows add to ..." and "section total
    found"), never in controls.json."""
    t = template.active()
    known = {r: (None if t.section_of(r) in incomplete else v) for r, v in rows.items()}
    instant = t.period(statement) == "instant"      # the balance sheet: its local controls stay c6's checks
    out = []
    for sid, c in t.controls.items():
        if t.statement_of(sid) != statement:
            continue
        normal = t.sections[sid]["normal_balance"]
        rec = {"section": sid, "filed": c["filed"], "kind": "local"}
        if t.local(sid) and (c["filed"] or instant):
            if instant or sid not in sections:
                continue
            chk = next((x for x in checks if x["section"] == sid and x["check"] == f"rows add to {c['filed']}"), None)
            _, _, why = filed_side(tax, f, key, [{"concept": c["filed"], "coef": 1}], normal)
            if why:
                out.append({**rec, "state": f"unavailable: {why}"})
            elif sid in incomplete:
                out.append({**rec, "state": "unavailable: incomplete input", "incomplete": incomplete[sid]})
            elif chk is None:
                out.append({**rec, "state": "failed", "why": "no section total to compare (see the section's checks)"})
            else:
                out.append({**rec, "state": "tied" if chk["ok"] else "differs",
                            **{k: chk[k] for k in ("built", "gap", "tolerance")}, "filed_value": chk["filed"]})
            continue
        rec["kind"] = "run-wide"
        need = sorted({t.section_of(r) for r in t.coefficients(c["compare"])} - set(sections))
        if need:
            out.append({**rec, "state": "not evaluated", "why": f"sections not in the run: {', '.join(need)}"})
            continue
        terms = [{"concept": c["filed"], "coef": 1}] if c["filed"] else c["reconstructed"]["terms"]
        if not c["filed"]:
            rec["label"] = "reconstructed, not a filed subtotal"
        value, filed, why = filed_side(tax, f, key, terms, normal)
        built, missing = t.evaluate(c["compare"], known)
        if why:
            out.append({**rec, "state": f"unavailable: {why}"})
        elif built is None:
            out.append({**rec, "state": "unavailable: incomplete input", "missing_rows": missing,
                        "incomplete": {s: incomplete[s] for s in sorted({t.section_of(m) for m in missing})
                                       if s in incomplete}})
        else:
            tol = tolerance(side(claims, c["compare"]), filed, canonical)
            gap = value - built
            out.append({**rec, "state": "tied" if abs(gap) <= tol else "differs", "filed_value": value, "built": built,
                        "gap": gap, "tolerance": tol, "facts": filed})
            found = sign_reversal(f, out[-1], t.coefficients(c["compare"]), claims)     # sign reversal: a diagnostic
            if found:
                out[-1]["sign_reversal"] = found
    return out


FILED_SIGN = "filed sign"   # sign reversal: shown in the exceptions, never counted in the outcome


def _printed_row_of(f, fid):
    """(row, why): the first printed row of the verified main document holding fact `fid`, as page.printed_row reads
    it, with "rows" the number of rows holding it; or (None, the reason)."""
    from . import page
    doc, why = page.main_document(f.dir)
    if doc is None:
        return None, "no verified main document"
    tables, _ = page.parsed(str(doc))
    found = [r for t in tables for r in t if any(fid in c["facts"] for c in r)]
    row = next((x for x in map(page.printed_row, found) if x and any(fid in c["facts"] for c in x["cells"])), None)
    if row is None:
        return None, "fact on no printed row"
    return {**row, "rows": len(found)}, None


def _amount(fact):
    """A fact's value as a Decimal; None for a nil or empty value (filing.Filing keeps those as '')."""
    if fact.get("nil") or not str(fact.get("value", "")).strip():
        return None
    return Decimal(fact["value"])


_NUM = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"     # one amount: digits, thousands commas, decimals
# a dollar sign and spaces only where a printed amount has them, so "(54.1 $1)" is not one amount
_SIGN_FORMS = (("parentheses", rf"\$? ?\( ?\$? ?{_NUM} ?\)"),
               ("minus", rf"\$? ?[-\u2212\u2013] ?\$? ?{_NUM}"),
               ("plain", rf"\$? ?{_NUM}"))


def _printed_form(text):
    """The sign form of a printed amount: "parentheses", "minus" or "plain"; None when the text is not one amount
    in one of those forms (a dash, a note, a footnote mark)."""
    t = " ".join(str(text).split())
    for form, pattern in _SIGN_FORMS:
        if re.fullmatch(pattern, t):
            return form
    return None


def sign_reversal(f, state, coefs, claims):
    """The printed-row check of a sign reversal: for a run-wide control that differs, the one claim whose filed
    fact is tagged negative and whose reversal alone would tie the control, checked against the printed row; None
    when no claim, or more than one, explains the gap. Code only; nothing here changes a state, value or check.
    The claim must represent the filed fact alone (a whole line, one original fact, its amount the filed value)."""
    if state.get("kind") != "run-wide" or state.get("state") != "differs":
        return None
    built, filed, tol = state["built"], state["filed_value"], state["tolerance"]
    hits = []
    for c in claims:     # every claim, pieces included: the one explaining claim must be unique among them all
        if c.get("row") not in coefs:
            continue
        x = coefs[c["row"]] * Decimal(str(c["signed"]))
        if abs(filed - (built - 2 * x)) <= tol:
            hits.append((c, x))
    if len(hits) != 1:
        return None
    c, x = hits[0]
    if "r" in c:     # a piece (six_rules: its direction "r") is never a whole line
        return None
    facts = {y["id"]: y for y in f.facts if y.get("id")}
    me = facts.get(c.get("fact"))
    if me is None or _amount(me) is None or _amount(me) >= 0:
        return None
    whole = (c.get("source") == c.get("concept") and len(c.get("facts") or []) == 1
             and (c["facts"][0][0] if c["facts"] else None) == me["id"]
             and Decimal(str(c["amount"])) == Decimal(me["value"]))
    if not whole:
        return None
    out = {"line": c["concept"], "fact": me["id"], "amount": Decimal(me["value"]), "contribution": x,
           "built_reversed": built - 2 * x, "printed_row": None, "corroborated": False,
           "calculation_tie": "not checked (not built)"}
    row, why = _printed_row_of(f, me["id"])
    if row is None:
        out["why"] = why
        return out
    ctx = f.contexts
    mine = ctx[me["context"]]

    def period(k):
        return k.get("instant"), k.get("start"), k.get("end")
    cell = {fid: cl["text"] for cl in row["cells"] for fid in cl["facts"]}
    comps = [y for fid in cell for y in [facts.get(fid)] if y and y["id"] != me["id"]
             and y["concept"] == me["concept"] and y.get("unit") == me.get("unit")
             and ctx[y["context"]]["entity"] == mine["entity"] and ctx[y["context"]]["dims"] == mine["dims"]
             and period(ctx[y["context"]]) != period(mine) and _amount(y) not in (None, 0)]     # nil: no comparable
    out["printed_row"] = {"label": row["label"], "cells": [cl["text"] for cl in row["cells"]],
                          "facts": [fid for cl in row["cells"] for fid in cl["facts"]], "rows_holding": row["rows"],
                          "comparables": [y["id"] for y in comps],
                          "compared": [cell[me["id"]]] + [cell[y["id"]] for y in comps]}     # the cells compared
    neg = [y["id"] for y in comps if _amount(y) < 0]
    form = _printed_form(cell[me["id"]])
    if not comps:
        out["why"] = "no comparable from another period on the row"
    elif neg and len(neg) < len(comps):
        out["why"] = f"comparables of mixed sign ({', '.join(neg)} negative)"
    elif neg:
        out["why"] = "comparables all negative"
    elif form is None or any(_printed_form(cell[y["id"]]) is None for y in comps):
        out["why"] = "a printed form not recognized"
    elif any(_printed_form(cell[y["id"]]) != form for y in comps):
        out["why"] = "printed forms differ"
    else:
        out["corroborated"] = True
    return out


def sign_exceptions(states, lines, f):
    """Sign reversal: one "filed sign" exception per corroborated sign_reversal on a control state; never counted (counted)."""
    label = {l["concept"]: l.get("label") for l in lines or ()}
    out = []
    for x in states:
        r = x.get("sign_reversal")
        if not r or not r.get("corroborated"):
            continue
        p = r["printed_row"]
        comps = p["comparables"]
        out.append({"step": "C6", "class": FILED_SIGN, "section": x["section"], "concept": r["line"],
                    "fact": r["fact"],
                    "why": f"the filed sign of '{label.get(r['line']) or f.label(r['line'])}' ({r['line']}, "
                           f"{r['fact']}, {r['amount']}) contradicts the printed statement: its printed row "
                           f"'{p['label']}' prints {', '.join(p['compared'])} in the same form, yet only {r['fact']} is "
                           f"tagged negative ({', '.join(comps)} positive); with its sign reversed, {x['section']} "
                           f"would tie (built {r['built_reversed']}, filed {x['filed_value']}). The filed value is "
                           f"kept."})
    return out


def report_checks(f, key, statement, sections, rows, claims, incomplete, canonical, disputed, variant=None):
    """C6b: the template's report checks of the statement's period, reported beside the spread, never among its
    checks or in its verdict. A run computes one when every row it names is a row of the statement and every spread
    section of the statement that applies to the filing's `variant` (all, when None) is in the run: CF.DIR does not
    apply to an indirect filer, so its absence does not stop a check."""
    t = template.active()
    known = {r: (None if t.section_of(r) in incomplete else v) for r, v in rows.items()}
    spread = {s for s, x in t.sections.items() if x["kind"] == "spread" and t.statement_of(s) == statement
              and (variant is None or variant in x["variants"])}
    out = []
    for rc in t.report_checks:
        if rc["period"] != t.period(statement):
            continue
        rec = {k: rc[k] for k in ("id", "name", "kind", "note")}
        named = {r for k in ("left", "right") for r in t.coefficients(rc[k])}
        if rc["deferred"]:
            out.append({**rec, "state": f"not done (deferred, {rc['deferred']})"})
            continue
        if spread - set(sections) or any(t.statement_of(t.section_of(r)) != statement for r in named):
            out.append({**rec, "state": "not done", "why": "the run does not hold every section of the statement"})
            continue
        got, why = {}, None
        for k in ("left", "right"):
            e = rc[k]
            if set(e) == {"concept"}:
                # the concept named; when it is not in the filing, the template's other printed totals for it, in
                # order (format v6). A named total whose facts disagree is never replaced.
                alts = (rc.get("alternatives") or {}).get(e["concept"], [])
                for c in [e["concept"]] + alts:
                    v, facts, w = filed_side(None, f, key, [{"concept": c, "coef": 1}], None)
                    if v is not None or not w.startswith("filed total not in the filing"):
                        break
                if v is not None and c != e["concept"]:
                    rec.setdefault("read_as", {})[k] = c
                elif v is None and alts and c == alts[-1] and w.startswith("filed total not in the filing"):
                    w = f"filed total not in the filing ({e['concept']}, nor {', '.join(alts)})"
            else:
                (v, m), facts = t.evaluate(e, known), side(claims, e)
                w = f"an input row unavailable: {', '.join(m)}" if m else None
            why = why or w
            got[k] = {"value": v, "facts": facts, "rows": sorted(t.coefficients(e))}
        if why:
            out.append({**rec, "state": "unavailable", "why": why, **got})
        elif disputed & named:
            out.append({**rec, "state": "disputed", "rows": sorted(disputed & named), **got})
        else:
            gap = got["left"]["value"] - got["right"]["value"]
            tol = tolerance(got["left"]["facts"], got["right"]["facts"], canonical)
            out.append({**rec, "state": "tied" if abs(gap) <= tol else "differs", "gap": gap, "tolerance": tol, **got})
    return out


def gap_lines(lines, unresolved, gap, tol, k=2):
    """Diagnostic only (C7 off): unresolved lines whose amounts, alone or in pairs, match the gap."""
    free = [l for l in lines if l["concept"] in unresolved]
    for n in range(1, k + 1):
        for combo in itertools.combinations(free, n):
            if abs(sum(l["amount"] for l in combo) - abs(gap)) <= tol:
                return [l["concept"] for l in combo]
    return []


B_LABEL = "an amount match can be coincidental; not the omitted line"


def gap_matches(f, role, key, filed, unit, candidates, gap, tol, k=2):
    """C4b's B: a record beside a failed control, never a repair. Every single and pair of the candidates (lines both
    models put OUTSIDE) is compared with the gap g. A candidate without a fact at the key or in another currency is
    excluded and counted. When each line of a combination has one weight w from the filed concept (usable
    relationships of one network, calc_network.reach_down; all paths giving the same product), the match is signed:
    |sum w x amount - g| <= T. Otherwise it is an absolute-magnitude match: ||sum amount| - |g|| <= T. The line's
    direction d is never applied. `gap`: filed - built in the filed concept's own arithmetic (the caller converts the
    check's gap). No effect on any check's ok, gap, verdict or coverage."""
    n = cn.networks(f, role)
    ws = defaultdict(set)
    for x in cn.reach_down(n, filed).values():
        for c, chains in x["reach"].items():
            for ch in chains:
                w = Decimal(1)
                for s in ch:
                    w *= s["weight"]
                ws[c].add(w)
    usable, excluded = [], []
    for l in candidates:
        xs = read(f, l["concept"], key)
        why = ("no fact at the key" if not xs else f"in {xs[0]['currency']}, the control in {unit}"
               if xs[0]["currency"] != unit else None)
        if why:
            excluded.append({"line": l["concept"], "why": why})
        else:
            usable.append(l)
    found, executed = [], 0
    for size in range(1, k + 1):
        for combo in itertools.combinations(usable, size):
            executed += 1
            one = [ws[l["concept"]] for l in combo]
            if all(len(w) == 1 for w in one):
                e = sum(next(iter(w)) * l["amount"] for w, l in zip(one, combo))
                hit, match = abs(e - gap) <= tol, "signed"
            else:
                e = sum(l["amount"] for l in combo)
                hit, match = abs(abs(e) - abs(gap)) <= tol, "absolute-magnitude match"
            if hit:
                found.append({"lines": [l["concept"] for l in combo], "match": match, "e": e, "label": B_LABEL})
    return {"filed": filed, "gap": gap, "tolerance": tol, "comparisons_executed": executed,
            "candidates": [l["concept"] for l in usable], "excluded": excluded, "gap_match_candidates": found}


# ---- the placement preparation: everything before the first model call ---------------------------------------------

def taken_rows(t):
    """The rows some total of the template takes: every row a relationship (recipe) or a control's comparison names.
    A queued line is offered only these: read from the template's own structure, this leaves out the beginning and
    ending cash rows, which no total takes and which are not spread. CF.DIR's rows are taken by CF.DIR's control;
    prepare drops that section for an indirect filer (C7)."""
    def rows_in(x):
        if isinstance(x, dict):
            return ({x["row"]} if "row" in x else set()).union(*map(rows_in, x.values()))
        if isinstance(x, list):
            return set().union(*map(rows_in, x))
        return set()
    return rows_in(list(t.recipes.values())) | rows_in([c["compare"] for c in t.controls.values()])
# the conservative application policy for prohibited and repeated pairs, recorded verbatim in the A walk
PROHIBITION = ("Keeping prohibited or repeated pairs unusable is a conservative application policy, not full XBRL "
               "relationship resolution. Under XBRL 2.1 (sections 3.5.3.9.7.3 to 3.5.3.9.7.5), prohibition and "
               "priority act on equivalent relationships within a base set. A highest-priority prohibition excludes "
               "them, a lower-priority prohibition need not, and equivalent duplicate arcs can resolve to one "
               "effective relationship. The policy can therefore discard a relationship that is in force. Rejecting "
               "edges cannot add graph reachability, so the policy itself adds no flag. That does not establish that "
               "every remaining path correctly represents the filing or the template's scope.")
BLOCKING = ("printed by member only", "incomplete subtotal", "extraction refused")


def setup(f, statement):
    """C1, the role and C7, before the taxonomy loads: (period key, role, variant). C7 asks each calculation network
    of the role separately whether a net-income starting item is under operating cash flow; the variant is then
    "indirect", else the statement's only variant (None when it has several and none is recognised)."""
    t = template.active()
    key = f.current(t.period(statement))     # C1: refused before C2 when ambiguous
    role = f.role(statement)
    if role is None:
        raise RuntimeError(f"C2: no {statement} role in FilingSummary: {f.found[statement]['why']}")
    start = t.indirect_start(statement)
    if start:
        if not any(under(net["kids"], *start) for net in cn.networks(f, role)["networks"].values()):
            raise RuntimeError("C7: indirect variant not recognised: no net-income starting item under operating "
                               f"cash flow; {statement} run refused")
        return key, role, "indirect"
    ids = [v["id"] for v in t.statements[statement]["variants"]]
    return key, role, ids[0] if len(ids) == 1 else None


# C4a's "excluded" reasons: the filing marks beginning and ending cash with the period start and end labels
# (preferredLabel); any other cash flow line at an instant is a balance the statement prints as a disclosure
BEGIN_END = {"periodStartLabel": "start", "periodEndLabel": "end"}
EXCLUDED_CASH = "beginning and ending cash are not spread"
EXCLUDED_BALANCE = ("a balance at a date printed on the cash flow statement, not a flow for the period; the cash "
                    "flow statement spreads flows only")


def excluded(f, concept, printed_row, key, role):
    """An excluded cash flow line's record beside its label and path: the reason, read from the filing's preferred
    label, the statement's role, and the amount at the date the line prints (the period's start for a start label,
    else its end) with its fact, so the line keeps its source amount and location. A date without one consistent
    amount says so; nothing is chosen."""
    from datetime import date, timedelta
    which = BEGIN_END.get((printed_row.get("preferred") or "").rsplit("/", 1)[-1])
    if which == "start":            # filed at the start date or the day before it
        dates = [(date.fromisoformat(key[1]) - timedelta(1)).isoformat(), key[1]]
    else:
        dates = [key[-1]]
    when, facts = next(((d, x) for d in dates for x in [f.instant_facts(concept, d)] if x), (dates[0], []))
    out = {"reason": EXCLUDED_CASH if which else EXCLUDED_BALANCE, "role": role, "date": when}
    if facts and not disagree(facts):
        x = chosen(facts)
        out.update(fact=x["id"], amount=str(Decimal(x["value"])), unit=x["currency"])
    else:
        out["amount_note"] = ("duplicate facts not consistent or in two currencies; no amount chosen" if facts else
                              "no amount at this date")
    return out


def coverage(tax, f, role, key, statement, canonical, org=None):
    """C4a (income and cash flow statements): one state for every printed occurrence, the first that applies:
    structure, not a currency amount, excluded (the cash flow's instant concepts, not spread: beginning and ending
    cash, and any balance printed as a disclosure; each with its reason, amount and fact, `excluded`), statement
    control, period type differs, printed by member only, no amount for the period, second
    occurrence, extraction refused (duplicate facts not consistent; disagree), routed (with its section), verified
    subtotal, incomplete subtotal, queued. The two subtotal states come from mark_controls on copies of the
    statement-wide first occurrences with amounts; a refused line is not among them, so its amount is never used. A
    line printed by member only and a refused line carry their section when the structural route (route_of, as
    section_lines routes) names one. With `org`, an item's section is the organizer's, and a blank is queued, a
    subtotal too (an item of the frozen inventory)."""
    t = template.active()
    controls, period = t.statement_controls(statement), t.period(statement)
    headings, (reach, _) = t.heading_to_section(), calc_routes(f, role)
    memo = {s for s in set(headings.values()) if t.memo(s)}
    occ, seen, firsts = [], set(), []
    for r in f.printed(role):
        c = r["concept"]
        k = tax.concepts.get(tag_of(c)) or f.extension.get(tag_of(c)) or {}
        facts = read(f, c, key)
        e = {"concept": c, "label": r["label"], "path": r["path"]}
        # an unknown type (neither taxonomy nor extension) is judged by the facts read for it
        amount = (tax.is_amount(tag_of(c)) if tag_of(c) in tax.concepts else
                  (k.get("type") or "").endswith("monetaryItemType") if k.get("type") else bool(facts))
        state = ("structure" if k.get("abstract") else "not a currency amount" if not amount else
                 "excluded" if statement == "CF" and k.get("period") == "instant" else
                 "statement control" if c in controls else
                 "period type differs" if k.get("period") and k["period"] != period else
                 "printed by member only" if not facts and member_only(f, c, key, role) else
                 "no amount for the period" if not facts else "second occurrence" if c in seen else
                 "extraction refused" if disagree(facts) else None)
        if state == "excluded":     # format v5 on: the reason from the filing, with amount and fact (excluded());
            # earlier formats keep their recorded reason, so saved runs replay
            e.update(excluded(f, c, r, key, role) if template.format_at_least(5) else {"reason": EXCLUDED_CASH})
        if state == "extraction refused":
            e["reason"] = "duplicate facts not consistent or in two currencies; no amount chosen"
        if state is None or state == "statement control" and facts and c not in seen:
            x = chosen(facts)
            firsts.append({"concept": c, "fact": x["id"], "amount": Decimal(x["value"]),
                           "decimals": x["decimals"], "unit": x["currency"], "occurrence": e})
        if state in (None, "printed by member only", "extraction refused"):
            sec = org[c] if org is not None and c in org else route_of(r, headings, reach, memo)[0]
            state = state or ("routed" if sec else "queued" if org is not None and c in org else "unrouted")
            if sec:
                e["section"] = sec
        if facts:
            seen.add(c)
        e["state"] = state
        occ.append(e)
    copies = [dict(x) for x in firsts]
    mark_controls(f, role, copies, [], canonical, controls, key)
    for x in copies:
        e = x["occurrence"]
        if e["state"] != "unrouted":
            continue
        e["state"] = ("verified subtotal" if x["decomposition"] == "verified" else
                      "incomplete subtotal" if x["decomposition"] == "unavailable" else "queued")
        if e["state"] == "incomplete subtotal":
            e["reason"] = f"subtotal not covered by its printed parts: {x['reason']}"
    counts = defaultdict(int)
    for e in occ:
        counts[e["state"]] += 1
    return {"occurrences": occ, "counts": dict(counts)}


def by_concept(rec):
    from .organizer import by_concept
    return by_concept(rec)


def handed(rec, org, owner, queued, st, sections, instant, exc):
    """The organizer hand-off's code check and open items. Each item is in exactly one place: its section (owned, or
    refused there when its amount is refused), the statement queue (an income or cash flow blank), or unplaced with a
    permitted cause (a balance-sheet blank, a section this run does not take, a blank whose amount is refused); no
    context line is a placement candidate. Anything else is a code fault (ValueError). Open items: a balance-sheet
    blank, with its cause and the organizer's reason; a check still differing; the lines a tie cannot see; an
    unchecked section of the run (a memo section is outside every total and is not one)."""
    q = [x["line"]["concept"] for x in queued]
    for c in q + [l["concept"] for s in sections for l in st[s]["queue"]]:
        if c not in org:
            raise ValueError(f"organizer hand-off: {c} is context in the frozen inventory, not a placement candidate")
    srt, lines = rec["sort"], {l["id"]: l for l in rec["inventory"]["lines"]}
    refused = {l["concept"] for l in lines.values() if l["state"] == "extraction refused"}
    for c, s in org.items():
        got = ([("owned", owner[c])] if c in owner else []) + [("queued", None)] * q.count(c) \
            + [("refused", x) for x in sections if c in st[x]["refused"]]
        want = ([("refused" if c in refused else "owned", s)] if s in sections else
                [("queued", None)] if s is None and not instant and c not in refused else [])
        if got != want:
            raise ValueError(f"organizer hand-off: {c} assigned to {s}, found {got or 'nowhere'}, expected "
                             f"{want or 'unplaced'}")
    said = {x["line"]: x["reason"] for c in srt["calls"] if c["answer"] for x in c["answer"]["lines"]}
    for i, cause in srt["blank"].items():
        why = (srt["citation_invalid"].get(i) or {}).get("why")
        if instant:
            exc.append({"step": "C2", "class": "organizer", "concept": lines[i]["concept"],
                        "why": f"no section from the organizer ({cause}{': ' + why if why else ''}); not placed"
                               + (f"; its reason: {said[i]}" if i in said else "")})
    for k, v in srt["ties"].items():
        if v["state"] == "differs" and set(v["covers"]) & set(sections):
            exc.append({"step": "C2", "class": "organizer", "section": v["covers"][0],
                        "why": f"before placement, {k} still differs by {v['difference']} on the organizer's sort"})
    for kind, ids in srt["unseen"].items():
        for i in ids:
            if srt["assignment"].get(i) in sections:
                exc.append({"step": "C2", "class": "organizer", "section": srt["assignment"][i],
                            "concept": lines[i]["concept"], "why": f"a tie cannot see this line ({kind})"})
    for s, v in srt["sections"].items():
        if s in sections and not v["checked"] and not v["why"].startswith("not applicable: memo"):
            exc.append({"step": "C2", "class": "organizer", "section": s,
                        "why": f"not checked before placement: {v['why']}"})


def a_walk(f, role, key, cov, controls):
    """C4b's A: from each statement control printed with a fact at the key, each calculation network is walked down
    through unprinted concepts, statement controls and verified subtotals, stopping at any other printed concept
    (reached); cycles are cut and recorded; unusable pairs are recorded, never followed. A queued line reached on a
    usable path is flagged when it has a fact at the key in the root's currency: its amount is, by the filing's own
    calculation, inside a statement total, so "OUTSIDE" from both models conflicts with the coverage. Up to 20 paths
    are kept per line, with their count, weight product and arcrole; `arcs` is the role's arc count per arcrole, so
    a mixed role is visible. The second walk (`reachable_only_through_unusable_relationships`) is a record only."""
    printed = {r["concept"] for r in f.printed(role)}
    occ = cov["occurrences"]
    queued = {e["concept"] for e in occ if e["state"] == "queued"}
    through = set(controls) | {e["concept"] for e in occ if e["state"] == "verified subtotal"}
    n = cn.networks(f, role)
    rec = {"policy": PROHIBITION, "role_state": n["state"], "arcs": n["arcs"], "networks": {}, "roots": {}, "lines": {},
           "reachable_only_through_unusable_relationships": {}, "cuts": {}}
    roots = {}
    for c in sorted(set(controls) & printed):
        xs = read(f, c, key)
        rec["roots"][c] = "root" if xs else "no fact at the key"
        if xs:
            roots[c] = xs[0]["currency"]
    hits, flagged = defaultdict(list), defaultdict(set)
    for ar, net in sorted(n["networks"].items()):
        if not roots:
            rec["networks"][ar] = {"state": "no root with a fact at the key" if net["kids"] else
                                   "no admissible relationships"}
            continue
        reached = set()
        for root, unit in roots.items():
            x = cn.reach_down(n, root, lambda c: c in printed and c not in through)[ar]
            if x["cuts"]:
                rec["cuts"].setdefault(ar, {})[root] = [list(p) for p in x["cuts"]]
            for q, chains in x["reach"].items():
                if q in queued:
                    reached.add(q)
                    for ch in chains:
                        w = Decimal(1)
                        for s in ch:
                            w *= s["weight"]
                        hits[q].append({"root": root, "arcrole": ar,
                                        "path": [ch[0]["parent"]] + [s["child"] for s in ch],
                                        "weight": str(w), "unit": unit})
            for q in x["unusable_reach"]:
                if q in queued:
                    rec["reachable_only_through_unusable_relationships"].setdefault(q, []).append(
                        {"root": root, "arcrole": ar,
                         "unusable": sorted({(s["parent"], s["child"], s["unusable"]) for ch in x["unusable_reach"][q]
                                             for s in ch if "unusable" in s})})
        for q in reached:
            xs = read(f, q, key)
            if xs and any(h["arcrole"] == ar and h["unit"] == xs[0]["currency"] for h in hits[q]):
                flagged[q].add(ar)
        # a network whose every pair is unusable is still walked for the second walk's record
        rec["networks"][ar] = ({"state": "no admissible relationships"} if not net["kids"] else
                               {"state": "walked", "reached": len(reached),
                                "conflicts": sum(ar in v for v in flagged.values())})
    for q, hs in sorted(hits.items()):
        xs = read(f, q, key)
        why = (None if q in flagged else "reached, not evidence: no fact at the key" if not xs else
               f"reached, not evidence: in {xs[0]['currency']}, not the root's currency")
        rec["lines"][q] = {"flag": "A" if q in flagged else None, "path_count": len(hs), "paths": hs[:20],
                           **({"why": why} if why else {})}
    return rec


def prepare(f, tax, statement, sections, exc=None, ctx=None, organizer=None):
    """Everything a run decides before its first model call, side-effect free: no model, no Search, no embedding.
    Raises RuntimeError on a fatal C1, C2 or C7 refusal. `ctx`: setup's (key, role, variant), when the caller already
    has it.

    Returns {key, role, variant, sections (the applicable ones, template order kept), not_applicable {section: why},
    st {section: {lines, signs, ids, offered, unresolved, queue, control, balance, member, refused}}, alloc, owner,
    unrouted, canonical, controls, coverage (C4a; None on the balance sheet), queued [{line, sign, ids (every
    applicable spread section's rows of the statement), start None, why}], a_walk (C4b's A; None on the balance sheet),
    offered, printed, exceptions (those this preparation added, each tagged with its section where it has one)}.
    Placement candidates are each section's `queue` (routed lines for C4) and `queued` (the statement queue).

    What the models are offered, the same on every surface: `st[s]["offered"]` is the routed start's rows and what
    the section accepts as an answer (a hand-off's included); `offered` is rows_of's sections and rows. On an income
    or cash flow statement both hold only applicable sections and only rows some total takes (taken_rows: beginning
    and ending cash are left out), as the queued `ids` do; `ids` stays every row of the section for the checks and
    the output. On the balance sheet: every row, and rows_of offers every section of the statement. `printed`: the
    shared printed-line protection of both roles, the routed lines and the queued lines (each read with a fact at
    the key).

    `organizer`: the statement's verified record (organizer.load), read, never asked. A section's lines are then the
    items it assigned there; controls and verified subtotals keep code's route and stay in the arithmetic. A blank
    goes to the statement queue on an income or cash flow statement and stays an unplaced open item on the balance
    sheet (handed)."""
    t = template.active()
    org = None if organizer is None else by_concept(organizer)
    exc = [] if exc is None else exc
    first_exc = len(exc)
    key, role, variant = ctx or setup(f, statement)
    instant = key[0] == "instant"      # the balance sheet's run
    not_applicable = {s: f"a section of the {'/'.join(t.sections[s]['variants'])} variant; the filing's {statement} "
                         f"is {variant} (C7)" for s in sections if variant and variant not in t.sections[s]["variants"]}
    sections = [s for s in sections if s not in not_applicable]
    rows = load_rows()
    taken = None if instant else taken_rows(t)

    def tag(s, start):
        for x in exc[start:]:
            x.setdefault("section", s)

    # C2 and C3 per section; `owner` is the run-wide ownership registry: concept -> its section
    st, owner, unrouted, alloc = {}, {}, [], {}
    controls = t.statement_controls(statement)     # C3, C4a's state 4: never placed, the BS's too
    from .steps import Eligibility
    canonical = Eligibility(tax, f, {"fact": None}).canonical     # filing-wide duplicate facts merged (C6)
    for s in sections:
        start, got, member, refused = len(exc), [], [], []
        lines = section_lines(f, role, s, exc, got, key, member, refused, org)
        unrouted += [c for c in got if c not in unrouted]
        for l in list(lines):
            if l["concept"] in owner:
                exc.append({"step": "C2", "class": "extraction", "concept": l["concept"],
                            "why": f"already a line of {owner[l['concept']]}; not added twice"})
                lines.remove(l)
            else:
                owner[l["concept"]] = s
        mark_controls(f, role, lines, exc, canonical, controls, key)
        balance = t.sections[s]["normal_balance"]
        signs = {l["concept"]: sign(balance_of(tax, f, l["concept"]), balance) for l in lines}
        if not instant:       # C3a: from the company's calculation, else the balance; the BS is unchanged
            got = path_signs(tax, f, role, key, statement, lines)
            for l in lines:
                l["sign"] = got[l["concept"]]
                signs[l["concept"]] = l["sign"]["d"]
        control = resolve_control(f, role, s, key, canonical) if instant else None   # IS.NOP's is run-wide (C6a)
        if control and not control.get("refused"):
            signs[control["concept"]] = sign(balance_of(tax, f, t.derived_totals()[s]["plus"]), balance)
            if signs[control["concept"]] is None:
                exc.append({"step": "C3", "class": "mapping", "concept": control["concept"],
                            "why": "reconstructed total's sign unknown"})
                signs[control["concept"]] = 1
        ids = [r["row_id"] for r in section_rows(rows, s)]
        unresolved, queue = set(), []
        for l in lines:
            c = l["concept"]
            if c in controls:       # C3 (C4a's state 4): a statement control is never placed
                l["statement_control"] = True       # a BS total that does not add up stays a failed control
            if l["control"] or c in controls:
                if signs[c] is None and t.memo(s):  # a memo section ties to no total: a total's sign is not used
                    continue
                if signs[c] is None and instant:
                    exc.append({"step": "C3", "class": "mapping", "concept": c, "why": "total's sign unknown"})
                    signs[c] = 1
                elif signs[c] is None:      # IS and CF: never forced to +1; the section cannot tie
                    exc.append({"step": "C3", "class": "mapping", "concept": c,
                                "why": f"total's sign unresolved: {l['sign'].get('why')}"})
                    unresolved.add(c)
                continue
            if org is not None and c not in org:    # context in the frozen inventory, whatever the decomposition does
                l["context"] = True                 # after reassignment: never placed
                continue
            if signs[c] is None:
                exc.append({"step": "C3", "class": "mapping", "concept": c,
                            "why": l["sign"]["why"] if "sign" in l else "sign unknown (no balance)"})
                unresolved.add(c)
                continue
            queue.append(l)
        st[s] = {"lines": lines, "signs": signs, "ids": ids, "unresolved": unresolved, "queue": queue,
                 "offered": ids if instant or t.memo(s) else [r for r in ids if r in taken],
                 "control": control, "balance": balance, "member": member, "refused": refused}
        tag(s, start)
    for s in sections:      # format v4: further reconstructions read the other sections' lines (alternative_control)
        if instant and t.alternatives(s) and st[s]["control"] is not None:
            control = alternative_control(f, role, s, key, canonical, st[s]["control"], st, owner)
            st[s]["control"] = control
            if control.get("chosen") not in (None, "R1"):
                plus = next(a["plus"] for a in t.alternatives(s) if a["id"] == control["chosen"])
                st[s]["signs"][control["concept"]] = sign(balance_of(tax, f, plus), st[s]["balance"])
                if st[s]["signs"][control["concept"]] is None:
                    exc.append({"step": "C3", "class": "mapping", "concept": control["concept"], "section": s,
                                "why": "reconstructed total's sign unknown"})
                    st[s]["signs"][control["concept"]] = 1
    if not any(st[s]["lines"] for s in sections) and (org is None or instant):     # organizer: all to the queue
        raise RuntimeError(f"C2: no printed line found in {', '.join(sections)}")
    for s in sections:
        if not st[s]["lines"]:
            exc.append({"step": "C2", "class": "extraction", "section": s, "why": f"no printed line found in {s}"})
    cov = walk = None
    queued = []
    if not instant:
        cov = coverage(tax, f, role, key, statement, canonical, org)
        walk = a_walk(f, role, key, cov, controls)
        spread_ids = [r for s in sections if t.sections[s]["kind"] == "spread" for r in st[s]["offered"]]
        qs = [e for e in cov["occurrences"] if e["state"] == "queued"]
        lines = []
        for e in qs:
            x = read(f, e["concept"], key)[0]
            lines.append({"concept": e["concept"], "label": e["label"], "fact": x["id"], "unit": x["currency"],
                          "decimals": x["decimals"], "amount": Decimal(x["value"]), "path": e["path"], "route": None,
                          "period": key, "control": False, "parts": [], "control_candidate": False,
                          "decomposition": "none", **raw(f, e["concept"])})     # C4a: neither a verified nor an
                                                                                 # incomplete subtotal
        signs = path_signs(tax, f, role, key, statement, lines)
        blanks = organizer_blanks(organizer)
        # on a statement with no spread rows, an organizer "can't tell" is offered the memo rows
        memo_ids = [] if spread_ids else [r for s in sections if t.memo(s) for r in st[s]["offered"]]
        queued = [{"line": l, "sign": signs[l["concept"]], "ids": spread_ids,
                   "flag": walk["lines"].get(l["concept"], {}).get("flag"),
                   **({"organizer": blanks[l["concept"]]} if l["concept"] in blanks else {}),
                   **({"memo_ids": memo_ids} if memo_ids and (blanks.get(l["concept"]) or {}).get("cause")
                      == "organizer can't tell" else {})} for l in lines]
    # a memo section's rows are offered to its own lines only: never through rows_of
    offered = (statement_rows(rows, statement) if instant else
               {s: st[s]["offered"] for s in sections if not t.memo(s)})
    printed = [l["concept"] for s in sections for l in st[s]["lines"]] + [q["line"]["concept"] for q in queued]
    if org is not None:
        handed(organizer, org, owner, queued, st, sections, instant, exc)
    return {"key": key, "role": role, "variant": variant, "sections": sections, "not_applicable": not_applicable,
            "st": st, "alloc": alloc, "owner": owner, "unrouted": unrouted, "canonical": canonical,
            "controls": controls, "coverage": cov, "queued": queued, "a_walk": walk, "offered": offered,
            "printed": printed, "exceptions": exc[first_exc:]}


def og_none_cause():
    from .organizer import NONE_CAUSE
    return NONE_CAUSE


def organizer_blanks(rec):
    """Item concept -> {cause, reason} for each line the organizer's record leaves blank (its reason from the last
    answer that gave one); {} without a record."""
    if rec is None:
        return {}
    srt, lines = rec["sort"], {l["id"]: l for l in rec["inventory"]["lines"]}
    said = {x["line"]: x["reason"] for c in srt["calls"] if c.get("answer") for x in c["answer"]["lines"]}
    return {lines[i]["concept"]: {"cause": cause, "reason": said.get(i)} for i, cause in srt["blank"].items()}


def answer_of(rec):
    """A loop record's answer as the outcome table names it: a row, COMBINED, OUTSIDE or NOT_TOLD, or None when the
    loop did not finish (F) or was not run."""
    if rec.get("termination") not in FINISHED:
        return None
    return rec.get("row") if rec.get("decision") == "row" else rec.get("decision")


CHECKER_OFF = "checker off (--no-checker): not run"


def queued_outcome(q, p, rec8, canonical):
    """One queued line's row of C4b's outcome table: {state, resolved, why, answers[, consistency, row, section,
    sections, same_row_totals, coverage_conflict, detail]}. `p` is the analyst's loop record, `rec8` the checker's
    (its termination None when the caller stopped before it). Only an analyst row is committed (`row`; the caller
    commits it after this comparison), and it stands whatever the checker said: agree, or qualified as disputed
    (pieces, row, section, option) or unreviewed. Both models' OUTSIDE resolves an unflagged line, with no commit;
    with flag A it is a coverage conflict. Analyst OUTSIDE against anything else, COMBINED, NOT_TOLD and an
    unfinished analyst leave the line unresolved, its section unknown. With the checker off (`rec8`'s why
    CHECKER_OFF) an analyst row or OUTSIDE stands as "checker off" (OUTSIDE with flag A still a coverage conflict);
    COMBINED, NOT_TOLD and an unfinished analyst stay as above."""
    t, a, k = template.active(), answer_of(p), answer_of(rec8)
    off = rec8.get("why") == CHECKER_OFF        # the checker-off test: the analyst's answer stands, no comparison
    o = {"answers": {"analyst": a, "checker": k}, "resolved": False}
    if q.get("memo_only"):
        return memo_outcome(q, p, rec8, canonical, a, k, off, o)
    if a is None:
        return {**o, "state": "not compared: analyst unfinished",
                "why": f"the analyst did not finish ({p.get('termination')}: {p.get('why')})"}
    if a == "OUTSIDE":
        if (k == "OUTSIDE" or off) and q["flag"]:
            return {**o, "state": "unresolved", **({} if off else {"consistency": "agree"}),
                    "coverage_conflict": "A",
                    "why": ("checker off: the analyst put it OUTSIDE" if off else "both models put it OUTSIDE")
                           + ", but the filing's calculation places it inside a statement "
                           "total (A)"}
        if k == "OUTSIDE":
            return {**o, "state": "agree", "resolved": True, "consistency": "agree", "why": "both models: OUTSIDE"}
        if off:
            return {**o, "state": "checker off", "resolved": True, "why": "checker off: the analyst's OUTSIDE stands"}
        return {**o, "state": "unreviewed" if k is None else "disputed (option)",
                "why": "the checker did not finish" if k is None else f"the checker answered {k}"}
    if a not in q["ids"]:       # COMBINED, NOT_TOLD (the filing does not tell), or no offered row
        return {**o, "state": "unresolved", "why": f"the analyst answered {a}"}
    if off:
        return {**o, "state": "checker off", "resolved": True, "row": a, "section": t.section_of(a),
                "why": f"checker off: the analyst's {a} stands"}
    result, detail = compare(p, rec8, canonical)
    o.update(resolved=True, row=a, section=t.section_of(a), consistency=result, detail=detail)
    if result == "agree":
        return {**o, "state": "agree", "why": f"both models: {a}"}
    if result == "unreviewed":
        return {**o, "state": "unreviewed", "why": f"the checker did not finish; the analyst's {a} stands, unreviewed"}
    if k in OPTIONS:
        kind = "option"
    elif k == a:
        kind = "pieces"
        o["same_row_totals"] = detail["same_row_totals"]
    elif t.section_of(k) == t.section_of(a):
        kind = "row"
    else:
        kind = "section"
        o["sections"] = {"analyst": t.section_of(a), "checker": t.section_of(k)}
    return {**o, "state": f"disputed ({kind})",
            "why": f"the checker answered {k}; the analyst's {a} stands, disputed ({kind})"}


MEMO_GATE = "memo-only agreement gate"


def memo_outcome(q, p, rec8, canonical, a, k, off, o):
    """A queued line of a statement read for its memo rows only, which the organizer
    could not tell. Stricter than queued_outcome's rule, since the memo row has almost no code check downstream: a
    row is committed only when the analyst and the checker give the same complete placement (decomposition: the row
    and every piece by canonical fact; and the same amount on every row). Both OUTSIDE resolves it, no commit (with
    flag A a coverage conflict). Anything else (a disagreement, an unfinished model, the checker off) leaves it
    unresolved, nothing committed. An agreement gate, not confirmation: agreeing models can both be wrong."""
    t, o = template.active(), {**o, "gate": MEMO_GATE}
    if a is None or k is None or off:
        who = ("the checker is off" if off else "the analyst did not finish" if a is None else
               "the checker did not finish")
        return {**o, "state": "unresolved", "why": f"{MEMO_GATE}: {who}; nothing committed"}
    if a == k == "OUTSIDE":
        if q["flag"]:
            return {**o, "state": "unresolved", "consistency": "agree", "coverage_conflict": "A",
                    "why": "both models put it OUTSIDE, but the filing's calculation places it inside a statement "
                           "total (A)"}
        return {**o, "state": "agree", "resolved": True, "consistency": "agree", "why": "both models: OUTSIDE"}
    if a in q["ids"] and a == k:
        result, detail = compare(p, rec8, canonical)
        if result == "agree" and row_totals(p) == row_totals(rec8):
            return {**o, "state": "agree", "resolved": True, "row": a, "section": t.section_of(a),
                    "consistency": "agree", "detail": detail, "why": f"both models: {a}, the same placement"}
        return {**o, "state": "unresolved", "consistency": result, "detail": detail,
                "why": f"{MEMO_GATE}: both answered {a} with different pieces or amounts; nothing committed"}
    return {**o, "state": "unresolved",
            "why": f"{MEMO_GATE}: the analyst answered {a}, the checker {k}; nothing committed"}


def queue_counts(outcomes):
    """C4b's counters over the outcome table."""
    return {"placed by the analyst": sum(bool(o.get("committed")) for o in outcomes.values()),
            "unresolved: ownership conflict": sum(o["state"] == "ownership conflict" for o in outcomes.values()),
            "unresolved: analyst OUTSIDE disputed or unreviewed":
                sum(o["answers"]["analyst"] == "OUTSIDE" and not o["resolved"] and not o.get("coverage_conflict")
                    for o in outcomes.values() if "answers" in o),
            "unresolved: OUTSIDE/OUTSIDE coverage conflict (A)":
                sum(bool(o.get("coverage_conflict")) for o in outcomes.values())}


def coverage_check(cov, outcomes, unresolved_signs):
    """(check, unknown): "every printed amount accounted for" (C4a), failing on a line printed by member only, an
    incomplete subtotal, a refused extraction, an unresolved queued line or an unresolved sign. `unknown` (concept ->
    why) is what section_unknown propagates to every offered section: the blocking states whose section is unknown and
    the unresolved queued lines, once. A blocking line whose route names its section marks that section only,
    through section_lines' `member` and `refused` lists; the check still fails on it."""
    blocking = {e["concept"]: e for e in cov["occurrences"] if e["state"] in BLOCKING}
    unknown = {c: e["state"] for c, e in blocking.items() if not e.get("section")}
    unknown.update({c: o["why"] for c, o in outcomes.items() if not o["resolved"]})
    return {"check": "every printed amount accounted for",
            "ok": not blocking and not unknown and not unresolved_signs, "section": "run", "states": cov["counts"],
            **{k: sorted(c for c, e in blocking.items() if e["state"] == w) for k, w in
               zip(("printed_by_member_only", "incomplete_subtotals", "extraction_refused"), BLOCKING)},
            "section_known": {c: e["section"] for c, e in sorted(blocking.items()) if e.get("section")},
            "queued_unresolved": sorted(c for c, o in outcomes.items() if not o["resolved"]),
            "unresolved_signs": sorted(unresolved_signs)}, unknown


# ---- the run ----------------------------------------------------------------------------------------

# the local analyst and checker; repair and bench_caps keep their own settings
LOOP_MODELS = {"analyst": "qwen3.5:9b-q4_K_M", "checker": "granite4.2:8b-q4_K_M"}
COMBINED_PROMPT = "c4_place_v27"    # the combined-fixes option, only with --combined-fixes (off by default)
# the root-cause-fixes option, only with --root-cause-fixes on top of --combined-fixes (off by default)
ROOT_CAUSE_PROMPT = "c4_place_v28"
# the statements-only option, only with --statements-only on top of
# --combined-fixes (off by default): no note search, no breakdowns or other_periods (Tools statements_only)
STATEMENTS_PROMPT = "c4_place_v29"
NO_ROW = {"COMBINED": "the line is left unresolved", "NOT_TOLD": "the line is left unresolved",
          "OUTSIDE": "the line is not an amount of the rows shown"}     # the second look's no-row display
LOOP_PROMPT = "c4_place_v26"    # the only placement prompt: the six rules with tags hidden (line IDs, the printed
                                # page tool, no definition tool; tagless.py)
def hosted_use(*users):
    """Hosted tokens and attempts with unknown tokens over a run's distinct callers and embedders (the placement
    caller, organizer, embedder; a local one counts none)."""
    users = list({id(u): u for u in users if u is not None}.values())
    return (sum(getattr(u, "hosted_tokens", 0) for u in users), sum(getattr(u, "hosted_unknown", 0) for u in users))


LOOP_NUM_CTX = 8192
DURATION_NUM_CTX = 16384    # the context of an income or cash flow run: its queued starts offer every taken row of
                            # the statement with full definitions
# the default hosted profile: the analyst on Qwen3.8-27B, the checker on DeepSeek V4.1 Flash (a different family);
# each pinned to one provider endpoint with no fallback, thinking off, at most the listed price (USD per input, per
# output token). LOOP_NUM_CTX stays the context guard of both loops, so the requests match the local ones
HOSTED = {"analyst": {"model": "qwen/qwen3.8-27b", "providers": ["deepinfra/bf16"], "price": (0.15e-6, 1.875e-6)},
          "checker": {"model": "deepseek/deepseek-v4.1-flash-1017", "providers": ["deepinfra/fp8"],
                      "price": (0.098e-6, 0.294e-6)}}
# Haiku on OpenRouter: the tiebreaker's settings (THIRD, below, where they are explained) and the Haiku-Luna analyst's
THIRD_HAIKU = {"model": "anthropic/claude-haiku-5.5", "providers": ["anthropic"], "price": (0.10e-6, 0.50e-6),
               "omit": ["seed", "temperature", "presence_penalty", "frequency_penalty"],
               "reasoning": "off", "allowance": 0, "provider_names": {"anthropic": ["Anthropic"]},
               "quantization": "unknown"}
# the flash line-up: the analyst on Qwen3.8 Flash with reasoning off, the checker on GLM-5.3-Flash at effort low (its
# reasoning is mandatory). A role's max_tokens is the answer cap plus its reasoning allowance. Pins, the exact
# provider names a response may carry, prices (the listed price, sent as the ceiling) and the deployed precision come
# from the listed endpoints: Qwen on alibaba, a base-provider pin that can match its variants (not one
# region, no uptime guarantee); GLM on wafer/us, chosen for uptime. The checker's reasoning allowance is 1,000 tokens;
# its 1,600 max_tokens fit
# wafer/us's 131,072 max completion. main refuses a profile while any of these is None
HOSTED_FLASH ={"analyst": {"model": "qwen/qwen3.8-flash", "reasoning": "off", "allowance": 0,
                            "providers": ["alibaba"], "provider_names": {"alibaba": ["Alibaba"]},
                            "price": (0.15e-6, 0.47e-6), "quantization": "unknown"},
                "checker": {"model": "z-ai/glm-5.3-flash", "reasoning": "low", "allowance": 1000,
                            "providers": ["wafer/us"], "provider_names": {"wafer/us": ["Wafer"]},
                            "price": (0.044e-6, 0.2e-6), "quantization": "nvfp4"}}
# the hosted line-up for a machine with no GPU: the analyst on Claude Haiku 5.5 with reasoning off, the checker on
# GPT-6 Luna (a different family) at effort low; each pinned to its maker's endpoint with no fallback, at the listed
# price. Haiku omits the four fields its endpoint does not advertise; Luna omits temperature and keeps its seed.
# Luna's 4,096 allowance is provisional headroom, not measured. The tiebreaker ("haiku", the role's name; also the
# tree's critic) is DeepSeek V4.1 Flash, a third family, thinking off, on two providers, which allows fallback if one
# endpoint fails (a model served by one provider only returned 429s); the ceiling is the dearer endpoint's price
HOSTED_HAIKU_LUNA = {"analyst": {**THIRD_HAIKU},
                     "checker": {"model": "openai/gpt-6-luna", "reasoning": "low", "allowance": 4096,
                                 "providers": ["openai"], "provider_names": {"openai": ["OpenAI"]},
                                 "price": (0.10e-6, 0.50e-6), "omit": ["temperature"], "quantization": "unknown"},
                     "haiku": {"model": "deepseek/deepseek-v4.1-flash-1017", "reasoning": "off", "allowance": 0,
                               "providers": ["deepinfra/fp8", "novita/fp8"],
                               "provider_names": {"deepinfra/fp8": ["DeepInfra"], "novita/fp8": ["Novita"]},
                               "price": (0.165e-6, 0.660e-6), "quantization": "fp8"}}
HOSTED_PROFILES = {"hosted": HOSTED, "flash": HOSTED_FLASH,     # --hosted-profile; HOSTED stays the default
                   "haiku_luna": HOSTED_HAIKU_LUNA}
# the roles outside the spread's loop that run on a profile's model when the run is hosted (a hosted job has no
# local role): the organizer and named totals on the analyst's; the tree's proposer and meaning reader on the
# analyst's, its controller on the checker's, its critic on the tiebreaker's (repair_run.hosted_tree)
BORROWED = {"organizer": "analyst", "name": "analyst", "generator": "analyst", "meaning": "analyst",
            "controller": "checker", "critic": "haiku"}


def borrowed(profile, role, answer_cap, spend_usd):
    """The Caller settings of one role outside the loop on its profile's entry (BORROWED): {model, engine, ...}, for
    organizer.caller and named_totals.caller. Roles the profile cannot fill (no tiebreaker entry) fall back to the
    analyst's."""
    p = HOSTED_PROFILES[profile]
    v = p.get(BORROWED[role]) or p["analyst"]
    return {"model": v["model"], "engine": "openrouter", **openrouter_kwargs({role: v}, answer_cap, spend_usd)}
ROW_SEED = 20261029             # the default seed of the checker's row order, recorded per run
# the third opinion on a disputed line: Haiku on OpenRouter, pinned to Anthropic with no fallback, thinking off,
# without the four fields its endpoint does not advertise; experimental. Price per input, per output token. Beside a
# per-role profile (HOSTED_FLASH) or the local one, Haiku's request goes through the per-role path with reasoning
# "off" and allowance 0, byte for byte the scalar path's request; beside HOSTED, through the scalar path
THIRD = {"haiku": THIRD_HAIKU}
# the tiebreaker's sub-cap per company inside --max-calls and --spend-usd; $0.25 covers 80 calls of about 13,000
# tokens at the tiebreaker's price ceiling, a ceiling, not an estimate
THIRD_CAPS = {"haiku": {"calls": 80, "spend_usd": 0.25}}
SETTLED = "settled by vote (experimental)"
# the checker-feedback retry (--checker-feedback): the analyst's one retry on a line the blind checker answered differently, with both
# answers as candidates A and B. "Settled after a second look" is its own class: never agreement, never pooled with
# the blind result, never verified (the checker is not rerun, to bound calls)
SECOND_LOOK = "settled after a second look"
FEEDBACK_TEXT = ("Two answers were proposed for this line. They are not evidence. Check each one's cited facts and "
                 "quote with the tools. Then give the answer the filing supports: A, B or another. Either may be "
                 "right, and keeping one unchanged is fine when the evidence supports it. Say what decided it.")
FEEDBACK_LABEL = "a second look after feedback, not a blind review; agreement after it is not verification"


def per_role(hosted):
    """Whether every role of a hosted profile carries its own reasoning setting (the Caller's per-role reasoning)."""
    return bool(hosted) and all("reasoning" in v for v in hosted.values())


def provider_names(profile):
    """Pin -> the exact provider names a response may carry, when every role of the profile names them (the frozen
    listing's names); else None, and the Caller's default comparison applies."""
    if not all(v.get("provider_names") for v in profile.values()):
        return None
    return {p: list(n) for v in profile.values() for p, n in v["provider_names"].items()}


def openrouter_kwargs(profile, answer_cap, spend_usd):
    """The Caller settings of an OpenRouter profile, for the spread's caller and the tree's (repair_roles.Roles
    caller_kwargs): spend cap, prices, pins, per-role reasoning (or thinking off for all), omitted fields, Haiku's
    sub-cap, and the exact provider names when the profile has them."""
    names = provider_names(profile)
    return {"spend_cap_usd": spend_usd, "price": {r: v["price"] for r, v in profile.items()},
            "providers": {r: v["providers"] for r, v in profile.items()},
            **({"reasoning": {r: v["reasoning"] for r, v in profile.items()},
                "max_tokens": {r: answer_cap + v["allowance"] for r, v in profile.items()}}
               if per_role(profile) else {"max_tokens": answer_cap, "reasoning_off": True}),
            "omit": {r: v["omit"] for r, v in profile.items() if v.get("omit")},
            "role_caps": {r: THIRD_CAPS[r] for r in profile if r in THIRD_CAPS},
            **({"provider_names": names} if names else {})}

SPLIT_PENDING = "split pending C8 premise review"
FINISHED = ("finished", "finished_at_last_step")


# finished answers that settle a memo section's line without a row: outcome_of's classes
MEMO_ANSWERED = ("outside", "the filing does not tell", "combined")
# a loop that ended unfinished keeps its termination as its class; any other end is "not run"
UNFINISHED =("invalid", "no_progress", "cap_unfinished", "allocation_pending", "context_exceeded")
FEEDBACK_UNPLACED = ("outside", "the filing does not tell")     # the unplaced answers the checker-feedback retry reviews: not
                                                                # COMBINED, a referral or an unfinished loop


def outcome_of(rec, ids):
    """(row or None, the exception's class, why) for one loop record. Only a whole-line row of the section is a
    claim; everything else is an exception outcome, kept apart by kind."""
    ans = rec.get("answer") or {}
    if rec["decision"] == "escalated":
        return None, "escalated", ans.get("reason")
    if rec["termination"] not in FINISHED:
        kind = rec["termination"] if rec["termination"] in UNFINISHED else "not run"
        return None, kind, f"{rec['termination']}: {rec.get('why')}"
    if rec["decision"] == "NOT_TOLD":
        return None, "the filing does not tell", ans.get("reason")
    if rec["decision"] in ("COMBINED", "OUTSIDE"):
        return None, rec["decision"].lower(), ans.get("reason")
    if rec["decision"] == "referred":
        return None, "mapping", f"referred to {rec['target']}, a row of another section"
    if rec["row"] not in ids:
        return None, "mapping", f"answer {rec['row']} is not a row ({ans.get('reason')})"
    if ans.get("allocations"):      # a split in the older allocations form waits for C8's claim review
        return None, "split", SPLIT_PENDING
    return rec["row"], None, None       # a whole line, or a split whose pieces passed rules 5 and 6


def decomposition(rec, canonical):
    """The canonical decomposition of a finished answer: the remainder row (or the answer: COMBINED, OUTSIDE,
    NOT_TOLD, a row of another section), and every piece as (fact, row), aliases normalised to their canonical fact.
    Amounts are code's, kept apart for the row totals."""
    ans = rec.get("answer") or {}
    head = rec["target"] if rec.get("decision") == "referred" else rec.get("row")
    pieces = sorted((canonical.get(p["fact"], p["fact"]), p["row"]) for p in ans.get("_pieces") or ())
    return {"row": head, "pieces": pieces}


def row_totals(rec):
    """Row -> amount of a finished answer: the pieces to their rows, what is left to the answer's row."""
    ans, out = rec.get("answer") or {}, defaultdict(Decimal)
    for p in ans.get("_pieces") or ():
        out[p["row"]] += Decimal(p["amount"])
    left = ans.get("_remainder")
    out[rec.get("row")] += Decimal(rec["amount"]) if left is None else Decimal(left)
    return {r: v for r, v in out.items() if v}


def compare(p, rec8, canonical):
    """(result, record) for one line the analyst placed: "agree", "disputed" (a usable checker answer whose
    decomposition differs) or "unreviewed" (the checker's loop did not finish: capped, invalid, no progress, context,
    not run). Agreement is a consistency result, not correctness. A same-row-totals flag says when the decompositions
    differ but every row gets the same amount (reported apart)."""
    if rec8["termination"] not in FINISHED:
        return "unreviewed", {"termination": rec8["termination"], "why": rec8.get("why")}
    a, c = decomposition(p, canonical), decomposition(rec8, canonical)
    if a == c:
        return "agree", {"decomposition": a, "label": "consistency result, not correctness"}
    totals = row_totals(p) == row_totals(rec8) if rec8["decision"] not in OPTIONS else False
    return "disputed", {"analyst": a, "checker": c, "same_row_totals": totals,
                        "checker_only_pieces": sorted(set(map(tuple, c["pieces"])) - set(map(tuple, a["pieces"]))),
                        "analyst_only_pieces": sorted(set(map(tuple, a["pieces"])) - set(map(tuple, c["pieces"])))}


def line_event(d, rec):
    """Record only: one event for the live jobs page, appended as one line to line_events.jsonl in the run's folder.
    The run never reads it, and a failure to write it changes nothing."""
    try:
        with open(Path(d) / "line_events.jsonl", "a", encoding="utf8") as fh:
            fh.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **rec}, default=str) + "\n")
    except OSError:
        pass


def organize(f, tax, statement, d, out):
    """The organizer step: the whole statement sorted once, before prepare, every send charged to the run's ledger
    and logged in its calls.jsonl, and the record frozen beside the run. Returns the record as written, or None for
    a refused statement, with out["refusal"] {refused_at, kind, reason}: a statement-level failure
    (organizer.sort_statement), or a balance sheet none of whose lines got a section. A cap or the stop switch
    raises, as in every step."""
    from . import organizer as og
    inv = og.inventory(f, tax, statement)
    m = out.get("meta", {})
    hosted = borrowed(m["hosted_profile"], "organizer", og.ANSWER_CAP, m["limits"]["spend_usd"]) if m.get(
        "hosted_profile") else {}
    out["organizer"] = og.caller(d / "calls.jsonl", ledger=out["ledger"], company=out["company"], stop=out["stop"],
                                 **hosted)
    res = og.sort_statement(inv, "option1", out["organizer"], og.system_prompt())
    rec = og.save(d, inv, res)
    blank = template.active().period(statement) == "instant" and not any(res["assignment"].values())   # balance sheet
    why = res["refused"] or (blank and {"kind": "all blank", "reason": "no line got a section"})
    if why:
        out["refusal"] = {"refused_at": "organizer", **why}
        return None
    return rec


def run(a, d, out):
    """C2-C10 for one or more sections of one statement, with the analyst's loop and the six-rule prompt. C4 runs the
    capped tool loop on each line: one answer, a row and any pieces by fact ID, checked by code for rules 5 and 6; a
    split that passes commits with its pieces. C8, the checker, runs the same prompt and loop on each line C4 placed
    in a row of the section, blind (its own tools, memory, transcript and working registry of owned amounts; template
    rows in an order shuffled by --row-seed; never C4's answer, reason, queries or trace), and code compares the
    whole canonical decomposition.

    Outcome precedence: an answer C4 did not give, a row outside the section (COMBINED, OUTSIDE, NOT_TOLD, another
    section's row) or a failed numerical check leaves the line unallocated, and C6 blocks the section; a
    disagreement never overrides that. A placed line whose checker gave a usable, different answer stands as C4
    placed it, with a C8 "disputed" exception (the section is qualified); a placed line whose checker did not finish
    stands with a C8 "unreviewed" exception, counted apart. Agreement is a consistency result, not correctness. The
    automatic A3 splits stay off.

    The third opinion (--third-opinion): on a disputed line, the tiebreaker (the hosted profile's own or Haiku on
    OpenRouter, or the --third-model local model; recorded under the role name "haiku" either way) re-solves it
    blind with the same prompt and loop, its own registry and its own row order. Only finished answers vote,
    compared by canonical decomposition. The tiebreaker equal to the analyst: the placement stands, settled by vote.
    The tiebreaker equal to the checker on a row of this section: the checker's answer is re-checked by code (rules
    5 and 6) against the analyst's registry without this line's own claims and, if it passes, replaces the line's
    claims at once, before the next line; settled by vote. Any other case leaves the line disputed. A settled line
    is an exception of its own class, so the section stays qualified.

    The hand-off: sections run in template order, and one run-wide registry says which section owns each line. A
    line C4 refers to a row of another section of the run is pending there: the receiving section's analyst places
    it again, blind (the receiver's rows, tools and definitions; never the sender's row or answer), and its checker
    reviews it as any line of that section. Placed in a receiver row, the line moves to the receiver. Anything else,
    a second referral included (one transfer per line), leaves it an unresolved transfer in the sender, which then
    fails its checks. A referral to a section already run is placed in a late pass. A referral to a section outside
    the run stays a mapping exception. C5 and C6 run per section after every transfer, with a check that no amount
    is claimed in two sections."""
    from .analyst import run_c4
    from .runrec import search_use
    from .llm import HostedEmbedder, OllamaEmbedder
    from .search import Search
    from .steps import fact_index, phase_relations
    from . import open_claims, six_rules
    from .tools import Tools
    exc, caller = out["exceptions"], out["caller"]
    rows = load_rows()
    if len(set(a.section)) != len(a.section):
        raise RuntimeError(f"a section is named twice: {a.section}")
    t = template.active()
    if set(a.section) - set(t.sections):
        raise RuntimeError(f"not a section of the template: {sorted(set(a.section) - set(t.sections))}")
    statement = t.statement_of(a.section[0])     # referrals stay within the line's own statement
    if any(t.statement_of(s) != statement for s in a.section):
        raise RuntimeError(f"one statement per run: {a.section}")
    first = {}
    for i, r in enumerate(rows):
        first.setdefault(t.section_of(r["row_id"]), i)
    missing = [s for s in a.section if s not in first]
    if missing:
        raise RuntimeError(f"no template rows for {missing}")
    sections = sorted(a.section, key=first.get)          # template order
    from .corrections import open_filing
    f = open_filing(a.filing, out["meta"].get("corrections"))     # the job's corrections record, if it names one
    tagless.start(f)        # the models see line IDs, never the tags (tagless.py)
    ctx = setup(f, statement)       # C1, the role and C7 (per network), before the taxonomy loads
    key, role, variant = ctx
    out["meta"]["statement_found"] = f.found.get(statement)     # how C2 chose the report (template.find_statement)
    if (f.found.get(statement) or {}).get("how") == "fallback":   # the analyst sees the title fallback was used
        exc.append({"step": "C2", "class": "statement_fallback", "concept": None, "section": "run",
                    "why": f.found[statement]["why"]})
    for (r, new), old in f.printed_as.items():   # a retagged total stays an open item for the analyst's review
        if r == role:
            exc.append({"step": "corrections", "class": "retag", "concept": new, "original_concept": old,
                        "why": f"the filing tags this printed total {old}; the job's corrections record (sha256 "
                               f"{f.corrections_sha256}) reads its facts as {new}, under their own fact IDs"})
    for e in getattr(f, "disagreements", ()):     # the company's tag was kept; the model's other answer is shown
        if e.get("statement") == statement:
            exc.append({"step": "corrections", "class": NOTE_DISAGREES, "concept": e.get("concept"), "section": "run",
                        "why": f"named total {e.get('total')}: the company tags line {e.get('tagged_line')}; "
                               f"{e['outcome']}; the model's reason: {(e.get('answer') or {}).get('reason')} "
                               f"(a note for review; it does not change the outcome)"})
    instant = key[0] == "instant"      # the balance sheet's run
    year = taxonomy_year(f)
    whole = t.run_sections(statement, variant)
    if not instant and set(whole) - set(sections):   # the full-statement contract: a queued line is offered every
        # applicable spread section's rows, so a part of an income or cash flow statement is refused before any chat
        # or embedding call; the balance sheet keeps its per-section runs. A memo section with headings is part of
        # the statement too: without it, its lines would be routed to no run
        raise RuntimeError(f"an income or cash flow run takes every applicable spread and memo section of the statement; "
                           f"missing {[s for s in whole if s not in sections]}")
    out["meta"].update(period_end=f.period_end, taxonomy_year=year,
                       period={"kind": key[0], "start": key[1] if len(key) > 2 else None, "end": key[-1],
                               "length": period_length(key[1], key[2]) if len(key) > 2 else None})
    tax = Taxonomy(edition(year))     # MissingEdition names the fetch command

    def tag(s, start):
        for x in exc[start:]:
            x.setdefault("section", s)

    rec = organize(f, tax, statement, d, out)
    if rec is None:
        return
    # C2, C3 and C4a by the shared, side-effect-free preparation; a section of a variant the filing does
    # not use (CF.DIR for an indirect filer) is not applicable: recorded, not run
    prep = prepare(f, tax, statement, sections, exc, ctx, rec)
    out["diagnostics"]["not_applicable"] = prep["not_applicable"]
    sections = prep["sections"]
    if not sections:
        raise RuntimeError(f"C7: no section of the run applies to this filing: {prep['not_applicable']}")
    st, owner, unrouted, alloc = prep["st"], prep["owner"], prep["unrouted"], prep["alloc"]
    canonical = prep["canonical"]
    out["diagnostics"]["unrouted_lines"] = unrouted
    if prep["coverage"] is not None:
        out["diagnostics"].update(coverage=prep["coverage"], a_walk=prep["a_walk"],
                                  queued=[{k: v for k, v in q.items() if k != "ids"} for q in prep["queued"]])

    printed = prep["printed"]       # routed and queued lines: both roles' printed-line protection
    # cross-dimension policy: a root-cause job's Relations does not carry a calculation across different dims (steps.Relations.overlap)
    # one Relations for placement and the vote path, with the source run's policy (steps.PHASES)
    index, rel = fact_index(f), phase_relations(f, "placement", getattr(a, "root_cause_fixes", False))
    rows_by_section = prep["offered"]     # rows_of: applicable sections, rows some total takes
    names = {r["row_id"]: r["name"] for r in rows}
    search = None
    if a.search and (any(st[s]["queue"] for s in sections) or prep["queued"]):
        # every embedding request under the ledger and the stop switch
        out["embedder"] = out.get("embedder") or (HostedEmbedder if a.engine == "openrouter" else OllamaEmbedder)(
            ledger=out["ledger"], company=out["company"], stop=out["stop"])
        search = out["search"] = Search.of(f, out["embedder"], **({"cache": a.cache} if a.cache else {}))

    # two working registries, run-wide: what C4's committed pieces own, and what the checker's own accepted pieces
    # own, so C4's allocations never constrain the blind checker
    owned, checker_owned, pieces, moved = [], [], [], {}
    haiku_owned = {}     # source line -> Haiku's active claims for it: a line re-solved after a hand-off replaces them
    counts = {"agree": 0, "disputed": 0, "disputed_same_row_totals": 0, "unreviewed": 0, "unreviewed_by": {},
              "not_reviewed_c4_unplaced": 0, "checker off": 0,
              "label": "agreement is a consistency result, not correctness"}
    votes = {}

    def loop(l, who, registry, s, ids=None, extra=""):
        """One role's loop on the line in section s, with its own tools, memory, transcript and registry. A queued
        line (s None) is offered `ids`, every offered row of the statement, and no rows_of; a memo section's line its
        memo rows, and no rows_of. `extra`: the retry's candidate block, shown after the start message and never part
        of it (the checker-feedback retry)."""
        tools = Tools(f, l, rows_by_section if s and not t.memo(s) else {}, s, names, search=search,
                      tax=tax, statement=statement, v20=True, fact_hint=a.combined_fixes,
                      root_cause=getattr(a, "root_cause_fixes", False),
                      statements_only=getattr(a, "statements_only", False))
        r7_start = len(rel.r7_log)
        rec = run_c4(caller, tax, f, l, s, st[s]["offered"] if s else ids, tools, printed, list(registry), role=who,
                     index=index, rel=rel, cap=a.cap, prompt_name=a.prompt, row_seed=a.row_seed,
                     **({"extra_text": extra} if extra else {}))
        for e in rel.r7_log[r7_start:]:      # the cross-dimension attempt log (empty without root-cause fixes): whose attempt it was
            e.update(role=who, section=s)
        rec["search_use"] = search_use(rec)
        return rec

    def keys(rec):
        return [ident(x) for x in (rec.get("answer") or {}).get("_pieces") or ()]

    def third(l, s, entry):
        """Haiku's blind answer on the line, with its own registry; the hash covers what it saw. The registry leaves out
        Haiku's own earlier claims for this line (made in the sending section before a hand-off), and a finished answer
        replaces them; both attempts stay in the record."""
        reg = [k for src, ks in haiku_owned.items() if src != l["concept"] for k in ks]
        h = entry["haiku"] = loop(l, "haiku", reg, s)
        h["inputs_hash"] = hashlib.sha256(json.dumps({"start": h.get("start"), "row_order": h.get("row_order"),
                                                      "registry": sorted(map(str, reg))}).encode()).hexdigest()
        if h["termination"] in FINISHED:
            haiku_owned[l["concept"]] = keys(h)
        return h

    def vote(c, s, l, p, rec8, entry, detail):
        """The vote on a disputed line (see above); returns the exception to record."""
        h = entry.get("haiku") or third(l, s, entry)
        who = getattr(a, "third_model", None) or "Haiku"       # the tiebreaker's model, named in its explanations
        v = entry["vote"] = {"analyst": detail["analyst"], "checker": detail["checker"], "label": "experimental"}
        if h["termination"] not in FINISHED:
            v.update(result="no vote", why=f"{who} did not finish ({h['termination']}: {h.get('why')})",
                     cap=h.get("stop_class") == "CapReached")
            votes["no vote, cap reached"] = votes.get("no vote, cap reached", 0) + v["cap"]
        else:
            v["haiku"] = dh = decomposition(h, canonical)
            if dh == detail["analyst"]:
                v.update(result="analyst", why=f"{who} agrees with the analyst; C4's placement stands")
            elif dh != detail["checker"]:
                v.update(result="no majority", why="the three answers differ")
            elif rec8["decision"] != "row" or rec8["row"] not in st[s]["offered"]:
                v.update(result="checker, not a row", why=f"{who} and the checker agree on {dh['row']}, not a row "
                         "of this section; C4's placement stands")
            else:
                others = list(owned)
                for k in keys(p):
                    others.remove(k)
                given = [{"fact_id": x["fact"], "row": x["row"]} for x in rec8["answer"].get("_pieces") or ()]
                r7_start, r7_v0 = len(rel.r7_log), rel.validated
                new, _, refused = six_rules.amounts(tax, f, l, given, printed, others, index, rel)
                rel.vote_validated += rel.validated - r7_v0      # the vote path's own denominator
                open_claims.closed(rel, r7_start, refused, role="vote", section=s)
                if refused:
                    v.update(result="checker, refused", refused=refused,
                             why="the majority answer fails code's checks against the analyst's registry: "
                                 + "; ".join(r["text"] for r in refused))
                else:        # replaced in one step, before the next line
                    for x in new:
                        x["by"] = "vote"
                    pieces[:] = [x for x in pieces if x["source"] != l["concept"]] + new
                    owned[:] = others + [ident(x) for x in new]
                    moved[c] = sum((x["amount"] for x in new), Decimal(0))
                    alloc[c] = {"row": rec8["row"], "by": "vote", "replaced": alloc[c]["row"],
                                **({"pieces": len(new)} if new else {}),
                                **({"from": alloc[c]["from"]} if "from" in alloc[c] else {})}
                    v.update(result="checker", why=f"{who} agrees with the checker; the checker's answer replaces "
                                                   "C4's")
        votes[v["result"]] = votes.get(v["result"], 0) + 1
        settled = v["result"] in ("analyst", "checker")
        return {"step": "C8", "class": SETTLED if settled else "disputed", "concept": c, "c4": detail["analyst"],
                "c8": detail["checker"], "haiku": v.get("haiku"), "vote": v["result"],
                "same_row_totals": detail["same_row_totals"], "why": v["why"]}

    feedback = getattr(a, "checker_feedback", False)
    if feedback:
        counts["second_look"] = {"disputed_retries": 0, "unplaced_retries": 0, "kept_retry": 0, "kept_first": 0,
                                 "not_run": 0, SECOND_LOOK: 0, "still disputed": 0, "label": FEEDBACK_LABEL}
        counts["unplaced_checked"] = {"agree": 0, "disputed": 0, "unreviewed": 0, "not run": 0,
                                      "label": "the checker on lines the analyst answered OUTSIDE or NOT_TOLD"}

    def halted():
        """Why nothing more may be sent: the caller's or the embedder's failure, or the stop switch already pressed
        (a reply that arrives after it is not kept); None otherwise. A halt is never cleared."""
        stop = getattr(caller, "stopped", None)
        return (caller.failed or (out.get("embedder") and out["embedder"].failed)
                or ("stop switch" if stop and stop() else None))

    def candidate(rec, k):
        """One candidate of the retry's block: the answer's row or outcome, its pieces by fact ID, quote and reason."""
        ans = rec.get("answer") or {}
        ps = ans.get("pieces") or []
        d = rec.get("row") or rec.get("decision")
        if a.combined_fixes and d in NO_ROW:      # the combined fixes' no-row candidate
            return "\n".join([f"CANDIDATE {k}", f"  no row allocated: {d} ({NO_ROW[d]})",
                              f'  quote: "{ans.get("quote") or ""}"', f"  reason: {ans.get('reason') or ''}"])
        return "\n".join([f"CANDIDATE {k}", f"  row: {d}",
                          "  pieces: " + ("; ".join(f"{x.get('fact_id')} to {x.get('row')}"
                                                    + (f' (quote "{x["quote"]}")' if x.get("quote") else "")
                                                    for x in ps) if ps else "none, the line kept whole"),
                          f'  quote: "{ans.get("quote") or ""}"', f"  reason: {ans.get('reason') or ''}"])

    def second_look(c, s, l, first, rec8, registry, unplaced=False):
        """The checker-feedback retry: the analyst's one retry on the line, in a fresh transcript with its own tools, step cap and
        `registry` (the analyst's without this line's own claims). The start message is the line's, then the two
        answers as candidates A and B in an order drawn from the row seed and the line, never saying whose is whose;
        the block is shown, never part of the start, so nothing in it is quotable or citable. The retry is kept when
        its loop finishes, code's checks (rules 5 and 6) having run in it, with a row of the section or an answer
        that settles the line without one (OUTSIDE, NOT_TOLD, COMBINED). The first answer stands, with the reason,
        when the retry refers the line or answers a row of another section, does not finish, or the caller is
        halted before it (not run: nothing sent) or during it (a late reply is not kept); a halt is never cleared.
        A retry whose loop sent nothing (the ledger's analyst sends unchanged: a refusal before the first request,
        context or ledger) is not run either; its loop record and reason are kept. Returns (the record that stands,
        the feedback record)."""
        order = ["analyst", "checker"]
        random.Random(f"{a.row_seed}:feedback:{c}").shuffle(order)
        fb = {"order": order, "path": "unplaced" if unplaced else "disputed", "label": FEEDBACK_LABEL}
        sl = counts["second_look"]
        if halted():
            sl["not_run"] += 1
            fb.update(ran=False, kept="first", why=f"not run: {halted()}; the first answer stands")
            return first, fb
        by = {"analyst": first, "checker": rec8}
        block = "\n\n".join([FEEDBACK_TEXT] + [candidate(by[w], k) for w, k in zip(order, "AB")])
        sends = lambda: sent_by_role(out["ledger"].state()).get("analyst", 0)     # noqa: E731
        n = sends()
        r = fb["retry"] = loop(l, "analyst", registry, s, extra=block)
        fb["ran"] = sends() > n
        if not fb["ran"]:
            sl["not_run"] += 1
            fb.update(kept="first", why=f"not run: the retry sent nothing ({r['termination']}: {r.get('why')}); "
                                        "the first answer stands")
            return first, fb
        sl["unplaced_retries" if unplaced else "disputed_retries"] += 1
        _, kind, _ = outcome_of(r, st[s]["offered"])
        if halted():
            fb.update(kept="first", why=f"halted during the retry ({halted()}); its answer is not kept and the first "
                                        "answer stands")
        elif r["termination"] not in FINISHED:
            fb.update(kept="first", why=f"the retry did not finish ({r['termination']}: {r.get('why')}); the first "
                                        "answer stands")
        elif kind is None or kind in MEMO_ANSWERED:
            fb.update(kept="retry", why="the retry finished and passed code's checks; its answer stands")
        else:
            fb.update(kept="first", why=f"the retry's answer settles nothing here ({kind}: "
                                        f"{r.get('target') or r.get('row') or r['decision']}); the first answer stands")
        sl["kept_" + fb["kept"]] += 1
        return (r if fb["kept"] == "retry" else first), fb

    def replace(c, first, rec, others):
        """The kept retry's claims replace the line's in one step, before the next line (as a vote's replacement).
        `by` says where the answer came from; `changed` whether its decomposition differs from the first's."""
        new = rec["answer"].get("_pieces") or []
        for x in new:
            x["by"] = "second look"
        changed = decomposition(first, canonical) != decomposition(rec, canonical)
        pieces[:] = [x for x in pieces if x["source"] != c] + new
        owned[:] = others + keys(rec)
        moved[c] = sum((x["amount"] for x in new), Decimal(0))
        alloc[c] = {"row": rec["row"], "by": "second look", "changed": changed,
                    **({"replaced": alloc[c]["row"]} if c in alloc and changed else {}),
                    **({"pieces": len(new)} if new else {}),
                    **({"from": alloc[c]["from"]} if c in alloc and "from" in alloc[c] else {})}

    def unplace(c, s, l, others, kind, why):
        """A kept retry that settles a placed line without a row: the line's claims and pieces go in one step, and the
        line is recorded as an unplaced answer is (a memo section's answered, any other's a C4 exception)."""
        pieces[:] = [x for x in pieces if x["source"] != c]
        owned[:] = others
        moved.pop(c, None)
        alloc.pop(c, None)
        if c in transfers:
            transfers[c]["second_look"] = kind
        if t.memo(s):
            l["memo_answered"] = kind
        else:
            exc.append({"step": "C4", "class": kind, "concept": c, "why": why})
            st[s]["unresolved"].add(c)

    def after_look(c, s, l, first, kept, rec8, entry, fb):
        """The C8 exception after the second look: settled after a second look, or still disputed (then the vote,
        when the tiebreaker is on and a row stands, on the answer that stands). A retry not run counts as not run
        only; its line stays disputed. The record keeps the first answer, the checker's and the one that stands."""
        after, detail = compare(kept, rec8, canonical)
        fb["after"] = {"result": after, **detail}
        if fb["ran"]:
            counts["second_look"][SECOND_LOOK if after == "agree" else "still disputed"] += 1
        looked = {"first": decomposition(first, canonical), "after": decomposition(kept, canonical),
                  "kept": fb["kept"], "feedback_path": fb["path"]}
        if after != "agree" and a.third_opinion and kept["decision"] == "row":
            return {**vote(c, s, l, kept, rec8, entry, detail), **looked}
        return {"step": "C8", "class": SECOND_LOOK if after == "agree" else "disputed", "concept": c,
                "c4": looked["first"], "c8": decomposition(rec8, canonical), **looked,
                "same_row_totals": detail.get("same_row_totals"),
                "why": ("after one retry with both answers as candidates, the analyst's answer equals the blind "
                        "checker's: a second look, not verified agreement" if after == "agree" else
                        f"the blind checker's decomposition differs ({fb['why']}); the analyst's answer stands, "
                        "disputed")}

    def unplaced_look(c, s, l, p, entry):
        """The checker-feedback retry on a line the analyst answered OUTSIDE or NOT_TOLD: the checker answers blind, as on a placed line.
        A disputed line gets its C8 exception: when the checker placed a row of the section, after the analyst's one
        retry; else at once, with no retry. A kept row is committed here (True); a kept OUTSIDE, NOT_TOLD or COMBINED
        replaces the line's answer, and the caller records the line's C4 outcome (False)."""
        nonlocal checker_owned
        uc = counts["unplaced_checked"]
        if halted():
            uc["not run"] += 1
            entry["review"] = {"termination": None, "why": f"not run: {halted()}"}
            return False
        rec8 = entry["review"] = loop(l, "checker", checker_owned, s)
        if rec8["termination"] in FINISHED:
            checker_owned += keys(rec8)
        result, detail = compare(p, rec8, canonical)
        entry["consistency"] = {"result": result, **detail, "path": "unplaced"}
        uc[result] += 1
        if result != "disputed":
            return False
        if rec8["decision"] != "row" or rec8["row"] not in st[s]["offered"]:
            exc.append({"step": "C8", "class": "disputed", "concept": c, "c4": detail["analyst"],
                        "c8": detail["checker"], "first": detail["analyst"], "after": detail["analyst"],
                        "kept": "first", "feedback_path": "unplaced",
                        "why": "the blind checker's answer differs but places no row of this section, so there is no "
                               "retry; the analyst's answer stands, disputed"})
            return False
        kept, fb = second_look(c, s, l, p, rec8, list(owned), unplaced=True)
        entry["feedback"] = fb
        row = None
        if kept is not p:
            row, kind, why = outcome_of(kept, st[s]["offered"])
            entry.update({"c4_first": p, "c4": kept, "row": row, "class": kind, "why": why})
            if row:
                replace(c, p, kept, list(owned))
        exc.append(after_look(c, s, l, p, kept, rec8, entry, fb))
        return row is not None

    transfers, pending, late, done = {}, {s: [] for s in sections}, [], set()
    stopped = None

    def unresolved_transfer(c, sender, s, why):
        st[sender]["unresolved"].add(c)
        transfers[c].update(status="unresolved", why=why)
        exc.append({"step": "C4", "class": "unresolved transfer", "section": sender, "concept": c,
                    "why": f"unresolved transfer to {s}: {why}"})

    def process(s, l, sender=None):
        """C4 and C8 on one line in section s; `sender`: the section that referred it here."""
        nonlocal owned, checker_owned, pieces, stopped
        c, start = l["concept"], len(exc)
        if stopped:
            if sender:
                unresolved_transfer(c, sender, s, f"not run: {stopped}")
            else:
                exc.append({"step": "C4", "class": "not run", "concept": c, "why": f"not run: {stopped}"})
                st[s]["unresolved"].add(c)
            tag(s, start)
            return
        if sender:
            st[s]["signs"][c] = (l["sign"]["d"] if "sign" in l     # C3a: one normal balance per statement
                                 else sign(balance_of(tax, f, c), st[s]["balance"]))
            if st[s]["signs"][c] is None:
                unresolved_transfer(c, sender, s, "sign unknown in the receiving section (no balance)")
                return
        p = loop(l, "analyst", owned, s)
        row, kind, why = outcome_of(p, st[s]["offered"])
        entry = {"section": s, "c4": p, "row": row, "class": kind, "why": why}
        if sender:
            out["c4"][c]["transfer"] = entry
        else:
            out["c4"][c] = entry
        looked = feedback and kind in FEEDBACK_UNPLACED and not sender and not t.memo(s)
        if looked and unplaced_look(c, s, l, p, entry):
            pass        # checker-feedback retry: the retry placed the line; committed and recorded in unplaced_look
        elif kind is None:
            if sender:          # accepted: ownership moves to the receiver
                st[sender]["lines"].remove(l)
                st[s]["lines"].append(l)
                owner[c] = s
                transfers[c].update(status="accepted", row=row,     # the hand-off's agreement measure
                                    receiver="row named" if row == transfers[c]["row_named"] else "another row")
            mine = p["answer"].get("_pieces") or []
            alloc[c] = {"row": row, "by": "C4", **({"pieces": len(mine)} if mine else {}),
                        **({"from": sender} if sender else {})}
            pieces += mine
            moved[c] = sum((x["amount"] for x in mine), Decimal(0))
            owned += keys(p)
            if a.no_checker:        # the checker-off test: no blind review; the analyst's answer stands
                counts["checker off"] += 1
            else:
                rec8 = entry["review"] = loop(l, "checker", checker_owned, s)
                if rec8["termination"] in FINISHED:
                    checker_owned += keys(rec8)
                result, detail = compare(p, rec8, canonical)
                entry["consistency"] = {"result": result, **detail}
                counts[result] += 1
                if result == "disputed" and feedback:    # checker-feedback retry: the retry first; the vote only if still disputed
                    counts["disputed_same_row_totals"] += bool(detail["same_row_totals"])
                    others = list(owned)
                    for k in keys(p):
                        others.remove(k)
                    kept, fb = second_look(c, s, l, p, rec8, others)
                    entry["feedback"] = fb
                    if kept is not p:
                        row, kind, why = outcome_of(kept, st[s]["offered"])
                        entry.update({"c4_first": p, "c4": kept, "row": row, "class": kind, "why": why})
                        if row:
                            replace(c, p, kept, others)
                        else:       # the retry settled the line without a row (OUTSIDE, NOT_TOLD, COMBINED)
                            unplace(c, s, l, others, kind, why)
                    exc.append(after_look(c, s, l, p, kept, rec8, entry, fb))
                elif result == "disputed":
                    counts["disputed_same_row_totals"] += bool(detail["same_row_totals"])
                    exc.append(vote(c, s, l, p, rec8, entry, detail) if a.third_opinion else
                               {"step": "C8", "class": "disputed", "concept": c, "c4": detail["analyst"],
                                "c8": detail["checker"], "same_row_totals": detail["same_row_totals"],
                                "why": "the blind checker's decomposition differs; C4's placement stands, disputed"})
                elif result == "unreviewed":
                    ub = counts["unreviewed_by"]
                    ub[rec8["termination"]] = ub.get(rec8["termination"], 0) + 1
                    exc.append({"step": "C8", "class": "unreviewed", "concept": c,
                                "why": f"no usable check ({rec8['termination']}: {rec8.get('why')}); C4's placement "
                                       "stands, unreviewed"})
        elif t.memo(s) and kind in MEMO_ANSWERED:
            # a memo line no memo row fits stays outside the spread: answered, not an exception
            counts["not_reviewed_c4_unplaced"] += 1
            entry["review"] = {"termination": None, "why": "not run: C4 placed no row of the section"}
            entry["memo"] = "answered: no memo row taken; outside every total"
            l["memo_answered"] = kind
        else:
            if looked:      # checker-feedback retry: the checker ran (or was stopped); the answer that stands is entry's
                p, kind, why = entry["c4"], entry["class"], entry["why"]
            else:
                counts["not_reviewed_c4_unplaced"] += 1
                entry["review"] = {"termination": None, "why": "not run: C4 placed no row of the section"}
            target = t.section_of(p["target"]) if p.get("decision") == "referred" else None
            if target and (t.memo(s) or t.memo(target)):     # no hand-off into or out of a memo section
                why, target = f"{why}; a memo section takes no referral and makes none", None
            if sender:
                unresolved_transfer(c, sender, s, "referred again to " + p["target"] + "; one transfer per line"
                                    if target else f"{kind}: {why}")
            elif target in st and target != s and p["target"] in st[target]["offered"]:
                transfers[c] = {"from": s, "to": target, "row_named": p["target"], "status": "pending",
                                "late": target in done}
                (late if target in done else pending[target]).append((target, l, s))
            else:
                exc.append({"step": "C4", "class": kind, "concept": c,      # receiver validation
                            "why": why if not target else f"{why}; that section is not in this run"
                                   if target not in st else f"{why}; {p['target']} is not offered in this run"
                                   if p["target"] not in st[target]["offered"] else why})
                st[s]["unresolved"].add(c)
        tag(s, start)
        stopped = caller.failed or (out.get("embedder") and out["embedder"].failed) or None

    def committed(l):
        """What the run has committed for the line at this point, as build will claim it: the line's row with the
        rest of its amount, then each piece with its row and amount (a vote's replacement included); [] when
        nothing is committed."""
        c = l["concept"]
        if c not in alloc:
            return []
        return [{"row": alloc[c]["row"], "amount": l["amount"] - moved.get(c, 0), "by": alloc[c]["by"]}] + [
            {"row": x["row"], "amount": x["amount"], "by": x.get("by") or alloc[c]["by"], "concept": x["concept"]}
            for x in pieces if x["source"] == c]

    def processed(s, l, sender=None):
        was = stopped       # the line's own state: skipped because the run had stopped, or a loop that ended not run
        process(s, l, sender)
        e = out["c4"].get(l["concept"]) or {}
        e = e.get("transfer") or {} if sender else e
        placed = committed(l)
        line_event(d, {"event": "line", "statement": statement, "section": s, "concept": l["concept"],
                       "label": l.get("label"), "from": sender, "row": placed[0]["row"] if placed else e.get("row"),
                       "placed": placed, "class": e.get("class"),
                       "review": (e.get("consistency") or {}).get("result"),
                       "vote": (e.get("vote") or {}).get("result"),
                       "not_run": was or (stopped if e.get("class") == "not run" else None)})

    line_event(d, {"event": "planned", "statement": statement, "sections": sections,
                   "lines": [l["concept"] for s in sections for l in st[s]["queue"]]
                   + [q["line"]["concept"] for q in prep["queued"]]})
    for s in sections:
        for l in st[s]["queue"]:
            processed(s, l)
        for s2, l, sender in pending[s]:
            processed(s2, l, sender)
        done.add(s)
    for s, l, sender in late:        # a referral to a section already run
        processed(s, l, sender)

    def place_queued(q):
        """C4b on one queued line, after every routed line and transfer and before C5 and C6: the analyst places it
        against every offered row of the statement with the run-wide registry; the checker answers blind on every
        queued line while the caller is not stopped, with its own registry (never merged into the analyst's); code
        compares (queued_outcome); an analyst row is then committed (commit_queued). The third opinion is off for
        queued lines."""
        nonlocal stopped, checker_owned
        l, c, start = q["line"], q["line"]["concept"], len(exc)
        said = q.get("organizer") or {}
        if not q["ids"] and said.get("cause") == og_none_cause():    # the organizer's own decision, with its reason
            outcomes[c] = {"state": "outside: the organizer: none of these sections", "resolved": True,
                           "organizer": said, "why": f"the organizer: none of these sections; its reason: "
                                                    f"{said.get('reason')}"}
            return
        if not q["ids"] and not q.get("memo_ids"):  # read for its memo rows only, and not a can't tell: no row to choose
            outcomes[c] = {"state": "outside: the statement offers no row", "resolved": True,
                           "why": "outside the template: this statement is read for its memo rows only"
                                  + (f" (organizer: {said['cause']})" if said.get("cause") else "")}
            return
        # the two outside exits above call no model, so they hold after a stop (dealer results v1: a stopped ledger
        # had turned five organizer "none" lines into unresolved ones). Below, a stop still comes before the sign
        # check, which calls no model either: an unsigned line reads "not run" after a stop, "not sent" otherwise;
        # both stay unresolved
        if stopped:
            outcomes[c] = {"state": "not run", "resolved": False, "why": f"not run: {stopped}"}
            exc.append({"step": "C4", "class": "not run", "concept": c, "why": f"queued line not run: {stopped}"})
            return
        if not q["ids"]:    # a can't tell on a memo-only statement goes to both models with the memo rows
            q = {**q, "ids": q["memo_ids"], "memo_only": True}
        if q["sign"]["d"] is None:      # nothing could be committed without a direction (C3a), so it is not sent
            outcomes[c] = {"state": "not sent: sign unresolved", "resolved": False,
                           "why": f"sign unresolved: {q['sign'].get('why')}"}
            return
        p = loop(l, "analyst", owned, None, q["ids"])
        stopped = caller.failed or (out.get("embedder") and out["embedder"].failed) or None
        rec8 = ({"termination": None, "why": f"not run: {stopped}"} if stopped else
                {"termination": None, "why": CHECKER_OFF} if a.no_checker else
                loop(l, "checker", checker_owned, None, q["ids"]))
        if rec8["termination"] in FINISHED:
            checker_owned += keys(rec8)
        o = outcomes[c] = queued_outcome(q, p, rec8, canonical)
        if said:
            o["organizer"] = said
        out["c4"][c] = {"section": o.get("section"), "queued": True, "c4": p, "review": rec8, "row": o.get("row"),
                        "outcome": o["state"], **({"gate": MEMO_GATE} if q.get("memo_only") else {})}
        if q.get("memo_only") and "row" in o and o["state"] != "agree":    # the gate: only an agreed placement
            raise ValueError(f"{MEMO_GATE}: {c} carries a row in state {o['state']}")
        if "row" in o:
            why = commit_queued(q, p, o["row"])
            if why:
                o.update(state="ownership conflict", resolved=False, why=why, row_answered=o.pop("row"))
                exc.append({"step": "C4b", "class": "ownership conflict", "concept": c, "why": why})
            else:
                o["committed"] = True
                if o["state"] not in ("agree", "checker off"):  # placed, qualified: a routed line's C8 exception
                    exc.append({"step": "C8", "class": "unreviewed" if o["state"] == "unreviewed" else "disputed",
                                "concept": c, "section": o["section"], "queued": o["state"], "why": o["why"],
                                **({"sections": o["sections"]} if "sections" in o else {})})
        tag(o.get("section") or "run", start)
        stopped = caller.failed or (out.get("embedder") and out["embedder"].failed) or None

    def commit_queued(q, p, row):
        """The one commit of a queued line: only the analyst's row, after the comparison. The line joins the row's
        section as an accepted hand-off does (its lines, signs, owner, alloc, pieces, moved and owned). A key already
        owned (a piece's, or the line's own amount) is an ownership conflict: nothing is committed, and the reason
        names the line that owns it."""
        nonlocal owned, pieces
        l, c = q["line"], q["line"]["concept"]
        mine = p["answer"].get("_pieces") or []
        held = {ident(x): x["source"] for x in pieces}
        clash = [k for k in [line_ident(f, c)] + keys(p) if k in owned]
        if clash:
            return f"ownership conflict with {held.get(clash[0], 'an amount already owned')} over {clash[0][0]}"
        s = t.section_of(row)
        st[s]["lines"].append(l)
        st[s]["signs"][c] = q["sign"]["d"]
        owner[c] = s
        alloc[c] = {"row": row, "by": "C4", "queued": True, **({"pieces": len(mine)} if mine else {})}
        pieces += mine
        moved[c] = sum((x["amount"] for x in mine), Decimal(0))
        owned += keys(p)
        return None

    outcomes = {}
    for q in prep["queued"]:
        n0 = len(exc)
        place_queued(q)
        o = outcomes.get(q["line"]["concept"]) or {}
        if not o.get("resolved") and not any(x.get("concept") == q["line"]["concept"] for x in exc[n0:]):
            # every unsettled queued line reaches the analyst's Exceptions sheet, whichever exit place_queued took
            exc.append({"step": "C4b", "class": "unresolved", "concept": q["line"]["concept"],
                        "section": o.get("section") or "run", "queued": o.get("state"), "why": o.get("why")})
        placed = committed(q["line"])
        line_event(d, {"event": "line", "statement": statement, "section": o.get("section"), "queued": True,
                       "concept": q["line"]["concept"], "label": q["line"].get("label"),
                       "row": placed[0]["row"] if placed else o.get("row"), "placed": placed,
                       "class": o.get("state"), "review": o.get("state"),
                       "not_run": stopped if o.get("state") == "not run" else None})
    out["diagnostics"]["review_counts"] = counts
    if a.third_opinion:
        out["diagnostics"]["votes"] = {**votes, "label": "experimental; a vote is not a correctness result"}
    out["diagnostics"]["transfers"] = transfers
    for c, p in alloc.items():
        print(f"{c:<55} -> {p['row']} ({p['by']})")

    # C5 and C6 per section, after every transfer; A3 is off, the only pieces are those C4 gave that passed rules
    # 5 and 6
    all_lines, all_claims, all_rows, all_checks, gaps, incomplete = [], [], {}, [], {}, {}

    def rival_amount(k):
        return rival_amount_of(f, key, [l for s2 in sections for l in st[s2]["lines"]]
                               + [q["line"] for q in prep["queued"]], pieces, k)

    for s in sections:
        x, start = st[s], len(exc)
        mine = {l["concept"] for l in x["lines"]}
        claims = build(f, x["lines"], {c: v for c, v in alloc.items() if c in mine},
                       [p for p in pieces if p["source"] in mine], moved, x["signs"], exc)
        if rel.cross_dims_unknown:      # cross-dimension policy: an unknown overlap stays an open claim, visible and unsettled
            open_claims.remainders(claims)
            exc.extend(open_claims.exceptions(claims, x["lines"], out["c4"], rival_amount))
        name = t.filed_total(s)
        own = instant or (t.local(s) and bool(name))     # IS and CF: the run computes the other controls (C6a)
        built, checks = c6(x["lines"], claims, x["ids"], x["unresolved"], x["signs"], s, control=x["control"],
                           canonical=canonical, filed_absent=not instant and bool(name) and not read(f, name, key),
                           runwide=not own)
        neg = sorted(r for r, v in built.items() if v < 0)
        if instant and neg:    # C5: a diagnostic on the balance sheet, no check; signed IS and CF rows are not
            out["diagnostics"].setdefault("negative_rows", {})[s] = neg
        partly = [c["check"] for c in checks if c["check"].endswith("fully allocated") and not c["ok"]]
        if x["unresolved"] or partly or x["member"] or x["refused"]:
            incomplete[s] = {"unallocated": sorted(x["unresolved"]), "partly_allocated": partly}
            if x["member"]:     # known missing input: totals over it are unavailable
                incomplete[s]["printed_by_member_only"] = list(x["member"])
            if x["refused"]:    # an amount not extracted: the same
                incomplete[s]["extraction_refused"] = list(x["refused"])
        recon = next((c for c in checks if c["check"].startswith("rows add to")), None)
        if recon and not recon["ok"] and recon["gap"] is not None:
            gaps[s] = gap_lines(x["lines"], x["unresolved"], recon["gap"], recon["tolerance"])
        for c in checks:
            c["section"] = s
        for l in x["lines"]:
            l["section"] = s
        for c in claims:
            c["section"] = s
        if x["control"]:
            out["diagnostics"].setdefault("controls", {})[s] = x["control"]
        tag(s, start)
        all_lines += x["lines"]
        all_claims += claims
        all_rows.update(built)
        all_checks += checks
    if len(sections) > 1:
        seen, twice = {}, []
        for c in all_claims:
            k = ident(c)
            if k in seen and seen[k] != c["section"]:
                twice.append({"amount": k, "sections": [seen[k], c["section"]]})
            seen.setdefault(k, c["section"])
        all_checks.append({"check": "no amount claimed in two sections", "ok": not twice, "section": "run",
                           "claimed_twice": twice})
    if gaps:
        out["diagnostics"]["gap_lines"] = gaps[sections[0]] if len(sections) == 1 else gaps
    # C6: an unresolved item whose section is unknown may belong to any offered section. C4a gives each printed
    # income or cash flow line a state; a line printed by member only, an incomplete subtotal and an unresolved queued
    # line are such items, propagated once. The balance sheet's unrouted lines stay a diagnostic
    unknown, outside = {}, []
    if prep["coverage"] is not None:
        qcounts = queue_counts(outcomes)
        chk, unknown = coverage_check(prep["coverage"], outcomes, [c for s in sections for c in st[s]["unresolved"]
                                                                   if st[s]["signs"].get(c) is None])
        outside = [q["line"] for q in prep["queued"]
                   if (outcomes[q["line"]["concept"]].get("answers") or {}).get("analyst") == "OUTSIDE"
                   and outcomes[q["line"]["concept"]]["answers"].get("checker") == "OUTSIDE"]
        all_checks.append(chk)
        out["diagnostics"]["queued_outcomes"] = outcomes
        out["diagnostics"]["queue_counts"] = qcounts
    all_checks += section_unknown(incomplete, [s for s in sections if t.sections[s]["kind"] == "spread"], unknown)
    out["computed_totals"] = computed_totals(all_rows, incomplete)
    if rel.cross_dims_unknown:      # cross-dimension policy: totals over a provisional row, and the run's record of its open claims
        provisional = open_claims.provisional_rows(all_claims)
        marked = open_claims.mark_totals(t, out["computed_totals"], provisional)
        out["diagnostics"]["open_claims"] = open_claims.summary(rel, all_claims, exc, provisional, marked)
    states = control_states(f, tax, key, statement, sections, all_rows, all_claims, all_checks, incomplete, canonical)
    for x in states:
        if x["state"].startswith("unavailable"):
            exc.append({"step": "C6", "class": "control unavailable", "section": x["section"],
                        "why": f"control {x['state']}"})
        elif x["kind"] == "run-wide" and x["state"] in ("tied", "differs"):
            all_checks.append({"check": f"{x['section']} control ties", "ok": x["state"] == "tied",
                               "section": x["section"],
                               **{k: x[k] for k in ("filed", "filed_value", "built", "gap", "tolerance")}})
    exc += sign_exceptions(states, all_lines, f)     # filed-sign notes
    all_checks += post_checks(all_checks, all_claims, {s: st[s]["control"] for s in sections}, canonical)
    # C4b's B: beside each control that differs, the amount matches among the lines both models put OUTSIDE; a
    # record on the failed check, never a change to its ok, gap, the verdict or the coverage
    if outside:
        matched = 0
        for x in states:
            if x["state"] != "differs":
                continue
            c = t.controls[x["section"]]
            filed = c["filed"] or next(y["concept"] for y in c["reconstructed"]["terms"] if y["coef"] > 0)
            xs = read(f, filed, key)
            if not xs:
                continue
            # the check's gap is in the section's normal-balance arithmetic (filed x its direction a, less built);
            # B compares in the filed concept's own arithmetic: g = a x gap, a = +-1
            a = sign(balance_of(tax, f, filed), t.sections[x["section"]]["normal_balance"]) or 1
            got = gap_matches(f, role, key, filed, xs[0]["currency"], outside, a * x["gap"], x["tolerance"])
            matched += len(got["gap_match_candidates"])
            out["diagnostics"].setdefault("gap_matches", {})[x["section"]] = got
            for chk in all_checks:
                if chk["section"] == x["section"] and not chk["ok"] and chk["check"] in (
                        f"{x['section']} control ties", f"rows add to {c['filed']}"):
                    chk["gap_match_candidates"] = got["gap_match_candidates"]
        out["diagnostics"]["queue_counts"]["gap-match candidates (B)"] = matched
    out["controls"] = {k: [x for x in states if x["state"].split(":")[0] == k]
                       for k in ("tied", "differs", "unavailable", "not evaluated", "failed")}
    disputed = {alloc[x["concept"]]["row"] for x in exc if x.get("class") == "disputed" and x.get("concept") in alloc}
    out["report_checks"] = report_checks(f, key, statement, sections, all_rows, all_claims, incomplete, canonical,
                                         disputed, variant)
    out.update(sections=sections, lines=all_lines, alloc=alloc, claims=all_claims, rows=all_rows, checks=all_checks)


def exceptions_summary(exc, checks, outcome, error=None):
    """What a person reads first: every exception counted by step and class, each with its line and reason, the
    failed checks behind a blocked outcome, and the fatal error behind 'not run'. Code only groups and counts; no
    case is resolved here."""
    by = {}
    for x in exc:
        k = f"{x['step']} {x['class']}"
        by.setdefault(k, []).append({"concept": x.get("concept") or x.get("contract"), "why": x.get("why")})
    return {"outcome": outcome, "exceptions": len(exc), "counts": {k: len(v) for k, v in sorted(by.items())},
            "cases": dict(sorted(by.items())), "failed_checks": [c for c in checks if not c["ok"]],
            "c7": "off: no repair was tried, so no case is withdrawn",
            **({"fatal": error["error"]} if error else {})}


# a section's checks of its rows against the subtotal the company printed for it (C6), and the run's tie of it
SUBTOTAL_CHECKS = ("section total found", "calculation weights agree with signs", "filed and reconstructed totals agree",
                   "reconstructions agree", "reconstructed total independent of the rows' facts")


def subtotal_check(c):
    """Whether a check compares a section's rows with a printed (or reconstructed) subtotal of that section."""
    n = str(c.get("check"))
    return (n.startswith("rows add to ") or n in SUBTOTAL_CHECKS or n == POST_CHECKS[1]
            or n == f"{c.get('section')} control ties")


def statement_checks():
    """The template's statement checks (format v7, `outcome_checks`), or None: every check decides the outcome."""
    return template.active().outcome_checks


def counts(c, gate):
    """Whether a check decides the outcome under `gate` (statement_checks()): every check without one; with one,
    a subtotal check counts only for a section the gate names, and every other check counts."""
    return not gate or not subtotal_check(c) or c.get("section") in gate["sections"]


NOTE_DISAGREES = "named total disagreement"     # shown in the exceptions, never counted in the outcome


def counted(checks, exc, reports=()):
    """(checks, exceptions) that decide a statement's outcome (C9). All of them, unless the template names statement
    checks (format v7): then a section's subtotal
    checks and its "control unavailable" exception count only for the sections named, and a named report check whose
    state is "differs" counts as a failed check ("statement check <id>"). A named check that is unavailable fails
    nothing. A named-total disagreement (NOTE_DISAGREES) and a filed sign (FILED_SIGN) are notes: they never
    count."""
    exc = [x for x in exc if x.get("class") not in (NOTE_DISAGREES, FILED_SIGN)]
    gate = statement_checks()
    if not gate:
        return checks, exc
    keep, ids = set(gate["sections"]), set(gate["report_checks"])
    checks = [c for c in checks if counts(c, gate)]
    exc = [x for x in exc if not (x.get("step") == "C6" and x.get("class") == "control unavailable"
                                  and x.get("section") not in keep)]
    checks = checks + [{"check": f"statement check {r['id']}", "ok": False, "section": "run"}
                       for r in reports or () if r.get("id") in ids and r.get("state") == "differs"]
    return checks, exc


def verdict(checks, exc, reports=()):
    """C9: blocked when a check that counts fails, qualified when an exception that counts stands, complete
    otherwise (counted)."""
    checks, exc = counted(checks, exc, reports)
    return "blocked" if not all(c["ok"] for c in checks) else "qualified" if exc else "complete"


def main(argv=None, caller=None, embedder=None, company=None):
    """`caller` and `embedder`: fakes for the offline tests; a real run builds both under the run's ledger.
    `company`: an open company ledger (ledger.Ledger, with Haiku's role limit) shared by every statement run of one
    company job, in place of --ledger: every attempt counts in it and in the run's own ledger, whose state is the
    run's delta (outcome.json ledger, and ledger.company_delta)."""
    from .runrec import StopFile
    from .ledger import Both, Ledger, company_delta, open_plan
    from .llm import NUM_PREDICT
    from .loop import STEP_CAP
    ap = argparse.ArgumentParser()
    ap.add_argument("filing")
    ap.add_argument("--section", nargs="+", required=True,
                    help="one or more sections of one statement; a referral between them is a hand-off")
    ap.add_argument("--a2", help="retired: the A2 rule book was removed; refused")
    ap.add_argument("--probe")
    ap.add_argument("--max-calls", type=int, required=True, help="most model requests, both roles and embeddings")
    ap.add_argument("--max-tokens", type=int, required=True, help="most local tokens, embeddings included")
    ap.add_argument("--wall-s", type=float, default=3600.0)
    ap.add_argument("--no-a2", action="store_true", help="accepted for older commands: every line goes to C4")
    ap.add_argument("--search", action="store_true", help="offer search_notes and read_passage to both loops")
    ap.add_argument("--cache", help="the embedding cache folder (default data/derived/search)")
    ap.add_argument("--cap", type=int, default=STEP_CAP)
    ap.add_argument("--num-ctx", type=int, default=None,
                    help=f"default {DURATION_NUM_CTX} for an income or cash flow run, {LOOP_NUM_CTX} for the balance "
                         "sheet")
    ap.add_argument("--answer-cap", type=int, default=NUM_PREDICT)
    ap.add_argument("--prompt", default=LOOP_PROMPT)
    ap.add_argument("--row-seed", type=int, default=ROW_SEED, help="the seed of the checker's template row order")
    ap.add_argument("--row-order", choices=("today", "template"), default="today",
                    help="today: the checker's rows shuffled by --row-seed; template: both roles in template order")
    ap.add_argument("--engine", choices=("ollama", "openrouter"), default="ollama",
                    help="openrouter: both roles on the hosted profile (--hosted-profile); embeddings stay local")
    ap.add_argument("--hosted-profile", choices=sorted(HOSTED_PROFILES), default="hosted",
                    help="with --engine openrouter: hosted (HOSTED, thinking off) or flash (HOSTED_FLASH, per-role "
                         "reasoning)")
    ap.add_argument("--spend-usd", type=float, default=0.0, help="most OpenRouter spend for this company, in USD")
    ap.add_argument("--ledger", help="an open plan ledger (python -m spreading.ledger open): every attempt counts "
                                     "in it too, so the plan's limits hold across its runs")
    ap.add_argument("--cold-start", action="store_true",
                    help="unload every model Ollama holds before the run, and refuse to start if one stays loaded")
    ap.add_argument("--third-opinion", action="store_true", help="the tiebreaker votes on each disputed line (experimental)")
    ap.add_argument("--third-opinion-all", action="store_true", help="retired; refused")
    ap.add_argument("--third-model", help="with --third-opinion: the tiebreaker on this local Ollama model instead of "
                                          "Haiku on OpenRouter; nothing paid (an option, untried)")
    ap.add_argument("--checker-feedback", action="store_true",
                    help="on a disputed line, and on one the analyst "
                         "answered OUTSIDE or NOT_TOLD where the checker places a row, the analyst gets one retry "
                         "with both answers as candidates")
    ap.add_argument("--combined-fixes", action="store_true",
                    help="the combined fixes (off by default): placement prompt c4_place_v27, "
                         "breakdowns' empty result points to calc_children (tools.FACT_HINT), and the second look "
                         "shows a COMBINED, NOT_TOLD or OUTSIDE candidate as no row allocated")
    ap.add_argument("--root-cause-fixes", action="store_true",
                    help="the root-cause fixes, not for ordinary operation; kept for reproducing "
                         "records and gates; needs --combined-fixes")
    ap.add_argument("--statements-only", action="store_true",
                    help="the statements-only option (off by "
                         "default): placement prompt c4_place_v29, every role's tools read the financial "
                         "statements only (no note search, no breakdowns or other_periods; calc_children over the "
                         "statements' calculation arcs); the tree reads it from this run's record; needs "
                         "--combined-fixes, not with --search or --root-cause-fixes")
    ap.add_argument("--no-checker", action="store_true",
                    help="the checker-off test: no blind review of any line; every analyst "
                         "answer stands, counted \"checker off\"; not with --third-opinion or --checker-feedback")
    ap.add_argument("--corrections", help="the job's filing-corrections record (named_totals.run), laid over the filing")
    ap.add_argument("--corrections-sha256", help="its sha256, sealed by the job before this run")
    ap.add_argument("--model", action="append", default=[], metavar="ROLE=MODEL",
                    help="the analyst or checker on this local Ollama model instead of LOOP_MODELS' (the small line-up test)")
    a = ap.parse_args(argv)
    if a.prompt != LOOP_PROMPT:
        ap.error(f"--prompt {a.prompt}: the tags-shown prompts were removed; {LOOP_PROMPT} is the "
                 "only placement prompt")
    if a.combined_fixes:
        a.prompt = COMBINED_PROMPT
    if a.root_cause_fixes:
        if not a.combined_fixes:
            ap.error("--root-cause-fixes: the root-cause fixes run on top of the combined fixes; give "
                     "--combined-fixes too")
        a.prompt = ROOT_CAUSE_PROMPT
    if a.statements_only:
        if not a.combined_fixes or a.root_cause_fixes or a.search:
            ap.error("--statements-only: the combined fixes with the statements only; give --combined-fixes, and not "
                     "--root-cause-fixes or --search")
        a.prompt = STATEMENTS_PROMPT
    if a.num_ctx is None:       # by the statement's period, before the caller and the frozen hashes read it
        t = template.active()
        a.num_ctx = (DURATION_NUM_CTX if a.section[0] in t.sections
                     and t.period(t.statement_of(a.section[0])) == "duration" else LOOP_NUM_CTX)
    if a.a2:
        ap.error("--a2: the A2 rule book was removed; every line goes to C4")
    if a.row_order == "template":       # no seed: analyst.row_order keeps the template order for both roles
        a.row_seed = None
    if a.third_opinion_all:
        ap.error("--third-opinion-all: the tiebreaker's all-lines arm was removed")
    if a.third_model is not None and not a.third_model.strip():
        ap.error("--third-model needs a model name")
    if a.third_model and not a.third_opinion:
        ap.error("--third-model needs --third-opinion")
    if bool(a.corrections) != bool(a.corrections_sha256):
        ap.error("--corrections and --corrections-sha256 go together")
    if a.no_checker and (a.third_opinion or a.checker_feedback):
        ap.error("--no-checker: nothing is reviewed, so no --third-opinion or --checker-feedback")
    given = {}
    for x in a.model:
        role, _, name = x.partition("=")
        if role not in LOOP_MODELS or role in given or not name.strip():
            ap.error("--model takes analyst=MODEL or checker=MODEL, each role once, with a model name")
        given[role] = name
    if given and a.engine == "openrouter":
        ap.error("--model is a local model: not with --engine openrouter")
    loop = {**LOOP_MODELS, **given}       # the run's analyst and checker; recorded and frozen with the run
    # a hosted run's tiebreaker: the profile's own when it has one, else Haiku (THIRD), or the local --third-model
    # beside them (llm.Mixed below); none without --third-opinion
    hosted = ({**{r: v for r, v in HOSTED_PROFILES[a.hosted_profile].items() if r != "haiku"},
               **({"haiku": HOSTED_PROFILES[a.hosted_profile].get("haiku", THIRD["haiku"])}
                  if a.third_opinion and not a.third_model else {})}
              if a.engine == "openrouter" else None)
    unset = sorted(f"{r}.{k}" for r, v in (hosted or {}).items() for k in ("providers", "price", "allowance")
                   if k in v and v[k] is None)
    if unset:
        ap.error(f"--hosted-profile {a.hosted_profile} has placeholders {unset}: fill in the providers, prices "
                 "and allowances first")
    if hosted and not a.spend_usd > 0:
        ap.error("--engine openrouter needs --spend-usd above 0")
    # the local analyst and checker with Haiku on OpenRouter
    mixed = THIRD if a.third_opinion and not hosted and not a.third_model and "haiku" not in loop else None
    # the local tiebreaker keeps the role's name "haiku" (its ledger limit, votes and records); its model is recorded.
    # Its 80-request limit is the company ledger's role limit (the full pipeline); a standalone run without one has
    # only the run's own limits
    local = {**loop, "haiku": a.third_model} if a.third_model else loop
    if mixed and not a.spend_usd > 0:
        ap.error("--third-opinion beside the local profile runs Haiku on OpenRouter: it needs --spend-usd above 0")
    beside = {"haiku": a.third_model} if hosted and a.third_model else {}     # the local tiebreaker of a hosted run
    models = ({**{r: v["model"] for r, v in hosted.items()}, **beside} if hosted else
              {**loop, **{r: v["model"] for r, v in mixed.items()}} if mixed else local)
    cold = llm.cold_start() if a.cold_start else None
    if cold and cold["still_loaded"]:
        ap.error(f"--cold-start: still loaded after the unload: {cold['still_loaded']}")
    limits = {"dispatches": a.max_calls, "local_tokens": a.max_tokens,
              "spend_usd": a.spend_usd if hosted or mixed else 0.0,
              "batch_wall_s": a.wall_s, "company_wall_s": a.wall_s}
    meta = {"filing": a.filing, "section": a.section[0] if len(a.section) == 1 else a.section, "models": models, "engine": a.engine,
            "engines": {r: "openrouter" if r in (hosted or mixed or {}) else "ollama" for r in models}, "hosted": hosted,
            "hosted_profile": a.hosted_profile if hosted else None, **({"third": mixed} if mixed else {}),
            "probe": a.probe, **({"corrections": {"record": a.corrections, "sha256": a.corrections_sha256}}
                                 if a.corrections else {}),
            "evidence": a.probe is None, "no_a2": True, "max_calls": a.max_calls, "limits": limits,
            "plan_ledger": a.ledger if company is None else str(company.dir),
            "search": a.search, "prompt": a.prompt, "cap": a.cap, "num_ctx": a.num_ctx,
            "num_batch": llm.batch_pins(beside.values()) if hosted else llm.batch_pins(local.values()),
            "answer_cap": a.answer_cap,
            "row_order": a.row_order, "row_seed": a.row_seed, "cold_start": cold, "third_opinion": a.third_opinion,
            "checker_feedback": a.checker_feedback, "no_checker": a.no_checker, "tags_hidden": True,
            "combined_fixes": a.combined_fixes, **({"root_cause_fixes": True} if a.root_cause_fixes else {}),
            **({"statements_only": True} if a.statements_only else {}),
            "options": None if hosted and not beside else {"presence_penalty": 0.0}, "c7": "off (diagnostic only)",
            "c8": "off (--no-checker): no line is reviewed; every analyst answer stands" if a.no_checker else
                  "the checker runs the same prompt and loop, blind, with its own registry and a shuffled row "
                  "order; code compares the whole canonical decomposition; a disagreement leaves C4's placement "
                  "disputed, a checker that does not finish leaves it unreviewed; agreement is consistency, not "
                  "correctness",
            "c8_exchange": "none" + (f"; {a.third_model or 'Haiku'} votes on disputed lines (experimental)"
                                     if a.third_opinion else
                                     "; no third opinion (off)")
                           + ("; on a disputed line, and on an OUTSIDE or NOT_TOLD line the checker places, the "
                              "analyst's one retry with both answers as candidates (the checker-feedback retry)" if a.checker_feedback
                              else ""),
            "a3": "off: the automatic A3 splits stay off; the only pieces are C4's that pass rules 5 and 6",
            "rounding": "C6 tolerance: each fact on the two sides, aliases merged into "
                        "the canonical fact, gets its net coefficient, built minus filed; T = sum of |coefficient| x "
                        "half a unit of its decimals; no intersection of duplicate intervals, no rounding mode (not "
                        "Calculations 1.1)",
            "hashes": freeze_hashes(prompts=[a.prompt], models=local,
                                    num_ctx=a.num_ctx, answer_cap=a.answer_cap, identity=caller is None,
                                    hosted=hosted, third=mixed)}
    if company is not None and a.ledger:
        ap.error("a company ledger and --ledger: pass one")
    try:
        plan = company if company is not None else open_plan(a.ledger) if a.ledger else None
    except ValueError as e:
        ap.error(str(e))
    # a company ledger holds Haiku's limit across the company's statement runs; without it each run would start anew
    paid_third = a.third_opinion and (mixed or hosted) and not a.third_model      # Haiku on OpenRouter
    missing = sorted(set(THIRD_CAPS) - set(plan.role_limits)) if plan is not None and paid_third else []
    if missing:
        ap.error(f"ledger {plan.dir} has no role limit for {missing}: open it with --role-limit "
                 + " ".join(f"{r} {THIRD_CAPS[r]['calls']} {THIRD_CAPS[r]['spend_usd']}" for r in missing))
    base = plan.state() if plan is not None else None     # the company ledger before this run (company_delta)
    from . import organizer
    if hosted:                          # the organizer on the profile's analyst model (BORROWED), no local digest
        meta["organizer"] = {**organizer.role(None, borrowed(a.hosted_profile, "organizer", 0, 0)["model"]),
                             "engine": "openrouter"}
    else:
        meta["organizer"] = organizer.role(llm.model_identity({"organizer": organizer.MODEL})
                                           if caller is None else None)
        meta["organizer"]["num_batch"] = llm.batch_pins([organizer.MODEL])[organizer.MODEL]
    d = new_run("spread", meta)
    # every local chat send of the run, the organizer's included, checked against the engine's load record
    watch, llm.LOADS = llm.LOADS, llm.LoadWatch()
    own = Ledger(d / "ledger", limits, role_limits=THIRD_CAPS if paid_third else None)   # Haiku's dollars recorded
    ledger = own if plan is None else Both(own, plan)
    stop = StopFile(d / "STOP")
    company = Path(a.filing).parent.name
    out = {"meta": meta, "exceptions": [], "c4": {}, "diagnostics": {}, "ledger": ledger, "stop": stop,
           "company": company, "embedder": embedder,
           "caller": caller or (
               Caller({r: models[r] for r in hosted}, d / "calls.jsonl", a.max_calls, num_ctx=a.num_ctx, stop=stop,
                      engine="openrouter",
                      answer_cap=a.answer_cap, ledger=ledger, company=company,
                      **openrouter_kwargs(hosted, a.answer_cap, a.spend_usd)) if hosted else
               Caller(local, d / "calls.jsonl", a.max_calls, num_ctx=a.num_ctx, stop=stop,
                      answer_cap=a.answer_cap, ledger=ledger, company=company, options=meta["options"]))}
    if mixed and caller is None:        # each role on its own engine, every attempt in the same ledger
        third = Caller({r: v["model"] for r, v in mixed.items()}, d / "calls.jsonl", a.max_calls, num_ctx=a.num_ctx,
                       stop=stop, engine="openrouter", answer_cap=a.answer_cap, ledger=ledger, company=company,
                       **openrouter_kwargs(mixed, a.answer_cap, a.spend_usd))
        out["caller"] = llm.Mixed({**{r: out["caller"] for r in loop}, **{r: third for r in mixed}})
    if beside and caller is None:       # the hosted analyst and checker with the local tiebreaker, one ledger
        third = Caller(beside, d / "calls.jsonl", a.max_calls, num_ctx=a.num_ctx, stop=stop, answer_cap=a.answer_cap,
                       ledger=ledger, company=company, options=meta["options"])
        out["caller"] = llm.Mixed({**{r: out["caller"] for r in hosted}, **{r: third for r in beside}})
    error = None
    provenance.begin(meta["hashes"]["approved_resources"])    # only the frozen inventory opens from here on
    try:
        out["meta"]["filing_inputs"] = filing_inputs(Path(a.filing))
        run(a, d, out)
    except Exception as e:
        error = {"error": repr(e), "trace": traceback.format_exc()[-3000:]}
    finally:
        opened = provenance.used()
        provenance.end()
        tagless.stop()
        loads, llm.LOADS = llm.LOADS.write(d), watch
    checks = out.get("checks", [])
    if error or out.get("refusal"):      # a refused statement is not placed
        outcome = "not run"
    else:
        outcome = verdict(checks, out["exceptions"], out.get("report_checks"))
    if isinstance(out["diagnostics"].get("open_claims"), dict):     # a root-cause job: why it is not "complete", kept apart
        from . import open_claims as _oc
        out["diagnostics"]["open_claims"]["qualification"] = _oc.qualification(outcome, out["exceptions"])
    if len(out.get("sections") or ()) > 1:     # each section's own outcome; the run's is the worst of them
        out["diagnostics"]["section_outcomes"] = {
            s: verdict([c for c in checks if c["section"] in (s, "run")],
                       [x for x in out["exceptions"] if x.get("section") in (s, None)])
            for s in out["sections"]}
    for c in checks:
        print(("PASS " if c["ok"] else "FAIL ") + json.dumps(c, default=str))
    print("outcome:", outcome, error["error"] if error else out.get("refusal") or "")
    j = lambda o: json.dumps(o, indent=1, default=str)
    state = ledger.state()
    files = [("lines.json", out.get("lines", [])), ("placements.json", out.get("alloc", {})),
             ("c4.json", out["c4"]), ("claims.json", out.get("claims", [])), ("rows.json", out.get("rows", {})),
             ("checks.json", checks), ("exceptions.json", out["exceptions"]),
             ("computed_totals.json", out.get("computed_totals", {})),
             ("controls.json", out.get("controls", {})), ("report_checks.json", out.get("report_checks", [])),
             ("unallocated.json", out.get("unallocated", [])),
             ("exceptions_summary.json", {**exceptions_summary(out["exceptions"], checks, outcome, error),
                                          **({"engine_loads": loads} if loads["flags"] or loads["not_observed"]
                                             else {})}),
             ("diagnostics.json", out["diagnostics"]),
             ("outcome.json", {"outcome": outcome, "error": error,     # calls: the organizer's sends included
                               "calls": out["caller"].calls + getattr(out.get("organizer"), "calls", 0),
                               # the company ledger's change over the run, or the run's own use without one
                               "ledger": {**state, "company_delta": company_delta(
                                   state["plan"] if base is not None else state, base or {},
                                   *hosted_use(out["caller"], out.get("organizer"), out.get("embedder")))},
                               "caller_failed": out["caller"].failed or getattr(out.get("organizer"), "failed", None),
                               # the caller first, as the stop check reads them; else the organizer's, else the
                               # embedder's (role embedding)
                               "caller_failure": (getattr(out["caller"], "failure", None)
                                                  or getattr(out.get("organizer"), "failure", None)
                                                  or getattr(out.get("embedder"), "failure", None)),
                               "refusal": out.get("refusal"),
                               "hashes": meta["hashes"], "opened_resources": opened, "engine_loads": loads,
                               "filing_inputs": out["meta"].get("filing_inputs"),
                               **{k: out["meta"].get(k) for k in ("period_end", "taxonomy_year")}})]
    if out.get("search") is not None:    # pins and settings, what was left out, every search, every embedding request
        s = out["search"]
        files.append(("search.json", {"pins": s.pins, "skipped": s.skipped, "searches": s.log,
                                      "embedding_calls": getattr(out["embedder"], "calls", [])}))
    elif getattr(out.get("embedder"), "calls", None):   # the index build failed: its requests are still recorded
        files.append(("embedding_calls.json", {"embedding_calls": out["embedder"].calls}))
    for name, obj in files:
        write(d, name, j(obj))
    print(d)
    return d


if __name__ == "__main__":
    main()
