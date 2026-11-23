"""C7 repair: a Tree of Thoughts on a draft that failed C6.

A thought changes one line's placement: it moves the line to a row, splits it by fact IDs (code reads each amount by
its fact ID and computes the remainder), or marks it combined. Code holds the hard gate at every edge (the invariants
I1-I5). It computes F(x), the failed C6 checks, and shows it to the controller; F(x) never prunes. Eligibility needs
the invariants, every filed amount owned exactly once, and F empty. The generator proposes up to 3 children per node.
The critic's verdict and reason go to the generator that expands that branch, as one reviewer's fallible opinion,
never as evidence. The controller receives no explicit critic verdict; generated reasons may reflect feedback. The
critic prunes nothing. The controller ranks, stops or selects through an enum over node IDs. Beam 2, depth 2.

The C8 review reserve is taken before the tree expands, and the tree spends only what is left. At commit, code replays
the path through the gate and re-checks eligibility and the evidence (G2). C8 then reviews each changed line. Any
result other than settled withdraws the repair, and the original draft stays. The root draft is never mutated: each
node holds its own registry, derived from its parent's. The whole tree goes into the record.

The roles are injected callables, and nothing here calls a model.
"""
import copy
import json
import os
import re
import time
from decimal import Decimal
from typing import Any, TypedDict

os.environ["LANGSMITH_TRACING"] = "false"       # no trace leaves the machine
os.environ["LANGCHAIN_TRACING_V2"] = "false"
from langgraph.graph import END, START, StateGraph   # noqa: E402

from . import steps, template                          # noqa: E402
from .ledger import LimitReached, allowance_refusal    # noqa: E402
from .ledger import count as count_check               # noqa: E402
from .llm import Stopped                               # noqa: E402
from .runrec import write                              # noqa: E402
from .spread import c6, clean, gap_lines, quote_parts  # noqa: E402
from .steps import CLAIM_KINDS, QUOTE, enum, obj       # noqa: E402

# Standalone defaults, used where a run's settings leave a tree role out (the parked hosted line-up takes its local
# critic from here); the shipped line-up sets every tree role itself.
ROLE_MODELS = {"generator": "qwen3.5:4b-q4_K_M", "controller": "qwen3.5:4b-q4_K_M",
               "critic": "granite4.2:8b-q4_K_M"}
BEAM, CHILDREN, DEPTH = 2, 3, 2
GENERATOR_STEPS = 4             # the generator's tool steps per call
CRITIC_REQUESTS = CONTROLLER_REQUESTS = 1
MEANING_REQUESTS = 8            # the one meaning question: one analyst loop, loop.STEP_CAP
CHECK_RETRIES = 1               # a refused reply is shown back once, as loop.CHECK_RETRIES
REVIEW_PER_LINE = {"own_loop": 49, "c4_evidence": 7}   # C8 + exchange per changed line, provisional
TOKENS_PER_REQUEST = 3072       # provisional, not measured: 75% of a 4,096 context; the tree's placeholder, while
                                # the review's ceiling comes from review_line
SECONDS_PER_REQUEST = 20        # placeholder, not measured
TERMINATIONS = ("controller_stopped", "selected_at_depth_limit", "no_eligible", "no_admitted_child", "cap_reached",
                "refused_at_commit", "invalid", "withdrawn", "role_failed", "selected_unreviewed", "interrupted")
STATEMENTS = {"BS": "balance sheet", "IS": "income statement", "CF": "cash-flow statement"}   # named in requests
INVARIANTS = {"I1": "source amounts unchanged, read by fact ID",
              "I2": "every reference valid (line, row, fact and evidence IDs)",
              "I3": "units and periods consistent",
              "I4": "each filed amount owned at most once",
              "I5": "a split adds up to its parent and passes the shared split checks (steps.split)"}
HINT = "lines whose amounts could close the gap, computed by code; you may propose any line"
ACTIONS = ("move", "split", "combined")
FIELDS = ("line", "action", "row", "allocations", "evidence_ids", "quote", "reason", "targets")
BASE = ("line", "action", "evidence_ids", "quote", "reason", "targets")
NEEDS = {"move": BASE + ("row",), "split": BASE + ("row", "allocations"), "combined": BASE}   # each action's fields
NO_SELECT = "none"              # the controller's `select` when it expands: the schema's strings hold no null
COMBINED = "COMBINED (C7)"
NOT_CITABLE = ("search_result",)  # a disclosed search-result block is not citable: cite the passage read
SETTLED = "settled"
USAGE_KEYS = ("requests", "tokens", "embedding_requests", "embedding_tokens")
STR = {"type": "string"}
REASON = {"type": "string", "maxLength": 300}


class Stop(Exception):
    """The tree ends with one of TERMINATIONS."""

    def __init__(self, termination, why):
        super().__init__(f"{termination}: {why}")
        self.termination, self.why = termination, why


def plain(o):
    """The record as plain JSON: amounts as strings, tuples and sets as lists."""
    if isinstance(o, Decimal):
        return str(o)
    if isinstance(o, dict):
        return {str(k): plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [plain(v) for v in o]
    if isinstance(o, (set, frozenset)):
        return sorted(plain(v) for v in o)
    return o


ALIASES_MEAN = "each ID in evidence_aliases cites exactly the same text as the evidence ID it names"


def evidence_view(blocks):
    """A request's citable evidence from (ID, {kind, text}) pairs: each distinct (kind, text) once, under its first
    ID; a later ID repeating it byte for byte goes to `evidence_aliases`, naming that ID, so repeated texts do not
    overflow a small generator's request. Without repeats, only `evidence`."""
    evidence, aliases, first = {}, {}, {}
    for e, b in blocks:
        key = (b["kind"], b["text"])
        if key in first:
            aliases[e] = first[key]
        else:
            first[key], evidence[e] = e, b["text"]
    return {"evidence": evidence, **({"evidence_aliases": aliases, "evidence_aliases_mean": ALIASES_MEAN}
                                     if aliases else {})}


def shared_run(a, b):
    """The longest run of consecutive words two texts share (words: runs of lower-case letters and digits), a
    diagnostic of copying fixed before any run: shared wording does not prove influence; a paraphrase scores low."""
    x, y = re.findall(r"[a-z0-9]+", (a or "").lower()), re.findall(r"[a-z0-9]+", (b or "").lower())
    best, prev = 0, [0] * (len(y) + 1)
    for w in x:
        cur = [0] * (len(y) + 1)
        for j, v in enumerate(y):
            if w == v:
                cur[j + 1] = prev[j] + 1
                best = max(best, cur[j + 1])
        prev = cur
    return best


def core(beam=BEAM, children=CHILDREN, depth=DEPTH):
    """Core tree requests before retries: generator steps on every expanded node, one critic request per scored
    child, one controller request per level. 23 at beam 2, 3 children, depth 2."""
    expanded = 1 + beam * (depth - 1)
    return expanded * GENERATOR_STEPS + children * expanded * CRITIC_REQUESTS + depth * CONTROLLER_REQUESTS


# ---- the registry, F(x) and the gate (code) -------------------------------------------------------------

def dims_of(x):
    d = x.get("dims") or ()
    return tuple(sorted(tuple(p) for p in (d.items() if isinstance(d, dict) else d)))


def rid(x):
    """(tag, dims) of the amount a line, piece or other section's claim names: a split piece's tag, else its
    concept (spread.ident)."""
    return (x.get("tag") or x["concept"], dims_of(x))


def fixed_line(l):
    """Not a placement line: a control, a statement control a whole-statement run never places (spread.run, C3;
    lines.json's `statement_control`), or a line the organizer's inventory holds as context (`context`)."""
    return l["control"] or l.get("statement_control", False) or l.get("context", False)


def placeable(draft):
    """Lines a thought may place: not controls, with a known sign. Not a hint: any of them may be proposed."""
    return [l["concept"] for l in draft["lines"] if not fixed_line(l) and draft["signs"].get(l["concept"]) is not None]


def ratio(x):
    return Decimal(str(x.get("r", 1)))


def claims_of(draft, reg):
    """Each placed line's remainder on its row and each piece on its own row, signed by the line's sign, with the
    original facts behind each claim as spread.build writes them (a remainder: the line's fact at d and each piece's
    at -d*r; a piece: its fact at d*r), so C6's tolerance is the run's."""
    by = {l["concept"]: l for l in draft["lines"]}
    out = []
    for c, p in reg["alloc"].items():
        mine = [x for x in reg["pieces"] if x["source"] == c]
        d = draft["signs"][c]
        out.append({"fact": by[c]["fact"], "concept": c, "dims": [], **{k: by[c][k] for k in ("tag", "dims") if k in by[c]},
                    "row": p["row"], "by": p["by"],
                    "amount": by[c]["amount"] - sum((x["amount"] for x in mine), Decimal(0)), "source": c,
                    "facts": [(by[c]["fact"], d, by[c].get("decimals"))]
                    + [(x["fact"], -d * ratio(x), x.get("decimals")) for x in mine]})
    out += [{**x, "facts": [(x["fact"], draft["signs"][x["source"]] * ratio(x), x.get("decimals"))]}
            for x in reg["pieces"]]
    for x in out:
        x["signed"] = x["amount"] * draft["signs"][x["source"]]
    return out


def rebuild(draft, reg):
    """(rows, claims, checks): C5 and C6 on one registry, with the run's C6 settings for the section (draft["c6"]:
    control, canonical, filed_absent, runwide; none on a draft from a single-section spread run)."""
    open_ = {l["concept"] for l in draft["lines"] if not fixed_line(l) and l["concept"] not in reg["alloc"]}
    claims = claims_of(draft, reg)
    rows, checks = c6(draft["lines"], claims, draft["ids"], open_, draft["signs"], draft["section"],
                      **draft.get("c6", {}))
    return rows, claims, checks


def state_of(draft, reg):
    """(F(x), complete). F(x) is every failed C6 check; C6 has no non-blocking check. Complete: every line that is
    not a control is placed. With I4 at most once, every filed amount is then owned exactly once."""
    _, _, checks = rebuild(draft, reg)
    complete = all(fixed_line(l) or l["concept"] in reg["alloc"] for l in draft["lines"])
    return [c for c in checks if not c["ok"]], complete


def ids_of(failures):
    return [c["check"] for c in failures]


def placement(reg, c, amounts=False):
    """One line's placement in a registry: its row and pieces, or unresolved with why."""
    if c in reg["alloc"]:
        ps = sorted((x for x in reg["pieces"] if x["source"] == c), key=lambda x: (x["fact"], x["row"]))
        return {"row": reg["alloc"][c]["row"],
                "pieces": [{"fact": x["fact"], "row": x["row"], **({"amount": x["amount"]} if amounts else {})}
                           for x in ps]}
    return {"unresolved": reg["unresolved"].get(c, "not placed")}


def wanted(t):
    """The placement a thought asks for."""
    if t["action"] == "combined":
        return {"unresolved": COMBINED}
    return {"row": t["row"], "pieces": [{"fact": f, "row": r}
                                        for f, r in sorted((a["fact_id"], a["row"]) for a in t["allocations"])]}


def answer_of(reg):
    """One answer, however reached: rows, pieces and unresolved lines."""
    return json.dumps({"alloc": {c: p["row"] for c, p in reg["alloc"].items()},
                       "pieces": sorted([x["source"], x["fact"], x["row"]] for x in reg["pieces"]),
                       "unresolved": sorted(reg["unresolved"])}, sort_keys=True)


def statement_name(draft):
    """The statement the draft's section is on, as the requests name it (balance sheet, income statement or cash-flow
    statement); None for a section the active template does not have."""
    try:
        return STATEMENTS.get(template.active().statement_of(draft["section"]))
    except KeyError:
        return None


def recorded_pieces(draft, c):
    """The pieces of line c the run recorded (draft["recorded"]), as the root registry holds them: amount,
    coefficient r and provenance (`by`) as the run wrote them."""
    fs = set(draft.get("recorded", {}).get(c, ()))
    return {p["fact"]: p for p in draft["registry"]["pieces"] if p["source"] == c and p["fact"] in fs}


def apply(draft, reg, t, by="C7"):
    """The child's registry: a copy of the parent's, which stays unchanged, with the one line's placement replaced.
    Amounts are read by fact ID, never taken from the thought. A split the run recorded is kept as recorded: a move
    keeps its pieces where they are, a split may name a recorded piece's fact only to change its row (its amount,
    coefficient and provenance stay the run's), and a split that leaves one out, or `combined`, drops it, which I5
    refuses. `by`: the placement's provenance, C7 for a tree repair (analyst_edits passes "analyst")."""
    reg = copy.deepcopy(reg)
    c = t["line"]
    rec = recorded_pieces(draft, c)
    rows = {a["fact_id"]: a["row"] for a in t["allocations"]} if t["action"] == "split" else {}
    keep = set(rec) if t["action"] == "move" else set(rows) & set(rec)
    mine = [x for x in reg["pieces"] if x["source"] == c and x["fact"] in keep]
    reg["pieces"] = [x for x in reg["pieces"] if x["source"] != c] + [
        {**x, "row": rows.get(x["fact"], x["row"])} for x in mine]
    if t["action"] == "combined":
        reg["alloc"].pop(c, None)
        reg["unresolved"][c] = COMBINED
        return reg
    reg["unresolved"].pop(c, None)
    reg["alloc"][c] = {"row": t["row"], "by": by}
    for a in t["allocations"]:
        if a["fact_id"] in rec:
            continue
        x = draft["facts"][a["fact_id"]]
        reg["pieces"].append({"fact": a["fact_id"], "concept": x["concept"], "dims": [list(d) for d in dims_of(x)],
                              "row": a["row"], "amount": x["amount"], "source": c, "by": by})
    return reg


def references(draft, scope, t):
    """I2 on the thought, before it is applied: the line, rows, facts and evidence exist; evidence is citable."""
    by = {l["concept"]: l for l in draft["lines"]}
    errs = []
    l = by.get(t["line"])
    if l is None:
        errs.append(("I2", f"{t['line']}: not a line of the section"))
    elif fixed_line(l):
        errs.append(("I2", f"{t['line']}: a control, placed through its parts"))
    elif draft["signs"].get(t["line"]) is None:
        errs.append(("I2", f"{t['line']}: sign unknown, it cannot be placed"))
    if t["action"] != "combined" and t["row"] not in draft["ids"]:
        errs.append(("I2", f"{t['row']}: not a row of the section"))
    for a in t["allocations"]:
        if a["fact_id"] not in draft["facts"]:
            errs.append(("I2", f"{a['fact_id']}: no such fact in the filing"))
        if a["row"] not in draft["ids"]:
            errs.append(("I2", f"{a['row']}: not a row of the section"))
    for e in t["evidence_ids"]:
        b = block_of(e, scope)
        if b is None:
            errs.append(("I2", f"{e}: no such evidence in this branch"))
        elif b["kind"] in NOT_CITABLE:
            errs.append(("I2", f"{e}: a search result is not citable; cite the passage read"))
    return errs


def conflicting(draft, fid):
    x = draft["facts"][fid]
    k = (x["concept"], dims_of(x), x["date"], x["unit"])
    return any(y["amount"] != x["amount"] for i, y in draft["facts"].items()
               if i != fid and (y["concept"], dims_of(y), y["date"], y["unit"]) == k)


def consistent(draft, t):
    """I3 and I1 on the allocations: each fact at the section's date and in the line's currency; no conflicting
    duplicate; an amount the thought states, if any, equal to the filed one (compared only, never used: a model's
    number never enters a calculation)."""
    line = next(l for l in draft["lines"] if l["concept"] == t["line"])
    errs = []
    for a in t["allocations"]:
        x = draft["facts"][a["fact_id"]]
        if x["unit"] != line["unit"]:
            errs.append(("I3", f"{a['fact_id']}: in {x['unit']}, the line is in {line['unit']}"))
        if x["date"] != draft["period_end"]:
            errs.append(("I3", f"{a['fact_id']}: at {x['date']}, the section is at {draft['period_end']}"))
        if conflicting(draft, a["fact_id"]):
            errs.append(("I1", f"{a['fact_id']}: duplicate facts disagree; no amount can be read"))
        if a.get("amount") is not None:
            try:
                same = Decimal(str(a["amount"]).replace(",", "")) == x["amount"]
            except ArithmeticError:
                same = False
            if not same:
                errs.append(("I1", f"{a['fact_id']}: stated {a['amount']}, filed {x['amount']}"))
    return errs


def registry_errors(draft, reg):
    """I1, I4 and I5 on a whole registry. I4 also refuses an amount another section of the run owns (its lines and
    pieces, draft["elsewhere"]); I5 also refuses dropping a split the run recorded (draft["recorded"])."""
    facts, errs = draft["facts"], []
    by = {l["concept"]: l for l in draft["lines"]}
    for l in draft["lines"]:
        x = facts.get(l["fact"])
        if x is not None and x["amount"] != l["amount"]:
            errs.append(("I1", f"{l['concept']}: line amount {l['amount']}, its fact {x['amount']}"))
    for p in reg["pieces"]:
        x = facts.get(p["fact"])
        if x is None or x["amount"] * ratio(p) != p["amount"]:
            errs.append(("I1", f"{p['fact']}: piece amount {p['amount']} is not the filed amount"))
    owner = {rid(l) + (draft["period_end"], l["unit"]): l["concept"] for l in draft["lines"]}
    away = {rid(o) + (o["date"], o["unit"]): o["section"] for o in draft.get("elsewhere", ())}
    seen = {}
    for p in reg["pieces"]:
        x = facts.get(p["fact"], {})
        k = rid(p) + (x.get("date"), x.get("unit"))
        if k in owner:
            errs.append(("I4", f"{p['fact']}: the amount of printed line {owner[k]}, which that line owns"))
        elif k in away:
            errs.append(("I4", f"{p['fact']}: an amount section {away[k]} of the run owns; never changed here"))
        elif k in seen:
            errs.append(("I4", f"{p['fact']}: already owned by a piece of {seen[k]}"))
        else:
            seen[k] = p["source"]
    for c in sorted({p["source"] for p in reg["pieces"]}):
        ps = [p for p in reg["pieces"] if p["source"] == c]
        if c not in reg["alloc"]:
            errs.append(("I5", f"{c}: pieces with no row for the remainder"))
        # six_rules.amounts' bound (signed splits, C3b): a negative piece counts as filed, and only a split with a
        # piece signed by balances (or recorded without a basis) must leave a rest of zero or of the line's sign
        total = sum((p["amount"] for p in ps), Decimal(0))
        if c in by and any(p.get("r_basis") != "path" for p in ps):
            v = by[c]["amount"]
            if v - total != 0 and (v == 0 or (v - total > 0) != (v > 0)):
                errs.append(("I5", f"{c}: pieces {total} exceed the line's {v}"))
    same = lambda p: (p["concept"], dims_of(p), p["amount"], ratio(p), p.get("by"), p.get("decimals"))  # noqa: E731
    for c in sorted(draft.get("recorded", {})):
        rec, now = recorded_pieces(draft, c), [p for p in reg["pieces"] if p["source"] == c]
        lost = sorted(set(rec) - {p["fact"] for p in now})
        if lost or c not in reg["alloc"]:
            errs.append(("I5", f"{c}: drops the split the run recorded ({', '.join(lost) or 'the line unplaced'})"))
        moved = sorted(p["fact"] for p in now if p["fact"] in rec and same(p) != same(rec[p["fact"]]))
        if moved:
            errs.append(("I5", f"{c}: changes the split the run recorded ({', '.join(moved)}): its amount, "
                               "coefficient and provenance stay as recorded; only its row may change"))
    return errs


def splitter(tax, f, index=None, rel=None, period=None, shadow=None, log=None):
    """The shared split validation (steps.split) bound to the draft's filing: check(line, allocations, printed,
    owned, evidence) -> (pieces, remainder, errors). C7's splits pass the same checks as C4's: the mechanical ones,
    known overlaps refused, provisional membership and overlap claims, and per-claim support. `period`: the draft's
    period key (draft["period"], one section of a whole-statement run), so a piece is a fact of the line's own
    period, a duration on the income statement and cash flows; None keeps the balance-sheet date.

    The overlap policy here is the legacy one in both arms (steps.PHASES["repair split"]); steps.split refuses an
    R7 Relations. `shadow` (an R source run's R7 Relations) and `log` (a list): each split check is counted, and a
    refused one is checked once more under R7, count only: whether R7 would have relaxed a refusal and whether the
    answer would then have passed. The record goes to `log`; the tree gets the legacy result, always."""
    index = index if index is not None else steps.fact_index(f)
    rel = rel if rel is not None else steps.phase_relations(f, "repair split")

    def check(line, allocations, printed, owned, evidence):
        got = steps.split(tax, f, line, allocations, printed, owned, index, rel, evidence, period=period)
        if shadow is not None and log is not None:
            from . import open_claims
            log.append(open_claims.repair_count(tax, f, line, allocations, printed, owned, index, shadow, evidence,
                                                period, got))
        return got
    return check


def gate(draft, reg, scope, t, split=None):
    """(child registry or None, [(invariant, why)]): the hard gate at one edge. A split then goes through `split`
    (splitter): its pieces, with their claims and support, replace the naive ones. With no split check connected a
    split is refused, never admitted on the weaker checks alone."""
    errs = references(draft, scope, t)
    if errs:
        return None, errs
    errs = consistent(draft, t)
    child = apply(draft, reg, t)
    errs += registry_errors(draft, child)
    if errs or t["action"] != "split":
        return (None if errs else child), errs
    if split is None:
        return None, [("I5", "no shared split check (steps.split) is connected; a split cannot be admitted")]
    line = next(l for l in draft["lines"] if l["concept"] == t["line"])
    rec = recorded_pieces(draft, t["line"])        # checked by the run already, kept as recorded (apply)
    owned = {rid(p) for p in child["pieces"] if p["source"] != t["line"] or p["fact"] in rec} | {
        rid(o) for o in draft.get("elsewhere", ())}      # other sections' amounts
    evidence = {e: b["text"] for e, b in scope.items() if b["kind"] not in NOT_CITABLE}
    new = [a for a in t["allocations"] if a["fact_id"] not in rec]
    pieces, _, bad = (split(line, new, [l["concept"] for l in draft["lines"]], owned, evidence) if new
                      else ([], None, []))
    if bad:
        return None, [("I5", w) for w in bad]
    child["pieces"] = [x for x in child["pieces"] if x["source"] != t["line"] or x["fact"] in rec] + [
        {**p, "dims": [list(d) for d in p["dims"]], "by": "C7"} for p in pieces]
    return child, []


def block_of(e, scope):
    """The branch block an evidence ID cites: its own, or else the first citable block that prints it as a fact ID
    in brackets, as the spread counts a fact ID shown by a tool as evidence (a role may cite the fact ID instead of
    its block). None when no block of the branch shows it."""
    if e in scope:
        return scope[e]
    shown = re.compile(rf"\[(?:[^\]\[]*, )?{re.escape(e)}(?:, [^\]\[]*)?\]")
    return next((b for b in scope.values() if b["kind"] not in NOT_CITABLE and shown.search(b["text"])), None)


def evidence_errors(t, scope):
    """G2 at commit: every cited ID exists and is citable in the branch, and every part of the quote occurs in one
    cited block (spread.quote_parts and clean, as spread.quote_found per block)."""
    errs, texts = [], []
    for e in t["evidence_ids"]:
        b = block_of(e, scope)
        if b is None or b["kind"] in NOT_CITABLE:
            errs.append(f"{t['line']}: {e} is not citable evidence of this branch")
        else:
            texts.append(clean(b["text"]))
    parts = quote_parts(t["quote"])
    if not parts or not any(all(p in x for p in parts) for x in texts):
        errs.append(f"{t['line']}: the quote is not found in the cited evidence")
    return errs


def draft_errors(draft):
    """The root's own checks; a failure is a code fault, not a tree outcome."""
    reg, ids, signs = draft["registry"], draft["ids"], draft["signs"]
    by = {l["concept"]: l for l in draft["lines"]}
    errs = []
    for l in draft["lines"]:
        c = l["concept"]
        if l["control"]:
            if signs.get(c) is None:
                errs.append(f"{c}: a control with no sign")
            if c in reg["alloc"] or c in reg["unresolved"]:
                errs.append(f"{c}: a control is placed through its parts")
        elif fixed_line(l):
            if c in reg["alloc"] or c in reg["unresolved"]:
                errs.append(f"{c}: a statement control is never placed")
        elif (c in reg["alloc"]) == (c in reg["unresolved"]):
            errs.append(f"{c}: must be placed or unresolved, once")
        elif c in reg["alloc"] and signs.get(c) is None:
            errs.append(f"{c}: placed with no sign")
    for c, p in reg["alloc"].items():
        if c not in by or p["row"] not in ids:
            errs.append(f"{c}: placed on {p['row']}, not a line and row of the section")
    for p in reg["pieces"]:
        x = draft["facts"].get(p["fact"])
        if x is None or p["row"] not in ids or p["source"] not in by:
            errs.append(f"{p['fact']}: a piece with an unknown fact, row or line")
        elif x["unit"] != by[p["source"]]["unit"] or x["date"] != draft["period_end"]:
            errs.append(f"{p['fact']}: a piece in another unit or period")
    if errs:
        return errs
    return [f"{i}: {w}" for i, w in registry_errors(draft, reg)]


def period_of(ctx):
    """A context's period as one comparable string: the date of an instant, 'start--end' of a duration."""
    return ctx["instant"] if ctx.get("instant") else f"{ctx.get('start')}--{ctx.get('end')}"


def key_period(key):
    """A spread run's period key (["instant", date] or ["duration", start, end]) as period_of dates a fact."""
    return key[1] if key[0] == "instant" else f"{key[1]}--{key[2]}"


def draft_of(f, section, ids, lines, signs, alloc, unresolved, evidence=None, period=None, pieces=(), elsewhere=(),
             c6_args=None):
    """The tree's input from a spread run (spread.run's lines, signs, alloc and unresolved) and its Filing. The facts
    are every numeric fact of the filing's own entity at an instant, in one currency, by fact ID. Pieces are none:
    A3 is off, so no split is committed before C7. `evidence`: ID to text, citable by every branch.

    One section of a whole-statement run (repair_run --section): `period` is the run's period key (["instant", date]
    or ["duration", start, end]); facts at durations are kept too, each dated by period_of, and the section's date is
    the key's, so a fact of another period is refused (I3). `pieces`: the splits the run recorded, kept as recorded
    (`recorded`: each line's piece facts, which no edge may drop). `elsewhere`: amounts other sections own ({concept,
    dims, date, unit, section}). `c6_args`: the run's C6 settings for the section (control, canonical, filed_absent,
    runwide)."""
    facts = {}
    for x in f.facts:
        ctx = f.contexts.get(x["context"]) or {}
        cur = f.currency(x["unit"]) if x.get("unit") else None
        if (not x.get("id") or cur is None or ctx.get("entity") != f.entity
                or (ctx.get("instant") is None and (period is None or not ctx.get("end")))):
            continue
        try:
            amount = Decimal(x["value"])
        except ArithmeticError:
            continue
        facts[x["id"]] = {"concept": x["concept"], "dims": sorted([d, m] for d, m in ctx["dims"].items()),
                          "unit": cur, "date": period_of(ctx), "amount": amount,
                          **({"decimals": x.get("decimals")} if period else {})}
    why = unresolved if isinstance(unresolved, dict) else {c: "unresolved" for c in unresolved}
    end = f.period_end if period is None else key_period(period)
    recorded = {}
    for p in pieces:
        recorded.setdefault(p["source"], []).append(p["fact"])
    return {"section": section, "period_end": end, "ids": list(ids), "lines": copy.deepcopy(lines),
            "signs": dict(signs), "facts": facts, "evidence": dict(evidence or {}),
            **({"period": list(period), "recorded": recorded, "elsewhere": [dict(o) for o in elsewhere],
                "c6": dict(c6_args or {})} if period else {}),
            "registry": {"alloc": {c: dict(p) for c, p in alloc.items()}, "pieces": [dict(p) for p in pieces],
                         "unresolved": dict(why)}}


# ---- structural checks on role replies --------------------------------------------------------------------

def support_entry_ok(s):
    """A support entry's shape, as steps.allocation_items offers it: {about: [{kind, target}], evidence_id, quote}."""
    return (isinstance(s, dict) and isinstance(s.get("about"), list) and isinstance(s.get("evidence_id"), str)
            and isinstance(s.get("quote"), str)
            and all(isinstance(c, dict) and c.get("kind") in CLAIM_KINDS and isinstance(c.get("target"), str)
                    for c in s["about"]))


def thought_errors(t):
    """Each action's own fields (NEEDS): a move a row, a split a row and allocations, combined neither."""
    if not isinstance(t, dict):
        return ["not an object"]
    if t.get("action") not in ACTIONS:
        return [f"`action`: one of {', '.join(ACTIONS)}"]
    miss = [k for k in NEEDS[t["action"]] if k not in t]
    if miss:
        return [f"missing {', '.join(miss)}"]
    t = {"row": None, "allocations": [], **t}
    errs = []
    if not isinstance(t["line"], str) or not t["line"]:
        errs.append("`line`: a concept")
    if not isinstance(t["allocations"], list) or not all(
            isinstance(a, dict) and isinstance(a.get("fact_id"), str) and isinstance(a.get("row"), str)
            for a in t["allocations"]):
        errs.append("each allocation: `fact_id` and `row`")
    elif not all(isinstance(a.get("support", []), list) and all(support_entry_ok(s) for s in a.get("support", []))
                 for a in t["allocations"]):
        errs.append("each support entry: `about` (claims {kind, target}), `evidence_id` and `quote`")
    if not isinstance(t["evidence_ids"], list) or not t["evidence_ids"] or not all(
            isinstance(e, str) for e in t["evidence_ids"]):
        errs.append("`evidence_ids`: at least one ID")
    if not isinstance(t["quote"], str) or not t["quote"].strip():
        errs.append("`quote`: words of the cited evidence")
    if not isinstance(t["reason"], str):
        errs.append("`reason`: text")
    if not isinstance(t["targets"], list) or not t["targets"] or not all(isinstance(x, str) for x in t["targets"]):
        errs.append("`targets`: at least one failure ID")
    if errs:
        return errs
    if t["action"] == "combined":
        if t["row"] is not None or t["allocations"]:
            errs.append("`combined` takes no row and no allocations")
    elif not isinstance(t["row"], str):
        errs.append(f"`{t['action']}` needs a row")
    elif t["action"] == "move" and t["allocations"]:
        errs.append("`move` places the whole line: no allocations")
    elif t["action"] == "split" and not t["allocations"]:
        errs.append("`split` needs at least one allocation")
    return errs


def reply_errors(reply, n, reg, scope):
    """Code's structural checks on a generator reply (refused, then resent once): the shape, each action's fields,
    distinct changes, no change that leaves the line as it is, and no disclosed block rewriting one the branch holds.
    References are the gate's (I2); an unknown target is recorded, not refused."""
    if not isinstance(reply, dict) or not isinstance(reply.get("thoughts"), list):
        return ["the reply must be an object with a list `thoughts`"]
    ts, errs = reply["thoughts"], []
    if not 1 <= len(ts) <= n:
        errs.append(f"give 1 to {n} thoughts, not {len(ts)}")
    d = reply.get("disclosed", {})
    if not isinstance(d, dict) or not all(isinstance(b, dict) and isinstance(b.get("kind"), str)
                                          and isinstance(b.get("text"), str) for b in d.values()):
        errs.append("`disclosed`: ID to {kind, text}")
    else:
        errs += [f"{e}: disclosed again with other text" for e, b in d.items()
                 if e in scope and scope[e]["text"] != b["text"]]
    seen = set()
    for i, t in enumerate(ts, 1):
        bad = thought_errors(t)
        errs += [f"thought {i}: {x}" for x in bad]
        if bad:
            continue
        t = normal(t)
        s = (t["line"], json.dumps(wanted(t), sort_keys=True))
        if s in seen:
            errs.append(f"thought {i}: the same change as an earlier thought")
        seen.add(s)
        if wanted(t) == placement(reg, t["line"]):
            errs.append(f"thought {i}: {t['line']} is already placed that way")
    return errs


def usable(reply, errs):
    """The thoughts of a reply with no error of their own, when every error belongs to one thought; else None."""
    if any(not e.startswith("thought ") for e in errs):
        return None
    ts = reply["thoughts"]
    bad = {i for i in range(1, len(ts) + 1) if any(e.startswith(f"thought {i}: ") for e in errs)}
    return [t for i, t in enumerate(ts, 1) if i not in bad] or None


def normal(t):
    """A checked thought with every field: row None and no allocations where its action takes none; support kept."""
    t = {"row": None, "allocations": [], **t}
    return {**{k: t[k] for k in FIELDS},
            "allocations": [{"fact_id": a["fact_id"], "row": a["row"],
                             **({"amount": a["amount"]} if a.get("amount") is not None else {}),
                             **({"support": [{"about": [{"kind": c["kind"], "target": c["target"]} for c in s["about"]],
                                              "evidence_id": s["evidence_id"], "quote": s["quote"]}
                                             for s in a["support"]]} if a.get("support") else {})}
                            for a in t["allocations"]]}


def controller_errors(reply, req):
    """Code's checks on a controller reply: node IDs from its enums only, a strict order over every offered node, the
    decision the level allows, and a reason."""
    if not isinstance(reply, dict):
        return ["the reply must be an object"]
    ids = [k["id"] for k in req["nodes"]]
    errs = []
    order = reply.get("order")
    if not (isinstance(order, list) and all(isinstance(x, str) for x in order)
            and len(set(order)) == len(order) and sorted(order) == sorted(ids)):
        errs.append(f"`order`: each of {', '.join(ids)} once, best first")
    if not isinstance(reply.get("reason"), str) or not reply["reason"].strip():
        errs.append("`reason`: why this order")
    if req["task"] == "select":
        if reply.get("select") not in ids:
            errs.append(f"`select`: one of {', '.join(ids)}")
        elif isinstance(order, list) and order and order[0] != reply["select"]:
            errs.append("`select` heads `order`")
    elif reply.get("decision") not in req["decisions"]:
        errs.append(f"`decision`: one of {', '.join(req['decisions'])}")
    elif reply["decision"] == "stop" and reply.get("select") not in req["eligible"]:
        errs.append(f"`stop` names an eligible node in `select`: one of {', '.join(req['eligible'])}")
    elif reply["decision"] == "expand" and reply.get("select") not in (None, NO_SELECT):
        errs.append(f"`expand` takes no node: `select` \"{NO_SELECT}\"")
    return errs


def generator_schema(draft, n):
    """The bound form llm.check_schema accepts: one branch per action with that action's fields only.
    Counts the subset cannot state (at least one thought, ID or target) are code's checks (reply_errors)."""
    ids = list(draft["ids"])
    about = {"type": "array", "items": obj({"kind": enum(CLAIM_KINDS), "target": STR})}
    alloc = obj({"fact_id": STR, "row": enum(ids),
                 "support": {"type": "array", "items": obj({"about": about, "evidence_id": STR, "quote": QUOTE})}})
    base = {"line": enum(placeable(draft)), "evidence_ids": {"type": "array", "items": STR}, "quote": QUOTE,
            "reason": REASON, "targets": {"type": "array", "items": STR}}
    branches = [obj({"action": steps.one("move"), "row": enum(ids), **base}),
                obj({"action": steps.one("split"), "row": enum(ids), "allocations": {"type": "array", "items": alloc},
                     **base}),
                obj({"action": steps.one("combined"), **base})]
    return obj({"thoughts": {"type": "array", "items": {"anyOf": branches}, "maxItems": n}})


def controller_schema(ids, eligible, final):
    """The bound form: node IDs as string enums, the order exactly as long as the nodes offered. At rank, `choice`
    binds the decision to its select: expand with NO_SELECT, or stop with an eligible node, a branch offered only
    when a node is eligible. The role wrapper turns it into the flat decision and select that controller_errors
    checks."""
    order = {"type": "array", "items": enum(ids), "minItems": len(ids), "maxItems": len(ids)}
    if final:
        return obj({"select": enum(ids), "order": order, "reason": REASON})
    choice = [obj({"action": steps.one("expand"), "select": steps.one(NO_SELECT)})]
    if eligible:
        choice.append(obj({"action": steps.one("stop"), "select": enum(eligible)}))
    return obj({"order": order, "choice": {"anyOf": choice}, "reason": REASON})


# ---- the budget -----------------------------------------------------------------------------------------

def usage_errors(usage):
    """Usage counts as reported: every one of USAGE_KEYS, a nonnegative integer (bools and floats refused, never
    coerced; a missing counter is not a zero)."""
    if not isinstance(usage, dict):
        return [f"usage is not a mapping: {usage!r}"]
    return ([f"{k}: missing (an explicit 0 is required)" for k in USAGE_KEYS if k not in usage]
            + [f"{k}: {usage[k]!r} is not a nonnegative integer" for k in USAGE_KEYS
               if k in usage and (isinstance(usage[k], bool) or not isinstance(usage[k], int) or usage[k] < 0)])


def refusal(e):
    """The refusal behind an exception, if any: the ledger's LimitReached (a wrapper's Caller or Embedder raises its
    own CapReached from it) or the stop switch (llm.Stopped), both ending the tree as cap_reached."""
    while e is not None:
        if isinstance(e, (LimitReached, Stopped)):
            return str(e)
        e = e.__cause__
    return None


class Meter:
    """The ledger a role's wrapper gets for one role call: the interface of ledger.Ledger (reserve, before_send,
    commit, commit_unknown, release), bound to the call's pool and ceiling. A wrapper hands it to its Caller and
    Embedder as their `ledger`, so every transport attempt, an embedding request or a retry included, is reserved
    against the call's ceiling and the pool before it is sent, then forwarded to the batch's shared ledger when there
    is one. A refusal raises ledger.LimitReached, as the shared ledger does; the call then stops the tree
    (`cap_reached`). `tokens`, when set, is the call's own token ceiling: the tokens it has spent (unknown usage
    counted at its reservation) and its open reservations must fit it, so one review line cannot draw on another
    line's reserve."""

    def __init__(self, budget, pool, role, ceiling, tokens=None):
        self.budget, self.pool, self.role, self.ceiling, self.tokens = budget, pool, role, ceiling, tokens
        self.open = {}                      # reservation ID -> {requests, tokens, sent, role}
        self.attempts = 0                   # attempts reserved by this call, settled or open
        self.spent = 0                      # tokens this call's settled attempts used, unknown ones in full
        self.refused = None

    def refuse(self, why):
        self.refused = why
        self.budget.events.append({"event": "refused", "pool": self.pool, "role": self.role, "why": why})
        raise LimitReached(why)

    def reserve(self, company, role, tokens, usd=0.0, requests=1, local=True):
        count_check(tokens, "tokens")
        count_check(requests, "requests", least=1)
        left, held = self.budget.left(self.pool), self.budget.held[self.pool]
        if self.attempts + requests > self.ceiling:
            self.refuse(f"{self.role}: attempt {self.attempts + requests} over its ceiling of {self.ceiling}")
        if self.tokens is not None:
            held_here = sum(r["tokens"] for r in self.open.values())
            if self.spent + held_here + tokens > self.tokens:
                self.refuse(f"{self.role}: {tokens} tokens over its token ceiling of {self.tokens} ({self.spent} "
                            f"spent, {held_here} reserved)")
        if held["requests"] + requests > left["requests"] or held["tokens"] + tokens > left["tokens"]:
            self.refuse(f"{self.role}: {requests} attempt(s) of {tokens} tokens do not fit what is left of the "
                        f"{self.pool} pool ({left}; {held['requests']} attempt(s), {held['tokens']} tokens reserved "
                        f"by calls in flight)")
        shared = self.budget.ledger
        rid = (shared.reserve(self.budget.company if company is None else company, role, tokens, usd, requests, local)
               if shared is not None else f"m{len(self.budget.events)}.{len(self.open)}")
        self.open[rid] = {"requests": requests, "tokens": tokens, "sent": 0, "role": role}
        self.hold(self.open[rid], 1)
        self.attempts += requests
        return rid

    def hold(self, r, sign):
        """Add (1) or take back (-1) a reservation in the pool's outstanding total, which every meter on the pool
        checks before it reserves."""
        h = self.budget.held[self.pool]
        h["requests"] += sign * r["requests"]
        h["tokens"] += sign * r["tokens"]

    def before_send(self, rid):
        r = self.open.get(rid)
        if r is None:
            raise ValueError(f"no open reservation {rid}")
        wall = self.budget.limits["wall_s"] - (self.budget.clock() - self.budget.started)
        if wall <= 0:
            self.refuse(f"{self.role}: wall time over the limit of {self.budget.limits['wall_s']} s")
        if r["sent"] >= r["requests"]:
            self.refuse(f"{self.role}: unreserved attempt on {rid}")
        left = self.budget.ledger.before_send(rid) if self.budget.ledger is not None else wall
        r["sent"] += 1
        return min(left, wall)

    def kind(self, r):
        return "embedding" if r["role"] == "embedding" else "chat"

    def commit(self, rid, tokens=0, usd=0.0):
        count_check(tokens, "tokens")
        r = self.open.pop(rid)
        self.hold(r, -1)
        self.spent += tokens
        try:
            if self.budget.ledger is not None:
                self.budget.ledger.commit(rid, tokens, usd)
        finally:
            self.budget.spend(self.pool, self.role, r["sent"], tokens, self.kind(r))

    def commit_unknown(self, rid, why):
        r = self.open.pop(rid)
        self.hold(r, -1)
        self.spent += r["tokens"]
        try:
            if self.budget.ledger is not None:
                self.budget.ledger.commit_unknown(rid, why)
        finally:
            self.budget.spend(self.pool, self.role, max(r["sent"], 1), r["tokens"], self.kind(r), why=f"unknown: {why}")

    def release(self, rid, why):
        r = self.open.get(rid)
        if r is not None and r["sent"]:
            raise ValueError(f"{rid} has {r['sent']} attempt(s) sent; commit it, do not release it")
        if self.budget.ledger is not None:
            self.budget.ledger.release(rid, why)
        if r is not None:
            self.hold(self.open.pop(rid), -1)
            self.attempts -= r["requests"]

    def claim_429(self, company, role, wait_s):
        """One recovery of an explicit HTTP 429 against the job allowance (llm.Caller.claim_429): through the batch's
        shared ledger when there is one, so a recovery in any tree or review role call counts against the same
        company allowance as the spread's, across role calls and child commands; without one (fakes, offline
        tests), against this Budget's own count under ledger.RATE_LIMIT_ALLOWANCE. The decision goes into the
        run's events either way."""
        shared = self.budget.ledger
        if shared is not None:
            reason, used = shared.claim_429(self.budget.company if company is None else company, role, wait_s)
        else:
            mine = self.budget.rate_limit_used
            refused = allowance_refusal(mine, wait_s)
            if refused is None:
                mine["recoveries"] += 1
                mine["wait_s"] = round(mine["wait_s"] + wait_s, 3)
            reason, used = (None, dict(mine)) if refused is None else \
                (refused[0], {**mine, "why": f"budget: {refused[1]}"})
        self.budget.events.append({"event": "rate_limit_claim", "pool": self.pool, "role": self.role, "wait_s": wait_s,
                                   "granted": reason is None, **({"reason": reason} if reason else {}),
                                   "used": dict(used)})
        return reason, used

    def close(self, why):
        """At the call's end, however it ended: an attempt sent and never settled is counted in full as unknown; one
        never sent is released. Returns how many were left open."""
        n = len(self.open)
        for rid, r in list(self.open.items()):
            if r["sent"]:
                self.commit_unknown(rid, f"not settled by the wrapper: {why}")
            else:
                self.release(rid, f"not sent: {why}")
        return n


class Counted:
    """A ledger as its user sees it, counting what was physically sent: a send is counted only once the reservation
    and before_send have both succeeded, the last step before the request is handed to transport
    (llm.Caller.reserve_attempt, llm.OllamaEmbedder.reserve). An attempt refused before that is recorded in `unsent`,
    never counted as a send, while a Caller's own `calls` counts it. A sent request that then fails or is
    interrupted stays counted; its use is charged by the ledger, in full when unknown. Every other attribute is the
    wrapped ledger's (a Meter's ceiling and refusal included)."""

    def __init__(self, ledger):
        self.ledger, self.roles = ledger, {}
        self.sends, self.unsent = {"chat": 0, "embedding": 0}, []

    def __getattr__(self, name):
        return getattr(self.ledger, name)

    def reserve(self, company, role, tokens, *a, **k):
        try:
            rid = self.ledger.reserve(company, role, tokens, *a, **k)
        except LimitReached as e:
            self.unsent.append({"stage": "reserve", "role": role, "why": str(e)})
            raise
        self.roles[rid] = role
        return rid

    def before_send(self, rid):
        try:
            left = self.ledger.before_send(rid)
        except LimitReached as e:
            self.unsent.append({"stage": "before_send", "role": self.roles.get(rid), "why": str(e)})
            raise
        self.sends["embedding" if self.roles.get(rid) == "embedding" else "chat"] += 1
        return left


class Budget:
    """C7's operating limit per failed check. The review reserve for the most lines a path can change is taken first,
    the tree spends only what is left, and C8 draws the reserve down; neither pool is charged for the other's use. A
    call starts only if its ceiling (its requests per dispatch times `dispatches`) fits what is left of its pool, a
    nested call's ceiling counted in.

    Each role call gets a Meter: every transport attempt, embedding requests included, is reserved against the
    call's ceiling and its pool before it is sent, outstanding reservations of every call on the pool counted in,
    and through the batch's shared ledger (`ledger`, ledger.Ledger) when there is one; the pool is charged with what
    the attempts used. The wrapper gets the meter as Counted: the call's physical sends and its attempts refused
    before sending go on its entry and into the run's `sends` and `unsent`. Every role reports all of USAGE_KEYS.
    Only without a shared ledger (fakes, offline tests) may a wrapper with no attempts through its meter be
    charged from the usage it reports, embedding requests counted against the ceiling; with a shared ledger,
    requests reported with none through the meter are a fault. A call that ends in any exception is charged its
    ceiling in full as unknown, unless its meter accounted for it, and the failure is recorded. Use over the
    ceiling, a pool overdrawn, the wall time spent or the stop switch stops the tree (`cap_reached`), its trace
    kept; a role that fails, or reports usage that is not valid, stops it as `role_failed`.

    `review_line` ({requests, tokens}): the review wrapper's own ceiling per changed line, in physical requests
    (chat attempts and query embeddings) and the tokens their reservations take, not multiplied again by
    `dispatches`. With it, the reserve is that ceiling times the lines and a review call's ceiling is that ceiling;
    without it, the provisional REVIEW_PER_LINE figures apply."""

    def __init__(self, arm="own_loop", dispatches=1, beam=BEAM, children=CHILDREN, depth=DEPTH, limits=None,
                 clock=time.monotonic, lines=None, ledger=None, company=None, review_line=None):
        if arm not in REVIEW_PER_LINE:
            raise ValueError(f"arm {arm!r}: one of {', '.join(REVIEW_PER_LINE)}")
        self.arm, self.dispatches, self.clock = arm, dispatches, clock
        self.ledger, self.company, self.review_line = ledger, company, review_line
        n = depth if lines is None else lines
        if review_line:
            self.reserve = {k: review_line[k] * n for k in ("requests", "tokens")}
        else:
            review = REVIEW_PER_LINE[arm] * n * dispatches
            self.reserve = {"requests": review, "tokens": review * TOKENS_PER_REQUEST}
        tree = (core(beam, children, depth) + MEANING_REQUESTS) * dispatches
        total = tree + self.reserve["requests"]
        self.derived = {"requests": total, "tokens": tree * TOKENS_PER_REQUEST + self.reserve["tokens"],
                        "wall_s": total * SECONDS_PER_REQUEST}
        self.limits = {**self.derived, **(limits or {})}
        self.spent = {"tree": {"requests": 0, "tokens": 0}, "review": {"requests": 0, "tokens": 0}}
        self.inflight = {pool: {"requests": 0, "tokens": 0} for pool in ("tree", "review")}   # ceilings of open calls
        self.held = {pool: {"requests": 0, "tokens": 0} for pool in ("tree", "review")}   # open reservations
        self.by_role = {}
        self.sends = {pool: {"chat": 0, "embedding": 0} for pool in ("tree", "review")}  # physical, by Counted
        self.unsent = []                    # attempts refused before sending, each with its pool and call
        self.rate_limit_used = {"recoveries": 0, "wait_s": 0.0}    # 429 recoveries granted, without a shared ledger
        self.started = clock()
        self.short = None
        if any(self.reserve[k] > self.limits[k] for k in ("requests", "tokens")):
            self.short = (f"the limit {self.limits} cannot hold the review reserve {self.reserve}; "
                          "the tree does not expand")
        self.events = [{"event": "reserve", "pool": "review", **self.reserve, "limits": self.limits,
                        "derived": self.derived, "arm": arm, "dispatches": dispatches, "review_line": review_line,
                        "shared_ledger": None if ledger is None else str(ledger.dir),
                        **({"short": self.short} if self.short else {})}]

    def left(self, pool):
        cap = self.reserve if pool == "review" else {k: self.limits[k] - self.reserve[k] for k in self.reserve}
        return {k: cap[k] - self.spent[pool][k] for k in cap}

    def ceiling(self, pool, steps):
        """A call's ceiling {requests, tokens}: the review line's own for a review call when it is set, otherwise its
        requests per dispatch times `dispatches`, at TOKENS_PER_REQUEST each."""
        if pool == "review" and self.review_line:
            return {k: self.review_line[k] for k in ("requests", "tokens")}
        need = steps * self.dispatches
        return {"requests": need, "tokens": need * TOKENS_PER_REQUEST}

    def fits(self, pool, need):
        left, held = self.left(pool), self.inflight[pool]
        return all(need[k] + held[k] <= left[k] for k in ("requests", "tokens"))

    def wall(self, role):
        spent = self.clock() - self.started
        if spent > self.limits["wall_s"]:
            raise Stop("cap_reached", f"wall time {spent:.0f} s over the limit of {self.limits['wall_s']} s at {role}")

    def role(self, role):
        return self.by_role.setdefault(role, {"calls": 0, **{k: 0 for k in USAGE_KEYS}})

    def spend(self, pool, role, requests, tokens, kind="chat", why=None):
        """Charge one pool with attempts sent and their tokens (`kind`: chat or embedding)."""
        self.spent[pool]["requests"] += requests
        self.spent[pool]["tokens"] += tokens
        r = self.role(role)
        r["requests" if kind == "chat" else "embedding_requests"] += requests
        r["tokens" if kind == "chat" else "embedding_tokens"] += tokens
        self.events.append({"event": "charge", "pool": pool, "role": role, "dispatches": requests, "tokens": tokens,
                            "kind": kind, **({"why": why} if why else {})})

    def count(self, pool, role, led, entry):
        """A call's physical sends and its attempts refused before sending (Counted), on its entry and in the run's
        totals, however the call ended."""
        entry["sends"] = dict(led.sends)
        for k, n in led.sends.items():
            self.sends[pool][k] += n
        if led.unsent:
            entry["unsent"] = list(led.unsent)
            self.unsent += [{"pool": pool, "call": role, **u} for u in led.unsent]

    def call(self, pool, role, steps, fn, entry):
        """fn(ledger) -> (reply, usage), `ledger` the call's meter as Counted. The reply, the usage and any failure go
        into `entry` before any stop, so the trace is kept."""
        if self.short:
            raise Stop("cap_reached", self.short)
        self.wall(role)
        ceil = self.ceiling(pool, steps)
        need = ceil["requests"]
        if not self.fits(pool, ceil):
            self.events.append({"event": "refused", "pool": pool, "role": role, "ceiling": ceil,
                                "left": self.left(pool), "in_flight": dict(self.inflight[pool])})
            entry.update(pool=pool, ceiling=need, refused="does not fit")
            raise Stop("cap_reached", f"{role}: its ceiling of {ceil} does not fit what is left of the "
                                      f"{pool} pool ({self.left(pool)}, {self.inflight[pool]} in flight)")
        entry.update(pool=pool, ceiling=need)
        meter = Meter(self, pool, role, need, ceil["tokens"] if pool == "review" and self.review_line else None)
        led = Counted(meter)
        self.role(role)["calls"] += 1
        for k in ceil:
            self.inflight[pool][k] += ceil[k]
        try:
            reply, usage = fn(led)
        except BaseException as e:
            left_open = meter.close(f"{type(e).__name__}: {e}")
            refused = meter.refused or refusal(e)
            if not meter.attempts and not refused:      # nothing went through the meter: what was sent is unknown
                self.spend(pool, role, need, ceil["tokens"], why="interrupted: counted in full")
            entry.update(usage=(f"interrupted: {meter.attempts} attempt(s) through the ledger, {left_open} left open"
                                if meter.attempts else "refused before sending" if refused else
                                "interrupted: counted in full"),
                         failure=f"{type(e).__name__}: {e}")
            if isinstance(e, Stop) or not isinstance(e, Exception):
                raise
            if refused:
                raise Stop("cap_reached", f"{role}: {refused}") from e
            raise Stop("role_failed", f"{role}: {type(e).__name__}: {e}") from e
        finally:
            for k in ceil:
                self.inflight[pool][k] -= ceil[k]
            self.count(pool, role, led, entry)
        entry.update(reply=reply, usage=usage)
        bad = usage_errors(usage)
        left_open = meter.close("the role returned")
        if left_open:
            bad.append(f"{left_open} reservation(s) left open by the wrapper, counted in full")
        reported = 0 if bad else usage.get("requests", 0) + usage.get("embedding_requests", 0)
        if not bad and meter.attempts and reported > meter.attempts:
            bad.append(f"reported {reported} requests, {meter.attempts} reserved through the ledger")
        elif not bad and not meter.attempts:
            if self.ledger is not None and reported:
                bad.append("requests reported with none reserved through the ledger")
            else:
                self.spend(pool, role, usage.get("requests", 0), usage.get("tokens", 0))
                if usage.get("embedding_requests") or usage.get("embedding_tokens"):
                    self.spend(pool, role, usage.get("embedding_requests", 0), usage.get("embedding_tokens", 0),
                               "embedding")
        if bad:
            if not meter.attempts:
                self.spend(pool, role, need, ceil["tokens"], why="usage not valid: counted in full")
            entry["usage_errors"] = bad
            raise Stop("role_failed", f"{role}: " + "; ".join(bad))
        dispatched = meter.attempts or reported
        if dispatched > need:
            raise Stop("cap_reached", f"{role} used {dispatched} requests, embeddings included, over its ceiling "
                                      f"of {need}")
        used = meter.spent if meter.attempts else usage.get("tokens", 0) + usage.get("embedding_tokens", 0)
        if meter.tokens is not None and used > meter.tokens:
            raise Stop("cap_reached", f"{role} used {used} tokens, embeddings included, over its token ceiling "
                                      f"of {meter.tokens}")
        left = self.left(pool)
        if left["requests"] < 0 or left["tokens"] < 0:
            raise Stop("cap_reached", f"{role} overdrew the {pool} pool: {left}")
        self.wall(role)
        return reply

    def state(self):
        return {"arm": self.arm, "dispatches": self.dispatches, "limits": self.limits, "derived": self.derived,
                "review_line": self.review_line, "reserve": self.reserve, "spent": copy.deepcopy(self.spent),
                "left": {"tree": self.left("tree"), "review": self.left("review")},
                "by_role": copy.deepcopy(self.by_role), "sends": copy.deepcopy(self.sends),
                "unsent": list(self.unsent), "events": list(self.events),
                "elapsed_s": self.clock() - self.started}


# ---- the tree -------------------------------------------------------------------------------------------

def new_node(nid, parent, depth, thought, scope):
    return {"id": nid, "parent": parent, "depth": depth, "thought": thought, "status": "proposed", "rejected": [],
            "why": [], "registry": None, "failures": None, "failures_change": None, "targets": None,
            "complete": None, "eligible": False, "changed_lines": [], "flags": None, "scored": False, "skipped": None,
            "critic": None, "fixed_rank": None, "controller_rank": None, "expanded": False, "same_answer_as": None,
            "scope": scope}


class Tree:
    """The tree's state: nodes, the beam, every call and the pruning record. The graph's nodes are its methods; each
    returns the next route, or raises Stop."""

    def __init__(self, draft, roles, models, budget, beam=BEAM, children=CHILDREN, depth=DEPTH, split=None,
                 review=True, feedback=True):
        self.draft, self.roles, self.models, self.budget = draft, roles, models, budget
        self.split = split      # the shared split check (splitter); None refuses every split
        self.review = review    # False: a component test with no C8; a selected candidate is recorded unreviewed
        self.feedback_on = feedback   # False: the off arm; the critic still runs, its verdict is logged only
        self.statement = statement_name(draft)
        self.beam, self.children, self.depth = beam, children, depth
        root = new_node("0", None, 0, None, {e: {"kind": "given", "text": t}
                                            for e, t in draft.get("evidence", {}).items()})
        reg = copy.deepcopy(draft["registry"])
        F, complete = state_of(draft, reg)
        root.update(status="root", registry=reg, failures=F, complete=complete, eligible=complete and not F)
        self.nodes = {"0": root}
        self.frontier, self.pending, self.scored, self.level = ["0"], [], [], 0
        self.calls = {"generator": [], "meaning": [], "critic": [], "controller": [], "c8": []}
        self.pruning = {"invariant": [], "beam": [], "depth": []}
        self.ineligible_last, self.beam_losses, self.departures = [], [], []
        self.termination = self.why = self.decision = self.chosen = self.committed = self.commit_check = None
        self.unreviewed = None
        self.meaning_used = False
        self.concepts = [l["concept"] for l in draft["lines"] if not fixed_line(l)]
        self.expansions, self.passing, self.picks = [], [], []     # per level: children, passing answers, picks

    def end(self, termination, why):
        if self.termination is None:
            self.termination, self.why = termination, why

    def path(self, n):
        out = []
        while n["parent"] is not None:
            out.append(n)
            n = self.nodes[n["parent"]]
        return out[::-1]

    def changed(self, reg):
        root = self.nodes["0"]["registry"]
        return [c for c in self.concepts if placement(reg, c) != placement(root, c)]

    def fixed_order(self, ids):
        """Code's fixed total order, logged beside the controller's rank: fewest failures, then fewest changed
        placements, then ID."""
        n = self.nodes
        return sorted(ids, key=lambda c: (len(n[c]["failures"]), len(n[c]["changed_lines"]),
                                          [int(x) for x in c.split(".")]))

    # -- generator --

    def feedback(self, node):
        """The critic's verdict and reason on each change already made on this branch, by the node that made it:
        that branch only, the critic's words unchanged, never evidence."""
        return {x["id"]: {"line": x["thought"]["line"], "verdict": x["critic"].get("verdict"),
                          "reason": x["critic"].get("reason")} for x in self.path(node) if x["critic"]}

    def generator_request(self, node):
        reg, F = node["registry"], node["failures"]
        feedback = self.feedback(node) if self.feedback_on else {}
        hint = None
        recon = next((c for c in F if c["check"].startswith("rows add to") and c.get("gap") is not None), None)
        if recon is not None:
            open_ = {c for c in self.concepts if c not in reg["alloc"]}
            hint = {"label": HINT, "gap": recon["gap"], "tolerance": recon["tolerance"],
                    "lines": gap_lines(self.draft["lines"], open_, recon["gap"], recon["tolerance"])}
        return plain({
            "role": "generator", "node": node["id"], "depth": node["depth"], "children": self.children,
            "statement": self.statement, "period": self.draft["period_end"],
            "section": self.draft["section"], "rows": self.draft["ids"],
            "lines": [{**{k: l.get(k) for k in ("concept", "label", "fact", "amount", "unit", "control")},
                       "sign": self.draft["signs"].get(l["concept"])} for l in self.draft["lines"]],
            "placements": {c: placement(reg, c, amounts=True) for c in self.concepts},
            "failures": F, "hint": hint, "path": [x["thought"] for x in self.path(node)],
            **({"feedback": feedback} if feedback else {}),
            **evidence_view([(e, b) for e, b in node["scope"].items() if b["kind"] not in NOT_CITABLE]),
            "actions": list(ACTIONS), "schema": generator_schema(self.draft, self.children),
            "ceiling": {"requests": GENERATOR_STEPS * self.budget.dispatches}, "refusal": None})

    def ask(self, line, question):
        """The generator's one meaning question to the analyst, from the tree's pool. A question that cannot be
        asked is refused with its reason; the generator's call goes on."""
        entry = {"line": line, "question": question}
        self.calls["meaning"].append(entry)
        why = ("no meaning question in this run" if self.roles.get("meaning") is None else
               "one meaning question per tree, already asked" if self.meaning_used else
               "it does not fit what is left of the tree's limit"
               if self.budget.short or not self.budget.fits("tree", self.budget.ceiling("tree", MEANING_REQUESTS))
               else None)
        if why:
            entry["refused"] = why
            return {"refused": why}
        self.meaning_used = True
        req = {"role": "meaning", "statement": self.statement, "section": self.draft["section"], "line": line,
               "question": question,
               "ceiling": {"requests": MEANING_REQUESTS * self.budget.dispatches}}
        return self.budget.call("tree", "meaning", MEANING_REQUESTS,
                                lambda m: self.roles["meaning"](req, ledger=m), entry)

    def propose(self, node):
        req = self.generator_request(node)
        left = {"reply": CHECK_RETRIES, "invariant": CHECK_RETRIES}     # each kind of refusal has its own retry
        carried = {}    # what a refused attempt's tools returned: shown to the invariant retry and in its scope
        for attempt in range(1 + sum(left.values())):
            scope = {**carried, **node["scope"]}
            entry = {"node": node["id"], "attempt": attempt + 1, "request": req}
            self.calls["generator"].append(entry)
            reply = self.budget.call("tree", "generator", GENERATOR_STEPS,
                                     lambda m: self.roles["generator"](req, self.ask, ledger=m), entry)
            if req.get("feedback"):     # each thought's longest word run shared with each critique it was shown
                entry["feedback_overlap"] = [
                    {e: shared_run(t.get("reason"), f["reason"]) for e, f in req["feedback"].items()}
                    for t in (reply.get("thoughts") or [] if isinstance(reply, dict) else [])
                    if isinstance(t, dict)]
            errs, kind = reply_errors(reply, self.children, node["registry"], scope), "reply"
            kept = usable(reply, errs) if errs and not left["reply"] else None
            if kept:    # the retry used: its broken thoughts drop out, the sound ones go on
                entry["dropped"], reply, errs = errs, {**reply, "thoughts": kept}, []
            if not errs:
                thoughts, disclosed = [normal(t) for t in reply["thoughts"]], {**carried, **reply.get("disclosed", {})}
                # thoughts that all break an invariant (perhaps one the generator was never told about) are shown
                # back once with the reasons, as C4's refusals are; one admissible thought keeps ordinary pruning,
                # and after the retry the gate prunes as usual
                errs, kind = (self.broken(node, thoughts, disclosed) if left["invariant"] else []), "invariant"
                if not errs:
                    entry["errors"] = []
                    return thoughts, disclosed
            entry["errors"] = errs
            if not left[kind]:
                break
            left[kind] -= 1
            if kind == "invariant":     # the retry sees and may cite what this attempt's tools returned
                carried = disclosed
                req = self.generator_request({**node, "scope": {**carried, **node["scope"]}})
            req = {**req, "refusal": errs, "attempt": attempt + 2}      # its own tool IDs (repair_roles)
        raise Stop("invalid", f"generator at node {node['id']}: {'; '.join(errs)}")

    def broken(self, node, thoughts, disclosed):
        """The gate's reasons when every thought breaks an invariant, else none (code only, nothing applied)."""
        scope = {**disclosed, **node["scope"]}
        gated = [gate(self.draft, node["registry"], scope, t, self.split) for t in thoughts]
        if any(child is not None for child, _ in gated):
            return []
        return [f"thought {i}: {inv}: {why}" for i, (_, errs) in enumerate(gated, 1) for inv, why in errs]

    def generate(self):
        """One generator call per node in the beam, up to `children` thoughts each. A child's evidence is its
        parent's plus what this call disclosed; a sibling branch's disclosures are not in it."""
        self.level += 1
        self.pending = []
        for nid in self.frontier:
            node = self.nodes[nid]
            node["expanded"] = True
            thoughts, disclosed = self.propose(node)
            scope = {**disclosed, **node["scope"]}
            asked = {json.dumps(plain([t["line"], wanted(t)]), sort_keys=True) for t in thoughts}
            self.expansions.append({"level": self.level, "node": nid, "children": len(thoughts),
                                    "distinct": len(asked)})
            for i, t in enumerate(thoughts, 1):
                cid = f"{nid}.{i}"
                self.nodes[cid] = new_node(cid, nid, self.level, t, scope)
                self.pending.append(cid)
        return "gate"

    # -- gate, F(x), eligibility --

    def gate(self):
        """The invariant gate on every child, then F(x), its change from the parent, the targets result, eligibility
        and the flags (code). At the last level only eligible children are scored."""
        root_ids = ids_of(self.nodes["0"]["failures"])
        admitted = []
        for cid in self.pending:
            n = self.nodes[cid]
            p = self.nodes[n["parent"]]
            child, errs = gate(self.draft, p["registry"], n["scope"], n["thought"], self.split)
            if child is None:
                n.update(status="rejected", rejected=sorted({i for i, _ in errs}),
                         why=[{"invariant": i, "why": w} for i, w in errs])
                self.pruning["invariant"].append({"id": cid, "invariants": n["rejected"]})
                continue
            F, complete = state_of(self.draft, child)
            f_ids, p_ids = ids_of(F), ids_of(p["failures"])
            n.update(status="admitted", registry=child, failures=F, complete=complete, eligible=complete and not F,
                     changed_lines=self.changed(child),
                     failures_change={"cleared": [x for x in p_ids if x not in f_ids],
                                      "new": [x for x in f_ids if x not in p_ids],
                                      "kept": [x for x in f_ids if x in p_ids]},
                     targets={x: ("cleared" if x in p_ids and x not in f_ids else "open" if x in p_ids
                                  else "not a failure of the parent") for x in n["thought"]["targets"]},
                     flags={"new_failures": [x for x in f_ids if x not in root_ids],
                            "non_improving": set(p_ids) <= set(f_ids), "targets_hit_later": None})
            admitted.append(cid)
        for nid in self.frontier:
            p = self.nodes[nid]
            if p["parent"] is not None and p["flags"]["non_improving"]:
                real = [x for x in p["thought"]["targets"] if x in ids_of(self.nodes[p["parent"]]["failures"])]
                kids = [self.nodes[c] for c in admitted if self.nodes[c]["parent"] == nid]
                p["flags"]["targets_hit_later"] = (None if not real else
                                                   any(all(x not in ids_of(k["failures"]) for x in real) for k in kids))
        passed = {"admitted": admitted, "eligible": [c for c in admitted if self.nodes[c]["eligible"]]}
        for k, ids in list(passed.items()):
            answers = {answer_of(self.nodes[c]["registry"]) for c in ids}
            passed[f"{k}_distinct"], passed[f"{k}_disagree"] = len(answers), len(answers) > 1
        answers = {}
        for c in passed["eligible"]:      # the distinct passing answers: arithmetic eligibility, not correctness
            answers.setdefault(answer_of(self.nodes[c]["registry"]), []).append(c)
        passed["eligible_answers"] = [{"nodes": ids, "changed": {x: placement(self.nodes[ids[0]]["registry"], x)
                                                                  for x in self.nodes[ids[0]]["changed_lines"]}}
                                      for ids in answers.values()]
        self.passing.append({"level": self.level, **passed})
        if not admitted:
            raise Stop("no_admitted_child", f"level {self.level}: every child broke an invariant: " + "; ".join(
                f"{c} {'/'.join(self.nodes[c]['rejected'])}" for c in self.pending))
        if self.level < self.depth:
            self.scored = admitted
            return "critic"
        self.pruning["depth"] += [{"id": c, "why": "depth limit: not expanded"} for c in admitted]
        self.scored = [c for c in admitted if self.nodes[c]["eligible"]]
        for c in admitted:
            if not self.nodes[c]["eligible"]:
                self.nodes[c]["skipped"] = "ineligible at last level"
                self.ineligible_last.append(c)
        if not self.scored:
            raise Stop("no_eligible", f"depth {self.level} reached with no eligible leaf")
        return "critic"

    # -- critic (feedback to the generator) --

    def critic(self):
        """The critic judges each scored child's evidence. Its verdict and reason go to the generator if that child
        is expanded (`feedback`); it prunes nothing and is not shown to the controller."""
        for cid in self.scored:
            n = self.nodes[cid]
            t, p = n["thought"], self.nodes[n["parent"]]
            req = plain({"role": "critic", "statement": self.statement, "line": t["line"],
                         "change": {"from": placement(p["registry"], t["line"], True),
                                    "to": placement(n["registry"], t["line"], True)},
                         "reason": t["reason"], "quote": t["quote"],
                         **evidence_view([(e, b) for e in t["evidence_ids"] if (b := block_of(e, n["scope"]))]),
                         "options": ["supported", "not supported"]})
            entry = {"node": cid, "request": req}
            self.calls["critic"].append(entry)
            n["critic"] = self.budget.call("tree", "critic", CRITIC_REQUESTS,
                                           lambda m: self.roles["critic"](req, ledger=m), entry)
            n["scored"] = True
        return "control"

    # -- controller --

    def view(self, cid):
        """What the controller sees of one child: code-written fields only, never the critic's verdict. The thought's
        own reason may reflect feedback the generator was shown; it is not stripped or rewritten."""
        n = self.nodes[cid]
        t, p = n["thought"], self.nodes[n["parent"]]
        return {"id": cid, "path": [x["id"] for x in self.path(n)], "line": t["line"], "action": t["action"],
                "change": {"from": placement(p["registry"], t["line"], True),
                           "to": placement(n["registry"], t["line"], True)},
                "reason": t["reason"], "evidence_ids": t["evidence_ids"], "quote": t["quote"],
                "failures": n["failures"], "failures_change": n["failures_change"], "targets": n["targets"],
                "eligible": n["eligible"], "complete": n["complete"], "changed_lines": n["changed_lines"]}

    def controller_request(self, final):
        eligible = [c for c in self.scored if self.nodes[c]["eligible"]]
        rejected = [{"id": c, "rejected": self.nodes[c]["rejected"]} for c in self.pending
                    if self.nodes[c]["status"] == "rejected"]
        req = {"role": "controller", "statement": self.statement, "level": self.level,
               "task": "select" if final else "rank",
               "nodes": [self.view(c) for c in self.scored], "rejected": rejected, "eligible": eligible,
               "schema": controller_schema(self.scored, eligible, final), "refusal": None}
        if not final:
            req.update(beam=self.beam, decisions=["expand", "stop"] if eligible else ["expand"])
        return plain(req)

    def control(self):
        """At a level before the last: rank every scored child, then expand the top `beam` or stop with an eligible
        one. At the last level: the final selection among eligible leaves, with a strict order."""
        final = self.level >= self.depth
        req = self.controller_request(final)
        for attempt in range(1 + CHECK_RETRIES):
            entry = {"level": self.level, "attempt": attempt + 1, "request": req}
            self.calls["controller"].append(entry)
            reply = self.budget.call("tree", "controller", CONTROLLER_REQUESTS,
                                     lambda m: self.roles["controller"](req, ledger=m), entry)
            errs = entry["errors"] = controller_errors(reply, req)
            if not errs:
                break
            req = {**req, "refusal": errs}
        else:
            raise Stop("invalid", f"controller at level {self.level}: {'; '.join(errs)}")
        order, fixed = reply["order"], self.fixed_order(self.scored)
        for i, c in enumerate(order, 1):
            self.nodes[c]["controller_rank"] = i
        for i, c in enumerate(fixed, 1):
            self.nodes[c]["fixed_rank"] = i
        if order != fixed:
            self.departures.append({"level": self.level, "fixed": fixed, "controller": order,
                                    "reason": reply["reason"]})
        seen = {}
        for c in order:
            n = self.nodes[c]
            if n["eligible"]:
                s = answer_of(n["registry"])
                n["same_answer_as"] = seen.get(s)
                seen.setdefault(s, c)
        picked = [reply["select"]] if final or reply["decision"] == "stop" else order[:self.beam]
        self.picks.append({"level": self.level, "decision": "select" if final else reply["decision"],
                           "shown": list(self.scored), "picked": picked,
                           "position_shown": [self.scored.index(c) + 1 for c in picked]})
        if final or reply["decision"] == "stop":
            self.decision, self.chosen = ("select" if final else "stop"), reply["select"]
            return "commit"
        self.decision = "expand"
        keep, drop = order[:self.beam], order[self.beam:]
        self.pruning["beam"] += [{"id": c, "level": self.level, "controller_rank": order.index(c) + 1,
                                  "reason": reply["reason"]} for c in drop]
        if fixed[0] in drop:
            self.beam_losses.append({"level": self.level, "dropped": fixed[0], "kept": keep, "fixed_order": fixed,
                                     "controller_order": order, "reason": reply["reason"]})
        self.frontier = keep
        return "generate"

    # -- commit --

    def recheck(self, n):
        """Replay the path from the root draft through the gate, re-check eligibility, and check every change's
        evidence (G2), preparatory ones included."""
        reg, errs = copy.deepcopy(self.draft["registry"]), []
        for x in self.path(n):
            child, bad = gate(self.draft, reg, x["scope"], x["thought"], self.split)
            if child is None:
                return errs + [f"{x['id']}: {i} {w}" for i, w in bad]
            errs += [f"{x['id']}: {w}" for w in evidence_errors(x["thought"], x["scope"])]
            reg = child
        if reg != n["registry"]:
            errs.append(f"{n['id']}: its registry is not what its path gives")
        F, complete = state_of(self.draft, reg)
        if F or not complete:
            errs.append(f"{n['id']}: not eligible on re-check (failed: {ids_of(F)}; complete: {complete})")
        return errs

    def c8_request(self, n, c):
        """One changed line for C8: `line`, `rows` and `questions` are what C8 sees, blind; `candidate` and
        `decomposition` are for code's comparison in the C8 wrapper, never shown to C8. The decomposition is
        complete: every piece with its claims, readings, support, provisional flag and basis, and the remainder. The
        wrapper's `settled` means its code-validated review procedure completed, any revision and the final candidate
        checks included. `order` is the node's changed lines in the fixed order they are reviewed, for the wrapper's
        ownership (code-side)."""
        reg, line = n["registry"], next(l for l in self.draft["lines"] if l["concept"] == c)
        ps = [p for p in reg["pieces"] if p["source"] == c]
        rest = line["amount"] - sum((p["amount"] for p in ps), Decimal(0))
        return plain({"role": "c8", "arm": self.budget.arm, "section": self.draft["section"],
                      "line": {k: line.get(k) for k in ("concept", "label", "fact", "amount", "unit")},
                      "rows": self.draft["ids"],
                      "questions": {"destinations": [p["fact"] for p in ps] + (["remainder"] if ps else ["line"]),
                                    "pieces": [{k: p[k] for k in ("fact", "concept", "dims", "amount")} for p in ps]},
                      "candidate": placement(reg, c, amounts=True),
                      "decomposition": {"pieces": copy.deepcopy(ps),
                                        "remainder": {"row": (reg["alloc"].get(c) or {}).get("row"), "amount": rest}},
                      "order": list(n["changed_lines"]),
                      "ceiling": self.budget.ceiling("review", REVIEW_PER_LINE[self.budget.arm])})

    def commit(self):
        """Commit re-check, then C8 on each changed line from the review reserve; anything but settled withdraws the
        repair and the root draft stays. With no review (a component test, review=False), a candidate that passes the
        re-check is recorded unreviewed and nothing is committed: the root draft stays. However a review ends, a cap
        and an interruption included, the lines after it are recorded not run before the ending is passed on."""
        n = self.nodes[self.chosen]
        errs = self.recheck(n)
        self.commit_check = {"node": n["id"], "errors": errs}
        if errs:
            raise Stop("refused_at_commit", "; ".join(errs))
        if not self.review:
            self.unreviewed = n["id"]
            self.end("selected_unreviewed", f"node {n['id']} selected ({self.decision}) and re-checked; no review in "
                                            "this run, so it is not committed and the original draft is kept")
            return "end"
        for i, c in enumerate(n["changed_lines"]):
            req = self.c8_request(n, c)
            entry = {"line": c, "request": req}
            self.calls["c8"].append(entry)
            later = lambda why: [{"line": x, "status": None, "why": f"not run: {why}"}   # noqa: E731
                                 for x in n["changed_lines"][i + 1:]]
            try:
                res = self.budget.call("review", "c8", REVIEW_PER_LINE[self.budget.arm],
                                       lambda m: self.roles["c8_review"](req, ledger=m), entry)
            except BaseException as e:      # the line's status and ending kept with its usage
                ending = e.termination if isinstance(e, Stop) else "interrupted"
                entry.update(status=None, ending=f"{ending}: {e.why if isinstance(e, Stop) else repr(e)}")
                self.calls["c8"] += later(ending)
                raise
            status = entry["status"] = res.get("status") if isinstance(res, dict) else None
            if status != SETTLED:
                self.calls["c8"] += later("the repair is withdrawn")
                raise Stop("withdrawn", f"C8 left {c} {status or 'with no status'}: the repair is withdrawn and the "
                                        f"original draft kept")
        self.committed = n["id"]
        self.end("selected_at_depth_limit" if self.decision == "select" else "controller_stopped",
                 f"node {n['id']} committed after the re-check and C8")
        return "end"

    def feedback_exposure(self):
        """Critiques produced, prepared in a later generator request, dispatched in one that was physically sent,
        and never dispatched (the controller stopped, the branch was not kept, it was the last level, or the call was
        refused before sending). Prepared is not dispatched. A call outside the meter (no shared ledger) counts as
        sent when it returned with requests reported. Sent is not received: a send that then failed stays counted."""
        def sent(c):
            u = c.get("usage")
            return bool(c.get("sends", {}).get("chat")) or (
                not c.get("unsent") and "failure" not in c and "usage_errors" not in c
                and isinstance(u, dict) and u.get("requests", 0) > 0)
        produced = [c for c, n in self.nodes.items() if n["critic"]]
        prepared = {e for c in self.calls["generator"] for e in c["request"].get("feedback", {})}
        dispatched = {e for c in self.calls["generator"] if sent(c) for e in c["request"].get("feedback", {})}
        return {"produced": produced, "prepared": [c for c in produced if c in prepared],
                "dispatched": [c for c in produced if c in dispatched],
                "never_dispatched": [c for c in produced if c not in dispatched]}

    def record(self):
        nodes = [{**{k: v for k, v in n.items() if k != "scope"}, "evidence_scope": sorted(n["scope"])}
                 for n in self.nodes.values()]
        return plain({
            "settings": {"models": self.models, "beam": self.beam, "children": self.children, "depth": self.depth,
                         "arm": self.budget.arm, "dispatches": self.budget.dispatches, "limits": self.budget.limits,
                         "hint": HINT, "invariants": INVARIANTS, "terminations": TERMINATIONS,
                         "critic": ("feedback to the generator expanding that branch, never evidence; not shown "
                                    "to the controller, prunes nothing" if self.feedback_on else
                                    "feedback off: logged only; not shown to the controller, prunes nothing"),
                         "feedback": "on" if self.feedback_on else "off",
                         "review": ("C8 on each changed line" if self.review else
                                    "none: component test; a selected, re-checked candidate is recorded unreviewed"),
                         "statement": self.statement,
                         "plan": "the repair tree"},
            "termination": self.termination, "why": self.why, "decision": self.decision, "chosen": self.chosen,
            "committed": self.committed, "unreviewed": self.unreviewed,
            "root_failures": ids_of(self.nodes["0"]["failures"]), "nodes": nodes,
            "pruning": {**self.pruning, "ineligible_at_last_level": self.ineligible_last},
            "beam_losses": self.beam_losses, "departures": self.departures, "calls": self.calls,
            "feedback": self.feedback_exposure(),
            "expansions": self.expansions, "passing": self.passing, "picks": self.picks,
            "commit_check": self.commit_check, "budget": self.budget.state()})


# ---- the graph ------------------------------------------------------------------------------------------

class State(TypedDict):
    tree: Any
    route: str


def step(fn):
    def run(state):
        t = state["tree"]
        try:
            route = fn(t)
        except Stop as e:
            t.end(e.termination, e.why)
            route = "end"
        return {"route": route}
    return run


def graph():
    """generator -> invariant gate -> critic -> controller, routed on the controller's decision."""
    g = StateGraph(State)
    for name, fn in (("generate", Tree.generate), ("gate", Tree.gate), ("critic", Tree.critic),
                     ("control", Tree.control), ("commit", Tree.commit)):
        g.add_node(name, step(fn))
    route = lambda s: s["route"]   # noqa: E731
    g.add_edge(START, "generate")
    g.add_conditional_edges("generate", route, {"gate": "gate", "end": END})
    g.add_conditional_edges("gate", route, {"critic": "critic", "end": END})
    g.add_conditional_edges("critic", route, {"control": "control", "end": END})
    g.add_conditional_edges("control", route, {"generate": "generate", "commit": "commit", "end": END})
    g.add_edge("commit", END)
    return g.compile(checkpointer=None, cache=None)


def repair(draft, generator, critic, controller, c8_review, meaning=None, models=None, arm="own_loop",
           dispatches=1, limits=None, beam=BEAM, children=CHILDREN, depth=DEPTH, clock=time.monotonic, split=None,
           ledger=None, company=None, review=True, review_line=None, feedback=True):
    """Run C7 on one C6-failed draft. Roles (fakes in tests; nothing here calls a model):
    generator(request, ask) -> (reply, usage), reply {thoughts, disclosed?}; `disclosed` (ID to {kind, text}) is
    what its tools returned, written by its wrapper; ask(line, question) is the one meaning question.
    critic(request) -> (reply, usage), feedback to the generator. controller(request) -> (reply, usage).
    c8_review(request) -> (result, usage), result {status: settled | disagreed | unfinished | unresolved | refused |
    unavailable, ...}; only settled commits.
    meaning(request) -> (reply, usage), optional. Usage: {requests, tokens, embedding_requests, embedding_tokens},
    every counter given. Each role is also passed `ledger=`, the call's Meter, which its wrapper hands to its Caller
    and Embedder. `ledger`, `company`: the batch's shared ledger.Ledger, which every attempt goes through.
    `limits`: {requests, tokens, wall_s}, by default derived (129 requests at the own-loop arm, one dispatch).
    `split`: the shared split check, splitter(tax, f), bound to the draft's filing; with none, every split is refused.
    `review`: True (runtime repair) needs C8 settled on every changed line before a commit, and keeps the review
    reserve. False (a component test): no reserve, no C8 call, and a selected, re-checked candidate ends
    `selected_unreviewed` with nothing committed. `review_line`: the review wrapper's ceiling per changed line
    ({requests, tokens}, see Budget). `feedback`: True gives the critic's verdict and reason to the generator
    expanding that branch; False (the off arm) leaves the request as without a critique, the critic still called.
    A KeyboardInterrupt ends the tree `interrupted`; the partial result rides on the exception as `.c7` and the
    interrupt is re-raised.

    Returns {termination, why, decision, committed, draft, record}: `draft` is the input object itself unless a node
    was committed, then a new draft with that node's registry."""
    errs = draft_errors(draft)
    if errs:
        raise ValueError("C7: the root draft fails its own checks: " + "; ".join(errs))
    budget = Budget(arm, dispatches, beam, children, depth, limits, clock, lines=None if review else 0,
                    ledger=ledger, company=company, review_line=review_line if review else None)
    roles = {"generator": generator, "critic": critic, "controller": controller, "c8_review": c8_review,
             "meaning": meaning}
    t = Tree(draft, roles, {**ROLE_MODELS, **(models or {})}, budget, beam, children, depth, split, review,
             feedback)
    if not t.nodes["0"]["failures"]:
        raise ValueError("C7: the draft passes C6; nothing to repair")
    try:
        graph().invoke({"tree": t, "route": ""}, config={"recursion_limit": 4 * depth + 4})
    except KeyboardInterrupt as e:      # keep the partial tree, then re-raise
        t.committed = None
        t.termination = None
        t.end("interrupted", f"KeyboardInterrupt at level {t.level}: the partial tree is kept, nothing committed")
        e.c7 = {"termination": t.termination, "why": t.why, "decision": t.decision, "committed": None,
                "unreviewed": t.unreviewed, "draft": draft, "record": t.record()}
        raise
    if t.termination is None:
        raise RuntimeError("C7: the graph ended without a termination")
    out = draft
    if t.committed is not None:
        out = {**copy.deepcopy({k: v for k, v in draft.items() if k != "registry"}),
               "registry": copy.deepcopy(t.nodes[t.committed]["registry"])}
    return {"termination": t.termination, "why": t.why, "decision": t.decision, "committed": t.committed,
            "unreviewed": t.unreviewed, "draft": out, "record": t.record()}


def write_record(d, result, name="c7_tree.json"):
    """The whole tree into the run's dated record (never overwritten)."""
    write(d, name, json.dumps(result["record"], indent=1))
