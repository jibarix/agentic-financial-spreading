"""The filing-corrections record and the code that named totals (named_totals.py) uses with it: one record per job,
written before the organizer, placement and repair read the filing. The labels-fix step that also wrote it (a Qwen
proposer and a Ling confirmer) was removed;
its records, at VERSION 1, are still read.

Retag. A record's retag moves a printed total's facts to the template's tag. Code compares the organizer's checks on
the code-only assignment before and after (`admit`): a check that tied and stops tying refuses the retag.

Split, code only. A line printed by member only (a department, a class of stock) becomes one line per member amount
the statement page prints (`split_of`), when the pieces with the line's siblings add up to the one printed total above
it (`split_adds_up`); otherwise it stays as it is, with the reason. A piece's key is '<tag>#<axis>=<member>'.

The record is immutable, and `apply` lays it over a fresh Filing in memory: the judged facts carry the new tag, with
the original tag kept on each fact, and the statement's presentation and calculation arcs name the new tag. Other
facts with the old tag are untouched. A split line's member facts carry its piece's key (`split_from` keeps the tag),
and the statement's arcs name its pieces in its place."""
import hashlib
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from . import keycheck, organizer as org, spread, template
from .filing import Filing, in_period

ROOT = Path(__file__).resolve().parents[1]
TOTAL_STATES = ("statement control", "verified subtotal", "incomplete subtotal")
MEMBER = "printed by member only"
RECORD = "corrections.json"
VERSION = 1
# The split rules. "add up": a split is
# admitted only when the pieces add up to the one printed total above the line, one axis only (templates before
# format v7). "printed rows": each member row the statement page prints is its own line, on any number of axes, bound
# to its fact by the printed cell; the add-up is recorded as information and admits nothing; the end checks and the
# line checks catch a missed or double-counted amount (templates with outcome_checks, format v7 on).
ADD_UP, PRINTED_ROWS = "add up", "printed rows"


def split_rule():
    """The split rule of the active template: printed rows when its outcome checks decide (format v7), else add up."""
    return PRINTED_ROWS if template.active().outcome_checks is not None else ADD_UP


# ---- the printed totals ---------------------------------------------------------------------------------------------

def money(x):
    return f"{int(round(float(x))):,}" if x not in (None, "") else "no amount"


def totals_of(inv, calc):
    """The statement's printed totals with an amount: a line the company's calculation adds up, or one code marks a
    total; the first occurrence of each tag."""
    seen, out = set(), []
    for ln in inv["lines"]:
        if "amount" in ln and ln["concept"] not in seen and (ln["concept"] in calc or ln["state"] in TOTAL_STATES):
            seen.add(ln["concept"])
            out.append(ln)
    return out


# ---- the overlay and the admission comparison ------------------------------------------------------------------------

class Refused(RuntimeError):
    pass


def dims_of(defref):
    """A member block's defref 'srt_ProductOrServiceAxis=us-gaap_CasinoMember' as (axis, member); None for a block
    that does not name one axis and one member."""
    parts = defref.split("=")
    return tuple(x.replace("_", ":", 1) for x in parts) if len(parts) == 2 else None


def piece_key(tag, dims):
    """'<tag>#<axis>=<member>', axes in sorted order joined by '&' when the facts carry more than one."""
    return tag + "#" + "&".join(f"{a}={m}" for a, m in sorted(dims))


def nested(f, members):
    """The members that sit under another of them, or are an axis's domain (the total of its members), in any of the
    filing's dimension networks: such pieces overlap, and blocks with distinct names and a sum that ties do not show
    otherwise."""
    up, domains = defaultdict(set), set()
    for a in f.arcs("def"):
        if a["arcrole"].endswith("/domain-member"):
            up[a["child"]].add(a["parent"])
        elif a["arcrole"].endswith(("/dimension-domain", "/dimension-default")):
            domains.add(a["child"])
    out = set()
    for m in members:
        seen, todo = set(), list(up[m])
        while todo:
            p = todo.pop()
            if p not in seen:
                seen.add(p)
                todo += up[p]
        if m in domains or seen & (set(members) - {m}):
            out.add(m)
    return sorted(out)


def norm(text):
    return " ".join(text.lower().split())


def resolved(f, h, text, groups):
    """Printed rows: the one set of members (a key of `groups`) a block names. Its defref names one axis and member
    (`h`); a block on several axes prints every member's label, ' | ' between them, so the other members of the set
    are those whose labels the block prints. (dims, None) or (None, why)."""
    parts = [norm(p) for p in text.split(" | ")]
    cand = [g for g in groups if h in g and len(g) == len(parts)
            and all(norm(f.label(m)) in parts for a, m in g if (a, m) != h)]
    if len(cand) != 1:
        return None, (f"the block {text} binds none of the line's facts" if not cand else
                      f"the block {text} matches {len(cand)} sets of members")
    return cand[0], None


def split_of(f, inv, ln, rows, scale, rule=ADD_UP):
    """(pieces, None) or (None, why): one piece per member block of the statement page (`rows`: keycheck.page_rows,
    signed) that prints the line, each bound to the line's facts at the statement's period on that one member, by
    its amount at the page's scale, its printed sign following the fact's as every other block's does (a negated
    label flips them all). Only what the statement prints counts: the filing's other breakdowns of the tag (its
    notes') are not lines here. One axis only (this release). Refused: the tag on two lines, a total of other lines,
    blocks on more than one axis, members nested in one another, a block that binds no fact, a nil amount, an amount
    of the line's own, not one currency unit, duplicates that disagree, a member not printed once on the statement,
    a printed amount the fact does not match, printed signs that do not follow the facts'. Under the printed rows
    rule a block may name several axes (`resolved`), and a member printed twice is judged by its whole set."""
    key, tag = tuple(inv["key"]), ln["concept"]
    ctx = lambda x: f.contexts[x["context"]]
    if sum(l["concept"] == tag for l in inv["lines"]) > 1:
        return None, "the same tag is printed on two lines"
    if f.calculation(inv["role"]).get(tag):
        return None, "the line is a total of other lines"
    cells = [(dims_of(block[0]), block[1], amount) for c, block, amount in rows if c == tag and block]
    if not cells:
        return None, "no member amount of this line is printed on the statement page"
    if rule == ADD_UP and (any(h is None or " | " in text for h, text, _ in cells) or
                           len({h[0] for h, _, _ in cells}) > 1):
        return None, "member blocks on more than one axis"
    if any(h is None for h, _, _ in cells):
        return None, "a member block names no axis and member"
    inner = nested(f, [h[1] for h, _, _ in cells])
    if inner:
        return None, f"members nested in one another: {', '.join(inner)}"
    if rule == ADD_UP and len({h for h, _, _ in cells}) != len(cells):
        return None, "a member block prints the line twice"
    here = [x for x in f.by_concept.get(tag, []) if in_period(ctx(x), key) and ctx(x)["entity"] == f.entity]
    if any(not ctx(x)["dims"] for x in here):
        return None, "a nil amount" if any(x["value"] == "" for x in here if not ctx(x)["dims"]) else             "the line has an amount of its own"
    groups = defaultdict(list)
    for x in here:
        groups[tuple(sorted(ctx(x)["dims"].items()))].append({**x, "currency": f.currency(x["unit"])})
    shown = defaultdict(int)
    for l in inv["lines"]:
        shown[l["concept"]] += 1
    out, flips, sets = [], set(), set()
    for h, text, printed in cells:
        if rule == ADD_UP:
            d = (h,)
            if d not in groups:
                return None, f"the block {text} binds none of the line's facts"
        else:
            d, why = resolved(f, h, text, groups)
            if why:
                return None, why
            if d in sets:
                return None, "a member block prints the line twice"
            sets.add(d)
        xs = groups[d]
        if any(x["value"] == "" for x in xs):
            return None, "a nil amount"
        if spread.disagree(xs):
            return None, f"duplicate facts disagree for {text}"
        x = spread.chosen(xs)
        if shown[h[1]] != 1:
            return None, f"{h[1]} is not printed once on the statement"
        if abs(printed) != abs(Decimal(x["value"])) / scale:
            return None, f"{text}: its fact {x['id']} ({x['value']}) is not the printed amount {printed}"
        if printed and Decimal(x["value"]):
            flips.add((printed > 0) == (Decimal(x["value"]) > 0))
        out.append({"key": piece_key(tag, d), "dims": [list(p) for p in d], "fact": x["id"],
                    "fact_ids": [y["id"] for y in xs], "label": f"{ln['label']} – {text}",
                    "amount": x["value"], "decimals": x["decimals"], "unit": x["unit"],
                    "cell": {"block": text, "printed": str(printed)}})
    if len(flips) > 1:
        return None, "the printed signs do not follow the facts' signs"
    if len({p["unit"] for p in out}) != 1 or not f.currency(out[0]["unit"]):
        return None, "not one currency unit"
    return out, None


def split_adds_up(f, inv, ln, split):
    """(True, how) when the one printed total above the line (its calculation parent) equals its calculation
    children, each split line counted by its signed pieces (`split`: tag -> pieces); else (False, why). Tolerance:
    half a unit of each amount's stated decimals. The sum is only the check: no fact is made from it."""
    calc = f.calculation(inv["role"])
    parents = [p for p in ln["calc_parents"] if p in calc]
    if len(parents) != 1:
        return False, f"{len(parents)} printed totals above it in the calculation, not one"
    parent = parents[0]
    amt = {}
    for l in inv["lines"]:
        if "amount" in l:
            amt.setdefault(l["concept"], l)
    if parent not in amt:
        return False, f"its total {parent} has no single amount"
    total, terms, words = Decimal(0), [amt[parent]["decimals"]], []
    for child, w in calc[parent]:
        if child in split:
            v = sum(Decimal(p["amount"]) for p in split[child])
            terms += [p["decimals"] for p in split[child]]
        elif child in amt:
            v = Decimal(amt[child]["amount"])
            terms.append(amt[child]["decimals"])
        else:
            return False, f"{child}, a part of {parent}, has no single amount"
        total += w * v
        words.append(f"{'+' if w > 0 else '-'}{abs(v):,}")
    tol = sum(Decimal(10) ** -int(d) / 2 for d in terms if d not in (None, "INF"))
    how = f"{' '.join(words)} = {total:,} against {parent} {Decimal(amt[parent]['amount']):,} (tolerance {tol:,})"
    return abs(total - Decimal(amt[parent]["amount"])) <= tol, how


def splits(directory, tax, record, name):
    """The split part of the step, on the filing with the record's applied retags laid over it: every line printed
    by member only, split or kept with its reason. [{statement, role, key, line, label, tag, applied, why, how,
    pieces}]."""
    f = apply(Filing(directory), {"retags": record.get("retags", [])})
    pages, problems = keycheck.statement_pages(Path(directory), template.active().data)
    out, rule = [], split_rule()
    for st in template.active().statements:
        inv = org.inventory(f, tax, st, name)
        members = [] if inv.get("refused") else [l for l in inv["lines"] if l["state"] == MEMBER]
        if not members:
            continue
        page = Path(directory) / pages[st] if st in pages else None
        probs = []
        rows, scale, _ = (keycheck.page_rows(page, f.period_end, probs, signed=True) if page and page.exists()
                          else ([], Decimal(1), None))
        found = {l["id"]: split_of(f, inv, l, rows, scale, rule) if rows else
                 (None, "no statement page: " + "; ".join(probs or problems or [f"{st} page not saved"]))
                 for l in members}
        # every split line's pieces count in each total they sit under (two such lines may share one total)
        split = {l["concept"]: found[l["id"]][0] for l in members if found[l["id"]][0]}
        for l in members:
            got, why = found[l["id"]]
            adds, how = split_adds_up(f, inv, l, split) if got else (False, None)
            if rule == ADD_UP:
                ok, said = adds, ("the pieces add up to the printed total" if adds else
                                  why or f"pieces do not add up: {how}")
            else:       # the add-up is information only: what it found, never a refusal
                ok, said = bool(got), ("each printed member row is its own line" if got else why)
            e = {"statement": st, "role": inv["role"], "key": inv["key"], "line": l["id"], "label": l["label"],
                 "tag": l["concept"], "applied": ok, "how": how, "why": said, "pieces": got or []}
            if rule == PRINTED_ROWS:
                e.update(rule=rule, adds_up=adds if got else None)
            out.append(e)
    return out


def open_filing(filing, binding=None):
    """The filing as a run reads it: a fresh Filing with the corrections record its run.json names laid over it.
    `binding`: the run's {"record", "sha256"} (run.json "corrections"), passed explicitly, never found in a folder;
    None for a run without corrections (legacy, or the step off), which reads the raw filing. Refused: a record
    missing, whose sha256 is not the sealed one, or of another filing. The Filing carries the sealed sha256
    (`corrections_sha256`), which organizer.load checks against the run's."""
    p = Path(filing)
    f = Filing(str(p if p.is_absolute() else ROOT / p))
    f.corrections_sha256 = None
    f.disagreements = []    # named totals where the model named another line than the company's tag (shown, not acted on)
    if not binding:
        return f
    r = Path(binding["record"])
    r = r if r.is_absolute() else ROOT / r
    if not r.is_file():
        raise Refused(f"corrections record {binding['record']} is missing")
    if sha(r) != binding["sha256"]:
        raise Refused(f"corrections record {binding['record']}: its sha256 is not the sealed one")
    record = json.loads(r.read_text(encoding="utf8"))
    there = Path(record["filing"])
    if (there if there.is_absolute() else ROOT / there).resolve() != f.dir.resolve():
        raise Refused(f"corrections record {binding['record']} is of {record['filing']}, not {filing}")
    apply(f, record)
    f.corrections_sha256 = binding["sha256"]
    f.disagreements = [e for e in record.get("named", []) if str(e.get("outcome", "")).startswith("disagrees")]
    return f


def apply(f, record):
    """Lay the record's applied retags and splits over a fresh Filing, in place, and return it. Retag: the judged facts
    carry the new tag (`retagged_from` keeps the original); the statement role's presentation and calculation arcs
    name it; the printed line keeps the company's label for the original tag. Split: each piece's facts carry its key
    (`split_from` keeps the tag, `piece` marks them); the role's arcs to the line name each piece in its place, in
    printed order; the piece is labelled '<line> \u2013 <member>' (Filing.pieces)."""
    retags = [r for r in record.get("retags", []) if r["applied"]]
    cuts = [s for s in record.get("splits", []) if s["applied"]]
    if not retags and not cuts:
        return f
    ids = {i: r for r in retags for i in r["fact_ids"]}
    for x in f.facts:
        if x["id"] in ids:
            x["retagged_from"], x["concept"] = x["concept"], ids[x["id"]]["new"]
    for s in cuts:
        for p in s["pieces"]:
            f.pieces[p["key"]] = {"tag": s["tag"], "dims": dict(map(tuple, p["dims"])), "fact": p["fact"],
                                  "label": p["label"], "role": s["role"]}
            for x in f.facts:
                if x["id"] in p["fact_ids"]:
                    x["split_from"], x["concept"], x["piece"] = x["concept"], p["key"], True
    f.by_concept.clear()
    for x in f.facts:
        f.by_concept[x["concept"]].append(x)
    names = {(r["role"], r["old"]): r["new"] for r in retags}
    keys = {(s["role"], s["tag"]): [p["key"] for p in s["pieces"]] for s in cuts}
    f.printed_as.update({(r["role"], r["new"]): r["old"] for r in retags})
    arcs = f.arcs

    def retagged(kind, role=None):
        out = arcs(kind, role)
        if kind not in ("pre", "cal"):
            return out
        return [dict(a, parent=names.get((a["role"], a["parent"]), a["parent"]),
                     child=names.get((a["role"], a["child"]), a["child"])) for a in out]

    def renamed(kind, role=None):
        out = retagged(kind, role)
        if kind not in ("pre", "cal"):
            return out
        return [dict(a, child=k, order=a["order"] + i / 1000) for a in out
                for i, k in enumerate(keys.get((a["role"], a["child"]), [a["child"]]))]
    f.arcs, f.tag_arcs = renamed, retagged
    return f


def code_assignment(inv):
    """item ID -> code's own section (None when code has none): the code-only assignment, by physical occurrence."""
    return {l["id"]: (l.get("code") or {}).get("section") for l in org.items_of(inv)}


def admit(directory, tax, record, inv, retag):
    """The organizer's checks on the frozen code-only assignment, before and after this retag, both with the retags
    already admitted (so admitted retags are judged together): (admitted, [{check, before, after, why_before,
    why_after, exposed}], why). A line with an amount that code places in no section leaves every check that would
    otherwise tie or differ unavailable: where it belongs is not known, so neither is what it adds. Only a zero amount is known to add nothing."""
    prior = [r for r in record["retags"] if r["applied"] and r["statement"] == retag["statement"]]
    side = lambda rs: org.inventory(apply(Filing(directory), {"retags": rs}), tax, inv["statement"], inv.get("company"))
    before, after = (side(prior) if prior else inv), side([*prior, {**retag, "applied": True}])
    if after.get("refused"):
        return False, [], f"statement refused after the retag: {after['refused']}"
    occ = lambda i: [(l["id"], l["concept"] if l["id"] != retag["line"] else None) for l in i["lines"]]
    if occ(after) != occ(before) or next(l for l in after["lines"] if l["id"] == retag["line"])["concept"] != retag["new"]:
        return False, [], "printed occurrences differ after the retag"
    asg = code_assignment(inv)
    zero = lambda l: "amount" in l and Decimal(l["amount"]) == 0
    unplaced = [l["id"] for l in org.items_of(inv) if asg[l["id"]] is None and not zero(l)]

    def ties(i):
        """org.ties, with every check that would tie or differ left unavailable while a line is unplaced."""
        out = org.ties(i, asg)
        for k, v in out.items():
            if unplaced and v["state"] in ("tied", "differs"):
                out[k] = {**v, "state": "unavailable", "why": "code places no section for " + ", ".join(unplaced)}
        return out
    b, a = ties(before), ties(after)
    rows = [{"check": c, "before": b[c]["state"], "after": a[c]["state"], "why_before": b[c].get("why"),
             "why_after": a[c].get("why"), "exposed": b[c]["state"] != "tied" and a[c]["state"] == "differs"}
            for c in b]
    worse = [r["check"] for r in rows if r["before"] == "tied" and r["after"] != "tied"]
    return not worse, rows, ("a tied check stops tying: " + ", ".join(worse)) if worse else "no tied check lost"


# ---- the record's file ------------------------------------------------------------------------------------------------

def tokens_used(log_path):
    p = Path(log_path)
    if not p.is_file():
        return 0
    return sum((r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0)
               for r in map(json.loads, p.read_text(encoding="utf8").splitlines()))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


