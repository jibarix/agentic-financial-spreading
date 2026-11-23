"""The C4 analyst's step format and the code checks behind it.

A step is {"why", "step"}. `step` is an anyOf of one object per action, each binding its own one-value `action` to
its own arguments, so a tool name can only come with that tool's arguments. The schema is rebuilt every step from
what the line has returned so far: the rows (the section's, plus any a `rows_of` call returned) and the four pools:
`recall_ids` (every observation, search included; `recall` only), `discovered_passages` (every passage a search or read
named; `read_passage` only), `read_passages` (opened by read_passage) and `citable_ids` (the line's fact, facts tools
returned, non-search O-IDs, read passages). At the last step only `answer` is offered.

Prompt v16, the six rules: one answer, {action "answer", quote, reason, row, pieces}, the quote first (evidence before
the decision). `row` is a section row (or a row a rows_of call returned) or COMBINED, OUTSIDE or NOT_TOLD; `pieces`
[{fact_id, quote, row}] draws fact IDs from `allocatable_ids` and rows from the section's own rows (`section_rows`), and
is offered only when some fact is allowed. A piece's quote is a required field that may be empty; code requires it for a
piece the networks do not place inside the line (six_rules). Code's checks are six_rules.py's rules 5 and 6.

`allocatable_ids`: tool-returned facts less the line's own amount and its duplicate aliases, known non-amounts (a known
non-monetary type, or a unit that is not one plain currency) and all but the canonical member of a complete duplicate
set (Eligibility). Membership is not filtered: inclusion means type-eligible, never "inside the line".

The split machinery below (split, chain_errors, support_errors and the claims) is what repair.py uses, and the
relations what six_rules.py uses: code finds the amounts by fact ID, checks each piece (date, entity, currency,
balance, owner, size) and keeps conservation: the pieces and the remainder add up to the line. Code never picks a
row.
"""
import difflib
import json
import re
from decimal import Decimal
from pathlib import Path

from .answer_keys import precise_duplicate
from .filing import fact_ident, line_ident, tag_of
from .spread import OPTIONS, balance_of, clean, norm, quote_found, quote_parts

STR = {"type": "string"}
# hard caps on the free-text fields, enforced while decoding (Ollama's format honours maxLength) and checked again by
# schema_errors: the prompt's word limits with room to spare, so a why used as a scratchpad cannot run the reply past
# the output cap
WHY = {"type": "string", "maxLength": 300}
QUOTE = {"type": "string", "maxLength": 400}
REASON = {"type": "string", "maxLength": 300}
TOOLS = ["calc_children", "breakdowns", "search_notes", "read_passage", "other_periods", "rows_of", "printed_page",
         "recall"]
SCOPES = ["block", "statement"]     # printed_page: the line's block, or every row
CLAIM_KINDS = ["line", "rival", "piece"]   # membership of the line; overlap with a printed or owned amount; with a piece


def obj(props):
    """The bound form: every property required, additionalProperties false (one form for both engines)."""
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def one(value):
    return {"type": "string", "enum": [value]}


def enum(values):
    return {"type": "string", "enum": list(values)}


def tool_branches(state):
    """One branch per tool that can be called now. A tool whose argument would be an empty enum (no passage or
    observation returned yet, no other section) is left out of this step, not offered with nothing to choose."""
    args = {"calc_children": {"concept": STR}, "breakdowns": {"concept": STR}, "search_notes": {"query": STR},
            "other_periods": {"concept": STR}, "printed_page": {"scope": enum(SCOPES)}}
    if state.get("page_line"):      # the tree: one tool set for a whole section, so the page names its line
        args["printed_page"] = {"concept": STR, **args["printed_page"]}
    if state.get("faces"):          # alternate statement lookup (--root-cause-fixes): "this statement", or another statement and a heading
        args["printed_page"] = {**args["printed_page"], "statement": STR, "heading": STR}
    pools = state.get("pools")      # read_passage from discovered passages, recall from every O-ID
    passages = state.get("discovered_passages") if pools else state.get("passage_ids")
    recallable = state.get("recall_ids") if pools else state.get("observation_ids")
    if passages:
        args["read_passage"] = {"passage_id": enum(passages)}
    if state.get("sections"):
        args["rows_of"] = {"section": enum(state["sections"])}
    if recallable:
        args["recall"] = {"observation_id": enum(recallable)}
    return [obj({"action": one(t), **args[t]}) for t in TOOLS if t in args and t in state.get("tools", TOOLS)]


def simple_answer_branch(state):
    """Prompt v16: the one answer form. The quote first, then the reason, the row and, when some fact is allowed,
    the pieces (each a fact ID, its own quote, which may be empty, and a section row: evidence before the verdict
    inside a piece too)."""
    props = {"action": one("answer"), "quote": QUOTE, "reason": REASON,
             "row": enum(list(state["rows"]) + OPTIONS)}
    if state.get("allocatable_ids"):
        item = obj({"fact_id": enum(state["allocatable_ids"]), "quote": QUOTE,
                    "row": enum(state.get("section_rows") or state["rows"])})
        props["pieces"] = {"type": "array", "items": item}
    return obj(props)


def step_schema(state, last=False):
    """The whole step schema for this step: the tools that can be called now (none at the last step) and the one
    answer form. `state` holds rows, fact_ids, the pools and sections, and tools (the tools of this arm)."""
    return obj({"why": WHY, "step": {"anyOf": ([] if last else tool_branches(state))
                                     + [simple_answer_branch(state)]}})


# ---- quote-failure messages and pointers: where a quote fails, and the text a pointer names ----

SENTENCE = re.compile(r"(?<=[a-z0-9)%][.;])\s+(?=[A-Z(])")     # a sentence end, not "[facts: F2]"


def units(text):
    """The lines of a passage as numbered for pointers: its own lines, each cut at sentence ends."""
    return [u.strip() for ln in text.split("\n") for u in SENTENCE.split(ln) if u.strip()]


def nearest(quote, blocks):
    """(block, line) whose words come closest to the quote, by code's string distance (the share of the quote's
    normalized characters matched in order, ties to the shorter line); None if no block has a line. It shows where
    the quote nearly is; it never stands in for the quote."""
    q, best = norm(quote), None
    for b, text in blocks:
        for ln in units(text):
            m = difflib.SequenceMatcher(None, q, norm(ln), autojunk=False)
            score = sum(x.size for x in m.get_matching_blocks()) / max(len(q), 1)
            if best is None or (score, -len(ln)) > best[0]:
                best = ((score, -len(ln)), b, ln)
    return best and (best[1], best[2])


def located(quote, blocks, label=None):
    """The message for a quote not found: the parts no cited block holds and the nearest line, verbatim, named by
    `label(block, line)` (its line ID, with pointers on) or by its block."""
    miss = [p for p in quote_parts(quote) if all(p not in clean(t) for _, t in blocks)]
    near = nearest(quote, blocks)
    out = f" Not found: {miss}." if miss else " Each part occurs, but not all in one block."
    if near:
        out += f" Nearest line in what you cite ({(label or (lambda b, ln: b))(*near)}): \"{near[1]}\""
    return out


def cited_text(s):
    """The text a support entry or answer stands on: the lines its pointer names (rendered by code) or its
    quote. v14: the same, read from the evidence object; a non-contiguous pointer renders with omission markers."""
    ev = s.get("evidence")
    if isinstance(ev, dict):
        return s.get("_rendered", "") if ev.get("form") == "pointer" else ev.get("text") or ""
    return s.get("_rendered", "") if (s.get("pointer") or "").strip() else s.get("quote", "")


def cited_found(s, text):
    """Whether what an entry cites occurs in `text`. A v14 pointer is checked range by range (`_segments`), so an
    omission marker is never looked for in the filing; anything else as quote_found on cited_text."""
    segs = s.get("_segments") if isinstance(s.get("evidence"), dict) else None
    if segs:
        return all(quote_found(g, text) for g in segs)
    return quote_found(cited_text(s), text)


# ---- what a fact may be allocated as, by type and identity only ----

OWN = "the line itself"
NOT_AMOUNT = "not an amount"
PIECE_MARKS = {OWN: "(the line itself: not a piece)", NOT_AMOUNT: "(not an amount: not a piece)"}     # prompt v16
MONETARY = {"monetaryItemType"}
# XBRL 2.1 and DTR item types that are not monetary, by local name; an unfamiliar name is never taken as one: its type
# stays unresolved and the fact is kept
NON_MONETARY = {
    "stringItemType", "normalizedStringItemType", "tokenItemType", "languageItemType", "textBlockItemType",
    "domainItemType", "enumerationItemType", "enumerationSetItemType", "booleanItemType", "yesNoItemType",
    "dateItemType", "dateTimeItemType", "timeItemType", "durationItemType", "gYearItemType", "gYearMonthItemType",
    "gMonthDayItemType", "gMonthItemType", "gDayItemType", "gYearListItemType", "sharesItemType", "pureItemType",
    "percentItemType", "perShareItemType", "perUnitItemType", "decimalItemType", "floatItemType", "doubleItemType",
    "integerItemType", "nonNegativeIntegerItemType", "positiveIntegerItemType", "nonPositiveIntegerItemType",
    "negativeIntegerItemType", "longItemType", "intItemType", "shortItemType", "byteItemType",
    "unsignedLongItemType", "unsignedIntItemType", "unsignedShortItemType", "unsignedByteItemType",
    "fractionItemType", "anyURIItemType", "QNameItemType", "NameItemType", "NCNameItemType", "base64BinaryItemType",
    "hexBinaryItemType", "areaItemType", "volumeItemType", "massItemType", "energyItemType", "flowItemType",
    "lengthItemType", "memberItemType"}
# prefixes an extension schema conventionally binds to the XBRL instance and DTR namespaces; a type under any other
# prefix is extension-derived and stays unresolved
TYPE_PREFIXES = {"xbrli", "dtr-types", "dtr", "num", "nonnum", "us-types"}


def natural(fid):
    """A fact ID's sort key, its digit runs read as numbers: f-919 before f-2043."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(fid))]


class Eligibility:
    """Whether a filed fact may be allocated, by type and identity only. Never membership: a monetary fact whose place
    is unknown stays allocatable, and inclusion is recorded as "type-eligible", never "inside the line".

    Excluded: the line's own amount, the line's fact and every fact with its concept, an equal context and the same
    unit measures (its aliases, whatever their IDs sort as); a non-amount, a concept whose type is known and not
    monetary (the standard taxonomy's type, or the extension schema's type under an XBRL or DTR prefix) or a unit
    that is not one plain currency (filing.currency); and every member of a complete duplicate set (same concept,
    context, unit, value and decimals) but its canonical ID, the lowest by `natural`. A type that cannot be resolved
    (an extension-derived type, a concept in neither schema) is kept and counted (`unresolved`). The unit test asks
    only whether the unit is one plain currency; the comparison with the line's currency stays split()'s hard check.
    Facts of one concept, context and unit that differ in value or decimals are not collapsed: each stays
    allocatable, and `group` names them for eligibility_errors' duplicate_group check.

    `of(fid)` returns a JSON-ready record: allocatable, why (OWN, NOT_AMOUNT, "alias of <id>" or None), canonical,
    amount_type ("monetary", "non_monetary", "unresolved"), basis, unresolved, group."""

    def __init__(self, tax, f, line):
        self.tax, self.f, self.line = tax, f, line
        self.index = fact_index(f)
        full, part = {}, {}
        for x in f.facts:
            if not x.get("id") or x["value"] == "":
                continue
            k = self.slot(x)
            part.setdefault(k, []).append(x["id"])
            full.setdefault(k + (x["value"], x.get("decimals")), []).append(x["id"])
        self.canonical = {i: min(ids, key=natural) for ids in full.values() for i in ids}
        self.groups = {}
        for ids in part.values():
            canon = sorted({self.canonical[i] for i in ids}, key=natural)
            if len(canon) > 1:
                for i in ids:
                    self.groups[i] = canon
        own = self.index.get(line["fact"])
        self.own = (set(part.get(self.slot(own), [])) if own else set()) | {line["fact"]}

    def slot(self, x):
        """(concept, context, unit measures): a fact's identity before its value and decimals."""
        ctx = json.dumps(self.f.contexts.get(x["context"]), sort_keys=True, default=str)
        return x["concept"], ctx, json.dumps(self.f.unit_measures.get(x["unit"]), default=str)

    def amount_type(self, concept):
        """(monetary | non_monetary | unresolved, basis), by tag (a split piece's key reads its tag)."""
        concept = tag_of(concept)
        c = self.tax.concepts.get(concept) if self.tax is not None else None
        if c is not None:
            t = c.get("type") or ""
            if c.get("abstract"):
                return "non_monetary", "abstract in the standard taxonomy"
            if t in MONETARY:
                return "monetary", f"standard taxonomy type {t}"
            if t in NON_MONETARY:
                return "non_monetary", f"standard taxonomy type {t}"
            return "unresolved", f"standard taxonomy type {t or 'missing'} not known as monetary or not"
        e = self.f.extension.get(concept)
        if e is not None:
            raw = e.get("type") or ""
            prefix, _, local = raw.rpartition(":")
            if e.get("abstract"):
                return "non_monetary", "abstract in the extension schema"
            if prefix in TYPE_PREFIXES and local in MONETARY:
                return "monetary", f"extension schema type {raw}"
            if prefix in TYPE_PREFIXES and local in NON_MONETARY:
                return "non_monetary", f"extension schema type {raw}"
            return "unresolved", f"extension schema type {raw or 'missing'} is extension-derived or unfamiliar"
        return "unresolved", "concept in neither the standard taxonomy nor the extension schema"

    def of(self, fid):
        x = self.index.get(fid)
        rec = {"fact": fid, "allocatable": False, "why": None, "canonical": None, "amount_type": None,
               "basis": None, "unresolved": False, "group": None}
        if fid in self.own:
            return {**rec, "why": OWN}
        if x is None:
            return {**rec, "why": "not a fact of this filing"}
        t, basis = self.amount_type(x["concept"])
        rec.update(amount_type=t, basis=basis, unresolved=t == "unresolved")
        if t == "non_monetary":
            return {**rec, "why": NOT_AMOUNT}
        if self.f.currency(x["unit"]) is None:
            return {**rec, "why": NOT_AMOUNT, "basis": f"unit {x['unit']} is not one plain currency"}
        canon = self.canonical.get(fid, fid)
        if canon != fid:
            return {**rec, "why": f"alias of {canon}", "canonical": canon}
        return {**rec, "allocatable": True, "canonical": fid, "group": self.groups.get(fid)}


def trivial_eligibility(line_fact):
    """Eligibility with no filing at hand (fake-model tests): the line's fact is the line itself, every other fact
    is allocatable. Never used where run_c4 can build the real one."""
    def of(fid):
        own = fid == line_fact
        return {"fact": fid, "allocatable": not own, "why": OWN if own else None, "canonical": None if own else fid,
                "amount_type": None, "basis": "no filing given", "unresolved": False, "group": None}
    return of


def fact_index(f):
    return {x["id"]: x for x in f.facts if x.get("id")}


DOMAIN_MEMBER = "http://xbrl.org/int/dim/arcrole/domain-member"
DIMENSION_DOMAIN = "http://xbrl.org/int/dim/arcrole/dimension-domain"


def positive(w):
    try:
        return w is not None and float(w) > 0
    except ValueError:
        return False


def networks(arcs):
    """role -> parent -> [(child, role of the walk below the child, weight)]. Each extended link role is its own
    network; a targetRole on an arc moves the walk below its child into that role (XBRL Dimensions §2.4). Networks
    are never merged, so no path is made up from arcs of different roles."""
    net = {}
    for a in arcs:
        net.setdefault(a["role"], {}).setdefault(a["parent"], []).append(
            (a["child"], a.get("target_role") or a["role"], a.get("weight")))
    return net


def walk(net, node, role, signed=False):
    """Every (node, role) below `node` in `role`'s network; with `signed`, only along arcs of positive weight."""
    seen, todo = set(), [(node, role)]
    while todo:
        n, r = todo.pop()
        for c, r2, w in net.get(r, {}).get(n, ()):
            if (signed and not positive(w)) or (c, r2) in seen:
                continue
            seen.add((c, r2))
            todo.append((c, r2))
    return seen


def withheld(arcs):
    """Roles whose effective network code does not resolve: an arc with use="prohibited", or one relationship
    stated more than once (an override by priority). XBRL 2.1 §3.5.3.9.7.5. Such a network gives no structural
    verdict; comparisons that would rest on it are "unknown"."""
    roles, seen = set(), set()
    for a in arcs:
        k = (a["role"], a["parent"], a["child"])
        if a.get("use") == "prohibited" or k in seen:
            roles.add(a["role"])
        seen.add(k)
    return roles


class Relations:
    """What the filing's own networks say about whether two amounts, each a key (concept, dims), overlap. Every
    answer is (result, basis) with result "overlap" (known), "disjoint" (supported by one calculation network) or
    "unknown". The absence of a link establishes neither disjointness nor separate coverage: it is "unknown", and a
    piece left unknown needs cited filing evidence and C8's review.

    Domain-member arcs define membership, not a summation (Dimensions 1.0), so sibling members are never "disjoint" and
    a dimensioned fact is never established as a part of the undimensioned amount: the axis may be a scenario, a
    measurement or another qualifier. A calculation relates amounts in one context, so it settles disjointness and
    membership only between keys with the same dims (XBRL 2.1 §5.2.5.2).

    Networks with prohibited or overridden arcs are withheld, not resolved (`withheld`). Members a definition arc
    marks usable="false" are recorded (`unusable`) and reported in the review; their subtrees are kept.

    Coverage: only the company's own linkbases are read. A relationship in a standard-taxonomy linkbase that is not
    loaded can override or prohibit a company arc, so the effective networks are not known to be complete (XBRL 2.1
    §3.5.3.9.7). Unless `complete` is set, which needs a reader that resolves the whole DTS, "disjoint" and "inside"
    are reported as "unverified": a structural reading is present, the effective network is not verified. That is
    evidence for review, not a verdict, and differs from "unknown", where no network reads the pair at all. Known
    overlaps still refuse."""

    COVERAGE = "company linkbases only; effective-network completeness not verified"

    def __init__(self, cal, dfn, complete=False, cross_dims_unknown=False):
        self.complete = complete
        # cross-dimension overlap policy (a root-cause job only, --root-cause-fixes): overlap() does not carry a calculation across different dims.
        # r7_log: the attempted pieces this relaxed, written by six_rules.amounts (open_claims.attempt)
        self.cross_dims_unknown, self.r7_log = cross_dims_unknown, []
        # the policy by name, the phase it was built for (phase_relations), the frozen filing's folder name (a
        # claim's identity, open_claims.claim_id) and the pieces validated under it (the per-path denominator)
        self.policy = R7_POLICY if cross_dims_unknown else LEGACY_POLICY
        self.phase = self.filing = None
        self.validated = self.vote_validated = 0
        cal = [a for a in cal if (a.get("arcrole") or "").endswith("summation-item")]
        dfn = [a for a in dfn if a.get("arcrole") in (DOMAIN_MEMBER, DIMENSION_DOMAIN)]
        self.withheld = {"calc": withheld(cal), "dim": withheld(dfn)}
        self.cal = networks(a for a in cal if a["role"] not in self.withheld["calc"])
        self.dfn = networks(a for a in dfn if a["role"] not in self.withheld["dim"])
        self.unusable = {a["child"] for a in dfn if a.get("usable") == "false"}
        self.memo = {}

    def note(self, kind):
        w = self.withheld[kind]
        return f"; networks withheld for prohibited or overridden arcs: {sorted(w)}" if w else ""

    def below(self, kind, x):
        """{x} and everything under x, walking each network of `kind` from x separately."""
        if (kind, x) not in self.memo:
            net = self.cal if kind == "calc" else self.dfn
            self.memo[kind, x] = {x} | {n for r in net for n, _ in walk(net, x, r)}
        return self.memo[kind, x]

    def apart_under(self, a, b):
        """Every (parent, role) with a under one of the parent's positive items and b under another and neither under
        both: the network then counts them once each. A parent whose walk enters a withheld network gives no
        verdict."""
        out = []
        for r in self.cal:
            for p, arcs in self.cal[r].items():
                under = [{(c, r2)} | walk(self.cal, c, r2, signed=True) for c, r2, w in arcs if positive(w)]
                if any(r2 in self.withheld["calc"] for u in under for _, r2 in u):
                    continue
                ia = {i for i, u in enumerate(under) if a in {n for n, _ in u}}
                ib = {i for i, u in enumerate(under) if b in {n for n, _ in u}}
                if ia and ib and not ia & ib:
                    out.append((p, r))
        return out

    def apart(self, a, b):
        """The first parent of apart_under, as a basis; None when no parent counts them apart."""
        found = self.apart_under(a, b)
        return f"under different items of {found[0][0]} in {found[0][1]}" if found else None

    def inside_roles(self, piece, line):
        """The roles whose network places the concept `piece` below `line` along positive calculation weights."""
        return [r for r in self.cal if piece in {n for n, _ in walk(self.cal, line, r, signed=True)}]

    def concepts(self, a, b):
        if a == b:
            return "same", None
        if self.below("calc", a) & self.below("calc", b):
            return "overlap", "one is below the other, or both contain a common item, in a calculation"
        basis = self.apart(a, b)
        return ("disjoint", basis) if basis else ("unknown", "no calculation network relates them" + self.note("calc"))

    def members(self, ax, m1, m2):
        if m1 == m2 or self.below("dim", m1) & self.below("dim", m2):
            return "overlap", f"{m1} and {m2} share members on {ax}"
        return "unknown", (f"{m1}, {m2} on {ax}: domain-member arcs define membership, not a summation"
                           + self.note("dim"))

    def dims(self, da, db):
        """"meet" when a slice lies inside both (every shared axis overlaps, one axis set contains the other; a
        missing axis is the total), else "unknown"."""
        da, db = dict(da), dict(db)
        shared = sorted(set(da) & set(db))
        found = [self.members(ax, da[ax], db[ax]) for ax in shared]
        if all(res == "overlap" for res, _ in found) and (set(da) <= set(db) or set(db) <= set(da)):
            return "meet", None
        return "unknown", "; ".join(b for res, b in found if res != "overlap") or "different axes"

    def settled(self, result, basis):
        """A "disjoint" or "inside" verdict, or, with networks not known to be complete, "unverified" with that
        reading as its basis."""
        if self.complete:
            return result, basis
        return "unverified", f"the company networks read {result}: {basis} ({self.COVERAGE})"

    def overlap(self, a, b):
        d, dbasis = self.dims(a[1], b[1])
        c, cbasis = self.concepts(a[0], b[0])
        if c == "disjoint" and dims_key(a[1]) == dims_key(b[1]):      # one representation (as the cross-dimension policy below)
            return self.settled("disjoint", cbasis)
        if c == "disjoint":
            return "unknown", f"{cbasis}, but in contexts with different dims"
        if self.cross_dims_unknown and c == "overlap" and d == "meet" and dims_key(a[1]) != dims_key(b[1]):
            # cross-dimension policy: a calculation binds facts of one context (XBRL 2.1 5.2.5.2), so it does not put a fact in other
            # dims inside the other concept's amount. Never "disjoint": the claim stays open (open_claims.py)
            return "unknown", R7_BASIS
        if c in ("same", "overlap") and d == "meet":
            return "overlap", cbasis or "the same concept on slices that meet"
        return "unknown", "; ".join(x for x in (cbasis, dbasis) if x)

    def r7_pair(self, a, b):
        """The cross-dimension policy's condition on a pair, whatever this object's policy: different concepts related by a calculation,
        on slices that meet, in contexts with different dims. The legacy policy reads such a pair "overlap"."""
        return (self.concepts(a[0], b[0])[0] == "overlap" and self.dims(a[1], b[1])[0] == "meet"
                and dims_key(a[1]) != dims_key(b[1]))

    def inside(self, piece, line):
        """Whether a piece is shown to be part of the line: below it along positive calculation weights in one
        network, in a context with the same dims. A dimensioned fact is not shown to be part of the line."""
        (cp, dp), (cl, dl) = piece, line
        if dp != dl:
            return "unknown", ("a context with other dims: the networks do not establish that the axis divides the "
                               "line into components (it may be a scenario or another qualifier)")
        for r in self.cal:
            if cp in {n for n, _ in walk(self.cal, cl, r, signed=True)}:
                return self.settled("inside", f"below the line along positive calculation weights in {r}")
        return "unknown", "the filing's networks do not place it inside the line" + self.note("calc")


R7_BASIS = "related by a calculation, in contexts with different dims"
LEGACY_POLICY = "legacy: a calculation between two concepts is an overlap in any dims"
R7_POLICY = "R7: a calculation between two concepts in contexts with different dims reads unknown"
# The overlap policy of each phase, written down.
# "source": the policy of the spread run the work belongs to, cross-dimension for a root-cause job, legacy otherwise.
# "legacy": always the legacy policy. Cross-dims is off only in steps.split, the check of a new repair split.
PHASES = {"placement": "source",                # six_rules.amounts through check_answer (spread.py)
          "vote path": "source",                # six_rules.amounts at the checker-majority step (spread.py)
          "source-run checking": "source",      # the C8 re-review of a source run's line (repair_review.Review)
          "repair split": "legacy"}             # steps.split through repair.splitter (repair_run, review_live)


def dims_key(d):
    """A key's dims in one representation (pairs or a mapping, tuples or lists, any order), for comparison."""
    return tuple(sorted((str(a), str(m)) for a, m in (d.items() if isinstance(d, dict) else d or ())))


def relations(f, cross_dims_unknown=False):
    """`cross_dims_unknown`: the cross-dimension policy, set for a root-cause job (--root-cause-fixes) and for work on its record."""
    arcs = getattr(f, "tag_arcs", f.arcs)        # by tag: a split piece is (tag, member dims) here
    rel = Relations(arcs("cal"), arcs("def"), cross_dims_unknown=cross_dims_unknown)
    d = getattr(f, "dir", None)
    rel.filing = Path(d).name if d is not None else None
    return rel


def phase_relations(f, phase, root_cause_fixes=False):
    """The Relations of one phase (PHASES), with its policy decided here and nowhere else. `root_cause_fixes`: the
    source run's setting; it has no effect on a phase whose policy is "legacy"."""
    rel = relations(f, cross_dims_unknown=PHASES[phase] == "source" and bool(root_cause_fixes))
    rel.phase = phase
    return rel


def name(key):
    """How a claim names an amount: the concept, with its dims when it has them."""
    return key[0] + (str(dict(key[1])) if key[1] else "")


def split(tax, f, line, allocations, printed, owned, index=None, rel=None, evidence=None, observations=None,
          period=None, r7_count_only=False):
    """(pieces, remainder, errors) for the model's allocations of one line. `period`: the line's period key
    (["instant", date] or ["duration", start, end]) when the caller is not at the balance-sheet date (C7 on one
    section of an income-statement or cash-flow run); None, C4's case, keeps the balance-sheet date. Each piece is a
    filed fact at the line's
    date for the filing's own entity, in the line's currency, with no conflicting duplicate (same concept,
    equivalent context, same unit measures), a known balance equal to the line's, a positive amount, and within
    what is left of the line. A piece may not be a calculation ancestor of the line, nor overlap, as far as the
    filing's networks show (Relations), another piece, an amount already owned or a printed line other than the
    line and its ancestors. Any failed check refuses the whole answer (no piece moves). The pieces and the
    remainder add up to the line's amount.

    Where the networks settle neither way, the piece is the model's judgement and provisional: its `review` lists
    each open claim, typed {kind, target}: kind "line" for membership of the line (target the line's concept),
    "rival" for overlap with a printed line or owned amount (target its name), "piece" for overlap with another
    piece (target its fact ID). Each claim records its `reading`: "unverified" when the company networks read it but
    the effective network is not verified, "none" when no network reads it. Only a claim with no reading needs C4's
    cited support (`needs_c4_support`); an unverified one goes to C8 with the networks' reading as its basis, and C4's
    answer still cites the evidence for its allocation. A piece read inside the line and a line read apart from a rival
    give a chained reading: kept on the claim as labelled evidence, with both paths and the dims, never a verdict, and
    the claim stays for C8.

    Support entries {about, evidence_id, quote}: `about` lists one or more claims, `evidence_id` is a returned
    passage or observation (`evidence`, ID to the text the model was shown; fact IDs are not support) and `quote`
    text found in that block. Only a claim between two allocated pieces may be supported on either. Code checks the
    IDs, the kinds and the quote, and keeps the reference on each claim the entry names, with how many it named: one
    quote is not taken to establish every claim; whether the evidence is relevant and sufficient for each is C8's
    judgement. A claim with no reading and no support refuses the answer; a provisional piece stays provisional
    until C8 settles it, and an overlap C8 does not resolve stays an exception.

    Chains {about, inside, apart}: C4 may rest a rival claim with no reading on the chain the networks read, by naming
    the observation that shows the piece inside the line and the one that shows the line and the rival apart under one
    parent (`observations`, ID to the observation record, only those shown in full). Code checks and binds them
    (chain_errors); the claim then reads "chained", with its two premises for C8.

    `owned` holds (concept, dims) keys already claimed in this filing at the line's date and currency; pieces add
    theirs only when the caller commits them. A piece carries its dims, so a breakdown member is a different
    amount from the line's total."""
    index = index if index is not None else fact_index(f)
    # this check's policy is the legacy one (PHASES["repair split"]): with no Relations given it builds that, and an
    # cross-dimension Relations is refused, so a caller cannot bring that policy here. `r7_count_only`: repair.splitter's count of the
    # refusals that policy would have relaxed; its result is recorded and never returned to a tree
    rel = rel if rel is not None else phase_relations(f, "repair split")
    if getattr(rel, "cross_dims_unknown", False) and not r7_count_only:
        raise ValueError("steps.split takes the legacy overlap policy: the cross-dimension policy is off here (steps.PHASES)")
    evidence = evidence or {}
    whole = line_ident(f, line["concept"])
    up = {p for r in rel.cal for p in rel.cal[r] if p != whole[0] and whole[0] in rel.below("calc", p)}
    rivals = [line_ident(f, c) for c in printed if c != line["concept"] and line_ident(f, c)[0] not in up] + list(owned)
    line_bal = balance_of(tax, f, line["concept"])
    pieces, errs, keys, left = [], [], [], Decimal(line["amount"])
    chainable = {}      # (fact ID, rival name) -> the rival's key, for claims the networks would chain
    for a in allocations:
        x = index.get(a["fact_id"])
        if x is None:
            errs.append(f"{a['fact_id']}: no such fact in the filing")
            continue
        ctx = f.contexts[x["context"]]
        key = fact_ident(f, x)
        dims = key[1]
        same = [y for y in f.by_concept.get(x["concept"], []) if f.contexts[y["context"]] == ctx
                and f.unit_measures.get(y["unit"]) == f.unit_measures.get(x["unit"]) and y["value"] != ""]
        bal = balance_of(tax, f, x["concept"])
        clash, review = None, []
        inside, ibasis = rel.inside(key, whole)
        if inside != "inside":
            odd = sorted(m for _, m in dims if m in rel.unusable)
            review.append(claim_entry({"kind": "line", "target": line["concept"]}, "inside the line",
                                      ibasis + (f"; members marked not usable: {odd}" if odd else ""), inside))
        for k, what, claim in ([(r, "printed line or owned amount", {"kind": "rival", "target": name(r)})
                                for r in rivals]
                               + [(k, "piece", {"kind": "piece", "target": i}) for k, i in keys]):
            res, basis = rel.overlap(key, k)
            if res == "overlap":
                clash = clash or f"overlaps {what} {name(k)}: {basis}"
            elif res != "disjoint":
                r = claim_entry(claim, f"overlap with {what} {name(k)}", basis, res)
                lres, lbasis = rel.overlap(whole, k) if what != "piece" else (None, None)
                kin = k[0] in rel.below("calc", whole[0]) or whole[0] in rel.below("calc", k[0])
                if inside in ("inside", "unverified") and lres in ("disjoint", "unverified") and not kin:
                    # labelled evidence, never a verdict: the claim stays for C8 until C4 selects the chain and its
                    # premises pass
                    r["chain"] = {"label": CHAIN, "piece_in_line": ibasis, "line_and_rival": lbasis,
                                  "dims": {"piece": dims, "line": whole[1], "rival": k[1]},
                                  "inside_roles": rel.inside_roles(key[0], whole[0]),
                                  "apart_under": [list(x) for x in rel.apart_under(whole[0], k[0])]}
                    chainable[(x["id"], claim["target"])] = k
                review.append(r)
        why = None
        if ctx["entity"] != f.entity or (period is None and ctx["instant"] != f.period_end):
            why = "not the filing's own entity at the balance-sheet date"
        elif period is not None and list(period[1:]) != (
                [ctx["instant"]] if ctx.get("instant") else [ctx.get("start"), ctx.get("end")]):
            why = "not in the line's period"
        elif f.currency(x["unit"]) != line["unit"]:
            why = f"currency {f.currency(x['unit'])}, line in {line['unit']}"
        elif precise_duplicate(same) is None:
            why = "duplicated with different amounts"
        elif Decimal(x["value"]) != Decimal(precise_duplicate(same)["value"]):
            why = (f"a lower-precision duplicate of {precise_duplicate(same)['id']}; to use this amount as a piece, "
                   f"use {precise_duplicate(same)['id']}")
        elif key[0] in up:
            why = "a calculation ancestor of the line"
        elif clash:
            why = clash
        elif bal is None or line_bal is None:
            why = "balance unknown"
        elif bal != line_bal:
            why = "balance differs from the line's"
        elif Decimal(x["value"]) <= 0:
            why = "amount not positive"
        elif Decimal(x["value"]) > left:
            why = "larger than what is left of the line"
        if why:
            errs.append(f"{a['fact_id']}: {why}")
            continue
        amount = Decimal(x["value"])
        left -= amount
        keys.append((key, x["id"]))
        pieces.append({"fact": x["id"], "concept": x["concept"], "dims": dims, "row": a["row"], "amount": amount,
                       **({"tag": key[0]} if x.get("piece") else {}), "source": line["concept"], "by": "C4", "review": review, "provisional": bool(review),
                       "basis": "the model's judgement, provisional until C8 review" if review
                       else "the filing's networks"})
    if not errs:
        errs = chain_errors(f, line, pieces, allocations, observations or {}, index, rel, chainable)
    if not errs:
        errs = support_errors(pieces, allocations, evidence)
    if errs:
        return [], Decimal(line["amount"]), errs
    if sum((p["amount"] for p in pieces), Decimal(0)) + left != Decimal(line["amount"]):
        raise RuntimeError(f"conservation broken for {line['concept']}")    # a code fault, never a model outcome
    return pieces, left, []


def listed(o):
    """The fact IDs an observation listed."""
    return {x for it in o.get("items", ()) for x in it.get("facts", ())}


def bound(f, index, fid, line):
    """Whether fact `fid` is the line's own amount: the same concept, entity, date and dimensions (an equivalent
    context), the same currency and the same amount as the line being split."""
    x, own = index.get(fid), index.get(line["fact"])
    if x is None or own is None or x["concept"] != line["concept"] or not x["value"]:
        return False
    return (f.contexts[x["context"]] == f.contexts[own["context"]] and f.currency(x["unit"]) == line["unit"]
            and Decimal(x["value"]) == Decimal(line["amount"]))


def chain_errors(f, line, pieces, allocations, observations, index, rel, chainable):
    """Code's checks on the chains C4 selected. Each entry names rival claims of its piece that no network reads but the
    networks would chain, an `inside` observation (calc_children or breakdowns of the line's concept, listing the
    piece's fact) and an `apart` observation (calc_children of a parent the networks count the line and the rival apart
    under, listing a fact bound to the line and the rival's fact), both shown in full. Dimensions identical throughout
    (none, for the line). Any failure refuses the whole answer; when all pass, each named claim reads "chained", needs
    no C4 citation and carries its two premises for C8. Code never selects a chain."""
    errs, marks = [], []
    by = {p["fact"]: p for p in pieces}
    for a in allocations:
        p = by.get(a["fact_id"])
        for ch in a.get("chains", ()):
            def bad(why):
                errs.append(f"{a['fact_id']}: chain ({ch['inside']}, {ch['apart']}): {why}")
            claims = {json.dumps(r["claim"], sort_keys=True): r for r in p["review"]} if p else {}
            named = [claims.get(json.dumps(dict(c), sort_keys=True)) for c in ch["about"]]
            oi, oa = observations.get(ch["inside"]), observations.get(ch["apart"])
            if not ch["about"]:
                bad("it names no claim in its about")
            elif any(r is None or r["claim"]["kind"] != "rival" for r in named):
                bad(f"about must list rival claims of this piece; its rival claims are "
                    f"{json.dumps([r['claim'] for r in p['review'] if r['claim']['kind'] == 'rival'] if p else [])}")
            elif any(r["reading"] != "none" for r in named):
                bad("a claim the networks already read needs no chain: "
                    + json.dumps([r["claim"] for r in named if r["reading"] != "none"]))
            elif any((a["fact_id"], r["claim"]["target"]) not in chainable for r in named):
                bad("the filing's networks read no chain for "
                    + json.dumps([r["claim"] for r in named if (a["fact_id"], r["claim"]["target"]) not in chainable]))
            elif oi is None or oa is None:
                bad("both must be tool results shown to you in full (recall one first if it is summarised)")
            elif oi["tool"] not in ("calc_children", "breakdowns") or oi["args"].get("concept", "").strip() != line["concept"]:
                bad(f"inside must be calc_children or breakdowns of the line, {line['concept']}")
            elif a["fact_id"] not in listed(oi):
                bad(f"{ch['inside']} does not list the piece {a['fact_id']}")
            elif oa["tool"] != "calc_children":
                bad("apart must be calc_children of a total that counts the line and the other line apart")
            elif not any(bound(f, index, x, line) for x in listed(oa)):
                bad(f"{ch['apart']} does not list the line's own amount {line['fact']}")
            else:
                parent = oa["args"]["concept"].strip()
                for r in named:
                    k = chainable[(a["fact_id"], r["claim"]["target"])]
                    rival = [x for x in listed(oa) if x in index and index[x]["concept"] == k[0]
                             and tuple(sorted(f.contexts[index[x]["context"]]["dims"].items())) == k[1]]
                    roles = [rr for pp, rr in rel.apart_under(line["concept"], k[0]) if pp == parent]
                    if p["dims"] != () or k[1] != ():
                        bad(f"dimensions differ along the chain to {r['claim']['target']}: a chain needs the piece, "
                            "the line and the other line all without dimensions")
                    elif not rival:
                        bad(f"{ch['apart']} does not list {r['claim']['target']}")
                    elif not roles:
                        bad(f"the filing's networks do not count the line and {r['claim']['target']} apart under "
                            f"{parent}")
                    else:
                        marks.append((p, r, ch, parent, roles))
    if errs or not marks:
        return errs
    own = index[line["fact"]]
    for p, r, ch, parent, roles in marks:
        member = next((x for x in p["review"] if x["claim"]["kind"] == "line"), None)
        r["chain"].update(selected_by="C4", inside={"observation": ch["inside"], "roles": r["chain"]["inside_roles"]},
                          apart={"observation": ch["apart"], "parent": parent, "roles": roles},
                          bound_line={"concept": line["concept"], "fact": line["fact"], "entity": f.entity,
                                      "date": f.contexts[own["context"]]["instant"], "dims": (), "unit": line["unit"],
                                      "amount": str(line["amount"])})
        r["reading"], r["needs_c4_support"] = "chained", False
        r["premises"] = [{"question": "part", "piece": p["fact"], "line": line["concept"],
                          "review": "the piece's membership claim" if member else "settled by the networks"},
                         {"question": "separate", "line": line["concept"], "rival": r["claim"]["target"]}]
    return []


def support_errors(pieces, allocations, evidence):
    """Every support entry names at least one claim, cites a returned passage or observation and quotes text found
    in it; every claim with no network reading has at least one such entry naming it, on its own piece or, for a
    claim between two pieces, on the other one. Each valid entry is kept on every claim it names, with how many it
    named (`claims_in_entry`); the missing claims of a piece are listed in one message.

    Every claim an entry names must first be one of the piece's own claims (or, for a claim between two pieces, the
    other piece's claim about it); one that is not refuses the whole answer, a mixed list included, and the error
    lists the piece's claims."""
    valid = open_claims(pieces)
    errs = []
    for a in allocations:
        mine = valid.get(a["fact_id"], [])
        bad = [c for s in a.get("support", ()) for c in s["about"] if claim_key(c) not in {claim_key(m) for m in mine}]
        if bad:
            errs.append(f"{a['fact_id']}: support names {json.dumps(bad)}, not "
                        + ("a claim of this piece" if len(bad) == 1 else "claims of this piece") + ". "
                        + (f"Its claims are {json.dumps(mine)}; name only those" if mine else
                           "This piece has no claims to support; give it no support entries"))
    if errs:
        return errs
    cited = {}
    for a in allocations:
        for s in a.get("support", ()):
            if not s["about"]:
                errs.append(f"{a['fact_id']}: a support entry names no claim in its about")
            elif s["evidence_id"] not in evidence:
                errs.append(f"{a['fact_id']}: support cites {s['evidence_id']}, not a returned passage or observation")
            elif not cited_found(s, evidence[s["evidence_id"]]):
                errs.append(f"{a['fact_id']}: support quote not found in {s['evidence_id']}."
                            + located(cited_text(s), [(s["evidence_id"], evidence[s["evidence_id"]])]))
            else:
                ptr = (s.get("pointer") or "").strip()     # with pointers on each entry records its form
                where = ({"form": s["_form"], "block": s["_block"], "lines": s["_lines"], "ranges": s["_ranges"]}
                         if s.get("_lines") else       # v14: a pointer keeps its block and its list of ranges
                         {**({"form": s["_form"]} if s.get("_form") else {}), **({"pointer": ptr} if ptr else {})})
                for c in s["about"]:
                    cited.setdefault((a["fact_id"],) + claim_key(c), []).append(
                        {"about": dict(c), "evidence_id": s["evidence_id"], "quote": cited_text(s), **where,
                         "claims_in_entry": len(s["about"])})
    facts = {p["fact"] for p in pieces}
    for p in pieces:
        missing = []
        for r in p["review"]:
            c = r["claim"]
            r["support"] = list(cited.get((p["fact"],) + claim_key(c), []))
            if c["kind"] == "piece" and c["target"] in facts:   # the pair's claim, cited on the other piece
                r["support"] += cited.get((c["target"], "piece", p["fact"]), [])
            r["distinct_quotes"] = len({(x["evidence_id"], x["quote"]) for x in r["support"]})
            if r["needs_c4_support"] and not r["support"]:
                missing.append(r)
        if missing:
            errs.append(f"{p['fact']}: no cited filing passage or observation for "
                        + ("this claim the filing's networks do not read: " if len(missing) == 1 else
                           f"these {len(missing)} claims the filing's networks do not read: ")
                        + "; ".join(f"'{r['what']}' ({r['basis']})" for r in missing)
                        + ". You may support them with entries whose about lists the claims each quote supports, e.g. "
                        + json.dumps([r["claim"] for r in missing])
                        + chain_hints([r for r in missing if r.get("chain")])
                        + "; or drop the allocation")
    return errs


CHAIN = "chained reading: evidence for review, not a verdict"
# how a refusal says a chain is available; the loop records a chain selected after it as hinted, so hinted and
# self-found chains are reported apart
CHAIN_HINT = "a chain through the filing's networks exists"


def chain_hints(claims):
    """The refusal's chain remedy: claims that share the same roles in one sentence, roles by their short names as
    the tools show them. It says a chain is available; it cites and selects nothing."""
    short = lambda role: role.rsplit("/", 1)[-1]
    groups = {}
    for r in claims:
        ch = r["chain"]
        k = (tuple(ch["inside_roles"]), tuple(tuple(x) for x in ch["apart_under"]))
        groups.setdefault(k, []).append(r["claim"])
    return "".join(f"; or, for {json.dumps(cs)}, {CHAIN_HINT} (the piece inside the line in "
                   f"{', '.join(short(s) for s in inside)}; the line apart from "
                   + ("it" if len(cs) == 1 else "each") + " under "
                   + ", ".join(f"{q} in {short(s)}" for q, s in apart)
                   + "): you may look up its premises and name them in chains if you judge them true"
                   for (inside, apart), cs in groups.items())


def claim_entry(claim, what, basis, reading):
    """A claim for C8's review, with what the networks read about it: "unverified" (a reading, the effective network
    not verified) or "none" (no network reads it, or the slices do not meet). Only "none" needs C4's citation."""
    reading = "unverified" if reading == "unverified" else "none"
    return {"claim": claim, "what": what, "basis": basis, "reading": reading, "needs_c4_support": reading == "none"}


def open_claims(pieces):
    """Fact ID to the claims a support entry on that piece may name: the piece's own, and the claim another piece
    holds about it (a claim between two pieces may be supported on either)."""
    out = {p["fact"]: [dict(r["claim"]) for r in p["review"]] for p in pieces}
    for p in pieces:
        for r in p["review"]:
            c = r["claim"]
            if c["kind"] == "piece" and c["target"] in out:
                out[c["target"]].append({"kind": "piece", "target": p["fact"]})
    return out


def claim_key(c):
    """A claim's identity: its kind and its target, membership of the line included (a membership claim named with
    another target is not that claim). Kinds keep a fact ID from colliding with "line" or a rival's name."""
    return (c["kind"], c["target"])
