"""Prompt v16, the six rules: code's checks for rules 5 and 6.

The model keeps rules 1 to 4: it picks the row, moves pieces of the line by fact ID, answers COMBINED, NOT_TOLD or
OUTSIDE, and quotes where it saw it. Code checks rules 5 and 6 on every answer the schema admitted, and each refusal is
one sentence that names its rule:

- Rule 5, nothing counted twice: each filed amount is used once; duplicates of one amount count once (an alias is
  never a piece, two members of one duplicate group never both); the line's own amount and its aliases are never a
  piece; a total that contains the line is never a piece; a known overlap between a piece and a printed line, an
  amount already owned or another piece refuses; the pieces never add up to more than the line.
- Rule 6, totals tie and the quote is real: every piece is a monetary fact a tool returned for this line, of the
  filing's own entity at the line's date, in the line's currency, signed against the line (amounts: from the
  calculation paths between them, else from the balances, else refused); a piece of the opposite direction, such as a
  negative-weight calculation child, is a piece, and a negative piece is used as filed; with pieces, the rest of the line goes to a row of the section and each piece to a row of the
  section; the quote occurs, after one fixed normalisation, within ONE block shown in full: the start message, a
  non-search tool result or a passage read. A search snippet is never quotable.

Models supply fact IDs and rows; code supplies every amount and the remainder, and keeps conservation (the pieces and
the remainder add up to the line, or a RuntimeError: a code fault, never a model outcome).

A piece the filing's calculation networks do not place inside the line ("unknown") is accepted on the model's
judgement (`model_judgement`) and must carry its own nonempty quote, which must occur like any quote. Network silence
is neither evidence nor certification; the quote records provenance, not support or membership. Overlaps the networks
leave unknown are recorded on the piece, not refused; known overlaps refuse.

What the trace says about a quote: code found it in one block shown in full (`_quote_source`, with the block's ID, its
kind, the normalisation and the original lines); code did not check that it supports the row or the pieces.

The quote is checked in two tiers, and the source names its tier: "exact", the quote occurs in a block as shown; else
"projection", it occurs within ONE line of the start message or of a tool result (never a passage, in any
representation) once markup the tool renderers print is removed from that line (PROJECTION). Every word, digit, sign and
bracket of the filing's text stays; nothing is removed from the quote but '>>' marks and cell bars, so a fact ID the
model typed must be on the line. Every projected line that holds the quote is recorded; when they show different items
(an item's shown text, its fact IDs in their order and place), the quote is refused as ambiguous; only an item shown
again counts once.
"""
import re
from decimal import Decimal
from types import SimpleNamespace

from . import calc_network as cn
from . import open_claims, passage_facts, steps, tagless
from .answer_keys import precise_duplicate
from .filing import fact_ident, in_period, line_ident, tag_of
from .spread import balance_of
from . import template

PROVENANCE = ("occurrence only: code found the quote in one block shown in full; it did not check that the quote "
              "supports the row or the pieces")
JUDGEMENT = ("model_judgement: the filing's calculation networks do not place this piece inside the line, so it stands "
             "on the model's judgement and its own quote; network silence is neither evidence nor certification, and "
             "the quote records provenance, not support")
NETWORKS = "the company's calculation network reads it inside the line (company linkbases only; not certification)"
# the movable-fact marker of the root-cause fixes (--root-cause-fixes): the mark a tool result shows beside a movable fact. It is the
# validator's own exemption (needs_quote of the reading amounts() takes from Relations.inside), never a second policy.
INSIDE_MARK = "(inside the line: quote may be empty)"
QUOTE_MARK = "(needs its own quote)"


def needs_quote(reading):
    """The one predicate of the empty-quote exemption: a piece whose membership reading is "unknown" stands on the
    model's judgement and needs its own quote. amounts() and the movable-fact mark both use it."""
    return reading == "unknown"


def quote_mark(f, line, index, rel, fid):
    """The mark for a movable fact, against the episode's current line: the reading of Relations.inside for the
    fact's own key (its dimensions included) and the line, exactly as amounts() reads it."""
    x = index.get(fid)
    if x is None:
        return QUOTE_MARK
    reading, _ = rel.inside(fact_ident(f, x), line_ident(f, line["concept"]))
    return QUOTE_MARK if needs_quote(reading) else INSIDE_MARK


def marked(eligible, f, line, index, rel):
    """Movable-fact marks: `eligible` (steps.Eligibility.of) with `quote_mark` added to each allocatable record; nothing else of the
    record changes."""
    def of(fid):
        r = eligible(fid)
        return {**r, "quote_mark": quote_mark(f, line, index, rel, fid)} if r.get("allocatable") else r
    return of


def relationship(p):
    """The marked refusal (--root-cause-fixes): how code reads the piece against the line, in words (no tag, no concept name)."""
    sign = "added to the line" if p["r"] == 1 else "subtracted from the line"
    how = "by a calculation path" if p["r_basis"] == "path" else "by the two balances"
    return f"{sign} {how}; membership {p['membership']['reading']}: {p['membership']['basis']}"


def without_shown_ids(st, blocks):
    """Quote matching without shown IDs: (the answer as checked, how many quotes changed). A quote that copied a fact ID read_passage shows inline
    is checked without it when, without it, it occurs in a passage read: the IDs are added by code, so they neither
    break the match nor enter it. The recorded answer keeps what the model wrote."""
    passages = [b for b in blocks if b[1] == "passage read"]
    n = 0

    def clean(q):
        nonlocal n
        if not q or not passage_facts.SHOWN_ID.search(q):
            return q
        c = passage_facts.without_ids(q)
        if find_quote(c, passages, [])[0] is None:
            return q
        n += 1
        return c
    out = {**st, "quote": clean(st.get("quote", ""))}
    if st.get("pieces"):
        out["pieces"] = [{**x, "quote": clean(x.get("quote") or "")} for x in st["pieces"]]
    return out, n

# The fixed quote normalisation: applied, in this order, to the quote and to every block alike.
# Nothing else is changed: no number, sign or word is removed to make a match. Gaps are allowed only where the quote
# marks them with "..." or "…"; each part must then occur in order within one block.
DASHES = re.compile("[‐‑‒–—―−]")
QUOTE_MARKS = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
                             "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"'})
DIGIT_COMMA = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
GAP = re.compile(r"\.\.\.|…")
EDGES = " .\"'"
# U+200B (zero-width space) becomes a space only in an otherwise empty table cell, the text after a '|' up to the next
# '|' or the end of the line holding nothing but spaces, tabs, no-break spaces and U+200B. It runs before whitespace is
# collapsed, so line and cell boundaries stay. Every other U+200B stays as it is (a known gap).
ZW = "​"
EMPTY_CELL = re.compile(r"(?<=\|)[ \t ​]+(?=\||$)", re.M)
# Within one line, a run of empty cells ("| | |") is one separator, so a quote that leaves out column padding still
# occurs. It runs after empty cells lose their U+200B and before whitespace is collapsed, so a line break is never
# crossed. Nothing holding a letter, a digit or a sign is removed.
EMPTY_RUN = re.compile(r"\|(?:[ \t ]+\|)+")      # spaces, tabs, no-break spaces; never a line break
NORMALISATION = (
    ("case", str.casefold),
    ("dashes", lambda s: DASHES.sub("-", s)),
    ("quote_marks", lambda s: s.translate(QUOTE_MARKS)),
    ("digit_grouping_commas", lambda s: DIGIT_COMMA.sub("", s)),
    ("dollar_sign", lambda s: s.replace("$", "")),
    ("zero_width_space_empty_cell", lambda s: EMPTY_CELL.sub(lambda m: m.group().replace(ZW, " "), s)),
    ("empty_cell_run", lambda s: EMPTY_RUN.sub("|", s)),
    ("whitespace", lambda s: " ".join(s.split())),
)
CONTENT = re.compile(r"[^\W_]")
POLICY = ("case folded; unicode dashes and minus to '-'; curly quotation marks to straight; commas between digit "
          "groups removed; '$' removed; U+200B in an otherwise empty table cell (spaces, tabs, no-break spaces and "
          "U+200B between '|' and the next '|' or the line's end) to a space, every other U+200B kept; a run of empty "
          "table cells within one line ('| | |') to one '|'; whitespace runs collapsed to one space; the quote split only at '...' or '…', each part trimmed of spaces, periods "
          "and quotation marks at its ends, every part holding a letter or a digit, and the parts found in order "
          "within one block, a number at either end of a part whole there: not continued by a digit or by '.' or "
          "',' and a digit, and not preceded by '(' or by '-' across spaces unless the '-' joins two digits (a "
          "range such as 2024-2025). Accepted limitations: a word cut short at its end (other asset in other "
          "assets) and a dropped percent sign still match; this is not a full reading of signs")


# The projection tier: only renderer-identified spans are removed, signs and source brackets stay, passages are out
# in every representation. A start-message line is used as shown. A tool result's item line (Memory.full: "  - " and
# item_line) loses its item prefix; on a printed_page row (tools.page_row) also the '>>' mark and the cell bars
# ("marks"), then the facts trailer, the rule marks after the cells and each bracket whose entries are all fact IDs the
# trailer lists ("marks_ids"); a calc_children or breakdowns item is also read as 'concept label amount unit'
# ("rendered"), only where it shows an amount in a currency. '(blank)' always stays: the shown text cannot tell an
# empty amount column from currency padding, so it is a boundary. The quote loses only '>>' marks and cell bars.
ITEM = "  - "
SHOWN_IN = re.compile(r"^\(shown in O\d+\) ")       # Memory.add, on an item shown before
FACTS = re.compile(r"  \[facts: ([^\]]*)\]$")
RULES_END = re.compile(r"(?: \((?:double )?rule (?:above|below)\))+$")
BRACKETS = re.compile(r"\[([^\[\]]*)\]")
HEAD = re.compile(r"^O\d+ ([a-z_]+)\(")
CALC = re.compile(r"^(\S+) \((.*)\) weight [-\d./]+: (-?\d[\d.]* [a-z]{3})\b")
LOCATION = re.compile(r"^\[location\] the filing states (\S+) \((.*)\) is included in \S+: (-?\d[\d.]* [a-z]{3})\b")
QUOTE_MARKS_OURS = re.compile(r"(?:^|(?<=\s))>>(?=\s|$)|\|")
PROJECTION = ("tried only when the quote occurs in no block as shown; within ONE line of the start message (as shown) "
              "or ONE item of a tool result other than read_passage, after the fixed normalisation; only markup our "
              "renderers print is removed from the item: 'marks' its item prefix ('- ', '(shown in O2)') and, on a "
              "printed_page row, "
              "the '>>' mark and the cell bars; 'marks_ids' also the facts trailer, the rule marks after the cells "
              "and brackets listing only that item's fact IDs; 'rendered' reads a calc_children or breakdowns item "
              "as 'concept label amount unit'. '(blank)' stays as a boundary. The quote loses only '>>' marks and "
              "cell bars; every other word, digit, sign and bracket of it must be on the line. Every line holding the "
              "quote is a candidate; candidates showing different items (the item's shown text, with its fact IDs "
              "in their order) refuse the quote as ambiguous, and only an item shown again counts once")
RENDERED = ("occurrence only, code-rendered evidence: code found the quote in one line of the start message or a tool "
            "result once its own markup was removed (projection); it did not check that the quote supports the row or "
            "the pieces")



def shown_name(key):
    """steps.name for a refusal the model reads: the line's ID and its members' labels (tags hidden, tagless.py)."""
    return tagless.show(key[0]) + "".join(f" at {tagless.member(a, a)}={tagless.member(m, m)}" for a, m in key[1])

def squash(s):
    return " ".join(s.split())


def projections(line, tool):
    """[(name, text)]: the projections of one shown line of a tool result whose call is `tool` (None: the start
    message) (PROJECTION)."""
    if tool is None:
        return [("marks", normal(line))]
    if not line.startswith(ITEM):
        return []
    page = tool == "printed_page"
    b = SHOWN_IN.sub("", line[len(ITEM):])
    b = b[3:] if page and b.startswith((">> ", ">~ ", "?> ")) else b     # the member marks too
    m = FACTS.search(b)
    ids = trailer_ids(m)
    body = b[:m.start()] if m else b
    body = RULES_END.sub("", body) if page else body
    plain = BRACKETS.sub(lambda x: " " if ids and {e.strip() for e in x.group(1).split(",")} <= ids else x.group(),
                         body)
    # bars go after the normalisation, which needs them to find empty cells (EMPTY_CELL, EMPTY_RUN)
    out = [(name, squash(normal(t).replace("|", " ") if page else normal(t))) for name, t in
           (("marks", b), ("marks_ids", plain))]
    if tool in ("calc_children", "breakdowns"):
        r = next((r for r in (p.match(normal(body)) for p in (CALC, LOCATION)) if r), None)
        if r:
            out.append(("rendered", " ".join(r.groups())))
    return out


def trailer_ids(m):
    """The fact IDs a facts trailer match lists (none without one)."""
    return {x.split(" (")[0].strip() for x in m.group(1).split(", ")} if m else set()


def binding(line, tool):
    """What a projected line binds a quote to: the item as shown, its fact IDs in their order and place (a set would
    hide two periods' facts swapped between columns); an item shown again ('(shown in O2)') binds the same as its first
    showing."""
    return squash(line if tool is None else SHOWN_IN.sub("", line[len(ITEM):]))


def projected(ps, blocks):
    """Every line of the start message or a tool result (never a passage, nor read_passage's result), in block order,
    holding the quote's parts `ps` once projected, under the first rule that holds them: [{block, kind, line, rule,
    text, span, binding}]."""
    norm = [x for x in (squash(QUOTE_MARKS_OURS.sub(" ", p)) for _, p in ps) if x]
    out = []
    if not norm:
        return out
    for b, kind, text in blocks:
        if kind not in ("start message", "tool result"):
            continue
        lines = text.split("\n")
        tool = None
        if kind == "tool result":
            h = HEAD.match(lines[0])
            if not h:
                continue
            tool = h.group(1)
        for ln in lines:
            for name, t in projections(ln, tool):
                at = match(norm, t)
                if at is not None:
                    out.append({"block": b, "kind": kind, "line": ln.strip(), "rule": name, "text": t,
                                "span": [[p, j] for p, j in zip(norm, at)], "binding": binding(ln, tool)})
                    break
    return out


def normal(s):
    for _, rule in NORMALISATION:
        s = rule(s)
    return s


def changed_by(s):
    """The rules of the fixed normalisation that change `s`, in order (logged on each quote)."""
    out = []
    for name, rule in NORMALISATION:
        t = rule(s)
        if t != s:
            out.append(name)
        s = t
    return out


def parts(quote):
    """[(raw part, normalised part)] of a quote, split only at marked gaps; empty parts dropped."""
    out = []
    for raw in GAP.split(quote or ""):
        n = normal(raw).strip(EDGES)
        if n:
            out.append((raw.strip(), n))
    return out


def whole(text, j, p):
    """Whether the occurrence of part `p` at `j` in normalised `text` keeps a number at either end of the part whole:
    a part ending in a digit is not continued by a digit, or by '.' or ',' and a digit; a part beginning with a digit
    is not preceded by a digit, by a digit and '.' or ',', by '(' across spaces, or by '-' across spaces unless the
    '-' sits directly between two digits (a range such as 2024-2025)."""
    end, before = j + len(p), text[max(0, j - 64):j]
    if p[-1:].isdigit() and (text[end:end + 1].isdigit() or
                             text[end:end + 1] in (".", ",") and text[end + 1:end + 2].isdigit()):
        return False
    if not p[:1].isdigit():
        return True
    if before[-1:].isdigit() or before[-1:] in (".", ",") and before[-2:-1].isdigit():
        return False
    sign = before.rstrip(" \t")
    if sign[-1:] == "(":
        return False
    return sign[-1:] != "-" or (sign == before and sign[-2:-1].isdigit())


def match(ps, text):
    """The offsets of the parts `ps`, in order, within `text`, each occurrence whole; every occurrence of a part is
    tried, earliest first. None when there is no such match."""
    i, at = 0, []
    for p in ps:
        j = text.find(p, i)
        while j >= 0 and not whole(text, j, p):
            j = text.find(p, j + 1)
        if j < 0:
            return None
        at.append(j)
        i = j + len(p)
    return at


def in_order(ps, text):
    return match(ps, text) is not None


def loose(ps, text):
    """The match before the whole-number guard; used only to name the guard on a refusal."""
    i = 0
    for p in ps:
        j = text.find(p, i)
        if j < 0:
            return False
        i = j + len(p)
    return True


def eligible_blocks(start, mem):
    """[(block ID, kind, text)]: the start message S, each passage read, each non-search observation shown in full.
    A passage comes before the read_passage result that showed it, so a quote from it keeps the passage's own ID;
    that result is kind "passage tool result", so the projection tier never reads a passage through it."""
    out = [("S", "start message", start)]
    out += [(p, "passage read", t) for p, t in mem.passage_texts().items()]
    out += [(o, "passage tool result" if mem.obs[o]["tool"] == "read_passage" else "tool result", t)
            for o, t in sorted(mem.quotable_texts().items(), key=lambda x: int(x[0][1:]))]
    return out


def search_blocks(mem):
    return [(o, t) for o, t in mem.shown_texts().items() if mem.obs[o]["tool"] == "search_notes"]


def find_quote(quote, blocks, searched, what="the quote"):
    """(source, refusal): `source` {block, kind, tier, normalisation, original_lines, label} when the quote occurs
    within one eligible block as shown (tier "exact") or within one projected line (tier "projection", with
    `projection`), else a refusal {rule 6, kind, text}: quote_empty, quote_no_content, quote_ambiguous (projected
    lines showing different items, all listed), quote_from_search or quote_not_found; a refusal of either of the last
    two kinds that the whole-number guard caused (the quote would have matched an eligible block without it) also
    carries guard "whole_number"."""
    ps = parts(quote)
    if not ps:
        return None, {"rule": 6, "kind": "quote_empty",
                      "text": f"Rule 6: {what} is empty; copy the words you relied on from the start message, a tool "
                              "result or a passage you read."}
    norm = [p for _, p in ps]
    if not all(CONTENT.search(p) for p in norm):
        return None, {"rule": 6, "kind": "quote_no_content",
                      "text": f"Rule 6: {what} has a part with no words or numbers; copy the words you relied on."}
    for b, kind, text in blocks:
        n = normal(text)
        at = match(norm, n)
        if at is not None:
            lines = [ln.strip() for ln in text.split("\n") if any(p in normal(ln) for p in norm)]
            return {"block": b, "kind": kind, "quote": quote, "tier": "exact",
                    "normalisation": {"policy": POLICY, "exact": in_order([r for r, _ in ps], text),
                                      "changed_quote": changed_by(quote), "changed_block": changed_by(text),
                                      "span": [[p, j] for p, j in zip(norm, at)]},
                    "original_lines": lines, "label": PROVENANCE}, None
    found = projected(ps, blocks)
    candidates = [{"block": c["block"], "line": c["line"], "rule": c["rule"], "binding": c["binding"]}
                  for c in found]
    bindings = {c["binding"] for c in found}
    if len(bindings) > 1:
        where = ", ".join(dict.fromkeys(c["block"] for c in found))
        return None, {"rule": 6, "kind": "quote_ambiguous", "quote_length": len(quote), "candidates": candidates,
                      "text": f"Rule 6: {what} fits {len(bindings)} different items once our markup is left out "
                              f"({where}); quote the item with its fact ID, or more of its line, so it names one."}
    if found:
        pj = found[0]
        return {"block": pj["block"], "kind": pj["kind"], "quote": quote, "tier": "projection",
                "projection": {"policy": PROJECTION, "rule": pj["rule"], "text": pj["text"],
                               "quote_length": len(quote), "candidates": candidates},
                "normalisation": {"policy": POLICY, "exact": False, "changed_quote": changed_by(quote),
                                  "changed_block": changed_by(pj["line"]), "span": pj["span"]},
                "original_lines": [pj["line"]], "label": RENDERED}, None
    texts = [normal(t) for _, _, t in blocks]
    # The whole-number guard, machine-readable beside the unchanged text: the quote would have matched an eligible
    # block without it. Kept on either refusal kind.
    guard = {"guard": "whole_number"} if any(loose(norm, t) for t in texts) else {}
    if any(in_order(norm, normal(t)) for _, t in searched):
        return None, {"rule": 6, "kind": "quote_from_search", **guard,
                      "text": f"Rule 6: {what} comes from a search result, which only points to a passage; read the "
                              "passage and quote it."}
    miss = next((r for r, p in ps if not any(match([p], t) is not None for t in texts)), None)
    near = steps.nearest(quote, [(b, t) for b, _, t in blocks])
    tail = f" Nearest shown line: \"{near[1]}\"" if near else ""
    if miss is None:
        return None, {"rule": 6, "kind": "quote_not_found", **guard,
                      "text": f"Rule 6: each part of {what} occurs, but not in this order within one block shown in "
                              f"full; quote from one block and mark a gap with '...'.{tail}"}
    return None, {"rule": 6, "kind": "quote_not_found", **guard,
                  "text": f"Rule 6: {what} is not in any one block shown in full ('{miss}' not found).{tail}"}


def refusal(rule, kind, text):
    return {"rule": rule, "kind": kind, "text": text}


def id_refusal(fid, state):
    """Rule 5 or 6 for a piece's fact ID outside `allocatable_ids`, by the reason code recorded for it."""
    why = state.get("excluded", {}).get(fid)
    if fid == state.get("line_fact") or why == steps.OWN:
        return refusal(5, "own_amount", f"Rule 5: {fid} is the line's own amount, so it cannot be a piece; to keep "
                                        "the line whole, give no pieces.")
    if why and why.startswith("alias of "):
        canon = why[len("alias of "):]
        return refusal(5, "alias", f"Rule 5: {fid} repeats the filed amount {canon}, so it counts once: use {canon}.")
    if why == steps.NOT_AMOUNT:
        return refusal(6, "not_amount", f"Rule 6: {fid} is not a monetary amount, so it cannot be a piece.")
    return refusal(6, "not_returned", f"Rule 6: {fid} was not returned as an amount by a tool for this line.")


def check_answer(tax, f, line, st, state, printed, owned, index, rel, blocks, searched, root_cause=False):
    """(source, pieces, remainder, refusals) for one admitted v16 answer. Nothing is committed here. `root_cause`
    (--root-cause-fixes): only the text of the judgement_without_quote refusal differs."""
    out = []
    src, r = find_quote(st.get("quote", ""), blocks, searched)
    if r:
        out.append(r)
    given = st.get("pieces") or []
    section = state.get("section_rows") or state["rows"]
    if state.get("queued"):     # a queued line sees every row of its statement; pieces and the rest stay in the
        t = template.active()   # answer row's section (C4b; C3b)
        own = t.section_of(st.get("row")) if st.get("row") in t.rows else None
        section = [r for r in section if own and t.section_of(r) == own]
    if given and st.get("row") in template.active().rows and st.get("row") not in section:   # C3b: one section
        out.append(refusal(6, "more_than_one_section", f"Rule 6: the rest goes to {st.get('row')}, a row of another "
                                                       "section; pieces and the rest must be rows of one section."))
    elif given and st.get("row") not in section:
        out.append(refusal(6, "remainder_row", f"Rule 6: with pieces, row is the section row that keeps the rest of "
                                               f"the line, so it cannot be {st.get('row')}."))
    ids, groups, seen_group = [], state.get("duplicate_groups", {}), {}
    ok = []
    for x in given:
        fid = x.get("fact_id")
        if x.get("row") in template.active().rows and x.get("row") not in section:   # C3b: one section (MVP limit)
            out.append(refusal(6, "more_than_one_section", f"Rule 6: {fid} goes to {x.get('row')}, a row of another "
                                                           "section; pieces and the rest must be rows of one section."))
            continue
        if x.get("row") not in section:
            out.append(refusal(6, "piece_row", f"Rule 6: {fid} goes to {x.get('row')}, which is not a row of this "
                                               "section, so the totals could not tie."))
            continue
        if fid in ids:
            out.append(refusal(5, "used_twice", f"Rule 5: {fid} is used twice; each filed amount counts once."))
            continue
        ids.append(fid)
        if fid not in state.get("allocatable_ids", ()):
            out.append(id_refusal(fid, state))
            continue
        g = tuple(groups.get(fid) or ())
        if g and g in seen_group:
            out.append(refusal(5, "duplicate_group", f"Rule 5: {seen_group[g]} and {fid} are one filed amount "
                                                     "reported twice with different decimals or values; use one."))
            continue
        if g:
            seen_group[g] = fid
        ok.append(x)
    pieces, left = [], Decimal(line["amount"])
    if not out:
        pieces, left, perrs = amounts(tax, f, line, ok, printed, owned, index, rel)
        out += perrs
    for p, x in zip(pieces, ok):
        q = x.get("quote") or ""
        if p["model_judgement"] and not q.strip():
            out.append(refusal(6, "judgement_without_quote",
                               f"{p['fact']} requires its own matching filing quote. Its calculation relationship "
                               f"to the line ({relationship(p)}) does not qualify for the empty-quote exemption. "
                               "Other eligibility checks still apply." if root_cause else
                               f"Rule 6: the filing's calculations do not place {p['fact']} inside the line, so that "
                               "piece needs its own quote from something shown in full."))
        elif q.strip():
            psrc, pr = find_quote(q, blocks, searched, what=f"the quote for {p['fact']}")
            if pr:
                out.append(pr)
            else:
                p["quote_source"] = psrc
    if out:
        return None, [], Decimal(line["amount"]), out
    if sum((p["amount"] for p in pieces), Decimal(0)) + left != Decimal(line["amount"]):
        raise RuntimeError(f"conservation broken for {line['concept']}")    # a code fault, never a model outcome
    return src, pieces, left, []


def ratios(f, line, piece):
    """The products of the weights on every calculation path from the line's concept down to the piece's (C3b). Each
    path lies within one calculation network of one role (calc_network.reach_down, usable relationships only, cycles
    cut); no path takes edges from two networks or crosses into another role. By tag, through the arcs as filed
    (Filing.tag_arcs): a split piece's key reads its tag, and compatible() keeps the path to amounts of one member
    context."""
    out, filed = set(), SimpleNamespace(arcs=getattr(f, "tag_arcs", f.arcs))
    line, piece = tag_of(line), tag_of(piece)
    for role in sorted({a["role"] for a in filed.arcs("cal")}):
        for x in cn.reach_down(cn.networks(filed, role), line).values():
            for chain in x["reach"].get(piece, ()):
                k = Decimal(1)
                for s in chain:
                    k *= s["weight"]
                out.add(k)
    return out


def compatible(f, index, line, x):
    """Whether a calculation path may set the piece's direction: the line's own fact and the piece's have the same
    context (entity, period, dimensions) and the same unit. A calculation relates amounts of one context and unit; a
    dimensioned piece of an undimensioned line is not one of its summed items."""
    own = index.get(line.get("fact"))
    return (own is not None and f.contexts[own["context"]] == f.contexts[x["context"]]
            and f.unit_measures.get(own["unit"]) == f.unit_measures.get(x["unit"]))


def amounts(tax, f, line, given, printed, owned, index, rel):
    """(pieces, what is left, refusals): code finds each piece's amount by fact ID and checks it (rules 5 and 6). C3b:
    each piece has a direction r relative to the line, from the calculation paths between them (each product +-1 and all
    equal, else "sign_conflict"), else from the balances (+1 same, -1 opposite), else "balance_unknown". The piece
    counts r x its filed value; the rest is the line less those. When an r comes from balances, the rest must be zero or
    of the line's sign ("exceeds_line"). A negative piece is used as filed."""
    whole = line_ident(f, line["concept"])
    up = {p for r in rel.cal for p in rel.cal[r] if p != whole[0] and whole[0] in rel.below("calc", p)}
    rivals = ([line_ident(f, c) for c in printed if c != line["concept"] and line_ident(f, c)[0] not in up]
              + [tuple(k) for k in owned])
    line_bal = balance_of(tax, f, line["concept"])
    pieces, out, keys, left, by_balance = [], [], [], Decimal(line["amount"]), False
    for a in given:
        fid = a["fact_id"]
        x = index.get(fid)
        if x is None:
            out.append(refusal(6, "not_returned", f"Rule 6: {fid} is not a fact of this filing."))
            continue
        ctx = f.contexts[x["context"]]
        key = fact_ident(f, x)
        dims = key[1]
        same = [y for y in f.by_concept.get(x["concept"], []) if f.contexts[y["context"]] == ctx
                and f.unit_measures.get(y["unit"]) == f.unit_measures.get(x["unit"]) and y["value"] != ""]
        bal = balance_of(tax, f, x["concept"])
        clash, unknown = None, []
        for k, what in [(r, "the printed line or owned amount") for r in rivals] + [(k, "the piece") for k, _ in keys]:
            res, basis = rel.overlap(key, k)
            if res == "overlap":
                clash = clash or (shown_name(k), what, basis)
            elif res != "disjoint":
                unknown.append(open_claims.entry(rel, k, res, basis, fid))
        # a path counts only between compatible facts; otherwise the balance, bounded by exceeds_line
        paths = (sorted(ratios(f, line["concept"], x["concept"]))
                 if x["concept"] != line["concept"] and compatible(f, index, line, x) else [])
        rel_dir = (int(paths[0]) if len(paths) == 1 and abs(paths[0]) == 1 else None) if paths else (
            None if bal is None or line_bal is None else 1 if bal == line_bal else -1)
        r = None
        if ctx["entity"] != f.entity:
            r = refusal(6, "entity", f"Rule 6: {fid} is not the company's own amount.")
        elif not in_period(ctx, line.get("period") or ("instant", f.period_end)):
            r = refusal(6, "period", f"Rule 6: {fid} is not an amount of the line's period.")
        elif f.currency(x["unit"]) != line["unit"]:
            r = refusal(6, "currency", f"Rule 6: {fid} is in {f.currency(x['unit'])}, the line in {line['unit']}.")
        elif precise_duplicate(same) is None:
            r = refusal(5, "conflicting_duplicates", f"Rule 5: {fid} is filed more than once with different "
                                                     "amounts, so it cannot be counted once.")
        elif Decimal(x["value"]) != Decimal(precise_duplicate(same)["value"]):
            p = precise_duplicate(same)["id"]
            r = refusal(5, "rounded_duplicate", f"Rule 5: {fid} is a lower-precision duplicate of {p}; if you "
                                                f"intend this amount as a piece, use {p}, and the other checks "
                                                "still apply.")
        elif key[0] in up:
            r = refusal(5, "ancestor", f"Rule 5: {fid} is a total that contains the line, so it cannot be a piece.")
        elif clash:
            r = refusal(5, "known_overlap", f"Rule 5: {fid} overlaps {clash[1]} {clash[0]} ({clash[2]}), so it would "
                                            "be counted twice.")
        elif paths and rel_dir is None:
            r = refusal(6, "sign_conflict", f"Rule 6: the filing's calculations relate {fid} to the line with "
                                            "different or partial weights, so its sign cannot be checked.")
        elif rel_dir is None:
            r = refusal(6, "balance_unknown", f"Rule 6: {fid} or the line has no known debit or credit balance, so "
                                              "its sign cannot be checked.")
        open_claims.attempt(rel, line, fid, unknown, r)      # cross-dimension policy: logged in a root-cause job only
        if r:
            out.append(r)
            continue
        by_balance = by_balance or not paths
        amount = rel_dir * Decimal(x["value"])
        left -= amount
        keys.append((key, fid))
        reading, basis = rel.inside(key, whole)
        judged = needs_quote(reading)
        pieces.append({"fact": fid, "concept": x["concept"], "dims": dims, **({"tag": key[0]} if x.get("piece") else {}),
                       "row": a["row"], "amount": amount,
                       "value": Decimal(x["value"]), "r": rel_dir, "r_basis": "path" if paths else "balance",
                       "decimals": x.get("decimals"), "source": line["concept"], "by": "C4", "membership": {"reading": reading, "basis": basis},
                       "model_judgement": judged, "basis": JUDGEMENT if judged else NETWORKS,
                       "unknown_overlaps": unknown, **open_claims.piece_marks(rel, unknown)})
    v = Decimal(line["amount"])
    if by_balance and not out and left != 0 and (v == 0 or (left > 0) != (v > 0)):
        out.append(refusal(5, "exceeds_line", f"Rule 5: the pieces add up to more than the line's {v:,} "
                                              f"{line['unit']}, so the rest would change sign."))
    return pieces, left, out


def checker(tax, f, line, printed, owned, start, index=None, rel=None, root_cause=False):
    """The check function of a v16 loop, bound to the line and to `owned`, the (concept, dims) keys already claimed in
    the caller's own working registry. On success the answer carries `_quote_source`, `_pieces` and `_remainder`; on
    a refusal, `_refusals` ({rule, kind, text}) for the loop's counts. Returns the refusal sentences. `root_cause`
    (--root-cause-fixes): the marked refusal text, and quotes are checked without the fact IDs read_passage adds (
    `_quote_ids_removed` counts the quotes this changed)."""
    index = index if index is not None else steps.fact_index(f)
    if rel is None and root_cause:      # a root-cause-fixes path names its policy: no default can pick it
        raise ValueError("six_rules.checker: a root-cause-fixes path (root_cause) is given its Relations; none was passed")
    rel = rel if rel is not None else steps.relations(f)     # otherwise: the legacy policy

    def check(st, state, shown, mem):
        blocks, given = eligible_blocks(start, mem), st
        if root_cause:
            given, n = without_shown_ids(st, blocks)
            if n:
                st["_quote_ids_removed"] = n
        r7_start = len(getattr(rel, "r7_log", ()))
        src, pieces, left, refused = check_answer(tax, f, line, given, state, printed, owned, index, rel,
                                                  blocks, search_blocks(mem), root_cause=root_cause)
        open_claims.closed(rel, r7_start, refused)
        st["_refusals"] = refused
        if refused:
            return [r["text"] for r in refused]
        st.update(_quote_source=src, _pieces=pieces, _remainder=left)
        return []
    return check
