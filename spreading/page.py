"""The statement as the company printed it: a text reconstruction of the statement's
table(s) in the filing's main document (inline XBRL HTML), read with the standard library, one entry per printed row.
It is not a browser rendering.

What is extracted from the document: each row's label and amount strings as printed, in printed order, the fact IDs
tagged on the amounts, and each cell's physical column (colspan and rowspan followed). What is read from inline styles
only, and reported as not detected when absent: a row's left offset (padding, margin-left, text-indent, leading
spaces) and the rules (borders) on its cells. Stylesheets, classes and bold are not read; bold is not used to
read structure. Amounts stay text: nothing
from this module enters a calculation. Only a document that verifies against the filing's manifest or
its supplement is read (edgar.verified_main_document).

    doc, why = page.main_document(dir)
    st = page.statement(doc, target_ids={"f-191"}, statement_keys={"f-191": (concept, context, unit), ...}, why=why)
    # {"rows": [...], "tables": [i, ...], "before": text printed before the first table, "not_joined": [...]}
    # or {"failed": code, "why": text}, code one of NO_MAIN_DOCUMENT, FACT_NOT_IN_TABLE, AMBIGUOUS_TABLE,
    # NOT_THE_STATEMENT
"""
import functools
import re
from html.parser import HTMLParser
from pathlib import Path

UNIT_PT = {"pt": 1.0, "px": 0.75, "in": 72.0, "em": 10.0, "rem": 10.0}
NBSP_PT = 2.5          # a leading non-breaking space as indent, about a quarter em at 10pt
FACT_TAGS = ("ix:nonfraction", "ix:nonnumeric")
ALONE = {"$", "(", ")", "%", ")%", "€", "£"}     # printed beside an amount in a cell of its own
BEFORE_CHARS = 300     # text printed between the previous table and this one, kept from its end
MIN_FACTS = 3          # a table joins the statement only with at least this many of the statement's facts ...
MIN_SHARE = 0.8        # ... making at least this share of its tagged facts, ...
BETWEEN_CHARS = 200    # ... with at most this much text printed between it and the statement (a page footer), ...
#                        and none of its amounts repeating a concept and date the statement already printed


def main_document(directory):
    """(path, None) for the filing's inline XBRL main document when it verifies, else (None, why)."""
    d = Path(directory)
    sig = tuple((q.name, q.stat().st_size, q.stat().st_mtime_ns) for q in sorted(d.iterdir()) if q.is_file())
    return _verified(str(d), sig)


@functools.lru_cache(maxsize=32)
def _verified(directory, sig):      # `sig`: the folder's files, sizes and times, so a change is read again
    from .edgar import verified_main_document
    return verified_main_document(directory)


def length_pt(value):
    m = re.fullmatch(r"(-?\d*\.?\d+)(pt|px|in|em|rem)?", value.strip())
    return float(m.group(1)) * UNIT_PT[m.group(2) or "px"] if m else 0.0


def props(style):
    out = {}
    for part in style.split(";"):
        k, _, v = part.partition(":")
        if k.strip():
            out[k.strip().lower()] = v.strip()
    return out


def left_pt(style):
    """The left offset a style gives: padding-left (or the padding shorthand's left), margin-left, text-indent."""
    p, total = props(style), 0.0
    if "padding-left" in p:
        total += length_pt(p["padding-left"])
    elif "padding" in p:
        v = p["padding"].split()
        total += length_pt(v[{1: 0, 2: 1, 3: 1, 4: 3}[len(v)]]) if 0 < len(v) <= 4 else 0.0
    for k in ("margin-left", "text-indent"):
        total += length_pt(p.get(k, ""))
    return total


def border(style, side):
    """'double', 'single' or None for one side of a cell's border, from its inline style."""
    p = props(style)
    v = p.get(f"border-{side}") or p.get("border") or ""
    if p.get(f"border-{side}-style") in ("none", "hidden") or not v:
        return None
    if re.search(r"\b(none|hidden)\b", v) or re.match(r"0(px|pt)?\b", v):
        return None
    return "double" if "double" in v else "single" if re.search(r"solid|dotted|dashed", v) else None


class _Tables(HTMLParser):
    """Top-level tables as rows of cells {text, style, inner, facts}, with the text printed before each. A table
    nested in a cell is read into that cell."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables, self.before, self.depth = [], [], 0
        self.rows = self.row = self.cell = None
        self.outside = ""
        self.spans, self.busy, self.col = {}, set(), 0     # rowspans still to run, columns they fill, next column

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "table":
            self.depth += 1
            if self.depth == 1:
                self.rows, self.spans = [], {}
                self.before.append(clean(self.outside)[-BEFORE_CHARS:])
                self.outside = ""
            return
        if self.depth == 0:
            if tag in ("div", "p", "br", "tr"):
                self.outside += "\n"
            return
        if tag == "tr" and self.depth == 1:
            self.row, self.col = [], 0
            self.busy = {c for c, n in self.spans.items() if n > 0}
            self.spans = {c: n - 1 for c, n in self.spans.items() if n > 1}
        elif tag in ("td", "th") and self.depth == 1 and self.row is not None:
            while self.col in self.busy:
                self.col += 1
            span, down = (max(1, int(x)) if (x or "").strip().isdigit() else 1
                          for x in (a.get("colspan"), a.get("rowspan")))
            self.cell = {"text": "", "style": a.get("style") or "", "inner": [], "facts": [], "col": self.col,
                         "span": span}
            if down > 1:
                for c in range(self.col, self.col + span):
                    self.spans[c] = down - 1
            self.col += span
            self.row.append(self.cell)
        elif self.cell is not None:
            self.cell["inner"].append(a.get("style") or "")
            if tag in FACT_TAGS and a.get("id"):
                self.cell["facts"].append(a["id"])

    def handle_endtag(self, tag):
        if tag == "table" and self.depth:
            if self.depth == 1 and self.rows is not None:
                self.tables.append(self.rows)
                self.rows = None
            self.depth -= 1
        elif tag == "tr" and self.depth == 1 and self.row is not None:
            if self.rows is not None:
                self.rows.append(self.row)
            self.row = None
        elif tag in ("td", "th") and self.depth == 1:
            self.cell = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell["text"] += data
        elif self.depth == 0:
            self.outside = (self.outside + data)[-4 * BEFORE_CHARS:]


ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿"), " ")     # spacer cells print only these


def clean(text):
    text = text.replace("\xa0", " ").translate(ZERO_WIDTH)
    return re.sub(r"[ \t\r\f\v]+", " ", re.sub(r"\s*\n\s*", " / ", text)).strip(" /")


def printed_row(cells):
    """One printed row: the label (the first cell with text, if it has a letter; the share counts some companies
    tag inside it are kept as its `label_facts`), its left offset, the rules on its cells, and the amount cells
    with bare '$', '(' and ')' cells joined to their amount; None if empty."""
    texts = [c for c in cells if clean(c["text"])]
    if not texts:
        return None
    first = texts[0]
    labelled = bool(re.search(r"[A-Za-z]", clean(first["text"])))
    lead = len(first["text"]) - len(first["text"].lstrip("\xa0 "))
    row = {"label": clean(first["text"]) if labelled else "",
           "header": [{"text": clean(c["text"]), "col": c["col"], "span": c["span"]} for c in texts],
           "indent_pt": round(left_pt(first["style"]) + sum(left_pt(s) for s in first["inner"][:1]) + lead * NBSP_PT,
                              2) if labelled else None,
           "label_facts": list(first["facts"]) if labelled else [],
           "rule_above": None, "rule_below": None, "cells": []}
    for c in cells:
        for side, key in (("top", "rule_above"), ("bottom", "rule_below")):
            b = border(c["style"], side)
            if b and (row[key] is None or b == "double"):
                row[key] = b
    pending = ""
    for c in (texts[1:] if labelled else texts):
        t = clean(c["text"])
        if t in ALONE and not c["facts"]:
            if t.startswith(")") and row["cells"]:
                row["cells"][-1]["text"] += t
            elif t != "$":
                pending += t
            continue
        row["cells"].append({"text": pending + t, "facts": list(c["facts"]), "col": c["col"], "span": c["span"]})
        pending = ""
    row["facts"] = [x for c in cells for x in c["facts"]]
    return row


@functools.lru_cache(maxsize=4)
def parsed(doc):
    """(tables, text before each), parsed once per document (a main document runs to a few megabytes)."""
    p = _Tables()
    p.feed(Path(doc).read_text(encoding="utf8", errors="replace"))
    return p.tables, p.before


def facts_of(table):
    return [x for r in table for c in r for x in c["facts"]]


def statement(doc, target_ids, statement_keys, why=None):
    """The printed statement holding the line: the table holding one of `target_ids` (the line's fact and its
    duplicates: same concept, context, unit and value) with the most facts of `statement_keys` (fact ID -> (concept,
    context, unit) for every fact of a concept the statement's presentation tree prints); a tie is ambiguous. Adjacent
    tables join (a statement split across pages) only with structural evidence besides shared facts (joins);
    otherwise they are listed in "not_joined" with the reason. {"rows", "tables", "before", "not_joined"} or
    {"failed": code, "why": text}."""
    if doc is None:
        return {"failed": "NO_MAIN_DOCUMENT", "why": why or "no main document in the filing folder"}
    tables, before = parsed(str(doc))
    hits = [(sum(x in statement_keys for x in facts_of(t)), i) for i, t in enumerate(tables)
            if any(x in target_ids for x in facts_of(t))]
    if not hits:
        return {"failed": "FACT_NOT_IN_TABLE", "why": "the line's fact is in no table of the main document"}
    n, at = max(hits, key=lambda h: (h[0], -h[1]))
    if n < MIN_FACTS:
        return {"failed": "NOT_THE_STATEMENT",
                "why": f"the table holding the line's fact has {n} of the statement's facts"}
    tied = [i for k, i in hits if k == n and i != at]
    if tied:
        return {"failed": "AMBIGUOUS_TABLE",
                "why": f"tables {[at] + tied} hold the line's fact and {n} of the statement's facts each"}
    keys = {statement_keys[x][:2] for x in facts_of(tables[at]) if x in statement_keys}
    lo = hi = at
    not_joined = []
    for step in (-1, 1):
        i = at + step
        while 0 <= i < len(tables):
            ok, reason = joins(tables[i], before[max(i, i - step)], keys, statement_keys)
            if not ok:
                if reason:
                    not_joined.append({"table": i, "why": reason})
                break
            keys |= {statement_keys[x][:2] for x in facts_of(tables[i]) if x in statement_keys}
            lo, hi = min(lo, i), max(hi, i)
            i += step
    rows = []
    for i in range(lo, hi + 1):
        rows += [dict(r, table=i) for r in (printed_row(c) for c in tables[i]) if r]
    return {"rows": rows, "tables": list(range(lo, hi + 1)), "before": before[lo], "not_joined": not_joined}


def joins(table, between, keys, statement_keys):
    """(True, None) if an adjacent table continues the statement: at least MIN_FACTS of the statement's facts making
    MIN_SHARE of its own, at most BETWEEN_CHARS of text printed between, and none of its statement facts repeating a
    concept and date already printed (a note or schedule repeats them; a continued page does not). (False, reason)
    when it holds the statement's facts but lacks the rest of the evidence; (False, None) when it holds too few."""
    xs = facts_of(table)
    mine = [x for x in xs if x in statement_keys]
    if len(mine) < MIN_FACTS or len(mine) < MIN_SHARE * len(xs):
        return False, None
    if len(between) > BETWEEN_CHARS:
        return False, f"{len(between)} characters of text are printed between it and the statement"
    again = sorted({statement_keys[x][0] for x in mine if statement_keys[x][:2] in keys})
    if again:
        return False, f"it repeats amounts the statement already printed ({', '.join(again[:3])})"
    return True, None


def amount_slots(rows):
    """Per table, its amount columns: the physical columns the amount cells of its rows with facts occupy, overlapping
    ranges merged, left to right. A row is shown slot by slot, so an empty amount column keeps its place and an amount
    is never shown under another column's heading."""
    spans = {}
    for r in rows:
        if r["facts"]:
            spans.setdefault(r["table"], []).extend((c["col"], c["col"] + c["span"] - 1) for c in r["cells"])
    out = {}
    for t, xs in spans.items():
        merged = []
        for lo, hi in sorted(xs):
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        out[t] = [tuple(m) for m in merged]
    return out


def indent_levels(rows):
    """Each labelled row's left offset as a level (0 for the smallest, by rank of the distinct offsets to the nearest
    half point), None for a row with no label; all None when the reader found no difference in offsets."""
    offsets = sorted({round(r["indent_pt"] * 2) / 2 for r in rows if r["label"]})
    if len(offsets) < 2:
        return [None] * len(rows)
    return [offsets.index(round(r["indent_pt"] * 2) / 2) if r["label"] else None for r in rows]


def heading(r):
    """A printed row with a label and no amount."""
    return bool(r["label"]) and not r["cells"]


def ruled(rows, i):
    """A rule printed above row i: on its own cells, or below the row before it."""
    return bool(rows[i]["rule_above"] or (i > 0 and rows[i - 1]["rule_below"]))


def block(rows, at):
    """(first, last) row index of the line's block, bounded by headings and totals (a ruled row with amounts): from
    the nearest heading above the line, or the row after the last total above it if that is nearer, to the first
    total after it, or the row before the next heading if that is nearer, else to the end."""
    total = lambda i: ruled(rows, i) and not heading(rows[i])     # noqa: E731
    first = max(next((i for i in range(at, -1, -1) if heading(rows[i])), 0),
                next((i + 1 for i in range(at - 1, -1, -1) if total(i)), 0))
    last = min(next((i for i in range(at + 1, len(rows)) if total(i)), len(rows) - 1),
               next((i - 1 for i in range(at + 1, len(rows)) if heading(rows[i])), len(rows) - 1))
    return first, last
