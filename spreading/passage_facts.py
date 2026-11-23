"""Fact IDs beside the tagged amounts a note passage prints (part of the root-cause fixes, behind
--root-cause-fixes). Document delivery only: code binds each tagged amount span of a passage to the fact the filing
tagged on that span, by the span's own element in the main document, never by comparing values (no reconciliation).
Nothing here enters a calculation, and nothing here decides whether a fact may be moved: loop.grow_pools and the
validator decide that, as for every other tool result.

How a span is bound. The filing's note text blocks (search.corpus) come from the extracted instance, which carries no
fact IDs. The main document carries each block as an ix:nonNumeric element (with its ix:continuation parts) and each
tagged amount inside it as an ix:nonFraction element with the fact's ID. The block's HTML is read again from the main
document with a private marker put right after each tagged amount, and cut into passages by the same code as the
corpus (search.blocks_of, search.chunk_block). A passage is covered only when its marked text, markers removed, is
byte-identical to the passage's text; otherwise it stays as it is and its spans are declared "not covered".

Declared outcomes per span (never silent): "bound"; "ambiguous, no ID shown"; "not covered, no ID shown";
"withheld: entry over the limit"; "withheld: passage over the limit". A result ends with one line counting the spans
left without an ID, by reason.

    ann = passage_facts.Annotator(f)
    r = ann.render(passage)         # passage: one dict of search.corpus(f)
    # {"text": the passage with " [f-12]" after each bound amount, "printed": the passage's own text,
    #  "entries": [{"fact", "text"}], "spans": [{"fact", "outcome", "why", ...}], "counts": {outcome: n},
    #  "count_line": text, "added": characters added}
"""
import functools
import html
import re
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
from pathlib import Path

from . import page, search, tagless

OPEN, CLOSE, DIGIT0 = "", "", 0xE100       # private-use characters: no whitespace, no word character
MARK = re.compile("([-]+)")
SHOWN_ID = re.compile(r" ?\[f-\d+\]")                   # an ID as this module shows it inline
# limits on what is added to one read_passage result. ID_CHARS caps one shown ID.
# ENTRY_CHARS caps one listed entry and PASSAGE_CHARS the characters added to one passage, sized from entry lengths in
# an offline render with no model call
# (4,000 is the cap a printed_page result already has, tools.PAGE_CHARS)
ID_CHARS = 12
ENTRY_CHARS = 300
PASSAGE_CHARS = 4000
PLAN_ENTRY_CHARS, PLAN_PASSAGE_CHARS = 160, 1500      # earlier smaller limits, not used
BOUND = "bound"
AMBIGUOUS = "ambiguous, no ID shown"
NOT_COVERED = "not covered, no ID shown"
OVER_ENTRY = "withheld: entry over the limit"
OVER_PASSAGE = "withheld: passage over the limit"
OUTCOMES = (BOUND, AMBIGUOUS, NOT_COVERED, OVER_ENTRY, OVER_PASSAGE)
HEADER = ("added by code from the filing's data, not printed: each [f-...] in the passage above is the fact ID the "
          "filing tagged on the amount just before it. One entry per ID follows: where the amount sits; its date or "
          "period; unit; the scale it is printed in; the sign of the filed value; its members, by the company's "
          "labels.")
SCALES = {0: "units", 3: "thousands", 6: "millions", 9: "billions", 12: "trillions", -2: "hundredths"}
# the trailer loop.item_line may add to an entry: "  [facts: f-12345 (inside the line: quote may be empty)]"
TRAILER_CHARS = len("  [facts:  (inside the line: quote may be empty)]")
# what loop.Memory.full and tools.with_fact_ids put around the added items: the bullet and line break of each item, and
# the header's "note: ". The count line is counted at its longest form, so the limit holds whatever it says.
BULLET_CHARS = len("  - ") + 1
NOTE_CHARS = len("note: ")


def marker(k):
    return OPEN + "".join(chr(DIGIT0 + int(d)) for d in str(k)) + CLOSE


def marks_in(text):
    return [int("".join(str(ord(c) - DIGIT0) for c in m.group(1))) for m in MARK.finditer(text)]


def unmarked(text):
    return MARK.sub("", text)


def squash(s):
    return " ".join(s.replace("\xa0", " ").split())


class _Document(HTMLParser):
    """The main document read once: for each ix:nonNumeric element with an ID, its content as HTML (continuations
    followed, ix:exclude dropped) with a marker after each ix:nonFraction element; and every ix:nonFraction span
    with its printed text, its ix attributes and its place (a top-level table's row and column, or a sentence)."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks = {}        # ix:nonNumeric ID -> list of HTML pieces
        self.waiting = {}       # continuation ID -> the block IDs it continues
        self.stack = []         # open ix elements: (tag, block IDs it feeds)
        self.spans = []         # every ix:nonFraction, by marker number
        self.open_spans = []    # marker numbers of the ix:nonFraction elements now open
        self.excluded = 0
        self.ids = {}           # element ID -> times seen on an ix:nonFraction
        self.depth = 0
        self.tables = []        # top-level tables: rows of cells {text, col, span}
        self.rows = self.row = self.cell = None
        self.spans_left, self.busy, self.col = {}, set(), 0

    # -- what the open blocks receive
    def feed_blocks(self, text):
        if self.excluded:
            return
        for _, ids in self.stack:
            for i in ids:
                self.blocks[i].append(text)

    def handle_startendtag(self, tag, attrs):
        if tag.startswith("ix:"):
            return              # an empty ix element prints nothing
        self.feed_blocks(self.get_starttag_text())
        self.table_start(tag, dict(attrs))
        self.table_end(tag)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "ix:exclude":
            self.excluded += 1
            return
        if tag == "ix:nonnumeric":
            self.feed_blocks(self.get_starttag_text())
            ids = []
            if a.get("id"):
                self.blocks.setdefault(a["id"], [])
                ids = [a["id"]]
                if a.get("continuedat"):
                    self.waiting.setdefault(a["continuedat"], []).append(a["id"])
            self.stack.append((tag, ids))
            return
        if tag == "ix:continuation":
            self.feed_blocks(self.get_starttag_text())
            ids = self.waiting.pop(a.get("id"), [])
            if a.get("continuedat"):
                self.waiting.setdefault(a["continuedat"], []).extend(ids)
            self.stack.append((tag, ids))
            return
        self.feed_blocks(self.get_starttag_text())
        if tag == "ix:nonfraction":
            k = len(self.spans)
            fid = a.get("id")
            if fid:
                self.ids[fid] = self.ids.get(fid, 0) + 1
            span = {"k": k, "fact": fid, "printed": "", "scale": a.get("scale"), "ix_sign": a.get("sign"),
                    "nested": bool(self.open_spans), "table": None, "row": None, "col": None, "colspan": None}
            for j in self.open_spans:
                self.spans[j]["nested"] = True
            if self.cell is not None and self.depth == 1:
                span.update(table=len(self.tables), row=len(self.rows), col=self.cell["col"],
                            colspan=self.cell["span"])
                self.cell["spans"].append(k)
            self.spans.append(span)
            self.open_spans.append(k)
            return
        self.table_start(tag, a)

    def handle_endtag(self, tag):
        if tag == "ix:exclude":
            self.excluded = max(0, self.excluded - 1)
            return
        if tag in ("ix:nonnumeric", "ix:continuation"):
            for i in range(len(self.stack) - 1, -1, -1):
                if self.stack[i][0] == tag:
                    del self.stack[i]
                    break
            self.feed_blocks(f"</{tag}>")
            return
        self.feed_blocks(f"</{tag}>")
        if tag == "ix:nonfraction" and self.open_spans:
            self.feed_blocks(marker(self.open_spans.pop()))
            return
        self.table_end(tag)

    def handle_data(self, data):
        self.feed_blocks(html.escape(data, quote=False))
        if not self.excluded:
            for k in self.open_spans:
                self.spans[k]["printed"] += data
        if self.cell is not None:
            self.cell["text"] += data

    # -- the place of a span: top-level tables with physical columns (as page._Tables)
    def table_start(self, tag, a):
        if tag == "table":
            self.depth += 1
            if self.depth == 1:
                self.rows, self.spans_left = [], {}
        elif tag == "tr" and self.depth == 1:
            self.row, self.col = [], 0
            self.busy = {c for c, n in self.spans_left.items() if n > 0}
            self.spans_left = {c: n - 1 for c, n in self.spans_left.items() if n > 1}
        elif tag in ("td", "th") and self.depth == 1 and self.row is not None:
            while self.col in self.busy:
                self.col += 1
            span, down = (max(1, int(x)) if (x or "").strip().isdigit() else 1
                          for x in (a.get("colspan"), a.get("rowspan")))
            self.cell = {"text": "", "col": self.col, "span": span, "spans": []}
            if down > 1:
                for c in range(self.col, self.col + span):
                    self.spans_left[c] = down - 1
            self.col += span
            self.row.append(self.cell)

    def table_end(self, tag):
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


def read_document(text):
    p = _Document()
    p.feed(text)
    p.close()
    return p


@functools.lru_cache(maxsize=4)
def _parsed(doc, sig):
    return read_document(Path(doc).read_text(encoding="utf8", errors="replace"))


def parsed(doc):
    q = Path(doc)
    return _parsed(str(q), (q.stat().st_size, q.stat().st_mtime_ns))


def place(doc, span):
    """Where a span sits: {"row", "column"} for a table cell (the row's first cell with a letter; the cells of the
    rows above the table's first tagged amount that overlap the span's columns), or {"sentence": True}."""
    if span["table"] is None or span["table"] >= len(doc.tables):
        return {"sentence": True}
    rows = doc.tables[span["table"]]
    row = rows[span["row"]] if span["row"] < len(rows) else []
    label = next((page.clean(c["text"]) for c in row if re.search(r"[A-Za-z]", page.clean(c["text"]))
                  and not c["spans"]), "")
    first = next((i for i, r in enumerate(rows) if any(c["spans"] for c in r)), 0)
    lo, hi = span["col"], span["col"] + span["colspan"] - 1
    heads = []
    for r in rows[:first]:
        for c in r:
            t = page.clean(c["text"])
            if t and c["col"] <= hi and c["col"] + c["span"] - 1 >= lo and t not in heads:
                heads.append(t)
    return {"row": label, "column": " / ".join(heads)}


def scale_words(scale):
    try:
        n = int(scale or 0)
    except ValueError:
        return f"scale {scale}"
    return SCALES.get(n, f"10^{n}")


def sign_words(value):
    try:
        v = Decimal(value)
    except (InvalidOperation, TypeError):
        return "sign not read"
    return "negative" if v < 0 else "zero" if v == 0 else "positive"


def unit_words(f, unit):
    cur = f.currency(unit)
    if cur:
        return cur
    ms = getattr(f, "unit_measures", {}).get(unit, ())
    return " per ".join(m.split("}")[-1] for m in ms) if ms else "no unit"


def member_words(member):
    """A member by the company's label for it (search.dims_label's naming), without the label role's closing
    " [Member]": the entry says "members:"."""
    m = tagless.member(member, member.split(":")[-1])
    return m[:-len(" [Member]")] if m.endswith(" [Member]") else m


def entry(f, x, where, span):
    """One metadata entry: where the amount sits, then period, unit, scale, sign and members. No concept name."""
    ctx = f.contexts.get(x["context"], {})
    when = f"at {ctx.get('instant')}" if ctx.get("instant") else f"{ctx.get('start')} to {ctx.get('end')}"
    dims = dict(ctx.get("dims") or {})
    if where.get("sentence"):
        at = "in a sentence"
    else:
        at = f"row \"{where['row'] or '(no label)'}\", column \"{where['column'] or '(no heading)'}\""
    members = "members: " + "; ".join(member_words(m) for _, m in sorted(dims.items())) if dims else "no members"
    return (f"{x['id']}: {at}; {when}; {unit_words(f, x.get('unit'))}; {scale_words(span['scale'])}; "
            f"{sign_words(x['value'])}; {members}")


class Annotator:
    """read_passage's fact IDs for one filing. `document`: the main document's text (tests); otherwise the filing's
    verified main document is read. Without a document every span is "not covered"."""

    def __init__(self, f, document=None, entry_chars=None, passage_chars=None):
        self.f = f
        self.entry_chars = ENTRY_CHARS if entry_chars is None else entry_chars
        self.passage_chars = PASSAGE_CHARS if passage_chars is None else passage_chars
        self.why = None
        if document is not None:
            self.doc = read_document(document)
        else:
            path, why = page.main_document(f.dir)
            self.doc, self.why = (parsed(path), None) if path else (None, why or "no main document")
        self.by_id = {}
        for x in f.facts:
            if x.get("id"):
                self.by_id.setdefault(x["id"], []).append(x)
        self._chunks = {}

    def chunks(self, p):
        """The marked passages of the text block `p` came from, cut as the corpus cuts them; None when the main
        document does not hold the block."""
        fid = p["fact"]
        if fid not in self._chunks:
            out = None
            x = next((x for x in self.by_id.get(fid, []) if x["concept"] == p["concept"]), None)
            if self.doc is not None and x is not None and fid in self.doc.blocks:
                plain = search.blocks_of(x["value"])
                title = (plain[0][1] if plain and plain[0][0] == "para" and search.is_heading(plain[0][1])
                         else self.f.label(x["concept"]))
                dims = dict(self.f.contexts.get(x["context"], {}).get("dims") or {})
                tag = f" [{search.dims_label(dims)}]" if dims else ""
                out = search.chunk_block(search.blocks_of("".join(self.doc.blocks[fid])), title, tag)
            self._chunks[fid] = out
        return self._chunks[fid]

    def spans_of(self, p):
        """(marked text or None, [span records]) for one passage, each span with its outcome; nothing is cut here."""
        chunks = self.chunks(p)
        n = p["part"]
        if chunks is None or n >= len(chunks) or unmarked(chunks[n]["text"]) != p["text"]:
            ks = marks_in(chunks[n]["text"]) if chunks is not None and n < len(chunks) else []
            why = (self.why or "the main document does not hold this passage's text block") if chunks is None \
                else "the main document's text of this passage differs from the passage"
            return None, [{"k": k, "fact": self.doc.spans[k]["fact"], "outcome": NOT_COVERED, "why": why}
                          for k in ks] or [{"k": None, "fact": None, "outcome": NOT_COVERED, "why": why,
                                            "unknown_count": True}]
        earlier = {k for c in chunks[:n] for k in marks_in(c["text"])}
        marked, out, here = chunks[n]["text"], [], set()
        for m in MARK.finditer(marked):
            k = marks_in(m.group())[0]
            s = self.doc.spans[k]
            rec = {"k": k, "fact": s["fact"], "printed": squash(s["printed"])}
            xs = self.by_id.get(s["fact"], [])
            before = squash(unmarked(marked[:m.start()]))
            if k in earlier or k in here:
                rec.update(outcome=NOT_COVERED, why="a row or caption repeated from an earlier passage of the same "
                                                    "table; its ID is shown where it is first printed")
            elif s["nested"]:
                rec.update(outcome=AMBIGUOUS, why="the span carries more than one tagged amount")
            elif not s["fact"] or self.doc.ids.get(s["fact"], 0) != 1:
                rec.update(outcome=AMBIGUOUS, why="the span's element has no ID of its own in the main document")
            elif len(xs) != 1:
                rec.update(outcome=AMBIGUOUS, why=f"{len(xs)} facts of the filing's data carry this ID")
            elif not rec["printed"] or not before.endswith(rec["printed"]):
                rec.update(outcome=AMBIGUOUS, why="the amount's printed text does not sit right before its place in "
                                                  "the passage")
            else:
                where = place(self.doc, s)
                rec.update(outcome=BOUND, why=None, where=where, entry=entry(self.f, xs[0], where, s),
                           binding={"fact": s["fact"], "printed": rec["printed"], **where,
                                    "context": xs[0]["context"], "unit": xs[0].get("unit"), "scale": s["scale"],
                                    "ix_sign": s["ix_sign"], "value": xs[0]["value"],
                                    "dims": dict(self.f.contexts.get(xs[0]["context"], {}).get("dims") or {})})
            here.add(k)
            out.append(rec)
        return marked, out

    def render(self, p):
        marked, spans = self.spans_of(p)
        entries, shown, full = [], {}, False
        used = NOTE_CHARS + len(HEADER) + BULLET_CHARS + COUNT_LINE_CHARS + BULLET_CHARS
        for s in spans:
            if s["outcome"] != BOUND:
                continue
            if full:        # reading order: once one span does not fit, no later span is shown
                s.update(outcome=OVER_PASSAGE, why=f"the passage's added text would pass {self.passage_chars} "
                                                   "characters")
                continue
            if len(s["entry"]) > self.entry_chars:
                s.update(outcome=OVER_ENTRY, why=f"its entry is {len(s['entry'])} characters, over "
                                                 f"{self.entry_chars}")
                continue
            inline = f" [{s['fact']}]"
            cost = len(inline) + len(s["entry"]) + BULLET_CHARS + TRAILER_CHARS + len(s["fact"])
            if len(inline) > ID_CHARS or used + cost > self.passage_chars:
                s.update(outcome=OVER_PASSAGE, why=f"the passage's added text would pass {self.passage_chars} "
                                                   "characters")
                full = True
                continue
            used += cost
            shown[s["k"]] = inline
            entries.append({"fact": s["fact"], "text": s["entry"]})
        text = p["text"] if marked is None else MARK.sub(lambda m: shown.get(marks_in(m.group())[0], ""), marked)
        counts = {o: sum(1 for s in spans if s["outcome"] == o and not s.get("unknown_count")) for o in OUTCOMES}
        unknown = any(s.get("unknown_count") for s in spans)
        return {"text": text, "printed": p["text"], "entries": entries, "spans": spans, "counts": counts,
                "count_line": count_line(counts, unknown, spans),
                "added": used if entries else COUNT_LINE_CHARS + BULLET_CHARS}


def count_line(counts, unknown, spans):
    """The result's last line: the tagged amount spans left without an ID, by reason."""
    none = {o: n for o, n in counts.items() if o != BOUND and n}
    if unknown:
        why = next(s["why"] for s in spans if s.get("unknown_count"))
        return (f"end: fact IDs could not be read for this passage ({why}); its tagged amounts, if any, are "
                f"{NOT_COVERED} and are not counted.")
    if not counts[BOUND] and not none:
        return "end: this passage prints no tagged amount; no fact ID is shown."
    if not none:
        return f"end: all {counts[BOUND]} tagged amounts of this passage carry their fact ID."
    return (f"end: {counts[BOUND]} tagged amounts of this passage carry their fact ID; "
            f"{sum(none.values())} carry none: " + "; ".join(f"{n} {o}" for o, n in none.items()) + ".")


# the longest count line: every reason present, each count at five digits (the unknown form is shorter than this)
COUNT_LINE_CHARS = len(count_line({o: 99999 for o in OUTCOMES}, False, [])) + 60


def without_ids(quote):
    """A quote with the IDs this module shows inline taken out (the IDs are added by code, not printed)."""
    return SHOWN_ID.sub("", quote or "")
