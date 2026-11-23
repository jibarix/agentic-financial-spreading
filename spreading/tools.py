"""The analyst's tools: local, pure Python over one filing. Each returns short evidence lines, each line with the fact
IDs it shows, so the model can cite them and allocate by fact ID. The pack checks of the v3 path live here: only the
line's currency is shown, an inconsistent duplicate is shown as a conflict with no amount (a consistent one, one
amount filed at two precisions, at its most precise fact), and an amount found only in
another currency is named, not shown.

A tool never decides anything about the line: it returns what the filing says. A list ends with an `end:` item stating
that every item this lookup found is listed, and what the lookup did not search (a model cannot know a list is
complete unless the tool says so; the marker is scoped to the lookup, since no single lookup covers calculations,
locations, axes and the notes, and company calculation arcs are raw, not verified effective relationships).
`search_notes` and `read_passage` run over a search.Search of the filing's notes; an arm built without one runs
without them and the schema does not offer them. Their items carry the passage ID they show (`passage`), and the
observation lists the passages it returned (`passages`), which become citable.

    tools = Tools(f, line, rows_by_section, section, search=search.Search.of(f))
    obs = tools.run("calc_children", {"concept": "us-gaap:LiabilitiesCurrent"})
    # {"tool", "args", "items": [{"text", "facts"}], "rows": {row_id: text} (rows_of only), "passages": [],
    #  "error": None}; search_notes adds "retrieval", the search's log
"""
from decimal import Decimal

from . import page, passage_facts, tagless, template
from .answer_keys import precise_duplicate
from .filing import in_period, period_length, period_words
from .packs import rows_block

BUILT = ["calc_children", "breakdowns", "other_periods", "rows_of", "recall"]
SEARCH_TOOLS = ["search_notes", "read_passage"]
PAGE_TOOLS = ["printed_page"]     # an arm built with the taxonomy (no definition tool: tags are hidden, tagless.py)
CALC_SCOPE = ("These are raw company arcs, not verified effective relationships; locations, axis breakdowns and "
              "narrative disclosures were not searched.")
BREAKDOWN_SCOPE = "Calculation relationships and narrative disclosures were not searched."
# the statements-only option (Tools statements_only): the tools that
# read facts from anywhere in the filing, the notes included, are not offered; calc_children reads the calculation
# arcs of the template's statements only (Filing.role of each), and note search is refused
NOT_STATEMENTS = ("breakdowns", "other_periods")
STATEMENTS_CALC_SCOPE = ("These are raw company arcs of the financial statements only, not verified effective "
                         "relationships; the notes and the rest of the filing were not read.")
# the combined-fixes option (--combined-fixes): added to breakdowns' "none" only, with --combined-fixes
FACT_HINT = ("If calculation children have not been checked, calc_children lists what the company's calculations "
             "add into this line, if anything, with fact IDs; only a fact ID a tool result lists can be moved as a "
             "piece.")
# member annotations: held at 222 characters, so a page near PAGE_CHARS is cut the same way
PAGE_SCOPE = ("end: among main-document tables holding a matching fact, this table held the most facts for this "
              "statement's concepts. It may be a note. Adjacent tables can continue it. Indentation and rules are the "
              "company's typesetting.")
# member annotations: a table with no row holding the line's own fact, but rows holding an equal amount tagged
# at members (member_matches). The marks and this item are added by code; the item and the member labels are
# annotations, never quotable (loop.Memory.quotable_texts).
MEMBER_MARK, MEMBER_MARKS = ">~", "?>"
MEMBER_NOTE = ("This table has no row holding the line's own fact {fact}. A row marked >~ (or ?>, when several rows "
               "qualify) holds an equal amount tagged at members: a distinct fact that may be a component of the "
               "line or a different item, not the line itself. The line under each marked row names the members. "
               "Judge from the labels whether it is the line; if the table does not establish it, say so.")
PAGE_LEGEND = ("legend (added by code, not printed): '>>' marks the line under review; [ids] are the fact IDs tagged "
               "on a printed amount; 'rule above' or 'rule below' is a line the company printed across the row; "
               "'(blank)' is an amount column the row leaves empty.")
BLANK = "(blank)"
NO_INDENT = ("indentation not detected by this reader (it reads padding, margin-left, text-indent and leading spaces "
             "in the cell's own style); all rows are shown flush left.")
TREE_SCOPE = ("end: the main document's printed table was not available, so this is the layout of the company's "
              "presentation tree, not the printed page: headings and depth, no typesetting. Notes and other "
              "statements were not searched.")
# alternate statement lookup (--root-cause-fixes only): printed_page's `statement` argument. A kind is resolved
# from the titles of the filing's own statement reports (FilingSummary), the way Filing.role finds the template's
# statements: those by the template's own names, title words and exclusions (face_kinds reads them from the template;
# no template literal is kept here), the other two by the words below. No company name, amount, line or expected
# answer is used.
THIS_STATEMENT = "this statement"
# `not_titled_as`: the template statement (by its name) whose title words also exclude a title from this kind
OTHER_FACES = {"equity statement": {"title_words": ["EQUITY", "SHAREHOLDERS", "STOCKHOLDERS", "DEFICIT",
                                                    "PARTNERS' CAPITAL", "MEMBERS' CAPITAL"],
                                    "title_excludes": ["PARENTHETICAL"], "not_titled_as": "balance sheet"},
               "comprehensive income statement": {"title_words": ["COMPREHENSIVE"],
                                                  "title_excludes": ["PARENTHETICAL"]}}


def face_kinds():
    """The alternate statement kinds: each template statement under its template name (the value is its statement ID), then
    OTHER_FACES with the title words of `not_titled_as` added to its exclusions."""
    sts = template.active().statements
    kinds = {s["name"]: k for k, s in sts.items()}
    for name, o in OTHER_FACES.items():
        more = [w for s in sts.values() if s["name"] == o.get("not_titled_as") for w in s["title_words"]]
        kinds[name] = {"title_words": list(o["title_words"]), "title_excludes": list(o["title_excludes"]) + more}
    return kinds
FACE_LEGEND = ("legend (added by code, not printed): rows of another statement are shown without fact IDs: they are "
               "evidence to quote and add no amount that can be moved as a piece; 'rule above' or 'rule below' is a "
               "line the company printed across the row; '(blank)' is an amount column the row leaves empty.")
FACE_SCOPE = ("end: the printed face of that statement only, as text read from the filing's main document; notes "
              "were not searched. Indentation and rules are the company's typesetting, not accounting "
              "classifications.")
PAGE_ROWS = 120       # a longer view is cut, and says so
# the whole result, every item counted: about 1,000 tokens of the 6,144 a request may use; a provisional engineering
# limit, not a measured best size
PAGE_CHARS = 4000


def size(items):
    return sum(len(i["text"]) + 1 for i in items)


def fit(required, order, compose, rows=None):
    """The rows to show: `required` first, then rows in `order` while the whole result `compose` makes of them stays
    within PAGE_CHARS characters and `rows` rows (PAGE_ROWS by default; less when items under rows take slots);
    None if the required rows alone do not fit."""
    rows = PAGE_ROWS if rows is None else rows
    shown = set(required)
    if len(shown) > rows or size(compose(shown)) > PAGE_CHARS:
        return None
    for i in order:
        if i in shown:
            continue
        if len(shown) >= rows or size(compose(shown | {i})) > PAGE_CHARS:
            break
        shown.add(i)
    return shown


def resolve_statement(asked, reports, prints=None):
    """The alternate statement resolver. `asked`: a kind of face_kinds() or a statement's printed title; `reports`: [(role, title)] of the
    filing's statement reports, in its order; `prints(role)`: the concepts a report prints (the template's title
    fallback). (outcome, [(role, title)]): "found" with the one report; "ambiguous" with every candidate; "missing"
    with none. A printed title names its report. Nothing is guessed."""
    a = " ".join(str(asked).split()).lower()
    named = [(k, t) for k, t in reports if " ".join(str(t).split()).lower() == a]
    if named:
        return ("found" if len(named) == 1 else "ambiguous"), named
    kind = face_kinds().get(a)
    if kind is None:
        return "missing", []
    if isinstance(kind, str):
        st = template.active().statements.get(kind)
        hits = template.find_statement(st, reports, prints or (lambda r: set()))[0] if st else []
    else:
        hits = [k for k, t in reports if template.titled(kind["title_words"], kind["title_excludes"], t)]
    found = [(k, t) for k, t in reports if k in hits]
    return ("found" if len(found) == 1 else "ambiguous" if found else "missing"), found


class Tools:
    def __init__(self, f, line, rows_by_section, section, names=None, search=None, tax=None, statement=None,
                 v20=False, fact_hint=False, root_cause=False, statements_only=False):
        """`line`: the C2 line (concept, unit, fact, amount). `rows_by_section`: section -> its item row IDs (the
        template); `names`: row ID -> template name. `section` is the line's own; rows_of offers the others,
        with the definitions the template gives them. `search`: the filing's
        search.Search, or None for an arm without search. `tax`: the us-gaap edition the filing uses
        (taxonomy.Taxonomy), or None for an arm without printed_page. `statement`: the line's
        statement, required for a queued line (section None, with rows_by_section empty, so rows_of is not offered);
        otherwise the section's. `v20` (STATEMENT_PROMPTS): rows_of names each catch-all's section. `fact_hint`
        (--combined-fixes): breakdowns' "none" adds FACT_HINT. `root_cause`
        (--root-cause-fixes): read_passage shows fact IDs beside tagged amounts and
        printed_page takes another statement; off, every result is as before. `statements_only`:
        NOT_STATEMENTS are not offered and refused, calc_children reads only the
        financial statements' calculation arcs, and a search is refused; off, every result is as before."""
        if section is None and statement is None:
            raise ValueError("a line without a section needs its statement")
        if statements_only and search is not None:
            raise ValueError("statements only: the notes' search is not offered")
        self.statements_only = statements_only
        self.f, self.line, self.section, self.v20, self.fact_hint = f, line, section, v20, fact_hint
        self.root_cause, self._annotator = root_cause, None
        self.statement = statement or template.active().statement_of(section)
        self.key = line.get("period") or ("instant", getattr(f, "period_end", None))      # the run's period (C1)
        self.rows_by_section, self.names, self.search, self.tax = rows_by_section, names or {}, search, tax
        self._arcs = {}
        self.lines = {}     # concept -> line, when the tools serve a whole section (repair_run.tools_of)
        self.own =getattr(f, "pieces", {}).get(line.get("concept"))       # the line's split piece, if it is one

    def available(self):
        """The tools of this arm, in the order the schema offers them."""
        built = [t for t in BUILT if t not in NOT_STATEMENTS] if self.statements_only else BUILT
        return (built + (PAGE_TOOLS if self.tax is not None else [])
                + (SEARCH_TOOLS if self.search is not None else []))

    def arcs(self, kind):
        if kind not in self._arcs:
            self._arcs[kind] = getattr(self.f, "tag_arcs", self.f.arcs)(kind)   # as filed, never split by code
            if self.statements_only and kind == "cal":      # the template's statements' own arcs only, not the
                # comprehensive income, equity or parenthetical pages FilingSummary also lists as Statements
                roles = {self.f.role(k) for k in template.active().statements} - {None}
                self._arcs[kind] = [a for a in self._arcs[kind] if a["role"] in roles]
        return self._arcs[kind]

    def other_sections(self):
        return sorted(s for s in self.rows_by_section if s != self.section and self.rows_by_section[s])

    def run(self, action, args):
        """One tool call. A call that cannot run (unknown concept, unknown section) returns an observation with an
        `error` and no facts: a tool error is evidence for the model, not a failure of the line. On a split piece's
        line (corrections.apply) its tag, which the line pack shows, and its key both name the line's own amount: the
        observation records the key, as code's checks compare it with the line."""
        args = dict(args)
        shown = "concept" not in args or tagless.known(args["concept"])   # tags hidden: only an ID shown passes
        if "concept" in args:       # tags hidden: the model names a line by its ID; the observation records the tag
            args["concept"] = tagless.back(str(args["concept"]))
        if self.own and str(args.get("concept", "")).strip() == self.own["tag"]:
            args["concept"] = self.line["concept"]
        out = {"tool": action, "args": args, "items": [], "rows": {}, "passages": [], "error": None}
        try:
            if not shown:           # a tag typed in place of an ID: refused, without repeating it
                raise ToolError(tagless.UNKNOWN)
            if self.statements_only and action not in self.available():
                raise ToolError(f"{action} is not a tool of this arm")
            if action == "calc_children":
                out["items"] = self.calc_children(args["concept"].strip())
            elif action == "breakdowns":
                out["items"] = self.breakdowns(args["concept"].strip())
            elif action == "other_periods":
                out["items"] = self.other_periods(args["concept"].strip())
            elif action == "rows_of":
                out["items"], out["rows"] = self.rows_of(args["section"])
            elif action == "printed_page" and self.tax is not None:
                other = " ".join(str(args.get("statement") or "").split())
                if self.root_cause and other and other.lower() != THIS_STATEMENT:      # alternate statement
                    out["items"], out["resolver"] = self.other_face(other, str(args.get("heading") or ""))
                else:
                    out["items"] = self.page_at(args)
            elif action in SEARCH_TOOLS and self.search is not None:
                out.update(self.search_notes(args["query"]) if action == "search_notes"
                           else self.read_passage(args["passage_id"].strip()))
            else:
                raise ToolError(f"{action} is not a tool of this arm")
        except ToolError as e:
            out["error"] = str(e)
            out["items"] = [{"text": f"tool error: {e}", "facts": []}]
        return out

    # ---- the notes' text ----

    def search_notes(self, query):
        try:
            return self.search.search(query)
        except ValueError as e:
            raise ToolError(str(e))

    def read_passage(self, pid):
        try:
            out = self.search.read(pid)
        except KeyError as e:
            raise ToolError(e.args[0])
        return self.with_fact_ids(pid, out) if self.root_cause else out

    def with_fact_ids(self, pid, out):
        """Passage fact IDs (--root-cause-fixes): the passage with the fact ID the filing tagged on each amount it prints, then one
        entry per ID (added by code, headed so), then the result's own note, then one line counting the tagged amounts
        left without an ID, by reason. The first item keeps the passage as printed (`printed`): what a quote is
        matched against. Each entry lists its fact, so code classifies it as any other tool-returned fact
        (loop.grow_pools). `fact_ids` is the binding record; it is never shown to the model."""
        if self._annotator is None:
            self._annotator = passage_facts.Annotator(self.f)
        p = self.search.by_id[pid]
        r = self._annotator.render(p)
        first, rest = out["items"][0], out["items"][1:]
        items = [{**first, "text": f"{pid} [{p['heading']}] {r['text']}", "printed": first["text"]}]
        if r["entries"]:
            items.append({"text": "note: " + passage_facts.HEADER, "facts": []})
            items += [{"text": e["text"], "facts": [e["fact"]]} for e in r["entries"]]
        items += rest + [{"text": r["count_line"], "facts": []}]
        spans = [{k: v for k, v in s.items() if k in ("fact", "printed", "outcome", "why", "binding")}
                 for s in r["spans"]]
        return {**out, "items": items,
                "fact_ids": {"spans": spans, "counts": r["counts"], "added": r["added"],
                             "limits": {"id": passage_facts.ID_CHARS, "entry": self._annotator.entry_chars,
                                        "passage": self._annotator.passage_chars}}}

    # ---- amounts, checked before they are shown ----

    def view(self, concept):
        """(tag, member dims) of the amount `concept` names: a split piece's key reads its tag at the piece's member
        (corrections.apply); any other concept, itself with no member. Lookups resolve through this, so a piece's
        amount at other dates and its tag's filed relationships are found."""
        p = getattr(self.f, "pieces", {}).get(concept)
        return (p["tag"], p["dims"]) if p else (concept, {})

    def facts_of(self, concept, own=None):
        """(fact, context, dims beyond the member) for the filing's own entity's non-nil facts of the tag `concept`
        names (view), as filed or split (a piece's facts carry its key), at the member `own` (default: the view's)."""
        tag, at = self.view(concept)
        own, f = at if own is None else own, self.f
        for c in [tag] + sorted(k for k, p in getattr(f, "pieces", {}).items() if p["tag"] == tag):
            for x in f.by_concept.get(c, []):
                ctx = f.contexts[x["context"]]
                if x["value"] != "" and ctx["entity"] == f.entity and all(ctx["dims"].get(a) == m
                                                                          for a, m in own.items()):
                    yield x, ctx, {a: m for a, m in ctx["dims"].items() if a not in own}

    def named(self, concept):
        tag, own = self.view(concept)
        return tagless.show(concept if own and tagless.ON else tag) + "".join(
            f" at {tagless.member(a, a.split(':')[-1])}={tagless.member(m, m.split(':')[-1])}"
            for a, m in sorted(own.items()))

    def known(self, concept):
        concept = self.view(concept)[0]
        if concept not in self.f.by_concept and not any(concept in (a["parent"], a["child"]) for a in self.arcs("cal")):
            raise ToolError(f"{tagless.show(concept)!r} is not a line of this filing; use an item ID as shown, e.g. "
                            "item_12")

    def amount(self, facts):
        """(text, fact IDs) for facts of one concept and context set, in the line's currency."""
        unit = self.line["unit"]
        same = [x for x in facts if self.f.currency(x["unit"]) == unit]
        if not same:
            other = sorted({self.f.currency(x["unit"]) or str(x["unit"]) for x in facts})
            none = "no amount at this date" if self.key[0] == "instant" else "no amount for this period"
            return (f"amount only in {', '.join(other)}, not shown" if other else none), []
        ids = sorted(str(x["id"]) for x in same if x.get("id"))
        if len({Decimal(x["value"]) for x in same}) > 1:
            best = precise_duplicate(same)
            if best is None:
                return f"conflict: {len(same)} facts with different amounts, none shown", ids
            # one amount filed at two precisions (consistent duplicates): the most precise is shown and offered
            v = Decimal(best["value"])
            rounded = sorted({x["value"] for x in same if Decimal(x["value"]) != v}, key=Decimal)
            keep = sorted(str(x["id"]) for x in same if x.get("id") and Decimal(x["value"]) == v)
            return (f"{best['value']} {unit} (also reported as {', '.join(rounded)} at lower precision; consistent "
                    "duplicate, not an additional amount)"), keep
        return f"{same[0]['value']} {unit}", ids

    def at_date(self, concept, dims=False, own=None):
        """The concept's facts of the filing's own entity for the run's period (C1); with `dims`, grouped by
        their dimensions beyond the member (facts_of), else those with none. A split piece's member facts are its
        undimensioned amount."""
        groups = {}
        for x, ctx, d in self.facts_of(concept, own):
            if in_period(ctx, self.key) and bool(d) == dims:
                groups.setdefault(tuple(sorted(d.items())), []).append(x)
        return groups

    # ---- the tools ----

    def calc_children(self, concept):
        """Concepts the company's calculations add into `concept`, balance-sheet network first, then the others,
        each child shown once with the roles that state it."""
        self.known(concept)
        tag, own = self.view(concept)       # the tag's filed arcs; a piece's children at its member
        by = {}
        for a in self.arcs("cal"):
            if a["parent"] == tag and (a.get("arcrole") or "").endswith("summation-item"):
                by.setdefault(a["child"], {"weights": set(), "roles": set()})
                by[a["child"]]["weights"].add(a["weight"])
                by[a["child"]]["roles"].add(a["role"].rsplit("/", 1)[-1])
        scope = STATEMENTS_CALC_SCOPE if self.statements_only else CALC_SCOPE
        if not by:
            return [{"text": f"none: no calculation arc in the company's own linkbases adds anything into "
                             f"{tagless.show(concept)}. "
                             f"{scope}", "facts": []}]
        items = []
        for c in sorted(by):
            amt, ids = self.amount(self.at_date(c, own=own).get((), []))
            w = "/".join(sorted(str(x) for x in by[c]["weights"]))
            items.append({"text": f"{tagless.show(c)} ({self.f.label(c)}) weight {w}: {amt} [in {', '.join(sorted(by[c]['roles']))}]",
                          "facts": ids})
        return items + [{"text": f"end: all {len(items)} children found in the company's own calculation arcs are "
                                 f"listed. {scope}", "facts": []}]

    def breakdowns(self, concept):
        """Amounts the filing places inside `concept`: extensible-list locations (a component the filing says is
        included in the line), then breakdowns by axis and member. A breakdown need not add to the line."""
        self.known(concept)
        f, items = self.f, []
        tag, own = self.view(concept)
        # an inclusion statement names the whole tag, never one member's amount: unavailable for a piece
        note = [{"text": f"[location] unavailable for {self.named(concept)}: the filing's inclusion statements name "
                         f"the whole {tagless.show(tag)}, not one member's amount", "facts": []}] if own else []
        for lc in sorted(c for c in f.by_concept if c.endswith("ExtensibleList") and not own):
            for x in f.by_concept[lc]:
                ctx = f.contexts[x["context"]]
                if not x["value"] or not in_period(ctx, self.key) or ctx["dims"] or ctx["entity"] != f.entity:
                    continue
                if concept in [f.qname(v) for v in x["value"].split()]:
                    comp = lc.replace("StatementOfFinancialPositionExtensibleList", "")
                    amt, ids = self.amount(self.at_date(comp).get((), []))
                    stated = [str(x["id"])] * bool(x.get("id"))
                    # which ID is which: the model cannot tell the inclusion statement from the amount otherwise
                    which = []
                    if ids and "conflict" in amt:
                        which.append(f"facts with different amounts {', '.join(ids)}")
                    elif ids:
                        which.append(f"amount fact {ids[0]}" if len(ids) == 1 else
                                     f"amount facts {', '.join(ids)}, one amount filed {len(ids)} times: use one")
                    if stated:
                        which.append(f"the inclusion statement itself {stated[0]}, not an amount")
                    items.append({"text": f"[location] the filing states {tagless.show(comp)} ({f.label(comp)}) is "
                                          f"included in {tagless.show(concept)}: {amt} ({'; '.join(which)})", "facts": stated + ids})
        rows = []
        for dims, xs in self.at_date(concept, dims=True).items():
            members = "; ".join(f"{tagless.member(d, d.split(':')[-1])}={tagless.member(m, m.split(':')[-1])}"
                                for d, m in dims)
            amt, ids = self.amount(xs)
            dup = f" (one amount filed {len(ids)} times: use one)" if len(ids) > 1 and "conflict" not in amt else ""
            rows.append({"text": f"[axis] {members}: {amt}{dup}", "facts": ids})
        items += sorted(rows, key=lambda r: r["text"])
        if not items:
            return note + [{"text": f"none: this {'axis' if own else 'location and axis'} lookup found no amount "
                                   f"inside {self.named(concept)} {period_words(self.key)}. {BREAKDOWN_SCOPE}"
                                   + (f" {FACT_HINT}" if self.fact_hint else ""),
                            "facts": []}]
        n = len(items)
        if rows:
            items.append({"text": "note: an axis breakdown is a disclosure; its members need not add to the line",
                          "facts": []})
        return note + items + [{"text": f"end: all {n} items found by this location and axis lookup are listed. "
                                 f"{BREAKDOWN_SCOPE}", "facts": []}]

    def other_periods(self, concept):
        """The concept, undimensioned, at every other date the filing shows (instants and period ends)."""
        self.known(concept)
        f, by = self.f, {}
        for x, ctx, d in self.facts_of(concept):      # a piece: its tag at its member, at every date
            if d or in_period(ctx, self.key):
                continue
            when = ctx["instant"] or f"{ctx['start']} to {ctx['end']}"
            if not ctx["instant"] and self.key[0] == "duration":      # IS and CF runs name each length (C1)
                when += f" ({period_length(ctx['start'], ctx['end'])})"
            by.setdefault(when, []).append(x)
        if not by:
            return [{"text": f"none: {self.named(concept)} appears at no other date in this filing", "facts": []}]
        out = []
        for when in sorted(by, reverse=True):
            amt, ids = self.amount(by[when])
            out.append({"text": f"{self.named(concept)} at {when}: {amt}", "facts": ids})
        return out

    def page_at(self, args):
        """printed_page around the tools' line, or, for tools serving a whole section (`lines`, the tree), around
        the line its `concept` names (otherwise the page comes back around the section's first line, with the line
        under review cut)."""
        if "concept" not in args:
            return self.printed_page(args.get("scope") or "block")
        line = self.lines.get(args["concept"].strip())
        if line is None:
            raise ToolError("not a line of this section")
        own, self.line = self.line, line
        try:
            return self.printed_page(args.get("scope") or "block")
        finally:
            self.line = own

    def printed_page(self, scope="block"):
        """The printed statement holding the line's fact, from the filing's verified main document, the line marked
        '>>'. `scope` 'block': the column headings, the line's block (its heading to its total) in full, and
        elsewhere only headings and ruled rows; 'statement': the whole statement. Either is subject to the caps: the
        whole result within PAGE_CHARS characters and PAGE_ROWS rows, the line's row, the column headings and the
        section headings kept first, then the rows nearest the line, the rest cut and said so. Without the printed
        table, or when the line and its headings alone exceed the cap: the presentation tree, with the reason (section
        headings kept)."""
        if scope not in ("block", "statement"):
            raise ToolError(f"scope {scope!r}: use 'block' or 'statement'")
        role = self.f.role(self.statement)
        me = next((x for x in self.f.facts if x["id"] == self.line["fact"]), None)
        if role is None:
            st = {"failed": "NO_ROLE", "why": "no presentation tree for this statement"}
        elif me is None:
            st = {"failed": "NO_FACT", "why": "the line's fact ID is not in the instance"}
        else:
            doc, why = page.main_document(self.f.dir)
            st = page.statement(doc, self.target_ids(me) | self.member_matches(me), self.statement_keys(role),
                                why=why)
        if "failed" in st:
            return self.printed_tree(role, f"{st['failed']}: {st['why']}")
        rows, targets = st["rows"], self.target_ids(me)
        exact = [i for i, r in enumerate(rows) if targets & set(r["facts"])]
        near = set() if exact else self.member_matches(me)
        marked = [] if exact else [i for i, r in enumerate(rows) if near & set(r["facts"])]
        at = (exact or marked)[0]
        mark = {i: MEMBER_MARK if len(marked) == 1 else MEMBER_MARKS for i in marked} or {at: True}
        labels = {i: [{"text": self.member_label(x, me), "facts": [], "annotation": True}
                      for x in dict.fromkeys(rows[i]["facts"]) if x in near] for i in marked}
        first_fact = next(i for i, r in enumerate(rows) if r["facts"])
        own_first = next(i for i, r in enumerate(rows) if r["facts"] and r["table"] == rows[at]["table"])
        levels = page.indent_levels(rows)
        if scope == "block":
            lo, hi = page.block(rows, at)
            keep = {i for i, r in enumerate(rows) if i < first_fact or lo <= i <= hi or page.heading(r)
                    or page.ruled(rows, i)}
        else:
            keep = set(range(len(rows)))
        keep |= set(marked)     # every marked row is required, as the line's row is
        heads = {i for i in keep if i < first_fact or (rows[i]["table"] == rows[at]["table"] and i < own_first)
                 or page.heading(rows[i])}     # section headings are kept under the cap
        slots = page.amount_slots(rows)
        texts = {i: self.page_row(rows[i], levels[i], mark.get(i, False), slots.get(rows[i]["table"])) for i in keep}
        head = [{"text": PAGE_LEGEND, "facts": []}]
        if marked:
            head.append({"text": MEMBER_NOTE.format(fact=me["id"]), "facts": [], "annotation": True})
        if all(lvl is None for lvl in levels):
            head.append({"text": NO_INDENT, "facts": []})
        if st["before"]:
            head.append({"text": f"printed above the table: {st['before']}", "facts": []})
        tail = [{"text": f"an adjacent table holding amounts of this statement was not joined: {x['why']}",
                 "facts": []} for x in st["not_joined"]]
        if scope == "block" and len(keep) < len(rows):
            tail.append({"text": "outside the line's block only headings and ruled rows are shown; "
                                 "printed_page(scope='statement') requests the whole statement, subject to the same "
                                 "caps", "facts": []})
        tail.append({"text": PAGE_SCOPE, "facts": []})

        def compose(shown):
            return head + self.window(len(rows), shown, len(keep), lambda i: {"text": texts[i],
                                                                                "facts": list(rows[i]["facts"])},
                                      after=labels) + tail
        required = {at} | heads | set(marked)
        too_long = ("TOO_LONG: the rows holding the line's amount at members exceed the cap" if marked else
                    f"TOO_LONG: the line's row and the column headings exceed the cap of {PAGE_CHARS} characters")
        # member annotations: the member labels and the note take row slots, reserved for the whole fit
        budget = PAGE_ROWS - (sum(map(len, labels.values())) + 1 if marked else 0)
        shown = fit(required, sorted(keep, key=lambda i: (abs(i - at), i)), compose, rows=budget)
        if shown is None:
            return self.printed_tree(role, too_long)
        return compose(shown)

    def statement_reports(self):
        return [(s["role"], s["name"]) for s in getattr(self.f, "statements", [])]

    def other_face(self, asked, heading=""):
        """Alternate statement lookup (--root-cause-fixes): (items, resolver outcome) for another statement's printed face. The first item
        starts "STATEMENT " and names the statement, the resolver's outcome and whether the line under review is on
        that face. "found": the face from its top, or from the row whose label holds `heading`, within the caps of the
        page; rows carry no fact IDs, so the face is evidence to quote and adds nothing to the pools. "ambiguous": the
        candidates by their printed titles, no face. "missing": said so, with the filing's statement titles."""
        reports = self.statement_reports()
        outcome, hits = resolve_statement(asked, reports, lambda r: {p["concept"] for p in self.f.printed(r)})
        titles = "; ".join(f"'{t}'" for _, t in reports) or "none"
        if outcome == "missing":
            return [{"text": f"STATEMENT {asked}: missing. No statement of this filing matches, and nothing was "
                             f"guessed. The filing's statements, by their printed titles: {titles}. A kind "
                             f"({', '.join(face_kinds())}) or one of these titles can be named.", "facts": []}], outcome
        if outcome == "ambiguous":
            return [{"text": f"STATEMENT {asked}: ambiguous. {len(hits)} statements of this filing match: "
                             + "; ".join(f"'{t}'" for _, t in hits) + ". No face is returned; name one of them by "
                             "its printed title.", "facts": []}], outcome
        role, title = hits[0]
        if role == self.f.role(self.statement):
            return ([{"text": f"STATEMENT {title}: found. It is the statement of the line under review; its page "
                              "follows as printed_page shows it.", "facts": []}]
                    + self.printed_page("statement")), outcome
        keys = self.statement_keys(role)
        doc, why = page.main_document(self.f.dir)
        st = page.statement(doc, set(keys), keys, why=why)
        if "failed" in st:
            return [{"text": f"STATEMENT {title}: found, but its printed face could not be read ({st['failed']}: "
                             f"{st['why']}). Nothing else is shown in its place.", "facts": []}], outcome
        rows = st["rows"]
        me = next((x for x in self.f.facts if x["id"] == self.line["fact"]), None)
        mine = self.target_ids(me) if me else set()
        on = [i for i, r in enumerate(rows) if mine & set(r["facts"])]
        first_fact = next((i for i, r in enumerate(rows) if r["facts"]), 0)
        levels, slots = page.indent_levels(rows), page.amount_slots(rows)
        texts = [self.page_row(r, levels[i], i in on, slots.get(r["table"]), ids=False) for i, r in enumerate(rows)]
        word = " ".join(heading.split()).lower()
        at = next((i for i, r in enumerate(rows) if word and word in " ".join(r["label"].split()).lower()), None)
        where = ("The line under review is not printed on it." if not on else
                 "The line under review's amount is printed on it, in the row marked '>>'.")
        part = ("" if not word else f" Shown around the row whose label holds '{heading.strip()}'." if at is not None
                else f" No row label holds '{heading.strip()}'; shown from the top.")
        head = [{"text": f"STATEMENT {title}: found, another statement of this filing. {where}{part}", "facts": []},
                {"text": FACE_LEGEND, "facts": []}]
        if st["before"]:
            head.append({"text": f"printed above the table: {st['before']}", "facts": []})
        tail = [{"text": f"an adjacent table holding amounts of this statement was not joined: {x['why']}",
                 "facts": []} for x in st["not_joined"]] + [{"text": FACE_SCOPE, "facts": []}]
        start = at if at is not None else 0

        def compose(shown):
            return head + self.window(len(rows), shown, len(rows), lambda i: {"text": texts[i], "facts": []}) + tail
        need = {i for i in range(len(rows)) if i < first_fact} | ({at} if at is not None else set())
        shown = fit(need, sorted(range(len(rows)), key=lambda i: (i < start, abs(i - start), i)), compose)
        if shown is None:
            return [{"text": f"STATEMENT {title}: found, but its column headings alone exceed the cap of "
                             f"{PAGE_CHARS} characters. Nothing else is shown in its place.", "facts": []}], outcome
        return compose(shown), outcome

    @staticmethod
    def window(n, shown, wanted, item, after=None):
        """Rows 0..n-1 that are `shown`, in order, with "(k rows not shown)" for each gap and a cut note when fewer
        than `wanted` rows are shown; `after`, row -> items shown directly under it (the member labels)."""
        out, gap = [], 0
        for i in range(n):
            if i not in shown:
                gap += 1
                continue
            if gap:
                out.append({"text": f"   ({gap} rows not shown)", "facts": []})
                gap = 0
            out.append(item(i))
            out += (after or {}).get(i, [])
        if gap:
            out.append({"text": f"   ({gap} rows not shown)", "facts": []})
        if wanted > len(shown):
            out.append({"text": f"cut: {wanted - len(shown)} rows of this view are not shown, to keep this result "
                                f"within {PAGE_ROWS} rows and {PAGE_CHARS} characters", "facts": []})
        return out

    @staticmethod
    def page_row(r, level, mine, slots=None, ids=True):
        """One printed row as text; a row with facts is shown slot by slot (page.amount_slots), BLANK where its
        amount column is empty. `ids` False (another statement's face): no fact ID is shown. `mine`: True for
        '>>', or a mark of its own (the member marks '>~' and '?>')."""
        label = "  " * (level or 0) + r["label"]
        if r.get("label_facts") and ids:
            label += f" [{', '.join(r['label_facts'])}]"

        def show(c):
            return c["text"] + (f" [{', '.join(c['facts'])}]" if c["facts"] and ids else "")
        if slots and r["facts"]:
            cells = " | ".join(" ".join(show(c) for c in r["cells"] if lo <= c["col"] <= hi) or BLANK
                               for lo, hi in slots)
        else:
            cells = " | ".join(show(c) for c in r["cells"])
        rules = "".join(f" ({'double ' if r[k] == 'double' else ''}rule {k[5:]})"
                        for k in ("rule_above", "rule_below") if r[k])
        lead = f"{mine} " if isinstance(mine, str) else ">> " if mine else "   "
        return f"{lead}{label}{' | ' if r['label'] and cells else ''}{cells}{rules}"

    def target_ids(self, me):
        """The line's fact and its duplicates: same concept, context, unit and value, any of which the page may
        print."""
        return {x["id"] for x in self.f.by_concept.get(me["concept"], [])
                if x["id"] and x["context"] == me["context"] and x.get("unit") == me.get("unit")
                and x["value"] == me["value"]}

    def member_matches(self, me):
        """Member annotations: the facts of the line's concept with the line's unit, value, period and entity at other dimensions,
        less target_ids. Distinct facts (XBRL OIM 1.0 6: duplicates only when every dimension is equal), used only to
        find the printed table and to label rows. A filing without `contexts` (a test fake): none."""
        ctxs = getattr(self.f, "contexts", None) or {}
        own = ctxs.get(me["context"])
        if own is None:
            return set()

        def key(c):
            return c.get("instant"), c.get("start"), c.get("end"), c.get("entity")
        same = self.target_ids(me)
        return {x["id"] for x in self.f.by_concept.get(me["concept"], [])
                if x["id"] and x["id"] not in same and x.get("unit") == me.get("unit") and x["value"] == me["value"]
                and x["context"] in ctxs and key(ctxs[x["context"]]) == key(own)
                and ctxs[x["context"]]["dims"] != own["dims"]}

    def member_label(self, fid, me):
        """Member annotations: the label under a marked row for one member match: its members, named as breakdowns names them."""
        x = next(y for y in self.f.by_concept[me["concept"]] if y["id"] == fid)
        dims = sorted(self.f.contexts[x["context"]]["dims"].items())
        members = "; ".join(f"{tagless.member(d, d.split(':')[-1])}={tagless.member(m, m.split(':')[-1])}"
                            for d, m in dims)
        return f"  \u21b3 {fid}: {members} (equal amount at members; not the line's fact {me['id']})"

    def statement_keys(self, role):
        """Fact ID -> (concept, context, unit) for every fact of a concept the statement's presentation tree prints."""
        concepts = {r["concept"] for r in self.f.printed(role)}
        return {x["id"]: (x["concept"], x["context"], x.get("unit")) for c in concepts
                for x in self.f.by_concept.get(c, []) if x["id"]}

    def printed_tree(self, role, reason):
        """The fallback: the company's presentation tree (order and depth, never printed layout), under the same caps
        as the page, the line's row kept first."""
        if role is None:
            raise ToolError(f"no printed table and no presentation tree for this statement ({reason})")
        tree = self.f.printed(role)
        items = []
        for r in tree:
            xs = self.at_date(r["concept"]).get((), [])
            amt, ids = self.amount(xs) if xs else ("", [])
            mark = ">> " if r["concept"] == self.line["concept"] else "   "
            items.append({"text": f"{mark}{'  ' * r['depth']}{r['label']}{' | ' + amt if amt else ''}", "facts": ids})
        head = [{"text": f"printed table not available ({reason}); the company's presentation tree follows",
                 "facts": []}]
        at = next((i for i, r in enumerate(tree) if r["concept"] == self.line["concept"]), 0)

        def compose(shown):
            return head + self.window(len(tree), shown, len(tree), lambda i: items[i]) + [{"text": TREE_SCOPE,
                                                                                           "facts": []}]
        shown = fit({at} if tree else set(), sorted(range(len(tree)), key=lambda i: (abs(i - at), i)), compose)
        if shown is None:
            raise ToolError(f"the line's row of the presentation tree exceeds the cap of {PAGE_CHARS} characters "
                            f"({reason})")
        return compose(shown)

    def rows_of(self, section):
        """The rows of another template section, shown as in the start state (a row without a definition says so).
        Every row returned becomes a valid answer (a referral)."""
        if section not in self.other_sections():
            raise ToolError(f"no other section {section!r}; sections: {self.other_sections()}")
        ids = self.rows_by_section[section]
        items = [{"text": t.strip(), "facts": []} for t in rows_block(ids, named=self.v20)[1:]]
        return items, {r: self.names.get(r, r) for r in ids}


class ToolError(Exception):
    pass
