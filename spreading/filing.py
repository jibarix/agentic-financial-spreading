"""C1-C2 parse: one filing's XBRL instance, presentation, calculation and labels, read by code.

The printed statement is read from the company's own presentation network for the statement role named in
FilingSummary.xml; the facts come from the extracted instance. Files are parsed only, never executed. A linkbase with
no file of its own is read from the schema that embeds it (spreading.edgar checks the schema holds it).
"""
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from . import tagless, template

XL = "{http://www.w3.org/1999/xlink}"
XBRLI = "{http://www.xbrl.org/2003/instance}"
XBRLDI = "{http://xbrl.org/2006/xbrldi}"
LINK = "{http://www.xbrl.org/2003/linkbase}"
XSD = "{http://www.w3.org/2001/XMLSchema}"
XSI = "{http://www.w3.org/2001/XMLSchema-instance}"


def _concept(href):
    """'...#us-gaap_Assets' -> 'us-gaap:Assets'."""
    prefix, _, name = href.split("#")[1].partition("_")
    return f"{prefix}:{name}"


def tag_of(concept):
    """The tag a line's concept names: a split piece's key '<tag>#<axis>=<member>' (corrections, splits) gives its
    original tag, any other concept itself. Semantic readers (definitions, balance, period type) use it; a piece's
    amount is never read through it."""
    return concept.split("#", 1)[0]


def line_ident(f, concept):
    """(tag, dims) of the amount a line's concept names: a split piece's original tag and its member dims (sorted
    pairs), any other concept with no dims. Ownership and overlap compare these, never the key."""
    p = getattr(f, "pieces", {}).get(concept)
    return (p["tag"], tuple(sorted(p["dims"].items()))) if p else (concept, ())


def fact_ident(f, x):
    """(tag, dims) of a fact: a split piece's facts give their original tag, every fact its context's dims."""
    return (x["split_from"] if x.get("piece") else x["concept"], tuple(sorted(f.contexts[x["context"]]["dims"].items())))


def in_period(ctx, key):
    """Whether a context's period is the key exactly: ("instant", date), or ("duration", start, end), both dates."""
    if key[0] == "instant":
        return ctx.get("instant") == key[1]
    return not ctx.get("instant") and ctx.get("start") == key[1] and ctx.get("end") == key[2]


def period_length(start, end):
    """The length in words, from inclusive days: 364 or 371 days are 52 or 53 weeks; the first of a month to the last
    of a month is months; anything else is days."""
    a, b = date.fromisoformat(start), date.fromisoformat(end)
    days = (b - a).days + 1
    if days in (364, 371):
        return f"{days // 7} weeks"
    if a.day == 1 and (b + timedelta(1)).day == 1:
        return f"{(b.year - a.year) * 12 + b.month - a.month + 1} months"
    return f"{days} days"


def period_words(key):
    """The one wording of the run's period, in the start message and every tool. An instant keeps the balance-sheet
    wording of prompt v19 ("at <date>"); its v20 suffix waits for the version gate (C10)."""
    if key[0] == "instant":
        return f"at {key[1]}"
    return f"for the fiscal year {key[1]} to {key[2]} ({period_length(key[1], key[2])}; a flow during the period)"


class Filing:
    def __init__(self, directory):
        self.dir = Path(directory)
        self.stem = next(p.name[:-4] for p in self.dir.glob("*.xsd"))
        self._roots = {}
        self.found = {}         # statement -> how role() found its report (template.find_statement)
        self.printed_as = {}    # (role, tag) -> the tag whose label the line prints: a retagged line (corrections.apply)
        self.pieces = {}        # key -> {tag, dims, fact, label, role}: a line split by member (corrections.apply)
        self._statements()
        self._instance()
        self._labels()
        self._extension()

    # -- FilingSummary: which roles are the statements
    def _statements(self):
        root = ET.parse(self.dir / "FilingSummary.xml").getroot()
        self.statements = [{"name": r.findtext("ShortName"), "role": r.findtext("Role")}
                           for r in root.iter("Report") if r.findtext("MenuCategory") == "Statements"]

    def role(self, kind):
        """kind: a statement of the template. The first statement role whose name the template's title words match
        and its exclusions do not, else the one its title fallback allows (template.find_statement); None when none.
        self.found[kind] keeps how it was found and why."""
        hits, how, why = template.find_statement(template.active().statements[kind],
                                                 [(s["role"], s["name"]) for s in self.statements],
                                                 lambda r: {p["concept"] for p in self.printed(r)})
        self.found[kind] = {"how": how, "role": hits[0] if hits else None, "why": why}
        return hits[0] if hits else None

    def _root(self, suffix):
        """The parsed linkbase <stem><suffix>, or the schema when the filing embeds that linkbase there."""
        path = self.dir / f"{self.stem}{suffix}"
        if not path.is_file():
            path = self.dir / f"{self.stem}.xsd"
        if path not in self._roots:
            self._roots[path] = ET.parse(path).getroot()
        return self._roots[path]

    # -- instance: contexts, units, facts (the one extracted instance, named after the main document)
    def _instance(self):
        inst = sorted(self.dir.glob("*_htm.xml"))
        if len(inst) != 1:
            raise ValueError(f"{self.dir}: {len(inst)} extracted instances")
        path = inst[0]
        root = ET.parse(path).getroot()
        nsmap = dict(re.findall(r'xmlns:([\w.-]+)="([^"]+)"', path.read_text(encoding="utf8")[:20000]))
        prefix_of = {v: k for k, v in nsmap.items()}
        self.nsmap = nsmap
        self.contexts = {}
        for c in root.iter(f"{XBRLI}context"):
            p = c.find(f"{XBRLI}period")
            inst = p.findtext(f"{XBRLI}instant")
            dims = {m.get("dimension"): (m.text or "").strip() for m in c.iter(f"{XBRLDI}explicitMember")}
            dims.update({m.get("dimension"): "typed" for m in c.iter(f"{XBRLDI}typedMember")})
            ident = c.find(f"{XBRLI}entity/{XBRLI}identifier")
            self.contexts[c.get("id")] = {"instant": inst, "start": p.findtext(f"{XBRLI}startDate"),
                                          "end": inst or p.findtext(f"{XBRLI}endDate"), "dims": dims,
                                          "entity": (ident.get("scheme"), (ident.text or "").strip())
                                          if ident is not None else None}
        self.units = {u.get("id"): " / ".join(m.text for m in u.iter(f"{XBRLI}measure"))
                      for u in root.iter(f"{XBRLI}unit")}
        # a measure's prefix is resolved through the unit element's own scope, so compare expanded names
        self.unit_measures = {}
        for u in root.iter(f"{XBRLI}unit"):
            ms = [m.text.strip() for m in u.iter(f"{XBRLI}measure")]
            self.unit_measures[u.get("id")] = tuple(self._expand(m, nsmap) for m in ms)
        self.facts = []
        for el in root:
            if el.get("contextRef") is None:
                continue
            ns, name = el.tag[1:].split("}")
            self.facts.append({"concept": f"{prefix_of.get(ns, ns)}:{name}", "context": el.get("contextRef"),
                               "unit": el.get("unitRef"), "decimals": el.get("decimals"),
                               "id": el.get("id"), "value": (el.text or "").strip(),
                               # an explicit xsi:nil (empty value too); kept apart from a malformed empty value
                               **({"nil": True} if el.get(f"{XSI}nil") == "true" else {})})
        self.by_concept = defaultdict(list)
        for f in self.facts:
            self.by_concept[f["concept"]].append(f)
        dei = self.by_concept.get("dei:DocumentPeriodEndDate")
        self.period_end = dei[0]["value"] if dei else None
        self.entity = self.contexts[dei[0]["context"]]["entity"] if dei else None

    @staticmethod
    def _expand(measure, nsmap):
        prefix, _, local = measure.rpartition(":")
        return f"{{{nsmap.get(prefix, prefix)}}}{local}"

    def currency(self, unit_id):
        """ISO 4217 code for a unit that is one plain currency measure, else None."""
        ms = self.unit_measures.get(unit_id, ())
        if len(ms) == 1 and ms[0].startswith("{http://www.xbrl.org/2003/iso4217}"):
            return ms[0].split("}")[1]
        return None

    # -- labels (company label linkbase: standard and extension concepts as the company labels them)
    def _labels(self):
        root = self._root("_lab.xml")
        loc, lab, arcs = {}, {}, []
        for lk in root.iter(f"{LINK}labelLink"):
            for e in lk:
                t = e.get(f"{XL}type")
                if t == "locator":
                    loc[e.get(f"{XL}label")] = _concept(e.get(f"{XL}href"))
                elif t == "resource":
                    lab.setdefault(e.get(f"{XL}label"), []).append((e.get(f"{XL}role"), (e.text or "").strip()))
                elif t == "arc":
                    arcs.append((e.get(f"{XL}from"), e.get(f"{XL}to")))
        self.labels = defaultdict(dict)
        for a, b in arcs:
            for role, text in lab.get(b, []):
                self.labels[loc[a]][role.rsplit("/", 1)[1]] = text

    def label(self, concept, preferred=None):
        if concept in self.pieces:
            return self.pieces[concept]["label"]
        d = self.labels.get(concept, {})
        if preferred and preferred.rsplit("/", 1)[1] in d:
            return d[preferred.rsplit("/", 1)[1]]
        return d.get("label") or tagless.show(concept)      # no label filed: the tag, or its ID with tags hidden

    # -- extension schema: balance and period type of the company's own concepts
    def _extension(self):
        root = ET.parse(self.dir / f"{self.stem}.xsd").getroot()
        prefix = self.stem.split("-")[0]
        self.extension = {f"{prefix}:{e.get('name')}": {
            "balance": e.get(f"{XBRLI}balance"), "period": e.get(f"{XBRLI}periodType"),
            "abstract": e.get("abstract") == "true", "type": e.get("type")}
            for e in root.iter(f"{XSD}element")}

    # -- presentation and calculation networks for one role
    def arcs(self, kind, role=None):
        link = {"pre": "presentationLink", "cal": "calculationLink", "def": "definitionLink"}[kind]
        arc = {"pre": "presentationArc", "cal": "calculationArc", "def": "definitionArc"}[kind]
        root = self._root(f"_{kind}.xml")
        out = []
        for lk in root.iter(f"{LINK}{link}"):
            if role and lk.get(f"{XL}role") != role:
                continue
            loc = {e.get(f"{XL}label"): _concept(e.get(f"{XL}href"))
                   for e in lk if e.get(f"{XL}type") == "locator"}
            for e in lk.iter(f"{LINK}{arc}"):
                out.append({"role": lk.get(f"{XL}role"), "parent": loc[e.get(f"{XL}from")],
                            "child": loc[e.get(f"{XL}to")], "order": float(e.get("order") or 0),
                            "weight": e.get("weight"), "preferred": e.get("preferredLabel"),
                            "arcrole": e.get(f"{XL}arcrole"),
                            "target_role": e.get("{http://xbrl.org/2005/xbrldt}targetRole"),
                            "use": e.get("use"), "priority": e.get("priority"),
                            "usable": e.get("{http://xbrl.org/2005/xbrldt}usable")})
        return out

    def printed(self, role):
        """The presentation tree in printed order, one entry per occurrence:
        [{concept, depth, parent, path (ancestors, root first), preferred, label}]."""
        arcs = self.arcs("pre", role)
        kids = defaultdict(list)
        for a in arcs:
            kids[a["parent"]].append(a)
        children = {a["child"] for a in arcs}
        out = []

        def walk(node, path, preferred):
            out.append({"concept": node, "depth": len(path), "parent": path[-1] if path else None,
                        "path": list(path), "preferred": preferred,
                        "label": self.label(self.printed_as.get((role, node), node), preferred)})
            if node in path:
                return
            for a in sorted(kids.get(node, []), key=lambda a: a["order"]):
                walk(a["child"], path + [node], a["preferred"])
        for r in [p for p in kids if p not in children]:
            walk(r, [], None)
        return out

    def calculation(self, role):
        """parent -> [(child, weight)] from the company's calculation network for the role."""
        out = defaultdict(list)
        for a in self.arcs("cal", role):
            out[a["parent"]].append((a["child"], Decimal(a["weight"])))
        return out

    def qname(self, expanded):
        """'http://fasb.org/us-gaap/2025#Name' -> 'us-gaap:Name', by the instance's own namespace map; None if
        the namespace is not declared."""
        ns, _, name = expanded.partition("#")
        prefix = next((p for p, uri in self.nsmap.items() if uri == ns), None)
        return f"{prefix}:{name}" if prefix and name else None

    def current(self, kind):
        """The run's period key, from the context of dei:DocumentPeriodEndDate: ("duration", start, end) or
        ("instant", end). An ambiguous period is refused with its reason, never guessed."""
        dei = self.by_concept.get("dei:DocumentPeriodEndDate") or []
        if not dei:
            raise RuntimeError("C1: no dei:DocumentPeriodEndDate; the run's period is not known")
        if len({x["context"] for x in dei}) > 1 or len({x["value"] for x in dei}) > 1:
            raise RuntimeError("C1: dei:DocumentPeriodEndDate is filed in two contexts or with two values")
        ctx = self.contexts[dei[0]["context"]]
        if ctx["instant"]:
            raise RuntimeError("C1: the context of dei:DocumentPeriodEndDate is an instant, not the fiscal period")
        if ctx["end"] != dei[0]["value"]:
            raise RuntimeError(f"C1: dei:DocumentPeriodEndDate is {dei[0]['value']}, its context ends {ctx['end']}")
        return ("instant", ctx["end"]) if kind == "instant" else ("duration", ctx["start"], ctx["end"])

    def tag_arcs(self, kind, role=None):
        """The arcs by tag, before any split (corrections.apply sets the retagged ones): the networks that say
        whether two (tag, dims) amounts overlap."""
        return self.arcs(kind, role)

    def facts_for(self, concept, key):
        """The concept's facts of the filing's own entity in the period `key` exactly (in_period), with no
        dimensions and a value; a split piece's key gives its one pinned fact, dimensioned (corrections.apply).
        Each carries its resolved currency, None for a unit that is not one currency."""
        out = []
        for f in self.by_concept.get(concept, []):
            ctx = self.contexts[f["context"]]
            if (in_period(ctx, key) and (not ctx["dims"] or f.get("piece")) and ctx["entity"] == self.entity
                    and f["value"] != ""):
                out.append({**f, "currency": self.currency(f["unit"])})
        return out

    def instant_facts(self, concept, date):
        """Balance-sheet facts: the filing's own entity, an instant on the date, no dimensions, a plain currency
        unit. Each fact carries its resolved currency."""
        return [x for x in self.facts_for(concept, ("instant", date)) if x["currency"]]

    def value(self, concept, date, dims=None):
        """Facts for a concept at an instant (or a duration ending on date) with exactly these dims."""
        dims = dims or {}
        return [f for f in self.by_concept.get(concept, [])
                if self.contexts[f["context"]]["end"] == date and self.contexts[f["context"]]["dims"] == dims
                and f["value"] != ""]
