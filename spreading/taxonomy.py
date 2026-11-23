"""Read the FASB US GAAP taxonomy package (a zip) with the standard library.

Gives, for every concept: label, documentation, balance, period type, data type, abstract flag,
and where the taxonomy places it in each network (presentation parent, calculation parent and weight).
"""
import re
import zipfile
import xml.etree.ElementTree as ET

from .runrec import ROOT

SOURCES = ROOT / "data/sources"
FETCH = ".venv-agents/Scripts/python.exe -s -E -m spreading.fasb_fetch fetch {year}"

XS = "{http://www.w3.org/2001/XMLSchema}"
XBRLI = "{http://www.xbrl.org/2003/instance}"
LINK = "{http://www.xbrl.org/2003/linkbase}"
XL = "{http://www.w3.org/1999/xlink}"
ROLE_LABEL = "http://www.xbrl.org/2003/role/label"
ROLE_TERSE = "http://www.xbrl.org/2003/role/terseLabel"
ROLE_DOC = "http://www.xbrl.org/2003/role/documentation"


def _concept(href):
    """'../elts/us-gaap-2025.xsd#us-gaap_Assets' -> 'us-gaap:Assets'."""
    frag = href.split("#", 1)[1]
    prefix, name = frag.split("_", 1)
    return f"{prefix}:{name}"


class Taxonomy:
    def __init__(self, zip_path):
        self.zip = zipfile.ZipFile(zip_path)
        self.names = self.zip.namelist()
        self.year = re.search(r"us-gaap-(\d{4})/", self.names[0] + "/".join(self.names[:5])).group(1)
        self.concepts = {}
        self._read_schema()
        self._read_labels(f"elts/us-gaap-lab-{self.year}.xml", {ROLE_LABEL: "label", ROLE_TERSE: "terse"})
        self._read_labels(f"elts/us-gaap-doc-{self.year}.xml", {ROLE_DOC: "documentation"})

    def _file(self, suffix):
        hits = [n for n in self.names if n.endswith(suffix)]
        if len(hits) != 1:
            raise FileNotFoundError(f"{suffix}: {len(hits)} matches")
        return ET.fromstring(self.zip.read(hits[0]))

    def _read_schema(self):
        root = self._file(f"elts/us-gaap-{self.year}.xsd")
        for el in root.iter(XS + "element"):
            self.concepts[f"us-gaap:{el.get('name')}"] = {
                "concept": f"us-gaap:{el.get('name')}",
                "type": el.get("type", "").split(":")[-1],
                "balance": el.get(XBRLI + "balance"),
                "period": el.get(XBRLI + "periodType"),
                "abstract": el.get("abstract") == "true",
            }

    def _read_labels(self, suffix, roles):
        root = self._file(suffix)
        for link in root.iter(LINK + "labelLink"):
            locs, res, arcs = {}, {}, []
            for e in link:
                kind = e.get(XL + "type")
                if kind == "locator":
                    locs[e.get(XL + "label")] = _concept(e.get(XL + "href"))
                elif kind == "resource" and e.get(XL + "role") in roles:
                    res.setdefault(e.get(XL + "label"), []).append((roles[e.get(XL + "role")], e.text or ""))
                elif kind == "arc":
                    arcs.append((e.get(XL + "from"), e.get(XL + "to")))
            for frm, to in arcs:
                c = self.concepts.get(locs.get(frm))
                for field, text in res.get(to, []):
                    if c is not None:
                        c[field] = text.strip()

    def network(self, stem, kind):
        """Arcs of one network file. kind 'pre' -> (role, parent, child, order, preferred label);
        kind 'cal' -> (role, parent, child, weight)."""
        root = self._file(f"{stem}-{kind}-{self.year}.xml")
        tag = LINK + ("presentationLink" if kind == "pre" else "calculationLink")
        out = []
        for link in root.iter(tag):
            role = link.get(XL + "role")
            locs = {e.get(XL + "label"): _concept(e.get(XL + "href"))
                    for e in link if e.get(XL + "type") == "locator"}
            for e in link:
                if e.get(XL + "type") != "arc":
                    continue
                p, c = locs[e.get(XL + "from")], locs[e.get(XL + "to")]
                if kind == "pre":
                    out.append((role, p, c, float(e.get("order", 0)), e.get("preferredLabel")))
                else:
                    out.append((role, p, c, float(e.get("weight"))))
        return out

    def is_amount(self, concept):
        c = self.concepts.get(concept)
        return bool(c) and not c["abstract"] and c["type"] == "monetaryItemType"


# ---- editions on disk: definitions and balances come from the edition the filing itself uses ------------------------

class MissingEdition(RuntimeError):
    pass


def edition_path(year, base=None):
    return (base or SOURCES) / f"fasb_us-gaap_{year}" / f"us-gaap-{year}.zip"


def edition(year, base=None):
    """The package of us-gaap `year`, or MissingEdition naming the command that fetches it. An unloaded edition is never a reason to skip a
    company; the batch fetches every edition it needs before its freeze."""
    p = edition_path(year, base)
    if not p.is_file():
        raise MissingEdition(f"us-gaap {year} is not on disk ({p}); fetch it with: {FETCH.format(year=year)}")
    return p


def editions(base=None):
    """Every package on disk, by year: what the freeze hashes."""
    base = base or SOURCES
    out = {}
    for d in sorted(base.glob("fasb_us-gaap_*")):
        year = d.name.rsplit("_", 1)[-1]
        if re.fullmatch(r"\d{4}", year) and edition_path(year, base).is_file():
            out[year] = edition_path(year, base)
    return out
