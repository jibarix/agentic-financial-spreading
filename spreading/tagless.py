"""Tags hidden (an analyst never reads the SEC tag): the models see no XBRL tag
name, no FASB or company definition of a tag and no balance type. A line is named by a neutral ID (item_1, item_2,
...) wherever a model would have read its tag, and a tool call naming an ID is read back as its tag. Code keeps the
tags for everything it computes and checks. Off (outside a spread or the tree: the job's parent, grading, the UI),
every name is the tag.

    tagless.start(f)            # spread.main and the tree, always
    tagless.show("us-gaap:Revenues")    # 'item_12'
    tagless.back("item_12")             # 'us-gaap:Revenues'
"""
import re

ON = False
_f, _ids, _tags = None, {}, {}
# a prefixed XBRL name (us-gaap:Revenues, hsy:Foo, hsy:lowercaseName, srt:ProductOrServiceAxis): what must not
# reach a model; with tags hidden a hit counts when its prefix is a standard one or one of the filing's
QNAME = re.compile(r"\b[A-Za-z][\w.-]*:[A-Za-z_][\w.-]*")
STANDARD = {"us-gaap", "dei", "srt", "ifrs-full", "country", "currency", "exch", "naics", "sic", "stpr", "ecd", "cyd",
            "invest", "xbrli", "xbrldi"}
UNKNOWN = "not a line ID of this filing; name a line by an item ID as shown, e.g. item_12"


def start(f):
    """Switch on for filing `f`: every concept it files or names in a calculation gets an ID in sorted order of the
    tag, then every split piece after them, so a split leaves every other line's ID as it was."""
    global ON, _f, _ids, _tags
    pieces = set(getattr(f, "pieces", {}))     # corrections.apply files each piece's facts under its own key too
    names = (set(f.by_concept) | {a[k] for a in f.arcs("cal") for k in ("parent", "child")}) - pieces
    ON, _f = True, f
    _ids = {c: f"item_{i}" for i, c in enumerate(sorted(names) + sorted(pieces), 1)}
    _tags = {v: k for k, v in _ids.items()}


def stop():
    global ON, _f, _ids, _tags
    ON, _f, _ids, _tags = False, None, {}, {}


def show(concept):
    """How a model sees a concept's name: its ID with tags hidden (a name outside the filing's list gets the next
    ID), else the tag."""
    if not ON:
        return concept
    if concept not in _ids:
        _ids[concept] = f"item_{len(_ids) + 1}"
        _tags[_ids[concept]] = concept
    return _ids[concept]


def ref(concept):
    """A concept named in a sentence: with tags hidden, its ID and the company's label for it (the ID alone when
    no label is filed: Filing.label gives the ID then)."""
    if not ON:
        return concept
    text = _f.label(concept)
    return text if text == show(concept) else f"{show(concept)} ({text})"


def member(qname, as_filed):
    """An axis or member: with tags hidden, the company's label for it (its ID when none is filed), else
    `as_filed`."""
    return _f.label(qname) if ON else as_filed


def known(name):
    """Whether a model's name for a concept is one of the IDs shown; off, every name passes."""
    return not ON or str(name).strip() in _tags


def back(name):
    """A model's name for a concept, read back as the tag (an unknown name is kept, and the tool says so)."""
    return _tags.get(name.strip(), name) if ON else name


def leaks(*texts):
    """The prefixed XBRL names in text bound for a model, with tags hidden; none when off."""
    if not ON:
        return []
    prefixes = STANDARD | {c.split(":")[0] for c in _ids if ":" in c}
    return sorted({m for t in texts for m in QNAME.findall(t or "") if m.split(":")[0] in prefixes})


def hide(x):
    """A request bound for a model (text, or JSON-like data, keys included) with every prefixed name of the filing
    or a standard taxonomy shown as its ID; unchanged when off. The tree's roles send their requests through it."""
    if not ON:
        return x
    if isinstance(x, str):
        prefixes = STANDARD | {c.split(":")[0] for c in _ids if ":" in c}
        return QNAME.sub(lambda m: show(m.group(0)) if m.group(0).split(":")[0] in prefixes else m.group(0), x)
    if isinstance(x, dict):
        return {hide(k): hide(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [hide(v) for v in x]
    return x


def unhide(x):
    """A model's reply with every string that is exactly an ID read back as its tag (text that only mentions an
    ID, a quote of a tool's output, is kept as written); unchanged when off."""
    if not ON:
        return x
    if isinstance(x, str):
        return _tags.get(x.strip(), x)
    if isinstance(x, dict):
        return {unhide(k): unhide(v) for k, v in x.items()}
    if isinstance(x, list):
        return [unhide(v) for v in x]
    return x
