"""XBRL's interval rule for duplicate facts at one key, used by the spread and the grader to tell a consistent
group (read at its most precise fact) from one that is not (refused). Pure code; no model call."""
from collections import defaultdict
from decimal import Decimal


def interval(v, acc):
    """The values v stands for at its accuracy: [v - h, v + h], h half a unit in the last declared place; a point
    for INF. Closed at both ends: both endpoints are included."""
    h = Decimal(0) if acc == "INF" else Decimal(5).scaleb(-acc - 1)
    return v - h, v + h


def consistent(members):
    """XBRL's interval rule over a whole group of (value, accuracy), as applied here: all the intervals share
    a point, and members of one accuracy have one value. Applied to the whole group, never pair by pair."""
    lo, hi = zip(*(interval(v, a) for v, a in members))
    values = defaultdict(set)
    for v, a in members:
        values[a].add(v)
    return max(lo) <= min(hi) and all(len(vs) == 1 for vs in values.values())


def accuracy(decimals):
    """A fact's decimals attribute as consistent() reads it: an int, or "INF" (also when absent)."""
    return "INF" if decimals in (None, "INF") else int(decimals)


def precise_duplicate(facts):
    """For filed facts of one concept, context and unit: the most precise when the group is consistent (XBRL's
    duplicates guidance, as spread.chosen reads it; the first among equals), else None. 1,774,000,000 at -6 and
    1,800,000,000 at -8 are one amount filed twice at two precisions: the first is read, never a conflict."""
    if not consistent([(Decimal(x["value"]), accuracy(x.get("decimals"))) for x in facts]):
        return None
    return max(facts, key=lambda x: float("inf") if accuracy(x.get("decimals")) == "INF"
               else accuracy(x.get("decimals")))
