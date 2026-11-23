"""C2n, the calculation-network helper: the selected role's calculation relationships, one network per summation-item arcrole, with the pairs that
cannot be used and why.

Only the two summation-item arcroles count; an arc of any other arcrole is counted under `unsupported` and never
used. Arcs are grouped by (arcrole, parent, child), and each arcrole is its own network (Calculations 1.1, 4.1). A
pair is unusable, with its reason kept, when any of its arcs is prohibited, when the network has more than one arc
for it, or when its weight is not a finite nonzero number. That is a conservative policy, not XBRL's resolution of
equivalent relationships by priority. Every question is asked of one network at a
time: no path, decomposition or reconstruction ever takes edges from two networks. Each network's answer is
"absent" (nothing in it bears on the question, usable or not), "unusable" (what bears on it is in `bad`) or
"supported" (built from its `kids` only). The readers decide what to do with the answers, each by its own rules.

The arcs are read with Filing.arcs("cal", role), which keeps each arc's use, priority and arcrole; the pooled
Filing.calculation is not used here.
"""
from collections import defaultdict
from decimal import Decimal, InvalidOperation

SUMMATION = ("http://www.xbrl.org/2003/arcrole/summation-item", "https://xbrl.org/2023/arcrole/summation-item")
NO_RELATIONSHIPS = "no calculation relationships in role"
BAD_WEIGHT = "weight not a finite nonzero number"


def weight(w):
    """The arc's weight as a finite nonzero Decimal, else None (missing, unparseable, NaN, infinite or zero)."""
    try:
        d = Decimal(w)
    except (TypeError, ValueError, InvalidOperation):
        return None
    return d if d.is_finite() and d != 0 else None


def networks(f, role):
    """{"role", "state", "arcs" (arcrole -> arc count, networks present only), "networks" (arcrole -> {"kids":
    parent -> [(child, Decimal weight)], "bad": (parent, child) -> reason}), "unsupported" (arcs of another
    arcrole)}."""
    groups, unsupported = defaultdict(list), 0
    for a in f.arcs("cal", role):
        if a["role"] != role:
            continue
        if a["arcrole"] not in SUMMATION:
            unsupported += 1
            continue
        groups[(a["arcrole"], a["parent"], a["child"])].append(a)
    nets, counts = {}, defaultdict(int)
    for (ar, p, c), xs in groups.items():
        net = nets.setdefault(ar, {"kids": defaultdict(list), "bad": {}})
        counts[ar] += len(xs)
        w = weight(xs[0]["weight"])
        # an arc with no use attribute is "optional" (XBRL 2.1 default), so .get
        why = ("prohibited" if any(x.get("use") == "prohibited" for x in xs) else "repeated" if len(xs) > 1
               else BAD_WEIGHT if w is None else None)
        if why:
            net["bad"][(p, c)] = why
        else:
            net["kids"][p].append((c, w))
    return {"role": role, "state": "present" if nets else NO_RELATIONSHIPS, "arcs": dict(counts),
            "networks": nets, "unsupported": unsupported}


def edges(net):
    """Every relationship of one network as (parent, child, weight or None, reason or None), usable ones first."""
    return ([(p, c, w, None) for p, kids in net["kids"].items() for c, w in kids]
            + [(p, c, None, why) for (p, c), why in net["bad"].items()])


def children(n, parent):
    """"What are `parent`'s children?", per network: absent when the parent has no outgoing relationship, usable or
    not; unusable when any outgoing relationship is in `bad` (the rest are never summed alone); else
    supported with its parts."""
    out = {}
    for ar, net in n["networks"].items():
        bad = sorted((p, c, why) for (p, c), why in net["bad"].items() if p == parent)
        kids = net["kids"].get(parent, [])
        out[ar] = ({"answer": "unusable", "bad": bad} if bad else
                   {"answer": "supported", "parts": list(kids)} if kids else {"answer": "absent"})
    return out


def paths_up(n, line, anchors, keep=lambda parent: True):
    """"Which paths lead up from `line` to an anchor?", per network. A path climbs one network's relationships to the
    first parent in `anchors`, through parents `keep` accepts, cycles cut. Supported paths use `kids` only; a path
    that needs a relationship in `bad` is listed as unusable. Answer: supported when a supported path exists,
    unusable when only unusable paths exist, absent when no path, usable or not, exists. Each path is (anchor,
    [parents from the line up], [weights]), and an unusable one also carries its bad pairs."""
    out = {}
    for ar, net in n["networks"].items():
        up = defaultdict(list)
        for p, c, w, why in edges(net):
            up[c].append((p, w, why))
        good, bad = [], []

        def walk(c, chain, ws, why, on):
            for p, w, reason in up.get(c, []):
                if p in on or not keep(p):
                    continue
                step = why + ([(p, c, reason)] if reason else [])
                if p in anchors:
                    (bad if step else good).append((p, chain + [p], ws + [w], step))
                else:
                    walk(p, chain + [p], ws + [w], step, on | {p})
        walk(line, [], [], [], {line})
        out[ar] = ({"answer": "supported", "paths": [x[:3] for x in good], "unusable_paths": bad} if good else
                   {"answer": "unusable", "paths": [], "unusable_paths": bad} if bad else {"answer": "absent"})
    return out


def reach_down(n, root, stop=lambda concept: False):
    """"What does `root` reach?", per network: every concept below it with its chains of {parent, child, weight},
    cycles cut, not descending below a concept `stop` accepts. `reach` holds chains over `kids` only;
    `unusable_reach` holds concepts reachable only through a relationship in `bad` (a diagnostic, never a route).
    Answer: absent when the root has no outgoing relationship, usable or not; unusable when all of them are in
    `bad`; else supported. `cuts`: the (parent, child) pairs not followed because the child is already on the
    chain (a cycle cut; part 2, for the A walk's record)."""
    out = {}
    for ar, net in n["networks"].items():
        down = defaultdict(list)
        for p, c, w, why in edges(net):
            down[p].append((c, w, why))
        reach, other, cuts = defaultdict(list), defaultdict(list), set()

        def walk(node, chain, tainted, on):
            for c, w, why in down.get(node, []):
                if c in on:
                    cuts.add((node, c))
                    continue
                step = chain + [{"parent": node, "child": c, "weight": w, **({"unusable": why} if why else {})}]
                (other if tainted or why else reach)[c].append(step)
                if not stop(c):
                    walk(c, step, tainted or bool(why), on | {c})
        walk(root, [], False, {root})
        outgoing = down.get(root, [])
        answer = ("absent" if not outgoing else "unusable" if all(why for _, _, why in outgoing) else "supported")
        out[ar] = {"answer": answer, "reach": dict(reach),
                   "unusable_reach": {c: v for c, v in other.items() if c not in reach}, "cuts": sorted(cuts)}
    return out
