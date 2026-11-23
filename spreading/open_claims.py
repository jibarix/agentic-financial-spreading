"""Open claims of the cross-dimension overlap policy. Code only: nothing here decides a claim.

In a root-cause job (--root-cause-fixes) the Relations object carries `cross_dims_unknown`. A piece whose overlap with a
printed line, an owned amount or another piece is "unknown" is then committed with an open claim per rival:
provisional: true and open_claims on the piece, on the rest of its line and on every total computed from them; one
"unresolved overlap" exception per committed piece with one decision card per claim; and the counts. Nothing in the
run settles a claim. A job without that option builds none of this and its records are unchanged.

Two kinds of unknown overlap are kept apart everywhere:
  RELAXED  the pair the policy relaxed: different concepts related by a calculation, in contexts with different dims. The legacy
           validator refuses these as known_overlap.
  ALREADY  a pair the validator already read as unknown (for example no calculation network relates them).
"""
import hashlib
import json
from decimal import Decimal
from types import SimpleNamespace

from . import steps

KIND = "unresolved overlap"
RELAXED, ALREADY = "newly relaxed (R7)", "already unknown"
ACTIONS = ["confirm", "edit", "escalate"]
ACTIONS_NOTE = "proposed actions; no analyst acts in this run"
STAKE_BASIS = "the piece's amount: the most this claim could count twice"
# what the legacy validator answers an attempt the policy relaxed, by the refusal it gave the same piece: the checks before
# known_overlap in six_rules.amounts are the same in both arms; the two after it are reached only without a clash
T_FIRST = ("entity", "period", "currency", "conflicting_duplicates", "rounded_duplicate", "ancestor")


SCOPE = ("the cross-dimension overlap policy applies to placement, the vote path and source-run checking; it is off only in steps.split, the check "
         "of a newly proposed repair split")
PLACEMENT, REPAIR = "placement", "repair split (legacy policy; reviewed by C8)"
CASE_NOTE = ("one case per piece; its claims concern one amount, so the amount at stake is the piece's amount once, "
             "never the sum over its claims")


def on(rel):
    """Whether this Relations object is a root-cause job's (policy on)."""
    return bool(getattr(rel, "cross_dims_unknown", False))


def claim_id(filing, fact, rival):
    """A claim's stable identity: the frozen filing (its folder name, the accession), the piece's fact and the amount
    it may overlap, by name (steps.name: concept and dims; a repair split's claim about another piece names that
    piece's fact). The run, the path, the attempt and the claim's kind are not part of it: they describe an
    occurrence, so a repeated notice or an inherited mark is the same claim."""
    return "oc-" + hashlib.sha256(json.dumps([filing, fact, rival]).encode("utf8")).hexdigest()[:16]


def entry(rel, k, res, basis, fid=None):
    """One unknown overlap of a piece with the amount `k`, as six_rules.amounts records it. The first three keys are
    the record as it always was; a root-cause job adds the kind, the rival's key and the claim's id."""
    e = {"with": steps.name(k), "reading": res, "basis": basis}
    if on(rel):
        e.update(kind=RELAXED if basis == steps.R7_BASIS else ALREADY, key=[k[0], [list(p) for p in k[1]]],
                 id=claim_id(getattr(rel, "filing", None), fid, e["with"]))
    return e


def marked(c):
    """Whether a committed claim carries the policy's mark. A repair split's piece has its own `provisional` (steps.split:
    "provisional until C8 review") and no open_claims; that is not this mark."""
    return c.get("provisional") is True and bool(c.get("open_claims"))


def recorded(c, filing):
    """Every unknown-overlap claim a committed claim records, in either arm: placement's `unknown_overlaps`
    (six_rules.amounts; kinds in a root-cause job, all already unknown in a legacy job, whose validator refuses the pairs the policy
    relaxes) and a repair split's `review` claims of overlap (steps.split, legacy policy: kind rival or piece). Each
    {id, kind, with, origin, reading, basis}. What a consumer does with them is the consumer's rule."""
    out = [{"id": u.get("id") or claim_id(filing, c["fact"], u["with"]), "kind": u.get("kind") or ALREADY,
            "with": u["with"], "origin": PLACEMENT, "reading": u.get("reading"), "basis": u.get("basis")}
           for u in c.get("unknown_overlaps") or ()]
    for r in c.get("review") or ():
        k = r.get("claim") or {}
        if k.get("kind") in ("rival", "piece"):
            out.append({"id": claim_id(filing, c["fact"], k.get("target")), "kind": ALREADY, "with": k.get("target"),
                        "origin": REPAIR, "reading": r.get("reading"), "basis": r.get("basis")})
    return out


def record_gaps(c, r_job=None):
    """What a committed claim's record of its unknown overlaps lacks, as reasons; [] when it reads. A record with
    a gap is never read as holding no claim, and its kind is never filled in.
    Either arm: `unknown_overlaps`, where the key is present, is a list of entries that name their amount.
    A piece with the policy's own mark (marked, and not the rest of a line, which carries `provisional_from`) records
    one unknown overlap per open claim, at least one.
    A root-cause record (`r_job` True; None: told from the record, by the piece's own mark or an entry's kind, id or
    key) gives every entry an explicit kind, RELAXED or ALREADY. A legacy record carries no kind and keeps its
    documented reading, all already unknown (recorded)."""
    if "unknown_overlaps" not in c:
        uo = None
    elif not isinstance(c["unknown_overlaps"], list):
        return ["unknown_overlaps is not a list"]
    else:
        uo = c["unknown_overlaps"]
        if not all(isinstance(u, dict) and isinstance(u.get("with"), str) and u["with"] for u in uo):
            return ["an unknown overlap does not name the amount it concerns"]
    gaps = []
    own = marked(c) and "provisional_from" not in c
    if own and not isinstance(c["open_claims"], list):
        gaps.append("the piece's open claims are not a list")
    elif own and not uo:
        gaps.append("the piece is marked provisional with open claims and records no unknown overlap")
    elif own and len(uo) != len(c["open_claims"]):
        gaps.append(f"the piece is marked with {len(c['open_claims'])} open claims and records {len(uo)} "
                    "unknown overlaps")
    if r_job is None:
        r_job = own or any(k in u for u in uo or () for k in ("kind", "id", "key"))
    if r_job:
        gaps += [f"the unknown overlap with {u['with']} has no valid kind" for u in uo or ()
                 if u.get("kind") not in (RELAXED, ALREADY)]
    return gaps


def assessable(c, line_fact):
    """Whether the record lets code read a committed claim's overlap claims. A whole line or the rest of one (the
    line's own fact) records none by design. A piece (another fact) is read from `unknown_overlaps` or `review`; a
    piece with neither as a list was written without its claims and cannot be assessed: it is never read as
    claim-free."""
    if c.get("fact") == line_fact:
        return True
    return isinstance(c.get("unknown_overlaps"), list) or isinstance(c.get("review"), list)


def claim_text(u):
    return f"{KIND} with {u['with']}"


def piece_marks(rel, unknown):
    """The keys a root-cause job's piece carries beside unknown_overlaps: none in a legacy job, none without an unknown overlap."""
    if not on(rel) or not unknown:
        return {}
    return {"provisional": True, "open_claims": [claim_text(u) for u in unknown]}


def attempt(rel, line, fid, unknown, refused):
    """Log one attempted piece that the policy relaxed (rule 5 would otherwise have met a known overlap). `refused`: the
    refusal this piece got from a later check, or None. The caller adds how the whole answer ended (closed)."""
    if on(rel):     # every piece validated under the cross-dimension policy: the denominator of the eligible attempts
        rel.validated = getattr(rel, "validated", 0) + 1
    relaxed = [u for u in unknown if u.get("kind") == RELAXED]
    if not on(rel) or not relaxed:
        return
    kind = refused["kind"] if refused else None
    rel.r7_log.append({"line": line["concept"], "fact": fid, "rivals": [u["with"] for u in relaxed],
                       "claims": [u.get("id") for u in relaxed], "piece_refused": kind,
                       "t_validator": kind if kind in T_FIRST else "known_overlap"})


def closed(rel, start, refused, **where):
    """Close the attempts logged since `start` with their answer's end: admitted, or the kinds that refused it."""
    if not on(rel):
        return []
    new = rel.r7_log[start:]
    for e in new:
        if "answer_admitted" not in e:
            e.update(answer_admitted=not refused and e["piece_refused"] is None,
                     answer_refusals=sorted({r["kind"] for r in refused or ()}))
        e.update({k: v for k, v in where.items() if v is not None and k not in e})
    return new


def is_piece(c):
    return "unknown_overlaps" in c


def remainders(claims, line_fact=None):
    """The rest of a line whose committed piece has an open claim is provisional too, with the piece's claims.
    `line_fact` (concept -> the line's own fact; a cumulative version, where a line may also hold repair pieces):
    the rest is the claim on the line's own fact, and its marks are set afresh, so a rest whose marked piece is gone
    loses them and one whose piece stays keeps them. Without it (a spread run): every claim that is not a piece."""
    by = {}
    for c in claims:
        if is_piece(c) and marked(c):
            by.setdefault(c["source"], []).append(c)
    for c in claims:
        rest = not is_piece(c) if line_fact is None else c.get("fact") == line_fact.get(c["source"])
        if not rest:
            continue
        if line_fact is not None:
            for k in ("provisional", "open_claims", "provisional_from"):
                c.pop(k, None)
        if c["source"] in by:
            c["provisional"] = True
            c["open_claims"] = [x for p in by[c["source"]] for x in p["open_claims"]]
            c["provisional_from"] = [p["fact"] for p in by[c["source"]]]


def inherit(before, after, line_fact):
    """A repair commit rebuilds its section's claims from the draft (repair.claims_of), which keeps a recorded
    piece's fact, row and amount and none of its record. Each rebuilt piece (a fact other than its line's) that the
    previous version held with the same line, fact and amount takes back every key the rebuild left out: its unknown
    overlaps, marks and open claims, its quote, membership and review. A row the repair changed stays changed. A
    piece the repair newly proposed has no previous record and takes nothing. Returns the facts carried. No review
    result is read here: C8's agreement on a line clears nothing."""
    old = {(c["source"], c["fact"]): c for c in before if c.get("fact") != line_fact.get(c["source"])}
    got = []
    for c in after:
        o = old.get((c["source"], c["fact"]))
        if (o is None or c.get("fact") == line_fact.get(c["source"])
                or Decimal(str(o["amount"])) != Decimal(str(c["amount"]))):
            continue
        for k, v in o.items():
            if k not in c and k != "section":
                c[k] = v
        if marked(o):       # the same fact and amount on the same line: the claim stands whoever proposed it again
            c.update(provisional=True, open_claims=o["open_claims"], unknown_overlaps=o["unknown_overlaps"])
        got.append(c["fact"])
    return got


def quote_of(piece, episode):
    """(quote, passage, whose): the piece's own checked quote, else the quote of the answer that gave the piece."""
    qs = piece.get("quote_source")
    if qs:
        return qs.get("quote"), qs.get("block"), "the piece's own quote"
    which = {"vote": "review", "C4": "c4"}.get(piece.get("by"))
    rec = (episode or {}).get(which) if which else None
    qs = ((rec or {}).get("answer") or {}).get("_quote_source") if isinstance(rec, dict) else None
    if qs:
        return qs.get("quote"), qs.get("block"), f"the {'checker' if which == 'review' else 'analyst'}'s answer quote"
    return None, None, "no quote recorded for this piece"


def exceptions(claims, lines, c4, amount_of, step="C5"):
    """One exception per committed piece with an open claim, with one decision card per claim, newly relaxed first.
    `amount_of(key)`: the rival's amount in the run's period, or None when the run holds none. The amount at stake
    is computed here: the piece's amount, without its sign."""
    by = {l["concept"]: l for l in lines}
    out = []
    for c in claims:
        if not (is_piece(c) and marked(c)):
            continue
        line = by[c["source"]]
        quote, passage, whose = quote_of(c, c4.get(c["source"]))
        cards = []
        for u in sorted(c["unknown_overlaps"], key=lambda u: u.get("kind") != RELAXED):
            key = (u["key"][0], tuple(tuple(p) for p in u["key"][1]))
            rival = amount_of(key)
            cards.append({"claim_id": u.get("id"), "kind": u["kind"], "line": c["source"],
                          "line_label": line.get("label"),
                          "line_amount": line["amount"], "row": c["row"],
                          "piece": {"fact": c["fact"], "concept": c["concept"], "dims": c.get("dims"),
                                    "amount": c["amount"]},
                          "rival": {"name": u["with"], "amount": rival,
                                    **({} if rival is not None else {"amount_note": "no amount of it in this run"})},
                          "basis": u["basis"], "reading": u["reading"], "quote": quote, "passage": passage,
                          "quote_of": whose, "amount_at_stake": abs(Decimal(c["amount"])),
                          "amount_at_stake_basis": STAKE_BASIS, "unit": line.get("unit"),
                          "actions": list(ACTIONS), "actions_note": ACTIONS_NOTE})
        n = sum(1 for x in cards if x["kind"] == RELAXED)
        out.append({"step": step, "class": KIND, "concept": c["source"],
                    "why": f"{c['fact']} ({c['amount']}) on {c['row']} is committed with {len(cards)} open claim"
                           f"{'s' if len(cards) != 1 else ''} ({n} {RELAXED}, {len(cards) - n} {ALREADY}); the "
                           "amounts stay and are provisional; nothing in this run settles a claim",
                    "piece": c["fact"], "open_claims": len(cards), "case": case_of(c, cards), "cards": cards})
    return out


def case_of(c, cards):
    """The review case of one piece: its claims as a list, newly relaxed first and counted apart from the already
    unknown ones. Grouping changes no claim: each card stays, with its own kind and id."""
    return {"piece": c["fact"], "line": c["source"], "row": c["row"],
            "claims": [{"claim_id": k.get("claim_id"), "kind": k["kind"], "rival": k["rival"]["name"]}
                       for k in cards],
            RELAXED: sum(1 for k in cards if k["kind"] == RELAXED),
            ALREADY: sum(1 for k in cards if k["kind"] == ALREADY),
            "amount_at_stake": abs(Decimal(str(c["amount"]))), "amount_at_stake_note": CASE_NOTE}


def carried(exc, claims):
    """(exceptions, dropped): a cumulative version's "unresolved overlap" exceptions after a repair. An exception
    stays, with every card, while its piece is still committed with its mark; where the repair changed the piece's
    row, the cards, the case and the sentence name the new row. An exception whose piece is no longer committed is
    dropped and listed with why. Every other exception is returned untouched, in order."""
    pieces = {(c["source"], c["fact"]): c for c in claims if is_piece(c) and marked(c)}
    out, dropped = [], []
    for x in exc:
        if x.get("class") != KIND:
            out.append(x)
            continue
        c = pieces.get((x.get("concept"), x.get("piece")))
        if c is None:
            dropped.append({"line": x.get("concept"), "piece": x.get("piece"), "open_claims": x.get("open_claims"),
                            "why": "the piece is no longer committed in this version"})
            continue
        old = (x.get("cards") or [{}])[0].get("row")
        if old is not None and old != c["row"]:
            x = json.loads(json.dumps(x, default=str))
            for k in x["cards"]:
                k["row"] = c["row"]
            if "case" in x:
                x["case"]["row"] = c["row"]
            x["why"] = x["why"].replace(f" on {old} is committed", f" on {c['row']} is committed")
        out.append(x)
    return out, dropped


def recipe_rows(t, rid, seen=()):
    """Every row a template total is computed from, through the totals among its terms."""
    out = set()
    for term in (t.recipes.get(rid) or {}).get("terms") or ():
        r = term.get("row")
        if r is None or r in seen:
            continue
        out.add(r)
        if r in t.recipes:
            out |= recipe_rows(t, r, seen + (rid,))
    return out


def provisional_rows(claims):
    """row -> its open claims, from every provisional claim on it (pieces and rests)."""
    rows = {}
    for c in claims:
        if marked(c):
            rows.setdefault(c["row"], []).extend(c["open_claims"])
    return rows


def mark_totals(t, totals, rows):
    """Every computed total over a provisional row is provisional, with the rows it takes the claims from."""
    marked = []
    for rid, v in totals.items():
        mine = sorted(recipe_rows(t, rid) & set(rows))
        if mine:
            v.update(provisional=True, provisional_rows=mine,
                     open_claims=sorted({x for r in mine for x in rows[r]}))
            marked.append(rid)
    return marked


def summary(rel, claims, exc, rows, totals):
    """The run's record of the cross-dimension policy: attempts eligible for it, the otherwise-refused answers admitted, the claims left
    open by kind, the provisional rows and totals, and the flag check (a committed piece with an unknown overlap
    that lacks its mark or its exception is a defect, listed and counted, never dropped)."""
    cards = [k for x in exc if x.get("class") == KIND for k in x["cards"]]
    flagged = {x["piece"] for x in exc if x.get("class") == KIND}
    pieces = [c for c in claims if is_piece(c) and c["unknown_overlaps"]]
    missing = [{"fact": c["fact"], "line": c["source"],
                "missing": [w for w, ok in (("provisional mark", c.get("provisional") is True and c.get("open_claims")),
                                            ("exception", c["fact"] in flagged)) if not ok]}
               for c in pieces if not (c.get("provisional") is True and c.get("open_claims") and c["fact"] in flagged)]
    log = list(getattr(rel, "r7_log", ()))
    kinds = {k: sum(1 for x in cards if x["kind"] == k) for k in (RELAXED, ALREADY)}
    cases = [x["case"] for x in exc if x.get("class") == KIND and "case" in x]
    ids = [k.get("claim_id") for k in cards]
    return {"r7": True, "basis": steps.R7_BASIS, "scope": SCOPE,
            "paths": paths(log, getattr(rel, "validated", None), getattr(rel, "vote_validated", None)),
            "final": {"note": "what this version commits, whatever path validated it; an attempt admitted at "
                              "validation is not a commitment",
                      "pieces_committed_with_open_claims": len(pieces),
                      "pieces_committed_with_newly_relaxed_claims": sum(
                          1 for c in pieces if any(u.get("kind") == RELAXED for u in c["unknown_overlaps"])),
                      "cases": len(cases), "cards": len(cards),
                      "unique_claims": None if None in ids else len(set(ids)),
                      "lines_affected": sorted({c["source"] for c in pieces})},
            "cases": cases,
            "eligible_attempts": log,
            "admitted_otherwise_refused": [e for e in log if e.get("answer_admitted")],
            "counts": {"eligible_attempts": len(log),
                       "admitted_otherwise_refused": sum(1 for e in log if e.get("answer_admitted")),
                       "open_claims": len(cards), **{f"open_claims, {k}": v for k, v in kinds.items()},
                       "pieces_with_open_claims": len(pieces), "exceptions": len(flagged),
                       "missing_flags": len(missing)},
            "open": [{"kind": k["kind"], "line": k["line"], "piece": k["piece"]["fact"], "rival": k["rival"]["name"],
                      "amount_at_stake": k["amount_at_stake"]} for k in cards],
            "provisional_rows": {r: len(v) for r, v in sorted(rows.items())},
            "provisional_totals": sorted(totals),
            "missing_flags": missing,
            "settled_in_this_run": 0, "note": "every open claim is unresolved by design; " + ACTIONS_NOTE}


def paths(log, validated, vote_validated):
    """The attempt counts of the two paths one spread run validates on, from its attempt log. An attempt is one
    piece of one answer; the same piece tried again is a repeat, so unique pieces are counted apart. On these paths
    the policy is applied to every eligible attempt. Admitted and refused are the answer's end at validation, not a
    commitment. A count the record cannot give is None, never 0."""
    def of(xs, n):
        adm = [e for e in xs if e.get("answer_admitted")]
        return {"pieces_validated": n, "eligible_attempts": len(xs), "r7_applied": len(xs),
                "admitted_at_validation": len(adm), "refused_at_validation": len(xs) - len(adm),
                "unique_pieces_eligible": len({(e["line"], e["fact"]) for e in xs}),
                "unique_pieces_admitted": len({(e["line"], e["fact"]) for e in adm})}
    vote = [e for e in log if e.get("role") == "vote"]
    rest = [e for e in log if e.get("role") != "vote"]
    known = isinstance(validated, int) and isinstance(vote_validated, int)
    return {"placement": of(rest, validated - vote_validated if known else None),
            "vote path": of(vote, vote_validated if known else None)}


def resummary(prev, claims, exc, rows, totals, dropped=()):
    """The open-claims record of a cumulative version: the attempt history is the spread run's, kept as recorded
    (`prev`, the previous version's record); what is committed now, the cases, the counts, the provisional rows and
    totals and the flag check are recomputed from this version's claims and exceptions."""
    log = SimpleNamespace(r7_log=list(prev.get("eligible_attempts") or ()))
    out = summary(log, claims, exc, rows, totals)
    for k in ("paths", "scope"):       # the caller sets the qualification, from this version's outcome
        if k in prev:
            out[k] = prev[k]
        else:
            out.pop(k, None)
    if dropped or prev.get("dropped_by_repair"):
        out["dropped_by_repair"] = list(prev.get("dropped_by_repair") or ()) + list(dropped)
    return out


def qualification(outcome, exc):
    """Why a run with open claims is not "complete", kept apart from every other reason: the flag can turn a
    complete run into a qualified one without changing one allocation."""
    n = sum(1 for x in exc if x.get("class") == KIND)
    return {"outcome": outcome, "open_claim_exceptions": n, "other_exceptions": len(exc) - n,
            "qualified_only_by_open_claims": outcome == "qualified" and n > 0 and n == len(exc),
            "note": "an open claim prevents 'complete'; it changes no allocation"}


def repair_count(tax, f, line, allocations, printed, owned, index, shadow, evidence, period, got):
    """One repair split check, for the scope record of a root-cause source run. `got`: the legacy result the tree received.
    A refused check is run once more under the policy, count only: `eligible` when an overlap refusal is gone or changed
    under it, `would_pass_as_proposed` when the same answer would then pass every check of steps.split. Nothing here
    reaches the tree."""
    errs = list(got[2])
    rec = {"line": line["concept"], "facts": sorted(a["fact_id"] for a in allocations),
           "legacy_refused": bool(errs), "eligible": False, "r7_applied": False}
    if not errs:
        return rec
    pieces, _, e2 = steps.split(tax, f, line, allocations, printed, owned, index, shadow, evidence, period=period,
                                r7_count_only=True)
    relaxed = [e for e in errs if ": overlaps " in e and e not in e2]
    rec.update(eligible=bool(relaxed), legacy_overlap_refusals_relaxed=relaxed, would_pass_as_proposed=not e2,
               r7_errors=list(e2),
               claims=sorted({claim_id(getattr(shadow, "filing", None), p["fact"], r["claim"]["target"])
                              for p in pieces for r in p["review"] if r["basis"] == steps.R7_BASIS}))
    return rec


SCOPE_LOST = ("interrupted", "not run")     # endings after which a started tree may have lost a split check


def repair_scope(log, ended=None, tree_started=None):
    """The scope record of one repair run on a root-cause source: every split check counted, and the refusals the policy would have
    relaxed, with repeats kept apart from unique pieces. The policy was applied to none.
    `ended`, `tree_started` (repair_run.report): the run's recorded outcome and whether its tree started. The
    record then says whether its counts hold: a run that ended by a
    recorded outcome made the checks listed, none when its tree never started; a run that was interrupted, or that
    failed after its tree started, may have lost a check, and its counts are a lower bound."""
    el = [r for r in log if r.get("eligible")]
    would = [r for r in el if r.get("would_pass_as_proposed")]
    uniq = lambda rs: len({(r["line"], tuple(r["facts"])) for r in rs})     # noqa: E731
    return {"scope": SCOPE, "path": "repair", "policy": steps.LEGACY_POLICY,
            "counts": {"split_checks": len(log), "refused": sum(1 for r in log if r["legacy_refused"]),
                       "eligible": len(el), "r7_applied": 0, "admitted_by_r7": 0,
                       "refused_that_would_pass_under_r7_as_proposed": len(would),
                       "unique_answers_eligible": uniq(el), "unique_answers_that_would_pass": uniq(would)},
            "note": "count only: each refused split was checked once more under the cross-dimension policy and the result recorded here; "
                    "the tree received the legacy result",
            **({} if ended is None and tree_started is None else {"run": scope_run(ended, tree_started)}),
            "eligible_checks": el}


def scope_run(ended, tree_started):
    """How the repair run of a scope record ended, and whether its counts hold."""
    known = isinstance(ended, str) and bool(ended) and isinstance(tree_started, bool)
    holds = known and not (ended == "interrupted" or (ended in SCOPE_LOST and tree_started))
    return {"outcome": ended, "tree_started": tree_started, "counts_hold": holds,
            "why": "the run ended by a recorded outcome: the split checks listed are the ones it made" if holds
            else "the run was interrupted, or failed after its tree started, or its ending is not recorded: a "
                 "split check may be missing, and the counts are a lower bound"}
