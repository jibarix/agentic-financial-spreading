"""The organizer: a model sorts the printed items of one statement into the template's sections, everywhere (option 1)
or only where code's route fails (option 2), and code checks each section against its printed total before any row
placement.

Code builds one frozen inventory per statement (`inventory`): every printed occurrence with its state, each item's
amount, sign evidence and code's route, and the statement's checks with their targets. Everything after it reads only
that JSON-ready record: `section_tie` and `ties` compare a section assignment with the checks, `failures` gives option
2's rules, the prompt builders word code's findings as observations, `sort_statement` makes the one sorting call and
at most one repair call and records the end state. The model returns a section per asked line and a short reason,
never a number."""
import hashlib
import json
import os
import re
import time
from decimal import Decimal
from pathlib import Path

from . import calc_network as cn
from . import llm, spread, tagless, template
from .filing import tag_of

ROOT = Path(__file__).resolve().parents[1]
PROMPT = ROOT / "prompts/organizer_v5.txt"     # without the XBRL concept: tags hidden (tagless.py); v5 adds NONE (J4)
MODEL = "qwen3.5:4b-q4_K_M"
NUM_CTX, ANSWER_CAP = 16384, 4096
BUDGET = NUM_CTX - ANSWER_CAP           # 12,288 input tokens, by estimate() before and the engine's count after
MAX_SENDS = 4                           # per statement: one sorting and one repair call, two sends each
CANT = "can't tell"
NONE = "none of these sections"     # the line belongs to no section offered, with its reason
NONE_CAUSE = "organizer: none of these sections"
REASON_CHARS = 160
ITEMS = ("routed", "queued", "incomplete subtotal", "extraction refused")
BLANK = ("organizer can't tell", NONE_CAUSE, "citation-invalid", "answer invalid", "statement unsorted (context)",
         "statement refused")
# a statement-level failure's kind, by the failure `ask` or the check gives
KINDS = {"cut": "structural", "invalid": "structural", "context": "context", "request": "request",
         "citation coverage": "citation coverage"}
# the stopping rule counts these kinds only: a balance sheet with no line sorted ("all blank") is a placement refusal,
# not one of them
COUNTED = tuple(dict.fromkeys(KINDS.values()))
# a memo section as the model reads it: its rows sit outside every template total; its lines can be printed anywhere,
# inside printed totals included (Template.memo). Read as "outside its totals" with no row listed, a memo such as
# the D&A memo was never filled
MEMO_NAME = ("{name}, a memo section: the lines printed under its heading, read for its memo rows only; they can sit "
             "inside the statement's printed totals, and the memo rows count in no template total")
RULES = {1: "no section", 2: "a heading and a calculation reach to another section",
         3: "no heading and two or more calculation reaches", 4: "the statement has no routed item"}


def failed(refusal):
    """Whether a statement's recorded refusal is an organizer failure the stopping rule counts."""
    return (refusal or {}).get("refused_at") == "organizer" and refusal.get("kind") in COUNTED


# ---- the inventory -------------------------------------------------------------------------------------------------

def descendants(nets, concept):
    """Every concept below `concept` in any calculation network of the role (usable relationships)."""
    out = set()
    for net in nets["networks"].values():
        todo = [concept]
        while todo:
            for c, _ in net["kids"].get(todo.pop(), []):
                if c not in out:
                    out.add(c)
                    todo.append(c)
    return out


def parents(nets, concept):
    return sorted({p for net in nets["networks"].values() for p, kids in net["kids"].items()
                   if any(c == concept for c, _ in kids)})


def eligibility(t, sid, sections):
    """(covered sections, None) when the control's expression reads every item row of each section it covers at +1
    (item rows no control or formula reads, Template.unread_items, are outside every total); else (covered, why). Total and derived rows are not
    components."""
    k = t.coefficients(t.controls[sid]["compare"])
    secs = {t.section_of(r) for r in k}
    covered = [s for s in sections if s in secs] + sorted(secs - set(sections))
    for s in covered:
        items = {r["row_id"] for r in t.data["rows"] if r["section"] == s and r["line_type"] == "item"} - t.unread_items()
        read = {r for r in k if t.section_of(r) == s}
        missing = sorted(items - read)
        odd = sorted(r for r in read if k[r] != 1)
        if missing or odd:
            return covered, ("not evaluated before placement: partial-section control ("
                             + "; ".join(([f"leaves out {', '.join(missing)}"] if missing else [])
                                         + ([f"reads {', '.join(odd)} at another coefficient"] if odd else [])) + ")")
    return covered, None


def first_line(lines, concept):
    return next((l for l in lines if l["concept"] == concept), None)


def filed_target(f, role, key, concept, lines, canonical, tax, direction):
    """The filed total of a check: printed in the role with a fact at the key, duplicates agreeing; else unavailable
    with why."""
    ln = first_line(lines, concept)
    if ln is None:
        return {"state": "unavailable", "why": f"total not printed ({concept})"}
    xs = spread.read(f, concept, key)
    if not xs:
        return {"state": "unavailable", "why": f"total {ln['id']} has no fact for the period", "line": ln["id"]}
    if spread.disagree(xs):
        return {"state": "unavailable", "why": f"total {ln['id']}: duplicate facts disagree", "line": ln["id"]}
    x = spread.chosen(xs)
    return {"state": "filed", "line": ln["id"], "amount": x["value"], "unit": x["currency"],
            "facts": [[canonical.get(x["id"], x["id"]), 1, x["decimals"]]],
            "sign": spread.sign(spread.balance_of(tax, f, concept), direction) or 1}


def target_of(f, tax, role, key, t, sid, lines, canonical, direction):
    c = t.controls[sid]
    if c["filed"]:
        got = filed_target(f, role, key, c["filed"], lines, canonical, tax, direction)
        if got["state"] == "filed" or sid not in t.derived_totals():
            return got
    if sid in t.derived_totals():
        r = spread.resolve_control(f, role, sid, key, canonical)
        if r.get("refused"):
            return {"state": "unavailable", "why": f"reconstruction refused: {r['refused']}"}
        d = t.derived_totals()[sid]
        fx, dec = r["reconstructed"]["facts"], r["reconstructed"]["decimals"]
        return {"state": "reconstructed", "amount": str(r["amount"]), "unit": r["unit"],
                "lines": [first_line(lines, d["plus"])["id"], first_line(lines, d["minus"])["id"]],
                "facts": [[canonical.get(fx[d["plus"]], fx[d["plus"]]), 1, dec[d["plus"]]],
                          [canonical.get(fx[d["minus"]], fx[d["minus"]]), -1, dec[d["minus"]]]],
                "sign": spread.sign(spread.balance_of(tax, f, d["plus"]), direction) or 1,
                "formula": r["reconstructed"]["formula"]}
    return {"state": "unavailable", "why": "no total in the template"}


def code_finding(sec, route, calc, sections):
    """Code's route of one item, as recorded fields: section (None when none, or when the route names a section the
    run does not take), heading, reaches [[section, anchor]], path count, route type (as in the route count)."""
    reaches = sorted({(s, a) for s, a, _ in calc})
    out = {"section": sec if sec in sections else None, "routed_to": sec, "basis": (route or {}).get("basis"),
           "heading": (route or {}).get("heading"), "memo": bool((route or {}).get("memo")),
           "memo_heading_set_aside": (route or {}).get("memo_heading_set_aside"),
           "reaches": [list(x) for x in reaches], "paths": len(calc)}
    if sec is not None and sec not in sections:
        out["why_none"] = f"routed to {sec}, not a section of this run"
    elif sec is None:
        out["why_none"] = "no listed heading and no calculation path" if not calc else \
            f"no listed heading and {len(calc)} calculation paths"
    return out


def inventory(f, tax, statement, company=None):
    """The frozen inventory of one statement, JSON-ready: {statement, refused | key, role, variant, sections,
    section_info, lines, checks}. A setup refusal (C2 no role, C7 not indirect) gives {statement, refused}."""
    t = template.active()
    try:
        key, role, variant = spread.setup(f, statement)
    except RuntimeError as e:
        return {"statement": statement, "company": company, "refused": str(e)}
    from .steps import Eligibility
    canonical = Eligibility(tax, f, {"fact": None}).canonical
    sections = t.run_sections(statement, variant)
    cov = spread.coverage(tax, f, role, key, statement, canonical)["occurrences"]
    printed = f.printed(role)
    headings = t.heading_to_section()
    memo = {s for s in set(headings.values()) if t.memo(s)}
    reach, _ = spread.calc_routes(f, role)
    nets = cn.networks(f, role)
    controls = t.statement_controls(statement)
    lines, seen, firsts = [], set(), []
    for i, (r, e) in enumerate(zip(printed, cov)):
        c = r["concept"]
        facts = spread.read(f, c, key)
        ln = {"id": f"L{i + 1}", "concept": c, "label": r["label"], "depth": r["depth"],
              "path": r["path"], "state": e["state"], "fact_ids": sorted(x["id"] for x in facts),
              "extension": tag_of(c) in f.extension, "calc_parents": parents(nets, c)}
        if e.get("reason"):
            ln["reason"] = e["reason"]
        if e["state"] == "printed by member only":
            # code's structural route, or None when unknown; its amount is never one fact
            ln["route_section"] = e.get("section")
        if e["state"] == "second occurrence":
            ln["first"] = next(l["id"] for l in lines if l["concept"] == c and l["fact_ids"])
        if facts and not spread.disagree(facts):
            x = spread.chosen(facts)
            ln.update(fact=x["id"], cfact=canonical.get(x["id"], x["id"]), amount=x["value"], unit=x["currency"],
                      decimals=x["decimals"])
        # the statement-wide first occurrences with amounts, as coverage collects them
        first = e["state"] in ITEMS + ("verified subtotal",) or e["state"] == "statement control" and c not in seen
        if first and "fact" in ln:
            firsts.append({"concept": c, "fact": ln["fact"], "amount": Decimal(ln["amount"]),
                           "decimals": ln["decimals"], "unit": ln["unit"], "line": ln})
        if e["state"] in ITEMS:
            sec, route, calc = spread.route_of(r, headings, reach, memo)
            ln["code"] = code_finding(sec, route, calc, sections)
            ln["code"]["type"] = (route_type(sec, route, calc) if ln["code"]["section"] else
                                  8 if sec is not None else 7 if statement == "BS" else 6)
        if facts:
            seen.add(c)
        lines.append(ln)
    # statement-wide decompositions: a routed subtotal its printed parts verify is context, one they do not is an
    # incomplete subtotal (coverage extended to routed lines)
    copies = [dict(x) for x in firsts]
    spread.mark_controls(f, role, copies, [], canonical, controls, key)
    for x in copies:
        ln = x["line"]
        if ln["state"] == "routed" and x["decomposition"] == "verified":
            ln["state"] = "verified subtotal"
        elif ln["state"] == "routed" and x["decomposition"] == "unavailable":
            ln["state"] = "incomplete subtotal"
            ln["reason"] = f"subtotal not covered by its printed parts: {x['reason']}"
    for ln in lines:
        ln["kind"] = "item" if ln["state"] in ITEMS else "context"
        if ln["kind"] == "context":
            ln.pop("code", None)
        if ln["state"] == "extraction refused":
            for k in ("fact", "cfact", "amount", "unit", "decimals"):
                ln.pop(k, None)
    items = [l for l in lines if l["kind"] == "item"]
    by_concept = {l["concept"]: l for l in items}
    for ln in items:
        if ln["state"] == "incomplete subtotal":
            below = descendants(nets, ln["concept"])
            ln["details"] = sorted({by_concept[c]["id"] for c in below if c in by_concept and c != ln["concept"]}
                                   | {l["id"] for l in items if ln["concept"] in l["path"]}, key=lambda s: int(s[1:]))
        if statement == "BS":
            ln["balance"] = spread.balance_of(tax, f, ln["concept"])
    if statement != "BS":
        got = spread.path_signs(tax, f, role, key, statement,
                                [{"concept": l["concept"], "unit": l["unit"]} for l in items if "amount" in l])
        for ln in items:
            if ln["concept"] in got:
                s = got[ln["concept"]]
                ln["sign"] = {"d": s["d"], "basis": s["basis"], "why": s.get("why") or s.get("why_fallback")}
    normal = {sid: t.sections[sid]["normal_balance"] for sid in sections}
    checks = []
    for sid, c in t.controls.items():
        if t.statement_of(sid) != statement:
            continue
        local = t.local(sid)
        if local and sid not in sections:
            continue
        covered, why = eligibility(t, sid, sections)
        if not set(covered) <= set(sections):
            continue
        # the check's direction: the section's normal balance; BS.TOT has none and takes its filed total's balance
        direction = t.sections[sid]["normal_balance"] or spread.balance_of(tax, f, c["filed"])
        chk = {"id": f"{'local' if local else 'span'}:{sid}", "kind": "local" if local else "span", "control": sid,
               "covers": covered, "direction": direction}
        if why:
            chk.update(eligible=False, why=why)
        else:
            chk.update(eligible=True, target=target_of(f, tax, role, key, t, sid, lines, canonical, direction))
            if not local and chk["target"]["state"] == "reconstructed":
                chk["target"] = {"state": "unavailable", "why": "a span is evaluated on its filed total only"}
        # a line printed by member only has no single amount to sum: a check covering its structural section, or
        # every check of the statement when its section is unknown, is unavailable
        member = next((l for l in lines if l["state"] == "printed by member only"
                       and (l["route_section"] is None or l["route_section"] in covered)), None)
        if member and chk["eligible"] and chk["target"]["state"] != "unavailable":
            chk["target"] = {"state": "unavailable", "why": f"printed by member only: line {member['id']}"
                             + ("" if member["route_section"] else ", section unknown")}
        checks.append(chk)
    unread = t.unread_items()
    info = {sid: {"name": t.sections[sid]["name"], "normal": normal[sid], "memo": t.memo(sid),
                  "rows": [r["name"] for r in t.data["rows"] if r["section"] == sid and r["row_id"] not in unread
                           and r["line_type"] == ("supplementary" if t.memo(sid) else "item")]}
            for sid in sections}
    return {"statement": statement, "company": company, "refused": None, "key": list(key), "role": role,
            "variant": variant, "sections": sections, "section_info": info, "lines": lines, "checks": checks}


def route_type(sec, route, calc):
    """Types 1-5 of a routed occurrence, by its source and routing evidence."""
    if route["basis"] == "calculation":
        return 5
    secs = {s for s, _, _ in calc}
    return 1 if not secs else 2 if secs == {sec} else 3 if sec not in secs else 4


def items_of(inv):
    return [l for l in inv["lines"] if l["kind"] == "item"]


# ---- the tie before placement ----------------------------------------------------------------------------------------

def item_sign(inv, ln, section, direction):
    """(sign or None, basis, why) of one component: balance sheet, its balance against the assigned section's normal
    balance, then that normal against the check's direction; income statement and cash flow, path_signs' d, where a
    cash flow sign by balance alone is unavailable."""
    if inv["statement"] == "BS":
        normal = inv["section_info"][section]["normal"]
        s = spread.sign(ln.get("balance"), normal)
        if s is None:
            return None, "balance", f"sign unknown: line {ln['id']}"
        return s * (spread.sign(normal, direction) or 1), "balance", None
    g = ln.get("sign") or {}
    if g.get("d") is None:
        return None, g.get("basis"), f"sign unknown: line {ln['id']}" + (f" ({g['why']})" if g.get("why") else "")
    if inv["statement"] == "CF" and g["basis"] == "balance":
        return None, "balance", f"sign by balance only: line {ln['id']}"
    return g["d"], g["basis"], None


def section_tie(inv, check, assignment):
    """One check against an assignment (item ID -> section or None): {check, state (tied, differs, unavailable,
    not evaluated), why, difference, tolerance, components, signs}. The same for every arm and for repair."""
    out = {"check": check["id"], "covers": check["covers"]}
    if not check["eligible"]:
        return {**out, "state": "not evaluated", "why": check["why"]}
    tg = check["target"]
    if tg["state"] == "unavailable":
        return {**out, "state": "unavailable", "why": tg["why"]}
    comps = [l for l in items_of(inv) if assignment.get(l["id"]) in check["covers"]]
    out["components"] = [l["id"] for l in comps]
    ids = set(out["components"])
    for l in comps:
        if "amount" not in l:
            return {**out, "state": "unavailable", "why": f"amount refused: line {l['id']}"}
    for l in comps:
        if l["state"] == "incomplete subtotal" and ids & set(l.get("details", [])):
            return {**out, "state": "unavailable", "why": f"overlap: subtotal {l['id']} with its details "
                                                         f"{', '.join(sorted(ids & set(l['details'])))}"}
    if len({l["unit"] for l in comps} | {tg["unit"]}) > 1:
        return {**out, "state": "unavailable", "why": "mixed units"}
    signs, built, total = {}, [], Decimal(0)
    for l in comps:
        s, basis, why = item_sign(inv, l, assignment[l["id"]], check["direction"])
        if s is None:
            return {**out, "state": "unavailable", "why": why}
        signs[l["id"]] = [s, basis]
        built.append((l["cfact"], s, l["decimals"]))
        total += s * Decimal(l["amount"])
    target = tg["sign"] * Decimal(tg["amount"])
    tol = spread.tolerance(built, [(x, tg["sign"] * c, d) for x, c, d in tg["facts"]], {})
    diff = total - target
    return {**out, "state": "tied" if abs(diff) <= tol else "differs", "difference": str(diff),
            "tolerance": str(tol), "sum": str(total), "target": str(target), "signs": signs}


def ties(inv, assignment):
    return {c["id"]: section_tie(inv, c, assignment) for c in inv["checks"]}


def to_repair(inv, assignment, tied):
    """(item IDs a differing check sends to repair, notes): a local check its section's items; a span the items of its
    covered sections with no tied local check. Blank items are never sent."""
    local = {c["control"]: tied[c["id"]]["state"] for c in inv["checks"] if c["kind"] == "local"}
    send, notes = set(), []
    for c in inv["checks"]:
        if tied[c["id"]]["state"] != "differs":
            continue
        secs = c["covers"] if c["kind"] == "local" else [s for s in c["covers"] if local.get(s) != "tied"]
        got = {i for i, s in assignment.items() if s in secs}
        if c["kind"] == "span" and not secs:
            notes.append(f"{c['id']} differs while every covered section ties: nothing sent")
        send |= got
    return sorted(send, key=lambda s: int(s[1:])), notes


def section_states(inv, tied):
    """Per section: checked (a tied or differing check covers it, its own or a span) or unchecked with why."""
    out = {}
    for s in inv["sections"]:
        mine = [tied[c["id"]] for c in inv["checks"] if s in c["covers"]]
        live = [x for x in mine if x["state"] in ("tied", "differs")]
        if inv["section_info"][s]["memo"]:
            out[s] = {"checked": False, "why": "not applicable: memo section, outside every total"}
        elif live:
            out[s] = {"checked": True, "by": [x["check"] for x in live],
                      "tied": all(x["state"] == "tied" for x in live)}
        else:
            out[s] = {"checked": False, "why": "; ".join(f"{x['check']} {x['state']}" for x in mine)
                      or "not applicable: no check covers the section"}
    return out


def unseen(inv, assignment, tied):
    """Lines a tie cannot see, on the final assignment: zero amounts, amounts within their section's tolerance, and
    equal amounts in two tied sections (a swap keeps both ties)."""
    states = section_states(inv, tied)
    tol = {}
    for c in inv["checks"]:
        x = tied[c["id"]]
        if "tolerance" in x:
            for s in c["covers"]:
                tol[s] = min(tol.get(s, Decimal(x["tolerance"])), Decimal(x["tolerance"]))
    out = {"zero": [], "within tolerance": [], "equal amounts in two tied sections": []}
    placed = [(l, assignment.get(l["id"])) for l in items_of(inv) if assignment.get(l["id"]) and "amount" in l]
    for l, s in placed:
        a = abs(Decimal(l["amount"]))
        if a == 0:
            out["zero"].append(l["id"])
        elif s in tol and a <= tol[s]:
            out["within tolerance"].append(l["id"])
    # equal amounts in two tied sections: flagged only when swapping the pair keeps every tied check tied, through the
    # same tie (signs follow the assigned section, so equal amounts alone do not hide a swap)
    tied_secs = {s for s, v in states.items() if v.get("checked") and v.get("tied")}
    was = [k for k, v in tied.items() if v["state"] == "tied"]
    swaps = set()
    for i, (l, s) in enumerate(placed):
        for m, t2 in placed[i + 1:]:
            if s != t2 and s in tied_secs and t2 in tied_secs \
                    and abs(Decimal(m["amount"])) == abs(Decimal(l["amount"])) != 0:
                now = ties(inv, {**assignment, l["id"]: t2, m["id"]: s})
                if all(now[k]["state"] == "tied" for k in was):
                    swaps |= {l["id"], m["id"]}
    out["equal amounts in two tied sections"] = [l["id"] for l, _ in placed if l["id"] in swaps]
    return out


# ---- option 2's rules ------------------------------------------------------------------------------------------------

def failures(inv):
    """Item ID -> (rule number, words) for the items option 2 sends to the organizer."""
    items = items_of(inv)
    if not any(l["code"]["section"] for l in items):
        return {l["id"]: (4, RULES[4]) for l in items}
    out = {}
    for l in items:
        c = l["code"]
        if not c["section"]:
            out[l["id"]] = (3, RULES[3]) if not c["heading"] and c["paths"] >= 2 else (1, RULES[1])
        elif c["heading"] and any(s != c["section"] for s, _ in c["reaches"]):
            out[l["id"]] = (2, RULES[2])
    return out


# ---- what the model sees -------------------------------------------------------------------------------------------

def amount_text(ln):
    if ln["kind"] == "item" and "amount" not in ln:
        return "amount refused (its duplicate facts disagree)"
    if "amount" not in ln:
        return ""
    return f"{Decimal(ln['amount']):,} {ln['unit']}"


def where(inv, concept):
    ln = first_line(inv["lines"], concept)
    return f"{ln['id']} '{ln['label']}'" if ln else f"'{tagless.ref(concept)}' (not printed)"


def heading_line(inv, ln):
    """The nearest occurrence above `ln` of its route's heading."""
    i = int(ln["id"][1:]) - 1
    return next((x for x in reversed(inv["lines"][:i]) if x["concept"] == ln["code"]["heading"]), None)


def findings(inv, ln):
    """Code's findings on one item, worded as observations."""
    c = ln["code"]
    out = []
    if c["heading"]:
        h = heading_line(inv, ln)
        out.append((f"the nearest listed heading above is {h['id']} '{h['label']}'" if h else
                    f"the nearest listed heading above is '{tagless.ref(c['heading'])}'") + f", a heading of {c['routed_to']}"
                   + ("" if c["section"] else " (not a section of this statement)"))
        if c["memo"]:
            out.append("that heading is a memo heading")
    elif not c["reaches"]:
        out.append("no listed heading above and no calculation path")
    if c.get("memo_heading_set_aside"):
        out.append("a memo heading above was set aside because a calculation path reaches it")
    if c["reaches"]:
        n = c["paths"]
        out.append(("one calculation path reaches it" if n == 1 else f"{n} calculation paths reach it") + " from "
                   + ", ".join(f"{where(inv, a)} (counted toward {s})" for s, a in c["reaches"]))
    if ln["state"] == "incomplete subtotal":
        out.append("it is a subtotal its printed parts do not add up to")
    return "; ".join(out)


def check_words(inv, chk):
    """A check as the model reads it: which printed total, or why none is used."""
    tg = chk.get("target") or {}
    if not chk["eligible"]:
        return "not checked before placement (its printed total leaves out part of the section)"
    if tg["state"] == "filed":
        ln = first_line_by_id(inv, tg["line"])
        return f"the printed total {ln['id']} '{ln['label']}'"
    if tg["state"] == "reconstructed":
        a, b = (first_line_by_id(inv, i) for i in tg["lines"])
        return f"{a['id']} '{a['label']}' less {b['id']} '{b['label']}'"
    why = tg["why"]
    return "no total available (" + ("the total is not printed" if why.startswith("total not printed") else
                                     "a line is printed only by class or member" if why.startswith("printed by member")
                                     else why) + ")"


def first_line_by_id(inv, i):
    return inv["lines"][int(i[1:]) - 1]


def sections_text(inv):
    out = []
    for s in inv["sections"]:
        x = inv["section_info"][s]
        mine = [check_words(inv, c) for c in inv["checks"] if s in c["covers"]]
        name = MEMO_NAME.format(name=x["name"]) if x["memo"] else x["name"]
        out.append(f"- {s}: {name}. Rows: {'; '.join(x['rows'])}."
                   + (f" Totals that include it: {' | '.join(mine)}." if mine else " No printed total checks it."))
    out.append(f"- {NONE}: the line belongs to none of the sections above; the reason cites the printed text that "
               "excludes it.")
    out.append(f"- {CANT}: the printed statement does not tell which section.")
    return "\n".join(out)


def statement_text(inv, mode, assignment=None, asked=()):
    """The printed statement in order, one line per occurrence. `mode`: findings shown for every item ("option1"),
    for the asked items with code's section on the others ("option2"), or none ("hidden"); with `assignment`, each
    item's current section (repair)."""
    asked = set(asked)
    out = []
    for ln in inv["lines"]:
        pad = "  " * ln["depth"]
        bits = [f"{ln['id']} {pad}{ln['label']}"]
        amt = amount_text(ln)
        if amt:
            bits.append(amt)
        if ln["kind"] == "context":
            bits.append({"structure": "heading", "statement control": "printed total",
                         "verified subtotal": "subtotal of the lines printed under it",
                         "excluded": ("a balance at a date, not a flow (not sorted)" if template.format_at_least(5) else
                                      "beginning or ending cash (not sorted)"),
                         "second occurrence": f"printed again (first at {ln.get('first')}; not sorted)"}
                        .get(ln["state"], f"not sorted: {ln['state']}"))
        else:
            # the filing's own field, in every arm: its calculation parent (no concept or extension mark, tags being
            # hidden); code's route findings follow per arm
            bits.append("calculation parent: " + ("; ".join(where(inv, p) for p in ln["calc_parents"])
                                                  if ln["calc_parents"] else "none"))
            if assignment is not None:
                bits.append(f"now in {assignment.get(ln['id']) or CANT}")
            elif mode == "option2" and ln["id"] not in asked:
                bits.append(f"sorted by code into {ln['code']['section']}")
            elif ln["id"] in asked:
                bits.append("TO SORT")
            if mode in ("option1", "option2"):
                fd = findings(inv, ln)
                if fd:
                    bits.append(f"code found: {fd}")
        out.append(" | ".join(bits))
    return "\n".join(out)


NAMES = {"BS": "balance sheet", "IS": "income statement", "CF": "cash flow statement"}


def sort_message(inv, mode, asked):
    return (f"Statement: the {NAMES[inv['statement']]}"
            + (f" of {inv['company']}" if inv.get("company") else "") + ".\n\n"
            f"Sections (answer with the ID):\n{sections_text(inv)}\n\n"
            f"Printed statement, in printed order (line ID, label indented by depth, amount, notes):\n"
            f"{statement_text(inv, mode, asked=asked)}\n\n"
            f"Lines to sort ({len(asked)}): {', '.join(asked)}\n"
            "Answer once for each line to sort, in the order listed.")


def repair_message(inv, mode, assignment, tied, asked):
    diffs = []
    for c in inv["checks"]:
        x = tied[c["id"]]
        if x["state"] == "differs":
            diffs.append(f"- {check_words(inv, c)}: the lines now in {', '.join(c['covers'])} sum to "
                         f"{Decimal(x['sum']):,}, the total is {Decimal(x['target']):,}, a difference of "
                         f"{Decimal(x['difference']):,}.")
    others = [f"- {', '.join(c['covers'])}, {check_words(inv, c)}: "
              + {"tied": "adds up", "unavailable": "not checked", "not evaluated": "not checked"}[tied[c["id"]]["state"]]
              for c in inv["checks"] if tied[c["id"]]["state"] != "differs"]
    return (f"Statement: the {NAMES[inv['statement']]}" + (f" of {inv['company']}" if inv.get("company") else "") +
            ".\n\nAfter sorting, these totals do not add up (sums use each line's sign):\n" + "\n".join(diffs) +
            ("\n\nOther checks:\n" + "\n".join(others) if others else "") +
            f"\n\nSections (answer with the ID):\n{sections_text(inv)}\n\n"
            f"Printed statement, in printed order, with each line's current section:\n"
            f"{statement_text(inv, mode, assignment=assignment)}\n\n"
            f"Lines to look at again ({len(asked)}): {', '.join(asked)}\n"
            "Answer once for each of these lines, in the order listed: keep its section or move it.")


def schema(asked, sections):
    item = {"type": "object", "properties": {
        "line": {"type": "string", "enum": list(asked)},
        "reason": {"type": "string", "minLength": 1, "maxLength": REASON_CHARS},
        "section": {"type": "string", "enum": list(sections) + [NONE, CANT]}},
        "required": ["line", "reason", "section"], "additionalProperties": False}
    return {"type": "object", "properties": {"lines": {"type": "array", "items": item, "minItems": len(asked),
                                                        "maxItems": len(asked)}},
            "required": ["lines"], "additionalProperties": False}


def blank_cause(answer):
    """The recorded blank cause of an abstention: "organizer can't tell" or NONE_CAUSE (J4)."""
    return NONE_CAUSE if answer == NONE else "organizer can't tell"


def validate(inv, answer, asked):
    """(structural problems, citation problems {line: why}) code finds that the schema cannot. Structural: an asked
    line missing or repeated, a line not asked. Per item: a reason that cites a
    line ID that does not exist, or no existing line ID and no printed label (case and spacing aside); a can't tell
    answer is blank anyway and is not checked."""
    got = [x["line"] for x in answer["lines"]]
    ids = {l["id"] for l in inv["lines"]}
    norm = lambda s: " ".join(s.casefold().split())
    labels = {norm(l["label"]) for l in inv["lines"] if l["label"].strip()}
    bad = [f"line {i} answered {got.count(i)} times" for i in sorted(set(got)) if got.count(i) > 1]
    bad += [f"line {i} not answered" for i in asked if i not in got]
    bad += [f"line {i} was not asked" for i in sorted(set(got) - set(asked))]
    cites = {}
    for x in answer["lines"]:
        cited = re.findall(r"\bL\d+\b", x["reason"])
        gone = [c for c in cited if c not in ids]
        if x["section"] == CANT:
            continue
        if gone:
            cites[x["line"]] = f"its reason cites {', '.join(gone)}, which does not exist"
        elif not cited and not any(b in norm(x["reason"]) for b in labels):
            cites[x["line"]] = "its reason cites no line ID and no printed label"
    return bad, cites


# ---- the calls -------------------------------------------------------------------------------------------------------

def ask(caller, system, user, sch, tag):
    """(answer or None, record, failure or None): failure is "context" (over the budget or the server's context),
    "cut", "invalid" or "request" (a timeout or an engine failure after llm.py's one retry), with why. A caller
    stopped by the time the call returns (STOP or the deadline, a reply or a timeout included) raises Stopped: nothing
    it returned is used. A cap (the ledger's included) raises CapReached: it ends the run, as every cap does."""
    def halt():
        if getattr(caller, "stopped", None) and caller.stopped():
            raise llm.Stopped("stop switch or deadline when the call returned; its answer is not used")
    try:
        ans, rec = caller.ask("organizer", system, user, sch, tag=tag, budget=BUDGET)
    except llm.ContextExceeded as e:
        halt()
        return None, {"error": str(e)}, "context"
    except llm.CapReached:      # Stopped included
        raise
    except llm.RequestFailed as e:
        halt()
        return None, {"error": str(e)}, "request"
    halt()
    if ans is None:
        return None, rec, "cut" if "output cut" in (rec.get("error") or "") else "invalid"
    return ans, rec, None


def call_record(rec, failure, est):
    load = (rec.get("engine_usage") or {}).get("load_duration")     # model loading, apart from the analyst's
    return {"failure": failure, "error": rec.get("error"), "estimated_tokens": est,
            "prompt_tokens": rec.get("prompt_tokens"), "answer_tokens": rec.get("answer_tokens"),
            "done_reason": rec.get("done_reason"), "seconds": rec.get("seconds"),
            "load_seconds": load / 1e9 if load is not None else None}


def sort_statement(inv, mode, caller, system):
    """One statement through one arm. `mode`: "option1", "option2" or "hidden". Returns {assignment (item ID ->
    section or None), blank (item ID -> cause), citation_invalid (item ID -> {section, why}), asked, calls,
    ties_initial, repair, ties, sections, unseen, refused, end}.

    The admission policy: a validated answer keeps its valid items; an item whose reason fails the citation check is
    blank, "citation-invalid", its proposed section and why kept in `citation_invalid` (a citation-coverage refusal's
    too, the sort call recording that failure). The statement is refused, `refused` {kind, reason}, on a structural
    failure (an ID missing, repeated or not asked; a cut or unparseable answer), over the context or budget, on a
    request failure, or by the citation-coverage refusal: citation-invalid items of the initial sort, counted once, more
    than half of the asked items (a policy heuristic, not a measured threshold). A refused statement gets no repair
    call. A repair that fails keeps the prior assignment and never refuses; its citation-invalid items keep their
    section before the repair."""
    out = {"statement": inv["statement"], "mode": mode, "calls": [], "repair": None, "refused": None,
           "citation_invalid": {}}
    if inv.get("refused"):
        return {**out, "end": "statement refused", "why": inv["refused"], "assignment": {}, "blank": {}}
    items = items_of(inv)
    sections = inv["sections"]
    fails = failures(inv) if mode == "option2" else {}
    if mode == "option2":
        assign = {l["id"]: l["code"]["section"] for l in items if l["id"] not in fails}
        asked = [l["id"] for l in items if l["id"] in fails]
        out["failures"] = {i: list(v) for i, v in fails.items()}
    else:
        assign, asked = {}, [l["id"] for l in items]
    out["asked"] = asked
    blank = {}
    if asked:
        user, sch = sort_message(inv, mode, asked), schema(asked, sections)
        est = llm.estimate(system, user, sch)
        ans, rec, failure = ask(caller, system, user, sch, f"sort {inv['statement']}")
        cites = {}
        if ans is not None:
            bad, cites = validate(inv, ans, asked)
            if bad:
                failure, rec = "invalid", {**rec, "error": "; ".join(bad)}
            else:       # filled before the threshold
                out["citation_invalid"] = {x["line"]: {"section": x["section"], "why": cites[x["line"]]}
                                           for x in ans["lines"] if x["line"] in cites}
                if 2 * len(cites) > len(asked):
                    failure, rec = "citation coverage", {**rec, "error": f"{len(cites)} of {len(asked)} asked lines "
                                                                         "are citation-invalid, more than half"}
        out["calls"].append({"kind": "sort", **call_record(rec, failure, est), "asked": len(asked),
                             "answer": ans})
        if failure:
            cause = "statement unsorted (context)" if failure == "context" else "answer invalid"
            for i in asked:
                assign[i], blank[i] = None, "citation-invalid" if i in out["citation_invalid"] else cause
            out["end_sort"] = cause
            out["refused"] = {"kind": KINDS[failure], "reason": rec.get("error")}
        else:
            for x in ans["lines"]:
                assign[x["line"]] = None if x["section"] in (CANT, NONE) or x["line"] in cites else x["section"]
                if x["line"] in cites:
                    blank[x["line"]] = "citation-invalid"
                elif x["section"] in (CANT, NONE):
                    blank[x["line"]] = blank_cause(x["section"])
    out["ties_initial"] = tied = ties(inv, assign)
    send, notes = to_repair(inv, assign, tied) if not out["refused"] else ([], [])
    out["repair_notes"] = notes
    if send:
        user, sch = repair_message(inv, mode, assign, tied, send), schema(send, sections)
        est = llm.estimate(system, user, sch)
        ans, rec, failure = ask(caller, system, user, sch, f"repair {inv['statement']}")
        cites = {}
        if ans is not None:
            bad, cites = validate(inv, ans, send)
            if bad:
                failure, rec = "invalid", {**rec, "error": "; ".join(bad)}
        out["calls"].append({"kind": "repair", **call_record(rec, failure, est), "asked": len(send), "answer": ans})
        rep = {"sent": send, "moves": [], "citation_invalid": cites}
        if failure:
            rep.update(outcome="repair failed", why=f"{failure}: {rec.get('error')}")
        else:
            after = dict(assign)
            for x in ans["lines"]:
                if x["line"] not in cites:
                    after[x["line"]] = None if x["section"] in (CANT, NONE) else x["section"]
            rep["moves"] = [{"line": i, "from": assign.get(i), "to": after[i]} for i in send
                            if after[i] != assign.get(i)]
            now = ties(inv, after)
            broken = [k for k, v in tied.items() if v["state"] == "tied" and now[k]["state"] != "tied"]
            if broken:
                rep.update(outcome="repair rejected", why=f"a tied check is no longer tied: {', '.join(broken)}",
                           ties_rejected=now)
            else:
                rep["outcome"] = "repair accepted"
                said = {x["line"]: x["section"] for x in ans["lines"] if x["line"] not in cites}
                for i in send:
                    if after[i] is None:
                        blank[i] = blank_cause(said.get(i, CANT)) if i in said else blank.get(i, "organizer can't tell")
                    else:
                        blank.pop(i, None)
                assign, tied = after, now
                rep["unresolved"] = [k for k, v in now.items() if v["state"] == "differs"]
        out["repair"] = rep
    out.update(assignment=assign, blank=blank, ties=tied, sections=section_states(inv, tied),
               unseen=unseen(inv, assign, tied))
    unchecked = {s for s, v in out["sections"].items() if not v["checked"]}
    out["unchecked_lines"] = sorted((i for i, s in assign.items() if s in unchecked), key=lambda s: int(s[1:]))
    out["end"] = ("refused" if out["refused"] else
                  "incomplete" if blank or any(v["state"] == "differs" for v in tied.values()) else "complete")
    return out


class Bounded(llm.Caller):
    """A caller that reads the run's STOP file and its deadline (time.monotonic) itself, before every send, and
    waits for a reply no longer than the time left. The wait is urlopen's socket timeout: nothing interrupts a read
    already under way, so a reply can still arrive after the deadline; `ask` then refuses it. In a spread the
    deadline is also the ledger's: the nearer of the batch and company deadlines, as the ledger gave it before the
    send."""

    def __init__(self, *a, stop_file=None, deadline=None, **k):
        super().__init__(*a, **k)
        self.stop_file, self.deadline = stop_file, deadline

    def stopped(self):
        late = self._left is not None and self.clock() - self._left[1] >= self._left[0]
        return (super().stopped() or late or (self.stop_file is not None and Path(self.stop_file).exists())
                or (self.deadline is not None and time.monotonic() >= self.deadline))

    def reserve_attempt(self, role, reserved):
        rid, timeout = super().reserve_attempt(role, reserved)
        if self.deadline is not None:
            left = self.deadline - time.monotonic()
            if left <= 0:
                raise llm.Stopped("deadline")
            timeout = min(timeout, left)
        return rid, timeout


def caller(log_path, stop_file=None, deadline=None, model=MODEL, **k):
    """A fresh caller for one statement: the organizer's model, thinking off, temperature 0, seed 7, at most four
    sends, stopped by `stop_file` or at `deadline`. In a spread, `k` passes the run's ledger, company and stop switch:
    every send, a failed one included, is charged there and logged in the run's calls.jsonl. A hosted run passes its
    model and engine settings (spread.borrowed)."""
    return Bounded(models={"organizer": model}, log_path=log_path, max_calls=MAX_SENDS, seed=7, num_ctx=NUM_CTX,
                   answer_cap=ANSWER_CAP, options={"presence_penalty": 0.0}, stop_file=stop_file, deadline=deadline,
                   **k)


def system_prompt():
    return PROMPT.read_text(encoding="utf8")


# ---- the frozen record ---------------------------------------------------------------------------------------------

RECORD = "organizer.json"
SETTINGS = {"think": False, "temperature": 0, "seed": 7, "num_ctx": NUM_CTX, "answer_cap": ANSWER_CAP,
            "budget": BUDGET, "max_sends": MAX_SENDS, "presence_penalty": 0.0}
HASHED = ("code", "prompt", "schema", "inventory", "template", "filing", "request")
RESULT = ("assignment", "blank", "citation_invalid", "calls", "ties", "sections", "unseen", "end")  # what readers read


class Refused(RuntimeError):
    """A replay the organizer's record does not allow, with the reason."""


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def role(identity=None, model=MODEL):
    """The organizer's own role in run.json: model, digest (from model_identity; None offline), settings, and the
    hashes of its prompt and of its code, which builds each statement's schema (that schema's hash is in the
    statement's record)."""
    return {"model": model, "digest": ((identity or {}).get("digests") or {}).get(model), "settings": SETTINGS,
            "prompt": spread.sha(PROMPT), "code": spread.sha(Path(__file__))}


def hashes(inv, filing, model_digest, corrections=None, model=MODEL):
    """What a record depends on: the organizer's code, prompt, initial schema, inventory, template and filing, and the
    initial request as sent, its messages and schema with model and settings beside them. A run whose run.json names
    a corrections record (corrections.open_filing) adds that record's sealed sha256."""
    from .edgar import filing_inputs
    asked = [l["id"] for l in items_of(inv)]
    sch = schema(asked, inv["sections"])
    return {"code": spread.sha(Path(__file__)), "prompt": spread.sha(PROMPT), "schema": digest(sch),
            "inventory": digest(inv), "template": spread.sha(template.active().path),
            "filing": digest(filing_inputs(Path(filing))),
            "request": digest({"messages": [system_prompt(), sort_message(inv, "option1", asked)], "schema": sch,
                               "model": model, "settings": SETTINGS}),
            "model": model, "digest": model_digest, "settings": SETTINGS,
            **({"corrections": corrections["sha256"]} if corrections else {})}


def save(d, inv, res):
    """The statement's record beside the run in folder `d` (its run.json names the run, filing and organizer role),
    written once and atomically (a temporary file replaced into place, so an interrupted write leaves none), and
    read back: the run goes on from exactly what replay will read."""
    meta = json.loads((Path(d) / "run.json").read_text(encoding="utf8"))
    rec = {"run": meta["run_id"], "company": Path(meta["filing"]).parent.name, "statement": inv["statement"],
           "filing": meta["filing"],
           "hashes": hashes(inv, meta["filing"], meta["organizer"]["digest"], meta.get("corrections"),
                            meta["organizer"]["model"]),
           "inventory": inv, "sort": res}
    p = Path(d) / RECORD
    tmp = p.with_name(RECORD + ".tmp")
    tmp.write_text(json.dumps(rec, indent=1, default=str), encoding="utf8")
    os.replace(tmp, p)
    return json.loads(p.read_text(encoding="utf8"))


def load(d, f, tax, statement):
    """The verified record of the run in folder `d` (a copy of the run included), read with no model call. Refused
    first for a run made with tags shown; None for a tags-hidden run from before the organizer (its run.json names
    no organizer role): the historical path, route_of. Refused, with the reason: a record missing, truncated or corrupt,
    without a result field (RESULT) or a hash, of a refused statement, from another run, company, statement or filing,
    made with tags shown, whose code, prompt, schema, template, filing, rebuilt inventory, request, model or settings do
    not match, whose result does not account for its items exactly (result_problems), or whose ties do not recompute."""
    d = Path(d)
    meta = json.loads((d / "run.json").read_text(encoding="utf8"))
    if meta.get("tags_hidden") is not True:
        raise Refused(f"run {d.name} was made with tags shown; tags-shown runs are no longer replayed")
    if "organizer" not in meta:
        return None
    try:
        rec = json.loads((d / RECORD).read_text(encoding="utf8"))
    except (OSError, ValueError) as e:
        raise Refused(f"organizer record of {d.name} unreadable: {e}") from None
    srt, h = rec.get("sort") or {}, rec.get("hashes") or {}
    if srt.get("refused"):
        raise Refused(f"{statement} was refused by the organizer ({srt['refused']['kind']}): nothing to replay")
    gone = [k for k in RESULT if k not in srt] + [k for k in HASHED if not h.get(k)]
    if gone or "inventory" not in rec:
        raise Refused(f"organizer record of {d.name} lacks {', '.join(gone or ['its inventory'])}")
    want = {"run": meta["run_id"], "company": Path(meta["filing"]).parent.name, "statement": statement,
            "filing": meta["filing"]}
    wrong = [k for k, v in want.items() if rec.get(k) != v]
    if digest(rec["inventory"]) != h["inventory"]:
        wrong.append("inventory as recorded")
    sealed = (meta.get("corrections") or {}).get("sha256")
    if getattr(f, "corrections_sha256", None) != sealed:    # the filing opened with another record, or none
        wrong.append("corrections record of the filing")
    hidden = not tagless.ON     # the run's inventory and request as sent, with tags hidden (a parent's replay)
    if hidden:
        tagless.start(f)
    try:
        inv = inventory(f, tax, statement, rec["inventory"].get("company"))
        now = hashes(inv, meta["filing"], meta["organizer"]["digest"], meta.get("corrections"),
                     meta["organizer"]["model"])
    finally:
        if hidden:
            tagless.stop()
    wrong += [k for k in [*now, *({"corrections"} - set(now))] if h.get(k) != now.get(k)]
    if wrong:
        raise Refused(f"organizer record of {d.name} does not match: {', '.join(wrong)}")
    bad = result_problems(inv, srt)
    if bad:
        raise Refused(f"organizer record of {d.name}: {'; '.join(bad)}")
    if digest(ties(inv, srt["assignment"])) != digest(srt["ties"]):
        raise Refused(f"organizer record of {d.name}: the ties recomputed on replay differ from the frozen ones")
    return rec


def result_problems(inv, srt):
    """What a frozen sort gets wrong against its inventory: every item
    assigned once, none missing and no other ID, to a section of the statement or to none; a blank exactly where the
    assignment is none, with a known cause."""
    items, a, b = [l["id"] for l in items_of(inv)], srt["assignment"], srt["blank"]
    out = [f"item {i} has no assignment entry" for i in items if i not in a]
    out += [f"{i} is assigned but is not an item" for i in a if i not in items]
    out += [f"{i} is assigned to {s}, not a section of the statement" for i, s in a.items()
            if s is not None and s not in inv["sections"]]
    out += [f"{i}: blank and assignment disagree" for i in list(a) + [i for i in b if i not in a]
            if (i in a and a[i] is None) != (i in b)]
    return out + [f"{i}: unknown blank cause {c!r}" for i, c in b.items() if c not in BLANK]


def by_concept(rec):
    """The record's assignment as prepare reads it: item concept -> section, or None for a blank. An item with no
    entry is a KeyError, never a blank."""
    return {l["concept"]: rec["sort"]["assignment"][l["id"]] for l in rec["inventory"]["lines"]
            if l["kind"] == "item"}
