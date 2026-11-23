"""Totals named by the model, identified from labels and amounts, without tags. One step
before any spread, in place of the removed labels-fix step, whose record (corrections.py) it writes in the same form.

Code finds a section's printed total by the template's tag (the controls' filed and reconstructed concepts). Here the
organizer's model reads each printed statement and, for each total the template needs, names the printed line that is
that total, or "not printed". It sees printed labels, amounts and what the company's calculation adds into each line,
by printed label: no tag, no definition (`leaks` guards every message). Code then acts only where the tag is
missing: the named line must be a total whose parts add up (a verified subtotal), not already another needed total,
of the same period type, claimed once; the organizer's ties must not get worse (corrections.admit). Such a line is
retagged to the template's total in the record (corrections.apply), so every later check reads it unchanged. Where the
company's tag is printed, it is kept and the model's answer is recorded as agreeing or not. The model never gives a
number that enters a calculation. Then code's department split (corrections.splits, no model) runs on the retagged filing."""
import json
import time
from pathlib import Path

from . import corrections as cr, llm, organizer as org, spread, tagless, template
from .filing import Filing

ROOT = Path(__file__).resolve().parents[1]
MODEL = org.MODEL
PROMPT = ROOT / "prompts/named_totals_v1.txt"
NUM_CTX, ANSWER_CAP = 8192, 600
BUDGET = NUM_CTX - ANSWER_CAP
MAX_SENDS, MAX_TOKENS, WALL_S = 60, 500_000, 300       # per company
RECORD, VERSION = cr.RECORD, 2       # 2: the labels-fix step's department split (I-02) after the retags
tokens_used, sha = cr.tokens_used, cr.sha


def needed(t, tax, statement):
    """[(concept, name)]: every total the template's controls of the statement read, filed or a reconstruction's term,
    first occurrence of each, named by the standard label (never the tag)."""
    out = []
    for sid, c in t.controls.items():
        if t.statement_of(sid) != statement:
            continue
        for x in [c["filed"], *[k["concept"] for k in (c.get("reconstructed") or {}).get("terms", [])]]:
            if x and x not in [o for o, _ in out]:
                out.append((x, (tax.concepts.get(x) or {}).get("label") or t.sections[sid]["name"]))
    return out


def message(inv, cands, calc, name, company):
    printed = {l["concept"]: l for l in inv["lines"] if "amount" in l}
    rows = []
    for ln in cands:
        rows.append(f"{ln['id']} \"{ln['label']}\" {cr.money(ln['amount'])}")
        for child, w in calc.get(ln["concept"], []):
            p = printed.get(child)
            rows.append(f"   {'+' if w > 0 else '-'} " + (f"\"{p['label']}\" {cr.money(p['amount'])}" if p
                                                       else "a line not printed on the statement"))
    return "\n".join([f"Statement: {org.NAMES[inv['statement']]} of {company}.",
                      f"The total asked for: {name}.", "Printed total lines:", *rows,
                      "Which printed line is this total?"])


def leaks(f, *texts):
    """The prefixed XBRL names (standard or the filing's own) in text bound for the model: none may be sent."""
    prefixes = tagless.STANDARD | {c.split(":")[0] for c in f.by_concept}
    return sorted({m for x in texts for m in tagless.QNAME.findall(x) if m.split(":")[0] in prefixes})


def schema(cands):
    return {"type": "object", "additionalProperties": False, "required": ["reason", "line"],
            "properties": {"reason": {"type": "string", "maxLength": 400},
                           "line": {"type": "string", "enum": [l["id"] for l in cands] + ["not printed"]}}}


def gate(f, tax, inv, key, ln, concept, needs):
    """Code's gates on a named line for a total whose tag is not printed: (True, "passed") or (False, why)."""
    if ln["concept"] in needs:
        return False, "the line is already another needed total"
    if ln["state"] != "verified subtotal":
        return False, f"its parts do not verifiably add up ({ln['state']})"
    if spread.read(f, concept, key):
        return False, "the total is filed for the period off the statement: kept"
    old_p = (tax.concepts.get(ln["concept"]) or f.extension.get(ln["concept"]) or {}).get("period")
    new_p = (tax.concepts.get(concept) or {}).get("period")
    if not old_p or old_p != new_p:
        return False, f"period type differs or unknown ({old_p} vs {new_p})"
    if not ln["fact_ids"] or any(not x for x in ln["fact_ids"]):
        return False, "a fact has no ID"
    return True, "passed"


def caller(log_path, stop_file=None, deadline=None, max_sends=MAX_SENDS, model=MODEL, **k):
    """A hosted job passes its model and engine settings (spread.borrowed)."""
    return org.Bounded(models={"name": model}, log_path=log_path, max_calls=max_sends, seed=7, num_ctx=NUM_CTX,
                       answer_cap=ANSWER_CAP, options={"presence_penalty": 0.0}, stop_file=stop_file,
                       deadline=deadline, **k)


def run(directory, tax, out, stop_file=None, deadline=None, max_sends=MAX_SENDS, token_cap=MAX_TOKENS, model=MODEL,
        **k):
    """The step for one filing; writes `out`/corrections.json (corrections' record form) and returns it.
    Caps and endings: the step's own caps end it "capped", totals not asked are "not examined";
    the job's STOP or deadline is re-raised."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "named_totals_calls.jsonl"
    step_end = time.monotonic() + WALL_S
    c = caller(str(log), stop_file=stop_file, deadline=min(x for x in (deadline, step_end) if x is not None),
               max_sends=max_sends, model=model, **k)
    f0, t = Filing(directory), template.active()
    company = (f0.by_concept.get("dei:EntityRegistrantName") or [{}])[0].get("value") or str(directory)
    system = PROMPT.read_text(encoding="utf8")
    record = {"version": VERSION, "step": "named totals", "filing": str(directory), "template": Path(t.path).name,
              "template_sha256": sha(t.path), "models": {"name": model}, "prompts": {PROMPT.name: sha(PROMPT)},
              "settings": {"num_ctx": NUM_CTX, "answer_cap": ANSWER_CAP, "budget": BUDGET, "max_sends": max_sends,
                           "max_tokens": token_cap, "wall_s": WALL_S, "seed": 7, "temperature": 0},
              "statements": {}, "named": [], "retags": [], "splits": [], "ending": "complete"}

    def job_ended():
        return (stop_file is not None and Path(stop_file).exists()) or (deadline is not None and time.monotonic() >= deadline)

    claims = []
    for st in t.statements:
        inv = org.inventory(f0, tax, st, company)
        if inv.get("refused"):
            record["statements"][st] = {"refused": inv["refused"]}
            continue
        calc, key = f0.calculation(inv["role"]), tuple(inv["key"])
        cands, needs = cr.totals_of(inv, calc), dict(needed(t, tax, st))
        record["statements"][st] = {"role": inv["role"], "key": inv["key"], "totals": len(cands)}
        tagged = {l["concept"]: l for l in inv["lines"] if "amount" in l}
        for concept, name in needs.items():
            entry = {"statement": st, "total": name, "concept": concept,
                     "tagged_line": tagged[concept]["id"] if concept in tagged else None}
            record["named"].append(entry)
            if record["ending"] != "complete" or not cands:
                entry["outcome"] = "not examined" if cands else "no printed total lines"
                continue
            if time.monotonic() >= step_end or tokens_used(log) >= token_cap:
                record["ending"] = "capped: " + ("wall" if time.monotonic() >= step_end else "tokens")
                entry["outcome"] = "not examined"
                continue
            user = message(inv, cands, calc, name, company)
            if leaks(f0, system, user):
                entry.update(outcome="refused", why="a tag in the message: " + ", ".join(leaks(f0, system, user)))
                continue
            try:
                ans, rec = c.ask("name", system, user, schema(cands), tag=f"name:{st}:{concept}", budget=BUDGET)
            except llm.ContextExceeded as e:
                entry.update(outcome="too large", why=str(e))
                continue
            except llm.RequestFailed as e:
                record["ending"] = f"failed: {e}"
                entry.update(outcome="call failed", why=str(e))
                continue
            except llm.CapReached as e:
                if job_ended() or str(c.failed or "").startswith("ledger") or str(e).startswith("ledger"):
                    raise
                record["ending"], entry["outcome"] = f"capped: {e}", "not examined"
                continue
            if job_ended():
                raise llm.Stopped("the job ended during the call: its answer is not used")
            if not ans:
                entry.update(outcome="call failed", why=rec.get("error"))
                continue
            entry.update(answer=ans)
            named = ans["line"]
            if concept in tagged:
                entry["outcome"] = "agrees" if named == tagged[concept]["id"] else \
                    f"disagrees: the model says {named}; the company's line kept"
                continue
            if named == "not printed":
                entry["outcome"] = "agrees: not printed"
                continue
            ln = next(l for l in cands if l["id"] == named)
            ok, why = gate(f0, tax, inv, key, ln, concept, needs)
            entry.update(line=named, label=ln["label"], gate=why)
            if not ok:
                entry["outcome"] = "refused"
                continue
            claims.append((st, inv, ln, concept, entry))
    for st, inv, ln, concept, entry in claims:
        rivals = [e for s, _, l, _, e in claims if s == st and l["id"] == ln["id"]]
        if len(rivals) > 1:
            entry.update(outcome="refused", why=f"the line is named for {len(rivals)} totals")
            continue
        retag = {"statement": st, "role": inv["role"], "key": inv["key"], "line": ln["id"], "label": ln["label"],
                 "old": ln["concept"], "new": concept, "fact_ids": ln["fact_ids"]}
        ok, checks, why = cr.admit(directory, tax, record, inv, retag)
        entry.update(admission=checks, admission_why=why, outcome="retagged" if ok else "refused")
        record["retags"].append({**retag, "applied": ok, "why": why, "checks": checks})
    if job_ended():
        raise llm.Stopped("the job ended before the record was written")
    record["splits"] = cr.splits(directory, tax, record, company)
    record["sends"], record["tokens"] = c.calls, tokens_used(log)
    (out / RECORD).write_text(json.dumps(record, indent=1), encoding="utf8")
    return record
