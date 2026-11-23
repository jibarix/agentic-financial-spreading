"""C7 repair as one command: load the draft a spread run left, and repair it with the real role wrappers
(repair_roles) under the batch's shared ledger.

    .venv-agents\\Scripts\\python.exe -m spreading.repair_run runs/<spread run> \\
        --max-calls 400 --max-tokens 2000000

The draft is rebuilt from the spread run's own record (run.json, lines.json, placements.json, exceptions.json) and
its filing: signs from the filing's taxonomy as spread.run sets them, facts by ID (repair.draft_of), and each line
left unresolved with the reasons the spread recorded. Code then decides only what needs no model: a root that fails its
own checks (repair.draft_errors) is recorded as a bad root, and a draft that already passes C6 as needing no repair,
both with no model call. Any other draft goes to repair.repair. Without --review, C8's slot is unavailable, so a repair
that reaches commit is withdrawn and the original draft kept.

Every model request goes through the run's Ledger (or a batch's, --ledger) and the call's meter; the stop switch is
STOP in the run folder (or the project's STOP). The run folder is new and dated; no file in it is overwritten
(runrec.write). Files: run.json, draft_in.json, c7_tree.json (when the tree ran), draft_out.json (when a node was
committed), exceptions.json, outcome.json, calls.jsonl and ledger/.

One section of a whole-statement run:

    .venv-agents\\Scripts\\python.exe -m spreading.repair_run runs/<statement run> --section CF.INV \\
        --max-calls 400 --max-tokens 2000000 [--dry]

reads that section from the run (section_draft): its lines, the run's own signs and period, its recorded splits,
its C6 settings, and every amount the run's other sections own. Before any call, code rebuilds the section with no
move applied and stops it (outcome "not run") when the recomputed failures or root total differ from the run's
(root check), when the rebuilt rows, claims, checks or unplaced lines differ from the run's, or when
no edge sequence within the depth can reach eligibility (structurally ineligible: its section total not found, or more lines unplaced
than the depth, or lines outside the section total, which no registry edit changes). A section that passes has
frozen.json written (models and their digests, num_ctx, the command, the effective budget and the section's results)
before the first dispatch. `--dry` stops there: no tree, no model, no network. Files added: section.json, frozen.json.

A --section run is a component test: no C8 review and no review reserve; the tree runs under the given limits
(--max-calls, --max-tokens, --wall-s), and a selected candidate that passes the commit re-check is recorded "selected,
unreviewed", with the run's draft unchanged. A repair run on a whole draft keeps C8: a candidate is committed only when
C8 settles every changed line. A KeyboardInterrupt keeps the partial tree, the outcome "interrupted" and the exceptions
in the run record, then is re-raised.

`--review` (with --section) runs the section with C8: the source run's checker re-solves each changed line blind
(repair_review.Review), built and checked before frozen.json, whose `review` holds its frozen configuration
(--review-model: another local model answers in the checker's place, its digest read and recorded); the review reserve
(repair_review.line_ceiling per line, times the depth) sits inside --max-calls, --max-tokens and --wall-s. Only C8
settled on every changed line commits ("repaired", with draft_out.json); anything else keeps the draft ("not repaired").
File added: c8_review.json (the configuration and every review, its checker's record included).

`--hosted-profile flash --spend-usd D` runs the tree with mixed engines: generator, controller and meaning on the
profile's hosted analyst (Qwen3.8 Flash, reasoning off), the critic local (hosted_tree); a --review of a hosted source
run re-solves on that run's hosted checker (repair_review: GLM at effort low). D is the section's dollar allocation: the
run ledger's dollar limit and each hosted Caller's own spend cap. Every Caller is fresh (one per role call, one per
reviewed line), so its own cap resets; the frozen allocations are held in ledgers instead: the run's own ledger holds
the section's (--max-calls, --max-tokens, --spend-usd) and each hosted role's (`--role-limit ROLE CALLS USD`, required
for generator, controller and meaning under a profile, and for the checker when the review's checker is hosted), and the
company ledger (`company`) the job's ceiling and its shared role sub-caps. Every reservation, retry and unknown-usage
settlement counts in each of them, so a new Caller cannot restore a spent role budget; a role's limit refuses that role
(the tree ends cap_reached, the refusal recorded as unsent). A profile with a placeholder (providers, price or allowance
unset), a malformed setting (repair_review.hosted_role_errors), no per-role reasoning or a hosted role without its
allocation is refused before the run folder is made; the identity check reads local models only.
"""
import argparse
import hashlib
import json
import traceback
from decimal import Decimal
from pathlib import Path

from . import llm, repair, tagless, template
from .packs import ROOT, load_rows, section_rows, statement_rows
from .ledger import company_delta, sent_by_role
from .runrec import new_run, write
from .spread import BORROWED, HOSTED_PROFILES, balance_of, hosted_use, openrouter_kwargs, per_role, sign, taxonomy_year
from .taxonomy import edition

HOSTED_TREE_ROLES = ("generator", "controller", "meaning", "critic")   # each on its profile entry (spread.BORROWED)


def load_spread(folder):
    """A spread run's record: its settings, lines (amounts and parts' weights as Decimal), placements, exceptions and
    checks. Files missing from the run are an error: a draft is never built from a partial record."""
    d = Path(folder)
    read = lambda n: json.loads((d / n).read_text(encoding="utf8"))    # noqa: E731
    lines = read("lines.json")
    for l in lines:
        l["amount"] = Decimal(str(l["amount"]))
        l["parts"] = [(c, Decimal(str(w))) for c, w in l.get("parts", [])]
    return {"dir": str(d), "run": read("run.json"), "lines": lines, "placements": read("placements.json"),
            "exceptions": read("exceptions.json"), "checks": read("checks.json")}


def unresolved_of(lines, placements, exceptions):
    """Each line that is neither a control nor placed, with the reasons the spread recorded for it, in order."""
    out = {}
    for l in lines:
        c = l["concept"]
        if repair.fixed_line(l) or c in placements:
            continue
        why = [f"{x['step']} {x['class']}: {x.get('why')}" for x in exceptions if x.get("concept") == c]
        out[c] = "; ".join(why) or "not placed"
    return out


def signs_of(lines, tax, f, section):
    """As spread.run: the sign from the concept's balance against the section's; a control with none gets 1, which
    is noted (a placed line with none is caught by repair.draft_errors)."""
    balance = template.active().sections[section]["normal_balance"]
    signs, notes = {}, []
    for l in lines:
        signs[l["concept"]] = sign(balance_of(tax, f, l["concept"]), balance)
        if l["control"] and signs[l["concept"]] is None:
            signs[l["concept"]] = 1
            notes.append({"step": "C3", "class": "mapping", "concept": l["concept"], "why": "total's sign unknown"})
    return signs, notes


def build_draft(spread, f, tax, section, ids):
    """The root draft and the notes made while building it (no model)."""
    signs, notes = signs_of(spread["lines"], tax, f, section)
    placements = {c: {k: v for k, v in p.items() if k in ("row", "by")} for c, p in spread["placements"].items()}
    why = unresolved_of(spread["lines"], placements, spread["exceptions"])
    return repair.draft_of(f, section, ids, spread["lines"], signs, placements, why), notes


def section_signs(spread, lines, tax, f, section, instant):
    """The run's own signs. Income statement and cash flows: lines.json's sign (C3a, the calculation path
    the run used), a queued line's from the run's queue (diagnostics.json), never the balance type. Balance sheet:
    the balance against the section's, which is how spread.prepare sets them on an instant (lines.json has none)."""
    if instant:
        return signs_of(lines, tax, f, section)
    queued = {q["line"]["concept"]: q["sign"]["d"] for q in spread["diagnostics"].get("queued") or []}
    return {l["concept"]: l["sign"]["d"] if "sign" in l else queued.get(l["concept"]) for l in lines}, []


def section_draft(spread, f, tax, section, ids):
    """(draft, notes): one section of a whole-statement run. Lines are those the run put in the
    section; the period is theirs; the splits are the run's claims of the section on another fact than their line's,
    kept as recorded; every amount of another section (its lines, and its pieces) is owned elsewhere; the C6
    settings are the run's for the section (spread.run: canonical facts, filed_absent, runwide). A section whose
    total the run reconstructed is refused: that control is not rebuilt here."""
    from .spread import read
    from .steps import Eligibility
    t = template.active()
    lines = [l for l in spread["lines"] if l.get("section") == section]
    if not lines:
        raise ValueError(f"{section}: no line of the run is in this section")
    key = tuple(lines[0]["period"])
    instant = key[0] == "instant"
    if (spread["diagnostics"].get("controls") or {}).get(section):
        raise ValueError(f"{section}: the run reconstructed its total; that control is not rebuilt here")
    signs, notes = section_signs(spread, lines, tax, f, section, instant)
    line_of = {l["concept"]: l for l in spread["lines"]}
    piece = lambda c: c["fact"] != line_of[c["source"]]["fact"]      # noqa: E731
    pieces = [{**{k: c[k] for k in ("fact", "concept", "tag", "dims", "row", "source", "by") if k in c},
               "amount": Decimal(str(c["amount"])), "r": c.get("r", 1), "decimals": c.get("decimals"),
               **({"r_basis": c["r_basis"]} if "r_basis" in c else {})}
              for c in spread["claims"] if c["section"] == section and piece(c)]
    date = repair.key_period(key)
    elsewhere = [{"concept": l["concept"], "dims": l.get("dims") or [], **({"tag": l["tag"]} if "tag" in l else {}),
                  "date": date, "unit": l["unit"],
                  "section": l.get("section") or "none (a queued line not committed)"}
                 for l in spread["lines"] if l.get("section") != section]
    elsewhere += [{"concept": c["concept"], "dims": c.get("dims") or [], **({"tag": c["tag"]} if "tag" in c else {}),
                   "date": date,
                   "unit": line_of[c["source"]]["unit"], "section": c["section"]}
                  for c in spread["claims"] if c["section"] != section and piece(c)]
    name = t.filed_total(section)
    c6_args = {"control": None, "canonical": Eligibility(tax, f, {"fact": None}).canonical,
               "filed_absent": not instant and bool(name) and not read(f, name, key),
               "runwide": not (instant or (t.local(section) and bool(name)))}
    mine = {l["concept"] for l in lines}
    alloc = {c: {k: v for k, v in p.items() if k in ("row", "by")} for c, p in spread["placements"].items()
             if c in mine}
    why = unresolved_of(lines, alloc, spread["exceptions"])
    return repair.draft_of(f, section, ids, lines, signs, alloc, why, period=list(key), pieces=pieces,
                           elsewhere=elsewhere, c6_args=c6_args), notes


def section_check(draft, spread, section):
    """The section checks, with no move applied and no model. Root check: the recomputed failures and the root
    total ("rows add to": filed, built, gap, tolerance) equal the run's. Reconstruction: the rebuilt rows of the section,
    its claims (fact, row, amount, signed) and every check equal the run's record. Unplaced lines: the root's unresolved
    lines are exactly the run's unplaced lines of the section. Structurally ineligible: the section total not
    found, more lines unplaced than the depth, or lines outside the section total: C6 lists them from the lines and the
    total's calculation alone, no move, split or combined mark changes either, and "rows add to" fails while any is
    listed. No reference label is read."""
    j = lambda o: json.dumps(repair.plain(o), sort_keys=True)     # noqa: E731
    rows, claims, checks = repair.rebuild(draft, draft["registry"])
    ran = [{k: v for k, v in c.items() if k != "section"} for c in spread["checks"] if c.get("section") == section]
    total = lambda cs: next(({k: j(c.get(k)) for k in ("filed", "built", "gap", "tolerance")}      # noqa: E731
                             for c in cs if c["check"].startswith("rows add to")), None)
    failed = lambda cs: sorted(c["check"] for c in cs if not c["ok"])                                # noqa: E731
    root = {"failures": failed(checks), "run_failures": failed(ran), "total": total(checks), "run_total": total(ran)}
    root["same"] = root["failures"] == root["run_failures"] and root["total"] == root["run_total"]
    claim = lambda c: [c["fact"], c["row"], str(c["amount"]), str(c["signed"])]                   # noqa: E731
    got = {"rows": {r: str(rows[r]) for r in draft["ids"]},
           "claims": sorted(claim(c) for c in claims), "checks": sorted(j(c) for c in checks)}
    want = {"rows": {r: str(spread["rows"].get(r)) for r in draft["ids"]},
            "claims": sorted(claim(c) for c in spread["claims"] if c["section"] == section),
            "checks": sorted(j(c) for c in ran)}
    differs = {k: {"rebuilt": got[k], "run": want[k]} for k in got if got[k] != want[k]}
    allocated = next((c for c in ran if c["check"] == "every line allocated"), {})
    unplaced = {"root": sorted(draft["registry"]["unresolved"]), "run": sorted(allocated.get("lines", [])),
                "reasons": draft["registry"]["unresolved"]}
    unplaced["same"] = unplaced["root"] == unplaced["run"]
    why = []
    if "section total found" in root["failures"]:
        why.append("its section total was not found; no move supplies it")
    if len(unplaced["root"]) > repair.DEPTH:
        why.append(f"{len(unplaced['root'])} unplaced lines, more than the depth of {repair.DEPTH} edits")
    outside = sorted({x for c in checks for x in c.get("lines_outside_the_total") or []})
    if outside:
        why.append(f"lines outside the section total ({', '.join(outside)}): no registry edit changes them, so "
                   "'rows add to' cannot pass")
    return {"root": root, "reconstruction": {"same": not differs, "differs": differs, "rows": len(got["rows"]),
                                             "claims": len(got["claims"]), "checks": len(got["checks"])},
            "unplaced": unplaced, "structural": {"ineligible": bool(why), "why": why,
                                                 "lines_outside_the_total": outside},
            "passed": root["same"] and not differs and unplaced["same"]}


def tools_of(f, draft, search=None, tax=None, statements_only=False):
    """The analyst's tools for the section, as spread.run builds them; the line given is only the currency the
    amounts are shown in (the section's lines share one, checked by C6) and the run's period. `tax`: the filing's
    us-gaap edition, which offers printed_page. `statements_only`: the source spread run's setting (its run.json),
    so the tree reads what the spread's roles read (tools.Tools statements_only)."""
    from .tools import Tools
    rows = load_rows()
    by = statement_rows(rows, template.active().statement_of(draft["section"]))
    line = next(l for l in draft["lines"] if not repair.fixed_line(l))
    if draft.get("period"):
        line = {**line, "period": draft["period"]}
    t = Tools(f, line, by, draft["section"], {r["row_id"]: r["name"] for r in rows}, search=search, tax=tax,
              statements_only=statements_only)
    t.lines = {l["concept"]: l for l in draft["lines"]}     # the tree's printed_page names its line
    return t


def hosted_tree(a, roles):
    """(engines, caller_kwargs, settings) of the tree's `roles` under --hosted-profile: each tree role on its profile
    entry (spread.BORROWED: proposer and meaning reader on the analyst's, controller on the checker's, critic on the
    tiebreaker's, else the analyst's) with its reasoning, pinned providers, price ceiling, omitted fields and max_tokens
    (the answer cap plus its allowance), each Caller capped at --spend-usd (the role allocations are the run ledger's
    role limits, shared by every Caller). Without a profile: all local, no settings."""
    if not a.hosted_profile:
        return "ollama", {}, None
    p = HOSTED_PROFILES[a.hosted_profile]
    v = {r: p.get(BORROWED[r]) or p["analyst"] for r in HOSTED_TREE_ROLES}
    return ({r: "openrouter" if r in HOSTED_TREE_ROLES else "ollama" for r in roles},
            openrouter_kwargs(v, a.answer_cap, a.spend_usd),
            {"roles": {r: {**x, "max_tokens": a.answer_cap + x["allowance"]} for r, x in v.items()}})


def prompt_hashes(statements_only=False):
    """The tree's prompts, by name and sha256; with `statements_only`, the statements-only versions the roles use
    (repair_roles.STATEMENTS_PROMPTS)."""
    from .repair_roles import GENERATOR_PLAIN, PROMPTS, STATEMENTS_PROMPTS
    names = [*PROMPTS.values(), GENERATOR_PLAIN]
    if statements_only:
        names = [STATEMENTS_PROMPTS.get(n, n) for n in names]
    return {n: hashlib.sha256((ROOT / "prompts" / f"{n}.txt").read_bytes()).hexdigest() for n in names}


def main(argv=None, filing=None, taxonomy=None, tools=None, identity=None, company=None):
    """`filing`, `taxonomy`, `tools`: fakes for the offline tests (the transport is faked at urllib); a real run reads
    the filing the spread run names, its taxonomy version and the analyst's tools over it. `identity`: models ->
    {ollama_version, digests}, by default llm.model_identity, read only by a --section run that is not --dry, before its
    first dispatch. `company`: the company job's ledger (a ledger.Ledger the job opened), held with the run's
    own as ledger.Both, so every attempt counts in the two and the nearer limit refuses first; the run's own ledger is
    its scope (reconciliation), the company's change over the run is recorded as `company_delta` (ledger.company_delta).
    Not with --ledger."""
    from .runrec import StopFile
    from .ledger import Both, Ledger, check_role_limits
    from .repair_review import hosted_role_errors
    from .llm import NUM_PREDICT
    from .repair_roles import MODELS, NUM_CTX
    ap = argparse.ArgumentParser()
    ap.add_argument("spread_run", help="the spread run folder whose draft failed C6")
    ap.add_argument("--filing", help="the filing folder, if not the one the spread run names")
    ap.add_argument("--max-calls", type=int, required=True, help="most model requests, embeddings included")
    ap.add_argument("--max-tokens", type=int, required=True, help="most local tokens, embeddings included")
    ap.add_argument("--wall-s", type=float, default=3600.0)
    ap.add_argument("--ledger", help="a batch's ledger folder, shared with other runs (default: this run's own)")
    ap.add_argument("--arm", default="own_loop", choices=sorted(repair.REVIEW_PER_LINE))
    ap.add_argument("--dispatches", type=int, default=2, help="requests per question: the Caller's retry included")
    ap.add_argument("--model", action="append", default=[], metavar="ROLE=MODEL", help="a run's model for a role")
    ap.add_argument("--num-ctx", type=int, default=NUM_CTX)
    ap.add_argument("--answer-cap", type=int, default=NUM_PREDICT)
    ap.add_argument("--role-num-ctx", action="append", default=[], metavar="ROLE=N",
                    help="a tree role's own context instead of --num-ctx (e.g. meaning=16384); repeat for more roles")
    ap.add_argument("--context-net", action="store_true", help="a step too large for the context: mask the oldest "
                                                              "result and ask for the final action")
    ap.add_argument("--search", action="store_true", help="offer search_notes and read_passage to the tools")
    ap.add_argument("--cache", help="the embedding cache folder (default data/derived/search)")
    ap.add_argument("--probe", help="why this run: a test or diagnostic run is labelled, never evidence")
    ap.add_argument("--section", help="one section of a whole-statement spread run")
    ap.add_argument("--dry", action="store_true", help="with --section: rebuild and check the section, then stop "
                                                       "before any model, tree or network")
    ap.add_argument("--review", action="store_true", help="with --section: C8 reviews each changed line, blind, as "
                                                          "the spread run's checker; only agreement commits")
    ap.add_argument("--review-model", metavar="MODEL", help="with --review: the review on this local model instead of "
                    "the spread run's checker, given the checker's line, rows, tools and settings")
    ap.add_argument("--hosted-profile", choices=sorted(HOSTED_PROFILES),
                    help="generator, controller and meaning on this profile's hosted analyst; the critic local")
    ap.add_argument("--spend-usd", type=float, default=0.0,
                    help="the run's dollar limit (OpenRouter): needed by --hosted-profile and a hosted source's review")
    ap.add_argument("--critic-feedback", choices=("on", "off"), default="on",
                    help="the critic's verdict and reason to the generator expanding that branch (default on); off "
                         "logs it only, the critic still called")
    ap.add_argument("--role-limit", nargs=3, action="append", default=[], metavar=("ROLE", "CALLS", "USD"),
                    help="a role's frozen allocation in the run's ledger, shared by every Caller of the role (each "
                         "hosted role needs one); repeat for more roles")
    a = ap.parse_args(argv)
    if a.dry and not a.section:
        ap.error("--dry needs --section")
    if a.review and not a.section:
        ap.error("--review needs --section")
    if a.review_model is not None and (not a.review or not a.review_model.strip()):
        ap.error("--review-model needs --review and a model name")
    if company is not None and a.ledger:
        ap.error("--ledger beside the company ledger: one shared ledger at most")
    given = dict(m.split("=", 1) for m in a.model)
    try:
        a.role_limits = {r: {"calls": int(n), "spend_usd": float(usd)} for r, n, usd in a.role_limit}
        check_role_limits(a.role_limits)
    except ValueError as e:
        ap.error(f"--role-limit ROLE CALLS USD: {e}")
    others = sorted(set(a.role_limits) - set(MODELS) - {"checker"})
    if len(a.role_limits) < len(a.role_limit) or others:
        ap.error(f"--role-limit: once per role, of {', '.join([*MODELS, 'checker'])}"
                 + (f"; not {others}" if others else ""))
    if a.role_limits and a.ledger:
        ap.error("--role-limit is held in the run's own ledger, not beside --ledger")
    if a.hosted_profile:        # as spread.main: nothing starts until the profile is filled and per role
        p = HOSTED_PROFILES[a.hosted_profile]
        unset = sorted(f"{r}.{k}" for r, v in p.items() for k in ("providers", "price", "allowance")
                       if k in v and v[k] is None)
        if not per_role(p):
            ap.error(f"--hosted-profile {a.hosted_profile} has no per-role reasoning settings; the tree needs them")
        if unset:
            ap.error(f"--hosted-profile {a.hosted_profile} has placeholders {unset}: fill in the providers, "
                     "prices and allowances first")
        bad = [w for r, v in p.items() for w in hosted_role_errors(r, v)]
        if bad:
            ap.error(f"--hosted-profile {a.hosted_profile}: {'; '.join(bad)}")
        missing = [r for r in HOSTED_TREE_ROLES if r not in a.role_limits]
        if missing:
            ap.error(f"--hosted-profile needs each hosted role's frozen allocation: --role-limit for "
                     f"{', '.join(missing)}")
        if not a.spend_usd > 0:
            ap.error("--hosted-profile needs --spend-usd above 0")
        if set(given) & set(HOSTED_TREE_ROLES):
            ap.error(f"--model: {', '.join(sorted(set(given) & set(HOSTED_TREE_ROLES)))} run on the profile's models")
    engines, hosted_kwargs, hosted = hosted_tree(a, MODELS)
    models = {**MODELS, **({r: v["model"] for r, v in hosted["roles"].items()} if hosted else {}), **given}
    unknown = sorted(set(models) - set(MODELS))
    if unknown:
        ap.error(f"--model: no role {', '.join(unknown)}; roles: {', '.join(MODELS)}")
    limits = {"dispatches": a.max_calls, "local_tokens": a.max_tokens, "spend_usd": a.spend_usd,
              "batch_wall_s": a.wall_s, "company_wall_s": a.wall_s}
    src = Path(a.spread_run) / "run.json"       # the source's settings: the tree reads what its roles read
    statements_only = src.is_file() and bool(json.loads(src.read_text(encoding="utf8")).get("statements_only"))
    if statements_only and a.search:
        ap.error("--search: the source spread run read the statements only (statements_only); no note search")
    meta = {"spread_run": a.spread_run, "models": models, "review_model": a.review_model or "the source run's checker", "arm": a.arm, "dispatches": a.dispatches,
            "limits": limits, "role_limits": a.role_limits, "ledger": a.ledger,
            "company_ledger": company and str(company.dir),
            "num_ctx": a.num_ctx, "role_num_ctx": role_ctx(a), "answer_cap": a.answer_cap,
            "num_batch": llm.batch_pins(m for r, m in models.items() if engines == "ollama" or engines[r] == "ollama"),
            "context_net": a.context_net, "search": a.search, "probe": a.probe,
            "engine": engines, "hosted_profile": a.hosted_profile, "hosted": hosted, "spend_usd": a.spend_usd,
            "evidence": a.probe is None and not a.dry,
            "section": a.section, "dry": a.dry,
            "beam": repair.BEAM, "children": repair.CHILDREN, "depth": repair.DEPTH,
            "c8": ((f"{a.review_model}, with the spread run's checker's prompt, rows, tools and settings,"
                    if a.review_model else "the spread run's checker") + " re-solves each changed line blind; only "
                   "agreement commits, from a review reserve inside the limits"
                   if a.review else
                   "not used: component test, no review reserve; a selected, re-checked candidate is recorded "
                   "unreviewed and the draft kept" if a.section else
                   "unavailable: the C8 review wrapper is not built; a repair that reaches commit is withdrawn"),
            **({"statements_only": True} if statements_only else {}),
            "prompts": prompt_hashes(statements_only), "plan": "the repair tree"}
    d = new_run("repair", meta)
    ledger = Ledger(Path(a.ledger) if a.ledger else d / "ledger", limits, role_limits=a.role_limits or None)
    if company is not None:
        ledger = Both(ledger, company)
    stop = StopFile(d / "STOP")
    out = {"outcome": None, "exceptions": [], "error": None, "draft": None, "result": None, "notes": [],
           "roles": None, "root": None, "section": None, "frozen": None, "review": None, "index": None,
           "base": ledger.state()}      # the ledger before this run: reconciliation's scope
    # every local chat send of the run checked against the engine's load record
    watch, llm.LOADS = llm.LOADS, llm.LoadWatch()
    try:
        run(a, d, out, models, ledger, stop, filing, taxonomy, tools, identity)
    except KeyboardInterrupt as e:
        tagless.stop()      # the record first, then the interrupt
        out["result"] = getattr(e, "c7", None) or out["result"]
        out.update(outcome="interrupted", error={"error": repr(e), "trace": traceback.format_exc()[-3000:]})
        out["exceptions"].append({"step": "C7", "class": "interrupted",
                                  "why": (out["result"] or {}).get("why") or "KeyboardInterrupt before the tree"})
        report(d, out, ledger, stop)
        llm.LOADS = watch
        raise
    except Exception as e:
        out.update(outcome="not run", error={"error": repr(e), "trace": traceback.format_exc()[-3000:]})
        out["exceptions"].append({"step": "C7", "class": "not run", "why": repr(e)})
    tagless.stop()
    report(d, out, ledger, stop)
    llm.LOADS = watch
    return d


def split_relations(filing, meta):
    """(rel, shadow) for a repair's split check. rel: the legacy overlap policy, in both arms: R7 is off in
    steps.split (steps.PHASES["repair split"]), so a newly proposed split piece is never admitted by R7. shadow: for
    an R source run (run.json root_cause_fixes) the R7 Relations, used only to count the refusals R7 would have
    relaxed (repair.splitter); None for a T source."""
    from .steps import phase_relations, relations
    is_r = meta.get("root_cause_fixes") is True     # run() has refused a flag that is not a Boolean (arm_of)
    rel = phase_relations(filing, "repair split", is_r)
    return rel, (relations(filing, cross_dims_unknown=True) if is_r else None)


def run(a, d, out, models, ledger, stop, filing, taxonomy, tools, identity=None):
    from .llm import CapReached, HostedEmbedder, OllamaEmbedder, model_identity
    from .repair_roles import Roles
    spread = load_spread(a.spread_run)
    meta = spread["run"]
    # the source's arm by the one contract (full_pipeline.arm_of): the flag a
    # Boolean, an R source with its overlap record, a T source without one. A record that does not hold is an
    # ArmRecord (a ValueError), raised before anything is opened: a text or a number is never read as a flag
    from .full_pipeline import arm_of
    try:
        source_diag = json.loads((Path(a.spread_run) / "diagnostics.json").read_text(encoding="utf8"))
    except (OSError, ValueError):
        source_diag = None
    if arm_of(meta, source_diag) == "R":    # an R source: the scope record, opened once the source policy is
        out["r7_scope"] = []    # known and before any early end; report() writes it on every exit (write_r7_scope)
    Embedder = HostedEmbedder if meta.get("hosted_profile") else OllamaEmbedder    # the spread run's own embedder
    if meta.get("tags_hidden") is not True:
        raise ValueError(f"{a.spread_run} was made with tags shown; tags-shown runs are no longer repaired")
    section = a.section or meta["section"]
    if a.section:
        ran = meta["section"] if isinstance(meta["section"], list) else [meta["section"]]
        if a.section not in ran:
            raise ValueError(f"--section {a.section}: the run spread {ran}")
        for n in ("claims", "rows", "diagnostics"):
            spread[n] = json.loads((Path(a.spread_run) / f"{n}.json").read_text(encoding="utf8"))
    path = a.filing or meta["filing"]
    path = Path(path) if Path(path).is_absolute() else ROOT / path
    company = path.parent.name           # the CIK folder, as spread.main
    if filing is None:
        from .corrections import open_filing
        filing = open_filing(str(path), meta.get("corrections"))   # the source run's corrections record, if any
    tagless.start(filing)       # the tree's models see line IDs, never tags (tagless.py)
    if taxonomy is None:
        from .taxonomy import Taxonomy
        year = taxonomy_year(filing)
        taxonomy = Taxonomy(edition(year))
    ids = [r["row_id"] for r in section_rows(load_rows(), section)]
    draft, notes = (section_draft(spread, filing, taxonomy, section, ids) if a.section else
                    build_draft(spread, filing, taxonomy, section, ids))
    out.update(draft=draft, notes=notes, company=company)
    errs = repair.draft_errors(draft)
    if errs:                    # a code fault in the draft, not a tree outcome: no model call
        out["outcome"] = "bad root"
        out["exceptions"] += [{"step": "C7", "class": "bad root", "why": w} for w in errs]
        return
    if a.review:                # built and checked before the freeze; a refusal is "not run"
        from .repair_review import Review
        pins = json.loads((Path(a.spread_run) / "search.json").read_text(encoding="utf8"))["pins"] if (
            a.dry and meta.get("search")) else {}
        out["review"] = Review(a.spread_run, filing, taxonomy,
                               lambda m: Embedder(ledger=m, company=company, stop=stop), d / "calls.jsonl",
                               stop=stop, company=company, digest=pins.get("embedding_digest"),
                               identity=None if a.dry else identity or model_identity, spend_usd=a.spend_usd,
                               role_limits=a.role_limits, model=a.review_model)
    if a.section and not section_ready(a, d, out, draft, spread, section, models, identity):
        return
    F, complete = repair.state_of(draft, draft["registry"])
    spread_failed = sorted(c["check"] for c in spread["checks"] if not c["ok"]
                           and (not a.section or c.get("section") == section))     # --section: its own checks
    out["root"] = {"failures": repair.ids_of(F), "complete": complete, "spread_failed_checks": spread_failed,
                   "same_as_spread": sorted(repair.ids_of(F)) == spread_failed}
    if not out["root"]["same_as_spread"]:       # what C7 sees must be what the spread failed (protection 1)
        out["exceptions"].append({"step": "C7", "class": "root differs from the spread's checks",
                                  "why": f"recomputed {repair.ids_of(F)}; the spread failed {spread_failed}"})
    if not F:                   # recorded as needing no repair, no model call
        out["outcome"] = "needs no repair"
        return
    if stop.is_set():
        raise CapReached("stop switch set before the repair started")
    if tools is None:
        search = None
        if a.search:            # the index's embedding requests go through the shared ledger and the stop switch
            from .search import Search
            emb = Embedder(ledger=repair.Counted(ledger), company=company, stop=stop)
            out["index"] = emb.ledger       # the index build's sends; a role call puts the embedder on its own meter
            out["index_embedder"] = emb
            search = Search.of(filing, emb, **({"cache": a.cache} if a.cache else {}))
            out["search"] = search
        tools = tools_of(filing, draft, search, tax=taxonomy, statements_only=bool(meta.get("statements_only")))
    engines, hosted_kwargs, _ = hosted_tree(a, models)
    roles = out["roles"] = Roles(d / "calls.jsonl", models=models, tools=tools, stop=stop, company=company,
                                 num_ctx=a.num_ctx, role_ctx=role_ctx(a), answer_cap=a.answer_cap,
                                 context_net=a.context_net,
                                 engine=engines, caller_kwargs=hosted_kwargs)
    component = bool(a.section) and not a.review    # no C8: the given limits, no review reserve
    review = out["review"]          # --review: C8 from a reserve inside the given limits
    rel, shadow = split_relations(filing, meta)
    if (shadow is None) != (out.get("r7_scope") is None):       # one policy, read once: never two answers
        raise ValueError("the repair's scope record and its split relations disagree on the source policy")
    res = out["result"] = repair.repair(
        draft, **{**roles.kwargs(), **({"c8_review": review} if review else {})}, models=models, arm=a.arm,
        dispatches=a.dispatches, split=repair.splitter(taxonomy, filing, rel=rel, period=draft.get("period"),
                                                       shadow=shadow, log=out.get("r7_scope")), ledger=ledger,
        company=company, review=not component, limits=section_limits(a) if a.section else None,
        review_line=review.config["line_ceiling"] if review else None, feedback=a.critic_feedback == "on")
    if res["committed"] is not None:
        out["outcome"] = "repaired"
    elif res.get("unreviewed") is not None:
        out["outcome"] = "selected, unreviewed"
        out["exceptions"].append({"step": "C7", "class": res["termination"], "why": res["why"]})
    else:
        out["outcome"] = "not repaired"
        out["exceptions"].append({"step": "C7", "class": res["termination"], "why": res["why"]})


def role_ctx(a):
    """--role-num-ctx as {role: context}; a role named twice or a context not above the answer cap is refused."""
    out = {}
    for x in a.role_num_ctx:
        role, _, n = x.partition("=")
        if role in out or not n.isdigit() or int(n) <= a.answer_cap:
            raise SystemExit(f"--role-num-ctx {x}: one ROLE=N per role, N a whole number above the answer cap")
        out[role] = int(n)
    return out


def section_limits(a):
    """The given limits of a --section component test, passed into the tree as they are."""
    return {"requests": a.max_calls, "tokens": a.max_tokens, "wall_s": a.wall_s}


def effective_budget(a, review_line=None):
    """The tree's effective budget for a --section run, as repair.repair builds it: no review reserve, or with
    --review the review line's ceiling times the depth, inside the same limits."""
    b = repair.Budget(a.arm, a.dispatches, limits=section_limits(a), lines=None if review_line else 0,
                      review_line=review_line)
    return {"limits": b.limits, "derived": b.derived, "reserve": b.reserve, "tree": b.left("tree"),
            "review": ({"line": review_line, "lines": repair.DEPTH, "pool": b.left("review")} if review_line else
                       "none: component test; a selected, re-checked candidate is recorded unreviewed"),
            "tokens_per_request": repair.TOKENS_PER_REQUEST, "short": b.short}


def section_ready(a, d, out, draft, spread, section, models, identity):
    """The section checks (section_check), then frozen.json before the first dispatch. False when the
    section stops here: "not run" with the reasons, or "dry run" after the freeze."""
    chk = out["section"] = section_check(draft, spread, section)
    stops = section_stops(chk)
    if stops:
        out["outcome"] = "not run"
        out["exceptions"] += [{"step": "C7", "class": k, "why": json.dumps(w, default=str)} for k, w in stops]
        return False
    engines, _, hosted = hosted_tree(a, models)
    if a.dry:
        ident = "not read: a dry run makes no network request"
    else:                       # local models only: a hosted one has no digest (the Caller checks what serves it)
        from .llm import model_identity
        local = {r: m for r, m in models.items() if engines == "ollama" or engines[r] == "ollama"}
        ident = (identity or model_identity)(local) if local else "not read: every role is hosted"
    out["frozen"] = {"models": models, "identity": ident, "num_ctx": a.num_ctx, "role_num_ctx": role_ctx(a),
                     "answer_cap": a.answer_cap, "num_batch": llm.batch_pins(
                         m for r, m in models.items() if engines == "ollama" or engines[r] == "ollama"),
                     "engines": engines, "hosted": hosted,
                     "command": vars(a), "beam": repair.BEAM, "children": repair.CHILDREN, "depth": repair.DEPTH,
                     "caps": {"requests": a.max_calls, "local_tokens": a.max_tokens, "wall_s": a.wall_s,
                              "spend_usd": a.spend_usd, "role_limits": a.role_limits},
                     "budget": effective_budget(a, out["review"] and out["review"].config["line_ceiling"]),
                     "review": out["review"] and out["review"].config, "arm": a.arm, "dispatches": a.dispatches,
                     "section": section, "passed": ["root check", "reconstruction",
                                                    "unplaced lines", "not structurally ineligible"],
                     "lines_outside_the_total": chk["structural"]["lines_outside_the_total"]}
    write(d, "frozen.json", json.dumps(out["frozen"], indent=1, default=str))
    if a.dry:
        out["outcome"] = "dry run"
        return False
    return True


def section_stops(chk):
    """[(class, why)]: what stops a section before any dispatch, from section_check."""
    return [(k, w) for k, bad, w in (
        ("root differs from the run's checks", not chk["root"]["same"], chk["root"]),
        ("reconstruction differs from the run", not chk["reconstruction"]["same"],
         sorted(chk["reconstruction"]["differs"])),
        ("unplaced lines differ from the run's", not chk["unplaced"]["same"], chk["unplaced"]),
        ("structurally ineligible", chk["structural"]["ineligible"], chk["structural"]["why"])) if bad]


def report(d, out, ledger, stop):
    """The run's record: every file written once; the exceptions counted by step and class."""
    j = lambda o: json.dumps(o, indent=1, default=str)     # noqa: E731
    res, roles = out["result"], out["roles"]
    if out["draft"] is not None:
        write(d, "draft_in.json", j(out["draft"]))
    if out["section"] is not None:
        write(d, "section.json", j(out["section"]))
    if res is not None:
        repair.write_record(d, res)
        if res["committed"] is not None:
            write(d, "draft_out.json", j(res["draft"]))
    if out["review"] is not None:       # each review with its embedding calls and searches
        write(d, "c8_review.json", j({"config": out["review"].config, "reviews": out["review"].log}))
    if out.get("r7_scope") is not None:     # an R source only; a T repair's folder is unchanged
        from . import open_claims
        write(d, "r7_repair_scope.json", j(open_claims.repair_scope(
            out["r7_scope"], ended=out.get("outcome"), tree_started=out.get("roles") is not None)))
    if out.get("search") is not None:
        s = out["search"]
        write(d, "search.json", j({"pins": s.pins, "skipped": s.skipped, "searches": s.log,
                                   "embedding_calls": getattr(s.embedder, "calls", [])}))
    elif getattr(out.get("index_embedder"), "calls", None):     # the index build failed: its requests still recorded
        write(d, "embedding_calls.json", j({"embedding_calls": out["index_embedder"].calls}))
    callers = (roles.callers if roles is not None else []) + (out["review"].callers if out["review"] else [])
    embedders = ([roles.embedder()] if roles is not None and roles.embedder() is not None else []) + (
        out["review"].embedders if out["review"] else [])
    # the failures structured (llm.Caller.failure, as spread.py writes it): the callers', then the embedders'
    failures = ([getattr(c, "failure", None) for c in callers if c.failed]
                + [e.failure for e in embedders if getattr(e, "failure", None)])
    hosted, unknown = hosted_use(*callers, *embedders)     # as OpenRouter reported them; the ledgers bound dollars
    ledger_state = ledger.state()
    rec = reconciliation(out, ledger_state, out["base"])
    if not rec["ok"]:
        out["exceptions"].append({"step": "C7", "class": "totals differ from the ledger", "why": json.dumps(rec)})
    exc = out["notes"] + out["exceptions"]
    loads = llm.LOADS.write(d) if llm.LOADS is not None else None
    by = {}
    for x in exc:
        by.setdefault(f"{x['step']} {x['class']}", []).append({"concept": x.get("concept"), "why": x.get("why")})
    write(d, "exceptions.json", j({"outcome": out["outcome"], "exceptions": len(exc),
                                   "counts": {k: len(v) for k, v in sorted(by.items())},
                                   "cases": dict(sorted(by.items())),
                                   **({"engine_loads": loads} if loads and (loads["flags"] or loads["not_observed"])
                                      else {})}))
    write(d, "outcome.json", j({
        "outcome": out["outcome"], "root": out["root"],
        "section": out["section"] and {k: out["section"][k] for k in ("passed", "structural")},
        "termination": res["termination"] if res else None, "why": res["why"] if res else None,
        "decision": res["decision"] if res else None, "committed": res["committed"] if res else None,
        "unreviewed": res.get("unreviewed") if res else None,
        "role_calls": len(callers),
        "requests": rec["counted"] and rec["counted"]["tree_chat"] + rec["counted"]["c8_chat"],
        "embedding_requests": rec["counted"] and rec["counted"]["embeddings"],
        "attempts": sum(c.calls for c in callers), "unsent_refusals": rec["unsent"],
        "spend_usd": ledger_state["spend_usd"] - out["base"]["spend_usd"],     # this run's, as the ledger counts it
        "hosted_tokens": hosted, "hosted_unknown": unknown,
        "caller_failures": [c.failed for c in callers if c.failed], "stopped": stop.is_set(),
        "caller_failure": next(iter(failures), None), "caller_failures_structured": failures,
        "context_net": roles.net_events if roles is not None else [], "reconciliation": rec,
        "ledger": {**ledger_state, **({"company_delta": company_delta(ledger_state["plan"], out["base"]["plan"],
                                                                      hosted, unknown)}
                                      if "plan" in ledger_state else {})},
        "engine_loads": loads, "error": out["error"]}))
    print("outcome:", out["outcome"], (res or {}).get("termination") or "", (out["error"] or {}).get("error") or "")
    print(d)


def reconciliation(out, state, base=None):
    """The run's requests physically sent, by kind, against the ledger's count over the same scope: tree chat (every
    role but C8's checker), C8 chat (role `checker`) and embeddings (the tree search's, the index build included, and
    each review's). Sends are counted where they are made (repair.Counted: after the reservation and before_send both
    succeed, just before transport; the tree's and the reviews' in the Budget's record, the index build's on its
    embedder), so an attempt refused before sending is in neither count and is listed in `unsent`. `base`: the ledger's
    state when this run started; only the change since then is this run's, so a company ledger used by runs one after
    another reconciles each run's own sends. Without the tree's record (a fault after the roles were built) the sends
    are unknown and the totals do not reconcile. A difference is never suppressed: report() records it as an
    exception."""
    led = sent_by_role(state)
    for k, n in sent_by_role(base or {}).items():
        led[k] = led.get(k, 0) - n
    led = {"tree_chat": sum(n for k, n in led.items() if k not in ("checker", "embedding")),
           "c8_chat": led.get("checker", 0), "embeddings": led.get("embedding", 0)}
    budget = ((out["result"] or {}).get("record") or {}).get("budget")
    index = out.get("index")
    unsent = [{"pool": "index", "call": "embedding", **u} for u in index.unsent] if index is not None else []
    built = {"chat": 0, "embedding": 0} if index is None else index.sends
    if budget is not None:
        s = budget["sends"]
        counted = {"tree_chat": s["tree"]["chat"], "c8_chat": s["review"]["chat"],
                   "embeddings": built["embedding"] + s["tree"]["embedding"] + s["review"]["embedding"]}
        unsent = budget["unsent"] + unsent
    elif out["roles"] is None:          # no role call was possible: only the index build can have sent
        counted = {"tree_chat": 0, "c8_chat": 0, "embeddings": built["embedding"]}
    else:
        counted = None
    return {"counted": counted, "ledger": led, "unsent": unsent, "ok": counted == led}


if __name__ == "__main__":
    main()
