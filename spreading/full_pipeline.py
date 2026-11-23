"""The full pipeline on one company and line-up as one command:

    .venv-agents\\Scripts\\python.exe -m spreading.full_pipeline <settings.json> \\
        --lineup <lineup> --dollars 0

The settings file holds the filing, the statements in their fixed order with each statement's context and limits,
the spread and tree settings, the company-wide limits, the line-ups and the grading inputs; code holds no company,
statement or section. The job:

1. runs each statement's spread (the line-up's arguments; the tiebreaker when its flags are there), each step's
   limits the smaller of its own and what is left of the company's (requests, tokens, wall time, dollars; the
   tiebreaker's requests and dollars);
2. copies each spread run's record into a new cumulative statement artifact, cumulative/<statement>_v0, and checks
   that the statement-level outputs recomputed from it (assemble) equal the run's own record; the run folder itself is
   never written to (fingerprints before and after);
3. fixes the tree's order before its first step (statements in the settings' order, sections in template order) and
   writes it (order.json); each failing section is attempted at most once, at most `max_sections` attempts per job
   in that order, each on the current cumulative artifact with --review; a failing section not attempted is recorded
   with its reason (a run-level check, the precheck repair_run would stop on, "not run, cap reached", a partial or
   unchecked record; with tree.on false, every section is recorded "the tree is switched off for this job", after the
   same precheck). There is no mode without commitment: a repair repair_run commits is applied;
4. applies each committed repair to a new cumulative version (<statement>_v<n+1>): the repair's input must be the
   current version, the committed registry passes repair.draft_errors against the current version's ownership (so
   an amount another section owns is refused at commit), each changed line has a settled review; the section's claims,
   rows and checks are rebuilt, the checker records of the settled reviews replace the line's in c4.json (so a later
   review's ownership holds them), and the claims, totals, coverage, controls and checks are recomputed;
5. grades the final versions, one per statement, when the settings name a reference and a frame;
6. writes each line's change chain, the transitions recorded in order (mechanism, before, after), by code; a chain
   records transitions, it does not attribute cause.

A step that stops at a limit the company set (the budget its recorded stop names: a ledger limit or a caller's cap
cut to the company's remainder, the company ledger's own, or the company's totals exhausted) stops the job, recorded
as capped, with every record kept. A spread that stops at its own statement limit is recorded capped, its sections
are not repaired, and the job goes on. The job's deadline counts from the command's start, preparation and
cumulative processing included; STOP in the job folder (or the project's STOP) stops the job: a watcher puts STOP in
the active child's run folder, and no later child starts. Each child's start is logged, with the folder its run is
made in, before it can send anything; its end is logged however it ends.

Each step's usage is the child's own (its ledger), reconciled within the child: the requests its ledger counts, by
role, equal the chat requests its call records show sent (refusals before sending and bookkeeping records excluded)
plus the embedding requests it recorded, and the child's own reconciliation, when it writes one, holds. A harness
defect stops the job, keeps every record and starts no later child: a step that cannot start, raises or leaves no
record (usage unknown is recorded unknown, never zero), a reconciliation that fails, or a source (a child's folder or
a cumulative version) changed after it was recorded; the command then exits unsuccessfully. The tree runs without
search; its review keeps the source run's search setting (repair_review reads it from the source run.json).

Files: run.json, steps.jsonl (a start and an end record per step, appended), children/NN_kind/ (each child's run
folder), order.json, cumulative/, job.json, exceptions.json (the job's exception report), chains.json and, when
graded, the grade run's folder name in job.json. Finalization is guarded: a failure in grading, chains or the
sources' check is recorded and job.json and exceptions.json are still written.

A child that ends on a caller transport failure (its request failed or was refused at the HTTP or network level, after
any permitted retry; the child's structured caller failure, kind "transport") stops the job: the job ends incomplete,
no later statement or tree section starts (each recorded "not run: the job ended incomplete"), the partial records
are kept, finalization runs, nothing is graded, and the command exits unsuccessfully. Every kind of caller failure
has its end (KIND_ENDS): an unreadable reply and a violation end the job incomplete too, STOP during a call ends it
stopped, a ledger refusal capped (the job goes on when the limit was the child's own), a request not sent and an
unknown kind are harness defects. Of a child's several failures, a STOP decides, then a harness defect, then
incomplete, then capped (SEVERITY); every one is kept in stop.json and job.json. Model outcomes (a schema-invalid
reply after its retry, a cut reply, a withdrawal) are not caller failures and do not stop the job. Any end other than
finished also writes stop.json beside exceptions.json: the end, why, when, and the triggering child, role, request,
kind and reason, HTTP status or cap.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import threading
import time
import traceback
from decimal import Decimal
from pathlib import Path

from . import open_claims, provenance, repair, template
from .llm import STOP_FILE
from .packs import ROOT, load_rows, section_rows
from .ledger import sent_by_role
from .runrec import new_run, write

CHAIN_LABEL = "recorded after earlier tiebreaker or second-look processing; counterfactual unavailable"
CHAIN_NOTE = "recorded transitions in order, not causal attribution"
CAP_REACHED = "not run, cap reached"            # a failing section past the job's max_sections
TREE_OFF = "the tree is switched off for this job"
DIMENSIONS = ("requests", "tokens", "wall_s", "dollars", "haiku_requests", "haiku_dollars")
ROLE_ONLY = ("haiku_requests", "haiku_dollars")     # one role's limits, never the whole job's
POLL_S = 0.5            # the watcher's look at the stop switch and the deadline
ANNOTATION = "gap_match_candidates"     # spread.run's B annotation on a failed check: carried, never recomputed
R_FLAG = "root_cause_fixes"             # run.json: the recorded source flag; the arm is read from it alone
SETUP_REFUSALS = ("C1:", "C2:", "C7:")  # spread.setup's refusals: the statement is recorded refused, the job goes on


# ---- settings, fingerprints ---------------------------------------------------------------------------------------

def load_settings(path):
    cfg = json.loads(Path(path).read_text(encoding="utf8"))
    missing = [k for k in ("filing", "statements", "spread_args", "tree", "company_limits", "lineups") if k not in cfg]
    if missing:
        raise ValueError(f"{path}: settings lack {missing}")
    if not isinstance(cfg["tree"].get("on", True), bool):
        raise ValueError(f"{path}: tree.on is {cfg['tree']['on']!r}, not true or false")
    for k in ("corrections", "named_totals"):
        if not isinstance(cfg.get(k, False), bool):
            raise ValueError(f"{path}: {k} is {cfg[k]!r}, not true or false")
    if cfg.get("corrections"):
        raise ValueError(f"{path}: corrections is on: the labels-fix step was removed; use "
                         "named_totals")
    from .spread import LOOP_PROMPT
    for where, args in [("spread_args", cfg["spread_args"])] + [(f"line-up {k}", lu.get("spread_args", []))
                                                                for k, lu in cfg["lineups"].items()]:
        names = [x.partition("=")[0] for x in args if x.startswith("--")]
        if any(n == "--tags-hidden" or len(n) > 5 and "--prompt".startswith(n) for n in names):
            raise ValueError(f"{path}: {where} {args}: tags are always hidden and {LOOP_PROMPT}, the default, is the "
                             "only placement prompt since the tags-shown side was removed; drop "
                             "--tags-hidden and --prompt in every form")
    return cfg


def parts(cfg, lu):
    """The job's two parts that can be switched off, read from what the job runs: the tiebreaker from the merged
    spread arguments (the settings' and the line-up's), the tree from tree.on (absent: on). The UI's manifest and
    job.json both take them from here."""
    spread = [*cfg["spread_args"], *lu.get("spread_args", [])]
    return {"tiebreaker": "on" if "--third-opinion" in spread else "off",
            "tree": "on" if cfg["tree"].get("on", True) else "off"}


def review(cfg, lu):
    """The job's review settings, read from the merged spread arguments: the blind checker ("off" with --no-checker)
    and the second look (--checker-feedback: the analyst retries a disputed line once, seeing both answers).
    The UI's manifest and job.json record them beside the parts."""
    spread = [*cfg["spread_args"], *lu.get("spread_args", [])]
    return {"checker": "off" if "--no-checker" in spread else "on",
            "second_look": "on" if "--checker-feedback" in spread else "off"}


def checker_off_resolved(cfg, lineup):
    """The settings with the second look removed from the copies of both argument lists when the checker is off: it
    needs the checker's answer, so a checker-off job runs without it and nobody is asked. Otherwise unchanged."""
    lu = cfg["lineups"][lineup]
    if "--no-checker" not in [*cfg["spread_args"], *lu.get("spread_args", [])]:
        return cfg
    drop = lambda args: [x for x in args if x != "--checker-feedback"]
    return {**cfg, "spread_args": drop(cfg["spread_args"]),
            "lineups": {**cfg["lineups"], lineup: {**lu, "spread_args": drop(lu.get("spread_args", []))}}}


def without_tiebreaker(args):
    """Spread arguments without the tiebreaker's: --third-opinion, --third-opinion-all and --third-model with its
    model, as the next argument or as --third-model=MODEL."""
    out, model = [], False
    for x in args:
        if not model and not (x in ("--third-opinion", "--third-opinion-all", "--third-model")
                              or x.startswith("--third-model=")):
            out.append(x)
        model = x == "--third-model"
    return out


def hosted_roles(cfg, lu):
    """The roles a line-up sends to OpenRouter beyond Haiku, and the ones a repair section limits: a spread run with
    --engine openrouter puts its analyst and checker there (and the review then asks that hosted checker); a tree
    with --hosted-profile puts all four of its roles there. Returns (company roles, section roles)."""
    from .repair_run import HOSTED_TREE_ROLES
    spread, tree = [*cfg["spread_args"], *lu.get("spread_args", [])], [*cfg["tree"].get("args", []),
                                                                        *lu.get("tree_args", [])]
    engines = [spread[i + 1] for i, x in enumerate(spread[:-1]) if x == "--engine"]
    hosted_spread = bool(engines) and engines[-1] == "openrouter"
    hosted_tree = "--hosted-profile" in tree
    company = [*(("analyst", "checker") if hosted_spread else ()), *(HOSTED_TREE_ROLES if hosted_tree else ())]
    section = [*(HOSTED_TREE_ROLES if hosted_tree else ()), *(("checker",) if hosted_spread else ())]
    return company, section


def hosted_profile(lu):
    """The line-up's --hosted-profile, or None for a local line-up."""
    args = lu.get("spread_args", [])
    return args[args.index("--hosted-profile") + 1] if "--hosted-profile" in args else None


def tree_dispatches(cfg, lu):
    """The arguments the driver adds to a tree's command for its requests per question: a hosted tree
    (--hosted-profile) reserves repair_roles.question_sends("openrouter"), two answer attempts and one 429 recovery,
    for every question (`--dispatches 3`); a local tree adds nothing and keeps repair_run's default. Settings that
    name --dispatches themselves add nothing; on a hosted tree below the hosted sends they are refused (ValueError)."""
    from .repair_roles import question_sends
    tree = [*cfg["tree"].get("args", []), *lu.get("tree_args", [])]
    if "--hosted-profile" not in tree:
        return []
    need, given = question_sends("openrouter"), [tree[i + 1] for i, x in enumerate(tree[:-1]) if x == "--dispatches"]
    if not given:
        return ["--dispatches", str(need)]
    if not given[-1].isdigit() or int(given[-1]) < need:
        raise ValueError(f"the hosted tree's --dispatches {given[-1]} is below a hosted question's {need} sends")
    return []


def load_allocation(path, company_roles, section_roles):
    """The frozen allocation of a hosted line-up (nothing here chooses it). `company`: the hosted roles' sub-caps
    shared over the whole job, held in the company ledger beside the tiebreaker's (which stays the settings'
    company_limits), one for every role in `company_roles`, the spread's analyst and checker included (section
    allocations alone set no sub-cap shared across sections). `section`: one repair section's dollar allocation
    (`spend_usd`) and its hosted roles' allocations (`roles`, every role in `section_roles`), passed to repair_run as
    --spend-usd and --role-limit. Malformed or short is refused (ValueError). Content only: the allocator checks its
    pending file with it too; the job also calls require_frozen."""
    from .ledger import check_role_limits
    q = Path(path)
    q = q if q.is_absolute() else ROOT / q
    raw = q.read_bytes()
    al = json.loads(raw)
    company, section = al.get("company"), al.get("section")
    check_role_limits(company)
    short = [r for r in company_roles if r not in company]
    if short:
        raise ValueError(f"the allocation's company part lacks the hosted roles {short}")
    if "haiku" in company:
        raise ValueError("Haiku's sub-cap is the settings' company_limits, not the allocation's")
    if not isinstance(section, dict) or isinstance(section.get("spend_usd"), bool) or \
            not isinstance(section.get("spend_usd"), (int, float)) or not section["spend_usd"] > 0:
        raise ValueError(f"the allocation's section needs spend_usd above 0, not {section!r}")
    check_role_limits(section.get("roles"))
    missing = [r for r in section_roles if r not in section["roles"]]
    if missing:
        raise ValueError(f"the allocation's section lacks the roles {missing}")
    return {"path": str(q), "sha256": hashlib.sha256(raw).hexdigest(), "company": company, "section": section}


def require_frozen(path):
    """The job takes an allocation only once the allocator froze it: the outcome.json beside it must be readable
    and say "frozen" (not interrupted, failed, refused or anything else). Otherwise ValueError."""
    q = Path(path)
    o = (q if q.is_absolute() else ROOT / q).parent / "outcome.json"
    try:
        done = json.loads(o.read_bytes())
    except (OSError, ValueError) as e:
        raise ValueError(f"the allocation needs a readable outcome.json beside it ({o}): {e}") from e
    if not isinstance(done, dict) or done.get("outcome") != "frozen":
        got = done.get("outcome") if isinstance(done, dict) else done
        raise ValueError(f"the allocation is not frozen: {o} says outcome {got!r}")


def fingerprint(folder):
    """{relative path: sha256} of every file under the folder."""
    d = Path(folder)
    return {p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.rglob("*")) if p.is_file()}


def read(d, name):
    return json.loads((Path(d) / name).read_text(encoding="utf8"))


def norm(o):
    return json.loads(json.dumps(o, default=str))


# ---- the children: the only place the driver calls spread and repair_run ------------------------------------------

def read_jsonl(p):
    p = Path(p)
    return [json.loads(x) for x in p.read_text(encoding="utf8").splitlines() if x.strip()] if p.is_file() else []


def physical(r):
    """A call record of a request that went out: it carries the messages sent and attempt 1 or more (a hosted
    question's 429 recovery is its third), and is not marked not sent. Refusals before sending (attempt 0, "not
    sent") and bookkeeping records (recovery markers, one-send outcomes, the no-retry note: no messages) are not
    sends."""
    return "system" in r and (r.get("attempt") or 0) >= 1 and "not sent" not in str(r.get("error") or "")


def embedding_sends(d):
    """Embedding requests the child recorded: its search's (the index build included) and each review's. One entry per
    physical request (a retried batch has one per attempt). embedding_calls.json is written only when the index build
    failed and no search.json exists, so no request is in both."""
    d, n = Path(d), 0
    for name in ("search.json", "embedding_calls.json"):
        if (d / name).is_file():
            n += len(read(d, name).get("embedding_calls") or [])
    if (d / "c8_review.json").is_file():
        n += sum(len(r.get("embedding_calls") or []) for r in read(d, "c8_review.json").get("reviews") or [])
    return n


def reconcile(d, o, led, calls):
    """The child's own reconciliation, its scope only (its ledger and its records, never the company's sums): the
    requests its ledger counts, by role, against the chat requests its call records show sent, by role, plus the
    embedding requests it recorded. The child's own reconciliation result, when it writes one, must hold too."""
    counted = {}
    for r in calls:
        if physical(r):
            counted[r.get("role")] = counted.get(r.get("role"), 0) + 1
    emb = embedding_sends(d)
    if emb:
        counted["embedding"] = emb
    ledger = {k: v for k, v in sent_by_role(led).items() if v}
    child = o.get("reconciliation")
    ok = counted == ledger and (child is None or child.get("ok") is True)
    why = None if ok else "; ".join(w for bad, w in (
        (counted != ledger, f"the ledger counts {ledger}, the records show {counted} sent"),
        (child is not None and child.get("ok") is not True, f"the child's own reconciliation: {child}")) if bad)
    return {"scope": "this child's own ledger and records", "ledger": ledger, "counted": counted,
            "unsent_or_bookkeeping": len(calls) - sum(1 for r in calls if physical(r)), "child": child, "ok": ok,
            "why": why}


LEDGER_LIMITS = {"dispatches": "requests", "local_tokens": "tokens", "spend_usd": "dollars",
                 "batch_wall_s": "wall_s", "company_wall_s": "wall_s"}


def stops_recorded(d, o, led):
    """The reasons the child recorded for stopping at a limit: its ledger's stop, its callers' failures and every
    line or step that ended on CapReached (spread's c4.json; repair_run's outcome)."""
    out = [led.get("stopped"), o.get("caller_failed"), *(o.get("caller_failures") or [])]
    if o.get("termination") == "cap_reached":
        out.append(o.get("why") or "cap_reached")
    if (Path(d) / "c4.json").is_file():
        def walk(x):
            if isinstance(x, dict):
                if x.get("stop_class") == "CapReached":
                    out.append(x.get("why") or x.get("error"))
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)
        walk(read(d, "c4.json"))
    return [str(w) for w in out if w]


def limiting(reasons):
    """The budget that stopped the child, read from its recorded reasons: which ledger (the run's own, or the plan or
    company ledger it counted in beside it), which limit, and the reason as recorded. A role's own limit (the
    tiebreaker's) refuses that role only and stops nothing else: kind "role". None when nothing stopped it at a
    limit."""
    found = []
    for why in reasons:
        text = why[len("ledger: "):] if why.startswith("ledger: ") else why
        ledger = "company" if text.startswith("plan ledger: ") else "stage" if text.startswith("stage ledger: ") \
            else "run"
        m = re.search(r"limit (?:reached|passed after commit): (\w+)", text)
        r = re.search(r"(\w+) (call|spend) limit", text)
        if m:
            found.append({"kind": "limit", "ledger": ledger, "limit": m.group(1),
                          "budget": LEDGER_LIMITS.get(m.group(1)), "reason": why})
        elif r:
            found.append({"kind": "role", "ledger": ledger, "limit": f"{r.group(1)} {r.group(2)}s",
                          "budget": f"{r.group(1)}_{'requests' if r.group(2) == 'call' else 'dollars'}",
                          "reason": why})
        elif re.search(r"\bcall cap \d+", text):
            found.append({"kind": "limit", "ledger": "caller", "limit": "call cap", "budget": "requests",
                          "reason": why})
        elif "spend cap" in text:
            found.append({"kind": "limit", "ledger": "caller", "limit": "spend cap", "budget": "dollars",
                          "reason": why})
    run = [x for x in found if x["kind"] == "limit"]
    return (run or found or [None])[0]


CALLER_FAILURES = "caller_failures_structured"   # repair_run: one structured failure per failed caller
CALLER_FAILURE = "caller_failure"   # the child's structured caller failure (llm.py's Caller.failure, as spread.py
#   and repair_run write it): a dict with kind (llm.FAILURE_KINDS, each with its end in KIND_ENDS), role, tag,
#   http_status, dispatch, attempt, error_code and, on a 429, rate_limit_class and rate_limit_reason (why it was not
#   recovered: second_429, job_allowance_recoveries, job_allowance_wait, stopped, interrupted or another of
#   llm.Caller.rate_limit's reasons) and rate_limit_stop (llm.RATE_LIMIT_STOPS' text for the two stop reasons, else
#   None)
FAILURE_FIELDS = ("kind", "role", "tag", "http_status", "dispatch", "attempt", "error_code", "rate_limit_class",
                  "rate_limit_reason", "rate_limit_stop")
KIND_ENDS = {   # a failed Caller sends nothing more: each kind of failure, the job's end and the reason recorded
    "transport": ("incomplete", "transport refusal"),      # HTTP error, a 200 carrying an error object, network
    "response": ("incomplete", "unreadable reply"),        # a 200 reply that could not be read: nothing usable came
    "violation": ("incomplete", "violation"),              # counts, cost, model, provider or tier broke a check
    "stopped": ("stopped", "stop switch during the call"),
    "ledger": ("capped", "ledger refusal"),
    "not_sent": ("harness defect", "not sent"),            # the vendor guard, no key, a request body that differs
}
SEVERITY = ("stopped", "harness defect", "incomplete", "capped")    # of a child's several failures, the first here
#   decides the job's end: an explicit STOP outranks every other failure, a transport failure included, and is never
#   read as a knock-on; the others are kept beside it in `failures`, never used to pick the label. The job's deadline
#   is not a STOP: the watcher records its cause (stopped_by "job deadline"), read first.


def failure_end(f):
    """(the job's end, the reason recorded) for a structured caller failure; an unknown kind is a harness defect."""
    return KIND_ENDS.get(f.get("kind"), ("harness defect", f"unknown caller failure kind {f.get('kind')!r}"))


def worst(fails):
    """The most severe of a child's structured caller failures (SEVERITY; the first recorded among equals), or None."""
    return min(fails, key=lambda f: SEVERITY.index(failure_end(f)[0]), default=None)


def caller_failures(o):
    """The structured caller failures a child's outcome records, never read from the failure's text: repair_run's
    whole list (one per failed caller) when the outcome has it, else the one failure (spread.py's, the first's)."""
    x = o[CALLER_FAILURES] if isinstance(o.get(CALLER_FAILURES), list) else o.get(CALLER_FAILURE)
    return [f for f in (x if isinstance(x, list) else [x]) if isinstance(f, dict)]


def salvage(d):
    """What a child that left no outcome used, from its ledger's state if it left one; None if unknown."""
    p = Path(d) / "ledger" / "state.json"
    if not p.is_file():
        return None
    s = json.loads(p.read_text(encoding="utf8"))
    by = sent_by_role(s)
    return {"requests": sum(by.values()), "tokens": s.get("local_tokens", 0), "dollars": s.get("spend_usd", 0.0),
            "haiku_requests": by.get("haiku", 0), "haiku_dollars": (s.get("role_usd") or {}).get("haiku", 0.0),
            "known": "the ledger's state only: hosted tokens unknown"}


class Children:
    """The adapter to the two child commands. Each child keeps its own run ledger, its scope (usage and
    reconciliation are read from its record), and the job's limits reach it as arguments (each the smaller of the
    step's own and the company's remainder). With `shared` (the real commands' default) each child also counts every
    attempt in the job's one company ledger (`spread.main(argv, company=)`, `repair_run.main(argv, company=)`: a
    ledger.Ledger the child holds beside its own through ledger.Both, which charges each send itself; the
    tiebreaker's limit is the company ledger's role limit). `spread_step` and `repair_step` return the child's run
    folder (fakes in the offline tests: called with argv only unless `shared`)."""

    def __init__(self, spread_step=None, repair_step=None, shared=None):
        real = spread_step is None and repair_step is None
        if spread_step is None:
            from .spread import main as spread_step
        if repair_step is None:
            from .repair_run import main as repair_step
        self.spread_step, self.repair_step = spread_step, repair_step
        self.shared = real if shared is None else shared

    def spread(self, argv, ledger):
        return self.spread_step(argv, company=ledger) if self.shared else self.spread_step(argv)

    def repair(self, argv, ledger):
        return self.repair_step(argv, company=ledger) if self.shared else self.repair_step(argv)

    def usage(self, d):
        """What the child used: requests, requests by role and dollars from its own ledger (its reconciliation scope,
        which the company ledger is checked against); local and hosted tokens, hosted attempts with unknown tokens,
        tokens and the tiebreaker's requests and dollars read directly from its company_delta (ledger.company_delta,
        which both write), each falling back to its own ledger or calls only when the delta lacks the key; its
        reconciliation; the budget that stopped it, if one did. None when the child left no outcome or ledger."""
        d = Path(d)
        if not (d / "outcome.json").is_file():
            return None
        o = read(d, "outcome.json")
        led = o.get("ledger")
        if not isinstance(led, dict) or "sent" not in led:
            return None
        calls = read_jsonl(d / "calls.jsonl")
        delta = led.get("company_delta") or {}
        by = sent_by_role(led)
        why = stops_recorded(d, o, led)
        local = delta["local_tokens"] if "local_tokens" in delta else led.get("local_tokens", 0)
        hosted = delta["hosted_tokens"] if "hosted_tokens" in delta else sum(
            (r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0)
            for r in calls if physical(r) and r.get("cost_usd") is not None)
        unknown = delta["hosted_unknown"] if "hosted_unknown" in delta else None   # None: the child did not say
        return {"requests": led["sent"], "tokens": delta["tokens"] if "tokens" in delta else local + hosted,
                "local_tokens": local, "hosted_tokens": hosted, "hosted_unknown": unknown,
                "dollars": led.get("spend_usd", 0.0),
                "haiku_requests": delta["haiku_requests"] if "haiku_requests" in delta else by.get("haiku", 0),
                "haiku_dollars": delta["haiku_dollars"] if "haiku_dollars" in delta else
                (led.get("role_usd") or {}).get(
                    "haiku", sum(r.get("cost_usd") or 0.0 for r in calls if r.get("role") == "haiku")),
                "by_role": by, "stopped": led.get("stopped"), "outcome": o.get("outcome"),
                "termination": o.get("termination"), "stop_reasons": why, "limiting": limiting(why),
                "caller_failures": caller_failures(o), "reconciliation": reconcile(d, o, led, calls)}


class Company:
    """The company-wide limits of the whole job, across statements and repairs: the sum of each child's own usage,
    checked before every step and passed down as each step's limits. Wall time is the job's elapsed time from its
    start, preparation and cumulative processing included."""

    def __init__(self, limits, dollars, clock, start):
        self.limits = {"requests": limits["requests"], "tokens": limits["tokens"], "wall_s": limits["wall_s"],
                       "dollars": dollars, "haiku_requests": limits["haiku_requests"],
                       "haiku_dollars": min(limits["haiku_dollars"], dollars)}
        self.used = {k: 0 for k in DIMENSIONS if k != "wall_s"}
        self.clock, self.start, self.unknown = clock, start, []
        self.hosted = {"local_tokens": 0, "hosted_tokens": 0, "hosted_unknown": 0, "not_reported": []}

    def elapsed(self):
        return self.clock() - self.start

    def left(self):
        return {k: self.limits[k] - (self.elapsed() if k == "wall_s" else self.used[k]) for k in DIMENSIONS}

    def exhausted(self, pays):
        """The company-wide limits with nothing left; a free line-up has no dollar limit to exhaust. The tiebreaker's
        requests and dollars are a role's limit: used up, they refuse the tiebreaker only (the company ledger holds
        them) and the job goes on."""
        return sorted(k for k, v in self.left().items()
                      if v <= 0 and k not in ROLE_ONLY and (pays or k != "dollars"))

    def caps(self, own, pays):
        """A step's limits: its own, cut to what the company has left; `cut` names the limits the company set."""
        left = self.left()
        got = {"requests": min(own["max_calls"], int(left["requests"])),
               "tokens": min(own["max_tokens"], int(left["tokens"])),
               "wall_s": round(min(float(own["wall_s"]), left["wall_s"]), 3)}
        cut = [k for k, o in (("requests", own["max_calls"]), ("tokens", own["max_tokens"]),
                              ("wall_s", float(own["wall_s"]))) if got[k] < o]
        if pays:
            got["dollars"] = round(left["dollars"], 6)
            got["haiku"] = {"requests": left["haiku_requests"], "dollars": round(left["haiku_dollars"], 6)}
            cut.append("dollars")     # a step's only dollar limit is the company's
        return got, cut

    def add(self, use, label):
        """A child's usage. Usage not known is recorded as unknown, never counted as zero."""
        if use is None:
            self.unknown.append(label)
            return
        for k in self.used:
            self.used[k] += use.get(k, 0)
        for k in ("local_tokens", "hosted_tokens"):
            self.hosted[k] += use.get(k, 0)
        if use.get("hosted_unknown") is None:
            self.hosted["not_reported"].append(label)       # a child that does not report it: not counted as zero
        else:
            self.hosted["hosted_unknown"] += use["hosted_unknown"]
        if use.get("known"):
            self.unknown.append(f"{label}: {use['known']}")

    def state(self):
        h = self.hosted
        return {"limits": self.limits, "used": {**self.used, "wall_s": round(self.elapsed(), 3)},
                "left": self.left(), "unknown_usage": self.unknown,
                "tokens": {"local": h["local_tokens"], "hosted": h["hosted_tokens"],
                           "hosted_unknown": h["hosted_unknown"],
                           "hosted_unknown_not_reported_by": h["not_reported"],
                           "is": "accounted local tokens plus known hosted tokens; hosted usage incomplete"
                           if h["hosted_unknown"] > 0 else
                           "not known complete" if h["not_reported"] else "the total"}}


def open_ledger(d, company, allocation=None):
    """The job's one company ledger, in the job's folder, opened once with the company's limits, the tiebreaker's
    role limit (its shared sub-cap) and, on a hosted line-up, the hosted roles' sub-caps from the frozen allocation;
    every child counts each attempt in it beside its own ledger, so the ledger charges, and `Company` only sums the
    children's own records for reporting. Its wall limits are what is left of the job's deadline when it opens (the
    driver's watcher holds the deadline itself)."""
    from .ledger import Ledger
    lim, wall = company.limits, max(0.0, round(company.left()["wall_s"], 3))
    return Ledger(d / "company_ledger", limits={
        "dispatches": lim["requests"], "local_tokens": lim["tokens"], "spend_usd": lim["dollars"],
        "batch_wall_s": wall, "company_wall_s": wall},
        role_limits={"haiku": {"calls": lim["haiku_requests"], "spend_usd": lim["haiku_dollars"]},
                     **(allocation["company"] if allocation else {})})


def charged(led):
    """What the company ledger has charged: requests (sent and the sends of open reservations), local tokens and
    dollars; and how many attempts it billed as unknown (commit_unknown: the whole reservation counted)."""
    s = led.state()
    return {"requests": sum(sent_by_role(s).values()), "local_tokens": s.get("local_tokens", 0),
            "dollars": s.get("spend_usd", 0.0), "stopped": s.get("stopped"),
            "unknown_billing": sum(1 for e in read_jsonl(led.dir / "events.jsonl")
                                   if e.get("event") == "commit_unknown")}


class Watch(threading.Thread):
    """While a child runs: the job's stop switch (STOP in the job folder or the project's STOP) or the job's deadline
    puts STOP in the child's run folder, which its Callers and embedder check before every request."""

    def __init__(self, halt, box):
        super().__init__(daemon=True)
        self.halt, self.box, self.done, self.why = halt, box, threading.Event(), None

    def run(self):
        while not self.done.wait(POLL_S):
            why = self.halt()
            if why:
                self.why = self.why or why
                for c in self.box.iterdir():
                    if c.is_dir() and not (c / "STOP").exists():
                        (c / "STOP").write_text(why + "\n", encoding="utf8")


# ---- the cumulative statement artifact ------------------------------------------------------------------------------

def snapshot(src, dest):
    """cumulative/<statement>_v0: every top-level JSON file of the spread run, byte for byte."""
    dest.mkdir(parents=True, exist_ok=False)
    for p in sorted(Path(src).glob("*.json")):
        shutil.copyfile(p, dest / p.name)


def load_state(d):
    """A record as repair_run's --section reads it (repair_run.load_spread: amounts as Decimal), with claims, rows,
    diagnostics, c4 and the outcome."""
    from .repair_run import load_spread
    s = load_spread(d)
    for n in ("claims", "rows", "diagnostics", "c4", "outcome"):
        s[n] = read(d, n + ".json")
    return s


def dec_claims(claims):
    return [{**c, "amount": Decimal(str(c["amount"])), "signed": Decimal(str(c["signed"])),
             "facts": [(f, Decimal(str(k)), dp) for f, k, dp in c["facts"]]} for c in claims]


def dec_rows(rows):
    return {r: None if v is None else Decimal(str(v)) for r, v in rows.items()}


def bare(c):
    return {k: v for k, v in c.items() if k != ANNOTATION}


class Statement:
    """What code needs to recompute a statement's outputs, read once from the filing (spread.prepare, as
    repair_review reads it: no model call). `d`: the record's folder, whose organizer record (organizer.load) sets the
    sections' lines; for a run from before the organizer, code's routes (an organizer-era one without `d` is
    refused)."""

    def __init__(self, f, tax, meta, d=None):
        from . import organizer
        from .spread import prepare, setup
        t = template.active()
        ran = meta["section"] if isinstance(meta["section"], list) else [meta["section"]]
        first = {}
        for i, r in enumerate(load_rows()):
            first.setdefault(t.section_of(r["row_id"]), i)
        self.statement = t.statement_of(ran[0])
        if d is None and "organizer" in meta:
            raise organizer.Refused("an organizer-era record is replayed from its folder, never by code's routes")
        self.prep = prepare(f, tax, self.statement, sorted(ran, key=first.get), [], setup(f, self.statement),
                            organizer.load(d, f, tax, self.statement) if d else None)
        self.sections = self.prep["sections"]
        self.f, self.tax = f, tax


class ArmRecord(ValueError):
    """A version whose arm cannot be read from its record, or whose record contradicts its arm. Never read as T."""


def arm_of(run, diagnostics):
    """"R" or "T": the arm of a version, from the recorded source flag (run.json root_cause_fixes), never from the presence of the overlap record. An R version must hold its overlap
    record (diagnostics.open_claims, which spread writes for every R run); a T version holds none. Anything else
    raises ArmRecord: a lost record is not a T version."""
    flag = run.get(R_FLAG, False) if isinstance(run, dict) else None
    if not isinstance(flag, bool):
        raise ArmRecord(f"the recorded source flag ({R_FLAG} in run.json) cannot be read: the arm is not known")
    has = isinstance(diagnostics, dict) and isinstance(diagnostics.get("open_claims"), dict)
    if flag and not has:
        raise ArmRecord(f"an R version ({R_FLAG} in run.json) without its overlap record "
                        "(diagnostics.open_claims): refused, never read as T")
    if not flag and isinstance(diagnostics, dict) and "open_claims" in diagnostics:
        raise ArmRecord(f"a T version (no {R_FLAG} in run.json) that carries an overlap record "
                        "(diagnostics.open_claims): the record contradicts its arm")
    return "R" if flag else "T"


def expected_arm(args):
    """The arm a job's spread arguments ask for."""
    return "R" if "--root-cause-fixes" in list(args) else "T"


def assemble(s, st, positive=False):
    """The statement-level outputs recomputed from a record whose per-section claims, rows and checks are as given:
    spread.run's tail after C6, with no model. The section checks in order, "no amount claimed in two sections", the
    coverage check, section_unknown, the run-wide control ties; the computed totals, controls, report checks, the C6
    "control unavailable" exceptions, gap lines, negative rows, section outcomes and the verdict. Returns plain JSON.
    The B annotation (gap-match candidates, a model's) is carried onto a check whose content is unchanged and dropped,
    and listed, otherwise."""
    from .repair_run import section_signs
    from .spread import (FILED_SIGN, POST_CHECKS, computed_totals, control_states, coverage_check,
                         exceptions_summary, gap_lines, ident, post_checks, report_checks, section_unknown,
                         sign_exceptions, verdict)
    t, f, tax, prep, sections = template.active(), st.f, st.tax, st.prep, st.sections
    key, canonical = prep["key"], prep["canonical"]
    instant = t.period(st.statement) == "instant"
    diag = json.loads(json.dumps(s["diagnostics"]))
    diag.pop("negative_rows", None)
    claims, rows, recorded = dec_claims(s["claims"]), dec_rows(s["rows"]), s["checks"]
    checks, incomplete, gaps, unresolved_signs = [], {}, {}, []
    for sec in sections:
        # a run-wide control tie carries its section's name but is recomputed below, after every section
        # so is a check after them (template format v4, spread.post_checks)
        mine = [bare(c) for c in recorded if c.get("section") == sec and c["check"] != f"{sec} control ties"
                and c["check"] not in POST_CHECKS]
        checks += mine
        unres = next((c["lines"] for c in mine if c["check"] == "every line allocated"), [])
        partly = [c["check"] for c in mine if c["check"].endswith("fully allocated") and not c["ok"]]
        x = prep["st"][sec]
        if unres or partly or x["member"] or x["refused"]:
            incomplete[sec] = {"unallocated": sorted(unres), "partly_allocated": partly}
            if x["member"]:
                incomplete[sec]["printed_by_member_only"] = list(x["member"])
            if x["refused"]:
                incomplete[sec]["extraction_refused"] = list(x["refused"])
        ls = [l for l in s["lines"] if l.get("section") == sec]
        if ls:
            signs, _ = section_signs(s, ls, tax, f, sec, instant)
            unresolved_signs += [c for c in unres if signs.get(c) is None]
        recon = next((c for c in mine if c["check"].startswith("rows add to")), None)
        if recon and not recon["ok"] and recon.get("gap") is not None:
            gaps[sec] = gap_lines(ls, set(unres), Decimal(str(recon["gap"])), Decimal(str(recon["tolerance"])))
        neg = sorted(r for r, v in rows.items() if t.section_of(r) == sec and v is not None and v < 0)
        if instant and neg:
            diag.setdefault("negative_rows", {})[sec] = neg
    if len(sections) > 1:
        seen, twice = {}, []
        for c in claims:
            k = ident(c)
            if k in seen and seen[k] != c["section"]:
                twice.append({"amount": k, "sections": [seen[k], c["section"]]})
            seen.setdefault(k, c["section"])
        checks.append({"check": "no amount claimed in two sections", "ok": not twice, "section": "run",
                       "claimed_twice": twice})
    diag.pop("gap_lines", None)
    if gaps:
        diag["gap_lines"] = gaps[sections[0]] if len(sections) == 1 else gaps
    unknown = {}
    if prep["coverage"] is not None:
        chk, unknown = coverage_check(prep["coverage"], diag.get("queued_outcomes") or {}, unresolved_signs)
        checks.append(chk)
    checks += section_unknown(incomplete, [x for x in sections if t.sections[x]["kind"] == "spread"], unknown,
                              positive)
    totals = computed_totals(rows, incomplete)
    states = control_states(f, tax, key, st.statement, sections, rows, claims, checks, incomplete, canonical)
    exc = [x for x in s["exceptions"] if not (x.get("step") == "C6" and x.get("class") in ("control unavailable",
                                                                                            FILED_SIGN))]
    # an R job's record: an open claim's exception stays while its piece is committed with its mark, the totals
    # over a provisional row are marked again and the record of open claims is recomputed; no review clears a claim
    r7 = arm_of(s["run"], s["diagnostics"]) == "R"      # the recorded flag; ArmRecord when the record is lost
    if r7:
        exc, r7_dropped = open_claims.carried(exc, claims)
        r7_rows = open_claims.provisional_rows(claims)
        r7_totals = open_claims.mark_totals(t, totals, r7_rows)
    for x in states:
        if x["state"].startswith("unavailable"):
            exc.append({"step": "C6", "class": "control unavailable", "section": x["section"],
                        "why": f"control {x['state']}"})
        elif x["kind"] == "run-wide" and x["state"] in ("tied", "differs"):
            checks.append({"check": f"{x['section']} control ties", "ok": x["state"] == "tied",
                           "section": x["section"],
                           **{k: x[k] for k in ("filed", "filed_value", "built", "gap", "tolerance")}})
    exc += sign_exceptions(states, s["lines"], f)     # replaced, never added twice
    checks += post_checks(checks, claims, {sec: prep["st"][sec]["control"] for sec in sections}, canonical)
    checks = norm(checks)
    before = {(c["section"], c["check"]): c for c in norm(recorded)}
    dropped = []
    for c in checks:
        old = before.get((c["section"], c["check"]))
        if old and ANNOTATION in old:
            if bare(old) == c:
                c[ANNOTATION] = old[ANNOTATION]
            else:
                dropped.append([c["section"], c["check"]])
    controls = {k: [x for x in states if x["state"].split(":")[0] == k]
                for k in ("tied", "differs", "unavailable", "not evaluated", "failed")}
    placed = s["placements"]
    disputed = {placed[x["concept"]]["row"] for x in exc if x.get("class") == "disputed"
                and x.get("concept") in placed and placed[x["concept"]].get("by") not in ("C7", "analyst")}
    reports = report_checks(f, key, st.statement, sections, rows, claims, incomplete, canonical, disputed,
                            prep["variant"])
    error = s["outcome"].get("error")
    outcome = "not run" if error else verdict(checks, exc, reports)
    if len(sections) > 1:
        diag["section_outcomes"] = {
            x: verdict([c for c in checks if c["section"] in (x, "run")],
                       [y for y in exc if y.get("section") in (x, None)]) for x in sections}
    if r7:
        diag["open_claims"] = open_claims.resummary(s["diagnostics"]["open_claims"], claims, exc, r7_rows, r7_totals,
                                                    r7_dropped)
        diag["open_claims"]["qualification"] = open_claims.qualification(outcome, exc)
    summary = exceptions_summary(exc, checks, outcome, error)
    return norm({"checks.json": checks, "exceptions.json": exc, "computed_totals.json": totals,
                 "controls.json": controls, "report_checks.json": reports, "diagnostics.json": diag,
                 "outcome": outcome, "summary": summary, "annotations_dropped": dropped})


def identity(d, st):
    """The v0 check: assemble on the copied record reproduces the run's own statement-level outputs. Returns what
    differs (empty: the same). A version whose arm cannot be read, or that contradicts it (ArmRecord), differs by
    that reason and is not reassembled."""
    s = load_state(d)
    try:
        got = assemble(s, st)
    except ArmRecord as e:
        return [f"arm: {e}"]
    differs = [n for n in ("checks.json", "exceptions.json", "computed_totals.json", "controls.json",
                           "report_checks.json") if got[n] != norm(read(d, n))]
    want = read(d, "diagnostics.json")
    differs += [f"diagnostics.{k}" for k in ("gap_lines", "negative_rows", "section_outcomes", "open_claims")
                if got["diagnostics.json"].get(k) != want.get(k)]
    if arm_of(s["run"], want) == "R":                   # an R record: the rests carry their pieces' marks
        again = norm(s["claims"])
        open_claims.remainders(again, {l["concept"]: l["fact"] for l in s["lines"]})
        if again != norm(s["claims"]):
            differs.append("claims.json provisional marks")
    if got["outcome"] != s["outcome"].get("outcome"):
        differs.append("outcome")
    return differs


def failing(d, sections):
    """(run-level failed checks, [(section, failed checks)] in the given order) of a record: the checks that decide
    the outcome (spread.counted), so the tree never repairs a subtotal the template reports only."""
    from .spread import counted
    checks, _ = counted(read(d, "checks.json"), [])
    run = sorted(c["check"] for c in checks if not c["ok"] and c.get("section") == "run")
    secs = [(x, sorted(c["check"] for c in checks if not c["ok"] and c.get("section") == x)) for x in sections]
    return run, [(x, cs) for x, cs in secs if cs]


def precheck(d, section, f, tax):
    """None when the section may go to the tree, else why not: the stops repair_run makes before any model
    (section_draft's refusal, section_check), read with its own functions on the same record. A bad root (the
    draft's own errors, a code fault) is a harness defect, raised with its errors as `diagnostics`."""
    from .repair_run import section_check, section_draft
    s = load_state(d)
    ids = [r["row_id"] for r in section_rows(load_rows(), section)]
    try:
        draft, _ = section_draft(s, f, tax, section, ids)
    except ValueError as e:
        return str(e)
    errs = repair.draft_errors(draft)
    if errs:
        e = Defect(f"precheck {section}: bad root: " + "; ".join(errs))
        e.diagnostics = errs
        raise e
    chk = section_check(draft, s, section)
    why = [w for bad, w in (
        (not chk["root"]["same"], "root differs from the record's checks"),
        (not chk["reconstruction"]["same"], f"reconstruction differs: {sorted(chk['reconstruction']['differs'])}"),
        (not chk["unplaced"]["same"], "unplaced lines differ from the record's")) if bad]
    why += [f"structurally ineligible: {w}" for w in chk["structural"]["why"]]
    return "; ".join(why) or None


def registry_of(raw):
    """A registry read back from draft_out.json (amounts written as strings)."""
    return {"alloc": {c: dict(p) for c, p in raw["alloc"].items()},
            "pieces": [{**p, "amount": Decimal(str(p["amount"]))} for p in raw["pieces"]],
            "unresolved": dict(raw["unresolved"])}


def answer(s, c):
    """A line's answer in a record: its claims [row, fact, signed], sorted, or unplaced with the recorded reasons."""
    if c in s["placements"]:
        return {"claims": sorted([x["row"], x["fact"], str(x["signed"])] for x in s["claims"] if x["source"] == c)}
    why = [f"{x['step']} {x['class']}" for x in s["exceptions"] if x.get("concept") == c]
    return {"unplaced": "; ".join(why) or "not placed"}


def refused(rec, why):
    return None, {**rec, "result": "refused at commit", "why": why}


def rebuilt_section(s, st, section, draft, new, reg, raw, recorded):
    """(s_new, kept): the record `s` with one section's claims, checks and rows rebuilt by code from registry `reg`
    (C5 and C6, repair.rebuild on `new`, the draft carrying it). Shared by commit (a C7 repair) and analyst_edits
    (an analyst's edits); no model and no review result is read. `raw`: the current version's claims.json,
    checks.json and rows.json; `recorded`: its arm."""
    rows, claims, checks = repair.rebuild(new, reg)
    # the rebuild keeps a recorded piece's fact, row and amount only: its recorded claims, marks and evidence are
    # taken back from the current version (either arm). No review result is read: C8's agreement clears no claim.
    line_fact = {l["concept"]: l["fact"] for l in draft["lines"]}
    kept = open_claims.inherit([c for c in raw["claims.json"] if c.get("section") == section], claims, line_fact)
    if recorded == "R":                                             # an R record: the rest of a marked piece's line
        open_claims.remainders(claims, line_fact)
    of = lambda xs, x: [c for c in xs if c.get("section") == x]   # noqa: E731
    s_new = dict(s)
    s_new["claims"] = norm([c for x in st.sections for c in (
        [{**c, "section": section} for c in claims] if x == section else of(raw["claims.json"], x))])
    s_new["checks"] = norm([c for x in st.sections for c in (
        [{**c, "section": section} for c in checks] if x == section else of(raw["checks.json"], x))])
    s_new["rows"] = {**raw["rows.json"], **norm({r: rows[r] for r in draft["ids"] if r in rows})}
    return s_new, kept


def commit(cur, section, rdir, dest, st, version, arm=None):
    """Apply a committed repair to a new cumulative version `dest`. Returns (dest or None, record). Refused, with
    nothing written, when the repair's input is not the current version, the committed registry fails
    repair.draft_errors against the current version (an amount another section owns included), or a changed line
    has no settled review. Refused too when the current version's arm cannot be read from its record (arm_of: an
    R version without its overlap record is never rebuilt as T), or is not `arm`, the job's own, when given."""
    from .repair_run import section_draft
    rec = {"section": section, "repair_run": str(rdir), "from": Path(cur).name}
    s = load_state(cur)
    try:
        recorded = arm_of(s["run"], s["diagnostics"])
    except ArmRecord as e:
        return refused(rec, f"arm: {e}")
    if arm is not None and recorded != arm:
        return refused(rec, f"arm: the version records arm {recorded} and the job's arm is {arm}")
    ids = [r["row_id"] for r in section_rows(load_rows(), section)]
    draft, _ = section_draft(s, st.f, st.tax, section, ids)
    din, dout = read(rdir, "draft_in.json"), read(rdir, "draft_out.json")
    if norm(draft["registry"]) != norm(din["registry"]) or norm(draft["lines"]) != norm(din["lines"]):
        return refused(rec, "the repair's input is not the current version")
    reg = registry_of(dout["registry"])
    new = {**draft, "registry": reg}
    # I5 on preservation is read against the version the repair started from: draft_errors(new) reads the
    # recorded pieces from the registry it is given, so a draft that leaves one out would pass it. Both arms.
    errs = repair.draft_errors(new) or [f"{i}: {w}" for i, w in repair.registry_errors(draft, reg) if i == "I5"]
    if errs:
        return refused(rec, errs)
    changed = [l["concept"] for l in draft["lines"] if not repair.fixed_line(l)
               and norm(repair.placement(draft["registry"], l["concept"])) != norm(repair.placement(reg, l["concept"]))]
    reviews = read(rdir, "c8_review.json")["reviews"] if (Path(rdir) / "c8_review.json").is_file() else []
    settled = {r["line"]: r for r in reviews if r.get("status") == repair.SETTLED and r.get("section") == section}
    if not changed:
        return refused(rec, "no line changed")
    unsettled = [c for c in changed if c not in settled]
    if unsettled:
        return refused(rec, f"changed lines without a settled review: {unsettled}")
    raw = {n: read(cur, n) for n in ("claims.json", "checks.json", "rows.json", "placements.json", "c4.json")}
    missing = [c for c in changed if c not in raw["c4.json"]]
    if missing:
        return refused(rec, f"not in c4.json: {missing}")
    before = {c: answer(s, c) for c in changed}
    s_new, kept = rebuilt_section(s, st, section, draft, new, reg, raw, recorded)
    placed = dict(raw["placements.json"])
    for c in changed:
        old = placed.pop(c, None)
        if c in reg["alloc"]:
            placed[c] = {"row": reg["alloc"][c]["row"], "by": "C7", "repair_run": Path(rdir).name, "before": old}
    s_new["placements"] = placed
    c4 = raw["c4.json"]
    for c in changed:
        r, e = settled[c], c4[c]
        target = e["transfer"] if (r.get("fidelity") or {}).get("episode") == "transfer" else e
        target["review_history"] = target.get("review_history", []) + [target.get("review")]
        target["review"] = r.get("checker")
        target["consistency"] = {"result": "settled (C7 repair, C8 review)", "before": target.get("consistency"),
                                 "repair_run": Path(rdir).name}
    s_new["diagnostics"] = {**s["diagnostics"], "c7": s["diagnostics"].get("c7", []) + [
        {"version": version, "section": section, "repair_run": str(rdir), "lines": changed}]}
    got = assemble(s_new, st)
    after = {c: answer(s_new, c) for c in changed}
    dest.mkdir(parents=True, exist_ok=False)
    for p in sorted(Path(cur).glob("*.json")):
        if p.name in ("lines.json", "run.json", "search.json", "unallocated.json", "calls.json", "organizer.json"):
            shutil.copyfile(p, dest / p.name)
    summary = {**got["summary"], "c7": s_new["diagnostics"]["c7"]}
    for name, obj in (("claims.json", s_new["claims"]), ("rows.json", s_new["rows"]), ("placements.json", placed),
                      ("c4.json", c4), ("checks.json", got["checks.json"]),
                      ("exceptions.json", got["exceptions.json"]), ("computed_totals.json", got["computed_totals.json"]),
                      ("controls.json", got["controls.json"]), ("report_checks.json", got["report_checks.json"]),
                      ("diagnostics.json", got["diagnostics.json"]),
                      ("outcome.json", {**s["outcome"], "outcome": got["outcome"]}),
                      ("exceptions_summary.json", summary),
                      ("cumulative.json", {
                          "version": version, "parent": Path(cur).name, "section": section, "repair_run": str(rdir),
                          "changed": changed, "annotations_dropped": got["annotations_dropped"],
                          **({"piece_records_kept": kept} if kept else {}),
                          "label": "a cumulative statement artifact: the spread run's record with the committed "
                                   "repairs applied in order; claims, totals, coverage, controls and checks "
                                   "recomputed by code"})):
        write(dest, name, json.dumps(obj, indent=1, default=str))
    chain = [{"line": c, "mechanism": "C7 tree repair, committed after a settled C8 review", "version": version,
              "repair_run": Path(rdir).name, "before": before[c], "after": after[c]} for c in changed]
    return dest, {**rec, "result": "committed", "to": dest.name, "changed": changed, "chain": chain,
                  "outcome": got["outcome"]}


def chains(v0, commits):
    """Each line's recorded transitions, in order: the spread's answer (a tiebreaker replacement as two entries, C4's
    answer then the vote's; a kept second look as C4's answer then the retry's, same or changed, then any vote), then
    each committed repair. A record, not causal attribution. A line C4 recorded after the statement's first tiebreaker
    replacement or kept second look carries CHAIN_LABEL (c4.json's order, the order C4 took the lines)."""
    s = {n: read(v0, n + ".json") for n in ("lines", "placements", "claims", "exceptions", "c4")}
    votes = {x["concept"]: x for x in s["exceptions"] if x.get("step") == "C8" and "vote" in x}
    looks = {x["concept"]: x for x in s["exceptions"] if x.get("step") == "C8" and "feedback_path" in x}
    order = list(s["c4"])
    replaced = [c for c in order if (s["placements"].get(c) or {}).get("by") == "vote"
                or (looks.get(c) or {}).get("kept") == "retry"]
    first = order.index(replaced[0]) if replaced else None
    out = {}
    for l in s["lines"]:
        if repair.fixed_line(l):
            continue
        c, p = l["concept"], s["placements"].get(l["concept"]) or {}
        now, v = answer(s, c), votes.get(c)
        x = looks.get(c) or {}
        if x.get("kept") == "retry":        # second look: the analyst's retry with both answers as candidates stood
            entries = [{"mechanism": "C4 analyst, disputed by the checker" if x["feedback_path"] != "unplaced"
                        else "C4 analyst: no row; the checker placed one", "before": None, "after": x["first"]},
                       {"mechanism": "a second look: the analyst's one retry with both answers stood, "
                                     + ("the same answer" if x["after"] == x["first"] else "a changed answer"),
                        "before": x["first"], "after": x["after"]}]
            if v:
                entries.append({"mechanism": f"tiebreaker vote: {v['vote']}"
                                + ("; the checker's answer replaced the retry's" if p.get("by") == "vote" else ""),
                                "before": x["after"], "after": now})
        elif p.get("by") == "vote":
            entries = [{"mechanism": "C4 analyst, disputed by the checker", "before": None, "after": v and v["c4"]},
                       {"mechanism": "tiebreaker: agreed with the checker; the checker's answer replaced C4's",
                        "before": v and v["c4"], "after": now}]
        else:
            entries = [{"mechanism": f"spread ({p.get('by') or 'not placed'})", "before": None, "after": now}]
            if v:
                entries.append({"mechanism": f"tiebreaker vote: {v['vote']}", "before": now, "after": now})
        for x in commits:
            entries += [e for e in x["chain"] if e["line"] == c]
        out[c] = {"section": l.get("section"), "transitions": entries,
                  "label": CHAIN_LABEL if first is not None and c in s["c4"] and order.index(c) > first else None}
    return {"note": CHAIN_NOTE, "lines": out}


# ---- the job ------------------------------------------------------------------------------------------------------

class End(Exception):
    """A job's end other than finished, with what triggered it for stop.json (`detail`: the child, role, request,
    HTTP status or cap, as far as known) and when it was raised."""

    def __init__(self, why, detail=None):
        super().__init__(why)
        self.detail, self.time = detail or {}, time.strftime("%Y-%m-%dT%H:%M:%S")


class Capped(End):
    """The company's limits or the job's deadline are reached: the job stops, every record kept."""


class Stopped(End):
    """The job's stop switch: the active child is stopped and no later child starts."""


class Defect(End):
    """A harness defect: a step that could not start, left no record, does not reconcile, or a changed source. The
    job stops, every record kept, and the command exits unsuccessfully."""


class Incomplete(End):
    """A child's caller failed at the transport level: the job stops, every record kept, nothing is graded, and the
    command exits unsuccessfully."""


class Progress:
    """The caller's progress callback, observational only: each event carries the job's ID, its stage and status, and
    counters named with their units. A callback that raises is recorded once and dropped; the job's own records never
    depend on it."""

    def __init__(self, fn, job_id):
        self.fn, self.job_id, self.error = fn, job_id, None

    def __call__(self, stage, status, **fields):
        if self.fn is None:
            return
        event = {"job_id": self.job_id, "stage": stage, "status": status, **fields}
        try:
            self.fn(event)
        except Exception as e:
            self.fn = None
            self.error = {"event": event, "error": repr(e), "traceback": traceback.format_exc()[-3000:],
                          "then": "the callback was dropped; the job went on"}


class Job:
    def __init__(self, d, cfg, lineup, a, clock, start, children, f, tax, allocation=None, stop=None, progress=None):
        self.d, self.cfg, self.lineup, self.a, self.clock = d, cfg, lineup, a, clock
        self.children, self.f, self.tax = children, f, tax
        self.company = Company(cfg["company_limits"], a.dollars, clock, start)
        self.pays = lineup.get("pays", False)
        self.skips, self.commits, self.statements, self.sources = [], [], {}, {}
        self.order, self.end, self.grade, self.attempts, self.n = None, None, None, 0, 0
        self.allocation, self.stop, self.template, self.corrections = allocation, stop, None, None
        self.tell = progress or Progress(None, d.name)
        self.ledger = open_ledger(d, self.company, allocation) if getattr(children, "shared", False) else None

    def log(self, rec):
        with open(self.d / "steps.jsonl", "a", encoding="utf8", newline="\n") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")

    def counters(self):
        return {"children_started": self.n, "statements_spread": sum("spread" in x for x in self.statements.values()),
                "statements_total": len(self.cfg["statements"]), "tree_attempts": self.attempts,
                "tree_attempts_most": self.cfg["tree"]["max_sections"]}

    def halt(self):
        """Why the job must stop now: its stop switch (STOP in its folder, or its caller's stop signal) or its
        deadline; None otherwise."""
        if (self.d / "STOP").exists() or (self.stop is not None and self.stop.is_set()):
            return "job STOP"
        if STOP_FILE.exists():
            return "project STOP"
        if self.company.left()["wall_s"] <= 0:
            return "job deadline"
        return None

    def keep(self, folder):
        """A finished child's folder or a written cumulative version: never changed afterwards."""
        self.sources[str(folder)] = fingerprint(folder)

    def changed(self):
        return sorted(p for p, fp in self.sources.items() if fingerprint(p) != fp)

    def check(self, label):
        """Before every child: an unchanged source, no stop, no deadline, company limits left."""
        moved = self.changed()
        if moved:
            raise Defect(f"before {label}: a source changed after it was recorded: {moved}")
        why = self.halt()
        if why == "job deadline":
            raise Capped(f"before {label}: the job's deadline ({self.company.limits['wall_s']} s) is reached",
                         {"before": label, "cap": "wall_s"})
        if why:
            raise Stopped(f"before {label}: {why}", {"before": label, "cap": why})
        gone = self.company.exhausted(self.pays)
        if gone:
            raise Capped(f"company limits reached before {label}: {gone}", {"before": label, "cap": gone})
        if self.ledger is not None and self.ledger.state()["stopped"]:
            raise Capped(f"the company ledger stopped before {label}: {self.ledger.state()['stopped']}",
                         {"before": label, "cap": self.ledger.state()["stopped"]})

    def step(self, kind, argv, own, label):
        """Run one child under the company's limits; returns (folder, usage, record). The child's start is logged
        and its run folder's parent made before it can send anything; its end is logged however it ends."""
        self.check(label)
        caps, cut = self.company.caps(own, self.pays)
        argv = argv + ["--max-calls", str(caps["requests"]), "--max-tokens", str(caps["tokens"]),
                       "--wall-s", str(caps["wall_s"])]
        if self.pays and kind == "tree" and self.allocation:     # a hosted repair: its section's allocation
            caps["section_dollars"] = self.allocation["section"]["spend_usd"]
            if caps["section_dollars"] < caps["dollars"]:
                caps["dollars"] = caps["section_dollars"]
                cut = [k for k in cut if k != "dollars"]    # the step's dollar limit is the section's, not the company's
        if self.pays:
            argv += ["--spend-usd", str(caps["dollars"])]      # the company's dollars left, or the section's
        if self.a.probe:
            argv += ["--probe", self.a.probe]
        self.n += 1
        box = self.d / "children" / f"{self.n:02d}_{kind}"
        box.mkdir(parents=True)
        rec = {"kind": kind, "label": label, "argv": argv, "caps": caps, "company_cut": cut, "box": str(box)}
        self.log({**rec, "event": "start"})
        self.tell(kind, "started", label=label, folder=str(box), counters=self.counters())
        env = os.environ.get("SPREADING_RUNS_DIR")
        os.environ["SPREADING_RUNS_DIR"] = str(box)      # the child's run folder is made in the box
        watch = Watch(self.halt, box)
        watch.start()
        t0, out, err = self.clock(), None, None
        try:
            out = (self.children.spread if kind == "spread" else self.children.repair)(argv, self.ledger)
        except BaseException as e:      # recorded below, then raised again as the job's end
            err = e
        finally:
            watch.done.set()
            watch.join()
            if env is None:
                os.environ.pop("SPREADING_RUNS_DIR", None)
            else:
                os.environ["SPREADING_RUNS_DIR"] = env
        made = sorted(p for p in box.iterdir() if p.is_dir())
        folder = Path(out) if out is not None else made[0] if len(made) == 1 else None
        rec.update(event="end", seconds=round(self.clock() - t0, 3), folder=str(folder) if folder else None,
                   stopped_by=watch.why)
        use = self.children.usage(folder) if folder is not None else None
        if err is not None or use is None:
            if use is None and folder is not None:
                use = salvage(folder)
            if use is None and folder is None and isinstance(err, SystemExit):
                use = {k: 0 for k in self.company.used}     # refused before the run: no folder, nothing sent
            self.company.add(use, label)
            rec.update(usage=use, company=self.company.state(), error=repr(err) if err is not None else None,
                       result="interrupted" if isinstance(err, KeyboardInterrupt) else
                       "refused before the run" if isinstance(err, SystemExit) and folder is None else
                       "raised" if err is not None else "left no outcome.json")
            self.log(rec)
            self.tell(kind, "ended", label=label, result=rec["result"], counters=self.counters())
            if folder is not None:
                self.keep(folder)
            if isinstance(err, KeyboardInterrupt):
                raise err
            raise Defect(f"{label}: {rec['result']}" + (f" ({err!r})" if err is not None else f" in {folder}"))
        self.company.add(use, label)
        self.keep(folder)
        lim = use["limiting"]
        rec["capped"] = bool(lim and lim["kind"] == "limit") or use.get("termination") == "cap_reached"
        gone = self.company.exhausted(self.pays)
        rec["company_capped"] = bool(rec["capped"] and lim and (lim["ledger"] == "company" or lim["budget"] in cut)) \
            or bool(gone)
        rec.update(usage=use, company=self.company.state())
        fails = use.get("caller_failures") or []
        hit = worst(fails)
        if hit:
            end, reason = failure_end(hit)
            rec["caller_failure"] = {"child": label, "folder": str(folder), **{k: hit.get(k) for k in FAILURE_FIELDS},
                                     "end": end, "reason": reason, "failures": [   # every simultaneous failure, kept
                                         {**{k: x.get(k) for k in FAILURE_FIELDS}, "why": x.get("why"),
                                          "end": failure_end(x)[0], "reason": failure_end(x)[1]} for x in fails]}
        if self.ledger is not None:
            rec["company_ledger"] = got = charged(self.ledger)
            mine = {"requests": self.company.used["requests"], "dollars": self.company.used["dollars"]}
            rec["company_ledger_matches"] = got["requests"] == mine["requests"] and \
                abs(got["dollars"] - mine["dollars"]) < 1e-9
        if rec["company_capped"]:
            rec["why_capped"] = (f"{label} stopped at {lim['budget']} ({lim['reason']}), a limit the company set"
                                 if rec["capped"] and lim else f"the company's limits are reached: {gone}")
        self.log(rec)
        self.tell(kind, "ended", label=label, result=use["outcome"], counters=self.counters())
        if not use["reconciliation"]["ok"]:
            raise Defect(f"{label}: its requests do not reconcile: {use['reconciliation']['why']}")
        if rec.get("company_ledger_matches") is False:
            raise Defect(f"{label}: the company ledger charged {rec['company_ledger']}, the children's own ledgers "
                         f"sum to {mine}")
        return folder, use, rec

    def after(self, rec):
        """After a step's record is kept (its snapshot or commit written): the job's end, if the step set one."""
        child = {"child": rec["label"], "folder": rec.get("folder"),
                 "failures": (rec.get("caller_failure") or {}).get("failures")}
        if rec.get("stopped_by") == "job deadline":
            raise Capped(f"{rec['label']}: the job's deadline stopped it", {**child, "cap": "wall_s"})
        if rec.get("stopped_by"):
            raise Stopped(f"{rec['label']}: {rec['stopped_by']} stopped it", {**child, "cap": rec["stopped_by"]})
        f = rec.get("caller_failure")
        if f and f["end"] != "capped":
            raise {"incomplete": Incomplete, "stopped": Stopped, "harness defect": Defect}[f["end"]](
                f"{rec['label']}: its {f['role']} request {f['tag']} ended on a {f['reason']} (kind {f['kind']!r}, "
                f"HTTP status {f['http_status']}, dispatch {f['dispatch']}, attempt {f['attempt']})", f)
        if rec["company_capped"]:   # a ledger refusal ends the job here; only at the child's own limit, it goes on
            lim = rec["usage"].get("limiting") or {}
            raise Capped(rec["why_capped"], {**child, **(f or {}), "cap": lim.get("budget") or rec["why_capped"],
                                             "role": lim.get("limit") if lim.get("kind") == "role" else
                                             (f or {}).get("role")})

    def run(self):
        """The job's stages in order: each statement's spread, the tree's order, then the tree on each section of
        that order. Finish, the grade and the job's records follow in run_job."""
        t = template.active()
        self.template = {"path": str(t.path), "sha256": hashlib.sha256(
            json.dumps(t.data, sort_keys=True).encode("utf8")).hexdigest(), "hash_of": "the content, keys sorted"}
        if self.cfg.get("named_totals"):
            self.correct()
        for own in self.cfg["statements"]:
            self.spread_statement(t, own)
        self.tree_order()
        for o in self.order:
            self.tree_section(o)

    def correct(self):
        """The filing-corrections step, named totals (named_totals.run), once, before any spread. Its record is
        written in its own child folder, which is then kept (fingerprinted); its sha256 is sealed here, before the first
        organizer send, and passed to every spread (--corrections); the job's filing is the corrected one from here on.
        Its sends and tokens count in the company's totals, and its caps are cut to what the company has left. The
        step's own caps end it capped and the job goes on; the job's stop or deadline ends the job."""
        from . import corrections, named_totals as step
        self.check("corrections")
        left = self.company.left()
        self.n += 1
        box = self.d / "children" / f"{self.n:02d}_corrections"
        out = box / "corrections"
        out.mkdir(parents=True)
        rec = {"kind": "corrections", "label": "corrections", "box": str(box),
               "caps": {"requests": min(step.MAX_SENDS, left["requests"]),
                        "tokens": min(step.MAX_TOKENS, left["tokens"]), "wall_s": step.WALL_S}}
        self.log({**rec, "event": "start"})
        self.tell("corrections", "started", label="corrections", folder=str(box), counters=self.counters())
        watch = Watch(self.halt, box)
        watch.start()
        t0, err, got = self.clock(), None, None
        # the step's dollars are what the company ledger charged it (its spend before and after), as a spread's are;
        # its call log has no cost for an attempt whose use is unknown, which the ledger counts in full
        before = charged(self.ledger) if self.ledger is not None else None
        prof = hosted_profile(self.lineup)      # a hosted job: named totals on the profile's model too
        try:
            # each send charged in the company ledger too, under the company's key, as a spread's are (spread.main)
            shared = {} if self.ledger is None else {"ledger": self.ledger,
                                                     "company": Path(self.cfg["filing"]).parent.name}
            if prof:
                from .spread import borrowed
                shared.update(borrowed(prof, "name", step.ANSWER_CAP, left["dollars"]))
            got = step.run(self.f.dir, self.tax, out, stop_file=out / "STOP",
                                  deadline=time.monotonic() + max(0.0, left["wall_s"]),
                                  max_sends=rec["caps"]["requests"], token_cap=rec["caps"]["tokens"], **shared)
        except BaseException as e:      # recorded below, then raised again as the job's end
            err = e
        finally:
            watch.done.set()
            watch.join()
        log = out / "named_totals_calls.jsonl"
        calls = read_jsonl(log) if log.is_file() else []
        tokens = step.tokens_used(log)
        logged = sum(r.get("cost_usd") or 0.0 for r in calls)
        after = charged(self.ledger) if before is not None else None
        use = {"requests": got["sends"] if got else len(calls), "tokens": tokens,
               "hosted_unknown": after["unknown_billing"] - before["unknown_billing"] if after else 0,
               **({"hosted_tokens": tokens, "local_tokens": 0} if prof else {"local_tokens": tokens}),
               "dollars": after["dollars"] - before["dollars"] if after else logged, "dollars_logged": logged}
        self.company.add(use, "corrections")
        rec.update(event="end", seconds=round(self.clock() - t0, 3), usage=use, stopped_by=watch.why,
                   error=repr(err) if err is not None else None, ending=got["ending"] if got else None)
        self.keep(box)
        self.log(rec)
        self.tell("corrections", "ended", label="corrections", result=rec["ending"] or rec["error"],
                  counters=self.counters())
        if err is not None:
            why = watch.why or self.halt()
            if isinstance(err, KeyboardInterrupt):
                raise err
            if why == "job deadline":
                raise Capped(f"corrections: the job's deadline stopped it ({err})", {"child": "corrections", "cap": "wall_s"})
            if why:
                raise Stopped(f"corrections: {why} stopped it ({err})", {"child": "corrections", "cap": why})
            raise Defect(f"corrections: raised {err!r}")
        path = out / step.RECORD
        where = path.resolve()
        self.corrections = {"record": where.relative_to(ROOT.resolve()).as_posix() if ROOT.resolve() in where.parents
                            else str(where), "sha256": step.sha(path), "ending": got["ending"],
                            "retags": [{k: r[k] for k in ("statement", "line", "label", "old", "new", "applied", "why")}
                                       for r in got["retags"]]}
        self.f = corrections.open_filing(self.f.dir, self.corrections)

    def spread_statement(self, t, own):
        """One statement's spread (its analyst, checker and tiebreaker run inside the child), kept as version 0. A
        statement refused at setup or by the organizer is recorded refused, with no version, and the job goes on; its
        child's records are kept and its ends still apply. An organizer failure of a counted kind (organizer.failed)
        that makes more than one over the queue (the settings' `organizer_failures_before`, from job_queue) stops the
        job at once (Stopped)."""
        from . import organizer
        from .spread import setup
        cfg, lu, cum = self.cfg, self.lineup, self.d / "cumulative"
        name = own["statement"]
        try:
            sections = t.run_sections(name, setup(self.f, name)[2])
        except RuntimeError as e:       # a recognized setup refusal; anything else stays a harness fault
            if not str(e).startswith(SETUP_REFUSALS):
                raise
            self.statements[name] = {"sections": None, "versions": [], "outcome": "not run",
                                     "refused": {"refused_at": "setup", "kind": str(e).split(":")[0], "reason": str(e)}}
            return
        argv = [cfg["filing"], "--section", *sections, *cfg["spread_args"], *lu.get("spread_args", []),
                "--num-ctx", str(lu.get("num_ctx", own["num_ctx"]))]     # a hosted line-up sets its own context
        if self.corrections:            # the sealed record, passed explicitly to the child
            argv += ["--corrections", self.corrections["record"], "--corrections-sha256", self.corrections["sha256"]]
        x = self.statements[name] = {"sections": sections, "versions": []}
        out, use, rec = self.step("spread", argv, own, f"spread {name}")
        x.update(spread=str(out), outcome=use["outcome"], capped=rec["capped"])
        refusal = read(out, "outcome.json").get("refusal")
        if refusal:
            x["refused"] = refusal
            self.after(rec)
            gone = [*self.cfg.get("organizer_failures_before", []),
                    *(n for n, s in self.statements.items() if organizer.failed(s.get("refused")))]
            if len(gone) > 1:       # known useless: stopped before any further model work
                raise Stopped(f"after spread {name}: the organizer failed on {len(gone)} statements over the queue "
                              f"({', '.join(gone)})", {"child": rec["label"], "cap": "organizer failures"})
            return
        if use["outcome"] == "not run" and not rec["capped"] and not rec["stopped_by"] and \
                not rec.get("caller_failure"):
            raise Defect(f"spread {name}: not run ({read(out, 'outcome.json').get('error')})")
        v0 = cum / f"{name}_v0"
        snapshot(out, v0)
        write(v0, "cumulative.json", json.dumps({"version": 0, "source": str(out), "parent": None,
                                                 "label": "the spread run's record, copied"}, indent=1))
        self.keep(v0)
        x["versions"].append(v0.name)
        want = expected_arm([*cfg["spread_args"], *lu.get("spread_args", [])])
        flag = read(v0, "run.json").get(R_FLAG, False)
        if flag is not (want == "R"):       # the record is not of the arm the job asked for: a harness defect
            raise Defect(f"spread {name}: run.json records {R_FLAG} {flag!r} and the job's arm is {want}")
        asked = "--statements-only" in [*cfg["spread_args"], *lu.get("spread_args", [])]
        if bool(read(v0, "run.json").get("statements_only")) is not asked:  # the tools it ran with, as recorded
            raise Defect(f"spread {name}: run.json records statements_only "
                         f"{bool(read(v0, 'run.json').get('statements_only'))} and the settings ask {asked}")
        try:
            x["statement"] = Statement(self.f, self.tax, read(v0, "run.json"), v0)
            x["identity"] = identity(v0, x["statement"])
        except organizer.Refused as e:  # ended before its organizer record was written: nothing to reassemble
            if use["outcome"] != "not run":
                raise
            x["identity"] = [str(e)]
        self.after(rec)                 # a partial record is kept, copied, and never repaired

    def tree_order(self):
        """Each statement's failing sections, in the settings' order of statements."""
        cfg, cum = self.cfg, self.d / "cumulative"
        self.order = []                         # the tree's order, fixed and written before its first step
        for own in cfg["statements"]:
            name = own["statement"]
            x = self.statements[name]
            if x.get("refused"):
                continue
            run, secs = failing(cum / x["versions"][-1], x["sections"])
            if run:
                self.skips.append({"statement": name, "section": "run", "failed": run,
                                   "why": "run-level checks; the tree repairs one section"})
            self.order += [{"statement": name, "section": sec, "failed": cs} for sec, cs in secs]
        write(self.d, "order.json", json.dumps(self.order, indent=1))

    def tree_section(self, o):
        """The tree on one section of the order, or its recorded skip; a repair is committed as the next version."""
        cfg, lu, cum, tree = self.cfg, self.lineup, self.d / "cumulative", self.cfg["tree"]
        name, sec = o["statement"], o["section"]
        x = self.statements[name]
        cur = cum / x["versions"][-1]
        why = (CAP_REACHED if self.attempts >= tree["max_sections"] else
               "its spread stopped at its own limit; the record is partial" if x["capped"] else
               f"its record could not be reassembled by code: {x['identity']}" if x["identity"] else
               None if failing(cur, [sec])[1] else "its checks pass in the current version")
        try:
            why = why or precheck(cur, sec, self.f, self.tax)
        except Defect as e:             # the diagnostics first, then the job's end
            self.log({"kind": "precheck", **o, "from": cur.name, "bad_root": getattr(e, "diagnostics", None)})
            raise
        if not tree.get("on", True):    # after the precheck, so a bad root still ends the job as it would with it on
            why = f"{TREE_OFF}; {why}" if why else TREE_OFF
        if why:
            self.skips.append({**o, "why": why})
            return
        self.attempts += 1
        argv = [str(cur), "--section", sec, "--review", *tree.get("args", []), *lu.get("tree_args", []),
                *tree_dispatches(cfg, lu)]
        argv += self.role_args(cum / x["versions"][0], argv, f"tree {sec}")
        rdir, use, rec = self.step("tree", argv, tree, f"tree {sec}")
        res = {**o, "repair_run": str(rdir), "outcome": use["outcome"], "termination": use["termination"],
               "from": cur.name}
        if use["outcome"] == "bad root":   # a code fault in the draft the child built: its folder is kept
            cases = read(rdir, "exceptions.json").get("cases", {}) if (rdir / "exceptions.json").is_file() else {}
            self.log({"kind": "tree result", **res, "bad_root": cases.get("C7 bad root")})
            raise Defect(f"tree {sec}: bad root, a code fault in the draft (diagnostics in {rdir})")
        error = (read(rdir, "outcome.json").get("error") or {}).get("error")
        if use["outcome"] == "not run" and error and not rec["capped"] and not rec["stopped_by"] and \
                not rec.get("caller_failure") and not str(error).startswith("CapReached("):
            self.log({"kind": "tree result", **res, "error": error})
            raise Defect(f"tree {sec}: the command failed ({error})")
        if use["outcome"] == "repaired" and (rec.get("caller_failure") or {}).get("end", "capped") != "capped":
            res["not_applied"] = "its caller failed; the job ends and the repair stays in the child's folder"
        elif use["outcome"] == "repaired" and not rec["stopped_by"]:
            got, c = commit(cur, sec, rdir, cum / f"{name}_v{len(x['versions'])}", x["statement"],
                            len(x["versions"]), expected_arm([*cfg["spread_args"], *lu.get("spread_args", [])]))
            res["commit"] = {k: v for k, v in c.items() if k != "chain"}
            if got is not None:
                self.keep(got)
                x["versions"].append(got.name)
                self.commits.append(c)
        self.log({"kind": "tree result", **res})
        self.after(rec)

    def role_args(self, v0, argv, label):
        """The allocations a repair needs: --role-limit for each hosted tree role when the tree runs hosted, and for
        the checker when the statement's spread ran its checker on OpenRouter (the review asks the source's checker).
        Needed and missing is a harness defect: the line-up was let through without its file."""
        from .repair_run import HOSTED_TREE_ROLES
        roles = [*HOSTED_TREE_ROLES] if "--hosted-profile" in argv else []
        if (read(v0, "run.json").get("engines") or {}).get("checker") == "openrouter":
            roles.append("checker")
        if not roles:
            return []
        if self.allocation is None:
            raise Defect(f"{label}: {roles} run on OpenRouter and need the frozen allocation; the line-up has none")
        lim = self.allocation["section"]["roles"]
        short = [r for r in roles if r not in lim]
        if short:
            raise Defect(f"{label}: the frozen allocation's section has no limit for {short}")
        return [x for r in roles for x in ("--role-limit", r, str(lim[r]["calls"]), str(lim[r]["spend_usd"]))]

    def finish(self):
        """The final versions, the sources' check and the grade. A changed source is a harness defect: nothing is
        graded. A refused statement has no final version, and the grade counts its frame lines as blank (no run)."""
        cum = self.d / "cumulative"
        finals = {n: str(cum / x["versions"][-1]) for n, x in self.statements.items() if x["versions"]}
        refused = [n for n, x in self.statements.items() if x.get("refused")]
        moved = self.changed()
        for x in self.statements.values():
            if "spread" in x:
                x["unchanged"] = x["spread"] not in moved
        if moved:
            self.grade = {"not run": f"a source changed after it was recorded: {moved}"}
            raise Defect(f"a source changed after it was recorded: {moved}")
        g = self.cfg.get("grade")
        if g and self.a.no_grade:
            self.grade = {"not run": "--no-grade"}
        elif g and self.end == "harness defect":
            self.grade = {"not run": "the job ended on a harness defect"}
        elif g and self.end == "incomplete":
            self.grade = {"not run": "the job ended incomplete"}
        elif g and len(finals) + len(refused) < len(self.cfg["statements"]):   # a refused one is graded blank
            self.grade = {"not run": "a statement has no final artifact"}
        elif g:
            from . import grade
            here = lambda p: str(p if Path(p).is_absolute() else ROOT / p)     # noqa: E731
            box = self.d / "children" / "grade"
            box.mkdir(parents=True, exist_ok=True)
            self.tell("grade", "started", counters=self.counters())
            env = os.environ.get("SPREADING_RUNS_DIR")
            os.environ["SPREADING_RUNS_DIR"] = str(box)
            try:
                self.grade = {"folder": str(grade.main(["grade", here(g["reference"]), here(g["frame"]),
                                                        *finals.values()]))}
            except (Exception, SystemExit) as e:      # the grade refuses: recorded, the finals stay
                self.grade = {"error": repr(e)}
            finally:
                if env is None:
                    os.environ.pop("SPREADING_RUNS_DIR", None)
                else:
                    os.environ["SPREADING_RUNS_DIR"] = env
            self.tell("grade", "ended", result="graded" if "folder" in self.grade else "refused",
                      counters=self.counters())
        return finals


ENDS = {Capped: "capped", Stopped: "stopped", Defect: "harness defect", Incomplete: "incomplete"}
STOP_FIELDS = ("child", "folder", "before", "kind", "reason", "role", "tag", "dispatch", "attempt", "http_status",
               "error_code", "rate_limit_class", "rate_limit_reason", "rate_limit_stop", "cap", "failures")


class Refused(ValueError):
    """The job's settings refused before its folder is made (the command shows it as a usage error)."""


def prepare(settings, lineup, dollars):
    """The job's settings, line-up and frozen allocation, checked before anything is made: (cfg, line-up,
    allocation). A `template` setting pins this process's template (provenance.pin) before anything reads it."""
    cfg = load_settings(settings)
    try:
        if cfg.get("template"):
            provenance.pin(cfg["template"])
    except provenance.Unapproved as e:
        raise Refused(str(e))
    lu = cfg["lineups"].get(lineup)
    if lu is None:
        raise Refused(f"--lineup {lineup}: the settings have {sorted(cfg['lineups'])}")
    if lu.get("pending"):
        raise Refused(f"--lineup {lineup} is not built: {lu['pending']}")
    if "--no-checker" in [*cfg["spread_args"], *lu.get("spread_args", [])] and cfg["tree"].get("on", True):
        raise Refused("--no-checker needs the tree off: the tree's review asks the spread run's checker")
    if not 0 <= dollars <= lu.get("dollars_most", 0):
        raise Refused(f"--dollars {dollars}: the line-up's ceiling is {lu.get('dollars_most', 0)}")
    try:
        tree_dispatches(cfg, lu)
    except ValueError as e:
        raise Refused(f"--lineup {lineup}: {e}")
    allocation = None
    company_roles, section_roles = hosted_roles(cfg, lu)
    if company_roles:
        if not lu.get("allocation"):
            raise Refused(f"--lineup {lineup} sends roles to OpenRouter: it needs its frozen allocation file "
                          f"(the line-up's 'allocation')")
        try:
            require_frozen(lu["allocation"])
            allocation = load_allocation(lu["allocation"], company_roles, section_roles)
        except (OSError, ValueError) as e:
            raise Refused(f"--lineup {lineup}: allocation {lu['allocation']}: {e}")
    return cfg, lu, allocation


def run_job(settings, lineup, dollars, probe=None, grade=True, stop=None, progress=None, children=None,
            filing=None, taxonomy=None, clock=time.monotonic, start=None):
    """One job, the entry the command and the UI call; one active job per process (a step sets the process's
    SPREADING_RUNS_DIR). `stop`: a threading.Event that stops the job as its STOP file does; `progress`: an optional
    callback, called with events (job ID, stage, status, counters named by their units), the first, before any step,
    with the job's folder. `children`, `filing`, `taxonomy`: fakes for the offline tests. Returns the job's ID, folder,
    end, why and record file; a stopped or failed job keeps its partial records. Refused settings raise Refused before
    any folder is made."""
    start = clock() if start is None else start     # the job's deadline counts from here, preparation included
    cfg, lu, allocation = prepare(settings, lineup, dollars)
    d = new_run(f"full_pipeline_{lineup}", {
        "settings": settings, "settings_sha256": hashlib.sha256(Path(settings).read_bytes()).hexdigest(),
        "settings_content": cfg, "lineup": lineup, "dollars": dollars, "allocation": allocation,
        "probe": probe,
        "evidence": probe is None, "plan": "the full pipeline, one company per job",
        "stop_switch": "STOP in this folder (or the project's STOP) stops the active child and every later one",
        "grade": grade})
    a = argparse.Namespace(dollars=dollars, probe=probe, no_grade=not grade)
    tell = Progress(progress, d.name)
    tell("job", "started", folder=str(d))
    job, end, error, problems, detail, when = None, "harness defect", None, [], {}, None
    try:
        children = children or Children()
        if filing is None:
            from .filing import Filing
            p = Path(cfg["filing"])
            filing = Filing(str(p if p.is_absolute() else ROOT / p))
        if taxonomy is None:
            from .spread import taxonomy_year
            from .taxonomy import Taxonomy, edition
            taxonomy = Taxonomy(edition(taxonomy_year(filing)))
        job = Job(d, cfg, lu, a, clock, start, children, filing, taxonomy, allocation, stop, tell)
        job.run()
        end = "finished"
    except End as e:
        end, error, detail, when = ENDS[type(e)], str(e), e.detail, e.time
    except KeyboardInterrupt as e:
        end, error = "interrupted", repr(e)
    except BaseException as e:      # anything else is the harness's: recorded, then the job's records are written
        end, error = "harness defect", f"{e!r}\n{traceback.format_exc()[-3000:]}"
    when = when or time.strftime("%Y-%m-%dT%H:%M:%S")
    # finalization: every part guarded, so that a failure in one is recorded and the job's outcome and exception
    # report are still written; a part that fails is a harness defect (an interrupted job stays interrupted)
    finals, out = {}, {}

    def part(name, fn):
        nonlocal end, error
        try:
            return fn()
        except BaseException as e:
            why = str(e) if isinstance(e, Defect) else repr(e)
            problems.append({"step": name, "class": "harness defect", "why": why})
            end = end if end == "interrupted" else "harness defect"
            error = f"{error}; {name}: {why}" if error else f"{name}: {why}"
            if job is not None:
                job.end = end

    def chain_file():
        cum = d / "cumulative"
        write(d, "chains.json", json.dumps(
            {n: chains(cum / x["versions"][0], [c for c in job.commits if c["to"].startswith(n + "_v")])
             for n, x in job.statements.items() if x["versions"]}, indent=1, default=str))

    def record():
        tried = {(r["statement"], r["section"]) for r in job.skips}
        tried |= {(r["statement"], r["section"]) for r in read_steps(d) if r.get("kind") == "tree result"}
        unreached = [{**o, "why": f"not run: the job ended {end}"} for o in job.order or []
                     if (o["statement"], o["section"]) not in tried and end != "finished"]
        not_run = [] if end == "finished" else [
            *({"statement": x["statement"], "step": "spread", "why": f"not run: the job ended {end}"}
              for x in job.cfg["statements"] if x["statement"] not in job.statements),
            *([{"statement": None, "step": "tree", "why": f"not run: the job ended {end} before the tree's order "
                                                           f"was fixed"}] if job.order is None else [])]
        out.update({
            "company": job.company.state(), "order": job.order, "skipped": job.skips + unreached, "not_run": not_run,
            "company_ledger": None if job.ledger is None else {"folder": str(job.ledger.dir), **charged(job.ledger)},
            "attempts": job.attempts, "commits": [{k: v for k, v in c.items() if k != "chain"} for c in job.commits],
            "finals": finals, "grade": job.grade, "chains_note": CHAIN_NOTE, "sources_kept": len(job.sources),
            "hosted_429": hosted_429(d), "template": job.template, "corrections": job.corrections,
            "statements": {n: {k: v for k, v in x.items() if k != "statement"} for n, x in job.statements.items()}})

    if job is not None:
        job.end = end
        finals = part("finish", job.finish) or {}
        part("chains", chain_file)
        part("job record", record)
    ended = {"end": end, "why": error, "time": when, **{k: detail.get(k) for k in STOP_FIELDS}}
    report = [*([{"step": "job", "class": end, **{k: v for k, v in ended.items() if k != "end"}}]
                if end != "finished" else []), *problems]
    if tell.error:
        out["progress_callback"] = tell.error
    for name, obj in (("exceptions.json", {"end": end, "exceptions": len(report), "cases": report}),
                      *([("stop.json", ended)] if end != "finished" else []),
                      ("job.json", {"end": end, "error": error, "stop": ended if end != "finished" else None,
                                    "parts": parts(cfg, lu), "review": review(cfg, lu), **out, "finalization": problems})):
        try:
            write(d, name, json.dumps(obj, indent=1, default=str))
        except BaseException as e:
            write(d, name, json.dumps({"end": end, "error": str(error), "unwritable": repr(e)}, indent=1))
    tell("job", end, folder=str(d), why=error)
    return {"job_id": d.name, "folder": d, "end": end, "why": error, "record": d / "job.json"}


def main(argv=None, children=None, filing=None, taxonomy=None, clock=time.monotonic, exit=False):
    """The command: its arguments, usage errors and exit status around run_job. `children`, `filing`, `taxonomy`:
    fakes for the offline tests. `exit`: the command's exit status (1 on a harness defect or an incomplete job)."""
    start = clock()                         # the job's deadline counts from here, preparation included
    ap = argparse.ArgumentParser(prog="python -m spreading.full_pipeline")
    ap.add_argument("settings", help="the job's settings (data/manifests/full_pipeline_*.json)")
    ap.add_argument("--lineup", required=True)
    ap.add_argument("--dollars", type=float, required=True, help="the job's dollar ceiling, every paid call included")
    ap.add_argument("--probe", help="why this job: a job run with this reason is labelled diagnostic, never evidence")
    ap.add_argument("--no-grade", action="store_true")
    a = ap.parse_args(argv)
    try:
        r = run_job(a.settings, a.lineup, a.dollars, probe=a.probe, grade=not a.no_grade, children=children,
                    filing=filing, taxonomy=taxonomy, clock=clock, start=start)
    except Refused as e:
        ap.error(str(e))
    print("job:", r["end"], r["why"] or "")
    print(r["folder"])
    if r["end"] == "interrupted":
        raise KeyboardInterrupt(r["why"])
    if exit and r["end"] in ("harness defect", "incomplete"):
        raise SystemExit(1)
    return r["folder"]


def read_steps(d):
    return read_jsonl(Path(d) / "steps.jsonl")


# ---- HTTP 429s and failed requests, read from the call records -------------------------------------------------------

def http_status(r):
    """A call record's recorded HTTP status: llm.py's `http.status`, or a top-level `http_status`; None when neither
    was recorded (an older record)."""
    h = r.get("http")
    return h.get("status") if isinstance(h, dict) and h.get("status") is not None else r.get("http_status")


LIMIT_CLASS = ("rate_limit", "class")     # a 429's class on its call record, set by llm.py: upstream, platform or
#                                           unknown
AFTER_429 = ("after_429", "retry_ok")      # the retry's record of the 429 that led to it, and whether it succeeded
LOST_TO_429 = "schema_retry_lost_to_429"  # llm.py's bookkeeping record of a question whose 429 retry used the second
#                                           dispatch and whose reply then failed its schema with no retry left
LIMIT_CLASSES = ("upstream", "platform", "unknown")


def is_429(r):
    """A call record of an HTTP 429: its recorded status, else the status in its error text (older records)."""
    if http_status(r) is not None:
        return http_status(r) == 429
    return "HTTPError 429" in str(r.get("error") or "")


def limit_class(r):
    """A 429's class as llm.py recorded it; a record without one (an older record, or no headers and no
    provider_code) is unknown, never assumed upstream."""
    rl = r.get(LIMIT_CLASS[0])
    c = rl.get(LIMIT_CLASS[1]) if isinstance(rl, dict) else None
    return c if c in LIMIT_CLASSES else "unknown"


def hosted_429(d):
    """The job's HTTP 429s from its children's call records, by class, by what the class was read from
    (rate_limit.class_source) and by the raw limit_source, with is_byok as recorded (missing kept apart from false);
    the retries they led to and how many succeeded, read from the retry's after_429 record (older records: the
    question's next send, same role and request tag, in the same child, and whether it got a reply); the questions
    whose schema retry was lost to a 429; llm.recovery_counts and llm.schema_retry_429s summed over the children, each
    counted once from the records; and the recoveries and planned wait claimed, reconciled with what the company
    ledger's allowance recorded (rate_limit_used)."""
    from .llm import recovery_counts, schema_retry_429s
    out = {"http_429": 0, "by_class": dict.fromkeys(LIMIT_CLASSES, 0), "by_class_source": {}, "limit_source": {},
           "is_byok": {"true": 0, "false": 0, "missing": 0}, "retried": 0, "retries_succeeded": 0,
           "schema_retry_lost_to_429": 0, "schema_retry_429s": {"count": 0, "recovery_ran": 0, "ended": {}},
           "recovery_counts": None, "ledger_allowance": None,
           "fields": {"class": ".".join(LIMIT_CLASS), "retry": ".".join(AFTER_429), "lost": LOST_TO_429,
                      "class_source": "rate_limit.class_source", "limit_source": "rate_limit.limit_source",
                      "is_byok": "rate_limit.is_byok"}}
    tally = lambda m, k: m.__setitem__(k, m.get(k, 0) + 1)        # noqa: E731
    counts = {}
    for _, run in child_runs(d):
        every = read_jsonl(run / "calls.jsonl") if run is not None else []
        out["schema_retry_lost_to_429"] += sum(bool(r.get(LOST_TO_429)) for r in every)
        s4 = schema_retry_429s(every)
        out["schema_retry_429s"]["count"] += s4["count"]
        out["schema_retry_429s"]["recovery_ran"] += s4["recovery_ran"]
        for k, n in s4["ended"].items():
            out["schema_retry_429s"]["ended"][k] = out["schema_retry_429s"]["ended"].get(k, 0) + n
        for k, v in recovery_counts(every).items():
            if isinstance(v, dict):
                counts[k] = {**counts.get(k, {})}
                for x, n in v.items():
                    counts[k][x] = counts[k].get(x, 0) + n
            else:
                counts[k] = counts.get(k, 0) + v
        rows = [r for r in every if physical(r)]
        for i, r in enumerate(rows):
            if not is_429(r):
                continue
            out["http_429"] += 1
            out["by_class"][limit_class(r)] += 1
            rl = r.get(LIMIT_CLASS[0]) if isinstance(r.get(LIMIT_CLASS[0]), dict) else {}
            tally(out["by_class_source"], str(rl.get("class_source") or "not recorded"))
            tally(out["limit_source"], str(rl.get("limit_source") or "not recorded"))
            byok = rl.get("is_byok")
            tally(out["is_byok"], "missing" if not isinstance(byok, bool) else str(byok).lower())
            nxt = next((x for x in rows[i + 1:] if (x.get("role"), x.get("tag")) == (r.get("role"), r.get("tag"))),
                       None)
            after = nxt.get(AFTER_429[0]) if nxt is not None else None
            if isinstance(r.get(LIMIT_CLASS[0]), dict) or isinstance(after, dict):   # recorded by llm.py
                out["retried"] += isinstance(after, dict)
                out["retries_succeeded"] += isinstance(after, dict) and after.get(AFTER_429[1]) is True
            else:
                out["retried"] += nxt is not None
                out["retries_succeeded"] += nxt is not None and not nxt.get("error")
    out["recovery_counts"] = {k: round(v, 3) if isinstance(v, float) else v for k, v in counts.items()} or None
    state = Path(d) / "company_ledger" / "state.json"
    if state.is_file():
        used = json.loads(state.read_text(encoding="utf8")).get("rate_limit_used") or {}
        led = {"recoveries": sum(x.get("recoveries") or 0 for x in used.values()),
               "wait_s": round(sum(x.get("wait_s") or 0.0 for x in used.values()), 3)}
        rec = {"recoveries": (counts or {}).get("claimed", 0), "wait_s": round((counts or {}).get("planned_wait_s",
                                                                                                0.0), 3)}
        out["ledger_allowance"] = {"ledger": led, "records": rec, "agree": led == rec}
    return out


def child_runs(d):
    """The job's children in the order they ran: (box name, the child's run folder or None)."""
    out, kids = [], Path(d) / "children"
    for box in sorted(p for p in kids.iterdir() if p.is_dir() and p.name[:2].isdigit()) if kids.is_dir() else []:
        runs = sorted(p for p in box.iterdir() if p.is_dir())
        out.append((box.name, runs[0] if len(runs) == 1 else None))
    return out


if __name__ == "__main__":
    main(exit=True)
