"""C8, the review step of C7's tree: the pipeline's own checker re-solves each changed line blind, as the source spread
run's checker would have, and code compares its answer with the candidate's. Only agreement settles a line;
repair.Tree.commit withdraws the repair on anything else and keeps the original draft.

Blind: each line gets a fresh run_c4(role="checker") built from the source run, never from the request: the line as
spread.prepare reads it, the section's offered rows (a queued line: every offered row of the statement), the
printed-line protection, the tools as spread.run builds them, search when the source had it, and the source's checker
model, context, answer cap, step cap, prompt, row seed and sampler options. Of the request only the line's concept, its
section and the review order are read; its questions, candidate and decomposition, and every tree and critic text, stay
code-side. Before any call, code rebuilds the analyst's start of the line and refuses the review unless it is
byte-identical to the start the source run recorded, with the same row order.

Comparison: spread.compare on the canonical decomposition, then the candidate's whole decomposition (each piece's amount
and coefficient, the remainder, the line's amount) against the checker's own (`agreement`). A candidate the adapter
cannot read whole ends `unresolved`, never simplified.

Ownership: the checker's registry is rebuilt from every finished checker review of the source run (routed, hand-off and
queued), each key with the line it came from (`ownership`); the claims of every line under review are removed, the lines
are reviewed in the request's fixed order, and each settled answer's keys are added before the next. A key claimed twice
refuses the run.

Budget: `line_ceiling` is one line's ceiling in physical requests, the step cap times one question's sends
(`checker_sends`: DISPATCHES_PER_QUESTION, plus the 429 recovery on a hosted checker) chat attempts and a query
embedding on every step but the last, and the tokens their reservations take (llm.Caller.attempt_tokens;
llm.OllamaEmbedder's UTF-8 bytes plus EMBED_OVERHEAD). Every attempt goes through the call's meter, which holds the line
to that token ceiling: each query embedding reserves its actual UTF-8 bytes before it is sent, and one that does not fit
what is left of the line stops the review (cap_reached), never truncated. The passage index is read from the cache only:
a miss refuses the run.

Endings: a refusal by the meter, the checker's Caller at its call cap and the stop switch are raised, so repair.Budget
records cap_reached, the stop's reason kept; a request that fails is raised as a role failure; an interrupt propagates.
However a review ends, its log entry keeps its status, usage, embedding calls and searches, and its Caller is kept in
`callers` for the run's totals. Its usage counts the requests physically sent (repair.Counted); an attempt refused
before sending is kept in the entry's `unsent`, beside the Caller's `attempts`, never counted as a request.

Input fidelity: a checker whose recorded start is not the rebuilt one ends `refused`, never settled; a live identity
check refuses the review unless both digests are present and equal.

Hosted checker: a source run on OpenRouter with a per-role profile (spread's HOSTED_FLASH: GLM at effort `low`) is
re-solved with the checker settings it recorded, its model, reasoning setting, pinned providers, price ceiling and
max_tokens (its answer cap plus the allowance), on the run's dollar limit. A hosted model has no local digest: the
Caller refuses a reply from another model or an unpinned provider instead. A source run whose hosted checker lacks any
of those settings, or records one malformed (`hosted_role_errors`: empty pins, a pin without its exact provider names, a
price that is not a pair of finite nonnegative numbers, an allowance that is not a nonnegative integer, a boolean
anywhere), is refused, as is a review without a dollar limit or without the checker's frozen allocation
(`role_limits["checker"]`), which the run's ledger holds for every line's Caller.

    review = Review(spread_run, f, tax, lambda m: OllamaEmbedder(ledger=m, company=company, stop=stop),
                    run / "calls.jsonl", stop=stop, company=company)
    repair.repair(draft, **{**roles.kwargs(), "c8_review": review}, review_line=review.config["line_ceiling"], ...)
"""
import hashlib
import json
import math
from decimal import Decimal, InvalidOperation
from pathlib import Path

from . import organizer, provenance, template
from .analyst import row_order, run_c4, start_text
from .runrec import search_use
from .ledger import LimitReached
from .llm import DISPATCHES_PER_QUESTION, EMBED_OVERHEAD, Stopped, count
from .loop import STEP_CAP
from .packs import ROOT, load_rows
from .repair import SETTLED, USAGE_KEYS, Counted, ratio
from .repair_roles import RecordingCaller, question_sends
from .search import QUERY_INSTRUCTION, Search, instruction
from .spread import (COMBINED_PROMPT, FINISHED, LOOP_PROMPT, ROOT_CAUSE_PROMPT, STATEMENTS_PROMPT, compare, ident,
                     prepare, setup, sha)
from .steps import fact_index, phase_relations
from .tools import Tools

QUERY_BYTES_PER_TOKEN = 4       # estimated: an engineering allowance, not a worst-case bound or a measurement: a query's
                                # UTF-8 bytes per answer token in the line ceiling; each query reserves its own bytes
SEED = 7                        # llm.Caller's default, the source run's (spread.freeze_hashes)


class Unsupported(Exception):
    """A candidate the adapter cannot read whole: the review ends unresolved."""


class Refused(Exception):
    """A review that cannot be run as the source run would have run it: no dispatch, the review ends refused."""


class CacheOnly:
    """The review search's embedder outside a review call: the model's pins, and no request (the passage index
    must be a validated cache hit)."""

    def __init__(self, model, dimensions, pinned, query_instruction=QUERY_INSTRUCTION):
        self.model, self.dimensions, self.pinned = model, dimensions, pinned
        self.query_instruction = query_instruction      # the source embedder's, so the cache keys match
        self.calls, self.failed = [], None

    def digest(self):
        return self.pinned

    def __call__(self, texts):
        raise Refused(f"{len(texts)} text(s) to embed outside a review call: the passage index is not a cache hit")


def line_ceiling(num_ctx, answer_cap, cap=STEP_CAP, search=True, sends=DISPATCHES_PER_QUESTION, embed_attempts=1):
    """One line's review ceiling: {requests, tokens, chat_requests, query_embeddings}. `sends`: one question's most
    requests (checker_sends); `embed_attempts`: one query's most embedding requests (the embedder's `attempts`: 3
    hosted, 1 local). query_embeddings counts queries, requests and tokens every attempt."""
    chat, queries = cap * sends, (cap - 1 if search else 0)
    query = len(QUERY_INSTRUCTION.encode("utf8")) + answer_cap * QUERY_BYTES_PER_TOKEN + EMBED_OVERHEAD
    return {"requests": chat + queries * embed_attempts,
            "tokens": chat * (num_ctx + answer_cap) + queries * embed_attempts * query,
            "chat_requests": chat, "query_embeddings": queries}


def keys(rec):
    """The (tag, dims) keys a finished answer's pieces own, as spread.run adds them to a registry (spread.ident)."""
    return [ident(x) for x in (rec.get("answer") or {}).get("_pieces") or ()]


def ownership(c4):
    """[{key, line, via}]: every key a finished checker review of the source run owns, in c4.json's order, with the
    line it came from and how (review, transfer.review, queued). Refused when a key is claimed twice or a piece has
    no key."""
    out, seen = [], {}
    for c, e in c4.items():
        for via, rec in (("queued" if e.get("queued") else "review", e.get("review")),
                         ("transfer.review", (e.get("transfer") or {}).get("review"))):
            if not rec or rec.get("termination") not in FINISHED:
                continue
            for p in (rec.get("answer") or {}).get("_pieces") or ():
                if "concept" not in p or "dims" not in p:
                    raise Refused(f"{c} ({via}): a piece without concept or dims")
                k = ident(p)
                if k in seen:
                    raise Refused(f"{k[0]} {list(k[1])} is claimed by {seen[k]} and by {c} ({via})")
                seen[k] = f"{c} ({via})"
                out.append({"key": k, "line": c, "via": via})
    return out


def dec(x):
    return Decimal(str(x))


def adapt(req):
    """The candidate's whole decomposition from the request's code-side `decomposition`: {concept, amount, row,
    pieces [(fact, row, amount, r)], remainder}. Unsupported when a field is missing, a row is None or the remainder
    is not the line less its pieces."""
    try:
        line, d = req["line"], req["decomposition"]
        pieces = [(p["fact"], p["row"], dec(p["amount"]), ratio(p)) for p in d["pieces"]]
        out = {"concept": line["concept"], "amount": dec(line["amount"]), "row": d["remainder"]["row"],
               "pieces": pieces, "remainder": dec(d["remainder"]["amount"])}
    except (KeyError, TypeError, InvalidOperation) as e:
        raise Unsupported(f"the candidate cannot be read whole: {type(e).__name__}: {e}") from e
    if out["row"] is None or any(r is None for _, r, _, _ in pieces):
        raise Unsupported("the candidate leaves a row unnamed")
    if out["remainder"] != out["amount"] - sum((a for _, _, a, _ in pieces), Decimal(0)):
        raise Unsupported(f"the candidate's remainder {out['remainder']} is not the line's {out['amount']} less its "
                          "pieces")
    return out


def agreement(cand, rec8, canonical):
    """(status, detail): `unfinished` when the checker did not finish; otherwise spread.compare on the canonical
    decomposition, then the whole decomposition: pieces as (canonical fact, row, amount, r), the remainder and the
    line's amount. `settled` only when all agree, else `disagreed`."""
    if rec8.get("termination") not in FINISHED:
        return "unfinished", {"termination": rec8.get("termination"), "why": rec8.get("why")}
    p = {"termination": "finished", "decision": "row", "row": cand["row"], "amount": str(cand["amount"]),
         "answer": {"_pieces": [{"fact": f, "row": r, "amount": a} for f, r, a, _ in cand["pieces"]],
                    "_remainder": cand["remainder"]}}
    result, detail = compare(p, rec8, canonical)
    if result != "agree":
        return "disagreed", {"compare": result, **detail}
    ans = rec8.get("answer") or {}
    left = ans.get("_remainder")
    mine = sorted((canonical.get(f, f), r, a, k) for f, r, a, k in cand["pieces"])
    theirs = sorted((canonical.get(x["fact"], x["fact"]), x["row"], dec(x["amount"]), ratio(x))
                    for x in ans.get("_pieces") or ())
    full = {"pieces": mine == theirs, "remainder": cand["remainder"] == dec(rec8["amount"] if left is None else left),
            "amount": cand["amount"] == dec(rec8["amount"])}
    detail = {"compare": result, **detail, "full": full,
              "candidate": [list(map(str, x)) for x in mine], "checker": [list(map(str, x)) for x in theirs]}
    return (SETTLED if all(full.values()) else "disagreed"), detail


def hosted_checker(m):
    """The source run's hosted checker settings (spread's per-role profile, as recorded), or None on the local
    engine."""
    return (m.get("hosted") or {}).get("checker") if m.get("engine") == "openrouter" else None


def checker_sends(m):
    """The most requests one question of the review's checker may send: repair_roles.question_sends of its engine,
    three on a hosted checker (two answer attempts and one 429 recovery), two locally."""
    return question_sends("openrouter" if hosted_checker(m) is not None else "ollama")


def embed_attempts(m):
    """The most requests one embedding batch of the review may send: the source run's embedder's `attempts`
    (repair_run picks HostedEmbedder when the run has a hosted profile)."""
    from .llm import HostedEmbedder, OllamaEmbedder
    return (HostedEmbedder if m.get("hosted_profile") else OllamaEmbedder).attempts


def source_engines(m):
    """Role -> engine of the source run, as recorded: on OpenRouter every role; on the local engine every role, but
    Haiku on OpenRouter beside it (spread's mixed run records it under "third")."""
    hosted = set(m.get("hosted") or {}) if m.get("engine") == "openrouter" else set(m.get("third") or {})
    return {r: "openrouter" if r in hosted else "ollama" for r in {**(m.get("models") or {}), **dict.fromkeys(hosted)}}


def hosted_role_errors(name, v):
    """Why one role's hosted settings (a per-role profile's entry, as recorded) cannot be sent as spread's Caller
    sent them: a model name, a reasoning setting, nonempty provider pins (empty ones would skip the Caller's check of
    what served the reply) with the exact provider names of every pin, a price pair of finite nonnegative numbers and
    a nonnegative integer allowance, booleans refused."""
    pins, names, price, allow = (v.get(k) for k in ("providers", "provider_names", "price", "allowance"))
    word = lambda x: isinstance(x, str) and bool(x)                                 # noqa: E731
    words = lambda x: isinstance(x, (list, tuple)) and bool(x) and all(map(word, x))    # noqa: E731
    num = lambda x: (not isinstance(x, bool) and isinstance(x, (int, float))     # noqa: E731
                     and math.isfinite(x) and x >= 0)
    return [f"{name}'s {w}" for bad, w in (
        (not word(v.get("model")), f"model {v.get('model')!r} is not a name"),
        (not word(v.get("reasoning")), "reasoning setting is missing (not a per-role profile)"),
        (not words(pins), f"provider pins {pins!r} are not a nonempty list of names"),
        (words(pins) and (not isinstance(names, dict) or bool(set(pins) - set(names))
                          or not all(words(names[p]) for p in pins)),
         f"exact provider names {names!r} do not name every pin"),
        (not isinstance(price, (list, tuple)) or len(price) != 2 or not all(map(num, price)),
         f"price {price!r} is not a pair of finite nonnegative numbers"),
        (isinstance(allow, bool) or not isinstance(allow, int) or allow < 0,
         f"allowance {allow!r} is not a nonnegative integer")) if bad]


def hosted_errors(m):
    """Why the source run's checker cannot be re-solved alike: local with no hosted profile, or OpenRouter with the
    checker's own model, reasoning setting, providers with their exact names, price and allowance all recorded and
    well formed (hosted_role_errors). Only the checker is re-solved, so another hosted role (Haiku beside a
    local checker) refuses nothing: its replies never reach the review's request (the checker's start and loop,
    blind)."""
    if m.get("engine") == "ollama" and not m.get("hosted"):
        return []
    h = hosted_checker(m)
    if h is None:
        return [f"engine {m.get('engine')}, hosted {m.get('hosted')}"]
    unset = [k for k in ("providers", "price", "allowance") if h.get(k) is None]
    return [w for bad, w in (
        (bool(unset), f"the hosted checker's {', '.join(unset)} not recorded"),
        (h.get("model") != (m.get("models") or {}).get("checker"),
         f"the hosted checker {h.get('model')} is not the run's checker {(m.get('models') or {}).get('checker')}"))
            if bad] + ([] if unset else hosted_role_errors("the hosted checker", h))


def source_prompt(m):
    """The placement prompt a source run's settings imply: v29 with statements_only, v28 with
    root_cause_fixes (the combined fixes plus the root-cause fixes), v27 with combined_fixes alone, else v26."""
    return (STATEMENTS_PROMPT if m.get("statements_only") else ROOT_CAUSE_PROMPT if m.get("root_cause_fixes") else
            COMBINED_PROMPT if m.get("combined_fixes") else LOOP_PROMPT)


def settings_errors(m):
    """Why the source run's checker cannot be re-solved alike from here: its engine, its routing and prompt, and the
    resources its starts were built from."""
    h = m.get("hashes") or {}
    errs = hosted_errors(m) + [w for bad, w in (
        (m.get("prompt") != source_prompt(m), f"prompt {m.get('prompt')} is not {source_prompt(m)}"),
        (bool(m.get("root_cause_fixes")) and not m.get("combined_fixes"),
         "the source has root_cause_fixes without combined_fixes"),
        (not m.get("no_a2"), "the source routed lines by A2"),
        (m.get("tags_hidden") is not True, "the source was made with tags shown"),
        (not (m.get("models") or {}).get("checker"), "no checker model")) if bad]
    if not errs:
        p = m["prompt"]
        errs += [w for bad, w in (
            (sha(ROOT / "prompts" / f"{p}.txt") != (h.get("prompts") or {}).get(p), f"prompt {p} changed"),
            (provenance.inventory() != h.get("approved_resources"), "approved resources changed")) if bad]
    return errs


class Review:
    """The c8_review role (repair.repair): review(request, ledger=meter) -> (result, usage). `f`, `tax`: the source
    run's filing and its us-gaap edition. `embedder(ledger)`: a new embedder on that ledger (None: only its model,
    dimensions and digest are read). `digest`: the embedding model's digest when it is not read from the engine (an
    offline test or dry run). `identity(models)`: the checker's {digests}, compared with the source run's; None
    leaves it unread, and a hosted checker has none. `spend_usd`: the run's dollar limit, a hosted checker's Caller's
    spend cap (required above 0 for one). `role_limits`: the run ledger's role limits; a hosted checker needs its frozen
    allocation there ("checker": {calls, spend_usd}), which the ledger holds across every line's fresh Caller, and its
    Caller's own cap is the smaller of the two dollar figures. `log`: every review's result, a raised ending included,
    for the run's record. `model`: the review's own local model instead of the source run's checker (line-ups name the
    review apart from the checker); it is given the checker's line, rows, tools, registry and settings, and its digest
    is read at the start and recorded (the source run froze none for it). None or the checker's own name: the
    checker."""

    def __init__(self, source_run, f, tax, embedder, log_path, stop=None, company=None, digest=None, identity=None,
                 spend_usd=None, role_limits=None, model=None):
        d = Path(source_run)
        m = self.run = json.loads((d / "run.json").read_text(encoding="utf8"))
        # the source's arm by the one contract (full_pipeline.arm_of), here too: before a prompt or a relation
        # is selected, so that a direct caller is refused as repair_run's entry refuses. A flag that is no Boolean, an R source without its overlap record and
        # a T source that carries one are refused; a lost record is never read as T
        from .full_pipeline import ArmRecord, arm_of
        try:
            diag = json.loads((d / "diagnostics.json").read_text(encoding="utf8"))
        except (OSError, ValueError):
            diag = None
        try:
            self.arm = arm_of(m, diag)
        except ArmRecord as e:
            raise ValueError(f"C8 review: the source run's arm cannot be read from its record: {e}") from e
        self.c4 = json.loads((d / "c4.json").read_text(encoding="utf8"))
        errs = settings_errors(m)
        if errs:
            raise ValueError("C8 review: the source run cannot be re-solved alike: " + "; ".join(errs))
        self.hosted = hosted_checker(m)
        if self.hosted is not None and not (spend_usd or 0) > 0:
            raise ValueError("C8 review: the source run's checker is hosted; the review needs a dollar limit above 0")
        self.allocation = (role_limits or {}).get("checker")
        if self.hosted is not None and self.allocation is None:
            raise ValueError("C8 review: the source run's checker is hosted; the review needs the checker's frozen "
                             "allocation in the run's ledger (--role-limit checker CALLS USD)")
        self.f, self.tax, self.embedder, self.log_path = f, tax, embedder, log_path
        self.stop, self.company, self.log, self.callers, self.embedders = stop, company, [], [], []
        self.checker = m["models"]["checker"]
        self.model = model or self.checker      # the model that answers the review
        own = self.model != self.checker
        if own and self.hosted is not None:
            raise ValueError("C8 review: a review model of its own is local only; the source run's checker is hosted")
        self.t = t = template.active()
        rows = load_rows()
        first = {}
        for i, r in enumerate(rows):
            first.setdefault(t.section_of(r["row_id"]), i)
        ran = m["section"] if isinstance(m["section"], list) else [m["section"]]
        self.statement = t.statement_of(ran[0])
        try:                            # the source run's frozen organizer record, read, never asked
            org = organizer.load(d, f, tax, self.statement)
        except organizer.Refused as x:
            raise ValueError(f"C8 review: {x}") from x
        prep = prepare(f, tax, self.statement, sorted(ran, key=first.get), [], setup(f, self.statement), org)
        self.prep, self.st, self.canonical, self.printed = prep, prep["st"], prep["canonical"], prep["printed"]
        # R7: the re-review of an R job's line reads overlaps as that job did
        self.index, self.rel = fact_index(f), phase_relations(f, "source-run checking", m.get("root_cause_fixes"))
        self.names = {r["row_id"]: r["name"] for r in rows}
        self.search = self.guard = None
        pins = None
        if m.get("search"):
            src = json.loads((d / "search.json").read_text(encoding="utf8"))["pins"]
            e = embedder(None)
            self.guard = CacheOnly(e.model, e.dimensions, digest if digest is not None else e.digest(),
                                   instruction(e))
            try:
                self.search = Search.of(f, self.guard, **({"cache": m["cache"]} if m.get("cache") else {}))
            except Refused as x:
                raise ValueError(f"C8 review: {x}") from x
            pins = self.search.pins
            # cache_use records how a run got its vectors (a miss embedded and wrote them), not a search setting
            diff = sorted(k for k in set(src) | set(pins) if k not in ("cache", "cache_use") and src.get(k) != pins.get(k))
            if diff or Path(src["cache"]).resolve() != Path(pins["cache"]).resolve():
                raise ValueError(f"C8 review: the search differs from the source run's: {diff or 'cache path'}")
        ident = "not read"
        if self.hosted is not None:
            ident = ("not applicable: a hosted model has no local digest; the Caller refuses a reply from another "
                     "model or an unpinned provider")
        elif identity is not None:          # fails closed: both digests present and equal
            ident = identity({"checker": self.model})
            want = ((m["hashes"].get("models") or {}).get("digests") or {}).get(self.model)
            got = ((ident or {}).get("digests") or {}).get(self.model)
            if own and not got:             # a model of its own: present, and recorded
                raise ValueError(f"C8 review: the review model {self.model}'s digest is missing")
            if own and want and got != want:    # a model the source run used (its tiebreaker): its pin holds
                raise ValueError(f"C8 review: the review model {self.model}'s digest {got!r} differs from the source "
                                 f"run's {want!r}")
            if not own and (not want or not got or got != want):
                raise ValueError(f"C8 review: {self.model}'s digest {got!r} is missing or differs from the source "
                                 f"run's {want!r}")
        try:
            self.owned = ownership(self.c4)
        except Refused as x:
            raise ValueError(f"C8 review: the checker's ownership is ambiguous: {x}") from x
        self.order, self.reviewed, self.added = None, [], []
        code = (m["hashes"].get("code") or {})
        self.config = {
            "source_run": str(d), "engine": m["engine"], "source_engines": source_engines(m),
            "checker_model": self.checker, "review_model": self.model, "review_is_checker": not own,
            "identity": ident,
            "hosted": self.hosted and {**self.hosted, "max_tokens": m["answer_cap"] + self.hosted["allowance"],
                                       "spend_cap_usd": min(spend_usd, self.allocation["spend_usd"])},
            "allocation": self.allocation,
            "source_digest": ((m["hashes"].get("models") or {}).get("digests") or {}).get(self.checker),
            "num_ctx": m["num_ctx"], "answer_cap": m["answer_cap"], "cap": m["cap"], "prompt": m["prompt"],
            "prompt_sha256": m["hashes"]["prompts"][m["prompt"]], "row_seed": m["row_seed"],
            "options": m.get("options") or {}, "seed": SEED, "statement": self.statement, "sections": ran,
            "line_ceiling": line_ceiling(m["num_ctx"], m["answer_cap"], m["cap"], bool(m.get("search")),
                                         sends=checker_sends(m), embed_attempts=embed_attempts(m)),
            "question_sends": checker_sends(m),
            "query_bytes_per_token": f"{QUERY_BYTES_PER_TOKEN} (estimated: an engineering allowance, not a worst-case "
                                     "bound; each query reserves its actual UTF-8 bytes)",
            "search": pins and {**pins, "digest_read": "given" if digest is not None else "from the engine"},
            "registry": {"initial": [{**o, "key": [o["key"][0], [list(x) for x in o["key"][1]]]}
                                     for o in self.owned]},
            "template_rows_sha256": m["hashes"]["template_rows"],
            "code_changed_since_source": sorted(n for n, v in code.items()
                                                if not (ROOT / "spreading" / n).is_file()
                                                or sha(ROOT / "spreading" / n) != v),
            "blind": "the checker is given the source run's line, rows, tools and registry only; the request's "
                     "questions, candidate and decomposition stay code-side"}

    def registry(self, req):
        """The checker's registry for this line: the source ownership less every line under review, plus each
        settled answer's keys, in the fixed order. Refused when the order changes or a line comes out of turn."""
        c, order = req["line"]["concept"], list(req.get("order") or ())
        if self.order is None:
            self.order = order
        if order != self.order:
            raise Refused(f"the review order changed: {order}, first {self.order}")
        if len(self.reviewed) >= len(order) or order[len(self.reviewed)] != c:
            raise Refused(f"{c} out of turn in {order} after {self.reviewed}")
        base = [o["key"] for o in self.owned if o["line"] not in order]
        return base + [k for a in self.added for k in a["keys"]], {
            "removed": [o["line"] for o in self.owned if o["line"] in order], "from_source": len(base),
            "added_before": [{"line": a["line"], "keys": len(a["keys"])} for a in self.added]}

    def inputs(self, req):
        """What the checker is given for the line, rebuilt from the source run: {line, section, ids, tools,
        checker_start, fidelity}. Refused when the line is not found where the request puts it or its rebuilt
        analyst start or row order differ from the source run's."""
        c, section, m = req["line"]["concept"], req["section"], self.run
        e = self.c4.get(c)
        if e is None:
            raise Refused(f"{c} is not a line of the source run")
        if e.get("queued"):
            q = next((q for q in self.prep["queued"] if q["line"]["concept"] == c), None)
            if q is None:
                raise Refused(f"{c}: queued in the source run, not in the rebuilt queue")
            src, l, s, ids = e, q["line"], None, q["ids"]
        else:
            src = e if e.get("section") == section else e.get("transfer") if (
                (e.get("transfer") or {}).get("section") == section) else None
            l = next((l for x in self.prep["sections"] for l in self.st[x]["queue"] if l["concept"] == c), None)
            if src is None or l is None:
                raise Refused(f"{c}: the source run placed no episode of it in {section}")
            s, ids = section, self.st[section]["offered"]
        tools = Tools(self.f, l, self.prep["offered"] if s and not self.t.memo(s) else {}, s, self.names,
                      search=self.search, tax=self.tax, statement=self.statement, v20=True,
                      fact_hint=bool(m.get("combined_fixes")), root_cause=bool(m.get("root_cause_fixes")),
                      statements_only=bool(m.get("statements_only")))
        other = tools.other_sections()
        analyst = start_text(self.f, l, s, ids, other)
        checker = start_text(self.f, l, s, row_order(ids, "checker", c, m["row_seed"], grouped=s is None), other)
        rec, old = src.get("c4") or {}, src.get("review") or {}
        fid = {"episode": "queued" if e.get("queued") else "transfer" if src is not e else "routed",
               "analyst_start_same": analyst == rec.get("start"), "row_order_same": list(ids) == rec.get("row_order"),
               **({"checker_start_same": checker == old["start"]} if old.get("start") else {}),
               "checker_start_sha256": hashlib.sha256(checker.encode("utf8")).hexdigest(),
               "tools": tools.available(), "rows": len(ids)}
        if not all(v for k, v in fid.items() if k.endswith("_same")):
            raise Refused(f"{c}: the rebuilt start differs from the source run's: {fid}")
        return {"line": l, "section": s, "ids": list(ids), "tools": tools, "checker_start": checker, "fidelity": fid}

    def __call__(self, req, ledger=None):
        c = req["line"]["concept"]
        res = {"line": c, "section": req["section"]}
        zero = {k: 0 for k in USAGE_KEYS}
        try:
            cand = adapt(req)
        except Unsupported as e:
            self.log.append({**res, "status": "unresolved", "why": str(e)})
            return self.log[-1], zero
        try:
            reg, info = self.registry(req)
            x = self.inputs(req)
        except Refused as e:
            self.log.append({**res, "status": "refused", "why": str(e)})
            return self.log[-1], zero
        self.reviewed.append(c)
        m = self.run
        if ledger is not None and not isinstance(ledger, Counted):     # physical sends counted
            ledger = Counted(ledger)
        res.update(fidelity=x["fidelity"], registry={**info, "keys": len(reg)})
        self.log.append(res)
        h = self.config["hosted"]           # the source run's hosted checker, as its spread Caller sent it
        caller = RecordingCaller({"checker": self.model}, self.log_path, self.config["line_ceiling"]["chat_requests"],
                                 seed=SEED, num_ctx=m["num_ctx"], stop=self.stop, answer_cap=m["answer_cap"],
                                 ledger=ledger, company=self.company, options=m.get("options") or {},
                                 **({"engine": "openrouter", "spend_cap_usd": h["spend_cap_usd"],
                                     "price": {"checker": h["price"]}, "providers": {"checker": h["providers"]},
                                     "reasoning": {"checker": h["reasoning"]},
                                     "max_tokens": {"checker": h["max_tokens"]},
                                     **({"omit": {"checker": h["omit"]}} if h.get("omit") else {}),
                                     # the frozen listing's exact provider names, when the run recorded them
                                     **({"provider_names": h["provider_names"]} if h.get("provider_names") else {})}
                                  if h else {}))
        self.callers.append(caller)
        emb = None
        n0 = len(self.search.log) if self.search is not None else 0
        if self.search is not None:
            emb = self.search.embedder = self.embedder(ledger)
            self.embedders.append(emb)

        def used():
            # requests physically sent; an attempt refused before sending is recorded in `unsent`, never counted.
            # Without a ledger nothing can refuse, and the Caller's and embedder's own counts are the sends
            embedded = emb.calls if emb is not None else []
            sends = ledger.sends if ledger is not None else {"chat": caller.calls, "embedding": len(embedded)}
            res.update(embedding_calls=list(embedded), searches=self.search.log[n0:] if self.search else [],
                       attempts=caller.calls, unsent=list(ledger.unsent) if ledger is not None else [])
            return {"requests": sends["chat"],
                    "tokens": sum(r["prompt_tokens"] + r["completion_tokens"] for r in caller.records
                                  if count(r.get("prompt_tokens")) and count(r.get("completion_tokens"))),
                    "embedding_requests": sends["embedding"],
                    "embedding_tokens": sum(e["tokens"] for e in embedded if count(e.get("tokens")))}

        def failure():
            # the caller's structured failure first, as spread.py writes it; else the embedder's (role embedding)
            return getattr(caller, "failure", None) or getattr(emb, "failure", None)
        r7_start, v0 = len(self.rel.r7_log), self.rel.validated
        try:
            rec = run_c4(caller, self.tax, self.f, x["line"], x["section"], x["ids"], x["tools"], self.printed,
                         list(reg), role="checker", index=self.index, rel=self.rel, cap=m["cap"],
                         prompt_name=m["prompt"], row_seed=m["row_seed"])
        except BaseException as e:          # the record first, then the ending
            ending = "role_failed" if isinstance(e, Exception) else "interrupted"
            res.update(status=None, usage=used(), caller_failed=caller.failed,
                       caller_failure=failure(), why=f"{ending}: {type(e).__name__}: {e}")
            raise
        finally:
            if self.search is not None:
                self.search.embedder = self.guard
        rec["search_use"] = search_use(rec)
        start = rec.get("start")
        x["fidelity"]["checker_start_used"] = bool(start) and start == x["checker_start"]
        usage = used()
        res.update(checker=rec, usage=usage)
        if self.rel.cross_dims_unknown:     # R7 on this path validates the checker's answer; it commits nothing
            res["r7"] = {"path": "source-run checking", "pieces_validated": self.rel.validated - v0,
                         "eligible_attempts": [dict(e, role="checker", section=x["section"])
                                               for e in self.rel.r7_log[r7_start:]],
                         "note": "validation of the checker's answer only; the commitment is the repair's "
                                 "registry, checked by steps.split under the legacy policy"}
        refused = getattr(ledger, "refused", None)
        if refused or rec.get("stop_class") in ("CapReached", "Stopped"):
            res.update(status=None, why=f"cap_reached: {refused or rec.get('why')}", caller_failure=failure())
            if rec.get("stop_class") == "Stopped":
                raise Stopped(f"C8 review of {c}: {rec.get('why')}")
            raise LimitReached(f"C8 review of {c}: {refused or rec.get('why')}")
        if rec.get("stop_class") == "RequestFailed":
            res.update(status=None, why=f"role_failed: {rec.get('why')}", caller_failure=failure())
            raise RuntimeError(f"C8 review of {c}: a request failed: {rec.get('why')}")
        if not x["fidelity"]["checker_start_used"]:         # never settled: the checker did not see the rebuilt start
            res.update(status="refused", why=f"the checker's start is not the rebuilt one (sha256 "
                                             f"{x['fidelity']['checker_start_sha256']})")
            reply = {k: v for k, v in res.items() if k != "checker"}
            return {**reply, "checker_termination": rec.get("termination")}, usage
        status, detail = agreement(cand, rec, self.canonical)
        res.update(status=status, comparison=detail, why={
            SETTLED: "the blind checker's whole decomposition equals the candidate's",
            "disagreed": "the blind checker's decomposition differs from the candidate's",
            "unfinished": f"the blind checker did not finish ({rec.get('termination')}: {rec.get('why')})"}[status])
        if status == SETTLED:
            self.added.append({"line": c, "keys": keys(rec)})
            res["registry"]["added"] = [[k[0], [list(d) for d in k[1]]] for k in keys(rec)]
        reply = {k: v for k, v in res.items() if k != "checker"}
        return {**reply, "checker_termination": rec.get("termination"), "checker_answer": rec.get("answer")}, usage
