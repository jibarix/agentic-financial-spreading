"""The capped tool loop: the C4 analyst under prompt v16, the six rules.

At each step the model sees the start state, the steps so far and the schema of this step, and returns one action: a
tool call or an answer. Code runs the tool, keeps the observation and rebuilds the schema from what the line has
returned so far. Each observation is shown in full once, in the step after its call; after that it is one ledger line,
and `recall` shows it in full again. The model decides which tools to call, when to stop and what to answer; code
executes, checks and logs.

Decision and termination are recorded apart. Terminations: finished, finished_at_last_step, cap_unfinished, no_progress,
context_exceeded, invalid, not_run. Decisions: a row of the section, COMBINED, OUTSIDE, NOT_TOLD, referred (a row of
another section, returned by rows_of), or none.

A failed answer check (the schema after the Caller's own retry, or code's checks on an admitted answer) is shown to the
model with its errors and costs a step; after CHECK_RETRIES such retries the line ends `invalid` (or `cap_unfinished` at
the last step, where no extra call is made).

Repeat guard: the first repeat of a tool call (same canonical action and arguments, `recall` exempt) is not run again;
the model gets a reminder and the earlier result in full, and the step counts. The second repeat (the third request for
the same call) ends the line `no_progress`. Neither resets a cap.

A quote is checked only against text actually rendered in a request of this line (the start state and each observation
shown in full when fresh or recalled), never against text the model never saw. Each reply also records diagnostics that
never refuse: field lengths against the schema caps, and where the quote occurs (`quote_where`).

The pools: each observation grows the four pools of steps.py (search O-IDs recallable only, their passages discoverable
only, read passages and non-search observations citable) and classifies the facts it returned (`eligible`,
steps.Eligibility.of): `allocatable_ids`, `excluded`, `canonical_of`, `duplicate_groups`. Rendered observations mark
each excluded fact where it appears and each search-result block. Per line the record keeps the eligibility records,
logged only, and each refusal says whether it was a schema or a content refusal and which of the allowed refusals it
used.
"""
import hashlib
import json
import re
import time

from . import steps, tagless
from .llm import CapReached, ContextExceeded, OverBudget, RequestFailed, estimate
from .spread import clean, quote_parts

STEP_CAP = 8                # the step cap per loop; a hard limit, recorded in each run
MARGIN = 0.75               # of num_ctx, the provisional context margin per step
CHECK_RETRIES = 1           # failed answers shown back before the line ends invalid
LEDGER_TEXT = 70            # characters of an observation's first item kept in its ledger line
# a result of these tools stays in full on every later step, the latest call only, not cut to its ledger line: the
# printed page is the frame the line is read in
KEPT_TOOLS = ("printed_page",)
REPEATS = 2                 # the second repeat of one call ends no_progress
# the prompt states the schema caps themselves, in characters, so there are no word warnings; each text field at its cap
# is counted (six_rules_log, "at_cap")
CAPS = {"why": steps.WHY, "quote": steps.QUOTE, "reason": steps.REASON}
TERMINATIONS = ("finished", "finished_at_last_step", "cap_unfinished", "no_progress", "context_exceeded", "invalid",
                "escalated", "not_run", "allocation_pending")
# the header above seeded evidence: m1 without pools; m2 with pools, naming the blocks that may be cited and the search
# results that may not. Hashed in the run record
SEEDED_M1 = "EVIDENCE ALREADY GATHERED (you may cite these IDs)"
SEEDED_M2 = "EVIDENCE ALREADY GATHERED (you may cite {cite}{search})"
SEEDED_M2_FACTS = "{ids} and the fact IDs they list"       # C4: its answer may cite facts
SEEDED_M2_SEARCH = "; {ids} {verb} search result{s}: recall only, not citable; read a passage to cite it"
SEEDED_M2_NONE = "only the line's own fact ID"              # C4, no citable block seeded
SEEDED_M2_NONE_C8 = "no block shown here"                   # C8 cites blocks only, never a fact
SEARCH_MARK = "(search result: read the passage to cite it)"
SEARCH_MARK_V16 = "(search result: read the passage to quote it)"      # prompt v16: there is nothing to cite by ID
M2 = [SEEDED_M2, SEEDED_M2_FACTS, SEEDED_M2_SEARCH, SEEDED_M2_NONE, SEEDED_M2_NONE_C8, SEARCH_MARK]
MESSAGE_SHA256 = {v: hashlib.sha256(t.encode("utf8")).hexdigest() for v, t in
                  (("m1", SEEDED_M1), ("m2", "\n".join(M2)))}

# A structural gap marker printed by Tools.window, "(k rows not shown)": it marks a position in a view, not evidence,
# so Memory.add never turns it into a pointer to an earlier observation (a repeated equal gap is still a gap).
GAP_MARKER = re.compile(r"^\s*\(\d+ rows not shown\)$")


def canonical(action, args):
    """A call's identity, with string arguments stripped as Tools.run strips them, so " X" and "X" are one call."""
    return action + json.dumps({k: v.strip() if isinstance(v, str) else v for k, v in args.items()}, sort_keys=True)


def item_line(it, marks=None):
    """An item as shown: its text, then its fact IDs, each excluded fact with its mark."""
    if not it["facts"]:
        return it["text"]
    return it["text"] + f"  [facts: {', '.join(f'{x} {marks[x]}' if marks and marks.get(x) else x for x in it['facts'])}]"


class Memory:
    """The line's observations. Each is kept in full and reachable by its ID; the transcript shows a ledger line once
    it has been seen. An item already shown in an earlier observation is shown as a pointer to it, once."""

    def __init__(self):
        self.obs = {}           # id -> {"tool", "args", "items", "error"}
        self.first_seen = {}    # item text -> observation ID that showed it first
        self.disclosed = set()  # IDs rendered in full in some request of the line (run_loop keeps it)
        self.marks = {}         # fact ID -> its mark where shown; empty unless pools
        self.marked = False     # pools on: search-result blocks carry SEARCH_MARK
        self.search_mark = SEARCH_MARK      # v16 (`simple`): SEARCH_MARK_V16

    def head(self, oid):
        """An observation's head: its ID and call, a search result marked as such with pools on."""
        o = self.obs[oid]
        # tags hidden: the call as the model named it (tools.run records the tag)
        args = {k: tagless.show(v) if k == "concept" else v for k, v in o["args"].items()}
        h = f"{oid} {o['tool']}({', '.join(f'{k}={v}' for k, v in args.items())})"
        return h + (f" {self.search_mark}" if self.marked and o["tool"] == "search_notes" else "")

    def add(self, o):
        oid = f"O{len(self.obs) + 1}"
        items = []
        for it in o["items"]:
            t = it["text"]
            if GAP_MARKER.match(t):
                items.append(it)
            elif t in self.first_seen and not t.startswith(("none:", "tool error:", "note:", "end:")):
                items.append({**it, "text": f"(shown in {self.first_seen[t]}) {t}"})
            else:
                self.first_seen.setdefault(t, oid)
                items.append(it)
        self.obs[oid] = {**o, "items": items}
        return oid

    def kept(self, oid):
        """True for the latest observation of a KEPT_TOOLS tool: it is shown in full at every later step."""
        o = self.obs.get(oid)
        return bool(o) and o["tool"] in KEPT_TOOLS and oid == max(
            (i for i, x in self.obs.items() if x["tool"] == o["tool"]), key=lambda i: int(i[1:]))

    def full(self, oid, quotable=False):
        """The observation as shown; `quotable`, without the items flagged "annotation" (the member labels and
        note), which are shown but never quotable."""
        o = self.obs[oid]
        return "\n".join([f"{self.head(oid)}:"] + ["  - " + item_line(it, self.marks) for it in o["items"]
                                                    if not (quotable and it.get("annotation"))])

    def ledger(self, oid):
        o = self.obs[oid]
        facts = sorted({x for it in o["items"] for x in it["facts"]})
        first = o["items"][0]["text"] if o["items"] else ""
        first = first if len(first) <= LEDGER_TEXT else first[:LEDGER_TEXT] + "..."
        return (f"{self.head(oid)}: {len(o['items'])} "
                f"item(s); facts {', '.join(facts) or 'none'}; first: {first} (recall {oid} to see it in full)")

    def texts(self, only=None):
        """oid -> full text; `only`, a set of IDs to keep."""
        return {oid: self.full(oid) for oid in self.obs if only is None or oid in only}

    def shown_texts(self):
        """oid -> full text of what the model was actually shown in full: what a citation may quote."""
        return self.texts(self.disclosed)

    def quotable_texts(self):
        """oid -> full text of the observations a quote may come from: those shown in full, less search results, whose
        snippets are pointers to passages, not evidence."""
        return {oid: self.full(oid, quotable=True) for oid in self.obs
                if oid in self.disclosed and self.obs[oid]["tool"] != "search_notes"}

    def passage_texts(self):
        """passage ID -> the passage in full, for each passage read_passage showed in a request that went out: what a
        support entry citing the passage may quote. A snippet, or a neighbour named by a read, is not a read."""
        out = {}
        for oid in sorted(self.disclosed, key=lambda o: int(o[1:])):
            o = self.obs.get(oid, {})
            if o.get("tool") != "read_passage" or o.get("error"):
                continue
            for it in o["items"]:
                if it.get("passage"):
                    out[it["passage"]] = it.get("printed", it["text"])      # the passage as printed, without IDs
        return out


def non_ascii(text):
    """[(position, character)] of every character outside ASCII: a stray script in an answer (a CJK character at a
    binding maxLength) shows here. Logged, never refused."""
    return [(i, c) for i, c in enumerate(text or "") if ord(c) > 127]


def cut_diagnostics(raw):
    """Recorded on a reply that did not parse (an answer cut at the answer cap, bad JSON), never a refusal: its length,
    how far the answer got (allocations, support and chain entries begun, as key counts in the raw text), the characters
    outside ASCII and the last characters before the cut. No field of the reply is decoded: nothing is extracted from
    cut JSON."""
    raw = raw or ""
    return {"chars": len(raw), "allocations_begun": raw.count('"fact_id"'), "support_begun": raw.count('"support"'),
            "chains_begun": raw.count('"chains"'), "non_ascii": non_ascii(raw), "tail": raw[-80:]}


def diagnostics(out, blocks):
    """Recorded on every admitted reply, never a refusal: word counts and character lengths against the schema caps
    (a field at its cap is an indicator, not proof it was cut), and for each quote, the answer's and each piece's,
    which shown blocks hold each of its parts. `blocks`: block name (start, O1, ...) -> the text shown. The prompt
    states the character caps, so there are no word warnings."""
    st = out["step"]
    fields = {"why": out.get("why", "")}
    fields.update({k: st[k] for k in ("quote", "reason") if isinstance(st.get(k), str)})
    d = {"chars": {k: len(v) for k, v in fields.items()}, "words": {k: len(v.split()) for k, v in fields.items()}}
    caps = {k: c["maxLength"] for k, c in CAPS.items()}
    d["at_limit"] = sorted(k for k, v in fields.items() if len(v) >= caps[k])
    d["warnings"] = []
    texts = {k: clean(v) for k, v in blocks.items()}

    def where(q):
        found = [sorted(k for k, t in texts.items() if p in t) for p in quote_parts(q)]
        return {"blocks": sorted(set().union(*map(set, found))) if found else [], "parts": len(found),
                "parts_in_no_block": sum(not w for w in found)}
    d["non_ascii"] = non_ascii(json.dumps(out, ensure_ascii=False))
    if fields.get("quote"):
        d["quote_where"] = where(fields["quote"])
    pq = [(x.get("fact_id"), x["quote"]) for x in st.get("pieces") or () if isinstance(x.get("quote"), str)]
    if pq:
        d["piece_quotes"] = [{"fact_id": i, "chars": len(q), "at_limit": len(q) >= CAPS["quote"]["maxLength"],
                              **(where(q) if q.strip() else {})} for i, q in pq]
    return d


def run_loop(caller, role, system, start, state, tools, check, *, tag, cap=STEP_CAP, num_ctx=None, trace=None,
             extra_text="", eligible=None):
    """One line through the loop. `start`: the start-state text; `state`: the step state of steps.py, the v16 pools
    state of analyst.start_state; `tools`: a tools.Tools or None (an answer-only loop); `check(answer, state, shown,
    memory)`: code's checks on an admitted answer, a list of errors. Returns the line's record: decision, answer,
    termination, steps used, failure counts and the per-step trace.

    The loop goes as chat turns: the start state, then each step's own JSON reply as an assistant turn followed by a
    user turn with what code did (the observation, or the refusal); only the last turn carries the schema.

    Each observation grows the pools (grow_pools) and, for a state that holds `allocatable_ids`, classifies the facts
    it returned (`eligible(fid)`, steps.Eligibility.of; default steps.trivial_eligibility, for fake-model tests).

    Prompt v16, the six rules: one answer form (row, pieces, quote, reason). admit runs only the role's check
    (six_rules.checker: rules 5 and 6, one sentence per refusal); search results carry SEARCH_MARK_V16; excluded
    facts carry steps.PIECE_MARKS; an answer without `pieces` moves nothing. Refusals are counted by rule
    (rule5_refusal, rule6_refusal), with quote-not-found and quote-from-search also counted apart; `six_rules` in
    the record keeps the logged-only flags (six_rules_log)."""
    state = {k: list(v) if isinstance(v, (list, tuple)) else v for k, v in state.items()}
    if not (state.get("simple") and state.get("pools")) or any(state.get(k) for k in ("staged", "pointers", "evidence")):
        raise ValueError("the v16 state runs with pools and without staged, pointers or evidence")
    state.setdefault("observation_ids", [])
    mem, steps_log, history = Memory(), [], []
    elig = {}                               # fact ID -> its eligibility record
    for key in ("recall_ids", "discovered_passages", "read_passages"):
        state.setdefault(key, [])
    state.setdefault("citable_ids", [state["line_fact"]] if state.get("line_fact") else [])
    mem.marked, mem.search_mark = True, SEARCH_MARK_V16
    classify = (eligible or steps.trivial_eligibility(state.get("line_fact"))) if "allocatable_ids" in state else None
    first_oid = {}                          # canonical call -> the observation its first request returned
    num_ctx = num_ctx or caller.num_ctx
    margin = int(MARGIN * num_ctx)
    seen, fails, pending = {}, 0, None       # canonical call -> times requested; failed answers; full obs to show
    counts = {"repeated_call": 0, "unreturned_id": 0, "answer_before_lookup": 0, "schema_failure": 0,
              "check_failure": 0, "tool_error": 0, "unread_citation": 0, "unbound_quote": 0, "bad_pointer": 0,
              "amount_type_unresolved": 0, "schema_refusal": 0, "content_refusal": 0,
              "rule5_refusal": 0, "rule6_refusal": 0, "quote_not_found": 0, "quote_from_search": 0,     # by rule
              "period": 0, "sign_conflict": 0, "more_than_one_section": 0, "exceeds_line": 0,
              "quote_no_content": 0, "quote_guard_whole_number": 0,         # the two quote guards
              "quote_ambiguous": 0}         # projection tier: lines binding different facts
    referred = {}
    rec = {"tag": tag, "role": role, "model": caller.models.get(role), "decision": None, "row": None, "target": None,
           "answer": None, "termination": None, "steps_used": 0, "counts": counts, "steps": steps_log,
           "num_ctx": num_ctx, "margin_tokens": margin, "machine": None}

    def end(termination, why=None):
        rec["termination"], rec["why"] = termination, why
        rec["observations"] = {oid: mem.obs[oid] for oid in mem.obs}
        rec["evidence_texts"] = mem.quotable_texts()  # what the model was shown, less search results: citable later
        rec["passage_texts"] = mem.passage_texts()    # the passages read in full
        rec["disclosed"] = sorted(mem.disclosed, key=lambda o: int(o[1:]))
        rec["state"] = state            # the line's final step state: what its answer could cite (exchange, review)
        rec["under_use"] = under_use(rec, steps_log)
        rec["interface"] = {"message_version": "m2", "message_sha256": MESSAGE_SHA256["m2"]}
        rec["eligibility"] = {"records": elig, "canonical_of": state.get("canonical_of", {}),
                              "excluded": state.get("excluded", {}),
                              "duplicate_groups": state.get("duplicate_groups", {}),
                              "unresolved": sorted(f for f, r in elig.items() if r.get("unresolved")),
                              "label": "type-eligible, never inside the line"}
        rec["six_rules"] = six_rules_log(rec, state)
        if trace is not None:
            trace.append(rec)
        return rec

    def commit(a, last):
        """Decision, row and answer written together, once."""
        row = a.get("row")
        decision = "referred" if row in referred else row if row in steps.OPTIONS else "row"
        rec.update(answer=a, row=row, decision=decision, target=row if decision == "referred" else None)
        return end("finished_at_last_step" if last else "finished")

    def admit(a, shown, log=None):
        """Code's checks on an answer: the role's check, its refusals counted by rule and kind and logged on the
        step."""
        errs = check(a, state, shown, mem)
        for r in a.get("_refusals") or ():
            counts[f"rule{r['rule']}_refusal"] += 1
            if r["kind"] in ("quote_not_found", "quote_from_search", "period", "sign_conflict",
                             "more_than_one_section", "exceeds_line", "quote_no_content", "quote_ambiguous"):
                counts[r["kind"]] += 1
            if r.get("guard"):
                counts["quote_guard_" + r["guard"]] += 1
        if log is not None and a.get("_refusals"):
            log["refusals"] = a["_refusals"]
        return errs

    for k in range(1, cap + 1):
        last = k == cap
        schema = steps.step_schema(state, last=last)
        turns, user = render_turns(start, history, mem, pending, k, cap, last, schema, extra_text)
        rendered = pending if pending in mem.obs else None     # in full in this request; disclosed once it is sent
        quotable = mem.quotable_texts()
        if rendered and mem.obs[rendered]["tool"] != "search_notes":
            quotable[rendered] = mem.full(rendered)
        shown = start + "\n" + "\n".join(quotable.values())
        est = estimate(system, "".join(t["content"] for t in turns) + user, schema)
        log = {"step": k, "last": last, "estimated_tokens": est, "margin_tokens": margin,
               "messages_sha256": hashlib.sha256("\x00".join([system] + [t["content"] for t in turns] + [user])
                                                 .encode("utf8")).hexdigest(),
               "user_chars": len(user), "turns": len(turns)}
        steps_log.append(log)
        rec["steps_used"] = k
        if est > margin:
            log["ended"] = "context margin"
            return end("context_exceeded", f"step {k}: about {est} tokens, margin {margin} of num_ctx {num_ctx}")
        t0 = time.time()
        try:
            out, call = caller.ask(role, system, user, schema, tag=f"{tag}:s{k}", budget=margin,
                                   **({"turns": turns} if turns else {}))
        except ContextExceeded as e:
            # not sent (OverBudget: the estimate was over) or sent and refused (the engine refused it, or counted it
            # over the budget after answering): a request that went out showed its observation in full
            sent = not isinstance(e, OverBudget)
            log.update(ended="context refused by the engine" if sent else "over the estimate limit, not sent",
                       sent=sent, error=str(e))
            if sent and rendered:
                mem.disclosed.add(rendered)
            return end("context_exceeded", str(e))
        except (CapReached, RequestFailed) as e:
            log.update(ended="caller stopped", error=str(e))
            rec["stop_class"] = type(e).__name__        # the run maps the end by this type
            return end("not_run", str(e))
        if rendered:
            mem.disclosed.add(rendered)     # the request went out with it in full
        log.update(wall_s=round(time.time() - t0, 2), prompt_eval_count=call.get("prompt_tokens"),
                   eval_count=call.get("completion_tokens"), raw=call.get("raw"), call_error=call.get("error"),
                   done_reason=call.get("done_reason"), recovery=call.get("recovery"),
                   dispatches=call.get("dispatches"))       # the step keeps its recovery and dispatches
        if len(steps_log) > 1 and "repeat_of" in steps_log[-2]:
            steps_log[-2]["next_action"] = out["step"]["action"] if out else "refused"   # after the reminder
        pending = None
        if out is None:
            counts["schema_failure"] += 1
            log["cut_diagnostics"] = cut_diagnostics(call.get("raw"))
            errs = [f"your reply was refused: {call.get('error')}"]
        else:
            st = out["step"]
            log.update(why=out["why"], action=st["action"], pieces=len(st.get("pieces") or ()),
                       diagnostics=diagnostics(out, {"start": start, **mem.shown_texts()}))
            if st["action"] != "answer":
                args = {a: v for a, v in st.items() if a != "action"}
                key = canonical(st["action"], args)
                log.update(tool=st["action"], args=args, repeat=seen.get(key, 0))
                if key in seen and st["action"] != "recall":
                    counts["repeated_call"] += 1
                    seen[key] += 1
                    if seen[key] > REPEATS:
                        log["ended"] = "repeated call"
                        return end("no_progress", f"{st['action']} {args} requested {seen[key]} times")
                    # the first repeat is not run again: a reminder and the earlier result in full
                    pending = first_oid[key]
                    log.update(ended="repeat reminder", repeat_of=pending, returned=[pending])
                    history.append({"k": k, "why": out["why"], "action": st["action"], "args": args, "oid": pending,
                                    "raw": call.get("raw"), "repeat_of": pending})
                    continue
                seen[key] = seen.get(key, 0) + 1
                history.append({"k": k, "why": out["why"], "action": st["action"], "args": args, "oid": None,
                                "raw": call.get("raw")})
                if st["action"] == "recall":
                    pending = args["observation_id"]
                    log.update(returned=[pending])
                else:
                    try:
                        o = tools.run(st["action"], args)
                    except (CapReached, RequestFailed) as e:     # an embedding refused by the ledger or the engine
                        log.update(ended="tool stopped", error=str(e))
                        rec["stop_class"] = type(e).__name__    # as above
                        return end("not_run", str(e))
                    counts["tool_error"] += bool(o["error"])
                    oid = mem.add(o)
                    pending = oid
                    facts = sorted({x for it in o["items"] for x in it["facts"]})
                    grow(state, "observation_ids", [oid])
                    grow(state, "fact_ids", facts)
                    grow(state, "evidence_ids", facts + [oid])
                    grow(state, "passage_ids", o.get("passages", []))     # read_passage may open these
                    grow(state, "evidence_ids", [it["passage"] for it in o["items"] if it.get("passage")])
                    grow_pools(state, oid, o, classify, elig, mem, counts)
                    if o["rows"]:
                        referred.update({r: args["section"] for r in o["rows"]})
                        grow(state, "rows", list(o["rows"]))
                    log.update(returned=[oid] + facts, result_chars=len(mem.full(oid)), tool_error=o["error"])
                    first_oid[key] = oid
                history[-1]["oid"] = pending
                log["ended"] = "tool called"
                continue
            if not mem.obs:
                counts["answer_before_lookup"] += 1
            st.setdefault("pieces", [])         # v16's answer offers pieces only when some fact is allowed
            errs = admit(st, shown, log)
            if not errs:
                log["ended"] = "answered"
                return commit(st, last)
            counts["check_failure"] += 1
        log.update(ended="answer refused", errors=errs)
        fails += 1
        kind = "schema" if out is None else "content"   # schema apart from content, and which allowed refusal
        counts[kind + "_refusal"] += 1
        log["refusal"] = {"kind": kind, "number": fails, "allowed": CHECK_RETRIES + 1}
        history.append({"k": k, "why": out["why"] if out else "", "action": "refused", "args": {"errors": errs},
                        "oid": None, "raw": call.get("raw")})
        if last:
            return end("cap_unfinished", "; ".join(errs))
        if fails > CHECK_RETRIES:
            return end("invalid", "; ".join(errs))
    return end("cap_unfinished", "no answer within the cap")      # not reached: the last step offers only answers


def grow_pools(state, oid, o, classify, elig, mem, counts):
    """One observation into the four pools. Every O-ID is recallable and every passage it names discoverable; a passage
    read_passage opened is read and citable; a non-search observation and the facts it lists are citable; a search
    result and its passages are never citable. For a C4 state (`classify` given), each fact is classified once:
    allocatable, or excluded with its reason and its mark where shown; an alias's canonical ID is allocatable in its
    place; a duplicate group is kept in the state. An excluded fact is shown with its steps.PIECE_MARKS mark."""
    facts = sorted({x for it in o["items"] for x in it["facts"]})
    grow(state, "recall_ids", [oid])
    grow(state, "discovered_passages", o.get("passages") or [])
    if o["tool"] in ("calc_children", "breakdowns"):     # kept in the recorded state
        grow(state, "chainable_ids", [oid])
    if o.get("read") and not o.get("error"):
        grow(state, "read_passages", [o["read"]])
        grow(state, "citable_ids", [o["read"]])
    if o["tool"] == "search_notes" or classify is None:
        if o["tool"] != "search_notes":
            grow(state, "citable_ids", [oid] + facts)
        return
    grow(state, "citable_ids", [oid] + facts)
    todo = list(facts)
    while todo:
        fid = todo.pop(0)
        if fid in elig:
            continue
        r = elig[fid] = classify(fid)
        counts["amount_type_unresolved"] += bool(r.get("unresolved"))
        why = r.get("why")
        if r.get("allocatable"):
            grow(state, "allocatable_ids", [fid])
        elif why:
            state.setdefault("excluded", {})[fid] = why
        if why and why.startswith("alias of "):
            state.setdefault("canonical_of", {})[fid] = r["canonical"]
            mem.marks[fid] = f"= {r['canonical']}"
            todo.append(r["canonical"])
        elif why in steps.PIECE_MARKS:
            mem.marks[fid] = steps.PIECE_MARKS[why]
        if r.get("allocatable") and r.get("quote_mark"):    # movable-fact quote (--root-cause-fixes): the validator's own exemption
            mem.marks[fid] = r["quote_mark"]
        if r.get("group"):
            state.setdefault("duplicate_groups", {})[fid] = list(r["group"])


def six_rules_log(rec, state):
    """Prompt v16, logged only, never a refusal: an answer whose reason or quote names parts while it gives no pieces;
    pieces that leave part of the line to the remainder row (the remainder, by code); the pieces accepted on the model's
    judgement; `at_cap`, how many admitted replies had each text field (why, quote, reason, a piece's quote) at its
    schema cap, out of `replies`. Refusals by rule and the two quote refusals are in `counts`."""
    diag = [s["diagnostics"] for s in rec.get("steps") or () if s.get("diagnostics")]
    at_cap = {k: sum(k in d["at_limit"] for d in diag) for k in ("why", "quote", "reason")}
    at_cap["piece_quote"] = sum(any(q["at_limit"] for q in d.get("piece_quotes") or ()) for d in diag)
    a = rec.get("answer") or {}
    words = " ".join(str(a.get(k) or "") for k in ("reason", "quote"))
    pieces = a.get("_pieces") or []
    left = a.get("_remainder")
    return {"names_parts_without_pieces": bool(a) and not pieces and bool(PARTS.search(words)),
            "pieces_short_of_line": ({"remainder": str(left), "row": a.get("row")} if pieces and left else None),
            "model_judgement_pieces": sorted(p["fact"] for p in pieces if p.get("model_judgement")),
            "at_cap": at_cap, "replies": len(diag),
            "label": "logged only; neither flag establishes a mistake"}


PARTS = re.compile(r"\b(includ\w*|portions?|parts?|split\w*|components?|consist\w*|compris\w*|combin\w*)\b", re.I)


def under_use(rec, steps_log):
    """Logging only: whether search was used and the outcome, the why before an answer given without search, and whether
    the answer's own words name parts while the line was committed whole. These flags are mechanical diagnostics, not a
    finding that the line was misplaced or a tool under-used."""
    tools_used = [s["tool"] for s in steps_log if s.get("tool")]
    searched = "search_notes" in tools_used
    a = rec.get("answer") or {}
    answered = [s for s in steps_log if s.get("action") == "answer"]
    before = [s for s in steps_log[:steps_log.index(answered[0])] if s.get("tool")] if answered else []
    words = " ".join(str(a.get(k) or "") for k in ("reason", "quote"))
    whole = rec["termination"] in ("finished", "finished_at_last_step") and not a.get("pieces")
    return {"searched": searched, "tools": tools_used, "outcome": rec.get("decision"),
            "why_at_answer": answered[0].get("why") if answered and not searched else None,
            "why_before_answer": before[-1].get("why") if before and not searched else None,
            "names_parts_committed_whole": bool(whole and (rec.get("decision") == "COMBINED" or PARTS.search(words))),
            "allocation_pending": False, "allocation_abandoned": False,     # no staged machine: kept as keys only
            "label": "diagnostic only"}


def grow(state, key, values):
    cur = state.setdefault(key, [])
    for v in values:
        if v not in cur:
            cur.append(v)


def reminder(history, h):
    """The repeat guard's message: scoped to the lookup, it says the result is unchanged, nothing about the line."""
    first = next(x["k"] for x in history if x["oid"] == h["repeat_of"] and not x.get("repeat_of"))
    return (f"You asked for this lookup at step {first}; it returned the same result, shown again below, and was "
            "not run again. Asking for it a third time ends this line.")


def where(k, cap, last, schema):
    """Where the loop stands, and the schema of this step's reply: the end of the last user message. The words match
    the actions the schema offers: at the last step, only an answer."""
    out = [f"THIS IS STEP {k} OF {cap}."]
    if last:
        out.append("This is the last step: only an answer is accepted now. NOT_TOLD is a valid answer if the "
                   "allowed statements do not establish it.")
    else:
        out.append("Call one tool, or answer.")
    return out + ["", "SCHEMA OF YOUR REPLY", json.dumps(schema, separators=(",", ":"))]


def render_turns(start, history, mem, pending, k, cap, last, schema, extra_text=""):
    """(earlier turns, last user message) of step k in the turns form. The first user turn is the start state; each
    earlier step is the model's own reply (assistant), then what code did with it (user): the observation in full
    if it is the one just returned, else its ledger line (a recall shows the recalled observation the same way), or
    the refusal with its errors. Only the last user message carries the schema."""
    roles, msgs = ["user"], [start + (f"\n\n{extra_text}" if extra_text else "")]
    for h in history:
        roles.append("assistant")
        msgs.append(h["raw"] or json.dumps({"why": h["why"], "step": {"action": h["action"]}}))
        if h["action"] == "refused":
            body = "Your reply was refused by code's checks: " + "; ".join(h["args"]["errors"])
        elif h["oid"] in mem.obs:
            fresh = (h["oid"] == pending and h is history[-1]) or (mem.kept(h["oid"]) and not h.get("repeat_of"))
            body = ("RESULT OF YOUR CALL\n" + (reminder(history, h) + "\n" if h.get("repeat_of") else "")
                    + (mem.full(h["oid"]) if fresh else mem.ledger(h["oid"])))
        else:
            body = "RESULT OF YOUR CALL\n(none)"
        roles.append("user")
        msgs.append(body)
    msgs[-1] += "\n\n" + "\n".join(where(k, cap, last, schema))
    return [{"role": r, "content": c} for r, c in zip(roles[:-1], msgs[:-1])], msgs[-1]
