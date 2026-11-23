"""C7's role wrappers: the generator with its one meaning question, the critic,
the controller and the meaning role, each a real model call through llm.Caller. C8's review wrapper is not built
here: `c8_unavailable` answers `unavailable`, which repair.Tree.commit turns into `withdrawn` with the cause kept and
the original draft unchanged.

Every wrapper follows one contract, repair.repair's: it is called as role(request, [ask,] ledger=meter), builds its
prompt from the request, asks through a Caller whose `ledger` is the call's repair.Meter (so every transport attempt,
a retry included, is reserved through the meter and forwarded to the batch's shared ledger before it is sent),
checks the reply against the role's schema (the Caller's check, retried once inside its two requests; a hosted
question may add one 429 recovery, `question_sends`), and returns
(reply, usage). Usage is read from what the Caller and the embedder recorded for this call, never from the model:
`requests` the attempts the Caller sent, `tokens` the engine's own input plus output counts, `embedding_requests` and
`embedding_tokens` the embedder's requests made inside this call. A reply that is still not valid after the retry
raises RoleFailed, which repair.Budget records and turns into `role_failed`.

Models are run settings (B1): repair.ROLE_MODELS, plus MEANING_MODEL for the meaning question, overridden by the
run's `models`; a model swap is a change of that setting, never of code. Nothing here decides a placement:
the generator and the meaning role choose their own tools, question and when to stop within a step cap; the
controller ranks, stops and selects; code runs the tools, checks shapes and counts.

    roles = Roles(run / "calls.jsonl", tools=Tools(f, line, rows_by_section, section, names), stop=stop)
    repair.repair(draft, **roles.kwargs(), models=roles.models, dispatches=2, ledger=ledger, company=company)
"""
import json
from contextlib import contextmanager

from . import repair, steps, tagless
from .llm import DISPATCHES_PER_QUESTION, NUM_PREDICT, RATE_LIMIT_RECOVERY, Caller, ContextExceeded, count
from .packs import prompt
from .spread import LOOP_MODELS

PROMPTS = {"generator": "c7_generator_v9", "critic": "c7_critic_v2", "controller": "c7_controller_v3",
           "meaning": "c7_meaning_v5"}     # the generator: the split's support rule and the
#                                            critic's feedback; printed_page names its line;
#                                            tags are hidden: no definition tool, lines named
#                                            by ID; the controller makes the rank choice;
#                                            the critic and the controller name no tag
#
GENERATOR_PLAIN = "c7_generator_v10"     # a generator request without feedback: the generator prompt
#                                         without the feedback paragraph, tags hidden
#                                         (used when feedback is off)
MEANING_MODEL = LOOP_MODELS["analyst"]      # the meaning question goes to the analyst
MODELS = {**repair.ROLE_MODELS, "meaning": MEANING_MODEL}
NUM_CTX = 4096                  # the context TOKENS_PER_REQUEST is taken from (repair)
OPTIONS = {"presence_penalty": 0.0}     # as spread.main: an option left out falls back to the modelfile's
QUESTION = {"type": "string", "maxLength": 600}    # as ANSWER: 600 characters, as questions can run
#                                                   past 300 characters
ANSWER = {"type": "string", "maxLength": 600}
# the statements-only mode (tools.Tools statements_only, the source spread run's): the same prompts without
# breakdowns, other_periods and the notes; a role whose tools are statements-only is asked with these
STATEMENTS_PROMPTS = {"c7_generator_v9": "c7_generator_v11", "c7_generator_v10": "c7_generator_v12",
                      "c7_meaning_v5": "c7_meaning_v6"}
NO_TOOLS = ("recall",)          # loop.py's recall needs its store; here every observation stays in the turns
KINDS = {"search_notes": "search_result", "read_passage": "passage"}    # disclosed kind by tool; others are "tool"
C8_UNAVAILABLE = "no checker review is configured for this repair"
MASKED = "[This earlier result ({}) was removed to make room in the context; it can no longer be cited.]"  # the net


class RoleFailed(Exception):
    """A role call ended with no usable reply: the Caller's answer retry used, or no final action given."""


class RecordingCaller(Caller):
    """llm.Caller that also keeps this call's log records in memory, so usage is read from what it recorded."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.records = []

    def log(self, rec):
        self.records.append(rec)
        super().log(rec)


def zero_usage():
    return {k: 0 for k in repair.USAGE_KEYS}


def c8_unavailable(req, ledger=None):
    """The C8 review slot when no checker review is configured: `unavailable`, no call, no use. repair.Tree.commit
    withdraws the repair and keeps the original draft."""
    return {"status": "unavailable", "why": C8_UNAVAILABLE}, zero_usage()


def text_of(obs):
    """An observation's items as the model sees them: each item's text with the fact IDs it shows."""
    lines = [it["text"] + (f" [facts: {', '.join(it['facts'])}]" if it.get("facts") else "")
             for it in obs.get("items", [])]
    lines += [f"{r}: {t}" for r, t in (obs.get("rows") or {}).items()]
    return "\n".join(lines)


class Roles:
    """The four model roles of one C7 run, with their settings. `log_path`: the run's calls.jsonl, shared by every
    Caller (append). `tools`: the analyst's tools for the section (tools.Tools: available(), run(action, args)), or
    None to offer the generator and the meaning role only their final actions and the question. With a search, its
    embedder is put on the call's meter while a role call runs, so query embeddings are reserved through it too."""

    def __init__(self, log_path, models=None, tools=None, stop=None, company=None, num_ctx=NUM_CTX,
                 answer_cap=NUM_PREDICT, options=None, engine="ollama", caller_kwargs=None, context_net=False,
                 role_ctx=None):
        self.models = {**MODELS, **(models or {})}
        self.log_path, self.tools, self.stop, self.company = log_path, tools, stop, company
        self.num_ctx, self.answer_cap, self.engine = num_ctx, answer_cap, engine
        # role_ctx: a role's own context, others at num_ctx (for example the meaning role
        # alone at 16,384); its Caller, reservation and input limit all take it
        self.role_ctx = dict(role_ctx or {})
        if set(self.role_ctx) - set(self.models):
            raise ValueError(f"role_ctx names no role: {sorted(set(self.role_ctx) - set(self.models))}")
        # engine: one for every role, or role -> engine for a mixed tree (for example generator,
        # controller and meaning on the hosted analyst, the critic local); caller_kwargs then
        # go to the OpenRouter roles only. Every role needs its engine
        if isinstance(engine, dict) and set(self.models) - set(engine):
            raise ValueError(f"a mixed tree names every role's engine: {sorted(set(self.models) - set(engine))} "
                             f"missing")
        self.options = dict(OPTIONS if options is None else options)
        self.caller_kwargs = dict(caller_kwargs or {})    # e.g. an OpenRouter run's spend cap, price and max_tokens
        self.callers = []                                 # every Caller made, in order, for the run's record
        self.context_net = context_net    # a step too large for the context: mask the oldest result, ask the final
        self.net_events = []              # every use of the net: role, node, step, masked blocks, the refusal
        self.loops = 0                    # loop calls so far: each call's ID ("generator#3") is its records' tag

    def kwargs(self):
        """repair.repair's role arguments."""
        return {"generator": self.generator, "critic": self.critic, "controller": self.controller,
                "c8_review": c8_unavailable, "meaning": self.meaning}

    # ---- plumbing: one Caller per role call, usage from its records ----

    def caller(self, role, ledger, ceiling):
        """A fresh Caller for one role call: its call cap is the call's ceiling and its ledger the call's meter, so
        a nested call (the meaning question inside the generator's) is metered on its own."""
        mixed = isinstance(self.engine, dict)
        engine = self.engine_of(role)
        c = RecordingCaller({role: self.models[role]}, self.log_path, ceiling, num_ctx=self.ctx(role), stop=self.stop,
                            engine=engine, answer_cap=self.answer_cap, ledger=ledger, company=self.company,
                            options=self.options,
                            **(self.caller_kwargs if not mixed or engine == "openrouter" else {}))
        self.callers.append(c)
        return c

    def ctx(self, role):
        return self.role_ctx.get(role, self.num_ctx)

    def engine_of(self, role):
        return self.engine[role] if isinstance(self.engine, dict) else self.engine

    def embedder(self):
        return getattr(getattr(self.tools, "search", None), "embedder", None)

    @contextmanager
    def metered(self, ledger):
        """The search's embedder on this call's meter for the call's length, then back on what it had."""
        e = self.embedder()
        if e is None or ledger is None:
            yield
            return
        old = e.ledger
        e.ledger = ledger
        try:
            yield
        finally:
            e.ledger = old

    def usage(self, caller, embedded):
        """The four counters from the records: requests the Caller sent, the engine's input plus output tokens, and
        the embedder requests (with their engine counts) made by this call, a nested call's excluded."""
        toks = sum(r["prompt_tokens"] + r["completion_tokens"] for r in caller.records
                   if count(r.get("prompt_tokens")) and count(r.get("completion_tokens")))
        return {"requests": caller.calls, "tokens": toks, "embedding_requests": len(embedded),
                "embedding_tokens": sum(c["tokens"] for c in embedded if count(c.get("tokens")))}

    def ask_once(self, caller, role, user, schema, turns=None, tag=None, system=None):
        """One question: the Caller checks the answer against the schema and retries a bad one once; still none
        is a role failure. The input limit is the context less the answer cap. The tag is logged, never sent.
        `system`: a prompt name other than the role's (the generator without feedback). The request goes with
        line IDs for tags and the answer comes back with them (tagless.unhide reads them back)."""
        user, schema, turns = tagless.hide(user), tagless.hide(schema), turns and tagless.hide(turns)
        name = system or PROMPTS[role]
        if getattr(self.tools, "statements_only", False):
            name = STATEMENTS_PROMPTS.get(name, name)
        answer, rec = caller.ask(role, prompt(name), user, schema, tag=tag or role, turns=turns,
                                 budget=self.ctx(role) - self.answer_cap)
        if answer is None:
            raise RoleFailed(f"{role}: no valid reply after the Caller's retry: {rec.get('error')}")
        return answer

    # ---- the step loop of the generator and the meaning role ----

    def tool_branches(self, passages):
        if self.tools is None:
            return []
        sections = self.tools.other_sections() if hasattr(self.tools, "other_sections") else []
        state = {"tools": [t for t in self.tools.available() if t not in NO_TOOLS], "passage_ids": passages,
                 "sections": sections, "page_line": bool(getattr(self.tools, "lines", None))}
        return steps.tool_branches(state)

    def observe(self, step, prefix, k, disclosed, passages):
        """Run one tool call. Its output is disclosed as one block (kind tool, search_result or passage); a passage
        read is disclosed under its own passage ID. Returns the text the model sees next."""
        args = {x: v for x, v in step.items() if x != "action"}
        obs = self.tools.run(step["action"], args)
        passages += [p for p in obs.get("passages", []) if p not in passages]
        kind = KINDS.get(step["action"], "tool")
        if kind == "passage" and not obs.get("error"):
            out = []
            for it in obs["items"]:
                if it.get("passage"):
                    disclosed[it["passage"]] = {"kind": "passage", "text": it["text"]}
                    out.append(f"{it['passage']} (passage, citable): {it['text']}")
                else:
                    out.append(it["text"])
            return "\n".join(out)
        bid = f"{prefix}{k}"
        disclosed[bid] = {"kind": "tool" if kind == "passage" else kind, "text": text_of(obs)}
        cite = "not citable: read a passage to cite it" if kind in repair.NOT_CITABLE else "citable"
        return f"{bid} ({step['action']}, {cite}):\n{disclosed[bid]['text']}"

    def loop(self, role, caller, start, final, steps_cap, per, prefix, ask=None, lines=(), nested=None, system=None):
        """The model's capped loop: each step one action of the bound step schema, a tool, the question (once) or
        the final action. The last step, or a step with too few requests left for one more question after it,
        offers only the final action. Returns (final step, disclosed blocks)."""
        disclosed, passages, turns, user = {}, [], [], start
        asked = ask is None
        self.loops += 1
        call = f"{role}#{self.loops}"   # this call's records and net events carry it
        results = []        # for the context net: each earlier result's message index in `turns` and its block IDs
        new = []            # the block IDs disclosed by the result `user` carries
        for k in range(1, steps_cap + 1):
            last = k == steps_cap or caller.max_calls - caller.calls < 2 * per
            branches = [] if last else self.tool_branches(passages)
            if not last and not asked:
                branches.append(steps.obj({"action": steps.one("ask"), "line": steps.enum(lines),
                                           "question": QUESTION}))
            branches.append(final)
            msg = user + f"\n\n[step {k} of at most {steps_cap}" + ("; the last: give your final action]" if last
                                                                    else "]")
            schema = steps.obj({"step": {"anyOf": branches}})
            while True:
                try:
                    answer = self.ask_once(caller, role, msg, schema, turns, tag=call, system=system)
                    break
                except ContextExceeded as e:
                    if not self.context_net or not results:
                        raise
                    # the context net: the oldest earlier result is masked and its blocks
                    # leave `disclosed`, so nothing unseen can be cited; the newest is kept; only the final action
                    i, ids = results.pop(0)
                    turns[i] = {"role": "user", "content": MASKED.format(", ".join(ids) or "no citable block")
                                + turns[i]["content"][turns[i]["content"].rfind("\n\n[step "):]}
                    for x in ids:
                        disclosed.pop(x, None)
                    passages[:] = [p for p in passages if p not in ids]
                    self.net_events.append({"role": role, "call": call, "node": prefix, "step": k, "masked": ids,
                                            "why": str(e)})
                    msg = user + f"\n\n[step {k}: no more steps; give your final action]"
                    schema = steps.obj({"step": {"anyOf": [final]}})
            step = answer["step"]
            if step["action"] == final["properties"]["action"]["enum"][0]:
                return tagless.unhide(step), disclosed       # a tool's step keeps its IDs: the tools read them
            turns += [{"role": "user", "content": msg}, {"role": "assistant", "content": json.dumps(answer)}]
            if k > 1:
                results.append((len(turns) - 2, new))
            seen = set(disclosed)
            if step["action"] == "ask":
                asked = True
                user = self.question(ask, step, prefix, disclosed, nested)
            else:
                user = self.observe(step, prefix, k, disclosed, passages)
            new = [x for x in disclosed if x not in seen]
        raise RoleFailed(f"{role}: no final action within {steps_cap} steps")

    def question(self, ask, step, prefix, disclosed, nested):
        """The one meaning question, through the tree (its own call and meter). The analyst's observations are
        disclosed under this call's IDs (a passage under its own); the analyst's answer is shown, not citable."""
        e = self.embedder()
        n0 = len(e.calls) if e is not None else 0
        try:
            got = ask(tagless.unhide(step["line"]), step["question"])
        finally:
            if e is not None and nested is not None:
                nested.append((n0, len(e.calls)))
        if not isinstance(got, dict) or "refused" in got:
            return f"question not asked: {got.get('refused') if isinstance(got, dict) else got}"
        out = [f"analyst's answer (not citable): {got.get('answer')}"]
        # byte-identical observations of one kind are shown once, under every ID; each ID stays disclosed with its
        # full text (six identical rows_of results once overflowed the generator)
        shown = {}
        for i, o in enumerate(got.get("observations", []), 1):
            bid = o["id"] if o["kind"] == "passage" else f"{prefix}Q{i}"
            disclosed[bid] = {"kind": o["kind"], "text": o["text"]}
            shown.setdefault((o["kind"], o["text"]), []).append(bid)
        for (kind, text), ids in shown.items():
            cite = "not citable" if kind in repair.NOT_CITABLE else "citable"
            same = f"; the same text under each of {len(ids)} IDs" if len(ids) > 1 else ""
            out.append(f"{', '.join(ids)} ({kind}, {cite}{same}):\n{text}")
        return "\n".join(out)

    def embedded(self, n0, nested=()):
        """The embedder's requests since n0, without those made inside a nested call."""
        e = self.embedder()
        if e is None:
            return []
        skip = {i for a, b in nested for i in range(a, b)}
        return [c for i, c in enumerate(e.calls) if i >= n0 and i not in skip]

    def mark(self):
        e = self.embedder()
        return len(e.calls) if e is not None else 0

    # ---- the roles ----

    def generator(self, req, ask, ledger=None):
        """The repair model with the analyst's tools: up to GENERATOR_STEPS steps of tool calls, the
        one question and the proposal. Reply {thoughts, disclosed}: `disclosed` is what its tools returned, written
        by code. The lines it may ask about are the ones its schema lets it place."""
        ceiling, schema = req["ceiling"]["requests"], req["schema"]["properties"]["thoughts"]
        lines = schema["items"]["anyOf"][0]["properties"]["line"]["enum"]
        with self.metered(ledger):
            n0, nested = self.mark(), []
            c = self.caller("generator", ledger, ceiling)
            user = json.dumps({k: v for k, v in req.items() if k not in ("role", "schema", "ceiling", "attempt")})
            final = steps.obj({"action": steps.one("propose"), "thoughts": schema})
            prefix = f"T{req['node']}." if "attempt" not in req else f"T{req['node']}r{req['attempt']}."   # a retry's
            step, disclosed = self.loop("generator", c, user, final, repair.GENERATOR_STEPS,   # IDs never reuse one
                                        max(1, ceiling // repair.GENERATOR_STEPS), prefix, ask=ask,
                                        lines=lines, nested=nested,
                                        system=None if req.get("feedback") else GENERATOR_PLAIN)
            return {"thoughts": step["thoughts"], "disclosed": disclosed}, self.usage(c, self.embedded(n0, nested))

    def meaning(self, req, ledger=None):
        """The one meaning question: one analyst loop of at most MEANING_REQUESTS steps over the same
        tools. Reply {answer, evidence_ids, quote, observations}; the observations are code-written tool output."""
        ceiling = req["ceiling"]["requests"]
        with self.metered(ledger):
            n0 = self.mark()
            c = self.caller("meaning", ledger, ceiling)
            user = json.dumps({k: req.get(k) for k in ("statement", "section", "line", "question")}
                              | {"line": tagless.ref(req["line"])})   # its ID and label
            final = steps.obj({"action": steps.one("answer"), "answer": ANSWER,
                               "evidence_ids": {"type": "array", "items": repair.STR}, "quote": steps.QUOTE})
            step, disclosed = self.loop("meaning", c, user, final, repair.MEANING_REQUESTS,
                                        max(1, ceiling // repair.MEANING_REQUESTS), "M")
            use = self.usage(c, self.embedded(n0))
        obs = [{"id": i, "kind": b["kind"], "text": b["text"]} for i, b in disclosed.items()]
        return {"answer": step["answer"], "evidence_ids": step["evidence_ids"], "quote": step["quote"],
                "observations": obs}, use

    def single(self, role, req, schema, ledger):
        """One question, no tools: the critic and the controller."""
        with self.metered(ledger):
            n0 = self.mark()
            c = self.caller(role, ledger, req_ceiling(ledger, question_sends(self.engine_of(role))))
            user = json.dumps({k: v for k, v in req.items() if k not in ("role", "schema")})
            return tagless.unhide(self.ask_once(c, role, user, schema)), self.usage(c, self.embedded(n0))

    def critic(self, req, ledger=None):
        """Judges one child's cited evidence; feedback to the generator expanding that child; the controller
        receives no explicit critic verdict, though generated reasons may reflect feedback (repair.Tree.critic)."""
        return self.single("critic", req, steps.obj({"verdict": steps.enum(req["options"]), "reason": repair.REASON}),
                           ledger)

    def controller(self, req, ledger=None):
        """Ranks, stops or selects through the request's own schema; repair.controller_errors checks the rest."""
        reply, use = self.single("controller", req, req["schema"], ledger)
        return flat_choice(reply), use


def flat_choice(reply):
    """A rank reply's `choice` as the flat decision and select the tree checks, the reply as answered kept beside
    them. Nothing is inferred or completed: a reply without a choice of exactly an action and a
    select (the select task's) is passed on as answered."""
    c = reply.get("choice") if isinstance(reply, dict) else None
    if not (isinstance(c, dict) and set(c) == {"action", "select"}):
        return reply
    return {"decision": c["action"], "order": reply.get("order"), "select": c["select"], "reason": reply.get("reason"),
            "as_answered": reply}


def question_sends(engine):
    """The most requests one question of a RecordingCaller on `engine` may send, as llm.Caller.question_sends()
    counts them: DISPATCHES_PER_QUESTION, plus RATE_LIMIT_RECOVERY on OpenRouter (the hosted
    question's 429 recovery); the local engine unchanged."""
    return DISPATCHES_PER_QUESTION + (RATE_LIMIT_RECOVERY if engine == "openrouter" else 0)


def req_ceiling(ledger, sends=DISPATCHES_PER_QUESTION):
    """The Caller's call cap for a one-question role: the meter's ceiling, or one question's requests (`sends`,
    question_sends of the role's engine) without one."""
    return getattr(ledger, "ceiling", None) or sends
