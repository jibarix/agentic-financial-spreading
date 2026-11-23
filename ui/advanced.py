"""The Advanced page (read only): the default configuration as it is today, read from the job settings file and the code's constants. It
changes nothing. Each part is a table, {title, about, columns, rows}, which the page draws as given.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spreading import llm, named_totals, organizer, spread as code  # noqa: E402
from setup_status import NAMES, SETTINGS, flag, label, roles  # noqa: E402

def columns(context):
    return ["Step", "Role", "Model", "Runs on", context, "Answer limit (tokens)", "Reasoning",
            "Price ceiling in / out ($ per million tokens)"]


def role_ctx(args, default):
    """Each tree role's context: the tree's --num-ctx, or a role's own --role-num-ctx (e.g. meaning=16384)."""
    own = {k: int(v) for k, v in (x.split("=", 1) for i, x in enumerate(args[1:]) if args[i] == "--role-num-ctx")}
    return lambda role: own.get(role, default)


def per_million(p):
    return f"{p[0] * 1e6:g} / {p[1] * 1e6:g}"


def hosted_rows(profile, answer_cap, spread_ctx, tree_ctx, tree_cap):
    """One row per role of a hosted line-up, from its profile (spread.HOSTED_PROFILES): the spread's three seats, the
    roles borrowed from them (spread.BORROWED), and the search's embeddings."""
    p = code.HOSTED_PROFILES[profile]
    seat = {"analyst": p["analyst"], "checker": p["checker"], "haiku": p.get("haiku") or code.THIRD["haiku"]}

    def row(step, role, v, ctx, cap):
        effort, room = v.get("reasoning", "off"), v.get("allowance", 0)
        return [step, role, v["model"], "OpenRouter: " + ", ".join(v["providers"]), ctx, cap,
                effort if effort == "off" else
                f"{effort}; {room:,} tokens of headroom; total completion limit {cap + room:,}",
                per_million(v["price"])]
    tree = {"generator": "Generator", "controller": "Controller", "meaning": "Meaning question", "critic": "Critic"}
    return [row("Spread", "Organizer", seat[code.BORROWED["organizer"]], organizer.NUM_CTX, organizer.ANSWER_CAP),
            row("Spread", "Named totals", seat[code.BORROWED["name"]], named_totals.NUM_CTX, named_totals.ANSWER_CAP),
            row("Spread", "Analyst", seat["analyst"], spread_ctx, answer_cap),
            row("Spread", "Checker", seat["checker"], spread_ctx, answer_cap),
            row("Spread", "Tiebreaker", seat["haiku"], spread_ctx, answer_cap),
            *[row("Tree of Thought", name, seat[code.BORROWED[r]], tree_ctx(r), tree_cap) for r, name in tree.items()],
            row("Tree of Thought", "Review", seat["checker"], spread_ctx, answer_cap),
            ["Search", "Embeddings", llm.HOSTED_EMBED["model"], "OpenRouter: " + ", ".join(llm.HOSTED_EMBED["providers"]),
             "–", "–", "–", f"{llm.HOSTED_EMBED['price'] * 1e6:g} / –"]]


def local_rows(spread_args, tree_args, spread_ctx, tree_ctx, tree_cap, answer_cap):
    """One row per role of a local line-up: the models its arguments name (the code's defaults otherwise), on this
    computer, free; the context Ollama is given and the batch pin each model runs with."""
    sp, tr = roles(spread_args), roles(tree_args)
    analyst = sp.get("analyst", code.LOOP_MODELS["analyst"])
    checker = sp.get("checker", code.LOOP_MODELS["checker"])
    third = flag(spread_args, "--third-model") if "--third-opinion" in spread_args else None

    def row(step, role, model, ctx, cap):
        name = NAMES.get(model, model)
        pin = llm.NUM_BATCH.get(model)
        return [step, role, f"{name} ({model})", f"this computer (Ollama), batch {pin or 'auto'}", ctx,
                cap, "off", "free"]
    return [row("Spread", "Organizer", organizer.MODEL, organizer.NUM_CTX, organizer.ANSWER_CAP),
            row("Spread", "Named totals", named_totals.MODEL, named_totals.NUM_CTX, named_totals.ANSWER_CAP),
            row("Spread", "Analyst", analyst, spread_ctx, answer_cap),
            row("Spread", "Checker", checker, spread_ctx, answer_cap),
            *([row("Spread", "Tiebreaker", third, spread_ctx, answer_cap)] if third else []),
            row("Tree of Thought", "Generator", tr.get("generator", analyst), tree_ctx("generator"), tree_cap),
            row("Tree of Thought", "Controller", tr.get("controller", checker), tree_ctx("controller"), tree_cap),
            row("Tree of Thought", "Meaning question", tr.get("meaning", analyst), tree_ctx("meaning"), tree_cap),
            row("Tree of Thought", "Critic", tr.get("critic", checker), tree_ctx("critic"), tree_cap),
            row("Tree of Thought", "Review", flag(tree_args, "--review-model") or checker, spread_ctx, answer_cap),
            ["Search", "Embeddings", f"{NAMES.get(llm.EMBED_MODEL, llm.EMBED_MODEL)} ({llm.EMBED_MODEL})",
             "this computer (Ollama)", "–", "–", "–", "free"]]


# Each setting explained, from the code that applies it and the project's settings record. Kept apart: the value
# configured; a limit or check the code enforces; a provider or hardware constraint; a project recommendation backed
# by recorded evidence, else NONE. Proposals that are not implemented are not shown as behaviour.
NONE = "no recommended range recorded"
NOT_HERE = "not recorded on this page: an open question"
NO_CHECK = "none on the value: the job uses the number in the settings file as given"
KEYS = ("for", "behaves", "why", "moves", "enforced", "constraint", "recommended")


CHECKED = "checked 2026-11-17"
# OpenRouter's endpoint listing per (model, endpoint): context, max_prompt_tokens, max_completion_tokens, source. A None
# max_prompt_tokens means no separate prompt maximum is published in that field, not unlimited input. An endpoint not
# listed here is not verified (the DeepSeek deepinfra/fp8 and novita/fp8 fields could not be confirmed).
ENDPOINTS = {
    ("anthropic/claude-haiku-5.5", "anthropic"):
        (1_000_000, None, 128_000, "https://openrouter.ai/api/v1/models/anthropic/claude-haiku-5.5/endpoints"),
    ("openai/gpt-6-luna", "openai"):         # the standard tier's listing
        (1_050_000, 922_000, 128_000, "https://openrouter.ai/api/v1/models/openai/gpt-6-luna/endpoints"),
}
MAX_PRICE = ("provider.max_price is a unit-price ceiling in US dollars per million tokens: providers above it are "
             "excluded, and a request fails if no eligible provider remains; the project's ledger supplies the "
             "total-spend limit (OpenRouter, https://openrouter.ai/blog/tutorials/"
             f"how-to-get-the-lowest-cost-llm-inference-on-openrouter/, {CHECKED})")
# the publishers' native windows, per installed model name; the installed quantized build is not verified
WINDOWS = {"qwen3.5:4b-q4_K_M": ("Qwen3.5-4B", 262_144, "https://huggingface.co/Qwen/Qwen3.5-4B"),
           "minicpm5:2b-q4_K_M": ("MiniCPM5-2B", 131_072, "https://huggingface.co/openbmb/MiniCPM5-2B")}


def endpoints(prof, part):
    """For each seat's pinned endpoint, the verified listing's `part` ("context", "prompt", "completion"), or not
    verified."""
    seats = [prof.get(k) or code.THIRD["haiku"] for k in ("analyst", "checker", "haiku")]
    pairs = list(dict.fromkeys((v["model"], e) for v in seats for e in v["providers"]))
    said = []
    for model, e in pairs:
        got = ENDPOINTS.get((model, e))
        if got is None:
            said.append(f"{model} on {e}: not verified ({NOT_HERE})")
            continue
        ctx, prompt, done, url = got
        tier = " (standard tier)" if (model, e) == ("openai/gpt-6-luna", "openai") else ""
        said.append(f"{model} on {e}{tier}: " + {
            "context": f"context {ctx:,}, max_prompt_tokens " + (f"{prompt:,}" if prompt is not None else
                                                                  "null (no separate prompt maximum "
                                                                  "published in that field, not unlimited input)"),
            "completion": f"max_completion_tokens {done:,}"}[part] + f" ({url}, {CHECKED})")
    return "; ".join(said)


def windows(models):
    said = [f"{NAMES.get(m, m)}: " + (f"{WINDOWS[m][1]:,} native tokens, the publisher's figure for {WINDOWS[m][0]} "
                                      f"({WINDOWS[m][2]}, {CHECKED}), the installed build not verified"
                                      if m in WINDOWS else "no published window checked")
            for m in dict.fromkeys(models)]
    return "; ".join(said)


def setting(name, value, unit, scope, source, **said):
    assert set(said) == set(KEYS), set(said) ^ set(KEYS)
    return {"setting": name, "value": value, "unit": unit, "scope": scope, "source": source, **said}


def explained(cfg, base_args, tree, answer_cap, hosted, local_models=()):
    """The settings explained, one entry per setting, with the values read from the settings and the code."""
    stmts, co, third = cfg["statements"], cfg.get("company_limits") or {}, code.THIRD_CAPS["haiku"]
    per = lambda key: " / ".join(f"{s['statement']} {s[key]:,}" for s in stmts)   # noqa: E731
    tree_args = tree.get("args", [])
    tree_ctx, tree_cap = int(flag(tree_args, "--num-ctx") or 8192), int(flag(tree_args, "--answer-cap") or answer_cap)
    from spreading import loop, repair
    margin = loop.MARGIN
    # the local spread's context, only while a line-up uses it: a local one, or an online one without its own context
    uses_local_ctx = bool(local_models) or (hosted is not None and not hosted["num_ctx"])
    out = [] if not uses_local_ctx else [
        setting("Context, local spread (num_ctx)", per("num_ctx"), "tokens", "per request; one value per statement",
                "settings: statements[].num_ctx",
                **{"for": "The window Ollama is given for each analyst, checker and tiebreaker request on that "
                          "statement.",
                   "behaves": f"Sent to Ollama with every request (num_ctx), with truncation and context shift off. "
                              f"Before each step the loop estimates the prompt; above {margin:g} of this value "
                              "(loop.MARGIN, labelled provisional in the code) the line ends context_exceeded and "
                              "nothing is sent. A prompt Ollama counts as over the window is refused there and the "
                              "line ends the same way; no prompt is cut.",
                   "why": "It decides how much of a filing's evidence one request can carry.",
                   "moves": "Higher: longer prompts are sent instead of ending context_exceeded, and each local "
                            "attempt reserves this value plus the answer limit from the statement's token limit, so "
                            "fewer attempts fit in it. Lower: the reverse.",
                   "enforced": f"the {margin:g} margin before sending; " + NO_CHECK,
                   "constraint": "hardware: the model must fit this computer's GPU memory at this window; the job "
                                 "does not check it, and the fit is not verified. The models' published windows: "
                                 + windows(local_models),
                   "recommended": NONE + ". The balance sheet keeps 16,384 because a larger window's fit "
                                  "in this computer's GPU memory is not checked by the job, the "
                                  "settings or the code."}),
    ]
    if hosted:
        ctx, prof = hosted["num_ctx"], code.HOSTED_PROFILES[hosted["profile"]]
        seats = (("analyst", "analyst"), ("checker", "checker"), ("tiebreaker", "haiku"))
        rooms = "; ".join(f"{who} {(prof.get(k) or code.THIRD['haiku']).get('allowance', 0):,}" for who, k in seats)
        out += [
            setting("Context guard, online line-up", f"{ctx:,}", "tokens",
                    "per request of the spread's loop (analyst, checker, tiebreaker) and of the tree's C8 review",
                    "settings: lineups[].num_ctx",
                    **{"for": "The job's own size check for the online line-up's spread requests and for the C8 review, "
                              "which re-solves each changed line with the source spread's checker and context. It is "
                              "not sent to OpenRouter.",
                       "behaves": f"The same {margin:g} margin: above {int(ctx * margin):,} estimated prompt tokens "
                                  "the line ends context_exceeded before a request is sent. After a reply, when "
                                  "OpenRouter's counted prompt tokens are over that budget, the answer is refused the "
                                  "same way, and its dollars are spent: such a line ends "
                                  "context_exceeded like one stopped before sending, and its "
                                  "reply is not used."
                                  " The tree's other roles (generator, controller, meaning question, "
                                  "critic) do not use this value: they take the line-up's tree --num-ctx, "
                                  f"{hosted['tree_ctx']:,} online, minus the answer limit (its role table above); the "
                                  "organizer and named totals keep their own (below).",
                       "why": "Controls which prompt sizes the project permits; provider limits apply separately.",
                       "moves": "Higher: larger prompts are sent, each costing more dollars, and the tree's review "
                                "reserves more tokens. Lower: more lines end "
                                "context_exceeded.",
                       "enforced": f"the {margin:g} margin before sending; " + NO_CHECK,
                       "constraint": "provider: each pinned endpoint's context length and largest prompt. The job "
                                     "does not check them. OpenRouter's listings: " + endpoints(prof, "context"),
                       "recommended": NONE + ". Kept at 32,768 until the review "
                                      "and tree reservations are made to fit, because at 65,536 the review's "
                                      "reservations for two lines would exceed the tree's token limit. "
                                      "Provider limits apply separately."}),
            setting("Reasoning allowance, online roles", rooms, "tokens",
                    "per request, per role (0 where reasoning is off)", f"code: spread.HOSTED_PROFILES[{hosted['profile']!r}]",
                    **{"for": "Room for the model's reasoning, added to the answer limit in the completion limit "
                              "sent to OpenRouter.",
                       "behaves": "max_tokens sent = the answer limit + this allowance (the role table above shows "
                                  "each role's total).",
                       "why": "Adds room for reasoning to the requested completion limit.",
                       "moves": "Higher: Increases the requested completion limit and dollar reservation; its effect "
                                "on cut answers has not been measured. Lower: the reverse.",
                       "enforced": NO_CHECK,
                       "constraint": "provider: each endpoint's largest completion. OpenRouter's listings: "
                                     + endpoints(prof, "completion"),
                       "recommended": NONE + ("; the code labels Luna's 4,096 provisional headroom, not "
                                              "measured (spreading/spread.py)." if "luna" in hosted["profile"]
                                              else ".")}),
            setting("Price ceiling, online roles", "per role (table above)", "dollars per million tokens, in / out",
                    "per request", "code: spread.HOSTED_PROFILES, llm.HOSTED_EMBED",
                    **{"for": "The most the project will pay per token for each role's model.",
                       "behaves": "Sent with every online chat request as provider.max_price. The job also "
                                  "reserves each request's dollars at this price before sending it.",
                       "why": "It is the most the project allows per token, and it sizes each request's dollar "
                              "reservation against the job's dollars.",
                       "moves": "Higher: a larger reservation per request, so the job's dollars cover fewer requests "
                                "in flight, and the last request is refused sooner, leaving dollars unused. What "
                                "OpenRouter does with it: see the constraint.",
                       "enforced": NO_CHECK,
                       "constraint": "provider: " + MAX_PRICE + ". What each endpoint charges "
                                     "today is not shown on this page (for the DeepSeek endpoints: " + NOT_HERE
                                     + "); the ceilings are frozen and not changed to today's prices.",
                       "recommended": NONE}),
            setting("Dollars per job, online line-up", f"0 to {hosted['dollars_most']:g} (set on the Run page)",
                    "US dollars", "per job (one company)", "settings: lineups[].dollars_most; the Run page",
                    **{"for": "The most one online job may spend.",
                       "behaves": "Before each request the job refuses to send when the dollars spent plus that "
                                  "request's reservation would pass it. The dollars are the company's limit, so a "
                                  "job that reaches them ends Capped with its reason.",
                       "why": "It is the job's spending limit.",
                       "moves": "Higher: a job can finish more lines before it stops, at more cost. Lower: it stops "
                                "sooner.",
                       "enforced": f"0 to {hosted['dollars_most']:g}, checked by the Run page and by the job",
                       "constraint": "none recorded",
                       "recommended": NONE}),
        ]
    out += [
        setting("Answer limit", f"{answer_cap:,}", "tokens", "per request of the analyst, checker, tiebreaker and "
                "the tree's review", "settings: spread_args --answer-cap",
                **{"for": "The longest answer one request may return.",
                   "behaves": "Sent to Ollama as num_predict; online, max_tokens is this plus the role's reasoning "
                              "allowance. An answer cut at the limit (Ollama's done_reason or "
                              "OpenRouter's finish_reason \"length\") is refused even when it parses, and that "
                              "request is not retried. The refusal is one of the two a loop allows: the role may "
                              "still answer at a later step, a second refusal ends the line invalid, and a refusal on "
                              "the last step ends it cap_unfinished.",
                   "why": "Too short cuts answers; each local attempt also reserves it from the token limits.",
                   "moves": "Higher: a larger requested completion limit and a larger reservation per local "
                            "attempt; its effect on cut answers has not been measured. Lower: the reverse.",
                   "enforced": NO_CHECK,
                   "constraint": "none recorded",
                   "recommended": NONE}),
        setting("Steps per placement pass", flag(base_args, "--cap") or loop.STEP_CAP, "steps",
                "per line, per role's pass (analyst, checker, second look, tiebreaker)", "settings: spread_args --cap",
                **{"for": "How many steps (search, read, ask, answer) one role's loop may take on one line.",
                   "behaves": "The last step offers only a final answer. A loop with no answer by then ends "
                              "cap_unfinished; a tiebreaker loop that ends so is counted as no vote. A retry can add "
                              "requests.",
                   "why": "It bounds the work and requests one line can take.",
                   "moves": "Higher: more room to look before answering, more requests. Lower: more lines end "
                            "cap_unfinished.",
                   "enforced": NO_CHECK,
                   "constraint": "none recorded",
                   "recommended": NONE + f"; the code's default {loop.STEP_CAP} is a hard limit per loop "
                                         "(spreading/loop.py)."}),
        setting("Statement limits: model calls, tokens, seconds",
                f"calls {per('max_calls')}; tokens {per('max_tokens')}; seconds {per('wall_s')}",
                "requests; tokens; seconds", "per statement of one company", "settings: statements[]",
                **{"for": "The ceilings one statement's spread may use. They are fixed ceilings, the same for "
                          "every filing, not scaled to a filing's size.",
                   "behaves": "Each is cut to what the company has left. Every attempt is reserved before it is "
                              "sent; a limit once reached stops the statement's ledger, which refuses every later "
                              "attempt, so the lines not yet answered end not run and the statement is recorded "
                              "capped. Tokens: each local attempt reserves its context plus the answer limit; online "
                              "attempts reserve none, so this limit does not bound online requests at send time (the "
                              "dollars do). Seconds: a deadline. A statement that reaches its own limit lets "
                              "the job go on; one cut by the company's limits ends the job Capped.",
                   "why": "They bound one statement's time and cost; a larger filing may reach one.",
                   "moves": "Higher: a larger filing can finish. Lower: more statements end capped.",
                   "enforced": NO_CHECK,
                   "constraint": "none recorded",
                   "recommended": NONE + ". A larger filing may need higher values; the job records each "
                                         "statement that reaches one."}),
        setting("Company limits: model calls, tokens, seconds",
                f"calls {co.get('requests', 0):,}; tokens {co.get('tokens', 0):,}; seconds {co.get('wall_s', 0):,}",
                "requests; tokens; seconds", "per job (one company), all its statements and the tree",
                "settings: company_limits",
                **{"for": "The ceilings of one company's whole job.",
                   "behaves": "Checked before every step; a step gets at most what is left. Online requests' "
                              "reported tokens are added to the company's tokens after each step, so online use "
                              "lowers what later steps get. When a limit is reached the job ends Capped with its "
                              "reason.",
                   "why": "They bound a whole job.",
                   "moves": "Higher: a large company can finish. Lower: jobs end Capped sooner.",
                   "enforced": NO_CHECK, "constraint": "none recorded", "recommended": NONE}),
        setting("Tiebreaker limits",
                f"company-wide {co.get('haiku_requests')} calls, ${co.get('haiku_dollars', 0):g}; "
                f"per spread run {third['calls']} calls, ${third['spend_usd']:g}",
                "requests; US dollars", "per company, and per spread run (one statement)",
                "settings: company_limits.haiku_requests, haiku_dollars (company-wide; the dollars cut to the job's "
                "dollars); code: spread.THIRD_CAPS (per spread run)",
                **{"for": "The tiebreaker's share, inside the job's own limits.",
                   "behaves": "Two limits apply. The company's ledger carries the company-wide pair; each spread run's "
                              "own ledger carries the per-run pair when the tiebreaker runs on OpenRouter. At any of "
                              "them the tiebreaker sends no more requests; the rest of the job goes on.",
                   "why": "It keeps an experimental role small.",
                   "moves": "Higher: more disputed lines get a third answer. Lower: fewer.",
                   "enforced": NO_CHECK, "constraint": "none recorded",
                   "recommended": NONE + f"; the code says ${third['spend_usd']:g} covers {third['calls']} calls "
                                         "of about 13,000 tokens at the tiebreaker's price ceiling, a ceiling, not an "
                                         "estimate (spread.THIRD_CAPS, whose comment calls them a per-company sub-cap; "
                                         "each spread run's own "
                                         "ledger uses the same figures)."}),
        setting("Tree of Thought: section attempts", tree.get("max_sections"), "sections", "per job",
                "settings: tree.max_sections",
                **{"for": "How many failing sections the tree may try to repair.",
                   "behaves": "Each failing section is tried at most once; sections past this number end \"not run, "
                              "cap reached\".",
                   "why": "It bounds the repair work of one job.",
                   "moves": "Higher: more sections are tried, at more requests. Lower: fewer.",
                   "enforced": NO_CHECK, "constraint": "none recorded", "recommended": NONE}),
        setting("Tree of Thought: calls, tokens, seconds per attempt",
                f"calls {tree.get('max_calls', 0):,}; tokens {tree.get('max_tokens', 0):,}; "
                f"seconds {tree.get('wall_s', 0):,}", "requests; tokens; seconds", "per section attempt",
                "settings: tree",
                **{"for": "The ceilings of one section's repair attempt, within what the company has left.",
                   "behaves": "The tree sizes its token ceiling at "
                              f"{repair.TOKENS_PER_REQUEST:,} tokens per call (labelled provisional, not measured, in "
                              "the code) and holds back room for its review; when that room does not fit, the tree "
                              "does not expand. Each local attempt then reserves its context plus the answer limit; "
                              "online attempts reserve none. At a limit the attempt ends cap_reached, and a repair is committed only when the "
                              "attempt ends repaired.",
                   "why": "They bound one repair.",
                   "moves": "Higher: a repair can search further. Lower: more attempts end cap_reached.",
                   "enforced": NO_CHECK, "constraint": "none recorded", "recommended": NONE}),
        setting("Tree of Thought: context and answer limit", f"context {tree_ctx:,}, answer {tree_cap:,} (a line-up may "
                "give a role its own context: its role table above)", "tokens", "per request of a tree role",
                "settings: tree.args --num-ctx, --answer-cap, --role-num-ctx",
                **{"for": "The window and the answer length of the tree's generator, controller, meaning question "
                          "and critic.",
                   "behaves": "A role's prompt may use its context minus the answer limit; the online line-up sets its "
                              "own (its role table above).",
                   "why": "Too small a window leaves out evidence a repair needs.",
                   "moves": "Higher: more evidence per request. Lower: the reverse.",
                   "enforced": "a role's own context must be above the answer limit (spreading/repair_run.py)",
                   "constraint": "hardware: GPU memory, as for the spread", "recommended": NONE}),
        setting("Organizer and named totals: context and answer limit",
                f"organizer {organizer.NUM_CTX:,} / {organizer.ANSWER_CAP:,}; named totals "
                f"{named_totals.NUM_CTX:,} / {named_totals.ANSWER_CAP:,}", "tokens (context / answer)",
                "per request of each role", "code: organizer.py, named_totals.py",
                **{"for": "The window and answer length of the two roles that prepare each statement.",
                   "behaves": "The prompt may use the context minus the answer limit (organizer: "
                              f"{organizer.NUM_CTX - organizer.ANSWER_CAP:,} tokens). These are code constants, the "
                              "same for every line-up.",
                   "why": "They decide how large a statement these roles can sort in one request.",
                   "moves": "Changed only in the code, not in the settings.",
                   "enforced": NO_CHECK.replace("the settings file", "the code"),
                   "constraint": "hardware: GPU memory, for a local line-up", "recommended": NONE}),
        setting("Batch pin (num_batch)", ", ".join(f"{NAMES.get(m, m)} {n:,}" for m, n in llm.NUM_BATCH.items()),
                "tokens per batch", "per local model, every chat request", "code: llm.NUM_BATCH",
                **{"for": "How many prompt tokens Ollama processes at once for each model.",
                   "behaves": "Sent with every local chat request; a model not listed is sent without one and "
                              "recorded \"batch not pinned\".",
                   "why": "It is part of what a local run is tested with; a change can change memory use and "
                          "speed.",
                   "moves": "Changed only in the code.",
                   "enforced": NO_CHECK.replace("the settings file", "the code"),
                   "constraint": "hardware: GPU memory",
                   "recommended": NONE + "; the code labels Qwen's 1,024 provisional."}),
    ]
    if not local_models:        # Ollama's batch pin is not sent for an online model (show only what is used)
        out = [x for x in out if not x["setting"].startswith("Batch pin")]
    return out


def hosted_profile(spread_args):
    """The profile an OpenRouter spread runs with: its --hosted-profile, else the CLI's default (spread.py's
    --hosted-profile default, "hosted"); None for a local spread."""
    if flag(spread_args, "--engine") != "openrouter":
        return None
    return flag(spread_args, "--hosted-profile") or "hosted"


# spread.py's --engine choices (kept the same as the spread's own command line)
ENGINES = ("ollama", "openrouter")


def refused(spread_args):
    """Why spread.py's command line would refuse a line-up's spread arguments, or None."""
    engine = flag(spread_args, "--engine") or "ollama"
    if engine not in ENGINES:
        return f"an --engine the spread does not accept ({engine}; it takes ollama or openrouter)"
    if engine == "openrouter" and "--model" in spread_args:
        return "--model with --engine openrouter, which the spread refuses (--model is a local model)"
    profile = hosted_profile(spread_args)
    if profile is not None and profile not in code.HOSTED_PROFILES:
        return f"a --hosted-profile the code does not know ({profile})"
    return None


def advanced(settings=SETTINGS, skip=()):
    cfg = json.loads(Path(settings).read_text(encoding="utf8"))
    base_args, tree = cfg.get("spread_args", []), cfg.get("tree") or {}
    answer_cap = int(flag(base_args, "--answer-cap") or llm.NUM_PREDICT)
    stmts = cfg["statements"]
    local_ctx = " / ".join(f"{s['statement']} {s['num_ctx']:,}" for s in stmts)
    parts = []
    for name, lu in sorted(cfg["lineups"].items()):
        if name in skip or lu.get("pending"):
            continue
        spread_args, tree_args = [*base_args, *lu.get("spread_args", [])], [*tree.get("args", []), *lu.get("tree_args", [])]
        tree_ctx = role_ctx(tree_args, int(flag(tree_args, "--num-ctx") or 8192))
        tree_cap = int(flag(tree_args, "--answer-cap") or answer_cap)
        profile, why = hosted_profile(spread_args), refused(spread_args)
        if why:
            parts.append({"title": label(name), "columns": ["Problem"], "rows": [[why]],
                          "about": f"This line-up has {why}; the spread would refuse it. Its roles are not shown, "
                                   "and it is neither local nor online here."})
            continue
        if profile is not None:
            rows = hosted_rows(profile, answer_cap, lu.get("num_ctx") or local_ctx, tree_ctx, tree_cap)
            ceiling = (f"Each job stops at the dollars set on the Run page, at most ${lu.get('dollars_most', 0):g}. "
                       "The context is the job's own guard, checked before a request is sent, not the provider's window; "
                       "a provider may charge less than the price ceiling.")
            context = "Context guard (tokens)"
        else:
            rows = local_rows(spread_args, tree_args, local_ctx, tree_ctx, tree_cap, answer_cap)
            ceiling = "Runs on this computer; nothing is paid. The context is the window Ollama is given."
            context = "Context (tokens)"
        if "--statements-only" in spread_args:      # no note search: no Search row
            rows = [x for x in rows if x[0] != "Search"]
        parts.append({"title": label(name), "about": "Each role's model and its limits. " + ceiling,
                      "columns": columns(context), "rows": rows})
    any_local = any(p.get("columns") and p["columns"][4:5] == ["Context (tokens)"] for p in parts)
    parts.append({"title": "Limits per statement",
                  "about": "Fixed ceilings per statement of one company, the same for every filing and not "
                           "scaled; a filing that "
                           "reaches one ends there with its reason." + (" The context is the local models'." if any_local else ""),
                  "columns": ["Statement", "Model calls", "Tokens", "Seconds", *(["Context (tokens, local)"] if any_local else [])],
                  "rows": [[s["statement"], s["max_calls"], s["max_tokens"], s["wall_s"], *([s["num_ctx"]] if any_local else [])]
                           for s in stmts]})
    co = cfg.get("company_limits") or {}
    third = code.THIRD_CAPS["haiku"]
    parts.append({"title": "Other limits",
                  "about": "The Tree of Thought (its calls, tokens and seconds per section attempt, within what is "
                           "left of the company's limits), one company in all, and the tiebreaker.",
                  "columns": ["Limit", "Value"],
                  "rows": [["Tree of Thought: section attempts per job", tree.get("max_sections")],
                           ["Tree of Thought: model calls per attempt", tree.get("max_calls")],
                           ["Tree of Thought: tokens per attempt", tree.get("max_tokens")],
                           ["Tree of Thought: seconds per attempt", tree.get("wall_s")],
                           ["Company: model calls", co.get("requests")], ["Company: tokens", co.get("tokens")],
                           ["Company: seconds", co.get("wall_s")],
                           ["Tiebreaker: calls per company", co.get("haiku_requests")],
                           ["Tiebreaker: dollars per company", f"${co.get('haiku_dollars', 0):g}"],
                           ["Tiebreaker: calls per spread run (online tiebreaker)", third["calls"]],
                           ["Tiebreaker: dollars per spread run (online tiebreaker)", f"${third['spend_usd']:g}"],
                           ["Steps per placement pass (analyst, checker, second look and tiebreaker each; a "
                            "retry can add requests)", flag(base_args, "--cap")],
                           ["Answer limit (tokens)", answer_cap]]})
    online = [(n, lu) for n, lu in cfg["lineups"].items() if n not in skip and not lu.get("pending")
              and flag([*base_args, *lu.get("spread_args", [])], "--engine") == "openrouter"]
    online = [(n, lu) for n, lu in online if refused([*base_args, *lu.get("spread_args", [])]) is None]
    hosted = None
    if online:
        lu = online[0][1]
        hosted = {"num_ctx": lu.get("num_ctx"), "dollars_most": lu.get("dollars_most", 0),
                  "profile": hosted_profile([*base_args, *lu.get("spread_args", [])]),
                  "tree_ctx": int(flag([*tree.get("args", []), *lu.get("tree_args", [])], "--num-ctx") or 8192)}
    local = [lu for n, lu in sorted(cfg["lineups"].items()) if n not in skip and not lu.get("pending")
             and hosted_profile([*base_args, *lu.get("spread_args", [])]) is None
             and refused([*base_args, *lu.get("spread_args", [])]) is None]
    models = []
    for lu in local:
        args = [*base_args, *lu.get("spread_args", [])]
        sp = roles(args)
        models += [sp.get("analyst", code.LOOP_MODELS["analyst"]), sp.get("checker", code.LOOP_MODELS["checker"])]
        models += [flag(args, "--third-model")] if "--third-opinion" in args and flag(args, "--third-model") else []
    return {"settings": Path(settings).name, "parts": parts,
            "explained": explained(cfg, base_args, tree, answer_cap, hosted, models),
            "explained_about": "Each setting: what it is for, how the pipeline behaves with it, why it matters and "
                               "what changes when it moves, kept apart from the value, any limit the code enforces, "
                               "any provider or hardware constraint, and any recommendation with its reason. Proposed "
                               "changes not yet made are not shown as current behaviour; where a default has a "
                               "stated reason for keeping today's value, that reason is given.",
            "any_local": any_local}
