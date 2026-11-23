"""C4, the analyst: one filed line through the capped tool loop. Whether the evidence supports the
row is the model's judgement.

Prompt v16, the six rules: the start message puts the printed label and the amount on one line, the
state is the pools state with `simple` and the section's own rows (`section_rows`), and code's check is
six_rules.checker (rules 5 and 6). With `row_seed`, the checker's template rows are shown in an order shuffled by a
recorded seed, so a row's place in the list does not steer the checker; the analyst keeps the template's order.
"""
import random

from . import six_rules, steps, template
from .loop import run_loop
from .packs import prompt
from .spread import STATEMENTS_PROMPT, line_pack
from .tools import BUILT

# the six-rule prompts that render the statement-level starts (the version gate):
# the ROWS SHOWN line, the catch-all with its section named, the OTHER TEMPLATE SECTIONS line with the rows_of
# sentence, and the queued line's every-row start; run_c4 refuses any other prompt
STATEMENT_PROMPTS = {"c4_place_v20", "c4_place_v21", "c4_place_v26", "c4_place_v27", "c4_place_v28",
                     "c4_place_v29"}
# the statements-only option: v27 without the notes, the note tools
# and breakdowns and other_periods (Tools statements_only); run_c4 refuses the prompt without those tools, and them
# without it (spread.STATEMENTS_PROMPT, the one name)
# the root-cause-fixes option (--root-cause-fixes): the prompt that describes what the tools show with --root-cause-fixes
# (Tools root_cause); run_c4 refuses the prompt without the tools, and the tools without the prompt
ROOT_CAUSE_PROMPT = "c4_place_v28"
ROWS_OF_SENTENCE = ("rows_of(section) shows that section's rows and makes them valid answers; use it when the line "
                    "or its pieces may belong to another section.")


def start_text(f, line, section, ids, other_sections):
    """The start message: the printed label and its amount on one line, without the AMOUNT label between them, so the
    natural quote "<label> <amount>" is found; line_pack's blocks, and the
    other sections on their own line with the rows_of sentence; a queued line
    (section None) has no other sections."""
    blocks = line_pack(f, line, section, ids)
    head = blocks["line"].rstrip("\n")
    label, amount, rest = head.split("\n", 2)
    head = f"{label} {amount.split(None, 1)[1]}\n{rest}"
    text = head + f"\nFACT ID     {line['fact']}\n\n" + blocks["rows"]
    if other_sections:
        text += "\nOTHER TEMPLATE SECTIONS: " + ", ".join(other_sections) + "\n" + ROWS_OF_SENTENCE
    return text


def start_state(line, ids, other_sections, tools=BUILT):
    """The line's first step state (v16): the four pools, `allocatable_ids` (empty until a tool returns an eligible
    fact), the line's fact excluded as the line itself, `simple`, and the section's own rows (`section_rows`, what a
    piece and the remainder of a split may go to)."""
    return {"rows": list(ids), "evidence_ids": [line["fact"]], "fact_ids": [line["fact"]], "line_fact": line["fact"],
            "sections": list(other_sections), "tools": list(tools), "observation_ids": [], "chains": False,
            "pools": True, "recall_ids": [], "discovered_passages": [], "read_passages": [],
            "citable_ids": [line["fact"]], "allocatable_ids": [], "excluded": {line["fact"]: steps.OWN},
            "canonical_of": {}, "duplicate_groups": {}, "line_concept": line.get("concept"), "simple": True,
            "section_rows": list(ids)}


def row_order(ids, role, concept, seed, grouped=False):
    """The template rows as this role is shown them: the template's order for the analyst, or with no seed; for the
    checker, shuffled by random.Random(f"{seed}:{role}:{concept}"), so the order is recorded and reproducible.
    `grouped` (a queued line's every-row start, C4b): the sections are shuffled, then the
    rows within each, so each section's rows stay together under their heading."""
    ids = list(ids)
    if seed is None or role == "analyst":
        return ids
    rng = random.Random(f"{seed}:{role}:{concept}")
    if not grouped:
        rng.shuffle(ids)
        return ids
    t, out = template.active(), []
    secs = list(dict.fromkeys(t.section_of(r) for r in ids))
    rng.shuffle(secs)
    for s in secs:
        rs = [r for r in ids if t.section_of(r) == s]
        rng.shuffle(rs)
        out += rs
    return out


def run_c4(caller, tax, f, line, section, ids, tools, printed, owned, *, prompt_name, role="analyst", index=None,
           rel=None, cap=None, trace=None, row_seed=None, extra_text=""):
    if prompt_name not in STATEMENT_PROMPTS:
        raise ValueError(f"prompt {prompt_name} does not render the statement-level start")
    if bool(getattr(tools, "root_cause", False)) != (prompt_name == ROOT_CAUSE_PROMPT):
        raise ValueError(f"prompt {prompt_name} with tools root_cause={bool(getattr(tools, 'root_cause', False))}: "
                         f"{ROOT_CAUSE_PROMPT} and the root-cause tools go together")
    if bool(getattr(tools, "statements_only", False)) != (prompt_name == STATEMENTS_PROMPT):
        raise ValueError(f"prompt {prompt_name} with tools statements_only="
                         f"{bool(getattr(tools, 'statements_only', False))}: {STATEMENTS_PROMPT} and the "
                         "statements-only tools go together")
    other = tools.other_sections() if tools else []
    return run_v16(caller, tax, f, line, section, ids, tools, printed, owned, other, role=role, index=index,
                   rel=rel, cap=cap, trace=trace, prompt_name=prompt_name, row_seed=row_seed, extra_text=extra_text)


def run_v16(caller, tax, f, line, section, ids, tools, printed, owned, other, *, role, index, rel, cap, trace,
            prompt_name, row_seed, extra_text=""):
    """One line through the v16 loop: the one answer form, rules 5 and 6 checked by six_rules.checker against
    `owned`, the caller's own working registry (the checker's is its own). No escalate.
    `extra_text` (the retry's candidate block): shown after the start
    message, never part of it, so a quote is never found in it and no fact ID in it becomes citable."""
    order = row_order(ids, role, line["concept"], row_seed, grouped=section is None)
    start = start_text(f, line, section, order, other)
    state = start_state(line, order, other, tools=tools.available() if tools else BUILT)
    if section is None:     # a queued line: pieces and the rest in the answer row's section (six_rules, rule 6)
        state["queued"] = True
    kw = {"cap": cap} if cap else {}
    eligible, root_cause = steps.Eligibility(tax, f, line).of, bool(getattr(tools, "root_cause", False))
    if root_cause:      # --root-cause-fixes: movable-fact marks, printed_page arguments, the marked refusal text
        index = index if index is not None else steps.fact_index(f)
        if rel is None:     # a root-cause-fixes path names its policy: no default can pick it
            raise ValueError("analyst: a root-cause-fixes path (--root-cause-fixes) is given its Relations; none was passed")
        eligible = six_rules.marked(eligible, f, line, index, rel)
        state["faces"] = True
    rec = run_loop(caller, role, prompt(prompt_name), start, state, tools,
                   six_rules.checker(tax, f, line, printed, owned, start, index, rel, **(
                       {"root_cause": True} if root_cause else {})), tag=f"C4:{line['concept']}",
                   trace=trace, eligible=eligible, extra_text=extra_text, **kw)
    rec.update(line=line["concept"], fact=line["fact"], amount=str(line["amount"]), unit=line["unit"],
               prompt=prompt_name, start=start, form="turns", row_order=order, row_seed=row_seed,
               quote_normalisation=six_rules.POLICY, **({"extra_text": extra_text} if extra_text else {}))
    return rec
