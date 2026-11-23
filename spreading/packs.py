"""Build the prompt pieces the spread shares: the template's rows, the concept block and the rows block.

Code picks every piece of context; a model sees one row or one concept per call. Each row's
definition, belongs and not_belongs come from the template itself (read by template.py); the only template
rule stated by code is the catch-all rule.
"""
from pathlib import Path

from . import template

ROOT = Path(__file__).resolve().parent.parent
# The catch-all template rule, stated by code on every "Other" item row, never left to a drafted definition.
CATCH_ALL = "Template rule: this row takes every amount of the section that no other row of the section takes."
# Shown for a row whose definition the template does not have yet, so the model knows it judges by the name alone.
NO_DEFINITION = "No definition yet: the template gives only this row's name."


def prompt(name):
    return (ROOT / "prompts" / f"{name}.txt").read_text(encoding="utf8").strip()


ROW_KEYS = ("row_id", "statement", "section", "name", "line_type", "definition", "belongs", "not_belongs")


def load_rows():
    """The template's rows, each with its statement and section name, read from the template alone."""
    t = template.active()
    rows = []
    for r in t.data["rows"]:
        s = t.sections[r["section"]]
        x = dict(r, statement=s["statement"], section=s["name"])
        rows.append({k: x[k] for k in ROW_KEYS})
    return rows


def section_rows(rows, section):
    """The rows a filed amount of this section can take: items only, in template order; a memo section's memo rows
    (template.memo). The section's members come from the template, never from the shape of a row
    ID."""
    t = template.active()
    kind = "supplementary" if t.memo(section) else "item"
    members = {r["row_id"] for r in t.section_rows(section) if r["line_type"] == kind}
    return [r for r in rows if r["row_id"] in members]


def statement_rows(rows, statement):
    """section -> its item row IDs, for every section of the statement, in template order."""
    t, out = template.active(), {}
    for r in rows:
        s = t.section_of(r["row_id"])
        if r["line_type"] == "item" and t.statement_of(s) == statement:
            out.setdefault(s, []).append(r["row_id"])
    return out


def catch_all(rid):
    r = template.active().rows.get(rid)
    return bool(r and r["policy"] == "catch_all")


def item_text(x):
    return f"{x['item']} (goes to {x['goes_to']})" if x["goes_to"] else f"{x['item']} (outside the template)"


# ---- Placement pieces ---------------------------------------------------------------------------------------------

def named_catch_all(section):
    """The catch-all rule with its section named (prompt v20): a statement-level
    block shows several catch-alls, so "the section" no longer identifies one."""
    return (f"Template rule: this row takes every amount of section {section} "
            f"({template.active().sections[section]['name']}) that no other row of that section takes.")


def rows_block(section_ids, named=False, headed=False):
    """Each row's name and definition from the template, the catch-all rule on a catch-all row, and what belongs and
    does not belong to it when the template says. `named` (STATEMENT_PROMPTS): the catch-all rule names its section.
    `headed` (a queued line's every-row block): a line "SECTION <ID> (<name>)" before each run of one section's rows.
    Without either, the block is the v19 one."""
    t = template.active()
    rows, out, last = t.rows, ["ROWS YOU MAY CHOOSE"], None
    for rid in section_ids:
        r = rows[rid]
        if headed and t.section_of(rid) != last:
            last = t.section_of(rid)
            out.append(f"SECTION {last} ({t.sections[last]['name']})")
        out.append(f"  {rid}: {r['name']}. {r['definition'] or NO_DEFINITION}")
        if catch_all(rid):
            out.append(f"      {named_catch_all(t.section_of(rid)) if named else CATCH_ALL}")
        if r["belongs"]:
            out.append(f"      Belongs: {'; '.join(r['belongs'])}")
        if r["not_belongs"]:
            out.append(f"      Does not belong: {'; '.join(item_text(x) for x in r['not_belongs'])}")
    return out
