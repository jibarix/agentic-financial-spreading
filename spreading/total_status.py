"""A printed total's status, from its statement's own records: whether a comparison ran on exactly that concept, or
only the run's code found it equal to its printed parts, or neither ("total: compared in Checks"
only when a comparison ran on that statement, for exactly that concept; one rule for every company). The workbook's
Lines sheet (result_workbook.snapshot) and Replay (ui/views.replay) both call it, so they say the same of each total.
Standard library only: Replay imports it where the workbook's Excel support is not installed."""

TOTAL = "total: compared in Checks"
TOTAL_PARTS = "total: its printed parts add up to it (checked by code, not listed in Checks)"
TOTAL_NOT = "a total that no check on this statement compares"
TOTALS = (TOTAL, TOTAL_PARTS, TOTAL_NOT)
PARTS = " equals its printed parts"
ROWS = "rows add to "


def control_state(s):
    """The state a recorded check is counted under: tied, failed (differs or failed), or unavailable."""
    s = str(s or "")
    if s == "tied":
        return "tied"
    if s in ("differs", "failed") or s.startswith("failed"):
        return "failed"
    return "unavailable"


def concept(x):
    """A recorded concept name, not a recorded amount: a string with its prefix ("us-gaap:Assets")."""
    return x if isinstance(x, str) and ":" in x else None


def ran(x):
    """Whether a recorded check or control compared two amounts: it records a gap (zero included), or it is a
    printed-parts check, written only for a total spread.mark_controls found equal to its printed parts (its parts
    recorded, or moved to another section after that arithmetic ran). A failure with no gap ("no section total to
    compare", a "rows add to" check whose operand was unavailable) compared nothing."""
    if x.get("gap") is not None:
        return True
    return str(x.get("check") or "").endswith(PARTS) and x.get("filed") is not None and (
        x.get("parts") is not None or bool(x.get("parts_in_another_section")))


def compared_concepts(checks, controls, report_checks=(), report_defs=(), facts=None):
    """concept -> [passed, ...]: every comparison a statement's records ran on exactly that concept.
    - a check that ran (`ran`) and names the concept as its total or filed concept, as "<concept> equals its printed
      parts" or "rows add to <concept>", or by its filed fact (`facts`: fact id -> concept; "filed and reconstructed
      totals agree" records the filed amount, not the concept);
    - a control that ran over its filed concept;
    - a report check that ran, for each side the template names as one filed concept (`report_defs`, the template's
      report checks), read as the concept the run recorded in `read_as` when it read an alternative.
    A check on a longer concept that begins with this one is not this one's."""
    out = {}

    def add(k, ok):
        if concept(k):
            out.setdefault(k, []).append(bool(ok))
    for x in checks or ():
        if not isinstance(x, dict) or not ran(x):
            continue
        name = str(x.get("check") or "")
        ks = {concept(x.get("total")), concept(x.get("filed")), (facts or {}).get(x.get("filed_fact"))}
        if name.endswith(PARTS):
            ks.add(name[:-len(PARTS)])
        if name.startswith(ROWS):
            ks.add(name[len(ROWS):])
        for k in ks - {None}:
            add(k, x.get("ok"))
    for bucket, xs in (controls.items() if isinstance(controls, dict) else ()):
        for x in xs or ():
            st = control_state(x.get("state") or bucket) if isinstance(x, dict) else "unavailable"
            if st != "unavailable" and ran(x):
                add(x.get("filed"), st == "tied")
    defs = {d.get("id"): d for d in report_defs or () if isinstance(d, dict)}
    for x in report_checks or ():
        if not isinstance(x, dict) or not ran(x) or control_state(x.get("state")) == "unavailable":
            continue
        d = defs.get(x.get("id")) or {}
        for side in ("left", "right"):
            e = d.get(side)
            if isinstance(e, dict) and set(e) == {"concept"}:
                add((x.get("read_as") or {}).get(side) or e["concept"], control_state(x.get("state")) == "tied")
    return out


def parts_verified(lines, diagnostics):
    """The printed totals the run's code found equal to their printed parts (spread.mark_controls: decomposition
    "verified"), whether or not a check records it: a line record so marked, or a coverage line in state "verified
    subtotal" (a cash flow subtotal the run read for no section)."""
    occ = ((diagnostics or {}).get("coverage") or {}).get("occurrences") or ()
    return {x.get("concept") for x in lines or () if isinstance(x, dict) and x.get("control")
            and x.get("decomposition") == "verified"} | {o.get("concept") for o in occ
                                                          if isinstance(o, dict) and o.get("state") == "verified subtotal"}


def fact_concepts(lines, diagnostics):
    """fact id -> concept, from the statement's line records and its queued lines' records."""
    recs = list(lines or ()) + [(q or {}).get("line") or {} for q in (diagnostics or {}).get("queued") or ()]
    return {x["fact"]: x.get("concept") for x in recs if isinstance(x, dict) and isinstance(x.get("fact"), str)}


def statement_totals(r, report_defs=()):
    """(compared_concepts, parts_verified) of one statement's records `r`: checks, controls, report_checks, lines,
    diagnostics (the `r` of result_workbook.snapshot, or Replay's child)."""
    facts = fact_concepts(r.get("lines"), r.get("diagnostics"))
    return (compared_concepts(r.get("checks"), r.get("controls"), r.get("report_checks"), report_defs, facts),
            parts_verified(r.get("lines"), r.get("diagnostics")))


def total_said(k, compared, verified):
    """A total's status: compared in Checks (a comparison ran on exactly this concept, tied or not: the Checks sheet
    shows which), its printed parts add up (checked by code where Checks does not list it), or no check on this
    statement compares it."""
    return TOTAL if compared.get(k) else TOTAL_PARTS if k in verified else TOTAL_NOT
