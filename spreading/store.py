"""Results store: finished spread run folders imported into one local SQLite file, by template version, company,
statement and year.

Code only: it reads what a run recorded and never decides a placement; no model writes here. Run folders
stay untouched as the full record. Each run imports in one transaction: the whole validated import commits, or nothing.
Re-importing a run whose files are unchanged adds nothing; a run whose files changed since its import is rejected.

    python -m spreading.store template add NAME PATH            a new template, version 1 = the file at PATH
    python -m spreading.store template version TEMPLATE_ID PATH  the next version of a template
    python -m spreading.store import RUN_FOLDER... [--label L]   import finished spread runs (dated load record)
    python -m spreading.store import-job JOB_FOLDER [--label L]  import a full-pipeline job's final statement versions
    python -m spreading.store select RUN_ID --by NAME            make a spread current (a person's choice)

--db PATH (before the command) uses another database file; the default is data/results/spreads.sqlite.
"""
import argparse
import contextlib
import datetime
import hashlib
import json
import re
import sqlite3
import sys
import time
import uuid
from decimal import MAX_PREC, Decimal, Inexact, localcontext
from pathlib import Path

from spreading import open_claims as oc
from spreading import runrec
from spreading.filing import Filing

ROOT = runrec.ROOT
DB = ROOT / "data/results/spreads.sqlite"
TEMPLATE_FORMATS = ("template_format_v2", "template_format_v3", "template_format_v4",  # v3: title fallback; v4: v11
                    "template_format_v5",                                          # v5: v12 (not_spread)
                    "template_format_v6",                                          # v6: v14 (report check alternatives)
                    "template_format_v7")                                          # v7: v15 (statement checks)
REQUIRED = ("run.json", "outcome.json", "rows.json", "lines.json", "placements.json", "claims.json", "c4.json",
            "checks.json", "computed_totals.json", "controls.json", "exceptions.json", "exceptions_summary.json",
            "report_checks.json", "diagnostics.json", "unallocated.json")
OPTIONAL = ("search.json", "calls.jsonl", "cumulative.json", "analyst_edits.json")

SCHEMA = """
CREATE TABLE IF NOT EXISTS templates (id TEXT PRIMARY KEY, name TEXT NOT NULL, created TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS template_versions (id TEXT PRIMARY KEY, template_id TEXT NOT NULL REFERENCES templates,
    version INTEGER NOT NULL, path TEXT NOT NULL, sha256 TEXT NOT NULL, created TEXT NOT NULL,
    UNIQUE (template_id, version));
CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, template_version_id TEXT REFERENCES template_versions,
    request TEXT, budget TEXT, started TEXT, ended TEXT, outcome TEXT);
CREATE TABLE IF NOT EXISTS companies (cik TEXT PRIMARY KEY, name TEXT, name_source TEXT, sic TEXT, sic_source TEXT);
CREATE TABLE IF NOT EXISTS filings (accession TEXT PRIMARY KEY, cik TEXT NOT NULL REFERENCES companies,
    form TEXT NOT NULL, period_end TEXT, fiscal_year TEXT, fiscal_period TEXT, taxonomy_year TEXT, folder TEXT NOT NULL,
    inputs_status TEXT);
CREATE TABLE IF NOT EXISTS spreads (run_id TEXT PRIMARY KEY, accession TEXT NOT NULL REFERENCES filings,
    statement TEXT NOT NULL, sections TEXT NOT NULL, not_applicable TEXT, period_kind TEXT NOT NULL,
    period_start TEXT, period_end TEXT NOT NULL, run_folder TEXT NOT NULL, run_date TEXT NOT NULL, models TEXT,
    template_version_id TEXT REFERENCES template_versions, template_note TEXT, template_path TEXT,
    template_sha256 TEXT, job_id TEXT REFERENCES jobs, job_note TEXT, outcome TEXT NOT NULL, label TEXT,
    files TEXT NOT NULL, load_id TEXT NOT NULL, imported TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS spread_rows (run_id TEXT NOT NULL REFERENCES spreads, seq INTEGER NOT NULL,
    row_code TEXT NOT NULL, row_name TEXT NOT NULL, section TEXT NOT NULL, placed TEXT, usable TEXT,
    availability TEXT NOT NULL, unit TEXT, PRIMARY KEY (run_id, row_code));
CREATE TABLE IF NOT EXISTS row_sources (run_id TEXT NOT NULL REFERENCES spreads, row_code TEXT NOT NULL,
    section TEXT, line TEXT NOT NULL, line_label TEXT, piece TEXT, fact TEXT, signed TEXT NOT NULL, facts TEXT,
    placed_by TEXT, unit TEXT, review_state TEXT NOT NULL, quote TEXT, quote_source TEXT);
CREATE TABLE IF NOT EXISTS totals (run_id TEXT NOT NULL REFERENCES spreads, row_code TEXT NOT NULL, value TEXT,
    unavailable TEXT, incomplete TEXT, label TEXT, unit TEXT, PRIMARY KEY (run_id, row_code));
CREATE TABLE IF NOT EXISTS checks (run_id TEXT NOT NULL REFERENCES spreads, kind TEXT NOT NULL, section TEXT,
    name TEXT, state TEXT NOT NULL, filed TEXT, computed TEXT, difference TEXT, tolerance TEXT, detail TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS exceptions (run_id TEXT NOT NULL REFERENCES spreads, step TEXT, class TEXT,
    section TEXT, concept TEXT, why TEXT, detail TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS current_selection (template_version_id TEXT NOT NULL REFERENCES template_versions,
    cik TEXT NOT NULL, statement TEXT NOT NULL, fiscal_year TEXT NOT NULL, run_id TEXT NOT NULL REFERENCES spreads,
    set_by TEXT NOT NULL, set_at TEXT NOT NULL, PRIMARY KEY (template_version_id, cik, statement, fiscal_year));
CREATE TABLE IF NOT EXISTS refusals (job_id TEXT NOT NULL REFERENCES jobs, accession TEXT NOT NULL,
    statement TEXT NOT NULL, refused_at TEXT NOT NULL, kind TEXT, reason TEXT NOT NULL,
    PRIMARY KEY (job_id, statement));
CREATE TABLE IF NOT EXISTS reviews (id INTEGER PRIMARY KEY AUTOINCREMENT,
    template_version_id TEXT NOT NULL REFERENCES template_versions, cik TEXT NOT NULL, fiscal_year TEXT NOT NULL,
    reviewer TEXT NOT NULL, export TEXT NOT NULL, export_sha256 TEXT NOT NULL, reviewed_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS open_claims (run_id TEXT NOT NULL REFERENCES spreads, seq INTEGER NOT NULL,
    kind TEXT NOT NULL, row_code TEXT, section TEXT, line TEXT NOT NULL, line_label TEXT, piece TEXT, fact TEXT,
    piece_amount TEXT, rival TEXT NOT NULL, rival_amount TEXT, basis TEXT, quote TEXT, passage TEXT, at_stake TEXT,
    unit TEXT, actions TEXT, detail TEXT NOT NULL, PRIMARY KEY (run_id, seq));
CREATE TABLE IF NOT EXISTS provisional_amounts (run_id TEXT NOT NULL REFERENCES spreads, what TEXT NOT NULL,
    row_code TEXT NOT NULL, line TEXT, fact TEXT, open_claims INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS open_claim_counts (run_id TEXT PRIMARY KEY REFERENCES spreads,
    open_claims INTEGER NOT NULL, newly_relaxed INTEGER NOT NULL, already_unknown INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS claim_status (run_id TEXT PRIMARY KEY REFERENCES spreads, status TEXT NOT NULL,
    reason TEXT, detail TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS recorded_claims (run_id TEXT NOT NULL REFERENCES spreads, seq INTEGER NOT NULL,
    claim_id TEXT NOT NULL, kind TEXT NOT NULL, origin TEXT NOT NULL, row_code TEXT, line TEXT NOT NULL, piece TEXT,
    fact TEXT NOT NULL, piece_amount TEXT, rival TEXT NOT NULL, basis TEXT, reading TEXT, PRIMARY KEY (run_id, seq));
"""
# open_claims, provisional_amounts and open_claim_counts hold an R-arm job's decision cards and marks. The use gate
# does not read its answer from them: claim_status and recorded_claims hold every unknown-overlap claim a spread
# records in either arm (claim_status()). A spread stored before those two tables has no row in them.
OPEN_KIND = oc.KIND
OPEN_RELAXED = oc.RELAXED
CLAIM_FREE, OPEN_CLAIMS, NOT_ASSESSABLE = "claim-free", "open claims", "not assessable"
# The origins of recorded claims that block use. A repair split's reviewed overlap claims (legacy policy, reviewed by
# C8) are counted here as open, a conservative reading; removing oc.REPAIR from this tuple is
# the one change that reverses it. Counts are kept by origin either way.
GATED_ORIGINS = (oc.PLACEMENT, oc.REPAIR)
# Acceptance: a spread with open claims is used only once
# accepted for use, bound to its content fingerprint and every recorded claim. It authorizes use and settles nothing.
# Not used by any gate: no gate reads an
# acceptance and the UI records none. The records, their triggers, accepted() and revoke() stay, so
# old acceptances remain readable and revocable; accept() stays and refuses (see DELEGATION).
ACCEPTANCE_SCHEMA = "acceptance/v1"
FINGERPRINT_SCHEMA = "spread-content/v1"
# Delegated acceptance is disabled in this copy: the record below names no acceptor, operator or spread, and its
# closing time has passed, so accept() refuses every request. Code enforces each check.
DELEGATION = {
    "accepted_by": "none (delegated acceptance disabled)",
    "operator": {"model": "none", "session": "none"},
    "role": "none: delegated acceptance is disabled",
    "authority": "none",
    "authority_sha256": "none",
    "run_ids": (),
    "decision": "none: no spread is accepted under this delegation",
    "scope": "no spread: delegated acceptance is disabled",
    "end_trigger": "the closing time (closes_at), which has passed",
    "closes_at": "1970-01-01T00:00:00+00:00",
    "monitoring_owner": "none",
    "relation": "none: no acceptor is delegated; an acceptance settles no claim, "
                "is not independent validation "
                "and is not human review"}
# The tables of one spread that the fingerprint covers and the triggers freeze once it is accepted.
FP_TABLES = ("spread_rows", "row_sources", "totals", "checks", "exceptions", "open_claims", "provisional_amounts",
             "open_claim_counts", "claim_status", "recorded_claims")
ACCEPT_SCHEMA = """
CREATE TABLE IF NOT EXISTS acceptances (id TEXT PRIMARY KEY, schema TEXT NOT NULL, run_id TEXT NOT NULL REFERENCES
    spreads, fingerprint TEXT NOT NULL, claim_ids TEXT NOT NULL, unique_claims INTEGER NOT NULL, accepted_by TEXT
    NOT NULL, operator TEXT NOT NULL, role TEXT NOT NULL, authority TEXT NOT NULL, authority_sha256 TEXT NOT NULL,
    decision TEXT NOT NULL, scope TEXT NOT NULL, residual_risk TEXT NOT NULL, reason TEXT NOT NULL, evidence TEXT
    NOT NULL, accepted_at TEXT NOT NULL, closes_at TEXT NOT NULL, end_trigger TEXT NOT NULL, monitoring_owner TEXT
    NOT NULL, relation TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS acceptance_revocations (id TEXT PRIMARY KEY, acceptance_id TEXT NOT NULL REFERENCES
    acceptances, revoked_at TEXT NOT NULL, revoked_by TEXT NOT NULL, reason TEXT NOT NULL);
CREATE TRIGGER IF NOT EXISTS acceptances_no_update BEFORE UPDATE ON acceptances
    BEGIN SELECT RAISE(ABORT, 'acceptances are append-only'); END;
CREATE TRIGGER IF NOT EXISTS acceptances_no_delete BEFORE DELETE ON acceptances
    BEGIN SELECT RAISE(ABORT, 'acceptances are append-only'); END;
CREATE TRIGGER IF NOT EXISTS revocations_no_update BEFORE UPDATE ON acceptance_revocations
    BEGIN SELECT RAISE(ABORT, 'revocations are append-only'); END;
CREATE TRIGGER IF NOT EXISTS revocations_no_delete BEFORE DELETE ON acceptance_revocations
    BEGIN SELECT RAISE(ABORT, 'revocations are append-only'); END;
CREATE TRIGGER IF NOT EXISTS filings_frozen_update BEFORE UPDATE ON filings WHEN EXISTS (SELECT 1 FROM acceptances a
    JOIN spreads s USING (run_id) WHERE s.accession = OLD.accession)
    BEGIN SELECT RAISE(ABORT, 'an accepted spread''s filing is frozen'); END;
CREATE TRIGGER IF NOT EXISTS filings_frozen_delete BEFORE DELETE ON filings WHEN EXISTS (SELECT 1 FROM acceptances a
    JOIN spreads s USING (run_id) WHERE s.accession = OLD.accession)
    BEGIN SELECT RAISE(ABORT, 'an accepted spread''s filing is frozen'); END;
CREATE TRIGGER IF NOT EXISTS template_versions_frozen_update BEFORE UPDATE ON template_versions WHEN EXISTS (SELECT 1
    FROM acceptances a JOIN spreads s USING (run_id) WHERE s.template_version_id = OLD.id)
    BEGIN SELECT RAISE(ABORT, 'an accepted spread''s template version is frozen'); END;
CREATE TRIGGER IF NOT EXISTS template_versions_frozen_delete BEFORE DELETE ON template_versions WHEN EXISTS (SELECT 1
    FROM acceptances a JOIN spreads s USING (run_id) WHERE s.template_version_id = OLD.id)
    BEGIN SELECT RAISE(ABORT, 'an accepted spread''s template version is frozen'); END;
""" + "".join(
    f"CREATE TRIGGER IF NOT EXISTS {t}_frozen_{op.lower()} BEFORE {op} ON {t} WHEN EXISTS (SELECT 1 FROM acceptances "
    f"WHERE run_id = OLD.run_id) BEGIN SELECT RAISE(ABORT, 'an accepted spread''s records are frozen'); END;\n"
    for t in ("spreads",) + FP_TABLES for op in ("UPDATE", "DELETE"))
CLAIM_FILES = ("claims.json", "lines.json", "exceptions.json", "computed_totals.json")
# Analyst versions and their sign-off. The tables are made when an analyst version is first
# imported or reviewed (ensure_analyst), so a store that never held one keeps its bytes. All are append-only.
# Fix and sign-off were removed since: the tables stay so old
# records remain readable; an old analyst version still imports and is exported like any spread.
ANALYST_TABLES = ("analyst_versions", "review_inventories", "review_readings", "signoffs", "signoff_revocations")
ANALYST_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyst_versions (run_id TEXT PRIMARY KEY REFERENCES spreads, job_id TEXT NOT NULL,
    version TEXT NOT NULL, statement TEXT NOT NULL, parent TEXT NOT NULL, parent_binding TEXT NOT NULL,
    by TEXT NOT NULL, edits_sha256 TEXT NOT NULL, made_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS review_inventories (id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES spreads,
    fingerprint TEXT NOT NULL, claim_ids TEXT NOT NULL, escalations TEXT NOT NULL, items TEXT NOT NULL,
    packet TEXT NOT NULL, packet_sha256 TEXT NOT NULL, frozen_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS review_readings (id TEXT PRIMARY KEY, inventory_id TEXT NOT NULL REFERENCES
    review_inventories, reviewer TEXT NOT NULL, reading TEXT NOT NULL, reading_sha256 TEXT NOT NULL,
    saved_at TEXT NOT NULL, comparison TEXT NOT NULL, agrees INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS signoffs (id TEXT PRIMARY KEY, schema TEXT NOT NULL, run_id TEXT NOT NULL REFERENCES
    spreads, inventory_id TEXT NOT NULL REFERENCES review_inventories, readings TEXT NOT NULL, fingerprint TEXT
    NOT NULL, claim_ids TEXT NOT NULL, escalations TEXT NOT NULL, signed_by TEXT NOT NULL, operator TEXT NOT NULL,
    role TEXT NOT NULL, authority TEXT NOT NULL, authority_sha256 TEXT NOT NULL, decision TEXT NOT NULL, scope TEXT
    NOT NULL, relation TEXT NOT NULL, code_checked TEXT NOT NULL, revisions TEXT NOT NULL, reason TEXT NOT NULL,
    residual_risk TEXT NOT NULL, signed_at TEXT NOT NULL, closes_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS signoff_revocations (id TEXT PRIMARY KEY, signoff_id TEXT NOT NULL REFERENCES signoffs,
    revoked_at TEXT NOT NULL, revoked_by TEXT NOT NULL, reason TEXT NOT NULL);
""" + "".join(
    f"CREATE TRIGGER IF NOT EXISTS {t}_no_{op.lower()} BEFORE {op} ON {t} "
    f"BEGIN SELECT RAISE(ABORT, '{t} are append-only'); END;\n"
    for t in ANALYST_TABLES for op in ("UPDATE", "DELETE"))


class Rejected(Exception):
    """An import that cannot be stored as recorded; nothing of it is written."""


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# spread.setup's refusals in plain words; the code stays as detail
SETUP_SAID = {"C1": "the filing's fiscal period cannot be read from its cover page date",
              "C2": "the filing has no page for this statement",
              "C7": "the cash flow's operating section does not start from net income in the filing's calculations"}


def refusal_said(at, kind, reason):
    """A refused statement's refusal after "refused": at setup in plain words, the code's reason as detail; else
    where, its kind and reason."""
    if at == "setup":
        return f"at setup: {SETUP_SAID.get(kind, kind)} (detail: {reason})"
    return f"at {at} ({kind}): {reason}"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p):
    """A path as recorded under the project folder (forward slashes), or as given when outside it."""
    p = Path(p)
    p = p if p.is_absolute() else ROOT / p
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.resolve().as_posix()


def exact_sum(xs):
    """Decimal amounts added without rounding, whatever their length: an inexact result raises."""
    with localcontext() as ctx:
        ctx.prec, ctx.traps[Inexact] = MAX_PREC, True
        return sum((Decimal(x) for x in xs), Decimal(0))


def js(x):
    return None if x is None else json.dumps(x, sort_keys=True, default=str)


def connect(path=DB):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, isolation_level=None)      # transactions are opened explicitly
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    con.executescript(ACCEPT_SCHEMA)
    return con


# ---- templates ---------------------------------------------------------------------------------------------------

def job_parts(con, job, key="parts"):
    """The parts a job ran with ({"tiebreaker": "on"|"off", "tree": ...}), or with key "review" its review settings
    ({"checker": ..., "second_look": ...}), from its stored request (the job's manifest); None for a job whose
    manifest does not name them, or no job."""
    row = con.execute("SELECT request FROM jobs WHERE id = ?", (job,)).fetchone() if job else None
    return (json.loads(row[0] or "null") or {}).get(key) if row else None


def add_version(con, template_id, path):
    """The next version of a template: the file at path, with its full fingerprint. A version's file is never
    overwritten once used; a changed file is a new version at a new path."""
    data = json.loads(Path(path).read_text(encoding="utf8"))
    if data.get("format") not in TEMPLATE_FORMATS:
        raise Rejected(f"{path} is not a template of format {' or '.join(TEMPLATE_FORMATS)}")
    other = con.execute("SELECT sha256 FROM template_versions WHERE path = ? AND sha256 != ?",
                        (rel(path), sha(path))).fetchone()
    if other:
        raise Rejected(f"{rel(path)} is registered with other content; a changed template is a new file at a new path")
    n = con.execute("SELECT coalesce(max(version), 0) + 1 FROM template_versions WHERE template_id = ?",
                    (template_id,)).fetchone()[0]
    vid = uuid.uuid4().hex
    con.execute("INSERT INTO template_versions VALUES (?, ?, ?, ?, ?, ?)",
                (vid, template_id, n, rel(path), sha(path), now()))
    return vid


def add_template(con, name, path):
    tid = uuid.uuid4().hex
    con.execute("BEGIN")
    try:
        con.execute("INSERT INTO templates VALUES (?, ?, ?)", (tid, name, now()))
        vid = add_version(con, tid, path)
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise
    return tid, vid


# ---- one run -----------------------------------------------------------------------------------------------------

def read_run(folder):
    d = Path(folder)
    missing = [n for n in REQUIRED if not (d / n).exists()]
    if missing:
        raise Rejected(f"not a finished spread folder: missing {missing}")
    got = {n: json.loads((d / n).read_text(encoding="utf8")) for n in REQUIRED}
    files = {n: sha(d / n) for n in REQUIRED + OPTIONAL if (d / n).exists()}
    return got, files


def run_template(con, run, outcome):
    """The run's active template, from the resources it recorded opening: its file (read to name rows and statements,
    only at the recorded fingerprint) and the registered version with that path and fingerprint. The association
    stays unresolved, never guessed, unless exactly one version matches; the legacy skeleton fingerprint is not used."""
    opened = outcome.get("opened_resources") or run.get("hashes", {}).get("approved_resources") or {}
    found = []
    for p, h in opened.items():
        try:
            data = json.loads(Path(p).read_text(encoding="utf8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("format") in TEMPLATE_FORMATS:
            found.append((p, h, data))
    if len(found) != 1:
        raise Rejected(f"the run recorded {len(found)} template files among its opened resources")
    p, h, data = found[0]
    if sha(p) != h:
        raise Rejected(f"the template file {rel(p)} no longer has the fingerprint the run recorded")
    ids = [r[0] for r in con.execute("SELECT id FROM template_versions WHERE path = ? AND sha256 = ?", (rel(p), h))]
    note = None if len(ids) == 1 else ("no registered template version has the recorded path and fingerprint"
                                       if not ids else f"{len(ids)} registered versions match; not chosen")
    return (ids[0] if len(ids) == 1 else None), note, rel(p), h, data


def cover(f, concept):
    vals = sorted({x["value"] for x in f.by_concept.get(concept, []) if not f.contexts[x["context"]]["dims"]})
    return vals[0] if len(vals) == 1 else None


def cover_period(f):
    """The filing's current period: the dates of the context of its dimensionless dei:DocumentPeriodEndDate."""
    ctxs = {(f.contexts[x["context"]]["start"], f.contexts[x["context"]]["end"] or f.contexts[x["context"]]["instant"])
            for x in f.by_concept.get("dei:DocumentPeriodEndDate", []) if not f.contexts[x["context"]]["dims"]}
    return ctxs.pop() if len(ctxs) == 1 else (None, None)


def episode(c4, src):
    """The review state and answer of a line's final episode, as the coordinator recorded it: an accepted transfer's
    episode replaces the sender's; a queued line carries its compared outcome; a vote that replaced the analyst's
    answer makes the checker's answer the final one. After a second look the state says how it ended: resolved
    after review, which is not a blind agreement, or still disputed."""
    e = c4.get(src) or {}
    e = e.get("transfer") or e
    if e.get("queued"):
        return e.get("outcome") or "not reviewed", (e.get("c4") or {}).get("answer")
    state = (e.get("consistency") or {}).get("result") or "not reviewed"
    fb = e.get("feedback") or {}
    if fb.get("ran"):
        state += ("; resolved after review (the analyst's second look, seeing the checker's answer, matches it; not a "
                  "blind agreement)" if (fb.get("after") or {}).get("result") == "agree"
                  else "; still disputed after the analyst's second look")
    vote = (e.get("vote") or {}).get("result")
    if vote:
        state += f"; third opinion: {vote}"
    final = e.get("review") if vote == "checker" else e.get("c4")
    return state, (final or {}).get("answer")


def sic_of(cik):
    """The SIC code from a stored SEC submissions response, when one was kept; otherwise not recorded."""
    for p in sorted((ROOT / "data/manifests").glob(f"*_responses/CIK{int(cik):010d}.json")):
        s = json.loads(p.read_text(encoding="utf8")).get("sic")
        if s:
            return str(s), rel(p)
    return None, "no stored SEC submissions response"


def build(con, folder, label, job=None):
    """Everything one run puts into the store, validated against its own files; nothing is written here. A job's
    statement version (`job`: the job's ID) is stored as <spread run>@<job>/<version folder>: the original (_v0) and
    each repaired version keep their own identities, so importing one never replaces another."""
    got, files = read_run(folder)
    run, outcome, rows = got["run.json"], got["outcome.json"], got["rows.json"]
    if run.get("kind") != "spread":
        raise Rejected(f"run kind {run.get('kind')!r}, not a spread")
    vid, tnote, tpath, tsha, t = run_template(con, run, outcome)
    sec_of = {s["id"]: s for s in t["sections"]}
    sections = run["section"]
    unknown = [s for s in sections if s not in sec_of]
    if unknown:
        raise Rejected(f"sections not in the template: {unknown}")
    statements = sorted({sec_of[s]["statement"] for s in sections})
    if len(statements) != 1:
        raise Rejected(f"one statement per spread; the run covers {statements}")
    statement = statements[0]
    names = {r["row_id"]: (r["name"], r["section"]) for r in t["rows"]}
    extra = sorted(set(rows) - set(names))
    if extra:
        raise Rejected(f"rows not in the template: {extra}")
    not_app = got["diagnostics.json"].get("not_applicable") or {}

    # the filing and its company, from the filing's own cover facts
    fdir = ROOT / run["filing"]
    f = Filing(fdir)
    form = cover(f, "dei:DocumentType")
    if form != "10-K":
        raise Rejected(f"form {form!r}; only 10-Ks are stored")
    cik, accession = fdir.parent.name, fdir.name
    sic, sic_source = sic_of(cik)
    name = cover(f, "dei:EntityRegistrantName")
    company = (cik, name, "dei:EntityRegistrantName in the filing" if name else "not recorded", sic, sic_source)
    filing = (accession, cik, form, outcome.get("period_end"), cover(f, "dei:DocumentFiscalYearFocus"),
              cover(f, "dei:DocumentFiscalPeriodFocus"), outcome.get("taxonomy_year"), rel(fdir),
              js(outcome.get("filing_inputs")))

    # the statement's period, from its printed lines
    periods = {tuple(ln["period"]) for ln in got["lines.json"] if ln.get("period")}
    if len(periods) != 1:
        raise Rejected(f"the lines carry {len(periods)} periods")
    (period,) = periods
    if period[0] != t["statements"][statement]["period"]:
        raise Rejected(f"period kind {period[0]} for a {t['statements'][statement]['period']} statement")
    start, end = (None, period[1]) if period[0] == "instant" else (period[1], period[2])
    cur = cover_period(f)       # checked against the filing's current period
    if end != cur[1] or (start is not None and start != cur[0]) or outcome.get("period_end") not in (None, end):
        raise Rejected(f"statement period {start} to {end}; the filing's current period is {cur[0]} to {cur[1]}")

    # rows: the raw placed sum and the usable amount kept apart
    lines = {ln["concept"]: ln for ln in got["lines.json"]}
    claims = got["claims.json"]
    by_row = {}
    for c in claims:
        by_row.setdefault(c["row"], []).append(c)
    totals = got["computed_totals.json"]
    unavailable = {r for x in totals.values() for r in x.get("unavailable") or []}
    incomplete = {s for x in totals.values() for s in x.get("incomplete_sections") or {}}
    incomplete |= {names[r][1] for r in unavailable if r in names}
    currency = {c["section"]: c["currencies"] for c in got["checks.json"]
                if c.get("check") == "one currency in the section" and c.get("ok")}
    out_rows, problems, row_unit = [], [], {}
    for seq, r in enumerate(x for x in t["rows"] if x["section"] in sections):
        code, sec = r["row_id"], r["section"]
        mine = by_row.get(code, [])
        units = {lines.get(c["source"], {}).get("unit") for c in mine}
        if None in units or len(units) > 1:     # reported, never filled in or combined
            problems.append(f"{code}: its contributions carry units {sorted(map(str, units))}")
        unit = next(iter(units)) if len(units) == 1 else None
        if not mine:                            # nothing placed: the section's one currency, when the run checked it
            unit = currency[sec][0] if len(currency.get(sec) or ()) == 1 else None
        row_unit[code] = unit
        if sec in not_app:
            out_rows.append((seq, code, r["name"], sec, None, None, "not applicable", None))
            continue
        if code not in rows:
            if r["line_type"] == "item":
                problems.append(f"{code}: an item row of a built section missing from rows.json")
            continue            # template totals and headings live in totals, not rows
        placed = rows[code]
        summed = exact_sum(c["signed"] for c in mine)
        if Decimal(placed) != summed:
            problems.append(f"{code}: rows.json {placed} but its claims add to {summed}")
        if sec in incomplete:
            state, usable = "incomplete section", None
        elif mine:
            state, usable = ("placed amount" if Decimal(placed) else "zero contribution"), placed
        elif sec_of[sec]["kind"] == "supplementary":     # an empty memo row is blank, never 0
            state, usable = "no memo amount", None
        else:
            state, usable = "no placement", placed
        out_rows.append((seq, code, r["name"], sec, placed, usable, state, unit))
    if problems:
        raise Rejected("; ".join(problems))

    c4 = got["c4.json"]

    def source(c):
        state, ans = episode(c4, c["source"])
        qs = c.get("quote_source")     # a piece's own evidence first, else the final episode's quote
        if qs is None and isinstance(ans, dict):
            qs = ans.get("_quote_source")
        return (c["row"], c.get("section"), c["source"], lines.get(c["source"], {}).get("label"),
                c["concept"] if c["concept"] != c["source"] else None, c.get("fact"), c["signed"], js(c.get("facts")),
                c.get("by"), lines.get(c["source"], {}).get("unit"), state, (qs or {}).get("quote"), js(qs))
    sources = [source(c) for c in claims]

    checks = []
    for c in got["checks.json"]:
        checks.append(("check", c.get("section"), c.get("check"), "ok" if c.get("ok") else "failed", c.get("filed"),
                       c.get("built", c.get("parts")), c.get("gap"), c.get("tolerance"), js(c)))
    for group, xs in got["controls.json"].items():
        for c in xs:
            checks.append(("control", c.get("section"), c.get("filed"), c.get("state", group), c.get("filed_value"),
                           c.get("built"), c.get("gap"), c.get("tolerance"), js(c)))
    for c in got["report_checks.json"]:
        checks.append(("report check", None, c.get("name"), c.get("state"), (c.get("left") or {}).get("value"),
                       (c.get("right") or {}).get("value"), c.get("difference", c.get("gap")), c.get("tolerance"),
                       js(c)))
    exceptions = [(e.get("step"), e.get("class"), e.get("section"), e.get("concept"), e.get("why"), js(e))
                  for e in got["exceptions.json"]]
    spread = dict(run_id=run["run_id"] if job is None else f"{run['run_id']}@{job}/{Path(folder).name}",
                  accession=accession, statement=statement, sections=js(sections),
                  not_applicable=js(not_app or None), period_kind=period[0], period_start=start, period_end=end,
                  run_folder=rel(folder), run_date=run["run_id"][:17], models=js(run.get("models")),
                  template_version_id=vid, template_note=tnote, template_path=tpath, template_sha256=tsha,
                  job_id=job, job_note=None if job else "no job recorded for this run", outcome=outcome["outcome"],
                  label=label,
                  files=js(files))
    terms = {r["row_id"]: [x["row"] for x in r["terms"]] for r in t["relationships"]}

    def units_of(code):
        """A total's unit, from its operands: the rows with a unit, through any totals among them."""
        return set().union(*map(units_of, terms[code])) if code in terms else {row_unit.get(code)} - {None}
    totals_rows = []
    for k, v in totals.items():
        u = units_of(k)
        if len(u) > 1:
            raise Rejected(f"{k}: its operands carry units {sorted(u)}")
        totals_rows.append((k, v.get("value"), js(v.get("unavailable")), js(v.get("incomplete_sections")),
                            v.get("label"), u.pop() if u else None))
    # the arm is the recorded source flag, never the presence of the overlap record: an R run without that record, or a T run with one, is rejected whole
    from .full_pipeline import ArmRecord, arm_of
    try:
        arm = arm_of(run, got["diagnostics.json"])
    except ArmRecord as e:
        raise Rejected(f"arm: {e}") from None
    cards, marks, counts = open_claims_of(claims, got["exceptions.json"], totals, terms,
                                          r7=arm == "R",
                                          line_fact={c: ln.get("fact") for c, ln in lines.items()})
    status = assess(claims, got["lines.json"], got["exceptions.json"], totals, accession)
    return dict(company=company, filing=filing, spread=spread, rows=out_rows, sources=sources, totals=totals_rows,
                checks=checks, exceptions=exceptions, open_claims=cards, provisional=marks, open_claim_counts=counts,
                claim_status=(status["status"], status["reason"], js(status["detail"])),
                recorded_claims=[(i, *r) for i, r in enumerate(status["records"])])


def assess(claims, lines, exceptions, totals, accession):
    """The claim status of one spread, from its committed claims, printed lines, exceptions and computed totals:
    {"status", "reason", "detail", "records"}. It reads the same in both arms: every unknown-overlap claim a committed
    claim records (open_claims.recorded: placement's unknown_overlaps, a repair split's reviewed overlap claims), R7's
    marks and its "unresolved overlap" exceptions.

    claim-free      every claim can be read and none records an unknown overlap.
    open claims     at least one recorded claim of a gated origin (GATED_ORIGINS), with the counts by kind and by
                    origin and the claim ids.
    not assessable  the record does not let code tell: a piece written without its overlap claims
                    (open_claims.assessable), a claim that cannot be told from its line, a mark or an exception
                    without a recorded claim, or anything unreadable. Never read as claim-free.
    Nothing here settles a claim, and no count is filled in for a record that cannot be read."""
    try:
        return assessed(claims, lines, exceptions, totals, accession)
    except (KeyError, TypeError, AttributeError, ValueError) as e:
        return {"status": NOT_ASSESSABLE, "reason": f"the record cannot be read: {e!r}",
                "detail": {"counts": None, "note": "no count: the record could not be read"}, "records": []}


def assessed(claims, lines, exceptions, totals, accession):
    if not (isinstance(claims, list) and isinstance(lines, list) and isinstance(exceptions, list)
            and isinstance(totals, dict)):
        raise TypeError("claims, lines and exceptions are lists and the computed totals a mapping")
    line_fact = {ln["concept"]: ln.get("fact") for ln in lines}
    problems, records = [], []
    for c in claims:
        src, fact = c["source"], c.get("fact")
        if not fact or line_fact.get(src) is None:
            problems.append(f"{fact or 'a claim'} on {src}: the record does not tell a piece from its line")
        elif not oc.assessable(c, line_fact[src]):
            problems.append(f"piece {fact} of {src} is recorded without its overlap claims")
        elif oc.record_gaps(c):
            problems.append(f"{fact} on {src}: " + "; ".join(oc.record_gaps(c)))
        else:
            for r in oc.recorded(c, accession):
                if not (isinstance(r["with"], str) and r["with"]):
                    raise ValueError(f"{fact}: an overlap claim does not name the amount it concerns")
                amount = c.get("amount")
                records.append((r["id"], r["kind"], r["origin"], c.get("row"), src, c.get("concept"), fact,
                                None if amount is None else str(amount), r["with"],
                                None if r["basis"] is None else str(r["basis"]),
                                None if r["reading"] is None else str(r["reading"])))
    held = {(r[4], r[6]) for r in records}
    for e in exceptions:
        if e.get("class") == OPEN_KIND and (e.get("concept"), e.get("piece")) not in held:
            problems.append(f"an {OPEN_KIND} exception names {e.get('piece')}, which records no claim")
    lines_held = {r[4] for r in records}
    for c in claims:
        if oc.marked(c) and c["source"] not in lines_held:
            problems.append(f"{c.get('fact')} on {c['source']} is marked provisional and its line records no claim")
    if not records:
        problems += [f"total {k} is marked provisional and the spread records no claim"
                     for k, v in totals.items() if v.get("provisional")]
    gated = [r for r in records if r[2] in GATED_ORIGINS]
    count = lambda xs, i, names: {n: sum(1 for r in xs if r[i] == n) for n in names}     # noqa: E731
    detail = {"counts": {"records": len(records), "unique_claims": len({r[0] for r in records}),
                         "pieces": len({(r[4], r[6]) for r in records}),
                         "by_kind": count(records, 1, (oc.RELAXED, oc.ALREADY)),
                         "by_origin": count(records, 2, (oc.PLACEMENT, oc.REPAIR)),
                         "gated_records": len(gated), "gated_unique_claims": len({r[0] for r in gated})},
              "gated_origins": list(GATED_ORIGINS), "claim_ids": sorted({r[0] for r in records}),
              "lines_affected": sorted(lines_held)}
    if problems:
        detail["note"] = "the counts cover only the claims that could be read; they are not the spread's count"
        more = f"; and {len(problems) - 5} more" if len(problems) > 5 else ""
        return {"status": NOT_ASSESSABLE, "reason": "; ".join(problems[:5]) + more, "detail": detail,
                "records": records}
    return {"status": OPEN_CLAIMS if gated else CLAIM_FREE, "reason": None, "detail": detail, "records": records}


def claim_status(con, run_id):
    """A stored spread's claim status: {"status", "reason", "detail", "source"}. From the claim record stored at
    import; for a spread stored before that record existed, from the run's own files, read only at the fingerprints
    stored at import (a missing or changed file gives "not assessable"). Either reading is then checked against what
    the store holds of R7: a spread read as claim-free whose store rows hold a decision card, a provisional mark, a
    counted open claim or an "unresolved overlap" exception contradicts itself and is "not assessable". This is the
    one function select() and the export ask."""
    row = con.execute("SELECT accession, run_folder, files FROM spreads WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        raise Rejected(f"{run_id} is not stored")
    accession, folder, files = row
    stored = con.execute("SELECT status, reason, detail FROM claim_status WHERE run_id = ?", (run_id,)).fetchone()
    if stored is not None:
        try:
            st = {"status": stored[0], "reason": stored[1], "detail": json.loads(stored[2])}
            if st["status"] not in (CLAIM_FREE, OPEN_CLAIMS, NOT_ASSESSABLE) or not isinstance(st["detail"], dict):
                raise ValueError(f"status {stored[0]!r}")
        except ValueError as e:
            st = {"status": NOT_ASSESSABLE, "reason": f"the stored claim record cannot be read: {e!r}",
                  "detail": {"counts": None}}
        st["source"] = "the claim record stored at import"
    else:
        st = status_from_files(accession, folder, files)
    if st["status"] == CLAIM_FREE:
        n = {"decision cards": con.execute("SELECT COUNT(*) FROM open_claims WHERE run_id = ?", (run_id,)),
             "provisional marks": con.execute("SELECT COUNT(*) FROM provisional_amounts WHERE run_id = ?", (run_id,)),
             "counted open claims": con.execute("SELECT COALESCE(SUM(open_claims), 0) FROM open_claim_counts WHERE "
                                                "run_id = ?", (run_id,)),
             f"{OPEN_KIND} exceptions": con.execute("SELECT COUNT(*) FROM exceptions WHERE run_id = ? AND class = ?",
                                                    (run_id, OPEN_KIND))}
        n = {k: v.fetchone()[0] for k, v in n.items()}
        if any(n.values()):
            st = {**st, "status": NOT_ASSESSABLE,
                  "reason": "the store's records disagree: read as claim-free, yet the store holds "
                            + ", ".join(f"{v} {k}" for k, v in n.items() if v)}
    return st


def status_from_files(accession, folder, files):
    """The claim status of a spread stored without its claim record, from the run's files. Each file is read once
    and must have the fingerprint stored at import; otherwise the spread cannot be assessed."""
    source = "the run's files, at the fingerprints stored at import (no claim record was stored with this spread)"

    def no(why):
        return {"status": NOT_ASSESSABLE, "detail": {"counts": None}, "source": source,
                "reason": f"the store holds no claim record for this spread (stored before the rule) and {why}"}
    try:
        fp = json.loads(files)
    except (TypeError, ValueError):
        fp = None
    if not isinstance(fp, dict):
        return no("its stored file fingerprints cannot be read")
    d = Path(folder)
    d = d if d.is_absolute() else ROOT / d
    got = {}
    for n in CLAIM_FILES:
        try:
            raw = (d / n).read_bytes()
        except OSError:
            return no(f"{n} is no longer in {folder}")
        if fp.get(n) != hashlib.sha256(raw).hexdigest():
            return no(f"{n} in {folder} does not have the fingerprint stored at import")
        try:
            got[n] = json.loads(raw.decode("utf8"))
        except ValueError:
            return no(f"{n} in {folder} cannot be read")
    st = assess(got["claims.json"], got["lines.json"], got["exceptions.json"], got["computed_totals.json"], accession)
    return {k: v for k, v in st.items() if k != "records"} | {"source": source}


def claims_said(st):
    """A spread's open claims in words: how many, on how many pieces, by kind and origin. They are not settled; the
    export marks the amounts they affect PROVISIONAL and lists them (no gate: downloading the result
    supersedes the use gate)."""
    c = st["detail"]["counts"]
    parts = [f"{v} {k}" for k, v in {**c["by_kind"], **{f"from {k}": v for k, v in c["by_origin"].items()}}.items()
             if v]
    return (f"{c['gated_unique_claims']} unknown-overlap claims on {c['pieces']} piece{'s' if c['pieces'] != 1 else ''}"
            f" ({'; '.join(parts)}); not settled")


def unassessable_said(run_id, st, use="be made current"):
    """The refusal of a spread whose claim record cannot be assessed: the store cannot say what it holds."""
    return (f"{run_id} cannot {use}: its claims cannot be assessed ({st['reason']}); a spread that cannot be assessed "
            f"is not treated as claim-free.")


def has_analyst(con):
    return con.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'analyst_versions'").fetchone() \
        is not None


def ensure_analyst(con):
    con.executescript(ANALYST_SCHEMA)


def analyst_version(con, run_id):
    """The stored analyst version record, or None. A spread whose folder sits under a job's analyst/ folder is an
    analyst version whether or not its record is found."""
    if has_analyst(con):
        cur = con.execute("SELECT * FROM analyst_versions WHERE run_id = ?", (run_id,))
        row = cur.fetchone()
        if row:
            return dict(zip([c[0] for c in cur.description], row))
    folder = con.execute("SELECT run_folder FROM spreads WHERE run_id = ?", (run_id,)).fetchone()
    if folder and Path(folder[0].replace("\\", "/")).parent.name == "analyst":
        return {"run_id": run_id, "missing": True}
    return None


def assessable(con, run_id):
    """(ok, claim status): whether the store can say what a spread's claim record holds, claim-free or open claims.
    Not a use gate: open claims are shown and marked, never a reason to refuse. A spread whose
    claims cannot be assessed is refused at selection and export, since its stored rows and files disagree."""
    st = claim_status(con, run_id)
    return st["status"] in (CLAIM_FREE, OPEN_CLAIMS), st


def open_cases(con, run_id):
    """One review case per piece with a recorded claim: the piece, its claims as a list (newly relaxed first, each
    with its own id, kind and origin), the two counts, and the amount at stake once: the piece's amount without its
    sign, computed here. A case's claims concern one amount, so nothing is added across them. Grouping changes no
    claim and is not read by the gate. From the recorded claims; for a spread stored before them, from its decision
    cards."""
    accession = con.execute("SELECT accession FROM spreads WHERE run_id = ?", (run_id,)).fetchone()[0]
    cards = {}
    for r in con.execute("SELECT kind, row_code, section, line, line_label, piece, fact, piece_amount, rival, "
                         "rival_amount, basis, quote, passage, at_stake, unit, actions, detail FROM open_claims WHERE "
                         "run_id = ? ORDER BY seq", (run_id,)):
        cards.setdefault((r[3], r[6], r[8]), r)
    recs = con.execute("SELECT claim_id, kind, origin, row_code, line, piece, fact, piece_amount, rival, basis FROM "
                       "recorded_claims WHERE run_id = ? ORDER BY seq", (run_id,)).fetchall()
    if not recs:        # stored before the recorded claims: the cards are the record
        recs = [(json.loads(k[16]).get("claim_id") or oc.claim_id(accession, k[6], k[8]), k[0], oc.PLACEMENT, k[1],
                 k[3], k[5], k[6], k[7], k[8], k[10]) for k in cards.values()]
    cases = {}
    for cid, kind, origin, row, line, piece, fact, amount, rival, basis in recs:
        k = cards.get((line, fact, rival))
        src = con.execute("SELECT line_label, unit, quote FROM row_sources WHERE run_id = ? AND line = ? AND fact = ?",
                          (run_id, line, fact)).fetchone() or (None, None, None)
        case = cases.setdefault((line, fact), {
            "line": line, "line_label": (k[4] if k else None) or src[0], "fact": fact, "piece": piece, "row_code": row,
            "piece_amount": amount, "unit": (k[14] if k else None) or src[1], "quote": (k[11] if k else None) or src[2],
            "passage": k[12] if k else None, "claims": []})
        case["claims"].append({"claim_id": cid, "kind": kind, "origin": origin, "rival": rival,
                               "rival_amount": k[9] if k else None, "basis": basis,
                               "actions": json.loads(k[15] or "[]") if k else []})
    out = []
    for case in cases.values():
        case["claims"].sort(key=lambda x: x["kind"] != OPEN_RELAXED)
        case[OPEN_RELAXED] = sum(1 for x in case["claims"] if x["kind"] == OPEN_RELAXED)
        case[oc.ALREADY] = len(case["claims"]) - case[OPEN_RELAXED]
        try:
            case["at_stake"] = abs(Decimal(case["piece_amount"]))
        except (TypeError, ArithmeticError):
            case["at_stake"] = None                     # unknown stays unknown: no amount is put in its place
        case["at_stake_note"] = oc.CASE_NOTE
        out.append(case)
    return out


def open_claims_of(claims, exceptions, totals, terms, r7=False, line_fact=None):
    """R7's open claims of one run, read from its own files: (cards, marks, counts). cards: one row per decision
    card of an "unresolved overlap" exception. marks: every provisional source (a piece or the rest of its line),
    row and computed total, with its number of open claims. counts: (open claims, newly relaxed, already unknown).
    A run without open claims gives ([], [], (0, 0, 0)). `r7`: the run recorded R7 (diagnostics.open_claims).

    The flag check, with its consequence: a run in which a provisional piece lacks its exception or its cards, an
    exception names a piece that is not provisional, the rest of a provisional piece's line is unmarked, or a
    computed total over a provisional row is unmarked (or marked without one), is rejected whole. No amount of such
    a run enters the store, so none can be exported without its mark.

    The mark is R7's (open_claims.marked: provisional with open claims). A repair split's piece carries its own
    `provisional` (steps.split, "provisional until C8 review") and no open claims; it is not this mark. `line_fact`
    (line -> its own fact): the rest of a line is the claim on the line's own fact, so a repair piece on the same
    line is not taken for it; without it, every claim that is not a placement piece."""
    excs = [e for e in exceptions if e.get("class") == OPEN_KIND]
    pieces = {c["fact"]: c for c in claims if "unknown_overlaps" in c and oc.marked(c)}
    problems = []
    for c in claims:
        if "unknown_overlaps" in c and c["unknown_overlaps"] and (r7 or excs or pieces) and not oc.marked(c):
            problems.append(f"{c['fact']} has unknown overlaps and no provisional mark")
    by_piece = {}
    for e in excs:
        by_piece.setdefault(e.get("piece"), []).append(e)
    for fid, c in pieces.items():
        n = sum(len(e.get("cards") or ()) for e in by_piece.get(fid, ()))
        if len(by_piece.get(fid, ())) != 1 or n != len(c["unknown_overlaps"]) or n != len(c.get("open_claims") or ()):
            problems.append(f"{fid}: {len(c['unknown_overlaps'])} unknown overlaps, "
                            f"{len(c.get('open_claims') or ())} open claims, {n} decision cards in "
                            f"{len(by_piece.get(fid, ()))} exceptions")
    problems += [f"an {OPEN_KIND} exception names {fid}, which is not a provisional piece"
                 for fid in by_piece if fid not in pieces]
    sources = {c["source"] for c in pieces.values()}
    def rest(c):
        if line_fact is None:
            return "unknown_overlaps" not in c
        return c.get("fact") is not None and c.get("fact") == line_fact.get(c["source"])
    problems += [f"the rest of {c['source']} is not marked provisional"
                 for c in claims if rest(c) and c["source"] in sources and not oc.marked(c)]
    rows = {}
    for c in claims:
        if oc.marked(c):
            rows[c["row"]] = rows.get(c["row"], 0) + len(c.get("open_claims") or ())

    def under(code, seen=()):
        out = set()
        for r in terms.get(code) or ():
            if r is not None and r not in seen:
                out |= {r} | under(r, seen + (code,))
        return out
    marks = [("source", c["row"], c["source"], c.get("fact"), len(c.get("open_claims") or ()))
             for c in claims if oc.marked(c)]
    marks += [("row", r, None, None, n) for r, n in sorted(rows.items())]
    for k, v in totals.items():
        mine = sorted(under(k) & set(rows))
        if bool(mine) != bool(v.get("provisional")):
            problems.append(f"total {k}: provisional rows {mine}, mark {v.get('provisional')!r}")
        elif mine:
            marks.append(("total", k, None, None, len(v.get("open_claims") or ())))
    if problems:
        raise Rejected("open claims and provisional marks disagree: " + "; ".join(problems))
    cards = []
    for e in excs:
        for k in e["cards"]:
            cards.append((len(cards), k["kind"], k.get("row"), e.get("section"), k["line"], k.get("line_label"),
                          k["piece"].get("concept"), k["piece"].get("fact"), k["piece"].get("amount"),
                          k["rival"]["name"], k["rival"].get("amount"), k.get("basis"), k.get("quote"),
                          k.get("passage"), k.get("amount_at_stake"), k.get("unit"), js(k.get("actions")), js(k)))
    relaxed = sum(1 for k in cards if k[1] == OPEN_RELAXED)
    return cards, marks, (len(cards), relaxed, len(cards) - relaxed)


def import_run(con, folder, label, load_id, job=None):
    """One run in one transaction. Returns 'imported' or 'already imported'; raises Rejected otherwise."""
    x = build(con, folder, label, job)
    s = x["spread"]
    av = analyst_origin(folder, job)
    old = con.execute("SELECT files FROM spreads WHERE run_id = ?", (s["run_id"],)).fetchone()
    if old:
        if old[0] != s["files"]:
            raise Rejected("already imported, and the run's files changed since")
        return "already imported"
    if av:
        ensure_analyst(con)
    con.execute("BEGIN")
    try:
        cik = x["company"][0]
        have = con.execute("SELECT name, sic FROM companies WHERE cik = ?", (cik,)).fetchone()
        if have is None:
            con.execute("INSERT INTO companies VALUES (?, ?, ?, ?, ?)", x["company"])
        have = con.execute("SELECT * FROM filings WHERE accession = ?", (x["filing"][0],)).fetchone()
        if have is None:
            con.execute("INSERT INTO filings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", x["filing"])
        elif tuple(have[:7]) != x["filing"][:7]:
            raise Rejected(f"the filing is stored with other cover facts: {tuple(have[:7])}")
        s = {**s, "load_id": load_id, "imported": now()}
        con.execute(f"INSERT INTO spreads ({', '.join(s)}) VALUES ({', '.join('?' * len(s))})", tuple(s.values()))
        rid = s["run_id"]
        con.executemany("INSERT INTO spread_rows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", [(rid, *r) for r in x["rows"]])
        con.executemany("INSERT INTO row_sources VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        [(rid, *r) for r in x["sources"]])
        con.executemany("INSERT INTO totals VALUES (?, ?, ?, ?, ?, ?, ?)", [(rid, *r) for r in x["totals"]])
        con.executemany("INSERT INTO checks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [(rid, *r) for r in x["checks"]])
        con.executemany("INSERT INTO exceptions VALUES (?, ?, ?, ?, ?, ?, ?)", [(rid, *r) for r in x["exceptions"]])
        # the spread's open claims, its provisional amounts and its count (0 for a spread without any)
        con.executemany(f"INSERT INTO open_claims VALUES ({', '.join('?' * 19)})",
                        [(rid, *r) for r in x["open_claims"]])
        con.executemany("INSERT INTO provisional_amounts VALUES (?, ?, ?, ?, ?, ?)",
                        [(rid, *r) for r in x["provisional"]])
        con.execute("INSERT INTO open_claim_counts VALUES (?, ?, ?, ?)", (rid, *x["open_claim_counts"]))
        # the use gate: the spread's claim status and every recorded claim, in either arm
        con.execute("INSERT INTO claim_status VALUES (?, ?, ?, ?)", (rid, *x["claim_status"]))
        con.executemany(f"INSERT INTO recorded_claims VALUES ({', '.join('?' * 13)})",
                        [(rid, *r) for r in x["recorded_claims"]])
        if av:
            con.execute(f"INSERT INTO analyst_versions VALUES ({', '.join('?' * 9)})", (rid, *av))
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise
    return "imported"


def analyst_origin(folder, job):
    """The analyst version's record from its own files, or None for a pipeline version; Rejected when they
    contradict. A folder is an analyst version when its run.json says origin "analyst", it sits under its job's
    analyst/ folder, or it carries analyst_edits.json: all three must hold, with one job, version and statement."""
    d = Path(folder)
    run = json.loads((d / "run.json").read_text(encoding="utf8"))
    p = d / "analyst_edits.json"
    marks = {"run.json origin": run.get("origin") == "analyst", "analyst/ folder": d.parent.name == "analyst",
             "analyst_edits.json": p.is_file()}
    if not any(marks.values()):
        if run.get("origin") is not None:
            raise Rejected(f"contradictory origin: run.json says {run.get('origin')!r}")
        return None
    if not all(marks.values()):
        raise Rejected("contradictory origin: an analyst version needs " + ", ".join(marks) + "; missing: "
                       + ", ".join(k for k, v in marks.items() if not v))
    e = json.loads(p.read_text(encoding="utf8"))
    stmt = d.name.rsplit("_a", 1)[0]
    bad = [k for k, ok in (("origin", e.get("origin") == "analyst"), ("version", e.get("version") == d.name),
                           ("run.json version", run.get("analyst_version") == d.name),
                           ("statement", e.get("statement") == stmt),
                           ("job", job is not None and d.parent.parent.name == job),
                           ("schema", e.get("schema") == "analyst-edits/v1"),
                           ("by", bool(e.get("by"))), ("parent", bool(e.get("parent")))) if not ok]
    if bad:
        raise Rejected(f"contradictory origin: the analyst record disagrees on {', '.join(bad)}")
    return (job, d.name, stmt, e["parent"], e.get("parent_binding") or "", e["by"], sha(p), e.get("made_at") or "")


# ---- a full-pipeline job's statement versions --------------------------------------------------------------------

def final_versions(job_dir):
    """A job's final statement versions as its job.json records them (`finals`), never the highest-numbered folder.
    Each must be the job's own cumulative/<statement>_v<n> folder, listed among that statement's versions; a recorded
    final that is not is refused (a job moved whole keeps its cumulative/<name> tail)."""
    d = Path(job_dir)
    try:
        job = json.loads((d / "job.json").read_text(encoding="utf8"))
    except (OSError, ValueError):
        raise Rejected(f"{d.name}: the job has not finished (no job.json)")
    out = []
    for st, p in sorted((job.get("finals") or {}).items()):
        name = Path(str(p)).name
        listed = ((job.get("statements") or {}).get(st) or {}).get("versions") or []
        if (Path(str(p)).parent.name != "cumulative" or not re.fullmatch(rf"{re.escape(st)}_v\d+", name)
                or name not in listed or not (d / "cumulative" / name).is_dir()):
            raise Rejected(f"{d.name}: the recorded final {st} ({p}) is not one of the job's own {st} versions")
        out.append(d / "cumulative" / name)
    return out


def import_job(con, job_dir, label, load_id, versions=None):
    """A finished job's final statement versions (or the `versions` named), each in its own transaction under its own
    identity (build), after the job's row: its manifest as the request, its company limits as the budget, its end;
    and its refused statements, kept apart: a refusal has no amounts, is never a spread and is never selected."""
    d = Path(job_dir)
    try:
        job = json.loads((d / "job.json").read_text(encoding="utf8"))
    except (OSError, ValueError):
        raise Rejected(f"{d.name}: the job has not finished (no job.json)")
    if con.execute("SELECT 1 FROM jobs WHERE id = ?", (d.name,)).fetchone() is None:
        try:
            manifest = json.loads((d / "ui_manifest.json").read_text(encoding="utf8"))
        except (OSError, ValueError):
            manifest = None
        vid = ((manifest or {}).get("template") or {}).get("version_id")
        if vid and con.execute("SELECT 1 FROM template_versions WHERE id = ?", (vid,)).fetchone() is None:
            vid = None
        con.execute("INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (d.name, vid, js(manifest), js((job.get("company") or {}).get("limits")), d.name[:17],
                     time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime((d / "job.json").stat().st_mtime)),
                     job.get("end")))
    filing = ((json.loads((d / "run.json").read_text(encoding="utf8")).get("settings_content") or {})
              .get("filing")) if (d / "run.json").is_file() else None
    con.executemany("INSERT OR IGNORE INTO refusals VALUES (?, ?, ?, ?, ?, ?)", [
        (d.name, Path(str(filing)).name, n, x["refused"]["refused_at"], x["refused"].get("kind"),
         x["refused"]["reason"]) for n, x in sorted((job.get("statements") or {}).items()) if x.get("refused")])
    out = []
    for p in versions or final_versions(d):
        try:
            out.append({"folder": rel(p), "result": import_run(con, p, label, load_id, job=d.name)})
        except Rejected as e:
            out.append({"folder": rel(p), "result": "rejected", "why": str(e)})
    return out


def load_job(con, db, job_dir, label=None, versions=None):
    """import_job under a dated load record (results.json, exceptions.json), as the import command keeps one;
    `versions`: the folders named instead of the job's finals (an analyst version)."""
    rec = runrec.new_run("store_load", {"db": rel(db), "job": rel(job_dir), "label": label,
                                        **({"versions": [rel(v) for v in versions]} if versions else {})})
    try:
        results = import_job(con, job_dir, label, rec.name, versions=versions)
    except Rejected as e:
        results = [{"folder": rel(job_dir), "result": "rejected", "why": str(e)}]
    except Exception as e:
        results = [{"folder": rel(job_dir), "result": "failed", "why": repr(e)}]
    bad = [r for r in results if r["result"] in ("rejected", "failed")]
    runrec.write(rec, "results.json", results)
    runrec.write(rec, "exceptions.json", {"rejected": bad, "unresolved": []})
    return {"record": rel(rec), "results": results, "rejected": len(bad)}


# ---- acceptance of a spread with open claims ---------------------------------------------------------------------

def canon(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def fingerprint(con, run_id):
    """The spread's content fingerprint: sha256 of canonical JSON of every stored record of it (its spreads row, with
    the per-file hashes, and every row of FP_TABLES), its filing and its template version (path and hash). Rows are
    sorted by their own canonical JSON, so tables without a primary key are ordered deterministically. Code only."""
    def table(name, where, args):
        cur = con.execute(f"SELECT * FROM {name} WHERE {where}", args)
        cols = [c[0] for c in cur.description]
        return sorted(canon(dict(zip(cols, r))) for r in cur)
    spread = table("spreads", "run_id = ?", (run_id,))
    if not spread:
        raise Rejected(f"{run_id} is not stored")
    s = json.loads(spread[0])
    doc = {"schema": FINGERPRINT_SCHEMA, "spreads": spread,
           "filing": table("filings", "accession = ?", (s["accession"],)),
           "template_version": table("template_versions", "id = ?", (s["template_version_id"],)),
           **{t: table(t, "run_id = ?", (run_id,)) for t in FP_TABLES}}
    return hashlib.sha256(canon(doc).encode("utf8")).hexdigest()


def claim_ids(con, run_id):
    """Every recorded claim ID of the spread, unique and sorted."""
    return sorted({r[0] for r in con.execute("SELECT claim_id FROM recorded_claims WHERE run_id = ?", (run_id,))})


def has_acceptances(con):
    return con.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'acceptances'").fetchone() is not None


def closed(at=None):
    """True once the delegation's closing time has passed (the enforced end trigger)."""
    t = at or datetime.datetime.now(datetime.timezone.utc)
    return t >= datetime.datetime.fromisoformat(DELEGATION["closes_at"])


def accepted(con, run_id):
    """Whether the spread's latest acceptance holds: {"ok", "why", "acceptance"}. It holds only while it is not
    revoked, the delegation is not closed, the authority note keeps its hash, and the spread's current fingerprint and
    claim-ID set equal the accepted ones."""
    if not has_acceptances(con):
        return {"ok": False, "why": "no acceptance is recorded", "acceptance": None}
    cur = con.execute("SELECT * FROM acceptances WHERE run_id = ? ORDER BY accepted_at DESC, rowid DESC LIMIT 1",
                      (run_id,))
    cols = [c[0] for c in cur.description]
    row = cur.fetchone()
    if row is None:
        return {"ok": False, "why": "no acceptance is recorded", "acceptance": None}
    a = dict(zip(cols, row))
    rv = con.execute("SELECT revoked_at, revoked_by, reason FROM acceptance_revocations WHERE acceptance_id = ? "
                     "ORDER BY revoked_at LIMIT 1", (a["id"],)).fetchone()
    p = ROOT / a["authority"]
    why = (f"acceptance {a['id']} was revoked at {rv[0]} by {rv[1]}: {rv[2]}" if rv
           else f"the delegation closed at {a['closes_at']}" if closed()
           else "the authority note is missing or changed" if not p.is_file() or sha(p) != a["authority_sha256"]
           else "the spread's content changed since acceptance (fingerprint)" if fingerprint(con, run_id) != a["fingerprint"]
           else "the spread's recorded claims changed since acceptance" if json.loads(a["claim_ids"]) != claim_ids(con, run_id)
           else None)
    return {"ok": why is None, "why": why or "the acceptance holds", "acceptance": a}


def accept(con, run_id, by, fingerprint_shown, claims_shown, reason, residual_risk, evidence, operator=None):
    """Record the delegated acceptance of one spread with open claims, bound to the fingerprint and the claim IDs the
    acceptor was shown. Refused, with the reason, unless: the acceptor, operator and spread are the delegation's; the
    authority note keeps its hash; the delegation is open; the spread's status is exactly "open claims" (claim-free
    needs none, "not assessable" is never accepted); no acceptance already holds; the shown fingerprint and claim set
    are the current ones (a stale or partial submission is refused); reason and residual risk are given; and the
    evidence is a file in the project (its path and hash are recorded). One transaction; append-only."""
    d = DELEGATION
    if by != d["accepted_by"]:
        raise Rejected(f"only {d['accepted_by']} may accept under the recorded delegation ({d['authority']})")
    if (operator or d["operator"]) != d["operator"]:
        raise Rejected(f"the operator is not the delegated one: {d['operator']}")
    if run_id not in d["run_ids"]:
        raise Rejected(f"{run_id} is outside the delegation's scope ({d['scope']})")
    p = ROOT / d["authority"]
    if not p.is_file() or sha(p) != d["authority_sha256"]:
        raise Rejected(f"the authority note {d['authority']} is missing or changed")
    if closed():
        raise Rejected(f"the delegation closed at {d['closes_at']}")
    for name, v in (("reason", reason), ("residual risk", residual_risk), ("evidence", evidence)):
        if not (isinstance(v, str) and v.strip()):
            raise Rejected(f"an acceptance gives its {name}")
    ev = Path(evidence.strip())
    ev = ev if ev.is_absolute() else ROOT / ev
    try:
        ev.resolve().relative_to(ROOT)
    except ValueError:
        raise Rejected("the evidence is a file in the project folder") from None
    if not ev.is_file():
        raise Rejected(f"no evidence file {rel(ev)}")
    if not (isinstance(claims_shown, list) and all(isinstance(x, str) for x in claims_shown)):
        raise Rejected("the claim IDs shown are a list of IDs")
    con.execute("BEGIN IMMEDIATE")
    try:
        st = claim_status(con, run_id)
        if st["status"] != OPEN_CLAIMS:
            raise Rejected(f"{run_id} is {st['status']!r}; only a spread with open claims is accepted")
        if accepted(con, run_id)["ok"]:
            raise Rejected(f"{run_id} already has an acceptance that holds")
        fp, ids = fingerprint(con, run_id), claim_ids(con, run_id)
        if fingerprint_shown != fp:
            raise Rejected("the fingerprint shown is not the spread's current one: the spread changed or the page is "
                           "stale; open it again")
        if len(set(claims_shown)) != len(claims_shown) or sorted(claims_shown) != ids:
            raise Rejected(f"the claim IDs shown ({len(claims_shown)}) are not the spread's {len(ids)} current claims")
        a = {"id": uuid.uuid4().hex, "schema": ACCEPTANCE_SCHEMA, "run_id": run_id, "fingerprint": fp,
             "claim_ids": canon(ids), "unique_claims": len(ids), "accepted_by": by, "operator": canon(d["operator"]),
             "role": d["role"], "authority": d["authority"], "authority_sha256": d["authority_sha256"],
             "decision": d["decision"], "scope": d["scope"], "residual_risk": residual_risk.strip(),
             "reason": reason.strip(), "evidence": canon({"path": rel(ev), "sha256": sha(ev)}),
             "accepted_at": now(), "closes_at": d["closes_at"], "end_trigger": d["end_trigger"],
             "monitoring_owner": d["monitoring_owner"], "relation": d["relation"]}
        con.execute(f"INSERT INTO acceptances ({', '.join(a)}) VALUES ({', '.join('?' * len(a))})", tuple(a.values()))
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise
    return a


def revoke(con, acceptance_id, by, reason):
    """Revoke an acceptance: an appended row; the acceptance stays in the record and no longer holds."""
    if not (isinstance(by, str) and by.strip() and isinstance(reason, str) and reason.strip()):
        raise Rejected("a revocation names who revokes and why")
    if not has_acceptances(con) or con.execute("SELECT 1 FROM acceptances WHERE id = ?",
                                               (acceptance_id,)).fetchone() is None:
        raise Rejected(f"no acceptance {acceptance_id}")
    r = (uuid.uuid4().hex, acceptance_id, now(), by.strip(), reason.strip())
    con.execute("INSERT INTO acceptance_revocations VALUES (?, ?, ?, ?, ?)", r)
    return dict(zip(("id", "acceptance_id", "revoked_at", "revoked_by", "reason"), r))


# ---- current selection -------------------------------------------------------------------------------------------

USABLE = ("complete", "qualified")         # spread.verdict: blocked when a check fails
SELECTABLE = USABLE + ("blocked",)


def select(con, run_id, by):
    """A person makes a stored spread current for its template version, company, statement and fiscal year. A blocked
    spread can be selected for inspection where none is current, but a run that did not finish, covers fewer sections,
    or is blocked where the current one is usable never replaces it; the previous one stays.

    A spread with open claims is selected like any other: its claims stay recorded and the export marks them (the
    earlier use gate is superseded). One whose claims cannot be assessed is refused with
    the reason and the previous selection stays."""
    row = con.execute("SELECT s.template_version_id, f.cik, s.statement, f.fiscal_year, s.outcome, s.sections "
                      "FROM spreads s JOIN filings f USING (accession) WHERE s.run_id = ?", (run_id,)).fetchone()
    if row is None:
        raise Rejected(f"{run_id} is not stored")
    vid, cik, statement, year, outcome, sections = row
    if vid is None:
        raise Rejected("the run's template version is unresolved")
    if year is None:
        raise Rejected("the filing has no fiscal-year label")
    if outcome not in SELECTABLE:
        raise Rejected(f"outcome {outcome!r} cannot be selected")
    cur = con.execute("SELECT c.run_id, s.sections, s.outcome FROM current_selection c JOIN spreads s USING (run_id) "
                      "WHERE "
                      "c.template_version_id = ? AND c.cik = ? AND c.statement = ? AND c.fiscal_year = ?",
                      (vid, cik, statement, year)).fetchone()
    if cur and not set(json.loads(cur[1])) <= set(json.loads(sections)):
        raise Rejected(f"{run_id} covers fewer sections than the current {cur[0]}")
    if cur and cur[2] in USABLE and outcome not in USABLE:
        raise Rejected(f"{run_id} is {outcome}; the current {cur[0]} is {cur[2]}")
    ok, st = assessable(con, run_id)
    if not ok:
        raise Rejected(unassessable_said(run_id, st))
    con.execute("INSERT OR REPLACE INTO current_selection VALUES (?, ?, ?, ?, ?, ?, ?)",
                (vid, cik, statement, year, run_id, by, now()))
    return cur[0] if cur else None


# ---- command -----------------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m spreading.store")
    ap.add_argument("--db", default=str(DB))
    sub = ap.add_subparsers(dest="cmd", required=True)
    tp = sub.add_parser("template").add_subparsers(dest="what", required=True)
    x = tp.add_parser("add")
    x.add_argument("name")
    x.add_argument("path")
    x = tp.add_parser("version")
    x.add_argument("template_id")
    x.add_argument("path")
    x = sub.add_parser("import")
    x.add_argument("folders", nargs="+")
    x.add_argument("--label", help="experimental arm or test label")
    x = sub.add_parser("import-job")
    x.add_argument("job")
    x.add_argument("--label", help="experimental arm or test label")
    x = sub.add_parser("select")
    x.add_argument("run_id")
    x.add_argument("--by", required=True)
    x = sub.add_parser("revoke", help="revoke an acceptance (the end trigger, or any reason)")
    x.add_argument("acceptance_id")
    x.add_argument("--by", required=True)
    x.add_argument("--reason", required=True)
    a = ap.parse_args(argv)
    with contextlib.closing(connect(a.db)) as con:
        return command(a, con)


def command(a, con):
    if a.cmd == "template" and a.what == "add":
        print(json.dumps(dict(zip(("template_id", "version_id"), add_template(con, a.name, a.path)))))
    elif a.cmd == "template":
        con.execute("BEGIN")
        vid = add_version(con, a.template_id, a.path)
        con.execute("COMMIT")
        print(json.dumps({"version_id": vid}))
    elif a.cmd == "select":
        print(json.dumps({"current": a.run_id, "replaced": select(con, a.run_id, a.by)}))
    elif a.cmd == "revoke":
        print(json.dumps(revoke(con, a.acceptance_id, a.by, a.reason)))
    elif a.cmd == "import-job":
        got = load_job(con, a.db, a.job, a.label)
        print(json.dumps(got, indent=1))
        return 1 if got["rejected"] else 0
    else:
        rec = runrec.new_run("store_load", {"db": rel(a.db), "folders": [rel(p) for p in a.folders],
                                            "label": a.label})
        results, exceptions, unresolved = [], [], []
        try:
            for p in a.folders:
                try:
                    results.append({"folder": rel(p), "result": import_run(con, p, a.label, rec.name)})
                except Exception as e:      # the import's own transaction is already rolled back
                    results.append({"folder": rel(p), "result": "rejected" if isinstance(e, Rejected) else "failed"})
                    exceptions.append({"folder": rel(p), "why": str(e) if isinstance(e, Rejected) else repr(e)})
            unresolved = [{"run_id": r[0], "template": r[1], "job": r[2]} for r in con.execute(
                "SELECT run_id, template_note, job_note FROM spreads WHERE load_id = ?", (rec.name,))
                if r[1] or r[2]]
        except Exception as e:
            exceptions.append({"folder": None, "why": f"the load stopped: {e!r}"})
        finally:
            runrec.write(rec, "results.json", results)
            runrec.write(rec, "exceptions.json", {"rejected": exceptions, "unresolved": unresolved})
        print(json.dumps({"record": rel(rec), "results": results, "rejected": len(exceptions)}, indent=1))
        return 1 if exceptions else 0


if __name__ == "__main__":
    sys.exit(main())
