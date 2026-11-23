"""Draw development companies: seeded simple random sample over
the SEC exchange ticker frame, walking the shuffled list and keeping the first n eligible companies.

    python -s -E -m spreading.devdraw --n 14 --seed 20261031

Population: companies in the dated NYSE/Nasdaq snapshot of the SEC ticker file (SEC does not guarantee its scope),
with SIC present and outside 6000-6799 (banks and insurers are left out; this also removes real-estate operators and REITs), and an
original 10-K with inline XBRL, the latest one. US GAAP is assumed from the form (a 10-K, not a
20-F or 40-F) and not checked. Companies in the cumulative exclusion manifest are removed before the shuffle.
Eligibility never depends on the project's parser; a fetch error stops the walk (it is not ineligibility). Every skip is
recorded with its reason. The frame and every Submissions response used are kept. The drawn companies enter the
exclusion manifest when the draw ends, complete or not, before any of their filings is opened. A fatal fetch error
(mcp_client.Fatal) keeps the incomplete manifest and the exclusions, then writes fatal.json in the run record and
exits nonzero: the draw stops after infrastructure uncertainty, and no further request is sent.

Every SEC byte comes through the pinned edgar-search-mcp server.
A draw opens its own recording snapshot, draw_<day>_seed<seed>, and fetches its frame afresh; the development run
on the draw records into the same snapshot, which is closed after it. Run with the pinned venv's python.
"""
import argparse
import hashlib
import json
import random
import re
import time

from .edgar import ROOT
from .mcp_client import Fatal, Fetcher, NotRun, open_snapshot
from .runrec import new_run, write

FRAME_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
EXCHANGES = {"NYSE", "Nasdaq"}
EXCLUSIONS = ROOT / "data/manifests/exclusions.json"
MAX_WALK = 200
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def fetch_json(url, keep, fetcher):
    """A JSON document through the server, its bytes kept in the draw's response folder."""
    body, _ = fetcher.document(url)
    (keep / url.rsplit("/", 1)[-1]).write_bytes(body)
    return json.loads(body)


def ten_ks(filings, year=None):
    """Every original 10-K with inline XBRL, latest first by report date then filing date; with `year`, only report
    dates in that year. Rows with a malformed date or a non-boolean inline flag are
    skipped, never guessed."""
    found = []
    for i, form in enumerate(filings.get("form", [])):
        rd, fd = filings["reportDate"][i], filings["filingDate"][i]
        inline = filings["isInlineXBRL"][i]
        if form == "10-K" and isinstance(rd, str) and DATE.fullmatch(rd) and (year is None or rd.startswith(year)) \
                and isinstance(fd, str) and DATE.fullmatch(fd) and inline in (True, 1):
            found.append((rd, fd, filings["accessionNumber"][i]))
    return [{"accession": acc, "report_date": rd, "filed": fd} for rd, fd, acc in sorted(found, reverse=True)]


def latest_10k(filings, year=None):
    """The latest of ten_ks, or None: a company's unit is its latest available filing."""
    found = ten_ks(filings, year)
    return found[0] if found else None


def eligibility(cik, keep, fetcher):
    sub = fetch_json(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json", keep, fetcher)
    sic = int(sub["sic"]) if str(sub.get("sic") or "").isdigit() else None
    k = latest_10k(sub.get("filings", {}).get("recent", {}))
    if k is None and sic and not 6000 <= sic <= 6799:
        for x in sub.get("filings", {}).get("files", []):
            k = latest_10k(fetch_json(f"https://data.sec.gov/submissions/{x['name']}", keep, fetcher))
            if k:
                break
    why = ("no SIC" if sic is None else "financial or real-estate SIC (6000-6799)" if 6000 <= sic <= 6799
           else "no original inline-XBRL 10-K" if k is None else None)
    return {"cik": cik, "name": sub.get("name"), "sic": sic, "sic_description": sub.get("sicDescription"),
            **(k or {})}, why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    day = time.strftime("%Y-%m-%d")
    out = ROOT / f"data/manifests/development_{day}.json"
    keep = ROOT / f"data/manifests/development_{day}_responses"
    if out.exists() or keep.exists():
        raise SystemExit(f"{out} or its responses exist; a manifest is never overwritten")
    snapshot = f"draw_{day}_seed{a.seed}"
    snap = open_snapshot(snapshot, f"development draw {out.name}: frame, submissions, then the run's filings")
    run = new_run("devdraw", {"n": a.n, "seed": a.seed, "snapshot": snapshot, "manifest": str(out)})
    keep.mkdir(parents=True)
    try:
        with Fetcher("record", snap, run / "mcp") as fetcher:
            manifest = walk(a, day, out, keep, fetcher)
    except Fatal as e:          # before the walk began (the frame) or at the server's start: nobody was drawn
        write(run, "fatal.json", {"code": e.code, "error": str(e), "where": "before the walk", "manifest": None})
        raise SystemExit(f"the draw stopped: {e} (record {run})")
    manifest["snapshot"] = snapshot
    manifest["fetch_record"] = str(run.relative_to(ROOT))
    exclusions = json.loads(EXCLUSIONS.read_text(encoding="utf8"))
    excluded = exclusions["companies"]
    # the drawn companies are excluded from every later frame, even when the draw is incomplete
    for r in manifest["development"]:
        excluded[r["cik"]] = {"name": r["name"], "since": day, "why": f"development draw {out.name}"}
    EXCLUSIONS.write_text(json.dumps(exclusions, indent=1), encoding="utf8")
    out.write_text(json.dumps(manifest, indent=1), encoding="utf8")
    print(out, len(manifest["development"]), "kept of", len(manifest["walk"]), "walked",
          "" if manifest["complete"] else f"INCOMPLETE: {manifest['stopped']}")
    if manifest["fatal"]:
        write(run, "fatal.json", {**manifest["fatal"], "where": "during the walk", "manifest": str(out),
                                  "kept": len(manifest["development"])})
        raise SystemExit(f"the draw stopped: {manifest['fatal']['error']} (record {run})")

def walk(a, day, out, keep, fetcher):
    frame = fetch_json(FRAME_URL, keep, fetcher)
    rows = [dict(zip(frame["fields"], r)) for r in frame["data"]]
    ciks = sorted({str(r["cik"]) for r in rows if r["exchange"] in EXCHANGES}, key=int)
    exclusions = json.loads(EXCLUSIONS.read_text(encoding="utf8"))
    excluded = exclusions["companies"]
    order = [c for c in ciks if c not in excluded]
    random.Random(a.seed).shuffle(order)
    walk, kept, stopped, fatal = [], [], None, None
    for c in order:
        if len(kept) == a.n:
            break
        if len(walk) == MAX_WALK:
            stopped = f"walk limit {MAX_WALK} reached: incomplete enumeration"
            break
        try:
            rec, why = eligibility(c, keep, fetcher)
        except (Fatal, NotRun) as e:
            walk.append({"cik": c, "error": str(e), "code": e.code})
            stopped = f"fetch error at CIK {c} ({e.code}); the walk stops (not ineligibility)"
            if isinstance(e, Fatal):
                fatal = {"cik": c, "code": e.code, "error": str(e)}
            break
        except Exception as e:
            walk.append({"cik": c, "error": repr(e)})
            stopped = f"response error at CIK {c}; the walk stops (not ineligibility)"
            break
        walk.append({**rec, "skip": why})
        if why is None:
            kept.append(rec)
    frame_bytes = (keep / FRAME_URL.rsplit("/", 1)[-1]).read_bytes()
    manifest = {"plan": "the development-sample draw", "frame_url": FRAME_URL,
                "frame_sha256": hashlib.sha256(frame_bytes).hexdigest(), "retrieved": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "responses": str(keep.relative_to(ROOT)), "frame_n": len(ciks),
                "excluded_before_shuffle": sorted(excluded, key=int), "seed": a.seed, "n": a.n, "walk": walk,
                "development": kept, "complete": len(kept) == a.n, "stopped": stopped, "fatal": fatal}
    return manifest


if __name__ == "__main__":
    main()
