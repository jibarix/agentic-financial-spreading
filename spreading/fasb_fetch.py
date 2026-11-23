"""Fetch and check a FASB US GAAP taxonomy package, or record the provenance of one already on disk.

    .venv-agents/Scripts/python.exe -s -E -m spreading.fasb_fetch fetch 2024
    .venv-agents/Scripts/python.exe -s -E -m spreading.fasb_fetch record 2025

`fetch` downloads the pinned package from xbrl.fasb.org only (no redirect followed) into a new folder
data/sources/fasb_us-gaap_<year>/, checks it, writes provenance.json and a dated run record; an existing folder is
never overwritten, and a package that fails a check is not published. `record` checks a package already on disk and
writes its provenance.json with the original download time "unknown". Every end of either, a refusal or a failure
included, writes outcome.json in the run record, and a failure exits nonzero. The package is read with zipfile only;
nothing in it is executed. This is the one module besides llm.py allowed a network import (the import
guard checks it): SEC bytes still come only through the pinned MCP server.

Still needs a person: a new edition needs its whole-archive hash pinned in PINS after it is verified (FASB publishes
no checksum).
"""
import argparse
import hashlib
import json
import os
import shutil
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from .llm import STOP_FILE
from .runrec import ROOT, new_run, write
from .taxonomy import Taxonomy, edition_path

HOST = "https://xbrl.fasb.org/"
# whole-archive pins, each computed from the official zip (FASB publishes none). 2025: a file already on disk, found
# equal to the official zip. 2022, 2023, 2026: official zips fetched once (HTTP 200, complete, integrity and
# reader checks passed) and pinned after its results were reviewed.
PINS = {
    "2022": {"url": "https://xbrl.fasb.org/us-gaap/2022/us-gaap-2022.zip", "bytes": 6485888,
             "sha256": "8af6fbbcec3818cd372f391cece9e62018e115b7b65ea75a6a38b9709af3ab37"},
    "2023": {"url": "https://xbrl.fasb.org/us-gaap/2023/us-gaap-2023.zip", "bytes": 6607657,
             "sha256": "b48fbb7be5cbaef5532ebff51394176ca6e7241ffce5fc88812485b8e6f9d6fd"},
    "2024": {"url": "https://xbrl.fasb.org/us-gaap/2024/us-gaap-2024.zip", "bytes": 7115653,
             "sha256": "decdd417d86ff7bfb5ca166c0ca1001017aea873673544a8d7f91c34bf5d82df"},
    "2025": {"url": "https://xbrl.fasb.org/us-gaap/2025/us-gaap-2025.zip", "bytes": 7101405,
             "sha256": "a3b835925ad74030eb5be865a26d7dfe44013081c4ab7204b6122316a685fff4"},
    "2026": {"url": "https://xbrl.fasb.org/us-gaap/2026/us-gaap-2026.zip", "bytes": 7387980,
             "sha256": "f4c8b8b5697ba7d825a8614b159611cd25a46640e98a9737cda1e4a672bd4c81"},
}
TIMEOUT = 120


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p):
    """A path relative to the project when it lies inside it (runs/ can be moved: runrec.runs_dir)."""
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def check(path, year):
    """Every check on a package, each with its result; a later check is skipped once the zip cannot be read."""
    pin, n, h = PINS[year], os.path.getsize(path), sha(path)
    out = [{"check": "bytes", "ok": n == pin["bytes"], "detail": f"{n} (pinned {pin['bytes']})"},
           {"check": "sha256", "ok": h == pin["sha256"], "detail": f"{h} (pinned {pin['sha256']})"}]
    try:
        z = zipfile.ZipFile(path)
    except zipfile.BadZipFile as e:
        return out + [{"check": "zip opens", "ok": False, "detail": str(e)}]
    with z:
        bad = z.testzip()
        out.append({"check": "member CRCs", "ok": bad is None, "detail": bad or f"{len(z.namelist())} members"})
        tops = sorted({n.split("/", 1)[0] for n in z.namelist()})
        out.append({"check": "one tree", "ok": tops == [f"us-gaap-{year}"], "detail": tops})
        need = [f"us-gaap-{year}/elts/us-gaap-{year}.xsd", f"us-gaap-{year}/elts/us-gaap-lab-{year}.xml",
                f"us-gaap-{year}/elts/us-gaap-doc-{year}.xml"]
        missing = [n for n in need if n not in z.namelist()]
        out.append({"check": "members read by Taxonomy", "ok": not missing, "detail": missing or need})
    if all(c["ok"] for c in out):
        try:
            t = Taxonomy(path)
            t.zip.close()          # release the file before the folder is renamed (Windows)
            n = sum(1 for c in t.concepts.values() if c.get("label"))
            out.append({"check": "parses", "ok": t.year == year and n > 0,
                        "detail": f"edition {t.year}, {len(t.concepts)} concepts, {n} with a standard label"})
        except Exception as e:
            out.append({"check": "parses", "ok": False, "detail": repr(e)})
    return out


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, f"redirect to {newurl} refused", headers, fp)


def download(url, dest):
    if not url.startswith(HOST):
        raise SystemExit(f"{url} is not on {HOST}")
    opener = urllib.request.build_opener(NoRedirect)
    req = urllib.request.Request(url, headers={"User-Agent": "agentic-financial-spreading taxonomy fetch"})
    with opener.open(req, timeout=TIMEOUT) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
        return {"status": r.status, "content_type": r.headers.get("Content-Type"),
                "last_modified": r.headers.get("Last-Modified")}


def ended(run, outcome, stage, e=None, partial=None, published=None):
    """The terminal record of every fetch or record, a refusal or a failure included: its outcome, the stage it reached, the error, any partial folder and what was published."""
    rec = {"outcome": outcome, "stage": stage, "published": str(published) if published else None,
           "error": None if e is None else str(e) if isinstance(e, SystemExit) else repr(e),
           "partial": str(partial) if partial is not None and partial.exists() else None}
    write(run, "outcome.json", rec)
    return rec


def fetch(year, base=None, get=download):
    run = new_run("fasb_fetch", {"year": year, "url": PINS.get(year, {}).get("url")})
    stage, part = "preconditions", None
    try:
        if year not in PINS:
            raise SystemExit(f"us-gaap {year} has no pinned hash; pin it in PINS after verifying it")
        if STOP_FILE.exists():
            raise SystemExit("stop switch set; nothing fetched")
        target = edition_path(year, base).parent
        if target.exists():
            raise SystemExit(f"{target} exists; a source folder is never overwritten")
        part = target.with_name(target.name + ".part")
        if part.exists():
            shutil.rmtree(part)        # this command's own unpublished leftover from a stopped fetch
        part.mkdir(parents=True)
        zp = part / f"us-gaap-{year}.zip"
        stage, started = "download", now()
        response = get(PINS[year]["url"], zp)
        stage = "check"
        checks = check(zp, year)
        prov = {"edition": year, "url": PINS[year]["url"], "retrieved": started, "response": response,
                "bytes": os.path.getsize(zp), "sha256": sha(zp), "pin": PINS[year], "checks": checks, "run": rel(run)}
        write(run, "provenance.json", prov)
        if not all(c["ok"] for c in checks):
            raise SystemExit(f"us-gaap {year} failed its checks; not published (left in {part})")
        stage = "publish"
        (part / "provenance.json").write_text(json.dumps(prov, indent=1), encoding="utf8")
        part.rename(target)
    except BaseException as e:      # a refusal, a network or check failure, the stop switch: all recorded
        ended(run, "not published", stage, e, part)
        raise SystemExit(f"us-gaap {year} not published at {stage}: "
                         f"{e if isinstance(e, SystemExit) else repr(e)} (record {run})") from e
    ended(run, "published", stage, published=target)
    return target, prov


def record(year, base=None):
    """Provenance for a package already on disk: verified now, original download time unknown."""
    run = new_run("fasb_record", {"year": year})
    stage = "preconditions"
    try:
        if year not in PINS:
            raise SystemExit(f"us-gaap {year} has no pinned hash")
        zp = edition_path(year, base)
        out = zp.parent / "provenance.json"
        if out.exists():
            raise SystemExit(f"{out} exists; provenance is never overwritten")
        if not zp.is_file():
            raise SystemExit(f"{zp} is not on disk")
        stage = "check"
        checks = check(zp, year)
        prov = {"edition": year, "url": PINS[year]["url"], "retrieved": "unknown",
                "file_modified": datetime.fromtimestamp(zp.stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "file_modified_note": "the file's modified time on disk, not its download time",
                "verified": now(), "bytes": os.path.getsize(zp), "sha256": sha(zp), "pin": PINS[year],
                "pin_source": "equal to the official zip, checked when the pin was made",
                "checks": checks, "run": rel(run)}
        write(run, "provenance.json", prov)
        if not all(c["ok"] for c in checks):
            raise SystemExit(f"us-gaap {year} on disk failed its checks; no provenance written")
        stage = "write provenance"
        out.write_text(json.dumps(prov, indent=1), encoding="utf8")
    except BaseException as e:
        ended(run, "not recorded", stage, e)
        raise SystemExit(f"us-gaap {year} provenance not recorded at {stage}: "
                         f"{e if isinstance(e, SystemExit) else repr(e)} (record {run})") from e
    ended(run, "recorded", stage, published=out)
    return out, prov


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("fetch", "record"))
    ap.add_argument("year")
    a = ap.parse_args(argv)
    path, prov = (fetch if a.action == "fetch" else record)(a.year)
    print(path)
    for c in prov["checks"]:
        print(f"  {'ok ' if c['ok'] else 'BAD'} {c['check']}: {c['detail']}")


if __name__ == "__main__":
    main()
