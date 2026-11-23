"""The live preparation behind the spreading pages:
Setup's Load, Update and Reset, company search and the 10-K list, the filing fetch, the default template, and each
job's settings and manifest. Everything goes through the existing code (spreading/edgar.py, mcp_client.py, devdraw.py,
taxonomy.py, store.py). Nothing here starts a job: ui/jobs.py does, with the spec job_spec makes.

A company is offered only after its industry code is read: banks, insurers and real estate (SIC 6000-6799) are removed
before any list. Setup offers only the approved models of the offered line-ups and the pinned FASB
editions; packages and the EDGAR server are never installed from the page."""
import contextlib
import json
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "ui"), str(ROOT)]
from spreading import devdraw, edgar, fasb_fetch, llm, mcp_client, provenance, result_workbook, store, taxonomy  # noqa: E402
from spreading.spread import load_rows  # noqa: E402
import jobs  # noqa: E402
import setup_status  # noqa: E402

BASE = setup_status.SETTINGS     # the job settings each UI job starts from: one path for Setup, Advanced and jobs
PARKED = {"hosted": "parked; its code is kept",
          "qwen_tree_minicpm_judge_fb_on": "parked: not offered in this configuration "
                                           "of the settings; its code is kept"}
# Models imported into Ollama by hand: Setup never downloads, updates or removes them (no recipe restores them yet).
# MiniCPM's tag is built locally from the publisher's GGUF; these local tags have no pull recipe in Setup.
LOCAL_ONLY = {"ling3-tiny:q4_K_M", "minicpm5:2b-q4_K_M"}
TEMPLATE_NAME = "Standard {}"     # with the template file's version, v8 the standard (or the settings' pin)
FILTER_NOTE = "Banks, insurers and real estate (industry codes 6000-6799) are removed before any list."
NO_REFERENCE = "no AI reference available"
CEILINGS = ("fixed limits on steps and cost, the same for every filing, "
            "not scaled or raised; a larger filing may reach one, and its job then ends with the reason")
CIK = re.compile(r"\d{1,10}")
RESULT_ID = re.compile(r"\d{4}-\d\d-\d\dT\d{6}_result_workbook")
ACCESSION = re.compile(r"\d{10}-\d\d-\d{6}")
MAX_HITS = 10            # companies per search, each with its industry code read
MAX_QUERY = 60
PULL_TIMEOUT = 3600      # seconds for one model download
FASB_TIMEOUT = 900       # seconds for one taxonomy edition


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def financial(sic):
    return sic is not None and 6000 <= sic <= 6799


def finished_jobs(runs):
    """The finished jobs in `runs`, newest first: a job ID folder whose job.json says it finished."""
    for d in sorted(Path(runs).glob("*_full_pipeline_*"), reverse=True):
        j = jobs.read_json(d / "job.json") if jobs.JOB_ID.fullmatch(d.name) else None
        if isinstance(j, dict) and j.get("end") == "finished":
            yield d


def example(runs):
    """The finished example Start here opens, chosen from the records on this computer: the newest finished job
    started from these pages (it has a manifest, not one reconstructed after a job started from the command line), with
    what its manifest says it is; None when there is none."""
    for d in finished_jobs(runs):
        m = jobs.read_json(d / "ui_manifest.json")
        if isinstance(m, dict) and "reconstructed" not in m:
            t = m.get("template") if isinstance(m.get("template"), dict) else {}
            return {"job_id": d.name, "said": f"The newest finished job started from these pages: {m.get('company')}, "
                    f"10-K {m.get('report_date')}, template {t.get('name')}, saved version {t.get('version')}, "
                    f"{m.get('lineup_label') or m.get('lineup')}"}
    return None


class NoFetch:
    """A fetcher for a folder that should already be complete: any request means it is not."""

    class Needed(Exception):
        pass

    def document(self, url):
        raise self.Needed(url)


def lineup_models(lu):
    """The models a line-up names in its own arguments (--third-model, --review-model, --model role=name)."""
    out = set()
    for args in (lu.get("spread_args", []), lu.get("tree_args", [])):
        for i, x in enumerate(args[:-1]):
            if x in ("--third-model", "--review-model"):
                out.add(args[i + 1])
            elif x == "--model":
                out.add(args[i + 1].split("=", 1)[-1])
    return out


class Session:
    """What the server holds for Live preparation: the companies looked up (with their industry code and 10-Ks), the
    background tasks (filing fetches and Setup items, one at a time) and their outcomes. `fetcher` (log folder ->
    a fetcher context), `installed` (model name -> digest), `pull`, `remove` and `command` can be replaced by fakes;
    `cache`, `db`, `fasb_base`, `base` too."""

    def __init__(self, ui_dir, db=None, cache=None, fasb_base=None, base=None, fetcher=None, installed=None,
                 pull=None, remove=None, command=None):
        self.ui_dir, self.db, self.cache = Path(ui_dir), Path(db or store.DB), Path(cache or edgar.CACHE)
        self.fasb_base, self.base = fasb_base, Path(base or BASE)
        self.make_fetcher = fetcher or self.record_fetcher
        self.installed = installed or (lambda: {m["name"]: m["digest"] for m in self.ollama("/api/tags")["models"]})
        self.pull = pull or self.ollama_pull
        self.remove = remove or (lambda name: self.ollama("/api/delete", {"model": name}, method="DELETE"))
        self.command = command or (lambda argv: subprocess.run(argv, cwd=ROOT, stdin=subprocess.DEVNULL,
                                                               capture_output=True, text=True, timeout=FASB_TIMEOUT))
        self.snapshot, self.frame, self.companies, self.tasks = None, None, {}, {}
        self.advisories = None       # the OSV row of the last Check now in this session; None: not checked
        self.key_tested = None       # the last Test the key row, with a digest of the key tested; None: not tested
        self.lock = threading.Lock()

    # ---- records and the EDGAR server ------------------------------------------------------------------------------

    def record_dir(self, what):
        d = self.ui_dir / "edgar" / f"{time.strftime('%Y-%m-%dT%H%M%S')}_{what}_{uuid.uuid4().hex[:6]}"
        d.mkdir(parents=True)
        return d

    def record_fetcher(self, log_dir):
        """The EDGAR server in record mode, into one snapshot per server session, opened at its first use."""
        with self.lock:
            if self.snapshot is None:
                name = "ui_" + time.strftime("%Y%m%dT%H%M%S")
                mcp_client.open_snapshot(name, "the spreading pages' company search and filing fetches")
                self.snapshot = name
        return mcp_client.Fetcher("record", mcp_client.snapshot_dir(self.snapshot, "record"), log_dir)

    @staticmethod
    def ollama(path, body=None, timeout=10, method=None):
        base = llm.OLLAMA.rsplit("/api/", 1)[0]
        req = urllib.request.Request(base + path, None if body is None else json.dumps(body).encode("utf8"),
                                     {"Content-Type": "application/json"}, method=method)
        return json.loads(urllib.request.urlopen(req, timeout=timeout).read() or b"{}")

    def ollama_pull(self, name):
        return self.ollama("/api/pull", {"model": name, "stream": False}, timeout=PULL_TIMEOUT)

    # ---- background tasks ------------------------------------------------------------------------------------------

    def active(self):
        with self.lock:
            return sorted(k for k, t in self.tasks.items() if t["state"] == "running")

    def task(self, key, fn, alone=False):
        """Run fn in the background as task `key`; one at a time per key, and with `alone` only while no task runs (the
        check and the registration under one lock). Its outcome stays on the task and goes to data/ui/tasks.jsonl."""
        with self.lock:
            running = sorted(k for k, t in self.tasks.items() if t["state"] == "running")
            if key in running:
                raise jobs.Busy(f"{key} is already running")
            if alone and running:
                raise jobs.Busy(f"wait for {', '.join(running)} to end")
            t = self.tasks[key] = {"task": key, "state": "running", "started": now()}

        def go():
            try:
                out = {"state": "done", "result": fn()}
            except Exception as e:  # noqa: BLE001  (every outcome is recorded on the task)
                out = {"state": "failed", "problem": f"{type(e).__name__}: {getattr(e, 'msg', e)}"}
            out["ended"] = now()
            with self.lock:
                t.update(out)
            self.ui_dir.mkdir(parents=True, exist_ok=True)
            with open(self.ui_dir / "tasks.jsonl", "a", encoding="utf8") as f:
                f.write(json.dumps(t, default=str) + "\n")

        threading.Thread(target=go, daemon=True).start()
        return dict(t)

    def task_view(self):
        with self.lock:
            return {"tasks": [dict(t) for t in self.tasks.values()]}

    # ---- companies and 10-Ks ---------------------------------------------------------------------------------------

    def search(self, q):
        """Companies whose ticker is q, then whose name contains it, from the SEC's company list; each one's industry
        code and 10-Ks read from its submissions. Financial and real-estate companies are removed and counted."""
        if not isinstance(q, str) or not q.strip() or len(q) > MAX_QUERY:
            raise ValueError(f"a search is 1 to {MAX_QUERY} characters")
        q = q.strip()
        log = self.record_dir("search")
        with self.make_fetcher(log) as f:
            if self.frame is None:
                self.frame = devdraw.fetch_json(devdraw.FRAME_URL, log, f)
            fields, seen, hits = self.frame["fields"], set(), []
            rows = [dict(zip(fields, r)) for r in self.frame["data"]]
            for r in [r for r in rows if str(r.get("ticker") or "").upper() == q.upper()] + \
                     [r for r in rows if q.casefold() in str(r.get("name") or "").casefold()]:
                if r["cik"] not in seen and len(seen) < MAX_HITS:
                    seen.add(r["cik"])
                    hits.append(r)
            out, removed = [], 0
            for r in hits:
                co = self.company(r, log, f)
                if financial(co["sic"]):
                    removed += 1
                else:
                    out.append(co)
        return {"query": q, "companies": out, "removed": removed, "filter": FILTER_NOTE,
                "capped": len(hits) == MAX_HITS, "record": log.name}

    def company(self, row, log, fetcher):
        cik = str(int(row["cik"]))
        sub = devdraw.fetch_json(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json", log, fetcher)
        sic = int(sub["sic"]) if str(sub.get("sic") or "").isdigit() else None
        co = {"cik": cik, "name": sub.get("name") or row.get("name"), "ticker": row.get("ticker"),
              "exchange": row.get("exchange"), "sic": sic, "sic_description": sub.get("sicDescription"),
              "ten_ks": [] if financial(sic) else devdraw.ten_ks(sub.get("filings", {}).get("recent", {}))}
        with self.lock:
            self.companies[cik] = co
        return co

    def known(self, cik, accession):
        """The company looked up in this session and one of its 10-Ks; anything else is refused."""
        if not isinstance(cik, str) or not CIK.fullmatch(cik) or not isinstance(accession, str) \
                or not ACCESSION.fullmatch(accession):
            raise ValueError("name the company by its CIK and the filing by its accession number")
        with self.lock:
            co = self.companies.get(str(int(cik)))
        if co is None:
            raise ValueError(f"CIK {cik}: search for the company first")
        if financial(co["sic"]):
            raise ValueError(f"{co['name']}: SIC {co['sic']}. {FILTER_NOTE}")
        k = next((k for k in co["ten_ks"] if k["accession"] == accession), None)
        if k is None:
            raise ValueError(f"{accession} is not one of {co['name']}'s 10-Ks with inline XBRL")
        return co, k

    # ---- the filing and its taxonomy edition -----------------------------------------------------------------------

    def folder(self, cik, accession):
        return self.cache / str(int(cik)) / accession

    def prepared(self, d):
        """The fetched filing's facts, when its files, main document and statement pages verify, without a request;
        raises NoFetch.Needed when something is missing, edgar's NotRun when something does not verify."""
        edgar.fetch_main_document(d, NoFetch())
        edgar.fetch_r_pages(d, NoFetch())
        from spreading.filing import Filing
        from spreading.spread import taxonomy_year
        f = Filing(str(d))
        year = taxonomy_year(f)
        try:
            taxonomy.edition(year, self.fasb_base)
            have = True
        except taxonomy.MissingEdition:
            have = False
        return {"folder": store.rel(d), "period_end": f.period_end, "taxonomy_year": year, "edition_on_disk": have}

    def fetch(self, cik, accession):
        """Start the preparation of one 10-K: its files, main document and statement pages through the EDGAR server
        (a folder already complete makes no request), then the taxonomy edition it uses when that is pinned and
        missing."""
        self.known(cik, accession)
        d = self.folder(cik, accession)

        def go():
            try:
                facts = self.prepared(d) if d.exists() else None
            except NoFetch.Needed:
                facts = None
            if facts is None:
                with self.make_fetcher(self.record_dir("filing")) as f:
                    if not d.exists():
                        edgar.fetch_filing(cik, accession, f, dest=d)
                    edgar.fetch_main_document(d, f)
                    edgar.fetch_r_pages(d, f)
                facts = self.prepared(d)
            if not facts["edition_on_disk"]:
                loaded = self.fasb(facts["taxonomy_year"])
                facts = {**self.prepared(d), "edition_load": loaded}
            return facts

        return self.task(f"fetch:{int(cik)}/{accession}", go)

    # ---- Setup's Load ----------------------------------------------------------------------------------------------

    def need_settings(self):
        """The settings file, or Missing: the pages say it is not on this computer and offer nothing that needs it."""
        if not self.base.is_file():
            raise jobs.Missing("settings", f"The job settings file {store.rel(self.base)} is not on this computer; the "
                               "line-ups, the models and the limits are read from it, so nothing that needs them is "
                               "offered.",
                               fix=f"The file ships with the project's repository: restore {store.rel(self.base)} "
                                   "from your copy of the repository (for example, git checkout -- "
                                   f"{store.rel(self.base)} in the project folder), then open this page again "
                                   "(on Setup, press Check again).")
        return self.base

    def settings(self):
        return json.loads(self.need_settings().read_text(encoding="utf8"))

    def lineups(self, cfg=None):
        """The line-ups a job may use: the settings' line-ups, less those parked or not built."""
        cfg = cfg or self.settings()
        return {k: v for k, v in cfg["lineups"].items() if k not in PARKED and not v.get("pending")}

    def approved_models(self, lineup=None):
        """The local models of the Setup table and of the offered line-ups: the only models Load offers. With a
        `lineup`, only the models that line-up uses."""
        import setup_status
        names = {r["model"] for r in setup_status.models(self.base, PARKED)["rows"]
                 if r["where"] == "local" and lineup in (None, r["lineup"])}
        for k, lu in self.lineups().items():
            if lineup in (None, k):
                names |= lineup_models(lu)
        return sorted(names)

    def offer(self, lineup=None):
        """What Setup can do now, one item per yes. Load: an approved model not installed, a pinned FASB edition not
        on disk. Update (download again) and Reset (remove): an installed approved model, except one imported locally,
        which is never downloaded: when it is missing, it is named as not restorable here. Reset: an edition's folder on
        disk; one that does not match its pin, or has no archive, says "Reset, then Load". With a `lineup` (Setup's
        selection), only its models; a line-up with no local model asks Ollama nothing. Load, Update and Reset
        themselves (act) check against the offer of every line-up, so the permissions do not move with the selection."""
        out, unavailable = [], []
        approved = self.approved_models(lineup)
        try:
            have = self.installed() if approved else {}
            for m in approved:
                if m not in have and m in LOCAL_ONLY:
                    unavailable.append(f"{m} is not installed; it was imported locally, and this page has no approved "
                                       "way to restore it")
                elif m not in have:
                    out.append({"action": "load", "item": f"model:{m}", "what": f"Download {m} from Ollama's library (its size is not shown here)."})
                elif m not in LOCAL_ONLY:
                    out.append({"action": "update", "item": f"model:{m}", "what": f"Download {m} again from Ollama's "
                                "library (its size is not shown here); if its weights changed, they are not the ones this setup was tested with."})
                    out.append({"action": "reset", "item": f"model:{m}", "what": f"Remove {m} from Ollama on this "
                                "computer, for every project that uses it."})
            models = None
        except Exception as e:  # noqa: BLE001  (Ollama not running: shown, not offered)
            models = f"{type(e).__name__}: {e}"
        on_disk = taxonomy.editions(self.fasb_base)
        for y, p in sorted(fasb_fetch.PINS.items()):
            if not taxonomy.edition_path(y, self.fasb_base).parent.exists():
                out.append({"action": "load", "item": f"fasb:{y}", "what": f"Download the US GAAP taxonomy {y} from "
                            f"xbrl.fasb.org ({p['bytes']:,} bytes; its hash must match the pin)."})
            else:
                z = on_disk.get(y)
                ok = bool(z) and z.stat().st_size == p["bytes"] and fasb_fetch.sha(z) == p["sha256"]
                why = "it does not match its pin" if y in on_disk else "its archive is missing"
                out.append({"action": "reset", "item": f"fasb:{y}", "matches_pin": ok, "what": f"Remove the US GAAP "
                            f"taxonomy {y} from this computer" + ("." if ok else f"; {why}: Reset, then Load.")})
        return {"offer": out, "models_unreadable": models, "unavailable": unavailable,
                "by_a_person": "Python packages and the EDGAR search server are installed by a person against their "
                               "locks; this page never installs them"}

    def act(self, action, item):
        """Load, update or reset one offered item; the person's yes on the page is the approval for it. One Setup task
        at a time, and none while a filing is fetched."""
        offered = {(x["action"], x["item"]) for x in self.offer()["offer"]}
        if (action, item) not in offered:
            raise ValueError(f"{action} {item!r} is not offered; Setup offers {sorted(offered)}")
        kind, name = item.split(":", 1)
        do = {("load", "model"): lambda: self.pull(name), ("load", "fasb"): lambda: self.fasb(name),
              ("update", "model"): lambda: self.update(name), ("reset", "model"): lambda: self.remove(name),
              ("reset", "fasb"): lambda: self.reset_fasb(name)}[action, kind]
        return self.task(f"{action}:{item}", do, alone=True)

    def update(self, name):
        """Download an installed model again: the digest this setup was tested with, before and after."""
        before = self.installed().get(name)
        self.pull(name)
        after = self.installed().get(name)
        job, digests = self.tested_job()
        tested = digests.get(name)
        said = ("unchanged" if after == before else "changed") + (
            "; the tested version is not recorded on this computer" if tested is None else
            "" if after == tested else ": not the version this setup was tested with")
        return {"model": name, "tested": tested, "before": before, "after": after, "said": said,
                **({"tested_in": job} if tested is not None and job else {})}

    @staticmethod
    def tested_job(runs=None):
        """The newest finished job whose first spread recorded local model digests, and those digests: the weights a
        finished job last ran with on this computer (ROOT/runs; nothing named in the code). (None, {}) when none."""
        for d in finished_jobs(runs or ROOT / "runs"):
            for run in sorted(d.glob("children/0*_spread/*_spread/run.json"))[:1]:
                got = (((jobs.read_json(run) or {}).get("hashes") or {}).get("models") or {}).get("digests")
                if got:
                    return d.name, got
        return None, {}

    def reset_fasb(self, year):
        """Remove the folder of one pinned edition, and nothing else: it must resolve directly inside the taxonomy
        folder, so a link to anywhere else is refused."""
        base = Path(self.fasb_base or taxonomy.SOURCES)
        d = taxonomy.edition_path(year, base).parent
        if year not in fasb_fetch.PINS or d.name != f"fasb_us-gaap_{year}":
            raise ValueError(f"us-gaap {year} is not a pinned edition")
        if d.resolve().parent != base.resolve():
            raise ValueError(f"{d} does not resolve inside {base}; nothing removed")
        shutil.rmtree(d)
        return {"edition": year, "removed": store.rel(d)}

    def fasb(self, year):
        if not year:
            raise ValueError("This 10-K names no US GAAP taxonomy edition, so it cannot be spread here.")
        if year not in fasb_fetch.PINS:
            raise ValueError(f"This 10-K uses US GAAP {year}. That edition is not supported. Supported taxonomy "
                             f"editions: {', '.join(sorted(fasb_fetch.PINS))}.")
        exe, *args = taxonomy.FETCH.format(year=year).split()     # Windows finds a relative program only by full path
        exe = jobs.PYTHON if exe.startswith(".venv-agents/") else ROOT / exe    # this system's interpreter path
        r = self.command([str(exe), *args])
        if r.returncode:
            raise RuntimeError(f"the taxonomy fetch ended {r.returncode}: {(r.stderr or r.stdout)[-1500:]}")
        return {"edition": year, "output": (r.stdout or "")[-1500:]}

    # ---- the default template --------------------------------------------------------------------------------------

    def default_template_path(self):
        """The settings' `template` pin (the job pins it too, full_pipeline.prepare), else the standard one. A pin must
        be an approved template (provenance.PINNED)."""
        pin = self.settings().get("template")
        if not pin:
            return provenance.TEMPLATE
        p = next((q for q in provenance.PINNED if q.resolve() == (ROOT / pin).resolve()), None)
        if p is None or p == provenance.SKELETON or not provenance.approved(p):
            raise ValueError(f"the settings' template {pin} is not a pinned template at its path and hash")
        return p

    def template_path(self):
        """The template the UI's jobs spread with: the default, the one template of this version (the
        template is a fixed standard, changed only in code; the Template page shows it, read only)."""
        return self.default_template_path()

    def template_label(self):
        """The one name the Template page, the Run page and the sidebar give the template ("Standard v14"): the name
        "Use default" registers it under."""
        return TEMPLATE_NAME.format(self.default_template_path().stem.rsplit("_", 1)[-1])

    def newest_taxonomy(self):
        """The newest us-gaap edition on disk, read once: the Template page's FASB names come from it."""
        if getattr(self, "_tax", None) is None:
            on_disk = taxonomy.editions(self.fasb_base)
            if not on_disk:
                raise jobs.Missing("fasb", "No us-gaap edition is on this computer; Setup fetches one.")
            self._tax = taxonomy.Taxonomy(on_disk[max(on_disk)])
        return self._tax

    def template_choices(self):
        """For each statement of the default template: the FASB headings a section can take and the printed totals a
        check can compare with, by label, from the newest us-gaap edition on disk; the default's own picks flagged."""
        if getattr(self, "_choices", None) is not None:
            return self._choices
        tax = self.newest_taxonomy()
        year = tax.year
        d = json.loads(self.default_template_path().read_text(encoding="utf8"))
        used_h = {h for s in d["sections"] for h in s.get("fasb_headings") or ()}
        used_t = {c["filed"] for c in d["controls"] if c.get("filed")} | {
            t.get("concept") for c in d["controls"] for t in (c.get("reconstructed") or {}).get("terms") or ()}
        out = {"edition": year, "statements": {}}
        for st, s in d["statements"].items():
            heads, totals = {}, {}
            for v in s["variants"]:
                for _, parent, child, *_ in tax.network(v["fasb_network"], "pre"):
                    for c in (parent, child):
                        x = tax.concepts.get(c) or {}
                        if x.get("abstract") and (x.get("label") or "").endswith("[Abstract]"):
                            heads[c] = x["label"][:-len(" [Abstract]")]
                try:
                    cal = tax.network(v["fasb_network"], "cal")
                except (FileNotFoundError, KeyError):
                    cal = []
                for _, parent, *_ in cal:
                    if tax.is_amount(parent):
                        totals[parent] = tax.concepts[parent].get("label") or parent
            for c in used_t:                       # a default's total outside the network (the pretax variants)
                if c and c not in totals and tax.is_amount(c) and any(
                        x["section"].split(".")[0] == st for x in d["controls"] if c in json.dumps(x)):
                    totals[c] = tax.concepts[c].get("label") or c
            key = lambda kv: (kv[0] not in used_h and kv[0] not in used_t, kv[1].lower())
            out["statements"][st] = {
                "name": s["name"],
                "headings": [{"concept": c, "label": l, "default": c in used_h} for c, l in sorted(heads.items(), key=key)],
                "totals": [{"concept": c, "label": l, "default": c in used_t} for c, l in sorted(totals.items(), key=key)]}
        self._choices = out
        return out

    def need_template(self):
        p = self.template_path()
        if not p.is_file():
            raise jobs.Missing("template", f"The template file {store.rel(p)} is not on this computer.")
        return p

    def need_template_sha(self):
        return store.sha(self.need_template())

    def stored_hashes(self, job_id, version):
        """The file hashes the results database stored when it imported `version` of job `job_id` (spreads.files),
        or None when it holds no such spread."""
        if not self.has_store():
            return None
        with contextlib.closing(self.reading()) as con:
            r = con.execute("SELECT files FROM spreads WHERE job_id = ? AND run_folder = ? ORDER BY imported DESC",
                            (job_id, f"runs/{job_id}/cumulative/{version}")).fetchone()
        try:
            return json.loads(r[0]) if r and r[0] else None
        except ValueError:
            return None

    def reading(self):
        """A read-only connection to the results database. A read never runs store.connect, which creates the file and
        executes the schema (a write: store.connect can migrate an older store); only a person's action (Use default, Import, Make current, an export, a review) writes."""
        return sqlite3.connect(f"{self.db.resolve().as_uri()}?mode=ro", uri=True)

    def has_store(self):
        """Whether the results database exists. A read never creates it: only Use default, Import, Make current, an
        export or a review (each a person's action) does."""
        return self.db.is_file()

    def template(self, register=False):
        """The registered version of the standard template (the file provenance.TEMPLATE at its current fingerprint);
        with `register`, registered first when absent ("Use default"). None when absent and not registering."""
        p = self.need_template()
        if not register and not self.has_store():
            return None
        with contextlib.closing(store.connect(self.db) if register else self.reading()) as con:
            q = ("SELECT v.id, v.template_id, v.version, t.name FROM template_versions v JOIN templates t "
                 "ON t.id = v.template_id WHERE v.path = ? AND v.sha256 = ?")
            rows = con.execute(q, (store.rel(p), store.sha(p))).fetchall()
            if not rows and register:
                try:
                    store.add_template(con, self.template_label(), p)
                except store.Rejected as e:
                    raise ValueError(str(e)) from None
                rows = con.execute(q, (store.rel(p), store.sha(p))).fetchall()
        if len(rows) > 1:
            raise ValueError(f"{len(rows)} registered versions have the standard template's file; none is chosen")
        if not rows:
            return None
        vid, tid, version, name = rows[0]
        return {"name": name, "template_id": tid, "version_id": vid, "version": version, "path": store.rel(p),
                "sha256": store.sha(p)}

    # ---- one job's settings and manifest ---------------------------------------------------------------------------

    def reference(self, grade, cik, accession):
        """The grading reference and frame of the base settings, only when both are for this company and filing and
        were written on the job's template, the settings' pin (else no compatible AI reference; the grader still
        refuses)."""
        if not grade:
            return None
        try:
            ref = json.loads((ROOT / grade["reference"]).read_text(encoding="utf8"))
            frame = json.loads((ROOT / grade["frame"] / "run.json").read_text(encoding="utf8"))
            p, tpl = self.template_path(), ref["template"]["sha256"]
            on_it = (ROOT / ref["template"]["path"]).resolve() == p.resolve() and store.sha(p) == tpl
        except (OSError, ValueError, KeyError, TypeError):
            return None
        same = on_it and str(ref.get("cik")) == str(int(cik)) and ref.get("accession") == accession \
            and Path(str(frame.get("filing"))).parts[-2:] == (str(int(cik)), accession) \
            and (frame.get("template") or {}).get("sha256") == tpl
        return grade if same else None

    def job_spec(self, body):
        """A start request (cik, accession, lineup, dollars, tiebreaker, tree) to the worker's request, the job's
        settings and its manifest, written once by ui/jobs.py. Refused unless the company was looked up, the filing is
        prepared, the template registered, the line-up offered and the dollars within its ceiling (`dollars`: the job's
        dollar ceiling, bounded by the offered line-up's dollars_most; 0 for the shipped line-up, which pays
        nothing, and the Run page offers no box). The tiebreaker and
        the tree are on unless the request says false: the tiebreaker off removes its arguments
        from the job's copies of both argument lists, the tree off sets tree.on false; the base settings stay as
        they are."""
        import setup_status
        from spreading.full_pipeline import checker_off_resolved, parts, review, without_tiebreaker
        on = {x: body.get(x, True) for x in ("tiebreaker", "tree")}
        for x, v in on.items():
            if not isinstance(v, bool):
                raise ValueError(f"{x}: true or false")
        co, k = self.known(body.get("cik"), body.get("accession"))
        cfg = self.settings()
        if not cfg["tree"].get("on", True):     # settings with the tree off (v29): the request records it off
            on["tree"] = False
        lineup, dollars = body.get("lineup"), body.get("dollars", 0)
        offered = self.lineups(cfg)
        if lineup not in offered:
            raise ValueError(f"line-up {lineup!r}: offered {sorted(offered)}")
        lu = offered[lineup]
        if isinstance(dollars, bool) or not isinstance(dollars, (int, float)) \
                or not 0 <= dollars <= lu.get("dollars_most", 0):
            raise ValueError(f"dollars: 0 to {lu.get('dollars_most', 0)} for line-up {lineup}")
        if lu.get("pays") and not setup_status.present_key():      # refused here, not as failed requests
            raise ValueError("these models run on OpenRouter and no OPENROUTER_API_KEY is set, in the server's "
                             "environment or in .env; save the key on the Setup page first")
        d = self.folder(co["cik"], k["accession"])
        try:
            facts = self.prepared(d)
        except (NoFetch.Needed, mcp_client.NotRun, OSError) as e:
            raise ValueError(f"{k['accession']} is not prepared; fetch it first ({type(e).__name__})") from None
        if not facts["edition_on_disk"]:
            raise ValueError(f"us-gaap {facts['taxonomy_year']} is not on disk; load it first")
        t = self.template()
        if t is None:
            raise ValueError("no registered default template; choose \"Use default\" first")
        grade = self.reference(cfg.get("grade"), co["cik"], k["accession"])
        settings = {k2: v for k2, v in cfg.items() if k2 != "grade"}
        settings["template"] = store.rel(self.template_path())     # the chosen template (the settings' pin by default)
        settings["filing"] = facts["folder"]
        if grade:
            settings["grade"] = grade
        if not on["tiebreaker"]:
            settings["spread_args"] = without_tiebreaker(settings["spread_args"])
            settings["lineups"] = {**settings["lineups"], lineup: {
                **lu, "spread_args": without_tiebreaker(lu.get("spread_args", []))}}
        if not on["tree"]:
            settings["tree"] = {**settings["tree"], "on": False}
        settings = checker_off_resolved(settings, lineup)
        manifest = {
            "company": co["name"], "cik": co["cik"], "ticker": co["ticker"], "sic": co["sic"],
            "sic_description": co["sic_description"], "form": "10-K", "accession": k["accession"],
            "report_date": k["report_date"], "filed": k["filed"], "filing": facts["folder"],
            "period": facts["period_end"], "taxonomy_year": facts["taxonomy_year"], "template": t,
            "lineup": lineup, "lineup_label": setup_status.label(lineup), "pays": bool(lu.get("pays")),
            "dollars": dollars, "parts": parts(settings, settings["lineups"][lineup]),
            "review": review(settings, settings["lineups"][lineup]),
            "caps": {"label": CEILINGS, "statements": cfg["statements"], "tree": {
                x: v for x, v in cfg["tree"].items() if x != "args"}, "company": cfg["company_limits"]},
            "grade": grade or NO_REFERENCE,
            "settings_from": {"path": store.rel(self.base), "sha256": store.sha(self.base)}, "written": now()}
        return {"request": {"lineup": lineup, "dollars": dollars, **on}, "settings": settings, "manifest": manifest}

    # ---- results: the store, the current selection and the exports ------------------------------------------------

    def stored(self, runs):
        """The stored spreads with their template, run date, job and whether each is current (the one chosen for its
        template, company, statement and year) and the exports made. A spread whose company row is
        missing is still listed, by CIK. With no results database, none, and `store` False; nothing is created."""
        spreads = []
        if self.has_store():
            with contextlib.closing(self.reading()) as con:
                cols = ("run_id", "company", "cik", "fiscal_year", "statement", "outcome", "job_id",
                        "template_version_id", "imported", "current", "template", "run_date")
                spreads = [{**dict(zip(cols, r)), "current": bool(r[9])} for r in con.execute(
                    "SELECT s.run_id, c.name, f.cik, f.fiscal_year, s.statement, s.outcome, s.job_id, "
                    "s.template_version_id, s.imported, cs.run_id IS NOT NULL, "
                    "COALESCE(t.name || ' v' || tv.version, 'no template recorded'), s.run_date FROM spreads s JOIN "
                    "filings f USING (accession) LEFT JOIN companies c USING (cik) LEFT JOIN template_versions tv ON "
                    "tv.id = s.template_version_id LEFT JOIN templates t ON t.id = tv.template_id LEFT JOIN "
                    "current_selection cs ON cs.run_id = s.run_id ORDER BY f.cik, f.fiscal_year, s.statement, "
                    "s.imported")]
        return {"spreads": spreads, "imports": self.imports(runs),
                "store": self.has_store()}

    def storage(self, runs):
        """File counts and sizes of the fixed roots, each marked present or not on this computer, and how many spread
        versions the results database stores and how many are current (the list itself is on Results), through a
        read-only connection. Creates, initialises and deletes nothing."""
        def size(files):
            files = [p for p in files if p.is_file()]
            return {"files": len(files), "bytes": sum(p.stat().st_size for p in files)}
        fasb = Path(self.fasb_base or taxonomy.SOURCES)
        roots = [("Results database", self.db, size([self.db]), self.db.is_file()),
                 ("Job and step records", runs, size(Path(runs).rglob("*")), Path(runs).is_dir()),
                 ("Downloaded 10-K filings", self.cache, size(self.cache.rglob("*")), self.cache.is_dir()),
                 ("FASB taxonomy editions", fasb / "fasb_us-gaap_*",
                  size(p for d in fasb.glob("fasb_us-gaap_*") for p in d.rglob("*")), any(fasb.glob("fasb_us-gaap_*")))]
        stored = None
        if self.db.is_file():
            with contextlib.closing(self.reading()) as con:
                n, cur = con.execute("SELECT COUNT(*), COUNT(cs.run_id) FROM spreads s LEFT JOIN current_selection cs "
                                     "ON cs.run_id = s.run_id").fetchone()
            stored = {"spreads": n, "current": cur}
        return {"measured": time.strftime("%Y-%m-%d %H:%M:%S"), "stored": stored,
                "roots": [{"root": n, "path": str(p), **s, "present": here} for n, p, s, here in roots]}

    def imports(self, runs):
        """Per finished job, its recorded final statement versions (store.final_versions) against what the store
        holds from it: "imported", "partial" (with the versions missing), "not imported", or "cannot import" and why;
        with `refused`, the statements it refused."""
        held = {}
        if self.has_store():
            with contextlib.closing(self.reading()) as con:
                for job_id, folder in con.execute("SELECT job_id, run_folder FROM spreads WHERE job_id IS NOT NULL"):
                    held.setdefault(job_id, set()).add(folder)
        out = {}
        for d in sorted(Path(runs).glob("*_full_pipeline_*")):
            if not (jobs.JOB_ID.fullmatch(d.name) and (d / "job.json").exists()):
                continue
            try:
                want = store.final_versions(d)
            except store.Rejected as e:
                out[d.name] = {"state": "cannot import", "why": str(e), "missing": []}
                continue
            if not want:
                out[d.name] = {"state": "cannot import", "why": "the job recorded no final statement version",
                               "missing": []}
                continue
            missing = [p.name for p in want if store.rel(p) not in held.get(d.name, set())]
            out[d.name] = {"state": "imported" if not missing else "partial" if len(missing) < len(want)
                           else "not imported", "missing": missing}
        for name, x in out.items():     # its refused statements: never imported as spreads, no amounts
            sts = (jobs.read_json(Path(runs) / name / "job.json") or {}).get("statements") or {}
            refused = [{"statement": n, **s["refused"], "said": store.refusal_said(
                s["refused"]["refused_at"], s["refused"].get("kind"), s["refused"]["reason"])}
                for n, s in sorted(sts.items()) if (s or {}).get("refused")]
            x.update({"refused": refused} if refused else {})
        return out

    def spread(self, run_id):
        """One stored spread as the store keeps it: its rows in template order, each row's sources, the totals, the
        checks, the exceptions and the job's parts as its stored request names them. Refused unless the run ID is
        stored. Each source keeps its fact, the facts it adds and where code found its quote (quote_source); the head
        keeps the filing's accession and the run folder the trace links read from."""
        if not self.has_store():
            raise jobs.Missing("store", "No results database on this computer yet: import a finished job first.")
        with contextlib.closing(self.reading()) as con:
            def rows(sql):
                cur = con.execute(sql, (run_id,))
                return [dict(zip([c[0] for c in cur.description], r)) for r in cur]
            head = rows("SELECT s.run_id, c.name AS company, f.cik, f.fiscal_year, s.statement, s.outcome, s.job_id, "
                        "s.run_folder, s.period_end, s.accession, s.run_date, s.files FROM spreads s JOIN filings f "
                        "USING (accession) LEFT JOIN companies c USING (cik) WHERE s.run_id = ?")
            if not head:
                raise ValueError(f"no stored spread {str(run_id)[:120]!r}")
            return {**head[0], "parts": store.job_parts(con, head[0]["job_id"]), "use": self.use(con, run_id),
                    "rows": rows("SELECT row_code, row_name, section, placed, usable, availability, unit "
                                 "FROM spread_rows WHERE run_id = ? ORDER BY seq"),
                    "sources": rows("SELECT row_code, section, line, line_label, piece, signed, placed_by, unit, "
                                    "review_state, quote, fact, facts, quote_source FROM row_sources WHERE run_id = ? "
                                    "ORDER BY rowid"),
                    "totals": rows("SELECT row_code, value, unavailable, incomplete, label, unit FROM totals "
                                   "WHERE run_id = ? ORDER BY rowid"),
                    "checks": rows("SELECT kind, section, name, state, filed, computed, difference, tolerance, detail "
                                   "FROM checks WHERE run_id = ? ORDER BY rowid"),
                    "exceptions": rows("SELECT step, class, section, concept, why FROM exceptions WHERE run_id = ? "
                                       "ORDER BY rowid")}

    @staticmethod
    def use(con, run_id):
        """What the spread's claim record holds, shown beside it: claim-free, open claims (not settled, marked
        PROVISIONAL in the export, by piece) or not assessable (refused at selection and export). Not a use gate:
        downloading the result supersedes it."""
        ok, st = store.assessable(con, run_id)
        out = {"usable": ok, "status": st["status"], "reason": st["reason"]}
        if store.analyst_version(con, run_id):
            out["analyst"] = True
        if st["status"] != store.OPEN_CLAIMS:
            return out
        pieces = {}
        # names a person reads: the row by its template name, a printed line by its label in this spread
        row_name = dict(con.execute("SELECT row_code, row_name FROM spread_rows WHERE run_id = ?", (run_id,)))
        label = dict(con.execute("SELECT line, line_label FROM row_sources WHERE run_id = ? AND line_label IS NOT NULL",
                                 (run_id,)))
        amount = dict(con.execute("SELECT fact, signed FROM row_sources WHERE run_id = ? AND fact IS NOT NULL", (run_id,)))
        words = lambda c: label.get(c) or re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(c).split(":")[-1]).capitalize()
        for cid, row, line, fact, rival in con.execute("SELECT claim_id, row_code, line, fact, rival FROM "
                                                       "recorded_claims WHERE run_id = ? ORDER BY seq", (run_id,)):
            p = pieces.setdefault(fact, {"fact": fact, "row": row, "line": line, "claims": 0, "rivals": [],
                                         "row_name": row_name.get(row, row), "line_label": words(line),
                                         "amount": amount.get(fact), "rival_labels": []})
            p["claims"] += 1
            p["rivals"].append(rival)
            if words(rival) not in p["rival_labels"]:
                p["rival_labels"].append(words(rival))
        return {**out, "said": store.claims_said(st), "pieces": list(pieces.values())}

    def import_job(self, runs, job_id):
        """A finished job's final statement versions into the store, under a dated load record (store.load_job)."""
        d = jobs.job_folder(runs, job_id)
        if self.template() is None:
            raise ValueError("no registered default template; choose \"Use default\" first")
        with contextlib.closing(store.connect(self.db)) as con:
            return store.load_job(con, self.db, d)

    def select(self, run_id, by):
        if not (isinstance(by, str) and by.strip()):
            raise ValueError("a selection names the person making it")
        with contextlib.closing(store.connect(self.db)) as con:
            try:
                return {"current": run_id, "replaced": store.select(con, str(run_id), by.strip())}
            except store.Rejected as e:
                raise ValueError(str(e)) from None

    def result(self, runs, job_id):
        """The download "Workflow result - unreviewed" of one finished job, made from the job's own records
        (spreading.result_workbook) into a dated record: its ID, the file, whether it read back as its snapshot, and
        the headline and per-statement counts."""
        out, diffs, summary = result_workbook.export(jobs.job_folder(runs, job_id), "ui")
        return {"export": out.parent.name, "file": out.name, "matches": not diffs, **said_of(summary)}

    def results_made(self, runs):
        """The workflow-result records on this computer, newest first: job, export ID, whether it read back, and its
        headline and counts as recorded."""
        out = []
        for d in sorted(Path(runs).glob("*_result_workbook"), reverse=True):
            run, back = jobs.read_json(d / "run.json") or {}, jobs.read_json(d / "read_back.json") or {}
            summary = jobs.read_json(d / "summary.json")
            if RESULT_ID.fullmatch(d.name) and back and isinstance(summary, dict):
                out.append({"export": d.name, "job_id": run.get("job"), "matches": back.get("matches"),
                            **said_of(summary)})
        return out

    def result_file(self, runs, export_id):
        """One workflow-result workbook by its record ID (never a path), only while it still matches the identity
        frozen when it was made (read_back.json's name and fingerprint)."""
        if not (isinstance(export_id, str) and RESULT_ID.fullmatch(export_id)):
            raise ValueError("a workflow result is named by its record ID")
        d = Path(runs) / export_id
        back = jobs.read_json(d / "read_back.json") or {}
        if not back.get("matches") or not back.get("workbook") or not back.get("sha256"):
            raise ValueError(f"{export_id}: no workflow result that read back as its snapshot")
        p = d / back["workbook"]
        if p.name != back["workbook"] or not p.is_file() or store.sha(p) != back["sha256"]:
            raise ValueError(f"{export_id}: the workbook no longer matches the one made; make it again")
        return p


def said_of(summary):
    """The headline and one line per statement, as the workbook's About sheet says them."""
    return {"headline": result_workbook.headline(summary),
            "statements": {st: result_workbook.said(s) for st, s in summary.items() if st != "all"},
            "summary": summary}
