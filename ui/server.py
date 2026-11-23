"""The local server for the spreading pages.

python ui/server.py [--port 8765]

Binds 127.0.0.1 only. Serves a fixed list of files from ui/ and a short list of named endpoints; never a path from the
request. Every action that changes something is a POST that carries the session token made at start, from a page on
this server (Host and Origin checked). Reads name a job by its ID (ui/jobs.py), never by a path, and serve only the
records on a fixed list. Preparation (Setup's Load, Update and Reset, company search, filing fetch, the default
template, each job's settings) is ui/live.py. A Setup action and a job never run at the same time.

At start, the server records its code identity (spreading/identity.py: commit, tracked-source status, loaded modules)
once in data/ui/server_identity/, and each job request it writes names that commit (expect_commit), so the worker
stops before any request when it runs other code.
"""
import argparse
import json
import secrets
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

UI = Path(__file__).resolve().parent
sys.path.insert(0, str(UI))
sys.path.insert(1, str(UI.parent))
import jobs  # noqa: E402
import views  # noqa: E402
from spreading import identity, store  # noqa: E402

STATIC = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/live.js": "live.js",
          "/style.css": "style.css"}
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8"}
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
MAX_BODY = 65536         # bytes; a larger request is not read
READ_TIMEOUT = 10        # seconds a connection may stall while sending a request


class State:
    """What the server holds: the session token, the live job (None when idle), the job that ended last, every action
    dispatched and every worker launched. `admit` is held while a Setup task or a job is checked and reserved, so one is
    never admitted beside the other."""

    def __init__(self):
        self.token, self.job, self.last = secrets.token_urlsafe(32), None, None
        self.dispatched, self.launches = [], []
        self.lock, self.admit = threading.Lock(), threading.Lock()


class Handler(BaseHTTPRequestHandler):
    server_version = "spreading-ui"
    timeout = READ_TIMEOUT   # applied to each connection's socket, so a stalled body cannot hold a thread

    def log_message(self, fmt, *args):     # quiet: the run records are the log
        pass

    def send(self, status, body, ctype="application/json; charset=utf-8", headers=()):
        data = body if isinstance(body, bytes) else json.dumps(body, default=str).encode("utf8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        for k, v in headers:
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "frame-ancestors 'none'")     # no other page may frame this one
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()
        self.wfile.write(data)

    def ours(self):
        """The addresses this server answers to; anything else (a renamed host, another site) is refused."""
        port = self.server.server_address[1]
        return {f"127.0.0.1:{port}", f"localhost:{port}"}

    def host_ok(self):
        return self.headers.get("Host") in self.ours()

    def do_GET(self):
        if not self.host_ok():
            return self.send(403, {"refused": "unknown host"})
        st, (path, _, query) = self.server.state, self.path.partition("?")
        if path in STATIC:
            f = UI / STATIC[path]
            return self.send(200, f.read_bytes(), TYPES[f.suffix]) if f.is_file() else self.send(404, {"refused": "missing"})
        if path == "/api/state":
            import live
            with st.lock:
                out = {"token": st.token, "job": st.job, "last": st.last}
            # the finished example Start here opens: the newest finished job these pages started on this computer,
            # found in runs/ (none named in the code); none when there is none
            ex = live.example(self.server.runs)
            out["example"], out["example_said"] = (ex["job_id"], ex["said"]) if ex else (None, None)
            # the server's identity as recorded at start (record_identity), for the Setup page; not its file list
            idt = getattr(self.server, "identity", None) or {}
            out["identity"] = {k: idt.get(k) for k in ("time", "pid", "commit", "git", "files_digest")} if idt else None
            if idt:
                out["identity"]["port"] = self.server.server_address[1]
            return self.send(200, out)
        read = READS.get(path.removeprefix("/api/"))
        if read is None:
            return self.send(404, {"refused": "not found"})
        q = {k: v[-1] for k, v in parse_qs(query).items()}
        try:
            if read is result_download:     # a workflow result's bytes, by its record ID, after its identity is checked
                f = self.server.live.result_file(self.server.runs, q.get("id"))
                return self.send(200, f.read_bytes(), XLSX, [("Content-Disposition", f'attachment; filename="{f.name}"')])
            return self.send(200, read(self.server, q))
        except jobs.Missing as e:   # a file or folder the read needs is not on this computer: said, nothing fetched
            return self.send(400, {"refused": str(e), "missing": e.what, **({"fix": e.fix} if e.fix else {})})
        except (jobs.BadId, ValueError) as e:
            return self.send(400, {"refused": str(e)})

    def do_POST(self):
        # Read the body before any refusal: answering with it unread can abort the connection on Windows
        # (WinError 10053) before the client sees the refusal. A body too large or a length that is not a number is not
        # read; a body shorter than its stated length or slower than READ_TIMEOUT is incomplete. Each is refused before
        # anything is checked or dispatched, and the connection closes.
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = -1
        try:
            raw = self.rfile.read(n) if 0 <= n <= MAX_BODY else None
        except OSError:          # the read timed out
            raw = None
        if raw is None or len(raw) != n:
            self.close_connection = True
            return self.send(400, {"refused": "the request is incomplete or too large"})
        if not self.host_ok():
            return self.send(403, {"refused": "unknown host"})
        if self.headers.get("Origin") not in {f"http://{h}" for h in self.ours()}:
            return self.send(403, {"refused": "not from this page"})
        st = self.server.state
        if not secrets.compare_digest(self.headers.get("X-Session-Token") or "", st.token):
            return self.send(403, {"refused": "no session token"})
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            body = None
        if not isinstance(body, dict):
            return self.send(400, {"refused": "the body must be a JSON object"})
        action = self.path.split("?", 1)[0].removeprefix("/api/")
        do = ACTIONS.get(action)
        if do is None:
            return self.send(404, {"refused": "not found"})
        with st.lock:
            st.dispatched.append(action)
        try:
            status, out = 200, do(self.server, body)
        except jobs.Busy as e:
            status, out = 409, {"refused": str(e)}
        except jobs.Missing as e:
            status, out = 400, {"refused": str(e), "missing": e.what, **({"fix": e.fix} if e.fix else {})}
        except (jobs.BadId, ValueError) as e:
            status, out = 400, {"refused": str(e)}
        except Exception as e:  # noqa: BLE001  (a fetch or server failure is answered, never a dropped connection)
            status, out = 500, {"refused": f"{action}: {type(e).__name__}: {getattr(e, 'msg', e)}"}
        receipt(self.server, action, status, out)
        return self.send(status, out)


# ---- reads (GET, by ID) and actions (POST) --------------------------------------------------------------------

def setup(srv, q):
    import setup_status
    import live
    base = srv.live.need_settings()   # the one settings path job preparation offers its line-ups from (live.BASE)
    # readiness for the line-up chosen on Setup (`lineup`), else the Run page's default (the first offered); a name
    # not offered is refused. The package advisories are asked of OSV only when a person presses Check now
    # (setup/advisories); until then they are "not checked", never "no advisories"
    chosen = setup_status.selected(base, live.PARKED, q.get("lineup"))
    return {"lineup": chosen, "lineups": {k: setup_status.label(k) for k in setup_status.offered(base, live.PARKED)},
            "items": setup_status.status(base, live.PARKED, chosen, srv.live.key_tested), "models": setup_status.models(base, live.PARKED),
            "load": srv.live.offer(chosen),
            "advisories": srv.live.advisories or {"item": setup_status.ADVISORIES[0], "what": setup_status.ADVISORIES[1],
                                                  "state": "not checked", "detail": "not checked: press Check now"}}


def advanced(srv, q):
    import advanced as view
    import live
    return view.advanced(srv.live.need_settings(), live.PARKED)


def tasks(srv, q):
    return srv.live.task_view()


def template(srv, q):
    """The standard template's registration, the line-ups, and the template itself, read only (this version has one
    template, changed only in code): its sections (headings, total, check), rows (subtotals with
    their formulas) and report checks, with FASB names in place of tags, and `label`, the one name every page gives it."""
    import setup_status
    p = srv.live.need_template()
    out = template_view(srv, json.loads(p.read_text(encoding="utf8")))
    key = bool(setup_status.present_key())     # present or not, never the value (the Run page lists it)
    return {"default": srv.live.template(), "lineups": {k: {"pays": bool(v.get("pays")),
                                                             "dollars_most": v.get("dollars_most", 0),
                                                             "label": setup_status.label(k),
                                                             "key_missing": bool(v.get("pays")) and not key}
                                                         for k, v in srv.live.lineups().items()},
            "label": srv.live.template_label(), "sha256": store.sha(p), **out,
            "settings": srv.live.need_settings().name,    # the settings file every job starts from
            # whether the settings run the tree at all; the Run page offers its switch only when they do
            "tree_on": bool(srv.live.settings()["tree"].get("on", True))}


def template_view(srv, t):
    """A template's sections, rows and report checks for the page, FASB labels in place of tags."""
    try:
        ch = srv.live.template_choices()["statements"]
        names = {x["concept"]: x["label"] for v in ch.values() for k in ("headings", "totals") for x in v[k]}
        tax = srv.live.newest_taxonomy()
        for x in t.get("report_checks") or ():
            for c in [x[k].get("concept") for k in ("left", "right") if isinstance(x[k], dict)] + [
                    c for v in (x.get("alternatives") or {}).values() for c in v]:
                if c and c not in names and c in tax.concepts:
                    names[c] = tax.concepts[c].get("label") or c
    except Exception:           # no edition on disk: tags are shown as they are
        names = {}
    rel = {x["row_id"]: x for x in t["relationships"]}
    ctl = {x["section"]: x for x in t["controls"]}

    rname = {r["row_id"]: r["name"] for r in t["rows"]}

    def one(x):
        if x.get("row"):
            return rname.get(x["row"], x["row"])
        return "printed " + (names.get(x.get("concept")) or str(x.get("concept")))

    def expr(e):
        """A formula in words: rows by name, printed totals by FASB label."""
        if not e:
            return None
        if "terms" not in e:
            return one(e)
        return "".join((("" if i == 0 else " + ") if x["coef"] > 0 else (" − " if i else "−")) + one(x)
                       for i, x in enumerate(e["terms"]))

    def check(c):
        if not c:
            return None
        return {"compare": expr(c["compare"]), "compare_raw": c["compare"], "filed": c.get("filed"),
                "filed_label": names.get(c.get("filed")) or c.get("filed"),
                "reconstructed": expr(c.get("reconstructed")), "alternatives": len(c.get("alternatives") or ())}

    return {"name": t["name"], "version": t["version"],
            "sections": [{"id": s["id"], "statement": s["statement"], "name": s["name"], "kind": s["kind"],
                          "normal_balance": s.get("normal_balance"), "total_row": s.get("output_total_row"),
                          "headings": [{"concept": h, "label": names.get(h) or h} for h in s.get("fasb_headings") or ()],
                          "check": check(ctl.get(s["id"]))}
                         for s in t["sections"]],
            "rows": [{"id": r["row_id"], "section": r["section"], "name": r["name"], "type": r["line_type"],
                      "catch_all": r.get("policy") == "catch_all",
                      "definition": r.get("definition") or "", "belongs": r.get("belongs") or [],
                      "not_belongs": [{"item": x.get("item", ""), "goes_to": x.get("goes_to")}
                                      for x in r.get("not_belongs") or []],
                      "formula": expr(rel[r["row_id"]]) if r["row_id"] in rel else None,
                      "terms": rel[r["row_id"]]["terms"] if r["row_id"] in rel else None} for r in t["rows"]],
            "report_checks": [{"id": x["id"], "name": x["name"], "kind": x["kind"], "left": expr(x["left"]),
                               "right": expr(x["right"]), "left_raw": x["left"], "right_raw": x["right"],
                               "period": x["period"], "note": x.get("note") or "",
                               "alternatives": {k: [{"concept": c, "label": names.get(c) or c} for c in v]
                                                for k, v in (x.get("alternatives") or {}).items()}}
                              for x in t.get("report_checks") or ()],
            # template format v7: the statement checks, the only checks that decide a statement's outcome
            "outcome_checks": t.get("outcome_checks")}


def job_view(srv, q):
    """A job's status and its progress events from line `from` on (complete lines only), by job ID."""
    d = jobs.job_folder(srv.runs, q.get("id"))
    start = int(q.get("from") or 0)
    events = jobs.complete_lines(d / "progress.jsonl", start)
    with srv.state.lock:
        running = bool(srv.state.job) and srv.state.job.get("job_id") == d.name
    return {"job_id": d.name, "running": running, "final": jobs.final_status(d), "from": start,
            "next": start + len(events), "events": events, "manifest": jobs.read_json(d / "ui_manifest.json"),
            "handshake": jobs.read_json(d / "ui_handshake.json")}


def job_progress(srv, q):
    """A job's line progress and model calls per child, from the children's own records (ui/views.py)."""
    return {"job_id": q.get("id"), **views.summary(jobs.job_folder(srv.runs, q.get("id")))}


def job_views(srv, q):
    """A job's tiebreaker lines and trees from its own records; graded only against the reference its manifest names."""
    d = jobs.job_folder(srv.runs, q.get("id"))
    grade = (jobs.read_json(d / "ui_manifest.json") or {}).get("grade")
    ref = jobs.ROOT / grade["reference"] if isinstance(grade, dict) and grade.get("reference") else None
    return views.job_views(d, ref, jobs.ROOT)


def job_replay(srv, q):
    """A job's spreads as recorded, for the Replay page, by job ID."""
    return views.replay(jobs.job_folder(srv.runs, q.get("id")))


def job_call(srv, q):
    """One model call of one of a job's children (corrections, a spread or a repair; its text included), by job ID, child name and number."""
    return views.call(jobs.job_folder(srv.runs, q.get("id")), q.get("child"), q.get("n"))


def job_calls(srv, q):
    """Every model call one child of a job recorded (corrections, a spread or a repair), without its text, by job ID
    and child name; each opens with job/call."""
    return views.call_list(jobs.job_folder(srv.runs, q.get("id")), q.get("child"))


def job_record(srv, q):
    """One record of a job, by job ID and a fixed name only: the job's own (jobs.RECORDS), a statement version's
    (`version`, views.VERSION; jobs.VERSION_RECORDS) or a child run's (`child`, jobs.CHILD; jobs.CHILD_RECORDS). An
    entry may be picked out: `line` (a concept) and `field` of it, `section` (checks), `node` (a tree node with its
    ancestors). Each answer carries the file's sha256 and, for a version, whether it matches the hash the results
    database stored for it. A record that is not on this computer, or an entry it does not hold, is said unavailable."""
    d = jobs.job_folder(srv.runs, q.get("id"))
    return jobs.record(d, q, srv.live.stored_hashes if srv.live.has_store() else None)


def estimate(srv, q):
    """What to expect on Run: the newest stored estimate for the selected line-up
    (scripts/run_estimate.py, data/ui/estimate/), or {"jobs": 0} when none was made."""
    lu = q.get("lineup")
    if lu not in srv.live.lineups():
        raise ValueError(f"not a line-up offered: {str(lu)[:80]!r}")
    files = sorted((srv.runs.parent / "data/ui/estimate").glob(f"*_{lu}.json"))
    if not files:
        return {"lineup": lu, "jobs": 0}
    rec = jobs.read_json(files[-1]) or {}
    return {k: rec.get(k) for k in ("made", "lineup", "jobs", "dollars", "seconds", "first", "last", "reconstructed")} | {"file": files[-1].name}


def job_list(srv, q):
    """The UI's jobs (those with a handshake), newest first, with their outcome from their own records."""
    out = []
    for d in sorted(srv.runs.glob("*_full_pipeline_*"), reverse=True):
        if jobs.JOB_ID.fullmatch(d.name) and (d / "ui_handshake.json").exists():
            out.append({"job_id": d.name, "final": jobs.final_status(d),
                        "running": bool(srv.state.job) and srv.state.job.get("job_id") == d.name,
                        "manifest": jobs.read_json(d / "ui_manifest.json"), "parts": views.parts_of(d)})
    return {"jobs": out}


def results(srv, q):
    return srv.live.stored(srv.runs)


def storage(srv, q):
    return srv.live.storage(srv.runs)


def stored_spread(srv, q):
    """One stored spread (rows, sources, totals, checks, exceptions) by run ID, with its job's recorded AI grade and
    parts: job.json's, else the stored request's when the job folder is gone; and the trace of each source to the
    job's own records (views.trace), each link one the records support, or said unavailable."""
    out = srv.live.spread(q.get("id"))
    try:
        d = jobs.job_folder(srv.runs, out["job_id"]) if out["job_id"] else None
    except jobs.BadId:
        d = None
    out["grade"] = views.grade_counts(d) if d else None
    out["parts"] = (views.parts_of(d) if d else None) or out["parts"]
    out["trace"] = views.trace(d, out, out["sources"])
    return out


def operations(srv, q):
    """The evals page's operations view of one job (ui/views.py), by job ID."""
    return views.operations(jobs.job_folder(srv.runs, q.get("id")))


def result_download(srv, q):
    """Served by do_GET as a workflow result's bytes (Session.result_file)."""
    raise ValueError("served as a file")


def results_made(srv, q):
    return {"results": srv.live.results_made(srv.runs)}


def make_result(srv, body):
    """The download "Workflow result - unreviewed" of one finished job (Session.result)."""
    jobs.job_folder(srv.runs, body.get("job_id"))
    return srv.live.result(srv.runs, body.get("job_id"))


def search(srv, body):
    return srv.live.search(body.get("q"))


def fetch(srv, body):
    return {"task": srv.live.fetch(body.get("cik"), body.get("accession"))}


def setup_item(action):
    """Setup's Load, Update or Reset of one offered item, refused while a job runs (start_job refuses the reverse)."""
    def do(srv, body):
        with srv.state.admit:
            with srv.state.lock:
                if srv.state.job is not None:
                    raise jobs.Busy("a job is running; Setup changes nothing until it ends")
            return {"task": srv.live.act(action, body.get("item"))}
    return do


def check_advisories(srv, body):
    """Setup's Check now: asks OSV for the advisories of the exact package versions the locks pin (the one network
    request Setup makes, and only on this press); the answer is kept with the time it was asked."""
    import setup_status
    srv.live.advisories = setup_status.advisories_item()
    return {"advisories": srv.live.advisories}


def test_key(srv, body):
    """Setup's Test the key: asks OpenRouter's key record whether it accepts the key (no model is called, no cost);
    the answer is kept in memory with a digest of the key tested, never the key, and never written."""
    import setup_status
    srv.live.key_tested = setup_status.key_test()
    return {"key": {k: v for k, v in srv.live.key_tested.items() if k != "digest"}}


def save_key(srv, body):
    """Setup's Save key: the key pasted on the page goes into the project's .env (setup_status.save_key), then it is
    tested as Test the key does. Refused while a job runs, which reads the key at each request. The key is never
    answered, logged or kept by the server."""
    import setup_status
    with srv.state.admit:
        with srv.state.lock:
            if srv.state.job is not None:
                raise jobs.Busy("a job is running; the key is not changed until it ends")
        setup_status.save_key(body.get("key"))
    return test_key(srv, body)


def save_ua(srv, body):
    """Setup's Save on the SEC User-Agent row: the name and email go into the project's .env
    (setup_status.save_ua). Refused while a job runs. The value is never answered or logged."""
    import setup_status
    with srv.state.admit:
        with srv.state.lock:
            if srv.state.job is not None:
                raise jobs.Busy("a job is running; the User-Agent is not changed until it ends")
        setup_status.save_ua(body.get("ua"))
    return {"saved": "EDGAR_USER_AGENT"}


def use_default(srv, body):
    return {"default": srv.live.template(register=True)}


def import_job(srv, body):
    return srv.live.import_job(srv.runs, body.get("job_id"))


def select(srv, body):
    return srv.live.select(body.get("run_id"), body.get("by"))


def start_job(srv, body):
    with srv.state.admit:
        if srv.live.active():
            raise jobs.Busy(f"wait for {', '.join(srv.live.active())} to end before starting a job")
        spec = srv.prepare(body)
        if srv.identity is not None:        # the worker checks itself against the server's commit
            spec = {**spec, "request": {**spec["request"], "expect_commit": identity.expected(srv.identity)}}
        return {"job": srv.runner.start(spec)}


def stop_job(srv, body):
    return {"job": srv.runner.stop(job_id=body.get("job_id"), request_id=body.get("request_id"))}


READS = {"setup": setup, "advanced": advanced, "tasks": tasks, "template": template, "job": job_view, "job/record": job_record,
         "job/progress": job_progress, "job/views": job_views, "job/replay": job_replay, "job/call": job_call, "job/calls": job_calls, "job/operations": operations, "jobs": job_list, "estimate": estimate,
         "results": results, "results/spread": stored_spread, "storage": storage,
         "result/list": results_made, "result/download": result_download}
ACTIONS = {"setup/advisories": check_advisories, "setup/key": test_key, "setup/key/save": save_key, "setup/ua/save": save_ua, "setup/load": setup_item("load"), "setup/update": setup_item("update"), "setup/reset": setup_item("reset"),
           "companies/search": search, "filings/fetch": fetch, "template/default": use_default,
           "store/import": import_job, "store/select": select, "result": make_result,
           "jobs/start": start_job, "jobs/stop": stop_job}


def make(port=8765, ui_dir=None, runs=None, worker=None, prepare=None, live=None, identity=None):
    """The server. `ui_dir` (data/ui), `runs`, `worker` (the worker's command before the request ID), `live` (a
    ui/live.py Session, or its keyword arguments) and `prepare` (a start request's body to its request, settings and
    manifest; by default the session's job_spec) can be replaced by fakes. `identity`: the server's code identity
    record (main records it); with one, each job request names its commit for the worker's check."""
    import live as preparation
    from spreading.runrec import runs_dir
    from spreading.ui_worker import ui_dir as default_ui
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.state = State()
    srv.runs = Path(runs or runs_dir())
    ui = Path(ui_dir or default_ui())
    srv.runner = jobs.Runner(srv.state, ui, srv.runs, worker)
    srv.live = live if isinstance(live, preparation.Session) else preparation.Session(ui, **(live or {}))
    srv.prepare = prepare or srv.live.job_spec
    srv.identity = identity
    return srv


def receipt(srv, action, status, out):
    """A lasting receipt of one action the server dispatched, appended to <ui_dir>/actions.jsonl: the time, the action
    name, the answer's status and, for a refusal, its reason. The request body is never written (it may hold a key).
    A receipt that cannot be written does not change the answer."""
    import time
    rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "action": action, "status": status}
    if status != 200 and isinstance(out, dict):
        rec["refused"] = str(out.get("refused"))[:500]
    try:
        srv.runner.ui.mkdir(parents=True, exist_ok=True)
        with srv.state.lock, open(srv.runner.ui / "actions.jsonl", "a", encoding="utf8", newline="\n") as f:
            f.write(json.dumps(rec) + "\n")
    except (OSError, AttributeError):
        pass


def record_identity(srv, record=None):
    """The server's code identity, recorded once in <ui_dir>/server_identity/ and kept on the server. `record`:
    identity.record, or a fake in its place."""
    import os
    import time
    rec = (record or identity.record)()
    d = srv.runner.ui / "server_identity"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{time.strftime('%Y-%m-%dT%H%M%S')}_{os.getpid()}.json", "x", encoding="utf8", newline="\n") as f:
        f.write(json.dumps({"port": srv.server_address[1], **rec}, indent=1))
    srv.identity = rec
    return rec


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args(argv)
    srv = make(a.port)
    record_identity(srv)
    print(f"Spreading: open http://127.0.0.1:{srv.server_address[1]}/  (Ctrl+C stops the server)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
