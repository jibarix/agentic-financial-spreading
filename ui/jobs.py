"""The server's side of a live job ("worker lifecycle"):
one job at a time across the server, the worker started with a fixed interpreter, working directory and argument list,
the handshake read back, a server log of a worker that fails to start, exits or dies, and the final status read from
the job's own records. Jobs are named by their IDs; a path is never accepted.

A request is written once in data/ui/requests/<request ID>/ (request.json, settings.json, manifest.json) before the
worker starts; the worker writes handshake.json and worker.json beside them and its output goes to worker.log. The
lock (data/ui/job.lock) holds the active request until its worker has ended; a lock left by an earlier server is
watched until that request's worker.json or its job's job.json appears.
"""
import hashlib
import json
import os
import re
import secrets
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The agents' interpreter: Windows keeps it under Scripts, other systems under bin.
PYTHON = ROOT / (".venv-agents/Scripts/python.exe" if os.name == "nt" else ".venv-agents/bin/python")
WORKER = [str(PYTHON), "-s", "-E", "-m", "spreading.ui_worker"]       # + the request ID, nothing else
JOB_ID = re.compile(r"\d{4}-\d\d-\d\dT\d{6}_full_pipeline_[a-z0-9_]+")
# The records a page may read from a job's own folder, by name; nothing else is served.
RECORDS = ("run.json", "job.json", "exceptions.json", "stop.json", "order.json", "chains.json", "ui_manifest.json",
           "ui_handshake.json")
# a statement version's records (runs/<job>/cumulative/<ST_vN>/) and a child run's (children/<NN_step>/<ts>_<kind>/)
# the pages may open, by fixed name only; never a path from the request
VERSION = re.compile(r"(IS|BS|CF)_v\d{1,3}")
CHILD = re.compile(r"\d\d_[a-z]+/\d{4}-\d\d-\d\dT\d{6}_(spread|repair)")
VERSION_RECORDS = ("run.json", "lines.json", "c4.json", "checks.json", "cumulative.json", "outcome.json",
                   "exceptions.json", "claims.json")
CHILD_RECORDS = ("run.json", "c7_tree.json", "c8_review.json", "outcome.json", "exceptions.json", "lines.json",
                 "c4.json", "checks.json")
FIELDS = ("c4", "c4_first", "vote", "haiku", "review", "consistency", "feedback")   # entries of a c4.json line
MAX_RECORD = 8 * 1024 * 1024   # bytes; a larger record is not read
POLL_S = 0.5
TAIL = 3000              # characters of the worker's output kept in the server log


class BadId(ValueError):
    pass


class Missing(ValueError):
    """A file or folder the read needs is not on this computer; `what` names it for the page, `fix` (optional) says
    how a person restores it."""

    def __init__(self, what, said, fix=None):
        super().__init__(said)
        self.what, self.fix = what, fix


class Busy(RuntimeError):
    pass


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def read_json(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf8"))
    except (OSError, ValueError):
        return None


def complete_lines(p, start=0):
    """The JSON lines of `p` from line `start` on, complete lines only: a line still being written (no newline yet) is
    left for the next read. A complete line that does not parse is returned as {"unreadable": text}."""
    try:
        data = Path(p).read_bytes()
    except OSError:
        return []
    lines = data.split(b"\n")[:-1]       # the part after the last newline is incomplete (or empty)
    out = []
    for raw in lines[start:]:
        try:
            out.append(json.loads(raw))
        except ValueError:
            out.append({"unreadable": raw.decode("utf8", "replace")[:500]})
    return out


def job_folder(runs, job_id):
    """A job's folder from its ID; anything that is not a job ID (a path, a parent reference) is refused."""
    if not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id):
        raise BadId(f"not a job ID: {str(job_id)[:80]!r}")
    d = Path(runs) / job_id
    if not d.is_dir():
        raise BadId(f"no such job: {job_id}")
    return d


def final_status(d):
    """The job's outcome from its own records (job.json, exceptions.json); None while neither is written."""
    job, exc = read_json(d / "job.json"), read_json(d / "exceptions.json")
    if job is None and exc is None:
        return None
    return {"end": (job or exc).get("end"), "why": (job or {}).get("error"),
            "exceptions": (exc or {}).get("exceptions"), "from": "job.json" if job else "exceptions.json",
            "progress_callback": (job or {}).get("progress_callback"),
            # Each statement's outcome, and each section whose repair was not attempted, with the reason
            "statements": {st: s.get("outcome") for st, s in ((job or {}).get("statements") or {}).items()},
            "skipped": [{"statement": x.get("statement"), "section": x.get("section"),
                         "why": str(x.get("why") or "").removeprefix(f"{x.get('section')}: ")}
                        for x in (job or {}).get("skipped") or ()]}


class Runner:
    """Starts and watches the one live job. `state` is the server's State: its `job` is the active job (None when idle)
    and `last` the job that ended last; `launches` counts every worker started."""

    def __init__(self, state, ui_dir, runs, worker=None):
        self.st, self.ui, self.runs, self.worker = state, Path(ui_dir), Path(runs), worker or WORKER
        self.lockfile = self.ui / "job.lock"
        left = read_json(self.lockfile)
        if left:      # an earlier server's job: watched until its records end it
            self.st.job = {"job_id": None, "folder": None, **left, "state": "left by an earlier server"}
            self.log({"event": "lock left by an earlier server", **left})
            threading.Thread(target=self.watch, args=(None, self.st.job), daemon=True).start()

    def log(self, rec):
        self.ui.mkdir(parents=True, exist_ok=True)
        with open(self.ui / "server_log.jsonl", "a", encoding="utf8", newline="\n") as f:
            f.write(json.dumps({"time": now(), **rec}, default=str) + "\n")

    def start(self, spec):
        """Writes the request once and starts its worker; raises Busy while a job is active. Returns the job record."""
        with self.st.lock:
            if self.st.job is not None:
                raise Busy("a job is already running; one job at a time")
            rid = f"r{time.strftime('%Y%m%dT%H%M%S')}_{secrets.token_hex(4)}"
            rq = self.ui / "requests" / rid
            rq.mkdir(parents=True)
            for name in ("request", "settings", "manifest"):
                (rq / f"{name}.json").write_text(json.dumps(spec[name], indent=1), encoding="utf8")
            job = {"request_id": rid, "job_id": None, "folder": None, "state": "starting", "started": now(),
                   "manifest": spec["manifest"]}
            with open(self.lockfile, "x", encoding="utf8") as f:
                f.write(json.dumps({"request_id": rid, "started": job["started"]}))
            env = {**os.environ, "SPREADING_UI_DIR": str(self.ui), "SPREADING_RUNS_DIR": str(self.runs)}
            try:
                with open(rq / "worker.log", "wb") as out:
                    proc = subprocess.Popen([*self.worker, rid], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                            stdout=out, stderr=subprocess.STDOUT)
            except OSError as e:
                self.lockfile.unlink()
                self.log({"event": "worker did not start", "request_id": rid, "error": repr(e)})
                raise
            self.st.launches.append(rid)
            job["pid"] = proc.pid
            self.st.job = job
        threading.Thread(target=self.watch, args=(proc, job), daemon=True).start()
        return job

    def handshake(self, job):
        hs = read_json(self.ui / "requests" / job["request_id"] / "handshake.json")
        if hs and job["job_id"] is None:
            with self.st.lock:
                job.update(job_id=hs["job_id"], folder=hs["folder"], state="running")
            self.log({"event": "handshake", **hs})

    def watch(self, proc, job):
        """Until the worker ends: the handshake read back; then its exit, its death, or a start that failed, logged,
        the final status read from the job's records, and the lock released."""
        rq = self.ui / "requests" / job["request_id"]
        while (proc.poll() is None) if proc else not ((rq / "worker.json").exists() or self.ended(job)):
            self.handshake(job)
            time.sleep(POLL_S)
        self.handshake(job)
        worker = read_json(rq / "worker.json")
        final = final_status(Path(job["folder"])) if job["folder"] else None
        code = proc.returncode if proc else None
        if worker is None and job["folder"] is None:
            event = "worker failed before the job started"
        elif worker is None:
            event = "worker died during the job"
        elif worker["end"] == "refused":
            event = "the job's settings were refused"
        else:
            event = "worker exited"
        try:
            tail = (rq / "worker.log").read_bytes()[-TAIL:].decode("utf8", "replace")
        except OSError:
            tail = None
        self.log({"event": event, "request_id": job["request_id"], "job_id": job["job_id"], "exit_status": code,
                  "worker": worker, "final": final, **({"output_tail": tail} if event != "worker exited" else {})})
        with self.st.lock:
            self.st.last = {**job, "state": "ended", "event": event, "exit_status": code, "worker": worker,
                            "final": final, "ended": now()}
            self.st.job = None
            self.lockfile.unlink(missing_ok=True)

    def ended(self, job):
        return bool(job["folder"]) and final_status(Path(job["folder"])) is not None

    def stop(self, job_id=None, request_id=None):
        """STOP in the active job's folder (its own switch), or, before the handshake, in its request folder, which
        the worker turns into the job's stop signal. Only the active job can be stopped."""
        with self.st.lock:
            job = self.st.job
            if job is None:
                raise BadId("no job is running")
            if job_id is not None:
                if job_id != job["job_id"]:
                    job_folder(self.runs, job_id)          # a path or an unknown ID is refused as such
                    raise BadId(f"{job_id} is not the running job")
                where = Path(job["folder"])
            elif request_id is not None and request_id == job["request_id"]:
                where = self.ui / "requests" / request_id
            else:
                raise BadId("name the running job by its job ID (or its request ID before it has one)")
            (where / "STOP").write_text(f"stopped from the UI at {now()}\n", encoding="utf8")
            job["stop_requested"] = now()
            return job


def record(d, q, stored_hashes=None):
    """One record of job folder `d`, by fixed name: the job's own (RECORDS), a version's (q["version"]) or a child
    run's (q["child"]); optionally one entry of it (q line/field, section or node). The answer carries the file's
    sha256 and, for a version when `stored_hashes(job, version)` gives them, the hash the results database stored and
    whether they match. Anything absent is said as unavailable; nothing outside `d` is read."""
    version, child, name = q.get("version"), q.get("child"), q.get("name")
    if version is not None and child is not None:
        raise BadId("name a version or a child run, not both")
    if version is not None:
        if not isinstance(version, str) or not VERSION.fullmatch(version):
            raise BadId(f"not a statement version: {str(version)[:40]!r}")
        allowed, where = VERSION_RECORDS, Path(d) / "cumulative" / version
    elif child is not None:
        if not isinstance(child, str) or not CHILD.fullmatch(child):
            raise BadId(f"not a child run: {str(child)[:80]!r}")
        allowed, where = CHILD_RECORDS, Path(d) / "children" / child
    else:
        allowed, where = RECORDS, Path(d)
    if name not in allowed:
        raise BadId(f"records served here: {list(allowed)}")
    out = {"job_id": Path(d).name, "name": name, **({"version": version} if version else {}),
           **({"child": child} if child else {})}
    f = where / name
    if not f.is_file():
        return {**out, "record": None, "unavailable": f"{name} is not on this computer for this "
                f"{'version' if version else 'run' if child else 'job'}"}
    if f.stat().st_size > MAX_RECORD:
        return {**out, "record": None, "unavailable": f"{name} is larger than {MAX_RECORD} bytes; not served"}
    raw = f.read_bytes()
    out["sha256"] = hashlib.sha256(raw).hexdigest()
    if version and stored_hashes is not None:
        stored = (stored_hashes(Path(d).name, version) or {}).get(name)
        out["stored_sha256"] = stored
        out["matches_store"] = None if stored is None else stored == out["sha256"]
    try:
        rec = json.loads(raw.decode("utf8"))
    except ValueError:
        return {**out, "record": None, "unavailable": f"{name} does not parse as JSON"}
    line, field, section, node = q.get("line"), q.get("field"), q.get("section"), q.get("node")
    if field is not None and field not in FIELDS:
        raise BadId(f"fields served: {list(FIELDS)}")
    if line is not None:
        if isinstance(rec, dict):
            got = rec.get(line)
        elif isinstance(rec, list):
            got = [x for x in rec if isinstance(x, dict) and x.get("concept") == line] or None
        else:
            got = None
        if got is not None and field is not None:
            got = got.get(field) if isinstance(got, dict) else None
        out.update(line=line, **({"field": field} if field else {}))
        if got is None:
            return {**out, "record": None, "unavailable": f"{name} holds no {field + ' for ' if field else ''}{line}"}
        rec = got
    if section is not None:
        got = [x for x in rec if isinstance(x, dict) and x.get("section") == section] if isinstance(rec, list) else []
        out["section"] = section
        if not got:
            return {**out, "record": None, "unavailable": f"{name} holds nothing for section {section}"}
        rec = got
    if node is not None:
        nodes = {n.get("id"): n for n in (rec.get("nodes") or []) if isinstance(n, dict)} if isinstance(rec, dict) else {}
        path, at = [], node
        while at in nodes and len(path) < 1000:
            path.append(nodes[at])
            at = nodes[at].get("parent")
        out["node"] = node
        if not path:
            return {**out, "record": None, "unavailable": f"{name} holds no node {node}"}
        rec = {"node": path[0], "ancestors": path[1:], "chosen": rec.get("chosen"), "committed": rec.get("committed")}
    return {**out, "record": rec}
