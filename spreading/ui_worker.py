"""The UI's job worker ("worker lifecycle"): one process
per job, started by ui/server.py with the project's interpreter, a fixed working directory and a fixed argument list:

    .venv-agents/Scripts/python.exe -s -E -m spreading.ui_worker REQUEST_ID

It reads the request the server wrote once (data/ui/requests/<ID>/request.json, settings.json, manifest.json), calls
run_job, and:
- on the job's first event, which names its folder, writes the handshake (request ID, job ID, folder, process ID) to the
  request folder and, with the manifest, into the job folder (ui_handshake.json, ui_manifest.json);
- appends every progress event, with a stable ID (<job ID>#<n>), to progress.jsonl in the job folder;
- turns STOP in the request folder into run_job's stop signal, so a stop pressed before the job's folder exists still
  stops it (once the folder exists, the server writes STOP there, the job's own switch);
- writes worker.json in the request folder when run_job returns or refuses.
When the request names the commit the server runs (expect_commit, written by ui/server.py), the worker records its
own code identity (spreading/identity.py) in identity.json before run_job, and stops there, before any model
request, when it does not match: worker.json then says refused, with the mismatches. The record is copied into the
job folder (ui_identity.json) with the handshake; identity_after.json records the modules loaded by the end of the
job.
Progress is observational only: a callback that fails is dropped and recorded by run_job, and the job's own records
(job.json, exceptions.json) stay its final status.
"""
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

from .runrec import ROOT

REQUEST_ID = re.compile(r"r\d{8}T\d{6}_[0-9a-f]{8}")
POLL_S = 0.5


def ui_dir():
    """data/ui in the project; SPREADING_UI_DIR moves it (the offline tests), as SPREADING_RUNS_DIR moves runs/."""
    return Path(os.environ.get("SPREADING_UI_DIR") or ROOT / "data/ui")


def request_dir(rid):
    if not REQUEST_ID.fullmatch(rid or ""):
        raise ValueError(f"not a request ID: {rid!r}")
    return ui_dir() / "requests" / rid


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def put(p, obj):
    """Written once: an existing file is never replaced."""
    with open(p, "x", encoding="utf8", newline="\n") as f:
        f.write(json.dumps(obj, indent=1, default=str))


class Events:
    """run_job's progress callback: the handshake on the first event, then each event appended as one line."""

    def __init__(self, rid, rq):
        self.rid, self.rq, self.folder, self.n = rid, rq, None, 0

    def __call__(self, event):
        if self.folder is None:
            self.folder = Path(event["folder"])
            hs = {"request_id": self.rid, "job_id": event["job_id"], "folder": str(self.folder), "pid": os.getpid(),
                  "time": now()}
            put(self.rq / "handshake.json", hs)
            put(self.folder / "ui_handshake.json", hs)
            put(self.folder / "ui_manifest.json", json.loads((self.rq / "manifest.json").read_text(encoding="utf8")))
            if (self.rq / "identity.json").exists():
                put(self.folder / "ui_identity.json", json.loads((self.rq / "identity.json").read_text(encoding="utf8")))
        self.n += 1
        rec = {"id": f"{event['job_id']}#{self.n}", "n": self.n, "time": now(), **event}
        with open(self.folder / "progress.jsonl", "a", encoding="utf8", newline="\n") as f:
            f.write(json.dumps(rec, default=str) + "\n")


def watch(rq, stop, done):
    while not done.wait(POLL_S):
        if (rq / "STOP").exists():
            stop.set()
            return


def check_identity(rq, expect, record=None):
    """identity.json in the request folder: this process's code identity against the commit the server runs. Returns
    the mismatches ([] when it matches). `record`: identity.record, or a fake for the offline tests."""
    from . import identity
    try:
        rec = (record or identity.record)()
        bad = identity.mismatches(rec, expect)
    except Exception as e:      # an identity that cannot be read is not a match
        rec, bad = None, [f"identity not readable: {e!r}"]
    put(rq / "identity.json", {"expect_commit": expect, "check": "stopped" if bad else "passed", "mismatches": bad,
                               "record": rec})
    return bad


def main(argv=None, run=None, record=None):
    """`run`: run_job, or a fake for the offline tests; `record`: identity.record, or a fake."""
    rid = (sys.argv[1:] if argv is None else argv)[0]
    rq = request_dir(rid)
    req = json.loads((rq / "request.json").read_text(encoding="utf8"))
    from .full_pipeline import Refused, run_job
    run = run or run_job
    if req.get("expect_commit") is not None:
        bad = check_identity(rq, req["expect_commit"], record)
        if bad:
            put(rq / "worker.json", {"request_id": rid, "end": "refused",
                                     "why": "identity: the worker is not the server's commit, stopped before any "
                                            "request: " + "; ".join(bad[:5]), "time": now()})
            return 2
    stop, done = threading.Event(), threading.Event()
    threading.Thread(target=watch, args=(rq, stop, done), daemon=True).start()
    try:
        r = run(str(rq / "settings.json"), req["lineup"], req["dollars"], stop=stop, progress=Events(rid, rq))
        out = {"end": r["end"], "why": r["why"], "job_id": r["job_id"], "folder": str(r["folder"])}
    except Refused as e:
        out = {"end": "refused", "why": str(e)}
    finally:
        done.set()
        if req.get("expect_commit") is not None:
            try:
                from . import identity
                put(rq / "identity_after.json", (record or identity.record)())
            except Exception as e:
                put(rq / "identity_after.json", {"error": repr(e)})
    put(rq / "worker.json", {"request_id": rid, **out, "time": now()})
    return 2 if out["end"] == "refused" else 0


if __name__ == "__main__":
    sys.exit(main())
