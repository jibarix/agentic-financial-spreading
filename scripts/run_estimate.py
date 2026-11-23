"""What a job costs and how long it takes, from the finished jobs in the runs folders, for the Run page's "What to expect".

One command:

    .venv-agents/Scripts/python.exe -s -E scripts/run_estimate.py [--lineup hosted_haiku_luna] [--runs DIR ...]

Reads every runs/*_full_pipeline_* job these pages started (ui_manifest.json) with the line-up, keeps those that
finished (job.json end "finished"), and takes each one's dollars and seconds from its own company ledger (job.json
company.used). Only statements-only jobs count (every spread step's run.json statements_only), as the current
settings run; a job that also reads the notes costs more. A job started from the command
line, whose manifest was written after the run, counts and is labelled reconstructed. Writes data/ui/estimate/<time>_<lineup>.json with each job's row and the range and mean; a file is never
overwritten. --runs may be given more than once (jobs may sit in more than one runs folder); a job ID found twice counts once. The Run page shows the newest file for its line-up. A job that did not finish is listed apart with its end
and is not in the figures. Nothing is sent anywhere.
"""
import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/ui/estimate"


def read(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf8"))
    except (OSError, ValueError):
        return None


def statements_only(d):
    """Every spread step of the job ran statements only, as its run.json records (False when none is recorded)."""
    runs = [read(r) or {} for r in Path(d).glob("children/*_spread/*/run.json")]
    return bool(runs) and all(r.get("statements_only") for r in runs)


def jobs(runs, lineup):
    """(finished rows, other rows) of the line-up's statements-only jobs in one runs folder or several, oldest
    first; a job ID found in two folders counts once (the first folder's)."""
    done, other, seen = [], [], {}
    for r in ([runs] if isinstance(runs, (str, Path)) else runs):
        for d in Path(r).glob("*_full_pipeline_*"):
            seen.setdefault(d.name, d)
    for d in (seen[k] for k in sorted(seen)):
        m = read(d / "ui_manifest.json") or {}
        if m.get("lineup") != lineup or not statements_only(d):
            continue
        job = read(d / "job.json") or {}
        used = (job.get("company") or {}).get("used") or {}
        row = {"job_id": d.name, "company": m.get("company"), "period": m.get("period"), "end": job.get("end"),
               "dollars": used.get("dollars"), "seconds": used.get("wall_s"),
               **({"reconstructed": True} if m.get("reconstructed") else {})}
        ok = job.get("end") == "finished" and all(isinstance(row[k], (int, float)) for k in ("dollars", "seconds"))
        (done if ok else other).append(row)
    return done, other


def summary(rows, key):
    xs = [r[key] for r in rows]
    return {"min": min(xs), "max": max(xs), "mean": statistics.fmean(xs)} if xs else None


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("--lineup", default="hosted_haiku_luna")
    a.add_argument("--runs", action="append", help="a runs folder; may be repeated (default: this copy's runs)")
    a.add_argument("--out", default=str(OUT))
    a = a.parse_args(argv)
    done, other = jobs(a.runs or [str(ROOT / "runs")], a.lineup)
    if not done:
        print(f"no finished job with {a.lineup}; nothing written", file=sys.stderr)
        return 1
    now = datetime.now(timezone.utc)
    rec = {"made": now.isoformat(timespec="seconds"), "lineup": a.lineup, "jobs": len(done),
           "dollars": summary(done, "dollars"), "seconds": summary(done, "seconds"),
           "first": done[0]["job_id"][:10], "last": done[-1]["job_id"][:10],
           "reconstructed": sum(1 for r in done if r.get("reconstructed")), "statements_only": True,
           "finished": done, "not_counted": other,
           "how": "each finished job's own company ledger (job.json company.used: dollars, wall_s); scripts/run_estimate.py"}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{now.strftime('%Y-%m-%dT%H%M%S')}_{a.lineup}.json"
    with open(p, "x", encoding="utf8") as fh:          # never overwritten
        json.dump(rec, fh, indent=1)
    print(f"{p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}: {len(done)} finished jobs, "
          f"${rec['dollars']['min']:.2f} to ${rec['dollars']['max']:.2f}, "
          f"{rec['seconds']['min'] / 60:.0f} to {rec['seconds']['max'] / 60:.0f} min; {len(other)} not counted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
