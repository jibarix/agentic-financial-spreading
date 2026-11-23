"""The code identity of a running process (provenance instrumentation). The server records it when it starts and passes its commit in
each job request; the UI's worker records its own before run_job and stops, before any model request, when it does
not match.

A record: the commit (git rev-parse HEAD), git status --short of the tracked files in the source folders, the
untracked files there (recorded, not a mismatch: uncommitted work may sit there, and an untracked file
the process loads is caught as a module), and each loaded module of the project with its file's git blob ID (git
hash-object, so line-ending filters apply as git applies them) beside the blob ID at the commit.

Mismatches: the commit is not the expected one; the tracked source is not clean; a loaded project module is not the
file at the commit (changed, untracked, or loaded from outside the project).

A folder that is not a git checkout (a copy downloaded as an archive)
gets a record of its files instead: the SHA-256 of every file in the source folders and of each loaded module, and a
digest of them all. The server expects that digest ("files:<digest>"), and the worker stops when its source files or a
loaded module are not the ones the server started with. The commit is None and `git` says why there is none.
"""
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

from .runrec import ROOT

SOURCE = ("ui", "spreading", "prompts", "rulebook", "data/manifests", "tests")
PACKAGES = ("spreading", "ui")


def git(*args, root=ROOT):
    r = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, stdin=subprocess.DEVNULL,
                       timeout=60)
    if r.returncode:
        raise RuntimeError(f"git {' '.join(args)}: exit {r.returncode}: {r.stderr.strip()[:300]}")
    return r.stdout


def loaded(root=ROOT, modules=None):
    """The project's loaded modules: name -> path relative to the project (posix), or the absolute path of one
    loaded from outside it."""
    out, root = {}, Path(root).resolve()
    for name, m in sorted((modules if modules is not None else sys.modules).items()):
        f = getattr(m, "__file__", None)
        if not f:
            continue
        p = Path(f).resolve()
        if name.partition(".")[0] not in PACKAGES and not any(p.is_relative_to(root / x) for x in PACKAGES):
            continue
        try:
            out[name] = p.relative_to(root).as_posix()
        except ValueError:
            out[name] = str(p)
    return out


def no_git(root=ROOT):
    """Why `root` cannot be read with git (no git program, or not a checkout), or None when it can."""
    try:
        git("rev-parse", "HEAD", root=root)
    except FileNotFoundError:
        return "no git program on this computer"
    except RuntimeError as e:
        if "not a git repository" in str(e):
            return "this folder is not a git checkout"
        raise
    return None


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def source_files(root):
    """SHA-256 of every file in the source folders, by path relative to `root` (posix); caches left out."""
    root, out = Path(root), {}
    for d in SOURCE:
        for p in sorted((root / d).rglob("*")) if (root / d).is_dir() else ():
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                out[p.relative_to(root).as_posix()] = sha256(p)
    return out


def digest(files):
    return hashlib.sha256("".join(f"{k}\t{v}\n" for k, v in sorted(files.items())).encode()).hexdigest()


def record_files(root, modules, why):
    files = source_files(root)
    mods = loaded(root, modules)
    return {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "pid": os.getpid(), "python": sys.version,
            "executable": sys.executable, "root": str(root), "commit": None, "git": why,
            "files_digest": digest(files), "files": files,
            "modules": {n: {"path": p, "sha256": None if Path(p).is_absolute() else files.get(p)}
                        for n, p in mods.items()}}


def expected(rec):
    """What a worker must match: the commit, or the source files' digest where there is no git."""
    return rec["commit"] if rec.get("commit") else f"files:{rec['files_digest']}"


def record(root=ROOT, modules=None):
    root = Path(root)
    why = no_git(root)
    if why:
        return record_files(root, modules, why)
    commit = git("rev-parse", "HEAD", root=root).strip()
    status = git("status", "--short", "--untracked-files=no", "--", *SOURCE, root=root).splitlines()
    untracked = git("ls-files", "--others", "--exclude-standard", "--", *SOURCE, root=root).splitlines()
    mods = loaded(root, modules)
    inside = sorted({p for p in mods.values() if not Path(p).is_absolute()})
    tree = {}
    for line in git("ls-tree", "-r", commit, "--", *inside, root=root).splitlines() if inside else []:
        meta, path = line.split("\t", 1)
        tree[path] = meta.split()[2]
    disk = dict(zip(inside, git("hash-object", "--", *inside, root=root).split())) if inside else {}
    return {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "pid": os.getpid(), "python": sys.version,
            "executable": sys.executable, "root": str(root), "commit": commit, "status": status,
            "untracked": untracked,
            "modules": {n: {"path": p, "blob": disk.get(p), "at_commit": tree.get(p)} for n, p in mods.items()}}


def mismatches(rec, expect_commit):
    """Why `rec` is not the code of `expect_commit` (a commit, or "files:<digest>" for a folder without git), or []."""
    if str(expect_commit).startswith("files:"):
        return file_mismatches(rec, expect_commit[len("files:"):])
    out = []
    if rec["commit"] != expect_commit:
        out.append(f"commit {rec['commit']}, not the expected {expect_commit}")
    if rec["status"]:
        out.append(f"tracked source is not clean: {rec['status'][:5]}")
    for n, m in rec["modules"].items():
        if m["at_commit"] is None:
            out.append(f"module {n} ({m['path']}) is not a file at the commit")
        elif m["blob"] != m["at_commit"]:
            out.append(f"module {n} ({m['path']}) differs from the commit")
    return out


def file_mismatches(rec, want):
    """Why `rec` is not the source files the server started with (digest `want`), or []."""
    if rec.get("commit") or "files_digest" not in rec:
        return [f"this process reads git (commit {rec.get('commit')}); the server read files without git"]
    out = []
    if rec["files_digest"] != want:
        out.append("the source files changed since the server started")
    for n, m in rec["modules"].items():
        if m["sha256"] is None:
            out.append(f"module {n} ({m['path']}) is not a file in the source folders")
    return out
