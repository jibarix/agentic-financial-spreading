"""C1 transport: every SEC byte the workflow reads comes through the pinned edgar-search-mcp server, as an MCP client
over stdio. This is the only module that talks to the SEC; it runs in the pinned venv, whose python also launches the
server.

The pin, check, open and close commands are in devtools/mcp_pin.py. A Fatal stops the whole command
and no further request is sent; a NotRun is one company's recorded outcome. The limiter is never reset here.
"""
import base64
import hashlib
import json
import logging
import os
import random
import re
import stat
import subprocess
import sys
import tempfile
import time
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .llm import STOP_FILE, env_file_value

ROOT = Path(__file__).resolve().parent.parent
PIN = ROOT / "mcp_pin.json"
AGENTS_ENV = ROOT / "agents_env.json"          # the client's own environment contract
RECORDINGS = ROOT / "data/recordings"
TOOL = "get_filing_document"
CONTRACT = "1.0"
PAGE = 1_000_000                # bytes per page, the tool's default
START_TIMEOUT = 60              # seconds for the launch handshake and for the listing
PAGE_TIMEOUT = 150              # seconds per page: an SEC pause (<= 60 s) + the lock (<= 5 s) + the fetch
COMPANY_TIMEOUT = 600           # seconds for all of one company's requests
RETRY_DELAYS = (2, 5)           # RATE_LIMITER_UNAVAILABLE only, each plus up to JITTER s; then the run stops
JITTER = 1.0
MAX_LIST_PAGES = 20
RESET = 'python -m edgar_mcp.ratelimit --reset --reason "<why>"'

# What each error code of the contract becomes. Any other code is a breaking change and stops the run.
NOT_RUN_CODES = {"NOT_FOUND", "NO_FROZEN_COPY", "INVALID_INPUT", "UNSUPPORTED_DOCUMENT"}
FATAL_CODES = {"RATE_LIMITED", "UPSTREAM_UNAVAILABLE", "NO_USER_AGENT", "AMBIGUOUS_COMPANY"}
RETRY_CODE = "RATE_LIMITER_UNAVAILABLE"
ERROR = re.compile(r"Error executing tool (?P<tool>\S+): (?P<code>[A-Z_]+): (?P<msg>.*)\Z", re.S)
SNAPSHOT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,80}")


class Fatal(Exception):
    """Stops the whole command: no further request is sent and every remaining company is not run."""
    def __init__(self, code, msg):
        super().__init__(f"{code}: {msg}")
        self.code, self.msg = code, msg


class Stopped(Fatal):
    def __init__(self, where):
        super().__init__("STOP", f"stop switch set {where}")


class NotRun(Exception):
    """One company's outcome, with the code that ended it."""
    def __init__(self, code, msg):
        super().__init__(f"{code}: {msg}")
        self.code, self.msg = code, msg


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def user_agent():
    """EDGAR_USER_AGENT from the environment, else the project's .env (Setup's Save writes it there), else the user's
    registry environment; None if unset. Never logged."""
    ua = os.environ.get("EDGAR_USER_AGENT") or env_file_value("EDGAR_USER_AGENT")
    if not ua and sys.platform == "win32":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                ua = winreg.QueryValueEx(k, "EDGAR_USER_AGENT")[0]
        except OSError:
            ua = None
    return (ua or "").strip() or None


# ---- the pin and the install ---------------------------------------------------------------------

def load_pin(path=PIN):
    try:
        return json.loads(Path(path).read_text(encoding="utf8"))
    except (OSError, ValueError) as e:
        raise Fatal("PIN", f"{path} cannot be read ({e})") from None


def write_pin(pin, path=PIN):
    """Write the pin whole: a temp file renamed over the old one (atomic on NTFS)."""
    path = Path(path)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(pin, indent=1), encoding="utf8")
    os.replace(tmp, path)


def venv_python(pin, root=ROOT):
    return Path(root) / pin["venv"] / "Scripts" / "python.exe"


def rate_file(pin):
    """The shared limiter path from our own non-secret configuration (the pin), %LOCALAPPDATA% expanded by code."""
    p = pin["launch"]["rate_file"]
    if "%LOCALAPPDATA%" in p:
        base = os.environ.get("LOCALAPPDATA")
        if not base or not os.path.isabs(base):
            raise Fatal("RATE_FILE", "LOCALAPPDATA is not set, so the shared limiter path cannot be formed")
        p = p.replace("%LOCALAPPDATA%", base)
    return p


def lock_versions(lock_text):
    """{normalized name: version} of every requirement line of the lock: `name==version --hash=...`, or a release
    wheel by URL, `name @ https://.../name-version-....whl --hash=sha256:...`, whose version is the wheel's. Any other
    line, or a wheel named for another package, is refused, so nothing is dropped silently."""
    out = {}
    for line in lock_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"([A-Za-z0-9_.-]+)==(\S+)", line)
        u = re.match(r"([A-Za-z0-9_.-]+) @ https://\S+/([A-Za-z0-9_.]+)-([^-/\s]+)-[^/\s]+\.whl "
                     r"--hash=sha256:[0-9a-f]{64}$", line)
        if m:
            out[norm(m.group(1))] = m.group(2)
        elif u and norm(u.group(1)) == norm(u.group(2)):
            out[norm(u.group(1))] = u.group(3)
        else:
            raise Fatal("LOCK", f"a lock line is neither name==version nor a release wheel by URL: {line[:120]}")
    return out


def lock_wheel(pin, lock_text):
    """The lock's line for the server must be the pin's release wheel by URL, with the pin's wheel hash."""
    want = f"{pin['package']} @ {pin['wheel_url']} --hash=sha256:{pin['wheel_sha256']}"
    if want not in (line.strip() for line in lock_text.splitlines()):
        raise Fatal("PIN_LOCK", f"the lock does not name the release wheel {pin['wheel_url']} with the pin's hash")


def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


PROBE = ("import importlib.metadata as m, importlib.util as u, json, sys; "
         "print(json.dumps({'dists': {d.metadata['Name']: d.version for d in m.distributions()}, "
         "'origin': u.find_spec('edgar_mcp').origin if u.find_spec('edgar_mcp') else None, "
         "'prefix': sys.prefix, 'python': sys.version.split()[0]}))")


def check_install(pin, root=ROOT, lock=None):
    """The venv the pin names, as the server will see it (-I): the lock's hash and its line naming the pin's release
    wheel, every locked distribution at its version and no other (pip aside), edgar_mcp imported from that venv and
    installed from that wheel, and its files equal to the release manifest (bytecode left out). Returns what was
    observed; raises Fatal on any difference."""
    root = Path(root)
    py = venv_python(pin, root)
    lock = Path(lock) if lock else root / pin["lock"]
    seen = {"venv": str(root / pin["venv"]), "python_exe": str(py)}
    if not py.exists():
        raise Fatal("PIN_VENV", f"no python at {py}")
    lock_bytes = lock.read_bytes()
    seen["lock_sha256"] = sha_bytes(lock_bytes)
    if seen["lock_sha256"] != pin["lock_sha256"]:
        raise Fatal("PIN_LOCK", f"{lock.name} hash {seen['lock_sha256']} is not the pin's {pin['lock_sha256']}")
    lock_wheel(pin, lock_bytes.decode("utf8"))
    p = subprocess.run([str(py), "-I", "-c", PROBE], capture_output=True, text=True, timeout=START_TIMEOUT,
                       cwd=tempfile.gettempdir())
    if p.returncode:
        raise Fatal("PIN_VENV", f"the venv probe failed: {p.stderr[-500:]}")
    probe = json.loads(p.stdout)
    seen.update(python=probe["python"], prefix=probe["prefix"], edgar_mcp_origin=probe["origin"])
    dists = {norm(k): v for k, v in probe["dists"].items()}
    want = lock_versions(lock_bytes.decode("utf8"))
    wrong = {k: {"lock": v, "venv": dists.get(k)} for k, v in want.items() if dists.get(k) != v}
    extra = sorted(set(dists) - set(want) - {"pip"})
    seen.update(distributions=len(dists), wrong_versions=wrong, extra_distributions=extra,
                mcp=dists.get("mcp"), edgar_search_mcp=dists.get("edgar-search-mcp"))
    if wrong or extra:
        raise Fatal("PIN_DISTRIBUTIONS", f"venv differs from the lock: wrong {wrong}, extra {extra}")
    site = root / pin["venv"] / "Lib" / "site-packages"
    pkg = site / "edgar_mcp"
    if probe["origin"] is None or Path(probe["origin"]).resolve() != (pkg / "__init__.py").resolve():
        raise Fatal("PIN_ORIGIN", f"edgar_mcp is imported from {probe['origin']}, not {pkg}")
    seen["wheel_sha256"] = installed_wheel(pin, site)
    seen.update(check_files(pin, pkg))
    return seen


def installed_wheel(pin, site):
    """The installed server came from the pin's wheel: the hash pip recorded for the archive it installed."""
    f = site / f"{pin['package'].replace('-', '_')}-{pin['version']}.dist-info" / "direct_url.json"
    try:
        got = json.loads(f.read_text(encoding="utf8"))["archive_info"]["hashes"]["sha256"]
    except (OSError, ValueError, KeyError, TypeError):
        got = None
    if got != pin["wheel_sha256"]:
        raise Fatal("PIN_ARTIFACT", f"the installed server came from the wheel {got}, not the pin's {pin['wheel_sha256']}")
    return got


def check_files(pin, pkg):
    """The installed package's files equal the release manifest's (bytecode left out), none missing, none added."""
    seen, installed = {}, {}
    for f in pkg.rglob("*"):
        if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc":
            installed["edgar_mcp/" + f.relative_to(pkg).as_posix()] = sha_bytes(f.read_bytes())
    trusted = {x["path"]: x["sha256"] for x in pin["files"]}
    mismatched = sorted(k for k in trusted if installed.get(k) != trusted[k])
    unexpected = sorted(set(installed) - set(trusted))
    seen.update(files_checked=len(trusted), files_mismatched=mismatched, files_unexpected=unexpected)
    if mismatched or unexpected:
        raise Fatal("PIN_FILES", f"edgar_mcp differs from the release manifest: {mismatched + unexpected}")
    return seen


def load_env(path=AGENTS_ENV):
    try:
        return json.loads(Path(path).read_text(encoding="utf8"))
    except (OSError, ValueError) as e:
        raise Fatal("CLIENT_ENV", f"{path} cannot be read ({e})") from None


def check_client(pin, env=None, root=ROOT, prefix=None, dists=None):
    """The client's own environment, against its contract (agents_env.json): this process runs in the contract's venv,
    every lock the contract names has its recorded hash, the server's lock is one of them with the pin's hash, the
    distributions are exactly the locks' union (pip aside), and the client's mcp is the version the server's lock pins,
    so client and server still share one mcp. The server keeps its own venv, checked by check_install. Returns what was
    observed; raises Fatal on any difference."""
    import importlib.metadata
    env = env if env is not None else load_env()
    root = Path(root)
    here, want = Path(prefix or sys.prefix).resolve(), (root / env["venv"]).resolve()
    seen = {"client_venv": str(want), "client_python": sys.executable, "client_version": sys.version.split()[0]}
    if here != want:
        raise Fatal("CLIENT_VENV", f"this command runs in {here}; run it with {want / 'Scripts' / 'python.exe'}")
    if pin["lock"] not in env["locks"] or env["lock_sha256"].get(pin["lock"]) != pin["lock_sha256"]:
        raise Fatal("CLIENT_LOCK", f"the client contract does not hold the server's lock {pin['lock']} at the pin's hash")
    texts = {}
    for name in env["locks"]:
        b = (root / name).read_bytes()
        if sha_bytes(b) != env["lock_sha256"].get(name):
            raise Fatal("CLIENT_LOCK", f"{name} hash {sha_bytes(b)} is not the contract's {env['lock_sha256'].get(name)}")
        texts[name] = b.decode("utf8")
    locked, clash = {}, {}
    for t in texts.values():
        for k, v in lock_versions(t).items():
            if locked.setdefault(k, v) != v:
                clash[k] = (locked[k], v)
    if clash:
        raise Fatal("CLIENT_LOCK", f"the locks pin different versions: {clash}")
    have = dists if dists is not None else {norm(d.metadata["Name"]): d.version
                                            for d in importlib.metadata.distributions()}
    wrong = {k: {"lock": v, "venv": have.get(k)} for k, v in locked.items() if have.get(k) != v}
    extra = sorted(set(have) - set(locked) - {"pip"})
    seen.update(distributions=len(have), locked=len(locked), wrong_versions=wrong, extra_distributions=extra,
                mcp=have.get("mcp"))
    if wrong or extra:
        raise Fatal("CLIENT_DISTRIBUTIONS", f"client venv differs from its locks: wrong {wrong}, extra {extra}")
    if have.get("mcp") != lock_versions(texts[pin["lock"]]).get("mcp"):
        raise Fatal("CLIENT_MCP", f"client mcp {have.get('mcp')} is not the server lock's")
    return seen


# ---- the listing: tools and server profile ---------------------------------------------------------

def tools_snapshot(tools):
    """Whole Tool objects, sorted by name, and their hash. Tool has no cache fields in mcp 2.3.0 (ttlMs and
    cacheScope are on results), so nothing is removed."""
    dumped = sorted((t if isinstance(t, dict) else t.model_dump(mode="json", by_alias=True, exclude_none=True)
                     for t in tools), key=lambda d: d["name"])
    return dumped, sha_bytes(canonical(dumped).encode("utf8"))


def check_listing(pages):
    """Listing pages, last one first-class: no next_cursor left, no name twice, the tool we call present."""
    if pages[-1].next_cursor is not None:
        raise Fatal("LISTING", f"the tool listing did not end within {MAX_LIST_PAGES} pages")
    names = [t.name for p in pages for t in p.tools]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise Fatal("LISTING", f"tools listed twice: {dup}")
    if TOOL not in names:
        raise Fatal("LISTING", f"{TOOL} is not listed")
    return [t for p in pages for t in p.tools]


def profile_snapshot(protocol_version, capabilities, instructions):
    caps = capabilities if isinstance(capabilities, dict) else \
        capabilities.model_dump(mode="json", by_alias=True, exclude_none=True)
    prof = {"protocol_version": protocol_version, "capabilities": caps, "instructions": instructions,
            "contract_version": CONTRACT}
    return prof, sha_bytes(canonical(prof).encode("utf8"))


def check_drift(pin, tools_sha, profile_sha):
    drift = [k for k, v in (("tools", tools_sha), ("profile", profile_sha)) if pin[f"{k}_sha256"] != v]
    if drift:
        raise Fatal("DRIFT", f"the server's {' and '.join(drift)} hash differs from the pin; review before any run")


def validator(pin, tool=TOOL):
    from jsonschema import Draft202012Validator
    schema = next(t["outputSchema"] for t in pin["tools"] if t["name"] == tool)
    return Draft202012Validator(schema)


# ---- one answer and one document -----------------------------------------------------------------

def text_of(result):
    return [c.text for c in result.content if getattr(c, "type", None) == "text"]


def error_code(result, tool=TOOL):
    """(code, message) of an isError answer; an answer whose text does not parse is an invalid protocol result."""
    texts = text_of(result)
    m = ERROR.match(texts[0]) if len(texts) == 1 else None
    if m is None or m.group("tool") != tool:
        raise Fatal("PROTOCOL", f"an error answer that does not parse: {texts!r:.300}")
    return m.group("code"), m.group("msg")


def success(result, validate):
    """The structured answer, after the checks every successful answer must pass."""
    sc = result.structured_content
    if not isinstance(sc, dict):
        raise Fatal("PROTOCOL", "a successful answer without structuredContent")
    texts = text_of(result)
    try:
        same = len(texts) == 1 and json.loads(texts[0]) == sc
    except ValueError:
        same = False
    if not same:
        raise Fatal("PROTOCOL", "the text block, parsed as JSON, is not the structuredContent")
    errors = sorted(validate.iter_errors(sc), key=lambda e: list(e.path))
    if errors:
        raise Fatal("PROTOCOL", f"the answer fails the pinned outputSchema: {errors[0].message:.300}")
    if sc.get("contract_version") != CONTRACT:
        raise Fatal("PROTOCOL", f"contract_version {sc.get('contract_version')!r}, not {CONTRACT}")
    return sc


def wait(seconds, stopped, sleep):
    end = time.monotonic() + seconds
    while (left := end - time.monotonic()) > 0:
        if stopped():
            raise Stopped("during a limiter wait")
        sleep(min(0.25, left))


def logged(log, rec, e):
    """Record a terminal exception once, with its latency, before it propagates."""
    if not getattr(e, "logged", False):
        code = e.code if isinstance(e, (Fatal, NotRun)) else "EXCEPTION"
        msg = e.msg if isinstance(e, (Fatal, NotRun)) else repr(e)
        log({**rec, "outcome": code, "message": f"{msg:.500}",
             "action": "not run" if isinstance(e, NotRun) else "stop"})
        e.logged = True
    return e


def answer(call, args, validate, log, stopped, sleep=time.sleep, rand=random.uniform, tool=TOOL):
    """One page: the stop switch before each attempt, two retries on RATE_LIMITER_UNAVAILABLE, every other code
    mapped. `call(args)` returns a CallToolResult or raises Fatal (timeout, protocol). Every attempt leaves one
    calls.jsonl line, also when the call or the checks of its answer raise."""
    for attempt in range(len(RETRY_DELAYS) + 1):
        if stopped():
            raise Stopped("before a request")
        t0 = time.monotonic()
        rec = {"t": now(), "tool": tool, **args, "attempt": attempt}
        try:
            result = call(args)
            rec["seconds"] = round(time.monotonic() - t0, 3)
            if not result.is_error:
                sc = success(result, validate)
                log({**rec, "outcome": "ok", **{k: sc.get(k) for k in (
                    "sha256", "total_bytes", "length", "next_offset", "contract_version", "representation",
                    "content_type")}, "sources": [{k: s.get(k) for k in ("url", "frozen", "cache_hit", "retrieved_at")}
                                                  for s in sc.get("sources", [])]})
                return sc
            code, msg = error_code(result, tool)
        except Exception as e:
            rec.setdefault("seconds", round(time.monotonic() - t0, 3))
            raise logged(log, rec, e)
        if code == RETRY_CODE and attempt < len(RETRY_DELAYS):
            delay = RETRY_DELAYS[attempt] + rand(0, JITTER)
            log({**rec, "outcome": code, "message": msg, "action": f"retry after {delay:.2f} s"})
            wait(delay, stopped, sleep)
            continue
        action = ("stop" if code == RETRY_CODE or code in FATAL_CODES or code not in NOT_RUN_CODES else "not run")
        log({**rec, "outcome": code, "message": msg, "action": action})
        if code == RETRY_CODE:
            e = Fatal(code, f"{msg} (after {len(RETRY_DELAYS)} retries; the limiter is never reset by code: "
                            f"see the contract, then {RESET})")
        elif code in NOT_RUN_CODES:
            e = NotRun(code, msg)
        elif code in FATAL_CODES:
            e = Fatal(code, msg)
        else:
            e = Fatal("UNKNOWN_CODE", f"{code}: {msg}")
        e.logged = True
        raise e


def document(call, url, validate, log, stopped, sleep=time.sleep, rand=random.uniform):
    """The whole file at `url`: its bytes and its first page's metadata. Every page must be the next one of the
    same file (url, offset, sha256, total_bytes, content_type), and the joined bytes must hash to the sha256."""
    try:
        return _document(call, url, validate, log, stopped, sleep, rand)
    except Fatal as e:
        raise logged(log, {"t": now(), "tool": TOOL, "url": url, "stage": "document"}, e)


def _document(call, url, validate, log, stopped, sleep, rand):
    parts, offset, first = [], 0, None
    while True:
        sc = answer(call, {"url": url, "offset": offset, "max_bytes": PAGE}, validate, log, stopped, sleep, rand)
        if sc.get("representation") != "source" or sc.get("url") != url or sc.get("offset") != offset:
            raise Fatal("PROTOCOL", f"page for {url} at {offset} answers {sc.get('url')} at {sc.get('offset')}")
        if not any(s.get("url") == url for s in sc.get("sources", [])):
            raise Fatal("PROTOCOL", f"no source of {url} names it")
        try:
            data = base64.b64decode(sc["data_base64"], validate=True)
        except ValueError:
            raise Fatal("PROTOCOL", f"page of {url} at {offset}: data_base64 does not decode") from None
        if first is None:
            first = sc
        elif any(sc.get(k) != first.get(k) for k in ("sha256", "total_bytes", "content_type")):
            raise Fatal("PROTOCOL", f"page of {url} at {offset} belongs to another file than its first page")
        total, nxt = sc["total_bytes"], sc["next_offset"]
        if len(data) != sc["length"] or offset + len(data) > total:
            raise Fatal("PROTOCOL", f"page of {url} at {offset}: length {sc['length']}, {len(data)} bytes, "
                                    f"total {total}")
        parts.append(data)
        offset += len(data)
        if nxt is None:
            break
        if nxt != offset or not data:
            raise Fatal("PROTOCOL", f"page of {url}: next_offset {nxt}, expected {offset} with progress")
    body = b"".join(parts)
    if offset != first["total_bytes"] or sha_bytes(body) != first["sha256"]:
        raise Fatal("PROTOCOL", f"{url}: {len(body)} bytes hashing to {sha_bytes(body)[:12]}, the answer says "
                                f"{first['total_bytes']} and {first['sha256'][:12]}")
    return body, {k: first.get(k) for k in ("url", "sha256", "total_bytes", "content_type", "sources")}


# ---- snapshots ---------------------------------------------------------------------------------------

def snapshot_path(name, base=RECORDINGS):
    if not SNAPSHOT.fullmatch(name):
        raise Fatal("SNAPSHOT", f"bad snapshot name {name!r}")
    return Path(base) / name


def open_snapshot(name, purpose, base=RECORDINGS):
    """A new recording folder, opened by an explicit, recorded act. An existing name is refused."""
    d = snapshot_path(name, base)
    meta = d.with_name(f"{name}.json")
    if d.exists() or meta.exists():
        raise Fatal("SNAPSHOT", f"snapshot {name} exists; re-recording means a new snapshot")
    d.mkdir(parents=True)
    meta.write_text(json.dumps({"snapshot": name, "opened": now(), "purpose": purpose}, indent=1), encoding="utf8")
    return d


def snapshot_dir(name, mode, base=RECORDINGS):
    """The folder a run may use: record needs an open snapshot, frozen any snapshot that exists."""
    d = snapshot_path(name, base)
    if not d.is_dir() or not d.with_name(f"{name}.json").exists():
        raise Fatal("SNAPSHOT", f"no snapshot {name}; open one first")
    if mode == "record" and d.with_name(f"{name}.closed.json").exists():
        raise Fatal("SNAPSHOT", f"snapshot {name} is closed; a missing address needs a new snapshot")
    return d


def close_snapshot(name, base=RECORDINGS):
    """Close a snapshot: check every recorded body against its description, list them with the capture window,
    and make the files read-only. Nothing is recorded into it afterwards."""
    d = snapshot_path(name, base)
    out = d.with_name(f"{name}.closed.json")
    if out.exists():
        raise Fatal("SNAPSHOT", f"snapshot {name} is already closed")
    entries = []
    for about in sorted(d.glob("*.json")):
        a = json.loads(about.read_text(encoding="utf8"))
        body = d / a["body"]
        if sha_bytes(body.read_bytes()) != a["sha256"]:
            raise Fatal("SNAPSHOT", f"{body.name} does not match its description")
        entries.append({"url": a["url"], "status": a["status"], "sha256": a["sha256"],
                        "retrieved_at": a["retrieved_at"], "description_sha256": sha_bytes(about.read_bytes())})
    stray = sorted(p.name for p in d.iterdir() if p.suffix not in (".json", ".body"))
    times = sorted(e["retrieved_at"] for e in entries)
    rec = {"snapshot": name, "closed": now(), "addresses": len(entries), "stray_files": stray,
           "capture_window": {"first": times[0] if times else None, "last": times[-1] if times else None,
                              "note": "the span of the captures, not a simultaneous SEC snapshot"},
           "entries": entries}
    for p in d.iterdir():
        p.chmod(stat.S_IREAD)
    out.write_text(json.dumps(rec, indent=1), encoding="utf8")
    return rec


# ---- the client ----------------------------------------------------------------------------------

class Fetcher:
    """One server process for one command. Synchronous for the workflow; the MCP session runs in an anyio portal.

    mode "record" uses EDGAR_RECORD_DIR, "frozen" EDGAR_FROZEN_DIR; `folder` comes from snapshot_dir() (or a temp
    folder for the start-up check). The log folder receives startup.json, calls.jsonl, server_log.jsonl,
    server_stderr.log, sdk.log and closed.json."""

    def __init__(self, mode, folder, log_dir, pin=None, rate_path=None, ua=None, stop_file=STOP_FILE,
                 root=ROOT, check_venv=True):
        if mode not in ("record", "frozen"):
            raise ValueError(mode)
        self.mode, self.folder, self.log_dir = mode, Path(folder), Path(log_dir)
        self.pin = pin if pin is not None else load_pin()
        self.pinning = self.pin["tools_sha256"] is None     # `pin` reads the listing it will store
        self.root, self.stop_file, self.check_venv = Path(root), Path(stop_file), check_venv
        self.rate_path = rate_path or rate_file(self.pin)
        self.ua = ua if ua is not None else user_agent()
        self.validate = None if self.pinning else validator(self.pin)
        self.deadline = None
        self.order = 0

    # -- logs
    def _append(self, name, rec):
        with open(self.log_dir / name, "a", encoding="utf8") as f:
            f.write(json.dumps(rec) + "\n")

    def log_call(self, rec):
        self._append("calls.jsonl", rec)

    def stopped(self):
        return self.stop_file.exists()

    # -- start and close
    def env(self):
        env = {"EDGAR_RATE_FILE": self.rate_path, "EDGAR_RATE_LIMIT": str(self.pin["launch"]["rate_limit"]),
               "EDGAR_RECORD_DIR" if self.mode == "record" else "EDGAR_FROZEN_DIR": str(self.folder)}
        if self.ua:
            env["EDGAR_USER_AGENT"] = self.ua
        return env

    def __enter__(self):
        from anyio.from_thread import start_blocking_portal
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.stopped():
            raise Stopped("before the server started")
        if self.mode == "record" and not self.ua:
            raise Fatal("NO_USER_AGENT", "EDGAR_USER_AGENT is not set; record mode would fetch live")
        client = check_client(self.pin, root=self.root) if self.check_venv else None
        self.startup = {"t": now(), "mode": self.mode, "folder": str(self.folder), "user_agent_set": bool(self.ua),
                        "rate_file": self.rate_path, "install": check_install(self.pin, self.root), "client": client}
        self.sdk_log = logging.FileHandler(self.log_dir / "sdk.log", encoding="utf8")
        self.sdk_log.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        logging.getLogger("mcp").addHandler(self.sdk_log)
        logging.getLogger("mcp").setLevel(logging.INFO)
        self.closed_how = "normal"
        self._portal_cm = start_blocking_portal()
        self.portal = self._portal_cm.__enter__()
        try:
            self._session_cm = self.portal.wrap_async_context_manager(self._session())
            self._session_cm.__enter__()
        except BaseException as e:
            self.closed_how = f"start failed: {e!r:.300}"
            self._finish()
            if isinstance(e, Fatal):
                raise
            raise Fatal("START", f"the server did not start and pass its checks: {e!r:.500}") from e
        return self

    @asynccontextmanager
    async def _session(self):
        import anyio
        from mcp import Client, StdioServerParameters, stdio_client
        from mcp.client.stdio import get_default_environment
        py = venv_python(self.pin, self.root)
        params = StdioServerParameters(command=str(py), args=self.pin["launch"]["args"], env=self.env(),
                                       cwd=str(self.log_dir))
        self.startup["inherited_env_keys"] = sorted(get_default_environment())
        self.startup["env_keys"] = sorted(self.env())

        async def on_log(p):
            self.order += 1
            self._append("server_log.jsonl", {"t": now(), "order": self.order, "level": p.level,
                                              "logger": p.logger, "data": p.data})
        with open(self.log_dir / "server_stderr.log", "a", encoding="utf8") as err:
            async with Client(stdio_client(params, errlog=err), read_timeout_seconds=START_TIMEOUT, cache=None,
                              logging_callback=on_log, log_level="info") as client:
                self.client = client
                with anyio.fail_after(START_TIMEOUT):
                    pages = [await client.list_tools(cache_mode="bypass")]
                    while pages[-1].next_cursor is not None and len(pages) < MAX_LIST_PAGES:
                        pages.append(await client.list_tools(cursor=pages[-1].next_cursor, cache_mode="bypass"))
                tools, tsha = tools_snapshot(check_listing(pages))
                prof, psha = profile_snapshot(client.protocol_version, client.server_capabilities,
                                              client.instructions)
                info = client.server_info
                self.startup.update(tools_sha256=tsha, profile_sha256=psha, profile=prof, tools=[t["name"] for t in tools],
                                    server_info=info.model_dump(mode="json", exclude_none=True) if info else None,
                                    protocol_version=client.protocol_version)
                self.tools_dump = tools
                self._write("startup.json", self.startup)
                self._write("tools.json", tools)
                if not self.pinning:
                    check_drift(self.pin, tsha, psha)
                yield client

    def _write(self, name, obj):
        p = self.log_dir / name
        if p.exists():
            raise Fatal("RUN_RECORD", f"{p} exists; run records are never overwritten")
        p.write_text(json.dumps(obj, indent=1), encoding="utf8")

    def __exit__(self, et, e, tb):
        if et is not None and self.closed_how == "normal":
            self.closed_how = ("stop switch" if et is Stopped else "interrupt" if et is KeyboardInterrupt
                               else f"{et.__name__}: {str(e):.300}")
        self._finish()
        return False

    def _finish(self):
        t0 = time.monotonic()
        try:
            if getattr(self, "_session_cm", None) is not None:
                self._session_cm.__exit__(None, None, None)
        except BaseException as e:
            self.closed_how += f"; close raised {e!r:.300}"
        finally:
            self._session_cm = None
            self._portal_cm.__exit__(None, None, None)
            logging.getLogger("mcp").removeHandler(self.sdk_log)
            self.sdk_log.close()
            rec = {"t": now(), "how": self.closed_how, "close_seconds": round(time.monotonic() - t0, 3),
                   "note": "the SDK closes stdin, waits 2 s, then kills the process tree (Job Object); "
                           "it does not report the exit status"}
            p = self.log_dir / "closed.json"
            if not p.exists():
                p.write_text(json.dumps(rec, indent=1), encoding="utf8")

    # -- calls
    @contextmanager
    def company(self, limit=COMPANY_TIMEOUT):
        """All requests inside share one time limit."""
        self.deadline = time.monotonic() + limit
        try:
            yield
        finally:
            self.deadline = None

    def call(self, args, tool=TOOL):
        """One tools/call, bounded; a timeout or a protocol failure is Fatal (the command stops, then closes)."""
        limit = PAGE_TIMEOUT
        if self.deadline is not None:
            limit = min(limit, self.deadline - time.monotonic())
            if limit <= 0:
                raise Fatal("COMPANY_TIMEOUT", f"the company's {COMPANY_TIMEOUT} s limit is reached")
        try:
            return self.portal.call(self._acall, tool, args, limit)
        except TimeoutError:
            self.closed_how = f"timeout after {limit:.0f} s on {tool} {json.dumps(args):.300}"
            raise Fatal("TIMEOUT", self.closed_how) from None
        except Fatal:
            raise
        except Exception as e:
            raise Fatal("PROTOCOL", f"the call raised {e!r:.500}") from e

    async def _acall(self, tool, args, limit):
        import anyio
        with anyio.fail_after(limit):
            return await self.client.session.call_tool(tool, args, read_timeout_seconds=limit + 5)

    def document(self, url):
        if self.validate is None:
            raise Fatal("PIN", "no pinned outputSchema; a pinning session fetches nothing")
        return document(self.call, url, self.validate, self.log_call, self.stopped)

    def source(self, tool, args):
        """The structured answer of another pinned tool that returns one SEC document unchanged (get_frames,
        get_companyconcept): checked against that tool's pinned outputSchema, with the same retries, error codes
        and calls.jsonl lines as a document page."""
        if self.validate is None:
            raise Fatal("PIN", "no pinned outputSchema; a pinning session fetches nothing")
        if tool not in {t["name"] for t in self.tools_dump}:
            raise Fatal("LISTING", f"{tool} is not listed")
        sc = answer(lambda a: self.call(a, tool), args, validator(self.pin, tool), self.log_call, self.stopped,
                    tool=tool)
        if sc.get("representation") != "source":
            raise Fatal("PROTOCOL", f"{tool} answered representation {sc.get('representation')!r}, not source")
        return sc
