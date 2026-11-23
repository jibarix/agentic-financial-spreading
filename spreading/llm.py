"""One wrapper for every model call: ask(role, system, user, schema).

Holds the call log, the call cap, the stop switch (a file named STOP in the project root, or an Event), a bound on
answer length, one retry on a bad answer, and the engine switch (Ollama, or OpenRouter with a spend cap). An answer is
accepted only if it parses and fits its JSON Schema (checked here by code, not trusted to the engine). Temperature 0 and
a fixed seed, so a repeated call adds no information.
"""
import email.utils
import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

from . import tagless
from .provenance import refuse_vendor

OLLAMA = "http://127.0.0.1:11434/api/chat"
OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"
STOP_FILE = Path(__file__).resolve().parent.parent / "STOP"
NUM_PREDICT = 600          # answer tokens; the answers are a few short fields
RETRY_WAIT = 10            # seconds before the one retry after the engine could not be reached (Ollama only)
# hosted chat: the one resend after a transient failure (HTTP
# 500/502/503/504, a broken connection, a timeout) waits this long, or a longer Retry-After that fits the wall time
# left; each request may take CHAT_TIMEOUT_HOSTED seconds (well above the slowest hosted success seen)
CHAT_RETRY_WAIT_HOSTED, CHAT_TIMEOUT_HOSTED = 20.0, 120
# OpenRouter reservation, an estimate and not a proven bound: input tokens are taken as the UTF-8 bytes of everything
# sent as text (system, user and the JSON schema, which a provider may add to the prompt), plus REQUEST_OVERHEAD for the
# chat template and provider wrapping. Neither part is established for every endpoint; a cost over the reservation is
# found only after the call is billed, and then stops the caller
REQUEST_OVERHEAD = 500
# requests sent for one question in total, the answer retry and the transport retry sharing them
DISPATCHES_PER_QUESTION = 2
# the one recovery of an explicit OpenRouter HTTP 429 per question: hosted calls only, a send of its own beside the two
# answer attempts, so at most DISPATCHES_PER_QUESTION + RATE_LIMIT_RECOVERY sends; a second 429 on the question is not
# recovered. Local calls keep DISPATCHES_PER_QUESTION. The wait is a usable Retry-After, else RATE_LIMIT_WAIT seconds
# (X-RateLimit-Reset is recorded, not used); a wait over RATE_LIMIT_MAX_WAIT or over the wall time the ledger has left
# is not taken, nor one the job allowance refuses (ledger.RATE_LIMIT_ALLOWANCE), and the question fails. Flat: no
# growth, no jitter; a Retry-After is never shortened to fit the allowance. The one application retry layer: no driver
# retries around Caller
RATE_LIMIT_RECOVERY = 1
RATE_LIMIT_WAIT = 60
# Anthropic's structured-output grammar-compilation limit, which OpenRouter returns as an HTTP 400: recovered as a
# 429 (grammar_limit), every other 400 stays terminal
GRAMMAR_LIMIT = "Grammar compilation rate limit exceeded."
RATE_LIMIT_MAX_WAIT = 300
# the stop reasons a 429 ends a question with, as Caller.failure's rate_limit_stop; the outcome stays incomplete. Other
# refusals (over_300, over_wall_budget, call_cap, ...) keep their reason only
RATE_LIMIT_STOPS = {"second_429": "second 429 on the same question",
                    "job_allowance_recoveries": "recovery refused: the job's recovery count or wait allowance cannot "
                                                "fund it",
                    "job_allowance_wait": "recovery refused: the job's recovery count or wait allowance cannot fund it"}
# OpenRouter's router metadata, asked for on every hosted request and recorded bounded and scrubbed on success and error
# replies. It may be absent; absence is recorded as None, never a failure. Its routing attempts are OpenRouter's, kept
# apart from our sends: one HTTP send is one client dispatch
ROUTER_METADATA_HEADER = {"X-OpenRouter-Metadata": "enabled"}
ROUTER_METADATA_LIMIT = 4000    # bytes of its JSON kept, after user_id is dropped and keys scrubbed
# what a failed OpenRouter attempt returned, as recorded in the call log: the body and each error field bounded, and the
# response headers on the allowlist only
HTTP_BODY_LIMIT = 2000      # bytes of the body, and of the error object's metadata, kept
HTTP_BODY_READ = 65536      # bytes read to parse the error object; only HTTP_BODY_LIMIT of them are kept
HEADER_VALUE_LIMIT = 200
HEADER_ALLOW = ("retry-after", "x-request-id", "x-generation-id", "cf-ray")
HEADER_PREFIX = "x-ratelimit-"
KEY_PATTERN = re.compile(r"sk-or-[A-Za-z0-9_-]+")    # an OpenRouter key's form, replaced wherever it appears
# the kinds of Caller.failure: what stopped the caller. transport: the request failed (an HTTP error, a 200 reply
# carrying an error object, a timeout or a broken connection) after any retry allowed; stopped: the stop switch ended
# the question after a failed attempt (during a 429 wait included); response: a 200 reply that could not be read;
# violation: the engine's counts, cost, provider or tier broke a check; ledger: the ledger refused or a commit passed a
# limit; not_sent: nothing could be sent (the vendor guard, no key, a body mismatch). A schema-invalid or cut answer
# does not stop the caller and sets no failure
FAILURE_KINDS = ("transport", "stopped", "response", "violation", "ledger", "not_sent")


ENV_FILE = STOP_FILE.parent / ".env"
LISTINGS = "https://openrouter.ai/api/v1/models"


def openrouter_listing(path=""):
    """OpenRouter's public model list (path "") or one model's endpoint list ("/<author>/<slug>/endpoints"), as JSON:
    a keyless GET, no model call (the hosted preflight's only network read)."""
    req = urllib.request.Request(LISTINGS + path, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def env_file_value(name, path=None):
    """The value of `name` in the project's .env (NAME=value lines), or None. Read when the request is built and never
    logged or printed."""
    p = Path(path) if path else ENV_FILE
    if not p.is_file():
        return None
    for line in p.read_text(encoding="utf8").splitlines():
        k, sep, v = line.strip().partition("=")
        if sep and k.strip() == name:
            return v.strip().strip('"').strip("'") or None
    return None


def count(x):
    """A token count: an int, and not a bool (Python counts True as 1)."""
    return isinstance(x, int) and not isinstance(x, bool)


def cost_valid(c):
    """A reported cost the ledger can be charged with: a finite, nonnegative number, and not a bool."""
    return isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(c) and c >= 0


def reported_cost(out):
    """usage.cost of a parsed OpenRouter body, or None; never raises, whatever the body is."""
    u = out.get("usage") if isinstance(out, dict) else None
    return u.get("cost") if isinstance(u, dict) else None


def estimate(system, user, schema):
    """Estimated input tokens of the whole request: system, user and the JSON schema as characters / 4, plus
    REQUEST_OVERHEAD / 4 for the chat template and framing."""
    return (len(system) + len(user) + len(json.dumps(schema)) + REQUEST_OVERHEAD) // 4



def model_identity(models):
    """The Ollama version and the digest of each model in use, from Ollama's local API; part of the batch freeze."""
    base = OLLAMA.rsplit("/api/", 1)[0]
    get = lambda path: json.loads(urllib.request.urlopen(base + path, timeout=30).read())
    tags = {m["name"]: m["digest"] for m in get("/api/tags")["models"]}
    missing = [m for m in models.values() if m not in tags]
    if missing:
        raise RuntimeError(f"models not installed in Ollama: {missing}")
    return {"ollama_version": get("/api/version")["version"], "digests": {m: tags[m] for m in models.values()}}


def loaded_models():
    """Ollama's loaded models and their memory (/api/ps), never a model call."""
    return json.loads(urllib.request.urlopen(OLLAMA.rsplit("/api/", 1)[0] + "/api/ps", timeout=30).read())


def cold_start(polls=5):
    """Every model Ollama holds in memory unloaded (keep_alive 0, no prompt: no model call), then /api/ps read until
    none is left, at most `polls` reads a second apart: the names unloaded and the names still loaded, so every compared
    arm starts cold. Not verified here: that the unload also drops the engine's prompt cache, and whether /api/ps can
    lag the unload."""
    base = OLLAMA.rsplit("/api/", 1)[0]
    names = lambda: [m["name"] for m in loaded_models()["models"]]
    before = names()
    for m in before:
        urllib.request.urlopen(urllib.request.Request(base + "/api/generate", json.dumps(
            {"model": m, "keep_alive": 0}).encode(), {"Content-Type": "application/json"}), timeout=120).read()
    for i in range(polls):
        left = names()
        if not left:
            break
        time.sleep(1)
    return {"unloaded": before, "still_loaded": left, "requests": 1 + len(before) + i + 1}


def send(req, timeout):
    """One request, sent once: (HTTP status, body bytes) for every HTTP response, success or error, each body read in
    full and closed here. A transport error or timeout raises. No answer validation or retry; the caller counts, stops
    and bounds the wait. `req` is a Request, or a path of Ollama's API such as "/api/ps"."""
    if isinstance(req, str):
        req = urllib.request.Request(OLLAMA.rsplit("/api/", 1)[0] + req)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        try:
            return e.code, e.read()
        finally:
            e.close()


# ---- the engine's batch size and its load record -------------------------------------------------------------------

# num_batch sent with every local chat request, by model. each value is set here by model;
# a local model not listed is sent without one and recorded NOT_PINNED. Embedding requests do not carry it
NUM_BATCH = {"qwen3.5:4b-q4_K_M": 1024, "minicpm5:2b-q4_K_M": 512}
NOT_PINNED = "batch not pinned"
NOT_OBSERVED = "loaded settings not observed"
# read only, from a remembered offset: never written, truncated or read whole (it can be large)
SERVER_LOG = Path(os.environ.get("SPREADING_SERVER_LOG")
                  or Path(os.environ.get("LOCALAPPDATA", "")) / "Ollama" / "server.log")
# the accepted profile: {"identity": {ollama_version, digests, backend}, "loads": {profile_key: fields}}; written only
# after a batch-pin check's results are accepted, never by a run
LOAD_PROFILE = STOP_FILE.parent / "data" / "engine_load_profile_v1.json"
LOAD_START, LOAD_END = 'msg="starting llama-server"', 'msg="loaded runners"'
MEMORY_WARNING = "unable to refresh free memory, using old values"
# field -> its line in a load block; a field read twice with different values makes the block unclear
LOAD_FIELDS = {"n_ctx": r"llama_context: n_ctx += (\d+)", "n_batch": r"llama_context: n_batch += (\d+)",
               "n_ubatch": r"llama_context: n_ubatch += (\d+)", "n_seq": r"llama_context: n_seq_max += (\d+)",
               "flash_attn": r"llama_context: flash_attn += (\w+)",
               "flash_attn_enabled": r"resolve_fused_ops: Flash Attention (\w+)",
               "offloaded": r"load_tensors: offloaded (\d+)/\d+ layers to GPU",
               "layers": r"load_tensors: offloaded \d+/(\d+) layers to GPU",
               "type_k": r"llama_kv_cache: size = .* K \((\w+)\)", "type_v": r"llama_kv_cache: size = .* V \((\w+)\)",
               "threads": r"system_info: n_threads = (\d+)", "threads_batch": r"system_info: .*n_threads_batch = (\d+)",
               "blob": r"--model \S*?sha256-([0-9a-f]{12})", "model": r'msg="template selection" model=(\S+)'}
STABLE = tuple(k for k in LOAD_FIELDS if k != "model")     # compared with the accepted profile and the first load
REQUIRED = tuple(LOAD_FIELDS)       # a block missing any of these is not observed (every saved block has them all)
BACKEND = ("ggml_cuda_init:", "  Device ", "load_backend:", "common_params_print_info: build")


def batch_pins(models):
    """model -> the num_batch it is sent with, or NOT_PINNED: frozen and recorded beside num_ctx."""
    return {m: NUM_BATCH.get(m, NOT_PINNED) for m in sorted(set(models))}


def load_fields(lines):
    """(fields, conflicts) of one load block: each LOAD_FIELDS value as read (None when absent), and the fields read
    with more than one value."""
    fields, conflicts = {}, []
    for k, pat in LOAD_FIELDS.items():
        got = {m.group(1) for s in lines for m in [re.search(pat, s)] if m}
        if len(got) > 1:
            conflicts.append(k)
        v = got.pop() if len(got) == 1 else None
        fields[k] = int(v) if v is not None and v.isdigit() else v
    return fields, conflicts


def load_blocks(lines):
    """Each model load in `lines`: {"first", "last" (line indexes; last None while the load is unfinished), "lines",
    "memory_warning" (MEMORY_WARNING since the previous load ended)}, from LOAD_START through LOAD_END."""
    out, cur, since = [], None, 0
    for i, s in enumerate(lines):
        if cur is None and LOAD_START in s:
            cur = {"first": i, "last": None,
                   "memory_warning": any(MEMORY_WARNING in x for x in lines[since:i])}
        if cur is not None and LOAD_END in s:
            cur["last"], since = i, i + 1
            out.append(cur)
            cur = None
    if cur is not None:
        out.append(cur)
    for b in out:
        b["lines"] = lines[b["first"]:(len(lines) if b["last"] is None else b["last"] + 1)]
    return out


def of_model(fields, model):
    """Whether a load block's model (its template selection line, e.g. registry.ollama.ai/library/<tag>) is `model`."""
    m = fields.get("model")
    return m is not None and (m == model or m.endswith("/" + model))


def load_problem(fields, conflicts):
    """Why a complete block is not evidence of a load, or None: a field missing or read twice with different values,
    or a count that is not positive (offloaded may be 0 to the total: a CPU load is observed, and flagged)."""
    missing = [k for k in REQUIRED if fields.get(k) is None]
    if missing or conflicts:
        return f"fields missing: {missing}" if missing else f"fields read twice with different values: {conflicts}"
    bad = [k for k in ("n_ctx", "n_batch", "n_ubatch", "n_seq", "layers", "threads", "threads_batch")
           if not fields[k] > 0] + ([] if 0 <= fields["offloaded"] <= fields["layers"] else ["offloaded"])
    return f"counts not positive or out of range: {bad}" if bad else None


def profile_applies(identity, model, ps, backend, version):
    """None when the accepted profile was taken on this engine version, model digest and backend; else why not."""
    if not isinstance(identity, dict):
        return "the profile names no identity"
    digest = next((m.get("digest") for m in ps if m.get("name") == model), None) if isinstance(ps, list) else None
    want = (identity.get("digests") or {}).get(model)
    why = ([] if version is not None and version == identity.get("ollama_version")
           else [f"engine {version} (profile {identity.get('ollama_version')})"])
    why += [] if digest is not None and digest == want else [f"digest {digest} (profile {want})"]
    why += [] if sorted(backend) == sorted(identity.get("backend") or []) else ["backend lines differ"]
    return "; ".join(why) or None


def engine_version():
    """Ollama's version (/api/version, no model call), or None when it cannot be read."""
    try:
        return json.loads(send("/api/version", 30)[1])["version"]
    except Exception:
        return None


def load_flags(fields, requested, profile=None, first=None):
    """How a load differs from the requested num_ctx and num_batch, from the accepted profile and from the run's
    first load of the same model, context and batch: one string each, none when nothing differs. A load with fewer
    layers on the GPU than the model has is flagged on its own, also on a run's first load with no profile."""
    out = [f"differs from requested: {f} {fields.get(f)} for {requested[k]}"
           for k, f in (("num_ctx", "n_ctx"), ("num_batch", "n_batch"))
           if isinstance(requested.get(k), int) and fields.get(f) != requested[k]]
    on, of = fields.get("offloaded"), fields.get("layers")
    if isinstance(on, int) and isinstance(of, int) and on < of:
        out.append(f"{'CPU fallback' if on == 0 else 'partial CPU fallback'}: {on} of {of} layers on the GPU")
    for name, ref in (("the accepted profile", profile), ("the run's first load", first)):
        diff = [f"{k} {ref.get(k)} -> {fields.get(k)}" for k in STABLE if ref is not None and ref.get(k) != fields.get(k)]
        if diff:
            out.append(f"differs from {name}: " + ", ".join(diff))
    return out


def profile_key(model, num_ctx, num_batch):
    return f"{model}|{num_ctx}|{num_batch}"


class LoadWatch:
    """What the engine loaded, read after every local chat send: the server log's complete new lines since the
    last read, and /api/ps once a load is tied. A send is tied to a load only when exactly one complete block names
    its model, has every REQUIRED field and positive counts, and no block in the read is unfinished or unnamed. No
    new block means the send ran on the model's last tied load, and only when that load was at the same context and
    batch ("no new load", not kept); otherwise, and in any other case or with a log that cannot be read, the send is
    NOT_OBSERVED: no value is guessed. Assumes no other process uses Ollama during the run: a load another process
    started would be tied as this send's (the log does not say who asked for a load)."""

    def __init__(self, path=None, profile=None):
        """`profile`: {"identity", "loads"} as in LOAD_PROFILE (default: that file, when it exists)."""
        self.path = Path(path or SERVER_LOG)
        if profile is None:
            profile = json.loads(LOAD_PROFILE.read_text(encoding="utf8")) if LOAD_PROFILE.exists() else {}
        self.profile, self.identity = profile.get("loads") or {}, profile.get("identity")
        self.version = engine_version() if self.profile else None
        self.first, self.current, self.records, self.unpinned, self.sends = {}, {}, [], set(), 0
        try:
            self.offset = self.path.stat().st_size
        except OSError:
            self.offset = None

    def read(self):
        """(complete new lines, the byte offset of each, and the end offset), the offset then moved past them."""
        if self.offset is None:
            self.offset = self.path.stat().st_size      # a log that was unreadable is read from here on
            raise OSError(f"server log {self.path} was unreadable at the last read")
        size = self.path.stat().st_size
        if size < self.offset:
            self.offset = size
            raise OSError(f"server log {self.path} shrank below the last offset (replaced or rotated)")
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            data = f.read(size - self.offset)
        data = data[:data.rfind(b"\n") + 1]
        at, starts = self.offset, []
        raw = data.splitlines(keepends=True)
        for s in raw:
            starts.append(at)
            at += len(s)
        self.offset = at
        return [s.decode("utf8", "replace") for s in raw], starts, at

    def after(self, model, num_ctx, num_batch, role=None):
        """The record of one send's load, kept in self.records unless it ran on a load already there."""
        self.sends += 1
        if num_batch is None:
            self.unpinned.add(model)
        rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "role": role, "model": model,
               "requested": {"num_ctx": num_ctx, "num_batch": NOT_PINNED if num_batch is None else num_batch}}
        key = profile_key(model, num_ctx, rec["requested"]["num_batch"])
        try:
            lines, starts, end = self.read()
        except OSError as e:
            return self.unobserved(rec, str(e))
        blocks = [(b, *load_fields(b["lines"])) for b in load_blocks(lines)]
        mine = [x for x in blocks if of_model(x[1], model)]
        unclear = [b for b, f, _ in blocks if b["last"] is None or f["model"] is None]

        def span(b):
            return [starts[b["first"]], end if b["last"] is None else
                    starts[b["last"]] + len(b["lines"][-1].encode("utf8"))]
        if unclear:
            return self.unobserved(rec, "a load is unfinished in the log" if unclear[0]["last"] is None
                                   else "a load whose model is not named", span(unclear[0]))
        if not mine:
            if self.current.get(model) == key:
                return {**rec, "status": "no new load"}
            return self.unobserved(rec, "no new load, and no load of this model at this context and batch was tied "
                                        "before in this run")
        if len(mine) > 1:
            return self.unobserved(rec, f"{len(mine)} loads since the last read", span(mine[0][0]))
        b, fields, conflicts = mine[0]
        why = load_problem(fields, conflicts)
        if why:
            return self.unobserved(rec, why, span(b))
        text = "".join(b["lines"]).encode("utf8")
        backend = [s.strip() for s in b["lines"] if s.startswith(BACKEND)]
        try:
            ps = [{k: m.get(k) for k in ("name", "digest", "size", "size_vram", "context_length")}
                  for m in loaded_models()["models"] if m.get("name") == model]
        except Exception as e:      # recorded, not guessed
            ps = f"not read: {e!r}"
        profile = self.profile.get(key)
        off = profile_applies(self.identity, model, ps, backend, self.version) if profile is not None else None
        flags = load_flags(fields, rec["requested"], None if off else profile, self.first.get(key))
        flags += [f"the accepted profile does not apply: {off}"] if off else []
        self.first.setdefault(key, fields)
        self.current[model] = key
        return self.keep({**rec, "status": "load", "fields": fields, "memory_warning": b["memory_warning"],
                          "backend": backend, "log_bytes": span(b), "block_sha256": hashlib.sha256(text).hexdigest(),
                          "ps": ps, "flags": flags + ([NOT_PINNED] if num_batch is None else [])})

    def unobserved(self, rec, why, span=None):
        """A NOT_OBSERVED record, kept; the model's next send without a new load cannot be tied either."""
        self.current.pop(rec["model"], None)
        return self.keep({**rec, "status": NOT_OBSERVED, "why": why, **({"log_bytes": span} if span else {})})

    def keep(self, rec):
        self.records.append(rec)
        return rec

    def summary(self):
        """The run's engine record: loads observed and not, the backend lines read and every flag."""
        loads = [r for r in self.records if r["status"] == "load"]
        return {"server_log": str(self.path), "sends": self.sends, "loads": len(loads),
                "not_observed": len(self.records) - len(loads),
                "backend": sorted({s for r in loads for s in r["backend"]}),
                "batch_not_pinned": sorted(self.unpinned),
                "flags": [f"{r['role']} {r['model']}: {f}" for r in loads for f in r["flags"]]}

    def write(self, d):
        """loads.jsonl in run folder `d`, one line per load or unobserved send; returns summary()."""
        with open(Path(d) / "loads.jsonl", "a", encoding="utf8") as f:
            for r in self.records:
                f.write(json.dumps(r) + "\n")
        return self.summary()


LOADS = None        # the LoadWatch of the run under way in this process (spread and repair runs set it), or None


# the keywords checked for each type; any other type, keyword or form is refused by check_schema
SCHEMA_KEYWORDS = {"object": {"type", "properties", "required", "additionalProperties"},
                   "array": {"type", "items", "minItems", "maxItems"}, "string": {"type", "enum", "maxLength", "minLength"}}
# the property that names an anyOf branch: "action" for step branches, "form" for v14 evidence branches; every branch of
# one anyOf uses the same one
DISCRIMINATORS = ("action", "form")


def discriminator(branches):
    """The discriminator key of a checked anyOf node's branches."""
    props = branches[0]["properties"]
    return next(k for k in DISCRIMINATORS if k in props)


def check_schema(s, path="$"):
    """Refuse (ValueError) a schema that schema_errors would not check whole: every node, visited by an answer or not,
    must be an object, array or string node with only the keywords checked for its type, in the form checked. The root
    is an object. Below it, a node may instead be the bound step form: {"anyOf": [...]} and nothing else, every branch
    an object with every property required, additionalProperties false, and an "action" property fixed to one value of
    its own, so the branch an answer means is never in doubt. The v14 evidence object uses the same bound form with
    "form" as its discriminator, placed first; one anyOf uses one discriminator in every branch. minLength (an integer
    of at least 1, at most any maxLength) is accepted only on a string its parent object requires, so it never sits
    under a property an answer may leave out."""
    def bad(why):
        raise ValueError(f"{path}: schema not checked here: {why}")
    if path == "$" and (not isinstance(s, dict) or s.get("type") != "object"):
        bad("the root must be an object")
    if path != "$" and isinstance(s, dict) and "anyOf" in s:
        branches = s["anyOf"]
        if set(s) != {"anyOf"} or not isinstance(branches, list) or not branches:
            bad("anyOf must be a non-empty list and the node's only keyword")
        actions, keys = [], set()
        for i, b in enumerate(branches):
            props = b.get("properties") if isinstance(b, dict) else None
            if (not isinstance(b, dict) or b.get("type") != "object" or b.get("additionalProperties") is not False
                    or not isinstance(props, dict) or sorted(b.get("required") or []) != sorted(props)):
                bad(f"anyOf[{i}]: an object with every property required and additionalProperties false")
            key = next((k for k in DISCRIMINATORS if k in props), None)
            if key == "action" and "form" in props:
                bad(f"anyOf[{i}]: both action and form; a branch has one discriminator")
            if key == "form" and next(iter(props)) != "form":
                bad(f"anyOf[{i}]: form must be the branch's first property")
            act = props.get(key) if key else None
            if (not isinstance(act, dict) or set(act) != {"type", "enum"} or act["type"] != "string"
                    or not isinstance(act["enum"], list) or len(act["enum"]) != 1
                    or not isinstance(act["enum"][0], str)):
                bad(f"anyOf[{i}]: an action property fixed to one value (or form, for evidence)")
            keys.add(key)
            actions.append(act["enum"][0])
            check_schema(b, f"{path}|{act['enum'][0]}")
        if len(keys) != 1:
            bad(f"anyOf branches mix discriminators: {sorted(keys)}")
        if len(set(actions)) != len(actions):
            bad(f"anyOf actions repeat: {actions}")
        return
    if not isinstance(s, dict) or s.get("type") not in SCHEMA_KEYWORDS:
        bad(f"type {s.get('type') if isinstance(s, dict) else s!r}")
    t = s["type"]
    if set(s) - SCHEMA_KEYWORDS[t]:
        bad(f"{sorted(set(s) - SCHEMA_KEYWORDS[t])} on a {t}")
    if t == "object":
        props = s.get("properties", {})
        if not isinstance(props, dict) or s.get("additionalProperties", False) is not False:
            bad("properties must be a mapping, additionalProperties absent or false")
        req = s.get("required", [])
        if not isinstance(req, list) or any(k not in props for k in req):
            bad(f"required {req!r} not all among the properties")
        for k, sub in props.items():
            if isinstance(sub, dict) and "minLength" in sub and k not in req:
                bad(f"minLength on {k}, a property an answer may leave out")
            check_schema(sub, f"{path}.{k}")
    elif t == "array":
        if "items" not in s:
            bad("an array with no items schema")
        m = s.get("maxItems", 0)
        if not isinstance(m, int) or isinstance(m, bool) or m < 0:
            bad(f"maxItems {m!r}")
        n = s.get("minItems", 0)        # the controller's order, exactly as long as the nodes offered
        if not isinstance(n, int) or isinstance(n, bool) or n < 0 or ("maxItems" in s and n > m):
            bad(f"minItems {n!r}")
        check_schema(s["items"], f"{path}[]")
    elif "enum" in s and (not isinstance(s["enum"], list) or not all(isinstance(v, str) for v in s["enum"])):
        bad(f"enum {s['enum']!r} is not a list of strings")
    elif "maxLength" in s and (not isinstance(s["maxLength"], int) or isinstance(s["maxLength"], bool)
                               or s["maxLength"] < 1):
        bad(f"maxLength {s['maxLength']!r}")
    if t == "string" and "minLength" in s:
        m = s["minLength"]
        if not isinstance(m, int) or isinstance(m, bool) or m < 1 or ("maxLength" in s and m > s["maxLength"]):
            bad(f"minLength {m!r}")


# explanatory fields: maxLength is a target, not a refusal, as hosted providers do not enforce it while decoding;
# "answer" is the tree's meaning reply, the only length-bound answer (a hosted reply ran past its cap of 600)
PROSE = ("why", "reason", "answer")


def schema_errors(x, s, path="$", long=None):
    """Errors of x against the subset of JSON Schema we use: object/properties/required/additionalProperties (false
    only), string/enum/maxLength/minLength, array/items/minItems/maxItems. The whole schema is checked first
    (check_schema), so no rule of a schema is ignored silently. A bound anyOf node is checked against the one branch
    its discriminator ("action", or "form" for v14 evidence) names. A PROSE field over its maxLength is no error
    (a hosted provider does not enforce maxLength while decoding; Ollama does): its path goes to `long`, if given."""
    if path == "$":
        check_schema(s)
    if "anyOf" in s:
        if not isinstance(x, dict):
            return [f"{path}: not an object"]
        key = discriminator(s["anyOf"])
        a = x.get(key)
        names = [b["properties"][key]["enum"][0] for b in s["anyOf"]]
        branch = dict(zip(names, s["anyOf"])).get(a) if isinstance(a, str) else None
        if branch is None:
            return [f"{path}.{key}: {x.get(key)!r} not among {names}"]
        return schema_errors(x, branch, path, long)
    t = s.get("type")
    if t == "object":
        if not isinstance(x, dict):
            return [f"{path}: not an object"]
        errs = [f"{path}.{k}: missing" for k in s.get("required", []) if k not in x]
        if s.get("additionalProperties") is False:
            errs += [f"{path}.{k}: not allowed" for k in x if k not in s.get("properties", {})]
        for k, sub in s.get("properties", {}).items():
            if k in x:
                errs += schema_errors(x[k], sub, f"{path}.{k}", long)
        return errs
    if t == "array":
        if not isinstance(x, list):
            return [f"{path}: not an array"]
        errs = [f"{path}: more than {s['maxItems']} items"] if "maxItems" in s and len(x) > s["maxItems"] else []
        errs += [f"{path}: fewer than {s['minItems']} items"] if "minItems" in s and len(x) < s["minItems"] else []
        for i, v in enumerate(x):
            errs += schema_errors(v, s.get("items", {}), f"{path}[{i}]", long)
        return errs
    if t == "string":
        if not isinstance(x, str):
            return [f"{path}: not a string"]
        if "enum" in s and x not in s["enum"]:
            return [f"{path}: {x!r} not in the enum"]
        if "maxLength" in s and len(x) > s["maxLength"]:
            if path.rsplit(".", 1)[-1] not in PROSE:
                return [f"{path}: longer than {s['maxLength']} characters"]
            if long is not None:
                long.append(f"{path}: {len(x)} of {s['maxLength']} characters")
        if "minLength" in s and len(x) < s["minLength"]:
            return [f"{path}: shorter than {s['minLength']} characters"]
    return []


class CapReached(Exception):
    pass


class Stopped(CapReached):
    """The stop switch refused the request: a refusal by a limit, not a failure of the role."""


class RequestFailed(Exception):
    """The engine could not be reached (after one retry) or answered with no message, or its own token counts were
    missing or over the budget or cap. The caller stops: every later question is refused with CapReached, so a
    builder records the rest as "not run", still writes its files and exits non-zero."""


class ErrorReply(RequestFailed):
    """OpenRouter answered HTTP 200 with an `error` object in its JSON body: a failed request (failure kind transport),
    never judged as an answer. Not retried."""


class ContextExceeded(RequestFailed):
    """One request too large for its input limit. Unlike other request failures it does not stop the caller: it ends
    only the question that was too large (termination code context_exceeded). Ollama refused the request as larger than
    num_ctx (truncate and shift off), or the engine's own count of the input was over the budget after the reply (the
    answer is not used, the attempt is charged and logged). Either way the log keeps the engine's count."""


class OverBudget(ContextExceeded):
    """A pack estimated over the input limit: not sent, and only that question ends."""


def context_exceeded(e):
    """The server's input token count when `e` is Ollama's refusal of a request over the context, else None."""
    if not isinstance(e, urllib.error.HTTPError) or e.code != 400:
        return None
    try:
        raw = e.read()
    except (AttributeError, OSError, ValueError):     # ValueError: a body already closed
        return None
    return refusal_count(e.code, raw)


def refusal_count(status, raw):
    """The server's input token count when (status, body bytes) is Ollama's refusal of a request over the context,
    else None: the one parser, for a captured body and for context_exceeded."""
    if status != 400:
        return None
    try:
        msg = json.loads(raw.decode("utf8", "replace")).get("error", "")
    except (ValueError, AttributeError):
        return None
    m = re.fullmatch(r"request \((\d+) tokens\) exceeds the available context size \((\d+) tokens\).*", msg)
    return int(m.group(1)) if m else None


def scrub(text, secret=None):
    """`text` with the key sent (`secret`) and anything of an OpenRouter key's form replaced: no record keeps a key."""
    text = KEY_PATTERN.sub("[redacted]", text)
    return text.replace(secret, "[redacted]") if secret else text


def bounded(text, limit=HTTP_BODY_LIMIT):
    """`text` cut to at most `limit` UTF-8 bytes (a character split at the cut is dropped)."""
    b = text.encode("utf8")
    return text if len(b) <= limit else b[:limit].decode("utf8", "ignore")


def error_object(err, secret=None):
    """OpenRouter's error object ({"code", "message", "metadata"}) as recorded: the code, the message bounded, the whole
    metadata object kept when its JSON fits HTTP_BODY_LIMIT (else its JSON text, cut), with provider_name, provider_code
    and limit_source also lifted out. Every string scrubbed of the key. None when `err` is not an object."""
    if not isinstance(err, dict):
        return None
    short = lambda v, n=HEADER_VALUE_LIMIT: v if v is None or count(v) else bounded(
        scrub(v if isinstance(v, str) else json.dumps(v), secret), n)
    meta = err.get("metadata")
    m = None if meta is None else scrub(json.dumps(meta), secret)
    kept = None if m is None else json.loads(m) if len(m.encode("utf8")) <= HTTP_BODY_LIMIT else \
        {"truncated_json": bounded(m)}
    lift = lambda k: short(meta.get(k)) if isinstance(meta, dict) else None
    byok = meta.get("is_byok") if isinstance(meta, dict) else None
    return {"code": short(err.get("code")), "message": short(err.get("message"), HTTP_BODY_LIMIT), "metadata": kept,
            "provider_name": lift("provider_name"), "provider_code": lift("provider_code"),
            "limit_source": lift("limit_source"), "is_byok": byok if isinstance(byok, bool) else None}


def without_user_id(v):
    """`v` with every "user_id" key dropped, at any depth: no record keeps the account's user_id."""
    if isinstance(v, dict):
        return {k: without_user_id(x) for k, x in v.items() if k != "user_id"}
    return [without_user_id(x) for x in v] if isinstance(v, list) else v


def router_metadata(j, secret=None):
    """OpenRouter's router metadata in a reply (its top-level openrouter_metadata, sent when the request carries
    ROUTER_METADATA_HEADER), as recorded: None when absent; else {"metadata": the object without user_id and scrubbed
    of the key, kept whole when its JSON fits ROUTER_METADATA_LIMIT (else {"truncated_json": its JSON text, cut}),
    "bytes": that JSON's size, "routing_attempts": how many routing attempts it lists (its "attempts", a list or a
    count; None when absent), "is_byok": its own top-level is_byok when a boolean, else None}. Its shape beyond that
    is not assumed. Routing attempts are OpenRouter's, never our sends."""
    meta = j.get("openrouter_metadata") if isinstance(j, dict) else None
    if meta is None:
        return None
    text = scrub(json.dumps(without_user_id(meta)), secret)
    n = len(text.encode("utf8"))
    tried = meta.get("attempts") if isinstance(meta, dict) else None
    byok = meta.get("is_byok") if isinstance(meta, dict) else None
    return {"metadata": json.loads(text) if n <= ROUTER_METADATA_LIMIT
            else {"truncated_json": bounded(text, ROUTER_METADATA_LIMIT)},
            "bytes": n, "routing_attempts": len(tried) if isinstance(tried, list)
            else tried if count(tried) and tried >= 0 else None,
            "is_byok": byok if isinstance(byok, bool) else None}


def http_failure(e, secret=None, body=None):
    """What a failed OpenRouter attempt returned: the exception type and HTTP status (None when no response came; 200
    for a reply received but failed), and for an HTTP error the allowlisted headers only (names lowercased, values
    bounded), the body's first HTTP_BODY_LIMIT bytes and its error object; for a 200 reply its error object; with a
    body, its router metadata (router_metadata, None when absent). `body`: the 200 reply's bytes, when one was received.
    The request's headers are never read, so the Authorization header cannot enter the record; the key is scrubbed from
    every value kept."""
    rec = {"type": type(e).__name__, "status": 200 if body is not None else None}
    if isinstance(e, urllib.error.HTTPError):
        rec["status"] = e.code
        try:
            items = list((e.headers or {}).items())
        except (AttributeError, TypeError):
            items = []
        rec["headers"] = {k.lower(): bounded(scrub(str(v), secret), HEADER_VALUE_LIMIT) for k, v in items
                          if k.lower() in HEADER_ALLOW or k.lower().startswith(HEADER_PREFIX)}
        try:
            body = e.read(HTTP_BODY_READ + 1)
        except (AttributeError, OSError, ValueError):
            body = None
    if isinstance(body, bytes):
        rec["body"] = bounded(scrub(body.decode("utf8", "replace"), secret))
        rec["body_bytes_read"] = len(body)
        rec["body_truncated"] = len(body) > HTTP_BODY_LIMIT
        try:
            j = json.loads(body)
        except ValueError:
            j = None
        rec["error"] = error_object(j.get("error"), secret) if isinstance(j, dict) else None
        rec["router_metadata"] = router_metadata(j, secret)
    return rec


def retry_after_seconds(value, now):
    """(seconds, unit) from a Retry-After value, delay-seconds or an HTTP date (RFC 9110, 10.2.3), measured from
    `now` (epoch seconds) and never below 0; None when absent or unusable."""
    v = (value or "").strip()
    if re.fullmatch(r"\d+(\.\d+)?", v):
        return float(v), "seconds"
    try:
        when = email.utils.parsedate_to_datetime(v)
    except (TypeError, ValueError, IndexError):
        return None
    if when is None or when.tzinfo is None:
        return None
    return max(0.0, when.timestamp() - now), "http_date"


def rate_limit_hint(headers, now):
    """(seconds to wait, source, the header value used, its unit) for an OpenRouter 429, from the allowlisted headers: a
    usable Retry-After, else RATE_LIMIT_WAIT (source "default_60"). X-RateLimit-Reset is not used: its unit is
    undocumented, and reading it from the value's size could give a zero wait; rate_limit keeps it raw in the event."""
    got = retry_after_seconds(headers.get("retry-after"), now)
    if got is not None:
        return got[0], "retry_after", headers.get("retry-after"), got[1]
    return float(RATE_LIMIT_WAIT), "default_60", None, None


UPSTREAM_LIMIT_SOURCES = ("upstream_provider_shared_pool",)    # metadata.limit_source values read as upstream


def grammar_limit(e, http):
    """True for Anthropic's grammar-compilation rate limit returned as an HTTP 400: provider Anthropic, the upstream
    error (metadata.raw, parsed) of type invalid_request_error with a message starting GRAMMAR_LIMIT. Anything else,
    an unreadable raw included, is not."""
    err = (http or {}).get("error") or {}
    if not isinstance(e, urllib.error.HTTPError) or e.code != 400 or err.get("provider_name") != "Anthropic":
        return False
    try:
        up = json.loads((err.get("metadata") or {}).get("raw") or "")["error"]
    except (ValueError, TypeError, KeyError):
        return False
    return isinstance(up, dict) and up.get("type") == "invalid_request_error" \
        and str(up.get("message") or "").startswith(GRAMMAR_LIMIT)


def rate_limit_class(http):
    """(whose limit a 429 came from, what that was read from, the other evidence that points elsewhere), from what was
    recorded, never assumed, in this order: "upstream" when the error metadata's limit_source is exactly one of
    UPSTREAM_LIMIT_SOURCES; else "platform" from X-RateLimit-* headers (OpenRouter's own limit); else "upstream" from a
    nonempty provider_code; else "unknown" (basis None). Another limit_source value classifies nothing by itself; the
    caller records it raw. Nothing is read from the prose of raw or remedy_hint. The conflicts: each later piece of
    evidence, as {"basis", "class"}, whose class differs."""
    err = http.get("error") or {}
    code = err.get("provider_code")
    found = [(basis, cls) for basis, cls, there in (
        ("limit_source", "upstream", err.get("limit_source") in UPSTREAM_LIMIT_SOURCES),
        ("x_ratelimit_headers", "platform", any(k.startswith(HEADER_PREFIX) for k in http.get("headers") or {})),
        ("provider_code", "upstream", code is not None and not (isinstance(code, str) and not code.strip())))
        if there]
    if not found:
        return "unknown", None, []
    basis, cls = found[0]
    return cls, basis, [{"basis": b, "class": c} for b, c in found[1:] if c != cls]


def schema_retry_429s(records):
    """The HTTP 429s received on a schema retry, from one run's call records in order (counted apart from
    schema_retry_lost_to_429): {"count", "recovery_ran", "ended": {how: n}}. How each ended: "not_run: <reason>"
    (rate_limit refused the recovery), "stopped" or "interrupted" (the wait), "not_sent" (the recovery was not sent
    after the wait: the stop switch, a cap or the ledger), "send_failed" (the recovery's request failed, a second 429
    included), "valid" or "invalid" (the recovery's answer)."""
    out = {"count": 0, "recovery_ran": 0, "ended": {}}
    for i, r in enumerate(records):
        ev = r.get("rate_limit")
        if "system" not in r or not isinstance(ev, dict) or ev.get("dispatch") != "schema_retry":
            continue
        same = [x for x in records[i + 1:] if (x.get("role"), x.get("tag")) == (r.get("role"), r.get("tag"))]
        wait = next((x["rate_limit_wait"] for x in same if "rate_limit_wait" in x), None)
        nxt = next((x for x in same if "system" in x and "after_429" in x), None)
        how = (f"not_run: {ev.get('reason')}" if not ev.get("retry")
               else wait["wait_ended"] if wait is not None and wait.get("wait_ended") != "completed"
               else "not_sent" if nxt is None
               else "send_failed" if "http" in nxt or str(nxt.get("error") or "").startswith("request failed")
               else "valid" if nxt.get("answer") is not None else "invalid")
        out["count"] += 1
        out["recovery_ran"] += how in ("send_failed", "valid", "invalid")
        out["ended"][how] = out["ended"].get(how, 0) + 1
    return out


def recovery_counts(records):
    """The 429 recoveries recounted from call records alone: the ledger enforces the job allowance, this reader
    reconciles it independently. Each count from one kind of record, so nothing is counted twice: "429s" and "claimed"
    (granted by the allowance; "refused" by reason) and the planned wait claimed from each 429's own record as written
    before its wait; "waits", "waited_s" and how each wait ended from the wait bookkeeping records (interrupted and
    stopped waits included; the copies in after_429 are not read); "recovery_sent" from the sends marked after_429, of
    which "recovery_replied" had an HTTP reply (any status, the 429 or 200 of a parsed answer included), "recovery_429"
    a second 429, and "recovery_valid" a schema-valid answer. Physical sends only (the drivers' rule: messages, attempt
    1 or 2, not marked not sent)."""
    sent = [r for r in records if "system" in r and (r.get("attempt") or 0) >= 1
            and "not sent" not in str(r.get("error") or "")]
    out = {"429s": 0, "claimed": 0, "refused": {}, "planned_wait_s": 0.0, "waits": 0, "waited_s": 0.0,
           "wait_ended": {}, "recovery_sent": 0, "recovery_replied": 0, "recovery_429": 0, "recovery_valid": 0}
    for r in sent:
        ev = r.get("rate_limit")
        status = (r["http"] or {}).get("status") if "http" in r else 200     # no http: a reply judged
        if isinstance(ev, dict):
            out["429s"] += 1
            if ev.get("reason") is None:
                out["claimed"] += 1
                out["planned_wait_s"] += ev.get("planned_wait_s") or 0.0
            else:
                out["refused"][ev["reason"]] = out["refused"].get(ev["reason"], 0) + 1
        if "after_429" in r:
            out["recovery_sent"] += 1
            out["recovery_replied"] += status is not None
            out["recovery_429"] += status == 429
            out["recovery_valid"] += r.get("answer") is not None
    for r in records:
        w = r.get("rate_limit_wait")
        if isinstance(w, dict):
            out["waits"] += 1
            out["waited_s"] += w.get("wait_s") or 0.0
            out["wait_ended"][w.get("wait_ended")] = out["wait_ended"].get(w.get("wait_ended"), 0) + 1
    out["planned_wait_s"], out["waited_s"] = round(out["planned_wait_s"], 3), round(out["waited_s"], 3)
    return out


OMITTABLE = frozenset({"seed", "temperature", "presence_penalty", "frequency_penalty"})


class Caller:
    # off for every run's caller; on only in tests
    one_send = False                # no answer retry: the first judged response is returned, whatever it is

    def __init__(self, models, log_path, max_calls, seed=7, num_ctx=4096, stop=None, engine="ollama",
                 spend_cap_usd=None, answer_cap=NUM_PREDICT, reasoning_effort=None, max_tokens=None, price=None,
                 providers=None, ledger=None, company=None, options=None, reasoning_off=False,
                 omit=None, role_caps=None, service_tier=None, provider_names=None, reasoning=None):
        self.models = models            # role -> model name, e.g. {"proposer": "qwen3.5:9b-q4_K_M"}
        self.log_path = log_path
        self.max_calls = max_calls
        self.calls = 0
        self.seed = seed
        self.num_ctx = num_ctx
        # explicit Ollama sampler options sent with every request and logged, e.g. {"presence_penalty": 0}: an option
        # left out falls back to the modelfile's PARAMETER lines (qwen3.5:9b ships presence_penalty 1.5)
        self.options = dict(options or {})
        if set(self.options) & {"temperature", "seed", "num_ctx", "num_predict", "num_batch"}:
            raise ValueError(f"options may not set the fixed sampler settings: {sorted(self.options)}")
        self.stop = stop                # a threading.Event or None: the stop switch
        self.failed = None              # the first request failure; no request is made after it
        # the same failure structured for drivers (fail(): kind in FAILURE_KINDS, why, role, tag, attempt, dispatch,
        # http_status, error_code), or None
        self.failure = None
        self.rate_limits = []           # one event per OpenRouter 429: class, wait hint, wait taken, retry or why not
        # this caller's 429 recoveries and planned wait (claim_429): the job allowance is the ledger's (held across
        # callers and processes); this count decides only when the ledger cannot (None, or without claim_429)
        self.rate_limit_used = {"recoveries": 0, "wait_s": 0.0}
        # the 429 wait's clock, sleep and epoch time (an HTTP-date hint), replaced in tests: no real sleeping there
        self.clock, self.sleep, self.now = time.monotonic, time.sleep, time.time
        self._at = {}                   # where ask() is (role, tag, attempt, dispatch), for self.failure
        self._left = None               # (seconds the ledger left before its nearer deadline, self.clock() then)
        self.max_prompt_tokens = 0      # largest engine-counted input and answer, for the run's success check
        self.max_answer_tokens = 0
        self.violations = []            # calls whose engine counts broke the budget or cap, or were missing
        self.context_refusals = []      # questions too large (Ollama's refusal, or the engine count over budget
                                        # after the reply); the caller goes on
        self.engine = engine            # "ollama" or "openrouter"
        self.spend_cap_usd = spend_cap_usd  # OpenRouter: no request once the reported cost reaches it
        self.spent_usd = 0.0
        self.unknown_usd = 0.0          # OpenRouter: failed attempts with no reported cost, at their full reservation
        # OpenRouter: (USD per input, per output token), sent as the price ceiling; or role -> that pair when the
        # roles use different models
        self.price = price
        self.answer_cap = answer_cap    # visible answer tokens (reasoning tokens excluded), checked on engine counts
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens    # OpenRouter: the request's limit on reasoning plus answer tokens
        # OpenRouter: the only provider endpoints allowed, no fallback (a pin); or role -> that list
        self.providers = providers
        # OpenRouter: send reasoning {"enabled": false}, thinking off as on the local engine
        self.reasoning_off = reasoning_off
        # OpenRouter, opt-in: role -> "off" (reasoning {"enabled": false}, checked as reasoning_off is) or an effort
        # ({"effort": ...}), with max_tokens then role -> that role's limit on reasoning plus answer tokens. None: the
        # scalar settings above
        self.reasoning = None if reasoning is None else dict(reasoning)
        self.ledger = ledger            # the batch ledger (spreading/ledger.py) or None: every attempt reserved first
        self.company = company          # the ledger's company key for this caller's attempts
        self._settled = set()           # reservations already committed
        self.hosted_tokens = 0          # OpenRouter's reported prompt plus completion tokens (the ledger limits dollars)
        self.hosted_unknown = 0         # OpenRouter attempts settled with no counts or cost: their tokens are unknown
        # OpenRouter: role -> request fields left out for that role's model, whose endpoint does not advertise them
        # (Haiku: seed, temperature and both penalties). Every other field, the pin, the price ceiling, the strict
        # schema and reasoning off included, is sent as for any role
        self.omit = {r: frozenset(v) for r, v in (omit or {}).items()}
        if any(v - OMITTABLE for v in self.omit.values()) or (self.omit and engine != "openrouter"):
            raise ValueError(f"omit is an OpenRouter setting over {sorted(OMITTABLE)} only: {omit}")
        # role -> {"calls": requests, "spend_usd": USD or None}: a sub-cap inside the caller's own caps (e.g. Haiku's).
        # Reaching it refuses that role's requests only (CapReached, the caller not failed), so the other roles go on
        self.role_caps = {r: dict(v) for r, v in (role_caps or {}).items()}
        if any(set(v) - {"calls", "spend_usd"} or not isinstance(v.get("calls"), int) or v["calls"] < 0
               for v in self.role_caps.values()):
            raise ValueError(f"role_caps takes calls (an integer) and spend_usd per role: {role_caps}")
        self.role_calls, self.role_spent = defaultdict(int), defaultdict(float)
        self.role_unknown = defaultdict(float)  # unknown_usd by role
        # OpenRouter, opt-in: the top-level service_tier sent with every request (e.g. "default", standard service). The
        # tier returned is recorded with the served provider and tested before the answer is read: another tier, or
        # none, stops the caller (judge). None: no field and no test
        self.service_tier = service_tier
        if service_tier is not None and (engine != "openrouter" or not isinstance(service_tier, str)
                                         or not service_tier):
            raise ValueError(f"service_tier is an OpenRouter setting, a non-empty string: {service_tier!r}")
        # OpenRouter, opt-in: pin -> the exact provider names a response may carry, from the frozen endpoint listing.
        # Every pin of every role needs its entry. None: the pin's provider slug is compared with the name returned
        self.provider_names = None if provider_names is None else {p: tuple(v) for p, v in provider_names.items()}
        if provider_names is not None:
            pins = [p for v in ((providers or {}).values() if isinstance(providers, dict) else [providers])
                    for p in (v or [])]
            if engine != "openrouter" or not pins or set(pins) - set(self.provider_names) or not all(
                    v and all(isinstance(n, str) and n for n in v) for v in self.provider_names.values()):
                raise ValueError(f"provider_names is an OpenRouter setting naming each pin's exact provider names: "
                                 f"{provider_names!r} for pins {pins}")
        if engine == "openrouter" and (spend_cap_usd is None or price is None or not max_tokens):
            raise ValueError("an OpenRouter caller needs a spend cap, a price ceiling and max_tokens")
        if reasoning_off and (engine != "openrouter" or reasoning_effort):
            raise ValueError("reasoning_off is an OpenRouter setting and excludes reasoning_effort")
        # every role of the caller needs its setting and its limit (at least the answer cap); refused when made,
        # before anything is sent
        if (reasoning is not None or isinstance(max_tokens, dict)) and (
                engine != "openrouter" or reasoning_off or reasoning_effort or not isinstance(reasoning, dict)
                or not isinstance(max_tokens, dict) or set(models) - set(reasoning)
                or set(models) - set(max_tokens)
                or not all(isinstance(reasoning[r], str) and reasoning[r] for r in models)
                or not all(type(max_tokens[r]) is int and max_tokens[r] >= answer_cap for r in models)):
            raise ValueError(f"per-role reasoning is an OpenRouter setting: each role's 'off' or effort and its "
                             f"max_tokens (an integer, at least the answer cap {answer_cap}), without reasoning_off or "
                             f"reasoning_effort: reasoning {reasoning!r}, max_tokens {max_tokens!r}")

    def role_cap_hit(self, role, reserved):
        """The reason this role's sub-cap, or the ledger's own limit for the role (a company ledger's, held across
        runs: ledger.role_refusal), refuses one more request, or None."""
        rc = self.role_caps.get(role)
        if rc is not None and self.role_calls[role] >= rc["calls"]:
            return f"{role} call cap {rc['calls']}"
        spent = self.role_spent[role] + self.role_unknown[role]
        if rc is not None and rc.get("spend_usd") is not None and reserved is not None and \
                spent + reserved > rc["spend_usd"]:
            return (f"{role} spend cap {rc['spend_usd']} USD: spent {spent:.4f}, one more call can "
                    f"cost {reserved:.4f}")
        why = self.ledger.role_refusal(role, reserved or 0.0) if hasattr(self.ledger, "role_refusal") else None
        return why and f"ledger: {why}"

    def over_spend(self, reserved):
        """OpenRouter: why one more send at `reserved` would pass the Caller's spend cap, counting failed attempts
        at their reservation (unknown_usd), or None."""
        spent = self.spent_usd + self.unknown_usd
        if self.engine == "openrouter" and spent + reserved > self.spend_cap_usd:
            return f"spend cap {self.spend_cap_usd} USD: spent {spent:.4f}, one more call can cost {reserved:.4f}"
        return None

    def role_price(self, role):
        """(USD per input, per output token) for this role's model."""
        return self.price[role] if isinstance(self.price, dict) else self.price

    def role_off(self, role):
        """Whether this role's requests send reasoning off (and are checked for zero reasoning)."""
        return self.reasoning_off if self.reasoning is None else self.reasoning[role] == "off"

    def role_effort(self, role):
        """The reasoning effort this role's requests send, or None."""
        if self.reasoning is None:
            return self.reasoning_effort
        return None if self.reasoning[role] == "off" else self.reasoning[role]

    def role_max_tokens(self, role):
        """The limit on reasoning plus answer tokens this role's requests send (OpenRouter)."""
        return self.max_tokens[role] if isinstance(self.max_tokens, dict) else self.max_tokens

    def role_providers(self, role):
        """The pinned provider endpoints for this role's model, or None."""
        return self.providers.get(role) if isinstance(self.providers, dict) else self.providers

    def reservation(self, system, user, schema, role=None):
        """OpenRouter: the estimated most one call costs at the price ceiling, from the whole request (REQUEST_OVERHEAD).
        An estimate, not a bound (see REQUEST_OVERHEAD)."""
        sent = len((system + user + json.dumps(schema)).encode("utf8")) + REQUEST_OVERHEAD
        price = self.role_price(role)
        return sent * price[0] + self.role_max_tokens(role) * price[1]

    def ask_reservation(self, role, system, user, schema, turns):
        """The reservation ask() takes for this question: reservation() over system, the earlier turns' text and
        user."""
        return self.reservation(system, "".join(t["content"] for t in turns) + user, schema, role)

    def attempt_tokens(self):
        """Local tokens to reserve in the batch ledger for one transport attempt: num_ctx plus the answer cap. num_ctx
        alone is not yet an established bound on input plus output as reported."""
        return self.num_ctx + self.answer_cap

    def reserve_attempt(self, role, reserved):
        """(ledger reservation ID or None, timeout in seconds) for one transport attempt. A ledger that refuses (a
        limit, a deadline, the batch stopped) stops this caller before anything is sent (CapReached). A role's own
        ledger limit refuses that role only: CapReached, the caller not failed."""
        self._left = None
        local = self.engine == "ollama"
        most = 600 if local else CHAT_TIMEOUT_HOSTED
        if self.ledger is None:
            return None, most
        from .ledger import LimitReached, RoleLimitReached
        try:
            rid = self.ledger.reserve(self.company, role, self.attempt_tokens() if local else 0,
                                      usd=0.0 if local else reserved, local=local)
        except RoleLimitReached as e:
            raise CapReached(f"ledger: {e}") from e
        except LimitReached as e:
            raise CapReached(self.fail("ledger", f"ledger: {e}")) from e
        try:
            left = self.ledger.before_send(rid)
        except LimitReached as e:
            self.ledger.release(rid, f"refused before send: {e}")
            raise CapReached(self.fail("ledger", f"ledger: {e}")) from e
        self._left = (left, self.clock())       # the wall a 429 wait is measured against
        return rid, max(1, min(most, left))

    def settle_attempt(self, rid, pt=None, total=None, cost=None, why=None):
        """Commit an attempt that went out: the engine's own counts when it reported them, else the whole
        reservation (commit_unknown). A limit passed on commit keeps this answer (it was received) and stops the
        caller for every later question."""
        if rid is None or rid in self._settled:
            return
        self._settled.add(rid)
        from .ledger import LimitReached
        try:
            if why is None and count(pt) and count(total) and (self.engine == "ollama" or isinstance(cost, (int, float))):
                if self.engine != "ollama":         # counted before the commit, which may stop the batch
                    self.hosted_tokens += pt + total
                self.ledger.commit(rid, tokens=pt + total if self.engine == "ollama" else 0,
                                   usd=0.0 if self.engine == "ollama" else float(cost))
            else:
                if self.engine != "ollama":
                    self.hosted_unknown += 1
                self.ledger.commit_unknown(rid, why or "engine counts or cost not reported")
        except LimitReached as e:
            self.fail("ledger", f"ledger: {e}", keep=True)

    def settle_not_billed(self, rid):
        """Commit an attempt that went out and was refused unbilled (an explicit HTTP 429): one request sent, no
        tokens, no dollars. A limit passed on commit stops the caller, as settle_attempt's does."""
        if rid is None or rid in self._settled:
            return
        self._settled.add(rid)
        from .ledger import LimitReached
        try:
            self.ledger.commit(rid, tokens=0, usd=0.0)
        except LimitReached as e:
            self.fail("ledger", f"ledger: {e}", keep=True)

    def fail(self, kind, why, keep=False, **at):
        """Stops the caller: self.failed, the string every driver reads, and self.failure, the same failure
        structured (kind in FAILURE_KINDS; role, tag, attempt and dispatch of the question being asked; http_status,
        error_code and any other field given in `at`). `keep`: a failure already recorded stays. Returns the
        caller's failure string."""
        if keep and self.failed:
            return self.failed
        self.failed = why
        self.failure = {"kind": kind, "why": why, "role": None, "tag": None, "attempt": None, "dispatch": None,
                        "http_status": None, "error_code": None, **self._at, **at}
        return why

    def wait_out(self, seconds):
        """(seconds waited, whether the stop switch ended the wait): sleeps in steps of at most a second, checking the
        stop switch before each (self.clock and self.sleep)."""
        start = self.clock()
        while True:
            if self.stopped():
                return self.clock() - start, True
            left = seconds - (self.clock() - start)
            if left <= 0:
                return self.clock() - start, False
            self.sleep(min(1.0, left))

    def question_sends(self):
        """The most requests one question may send: DISPATCHES_PER_QUESTION, plus RATE_LIMIT_RECOVERY on a hosted caller
        that retries (not on the local engine or in one-send mode)."""
        return DISPATCHES_PER_QUESTION + (RATE_LIMIT_RECOVERY if self.engine == "openrouter" and not self.one_send
                                          else 0)

    def claim_429(self, role, wait):
        """(None, used) when the job allowance grants one more 429 recovery waiting `wait` seconds, else (reason,
        used). The ledger decides when there is one (ledger.claim_429: it holds the job's count across callers and
        child processes, and records every decision); without one, this caller's own count against
        ledger.RATE_LIMIT_ALLOWANCE. A grant is also counted in the caller's own count."""
        from .ledger import allowance_refusal
        mine = self.rate_limit_used
        if hasattr(self.ledger, "claim_429"):
            reason, used = self.ledger.claim_429(self.company, role, wait)
        else:
            refused = allowance_refusal(mine, wait)
            reason, used = (None, None) if refused is None else (refused[0], {**mine, "why": f"caller: {refused[1]}"})
        if reason is None:
            mine["recoveries"] += 1
            mine["wait_s"] = round(mine["wait_s"] + wait, 3)
        return reason, used if used is not None else dict(mine)

    def rate_limit(self, http, sent, cap, role, tag, consumer, recovered=False, resent=False):
        """The decision on an explicit OpenRouter HTTP 429, taken before any wait so the 429's record is written first.
        The event is kept in self.rate_limits and returned for the attempt's log record: class and its source
        (rate_limit_class) with any conflicting evidence and the raw limit_source, is_byok as supplied (True or False;
        None when missing or not a boolean, never inferred) and where it came from (error_metadata, else
        router_metadata; None), the hint (source retry_after or default_60, the value and its unit), the raw
        X-RateLimit-Reset (kept for diagnosis, never used: its unit is undocumented), the planned wait, the wall left,
        retry or the reason for none (second_429 when `recovered`: the question has had its recovery, then
        retry_used when `resent`: the 429 came on the attempt's one resend, then dispatch_budget_used, one_send, over_300, over_wall_budget, call_cap, job_allowance_recoveries or
        job_allowance_wait), and the job allowance as used ("allowance", with "why" on a refusal). The allowance is
        claimed last, only when nothing else refuses. wait_s and wait_ended are set by wait_429, retry_ok when the
        recovery's send is known to have had a reply or not."""
        wait, source, hint, unit = rate_limit_hint(http.get("headers") or {}, self.now())
        left = None if self._left is None else self._left[0] - (self.clock() - self._left[1])
        cls, why_cls, conflict = rate_limit_class(http)
        err, routed = http.get("error") or {}, http.get("router_metadata") or {}
        byok = next(((v, where) for v, where in ((err.get("is_byok"), "error_metadata"),
                                                 (routed.get("is_byok"), "router_metadata")) if isinstance(v, bool)),
                    (None, None))
        ev = {"role": role, "tag": tag, "dispatch": consumer, "http_status": http.get("status"), "class": cls,
              "class_source": why_cls,
              "class_conflict": conflict, "limit_source": err.get("limit_source"),
              "is_byok": byok[0], "is_byok_source": byok[1], "source": source,
              "hint": hint, "hint_unit": unit,
              "x_ratelimit_reset": (http.get("headers") or {}).get("x-ratelimit-reset"),     # raw, never used
              "planned_wait_s": round(wait, 3), "wall_left_s": None if left is None else round(left, 3),
              "wait_s": 0.0, "wait_ended": None, "allowance": None}
        reason = ("second_429" if recovered else "retry_used" if resent
                  else "dispatch_budget_used" if sent >= self.question_sends() else "one_send" if self.one_send
                  else "over_300" if wait > RATE_LIMIT_MAX_WAIT
                  else "over_wall_budget" if left is not None and wait > left
                  else "call_cap" if self.calls >= cap else None)
        if reason is None:
            reason, ev["allowance"] = self.claim_429(role, round(wait, 3))
        ev.update(retry=reason is None, reason=reason, retry_ok=None)
        self.rate_limits.append(ev)
        return ev

    def wait_429(self, ev, e, http, attempt):
        """The wait before a 429's retry, after the 429's record is written and its reservation settled. How it ended
        goes into the event (wait_s, wait_ended: completed, stopped or interrupted) and into a bookkeeping record,
        which is not a request (no messages, no dispatch). Ended by the stop switch: failure kind "stopped" with the
        429's evidence, Stopped raised. Ended by an exception (KeyboardInterrupt): the same failure, and the exception
        goes on. Nothing is sent from here."""
        start, ended = self.clock(), "interrupted"
        try:
            ended = "stopped" if self.wait_out(ev["planned_wait_s"])[1] else "completed"
        finally:
            ev["wait_s"], ev["wait_ended"] = round(self.clock() - start, 3), ended
            if ended != "completed":
                ev.update(retry=False, reason=ended)
            self.log({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "role": ev["role"], "model": self.models[ev["role"]],
                      "tag": ev["tag"], "attempt": attempt, "rate_limit_wait": dict(ev)})
            if ended != "completed":
                self.fail("stopped", f"{e!r}; billing outcome unknown; no retry: {ended}",
                          http_status=ev.get("http_status") or 429,
                          error_code=(http.get("error") or {}).get("code"), rate_limit_class=ev["class"],
                          rate_limit_reason=ended, wait_ended=ended)
        if ended == "stopped":
            raise Stopped(self.failed) from e

    def sent_options(self, role=None):
        """The Ollama options of every request: the explicit ones, then the fixed settings. The requested output is
        the answer cap (NUM_PREDICT unless the caller sets one), so the cap checked is the cap requested. num_batch
        last, from NUM_BATCH for the role's model (with no role, the caller's one model), when it is listed."""
        batch = self.batch(role)
        return {**self.options, "temperature": 0, "seed": self.seed, "num_ctx": self.num_ctx,
                "num_predict": self.answer_cap, **({} if batch is None else {"num_batch": batch})}

    def batch(self, role=None):
        """The pinned num_batch of the role's model on the local engine, or None (hosted, or not listed)."""
        models = {self.models[role]} if role is not None else set(self.models.values())
        return NUM_BATCH.get(models.pop()) if self.engine == "ollama" and len(models) == 1 else None

    def _open(self, req, timeout, role):
        """One send's response bytes; then, on the local engine with a run's LoadWatch set, the load it used. The
        options object of the bytes dispatched (None on a hosted body, which has none) is kept for the call's record,
        "sent_options"; the bytes are read, never changed."""
        self._sent = json.loads(req.data).get("options")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        finally:
            if LOADS is not None and self.engine == "ollama":
                LOADS.after(self.models[role], self.num_ctx, self.batch(role), role)

    def stopped(self):
        return STOP_FILE.exists() or (self.stop is not None and self.stop.is_set())

    def log(self, rec):
        if getattr(self, "_turns", None) and "system" in rec:
            rec["turns"] = self._turns          # earlier chat turns of this question, before its last user message
        if "system" in rec:                     # the rest of what the model was sent
            rec["schema"] = getattr(self, "_schema", None)
            rec["options"] = self.options
            rec["sent_options"] = getattr(self, "_sent", None)      # what was dispatched; None when nothing was
        with open(self.log_path, "a", encoding="utf8") as f:
            f.write(json.dumps(rec) + "\n")

    def ask(self, role, system, user, schema, tag="", ceiling=None, budget=None, turns=None):
        """One question, one answer. `turns`: earlier chat turns ({"role": "user" or "assistant", "content"}) sent
        between the system message and `user`, counted in the estimate and the reservation (a loop's conversation).
        Returns (parsed JSON or None, record). At most DISPATCHES_PER_QUESTION requests are sent for it, shared between
        the retry of a bad answer and the retry of an unreachable engine: after a transport retry a bad answer is not
        retried, and a transport failure on the second request stops the caller. `ceiling` is a lower call cap for this
        question (keeps calls in reserve for a later step); `budget` is the input limit in tokens, estimated on the
        whole request by estimate(): a pack over it is not sent. Characters / 4 of system and user alone was an upper
        bound on the engine's own count on the recorded calls (engine / estimate 0.835 to 0.989 over 966 local calls),
        observed, not guaranteed, so the engine counts are also checked after every call: a missing count, an input over
        `budget` or an answer over the cap stops the caller (RequestFailed), with the response kept in the log. Every
        request's record names what consumed it ("dispatch": first, transport_retry, schema_retry, recovery).
        OpenRouter: every failed attempt's record keeps what came back ("http", http_failure). The question has two
        answer attempts and one recovery of an explicit HTTP 429, a send of its own (question_sends: at most three), so
        invalid → 429 → valid and 429 → invalid → valid both complete. The recovery repeats the same request as the
        transport retry, when rate_limit allows it (the job allowance included): the 429's record ("rate_limit") is
        written and its attempt settled before the wait (wait_429, whose ending has its own bookkeeping record), and the
        recovery's record carries "after_429". A transient failure (HTTP 500/502/503/504, a broken connection, a
        timeout; retry_wait) is resent once (the transport retry) after CHAT_RETRY_WAIT_HOSTED seconds or a longer
        Retry-After that fits the wall time left, the failed attempt charged at its full reservation and the resend
        reserved on its own. A second 429 on the question, a 429 on a resend, every other HTTP error (402 included:
        our bounded policy) and a 200 reply carrying an error object (ErrorReply) are not retried. The stop switch
        during either wait raises Stopped, with nothing more sent."""
        if self.failed:
            raise CapReached(f"caller stopped: {self.failed}")
        check_schema(schema)            # before anything is sent: an answer could not be checked against it whole
        self._turns, self._schema = list(turns or []), schema
        self._at = {"role": role, "tag": tag, "attempt": 0, "dispatch": None}
        self._sent = None
        prior = "".join(t["content"] for t in self._turns)
        vendor = refuse_vendor(system, prior, user, json.dumps(schema))   # the literal vendor guard
        if vendor:
            self.log({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "role": role, "model": self.models[role], "tag": tag,
                      "attempt": 0, "system": system, "user": user, "answer": None, "error": f"{vendor}; not sent"})
            raise RequestFailed(self.fail("not_sent", f"{vendor}; not sent"))
        leaked = tagless.leaks(system, prior, user, json.dumps(schema))   # tags hidden: recorded, the request still sent
        if leaked:      # supplied: in what code sent; the model's own: only in its earlier answers
            own = tagless.leaks("".join(t["content"] for t in self._turns if t["role"] == "assistant"))
            supplied = tagless.leaks(system, user, json.dumps(schema),
                                     "".join(t["content"] for t in self._turns if t["role"] != "assistant"))
            with open(Path(self.log_path).with_name("tag_leaks.jsonl"), "a", encoding="utf8") as f:
                f.write(json.dumps({"role": role, "tag": tag, "names": leaked, "supplied": supplied,
                                    "model_own": sorted(set(own) - set(supplied))}) + "\n")
        est = estimate(system, prior + user, schema)
        if budget is not None and est > budget:
            rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "role": role, "model": self.models[role], "tag": tag,
                   "attempt": 0, "estimated_tokens": est, "system": system, "user": user, "answer": None,
                   "error": f"input over budget: about {est} tokens, budget {budget}; not sent"}
            self.log(rec)
            raise OverBudget(rec["error"])
        cap = min(self.max_calls, ceiling) if ceiling is not None else self.max_calls
        reserved = self.ask_reservation(role, system, user, schema, self._turns) if self.engine == "openrouter" \
            else None
        sent = 0                        # requests sent for this question, shared by both kinds of retry
        most = self.question_sends()    # hosted: the two answer attempts and the one 429 recovery
        dispatches = []                 # what consumed each request sent: first, transport_retry, schema_retry
        rate = None                     # this question's 429 event, when its retry was allowed
        for attempt in (1, 2):
            self._at = {"role": role, "tag": tag, "attempt": attempt, "dispatch": None}
            self._sent = None
            if sent >= most:
                # a transport retry used the second request: the bad answer of attempt 1 is not retried. Marked when
                # that retry followed a 429, so a job can count the answer retries lost to rate limits
                lost = rate is not None and rate["retry_ok"] is True
                if lost:
                    rate["schema_retry_lost"] = True
                self.log({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "role": role, "model": self.models[role],
                          "tag": tag, "attempt": attempt, "answer": None,
                          "error": f"no answer retry: {most} requests per question used",
                          **({"schema_retry_lost_to_429": True} if lost else {})})
                break
            if self.stopped():
                raise Stopped("stop switch")
            if self.calls >= cap:
                raise CapReached(f"call cap {cap}" + (" (reserved for review)" if cap < self.max_calls else ""))
            if self.over_spend(reserved):
                raise CapReached(self.over_spend(reserved))
            if self.role_cap_hit(role, reserved):
                raise CapReached(self.role_cap_hit(role, reserved))
            try:
                req = self.request(role, system, user, schema, self._turns)
            except RequestFailed:
                self.log({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "role": role, "model": self.models[role],
                          "tag": tag, "attempt": attempt, "system": system, "user": user, "answer": None,
                          "error": f"not sent: {self.failed}"})
                raise
            self.calls += 1
            self.role_calls[role] += 1
            sent += 1
            t0 = time.time()
            for transport in (1, 2):
                recovering = rate is not None and transport == 2    # this send is the question's 429 recovery
                if transport == 2:
                    # the one retry: the stop switch, the cap and the per-question limit are checked again just
                    # before it is sent
                    if self.stopped() or self.calls >= cap or sent >= most:
                        why = ("stop switch" if self.stopped() else f"call cap {cap}" if self.calls >= cap
                               else f"{most} requests per question used")
                        self.fail("stopped" if why == "stop switch" else "transport", f"{first!r}; no retry: {why}",
                                  http_status=getattr(first, "code", None))
                        if rate is not None and why == "stop switch":     # a 429 wait ended by the stop switch
                            rate.update(retry=False, reason="stopped")
                            raise Stopped(self.failed) from first
                        raise RequestFailed(self.failed) from first
                    if self.role_cap_hit(role, reserved):      # this role only: the caller goes on
                        raise CapReached(f"{first!r}; no retry: {self.role_cap_hit(role, reserved)}") from first
                    if self.over_spend(reserved):              # the dropped attempt already counted at its reservation
                        raise CapReached(f"{first!r}; no retry: {self.over_spend(reserved)}") from first
                    self.calls += 1
                    self.role_calls[role] += 1
                    sent += 1
                consumer = "transport_retry" if transport == 2 else "first" if attempt == 1 else "schema_retry"
                self._at["dispatch"] = consumer
                rid, timeout = self.reserve_attempt(role, reserved)     # nothing is sent if the ledger refuses
                dispatches.append(consumer)
                out = body = None
                try:
                    body = self._open(req, timeout, role)
                    out = json.loads(body)
                    if self.engine == "openrouter" and isinstance(out, dict) and out.get("error") is not None:
                        code = out["error"].get("code") if isinstance(out["error"], dict) else None
                        raise ErrorReply(f"HTTP 200 with an error object, code {code!r}")
                    text, pt, total, thought_tokens, cost = self.read(out)
                    thought = self.thinking(out)
                    self.settle_attempt(rid, pt, total, cost)
                    if recovering:
                        rate["retry_ok"] = True
                    break
                except Exception as e:
                    # an explicit HTTP 429 from OpenRouter is not billed (its pricing FAQ: a rate-limited request
                    # incurs no token charge): one request sent, zero tokens, zero dollars.
                    # Otherwise the cost of a body that parsed is passed on; the whole reservation is still charged
                    # here (why is set)
                    not_billed = (self.engine == "openrouter" and isinstance(e, urllib.error.HTTPError)
                                  and e.code == 429)
                    if not_billed:
                        self.settle_not_billed(rid)
                    else:
                        self.settle_attempt(rid, cost=reported_cost(out), why=repr(e))
                    unknown = None
                    if self.engine == "openrouter" and not not_billed:     # the Caller's own caps count it too
                        if cost_valid(reported_cost(out)):
                            self.spent_usd += reported_cost(out)
                            self.role_spent[role] += reported_cost(out)
                        else:
                            unknown = reserved
                            self.unknown_usd += reserved
                            self.role_unknown[role] += reserved
                    over = context_exceeded(e) if self.engine == "ollama" else None
                    if over is not None:
                        # charged (self.calls counts it) but local to this question: self.failed is not set
                        why = f"context exceeded: the server counted {over} input tokens, num_ctx {self.num_ctx}"
                        self.context_refusals.append({"tag": tag, "role": role, "server_prompt_tokens": over,
                                                      "estimated_tokens": est})
                        self.log({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "role": role, "model": self.models[role],
                                  "tag": tag, "attempt": attempt, "transport": transport, "dispatch": consumer,
                                  "estimated_tokens": est, "server_prompt_tokens": over,
                                  "system": system, "user": user, "answer": None, "error": why})
                        raise ContextExceeded(why) from e
                    # one retry, inside the call cap, when Ollama could not be reached. OpenRouter: one retry of an
                    # explicit HTTP 429, after its wait (rate_limit), and one of a dropped connection with nothing read
                    # back, so one reset does not end a whole job; the dropped attempt stays charged at its full
                    # reservation above, since whether it was billed is unknown
                    # hosted: one resend of a transient failure (retry_wait: 500/502/503/504, a broken connection, a
                    # timeout), after `wait` seconds that fit the wall time left; a 429 is rate_limit's decision
                    wait = None
                    if self.engine != "openrouter":
                        again = transport == 1 and not isinstance(e, urllib.error.HTTPError) and isinstance(
                            e, (urllib.error.URLError, ConnectionError))
                    else:
                        left = RATE_LIMIT_MAX_WAIT if self._left is None else min(
                            RATE_LIMIT_MAX_WAIT, self._left[0] - (self.clock() - self._left[1]))
                        wait = retry_wait(e, 1, (CHAT_RETRY_WAIT_HOSTED,), left) \
                            if transport == 1 and sent < most else None
                        again = wait is not None
                    http = rl = None
                    if self.engine == "openrouter":     # what came back, bounded and allowlisted
                        key = (req.get_header("Authorization") or "").removeprefix("Bearer ") or None
                        http = http_failure(e, key, body)
                        if isinstance(e, urllib.error.HTTPError) and (e.code == 429 or grammar_limit(e, http)):
                            rl = self.rate_limit(http, sent, cap, role, tag, consumer, recovered=rate is not None,
                                                 resent=transport == 2)
                            again, wait = rl["retry"], None
                    after = recovering                  # this attempt was the recovery after a 429
                    if after:
                        rate["retry_ok"] = False
                    self.log({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "role": role, "model": self.models[role],
                              "tag": tag, "attempt": attempt, "transport": transport, "dispatch": consumer,
                              "seconds": round(time.time() - t0, 2), "system": system, "user": user, "answer": None,
                              "error": f"request failed: {e!r}" + ("; one retry allowed" if again else "")
                              + ("; not billed (HTTP 429)" if not_billed else
                                 "; billing outcome unknown" if self.engine == "openrouter" else ""),
                              **({"cost_usd": 0.0} if not_billed else {}),
                              **({"http": http} if http is not None else {}),
                              **({"unknown_usd": unknown} if unknown is not None else {}),
                              **({"rate_limit": rl} if rl is not None else {}),
                              **({"retry_wait_s": round(wait, 3)} if wait is not None else {}),
                              **({"after_429": dict(rate)} if after else {})})
                    if not again:
                        status = http["status"] if http is not None else getattr(e, "code", None)
                        kind = "response" if body is not None and not isinstance(e, ErrorReply) else "transport"
                        self.fail(kind, repr(e) + ("; not billed (HTTP 429)" if not_billed else
                                                  "; billing outcome unknown" if self.engine == "openrouter" else "")
                                  + (f"; no retry: {rl['reason']}" if rl is not None else "")
                                  + (f" ({rl['allowance']['why']})" if rl is not None and rl["allowance"]
                                     and rl["allowance"].get("why") else ""),
                                  http_status=status, error_code=((http or {}).get("error") or {}).get("code"),
                                  **({"rate_limit_class": rl["class"], "rate_limit_reason": rl["reason"],
                                      "rate_limit_stop": RATE_LIMIT_STOPS.get(rl["reason"])}
                                     if rl is not None else {}))
                        raise RequestFailed(self.failed) from e
                    first = e
                    if rl is None and wait is None:
                        time.sleep(RETRY_WAIT)  # Ollama
                    elif rl is None:            # hosted transient failure: the stop switch ends the wait, nothing sent
                        if self.wait_out(wait)[1]:
                            self.fail("stopped", f"{e!r}; billing outcome unknown; no retry: stopped",
                                      http_status=getattr(e, "code", None))
                            raise Stopped(self.failed) from e
                    else:                       # the 429's record is written and its attempt settled: now the wait
                        self.wait_429(rl, e, http, attempt)
                        rate = rl
            answer, rec, cut = self.judge(role, system, user, schema, tag, attempt, t0, out,
                                          (text, pt, total, thought_tokens, cost, thought), reserved, budget,
                                          self.answer_cap, {"dispatch": consumer, "dispatches": list(dispatches),
                                                            **({"after_429": dict(rate)} if recovering else {}),
                                                            **({"router_metadata": router_metadata(
                                                                out, (req.get_header("Authorization") or "")
                                                                .removeprefix("Bearer ") or None)}
                                                               if self.engine == "openrouter" else {})})
            if answer is not None or cut or self.one_send:
                return answer, rec
        return None, rec

    def judge(self, role, system, user, schema, tag, attempt, t0, out, got, reserved, budget, asked, extra):
        """(answer or None, record, cut) for one response: the engine's counts checked against the budget and
        `asked`, the answer cap this request sent (a violation stops the caller, RequestFailed), then the reply
        parsed and checked against the schema. `got`: (text, prompt, completion, reasoning tokens, cost, reasoning
        text) as read. `extra`: fields added to the record (the dispatch consumer)."""
        text, pt, total, thought_tokens, cost, thought = got
        cost_ok = isinstance(cost, (int, float)) and not isinstance(cost, bool) and math.isfinite(cost) \
            and cost >= 0
        if self.engine == "openrouter" and cost_ok:
            self.spent_usd += cost
            self.role_spent[role] += cost
        at, breakdown, bad = self.visible(total, thought_tokens)
        violation = None
        if not count(pt) or not count(total):
            violation = f"engine counts unavailable: prompt {pt!r}, answer {total!r}"
        elif pt < 0 or bad:
            violation = f"impossible engine counts: prompt {pt!r}, {bad or 'negative'}"
        elif self.reasoning is not None and total > self.max_tokens[role]:
            # per-role only: reasoning and answer together within the limit the role asked for
            violation = f"engine counted {total} completion tokens, limit {self.max_tokens[role]}"
        elif self.engine == "openrouter" and not cost_ok:
            violation = f"engine cost unavailable or invalid: {cost!r}; the spend cap cannot be kept"
        elif self.engine == "openrouter" and cost > reserved:
            violation = f"budget violation: cost {cost} USD over the reservation {reserved:.4f}"
        elif self.role_off(role) and (type(thought_tokens) is not int or thought_tokens != 0):
            # thinking off is shown only by an explicit count of 0; a missing count is unavailable telemetry, not zero
            violation = (f"reasoning off, but the engine counted {thought_tokens!r} reasoning tokens"
                         if count(thought_tokens) else
                         f"reasoning off, but the reasoning count is unavailable or invalid: {thought_tokens!r}")
        elif self.role_off(role) and thought:
            violation = f"reasoning off, but the engine returned reasoning text ({len(thought)} characters)"
        elif self.unpinned(role, out):
            violation = self.unpinned(role, out)
        elif self.wrong_tier(out):
            violation = self.wrong_tier(out)
        elif at > asked and thought_tokens is None and self.engine == "openrouter":
            # the whole completion is only an upper bound on the answer: over the cap, compliance is not shown, not a
            # measured overrun
            violation = (f"visible-cap compliance unverified: completion {total} over cap {asked} with no "
                         f"reasoning count")
        elif at > asked:
            violation = f"engine counted {at} answer tokens, cap {asked}"
        # an input over the budget on the engine's own count ends only this question (charged, answer not used); the
        # checks above still stop the caller
        over = (None if violation or budget is None or pt <= budget
                else f"engine counted {pt} input tokens, budget {budget}; this question only")
        if over:
            self.context_refusals.append({"tag": tag, "role": role, "engine_prompt_tokens": pt, "budget": budget})
            # the whole response record, telemetry included
            self.log({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "role": role, "model": self.models[role],
                      "tag": tag, "attempt": attempt, "seconds": round(time.time() - t0, 2),
                      "prompt_tokens": pt, "completion_tokens": total, "answer_tokens": at,
                      "token_breakdown": breakdown, "cost_usd": cost, "reserved_usd": reserved,
                      "done_reason": self.done(out), "thinking_requested": self.thinking_requested(role),
                      "reasoning": thought, "reasoning_tokens": thought_tokens, "engine_usage": self.usage(out),
                      "served": self.served(out),
                      "system": system, "user": user, "raw": text, "answer": None, "error": over, **extra,
                      "context_refusal": True})
            raise ContextExceeded(over)
        if violation:
            # the response is kept in the log; no request is made after it and the builder exits non-zero
            self.violations.append({"tag": tag, "role": role, "violation": violation})
            self.fail("violation", violation)
            self.log({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "role": role, "model": self.models[role],
                      "tag": tag, "attempt": attempt, "seconds": round(time.time() - t0, 2),
                      "prompt_tokens": pt, "completion_tokens": total, "answer_tokens": at,
                      "token_breakdown": breakdown, "cost_usd": cost, "reserved_usd": reserved,
                      "done_reason": self.done(out), "thinking_requested": self.thinking_requested(role),
                      "reasoning": thought, "reasoning_tokens": thought_tokens, "engine_usage": self.usage(out),
                      "served": self.served(out),
                      "system": system, "user": user, "raw": text, "answer": None, "error": violation, **extra})
            raise RequestFailed(violation)
        self.max_prompt_tokens = max(self.max_prompt_tokens, pt)
        self.max_answer_tokens = max(self.max_answer_tokens, at)
        cut, long = self.done(out) == "length", []
        if not isinstance(text, str) or not text.strip():
            # e.g. reasoning used the whole output limit: a bad answer, retried once like one that fails the schema
            answer, error = None, f"no visible answer: content {text!r}"
        else:
            try:
                answer = json.loads(text)
                errs = schema_errors(answer, schema, long=long)
                error = "; ".join(errs) if errs else None
                if errs:
                    answer = None
            except json.JSONDecodeError as e:
                answer, error = None, f"bad JSON: {e}"
        if cut:
            # generation hit the output limit or the end of the context (shift off): the answer is refused even when it
            # parses and fits the schema, because it may be a valid prefix of a longer one. Not retried, a conservative
            # policy: an output-cap refusal is lost coverage, reported as such, not an established wrong answer
            answer, error = None, "output cut: done_reason length" + (f"; {error}" if error else "; schema-valid")
        rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "role": role, "model": self.models[role],
               "tag": tag, "attempt": attempt, "seconds": round(time.time() - t0, 2),
               "prompt_tokens": pt, "completion_tokens": total, "answer_tokens": at,
               "token_breakdown": breakdown, "cost_usd": cost, "reserved_usd": reserved,
               "done_reason": self.done(out), "thinking_requested": self.thinking_requested(role),
               "reasoning": thought, "reasoning_tokens": thought_tokens, "engine_usage": self.usage(out),
                      "served": self.served(out),
               "system": system, "user": user, "raw": text, "answer": answer, "error": error,
               **({"over_length": long} if long else {}), **extra}
        self.log(rec)
        return answer, rec, cut

    def request(self, role, system, user, schema, turns=()):
        if self.engine == "ollama":
            return urllib.request.Request(OLLAMA, self.payload(role, system, user, schema, turns),
                                          {"Content-Type": "application/json"})
        key = os.environ.get("OPENROUTER_API_KEY") or env_file_value("OPENROUTER_API_KEY")
        if not key:
            raise RequestFailed(self.fail("not_sent", "OPENROUTER_API_KEY is not set"))
        return urllib.request.Request(OPENROUTER, self.payload(role, system, user, schema, turns),
                                      {"Content-Type": "application/json", "Authorization": f"Bearer {key}",
                                       **ROUTER_METADATA_HEADER})

    def payload(self, role, system, user, schema, turns=()):
        """The request body bytes request() sends, built with no key, log, count or reservation (the one builder,
        so the frozen bytes and the sent bytes cannot differ)."""
        messages = [{"role": "system", "content": system}, *turns, {"role": "user", "content": user}]
        if self.engine == "ollama":
            # truncate and shift off, at the top level: a prompt over num_ctx is refused with HTTP 400 and the server's
            # own token count, not cut from the front. Every call sends both: a change of shift reloads the runner
            body = {"model": self.models[role], "stream": False, "format": schema, "think": False,
                    "messages": messages, "truncate": False, "shift": False,
                    "options": self.sent_options(role)}
            return json.dumps(body).encode()
        # require_parameters: only a provider that honours the schema, seed and temperature may serve the call
        price, providers = self.role_price(role), self.role_providers(role)
        body = {"model": self.models[role], "messages": messages, "temperature": 0, "seed": self.seed,
                # sent, not left to provider defaults, as the local engine gets presence_penalty 0; only with reasoning
                # off, so requests with reasoning on are unchanged
                **({"presence_penalty": 0, "frequency_penalty": 0} if self.role_off(role) else {}),
                "max_tokens": self.role_max_tokens(role) or self.answer_cap, "usage": {"include": True},
                "response_format": {"type": "json_schema",
                                    "json_schema": {"name": "answer", "strict": True, "schema": schema}},
                "provider": {"require_parameters": True,
                             # a seat pinned to two or more endpoints may move between them, never beyond them
                             **({"only": list(providers), "allow_fallbacks": len(providers) > 1}
                                if providers else {}),
                             # USD per million tokens; no provider above the declared price may serve the call
                             "max_price": {"prompt": round(price[0] * 1e6, 6),
                                           "completion": round(price[1] * 1e6, 6)}}}
        if self.service_tier is not None:
            body["service_tier"] = self.service_tier
        for k in self.omit.get(role, ()):
            body.pop(k, None)
        if self.role_effort(role):
            # returned, not excluded: excluded reasoning is billed all the same, and the trace keeps it
            body["reasoning"] = {"effort": self.role_effort(role), "exclude": False}
        elif self.role_off(role):
            body["reasoning"] = {"enabled": False}
        return json.dumps(body).encode()

    def read(self, out):
        """(answer text, input tokens, total completion tokens, reasoning tokens, cost in USD), as the engine reports
        them. Ollama runs with think off and reports no reasoning count (None); OpenRouter's count may be missing."""
        if self.engine == "ollama":
            return out["message"]["content"], out.get("prompt_eval_count"), out.get("eval_count"), None, None
        u = out.get("usage") or {}
        return ((out["choices"][0].get("message") or {}).get("content"), u.get("prompt_tokens"),
                u.get("completion_tokens"), (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                u.get("cost"))

    def usage(self, out):
        """The engine's own usage report, kept as returned so local and OpenRouter use can be compared later: Ollama's
        counts and durations (nanoseconds), OpenRouter's usage object (its cached-token and cost details included).
        Recorded only; nothing reads it."""
        if not isinstance(out, dict):
            return None
        if self.engine == "ollama":
            return {k: out.get(k) for k in ("total_duration", "load_duration", "prompt_eval_count",
                                            "prompt_eval_duration", "eval_count", "eval_duration")}
        return out.get("usage")

    def unpinned(self, role, out):
        """OpenRouter: why the serving provider is not one of this role's pinned endpoints, or the serving model not the
        one asked for, or None. The pin is a provider slug with its precision ("deepinfra/bf16"); the response names the
        provider only ("DeepInfra"), so the precision itself is not confirmed by the response. The model must come back
        exactly as requested: no alias is documented, so none is allowed."""
        pins = self.role_providers(role) if self.engine == "openrouter" else None
        if not pins:
            return None
        model = (self.served(out) or {}).get("model")
        if model != self.models[role]:
            return f"served model {model!r}, not the requested {self.models[role]!r}"
        got = (self.served(out) or {}).get("provider")
        if self.provider_names is not None:     # exact names from the frozen listing, nothing else
            if not isinstance(got, str) or got not in {n for p in pins for n in self.provider_names[p]}:
                return f"served by provider {got!r}, not the pinned {sorted(pins)}"
            return None
        names = {p.split("/")[0].lower() for p in pins}
        if not isinstance(got, str) or got.replace(" ", "").lower() not in names:
            return f"served by provider {got!r}, not the pinned {sorted(pins)}"
        return None

    def wrong_tier(self, out):
        """OpenRouter, when this caller declares a service tier: why the response's top-level service_tier is not the
        declared one, or None. A missing tier is unavailable telemetry and stops the caller pending review; any other
        tier is a violation."""
        if self.service_tier is None or self.engine != "openrouter":
            return None
        got = out.get("service_tier") if isinstance(out, dict) else None
        if got is None:
            return f"service tier unavailable: the response returned none (declared {self.service_tier!r}); " \
                   f"stopped pending review"
        if got != self.service_tier:
            return f"served on service tier {got!r}, not the declared {self.service_tier!r}"
        return None

    def served(self, out):
        """OpenRouter: the provider and model that actually served the call, as returned, and the returned top-level
        service_tier (None when missing) when this caller sends one; None locally. Recorded here; tested in unpinned
        and wrong_tier."""
        if self.engine != "openrouter" or not isinstance(out, dict):
            return None
        return {"provider": out.get("provider"), "model": out.get("model"),
                **({"service_tier": out.get("service_tier")} if self.service_tier is not None else {})}

    def visible(self, total, reasoning):
        """(visible answer tokens, how the split was found, what is impossible or None), kept apart from the total and
        the reasoning count. With no reasoning count the whole completion is counted as answer, so the answer cap is
        checked on an upper bound."""
        if not count(total):
            return total, None, None
        if total < 0:
            return total, None, f"completion {total} is negative"
        if reasoning is None:
            return total, ("thinking off: no reasoning count" if self.engine == "ollama"
                           else "reasoning count not reported: answer counted as the whole completion"), None
        if not count(reasoning) or reasoning < 0 or reasoning > total:
            return None, None, f"reasoning {reasoning!r} with completion {total}"
        return total - reasoning, "reported", None

    def thinking_requested(self, role=None):
        """The thinking setting sent with this role's requests (with per-role reasoning) or with every request of
        this caller, recorded on each call."""
        if self.engine == "ollama":
            return {"think": False}
        if self.reasoning is not None:
            return ({"reasoning": {"enabled": False}} if self.role_off(role)
                    else {"reasoning_effort": self.role_effort(role)})
        if self.reasoning_off:
            return {"reasoning": {"enabled": False}}
        return {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else None

    def thinking(self, out):
        """Returned reasoning text or None. Kept in its own log field, never in another model's prompt; written
        reasoning is not taken as a faithful account of the decision."""
        if self.engine == "ollama":
            return (out.get("message") or {}).get("thinking") or None
        msg = (out.get("choices") or [{}])[0].get("message") or {}
        text = msg.get("reasoning")
        if not text and msg.get("reasoning_details"):
            text = "\n".join(d.get("text") or d.get("summary") or "" for d in msg["reasoning_details"]
                             if isinstance(d, dict)).strip()
        return text or None

    def done(self, out):
        if self.engine == "ollama":
            return out.get("done_reason")
        return (out.get("choices") or [{}])[0].get("finish_reason")


class Mixed:
    """Several Callers as one, for roles on different engines (e.g. analyst and checker on Ollama, Haiku on OpenRouter).
    `callers`: role -> its Caller; each question goes to its role's Caller, under that Caller's own caps, settings and
    checks. The callers share one context size (the loop reads it); the first failure of any stops them all."""

    def __init__(self, callers):
        self.callers = dict(callers)
        self.distinct = list({id(c): c for c in self.callers.values()}.values())
        self.models = {r: c.models[r] for r, c in self.callers.items()}
        if len({c.num_ctx for c in self.distinct}) != 1:
            raise ValueError(f"mixed callers need one num_ctx: {[c.num_ctx for c in self.distinct]}")
        self.num_ctx = self.distinct[0].num_ctx
        self.max_calls = max(c.max_calls for c in self.distinct)

    @property
    def calls(self):
        return sum(c.calls for c in self.distinct)

    @property
    def hosted_tokens(self):
        return sum(c.hosted_tokens for c in self.distinct)

    @property
    def hosted_unknown(self):
        return sum(c.hosted_unknown for c in self.distinct)

    @property
    def failed(self):
        return next((c.failed for c in self.distinct if c.failed), None)

    @property
    def failure(self):
        """The structured failure of the caller whose failure `failed` gives (Caller.fail)."""
        return next((c.failure for c in self.distinct if c.failed), None)

    @property
    def rate_limits(self):
        return [ev for c in self.distinct for ev in c.rate_limits]

    def ask(self, role, *a, **k):
        if self.failed:
            raise CapReached(f"caller stopped: {self.failed}")
        return self.callers[role].ask(role, *a, **k)


EMBED_MODEL = "qwen3-embedding:0.6b-fp16"
EMBED_BATCH = 32              # texts per /api/embed request


EMBED_DIMENSIONS = 1024      # qwen3-embedding:0.6b; every vector is checked against it (search.vector_errors)
# per input, added to its UTF-8 bytes for the ledger reservation: the instruction template and special tokens. An
# estimate, not a proven bound; use above it stops the batch (ledger.commit)
EMBED_OVERHEAD = 32


class OllamaEmbedder:
    """Embeddings from the local Ollama server, for search.Search (network calls stay in this module). `truncate`
    is off: an input too long for the model is an error, never silently cut.

    Each request goes through the batch ledger like a chat attempt: reserved as role `embedding` (its texts' UTF-8 bytes
    plus EMBED_OVERHEAD per text), checked by `before_send` for the stop and the deadlines, and committed with the
    engine's own token count (`prompt_eval_count`), or in full when that is not reported. The stop switch is checked
    before every request. Every request is logged in `calls`. A failed request is not retried here (`attempts` 1; the
    hosted embedder retries, see request()); its failure is kept as a string (self.failed) and structured
    (self.failure, as Caller.fail, role `embedding`)."""

    def __init__(self, model=EMBED_MODEL, *, dimensions=EMBED_DIMENSIONS, ledger=None, company=None, stop=None):
        self.model, self.dimensions = model, dimensions
        self.base = OLLAMA.rsplit("/api/", 1)[0]
        self.ledger, self.company, self.stop = ledger, company, stop
        self.calls, self.failed, self.failure = [], None, None

    def fail(self, kind, why, keep=False, **at):
        """As Caller.fail: self.failed and self.failure together (kind `transport`, `response` or `ledger`, role
        `embedding`, the batch being embedded). Returns the failure string."""
        if keep and self.failed:
            return self.failed
        self.failed = why
        self.failure = {"kind": kind, "why": why, "role": "embedding", "tag": None, "attempt": 1, "dispatch": None,
                        "http_status": None, "error_code": None, "batch": None, **at}
        return why

    def stopped(self):
        return STOP_FILE.exists() or (self.stop is not None and self.stop.is_set())

    def reserve(self, texts):
        """(ledger reservation ID or None, timeout in seconds) for one request; CapReached if refused."""
        if self.ledger is None:
            return None, 600
        from .ledger import LimitReached
        tokens = sum(len(t.encode("utf8")) for t in texts) + EMBED_OVERHEAD * len(texts)
        try:
            rid = self.ledger.reserve(self.company, "embedding", tokens, local=True)
        except LimitReached as e:
            raise CapReached(self.fail("ledger", f"ledger: {e}")) from e
        try:
            left = self.ledger.before_send(rid)
        except LimitReached as e:
            self.ledger.release(rid, f"refused before send: {e}")
            raise CapReached(self.fail("ledger", f"ledger: {e}")) from e
        return rid, max(1, min(600, left))

    def send(self, texts, timeout):
        body = json.dumps({"model": self.model, "input": texts, "truncate": False}).encode()
        req = urllib.request.Request(self.base + "/api/embed", body, {"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=timeout).read())

    def digest(self):
        tags = json.loads(urllib.request.urlopen(self.base + "/api/tags", timeout=30).read())["models"]
        d = [m["digest"] for m in tags if m["name"] == self.model]
        if not d:
            raise RuntimeError(f"embedding model not installed in Ollama: {self.model}")
        return d[0]

    attempts, waits = 1, ()     # the local engine: one attempt, no retry

    def wait(self, seconds):
        """Wait `seconds` before a retry; False when the stop switch is set meanwhile."""
        end = time.time() + seconds
        while time.time() < end:
            if self.stopped():
                return False
            time.sleep(min(0.5, max(0.0, end - time.time())))
        return not self.stopped()

    def __call__(self, texts):
        out = []
        for i in range(0, len(texts), EMBED_BATCH):
            if self.failed:
                raise CapReached(f"embedder stopped: {self.failed}")
            if self.stopped():
                raise Stopped("stop switch")
            out += self.request(texts[i:i + EMBED_BATCH], i // EMBED_BATCH)
        return out

    def request(self, batch, b):
        """One batch's embeddings, in up to `attempts` physical requests. Each attempt is reserved, checked by
        before_send, logged in `calls` (attempt number, error and wait included) and charged on its own; a failed one
        in full (commit_unknown). Retried: HTTP 500/502/503/504, a broken connection or a timeout, after `waits`, or a
        longer Retry-After when it fits what is left of the run's wall time. The stop switch is checked after each
        wait, and a reply that comes back on a retry after the stop is charged and logged, not used. Only the last
        failure is kept in self.failed (concurrent batches each keep their own attempt state)."""
        for n in range(1, self.attempts + 1):
            rid, timeout = self.reserve(batch)
            t0 = time.time()
            rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "role": "embedding", "model": self.model,
                   "texts": len(batch), "reservation": rid, "attempt": n}
            try:
                got = self.send(batch, timeout)
            except Exception as e:
                if rid is not None:
                    self.ledger.commit_unknown(rid, f"request failed: {e}")
                status = getattr(e, "code", None)
                wait = retry_wait(e, n, self.waits, timeout - (time.time() - t0)) if n < self.attempts else None
                self.calls.append({**rec, "wall_s": round(time.time() - t0, 3), "error": str(e),
                                   **({"retry_after_s": wait} if wait is not None else {})})
                if wait is not None and self.wait(wait):
                    continue
                # a reply that came back but is not JSON is `response`; unreachable, HTTP error or timeout `transport`
                why = f"embedding request failed: {e}" + (f" (attempt {n} of {self.attempts})" if n > 1 else "")
                self.fail("response" if isinstance(e, json.JSONDecodeError) else "transport", why,
                          http_status=status, attempt=n, batch=b)
                if wait is not None:        # the stop switch was set during the wait
                    raise Stopped("stop switch") from e
                raise RequestFailed(why) from e
            if rid is not None:
                from .ledger import LimitReached
                try:
                    self.settle(rid, got)
                except LimitReached as e:
                    self.fail("ledger", f"ledger: {e}", keep=True, attempt=n, batch=b)
            self.calls.append({**rec, "wall_s": round(time.time() - t0, 3), "tokens": got.get("prompt_eval_count"),
                               **({"cost_usd": got["cost"], "cost_source": got.get("cost_source"),
                                   "served": got.get("provider")} if "cost" in got else {})})
            if n > 1 and self.stopped():
                raise Stopped("stop switch")
            return got.get("embeddings") or []

    def settle(self, rid, got):
        """Commit one request: the engine's own token count, or in full when that is not reported."""
        if count(got.get("prompt_eval_count")):
            self.ledger.commit(rid, tokens=got["prompt_eval_count"])
        else:
            self.ledger.commit_unknown(rid, "prompt_eval_count not reported")


# OpenRouter's embeddings, for hosted runs on a machine without a GPU: OpenAI's small model on its own endpoint, no
# fallback, 1536 dimensions asked for explicitly
OPENROUTER_EMBED = "https://openrouter.ai/api/v1/embeddings"
HOSTED_EMBED = {"model": "openai/text-embedding-3-small", "providers": ["openai"], "dimensions": 1536,
                "price": 0.02e-6}     # USD per input token, the listed price
# hosted embedding requests fail now and then with HTTP 503: at most three attempts per batch, waiting 5 s then 20 s, or a longer Retry-After that fits
EMBED_ATTEMPTS_HOSTED, EMBED_WAITS_HOSTED = 3, (5.0, 20.0)
RETRY_STATUSES = (500, 502, 503, 504)


def retry_wait(e, n, waits, left):
    """Seconds to wait before retrying a failed embedding attempt `n` (1-based), or None when it is not retried: an
    HTTP status outside RETRY_STATUSES, an answer that is not JSON, or a wait that does not fit `left` seconds of wall
    time. Retried: those statuses, a broken connection and a timeout. A Retry-After longer than the default wait is
    honoured."""
    import socket
    status = getattr(e, "code", None)
    if isinstance(e, urllib.error.HTTPError):
        if status not in RETRY_STATUSES:
            return None
    elif isinstance(e, json.JSONDecodeError) or not isinstance(e, (urllib.error.URLError, ConnectionError,
                                                                     TimeoutError, socket.timeout)):
        return None
    wait = waits[min(n, len(waits)) - 1] if waits else 0.0
    hint = retry_after_seconds((getattr(e, "headers", None) or {}).get("Retry-After"), time.time())
    if hint is not None and hint[0] > wait:
        wait = hint[0]
    return wait if wait < left else None


class HostedEmbedder(OllamaEmbedder):
    """OllamaEmbedder's batching, stop switch, logging and failures, sent to OpenRouter and charged in dollars: each
    request reserved at its texts' UTF-8 bytes (no fewer than its tokens) times the price, committed at the reported
    cost, else at the reported tokens times the price, else in full. Its identity is its settings (no weight digest);
    its queries go without the local model's instruction."""

    query_instruction = ""
    attempts, waits = EMBED_ATTEMPTS_HOSTED, EMBED_WAITS_HOSTED

    def __init__(self, settings=HOSTED_EMBED, *, ledger=None, company=None, stop=None):
        super().__init__(settings["model"], dimensions=settings["dimensions"], ledger=ledger, company=company, stop=stop)
        self.settings = settings

    def reserve(self, texts):
        if self.ledger is None:
            return None, 600
        from .ledger import LimitReached
        usd = (sum(len(t.encode("utf8")) for t in texts) + EMBED_OVERHEAD * len(texts)) * self.settings["price"]
        try:
            rid = self.ledger.reserve(self.company, "embedding", 0, usd=usd, local=False)
        except LimitReached as e:
            raise CapReached(self.fail("ledger", f"ledger: {e}")) from e
        try:
            left = self.ledger.before_send(rid)
        except LimitReached as e:
            self.ledger.release(rid, f"refused before send: {e}")
            raise CapReached(self.fail("ledger", f"ledger: {e}")) from e
        return rid, max(1, min(600, left))

    def send(self, texts, timeout):
        key = os.environ.get("OPENROUTER_API_KEY") or env_file_value("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        s = self.settings
        body = json.dumps({"model": s["model"], "input": texts, "dimensions": s["dimensions"],
                           "provider": {"only": s["providers"], "allow_fallbacks": False}}).encode()
        req = urllib.request.Request(OPENROUTER_EMBED, body, {"Content-Type": "application/json",
                                                              "Authorization": f"Bearer {key}"})
        got = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        rows = sorted(got.get("data") or [], key=lambda r: r.get("index", 0))
        usage = got.get("usage") or {}
        cost, source = usage.get("cost"), "reported"
        if not isinstance(cost, (int, float)) and count(usage.get("prompt_tokens")):
            cost, source = usage["prompt_tokens"] * s["price"], "price"     # the reported tokens times the price
        return {"embeddings": [r.get("embedding") for r in rows], "prompt_eval_count": usage.get("prompt_tokens"),
                "cost": cost, "cost_source": source if isinstance(cost, (int, float)) else None,
                "provider": got.get("provider")}

    def settle(self, rid, got):
        """Commit one request at its cost (reported, else computed from the reported tokens), or in full."""
        if isinstance(got.get("cost"), (int, float)):
            self.ledger.commit(rid, tokens=0, usd=float(got["cost"]))
        else:
            self.ledger.commit_unknown(rid, "embedding cost not reported")

    def digest(self):
        return "openrouter:" + json.dumps(self.settings, sort_keys=True)

    @property
    def hosted_tokens(self):        # as OpenRouter reported them, as llm.Caller counts its own
        return sum(c["tokens"] for c in self.calls if count(c.get("tokens")))

    @property
    def hosted_unknown(self):       # requests answered with no reported tokens
        return sum(1 for c in self.calls if "error" not in c and not count(c.get("tokens")))
