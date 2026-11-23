"""What is installed, for the Setup page in Live, and the models each job option uses. Reads only: it downloads, installs and starts nothing, and reports whether
the OpenRouter key is present, never its value. OSV is asked about the locked packages only when a
person presses Check now (advisories_item); until then Setup says "not checked". The key is tested only by
key_test (Test the key), which asks OpenRouter's key record and calls no model.
"""
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spreading import llm, mcp_client, organizer, repair, repair_roles, spread as code, taxonomy  # noqa: E402

# The job settings: the one path Setup, Advanced and each job's preparation read (ui/live.py BASE is this object).
SETTINGS = ROOT / "data/manifests/full_pipeline_monarch_v31.json"  # the default settings for Setup, Advanced and job preparation
# What a person sees for a line-up and a model; the settings and the records keep the codes.
LABELS = {"qwen_tree_minicpm_judge_fb_on": "Recommended: small local models (tested)",
          "hosted_haiku_luna": "Online: Claude Haiku 5.5 analyst, GPT-6 Luna checker (paid, OpenRouter)"}
NAMES = {"qwen3.5:4b-q4_K_M": "Qwen 3.5 4B", "minicpm5:2b-q4_K_M": "MiniCPM 5 2B", "ling3-tiny:q4_K_M": "Ling 3.0 Tiny",
         "qwen3-embedding:0.6b-fp16": "Qwen3 Embedding 0.6B", "anthropic/claude-haiku-5.5": "Claude Haiku 5.5",
         "openai/gpt-6-luna": "GPT-6 Luna", "deepseek/deepseek-v4.1-flash-1017": "DeepSeek V4.1 Flash",
         "openai/text-embedding-3-small": "OpenAI text-embedding-3-small",
         "qwen/qwen3.8-flash": "Qwen 3.8 Flash", "z-ai/glm-5.3-flash": "GLM 5.3 Flash"}
# The kind of model each role needs, by role.
NEEDS = {
    "Organizer": "reads the whole statement and sorts each line into a template section, citing a line or printed label",
    "Analyst": "follows long instructions, reads statement wording and notes, calls the search tool, answers in a fixed "
               "format",
    "Checker": "solves the line separately using filing tools and the required answer format",
    "Tiebreaker": "answers disputed lines blind using filing tools; code compares its answer with the other two",
    "Generator": "proposes evidence-backed placement changes for sections whose code checks fail",
    "Controller": "compares a few short proposals and picks one or stops",
    "Meaning question": "answers focused questions about filing evidence; this setup reuses the analyst model",
    "Critic": "assesses a proposed change against its cited evidence; its opinion may be wrong",
    "Review": "solves the repaired line blind using the checker model; matching answers are required to commit",
    "Embeddings": "an embedding model that turns note passages into vectors for search",
}


def label(lineup):
    return LABELS.get(lineup, lineup)


def flag(args, name):
    """The last value given to `name` in a command's arguments, or None."""
    got = [args[i + 1] for i, x in enumerate(args[:-1]) if x == name]
    return got[-1] if got else None


def roles(args):
    return dict(x.split("=", 1) for i, x in enumerate(args[1:]) if args[i] == "--model" and "=" in x)


def where(model):
    """Where a model runs, from its name as the commands give it: OpenRouter names carry a provider ("anthropic/…")."""
    return "online, paid" if "/" in model else "local"


def offered(settings=SETTINGS, skip=()):
    """The line-ups a job may use, in the settings' order (the Run page offers them in this order and checks the
    first): the settings' line-ups, less those in `skip` (parked) or not built."""
    cfg = json.loads(Path(settings).read_text(encoding="utf8"))
    return {k: v for k, v in cfg["lineups"].items() if k not in skip and not v.get("pending")}


def selected(settings=SETTINGS, skip=(), lineup=None):
    """The line-up Setup checks readiness for: the one named, else the Run page's default (the first offered). A name
    that is not offered is refused."""
    names = list(offered(settings, skip))
    if lineup is None:
        if not names:
            raise ValueError("the settings offer no line-up: each one in the file is parked or not built, so there "
                             "is nothing to check readiness for")
        return names[0]
    if lineup not in names:
        raise ValueError(f"line-up {lineup!r} is not offered; the settings offer {names}")
    return lineup


def models(settings=SETTINGS, skip=()):
    """The configured models of each job option (line-up) offered, by step and role, read from the code's settings and
    the line-up's own arguments; line-ups in `skip` (parked) or not built are left out."""
    cfg = json.loads(Path(settings).read_text(encoding="utf8"))
    offered = {k: v for k, v in cfg["lineups"].items() if k not in skip and not v.get("pending")}
    rows, statements_only = [], set()
    for name, lu in sorted(offered.items()):
        spread_args = [*cfg.get("spread_args", []), *lu.get("spread_args", [])]
        if "--statements-only" in spread_args:     # no note search: no Search row
            statements_only.add(name)
        tree_args = [*(cfg.get("tree") or {}).get("args", []), *lu.get("tree_args", [])]
        if flag(spread_args, "--engine") == "openrouter" or "--hosted-profile" in tree_args:
            prof = code.HOSTED_PROFILES[flag(tree_args, "--hosted-profile") or flag(spread_args, "--hosted-profile")
                                        or "hosted"]
            on = lambda role: (prof.get(code.BORROWED.get(role, role)) or prof["analyst"])["model"]
            third = prof.get("haiku", {}).get("model") if "--third-opinion" in spread_args else None
            rows += [
                (name, "Spread", "Organizer", "sorts eligible statement lines into template sections before placement",
                 on("organizer"), "online"),
                (name, "Spread", "Analyst", "reads each printed line into a template row", prof["analyst"]["model"],
                 "online"),
                (name, "Spread", "Checker", "solves each line again on its own; a dispute when the two differ",
                 prof["checker"]["model"], "online"),
                (name, "Spread", "Tiebreaker", "votes on disputed lines" if third else "off: no third vote in this "
                 "line-up", third or "\u2013", "online" if third else "off"),
                (name, "Tree of Thought", "Generator", "proposes placement changes in a section that fails its checks",
                 on("generator"), "online"),
                (name, "Tree of Thought", "Controller", "ranks the proposals, picks one or stops", on("controller"),
                 "online"),
                (name, "Tree of Thought", "Meaning question", "the generator may ask the analyst what a line means",
                 on("meaning"), "online"),
                (name, "Tree of Thought", "Critic", "assesses proposals that reach this step", on("critic"), "online"),
                (name, "Tree of Thought", "Review", "the spread's checker solves the repaired line again, blind; only "
                 "agreement keeps it", prof["checker"]["model"], "online"),
                (name, "Search", "Embeddings", "finds passages in the filing's notes", llm.HOSTED_EMBED["model"],
                 "online"),
            ]
            continue
        sp, tr = roles(spread_args), roles(tree_args)
        analyst, checker = sp.get("analyst", code.LOOP_MODELS["analyst"]), sp.get("checker", code.LOOP_MODELS["checker"])
        third = (flag(spread_args, "--third-model") or code.THIRD["haiku"]["model"]) if "--third-opinion" in spread_args             else None
        tree = {k: tr.get(k, v) for k, v in repair.ROLE_MODELS.items()}
        rows += [
            (name, "Spread", "Organizer", "sorts eligible statement lines into template sections before placement",
             organizer.MODEL, "local"),
            (name, "Spread", "Analyst", "reads each printed line into a template row", analyst, "local"),
            (name, "Spread", "Checker", "solves each line again on its own; a dispute when the two differ", checker,
             "local"),
            (name, "Spread", "Tiebreaker", "votes on disputed lines" if third else "off: no third vote in this line-up",
             third or "–", where(third) if third else "off"),
            (name, "Tree of Thought", "Generator", "proposes placement changes in a section that fails its checks",
             tree["generator"], "local"),
            (name, "Tree of Thought", "Controller", "ranks the proposals, picks one or stops", tree["controller"],
             "local"),
            (name, "Tree of Thought", "Meaning question", "the generator may ask the analyst what a line means",
             tr.get("meaning", repair_roles.MEANING_MODEL), "local"),
            (name, "Tree of Thought", "Critic", "assesses proposals that reach this step; its verdict is recorded only "
             "and does not guide ranking or selection" if flag(tree_args, "--critic-feedback") == "off" else
             "assesses proposals that reach this step; its verdict and reason go to the generator growing that branch, "
             "as an opinion, not evidence; the controller receives no explicit verdict", tree["critic"], "local"),
            (name, "Tree of Thought", "Review", "the spread's checker solves the repaired line again, blind; only "
             "agreement keeps it", flag(tree_args, "--review-model") or checker, "local"),
            (name, "Search", "Embeddings", "finds passages in the filing's notes", llm.EMBED_MODEL, "local"),
        ]
    rows = [r for r in rows if not (r[0] in statements_only and r[1] == "Search")]
    return {"settings": Path(settings).name, "lineups": sorted(offered), "parked": sorted(skip),
            "labels": {k: label(k) for k in sorted(offered)},
            "rows": [{**dict(zip(("lineup", "step", "role", "does", "model", "where"), r)), "needs": NEEDS.get(r[2]),
                      "name": NAMES.get(r[4], r[4])} for r in rows]}


class NotReady(RuntimeError):
    """An expected not-ready state, said in plain words on the page (no exception name)."""


def item(name, what, fn, fix=None):
    """One Setup row: what was seen, or the reason it is not ready and, when given, how a person fixes it. A check that
    fails is a row, never a crash."""
    how = {"fix": fix} if fix else {}
    try:
        return {"item": name, "what": what, "ready": True, "seen": fn()}
    except mcp_client.Fatal as e:      # "not_yet": expected before the first filing, not a gap
        return {"item": name, "what": what, "ready": False, "problem": e.msg, **how,
                **({"later": True} if e.code == "not_yet" else {})}
    except NotReady as e:
        return {"item": name, "what": what, "ready": False, "problem": str(e), **how}
    except Exception as e:  # noqa: BLE001  (a missing server or file shows on the page as not ready)
        return {"item": name, "what": what, "ready": False, "problem": f"{type(e).__name__}: {e}", **how}


KEY_URL = "https://openrouter.ai/api/v1/key"   # the key's own record; no model is called
KEY_ITEM = ("OpenRouter key", "the key for OpenRouter, where these online models run; never shown")


def present_key():
    """The key a job would use, in the jobs' own order (spreading/llm.py): the server's environment, then .env."""
    return os.environ.get("OPENROUTER_API_KEY") or llm.env_file_value("OPENROUTER_API_KEY")


KEY_SHAPE = re.compile(r"sk-or-[A-Za-z0-9_-]{16,200}")
# the SEC's form: a name, then a contact email ("Sample Company Name AdminContact@<sample company domain>.com")
UA_SHAPE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 .,&'()-]{0,150} [A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def env_first(name):
    """Refuses a save the code would not use: the server's environment sets `name`, and it is read before .env."""
    if os.environ.get(name):
        raise ValueError(f"the server's environment sets {name}, and it is used before .env, so a value saved here "
                         "would not be used. Remove it from the environment, start the server again, then save it "
                         "here. Nothing was saved")


def save_env_line(name, value, path=None):
    """Writes NAME=value into the project's .env (llm.ENV_FILE), replacing an earlier NAME line and keeping every
    other line and the file's line endings. The file is written whole to a temporary file beside it and then swapped
    in, so a failed write leaves the old one. Refused when the server's environment sets NAME: the code reads the
    environment before .env, so a value saved here would not be used. No refusal names the value."""
    env_first(name)
    p = Path(path) if path else llm.ENV_FILE
    old = p.read_bytes().decode("utf8") if p.is_file() else ""
    nl = "\r\n" if "\r\n" in old else "\n"
    lines, done = [], False
    for line in old.splitlines():
        if line.strip().partition("=")[0].strip() == name:
            if not done:
                lines.append(f"{name}={value}")
                done = True
        else:
            lines.append(line)
    if not done:
        lines.append(f"{name}={value}")
    tmp = p.with_name(p.name + ".saving")
    tmp.write_bytes((nl.join(lines) + nl).encode("utf8"))
    os.replace(tmp, p)


def save_key(key, path=None):
    """Setup's Save the key: OPENROUTER_API_KEY=<key> into .env (save_env_line). The key is never logged, shown or
    returned. Refused: anything not shaped like an OpenRouter key, and a key set in the server's environment."""
    env_first("OPENROUTER_API_KEY")
    key = key.strip() if isinstance(key, str) else ""
    if not KEY_SHAPE.fullmatch(key):
        raise ValueError("this is not shaped like an OpenRouter API key (it starts with sk-or- and has only letters, "
                         "digits, - and _). Nothing was saved")
    save_env_line("OPENROUTER_API_KEY", key, path)


def save_ua(ua, path=None):
    """Setup's Save on the SEC User-Agent row: EDGAR_USER_AGENT=<name and email> into .env (save_env_line), which
    spreading/mcp_client.py reads after the server's environment. Refused: anything not a name followed by an email,
    and a User-Agent set in the server's environment."""
    env_first("EDGAR_USER_AGENT")
    ua = " ".join(ua.split()) if isinstance(ua, str) else ""
    if not UA_SHAPE.fullmatch(ua):
        raise ValueError("the SEC asks for a name followed by a contact email, for example: Jane Doe "
                         "jane.doe@example.com. Nothing was saved")
    save_env_line("EDGAR_USER_AGENT", ua, path)


UA_ITEM = ("SEC User-Agent", "the name and contact email the SEC asks every automated client to send; not shown")


def ua_where():
    """Where the User-Agent the EDGAR client will send comes from, in its own order (mcp_client.user_agent); never
    the value. Raises when none is set."""
    if os.environ.get("EDGAR_USER_AGENT", "").strip():
        return "set in the server's environment"
    if llm.env_file_value("EDGAR_USER_AGENT"):
        return "set in the file .env"
    if mcp_client.user_agent():
        return "set in your Windows user environment"
    raise NotReady("missing: no EDGAR_USER_AGENT in the server's environment, in .env or in your Windows user "
                   "environment; filings cannot be fetched from the SEC without it")


def key_digest(key):
    """Which key was tested, kept in memory only: a changed key is untested again. Never shown or written."""
    return hashlib.sha256(key.encode("utf8")).hexdigest()


def key_get(key):
    """(HTTP status, parsed body or None) of OpenRouter's key record for `key`. Only the Authorization header carries
    the key; nothing of the answer but the fields key_test reads is kept."""
    req = urllib.request.Request(KEY_URL, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, None


def key_test():
    """Setup's Test the key: asks OpenRouter whether it accepts the key, without calling a model (free). Kept: works or
    not, the key's own spending cap left, the time asked, and a digest of the key tested. Not kept: the key, its label,
    its spending or the account. Accepted means the key authenticates; it does not show the account has money for a job.
    A refusal (401, 403) is "not ready"; a limit, a server error or no answer is "could not verify"."""
    asked = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    key = present_key()
    if not key:
        return {"item": KEY_ITEM[0], "what": KEY_ITEM[1], "checked": asked, "state": "missing", "ready": False,
                "problem": "missing: nothing to test", "fix": FIX[KEY_ITEM[0]]}
    base = {"item": KEY_ITEM[0], "what": KEY_ITEM[1], "checked": asked, "digest": key_digest(key)}
    try:
        code, body = key_get(key)
    except Exception as e:  # noqa: BLE001  (no answer: not a verdict on the key)
        return {**base, "ready": True, "untested": True, "state": "not verified",
                "seen": f"present; could not verify on {asked}: {type(e).__name__}"}
    if code in (401, 403):
        return {**base, "ready": False, "state": "refused",
                "problem": f"OpenRouter refused the key on {asked} (HTTP {code}): it is wrong, disabled or revoked"}
    data = (body or {}).get("data") if code == 200 else None
    if not isinstance(data, dict):
        return {**base, "ready": True, "untested": True, "state": "not verified",
                "seen": f"present; could not verify on {asked}: HTTP {code}"}
    if data.get("is_management_key"):
        return {**base, "ready": False, "state": "refused",
                "problem": "this is a management key: it cannot run models. Use an ordinary API key"}
    left = data.get("limit_remaining")
    cap = ("no spending cap of its own" if data.get("limit") is None else
           f"its own spending cap has ${left:.2f} left" if isinstance(left, (int, float)) else "it has a spending cap")
    return {**base, "ready": True, "state": "accepted",
            "seen": f"accepted by OpenRouter on {asked} (no model called, no cost); {cap}. This does not show the "
                    "account has credit for a job"}


def not_needed(name, what, why):
    """A Setup row the selected line-up does not use: nothing is asked for it, and it is neither ready nor missing."""
    return {"item": name, "what": what, "ready": None, "needed": False, "seen": f"not needed for this line-up: {why}"}


def ollama(path):
    base = llm.OLLAMA.rsplit("/api/", 1)[0]
    return json.loads(urllib.request.urlopen(base + path, timeout=10).read())


OSV = "https://api.osv.dev/v1/querybatch"


def osv_post(body):
    req = urllib.request.Request(OSV, json.dumps(body).encode("utf8"), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=10).read()


def advisories(out=None):
    """Known advisories for the locked packages: one OSV batch query of the exact name==version pairs of the locks the
    agents' environment names (its own and the EDGAR server's); only names and versions are sent. An advisory found, or
    an answer that is not complete, is a row that is not ready."""
    locked = {}
    for name in mcp_client.load_env()["locks"]:
        locked |= mcp_client.lock_versions((ROOT / name).read_text(encoding="utf8"))
    pairs = sorted(locked.items())
    if not pairs:        # nothing asked is never "no advisories"
        raise mcp_client.Fatal("OSV", "Not checked: the locks name no package version")
    when = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    try:
        res = json.loads(osv_post({"queries": [{"package": {"ecosystem": "PyPI", "name": n}, "version": v}
                                               for n, v in pairs]}))["results"]
    except Exception as e:  # noqa: BLE001  (unreachable, not JSON, or no results)
        raise mcp_client.Fatal("OSV", f"Not checked completely: {type(e).__name__}: {e}") from None
    if not isinstance(res, list) or len(res) != len(pairs):
        raise mcp_client.Fatal("OSV", f"Not checked completely: {len(pairs)} packages asked, the answer does not match")
    found, packages = [], []
    for (n, v), r in zip(pairs, res):
        vulns = r.get("vulns", []) if isinstance(r, dict) else None
        if isinstance(r, dict) and "next_page_token" in r:
            raise mcp_client.Fatal("OSV", f"Not checked completely: OSV's answer for {n} has more pages")
        if not isinstance(vulns, list) or not all(isinstance(x, dict) and x.get("id") for x in vulns):
            raise mcp_client.Fatal("OSV", f"Not checked completely: OSV's answer for {n} is malformed")
        packages.append({"name": n, "version": v, "advisories": [x["id"] for x in vulns]})
        if vulns:
            found.append(f"{n}=={v}: " + ", ".join(f"{x['id']} (https://osv.dev/vulnerability/{x['id']})"
                                                     for x in vulns))
    if out is not None:  # each package's own answer, for the page's list
        out["packages"] = packages
    if found:
        raise mcp_client.Fatal("OSV", f"OSV reported advisories, checked {when}: " + "; ".join(found))
    return f"OSV reported no matching advisories for these {len(pairs)} locked versions, checked {when}"


# How a person fixes each item that is not ready (the README's platform section and the code that checks it). This page
# never installs a package or a program; it downloads only an item whose own button names it (Load).
FIX = {
    "Agents' Python environment": "A person creates .venv-agents with CPython 3.13 and installs the locks that "
                                  "agents_env.json names, with their hashes; this page never installs packages. Then "
                                  "press Check again.",
    "EDGAR search server": "A person creates .venv and installs the pinned edgar-search-mcp release wheel from "
                           "requirements.lock, checked against mcp_pin.json; this page never installs it. Then press "
                           "Check again.",
    "SEC User-Agent": "Type your name and contact email in the field below (for example: Jane Doe "
                      "jane.doe@example.com) and press Save: it is written to the file .env in the project folder. "
                      "Or set EDGAR_USER_AGENT in the server's environment and start the server again.",
    "Ollama": "Install Ollama on this computer and start it; Setup asks it at {ollama}. This page does not install "
              "it. Then press Check again.",
    "Local models": "Once Ollama answers (its step above), each missing model that Setup may download has its own "
                    "button here, which names it and downloads it from Ollama's library when pressed. A model imported "
                    "into Ollama by hand is named under the table as not offered: Setup cannot load it, and the "
                    "project records no recipe that restores it yet; it is imported into Ollama by hand.",
    "FASB taxonomy": "Each pinned edition has its own button here, which names it and downloads it from "
                     "xbrl.fasb.org when pressed (its size is shown). Choose and fetch on Companies and years also "
                     "loads the edition its 10-K uses when that edition is pinned and missing.",
    "OpenRouter key": "Paste the key in the field below and press Save the key: it is written to the file .env in "
                      "the project folder and tested, and the page never shows it again. Or add the line "
                      "OPENROUTER_API_KEY=<your key> to that file yourself, then press Check again.",
}


def status(settings=SETTINGS, skip=(), lineup=None, tested=None):
    """Every Setup item in page order, for the selected line-up (`lineup`, else the Run page's default), numbered as
    steps: an item the line-up does not use is "not needed" and is not asked (an online line-up never asks Ollama).
    Each item that is not ready says how to fix it. Missing models are listed by name; their download sizes come with
    Load. The key is reported present or missing, never its value, and is never tested here: `tested`, the row of the
    last Test the key, is shown instead while the key present is the one it tested."""
    chosen = selected(settings, skip, lineup)
    rows = [r for r in models(settings, skip)["rows"] if r["lineup"] == chosen]
    need = sorted({r["model"] for r in rows if r["where"] == "local"})
    online = any(r["where"] not in ("local", "off") for r in rows)

    def client():
        seen = mcp_client.check_client(mcp_client.load_pin())
        # the locked count, as the advisories card counts it; pip is installed and not locked
        n = seen.get("locked", seen["distributions"])
        return f"Python {seen['client_version']}, {n} packages as locked" + (
            f" (pip, installed and not locked, aside)" if seen["distributions"] > n else "")

    def server():
        seen = mcp_client.check_install(mcp_client.load_pin())
        return f"edgar-search-mcp {seen['edgar_search_mcp']}, {seen['files_checked']} files as released"

    def engine():
        return f"Ollama {ollama('/api/version')['version']}"

    def installed():
        have = {m["name"] for m in ollama("/api/tags")["models"]}
        missing = [m for m in need if m not in have]
        if missing:
            raise NotReady("not installed: " + ", ".join(missing))
        return f"all {len(need)} installed"

    def fasb():
        years = sorted(taxonomy.editions())
        if not years:
            raise mcp_client.Fatal("not_yet", "none on disk yet. This is expected before your first filing: Choose "
                                   "and fetch on Companies and years downloads the edition its 10-K uses, and this "
                                   "row then reads ready (press Check again to see it). Or load one with its button.")
        return "editions on disk: " + ", ".join(years)

    def key():          # looked up only for a line-up with an online model; never shown, never sent anywhere
        if present_key():
            return "present, not tested yet: press Test the key (free, no model is called)"
        raise NotReady("missing: no OPENROUTER_API_KEY in the server's environment and none in .env; this line-up "
                       "sends requests to OpenRouter and cannot run without it")

    def check(name, what, fn):
        return {**item(name, what, fn, FIX[name].format(ollama=llm.OLLAMA.rsplit("/api/", 1)[0])), "needed": True}

    items = [check("Agents' Python environment", "the packages the agents run with, against their locks", client),
             check("EDGAR search server", "the MCP server (the pinned edgar-search-mcp release) that reads SEC filings",
                   server),
             check(*UA_ITEM, ua_where)]
    if need:            # an online line-up lists no Ollama step at all (fewer steps)
        items += [check("Ollama", "runs the local models on this computer", engine),
                  check("Local models", f"the {len(need)} local models this line-up uses (its rows in the models table below)", installed)]
    items.append(check("FASB taxonomy", "US GAAP, the edition each filing uses", fasb))
    k = present_key() if online else None
    if online and tested and k and tested.get("digest") == key_digest(k):
        items.append({**{x: v for x, v in tested.items() if x != "digest"}, "needed": True})
    else:
        items.append({**check(*KEY_ITEM, key), "untested": True} if online else
                     not_needed("OpenRouter key", "the key for OpenRouter, used only by a line-up with an online model",
                                "every model of this line-up runs on this computer; the key is not looked up"))
    return [{"step": i, **x} for i, x in enumerate(items, 1)]


ADVISORIES = ("Package advisories", "known advisories (OSV) for the exact package versions the locks pin")


def advisories_item():
    """The OSV row, asked only when a person presses Check now (it sends the locked names and versions to OSV); until
    then the page shows it as not checked. The time asked is kept with the answer, whatever it is."""
    asked = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    out = {}
    return {**item(*ADVISORIES, lambda: advisories(out)), "checked": asked, "packages": out.get("packages", [])}
