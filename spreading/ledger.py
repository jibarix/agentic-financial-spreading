"""One budget ledger per batch (every loop has a cost cap).

Every company process of a batch writes to the same ledger folder: `state.json` holds the totals, `events.jsonl`
every reservation, attempt, commit and release. Every transport attempt, retries included, is reserved before it is
sent. `before_send` rechecks the stop and the deadlines immediately before each attempt, counts it, and returns the
time left for its timeout. The reservation is then committed with the engine's own counts, or committed as unknown
(counted in full) when an attempt went out and its use is not known. A reservation never sent is released.
Reserving checks every limit with what is already reserved counted in, so calls kept for reviews still to come
cannot be spent by the analyst. A limit, once reached, stops the whole batch: the ledger records which limit and
refuses every later reservation and attempt. Use above a reservation also stops the batch: a reservation is the
most an attempt can use (context plus output cap, or the price cap), so going over it means a fault.

Counts: requests actually sent (retries and resends each count), per role (analyst, reviewer, generator, critic,
controller, reference, embedding), local tokens (input plus output), OpenRouter spend. Limits: dispatches, local
tokens, spend, batch wall time, and wall time per company. Limits, reservations and commits are finite and
nonnegative; request and token counts are integers.

Processes take turns through a lock file created with O_EXCL; a lock left by a dead process is not broken by code,
it stops the batch with a message (a person removes it after checking).

A plan ledger holds a whole plan's limits across its stages: opened once by
`python -m spreading.ledger open <folder> ...`, then named by each stage's run (`--ledger <folder>`). The run keeps its
own ledger with its own caps, and `Both` counts every attempt in the two, so the nearer limit refuses first. A stage
ledger, opened the same way, holds one stage's limits across its runs (a release drafted and redrafted in several
invocations); `Both` then counts every attempt in the run's, the stage's and the plan's ledgers.

A ledger may also hold role limits, stored with it: role -> {"calls", "spend_usd"} (the tiebreaker's: 80 requests, and
$0.08 for Haiku on OpenRouter, spread.THIRD_CAPS; "haiku" is the tiebreaker role's record name, whatever model fills
it). A role's limit, with what is reserved for that role counted in, refuses that role only (RoleLimitReached,
recorded): the batch is not stopped and the other roles go on. A company ledger opened with the tiebreaker's limit and
named by every statement run of one company job (`--ledger`) holds it across the runs and their processes; each run's
own ledger records that run's delta.

A ledger also holds the job allowance for HTTP 429 recoveries per company key (RATE_LIMIT_ALLOWANCE: recoveries and
cumulative planned wait). The company ledger, named by every child of a company job, holds it across the children and
their processes; `claim_429` grants or refuses one recovery.
"""
import argparse
import contextlib
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path

LIMITS = ("dispatches", "local_tokens", "spend_usd", "batch_wall_s", "company_wall_s")
LOCK_WAIT_S = 30
# the job allowance for recoveries of an explicit OpenRouter HTTP 429, per company key: bounded allowances, not
# measured provider limits. A recovery claims its planned wait when granted (claim_429); a refusal ends that question,
# not the batch
RATE_LIMIT_ALLOWANCE = {"recoveries": 5, "wait_s": 300.0}


class LimitReached(Exception):
    """A ledger limit was reached; nothing more is sent in this batch. The message names the limit."""


class RoleLimitReached(LimitReached):
    """A role's own limit was reached: that role is refused; the batch goes on."""


class Ledger:
    def __init__(self, folder, limits=None, clock=time.time, role_limits=None, role_groups=None):
        """Open the batch ledger in `folder`, creating it with `limits` (and `role_limits`, if given) if new. An
        existing ledger keeps its own limits; different limits passed for it are refused, so two processes cannot run
        one batch on two budgets. `role_groups` (role -> group) counts a role under its group's role limit, for an
        allocation per model across roles (e.g. {"generator": "analyst"}); `by_role` still counts each role."""
        self.dir = Path(folder)
        self.clock = clock
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.lock():
            p = self.dir / "state.json"
            if p.exists():
                s = json.loads(p.read_text(encoding="utf8"))
                if limits is not None and limits != s["limits"]:
                    raise ValueError(f"ledger {self.dir} has limits {s['limits']}, not {limits}")
                if role_limits is not None and role_limits != s.get("role_limits", {}):
                    raise ValueError(f"ledger {self.dir} has role limits {s.get('role_limits', {})}, "
                                     f"not {role_limits}")
                if role_groups is not None and role_groups != s.get("role_groups", {}):
                    raise ValueError(f"ledger {self.dir} has role groups {s.get('role_groups', {})}, "
                                     f"not {role_groups}")
            else:
                check_limits(limits)
                s = {"limits": limits, "started": self.clock(), "company_started": {}, "sent": 0, "by_role": {},
                     "local_tokens": 0, "spend_usd": 0.0, "reserved": {}, "stopped": None,
                     "rate_limit_allowance": dict(RATE_LIMIT_ALLOWANCE)}
                if role_limits is not None:
                    check_role_limits(role_limits)
                    s["role_limits"] = role_limits
                if role_groups is not None:
                    check_role_groups(role_groups, role_limits or {})
                    s["role_groups"] = role_groups
                self._save(s)
                self._event({"event": "open", "limits": limits,
                             **({"role_limits": role_limits} if role_limits is not None else {}),
                             **({"role_groups": role_groups} if role_groups is not None else {})})
        self.role_limits = s.get("role_limits", {})
        self.role_groups = s.get("role_groups", {})

    def group(self, role):
        """The name a role's limit is counted under: its group, or the role itself."""
        return self.role_groups.get(role, role)

    # ---- file plumbing ----

    def lock(self):
        ledger = self

        class Lock:
            def __enter__(self):
                t0 = time.time()
                while True:
                    try:
                        self.fd = os.open(ledger.dir / "lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                        os.write(self.fd, str(os.getpid()).encode())
                        return self
                    except (FileExistsError, PermissionError):  # Windows: a lock being deleted refuses access
                        if time.time() - t0 > LOCK_WAIT_S:
                            raise LimitReached(f"ledger lock held over {LOCK_WAIT_S} s ({ledger.dir / 'lock'}); "
                                               "a person checks the holder and removes it")
                        time.sleep(0.05)

            def __exit__(self, *exc):
                os.close(self.fd)
                os.remove(ledger.dir / "lock")
        return Lock()

    def _load(self):
        return json.loads((self.dir / "state.json").read_text(encoding="utf8"))

    def _save(self, s):
        tmp = self.dir / f"state.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(s, indent=1), encoding="utf8")
        os.replace(tmp, self.dir / "state.json")

    def _event(self, rec):
        with open(self.dir / "events.jsonl", "a", encoding="utf8") as f:
            f.write(json.dumps({"time": self.clock(), "pid": os.getpid(), **rec}) + "\n")

    def _stop(self, s, why):
        if s["stopped"] is None:
            s["stopped"] = why
            self._event({"event": "stopped", "why": why})
        self._save(s)
        raise LimitReached(why)

    # ---- the budget ----

    def state(self):
        with self.lock():
            return self._load()

    def _wall(self, s, company):
        """Seconds left before the nearer of the batch and company deadlines, and the deadline passed, if any."""
        now, lim = self.clock(), s["limits"]
        left = {"batch_wall_s": lim["batch_wall_s"] - (now - s["started"]),
                "company_wall_s": lim["company_wall_s"] - (now - s["company_started"].get(company, now))}
        return min(left.values()), next((k for k in LIMITS if k in left and left[k] <= 0), None)

    def _over(self, s, extra=None):
        """Limits passed by actual use plus every outstanding reservation (plus `extra`, a reservation being asked)."""
        held = list(s["reserved"].values()) + ([extra] if extra else [])
        want = {"dispatches": s["sent"] + sum(r["requests"] for r in held),     # sent ones are committed later
                "local_tokens": s["local_tokens"] + sum(r["tokens"] for r in held if r["local"]),
                "spend_usd": s["spend_usd"] + sum(r["usd"] for r in held)}
        return [(k, want[k]) for k in ("dispatches", "local_tokens", "spend_usd") if want[k] > s["limits"][k]]

    def _role_over(self, s, role, usd=0.0, requests=1):
        """Why the role's own limit refuses `requests` more attempts costing at most `usd`, with what is already
        sent, spent and reserved for the role counted in; None if it does not, or the role has no limit."""
        g = self.group(role)
        lim = self.role_limits.get(g)
        if lim is None:
            return None
        name = role if g == role else f"{g} (with {role})"
        held = [r for r in s["reserved"].values() if self.group(r["role"]) == g]
        sent = sum(n for x, n in s["by_role"].items() if self.group(x) == g)
        held_n = sum(r["requests"] for r in held)
        if sent + held_n + requests > lim["calls"]:
            return f"{name} call limit {lim['calls']}: {sent} sent, {held_n} reserved"
        spent, held_usd = s.get("role_usd", {}).get(g, 0.0), sum(r["usd"] for r in held)
        if spent + held_usd + usd > lim["spend_usd"]:
            return (f"{name} spend limit {lim['spend_usd']} USD: {spent:.6f} spent, {held_usd:.6f} reserved, "
                    f"{usd:.6f} asked")
        return None

    def role_refusal(self, role, usd=0.0):
        """Why the role's own limit would refuse one more attempt costing at most `usd`, or None; nothing is
        reserved or recorded. A role without a limit is never refused here."""
        if self.group(role) not in self.role_limits:
            return None
        with self.lock():
            return self._role_over(self._load(), role, usd)

    def reserve(self, company, role, tokens, usd=0.0, requests=1, local=True):
        """Reserve `requests` transport attempts of `role` for `company`, with their most tokens (local engine: the
        context plus the output cap) or most cost (hosted). Every attempt, a retry included, is reserved before it
        is sent. Returns the reservation ID. Raises LimitReached, recorded, when any limit would be passed."""
        count(tokens, "tokens")
        count(requests, "requests", least=1)
        amount(usd, "usd")
        with self.lock():
            s = self._load()
            if s["stopped"]:
                raise LimitReached(s["stopped"])
            s["company_started"].setdefault(company, self.clock())
            r = {"company": company, "role": role, "tokens": tokens if local else 0, "usd": usd,
                 "requests": requests, "local": local, "sent": 0}
            _, late = self._wall(s, company)
            over = ([(late, "deadline passed")] if late else []) + self._over(s, r)
            if over:
                self._stop(s, f"limit reached: {over[0][0]} ({over[0][1]}), asked by {company}/{role}")
            why = self._role_over(s, role, usd, requests)
            if why:                         # this role only: the batch goes on
                self._save(s)
                self._event({"event": "role_refused", "company": company, "role": role, "why": why})
                raise RoleLimitReached(f"role limit reached: {why}, asked by {company}/{role}")
            rid = uuid.uuid4().hex[:12]
            s["reserved"][rid] = r
            self._save(s)
            self._event({"event": "reserve", "id": rid, **r})
            return rid

    def before_send(self, rid):
        """Called immediately before each transport attempt of a reservation. Refuses when the batch has stopped,
        a deadline has passed or the reservation's attempts are used up; otherwise counts the attempt as sent (so
        an attempt that fails without a reply is still counted) and returns the seconds left before the nearer
        deadline, which the caller uses as the request's timeout."""
        with self.lock():
            s = self._load()
            return self._sent(s, rid, *self._send_check(s, rid))

    def _sent(self, s, rid, r, left):
        r["sent"] += 1
        self._save(s)
        self._event({"event": "send", "id": rid, "attempt": r["sent"], "seconds_left": left})
        return left

    def _send_check(self, s, rid):
        r = s["reserved"].get(rid)
        if r is None:
            raise ValueError(f"no open reservation {rid}")
        if s["stopped"]:
            raise LimitReached(s["stopped"])
        left, late = self._wall(s, r["company"])
        if late:
            self._stop(s, f"limit reached: {late} (deadline passed), before sending {rid}")
        if r["sent"] >= r["requests"]:
            self._stop(s, f"unreserved attempt: {rid} has used its {r['requests']} attempt(s)")
        return r, left

    def commit(self, rid, tokens=0, usd=0.0):
        """The reservation's attempts are over, with the engine's own token count and cost; the requests sent are
        the attempts before_send counted. Use above the reservation is kept as counted (never cut) and stops the
        batch; so does actual use plus the outstanding reservations passing a limit."""
        count(tokens, "tokens")
        amount(usd, "usd")
        with self.lock():
            s = self._load()
            r = s["reserved"].pop(rid)
            self._count(s, r, r["sent"], tokens if r["local"] else 0, usd)
            over_res = (r["local"] and tokens > r["tokens"]) or usd > r["usd"]
            self._event({"event": "commit", "id": rid, "sent": r["sent"], "tokens": tokens, "usd": usd,
                         "over_reservation": over_res})
            if over_res:
                self._stop(s, f"reservation exceeded: {rid} used {tokens} tokens of {r['tokens']}, "
                              f"{usd} USD of {r['usd']}")
            over = self._over(s)
            if over:
                self._stop(s, f"limit passed after commit: {over[0][0]} ({over[0][1]})")
            self._save(s)

    def commit_unknown(self, rid, why):
        """An attempt went out but its use is unknown (a reset, a timeout, no usage in the reply). The whole
        reservation is counted as used, the conservative outcome; it is never released as unsent."""
        with self.lock():
            s = self._load()
            r = s["reserved"].pop(rid)
            sent = max(r["sent"], 1)
            self._count(s, r, sent, r["tokens"], r["usd"])
            self._event({"event": "commit_unknown", "id": rid, "why": why, "sent": sent, "tokens": r["tokens"],
                         "usd": r["usd"]})
            over = self._over(s)
            if over:
                self._stop(s, f"limit passed after commit: {over[0][0]} ({over[0][1]})")
            self._save(s)

    def release(self, rid, why):
        """The reservation was never sent (stop switch, a check before dispatch); nothing is counted. A reservation
        with an attempt already counted is not released: it is committed, or committed as unknown."""
        with self.lock():
            s = self._load()
            r = s["reserved"].get(rid)
            if r is not None and r["sent"]:
                raise ValueError(f"{rid} has {r['sent']} attempt(s) sent; commit it, do not release it")
            s["reserved"].pop(rid, None)
            self._save(s)
            self._event({"event": "release", "id": rid, "why": why, "found": r is not None})

    def claim_429(self, company, role, wait_s):
        """One recovery of an explicit HTTP 429 for `company`, waiting `wait_s`, against the job allowance
        (RATE_LIMIT_ALLOWANCE; a ledger made before it has the same): (None, what the company has used with this
        claim) when granted, counted at once with its planned wait; (reason, used) when refused, reason
        "job_allowance_recoveries" or "job_allowance_wait" (`why` names the figures). Both recorded; a refusal does
        not stop the batch. The recovery's send still takes its own reservation."""
        amount(wait_s, "wait_s")
        company = str(company)          # the key as JSON keeps it
        with self.lock():
            s = self._load()
            return self._claimed(s, company, role, wait_s, self._claim_check(s, company, wait_s))

    def _claim_check(self, s, company, wait_s):
        return allowance_refusal(s.get("rate_limit_used", {}).get(company, {"recoveries": 0, "wait_s": 0.0}), wait_s,
                                 s.get("rate_limit_allowance", RATE_LIMIT_ALLOWANCE))

    def _claimed(self, s, company, role, wait_s, refused):
        used = s.setdefault("rate_limit_used", {}).setdefault(company, {"recoveries": 0, "wait_s": 0.0})
        if refused is None:
            used["recoveries"] += 1
            used["wait_s"] = round(used["wait_s"] + wait_s, 3)
            self._save(s)
        self._event({"event": "rate_limit_claim", "company": company, "role": role, "wait_s": wait_s,
                     "granted": refused is None, **({"reason": refused[0], "why": refused[1]} if refused else {}),
                     "used": dict(used)})
        return (None if refused is None else refused[0]), {**dict(used), **({"why": refused[1]} if refused else {})}

    def _count(self, s, r, sent, tokens, usd):
        s["sent"] += sent
        s["by_role"][r["role"]] = s["by_role"].get(r["role"], 0) + sent
        s["local_tokens"] += tokens
        s["spend_usd"] += usd
        g = self.group(r["role"])
        if g in self.role_limits:          # a group's dollars are kept under the group
            s.setdefault("role_usd", {})[g] = s.get("role_usd", {}).get(g, 0.0) + usd


def open_plan(folder):
    """An existing plan ledger, never a new one: a stage cannot open the plan's budget by itself."""
    if not (Path(folder) / "state.json").exists():
        raise ValueError(f"no plan ledger in {folder}: open it first (python -m spreading.ledger open)")
    return Ledger(folder)


class Both:
    """The run's ledger, the stage's (if any) and the plan's, with the Ledger interface: every reservation, attempt,
    commit and release goes to each, in order (the run's first, the plan's last). A reservation either ledger refuses is released in the other, so
    nothing is held for an attempt that will not go out. An attempt is checked and counted with every ledger's lock
    held (always taken in the same order), every ledger checked before any counts: an attempt one refuses is counted
    in none, and one counted in one is counted in all.
    A limit reached on commit is raised after every ledger has committed."""

    def __init__(self, run, plan, stage=None):
        self.ledgers = [run, *([stage] if stage else []), plan]
        self.stage = stage
        self.dir = run.dir
        self.ids = {}            # our ID -> the IDs in each ledger, in order
        self.done = {}           # our ID -> indexes of ledgers already settled

    def reserve(self, company, role, tokens, usd=0.0, requests=1, local=True):
        got = []
        try:
            for led in self.ledgers:
                got.append(led.reserve(company, role, tokens, usd=usd, requests=requests, local=local))
        except LimitReached:
            for led, r in zip(self.ledgers, got):
                led.release(r, "the other ledger refused this reservation")
            raise
        rid = got[0]
        self.ids[rid], self.done[rid] = got, set()
        return rid

    def before_send(self, rid):
        with contextlib.ExitStack() as held:
            for led in self.ledgers:
                held.enter_context(led.lock())
            states = [led._load() for led in self.ledgers]
            checked = [led._send_check(s, r) for led, s, r in zip(self.ledgers, states, self.ids[rid])]
            return min(led._sent(s, r, *c) for led, s, r, c in zip(self.ledgers, states, self.ids[rid], checked))

    def _each(self, rid, f):
        first = None
        for i, (led, r) in enumerate(zip(self.ledgers, self.ids[rid])):
            if i in self.done[rid]:
                continue
            self.done[rid].add(i)
            try:
                f(led, r)
            except LimitReached as e:
                first = first or e
        if first:
            raise first

    def commit(self, rid, tokens=0, usd=0.0):
        self._each(rid, lambda led, r: led.commit(r, tokens, usd))

    def commit_unknown(self, rid, why):
        self._each(rid, lambda led, r: led.commit_unknown(r, why))

    def release(self, rid, why):
        self._each(rid, lambda led, r: led.release(r, why))

    def claim_429(self, company, role, wait_s):
        """Ledger.claim_429 in every ledger, with every lock held and every ledger checked first: a claim one refuses
        is counted in none (the refusal recorded in each, naming the ledger), one granted is counted in all. Returns
        the refusing ledger's (reason, used), else (None, the plan ledger's used: the job's, across its children)."""
        amount(wait_s, "wait_s")
        company = str(company)          # the key as JSON keeps it
        names = ["run ledger", *(["stage ledger"] if self.stage else []), "plan ledger"]
        with contextlib.ExitStack() as held:
            for led in self.ledgers:
                held.enter_context(led.lock())
            states = [led._load() for led in self.ledgers]
            checks = [led._claim_check(s, company, wait_s) for led, s in zip(self.ledgers, states)]
            i = next((i for i, c in enumerate(checks) if c), None)
            refused = None if i is None else (checks[i][0], f"{names[i]}: {checks[i][1]}")
            got = [led._claimed(s, company, role, wait_s, refused) for led, s in zip(self.ledgers, states)]
            return got[-1 if i is None else i]

    def role_refusal(self, role, usd=0.0):
        """The first ledger's refusal of one more attempt of the role (Ledger.role_refusal), naming that ledger."""
        names = ["", *(["stage ledger: "] if self.stage else []), "plan ledger: "]
        for name, led in zip(names, self.ledgers):
            why = led.role_refusal(role, usd)
            if why:
                return name + why
        return None

    def state(self):
        """The run's state, with the plan's under "plan" and the stage's under "stage"; stopped if any has."""
        s, p = self.ledgers[0].state(), self.ledgers[-1].state()
        out = {**s, "plan": {"folder": str(self.ledgers[-1].dir), **p}}
        stops = [s["stopped"]]
        if self.stage:
            g = self.stage.state()
            out["stage"] = {"folder": str(self.stage.dir), **g}
            stops.append(g["stopped"] and f"stage ledger: {g['stopped']}")
        stops.append(p["stopped"] and f"plan ledger: {p['stopped']}")
        out["stopped"] = next((x for x in stops if x), None)
        return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m spreading.ledger",
                                 description="open a plan ledger, or show a ledger's state")
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("open", help="open a new plan ledger with the whole plan's limits")
    o.add_argument("folder")
    o.add_argument("--dispatches", type=int, required=True, help="most requests over the whole plan, every role")
    o.add_argument("--local-tokens", type=int, required=True, help="most local tokens (embeddings included)")
    o.add_argument("--spend-usd", type=float, required=True, help="most OpenRouter spend over the whole plan")
    o.add_argument("--batch-wall-s", type=float, required=True, help="most wall time from opening, in seconds")
    o.add_argument("--company-wall-s", type=float, required=True,
                   help="most wall time per company (or stage key), in seconds")
    o.add_argument("--role-limit", nargs=3, action="append", default=[], metavar=("ROLE", "CALLS", "USD"),
                   help="a role's own limit over the whole ledger (e.g. haiku 80 0.08); repeat for more roles")
    o.add_argument("--role-group", nargs=2, action="append", default=[], metavar=("ROLE", "GROUP"),
                   help="count ROLE under GROUP's role limit (e.g. generator analyst); repeat for more roles")
    st = sub.add_parser("state", help="print a ledger's state")
    st.add_argument("folder")
    a = ap.parse_args(argv)
    if a.cmd == "open":
        if (Path(a.folder) / "state.json").exists():
            ap.error(f"{a.folder} already holds a ledger; a plan's budget is opened once")
        try:
            roles = {r: {"calls": int(n), "spend_usd": float(usd)} for r, n, usd in a.role_limit}
            check_role_limits(roles)
        except ValueError as e:
            ap.error(f"--role-limit ROLE CALLS USD: {e}")
        groups = dict(a.role_group)
        try:
            check_role_groups(groups, roles)
        except ValueError as e:
            ap.error(f"--role-group ROLE GROUP: {e}")
        Ledger(a.folder, {"dispatches": a.dispatches, "local_tokens": a.local_tokens, "spend_usd": a.spend_usd,
                          "batch_wall_s": a.batch_wall_s, "company_wall_s": a.company_wall_s},
               role_limits=roles or None, role_groups=groups or None)
    print(json.dumps(open_plan(a.folder).state(), indent=1))


def allowance_refusal(used, wait_s, lim=RATE_LIMIT_ALLOWANCE):
    """(reason, why) when one more 429 recovery waiting `wait_s` does not fit the allowance `lim` after `used`
    ({recoveries, wait_s}), else None: the one rule the ledger, a Caller without one and a role call's Meter apply. The
    wait is the planned one, never shortened to fit: a wait that does not fit is refused. Five default 60 s waits use
    the whole 300 s."""
    if used["recoveries"] >= lim["recoveries"]:
        return "job_allowance_recoveries", f"{used['recoveries']} of {lim['recoveries']} 429 recoveries used"
    if used["wait_s"] + wait_s > lim["wait_s"]:
        return "job_allowance_wait", (f"{used['wait_s']:.3f} s of {lim['wait_s']} s of 429 waiting used, "
                                      f"{wait_s:.3f} s asked")
    return None


def count(x, name, least=0):
    if isinstance(x, bool) or not isinstance(x, int) or x < least:
        raise ValueError(f"{name} must be an integer of at least {least}, not {x!r}")


def amount(x, name):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0:
        raise ValueError(f"{name} must be a finite nonnegative number, not {x!r}")


def check_limits(limits):
    if not isinstance(limits, dict) or set(limits) != set(LIMITS):
        raise ValueError(f"a new ledger needs every limit: {LIMITS}")
    for k in ("dispatches", "local_tokens"):
        count(limits[k], k)
    for k in ("spend_usd", "batch_wall_s", "company_wall_s"):
        amount(limits[k], k)


def check_role_limits(role_limits):
    if not isinstance(role_limits, dict) or not all(
            isinstance(r, str) and r and isinstance(v, dict) and set(v) == {"calls", "spend_usd"}
            for r, v in role_limits.items()):
        raise ValueError(f"role limits are role -> {{'calls', 'spend_usd'}}, not {role_limits!r}")
    for r, v in role_limits.items():
        count(v["calls"], f"{r} calls")
        amount(v["spend_usd"], f"{r} spend_usd")


def check_role_groups(role_groups, role_limits):
    """Each grouped role names a group that has a role limit, has no limit of its own, and is no group itself."""
    if not isinstance(role_groups, dict) or not all(
            isinstance(r, str) and r and isinstance(g, str) and g for r, g in role_groups.items()):
        raise ValueError(f"role groups are role -> group, not {role_groups!r}")
    for r, g in role_groups.items():
        if g not in role_limits or g in role_groups:
            raise ValueError(f"{r}'s group {g!r} needs its own role limit and cannot be grouped itself")
        if r in role_limits:
            raise ValueError(f"{r} has its own role limit; it cannot also count under {g!r}")


def sent_by_role(state):
    """Requests the ledger counts as sent, by role: committed ones and the sends of reservations still open."""
    by = dict(state.get("by_role") or {})
    for r in (state.get("reserved") or {}).values():
        by[r["role"]] = by.get(r["role"], 0) + r["sent"]
    return by


def company_delta(now, base, hosted, unknown=0):
    """What a company job sums: the ledger's change from state `base` to state `now` (an empty base gives the run's
    own use): requests sent (by role), local tokens, dollars, Haiku's requests and dollars; and the run's hosted
    tokens as OpenRouter reported them (no ledger counts them: it bounds hosted use by dollars). `tokens` is local
    plus the reported hosted tokens only, a lower bound when `hosted_unknown` > 0: the hosted attempts with no
    reported usage, never counted as zero (their whole reservation is charged). Every attempt of a run counts in
    both its own and the company ledger, so the change equals the run's own count unless another process used the
    company ledger meanwhile."""
    b = sent_by_role(base)
    by = {r: n - b.get(r, 0) for r, n in sent_by_role(now).items()}
    usd = lambda s: (s.get("role_usd") or {}).get("haiku", 0.0)     # noqa: E731
    local = now["local_tokens"] - base.get("local_tokens", 0)
    return {"requests": sum(by.values()), "by_role": {r: n for r, n in by.items() if n}, "local_tokens": local,
            "dollars": now["spend_usd"] - base.get("spend_usd", 0.0), "hosted_tokens": hosted,
            "hosted_unknown": unknown, "tokens": local + hosted,
            "haiku_requests": by.get("haiku", 0), "haiku_dollars": usd(now) - usd(base)}


if __name__ == "__main__":
    sys.exit(main())
