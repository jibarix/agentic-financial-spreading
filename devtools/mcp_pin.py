"""The edgar-search-mcp client's maintenance commands (kept apart from spreading/mcp_client.py): make a new
pin from a release (the deliberate upgrade), check the pinned server's start-up, and open or close a recording.

    .venv-agents\\Scripts\\python.exe -s -E -m devtools.mcp_pin pin --release-manifest <manifest.json>
    .venv-agents\\Scripts\\python.exe -s -E -m devtools.mcp_pin check
    .venv-agents\\Scripts\\python.exe -s -E -m devtools.mcp_pin open <snapshot> --purpose "why"
    .venv-agents\\Scripts\\python.exe -s -E -m devtools.mcp_pin close <snapshot>

`pin` and `check` send no SEC request: the server runs frozen on an empty folder. The limiter is never reset here.
"""
import argparse
import json
import tempfile
from pathlib import Path

from spreading.mcp_client import (CONTRACT, PIN, ROOT, Fatal, Fetcher, close_snapshot, load_pin, lock_wheel, now,
                                  open_snapshot, sha_bytes, write_pin)


def probe_server(pin, log_dir, check_venv=True, root=ROOT):
    """Launch the pinned server frozen on an empty folder, with a throwaway limiter file (no SEC request, the shared
    limiter untouched), and return the start-up record and the tools."""
    with tempfile.TemporaryDirectory(prefix="edgar_frozen_empty_") as empty:
        frozen, rate = Path(empty) / "frozen", Path(empty) / "rate" / "state"
        frozen.mkdir()
        with Fetcher("frozen", frozen, log_dir, pin=pin, rate_path=str(rate), ua="", check_venv=check_venv,
                     root=root) as f:
            return f.startup, f.tools_dump


def make_pin(release_manifest, venv=".venv", lock="requirements.lock", root=ROOT):
    """A new pin from the release's manifest asset and the live listing of the installed server."""
    man_bytes = Path(release_manifest).read_bytes()
    man = json.loads(man_bytes)
    lock_bytes = (Path(root) / lock).read_bytes()
    pin = {"package": man["name"], "version": man["version"], "wheel_sha256": man["wheel_sha256"],
           "wheel_url": f"https://github.com/jibarix/edgar-search-mcp/releases/download/v{man['version']}/{man['wheel']}",
           "source_commit": man["source_commit"],
           "release_url": f"https://github.com/jibarix/edgar-search-mcp/releases/tag/v{man['version']}",
           "release_manifest_sha256": sha_bytes(man_bytes), "files": man["files"],
           "venv": venv, "lock": lock, "lock_sha256": sha_bytes(lock_bytes),
           "launch": {"args": ["-I", "-m", "edgar_mcp"], "rate_file": "%LOCALAPPDATA%\\sec-edgar-rate\\state",
                      "rate_limit": 8},
           "contract_version": CONTRACT, "tools": [], "tools_sha256": None, "profile": None,
           "profile_sha256": None, "made": now(), "made_by": "python -m devtools.mcp_pin pin"}
    lock_wheel(pin, lock_bytes.decode("utf8"))
    return pin


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pin")
    p.add_argument("--release-manifest", required=True)
    p.add_argument("--replace", action="store_true")
    sub.add_parser("check")
    p = sub.add_parser("open")
    p.add_argument("snapshot")
    p.add_argument("--purpose", required=True)
    p = sub.add_parser("close")
    p.add_argument("snapshot")
    a = ap.parse_args()
    from spreading.runrec import new_run, write
    try:
        if a.cmd == "pin":
            if PIN.exists() and not a.replace:
                raise Fatal("PIN", f"{PIN} exists; pass --replace (the upgrade's job)")
            pin = make_pin(a.release_manifest)
            d = new_run("mcp_pin", {"release_manifest": a.release_manifest})
            st, tools = probe_server(pin, d / "server")
            pin.update(tools=tools, tools_sha256=st["tools_sha256"], profile=st["profile"],
                       profile_sha256=st["profile_sha256"], server_info=st["server_info"])
            write_pin(pin)
            write(d, "pin.json", pin)
            print(PIN, "tools", pin["tools_sha256"][:16], "profile", pin["profile_sha256"][:16])
        elif a.cmd == "check":
            d = new_run("mcp_check", {})
            pin = load_pin()
            st, _ = probe_server(pin, d / "server")
            print("start-up checks pass:", json.dumps({k: st[k] for k in ("tools_sha256", "profile_sha256",
                                                                         "protocol_version", "server_info")}))
            print(d)
        elif a.cmd == "open":
            print(open_snapshot(a.snapshot, a.purpose))
        elif a.cmd == "close":
            rec = close_snapshot(a.snapshot)
            print(a.snapshot, rec["addresses"], "addresses", rec["capture_window"])
    except Fatal as e:
        raise SystemExit(f"stopped: {e}")


if __name__ == "__main__":
    main()
