"""C1 fetch: one filing's XBRL files through the pinned edgar-search-mcp server (spreading.mcp_client), each checked
by its SHA-256. Downloads are untrusted data: each filing gets its own folder and is only parsed.

The set is what filing.py opens: FilingSummary.xml, the one company schema <stem>.xsd, the one extracted instance
(*_htm.xml, named after the main document, which need not be the schema's stem) and each of <stem>_lab.xml, _cal.xml,
_def.xml and _pre.xml that is filed as its own file, found in the filing's index.json by exact name. A linkbase with no
file of its own is read as embedded in the schema (EDGAR's "schema with embedded linkbases"), and the schema must then
hold it. A missing, ambiguous or unheld name ends the company as not run; nothing is guessed. The whole listing is kept in the manifest, and index.json is saved with the
files. The folder is written as <folder>.part and renamed only after every file is verified, manifest last.

The main document (inline XBRL HTML, for the analyst's printed page) is added to a fetched folder by --main-documents,
with a supplement manifest of its own; the filing's manifest and files are not touched (fetch_main_document).

    .venv\\Scripts\\python.exe -s -E -m spreading.edgar 47111 0001628280-26-008586 --snapshot <name> --mode frozen
    .venv\\Scripts\\python.exe -s -E -m spreading.edgar --verify data/filings/47111/0001628280-26-008586
    .venv\\Scripts\\python.exe -s -E -m spreading.edgar --main-documents data/filings/*/* --snapshot <name> --mode record
    .venv\\Scripts\\python.exe -s -E -m spreading.edgar --r-pages <folder> ... --snapshot <name> --mode record

--r-pages adds the statement R pages FilingSummary lists (fetch_r_pages), with a supplement manifest of its own.
"""
import argparse
import hashlib
import json
import re
import shutil
import stat
import xml.etree.ElementTree as ET
from pathlib import Path

from .mcp_client import Fatal, NotRun, now

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data/filings"
LINKBASES = ("_lab.xml", "_cal.xml", "_def.xml", "_pre.xml")
LINK = "{http://www.xbrl.org/2003/linkbase}"
LINK_ELEMENT = {"_lab.xml": "labelLink", "_cal.xml": "calculationLink", "_def.xml": "definitionLink",
                "_pre.xml": "presentationLink"}
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,200}")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def listing(body):
    """The file names of an index.json, or NotRun when it is not the SEC's directory listing."""
    try:
        items = json.loads(body)["directory"]["item"]
        names = [x["name"] for x in items]
    except (ValueError, KeyError, TypeError):
        raise NotRun("INDEX", "index.json is not a directory listing") from None
    if not all(isinstance(n, str) for n in names):
        raise NotRun("INDEX", "index.json lists a name that is not a string")
    return names


def file_set(names, accession):
    """The names the spread opens, from the listing: FilingSummary.xml, the schema, the one extracted instance and
    the linkbases filed as files of their own. Every name must appear exactly once and each linkbase file must share
    the one company schema's stem; anything else is reported, never resolved. Use `embedded` for the linkbases the
    schema must hold."""
    acc = accession.replace("-", "")
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise NotRun("FILE_SET", f"names listed twice: {dup}")
    if "FilingSummary.xml" not in names:
        raise NotRun("FILE_SET", "FilingSummary.xml is not in the filing")
    xsd = [n for n in names if n.lower().endswith(".xsd") and not n.startswith(acc)]
    if len(xsd) != 1:
        raise NotRun("FILE_SET", f"{'no' if not xsd else 'more than one'} company schema: {xsd}")
    stem = xsd[0][:-4]
    inst = [n for n in names if n.lower().endswith("_htm.xml")]
    if len(inst) != 1:
        raise NotRun("FILE_SET", f"{'no' if not inst else 'more than one'} extracted instance: {inst}")
    chosen = ["FilingSummary.xml", xsd[0], inst[0]]
    for suffix in LINKBASES:
        hits = [n for n in names if n.lower().endswith(suffix)]
        if hits and hits != [stem + suffix]:
            raise NotRun("FILE_SET", f"{stem}{suffix} ambiguous or not on the schema's stem: {hits}")
        chosen += hits
    bad = [n for n in chosen if not SAFE_NAME.fullmatch(n)]
    if bad:
        raise NotRun("FILE_SET", f"file names that are not plain names: {bad}")
    return chosen


def embedded(chosen):
    """The linkbase suffixes with no file of their own in `chosen`: the schema must hold them."""
    stem = next(n for n in chosen if n.lower().endswith(".xsd"))[:-4]
    return [x for x in LINKBASES if stem + x not in chosen]


def check_embedded(xsd_path, suffixes):
    """NotRun unless the schema holds a link element of each embedded linkbase (parsed, never executed)."""
    try:
        root = ET.parse(xsd_path).getroot()
    except ET.ParseError as e:
        raise NotRun("FILE_SET", f"the schema does not parse ({e})") from None
    held = {e.tag for e in root.iter() if isinstance(e.tag, str) and e.tag.startswith(LINK)}
    gone = [x for x in suffixes if LINK + LINK_ELEMENT[x] not in held]
    if gone:
        raise NotRun("FILE_SET", f"linkbases neither filed nor embedded in the schema: {gone}")


def provenance(fetcher):
    pin = fetcher.pin
    return {"via": pin["package"], "version": pin["version"], "wheel_sha256": pin["wheel_sha256"],
            "tools_sha256": pin["tools_sha256"], "contract_version": pin["contract_version"],
            "mode": fetcher.mode, "snapshot": fetcher.folder.name}


def verify_folder(d):
    """Cache-only use: the folder's manifest, after every file it lists is re-hashed. NotRun on any difference."""
    d = Path(d)
    try:
        man = json.loads((d / "manifest.json").read_text(encoding="utf8"))
    except (OSError, ValueError) as e:
        raise NotRun("CACHE", f"{d} has no readable manifest ({e})") from None
    bad = []
    for name, f in man["files"].items():
        p = d / name
        if not p.is_file() or sha(p.read_bytes()) != f["sha256"]:
            bad.append(name)
    if bad:
        raise NotRun("CACHE", f"{d}: files missing or not matching the manifest: {bad}")
    return man


def fetch_filing(cik, accession, fetcher, dest=None):
    """The filing's files (file_set) and index.json in `dest` (default data/filings/<cik>/<accession>): (folder,
    manifest). An existing folder is used only when it verifies and was made from the same snapshot."""
    cik = str(int(cik))
    acc = accession.replace("-", "")
    d = Path(dest) if dest else CACHE / cik / accession
    prov = provenance(fetcher)
    if d.exists():
        man = verify_folder(d)
        if (man.get("provenance") or {}).get("snapshot") != prov["snapshot"]:
            raise NotRun("PROVENANCE", f"{d} was not made from snapshot {prov['snapshot']} "
                                       f"({(man.get('provenance') or {}).get('snapshot') or 'legacy fetcher'})")
        return d, man
    base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/"
    index, about = fetcher.document(base + "index.json")
    names = listing(index)
    chosen = file_set(names, accession)
    part = d.with_name(d.name + ".part")
    if part.exists():
        shutil.rmtree(part)        # this command's own unpublished leftover from a stopped run
    part.mkdir(parents=True)
    files = {}
    for name, body, ab in [("index.json", index, about)] + [(n, *fetcher.document(base + n)) for n in chosen]:
        (part / name).write_bytes(body)
        if sha((part / name).read_bytes()) != ab["sha256"]:
            raise Fatal("WRITE", f"{part / name} does not read back with its hash")
        src = next(s for s in ab["sources"] if s["url"] == base + name)
        files[name] = {"url": base + name, "sha256": ab["sha256"], "bytes": len(body),
                       "retrieved_at": src.get("retrieved_at"), "frozen": src.get("frozen"),
                       "cache_hit": src.get("cache_hit")}
    emb = embedded(chosen)
    if emb:
        check_embedded(part / next(n for n in chosen if n.lower().endswith(".xsd")), emb)
    manifest = {"cik": cik, "accession": accession, "written": now(), "files": files, "listed": names,
                "selected": chosen, "embedded": emb, "provenance": prov}
    (part / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf8")
    part.rename(d)
    return d, manifest


SUPPLEMENT = "main_document_manifest_v1.json"


def main_document_name(d):
    """The main document of a fetched filing, chosen by evidence, never guessed: the one InputFiles entry of
    FilingSummary.xml that carries a doctype, which must also be the document the extracted instance is named after
    (<name>_htm.xml) and a name in the filing's listing. (name, evidence); NotRun when they do not all agree."""
    d = Path(d)
    man = verify_folder(d)
    summary = ET.parse(d / "FilingSummary.xml").getroot()
    entries = [(e.get("doctype"), (e.text or "").strip()) for e in summary.iter("File") if e.get("doctype")]
    inst = [p.name[:-len("_htm.xml")] + ".htm" for p in d.glob("*_htm.xml")]
    if len(entries) != 1:
        raise NotRun("MAIN_DOCUMENT", f"{d}: FilingSummary lists {len(entries)} documents with a doctype")
    doctype, name = entries[0]
    if not SAFE_NAME.fullmatch(name) or not name.lower().endswith(".htm"):
        raise NotRun("MAIN_DOCUMENT", f"{d}: unsafe or non-HTML main document name {name!r}")
    if inst != [name]:
        raise NotRun("MAIN_DOCUMENT", f"{d}: FilingSummary names {name}, the extracted instance names {inst}")
    if man.get("listed") and name not in man["listed"]:
        raise NotRun("MAIN_DOCUMENT", f"{d}: {name} is not in the filing's listing")
    return name, {"filing_summary": "InputFiles/File with a doctype", "doctype": doctype,
                  "filing_summary_sha256": man["files"]["FilingSummary.xml"]["sha256"],
                  "instance": next(p.name for p in d.glob("*_htm.xml")), "listed": name in (man.get("listed") or [])}


def check_supplement(d, man, name):
    """The supplement of folder `d`, verified: it names this filing's accession and CIK and the main document chosen
    by evidence, the filing manifest at the hash it recorded, and the document at its hash. NotRun otherwise."""
    try:
        s = json.loads((d / SUPPLEMENT).read_text(encoding="utf8"))
    except (OSError, ValueError) as e:
        raise NotRun("SUPPLEMENT", f"{d}: unreadable supplement ({e})") from None
    bad = [k for k, ok in (("accession", s.get("accession") == man["accession"]),
                           ("cik", str(s.get("cik")) == str(man["cik"])),
                           ("document name", (s.get("file") or {}).get("name") == name),
                           ("filing manifest hash",
                            s.get("filing_manifest_sha256") == sha((d / "manifest.json").read_bytes())),
                           ("document hash", (d / name).is_file()
                            and sha((d / name).read_bytes()) == (s.get("file") or {}).get("sha256")))
           if not ok]
    if bad:
        raise NotRun("SUPPLEMENT", f"{d}: the supplement does not hold: {', '.join(bad)}")
    return s


def clear_parts(d, names):
    """Remove this fetch's own unpublished files from a stopped run: exactly <name>.part for each name, regular
    files only, never a wildcard or a folder's contents. A directory, link or reparse point there is refused, not
    removed."""
    for n in names:
        p = d / (n + ".part")
        try:
            st = p.lstat()             # the entry itself, a dangling link or reparse point included
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(st.st_mode) or getattr(st, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise NotRun("LEFTOVER", f"{p}: not a file this fetch wrote; not removed")
        p.unlink()


def leftovers(d, name):
    """Files of an unfinished supplement: the document's or the supplement's .part, or the document without a
    supplement (published by a run stopped before its supplement)."""
    out = [p.name for p in (d / (name + ".part"), d / (SUPPLEMENT + ".part")) if p.exists()]
    if (d / name).exists() and not (d / SUPPLEMENT).exists():
        out.append(name)
    return out


def fetch_main_document(d, fetcher, stage=None):
    """Add the main document (inline XBRL HTML) to a fetched filing folder: the filing's own manifest and files stay as
    they are; the document and a supplement manifest of its own are written, each first as .part, and published by
    rename once the document reads back with its hash, the document first. A folder whose manifest already holds the
    document is returned as it is; one whose supplement verifies (check_supplement) too. A published document without
    its supplement is refused, never overwritten: a person removes it; then this fetch's own .part files of a stopped
    run are removed (clear_parts) and the fetch runs.
    `stage` (a dict) is set to the step under way, for the outcome record. (name, supplement or None)."""
    d, stage = Path(d), stage if stage is not None else {}
    stage["at"] = "verify filing"
    man = verify_folder(d)
    stage["at"] = "choose document"
    name, evidence = main_document_name(d)
    if name in man["files"]:
        return name, None          # fetched with the filing (legacy folder): its own manifest covers it
    stage["at"] = "check supplement"
    if (d / SUPPLEMENT).exists():
        return name, check_supplement(d, man, name)
    if (d / name).exists():
        raise NotRun("LEFTOVER", f"{d}: {name} is published without its supplement; not overwritten (delete it "
                               "to fetch it again)")
    clear_parts(d, [name, SUPPLEMENT])
    stage["at"] = "fetch"
    base = f"https://www.sec.gov/Archives/edgar/data/{man['cik']}/{man['accession'].replace('-', '')}/"
    body, ab = fetcher.document(base + name)
    stage["at"] = "write"
    part = d / (name + ".part")
    part.write_bytes(body)
    if sha(part.read_bytes()) != ab["sha256"]:
        raise Fatal("WRITE", f"{part} does not read back with its hash")
    src = next(x for x in ab["sources"] if x["url"] == base + name)
    s = {"cik": man["cik"], "accession": man["accession"], "written": now(), "selected_by": evidence,
         "file": {"name": name, "url": base + name, "sha256": ab["sha256"], "bytes": len(body),
                  "retrieved_at": src.get("retrieved_at"), "frozen": src.get("frozen"),
                  "cache_hit": src.get("cache_hit")},
         "filing_manifest_sha256": sha((d / "manifest.json").read_bytes()), "provenance": provenance(fetcher)}
    sup_part = d / (SUPPLEMENT + ".part")
    sup_part.write_text(json.dumps(s, indent=1), encoding="utf8")
    stage["at"] = "publish"
    part.replace(d / name)
    sup_part.replace(d / SUPPLEMENT)
    stage["at"] = "verify published"
    return name, check_supplement(d, man, name)


def verified_main_document(d):
    """(path, None) for the folder's main document when it verifies, by the filing's manifest or its supplement;
    (None, why) otherwise. The page reader reads only a document this returns."""
    d = Path(d)
    try:
        man = verify_folder(d)
        name, _ = main_document_name(d)
        if name not in man["files"]:
            if not (d / SUPPLEMENT).exists():
                return None, f"{name} has not been fetched (no supplement manifest)"
            check_supplement(d, man, name)
        return d / name, None
    except (NotRun, OSError, ValueError, ET.ParseError) as e:
        return None, f"{type(e).__name__}: {e}"


def filing_inputs(d):
    """The hashes of the filing's manifest, the supplement and the main document, as a run records them; a file
    absent is None, and nothing here raises."""
    d = Path(d)
    out = {}
    for k, p in (("manifest", d / "manifest.json"), ("supplement", d / SUPPLEMENT)):
        out[k] = sha(p.read_bytes()) if p.is_file() else None
    doc, why = verified_main_document(d)
    out["main_document"] = {"name": doc.name, "sha256": sha(doc.read_bytes())} if doc else {"unverified": why}
    return out


def main_documents(folders, fetcher, run):
    """The supplement for many folders, each its own outcome; one folder's failure never stops the rest, and only a
    Fatal (the stop switch, a write that does not read back) stops the command. Each folder's outcome is
    appended to run/progress.jsonl as it ends; on any exit run/outcomes.json holds every outcome, with the stage and
    leftover files of a folder not done, and the folders not reached, and run/exceptions.json every folder not done."""
    from .runrec import write
    out, exc, todo = [], [], [Path(x) for x in folders]

    def progress(entry):            # append-only, so a killed run still shows each folder it finished
        with open(Path(run) / "progress.jsonl", "a", encoding="utf8") as fh:
            fh.write(json.dumps(entry) + "\n")

    def record(remaining):
        write(run, "outcomes.json", {"outcomes": out, "not_reached": [str(x) for x in remaining]})
        write(run, "exceptions.json", exc)
    try:
        while todo:
            d, stage = todo[0], {}
            try:
                had = (d / SUPPLEMENT).exists()
                with fetcher.company():
                    name, s = fetch_main_document(d, fetcher, stage)
                out.append({"folder": str(d), "outcome": "fetched" if s and not had else "present", "document": name})
            except Exception as e:      # every failure of one folder is its outcome; the next folder still runs
                try:
                    name, _ = main_document_name(d)
                    left = leftovers(d, name)
                except Exception:
                    left = None
                out.append({"folder": str(d), "outcome": "not run", "stage": stage.get("at"),
                            "why": f"{type(e).__name__}: {e}", "leftover_files": left})
                exc.append(out[-1])
                if isinstance(e, Fatal):    # the stop switch or a failed write stops the command: the rest not reached
                    todo.pop(0)
                    progress(out[-1])
                    raise
            todo.pop(0)
            progress(out[-1])
    finally:
        record(todo)
    return out


R_SUPPLEMENT = "r_pages_manifest_v1.json"
R_PAGE = re.compile(r"R[0-9]{1,4}\.htm")


def statement_pages(d):
    """The R pages FilingSummary.xml files under MenuCategory "Statements" (the face statements and their
    parentheticals), in its order. Which page is which statement is not decided here."""
    names = [r.findtext("HtmlFileName") for r in ET.parse(Path(d) / "FilingSummary.xml").getroot().iter("Report")
             if r.findtext("MenuCategory") == "Statements"]
    bad = [n for n in names if not n or not R_PAGE.fullmatch(n)]
    if bad or not names:
        raise NotRun("R_PAGES", f"{d}: FilingSummary lists no statement pages or names that are not R pages: {bad}")
    return names


def check_r_supplement(d, man, todo):
    """The R-page supplement of folder `d`, verified for the pages `todo` the filing's manifest does not hold."""
    s = json.loads((d / R_SUPPLEMENT).read_text(encoding="utf8"))
    bad = [n for n in todo if n not in s["files"] or not (d / n).is_file()
           or sha((d / n).read_bytes()) != s["files"][n]["sha256"]]
    if bad or s.get("accession") != man["accession"]:
        raise NotRun("R_SUPPLEMENT", f"{d}: the R-page supplement does not hold: {bad}")
    return s


def verified_filing(d):
    """A filing folder that must already be complete, checked without a write or a request: its files against the
    manifest, the main document and the statement R pages by the manifest or their supplements. NotRun otherwise."""
    d = Path(d)
    man = verify_folder(d)
    name, _ = main_document_name(d)
    if name not in man["files"]:
        if not (d / SUPPLEMENT).exists():
            raise NotRun("SUPPLEMENT", f"{d}: {name} has not been fetched")
        check_supplement(d, man, name)
    todo = [n for n in statement_pages(d) if n not in man["files"]]
    if todo:
        if not (d / R_SUPPLEMENT).exists():
            raise NotRun("R_SUPPLEMENT", f"{d}: R pages not fetched: {todo}")
        check_r_supplement(d, man, todo)
    return man


def fetch_r_pages(d, fetcher, stage=None):
    """Add the statement R pages (statement_pages) to a fetched filing folder, as fetch_main_document does for the main
    document: the filing's files stay as they are; each page and a supplement of its own (R_SUPPLEMENT) are written
    first as .part and published once every page reads back with its hash, the supplement last. Pages the filing's
    manifest already holds are not fetched; a folder whose supplement exists is checked, never refetched; published
    pages without the supplement are refused, this fetch's own .part files are removed (clear_parts). (names,
    supplement or None)."""
    d, stage = Path(d), stage if stage is not None else {}
    stage["at"] = "verify filing"
    man = verify_folder(d)
    names = statement_pages(d)
    todo = [n for n in names if n not in man["files"]]
    if not todo:
        return names, None
    stage["at"] = "check supplement"
    if (d / R_SUPPLEMENT).exists():
        return names, check_r_supplement(d, man, todo)
    left = [n for n in todo if (d / n).exists()]
    if left:
        raise NotRun("LEFTOVER", f"{d}: R pages published without their supplement, not overwritten (delete them "
                               f"to fetch them again): {left}")
    clear_parts(d, todo + [R_SUPPLEMENT])
    base = f"https://www.sec.gov/Archives/edgar/data/{man['cik']}/{man['accession'].replace('-', '')}/"
    files = {}
    for n in todo:
        stage["at"] = f"fetch {n}"
        body, ab = fetcher.document(base + n)
        (d / (n + ".part")).write_bytes(body)
        if sha((d / (n + ".part")).read_bytes()) != ab["sha256"]:
            raise Fatal("WRITE", f"{d / n}.part does not read back with its hash")
        src = next(x for x in ab["sources"] if x["url"] == base + n)
        files[n] = {"url": base + n, "sha256": ab["sha256"], "bytes": len(body),
                    "retrieved_at": src.get("retrieved_at"), "frozen": src.get("frozen"),
                    "cache_hit": src.get("cache_hit")}
    s = {"cik": man["cik"], "accession": man["accession"], "written": now(),
         "selected_by": "FilingSummary.xml Report entries with MenuCategory Statements",
         "filing_summary_sha256": man["files"]["FilingSummary.xml"]["sha256"], "files": files,
         "filing_manifest_sha256": sha((d / "manifest.json").read_bytes()), "provenance": provenance(fetcher)}
    stage["at"] = "publish"
    (d / (R_SUPPLEMENT + ".part")).write_text(json.dumps(s, indent=1), encoding="utf8")
    for n in todo:
        (d / (n + ".part")).replace(d / n)
    (d / (R_SUPPLEMENT + ".part")).replace(d / R_SUPPLEMENT)
    return names, s


def r_pages(folders, fetcher, run):
    """fetch_r_pages for many folders, each its own outcome; a Fatal stops the command. run/outcomes.json and
    run/exceptions.json hold every outcome and every folder not done."""
    from .runrec import write
    out = []
    try:
        for d in map(Path, folders):
            stage, had = {}, (d / R_SUPPLEMENT).exists()
            try:
                with fetcher.company():
                    names, s = fetch_r_pages(d, fetcher, stage)
                out.append({"folder": str(d), "outcome": "fetched" if s and not had else "present", "pages": names})
            except Exception as e:
                out.append({"folder": str(d), "outcome": "not run", "stage": stage.get("at"),
                            "why": f"{type(e).__name__}: {e}"})
                if isinstance(e, Fatal):
                    raise
    finally:
        write(run, "outcomes.json", {"outcomes": out, "not_reached": [str(x) for x in folders[len(out):]]})
        write(run, "exceptions.json", [o for o in out if o["outcome"] == "not run"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cik", nargs="?")
    ap.add_argument("accession", nargs="?")
    ap.add_argument("--snapshot")
    ap.add_argument("--mode", choices=("record", "frozen"))
    ap.add_argument("--dest")
    ap.add_argument("--verify")
    ap.add_argument("--main-documents", nargs="+", metavar="FOLDER", help="add each folder's main document")
    ap.add_argument("--r-pages", nargs="+", metavar="FOLDER", help="add each folder's statement R pages")
    a = ap.parse_args()
    if a.verify:
        man = verify_folder(a.verify)
        print(a.verify, len(man["files"]), "files verified")
        return
    from .mcp_client import Fetcher, snapshot_dir
    from .runrec import new_run
    if a.r_pages:
        run = new_run("fetch_r_pages", {"folders": a.r_pages, "snapshot": a.snapshot, "mode": a.mode})
        with Fetcher(a.mode, snapshot_dir(a.snapshot, a.mode), run / "mcp") as f:
            for o in r_pages(a.r_pages, f, run):
                print(o)
        print("record", run)
        return
    if a.main_documents:
        run = new_run("fetch_main_documents", {"folders": a.main_documents, "snapshot": a.snapshot, "mode": a.mode})
        with Fetcher(a.mode, snapshot_dir(a.snapshot, a.mode), run / "mcp") as f:
            for o in main_documents([Path(x) for x in a.main_documents], f, run):
                print(o)
        print("record", run)
        return
    run = new_run("fetch", {"cik": a.cik, "accession": a.accession, "snapshot": a.snapshot, "mode": a.mode})
    try:
        with Fetcher(a.mode, snapshot_dir(a.snapshot, a.mode), run / "mcp") as f, f.company():
            d, m = fetch_filing(a.cik, a.accession, f, a.dest)
    except (Fatal, NotRun) as e:
        raise SystemExit(f"{type(e).__name__}: {e} (record {run})")
    print(d)
    for k, v in m["files"].items():
        print(f"  {k:<40} {v['bytes']:>10}  {v['sha256'][:16]}")


if __name__ == "__main__":
    main()
