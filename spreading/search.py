"""Hybrid search over one filing's note text, for the analyst's `search_notes` and `read_passage`
tools. The model decides whether to search, writes the query, judges the passages and may search again; code builds the
passages, ranks them, logs every search and checks quotes.

Corpus: the filing's note, policy and table text blocks (XBRL TextBlock facts of the extracted instance, so every
passage comes from a fact of this filing, with its fact ID, concept and context). Each block's HTML is cut into
paragraphs (div, p, li) and tables; prose is grouped into passages of at most PASSAGE_WORDS words, tables are kept
as rows of cells ("a | b | c") and cut by rows with the header row repeated. A block in a dimensional context is
kept, with its dimensions in its heading and its entity, period and dimensions on each passage. A policy or table
block whose text is already inside a note block is left out, and a passage whose text repeats an earlier one is kept
once with every source listed (`also_in`); both are recorded. Repeats are found by `dedup_key`, the text with only
whitespace removed, so "$100" and "$(100)" stay apart. Blocks outside the notes (dei, cybersecurity, executive pay) are
not searched.

Ranking: BM25 (k1, b the usual defaults; `tokens`: lowercase words, a short stopword list and a light stemmer for
plurals and possessives, so the keyword baseline is fair) and dense cosine over local embeddings
(`qwen3-embedding:0.6b-fp16`, one model for passages and queries, the query with its instruction), merged by reciprocal
rank fusion, score(d) = sum 1 / (RRF_C + rank(d)) (Cormack et al. 2009; c untuned). Ties go to the earlier passage. The
settings are fixed and recorded with every search. A top rank is not evidence: whether a passage helps is the model's
judgement, and a search that finds nothing does not show that the filing omits it. Each returned passage records its
provenance: in the keyword ranking only, the dense one only, or both.

Pins (embedding model and digest, dimensions, query instruction, chunker and preprocessing versions) form the key
of the embedding cache; vectors read from the cache or the engine are checked (count, dimensions, finite values)
before any ranking.

    s = Search.of(f)                       # builds the passages; embeds them through Ollama, or reads the cache
    o = s.search("revolving credit facility commercial paper")
    # {"items": [{"text", "facts": [], "passage": "P12"}...], "passages": ["P12", ...], "retrieval": {...}}
"""
import hashlib
import html
import json
import math
import re
import time
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

from . import tagless
from .llm import OllamaEmbedder

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data/derived/search"
QUERY_INSTRUCTION = ("Instruct: Given a question about a line of a company's balance sheet, retrieve passages of "
                     "the company's 10-K notes that answer it\nQuery: ")
# notes_v3: a lead-in sentence ending ":" goes with its table, whitespace-only dedup with every source kept, and
# dimensional blocks kept with their dimensions
CHUNKER_VERSION = "notes_v3"
PREPROCESS_VERSION = "html_text_v1"
TOKENIZER_VERSION = "light_stem_v1"     # BM25, snippet and mismatch tokens
PASSAGE_WORDS = 150         # most words in one passage (prose or table rows)
SNIPPET_WORDS = 45          # words of a passage shown in a search result
DEPTH = 50                  # candidates per ranker before fusion
RRF_C = 60                  # reciprocal rank fusion constant (Cormack et al. 2009; an untuned default)
SHOWN = 5                   # passages a search returns
K1, B = 1.2, 0.75           # BM25
STOPWORDS = frozenset("a an and are as at be by for from has have in is it its of on or s that the this these those "
                      "to was were which will with".split())
NOT_NOTES = {"dei", "cyd", "ecd"}           # cover page, cybersecurity (Item 1C), executive pay: not the notes
BLOCK_TAGS = {"div", "p", "li", "h1", "h2", "h3", "h4", "h5", "h6"}
SCOPE = ("end: these are the best-ranked passages of this filing's notes for your query, by keyword and by meaning; "
         "a passage not listed may still matter, and finding nothing does not show the filing omits it. "
         "read_passage shows a passage in full.")


def words(text):
    return text.split()


def plain_tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def stem(t):
    """Plurals and possessives only ("payables" -> "payable", "liabilities" -> "liability", "taxes" -> "tax");
    numbers and short words are left alone."""
    if t.isdigit() or len(t) <= 3:
        return t
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith(("sses", "xes", "ches", "shes")):
        return t[:-2]
    if t.endswith("s") and not t.endswith(("ss", "us", "is")):
        return t[:-1]
    return t


def tokens(text):
    """The ranking tokens: lowercase words, stopwords out, plurals and possessives stemmed (TOKENIZER_VERSION)."""
    return [stem(t) for t in plain_tokens(text) if t not in STOPWORDS]


def norm_text(text):
    return " ".join(plain_tokens(text))


def dedup_key(text):
    """What two passages must share to be one passage: the text with whitespace removed, case and punctuation kept."""
    return re.sub(r"\s+", "", text)


def block_text(value):
    return html.unescape(re.sub(r"<[^>]+>", " ", value))


def dims_label(dims):
    """Axis=member by local name; with tags hidden, by the company's labels."""
    return "; ".join(f"{tagless.member(d, d.split(':')[-1])}={tagless.member(m, m.split(':')[-1])}"
                     for d, m in sorted(dims.items()))


def sha(text):
    return hashlib.sha256(text.encode("utf8")).hexdigest()


class Blocks(HTMLParser):
    """A text block's HTML as a list of ("para", text) and ("table", rows of cells), in document order. Text before
    the first block element (a note's title) is a paragraph of its own."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.buf, self.rows, self.row, self.cell, self.depth = [], [], None, None, None, 0

    def flush(self):
        t = " ".join("".join(self.buf).replace("\xa0", " ").split())
        if t:
            self.out.append(("para", t))
        self.buf = []

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            if self.depth == 0:
                self.flush()
                self.rows = []
            self.depth += 1
        elif self.depth:
            if tag == "tr":
                self.row = []
            elif tag in ("td", "th"):
                self.cell = []
        elif tag in BLOCK_TAGS or tag == "br":
            self.flush()

    def handle_endtag(self, tag):
        if tag == "table" and self.depth:
            self.depth -= 1
            if self.depth == 0:
                rows = [r for r in self.rows if r]
                if rows:
                    self.out.append(("table", rows))
                self.rows = None
        elif self.depth:
            if tag in ("td", "th") and self.cell is not None and self.row is not None:
                self.row.append(" ".join("".join(self.cell).replace("\xa0", " ").split()))
                self.cell = None
            elif tag == "tr" and self.row is not None:
                self.rows.append(merge_cells(self.row))
                self.row = None
        elif tag in BLOCK_TAGS:
            self.flush()

    def handle_data(self, data):
        if self.depth:
            if self.cell is not None:
                self.cell.append(data)
        else:
            self.buf.append(data)

    def close(self):
        super().close()
        self.flush()
        return self.out


def merge_cells(cells):
    """Joins the cells filings split for layout: a lone "$" goes onto the next cell, a lone ")" or "%" onto the one
    before. Empty cells go."""
    out = []
    for c in cells:
        if not c:
            continue
        if out and c in (")", "%", ")%"):
            out[-1] += c
        elif out and out[-1] in ("$", "("):
            out[-1] += c
        else:
            out.append(c)
    return out


def blocks_of(html):
    p = Blocks()
    p.feed(html)
    return p.close()


def is_heading(text):
    return len(words(text)) <= 12 and not text.rstrip().endswith((".", ":", ";"))


def kind_of(concept):
    name = concept.split(":", 1)[1]
    return "policy" if "Policy" in name else "table" if name.endswith("TableTextBlock") else "note"


def chunk_block(blocks, title, tag=""):
    """Passages of one text block: prose grouped up to PASSAGE_WORDS (a longer paragraph cut at sentences), tables
    cut by rows with their lead-in sentence and first row repeated. Each passage keeps its heading path; `tag` (a
    dimensional block's members) follows the title."""
    out, sub, prose = [], None, []

    def head():
        return " > ".join(x for x in (title + tag if title else tag, sub) if x)

    def flush_prose():
        if prose:
            out.append({"kind": "prose", "heading": head(), "text": " ".join(prose)})
            prose.clear()

    for i, (kind, body) in enumerate(blocks):
        if kind == "para":
            if i == 0 and title and norm_text(body) == norm_text(title):
                continue
            if is_heading(body):
                flush_prose()
                sub = body
                continue
            for piece in sentences_within(body, PASSAGE_WORDS):
                if prose and len(words(" ".join(prose))) + len(words(piece)) > PASSAGE_WORDS:
                    flush_prose()
                prose.append(piece)
        else:
            caption = take_caption(prose)
            flush_prose()
            lines = [" | ".join(r) for r in body]
            top = [caption, lines[0]] if caption else [lines[0]]
            part = list(top)
            for ln in lines[1:]:
                if len(part) > len(top) and len(words(" ".join(part))) + len(words(ln)) > PASSAGE_WORDS:
                    out.append({"kind": "table", "heading": head(), "text": "\n".join(part)})
                    part = list(top)
                part.append(ln)
            out.append({"kind": "table", "heading": head(), "text": "\n".join(part)})
    flush_prose()
    return out


def take_caption(prose):
    """The sentence that introduces a table ("... consisted of the following:"), taken off the end of the prose so
    it stays with its table: alone, a lead-in matches the query and shows nothing."""
    if not prose or not prose[-1].rstrip().endswith(":"):
        return None
    *before, last = re.split(r"(?<=[.;])\s+", prose.pop().strip())
    if before:
        prose.append(" ".join(before))
    return last


def sentences_within(text, limit):
    """A paragraph as pieces of at most `limit` words, cut at sentence ends (a sentence longer than the limit is
    cut at the limit)."""
    if len(words(text)) <= limit:
        return [text]
    out, cur = [], []
    for s in re.split(r"(?<=[.;])\s+", text):
        w = words(s)
        while len(w) > limit:
            if cur:
                out.append(" ".join(cur))
                cur = []
            out.append(" ".join(w[:limit]))
            w = w[limit:]
        if cur and len(cur) + len(w) > limit:
            out.append(" ".join(cur))
            cur = []
        cur += w
    if cur:
        out.append(" ".join(cur))
    return out


def corpus(f):
    """The passages of one filing, in document order, with IDs P1, P2, ... (stable for one filing and one chunker
    version), and the record of what was left out."""
    entity = getattr(f, "entity", None)
    blocks, skipped = [], []
    for x in f.facts:
        concept = x["concept"]
        if not concept.endswith("TextBlock"):
            continue
        prefix = concept.split(":", 1)[0]
        ctx = f.contexts.get(x["context"], {})
        if prefix in NOT_NOTES:
            skipped.append({"fact": x["id"], "concept": concept, "why": "not the notes"})
        elif entity is not None and ctx.get("entity") != entity:
            skipped.append({"fact": x["id"], "concept": concept, "why": "another entity"})
        else:
            blocks.append(x)
    # a dimensional block may repeat a note too, so only undimensioned note blocks hold others
    notes = [(x["id"], dedup_key(block_text(x["value"]))) for x in blocks
             if kind_of(x["concept"]) == "note" and not f.contexts.get(x["context"], {}).get("dims")]
    passages, seen = [], {}
    for x in blocks:
        kind = kind_of(x["concept"])
        ctx = f.contexts.get(x["context"], {})
        dims = dict(ctx.get("dims") or {})
        if kind != "note" or dims:
            t = dedup_key(block_text(x["value"]))
            within = next((nid for nid, n in notes if t and nid != x["id"] and t in n), None)
            if within:
                skipped.append({"fact": x["id"], "concept": x["concept"], "context": x["context"],
                                "why": f"inside note block {within}"})
                continue
        parts = blocks_of(x["value"])
        title = parts[0][1] if parts and parts[0][0] == "para" and is_heading(parts[0][1]) else f.label(x["concept"])
        tag = f" [{dims_label(dims)}]" if dims else ""
        source = {"fact": x["id"], "concept": x["concept"], "context": x["context"]}
        for n, p in enumerate(chunk_block(parts, title, tag)):
            key = dedup_key(p["text"])
            if not key:
                continue
            if key in seen:
                seen[key]["also_in"].append({**source, "part": n})
                skipped.append({**source, "why": f"repeats {seen[key]['id']}"})
                continue
            pid = f"P{len(passages) + 1}"
            passages.append({"id": pid, **source, "block_kind": kind, "part": n, "entity": ctx.get("entity"),
                             "period": ctx.get("instant") or f"{ctx.get('start')} to {ctx.get('end')}",
                             "dims": dims, "also_in": [], **p})
            seen[key] = passages[-1]
    for i, p in enumerate(passages):
        same = [q["id"] for q in passages if q["fact"] == p["fact"]]
        k = same.index(p["id"])
        p["prev"] = same[k - 1] if k else None
        p["next"] = same[k + 1] if k + 1 < len(same) else None
    return passages, skipped


def search_text(p):
    """What the rankers see: the heading path and the passage."""
    return (p["heading"] + "\n" + p["text"]).strip()


class BM25:
    def __init__(self, docs):
        self.docs = [Counter(tokens(d)) for d in docs]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lens) / len(self.lens) if self.lens else 0
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query):
        q = set(tokens(query))
        out = []
        for d, ln in zip(self.docs, self.lens):
            s = 0.0
            for t in q:
                if t in d:
                    tf = d[t]
                    s += self.idf[t] * tf * (K1 + 1) / (tf + K1 * (1 - B + B * ln / self.avg))
            out.append(s)
        return out


def unit(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def vector_errors(vectors, n, dimensions):
    """What is wrong with a set of vectors for n texts: the count, a dimension or a value that is not finite. A
    mismatch must never reach a ranking, where zip() would cut it silently."""
    if not isinstance(vectors, list) or len(vectors) != n:
        return [f"{len(vectors) if isinstance(vectors, list) else 'no'} vectors for {n} texts"]
    errs = []
    for i, v in enumerate(vectors):
        if not isinstance(v, list) or len(v) != dimensions:
            errs.append(f"vector {i} has {len(v) if isinstance(v, list) else 'no'} dimensions, not {dimensions}")
        elif not all(isinstance(x, (int, float)) and math.isfinite(x) for x in v):
            errs.append(f"vector {i} has a value that is not a finite number")
    return errs


def checked(vectors, n, dimensions, where):
    errs = vector_errors(vectors, n, dimensions)
    if errs:
        raise ValueError(f"{where}: {'; '.join(errs[:3])}")
    return vectors


def settings():
    """The ranking and display settings, fixed and recorded with every search, with the hash of this file."""
    return {"tokenizer": TOKENIZER_VERSION, "stopwords": sorted(STOPWORDS), "bm25": {"k1": K1, "b": B},
            "depth": DEPTH, "rrf_c": RRF_C, "shown": SHOWN, "snippet_words": SNIPPET_WORDS,
            "ties": "earlier passage first", "snippet": "from a little before the first query token, else the start",
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def instruction(embedder):
    """The query's lead-in: the local model's instruction, or the embedder's own (none for the hosted model)."""
    return getattr(embedder, "query_instruction", QUERY_INSTRUCTION)


class Search:
    def __init__(self, passages, embedder, *, pins, vectors=None, skipped=()):
        self.passages = passages
        self.by_id = {p["id"]: p for p in passages}
        self.embedder, self.pins, self.skipped = embedder, dict(pins), list(skipped)
        self.dimensions = embedder.dimensions
        self.bm25 = BM25([search_text(p) for p in passages])
        if vectors is None:
            vectors = embedder([search_text(p) for p in passages])
        self.vectors = [unit(v) for v in checked(vectors, len(passages), self.dimensions, "passage vectors")]
        self.log, self.seconds = [], {}

    @classmethod
    def of(cls, f, embedder=None, cache=CACHE):
        """The filing's search, embedding its passages once and reading them from the cache after that. The expected
        dimensions are in the key; a cached file that fails the vector checks is recorded and embedded again."""
        embedder = embedder or OllamaEmbedder()
        passages, skipped = corpus(f)
        pins = {"embedding_model": embedder.model, "embedding_digest": embedder.digest(),
                "dimensions": embedder.dimensions, "query_instruction": instruction(embedder),
                "chunker": CHUNKER_VERSION, "preprocessing": PREPROCESS_VERSION, "passage_words": PASSAGE_WORDS}
        text_sha = [sha(search_text(p)) for p in passages]
        key = sha(json.dumps(pins, sort_keys=True) + "".join(text_sha))[:16]
        path = Path(cache) / Path(f.dir).parent.name / Path(f.dir).name / f"embeddings_{key}.json"
        vectors, cache_use = None, "miss"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf8"))
            errs = vector_errors(data.get("vectors"), len(passages), embedder.dimensions)
            if data.get("pins") == pins and data.get("text_sha") == text_sha and not errs:
                vectors, cache_use = data["vectors"], "hit"
            else:
                cache_use = "refused: " + ("; ".join(errs[:3]) if errs else "pins or texts differ")
        if vectors is None:
            vectors = checked(embedder([search_text(p) for p in passages]), len(passages), embedder.dimensions,
                              "passage vectors from the engine")
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"pins": pins, "text_sha": text_sha, "ids": [p["id"] for p in passages],
                                       "vectors": vectors}), encoding="utf8")
            tmp.replace(path)
        pins.update(cache=str(path), cache_use=cache_use, passages=len(passages), settings=settings())
        return cls(passages, embedder, pins=pins, vectors=vectors, skipped=skipped)

    def rank(self, query):
        """Both rankings to DEPTH and their fusion: lists of (passage ID, score), ties to the earlier passage."""
        t0 = time.perf_counter()
        kw = self.bm25.scores(query)
        kw_rank = sorted((i for i, s in enumerate(kw) if s > 0), key=lambda i: (-kw[i], i))[:DEPTH]
        t1 = time.perf_counter()
        q = unit(checked(self.embedder([instruction(self.embedder) + query]), 1, self.dimensions, "query vector")[0])
        dense = [sum(a * b for a, b in zip(q, v)) for v in self.vectors]
        dn_rank = sorted(range(len(dense)), key=lambda i: (-dense[i], i))[:DEPTH]
        t2 = time.perf_counter()
        fused = Counter()
        for r in (kw_rank, dn_rank):
            for k, i in enumerate(r, 1):
                fused[i] += 1 / (RRF_C + k)
        fu_rank = sorted(fused, key=lambda i: (-fused[i], i))
        ids = lambda r, s: [(self.passages[i]["id"], round(s[i], 6)) for i in r]
        self.seconds = {"keyword": round(t1 - t0, 4), "dense": round(t2 - t1, 4)}
        return ids(kw_rank, kw), ids(dn_rank, dense), ids(fu_rank, fused)

    def search(self, query):
        query = " ".join(query.split())
        if not plain_tokens(query):
            raise ValueError("the query is empty: write the words to look for")
        kw, dn, fu = self.rank(query)
        top = [pid for pid, _ in fu[:SHOWN]]
        rank_of = lambda r: {pid: k for k, (pid, _) in enumerate(r, 1)}
        kr, dr = rank_of(kw), rank_of(dn)
        items = []
        for pid in top:
            p = self.by_id[pid]
            items.append({"text": f"{pid} [{p['heading']}] {snippet(p['text'], query)}", "facts": [],
                          "passage": pid, "snippet": True})
        items.append({"text": SCOPE, "facts": []})
        ranks = {pid: {"keyword": kr.get(pid), "dense": dr.get(pid)} for pid in top}
        rec = {"query": query, "keyword": kw, "dense": dn, "fused": fu[:DEPTH], "returned": top, "ranks": ranks,
               "provenance": {pid: provenance(r) for pid, r in ranks.items()}, "seconds": self.seconds,
               "pins": self.pins}
        self.log.append(rec)
        return {"items": items, "passages": top, "retrieval": rec}

    def read(self, pid):
        """One passage in full. Only `pid` is read: the neighbours it names are pointers, opened by reading them."""
        p = self.by_id.get(pid)
        if p is None:
            raise KeyError(f"{pid} is not a passage of this filing")
        where = f"{tagless.member(p['concept'], p['concept'])} (fact {p['fact']}), part {p['part'] + 1}"
        if p["dims"]:
            where += f", {dims_label(p['dims'])}"
        items = [{"text": f"{pid} [{p['heading']}] {p['text']}", "facts": [], "passage": pid},
                 {"text": "note: source " + where + "; before it " + (p["prev"] or "nothing") + ", after it "
                          + (p["next"] or "nothing"), "facts": []}]
        near = [x for x in (p["prev"], p["next"]) if x]
        return {"items": items, "passages": [pid] + near, "read": pid}


def provenance(ranks):
    """Which ranking found a returned passage within DEPTH: keyword only, dense only, or both."""
    k, d = ranks["keyword"] is not None, ranks["dense"] is not None
    return "both" if k and d else "keyword" if k else "dense" if d else "neither"


def snippet(text, query):
    """SNIPPET_WORDS words of the passage starting a little before its first word that the query also has, else
    at its start; '...' marks a cut."""
    w = words(text)
    if len(w) <= SNIPPET_WORDS:
        return text
    q = set(tokens(query))
    hits = [i for i, x in enumerate(w) if set(tokens(x)) & q]
    start = max(0, min(hits[0] - SNIPPET_WORDS // 3, len(w) - SNIPPET_WORDS)) if hits else 0
    body = " ".join(w[start:start + SNIPPET_WORDS])
    return ("... " if start else "") + body + (" ..." if start + SNIPPET_WORDS < len(w) else "")
