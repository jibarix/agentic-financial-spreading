// Pages: plain JavaScript, served by ui/server.py. Every page reads this computer through the server (live.js draws
// them); this file holds the shared helpers, Start here, the template browser and the drawings of a job's tiebreaker
// lines and trees.
// Every text from a filing, a model or a person goes through esc() before it reaches innerHTML, attributes included.
const view = document.getElementById("view");
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const money = a => {
  if (a === null || a === undefined || a === "" || isNaN(Number(a))) return "";
  const n = Number(a), s = Math.abs(n).toLocaleString("en-US");
  return n < 0 ? `(${s})` : s;
};
// The shipped template's sections and rows, read from the server at start: the Template page and every row name.
let TPL = {name: "", version: "", sections: [], rows: []}, rowsById = {};
const SPECIAL = {NOT_TOLD: "the filing does not tell", COMBINED: "the line combines several items", OUTSIDE: "the line falls outside this statement's rows"};
const rowName = id => SPECIAL[id] || (rowsById[id] || {}).name || id || "";
const STMT = {IS: "Income statement", BS: "Balance sheet", CF: "Cash flow statement"};
// Code and model texts name rows by ID and rules by number; on screen they read as row names, without the rule number.
const human = s => String(s ?? "").replace(/\bRule \d+:\s*/g, "").replace(/\b(?:IS|BS|CF)\.[A-Z0-9_.]*[A-Z0-9_]/g, id => rowsById[id] ? `“${rowsById[id].name}”` : id);
const CHECK = {"every line allocated": "some printed lines were not placed", "section total found": "no printed section total could be used as a check",
  "rows add to the printed total": "the rows do not add to the printed total"};
// The status words the pages show, by what they describe: a job's end as its record names
// it, a stored spread's outcome, a row's availability (spreading/store.py) and the comparison with the AI reference.
const GLOSSARY = [
  ["Jobs: how the job ended", [
    ["running", "the job is still working"],
    ["finished", "the job ran to its end; each statement's outcome is on Results"],
    ["capped", "the job reached a recorded limit (requests, tokens, time or dollars) and stopped; every record is kept"],
    ["stopped", "someone pressed Stop: the step under way was stopped and no later step started"],
    ["interrupted", "the program running the job was closed or broke off before the end"],
    ["incomplete", "a model request failed or its response broke a required check; the job stopped"],
    ["harness defect", "the program found a fault in its own records (a step that could not start or left no record, usage that does not add up, or a changed source); the job stopped"],
    ["no final record", "the job has not written its final record, or it was lost"]]],
  ["Results: each stored spread", [
    ["complete", "every code check passed and no exception stands"],
    ["qualified", "every code check passed, but some lines carry an exception (for example a total that could not be compared)"],
    ["blocked", "at least one code check failed: an unresolved line, an unavailable comparison or a numerical mismatch. Open the spread on Results to see its checks. It is a recorded outcome, not a crash"]]],
  ["Results: each row", [
    ["placed amount", "printed lines were put in the row and they add to an amount other than zero"],
    ["zero contribution", "printed lines were put in the row and they add to zero"],
    ["no placement", "no printed line was put in the row"],
    ["incomplete section", "the row's section was not finished, so its amount is not usable"],
    ["no memo amount", "a memo row with no amount; it stays blank, never 0"],
    ["not applicable", "the row's section does not apply to this statement, as the run recorded it"]]],
  ["Results, Tiebreaker and Tree of thought: the answer key, where one is set up on this computer (a copy without answer keys reads “not compared” on every job)", [
    ["matches the answer key", "the line went to the row the answer key chose, or one it also accepts"],
    ["differs from the answer key", "the line went to another row"],
    ["no answer", "the job gave no final answer for the line, so there is nothing to compare"],
    ["unresolved", "the reference could not be used for the line (for example, it does not place the line, or several of its lines share the same reported fact)"],
    ["not compared", "the job was not compared with an answer key: none was set up for that company's 10-K. An answer key is a stronger model's reviewed spread, not proof"]]],
];

// The session token, the running job and the finished example, as the server holds them; the page never decides them.
const SRV = {on: false, token: "", job: null, example: null};
async function serverState() {
  try {
    const [r, t] = await Promise.all([fetch("/api/state", {cache: "no-store"}), fetch("/api/template", {cache: "no-store"})]);
    if (r.ok) Object.assign(SRV, await r.json(), {on: true});
    if (t.ok) { TPL = await t.json(); rowsById = Object.fromEntries(TPL.rows.map(x => [x.id, x])); }
  } catch (e) { /* no server: render says so */ }
}
async function post(path, body) {
  const r = await fetch(path, {method: "POST", headers: {"Content-Type": "application/json", "X-Session-Token": SRV.token},
    body: JSON.stringify(body || {})});
  return {ok: r.ok, status: r.status, body: await r.json().catch(() => ({}))};
}

function badge(t, tone) { return `<span class="badge ${tone || ""}">${esc(t)}</span>`; }
function outcomeBadge(o) { return badge(o, {qualified: "green", complete: "green", blocked: "red", stopped: "amber"}[o] || "grey"); }
function head(title, sub, right = "") {
  return `<div class="page-head"><div><h1>${esc(title)}</h1><p class="sub">${sub}</p></div><div class="row-flex">${right}</div></div>`;
}
function drawer(title, html) {
  document.getElementById("drawer-title").textContent = title;
  document.getElementById("drawer-body").innerHTML = html;
  document.getElementById("drawer").classList.add("open");
  document.getElementById("shade").classList.add("open");
}
function closeDrawer() {
  document.getElementById("drawer").classList.remove("open");
  document.getElementById("shade").classList.remove("open");
}
document.getElementById("drawer-close").onclick = closeDrawer;
document.getElementById("shade").onclick = closeDrawer;

// ---------- Start here ----------
// A step in the Start page's diagram, coloured by who does it.
const fbox = (who, t, d) => `<div class="fbox ${who}"><b>${esc(t)}</b><div>${esc(d)}</div></div>`;
const fdown = () => `<div class="fdown">↓</div>`;
function pageHome() {
  view.innerHTML = `<div class="page">
  ${head("Spreading financial statements", "Agents read a company's 10-K and put the printed lines of its income statement, balance sheet and cash flow statement, other than totals and excluded lines, into the rows of a standard template.")}
  <div class="card"><div class="hd"><h2>How it works</h2><p class="sub">Who does each step: <span class="fkey person">a person</span> <span class="fkey code">code</span> <span class="fkey model">a model</span> <span class="fkey record">a record</span></p></div>
  <div class="bd"><div class="flow">
    <div class="fcol"><h3>What goes in</h3>
      ${fbox("person", "You choose", "a template, the companies and years, a budget; you can stop a job at any time")}
      ${fbox("record", "The 10-K", "each company's filing from the SEC; the models read its tagged financial statements only, not the notes")}</div>
    <div class="farrow">→</div>
    <div class="fcol"><h3>What the agents do</h3>
      ${fbox("code", "Find the lines", "reads each statement and sets aside totals and excluded lines")}${fdown()}
      ${fbox("model", "Organizer", "sorts the lines into the template's sections")}${fdown()}
      ${fbox("model", "Analyst", "puts each line in a row, with a quote from the filing as evidence")}${fdown()}
      ${fbox("model", "Checker", "places the same line again on its own")}${fdown()}
      ${fbox("code", "Compare", "when the two differ, the analyst takes a second look, seeing both answers")}${fdown()}
      ${fbox("model", "Tiebreaker", "can be switched off: a third model answers a line still disputed")}${fdown()}
      ${fbox("code", "Check", "the quotes, the amounts and the totals")}${fdown()}
      ${fbox("model", "Tree of thought", "off by default; when switched on, tries repairs to a section whose checks fail")}</div>
    <div class="farrow">→</div>
    <div class="fcol"><h3>What comes out</h3>
      ${fbox("record", "A spread", "per company, year and statement: each amount with its source, the lines not placed and why, and the checks")}${fdown()}
      ${fbox("person", "You review", "import it, choose which version counts and download the Excel workbook: the workflow's result, unreviewed; the replay shows how each line was classified")}</div>
  </div></div></div>
  <div class="card" style="margin-top:16px"><div class="hd"><h2>Your first job</h2></div><div class="bd"><ol style="margin:0;padding-left:20px">
    <li><a href="#/setup">Setup</a>: every item the models need is ready.</li>
    <li><a href="#/template">Template</a>: press <b>Use default</b> once. It records the standard template in the results database on this computer; the Run page refuses a start until it is.</li>
    <li><a href="#/companies">Companies and years</a>: search, choose one 10-K and press <b>Choose and fetch</b>.</li>
    <li><a href="#/run">Run</a>: check the template and filing, then press <b>Start</b>.</li></ol></div></div>
  <div class="card" style="margin-top:16px"><div class="hd"><h2>A finished example</h2></div><div class="bd row-flex">
    ${SRV.example ? `<span>${esc(SRV.example_said)}. It opens in the normal pages.</span>
    <span class="spacer"></span>${[["jobs", "Jobs"], ["tree", "Tree of thought"], ["tiebreak", "Tiebreaker"], ["evals", "Evals"]].map(([k, t]) => `<button class="btn" data-ex="${k}">${t}</button>`).join("")}`
      : `<span class="note">The example job is not on this computer.</span>`}</div></div>
  <div class="card" style="margin-top:16px"><div class="hd"><h2>What the status words mean</h2></div><div class="bd">
    ${GLOSSARY.map(([g, words]) => `<h3 class="gl">${esc(g)}</h3><div class="kv">${words.map(([k, v]) => `<div class="k">${esc(k)}</div><div>${esc(v)}</div>`).join("")}</div>`).join("")}</div></div></div>`;
  view.querySelectorAll("[data-ex]").forEach(b => b.onclick = () => { chooseJob(SRV.example); location.hash = `#/${b.dataset.ex}`; });
}

// ---------- 1. Setup: the models table (live.js draws the page) ----------
function modelsTable(models) {
  // Each role and the model it runs on, one table per job option offered, from the settings.
  const opts = Object.keys(models.labels);
  return `<div class="card" style="margin-top:18px"><div class="hd"><h2>Which model does what</h2><p class="sub">Each role in a job and the model it runs on. Each job records the models it actually used.</p></div>
    <div class="bd">${opts.map((k, i) => { const rows = models.rows.filter(r => r.lineup === k);
      return `<h3 class="${i ? "sep" : ""}">${esc(models.labels[k])}</h3>
      <table class="wrap"><tr><th>Part of the job</th><th>Role</th><th>What it does</th><th>Model</th></tr>
      ${rows.map(r => `<tr><td class="note">${esc(r.step)}</td><td><b>${esc(r.role)}</b></td><td>${esc(r.does)}</td>
        <td>${r.where === "off" ? `<span class="note">not used</span>` : `${esc(r.name)}<div class="note mono">${esc(r.model)}</div>`}</td></tr>`).join("")}
      </table>`; }).join("")}</div></div>`;
}

// ---------- 2. Template: the shipped template's rows, read only (live.js draws the page) ----------
const sectionName = id => (TPL.sections.find(s => s.id === id) || {}).name || id;
function tplBrowser() {
  return `<div class="grid" style="grid-template-columns:minmax(0,1fr) minmax(0,1.2fr)">
    <div class="card"><div class="bd" style="padding-top:14px"><input type="text" id="rowfilter" placeholder="Filter rows">
    <div class="scroll" style="margin-top:10px">${TPL.sections.map(s => `<div class="nav-label" style="padding-left:10px">${esc(STMT[s.statement])}: ${esc(s.name)}</div>
      ${TPL.rows.filter(r => r.section === s.id).map(r => `<div class="tpl-row" data-row="${esc(r.id)}" data-text="${esc((r.name + " " + r.definition).toLowerCase())}">
        <div><div class="rn">${esc(r.name)}</div><div class="d">${r.definition ? esc(r.definition) : r.formula ? `= ${esc(r.formula)}` : "<i>no definition</i>"}</div></div>
        <div class="num">${badge(r.type, r.type === "item" ? "grey" : "violet")}</div></div>`).join("")}`).join("")}</div></div></div>
    <div class="card"><div class="bd" id="rowpane" style="padding-top:16px"><p class="note">Choose a row on the left.</p></div></div></div>`;
}
function rowPane(r) {
  if (!r) return `<p class="note">Choose a row on the left.</p>`;
  return `<h2>${esc(r.name)}</h2><p class="note">${esc(sectionName(r.section))} · ${esc(r.type)}</p>
    <label class="f">Definition</label><p>${esc(r.definition) || "<i>empty</i>"}</p>
    <label class="f">Belongs here</label><ul>${r.belongs.map(b => `<li>${esc(b)}</li>`).join("") || "<li class='note'>none listed</li>"}</ul>
    <label class="f">Does not belong here, and where it goes</label><ul>${r.not_belongs.map(x => `<li>${esc(x.item)} <span class="note">→ ${esc(rowName(x.goes_to))}</span></li>`).join("") || "<li class='note'>none listed</li>"}</ul>`;
}
function wireBrowser() {
  const pane = document.getElementById("rowpane");
  const open = el => {
    view.querySelectorAll(".tpl-row").forEach(x => x.classList.toggle("sel", x === el));
    pane.innerHTML = rowPane(rowsById[el.dataset.row]);
  };
  view.querySelectorAll(".tpl-row").forEach(el => el.onclick = () => open(el));
  const first = view.querySelector(".tpl-row");
  if (first) open(first);
  document.getElementById("rowfilter").oninput = ev => {
    const q = ev.target.value.toLowerCase();
    view.querySelectorAll(".tpl-row").forEach(el => el.style.display = el.dataset.text.includes(q) ? "" : "none");
  };
}

// ---------- 5. Jobs (live.js draws the page) ----------
function bar(done, total) { return `<div class="bar"><span class="ok" style="width:${total ? 100 * done / total : 0}%"></span></div>`; }

// ---------- 5c. Tiebreaker (the recorded lines of the shown job, read by live.js) ----------
let V = null;   // the shown job's views, set by live.js before it draws the tiebreaker or a tree
// The job shown, by company, 10-K and start (never its folder ID), and where to choose another.
const jobShown = L => `<div class="card pad" style="margin:-6px 0 14px"><div class="row-flex"><span class="note">Job shown</span><b>${esc(L.title)}</b><span class="note">${esc(L.sub)}</span>
  <span class="spacer"></span><a class="btn sm" href="#/jobs">Choose another job</a></div></div>`;
const viewNote = () => `${V.label ? jobShown(V.label) : ""}<p class="note" style="margin:-4px 0 14px">Read from the job's own records now. Model explanations are display text, with tag names replaced by the line's label.</p>`;
const TOOL = {printed_page: "reads the printed page", definition: "reads the accounting definition", search_notes: "searches the notes",
  read_passage: "reads a passage", calc_children: "checks which lines add up", breakdowns: "looks at the reported parts",
  rows_of: "lists template rows", recall: "looks up an earlier decision"};
const rowList = ids => ids.map(r => `“${esc(rowName(r))}”`).join(" and ");
const rowsSaid = r => Array.isArray(r) ? `pieces in ${r.map(x => `“${rowName(x)}”`).join(" and ")}` : `“${rowName(r)}”`;
function stepPlain(s, row) {
  if (/context refused/.test(s.ended || "")) return "ran out of room: the question was longer than the model's allowance";
  if (s.call_error) return "the model call failed";
  const r = s.row || row;
  if (s.ended === "answer refused") return r ? `proposes ${rowsSaid(r)}; code refuses the answer` : "answers; code refuses the answer";
  if (s.action === "answer") return r ? `answers ${rowsSaid(r)}` : "answers";
  if (s.ended === "repeat reminder") return "repeats an earlier request; reminded to move on";
  const t = TOOL[s.action] || {other_periods: "reads other years' amounts"}[s.action] || s.action || "step";
  return s.tool_error ? `${t}: the tool returned an error` : t;
}
// One step in a model's list: what it did, its recorded time and tokens, its stated reason, and what code refused or the tool reported.
const stepItem = (s, row, attr = "") => `<li ${attr}><b>${esc(stepPlain(s, row))}</b>${s.seconds ? ` <span class="note">${esc(s.seconds)} s</span>` : ""}${s.tokens ? ` <span class="note">· ${s.tokens.toLocaleString("en-US")} tokens</span>` : ""}
  ${s.why ? `<div class="note">${esc(human(s.why))}</div>` : ""}${(s.refused || []).map(r => `<div class="note" style="color:var(--red)">Refused: ${esc(human(r))}</div>`).join("")}
  ${s.tool_error ? `<div class="note" style="color:var(--amber)">Tool error: ${esc(human(s.tool_error))}</div>` : ""}</li>`;
function gradeBadge(g) {
  if (!g) return "";
  return badge(g.correctness === "correct" ? "matches the answer key" : g.correctness === "wrong" ? "differs from the answer key" : g.correctness,
    g.correctness === "correct" ? "green" : g.correctness === "wrong" ? "red" : "grey");
}
const gradeText = g => g ? `Answer key: ${rowList(g.rows)}${g.accepted.length ? `; also accepted: ${rowList(g.accepted)}` : ""}. An answer key is a stronger model's reviewed spread, not proof.` : "";
// The tiebreaker as the job's records name it (model and where it ran); "haiku" is only the role's name in the records.
// A model's plain name; an ID not listed is shown as recorded.
const MODEL_NAMES = {"anthropic/claude-haiku-5.5": "Claude Haiku 5.5", "openai/gpt-6-luna": "GPT-6 Luna", "deepseek/deepseek-v4.1-flash-1017": "DeepSeek V4.1 Flash",
  "qwen/qwen3.8-flash": "Qwen 3.8 Flash", "z-ai/glm-5.3-flash": "GLM 5.3 Flash", "openai/text-embedding-3-small": "OpenAI text-embedding-3-small",
  "qwen3.5:4b-q4_K_M": "Qwen 3.5 4B", "minicpm5:2b-q4_K_M": "MiniCPM 5 2B", "ling3-tiny:q4_K_M": "Ling 3.0 Tiny"};
const modelName = id => MODEL_NAMES[id] || id || "";
const tbName = t => t.tiebreaker ? `${modelName(t.tiebreaker.model)} (${t.tiebreaker.where})` : "the tiebreaker";
// A model's placement: its row, or the pieces it split the line into, each with its amount.
const placedText = o => o && o.pieces && o.pieces.length ? o.pieces.map(x => `“${esc(rowName(x.row))}” ${money(x.amount)}`).join(" and ")
  : o && o.row ? `“${esc(rowName(o.row))}”` : `<span class="note">no row</span>`;
const roomText = (why, name = "The tiebreaker") => {
  const m = /counted (\d+) input tokens, budget (\d+)/.exec(why || "");
  return m ? `${esc(name)} ran out of room: the question came to ${Number(m[1]).toLocaleString("en-US")} tokens, over its allowance of ${Number(m[2]).toLocaleString("en-US")}.` : esc(why || "");
};
function tbResult(t) {
  const name = esc(tbName(t)), c = t.committed || [];
  const used = c.length ? `The spread commits ${c.map(x => `${money(x.amount)} to “${esc(rowName(x.row))}”`).join(" and ")}.`
    : t.used.length ? `The spread uses ${rowList(t.used)}.` : "The spread places the line nowhere.";
  if (t.vote.result === "checker") return `${name} agreed with the checker, so the checker's placement replaced the analyst's. ${used}`;
  if (t.vote.result === "analyst") return `${name} agreed with the analyst, so the analyst's placement stands. ${used}`;
  if (t.vote.result === "no vote") return `No vote. ${/did not finish/.test(t.vote.why || "") ? roomText(t.vote.why, tbName(t)) : esc(t.vote.why || "")}
    ${t.used.length === 1 && t.used[0] === t.analyst.row ? "Without a vote, the analyst's row stands." : ""} ${used}`;
  return `${esc(t.vote.why || t.vote.result)} ${used}`;
}
let TB = {gen: 0, timer: null};
function pageTiebreak() {
  const T = V.tiebreaks, votes = T.filter(t => t.vote.result === "analyst" || t.vote.result === "checker");
  const names = [...new Set(T.map(tbName))].join(", ");
  view.innerHTML = `<div class="page">
  ${head("Tiebreaker", "When the analyst and the checker put a line in different rows, the line is not resolved. A third model answers the same question on its own, with the same tools; its answer decides between the two. In this job the tiebreaker was " + esc(names) + ".")}
  ${viewNote()}
  <div class="grid g4" style="margin-bottom:16px">
    <div class="card pad stat"><div class="v">${T.length}</div><div class="l">lines not resolved, sent to the tiebreaker</div></div>
    <div class="card pad stat"><div class="v">${votes.length}</div><div class="l">decided by the tiebreaker's vote</div></div>
    <div class="card pad stat"><div class="v">${T.length - votes.length}</div><div class="l">no vote${T.length - votes.length && T.filter(t => !votes.includes(t)).every(t => /did not finish/.test(t.vote.why || "")) ? ": the tiebreaker did not finish" : ""}</div></div>
    <div class="card pad stat"><div class="v">$${T.reduce((a, t) => a + t.cost_usd, 0).toFixed(4)}</div><div class="l">the tiebreaker's cost on these lines</div></div></div>
  <p class="note" style="margin:0 0 14px">Two models out of three agreeing is not proof the row is right. The badge on each line compares the row the spread used with the answer key.</p>
  ${T.map((t, i) => tbCard(t, i)).join("")}</div>`;
  view.querySelectorAll("[data-tbopen]").forEach(b => b.onclick = () => {
    const d = document.getElementById(`tbd-${b.dataset.tbopen}`); d.hidden = !d.hidden; b.textContent = d.hidden ? "Show every step" : "Hide the steps";
  });
  view.querySelectorAll("[data-tbplay]").forEach(b => b.onclick = () => tbPlay(+b.dataset.tbplay));
}
function tbCol(title, o, cls, seq) {
  if (!o) return `<div><h3>${esc(title)}</h3><p class="note">no record</p></div>`;
  const last = o.steps.length - 1;
  return `<div class="tb-col ${cls}"><h3>${esc(title)}</h3><p class="note" style="margin:0 0 6px">${esc(modelName(o.model))}</p>
    <ol class="tb-steps">${o.steps.map((s, k) => stepItem(s, k === last ? o.row : null, `data-seq="${seq()}"`)).join("")}</ol>
    ${o.reason ? `<div class="why" data-seq="${seq()}"><b>Its reason:</b> ${esc(human(o.reason))}${o.quote ? `<br><b>Quote:</b> “${esc(o.quote)}”` : ""}</div>` : ""}</div>`;
}
function tbCard(t, i) {
  let n = 0;
  const seq = () => n++;
  // The detail columns take their sequence numbers first, interleaved with the summary boxes in the order the job ran them.
  // With a second look, the analyst's blind answer first and its revised one after the checker's, as the job ran them.
  // A kept retry can repeat the first placement; only `revised` says the placement changed.
  const retry = !!t.analyst_first, sl = retry || !!(t.second_look && t.second_look.ran), k = sl ? 4 : 3;
  const colA = tbCol("1 · Analyst", retry ? t.analyst_first : t.analyst, "", seq), boxA = seq();
  const colC = tbCol("2 · Checker", t.checker, "chk", seq), boxC = seq(), flag = seq();
  const colS = retry ? tbCol("3 · Analyst's second look, seeing the checker's answer", t.analyst, "", seq) : "";
  const boxS = sl ? seq() : null;
  const colH = tbCol(`${k} · Tiebreaker: ${tbName(t)}`, t.haiku, "hk", seq), boxH = seq(), res = seq();
  const hk = t.haiku && (t.haiku.row || (t.haiku.pieces || []).length) ? placedText(t.haiku) : `<span class="note">did not finish</span>`;
  return `<div class="card tb" id="tb-${i}" style="margin-bottom:14px"><div class="hd row-flex"><div><h2>${esc(t.label)} <span class="note" style="font-weight:400">${money(t.amount)}</span></h2>
      <p class="sub">${esc(STMT[t.statement])} · ${esc(sectionName(t.section))}</p></div><span class="spacer"></span>${gradeBadge(t.grade)}</div>
    <div class="bd"><div class="tb-flow">
      <div class="tb-box" data-seq="${boxA}"><div class="note">1 · Analyst</div>${placedText(retry ? t.analyst_first : t.analyst)}</div><div class="tb-arrow">→</div>
      <div class="tb-box chk" data-seq="${boxC}"><div class="note">2 · Checker</div>${placedText(t.checker)}</div><div class="tb-arrow">→</div>
      <div class="tb-box flag" data-seq="${flag}"><div class="note">Code compares</div>Not resolved: the rows differ</div><div class="tb-arrow">→</div>
      ${sl ? `<div class="tb-box" data-seq="${boxS}"><div class="note">3 · Analyst's second look, seeing the checker's answer</div>${t.revised ? `${placedText(t.analyst)}; still differs from the checker` : "kept its first placement"}</div><div class="tb-arrow">→</div>` : ""}
      <div class="tb-box hk" data-seq="${boxH}"><div class="note">${k} · Tiebreaker (${esc(tbName(t))})</div>${hk}</div></div>
      <div class="tb-result" data-seq="${res}"><b>Result.</b> ${tbResult(t)}</div>
      <p class="note" style="margin:6px 0 0">${gradeText(t.grade)} The tiebreaker's cost on this line: $${t.cost_usd.toFixed(4)}.</p>
      <div class="row-flex" style="margin-top:10px"><button class="btn sm primary" data-tbplay="${i}">Play this line</button><button class="btn sm" data-tbopen="${i}">Show every step</button></div>
      <div class="tb-detail grid ${retry ? "g4" : "g3"}" id="tbd-${i}" hidden>${colA}${colC}${colS}${colH}</div></div></div>`;
}
// Plays one line in the order the job ran it: the analyst's steps, the checker's, the comparison, the analyst's second
// look when one ran, the tiebreaker's, the vote.
function tbPlay(i) {
  clearTimeout(TB.timer);
  const gen = ++TB.gen, card = document.getElementById(`tb-${i}`), d = document.getElementById(`tbd-${i}`);
  d.hidden = false; card.querySelector("[data-tbopen]").textContent = "Hide the steps";
  const els = [...card.querySelectorAll("[data-seq]")].sort((a, b) => a.dataset.seq - b.dataset.seq);
  els.forEach(e => { e.classList.add("dim"); e.classList.remove("lit"); });
  let k = 0;
  const next = () => {
    if (gen !== TB.gen || location.hash !== "#/tiebreak") return;
    const e = els[k++]; if (!e) return;
    e.classList.remove("dim"); e.classList.add("lit"); setTimeout(() => e.classList.remove("lit"), 900);
    TB.timer = setTimeout(next, e.matches(".tb-box, .tb-result") ? 1400 : 900);
  };
  next();
}

// ---------- 5d. Tree of thought (the recorded repair of one section, read by live.js) ----------
const nodeName = id => id === "0" ? "Start" : `Branch ${id.slice(2)}`;
// Runs recorded before the feedback setting existed carry none: the critic was recorded only. With feedback on, a verdict
// reaches a proposer only when its branch is grown; the tree records which ones did (none recorded: delivery unknown).
const criticSent = (T, id) => {
  const F = T.feedback;
  if (!F) return "delivery not recorded";
  if ((F.dispatched || []).includes(id)) return `passed on to the proposer growing ${nodeName(id)}`;
  if ((F.prepared || []).includes(id)) return `prepared for the proposer, not sent`;
  return `not passed on: ${nodeName(id)} was not grown`;
};
const criticUse = (T, ids) => T.settings.feedback === "on"
  ? `With feedback on, a verdict goes to the proposer when its branch is grown, as an opinion, not evidence; the chooser
    gets no explicit verdict. ${ids.map(id => `${esc(nodeName(id))}: ${criticSent(T, id)}.`).join(" ")}`
  : "It is recorded only: neither the proposer nor the chooser sees it.";
function nodeTitle(n) {
  if (n.status === "root") return "The section as drafted";
  if (n.action === "move") return `Move to “${esc(rowName(n.row))}”`;
  if (n.action === "split") return `Split it into pieces`;
  if (n.action === "combined") return "Call it a combined line (leave it unplaced)";
  return esc(n.action || "");
}
const failText = f => `${esc(CHECK[f.check.startsWith("rows add to") ? "rows add to the printed total" : f.check] || f.check)}${f.gap ? ` (off by ${money(Math.abs(Number(f.gap)))})` : ""}`;
function nodeStatus(n) {
  if (n.status === "root") return badge(`${n.failures.length} code checks fail`, "red");
  if (n.status === "rejected") return badge("broke a code rule: pruned by code", "red");
  if (n.complete && n.eligible) return badge("passes every code check", "green");
  return badge(`still fails ${n.failures.length} check${n.failures.length === 1 ? "" : "s"}`, "amber");
}
// Every line the checker reviewed after the tree (none, one or several); a view built before `rechecks` has one.
const rechecksOf = T => T.rechecks || (T.recheck ? [{line: T.line, line_amount: T.line_amount, first: T.first, recheck: T.recheck, grade: T.grade}] : []);
const firstSaid = f => f.class ? `“${esc(f.class)}”, not placed` : f.pieces && f.pieces.length ? placedText(f) : f.row ? `“${esc(rowName(f.row))}”` : "not placed";
// A whole decomposition as recorded: every piece with its row and amount, and the rest on the answer's row.
const wholeSaid = w => (w.pieces || []).length ? w.pieces.map(x => `“${esc(rowName(x.row))}” ${money(x.amount)}`).join(", ")
  + (w.remainder != null && Number(w.remainder) !== 0 ? `, the rest ${money(w.remainder)} in “${esc(rowName(w.row))}”` : "") : `the whole line in “${esc(rowName(w.row))}”`;
// The chosen branch and the checker after the tree; a view built before whole decompositions has rows only. A checker
// that did not finish gave no answer and is never shown as having answered.
const branchSaid = x => x.candidate ? wholeSaid(x.candidate) : x.recheck.candidate_row ? `“${esc(rowName(x.recheck.candidate_row))}”` : "not recorded";
// A review that settled: "settled" as the run records it (repair.SETTLED), "agreed" in views made before.
const settled = R => ["settled", "agreed"].includes(R.status);
const answered = R => R.answer !== undefined ? !!R.answer : settled(R) || R.status === "disagreed";
const checkerSaid = R => !answered(R) ? `no answer${R.termination ? ` (it stopped: ${esc(human(R.termination))})` : ""}` : R.answer ? wholeSaid(R.answer) : `“${esc(rowName(R.row))}”`;
// The review's recorded status and reason, in words.
const reviewSaid = R => (settled(R) ? "the checker's whole answer matches the chosen branch" : {disagreed: "the checker's whole answer differs from the chosen branch",
  unfinished: "the checker did not finish, so there was no answer to compare", unresolved: "the review could not be settled", refused: "code refused the review"}[R.status]
  || `the review did not complete (status ${esc(R.status == null ? "none" : R.status)})`) + (R.why && !settled(R) && R.status !== "disagreed" ? `: ${esc(human(R.why))}` : "");
// The chooser's earlier attempts at one level, when code refused them and asked again.
const attemptsSaid = p => (p.attempts || []).length > 1 ? ` It took ${p.attempts.length} attempts: ${p.attempts.slice(0, -1).map(a => `attempt ${esc(a.attempt)} was refused${a.errors.length ? ` (${a.errors.map(e => esc(human(typeof e === "string" ? e : JSON.stringify(e)))).join("; ")})` : ""}`).join("; ")}.` : "";
function treeEvents(T) {
  const kids = id => T.nodes.filter(n => n.parent === id), byId = id => T.nodes.find(n => n.id === id), m = T.settings.models;
  const ev = [{show: ["0"], checked: ["0"], say: `<b>Start.</b> ${T.line ? `In the first pass, “${esc(T.line)}” (${money(T.line_amount)}) was ${T.first.class ? `not placed: the analyst answered that ${esc(T.first.class)}` : `placed in ${firstSaid(T.first)}`}. ` : ""}
    Code checks on ${esc(sectionName(T.section))} fail: ${T.root_failures.map(failText).join("; ")}. The tree tries small changes to the draft.`}];
  T.picks.forEach((p, li) => {
    const parents = li === 0 ? ["0"] : T.picks[li - 1].picked, level = parents.flatMap(id => kids(id).map(n => n.id));
    if (li === 0) T.meaning.forEach(q => ev.push({say: `The proposer asked what the filing means: “${esc(q.question)}” A second model (${esc(modelName(m.meaning))}) answered from the filing:
      “${esc(human(q.answer))}”`}));
    parents.forEach(id => ev.push({show: kids(id).map(n => n.id), say: `The proposer (${esc(modelName(m.generator))}) suggests ${kids(id).length} change${kids(id).length === 1 ? "" : "s"} starting from ${esc(nodeName(id))}:
      ${kids(id).map(n => `${esc(nodeName(n.id))}, ${nodeTitle(n).toLowerCase()}`).join("; ")}.`}));
    ev.push({checked: level, say: `The evaluator (code checks) tests every proposal: ${level.map(byId).map(n => `${esc(nodeName(n.id))} ${n.status === "rejected" ? "broke a code rule and is pruned"
      : n.complete && n.eligible ? "passes every check" : `still fails ${n.failures.length}`}`).join("; ")}.`});
    const crit = level.filter(id => byId(id).critic);
    if (crit.length) ev.push({critic: crit, say: `A second opinion (${esc(modelName(m.critic))}) is recorded: ${crit.map(id => `${esc(nodeName(id))} “${esc(byId(id).critic.verdict)}”`).join("; ")}.
      ${criticUse(T, crit)}`});
    const saw = `It saw ${p.shown.map(byId).map(n => `${esc(nodeName(n.id))} (${n.complete && n.eligible ? "passes every check" : `still fails ${n.failures.length}`})`).join(", ")}.`;
    ev.push({picked: p, say: (p.decision === "expand" ? `The chooser (${esc(modelName(m.controller))}) keeps ${p.picked.map(nodeName).join(" and ")} to grow.`
      : `The chooser (${esc(modelName(m.controller))}) selects ${p.picked.map(nodeName).join(", ")}${p.decision === "stop"
        ? ` and stops the tree at level ${esc(p.level)} of at most ${esc(T.settings.depth)}.` : p.level >= T.settings.depth
        ? ` at level ${esc(p.level)}, the maximum.` : ` at level ${esc(p.level)}.`}`) + `${attemptsSaid(p)} ${saw} Its reason: “${esc(human(p.reason || ""))}”`});
  });
  const RS = rechecksOf(T);
  if (!RS.length) ev.push({recheck: true, say: "After the tree, no line was rechecked: the records hold no review by the checker."});
  RS.forEach(x => ev.push({recheck: true, say: `After the tree, verification of “${esc(x.line)}”: the checker${x.recheck.model ? ` (${esc(modelName(x.recheck.model))})` : ""} is asked to place the line again on its own, without seeing the chosen branch (${x.recheck.steps.length} steps).
    ${answered(x.recheck) ? `It answers ${checkerSaid(x.recheck)}; the chosen branch said ${branchSaid(x)}.` : `It gave ${checkerSaid(x.recheck)}. The chosen branch said ${branchSaid(x)}.`} Result: ${reviewSaid(x.recheck)}.`}));
  const causes = RS.filter(x => !settled(x.recheck)).map(x => `for “${esc(x.line)}”, ${reviewSaid(x.recheck)}`).join("; ");
  const ending = T.termination === "withdrawn" ? `<b>The repair is withdrawn and the first draft is kept</b>: ${causes || esc(human(T.why || "the records give no reason"))}. ${RS.map(x => `“${esc(x.line)}” stays as the first pass left it: ${firstSaid(x.first)}.`).join(" ")}`
    : `<b>The tree ended: ${esc(T.termination)}.</b> ${esc(human(T.why || ""))}`;
  ev.push({end: true, say: `${ending}
    ${RS.filter(x => x.grade).map(x => `Answer key for “${esc(x.line)}”: ${rowList(x.grade.rows)}${x.grade.accepted.length ? `, also accepted ${rowList(x.grade.accepted)}` : ""}.`).join(" ")}`});
  return ev;
}
let TR = {gen: 0, timer: null, i: 0, playing: false, speed: 1};
function pageTree() {
  const T = V.tree, s = T.settings;
  TR = {gen: TR.gen + 1, timer: null, i: 0, playing: true, speed: 1, ev: treeEvents(T)};
  view.innerHTML = `<div class="page">
  ${head("Tree of thought", `When a section still fails its code checks after the first spread, including because lines remain unplaced, the tree tries small changes to the draft: a proposer suggests branches, code checks each one (the evaluator), and a chooser decides which branches to grow and which to pick. A check after the tree decides whether the change goes in.`)}
  <p class="note" style="margin:-10px 0 10px">One step of the tree is one change to the draft: move a line to a row, split it into pieces, or call it a combined line. Each step proposes up to ${s.children} changes; the chooser keeps up to ${s.beam} per level to grow; the tree can grow through at most ${s.depth} levels; the chooser may stop earlier, and it selects one branch.</p>
  ${viewNote()}
  <div class="card pad" style="margin-bottom:14px"><div class="row-flex"><b>${esc(T.company_name || T.company)}</b><span class="note">${esc(STMT[T.statement])} · ${esc(sectionName(T.section))}</span>
    <span class="spacer"></span><span class="note">Speed</span><div class="tabs">${[1, 2, 4].map(x => `<button data-ts="${x}" class="${x === 1 ? "on" : ""}">${x}×</button>`).join("")}</div>
    <button class="btn sm" id="tr-play">Pause</button><button class="btn sm" id="tr-next">Next step</button><button class="btn sm" id="tr-all">Show all</button><button class="btn sm" id="tr-restart">Restart</button></div>
    <div class="legend"><span><i style="background:var(--green-bg);border:1px solid var(--green)"></i>passes every code check</span><span><i style="background:var(--amber-bg);border:1px solid var(--amber)"></i>still fails a check</span>
      <span><i style="background:var(--red-bg);border:1px dashed var(--red)"></i>broke a code rule: pruned by code</span><span><i style="background:#fff;border:2px solid var(--blue)"></i>chosen</span></div></div>
  <div class="tree-wrap"><div class="card tree-card"><div class="tree" id="tree"></div></div>
    <div class="card"><div class="hd"><h3>What happened, step by step</h3><p class="note">Click a branch for its proposal summary, code checks and the second opinion.</p></div><div class="timeline" id="tr-log"></div></div></div>
  <div id="tr-final"></div></div>`;
  document.getElementById("tr-play").onclick = b => { TR.playing = !TR.playing; b.target.textContent = TR.playing ? "Pause" : "Play"; clearTimeout(TR.timer); if (TR.playing) trTick(TR.gen); };
  document.getElementById("tr-next").onclick = () => { trStop(); trShow(Math.min(TR.i + 1, TR.ev.length)); };
  document.getElementById("tr-all").onclick = () => { trStop(); trShow(TR.ev.length); };
  document.getElementById("tr-restart").onclick = () => { trStop(); trShow(1); TR.playing = true; document.getElementById("tr-play").textContent = "Pause"; trTick(TR.gen); };
  view.querySelectorAll("[data-ts]").forEach(b => b.onclick = () => { TR.speed = +b.dataset.ts; view.querySelectorAll("[data-ts]").forEach(x => x.classList.toggle("on", x === b)); });
  trShow(1); trTick(TR.gen);
}
function trStop() { TR.playing = false; clearTimeout(TR.timer); document.getElementById("tr-play").textContent = "Play"; }
function trTick(gen) {
  clearTimeout(TR.timer);
  TR.timer = setTimeout(() => {
    if (gen !== TR.gen || !TR.playing || location.hash !== "#/tree") return;
    if (TR.i >= TR.ev.length) { trStop(); return; }
    trShow(TR.i + 1); trTick(gen);
  }, 2600 / TR.speed);
}
// Draws the tree as it stood after the first i recorded steps.
function trShow(i) {
  const T = V.tree, prevShown = new Set(TR.state ? TR.state.shown : []);
  const st = {shown: new Set(), checked: new Set(), critic: new Set(), grown: new Set(), chosen: null, recheck: false, end: false};
  TR.ev.slice(0, i).forEach(e => {
    (e.show || []).forEach(x => st.shown.add(x)); (e.checked || []).forEach(x => st.checked.add(x)); (e.critic || []).forEach(x => st.critic.add(x));
    if (e.picked) { if (e.picked.decision === "expand") e.picked.picked.forEach(x => st.grown.add(x)); else st.chosen = e.picked.picked[0]; }
    if (e.recheck) st.recheck = true; if (e.end) st.end = true;
  });
  TR.i = i; TR.state = {shown: [...st.shown]};
  // Layout: leaves take the next column left to right; a parent sits above the middle of its children.
  const kids = id => T.nodes.filter(n => n.parent === id).sort((a, b) => a.id.localeCompare(b.id, undefined, {numeric: true}));
  const pos = {}; let slot = 0;
  const place = n => { const k = kids(n.id); if (!k.length) pos[n.id] = slot++; else { k.forEach(place); pos[n.id] = (pos[k[0].id] + pos[k[k.length - 1].id]) / 2; } };
  place(T.nodes.find(n => n.parent === null));
  // Boxes shrink to fit the card's width so the whole tree shows without scrolling sideways.
  const GX = 12, H = Math.max(150, TR.H || 0), GY = 56, W = Math.max(140, Math.min(210, Math.floor((document.getElementById("tree").clientWidth - (slot - 1) * GX) / slot))), X = id => pos[id] * (W + GX), Y = n => n.depth * (H + GY);
  const depth = Math.max(...T.nodes.map(n => n.depth)), width = slot * (W + GX) - GX, height = (depth + 1) * (H + GY) - GY;
  const path = new Set(); for (let id = st.chosen; id; id = (T.nodes.find(n => n.id === id) || {}).parent) path.add(id);
  const vis = T.nodes.filter(n => st.shown.has(n.id));
  const edges = vis.filter(n => n.parent).map(n => {
    const p = T.nodes.find(x => x.id === n.parent), x1 = X(p.id) + W / 2, y1 = Y(p) + H, x2 = X(n.id) + W / 2, y2 = Y(n), my = (y1 + y2) / 2;
    const cls = path.has(n.id) ? "on-path" : st.checked.has(n.id) && n.status === "rejected" ? "dropped" : "";
    return `<path class="${cls}" d="M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}"/>`;
  }).join("");
  const box = n => {
    const ck = st.checked.has(n.id), cls = !ck ? "fresh" : n.status === "root" ? "root" : n.status === "rejected" ? "bad" : n.complete && n.eligible ? "good" : "warn";
    const other = n.line && n.line !== T.line ? `<div class="tn-l">line: ${esc(n.line)}</div>` : "";
    const gap = ck && n.status !== "root" ? n.failures.find(f => f.gap) : null;
    return `<div class="tn ${cls} ${prevShown.has(n.id) ? "" : "new"} ${st.chosen === n.id ? "chosen" : ""}" data-n="${esc(n.id)}" style="left:${X(n.id)}px;top:${Y(n)}px;width:${W}px;height:${H}px">
      <div class="tn-id">${esc(nodeName(n.id))}</div><div class="tn-h">${nodeTitle(n)}</div>${other}
      <div class="tn-b">${ck ? nodeStatus(n) : `<span class="note">code is checking…</span>`}
      ${gap ? `<span class="note">off by ${money(Math.abs(Number(gap.gap)))}</span>` : ""}
      ${st.critic.has(n.id) && n.critic ? badge(`2nd opinion: ${n.critic.verdict}`, "violet") : ""}
      ${st.grown.has(n.id) ? badge("grown", "blue") : ""}${st.chosen === n.id ? badge("chosen", "blue") : ""}
      ${ck && n.depth === depth && n.status !== "rejected" && st.chosen !== n.id && (st.chosen || st.end) ? `<span class="note">last level: not chosen</span>` : ""}</div></div>`;
  };
  document.getElementById("tree").innerHTML = `<div style="position:relative;width:${width}px;height:${height}px;margin:0 auto">
    <svg width="${width}" height="${height}" style="position:absolute;inset:0">${edges}</svg>${vis.map(box).join("")}</div>`;
  // Every box takes the height of the fullest one; when a badge would be cut off, draw again taller.
  const over = [...view.querySelectorAll(".tn")].filter(e => e.scrollHeight > e.clientHeight + 1);
  const need = over.length ? Math.max(...over.map(e => e.scrollHeight + 4)) : 0;
  if (need > H) { TR.H = need; TR.state = {shown: [...prevShown]}; return trShow(i); }
  view.querySelectorAll(".tn").forEach(el => el.onclick = () => nodeDetail(T.nodes.find(n => n.id === el.dataset.n)));
  const log = document.getElementById("tr-log");
  log.innerHTML = TR.ev.slice(0, i).map((e, k) => `<div class="ev ${k === i - 1 ? "cur" : ""}" style="grid-template-columns:28px 1fr;cursor:default"><span class="t">${k + 1}</span><span>${e.say}</span></div>`).join("");
  log.scrollTop = log.scrollHeight;
  const RS = rechecksOf(T);
  document.getElementById("tr-final").innerHTML = st.recheck ? `<div class="card" style="margin-top:14px"><div class="hd row-flex"><div><h2>Verification after the tree</h2>
      <p class="sub">The checker solves each reviewed line again without seeing the chosen branch. Only agreement lets the change in.</p></div><span class="spacer"></span>
      ${st.end ? (T.termination === "withdrawn" ? badge("withdrawn: first draft kept", "amber") : badge(T.termination, "grey")) : ""}</div>
    <div class="bd">${RS.map(x => recheckCard(x, st.end)).join("") || `<span class="note">No line was rechecked after the tree.</span>`}</div></div>` : "";
}
// One reviewed line in the verification card: both whole decompositions, the review's result, the reference at the end.
function recheckCard(x, end) {
  const R = x.recheck;
  return `<h3 style="margin-top:6px">“${esc(x.line)}” ${money(x.line_amount)}</h3><div class="kv"><div class="k">Chosen branch says</div><div>${branchSaid(x)}</div>
      <div class="k">Checker says</div><div>${checkerSaid(R)}</div><div class="k">Review result</div><div>${reviewSaid(R)}</div>${end && x.grade ? `<div class="k">Answer key</div><div>${rowList(x.grade.rows)}${x.grade.accepted.length ? `; also accepted: ${rowList(x.grade.accepted)}` : ""} <span class="note">(another model's answer key, not proof)</span></div>` : ""}</div>
      <details style="margin-top:10px"><summary><b>The checker's ${R.steps.length} steps</b></summary><ol class="tb-steps">${R.steps.map((s, k) => stepItem(s, k === R.steps.length - 1 && answered(R) ? R.row : null)).join("")}</ol>
        ${R.reason ? `<div class="why"><b>Its reason:</b> ${esc(human(R.reason))}</div>` : ""}</details>`;
}
function nodeDetail(n) {
  const T = V.tree, ch = n.change || {};
  drawer(`${nodeName(n.id)}: ${n.status === "root" ? "the section as drafted" : nodeTitle(n).replace(/&[^;]+;/g, "")}`, `
    <div class="kv" style="margin-bottom:12px">${n.line ? `<div class="k">Line</div><div>${esc(n.line)}</div>` : ""}
      ${n.parent ? `<div class="k">Grew from</div><div>${esc(nodeName(n.parent))}</div>` : ""}
      <div class="k">Code checks</div><div>${nodeStatus(n)}</div>
      ${n.failures.length ? `<div class="k">Still failing</div><div>${n.failures.map(failText).join("<br>")}</div>` : ""}
      ${ch.cleared && ch.cleared.length ? `<div class="k">Cleared</div><div>${ch.cleared.map(c => failText({check: c})).join("<br>")}</div>` : ""}
      ${ch.new && ch.new.length ? `<div class="k">Brought back</div><div>${ch.new.map(c => failText({check: c})).join("<br>")}</div>` : ""}
      ${n.rejected.length ? `<div class="k">Rules broken</div><div>${n.rejected.map(r => `${esc(r.rule)}<div class="note">${esc(human(r.why))}</div>`).join("")}</div>` : ""}</div>
    ${n.status !== "root" ? `<p class="note">Proposal summary: the change, its reason and quote. Split amounts and evidence references are in the run record.</p>` : ""}
    ${n.reason ? `<h3>The proposer's reason</h3><div class="why">${esc(human(n.reason))}${n.quote ? `<br><b>Quote:</b> “${esc(n.quote)}”` : ""}</div>` : ""}
    ${n.critic ? `<h3 style="margin-top:14px">Second opinion${T.settings.feedback === "on" ? "" : " (recorded only, not used)"}: ${esc(n.critic.verdict)}</h3><div class="why">${esc(human(n.critic.reason))}</div>
      <p class="note">${criticUse(T, [n.id])}</p>` : ""}
    ${T.picks.filter(p => p.shown.includes(n.id)).map(p => `<h3 style="margin-top:14px">The chooser, level ${esc(p.level)}: ${p.decision === "expand" ? (p.picked.includes(n.id) ? "grew this branch" : "did not grow it") : (p.picked.includes(n.id) ? "selected this branch" : "did not select it")}</h3>
      <div class="why">${esc(human(p.reason || ""))}${attemptsSaid(p)}</div>`).join("")}
    ${n.status === "root" ? rechecksOf(T).filter(x => x.first.why).map(x => `<h3 style="margin-top:14px">The first pass on “${esc(x.line)}”</h3><div class="why">${esc(human(x.first.why))}</div>`).join("") : ""}`);
}

// ---------- 7. Evals: the operations view of the shown job (live.js draws the page) ----------
// The parts a job ran with, from its records; a job recorded before the Run page's switches does not name them.
const partsSaid = p => !p ? "parts not recorded" : p.tiebreaker === "on" && p.tree === "on" ? "tiebreaker and tree on"
  : ["tiebreaker", "tree"].filter(k => p[k] === "off").map(k => `${k} off`).join(", ");
// The job's recorded grade against the AI reference, for the statements named and overall.
function gradeCard(gr, sts) {
  if (!gr) return `<p>${badge("not compared", "grey")} This job was not compared with an answer key.</p>
    <p class="note">An answer key is a reference spread of the same 10-K, made by a stronger model and reviewed; it is used to measure the agents, not as proof. The comparison runs at the end of a job only when an answer key for that company's 10-K is set up for the job.</p>`;
  const row = k => { const c = gr.counts[k]; return c ? `<tr><td>${esc(STMT[k] || "All statements")}</td><td class="num">${Number(c.correct || 0)}</td><td class="num">${Number(c.wrong || 0)}</td>
    <td class="num">${Number(c["no answer"] || 0)}</td><td class="num">${Number(c.completed || 0)}</td></tr>` : ""; };
  return `<p class="note">${badge(`compared; ${partsSaid(gr.parts)}`, "grey")} ${esc(gr.label || "")}. Answer key: <span class="mono">${esc(gr.reference)}</span>, a stronger model's reviewed spread, not proof.</p>
    <table><tr><th></th><th class="num">Agrees</th><th class="num">Differs</th><th class="num">No answer</th><th class="num">Completed</th></tr>
    ${[...sts, "all"].map(row).join("")}</table>`;
}
// A job in plain words, from its manifest and the start time in its ID: "ACME CORP, 10-K 2025-12-31" and
// "Claude Haiku 5.5 analyst, GPT-6 Luna checker · started 18 Nov 2026 06:35"; the ID itself only as a fallback.
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
function jobStarted(id) {
  const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2})(\d{2})/.exec(id || "");
  return m ? `${Number(m[3])} ${MONTHS[Number(m[2]) - 1]} ${m[1]} ${m[4]}:${m[5]}` : "";
}
function jobLabel(x) {
  const m = x.manifest;
  const models = m ? String(m.lineup_label || m.lineup || "").replace(/^Online:\s*|^Local:\s*/, "").replace(/\s*\(.*\)\s*$/, "") : "";
  const tpl = m && m.template ? `${m.template.name} v${m.template.version}` : "";
  return {title: m ? `${m.company}, 10-K ${m.report_date}` : "Job started outside this app",
          sub: [jobStarted(x.job_id) && `started ${jobStarted(x.job_id)}`, tpl && `template ${tpl}`, models].filter(Boolean).join(" · ")};
}
// The job's caps in plain words: what was used against each limit, once the job has ended.
const CAP_NAMES = {requests: "Model requests", tokens: "Tokens", wall_s: "Time", dollars: "Dollars",
  haiku_requests: "Tiebreaker requests", haiku_dollars: "Tiebreaker dollars"};
const capValue = (k, v) => v === null || v === undefined ? "–" : k === "wall_s" ? `${(Number(v) / 60).toFixed(1)} min`
  : /dollars/.test(k) ? `$${Number(v).toFixed(2)}` : Number(v).toLocaleString("en-US");
// One limit as a bar: the track is the limit, the fill what was used; near the limit (90% or more) it says so in words.
function meter(c) {
  const known = c.used !== null && c.used !== undefined && Number(c.limit) > 0;
  const pct = known ? 100 * Number(c.used) / Number(c.limit) : 0, near = known && pct >= 90;
  const said = known ? `${capValue(c.cap, c.used)} of ${capValue(c.cap, c.limit)}` : `not recorded · limit ${capValue(c.cap, c.limit)}`;
  return `<div class="meter" title="${esc(CAP_NAMES[c.cap] || c.cap)}: ${esc(said)}${known ? ` (${pct.toFixed(1)}%)` : ""}">
    <div class="meter-l">${esc(CAP_NAMES[c.cap] || c.cap)}</div>
    <div class="meter-track"><div class="meter-fill${near ? " near" : ""}" style="width:${Math.min(100, pct).toFixed(1)}%"></div></div>
    <div class="meter-v"><b>${known ? `${pct < 1 && pct > 0 ? "<1" : pct.toFixed(0)}%` : "–"}</b> <span class="note">${esc(said)}${near ? " · near the limit" : ""}</span></div></div>`;
}
function operationsCards(ops) {
  const ms = Object.entries(ops.models);
  return `<div class="card"><div class="hd"><h2>Per model</h2></div><div class="bd"><table><tr><th>Model</th><th class="num">Calls</th><th class="num">Tokens</th><th class="num">Seconds</th><th class="num">Dollars</th></tr>
    ${ms.map(([k, v]) => `<tr><td>${esc(modelName(k))}<div class="note mono">${esc(k)}</div></td><td class="num">${Number(v.calls)}</td><td class="num">${Number(v.tokens).toLocaleString("en-US")}</td>
      <td class="num">${Number(v.seconds).toFixed(0)}</td><td class="num">${Number(v.cost_usd).toFixed(4)}</td></tr>`).join("") || `<tr><td colspan="5" class="note">no model calls recorded</td></tr>`}</table></div></div>
  <div class="card" style="margin-top:16px"><div class="hd"><h2>Budget used by this job</h2><p class="sub">A job is one company's 10-K, all three statements. Each bar is one limit the job ran under: the grey track is the limit, the filled part what the job used. The tiebreaker has its own smaller limits inside the job's.</p></div><div class="bd">
    ${ops.caps ? `${ops.caps.map(meter).join("")}<details style="margin-top:10px"><summary class="note">As a table</summary><table><tr><th>Limit on</th><th class="num">Used</th><th class="num">Limit</th><th class="num">Share used</th></tr>${ops.caps.map(c => `<tr><td>${esc(CAP_NAMES[c.cap] || c.cap)}</td>
      <td class="num">${esc(capValue(c.cap, c.used))}</td><td class="num">${esc(capValue(c.cap, c.limit))}</td>
      <td class="num">${c.used !== null && c.used !== undefined && Number(c.limit) ? `${(100 * Number(c.used) / Number(c.limit)).toFixed(1)}%` : "–"}</td></tr>`).join("")}</table></details>`
      : `<span class="note">not recorded yet: the job has not ended</span>`}</div></div>
  <div class="card" style="margin-top:16px"><div class="hd"><h2>Agreement with the answer key</h2><p class="sub">How many of the job's placements match the answer key, per statement, as the job's last step recorded it.</p></div><div class="bd">
    ${gradeCard(ops.grade, ["IS", "BS", "CF"])}</div></div>`;
}

// ---------- router ----------
function render() {
  const key = location.hash.replace("#/", "") || "home";
  if (ALIAS[key]) { location.replace("#/" + ALIAS[key]); return; }     // a typed address sent on to the page's own
  // Replies and timers still on their way to the page left draw nothing.
  LS.gen += 1; TB.gen += 1; clearTimeout(TB.timer); TR.gen += 1; clearTimeout(TR.timer);
  V = null;
  document.getElementById("tpl-current").textContent = TPL.label || "";
  document.querySelectorAll("#nav a").forEach(a => a.classList.toggle("active", a.getAttribute("href") === "#/" + key));
  closeDrawer();
  if (!SRV.on) {
    view.innerHTML = `<div class="page">${head("Spreading", "")}<div class="callout">These pages need the local server: run <span class="mono">python ui/server.py</span> and open the address it prints.</div></div>`;
    return;
  }
  (LIVE_PAGES[key] || pageHome)();
  // an address with no page says so above Start here
  if (key !== "home" && !LIVE_PAGES[key]) view.innerHTML = `<div class="callout" id="no-page">There is no page at <span class="mono">#/${esc(key)}</span>; this is Start here. The pages are in the sidebar.</div>` + view.innerHTML;
  document.querySelector("main").scrollTop = 0;
}
// Start: render() needs live.js (ALIAS, LS, the pages), a later script. A browser may run the state reply's
// callback while it is still fetching live.js, so nothing is drawn until live.js, at its end, calls pagesReady()
// and the state has been read.
let pagesLoaded = false, stateRead = false;
function startWhenReady() { if (pagesLoaded && stateRead) render(); }
function pagesReady() { pagesLoaded = true; startWhenReady(); }
window.addEventListener("hashchange", () => { if (pagesLoaded) render(); });
serverState().then(() => { stateRead = true; startWhenReady(); });
