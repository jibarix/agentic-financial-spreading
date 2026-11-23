// The pages. Loaded after app.js, whose router draws them. Every read is a GET by
// job or template ID, every action a POST with the session token (app.js `post`). Every value from the server is
// escaped (esc) before it goes into the page; no browser dialogs.

// What the pages chose. app.js render() adds one to gen before it draws a page, which drops every reply and timer still
// on its way to the page left.
const LS = {gen: 0, search: null, chosen: null, job: null, shown: null, events: [], from: 0, tree: 0, ask: null, by: "",
  spread: null, importSaid: null, parts: {tiebreaker: true, tree: true}, line: null, treeFolder: null,
  setupLineup: null};
// A job chosen (Start here's example, Show on Jobs, a job started on Run): its progress cursor, tree tab, drawn views and
// the open Results detail start over, so nothing of the job shown before carries to it.
function chooseJob(id) { Object.assign(LS, {job: id, shown: null, events: [], from: 0, tree: 0, spread: null, line: null, treeFolder: null}); V = null; }

async function get(path) {
  try {
    const r = await fetch(path, {cache: "no-store"});
    return {ok: r.ok, status: r.status, body: await r.json().catch(() => ({}))};
  } catch (e) {
    return {ok: false, status: 0, body: {refused: "the server did not answer"}};
  }
}
// A read that needs a file or folder this computer does not have says so; nothing is downloaded or started for it.
const refused = r => r.body.missing ? `${badge("not on this computer", "amber")} ${esc(r.body.refused)}${r.body.fix ? `<p class="note"><b>How to fix:</b> ${esc(r.body.fix)}</p>` : ""}`
  : `${badge("refused", "red")} ${esc(r.body.refused || `the server answered ${r.status}`)}`;
const stillHere = (g, key) => g === LS.gen && location.hash.replace("#/", "") === key;
const later = (g, key, fn, ms) => setTimeout(() => stillHere(g, key) && fn(), ms);
const notYet = what => `<p class="note">${badge("not yet available", "grey")} ${esc(what)}</p>`;
function page(title, sub, body, right = "") {
  view.innerHTML = `<div class="page">${head(title, esc(sub), right)}${body}</div>`;  // sub is plain text here
}
const taskRow = t => `<tr><td class="mono">${esc(t.task)}</td><td>${badge(t.state, {done: "green", failed: "red", running: "blue"}[t.state] || "grey")}</td>
  <td>${esc(t.started)}</td><td>${esc(t.ended || "")}</td><td class="note">${esc(t.problem || "")}</td></tr>`;
// The list of jobs at the top of Jobs and Evals: a search over company, 10-K date, start time, template and models,
// a few jobs to a page, newest first, the shown job marked. Only the jobs on the page are drawn (and, on Evals, read).
const PICK = {q: "", page: 0};
const pickMatch = (x, q) => !q || `${jobLabel(x).title} ${jobLabel(x).sub}`.toLowerCase().includes(q.trim().toLowerCase());
const pickerCard = (title, sub) => `<div class="card" style="margin-bottom:16px"><div class="hd row-flex"><div><h2>${title}</h2><p class="sub">${sub}</p></div>
  <span class="spacer"></span><input id="pick-q" type="search" placeholder="Search company, date or model" value="${esc(PICK.q)}" style="width:280px"></div><div class="bd" id="pick-rows"></div></div>`;
// o: {jobs, shown, size, head (extra column headers), cell(job) (extra cells), onShow(id), drawn(visible jobs)}
function drawPicker(o) {
  const box = document.getElementById("pick-rows");
  if (!box) return;
  const hits = o.jobs.filter(x => pickMatch(x, PICK.q)), pages = Math.max(1, Math.ceil(hits.length / o.size));
  PICK.page = Math.max(0, Math.min(PICK.page, pages - 1));
  const vis = hits.slice(PICK.page * o.size, (PICK.page + 1) * o.size);
  box.innerHTML = (vis.length ? `<table><tr><th>Job</th><th>Outcome</th>${o.head || ""}<th></th></tr>${vis.map(x => { const l = jobLabel(x);
    return `<tr><td><b>${esc(l.title)}</b><div class="note">${esc(l.sub)}</div></td><td>${x.final ? outcomeBadge(x.final.end) : x.running ? badge("running", "blue") : badge("no final record", "grey")}</td>
    ${o.cell ? o.cell(x) : ""}<td class="num">${x.job_id === o.shown ? badge("shown below", "blue") : `<button class="btn sm" data-show="${esc(x.job_id)}">Show</button>`}</td></tr>`; }).join("")}</table>`
    : `<p class="note">No job matches “${esc(PICK.q)}”.</p>`)
    + `<div class="row-flex" style="margin-top:10px"><span class="note">${PICK.q ? `${hits.length} of ${o.jobs.length} jobs match` : `${o.jobs.length} jobs`}${pages > 1 ? ` · page ${PICK.page + 1} of ${pages}` : ""}</span><span class="spacer"></span>
    ${pages > 1 ? `<button class="btn sm" data-pg="-1" ${PICK.page === 0 ? "disabled" : ""}>‹ Newer</button> <button class="btn sm" data-pg="1" ${PICK.page >= pages - 1 ? "disabled" : ""}>Older ›</button>` : ""}</div>`;
  box.querySelectorAll("[data-pg]").forEach(b => b.onclick = () => { PICK.page += Number(b.dataset.pg); drawPicker(o); });
  box.querySelectorAll("[data-show]").forEach(b => b.onclick = () => o.onShow(b.dataset.show));
  const q = document.getElementById("pick-q");
  if (q && !q.dataset.on) { q.dataset.on = "1"; q.oninput = () => { PICK.q = q.value; PICK.page = 0; drawPicker(o); }; }
  if (o.drawn) o.drawn(vis);
}
// A page that redraws itself (Jobs, every 2 s while a job runs) keeps the search box's focus and caret.
function keepSearchFocus(draw) {
  const a = document.activeElement, on = a && a.id === "pick-q", s = on ? [a.selectionStart, a.selectionEnd] : null;
  draw();
  const q = on && document.getElementById("pick-q");
  if (q) { q.focus(); try { q.setSelectionRange(s[0], s[1]); } catch (e) { /* type=search may refuse */ } }
}
// A job step's plain name, by the kind its folder records.
const STEP_NAMES = {corrections: "Corrections to the statement", spread: "Spread: placing the lines", repair: "Tree of thought: repairing failed checks", grade: "Comparison with the answer key"};
const STAGE_NAMES = {job: "Job", corrections: "Corrections", spread: "Spread", tree: "Tree of thought", repair: "Tree of thought", grade: "Comparison with the answer key"};
// The job's steps on one line: each from its first to its last recorded event, on the span of all the job's events.
const TL_COLOR = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"];
function timeline(events) {
  const t = e => Date.parse(e.time), ev = events.filter(e => e.time && !isNaN(t(e)));
  if (ev.length < 2) return "";
  const t0 = Math.min(...ev.map(t)), span = Math.max(...ev.map(t)) - t0;
  if (span <= 0) return "";
  const steps = new Map();
  ev.filter(e => e.stage && e.stage !== "job").forEach(e => { const k = STAGE_NAMES[e.stage] || e.stage, s = steps.get(k);
    steps.set(k, s ? {a: Math.min(s.a, t(e)), z: Math.max(s.z, t(e))} : {a: t(e), z: t(e)}); });
  if (!steps.size) return "";
  const mins = ms => ms < 60000 ? `${Math.round(ms / 1000)} s` : `${(ms / 60000).toFixed(1)} min`;
  const rows = [...steps.entries()];
  return `<div style="margin-top:14px"><div class="note" style="margin-bottom:4px">Steps over time: ${esc(mins(span))} from the first record to the last</div>
    ${rows.map(([k, s], i) => `<div class="tl-row"><div class="tl-l">${esc(k)}</div><div class="tl-track"><div class="tl-seg" style="left:${(100 * (s.a - t0) / span).toFixed(2)}%;width:${(100 * (s.z - s.a) / span).toFixed(2)}%;background:${TL_COLOR[i % 4]}" title="${esc(k)}: ${esc(mins(s.z - s.a))}"></div></div><div class="tl-v note">${esc(mins(s.z - s.a))}</div></div>`).join("")}</div>`;
}
const taskTable = ts => ts.length ? `<table><tr><th>Task</th><th>State</th><th>Started</th><th>Ended</th><th>Problem</th></tr>${ts.map(taskRow).join("")}</table>`
  : `<span class="note">none yet</span>`;

// ---------- 1. Setup: what this computer has; Load, Update and Reset of the offered items, one item per click ----------
const SETUP_ACTS = {
  load: ["Load", "Download to this computer. Clicking Load on a row starts that download."],
  update: ["Update", "Download an installed model again. If its weights changed, they are not the ones this setup was tested with; the result below shows the tested, previous and new versions."],
  reset: ["Reset", "Remove an item from this computer. A removed model is gone for every project on this computer that uses it."]};
// What a Load card says, naming only the sources of the items offered (no Ollama when no local model is offered)
const actSaid = (a, offer) => a !== "load" ? SETUP_ACTS[a][1] : SETUP_ACTS.load[1].replace("computer.", `computer: ${[
  offer.some(o => o.action === "load" && o.item.startsWith("model:")) ? "models from Ollama's library" : "",
  offer.some(o => o.action === "load" && o.item.startsWith("fasb:")) ? "FASB taxonomy editions (US GAAP) from xbrl.fasb.org" : ""].filter(Boolean).join(", ")}.`);
// A Load, Update or Reset button names what it fetches or removes: "Load model qwen3.5:4b-q4_K_M", "Load US GAAP 2025".
const actLabel = (a, item) => { const [kind, name] = String(item).split(/:(.*)/s);
  return `${SETUP_ACTS[a][0]} ${kind === "fasb" ? `US GAAP ${name}` : kind === "model" ? `model ${name}` : item}`; };
// An Update's result: the version the setup was tested with, before and after.
const updateSaid = x => x && x.said ? `${x.said} · tested ${x.tested || "–"}${x.tested_in ? ` (in job ${x.tested_in})` : ""} · before ${x.before || "–"} · after ${x.after || "–"}` : "";
// The package advisories: asked of OSV once per session when Setup first opens (the one request Setup sends to OSV;
// names and versions only), again on Check now; then the time it was asked and what came back, a failure or a partial
// answer included, and each locked package's own result. Never "no advisories" unasked.
const advisoriesSaid = a => !a || a.state === "not checked" ? `${badge("not checked", "grey")} <span class="note">${esc((a && a.detail) || "not checked yet")}</span>`
  : `${badge(a.ready ? "no known advisories" : "problem", a.ready ? "green" : "red")} <span class="note">asked ${esc(a.checked)} · ${esc(a.ready ? a.seen : a.problem)}</span>
    ${(a.packages || []).length ? `<details style="margin-top:8px"><summary class="note">The ${a.packages.length} locked packages</summary>
    <table class="wrap"><tr><th>Package</th><th>Version</th><th>Known advisories</th></tr>${a.packages.map(x => `<tr><td>${esc(x.name)}</td><td class="mono">${esc(x.version)}</td>
    <td>${x.advisories.length ? x.advisories.map(i => `<a href="https://osv.dev/vulnerability/${encodeURIComponent(i)}" target="_blank" rel="noopener">${esc(i)}</a>`).join(", ") : `<span class="note">none</span>`}</td></tr>`).join("")}</table></details>` : ""}`;
// The server's identity as it recorded it at start (data/ui/server_identity/), so a person can name the server from a
// page
const serverSaid = i => i ? `<div class="card" style="margin-top:16px"><div class="hd"><h2>This server</h2><p class="sub">As the server recorded itself at start, in data/ui/server_identity/.</p></div><div class="bd kv"><div class="k">Port</div><div>${esc(i.port)}</div><div class="k">Started</div><div>${esc(i.time)} (this computer's time)</div><div class="k">Process</div><div>${esc(i.pid)}</div><div class="k">Code</div><div class="mono">${i.commit ? `commit ${esc(i.commit)}` : esc(i.git || "no commit recorded")}</div>${i.files_digest ? `<div class="k">Files digest</div><div class="mono">${esc(i.files_digest)}</div>` : ""}</div></div>` : "";
async function liveSetup() {
  const g = ++LS.gen;
  if (!stillHere(g, "setup")) return;
  page("Setup", "What this computer has, read now.", `<p class="note">Checking… Each item is read now (the packages against their locks, the EDGAR server, the key); this takes up to about half a minute.</p>`);
  const q = LS.setupLineup ? `?lineup=${encodeURIComponent(LS.setupLineup)}` : "";
  const [r, t] = await Promise.all([get(`/api/setup${q}`), get("/api/tasks")]);
  if (!stillHere(g, "setup")) return;
  if (!r.ok) {   // refused (no settings, a line-up no longer offered): Check again asks for the Run page's default
    const was = LS.setupLineup;
    LS.setupLineup = null;
    page("Setup", "What this computer has, read now.", `<div class="card pad">${refused(r)}${was ? `<p class="note">Check again checks the Run page's default models instead of ${esc(was)}.</p>` : ""}</div>`,
      `<button class="btn" id="setup-again">Check again</button>`);
    document.getElementById("setup-again").onclick = liveSetup;
    return;
  }
  const got = r.body, tasks = (t.body.tasks || []).filter(x => /^(load|update|reset):/.test(x.task));
  // a present key is not tested: counted apart from the items that are ready
  // an item expected later (the FASB taxonomy, fetched with the first filing) is not counted as a gap
  const needed = got.items.filter(i => i.needed !== false && !i.later), later = got.items.filter(i => i.needed !== false && i.later), offer = got.load.offer;
  const ready = needed.filter(i => i.ready && !i.untested).length, present = needed.filter(i => i.ready && i.untested).length;
  const next = needed.find(i => !i.ready);
  const after = later.length ? ` ${later.map(i => `Step ${Number(i.step)}, ${esc(i.item)}, comes with your first filing.`).join(" ")}` : "";
  // the Load buttons a missing item's own row shows: each names the one model or edition it downloads
  const loadsFor = name => offer.filter(o => o.action === "load" && o.item.startsWith({"Local models": "model:", "FASB taxonomy": "fasb:"}[name] || "-"));
  const acts = Object.keys(SETUP_ACTS).filter(a => offer.some(o => o.action === a));
  const ask = acts.includes(LS.ask) ? LS.ask : null;
  page("Setup", "What this computer has, read now.", `
  <div class="card" style="margin-bottom:16px"><div class="hd"><h2>Models to check</h2><p class="sub">Readiness below is for these models. The Run page starts with the first one checked; choosing here does not change it.</p></div>
    <div class="bd row-flex">${Object.entries(got.lineups || {}).map(([k, label]) => `<button class="btn sm${k === got.lineup ? " primary" : ""}" data-lu="${esc(k)}">${esc(label)}</button>`).join(" ")}</div></div>
  ${ask ? `<div class="card" style="margin-bottom:16px"><div class="hd"><h2>${SETUP_ACTS[ask][0]}</h2><p class="sub">${esc(actSaid(ask, offer))}</p></div>
    <div class="bd"><table><tr><th>Item</th><th>What happens</th><th></th></tr>${offer.filter(o => o.action === ask).map(o => `<tr><td class="mono">${esc(o.item)}</td><td>${esc(o.what)}</td>
      <td class="num"><button class="btn sm primary" data-act="${esc(o.action)}" data-item="${esc(o.item)}">${esc(actLabel(ask, o.item))}</button></td></tr>`).join("")}</table>
    </div></div>` : ""}
  <div class="card"><div class="hd"><h2>This computer, step by step</h2><p class="sub">${ready} of ${needed.length} items these models need are ready${present ? `, ${present} present but not tested` : ""}${next ? `; next: step ${Number(next.step)}, ${esc(next.item)}.` : `; ready to choose a filing on Companies and years.`}${after}</p></div>
    <div class="bd"><table><tr><th>Step</th><th>Item</th><th>What it is</th><th>Status</th><th>Seen, or what is missing and how to fix it</th></tr>
    ${got.items.map(i => `<tr><td class="num">${esc(i.step)}</td><td><b>${esc(i.item)}</b></td><td>${esc(i.what)}</td>
      <td>${i.needed === false ? badge("not needed", "grey") : i.ready && i.untested ? badge("present, not tested", "amber") : badge(i.ready ? "ready" : "not ready", i.ready ? "green" : "red")}</td>
      <td>${esc(i.ready || i.needed === false ? i.seen : i.problem)}${!i.ready && i.fix ? `<div class="note"><b>How to fix:</b> ${esc(i.fix)}</div>` : ""}
      ${i.item === "OpenRouter key" && i.needed !== false && i.state !== "missing" && (i.ready || i.state) ? `<div class="note"><button class="btn sm" id="key-test">${i.state ? "Test again" : "Test the key"}</button> asks OpenRouter whether it accepts the key; no model is called and it costs nothing.</div>` : ""}
      ${i.item === "OpenRouter key" && i.needed !== false ? `<div class="note"><input type="password" id="key-in" autocomplete="off" spellcheck="false" placeholder="paste the key (sk-or-...)" style="max-width:320px"> <button class="btn sm" id="key-save">${i.state === "missing" || !i.ready && !i.state ? "Save the key" : "Replace the key"}</button> writes it to the file .env in the project folder on this computer, then tests it. It is not shown again.<div id="key-said"></div></div>` : ""}
      ${i.item === "SEC User-Agent" ? `<div class="note"><input type="text" id="ua-in" autocomplete="off" spellcheck="false" placeholder="Jane Doe jane.doe@example.com" style="max-width:320px"> <button class="btn sm" id="ua-save">${i.ready ? "Replace" : "Save"}</button> writes it to the file .env in the project folder on this computer. It is not shown again.<div id="ua-said"></div></div>` : ""}
      ${!i.ready && i.needed !== false ? loadsFor(i.item).map(o => `<div class="note">${esc(o.what)} <button class="btn sm" data-act="load" data-item="${esc(o.item)}">${esc(actLabel("load", o.item))}</button></div>`).join("") : ""}</td></tr>`).join("")}</table>
    <div id="load-said"></div>
    ${got.load.models_unreadable ? `<p class="note">${badge("models not read", "amber")} ${esc(got.load.models_unreadable)}</p>` : ""}
    ${got.load.unavailable.map(u => `<p class="note">${badge("not offered", "grey")} ${esc(u)}</p>`).join("")}
    <p class="note">${esc(got.load.by_a_person)}.</p>
    <h3 style="margin-top:12px">Downloads and removals since the server started</h3><p class="note">Each Load, Update or Reset you press on this page is listed here with its result. The list starts empty each time the server starts.</p>${taskTable(tasks.map(x => ({...x, problem: x.problem || updateSaid(x.result)})))}</div></div>
  <div class="card" style="margin-top:16px"><div class="hd"><h2>Package advisories</h2><p class="sub">${esc((got.advisories && got.advisories.what) || "")}</p></div>
    <div class="bd"><div class="row-flex" style="align-items:flex-start"><div id="adv-said" style="flex:1">${advisoriesSaid(got.advisories)}</div><button class="btn sm" id="adv-check">Check again</button></div></div></div>
  ${modelsTable(got.models)}${serverSaid(SRV.identity)}`, `<button class="btn" id="setup-again">Check again</button>
    ${acts.map(a => `<button class="btn${a === ask ? " primary" : ""}" data-ask="${a}">${SETUP_ACTS[a][0]}</button>`).join(" ")}`);
  document.getElementById("setup-again").onclick = liveSetup;
  const askOsv = async () => {
    document.getElementById("adv-said").innerHTML = `<span class="note">Asking OSV…</span>`;
    const a = await post("/api/setup/advisories", {});
    if (!stillHere(g, "setup")) return;
    document.getElementById("adv-said").innerHTML = a.ok ? advisoriesSaid(a.body.advisories) : refused(a);
  };
  document.getElementById("adv-check").onclick = askOsv;
  const keyBtn = document.getElementById("key-test");
  const testKey = async () => {
    if (keyBtn) { keyBtn.disabled = true; keyBtn.textContent = "Asking OpenRouter…"; }
    await post("/api/setup/key", {});
    if (stillHere(g, "setup")) liveSetup();
  };
  if (keyBtn) keyBtn.onclick = testKey;
  const keySave = document.getElementById("key-save");
  if (keySave) keySave.onclick = async () => {
    const box = document.getElementById("key-in");
    const key = box.value;
    box.value = "";
    if (!key.trim()) { document.getElementById("key-said").innerHTML = `<span class="note">Paste the key first.</span>`; return; }
    const label = keySave.textContent;
    keySave.disabled = true; keySave.textContent = "Saving and testing…";
    const r = await post("/api/setup/key/save", {key});
    if (!stillHere(g, "setup")) return;
    if (!r.ok) { keySave.disabled = false; keySave.textContent = label; document.getElementById("key-said").innerHTML = refused(r); return; }
    LS.keyAsked = true;
    liveSetup();
  };
  const uaSave = document.getElementById("ua-save");
  if (uaSave) uaSave.onclick = async () => {
    const box = document.getElementById("ua-in");
    const ua = box.value;
    if (!ua.trim()) { document.getElementById("ua-said").innerHTML = `<span class="note">Type your name and email first.</span>`; return; }
    const label = uaSave.textContent;
    uaSave.disabled = true; uaSave.textContent = "Saving…";
    const r = await post("/api/setup/ua/save", {ua});
    if (!stillHere(g, "setup")) return;
    if (!r.ok) { uaSave.disabled = false; uaSave.textContent = label; document.getElementById("ua-said").innerHTML = refused(r); return; }
    box.value = "";
    liveSetup();
  };
  const keyRow = got.items.find(i => i.item === "OpenRouter key");
  if (keyBtn && keyRow && keyRow.untested && !keyRow.state && !LS.keyAsked) { LS.keyAsked = true; testKey(); }
  if ((!got.advisories || got.advisories.state === "not checked") && !LS.osvAsked) { LS.osvAsked = true; askOsv(); }
  view.querySelectorAll("[data-ask]").forEach(b => b.onclick = () => { LS.ask = LS.ask === b.dataset.ask ? null : b.dataset.ask; liveSetup(); });
  view.querySelectorAll("[data-lu]").forEach(b => b.onclick = () => { LS.setupLineup = b.dataset.lu; LS.ask = null; liveSetup(); });
  view.querySelectorAll("[data-act]").forEach(b => b.onclick = async () => {
    const r2 = await post(`/api/setup/${b.dataset.act}`, {item: b.dataset.item});
    if (!stillHere(g, "setup")) return;
    if (!r2.ok) { document.getElementById("load-said").innerHTML = `<p>${refused(r2)}</p>`; return; }
    LS.ask = null;
    liveSetup();
  });
  if (tasks.some(x => x.state === "running")) later(g, "setup", liveSetup, 2000);
}

// ---------- 2. Template: the standard template, read only ----------
// This version has one template, a fixed standard changed only in code, shown read only on this page.
// "Use default" registers it in the results database, which a job needs.
function tplStructure() {
  return ["BS", "IS", "CF"].map(st => `<h3 style="margin-top:14px">${esc(STMT[st])}</h3>${TPL.sections.filter(s => s.statement === st).map(s => {
    const rows = TPL.rows.filter(r => r.section === s.id);
    const kind = s.kind === "spread" ? "printed lines" : s.kind === "totals" ? "totals only" : "memo";
    return `<div class="card pad" style="margin-top:8px"><div class="row-flex"><b>${esc(s.name)}</b>
      <span class="note">${esc(kind)}${s.normal_balance ? ` · normally ${esc(s.normal_balance)}s` : ""}</span></div>
      ${s.headings.length ? `<p class="note">Reads the lines printed under: ${s.headings.map(h => esc(h.label)).join("; ")}</p>` : ""}
      <ul style="margin:6px 0">${rows.map(r => `<li>${esc(r.name)} ${r.formula ? `<span class="note">= ${esc(r.formula)}</span>` : r.catch_all ? `<span class="note">(catch-all)</span>` : ""}</li>`).join("") || "<li class='note'>no rows</li>"}</ul>
      ${s.check ? `<p>${TPL.outcome_checks ? (TPL.outcome_checks.sections.includes(s.id) ? badge("statement check", "blue") : `<span class="note">shown only, decides nothing:</span>`) : badge("check", "blue")} ${esc(s.check.compare)} must equal the printed ${esc(s.check.filed_label || "total, reconstructed as " + s.check.reconstructed)}</p>` : `<p class="note">No check.</p>`}</div>`;
  }).join("") || `<p class="note">No sections.</p>`}`).join("")
    + ((TPL.report_checks || []).length ? `<h3 style="margin-top:14px">Checks on the finished spread</h3><p class="note">${TPL.outcome_checks ? "Statement checks decide a statement's outcome: a difference flags it as blocked, while every line stays placed and the workbook is still made. A check that cannot be made (the company prints no such total) fails nothing. The others are shown only." : "Reported with the spread; a difference never stops it."}</p><ul>${TPL.report_checks.map(x => `<li>${TPL.outcome_checks && TPL.outcome_checks.report_checks.includes(x.id) ? badge("statement check", "blue") + " " : ""}${esc(x.id)} ${esc(x.name)}: ${esc(x.left)} against ${esc(x.right)} <span class="note">(${x.period === "instant" ? "at the end of the period" : "over the period"}${Object.values(x.alternatives || {}).flat().length ? "; if not printed, reads " + Object.values(x.alternatives).flat().map(a => esc(a.label)).join(", ") : ""})</span></li>`).join("")}</ul>` : "");
}
async function liveTemplate() {
  const g = ++LS.gen;
  const r = await get("/api/template");
  if (!stillHere(g, "template")) return;
  if (!r.ok) return page("Template", "", `<div class="card pad">${refused(r)}</div>`);
  TPL = r.body; rowsById = Object.fromEntries(TPL.rows.map(x => [x.id, x]));
  const side = document.getElementById("tpl-current");      // the sidebar names the same template
  if (side) side.textContent = r.body.label || "";
  const d = r.body.default;
  page("Template", "The template is the list of rows every spread fills, the sections they sit in, the subtotals code computes from them, and the checks that compare those subtotals with totals the company prints. Every job spreads with the standard template below.", `
  <div class="card"><div class="hd"><h2>Jobs spread with</h2></div><div class="bd">
    <div class="kv"><div class="k">Template</div><div>${esc(r.body.label)} ${d ? badge("registered", "green") : badge("not registered", "red")}</div>
      <div class="k">Rows</div><div>${TPL.rows.length}</div>
      <div class="k">Settings</div><div class="mono">${esc(r.body.settings || "–")}</div></div>
    ${d ? `<details style="margin-top:8px"><summary class="note">Record details</summary><div class="kv"><div class="k">Version ID</div><div class="mono">${esc(d.version_id)}</div>
      <div class="k">File</div><div class="mono">${esc(d.path)}</div><div class="k">SHA-256</div><div class="mono">${esc(d.sha256)}</div></div></details>`
      : `<p>Not registered yet. A job needs the template recorded in the results database once.</p><button class="btn primary" id="use-default">Use default</button><div id="tpl-said"></div>`}</div></div>
  <div class="card" style="margin-top:16px"><div class="bd">${notYet("Changing the template on this page. In this version the template is a fixed standard, changed only in code; a later version may let you edit it here and save each change as a new version.")}</div></div>
  <h2 style="margin-top:20px">Sections and checks</h2>${tplStructure()}
  <h2 style="margin-top:20px">Rows of the standard template</h2>${tplBrowser()}`);
  wireBrowser();
  const b = document.getElementById("use-default");
  if (b) b.onclick = async () => {
    b.disabled = true; b.textContent = "Registering…";
    document.getElementById("tpl-said").innerHTML = `<p class="note">Checking the standard template and recording it in the results database; this takes a few seconds.</p>`;
    const r2 = await post("/api/template/default", {});
    if (!stillHere(g, "template")) return;
    if (!r2.ok) { b.disabled = false; b.textContent = "Use default"; document.getElementById("tpl-said").innerHTML = `<p>${refused(r2)}</p>`; return; }
    liveTemplate();
  };
}

// ---------- 3. Companies and years: search, the 10-Ks, fetch one ----------
async function liveCompanies() {
  const g = ++LS.gen;
  const t = await get("/api/tasks");
  if (!stillHere(g, "companies")) return;
  const fetches = (t.body.tasks || []).filter(x => x.task.startsWith("fetch:"));
  const s = LS.search, ch = LS.chosen;
  const task = ch && fetches.find(x => x.task === `fetch:${Number(ch.cik)}/${ch.accession}`);
  page("Companies and years", "Search the SEC's company list by ticker or name, then choose one 10-K.", `
  <div class="card"><div class="bd"><div class="row-flex"><input type="text" id="q" placeholder="ticker or name" value="${esc(s ? s.query : "")}" style="max-width:320px">
    <button class="btn primary" id="go">Search</button><span id="q-said"></span></div>
    ${s ? `<p class="note">${esc(s.filter)} Removed from this search: ${Number(s.removed)}.${s.capped ? " Only the first hits are shown; search more narrowly." : ""}</p>` : `<p class="note">Banks, insurers and real estate (SIC 6000-6799) are removed before the list.</p>`}
    <p class="note">Any listed year can be chosen. Choose and fetch downloads the 10-K from SEC EDGAR, and with it the US GAAP taxonomy edition the 10-K uses (2022 to 2026), from xbrl.fasb.org, when this computer does not have them yet.</p></div></div>
  ${s ? s.companies.map(c => `<div class="card" style="margin-top:12px"><div class="hd"><h2>${esc(c.name)} <span class="note">${esc(c.ticker)} · CIK ${esc(c.cik)}</span></h2>
      <p class="sub">SIC ${esc(c.sic)} ${esc(c.sic_description)}</p></div><div class="bd">
      ${c.ten_ks.length ? `<table><tr><th title="Any listed 10-K can be chosen; the newest is not required">10-K for the period ending</th><th>Filed</th><th>Accession</th><th></th></tr>${c.ten_ks.map(k => `<tr><td>${esc(k.report_date)}</td><td>${esc(k.filed)}</td>
        <td class="mono">${esc(k.accession)}</td><td class="num">${ch && ch.cik === c.cik && ch.accession === k.accession ? badge("chosen", "blue")
          : `<button class="btn sm" data-cik="${esc(c.cik)}" data-acc="${esc(k.accession)}">Choose and fetch</button>`}</td></tr>`).join("")}</table>`
        : `<span class="note">No 10-K with inline XBRL.</span>`}</div></div>`).join("") || `<p class="note">No company found.</p>` : ""}
  ${ch ? `<div class="card" style="margin-top:16px"><div class="hd"><h2>Currently chosen filing</h2><p class="sub">Kept from your last choice, not part of the search above; the Run page uses it until you choose another.</p></div><div class="bd">
    <div class="kv"><div class="k">Company</div><div>${esc(ch.name)}</div><div class="k">10-K</div><div>${esc(ch.report_date)} · <span class="mono">${esc(ch.accession)}</span></div>
    <div class="k">Preparation</div><div>${task ? badge(task.state, {done: "green", failed: "red"}[task.state] || "blue") : badge("not started", "grey")}
      ${task && task.state === "running" ? ` <span class="note">downloading the 10-K from SEC EDGAR, then its US GAAP edition if missing; this can take a minute</span>` : ""}
      ${task && task.result ? ` period ${esc(task.result.period_end)}` : ""}
      ${task && task.problem ? `<div class="note">${esc(task.problem)}</div>` : ""}</div>
    ${task && task.result ? `<div class="k">US GAAP taxonomy</div><div>${esc(task.result.taxonomy_year)}: ${task.result.edition_on_disk
      ? (task.result.edition_load ? "downloaded now from xbrl.fasb.org; Setup's taxonomy row now reads ready" : "already on this computer")
      : badge("not on this computer", "red")}</div>` : ""}</div>
    ${task && task.state === "done" ? `<div style="margin-top:10px"><a class="btn primary" href="#/run">Next: run</a></div>` : ""}</div></div>` : ""}`);
  const go = async () => {
    const q = document.getElementById("q").value;
    document.getElementById("q-said").innerHTML = `<span class="note">Searching…</span>`;
    const r = await post("/api/companies/search", {q});
    if (!stillHere(g, "companies")) return;
    if (!r.ok) { document.getElementById("q-said").innerHTML = refused(r); return; }
    LS.search = r.body;
    liveCompanies();
  };
  document.getElementById("go").onclick = go;
  document.getElementById("q").onkeydown = e => { if (e.key === "Enter") go(); };
  view.querySelectorAll("[data-acc]").forEach(b => b.onclick = async () => {
    const c = s.companies.find(x => x.cik === b.dataset.cik), k = c.ten_ks.find(x => x.accession === b.dataset.acc);
    b.disabled = true; b.textContent = "Starting…";
    const r = await post("/api/filings/fetch", {cik: c.cik, accession: k.accession});
    if (!stillHere(g, "companies")) return;
    if (!r.ok) { b.disabled = false; b.textContent = "Choose and fetch"; document.getElementById("q-said").innerHTML = refused(r); return; }
    LS.chosen = {cik: c.cik, name: c.name, accession: k.accession, report_date: k.report_date};
    liveCompanies();
  });
  if (task && task.state === "running") later(g, "companies", liveCompanies, 2000);
}

// ---------- 4. Run: the line-up for the chosen filing, its two parts, then start ----------
// The parts a person can switch off, both ticked when the page first opens and as the person left them after that,
// and the line an unticked box shows.
const SWITCHES = [["tiebreaker", "Tiebreaker (a third model answers a disputed line blind)", "Off: lines still disputed after the analyst's second look keep the analyst's latest placement and stay marked disputed."],
  ["tree", "Tree of thought (attempts repairs to sections whose code checks fail)", "Off: sections whose checks fail are left as the spread made them."]];
async function liveRun() {
  const g = ++LS.gen;
  const r = await get("/api/template");
  if (!stillHere(g, "run")) return;
  if (!r.ok) return page("Run", "", `<div class="card pad">${refused(r)}</div>`);
  const ch = LS.chosen, lineups = Object.entries(r.body.lineups), d = r.body.default;
  // every reason Start cannot run yet, the key included for models that need it
  const why = lu => [!d && "a registered template (Template page)", !ch && "a chosen 10-K (Companies and years page)",
    lu && r.body.lineups[lu] && r.body.lineups[lu].key_missing && "an OpenRouter key for these models (Setup page)"].filter(Boolean);
  const whySaid = w => w.length ? `<p class="note">${badge("not ready", "amber")} Start needs ${w.length > 1 ? `${w.slice(0, -1).join(", ")} and ${w[w.length - 1]}` : w[0]}.</p>` : "";
  const first = lineups.length ? lineups[0][0] : null;
  page("Run", "One company and one 10-K per job. The server refuses a start until the filing is prepared and the template registered.", `
  <div class="card"><div class="bd"><div class="kv">
    <div class="k">Template</div><div>${d ? esc(r.body.label) : `${esc(r.body.label || "")} ${badge("not registered", "red")} <a href="#/template">choose “Use default”</a>`}</div>
    <div class="k">Settings</div><div class="mono">${esc(r.body.settings || "–")}</div>
    <div class="k">Filing</div><div>${ch ? `${esc(ch.name)}, 10-K ${esc(ch.report_date)} <span class="mono">${esc(ch.accession)}</span>` : `<a href="#/companies">choose a 10-K first</a>`}</div></div>
    <label class="f">Models</label><div>${lineups.map(([k, v], i) => `<label style="display:block"><input type="radio" name="lu" value="${esc(k)}" ${i ? "" : "checked"}>
      ${esc(v.label || k)} ${v.pays ? badge(`paid, at most $${Number(v.dollars_most).toFixed(2)}`, "amber") : badge("nothing paid", "grey")}</label>`).join("")}</div>
    <label class="f">Parts</label><div>${SWITCHES.filter(([k]) => k !== "tree" || r.body.tree_on !== false).map(([k, label, off]) => `<label style="display:block"><input type="checkbox" data-part="${k}" ${LS.parts[k] ? "checked" : ""}> ${esc(label)}
      <span class="note" id="off-${k}" ${LS.parts[k] ? "hidden" : ""}>${esc(off)}</span></label>`).join("")}${r.body.tree_on === false ? `<p class="note">Tree of thought: off in these settings (statement checks; it has almost nothing it can repair).</p>` : ""}</div>
    <div id="expect" class="callout" style="margin-top:12px"><span class="note">Reading what earlier jobs with these models used\u2026</span></div>
    <p class="note">The job stops at its caps and keeps its full record. STOP is on the Jobs page.</p>
    <div id="not-ready">${whySaid(why(first))}</div>
    <div style="margin-top:10px"><button class="btn primary" id="start" ${why(first).length ? "disabled" : ""}>Start</button> <span id="run-said"></span></div></div></div>`);
  const ready = () => {         // again when another line-up is chosen
    const w = why((view.querySelector("input[name=lu]:checked") || {}).value);
    document.getElementById("not-ready").innerHTML = whySaid(w);
    document.getElementById("start").disabled = w.length > 0;
  };
  const part = k => view.querySelector(`[data-part=${k}]`).checked;
  // What to expect: a historical estimate for this model configuration (scripts/run_estimate.py), the
  // same for everyone who has this copy; not this computer's jobs.
  const expect = async () => {
    const lu = (view.querySelector("input[name=lu]:checked") || {}).value, box = document.getElementById("expect");
    const r = await get(`/api/estimate?lineup=${encodeURIComponent(lu || "")}`);
    if (!stillHere(g, "run") || !box) return;
    const e = r.ok ? r.body : null;
    if (!e || !e.jobs) { box.innerHTML = `<span class="note">No historical estimate exists for these models yet. The limit beside the models still applies.</span>`; return; }
    const usd = v => `$${v.toFixed(2)}`, min = v => `${Math.round(v / 60)} minutes`;
    box.innerHTML = `<b>Historical estimate, not a promise:</b> ${e.jobs} finished statements-only jobs with these models (${esc(e.first)} to ${esc(e.last)}${e.reconstructed ? `; ${e.reconstructed === e.jobs ? "all" : Number(e.reconstructed)} started from the command line, not from these pages` : ""}) cost from ${usd(e.dollars.min)} to ${usd(e.dollars.max)}, about ${usd(e.dollars.mean)} on average, and took from ${Math.round(e.seconds.min / 60)} to ${min(e.seconds.max)}.
      <span class="note">Your 10-K may cost and take more, for example when it has more lines or the models are slower that day. The limit beside the models still applies: the job stops when it is reached.</span>`;
  };
  expect();
  view.querySelectorAll("input[name=lu]").forEach(b => b.onchange = () => { ready(); expect(); });
  view.querySelectorAll("[data-part]").forEach(b => b.onchange = () => { LS.parts[b.dataset.part] = b.checked; document.getElementById(`off-${b.dataset.part}`).hidden = b.checked; });
  if (LS.runSaid) { document.getElementById("run-said").innerHTML = LS.runSaid; LS.runSaid = null; }
  document.getElementById("start").onclick = async () => {
    const btn = document.getElementById("start"), said = x => { const el = document.getElementById("run-said"); if (el && stillHere(g, "run")) el.innerHTML = x; else LS.runSaid = x; };
    const lineup = view.querySelector("input[name=lu]:checked").value;
    const lu = r.body.lineups[lineup];   // a paid line-up runs at its stated ceiling, shown beside it
    btn.disabled = true; btn.textContent = "Starting…";
    said(`<span class="note">Sent. The server is checking the filing and the template; the Jobs page opens when the job starts.</span>`);
    let res;
    try {
      res = await post("/api/jobs/start", {cik: ch.cik, accession: ch.accession, lineup, dollars: lu.pays ? Number(lu.dollars_most) : 0, tiebreaker: part("tiebreaker"), tree: view.querySelector("[data-part=tree]") ? part("tree") : false});
    } catch (e) { res = {ok: false, status: 0, body: {refused: `the server did not answer (${e.message || e}); nothing was started by this page`}}; }
    btn.disabled = false; btn.textContent = "Start";
    // a refusal is kept if the person has left the page, and shown when the Run page opens again
    if (!res.ok) { said(refused(res)); return; }
    chooseJob(res.body.job.job_id || null);
    if (stillHere(g, "run")) location.hash = "#/jobs";
  };
}

// A chat request counts only when it was sent; records of waits and recoveries, and requests never sent, are shown apart.
const uncounted = c => [[c.not_sent, "not sent"], [c.bookkeeping, "waits and recoveries"], [c.unreadable, "unreadable"]]
  .filter(([n]) => n).map(([n, w]) => `<div class="note">${Number(n)} ${w}, not counted</div>`).join("");
// What the run committed for a line when it was done: each row with its amount (pieces and a vote's replacement
// included); an event written before the run recorded this shows its row only.
// "not run" belongs to a line the stopped run never handled: not to one with a recorded row (placed, or the older
// row field) or its own class; older events carried the run's stop on every line written after it
const lineNotRun = e => Boolean(e.not_run) && !((e.placed || []).length || e.row) && (!e.class || e.class === "not run");
// Replay's reason for a line with no row: its own class first, then the run's stop
const noRowSaid = l => l.class && l.class !== "not run" ? l.class : (l.not_run || l.class || "not placed");
const lineWent = e => e.placed ? (e.placed.length ? e.placed.map(x => `“${esc(rowName(x.row))}” ${money(x.amount)}`).join(" and ") : "nothing committed")
  : e.row ? `“${esc(rowName(e.row))}”` : "no row";

// ---------- 5. Jobs: the running or chosen job, from progress.jsonl and the children's own records ----------
async function liveJobs() {
  const g = ++LS.gen;
  const [st, list] = await Promise.all([get("/api/state"), get("/api/jobs")]);
  if (!stillHere(g, "jobs")) return;
  if (!list.ok) return page("Jobs", "", `<div class="card pad">${refused(list)}</div>`);
  const run = st.body.job, last = st.body.last, jobs = list.body.jobs;
  const id = LS.job || (run && run.job_id) || (jobs[0] && jobs[0].job_id);
  if (id !== LS.shown) Object.assign(LS, {shown: id, events: [], from: 0, tree: 0});
  let ev = null, pr = null;
  if (id) {
    [ev, pr] = await Promise.all([get(`/api/job?id=${encodeURIComponent(id)}&from=${LS.from}`), get(`/api/job/progress?id=${encodeURIComponent(id)}`)]);
    if (!stillHere(g, "jobs")) return;
    if (ev.ok) { LS.events = LS.events.concat(ev.body.events); LS.from = ev.body.next; }
  }
  const running = run && (!id || run.job_id === id), p = pr && pr.ok ? pr.body : null, j = ev && ev.ok ? ev.body : null;
  const recent = p ? p.children.flatMap(c => c.recent || []).sort((a, b) => String(a.time).localeCompare(String(b.time))).slice(-12) : [];
  keepSearchFocus(() => page("Jobs", "A job is one run of the pipeline on one company's 10-K: it fixes missing tags, then spreads the income statement, cash flow statement and balance sheet into the template. Choose a job in the list; the page shows the running job, else the one chosen, else the newest.", `
  ${jobs.length ? pickerCard("Jobs", "Newest first. Search by company, 10-K date, start time or model.") : ""}
  ${run ? `<div class="card pad" style="margin-bottom:14px"><div class="row-flex"><b>${run.job_id ? esc(jobLabel(jobs.find(x => x.job_id === run.job_id) || {job_id: run.job_id}).title) : "Starting…"}</b>${badge(run.state, "blue")}
    ${run.stop_requested ? badge("stop requested", "amber") : ""}<span class="spacer"></span><button class="btn" id="stop" ${run.stop_requested ? "disabled" : ""}>Stop</button></div>
    <p class="note">Stop writes STOP: before the job's handshake into its request folder, after it into the job's folder; the job ends at its next check and keeps its records.</p></div>` : ""}
  ${!run && last ? `<div class="card pad" style="margin-bottom:14px">Last job: <b>${esc(last.job_id ? jobLabel(jobs.find(x => x.job_id === last.job_id) || {job_id: last.job_id}).title : last.request_id)}</b> ${badge(last.event || "ended", "grey")}
    ${last.final ? `ended “${esc(last.final.end)}”${last.final.why ? `: ${esc(last.final.why)}` : ""}` : esc((last.worker && last.worker.why) || "")}</div>` : ""}
  ${id ? `<div class="card"><div class="hd row-flex"><div><h2>${esc(jobLabel({job_id: id, manifest: j && j.manifest}).title)}</h2><p class="sub">${j && j.manifest ? `${esc(jobLabel({job_id: id, manifest: j.manifest}).sub)} · ${esc(partsSaid(j.manifest.parts))}${j.manifest.review ? ` · second look ${esc(j.manifest.review.second_look)}` : ""}` : "no manifest (started outside this app)"}</p></div>
      <span class="spacer"></span>${j && j.final ? outcomeBadge(j.final.end) : running ? badge("running", "blue") : badge("no final record", "grey")}</div>
    <div class="bd">${p ? `<div class="row-flex"><b>${p.percent === null ? "–" : `${Number(p.percent)}%`}</b><span class="note">${Number(p.processed)} of ${Number(p.planned)} lines processed in the statements started (processed means handled, not placed correctly)${p.not_started.length ? `; not started: ${p.not_started.map(x => esc(STMT[x] || x)).join(", ")}` : ""}</span></div>
      ${bar(p.processed, p.planned)}${j && j.final ? finalSaid(j.final) : ""}${timeline(LS.events)}
      <table style="margin-top:14px"><tr><th>Step</th><th style="width:30%">Lines</th><th class="num">Chat requests</th><th class="num">Embeddings</th><th class="num">Tokens</th><th class="num">Dollars</th></tr>
      ${p.children.map(c => `<tr><td title="${esc(c.child)}"><b>${esc(STEP_NAMES[c.kind] || c.kind)}</b>${c.statement ? `<div class="note">${esc(STMT[c.statement] || c.statement)}</div>` : ""}</td>
        <td>${c.planned ? `${bar(c.processed, c.planned)}<span class="note">${Number(c.processed)} / ${Number(c.planned)}</span>` : `<span class="note">–</span>`}</td>
        <td class="num">${Number(c.calls.calls) ? `<button class="btn sm" data-calls="${esc(c.child)}/${esc(c.run)}" data-step="${esc(STEP_NAMES[c.kind] || c.kind)}${c.statement ? ` · ${esc(STMT[c.statement] || c.statement)}` : ""}" title="List this step's model calls; each opens with what it was sent and what it answered">${Number(c.calls.calls)} · open</button>` : 0}${c.calls.errors ? ` ${badge(`${Number(c.calls.errors)} errors`, "red")}` : ""}${uncounted(c.calls)}</td>
        <td class="num">${Number(c.calls.embeddings || 0)}</td><td class="num">${Number(c.calls.tokens).toLocaleString("en-US")}</td>
        <td class="num">${Number(c.calls.cost_usd || 0).toFixed(4)}</td></tr>`).join("")}</table>` : `<span class="note">${pr ? refused(pr) : ""}</span>`}</div></div>
  <div class="grid g2" style="margin-top:16px">
    <div class="card"><div class="hd"><h2>Latest lines</h2><p class="sub">Where each processed line went, as recorded. A recorded row is not thereby correct.</p></div><div class="bd">
      ${recent.map(e => `<div style="padding:4px 0;border-bottom:1px solid var(--border)"><div class="row-flex">${esc(e.label)}<span class="spacer"></span>${badge(e.class || e.review || "processed", "grey")}</div>
        <div class="note">${lineWent(e)} · ${esc(sectionName(e.section))}${e.from ? ` · from ${esc(sectionName(e.from))}` : ""}${e.vote ? ` · vote: ${esc(e.vote)}` : ""}${lineNotRun(e) ? " · not run" : ""}</div></div>`).join("") || `<span class="note">none yet</span>`}</div></div>
    <div class="card"><div class="hd"><h2>Job steps</h2><p class="sub">From the job's progress record.</p></div><div class="bd">
      ${LS.events.slice(-15).map(e => `<div class="note">${esc(String(e.time || "").slice(11, 19))} · <b>${esc(STAGE_NAMES[e.stage] || e.stage)}</b> ${esc(e.status)}${e.label && !/^\w+ (IS|BS|CF)$/.test(e.label) && e.label !== e.stage ? ` · ${esc(e.label)}` : /(IS|BS|CF)$/.test(e.label || "") ? ` · ${esc(STMT[e.label.slice(-2)])}` : ""}${e.result ? ` · ${esc(e.result)}` : ""}${e.why ? ` · ${esc(e.why)}` : ""}</div>`).join("") || `<span class="note">none yet</span>`}</div></div></div>
  <div class="row-flex" style="margin-top:14px">${j && j.final ? `<a class="btn primary" href="#/results">Review the result</a>` : ""}<a class="btn" href="#/replay">Replay</a><a class="btn" href="#/tiebreak">Tiebreaker</a><a class="btn" href="#/tree">Tree of thought</a><a class="btn" href="#/evals">Evals</a></div>` : `<p class="note">No job yet. Start one on the Run page.</p>`}
`));
  drawPicker({jobs, shown: id, size: 5, onShow: x => { chooseJob(x); liveJobs(); }});
  view.querySelectorAll("[data-calls]").forEach(b => b.onclick = () => callList(id, b.dataset.calls, b.dataset.step));
  const stop = document.getElementById("stop");
  if (stop) stop.onclick = async () => {
    const r = await post("/api/jobs/stop", run.job_id ? {job_id: run.job_id} : {request_id: run.request_id});
    if (!stillHere(g, "jobs")) return;
    if (!r.ok) stop.insertAdjacentHTML("afterend", ` ${refused(r)}`); else liveJobs();
  };
  if (run || (id && !(j && j.final))) later(g, "jobs", liveJobs, 2000);
}

// A finished job's outcome per statement, and why no repair ran on a blocked section.
function finalSaid(f) {
  const sts = Object.entries(f.statements || {});
  if (!sts.length) return "";
  const skips = st => (f.skipped || []).filter(x => x.statement === st);
  return `<div class="callout" style="margin:10px 0">${sts.map(([st, o]) => `<div>${esc(STMT[st] || st)}: ${outcomeBadge(o)}${skips(st).map(x => `<div class="note">repair not attempted on ${esc(x.section === "run" ? "the statement as a whole" : sectionName(x.section))}: ${esc(x.why)}</div>`).join("")}</div>`).join("")}${sts.some(([, o]) => o === "blocked") ? `<div class="note">A blocked statement failed a code check. Its failed checks and their reasons are on Results, in the job's line and in the workbook's Checks sheet.</div>` : ""}</div>`;
}

// ---------- 5c, 5d. Tiebreaker and tree: the shown job's own records ----------
// The job a page shows, in the Jobs page's order: the one chosen (on Jobs, Run or Start here), else the running job,
// else the newest.
async function shownJob() {
  if (LS.job) return LS.job;
  const [st, list] = await Promise.all([get("/api/state"), get("/api/jobs")]);
  const run = st.body.job, jobs = list.body.jobs || [];
  return (run && run.job_id) || (jobs[0] && jobs[0].job_id) || null;
}
async function jobViews(key, title) {
  const g = ++LS.gen, id = await shownJob();
  if (!stillHere(g, key)) return null;
  if (!id) { page(title, "", `<p class="note">No job yet. Start one on the Run page.</p>`); return null; }
  const [r, j] = await Promise.all([get(`/api/job/views?id=${encodeURIComponent(id)}`), get(`/api/job?id=${encodeURIComponent(id)}&from=999999999`)]);
  if (!stillHere(g, key)) return null;
  if (!r.ok) { page(title, "", `<div class="card pad">${refused(r)}</div>`); return null; }
  const m = j.ok && j.body.manifest;
  return {g, id, v: r.body, running: j.ok && !j.body.final, company: (m && m.company) || "", parts: (m && m.parts) || null,
          label: jobLabel({job_id: id, manifest: m || null})};
}
// A part switched off for the shown job, said where its page would draw it.
const switchedOff = (got, k) => got.parts && got.parts[k] === "off"
  ? `<p class="note">${badge("switched off for this job", "grey")} ${esc(SWITCHES.find(s => s[0] === k)[2])}</p>` : "";
// With the tiebreaker off: the disputed lines, each kept with the analyst's placement.
const disputedTable = T => `<div class="card"><div class="hd"><h2>Lines not resolved: ${T.length}</h2></div><div class="bd">
  <table><tr><th>Line</th><th>Amount</th><th>Section</th><th>Analyst (kept)</th><th>Checker</th></tr>${T.map(t => `<tr><td>${esc(t.label)}</td>
  <td>${money(t.amount)}</td><td>${esc(sectionName(t.section))}</td><td>${placedText(t.analyst)}${t.revised ? ` <span class="note">(revised after seeing the checker's answer; first: ${placedText(t.analyst_first)})</span>` : ""}</td><td>${placedText(t.checker)}</td></tr>`).join("")}</table></div></div>`;
async function liveTiebreak() {
  const got = await jobViews("tiebreak", "Tiebreaker");
  if (!got) return;
  if (!got.v.tiebreaks) {
    page("Tiebreaker", "", jobShown(got.label) + (switchedOff(got, "tiebreaker") || `<p class="note">${got.running ? "No tiebreaker line recorded yet." : !got.parts ? "This job recorded no tiebreaker line. Its record is older than the on/off switches, so it does not say whether the tiebreaker was on." : "The tiebreaker was on and got no line: the analyst and the checker agreed on every line after the second look."}</p>`));
  } else if (switchedOff(got, "tiebreaker")) {
    page("Tiebreaker", "", jobShown(got.label) + switchedOff(got, "tiebreaker") + disputedTable(got.v.tiebreaks));
  } else {
    V = {job: got.id, label: got.label, tiebreaks: got.v.tiebreaks, tree: null};
    pageTiebreak();
    view.querySelector(".page").insertAdjacentHTML("afterbegin", switchedOff(got, "tiebreaker") + lineTarget(got.v.tiebreaks));
  }
  if (got.running) later(got.g, "tiebreak", liveTiebreak, 5000);
}
// The line a Results source opened this page at: its card scrolled to, or said absent from this job's tiebreaker lines.
function lineTarget(T) {
  const want = LS.line;
  LS.line = null;
  if (!want) return "";
  const i = T.findIndex(t => t.concept === want);
  if (i < 0) return `<p class="note">${badge("not here", "grey")} The line ${esc(want)} has no tiebreaker record in this job.</p>`;
  setTimeout(() => { const c = document.getElementById(`tb-${i}`); if (c) c.scrollIntoView(); }, 0);
  return `<p class="note">${badge("opened from Results", "blue")} The line \u201c${esc(T[i].label)}\u201d is card ${i + 1} of ${T.length}.</p>`;
}
async function liveTree() {
  const got = await jobViews("tree", "Tree of thought");
  if (!got) return;
  const trees = got.v.trees;
  if (!trees) {
    page("Tree of thought", "", jobShown(got.label) + (switchedOff(got, "tree") || `<p class="note">${got.running ? "No tree repair recorded yet." : !got.parts ? "This job recorded no tree of thought. Its record is older than the on/off switches, so it does not say whether the tree was on." : "The tree of thought was on and recorded no repair. It runs only on a section whose checks fail; see the checks on Results."}</p>`));
    if (got.running) later(got.g, "tree", liveTree, 3000);
  } else {
    // a Results source opens the tree of the repair that changed it, by the repair's folder name
    const named = LS.treeFolder ? trees.findIndex(t => String(t.folder).split(/[\\/]/).pop() === LS.treeFolder) : -1;
    const missed = LS.treeFolder && named < 0 ? `<p class="note">${badge("not here", "grey")} The repair ${esc(LS.treeFolder)} has no tree in this job's records.</p>` : "";
    if (named >= 0) LS.tree = named;
    LS.treeFolder = null;
    const i = Math.min(LS.tree, trees.length - 1), T = trees[i];
    const tabs = trees.length > 1 ? `<div class="tabs" style="margin-bottom:12px">${trees.map((t, k) => `<button data-tree="${k}" class="${k === i ? "on" : ""}">${esc(sectionName(t.section))}${t.growing ? " (growing)" : ""}</button>`).join("")}</div>` : "";
    const tchild = String(T.folder).split(/[\\/]/).slice(-2).join("/");
    const calls = `<div style="margin-bottom:12px"><button class="btn sm" id="tree-calls" title="Every model call this repair recorded, each with what it was sent and what it answered">Model calls of this repair</button></div>`;
    if (T.growing) growingTree(got.label, T, missed + tabs + calls);
    else {
      V = {job: got.id, label: got.label, tiebreaks: null, tree: {...T, company_name: got.company}};
      pageTree();
      view.querySelector(".page").insertAdjacentHTML("afterbegin", missed + tabs + calls);
    }
    const tc = document.getElementById("tree-calls");
    if (tc) tc.onclick = () => callList(got.id, tchild, `Tree of thought · ${sectionName(T.section)}`);
    view.querySelectorAll("[data-tree]").forEach(b => b.onclick = () => { LS.tree = +b.dataset.tree; liveTree(); });
    if (got.running && T.growing) later(got.g, "tree", liveTree, 3000);
  }
}
function growingTree(L, T, tabs) {
  const kids = p => T.nodes.filter(n => n.parent === p);
  const draw = n => `<li><b>${esc(nodeName(n.id))}</b> <span class="note">level ${esc(n.depth)}${n.asked ? ` · asked ${Number(n.asked)} questions` : ""}</span>
    ${n.proposed.length ? `<ul class="clean">${n.proposed.map(x => `<li class="note">proposed (provisional): ${esc(x.action)} “${esc(x.line)}”${x.row ? ` to ${Array.isArray(x.row) ? x.row.map(r => `“${esc(rowName(r))}”`).join(" and ") : `“${esc(rowName(x.row))}”`}` : ""}${x.reason ? ` · ${esc(human(x.reason))}` : ""}</li>`).join("")}</ul>` : ""}
    ${kids(n.id).length ? `<ul>${kids(n.id).map(draw).join("")}</ul>` : ""}</li>`;
  page("Tree of thought", `${STMT[T.statement] || T.statement}, ${sectionName(T.section)}. Growing: drawn from the repair's model calls as they are recorded.`, `${jobShown(L)}${tabs}
  <p style="margin:-10px 0 8px">${badge("provisional", "amber")} <span class="note">${esc(T.provisional || "proposals only")}</span></p>
  <p class="note" style="margin:0 0 14px">Node IDs are the ones code sent to the proposer. Which proposals code kept, and the finished tree, appear when the repair ends.</p>
  <div class="grid g2"><div class="card"><div class="hd"><h2>Branches so far</h2></div><div class="bd">${T.nodes.length ? `<ul class="clean">${T.nodes.filter(n => !n.parent || !T.nodes.some(x => x.id === n.parent)).map(draw).join("")}</ul>` : `<span class="note">no branch yet</span>`}</div></div>
    <div class="card"><div class="hd"><h2>Second opinions and choices</h2></div><div class="bd">
      ${T.critic.map(c => `<div class="note">Second opinion on “${esc(c.line)}”: ${esc(c.verdict)}${c.reason ? ` · ${esc(human(c.reason))}` : ""}</div>`).join("") || `<span class="note">no second opinion yet</span>`}
      ${T.picks.map(p => `<div class="note">The chooser, level ${esc(p.level)}: ${esc(human(p.reason || ""))}</div>`).join("")}
      <p class="note">Checks after the tree: ${Number(T.rechecked ?? T.recheck_calls ?? 0)} model calls so far.</p></div></div></div>`);
}

// ---------- 6. Results: import a finished job, choose the current spread, inspect it, export, download, review ----------
const IMPORT_TONE = {imported: "green", partial: "amber", "not imported": "grey", "cannot import": "red"};
// A finished job's import state as the server compares it (its recorded finals against the store); Import stays offered
// while any final version is not stored.
function importState(x, st) {
  st = st || {state: "not imported", missing: []};
  const again = st.state === "partial" || st.state === "not imported";
  return `${badge(st.state, IMPORT_TONE[st.state] || "grey")}${st.state === "partial" && st.missing.length ? ` <span class="note">not stored: ${st.missing.map(esc).join(", ")}</span>` : ""}
    ${st.why ? ` <span class="note">${esc(st.why)}</span>` : ""}${again ? ` <button class="btn sm" data-imp="${esc(x.job_id)}">${st.state === "partial" ? "Import the rest" : "Import"}</button>` : ""}`;
}
// A job's refused statements: the refusal, amounts unknown, nothing to import or choose.
function refusedSaid(st) {
  return ((st && st.refused) || []).map(r => `<div class="note" style="padding:0 0 4px">${esc(STMT[r.statement] || r.statement)} ${badge("refused", "red")} ${esc(r.said)}; amounts unknown</div>`).join("");
}
// What the last import answered, folder by folder, refusals included.
function importSaid(s) {
  if (!s) return "";
  return `<div class="card pad" style="margin-top:12px"><b>Import of ${esc(s.job)}</b> ${s.rejected ? badge(`${Number(s.rejected)} not imported`, "red") : badge("nothing refused", "green")}
    ${(s.results || []).map(x => `<div class="note">${esc(x.folder)} · ${badge(x.result, x.result === "rejected" || x.result === "failed" ? "red" : "green")}${x.why ? ` ${esc(x.why)}` : ""}</div>`).join("")}
    <div class="note">Load record: <span class="mono">${esc(s.record)}</span></div></div>`;
}
// The trace of a stored source to the job's records (server views.trace): the filing, the fact as the recorded line
// names it, where code found the quote, and the record links the records support. What is absent is said.
const periodSaid = p => !Array.isArray(p) ? "" : p[0] === "instant" ? `at ${p[1]}` : p[0] === "duration" ? `${p[1]} to ${p[2]}` : p.join(" ");
const recordLink = (href, text) => href ? `<a href="${esc(href)}" target="_blank" rel="noopener">${esc(text)}</a>` : "";
function sourceTrace(t) {
  if (!t) return `<br><span class="note">${badge("unavailable", "grey")} no trace for this source</span>`;
  const f = t.fact, q = t.quote_source;
  const links = [recordLink(t.line, "recorded line"), recordLink(t.answer, "answer"), recordLink(t.checks, "section checks"),
    t.vote ? `${recordLink(t.vote.href, "vote")} <button class="btn sm" data-tb="${esc(t.vote.concept)}">Tiebreaker page</button>` : "",
    t.tree ? `${recordLink(t.tree.href, t.tree.node ? `tree node ${t.tree.node}` : "tree")} <button class="btn sm" data-tr="${esc(t.tree.folder)}">Tree page</button>` : ""].filter(Boolean);
  return `<br><span class="mono">Filing ${esc(t.accession || "–")}</span>${f ? ` · fact ${esc(f.concept)} · ${esc(periodSaid(f.period))} · ${esc(f.unit)} ${esc(f.amount)}` : ""}${t.other_facts ? ` · and ${Number(t.other_facts)} more fact${t.other_facts === 1 ? "" : "s"}` : ""}
    ${q ? `<br>Quote found: ${esc([q.tier, q.kind, q.block && `block ${q.block}`].filter(Boolean).join(" · "))}${q.label ? ` <span class="note">(${esc(q.label)})</span>` : ""}` : `<br>Quote found: ${badge("not recorded", "grey")}`}
    ${links.length ? `<br>${links.join(" · ")}` : ""}${t.unavailable.map(u => `<br>${badge("unavailable", "grey")} ${esc(u)}`).join("")}`;
}
// A viewed spread's open questions, in plain words: each amount the spread took out of a larger printed line into its
// own row, the line it came from, and the lines the filing's calculations do not rule it out of (so it might be
// counted twice).
const cut = (s, n = 70) => { s = String(s || ""); return s.length > n ? `${s.slice(0, n - 1)}…` : s; };
// A tag in words: the line's own label where the spread has it, else the tag split ("MinorityInterest" -> "minority interest").
const TAG_WORDS = {MinorityInterest: "noncontrolling interest"};
const tagWords = t => { const k = String(t || "").replace(/^[a-z-]+:/, ""); return TAG_WORDS[k] || k.replace(/([a-z])([A-Z])/g, "$1 $2").toLowerCase(); };
const capFirst = s => String(s).charAt(0).toUpperCase() + String(s).slice(1);
const parsed = s => { try { return typeof s === "string" ? JSON.parse(s) : (s || {}); } catch (e) { return {}; } };
// Each check's name in plain words; an unknown one keeps its name with tags and row codes in words.
function checkSaid(c) {
  const n = String(c.name || c.kind || "");
  if (c.kind === "control") return "The spread's grand total matches the filed total";
  const known = [[/^one currency in the section$/, "All amounts are in one currency"],
    [/ equals its printed parts$/, "The filed section total equals the lines printed above it"],
    [/^every line allocated$/, "No line is left unresolved: each is placed in a row or put outside the template"],
    [/^section total found$/, "A section total was found"],
    [/^rows add to .*\(reconstructed\)$/, "The rows add up to the section total (rebuilt from the printed lines: the filing prints no subtotal)"],
    [/^rows add to /, "The rows add up to the filed section total"],
    [/^reconstructed total independent/, "The rebuilt total does not reuse the rows' own amounts"],
    [/^no amount claimed in two sections$/, "No amount is counted in two sections"],
    [/ control ties$/, "The grand total matches the filing"]];
  const k = known.find(([re]) => re.test(n));
  return k ? k[1] : capFirst(human(n.replace(/\b[a-z-]+:[A-Z]\w+/g, tagWords)));
}
// Why a failed check failed, from its own record; "" when the record does not say.
function whyFailed(c, labelOf) {
  const r = parsed(c.detail), list = xs => xs.map(labelOf).join(", ");
  if ((r.lines_outside_the_total || []).length) return `${Number(r.gap) === 0 ? "The rows match the filed total, but " : ""}${list(r.lines_outside_the_total)} ${r.lines_outside_the_total.length === 1 ? "is" : "are"} placed in this section and ${r.lines_outside_the_total.length === 1 ? "is" : "are"} not part of that total.`;
  if ((r.lines || []).length) return `Not placed in any row: ${list(r.lines)}.`;
  if ((r.claimed_twice || []).length) return `Counted in two sections: ${list(r.claimed_twice)}.`;
  if (r.gap !== undefined && Number(r.gap) !== 0) return `Off by ${money(r.gap)}, more than the ${money(r.tolerance)} allowed for rounding.`;
  return "";
}
const checkState = st => /^not done/.test(st) ? "not done yet" : st === "ok" ? "passed" : st === "tied" ? "ties" : st;
// The exceptions in words: the step by its role, the printed line by its label.
const STEP_ROLE = {C2: "Organizer", C4: "Analyst", C7: "Tree of thought", C8: "Tiebreaker"};
const exceptionSaid = s => human(String(s || "")
  .replace(/a tie cannot see this line \(zero\)/, "the totals cannot confirm where this line goes: its amount is zero")
  .replace(/a tie cannot see this line \(within tolerance\)/, "the totals cannot confirm where this line goes: its amount is within the rounding allowance")
  .replace(/\bHaiku\b/g, "the tiebreaker").replace(/C4's placement/g, "the analyst's placement"));
// What the spread's claim record holds (live.Session.use). Not a gate: the download is the workflow's result,
// unreviewed; open questions are listed here and marked PROVISIONAL in the workbook.
function useCard(u, failed = 0) {
  if (!u) return "";
  const old = u.analyst ? `<p class="note">An edited version made by an earlier version of this app; it is exported like any other version.</p>` : "";
  const fails = failed ? ` ${failed} check${failed === 1 ? "" : "s"} failed: see why at the top and under Checks.` : "";
  if (!u.usable) return `<p>${badge("cannot be exported", "red")} ${esc(u.reason || "")}</p>${old}`;
  if (u.status !== "open claims" || !(u.pieces || []).length) return `<p>${badge(u.status === "open claims" ? "open questions" : "no open questions", u.status === "open claims" ? "amber" : "green")}
    This version can be chosen in step 2 and exported. The workbook is the workflow's result, unreviewed.${fails}</p>${old}`;
  const n = u.pieces.length;
  return `<p>${badge("open questions", "amber")} This version can be chosen and exported; the questions stay open and the amounts they touch are marked PROVISIONAL in the workbook.${fails}</p>
    <p>The spread took ${n} amount${n === 1 ? "" : "s"} out of a larger printed line and gave ${n === 1 ? "it its" : "each its"} own row.
    The filing's own calculations do not show that ${n === 1 ? "this amount is" : "these amounts are"} kept apart from some other lines on the statement,
    so code cannot rule out that ${n === 1 ? "it is" : "one is"} counted twice. Each amount paired with one such line is one open question (${esc(u.said || "")}).</p>
    <table class="wrap" style="margin-top:6px"><tr><th class="num">Amount</th><th>Given its own row</th><th>Taken out of the printed line</th><th>Not ruled out of</th></tr>
    ${u.pieces.map(p => `<tr><td class="num">${money(p.amount)}</td><td>${esc(p.row_name || p.row)}</td><td>${esc(p.line_label || p.line)}</td>
      <td><details><summary>${(p.rival_labels || []).length} lines</summary><ul class="clean">${(p.rival_labels || []).map(x => `<li class="note" title="${esc(x)}">${esc(cut(x))}</li>`).join("")}</ul></details></td></tr>`).join("")}</table>${old}`;
}
// One section of the viewed spread, with a heading and a line above it.
const part = (title, sub, body, open = true) => `<details class="part" ${open ? "open" : ""}><summary><h3>${title}</h3>${sub ? `<span class="note">${sub}</span>` : ""}</summary><div>${body}</div></details>`;
function spreadDetail(r) {
  if (!r) return "";
  if (!r.ok) return `<div class="card pad" style="margin-top:16px">${refused(r)}</div>`;
  const d = r.body, tr = d.trace || {sources: [], unavailable: [], pages: []};
  const sources = d.sources.map((x, i) => ({...x, t: tr.sources[i]})), src = code => sources.filter(x => x.row_code === code);
  const tone = st => /fail|differ/.test(st) ? "red" : /tied|pass|ok/.test(st) ? "green" : "grey";
  const filled = d.rows.filter(x => x.availability !== "no placement");
  const labelOf = t => (d.sources.find(s => s.line === t) || {}).line_label || tagWords(t);
  const failed = d.checks.filter(c => c.state === "failed"), checks = [...failed, ...d.checks.filter(c => c.state !== "failed")];
  return `<div class="card step" id="spread-detail"><div class="hd row-flex"><div><span class="stepno">3</span><h2 style="display:inline">Viewing: ${esc(d.company || `CIK ${d.cik}`)} · ${esc(STMT[d.statement] || d.statement)} · ${esc(d.fiscal_year)}</h2>
      <p class="sub">${esc(jobStarted(d.job_id || "") ? `from the job started ${jobStarted(d.job_id)}` : "")}${d.parts ? ` · ${esc(partsSaid(d.parts))}` : ""}</p></div><span class="spacer"></span>${outcomeBadge(d.outcome)} <button class="btn sm" id="spread-close">Close</button></div>
    <div class="bd">
    ${failed.length ? `<div class="callout" style="margin-bottom:12px"><b>${d.outcome === "blocked" ? "Blocked because" : "Failed checks"}:</b> ${failed.map(c => `${esc(capFirst(sectionName(c.section)))}: ${esc(whyFailed(c, labelOf) || checkSaid(c))}`).join(" ")}</div>` : ""}
    ${part("Open questions", "", useCard(d.use, failed.length))}
    ${part("Rows", `${filled.length} of ${d.rows.length} template rows have an amount`, `<table class="wrap"><tr><th>Row</th><th>Section</th><th class="num">Amount</th><th>Printed lines placed here</th></tr>
    ${(LS.allRows ? d.rows : filled).map(x => { const s = src(x.row_code), none = x.availability === "no placement"; return `<tr${none ? ' class="note"' : ""}><td><b>${esc(x.row_name)}</b></td><td class="note">${esc(sectionName(x.section))}</td><td class="num">${none ? "–" : money(x.placed)}</td>
      <td>${s.length ? `<details><summary>${s.length} line${s.length === 1 ? "" : "s"}</summary>${s.map(y => `<div class="note" style="padding:4px 0;border-top:1px solid var(--border)"><b>${esc(y.line_label || y.line)}</b>: ${money(y.signed)}
        · placed by ${esc(placerName(y.placed_by))}${y.review_state ? ` · ${esc(y.review_state)}` : ""}${y.quote ? `<br>“${esc(y.quote)}”` : ""}
        <details><summary>Record details</summary><div class="wrapall">${sourceTrace(y.t)}</div></details></div>`).join("")}</details>` : `<span class="note">none</span>`}</td></tr>`; }).join("")}</table>
    ${filled.length < d.rows.length ? `<button class="btn sm" id="all-rows" style="margin-top:8px">${LS.allRows ? "Hide the rows with no amount" : `Show the ${d.rows.length - filled.length} rows with no amount`}</button>` : ""}`)}
    ${part("Checks", `${failed.length ? `${failed.length} failed, listed first; ` : "none failed; "}${d.checks.length - failed.length} passed or not applicable`, `<table class="wrap"><tr><th>Check</th><th>Result</th><th class="num">Filed</th><th class="num">Computed</th><th class="num">Difference</th></tr>
    ${checks.map(c => `<tr><td>${esc(checkSaid(c))}${c.section && sectionName(c.section) !== c.section ? ` <span class="note">· ${esc(sectionName(c.section))}</span>` : ""}${c.state === "failed" && whyFailed(c, labelOf) ? `<div><b>${esc(whyFailed(c, labelOf))}</b></div>` : ""}${c.detail ? `<details><summary class="note">Record details</summary><div class="note mono wrapall">${esc(c.detail)}</div></details>` : ""}</td><td>${badge(checkState(c.state), tone(c.state))}</td>
      <td class="num">${money(c.filed)}</td><td class="num">${money(c.computed)}</td><td class="num">${money(c.difference)}</td></tr>`).join("") || `<tr><td colspan="5" class="note">no check recorded</td></tr>`}</table>`, failed.length > 0)}
    ${part("Agreement with the answer key", "where an answer key for this 10-K is set up", gradeCard(d.grade, [d.statement]), false)}
    ${d.exceptions.length ? part("Notes from the steps", `${d.exceptions.length} recorded: lines a step could not confirm or settled another way`, `<table class="wrap"><tr><th>Step</th><th>Printed line</th><th>What happened</th></tr>${d.exceptions.map(e => `<tr><td>${esc(STEP_ROLE[e.step] || e.step)}</td>
      <td>${esc(e.concept ? labelOf(e.concept) : "–")}${e.section ? `<div class="note">${esc(sectionName(e.section))}</div>` : ""}</td><td>${esc(exceptionSaid(e.why))}${/vote/.test(e.class || "") ? ` <span class="note">(settled by the tiebreaker's vote)</span>` : ""}</td></tr>`).join("")}</table>`, false) : ""}
    ${part("Where it came from", "the filing and the job's records", `
      <div class="row-flex">${tr.pages.map(k => `<button class="btn sm" data-goto="${k}">${{jobs: "Jobs", tiebreak: "Tiebreaker", tree: "Tree of thought"}[k]} page for this job</button>`).join(" ")}</div>
      <div class="note wrapall" style="margin-top:6px">Filing ${esc(d.accession || "–")} · ${[recordLink(tr.run, "run record"), recordLink(tr.cumulative, "all placements")].filter(Boolean).join(" · ")}</div>
      ${tr.unavailable.map(u => `<div class="note">${badge("unavailable", "grey")} ${esc(u)}</div>`).join("")}`, false)}
    </div></div>`;
}
// Who placed a line, in words.
const placerName = b => ({C4: "the analyst", C7: "the tree of thought", vote: "the tiebreaker's vote", "second look": "the analyst's second look", code: "code"}[b] || b || "–");
async function liveResults() {
  const g = ++LS.gen, h1 = view.querySelector(".page-head h1");
  if (!h1 || h1.textContent !== "Results") page("Results", "Each finished job's Excel workbook, the workflow's result, unreviewed: you review it, and the job's replay shows how each line was classified. Below it, the results database keeps every imported version of a spread, in three steps.", `<p class="note">Reading the results database\u2026</p>`);
  const [r, list, one, made] = await Promise.all([get("/api/results"), get("/api/jobs"),
    LS.spread ? get(`/api/results/spread?id=${encodeURIComponent(LS.spread)}`) : null, get("/api/result/list")]);
  if (!stillHere(g, "results")) return;
  if (!r.ok) return page("Results", "", `<div class="card pad">${refused(r)}</div>`);
  const {spreads} = r.body, imports = r.body.imports || {};
  const finished = (list.body.jobs || []).filter(x => x.final);
  const waiting = finished.filter(x => !imports[x.job_id] || imports[x.job_id].state !== "imported"), done = finished.length - waiting.length;
  const current = spreads.filter(s => s.current);
  const groups = [...new Map(current.map(s => [`${s.template_version_id}|${s.cik}|${s.fiscal_year}`, s])).values()];
  const q = (LS.resQ || "").trim().toLowerCase(), shownSpreads = spreads.filter(s => !q || `${s.company} ${s.fiscal_year} ${STMT[s.statement] || s.statement}`.toLowerCase().includes(q));
  const cellOf = (s0, st) => { const s = current.find(x => x.template_version_id === s0.template_version_id && String(x.cik) === String(s0.cik) && x.fiscal_year === s0.fiscal_year && x.statement === st);
    return s ? `<button class="btn sm" data-open="${esc(s.run_id)}" title="View this version in step 3">${outcomeBadge(s.outcome)}</button>` : `<span class="note">none counts</span>`; };
  page("Results", "Each finished job's Excel workbook, the workflow's result, unreviewed: you review it, and the job's replay shows how each line was classified. Below it, the results database keeps every imported version of a spread, in three steps.", `
  ${workflowResults(finished, (made && made.ok && made.body.results) || [])}
  ${groups.length ? `<div class="card" style="margin-bottom:16px"><div class="hd"><h2>At a glance</h2><p class="sub">The version that counts for each company, year and statement. Press one to view it in step 3.</p></div><div class="bd">
    <table><tr><th>Company</th><th>Year</th><th>Income statement</th><th>Balance sheet</th><th>Cash flow statement</th></tr>
    ${groups.map(s => `<tr><td>${esc(s.company || `CIK ${s.cik}`)}<div class="note">template ${esc(s.template)}</div></td><td>${esc(s.fiscal_year)}</td>${["IS", "BS", "CF"].map(st => `<td>${cellOf(s, st)}</td>`).join("")}</tr>`).join("")}</table></div></div>` : ""}
  <div class="card step"><div class="hd"><span class="stepno">1</span><h2 style="display:inline">Import a finished job</h2><p class="sub">A finished job keeps its spreads in its own folder. Import copies them into the results database, where the next steps choose and view them.</p></div><div class="bd">
    ${waiting.length ? `<table class="wrap"><tr><th>Job</th><th>Job outcome</th><th>In the database</th></tr>${(LS.allWaiting ? waiting : waiting.slice(0, 5)).map(x => `<tr><td><b>${esc(jobLabel(x).title)}</b><div class="note">${esc(jobLabel(x).sub)}</div>${refusedSaid(imports[x.job_id])}</td>
      <td>${outcomeBadge(x.final.end)}</td><td>${importState(x, imports[x.job_id])}</td></tr>`).join("")}</table>` : `<span class="note">Every finished job is imported.</span>`}
    ${waiting.length > 5 ? `<button class="btn sm" id="all-waiting" style="margin-top:8px">${LS.allWaiting ? "Show five" : `Show all ${waiting.length} not yet imported`}</button>` : ""}
    ${done ? `<p class="note" style="margin-top:6px">${done} finished job${done === 1 ? " is" : "s are"} already imported.</p>` : ""}
    ${importSaid(LS.importSaid)}</div></div>
  <div class="card step"><div class="hd row-flex"><div><span class="stepno">2</span><h2 style="display:inline">Choose the version that counts</h2><p class="sub">The database keeps every imported version of a statement. For each company, year and statement, one counts: it is the one shown under At a glance. Choosing it approves nothing.</p></div>
    <span class="spacer"></span><input id="res-q" type="search" placeholder="Search company or year" value="${esc(LS.resQ || "")}" style="width:220px"></div><div class="bd">
    <label class="f">Your name</label><input type="text" id="by" placeholder="your name" value="${esc(LS.by)}" style="max-width:240px">
    <p class="note">Needed before Make this count: the results database records who made each choice, beside the time. It stays on this computer.</p>
    ${r.body.store === false ? `<p class="note">${badge("not on this computer", "amber")} No results database yet: import a finished job in step 1.</p>` : ""}
    <table style="margin-top:8px"><tr><th>Company</th><th>Year</th><th>Statement</th><th title="How the job's code checks ended for this statement; it does not say whether the version may count">Checks outcome</th><th>Made</th><th>Counts</th><th></th></tr>
    ${(LS.allSpreads ? shownSpreads : shownSpreads.slice(0, 15)).map(s => `<tr${LS.spread === s.run_id ? ' style="background:var(--blue-bg)"' : ""}><td>${esc(s.company || `CIK ${s.cik}`)}</td><td>${esc(s.fiscal_year)}</td><td>${esc(STMT[s.statement] || s.statement)}</td><td>${outcomeBadge(s.outcome)}</td>
      <td class="note">${esc(jobStarted(s.run_date || "") || s.run_date || "")}<div>template ${esc(s.template)}</div></td>
      <td>${s.current ? badge("counts", "blue") : `<button class="btn sm" data-sel="${esc(s.run_id)}">Make this count</button>`}</td>
      <td class="num">${LS.spread === s.run_id ? badge("viewing below", "blue") : `<button class="btn sm" data-open="${esc(s.run_id)}">View</button>`}</td></tr>`).join("") || `<tr><td colspan="7" class="note">${q ? "No spread matches." : "none yet"}</td></tr>`}</table>
    ${shownSpreads.length > 15 ? `<button class="btn sm" id="all-spreads" style="margin-top:8px">${LS.allSpreads ? "Show 15" : `Show all ${shownSpreads.length}`}</button>` : ""}
    <p class="note">A version counts per company, year, statement and template. The checks outcome says how the code checks ended. A version with failed checks or open questions can still be chosen. View shows its rows, checks, sources and open questions in step 3.</p></div></div>
  ${one ? spreadDetail(one) : `<div class="card step pad"><span class="stepno">3</span><b>View a spread</b> <span class="note">Press View on a spread in step 2 to see its rows, checks and where each amount came from.</span></div>`}
  <div id="res-said" style="margin-top:12px"></div>`);
  // the answer is written below every card, so it is brought into view (a refusal there was off-screen)
  const said = x => { const el = document.getElementById("res-said"); el.innerHTML = x; el.scrollIntoView({block: "nearest"}); };
  // a refusal is shown beside the button pressed (it was written far below, and missed), and in the box at the bottom
  const act = async (path, body, btn) => {
    if (btn) { btn.disabled = true; btn.nextElementSibling && btn.nextElementSibling.classList.contains("btn-said") && btn.nextElementSibling.remove(); }
    const a = await post(path, body);
    if (!stillHere(g, "results")) return;
    if (btn) btn.disabled = false;
    if (a.ok) return liveResults();
    if (btn) btn.insertAdjacentHTML("afterend", `<div class="btn-said note" style="text-align:left;max-width:420px">${refused(a)}</div>`);
    else said(refused(a));
  };
  view.querySelectorAll("[data-wr]").forEach(b => b.onclick = () => act("/api/result", {job_id: b.dataset.wr}, b));
  view.querySelectorAll("[data-dl]").forEach(x => x.addEventListener("click", () => {
    if (!x.nextElementSibling) x.insertAdjacentHTML("afterend", `<div class="note">Sent to your browser's downloads.</div>`); }));
  document.getElementById("by").oninput = e => { LS.by = e.target.value; };
  const rq = document.getElementById("res-q");
  rq.oninput = () => { LS.resQ = rq.value; keepSearchFocusOn("res-q", liveResults); };
  const flip = (id, k) => { const x = document.getElementById(id); if (x) x.onclick = () => { LS[k] = !LS[k]; liveResults(); }; };
  flip("all-rows", "allRows"); flip("all-waiting", "allWaiting"); flip("all-spreads", "allSpreads"); flip("all-results", "allResults");
  const close = document.getElementById("spread-close");
  if (close) close.onclick = () => { LS.spread = null; liveResults(); };
  view.querySelectorAll("[data-open]").forEach(b => b.onclick = async () => { LS.spread = b.dataset.open; LS.scrollTo = "spread-detail"; await liveResults(); });
  if (LS.scrollTo && one) { const el = document.getElementById(LS.scrollTo); LS.scrollTo = null; if (el) el.scrollIntoView({block: "start"}); }
  // the viewed spread's job pages: that job is chosen, then its page drawn (at the line or repair a source names)
  const toJob = (key, set) => { const id = one && one.ok && one.body.job_id; if (!id) return; chooseJob(id); Object.assign(LS, set); location.hash = `#/${key}`; };
  view.querySelectorAll("[data-goto]").forEach(b => b.onclick = () => toJob(b.dataset.goto, {}));
  view.querySelectorAll("[data-tb]").forEach(b => b.onclick = () => toJob("tiebreak", {line: b.dataset.tb}));
  view.querySelectorAll("[data-tr]").forEach(b => b.onclick = () => toJob("tree", {treeFolder: b.dataset.tr}));
  view.querySelectorAll("[data-imp]").forEach(b => b.onclick = async () => {
    const a = await post("/api/store/import", {job_id: b.dataset.imp});
    if (!stillHere(g, "results")) return;
    if (!a.ok) return said(refused(a));
    LS.importSaid = {job: b.dataset.imp, ...a.body};
    liveResults();
  });
  view.querySelectorAll("[data-sel]").forEach(b => b.onclick = () => act("/api/store/select", {run_id: b.dataset.sel, by: LS.by}, b));
}
// The download "Workflow result - unreviewed" of each finished job: the headline and
// per-statement counts the workbook's About sheet says, and the workbook made from the job's own records, never shown
// as approved. Edits made in Excel do not return here.
function workflowResults(finished, made) {
  const latest = id => made.find(m => m.job_id === id);
  const shown = LS.allResults ? finished : finished.slice(0, 5);
  const row = x => { const m = latest(x.job_id);
    return `<tr><td><b>${esc(jobLabel(x).title)}</b><div class="note">${esc(jobLabel(x).sub)}</div></td>
      <td>${m ? `${esc(m.headline)}${Object.entries(m.statements).map(([st, t]) => `<div class="note">${esc(STMT[st] || st)}: ${esc(t)}</div>`).join("")}` : `<span class="note">not made yet</span>`}</td>
      <td class="num">${m && m.matches ? `<a class="btn sm" data-dl href="/api/result/download?id=${encodeURIComponent(m.export)}" download>Download</a> ` : ""}<button class="btn sm" data-wr="${esc(x.job_id)}">${m ? "Make again" : "Make the workbook"}</button></td></tr>`; };
  return `<div class="card" style="margin-bottom:16px"><div class="hd"><h2>Workflow result — unreviewed</h2><p class="sub">For each finished job, the Excel workbook of what the workflow did: every printed line with its proposed row, evidence and status, the template rows and the checks, and a review sheet. Nobody has reviewed it. Correct it in Excel: the totals and checks recalculate from your corrections, which do not return to the system.</p></div><div class="bd">
    ${finished.length ? `<table class="wrap"><tr><th>Job</th><th>What the workflow classified</th><th></th></tr>${shown.map(row).join("")}</table>
    ${finished.length > 5 ? `<button class="btn sm" id="all-results" style="margin-top:8px">${LS.allResults ? "Show five" : `Show all ${finished.length}`}</button>` : ""}` : `<span class="note">No finished job yet.</span>`}</div></div>`;
}
// A search box whose page redraws on each key keeps its focus and caret.
async function keepSearchFocusOn(id, redraw) {
  const a = document.getElementById(id), s = a ? [a.selectionStart, a.selectionEnd] : null;
  await redraw();
  const q = document.getElementById(id);
  if (q) { q.focus(); try { q.setSelectionRange(s[0], s[1]); } catch (e) { /* type=search may refuse */ } }
}

// ---------- 7. Evals: every job side by side, then the shown job's operations ----------
// The grade's overall agreement with the answer key, or why there is none.
const agreement = gr => {
  const c = gr && gr.counts && gr.counts.all;
  return c ? `${Number(c.correct || 0)} of ${Number(c.correct || 0) + Number(c.wrong || 0) + Number(c["no answer"] || 0)} agree`
    : `<span class="note">no answer key</span>`;
};
const OPS = new Map();   // each job's operations, read once per visit to Evals; a running job is read again
const usedOf = (o, k) => { const c = o && o.caps && o.caps.find(c => c.cap === k);    // the job's ledger, embeddings included
  return c && c.used != null && c.used !== "" && Number.isFinite(Number(c.used)) ? Number(c.used) : null; };
async function liveEvals() {
  const g = ++LS.gen;
  const [id, list] = await Promise.all([shownJob(), get("/api/jobs")]);
  if (!stillHere(g, "evals")) return;
  if (!list.ok) return page("Evals", "", `<div class="card pad">${refused(list)}</div>`);
  const jobs = list.body.jobs || [];
  if (!jobs.length) return page("Evals", "", `<p class="note">No job yet. Start one on the Run page.</p>`);
  jobs.forEach(x => { if (!x.final) OPS.delete(x.job_id); });
  const shown = jobs.find(x => x.job_id === id) || jobs[0], L = jobLabel(shown);
  if (!OPS.has(shown.job_id) || !OPS.get(shown.job_id)) OPS.set(shown.job_id, await get(`/api/job/operations?id=${encodeURIComponent(shown.job_id)}`));
  if (!stillHere(g, "evals")) return;
  const one = OPS.get(shown.job_id);
  page("Evals", "What each job cost, how long it took and, where the company has an answer key (another model's reference spread, not proof), how many lines agree with it.", `
  ${pickerCard("Jobs", "Newest first. Search by company, 10-K date, start time or model. Choose one to see its details below.")}
  <div class="page-head" style="margin-top:22px"><div><h2>${esc(L.title)}</h2><p class="sub">${esc(L.sub)} · ${esc(partsSaid(shown.parts))}</p></div></div>
  ${one.ok ? operationsCards(one.body) : `<div class="card pad">${refused(one)}</div>`}`);
  // Dollars and minutes each with a bar on its own scale: the largest among the jobs read so far fills the cell.
  const most = k => Math.max(0, ...[...OPS.values()].map(r => usedOf(r && r.ok ? r.body : null, k)).filter(v => v != null));
  const bar = (v, k, said) => { const m = most(k);
    return `<div class="cellbar"><span>${said}</span><i style="width:${m ? (100 * v / m).toFixed(1) : 0}%"></i></div>`; };
  const cellOf = x => { const r = OPS.get(x.job_id), o = r && r.ok ? r.body : null, w = r ? "–" : "…";
    return `<td class="num">${usedOf(o, "requests") != null ? usedOf(o, "requests").toLocaleString("en-US") : w}</td>
      <td class="num">${usedOf(o, "dollars") != null ? bar(usedOf(o, "dollars"), "dollars", `$${usedOf(o, "dollars").toFixed(2)}`) : w}</td>
      <td class="num">${usedOf(o, "wall_s") != null ? bar(usedOf(o, "wall_s"), "wall_s", (usedOf(o, "wall_s") / 60).toFixed(1)) : w}</td>
      <td class="num">${o ? agreement(o.grade) : w}</td>`; };
  const o = {jobs, shown: shown.job_id, size: 10, cell: cellOf,
    head: `<th class="num" title="From the job's budget ledger: chat and embedding requests">Model requests</th><th class="num">Dollars</th><th class="num">Minutes</th><th class="num">Agreement with answer key</th>`,
    onShow: x => { chooseJob(x); liveEvals(); },
    drawn: async vis => {
      const need = vis.filter(x => !OPS.has(x.job_id));
      if (!need.length) return;
      need.forEach(x => OPS.set(x.job_id, null));
      const got = await Promise.all(need.map(x => get(`/api/job/operations?id=${encodeURIComponent(x.job_id)}`)));
      need.forEach((x, k) => OPS.set(x.job_id, got[k]));
      if (stillHere(g, "evals")) drawPicker(o);
    }};
  drawPicker(o);
}

// ---------- 8. Storage: file sizes of the fixed folders and a count of the stored spreads (listed on Results), read only ----------
// What each measured part holds, by the name the server gives it; the picture shows how a job uses them.
const PARTS = {"Downloaded 10-K filings": "the filings' files and statement pages, fetched once and verified",
  "FASB taxonomy editions": "the US GAAP editions the filings use, pinned by hash",
  "Job and step records": "every job's settings, progress, model calls and checks",
  "Results database": "the imported spreads, and which one counts for each statement and year"};
const mb = n => n >= 1073741824 ? `${(n / 1073741824).toLocaleString("en-US", {maximumFractionDigits: 1})} GiB` : `${(n / 1048576).toLocaleString("en-US", {maximumFractionDigits: 1})} MiB`;
async function liveStorage() {
  const g = ++LS.gen;
  page("Storage", "", `<p class="note">Measuring…</p>`);
  const r = await get("/api/storage");
  if (!stillHere(g, "storage")) return;
  if (!r.ok) return page("Storage", "", `<div class="card pad">${refused(r)}</div>`);
  const {roots, stored, measured} = r.body;
  const part = Object.fromEntries(roots.map(x => [x.root, x]));
  const box = n => `<div class="flow-box"><b>${esc(n)}</b><div class="note">${esc(PARTS[n])}</div>
    <div class="note">${part[n] ? `${part[n].files.toLocaleString("en-US")} file${part[n].files === 1 ? "" : "s"} · ${mb(part[n].bytes)}` : ""}</div></div>`;
  const arrow = t => `<div class="flow-arrow"><span class="note">${t}</span><span>→</span></div>`;
  // The four parts as one split bar, each segment its share of the bytes measured; legend with the same numbers.
  const total = roots.reduce((s, x) => s + Number(x.bytes || 0), 0);
  const SEG = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"];
  const share = x => total ? 100 * Number(x.bytes || 0) / total : 0;
  const pctSaid = x => { const p = share(x); return p > 0 && p < 0.1 ? "<0.1%" : `${p.toFixed(1)}%`; };
  page("Storage", `File sizes, not total disk usage, measured ${measured}.`, `
  <div class="card"><div class="hd"><h2>Space used: ${esc(mb(total))}</h2><p class="sub">Each part's share of the files this app keeps.</p></div><div class="bd">
    <div class="segbar">${roots.map((x, i) => share(x) ? `<div style="flex:${share(x)} 0 0;background:${SEG[i % 4]}" title="${esc(x.root)}: ${esc(mb(x.bytes))} (${pctSaid(x)})"></div>` : "").join("")}</div>
    <div class="seglegend">${roots.map((x, i) => `<div><i style="background:${SEG[i % 4]}"></i><b>${esc(x.root)}</b> <span class="note">${pctSaid(x)} · ${esc(mb(x.bytes))} · ${x.files.toLocaleString("en-US")} file${x.files === 1 ? "" : "s"}</span></div>`).join("")}</div></div></div>
  <div class="card" style="margin-top:16px"><div class="hd"><h2>How the parts are used</h2></div><div class="bd"><div class="flow"><div class="flow-col">${box("Downloaded 10-K filings")}${box("FASB taxonomy editions")}</div>
    ${arrow("a job reads them and writes")}${box("Job and step records")}${arrow("Import, on Results")}${box("Results database")}</div></div></div>
  <div class="card" style="margin-top:16px"><div class="bd"><table><tr><th>What</th><th>Where</th><th class="num">Files</th><th class="num">Size</th></tr>
    ${roots.map(x => `<tr><td>${esc(x.root)}${x.present ? "" : ` ${badge("not on this computer", "amber")}`}</td><td class="mono note">${esc(x.path)}</td><td class="num">${x.files.toLocaleString()}</td><td class="num" title="${x.bytes.toLocaleString("en-US")} bytes">${esc(mb(x.bytes))}</td></tr>`).join("")}</table>
    <p class="note">${stored ? `The results database stores ${Number(stored.spreads).toLocaleString("en-US")} spread version${stored.spreads === 1 ? "" : "s"}, ${Number(stored.current).toLocaleString("en-US")} of them current.` : "No results database yet."}
      <a href="#/results">The stored spreads are listed on Results</a>.</p></div></div>`);
}

// ---------- Advanced settings: the default configuration, read only (ui/advanced.py) ----------
const cell = x => typeof x === "number" ? x.toLocaleString("en-US") : esc(x ?? "–");
// Each setting's explanation, in this order: the value and where it applies, then what it does, then what the code
// enforces, the provider or hardware constraint and the recommendation with its reason, each in its own line.
const EXPLAIN = [["value", "Value"], ["unit", "Unit"], ["scope", "Applies"], ["source", "Read from"], ["for", "What it is for"],
  ["behaves", "How the pipeline behaves with it"], ["why", "Why it matters"], ["moves", "What changes when it moves"],
  ["enforced", "Limit the code enforces"], ["constraint", "Provider or hardware constraint"], ["recommended", "Recommendation"]];
async function liveAdvanced() {
  const g = ++LS.gen, sub = "The default configuration every job starts from, read now from the settings and the code. Nothing here can be changed.";
  page("Advanced settings", sub, `<p class="note">Reading…</p>`);
  const r = await get("/api/advanced");
  if (!stillHere(g, "advanced")) return;
  if (!r.ok) return page("Advanced settings", sub, `<div class="card pad">${refused(r)}</div>`);
  const intro = `<div class="callout"><b>You do not need this page to run a job.</b> The Run page has every choice a job needs. This page is the reference behind it: which model does each step, and the limits every job runs under.
    <div class="kv" style="margin-top:8px"><div class="k">token</div><div>a piece of text, about three quarters of a word; models read and write in tokens and are paid by them</div>
    <div class="k">context</div><div>how much text one request to a model may carry</div><div class="k">answer limit</div><div>the longest answer one request may return</div>
    <div class="k">limit, cap</div><div>a ceiling; a job that reaches one stops and keeps its records</div>${r.body.any_local ? `<div class="k">local, online</div><div>a model run on this computer (Ollama, free) or through OpenRouter (paid)</div>` : `<div class="k">online</div><div>a model run through OpenRouter (paid)</div>`}
    <div class="k">settings</div><div class="mono">${esc(r.body.settings || "–")}</div></div></div>`;
  page("Advanced settings", sub, intro + r.body.parts.map(p => `<div class="card" style="margin-top:18px"><div class="hd"><h2>${esc(p.title)}</h2><p class="sub">${esc(p.about)}</p></div>
    <div class="bd"><table><tr>${p.columns.map(c => `<th>${esc(c)}</th>`).join("")}</tr>
    ${p.rows.map(row => `<tr>${row.map(cell).map(c => `<td>${c}</td>`).join("")}</tr>`).join("")}</table></div></div>`).join("")
    + (r.body.explained ? `<div class="card" style="margin-top:18px"><div class="hd"><h2>What each setting does</h2><p class="sub">${esc(r.body.explained_about)}</p></div><div class="bd">
    ${r.body.explained.map((x, i) => `<details class="${i ? "sep" : ""}" style="padding:6px 0"><summary><b>${esc(x.setting)}</b> <span class="note">${cell(x.value)}${x.unit ? ` ${esc(x.unit)}` : ""}</span></summary><div class="kv" style="margin-top:6px">${EXPLAIN.map(([k, label]) => `<div class="k">${esc(label)}</div><div>${cell(x[k])}</div>`).join("")}</div></details>`).join("")}</div></div>` : "")
    + `<p class="note" style="margin-top:12px">Each job records the settings it actually used, on its Jobs page.</p>`);
}

// ---------- 5e. Replay: a job played back from its own records ----------
// Each statement's printed lines are read in the order the job took them, and each goes to the template row(s) its
// record names, with the checker's review and any vote; at the end the totals are checked as the job checked them.
// The records are read once (/api/job/replay); nothing is sent to a model. A model call opens what the model was sent
// and what it answered (/api/job/call). A replay is the same every time for the same job.
const RP = {gen: 0, tok: 0, data: null, job: null, label: null, st: 0, i: -1, playing: false, speed: 1};
const ROLE_SAID = {analyst: "Analyst", checker: "Checker", organizer: "Organizer", tiebreaker: "Tiebreaker"};
const REVIEW_SAID = {agree: ["checker agreed", "green"], disagree: ["checker disagreed", "amber"]};
const rpMs = ms => ms / RP.speed;
const rpLive = t => t === RP.tok && stillHere(RP.gen, "replay");
const rpText = x => x === null || x === undefined ? "" : typeof x === "string" ? (() => { try { return JSON.stringify(JSON.parse(x), null, 2); } catch { return x; } })() : JSON.stringify(x, null, 2);

async function liveReplay() {
  const g = ++LS.gen, id = await shownJob();
  if (!stillHere(g, "replay")) return;
  if (!id) { page("Replay", "", `<p class="note">No job yet. Start one on the Run page.</p>`); return; }
  if (RP.job !== id || !RP.data) {
    page("Replay", "", `<p class="note">Reading the job's records…</p>`);
    const [r, j] = await Promise.all([get(`/api/job/replay?id=${encodeURIComponent(id)}`), get(`/api/job?id=${encodeURIComponent(id)}&from=999999999`)]);
    if (!stillHere(g, "replay")) return;
    if (!r.ok) { page("Replay", "", `<div class="card pad">${refused(r)}</div>`); return; }
    Object.assign(RP, {data: r.body, job: id, st: 0, i: -1, label: jobLabel({job_id: id, manifest: j.ok ? j.body.manifest : null}),
                       running: j.ok && !j.body.final});
  }
  RP.gen = g; RP.tok += 1; RP.playing = false;
  drawReplay();
}

// What a placed piece adds to its row: its signed amount (views.replay_signed, the claim's "signed", the amount the
// workbook's Lines sheet and the run's rows add up); nothing for a claim the run dropped; null when not recorded. The
// line event's amount carries no row sign: adding it read wrong (a common equity
// total read far too high).
const rpAdds = p => p.signed != null ? Number(p.signed) : p.dropped ? 0 : null;
// What line l adds to each of its rows (pieces in one row summed); a row with a piece not recorded is left out.
const rpLineAdds = l => { const by = {}, unknown = new Set(); (l.placed || []).forEach(p => { const a = rpAdds(p); if (a == null) unknown.add(p.row); else by[p.row] = (by[p.row] || 0) + a; }); unknown.forEach(r => delete by[r]); return by; };
// The state after line i of statement st: the lines taken so far, what each row holds, and the rows whose total is
// incomplete (a piece in it whose contribution is not recorded).
function rpState(s, i) {
  const done = new Set(), rows = {}, incomplete = new Set();
  s.lines.slice(0, Math.max(0, i + 1)).forEach(l => {
    done.add(l.concept);
    // a signed piece fills its row; a dropped claim adds nothing and fills nothing (the workbook leaves a row no claim
    // fills unknown); a piece with no recorded contribution, or an older event with a row and no pieces, marks it
    (l.placed || []).forEach(p => { if (p.signed != null) rows[p.row] = (rows[p.row] || 0) + Number(p.signed); else if (!p.dropped) incomplete.add(p.row); });
    if (!(l.placed || []).length && l.row) incomplete.add(l.row);
  });
  return {done, rows, incomplete};
}
// A row's amount in the template panel: its total, marked when a contribution is not recorded.
const rpRowSaid = (st, id) => `${money(st.rows[id])}${st.incomplete.has(id) ? " (incomplete)" : ""}`;
// A total's checks: the comparisons the records ran on exactly this concept (views.replay_compared), each passed or not.
const rpTotalChecks = (s, concept) => ((s.printed.find(p => p.concept === concept) || {}).checked || []).map(ok => ({ok}));
const rpN = (n, one, many) => `${n} ${n === 1 ? one : many}`;
// The sentence before the first line, from the counts of the records themselves: the lines taken (line events), the
// printed totals, and the printed lines neither taken nor a total (an earlier
// sentence took printed minus taken as the totals and read "the other -37")
const rpBefore = s => { const took = new Set(s.lines.map(l => l.concept)), tot = s.printed.filter(p => p.total && !took.has(p.concept)).length, totc = s.printed.filter(p => p.total && !took.has(p.concept) && (p.checked || []).length).length, rest = s.printed.filter(p => !p.total && !took.has(p.concept)).length, out = s.lines.filter(l => l.outside).length, tl = s.lines.filter(l => l.total && !(l.placed || []).length).length, tc = s.lines.filter(l => l.total && l.compared && !(l.placed || []).length).length; const nr = s.lines.filter(lineNotRun).length; return `<b>Before the lines.</b> The job recorded ${rpN(s.lines.length, "line", "lines")}, shown one at a time below in the order it recorded them, toward the template's ${rpN(s.sections.length, "section", "sections")} on this statement (${s.sections.map(x => esc(sectionName(x))).join(", ")})${out ? `; ${rpN(out, "of them goes", "of them go")} outside the template` : ""}${tl ? `; ${rpN(tl, "of them is a printed total", "of them are printed totals")} (${tc === tl ? (tl === 1 ? "compared" : "all compared") : tc ? `${tc} compared` : tl === 1 ? "not compared" : "none compared"} in this statement's checks)` : ""}${nr ? `; ${rpN(nr, "of them was", "of them were")} not run, each with its recorded reason` : ""}.${tot ? ` ${rpN(tot, "more printed total is", "more printed totals are")} shown (${totc === tot ? (tot === 1 ? "compared" : "all compared") : totc ? `${totc} compared` : tot === 1 ? "not compared" : "none compared"} in this statement's checks, at the end).` : ""}${rest ? ` ${rpN(rest, "more printed line is", "more printed lines are")} shown but not taken one by one (a subtotal, for example).` : ""}`; };
// Where a line went: its rows, or outside the template with the recorded reason, or no row with its reason
const rpWhy = l => String(l.why || "").replace(/^both models: OUTSIDE$/, "the analyst and the checker each put it outside");
const rpWhere = l => (l.placed || []).length ? l.placed.map(p => p.signed != null ? `<b>${esc(rowName(p.row))}</b> ${money(p.signed)}${Number(p.signed) !== Number(p.amount) ? ` <span class="note">(recorded on the line as ${money(p.amount)}; the row adds it with its sign)</span>` : ""}` : p.dropped ? `<b>${esc(rowName(p.row))}</b> <span class="note">${money(p.amount)} recorded on the line; the run dropped this claim (its amount was already claimed), so the row adds nothing</span>` : `<b>${esc(rowName(p.row))}</b> <span class="note">${money(p.amount)} recorded on the line; what it adds to the row is not recorded</span>`).join(" + ") : l.row ? `<b>${esc(rowName(l.row))}</b> <span class="note">(amount not recorded in this older event)</span>` : l.total ? `<span class="note">${l.total_said === "parts" ? "a printed total: its printed parts add up to it (checked by code; no check on this statement lists it)" : l.total_said === "none" ? "a printed total (the template compares it as one; no check on this statement compares it)" : l.compared ? "a printed total: compared in this statement's checks" : "a printed total (the template compares it as one; no check on this statement compares it)"}</span>`: l.outside ? `<span class="note">outside the template: ${esc(rpWhy(l) || l.class)}</span>` : `<span class="note">no row: ${esc(rpWhy(l) || noRowSaid(l))}</span>`;
// The line's badges, each said once: the review (unless the outside text says it), then a class that differs from it
const rpBadges = l => { const rev = REVIEW_SAID[l.review], own = l.review && !(l.outside && String(l.review).startsWith("outside")); return `${rev ? badge(rev[0], rev[1]) : own ? badge(l.review, "grey") : ""}${l.vote ? ` ${badge(`tiebreaker: ${l.vote}`, "blue")}` : ""}${l.class && l.class !== l.review ? ` ${badge(l.class, "grey")}` : ""}`; };
// The checks line at a statement's end: every check the run recorded, and the template's report checks, so it can be
// read beside Results, which counts the statement checks that decide the outcome
const rpChecks = s => { const where = x => x === "run" ? "the whole statement" : sectionName(x), both = [...(s.controls || []).map(c => ({...c, said: `${capFirst(where(c.section))}: ${c.name ? checkSaid({name: c.name}) : "the spread's total against the filed total"}`})), ...(s.reports || []).map(r => ({...r, said: `${r.id} ${r.name || ""}`}))], known = both.some(c => typeof c.decides === "boolean"), dec = both.filter(c => c.decides === true), shown = both.filter(c => c.decides === false), tally = xs => ["tied", "failed", "unavailable"].map(k => [k, xs.filter(c => c.state === k).length]).filter(x => x[1]).map(x => `${x[1]} ${x[0]}`).join(", ") || "none", item = c => `<div class="note">${badge(c.state === "tied" ? "ties" : c.state, c.state === "tied" ? "green" : c.state === "failed" ? "red" : "grey")} ${esc(c.said)}${c.recorded && c.recorded !== c.state ? ` (recorded: ${esc(checkState(String(c.recorded)))}${c.why ? `; ${esc(c.why)}` : ""})` : c.why ? ` (${esc(c.why)})` : ""}</div>`, line = s.checks.filter(c => !c.control), failed = line.filter(c => !c.ok), ok = line.filter(c => c.ok), one = (c, b) => `<div class="note">${b} ${esc(checkSaid({name: c.check}))} · ${esc(where(c.section))}</div>`; return `<b>Checks by code</b> · outcome ${outcomeBadge(s.outcome || "not recorded")}${known ? `<div style="margin-top:4px">${s.gate === "none" ? "The template names no statement checks, so every check the run recorded counts for the outcome, these totals compared with the filing included" : "Statement checks, the ones Results counts for the outcome (totals compared with the filing and the template's report checks)"}: ${tally(dec)}.</div>${dec.map(item).join("")}${shown.length ? `<div style="margin-top:4px">Shown only, deciding nothing${s.gate === "none" ? " (report checks count only when the template names them)" : ""}: ${tally(shown)}.</div>${shown.map(item).join("")}` : ""}` : both.length ? `<div style="margin-top:4px">Totals compared with the filing and the template's report checks: ${tally(both)}. Which of them decide the outcome is not known here: the template the job ran with could not be opened.</div>${both.map(item).join("")}` : ""}<div style="margin-top:4px">Section checks the run recorded (checks.json), as recorded: ${ok.length} of ${rpN(line.length, "check", "checks")} passed.</div>${failed.map(c => one(c, badge("failed", "red"))).join("")}${ok.length ? `<details style="margin-top:4px"><summary class="note">The ${rpN(ok.length, "check", "checks")} that passed</summary>${ok.map(c => one(c, badge("passed", "green"))).join("")}</details>` : ""}`; };

function drawReplay() {
  const S = RP.data.statements;
  const sub = "A finished job played back from its own records: each line is shown in the order the job recorded it and goes to the template row its record names, or outside the template. Nothing is sent to a model.";
  if (!S.length) { page("Replay", sub, jobShown(RP.label) + `<p class="note">${RP.running ? "No statement has been planned yet." : "This job recorded no spread."}</p>`); return; }
  const s = S[RP.st], L = s.lines.length, st = rpState(s, Math.min(RP.i, L - 1)), atEnd = RP.i >= L;
  const company = (RP.label.title || "").replace(/, 10-K .*$/, "");
  const secs = TPL.sections.filter(x => x.statement === s.statement);
  page("Replay", sub, `${jobShown(RP.label)}
  <div class="card"><div class="hd row-flex">
    <div class="row-flex">${S.map((x, k) => `<button class="btn sm ${k === RP.st ? "primary" : ""}" data-st="${k}">${esc(STMT[x.statement] || x.statement)}</button>`).join(" ")}</div>
    <span class="note">in the order the job ran them</span><span class="spacer"></span>
    <button class="btn sm" id="rp-back" ${RP.i < 0 ? "disabled" : ""}>‹ Line</button>
    <button class="btn sm primary" id="rp-play">${rpPlay()}</button>
    <button class="btn sm" id="rp-next" ${atEnd && RP.st >= S.length - 1 ? "disabled" : ""}>Line ›</button>
    <button class="btn sm" id="rp-restart">Start over</button>
    <select id="rp-speed" title="Speed">${[1, 2, 4].map(v => `<option value="${v}" ${v === RP.speed ? "selected" : ""}>${v}×</option>`).join("")}</select></div>
  <div class="bd">
    <div class="rp-now" id="rp-now" style="margin-top:0">${rpNow(s)}</div>
    <div id="rp-bar">${bar(Math.max(0, Math.min(RP.i + 1, L)), L)}</div><p class="note" id="rp-count">${esc(rpCount(s))}</p>
    <div class="rp-stage" id="rp-stage">
      <div><div class="rp-cap">The filing: ${esc(STMT[s.statement] || s.statement)}, the lines the job read, ${s.printed_order ? "in printed order" : "in the order recorded (no printed-order record)"}</div>
        <div class="rp-paper" id="rp-paper"><div class="rp-title">${esc(company)}<div class="note">${esc(STMT[s.statement] || s.statement)}</div></div>
        ${s.printed.map((p, k) => { const ck = p.total && atEnd ? rpTotalChecks(s, p.concept) : [];
          return `<div class="rp-line ${p.total ? "total" : ""} ${st.done.has(p.concept) ? "done" : ""}" data-c="${esc(p.concept)}" id="rp-l-${k}"><span>${esc(p.label)}</span>
          <span class="num">${ck.length ? (ck.every(c => c.ok) ? badge("checked", "green") : badge("check failed", "red")) + " " : ""}${money(p.amount)}</span></div>`; }).join("")}
        <div class="rp-scan" id="rp-scan"></div></div></div>
      <div><div class="rp-cap">The template: where each line goes</div>
        <div class="rp-tpl" id="rp-tpl">${secs.map(x => `<div class="rp-sec">${esc(x.name)}</div>
        ${TPL.rows.filter(r => r.section === x.id).map(r => `<div class="rp-row ${r.id in st.rows ? "filled" : ""}" data-row="${esc(r.id)}"><span>${esc(r.name)}</span><span class="num">${r.id in st.rows ? rpRowSaid(st, r.id) : ""}</span></div>`).join("")}`).join("")}</div></div>
    </div>
  </div></div>`);
  wireReplay();
  const cur = RP.i >= 0 && RP.i < L ? rpLineEl(s.lines[RP.i].concept) : null;
  if (cur) rpScanTo(cur, true);
}

function rpNow(s) {
  const L = s.lines.length;
  const callBtns = cs => cs.map(c => `<button class="btn sm" data-call="${Number(c.n)}" data-child="${esc(s.child)}">${esc(ROLE_SAID[c.role] || c.role)} · ${Number(c.seconds || 0).toFixed(1)} s${c.cost_usd ? ` · $${Number(c.cost_usd).toFixed(4)}` : ""}${c.error ? " · error" : ""}</button>`).join(" ");
  if (RP.i < 0) return `${rpBefore(s)}
    ${s.other_calls.length ? `<div style="margin-top:8px"><span class="note">Its model call:</span> ${callBtns(s.other_calls)}</div>` : ""}`;
  if (RP.i >= L) {
    return `${rpChecks(s)}
      ${RP.st < RP.data.statements.length - 1 ? `<div style="margin-top:8px"><button class="btn sm" id="rp-turn">Next statement ›</button></div>` : `<div style="margin-top:8px"><a class="btn sm" href="#/results">See the result</a></div>`}`;
  }
  const l = s.lines[RP.i];
  return `<div class="row-flex"><b>${esc(l.label)}</b><span class="note">${l.section ? esc(sectionName(l.section)) : l.queued ? "no section from the organizer" : ""}${l.from ? ` · moved from ${esc(sectionName(l.from))}` : ""}</span><span class="spacer"></span>
    ${rpBadges(l)}</div>
    <div style="margin-top:4px">→ ${rpWhere(l)}</div>
    ${l.calls.length ? `<div style="margin-top:8px"><span class="note">The model calls on this line, in order (press one to see what it was sent and answered):</span><div class="row-flex" style="flex-wrap:wrap;gap:6px;margin-top:4px">${callBtns(l.calls)}</div></div>` : ""}`;
}

const rpLineEl = c => [...document.querySelectorAll("#rp-paper .rp-line")].find(e => e.dataset.c === c) || null;
const rpRowEl = id => [...document.querySelectorAll("#rp-tpl .rp-row")].find(e => e.dataset.row === id) || null;
function rpCenter(box, el) { box.scrollTop = Math.max(0, el.offsetTop - box.clientHeight / 2 + el.offsetHeight / 2); }
function rpScanTo(el, now) {
  const scan = document.getElementById("rp-scan"), paper = document.getElementById("rp-paper");
  if (!scan || !el) return;
  rpCenter(paper, el);
  scan.style.transition = now ? "none" : `top ${rpMs(380)}ms ease, height ${rpMs(380)}ms ease`;
  scan.style.top = `${el.offsetTop}px`; scan.style.height = `${el.offsetHeight}px`; scan.style.opacity = "1";
}
// A chip with the line's label and amount moves from the printed line to each row it went to.
function rpFly(fromEl, rowIds, amountOf, t, then) {
  const stage = document.getElementById("rp-stage"), tpl = document.getElementById("rp-tpl");
  if (!stage || !fromEl || !rowIds.length) { then(); return; }
  const sr = stage.getBoundingClientRect(), f = fromEl.getBoundingClientRect();
  let left = rowIds.length;
  rowIds.forEach((id, k) => {
    const row = rpRowEl(id);
    if (!row) { if (--left === 0) then(); return; }
    if (k === 0) rpCenter(tpl, row);
    const chip = document.createElement("div");
    chip.className = "rp-chip";
    const a = amountOf(id);
    chip.innerHTML = `${esc(fromEl.firstElementChild.textContent)}${a == null ? "" : ` <b>${money(a)}</b>`}`;
    chip.style.left = `${f.left - sr.left}px`; chip.style.top = `${f.top - sr.top}px`; chip.style.width = `${f.width}px`;
    stage.appendChild(chip);
    requestAnimationFrame(() => {
      const r = row.getBoundingClientRect();
      chip.style.transition = `transform ${rpMs(700)}ms cubic-bezier(.2,.8,.2,1), opacity ${rpMs(700)}ms`;
      chip.style.transform = `translate(${r.left - f.left}px, ${r.top - f.top}px) scale(.96)`;
    });
    setTimeout(() => {
      chip.remove();
      if (!rpLive(t)) return;
      row.classList.add("filled", "hit");
      setTimeout(() => row.classList.remove("hit"), rpMs(900));
      if (--left === 0) then();
    }, rpMs(720));
  });
}

// One step forward, animated: the next line (scan, then move to its rows), or the totals check, or the next statement.
function rpStep(t) {
  const S = RP.data.statements, s = S[RP.st], L = s.lines.length;
  if (RP.i >= L) {
    if (RP.st >= S.length - 1) { RP.playing = false; drawReplay(); return; }
    rpTurn(t); return;
  }
  RP.i += 1;
  const now = document.getElementById("rp-now");
  if (now) now.innerHTML = rpNow(s);
  drawProgress(s);
  wireCalls();
  if (RP.i >= L) { const pb = document.getElementById("rp-play"); if (pb) pb.textContent = rpPlay(); rpTotals(t, s); return; }
  const l = s.lines[RP.i], el = rpLineEl(l.concept);
  if (el) rpScanTo(el);
  const pieces = l.placed || [], st = rpState(s, RP.i), adds = rpLineAdds(l);
  setTimeout(() => {
    if (!rpLive(t)) return;
    // the chip carries what this line adds to the row; the row's cell then shows its running total
    rpFly(el, [...new Set(pieces.map(p => p.row))], id => id in adds ? adds[id] : null, t, () => {
      if (!rpLive(t)) return;
      if (el) el.classList.add("done");
      pieces.forEach(p => { const r = rpRowEl(p.row); if (r) r.lastElementChild.textContent = rpRowSaid(st, p.row); });
      if (RP.playing) rpNext(t, 900); else drawReplay();
    });
  }, rpMs(450));
}
// The Play button's label: what pressing it does now (wireReplay's rp-play)
const rpPlay = () => { const S = RP.data.statements, atEnd = RP.i >= S[RP.st].lines.length; return RP.playing ? "Pause" : RP.i < 0 && RP.st === 0 ? "Play" : atEnd && RP.st < S.length - 1 ? "Play the next statement" : atEnd ? "Replay from start" : "Play from here"; };
// "N of L recorded lines shown": the lines whose step has started, the line shown included (the same count as drawReplay's)
const rpCount = s => `${Math.max(0, Math.min(RP.i + 1, s.lines.length))} of ${s.lines.length} recorded lines shown in ${STMT[s.statement] || s.statement} · statement ${RP.st + 1} of ${RP.data.statements.length}`;
function drawProgress(s) {
  const L = s.lines.length, b = document.getElementById("rp-bar"), p = document.getElementById("rp-count");
  if (b) b.innerHTML = bar(Math.max(0, Math.min(RP.i + 1, L)), L);
  if (p) p.textContent = rpCount(s);
}
// The totals, one by one, as the code checked them.
function rpTotals(t, s) {
  const totals = s.printed.filter(p => p.total);
  let k = 0;
  const one = () => {
    if (!rpLive(t)) return;
    if (k >= totals.length) { RP.playing = false; drawReplay(); return; }
    const p = totals[k++], el = rpLineEl(p.concept), ck = rpTotalChecks(s, p.concept);
    if (el) {
      rpScanTo(el);
      if (ck.length) el.lastElementChild.insertAdjacentHTML("afterbegin", (ck.every(c => c.ok) ? badge("checked", "green") : badge("check failed", "red")) + " ");
    }
    setTimeout(one, rpMs(500));
  };
  one();
}
// The page turns to the next statement.
function rpTurn(t) {
  const paper = document.getElementById("rp-paper");
  if (paper) paper.classList.add("turn-out");
  setTimeout(() => {
    if (!rpLive(t)) return;
    RP.st += 1; RP.i = -1;
    drawReplay();
    const p = document.getElementById("rp-paper");
    if (p) p.classList.add("turn-in");
    if (RP.playing) rpNext(t, 1600);
  }, rpMs(520));
}
function rpNext(t, ms) { setTimeout(() => { if (rpLive(t) && RP.playing) rpStep(t); }, rpMs(ms)); }

function wireCalls() {
  document.querySelectorAll("[data-call]").forEach(b => b.onclick = () => rpCall(b.dataset.child, b.dataset.call));
  const turn = document.getElementById("rp-turn");
  if (turn) turn.onclick = () => { RP.tok += 1; rpTurn(RP.tok); };
}
function wireReplay() {
  const stop = () => { RP.tok += 1; RP.playing = false; };
  document.querySelectorAll("[data-st]").forEach(b => b.onclick = () => { stop(); RP.st = +b.dataset.st; RP.i = -1; drawReplay(); });
  document.getElementById("rp-play").onclick = () => {
    if (RP.playing) { stop(); drawReplay(); return; }
    const S = RP.data.statements;
    if (RP.st >= S.length - 1 && RP.i >= S[RP.st].lines.length) { RP.st = 0; RP.i = -1; }
    RP.tok += 1; RP.playing = true; drawReplay(); rpStep(RP.tok);
  };
  // a step cut short by the next press never reaches its end, so the page is first drawn as the records stand
  document.getElementById("rp-next").onclick = () => { stop(); drawReplay(); rpStep(RP.tok); };
  document.getElementById("rp-back").onclick = () => { stop(); RP.i = Math.max(-1, Math.min(RP.i, RP.data.statements[RP.st].lines.length) - 1); drawReplay(); };
  document.getElementById("rp-restart").onclick = () => { stop(); RP.st = 0; RP.i = -1; drawReplay(); };
  document.getElementById("rp-speed").onchange = e => { RP.speed = Number(e.target.value) || 1; };
  wireCalls();
}

// A step's model calls, from the Jobs page: one row per recorded call, each opening what it was sent and answered.
async function callList(job, child, step) {
  drawer(`Model calls · ${step}`, `<p class="note">Reading the step's call log…</p>`);
  const r = await get(`/api/job/calls?id=${encodeURIComponent(job)}&child=${encodeURIComponent(child)}`);
  if (!r.ok) { drawer(`Model calls · ${step}`, refused(r)); return; }
  const cs = r.body.calls;
  drawer(`Model calls · ${step}`, `<p class="note">${cs.length} recorded call${cs.length === 1 ? "" : "s"}, in the order made. Press one to see the instructions, the message sent and the reply.</p>
    <table class="wrap"><tr><th>#</th><th>Time</th><th>Role</th><th>What for</th><th class="num">Tokens</th><th class="num">Dollars</th></tr>
    ${cs.map(c => `<tr><td><button class="btn sm" data-one="${Number(c.n)}">${Number(c.n) + 1}</button></td><td class="note">${esc(String(c.time || "").slice(11, 19))}</td>
      <td>${esc(ROLE_SAID[c.role] || c.role)}<div class="note mono">${esc(c.model)}</div></td><td class="note">${esc(c.tag || "")}${c.attempt > 1 ? ` · attempt ${Number(c.attempt)}` : ""}${c.error ? ` ${badge("error", "red")}` : ""}</td>
      <td class="num">${(Number(c.prompt_tokens || 0) + Number(c.completion_tokens || 0)).toLocaleString("en-US")}</td><td class="num">${c.cost_usd ? Number(c.cost_usd).toFixed(5) : "–"}</td></tr>`).join("")}</table>`);
  document.querySelectorAll("#drawer-body [data-one]").forEach(b => b.onclick = () => openCall(job, child, b.dataset.one, () => callList(job, child, step)));
}

// One model call: what it was sent and what it answered, as recorded.
const rpCall = (child, n) => openCall(RP.job, child, n);
async function openCall(job, child, n, back) {
  const r = await get(`/api/job/call?id=${encodeURIComponent(job)}&child=${encodeURIComponent(child)}&n=${encodeURIComponent(n)}`);
  if (!r.ok) { drawer("Model call", refused(r)); return; }
  const c = r.body, pre = x => `<pre>${esc(rpText(x))}</pre>`;
  drawer(`${ROLE_SAID[c.role] || c.role} · ${c.tag || ""}`, `
    <div class="kv" style="margin-bottom:12px"><div class="k">Time</div><div>${esc(c.time)} (${esc(c.seconds)} s)</div>
      <div class="k">Model</div><div class="mono">${esc(c.model)}</div>
      <div class="k">Tokens</div><div>${Number(c.prompt_tokens || 0).toLocaleString("en-US")} sent, ${Number(c.completion_tokens || 0).toLocaleString("en-US")} returned</div>
      <div class="k">Dollars</div><div>${c.cost_usd ? `$${Number(c.cost_usd).toFixed(5)}` : "–"}</div>
      ${c.error ? `<div class="k">Error</div><div>${esc(rpText(c.error))}</div>` : ""}</div>
    <details><summary><b>Instructions (system prompt)</b></summary>${pre(c.system)}</details>
    <details style="margin-top:8px"><summary><b>Answer format the model must follow</b></summary>${pre(c.schema)}</details>
    <h3 style="margin-top:14px">What it was sent</h3>${pre(c.user)}
    <h3 style="margin-top:14px">The reply</h3>${pre(c.raw ?? c.answer)}
    ${c.reasoning ? `<h3 style="margin-top:14px">Reasoning returned by the model</h3>${pre(c.reasoning)}` : `<p class="note">The model returned no separate reasoning for this call.</p>`}
    <p class="note" style="margin-top:12px">From the job's record ${esc(child)}, call log entry ${Number(c.n) + 1}.</p>${back ? `<button class="btn sm" id="call-back">Back to the list</button>` : ""}`);
  const bk = document.getElementById("call-back");
  if (bk) bk.onclick = back;
}

// Addresses a person may type for a page whose own address is shorter.
const ALIAS = {tiebreaker: "tiebreak"};
window.LIVE_PAGES = {setup: liveSetup, advanced: liveAdvanced, template: liveTemplate, companies: liveCompanies, run: liveRun, jobs: liveJobs,
  tiebreak: liveTiebreak, tree: liveTree, replay: liveReplay, results: liveResults, evals: liveEvals, storage: liveStorage};
// The last script: the pages are defined, so app.js may draw.
pagesReady();
