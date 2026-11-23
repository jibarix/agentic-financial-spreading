# Agentic financial spreading

Repository: https://github.com/jibarix/agentic-financial-spreading

Agentic spreading turns one company's annual report (a 10-K filed with the SEC, with inline XBRL tags) into a fixed
template of income statement and balance sheet rows, with depreciation and amortization read from the cash flow
statement as a memo row. Credit analysts call this *spreading*. The spread serves two separate uses: comparing
peers on valuation multiples, and discounted cash flow valuations.

The models see the financial statements only: no notes, no other part of the filing, no search. A line the
statements do not settle is left *unresolved: not established from the allowed statements*.

Models decide where each printed line goes. Code does all the arithmetic and all the checks. Every amount keeps a
link to the fact in the filing it came from. A line the system cannot place is recorded with its reason, not hidden.

## What it does

For one 10-K, a job runs these steps:

1. **Corrections (model).** Before anything else, a model reads each statement and points to the printed lines that
   are the totals the template needs. Code accepts a choice only if it passes its checks.
2. **Extraction (code).** The statement lines and amounts are read from the filing's inline XBRL data.
3. **Organizer (model).** It sorts each line into a section of the template. Code checks each section against its
   printed total.
4. **Analyst (model).** For each line, it picks a template row and cites its evidence. It can split a line into
   pieces for different rows, or say the statements combine items, do not establish the answer (the line is then
   *unresolved: not established from the allowed statements*), or that the line falls outside the template. The analyst
   can read calculation links, open a printed page, recall its earlier tool results and look up another template
   section's rows. The printed page can include a note table.
5. **Checker (model).** A model from a different family answers the same line without seeing the analyst's answer.
6. **Second look.** For some disagreements, the analyst sees both answers and takes a second look.
7. **Tiebreaker (model).** A third model may answer a disputed line independently. A placement can remain
   disputed, with the reason recorded.
8. **Checks (code).** Code builds the spread and checks it. The statement checks decide each statement's outcome:
   total assets and total liabilities and equity against the company's printed totals, assets equal to liabilities
   plus equity as spread, and cash, revenue, operating income and net income against the printed figures. With them,
   every line must be placed once and nothing counted twice. A section's tie to its own printed subtotal (current
   assets, equity, other income and the rest) is computed and shown, and decides nothing. A check the filing cannot
   support (the company prints no such total) is *unavailable* and fails nothing. Each statement ends *complete*,
   *qualified* (the checks pass but an exception remains) or *blocked*; a blocked statement keeps every line placed
   and is still in the workbook, with the failed check named.
9. **Repair (models and code), off in this version.** Up to four sections with a failed statement check per job
   can get one repair attempt each: a proposer suggests changes, a critic comments, a controller decides how to
   continue, and a change is kept only if it passes the code checks and a fresh review by the checker's model. The
   default settings switch it off: under the statement checks it has almost nothing it can repair (most of those
   checks cover a whole statement, not one section), so each failed section is recorded as skipped, with no model
   call.

Each line is shown to the models by a neutral ID, with its printed label and amount.
Code keeps the tags for its own checks. A model never produces a number that enters a calculation: it chooses rows
and facts, and code fetches and adds the amounts.

The default template is engine template v16: the statement inputs of a valuation engine. Its rows are engine
template v14's; v15 changed only its checks, and v16 only the D&A memo row, which now takes every depreciation and
amortization line the cash flow statement prints, summed. It has 48 rows in 11 sections (4 for the income statement, 6 for the balance sheet and 1 cash flow memo section for D&A). 33 rows take
lines, 7 of them catch-all rows; 14 are derived by code: gross profit, the engine's G&A, operating income, other
income excluding interest expense, EBIT, pretax income, income from continuing operations, consolidated and parent
net income, and the balance sheet totals (there is no EBITDA row). A *not spread* list names 8 inputs the template
leaves out on purpose, each with the reason. Six report checks compare the finished spread with the filing:
total assets against total liabilities and capital, as spread and as filed; the cash row against the filed cash; the
equity rows against the filed total equity; the total assets row against the filed total assets; and the revenue row
against the filed revenue (when the filing prints no such total, the check reads the named alternative and says which
total it read). Four of them (assets against liabilities and capital as spread, cash, total assets, revenue), with
the checks of total liabilities and equity, operating income and net income against the printed figures, are the
statement checks.

The UI shows the template on its Template page: its sections, subtotals and checks, and each row's definition. In
this version the template is a fixed standard, changed only in code; the Template page shows it read only.

## The tested configuration

The UI starts jobs with one configuration: settings v31, engine template v16, statements only, and the model
line-up below. Settings v31 is v30 with template v16; v30 is v29 with the cash flow statement's request limit raised
from 50 to 100. All its models run on OpenRouter. An earlier version, settings v28 with engine template v14 (the same rows;
every section's subtotal check decided the outcome), was run once on each of three companies (Design Therapeutics,
Genco Shipping & Trading and Acurx Pharmaceuticals): every balance sheet check tied, and every line agreed with the
answer key (15 of 15, 31 of 31 and 10 of 10 lines). That is not an independent test: template v14's wording changes
were written from these same three filings. How it does on other companies has not been measured.

| Seat | Model |
|---|---|
| Corrections, organizer, analyst (and its second look) | Claude Haiku 5.5, reasoning off |
| Checker | GPT-6 Luna, low reasoning effort |
| Tiebreaker | DeepSeek V4.1 Flash, reasoning off |
| Repair: proposer and its questions about a line's meaning | Claude Haiku 5.5 |
| Repair: controller, and the review of a repaired line | GPT-6 Luna |
| Repair: critic | DeepSeek V4.1 Flash |

Each model is pinned to named providers on OpenRouter, without fallback to others (DeepSeek is pinned to two, and
may fall back between those two only). The analyst, checker and
tiebreaker come from different model families, to obtain separate answers; independence is not established, and
they can still share errors.

**Cost cap: at most $3 per job** (one company, one 10-K). A job also has limits on requests, tokens and time: 400
requests, 4.8 million tokens and 3,600 seconds for the whole job, and smaller limits for each step (for example 100
requests for the income statement, 100 for the cash flow statement, 150 for the balance sheet, and 100 for each
repair attempt). A step that reaches its own limit stops, keeps its full record and says which limit it reached;
the job goes on to its next statement, and that statement's outcome records the stop. A job that reaches a
whole-job limit stops.

The tiebreaker and the repair step can each be switched off on the Run page.

## Requirements

- **Windows and CPython 3.13.** The lock files hold the hashes of the Windows wheels for CPython 3.13, the only
  platform tested.
- **Two Python environments**, both installed from the lock files with their hashes:
  - `.venv` runs the SEC search server, the `edgar-search-mcp` 1.0.1 release wheel, which `requirements.lock`
    names by URL and hash and `mcp_pin.json` pins file by file.
  - `.venv-agents` runs everything else. `agents_env.json` lists its three lock files:
    - `requirements.lock`: the MCP client and its dependencies (`mcp` 2.3.0, `httpx` 0.28.1, `pydantic` 2.14.1 and
      others);
    - `requirements-langgraph-additions.lock`: `langgraph` 1.2.18 and its dependencies, used by the repair step;
    - `requirements-results-store-v1.lock`: `openpyxl` 3.1.5 and `et-xmlfile` 2.0.0, for the Excel workbook.

  Create both from the project folder and install them with hashes required:

  ```
  py -3.13 -m venv .venv
  py -3.13 -m venv .venv-agents
  .venv\Scripts\python.exe -m pip install --only-binary=:all: --require-hashes -r requirements.lock
  .venv-agents\Scripts\python.exe -m pip install --only-binary=:all: --require-hashes -r requirements.lock -r requirements-langgraph-additions.lock -r requirements-results-store-v1.lock
  ```

  The code checks at start that each environment matches its locks exactly and stops if it does not.
- **An OpenRouter API key.** Paste it on the Setup page and press *Save the key*: the server writes it to a `.env`
  file in the project folder, on this computer only. You can also write the line `OPENROUTER_API_KEY=<your key>` in
  that file yourself, or set the environment variable `OPENROUTER_API_KEY` before you start the server (a key in the
  environment is used first, and the page then refuses to save another). The key is never logged or shown, and git
  ignores `.env`. The Setup page tests it once, for free: it asks OpenRouter's own record of the key whether
  the key is accepted, without calling a model, and shows only accepted or refused and the key's own spending cap.
- **An SEC User-Agent.** The SEC asks automated clients to identify themselves with a name and a contact email.
  Type them on the Setup page and press *Save* (written to `.env` like the key), or set the environment variable
  `EDGAR_USER_AGENT` before you start the server, for example:

  ```
  EDGAR_USER_AGENT=Your Name your.email@example.com
  ```

  Without it the system refuses to fetch filings.
- **The US GAAP taxonomy** edition the filing uses. The Setup page, and choosing a filing, download a pinned edition
  (2022 to 2026) from xbrl.fasb.org when it is not on the computer.

This copy runs its models online through OpenRouter. It needs no local model server, and no git: in a folder that
is not a git checkout, the server records a fingerprint of its own source files at start and each job checks it.

## Run one company

1. **Start the UI server** from the project folder:

   ```
   .venv-agents\Scripts\python.exe -s -E ui\server.py
   ```

   `-s -E` keep your own Python settings out of the agents' environment. The server listens on 127.0.0.1 only, on
   port 8765 by default (add `--port 8767`, for example, for another), and prints the address to open in a browser
   (http://127.0.0.1:8765 by default). Ctrl+C stops it.
2. **Setup.** Check that every item the line-up needs is ready: the two Python environments, the SEC server, the
   OpenRouter key and the taxonomy. Each item that is not ready says how to fix it. The taxonomy is not counted
   as missing before your first filing: *Choose and fetch* (step 4) downloads the edition that filing uses.
3. **Template.** Choose *Use default* once to register the standard template (engine template v16) before the first
   job; on a fresh copy the Run page refuses a start until it is registered. *Start here* lists these first-job steps.
   Every page names it *Standard v16*. The page shows its sections, subtotals and checks, and each row's definition,
   read only: in this version the template is fixed.
4. **Companies and years.** Search by ticker or name, then choose one 10-K and press *Choose and fetch*. This
   downloads the filing's files from SEC EDGAR, and the US GAAP edition it uses when that is missing; the
   *Chosen filing* card says which. Any listed year can be chosen.
5. **Run.** Check the template and filing, choose which parts to run (tiebreaker, repair) and press *Start*. The
   paid line-up runs with its $3 ceiling, which the page shows. The cost and time shown beside it are the range of
   earlier finished jobs with this line-up, not a promise. Until the job can run, the page lists what is missing (a registered
   template, a chosen filing, the OpenRouter key) and *Start* stays off. The button reads *Starting…* until the job starts, then the Jobs page opens.
6. **Jobs (the live view).** Watch the job's progress: lines processed per statement, where each line went, the
   model requests, tokens and dollars per step. Press a step's request count to list its model calls (the
   corrections, each statement's spread, the repairs); each call opens with the instructions and message it was
   sent and the reply. *Replay* replays a statement line by line. *Stop* ends the job at its next check and keeps
   its records.
   The *Tiebreaker* and *Tree of thought* pages show the disputed lines and the repair attempts as they happen.
   *Evals* shows calls, tokens, time and cost per model, and the caps against what was used.
7. **Results (the workbook and the trail).** When the job has finished, press *Review the result* on the Jobs page
   (or open *Results*). The card at the top, *Workflow result — unreviewed*, makes one Excel workbook per finished
   job from the job's own records: press *Make the workbook*, then *Download*. The workbook is read back against the
   records it was made from before it is offered, and says on its *About* sheet that it is the workflow's result,
   unreviewed: no step of this build reviews, approves or signs off a spread; you review it. It is saved as
   `workflow_result_<CIK>_<accession>_<job start>.xlsx` where your browser saves downloads (usually the Downloads folder).
   Below the card is the results database: *Import* the job, type your name (it is recorded with each choice), and
   press *Make this count* for each statement, or *View* first to see it. The *Checks outcome* column says how the
   code checks ended. For each line you see the row it went to, the source fact of every amount, the lines that
   could not be placed and why, the open exceptions and the checks. A spread with *open claims* (a placed amount the
   checks cannot rule out also sitting in another line) can be made to count; its claims are listed under *Open
   questions*. Only a spread whose claims cannot be assessed at all is refused, with the reason. *Replay* shows how
   each line was classified.

The other pages: *Start here* explains how a job works, with a finished example and a glossary; *Advanced settings*
shows the default configuration, read only.

## Verification

This copy was checked with a fresh install from the lock files, the Setup page and one paid job run through the UI. The development test suite is not part of this copy. The dealership jobs under *Five car dealers* were run through the UI of earlier versions of this code.

## Five car dealers

The tested configuration was also run on five US car dealership groups, one FY2025 10-K each (fiscal year ended
December 31, 2025): Penske Automotive Group (PAG), Group 1 Automotive (GPI), AutoNation (AN), Lithia Motors (LAD) and
Sonic Automotive (SAH). CarMax is not used. Each job ran through the UI with engine template v15, settings v29, and
cost from $0.06 to $0.07. Penske was run again with settings v30, whose only change is the cash flow statement's
request limit, raised from 50 to 100 after the first Penske job's cash flow statement reached it; that rerun cost
$0.085.

| Company | Settings | Income statement | Balance sheet | Cash flow (D&A memo) | Workbook |
|---|---|---|---|---|---|
| Penske (PAG) | v29 | qualified | qualified | blocked (reached its request limit) | `workflow_result_1019849_0001628280-26-012830.xlsx` |
| Penske (PAG), rerun | v30 | qualified | qualified | complete | `workflow_result_1019849_0001628280-26-012830.xlsx` |
| Group 1 (GPI) | v29 | qualified | qualified | qualified | `workflow_result_1031203_0001031203-26-000064.xlsx` |
| AutoNation (AN) | v29 | blocked | qualified | blocked | `workflow_result_350698_0001628280-26-007800.xlsx` |
| AutoNation (AN), rerun | v30 | qualified | qualified | qualified | `workflow_result_350698_0001628280-26-007800_<start time>.xlsx` |
| Lithia (LAD) | v29 | qualified | qualified | qualified | `workflow_result_1023128_0001023128-26-000015.xlsx` |
| Sonic (SAH) | v29 | blocked | qualified | qualified | `workflow_result_1043509_0001628280-26-010570.xlsx` |
| Sonic (SAH), rerun | v30 | blocked (net income) | qualified | qualified | `workflow_result_1043509_0001628280-26-010570_<start time>.xlsx` |

AutoNation and Sonic were run again with settings v30. AutoNation: all
three statements qualified; 146 requests, $0.073, 7.7 minutes. Sonic: the income statement
is still blocked on net income (filed 118.7 million, built 226.9 million). The filed sign of the tax line contradicts
the printed statement; the workbook's Review sheet flags it and the filed value is kept. Its balance sheet and cash
flow statement qualified; 141 requests, $0.065, 6.5 minutes.

Each workbook from the first runs and the Penske rerun was read back against the records it was made from and
matched. Those were saved by earlier versions as `workflow_result_<CIK>_<accession>.xlsx`; the AutoNation and Sonic
reruns carry the job's start time in the name, as this build does. None of them is included in this copy. The outcomes are the code checks' own; no answer key covers these five filings, and no
person has yet checked examples from them.

## Outputs and where they go

| What | Where |
|---|---|
| Downloaded filings | `data/filings/<CIK>/<accession>/` |
| Downloaded taxonomy editions | `data/sources/` |
| Each job's full record: every model request and reply, cost, check and outcome, in a dated folder that is never overwritten | `runs/` (or the folder named by `SPREADING_RUNS_DIR`) |
| The UI's job requests and its own state | `data/ui/` (or `SPREADING_UI_DIR`) |
| The results store: every imported spread version, and which one is current | `data/results/spreads.sqlite` |
| Workflow result workbooks, each with the snapshot it came from and the read-back result | `runs/<date and time>_result_workbook/workflow_result_<CIK>_<accession>_<job start>.xlsx` |

The workbook ("Workflow result — unreviewed") has *About* (the job, filing, template and status), *Lines* (every
printed line with its proposed row, evidence and status, and the correction columns *Your row* and *Your signed
amount*), *Template* (the rows, recalculated from your corrections), *Checks* (each check as recorded, with an
*Assessment* column), *Review* (the lines and checks to look at first) and *Not spread* (the inputs the template
leaves out). Edits made in Excel do not flow back to the system.

## Limits

- **Non-financial companies only.** Banks, insurers and real estate companies (SEC industry codes 6000 to 6799) are
  removed before any company list. The template is built for non-financial statements.
- **Tagged filings only.** The system reads the statements from inline XBRL. A filing without tagged statements is
  out of scope.
- **Evaluation uses approved labels, not ground truth.** Results are graded against answer keys drafted by Claude
  Opus and checked and approved by a person. The keys are labels, and they can be wrong. The answer keys, the tools that draft them and the offline evaluation tools are not part of this copy. The
  in-job grader (`spreading/grade.py`) is included; it runs only when the job's settings name a reference, and the
  default settings name none.
- **Agreement is not correctness.** When the analyst and the checker agree and the arithmetic checks pass, the models
  can still share the same wrong reading. Code checks that a quoted passage was really shown to the model, not that
  it supports the chosen row.
- **Model calls are not deterministic.** Two runs of the same filing with the same settings can place some lines
  differently and end with different sections blocked.
- **A spread is not always complete.** Extraction and placement can both fail. Blocked sections and unplaced lines
  stay visible in the results and the workbook.
- **Open claims are not settled.** A spread with open claims is kept with them listed; the workbook's *Review*
  sheet lists them, and nothing in this build settles or approves them.
- **No review or sign-off step.** The workbook is the workflow's result, unreviewed; you review it.
- **Statements only.** The models do not read the notes or the rest of the filing. A line that only the notes
  would settle stays unresolved.
- **Hosted models and cost.** Every job of this configuration calls paid models on OpenRouter, up to $3 per
  job. Model names, prices and availability on OpenRouter can change.

## Documents

One-page explanations, as PDFs in `docs/`:

- `workflow_overview.pdf`: one 10-K, from the template to the unreviewed workbook.
- `architecture_today.pdf`: how it works, with the template file at the centre.
- `code_map.pdf`: each stage of a job and the code that does it.
- `step_2_3_choose.pdf`: choosing the company, its 10-K and the period.
- `step_4_run.pdf`: what a job does.
- `sorting_lines.pdf`: how the organizer sorts lines into sections and code checks the totals.
- `step_5_results.pdf`: what is stored and what the analyst downloads.
- `decision_diagrams.pdf`: every decision in one job, who makes it, and what happens on each outcome.
- `ui.pdf`: what each page of the user interface does.
- `system_prompt.pdf`: the written instructions every model that places a line reads.
- `tiebreaker.pdf`: how a third model settles the lines the two models dispute.
- `evaluation_and_monitoring.pdf`: what a job records, what you watch, and how a spread is judged.
- `tree_of_thought.pdf`: the optional repair of a section that fails its checks (off by default).

`docs/code_rules.md` lists the places where the code itself decides something in a default job, with the file and
lines of each rule.

## Licence

MIT. See `LICENSE`.
