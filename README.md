# t2-3po

> *"Sir, the possibility of this tech allocation overconcentrating your ISA portfolio is
> approximately 3,720 to 1!"*

An asynchronous, multi-agent advisory engine for `t2-d2`, the
Trading212 execution engine. t2-3po reviews investment theses and writes recommendations for a human
to approve. It can't place orders: its Trading212 client has no way to send anything but a GET.

## Part of the T2 system

This repo is one of the T2 droids, coordinated from the `t2-hq` repo, which holds the system
architecture (`ARCHITECTURE.md`) and plan (`PLAN.md`). All the T2 repos live under the same
GitHub owner as this one.

| Repo | Role |
|---|---|
| `t2-d2` | buys dips; places buy orders on its own timers |
| `t2-k2` | harvests gains from overbought holdings; places sell orders |
| `t2-bb8` | harvests the Trading212 instrument list; metadata only |
| **t2-3po** (this repo) | advisory council; reviews theses, strategy and configuration; read-only |
| `t2-hq` | hub: plan, architecture, and the orchestrator and Telegram interface (planned) |

t2-3po is the advisory layer. It reads the other droids' configuration and run reports and suggests changes, such as moving a stock in t2-d2 from `mild` to `moderate` or adding a holding. It never edits their files and never places orders. It runs on its own and doesn't need the other repos to be running.

## How it works

1. **Data protocol:** reads a thesis, then fetches portfolio weights from Trading212 and fundamentals
   from Yahoo Finance. Concentration limits are checked in code.
2. **Advisory council:** the Bull and Bear analysts argue in parallel, then the Quantitative Analyst
   checks the thesis and both cases against the figures. Any long documents are digested first.
3. **Protocol officer:** synthesises a verdict and proposed allocations. Code then checks those
   proposals against the limits and the actual portfolio, and adds warnings for the reviewer.

If an evaluation fails at any stage, no proposal is written for that thesis.

t2-3po also advises on how the trading droids are configured; see
[Advising on t2-d2 and t2-k2](#advising-on-t2-d2-and-t2-k2).

## Quick start

Requires [uv](https://github.com/astral-sh/uv).

```bash
uv sync
cp .env.example .env    # then fill in your keys

# Write a thesis
cp theses/NVDA.example.md theses/NVDA.md

# Run the full evaluation and print the report, without writing reports or proposals.
# This makes the same model calls as a normal run.
uv run t2-3po evaluate theses/NVDA.md --dry-run

# Run the evaluation and write the report and proposal
uv run t2-3po evaluate theses/NVDA.md

# Or every thesis in theses/ (*.example.md files are skipped)
uv run t2-3po evaluate
```

Each evaluation makes four model calls, plus one digest call per referenced document, and logs the
token usage of each.

### Choosing a model

Both options are free. `MODEL_PROVIDER` picks one.

**Gemini on Google's free tier** (`MODEL_PROVIDER=google`, the default). Get a key from
[Google AI Studio](https://aistudio.google.com/apikey) and set `GOOGLE_API_KEY`. `GEMINI_MODEL`
defaults to `gemini-3.7-flash`. Google retires model names, and the `-latest` aliases aren't
offered to every key, so if a run fails with a 404 for the model, list the Flash models your key
can use and set `GEMINI_MODEL` to one of them:

```bash
uv run t2-3po models
```

- **Rate limits.** The free tier allows only a few requests a minute and a daily quota. t2-3po starts
  requests at least `GEMINI_MIN_INTERVAL_SECONDS` apart (7s by default, so under 10 a minute), even
  when agents run in parallel, and retries a rate-limited or failed request up to four more times,
  waiting 15s, then longer, up to 2 minutes. A `review` of 18 stocks takes about two and a half
  minutes. If you still see 429 errors, raise the interval; if the daily quota runs out, the run fails
  and can be repeated the next day. The free daily quota can be as low as 20 requests per model, and a
  `review` makes one per stock plus one for the overview, so use `--batch` for more than a few stocks.
- **Privacy.** Google may use free-tier prompts and responses to improve its products. Prompts
  contain your thesis, the stocks and your portfolio weights; with a portfolio they also contain
  position values. Use `--no-portfolio` or Ollama to keep those private.

**A local model served by [Ollama](https://ollama.com)** (`MODEL_PROVIDER=ollama`). Nothing leaves the
machine, and there are no rate limits:

```bash
ollama pull qwen3:4b
# Ollama's default context window is too small for the council's prompts
OLLAMA_CONTEXT_LENGTH=16384 ollama serve

MODEL_PROVIDER=ollama uv run t2-3po evaluate theses/NVDA.md --dry-run
```

Set `OLLAMA_MODEL` to use another model, for example `qwen3:1.7b` on a machine with little memory. A
small local model reasons much less well than Gemini Flash and is more likely to fail its output
schema, so treat its verdicts as a rough first pass.

Add `--no-portfolio` to run without Trading212 credentials. Concentration isn't checked in that mode,
and the proposal says so.

## Reviewing your stocks against a macro thesis

`review` profiles every traded stock in t2-d2's `config/stocks.yaml` as a vehicle for one macro
thesis, on the horizon the thesis states, then looks across all of them together. The watchlist is
left out.

```bash
# Try two stocks first
uv run t2-3po review --only RIO HOOD --dry-run

# Review every stock, then write the report and a proposal per stock
uv run t2-3po review

# Review only the 5 stocks reviewed least recently (what the daily timer runs)
uv run t2-3po review --batch 5
```

`--batch N` spreads the stocks over several days, so each run stays inside Gemini's free-tier daily
quota. It reads this droid's earlier run reports to find when each stock was last reviewed, then
takes stocks never reviewed first and the oldest after that, in file order on a tie. A stock whose
review failed isn't recorded as reviewed, so it's picked again the next day. Dry runs don't count.
The overview then covers that day's batch.

The thesis is plain markdown with no front matter, by default at `theses/macro/core.md`. Name its
pillars, what would count as serving each one, and what would change your mind.

For each stock, the Vehicle Analyst decides:

- **Fit:** which pillars it serves and how much of the business does, as a verdict from `VALIDATED`
  to `REJECTED`. A stock that serves no pillar is `REJECTED`, which means it needs its own reason to be
  held rather than that it's a bad company. The analyst is told not to propose selling for that alone.
- **Role:** `GROWTH` if shareholders benefit as the business grows with the theme, `STABILITY` if they
  benefit from steady, cash-covered earnings and dividends, or `NEITHER`.
- **Key metrics:** the four to eight figures that decide the role, such as revenue CAGR, cash runway
  and break-even revenue for growth, or years profitable and dividend cover after capex for stability.
- **Outlook:** what has to happen over the horizon, and what would show it isn't happening.
- **Proposal:** a target weight and action, and a t2-d2 strategy change where the verdict justifies one.

Alongside Yahoo's snapshot, the analyst gets four or five years of annual revenue, net income and free
cash flow. Revenue CAGR and the counts of profitable and cash-positive years are computed in code.

The Portfolio Reviewer then reads every profile and reports how well each pillar is covered, the
balance between growth and stability vehicles, overlaps where stocks share a driver, and gaps.

That's one model call per stock plus one more. Up to five stocks are profiled at a time, but on
Gemini the requests are still spaced out to respect the free tier's limits. Each stock is sized on its
own, so the report says what the target weights add up to. For a full Bull, Bear and Quant debate on
one stock, write a single-stock thesis and use `evaluate`.

## Advising on t2-d2 and t2-k2

`advise` reviews how the trading droids are configured and suggests changes: move a t2-d2 stock to
another mode, add a stock to t2-d2, move one to the watchlist, or change one of t2-d2's or t2-k2's
thresholds. It never edits their files. You apply a suggestion yourself, then record your answer.

```bash
# Print the advice without recording anything
uv run t2-3po advise --dry-run

# Record the advice: var/suggestions/YYYY-MM-DD_advice.json and reports/YYYY-MM-DD_advice.md
uv run t2-3po advise
uv run t2-3po advise --only k2           # one droid only: d2 or k2

# Open suggestions, and your answers
uv run t2-3po suggestions
uv run t2-3po suggestions accept e3a79f4f --note "applied in PR #31"
uv run t2-3po suggestions reject 1b2c3d4e --note "TSLA stays on moderate"
uv run t2-3po suggestions --all
```

How it works:

1. **Code gathers the figures.** It reads t2-d2's `config/stocks.yaml` and `config/strategy.yaml`
   and t2-k2's `config/strategy.yaml`, from the sibling directories (`D2_DIR`, `K2_DIR`). For each
   t2-d2 stock it fetches a year of prices and computes the return, volatility, distance from the
   1-year high, the 200-day trend, and **dip days by mode**: how often each mode's price rule
   would have fired in the last year. It tallies what t2-d2 actually did over the last 30 days of
   its run reports (evaluations, buy signals, buys, top skip reasons) and reads t2-k2's last run.
   With a portfolio, it adds position weights, never account values.
2. **Two model calls.** The Strategy Analyst drafts up to eight suggestions from those figures; the
   Risk Reviewer then challenges each one and sets its confidence.
3. **Code checks every draft** and drops what fails, saying why in the report:
   - a mode must exist, and a stock must be in `stocks.yaml`;
   - a new holding must match exactly one tradable instrument in t2-bb8's instrument list
     (`BB8_DIR`) in the stated market, not already be listed, and have a year of Yahoo prices;
   - a threshold must be one of the adjustable keys, keep its sign and change by at most 50% at a
     time (`overbought.min_signals` must be 1 to 3, `harvest.gain_fraction` above 0 and at most 1);
   - a suggestion you rejected in the last 90 days, or already accepted, isn't made again. One
     that is still open is marked as a repeat.

   At most five survive (`MAX_SUGGESTIONS`), highest confidence first.

Each suggestion has a stable id: the same change suggested again keeps the same id. Answers are kept
in `var/suggestions/decisions.json`.

## Output

| File | Contents |
|---|---|
| `reports/YYYY-MM-DD_ticker.md` | `evaluate`: verdict, critique, risks, each analyst's case and the claim checks |
| `reports/YYYY-MM-DD_macro_review.md` | `review`: the overview, a comparison table, then each stock's profile and history |
| `proposals/proposal_YYYYMMDD.json` | The machine-readable proposal, one per thesis or stock |

A later run on the same day adds `_2`, `_3` and so on rather than overwriting. These directories are
gitignored, along with single-stock theses, because they contain personal financial data. The macro
thesis in `theses/macro/` is tracked, so keep positions and amounts out of it.

```json
{
  "timestamp": "2026-11-01T20:00:00Z",
  "thesis_status": "VALIDATED_WITH_RISKS",
  "portfolio_warnings": ["Technology sector concentration 35.0% exceeds 30% ISA limit threshold."],
  "proposed_allocations": [
    {
      "ticker": "NVDA",
      "target_weight": 0.08,
      "action": "TRIM",
      "reasoning": "P/E expanded past target threshold; trimming to lock in gains."
    }
  ],
  "suggested_config_updates": [
    { "ticker": "NVDA", "strategy": "mild", "reasoning": "Slow dip-buying while overweight." }
  ],
  "ticker": "NVDA",
  "thesis_file": "NVDA.md"
}
```

`thesis_status` is `VALIDATED`, `VALIDATED_WITH_RISKS`, `NEEDS_REVISION` or `REJECTED`. `action` is
`BUY`, `HOLD`, `TRIM` or `SELL`.

## Run reports, exit codes and locks

t2-3po follows the T2 droid contract (t2-hq `CONTRACT.md`):

- **Run report.** Every run writes a JSON report to `var/runs/<UTC timestamp>-<command>.json`
  (or `$T2_RUNS_DIR`). It records the version, command, start and end time, dry-run flag, status,
  exit code, a one-line summary and errors. Its `result` lists each thesis or stock with its verdict and the files written. `evaluate` and `review` fail with `1` if any thesis or stock fails, or the portfolio can't be fetched; a missing key or thesis is `2`. `export` writes a report too, and `--dry-run` still writes the run report. `--report PATH` writes it somewhere else.
- **Exit codes:** `0` success, `1` the run failed, `2` configuration or usage error, `3` skipped
  (another evaluation or review is running, or the Trading212 lock is held).
- **One run at a time.** A second run while one is in progress exits `3` at once.
- **The Trading212 account is shared.** Before fetching the portfolio a run waits for the account lock,
  which every droid takes, so droids on independent timers don't trip the per-account rate limits.
  If it is still held after 10 minutes (`T2_T212_LOCK_TIMEOUT`, in seconds), the run exits `3`.

Locks live in `/run/user/<uid>/t2` (or `$T2_LOCK_DIR`).

```bash
uv run t2-3po review --only RIO --dry-run
ls var/runs/
```

## Handing over to t2-d2

After reviewing a report, export its proposal:

```bash
uv run t2-3po export proposals/proposal_20261101.json --approve
```

This writes a copy to `exports/` with an `approved_at` stamp. Set `EXPORT_DIR` to change where it goes.

t2-d2 doesn't read these files yet, and export doesn't change its config, so for now apply
`suggested_config_updates` to t2-d2's `config/stocks.yaml` by hand. t2-d2 also only places buy orders,
so `TRIM` and `SELL` proposals have to be carried out manually.

## Writing a thesis

A thesis is a markdown file with YAML front matter. See
[`theses/NVDA.example.md`](theses/NVDA.example.md).

| Field | Required | Description |
|---|---|---|
| `ticker` | Yes | Name for the ticker, used in filenames and proposals |
| `t212_ticker` | No | Trading212 code, as in t2-d2's `stocks.yaml`, e.g. `RIOl_EQ` |
| `yahoo_ticker` | No | Yahoo symbol, for when it can't be derived from `t212_ticker` |
| `documents` | No | UTF-8 text files, relative to the thesis, e.g. transcripts or 10-Ks |

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `T212_API_KEY`, `T212_API_SECRET` | | Trading212 credentials. Use a key with read-only permissions |
| `T212_BASE_URL` | `https://live.trading212.com` | Or `https://demo.trading212.com` |
| `MODEL_PROVIDER` | `google` | `google` (Gemini free tier) or `ollama` (local). See [Choosing a model](#choosing-a-model) |
| `GOOGLE_API_KEY` | | Required with `MODEL_PROVIDER=google` for `evaluate` and `review`, including with `--dry-run` |
| `MACRO_THESIS_FILE` | `theses/macro/core.md` | The macro thesis for `review` |
| `STOCKS_FILE` | `../t2-d2/config/stocks.yaml` | The stocks for `review` |
| `D2_DIR`, `K2_DIR`, `BB8_DIR` | `../t2-d2`, `../t2-k2`, `../t2-bb8` | The other droids, for `advise`: their `config/`, `var/runs/` and t2-bb8's instrument list |
| `SUGGESTIONS_DIR` | `var/suggestions` | Advice runs and your answers |
| `ADVISE_LOOKBACK_DAYS` | `30` | Days of run reports the advice considers |
| `MAX_SUGGESTIONS` | `5` | Most suggestions kept per run |
| `REJECTED_MEMORY_DAYS` | `90` | How long a rejected suggestion isn't repeated |
| `GEMINI_MODEL` | `gemini-3.7-flash` | Gemini model for every agent, including document digests |
| `GEMINI_MIN_INTERVAL_SECONDS` | `7` | Minimum gap between the starts of two Gemini requests |
| `OLLAMA_MODEL` | `qwen3:4b` | Local model, with `MODEL_PROVIDER=ollama` |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama server |
| `MAX_SECTOR_WEIGHT` | `0.30` | Sector limit, as a share of total ISA value |
| `MAX_POSITION_WEIGHT` | `0.175` | Position limit. The default is t2-d2's highest per-mode cap |
| `THESES_DIR`, `REPORTS_DIR`, `PROPOSALS_DIR`, `EXPORT_DIR` | `theses`, `reports`, `proposals`, `exports` | Directories |
| `T2_RUNS_DIR` | `var/runs` | Where run reports go |
| `T2_LOCK_DIR` | `/run/user/<uid>/t2` | Where the run and Trading212 locks live |
| `T2_T212_LOCK_TIMEOUT` | `600` | Seconds to wait for the shared Trading212 lock before skipping |
| `LOG_LEVEL` | `INFO` | Log level |

Yahoo Finance data comes from `yfinance`, which needs no key.

## Deployment

t2-3po is deployed with t2-hq's `bin/t2-deploy`, which checks out a release tag in
`$T2_APPS/t2-3po`, builds its virtual environment and installs the systemd user units in
`deploy/systemd/`:

```bash
# From the t2-hq checkout
bin/t2-deploy release t2-3po v0.2.0   # tag a clean, tested main and push the tag
bin/t2-deploy deploy t2-3po           # deploy the newest tag
bin/t2-deploy status
```

The units run `t2-3po review --batch 5` every day at 07:00 UK time (`t2-3po.timer`), so about 25
stocks are each reviewed once a week, and `t2-3po advise` on Sundays at 08:00
(`t2-3po-advise.timer`), an hour after the review and two after T2-BB8's harvest. Read the advice in
`$T2_APPS/t2-3po/reports/` and answer it with `t2-3po suggestions`, run from that directory. A run missed while the machine was off or asleep runs at the next start-up. It
reads the stocks from the deployed T2-D2 (`../t2-d2/config/stocks.yaml`), so deploy that first,
and its `.env` needs `GOOGLE_API_KEY` (or `MODEL_PROVIDER=ollama`).

The deploy never creates, reads or edits `.env`: write `$T2_APPS/t2-3po/.env` from `.env.example`
first, or the deploy stops before enabling anything.

## Development

```bash
uv run pytest         # no network or model calls
uv run ruff check .
uv run ruff format .
uv run mypy
```

### Pre-commit hooks

```bash
uv sync
uv run pre-commit install     # once after cloning
uv run pre-commit run --all-files
```

Hooks run on every commit: `detect-secrets`, `ruff`, `bandit`, `semgrep`, `pip-audit`, `mypy` and
`pytest`, plus a branch name check. `semgrep` and `pip-audit` need network access. Any finding blocks
the commit.

Commit on `feature/<name>`, `bug/<name>`, `task/<name>` or `hotfix/<name>`, with `<name>` in
lowercase-with-hyphens. The hook rejects commits on any other branch, including `main`.

If `detect-secrets` flags a false positive, update the baseline:

```bash
uvx detect-secrets==1.5.0 scan --baseline .secrets.baseline
```

If `pip-audit` flags a vulnerable dependency, upgrade it. `uv` ignores versions published in the last
7 days, so a very recent fix may need a short wait.

```bash
uv lock --upgrade-package <package>
uv sync
```

### CI

| Workflow | Runs | Checks |
|---|---|---|
| `ci.yml` | Pushes and pull requests to `main` | `ruff`, `mypy`, `pytest` |
| `branch-name.yml` | Pull requests to `main` | Branch naming |
| `dependency-audit.yml` | Daily, or `gh workflow run dependency-audit.yml` | `pip-audit` |

Dependabot checks weekly for newer GitHub Actions versions.

## Known limitations

- **Historical valuation:** Yahoo's snapshot has no historical multiples, so the Quantitative Analyst
  marks claims that depend on them as `UNVERIFIABLE`.
- **Sector limits:** ETFs have no sector, so they're excluded from sector checks. The warnings report
  how much of the ISA that covers.
- **Exchanges:** Trading212 codes map to Yahoo symbols for US, London, Xetra, Amsterdam and Paris
  listings. For other holdings, the sector is reported as unknown.
- **Renamed tickers:** a Trading212 code can outlive a rename, such as Rocket Lab's `VACQ_US_EQ`.
  `review` looks these up by symbol, but portfolio sector checks still report the sector as unknown.
- **Revenue breakdown:** Yahoo has no revenue by segment, so `review` often can't confirm how much of
  a mixed business serves a pillar.
- **Yahoo Finance:** `yfinance` is an unofficial API and can break or return incomplete data.

## Disclaimer

For personal educational use. This is model-generated analysis, not financial advice, and it can be
wrong. Review every recommendation before acting on it.
