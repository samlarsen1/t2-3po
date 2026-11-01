# t2-3po

> *"Sir, the possibility of this tech allocation overconcentrating your ISA portfolio is
> approximately 3,720 to 1!"*

An asynchronous, multi-agent advisory engine for [t2-d2](https://github.com/samlarsen1/t2-d2), the
Trading212 execution engine. t2-3po reviews investment theses and writes recommendations for a human
to approve. It can't place orders: its Trading212 client has no way to send anything but a GET.

## How it works

1. **Data protocol:** reads a thesis, then fetches portfolio weights from Trading212 and fundamentals
   from Yahoo Finance. Concentration limits are checked in code.
2. **Advisory council:** the Bull and Bear analysts argue in parallel, then the Quantitative Analyst
   checks the thesis and both cases against the figures. Any long documents are digested first.
3. **Protocol officer:** synthesises a verdict and proposed allocations. Code then checks those
   proposals against the limits and the actual portfolio, and adds warnings for the reviewer.

If an evaluation fails at any stage, no proposal is written for that thesis.

## Quick start

Requires [uv](https://github.com/astral-sh/uv).

```bash
uv sync
cp .env.example .env    # then fill in your keys

# Write a thesis
cp theses/NVDA.example.md theses/NVDA.md

# Run the full evaluation and print the report, without writing any files.
# This makes the same paid model calls as a normal run.
uv run t2-3po evaluate theses/NVDA.md --dry-run

# Run the evaluation and write the report and proposal
uv run t2-3po evaluate theses/NVDA.md

# Or every thesis in theses/ (*.example.md files are skipped)
uv run t2-3po evaluate
```

Each evaluation makes four Claude calls, plus one digest call per referenced document, and logs the
token usage of each.

Add `--no-portfolio` to run without Trading212 credentials. Concentration isn't checked in that mode,
and the proposal says so.

## Output

| File | Contents |
|---|---|
| `reports/YYYY-MM-DD_ticker.md` | Verdict, critique, risks, each analyst's case and the claim checks |
| `proposals/proposal_YYYYMMDD.json` | The machine-readable proposal |

A later run on the same day adds `_2`, `_3` and so on rather than overwriting. These directories are
gitignored, along with your theses, because they contain personal financial data.

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
| `ANTHROPIC_API_KEY` | | Required for `evaluate`, including with `--dry-run` |
| `CLAUDE_MODEL` | `claude-sonnet-5-5` | Model for every agent, including document digests |
| `DIGEST_CACHE_TTL` | `5m` | Prompt cache lifetime for document digests: `5m`, `1h` or `off`. See [Prompt caching](#prompt-caching) |
| `MAX_SECTOR_WEIGHT` | `0.30` | Sector limit, as a share of total ISA value |
| `MAX_POSITION_WEIGHT` | `0.175` | Position limit. The default is t2-d2's highest per-mode cap |
| `THESES_DIR`, `REPORTS_DIR`, `PROPOSALS_DIR`, `EXPORT_DIR` | `theses`, `reports`, `proposals`, `exports` | Directories |
| `LOG_LEVEL` | `INFO` | Log level |

Yahoo Finance data comes from `yfinance`, which needs no key.

## Development

```bash
uv run pytest         # no network or API spend
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

## Prompt caching

Only document digests use prompt caching, so rerunning on the same document can reuse it even if the
thesis has changed. Each agent's log line reports `usage_cache_read` and `usage_cache_write`.

The council isn't cached. On a measured run, output made up 80% of the cost, and caching its small
input saved nothing. Agents with different output schemas also can't share a cache.

For Sonnet 5.5, a cache read costs 0.05× the normal input price. A write costs 1.25× with `5m` or 2×
with `1h`, and a write that's never read back is a surcharge:

| `DIGEST_CACHE_TTL` | Best for |
|---|---|
| `5m` | The default. Pays off if the document is read back at least once within 5 minutes |
| `1h` | Repeated reruns with longer gaps. Needs at least two reads to pay off |
| `off` | One-off digests, to avoid the write surcharge |

## Known limitations

- **Historical valuation:** Yahoo's snapshot has no historical multiples, so the Quantitative Analyst
  marks claims that depend on them as `UNVERIFIABLE`.
- **Sector limits:** ETFs have no sector, so they're excluded from sector checks. The warnings report
  how much of the ISA that covers.
- **Exchanges:** Trading212 codes map to Yahoo symbols for US, London, Xetra, Amsterdam and Paris
  listings. For other holdings, the sector is reported as unknown.
- **Yahoo Finance:** `yfinance` is an unofficial API and can break or return incomplete data.

## Disclaimer

For personal educational use. This is model-generated analysis, not financial advice, and it can be
wrong. Review every recommendation before acting on it.
