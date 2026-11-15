# t2-3po

## System context

Part of the T2 droid system. The hub repo is `../t2-hq` (https://github.com/samlarsen1/t2-hq):
read its `ARCHITECTURE.md` and `PLAN.md` before changing how this repo interacts with the others.
Repos sit side by side under `~/dev/git/samlarsen1/t2/` (development) and `~/dev/apps/` (deployed).

- `../t2-d2`: buys dips; places buy orders on its own timers
- `../t2-k2`: harvests gains from overbought holdings; places sell orders
- `../t2-bb8`: harvests the Trading212 instrument list; metadata only
- `../t2-hq`: hub; plan, architecture, orchestrator and Telegram interface (planned)

t2-3po is the advisory layer. It reads the other droids' configuration and run reports and suggests changes, such as moving a stock in t2-d2 from `mild` to `moderate` or adding a holding. It never edits their files and never places orders.
This repo must keep working standalone: never import a sibling repo's code. Changes to the
shared droid contract (run reports, exit codes, locks) are agreed in `t2-hq` first.

## Rules

- **Read-only.** Never add a Trading212 call other than GET, and never edit another droid's files.
  Suggestions are written as reports and proposals for Sam to act on.
- **Free models only.** The council runs on Gemini's free tier or a local Ollama model
  (`MODEL_PROVIDER=google|ollama`). Don't add a paid provider. Keep Gemini requests paced and
  retried (`pacing.py`, `GEMINI_RETRY` in `agents.py`) so a run stays inside the free tier's limits.
- **Never create, read or edit `.env`.** Document variables in `.env.example` with placeholders.
- Tests make no network or model calls: use mock transports or `FunctionModel`.
- Commit only on `feature/<name>`, `bug/<name>`, `task/<name>` or `hotfix/<name>` branches.
