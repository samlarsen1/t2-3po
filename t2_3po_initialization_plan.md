# `t2-3po` — Strategic & Protocol Advisory Droid for `t2-d2`

> *"Sir, the possibility of this tech allocation overconcentrating your ISA portfolio is approximately 3,720 to 1!"*

## 1. Overview & Purpose

`t2-3po` is an asynchronous, multi-agent strategic advisory engine built to complement `t2-d2` (the Trading212 execution engine).

While `t2-d2` handles deterministic execution, portfolio state polling, and order placement, `t2-3po` acts as the probabilistic intelligence layer. It analyzes investment theses, runs multi-perspective debates (Bull, Bear/Devil's Advocate, and Quantitative), cross-checks macro/market data, evaluates Trading212 ISA concentration risks, and outputs structured strategy recommendations for human approval.

## 2. Architectural Boundaries & System Flow

```
                      ┌────────────────────────────────────────┐
                      │                t2-3po                  │
                      │   (Strategic Advisory Microservice)    │
                      └──────────────────┬─────────────────────┘
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        ▼                                ▼                                ▼
┌──────────────────┐           ┌──────────────────┐           ┌──────────────────┐
│  Trading212 ISA  │           │   Yahoo Finance  │           │   Multi-Agent    │
│ (Read Positions) │           │ (Market Metrics) │           │  Review Engine   │
└──────────────────┘           └──────────────────┘           └──────────────────┘
                                         │
                                         ▼ (Generates Proposal)
                             ┌────────────────────────┐
                             │  strategy_proposal.json│
                             │  / Thesis Review (.md) │
                             └───────────┬────────────┘
                                         │
                                         ▼ (Human-in-the-Loop Review)
                      ┌────────────────────────────────────────┐
                      │                 t2-d2                  │
                      │     (Trading Execution Engine)         │
                      └────────────────────────────────────────┘
```

## 3. Core Multi-Agent Architecture

`t2-3po` runs a structured 3-stage multi-agent orchestration pipeline:

1. **The Data Protocol (Fetcher):**
   * Reads target tickers and investment thesis files from `./theses/*.md`.
   * Queries Trading212 API (read-only) for current portfolio weightings, cash balance, and ISA constraints.
   * Calls Yahoo Finance / Market APIs for fundamentals (P/E, Debt/Equity, FCF Yield, forward consensus).

2. **The Advisory Council (Debate):**
   * **Bear Analyst / Devil's Advocate:** Focuses strictly on downside risks, thesis fallacies, valuation traps, and macro headwinds.
   * **Bull Analyst:** Evaluates TAM expansion, earnings catalysts, and growth momentum.
   * **Quantitative Analyst:** Cross-checks claims against financial metrics, historical valuation multiples, and balance sheet strength.

3. **Lead Orchestrator (Protocol Officer):**
   * Synthesizes agent debates into a clear, structured critique.
   * Evaluates ISA portfolio concentration risk.
   * Outputs an actionable recommendation with suggested config updates for `t2-d2`.

## 4. Tech Stack & Integration

* **Repository:** `samlarsen1/t2-3po`
* **Language/Framework:** Python 3.11+ using `PydanticAI` / `LangGraph`
* **LLM Engine:**
  * Anthropic Claude 3.5/3.7 Sonnet (Primary reasoning, thesis synthesis, advice)
  * Google Gemini 2.0 Flash / 1.5 Pro (Large context parsing for earnings transcripts & 10-Ks)
* **APIs & Credentials:**
  * `T212_API_KEY`: Read-only access to Trading212 ISA
  * `YFINANCE_API_KEY` / `yfinance`: Market fundamental data
  * `ANTHROPIC_API_KEY` & `GEMINI_API_KEY`: Model inference

## 5. Output Specifications

`t2-3po` generates two primary artifacts on every evaluation run:

1. **Human Review Report (`./reports/YYYY-MM-DD_ticker.md`):** Deep-dive text critique, key risk highlights, and strategic reasoning.
2. **Machine Structured Proposal (`./proposals/proposal_YYYYMMDD.json`):**
   ```json
   {
     "timestamp": "2026-11-01T20:00:00Z",
     "thesis_status": "VALIDATED_WITH_RISKS",
     "portfolio_warnings": [
       "Tech sector concentration exceeds 30% ISA limit threshold."
     ],
     "proposed_allocations": [
       {
         "ticker": "NVDA",
         "target_weight": 0.08,
         "action": "TRIM",
         "reasoning": "P/E expanded past target threshold; trimming to lock in gains."
       }
     ]
   }
   ```

## 6. Development Roadmap

* [ ] **Phase 1: Environment & API Connectors**
  * Set up `.env` and configuration management.
  * Implement read-only Trading212 API client and Yahoo Finance tool helper functions.
* [ ] **Phase 2: Multi-Agent Prompt Pipeline**
  * Implement Bear, Bull, and Quantitative sub-agent prompts.
  * Wire orchestration logic using Claude Sonnet / Gemini.
* [ ] **Phase 3: Thesis Parser & Report Generator**
  * Build markdown thesis reader and structured JSON proposal generator.
* [ ] **Phase 4: Integration with `t2-d2`**
  * Establish proposal export directory or Git PR workflow for `t2-d2` config sync.