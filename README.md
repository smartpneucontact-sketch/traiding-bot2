# combo-v2-bot

Live paper-trading bot running the **combo_v2** strategy — a 3-signal momentum blend that backtested at **4.96 %/mo, Sharpe 1.06, Calmar 0.87** over 10 years (2016-04 → 2026-03, 1,040 US stocks, 5 bp/side TC, 2.0x leverage). Best single result across 45 strategies tested in the [`/Traiding 11/`](../Traiding%2011/REPORT.md) research repo.

> **This is a paper-trading bot.** It places orders against an Alpaca paper account. Do not point it at a live brokerage without changing the code path and acknowledging the -65 % historical max drawdown.

---

## What it does

Each weekday at 13:35 UTC (09:35 ET, 5 min after market open) the bot:

1. Downloads ~1 yr of daily OHLCV bars for the full US large/mid-cap universe + ~20 macro/sector ETFs via yfinance.
2. Computes the **combo_v2** target weights:
   - 1/3 [`xs_momentum_top30`](core/combo_strategy.py) — long top-30 stocks by 12-1 month return
   - 1/3 [`dual_momentum_voltarget`](core/combo_strategy.py) — top-30 by 6-mo abs-momentum, inverse-vol, target 15 % vol
   - 1/3 [`adaptive_voltarget_momentum`](core/combo_strategy.py) — xs_momentum with VIX-percentile leverage 0.5x / 1.0x / 1.5x
   - SPY drawdown gate ramps exposure to 0 when SPY is ≥18 % below its 60-day high
3. Multiplies by `target_leverage` (default 2.0x) and submits the rebalance to Alpaca.
4. The runner's pre-flight auto-scales allocations to fit `buying_power` so 2.0x always succeeds on standard Alpaca paper accounts (≈ 2.37x bp).
5. Logs the trade to `data/state/` and exposes the result via `/v2`.

---

## Deploy to Railway

### 1. Push to GitHub
```bash
cd combo-v2-bot
git init
git add .
git commit -m "Initial combo_v2 bot"
gh repo create smartpneucontact-sketch/combo-v2-bot --public --source=. --push
# or manually: git remote add origin git@github.com:smartpneucontact-sketch/combo-v2-bot.git && git push -u origin main
```

### 2. Create a new Railway service
- New Project → Deploy from GitHub repo → pick `combo-v2-bot`
- Add a **Volume** mounted at `/app/data` (state, logs, model_config.json persist here)
- Add env vars (copy from `.env.example`):
  - `MODEL_COMBO_V2_ALPACA_KEY` — your Alpaca paper key
  - `MODEL_COMBO_V2_ALPACA_SECRET` — your Alpaca paper secret
  - `DASHBOARD_AUTH_TOKEN` — long random string for dashboard + internal-API protection (**required**; without it the internal `/api` is unauthenticated). The env var is `DASHBOARD_AUTH_TOKEN`, not `DASHBOARD_ADMIN_TOKEN`.

### 3. After first deploy
- Visit `https://<railway-url>/v2` for the clean dashboard
- Visit `https://<railway-url>/` for the multi-slot operator dashboard
- Either trigger the cron manually with `/run?model=combo_v2&dry_run=true` to validate the pipeline
- Live rebalance kicks off at 13:35 UTC on the next weekday

---

## Local development

```bash
cd combo-v2-bot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in Alpaca paper creds

# Rebuild the model bundle (only needed if you change core/combo_strategy.py)
python scripts/build_combo_v2.py

# Run the dashboard locally
python dashboard.py
# → http://localhost:8080/v2
```

---

## Backtest reference

The bundle (`model/combo_v2/model.pkl`) carries the backtest figures inline under
`backtest_reference`. Surfaced by the dashboard so the live-vs-backtest comparison
is one click away.

| Metric | Value (2.0x leverage) |
|---|---|
| Window | 2016-04-01 → 2026-03-27 (10 yr, 2,512 trading days) |
| Universe | 1,040 US large/mid-cap stocks |
| TC | 5 bp per side |
| Mean monthly return | **4.96 %** |
| Sharpe | 1.06 |
| Max drawdown | -65.3 % |
| Calmar (return / max DD) | 0.87 |
| Rank in 45-strategy search | **1st** |
| Worst calendar year | 2022 (momentum reversal) |

Live expectation after 15-30 % friction haircut: **3.5-4.2 %/mo**.

---

## Architecture

```
combo-v2-bot/
├── core/
│   ├── combo_strategy.py     # ComboStrategy + ComboConfig (3-sleeve blend)
│   ├── runner.py             # Daily orchestrator (direct_weights dispatch)
│   ├── data.py               # yfinance downloader (stocks + macro)
│   ├── orders.py             # Alpaca rebalance with auto-scale to bp
│   ├── config.py             # Per-slot config + MODEL_REGISTRY
│   ├── state.py              # Per-slot persistent state
│   ├── risk.py               # Cut-loss scanner (-8% / -5% / -3% tiered)
│   ├── market.py             # is_market_open (DST + holidays)
│   ├── journal.py            # JSONL trade journal
│   └── ...                   # alpaca client, portfolio, run_report, etc.
├── dashboard.py              # Flask app (legacy / + clean /v2)
├── pipeline.py               # Backwards-compat shim importing from core/
├── model/combo_v2/model.pkl  # The strategy bundle (1.1 KB)
├── scripts/build_combo_v2.py # Re-bundler (re-run after strategy edits)
├── Dockerfile                # Python 3.11-slim base
├── railway.toml              # Railway service config
└── requirements.txt          # Trimmed: no lightgbm/xgboost/catboost
```

The slot framework is preserved from the parent `trading-bot` repo so additional
strategies can be added later by registering them in `core/config.py:MODEL_REGISTRY`
and dropping a `model/<name>/model.pkl` bundle.

---

## Provenance

- Strategy: derived from V5 research in [`/Users/arsenkhanguieldyan/Documents/Trading/Traiding 11/`](../Traiding%2011/)
- 45-strategy comparison: [`/Traiding 11/results/summary.csv`](../Traiding%2011/results/summary.csv)
- Original deploy framework: [`smartpneucontact-sketch/trading-bot`](https://github.com/smartpneucontact-sketch/trading-bot) (this repo is a stripped sibling)
- License: private. Do not redistribute without permission.
