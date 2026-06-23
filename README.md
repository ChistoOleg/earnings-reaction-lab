# Market Intelligence Bot

A self-hosted Telegram bot that delivers a scheduled cross-asset **market digest** and **instant, analysed alerts** for the tickers you track. It runs as a single-tenant, owner-locked instance: you supply the keys, and it serves you (plus an optional allowlist) only.

> Everything it produces is information and analysis, **not personalised investment advice.**

## Demo

Per-ticker news alerts, each carrying an LLM analysis bar — **sentiment**, **impact**, **why it matters**, and **what to watch**. The same pipeline handles the full sentiment range:

| 🟢 Bullish | ⚪ Neutral | 🔴 Bearish |
|---|---|---|
| ![Bullish alert](docs/alert_bullish.png) | ![Neutral alert](docs/alert_neutral.png) | ![Bearish alert](docs/alert_bearish.png) |

*Real output for tracked tickers. Each article is fetched, deduplicated per user, classified, and pushed once with a structured read — not just a headline.*

## What it does

- **4-hour digest** (configurable times): one brief covering US/UK/global indices, major FX, BTC/ETH, WTI & gold, US 10Y, VIX and DXY — with moves vs prior close and vs the last digest, outlier flags, upcoming earnings for your tracked names, and an LLM synthesis (what changed, why it matters, what to watch). Computed once and broadcast.
- **Whitelist news alerts**: polls news for the union of tracked tickers, and for each fresh article pushes a tight analysis (summary, sentiment, impact direction/magnitude, materiality, why-it-matters, what-to-watch) to every user holding that ticker — once each, with strict per-user dedup.
- **Price alerts**: `>`/`<` level crossings and daily `%` moves, one-shot or repeating.

## How it works

- **Compute-once → broadcast** for the digest; **poll-union → fan-out** for alerts.
- **Owner-lock:** every handler is gated; non-authorised users never reach the data providers or the LLM, so strangers can't run up your API bill.
- **Data:** Yahoo Finance via `yfinance` for all prices and news (no key). Anthropic (Claude) is the only paid dependency. An optional Finnhub key adds a US-news fallback.
- **Storage:** SQLite, keyed by Telegram user ID, structured so a move to PostgreSQL is a connection change rather than a rewrite.

## Requirements

- Python 3.10+
- A Telegram bot token
- An Anthropic API key
- (Optional) a Finnhub API key for a news fallback

## Quick start

```bash
git clone <your-fork-url> market-intel-bot
cd market-intel-bot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env: TELEGRAM_BOT_TOKEN, OWNER_TELEGRAM_ID, ANTHROPIC_API_KEY
python -m app.main
```

Then message your bot `/start`.

## Getting the keys and IDs

- **Telegram bot token:** open [@BotFather](https://t.me/BotFather), `/newbot`, copy the token. To accept Star donations, also run `/mybots → your bot → Payments` once.
- **Your Telegram user ID:** message [@userinfobot](https://t.me/userinfobot); it replies with your numeric ID. Put it in `OWNER_TELEGRAM_ID`.
- **Anthropic API key:** create one at <https://console.anthropic.com>.
- **Finnhub (optional):** free key at <https://finnhub.io>.

## Configuration

| Variable | Required | Default | Notes |
|---|---|---|---|
| `TELEGRAM_BOT_TOKEN` | yes | — | From BotFather |
| `OWNER_TELEGRAM_ID` | yes | — | The only user served by default |
| `ANTHROPIC_API_KEY` | yes | — | The only guaranteed cost |
| `ALLOWED_USER_IDS` | no | empty | Comma-separated extra IDs allowed on this instance |
| `DIGEST_MODEL` | no | `claude-sonnet-4-6` | Digest synthesis |
| `CLASSIFY_MODEL` | no | `claude-haiku-4-5-20251001` | Per-article classification |
| `TIMEZONE` | no | `Europe/London` | IANA zone |
| `DIGEST_TIMES` | no | `06:00,10:00,14:00,18:00,22:00` | Local broadcast times |
| `MARKET_OPEN_HOUR` / `MARKET_CLOSE_HOUR` | no | `8` / `21` | Local hours defining the fast-poll window |
| `POLL_INTERVAL_MARKET_SECONDS` | no | `300` | Poll interval in-window |
| `POLL_INTERVAL_OFFHOURS_SECONDS` | no | `1200` | Effective interval off-hours (equities); crypto stays live |
| `NEWS_LOOKBACK_HOURS` | no | `24` | Bounds the news fetch window |
| `FINNHUB_API_KEY` | no | empty | Enables the US-news fallback |
| `FMP_API_KEY` | no | empty | Reserved for a future macro calendar; unused |
| `DONATE_URL` / `DONATE_TEXT` | no | — | External donation link shown by `/donate` |
| `DONATE_STARS_AMOUNTS` | no | `50,100,250` | Telegram Stars amounts offered |
| `ALERT_LLM_COMMENTARY` | no | `false` | Reserved toggle for per-trigger LLM notes |
| `MAX_TRACKED_SYMBOLS` | no | `100` | Per-user watchlist cap |
| `DATABASE_PATH` | no | `data/bot.db` | SQLite path |
| `LOG_LEVEL` | no | `INFO` | Structured JSON logs |
| `HTTP_TIMEOUT_SECONDS` / `HTTP_MAX_RETRIES` | no | `15` / `3` | Network behaviour |

## Commands

```
/add <TICKER>          track a symbol (AAPL, VOD.L, BTC-USD)
/remove <TICKER>       stop tracking
/list                  show your watchlist
/digest                send the latest digest now
/alert <T> > <PRICE>   alert above a level (use < for below)
/alert <T> +5%         alert on a daily move (-5% for down); add 'repeat' to re-arm
/alerts                list active alerts
/alert_remove <ID>     delete an alert
/donate                support development
/help                  show help
```

## Running with Docker

```bash
docker build -t market-intel-bot .
docker run -d --name market-intel-bot \
  --env-file .env \
  -v "$(pwd)/data:/app/data" \
  market-intel-bot
```

The volume persists the SQLite database across restarts.

## Deployment

**VPS (systemd):**

```ini
[Unit]
Description=Market Intelligence Bot
After=network-online.target
Wants=network-online.target

[Service]
WorkingDirectory=/opt/market-intel-bot
EnvironmentFile=/opt/market-intel-bot/.env
ExecStart=/opt/market-intel-bot/.venv/bin/python -m app.main
Restart=on-failure
RestartSec=5
User=botuser

[Install]
WantedBy=multi-user.target
```

**Railway / Fly.io / Render:** deploy the Dockerfile, set the environment variables in the platform dashboard, and attach a persistent volume mounted at `/app/data`. The bot uses long polling, so no inbound port or public URL is required.

## Data and privacy

- Single-tenant by design. The bot only responds to `OWNER_TELEGRAM_ID` and any `ALLOWED_USER_IDS`; everyone else gets a "private instance" reply and is never passed to the providers or the LLM.
- All state lives in your local SQLite file: registered users, per-user watchlists, per-user sent-article hashes (for dedup), price alerts, and the latest market snapshot. No analytics, no third-party storage.
- Keys live only in your `.env`. Never commit it.

## Donations

`/donate` shows your `DONATE_URL` (if set) and Telegram Stars buttons. Stars use Telegram's native in-app payment (`currency=XTR`, no payment provider needed); note that Apple/Google take a cut when users buy Stars on mobile, so desktop tips net more. Forks can point `DONATE_URL` at their own page.

## Limitations and data sources

- **Yahoo via `yfinance` is unofficial and best-effort.** Quotes are delayed (~15 min) and the endpoints can change without notice. The bot caches, retries, and degrades per-section rather than crashing — but it is not a real-time tape, and price alerts are delayed-price alerts.
- **UK 10Y gilt yield** is not available from Yahoo (only gilt price indices are), so it shows as unavailable in the digest until you wire a dedicated source.
- **No macro economic calendar.** The digest surfaces upcoming **earnings** for your tracked names from Yahoo instead. `FMP_API_KEY` is reserved for adding a true macro calendar later.
- **News latency/coverage:** Yahoo news is good for US large caps, patchier for UK/small caps and not as fresh as a dedicated feed. Set `FINNHUB_API_KEY` to add a US fallback.
- **Cost scales with your watchlist.** The news poll and per-article classification grow with the number of tracked tickers; keep `MAX_TRACKED_SYMBOLS` sensible.

## Development

Run the test suite (real SQLite, fake providers/bot — no keys or network needed for the logic tests):

```bash
pip install -r requirements-dev.txt
pytest -q
```

The suite covers config parsing/validation, storage and all repositories, the formatters, the analysis layer (JSON parsing + classification coercion), and the job logic (digest broadcast, news fan-out with dedup, and the price-alert lifecycle). A GitHub Actions workflow (`.github/workflows/ci.yml`) runs it on every push and pull request across Python 3.10–3.12.

## License

MIT — see [LICENSE](LICENSE).
