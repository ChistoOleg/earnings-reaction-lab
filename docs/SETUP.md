# Setup & Getting Started

This guide takes you from a fresh machine to a working pilot run. It covers prerequisites,
getting a Financial Modeling Prep (FMP) API key, installation on both a CPU machine and a
GPU machine, configuration, the first run, scaling to the full universe, and troubleshooting.

If you only want the one-paragraph version: install Python 3.10+, create a virtual
environment, `pip install -r requirements.txt`, run `pytest` to confirm 47 tests pass, put
an FMP key in `.env`, then run `python -m erl.pipeline all --universe pilot`.

---

## 1. What you are installing

The Earnings Reaction Lab is a research pipeline that studies why similar earnings beats
produce different stock-price reactions. It has two kinds of work:

- **CPU work** — data harvesting, the event panel, econometric inference (double lasso,
  causal forest), and gradient-boosting prediction. Runs anywhere, including a Mac mini.
- **GPU work** — an FT-Transformer benchmark and transcript embeddings with a FinBERT-class
  model. Needs an NVIDIA GPU with CUDA (e.g. an RTX 4090).

A sensible split is to run the slow, always-on harvest on a Mac mini and the GPU track on a
Windows/Linux PC with the 4090. The code is OS-agnostic and the two machines share data
through parquet files.

---

## 2. Prerequisites

- **Python 3.10, 3.11, or 3.12.** Check with `python3 --version`.
- **git** (for cloning and pushing to GitHub).
- About **2 GB of free disk** for the full-universe data cache (transcripts are the bulk).
- For the GPU track only: an **NVIDIA GPU + recent driver** and a matching CUDA-enabled
  PyTorch build.

No database server, no Docker, and no API keys other than FMP are required.

---

## 3. Get a Financial Modeling Prep (FMP) API key

The pipeline uses FMP as its data source. Getting a key takes a few minutes.

1. Go to the FMP website (`https://site.financialmodelingprep.com`).
2. Find the sign-up box, enter your email, set a password, and submit.
3. Verify your email from the message FMP sends. Verification is what activates the account.
4. Sign in and open your dashboard at
   `https://site.financialmodelingprep.com/developer/docs/dashboard`. Your API key is shown
   there — copy it. Treat it like a password; it authenticates every request as
   `...?apikey=YOUR_KEY`, which the code adds for you.

### Which plan do you need?

- The **free plan** (around 250 requests/day — confirm the current figure on the dashboard)
  is enough to verify that your key works and that connectivity is fine, but it generally
  does **not** include the point-in-time historical S&P 500 constituents endpoint or the
  deep history this project depends on.
- **Starter or Premium** covers everything for the main analysis (notebooks 01–04): earnings
  surprises, historical constituents, deep prices, and analyst estimates. Start here.
- **Ultimate** is needed only for **earnings-call transcripts** (notebook 05). The
  cost-efficient pattern is to subscribe to Ultimate for a single month, run the transcript
  harvest once (it caches locally as parquet), then downgrade — your GPU works off the cache
  afterward.

Prices and plan contents change, and FMP runs frequent discounts, so check the current
pricing page before subscribing rather than trusting a number quoted elsewhere.

### How the code reacts to plan limits

If you call an endpoint your plan does not include, FMP returns a 402 or 403. The client
turns that into a clear, actionable error naming the endpoint, instead of failing silently —
so if you see a message about an endpoint "not included in your FMP plan," upgrade the tier
or skip that stage. Transcript calls specifically tell you to use the Ultimate plan.

---

## 4. Get the code

If you received the project as `earnings-reaction-lab.zip`, unzip it:

```
unzip earnings-reaction-lab.zip
cd earnings-reaction-lab
```

Or clone it from GitHub if you have already pushed it:

```
git clone https://github.com/YOUR_USERNAME/earnings-reaction-lab.git
cd earnings-reaction-lab
```

---

## 5. Install (CPU machine — e.g. the Mac mini)

1. Create and activate a virtual environment:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```
   On Windows PowerShell the activation line is `.venv\Scripts\Activate.ps1` instead.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Confirm everything works before touching any data or paying for a key:
   ```
   python -m pytest -q -rs
   ```
   You should see `47 passed, 1 skipped`. The single skip is the GPU test, which is expected
   to skip on a machine without PyTorch.

## 6. Install (GPU machine — the 4090 PC)

Do the steps in section 5 first, then add the GPU dependencies:

```
pip install -r requirements-gpu.txt
```

`requirements-gpu.txt` pulls in `torch`, `transformers`, and `accelerate`. If you need a
specific CUDA build of PyTorch for your driver, install that build first following the
official PyTorch instructions, then run the line above. Verify the GPU is visible:

```
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

It should print `True`. Re-running `python -m pytest tests/test_dl_text.py -q` will now
execute the FT-Transformer test instead of skipping it.

---

## 7. Configure (`.env`)

Configuration is read from a `.env` file in the project root. Copy the template and edit it:

```
cp .env.example .env
```

Then set at least your key. The variables (all prefixed `ERL_`) are:

| Variable | Meaning | Default |
|---|---|---|
| `ERL_FMP_API_KEY` | Your FMP key from the dashboard | empty (required) |
| `ERL_DATA_DIR` | Where parquet data and the request cache live | `data` |
| `ERL_START_DATE` | Earliest date to harvest (YYYY-MM-DD) | `2015-01-01` |
| `ERL_REQUEST_INTERVAL_SECONDS` | Minimum gap between FMP calls (rate-limit politeness) | `0.25` |

`.env` is listed in `.gitignore`, so your key never gets committed. The `data/` directory is
also ignored, so cached pulls stay local.

---

## 8. First run — the pilot

The pilot uses 30 tickers (plus Shopify as an out-of-universe comparison) so you can validate
the whole chain quickly and cheaply:

```
python -m erl.pipeline all --universe pilot
```

This harvests data, builds the event panel, runs inference, and runs prediction, writing
results to `data/processed/`. Then open `notebooks/01_event_study` and confirm:

- the event count is roughly 1,000–1,200,
- average abnormal return rises across surprise quintiles (the top quintile significant),
- the run completed without a leakage error (the pipeline raises if day-0 alignment is wrong),
- META and SHOP show the opposite-reaction pattern that motivates the project.

If those look right, the data and methodology are sound.

### Running notebooks

The notebooks are stored as `.py` files in jupytext "percent" format, which keeps clean diffs
in git. Open them directly in VS Code (it renders cells), or convert to a classic notebook:

```
pip install jupytext jupyter
jupytext --to notebook notebooks/01_event_study.py
jupyter lab
```

---

## 9. Scale to the full S&P 500

Once the pilot checks out and you are on at least the Starter plan:

```
python -m erl.pipeline harvest --universe sp500
python -m erl.pipeline panel
python -m erl.pipeline inference
python -m erl.pipeline predict
```

The full harvest fetches ~500–600 names across several endpoints. It is network-bound and
takes up to an hour or two, but it is **resumable**: the on-disk cache means that if it is
interrupted or rate-limited, re-running the same command picks up where it left off. Expect
roughly 18,000–20,000 events in the final panel.

---

## 10. GPU and transcript track

Run this on the 4090 machine, after section 6.

1. Subscribe to FMP Ultimate (one month is enough).
2. Harvest and embed transcripts via `notebooks/05_dl_and_text`. The embeddings cache to
   `data/processed/embeddings.parquet` as they are produced, so this is resumable too.
3. Downgrade FMP back to Starter once the transcripts are cached.

The FinBERT-class model downloads automatically the first time from the model hub; this needs
internet access and a few hundred MB of disk.

---

## 11. Troubleshooting

- **"endpoint is not included in your FMP plan" (402/403).** Your tier lacks that endpoint.
  Upgrade (Starter for constituents/history, Ultimate for transcripts) or skip that stage.
- **Repeated 429 responses.** You are hitting the rate limit. The client already backs off and
  retries; if it persists, raise `ERL_REQUEST_INTERVAL_SECONDS` (e.g. to `0.5`) and re-run —
  the cache means you do not lose completed work.
- **`benchmark ^GSPC not found in prices`.** The price harvest did not include the benchmark.
  Re-run the harvest stage; the pipeline requests benchmarks alongside tickers automatically.
- **Empty or tiny panel.** Usually means the surprise or price harvest returned little — check
  your plan covers the date range in `ERL_START_DATE` and that the harvest completed.
- **`torch` not found / GPU test skips.** Expected on a CPU machine. Install
  `requirements-gpu.txt` on the GPU machine to enable the deep-learning track.
- **CUDA available prints `False`.** Your PyTorch build does not match your driver. Reinstall
  the CUDA-specific PyTorch build from the official instructions, then reinstall the rest.
- **Tests fail after editing code.** Run `python -m pytest -q -rs` to see which stage broke;
  the suite is fast and the failure messages point at the module.

---

## 12. Cost summary

- FMP **Starter** for the active build period (covers notebooks 01–04).
- One month of FMP **Ultimate** for transcripts (notebook 05), then downgrade.
- No other paid services. Electricity for an always-on Mac mini is negligible.

Verify the exact prices on the FMP pricing page when you subscribe, as they change and are
frequently discounted.
