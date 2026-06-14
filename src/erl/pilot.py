from __future__ import annotations

PILOT_TICKERS: list[str] = [
    "AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMZN",
    "AMD", "AVGO", "INTC",
    "CRM", "ADBE", "NOW",
    "NFLX", "DIS", "TSLA", "NKE", "MCD",
    "JNJ", "PG", "KO", "WMT",
    "JPM", "GS",
    "XOM", "CVX",
    "CAT", "HON",
    "UNH", "LLY",
    "NEE",
]

OUT_OF_UNIVERSE_EXTRAS: list[str] = ["SHOP"]


# Curated, diversified subset of well-known current S&P 500 names across all 11
# sectors. Used as a zero-dependency fallback universe on plans that gate the
# index-constituents endpoints. Survivorship bias is present (documented) since
# this is a current-membership snapshot, not point-in-time.
SP500_SUBSET: list[str] = sorted(set([
    # Technology & semis
    "AAPL", "MSFT", "NVDA", "AVGO", "AMD", "INTC", "QCOM", "TXN", "MU", "ORCL",
    "CRM", "ADBE", "NOW", "CSCO", "IBM", "ACN", "AMAT", "INTU",
    # Communication & consumer discretionary
    "GOOGL", "META", "NFLX", "DIS", "CMCSA", "TSLA", "AMZN", "HD", "NKE", "MCD",
    "SBUX", "LOW", "TGT", "BKNG",
    # Consumer staples
    "PG", "KO", "PEP", "WMT", "COST", "MDLZ", "CL",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SPGI", "AXP", "V", "MA",
    # Health care
    "JNJ", "LLY", "UNH", "PFE", "MRK", "ABBV", "TMO", "ABT", "DHR", "BMY", "AMGN",
    # Industrials
    "CAT", "HON", "UNP", "BA", "GE", "RTX", "LMT", "DE", "UPS",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG",
    # Utilities
    "NEE", "DUK", "SO",
    # Materials
    "LIN", "SHW", "FCX",
    # Real estate
    "PLD", "AMT", "EQIX",
]))
