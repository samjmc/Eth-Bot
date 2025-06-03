# download_15m_data.py

import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta

# ─── CONFIG ─────────────────────────────────────────────────────────
# If you want Testnet, CCXT has a special symbol for Bybit testnet:
#     ccxt.bybit({ 'options': { 'defaultType': 'future' }, 'urls': { 'api': { 'public': 'https://api-testnet.bybit.com' } } })
# For simplicity, this example pulls from mainnet:
EXCHANGE_ID    = "bybit"
SYMBOL         = "ETH/USDT"
TIMEFRAME      = "15m"
OUTPUT_CSV     = "eth_usdt_15m.csv"

# How many candles to fetch per request: Bybit allows up to 200
LIMIT_PER_CALL = 200

# How many total candles you want (approx). To cover 5 years of 15m bars:
#  5 years ≈ 365 days × 24 hours × (60/15) bars = 35,040 bars → so 36,000 is a comfortable round number.
TOTAL_BARS_WANTED = 36000

# ─── INITIALIZE CCXT EXCHANGE ─────────────────────────────────────────
exchange_class = getattr(ccxt, EXCHANGE_ID)
exchange = exchange_class({
    "enableRateLimit": True,
    # If you wanted testnet instead of mainnet, you could do:
    # "urls": { "api": { "public": "https://api-testnet.bybit.com",
    #                      "private": "https://api-testnet.bybit.com" } }
})
# (No API key is needed just to fetch public OHLCV data.)


def fetch_historical_15m(symbol, timeframe, since_timestamp, limit):
    """
    Wrapper around exchange.fetch_ohlcv(...) for convenience.
    Returns a list of lists: [ timestamp, open, high, low, close, volume ].
    """
    return exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_timestamp, limit=limit)


if __name__ == "__main__":
    # 1) Calculate how many loops we need:
    loops_needed = (TOTAL_BARS_WANTED // LIMIT_PER_CALL) + 1

    # 2) CCXT wants "since" as milliseconds. We'll start by pulling the most recent 'limit' bars,
    #    then step backward in time by timeframe × limit. 15m in ms is 15 * 60 * 1000 = 900_000 ms.
    timeframe_ms   = exchange.parse_timeframe(TIMEFRAME) * 1000
    all_candles    = []
    now_ms         = int(time.time() * 1000)

    # 3) Loop to fetch backwards until we've collected enough bars
    since_ms = now_ms - (timeframe_ms * LIMIT_PER_CALL)
    for i in range(loops_needed):
        try:
            candles = fetch_historical_15m(SYMBOL, TIMEFRAME, since_ms, LIMIT_PER_CALL)
        except Exception as e:
            print(f"[Loop {i}] Error fetching OHLCV: {e}. Retrying in 2s…")
            time.sleep(2)
            candles = fetch_historical_15m(SYMBOL, TIMEFRAME, since_ms, LIMIT_PER_CALL)

        if not candles:
            break

        # CCXT returns: [ [timestamp, open, high, low, close, volume], … ]
        all_candles.extend(candles)
        
        # Move "since_ms" backward by LIMIT_PER_CALL × timeframe_ms
        earliest_ts = candles[0][0]
        since_ms = earliest_ts - timeframe_ms

        # Respect CCXT's rate limit:
        time.sleep(exchange.rateLimit / 1000)

    # 4) Convert to DataFrame and dedupe/sort
    df = pd.DataFrame(
        all_candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    # Some bars may overlap (because of how CCXT/Bybit pagination works), so drop duplicates on timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)

    # 5) Keep only the most recent TOTAL_BARS_WANTED bars
    if len(df) > TOTAL_BARS_WANTED:
        df = df.iloc[-TOTAL_BARS_WANTED :].copy().reset_index(drop=True)

    # 6) Save to CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} bars to {OUTPUT_CSV} (from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]})")
