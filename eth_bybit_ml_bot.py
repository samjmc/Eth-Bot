# ─────────────────────────────────────────────────────────────────────────────
# trade_ethusd_perp_ml.py
#
# A “live” ETHUSD Perp (USDT) trading bot using Bybit’s unified (USDT) perpetual API
# and a RandomForestClassifier trained on 15 min ETHUSD bars.
#
# Before running:
#   1) pip install pybit pandas numpy scikit-learn
#   2) export BYBIT_API_KEY=<your_key>
#      export BYBIT_API_SECRET=<your_secret>
# ─────────────────────────────────────────────────────────────────────────────

import os
import time
import pandas as pd
import numpy as np

from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier

from pybit import HTTP     # ← correct import in PyBIT v5.x

# ──────────────────────────── 1) READ API CREDENTIALS ───────────────────────────
BYBIT_API_KEY    = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")

if not BYBIT_API_KEY or not BYBIT_API_SECRET:
    raise RuntimeError("Missing BYBIT_API_KEY or BYBIT_API_SECRET in environment.")

# ────────────────────────────── 2) BOT CONSTANTS ────────────────────────────────
SYMBOL             = "ETHUSDT"      # Unified Perpetual contract symbol on Bybit
INTERVAL           = "15"           # 15-minute candles
HIST_MINUTES       = 15 * 1000      # ~1000 * 15m bars
BOOTSTRAP_DAYS     = 5              # days of history to bootstrap
TAKE_PROFIT_PCT    = 0.07           # +7% take-profit
STOP_LOSS_PCT      = 0.03           # -3% stop-loss
LABEL_DELAY_BARS   = 5              # label ~5 bars later (5*15m ≈ 75 min)
RETRAIN_FREQ_M     = 60             # retrain every 60 minutes
CAPITAL_ALLOCATION = 0.10           # 10% of USDT balance per new trade
POLL_INTERVAL_S    = 5              # poll every 5 sec until order fills/cancels

# ────────────────────────────── 3) CONNECT TO BYBIT ─────────────────────────────
session = HTTP(
    endpoint="https://api.bybit.com",
    api_key=BYBIT_API_KEY,
    api_secret=BYBIT_API_SECRET
)

# ───────────────────────────── 4) ML MODEL & SCALER ─────────────────────────────
model   = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
scaler  = StandardScaler()

# ―――――― Flags & containers for training/bootstrap
_bootstrap_done    = False
_next_retrain_time = datetime.utcnow()

training_data_X    = []   # features
training_data_y    = []   # labels (“1” if price ≥ +7% in next ~75 min; else 0)
prediction_buffer  = []   # pending-label records

# ───────────────────────────── 5) HISTORICAL BOOTSTRAP ────────────────────────────
def bootstrap_training():
    """
    Fetch ~1000 15m bars from Bybit and do an initial “bootstrap” training
    of the RandomForest on the last ~1000 bars.  Once done, set _bootstrap_done.
    """

    global _bootstrap_done, model, scaler, training_data_X, training_data_y

    print(f"[{datetime.utcnow():%Y-%m-%d %H:%M:%S}]  Starting bootstrap training ...")

    # 1) Fetch ~HIST_MINUTES of 15m ETHUSD bars
    resp = session.query_kline(
        symbol=SYMBOL,
        interval=INTERVAL,
        limit=1000  # Bybit allows up to 200 points per call; v5 unified returns up to 2000 if limit=2000
    )
    kdata = resp["result"]
    if not kdata or len(kdata) < 600:
        raise RuntimeError("Insufficient history from Bybit to bootstrap.")

    df = pd.DataFrame(kdata)
    # Bybit’s unified kline fields: "open_time","open","high","low","close","volume", ...
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("open_time", inplace=True)

    # We want the most recent ~1000 bars; they are already in chronological order.
    closes  = df["close"].astype(float).values
    highs   = df["high"].astype(float).values
    lows    = df["low"].astype(float).values
    volumes = df["volume"].astype(float).values
    n       = len(closes)

    # Precompute MACD on the close array
    close_series   = pd.Series(closes)
    ema12          = close_series.ewm(span=12, adjust=False).mean()
    ema26          = close_series.ewm(span=26, adjust=False).mean()
    macd_line      = ema12 - ema26
    signal_line    = macd_line.ewm(span=9, adjust=False).mean()
    macd_vals      = macd_line.values
    macd_hist_vals = (macd_line - signal_line).values

    Xb = []
    yb = []

    # Slide a 7-bar-ahead labeling window (7*15m ≈ 105 min, but label_delay=5*15m=75 min for ML target)
    for i in range(n - (LABEL_DELAY_BARS + 2)):
        # (A) Use the last 4 fifteen-minute closes to build features
        c4   = closes[i : i + 4]
        v4   = volumes[i : i + 4]
        present_price = closes[i + 4]
        future_price  = closes[i + LABEL_DELAY_BARS + 2]  # ~5 bars after present ≈ 75 min

        if present_price == 0.0:
            continue

        ret_pct = (future_price - present_price) / present_price
        label   = 1 if ret_pct >= TAKE_PROFIT_PCT else 0

        # (1) Log‐returns over those 4 bars
        returns_4 = np.diff(np.log(c4))
        avg_r     = np.mean(returns_4)
        std_r     = np.std(returns_4)

        # (2) Volume ratio: last 4 bars / last 8 bars
        vol4 = np.sum(v4)
        vol8 = np.sum(volumes[i : i + 8]) if np.sum(volumes[i : i + 8]) != 0 else 1
        vol_ratio = vol4 / vol8

        # (3) MACD & MACD‐hist at “present” (i+4)
        macd_val  = macd_vals[i + 4]
        macd_hist = macd_hist_vals[i + 4]

        # (4) Moving‐average ratio: fast (2‐bar) vs slow (4‐bar) from that same slice
        fast_ma = np.mean(c4[-2:])
        slow_ma = np.mean(c4) if np.mean(c4) != 0 else 1e-6
        ma_ratio = fast_ma / slow_ma

        # (5) Pivot distance: pivot = (high+low+close)/3 from bar (i+3)
        ph = highs[i + 3]
        pl = lows[i + 3]
        pc = closes[i + 3]
        pivot = (ph + pl + pc) / 3 if (ph + pl + pc) != 0 else 1e-6
        currp = c4[-1]
        pivot_dist = (currp - pivot) / pivot

        feat = [avg_r, std_r, vol_ratio, macd_val, macd_hist, ma_ratio, pivot_dist]
        Xb.append(feat)
        yb.append(label)

    Xb = np.array(Xb)
    yb = np.array(yb)

    # -- Balance classes if possible
    ones  = Xb[yb == 1]
    zeros = Xb[yb == 0]
    if len(ones) == 0 or len(zeros) == 0:
        raise RuntimeError("Bootstrap labels contain only one class; cannot train.")
    mn = min(len(ones), len(zeros))
    X1r = resample(ones, n_samples=mn, random_state=42)
    X0r = resample(zeros, n_samples=mn, random_state=42)
    X_bal = np.vstack([X1r, X0r])
    y_bal = np.array([1]*mn + [0]*mn)

    # Scale and train
    X_scaled = scaler.fit_transform(X_bal)
    model.fit(X_scaled, y_bal)

    # Save these bootstrap samples so future “retrain” uses them
    training_data_X.extend(X_bal.tolist())
    training_data_y.extend(y_bal.tolist())

    _bootstrap_done = True
    print(f"[{datetime.utcnow():%Y-%m-%d %H:%M:%S}]  Bootstrap training complete on {len(y_bal)} samples.\n")


# ─────────────────────────────────────────────────────────────────────────────
# SymbolData15
#
# Maintains a rolling 15-bar history of (close, volume, high, low) using Bybit WS or REST.
# For simplicity, we’ll just fetch most recent 15m candles each loop iteration instead of
# subscribing to WebSockets.
# ─────────────────────────────────────────────────────────────────────────────
class SymbolData15:
    def __init__(self, symbol):
        self.symbol = symbol
        self.history = []  # List of (close, volume, high, low) tuples, maxlen=60

    def update(self):
        """
        Fetch the last 8 fifteen-minute bars (to build features) and store into self.history.
        """
        resp = session.query_kline(
            symbol=self.symbol,
            interval=INTERVAL,
            limit=8
        )
        kdata = resp["result"]
        if not kdata or len(kdata) < 8:
            return False

        # We only need close, volume, high, low
        self.history = [
            (
                float(bar["close"]),
                float(bar["volume"]),
                float(bar["high"]),
                float(bar["low"])
            )
            for bar in kdata[-8:]
        ]
        return True

    def is_ready(self):
        return len(self.history) >= 8

    def get_features(self):
        """
        Build the same 7 features in realtime:
          1) avg_r:  mean(log-returns over last 4 bars)
          2) std_r:  std(log-returns over last 4 bars)
          3) vol_ratio: sum(volume last 4 bars) / sum(volume last 8 bars)
          4) macd_val: current MACD line
          5) macd_hist:  MACD_hist (MACD minus signal), but we’ll approximate via 8-bar slice
             (for simplicity let’s re-compute MACD on those 8 closes each time)
          6) ma_ratio: (2-bar SMA of last 8 closes) / (4-bar SMA of last 8 closes)
          7) pivot_dist: (close(“last”) – pivot(prev bar)) / pivot
        """
        arr = np.array(self.history)  # shape (8,4)
        closes  = arr[:, 0]
        volumes = arr[:, 1]
        highs   = arr[:, 2]
        lows    = arr[:, 3]

        # last 4 bars (indices 4..7) for returns & vol
        c4   = closes[-4:]
        v4   = volumes[-4:]
        present_price = c4[-1]

        # (A) Log-returns over last 4 bars
        returns_4 = np.diff(np.log(c4))
        avg_r = float(np.mean(returns_4)) if len(returns_4) > 0 else 0.0
        std_r = float(np.std(returns_4))  if len(returns_4) > 0 else 0.0

        # (B) Volume ratio
        vol4 = float(np.sum(v4))
        vol8 = float(np.sum(volumes)) if np.sum(volumes) != 0 else 1.0
        vol_ratio = vol4 / vol8

        # (C) MACD & MACD_hist on the last 8 closes
        cs = pd.Series(closes)
        ema12 = cs.ewm(span=12, adjust=False).mean()
        ema26 = cs.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal = macd_line.ewm(span=9, adjust=False).mean()
        macd_val = float(macd_line.values[-1])
        macd_hist = float((macd_line - signal).values[-1])

        # (D) SMA ratio: fast=2-bar SMA of last 8 closes, slow=4-bar SMA
        fast_ma = float(np.mean(c4[-2:]))
        slow_ma = float(np.mean(c4)) if float(np.mean(c4)) != 0 else 1.0
        ma_ratio = fast_ma / slow_ma

        # (E) Pivot distance using prev bar (index -2)
        prev_h = float(highs[-2])
        prev_l = float(lows[-2])
        prev_c = float(closes[-2])
        pivot = (prev_h + prev_l + prev_c) / 3 if (prev_h + prev_l + prev_c) != 0 else 1.0
        currp = present_price
        pivot_dist = (currp - pivot) / pivot

        return [avg_r, std_r, vol_ratio, macd_val, macd_hist, ma_ratio, pivot_dist]


# ─────────────────────────────────────────────────────────────────────────────
# BUY / SELL UTILS (USDT Perpetual)
# ─────────────────────────────────────────────────────────────────────────────
def place_market_buy(usdt_amount: float):
    """
    Places a market‐buy for ETHUSDT perp, spending ~usdt_amount USDT.
    Returns the order ID if successful.
    """
    # First get the current ETH price
    ticker = session.latest_information_for_symbol(symbol=SYMBOL)["result"]
    price  = float(ticker[0]["last_price"])

    qty = usdt_amount / price
    qty = round(qty, 3)  # Bybit’s min lot size is 0.001 for ETHUSDT perp

    resp = session.place_active_order(
        symbol=SYMBOL,
        side="Buy",
        order_type="Market",
        qty=qty,
        time_in_force="ImmediateOrCancel",
        reduce_only=False,
        close_on_trigger=False
    )

    if resp.get("ret_code", 0) != 0:
        raise RuntimeError(f"Buy order failed: {resp}")
    return resp["result"]["order_id"]


def place_market_sell(qty: float):
    """
    Places a market-sell to liquidate an existing ETHUSD position of size=qty.
    """
    resp = session.place_active_order(
        symbol=SYMBOL,
        side="Sell",
        order_type="Market",
        qty=qty,
        time_in_force="ImmediateOrCancel",
        reduce_only=True,
        close_on_trigger=False
    )
    if resp.get("ret_code", 0) != 0:
        raise RuntimeError(f"Sell order failed: {resp}")
    return resp["result"]["order_id"]


def fetch_position_qty():
    """
    Returns the current ETHUSDT perpetual position size (positive means long size).
    """
    pos = session.get_position_list(
        symbol=SYMBOL
    )["result"][0]
    return float(pos["size"])     # positive = long, negative = short


# ─────────────────────────────────────────────────────────────────────────────
# POLL ORDER STATUS
# ─────────────────────────────────────────────────────────────────────────────
def wait_for_fill(order_id: str, timeout: int = 30):
    """
    Polls Bybit until the order with order_id is filled (or timeout expires).
    Returns True if filled, False if timed out or not filled.
    """
    start = time.time()
    while time.time() - start < timeout:
        ob = session.query_active_order(
            symbol=SYMBOL,
            order_id=order_id
        )["result"]
        if not ob:  # empty list → fully filled or canceled
            return True
        time.sleep(POLL_INTERVAL_S)
    return False


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
def main_loop():
    """
    1) Keep updating 15m history.
    2) Make ML prediction every INTERVAL; if “buy” and no position, place market buy.
    3) Track entry price & time.
    4) Check exit conditions each loop iteration (TP/SL/time).
    5) Re-train model every RETRAIN_FREQ_M minutes.
    """
    sym15 = SymbolData15(SYMBOL)
    last_action_time = datetime.utcnow()

    while True:
        now = datetime.utcnow()

        # ── (A) Update 15m history
        if not sym15.update():
            time.sleep(5)
            continue
        if not sym15.is_ready():
            time.sleep(5)
            continue

        # ── (B) If bootstrap not done yet, do it now
        global _bootstrap_done, _next_retrain_time
        if not _bootstrap_done:
            bootstrap_training()

        # ── (C) Build features & predict
        feats = sym15.get_features()
        if feats is None:
            time.sleep(5)
            continue

        feats_scaled = scaler.transform([feats]) if _bootstrap_done else None
        pred = model.predict(feats_scaled)[0] if _bootstrap_done else 0

        # 5% random exploration
        if not fetch_position_qty() and np.random.random() < 0.05:
            pred = 1

        # ── (D) If model says “buy” and no existing position → place Market Buy
        if pred == 1 and fetch_position_qty() == 0:
            # Get current USDT balance
            wallet = session.get_wallet_balance(coin="USDT")["result"]["USDT"]
            free_usdt = float(wallet["available_balance"])
            alloc     = free_usdt * CAPITAL_ALLOCATION
            if alloc > 10:  # only trade if >10 USDT allocated
                order_id = place_market_buy(alloc)
                filled   = wait_for_fill(order_id, timeout=30)
                if filled:
                    # Record training label buffer: we’ll label this buy ~ LABEL_DELAY_BARS*15m later
                    entry_price = float(session.latest_information_for_symbol(symbol=SYMBOL)["result"][0]["last_price"])
                    training_data_X.append(feats)
                    prediction_buffer.append({
                        "timestamp": now,
                        "entry_price": entry_price
                    })
                    print(f"[{now:%Y-%m-%d %H:%M:%S}]  MARKET BUY filled, price={entry_price:.3f}, timestamp={now}")
                else:
                    print(f"[{now:%Y-%m-%d %H:%M:%S}]  BUY not filled in time, skipping.")

        # ── (E) If currently long → check exit conditions (TP/SL/time)
        position_size = fetch_position_qty()
        if position_size > 0:
            # Fetch live price
            price = float(session.latest_information_for_symbol(symbol=SYMBOL)["result"][0]["last_price"])
            # Get the entry price for this round from prediction_buffer (we assume only one open at a time)
            recs_to_keep = []
            exit_now = False
            for rec in prediction_buffer:
                # If now ≥ rec["timestamp"] + LABEL_DELAY_BARS*15m → label & remove
                if now >= rec["timestamp"] + timedelta(minutes=LABEL_DELAY_BARS * 15):
                    entry_px = rec["entry_price"]
                    ret = (price - entry_px) / entry_px
                    label = 1 if ret >= TAKE_PROFIT_PCT else 0
                    training_data_y.append(label)
                    # We keep this rec around to measure TP/SL/time exit:
                    recs_to_keep.append(rec)
                else:
                    recs_to_keep.append(rec)
            prediction_buffer[:] = recs_to_keep

            # Always compare against the oldest “entry” that is still pending exit
            if prediction_buffer:
                epx = prediction_buffer[0]["entry_price"]
                dt0 = prediction_buffer[0]["timestamp"]
                ret_pct = (price - epx) / epx
                hold_mins = (now - dt0).total_seconds() / 60.0

                if ret_pct >= TAKE_PROFIT_PCT:
                    exit_now = True
                    print(f"[{now:%Y-%m-%d %H:%M:%S}]  TAKE-PROFIT hit (@{price:.3f})")
                elif ret_pct <= -STOP_LOSS_PCT:
                    exit_now = True
                    print(f"[{now:%Y-%m-%d %H:%M:%S}]  STOP-LOSS hit (@{price:.3f})")
                elif hold_mins >= (LABEL_DELAY_BARS + 2) * 15:  # ~105 min
                    exit_now = True
                    print(f"[{now:%Y-%m-%d %H:%M:%S}]  TIME-BASED exit after {int(hold_mins)} min (@{price:.3f})")

                if exit_now:
                    # Place market sell of the entire position_size
                    sell_id = place_market_sell(position_size)
                    filled = wait_for_fill(sell_id, timeout=30)
                    if filled:
                        print(f"[{now:%Y-%m-%d %H:%M:%S}]  POSITION CLOSED, qty={position_size} ETH @ {price:.3f}")
                        # Remove that record from prediction_buffer
                        prediction_buffer.pop(0)
                    else:
                        print(f"[{now:%Y-%m-%d %H:%M:%S}]  Exit order timed out → might stay open.")

        # ── (F) RETRAIN check
        if now >= _next_retrain_time and len(training_data_y) >= 40:
            arrX = np.array(training_data_X)
            arry = np.array(training_data_y)
            wins  = np.sum(arry == 1)
            losses= np.sum(arry == 0)
            if wins >= 20 and losses >= 20:
                # Re-balance classes and retrain
                X1 = arrX[arry == 1]
                X0 = arrX[arry == 0]
                mn = min(len(X1), len(X0))
                X1r = resample(X1, n_samples=mn, random_state=42)
                X0r = resample(X0, n_samples=mn, random_state=42)
                X_bal = np.vstack([X1r, X0r])
                y_bal = np.array([1]*mn + [0]*mn)

                Xs = scaler.fit_transform(X_bal)
                model.fit(Xs, y_bal)
                print(f"[{now:%Y-%m-%d %H:%M:%S}]  Model retrained on {len(y_bal)} samples.")
                # Keep only the last 1000 training points
                training_data_X[:] = training_data_X[-1000:]
                training_data_y[:] = training_data_y[-1000:]
            _next_retrain_time = now + timedelta(minutes=RETRAIN_FREQ_M)

        # ── (G) Sleep until next minute’s 15m boundary
        #    (bybit klines update ~at :00, :15, :30, :45)
        time_to_next_15m = 60 - (now.minute % 15)*60 - now.second
        sleep_secs = max(5, time_to_next_15m + 1)
        time.sleep(sleep_secs)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1) Bootstrap training (if not done)
    if not _bootstrap_done:
        bootstrap_training()

    # 2) Enter the never-ending main loop
    main_loop()
