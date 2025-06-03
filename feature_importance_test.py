import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# ─── CONFIG ─────────────────────────────────────────────────────────
CSV_PATH = "eth_usdt_15m.csv"    # 15m OHLCV CSV from Bybit (produced by download_15m_data.py)
BOOTSTRAP_BARS = 2000            # how many 15m bars to use for bootstrapping
                                  # (adjust if your CSV is shorter)

# ─── 1) LOAD DATA ────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)

# Only keep the last (BOOTSTRAP_BARS + 7) bars
if len(df) > BOOTSTRAP_BARS + 7:
    df = df.tail(BOOTSTRAP_BARS + 7)

# ─── 2) PRECOMPUTE INDICATORS: MACD, SMA fast/slow ────────────────────
closes = df["close"].astype(float)
ema12 = closes.ewm(span=12, adjust=False).mean()
ema26 = closes.ewm(span=26, adjust=False).mean()
macd_line = ema12 - ema26
signal_line = macd_line.ewm(span=9, adjust=False).mean()

df["macd"] = macd_line
df["macd_hist"] = macd_line - signal_line
df["sma_fast"] = closes.rolling(window=2).mean()   # 2‐bar SMA (~30 minutes if 15m bars)
df["sma_slow"] = closes.rolling(window=4).mean()   # 4‐bar SMA (~60 minutes)

# ─── 3) BUILD FEATURES & LABELS (7 features) ────────────────────────
X_list = []
y_list = []

arr_h = df["high"].values
arr_l = df["low"].values
arr_c = df["close"].values
arr_v = df["volume"].values
arr_m = df["macd"].values
arr_hm = df["macd_hist"].values
arr_sf = df["sma_fast"].values
arr_ss = df["sma_slow"].values

n = len(df)
for i in range(n - 7):
    window_closes = arr_c[i : i + 4]   # 4-bar window
    window_vols   = arr_v[i : i + 4]
    future_price  = arr_c[i + 7]
    present_price = arr_c[i + 4]
    if present_price == 0:
        continue

    # ── Label: did price move ≥ 0.5% over next ~75 minutes?
    ret_pct = (future_price - present_price) / present_price
    label = 1 if ret_pct >= 0.005 else 0

    # ── (A) Log‐returns over those 4 bars
    returns4 = np.diff(np.log(window_closes))
    avg_r = np.mean(returns4)
    std_r = np.std(returns4)

    # ── (B) Volume ratio: last 4 bars vs last 8 bars
    vol4 = np.sum(window_vols)
    vol8 = np.sum(arr_v[i : i + 8]) if np.sum(arr_v[i : i + 8]) != 0 else 1
    vol_ratio = vol4 / vol8

    # ── (C) MACD & MACD‐hist at bar (i + 4)
    macd_val = arr_m[i + 4]
    macd_hist = arr_hm[i + 4]

    # ── (D) SMA ratio (fast/slow) at bar (i + 4)
    fast_ma = arr_sf[i + 4]
    slow_ma = arr_ss[i + 4] if arr_ss[i + 4] != 0 else 1e-6
    ma_ratio = fast_ma / slow_ma

    # ── (E) Pivot distance using bar (i + 3)
    prev_high = arr_h[i + 3]
    prev_low = arr_l[i + 3]
    prev_close = arr_c[i + 3]
    pivot = (prev_high + prev_low + prev_close) / 3 if (prev_high + prev_low + prev_close) != 0 else 0
    currp = window_closes[-1]
    pivot_dist = (currp - pivot) / pivot if pivot != 0 else 0

    # Compose the 7‐dimensional feature vector
    feat = [
        avg_r,      # log‐return mean
        std_r,      # log‐return std
        vol_ratio,  # volume ratio
        macd_val,   # MACD line
        macd_hist,  # MACD histogram
        ma_ratio,   # fast/slow SMA ratio
        pivot_dist  # pivot point distance
    ]

    X_list.append(feat)
    y_list.append(label)

Xb = np.array(X_list)
yb = np.array(y_list)
print(f"Built {len(yb)} samples → {np.sum(yb==1)} positives, {len(yb)-np.sum(yb==1)} negatives")

# ─── 4) BALANCE & TRAIN ───────────────────────────────────────────────
ones = np.sum(yb == 1)
zeros = np.sum(yb == 0)
mn = min(ones, zeros)
if mn == 0:
    raise RuntimeError("All labels are one class—cannot train.")

X1 = resample(Xb[yb == 1], n_samples=mn, random_state=42)
X0 = resample(Xb[yb == 0], n_samples=mn, random_state=42)
X_bal = np.vstack((X1, X0))
y_bal = np.array([1]*mn + [0]*mn)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_bal)

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_scaled, y_bal)
print("RandomForest trained on balanced bootstrap data.")

# ─── 5) REPORT FEATURE IMPORTANCES ───────────────────────────────────
feature_names = [
    "avg_r",
    "std_r",
    "vol_ratio",
    "macd_val",
    "macd_hist",
    "ma_ratio",
    "pivot_dist"
]
importances = model.feature_importances_
feat_imp = list(zip(feature_names, importances))
feat_imp.sort(key=lambda x: x[1], reverse=True)

print("\n=== Feature Importances ===")
for name, score in feat_imp:
    print(f"{name.ljust(10)} : {score:.4f}")

# Plot a horizontal bar‐chart
plt.figure(figsize=(8, 5))
plt.barh(
    [n for n, _ in reversed(feat_imp)],
    [s for _, s in reversed(feat_imp)],
    color="skyblue"
)
plt.title("Feature Importances (Bootstrap Dataset, 7 features)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()
