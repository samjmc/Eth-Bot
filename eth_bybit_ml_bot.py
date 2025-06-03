from AlgorithmImports import *
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
import random

class ETHMinuteMLBot(QCAlgorithm):

    def Initialize(self):
        # ─── BACKTEST PERIOD ───────────────────────────────────
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 9, 1)
        self.SetCash(100000)

        # ─── RISK/CAPITAL PARAMETERS ───────────────────────────
        self.capitalAllocation       = 0.1    # 10% of available cash per trade
        self.take_profit_pct         = 0.07   # 7% take-profit
        self.stop_loss_pct           = 0.03   # 3% stop-loss
        self.label_delay_minutes     = 5      # label after 5 fifteen-minute bars (~75 min)
        self.time_based_exit_minutes = 105    # exit after 7 fifteen-minute bars (~105 min)

        # ─── CONTAINERS FOR DATA & MODEL ──────────────────────
        self.symbol_data        = {}
        self.training_data_X    = []   # list of feature lists
        self.training_data_y    = []   # list of labels
        self.prediction_buffer  = []   # pending labels
        self.bootstrap_training = True
        self.next_model_train   = self.Time

        # ─── MACHINE LEARNING MODEL ───────────────────────────
        self.model  = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.scaler = StandardScaler()
        self._is_model_trained = False

        # ─── ASSET SETUP: ETHUSD at minute resolution, consolidate to 15-min bars
        self.eth = self.AddCrypto("ETHUSD", Resolution.Minute, Market.GDAX).Symbol
        self.symbol_data[self.eth] = SymbolData(self, self.eth)

        # ─── WARM-UP ────────────────────────────────────────────
        # 5 days of minute data to warm up consolidators & indicators
        self.SetWarmUp(timedelta(days=5))

        self.Debug("Initialize complete")

    def OnData(self, data: Slice):
        if self.IsWarmingUp:
            return

        # ── (0) BOOTSTRAP PHASE: initial labeling & training from history
        if self.bootstrap_training and self.Time > self.StartDate + timedelta(days=5):
            hist = self.History(self.eth, 7000, Resolution.Minute)
            if hist.empty or len(hist) < 600:
                return

            minute_df = hist.loc[self.eth].reset_index()
            minute_df['time_dt'] = minute_df['time'].dt.floor('15T')
            grouped = minute_df.groupby('time_dt').agg({
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).tail(1000)  # most recent ~1000 15-min bars

            highs   = grouped['high'].values
            lows    = grouped['low'].values
            closes  = grouped['close'].values
            volumes = grouped['volume'].values
            n = len(closes)

            # precompute MACD lines for the entire 15m series
            close_series   = pd.Series(closes)
            ema12          = close_series.ewm(span=12, adjust=False).mean()
            ema26          = close_series.ewm(span=26, adjust=False).mean()
            macd_line      = ema12 - ema26
            signal_line    = macd_line.ewm(span=9, adjust=False).mean()
            macd_vals      = macd_line.values
            macd_hist_vals = (macd_line - signal_line).values

            X_bootstrap = []
            y_bootstrap = []

            for i in range(n - 7):
                window_closes = closes[i : i + 4]    # 4 bars = 1 hour
                window_vols   = volumes[i : i + 4]
                future_price  = closes[i + 7]
                present_price = closes[i + 4]
                if present_price == 0:
                    continue

                # ── Label: did price move ≥ 0.5% over next ~75 min?
                ret_pct = (future_price - present_price) / present_price
                label   = 1 if ret_pct >= 0.005 else 0

                # ── (A) Log‐returns over those 4 bars
                returns_4 = np.diff(np.log(window_closes))
                avg_r     = np.mean(returns_4)
                std_r     = np.std(returns_4)

                # ── (B) Volume ratio: last 4 bars vs last 8 bars
                vol4 = np.sum(window_vols)
                vol8 = np.sum(volumes[i : i + 8]) if np.sum(volumes[i : i + 8]) != 0 else 1
                vol_ratio = vol4 / vol8

                # ── (C) MACD & MACD‐hist at bar (i+4)
                macd_val  = macd_vals[i + 4]
                macd_hist = macd_hist_vals[i + 4]

                # ── (D) Moving average ratio: fast (2‐bar) vs slow (4‐bar)
                fast_ma = np.mean(window_closes[-2:])
                slow_ma = np.mean(window_closes[-4:]) if np.mean(window_closes[-4:]) != 0 else 1e-6
                ma_ratio = fast_ma / slow_ma

                # ── (E) Pivot point distance using bar (i+3)
                prev_high  = highs[i + 3]
                prev_low   = lows[i + 3]
                prev_close = closes[i + 3]
                pivot      = (prev_high + prev_low + prev_close) / 3 if (prev_high + prev_low + prev_close) != 0 else 0
                currp      = window_closes[-1]
                pivot_dist = (currp - pivot) / pivot if pivot != 0 else 0

                feat = [
                    avg_r,
                    std_r,
                    vol_ratio,
                    macd_val,
                    macd_hist,
                    ma_ratio,
                    pivot_dist
                ]
                X_bootstrap.append(feat)
                y_bootstrap.append(label)

            if len(X_bootstrap) > 0:
                Xb = np.array(X_bootstrap)
                yb = np.array(y_bootstrap)
                num_ones  = np.sum(yb == 1)
                num_zeros = np.sum(yb == 0)

                if num_ones > 0 and num_zeros > 0:
                    mn = min(num_ones, num_zeros)
                    X1r = resample(Xb[yb == 1], n_samples=mn, random_state=42)
                    X0r = resample(Xb[yb == 0], n_samples=mn, random_state=42)
                    X_bal = np.vstack((X1r, X0r))
                    y_bal = np.array([1] * mn + [0] * mn)

                    X_scaled = self.scaler.fit_transform(X_bal)
                    self.model.fit(X_scaled, y_bal)
                    self._is_model_trained   = True
                    self.bootstrap_training  = False
                    self.Log(f"Bootstrap training complete on {len(X_bal)} samples")
                    self.LogFeatureImportances()
                else:
                    return
            else:
                return

        # ── (1) DELAYED LABELING: assign labels ~75 minutes after each buy
        for record in list(self.prediction_buffer):
            if self.Time >= record["timestamp"] + timedelta(minutes=15 * self.label_delay_minutes):
                entry_price   = record["entry_price"]
                current_price = self.Securities[self.eth].Price
                if entry_price == 0:
                    self.prediction_buffer.remove(record)
                    continue

                # Align labeling with our 7% take‐profit threshold
                ret   = (current_price - entry_price) / entry_price
                label = 1 if ret >= self.take_profit_pct else 0
                self.training_data_X.append(record["features"])
                self.training_data_y.append(label)
                self.prediction_buffer.remove(record)

        # ── (2) AUTO-RETRAIN: every 60 minutes if balanced classes available
        if self.Time >= self.next_model_train and len(self.training_data_X) > 20:
            X_np = np.array(self.training_data_X)
            y_np = np.array(self.training_data_y)

            wins   = np.sum(y_np == 1)
            losses = np.sum(y_np == 0)
            # Only retrain when at least 20 wins and 20 losses exist
            if wins < 20 or losses < 20:
                self.next_model_train = self.Time + timedelta(minutes=60)
                return

            X_1 = X_np[y_np == 1]
            X_0 = X_np[y_np == 0]

            mn  = min(len(X_1), len(X_0))
            X1r = resample(X_1, n_samples=mn, random_state=42)
            X0r = resample(X_0, n_samples=mn, random_state=42)
            X_bal = np.vstack((X1r, X0r))
            y_bal = np.array([1] * mn + [0] * mn)

            X_scaled = self.scaler.fit_transform(X_bal)
            self.model.fit(X_scaled, y_bal)
            self._is_model_trained = True

            # ── sliding window: keep most recent 1000 samples
            max_samples = 1000
            self.training_data_X = self.training_data_X[-max_samples:]
            self.training_data_y = self.training_data_y[-max_samples:]

            self.next_model_train = self.Time + timedelta(minutes=60)
            self.Log(f"Model retrained on {len(y_bal)} samples at {self.Time}")
            self.LogFeatureImportances()

        if not self._is_model_trained:
            return

        # ── (3) ML PREDICTION + TRADE EXECUTION ──────────────────
        sd = self.symbol_data[self.eth]
        if not sd.IsReady():
            return

        features = sd.GetFeatures()
        if features is None:
            return

        features_scaled = self.scaler.transform([features])
        prediction = self.model.predict(features_scaled)[0]

        # --- (Step 3) Exploration: 5% random buys even if prediction == 0
        if not self.Portfolio[self.eth].Invested:
            if random.random() < 0.05:
                prediction = 1

        # Buy if model (or exploration) predicts 1 and no existing position
        if prediction == 1 and not self.Portfolio[self.eth].Invested:
            free_cash  = self.Portfolio.Cash
            allocation = free_cash * self.capitalAllocation
            price      = self.Securities[self.eth].Price
            quantity   = int(allocation / price)
            if quantity > 0:
                self.MarketOrder(self.eth, quantity)
                self.entry_time = self.Time
                self.prediction_buffer.append({
                    "symbol": self.eth,
                    "features": features,
                    "entry_price": price,
                    "timestamp": self.Time
                })
                self.Debug(f"Placed BUY order for ETHUSD: qty={quantity}, price={price}, time={self.Time}")

        # ── (4) EXIT LOGIC: TP 7%, SL 3%, or time-based backstop (105 min)
        if self.Portfolio[self.eth].Invested and getattr(self, "entry_time", None) is not None:
            current_price = self.Securities[self.eth].Price
            avg_price     = self.Portfolio[self.eth].AveragePrice
            hold_minutes  = (self.Time - self.entry_time).total_seconds() / 60.0

            # Take-Profit: +7%
            if current_price >= avg_price * (1 + self.take_profit_pct):
                self.Liquidate(self.eth)
                self.Debug(f"TP exit at {self.Time}, price={current_price}")

            # Stop-Loss: -3%
            elif current_price <= avg_price * (1 - self.stop_loss_pct):
                self.Liquidate(self.eth)
                self.Debug(f"SL exit at {self.Time}, price={current_price}")

            # Time-based exit: hold ≥ 105 minutes
            elif hold_minutes >= self.time_based_exit_minutes:
                self.Liquidate(self.eth)
                self.Debug(f"Time-based exit at {self.Time}, price={current_price}")


    def LogFeatureImportances(self):
        """
        Logs the feature importances from the RandomForest model (7 features).
        """
        if not self._is_model_trained:
            return

        importances = self.model.feature_importances_
        names = [
            "avg_r",
            "std_r",
            "vol_ratio",
            "macd_val",
            "macd_hist",
            "ma_ratio",
            "pivot_dist"
        ]

        paired = list(zip(names, importances))
        paired.sort(key=lambda x: x[1], reverse=True)

        self.Log("=== Feature Importances ===")
        for name, score in paired:
            self.Log(f"{name.ljust(10)} : {score:.4f}")
        self.Log("===========================")


class SymbolData:
    def __init__(self, algo, symbol):
        self.symbol = symbol
        self.algo   = algo
        # Store tuples: (close, volume, high, low) for each 15-min bar
        self.history_15 = deque(maxlen=60)

        consolidator = TradeBarConsolidator(timedelta(minutes=15))
        consolidator.DataConsolidated += self.OnConsolidated
        algo.SubscriptionManager.AddConsolidator(symbol, consolidator)

        self.macd    = MovingAverageConvergenceDivergence(12, 26, 9, MovingAverageType.Exponential)
        self.fastSMA = SimpleMovingAverage(10)
        self.slowSMA = SimpleMovingAverage(30)

        algo.RegisterIndicator(symbol, self.macd, consolidator)
        algo.RegisterIndicator(symbol, self.fastSMA, consolidator)
        algo.RegisterIndicator(symbol, self.slowSMA, consolidator)

    def OnConsolidated(self, sender, bar: TradeBar):
        # Maintain history of 15-min (close, volume, high, low)
        self.history_15.append((bar.Close, bar.Volume, bar.High, bar.Low))

    def IsReady(self):
        return (
            len(self.history_15) >= 8
            and self.macd.IsReady
            and self.fastSMA.IsReady
            and self.slowSMA.IsReady
        )

    def GetFeatures(self):
        """
        Build the same 7 features in realtime:
          1) avg_r:  mean(log‐returns over last 4 bars)
          2) std_r:  std(log‐returns over last 4 bars)
          3) vol_ratio: sum(volume last 4 bars) / sum(volume last 8 bars)
          4) macd_val: current MACD line
          5) macd_hist: MACD‐hist (MACD minus signal)
          6) ma_ratio: last‐bar fast SMA (2 bars) / slow SMA (4 bars)
          7) pivot_dist: (close − pivot) / pivot, where pivot = (high+low+close)/3 of prior bar
        """
        if len(self.history_15) < 8:
            return None

        arr     = np.array(self.history_15)  # shape: (num_bars, 4)
        closes  = arr[:, 0]
        volumes = arr[:, 1]
        highs   = arr[:, 2]
        lows    = arr[:, 3]

        # last 4 bars for returns & volume ratio
        window_closes = closes[-4:]
        window_vols   = volumes[-4:]

        # (A) Log-returns over last 4 bars
        returns_4 = np.diff(np.log(window_closes))
        avg_r = np.mean(returns_4) if len(returns_4) > 0 else 0
        std_r = np.std(returns_4) if len(returns_4) > 0 else 0

        # (B) Volume ratio
        vol4 = np.sum(window_vols)
        vol8 = np.sum(volumes[-8:]) if np.sum(volumes[-8:]) != 0 else 1
        vol_ratio = vol4 / vol8

        # (C) MACD & MACD‐hist
        macd_val = self.macd.Current.Value
        macd_hist = macd_val - self.macd.Signal.Current.Value

        # (D) SMA ratio (fast vs slow)
        fast_val = self.fastSMA.Current.Value
        slow_val = self.slowSMA.Current.Value if self.slowSMA.Current.Value != 0 else 1e-6
        ma_ratio = fast_val / slow_val

        # (E) Pivot distance (use previous bar’s high/low/close)
        prev_high  = highs[-2]
        prev_low   = lows[-2]
        prev_close = closes[-2]
        pivot = (prev_high + prev_low + prev_close) / 3 if (prev_high + prev_low + prev_close) != 0 else 0
        currp = window_closes[-1]
        pivot_dist = (currp - pivot) / pivot if pivot != 0 else 0

        return [
            avg_r,
            std_r,
            vol_ratio,
            macd_val,
            macd_hist,
            ma_ratio,
            pivot_dist
        ]
