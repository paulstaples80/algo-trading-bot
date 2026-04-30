"""
Multi-Timeframe EMA 20/50 Crossover Strategy
=============================================

Rules:
  Daily  : EMA20 > EMA50 (bullish) sustained ≥ min_cycle_bars as a proxy for
            "2 cycles of Phase 1 / Phase 2 cyclicity"
  4H     : EMA20 > EMA50 > EMA200 fully stacked (bullish)
  1H     : EMA20 > EMA50 (aligned with higher TFs) → entry

  Bearish mirror: all conditions reversed, short entries.

Trade management:
  SL  = entry ± (ATR14_1H × atr_sl_mult)
  TP1 = entry ± (SL_distance × tp_rr)  → close tp1_close_pct (75%) of position
  Runner (remaining 25%): stop moved to BE after TP1
  Runner exit: BE stop hit  OR  Daily EMA20 crosses opposite direction
"""

import backtrader as bt


class MultiTFEmaCross(bt.Strategy):
    params = (
        ("ema_fast", 20),
        ("ema_slow", 50),
        ("ema_trend", 200),
        ("atr_period", 14),
        ("atr_sl_mult", 1.5),
        ("tp_rr", 2.0),
        ("tp1_close_pct", 0.75),
        ("risk_pct", 0.01),
        ("capital", 10000.0),
        ("min_cycle_bars", 10),
        ("allow_shorts", True),
        # ── Additional filters (set to 0/False to disable) ────────────
        ("use_ema200_daily", False),  # Long only above Daily EMA200
        ("adx_period", 14),
        ("adx_threshold", 0.0),       # 0 = disabled; 20+ = active
        ("adx_rising", False),        # Require ADX slope to be positive
        ("atr_regime_bars", 0),       # 0 = disabled; 50 = use 50-bar ATR pct
        ("atr_regime_low", 20.0),     # Ignore bottom N-pct ATR (dead market)
        ("atr_regime_high", 80.0),    # Ignore top N-pct ATR (extreme volatility)
        ("session_filter", False),    # Only trade London / NY sessions
    )

    def __init__(self):
        # ── 4H indicators (datas[0]) — entry, stacking, ADX, ATR ────────
        # All optional filters use 4H indicators to avoid the long warmup
        # that daily EMA200 (200 daily bars ≈ 10 months) would impose.
        self.ema20_4h  = bt.indicators.EMA(self.datas[0].close, period=self.p.ema_fast)
        self.ema50_4h  = bt.indicators.EMA(self.datas[0].close, period=self.p.ema_slow)
        self.ema200_4h = bt.indicators.EMA(self.datas[0].close, period=self.p.ema_trend)
        self.atr_4h    = bt.indicators.ATR(self.datas[0], period=self.p.atr_period)
        self.adx_4h    = bt.indicators.AverageDirectionalMovementIndex(
                             self.datas[0], period=self.p.adx_period)

        # ── Daily indicators (datas[1]) — trend direction only ───────
        # Only EMA20/50 used here (needs 50 daily bars ≈ 10 weeks — fast to init)
        self.ema20_d = bt.indicators.EMA(self.datas[1].close, period=self.p.ema_fast)
        self.ema50_d = bt.indicators.EMA(self.datas[1].close, period=self.p.ema_slow)

        # Rolling ATR percentile window (volatility regime filter)
        self._atr_window = []

        # Internal state
        self._daily_bull_bars  = 0   # consecutive bars EMA20>EMA50 on daily
        self._daily_bear_bars  = 0
        self._in_trade         = False
        self._trade_direction  = 0   # +1 long, -1 short
        self._entry_price      = 0.0
        self._sl_price         = 0.0
        self._tp1_price        = 0.0
        self._be_price         = 0.0
        self._tp1_hit          = False
        self._entry_size       = 0
        self._runner_size      = 0

    # ─────────────────────────────────────────────────────────────────
    def next(self):
        # Update daily cycle counters (only changes when daily bar closes)
        if self.ema20_d[0] > self.ema50_d[0]:
            self._daily_bull_bars += 1
            self._daily_bear_bars  = 0
        else:
            self._daily_bear_bars += 1
            self._daily_bull_bars  = 0

        # ── Manage open trade ────────────────────────────────────────
        if self._in_trade:
            self._manage_open_trade()
            return

        # ── Look for entries ─────────────────────────────────────────
        # ── Maintain ATR regime window ────────────────────────────────
        if self.p.atr_regime_bars > 0:
            self._atr_window.append(self.atr_4h[0])
            if len(self._atr_window) > self.p.atr_regime_bars:
                self._atr_window.pop(0)

        if self._daily_bull_bars >= self.p.min_cycle_bars:
            if self._4h_bullish() and self._entry_bullish():
                if self._passes_filters(direction=1):
                    self._enter_long()

        elif self.p.allow_shorts and self._daily_bear_bars >= self.p.min_cycle_bars:
            if self._4h_bearish() and self._entry_bearish():
                if self._passes_filters(direction=-1):
                    self._enter_short()

    # ─────────────────────────────────────────────────────────────────
    def _passes_filters(self, direction: int) -> bool:
        """Return True if all active optional filters pass for the given direction."""

        # ── EMA200 alignment (uses 4H EMA200 — always has sufficient bars) ──
        if self.p.use_ema200_daily:
            price_4h = self.datas[0].close[0]
            ema200   = self.ema200_4h[0]
            if direction == +1 and price_4h < ema200:
                return False
            if direction == -1 and price_4h > ema200:
                return False

        # ── ADX threshold (uses 4H ADX — consistent with all other filters) ──
        if self.p.adx_threshold > 0:
            adx_val = self.adx_4h[0]
            if adx_val < self.p.adx_threshold:
                return False
            if self.p.adx_rising and len(self.adx_4h) > 1:
                if self.adx_4h[0] <= self.adx_4h[-1]:
                    return False

        # ── ATR volatility regime ─────────────────────────────────────
        if self.p.atr_regime_bars > 0 and len(self._atr_window) >= self.p.atr_regime_bars:
            cur_atr = self.atr_4h[0]
            sorted_w = sorted(self._atr_window)
            n = len(sorted_w)
            low_cut  = sorted_w[int(n * self.p.atr_regime_low  / 100)]
            high_cut = sorted_w[int(min(n - 1, int(n * self.p.atr_regime_high / 100)))]
            if cur_atr < low_cut or cur_atr > high_cut:
                return False

        # ── Session filter (London 07-17 UTC, NY overlap 13-21 UTC) ──
        if self.p.session_filter:
            bar_hour = self.datas[0].datetime.datetime(0).hour
            in_london = 7 <= bar_hour < 17
            in_ny     = 13 <= bar_hour < 21
            if not (in_london or in_ny):
                return False

        return True

    def _4h_bullish(self):
        """EMA20 > EMA50 > EMA200 fully stacked bullish on 4H."""
        return (self.ema20_4h[0] > self.ema50_4h[0] > self.ema200_4h[0])

    def _4h_bearish(self):
        """EMA20 < EMA50 < EMA200 fully stacked bearish on 4H."""
        return (self.ema20_4h[0] < self.ema50_4h[0] < self.ema200_4h[0])

    def _entry_bullish(self):
        """EMA20 > EMA50 on 4H — entry signal (proxy for 1H alignment)."""
        return self.ema20_4h[0] > self.ema50_4h[0]

    def _entry_bearish(self):
        """EMA20 < EMA50 on 4H — entry signal (proxy for 1H alignment)."""
        return self.ema20_4h[0] < self.ema50_4h[0]

    # ─────────────────────────────────────────────────────────────────
    def _calc_position_size(self, sl_distance: float) -> int:
        """Risk 1% of capital; position size = risk_amount / sl_distance."""
        if sl_distance <= 0:
            return 0
        risk_amount = self.p.capital * self.p.risk_pct
        size = int(risk_amount / sl_distance)
        return max(size, 1)

    def _enter_long(self):
        price = self.datas[0].close[0]
        atr   = self.atr_4h[0]
        if atr <= 0:
            return

        sl  = price - atr * self.p.atr_sl_mult
        sl_dist = price - sl
        tp1 = price + sl_dist * self.p.tp_rr

        size = self._calc_position_size(sl_dist)
        if size < 1:
            return

        self.buy(size=size)
        self._record_trade(+1, price, sl, tp1, size)

    def _enter_short(self):
        price = self.datas[0].close[0]
        atr   = self.atr_4h[0]
        if atr <= 0:
            return

        sl  = price + atr * self.p.atr_sl_mult
        sl_dist = sl - price
        tp1 = price - sl_dist * self.p.tp_rr

        size = self._calc_position_size(sl_dist)
        if size < 1:
            return

        self.sell(size=size)
        self._record_trade(-1, price, sl, tp1, size)

    def _record_trade(self, direction, price, sl, tp1, size):
        self._in_trade        = True
        self._trade_direction = direction
        self._entry_price     = price
        self._sl_price        = sl
        self._tp1_price       = tp1
        self._be_price        = price
        self._tp1_hit         = False
        self._entry_size      = size
        self._runner_size     = int(size * (1 - self.p.tp1_close_pct))

    # ─────────────────────────────────────────────────────────────────
    def _manage_open_trade(self):
        price = self.datas[0].close[0]
        d     = self._trade_direction

        if not self._tp1_hit:
            # ── Check SL ─────────────────────────────────────────────
            sl_hit = (d == +1 and price <= self._sl_price) or \
                     (d == -1 and price >= self._sl_price)
            if sl_hit:
                self._close_all()
                return

            # ── Check TP1 ─────────────────────────────────────────────
            tp1_hit = (d == +1 and price >= self._tp1_price) or \
                      (d == -1 and price <= self._tp1_price)
            if tp1_hit:
                close_size = self._entry_size - self._runner_size
                if d == +1:
                    self.sell(size=close_size)
                else:
                    self.buy(size=close_size)
                self._tp1_hit = True
                return

        else:
            # ── Runner management: BE stop ────────────────────────────
            be_hit = (d == +1 and price <= self._be_price) or \
                     (d == -1 and price >= self._be_price)
            if be_hit:
                self._close_runner()
                return

            # ── Runner management: Daily EMA flip ────────────────────
            daily_flip = (d == +1 and self.ema20_d[0] < self.ema50_d[0]) or \
                         (d == -1 and self.ema20_d[0] > self.ema50_d[0])
            if daily_flip:
                self._close_runner()
                return

    def _close_all(self):
        self.close(data=self.datas[0])
        self._reset_state()

    def _close_runner(self):
        if self._runner_size > 0:
            if self._trade_direction == +1:
                self.sell(size=self._runner_size)
            else:
                self.buy(size=self._runner_size)
        self._reset_state()

    def _reset_state(self):
        self._in_trade        = False
        self._trade_direction = 0
        self._entry_price     = 0.0
        self._sl_price        = 0.0
        self._tp1_price       = 0.0
        self._be_price        = 0.0
        self._tp1_hit         = False
        self._entry_size      = 0
        self._runner_size     = 0
