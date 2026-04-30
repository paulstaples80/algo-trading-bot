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
        ("ema_trend", 200),       # 4H trend EMA
        ("atr_period", 14),
        ("atr_sl_mult", 1.5),     # SL = ATR × this
        ("tp_rr", 2.0),           # TP at 2:1 RR
        ("tp1_close_pct", 0.75),  # Close 75% at TP1
        ("risk_pct", 0.01),       # 1% risk per trade
        ("capital", 10000.0),     # Account capital in £
        ("min_cycle_bars", 10),   # Daily EMA cross must be sustained ≥ N bars
        ("allow_shorts", True),   # Trade both directions
    )

    def __init__(self):
        # ── 4H indicators (datas[0]) — entry + stacking ──────────────
        # Note: TradingView limits 1H forex history to ~10 months.
        # 4H data gives 3+ years, so 4H is used as the execution TF.
        # EMA20/50 on 4H acts as the "1H aligned" condition.
        # EMA20/50/200 on 4H is the stacking check.
        self.ema20_4h  = bt.indicators.EMA(self.datas[0].close, period=self.p.ema_fast)
        self.ema50_4h  = bt.indicators.EMA(self.datas[0].close, period=self.p.ema_slow)
        self.ema200_4h = bt.indicators.EMA(self.datas[0].close, period=self.p.ema_trend)
        self.atr_4h    = bt.indicators.ATR(self.datas[0], period=self.p.atr_period)

        # ── Daily indicators (datas[1]) — trend + cyclicity ──────────
        self.ema20_d = bt.indicators.EMA(self.datas[1].close, period=self.p.ema_fast)
        self.ema50_d = bt.indicators.EMA(self.datas[1].close, period=self.p.ema_slow)

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
        if self._daily_bull_bars >= self.p.min_cycle_bars:
            if self._4h_bullish() and self._entry_bullish():
                self._enter_long()

        elif self.p.allow_shorts and self._daily_bear_bars >= self.p.min_cycle_bars:
            if self._4h_bearish() and self._entry_bearish():
                self._enter_short()

    # ─────────────────────────────────────────────────────────────────
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
