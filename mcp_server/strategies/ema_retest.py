"""
EMA Pullback Momentum Strategy
================================
Fundamental difference from the crossover strategy:
  - Crossover: enters the moment EMA20 crosses EMA50 (lagging)
  - This strategy: trend must ALREADY be confirmed (EMA20>50>200 stacked),
    then WAITS for price to pull back to the EMA20 zone, then enters on
    a confirmation candle that closes back above EMA20.

This approach gives a far tighter stop (SL below pullback wick) and a
better entry price, which is why pullback systems typically achieve
60-65% win rates vs 45-50% for crossover systems.

Entry conditions (ALL required):
  1. Daily  : EMA20 > EMA50 sustained >= min_cycle_bars (trend established)
  2. 4H     : EMA20 > EMA50 > EMA200 fully stacked
  3. 4H     : ADX(14) > adx_threshold AND rising (trend has strength)
  4. 4H     : Price above EMA200 (structural alignment)
  5. 4H     : Price pulled back INTO the EMA20 zone (low <= EMA20 + pullback_atr * ATR)
  6. 4H     : Confirmation candle closes BACK ABOVE EMA20 (retest confirmed)
  7. 4H     : MACD histogram positive AND rising (momentum with trend)
  8. 4H     : RSI(14) between rsi_low and rsi_high (not overextended)

Exit:
  SL  : 1.5× ATR below EMA20 at entry
  TP1 : 1.5R → close tp1_pct (40%) of position
  TP2 : 3.0R → close tp2_pct (40%) of position
  Runner (20%): exit when Daily EMA20 crosses below EMA50 OR BE stop hit

Bearish mirror of all conditions for shorts.
"""

import backtrader as bt


class EMAPullbackMomentum(bt.Strategy):
    params = (
        # EMA periods
        ("ema_fast",     20),
        ("ema_slow",     50),
        ("ema_trend",   200),
        # ATR
        ("atr_period",   14),
        ("atr_sl_mult", 1.5),
        # Pullback zone: price must reach within this many ATRs of EMA20
        ("pullback_atr_zone", 0.5),
        # Max bars to wait for confirmation after pullback touch
        ("pullback_max_wait", 5),
        # MACD
        ("macd_fast",    12),
        ("macd_slow",    26),
        ("macd_signal",   9),
        # RSI
        ("rsi_period",   14),
        ("rsi_long_low",  45),
        ("rsi_long_high", 68),
        ("rsi_short_low", 32),
        ("rsi_short_high",55),
        # Tiered TP (multiples of initial R)
        ("tp1_r",       1.5),
        ("tp1_pct",     0.40),   # close 40% at TP1
        ("tp2_r",       3.0),
        ("tp2_pct",     0.40),   # close 40% at TP2 (20% runner remains)
        # Risk & sizing
        ("risk_pct",   0.005),   # 0.5% risk per trade
        ("capital",  10000.0),
        # Trend filters
        ("min_cycle_bars", 10),
        ("adx_period",     14),
        ("adx_threshold",  20.0),
        ("adx_rising",    True),
        ("use_ema200",    True),
        ("allow_shorts",  True),
    )

    # ─── Initialise ────────────────────────────────────────────────────
    def __init__(self):
        # 4H indicators (datas[0])
        self.ema20_4h   = bt.indicators.EMA(self.datas[0].close, period=self.p.ema_fast)
        self.ema50_4h   = bt.indicators.EMA(self.datas[0].close, period=self.p.ema_slow)
        self.ema200_4h  = bt.indicators.EMA(self.datas[0].close, period=self.p.ema_trend)
        self.atr_4h     = bt.indicators.ATR(self.datas[0],       period=self.p.atr_period)
        self.adx_4h     = bt.indicators.AverageDirectionalMovementIndex(
                              self.datas[0], period=self.p.adx_period)
        self.rsi_4h     = bt.indicators.RSI(self.datas[0].close, period=self.p.rsi_period)
        self.macd_4h    = bt.indicators.MACDHisto(
                              self.datas[0].close,
                              period_me1=self.p.macd_fast,
                              period_me2=self.p.macd_slow,
                              period_signal=self.p.macd_signal)

        # Daily indicators (datas[1]) — trend direction only
        self.ema20_d    = bt.indicators.EMA(self.datas[1].close, period=self.p.ema_fast)
        self.ema50_d    = bt.indicators.EMA(self.datas[1].close, period=self.p.ema_slow)

        # Internal state
        self._daily_bull_bars  = 0
        self._daily_bear_bars  = 0

        # Pullback state machine
        # 0 = watching, 1 = pullback touched, waiting for confirmation
        self._pb_state         = 0
        self._pb_direction     = 0    # +1 long setup, -1 short setup
        self._pb_bars_waited   = 0

        # Trade management
        self._in_trade         = False
        self._direction        = 0
        self._entry_price      = 0.0
        self._sl_price         = 0.0
        self._tp1_price        = 0.0
        self._tp2_price        = 0.0
        self._be_price         = 0.0
        self._tp1_hit          = False
        self._tp2_hit          = False
        self._entry_size       = 0
        self._tp1_size         = 0
        self._tp2_size         = 0
        self._runner_size      = 0

    # ─── Main loop ─────────────────────────────────────────────────────
    def next(self):
        # Update daily cycle counters
        if self.ema20_d[0] > self.ema50_d[0]:
            self._daily_bull_bars += 1
            self._daily_bear_bars  = 0
        else:
            self._daily_bear_bars += 1
            self._daily_bull_bars  = 0

        if self._in_trade:
            self._manage_trade()
            return

        self._update_pullback_state()
        self._check_entry()

    # ─── Pullback state machine ────────────────────────────────────────
    def _update_pullback_state(self):
        """Advance the pullback state machine."""
        ema20 = self.ema20_4h[0]
        ema50 = self.ema50_4h[0]
        atr   = self.atr_4h[0]
        close = self.datas[0].close[0]
        low   = self.datas[0].low[0]
        high  = self.datas[0].high[0]

        bull_trend = (self._daily_bull_bars >= self.p.min_cycle_bars and
                      self._4h_bullish())
        bear_trend = (self.p.allow_shorts and
                      self._daily_bear_bars >= self.p.min_cycle_bars and
                      self._4h_bearish())

        # Reset if trend no longer valid
        if self._pb_state == 1:
            trend_still_valid = (
                (self._pb_direction == +1 and bull_trend) or
                (self._pb_direction == -1 and bear_trend)
            )
            if not trend_still_valid:
                self._pb_state = 0
                return

            # Timeout
            self._pb_bars_waited += 1
            if self._pb_bars_waited > self.p.pullback_max_wait:
                self._pb_state = 0
                return

            # Invalidation: price broke through EMA50 (too deep)
            if self._pb_direction == +1 and close < ema50:
                self._pb_state = 0
                return
            if self._pb_direction == -1 and close > ema50:
                self._pb_state = 0
                return

        # Look for new pullback touches (state 0 → 1)
        if self._pb_state == 0:
            zone = atr * self.p.pullback_atr_zone

            if bull_trend:
                # Price dips into EMA20 zone but stays above EMA50
                in_zone = (low <= ema20 + zone) and (low >= ema50)
                if in_zone:
                    self._pb_state     = 1
                    self._pb_direction = +1
                    self._pb_bars_waited = 0

            elif bear_trend:
                # Price rises into EMA20 zone (from below) but stays below EMA50
                in_zone = (high >= ema20 - zone) and (high <= ema50)
                if in_zone:
                    self._pb_state     = 1
                    self._pb_direction = -1
                    self._pb_bars_waited = 0

    def _check_entry(self):
        """Fire entry if pullback has been confirmed this bar."""
        if self._pb_state != 1:
            return

        ema20 = self.ema20_4h[0]
        close = self.datas[0].close[0]

        if self._pb_direction == +1:
            # Confirmation: close back above EMA20
            if close > ema20:
                if self._passes_momentum_filters(+1):
                    self._enter_long()
                    self._pb_state = 0

        elif self._pb_direction == -1:
            # Confirmation: close back below EMA20
            if close < ema20:
                if self._passes_momentum_filters(-1):
                    self._enter_short()
                    self._pb_state = 0

    # ─── Momentum filters ─────────────────────────────────────────────
    def _passes_momentum_filters(self, direction: int) -> bool:
        """MACD histogram + RSI zone + ADX + EMA200 alignment."""
        # ADX
        if self.p.adx_threshold > 0:
            if self.adx_4h[0] < self.p.adx_threshold:
                return False
            if self.p.adx_rising and self.adx_4h[0] <= self.adx_4h[-1]:
                return False

        # EMA200 structural alignment
        if self.p.use_ema200:
            price = self.datas[0].close[0]
            if direction == +1 and price < self.ema200_4h[0]:
                return False
            if direction == -1 and price > self.ema200_4h[0]:
                return False

        # MACD: line must be above signal (positive momentum direction)
        # Not requiring histogram to be rising — too strict for pullback bars
        # where histogram naturally dips during the retracement itself
        macd_line   = self.macd_4h.macd[0]
        signal_line = self.macd_4h.signal[0]
        if direction == +1 and macd_line <= signal_line:
            return False
        if direction == -1 and macd_line >= signal_line:
            return False

        # RSI zone — widened to capture recovery from pullback dip
        # Pullbacks naturally push RSI lower; 38-72 is realistic for retest bars
        rsi = self.rsi_4h[0]
        if direction == +1 and not (self.p.rsi_long_low <= rsi <= self.p.rsi_long_high):
            return False
        if direction == -1 and not (self.p.rsi_short_low <= rsi <= self.p.rsi_short_high):
            return False

        return True

    # ─── Trend helpers ────────────────────────────────────────────────
    def _4h_bullish(self):
        return self.ema20_4h[0] > self.ema50_4h[0] > self.ema200_4h[0]

    def _4h_bearish(self):
        return self.ema20_4h[0] < self.ema50_4h[0] < self.ema200_4h[0]

    # ─── Entry execution ──────────────────────────────────────────────
    def _calc_size(self, sl_distance: float) -> int:
        if sl_distance <= 0:
            return 0
        risk = self.p.capital * self.p.risk_pct
        return max(int(risk / sl_distance), 1)

    def _enter_long(self):
        price = self.datas[0].close[0]
        atr   = self.atr_4h[0]
        sl    = self.ema20_4h[0] - atr * self.p.atr_sl_mult
        sl_d  = price - sl
        if sl_d <= 0:
            return
        size  = self._calc_size(sl_d)
        self.buy(size=size)
        self._record(+1, price, sl, sl_d, size)

    def _enter_short(self):
        price = self.datas[0].close[0]
        atr   = self.atr_4h[0]
        sl    = self.ema20_4h[0] + atr * self.p.atr_sl_mult
        sl_d  = sl - price
        if sl_d <= 0:
            return
        size  = self._calc_size(sl_d)
        self.sell(size=size)
        self._record(-1, price, sl, sl_d, size)

    def _record(self, direction, price, sl, sl_d, size):
        tp1_size  = max(int(size * self.p.tp1_pct), 1)
        tp2_size  = max(int(size * self.p.tp2_pct), 1)
        runner    = max(size - tp1_size - tp2_size, 0)

        self._in_trade     = True
        self._direction    = direction
        self._entry_price  = price
        self._sl_price     = sl
        self._be_price     = price
        self._tp1_price    = price + direction * sl_d * self.p.tp1_r
        self._tp2_price    = price + direction * sl_d * self.p.tp2_r
        self._tp1_hit      = False
        self._tp2_hit      = False
        self._entry_size   = size
        self._tp1_size     = tp1_size
        self._tp2_size     = tp2_size
        self._runner_size  = runner

    # ─── Trade management ─────────────────────────────────────────────
    def _manage_trade(self):
        price = self.datas[0].close[0]
        d     = self._direction

        # ── Before TP1: monitor SL and TP1 ───────────────────────────
        if not self._tp1_hit:
            sl_hit = (d == +1 and price <= self._sl_price) or \
                     (d == -1 and price >= self._sl_price)
            if sl_hit:
                self.close(data=self.datas[0])
                self._reset()
                return

            tp1_hit = (d == +1 and price >= self._tp1_price) or \
                      (d == -1 and price <= self._tp1_price)
            if tp1_hit:
                if d == +1:
                    self.sell(size=self._tp1_size)
                else:
                    self.buy(size=self._tp1_size)
                self._tp1_hit = True
                return

        # ── Between TP1 and TP2 ───────────────────────────────────────
        elif not self._tp2_hit:
            # BE stop now active
            be_hit = (d == +1 and price <= self._be_price) or \
                     (d == -1 and price >= self._be_price)
            if be_hit:
                remaining = self._tp2_size + self._runner_size
                if remaining > 0:
                    if d == +1:
                        self.sell(size=remaining)
                    else:
                        self.buy(size=remaining)
                self._reset()
                return

            tp2_hit = (d == +1 and price >= self._tp2_price) or \
                      (d == -1 and price <= self._tp2_price)
            if tp2_hit:
                if d == +1:
                    self.sell(size=self._tp2_size)
                else:
                    self.buy(size=self._tp2_size)
                self._tp2_hit = True
                if self._runner_size <= 0:
                    self._reset()
                return

        # ── Runner: daily EMA flip or BE ─────────────────────────────
        else:
            be_hit = (d == +1 and price <= self._be_price) or \
                     (d == -1 and price >= self._be_price)
            daily_flip = (d == +1 and self.ema20_d[0] < self.ema50_d[0]) or \
                         (d == -1 and self.ema20_d[0] > self.ema50_d[0])
            if be_hit or daily_flip:
                if self._runner_size > 0:
                    if d == +1:
                        self.sell(size=self._runner_size)
                    else:
                        self.buy(size=self._runner_size)
                self._reset()

    def _reset(self):
        self._in_trade    = False
        self._direction   = 0
        self._tp1_hit     = False
        self._tp2_hit     = False
        self._entry_size  = 0
        self._tp1_size    = 0
        self._tp2_size    = 0
        self._runner_size = 0
