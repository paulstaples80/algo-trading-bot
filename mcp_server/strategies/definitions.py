import backtrader as bt


class SmaCross(bt.Strategy):
    params = (
        ("fast_period", 10),
        ("slow_period", 30),
        ("size_pct", 0.95),
    )

    def __init__(self):
        self.fast = bt.indicators.SMA(self.data.close, period=self.p.fast_period)
        self.slow = bt.indicators.SMA(self.data.close, period=self.p.slow_period)
        self.cross = bt.indicators.CrossOver(self.fast, self.slow)

    def next(self):
        if not self.position:
            if self.cross > 0:
                size = int(self.broker.getcash() * self.p.size_pct / self.data.close[0])
                if size > 0:
                    self.buy(size=size)
        elif self.cross < 0:
            self.close()


class EmaCross(bt.Strategy):
    params = (
        ("fast_period", 9),
        ("slow_period", 21),
        ("atr_stop_mult", 2.0),
        ("size_pct", 0.95),
    )

    def __init__(self):
        self.fast = bt.indicators.EMA(self.data.close, period=self.p.fast_period)
        self.slow = bt.indicators.EMA(self.data.close, period=self.p.slow_period)
        self.cross = bt.indicators.CrossOver(self.fast, self.slow)
        self.atr = bt.indicators.ATR(self.data, period=14)
        self.stop_price = None

    def next(self):
        if not self.position:
            if self.cross > 0:
                size = int(self.broker.getcash() * self.p.size_pct / self.data.close[0])
                if size > 0:
                    self.buy(size=size)
                    if self.p.atr_stop_mult > 0:
                        self.stop_price = self.data.close[0] - self.atr[0] * self.p.atr_stop_mult
        else:
            if self.cross < 0:
                self.close()
                self.stop_price = None
            elif self.p.atr_stop_mult > 0 and self.stop_price and self.data.close[0] < self.stop_price:
                self.close()
                self.stop_price = None


class RsiMeanReversion(bt.Strategy):
    params = (
        ("rsi_period", 14),
        ("oversold", 30),
        ("overbought", 70),
        ("size_pct", 0.95),
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)

    def next(self):
        if not self.position:
            if self.rsi < self.p.oversold:
                size = int(self.broker.getcash() * self.p.size_pct / self.data.close[0])
                if size > 0:
                    self.buy(size=size)
        elif self.rsi > self.p.overbought:
            self.close()


class MacdStrategy(bt.Strategy):
    params = (
        ("fast_period", 12),
        ("slow_period", 26),
        ("signal_period", 9),
        ("size_pct", 0.95),
    )

    def __init__(self):
        macd = bt.indicators.MACD(
            self.data.close,
            period1=self.p.fast_period,
            period2=self.p.slow_period,
            period_signal=self.p.signal_period,
        )
        self.macd_line = macd.macd
        self.signal_line = macd.signal
        self.cross = bt.indicators.CrossOver(self.macd_line, self.signal_line)

    def next(self):
        if not self.position:
            if self.cross > 0:
                size = int(self.broker.getcash() * self.p.size_pct / self.data.close[0])
                if size > 0:
                    self.buy(size=size)
        elif self.cross < 0:
            self.close()


class BollingerBand(bt.Strategy):
    params = (
        ("period", 20),
        ("devfactor", 2.0),
        ("size_pct", 0.95),
    )

    def __init__(self):
        self.bb = bt.indicators.BollingerBands(
            self.data.close, period=self.p.period, devfactor=self.p.devfactor
        )

    def next(self):
        if not self.position:
            if self.data.close[0] < self.bb.lines.bot[0]:
                size = int(self.broker.getcash() * self.p.size_pct / self.data.close[0])
                if size > 0:
                    self.buy(size=size)
        elif self.data.close[0] > self.bb.lines.mid[0]:
            self.close()


STRATEGY_REGISTRY = {
    "SmaCross": {
        "class": SmaCross,
        "description": "SMA crossover — buy when fast SMA crosses above slow SMA",
        "params": {
            "fast_period": {"type": "int", "default": 10, "low": 3, "high": 50, "desc": "Fast SMA period"},
            "slow_period": {"type": "int", "default": 30, "low": 10, "high": 200, "desc": "Slow SMA period"},
            "size_pct": {"type": "float", "default": 0.95, "low": 0.5, "high": 1.0, "desc": "Fraction of cash to deploy"},
        },
    },
    "EmaCross": {
        "class": EmaCross,
        "description": "EMA crossover with optional ATR-based trailing stop",
        "params": {
            "fast_period": {"type": "int", "default": 9, "low": 3, "high": 50, "desc": "Fast EMA period"},
            "slow_period": {"type": "int", "default": 21, "low": 10, "high": 200, "desc": "Slow EMA period"},
            "atr_stop_mult": {"type": "float", "default": 2.0, "low": 0.0, "high": 5.0, "desc": "ATR stop multiplier (0 = disabled)"},
            "size_pct": {"type": "float", "default": 0.95, "low": 0.5, "high": 1.0, "desc": "Fraction of cash to deploy"},
        },
    },
    "RsiMeanReversion": {
        "class": RsiMeanReversion,
        "description": "Buy on RSI oversold, sell on RSI overbought",
        "params": {
            "rsi_period": {"type": "int", "default": 14, "low": 5, "high": 30, "desc": "RSI lookback period"},
            "oversold": {"type": "int", "default": 30, "low": 10, "high": 45, "desc": "Oversold threshold to buy"},
            "overbought": {"type": "int", "default": 70, "low": 55, "high": 90, "desc": "Overbought threshold to sell"},
            "size_pct": {"type": "float", "default": 0.95, "low": 0.5, "high": 1.0, "desc": "Fraction of cash to deploy"},
        },
    },
    "MacdStrategy": {
        "class": MacdStrategy,
        "description": "MACD line / signal line crossover",
        "params": {
            "fast_period": {"type": "int", "default": 12, "low": 5, "high": 30, "desc": "MACD fast EMA period"},
            "slow_period": {"type": "int", "default": 26, "low": 15, "high": 60, "desc": "MACD slow EMA period"},
            "signal_period": {"type": "int", "default": 9, "low": 3, "high": 20, "desc": "MACD signal EMA period"},
            "size_pct": {"type": "float", "default": 0.95, "low": 0.5, "high": 1.0, "desc": "Fraction of cash to deploy"},
        },
    },
    "BollingerBand": {
        "class": BollingerBand,
        "description": "Buy when price touches lower BB, sell when price returns to midline",
        "params": {
            "period": {"type": "int", "default": 20, "low": 10, "high": 50, "desc": "BB lookback period"},
            "devfactor": {"type": "float", "default": 2.0, "low": 1.0, "high": 3.5, "desc": "Standard deviation multiplier for bands"},
            "size_pct": {"type": "float", "default": 0.95, "low": 0.5, "high": 1.0, "desc": "Fraction of cash to deploy"},
        },
    },
}
