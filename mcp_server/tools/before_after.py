"""
Before / After comparison:
  BEFORE : MultiTFEmaCross — Config C (0.5% risk + EMA200 + ADX>20)
  AFTER  : EMAPullbackMomentum — pullback retest + MACD + RSI + tiered TP
"""

import io, sys
import pandas as pd
import backtrader as bt
from dateutil.relativedelta import relativedelta

from .tradingview import tv_get_bars
from .multi_tf_backtest import _resample_ohlcv, _ProfitFactor
from ..strategies.multi_tf_ema import MultiTFEmaCross
from ..strategies.ema_retest import EMAPullbackMomentum


def _run(strategy_class, df: pd.DataFrame, params: dict,
         capital: float, commission: float) -> dict:
    """Generic cerebro runner for 2-feed strategies (4H + Daily)."""
    d = df.copy()
    d["datetime"] = pd.to_datetime(d["datetime"])
    d = d.set_index("datetime")[["open","high","low","close","volume"]].dropna()
    d_daily = _resample_ohlcv(d, "1D")

    if len(d) < 210 or len(d_daily) < 55:
        return {"error": f"Insufficient bars: 4H={len(d)}, Daily={len(d_daily)}"}

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(capital)
    cerebro.broker.setcommission(commission=commission)
    cerebro.adddata(bt.feeds.PandasData(dataname=d),       name="4H")
    cerebro.adddata(bt.feeds.PandasData(dataname=d_daily), name="Daily")
    cerebro.addstrategy(strategy_class, **params)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe",
                        riskfreerate=0.02, annualize=True,
                        timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.DrawDown,      _name="dd")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.Returns,       _name="returns")
    cerebro.addanalyzer(_ProfitFactor,              _name="pf")

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        results = cerebro.run()
    finally:
        sys.stdout = old_stdout

    s     = results[0]
    final = s.broker.getvalue()
    ret   = round((final - capital) / capital * 100, 2)

    sharpe_raw = s.analyzers.sharpe.get_analysis().get("sharperatio") or 0.0
    sharpe     = round(float(sharpe_raw), 4)

    dd   = s.analyzers.dd.get_analysis()
    max_dd = round(dd.get("max", {}).get("drawdown", 0.0) or 0.0, 2)

    ta   = s.analyzers.trades.get_analysis()
    total  = ta.get("total", {}).get("closed", 0) or 0
    won    = ta.get("won",   {}).get("total",  0) or 0
    wr     = round(won / total * 100, 2) if total > 0 else 0.0

    pf_val = s.analyzers.pf.get_analysis().get("profit_factor", 0.0)
    rnorm  = s.analyzers.returns.get_analysis().get("rnorm100", 0.0) or 0.0

    avg_win_r  = ta.get("won",  {}).get("pnl", {}).get("average", 0.0) or 0.0
    avg_loss_r = ta.get("lost", {}).get("pnl", {}).get("average", 0.0) or 0.0

    return {
        "final_value":          round(final, 2),
        "total_return_pct":     ret,
        "annualised_return_pct":round(rnorm, 2),
        "sharpe_ratio":         sharpe,
        "max_drawdown_pct":     max_dd,
        "total_trades":         total,
        "win_rate_pct":         wr,
        "profit_factor":        round(pf_val, 4),
        "avg_win":              round(avg_win_r, 2),
        "avg_loss":             round(avg_loss_r, 2),
    }


def bt_before_after(
    symbol:       str   = "EURUSD",
    exchange:     str   = "OANDA",
    n_bars:       int   = 5000,
    oos_months:   int   = 6,
    capital:      float = 10000.0,
    commission:   float = 0.00007,
) -> dict:
    """Run Config-C (before) vs EMAPullbackMomentum (after) on the same data.

    Returns a side-by-side comparison with backtest and OOS metrics.
    """
    bar_data = tv_get_bars(symbol.upper(), exchange, "4h", n_bars)
    if "error" in bar_data:
        return bar_data

    df = pd.DataFrame(bar_data["data"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    latest    = df["datetime"].iloc[-1]
    oos_start = latest - relativedelta(months=oos_months)

    df_bt      = df[df["datetime"] <  oos_start].copy()
    df_oos_raw = df[df["datetime"] >= oos_start].copy()

    # Prepend 210 warmup bars so EMA200 can initialise in OOS window
    warmup = df_bt.tail(210).copy()
    df_oos = pd.concat([warmup, df_oos_raw], ignore_index=True)

    # ── BEFORE params: Config C ───────────────────────────────────────
    before_params = dict(
        risk_pct       = 0.005,
        capital        = capital,
        atr_sl_mult    = 1.5,
        tp_rr          = 2.0,
        tp1_close_pct  = 0.75,
        use_ema200_daily = True,
        adx_threshold  = 20.0,
        adx_rising     = True,
        atr_regime_bars= 0,
        session_filter = False,
        min_cycle_bars = 10,
        allow_shorts   = True,
        adx_period     = 14,
        atr_regime_low = 20.0,
        atr_regime_high= 80.0,
    )

    # ── AFTER params: EMAPullbackMomentum ─────────────────────────────
    after_params = dict(
        risk_pct          = 0.005,
        capital           = capital,
        atr_sl_mult       = 1.5,
        pullback_atr_zone = 0.5,
        pullback_max_wait = 5,
        macd_fast         = 12,
        macd_slow         = 26,
        macd_signal       = 9,
        rsi_period        = 14,
        rsi_long_low      = 38,
        rsi_long_high     = 72,
        rsi_short_low     = 28,
        rsi_short_high    = 62,
        tp1_r             = 1.5,
        tp1_pct           = 0.40,
        tp2_r             = 3.0,
        tp2_pct           = 0.40,
        min_cycle_bars    = 10,
        adx_period        = 14,
        adx_threshold     = 20.0,
        adx_rising        = True,
        use_ema200        = True,
        allow_shorts      = True,
    )

    print("  Running BEFORE (Config C — EMA Crossover)...")
    bt_before  = _run(MultiTFEmaCross,     df_bt,  before_params, capital, commission)
    oos_before = _run(MultiTFEmaCross,     df_oos, before_params, capital, commission)

    print("  Running AFTER  (EMA Pullback Momentum)...")
    bt_after   = _run(EMAPullbackMomentum, df_bt,  after_params,  capital, commission)
    oos_after  = _run(EMAPullbackMomentum, df_oos, after_params,  capital, commission)

    def oos_eff(bt_m, oos_m):
        bs = bt_m.get("sharpe_ratio", 0)
        os = oos_m.get("sharpe_ratio", 0)
        return round(os / bs, 3) if bs and bs != 0 else None

    return {
        "symbol":   symbol.upper(),
        "exchange": exchange,
        "capital":  capital,
        "periods": {
            "backtest_start": str(df_bt["datetime"].iloc[0]),
            "backtest_end":   str(df_bt["datetime"].iloc[-1]),
            "oos_start":      str(df_oos_raw["datetime"].iloc[0]),
            "oos_end":        str(df_oos_raw["datetime"].iloc[-1]),
            "backtest_bars":  len(df_bt),
            "oos_bars":       len(df_oos_raw),
        },
        "before": {
            "label":        "Config C — EMA Crossover (0.5% risk, EMA200, ADX>20)",
            "backtest":     bt_before,
            "oos":          oos_before,
            "oos_efficiency": oos_eff(bt_before, oos_before),
        },
        "after": {
            "label":        "EMA Pullback Momentum (MACD + RSI + 3-level TP)",
            "backtest":     bt_after,
            "oos":          oos_after,
            "oos_efficiency": oos_eff(bt_after, oos_after),
        },
    }
