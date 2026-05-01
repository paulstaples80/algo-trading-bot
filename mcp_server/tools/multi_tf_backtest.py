"""
Multi-Timeframe EMA Backtest Tool
==================================
Fetches 1H data, resamples to 4H and Daily internally, runs the
MultiTFEmaCross strategy, then validates on a held-out OOS window.
"""

import io
import sys
import logging
import json
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import pandas as pd
import backtrader as bt

from ..strategies.multi_tf_ema import MultiTFEmaCross
from .backtest import _ProfitFactor

logger = logging.getLogger(__name__)

MAJOR_PAIRS = {
    "EURUSD": "FX_IDC",
    "GBPUSD": "FX_IDC",
    "USDJPY": "FX_IDC",
    "USDCHF": "FX_IDC",
    "AUDUSD": "FX_IDC",
    "NZDUSD": "FX_IDC",
    "USDCAD": "FX_IDC",
}

# OANDA exchange is more reliable for forex data than FX_IDC
MAJOR_PAIRS_OANDA = {
    "EURUSD": "OANDA",
    "GBPUSD": "OANDA",
    "USDJPY": "OANDA",
    "USDCHF": "OANDA",
    "AUDUSD": "OANDA",
    "NZDUSD": "OANDA",
    "USDCAD": "OANDA",
}


def _df_to_bt_feed(df: pd.DataFrame) -> bt.feeds.PandasData:
    d = df.copy()
    if "datetime" in d.columns:
        d["datetime"] = pd.to_datetime(d["datetime"])
        d = d.set_index("datetime")
    d.index = pd.DatetimeIndex(d.index)
    d = d[["open", "high", "low", "close", "volume"]].dropna()
    return bt.feeds.PandasData(dataname=d)


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample a 1H OHLCV DataFrame to a coarser timeframe using pandas."""
    d = df.copy()
    d.index = pd.DatetimeIndex(d.index)
    resampled = d.resample(rule).agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna(subset=["open", "close"])
    return resampled


def _run_multitf_cerebro(df: pd.DataFrame, params: dict, capital: float, commission: float) -> dict:
    """Build cerebro with 2 timeframes (4H execution / Daily trend), run, return metrics.

    Note: TradingView limits 1H forex data to ~10 months. 4H data provides 3+ years
    of history, so 4H is used as the entry/execution timeframe. Daily is resampled
    from 4H in pandas for reliability.
    """
    d = df.copy()
    d["datetime"] = pd.to_datetime(d["datetime"])
    d = d.set_index("datetime")
    d = d[["open", "high", "low", "close", "volume"]].dropna()

    # Resample 4H → Daily
    d_daily = _resample_ohlcv(d, "1D")

    if len(d) < 210:
        return {"error": f"Insufficient 4H bars: {len(d)} (need ≥210 for EMA200)"}
    if len(d_daily) < 60:
        return {"error": f"Insufficient Daily bars after resample: {len(d_daily)} (need ≥60)"}

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(capital)
    cerebro.broker.setcommission(commission=commission)

    # datas[0] = 4H  (execution + EMA20/50/200 stacking)
    # datas[1] = Daily (trend + cyclicity via EMA20/50)
    cerebro.adddata(bt.feeds.PandasData(dataname=d),       name="4H")
    cerebro.adddata(bt.feeds.PandasData(dataname=d_daily), name="Daily")

    cerebro.addstrategy(MultiTFEmaCross, **params)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe",
                        riskfreerate=0.02, annualize=True, timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.DrawDown,     _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.Returns,       _name="returns")
    cerebro.addanalyzer(_ProfitFactor,              _name="profitfactor")

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        results = cerebro.run()
    finally:
        sys.stdout = old_stdout

    strat = results[0]
    final = strat.broker.getvalue()
    ret   = round((final - capital) / capital * 100, 2)

    sharpe_raw = strat.analyzers.sharpe.get_analysis().get("sharperatio") or 0.0
    sharpe     = round(float(sharpe_raw), 4)

    dd_raw = strat.analyzers.drawdown.get_analysis()
    max_dd = round(dd_raw.get("max", {}).get("drawdown", 0.0) or 0.0, 2)

    ta = strat.analyzers.trades.get_analysis()
    total_trades = ta.get("total", {}).get("closed", 0) or 0
    won          = ta.get("won",   {}).get("total", 0) or 0
    win_rate     = round(won / total_trades * 100, 2) if total_trades > 0 else 0.0

    pf_raw = strat.analyzers.profitfactor.get_analysis().get("profit_factor", 0.0)

    rnorm = strat.analyzers.returns.get_analysis().get("rnorm100", 0.0) or 0.0

    return {
        "initial_capital": capital,
        "final_value": round(final, 2),
        "total_return_pct": ret,
        "annualised_return_pct": round(rnorm, 2),
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_dd,
        "total_trades": total_trades,
        "win_rate_pct": win_rate,
        "profit_factor": round(pf_raw, 4),
    }


def bt_forex_multitf(
    symbol: str,
    exchange: str = "FX_IDC",
    n_bars: int = 5000,
    oos_months: int = 6,
    capital: float = 10000.0,
    commission: float = 0.00007,
    atr_sl_mult: float = 1.5,
    tp_rr: float = 2.0,
    min_cycle_bars: int = 10,
    allow_shorts: bool = True,
) -> dict:
    """Run a multi-timeframe EMA 20/50 backtest + OOS validation for a forex pair.

    Fetches 1H data, resamples to 4H and Daily inside backtrader.
    Splits data: last `oos_months` months held out for OOS validation.
    Backtests on everything before that cutoff.

    Args:
        symbol: Forex pair e.g. 'EURUSD', 'GBPUSD'
        exchange: TradingView exchange, default 'FX_IDC'
        n_bars: 1H bars to fetch (default 5000 ≈ 10 months for forex)
        oos_months: Months to hold out for OOS validation (default 6)
        capital: Starting capital in account currency (default 10000)
        commission: Per-trade commission fraction (default 0.00007 ≈ 0.7 pip spread)
        atr_sl_mult: ATR multiplier for stop loss (default 1.5)
        tp_rr: Take profit risk:reward ratio (default 2.0)
        min_cycle_bars: Daily bars EMA20>EMA50 must be sustained (default 10 = 2 weeks)
        allow_shorts: Trade short setups as well as long (default True)
    """
    from .tradingview import tv_get_bars

    sym = symbol.upper()
    if sym not in MAJOR_PAIRS and exchange == "FX_IDC":
        logger.warning(f"{sym} not in major pairs list — proceeding anyway")

    # TradingView limits 1H forex history to ~10 months regardless of plan.
    # 4H gives 3+ years — sufficient for meaningful backtesting.
    # Entry signals use 4H; daily trend uses 4H resampled to 1D.
    bar_data = tv_get_bars(sym, exchange, "4h", n_bars)
    if "error" in bar_data:
        return bar_data

    df = pd.DataFrame(bar_data["data"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    if len(df) < 300:
        return {"error": f"Too few bars returned ({len(df)}). Try a higher n_bars value."}

    # ── Split backtest / OOS ──────────────────────────────────────────
    latest    = df["datetime"].iloc[-1]
    oos_start = latest - relativedelta(months=oos_months)

    df_bt  = df[df["datetime"] < oos_start].copy()
    df_oos = df[df["datetime"] >= oos_start].copy()

    if len(df_bt) < 100:
        return {
            "error": (
                f"Not enough bars before OOS cutoff ({len(df_bt)} bars). "
                f"Increase n_bars — you need at least {oos_months} months OOS + "
                f"backtest history. For forex 1H, ~5000 bars covers ~10 months total."
            )
        }

    params = dict(
        atr_sl_mult   = atr_sl_mult,
        tp_rr         = tp_rr,
        min_cycle_bars= min_cycle_bars,
        allow_shorts  = allow_shorts,
        capital       = capital,
        risk_pct      = 0.01,
    )

    # ── Backtest ──────────────────────────────────────────────────────
    bt_metrics = _run_multitf_cerebro(df_bt, params, capital, commission)
    if "error" in bt_metrics:
        return bt_metrics

    # ── OOS validation ────────────────────────────────────────────────
    oos_metrics = _run_multitf_cerebro(df_oos, params, capital, commission)
    if "error" in oos_metrics:
        return oos_metrics

    # ── OOS efficiency ────────────────────────────────────────────────
    bt_sharpe  = bt_metrics["sharpe_ratio"]
    oos_sharpe = oos_metrics["sharpe_ratio"]
    if bt_sharpe != 0:
        oos_efficiency = round(oos_sharpe / bt_sharpe, 4)
        if oos_efficiency >= 0.5:
            oos_interp = "Good — OOS captures ≥50% of backtest performance"
        elif oos_efficiency >= 0.25:
            oos_interp = "Marginal — some robustness but notable decay OOS"
        else:
            oos_interp = "Poor — significant OOS degradation"
    else:
        oos_efficiency = None
        oos_interp = "Cannot calculate — backtest Sharpe is zero"

    return {
        "symbol": sym,
        "exchange": exchange,
        "strategy": "MultiTFEmaCross (Daily/4H/1H EMA 20/50)",
        "params": {
            "ema_fast": 20,
            "ema_slow": 50,
            "ema_trend_4h": 200,
            "atr_sl_mult": atr_sl_mult,
            "tp_rr": tp_rr,
            "tp1_close_pct": 0.75,
            "min_cycle_bars_daily": min_cycle_bars,
            "allow_shorts": allow_shorts,
            "risk_pct": "1%",
            "capital": capital,
        },
        "data": {
            "total_1h_bars": len(df),
            "backtest_bars": len(df_bt),
            "oos_bars": len(df_oos),
            "backtest_period": {
                "start": str(df_bt["datetime"].iloc[0]),
                "end":   str(df_bt["datetime"].iloc[-1]),
            },
            "oos_period": {
                "start": str(df_oos["datetime"].iloc[0]),
                "end":   str(df_oos["datetime"].iloc[-1]),
            },
        },
        "backtest_metrics": bt_metrics,
        "oos_metrics": oos_metrics,
        "oos_efficiency": {
            "ratio": oos_efficiency,
            "interpretation": oos_interp,
        },
    }


def bt_forex_screen_multitf(
    symbols: list = None,
    oos_months: int = 6,
    capital: float = 10000.0,
    n_bars: int = 5000,
) -> dict:
    """Run the MultiTFEmaCross strategy across all major forex pairs and rank by OOS Sharpe.

    Args:
        symbols: List of pairs to test — defaults to all 7 majors
        oos_months: Months held out for OOS validation (default 6)
        capital: Starting capital (default 10000)
        n_bars: 1H bars to fetch per pair (default 5000)
    """
    pairs = symbols if symbols else list(MAJOR_PAIRS.keys())
    results = {}

    for pair in pairs:
        exchange = MAJOR_PAIRS.get(pair.upper(), "FX_IDC")
        logger.info(f"Testing {pair}...")
        result = bt_forex_multitf(
            symbol=pair, exchange=exchange,
            n_bars=n_bars, oos_months=oos_months,
            capital=capital,
        )
        if "error" in result:
            results[pair] = {"error": result["error"]}
        else:
            results[pair] = {
                "bt_return_pct":    result["backtest_metrics"]["total_return_pct"],
                "bt_sharpe":        result["backtest_metrics"]["sharpe_ratio"],
                "bt_max_dd_pct":    result["backtest_metrics"]["max_drawdown_pct"],
                "bt_trades":        result["backtest_metrics"]["total_trades"],
                "bt_win_rate":      result["backtest_metrics"]["win_rate_pct"],
                "bt_profit_factor": result["backtest_metrics"]["profit_factor"],
                "oos_return_pct":   result["oos_metrics"]["total_return_pct"],
                "oos_sharpe":       result["oos_metrics"]["sharpe_ratio"],
                "oos_max_dd_pct":   result["oos_metrics"]["max_drawdown_pct"],
                "oos_trades":       result["oos_metrics"]["total_trades"],
                "oos_win_rate":     result["oos_metrics"]["win_rate_pct"],
                "oos_efficiency":   result["oos_efficiency"]["ratio"],
            }

    # Rank by OOS Sharpe (best first)
    ranked = sorted(
        [(k, v) for k, v in results.items() if "error" not in v],
        key=lambda x: x[1].get("oos_sharpe", -999),
        reverse=True,
    )

    return {
        "strategy": "MultiTFEmaCross (Daily/4H/1H EMA 20/50)",
        "pairs_tested": len(pairs),
        "oos_months": oos_months,
        "ranked_by_oos_sharpe": [{"pair": k, **v} for k, v in ranked],
        "errors": {k: v["error"] for k, v in results.items() if "error" in v},
    }


def bt_config_c_screen(
    symbols: list = None,
    oos_months: int = 6,
    capital: float = 10000.0,
    n_bars: int = 5000,
    commission: float = 0.00007,
) -> dict:
    """Run Config C (0.5% risk + EMA200 + ADX>20 rising) across all 7 major forex pairs via OANDA.

    Config C params:
      - risk_pct      = 0.5% per trade
      - EMA200 alignment filter on 4H (structural trend confirmation)
      - ADX(14) > 20 AND rising (trend strength filter)
      - SL = 1.5× ATR, TP = 2:1 RR, close 75% at TP, 25% runner to BE/daily EMA flip

    Each pair uses a 6-month OOS window with 210 warm-up bars prepended.
    Results ranked by OOS Sharpe ratio (best → worst).
    """
    from .tradingview import tv_get_bars

    pairs = symbols if symbols else list(MAJOR_PAIRS_OANDA.keys())

    config_c_params = dict(
        risk_pct          = 0.005,
        capital           = capital,
        atr_sl_mult       = 1.5,
        tp_rr             = 2.0,
        tp1_close_pct     = 0.75,
        use_ema200_daily  = True,
        adx_threshold     = 20.0,
        adx_rising        = True,
        atr_regime_bars   = 0,
        session_filter    = False,
        min_cycle_bars    = 10,
        allow_shorts      = True,
        adx_period        = 14,
        atr_regime_low    = 20.0,
        atr_regime_high   = 80.0,
    )

    results = {}

    for pair in pairs:
        exchange = MAJOR_PAIRS_OANDA.get(pair.upper(), "OANDA")
        print(f"  Running Config C on {pair} ({exchange})...")

        bar_data = tv_get_bars(pair.upper(), exchange, "4h", n_bars)
        if "error" in bar_data:
            results[pair] = {"error": bar_data["error"]}
            continue

        df = pd.DataFrame(bar_data["data"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

        if len(df) < 400:
            results[pair] = {"error": f"Too few bars ({len(df)})"}
            continue

        latest    = df["datetime"].iloc[-1]
        oos_start = latest - relativedelta(months=oos_months)

        df_bt      = df[df["datetime"] <  oos_start].copy()
        df_oos_raw = df[df["datetime"] >= oos_start].copy()

        if len(df_bt) < 210:
            results[pair] = {"error": f"Insufficient backtest bars ({len(df_bt)})"}
            continue

        # Prepend 210 warmup bars so EMA200 can initialise in OOS window
        warmup = df_bt.tail(210).copy()
        df_oos = pd.concat([warmup, df_oos_raw], ignore_index=True)

        params = {**config_c_params}   # copy

        bt_m  = _run_multitf_cerebro(df_bt,  params, capital, commission)
        oos_m = _run_multitf_cerebro(df_oos, params, capital, commission)

        if "error" in bt_m or "error" in oos_m:
            results[pair] = {"error": bt_m.get("error") or oos_m.get("error")}
            continue

        bs = bt_m.get("sharpe_ratio", 0)
        os_ = oos_m.get("sharpe_ratio", 0)
        oos_eff = round(os_ / bs, 3) if bs and bs != 0 else None

        results[pair] = {
            "exchange":          exchange,
            "backtest_bars":     len(df_bt),
            "oos_bars":          len(df_oos_raw),
            "backtest_start":    str(df_bt["datetime"].iloc[0]),
            "backtest_end":      str(df_bt["datetime"].iloc[-1]),
            "oos_start":         str(df_oos_raw["datetime"].iloc[0]),
            "oos_end":           str(df_oos_raw["datetime"].iloc[-1]),
            "bt_return_pct":     bt_m["total_return_pct"],
            "bt_annual_pct":     bt_m["annualised_return_pct"],
            "bt_sharpe":         bt_m["sharpe_ratio"],
            "bt_max_dd_pct":     bt_m["max_drawdown_pct"],
            "bt_trades":         bt_m["total_trades"],
            "bt_win_rate_pct":   bt_m["win_rate_pct"],
            "bt_profit_factor":  bt_m["profit_factor"],
            "oos_return_pct":    oos_m["total_return_pct"],
            "oos_annual_pct":    oos_m["annualised_return_pct"],
            "oos_sharpe":        oos_m["sharpe_ratio"],
            "oos_max_dd_pct":    oos_m["max_drawdown_pct"],
            "oos_trades":        oos_m["total_trades"],
            "oos_win_rate_pct":  oos_m["win_rate_pct"],
            "oos_profit_factor": oos_m["profit_factor"],
            "oos_efficiency":    oos_eff,
        }

    # Rank successful results by OOS Sharpe
    ranked = sorted(
        [(k, v) for k, v in results.items() if "error" not in v],
        key=lambda x: x[1].get("oos_sharpe", -999),
        reverse=True,
    )

    # Portfolio-level summary (treating all 7 pairs as a portfolio, equal capital)
    valid = [v for v in results.values() if "error" not in v]
    if valid:
        avg_bt_return   = round(sum(v["bt_return_pct"]   for v in valid) / len(valid), 2)
        avg_oos_return  = round(sum(v["oos_return_pct"]  for v in valid) / len(valid), 2)
        avg_bt_sharpe   = round(sum(v["bt_sharpe"]       for v in valid) / len(valid), 4)
        avg_oos_sharpe  = round(sum(v["oos_sharpe"]      for v in valid) / len(valid), 4)
        avg_bt_dd       = round(sum(v["bt_max_dd_pct"]   for v in valid) / len(valid), 2)
        avg_oos_dd      = round(sum(v["oos_max_dd_pct"]  for v in valid) / len(valid), 2)
        total_bt_trades = sum(v["bt_trades"]  for v in valid)
        total_oos_trades= sum(v["oos_trades"] for v in valid)
        eff_vals        = [v["oos_efficiency"] for v in valid if v["oos_efficiency"] is not None]
        avg_eff         = round(sum(eff_vals) / len(eff_vals), 3) if eff_vals else None
    else:
        avg_bt_return = avg_oos_return = avg_bt_sharpe = avg_oos_sharpe = None
        avg_bt_dd = avg_oos_dd = total_bt_trades = total_oos_trades = avg_eff = None

    return {
        "strategy":          "MultiTFEmaCross — Config C (0.5% risk, EMA200 4H, ADX>20 rising)",
        "pairs_tested":      len(pairs),
        "pairs_successful":  len(valid),
        "oos_months":        oos_months,
        "config_c_params":   config_c_params,
        "portfolio_summary": {
            "avg_bt_return_pct":   avg_bt_return,
            "avg_oos_return_pct":  avg_oos_return,
            "avg_bt_sharpe":       avg_bt_sharpe,
            "avg_oos_sharpe":      avg_oos_sharpe,
            "avg_bt_max_dd_pct":   avg_bt_dd,
            "avg_oos_max_dd_pct":  avg_oos_dd,
            "total_bt_trades":     total_bt_trades,
            "total_oos_trades":    total_oos_trades,
            "avg_oos_efficiency":  avg_eff,
        },
        "ranked_by_oos_sharpe": [{"pair": k, **v} for k, v in ranked],
        "errors": {k: v["error"] for k, v in results.items() if "error" in v},
    }
