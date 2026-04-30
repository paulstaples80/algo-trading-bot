"""
Run 5 strategy configurations against the same EURUSD dataset and
produce a side-by-side comparison table.
"""

import os
import json
import pandas as pd
from .multi_tf_backtest import _run_multitf_cerebro, _resample_ohlcv
from .tradingview import tv_get_bars
from dateutil.relativedelta import relativedelta

CONFIGS = {
    "Baseline": dict(
        risk_pct=0.01,
        atr_sl_mult=1.5,
        tp_rr=2.0,
        use_ema200_daily=False,
        adx_threshold=0.0,
        adx_rising=False,
        atr_regime_bars=0,
        session_filter=False,
    ),
    "A — 0.5% Risk": dict(
        risk_pct=0.005,
        atr_sl_mult=1.5,
        tp_rr=2.0,
        use_ema200_daily=False,
        adx_threshold=0.0,
        adx_rising=False,
        atr_regime_bars=0,
        session_filter=False,
    ),
    "B — 0.5% + EMA200 Daily": dict(
        risk_pct=0.005,
        atr_sl_mult=1.5,
        tp_rr=2.0,
        use_ema200_daily=True,
        adx_threshold=0.0,
        adx_rising=False,
        atr_regime_bars=0,
        session_filter=False,
    ),
    "C — 0.5% + EMA200 + ADX>20": dict(
        risk_pct=0.005,
        atr_sl_mult=1.5,
        tp_rr=2.0,
        use_ema200_daily=True,
        adx_threshold=20.0,
        adx_rising=True,
        atr_regime_bars=0,
        session_filter=False,
    ),
    "D — Full Stack + 3:1 TP": dict(
        risk_pct=0.005,
        atr_sl_mult=1.5,
        tp_rr=3.0,
        use_ema200_daily=True,
        adx_threshold=20.0,
        adx_rising=True,
        atr_regime_bars=50,
        atr_regime_low=20.0,
        atr_regime_high=80.0,
        session_filter=True,
    ),
}


def bt_compare_configs(
    symbol: str = "EURUSD",
    exchange: str = "OANDA",
    n_bars: int = 5000,
    oos_months: int = 6,
    capital: float = 10000.0,
    commission: float = 0.00007,
) -> dict:
    """Run all 5 drawdown-reduction configurations against the same dataset.

    Returns a side-by-side comparison of backtest and OOS metrics for each config.
    """
    bar_data = tv_get_bars(symbol.upper(), exchange, "4h", n_bars)
    if "error" in bar_data:
        return bar_data

    df = pd.DataFrame(bar_data["data"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    latest    = df["datetime"].iloc[-1]
    oos_start = latest - relativedelta(months=oos_months)

    df_bt  = df[df["datetime"] < oos_start].copy()
    df_oos_raw = df[df["datetime"] >= oos_start].copy()

    # Prepend 300 warmup bars from backtest so EMA/ADX indicators can initialise.
    # The broker starts with full capital so warmup trades don't affect OOS metrics
    # — but we reset cash before the real OOS period by running warmup with size=0.
    # Simpler approach: just prepend bars; backtrader computes indicators then starts
    # trading naturally. The warmup period has no valid signals because all the
    # entry conditions require the cross to be sustained for min_cycle_bars anyway.
    # 210 warmup bars covers the longest indicator period (EMA200 on 4H = 200 bars).
    warmup = df_bt.tail(210).copy()
    df_oos = pd.concat([warmup, df_oos_raw], ignore_index=True)

    results = {}

    for name, extra_params in CONFIGS.items():
        print(f"  Running: {name}...")
        base = dict(
            capital=capital,
            min_cycle_bars=10,
            allow_shorts=True,
            adx_period=14,
            atr_regime_low=20.0,
            atr_regime_high=80.0,
        )
        base.update(extra_params)   # extra_params overrides base defaults
        params = base
        bt_m  = _run_multitf_cerebro(df_bt,  params, capital, commission)
        oos_m = _run_multitf_cerebro(df_oos, params, capital, commission)

        if "error" in bt_m or "error" in oos_m:
            results[name] = {"error": bt_m.get("error") or oos_m.get("error")}
            continue

        bt_sharpe  = bt_m["sharpe_ratio"]
        oos_sharpe = oos_m["sharpe_ratio"]
        oos_eff    = round(oos_sharpe / bt_sharpe, 3) if bt_sharpe != 0 else None

        results[name] = {
            "backtest": bt_m,
            "oos":      oos_m,
            "oos_efficiency": oos_eff,
        }

    # Build flat comparison table
    rows = []
    for name, r in results.items():
        if "error" in r:
            continue
        bt  = r["backtest"]
        oos = r["oos"]
        rows.append({
            "config":              name,
            "bt_return_%":         bt["total_return_pct"],
            "bt_annual_%":         bt["annualised_return_pct"],
            "bt_sharpe":           bt["sharpe_ratio"],
            "bt_max_dd_%":         bt["max_drawdown_pct"],
            "bt_trades":           bt["total_trades"],
            "bt_win_rate_%":       bt["win_rate_pct"],
            "bt_profit_factor":    bt["profit_factor"],
            "oos_return_%":        oos["total_return_pct"],
            "oos_sharpe":          oos["sharpe_ratio"],
            "oos_max_dd_%":        oos["max_drawdown_pct"],
            "oos_trades":          oos["total_trades"],
            "oos_win_rate_%":      oos["win_rate_pct"],
            "oos_profit_factor":   oos["profit_factor"],
            "oos_efficiency":      r["oos_efficiency"],
        })

    return {
        "symbol":   symbol.upper(),
        "exchange": exchange,
        "backtest_period": {
            "start": str(df_bt["datetime"].iloc[0]),
            "end":   str(df_bt["datetime"].iloc[-1]),
        },
        "oos_period": {
            "start": str(df_oos["datetime"].iloc[0]),
            "end":   str(df_oos["datetime"].iloc[-1]),
        },
        "comparison_table": rows,
        "raw": results,
    }
