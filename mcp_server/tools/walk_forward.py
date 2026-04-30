import math
import logging
import pandas as pd
import optuna
from typing import Optional

from ..strategies import STRATEGY_REGISTRY
from .backtest import run_backtest_on_df

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


def _sample_params(trial: optuna.Trial, param_ranges: dict) -> dict:
    """Sample strategy params from Optuna trial using the param_ranges spec."""
    sampled = {}
    for name, spec in param_ranges.items():
        ptype = spec.get("type", "int")
        low = spec["low"]
        high = spec["high"]
        if ptype == "int":
            sampled[name] = trial.suggest_int(name, int(low), int(high))
        else:
            sampled[name] = trial.suggest_float(name, float(low), float(high))
    return sampled


def _optimize_on_df(
    strategy_name: str,
    df: pd.DataFrame,
    param_ranges: dict,
    n_trials: int,
    initial_cash: float,
    commission: float,
) -> tuple[dict, float]:
    """Run Optuna optimization on a DataFrame slice. Returns (best_params, best_sharpe)."""
    registry_params = STRATEGY_REGISTRY[strategy_name]["params"]

    # Merge registry defaults into param_ranges so fixed params pass through
    full_param_ranges = {}
    for k, v in registry_params.items():
        if k in param_ranges:
            full_param_ranges[k] = {**v, **param_ranges[k]}
        else:
            full_param_ranges[k] = {**v, "low": v["default"], "high": v["default"]}

    def objective(trial):
        params = _sample_params(trial, full_param_ranges)
        metrics = run_backtest_on_df(strategy_name, df, params, initial_cash, commission)
        if "error" in metrics:
            return -999.0
        sharpe = metrics.get("sharpe_ratio", 0.0) or 0.0
        # Penalise strategies with fewer than 5 trades to avoid overfitting edge cases
        if metrics.get("total_trades", 0) < 5:
            return -10.0
        return sharpe

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_value = study.best_value
    return best_params, best_value


def _aggregate_oos(windows: list[dict]) -> dict:
    """Compute aggregate OOS statistics across all walk-forward windows."""
    oos_metrics_list = [w["oos_metrics"] for w in windows if "error" not in w.get("oos_metrics", {})]

    if not oos_metrics_list:
        return {"error": "No valid OOS windows to aggregate"}

    n = len(oos_metrics_list)
    total_return = sum(m["total_return_pct"] for m in oos_metrics_list)
    avg_sharpe = sum(m["sharpe_ratio"] for m in oos_metrics_list) / n
    avg_drawdown = sum(m["max_drawdown_pct"] for m in oos_metrics_list) / n
    avg_win_rate = sum(m["win_rate_pct"] for m in oos_metrics_list) / n
    total_trades = sum(m["total_trades"] for m in oos_metrics_list)

    pf_values = [m["profit_factor"] for m in oos_metrics_list if m["profit_factor"] != float("inf")]
    avg_pf = sum(pf_values) / len(pf_values) if pf_values else float("inf")

    # Compounded OOS return: chain window returns
    compounded = 1.0
    for m in oos_metrics_list:
        compounded *= (1 + m["total_return_pct"] / 100)
    compounded_return_pct = round((compounded - 1) * 100, 2)

    return {
        "windows_tested": n,
        "compounded_oos_return_pct": compounded_return_pct,
        "sum_oos_return_pct": round(total_return, 2),
        "avg_sharpe": round(avg_sharpe, 4),
        "avg_max_drawdown_pct": round(avg_drawdown, 2),
        "avg_win_rate_pct": round(avg_win_rate, 2),
        "avg_profit_factor": round(avg_pf, 4),
        "total_oos_trades": total_trades,
    }


def _calc_wfe(windows: list[dict]) -> dict:
    """Walk-Forward Efficiency: ratio of OOS Sharpe to IS Sharpe per window.
    WFE > 0.5 suggests the optimised parameters are robust out-of-sample.
    """
    ratios = []
    for w in windows:
        is_sharpe = w.get("is_metrics", {}).get("sharpe_ratio", 0.0) or 0.0
        oos_sharpe = w.get("oos_metrics", {}).get("sharpe_ratio", 0.0) or 0.0
        if is_sharpe != 0:
            ratios.append(oos_sharpe / is_sharpe)

    if not ratios:
        return {"wfe": None, "interpretation": "Could not calculate WFE (zero IS Sharpe in all windows)"}

    wfe = round(sum(ratios) / len(ratios), 4)

    if wfe >= 1.0:
        interpretation = "Suspicious — OOS outperforms IS; possible data anomaly"
    elif wfe >= 0.5:
        interpretation = "Good — strategy is robust; OOS captures >50% of IS performance"
    elif wfe >= 0.25:
        interpretation = "Marginal — strategy shows some robustness but significant decay OOS"
    else:
        interpretation = "Poor — strategy is likely overfit; OOS performance degrades severely"

    return {
        "wfe": wfe,
        "window_ratios": [round(r, 4) for r in ratios],
        "interpretation": interpretation,
    }


def bt_walk_forward(
    strategy: str,
    symbol: str,
    exchange: str,
    timeframe: str,
    n_bars: int = 2000,
    param_ranges: Optional[dict] = None,
    n_windows: int = 5,
    in_sample_ratio: float = 0.7,
    anchored: bool = False,
    n_trials: int = 50,
    initial_cash: float = 10000.0,
    commission: float = 0.001,
) -> dict:
    """Walk-forward optimisation and testing.

    Each window: optimise parameters on the in-sample (IS) period using Optuna,
    then evaluate the best parameters on the held-out out-of-sample (OOS) period.
    Produces per-window results + aggregate OOS stats + Walk-Forward Efficiency (WFE).

    Args:
        strategy: Strategy name from bt_list_strategies e.g. 'SmaCross'
        symbol: Ticker symbol e.g. 'AAPL'
        exchange: Exchange e.g. 'NASDAQ'
        timeframe: Timeframe e.g. '1d', '4h', '1h'
        n_bars: Total bars to fetch (more bars = more windows possible)
        param_ranges: Dict of {param_name: {type, low, high}} to optimise.
                      Unspecified params use their registry defaults.
                      Example: {"fast_period": {"type": "int", "low": 5, "high": 30}}
        n_windows: Number of walk-forward windows (default 5)
        in_sample_ratio: Fraction of each window used for IS optimisation (default 0.7)
        anchored: If True, IS always starts from bar 0 and grows each window.
                  If False (default), rolling windows of equal size.
        n_trials: Optuna trials per IS window (default 50; increase for finer search)
        initial_cash: Starting capital per window (default 10000)
        commission: Commission per trade as fraction (default 0.001)
    """
    from .tradingview import tv_get_bars

    if strategy not in STRATEGY_REGISTRY:
        return {"error": f"Unknown strategy '{strategy}'. Call bt_list_strategies first."}

    bar_data = tv_get_bars(symbol, exchange, timeframe, n_bars)
    if "error" in bar_data:
        return bar_data

    df = pd.DataFrame(bar_data["data"])
    total_bars = len(df)

    if total_bars < n_windows * 20:
        return {
            "error": f"Too few bars ({total_bars}) for {n_windows} windows. "
                     f"Reduce n_windows or increase n_bars."
        }

    if param_ranges is None:
        param_ranges = {}

    windows_output = []

    if anchored:
        # Anchored: IS always starts at 0, OOS is a fixed-size rolling chunk
        oos_size = total_bars // (n_windows + 1)
        for i in range(n_windows):
            oos_start = oos_size * (i + 1)
            oos_end = oos_start + oos_size
            if oos_end > total_bars:
                break
            is_data = df.iloc[:oos_start]
            oos_data = df.iloc[oos_start:oos_end]
            _run_window(
                i + 1, strategy, is_data, oos_data,
                param_ranges, n_trials, initial_cash, commission, windows_output
            )
    else:
        # Rolling: equal-size windows, non-overlapping OOS periods
        window_size = total_bars // n_windows
        for i in range(n_windows):
            w_start = i * window_size
            w_end = w_start + window_size
            if w_end > total_bars:
                w_end = total_bars
            split = w_start + int((w_end - w_start) * in_sample_ratio)
            is_data = df.iloc[w_start:split]
            oos_data = df.iloc[split:w_end]
            _run_window(
                i + 1, strategy, is_data, oos_data,
                param_ranges, n_trials, initial_cash, commission, windows_output
            )

    aggregate = _aggregate_oos(windows_output)
    wfe = _calc_wfe(windows_output)

    return {
        "strategy": strategy,
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": timeframe,
        "total_bars": total_bars,
        "period": {"start": bar_data["start"], "end": bar_data["end"]},
        "config": {
            "n_windows": n_windows,
            "in_sample_ratio": in_sample_ratio,
            "anchored": anchored,
            "n_trials": n_trials,
            "param_ranges": param_ranges,
        },
        "windows": windows_output,
        "aggregate_oos": aggregate,
        "walk_forward_efficiency": wfe,
    }


def _run_window(
    window_num: int,
    strategy_name: str,
    is_data: pd.DataFrame,
    oos_data: pd.DataFrame,
    param_ranges: dict,
    n_trials: int,
    initial_cash: float,
    commission: float,
    output_list: list,
) -> None:
    """Optimise on IS, evaluate on OOS, append result to output_list."""
    logger.info(f"Window {window_num}: IS={len(is_data)} bars, OOS={len(oos_data)} bars")

    if len(is_data) < 10 or len(oos_data) < 5:
        output_list.append({
            "window": window_num,
            "error": "Insufficient data in this window",
        })
        return

    best_params, best_is_sharpe = _optimize_on_df(
        strategy_name, is_data, param_ranges, n_trials, initial_cash, commission
    )
    is_metrics = run_backtest_on_df(strategy_name, is_data, best_params, initial_cash, commission)
    oos_metrics = run_backtest_on_df(strategy_name, oos_data, best_params, initial_cash, commission)

    output_list.append({
        "window": window_num,
        "is_bars": len(is_data),
        "oos_bars": len(oos_data),
        "best_params": best_params,
        "is_metrics": is_metrics,
        "oos_metrics": oos_metrics,
    })
