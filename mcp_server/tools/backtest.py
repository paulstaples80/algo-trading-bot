import io
import sys
import logging
import pandas as pd
import backtrader as bt
from typing import Optional

from ..strategies import STRATEGY_REGISTRY

logging.getLogger("backtrader").setLevel(logging.WARNING)


class _ProfitFactor(bt.Analyzer):
    def start(self):
        self.gross_profit = 0.0
        self.gross_loss = 0.0

    def notify_trade(self, trade):
        if trade.isclosed:
            if trade.pnlcomm > 0:
                self.gross_profit += trade.pnlcomm
            else:
                self.gross_loss += abs(trade.pnlcomm)

    def get_analysis(self):
        if self.gross_loss == 0:
            pf = float("inf") if self.gross_profit > 0 else 0.0
        else:
            pf = self.gross_profit / self.gross_loss
        return {"profit_factor": round(pf, 4)}


def _df_to_bt_feed(df: pd.DataFrame) -> bt.feeds.PandasData:
    """Convert OHLCV DataFrame to a backtrader data feed."""
    df = df.copy()
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
    df.index = pd.DatetimeIndex(df.index)
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    return bt.feeds.PandasData(dataname=df)


def _extract_metrics(cerebro_results, initial_cash: float) -> dict:
    strat = cerebro_results[0]
    final_value = strat.broker.getvalue()
    total_return_pct = round((final_value - initial_cash) / initial_cash * 100, 2)

    sharpe = strat.analyzers.sharpe.get_analysis()
    sharpe_ratio = sharpe.get("sharperatio") or 0.0
    if sharpe_ratio is None:
        sharpe_ratio = 0.0

    drawdown = strat.analyzers.drawdown.get_analysis()
    max_dd = drawdown.get("max", {}).get("drawdown", 0.0) or 0.0

    trades = strat.analyzers.trades.get_analysis()
    total_trades = trades.get("total", {}).get("closed", 0) or 0
    won = trades.get("won", {}).get("total", 0) or 0
    win_rate = round(won / total_trades * 100, 2) if total_trades > 0 else 0.0

    pf = strat.analyzers.profitfactor.get_analysis()
    profit_factor = pf.get("profit_factor", 0.0)

    returns_data = strat.analyzers.returns.get_analysis()
    rnorm = returns_data.get("rnorm100", 0.0) or 0.0

    return {
        "initial_cash": initial_cash,
        "final_value": round(final_value, 2),
        "total_return_pct": total_return_pct,
        "annualized_return_pct": round(rnorm, 2),
        "sharpe_ratio": round(sharpe_ratio, 4),
        "max_drawdown_pct": round(max_dd, 2),
        "total_trades": total_trades,
        "win_rate_pct": win_rate,
        "profit_factor": profit_factor,
    }


def run_backtest_on_df(
    strategy_name: str,
    df: pd.DataFrame,
    params: dict,
    initial_cash: float = 10000.0,
    commission: float = 0.001,
) -> dict:
    """Run a single backtest on a pre-loaded DataFrame. Returns metrics dict."""
    if strategy_name not in STRATEGY_REGISTRY:
        return {"error": f"Unknown strategy '{strategy_name}'. Use bt_list_strategies to see options."}

    strategy_class = STRATEGY_REGISTRY[strategy_name]["class"]

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)

    feed = _df_to_bt_feed(df)
    cerebro.adddata(feed)
    cerebro.addstrategy(strategy_class, **params)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.02, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(_ProfitFactor, _name="profitfactor")

    # Suppress backtrader stdout
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        results = cerebro.run()
    finally:
        sys.stdout = old_stdout

    return _extract_metrics(results, initial_cash)


def bt_list_strategies() -> dict:
    """Return all available strategies with their parameter descriptions and default values."""
    output = {}
    for name, meta in STRATEGY_REGISTRY.items():
        output[name] = {
            "description": meta["description"],
            "params": {
                k: {
                    "type": v["type"],
                    "default": v["default"],
                    "range": [v["low"], v["high"]],
                    "desc": v["desc"],
                }
                for k, v in meta["params"].items()
            },
        }
    return output


def bt_run_backtest(
    strategy: str,
    symbol: str,
    exchange: str,
    timeframe: str,
    n_bars: int = 1000,
    params: Optional[dict] = None,
    initial_cash: float = 10000.0,
    commission: float = 0.001,
) -> dict:
    """Run a single backtest for a strategy against TradingView data.

    Args:
        strategy: Strategy name from bt_list_strategies e.g. 'SmaCross'
        symbol: Ticker symbol e.g. 'AAPL'
        exchange: Exchange e.g. 'NASDAQ'
        timeframe: Timeframe e.g. '1d', '4h', '1h'
        n_bars: Number of bars to fetch (default 1000)
        params: Strategy parameters dict. Defaults used if not provided.
        initial_cash: Starting capital (default 10000)
        commission: Commission per trade as fraction (default 0.001 = 0.1%)
    """
    from .tradingview import tv_get_bars

    if strategy not in STRATEGY_REGISTRY:
        return {"error": f"Unknown strategy '{strategy}'. Call bt_list_strategies first."}

    bar_data = tv_get_bars(symbol, exchange, timeframe, n_bars)
    if "error" in bar_data:
        return bar_data

    df = pd.DataFrame(bar_data["data"])

    # Fill missing params with defaults
    merged_params = {
        k: v["default"] for k, v in STRATEGY_REGISTRY[strategy]["params"].items()
    }
    if params:
        merged_params.update(params)

    metrics = run_backtest_on_df(strategy, df, merged_params, initial_cash, commission)
    if "error" in metrics:
        return metrics

    return {
        "strategy": strategy,
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": timeframe,
        "bars": bar_data["bars"],
        "period": {"start": bar_data["start"], "end": bar_data["end"]},
        "params": merged_params,
        "metrics": metrics,
    }
