"""TradingView + Backtest MCP Server.

Exposes TradingView data fetching and backtrader backtesting (including
walk-forward optimisation) as MCP tools for use with Claude agents.

Run with:
    python -m mcp_server

or configure in Claude Desktop's claude_desktop_config.json.
"""
import json
from mcp.server.fastmcp import FastMCP
from .tools import (
    tv_get_bars as _tv_get_bars,
    tv_get_indicators as _tv_get_indicators,
    tv_screen as _tv_screen,
    bt_list_strategies as _bt_list_strategies,
    bt_run_backtest as _bt_run_backtest,
    bt_walk_forward as _bt_walk_forward,
)

mcp = FastMCP(
    "tradingview-backtest",
    instructions=(
        "You are a backtesting agent with access to TradingView market data and a "
        "Python backtrader engine. Use tv_get_bars and tv_get_indicators to research "
        "symbols, bt_list_strategies to see available strategies, bt_run_backtest for "
        "quick single-run tests, and bt_walk_forward for robust walk-forward optimisation. "
        "Always check walk-forward results before recommending a strategy."
    ),
)


@mcp.tool()
def tv_get_bars(symbol: str, exchange: str, timeframe: str, n_bars: int = 1000) -> str:
    """Fetch OHLCV price bars from TradingView.

    Args:
        symbol: Ticker e.g. 'AAPL', 'EURUSD', 'BTCUSDT'
        exchange: Exchange e.g. 'NASDAQ', 'FOREX', 'BINANCE'
        timeframe: Bar size — 1m/3m/5m/15m/30m/45m/1h/2h/3h/4h/1d/1w/1M
        n_bars: Number of bars to fetch (default 1000, max ~5000)
    """
    result = _tv_get_bars(symbol, exchange, timeframe, n_bars)
    return json.dumps(result, default=str)


@mcp.tool()
def tv_get_indicators(symbol: str, screener: str, exchange: str, timeframe: str) -> str:
    """Get all TradingView technical indicators and TA summary for a symbol.

    Returns RSI, MACD, Bollinger Bands, 50+ indicators plus buy/sell/neutral counts.

    Args:
        symbol: Ticker e.g. 'AAPL'
        screener: 'america', 'forex', 'crypto', 'cfd', 'india', etc.
        exchange: Exchange e.g. 'NASDAQ', 'NYSE', 'BINANCE'
        timeframe: 1m/5m/15m/30m/1h/2h/4h/1d/1w/1M
    """
    result = _tv_get_indicators(symbol, screener, exchange, timeframe)
    return json.dumps(result, default=str)


@mcp.tool()
def tv_screen(symbols: list, screener: str, exchange: str, timeframe: str) -> str:
    """Screen multiple symbols for buy/sell signals and key indicators at once.

    Args:
        symbols: List of tickers e.g. ['AAPL', 'MSFT', 'GOOGL']
        screener: 'america', 'forex', 'crypto', 'cfd', etc.
        exchange: Exchange e.g. 'NASDAQ'
        timeframe: 1m/5m/15m/30m/1h/2h/4h/1d/1w/1M
    """
    result = _tv_screen(symbols, screener, exchange, timeframe)
    return json.dumps(result, default=str)


@mcp.tool()
def bt_list_strategies() -> str:
    """List all available backtest strategies with their parameter names, types, defaults, and optimisation ranges."""
    result = _bt_list_strategies()
    return json.dumps(result, default=str)


@mcp.tool()
def bt_run_backtest(
    strategy: str,
    symbol: str,
    exchange: str,
    timeframe: str,
    n_bars: int = 1000,
    params: dict = None,
    initial_cash: float = 10000.0,
    commission: float = 0.001,
) -> str:
    """Run a single backtest and return performance metrics.

    Returns: total return %, Sharpe ratio, max drawdown %, win rate %, profit factor, trade count.

    Args:
        strategy: Strategy name from bt_list_strategies e.g. 'SmaCross'
        symbol: Ticker e.g. 'AAPL'
        exchange: Exchange e.g. 'NASDAQ'
        timeframe: Timeframe e.g. '1d', '4h'
        n_bars: Bars of history to backtest over (default 1000)
        params: Strategy params dict — omit to use defaults
        initial_cash: Starting capital (default 10000)
        commission: Round-trip commission fraction (default 0.001 = 0.1%)
    """
    result = _bt_run_backtest(strategy, symbol, exchange, timeframe, n_bars, params, initial_cash, commission)
    return json.dumps(result, default=str)


@mcp.tool()
def bt_walk_forward(
    strategy: str,
    symbol: str,
    exchange: str,
    timeframe: str,
    n_bars: int = 2000,
    param_ranges: dict = None,
    n_windows: int = 5,
    in_sample_ratio: float = 0.7,
    anchored: bool = False,
    n_trials: int = 50,
    initial_cash: float = 10000.0,
    commission: float = 0.001,
) -> str:
    """Walk-forward optimisation: optimise on in-sample, test on out-of-sample, repeat across N windows.

    Returns per-window IS/OOS metrics, aggregate OOS statistics, and Walk-Forward Efficiency (WFE).
    WFE >= 0.5 indicates a robust strategy; < 0.25 suggests overfitting.

    Args:
        strategy: Strategy name e.g. 'SmaCross'
        symbol: Ticker e.g. 'AAPL'
        exchange: Exchange e.g. 'NASDAQ'
        timeframe: Timeframe e.g. '1d', '4h', '1h'
        n_bars: Total bars to fetch — more = more windows possible (default 2000)
        param_ranges: Dict of params to optimise.
            Format: {"param_name": {"type": "int"|"float", "low": X, "high": Y}}
            Unspecified params use their defaults. Omit to optimise all params.
        n_windows: Number of walk-forward windows (default 5)
        in_sample_ratio: IS fraction of each window (default 0.7 = 70% IS / 30% OOS)
        anchored: True = growing IS window; False = rolling equal-size windows (default)
        n_trials: Optuna optimisation trials per IS window (default 50)
        initial_cash: Starting capital per window (default 10000)
        commission: Commission fraction (default 0.001)
    """
    result = _bt_walk_forward(
        strategy, symbol, exchange, timeframe, n_bars,
        param_ranges, n_windows, in_sample_ratio, anchored,
        n_trials, initial_cash, commission
    )
    return json.dumps(result, default=str)
