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
    bt_forex_multitf as _bt_forex_multitf,
    bt_forex_screen_multitf as _bt_forex_screen_multitf,
    bt_config_c_screen as _bt_config_c_screen,
    bt_compare_configs as _bt_compare_configs,
    bt_before_after as _bt_before_after,
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


@mcp.tool()
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
) -> str:
    """Run the Multi-Timeframe EMA 20/50 forex strategy backtest + 6-month OOS validation.

    Strategy rules:
      - Daily: EMA20 > EMA50 sustained for min_cycle_bars (bullish trend / cyclicity proxy)
      - 4H   : EMA20 > EMA50 > EMA200 fully stacked
      - 1H   : EMA20 > EMA50 — entry bar
      - SL   : 1.5× ATR(14) on 1H
      - TP1  : 2:1 RR → close 75%, runner stays open at BE
      - Runner exit: BE stop hit OR Daily EMA20/50 flips opposite

    Last oos_months of data is held out and never used in backtesting.

    Args:
        symbol: Forex pair e.g. 'EURUSD', 'GBPUSD', 'USDJPY'
        exchange: TradingView exchange (default 'FX_IDC')
        n_bars: 1H bars to fetch — 5000 ≈ 10 months for forex (default 5000)
        oos_months: Months held out for OOS validation (default 6)
        capital: Starting capital in account currency (default 10000)
        commission: Per-trade commission as price fraction (default 0.00007 ≈ 0.7 pip)
        atr_sl_mult: ATR multiplier for stop loss distance (default 1.5)
        tp_rr: Take profit risk:reward ratio (default 2.0)
        min_cycle_bars: Minimum daily bars EMA cross must be sustained (default 10)
        allow_shorts: Include bearish setups (default True)
    """
    result = _bt_forex_multitf(
        symbol, exchange, n_bars, oos_months, capital,
        commission, atr_sl_mult, tp_rr, min_cycle_bars, allow_shorts,
    )
    return json.dumps(result, default=str)


@mcp.tool()
def bt_forex_screen_multitf(
    symbols: list = None,
    oos_months: int = 6,
    capital: float = 10000.0,
    n_bars: int = 5000,
) -> str:
    """Run the Multi-Timeframe EMA strategy across all 7 major forex pairs and rank by OOS Sharpe.

    Pairs tested by default: EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, NZDUSD, USDCAD.
    Results ranked best → worst by out-of-sample Sharpe ratio.

    Args:
        symbols: Subset of pairs to test — omit to test all 7 majors
        oos_months: Months held out for OOS validation (default 6)
        capital: Starting capital per pair (default 10000)
        n_bars: 1H bars to fetch per pair (default 5000)
    """
    result = _bt_forex_screen_multitf(symbols, oos_months, capital, n_bars)
    return json.dumps(result, default=str)


@mcp.tool()
def bt_config_c_screen(
    symbols: list = None,
    oos_months: int = 6,
    capital: float = 10000.0,
    n_bars: int = 5000,
    commission: float = 0.00007,
) -> str:
    """Run Config C across all 7 major forex pairs via OANDA and rank by OOS Sharpe.

    Config C = best-performing drawdown-reduced configuration:
      - 0.5% risk per trade (vs 1% baseline)
      - EMA200 4H alignment filter (structural trend only)
      - ADX(14) > 20 AND rising (trend strength — no chop)
      - SL = 1.5× ATR, TP = 2:1 RR, close 75% at TP, 25% runner

    All 7 majors (EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, NZDUSD, USDCAD) are run
    against OANDA data (more reliable than FX_IDC for history).
    Returns per-pair backtest + OOS metrics plus a portfolio-level summary.

    Args:
        symbols: Subset of pairs to test — omit to test all 7 majors
        oos_months: Months held out for OOS validation (default 6)
        capital: Starting capital per pair (default 10000)
        n_bars: 4H bars to fetch per pair (default 5000 ≈ 3 years)
        commission: Per-trade commission fraction (default 0.00007 ≈ 0.7 pip)
    """
    result = _bt_config_c_screen(symbols, oos_months, capital, n_bars, commission)
    return json.dumps(result, default=str)


@mcp.tool()
def bt_compare_configs(
    symbol: str = "EURUSD",
    exchange: str = "OANDA",
    n_bars: int = 5000,
    oos_months: int = 6,
    capital: float = 10000.0,
    commission: float = 0.00007,
) -> str:
    """Run all 5 drawdown-reduction configurations side-by-side on the same dataset.

    Configs: Baseline (1% risk), A (0.5% risk), B (+EMA200), C (+ADX>20), D (+3:1 TP full stack).
    Returns a comparison table of backtest and OOS metrics for each config.

    Args:
        symbol: Forex pair to test (default EURUSD)
        exchange: TradingView exchange (default OANDA)
        n_bars: 4H bars to fetch (default 5000)
        oos_months: Months held out for OOS (default 6)
        capital: Starting capital (default 10000)
        commission: Per-trade commission fraction (default 0.00007)
    """
    result = _bt_compare_configs(symbol, exchange, n_bars, oos_months, capital, commission)
    return json.dumps(result, default=str)


@mcp.tool()
def bt_before_after(
    symbol: str = "EURUSD",
    exchange: str = "OANDA",
    n_bars: int = 5000,
    oos_months: int = 6,
    capital: float = 10000.0,
    commission: float = 0.00007,
) -> str:
    """Compare Config C (EMA crossover) vs EMAPullbackMomentum strategy side-by-side.

    BEFORE: MultiTFEmaCross Config C — 0.5% risk, EMA200, ADX>20, 2:1 TP
    AFTER:  EMAPullbackMomentum — pullback retest, MACD+RSI filters, 3-level TP (1.5R/3R/runner)

    Returns backtest and OOS metrics for both strategies, plus OOS efficiency ratio.

    Args:
        symbol: Forex pair (default EURUSD)
        exchange: TradingView exchange (default OANDA)
        n_bars: 4H bars to fetch (default 5000)
        oos_months: Months held out for OOS (default 6)
        capital: Starting capital (default 10000)
        commission: Per-trade commission fraction (default 0.00007)
    """
    result = _bt_before_after(symbol, exchange, n_bars, oos_months, capital, commission)
    return json.dumps(result, default=str)
