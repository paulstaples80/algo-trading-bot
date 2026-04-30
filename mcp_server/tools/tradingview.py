import os
from typing import Optional
import pandas as pd

# Lazy imports — only loaded when tools are called
_tv_client = None


def _get_tv_client():
    global _tv_client
    if _tv_client is None:
        from tvDatafeed import TvDatafeed
        username = os.getenv("TV_USERNAME", "")
        password = os.getenv("TV_PASSWORD", "")
        if username and password:
            _tv_client = TvDatafeed(username=username, password=password)
        else:
            _tv_client = TvDatafeed()
    return _tv_client


_TV_INTERVAL_MAP = {
    "1m": "in_1_minute",
    "3m": "in_3_minute",
    "5m": "in_5_minute",
    "15m": "in_15_minute",
    "30m": "in_30_minute",
    "45m": "in_45_minute",
    "1h": "in_1_hour",
    "2h": "in_2_hour",
    "3h": "in_3_hour",
    "4h": "in_4_hour",
    "1d": "in_daily",
    "1w": "in_weekly",
    "1M": "in_monthly",
}

_TA_INTERVAL_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "1d": "1D",
    "1w": "1W",
    "1M": "1M",
}


def tv_get_bars(symbol: str, exchange: str, timeframe: str, n_bars: int) -> dict:
    """Fetch OHLCV bars from TradingView.

    Args:
        symbol: Ticker symbol e.g. 'AAPL', 'EURUSD', 'BTCUSDT'
        exchange: Exchange e.g. 'NASDAQ', 'FOREX', 'BINANCE'
        timeframe: One of 1m/3m/5m/15m/30m/45m/1h/2h/3h/4h/1d/1w/1M
        n_bars: Number of bars to fetch (max ~5000)
    """
    from tvDatafeed import Interval

    if timeframe not in _TV_INTERVAL_MAP:
        return {"error": f"Invalid timeframe '{timeframe}'. Valid: {list(_TV_INTERVAL_MAP.keys())}"}

    interval_attr = _TV_INTERVAL_MAP[timeframe]
    interval = getattr(Interval, interval_attr)

    try:
        client = _get_tv_client()
        df = client.get_hist(
            symbol=symbol.upper(),
            exchange=exchange.upper(),
            interval=interval,
            n_bars=n_bars,
        )
    except Exception as e:
        return {"error": f"TvDatafeed error: {e}"}

    if df is None or df.empty:
        return {"error": f"No data returned for {symbol} on {exchange} ({timeframe})"}

    df = df.reset_index()
    df["datetime"] = df["datetime"].astype(str)
    df = df.rename(columns={"symbol": "symbol_col"})

    return {
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": timeframe,
        "bars": len(df),
        "start": str(df["datetime"].iloc[0]),
        "end": str(df["datetime"].iloc[-1]),
        "columns": list(df.columns),
        "data": df[["datetime", "open", "high", "low", "close", "volume"]].to_dict(orient="records"),
    }


def tv_get_indicators(symbol: str, screener: str, exchange: str, timeframe: str) -> dict:
    """Get technical analysis indicators from TradingView.

    Args:
        symbol: Ticker symbol e.g. 'AAPL'
        screener: Market screener — 'america', 'forex', 'crypto', 'cfd', 'australia', 'canada', 'egypt', 'hongkong', 'india', 'indonesia', 'malaysia', 'pakistan', 'philippines', 'saudi_arabia', 'singapore', 'taiwan', 'thailand', 'turkey', 'vietnam'
        exchange: Exchange e.g. 'NASDAQ', 'NYSE', 'BINANCE'
        timeframe: One of 1m/5m/15m/30m/1h/2h/4h/1d/1w/1M
    """
    from tradingview_ta import TA_Handler

    if timeframe not in _TA_INTERVAL_MAP:
        return {"error": f"Invalid timeframe '{timeframe}'. Valid: {list(_TA_INTERVAL_MAP.keys())}"}

    ta_interval = _TA_INTERVAL_MAP[timeframe]

    try:
        handler = TA_Handler(
            symbol=symbol.upper(),
            screener=screener.lower(),
            exchange=exchange.upper(),
            interval=ta_interval,
        )
        analysis = handler.get_analysis()
    except Exception as e:
        return {"error": f"tradingview-ta error: {e}"}

    return {
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": timeframe,
        "summary": {
            "recommendation": analysis.summary.get("RECOMMENDATION"),
            "buy": analysis.summary.get("BUY"),
            "sell": analysis.summary.get("SELL"),
            "neutral": analysis.summary.get("NEUTRAL"),
        },
        "oscillators": analysis.oscillators,
        "moving_averages": analysis.moving_averages,
        "indicators": analysis.indicators,
    }


def tv_screen(symbols: list[str], screener: str, exchange: str, timeframe: str) -> dict:
    """Run TradingView TA analysis across multiple symbols at once.

    Args:
        symbols: List of ticker symbols e.g. ['AAPL', 'MSFT', 'GOOGL']
        screener: Market screener e.g. 'america', 'forex', 'crypto'
        exchange: Exchange e.g. 'NASDAQ'
        timeframe: One of 1m/5m/15m/30m/1h/2h/4h/1d/1w/1M
    """
    from tradingview_ta import TA_Handler

    if timeframe not in _TA_INTERVAL_MAP:
        return {"error": f"Invalid timeframe '{timeframe}'. Valid: {list(_TA_INTERVAL_MAP.keys())}"}

    ta_interval = _TA_INTERVAL_MAP[timeframe]
    results = {}

    for symbol in symbols:
        try:
            handler = TA_Handler(
                symbol=symbol.upper(),
                screener=screener.lower(),
                exchange=exchange.upper(),
                interval=ta_interval,
            )
            analysis = handler.get_analysis()
            results[symbol] = {
                "recommendation": analysis.summary.get("RECOMMENDATION"),
                "buy": analysis.summary.get("BUY"),
                "sell": analysis.summary.get("SELL"),
                "neutral": analysis.summary.get("NEUTRAL"),
                "rsi": analysis.indicators.get("RSI"),
                "macd": analysis.indicators.get("MACD.macd"),
                "macd_signal": analysis.indicators.get("MACD.signal"),
                "ema_20": analysis.indicators.get("EMA20"),
                "ema_50": analysis.indicators.get("EMA50"),
                "ema_200": analysis.indicators.get("EMA200"),
                "volume": analysis.indicators.get("volume"),
            }
        except Exception as e:
            results[symbol] = {"error": str(e)}

    return {
        "screener": screener,
        "exchange": exchange,
        "timeframe": timeframe,
        "results": results,
    }
