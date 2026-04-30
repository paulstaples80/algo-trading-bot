#!/usr/bin/env python3
"""
FX Daily Bias Scanner
=====================
28 pairs | Daily swing high/low (3-candle pullback confirmation)
         | 4H EMA 20/50/200 alignment filter

Bias classifications:
  BULL    — confirmed swing low + 3-candle rally; HH/HL structure
  BEAR    — confirmed swing high + 3-candle pullback; LH/LL structure
  RANGING — no clear structure

Priority markets = Daily bias confirmed AND 4H EMAs aligned in same direction.

Data source : Alpha Vantage FX_DAILY + FX_INTRADAY (60min resampled to 4H)
              Cache: JSON files in .cache/ — one fetch per pair per day.
Rate limits : Free tier = 25 req/day (covers ~12 pairs). Premium = 75–300+/min.
              Script sleeps 12s between requests on free tier to stay safe.
              Set AV_PREMIUM=1 in .env to disable the sleep.
Notifications: Slack + optional email
"""
from __future__ import annotations

import json
import os
import smtplib
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# ── 28 FX pairs ───────────────────────────────────────────────────────────────
PAIRS = [
    # 8 majors + EURAUD
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
    "USDCAD", "AUDUSD", "NZDUSD", "EURJPY",
    "EURAUD",
]

# ── Config ────────────────────────────────────────────────────────────────────
SWING_BARS       = 2     # bars each side to confirm a swing point
PULLBACK_CANDLES = 3     # consecutive candles needed to confirm pullback
CACHE_DIR        = Path(".cache")

# Alpha Vantage
AV_API_KEY  = os.getenv("ALPHA_VANTAGE_API_KEY", "")
AV_BASE_URL = "https://www.alphavantage.co/query"
AV_PREMIUM  = os.getenv("AV_PREMIUM", "0") == "1"   # set to skip rate-limit sleep
AV_SLEEP    = 12.0   # seconds between requests on free tier (25/day = ~1/57s; 12s is safe)

# Slack / email — all read from .env
SLACK_BOT_TOKEN  = os.getenv("SLACK_BOT_TOKEN")
SLACK_CHANNEL    = os.getenv("SLACK_CHANNEL", "#trading-bias")
EMAIL_SENDER     = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD   = os.getenv("EMAIL_PASSWORD")
EMAIL_RECIPIENT  = os.getenv("EMAIL_RECIPIENT")
SMTP_HOST        = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT        = int(os.getenv("SMTP_PORT", "587"))


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class PairResult:
    pair: str
    daily_bias: str          # BULL | BEAR | RANGING
    h4_aligned: bool         # True if 4H EMAs confirm daily bias
    daily_note: str          # brief reason string
    h4_note: str             # EMA levels / alignment detail
    priority: bool           # daily_bias != RANGING and h4_aligned


# ── Alpha Vantage helpers ─────────────────────────────────────────────────────

def _av_request(params: dict) -> Optional[dict]:
    """Fire one Alpha Vantage request, return parsed JSON or None on error."""
    params["apikey"] = AV_API_KEY
    try:
        resp = requests.get(AV_BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        # AV returns {"Note": "..."} when rate-limited, {"Information": "..."} on bad key
        if "Note" in data:
            print("  [WARN] Alpha Vantage rate limit hit — consider upgrading plan.", file=sys.stderr)
            return None
        if "Information" in data:
            print(f"  [WARN] Alpha Vantage: {data['Information']}", file=sys.stderr)
            return None
        return data
    except Exception as e:
        print(f"  [WARN] AV request failed: {e}", file=sys.stderr)
        return None


def _rate_limit_sleep():
    """Sleep between requests unless on a premium plan."""
    if not AV_PREMIUM:
        time.sleep(AV_SLEEP)


def _cache_path(pair: str, suffix: str) -> Path:
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"{pair}_{suffix}_{today}.json"


def _load_cache(path: Path) -> Optional[dict]:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _save_cache(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f)


def _parse_av_ohlc(raw: dict, series_key: str) -> Optional[pd.DataFrame]:
    """Convert Alpha Vantage time-series dict to a sorted OHLC DataFrame."""
    ts = raw.get(series_key)
    if not ts:
        return None
    records = []
    for dt_str, bar in ts.items():
        try:
            records.append({
                "datetime": pd.Timestamp(dt_str),
                "open":  float(bar["1. open"]),
                "high":  float(bar["2. high"]),
                "low":   float(bar["3. low"]),
                "close": float(bar["4. close"]),
            })
        except (KeyError, ValueError):
            continue
    if not records:
        return None
    df = pd.DataFrame(records).set_index("datetime").sort_index()
    return df


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_daily(pair: str) -> Optional[pd.DataFrame]:
    """
    Fetch daily OHLC via Alpha Vantage FX_DAILY.
    Uses today's cache if available (avoids burning API quota on re-runs).
    """
    cache = _cache_path(pair, "daily")
    raw = _load_cache(cache)

    if raw is None:
        from_sym, to_sym = pair[:3], pair[3:]
        raw = _av_request({
            "function":    "FX_DAILY",
            "from_symbol": from_sym,
            "to_symbol":   to_sym,
            "outputsize":  "compact",   # last 100 bars — enough for swing detection
        })
        _rate_limit_sleep()
        if raw is None:
            return None
        _save_cache(cache, raw)

    df = _parse_av_ohlc(raw, "Time Series FX (Daily)")
    if df is None or len(df) < 30:
        return None
    return df


def fetch_4h(pair: str) -> Optional[pd.DataFrame]:
    """
    Fetch 60-min OHLC via Alpha Vantage FX_INTRADAY, resample to 4H.
    Uses today's cache if available.
    """
    cache = _cache_path(pair, "60min")
    raw = _load_cache(cache)

    if raw is None:
        from_sym, to_sym = pair[:3], pair[3:]
        raw = _av_request({
            "function":    "FX_INTRADAY",
            "from_symbol": from_sym,
            "to_symbol":   to_sym,
            "interval":    "60min",
            "outputsize":  "full",      # ~30 days of hourly bars
        })
        _rate_limit_sleep()
        if raw is None:
            return None
        _save_cache(cache, raw)

    df = _parse_av_ohlc(raw, "Time Series FX (60min)")
    if df is None or len(df) < 50:
        return None

    # Resample to 4H
    df_4h = df.resample("4h").agg({
        "open": "first", "high": "max", "low": "min", "close": "last"
    }).dropna()
    return df_4h if len(df_4h) >= 50 else None


# ── Swing point detection ─────────────────────────────────────────────────────

def find_swing_highs(highs: np.ndarray, n: int = SWING_BARS) -> list[int]:
    """Return indices of confirmed swing highs (n bars each side)."""
    result = []
    for i in range(n, len(highs) - n):
        if all(highs[i] > highs[i - j] for j in range(1, n + 1)) and \
           all(highs[i] > highs[i + j] for j in range(1, n + 1)):
            result.append(i)
    return result


def find_swing_lows(lows: np.ndarray, n: int = SWING_BARS) -> list[int]:
    """Return indices of confirmed swing lows (n bars each side)."""
    result = []
    for i in range(n, len(lows) - n):
        if all(lows[i] < lows[i - j] for j in range(1, n + 1)) and \
           all(lows[i] < lows[i + j] for j in range(1, n + 1)):
            result.append(i)
    return result


def has_pullback_after(values: np.ndarray, pivot_idx: int,
                       direction: str, k: int = PULLBACK_CANDLES) -> bool:
    """
    Check if there are k consecutive bars after pivot_idx showing a pullback.

    direction='down'  — after swing HIGH: k bars of lower highs (pullback down)
    direction='up'    — after swing LOW:  k bars of higher lows (pullback up)
    """
    start = pivot_idx + 1
    if start + k > len(values):
        return False
    window = values[start: start + k]
    if direction == "down":
        return all(window[i] < window[i - 1] for i in range(1, k))
    else:  # up
        return all(window[i] > window[i - 1] for i in range(1, k))


# ── Daily bias ────────────────────────────────────────────────────────────────

def daily_bias(df: pd.DataFrame) -> tuple[str, str]:
    """
    Returns (bias, note) where bias is BULL | BEAR | RANGING.

    Logic
    -----
    1. Find all confirmed swing highs and lows (2 bars each side).
    2. A swing high is 'pullback-confirmed' if the 3 candles after it
       each have a lower high (price pulls away from the high).
    3. A swing low is 'pullback-confirmed' if the 3 candles after it
       each have a higher low (price bounces away from the low).
    4. Collect the two most recent confirmed highs and two most recent
       confirmed lows.
    5. BULL  = most recent confirmed SH > previous confirmed SH  (HH)
               AND most recent confirmed SL > previous confirmed SL (HL)
    6. BEAR  = most recent confirmed SH < previous confirmed SH  (LH)
               AND most recent confirmed SL < previous confirmed SL (LL)
    7. RANGING = anything else.
    """
    highs  = df["high"].values
    lows   = df["low"].values

    sh_idxs = find_swing_highs(highs)
    sl_idxs = find_swing_lows(lows)

    # Filter to those with a confirmed 3-candle pullback
    confirmed_sh = [i for i in sh_idxs
                    if has_pullback_after(highs, i, "down")]
    confirmed_sl = [i for i in sl_idxs
                    if has_pullback_after(lows,  i, "up")]

    if len(confirmed_sh) < 2 or len(confirmed_sl) < 2:
        return "RANGING", "insufficient confirmed swing points"

    # Most recent two of each
    sh1, sh2 = confirmed_sh[-1], confirmed_sh[-2]   # sh1 is more recent
    sl1, sl2 = confirmed_sl[-1], confirmed_sl[-2]

    sh1_val, sh2_val = highs[sh1], highs[sh2]
    sl1_val, sl2_val = lows[sl1],  lows[sl2]

    hh = sh1_val > sh2_val   # Higher High
    hl = sl1_val > sl2_val   # Higher Low
    lh = sh1_val < sh2_val   # Lower High
    ll = sl1_val < sl2_val   # Lower Low

    if hh and hl:
        note = f"HH({sh2_val:.5f}→{sh1_val:.5f}) HL({sl2_val:.5f}→{sl1_val:.5f})"
        return "BULL", note
    elif lh and ll:
        note = f"LH({sh2_val:.5f}→{sh1_val:.5f}) LL({sl2_val:.5f}→{sl1_val:.5f})"
        return "BEAR", note
    else:
        note = f"HH={hh} HL={hl} — mixed structure"
        return "RANGING", note


# ── 4H EMA alignment ─────────────────────────────────────────────────────────

def h4_ema_aligned(df: pd.DataFrame, bias: str) -> tuple[bool, str]:
    """
    Returns (aligned, note).

    BULL aligned : EMA20 > EMA50 > EMA200
    BEAR aligned : EMA20 < EMA50 < EMA200
    """
    close = df["close"]
    ema20  = close.ewm(span=20,  adjust=False).mean().iloc[-1]
    ema50  = close.ewm(span=50,  adjust=False).mean().iloc[-1]
    ema200 = close.ewm(span=200, adjust=False).mean().iloc[-1]

    note = f"EMA20={ema20:.5f} EMA50={ema50:.5f} EMA200={ema200:.5f}"

    if bias == "BULL":
        aligned = ema20 > ema50 > ema200
    elif bias == "BEAR":
        aligned = ema20 < ema50 < ema200
    else:
        aligned = False

    return aligned, note


# ── Analyse one pair ──────────────────────────────────────────────────────────

def analyse_pair(pair: str) -> PairResult:
    daily_df = fetch_daily(pair)
    h4_df    = fetch_4h(pair)

    if daily_df is None:
        return PairResult(pair, "RANGING", False, "no data", "no data", False)

    bias, d_note = daily_bias(daily_df)

    if h4_df is None or bias == "RANGING":
        h4_ok, h4_note = False, "n/a"
    else:
        h4_ok, h4_note = h4_ema_aligned(h4_df, bias)

    return PairResult(
        pair       = pair,
        daily_bias = bias,
        h4_aligned = h4_ok,
        daily_note = d_note,
        h4_note    = h4_note,
        priority   = (bias != "RANGING" and h4_ok),
    )


# ── Report formatting ─────────────────────────────────────────────────────────

BULL_EMOJI = "📈"
BEAR_EMOJI = "📉"
RANGE_EMOJI = "↔️"
TICK = "✅"
CROSS = "❌"


def format_slack_report(results: list[PairResult]) -> str:
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    priority_bull  = [r for r in results if r.priority and r.daily_bias == "BULL"]
    priority_bear  = [r for r in results if r.priority and r.daily_bias == "BEAR"]
    bias_only_bull = [r for r in results if not r.priority and r.daily_bias == "BULL"]
    bias_only_bear = [r for r in results if not r.priority and r.daily_bias == "BEAR"]
    ranging        = [r for r in results if r.daily_bias == "RANGING"]

    lines = [
        f"🧭 *FX DAILY BIAS REPORT — {now}*",
        "",
        f"*PRIORITY MARKETS* (Daily bias + 4H EMA aligned) — {len(priority_bull)+len(priority_bear)} pairs",
        "━" * 42,
    ]

    if priority_bull:
        lines.append(f"\n{BULL_EMOJI} *BULLISH SETUPS*")
        for r in priority_bull:
            lines.append(f"  `{r.pair:<8}` {TICK} Daily: {r.daily_note}")
            lines.append(f"  {'':8}  4H:    {r.h4_note}")
    else:
        lines.append(f"\n{BULL_EMOJI} *BULLISH SETUPS* — none")

    if priority_bear:
        lines.append(f"\n{BEAR_EMOJI} *BEARISH SETUPS*")
        for r in priority_bear:
            lines.append(f"  `{r.pair:<8}` {TICK} Daily: {r.daily_note}")
            lines.append(f"  {'':8}  4H:    {r.h4_note}")
    else:
        lines.append(f"\n{BEAR_EMOJI} *BEARISH SETUPS* — none")

    lines += ["", "─" * 42,
              "*DAILY BIAS ONLY* (4H EMAs not aligned — watch list)"]

    if bias_only_bull:
        pairs_str = "  " + "  ".join(
            f"`{r.pair}` {CROSS}4H" for r in bias_only_bull
        )
        lines.append(f"{BULL_EMOJI} Bullish watch: {pairs_str}")
    if bias_only_bear:
        pairs_str = "  " + "  ".join(
            f"`{r.pair}` {CROSS}4H" for r in bias_only_bear
        )
        lines.append(f"{BEAR_EMOJI} Bearish watch: {pairs_str}")
    if not bias_only_bull and not bias_only_bear:
        lines.append("  None")

    ranging_str = "  ".join(f"`{r.pair}`" for r in ranging)
    lines += [
        "",
        f"{RANGE_EMOJI} *RANGING* ({len(ranging)} pairs) — skip",
        f"  {ranging_str}" if ranging_str else "  None",
        "",
        f"_Scan complete. {len(results)} pairs analysed._",
    ]

    return "\n".join(lines)


def format_plain_report(results: list[PairResult]) -> str:
    """Plain-text version for email."""
    slack_text = format_slack_report(results)
    # Strip Slack markdown
    for ch in ["*", "`", "_"]:
        slack_text = slack_text.replace(ch, "")
    return slack_text


# ── Delivery ──────────────────────────────────────────────────────────────────

def send_slack(message: str) -> bool:
    if not SLACK_BOT_TOKEN:
        print("SLACK_BOT_TOKEN not set — skipping Slack.")
        return False
    try:
        from slack_sdk import WebClient
        client = WebClient(token=SLACK_BOT_TOKEN)
        client.chat_postMessage(channel=SLACK_CHANNEL, text=message)
        print(f"Slack message sent to {SLACK_CHANNEL}")
        return True
    except Exception as e:
        print(f"Slack error: {e}", file=sys.stderr)
        return False


def send_email(subject: str, body: str) -> bool:
    if not EMAIL_SENDER or not EMAIL_PASSWORD or not EMAIL_RECIPIENT:
        print("Email credentials not set — skipping email.")
        return False
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"]    = EMAIL_SENDER
        msg["To"]      = EMAIL_RECIPIENT
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.sendmail(EMAIL_SENDER, EMAIL_RECIPIENT, msg.as_string())
        print(f"Email sent to {EMAIL_RECIPIENT}")
        return True
    except Exception as e:
        print(f"Email error: {e}", file=sys.stderr)
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"FX Bias Scanner — {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Scanning {len(PAIRS)} pairs...\n")

    results: list[PairResult] = []
    for pair in PAIRS:
        print(f"  {pair}...", end=" ", flush=True)
        r = analyse_pair(pair)
        results.append(r)
        icon = BULL_EMOJI if r.daily_bias == "BULL" else (BEAR_EMOJI if r.daily_bias == "BEAR" else RANGE_EMOJI)
        aligned_str = f" + 4H {TICK}" if r.h4_aligned else ""
        print(f"{icon} {r.daily_bias}{aligned_str}")

    # Sort: priority first, then by bias, then alphabetical
    results.sort(key=lambda r: (
        0 if r.priority else (1 if r.daily_bias != "RANGING" else 2),
        r.daily_bias,
        r.pair,
    ))

    slack_report = format_slack_report(results)
    plain_report = format_plain_report(results)

    print("\n" + "=" * 50)
    print(plain_report)
    print("=" * 50 + "\n")

    # Send
    send_slack(slack_report)
    subject = f"FX Bias Report — {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d')}"
    send_email(subject, plain_report)


if __name__ == "__main__":
    main()
