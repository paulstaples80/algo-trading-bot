#!/usr/bin/env python3
"""
tradelocker_sync.py — Pull today's closed trades from Tradelocker.

Authenticates with the Tradelocker API, fetches filled orders for a given date,
pairs open/close orders into complete trades, and outputs pre-fill JSON for
the trade log agent to consume.

Usage:
    python3 tradelocker_sync.py              # today's trades
    python3 tradelocker_sync.py 2026-05-27   # specific date

Requires in .env:
    TRADELOCKER_EMAIL
    TRADELOCKER_PASSWORD
    TRADELOCKER_SERVER    (shown on the login screen, e.g. "OSP-DEMO")
    TRADELOCKER_BASE_URL  (e.g. "https://demo.tradelocker.com/backend-api")
"""

import os
import sys
import json
import requests
from pathlib import Path
from datetime import datetime, timezone, date
from typing import Optional
from dotenv import load_dotenv

# Always load .env from the same directory as this script,
# regardless of where the agent's working directory is.
load_dotenv(Path(__file__).parent / ".env")

BASE_URL    = os.getenv("TRADELOCKER_BASE_URL", "https://demo.tradelocker.com/backend-api")
EMAIL       = os.getenv("TRADELOCKER_EMAIL", "")
PASSWORD    = os.getenv("TRADELOCKER_PASSWORD", "")
SERVER      = os.getenv("TRADELOCKER_SERVER", "")
MIN_DATE    = date(2026, 5, 28)   # only import trades from this date forward
ACCOUNT_ID  = 1952884             # prop firm trading account — do not change

# Map Tradelocker instrument names to trade log names
INSTRUMENT_MAP = {
    "NAS100": "NQ", "NASDAQ100": "NQ", "US100": "NQ", "NQ100": "NQ",
    "USTEC": "NQ", "USTEC.M": "NQ",
    "GER40": "DAX", "GER30": "DAX", "DAX40": "DAX", "DE40": "DAX", "DE40.M": "DAX",
    "SPX500": "ES", "US500": "ES", "SP500": "ES", "SPX": "ES", "US500.M": "ES",
    "UK100": "FTSE100", "FTSE": "FTSE100", "UK100.M": "FTSE100",
    "US30": "DOW", "US30.M": "DOW",
}


def _col_ids(config_section: dict) -> list[str]:
    """Extract column id strings from a config section's 'columns' list."""
    return [col["id"] for col in config_section.get("columns", [])]


def _rows_to_dicts(rows: list, col_ids: list[str]) -> list[dict]:
    """Convert Tradelocker column-based response rows to list of dicts."""
    return [dict(zip(col_ids, row)) for row in rows]


def auth() -> tuple[str, int, int]:
    """Authenticate and return (access_token, account_id, acc_num)."""
    resp = requests.post(
        f"{BASE_URL}/auth/jwt/token",
        json={"email": EMAIL, "password": PASSWORD, "server": SERVER},
        timeout=15,
    )
    resp.raise_for_status()
    access_token = resp.json()["accessToken"]

    headers = {"Authorization": f"Bearer {access_token}"}
    resp = requests.get(f"{BASE_URL}/auth/jwt/all-accounts", headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    # Handle both list and dict response shapes
    accounts = data if isinstance(data, list) else data.get("accounts", [data])

    # Find the specific account — never fall back to another account
    account = next((a for a in accounts if int(a["id"]) == ACCOUNT_ID), None)
    if account is None:
        raise ValueError(
            f"Account {ACCOUNT_ID} not found. Available accounts: "
            + ", ".join(str(a["id"]) for a in accounts)
        )

    return access_token, int(account["id"]), int(account["accNum"])


def get_config(token: str, acc_num: int) -> dict:
    """Fetch /trade/config — returns column definitions for all data types."""
    headers = {"Authorization": f"Bearer {token}", "accNum": str(acc_num)}
    resp = requests.get(f"{BASE_URL}/trade/config", headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json().get("d", {})


def get_instruments(token: str, account_id: int, acc_num: int) -> dict[int, str]:
    """Return {tradableInstrumentId: name} from the instruments endpoint."""
    headers = {
        "Authorization": f"Bearer {token}",
        "accNum": str(acc_num),
    }
    resp = requests.get(
        f"{BASE_URL}/trade/accounts/{account_id}/instruments",
        headers=headers,
        params={"routeId": "INFO"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json().get("d", {})

    instruments: dict[int, str] = {}

    # Response may be column-based or a list of dicts
    if "instruments" in data:
        raw = data["instruments"]
        if raw and isinstance(raw[0], list):
            # Column-based: need field names from the same response
            fields = [f["id"] for f in data.get("instrumentsFields", [])]
            if not fields:
                fields = ["tradableInstrumentId", "id", "name", "description", "type"]
            for row in raw:
                item = dict(zip(fields, row))
                iid = item.get("tradableInstrumentId") or item.get("id")
                name = item.get("name", "")
                if iid:
                    instruments[int(iid)] = name
        else:
            for item in raw:
                iid = item.get("tradableInstrumentId") or item.get("id")
                name = item.get("name", "")
                if iid:
                    instruments[int(iid)] = name

    return instruments


def get_orders_history(
    token: str,
    account_id: int,
    acc_num: int,
    col_ids: list[str],
    target_date: date,
) -> list[dict]:
    """Return filled orders for target_date as list of dicts."""
    start_ms = int(datetime(
        target_date.year, target_date.month, target_date.day, tzinfo=timezone.utc
    ).timestamp() * 1000)
    end_ms = start_ms + 86_400_000

    headers = {
        "Authorization": f"Bearer {token}",
        "accNum": str(acc_num),
    }
    resp = requests.get(
        f"{BASE_URL}/trade/accounts/{account_id}/ordersHistory",
        headers=headers,
        params={"from": start_ms, "to": end_ms},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json().get("d", {})

    raw_orders = data.get("ordersHistory", [])
    if not raw_orders:
        return []

    # Convert column-based rows to dicts if needed
    if isinstance(raw_orders[0], list):
        orders = _rows_to_dicts(raw_orders, col_ids)
    else:
        orders = raw_orders

    # Filter for filled status only
    return [
        o for o in orders
        if str(o.get("status", "")).lower() in ("filled", "2", "fill")
    ]




def pair_trades(
    orders: list[dict],
    instruments: dict[int, str],
) -> list[dict]:
    """
    Group orders by positionId into complete trades.
    Each trade has one opening order and one (or more) closing orders.
    """

    by_position: dict = {}
    for order in orders:
        pos_id = order.get("positionId") or order.get("id")
        by_position.setdefault(pos_id, []).append(order)

    trades = []
    for pos_id, pos_orders in by_position.items():
        pos_orders.sort(key=lambda o: o.get("createdDate", 0))

        opening = pos_orders[0]
        closing = pos_orders[-1] if len(pos_orders) > 1 else None

        instrument_id = int(opening.get("tradableInstrumentId") or opening.get("instrumentId") or 0)
        raw_name = instruments.get(instrument_id, str(instrument_id))
        log_name = INSTRUMENT_MAP.get(raw_name.upper(), raw_name)

        side = str(opening.get("side", "")).lower()
        entry_price = float(opening.get("avgPrice") or opening.get("price") or 0)
        exit_price  = float((closing.get("avgPrice") or closing.get("price") or 0)) if closing else 0.0
        qty         = float(opening.get("filledQty") or opening.get("qty") or 0)
        sl          = float(opening.get("stopLoss") or 0)
        tp          = float(opening.get("takeProfit") or 0)

        entry_ts = int(opening.get("createdDate") or 0)
        entry_time = (
            datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).strftime("%H:%M UTC")
            if entry_ts > 0 else ""
        )

        pnl_pts = 0.0
        if exit_price and entry_price:
            pnl_pts = round(
                exit_price - entry_price if side == "buy" else entry_price - exit_price, 2
            )

        pts_at_risk = round(abs(entry_price - sl), 2) if sl and entry_price else 0.0
        actual_rr   = round(pnl_pts / pts_at_risk, 2) if pts_at_risk else 0.0

        outcome = "win" if pnl_pts > 0 else ("loss" if pnl_pts < 0 else "no_trade")
        is_closed = closing is not None

        # ── Account-currency P&L ──────────────────────────────────────────────
        # This broker's API (demo.tradelocker.com) does not return account-currency
        # P&L in ordersHistory and the positionsHistory endpoint returns 404.
        # pnl_gbp must be entered manually when logging the trade.

        trades.append({
            "position_id":        pos_id,
            "instrument":         log_name,
            "instrument_raw":     raw_name,
            "side":               side,
            "is_closed":          is_closed,
            "entry_price":        entry_price,
            "exit_price":         exit_price if is_closed else None,
            "entry_time":         entry_time,
            "position_size_lots": qty,
            "stop_loss":          sl,
            "take_profit":        tp,
            "pts_at_risk":        pts_at_risk,
            "pnl_points":         pnl_pts if is_closed else None,
            "pnl_gbp":            None,   # not available from this broker's API — enter manually
            "actual_rr_achieved": actual_rr if is_closed and pts_at_risk else None,
            "outcome":            outcome if is_closed else "open",
        })

    return trades


def main() -> None:
    if not all([EMAIL, PASSWORD, SERVER]):
        print(json.dumps({
            "error": "Missing credentials. Set TRADELOCKER_EMAIL, TRADELOCKER_PASSWORD, "
                     "and TRADELOCKER_SERVER in .env"
        }))
        sys.exit(1)

    target_date = date.today()
    if len(sys.argv) > 1:
        try:
            target_date = date.fromisoformat(sys.argv[1])
        except ValueError:
            print(json.dumps({"error": f"Invalid date: {sys.argv[1]} — use YYYY-MM-DD"}))
            sys.exit(1)

    if target_date < MIN_DATE:
        print(json.dumps({
            "error": f"Date {target_date} is before the sync start date ({MIN_DATE}). "
                     "Only trades from 28 May 2026 onwards are imported."
        }))
        sys.exit(1)

    try:
        token, account_id, acc_num = auth()
        config = get_config(token, acc_num)

        history_col_ids = _col_ids(config.get("ordersHistoryConfig", {}))
        if not history_col_ids:
            # Fallback to known field names from the Python library types
            history_col_ids = [
                "id", "tradableInstrumentId", "routeId", "qty", "side", "type",
                "status", "filledQty", "avgPrice", "price", "stopPrice", "validity",
                "expireDate", "createdDate", "lastModified", "isOpen", "positionId",
                "stopLoss", "stopLossType", "takeProfit", "takeProfitType", "strategyId",
            ]

        instruments = get_instruments(token, account_id, acc_num)
        orders      = get_orders_history(token, account_id, acc_num, history_col_ids, target_date)
        trades      = pair_trades(orders, instruments)

        print(json.dumps({
            "date":          target_date.isoformat(),
            "account_id":    account_id,
            "trades_found":  len(trades),
            "trades":        trades,
        }, indent=2))

    except requests.HTTPError as e:
        body = e.response.text[:300] if e.response else ""
        print(json.dumps({"error": f"HTTP {e.response.status_code}: {body}"}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
