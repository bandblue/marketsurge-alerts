import json
import os
import sys
from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

WATCHLIST_FILE = "marketsurge_actionable_setups.csv"
STATE_FILE = "alerted_today.json"

EARLY_PROXIMITY_PCT = 0.02     # Within 2.0% below Pivot_Buy
EARLY_VOL_PACE_MIN = 1.25      # +25% above expected intraday volume pace
BREAKOUT_VOL_PACE_MIN = 1.40   # +40% above expected intraday volume pace


def get_market_progress() -> tuple[bool, float, str]:
    et = ZoneInfo("America/New_York")
    now_et = datetime.now(et)
    today_str = now_et.strftime("%Y-%m-%d")

    if now_et.weekday() >= 5:
        return False, 0.0, today_str

    market_open = datetime.combine(now_et.date(), time(9, 30), tzinfo=et)
    market_close = datetime.combine(now_et.date(), time(16, 0), tzinfo=et)

    if now_et < market_open or now_et > market_close:
        return False, 0.0, today_str

    elapsed_minutes = (now_et - market_open).total_seconds() / 60.0
    # Account for ~15-min Yahoo Finance intraday volume delay
    effective_minutes = max(elapsed_minutes - 15.0, 15.0)
    progress = min(effective_minutes / 390.0, 1.0)
    return True, progress, today_str


def load_state(today_str: str) -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("date") == today_str:
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {"date": today_str, "sent": []}


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def send_ntfy(topic: str, title: str, message: str, priority: str = "high", tags: str = "chart_with_upwards_trend") -> None:
    url = f"https://ntfy.sh/{topic}"
    headers = {"Title": title, "Priority": priority, "Tags": tags}
    resp = requests.post(url, data=message.encode("utf-8"), headers=headers, timeout=15)
    resp.raise_for_status()


def format_alert_status(ticker: str, sent_keys: set[str]) -> str:
    if f"{ticker}_BREAKOUT" in sent_keys:
        return "🚀 TIER 2: BREAKOUT"
    if f"{ticker}_APPROACHING" in sent_keys:
        return "👀 TIER 1: EARLY SETUP"
    return "⏳ Watching"


def main() -> None:
    ntfy_topic = os.environ.get("NTFY_TOPIC", "").strip()
    if not ntfy_topic:
        raise ValueError("Missing NTFY_TOPIC environment variable.")

    if "--test" in sys.argv:
        send_ntfy(
            ntfy_topic,
            "TEST BREAKOUT: NVDA crossed $140.00",
            "Test alert from MarketSurge Breakout Watcher.\nYour Galaxy S24 ntfy channel is active and working.",
            priority="urgent",
            tags="rocket,white_check_mark",
        )
        print("[SENT] Test ntfy notification delivered.")
        return

    is_open, session_progress, today_str = get_market_progress()
    if not is_open:
        print("Market is currently closed. Exiting.")
        return

    if not os.path.exists(WATCHLIST_FILE):
        print(f"{WATCHLIST_FILE} not found. Exiting.")
        return

    df = pd.read_csv(WATCHLIST_FILE)
    if df.empty:
        print("Watchlist is empty. Exiting.")
        return

    pivot_col = "Pivot_Buy" if "Pivot_Buy" in df.columns else "52W_High_Pivot"
    ticker_col = "Ticker" if "Ticker" in df.columns else "Symbol"

    state = load_state(today_str)
    sent_keys = set(state.get("sent", []))
    state_changed = False

    tickers = df[ticker_col].dropna().astype(str).str.strip().unique().tolist()
    hist = yf.download(tickers, period="60d", interval="1d", group_by="ticker", progress=False, threads=True)
    is_multi = isinstance(hist.columns, pd.MultiIndex)

    for _, row in df.iterrows():
        ticker = str(row[ticker_col]).strip()
        try:
            pivot_buy = float(row[pivot_col])
        except (ValueError, TypeError):
            continue

        if pivot_buy <= 0:
            continue

        try:
            t_df = hist[ticker].dropna() if is_multi else hist.dropna()
        except KeyError:
            continue

        if len(t_df) < 22:
            continue

        today_bar = t_df.iloc[-1]
        prior_bars = t_df.iloc[:-1]

        last_price = float(today_bar["Close"])
        today_vol = float(today_bar["Volume"])
        prior_high = float(prior_bars["High"].iloc[-1])
        avg_vol_50d = float(prior_bars["Volume"].tail(50).mean())

        if avg_vol_50d <= 0:
            continue

        expected_vol = avg_vol_50d * session_progress
        vol_pace = today_vol / expected_vol if expected_vol > 0 else 0.0
        vol_pct_change = (vol_pace - 1.0) * 100.0
        pct_from_pivot = ((last_price - pivot_buy) / pivot_buy) * 100.0
        early_floor = pivot_buy * (1.0 - EARLY_PROXIMITY_PCT)

        # TIER 2: Official Breakout Above Pivot_Buy
        breakout_key = f"{ticker}_BREAKOUT"
        if last_price >= pivot_buy and vol_pace >= BREAKOUT_VOL_PACE_MIN:
            if breakout_key not in sent_keys:
                title = f"BREAKOUT: {ticker} crossed ${pivot_buy:.2f}"
                msg = (
                    f"{ticker} is at ${last_price:.2f} ({pct_from_pivot:+.2f}% vs pivot ${pivot_buy:.2f}).\n"
                    f"Vol Pace: {vol_pct_change:+.0f}% vs 50d avg | Prior High: ${prior_high:.2f}"
                )
                send_ntfy(ntfy_topic, title, msg, priority="urgent", tags="rocket,chart_with_upwards_trend")
                sent_keys.add(breakout_key)
                state_changed = True
                print(f"[SENT] {title}")

        # TIER 1: Early Cheat / Approaching Pivot
        elif early_floor <= last_price < pivot_buy and last_price > prior_high and vol_pace >= EARLY_VOL_PACE_MIN:
            early_key = f"{ticker}_APPROACHING"
            if early_key not in sent_keys:
                title = f"EARLY SETUP: {ticker} approaching ${pivot_buy:.2f}"
                msg = (
                    f"{ticker} at ${last_price:.2f} ({pct_from_pivot:+.2f}% below pivot ${pivot_buy:.2f}).\n"
                    f"Cleared Prior High (${prior_high:.2f}) | Vol Pace: {vol_pct_change:+.0f}%"
                )
                send_ntfy(ntfy_topic, title, msg, priority="high", tags="eyes,chart_with_upwards_trend")
                sent_keys.add(early_key)
                state_changed = True
                print(f"[SENT] {title}")

    # Sync Alert_Status column into marketsurge_actionable_setups.csv for Dashboard & Drive
    new_statuses = [format_alert_status(str(t).strip(), sent_keys) for t in df[ticker_col]]
    old_statuses = df["Alert_Status"].tolist() if "Alert_Status" in df.columns else []

    if state_changed or new_statuses != old_statuses:
        df["Alert_Status"] = new_statuses
        cols = df.columns.tolist()
        cols.remove("Alert_Status")
        insert_idx = 2 if len(cols) >= 2 else len(cols)
        cols.insert(insert_idx, "Alert_Status")
        df = df[cols]
        df.to_csv(WATCHLIST_FILE, index=False)

    if state_changed:
        state["sent"] = sorted(sent_keys)
        save_state(state)


if __name__ == "__main__":
    main()
