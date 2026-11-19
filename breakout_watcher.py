import os
import json
from datetime import datetime, time
from zoneinfo import ZoneInfo
import pandas as pd
import yfinance as yf
import requests

CSV_FILE = "marketsurge_actionable_setups.csv"
STATE_FILE = "alerted_today.json"
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "").strip()

def is_market_open_et():
    now_et = datetime.now(ZoneInfo("America/New_York"))
    if now_et.weekday() >= 5:  # Saturday/Sunday
        return False, now_et
    market_start = time(9, 30)
    market_end = time(16, 5)
    return (market_start <= now_et.time() <= market_end), now_et

def load_state(today_str):
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
            if data.get("date") == today_str:
                return set(data.get("alerted", []))
        except Exception:
            pass
    return set()

def save_state(today_str, alerted_set):
    with open(STATE_FILE, "w") as f:
        json.dump({"date": today_str, "alerted": sorted(list(alerted_set))}, f)

def send_ntfy(title, message, priority="default"):
    if not NTFY_TOPIC:
        print("NTFY_TOPIC secret not set; skipping push notification.")
        return
    url = f"https://ntfy.sh/{NTFY_TOPIC}"
    headers = {
        "Title": title,
        "Priority": priority,
        "Tags": "chart_with_upwards_trend,rocket"
    }
    resp = requests.post(url, data=message.encode("utf-8"), headers=headers, timeout=15)
    resp.raise_for_status()

def check_breakouts():
    open_now, now_et = is_market_open_et()
    if not open_now:
        print(f"[{now_et.strftime('%Y-%m-%d %H:%M %Z')}] Market closed. Exiting.")
        return

    if not os.path.exists(CSV_FILE):
        print(f"Missing {CSV_FILE}. Exiting.")
        return

    df = pd.read_csv(CSV_FILE)
    if df.empty or "Pivot_Buy" not in df.columns:
        print("No actionable setups in CSV.")
        return

    today_str = now_et.strftime("%Y-%m-%d")
    alerted = load_state(today_str)

    # Calculate fraction of trading day elapsed (390 total minutes from 9:30 to 16:00 ET)
    mins_since_open = (now_et.hour * 60 + now_et.minute) - (9 * 60 + 30)
    mins_since_open = max(5, min(390, mins_since_open))
    day_fraction = mins_since_open / 390.0

    for _, row in df.iterrows():
        ticker = str(row["Ticker"]).strip()
        if ticker in alerted or pd.isna(row["Pivot_Buy"]):
            continue

        pivot = float(row["Pivot_Buy"])
        buy_max = float(row.get("Buy_Zone_Max", pivot * 1.05))
        stop_loss = float(row.get("Stop_Loss_7.5Pct", pivot * 0.925))

        try:
            hist = yf.Ticker(ticker).history(period="3mo")
            if len(hist) < 20:
                continue

            live_price = float(hist["Close"].iloc[-1])
            today_vol = float(hist["Volume"].iloc[-1])
            avg_vol_50 = float(hist["Volume"].iloc[:-1].tail(50).mean())

            if avg_vol_50 <= 0:
                continue

            # Project full-day volume pace vs 50-day average
            projected_vol = today_vol / day_fraction
            vol_pace_pct = ((projected_vol - avg_vol_50) / avg_vol_50) * 100

            # Trigger when price crosses Pivot (within 5% buy zone) AND volume pace >= +40%
            if pivot <= live_price <= buy_max and vol_pace_pct >= 40.0:
                msg = (
                    f"{ticker} broke above ${pivot:.2f} pivot!\n"
                    f"Live Price: ${live_price:.2f} (Buy Zone up to ${buy_max:.2f})\n"
                    f"Vol Pace: +{vol_pace_pct:.0f}% vs 50D Avg\n"
                    f"7.5% Stop-Loss: ${stop_loss:.2f}"
                )
                send_ntfy(f"CAN SLIM Breakout: {ticker}", msg)
                print(f"ALERT SENT: {ticker} @ ${live_price:.2f} (Vol +{vol_pace_pct:.0f}%)")
                alerted.add(ticker)
            else:
                print(f"{ticker}: ${live_price:.2f} (Pivot ${pivot:.2f}) | Vol Pace {vol_pace_pct:+.0f}%")
        except Exception as e:
            print(f"Error checking {ticker}: {e}")

    save_state(today_str, alerted)

if __name__ == "__main__":
    check_breakouts()
