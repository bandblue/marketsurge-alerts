import pandas as pd
import yfinance as yf

MIN_ATR_PCT = 1.2  # Minimum 14-day ATR / Close (%) to exclude buyout-pinned or dead-range stocks

df_rank = pd.read_csv("marketsurge_rankings.csv")
candidates = df_rank[(df_rank["Stage2_Trend"] == True) & (df_rank["RS_Rating"] >= 85)].copy()

print(f"Found {len(candidates)} Stage 2 leaders (RS >= 85) to scan for volatility, bases, and EPS growth...\n")


def analyze_price_and_atr(symbol: str):
    """
    Fetches 9 months of daily history, calculates 14-day Wilder ATR%,
    and detects Flat Base or Cup / Handle Base patterns.
    """
    df = yf.Ticker(symbol).history(period="9mo")
    if df.empty or len(df) < 60:
        return None, "No Base", None, None, None

    latest_close = float(df["Close"].iloc[-1])
    if latest_close <= 0:
        return None, "No Base", None, None, None

    # --- 14-Day Wilder ATR & ATR% ---
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_14 = float(tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
    atr_pct = round((atr_14 / latest_close) * 100, 2)

    # --- Base Pattern & Pivot Detection ---
    flat_win = df.tail(55)
    flat_high = float(flat_win["High"].max())
    flat_low = float(flat_win["Low"].min())
    flat_depth = ((flat_high - flat_low) / flat_high) * 100

    cup_win = df.tail(110)
    cup_high = float(cup_win["High"].max())
    cup_low = float(cup_win["Low"].min())
    cup_depth = ((cup_high - cup_low) / cup_high) * 100

    vol_10d = float(df["Volume"].tail(10).mean())
    vol_50d = float(df["Volume"].tail(50).mean())
    vol_dryup = vol_10d < vol_50d

    pattern = "Extended / No Base"
    pivot = flat_high

    if flat_depth <= 15.0 and latest_close >= flat_high * 0.92:
        pattern = "Flat Base" + (" (Vol Dry-Up)" if vol_dryup else "")
        pivot = flat_high
    elif 12.0 < cup_depth <= 33.0 and latest_close >= cup_high * 0.90:
        pattern = "Cup / Handle Base" + (" (Vol Dry-Up)" if vol_dryup else "")
        pivot = cup_high

    buy_zone_top = round(pivot * 1.05, 2)
    stop_loss = round(pivot * 0.925, 2)
    return atr_pct, pattern, round(pivot, 2), buy_zone_top, stop_loss


def get_quarterly_eps_growth(symbol: str):
    try:
        t = yf.Ticker(symbol)
        q_inc = t.quarterly_income_stmt
        if q_inc is None or q_inc.empty or q_inc.shape[1] < 5:
            info = t.info
            eg = info.get("earningsQuarterlyGrowth")
            rg = info.get("revenueGrowth")
            return (
                round(eg * 100, 1) if eg is not None else None,
                round(rg * 100, 1) if rg is not None else None,
            )

        eps_row = next((r for r in ["Diluted EPS", "Basic EPS"] if r in q_inc.index), None)
        rev_row = next((r for r in ["Total Revenue", "Operating Revenue"] if r in q_inc.index), None)

        eps_yoy = None
        rev_yoy = None

        if eps_row:
            eps_now = q_inc.loc[eps_row].iloc[0]
            eps_prev = q_inc.loc[eps_row].iloc[4]
            if pd.notna(eps_now) and pd.notna(eps_prev) and eps_prev != 0:
                eps_yoy = round(((eps_now - eps_prev) / abs(eps_prev)) * 100, 1)

        if rev_row:
            rev_now = q_inc.loc[rev_row].iloc[0]
            rev_prev = q_inc.loc[rev_row].iloc[4]
            if pd.notna(rev_now) and pd.notna(rev_prev) and rev_prev > 0:
                rev_yoy = round(((rev_now - rev_prev) / abs(rev_prev)) * 100, 1)

        return eps_yoy, rev_yoy
    except Exception:
        return None, None


enriched_rows = []
for _, row in candidates.iterrows():
    ticker = row["Ticker"]
    atr_pct, pattern, pivot, buy_top, stop_loss = analyze_price_and_atr(ticker)

    # Filter out low-volatility / buyout-pinned stocks (e.g., ATR_Pct < 1.2%)
    if atr_pct is None or atr_pct < MIN_ATR_PCT:
        print(f"Skipping {ticker}: ATR_Pct ({atr_pct}%) below minimum {MIN_ATR_PCT}% threshold.")
        continue

    print(f"Checking base structure & quarterly EPS for {ticker} (ATR%: {atr_pct}%)...")
    eps_yoy, rev_yoy = get_quarterly_eps_growth(ticker)
    pct_to_pivot = round(((row["Price"] - pivot) / pivot) * 100, 2) if pivot else None
    blue_dot = bool(row["Blue_Dot"]) if "Blue_Dot" in row else False
    macd_sig = row["MACD_Signal"] if "MACD_Signal" in row else "—"

    enriched_rows.append({
        "Ticker": ticker,
        "Company": row["Company"] if "Company" in row else ticker,
        "RS_Rating": row["RS_Rating"],
        "Blue_Dot": "🔵 YES" if blue_dot else "—",
        "MACD_Signal": macd_sig,
        "Acc_Dist": row["Acc_Dist"],
        "ATR_Pct": atr_pct,
        "Vol_vs_50D_Pct": row["Vol_vs_50D_Pct"] if "Vol_vs_50D_Pct" in row else None,
        "EPS_YoY_Pct": eps_yoy,
        "Sales_YoY_Pct": rev_yoy,
        "Pattern": pattern,
        "Price": row["Price"],
        "Pivot_Buy": pivot,
        "Pct_From_Pivot": pct_to_pivot,
        "Buy_Zone_Max": buy_top,
        "Stop_Loss_7.5Pct": stop_loss,
    })

df_actionable = pd.DataFrame(enriched_rows)
df_actionable.to_csv("marketsurge_actionable_setups.csv", index=False)
print(f"\nSaved {len(df_actionable)} actionable setups (filtered to ATR_Pct >= {MIN_ATR_PCT}%).")
