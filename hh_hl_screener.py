import io
import os
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy.signal import argrelextrema

MIN_PRICE = 15.0
MIN_AVG_VOL_50D = 400_000
MIN_ATR_PCT = 1.2
SWING_ORDER = 5  # 5 trading days on each side to confirm a swing pivot
BATCH_SIZE = 250


def get_broad_universe() -> pd.DataFrame:
    """
    Fetches S&P 500, S&P MidCap 400, and S&P SmallCap 600 constituents (~1,500 US equities).
    Falls back to marketsurge_rankings.csv if Wikipedia is unreachable.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    sources = [
        ("S&P 500", "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol", "Security"),
        ("S&P 400", "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", "Symbol", "Security"),
        ("S&P 600", "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", "Symbol", "Security"),
    ]

    frames = []
    for index_name, url, sym_col, name_col in sources:
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            tables = pd.read_html(io.StringIO(resp.text))
            df_idx = tables[0][[sym_col, name_col]].copy()
            df_idx.columns = ["Ticker", "Company"]
            df_idx["Index"] = index_name
            df_idx["Ticker"] = df_idx["Ticker"].astype(str).str.strip().str.replace(".", "-", regex=False)
            frames.append(df_idx)
        except Exception as e:
            print(f"Warning: Could not fetch {index_name}: {e}")

    if frames:
        universe = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["Ticker"])
        return universe

    if os.path.exists("marketsurge_rankings.csv"):
        df_ms = pd.read_csv("marketsurge_rankings.csv")[["Ticker", "Company"]].copy()
        df_ms["Index"] = "MarketSurge Fallback"
        return df_ms

    raise RuntimeError("Unable to load any stock universe.")


def evaluate_df(df: pd.DataFrame) -> dict | None:
    df = df.dropna(subset=["Close", "High", "Low", "Volume"])
    if len(df) < 210:
        return None

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    last_close = float(close.iloc[-1])
    avg_vol_50d = float(volume.tail(50).mean())
    if last_close < MIN_PRICE or avg_vol_50d < MIN_AVG_VOL_50D:
        return None

    # 1. Moving Average Stack: Close > 21 EMA > 50 SMA > 200 SMA
    ema21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    sma200 = float(close.rolling(200).mean().iloc[-1])
    if not (last_close > ema21 > sma50 > sma200):
        return None

    # 2. 14-Day Wilder ATR%
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr14 = float(tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
    atr_pct = round((atr14 / last_close) * 100, 2)
    if atr_pct < MIN_ATR_PCT:
        return None

    # 3. Momentum & Accumulation: RSI(14) 50-70, MACD > Signal, Rising 20d OBV
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs_val = gain / loss.replace(0, np.nan)
    rsi14 = float((100 - (100 / (1 + rs_val))).iloc[-1])
    if not (50.0 <= rsi14 <= 70.0):
        return None

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    if float(macd.iloc[-1]) <= float(macd_signal.iloc[-1]):
        return None

    obv = (np.sign(close.diff()).fillna(0) * volume).cumsum()
    if float(obv.iloc[-1]) <= float(obv.iloc[-20]):
        return None

    # 4. Confirmed Fractal Swing Highs & Swing Lows (Higher Highs + Higher Lows)
    highs_arr = high.values
    lows_arr = low.values
    sh_idx = argrelextrema(highs_arr, np.greater_equal, order=SWING_ORDER)[0]
    sl_idx = argrelextrema(lows_arr, np.less_equal, order=SWING_ORDER)[0]

    if len(sh_idx) < 2 or len(sl_idx) < 2:
        return None

    sh1, sh2 = float(highs_arr[sh_idx[-2]]), float(highs_arr[sh_idx[-1]])
    sl1, sl2 = float(lows_arr[sl_idx[-2]]), float(lows_arr[sl_idx[-1]])

    if not (sh2 > sh1 and sl2 > sl1 and last_close > sl2):
        return None

    # 5. Classify Entry Zone
    pct_from_ema21 = round(((last_close - ema21) / ema21) * 100, 2)
    pct_from_sh2 = round(((last_close - sh2) / sh2) * 100, 2)

    if -1.0 <= pct_from_ema21 <= 2.5:
        setup_type = "PULLBACK_ENTRY (21 EMA)"
    elif -3.0 <= pct_from_sh2 <= 0.5:
        setup_type = "BREAKOUT_READY (Swing High)"
    else:
        setup_type = "UPTREND_EXTENDED"

    risk_to_sl2_pct = round(((last_close - sl2) / last_close) * 100, 2)

    return {
        "Setup_Type": setup_type,
        "ATR_Pct": atr_pct,
        "RSI_14": round(rsi14, 1),
        "Avg_Vol_50D_K": round(avg_vol_50d / 1000, 0),
        "Price": round(last_close, 2),
        "EMA_21": round(ema21, 2),
        "Pct_From_EMA21": pct_from_ema21,
        "Prev_Swing_High": round(sh1, 2),
        "Last_Swing_High": round(sh2, 2),
        "Pct_From_SH2": pct_from_sh2,
        "Prev_Swing_Low": round(sl1, 2),
        "Last_Swing_Low_Stop": round(sl2, 2),
        "Risk_To_Stop_Pct": risk_to_sl2_pct,
    }


def main():
    universe = get_broad_universe()
    tickers = universe["Ticker"].tolist()
    meta_map = universe.set_index("Ticker").to_dict("index")

    ms_map = {}
    if os.path.exists("marketsurge_rankings.csv"):
        df_ms = pd.read_csv("marketsurge_rankings.csv")
        if "Ticker" in df_ms.columns:
            ms_map = df_ms.set_index("Ticker").to_dict("index")

    print(f"Scanning {len(tickers)} broad-market stocks (S&P 500 + 400 + 600) for Higher Highs & Higher Lows...\n")

    results = []
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i : i + BATCH_SIZE]
        data = yf.download(
            batch,
            period="1y",
            interval="1d",
            group_by="ticker",
            progress=False,
            threads=True,
        )
        if data.empty:
            continue

        for ticker in batch:
            try:
                t_df = data[ticker] if len(batch) > 1 else data
            except KeyError:
                continue

            metrics = evaluate_df(t_df)
            if metrics:
                ms_info = ms_map.get(ticker, {})
                u_info = meta_map.get(ticker, {})
                results.append({
                    "Ticker": ticker,
                    "Company": u_info.get("Company", ticker),
                    "Index": u_info.get("Index", "—"),
                    "RS_Rating": ms_info.get("RS_Rating", "—"),
                    "Acc_Dist": ms_info.get("Acc_Dist", "—"),
                    **metrics,
                })

    df_out = pd.DataFrame(results)
    if not df_out.empty:
        priority = {"PULLBACK_ENTRY (21 EMA)": 0, "BREAKOUT_READY (Swing High)": 1, "UPTREND_EXTENDED": 2}
        df_out["_sort"] = df_out["Setup_Type"].map(priority).fillna(2)
        df_out = df_out.sort_values(["_sort", "RSI_14"], ascending=[True, False]).drop(columns=["_sort"])

    df_out.to_csv("marketsurge_hh_hl_setups.csv", index=False)
    print(f"Saved {len(df_out)} Higher High / Higher Low setups across the broad market to marketsurge_hh_hl_setups.csv.")


if __name__ == "__main__":
    main()
