import io
import re
import requests
import pandas as pd
import yfinance as yf

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def fetch_index_tickers():
    """Fetches current S&P 500 and Nasdaq-100 constituents + company names."""
    name_map = {}

    # 1. S&P 500 Constituents + Security Name
    try:
        sp_html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=HEADERS, timeout=15).text
        sp_tables = pd.read_html(io.StringIO(sp_html))
        sp_df = sp_tables[0]
        for _, r in sp_df.iterrows():
            sym = str(r["Symbol"]).strip().upper().replace(".", "-")
            if sym and sym != "NAN":
                name_map[sym] = str(r["Security"]).strip()
        print(f"Loaded {len(sp_df)} S&P 500 constituents.")
    except Exception as e:
        print(f"Warning: Could not fetch S&P 500 list ({e})")

    # 2. Nasdaq-100 Constituents + Company Name
    try:
        ndx_html = requests.get("https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies", headers=HEADERS, timeout=15).text
        ndx_tables = pd.read_html(io.StringIO(ndx_html))
        for tbl in ndx_tables:
            col = next((c for c in ["Ticker", "Symbol"] if c in tbl.columns), None)
            name_col = next((c for c in ["Company", "Security", "Name"] if c in tbl.columns), None)
            if col and len(tbl) >= 95:
                for _, r in tbl.iterrows():
                    sym = str(r[col]).strip().upper().replace(".", "-")
                    if sym and sym != "NAN" and sym not in name_map:
                        name_map[sym] = str(r[name_col]).strip() if name_col else sym
                print(f"Loaded {len(tbl)} Nasdaq-100 constituents.")
                break
    except Exception as e:
        print(f"Warning: Could not fetch Nasdaq-100 list ({e})")

    # 3. Merge any extra custom symbols from watchlist.txt
    try:
        with open("watchlist.txt", "r") as f:
            custom = [t.strip().upper().replace(".", "-") for t in re.split(r"[,\s]+", f.read()) if t.strip()]
        for sym in custom:
            if sym and sym not in name_map:
                try:
                    info = yf.Ticker(sym).info
                    name_map[sym] = info.get("shortName") or info.get("longName") or sym
                except Exception:
                    name_map[sym] = sym
    except FileNotFoundError:
        pass

    return name_map

print("Building S&P 500 + Nasdaq-100 + Custom Watchlist universe...")
company_names = fetch_index_tickers()
universe = list(company_names.keys())
print(f"Total unique tickers to scan: {len(universe)}\n")

if not universe:
    raise RuntimeError("No tickers loaded! Check internet connection or watchlist.txt.")

print("Fetching S&P 500 (SPY) benchmark...")
spy_df = yf.Ticker("SPY").history(period="1y")
spy_close = spy_df["Close"]
spy_close.index = spy_close.index.tz_localize(None).normalize()

def analyze_dataframe(symbol, df):
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    required_cols = {"Close", "High", "Volume"}
    if not required_cols.issubset(set(df.columns)):
        return None

    df = df.dropna(subset=["Close", "High", "Volume"]).copy()
    if len(df) < 130:
        return None

    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    close = df["Close"]
    volume = df["Volume"]
    latest_close = float(close.iloc[-1])

    # 1. IBD Weighted Performance (40% Q1, 20% Q2, 20% Q3, 20% Q4)
    p_3m = float(close.iloc[-min(63, len(df))])
    p_6m = float(close.iloc[-min(126, len(df))])
    p_9m = float(close.iloc[-min(189, len(df))])
    p_12m = float(close.iloc[-min(252, len(df))])

    q1 = (latest_close - p_3m) / p_3m
    q2 = (p_3m - p_6m) / p_6m
    q3 = (p_6m - p_9m) / p_9m
    q4 = (p_9m - p_12m) / p_12m
    raw_rs = (0.4 * q1 + 0.2 * q2 + 0.2 * q3 + 0.2 * q4) * 100

    # 2. MarketSurge "Blue Dot": RS Line (Stock / SPY) at 52W High
    aligned_spy = spy_close.reindex(close.index).ffill()
    rs_line = (close / aligned_spy).dropna()
    if rs_line.empty:
        return None
    rs_line_latest = float(rs_line.iloc[-1])
    rs_line_52w_high = float(rs_line.max())
    rs_line_pct_off = ((rs_line_latest - rs_line_52w_high) / rs_line_52w_high) * 100
    is_blue_dot = rs_line_pct_off >= -0.5

    # 3. Moving Averages & 52-Week High Proximity
    sma_50 = float(close.rolling(50).mean().iloc[-1])
    sma_200 = float(close.rolling(200).mean().iloc[-1]) if len(df) >= 200 else float(close.mean())
    high_52w = float(df["High"].max())
    pct_off_high = ((latest_close - high_52w) / high_52w) * 100

    in_uptrend = (latest_close > sma_50) and (sma_50 > sma_200) and (pct_off_high >= -15.0)

    # 4. MACD (12, 26, 9) Momentum Status
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line

    h_now = float(macd_hist.iloc[-1])
    h_prev = float(macd_hist.iloc[-2])
    h_3d_ago = float(macd_hist.iloc[-4])

    if h_now > 0 and h_3d_ago <= 0:
        macd_status = "🟢 Bullish Cross"
    elif h_now > 0:
        macd_status = "🟢 Bullish"
    elif h_now > h_prev:
        macd_status = "🟡 Turning Up"
    else:
        macd_status = "🔴 Bearish"

    # 5. 13-Week (65-day) Institutional Accumulation / Distribution Ratio
    recent = df.tail(65).copy()
    recent["change"] = recent["Close"].diff()
    up_vol = recent.loc[recent["change"] > 0, "Volume"].sum()
    down_vol = recent.loc[recent["change"] < 0, "Volume"].sum()
    ud_ratio = round(up_vol / down_vol, 2) if down_vol > 0 else 1.0

    if ud_ratio >= 1.35:
        acc_dist = "A"
    elif ud_ratio >= 1.15:
        acc_dist = "B"
    elif ud_ratio >= 0.90:
        acc_dist = "C"
    elif ud_ratio >= 0.75:
        acc_dist = "D"
    else:
        acc_dist = "E"

    # 6. Recent Volume Surge vs 50-day Average Volume
    avg_vol_50 = float(volume.tail(50).mean())
    vol_pct_change = ((float(volume.iloc[-1]) - avg_vol_50) / avg_vol_50) * 100 if avg_vol_50 > 0 else 0

    return {
        "Ticker": symbol,
        "Company": company_names.get(symbol, symbol),
        "Blue_Dot": is_blue_dot,
        "MACD_Signal": macd_status,
        "Price": round(latest_close, 2),
        "Raw_RS": round(raw_rs, 2),
        "Acc_Dist": acc_dist,
        "Up_Down_Vol": ud_ratio,
        "Pct_Off_52W_High": round(pct_off_high, 2),
        "52W_High_Pivot": round(high_52w, 2),
        "Stage2_Trend": in_uptrend,
        "Vol_vs_50D_Pct": round(vol_pct_change, 1),
    }

results = []
batch_size = 100
for start in range(0, len(universe), batch_size):
    batch = universe[start : start + batch_size]
    print(f"Downloading batch {start + 1}–{min(start + batch_size, len(universe))} of {len(universe)}...")
    bulk = yf.download(batch, period="1y", group_by="ticker", threads=True, progress=False)
    for sym in batch:
        try:
            if isinstance(bulk.columns, pd.MultiIndex):
                if sym in bulk.columns.get_level_values(0):
                    sub_df = bulk[sym]
                elif sym in bulk.columns.get_level_values(1):
                    sub_df = bulk.xs(sym, axis=1, level=1)
                else:
                    continue
            else:
                sub_df = bulk
            data = analyze_dataframe(sym, sub_df)
            if data:
                results.append(data)
        except Exception:
            continue

if not results:
    raise RuntimeError("No valid stock data returned from Yahoo Finance.")

df_results = pd.DataFrame(results)
df_results["RS_Rating"] = (df_results["Raw_RS"].rank(pct=True) * 98 + 1).round().astype(int)

cols = [
    "Ticker", "Company", "RS_Rating", "Blue_Dot", "MACD_Signal", "Acc_Dist", "Price", "52W_High_Pivot",
    "Pct_Off_52W_High", "Stage2_Trend", "Up_Down_Vol", "Vol_vs_50D_Pct", "Raw_RS"
]
df_results = df_results[cols].sort_values(by=["RS_Rating", "Up_Down_Vol"], ascending=False)
df_results.to_csv("marketsurge_rankings.csv", index=False)

print(f"\nScan complete! Ranked {len(df_results)} stocks and saved to 'marketsurge_rankings.csv'.")