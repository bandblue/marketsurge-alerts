import io
import os
import numpy as np
import pandas as pd
import requests
import yfinance as yf

MIN_MARKET_CAP = 10_000_000_000  # $10B+ institutional liquidity
MIN_AVG_VOL_50D = 1_000_000      # 1M+ shares/day
MIN_ATR_PCT = 1.2                # Exclude buyout-pinned stocks
MAX_PCT_OFF_52W_HIGH = 10.0      # Within 10% of 52-week high
BATCH_SIZE = 200


def get_institutional_universe() -> pd.DataFrame:
    """Fetches S&P 500 and S&P MidCap 400 constituents (~900 stocks)."""
    headers = {"User-Agent": "Mozilla/5.0"}
    sources = [
        ("S&P 500", "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol", "Security", "GICS Sector"),
        ("S&P 400", "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", "Symbol", "Security", "GICS Sector"),
    ]
    frames = []
    for index_name, url, sym_col, name_col, sec_col in sources:
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            tables = pd.read_html(io.StringIO(resp.text))
            cols = [c for c in [sym_col, name_col, sec_col] if c in tables[0].columns]
            df_idx = tables[0][cols].copy()
            rename_map = {sym_col: "Ticker", name_col: "Company"}
            if sec_col in df_idx.columns:
                rename_map[sec_col] = "Sector"
            df_idx = df_idx.rename(columns=rename_map)
            if "Sector" not in df_idx.columns:
                df_idx["Sector"] = "—"
            df_idx["Index"] = index_name
            df_idx["Ticker"] = df_idx["Ticker"].astype(str).str.strip().str.replace(".", "-", regex=False)
            frames.append(df_idx)
        except Exception as e:
            print(f"Warning: Could not fetch {index_name}: {e}")

    if frames:
        return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["Ticker"])
    raise RuntimeError("Unable to load S&P 500 / 400 universe.")


def evaluate_technical(df: pd.DataFrame, spy_6m_ret: float, spy_12m_ret: float) -> dict | None:
    df = df.dropna(subset=["Close", "High", "Low", "Volume"])
    if len(df) < 252:
        return None

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    last_close = float(close.iloc[-1])
    avg_vol_50d = float(volume.tail(50).mean())
    if avg_vol_50d < MIN_AVG_VOL_50D:
        return None

    # 1. Moving Averages & Slopes (Price > 50 SMA > 200 SMA, both MAs rising)
    sma50_series = close.rolling(50).mean()
    sma200_series = close.rolling(200).mean()
    sma50 = float(sma50_series.iloc[-1])
    sma200 = float(sma200_series.iloc[-1])
    sma50_prev = float(sma50_series.iloc[-10])
    sma200_prev = float(sma200_series.iloc[-20])

    if not (last_close > sma50 > sma200 and sma50 > sma50_prev and sma200 > sma200_prev):
        return None

    # 2. 52-Week High Proximity (Within 10%)
    high_52w = float(high.tail(252).max())
    pct_off_high = ((high_52w - last_close) / high_52w) * 100.0
    if pct_off_high > MAX_PCT_OFF_52W_HIGH:
        return None

    # 3. 14-Day Wilder ATR%
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr14 = float(tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
    atr_pct = round((atr14 / last_close) * 100.0, 2)
    if atr_pct < MIN_ATR_PCT:
        return None

    # 4. Constructive RSI (52 - 72)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs_val = gain / loss.replace(0, np.nan)
    rsi14 = float((100.0 - (100.0 / (1.0 + rs_val))).iloc[-1])
    if not (52.0 <= rsi14 <= 72.0):
        return None

    # 5. Relative Outperformance vs SPY (6-Month & 12-Month)
    ret_3m = ((last_close / float(close.iloc[-63])) - 1.0) * 100.0
    ret_6m = ((last_close / float(close.iloc[-126])) - 1.0) * 100.0
    ret_12m = ((last_close / float(close.iloc[-252])) - 1.0) * 100.0
    if ret_6m <= spy_6m_ret or ret_12m <= spy_12m_ret:
        return None

    pct_above_50d = round(((last_close - sma50) / sma50) * 100.0, 2)
    if pct_off_high <= 3.0:
        entry_zone = "BREAKOUT_ZONE (Near 52W High)"
    elif pct_above_50d <= 4.0:
        entry_zone = "PULLBACK_SUPPORT (Near 50D SMA)"
    else:
        entry_zone = "TRENDING_UP"

    return {
        "Entry_Zone": entry_zone,
        "Price": round(last_close, 2),
        "52W_High": round(high_52w, 2),
        "Pct_Off_52W_High": round(-pct_off_high, 2),
        "SMA_50_Stop": round(sma50, 2),
        "Pct_Above_50D": pct_above_50d,
        "SMA_200": round(sma200, 2),
        "RSI_14": round(rsi14, 1),
        "ATR_Pct": atr_pct,
        "Ret_3M_Pct": round(ret_3m, 1),
        "Ret_6M_Pct": round(ret_6m, 1),
        "Ret_12M_Pct": round(ret_12m, 1),
    }


def check_fundamentals(symbol: str) -> dict | None:
    """Verifies $10B+ market cap and positive YoY revenue & earnings growth."""
    try:
        info = yf.Ticker(symbol).info
        mkt_cap = info.get("marketCap") or 0
        if mkt_cap < MIN_MARKET_CAP:
            return None

        rev_growth = info.get("revenueGrowth")
        eps_growth = info.get("earningsQuarterlyGrowth")
        rev_pct = round(rev_growth * 100.0, 1) if rev_growth is not None else None
        eps_pct = round(eps_growth * 100.0, 1) if eps_growth is not None else None

        # Require positive top-line and bottom-line momentum (> 5% YoY)
        if rev_pct is None or eps_pct is None or rev_pct < 5.0 or eps_pct < 5.0:
            return None

        return {
            "Mkt_Cap_B": round(mkt_cap / 1e9, 1),
            "Rev_YoY_Pct": rev_pct,
            "EPS_YoY_Pct": eps_pct,
        }
    except Exception:
        return None


def main():
    universe = get_institutional_universe()
    tickers = universe["Ticker"].tolist()
    meta_map = universe.set_index("Ticker").to_dict("index")

    spy_df = yf.Ticker("SPY").history(period="15mo")
    spy_6m = ((float(spy_df["Close"].iloc[-1]) / float(spy_df["Close"].iloc[-126])) - 1.0) * 100.0
    spy_12m = ((float(spy_df["Close"].iloc[-1]) / float(spy_df["Close"].iloc[-252])) - 1.0) * 100.0
    print(f"Benchmark SPY returns — 6M: {spy_6m:+.1f}%, 12M: {spy_12m:+.1f}%")
    print(f"Scanning {len(tickers)} institutional stocks (S&P 500 + 400)...\n")

    tech_candidates = []
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i : i + BATCH_SIZE]
        data = yf.download(batch, period="15mo", interval="1d", group_by="ticker", progress=False, threads=True)
        if data.empty:
            continue

        for ticker in batch:
            try:
                t_df = data[ticker] if len(batch) > 1 else data
            except KeyError:
                continue
            tech = evaluate_technical(t_df, spy_6m, spy_12m)
            if tech:
                tech_candidates.append((ticker, tech))

    print(f"{len(tech_candidates)} stocks passed technical hurdles; checking fundamental growth...\n")

    rows = []
    for ticker, tech in tech_candidates:
        fund = check_fundamentals(ticker)
        if not fund:
            continue

        u_info = meta_map.get(ticker, {})
        # Composite score emphasizing 6M/12M alpha + fundamental growth
        score = round(
            min(tech["Ret_6M_Pct"], 80) * 0.4
            + min(tech["Ret_12M_Pct"], 120) * 0.3
            + min(fund["Rev_YoY_Pct"], 50) * 0.4
            + min(fund["EPS_YoY_Pct"], 60) * 0.3,
            1,
        )
        rows.append({
            "Ticker": ticker,
            "Company": u_info.get("Company", ticker),
            "Sector": u_info.get("Sector", "—"),
            "Best_Stock_Score": score,
            **fund,
            **tech,
        })

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out = df_out.sort_values("Best_Stock_Score", ascending=False)

    df_out.to_csv("josh_brown_best_stocks.csv", index=False)
    print(f"Saved {len(df_out)} 'Best Stocks in the Market' candidates to josh_brown_best_stocks.csv.")


if __name__ == "__main__":
    main()
