import os
import sys
import base64
import subprocess
from datetime import datetime
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from google import genai
from google.genai import types

st.set_page_config(page_title="AI MarketSurge & Best Stocks Terminal", layout="wide")
st.title("📈 AI MarketSurge & Best Stocks Terminal")

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/bandblue/marketsurge-alerts/main"
SYNC_FILES = [
    "marketsurge_rankings.csv",
    "marketsurge_actionable_setups.csv",
    "marketsurge_hh_hl_setups.csv",
    "josh_brown_best_stocks.csv",
]


def sync_from_github_if_newer():
    """Pulls the latest cloud-scanned CSVs from GitHub on startup so local matches cloud."""
    for fname in SYNC_FILES:
        try:
            r = requests.get(f"{GITHUB_RAW_BASE}/{fname}", timeout=8)
            if r.status_code == 200 and len(r.content) > 20:
                with open(fname, "wb") as f:
                    f.write(r.content)
        except Exception:
            pass


if "initial_cloud_pull_done" not in st.session_state:
    sync_from_github_if_newer()
    st.session_state["initial_cloud_pull_done"] = True


# --- MARKET DIRECTION ("M" IN CAN SLIM) ---
@st.cache_data(ttl=900)
def get_market_pulse():
    status_data = {}
    for idx in ["SPY", "QQQ"]:
        df = yf.Ticker(idx).history(period="6mo")
        close = float(df["Close"].iloc[-1])
        ema21 = float(df["Close"].ewm(span=21).mean().iloc[-1])
        sma50 = float(df["Close"].rolling(50).mean().iloc[-1])
        status_data[idx] = {
            "close": round(close, 2),
            "above_21": close > ema21,
            "above_50": close > sma50,
        }
    return status_data


pulse = get_market_pulse()
spy_ok = pulse["SPY"]["above_21"] and pulse["SPY"]["above_50"]
qqq_ok = pulse["QQQ"]["above_21"] and pulse["QQQ"]["above_50"]
above_50s = pulse["SPY"]["above_50"] and pulse["QQQ"]["above_50"]

if spy_ok and qqq_ok:
    st.success(
        f"**Market Pulse (M): CONFIRMED UPTREND** — "
        f"SPY (${pulse['SPY']['close']}) & QQQ (${pulse['QQQ']['close']}) are both trading above their 21-day EMA and 50-day SMA. Normal breakout buying is favored."
    )
elif above_50s:
    st.warning(
        f"**Market Pulse (M): UPTREND UNDER PRESSURE** — "
        f"SPY (${pulse['SPY']['close']}) & QQQ (${pulse['QQQ']['close']}) are holding above their 50-day SMA but testing short-term 21-day EMA support. Be selective with new buys."
    )
else:
    st.error(
        f"**Market Pulse (M): MARKET IN CORRECTION** — "
        f"SPY (${pulse['SPY']['close']}) or QQQ (${pulse['QQQ']['close']}) has broken below its 50-day SMA. Protect capital and wait for a Follow-Through Day."
    )


# --- SIDEBAR: ONE-CLICK DATA REFRESH, GITHUB SYNC & FILTERS ---
st.sidebar.header("🔄 Market Data Controls")

if os.path.exists("marketsurge_actionable_setups.csv"):
    last_mod = datetime.fromtimestamp(os.path.getmtime("marketsurge_actionable_setups.csv"))
    st.sidebar.caption(f"Last synced: {last_mod.strftime('%b %d, %Y at %I:%M %p')}")

if st.sidebar.button("⬇️ Pull Latest Watchlists from GitHub", use_container_width=True):
    sync_from_github_if_newer()
    st.rerun()

if st.sidebar.button("🔄 Refresh All Screeners Locally", type="primary", use_container_width=True):
    with st.sidebar.status("Running full 4-step market scan...", expanded=True) as status:
        steps = [
            ("1/4 RS & Technical Screener (`screener.py`)", "screener.py"),
            ("2/4 Base Pattern & EPS Scanner (`base_and_eps_scanner.py`)", "base_and_eps_scanner.py"),
            ("3/4 Higher High / Higher Low (`hh_hl_screener.py`)", "hh_hl_screener.py"),
            ("4/4 Josh Brown Best Stocks (`best_stocks_screener.py`)", "best_stocks_screener.py"),
        ]
        for label, script in steps:
            if not os.path.exists(script):
                continue
            st.write(f"Running {label}...")
            res = subprocess.run([sys.executable, script], capture_output=True, text=True)
            if res.returncode != 0:
                status.update(label=f"Error in {script}", state="error")
                st.sidebar.error(res.stderr)
                st.stop()

        status.update(label="All scans complete! Reloading dashboard...", state="complete")
        st.rerun()

if st.sidebar.button("☁️ Sync Watchlist to GitHub", use_container_width=True):
    gh_token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not gh_token:
        st.sidebar.error("GITHUB_TOKEN not found in environment variables.")
    elif not os.path.exists("marketsurge_actionable_setups.csv"):
        st.sidebar.error("No setups CSV found to sync.")
    else:
        with st.sidebar.status("Uploading setups to GitHub cloud...", expanded=True) as sync_status:
            api_url = "https://api.github.com/repos/bandblue/marketsurge-alerts/contents/marketsurge_actionable_setups.csv"
            headers = {
                "Authorization": f"Bearer {gh_token}",
                "Accept": "application/vnd.github+json",
            }
            get_resp = requests.get(api_url, headers=headers, timeout=15)
            sha = get_resp.json().get("sha") if get_resp.status_code == 200 else None

            with open("marketsurge_actionable_setups.csv", "rb") as f:
                encoded_csv = base64.b64encode(f.read()).decode("utf-8")

            payload = {
                "message": f"Update setups ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
                "content": encoded_csv,
                "branch": "main",
            }
            if sha:
                payload["sha"] = sha

            put_resp = requests.put(api_url, headers=headers, json=payload, timeout=15)
            if put_resp.status_code in (200, 201):
                sync_status.update(label="Synced to GitHub! Cloud alerts updated.", state="complete")
            else:
                sync_status.update(label="GitHub upload failed", state="error")
                st.sidebar.error(f"Error {put_resp.status_code}: {put_resp.text}")


# --- LOAD SAVED CSVs SAFELY ---
def load_csv_safe(fname: str) -> pd.DataFrame:
    if os.path.exists(fname) and os.path.getsize(fname) > 5:
        try:
            return pd.read_csv(fname)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


df_all = load_csv_safe("marketsurge_rankings.csv")
df_setups = load_csv_safe("marketsurge_actionable_setups.csv")
df_hh_hl = load_csv_safe("marketsurge_hh_hl_setups.csv")
df_best = load_csv_safe("josh_brown_best_stocks.csv")


# --- HELPER FOR JOSH BROWN / THE COMPOUND TEAR-SHEET CHART ---
@st.cache_data(ttl=600)
def fetch_compound_chart_data(symbol: str):
    hist = yf.Ticker(symbol).history(period="2y")
    iwb = yf.Ticker("IWB").history(period="2y")
    if hist.empty or len(hist) < 60:
        return None

    hist.index = hist.index.tz_localize(None).normalize()
    iwb.index = iwb.index.tz_localize(None).normalize()

    hist["SMA_50"] = hist["Close"].rolling(50).mean()
    hist["SMA_200"] = hist["Close"].rolling(200).mean()
    hist["High_52W_Step"] = hist["High"].rolling(252, min_periods=20).max()

    # Relative to Russell 1000 (IWB)
    iwb_aligned = iwb["Close"].reindex(hist.index).ffill()
    hist["Rel_IWB"] = (hist["Close"] / iwb_aligned) * 100.0

    # Wilder RSI (14)
    delta = hist["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    hist["RSI_14"] = 100.0 - (100.0 / (1.0 + rs))

    # Aroon Oscillator (25)
    period = 25
    aroon_up = hist["High"].rolling(period + 1).apply(lambda x: (np.argmax(x) / period) * 100.0, raw=True)
    aroon_dn = hist["Low"].rolling(period + 1).apply(lambda x: (np.argmin(x) / period) * 100.0, raw=True)
    hist["Aroon_Osc"] = aroon_up - aroon_dn

    return hist


def render_compound_tear_sheet(symbol: str, best_df: pd.DataFrame):
    hist = fetch_compound_chart_data(symbol)
    if hist is None:
        st.warning(f"No historical price data found for `{symbol}`.")
        return

    plot_df = hist.tail(252).copy()  # 1-year window matching The Compound chart
    last_row = plot_df.iloc[-1]
    last_date_str = plot_df.index[-1].strftime("%m/%d/%y")

    last_close = float(last_row["Close"])
    sma50 = float(last_row["SMA_50"]) if pd.notna(last_row["SMA_50"]) else last_close
    sma200 = float(last_row["SMA_200"]) if pd.notna(last_row["SMA_200"]) else last_close
    high_52w = float(plot_df["High"].max())
    rsi_val = float(last_row["RSI_14"]) if pd.notna(last_row["RSI_14"]) else 50.0
    aroon_val = int(round(float(last_row["Aroon_Osc"]))) if pd.notna(last_row["Aroon_Osc"]) else 0

    pct_below_high = ((last_close - high_52w) / high_52w) * 100.0
    vs_50d = ((last_close - sma50) / sma50) * 100.0
    vs_200d = ((last_close - sma200) / sma200) * 100.0

    # Returns: 1M (21d), 3M (63d), 6M (126d), YTD
    ret_1m = ((last_close / float(hist["Close"].iloc[-22])) - 1.0) * 100.0 if len(hist) > 22 else 0.0
    ret_3m = ((last_close / float(hist["Close"].iloc[-64])) - 1.0) * 100.0 if len(hist) > 64 else 0.0
    ret_6m = ((last_close / float(hist["Close"].iloc[-127])) - 1.0) * 100.0 if len(hist) > 127 else 0.0

    ytd_df = plot_df[plot_df.index.year == plot_df.index[-1].year]
    ret_ytd = ((last_close / float(ytd_df["Close"].iloc[0])) - 1.0) * 100.0 if not ytd_df.empty else 0.0

    # 1Y Max Drawdown
    rolling_peak = plot_df["High"].cummax()
    drawdowns = ((plot_df["Low"] - rolling_peak) / rolling_peak) * 100.0
    max_dd_1y = int(round(float(drawdowns.min())))

    # Pull metadata from josh_brown_best_stocks.csv if present
    row_match = best_df[best_df["Ticker"] == symbol] if (not best_df.empty and "Ticker" in best_df.columns) else pd.DataFrame()
    if not row_match.empty:
        meta = row_match.iloc[0]
        company_name = meta.get("Company", symbol)
        sector_name = meta.get("Sector", "—")
        chart_score = meta.get("Best_Stock_Score", "—")
        mkt_cap_str = f"${meta['Mkt_Cap_B']:.1f}B" if pd.notna(meta.get("Mkt_Cap_B")) else "—"
        rev_yoy = meta.get("Rev_YoY_Pct")
        eps_yoy = meta.get("EPS_YoY_Pct")
        rev_eps_str = (
            f"{rev_yoy:+.1f}% / {eps_yoy:+.1f}%"
            if pd.notna(rev_yoy) and pd.notna(eps_yoy)
            else (f"{rev_yoy:+.1f}% / —" if pd.notna(rev_yoy) else "—")
        )
        rsi_status = meta.get("RSI_Status", "")
    else:
        company_name = symbol
        sector_name = "Custom Lookup"
        chart_score = "—"
        mkt_cap_str = "—"
        rev_eps_str = "—"
        rsi_status = "POST_EARNINGS_RSI (>72)" if rsi_val > 72 else ""

    # Detect most recent bullish crossover above 50D SMA (for green "Added to list" dot)
    above_50_series = plot_df["Close"] > plot_df["SMA_50"]
    cross_up_dates = plot_df.index[above_50_series & (~above_50_series.shift(1).fillna(True))]
    added_idx = cross_up_dates[-1] if len(cross_up_dates) > 0 else None

    # --- BUILD 2-COLUMN TEAR SHEET ---
    c_left, c_right = st.columns([2.15, 1.0], gap="large")

    with c_left:
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.045,
            row_heights=[0.62, 0.19, 0.19],
            subplot_titles=("", "Relative to Russell 1000 (IWB)", "RSI (14)"),
        )

        # Pane 1: OHLC Bars (Blue up / Orange down matching The Compound)
        fig.add_trace(
            go.Ohlc(
                x=plot_df.index,
                open=plot_df["Open"],
                high=plot_df["High"],
                low=plot_df["Low"],
                close=plot_df["Close"],
                increasing_line_color="#2563EB",
                decreasing_line_color="#EA580C",
                line_width=1.4,
                name=symbol,
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # 50-Day SMA (Amber/Gold)
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df["SMA_50"],
                line=dict(color="#F59E0B", width=2.2),
                name="50-day",
            ),
            row=1,
            col=1,
        )

        # 200-Day SMA (Dark Navy)
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df["SMA_200"],
                line=dict(color="#0F172A", width=2.2),
                name="200-day",
            ),
            row=1,
            col=1,
        )

        # Stepped 52-Week High Dashed Line
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df["High_52W_Step"],
                line=dict(color="#94A3B8", width=1.2, dash="dash"),
                name="52-week high",
            ),
            row=1,
            col=1,
        )

        # Green Dot Marker ("Trend Reclaim / Added")
        if added_idx is not None:
            fig.add_trace(
                go.Scatter(
                    x=[added_idx],
                    y=[float(plot_df.loc[added_idx, "Close"])],
                    mode="markers",
                    marker=dict(color="#16A34A", size=9, line=dict(color="white", width=1.5)),
                    name="50D Reclaim",
                ),
                row=1,
                col=1,
            )

        # Right-axis price badges ($Close, $50D, $200D)
        last_x = plot_df.index[-1]
        badges = [
            (last_close, "#EA580C", f"${last_close:,.0f}"),
            (sma50, "#F59E0B", f"${sma50:,.0f}"),
            (sma200, "#0F172A", f"${sma200:,.0f}"),
        ]
        for y_val, bg_color, label_txt in badges:
            fig.add_annotation(
                x=last_x,
                y=y_val,
                text=f"<b>{label_txt}</b>",
                showarrow=False,
                xanchor="left",
                xshift=6,
                font=dict(color="white", size=11),
                bgcolor=bg_color,
                borderpad=3,
                row=1,
                col=1,
            )

        # Pane 2: Relative to Russell 1000 (IWB)
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df["Rel_IWB"],
                line=dict(color="#2563EB", width=1.8),
                name="Rel to IWB",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # Pane 3: RSI (14) with 70/30 lines
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df["RSI_14"],
                line=dict(color="#1D4ED8", width=1.7),
                name="RSI (14)",
                showlegend=False,
            ),
            row=3,
            col=1,
        )
        fig.add_hline(y=70, line_dash="dot", line_color="#CBD5E1", line_width=1, row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#CBD5E1", line_width=1, row=3, col=1)
        fig.add_annotation(
            x=last_x,
            y=rsi_val,
            text=f"<b>{int(round(rsi_val))}</b>",
            showarrow=False,
            xanchor="left",
            xshift=6,
            font=dict(color="#0F172A", size=11),
            row=3,
            col=1,
        )

        fig.update_layout(
            height=640,
            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=15, r=55, t=40, b=20),
            font=dict(color="#1E293B", size=12),
        )
        fig.update_yaxes(side="right", tickprefix="$", gridcolor="#F1F5F9", row=1, col=1)
        fig.update_yaxes(side="right", showticklabels=False, gridcolor="#F1F5F9", row=2, col=1)
        fig.update_yaxes(side="right", range=[15, 90], tickvals=[30, 70], gridcolor="#F1F5F9", row=3, col=1)
        fig.update_xaxes(showgrid=False, tickformat="%b '%y")

        st.plotly_chart(fig, use_container_width=True)

    with c_right:
        with st.container(border=True):
            st.markdown(f"### **{symbol}** &nbsp; {company_name}, *{sector_name}*")

            # Dynamic Status Pills (Matching The Compound Header Badges)
            pills_html = []
            if abs(pct_below_high) <= 5.0:
                pills_html.append(
                    "<span style='background:#0F172A;color:#fff;padding:4px 10px;border-radius:14px;font-size:12px;font-weight:600;margin-right:6px;'>Within 5% of high</span>"
                )
            elif abs(pct_below_high) <= 10.0:
                pills_html.append(
                    "<span style='background:#1E3A8A;color:#fff;padding:4px 10px;border-radius:14px;font-size:12px;font-weight:600;margin-right:6px;'>Within 10% of high</span>"
                )
            if 0.0 <= vs_50d <= 4.5:
                pills_html.append(
                    "<span style='background:#F59E0B;color:#0F172A;padding:4px 10px;border-radius:14px;font-size:12px;font-weight:700;margin-right:6px;'>At 50-day</span>"
                )
            if last_close > sma50 > sma200:
                pills_html.append(
                    "<span style='background:#334155;color:#fff;padding:4px 10px;border-radius:14px;font-size:12px;font-weight:600;margin-right:6px;'>Higher highs & lows</span>"
                )
            if "POST_EARNINGS" in str(rsi_status):
                pills_html.append(
                    "<span style='background:#16A34A;color:#fff;padding:4px 10px;border-radius:14px;font-size:12px;font-weight:600;'>Post-earnings RSI</span>"
                )
            if pills_html:
                st.markdown("".join(pills_html), unsafe_allow_html=True)
                st.write("")

            # Clean Data | Value Table
            summary_rows = [
                ("Best Stock Score", f"{chart_score}"),
                (f"Price ({last_date_str} close)", f"${last_close:,.2f}"),
                ("% below 52-week high", f"{pct_below_high:+.1f}%"),
                ("vs 50-day", f"{vs_50d:+.1f}%"),
                ("vs 200-day", f"{vs_200d:+.1f}%"),
                ("RSI (14)", f"{int(round(rsi_val))}"),
                ("Aroon oscillator", f"{aroon_val}"),
                ("1M / 3M / 6M / YTD", f"{ret_1m:+.0f}% / {ret_3m:+.0f}% / {ret_6m:+.0f}% / {ret_ytd:+.0f}%"),
                ("1Y max drawdown", f"{max_dd_1y}%"),
                ("Market cap", mkt_cap_str),
                ("Rev / EPS YoY", rev_eps_str),
                ("50-day SMA stop", f"${sma50:,.2f}"),
            ]
            df_metrics = pd.DataFrame(summary_rows, columns=["Data", "Value"])
            st.dataframe(df_metrics, use_container_width=True, hide_index=True, height=455)

            # Bottom Summary Callout Box (Matching Screenshot Narrative)
            st.info(
                f"**{symbol}:** ${last_close:,.0f}, with the 50-day at ${sma50:,.0f} and the 200-day at${sma200:,.0f}. "
                f"RSI at {int(round(rsi_val))}. The stock is {abs(pct_below_high):.0f}% below its 52-week high of ${high_52w:,.0f}."
            )


# ==============================================================================
# MAIN 3-TAB TERMINAL
# ==============================================================================
tab_best, tab_canslim, tab_hh_hl = st.tabs([
    "🏆 Josh Brown Best Stocks",
    "🎯 CAN SLIM Base Setups",
    "📈 Higher Highs & Higher Lows",
])

# ------------------------------------------------------------------------------
# TAB 1: JOSH BROWN BEST STOCKS IN THE MARKET (THE COMPOUND TEAR SHEET)
# ------------------------------------------------------------------------------
with tab_best:
    st.subheader("🏆 Best Stocks in the Market — Institutional Trend & Tear Sheet")

    c_sel1, c_sel2, c_sel3 = st.columns([2, 1.2, 1.2])
    best_tickers = df_best["Ticker"].tolist() if (not df_best.empty and "Ticker" in df_best.columns) else ["SN", "TGT", "DGX", "ABBV", "BDX"]
    best_name_map = (
        dict(zip(df_best["Ticker"], df_best["Company"]))
        if (not df_best.empty and "Ticker" in df_best.columns and "Company" in df_best.columns)
        else {}
    )

    chosen_best = c_sel1.selectbox(
        "Select Screened Best Stock:",
        best_tickers,
        format_func=lambda t: f"{t} — {best_name_map.get(t, t)}",
        key="best_stock_selector",
    )
    custom_ticker = c_sel2.text_input("Or inspect any ticker
