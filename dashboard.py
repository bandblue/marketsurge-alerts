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

st.set_page_config(page_title="AI MarketSurge Terminal", layout="wide")
app_pw = os.environ.get("APP_PASSWORD", "")
if app_pw and st.sidebar.text_input("Dashboard Password", type="password") != app_pw:
    st.info("Enter your dashboard password in the left sidebar to unlock.")
    st.stop()
st.title("📈 AI MarketSurge — CAN SLIM Screener & Chart Terminal")

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/bandblue/marketsurge-alerts/main"

def sync_from_github_if_newer():
    """Pulls the latest cloud-scanned CSVs from GitHub on startup so local matches cloud."""
    for fname in ["marketsurge_rankings.csv", "marketsurge_actionable_setups.csv"]:
        try:
            r = requests.get(f"{GITHUB_RAW_BASE}/{fname}", timeout=8)
            if r.status_code == 200 and len(r.content) > 50:
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

if st.sidebar.button("⬇️ Pull Latest Watchlist from GitHub", use_container_width=True):
    sync_from_github_if_newer()
    st.rerun()

if st.sidebar.button("🔄 Refresh Market Data Now", type="primary", use_container_width=True):
    with st.sidebar.status("Running CAN SLIM market scan...", expanded=True) as status:
        st.write("1/2 Ranking 1–99 RS, Blue Dots, MACD & Acc/Dist (`screener.py`)...")
        res1 = subprocess.run([sys.executable, "screener.py"], capture_output=True, text=True)
        if res1.returncode != 0:
            status.update(label="Screener error", state="error")
            st.sidebar.error(res1.stderr)
            st.stop()

        st.write("2/2 Scanning base patterns & EPS (`base_and_eps_scanner.py`)...")
        res2 = subprocess.run([sys.executable, "base_and_eps_scanner.py"], capture_output=True, text=True)
        if res2.returncode != 0:
            status.update(label="Base scanner error", state="error")
            st.sidebar.error(res2.stderr)
            st.stop()

        status.update(label="Scan complete! Reloading dashboard...", state="complete")
    st.rerun()

if st.sidebar.button("☁️ Sync Watchlist to GitHub", use_container_width=True):
    gh_token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not gh_token:
        st.sidebar.error("GITHUB_TOKEN not found in Windows environment variables. Restart launch_dashboard.bat after setting it.")
    elif not os.path.exists("marketsurge_actionable_setups.csv"):
        st.sidebar.error("No setups CSV found to sync.")
    else:
        with st.sidebar.status("Uploading setups to GitHub cloud...", expanded=True) as sync_status:
            api_url = "https://api.github.com/repos/bandblue/marketsurge-alerts/contents/marketsurge_actionable_setups.csv"
            headers = {
                "Authorization": f"Bearer {gh_token}",
                "Accept": "application/vnd.github+json"
            }
            get_resp = requests.get(api_url, headers=headers, timeout=15)
            sha = get_resp.json().get("sha") if get_resp.status_code == 200 else None

            with open("marketsurge_actionable_setups.csv", "rb") as f:
                encoded_csv = base64.b64encode(f.read()).decode("utf-8")

            payload = {
                "message": f"Update setups ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
                "content": encoded_csv,
                "branch": "main"
            }
            if sha:
                payload["sha"] = sha

            put_resp = requests.put(api_url, headers=headers, json=payload, timeout=15)
            if put_resp.status_code in (200, 201):
                sync_status.update(label="Synced to GitHub! Cloud alerts updated.", state="complete")
            else:
                sync_status.update(label="GitHub upload failed", state="error")
                st.sidebar.error(f"Error {put_resp.status_code}: {put_resp.text}")

st.sidebar.divider()
st.sidebar.header("CAN SLIM Screener Filters")
min_rs = st.sidebar.slider("Minimum RS Rating", 1, 99, 85)
blue_dot_only = st.sidebar.checkbox("🔵 Blue Dot Only (RS Line at 52W High)", value=False)
bullish_macd_only = st.sidebar.checkbox("🟢 Bullish MACD Only", value=False)
stage2_only = st.sidebar.checkbox("Stage 2 Uptrend Only (Price > 50SMA > 200SMA)", value=True)
acc_filter = st.sidebar.multiselect("Acc/Dist Grade", ["A", "B", "C", "D", "E"], default=["A", "B", "C"])

# Load saved scanner CSVs
try:
    df_all = pd.read_csv("marketsurge_rankings.csv")
    df_setups = pd.read_csv("marketsurge_actionable_setups.csv")
except FileNotFoundError:
    st.warning("No scan files found yet. Click **Refresh Market Data Now** in the left sidebar to run your first scan.")
    st.stop()

filtered_df = df_all[
    (df_all["RS_Rating"] >= min_rs) &
    (df_all["Acc_Dist"].isin(acc_filter))
]
if stage2_only:
    filtered_df = filtered_df[filtered_df["Stage2_Trend"] == True]
if blue_dot_only and "Blue_Dot" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Blue_Dot"] == True]
if bullish_macd_only and "MACD_Signal" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["MACD_Signal"].str.contains("Bullish", na=False)]

display_setups = df_setups.copy()
if blue_dot_only and "Blue_Dot" in display_setups.columns:
    display_setups = display_setups[display_setups["Blue_Dot"] == "🔵 YES"]
if bullish_macd_only and "MACD_Signal" in display_setups.columns:
    display_setups = display_setups[display_setups["MACD_Signal"].str.contains("Bullish", na=False)]

# --- TOP SECTION: ACTIONABLE BREAKOUT WATCHLIST ---
st.subheader("🎯 Actionable Base Setups (Stage 2 + RS ≥ 85)")
st.dataframe(display_setups, use_container_width=True, hide_index=True)
if not display_setups.empty:
    c_exp1, c_exp2 = st.columns([1, 2])
    fidelity_csv = display_setups[["Ticker"]].rename(columns={"Ticker": "Symbol"}).to_csv(index=False)
    c_exp1.download_button(
        label="📥 Download Fidelity Watchlist CSV",
        data=fidelity_csv,
        file_name="MarketSurge_Breakouts.csv",
        mime="text/csv",
        use_container_width=True
    )
    ticker_list_str = ", ".join(display_setups["Ticker"].tolist())
    c_exp2.text_input("Or copy & paste tickers into Fidelity Watch List:", value=ticker_list_str)
st.divider()

# --- MIDDLE SECTION: 5-PANE CHART + POSITION CALCULATOR & AI ANALYST ---
col_chart, col_right = st.columns([2.2, 1])

with col_chart:
    st.subheader("📊 Multi-Pane Chart: Price, Volume, OBV, RSI & MACD")
    source_df = display_setups if not display_setups.empty else df_all
    ticker_options = source_df["Ticker"].tolist()
    name_lookup = dict(zip(df_all["Ticker"], df_all["Company"])) if "Company" in df_all.columns else {}

    selected_ticker = st.selectbox(
        "Select Stock to Inspect:",
        ticker_options,
        format_func=lambda t: f"{t} — {name_lookup.get(t, t)}"
    )

    hist = yf.Ticker(selected_ticker).history(period="9mo")
    spy_hist = yf.Ticker("SPY").history(period="9mo")

    hist.index = hist.index.tz_localize(None).normalize()
    spy_hist.index = spy_hist.index.tz_localize(None).normalize()

    # Moving Averages & RS Line
    hist["SMA_21"] = hist["Close"].ewm(span=21).mean()
    hist["SMA_50"] = hist["Close"].rolling(50).mean()
    hist["RS_Line"] = (hist["Close"] / spy_hist["Close"].reindex(hist.index).ffill()) * 100
    hist["Vol_50_Avg"] = hist["Volume"].rolling(50).mean()

    # On-Balance Volume (OBV) + 20-Day OBV EMA
    price_dir = np.sign(hist["Close"].diff()).fillna(0)
    hist["OBV"] = (price_dir * hist["Volume"]).cumsum()
    hist["OBV_EMA20"] = hist["OBV"].ewm(span=20, adjust=False).mean()

    # 14-Day Wilder's RSI
    delta = hist["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    hist["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    ema12 = hist["Close"].ewm(span=12, adjust=False).mean()
    ema26 = hist["Close"].ewm(span=26, adjust=False).mean()
    hist["MACD"] = ema12 - ema26
    hist["MACD_Signal"] = hist["MACD"].ewm(span=9, adjust=False).mean()
    hist["MACD_Hist"] = hist["MACD"] - hist["MACD_Signal"]

    plot_df = hist.tail(130).copy()
    setup_row = df_setups[df_setups["Ticker"] == selected_ticker]

    # 5-row synchronized chart
    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.025,
        row_heights=[0.44, 0.13, 0.14, 0.14, 0.15],
        specs=[
            [{"secondary_y": True}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
        ]
    )

    # Row 1: Candlesticks, Moving Averages, and RS Line
    fig.add_trace(
        go.Candlestick(
            x=plot_df.index, open=plot_df["Open"], high=plot_df["High"],
            low=plot_df["Low"], close=plot_df["Close"], name=selected_ticker
        ),
        row=1, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df["SMA_21"], line=dict(color="#00b4d8", width=1.2), name="21-Day EMA"),
        row=1, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df["SMA_50"], line=dict(color="#e63946", width=1.5), name="50-Day SMA"),
        row=1, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df.index, y=plot_df["RS_Line"],
            line=dict(color="#3b82f6", width=2.2),
            name="RS Line vs SPY"
        ),
        row=1, col=1, secondary_y=True
    )

    # Row 2: Daily Volume Bars + 50-Day Volume Average
    vol_colors = [
        "#22c55e" if c >= o else "#ef4444"
        for c, o in zip(plot_df["Close"], plot_df["Open"])
    ]
    fig.add_trace(
        go.Bar(x=plot_df.index, y=plot_df["Volume"], marker_color=vol_colors, opacity=0.75, name="Volume"),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df["Vol_50_Avg"], line=dict(color="#eab308", width=1.4), name="50D Vol Avg"),
        row=2, col=1
    )

    # Row 3: On-Balance Volume (OBV) + 20-Day EMA
    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df["OBV"], line=dict(color="#10b981", width=1.8), name="OBV"),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df["OBV_EMA20"], line=dict(color="#94a3b8", width=1.1, dash="dot"), name="OBV 20 EMA"),
        row=3, col=1
    )

    # Row 4: 14-Day RSI + 70/30 Thresholds
    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df["RSI_14"], line=dict(color="#c084fc", width=1.7), name="RSI (14)"),
        row=4, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", line_width=1, row=4, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#22c55e", line_width=1, row=4, col=1)

    # Row 5: MACD (12, 26, 9)
    hist_colors = ["#22c55e" if val >= 0 else "#ef4444" for val in plot_df["MACD_Hist"]]
    fig.add_trace(
        go.Bar(x=plot_df.index, y=plot_df["MACD_Hist"], marker_color=hist_colors, opacity=0.5, name="MACD Hist"),
        row=5, col=1
    )
    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df["MACD"], line=dict(color="#38bdf8", width=1.6), name="MACD (12,26)"),
        row=5, col=1
    )
    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df["MACD_Signal"], line=dict(color="#f97316", width=1.6), name="Signal (9)"),
        row=5, col=1
    )

    # Pivot, 5% Buy Zone & 7.5% Stop-Loss overlays
    default_entry = float(plot_df["Close"].iloc[-1])
    default_stop = round(default_entry * 0.925, 2)

    if not setup_row.empty and pd.notna(setup_row.iloc[0]["Pivot_Buy"]):
        pivot = float(setup_row.iloc[0]["Pivot_Buy"])
        buy_max = float(setup_row.iloc[0]["Buy_Zone_Max"])
        stop_loss = float(setup_row.iloc[0]["Stop_Loss_7.5Pct"])
        default_entry = pivot
        default_stop = stop_loss

        fig.add_hrect(y0=pivot, y1=buy_max, fillcolor="green", opacity=0.15, line_width=0, annotation_text="5% Buy Zone", row=1, col=1, secondary_y=False)
        fig.add_hline(y=pivot, line_dash="dash", line_color="green", annotation_text=f"Pivot: ${pivot}", row=1, col=1, secondary_y=False)
        fig.add_hline(y=stop_loss, line_dash="dot", line_color="red", annotation_text=f"7.5% Stop: ${stop_loss}", row=1, col=1, secondary_y=False)

    fig.update_layout(
        height=980,
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
        xaxis3_rangeslider_visible=False,
        xaxis4_rangeslider_visible=False,
        xaxis5_rangeslider_visible=False,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=25, b=10)
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="RS Line", showgrid=False, row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    fig.update_yaxes(title_text="OBV", row=3, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=4, col=1)
    fig.update_yaxes(title_text="MACD", row=5, col=1)
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    # --- POSITION SIZE & 7.5% STOP-LOSS RISK CALCULATOR ---
    st.subheader("🧮 Position Size & Risk Calculator")
    with st.container(border=True):
        st.caption(f"Auto-loaded for **{selected_ticker}** (7.5% CAN SLIM Stop Rule)")
        acct_size = st.number_input("Account Size ($)", min_value=1000.0, value=100000.0, step=5000.0, format="%.2f")
        risk_pct = st.slider("Max Account Risk per Trade (%)", min_value=0.25, max_value=3.0, value=1.0, step=0.25)

        c1, c2 = st.columns(2)
        entry_price = c1.number_input("Entry / Pivot ($)", min_value=0.01, value=float(round(default_entry, 2)), step=0.50, format="%.2f")
        stop_price = c2.number_input("Stop-Loss ($)", min_value=0.01, value=float(round(default_stop, 2)), step=0.50, format="%.2f")

        risk_per_share = entry_price - stop_price
        if risk_per_share <= 0:
            st.error("Stop-Loss must be lower than Entry Price.")
        else:
            max_dollar_risk = acct_size * (risk_pct / 100.0)
            shares_to_buy = int(max_dollar_risk // risk_per_share)
            position_cost = shares_to_buy * entry_price
            portfolio_wt = (position_cost / acct_size) * 100
            stop_pct_actual = (risk_per_share / entry_price) * 100
            target_20_price = entry_price * 1.20
            est_20_gain = shares_to_buy * (target_20_price - entry_price)

            m1, m2 = st.columns(2)
            m1.metric("Shares to Buy", f"{shares_to_buy:,}")
            m2.metric("Position Cost", f"${position_cost:,.2f}", f"{portfolio_wt:.1f}% of Acct")

            m3, m4 = st.columns(2)
            m3.metric("Max Dollar Risk", f"-${shares_to_buy * risk_per_share:,.2f}", f"-{stop_pct_actual:.1f}% Stop")
            m4.metric("+20% Target Gain", f"+${est_20_gain:,.2f}", f"${target_20_price:,.2f}")

    st.divider()

    # --- GEMINI CAN SLIM ASSISTANT ---
    st.subheader("🤖 Gemini CAN SLIM Assistant")
    latest_rsi = round(float(plot_df["RSI_14"].iloc[-1]), 1)
    obv_rising = float(plot_df["OBV"].iloc[-1]) > float(plot_df["OBV_EMA20"].iloc[-1])
    obv_trend_str = "Rising above 20D EMA" if obv_rising else "Below 20D EMA"

    user_q = st.text_area(
        "Ask about your watchlist or selected stock:",
        value=f"Analyze {selected_ticker}'s setup, Blue Dot RS status, OBV ({obv_trend_str}), RSI ({latest_rsi}), MACD momentum, EPS growth, and exact buy/stop levels."
    )
    if st.button("Ask AI Analyst", type="primary"):
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_key:
            st.error("GEMINI_API_KEY not found in environment variables.")
        else:
            with st.spinner("Analyzing watchlist & chart data..."):
                client = genai.Client(api_key=gemini_key, http_options=types.HttpOptions(timeout=15000))
                context_csv = df_setups.to_csv(index=False)
                prompt = (
                    f"You are an IBD MarketSurge portfolio analyst. Here is today's screened CAN SLIM setup table:\n"
                    f"{context_csv}\n\nSelected ticker live technicals: {selected_ticker} | RSI(14): {latest_rsi} | OBV: {obv_trend_str}.\n"
                    f"User question: {user_q}\n"
                    f"Give a concise, actionable CAN SLIM assessment citing exact RS, Blue Dot status, OBV & RSI readings, "
                    f"MACD momentum, EPS/Sales YoY %, Acc/Dist grade, pivot price, 5% buy zone, and 7.5% stop-loss."
                )
                for m in ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]:
                    try:
                        resp = client.models.generate_content(model=m, contents=prompt)
                        st.markdown(resp.text)
                        break
                    except Exception:
                        continue

# --- BOTTOM SECTION: FULL RANKED UNIVERSE ---
with st.expander("📋 View Full Ranked Watchlist"):
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)
