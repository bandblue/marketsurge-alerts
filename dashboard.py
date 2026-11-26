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
    if hist.empty or len(hist) < 60:
        return None

    hist.index = hist.index.tz_localize(None).normalize()
    hist["SMA_50"] = hist["Close"].rolling(50).mean()
    hist["SMA_200"] = hist["Close"].rolling(200).mean()
    hist["High_52W_Step"] = hist["High"].rolling(252, min_periods=20).max()

    try:
        iwb = yf.Ticker("IWB").history(period="2y")
        if not iwb.empty:
            iwb.index = iwb.index.tz_localize(None).normalize()
            iwb_aligned = iwb["Close"].reindex(hist.index).ffill()
            hist["Rel_IWB"] = (hist["Close"] / iwb_aligned) * 100.0
        else:
            hist["Rel_IWB"] = 100.0
    except Exception:
        hist["Rel_IWB"] = 100.0

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

    plot_df = hist.tail(252).copy()
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

    ret_1m = ((last_close / float(hist["Close"].iloc[-22])) - 1.0) * 100.0 if len(hist) > 22 else 0.0
    ret_3m = ((last_close / float(hist["Close"].iloc[-64])) - 1.0) * 100.0 if len(hist) > 64 else 0.0
    ret_6m = ((last_close / float(hist["Close"].iloc[-127])) - 1.0) * 100.0 if len(hist) > 127 else 0.0

    ytd_df = plot_df[plot_df.index.year == plot_df.index[-1].year]
    ret_ytd = ((last_close / float(ytd_df["Close"].iloc[0])) - 1.0) * 100.0 if not ytd_df.empty else 0.0

    rolling_peak = plot_df["High"].cummax()
    drawdowns = ((plot_df["Low"] - rolling_peak) / rolling_peak) * 100.0
    max_dd_1y = int(round(float(drawdowns.min())))

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

    above_50_series = plot_df["Close"] > plot_df["SMA_50"]
    cross_up_dates = plot_df.index[above_50_series & (~above_50_series.shift(1).fillna(True))]
    added_idx = cross_up_dates[-1] if len(cross_up_dates) > 0 else None

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

            st.info(
                f"**{symbol}:** ${last_close:,.0f}, with the 50-day at ${sma50:,.0f} and the 200-day at ${sma200:,.0f}. "
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
# TAB 1: JOSH BROWN BEST STOCKS IN THE MARKET (WATCHLIST FILTERS + CLICK-TO-CHART)
# ------------------------------------------------------------------------------
with tab_best:
    st.subheader("🏆 Best Stocks in the Market — Institutional Trend & Tear Sheet")

    # Placeholder renders the tear-sheet chart at the top after reading watchlist row clicks below
    tear_sheet_slot = st.empty()

    st.divider()

    # Watchlist Header + Filters placed directly on the Watchlist section
    wl_head, wl_f1, wl_f2, wl_f3 = st.columns([1.8, 1.2, 1.1, 1.1])
    wl_head.markdown("### 📋 `josh_brown_best_stocks.csv` Watchlist")
    zone_filter = wl_f1.selectbox(
        "Entry Zone Filter:",
        ["All Zones", "BREAKOUT_ZONE", "PULLBACK_SUPPORT", "TRENDING_UP"],
        key="jb_zone_filter",
    )
    custom_ticker = wl_f2.text_input(
        "Lookup Any Ticker (e.g. A, BIIB):",
        value="",
        key="jb_custom_ticker",
    ).strip().upper()
    filter_table_to_selected = wl_f3.checkbox(
        "Filter table to selected row",
        value=False,
        key="jb_filter_table_to_selected",
    )
    st.caption("Click the selection box on the left of any stock row below to update the tear-sheet chart above.")

    default_ticker = (
        str(df_best["Ticker"].iloc[0])
        if (not df_best.empty and "Ticker" in df_best.columns)
        else "SN"
    )
    table_clicked_ticker = None

    if df_best.empty:
        st.info("No rows in `josh_brown_best_stocks.csv` yet. Run **MarketSurge Daily Pre-Market Scan** in GitHub Actions or click **Pull Latest Watchlists from GitHub**.")
    else:
        filtered_best = df_best.copy()
        if zone_filter != "All Zones" and "Entry_Zone" in filtered_best.columns:
            filtered_best = filtered_best[filtered_best["Entry_Zone"].str.contains(zone_filter, na=False)]
        filtered_best = filtered_best.reset_index(drop=True)

        # Read previous selection from session state if filtering the table down to 1 row
        prev_sel = st.session_state.get("jb_watchlist_table", {}).get("selection", {}).get("rows", [])
        if filter_table_to_selected and prev_sel and prev_sel[0] < len(filtered_best):
            table_clicked_ticker = str(filtered_best.iloc[prev_sel[0]]["Ticker"])
            display_best_df = filtered_best.iloc[[prev_sel[0]]].reset_index(drop=True)
        else:
            display_best_df = filtered_best

        best_table_event = st.dataframe(
            display_best_df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="jb_watchlist_table",
        )
        selected_rows = best_table_event.selection.rows if best_table_event else []
        if selected_rows and "Ticker" in display_best_df.columns:
            table_clicked_ticker = str(display_best_df.iloc[selected_rows[0]]["Ticker"])

        b_exp1, b_exp2 = st.columns([1, 2])
        jb_fidelity_csv = display_best_df[["Ticker"]].rename(columns={"Ticker": "Symbol"}).to_csv(index=False)
        b_exp1.download_button(
            label="📥 Download Best Stocks CSV (Fidelity)",
            data=jb_fidelity_csv,
            file_name="JoshBrown_BestStocks.csv",
            mime="text/csv",
            use_container_width=True,
        )
        b_exp2.text_input(
            "Copy & paste Best Stocks tickers into Fidelity Watch List:",
            value=", ".join(display_best_df["Ticker"].tolist()),
            key="jb_copy_tickers",
        )

    # Priority: Clicked Watchlist Row > Custom Lookup Input > Top Ranked Stock
    active_best_ticker = table_clicked_ticker or custom_ticker or default_ticker
    with tear_sheet_slot.container():
        render_compound_tear_sheet(active_best_ticker, df_best)


# ------------------------------------------------------------------------------
# TAB 2: CAN SLIM BASE SETUPS (CLICK-TO-CHART + 5-PANE CHART + AI ANALYST)
# ------------------------------------------------------------------------------
with tab_canslim:
    st.sidebar.divider()
    st.sidebar.header("CAN SLIM Screener Filters")
    min_rs = st.sidebar.slider("Minimum RS Rating", 1, 99, 85)
    blue_dot_only = st.sidebar.checkbox("🔵 Blue Dot Only (RS Line at 52W High)", value=False)
    bullish_macd_only = st.sidebar.checkbox("🟢 Bullish MACD Only", value=False)
    stage2_only = st.sidebar.checkbox("Stage 2 Uptrend Only (Price > 50SMA > 200SMA)", value=True)
    acc_filter = st.sidebar.multiselect("Acc/Dist Grade", ["A", "B", "C", "D", "E"], default=["A", "B", "C"])

    if df_all.empty or df_setups.empty:
        st.warning("No CAN SLIM scan files found yet. Click **Pull Latest Watchlists from GitHub** in the sidebar.")
    else:
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
        display_setups = display_setups.reset_index(drop=True)

        st.subheader("🎯 Actionable Base Setups (Stage 2 + RS ≥ 85 + ATR% ≥ 1.2%) *(Click any row to filter chart below)*")
        cs_table_event = st.dataframe(
            display_setups,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="cs_watchlist_table",
        )
        cs_selected_rows = cs_table_event.selection.rows if cs_table_event else []
        cs_clicked_ticker = (
            str(display_setups.iloc[cs_selected_rows[0]]["Ticker"])
            if (cs_selected_rows and "Ticker" in display_setups.columns)
            else None
        )

        if not display_setups.empty:
            c_exp1, c_exp2 = st.columns([1, 2])
            fidelity_csv = display_setups[["Ticker"]].rename(columns={"Ticker": "Symbol"}).to_csv(index=False)
            c_exp1.download_button(
                label="📥 Download Fidelity Watchlist CSV",
                data=fidelity_csv,
                file_name="MarketSurge_Breakouts.csv",
                mime="text/csv",
                use_container_width=True,
            )
            ticker_list_str = ", ".join(display_setups["Ticker"].tolist())
            c_exp2.text_input("Or copy & paste tickers into Fidelity Watch List:", value=ticker_list_str)
        st.divider()

        col_chart, col_right = st.columns([2.2, 1])

        with col_chart:
            st.subheader("📊 Multi-Pane Chart: Price, Volume, OBV, RSI & MACD")
            source_df = display_setups if not display_setups.empty else df_all
            ticker_options = source_df["Ticker"].tolist()
            name_lookup = dict(zip(df_all["Ticker"], df_all["Company"])) if "Company" in df_all.columns else {}

            dropdown_ticker = st.selectbox(
                "Or Choose Stock From Dropdown:",
                ticker_options,
                format_func=lambda t: f"{t} — {name_lookup.get(t, t)}",
            )
            selected_ticker = cs_clicked_ticker if cs_clicked_ticker else dropdown_ticker

            hist = yf.Ticker(selected_ticker).history(period="9mo")
            spy_hist = yf.Ticker("SPY").history(period="9mo")

            hist.index = hist.index.tz_localize(None).normalize()
            spy_hist.index = spy_hist.index.tz_localize(None).normalize()

            hist["SMA_21"] = hist["Close"].ewm(span=21).mean()
            hist["SMA_50"] = hist["Close"].rolling(50).mean()
            hist["RS_Line"] = (hist["Close"] / spy_hist["Close"].reindex(hist.index).ffill()) * 100
            hist["Vol_50_Avg"] = hist["Volume"].rolling(50).mean()

            price_dir = np.sign(hist["Close"].diff()).fillna(0)
            hist["OBV"] = (price_dir * hist["Volume"]).cumsum()
            hist["OBV_EMA20"] = hist["OBV"].ewm(span=20, adjust=False).mean()

            delta = hist["Close"].diff()
            gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
            loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
            rs = gain / loss.replace(0, np.nan)
            hist["RSI_14"] = 100 - (100 / (1 + rs))

            ema12 = hist["Close"].ewm(span=12, adjust=False).mean()
            ema26 = hist["Close"].ewm(span=26, adjust=False).mean()
            hist["MACD"] = ema12 - ema26
            hist["MACD_Signal"] = hist["MACD"].ewm(span=9, adjust=False).mean()
            hist["MACD_Hist"] = hist["MACD"] - hist["MACD_Signal"]

            plot_df = hist.tail(130).copy()
            setup_row = df_setups[df_setups["Ticker"] == selected_ticker]

            fig = make_subplots(
                rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.025,
                row_heights=[0.44, 0.13, 0.14, 0.14, 0.15],
                specs=[
                    [{"secondary_y": True}],
                    [{"secondary_y": False}],
                    [{"secondary_y": False}],
                    [{"secondary_y": False}],
                    [{"secondary_y": False}],
                ],
            )

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

            vol_colors = ["#22c55e" if c >= o else "#ef4444" for c, o in zip(plot_df["Close"], plot_df["Open"])]
            fig.add_trace(
                go.Bar(x=plot_df.index, y=plot_df["Volume"], marker_color=vol_colors, opacity=0.75, name="Volume"),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=plot_df.index, y=plot_df["Vol_50_Avg"], line=dict(color="#eab308", width=1.4), name="50D Vol Avg"),
                row=2, col=1
            )

            fig.add_trace(
                go.Scatter(x=plot_df.index, y=plot_df["OBV"], line=dict(color="#10b981", width=1.8), name="OBV"),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=plot_df.index, y=plot_df["OBV_EMA20"], line=dict(color="#94a3b8", width=1.1, dash="dot"), name="OBV 20 EMA"),
                row=3, col=1
            )

            fig.add_trace(
                go.Scatter(x=plot_df.index, y=plot_df["RSI_14"], line=dict(color="#c084fc", width=1.7), name="RSI (14)"),
                row=4, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", line_width=1, row=4, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="#22c55e", line_width=1, row=4, col=1)

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

        with st.expander("📋 View Full Ranked Watchlist"):
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)


# ------------------------------------------------------------------------------
# TAB 3: HIGHER HIGHS & HIGHER LOWS (CLICK-TO-CHART + BROAD-MARKET TABLE)
# ------------------------------------------------------------------------------
with tab_hh_hl:
    st.subheader("📈 Broad-Market Higher Highs & Higher Lows (`marketsurge_hh_hl_setups.csv`)")
    if df_hh_hl.empty:
        st.info("No Higher High / Higher Low setups loaded yet. Click **Pull Latest Watchlists from GitHub** in the sidebar.")
    else:
        hh_filter = st.radio(
            "Filter by Setup Type:",
            ["All Setups", "PULLBACK_ENTRY (21 EMA)", "BREAKOUT_READY (Swing High)"],
            horizontal=True,
        )
        disp_hh = df_hh_hl.copy()
        if hh_filter != "All Setups" and "Setup_Type" in disp_hh.columns:
            disp_hh = disp_hh[disp_hh["Setup_Type"] == hh_filter]
        disp_hh = disp_hh.reset_index(drop=True)

        st.caption("Click the selection box on the left of any row to view its tear-sheet chart below.")
        hh_table_event = st.dataframe(
            disp_hh,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="hh_watchlist_table",
        )
        hh_selected_rows = hh_table_event.selection.rows if hh_table_event else []

        hh_exp1, hh_exp2 = st.columns([1, 2])
        hh_csv = disp_hh[["Ticker"]].rename(columns={"Ticker": "Symbol"}).to_csv(index=False)
        hh_exp1.download_button(
            label="📥 Download HH/HL Watchlist CSV (Fidelity)",
            data=hh_csv,
            file_name="HH_HL_Setups.csv",
            mime="text/csv",
            use_container_width=True,
        )
        hh_exp2.text_input(
            "Copy & paste HH/HL tickers into Fidelity Watch List:",
            value=", ".join(disp_hh["Ticker"].tolist()),
            key="hh_copy_tickers",
        )

        if hh_selected_rows and "Ticker" in disp_hh.columns:
            st.divider()
            render_compound_tear_sheet(str(disp_hh.iloc[hh_selected_rows[0]]["Ticker"]), df_best)
