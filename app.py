import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="Stock Analyzer MVP", page_icon="📈", layout="wide")

# ----------------- SESSION STATE -----------------
if "results" not in st.session_state:
    st.session_state["results"] = None

if "last_query" not in st.session_state:
    st.session_state["last_query"] = ""

# ----------------- SIMPLE USAGE LOGGER -----------------
LOG_FILE = "events_log.csv"


def log_event(event_type: str, tickers: str):
    try:
        ts = datetime.utcnow().isoformat()
        row = f"{ts}|{event_type}|{tickers}\n"
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(row)
    except Exception:
        # Never let logging break the app
        pass


# ----------------- INDICATOR HELPERS -----------------
def to_float(x):
    """Force pandas/numpy objects into a clean float."""
    try:
        return float(x)
    except Exception:
        return np.nan


def compute_rsi(series, period: int = 14) -> float:
    arr = np.asarray(series, dtype="float64").reshape(-1)
    arr = arr[~np.isnan(arr)]
    if arr.size < period + 1:
        return np.nan

    delta = np.diff(arr)
    gain = np.clip(delta, 0, None)
    loss = -np.clip(delta, None, 0)

    avg_gain = pd.Series(gain).rolling(period).mean()
    avg_loss = pd.Series(loss).rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def safe_pct_change(a, b) -> float:
    try:
        a = float(a)
        b = float(b)
    except Exception:
        return np.nan

    if b == 0 or np.isnan(a) or np.isnan(b):
        return np.nan
    return (a / b - 1.0) * 100.0


def simple_label_from_score(score: float) -> str:
    if score >= 65:
        return "Bullish"
    elif score <= 35:
        return "Bearish"
    else:
        return "Neutral"


def label_with_emoji(label: str) -> str:
    if label == "Bullish":
        return "🟢 Bullish"
    if label == "Bearish":
        return "🔴 Bearish"
    return "⚪ Neutral"


def fetch_price_data(ticker: str, period: str = "3mo") -> pd.DataFrame:
    data = yf.download(ticker, period=period, progress=False)
    return data if data is not None else pd.DataFrame()


# ----------------- CORE CALCULATION ENGINE -----------------
def compute_indicators_for_ticker(ticker: str) -> dict:
    data = fetch_price_data(ticker)
    if data.empty or len(data) < 30:
        return {"error": "Not enough data"}

    close = data["Close"].astype(float)
    volume = data["Volume"].astype(float)

    # --- Day-to-day momentum ---
    today_change = safe_pct_change(close.iloc[-1], close.iloc[-2])
    five_day_change = safe_pct_change(close.iloc[-1], close.iloc[-6]) if len(close) >= 6 else np.nan

    # --- Trend regression ---
    if len(close) >= 20:
        recent = close.iloc[-20:]
        x = np.arange(len(recent))
        slope, _ = np.polyfit(x, recent.values, 1)
        trend_20 = (slope * 20 / recent.iloc[-1]) * 100
    else:
        trend_20 = np.nan

    # --- RSI ---
    rsi_14 = compute_rsi(close)

    # --- Volume spike ---
    if len(volume) >= 21:
        vol_avg = float(volume.iloc[-21:-1].mean())
        last_vol = float(volume.iloc[-1])
        vol_spike = last_vol / vol_avg if vol_avg > 0 else np.nan
    else:
        vol_spike = np.nan

    # --- SMAs ---
    sma_20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else np.nan
    sma_50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else np.nan

    if not np.isnan(sma_20) and not np.isnan(sma_50):
        if sma_20 > sma_50:
            sma_crossover = "Bullish (20 > 50)"
        elif sma_20 < sma_50:
            sma_crossover = "Bearish (20 < 50)"
        else:
            sma_crossover = "Neutral"
    else:
        sma_crossover = "Not enough data"

    # --- Normalize to floats ---
    today_change = to_float(today_change)
    five_day_change = to_float(five_day_change)
    trend_20 = to_float(trend_20)
    rsi_14 = to_float(rsi_14)
    vol_spike = to_float(vol_spike)

    # --- Scoring ---
    def clamp(x, lo, hi):
        return max(lo, min(hi, x))

    today_score = clamp((today_change + 4) / 8 * 100, 0, 100) if not np.isnan(today_change) else 50
    five_score = clamp((five_day_change + 10) / 20 * 100, 0, 100) if not np.isnan(five_day_change) else 50
    trend_score = clamp((trend_20 + 15) / 30 * 100, 0, 100) if not np.isnan(trend_20) else 50

    if not np.isnan(rsi_14):
        if rsi_14 < 30:
            rsi_score = 70
        elif rsi_14 > 70:
            rsi_score = 30
        else:
            rsi_score = 60 - ((rsi_14 - 30) / 40) * 20
    else:
        rsi_score = 50

    vol_score = clamp((vol_spike - 0.5) / 2.5 * 100, 0, 100) if not np.isnan(vol_spike) else 50

    overall = (
        today_score * 0.25 +
        five_score * 0.20 +
        trend_score * 0.25 +
        rsi_score * 0.15 +
        vol_score * 0.15
    )

    label = simple_label_from_score(overall)

    explanation = (
        f"Today change: **{today_change:.2f}%**; 5-day: **{five_day_change:.2f}%**.  \n"
        f"20-day trend: **{trend_20:.2f}%**.  \n"
        f"RSI(14): **{rsi_14:.1f}**.  \n"
        f"Volume spike: **{vol_spike:.2f}x**.  \n"
        f"SMA status: **{sma_crossover}**.  \n\n"
        f"Overall strength: **{overall:.1f} / 100** → **{label}**."
    )

    return {
        "today_change": today_change,
        "five_day_change": five_day_change,
        "trend_20": trend_20,
        "rsi_14": rsi_14,
        "vol_spike": vol_spike,
        "sma_20": sma_20,
        "sma_50": sma_50,
        "sma_crossover": sma_crossover,
        "overall_score": overall,
        "label": label,
        "explanation": explanation,
    }


# ----------------- UI: SIDEBAR -----------------
st.sidebar.title("📊 Stock Analyzer MVP")
page = st.sidebar.radio("Navigation", ["Analyzer", "Top Movers"])
st.sidebar.markdown("---")
st.sidebar.caption("v0.3 – Momentum + RSI + Volume + SMA")


# ----------------- PAGE 1: ANALYZER -----------------
if page == "Analyzer":
    st.title("📈 Stock Analyzer")
    st.write("Enter tickers like `AAPL, TSLA, NVDA` and click **Analyze**.")

    tickers_input = st.text_input("Tickers", "AAPL, TSLA, NVDA")
    analyze_clicked = st.button("🔍 Analyze")

    if analyze_clicked:
        tickers_str = tickers_input.strip()
        log_event("analyze_clicked", tickers_str)

        tickers = [t.upper() for t in tickers_str.replace(" ", "").split(",") if t]

        results = []
        for t in tickers:
            with st.spinner(f"Analyzing {t}..."):
                info = compute_indicators_for_ticker(t)
                row = {"Ticker": t, **info}
                results.append(row)

        st.session_state["results"] = results

    if st.session_state["results"]:
        results = st.session_state["results"]

        st.markdown("### 🔦 Summary Signals")
        for row in results:
            if "error" in row:
                st.error(f"{row['Ticker']}: {row['error']}")
                continue

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"**{row['Ticker']}**")
                st.markdown(f"### {label_with_emoji(row['label'])}")
                st.markdown(f"**Signal Strength:** {row['overall_score']:.1f}/100")
            with col2:
                st.metric("Today %", f"{row['today_change']:.2f}%")
                st.metric("5-Day %", f"{row['five_day_change']:.2f}%")
            with col3:
                st.metric("20-Day Trend %", f"{row['trend_20']:.2f}%")
                st.metric("RSI(14)", f"{row['rsi_14']:.1f}")
            with col4:
                st.metric("Volume Spike (x)", f"{row['vol_spike']:.2f}")
                st.markdown(f"**SMA:** {row['sma_crossover']}")

            with st.expander("View explanation"):
                st.markdown(row["explanation"])

            st.markdown("---")


# ----------------- PAGE 2: TOP MOVERS -----------------
elif page == "Top Movers":
    st.title("🔥 Top Movers (MVP)")

    st.write(
        "This scans a fixed list of popular tickers and ranks them by today's move "
        "and overall signal strength using the same engine as the Analyzer."
    )

    # Simple hard-coded universe for now (you can edit this list anytime)
    default_universe = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
        "META", "NVDA", "NFLX", "AMD", "INTC",
        "JPM", "BAC", "XOM", "CVX", "WMT",
        "COST", "DIS", "QQQ", "SPY", "IWM",
    ]

    tickers_str = st.text_input(
        "Universe (comma-separated)", ", ".join(default_universe)
    )

    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        scan_clicked = st.button("🚀 Scan Top Movers")

    if scan_clicked:
        universe = [
            t.strip().upper()
            for t in tickers_str.replace(" ", "").split(",")
            if t.strip()
        ]

        # Track this event (still private)
        log_event("top_movers_scan", ",".join(universe))

        movers = []
        with st.spinner("Scanning universe..."):
            for t in universe:
                info = compute_indicators_for_ticker(t)
                row = {"Ticker": t, **info}
                movers.append(row)

        # Filter out errors
        movers_ok = [m for m in movers if "error" not in m]

        if not movers_ok:
            st.warning("No valid data for the selected universe.")
        else:
            # Sort by signal strength descending
            top_by_score = sorted(
                movers_ok, key=lambda r: r["overall_score"], reverse=True
            )[:10]

            # Sort by today's % change descending
            top_by_today = sorted(
                movers_ok, key=lambda r: r["today_change"], reverse=True
            )[:10]

            st.markdown("### 🔝 Top 10 by Signal Strength")
            for row in top_by_score:
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.markdown(f"**{row['Ticker']}** — {label_with_emoji(row['label'])}")
                    st.markdown(
                        f"Signal: **{row['overall_score']:.1f} / 100**  \n"
                        f"SMA: {row['sma_crossover']}"
                    )
                with col2:
                    st.metric("Today %", f"{row['today_change']:.2f}%")
                    st.metric("5-Day %", f"{row['five_day_change']:.2f}%")
                with col3:
                    st.metric("RSI(14)", f"{row['rsi_14']:.1f}")
                    st.metric("Vol Spike (x)", f"{row['vol_spike']:.2f}")
                st.markdown("---")

            st.markdown("### 📈 Top 10 by Today % Move")
            for row in top_by_today:
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.markdown(f"**{row['Ticker']}** — {label_with_emoji(row['label'])}")
                    st.markdown(
                        f"Signal: **{row['overall_score']:.1f} / 100**  \n"
                        f"SMA: {row['sma_crossover']}"
                    )
                with col2:
                    st.metric("Today %", f"{row['today_change']:.2f}%")
                    st.metric("5-Day %", f"{row['five_day_change']:.2f}%")
                with col3:
                    st.metric("RSI(14)", f"{row['rsi_14']:.1f}")
                    st.metric("Vol Spike (x)", f"{row['vol_spike']:.2f}")
                st.markdown("---")

