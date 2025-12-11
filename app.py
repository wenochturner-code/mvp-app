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
    """
    Lightweight event logger.
    Writes events to a local CSV file for usage analytics.
    Uses '|' as separator so commas in tickers are safe.
    """
    try:
        ts = datetime.utcnow().isoformat()
        row = f"{ts}|{event_type}|{tickers}\n"
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(row)
    except Exception:
        # Never let logging break the app
        pass


# ----------------- INDICATOR HELPERS -----------------
def compute_rsi(series, period: int = 14) -> float:
    """
    Robust RSI(14) that:
    - flattens any input to 1D
    - handles NaNs safely
    """
    # Force to 1D float array
    arr = np.asarray(series, dtype="float64").reshape(-1)

    # Drop NaNs
    arr = arr[~np.isnan(arr)]

    if arr.size < period + 1:
        return np.nan

    # Price changes
    delta = np.diff(arr)

    # Gains (up moves) and losses (down moves)
    gain = np.clip(delta, 0, None)
    loss = -np.clip(delta, None, 0)

    gain_s = pd.Series(gain)
    loss_s = pd.Series(loss)

    avg_gain = gain_s.rolling(window=period, min_periods=period).mean()
    avg_loss = loss_s.rolling(window=period, min_periods=period).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return float(rsi.iloc[-1])


def safe_pct_change(a, b) -> float:
    """
    Safe percentage change that works even if a/b are pandas scalars / arrays.
    This is what was throwing your ValueError before.
    """
    try:
        # Force to plain floats – avoids "truth value of a Series is ambiguous"
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
    if data is None or data.empty:
        return pd.DataFrame()
    return data


def compute_indicators_for_ticker(ticker: str) -> dict:
    """
    Returns a dict with:
    - today_change, five_day_change, trend_20
    - rsi_14, volume_spike, sma_20, sma_50, sma_crossover
    - overall_score, label, explanation
    """
    data = fetch_price_data(ticker)
    if data.empty or len(data) < 30:
        return {"error": "Not enough data"}

    close = data["Close"]
    volume = data["Volume"]

    # Today vs previous close
    today_change = safe_pct_change(close.iloc[-1], close.iloc[-2])

    # 5-day change
    if len(close) >= 6:
        five_day_change = safe_pct_change(close.iloc[-1], close.iloc[-6])
    else:
        five_day_change = float("nan")

    # 20-day trend via linear regression on last 20 closes
    if len(close) >= 20:
        recent_20 = close.iloc[-20:]
        x = np.arange(len(recent_20))
        slope, _ = np.polyfit(x, recent_20.values, 1)
        # normalize slope to % over 20 days
        trend_20 = (slope * 20 / recent_20.iloc[-1]) * 100.0
    else:
        trend_20 = float("nan")

    # RSI(14)
    rsi_14 = compute_rsi(close, period=14)

    # Volume spike: today's volume vs 20-day average
    if len(volume) >= 21:
        # force everything to plain floats so pandas can't be weird
        vol_avg = float(volume.iloc[-21:-1].mean())
        last_vol = float(volume.iloc[-1])

        if np.isnan(vol_avg) or vol_avg <= 0:
            vol_spike = float("nan")
        else:
            vol_spike = last_vol / vol_avg
    else:
        vol_spike = float("nan")

        # SMA crossover
    if len(close) >= 20:
        sma_20 = float(close.rolling(20).mean().iloc[-1])
    else:
        sma_20 = float("nan")

    if len(close) >= 50:
        sma_50 = float(close.rolling(50).mean().iloc[-1])
    else:
        sma_50 = float("nan")

    if not np.isnan(sma_20) and not np.isnan(sma_50):
        if sma_20 > sma_50:
            sma_crossover = "Bullish (20 > 50)"
        elif sma_20 < sma_50:
            sma_crossover = "Bearish (20 < 50)"
        else:
            sma_crossover = "Neutral"
    else:
        sma_crossover = "Not enough data"

    # ----------------- SCORING MODEL -----------------
    # Normalize each factor into 0–100, then average with weights
    # These thresholds are intentionally moderate so not everything is "Neutral".

    def clamp(x, lo, hi):
        return max(lo, min(hi, x))

    # Today change: -4% to +4% → 0–100
    if not np.isnan(today_change):
        today_score = clamp((today_change + 4) / 8 * 100, 0, 100)
    else:
        today_score = 50

    # 5-day change: -10% to +10% → 0–100
    if not np.isnan(five_day_change):
        five_score = clamp((five_day_change + 10) / 20 * 100, 0, 100)
    else:
        five_score = 50

    # 20-day trend: -15% to +15% → 0–100
    if not np.isnan(trend_20):
        trend_score = clamp((trend_20 + 15) / 30 * 100, 0, 100)
    else:
        trend_score = 50

    # RSI: classic 30–70 range
    if not np.isnan(rsi_14):
        if rsi_14 < 30:
            rsi_score = 70  # oversold – bullish tilt
        elif rsi_14 > 70:
            rsi_score = 30  # overbought – bearish tilt
        else:
            # map 30–70 to 60–40 (neutral-ish)
            rsi_score = 60 - ((rsi_14 - 30) / 40) * 20
    else:
        rsi_score = 50

    # Volume spike: 0.5x–3x → 0–100
    if not np.isnan(vol_spike):
        vol_score = clamp((vol_spike - 0.5) / 2.5 * 100, 0, 100)
    else:
        vol_score = 50

    # Weights
    w_today = 0.25
    w_five = 0.2
    w_trend = 0.25
    w_rsi = 0.15
    w_vol = 0.15

    overall_score = (
        today_score * w_today
        + five_score * w_five
        + trend_score * w_trend
        + rsi_score * w_rsi
        + vol_score * w_vol
    )

    label = simple_label_from_score(overall_score)

    # ----------------- EXPLANATION -----------------
    explanation_parts = []

    explanation_parts.append(
        f"Today change: **{today_change:.2f}%**; 5-day change: **{five_day_change:.2f}%**."
    )
    explanation_parts.append(
        f"20-day price trend (regression): **{trend_20:.2f}%** over the last 20 sessions."
    )
    explanation_parts.append(f"RSI(14): **{rsi_14:.1f}**.")
    explanation_parts.append(
        f"Volume spike factor: **{vol_spike:.2f}x** vs 20-day average."
    )
    explanation_parts.append(f"SMA status: **{sma_crossover}**.")

    explanation_parts.append(
        f"\nOverall signal strength: **{overall_score:.1f} / 100** → **{label}**."
    )

    explanation = "  \n".join(explanation_parts)

    return {
        "today_change": today_change,
        "five_day_change": five_day_change,
        "trend_20": trend_20,
        "rsi_14": rsi_14,
        "vol_spike": vol_spike,
        "sma_20": sma_20,
        "sma_50": sma_50,
        "sma_crossover": sma_crossover,
        "overall_score": overall_score,
        "label": label,
        "explanation": explanation,
    }


    label = simple_label_from_score(overall_score)

    # ----------------- EXPLANATION -----------------
    explanation_parts = []

    explanation_parts.append(
        f"Today change: **{today_change:.2f}%**; 5-day change: **{five_day_change:.2f}%**."
    )
    explanation_parts.append(
        f"20-day price trend (regression): **{trend_20:.2f}%** over the last 20 sessions."
    )
    explanation_parts.append(f"RSI(14): **{rsi_14:.1f}**.")
    explanation_parts.append(
        f"Volume spike factor: **{vol_spike:.2f}x** vs 20-day average."
    )
    explanation_parts.append(f"SMA status: **{sma_crossover}**.")

    explanation_parts.append(
        f"\nOverall signal strength: **{overall_score:.1f} / 100** → **{label}**."
    )

    explanation = "  \n".join(explanation_parts)

    return {
        "today_change": today_change,
        "five_day_change": five_day_change,
        "trend_20": trend_20,
        "rsi_14": rsi_14,
        "vol_spike": vol_spike,
        "sma_20": sma_20,
        "sma_50": sma_50,
        "sma_crossover": sma_crossover,
        "overall_score": overall_score,
        "label": label,
        "explanation": explanation,
    }


# ----------------- UI: SIDEBAR -----------------
st.sidebar.title("📊 Stock Analyzer MVP")
page = st.sidebar.radio("Navigation", ["Analyzer", "Analytics"])

st.sidebar.markdown("---")
st.sidebar.caption("v0.2 – Momentum + RSI + Volume spikes")


# ----------------- PAGE 1: ANALYZER -----------------
if page == "Analyzer":
    st.title("📈 Stock Analyzer")
    st.write(
        "Enter one or more tickers like `AAPL, TSLA, NVDA` and click **Analyze** "
        "to get a multi-factor momentum signal with a 0–100 strength score."
    )

    default_tickers = "AAPL, TSLA, NVDA"
    tickers_input = st.text_input("Tickers", default_tickers)

    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        analyze_clicked = st.button("🔍 Analyze")

    if analyze_clicked:
        tickers_str = tickers_input.strip()
        st.session_state["last_query"] = tickers_str

        # Log event
        log_event("analyze_clicked", tickers_str)

        tickers = [
            t.strip().upper()
            for t in tickers_str.replace(" ", "").split(",")
            if t.strip()
        ]

        results = []
        for t in tickers:
            with st.spinner(f"Analyzing {t}..."):
                info = compute_indicators_for_ticker(t)
                row = {"Ticker": t}
                if "error" in info:
                    row["Error"] = info["error"]
                else:
                    row["Signal Strength"] = round(info["overall_score"], 1)
                    row["Label"] = info["label"]
                    row["Today %"] = round(info["today_change"], 2)
                    row["5-Day %"] = round(info["five_day_change"], 2)
                    row["20-Day Trend %"] = round(info["trend_20"], 2)
                    row["RSI(14)"] = round(info["rsi_14"], 1)
                    row["Volume Spike (x)"] = round(info["vol_spike"], 2)
                    row["SMA Crossover"] = info["sma_crossover"]
                    row["Explanation"] = info["explanation"]
                results.append(row)

        st.session_state["results"] = results

    # Display results if present
    if st.session_state["results"] is not None:
        results = st.session_state["results"]

        # Summary cards at top: one card per ticker
        st.markdown("### 🔦 Summary Signals")
        for row in results:
            if "Error" in row:
                with st.container():
                    st.error(f"{row['Ticker']}: {row['Error']}")
                continue

            label = row["Label"]
            emoji_label = label_with_emoji(label)

            col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
            with col1:
                st.markdown(f"**{row['Ticker']}**")
                st.markdown(f"### {emoji_label}")
                st.markdown(
                    f"**Signal Strength:** {row['Signal Strength']:.1f} / 100"
                )
            with col2:
                st.metric("Today %", f"{row['Today %']:.2f}%")
                st.metric("5-Day %", f"{row['5-Day %']:.2f}%")
            with col3:
                st.metric("20-Day Trend %", f"{row['20-Day Trend %']:.2f}%")
                st.metric("RSI(14)", f"{row['RSI(14)']:.1f}")
            with col4:
                st.metric("Volume Spike (x)", f"{row['Volume Spike (x)']:.2f}")
                st.markdown(f"**SMA:** {row['SMA Crossover']}")

            with st.expander("View explanation"):
                st.markdown(row["Explanation"])

            st.markdown("---")

        # Table view
        df_rows = [r for r in results if "Error" not in r]
        if df_rows:
            table_cols = [
                "Ticker",
                "Signal Strength",
                "Label",
                "Today %",
                "5-Day %",
                "20-Day Trend %",
                "RSI(14)",
                "Volume Spike (x)",
                "SMA Crossover",
            ]
            df = pd.DataFrame(df_rows)[table_cols]
            st.markdown("### 📋 Table View")
            st.dataframe(df, use_container_width=True)

# ----------------- PAGE 2: ANALYTICS -----------------
elif page == "Analytics":
    st.title("📊 Usage Analytics")

    if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
        st.info("No usage data yet. Run some analyses on the **Analyzer** page first.")
    else:
        # Read log file – some old rows may be malformed, so be defensive
        df_log = pd.read_csv(
            LOG_FILE,
            sep="|",
            header=None,
            names=["timestamp", "event_type", "tickers"],
        )

        # Drop totally empty rows
        df_log = df_log.dropna(how="all")

        # Robust datetime parsing
        df_log["timestamp"] = pd.to_datetime(
            df_log["timestamp"], errors="coerce"
        )
        df_log = df_log.dropna(subset=["timestamp"])
        df_log["date"] = df_log["timestamp"].dt.date

        if df_log.empty:
            st.info("No valid usage events yet.")
        else:
            # Basic stats
            st.markdown("### Overview")
            total_events = len(df_log)
            total_analyzes = (df_log["event_type"] == "analyze_clicked").sum()

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Total logged events", total_events)
            with col_b:
                st.metric("Total analyses run", total_analyzes)

            # Events per day
            per_day = df_log.groupby("date").size().reset_index(name="events")

            st.markdown("### Activity Over Time")
            st.line_chart(per_day.set_index("date"))

            # Most popular tickers
            def split_tickers(x):
                if pd.isna(x):
                    return []
                return [t.strip().upper() for t in str(x).split(",") if t.strip()]

            exploded = df_log["tickers"].dropna().apply(split_tickers)
            tickers_flat = [t for sub in exploded for t in sub]
            if tickers_flat:
                df_t = pd.Series(tickers_flat).value_counts().reset_index()
                df_t.columns = ["Ticker", "Count"]
                st.markdown("### Most Analyzed Tickers")
                st.bar_chart(df_t.set_index("Ticker"))
            else:
                st.info("No tickers recorded yet.")







