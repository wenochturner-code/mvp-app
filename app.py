import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime

# ---------- Page config ----------
st.set_page_config(page_title="Friendly Ticker", page_icon="📊")
st.title("📊 Friendly Ticker")

st.write(
    "Type a stock symbol like **AAPL**, **TSLA**, or **NVDA** and Friendly Ticker will instantly explain the trend, "
    "risk level, and recent movement in simple, beginner-friendly language. No charts. No jargon."
)

# ---------- Session state ----------
if "results" not in st.session_state:
    st.session_state["results"] = None

# ---------- Simple usage logging ----------
def log_event(event_type: str, tickers: str):
    """Append simple usage events to a local CSV-style log file."""
    try:
        ts = datetime.utcnow().isoformat()
        row = f"{ts}|{event_type}|{tickers}\n"
        with open("events_log.csv", "a", encoding="utf-8") as f:
            f.write(row)
    except Exception:
        # Logging should never break the app
        pass


# ---------- Helper functions for labels / text ----------

def label_from_score(score: int) -> str:
    if score >= 70:
        return "🟢 Strong Uptrend"
    if score >= 40:
        return "⚪ Sideways / Mixed"
    return "🔴 Weak / Downtrend"


def risk_label_from_vol(vol_pct: float) -> str:
    """Rough risk label based on daily volatility (std dev of daily returns in %)."""
    if vol_pct < 1.2:
        return "🟢 Low – moves are usually small"
    if vol_pct < 2.5:
        return "⚪ Medium – normal ups and downs"
    return "🔴 High – price swings can be big"


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def compute_scores(df: pd.DataFrame) -> dict | None:
    """Compute all numeric scores needed for the beginner view.
    Returns a dict or None if we don't have enough data.
    """
    if df is None or df.empty or len(df) < 25:
        return None

    close = df["Close"]
    last = close.iloc[-1]
    prev = close.iloc[-2]

    # Today % change
    today_change_pct = (last - prev) / prev * 100

    # 5-day change (use 5 trading days ago)
    if len(close) >= 6:
        five_ago = close.iloc[-6]
        five_day_change_pct = (last - five_ago) / five_ago * 100
    else:
        five_day_change_pct = 0.0

    # 20-day trend vs 20-day average
    if len(close) >= 20:
        sma20 = close.tail(20).mean()
        trend_20_pct = (last - sma20) / sma20 * 100
    else:
        trend_20_pct = 0.0

    # Volatility (daily std dev in %)
    daily_returns = close.pct_change().dropna()
    vol_pct = float(daily_returns.std() * 100.0)

    # ---- Convert raw metrics to 0–100 scores ----
    # Today score: clamp -4% to +4% into 0–100
    today_clamped = clamp(today_change_pct, -4.0, 4.0)
    today_score = int(round((today_clamped + 4.0) / 8.0 * 100.0))

    # 5-day score: clamp -12% to +12% into 0–100
    five_clamped = clamp(five_day_change_pct, -12.0, 12.0)
    five_score = int(round((five_clamped + 12.0) / 24.0 * 100.0))

    # 20-day trend score: clamp -20% to +20% into 0–100
    trend_clamped = clamp(trend_20_pct, -20.0, 20.0)
    trend_score = int(round((trend_clamped + 20.0) / 40.0 * 100.0))

    # Volatility score: medium volatility (around 1.5–2.5%) is “best” for beginners.
    # Penalize how far we are from 2% daily std dev.
    ideal_vol = 2.0
    vol_diff = abs(vol_pct - ideal_vol)
    # If vol_diff is 0 → 100, if vol_diff is 4 or more → 0
    vol_score = int(round(clamp(100.0 - (vol_diff / 4.0) * 100.0, 0.0, 100.0)))

    # Overall score: weighted average
    overall_score = int(
        round(
            0.4 * trend_score
            + 0.25 * five_score
            + 0.2 * today_score
            + 0.15 * vol_score
        )
    )

    return {
        "price": float(last),
        "today_change_pct": float(today_change_pct),
        "five_day_change_pct": float(five_day_change_pct),
        "trend_20_pct": float(trend_20_pct),
        "vol_pct": float(vol_pct),
        "today_score": today_score,
        "five_score": five_score,
        "trend_score": trend_score,
        "vol_score": vol_score,
        "overall_score": overall_score,
    }


def beginner_summary(ticker: str, s: dict) -> str:
    """Plain-English explanation for beginners based on the scores."""
    direction = label_from_score(s["overall_score"])

    # Direction fragment
    if s["overall_score"] >= 70:
        dir_text = (
            f"{ticker} has been in a **strong overall uptrend** recently. "
            "Both the short-term and 20-day trend look healthy."
        )
    elif s["overall_score"] >= 40:
        dir_text = (
            f"{ticker} is **moving more sideways** right now. "
            "Some days are up, some are down, and there isn't a super clear direction yet."
        )
    else:
        dir_text = (
            f"{ticker} has been **weak or trending down** lately. "
            "Recent price action has leaned more negative than positive."
        )

    # Volatility fragment
    if s["vol_pct"] < 1.2:
        vol_text = (
            "The stock's daily moves are usually **small**, which can feel calmer for beginners."
        )
    elif s["vol_pct"] < 2.5:
        vol_text = (
            "The day-to-day moves are **normal-sized** — not too calm, not too wild."
        )
    else:
        vol_text = (
            "Price swings are **pretty large**, so it's normal to see sharp up and down days. "
            "Beginners should be ready for bigger moves."
        )

    # Today fragment
    if s["today_change_pct"] > 0.8:
        today_text = "Today is a **solid green day**, which adds to the bullish side."
    elif s["today_change_pct"] < -0.8:
        today_text = "Today is a **red day**, which pulls the short-term mood down a bit."
    else:
        today_text = "Today is fairly **flat**, so the bigger trend matters more than just today."

    return (
        f"{direction}  \n\n"
        f"{dir_text}  \n\n"
        f"{vol_text}  \n\n"
        f"{today_text}  \n\n"
        "This is **not financial advice** — it just describes how the price has been behaving lately in simple terms."
    )


# ---------- UI: Input ----------
tickers_input = st.text_input(
    "Stock symbol (or multiple, separated by commas)",
    "AAPL",
    help="Type one or more tickers like AAPL, TSLA, NVDA",
)

analyze_clicked = st.button("Analyze")

if analyze_clicked:
    cleaned = tickers_input.upper().replace(" ", "")
    log_event("analyze", cleaned)
    tickers = [t for t in cleaned.split(",") if t]

    if not tickers:
        st.warning("Please enter at least one valid ticker symbol.")
    else:
        all_results = {}
        for t in tickers:
            try:
                df = yf.Ticker(t).history(period="6mo")
                scores = compute_scores(df)
                all_results[t] = scores
            except Exception:
                all_results[t] = None
        st.session_state["results"] = all_results

results = st.session_state.get("results")

# ---------- UI: Results ----------
if results:
    for ticker, scores in results.items():
        st.markdown("---")
        st.subheader(f"📈 {ticker}")
        if scores is None:
            st.write(
                "Couldn't load enough data for this symbol. It might be invalid or too new."
            )
            continue

        overall = scores["overall_score"]
        today_pct = scores["today_change_pct"]
        five_pct = scores["five_day_change_pct"]
        trend20 = scores["trend_20_pct"]
        vol_pct = scores["vol_pct"]

        # Overall trend block
        st.markdown("### Overall Trend (for beginners)")
        col_main, col_label = st.columns([3, 2])
        with col_main:
            st.metric(
                label="Trend Score (0–100)",
                value=f"{overall}",
            )
            st.progress(overall / 100.0)
        with col_label:
            st.write("**Signal:**", label_from_score(overall))
            st.write("**Risk:**", risk_label_from_vol(vol_pct))

        # Key numbers
        st.markdown("### Key Recent Moves")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                label="Today",
                value=f"{today_pct:+.2f}%",
            )
        with c2:
            st.metric(
                label="Last 5 trading days",
                value=f"{five_pct:+.2f}%",
            )
        with c3:
            st.metric(
                label="20-day vs average",
                value=f"{trend20:+.2f}%",
            )

        st.caption(
            "Positive percentages mean the stock has gone up over that time. Negative means it has gone down."
        )

        # Beginner summary
        st.markdown("### Beginner-Friendly Summary")
        st.write(beginner_summary(ticker, scores))


