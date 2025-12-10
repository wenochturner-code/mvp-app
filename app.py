import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime   # <-- added for logging

st.set_page_config(page_title="Stock Analyzer MVP", page_icon="📈")

st.title("Stock Analyzer MVP")

# ---- Session state for results / query ----
if "results" not in st.session_state:
    st.session_state["results"] = None
if "last_query" not in st.session_state:
    st.session_state["last_query"] = ""
if "pending_query" not in st.session_state:
    st.session_state["pending_query"] = ""

# ---------- Simple usage logging ----------

def log_event(event_type: str, tickers: str):
    """
    Lightweight event logger.
    Writes events to a local CSV file for usage analytics.
    """
    try:
        ts = datetime.utcnow().isoformat()
        row = f"{ts},{event_type},{tickers}\n"
        with open("events_log.csv", "a", encoding="utf-8") as f:
            f.write(row)
    except Exception:
        pass  # never break the app if logging fails

# ---------- Small helper utilities ----------

def label_with_emoji(label: str) -> str:
    if label == "Bullish":
        return "🟢 Bullish"
    if label == "Bearish":
        return "🔴 Bearish"
    return "⚪ Neutral"


def classify_risk(vol_factor: float) -> str:
    if vol_factor <= 0.8:
        return "Low"
    elif vol_factor <= 1.5:
        return "Medium"
        return "High"

def classify_timeframe(today_change: float, five_day_change: float, trend_20: float) -> str:
    if abs(today_change) > 2.0 and abs(trend_20) < 4.0:
        return "Short-term (1–3 days)"
    if abs(five_day_change) > 3.0 and (five_day_change * trend_20) > 0:
        return "Swing (3–10 days)"
    if abs(trend_20) > 5.0 and abs(today_change) < 2.0:
        return "Trend / Position"
    return "Mixed / Unclear"


def classify_setup_and_profile(
    signal: str,
    today_change: float,
    five_day_change: float,
    trend_20: float,
    vol_factor: float,
    risk: str,
    timeframe: str,
):
    setup = "Mixed / noisy"

    if abs(today_change) > 3 and abs(trend_20) < 4:
        setup = "Short-term spike"

    if abs(five_day_change) > 4 and (five_day_change * trend_20) > 0:
        if signal == "Bullish":
            setup = "Momentum continuation"
        elif signal == "Bearish":
            setup = "Downtrend continuation"

    if abs(trend_20) > 6 and abs(today_change) < 1.5 and abs(five_day_change) < 3:
        setup = "Steady trend"

    if abs(today_change) < 0.8 and abs(five_day_change) < 2 and abs(trend_20) < 4:
        setup = "Sideways / consolidation"

    best_for = "Watchlist only"

    if setup in ["Short-term spike", "Momentum continuation"] and risk in ["Medium", "High"]:
        best_for = "Aggressive short-term traders"
    elif setup in ["Momentum continuation", "Steady trend"] and risk in ["Low", "Medium"]:
        if "Swing" in timeframe:
            best_for = "Swing traders"
        elif "Trend" in timeframe or "Position" in timeframe:
            best_for = "Trend / position holders"
        else:
            best_for = "Active swing traders"
    elif setup in ["Sideways / consolidation"]:
        best_for = "Range / mean-reversion traders"

    return setup, best_for


# ---------- Signal engine ----------

def compute_signal_and_explanation(
    ticker: str, today_change: float, five_day_change: float, trend_20: float, vol_factor: float
):
    def safe(x, default=0.0):
        try:
            if x is None:
                return default
            if isinstance(x, float) and (x != x):
                return default
            return float(x)
        except Exception:
            return default

    today_change = safe(today_change)
    five_day_change = safe(five_day_change)
    trend_20 = safe(trend_20)
    vol_factor = max(safe(vol_factor, 1.0), 0.01)

    def clamp(x, lo, hi):
        return max(lo, min(hi, x))

    t1_norm = clamp(today_change, -4, 4) / 4.0
    t5_norm = clamp(five_day_change, -10, 10) / 10.0
    t20_norm = clamp(trend_20, -20, 20) / 20.0

    score = 0.20 * t1_norm + 0.45 * t5_norm + 0.35 * t20_norm

    bullish_threshold = 0.25
    bearish_threshold = -0.25

    if score >= bullish_threshold:
        signal = "Bullish"
    elif score <= bearish_threshold:
        signal = "Bearish"
    else:
        signal = "Neutral"

    def dir_from_val(v, eps=0.005):
        if v > eps: return "up"
        if v < -eps: return "down"
        return "flat"

    directions = [dir_from_val(today_change), dir_from_val(five_day_change), dir_from_val(trend_20)]
    up_count = directions.count("up")
    down_count = directions.count("down")

    abs_score = abs(score)
    if abs_score >= 0.50:
        base_conf = "High"
    elif abs_score >= 0.25:
        base_conf = "Medium"
    else:
        base_conf = "Low"

    if signal == "Bullish":
        agreement = up_count
    elif signal == "Bearish":
        agreement = down_count
    else:
        agreement = max(up_count, down_count)

    if agreement >= 3 and abs_score >= 0.4:
        confidence = "High"
    elif agreement >= 2 and abs_score >= 0.25:
        confidence = "Medium"
    else:
        confidence = base_conf

    if vol_factor > 1.4:
        vol_note = "Volatility is elevated, so expect bigger swings."
    elif vol_factor < 0.7:
        vol_note = "Price moves have been relatively calm."
    else:
        vol_note = "Volatility is in a normal range."

    def fmt_pct(x):
        return f"{x:+.1f}%"

    trend_summary = f"Today: {fmt_pct(today_change)}, 5-day: {fmt_pct(five_day_change)}, 20-day: {fmt_pct(trend_20)}"

    if signal == "Bullish":
        reason = f"{ticker} is showing a **Bullish** trend overall."
    elif signal == "Bearish":
        reason = f"{ticker} is showing a **Bearish** trend overall."
    else:
        reason = f"{ticker} looks **Neutral** right now."

    explanation = f"{reason} Recent performance → {trend_summary}. {vol_note} (Score: {score:+.2f}, confidence: {confidence})"

    return signal, confidence, explanation, score


# ---------- Analysis function ----------

def run_analysis(ticker_string: str):
    tickers = [t.strip().upper() for t in ticker_string.split(",") if t.strip()]

    if not tickers:
        st.warning("Please enter at least one ticker symbol.")
        return None

    # Log the scan
    log_event("scan", ticker_string)

    results = []

    with st.spinner("Fetching data and computing signals..."):
        for ticker in tickers:
            try:
                data = yf.Ticker(ticker).history(period="30d")
                if data.empty or len(data) < 10:
                    st.warning(f"{ticker}: Not enough data.")
                    continue

                closes = data["Close"]
                latest = closes.iloc[-1]
                prev = closes.iloc[-2]
                today_change = (latest / prev - 1) * 100

                five_day_change = (
                    (latest / closes.iloc[-6] - 1) * 100
                    if len(closes) >= 6 else 0
                )

                trend_20 = (
                    (latest / closes.iloc[-21] - 1) * 100
                    if len(closes) >= 21 else 0
                )

                returns = closes.pct_change().dropna()
                daily_vol_pct = (returns.iloc[-20:].std() * 100) if len(returns) >= 10 else 0
                vol_factor = daily_vol_pct / 2.0

                signal, confidence, explanation, score = compute_signal_and_explanation(
                    ticker, today_change, five_day_change, trend_20, vol_factor
                )

                risk = classify_risk(vol_factor)
                timeframe = classify_timeframe(today_change, five_day_change, trend_20)
                setup, best_for = classify_setup_and_profile(
                    signal, today_change, five_day_change, trend_20, vol_factor, risk, timeframe
                )

                results.append({
                    "Ticker": ticker,
                    "Signal": signal,
                    "Confidence": confidence,
                    "Score": score,
                    "Today %": today_change,
                    "5-day %": five_day_change,
                    "20-day %": trend_20,
                    "Vol factor": vol_factor,
                    "Risk": risk,
                    "Timeframe": timeframe,
                    "Setup": setup,
                    "Best for": best_for,
                    "Explanation": explanation,
                })

            except Exception as e:
                st.error(f"Error analyzing {ticker}: {e}")

    return sorted(results, key=lambda r: r["Score"], reverse=True)


# ---------- Watchlist / trending ----------

TRENDING_TICKERS = ["AAPL", "NVDA", "TSLA", "META", "AVGO", "SMCI", "SPY", "QQQ"]
UNIVERSE_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "META", "AMZN", "GOOGL",
                    "AVGO", "SMCI", "SPY", "QQQ", "NFLX", "AMD", "INTC"]

def get_top_bullish_setups(limit=5):
    log_event("scan_watchlist", ",".join(UNIVERSE_TICKERS))
    results = run_analysis(",".join(UNIVERSE_TICKERS))
    if not results:
        return []
    bullish = [r for r in results if r["Signal"] == "Bullish"]
    return sorted(bullish, key=lambda r: r["Score"], reverse=True)[:limit]


# ---------- UI logic ----------

if st.session_state["results"] is None:
    # HERO MODE
    st.info("Beta – experiment screener using 1/5/20-day momentum and volatility.")

    default_query = st.session_state["pending_query"] or "AAPL, TSLA, NVDA"

    with st.form("initial_search"):
        tickers_input = st.text_input("Tickers", default_query)
        submitted = st.form_submit_button("Analyze")

    st.markdown("##### Or tap a trending ticker")
    clicked_ticker = None
    cols = st.columns(4)
    for i, t in enumerate(TRENDING_TICKERS):
        if cols[i % 4].button(t, key=f"trend_{t}"):
            clicked_ticker = t
            log_event("trending_click", t)

    # Watchlist section
    with st.expander("Today's top bullish setups (watchlist)"):
        top_setups = get_top_bullish_setups()
        for row in top_setups:
            st.markdown(
                f"- **{row['Ticker']}** – {row['Signal']} · Score: {row['Score']:.2f} · {row['Timeframe']} · {row['Setup']}"
            )

    # Manual search submit
    if submitted:
        log_event("manual_search", tickers_input)
        results_sorted = run_analysis(tickers_input)
        if results_sorted is not None:
            st.session_state["results"] = results_sorted
            st.session_state["last_query"] = tickers_input
            st.session_state["pending_query"] = ""
            st.rerun()

    # Trending click submit
    if clicked_ticker:
        results_sorted = run_analysis(clicked_ticker)
        if results_sorted is not None:
            st.session_state["results"] = results_sorted
            st.session_state["last_query"] = clicked_ticker
            st.session_state["pending_query"] = ""
            st.rerun()

else:
    # RESULTS MODE
    query = st.session_state["last_query"]
    st.markdown(f"#### Results for: `{query}`")

    col1, col2 = st.columns(2)
    if col1.button("New search"):
        st.session_state["pending_query"] = st.session_state["last_query"]
        st.session_state["results"] = None
        st.rerun()
    if col2.button("Clear & go back"):
        st.session_state["pending_query"] = ""
        st.session_state["last_query"] = ""
        st.session_state["results"] = None
        st.rerun()

    st.write("---")


# ---------- Render results ----------

if st.session_state["results"] is not None:
    results_sorted = st.session_state["results"]

    st.subheader("Summary")
    df = pd.DataFrame(results_sorted)[
        ["Ticker", "Signal", "Confidence", "Risk", "Timeframe", "Score"]
    ]
    df["Score"] = df["Score"].round(2)
    st.dataframe(df, use_container_width=True)

    st.write("---")
    st.subheader("Chart Brain Read")

    for row in results_sorted:
        st.markdown(f"### {row['Ticker']} – {label_with_emoji(row['Signal'])}")
        st.markdown(
            f"**Score:** {row['Score']:.2f} · **Confidence:** {row['Confidence']} · "
            f"**Risk:** {row['Risk']} · **Timeframe:** {row['Timeframe']}"
        )
        st.markdown(
            f"• Today: {row['Today %']:+.2f}% | 5-day: {row['5-day %']:+.2f}% | "
            f"20-day: {row['20-day %']:+.2f}% | Vol: {row['Vol factor']:.2f}"
        )
        st.markdown(f"**Setup:** {row['Setup']} · **Best for:** {row['Best for']}")
        st.caption(row["Explanation"])
        st.write("---")


# ---------- Footer ----------
st.write("---")
st.caption(
    "Disclaimer: This tool provides automated market analysis for educational "
    "purposes only and is not financial advice."
)

# ---------- Admin / debug panel ----------
with st.expander("📊 Admin: View usage log"):
    try:
        df_log = pd.read_csv("events_log.csv", header=None, names=["timestamp", "event", "tickers"])
        st.dataframe(df_log, use_container_width=True)
    except FileNotFoundError:
        st.caption("No log file found yet.")





