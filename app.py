import streamlit as st
import pandas as pd
import yfinance as yf

st.set_page_config(page_title="Stock Analyzer MVP", page_icon="📈")

st.title("Stock Analyzer MVP")

# ---- Session state for results / query ----
if "results" not in st.session_state:
    st.session_state["results"] = None
if "last_query" not in st.session_state:
    st.session_state["last_query"] = ""
if "pending_query" not in st.session_state:
    st.session_state["pending_query"] = ""

# ---------- Small helper utilities ----------

def label_with_emoji(label: str) -> str:
    if label == "Bullish":
        return "🟢 Bullish"
    if label == "Bearish":
        return "🔴 Bearish"
    return "⚪ Neutral"


def classify_risk(vol_factor: float) -> str:
    """
    Simple risk tag based on relative volatility.
    vol_factor ~1 = normal, >1.5 = spicy, <0.8 = calm.
    """
    if vol_factor <= 0.8:
        return "Low"
    elif vol_factor <= 1.5:
        return "Medium"
    else:
        return "High"


def classify_timeframe(today_change: float, five_day_change: float, trend_20: float) -> str:
    """
    Rough guess of what timeframe the setup looks best for.
    This is just heuristics for now (ML can replace later).
    """
    # Strong short pop, not much 20d trend → day/small swing
    if abs(today_change) > 2.0 and abs(trend_20) < 4.0:
        return "Short-term (1–3 days)"

    # 5d and 20d pointing same way with some size → swing
    if abs(five_day_change) > 3.0 and (five_day_change * trend_20) > 0:
        return "Swing (3–10 days)"

    # Modest 20d but not crazy short-term moves → longer hold
    if abs(trend_20) > 5.0 and abs(today_change) < 2.0:
        return "Trend / Position"

    return "Mixed / Unclear"


# ---------- Signal engine (chart brain v0) ----------

def compute_signal_and_explanation(
    ticker: str,
    today_change: float,
    five_day_change: float,
    trend_20: float,
    vol_factor: float,
):
    """
    Returns:
        signal: 'Bullish' | 'Bearish' | 'Neutral'
        confidence: 'High' | 'Medium' | 'Low'
        explanation: str (AI-style "chart read" text)
        score: float  # overall numeric score for debugging / sorting
    """

    # Defensive defaults if any value is None / NaN
    def safe(x, default=0.0):
        try:
            if x is None:
                return default
            if isinstance(x, float) and (x != x):  # NaN check
                return default
            return float(x)
        except Exception:
            return default

    today_change = safe(today_change)
    five_day_change = safe(five_day_change)
    trend_20 = safe(trend_20)
    vol_factor = max(safe(vol_factor, 1.0), 0.01)

    # --- Normalize & weight factors ---
    def clamp(x, lo, hi):
        return max(lo, min(hi, x))

    t1_norm = clamp(today_change, -4, 4) / 4.0        # -1 .. 1
    t5_norm = clamp(five_day_change, -10, 10) / 10.0  # -1 .. 1
    t20_norm = clamp(trend_20, -20, 20) / 20.0        # -1 .. 1

    score = (
        0.20 * t1_norm +
        0.45 * t5_norm +
        0.35 * t20_norm
    )

    # --- Determine signal (direction) ---
    bullish_threshold = 0.25
    bearish_threshold = -0.25

    if score >= bullish_threshold:
        signal = "Bullish"
    elif score <= bearish_threshold:
        signal = "Bearish"
    else:
        signal = "Neutral"

    # --- Agreement between factors helps confidence ---
    def dir_from_val(v, eps=0.005):
        if v > eps:
            return "up"
        if v < -eps:
            return "down"
        return "flat"

    d1 = dir_from_val(today_change)
    d5 = dir_from_val(five_day_change)
    d20 = dir_from_val(trend_20)
    directions = [d1, d5, d20]

    up_count = directions.count("up")
    down_count = directions.count("down")

    # --- Base confidence from score strength ---
    abs_score = abs(score)
    if abs_score >= 0.50:
        base_conf = "High"
    elif abs_score >= 0.25:
        base_conf = "Medium"
    else:
        base_conf = "Low"

    # --- Adjust confidence with agreement between timeframes ---
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

    # Volatility note
    if vol_factor > 1.4:
        vol_note = "Volatility is elevated, so expect bigger swings."
    elif vol_factor < 0.7:
        vol_note = "Price moves have been relatively calm."
    else:
        vol_note = "Volatility is in a normal range."

    def fmt_pct(x):
        return f"{x:+.1f}%"

    trend_bits = [
        f"Today: {fmt_pct(today_change)}",
        f"5-day: {fmt_pct(five_day_change)}",
        f"20-day: {fmt_pct(trend_20)}",
    ]
    trend_summary = ", ".join(trend_bits)

    # High-level “chart brain” explanation
    if signal == "Bullish":
        reason = (
            f"{ticker} is showing a **Bullish** trend overall. "
            f"Short- and medium-term momentum lean to the upside."
        )
    elif signal == "Bearish":
        reason = (
            f"{ticker} is showing a **Bearish** trend overall. "
            f"Short- and medium-term momentum lean to the downside."
        )
    else:
        reason = (
            f"{ticker} looks **Neutral** right now. "
            f"Recent moves don’t strongly favor bulls or bears."
        )

    explanation = (
        f"{reason} "
        f"Recent performance → {trend_summary}. "
        f"{vol_note} "
        f"(Internal score: {score:+.2f}, confidence: {confidence}.)"
    )

    return signal, confidence, explanation, score


# ---------- Analysis function ----------

def run_analysis(ticker_string: str):
    raw_tickers = [t.strip().upper() for t in ticker_string.split(",")]
    tickers = [t for t in raw_tickers if t]

    if not tickers:
        st.warning("Please enter at least one ticker symbol.")
        return None

    results = []

    with st.spinner("Fetching data and computing signals..."):
        for ticker in tickers:
            try:
                data = yf.Ticker(ticker).history(period="30d")

                if data.empty or len(data) < 10:
                    st.warning(f"{ticker}: Not enough data to analyze.")
                    continue

                closes = data["Close"]

                latest = closes.iloc[-1]
                prev = closes.iloc[-2]
                today_change = (latest / prev - 1.0) * 100.0

                if len(closes) >= 6:
                    five_day_base = closes.iloc[-6]
                    five_day_change = (latest / five_day_base - 1.0) * 100.0
                else:
                    five_day_change = 0.0

                if len(closes) >= 21:
                    twenty_day_base = closes.iloc[-21]
                    trend_20 = (latest / twenty_day_base - 1.0) * 100.0
                else:
                    trend_20 = 0.0

                returns = closes.pct_change().dropna()
                if len(returns) >= 10:
                    recent = returns.iloc[-20:] if len(returns) >= 20 else returns
                    daily_vol_pct = recent.std() * 100.0
                else:
                    daily_vol_pct = 0.0

                baseline_vol = 2.0
                vol_factor = (daily_vol_pct / baseline_vol) if baseline_vol > 0 else 1.0

                signal, confidence, explanation, score = compute_signal_and_explanation(
                    ticker,
                    today_change,
                    five_day_change,
                    trend_20,
                    vol_factor,
                )

                risk = classify_risk(vol_factor)
                timeframe = classify_timeframe(today_change, five_day_change, trend_20)

                results.append(
                    {
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
                        "Explanation": explanation,
                    }
                )

            except Exception as e:
                st.error(f"Error analyzing {ticker}: {e}")

    if not results:
        st.info("No valid results to show yet.")
        return None

    return sorted(results, key=lambda x: x["Score"], reverse=True)


# ---------- UI logic: hero vs results mode ----------

TRENDING_TICKERS = ["AAPL", "NVDA", "TSLA", "META", "AVGO", "SMCI", "SPY", "QQQ"]

if st.session_state["results"] is None:
    # ---- HERO / FIRST VISIT MODE ----
    st.info("Beta – experimental trend screener using 1/5/20-day momentum and volatility. Feedback welcome.")

    st.write(
        "Enter one or more tickers like `AAPL, TSLA, NVDA` and click **Analyze** "
        "to see multi-factor momentum-based signals."
    )

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

    st.write("---")

    # Handle manual search
    if submitted:
        results_sorted = run_analysis(tickers_input)
        if results_sorted is not None:
            st.session_state["results"] = results_sorted
            st.session_state["last_query"] = tickers_input
            st.session_state["pending_query"] = ""
            st.experimental_rerun()

    # Handle trending ticker click
    if clicked_ticker:
        results_sorted = run_analysis(clicked_ticker)
        if results_sorted is not None:
            st.session_state["results"] = results_sorted
            st.session_state["last_query"] = clicked_ticker
            st.session_state["pending_query"] = ""
            st.experimental_rerun()

else:
    # ---- RESULTS MODE ----
    query = st.session_state["last_query"]

    st.markdown(f"#### Results for: `{query}`")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("New search"):
            # Go back to hero with last query prefilled
            st.session_state["pending_query"] = st.session_state["last_query"]
            st.session_state["results"] = None
            st.experimental_rerun()

    with col2:
        if st.button("Clear & go back"):
            # Go back to hero with empty field
            st.session_state["pending_query"] = ""
            st.session_state["last_query"] = ""
            st.session_state["results"] = None
            st.experimental_rerun()

    st.write("---")

# If we have results in session_state, render them
if st.session_state["results"] is not None:
    results_sorted = st.session_state["results"]

    # ---------- Summary table ----------
    st.subheader("Summary")

    df = pd.DataFrame(results_sorted)[
        [
            "Ticker",
            "Signal",
            "Confidence",
            "Risk",
            "Timeframe",
            "Score",
        ]
    ]
    df["Score"] = df["Score"].round(2)

    st.dataframe(df, use_container_width=True)

    # ---------- Card-style detailed view ----------
    st.write("---")
    st.subheader("Chart Brain Read (per ticker)")

    for row in results_sorted:
        signal_label = label_with_emoji(row["Signal"])
        with st.container():
            st.markdown(f"### {row['Ticker']} – {signal_label}")
            st.markdown(
                f"**Score:** {row['Score']:.2f} · "
                f"**Confidence:** {row['Confidence']} · "
                f"**Risk:** {row['Risk']} · "
                f"**Timeframe:** {row['Timeframe']}"
            )

            # Compact “stat line”
            st.markdown(
                f"• Today: {row['Today %']:+.2f}%  |  "
                f"5-day: {row['5-day %']:+.2f}%  |  "
                f"20-day: {row['20-day %']:+.2f}%  |  "
                f"Vol factor: {row['Vol factor']:.2f}"
            )

            # This is your AI-style context for now
            st.caption(row["Explanation"])

            st.write("---")

# ---------- Global disclaimer footer ----------
st.write("---")
st.caption(
    "Disclaimer: This tool provides automated market analysis for educational "
    "and informational purposes only and does not constitute financial, "
    "investment, or trading advice. No recommendations to buy, sell, or hold "
    "any security are being made. You are solely responsible for your own "
    "investment decisions."
)

