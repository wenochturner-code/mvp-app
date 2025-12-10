import streamlit as st
import pandas as pd
import yfinance as yf

st.set_page_config(page_title="Stock Analyzer MVP", page_icon="📈")

st.title("Stock Analyzer MVP")
st.info("Beta version – experimental trend screener using 1/5/20-day momentum and volatility. Feedback welcome.")

st.write(
    "Enter one or more tickers like `AAPL, TSLA, NVDA` and click **Analyze** "
    "to see multi-factor momentum-based signals."
)

tickers_input = st.text_input("Tickers", "AAPL, TSLA, NVDA")


def label_with_emoji(label: str) -> str:
    if label == "Bullish":
        return "🟢 Bullish"
    if label == "Bearish":
        return "🔴 Bearish"
    return "⚪ Neutral"


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
        explanation: str
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

    # Slightly more aggressive scaling so normal trends push the score more
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
        # otherwise Low
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

    if signal == "Bullish":
        reason = (
            f"{ticker} is showing a **Bullish** trend overall. "
            f"Short- and medium-term momentum are skewed to the upside."
        )
    elif signal == "Bearish":
        reason = (
            f"{ticker} is showing a **Bearish** trend overall. "
            f"Short- and medium-term momentum are skewed to the downside."
        )
    else:
        reason = (
            f"{ticker} is **Neutral** right now. "
            f"Recent moves don’t strongly favor bulls or bears."
        )

    explanation = (
        f"{reason} "
        f"Recent performance → {trend_summary}. "
        f"{vol_note} "
        f"(Overall score: {score:+.2f}, confidence: {confidence}.)"
    )

    return signal, confidence, explanation, score


st.write("---")

if st.button("Analyze"):
    raw_tickers = [t.strip().upper() for t in tickers_input.split(",")]
    tickers = [t for t in raw_tickers if t]

    if not tickers:
        st.warning("Please enter at least one ticker symbol.")
    else:
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
                            "Explanation": explanation,
                        }
                    )

                except Exception as e:
                    st.error(f"Error analyzing {ticker}: {e}")

        if not results:
            st.info("No valid results to show yet.")
        else:
            results_sorted = sorted(results, key=lambda x: x["Score"], reverse=True)

            st.subheader("Summary Table")

            df = pd.DataFrame(results_sorted)[
                ["Ticker", "Signal", "Confidence", "Score", "Today %", "5-day %", "20-day %", "Vol factor"]
            ]

            df["Score"] = df["Score"].round(2)
            df["Today %"] = df["Today %"].round(2)
            df["5-day %"] = df["5-day %"].round(2)
            df["20-day %"] = df["20-day %"].round(2)
            df["Vol factor"] = df["Vol factor"].round(2)

            st.dataframe(df, use_container_width=True)

            st.write("---")
            st.subheader("Detailed Signals")

            for row in results_sorted:
                signal_label = label_with_emoji(row["Signal"])
                st.markdown(
                    f"### {row['Ticker']} – {signal_label} (Confidence: {row['Confidence']})"
                )
                st.caption(row["Explanation"])


