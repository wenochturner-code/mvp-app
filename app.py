import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Friendly Ticker",
    page_icon="📈",
    layout="wide"
)

# ---------------- Basic styling / layout ----------------
st.markdown(
    """
    <style>
    .centered {
        text-align: center;
    }
    .small-text {
        font-size: 0.85rem;
        color: #777777;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Header ----------------
st.markdown("<h1 class='centered'>Friendly Ticker</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='centered'>Quick sanity checks on your stocks before you buy or sell.</p>",
    unsafe_allow_html=True,
)

st.write("")  # spacing

# ---------------- Input / layout ----------------
left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader("Analyze tickers")

    tickers_input = st.text_input(
        "Enter one or more symbols (comma separated)",
        value="AAPL, TSLA, NVDA"
    )

    run_button = st.button("Analyze", type="primary")

with right:
    st.subheader("What this tool does")
    st.write(
        """
        - Looks at recent price action for each ticker  
        - Scores short-term momentum & trend  
        - Labels each as **Bullish / Neutral / Bearish**  
        - Gives a short explanation you can skim quickly  
        """
    )
    st.markdown(
        "<p class='small-text'>For education only. This is not investment advice.</p>",
        unsafe_allow_html=True,
    )

st.write("---")

# ---------------- Signal logic ----------------

def compute_signal_row(ticker: str) -> dict:
    """
    Fetch recent data and compute a simple momentum-style signal.
    Returns a dict for a single ticker.
    """
    try:
        data = yf.Ticker(ticker).history(period="3mo", interval="1d")
        if data.empty or len(data) < 20:
            return {
                "Ticker": ticker.upper(),
                "Today %": None,
                "5D %": None,
                "20D Trend %": None,
                "Volatility": None,
                "Signal": "No data",
                "Score": None,
                "Why": "Not enough recent price history."
            }

        # basic returns
        latest = data["Close"].iloc[-1]
        prev = data["Close"].iloc[-2]

        today_change = (latest - prev) / prev * 100

        # 5-day change (last close vs 5 days ago close)
        if len(data) >= 6:
            five_ago = data["Close"].iloc[-6]
            five_day_change = (latest - five_ago) / five_ago * 100
        else:
            five_day_change = 0.0

        # 20-day trend (simple)
        if len(data) >= 21:
            twenty_ago = data["Close"].iloc[-21]
            trend_20 = (latest - twenty_ago) / twenty_ago * 100
        else:
            trend_20 = 0.0

        # volatility proxy
        vol_factor = data["Close"].pct_change().rolling(20).std().iloc[-1]
        if pd.isna(vol_factor):
            vol_factor = 0.0

        # Score: simple weighted sum (you can tweak these later)
        score = (
            0.4 * today_change +
            0.3 * five_day_change +
            0.3 * trend_20 -
            50 * vol_factor  # penalize high volatility
        )

        # Cap score for sanity
        score = max(min(score, 100), -100)

        # Label
        if score >= 20:
            signal = "Bullish"
        elif score <= -20:
            signal = "Bearish"
        else:
            signal = "Neutral"

        # Short explanation
        reasons = []
        if today_change > 1:
            reasons.append("strong move today")
        elif today_change < -1:
            reasons.append("weak move today")

        if five_day_change > 3:
            reasons.append("solid 1-week strength")
        elif five_day_change < -3:
            reasons.append("weak over the last week")

        if trend_20 > 5:
            reasons.append("uptrend over the last month")
        elif trend_20 < -5:
            reasons.append("downtrend over the last month")

        if vol_factor > 0.04:
            reasons.append("high volatility")

        if not reasons:
            reasons_text = "Mixed recent price action."
        else:
            reasons_text = ", ".join(reasons).capitalize() + "."

        return {
            "Ticker": ticker.upper(),
            "Today %": round(today_change, 2),
            "5D %": round(five_day_change, 2),
            "20D Trend %": round(trend_20, 2),
            "Volatility": round(vol_factor, 4),
            "Signal": signal,
            "Score": round(score, 1),
            "Why": reasons_text,
        }

    except Exception as e:
        return {
            "Ticker": ticker.upper(),
            "Today %": None,
            "5D %": None,
            "20D Trend %": None,
            "Volatility": None,
            "Signal": "Error",
            "Score": None,
            "Why": f"Error loading data: {e}"
        }


# ---------------- Run analysis ----------------
results_df = None

if run_button:
    raw = tickers_input.strip()
    if not raw:
        st.warning("Please enter at least one ticker symbol.")
    else:
        tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
        rows = [compute_signal_row(t) for t in tickers]
        results_df = pd.DataFrame(rows)

        st.subheader("Results")
        st.dataframe(
            results_df.set_index("Ticker"),
            use_container_width=True,
        )

        # Quick summary
        bull = (results_df["Signal"] == "Bullish").sum()
        bear = (results_df["Signal"] == "Bearish").sum()
        neutral = (results_df["Signal"] == "Neutral").sum()

        st.markdown(
            f"**Summary:** 🟢 {bull} Bullish · ⚪ {neutral} Neutral · 🔴 {bear} Bearish"
        )

# ---------------- Footer ----------------
st.write("")
st.markdown(
    "<p class='small-text centered'>Friendly Ticker is a simple research tool. "
    "Always do your own homework before investing.</p>",
    unsafe_allow_html=True,
)












