import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Friendly Ticker",
    page_icon="📈",
    layout="wide",
)

# ---------------- Simple styling ----------------
st.markdown(
    """
    <style>
    .centered {
        text-align: center;
    }
    .small-text {
        font-size: 0.85rem;
        color: #777777;
        text-align: center;
    }
    /* tighten top padding so content sits higher */
    .main {
        padding-top: 30px !important;
    }
    /* simple card style for results */
    .card {
        padding: 20px;
        border-radius: 12px;
        background-color: #fafafa;
        border: 1px solid #eeeeee;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Lightweight logging ----------------
def log_event(event_type: str, tickers: str = ""):
    """
    Very simple event logger.
    Appends a line to events_log.csv:
    timestamp | event_type | tickers
    """
    try:
        ts = datetime.utcnow().isoformat()
        row = f"{ts}|{event_type}|{tickers}\n"
        with open("events_log.csv", "a", encoding="utf-8") as f:
            f.write(row)
    except Exception:
        # logging should NEVER crash the app
        pass


# Make sure we only log one page_view per browser session
if "page_view_logged" not in st.session_state:
    log_event("page_view", "")
    st.session_state["page_view_logged"] = True

# ---------------- Header ----------------
st.markdown("<h1 class='centered'>Friendly Ticker</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='centered'>Quick sanity checks on your stocks before you buy or sell.</p>",
    unsafe_allow_html=True,
)
# pull content slightly closer to the header
st.markdown("<div style='margin-top:-10px;'></div>", unsafe_allow_html=True)

# ---------------- Layout ----------------
with st.container():
    left, right = st.columns([2, 1], gap="large")

    with left:
        st.markdown("### Analyze tickers")

        tickers_input = st.text_input(
            "Enter one or more symbols (comma separated)",
            value="AAPL, TSLA, NVDA",
            help="Example: AAPL, TSLA, NVDA",
        )

        analyze_button = st.button("Analyze", type="primary")

    with right:
        st.markdown("### What this tool does")
        st.write(
            """
            - Looks at recent price action for each ticker  
            - Scores short-term momentum & trend  
            - Labels each as **Bullish / Neutral / Bearish**  
            - Gives a short explanation you can skim in seconds  
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
    Returns a dict with everything needed for the results table.
    """
    ticker_clean = ticker.upper()

    try:
        data = yf.Ticker(ticker_clean).history(period="3mo", interval="1d")

        if data.empty or len(data) < 20:
            return {
                "Ticker": ticker_clean,
                "Today %": None,
                "5D %": None,
                "20D Trend %": None,
                "Volatility": None,
                "Score": None,
                "Signal": "No data",
                "Why": "Not enough recent price history.",
            }

        # latest close vs previous close
        latest = data["Close"].iloc[-1]
        prev = data["Close"].iloc[-2]
        today_change = (latest - prev) / prev * 100

        # 5-day change: last close vs 5 trading days ago
        if len(data) >= 6:
            five_ago = data["Close"].iloc[-6]
            five_day_change = (latest - five_ago) / five_ago * 100
        else:
            five_day_change = 0.0

        # 20-day trend: last close vs 20 trading days ago
        if len(data) >= 21:
            twenty_ago = data["Close"].iloc[-21]
            trend_20 = (latest - twenty_ago) / twenty_ago * 100
        else:
            trend_20 = 0.0

        # volatility: 20-day rolling std of daily returns
        vol_factor = (
            data["Close"].pct_change().rolling(20).std().iloc[-1]
        )
        if pd.isna(vol_factor):
            vol_factor = 0.0

        # Simple score (tweak later if you want)
        score = (
            0.4 * today_change +
            0.3 * five_day_change +
            0.3 * trend_20 -
            50 * vol_factor  # penalize high volatility
        )
        score = max(min(score, 100), -100)  # clamp to [-100, 100]

        # Label based on score
        if score >= 20:
            signal = "Bullish"
        elif score <= -20:
            signal = "Bearish"
        else:
            signal = "Neutral"

        # Explanation pieces
        reasons = []

        if today_change > 1.5:
            reasons.append("strong move today")
        elif today_change < -1.5:
            reasons.append("weak move today")

        if five_day_change > 3:
            reasons.append("solid strength over the last week")
        elif five_day_change < -3:
            reasons.append("weak over the last week")

        if trend_20 > 5:
            reasons.append("uptrend over the last month")
        elif trend_20 < -5:
            reasons.append("downtrend over the last month")

        if vol_factor > 0.04:
            reasons.append("high volatility recently")

        if not reasons:
            reasons_text = "Mixed recent price action."
        else:
            reasons_text = ", ".join(reasons).capitalize() + "."

        return {
            "Ticker": ticker_clean,
            "Today %": round(today_change, 2),
            "5D %": round(five_day_change, 2),
            "20D Trend %": round(trend_20, 2),
            "Volatility": round(vol_factor, 4),
            "Score": round(score, 1),
            "Signal": signal,
            "Why": reasons_text,
        }

    except Exception as e:
        return {
            "Ticker": ticker_clean,
            "Today %": None,
            "5D %": None,
            "20D Trend %": None,
            "Volatility": None,
            "Score": None,
            "Signal": "Error",
            "Why": f"Error loading data: {e}",
        }


# ---------------- Run analysis ----------------
if analyze_button:
    raw = tickers_input.strip()
    if not raw:
        st.warning("Please enter at least one ticker symbol.")
    else:
        tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
        # log analyze event
        log_event("analyze", ",".join(tickers))

        rows = [compute_signal_row(t) for t in tickers]
        results_df = pd.DataFrame(rows)

        st.markdown("### Results")
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.dataframe(
            results_df.set_index("Ticker"),
            use_container_width=True,
        )

        # Summary row
        bull = (results_df["Signal"] == "Bullish").sum()
        bear = (results_df["Signal"] == "Bearish").sum()
        neutral = (results_df["Signal"] == "Neutral").sum()

        st.markdown(
            f"**Summary:** 🟢 {bull} Bullish · ⚪ {neutral} Neutral · 🔴 {bear} Bearish"
        )

        st.markdown(
            "<p class='small-text'>Always double-check before making real trades.</p>",
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)









