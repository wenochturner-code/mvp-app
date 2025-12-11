import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# ----------------- Page config & brand -----------------
st.set_page_config(
    page_title="Friendly Ticker • Beginner Stock Analyzer",
    page_icon="📈",
    layout="wide",
)

PRIMARY = "#2563EB"   # brand blue
ACCENT = "#22C55E"    # mint accent
WARNING = "#F97316"
TEXT_MAIN = "#F9FAFB"
TEXT_MUTED = "#9CA3AF"

custom_css = f"""
<style>
    /* Global background / typography */
    .main {{
        background: #020617;
        color: {TEXT_MAIN};
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
    }}

    section[data-testid="stSidebar"] {{
        background-color: #020617;
        border-right: 1px solid #111827;
    }}

    /* Brand lockup in sidebar */
    .friendly-logo {{
        display:flex;
        align-items:center;
        gap:0.45rem;
        font-weight:700;
        letter-spacing:-0.03em;
        font-size:1rem;
        color:{TEXT_MAIN};
        margin-bottom:0.1rem;
    }}
    .friendly-logo-mark {{
        width:1.7rem;
        height:1.7rem;
        border-radius:0.6rem;
        background:linear-gradient(135deg,{PRIMARY}, {ACCENT});
        display:flex;
        align-items:center;
        justify-content:center;
        font-size:1.05rem;
        box-shadow:0 8px 18px rgba(15,23,42,0.9);
    }}

    .friendly-hero-title {{
        font-size: 2.35rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        margin-bottom: 0.35rem;
        color:#FFFFFF;
        text-shadow:0 0 18px rgba(0,0,0,0.75);
    }}
    .friendly-hero-sub {{
        font-size: 1rem;
        color: {TEXT_MUTED};
        margin-bottom: 0.7rem;
    }}

    .pill {{
        display:inline-flex;
        align-items:center;
        gap:0.4rem;
        padding:0.2rem 0.7rem;
        border-radius:999px;
        border:1px solid rgba(148,163,184,0.55);
        background:#020617;
        font-size:0.78rem;
        color:{TEXT_MAIN};
    }}
    .pill span.badge {{
        padding:0.05rem 0.5rem;
        border-radius:999px;
        font-size:0.7rem;
        font-weight:600;
        background:rgba(37,99,235,0.2);
        color:{PRIMARY};
    }}

    .hero-card {{
        border-radius: 1.35rem;
        padding: 1.4rem 1.6rem;
        border: 1px solid rgba(148,163,184,0.25);
        background: #020617;
        box-shadow: 0 18px 40px rgba(15,23,42,0.9);
    }}

    .hero-side-card {{
        border-radius: 1.2rem;
        padding: 1.05rem 1.25rem;
        border: 1px solid rgba(148,163,184,0.25);
        background: #020617;
        box-shadow: 0 14px 35px rgba(15,23,42,0.9);
    }}

    .hero-side-card * {{
        color:{TEXT_MAIN};
        font-size:0.82rem;
    }}
    .hero-side-title {{
        font-size:0.85rem;
        font-weight:600;
        letter-spacing:0.12em;
        text-transform:uppercase;
        color:{TEXT_MUTED} !important;
        margin-bottom:0.2rem;
    }}

    /* Input area */
    .input-card {{
        border-radius: 1.1rem;
        padding: 1.0rem 1.1rem 0.6rem 1.1rem;
        border: 1px solid rgba(55,65,81,0.85);
        background: #020617;
        margin-top:0.7rem;
        margin-bottom:0.5rem;
    }}

    .stTextInput>div>div>input {{
        background-color: #020617;
        border-radius: 0.7rem;
        border: 1px solid #374151;
        color: {TEXT_MAIN};
        font-size:0.9rem;
    }}
    .stTextInput>div>div>input::placeholder {{
        color:#6B7280;
    }}

    .help-text {{
        font-size:0.78rem;
        color:{TEXT_MUTED};
        margin-top:0.12rem;
    }}

    /* Buttons */
    .stButton>button {{
        border-radius: 999px;
        border: 1px solid rgba(37,99,235,0.9);
        background: linear-gradient(135deg, {PRIMARY}, #1D4ED8);
        color: white;
        font-weight:600;
        padding:0.4rem 1.5rem;
        font-size:0.9rem;
        box-shadow:0 10px 25px rgba(37,99,235,0.35);
    }}
    .stButton>button:hover {{
        border-color:#60A5FA;
        background: linear-gradient(135deg, #1D4ED8, #1E3A8A);
        box-shadow:0 12px 28px rgba(37,99,235,0.45);
    }}

    /* Result cards */
    .result-card {{
        border-radius: 1.05rem;
        padding: 0.9rem 1.0rem;
        border: 1px solid rgba(55,65,81,0.9);
        background: #020617;
        margin-bottom: 0.6rem;
        box-shadow:0 10px 28px rgba(15,23,42,0.9);
    }}

    .ticker-header {{
        display:flex;
        justify-content:space-between;
        align-items:flex-start;
        gap:0.65rem;
        margin-bottom:0.2rem;
    }}
    .ticker-main {{
        font-size:1.05rem;
        font-weight:700;
        letter-spacing:0.03em;
        color:{TEXT_MAIN};
    }}
    .ticker-sub {{
        font-size:0.78rem;
        color:{TEXT_MUTED};
    }}

    .score-label {{
        font-size:0.75rem;
        color:{TEXT_MUTED};
    }}

    .score-badge {{
        border-radius:999px;
        padding:0.15rem 0.7rem;
        font-size:0.78rem;
        font-weight:600;
        display:inline-flex;
        align-items:center;
        gap:0.25rem;
    }}

    .metric-grid {{
        display:flex;
        flex-wrap:wrap;
        gap:0.35rem;
        margin-top:0.35rem;
    }}
    .metric-pill {{
        border-radius:999px;
        padding:0.18rem 0.7rem;
        font-size:0.75rem;
        border:1px solid rgba(75,85,99,0.9);
        background:rgba(30,41,59,0.95);
        color:{TEXT_MAIN};
    }}

    .explanation {{
        font-size:0.8rem;
        color:{TEXT_MAIN};
        margin-top:0.25rem;
    }}

    .neutral-pill {{
        background:rgba(55,65,81,0.9);
        border:1px solid rgba(75,85,99,1);
        color:{TEXT_MAIN};
    }}
    .bull-pill {{
        background:rgba(22,163,74,0.16);
        border:1px solid rgba(34,197,94,0.95);
        color:#BBF7D0;
    }}
    .bear-pill {{
        background:rgba(220,38,38,0.16);
        border:1px solid rgba(248,113,113,0.95);
        color:#FECACA;
    }}

    .stDataFrame div[data-testid="stTable"] {{
        border-radius:0.8rem;
        overflow:hidden;
    }}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# ----------------- Sidebar -----------------
with st.sidebar:
    st.markdown(
        """
        <div class="friendly-logo">
            <div class="friendly-logo-mark">📈</div>
            <div>Friendly Ticker</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:0.85rem;color:#9CA3AF;'>"
        "Beginner-friendly momentum insights for U.S.-listed stocks. "
        "Designed for education, practice, and learning how markets move."
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("#### ⚙️ Signal settings")

    sensitivity = st.slider(
        "Signal sensitivity (Bullish vs Bearish)",
        min_value=1,
        max_value=5,
        value=3,
        help="Higher = stricter Bullish/Bearish labels. Lower = more signals.",
    )

    st.markdown("#### ⏱️ Lookback windows")
    st.caption("Currently using 1-day, 5-day, and 20-day performance windows.")

    st.markdown("---")
    st.markdown("#### 🔒 Roadmap preview")
    st.caption(
        "Planned upgrades include watchlists, alerts, richer explanations, "
        "and simple historical testing. The current release focuses on core momentum scoring."
    )


# ----------------- Helper functions -----------------
def simple_label(value: float, bull_thr: float, bear_thr: float) -> str:
    if value is None:
        return "No signal"
    if value >= bull_thr:
        return "Bullish"
    if value <= bear_thr:
        return "Bearish"
    return "Neutral"


def label_to_emoji(label: str) -> str:
    if label == "Bullish":
        return "🟢"
    if label == "Bearish":
        return "🔴"
    if label == "Neutral":
        return "⚪"
    return "⚪"


def compute_signal(
    ticker: str,
    hist: pd.DataFrame,
    sensitivity: int,
):
    # Need at least 21 trading days for 20-day trend
    if len(hist) < 21:
        return {
            "ticker": ticker.upper(),
            "score": None,
            "label": "No signal",
            "emoji": "⚪",
            "today_change": None,
            "five_day_change": None,
            "trend_20": None,
            "vol_factor": None,
            "explanation": "Insufficient recent history to produce a momentum snapshot.",
        }

    hist = hist.sort_index()
    close = hist["Close"]

    today_change = (close.iloc[-1] / close.iloc[-2] - 1) * 100
    five_day_change = (close.iloc[-1] / close.iloc[-6] - 1) * 100
    trend_20 = (close.iloc[-1] / close.iloc[-21] - 1) * 100

    returns = close.pct_change()
    recent_vol = returns.iloc[-10:].std()
    long_vol = returns.std()
    vol_factor = float(recent_vol / long_vol) if long_vol and not pd.isna(long_vol) else 1.0

    # Sensitivity scaling
    base_bull_1d = 0.4 + 0.3 * (sensitivity - 3)
    base_bear_1d = -0.4 - 0.3 * (sensitivity - 3)

    base_bull_5d = 1.5 + 0.6 * (sensitivity - 3)
    base_bear_5d = -1.5 - 0.6 * (sensitivity - 3)

    base_bull_20d = 4.0 + 1.5 * (sensitivity - 3)
    base_bear_20d = -4.0 - 1.5 * (sensitivity - 3)

    score = 50

    if today_change > base_bull_1d:
        score += 8
    elif today_change < base_bear_1d:
        score -= 8

    if five_day_change > base_bull_5d:
        score += 16
    elif five_day_change < base_bear_5d:
        score -= 16

    if trend_20 > base_bull_20d:
        score += 26
    elif trend_20 < base_bear_20d:
        score -= 26

    if vol_factor > 1.5:
        score -= 8
    elif vol_factor < 0.7:
        score += 6

    score = max(0, min(100, int(round(score))))

    bull_cut = 60 + 2 * (sensitivity - 3)
    bear_cut = 40 - 2 * (sensitivity - 3)

    label = simple_label(score, bull_cut, bear_cut)
    emoji = label_to_emoji(label)

    parts = []
    if label == "Bullish":
        parts.append("Recent price behaviour leans positive across the short and medium term.")
    elif label == "Bearish":
        parts.append("Recent price behaviour has skewed negative over the last few weeks.")
    else:
        parts.append("Signals are mixed — momentum is not clearly directional.")

    if today_change > 0.5:
        parts.append(f"Session move: up {today_change:.1f}% today.")
    elif today_change < -0.5:
        parts.append(f"Session move: down {today_change:.1f}% today.")

    if five_day_change > 2:
        parts.append(f"Five-day performance: +{five_day_change:.1f}%.")
    elif five_day_change < -2:
        parts.append(f"Five-day performance: {five_day_change:.1f}%.")

    if trend_20 > 5:
        parts.append("Twenty-day trend is firmly positive.")
    elif trend_20 < -5:
        parts.append("Twenty-day trend is firmly negative.")

    if vol_factor > 1.5:
        parts.append("Volatility is elevated relative to this name's recent history.")
    elif vol_factor < 0.7:
        parts.append("Volatility is subdued relative to recent history.")

    explanation = " ".join(parts)

    return {
        "ticker": ticker.upper(),
        "score": score,
        "label": label,
        "emoji": emoji,
        "today_change": today_change,
        "five_day_change": five_day_change,
        "trend_20": trend_20,
        "vol_factor": vol_factor,
        "explanation": explanation,
    }


# ----------------- Layout: hero & input -----------------
hero_col, side_col = st.columns([1.6, 1.1])

with hero_col:
    st.markdown(
        """
        <div class="hero-card">
            <div class="pill">
                <span class="badge">NEW</span>
                <span>Beginner-friendly stock signals · Educational only</span>
            </div>
            <h1 class="friendly-hero-title">
                See Bullish, Neutral, or Bearish signals in seconds.
            </h1>
            <p class="friendly-hero-sub">
                Friendly Ticker summarizes recent price movement and volatility into one,
                easy-to-read momentum-style score — so new investors can practice reading
                trends without wading through charts and jargon.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with side_col:
    st.markdown(
        """
        <div class="hero-side-card">
            <div class="hero-side-title">HOW IT WORKS</div>
            <ul>
                <li>Enter a list of stock symbols (for example: <code>AAPL, TSLA, NVDA</code>).</li>
                <li>Friendly Ticker looks at recent price history for each symbol.</li>
                <li>Short, medium, and 20-day changes are combined into a 0–100 score.</li>
                <li>You see a simple Bullish / Neutral / Bearish label plus a concise explanation.</li>
            </ul>
            <p style="margin-top:0.4rem;color:#9CA3AF;font-size:0.78rem;">
                Built for education and market practice. Not a trading signal and not investment advice.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.86rem;color:#9CA3AF;margin-bottom:0.2rem;'>Tickers to analyze</div>",
        unsafe_allow_html=True,
    )
    tickers_input = st.text_input(
        "",
        value="AAPL, TSLA, NVDA",
        placeholder="e.g. AAPL, TSLA, NVDA, MSFT",
        help="Separate tickers with commas or spaces.",
    )
    st.markdown(
        "<div class='help-text'>U.S.-listed equities with sufficient recent trading history generally work best.</div>",
        unsafe_allow_html=True,
    )

    btn_col, hint_col = st.columns([0.25, 0.75])
    with btn_col:
        run = st.button("Analyze tickers")
    with hint_col:
        st.markdown(
            "<div class='help-text'>Example watchlist: <code>AAPL, MSFT, META, SOXL</code></div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

# ----------------- Main logic -----------------
results = []

if run:
    cleaned = [t.strip().upper() for t in tickers_input.replace(";", ",").replace(" ", ",").split(",") if t.strip()]
    cleaned = list(dict.fromkeys(cleaned))  # dedupe

    if not cleaned:
        st.warning("Please enter at least one stock symbol (for example: AAPL or TSLA).")
    else:
        with st.spinner("Running momentum checks on your watchlist..."):
            end = datetime.today()
            start = end - timedelta(days=60)

            for ticker in cleaned:
                try:
                    hist = yf.download(ticker, start=start, end=end, progress=False)
                    if hist.empty:
                        results.append(
                            {
                                "ticker": ticker.upper(),
                                "score": None,
                                "label": "No data",
                                "emoji": "⚪",
                                "today_change": None,
                                "five_day_change": None,
                                "trend_20": None,
                                "vol_factor": None,
                                "explanation": "No recent trading data was available for this symbol.",
                            }
                        )
                    else:
                        res = compute_signal(ticker, hist, sensitivity=sensitivity)
                        results.append(res)
                except Exception:
                    results.append(
                        {
                            "ticker": ticker.upper(),
                            "score": None,
                            "label": "Error",
                            "emoji": "⚪",
                            "today_change": None,
                            "five_day_change": None,
                            "trend_20": None,
                            "vol_factor": None,
                            "explanation": "Something went wrong while loading market data.",
                        }
                    )

if results:
    st.markdown("### 🔍 Results overview")

    df_rows = []
    for r in results:
        df_rows.append(
            {
                "Ticker": r["ticker"],
                "Score (0–100)": r["score"],
                "Label": r["label"],
                "Today %": None if r["today_change"] is None else round(r["today_change"], 2),
                "5-day %": None if r["five_day_change"] is None else round(r["five_day_change"], 2),
                "20-day %": None if r["trend_20"] is None else round(r["trend_20"], 2),
                "Vol vs recent": None if r["vol_factor"] is None else round(r["vol_factor"], 2),
            }
        )
    df = pd.DataFrame(df_rows)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)
    st.markdown("#### 🧠 Ticker breakdowns")

    left_col, right_col = st.columns(2)

    for idx, r in enumerate(results):
        col = left_col if idx % 2 == 0 else right_col
        with col:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)

            if r["label"] == "Bullish":
                badge_class = "bull-pill"
                badge_text = "Bullish"
            elif r["label"] == "Bearish":
                badge_class = "bear-pill"
                badge_text = "Bearish"
            elif r["label"] == "Neutral":
                badge_class = "neutral-pill"
                badge_text = "Neutral"
            else:
                badge_class = "neutral-pill"
                badge_text = r["label"]

            st.markdown(
                f"""
                <div class="ticker-header">
                    <div>
                        <div class="ticker-main">{r["emoji"]} {r["ticker"]}</div>
                        <div class="ticker-sub">Condensed momentum snapshot based on recent market behaviour.</div>
                    </div>
                    <div style="text-align:right;">
                        <div class="score-label">Score</div>
                        <div class="score-badge {badge_class}">
                            <span>{r.get("score", "–") if r.get("score") is not None else "–"}</span>
                            <span style="font-size:0.75rem;">/ 100</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if r["today_change"] is None:
                st.markdown(
                    f"<div class='explanation'>{r['explanation']}</div>",
                    unsafe_allow_html=True,
                )
            else:
                metric_bits = [
                    f"Today: {r['today_change']:+.2f}%",
                    f"5-day: {r['five_day_change']:+.2f}%",
                    f"20-day: {r['trend_20']:+.2f}%",
                    f"Volatility vs recent: {r['vol_factor']:.2f}×",
                ]

                st.markdown(
                    "<div class='metric-grid'>" +
                    "".join([f"<div class='metric-pill'>{m}</div>" for m in metric_bits]) +
                    "</div>",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"<div class='explanation'>{r['explanation']}</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<p style='font-size:0.75rem;color:#6B7280;'>"
    "Friendly Ticker is an educational tool that summarizes historical market data into simple momentum-style scores. "
    "Outputs are not investment recommendations and should not be used as the sole basis for any trading decision."
    "</p>",
    unsafe_allow_html=True,
)












