from datetime import datetime
from pathlib import Path
import random

import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.data.fi2010 import FI2010Loader
from src.models.multihorizon_model import TemporalCNNGRUAttention


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parent

DATASET_ROOT = ROOT / "deployment_data"

CHECKPOINT_PATH = (
    ROOT / "results" / "checkpoints" / "best_cf1_model.pt"
)

HORIZONS = [1, 2, 3, 5, 10]
CLASSES = ["UP", "STATIONARY", "DOWN"]

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

st.set_page_config(
    page_title="LOB Intelligence | Quant Terminal",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PREMIUM WEB3 DESIGN
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root{
 --bg:#05080e;--panel:#0a111c;--panel2:#0d1927;--line:#1b3044;
 --cyan:#52e6ff;--blue:#568dff;--green:#36e2a1;--red:#ff597d;
 --amber:#f4c66b;--muted:#758da9;--white:#eaf4ff;
}
html,body,[class*="css"]{font-family:'Space Grotesk',sans-serif}
.stApp{
 background:
 radial-gradient(ellipse at 50% -20%,rgba(26,118,164,.19),transparent 42%),
 linear-gradient(180deg,#060b13,#04070d 70%);
 color:var(--white)
}
.block-container{max-width:1800px;padding:1rem 1.35rem 2rem}
[data-testid="stHeader"]{background:rgba(5,8,14,.92)}
[data-testid="stSidebar"]{background:#080f19;border-right:1px solid #1b3044}
[data-testid="stSidebar"] *{color:#d5e5f6}
h1,h2,h3,h4{font-family:'Space Grotesk',sans-serif!important;color:#f0f7ff!important;letter-spacing:-.04em}
h2{font-size:1.25rem!important}
h3{font-size:1.03rem!important}
p{color:#a4b7ce}
hr{border-color:#1b3044}
[data-testid="stMetric"]{
 background:linear-gradient(145deg,#0e1b2b,#09121e);
 border:1px solid #1e3a51;border-radius:7px;padding:11px 13px;
 box-shadow:inset 0 1px rgba(255,255,255,.025)
}
[data-testid="stMetricLabel"]{color:#829bb6;font:10px 'DM Mono',monospace;text-transform:uppercase;letter-spacing:1px}
[data-testid="stMetricValue"]{color:#f2f8ff;font:500 22px 'DM Mono',monospace}
[data-testid="stMetricDelta"]{font:10px 'DM Mono',monospace}
[data-testid="stDataFrame"]{border:1px solid #1b3044;border-radius:7px}
.stButton>button{
 background:linear-gradient(100deg,#087e9c,#2458bd);border:1px solid #3aa6c6;
 color:white;border-radius:5px;font:600 11px 'DM Mono',monospace;text-transform:uppercase
}
.stButton>button:hover{color:white;border-color:var(--cyan);box-shadow:0 0 16px #28c6ed33}
[data-testid="stRadio"] label{font-size:13px}
.terminal-top{
 display:flex;align-items:center;justify-content:space-between;gap:12px;
 border:1px solid #1b354a;background:#08121e;padding:9px 13px;border-radius:6px;
 font:10px 'DM Mono',monospace;color:#8ba5c0;letter-spacing:.7px
}
.brandmark{font:700 14px 'Space Grotesk',sans-serif;color:#f1f8ff;letter-spacing:1.4px}
.brandmark span{color:var(--cyan)}
.eyebrow{font:10px 'DM Mono',monospace;color:#58cde7;letter-spacing:1.8px;text-transform:uppercase}
.hero-title{font:700 clamp(25px,3vw,39px) 'Space Grotesk',sans-serif;color:#f4f8ff;letter-spacing:-1.8px;margin:8px 0 4px}
.hero-sub{font-size:12px;color:#8fa7c2}
.hero-box{
 border:1px solid #24445d;border-radius:8px;padding:19px 21px;margin:12px 0 15px;
 background:linear-gradient(110deg,#0b1c2b,#0c2033 55%,#171b35);
 box-shadow:0 12px 35px #0006;position:relative;overflow:hidden
}
.hero-box:after{
 content:"";position:absolute;width:250px;height:250px;right:-90px;top:-150px;
 border:1px solid #52e6ff33;border-radius:50%;box-shadow:0 0 0 22px #52e6ff09,0 0 0 48px #52e6ff06
}
.micro-label{font:9px 'DM Mono',monospace;color:#6f8aa8;letter-spacing:1.4px;text-transform:uppercase}
.panel{
 background:linear-gradient(145deg,#0b1623,#08111b);border:1px solid #1b344a;
 border-radius:7px;padding:12px 13px;height:100%
}
.panel-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:9px}
.panel-title{font:600 12px 'Space Grotesk',sans-serif;color:#dceaff}
.panel-tag{font:9px 'DM Mono',monospace;color:#6685a5}
.ticker{
 background:#0a1420;border:1px solid #1b354b;border-radius:6px;padding:9px 11px;
 min-height:64px
}
.ticker-label{font:9px 'DM Mono',monospace;color:#7893af;letter-spacing:1px}
.ticker-value{font:500 18px 'DM Mono',monospace;color:#eaf5ff;margin-top:5px}
.ticker-delta{font:9px 'DM Mono',monospace;color:#36e2a1}
.signal{
 border:1px solid #234158;border-radius:6px;background:linear-gradient(150deg,#0e1d2d,#09121e);
 padding:13px 6px;text-align:center;position:relative;overflow:hidden
}
.signal:before{content:"";height:2px;position:absolute;top:0;left:0;right:0;background:var(--accent)}
.signal-h{font:9px 'DM Mono',monospace;color:#829bb6;letter-spacing:1px}
.signal-v{font:700 18px 'DM Mono',monospace;margin:8px 0 5px;color:var(--accent)}
.signal-p{font:9px 'DM Mono',monospace;color:#91a8c2}
.status-pill{font:9px 'DM Mono',monospace;color:#48e2a7;border:1px solid #245943;background:#0b241d;padding:4px 7px;border-radius:3px}
.chart-wrap{border:1px solid #1b354b;border-radius:7px;background:#08121d;padding:8px}
.depth-row{display:grid;grid-template-columns:1fr 1fr 55px 1fr 1fr;gap:5px;align-items:center;font:9px 'DM Mono',monospace;color:#9bb0c7;padding:5px 3px;border-bottom:1px solid #142638}
.depth-bar{height:5px;border-radius:2px}
.footnote{font:9px 'DM Mono',monospace;color:#607b99;letter-spacing:.5px;border-top:1px solid #1b3044;padding-top:10px}
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA AND MODEL
# =========================================================

@st.cache_data
def load_test_data():
    loader = FI2010Loader(str(DATASET_ROOT))
    X, y = loader.load_fold(1, split="Testing")
    return X, y


@st.cache_resource
def load_model():
    model = TemporalCNNGRUAttention().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


# The trained model is required for the dashboard. The FI-2010 dataset
# is optional in cloud deployments until its redistribution terms are confirmed.
try:
    with st.spinner("Initializing Quant Research Engine..."):
        model, checkpoint = load_model()
except Exception as error:
    st.error(f"Model initialization failed: {error}")
    st.info(
        "Check the checkpoint location, model architecture, "
        "and installed package versions."
    )
    st.stop()

X_test = None
y_test = None
DATA_AVAILABLE = False
DATA_ERROR = None

try:
    X_test, y_test = load_test_data()
    DATA_AVAILABLE = len(X_test) >= 100
    if not DATA_AVAILABLE:
        DATA_ERROR = "Insufficient test data for a 100-step input window."
except Exception as error:
    DATA_ERROR = str(error)

MAX_INDEX = len(X_test) - 100 if DATA_AVAILABLE else -1


# =========================================================
# ACTUAL MODEL INFERENCE
# =========================================================

@st.cache_data(max_entries=256)
def run_prediction(index):
    window = X_test[index:index + 100]

    tensor = torch.from_numpy(
        window.astype(np.float32)
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits, attention = model(tensor)

        probabilities = torch.softmax(
            logits,
            dim=-1
        )[0].detach().cpu().numpy()

        attention_values = attention[0].detach().cpu().numpy()

    predicted = probabilities.argmax(axis=-1)
    actual = y_test[index + 99].astype(int) - 1

    return probabilities, attention_values, predicted, actual


# =========================================================
# LIVE CLOCK
# =========================================================

@st.fragment(run_every="1s")
def clock():
    now = datetime.now().strftime("%d %b %Y  |  %H:%M:%S")

    st.markdown(
        f"""
        <div class="terminal-top">
          <span>◷ SYSTEM TIME&nbsp;&nbsp; {now}</span>
          <span><span class="live-indicator"></span>MODEL ENGINE ONLINE</span>
          <span style="color:#f4c66b">● DATA SOURCE: HISTORICAL / SIMULATED</span>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# SIMULATED MARKET TICKER
# =========================================================

@st.fragment(run_every="2s")
def market_ticker():
    if "demo_price" not in st.session_state:
        st.session_state.demo_price = 100.00

    old_price = st.session_state.demo_price
    new_price = max(1, old_price + random.uniform(-0.25, 0.25))
    st.session_state.demo_price = new_price
    change = ((new_price - old_price) / old_price) * 100

    items = [
        ("BTC / USD", f"${new_price:,.2f}", f"{change:+.2f}%"),
        ("BID VOLUME", f"{random.randint(12, 48)}.{random.randint(1,9)}K", "SIMULATED"),
        ("ASK VOLUME", f"{random.randint(10, 45)}.{random.randint(1,9)}K", "SIMULATED"),
        ("SPREAD", f"{random.uniform(0.01, 0.09):.3f}", "SIMULATED"),
        ("VOLATILITY", f"{random.uniform(0.2, 1.8):.2f}%", "SIMULATED"),
    ]
    cols = st.columns(5, gap="small")
    for col, (label, value, delta) in zip(cols, items):
        with col:
            st.markdown(f"""
            <div class="ticker">
              <div class="ticker-label">{label}</div>
              <div class="ticker-value">{value}</div>
              <div class="ticker-delta">{delta}</div>
            </div>
            """, unsafe_allow_html=True)



# =========================================================
# SIMULATED PRICE CHART
# =========================================================

@st.fragment(run_every="3s")
def price_chart():
    if "price_history" not in st.session_state:
        st.session_state.price_history = [
            100 + random.uniform(-0.5, 0.5) for _ in range(48)
        ]

    history = st.session_state.price_history
    history.append(max(1, history[-1] + random.uniform(-0.42, 0.42)))
    history = history[-60:]
    st.session_state.price_history = history

    width, height = 900, 300
    pad_l, pad_r, pad_t, pad_b = 48, 18, 20, 30
    lo, hi = min(history) - .5, max(history) + .5
    span = max(hi - lo, .01)

    points = []
    for i, val in enumerate(history):
        x = pad_l + i * (width - pad_l - pad_r) / (len(history) - 1)
        y = pad_t + (hi - val) * (height - pad_t - pad_b) / span
        points.append((x, y))

    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    area = f"{pad_l},{height-pad_b} " + line + f" {points[-1][0]:.1f},{height-pad_b}"
    grid = ""
    for j in range(5):
        y = pad_t + j * (height - pad_t - pad_b) / 4
        label = hi - j * span / 4
        grid += f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width-pad_r}" y2="{y:.1f}" stroke="#1b3044" stroke-width="1"/>'
        grid += f'<text x="5" y="{y+4:.1f}" fill="#66809d" font-size="10" font-family="monospace">{label:.2f}</text>'

    last_x, last_y = points[-1]
    svg = f"""
    <div class="chart-wrap">
    <svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Simulated reference price chart">
      <defs>
        <linearGradient id="priceArea" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stop-color="#43dafa" stop-opacity=".28"/>
          <stop offset="100%" stop-color="#43dafa" stop-opacity="0"/>
        </linearGradient>
        <filter id="glow"><feGaussianBlur stdDeviation="3" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
      </defs>
      {grid}
      <polygon points="{area}" fill="url(#priceArea)"/>
      <polyline points="{line}" fill="none" stroke="#52e6ff" stroke-width="2.2" filter="url(#glow)"/>
      <line x1="{last_x:.1f}" y1="{last_y:.1f}" x2="{width-pad_r}" y2="{last_y:.1f}" stroke="#52e6ff" stroke-dasharray="4 5" opacity=".6"/>
      <circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="4" fill="#52e6ff" stroke="#eaffff" stroke-width="1.5"/>
      <text x="{width-105}" y="{last_y-9:.1f}" fill="#52e6ff" font-size="11" font-family="monospace">{history[-1]:.2f}</text>
      <text x="{pad_l}" y="{height-7}" fill="#66809d" font-size="10" font-family="monospace">-60 TICKS</text>
      <text x="{width-pad_r-65}" y="{height-7}" fill="#66809d" font-size="10" font-family="monospace">LATEST</text>
    </svg>
    </div>
    """
    st.markdown(svg, unsafe_allow_html=True)



# =========================================================
# CIRCULAR ANALYTICS
# =========================================================

def circular_metric(label, value, color):
    percentage = max(0, min(value, 100))

    st.markdown(f"""
    <div style="
        background:linear-gradient(145deg,#172641,#10192b);
        border:1px solid #2b4165;
        border-radius:16px;
        padding:18px 10px;
        text-align:center;
        margin-bottom:12px;
    ">
        <div style="
            width:94px;height:94px;border-radius:50%;
            margin:auto;display:grid;place-items:center;
            background:conic-gradient(
                {color} {percentage * 3.6}deg,
                #263653 0deg
            );
        ">
            <div style="
                width:73px;height:73px;border-radius:50%;
                background:#101a2c;display:grid;place-items:center;
                color:white;font-size:18px;font-weight:700;
            ">{percentage:.1f}%</div>
        </div>
        <div style="
            color:#b9c9e1;font-size:13px;
            margin-top:12px;font-weight:600;
        ">{label}</div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# SIMULATED ORDER BOOK
# =========================================================

@st.fragment(run_every="3s")
def order_book():
    base = st.session_state.get("demo_price", 100.0)
    bids = [(round(base - i * .05, 3), random.randint(12, 100)) for i in range(5)]
    asks = [(round(base + (i + 1) * .05, 3), random.randint(12, 100)) for i in range(5)]
    max_qty = max([q for _, q in bids + asks])

    html = """
    <div class="panel">
      <div class="panel-head">
        <span class="panel-title">DEPTH MATRIX</span>
        <span class="panel-tag">5 LEVELS / SIMULATED</span>
      </div>
      <div class="depth-row" style="color:#66809d">
        <span>BID SIZE</span><span>BID BAR</span><span style="text-align:center">SPREAD</span><span>ASK BAR</span><span>ASK SIZE</span>
      </div>
    """
    for i in range(5):
        bp, bq = bids[i]
        ap, aq = asks[i]
        bw = bq / max_qty * 100
        aw = aq / max_qty * 100
        html += f"""
        <div class="depth-row">
          <span style="color:#39dfa1">{bq:>4} · {bp:.3f}</span>
          <span><div class="depth-bar" style="width:{bw:.1f}%;background:linear-gradient(90deg,#168c72,#39dfa1);margin-left:auto"></div></span>
          <span style="text-align:center;color:#6d87a5">{ap-bp:.3f}</span>
          <span><div class="depth-bar" style="width:{aw:.1f}%;background:linear-gradient(90deg,#ff6685,#a83e62)"></div></span>
          <span style="color:#ff7a95">{ap:.3f} · {aq:>4}</span>
        </div>
        """
    html += "</div>"
    st.html(html)



# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("""
    <div style="padding:10px 0 22px 0;">
        <div style="
            font-family:'Space Grotesk';
            font-size:24px;font-weight:700;color:white;
        ">◈ LOB INTELLIGENCE</div>
        <div style="
            color:#63caff;font-size:10px;
            letter-spacing:2px;margin-top:5px;
        ">QUANT RESEARCH TERMINAL</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Forecast Explorer",
            "Explainability",
            "Performance Lab",
            "Architecture & Research"
        ]
    )

    st.divider()

    st.markdown("**ENGINE STATUS**")
    st.markdown("🟢 MODEL CHECKPOINT AVAILABLE")
    st.caption(f"Device: {DEVICE}")
    st.caption("Dataset: FI-2010")
    st.caption("Fold: 1")
    st.caption(
        f"Test windows: {MAX_INDEX + 1:,}"
        if DATA_AVAILABLE else "Test windows: Not configured"
    )
    if not DATA_AVAILABLE:
        st.caption("Historical inference: unavailable in this deployment")

    st.divider()
    st.caption("Historical research model")
    st.caption("No live exchange connection")


# =========================================================
# GLOBAL STATUS
# =========================================================

clock()


# =========================================================
# OVERVIEW
# =========================================================

if page == "Overview":

    st.markdown("""
    <div class="hero-box">
      <div class="eyebrow">LOB INTELLIGENCE / INSTITUTIONAL RESEARCH TERMINAL</div>
      <div class="hero-title">Order Flow Intelligence</div>
      <div class="hero-sub">Deep learning signals across five forecast horizons · CNN / BiGRU / Temporal Attention</div>
      <div style="margin-top:14px;display:flex;gap:12px;align-items:center;flex-wrap:wrap">
        <span class="status-pill">● MODEL ENGINE READY</span>
        <span class="micro-label">FI-2010 / NO-AUCTION / FOLD 1</span>
        <span class="micro-label">HISTORICAL REPLAY MODE</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    market_ticker()

    st.markdown("")
    left, right = st.columns([2.15, 1], gap="medium")

    with left:
        st.markdown('<div class="panel-head"><span class="panel-title">REFERENCE PRICE / MICROSTRUCTURE TRACE</span><span class="panel-tag">SYNTHETIC FEED · AUTO REFRESH</span></div>', unsafe_allow_html=True)
        price_chart()

    with right:
        st.markdown('<div class="panel-head"><span class="panel-title">MODEL PERFORMANCE</span><span class="panel-tag">HELD-OUT TEST</span></div>', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.metric("MACRO F1", "0.6091")
        m2.metric(
            "TEST WINDOWS",
            f"{MAX_INDEX + 1:,}" if DATA_AVAILABLE else "N/A"
        )
        m3, m4 = st.columns(2)
        m3.metric("1-STEP ACC.", "63.42%")
        m4.metric("10-STEP ACC.", "64.78%")
        st.markdown('<div class="micro-label" style="margin-top:10px">FIVE-HORIZON CLASSIFICATION / THREE CLASSES</div>', unsafe_allow_html=True)
        st.progress(0.6091, text="Overall Macro F1")

    st.markdown("")
    st.markdown('<div class="panel-head"><span class="panel-title">ORDER BOOK DEPTH</span><span class="panel-tag">SIMULATED BID / ASK LADDER</span></div>', unsafe_allow_html=True)
    order_book()

    st.markdown("")
    st.markdown('<div class="panel-head"><span class="panel-title">MODEL SIGNAL MATRIX</span><span class="panel-tag">ACTUAL FI-2010 INFERENCE · HISTORICAL WINDOW</span></div>', unsafe_allow_html=True)

    if DATA_AVAILABLE:
        if "overview_index" not in st.session_state:
            st.session_state.overview_index = min(1000, MAX_INDEX)

        index = min(st.session_state.overview_index, MAX_INDEX)
        probs, attention, predicted, actual = run_prediction(index)

        cards = st.columns(5, gap="small")
        for i, col in enumerate(cards):
            prediction = CLASSES[int(predicted[i])]
            confidence = float(probs[i].max())
            color = {"UP": "#39e2a1", "DOWN": "#ff597d", "STATIONARY": "#f4c66b"}[prediction]
            with col:
                st.markdown(f"""
                <div class="signal" style="--accent:{color}">
                  <div class="signal-h">FORECAST / T+{HORIZONS[i]}</div>
                  <div class="signal-v">{prediction}</div>
                  <div class="signal-p">{confidence:.1%} MODEL PROBABILITY</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("")
        index_new = st.slider("Historical replay index", 0, MAX_INDEX, int(index), key="overview_sample_slider")
        if index_new != index:
            st.session_state.overview_index = index_new
            st.rerun()

        c1, c2 = st.columns([1.35, 1], gap="medium")
        with c1:
            st.markdown("#### Probability surface")
            prob_df = pd.DataFrame(probs, columns=CLASSES, index=[f"T+{h}" for h in HORIZONS])
            st.bar_chart(prob_df, height=230)
        with c2:
            st.markdown("#### Window diagnostics")
            st.metric("Correct horizons", f"{int(np.sum(predicted == actual))} / 5")
            st.metric("Sequence tensor", "100 × 144")
            st.metric("Attention mass · last 5", f"{attention[-5:].sum():.2%}")
    else:
        st.info(
            "Historical FI-2010 inference is temporarily unavailable in this "
            "public deployment because the dataset has not been configured. "
            "The trained model checkpoint is loaded, but no prediction is "
            "shown without the required historical input data."
        )

    st.markdown("""
    <div class="footnote">
      DATA DISCLOSURE · Price trace, ticker values and depth ladder are simulated interface data.
      Prediction signals are produced by the trained model from historical FI-2010 test windows.
      No live exchange feed is connected.
    </div>
    """, unsafe_allow_html=True)

elif page == "Forecast Explorer":

    st.title("Forecast Explorer")

    st.caption(
        "Inspect predictions on historical FI-2010 test observations."
    )

    st.info(
        "This is historical model inference, not live market forecasting."
    )

    if not DATA_AVAILABLE:
        st.warning(
            "Historical FI-2010 test data is not configured in this deployment. "
            "The forecast explorer will be available when the dataset is added."
        )
    else:
        index = st.slider(
            "Test sample",
            0,
            MAX_INDEX,
            min(1000, MAX_INDEX)
        )

        probs, attention, predicted, actual = run_prediction(index)

        rows = []

        for i, horizon in enumerate(HORIZONS):
            rows.append({
                "Horizon": horizon,
                "Prediction": CLASSES[int(predicted[i])],
                "Actual": CLASSES[int(actual[i])],
                "Up": probs[i, 0],
                "Stationary": probs[i, 1],
                "Down": probs[i, 2],
                "Correct": bool(predicted[i] == actual[i])
            })

        st.dataframe(
            pd.DataFrame(rows).style.format({
                "Up": "{:.2%}",
                "Stationary": "{:.2%}",
                "Down": "{:.2%}"
            }),
            hide_index=True,
            use_container_width=True
        )

        horizon = st.selectbox("Select horizon", HORIZONS, index=4)
        h = HORIZONS.index(horizon)

        st.markdown("### Class Probabilities")

        st.bar_chart(
            pd.DataFrame({
                "Class": CLASSES,
                "Probability": probs[h]
            }),
            x="Class",
            y="Probability"
        )


# =========================================================
# EXPLAINABILITY
# =========================================================

elif page == "Explainability":

    st.title("Temporal Explainability")

    st.info(
        "Attention weights show learned temporal weighting, not causal explanations."
    )

    if not DATA_AVAILABLE:
        st.warning(
            "Temporal attention visualizations require the FI-2010 test dataset, "
            "which is not configured in this deployment."
        )
    else:
        index = st.slider(
            "Select sample",
            0,
            MAX_INDEX,
            min(1000, MAX_INDEX),
            key="explainability_index"
        )

        probs, attention, predicted, actual = run_prediction(index)

        attention_df = pd.DataFrame({
            "Time Step": np.arange(1, 101),
            "Attention": attention
        })

        st.markdown("### Temporal Attention Profile")

        st.line_chart(
            attention_df,
            x="Time Step",
            y="Attention",
            height=350
        )

        a, b, c = st.columns(3)

        a.metric("Final Step Attention", f"{attention[-1] * 100:.2f}%")
        b.metric("Final Five Steps", f"{attention[-5:].sum() * 100:.2f}%")
        c.metric("Attention Sum", f"{attention.sum():.4f}")


# =========================================================
# PERFORMANCE LAB
# =========================================================

elif page == "Performance Lab":

    st.title("Performance Lab")

    st.caption(
        "Previously computed complete held-out Fold 1 test metrics."
    )

    performance = pd.DataFrame({
        "Horizon": ["1", "2", "3", "5", "10"],
        "Accuracy": [0.6342, 0.6018, 0.6083, 0.6207, 0.6478],
        "Macro F1": [0.5963, 0.5889, 0.6059, 0.6209, 0.6337]
    })

    a, b = st.columns(2)
    a.metric("Overall Macro F1", "0.6091")
    b.metric("Test Sequences", "38,298")

    st.markdown("### Accuracy by Horizon")
    st.bar_chart(performance, x="Horizon", y="Accuracy")

    st.markdown("### Macro F1 by Horizon")
    st.bar_chart(performance, x="Horizon", y="Macro F1")

    st.dataframe(
        performance.style.format({
            "Accuracy": "{:.2%}",
            "Macro F1": "{:.4f}"
        }),
        hide_index=True,
        use_container_width=True
    )


# =========================================================
# ARCHITECTURE & RESEARCH
# =========================================================

elif page == "Architecture & Research":

    st.title("Architecture & Research")

    st.markdown("### Model Architecture")

    architecture = [
        ("Input", "100 time steps × 144 features"),
        ("CNN", "Conv1D: 144 → 64 → 64"),
        ("BiGRU", "Bidirectional GRU, hidden size 64"),
        ("Attention", "128 → 64 → 1"),
        ("Context", "Weighted temporal representation"),
        ("Classifier", "128 → 128 → 15"),
        ("Output", "5 horizons × 3 classes")
    ]

    for title, description in architecture:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(description)

    st.divider()

    st.markdown("### Dataset")

    st.write("""
    - Dataset: FI-2010
    - Subset: NoAuction
    - Fold: 1
    - Input features: 144
    - Sequence length: 100
    - Horizons: 1, 2, 3, 5, 10
    - Classes: Up, Stationary, Down
    """)

    st.divider()

    st.markdown("### Research Limitations")

    st.write("""
    - No live exchange connection.
    - No real-time market-data ingestion.
    - No exact future-price regression.
    - No guaranteed prediction accuracy.
    - Attention weights are not causal explanations.
    """)

    st.markdown("### Future Research Directions")

    st.write("""
    - Real-time order-book integration.
    - Uncertainty-aware forecasting.
    - Distribution-shift analysis.
    - Cross-asset generalization.
    - Mid-price regression and prediction intervals.
    """)

    st.markdown("### Checkpoint Information")

    st.json({
        "epoch": checkpoint.get("epoch", "N/A"),
        "validation_macro_f1": checkpoint.get("val_macro_f1", None),
        "device": str(DEVICE)
    })


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown("""
<div style="
    text-align:center;
    color:#7186a8;
    padding:15px;
    font-size:12px;
">
    ◈ LOB INTELLIGENCE · QUANT RESEARCH TERMINAL<br>
    CNN · BIDIRECTIONAL GRU · TEMPORAL ATTENTION<br>
    HISTORICAL FI-2010 MODEL · SIMULATED MARKET VISUALS
</div>
""", unsafe_allow_html=True)
