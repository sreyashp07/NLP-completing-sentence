"""CustomerIntent AI - Hugging Face Spaces standalone version."""
import sys
import os
import re
import time
import pickle
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import nltk
NLTK_DIR = Path("/tmp/nltk_data")
NLTK_DIR.mkdir(exist_ok=True)
os.environ["NLTK_DATA"] = str(NLTK_DIR)
nltk.data.path.insert(0, str(NLTK_DIR))
for resource in ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger", "punkt_tab"]:
    try:
        nltk.download(resource, download_dir=str(NLTK_DIR), quiet=True)
    except Exception:
        pass

import streamlit as st

st.set_page_config(
    page_title="CustomerIntent",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

import plotly.graph_objects as go
import plotly.express as px

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

TMP_MODEL_DIR = Path("/tmp/customerintent_model")
TMP_PIPELINE  = TMP_MODEL_DIR / "pipeline.pkl"
TMP_LE        = TMP_MODEL_DIR / "label_encoder.pkl"

LOCAL_MODEL_DIR = ROOT / "ml" / "saved_models" / "baseline"
LOCAL_PIPELINE  = LOCAL_MODEL_DIR / "pipeline.pkl"
LOCAL_LE        = LOCAL_MODEL_DIR / "label_encoder.pkl"

ACCENT_COLORS = {
    "critical": "#E63946",
    "high":     "#F4A261",
    "medium":   "#2A9D8F",
    "low":      "#264653",
}

INTENT_META = {
    "payment_issue":       {"department": "Billing Team",     "priority": "critical", "label": "Payment Issue"},
    "refund_request":      {"department": "Billing Team",     "priority": "high",     "label": "Refund Request"},
    "account_locked":      {"department": "Security Team",    "priority": "critical", "label": "Account Locked"},
    "technical_bug":       {"department": "Engineering Team", "priority": "high",     "label": "Technical Bug"},
    "feature_request":     {"department": "Product Team",     "priority": "low",      "label": "Feature Request"},
    "subscription_cancel": {"department": "Retention Team",   "priority": "high",     "label": "Subscription Cancel"},
    "invoice_problem":     {"department": "Finance Team",     "priority": "medium",   "label": "Invoice Problem"},
    "shipping_delay":      {"department": "Logistics Team",   "priority": "medium",   "label": "Shipping Delay"},
    "general_inquiry":     {"department": "General Support",  "priority": "low",      "label": "General Inquiry"},
}

EXAMPLES = [
    "My payment failed but money was deducted",
    "I want to cancel my subscription",
    "App keeps crashing on dashboard",
    "Need GST invoice for last month",
    "Package not delivered, it's been 7 days",
    "Account locked after wrong password",
    "Please add dark mode to mobile app",
    "Refund not received after 10 days",
]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, .stApp {
    background: #0A0E1A !important;
    font-family: 'Space Grotesk', -apple-system, sans-serif;
    color: #DDE3EC;
}

.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(circle at 15% 25%, rgba(64, 134, 245, 0.06) 0%, transparent 40%),
        radial-gradient(circle at 85% 75%, rgba(168, 85, 247, 0.05) 0%, transparent 40%);
    pointer-events: none;
    z-index: 0;
}

[data-testid="stSidebar"] {
    background: #0D1220 !important;
    border-right: 1px solid #1A2332;
}
[data-testid="stSidebar"] * { color: #B8C2D1 !important; font-family: 'Space Grotesk', sans-serif !important; }

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

[data-testid="metric-container"] {
    background: #131A2A;
    border: 1px solid #1F2A3D;
    border-radius: 10px;
    padding: 16px 20px;
    transition: all 0.2s ease;
}
[data-testid="metric-container"]:hover {
    border-color: #4086F5;
    background: #161E30;
}
[data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600;
    font-size: 1.15rem !important;
    color: #FFFFFF !important;
    letter-spacing: -0.2px;
}
[data-testid="stMetricLabel"] {
    color: #6B7A94 !important;
    font-size: 0.68rem !important;
    text-transform: uppercase;
    letter-spacing: 1.3px;
    font-weight: 500;
    font-family: 'IBM Plex Mono', monospace !important;
}

.stButton > button {
    background: #1A2332;
    color: #DDE3EC !important;
    border: 1px solid #2A3548;
    border-radius: 8px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 500;
    font-size: 0.83rem;
    padding: 9px 16px;
    letter-spacing: 0.2px;
    transition: all 0.18s ease;
}
.stButton > button:hover {
    background: #1F2A3D;
    border-color: #4086F5;
    color: #FFFFFF !important;
}
.stButton > button:active {
    transform: scale(0.98);
}

button[kind="primary"] {
    background: #4086F5 !important;
    color: white !important;
    border: 1px solid #4086F5 !important;
    font-weight: 600 !important;
}
button[kind="primary"]:hover {
    background: #2E6DD9 !important;
    border-color: #2E6DD9 !important;
}

.stTextArea textarea {
    background: #131A2A !important;
    border: 1px solid #1F2A3D !important;
    border-radius: 10px !important;
    color: #FFFFFF !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.95rem !important;
    transition: all 0.2s ease;
}
.stTextArea textarea:focus {
    border-color: #4086F5 !important;
    box-shadow: 0 0 0 3px rgba(64, 134, 245, 0.12) !important;
}

[data-testid="stRadio"] label {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.87rem !important;
}

hr { border-color: #1A2332 !important; margin: 1rem 0 !important; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1F2A3D; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2A3548; }

.main-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: -1.5px;
    margin: 0;
    line-height: 1.1;
}
.main-title .accent {
    color: #4086F5;
}

.subtitle {
    color: #6B7A94;
    font-size: 0.9rem;
    margin-top: 8px;
    font-weight: 400;
    font-family: 'Space Grotesk', sans-serif;
}

.section-label {
    font-size: 0.66rem;
    font-weight: 500;
    color: #6B7A94;
    text-transform: uppercase;
    letter-spacing: 1.6px;
    margin-bottom: 14px;
    font-family: 'IBM Plex Mono', monospace;
}

.brand-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: -0.4px;
}

.brand-version {
    font-size: 0.66rem;
    color: #6B7A94;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.3px;
    margin-top: 4px;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 11px;
    border-radius: 6px;
    font-size: 0.74rem;
    font-weight: 500;
    background: rgba(42, 157, 143, 0.12);
    border: 1px solid rgba(42, 157, 143, 0.25);
    color: #2A9D8F;
    font-family: 'IBM Plex Mono', monospace;
}

.keyword-chip {
    display: inline-block;
    background: #131A2A;
    border: 1px solid #1F2A3D;
    border-radius: 5px;
    padding: 4px 10px;
    margin: 3px 3px 3px 0;
    font-size: 11.5px;
    color: #8AB4F8;
    font-family: 'IBM Plex Mono', monospace;
    transition: all 0.15s ease;
}
.keyword-chip:hover {
    border-color: #4086F5;
    color: #FFFFFF;
}

.divider-line {
    height: 1px;
    background: #1A2332;
    margin: 24px 0;
}

.system-info {
    font-size: 0.7rem;
    color: #6B7A94;
    line-height: 2;
    font-family: 'IBM Plex Mono', monospace;
}
.system-info-header {
    color: #8AB4F8;
    font-weight: 500;
    margin-bottom: 6px;
    letter-spacing: 1px;
    font-size: 0.66rem;
}

div[data-baseweb="spinner"] > div {
    border-color: #4086F5 transparent transparent transparent !important;
}

.stProgress > div > div > div > div {
    background: #4086F5 !important;
}

.priority-badge {
    display: inline-block;
    border-radius: 6px;
    padding: 5px 12px;
    font-size: 10.5px;
    font-weight: 600;
    letter-spacing: 1.1px;
    font-family: 'IBM Plex Mono', monospace;
}

.empty-state {
    text-align: center;
    padding: 80px 20px;
    color: #6B7A94;
}
.empty-state-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 500;
    color: #8A95AA;
    margin-bottom: 8px;
}
.empty-state-subtitle {
    font-size: 0.85rem;
    color: #6B7A94;
}
</style>
""", unsafe_allow_html=True)


def get_model_paths():
    if TMP_PIPELINE.exists() and TMP_LE.exists():
        return TMP_PIPELINE, TMP_LE
    if LOCAL_PIPELINE.exists() and LOCAL_LE.exists():
        return LOCAL_PIPELINE, LOCAL_LE
    return None, None


def model_exists() -> bool:
    p, _ = get_model_paths()
    return p is not None


def train_model() -> bool:
    TMP_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    placeholder = st.empty()
    with placeholder.container():
        st.markdown('<h1 class="main-title">First-Time <span class="accent">Setup</span></h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Training the intent classifier — takes about 2 minutes on first run only.</p>', unsafe_allow_html=True)
        prog = st.progress(0)
        status = st.empty()
        try:
            status.text("Generating training data (1800 samples, 9 classes)...")
            prog.progress(5)
            env = os.environ.copy()
            env["NLTK_DATA"] = str(NLTK_DIR)
            r1 = subprocess.run(
                [sys.executable, str(ROOT / "data" / "generate_dataset.py")],
                capture_output=True, text=True, cwd=str(ROOT), timeout=180, env=env,
            )
            if r1.returncode != 0:
                st.error(f"Dataset generation failed:\n```\n{r1.stderr[-800:]}\n```")
                return False
            prog.progress(35)
            status.text("Training TF-IDF + Logistic Regression...")
            r2 = subprocess.run(
                [sys.executable, str(ROOT / "ml" / "training" / "train_baseline.py")],
                capture_output=True, text=True, cwd=str(ROOT), timeout=600, env=env,
            )
            if r2.returncode != 0:
                st.error(f"Training failed:\n```\n{r2.stderr[-800:]}\n```")
                return False
            prog.progress(80)
            status.text("Copying artifacts for persistence...")
            import shutil
            for fname in ["pipeline.pkl", "label_encoder.pkl", "metrics.yaml", "classes.txt"]:
                src = LOCAL_MODEL_DIR / fname
                if src.exists():
                    shutil.copy2(src, TMP_MODEL_DIR / fname)
            if not TMP_PIPELINE.exists():
                st.error("Model file not found after training")
                return False
            prog.progress(100)
            status.text("Complete")
            time.sleep(1)
        except subprocess.TimeoutExpired:
            st.error("Training timed out")
            return False
        except Exception as e:
            st.error(f"Error: {e}")
            return False
    placeholder.empty()
    return True


@st.cache_resource(show_spinner="Initializing model...")
def load_model():
    p, l = get_model_paths()
    if p is None:
        return None, None
    try:
        with open(p, "rb") as f:
            pipeline = pickle.load(f)
        with open(l, "rb") as f:
            le = pickle.load(f)
        return pipeline, le
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None, None


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    try:
        from nltk.tokenize import word_tokenize
        from nltk.stem import WordNetLemmatizer
        tokens = word_tokenize(text)
        lem = WordNetLemmatizer()
        tokens = [lem.lemmatize(t) for t in tokens if len(t) > 1]
        return " ".join(tokens)
    except Exception:
        return text


def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    try:
        from nltk.tokenize import word_tokenize
        from nltk.stem import WordNetLemmatizer
        from nltk.corpus import stopwords
        stop = set(stopwords.words("english"))
        lem = WordNetLemmatizer()
        tokens = [lem.lemmatize(t) for t in word_tokenize(text.lower())
                  if len(t) > 2 and t not in stop]
        freq: Dict = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        return [t for t, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]]
    except Exception:
        return []


def predict(text: str) -> Optional[Dict]:
    pipeline, le = load_model()
    if pipeline is None:
        return None
    try:
        start = time.time()
        cleaned = clean_text(text)
        probs = pipeline.predict_proba([cleaned])[0]
        ranked = sorted(zip(le.classes_, probs), key=lambda x: x[1], reverse=True)
        intent, conf = ranked[0]
        meta = INTENT_META.get(intent, {"department": "General Support", "priority": "low", "label": intent})
        return {
            "primary_intent": {
                "intent": intent, "confidence": float(conf),
                "department": meta["department"], "priority": meta["priority"],
                "display_label": meta["label"],
            },
            "top_intents": [
                {"intent": i, "confidence": float(c),
                 "display_label": INTENT_META.get(i, {}).get("label", i)}
                for i, c in ranked[:3]
            ],
            "keywords": extract_keywords(text),
            "model_used": "TF-IDF + Logistic Regression",
            "processing_time_ms": round((time.time() - start) * 1000, 2),
        }
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


if "history" not in st.session_state:
    st.session_state.history: List[Dict] = []
if "last_pred" not in st.session_state:
    st.session_state.last_pred = None
if "prefilled_text" not in st.session_state:
    st.session_state.prefilled_text = ""


def add_ticket(text: str, pred: Dict):
    n = len(st.session_state.history) + 1001
    st.session_state.history.insert(0, {
        "id": f"TKT-{n:04d}",
        "ts": datetime.now().strftime("%H:%M"),
        "date": datetime.now().strftime("%b %d"),
        "text": text[:100] + ("..." if len(text) > 100 else ""),
        **pred["primary_intent"],
    })


if not model_exists():
    ok = train_model()
    if ok:
        st.cache_resource.clear()
        st.rerun()
    else:
        st.error("Setup failed. Please restart the Space.")
        st.stop()


with st.sidebar:
    st.markdown("""
    <div style="padding:14px 0 18px 0;">
        <div class="brand-title">CustomerIntent</div>
        <div class="brand-version">v1.0.0 · production</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="status-pill">● Model Ready</div>', unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Live Classifier", "Analytics", "Ticket History"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown("""
    <div class="system-info">
    <div class="system-info-header">SYSTEM</div>
    model  &nbsp;&nbsp;tfidf+lr<br>
    intents&nbsp;&nbsp;9 classes<br>
    env    &nbsp;&nbsp;&nbsp;&nbsp;hf spaces<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Clear", use_container_width=True):
            st.session_state.history = []
            st.session_state.last_pred = None
            st.session_state.prefilled_text = ""
            st.rerun()
    with col_b:
        ticket_count = len(st.session_state.history)
        st.markdown(
            f'<div style="text-align:center;padding:8px;background:#131A2A;'
            f'border:1px solid #1F2A3D;border-radius:6px;font-size:0.74rem;'
            f'color:#6B7A94;font-family:IBM Plex Mono,monospace;">'
            f'{ticket_count} tickets</div>',
            unsafe_allow_html=True,
        )


if "Live Classifier" in page:
    st.markdown("""
    <div style="padding:12px 0 24px 0;">
        <h1 class="main-title">Live Intent <span class="accent">Classifier</span></h1>
        <p class="subtitle">Real-time customer support classification with smart routing</p>
    </div>
    """, unsafe_allow_html=True)

    lc, rc = st.columns([3, 2], gap="large")
    with lc:
        txt = st.text_area(
            "message",
            value=st.session_state.prefilled_text,
            placeholder="Type a customer support message...",
            height=140, key="inp_field", label_visibility="collapsed",
        )
        btn = st.button("Classify Intent", type="primary")

    with rc:
        st.markdown('<div class="section-label">Quick Examples</div>', unsafe_allow_html=True)
        for ex in EXAMPLES:
            if st.button(ex[:46] + ('...' if len(ex)>46 else ''),
                        key=f"e{hash(ex)}", use_container_width=True):
                st.session_state.prefilled_text = ex
                st.rerun()

    if btn:
        if not txt or len(txt.strip()) < 5:
            st.warning("Enter at least 5 characters")
        else:
            with st.spinner("Analyzing intent..."):
                pred = predict(txt)
            if pred:
                st.session_state.last_pred = pred
                add_ticket(txt, pred)
                st.session_state.prefilled_text = ""

    if st.session_state.last_pred:
        p = st.session_state.last_pred
        pi = p["primary_intent"]

        st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Intent Detected", pi["display_label"])
        c2.metric("Routed To",       pi["department"])
        c3.metric("Priority",        pi["priority"].upper())
        c4.metric("Confidence",      f"{pi['confidence']*100:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)

        cc, ic = st.columns([3, 2], gap="large")

        with cc:
            st.markdown('<div class="section-label">Confidence Distribution</div>', unsafe_allow_html=True)
            top = p["top_intents"]
            values = [round(t["confidence"]*100, 2) for t in top]
            fig = go.Figure(go.Bar(
                x=values,
                y=[t["display_label"] for t in top],
                orientation="h",
                marker=dict(
                    color=["#4086F5", "#2A5DB0", "#1F3F7A"],
                    line=dict(width=0),
                ),
                text=[f"{v:.1f}%" for v in values],
                textposition="outside",
                textfont=dict(color="#FFFFFF", size=12, family="Space Grotesk"),
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#6B7A94", family="Space Grotesk"),
                height=200,
                margin=dict(l=0, r=40, t=8, b=8),
                xaxis=dict(showgrid=False, showticklabels=False, range=[0, 115]),
                yaxis=dict(
                    showgrid=False,
                    tickfont=dict(size=12, color="#DDE3EC", family="Space Grotesk"),
                ),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with ic:
            st.markdown('<div class="section-label">Key Signals</div>', unsafe_allow_html=True)
            if p.get("keywords"):
                kw_html = "".join([
                    f'<span class="keyword-chip">{kw}</span>'
                    for kw in p["keywords"]
                ])
                st.markdown(kw_html, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            pc = ACCENT_COLORS.get(pi["priority"], "#6B7A94")
            st.markdown(
                f'<div style="margin-top:8px;">'
                f'<span class="priority-badge" style="background:{pc}1A;'
                f'border:1px solid {pc}40;color:{pc};">'
                f'{pi["priority"].upper()} PRIORITY</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<p style="color:#6B7A94;font-size:10.5px;margin-top:14px;'
                f'font-family:IBM Plex Mono,monospace;letter-spacing:0.2px;">'
                f'{p["processing_time_ms"]}ms · {p["model_used"]}</p>',
                unsafe_allow_html=True,
            )


elif "Analytics" in page:
    st.markdown("""
    <div style="padding:12px 0 24px 0;">
        <h1 class="main-title"><span class="accent">Analytics</span></h1>
        <p class="subtitle">Real-time insights from classified tickets</p>
    </div>
    """, unsafe_allow_html=True)

    h = st.session_state.history
    if not h:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-title">No data yet</div>
            <div class="empty-state-subtitle">Classify some messages to see analytics</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Tickets",  len(h))
        m2.metric("Critical",       sum(1 for t in h if t["priority"]=="critical"))
        m3.metric("Avg Confidence", f"{sum(t['confidence'] for t in h)/len(h)*100:.1f}%")
        m4.metric("Departments",    len(set(t["department"] for t in h)))

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="large")

        with c1:
            st.markdown('<div class="section-label">Intent Distribution</div>', unsafe_allow_html=True)
            ic: Dict = {}
            for t in h:
                ic[t["display_label"]] = ic.get(t["display_label"], 0) + 1
            fig = px.pie(
                names=list(ic.keys()), values=list(ic.values()), hole=0.6,
                color_discrete_sequence=[
                    "#4086F5","#5B9BF8","#2E6DD9","#1F4FA8",
                    "#7AB1FB","#3A78E0","#1A3D80","#5181E5","#2553B8",
                ],
            )
            fig.update_traces(
                textfont=dict(family="Space Grotesk", size=11, color="white"),
                marker=dict(line=dict(color="#0A0E1A", width=2)),
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#6B7A94", family="Space Grotesk"),
                legend=dict(font=dict(size=10, color="#8A95AA", family="Space Grotesk")),
                height=320, margin=dict(l=0, r=0, t=10, b=10),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with c2:
            st.markdown('<div class="section-label">Priority Breakdown</div>', unsafe_allow_html=True)
            pc = {"critical": 0, "high": 0, "medium": 0, "low": 0}
            for t in h:
                if t["priority"] in pc:
                    pc[t["priority"]] += 1
            fig2 = go.Figure(go.Bar(
                x=list(pc.keys()), y=list(pc.values()),
                marker=dict(
                    color=[ACCENT_COLORS[p] for p in pc],
                    line=dict(width=0),
                ),
                text=list(pc.values()),
                textposition="outside",
                textfont=dict(color="#DDE3EC", size=13, family="Space Grotesk"),
            ))
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#6B7A94", family="Space Grotesk"),
                height=320, margin=dict(l=0, r=0, t=10, b=10),
                xaxis=dict(showgrid=False, tickfont=dict(size=11, color="#8A95AA", family="Space Grotesk")),
                yaxis=dict(showgrid=False, showticklabels=False),
                bargap=0.5,
            )
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})


elif "Ticket History" in page:
    st.markdown("""
    <div style="padding:12px 0 24px 0;">
        <h1 class="main-title">Ticket <span class="accent">History</span></h1>
        <p class="subtitle">All classified tickets from this session</p>
    </div>
    """, unsafe_allow_html=True)

    h = st.session_state.history
    if not h:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-title">No tickets yet</div>
            <div class="empty-state-subtitle">Classify messages to populate ticket history</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for t in h:
            pc = ACCENT_COLORS.get(t["priority"], "#6B7A94")
            conf_pct = t["confidence"] * 100
            st.markdown(f"""
            <div style="background:#131A2A;
            border-left:3px solid {pc};
            border-radius:0 10px 10px 0;padding:14px 18px;margin:10px 0;
            border-top:1px solid #1F2A3D;
            border-right:1px solid #1F2A3D;
            border-bottom:1px solid #1F2A3D;
            transition: all 0.2s ease;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                    <span style="font-family:IBM Plex Mono,monospace;font-size:11.5px;
                    color:#8AB4F8;font-weight:500;letter-spacing:0.3px;">{t['id']}</span>
                    <span style="color:#6B7A94;font-size:10.5px;font-family:IBM Plex Mono,monospace;">
                    {t['date']} · {t['ts']}</span>
                </div>
                <div style="color:#DDE3EC;font-size:0.9rem;margin:8px 0 12px 0;
                line-height:1.5;font-weight:400;">{t['text']}</div>
                <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;">
                    <span style="background:#1A2332;border:1px solid #2A3548;color:#8AB4F8;
                    border-radius:5px;padding:4px 10px;font-size:11px;font-weight:500;
                    font-family:Space Grotesk,sans-serif;">{t['display_label']}</span>
                    <span style="background:#1A2332;border:1px solid #2A3548;color:#A8B5C8;
                    border-radius:5px;padding:4px 10px;font-size:11px;font-weight:500;
                    font-family:Space Grotesk,sans-serif;">{t['department']}</span>
                    <span style="background:{pc}1A;border:1px solid {pc}40;
                    color:{pc};border-radius:5px;padding:4px 10px;
                    font-size:10.5px;font-weight:600;letter-spacing:0.7px;
                    font-family:IBM Plex Mono,monospace;">{t['priority'].upper()}</span>
                    <span style="color:#6B7A94;font-size:11px;font-family:IBM Plex Mono,monospace;
                    margin-left:auto;letter-spacing:0.2px;">{conf_pct:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
