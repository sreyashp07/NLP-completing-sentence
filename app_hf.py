"""
CustomerIntent AI - Hugging Face Spaces Version
Trains model on startup if not present, then serves predictions.
"""
import os
import sys
import pickle
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Download NLTK data first
import nltk
for resource in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'punkt_tab']:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CustomerIntent AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_DIR = Path("ml/saved_models/baseline")
PIPELINE_PATH = MODEL_DIR / "pipeline.pkl"
LE_PATH = MODEL_DIR / "label_encoder.pkl"

PRIORITY_COLORS = {
    "critical": "#FF2D55", "high": "#FF6B35",
    "medium": "#FFD700", "low": "#34C759",
}
PRIORITY_EMOJI = {
    "critical": "🚨", "high": "🔴", "medium": "🟡", "low": "🟢"
}
INTENT_EMOJI = {
    "payment_issue": "💳", "refund_request": "💰",
    "account_locked": "🔒", "technical_bug": "🐛",
    "feature_request": "⭐", "subscription_cancel": "❌",
    "invoice_problem": "📄", "shipping_delay": "📦",
    "general_inquiry": "❓",
}
INTENT_METADATA = {
    "payment_issue":       {"department": "Billing Team",    "priority": "critical", "label": "Payment Issue"},
    "refund_request":      {"department": "Billing Team",    "priority": "high",     "label": "Refund Request"},
    "account_locked":      {"department": "Security Team",   "priority": "critical", "label": "Account Locked"},
    "technical_bug":       {"department": "Engineering Team","priority": "high",     "label": "Technical Bug"},
    "feature_request":     {"department": "Product Team",    "priority": "low",      "label": "Feature Request"},
    "subscription_cancel": {"department": "Retention Team",  "priority": "high",     "label": "Subscription Cancel"},
    "invoice_problem":     {"department": "Finance Team",    "priority": "medium",   "label": "Invoice Problem"},
    "shipping_delay":      {"department": "Logistics Team",  "priority": "medium",   "label": "Shipping Delay"},
    "general_inquiry":     {"department": "General Support", "priority": "low",      "label": "General Inquiry"},
}
EXAMPLE_MESSAGES = [
    "My payment failed but money was deducted",
    "I want to cancel my subscription",
    "App keeps crashing on dashboard",
    "Need GST invoice for last month",
    "Package not delivered, it's been 7 days",
    "Account locked after wrong password attempts",
    "Please add dark mode to the mobile app",
    "Refund not received after 10 days",
]

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&display=swap');
html, body, .stApp { background-color: #080C14 !important; font-family: 'Sora', sans-serif; color: #E8EDF5; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0D1526 0%, #080C14 100%) !important; border-right: 1px solid #1C2A3A; }
[data-testid="stSidebar"] * { color: #C8D3E0 !important; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="metric-container"] { background: linear-gradient(135deg, #0F1B2D 0%, #0A1520 100%); border: 1px solid #1C2A3A; border-radius: 12px; padding: 16px 20px; }
[data-testid="stMetricValue"] { font-weight: 700; font-size: 1.1rem !important; color: #E8EDF5 !important; }
[data-testid="stMetricLabel"] { color: #6B7C93 !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 1px; }
.stButton > button { background: linear-gradient(135deg, #0066FF, #0044CC); color: white !important; border: none; border-radius: 8px; font-weight: 600; }
.stTextArea textarea { background: #0D1526 !important; border: 1px solid #1C2A3A !important; border-radius: 10px !important; color: #E8EDF5 !important; }
hr { border-color: #1C2A3A !important; }
</style>
""", unsafe_allow_html=True)


# ── Model Training on Startup ──────────────────────────────────────────────────
def ensure_model_exists():
    """Train model if artifacts don't exist."""
    if PIPELINE_PATH.exists() and LE_PATH.exists():
        return True

    st.info("🔧 First run: Training the model... This takes about 30 seconds.")
    progress = st.progress(0)
    status = st.empty()

    try:
        status.text("📊 Generating training dataset...")
        progress.progress(10)
        result = subprocess.run(
            [sys.executable, "data/generate_dataset.py"],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            st.error(f"Dataset generation failed: {result.stderr}")
            return False

        progress.progress(40)
        status.text("🤖 Training TF-IDF + Logistic Regression model...")

        result = subprocess.run(
            [sys.executable, "ml/training/train_baseline.py"],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            st.error(f"Training failed: {result.stderr}")
            return False

        progress.progress(100)
        status.text("✅ Model trained successfully!")
        time.sleep(1)
        progress.empty()
        status.empty()
        st.success("Model ready! Reloading...")
        st.rerun()
        return True

    except subprocess.TimeoutExpired:
        st.error("Training timed out. Please refresh the page.")
        return False
    except Exception as e:
        st.error(f"Error: {e}")
        return False


# ── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not PIPELINE_PATH.exists():
        return None, None
    with open(PIPELINE_PATH, "rb") as f:
        pipeline = pickle.load(f)
    with open(LE_PATH, "rb") as f:
        le = pickle.load(f)
    return pipeline, le


@st.cache_resource
def get_cleaner():
    from ml.preprocessing.text_cleaner import TextCleaner
    return TextCleaner(remove_stopwords=False, lemmatize=True)


# ── Predict ────────────────────────────────────────────────────────────────────
def predict(text: str) -> Optional[Dict]:
    pipeline, le = load_model()
    if pipeline is None:
        return None

    from ml.preprocessing.text_cleaner import extract_keywords
    cleaner = get_cleaner()
    start = time.time()
    cleaned = cleaner.clean(text)
    probs = pipeline.predict_proba([cleaned])[0]
    classes = le.classes_
    ranked = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
    primary_intent, primary_conf = ranked[0]
    meta = INTENT_METADATA.get(primary_intent, {
        "department": "General Support", "priority": "low", "label": primary_intent
    })
    elapsed_ms = (time.time() - start) * 1000

    return {
        "primary_intent": {
            "intent": primary_intent,
            "confidence": float(primary_conf),
            "department": meta["department"],
            "priority": meta["priority"],
            "display_label": meta["label"],
        },
        "top_intents": [
            {"intent": i, "confidence": float(c),
             "display_label": INTENT_METADATA.get(i, {}).get("label", i)}
            for i, c in ranked[:3]
        ],
        "keywords": extract_keywords(text, top_n=5),
        "model_used": "TF-IDF + Logistic Regression",
        "processing_time_ms": round(elapsed_ms, 2),
    }


# ── Session State ──────────────────────────────────────────────────────────────
if "ticket_history" not in st.session_state:
    st.session_state.ticket_history: List[Dict] = []
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


def add_to_history(text: str, prediction: Dict):
    ticket_num = len(st.session_state.ticket_history) + 1001
    ticket = {
        "id": f"TKT-{ticket_num:04d}",
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "date": datetime.now().strftime("%b %d"),
        "text": text[:100] + ("..." if len(text) > 100 else ""),
        "intent": prediction["primary_intent"]["intent"],
        "display_label": prediction["primary_intent"]["display_label"],
        "department": prediction["primary_intent"]["department"],
        "priority": prediction["primary_intent"]["priority"],
        "confidence": prediction["primary_intent"]["confidence"],
    }
    st.session_state.ticket_history.insert(0, ticket)


# ── Check model exists, train if not ──────────────────────────────────────────
if not PIPELINE_PATH.exists():
    model_ok = ensure_model_exists()
    if not model_ok:
        st.stop()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:8px 0 16px 0;">
        <div style="font-size:1.1rem;font-weight:800;color:#E8EDF5;">🎯 CustomerIntent</div>
        <div style="font-size:0.7rem;color:#4A6080;">v1.0.0 · AI Ticket Router</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#0A2A1A;border:1px solid #1A5A30;border-radius:8px;
    padding:8px 12px;margin-bottom:12px;font-size:0.78rem;color:#34C759;">
    ● Model Loaded — Standalone Mode
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠  Live Classifier", "📊  Analytics", "🎫  Ticket History"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem;color:#3A5070;line-height:1.8;">
    <div style="color:#4A7090;font-weight:600;margin-bottom:4px;">SYSTEM INFO</div>
    Model &nbsp;&nbsp;&nbsp;TF-IDF + LR<br>
    Intents &nbsp;&nbsp;9 classes<br>
    Mode &nbsp;&nbsp;&nbsp;&nbsp;Standalone<br>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.ticket_history = []
        st.session_state.last_prediction = None
        st.rerun()

# ── PAGE 1: Live Classifier ────────────────────────────────────────────────────
if "Live Classifier" in page:
    st.markdown("""
    <div style="padding:8px 0 24px 0;">
        <h1 style="font-size:2rem;font-weight:800;color:#E8EDF5;margin:0;letter-spacing:-1px;">
        Live Intent Classifier</h1>
        <p style="color:#4A6080;font-size:0.88rem;margin:6px 0 0 0;">
        Real-time customer support intent classification and smart ticket routing
        </p>
    </div>
    """, unsafe_allow_html=True)

    left_col, right_col = st.columns([3, 2], gap="large")

    with left_col:
        user_input = st.text_area(
            "Customer Message",
            placeholder="e.g. My payment failed but money got deducted...",
            height=130,
            key="main_input",
            label_visibility="collapsed",
        )
        classify_btn = st.button("🎯  Classify Intent", type="primary")

    with right_col:
        st.markdown("""<div style="font-size:0.75rem;font-weight:600;color:#4A6080;
        text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">
        Quick Examples</div>""", unsafe_allow_html=True)
        for ex in EXAMPLE_MESSAGES:
            if st.button(
                f"{ex[:45]}{'...' if len(ex)>45 else ''}",
                key=f"ex_{hash(ex)}",
                use_container_width=True
            ):
                st.session_state["main_input"] = ex
                st.rerun()

    if classify_btn:
        if not user_input or len(user_input.strip()) < 5:
            st.warning("Please enter at least 5 characters.")
        else:
            with st.spinner("Analyzing..."):
                prediction = predict(user_input)
            if prediction:
                st.session_state.last_prediction = prediction
                add_to_history(user_input, prediction)

    if st.session_state.last_prediction:
        pred = st.session_state.last_prediction
        primary = pred["primary_intent"]
        intent = primary["intent"]
        priority = primary["priority"]
        conf = primary["confidence"]

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<div style="height:1px;background:linear-gradient(90deg,
        #0066FF22,#0066FF88,#0066FF22);margin-bottom:24px;"></div>""", unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Intent", f"{INTENT_EMOJI.get(intent,'🎯')} {primary['display_label']}")
        k2.metric("Department", primary["department"])
        k3.metric("Priority", f"{PRIORITY_EMOJI.get(priority,'')} {priority.upper()}")
        k4.metric("Confidence", f"{conf*100:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)
        chart_col, info_col = st.columns([3, 2], gap="large")

        with chart_col:
            top = pred["top_intents"]
            fig = go.Figure(go.Bar(
                x=[round(t["confidence"]*100, 2) for t in top],
                y=[t["display_label"] for t in top],
                orientation="h",
                marker=dict(color=["#0066FF","#0044AA","#002266"], line=dict(width=0)),
                text=[f"{round(t['confidence']*100,1)}%" for t in top],
                textposition="inside",
                textfont=dict(color="white", size=12),
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#6B7C93"), height=180,
                margin=dict(l=0, r=10, t=5, b=5),
                xaxis=dict(showgrid=False, showticklabels=False, range=[0,110]),
                yaxis=dict(showgrid=False, tickfont=dict(size=12, color="#C8D3E0")),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with info_col:
            if pred.get("keywords"):
                kw_html = "".join([
                    f'<span style="display:inline-block;background:#0D1D33;'
                    f'border:1px solid #1C3A5A;border-radius:5px;padding:4px 10px;'
                    f'margin:3px;font-size:12px;color:#7AADFF;">#{kw}</span>'
                    for kw in pred["keywords"]
                ])
                st.markdown(kw_html, unsafe_allow_html=True)
            pcolor = PRIORITY_COLORS.get(priority, "#888")
            st.markdown(
                f'<div style="margin-top:12px;"><span style="background:{pcolor}22;'
                f'border:1px solid {pcolor}55;color:{pcolor};border-radius:6px;'
                f'padding:5px 14px;font-size:12px;font-weight:700;">'
                f'{PRIORITY_EMOJI.get(priority,"")} {priority.upper()} PRIORITY</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<p style="color:#2A4060;font-size:11px;margin-top:12px;">'
                f'⚡ {pred["processing_time_ms"]}ms · {pred["model_used"]}</p>',
                unsafe_allow_html=True,
            )

# ── PAGE 2: Analytics ──────────────────────────────────────────────────────────
elif "Analytics" in page:
    st.markdown("""
    <h1 style="font-size:2rem;font-weight:800;color:#E8EDF5;margin:0 0 24px 0;">
    Analytics Dashboard</h1>""", unsafe_allow_html=True)

    history = st.session_state.ticket_history
    if not history:
        st.info("No data yet. Classify some messages in Live Classifier first.")
    else:
        total = len(history)
        critical_cnt = sum(1 for t in history if t["priority"] == "critical")
        avg_conf = sum(t["confidence"] for t in history) / total
        dept_count = len(set(t["department"] for t in history))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Tickets", total)
        m2.metric("🚨 Critical", critical_cnt)
        m3.metric("Avg Confidence", f"{avg_conf*100:.1f}%")
        m4.metric("Departments Hit", dept_count)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="large")

        with c1:
            intent_counts: Dict = {}
            for t in history:
                intent_counts[t["display_label"]] = intent_counts.get(t["display_label"], 0) + 1
            fig_pie = px.pie(
                names=list(intent_counts.keys()),
                values=list(intent_counts.values()),
                title="Intent Distribution",
                color_discrete_sequence=["#0066FF","#0044AA","#0088FF","#00AAFF",
                                          "#0033CC","#3377FF","#005599","#0099DD","#002299"],
                hole=0.45,
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#6B7C93"),
                height=300, margin=dict(l=0,r=0,t=30,b=10)
            )
            st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

        with c2:
            prio_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
            for t in history:
                if t["priority"] in prio_counts:
                    prio_counts[t["priority"]] += 1
            fig_bar = go.Figure(go.Bar(
                x=list(prio_counts.keys()), y=list(prio_counts.values()),
                marker=dict(color=[PRIORITY_COLORS[p] for p in prio_counts.keys()]),
                text=list(prio_counts.values()), textposition="outside",
            ))
            fig_bar.update_layout(
                title="Priority Breakdown",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#6B7C93"), height=300,
                xaxis=dict(showgrid=False), yaxis=dict(showgrid=False, showticklabels=False),
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

# ── PAGE 3: Ticket History ─────────────────────────────────────────────────────
elif "Ticket History" in page:
    st.markdown("""
    <h1 style="font-size:2rem;font-weight:800;color:#E8EDF5;margin:0 0 24px 0;">
    Ticket History</h1>""", unsafe_allow_html=True)

    history = st.session_state.ticket_history
    if not history:
        st.info("No tickets yet. Classify messages to see them here.")
    else:
        for ticket in history:
            priority = ticket["priority"]
            pcolor = PRIORITY_COLORS.get(priority, "#888")
            emoji_i = INTENT_EMOJI.get(ticket["intent"], "🎯")
            emoji_p = PRIORITY_EMOJI.get(priority, "")
            st.markdown(f"""
            <div style="background:#0D1526;border-left:3px solid {pcolor};
            border-radius:0 10px 10px 0;padding:14px 18px;margin:8px 0;
            border-top:1px solid #1C2A3A;border-right:1px solid #1C2A3A;
            border-bottom:1px solid #1C2A3A;">
                <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                    <span style="font-size:12px;color:#0066FF;font-weight:600;">
                    {ticket['id']}</span>
                    <span style="color:#2A4060;font-size:11px;">
                    {ticket['date']} · {ticket['timestamp']}</span>
                </div>
                <div style="color:#C8D3E0;font-size:0.9rem;margin:6px 0 10px 0;">
                {ticket['text']}</div>
                <div style="display:flex;gap:8px;flex-wrap:wrap;">
                    <span style="background:#0D1D33;border:1px solid #1C3A5A;color:#7AADFF;
                    border-radius:5px;padding:3px 10px;font-size:11px;font-weight:600;">
                    {emoji_i} {ticket['display_label']}</span>
                    <span style="background:{pcolor}15;border:1px solid {pcolor}40;
                    color:{pcolor};border-radius:5px;padding:3px 10px;font-size:11px;
                    font-weight:700;">{emoji_p} {priority.upper()}</span>
                    <span style="color:#2A4060;font-size:11px;">
                    {ticket['confidence']*100:.1f}% conf</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
