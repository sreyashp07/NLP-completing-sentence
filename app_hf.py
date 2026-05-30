"""
CustomerIntent AI - Hugging Face Spaces
Production-stable standalone version.
Model trains once and persists in /tmp across restarts.
"""
import sys
import os
import re
import time
import pickle
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# ── NLTK first ─────────────────────────────────────────────────────────────────
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

# ── Streamlit ──────────────────────────────────────────────────────────────────
import streamlit as st

st.set_page_config(
    page_title="CustomerIntent AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

import plotly.graph_objects as go
import plotly.express as px

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Model stored in /tmp for HF persistence
TMP_MODEL_DIR = Path("/tmp/customerintent_model")
TMP_PIPELINE  = TMP_MODEL_DIR / "pipeline.pkl"
TMP_LE        = TMP_MODEL_DIR / "label_encoder.pkl"

# Also check local path
LOCAL_MODEL_DIR = ROOT / "ml" / "saved_models" / "baseline"
LOCAL_PIPELINE  = LOCAL_MODEL_DIR / "pipeline.pkl"
LOCAL_LE        = LOCAL_MODEL_DIR / "label_encoder.pkl"

# ── Constants ──────────────────────────────────────────────────────────────────
PRIORITY_COLORS = {"critical":"#FF2D55","high":"#FF6B35","medium":"#FFD700","low":"#34C759"}
PRIORITY_EMOJI  = {"critical":"🚨","high":"🔴","medium":"🟡","low":"🟢"}
INTENT_EMOJI    = {
    "payment_issue":"💳","refund_request":"💰","account_locked":"🔒",
    "technical_bug":"🐛","feature_request":"⭐","subscription_cancel":"❌",
    "invoice_problem":"📄","shipping_delay":"📦","general_inquiry":"❓",
}
INTENT_META = {
    "payment_issue":       {"department":"Billing Team",    "priority":"critical","label":"Payment Issue"},
    "refund_request":      {"department":"Billing Team",    "priority":"high",   "label":"Refund Request"},
    "account_locked":      {"department":"Security Team",   "priority":"critical","label":"Account Locked"},
    "technical_bug":       {"department":"Engineering Team","priority":"high",   "label":"Technical Bug"},
    "feature_request":     {"department":"Product Team",    "priority":"low",    "label":"Feature Request"},
    "subscription_cancel": {"department":"Retention Team",  "priority":"high",   "label":"Subscription Cancel"},
    "invoice_problem":     {"department":"Finance Team",    "priority":"medium", "label":"Invoice Problem"},
    "shipping_delay":      {"department":"Logistics Team",  "priority":"medium", "label":"Shipping Delay"},
    "general_inquiry":     {"department":"General Support", "priority":"low",    "label":"General Inquiry"},
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

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html,body,.stApp{background:#080C14!important;color:#E8EDF5;}
[data-testid="stSidebar"]{background:#0D1526!important;border-right:1px solid #1C2A3A;}
[data-testid="stSidebar"] *{color:#C8D3E0!important;}
#MainMenu,footer,header{visibility:hidden;}
[data-testid="metric-container"]{background:#0F1B2D;border:1px solid #1C2A3A;border-radius:12px;padding:16px;}
[data-testid="stMetricValue"]{font-weight:700;color:#E8EDF5!important;}
[data-testid="stMetricLabel"]{color:#6B7C93!important;font-size:.75rem!important;}
.stButton>button{background:linear-gradient(135deg,#0066FF,#0044CC);color:#fff!important;border:none;border-radius:8px;font-weight:600;}
.stTextArea textarea{background:#0D1526!important;border:1px solid #1C2A3A!important;border-radius:10px!important;color:#E8EDF5!important;}
hr{border-color:#1C2A3A!important;}
</style>
""", unsafe_allow_html=True)


# ── Find model paths ───────────────────────────────────────────────────────────
def get_model_paths():
    """Return (pipeline_path, le_path) — prefers /tmp then local."""
    if TMP_PIPELINE.exists() and TMP_LE.exists():
        return TMP_PIPELINE, TMP_LE
    if LOCAL_PIPELINE.exists() and LOCAL_LE.exists():
        return LOCAL_PIPELINE, LOCAL_LE
    return None, None


def model_exists() -> bool:
    p, l = get_model_paths()
    return p is not None


# ── Train model ────────────────────────────────────────────────────────────────
def train_model() -> bool:
    """Train model and save to /tmp for persistence."""
    TMP_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    placeholder = st.empty()
    with placeholder.container():
        st.markdown("## 🔧 First-Time Setup")
        st.info("Training the intent classifier — takes about 2 minutes on first run only.")
        prog = st.progress(0)
        status = st.empty()

        try:
            status.text("📊 Generating training data (1800 samples, 9 classes)...")
            prog.progress(5)

            env = os.environ.copy()
            env["NLTK_DATA"] = str(NLTK_DIR)

            r1 = subprocess.run(
                [sys.executable, str(ROOT / "data" / "generate_dataset.py")],
                capture_output=True, text=True, cwd=str(ROOT),
                timeout=180, env=env,
            )
            if r1.returncode != 0:
                st.error(f"Dataset generation failed:\n```\n{r1.stderr[-800:]}\n```")
                return False

            prog.progress(35)
            status.text("🤖 Training TF-IDF + Logistic Regression...")

            r2 = subprocess.run(
                [sys.executable, str(ROOT / "ml" / "training" / "train_baseline.py")],
                capture_output=True, text=True, cwd=str(ROOT),
                timeout=600, env=env,
            )
            if r2.returncode != 0:
                st.error(f"Training failed:\n```\n{r2.stderr[-800:]}\n```")
                return False

            prog.progress(80)
            status.text("💾 Copying artifacts to /tmp for persistence...")

            # Copy to /tmp
            import shutil
            for fname in ["pipeline.pkl", "label_encoder.pkl", "metrics.yaml", "classes.txt"]:
                src = LOCAL_MODEL_DIR / fname
                if src.exists():
                    shutil.copy2(src, TMP_MODEL_DIR / fname)

            if not TMP_PIPELINE.exists():
                st.error("Model file not found after training!")
                return False

            prog.progress(100)
            status.text("✅ Done!")
            time.sleep(1)

        except subprocess.TimeoutExpired:
            st.error("Training timed out. Please restart the Space.")
            return False
        except Exception as e:
            st.error(f"Error during training: {e}")
            return False

    placeholder.empty()
    return True


# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
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


# ── NLP helpers ────────────────────────────────────────────────────────────────
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


# ── Predict ────────────────────────────────────────────────────────────────────
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
        meta = INTENT_META.get(intent, {"department":"General Support","priority":"low","label":intent})
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


# ── Session state ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history: List[Dict] = []
if "last_pred" not in st.session_state:
    st.session_state.last_pred = None


def add_ticket(text: str, pred: Dict):
    n = len(st.session_state.history) + 1001
    st.session_state.history.insert(0, {
        "id": f"TKT-{n:04d}",
        "ts": datetime.now().strftime("%H:%M"),
        "date": datetime.now().strftime("%b %d"),
        "text": text[:100] + ("..." if len(text) > 100 else ""),
        **pred["primary_intent"],
    })


# ── Boot: train if needed ──────────────────────────────────────────────────────
if not model_exists():
    ok = train_model()
    if ok:
        st.cache_resource.clear()
        st.rerun()
    else:
        st.error("Could not start. Please click the restart button in Settings.")
        st.stop()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎯 CustomerIntent AI")
    st.markdown('<div style="background:#0A2A1A;border:1px solid #1A5A30;border-radius:8px;padding:8px 12px;font-size:.78rem;color:#34C759;">● Model Ready</div>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Nav", ["🏠 Live Classifier","📊 Analytics","🎫 Ticket History"], label_visibility="collapsed")
    st.markdown("---")
    st.caption("TF-IDF + LR | 9 classes | v1.0")
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.history = []
        st.session_state.last_pred = None
        st.rerun()


# ── PAGE 1 ─────────────────────────────────────────────────────────────────────
if "Live Classifier" in page:
    st.markdown("## 🎯 Live Intent Classifier")
    st.caption("Real-time customer support intent classification and smart ticket routing")

    lc, rc = st.columns([3, 2], gap="large")
    with lc:
        txt = st.text_area("msg", placeholder="e.g. My payment failed but money got deducted...", height=130, key="inp", label_visibility="collapsed")
        btn = st.button("🎯 Classify Intent", type="primary")
    with rc:
        st.caption("QUICK EXAMPLES")
        for ex in EXAMPLES:
            if st.button(f"{ex[:44]}{'...' if len(ex)>44 else ''}", key=f"e{hash(ex)}", use_container_width=True):
                st.session_state["inp"] = ex
                st.rerun()

    if btn:
        if not txt or len(txt.strip()) < 5:
            st.warning("Enter at least 5 characters.")
        else:
            with st.spinner("Classifying..."):
                pred = predict(txt)
            if pred:
                st.session_state.last_pred = pred
                add_ticket(txt, pred)

    if st.session_state.last_pred:
        p = st.session_state.last_pred
        pi = p["primary_intent"]
        st.divider()
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Intent",      f"{INTENT_EMOJI.get(pi['intent'],'🎯')} {pi['display_label']}")
        c2.metric("Department",  pi["department"])
        c3.metric("Priority",    f"{PRIORITY_EMOJI.get(pi['priority'],'')} {pi['priority'].upper()}")
        c4.metric("Confidence",  f"{pi['confidence']*100:.1f}%")

        cc, ic = st.columns([3, 2])
        with cc:
            top = p["top_intents"]
            fig = go.Figure(go.Bar(
                x=[round(t["confidence"]*100,2) for t in top],
                y=[t["display_label"] for t in top],
                orientation="h",
                marker=dict(color=["#0066FF","#0044AA","#002266"]),
                text=[f"{round(t['confidence']*100,1)}%" for t in top],
                textposition="inside", textfont=dict(color="white",size=12),
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#6B7C93"), height=180,
                margin=dict(l=0,r=10,t=5,b=5),
                xaxis=dict(showgrid=False,showticklabels=False,range=[0,110]),
                yaxis=dict(showgrid=False,tickfont=dict(size=12,color="#C8D3E0")),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        with ic:
            if p.get("keywords"):
                st.markdown(" ".join([f'<span style="background:#0D1D33;border:1px solid #1C3A5A;border-radius:5px;padding:4px 10px;margin:3px;display:inline-block;font-size:12px;color:#7AADFF;">#{k}</span>' for k in p["keywords"]]), unsafe_allow_html=True)
            pc = PRIORITY_COLORS.get(pi["priority"],"#888")
            st.markdown(f'<div style="margin-top:12px;"><span style="background:{pc}22;border:1px solid {pc}55;color:{pc};border-radius:6px;padding:5px 14px;font-size:12px;font-weight:700;">{PRIORITY_EMOJI.get(pi["priority"],"")} {pi["priority"].upper()} PRIORITY</span></div>', unsafe_allow_html=True)
            st.caption(f"⚡ {p['processing_time_ms']}ms · {p['model_used']}")


# ── PAGE 2 ─────────────────────────────────────────────────────────────────────
elif "Analytics" in page:
    st.markdown("## 📊 Analytics Dashboard")
    h = st.session_state.history
    if not h:
        st.info("No data yet. Classify some messages first.")
    else:
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Tickets", len(h))
        m2.metric("🚨 Critical", sum(1 for t in h if t["priority"]=="critical"))
        m3.metric("Avg Conf", f"{sum(t['confidence'] for t in h)/len(h)*100:.1f}%")
        m4.metric("Depts", len(set(t["department"] for t in h)))
        c1,c2 = st.columns(2)
        with c1:
            ic: Dict = {}
            for t in h: ic[t["display_label"]] = ic.get(t["display_label"],0)+1
            fig = px.pie(names=list(ic.keys()), values=list(ic.values()), title="Intent Distribution", hole=0.45, color_discrete_sequence=["#0066FF","#0044AA","#0088FF","#00AAFF","#0033CC","#3377FF","#005599","#0099DD","#002299"])
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#6B7C93"), height=300)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        with c2:
            pc = {"critical":0,"high":0,"medium":0,"low":0}
            for t in h:
                if t["priority"] in pc: pc[t["priority"]] += 1
            fig2 = go.Figure(go.Bar(x=list(pc.keys()), y=list(pc.values()), marker=dict(color=[PRIORITY_COLORS[p] for p in pc]), text=list(pc.values()), textposition="outside"))
            fig2.update_layout(title="Priority Breakdown", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#6B7C93"), height=300, xaxis=dict(showgrid=False), yaxis=dict(showgrid=False,showticklabels=False))
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})


# ── PAGE 3 ─────────────────────────────────────────────────────────────────────
elif "Ticket History" in page:
    st.markdown("## 🎫 Ticket History")
    h = st.session_state.history
    if not h:
        st.info("No tickets yet.")
    else:
        for t in h:
            pc = PRIORITY_COLORS.get(t["priority"],"#888")
            ei = INTENT_EMOJI.get(t["intent"],"🎯")
            ep = PRIORITY_EMOJI.get(t["priority"],"")
            st.markdown(f"""<div style="background:#0D1526;border-left:3px solid {pc};border-radius:0 10px 10px 0;padding:14px 18px;margin:8px 0;border:1px solid #1C2A3A;border-left:3px solid {pc};">
<div style="display:flex;justify-content:space-between;margin-bottom:6px;">
<span style="font-size:12px;color:#0066FF;font-weight:600;">{t['id']}</span>
<span style="color:#2A4060;font-size:11px;">{t['date']} · {t['ts']}</span></div>
<div style="color:#C8D3E0;font-size:.9rem;margin:6px 0 10px 0;">{t['text']}</div>
<div style="display:flex;gap:8px;flex-wrap:wrap;">
<span style="background:#0D1D33;border:1px solid #1C3A5A;color:#7AADFF;border-radius:5px;padding:3px 10px;font-size:11px;">{ei} {t['display_label']}</span>
<span style="background:{pc}22;border:1px solid {pc}55;color:{pc};border-radius:5px;padding:3px 10px;font-size:11px;font-weight:700;">{ep} {t['priority'].upper()}</span>
<span style="color:#2A4060;font-size:11px;">{t['confidence']*100:.1f}% conf</span>
</div></div>""", unsafe_allow_html=True)
