"""
CustomerIntent — Streamlit Frontend

Professional AI-powered customer support intent classification UI.
Features: live typing prediction, confidence bars, ticket history, analytics.
"""
import time
from datetime import datetime
from typing import Dict, List, Optional

import httpx
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CustomerIntent AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000/api/v1"

PRIORITY_COLORS = {
    "critical": "#FF0000",
    "high": "#FF6B35",
    "medium": "#FFD700",
    "low": "#4CAF50",
}

PRIORITY_EMOJI = {
    "critical": "🚨",
    "high": "🔴",
    "medium": "🟡",
    "low": "🟢",
}

INTENT_EMOJI = {
    "payment_issue": "💳",
    "refund_request": "💰",
    "account_locked": "🔒",
    "technical_bug": "🐛",
    "feature_request": "⭐",
    "subscription_cancel": "❌",
    "invoice_problem": "📄",
    "shipping_delay": "📦",
    "general_inquiry": "❓",
}

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main styling */
.stApp { background-color: #0E1117; }

.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 20px;
    margin: 8px 0;
}

.intent-badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.5px;
}

.priority-critical { background: #FF0000; color: white; }
.priority-high     { background: #FF6B35; color: white; }
.priority-medium   { background: #FFD700; color: black; }
.priority-low      { background: #4CAF50; color: white; }

.ticket-card {
    background: #1a1a2e;
    border-left: 4px solid #0f3460;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 14px;
}

.confidence-label {
    font-size: 12px;
    color: #888;
    margin-bottom: 2px;
}

.main-title {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00D4FF, #7B2FBE);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}

.subtitle {
    color: #888;
    font-size: 1rem;
    margin-top: 4px;
    margin-bottom: 24px;
}
</style>
""", unsafe_allow_html=True)

# ── Session State ──────────────────────────────────────────────────────────────
if "ticket_history" not in st.session_state:
    st.session_state.ticket_history = []
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "api_status" not in st.session_state:
    st.session_state.api_status = "unknown"

# ── Helper Functions ───────────────────────────────────────────────────────────
def check_api_health() -> bool:
    """Check if FastAPI backend is running."""
    try:
        r = httpx.get(f"{API_BASE}/health", timeout=3.0)
        if r.status_code == 200:
            st.session_state.api_status = "online"
            return True
    except Exception:
        pass
    st.session_state.api_status = "offline"
    return False


def predict_intent(text: str) -> Optional[Dict]:
    """Call the prediction API."""
    try:
        r = httpx.post(
            f"{API_BASE}/predict",
            json={"text": text, "model_type": "baseline"},
            timeout=10.0,
        )
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        st.error(f"API Error: {e}")
    return None


def add_to_history(text: str, prediction: Dict) -> None:
    """Add a prediction to the ticket history."""
    ticket = {
        "id": f"TKT-{len(st.session_state.ticket_history) + 1001:04d}",
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "text": text[:80] + ("..." if len(text) > 80 else ""),
        "intent": prediction["primary_intent"]["intent"],
        "display_label": prediction["primary_intent"]["display_label"],
        "department": prediction["primary_intent"]["department"],
        "priority": prediction["primary_intent"]["priority"],
        "confidence": prediction["primary_intent"]["confidence"],
    }
    st.session_state.ticket_history.insert(0, ticket)
    if len(st.session_state.ticket_history) > 50:
        st.session_state.ticket_history = st.session_state.ticket_history[:50]


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 CustomerIntent AI")
    st.markdown("---")

    # API Status
    api_ok = check_api_health()
    status_color = "🟢" if api_ok else "🔴"
    status_text = "API Online" if api_ok else "API Offline"
    st.markdown(f"**Status:** {status_color} {status_text}")

    if not api_ok:
        st.warning("Start the API:\n```\nuvicorn app.main:app --reload\n```")

    st.markdown("---")

    # Navigation
    page = st.radio(
        "Navigate",
        ["🏠 Live Classifier", "📊 Analytics", "🎫 Ticket History"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Model:** TF-IDF + LR (Baseline)")
    st.markdown("**Intents:** 9 classes")
    st.markdown("**Version:** v1.0.0")

    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.ticket_history = []
        st.session_state.last_prediction = None
        st.rerun()

# ── Main Content ───────────────────────────────────────────────────────────────

# ── PAGE 1: Live Classifier ────────────────────────────────────────────────────
if "Live Classifier" in page:
    st.markdown('<p class="main-title">CustomerIntent AI</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Real-time customer support intent classification & smart ticket routing</p>',
        unsafe_allow_html=True,
    )

    # Input
    col1, col2 = st.columns([2, 1])

    with col1:
        user_input = st.text_area(
            "📝 Enter customer message",
            placeholder="e.g. My payment failed but money got deducted from my bank...",
            height=120,
            key="main_input",
        )

        col_btn1, col_btn2 = st.columns([1, 3])
        with col_btn1:
            classify_btn = st.button(
                "🎯 Classify", type="primary", use_container_width=True
            )
        with col_btn2:
            live_mode = st.toggle("⚡ Live typing mode", value=False)

    with col2:
        st.markdown("**📌 Example messages:**")
        examples = [
            "My payment failed but money was deducted",
            "I want to cancel my subscription",
            "App keeps crashing on dashboard",
            "Need GST invoice for last month",
            "Package not delivered, it's been 7 days",
        ]
        for ex in examples:
            if st.button(ex[:45] + "...", key=f"ex_{ex[:10]}", use_container_width=True):
                st.session_state["main_input"] = ex
                st.rerun()

    # Live mode: predict on every keystroke change
    if live_mode and user_input and len(user_input) >= 10:
        prediction = predict_intent(user_input)
        if prediction:
            st.session_state.last_prediction = prediction

    # Button click prediction
    if classify_btn:
        if not user_input or len(user_input.strip()) < 5:
            st.warning("Please enter at least 5 characters.")
        elif not api_ok:
            st.error("API is offline. Start the backend first.")
        else:
            with st.spinner("Analyzing intent..."):
                prediction = predict_intent(user_input)

            if prediction:
                st.session_state.last_prediction = prediction
                add_to_history(user_input, prediction)
                st.success("✅ Classification complete!")

    # ── Results Section ────────────────────────────────────────────────────────
    if st.session_state.last_prediction:
        pred = st.session_state.last_prediction
        primary = pred["primary_intent"]

        st.markdown("---")
        st.markdown("### 🎯 Classification Result")

        # KPI Cards
        k1, k2, k3, k4 = st.columns(4)

        with k1:
            emoji = INTENT_EMOJI.get(primary["intent"], "🎯")
            st.metric(
                label="Intent Detected",
                value=f"{emoji} {primary['display_label']}",
            )

        with k2:
            st.metric(
                label="Department",
                value=f"🏢 {primary['department']}",
            )

        with k3:
            priority = primary["priority"]
            emoji_p = PRIORITY_EMOJI.get(priority, "")
            st.metric(
                label="Priority",
                value=f"{emoji_p} {priority.upper()}",
            )

        with k4:
            conf_pct = round(primary["confidence"] * 100, 1)
            st.metric(
                label="Confidence",
                value=f"{conf_pct}%",
            )

        # Confidence chart for top intents
        st.markdown("#### 📊 Intent Confidence Distribution")
        top_intents = pred["top_intents"]

        labels = [t["display_label"] for t in top_intents]
        values = [round(t["confidence"] * 100, 2) for t in top_intents]

        fig = go.Figure(go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(
                color=["#00D4FF", "#7B2FBE", "#4CAF50"],
                line=dict(width=0),
            ),
            text=[f"{v}%" for v in values],
            textposition="inside",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=200,
            margin=dict(l=0, r=0, t=10, b=10),
            xaxis=dict(showgrid=False, showticklabels=False, range=[0, 105]),
            yaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Keywords
        if pred.get("keywords"):
            st.markdown("#### 🔍 Key Signals Detected")
            keyword_html = " ".join([
                f'<span style="background:#1a1a2e;border:1px solid #0f3460;'
                f'border-radius:12px;padding:4px 12px;margin:3px;'
                f'display:inline-block;font-size:13px;">#{kw}</span>'
                for kw in pred["keywords"]
            ])
            st.markdown(keyword_html, unsafe_allow_html=True)

        # Meta
        st.markdown(
            f'<p style="color:#555;font-size:12px;margin-top:12px;">'
            f'⚡ Processed in {pred["processing_time_ms"]}ms | '
            f'Model: {pred["model_used"]}</p>',
            unsafe_allow_html=True,
        )

# ── PAGE 2: Analytics ──────────────────────────────────────────────────────────
elif "Analytics" in page:
    st.markdown("## 📊 Analytics Dashboard")

    if not st.session_state.ticket_history:
        st.info("No tickets yet. Go to **Live Classifier** and classify some messages!")
    else:
        history = st.session_state.ticket_history

        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Tickets", len(history))

        critical_count = sum(1 for t in history if t["priority"] == "critical")
        m2.metric("🚨 Critical", critical_count)

        avg_conf = sum(t["confidence"] for t in history) / len(history)
        m3.metric("Avg Confidence", f"{avg_conf*100:.1f}%")

        departments = set(t["department"] for t in history)
        m4.metric("Departments Routed", len(departments))

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            # Intent distribution pie
            intent_counts: Dict = {}
            for t in history:
                intent_counts[t["display_label"]] = intent_counts.get(t["display_label"], 0) + 1

            fig_pie = px.pie(
                names=list(intent_counts.keys()),
                values=list(intent_counts.values()),
                title="Intent Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Priority distribution bar
            priority_counts: Dict = {"critical": 0, "high": 0, "medium": 0, "low": 0}
            for t in history:
                priority_counts[t["priority"]] = priority_counts.get(t["priority"], 0) + 1

            fig_bar = go.Figure(go.Bar(
                x=list(priority_counts.keys()),
                y=list(priority_counts.values()),
                marker_color=[PRIORITY_COLORS[p] for p in priority_counts.keys()],
            ))
            fig_bar.update_layout(
                title="Priority Distribution",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                yaxis=dict(showgrid=False),
                xaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

# ── PAGE 3: Ticket History ─────────────────────────────────────────────────────
elif "Ticket History" in page:
    st.markdown("## 🎫 Ticket History")

    if not st.session_state.ticket_history:
        st.info("No tickets yet. Classify some messages to see them here.")
    else:
        for ticket in st.session_state.ticket_history:
            priority = ticket["priority"]
            color = PRIORITY_COLORS.get(priority, "#888")
            emoji_i = INTENT_EMOJI.get(ticket["intent"], "🎯")
            emoji_p = PRIORITY_EMOJI.get(priority, "")

            st.markdown(
                f"""
                <div style="background:#1a1a2e;border-left:4px solid {color};
                border-radius:8px;padding:12px 16px;margin:6px 0;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span style="font-weight:700;color:white;">{ticket['id']}</span>
                        <span style="color:#888;font-size:12px;">{ticket['timestamp']}</span>
                    </div>
                    <div style="color:#ccc;font-size:14px;margin:6px 0;">{ticket['text']}</div>
                    <div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:8px;">
                        <span style="background:#0f3460;color:white;border-radius:12px;
                        padding:3px 10px;font-size:12px;">{emoji_i} {ticket['display_label']}</span>
                        <span style="background:#162447;color:white;border-radius:12px;
                        padding:3px 10px;font-size:12px;">🏢 {ticket['department']}</span>
                        <span style="background:{color};color:white;border-radius:12px;
                        padding:3px 10px;font-size:12px;">{emoji_p} {priority.upper()}</span>
                        <span style="color:#888;font-size:12px;padding:3px 0;">
                        Confidence: {ticket['confidence']*100:.1f}%</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
