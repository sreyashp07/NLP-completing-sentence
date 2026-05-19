"""
CustomerIntent AI — Streamlit Frontend
Complete 4-page professional UI:
  1. Live Classifier
  2. Analytics Dashboard
  3. Ticket History
  4. Model Evaluation
"""

import time
import yaml
import pandas as pd
from datetime import datetime
from pathlib import Path
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
    "critical": "#FF2D55",
    "high":     "#FF6B35",
    "medium":   "#FFD700",
    "low":      "#34C759",
}

PRIORITY_EMOJI = {
    "critical": "🚨",
    "high":     "🔴",
    "medium":   "🟡",
    "low":      "🟢",
}

INTENT_EMOJI = {
    "payment_issue":       "💳",
    "refund_request":      "💰",
    "account_locked":      "🔒",
    "technical_bug":       "🐛",
    "feature_request":     "⭐",
    "subscription_cancel": "❌",
    "invoice_problem":     "📄",
    "shipping_delay":      "📦",
    "general_inquiry":     "❓",
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

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;400;600;700;800&display=swap');

/* ── Base ── */
html, body, .stApp {
    background-color: #080C14 !important;
    font-family: 'Sora', sans-serif;
    color: #E8EDF5;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1526 0%, #080C14 100%) !important;
    border-right: 1px solid #1C2A3A;
}
[data-testid="stSidebar"] * { color: #C8D3E0 !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0F1B2D 0%, #0A1520 100%);
    border: 1px solid #1C2A3A;
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="stMetricValue"] {
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    font-size: 1.3rem !important;
    color: #E8EDF5 !important;
}
[data-testid="stMetricLabel"] {
    color: #6B7C93 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #0066FF, #0044CC);
    color: white !important;
    border: none;
    border-radius: 8px;
    font-family: 'Sora', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 8px 16px;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0077FF, #0055DD);
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(0, 102, 255, 0.3);
}

/* ── Text area ── */
.stTextArea textarea {
    background: #0D1526 !important;
    border: 1px solid #1C2A3A !important;
    border-radius: 10px !important;
    color: #E8EDF5 !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.95rem !important;
    resize: vertical;
}
.stTextArea textarea:focus {
    border-color: #0066FF !important;
    box-shadow: 0 0 0 2px rgba(0, 102, 255, 0.15) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1C2A3A;
    border-radius: 10px;
    overflow: hidden;
}

/* ── Radio buttons ── */
[data-testid="stRadio"] label {
    font-family: 'Sora', sans-serif !important;
    font-size: 0.88rem !important;
}

/* ── Alerts ── */
.stAlert {
    background: #0D1526 !important;
    border-radius: 10px !important;
    border: 1px solid #1C2A3A !important;
}

/* ── Divider ── */
hr { border-color: #1C2A3A !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #080C14; }
::-webkit-scrollbar-thumb { background: #1C2A3A; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2A3F5A; }
</style>
""", unsafe_allow_html=True)

# ── Session State ──────────────────────────────────────────────────────────────
if "ticket_history" not in st.session_state:
    st.session_state.ticket_history: List[Dict] = []
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction: Optional[Dict] = None
if "api_online" not in st.session_state:
    st.session_state.api_online = False

# ── Helper Functions ───────────────────────────────────────────────────────────
def check_api_health() -> bool:
    try:
        r = httpx.get(f"{API_BASE}/health", timeout=3.0)
        ok = r.status_code == 200
        st.session_state.api_online = ok
        return ok
    except Exception:
        st.session_state.api_online = False
        return False


def predict_intent(text: str) -> Optional[Dict]:
    try:
        r = httpx.post(
            f"{API_BASE}/predict",
            json={"text": text, "model_type": "baseline"},
            timeout=15.0,
        )
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        st.error(f"API Error: {e}")
    return None


def add_to_history(text: str, prediction: Dict) -> None:
    ticket_num = len(st.session_state.ticket_history) + 1001
    ticket = {
        "id":            f"TKT-{ticket_num:04d}",
        "timestamp":     datetime.now().strftime("%H:%M:%S"),
        "date":          datetime.now().strftime("%b %d"),
        "text":          text[:100] + ("..." if len(text) > 100 else ""),
        "intent":        prediction["primary_intent"]["intent"],
        "display_label": prediction["primary_intent"]["display_label"],
        "department":    prediction["primary_intent"]["department"],
        "priority":      prediction["primary_intent"]["priority"],
        "confidence":    prediction["primary_intent"]["confidence"],
    }
    st.session_state.ticket_history.insert(0, ticket)
    if len(st.session_state.ticket_history) > 100:
        st.session_state.ticket_history = st.session_state.ticket_history[:100]


def render_badge(text: str, bg: str, color: str = "white", radius: str = "6px") -> str:
    return (
        f'<span style="background:{bg};color:{color};border-radius:{radius};'
        f'padding:3px 10px;font-size:11px;font-weight:600;'
        f'font-family:Sora,sans-serif;letter-spacing:0.5px;">{text}</span>'
    )


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:8px 0 16px 0;">
        <div style="font-family:Sora,sans-serif;font-size:1.1rem;font-weight:800;
        color:#E8EDF5;letter-spacing:-0.5px;">🎯 CustomerIntent</div>
        <div style="font-size:0.7rem;color:#4A6080;margin-top:2px;font-family:JetBrains Mono,monospace;">
        v1.0.0 · AI Ticket Router</div>
    </div>
    """, unsafe_allow_html=True)

    # API status
    api_ok = check_api_health()
    if api_ok:
        st.markdown("""
        <div style="background:#0A2A1A;border:1px solid #1A5A30;border-radius:8px;
        padding:8px 12px;margin-bottom:12px;font-size:0.78rem;color:#34C759;">
        ● API Online — localhost:8000
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#2A0A0A;border:1px solid #5A1A1A;border-radius:8px;
        padding:8px 12px;margin-bottom:12px;font-size:0.78rem;color:#FF2D55;">
        ● API Offline
        </div>""", unsafe_allow_html=True)
        st.code("py -m uvicorn app.main:app --reload", language="bash")

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠  Live Classifier", "📊  Analytics", "🎫  Ticket History", "🔬  Model Evaluation"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown("""
    <div style="font-size:0.72rem;color:#3A5070;line-height:1.8;">
    <div style="color:#4A7090;font-weight:600;margin-bottom:4px;">SYSTEM INFO</div>
    Model &nbsp;&nbsp;&nbsp;TF-IDF + LR<br>
    Intents &nbsp;&nbsp;9 classes<br>
    Env &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;development<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.ticket_history = []
            st.session_state.last_prediction = None
            st.rerun()
    with col_b:
        ticket_count = len(st.session_state.ticket_history)
        st.markdown(
            f'<div style="text-align:center;padding:8px;background:#0D1526;'
            f'border:1px solid #1C2A3A;border-radius:8px;font-size:0.78rem;'
            f'color:#6B7C93;">{ticket_count} tickets</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — LIVE CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
if "Live Classifier" in page:

    # Header
    st.markdown("""
    <div style="padding:8px 0 24px 0;">
        <h1 style="font-family:Sora,sans-serif;font-size:2rem;font-weight:800;
        color:#E8EDF5;margin:0;letter-spacing:-1px;">
        Live Intent Classifier
        </h1>
        <p style="color:#4A6080;font-size:0.88rem;margin:6px 0 0 0;">
        Type a customer message and get instant intent classification with department routing
        </p>
    </div>
    """, unsafe_allow_html=True)

    left_col, right_col = st.columns([3, 2], gap="large")

    with left_col:
        user_input = st.text_area(
            "Customer Message",
            placeholder="e.g. My payment failed but money got deducted from my bank account...",
            height=130,
            key="main_input",
            label_visibility="collapsed",
        )

        btn_col, toggle_col = st.columns([1, 2])
        with btn_col:
            classify_btn = st.button("🎯  Classify Intent", type="primary", use_container_width=True)
        with toggle_col:
            live_mode = st.toggle("⚡ Live typing mode", value=False)

        # Live mode
        if live_mode and user_input and len(user_input.strip()) >= 10 and api_ok:
            prediction = predict_intent(user_input)
            if prediction:
                st.session_state.last_prediction = prediction

        # Classify button
        if classify_btn:
            if not user_input or len(user_input.strip()) < 5:
                st.warning("Please enter at least 5 characters.")
            elif not api_ok:
                st.error("API is offline. Start the backend with: `py -m uvicorn app.main:app --reload`")
            else:
                with st.spinner("Analyzing..."):
                    prediction = predict_intent(user_input)
                if prediction:
                    st.session_state.last_prediction = prediction
                    add_to_history(user_input, prediction)

    with right_col:
        st.markdown("""
        <div style="font-size:0.75rem;font-weight:600;color:#4A6080;
        text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">
        Quick Examples
        </div>""", unsafe_allow_html=True)

        for ex in EXAMPLE_MESSAGES:
            if st.button(
                f"{INTENT_EMOJI.get('general_inquiry', '💬')}  {ex[:42]}{'...' if len(ex)>42 else ''}",
                key=f"ex_{hash(ex)}",
                use_container_width=True,
            ):
                st.session_state["main_input"] = ex
                st.rerun()

    # ── Result Panel ──────────────────────────────────────────────────────────
    if st.session_state.last_prediction:
        pred    = st.session_state.last_prediction
        primary = pred["primary_intent"]
        intent  = primary["intent"]
        priority = primary["priority"]
        conf    = primary["confidence"]

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="height:1px;background:linear-gradient(90deg,#0066FF22,#0066FF88,#0066FF22);
        margin-bottom:24px;"></div>
        """, unsafe_allow_html=True)

        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Intent Detected",  f"{INTENT_EMOJI.get(intent,'🎯')} {primary['display_label']}")
        k2.metric("Routed To",         primary["department"])
        k3.metric("Priority",          f"{PRIORITY_EMOJI.get(priority,'')} {priority.upper()}")
        k4.metric("Confidence",        f"{conf*100:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)

        chart_col, info_col = st.columns([3, 2], gap="large")

        with chart_col:
            st.markdown("""
            <div style="font-size:0.75rem;font-weight:600;color:#4A6080;
            text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">
            Confidence Distribution
            </div>""", unsafe_allow_html=True)

            top = pred["top_intents"]
            labels = [t["display_label"] for t in top]
            values = [round(t["confidence"] * 100, 2) for t in top]
            bar_colors = ["#0066FF", "#0044AA", "#002266"]

            fig = go.Figure(go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker=dict(color=bar_colors, line=dict(width=0)),
                text=[f"{v:.1f}%" for v in values],
                textposition="inside",
                textfont=dict(color="white", size=12, family="Sora"),
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#6B7C93", family="Sora"),
                height=180,
                margin=dict(l=0, r=10, t=5, b=5),
                xaxis=dict(showgrid=False, showticklabels=False, range=[0, 110]),
                yaxis=dict(showgrid=False, tickfont=dict(size=12, color="#C8D3E0")),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with info_col:
            st.markdown("""
            <div style="font-size:0.75rem;font-weight:600;color:#4A6080;
            text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">
            Key Signals
            </div>""", unsafe_allow_html=True)

            if pred.get("keywords"):
                kw_html = "".join([
                    f'<span style="display:inline-block;background:#0D1D33;'
                    f'border:1px solid #1C3A5A;border-radius:5px;padding:4px 10px;'
                    f'margin:3px;font-size:12px;color:#7AADFF;'
                    f'font-family:JetBrains Mono,monospace;">#{kw}</span>'
                    for kw in pred["keywords"]
                ])
                st.markdown(kw_html, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Priority pill
            pcolor = PRIORITY_COLORS.get(priority, "#888")
            st.markdown(
                f'<div style="margin-top:4px;">'
                f'<span style="background:{pcolor}22;border:1px solid {pcolor}55;'
                f'color:{pcolor};border-radius:6px;padding:5px 14px;'
                f'font-size:12px;font-weight:700;font-family:Sora,sans-serif;">'
                f'{PRIORITY_EMOJI.get(priority,"")} {priority.upper()} PRIORITY</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<p style="color:#2A4060;font-size:11px;margin-top:14px;'
                f'font-family:JetBrains Mono,monospace;">'
                f'⚡ {pred["processing_time_ms"]}ms · {pred["model_used"]}</p>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif "Analytics" in page:

    st.markdown("""
    <div style="padding:8px 0 24px 0;">
        <h1 style="font-family:Sora,sans-serif;font-size:2rem;font-weight:800;
        color:#E8EDF5;margin:0;letter-spacing:-1px;">Analytics Dashboard</h1>
        <p style="color:#4A6080;font-size:0.88rem;margin:6px 0 0 0;">
        Real-time insights from classified tickets this session
        </p>
    </div>
    """, unsafe_allow_html=True)

    history = st.session_state.ticket_history

    if not history:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#3A5070;">
            <div style="font-size:3rem;margin-bottom:12px;">📊</div>
            <div style="font-size:1rem;font-weight:600;color:#4A6080;">No data yet</div>
            <div style="font-size:0.83rem;margin-top:6px;">
            Go to Live Classifier and classify some messages to see analytics here.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Summary KPIs
        total         = len(history)
        critical_cnt  = sum(1 for t in history if t["priority"] == "critical")
        avg_conf      = sum(t["confidence"] for t in history) / total
        dept_count    = len(set(t["department"] for t in history))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Tickets",    total)
        m2.metric("🚨 Critical",      critical_cnt)
        m3.metric("Avg Confidence",   f"{avg_conf*100:.1f}%")
        m4.metric("Departments Hit",  dept_count)

        st.markdown("<br>", unsafe_allow_html=True)

        chart1, chart2 = st.columns(2, gap="large")

        with chart1:
            st.markdown("""
            <div style="font-size:0.75rem;font-weight:600;color:#4A6080;
            text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">
            Intent Distribution
            </div>""", unsafe_allow_html=True)

            intent_counts: Dict = {}
            for t in history:
                intent_counts[t["display_label"]] = intent_counts.get(t["display_label"], 0) + 1

            fig_pie = px.pie(
                names=list(intent_counts.keys()),
                values=list(intent_counts.values()),
                color_discrete_sequence=[
                    "#0066FF","#0044AA","#0088FF","#00AAFF",
                    "#0033CC","#3377FF","#005599","#0099DD","#002299",
                ],
                hole=0.45,
            )
            fig_pie.update_traces(
                textfont=dict(family="Sora", size=11),
                marker=dict(line=dict(color="#080C14", width=2)),
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#6B7C93", family="Sora"),
                legend=dict(font=dict(size=10, color="#8A9BB0")),
                height=300,
                margin=dict(l=0, r=0, t=10, b=10),
            )
            st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

        with chart2:
            st.markdown("""
            <div style="font-size:0.75rem;font-weight:600;color:#4A6080;
            text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">
            Priority Breakdown
            </div>""", unsafe_allow_html=True)

            prio_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
            for t in history:
                p = t["priority"]
                if p in prio_counts:
                    prio_counts[p] += 1

            fig_bar = go.Figure(go.Bar(
                x=list(prio_counts.keys()),
                y=list(prio_counts.values()),
                marker=dict(
                    color=[PRIORITY_COLORS[p] for p in prio_counts.keys()],
                    line=dict(width=0),
                ),
                text=list(prio_counts.values()),
                textposition="outside",
                textfont=dict(color="#8A9BB0", size=12),
            ))
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#6B7C93", family="Sora"),
                height=300,
                margin=dict(l=0, r=0, t=10, b=10),
                xaxis=dict(showgrid=False, tickfont=dict(size=11, color="#8A9BB0")),
                yaxis=dict(showgrid=False, showticklabels=False),
                bargap=0.35,
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

        # Department routing table
        st.markdown("""
        <div style="font-size:0.75rem;font-weight:600;color:#4A6080;
        text-transform:uppercase;letter-spacing:1px;margin:16px 0 8px 0;">
        Department Routing Summary
        </div>""", unsafe_allow_html=True)

        dept_summary: Dict = {}
        for t in history:
            d = t["department"]
            if d not in dept_summary:
                dept_summary[d] = {"tickets": 0, "critical": 0, "avg_conf": []}
            dept_summary[d]["tickets"] += 1
            if t["priority"] == "critical":
                dept_summary[d]["critical"] += 1
            dept_summary[d]["avg_conf"].append(t["confidence"])

        rows = []
        for dept, data in dept_summary.items():
            rows.append({
                "Department":    dept,
                "Tickets":       data["tickets"],
                "Critical":      data["critical"],
                "Avg Confidence": f"{sum(data['avg_conf'])/len(data['avg_conf'])*100:.1f}%",
            })

        df_dept = pd.DataFrame(rows).sort_values("Tickets", ascending=False)
        st.dataframe(df_dept, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — TICKET HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif "Ticket History" in page:

    st.markdown("""
    <div style="padding:8px 0 24px 0;">
        <h1 style="font-family:Sora,sans-serif;font-size:2rem;font-weight:800;
        color:#E8EDF5;margin:0;letter-spacing:-1px;">Ticket History</h1>
        <p style="color:#4A6080;font-size:0.88rem;margin:6px 0 0 0;">
        All classified tickets from this session
        </p>
    </div>
    """, unsafe_allow_html=True)

    history = st.session_state.ticket_history

    if not history:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#3A5070;">
            <div style="font-size:3rem;margin-bottom:12px;">🎫</div>
            <div style="font-size:1rem;font-weight:600;color:#4A6080;">No tickets yet</div>
            <div style="font-size:0.83rem;margin-top:6px;">
            Classify messages in the Live Classifier to populate ticket history.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Filter row
        f1, f2, _ = st.columns([2, 2, 4])
        with f1:
            all_priorities = ["All"] + list(set(t["priority"] for t in history))
            filter_priority = st.selectbox("Priority", all_priorities, label_visibility="collapsed")
        with f2:
            all_depts = ["All Departments"] + list(set(t["department"] for t in history))
            filter_dept = st.selectbox("Department", all_depts, label_visibility="collapsed")

        filtered = history
        if filter_priority != "All":
            filtered = [t for t in filtered if t["priority"] == filter_priority]
        if filter_dept != "All Departments":
            filtered = [t for t in filtered if t["department"] == filter_dept]

        st.markdown(
            f'<p style="color:#3A5070;font-size:0.78rem;margin:8px 0 12px 0;">'
            f'Showing {len(filtered)} of {len(history)} tickets</p>',
            unsafe_allow_html=True,
        )

        for ticket in filtered:
            priority  = ticket["priority"]
            pcolor    = PRIORITY_COLORS.get(priority, "#888")
            emoji_i   = INTENT_EMOJI.get(ticket["intent"], "🎯")
            emoji_p   = PRIORITY_EMOJI.get(priority, "")
            conf_pct  = ticket["confidence"] * 100

            st.markdown(f"""
            <div style="background:#0D1526;border-left:3px solid {pcolor};
            border-radius:0 10px 10px 0;padding:14px 18px;margin:8px 0;
            border-top:1px solid #1C2A3A;border-right:1px solid #1C2A3A;
            border-bottom:1px solid #1C2A3A;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                    <span style="font-family:JetBrains Mono,monospace;font-size:12px;
                    color:#0066FF;font-weight:600;">{ticket['id']}</span>
                    <span style="color:#2A4060;font-size:11px;font-family:JetBrains Mono,monospace;">
                    {ticket['date']} · {ticket['timestamp']}</span>
                </div>
                <div style="color:#C8D3E0;font-size:0.9rem;margin:6px 0 10px 0;
                line-height:1.4;">{ticket['text']}</div>
                <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;">
                    <span style="background:#0D1D33;border:1px solid #1C3A5A;color:#7AADFF;
                    border-radius:5px;padding:3px 10px;font-size:11px;font-weight:600;">
                    {emoji_i} {ticket['display_label']}</span>
                    <span style="background:#0A1A0A;border:1px solid #1A3A1A;color:#34C759;
                    border-radius:5px;padding:3px 10px;font-size:11px;font-weight:600;">
                    🏢 {ticket['department']}</span>
                    <span style="background:{pcolor}15;border:1px solid {pcolor}40;
                    color:{pcolor};border-radius:5px;padding:3px 10px;
                    font-size:11px;font-weight:700;">
                    {emoji_p} {priority.upper()}</span>
                    <span style="color:#2A4060;font-size:11px;font-family:JetBrains Mono,monospace;
                    margin-left:4px;">{conf_pct:.1f}% conf</span>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
elif "Model Evaluation" in page:

    st.markdown("""
    <div style="padding:8px 0 24px 0;">
        <h1 style="font-family:Sora,sans-serif;font-size:2rem;font-weight:800;
        color:#E8EDF5;margin:0;letter-spacing:-1px;">Model Evaluation</h1>
        <p style="color:#4A6080;font-size:0.88rem;margin:6px 0 0 0;">
        Baseline model performance metrics and confusion matrix analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

    report_dir = Path("ml/evaluation/reports")
    cm_path    = report_dir / "confusion_matrix.png"
    pc_path    = report_dir / "per_class_metrics.png"
    csv_path   = report_dir / "metrics_report.csv"
    yaml_path  = report_dir / "summary.yaml"

    if not report_dir.exists() or not cm_path.exists():
        st.markdown("""
        <div style="background:#0D1526;border:1px solid #1C2A3A;border-radius:10px;
        padding:24px;text-align:center;">
            <div style="font-size:2rem;margin-bottom:10px;">🔬</div>
            <div style="color:#4A6080;font-size:0.9rem;">No evaluation reports found.</div>
            <div style="color:#2A4060;font-size:0.8rem;margin-top:8px;">Run the evaluator first:</div>
        </div>
        """, unsafe_allow_html=True)
        st.code("py ml/evaluation/evaluator.py", language="bash")
    else:
        # Summary metrics
        if yaml_path.exists():
            with open(yaml_path) as f:
                summary = yaml.safe_load(f)

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Accuracy",        f"{summary.get('accuracy', 0)*100:.1f}%")
            m2.metric("F1 Weighted",     f"{summary.get('f1_weighted', 0)*100:.1f}%")
            m3.metric("F1 Macro",        f"{summary.get('f1_macro', 0)*100:.1f}%")
            m4.metric("Test Samples",    summary.get("test_samples", "—"))
            m5.metric("Classes",         summary.get("num_classes", 9))

        st.markdown("<br>", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["🟦 Confusion Matrix", "📊 Per-Class Metrics", "📋 Metrics Table"])

        with tab1:
            st.image(str(cm_path), use_column_width=True)

        with tab2:
            st.image(str(pc_path), use_column_width=True)

        with tab3:
            if csv_path.exists():
                df_report = pd.read_csv(csv_path)

                # Style the dataframe
                def color_score(val):
                    if isinstance(val, float):
                        if val >= 0.9:
                            return "color: #34C759"
                        elif val >= 0.7:
                            return "color: #FFD700"
                        else:
                            return "color: #FF2D55"
                    return ""

                styled = df_report.style.applymap(
                    color_score,
                    subset=["precision", "recall", "f1-score"]
                ).format({
                    "precision": "{:.4f}",
                    "recall":    "{:.4f}",
                    "f1-score":  "{:.4f}",
                })
                st.dataframe(styled, use_container_width=True, hide_index=True)

        # Run evaluator button
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.75rem;font-weight:600;color:#4A6080;
        text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">
        Refresh Evaluation
        </div>""", unsafe_allow_html=True)
        st.code("py ml/evaluation/evaluator.py", language="bash")