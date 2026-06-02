"""CustomerIntent AI - Professional dark UI without AI-generated emoji feel."""
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

st.set_page_config(
    page_title="CustomerIntent",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8000/api/v1"

PRIORITY_COLORS = {
    "critical": "#FF2D55",
    "high":     "#FF6B35",
    "medium":   "#FFD700",
    "low":      "#34C759",
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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, .stApp {
    background: #050810 !important;
    font-family: 'Inter', -apple-system, sans-serif;
    color: #E8EDF5;
}

/* Animated gradient background */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(circle at 20% 30%, rgba(0, 102, 255, 0.08) 0%, transparent 50%),
        radial-gradient(circle at 80% 70%, rgba(123, 47, 190, 0.06) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080C18 0%, #050810 100%) !important;
    border-right: 1px solid rgba(28, 42, 58, 0.5);
    backdrop-filter: blur(20px);
}
[data-testid="stSidebar"] * { color: #C8D3E0 !important; }

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(15, 27, 45, 0.6) 0%, rgba(10, 21, 32, 0.4) 100%);
    border: 1px solid rgba(28, 42, 58, 0.5);
    border-radius: 14px;
    padding: 18px 22px;
    backdrop-filter: blur(10px);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
[data-testid="metric-container"]:hover {
    border-color: rgba(0, 102, 255, 0.4);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 102, 255, 0.12);
}
[data-testid="stMetricValue"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 700;
    font-size: 1.25rem !important;
    color: #E8EDF5 !important;
    letter-spacing: -0.3px;
}
[data-testid="stMetricLabel"] {
    color: #5A6B85 !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
}

.stButton > button {
    background: linear-gradient(135deg, #0066FF 0%, #0044CC 100%);
    color: white !important;
    border: 1px solid rgba(0, 102, 255, 0.3);
    border-radius: 10px;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 10px 20px;
    letter-spacing: 0.3px;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 2px 8px rgba(0, 102, 255, 0.15);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0077FF 0%, #0055DD 100%);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 102, 255, 0.35);
    border-color: rgba(0, 102, 255, 0.6);
}
.stButton > button:active {
    transform: translateY(0);
}

.stTextArea textarea {
    background: rgba(13, 21, 38, 0.8) !important;
    border: 1px solid rgba(28, 42, 58, 0.6) !important;
    border-radius: 12px !important;
    color: #E8EDF5 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    transition: all 0.25s ease;
    backdrop-filter: blur(10px);
}
.stTextArea textarea:focus {
    border-color: #0066FF !important;
    box-shadow: 0 0 0 3px rgba(0, 102, 255, 0.15) !important;
    background: rgba(13, 21, 38, 0.95) !important;
}

[data-testid="stRadio"] label {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    transition: color 0.2s ease;
}

hr { border-color: rgba(28, 42, 58, 0.5) !important; }

::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #1C2A3A, #0F1B2D);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover { background: #2A3F5A; }

.main-title {
    font-family: 'Inter', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #FFFFFF 0%, #7AADFF 50%, #0066FF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1.5px;
    margin: 0;
    line-height: 1.1;
}

.subtitle {
    color: #5A6B85;
    font-size: 0.9rem;
    margin-top: 8px;
    font-weight: 400;
    letter-spacing: 0.2px;
}

.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    color: #5A6B85;
    text-transform: uppercase;
    letter-spacing: 1.8px;
    margin-bottom: 12px;
}

.keyword-chip {
    display: inline-block;
    background: linear-gradient(135deg, rgba(13, 29, 51, 0.9) 0%, rgba(13, 29, 51, 0.6) 100%);
    border: 1px solid rgba(28, 58, 90, 0.6);
    border-radius: 6px;
    padding: 5px 12px;
    margin: 3px;
    font-size: 12px;
    color: #7AADFF;
    font-family: 'JetBrains Mono', monospace;
    transition: all 0.2s ease;
    backdrop-filter: blur(10px);
}
.keyword-chip:hover {
    border-color: rgba(122, 173, 255, 0.6);
    transform: translateY(-1px);
}

.gradient-line {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0, 102, 255, 0.5), transparent);
    margin: 28px 0;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border-radius: 8px;
    font-size: 0.78rem;
    font-weight: 500;
    font-family: 'Inter', sans-serif;
    transition: all 0.2s ease;
}
.status-online {
    background: rgba(52, 199, 89, 0.1);
    border: 1px solid rgba(52, 199, 89, 0.3);
    color: #34C759;
}
.status-offline {
    background: rgba(255, 45, 85, 0.1);
    border: 1px solid rgba(255, 45, 85, 0.3);
    color: #FF2D55;
}
</style>
""", unsafe_allow_html=True)

if "ticket_history" not in st.session_state:
    st.session_state.ticket_history: List[Dict] = []
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction: Optional[Dict] = None
if "api_online" not in st.session_state:
    st.session_state.api_online = False


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


with st.sidebar:
    st.markdown("""
    <div style="padding:12px 0 18px 0;">
        <div style="font-family:Inter,sans-serif;font-size:1.15rem;font-weight:800;
        color:#E8EDF5;letter-spacing:-0.5px;">CustomerIntent</div>
        <div style="font-size:0.7rem;color:#3A5070;margin-top:3px;
        font-family:JetBrains Mono,monospace;letter-spacing:0.5px;">v1.0.0 / production</div>
    </div>
    """, unsafe_allow_html=True)

    api_ok = check_api_health()
    if api_ok:
        st.markdown(
            '<div class="status-pill status-online">● Connected</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-pill status-offline">● Disconnected</div>',
            unsafe_allow_html=True,
        )
        st.code("uvicorn app.main:app --reload", language="bash")

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Live Classifier", "Analytics", "Ticket History", "Model Evaluation"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown("""
    <div style="font-size:0.7rem;color:#3A5070;line-height:2;
    font-family:JetBrains Mono,monospace;">
    <div style="color:#4A7090;font-weight:600;margin-bottom:6px;letter-spacing:1px;">SYSTEM</div>
    model  &nbsp;&nbsp;tfidf+lr<br>
    intents&nbsp;&nbsp;9 classes<br>
    env    &nbsp;&nbsp;&nbsp;&nbsp;production<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Clear", use_container_width=True):
            st.session_state.ticket_history = []
            st.session_state.last_prediction = None
            st.rerun()
    with col_b:
        ticket_count = len(st.session_state.ticket_history)
        st.markdown(
            f'<div style="text-align:center;padding:9px;background:rgba(13,21,38,0.6);'
            f'border:1px solid rgba(28,42,58,0.5);border-radius:8px;font-size:0.78rem;'
            f'color:#6B7C93;font-family:JetBrains Mono,monospace;">{ticket_count} tickets</div>',
            unsafe_allow_html=True,
        )


if "Live Classifier" in page:
    st.markdown("""
    <div style="padding:12px 0 28px 0;">
        <h1 class="main-title">Live Intent Classifier</h1>
        <p class="subtitle">Real-time customer support classification with smart routing</p>
    </div>
    """, unsafe_allow_html=True)

    left_col, right_col = st.columns([3, 2], gap="large")

    with left_col:
        user_input = st.text_area(
            "Customer Message",
            placeholder="Type a customer support message...",
            height=140,
            key="main_input",
            label_visibility="collapsed",
        )

        btn_col, toggle_col = st.columns([1, 2])
        with btn_col:
            classify_btn = st.button("Classify Intent", type="primary", use_container_width=True)
        with toggle_col:
            live_mode = st.toggle("Live typing mode", value=False)

        if live_mode and user_input and len(user_input.strip()) >= 10 and api_ok:
            prediction = predict_intent(user_input)
            if prediction:
                st.session_state.last_prediction = prediction

        if classify_btn:
            if not user_input or len(user_input.strip()) < 5:
                st.warning("Enter at least 5 characters")
            elif not api_ok:
                st.error("API offline. Run: uvicorn app.main:app --reload")
            else:
                with st.spinner("Analyzing"):
                    prediction = predict_intent(user_input)
                if prediction:
                    st.session_state.last_prediction = prediction
                    add_to_history(user_input, prediction)

    with right_col:
        st.markdown('<div class="section-label">Quick Examples</div>', unsafe_allow_html=True)
        for ex in EXAMPLE_MESSAGES:
            if st.button(ex[:45] + ('...' if len(ex)>45 else ''),
                        key=f"ex_{hash(ex)}", use_container_width=True):
                st.session_state["main_input"] = ex
                st.rerun()

    if st.session_state.last_prediction:
        pred = st.session_state.last_prediction
        primary = pred["primary_intent"]
        intent = primary["intent"]
        priority = primary["priority"]
        conf = primary["confidence"]

        st.markdown('<div class="gradient-line"></div>', unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Intent Detected", primary['display_label'])
        k2.metric("Routed To",        primary["department"])
        k3.metric("Priority",         priority.upper())
        k4.metric("Confidence",       f"{conf*100:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)

        chart_col, info_col = st.columns([3, 2], gap="large")

        with chart_col:
            st.markdown('<div class="section-label">Confidence Distribution</div>', unsafe_allow_html=True)

            top = pred["top_intents"]
            labels = [t["display_label"] for t in top]
            values = [round(t["confidence"] * 100, 2) for t in top]
            bar_colors = ["#0066FF", "#0044AA", "#1F3A6B"]

            fig = go.Figure(go.Bar(
                x=values, y=labels, orientation="h",
                marker=dict(color=bar_colors, line=dict(width=0)),
                text=[f"{v:.1f}%" for v in values],
                textposition="inside",
                textfont=dict(color="white", size=13, family="Inter"),
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#5A6B85", family="Inter"),
                height=200,
                margin=dict(l=0, r=10, t=8, b=8),
                xaxis=dict(showgrid=False, showticklabels=False, range=[0, 110]),
                yaxis=dict(showgrid=False, tickfont=dict(size=12, color="#C8D3E0")),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with info_col:
            st.markdown('<div class="section-label">Key Signals</div>', unsafe_allow_html=True)

            if pred.get("keywords"):
                kw_html = "".join([
                    f'<span class="keyword-chip">#{kw}</span>'
                    for kw in pred["keywords"]
                ])
                st.markdown(kw_html, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            pcolor = PRIORITY_COLORS.get(priority, "#888")
            st.markdown(
                f'<div style="margin-top:6px;">'
                f'<span style="background:linear-gradient(135deg,{pcolor}22,{pcolor}11);'
                f'border:1px solid {pcolor}55;color:{pcolor};border-radius:8px;'
                f'padding:7px 16px;font-size:11px;font-weight:700;letter-spacing:1.2px;'
                f'font-family:Inter,sans-serif;display:inline-block;">'
                f'{priority.upper()} PRIORITY</span></div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<p style="color:#2A4060;font-size:10.5px;margin-top:16px;'
                f'font-family:JetBrains Mono,monospace;letter-spacing:0.3px;">'
                f'/ {pred["processing_time_ms"]}ms · {pred["model_used"]}</p>',
                unsafe_allow_html=True,
            )


elif "Analytics" in page:
    st.markdown("""
    <div style="padding:12px 0 28px 0;">
        <h1 class="main-title">Analytics</h1>
        <p class="subtitle">Real-time insights from classified tickets</p>
    </div>
    """, unsafe_allow_html=True)

    history = st.session_state.ticket_history

    if not history:
        st.markdown("""
        <div style="text-align:center;padding:80px 20px;color:#3A5070;">
            <div style="font-family:Inter;font-size:1.05rem;font-weight:600;color:#4A6080;">No data yet</div>
            <div style="font-size:0.85rem;margin-top:8px;color:#3A5070;">
            Classify some messages to see analytics</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        total = len(history)
        critical_cnt = sum(1 for t in history if t["priority"] == "critical")
        avg_conf = sum(t["confidence"] for t in history) / total
        dept_count = len(set(t["department"] for t in history))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Tickets",   total)
        m2.metric("Critical",        critical_cnt)
        m3.metric("Avg Confidence",  f"{avg_conf*100:.1f}%")
        m4.metric("Departments",     dept_count)

        st.markdown("<br>", unsafe_allow_html=True)

        chart1, chart2 = st.columns(2, gap="large")

        with chart1:
            st.markdown('<div class="section-label">Intent Distribution</div>', unsafe_allow_html=True)

            intent_counts: Dict = {}
            for t in history:
                intent_counts[t["display_label"]] = intent_counts.get(t["display_label"], 0) + 1

            fig_pie = px.pie(
                names=list(intent_counts.keys()),
                values=list(intent_counts.values()),
                color_discrete_sequence=[
                    "#0066FF","#0044AA","#0088FF","#00AAFF",
                    "#0033CC","#3377FF","#005599","#0099DD","#1F3A6B",
                ],
                hole=0.55,
            )
            fig_pie.update_traces(
                textfont=dict(family="Inter", size=11, color="white"),
                marker=dict(line=dict(color="#050810", width=2)),
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#5A6B85", family="Inter"),
                legend=dict(font=dict(size=10, color="#8A9BB0")),
                height=320,
                margin=dict(l=0, r=0, t=10, b=10),
            )
            st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

        with chart2:
            st.markdown('<div class="section-label">Priority Breakdown</div>', unsafe_allow_html=True)

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
                textfont=dict(color="#8A9BB0", size=13, family="Inter"),
            ))
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#5A6B85", family="Inter"),
                height=320,
                margin=dict(l=0, r=0, t=10, b=10),
                xaxis=dict(showgrid=False, tickfont=dict(size=11, color="#8A9BB0")),
                yaxis=dict(showgrid=False, showticklabels=False),
                bargap=0.4,
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})


elif "Ticket History" in page:
    st.markdown("""
    <div style="padding:12px 0 28px 0;">
        <h1 class="main-title">Ticket History</h1>
        <p class="subtitle">All classified tickets from this session</p>
    </div>
    """, unsafe_allow_html=True)

    history = st.session_state.ticket_history

    if not history:
        st.markdown("""
        <div style="text-align:center;padding:80px 20px;color:#3A5070;">
            <div style="font-family:Inter;font-size:1.05rem;font-weight:600;color:#4A6080;">No tickets yet</div>
            <div style="font-size:0.85rem;margin-top:8px;color:#3A5070;">
            Classify messages to populate ticket history</div>
        </div>
        """, unsafe_allow_html=True)
    else:
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
            f'<p style="color:#3A5070;font-size:0.78rem;margin:12px 0 16px 0;'
            f'font-family:JetBrains Mono,monospace;">'
            f'{len(filtered)} of {len(history)} tickets</p>',
            unsafe_allow_html=True,
        )

        for ticket in filtered:
            priority  = ticket["priority"]
            pcolor    = PRIORITY_COLORS.get(priority, "#888")
            conf_pct  = ticket["confidence"] * 100

            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(13,21,38,0.7),rgba(13,21,38,0.4));
            border-left:3px solid {pcolor};
            border-radius:0 12px 12px 0;padding:16px 22px;margin:10px 0;
            border-top:1px solid rgba(28,42,58,0.5);
            border-right:1px solid rgba(28,42,58,0.5);
            border-bottom:1px solid rgba(28,42,58,0.5);
            transition: all 0.25s ease;
            backdrop-filter: blur(10px);">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                    <span style="font-family:JetBrains Mono,monospace;font-size:12px;
                    color:#7AADFF;font-weight:600;letter-spacing:0.5px;">{ticket['id']}</span>
                    <span style="color:#2A4060;font-size:10.5px;font-family:JetBrains Mono,monospace;">
                    {ticket['date']} / {ticket['timestamp']}</span>
                </div>
                <div style="color:#C8D3E0;font-size:0.92rem;margin:8px 0 12px 0;
                line-height:1.5;font-weight:400;">{ticket['text']}</div>
                <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;">
                    <span style="background:rgba(13,29,51,0.8);border:1px solid rgba(28,58,90,0.6);color:#7AADFF;
                    border-radius:6px;padding:4px 11px;font-size:11px;font-weight:500;
                    font-family:Inter,sans-serif;">{ticket['display_label']}</span>
                    <span style="background:rgba(10,26,10,0.6);border:1px solid rgba(26,58,26,0.6);color:#34C759;
                    border-radius:6px;padding:4px 11px;font-size:11px;font-weight:500;
                    font-family:Inter,sans-serif;">{ticket['department']}</span>
                    <span style="background:{pcolor}15;border:1px solid {pcolor}40;
                    color:{pcolor};border-radius:6px;padding:4px 11px;
                    font-size:11px;font-weight:700;letter-spacing:0.8px;
                    font-family:Inter,sans-serif;">{priority.upper()}</span>
                    <span style="color:#2A4060;font-size:11px;font-family:JetBrains Mono,monospace;
                    margin-left:auto;letter-spacing:0.3px;">{conf_pct:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)


elif "Model Evaluation" in page:
    st.markdown("""
    <div style="padding:12px 0 28px 0;">
        <h1 class="main-title">Model Evaluation</h1>
        <p class="subtitle">Performance metrics and confusion matrix analysis</p>
    </div>
    """, unsafe_allow_html=True)

    report_dir = Path("ml/evaluation/reports")
    cm_path = report_dir / "confusion_matrix.png"
    pc_path = report_dir / "per_class_metrics.png"
    csv_path = report_dir / "metrics_report.csv"
    yaml_path = report_dir / "summary.yaml"

    if not report_dir.exists() or not cm_path.exists():
        st.markdown("""
        <div style="background:rgba(13,21,38,0.6);border:1px solid rgba(28,42,58,0.5);
        border-radius:12px;padding:32px;text-align:center;backdrop-filter:blur(10px);">
            <div style="color:#4A6080;font-size:0.95rem;font-weight:500;">No evaluation reports found</div>
            <div style="color:#2A4060;font-size:0.82rem;margin-top:10px;">Run the evaluator first</div>
        </div>
        """, unsafe_allow_html=True)
        st.code("python ml/evaluation/evaluator.py", language="bash")
    else:
        if yaml_path.exists():
            with open(yaml_path) as f:
                summary = yaml.safe_load(f)

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Accuracy",     f"{summary.get('accuracy', 0)*100:.1f}%")
            m2.metric("F1 Weighted",  f"{summary.get('f1_weighted', 0)*100:.1f}%")
            m3.metric("F1 Macro",     f"{summary.get('f1_macro', 0)*100:.1f}%")
            m4.metric("Test Samples", summary.get("test_samples", "—"))
            m5.metric("Classes",      summary.get("num_classes", 9))

        st.markdown("<br>", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Per-Class Metrics", "Metrics Table"])

        with tab1:
            st.image(str(cm_path), use_column_width=True)
        with tab2:
            st.image(str(pc_path), use_column_width=True)
        with tab3:
            if csv_path.exists():
                df_report = pd.read_csv(csv_path)
                st.dataframe(df_report, use_container_width=True, hide_index=True)
