"""
Reusable ticket card component for Streamlit.
Renders a styled ticket card with priority color coding.
"""

PRIORITY_COLORS = {
    "critical": "#FF2D55",
    "high": "#FF6B35",
    "medium": "#FFD700",
    "low": "#34C759",
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

PRIORITY_EMOJI = {
    "critical": "🚨", "high": "🔴",
    "medium": "🟡", "low": "🟢",
}


def render_ticket_card(ticket: dict) -> str:
    """
    Generate HTML for a styled ticket card.

    Args:
        ticket: Ticket dict with id, text, intent, priority, department, confidence

    Returns:
        HTML string for rendering with st.markdown(..., unsafe_allow_html=True)
    """
    priority = ticket.get("priority", "low")
    pcolor = PRIORITY_COLORS.get(priority, "#888")
    emoji_i = INTENT_EMOJI.get(ticket.get("intent", ""), "🎯")
    emoji_p = PRIORITY_EMOJI.get(priority, "")
    conf_pct = ticket.get("confidence", 0) * 100

    return f"""
    <div style="background:#0D1526;border-left:3px solid {pcolor};
    border-radius:0 10px 10px 0;padding:14px 18px;margin:8px 0;
    border-top:1px solid #1C2A3A;border-right:1px solid #1C2A3A;
    border-bottom:1px solid #1C2A3A;">
        <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
            <span style="font-family:monospace;font-size:12px;color:#0066FF;font-weight:600;">
            {ticket.get('id', 'TKT-0000')}</span>
            <span style="color:#2A4060;font-size:11px;font-family:monospace;">
            {ticket.get('date', '')} · {ticket.get('timestamp', '')}</span>
        </div>
        <div style="color:#C8D3E0;font-size:0.9rem;margin:6px 0 10px 0;">
        {ticket.get('text', '')}</div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;">
            <span style="background:#0D1D33;border:1px solid #1C3A5A;color:#7AADFF;
            border-radius:5px;padding:3px 10px;font-size:11px;font-weight:600;">
            {emoji_i} {ticket.get('display_label', '')}</span>
            <span style="background:{pcolor}15;border:1px solid {pcolor}40;
            color:{pcolor};border-radius:5px;padding:3px 10px;font-size:11px;font-weight:700;">
            {emoji_p} {priority.upper()}</span>
            <span style="color:#2A4060;font-size:11px;font-family:monospace;">
            {conf_pct:.1f}% conf</span>
        </div>
    </div>
    """
