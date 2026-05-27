"""Unit tests for session manager."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from app.utils.session_manager import SessionManager

def make_pred(intent, priority, confidence):
    return {"primary_intent": {
        "intent": intent, "priority": priority,
        "confidence": confidence, "department": "Billing Team",
    }}

def test_create_session_returns_id():
    sm = SessionManager()
    sid = sm.create_session()
    assert isinstance(sid, str)
    assert len(sid) > 0

def test_get_session_returns_dict():
    sm = SessionManager()
    sid = sm.create_session()
    session = sm.get_session(sid)
    assert isinstance(session, dict)
    assert session["id"] == sid

def test_add_prediction_increments_count():
    sm = SessionManager()
    sid = sm.create_session()
    sm.add_prediction(sid, make_pred("payment_issue", "critical", 0.9))
    session = sm.get_session(sid)
    assert session["ticket_count"] == 1

def test_get_session_stats():
    sm = SessionManager()
    sid = sm.create_session()
    sm.add_prediction(sid, make_pred("payment_issue", "critical", 0.9))
    sm.add_prediction(sid, make_pred("payment_issue", "critical", 0.8))
    stats = sm.get_session_stats(sid)
    assert stats["ticket_count"] == 2
    assert stats["top_intent"] == "payment_issue"

def test_list_sessions():
    sm = SessionManager()
    sm.create_session()
    sm.create_session()
    assert len(sm.list_sessions()) >= 2

def test_clear_session():
    sm = SessionManager()
    sid = sm.create_session()
    sm.clear_session(sid)
    assert sm.get_session(sid) is None
