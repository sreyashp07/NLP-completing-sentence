"""
Session manager for tracking user prediction sessions.

Maintains session state for analytics and ticket history.
In production this would use Redis or a database backend.
"""
import uuid
from datetime import datetime
from typing import Dict, List, Optional


class SessionManager:
    """
    Manages prediction sessions for analytics tracking.
    One session = one user's interaction period.
    """

    def __init__(self):
        self._sessions: Dict[str, Dict] = {}

    def create_session(self) -> str:
        """Create a new session and return session ID."""
        session_id = str(uuid.uuid4())[:12]
        self._sessions[session_id] = {
            "id": session_id,
            "created_at": datetime.utcnow().isoformat(),
            "predictions": [],
            "ticket_count": 0,
        }
        return session_id

    def add_prediction(
        self,
        session_id: str,
        prediction: Dict,
    ) -> None:
        """Add a prediction result to a session."""
        if session_id not in self._sessions:
            self.create_session()
        session = self._sessions[session_id]
        primary = prediction.get("primary_intent", {})
        session["predictions"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "intent": primary.get("intent"),
            "confidence": primary.get("confidence"),
            "priority": primary.get("priority"),
        })
        session["ticket_count"] += 1

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session data by ID."""
        return self._sessions.get(session_id)

    def get_session_stats(self, session_id: str) -> Dict:
        """Get statistics for a session."""
        session = self._sessions.get(session_id)
        if not session:
            return {}
        preds = session["predictions"]
        if not preds:
            return {"ticket_count": 0}
        intents = [p["intent"] for p in preds]
        intent_counts: Dict = {}
        for intent in intents:
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        avg_conf = sum(
            p["confidence"] for p in preds if p["confidence"]
        ) / len(preds)
        return {
            "session_id": session_id,
            "ticket_count": session["ticket_count"],
            "avg_confidence": round(avg_conf, 4),
            "intent_distribution": intent_counts,
            "top_intent": max(intent_counts, key=intent_counts.get),
        }

    def list_sessions(self) -> List[str]:
        """Return all active session IDs."""
        return list(self._sessions.keys())

    def clear_session(self, session_id: str) -> None:
        """Remove a session."""
        self._sessions.pop(session_id, None)
