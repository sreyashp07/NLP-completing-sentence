"""
Batch prediction demo.
Run: py notebooks/04_batch_prediction_demo.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.inference.predictor import IntentRouter

router = IntentRouter(model_type="baseline")

batch = [
    "My payment failed but money got deducted",
    "I want to cancel my subscription",
    "App keeps crashing when I open dashboard",
    "Need GST invoice for last month",
    "Package not delivered for 7 days",
    "Account locked after wrong password",
    "Please add dark mode to mobile app",
    "Refund not received after 10 days",
    "What are your support hours",
]

print("\n=== BATCH PREDICTION DEMO ===\n")
results = router.route_batch(batch)

for r in results:
    p = r["primary_intent"]
    print(f"Text     : {r['text'][:55]}")
    print(f"Intent   : {p['display_label']} ({p['confidence']*100:.1f}%)")
    print(f"Priority : {p['priority'].upper()} → {p['department']}")
    print()
