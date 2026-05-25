"""
Sentiment-aware routing demo.

Shows how combining intent classification with
sentiment analysis improves ticket routing decisions.

Run: py notebooks/08_sentiment_routing_demo.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.inference.predictor import IntentRouter
from ml.inference.routing_engine import determine_queue
from ml.preprocessing.sentiment_analyzer import analyze_sentiment

router = IntentRouter(model_type="baseline")

TEST_CASES = [
    {
        "text": "My payment failed but money got deducted",
        "description": "Standard payment issue",
    },
    {
        "text": "THIS IS FRAUD!!! My money was STOLEN!!! I am calling police!!!",
        "description": "Escalated frustrated customer",
    },
    {
        "text": "Can you please add dark mode? Would be nice to have",
        "description": "Polite feature request",
    },
    {
        "text": "urgent!!! account locked cannot login deadline today",
        "description": "Urgent account issue",
    },
    {
        "text": "I want to cancel my subscription thank you",
        "description": "Calm cancellation request",
    },
]

print("\n" + "="*70)
print("  SENTIMENT-AWARE ROUTING DEMO")
print("="*70)

for case in TEST_CASES:
    text = case["text"]
    prediction = router.route(text)
    prediction["text"] = text
    sentiment = analyze_sentiment(text)
    routing = determine_queue(prediction)

    print(f"\n  Scenario  : {case['description']}")
    print(f"  Text      : {text[:60]}")
    print(f"  Intent    : {prediction['primary_intent']['display_label']}")
    print(f"  Sentiment : {sentiment['sentiment']} (urgent={sentiment['is_urgent']})")
    print(f"  Queue     : {routing['queue'].upper()}")
    print(f"  SLA       : {routing['sla_minutes']} minutes")
    print(f"  Reason    : {routing['reason']}")
    print("-"*70)
