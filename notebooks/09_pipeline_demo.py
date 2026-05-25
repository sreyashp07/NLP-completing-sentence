"""
Full pipeline demo showing end-to-end flow.

Demonstrates the complete journey of a customer message:
raw text -> normalize -> clean -> classify -> route -> queue

Run: py notebooks/09_pipeline_demo.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.preprocessing.text_pipeline import TextPipeline
from ml.inference.predictor import IntentRouter
from ml.inference.routing_engine import determine_queue
from ml.preprocessing.priority_detector import detect_priority_signals

router = IntentRouter(model_type="baseline")
pipeline = TextPipeline()

MESSAGES = [
    "My payment failed but money got deducted URGENT!!!",
    "pls help asap acc locked cant login",
    "I recieved wrong item and want refund",
    "Can u add dark mode? thx",
]

print("\n" + "="*70)
print("  END-TO-END PIPELINE DEMO")
print("="*70)

for msg in MESSAGES:
    processed = pipeline.process(msg)
    prediction = router.route(msg)
    prediction["text"] = msg
    routing = determine_queue(prediction)
    signals = detect_priority_signals(msg)

    print(f"\n  Raw Input   : {msg}")
    print(f"  Normalized  : {processed['normalized'][:60]}")
    print(f"  Cleaned     : {processed['cleaned'][:60]}")
    print(f"  Sentiment   : {processed['sentiment']['sentiment']}")
    print(f"  Keywords    : {', '.join(processed['keywords'][:4])}")
    print(f"  Intent      : {prediction['primary_intent']['display_label']}")
    print(f"  Confidence  : {prediction['primary_intent']['confidence']*100:.1f}%")
    print(f"  Queue       : {routing['queue'].upper()}")
    print(f"  SLA         : {routing['sla_minutes']} mins")
    if signals["should_escalate"]:
        print(f"  ESCALATE    : YES - {signals['critical_keywords_found']}")
    print("-"*70)
