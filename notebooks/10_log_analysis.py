"""
Prediction log analysis notebook.

Analyzes the JSONL prediction log to surface
patterns in customer support ticket classification.

Run: py notebooks/10_log_analysis.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.inference.prediction_logger import get_log_stats, read_prediction_logs

print("\n=== PREDICTION LOG ANALYSIS ===\n")

stats = get_log_stats()

if stats["total_logged"] == 0:
    print("No predictions logged yet.")
    print("Run the API and make some predictions first.")
    print("\nExample:")
    print("  py -m uvicorn app.main:app --reload")
    print("  py -m streamlit run streamlit_app/app.py")
else:
    print(f"Total predictions logged : {stats['total_logged']}")
    print(f"Average confidence       : {stats['avg_confidence']*100:.1f}%")
    print(f"Most common intent       : {stats['most_common_intent']}")
    print("\nIntent distribution:")
    for intent, count in sorted(
        stats["intent_distribution"].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        bar = "█" * count
        print(f"  {intent:25} {bar} ({count})")
