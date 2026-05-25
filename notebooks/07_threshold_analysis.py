"""
Threshold analysis runner notebook.
Run: py notebooks/07_threshold_analysis.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.evaluation.threshold_analyzer import print_threshold_report

print("Running confidence threshold analysis...")
print_threshold_report()
print("Done! Use this to pick the best threshold for your use case.")
print("Recommendation: 0.75 gives good balance of auto-routing and accuracy.")
