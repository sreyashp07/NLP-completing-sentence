"""
Exploratory Data Analysis for CustomerIntent dataset.
Run: py notebooks/01_eda.py
"""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/raw/customer_support_dataset.csv")

print("=== Dataset Overview ===")
print(f"Shape: {df.shape}")
print(f"\nIntent distribution:\n{df['intent'].value_counts()}")
print(f"\nPriority distribution:\n{df['priority'].value_counts()}")
print(f"\nDepartment distribution:\n{df['department'].value_counts()}")
print(f"\nAvg text length: {df['text'].str.len().mean():.1f} chars")
print(f"Min text length: {df['text'].str.len().min()} chars")
print(f"Max text length: {df['text'].str.len().max()} chars")
