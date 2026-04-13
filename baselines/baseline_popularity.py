"""
baselines/baseline_popularity.py
─────────────────────────────────
Baseline 1 — Most Popular Items

Simple idea:
    Recommend the apps that appear most often in the training data.
    Same ranking for every user, but skip anything they've already seen.

Reference (for this kind of baseline approach):
    Cremonesi, Koren, & Turrin (2010),
    "Performance of recommender algorithms on top-N recommendation tasks"
    RecSys '10. https://doi.org/10.1145/1864708.1864721
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import joblib
import time

from baselines.evaluate import evaluate_recommendations, print_results_table
K_MAX = 10   # we only need top-10 recommendations at most

# loading pipeline outputs 
print("Loading pipeline outputs …")
train_df          = pd.read_csv("pipeline_output/train.csv")
test_df           = pd.read_csv("pipeline_output/test.csv")
user_train_history = joblib.load("pipeline_output/user_train_history.pkl")
print(f"  Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")

# Building global popularity ranking
print("\nComputing global popularity ranking …")

# counting how often each app shows up in the training data
popularity = (
    train_df.groupby("app_id")
    .size()
    .sort_values(ascending=False)
    .reset_index(name="count")
)

# this gives us a global ranking from most to least popular
ranked_apps = popularity["app_id"].tolist()

print(f"  Total apps ranked: {len(ranked_apps):,}")
print(f"  Top-10 popular app ids: {ranked_apps[:10]}")

# Building the ground truth from test set
# what users actually interacted with in the test set
ground_truth = (
    test_df.groupby("user_id")["app_id"]
    .apply(list)
    .to_dict()
)

print(f"\n  Test users with ground truth: {len(ground_truth):,}")

# Generating recommendations
print("Generating recommendations …")

t0 = time.time()
recommendations = {}

for user_id in ground_truth:
    # items the user has already seen during training
    seen = set(user_train_history.get(user_id, []))

    recs = []

    # go down the popularity list and skip seen items
    for app_id in ranked_apps:
        if app_id not in seen:
            recs.append(app_id)
        if len(recs) == K_MAX:
            break

    recommendations[user_id] = recs

elapsed = time.time() - t0
print(f"  Done in {elapsed:.2f}s for {len(recommendations):,} users")

# quick sanity check for one user
sample_user = list(recommendations.keys())[0]
print(f"\n  Sanity check — user {sample_user}:")
print(f"    Seen in train: {len(user_train_history.get(sample_user, []))} items")
print(f"    Recommended:   {recommendations[sample_user]}")
print(f"    Ground truth:  {ground_truth[sample_user]}")

# making sure we did not recommend anything already seen
overlap = set(recommendations[sample_user]) & set(user_train_history.get(sample_user, []))
print(f"    Train overlap (must be 0): {len(overlap)}")

# Evaluating
print("\nEvaluating …")

results = evaluate_recommendations(
    recommendations=recommendations,
    ground_truth=ground_truth,
    k_values=[5, 10]
)

print_results_table("Baseline 1 — Most Popular Items", results)

# Saving Results
import json

os.makedirs("pipeline_output/results", exist_ok=True)

with open("pipeline_output/results/baseline_popularity.json", "w") as f:
    json.dump(results, f, indent=2)

print("Results saved -> pipeline_output/results/baseline_popularity.json")