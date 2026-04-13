"""
baselines/baseline_knn_cf.py
User-Based KNN Collaborative Filtering (baseline)

What this does:
    1. Learn user-user similarity using scikit-surprise (cosine similarity).
    2. Grab the similarity matrix from the trained model.
    3. For each user, find the most similar neighbours.
    4. Score items based on what those neighbours interacted with.
    5. Recommend the top items the user has not seen yet.

Uses Surprise for similarity + numpy/scipy for fast scoring.

References:
    Hug (2020) - Surprise library (JOSS)
    Herlocker et al. (1999) - classic collaborative filtering paper (SIGIR)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import joblib
import json
import time
from scipy.sparse import csr_matrix, load_npz

from surprise import Dataset, Reader, KNNWithMeans

from baselines.evaluate import evaluate_recommendations, print_results_table

# config
N_NEIGHBORS = 20   # number of neighbours to look at
K_MAX       = 10   # max number of recommendations per user

# loading data
print("Loading pipeline outputs …")
train_df            = pd.read_csv("pipeline_output/train.csv")
test_df             = pd.read_csv("pipeline_output/test.csv")
train_matrix        = load_npz("pipeline_output/train_interaction_matrix.npz")  # (10000, 7988)
user_train_history  = joblib.load("pipeline_output/user_train_history.pkl")

n_users, n_items = train_matrix.shape
print(f"  Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")
print(f"  Interaction matrix: {n_users} × {n_items}")

# step 1: train similarity model
print("\nStep 1: Training KNNWithMeans (user cosine, k=%d) …" % N_NEIGHBORS)

# Surprise expects a rating column, so we just use 1.0 for all interactions
train_ratings           = train_df[["user_id", "app_id"]].copy()
train_ratings["rating"] = 1.0

reader        = Reader(rating_scale=(1, 1))
surprise_data = Dataset.load_from_df(train_ratings[["user_id", "app_id", "rating"]], reader)
trainset      = surprise_data.build_full_trainset()

print(f"  Surprise trainset: {trainset.n_users} users, {trainset.n_items} items")

t0  = time.time()
algo = KNNWithMeans(
    k=N_NEIGHBORS,
    sim_options={"name": "cosine", "user_based": True, "min_support": 1},
    verbose=False
)
algo.fit(trainset)
print(f"  Training done in {time.time() - t0:.1f}s")

# step 2: get similarity matrix
print("\nStep 2: Extracting similarity matrix …")

# algo.sim -> (num_users, num_users)
sim_matrix = algo.sim
print(f"  Similarity matrix shape: {sim_matrix.shape}")

# build mapping between raw user ids and Surprise internal ids
n_surprise_users = trainset.n_users
raw_uid_to_inner = {}   # raw user_id -> internal index
inner_to_raw_uid = {}   # internal index -> raw user_id

for inner_uid in range(n_surprise_users):
    raw_uid = trainset.to_raw_uid(inner_uid)
    raw_uid_to_inner[raw_uid] = inner_uid
    inner_to_raw_uid[inner_uid] = raw_uid

print(f"  Surprise users: {n_surprise_users}")

# step 3: scoring recommendations
print("\nStep 3: Running user-based CF scoring …")

# ground truth from test set
ground_truth = (
    test_df.groupby("user_id")["app_id"]
    .apply(list)
    .to_dict()
)
print(f"  Test users: {len(ground_truth)}")

# convert to float32 for faster math
# rows -> users, cols -> items, values -> interaction (1)
X = train_matrix.astype(np.float32)

t0 = time.time()
recommendations = {}

for user_id in ground_truth:
    seen = set(user_train_history.get(user_id, []))

    # if user wasn’t in training, we can’t compute similarity
    if user_id not in raw_uid_to_inner:
        recommendations[user_id] = []   # cold start
        continue

    inner_uid = raw_uid_to_inner[user_id]
    sim_vec   = sim_matrix[inner_uid]

    # remove self similarity and keep top neighbours
    sim_copy = sim_vec.copy()
    sim_copy[inner_uid] = 0.0
    top_k_inner = np.argsort(sim_copy)[-N_NEIGHBORS:]

    # get neighbour data
    top_k_raw_uids = [inner_to_raw_uid[i] for i in top_k_inner]
    top_k_sims     = sim_copy[top_k_inner]

    neighbour_rows  = X[top_k_raw_uids, :].toarray()
    weighted_scores = top_k_sims @ neighbour_rows

    # remove items the user already saw
    seen_arr = np.array(list(seen), dtype=int)
    valid_seen = seen_arr[(seen_arr >= 0) & (seen_arr < n_items)]
    weighted_scores[valid_seen] = 0.0

    # pick top items
    top_items = np.argsort(weighted_scores)[-K_MAX:][::-1].tolist()
    recommendations[user_id] = [int(i) for i in top_items if weighted_scores[i] > 0]

    # note: some users might end up with fewer than K items
    # if their neighbours have very weak similarity

total_elapsed = time.time() - t0
print(f"  Scoring done in {total_elapsed:.1f}s")

# quick sanity check
sample_user = list(recommendations.keys())[0]
print(f"\n  Sanity check — user {sample_user}:")
print(f"    Seen in train:  {len(user_train_history.get(sample_user, []))} items")
print(f"    Recommended:    {recommendations[sample_user]}")
print(f"    Ground truth:   {ground_truth[sample_user]}")

overlap = set(recommendations[sample_user]) & set(user_train_history.get(sample_user, []))
print(f"    Train overlap (must be 0): {len(overlap)}")

cold_starts = sum(1 for u in ground_truth if not recommendations.get(u))
print(f"    Cold-start users (no recs): {cold_starts}")

# evaluating
print("\nEvaluating …")
results = evaluate_recommendations(
    recommendations=recommendations,
    ground_truth=ground_truth,
    k_values=[5, 10]
)

print_results_table("Baseline 2 — User-Based KNN-CF (scikit-surprise)", results)

# saving
os.makedirs("pipeline_output/results", exist_ok=True)
with open("pipeline_output/results/baseline_knn_cf.json", "w") as f:
    json.dump(results, f, indent=2)

print("Results saved -> pipeline_output/results/baseline_knn_cf.json")