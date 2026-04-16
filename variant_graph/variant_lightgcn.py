"""
LightGCN-based collaborative filtering using PyTorch Geometric.

This script loads the training interaction matrix, builds a user-item
graph, trains a LightGCN model with BPR loss, generates Top-K
recommendations, and evaluates the results using the shared evaluation
framework.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import joblib
import json
import time

import torch
from scipy.sparse import load_npz

try:
    from torch_geometric.nn import LightGCN
    from torch_geometric.utils import structured_negative_sampling
except ImportError:
    print("Please install PyTorch Geometric first: pip install torch-geometric")
    sys.exit(1)

from baselines.evaluate import evaluate_recommendations, print_results_table

HIDDEN_DIM = 256
NUM_LAYERS = 2
EPOCHS = 1000
LR = 0.005

# Use CUDA if available. We deliberately avoid MPS here because sparse
# graph operations can be unreliable on Apple Metal for this setup.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

print("\nLoading pipeline outputs...")

test_df = pd.read_csv("pipeline_output/test.csv")
train_matrix = load_npz("pipeline_output/train_interaction_matrix.npz")
user_train_history = joblib.load("pipeline_output/user_train_history.pkl")

num_users, num_items = train_matrix.shape
num_nodes = num_users + num_items
print(f"Users: {num_users} | Items: {num_items} | Total Nodes: {num_nodes}")

print("\nBuilding the bipartite graph...")

train_matrix = train_matrix.tocoo()

# Users stay in the range [0, num_users - 1].
# Items are offset so they occupy a separate index range.
row_idx = train_matrix.row
col_idx = train_matrix.col + num_users

# These directed edges are used during negative sampling for BPR training.
directed_edges = np.vstack((row_idx, col_idx))
train_edge_index = torch.tensor(directed_edges, dtype=torch.long).to(device)

# LightGCN message passing is done on an undirected graph,
# so we add the reverse item -> user edges as well.
reversed_edges = np.vstack((col_idx, row_idx))
edge_index_np = np.hstack((directed_edges, reversed_edges))
edge_index = torch.tensor(edge_index_np, dtype=torch.long).to(device)

print(f"Total graph edges after adding reverse links: {edge_index.shape[1]:,}")

# Initialize the graph recommender.
model = LightGCN(
    num_nodes=num_nodes,
    embedding_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(f"\nStarting LightGCN training for {EPOCHS} epochs...")
t0 = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()

    # For every positive user-item interaction, sample a negative item.
    i, j, k = structured_negative_sampling(train_edge_index, num_nodes=num_nodes)

    # Run message passing and fetch the current node embeddings.
    emb = model.get_embedding(edge_index)
    user_emb, pos_emb, neg_emb = emb[i], emb[j], emb[k]

    # BPR loss trains the model to score positive items above negative ones.
    pos_scores = (user_emb * pos_emb).sum(dim=-1)
    neg_scores = (user_emb * neg_emb).sum(dim=-1)
    bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

    # Add light regularization to keep the embeddings stable.
    l2_reg = (
        user_emb.norm(2).pow(2) +
        pos_emb.norm(2).pow(2) +
        neg_emb.norm(2).pow(2)
    ) / (2 * len(i))

    loss = bpr_loss + 1e-4 * l2_reg

    loss.backward()
    optimizer.step()

    if epoch % 25 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d}/{EPOCHS} | BPR Loss: {loss.item():.4f}")

total_elapsed = time.time() - t0
print(f"Training completed in {total_elapsed:.1f} seconds")

print("\nGenerating Top-K recommendations for evaluation...")
model.eval()

with torch.no_grad():
    # Split final embeddings back into user and item blocks.
    final_embs = model.get_embedding(edge_index)
    final_user_embs = final_embs[:num_users].cpu().numpy()
    final_item_embs = final_embs[num_users:].cpu().numpy()

# Ground truth contains the held-out test apps for each user.
ground_truth = (
    test_df.groupby("user_id")["app_id"]
    .apply(list)
    .to_dict()
)

recommendations = {}
K_MAX = 10
pred_time = time.time()

for user_id in ground_truth:
    seen_items = user_train_history.get(user_id, [])

    # Score every candidate item for this user.
    user_vector = final_user_embs[user_id]
    scores = np.dot(final_item_embs, user_vector)

    # Remove training items so we only recommend unseen apps.
    valid_seen = [i for i in seen_items if i < num_items]
    scores[valid_seen] = -np.inf

    # Select the Top-K highest-scoring unseen items.
    top_items = np.argpartition(scores, -K_MAX)[-K_MAX:]
    top_items = top_items[np.argsort(scores[top_items])[::-1]]

    recommendations[user_id] = [int(i) for i in top_items]

print(f"Predictions generated in {time.time() - pred_time:.1f} seconds")

results = evaluate_recommendations(
    recommendations=recommendations,
    ground_truth=ground_truth,
    k_values=[5, 10]
)

print_results_table("Variant C — Graph Neural Network (LightGCN)", results)

os.makedirs("pipeline_output/results", exist_ok=True)
with open("pipeline_output/results/variant_lightgcn.json", "w") as f:
    json.dump(results, f, indent=2)

print("Results saved to pipeline_output/results/variant_lightgcn.json")