"""
LightGCN implementation for Google Colab with CUDA support.

This version is designed to run in Colab and use the GPU when one is available.
It trains a graph-based collaborative filtering model on the user-app interaction graph,
generates Top-K recommendations, and evaluates them with the shared project code.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))#used to find parent directory

import pandas as pd
import numpy as np
import joblib #used for serializing python objects like dictionaries and scalers, it loads user_train_history.pkl
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

from baselines.evaluate import evaluate_recommendations, print_results_table #imports the evaluation functions from the parent directory's baselines.evaluate module #used to evaluate the recommendations

#Hyperparameters for LightGCN
HIDDEN_DIM = 256 #The dimensionality of the learned embeddings and the hidden layers
NUM_LAYERS = 2 #Number of propagation layers in LightGCN
EPOCHS = 500 #Number of passes through the training data
LR = 0.001 #Learning rate for the optimizer

# In Colab, use CUDA if a GPU is attached.
# If not, continue on CPU
if not torch.cuda.is_available():
    print("CUDA is not available.")
    device = torch.device("cpu")
else:
    device = torch.device("cuda")

print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

print("\nLoading pipeline outputs...")

# This block verifies that preprocessing has already been run and that required input files exist.
if not os.path.exists("pipeline_output/train_interaction_matrix.npz"):
    print("Pipeline outputs not found. Run `python src/build_npy.py` first.")
    sys.exit(1)

test_df = pd.read_csv("pipeline_output/test.csv")
train_matrix = load_npz("pipeline_output/train_interaction_matrix.npz")
user_train_history = joblib.load("pipeline_output/user_train_history.pkl")

num_users, num_items = train_matrix.shape
num_nodes = num_users + num_items
print(f"Users: {num_users} | Items: {num_items} | Total Nodes: {num_nodes}")

print("\nBuilding the bipartite graph...")

train_matrix = train_matrix.tocoo()
# converts the sparse matrix to coordinate format
# COO format stores nonzero entries as row indices, column indices, and values.
# this is convenient for extracting graph edges.

# Users keep their original ids.
row_idx = train_matrix.row

# Item ids are shifted by num_users so users and items share one node space.
col_idx = train_matrix.col + num_users

# These user -> item edges are the observed positive interactions used in training.
directed_edges = np.vstack((row_idx, col_idx))#stacks the row and column indices to create an array of edges.
train_edge_index = torch.tensor(directed_edges, dtype=torch.long).to(device)#converts the array of edges to a PyTorch tensor and moves it to the specified device.
# This edge list is used by structured_negative_sampling to sample training triplets.

# LightGCN propagates messages on an undirected graph,
# so we also add the reverse item -> user edges.
# adding reverse edges allows information to flow in both directions between users and items.
reversed_edges = np.vstack((col_idx, row_idx))
edge_index_np = np.hstack((directed_edges, reversed_edges))
edge_index = torch.tensor(edge_index_np, dtype=torch.long).to(device)

print(f"Total graph edges after adding reverse links: {edge_index.shape[1]:,}")

# LightGCN learns user and item embeddings directly from the interaction graph.
model = LightGCN(
    num_nodes=num_nodes,
    embedding_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(f"\nStarting LightGCN training for {EPOCHS} epochs on {device}...")
t0 = time.time()

# Main training loop over all epochs
for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()

    # For each positive user-item edge, sample one negative item
    # that the same user has not interacted with.
    # this creates training triplets for pairwise ranking loss.
    i, j, k = structured_negative_sampling(train_edge_index, num_nodes=num_nodes)#samples triplets of (user, positive item, negative item)

    # Recompute node embeddings after graph propagation for the current step.
    # computes node embeddings and selects the sampled user/positive/negative vectors.
    emb = model.get_embedding(edge_index)
    user_emb, pos_emb, neg_emb = emb[i], emb[j], emb[k]

    # BPR loss encourages positive items to score higher than sampled negatives.
    # score(user,positive) > score(user,negative)
    # teaches the model to rank true items above sampled negatives.
    pos_scores = (user_emb * pos_emb).sum(dim=-1)#dot product between user and positive item embeddings
    neg_scores = (user_emb * neg_emb).sum(dim=-1)#dot product between user and negative item embeddings
    bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

    # A small L2 penalty helps keep embedding values from growing too large.
    # improves stability and prevents overfitting.
    l2_reg = (
        user_emb.norm(2).pow(2) +
        pos_emb.norm(2).pow(2) +
        neg_emb.norm(2).pow(2)
    ) / (2 * len(i))

    loss = bpr_loss + 1e-4 * l2_reg
    # bpr_loss: The main loss pushing correct items to have higher scores.
    # 1e-4 * l2_reg: A tiny weight on L2 regularization to keep embeddings small and stable.
    # updates model weights to reduce the computed loss.
    loss.backward()
    optimizer.step()
    if epoch % 25 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d}/{EPOCHS} | BPR Loss: {loss.item():.4f}")

total_elapsed = time.time() - t0
print(f"Training completed in {total_elapsed:.1f} seconds")

print("\nGenerating Top-K recommendations for evaluation...")
model.eval()

with torch.no_grad():
    # Compute final embeddings once, then reuse them for ranking.
    final_embs = model.get_embedding(edge_index)
    final_user_embs = final_embs[:num_users].cpu().numpy()
    final_item_embs = final_embs[num_users:].cpu().numpy()

# Build the ground-truth test dictionary: user_id -> held-out app_ids.
# this is needed because evaluation metrics compare the model’s recommendations with the user’s held-out test items.
ground_truth = (
    test_df.groupby("user_id")["app_id"] #groups the test data by user_id and selects the app_id column.
    .apply(list) #applies a function to each group, in this case, converting the app_ids to a list.
    .to_dict()
)
# Converts the result into a dictionary where keys are user_ids and values are lists of held-out app_ids.
recommendations = {}
K_MAX = 10 # number of recommendations to generate for each user.
pred_time = time.time()

for user_id in ground_truth:
    seen_items = user_train_history.get(user_id, []) #gets the list of items the user has already interacted with.

    # Score every item for this user using a dot product in embedding space.
    # The higher the dot product, the more aligned the user and item embeddings are, meaning higher predicted preference.
    user_vector = final_user_embs[user_id]
    scores = np.dot(final_item_embs, user_vector)

    # Mask training items so we only recommend unseen apps.
    valid_seen = [i for i in seen_items if i < num_items]
    scores[valid_seen] = -np.inf

    # Select the Top-K highest-scoring items.
    top_items = np.argpartition(scores, -K_MAX)[-K_MAX:] #finds indices that would sort the scores array and then takes the last K_MAX indices
    top_items = top_items[np.argsort(scores[top_items])[::-1]] #sorts the top_items based on their scores in descending order

    recommendations[user_id] = [int(i) for i in top_items] #converts the top_items to a list of integers and stores it in the recommendations dictionary.

print(f"Predictions generated in {time.time() - pred_time:.1f} seconds")

# Evaluate the recommendation lists using the shared project evaluation code.
results = evaluate_recommendations(
    recommendations=recommendations,
    ground_truth=ground_truth,
    k_values=[5, 10]
)
# Takes the generated recommendations and ground-truth (true) app usages, then calculates metrics like recall@k and NDCG@k using the shared evaluation helper.

print_results_table("Variant C.1: Pure LightGCN (Colab/CUDA)", results)

# Save both the evaluation metrics and the generated recommendation lists.
output_data = {
    "metrics": results,
    "recommendations": recommendations
}

os.makedirs("pipeline_output/results", exist_ok=True)
with open("pipeline_output/results/variant_lightgcn.json", "w") as f:
    json.dump(output_data, f, indent=2)

print("Results saved to pipeline_output/results/variant_lightgcn.json")

# This script turns 
# user-item interactions into a graph, 
# learns vector embeddings for users and items with LightGCN, 
# then recommends items whose embeddings are most similar to each user’s embedding, 
# while excluding items the user already interacted with.