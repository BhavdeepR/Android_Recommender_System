"""
LightGCN implementation for Google Colab with CUDA support, enhanced with item metadata.

This version is designed for Colab and will use the GPU whenever one is available.
It trains a graph-based collaborative filtering model on the user-app interaction graph,
combines LightGCN item embeddings with metadata-based item embeddings learned from app features,
produces Top-K recommendations, and evaluates them with the shared project evaluation code.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd #used to load the test data
import numpy as np  #used for numerical operations and array manipulation
import joblib       #used to save and load model weights and other Python objects
import json         #used to save and load the evaluation results in JSON format
import time         #used to measure the execution time of different parts of the script

import torch        #The core deep learning framework used for building and training the neural network model.
from scipy.sparse import load_npz #used to load the sparse training interaction matrix from disk

try:
    from torch_geometric.nn import LightGCN #The LightGCN layer implementation from PyTorch Geometric, used for graph convolutional operations.
    from torch_geometric.utils import structured_negative_sampling #A helper function from PyTorch Geometric used to efficiently sample negative examples for training.
except ImportError:
    print("Please install PyTorch Geometric first: pip install torch-geometric")
    sys.exit(1)

from baselines.evaluate import evaluate_recommendations, print_results_table

HIDDEN_DIM = 256 # dimensionality of the learned embeddings and hidden layers
NUM_LAYERS = 2   # number of propagation layers in LightGCN
EPOCHS = 500     # number of passes through the training data
LR = 0.001       # learning rate for the optimizer

# In Google Colab, use the GPU when one is attached.
# If CUDA is not available, keep going on CPU.
if not torch.cuda.is_available():
    print("\n[WARNING] CUDA is not available.")
    print("In Google Colab, go to Runtime > Change runtime type > Hardware accelerator > T4 GPU\n")
    device = torch.device("cpu")
else:
    device = torch.device("cuda")

print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

print("\nLoading pipeline outputs...")

# This model expects the shared preprocessing pipeline to have already run.
# If those artifacts are missing, it stops early
if not os.path.exists("pipeline_output/train_interaction_matrix.npz"):
    print("Pipeline outputs not found. Run `python graph_build_npy.py` first.")
    sys.exit(1)

test_df = pd.read_csv("pipeline_output/test.csv")
train_matrix = load_npz("pipeline_output/train_interaction_matrix.npz") #This line loads the preprocessed training interaction data from a compressed sparse matrix format (.npz).
user_train_history = joblib.load("pipeline_output/user_train_history.pkl") #This line loads the history of interactions for each user from a Python pickle file.
app_feature_matrix = np.load("pipeline_output/app_feature_matrix.npy") #This line loads the computed metadata features for each app from a NumPy array file.

num_users, num_items = train_matrix.shape #This extracts the number of users and items from the shape of the training interaction matrix.
num_nodes = num_users + num_items          #This calculates the total number of nodes in the bipartite graph (users + items).
print(f"Users: {num_users} | Items: {num_items} | Total Nodes: {num_nodes}")

print("\nBuilding the bipartite graph...")

# converts the sparse matrix to coordinate format
# COO format stores nonzero entries as row indices, column indices, and values.
# this is convenient for extracting graph edges.
train_matrix = train_matrix.tocoo()

# User nodes keep their original IDs: [0, num_users - 1]
row_idx = train_matrix.row

# Item nodes are shifted by num_users so users and items live in the same node space.
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

# Item metadata features are loaded as a dense tensor and moved to the same device.
item_features = torch.tensor(app_feature_matrix, dtype=torch.float32).to(device)
feature_dim = item_features.shape[1]

print(f"Total graph edges after adding reverse links: {edge_index.shape[1]:,}")

# This model blends standard LightGCN ID embeddings with a learned metadata representation for items.
import torch
import torch.nn as nn #PyTorch neural network layers and modules.
import torch.nn.functional as F #functional API, used here for normalization
from torch_geometric.nn import LightGCN

class LightGCN_Metadata(nn.Module):
    # Constructor for the model which receives graph size, counts, feature size, embedding size, and number of graph layers.
    def __init__(self, num_nodes, num_users, num_items, feature_dim, hidden_dim, num_layers):#
        super().__init__() # calls the __init__ method of the parent class LightGCN. In this case, nn.Module
        self.num_users = num_users # Number of users in the graph
        self.num_items = num_items # Number of items in the graph   
        
        # Core LightGCN block for collaborative filtering over the graph structure.
        # num_nodes: includes both users and items.
        # embedding_dim: the latent dimension.
        # num_layers: the number of propagation steps.
        self.lightgcn = LightGCN(num_nodes=num_nodes, embedding_dim=hidden_dim, num_layers=num_layers) 
        
        # A deeper MLP maps raw app metadata into the same embedding space as LightGCN.
        # The bottleneck and regularization help the model learn a cleaner metadata signal.
        self.item_mlp = nn.Sequential(
            nn.Linear(feature_dim, 512), # projects raw metadata features into a 512-dimensional hidden representation.
            nn.LayerNorm(512),           # Normalizes the 512-dimensional output to stabilize training.
            nn.ReLU(),                   # Introduces non-linearity to allow the model to learn complex patterns.
            nn.Dropout(0.4),             # Randomly sets 40% of the activations to zero during training to prevent overfitting.
            nn.Linear(512, hidden_dim),  # Projects the 512-dimensional output to the embedding dimension.
            nn.LayerNorm(hidden_dim),    # Normalizes the output to the embedding dimension to stabilize training.
            nn.ReLU(),                   # Introduces non-linearity to allow the model to learn complex patterns.
            nn.Linear(hidden_dim, hidden_dim) # Maps the embedding to the final dimension used throughout the model.
        )
        # This learnable scalar controls how strongly metadata influences the item embeddings.
        self.meta_weight = nn.Parameter(torch.tensor(0.01))

        # This controls how strongly metadata embeddings affect item embeddings.
        # Starting value is 0.01, meaning metadata starts with very small influence.
        self.temperature = 0.2 

    def get_embedding(self, edge_index, item_features, train_edge_index = None):
        # Turn raw item features into metadata embeddings and normalize them
        # so they stay on a comparable scale.
        item_meta = self.item_mlp(item_features)  #Processes item features through the MLP to produce metadata-based embeddings.
        item_meta = F.normalize(item_meta, p=2, dim=-1) #Normalizes the metadata embeddings to have unit L2 norm.
        
        #Retrieves the learned ID embeddings from the LightGCN component.
        id_embs = self.lightgcn.embedding.weight 

        # Takes the user part of the embeddings
        user_e0 = id_embs[:self.num_users]

        # Takes the item part of the embeddings and adds a scaled metadata signal.
        item_e0 = id_embs[self.num_users:] + (self.meta_weight * item_meta)

        # Concatenates the user and item embeddings
        e0 = torch.cat([user_e0, item_e0], dim=0)
        
        # Initializes the output with the first layer's alpha-weighted embeddings.
        out = e0 * self.lightgcn.alpha[0] 
        #Performs one layer of LightGCN propagation.
        for i in range(self.lightgcn.num_layers):
            e0 = self.lightgcn.convs[i](e0, edge_index) 
            #Adds the result to the output, weighted by the next alpha value.
            out = out + e0 * self.lightgcn.alpha[i + 1] 
        #Returns the final embedding for all nodes.
        return out

#Initializing the model
model = LightGCN_Metadata(
    num_nodes=num_nodes,
    num_users=num_users,
    num_items=num_items,
    feature_dim=feature_dim,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
).to(device)


# The metadata MLP gets stronger weight decay to keep it from overfitting noisy high-cardinality features.
optimizer = torch.optim.Adam([
    #LightGCN parameters.
    {'params': model.lightgcn.parameters(), 'lr': 1e-3},
    # metadata network learns a bit more cautiously and is more strongly regularized.
    {'params': model.item_mlp.parameters(), 'lr': 5e-4, 'weight_decay': 5e-3},
    # Learnable weight for metadata.
    {'params': [model.meta_weight], 'lr': 1e-2}
])

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

print_results_table("Variant C.2: LightGCN + Metadata MLP (Colab/CUDA)", results)

output_data = {
    "metrics": results,
    "recommendations": recommendations
}

os.makedirs("pipeline_output/results", exist_ok=True)
with open("pipeline_output/results/variant_lightgcn_metadata.json", "w") as f:
    json.dump(output_data, f, indent=2)

print("Results saved to pipeline_output/results/variant_lightgcn_metadata.json")




# Diagnosis block: check whether metadata is actually helping
# or being ignored by the model.
print("RUNNING DIAGNOSIS")

# 1. Check how much of the training data comes from apps with missing metadata.
try:
    train_df = pd.read_csv("pipeline_output/train.csv")

    # Rows with all-zero features are treated as apps with missing metadata.
    is_missing_metadata = np.all(app_feature_matrix == 0, axis=1)
    missing_metadata_app_ids = np.where(is_missing_metadata)[0]
    
    interactions_with_missing = train_df[train_df['app_id'].isin(missing_metadata_app_ids)]
    coverage_perc = (len(interactions_with_missing) / len(train_df)) * 100
    
    print(f"[1] Metadata Coverage")
    print(f"    Total Interactions: {len(train_df):,}")
    print(f"    Interactions with Missing Metadata: {len(interactions_with_missing):,}")
    print(f"    Missing Metadata Percentage: {coverage_perc:.2f}%")
    
    if coverage_perc > 30:
        print("    RESULT: A large share of interactions involve missing metadata (>30%), so the metadata branch may be too weak to move the metrics.")
    else:
        print("    RESULT: Metadata coverage looks reasonably healthy (<30%).")
except Exception as e:
    print(f"    [ERROR] Could not calculate metadata coverage: {e}")

# 2. Compare the magnitude of collaborative-filtering embeddings
# against metadata-derived embeddings to see which signal dominates.
try:
    model.eval()
    with torch.no_grad():
        # Get the standard LightGCN embeddings.
        cf_embs_all = model.lightgcn.get_embedding(edge_index)#gets the final embedding
        user_cf_embs = cf_embs_all[:num_users]#isolates user embeddings from the combined matrix
        
        # Rebuild user-side metadata embeddings by averaging metadata embeddings
        # of the items each user interacted with.
        users = train_edge_index[0] #Gets user node indices from positive training edges.
        items = train_edge_index[1] - num_users #Gets item node indices (adjusted by num_users offset)
        item_meta_embs = model.item_mlp(item_features)#computes item embeddings using the MLP
        user_meta_embs = torch.zeros(num_users, item_meta_embs.size(1), device=item_meta_embs.device)   #initializes the user embedding matrix with zeros
        user_meta_embs.index_add_(0, users, item_meta_embs[items])#sums the metadata embeddings of all items interacted with by each user
        user_degrees = torch.bincount(users, minlength=num_users).float().unsqueeze(1)#calculates the number of interactions for each user
        user_degrees = torch.clamp(user_degrees, min=1.0)#prevents division by zero
        user_meta_embs = user_meta_embs / user_degrees #averages the metadata embeddings for each user
        
        # Measure overall and per-user embedding magnitudes.
        cf_norm = torch.norm(user_cf_embs).item() #computes the Euclidean norm of the CF embeddings
        meta_norm = torch.norm(user_meta_embs).item()#computes the Euclidean norm of the metadata embeddings
        
        avg_cf_norm = torch.norm(user_cf_embs, dim=1).mean().item()#computes the average norm per user for CF embeddings
        avg_meta_norm = torch.norm(user_meta_embs, dim=1).mean().item()#computes the average norm per user for metadata embeddings
        
        print(f"\n[2] Embedding Norms")
        print(f"    Total User CF Norm:   {cf_norm:.4f} (Avg/User: {avg_cf_norm:.4f})")
        print(f"    Total User Meta Norm: {meta_norm:.4f} (Avg/User: {avg_meta_norm:.4f})")
        
        ratio = cf_norm / (meta_norm + 1e-8)#calculates the ratio of CF to metadata embedding norms
        print(f"    CF-to-Meta Ratio:     {ratio:.2f}x")
        
        if ratio > 10:
            print("    RESULT: The collaborative-filtering signal is much larger, so the model is probably leaning heavily on the graph and mostly ignoring metadata.")
        elif ratio < 0.1:
            print("    RESULT: The metadata signal is much larger, so the model may be relying too much on metadata.")
        else:
            print("    RESULT: The two signals are in a fairly balanced range.")
except Exception as e:
    print(f"    [ERROR] Could not calculate embedding norms: {e}")

print("="*50)