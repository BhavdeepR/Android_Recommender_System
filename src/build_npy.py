"""
build_npy.py
────────────
Encodes raw user/app IDs, builds app_info_sample.npy aligned to app integer
indices, and runs the full data pipeline — saving all outputs to pipeline_output/.

Run with:
    python3.12 build_npy.py
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz
from sklearn.preprocessing import StandardScaler
import joblib
import os

os.makedirs("pipeline_output", exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# STEP 0 — Load Raw Interactions & Encode IDs
# ═══════════════════════════════════════════════════════════════
print("Loading raw myket.csv …")
df_raw = pd.read_csv("data/myket.csv", index_col=False, on_bad_lines="skip")
df_raw.columns = ["user_id", "app_name", "timestamp", "state_label", "features"]
df_raw = df_raw[["user_id", "app_name", "timestamp"]].copy()
df_raw["timestamp"] = pd.to_numeric(df_raw["timestamp"], errors="coerce")
df_raw = df_raw.dropna(subset=["timestamp"])

print(f"  Raw rows:  {len(df_raw):,}")
print(f"  Raw users: {df_raw['user_id'].nunique():,}")
print(f"  Raw apps:  {df_raw['app_name'].nunique():,}")

# Build deterministic int encodings (sorted for reproducibility)
unique_users = sorted(df_raw["user_id"].unique())
unique_apps  = sorted(df_raw["app_name"].unique())

user2idx = {u: i for i, u in enumerate(unique_users)}  # 0 … 9,999
app2idx  = {a: i for i, a in enumerate(unique_apps)}   # 0 … 7,987
idx2app  = {i: a for a, i in app2idx.items()}

df_raw["user_id"] = df_raw["user_id"].map(user2idx)
df_raw["app_id"]  = df_raw["app_name"].map(app2idx)
df = df_raw[["user_id", "app_id", "timestamp"]].sort_values(
    ["user_id", "timestamp"]
).reset_index(drop=True)

print(f"\nAfter encoding:")
print(f"  Interactions: {len(df):,}")
print(f"  Users: {df['user_id'].nunique():,}  (max={df['user_id'].max()})")
print(f"  Apps:  {df['app_id'].nunique():,}  (max={df['app_id'].max()})")

# Save encodings for later use by models
joblib.dump(user2idx, "pipeline_output/user2idx.pkl")
joblib.dump(app2idx,  "pipeline_output/app2idx.pkl")
joblib.dump(idx2app,  "pipeline_output/idx2app.pkl")
print("  Saved: user2idx.pkl, app2idx.pkl, idx2app.pkl")

# ═══════════════════════════════════════════════════════════════
# STEP 1 — Build app_info_sample.npy
#   Shape: (7988, 33) — 3 numeric + 30 one-hot category dummies
#   Indexed by app integer id (0 … 7987)
#   Zero vector for apps not found in app_info_sample.csv
# ═══════════════════════════════════════════════════════════════
print("\nBuilding app_info_sample.npy from CSV …")
app_info = pd.read_csv("data/app_info_sample.csv")
# One-hot encode 30 categories
categories = sorted(app_info["category_en"].dropna().unique())   # 30 unique
print(f"  Categories ({len(categories)}): {categories[:5]} …")

cat_dummies = pd.get_dummies(app_info["category_en"], prefix="cat")
# Ensure all 30 categories are present as columns
expected_cols = [f"cat_{c}" for c in categories]
for col in expected_cols:
    if col not in cat_dummies.columns:
        cat_dummies[col] = 0
cat_dummies = cat_dummies[expected_cols]  # enforce column order

# Numeric block: installs, rating, rating_count
num_block = app_info[["installs", "rating", "rating_count"]].fillna(0).values

# Full feature matrix for apps IN the CSV
feature_block = np.hstack([num_block, cat_dummies.values.astype(np.float32)])  # (7606, 33)

# Allocate output array (7988, 33) — zero fallback for missing apps
n_apps  = len(unique_apps)
n_feats = feature_block.shape[1]  # 33
app_features = np.zeros((n_apps, n_feats), dtype=np.float32)

# Map CSV app_name → app integer id and fill rows
app_name_col = app_info["app_name"].values
filled = 0
for row_idx, app_name in enumerate(app_name_col):
    if app_name in app2idx:
        app_features[app2idx[app_name]] = feature_block[row_idx]
        filled += 1

print(f"  Apps matched & filled: {filled:,} / {n_apps:,}")
print(f"  Apps using zero fallback: {n_apps - filled:,}")
print(f"  Feature matrix shape: {app_features.shape}")

np.save("data/app_info_sample.npy", app_features)
print("  Saved: data/app_info_sample.npy")

# ═══════════════════════════════════════════════════════════════
# STEP 2 — Normalize App Features (same as original pipeline)
# ═══════════════════════════════════════════════════════════════
num_features = app_features[:, :3].copy()
cat_features = app_features[:, 3:].copy()

num_features[:, 0] = np.log1p(num_features[:, 0])  # log(installs)
num_features[:, 2] = np.log1p(num_features[:, 2])  # log(rating_count)

has_features = ~np.all(app_features == 0, axis=1)
print(f"\nApps with features: {has_features.sum():,} / {len(app_features):,}")
print(f"Apps using zero fallback: {(~has_features).sum():,}")

scaler = StandardScaler()
num_features[has_features] = scaler.fit_transform(num_features[has_features])

app_feature_matrix = np.hstack([num_features, cat_features]).astype(np.float32)
print(f"App feature matrix shape: {app_feature_matrix.shape}")

# ═══════════════════════════════════════════════════════════════
# STEP 3 — Temporal Train / Val / Test Split (N=5 per user)
# ═══════════════════════════════════════════════════════════════
N = 5
train_rows, val_rows, test_rows = [], [], []

for user_id, group in df.groupby("user_id"):
    group = group.sort_values("timestamp")
    n = len(group)
    if n < N * 2 + 1:
        train_rows.append(group)
        continue
    train_rows.append(group.iloc[:-(N * 2)])
    val_rows.append(group.iloc[-(N * 2):-N])
    test_rows.append(group.iloc[-N:])

train_df = pd.concat(train_rows).reset_index(drop=True)
val_df   = pd.concat(val_rows).reset_index(drop=True)
test_df  = pd.concat(test_rows).reset_index(drop=True)

print(f"\nN = {N} held-out interactions per user per split")
print(f"  Train: {len(train_df):,} interactions | {train_df['user_id'].nunique():,} users")
print(f"  Val:   {len(val_df):,} interactions  | {val_df['user_id'].nunique():,} users")
print(f"  Test:  {len(test_df):,} interactions  | {test_df['user_id'].nunique():,} users")

# ═══════════════════════════════════════════════════════════════
# STEP 4 — User History Dict
# ═══════════════════════════════════════════════════════════════
user_train_history = (
    train_df.sort_values("timestamp")
    .groupby("user_id")["app_id"]
    .apply(list)
    .to_dict()
)

sample_user = list(user_train_history.keys())[0]
print(f"\nSample user {sample_user} train history "
      f"({len(user_train_history[sample_user])} interactions):")
print(f"  {user_train_history[sample_user][:10]} …")

# ═══════════════════════════════════════════════════════════════
# STEP 5 — Sparse Interaction Matrix (train only)
# ═══════════════════════════════════════════════════════════════
n_users = df["user_id"].max() + 1   # 10,000
n_items = df["app_id"].max() + 1    # 7,988

train_matrix = csr_matrix(
    (np.ones(len(train_df)),
     (train_df["user_id"].values, train_df["app_id"].values)),
    shape=(n_users, n_items)
)

print(f"\nInteraction matrix: {train_matrix.shape}")
print(f"Sparsity: {1 - train_matrix.nnz / (n_users * n_items):.4%}")

# ═══════════════════════════════════════════════════════════════
# STEP 6 — Save All Pipeline Outputs
# ═══════════════════════════════════════════════════════════════
train_df.to_csv("pipeline_output/train.csv", index=False)
val_df.to_csv("pipeline_output/val.csv",     index=False)
test_df.to_csv("pipeline_output/test.csv",   index=False)

np.save("pipeline_output/app_feature_matrix.npy", app_feature_matrix)
save_npz("pipeline_output/train_interaction_matrix.npz", train_matrix)
joblib.dump(scaler,              "pipeline_output/feature_scaler.pkl")
joblib.dump(user_train_history,  "pipeline_output/user_train_history.pkl")

print("\n── Pipeline outputs saved ──")
print("  train.csv                     → temporal model + CF")
print("  val.csv                       → validation")
print("  test.csv                      → final evaluation")
print("  app_feature_matrix.npy        → hybrid model content features")
print("  train_interaction_matrix.npz  → hybrid model CF component")
print("  feature_scaler.pkl            → consistent feature scaling")
print("  user_train_history.pkl        → context window / seen-items mask")
print("  user2idx.pkl / app2idx.pkl / idx2app.pkl  → ID encodings")
print("\nDone ✓")
