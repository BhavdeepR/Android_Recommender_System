"""
graph_build_npy.py

Builds the full preprocessing pipeline:
- encodes raw user/app IDs into integer indices
- creates an app feature matrix aligned to app indices
- performs a temporal train/validation/test split
- saves all generated artifacts into pipeline_output/

Run with:
    python3.12 graph_build_npy.py
"""

import os #used for file/directory operations
import re #used for regular expression operations

import joblib #used for serializing python objects like dictionaries and scalers
import numpy as np #used for numerical operations
import pandas as pd #used for data manipulation and analysis
from scipy.sparse import csr_matrix, save_npz #used for sparse matrix operations
from sklearn.preprocessing import StandardScaler #used for standardizing features

os.makedirs("pipeline_output", exist_ok=True)

# Loads the raw interaction file and keeps only the columns needed for modeling.
print("Loading raw myket.csv...")
df_raw = pd.read_csv("data/myket.csv", index_col=False, on_bad_lines="skip")
df_raw.columns = ["user_id", "app_name", "timestamp", "state_label", "features"]
df_raw = df_raw[["user_id", "app_name", "timestamp"]].copy()

# Convert timestamps safely and drop rows where the timestamp is invalid.
df_raw["timestamp"] = pd.to_numeric(df_raw["timestamp"], errors="coerce") #coerce invalid timestamp to NaN
df_raw = df_raw.dropna(subset=["timestamp"])

print(f"  Raw rows:  {len(df_raw):,}")
print(f"  Raw users: {df_raw['user_id'].nunique():,}")
print(f"  Raw apps:  {df_raw['app_name'].nunique():,}")

# Create stable integer IDs so users and apps can be used by downstream models.
# Sorting keeps the mapping reproducible across runs.
unique_users = sorted(df_raw["user_id"].unique())
unique_apps = sorted(df_raw["app_name"].unique())

# Creates a dict mapping each raw user ID to integer index, same for apps and vice versa.
user2idx = {u: i for i, u in enumerate(unique_users)}
app2idx = {a: i for i, a in enumerate(unique_apps)} #maps each app name to an integer app index.
idx2app = {i: a for a, i in app2idx.items()} #maps the integer app indices back to app names.

df_raw["user_id"] = df_raw["user_id"].map(user2idx) #replaces raw user ids with integer indices.
df_raw["app_id"] = df_raw["app_name"].map(app2idx) #replaces raw app names with integer indices.

# Keep the encoded interaction table sorted by user and time, this is required for the sequence models to learn the temporal patterns in app usage.
df = df_raw[["user_id", "app_id", "timestamp"]].sort_values(
    ["user_id", "timestamp"]
).reset_index(drop=True)

print("\nAfter encoding:")
print(f"Interactions: {len(df):,}")
print(f"Users: {df['user_id'].nunique():,}  (max={df['user_id'].max()})")
print(f"Apps:  {df['app_id'].nunique():,}  (max={df['app_id'].max()})")

# Save the ID mappings so training and inference can use the same indexing.
joblib.dump(user2idx, "pipeline_output/user2idx.pkl") #used to convert raw user ids to integer indices.
joblib.dump(app2idx, "pipeline_output/app2idx.pkl") #used to convert raw app names to integer indices.
joblib.dump(idx2app, "pipeline_output/idx2app.pkl") #used to convert integer app indices back to app names.
print("Saved: user2idx.pkl, app2idx.pkl, idx2app.pkl")

print("\nBuilding refined app features from CSV ...")
app_info = pd.read_csv("data/app_info_sample.csv") #metadata

# Start from the master app list so feature rows stay perfectly aligned with app_id.
features_df = pd.DataFrame({"app_name": unique_apps})
features_df = features_df.merge(app_info, on="app_name", how="left")

# Track which apps are missing metadata so the model can distinguish real zeros
# from rows where metadata was unavailable.
features_df["is_missing_metadata"] = features_df["category_en"].isna().astype(int)

# Fill missing numeric values before feature engineering.
features_df["installs"] = features_df["installs"].fillna(0)
features_df["rating"] = features_df["rating"].fillna(0)
features_df["rating_count"] = features_df["rating_count"].fillna(0)

# Bucket installs into coarse tiers so the model can capture broad popularity bands.
bins_installs = [-1, 1, 10000, 100000, 1000000, np.inf]
labels_installs = ["zero", "low", "medium", "high", "mega"]
features_df["install_tier"] = pd.cut(
    features_df["installs"], bins=bins_installs, labels=labels_installs
)

# Bucket ratings for a more stable categorical signal than raw values alone.
bins_rating = [-1, 0.1, 3.0, 4.0, 4.5, 5.1]
labels_rating = ["zero", "poor", "average", "good", "excellent"]
features_df["rating_tier"] = pd.cut(
    features_df["rating"], bins=bins_rating, labels=labels_rating
)

# Extract a publisher-like token from the package name.
# This gives the model a developer-level feature that is often more specific than category.
# e.g. "com.spotify.music" -> "spotify"

def extract_publisher(pkg):
    parts = str(pkg).split(".")
    if len(parts) >= 3 and parts[0] in ["com", "air", "org", "net", "ir"]:
        return parts[1]
    elif len(parts) >= 2:
        return parts[0]
    return "unknown"

features_df["publisher"] = features_df["app_name"].apply(extract_publisher)

# Keep only the most frequent publishers and collapse the rest into "other"
# so the feature space does not become too sparse.
top_publishers = features_df["publisher"].value_counts().nlargest(500).index
features_df["publisher"] = features_df["publisher"].where(
    features_df["publisher"].isin(top_publishers), "other"
)

# Extract simple keyword tokens from app names/package names.
# These act as lightweight content features.
def extract_keywords(pkg):
    tokens = re.split(r"[^a-zA-Z0-9]", str(pkg).lower())
    stop_words = {"com", "air", "org", "net", "ir", "app", "www", "android", "google", "play"}
    return [t for t in tokens if len(t) > 2 and t not in stop_words]

#list of all the unique keywords in the app names along with their frequencies.
all_keywords = features_df["app_name"].apply(extract_keywords).explode().value_counts()#list of all the unique keywords in the app names along with their frequencies.

# Keep only the most common keywords to avoid adding too much noise.
top_keywords = all_keywords.nlargest(20).index

keyword_cols = []
for kw in top_keywords: #e.g. kw_photo, kw_music, kw_game, kw_news etc.
    col_name = f"kw_{kw}"
    features_df[col_name] = (
        features_df["app_name"].str.lower().str.contains(kw, na=False).astype(int)#true if the app_name contains the keyword
    )
    keyword_cols.append(col_name) #adds the new column to the list of keyword columns

# One-hot encode the categorical metadata features.
features_df["category_en"] = features_df["category_en"].fillna("Unknown")
categorical_cols = ["category_en", "install_tier", "rating_tier", "publisher"]
cat_dummies = pd.get_dummies(features_df[categorical_cols], sparse=False)

# Scale numeric features after log-transforming the heavy-tailed ones.
# Fit the scaler only on rows that actually have metadata.
num_features = features_df[["installs", "rating", "rating_count"]].copy()
num_features["installs"] = np.log1p(num_features["installs"]) #log1p is used to handle the skewness of the data
num_features["rating_count"] = np.log1p(num_features["rating_count"])

has_features = features_df["is_missing_metadata"] == 0
scaler = StandardScaler()
if has_features.sum() > 0:
    num_features.loc[has_features] = scaler.fit_transform(num_features.loc[has_features]) #standardizes the features (mean 0 and variance 1)

# Stack all feature blocks into one final matrix aligned with app_id order.
final_blocks = [
    num_features.values.astype(np.float32),
    features_df[["is_missing_metadata"]].values.astype(np.float32),
    features_df[keyword_cols].values.astype(np.float32),
    cat_dummies.values.astype(np.float32),
]

app_feature_matrix = np.hstack(final_blocks) #stack all feature blocks into one final matrix aligned with app_id order.
print(f"  Total apps processed: {len(app_feature_matrix):,}")
print(f"  Final feature dimension: {app_feature_matrix.shape[1]}")
print(f"  Apps using zero fallback (missing metadata): {features_df['is_missing_metadata'].sum():,}")

np.save("data/app_info_sample.npy", app_feature_matrix)
print("  Saved: data/app_info_sample.npy")

# Split each user's history in time order.
# Users with too little history stay entirely in training.
N = 5 #number of held-out interactions per user per split
train_rows, val_rows, test_rows = [], [], [] #lists to store the training, validation, and test data.

for user_id, group in df.groupby("user_id"): 
    group = group.sort_values("timestamp") 
    n = len(group) #get the number of interactions for the current user

    if n < N * 2 + 1:
        train_rows.append(group)
        continue

    train_rows.append(group.iloc[:-(N * 2)]) #takes all interactions except the last 2*N interactions
    val_rows.append(group.iloc[-(N * 2):-N]) #takes the last 2*N interactions except the last N interactions
    test_rows.append(group.iloc[-N:]) #takes the last N interactions

train_df = pd.concat(train_rows).reset_index(drop=True)
val_df = pd.concat(val_rows).reset_index(drop=True)
test_df = pd.concat(test_rows).reset_index(drop=True)

print(f"\nN = {N} held-out interactions per user per split")
print(f"  Train: {len(train_df):,} interactions | {train_df['user_id'].nunique():,} users")
print(f"  Val:   {len(val_df):,} interactions  | {val_df['user_id'].nunique():,} users")
print(f"  Test:  {len(test_df):,} interactions  | {test_df['user_id'].nunique():,} users")

# Build each user's training history for sequence context and seen-item filtering.
user_train_history = (
    train_df.sort_values("timestamp")
    .groupby("user_id")["app_id"]
    .apply(list)
    .to_dict()
)

sample_user = list(user_train_history.keys())[0]
print(f"\nSample user {sample_user} train history ({len(user_train_history[sample_user])} interactions):")
print(f"  {user_train_history[sample_user][:10]} ...")

# Build the sparse user-item interaction matrix from training data only.
n_users = df["user_id"].max() + 1
n_items = df["app_id"].max() + 1

train_matrix = csr_matrix(
    (np.ones(len(train_df)), 
    (train_df["user_id"].values, train_df["app_id"].values)),#creates a sparse matrix with 1s at the positions where there are interactions
    shape=(n_users, n_items),#the shape of the matrix is the number of users by the number of items
) 
# So if user 3 interacted with item 10 in training, the matrix gets a 1 at position (3, 10).


print(f"\nInteraction matrix: {train_matrix.shape}")
print(f"Sparsity: {1 - train_matrix.nnz / (n_users * n_items):.4%}")

# Save every artifact needed by downstream models and evaluation scripts.
train_df.to_csv("pipeline_output/train.csv", index=False)
val_df.to_csv("pipeline_output/val.csv", index=False)
test_df.to_csv("pipeline_output/test.csv", index=False)

np.save("pipeline_output/app_feature_matrix.npy", app_feature_matrix)
save_npz("pipeline_output/train_interaction_matrix.npz", train_matrix)
joblib.dump(scaler, "pipeline_output/feature_scaler.pkl")
joblib.dump(user_train_history, "pipeline_output/user_train_history.pkl")

print("\nPipeline outputs saved")
print("  train.csv                     -> temporal model + CF")
print("  val.csv                       -> validation")
print("  test.csv                      -> final evaluation")
print("  app_feature_matrix.npy        -> hybrid model content features")
print("  train_interaction_matrix.npz  -> hybrid model CF component")
print("  feature_scaler.pkl            -> consistent feature scaling")
print("  user_train_history.pkl        -> context window / seen-items mask")
print("  user2idx.pkl / app2idx.pkl / idx2app.pkl  -> ID encodings")
print("\nDone")


