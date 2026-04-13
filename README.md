# Android App Recommender System

A recommender system built to recommend Android apps to users based on their implicit interactions (installs/views). The project uses a dataset from the Myket Android App Store.

## Setup & Installation

Install the required packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```
*(Note: `scikit-surprise` requires `numpy<2` to run correctly on current systems).*

## 1. The Data Pipeline (`src/build_npy.py`)

The `src/build_npy.py` script serves as the main data ingestion and preprocessing pipeline. Its primary use cases are:

1.  **ID Encoding:** Reads the raw dataset (`data/myket.csv`) and encodes the raw large integer User IDs and string Category/App package names into contiguous integer indices (e.g., 0 to 9,999 for users, 0 to 7,987 for apps).
2.  **Feature Processing:** Reads `data/app_info_sample.csv` to one-hot encode app categories and log-scale continuous features (like installs and ratings). It then builds a normalized dense feature matrix (`app_info_sample.npy`).
3.  **Data Splitting:** Uses a user-based temporal split. By ordering each user's interactions by timestamp, the last *N=5* apps go to the test set, the prior *N=5* go to the validation set, and the rest remain in the training set.
4.  **Sparse Matrices & Output generation:** It creates the CSR sparse interaction matrix used by models to compute similarities quickly. All resulting artifacts (CSV splits, `.npz` sparse matrices, and pickled ID mappers) are finally saved into `pipeline_output/`.

**To run the pipeline:**
```bash
python3 src/build_npy.py
```

## 2. Baseline Models

We implemented two primary baselines to establish a performance floor for the recommender space. They read from `pipeline_output/` to ensure a consistent evaluation setup.

### Baseline 1: Most Popular Items (`baselines/baseline_popularity.py`)
- **How it works:** It acts as a non-personalized baseline. It simply counts how many interactions each app had in the training datasets, ranks them in descending order, and recommends the overall top-K most popular unread apps to every user.
- **Why it matters:** In environments with very high dataset sparsity, global popularity is exceptionally difficult to beat. Any truly personalized algorithm needs to outperform this model to be considered useful.

### Baseline 2: User-Based KNN Collaborative Filtering (`baselines/baseline_knn_cf.py`)
- **How it works:** A classic memory-based collaborative filtering approach using `scikit-surprise`. It treats all interactions as binary "implicit feedback" (score 1) and calculates cosine similarity between the users. It extracts this fitted similarity matrix, and for each inference, scores unsampled apps based on the weighted preference similarity to other K-nearest users.
- **Why it matters:** It serves as the standard memory-based reference point. While it offers personalization, it often struggles against the popularity baseline if intersection levels between users (sparsity) are too severely low.

**To run the baselines:**
*(Ensure the pipeline has been run first!)*
```bash
python3 -m baselines.baseline_popularity
python3 -m baselines.baseline_knn_cf
```