# 📱 Android App Recommender System

A recommendation system designed to suggest Android apps based on users’ **implicit interactions** (e.g., installs and views). This project uses real-world data from the **Myket Android App Store**.

---

## 🚀 Setup & Installation

Install dependencies using:

```bash
pip install -r requirements.txt
```

> ⚠️ **Note:** `scikit-surprise` currently requires `numpy < 2`.

---

## 🧱 1. Data Pipeline (`src/build_npy.py`)

This script handles **data ingestion, preprocessing, and artifact generation**.

### Key Responsibilities

* **ID Encoding**

  * Converts raw user IDs and app identifiers into contiguous integer indices
  * Example: Users → `0 ... N`, Apps → `0 ... M`

* **Feature Engineering**

  * One-hot encodes app categories
  * Applies log-scaling to numerical features (e.g., installs, ratings)
  * Outputs a normalized feature matrix: `app_info_sample.npy`

* **Temporal Data Splitting**

  * Per-user chronological split:

    * Last **5 interactions** → **Test set**
    * Previous **5 interactions** → **Validation set**
    * Remaining → **Training set**

* **Sparse Matrix Construction**

  * Builds CSR interaction matrices for efficient computation

* **Output Artifacts**

  * Saved in `pipeline_output/`:

    * CSV splits
    * `.npz` sparse matrices
    * Pickled ID mappings

### ▶️ Run the Pipeline

```bash
python3 src/build_npy.py
```

---

## 📊 2. Baseline Models

All baselines use data from `pipeline_output/` for fair comparison.

### 🔹 Baseline 1: Most Popular Items

**File:** `baselines/baseline_popularity.py`

* **Approach:**

  * Ranks apps by total interaction count
  * Recommends top-K unseen apps globally

* **Why it matters:**

  * Strong non-personalized benchmark
  * Hard to beat in sparse datasets

---

### 🔹 Baseline 2: User-Based KNN (Collaborative Filtering)

**File:** `baselines/baseline_knn_cf.py`

* **Approach:**

  * Computes cosine similarity between users (`scikit-surprise`)
  * Treats interactions as binary implicit feedback
  * Scores apps based on similar users’ preferences

* **Why it matters:**

  * Classic personalization baseline
  * Can struggle with extreme sparsity

---

### ▶️ Run Baselines

```bash
python3 -m baselines.baseline_popularity
python3 -m baselines.baseline_knn_cf
```

> ✅ Ensure the data pipeline has been executed first.

---

## 📏 3. Evaluation Framework

A shared evaluation setup ensures consistent model comparison.

### 🔄 Split Strategy

Time-based per-user split:

* **Test set:** last 5 interactions
* **Validation set:** previous 5
* **Training set:** remaining interactions

This setup evaluates how well models predict **future behavior**.

---

### 📐 Metrics

* **Recall@10**
* **Precision@10**
* **HitRate@10**

---

### ⚙️ Dataset Assumptions

* **Implicit Feedback Only**

  * Interactions = positive signals
  * No explicit negative feedback

* **Evaluation Rules**

  * Previously seen apps are filtered out
  * Only **new recommendations** are evaluated

---

### 🧩 Shared Evaluation Code

```bash
baselines/evaluate.py
```
---

## 🤖 4. Advanced Models

We explored three distinct advanced architectures to push beyond the baselines:

### 🔹 Variant A: BPR Matrix Factorization with Time Decay
**File:** `BPR_MF_with_Time_Decay.ipynb`
* Implements Bayesian Personalized Ranking (BPR) to optimize for ranking items correctly.
* Introduces a time decay factor to give more weight to recent user interactions, acknowledging that user app preferences change over time.

### 🔹 Variant B: Atilla Hybrid
**Directory:** `variant_atilla_hybrid/`
* Combines content-based filtering with collaborative filtering signals.
* Attempts to bridge the gap between user behavior and app semantic properties to improve predictions when interactions are sparse.

### 🔹 Variant C: Graph Neural Networks (LightGCN)
**Directory:** `variant_graph/`
* **Pure LightGCN:** Captures high-order connectivity between users and apps by learning embeddings through graph propagation on the user-app bipartite graph.
* **LightGCN with Metadata:** A hybrid approach that adds a Multi-Layer Perceptron (MLP) branch. It injects app metadata directly into the graph learning process, grounding the collaborative signals in actual app semantics and helping alleviate cold-start problems.

---

### 📈 Results Comparison (@10)

| Model                     | Recall@10 | Precision@10 | Hit Rate@10 |
|---------------------------|-----------|--------------|-------------|
| Popularity Baseline       | 0.0553    | 0.0255       | 0.2205      |
| User-KNN Baseline         | 0.0440    | 0.0204       | 0.1825      |
| Variant B - HYBRID        | 0.0490    | 0.0228       | 0.2046      |
| LightGCN                  | 0.0586    | 0.0270       | 0.2327      |
| LightGCN With metadata    | 0.0618    | 0.0285       | 0.2456      |

✅ The LightGCN with Metadata variant achieved the highest performance across all tracked metrics, significantly outperforming the baselines and other variants.

---

### ▶️ Run Full Pipeline (End-to-End)

```bash
./graph_run_pipeline.sh
```

---

## 📌 Summary

* Built a full recommendation pipeline from raw data to evaluation
* Established strong baselines (Popularity, User-KNN)
* Improved performance using **Graph Neural Networks (LightGCN)**
* Demonstrated gains in all key ranking metrics

---

## 🙌 Acknowledgements

* Dataset: Myket Android App Store
* Libraries: NumPy, SciPy, scikit-surprise, PyTorch (for LightGCN)

---
