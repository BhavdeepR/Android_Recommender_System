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

### 🔹 LightGCN (Graph-Based Model)

A Graph Neural Network designed for recommendation tasks on bipartite graphs.

* Captures **high-order connectivity** between users and apps
* Handles **extreme sparsity** better than traditional methods
* Learns embeddings through **graph propagation**

---

### 📈 Results Comparison (@10)

| Metric    | LightGCN | Popularity | User-KNN |
| --------- | -------- | ---------- | -------- |
| Recall    | 0.0627   | 0.0553     | 0.0440   |
| Precision | 0.0365   | 0.0255     | 0.0204   |
| Hit Rate  | 0.2517   | 0.2205     | 0.1825   |

✅ LightGCN outperforms both baselines across all metrics.

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

## 📁 Project Structure (Optional)

```bash
.
├── data/
├── src/
├── baselines/
├── pipeline_output/
├── requirements.txt
└── graph_run_pipeline.sh
```

---

## 🙌 Acknowledgements

* Dataset: Myket Android App Store
* Libraries: NumPy, SciPy, scikit-surprise, PyTorch (for LightGCN)

---
