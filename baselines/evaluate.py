"""
baselines/evaluate.py
─────────────────────
Shared evaluation functions for all baseline and main models.

Metrics computed at K=5 and K=10:
  - Recall@K       : fraction of test items found in top-K
  - NDCG@K         : rank-weighted recall (position-sensitive)
  - Precision@K    : fraction of top-K items that are relevant
  - Hit Rate@K     : binary (did at least one test item appear in top-K)

Usage:
    from evaluate import evaluate_recommendations
    results = evaluate_recommendations(recommendations, ground_truth)
    print(results)
"""

import numpy as np
from collections import defaultdict


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of relevant items found in top-K recommendations."""
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    Normalised Discounted Cumulative Gain at K.
    Ideal DCG assumes all relevant items ranked first.
    """
    if not relevant:
        return 0.0
    dcg = 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            dcg += 1.0 / np.log2(rank + 1)
    # Ideal DCG: top min(|relevant|, K) positions all hit
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of top-K recommendations that are relevant."""
    if k == 0:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def hit_rate_at_k(recommended: list, relevant: set, k: int) -> float:
    """1 if at least one relevant item is in top-K, else 0."""
    return float(any(item in relevant for item in recommended[:k]))


def evaluate_recommendations(
    recommendations: dict,   # {user_id: [item_id, …]} (already ordered, top-K)
    ground_truth: dict,      # {user_id: [item_id, …]} (test set items)
    k_values: list = (5, 10)
) -> dict:
    """
    Evaluate how good the recommendations are compared to what users actually interacted with.

    Parameters
    recommendations : dict
        Mapping of user_id -> list of recommended item ids, ordered from best to worst.
    ground_truth : dict
        Mapping of user_id -> list of actual items from the test set.
    k_values : iterable of ints
        The cutoff points to evaluate (e.g., top-5, top-10).
    Returns
    dict
    Dictionary of metric scores, averaged across all users who have ground truth data.

    """
    metrics = defaultdict(list)

    for user_id, relevant_items in ground_truth.items():
        relevant = set(relevant_items)
        recs = recommendations.get(user_id, [])

        for k in k_values:
            metrics[f"Recall@{k}"].append(recall_at_k(recs, relevant, k))
            metrics[f"NDCG@{k}"].append(ndcg_at_k(recs, relevant, k))
            metrics[f"Precision@{k}"].append(precision_at_k(recs, relevant, k))
            metrics[f"HitRate@{k}"].append(hit_rate_at_k(recs, relevant, k))

    return {metric: float(np.mean(vals)) for metric, vals in metrics.items()}


def print_results_table(model_name: str, results: dict) -> None:
    """Pretty-print evaluation results."""
    print(f"\n{'='*58}")
    print(f"  {model_name}")
    print(f"{'='*58}")
    print(f"  {'Metric':<18} {'@5':>10} {'@10':>10}")
    print(f"  {'-'*38}")
    for base in ["Recall", "NDCG", "Precision", "HitRate"]:
        v5  = results.get(f"{base}@5",  0.0)
        v10 = results.get(f"{base}@10", 0.0)
        print(f"  {base:<18} {v5:>10.4f} {v10:>10.4f}")
    print(f"{'='*58}\n")
