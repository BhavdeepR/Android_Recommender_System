"""
Atilla Variant Version 2, Attempt at a Hybrid Content + Item-CF Recommender

This version builds on my basic content model (version 1).
It still uses app metadata, but it also adds an item-item CF score based on
apps that are installed by similar users.

Run from the project root
(python3 -m variant_atilla_hybrid.atilla_hybrid)

Make sure you run the data pipeline first
(python3 src/build_npy.py)

Goals of Variant 2, Version 2:

1) Build the content score the same way as version 1.
2) Build an item-CF score from the train interaction matrix.
3) Normalize both scores and blend them together.

I used alpha = 0.2 for the normal runs. Which uses 20% content score and 80% item-tem CF score
Setting the content score to 0% yileded the best results (the metadata is too simple).  
I kept the alpha tuning function in the the project but it is deactivated by default as 
the grid search takes some time to run.
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, diags

#Importing the evaluator and shared utilities from version 1
sys.path.insert(0,os.path.join(os.path.dirname(__file__),".."))
from baselines.evaluate import evaluate_recommendations
from variant_atilla_hybrid.atilla_basic_content import (
    normalize_rows,
    build_ground_truth,
    DATA_DIR,
    K,
    EPS,
)

ALPHA_DEFAULT=0.2

#Optional alpha tuning grid
#I do not run this by default because it takes too much time to run it each time, so I chose 0.2 alpha after some manual experimentation.
ALPHA_GRID=[0.2,0.3,0.4,0.5,0.6,0.7,0.8]


#change do_eda=False to do_eda=True in run_hybrid_model() to see the EDA
def run_eda(app_features,R):
    n_apps=app_features.shape[0]
    has_meta=~np.all(app_features == 0,axis=1)
    app_installs=np.array(R.sum(axis=0)).flatten()
    user_installs=np.array(R.sum(axis=1)).flatten()
    top1_pct=int(n_apps*0.01)+1
    top1_share=app_installs[np.argsort(-app_installs)[:top1_pct]].sum()/app_installs.sum()


    print(f"\nEDA:")
    print(f"Apps with metadata: {has_meta.sum():,} / {n_apps:,} ({100*(~has_meta).mean():.1f}% missing)")
    print(f"Installs per app : mean={app_installs.mean():.1f}, median={np.median(app_installs):.0f}, max={app_installs.max():.0f}")
    print(f"Installs per user : mean={user_installs.mean():.1f}, median={np.median(user_installs):.0f}, max={user_installs.max():.0f}")
    print(f"Top 1% of apps account for {100*top1_share:.1f}% of all installs")


def build_item_sim_matrix(R):
    #Compute item-item cosine similarities from the train matrix
    col_norms=np.sqrt(np.array(R.power(2).sum(axis=0))).flatten()
    col_norms=np.maximum(col_norms, EPS)
    R_cn=R @ diags(1.0 / col_norms)
    sim=(R_cn.T @ R_cn).toarray().astype(np.float32)
    np.fill_diagonal(sim, 0.0)
    return sim


def minmax(arr):
    #Scale array to [0, 1] as both of the score types are are on different rangers and we would like to blend them together.
    lo, hi = arr.min(), arr.max()
    return arr if (hi - lo) < EPS else (arr - lo) / (hi - lo)


def recommend_user(user_id, user_train_history, app_features_norm, item_sim, alpha=ALPHA_DEFAULT, k=K):
    #Get seen apps
    seen_items=user_train_history.get(user_id, [])

    if not seen_items:
        return []

    #Build content scores (from version 1)
    user_profile = app_features_norm[seen_items].mean(axis=0)
    profile_norm = np.linalg.norm(user_profile)

    if profile_norm < EPS:
        content_s=None
    else:
        user_profile = user_profile / profile_norm
        content_s=minmax(app_features_norm @ user_profile)

    #Build CF scores
    cf_s = minmax(item_sim[seen_items, :].mean(axis=0))

    #Blend both scores
    if content_s is None:
        scores=cf_s
    else:
        scores=alpha*content_s+(1.0 - alpha) * cf_s

    #Remove seen apps
    scores[seen_items]=-np.inf

    #Pick top k
    top_items = np.argsort(-scores)[:k]
    return top_items.tolist()


def generate_recommendations(ground_truth, user_train_history, app_features_norm, item_sim, alpha=ALPHA_DEFAULT, verbose=True):
    recommendations={}
    skipped=0

    for user_id in ground_truth:
        recs=recommend_user(user_id, user_train_history, app_features_norm, item_sim, alpha=alpha)

        if not recs:
            skipped+=1

        recommendations[user_id] = recs

    if verbose:
        print(f"Empty recommendations: {skipped:,} / {len(ground_truth):,}")
    return recommendations


def tune_alpha(ground_truth,user_train_history,app_features_norm, item_sim):
    #Optional grid search on validation, takes some time.
    print("Tuning the alpha")
    best_alpha,best_recall=ALPHA_DEFAULT, -1.0

    for alpha in ALPHA_GRID:
        recs=generate_recommendations(ground_truth,user_train_history,app_features_norm,item_sim,alpha=alpha,verbose=False)
        recall=evaluate_recommendations(recs,ground_truth,k_values=[10])["Recall@10"]
        if recall > best_recall:
            best_recall,best_alpha=recall,alpha

    print(f"Best alpha: {best_alpha} (Recall@10 = {best_recall:.4f})")
    return best_alpha


def run_hybrid_model(split_name="val", tune=False, do_eda=True):
    if split_name not in {"val", "test"}:
        raise ValueError("split_name must be 'val' or 'test'")

    #Loading the data
    eval_df=pd.read_csv(os.path.join(DATA_DIR, f"{split_name}.csv"))
    app_features=np.load(os.path.join(DATA_DIR, "app_feature_matrix.npy"))
    R=load_npz(os.path.join(DATA_DIR,"train_interaction_matrix.npz"))
    user_train_history=joblib.load(os.path.join(DATA_DIR,"user_train_history.pkl"))

    if do_eda:
        run_eda(app_features,R)

    #Precompute shared structures
    app_features_norm = normalize_rows(app_features.astype(np.float32))

    print("Building the item to item similarity matrix")
    t=time.time()
    item_sim=build_item_sim_matrix(R)
    print(f"Done ({time.time()-t:.1f}s)")

    #Build answers
    ground_truth=build_ground_truth(eval_df)
    print(f"Users in {split_name}: {len(ground_truth):,}")

    #Use fixed alpha unless tuning is turned on
    alpha = ALPHA_DEFAULT
    if tune and split_name == "val":
        alpha = tune_alpha(ground_truth, user_train_history, app_features_norm, item_sim)

    #Recommend
    print(f"\nGenerating recommendations (alpha={alpha})...")
    start=time.time()
    recommendations=generate_recommendations(ground_truth, user_train_history, app_features_norm, item_sim, alpha=alpha)
    print(f"Time: {time.time() - start:.2f} seconds")

    #Evaluate
    results = evaluate_recommendations(
        recommendations=recommendations,
        ground_truth=ground_truth,
        k_values=[10],
    )

    print(f"\nAtilla version 2, Hybrid Content + Item-CF ({split_name}, alpha={alpha})")
    print(f"Recall@10: {results['Recall@10']:.4f}")
    print(f"Precision@10: {results['Precision@10']:.4f}")
    print(f"HitRate@10: {results['HitRate@10']:.4f}")

    #Save results
    results_dir=os.path.join(DATA_DIR,"results")
    os.makedirs(results_dir, exist_ok=True)
    output_path=os.path.join(results_dir,f"atilla_hybrid_{split_name}.json")
    save_results={k: results[k] for k in ["Recall@10","Precision@10","HitRate@10"]}
    with open(output_path, "w") as f:
        json.dump({"alpha": alpha, "split": split_name, "results": save_results},f,indent=2)
    print(f"Saved to: {output_path}")
    return results


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "val"
    run_hybrid_model(split_name=split,tune=False,do_eda=False)
