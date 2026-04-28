"""
Atilla Variant Version 1, Basic Content Based Recommender

This file loads the shared pipeline outputs and creates a basic user profile
by averaging the metadata features of the apps that each user has installed during training,
then it recommends apps not seen by the user by comparing them to the user profile using the
cosine similarity.
(uses the same shared evaluation code as the rest of the project)

Run from the project root
(python3 -m variant_atilla_hybrid.atilla_basic_content)

Make sure you run the data pipeline first
(python3 src/build_npy.py)
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd

#Evaluator
sys.path.insert(0,os.path.join(os.path.dirname(__file__),".."))
from baselines.evaluate import evaluate_recommendations


DATA_DIR="pipeline_output"
K=10
EPS=1e-12


def normalize_rows(matrix):
    #Normalize app vectors
    norms=np.linalg.norm(matrix,axis=1,keepdims=True)
    return matrix/np.maximum(norms, EPS)


def build_ground_truth(df):
    #Map each user to the correct apps
    return df.groupby("user_id")["app_id"].apply(list).to_dict()


def recommend_user(user_id, user_train_history,app_features_norm,k=K):
    #Get apps that have been seen
    seen_items=user_train_history.get(user_id,[])

    if not seen_items:
        return []

    #Build user profile
    user_profile=app_features_norm[seen_items].mean(axis=0)

    #Normalize profile
    profile_norm=np.linalg.norm(user_profile)
    if profile_norm<EPS:
        return []

    user_profile=user_profile/profile_norm

    #Score all apps
    scores=app_features_norm @ user_profile

    #Remove the seen apps
    scores[seen_items]=-np.inf

    #Picking top k
    top_items=np.argsort(-scores)[:k]
    return top_items.tolist()


def generate_recommendations(ground_truth,user_train_history,app_features):
    #Normalize features once
    app_features_norm=normalize_rows(app_features.astype(np.float32))

    recommendations={}
    skipped=0

    for user_id in ground_truth:
        recs=recommend_user(user_id,user_train_history,app_features_norm)

        if not recs:
            skipped+=1

        recommendations[user_id]=recs

    print(f"Empty recommendations: {skipped:,} / {len(ground_truth):,}")
    return recommendations


def run_basic_content_model(split_name="val"):
    if split_name not in {"val", "test"}:
        raise ValueError("split_name must be 'val' or 'test'")

    #Load data
    eval_df=pd.read_csv(os.path.join(DATA_DIR,f"{split_name}.csv"))
    app_features=np.load(os.path.join(DATA_DIR,"app_feature_matrix.npy"))
    user_train_history=joblib.load(os.path.join(DATA_DIR,"user_train_history.pkl"))

    #Build answers
    ground_truth=build_ground_truth(eval_df)
    print(f"Users in {split_name}: {len(ground_truth):,}")

    #Recommend
    start=time.time()
    recommendations=generate_recommendations(
        ground_truth,
        user_train_history,
        app_features,
    )
    print(f"Time: {time.time() - start:.2f} seconds")

    #Evaluate
    results=evaluate_recommendations(
        recommendations=recommendations,
        ground_truth=ground_truth,
        k_values=[10],
    )

    print(f"\nAtilla version 1, Basic Content ({split_name})")
    print(f"Recall@10: {results['Recall@10']:.4f}")
    print(f"Precision@10: {results['Precision@10']:.4f}")
    print(f"HitRate@10: {results['HitRate@10']:.4f}")

    #Save results
    results_dir=os.path.join(DATA_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    output_path=os.path.join(results_dir, f"atilla_basic_content_{split_name}.json")
    save_results={k: results[k] for k in ["Recall@10","Precision@10","HitRate@10"]}
    with open(output_path,"w") as f:
        json.dump(save_results,f,indent=2)
    print(f"Saved to: {output_path}")
    return results


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "val"
    run_basic_content_model(split_name=split)
