## Atilla Variant 2, Hybrid Recommender

This folder contains my individual variant that is a content based recommender that uses app metadata
and the updated hybrid version that also uses item to item collaborative filtering. 
#Version 1: Basic Content-Based Model

Run from the project root after running the shared pipeline:

```bash
python3 src/build_npy.py
python3 -m variant_atilla_hybrid.atilla_basic_content
```

The metadata used includes app category, install count, average rating and rating count.

The model builds an user profile by averaging the metadata vectors of apps that the
user has installed in the training set. Then it recommends apps that are unseen by the user using
cosine similarity.

The script evaluates on the validation split and saves results to;

pipeline_output/results/atilla_basic_content_val.json

Current validation result:

Recall@10: 0.0218
Precision@10: 0.0103
HitRate@10: 0.0957

## Variant 2 version 2,  Hybrid Content + Item based CF Model

Run from the project root after running the shared pipeline:

```bash
python3 src/build_npy.py
python3 -m variant_atilla_hybrid.atilla_hybrid val
python3 -m variant_atilla_hybrid.atilla_hybrid test
```

Version 2 keeps the same content from version 1, but adds the 
item to item colloaborative filtering score. The item based CF is used here
to look at which apps tend to be installed by the same group of users. 

The final score is a blend of the two parts.
(final score = 0.2 * content score + 0.8 * item-CF score)

I chose to go with 0.2 alpha so the model stull gives some weight to metadata
while giving majority of it to item based CF (testings showed that the alpha of 0.0 yields the best results but that would defeat the purpose of my variant).
Item interaction signal was stronger than the metadata in this case.

Results are saved to:
pipeline_output/results/atilla_hybrid_val.json
pipeline_output/results/atilla_hybrid_test.json

Current test results:

Recall@10: 0.0490
Precision@10: 0.0228
HitRate@10: 0.2046


Compared to the baselines, the version 2 of my model beats the User KNN baselines,
but it does not beat the popularity baseline.

It does make sense as app installs can be very popularity driven,
hence why a simple popularity model can still show some strong results. 
