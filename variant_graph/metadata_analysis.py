"""
Metadata exploration script for app recommendation features.

This script analyzes app metadata to understand how useful different metadata
fields may be for a recommendation model. It examines category balance,
numeric feature variance, and publisher-level granularity to assess whether
the metadata is likely to provide meaningful signal.
"""

import pandas as pd             #used to load and manipulate tabular data
import matplotlib.pyplot as plt #used for plotting and data visualization
import seaborn as sns           #used for statistical data visualization
import numpy as np              #used for numerical operations and array manipulation

# Load the app metadata file containing descriptive information about each app.
app_info = pd.read_csv("data/app_info_sample.csv")

print("\nMETADATA OVERVIEW")
print(f"Total Apps in Metadata: {len(app_info):,}")

# 1. Category Distribution
# Examine how apps are distributed across categories.
# If only a few categories dominate the dataset, category may provide limited
# discriminative value as a metadata feature.
cat_counts = app_info['category_en'].value_counts()
cat_perc = app_info['category_en'].value_counts(normalize=True) * 100

print("\nTop 10 Categories by Volume:")
eda_df = pd.DataFrame({'Count': cat_counts, 'Percentage': cat_perc}).head(10)
print(eda_df)

# 2. Information Gain Check
# Measure whether category is likely to be informative.
# If one category occupies too much of the dataset, the feature may contribute
# less useful signal because it does not separate items well.
top_cat_share = cat_perc.iloc[0]
if top_cat_share > 30:
    print(f"\nWARNING: The '{cat_counts.index[0]}' category accounts for {top_cat_share:.1f}% of your data.")
    print("This may reduce the usefulness of category as a metadata signal and encourage over-generalization.")

# 3. Category Distribution Visualization
# Plot the top categories to visually inspect whether the metadata is balanced
# or heavily skewed toward a small number of groups.
plt.figure(figsize=(12, 6))
sns.barplot(x=cat_perc.head(15).values, y=cat_perc.head(15).index, palette='viridis')
plt.title("Top 15 App Categories (%)")
plt.xlabel("Percentage of Total Apps")
plt.show()

# 4. Numeric Metadata Feature Variance
# Inspect summary statistics for numeric metadata fields.
# Features with very low variance may offer limited value for distinguishing apps.
print("\nNUMERIC FEATURE VARIANCE")
stats = app_info[['rating', 'installs', 'rating_count']].describe()
print(stats.loc[['mean', 'std', '50%', 'max']])

# 5. High-Cardinality Metadata Feature Check (Publisher Proxy)
# Derive a rough publisher-like feature from the app identifier/name structure.
# This helps assess whether a more fine-grained metadata field may provide
# stronger signal than broad categories.
def extract_publisher(pkg):
    parts = str(pkg).split('.')
    return parts[1] if len(parts) > 2 else parts[0]

app_info['publisher_temp'] = app_info['app_name'].apply(extract_publisher)

# Count the number of unique publisher values.
unique_pubs = app_info['publisher_temp'].nunique()

print(f"\nUnique Publishers: {unique_pubs}")

# Compute the average number of apps per publisher to estimate sparsity.
print(f"Apps per Publisher (Avg): {len(app_info)/unique_pubs:.2f}")

# If publisher-like identifiers are much more granular than categories,
# they may offer stronger personalization signal in a metadata-aware model.
if unique_pubs > app_info['category_en'].nunique() * 10:
    print("The 'Publisher' feature is much more detailed than 'Category'.")
    print("The model may benefit from relying more on Publisher for finer recommendations.")