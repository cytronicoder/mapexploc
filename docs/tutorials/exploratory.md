# Exploratory Data Analysis Tutorial

This tutorial demonstrates how to perform comprehensive exploratory data analysis (EDA) on protein datasets using MAP-ExPLoc and associated visualization tools.

## Overview

Exploratory data analysis is crucial for understanding protein datasets before building machine learning models. This tutorial covers data distribution analysis, feature correlations, sequence properties, and localization patterns.

## Prerequisites

- Processed protein dataset (from preprocessing tutorial)
- Feature matrix (from feature engineering)
- Visualization libraries (matplotlib, seaborn)

## Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up visualization style
plt.style.use('default')
sns.set_palette("husl")

# Configure matplotlib
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
```

## Loading the Dataset

```python
# Load processed protein data
from mapexploc.preprocessing import extract_protein_data
from mapexploc.features import build_feature_matrix

# Option 1: Load from saved files
df = pd.read_csv('processed_proteins.csv')
features_df = pd.read_csv('protein_features.csv')

# Option 2: Generate fresh data
# protein_data = extract_protein_data("uniprot_sprot.dat")
# df = pd.DataFrame(protein_data)

print(f"Dataset shape: {df.shape}")
print(f"Features shape: {features_df.shape}")
```

## Basic Dataset Statistics

### Overview Statistics

```python
# Basic information about the dataset
print("Dataset Overview")
print("=" * 50)
print(f"Total proteins: {len(df)}")
print(f"Total features: {features_df.shape[1] - 2}")  # Excluding entry_name and localization
print(f"Unique localizations: {df['localization'].nunique()}")

# Memory usage
print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Missing values
missing_vals = df.isnull().sum()
if missing_vals.sum() > 0:
    print(f"\nMissing values:\n{missing_vals[missing_vals > 0]}")
else:
    print("\nNo missing values found")
```

### Sequence Length Analysis

```python
# Calculate sequence lengths
df['sequence_length'] = df['sequence'].str.len()

# Basic statistics
print("Sequence Length Statistics")
print("=" * 30)
print(df['sequence_length'].describe())

# Visualize sequence length distribution
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Histogram
axes[0].hist(df['sequence_length'], bins=50, alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Sequence Length')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Protein Sequence Lengths')
axes[0].axvline(df['sequence_length'].median(), color='red', linestyle='--',
                label=f'Median: {df["sequence_length"].median():.0f}')
axes[0].legend()

# Box plot
sns.boxplot(data=df, y='sequence_length', ax=axes[1])
axes[1].set_title('Sequence Length Box Plot')
axes[1].set_ylabel('Sequence Length')

plt.tight_layout()
plt.show()
```

## Subcellular Localization Analysis

### Distribution of Localizations

```python
# Localization frequency analysis
loc_counts = df['localization'].value_counts()

print("Subcellular Localization Distribution")
print("=" * 40)
print(loc_counts)
print(f"\nClass balance ratio (min/max): {loc_counts.min():.0f}/{loc_counts.max():.0f} = {loc_counts.min()/loc_counts.max():.3f}")

# Visualize localization distribution
plt.figure(figsize=(14, 8))
loc_counts.plot(kind='bar')
plt.title('Distribution of Subcellular Localizations')
plt.xlabel('Localization')
plt.ylabel('Number of Proteins')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Pie chart for proportions
plt.figure(figsize=(10, 10))
colors = plt.cm.Set3(np.linspace(0, 1, len(loc_counts)))
wedges, texts, autotexts = plt.pie(loc_counts.values, labels=loc_counts.index,
                                  autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('Proportion of Subcellular Localizations')
plt.axis('equal')
plt.show()
```

### Sequence Length by Localization

```python
# Analyze sequence length patterns by localization
plt.figure(figsize=(15, 8))
sns.boxplot(data=df, x='localization', y='sequence_length')
plt.title('Sequence Length Distribution by Subcellular Localization')
plt.xlabel('Localization')
plt.ylabel('Sequence Length')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Statistical comparison
from scipy import stats

print("Sequence Length by Localization")
print("=" * 35)
for loc in df['localization'].unique():
    subset = df[df['localization'] == loc]['sequence_length']
    print(f"{loc:20s}: {subset.mean():.1f} ± {subset.std():.1f} (n={len(subset)})")
```

## Feature Analysis

### Feature Distribution and Correlations

```python
# Load feature matrix (excluding metadata columns)
feature_cols = [col for col in features_df.columns
                if col not in ['entry_name', 'localization']]
X = features_df[feature_cols]

# Basic feature statistics
print("Feature Statistics")
print("=" * 20)
print(X.describe())

# Feature correlation matrix
plt.figure(figsize=(20, 16))
correlation_matrix = X.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', center=0, square=True, cbar_kws={"shrink": .8})
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# Identify highly correlated features
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            high_corr_pairs.append((
                correlation_matrix.columns[i],
                correlation_matrix.columns[j],
                correlation_matrix.iloc[i, j]
            ))

print(f"\nHighly correlated feature pairs (|r| > 0.8):")
for feat1, feat2, corr in high_corr_pairs:
    print(f"  {feat1} - {feat2}: {corr:.3f}")
```

### Key Physicochemical Properties

```python
# Analyze key physicochemical properties
key_features = ['molecular_weight', 'isoelectric_point', 'gravy', 'aromaticity']

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for i, feature in enumerate(key_features):
    if feature in X.columns:
        axes[i].hist(X[feature], bins=30, alpha=0.7, edgecolor='black')
        axes[i].set_xlabel(feature.replace('_', ' ').title())
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'Distribution of {feature.replace("_", " ").title()}')
        axes[i].axvline(X[feature].median(), color='red', linestyle='--',
                       label=f'Median: {X[feature].median():.2f}')
        axes[i].legend()

plt.tight_layout()
plt.show()
```

### Amino Acid Composition Analysis

```python
# Analyze amino acid composition
aa_cols = [col for col in X.columns if col.startswith('aa_')]
aa_composition = X[aa_cols]

# Average amino acid composition
avg_composition = aa_composition.mean().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
avg_composition.plot(kind='bar')
plt.title('Average Amino Acid Composition Across All Proteins')
plt.xlabel('Amino Acid')
plt.ylabel('Average Frequency')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

print("Most common amino acids:")
print(avg_composition.head(10))
```

## Localization-Specific Feature Analysis

### Feature Differences by Localization

```python
# Analyze features by localization
features_with_loc = features_df[feature_cols + ['localization']]

# Calculate mean features by localization
loc_features = features_with_loc.groupby('localization')[key_features].mean()

print("Key Features by Localization")
print("=" * 30)
print(loc_features)

# Visualize key features by localization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for i, feature in enumerate(key_features):
    if feature in features_df.columns:
        sns.boxplot(data=features_df, x='localization', y=feature, ax=axes[i])
        axes[i].set_title(f'{feature.replace("_", " ").title()} by Localization')
        axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

### Statistical Testing

```python
# Statistical tests for feature differences
from scipy.stats import kruskal
from itertools import combinations

def test_feature_differences(df, feature, group_col='localization'):
    """Test if feature values differ significantly between groups."""
    groups = [df[df[group_col] == loc][feature].values
              for loc in df[group_col].unique()]

    # Remove empty groups
    groups = [g for g in groups if len(g) > 0]

    if len(groups) < 2:
        return None

    # Kruskal-Wallis test (non-parametric ANOVA)
    stat, p_value = kruskal(*groups)
    return {'statistic': stat, 'p_value': p_value}

print("Statistical Tests for Feature Differences")
print("=" * 45)

for feature in key_features:
    if feature in features_df.columns:
        result = test_feature_differences(features_df, feature)
        if result:
            significance = "***" if result['p_value'] < 0.001 else ("**" if result['p_value'] < 0.01 else ("*" if result['p_value'] < 0.05 else ""))
            print(f"{feature:20s}: p={result['p_value']:.2e} {significance}")
```

## Dimensionality Reduction and Clustering

### Principal Component Analysis (PCA)

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize features for PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Plot explained variance
plt.figure(figsize=(12, 5))

# Cumulative explained variance
plt.subplot(1, 2, 1)
plt.plot(range(1, min(21, len(pca.explained_variance_ratio_) + 1)),
         np.cumsum(pca.explained_variance_ratio_[:20]), 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('PCA: Cumulative Explained Variance')
plt.grid(True, alpha=0.3)

# Individual component variance
plt.subplot(1, 2, 2)
plt.plot(range(1, min(21, len(pca.explained_variance_ratio_) + 1)),
         pca.explained_variance_ratio_[:20], 'ro-')
plt.xlabel('Component Number')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA: Individual Component Variance')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"First 10 components explain {np.sum(pca.explained_variance_ratio_[:10]):.1%} of variance")
print(f"First 20 components explain {np.sum(pca.explained_variance_ratio_[:20]):.1%} of variance")
```

### 2D Visualization with PCA

```python
# 2D PCA visualization colored by localization
plt.figure(figsize=(12, 10))

colors = plt.cm.tab10(np.linspace(0, 1, features_df['localization'].nunique()))
loc_color_map = dict(zip(features_df['localization'].unique(), colors))

for loc in features_df['localization'].unique():
    mask = features_df['localization'] == loc
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=[loc_color_map[loc]], label=loc, alpha=0.6, s=50)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('PCA Visualization of Protein Features by Localization')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
```

### UMAP Visualization (if available)

```python
try:
    import umap

    # UMAP for non-linear dimensionality reduction
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)

    plt.figure(figsize=(12, 10))

    for loc in features_df['localization'].unique():
        mask = features_df['localization'] == loc
        plt.scatter(X_umap[mask, 0], X_umap[mask, 1],
                    c=[loc_color_map[loc]], label=loc, alpha=0.6, s=50)

    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.title('UMAP Visualization of Protein Features by Localization')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

except ImportError:
    print("UMAP not available. Install with: pip install umap-learn")
```

## Data Quality Assessment

### Outlier Detection

```python
from sklearn.ensemble import IsolationForest

# Detect outliers using Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
outliers = iso_forest.fit_predict(X_scaled)

print(f"Detected {np.sum(outliers == -1)} outliers ({np.mean(outliers == -1):.1%} of data)")

# Visualize outliers in PCA space
plt.figure(figsize=(10, 8))
colors = ['red' if x == -1 else 'blue' for x in outliers]
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.6, s=50)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('Outlier Detection in PCA Space (Red = Outliers)')
plt.show()
```

### Data Balance Analysis

```python
# Analyze class balance for machine learning
from collections import Counter

loc_counts = Counter(features_df['localization'])
total_samples = len(features_df)

print("Class Balance Analysis")
print("=" * 25)
print(f"{'Localization':20s} {'Count':>8s} {'Percentage':>10s}")
print("-" * 45)

for loc, count in loc_counts.most_common():
    percentage = count / total_samples * 100
    print(f"{loc:20s} {count:8d} {percentage:9.1f}%")

# Calculate imbalance ratio
min_class = min(loc_counts.values())
max_class = max(loc_counts.values())
imbalance_ratio = max_class / min_class

print(f"\nImbalance ratio: {imbalance_ratio:.1f}:1")

if imbalance_ratio > 10:
    print("⚠️  Highly imbalanced dataset - consider using SMOTE or class weighting")
elif imbalance_ratio > 5:
    print("⚠️  Moderately imbalanced dataset - consider balancing techniques")
else:
    print("✓ Reasonably balanced dataset")
```

## Summary and Recommendations

```python
# Generate summary report
def generate_eda_summary(df, features_df):
    """Generate a comprehensive EDA summary."""
    summary = {
        'dataset_size': len(df),
        'n_features': len([col for col in features_df.columns
                          if col not in ['entry_name', 'localization']]),
        'n_localizations': df['localization'].nunique(),
        'median_sequence_length': df['sequence_length'].median(),
        'imbalance_ratio': max(df['localization'].value_counts()) / min(df['localization'].value_counts()),
        'missing_values': df.isnull().sum().sum(),
        'outlier_percentage': np.mean(outliers == -1) * 100
    }

    return summary

summary = generate_eda_summary(df, features_df)

print("Exploratory Data Analysis Summary")
print("=" * 35)
print(f"Dataset size: {summary['dataset_size']:,} proteins")
print(f"Number of features: {summary['n_features']}")
print(f"Number of localization classes: {summary['n_localizations']}")
print(f"Median sequence length: {summary['median_sequence_length']:.0f} amino acids")
print(f"Class imbalance ratio: {summary['imbalance_ratio']:.1f}:1")
print(f"Missing values: {summary['missing_values']}")
print(f"Potential outliers: {summary['outlier_percentage']:.1f}%")

print("\nRecommendations for Model Training:")
if summary['imbalance_ratio'] > 5:
    print("- Consider using SMOTE or class weighting for imbalanced classes")
if summary['outlier_percentage'] > 10:
    print("- Review and potentially remove outliers")
if summary['n_features'] > 50:
    print("- Consider feature selection or dimensionality reduction")
print("- Use stratified cross-validation for evaluation")
print("- Consider ensemble methods for better performance")
```

## Next Steps

After completing exploratory data analysis, proceed to:

1. **[Feature Engineering](features.md)**: Optimize feature selection and engineering
2. **[k-NN Classification](knn.md)**: Train k-nearest neighbors models
3. **[Random Forest](rf.md)**: Train ensemble models with the insights gained
4. **[Model Evaluation](evaluation.md)**: Implement comprehensive evaluation strategies

The insights from this EDA will guide feature selection, model choice, and evaluation strategies in subsequent tutorials.
