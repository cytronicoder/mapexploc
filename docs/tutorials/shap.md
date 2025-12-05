# SHAP Analysis Tutorial

This tutorial demonstrates how to use SHAP (SHapley Additive exPlanations) to interpret and explain machine learning models for protein subcellular localization prediction, providing deep insights into feature contributions and model behavior.

## Overview

SHAP provides a unified framework for interpreting machine learning models by computing the contribution of each feature to individual predictions. This tutorial covers global and local explanations, feature interactions, and model debugging techniques.

## Prerequisites

- Trained machine learning model (Random Forest, XGBoost, etc.)
- Understanding of model interpretability concepts
- Knowledge of SHAP theory and Shapley values

## Setup and Installation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
import joblib

# Configure SHAP and plotting
shap.initjs()
plt.style.use('default')
sns.set_palette("husl")

print(f"SHAP version: {shap.__version__}")
```

## Data Preparation

### Loading Model and Data

```python
# Load the trained model (assuming Random Forest from previous tutorial)
model_path = "models/random_forest_best.pkl"
model = joblib.load(model_path)

# Load the dataset
features_df = pd.read_csv('protein_features.csv')

# Prepare features and targets
X = features_df.drop(['entry_name', 'localization'], axis=1)
y = features_df['localization']

# Train-test split (same as used for training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Dataset shape: {X.shape}")
print(f"Feature names: {list(X.columns)}")
print(f"Classes: {sorted(y.unique())}")
```

### Understanding the Model Structure

```python
# Analyze the model structure for SHAP compatibility
print("Model Structure Analysis")
print("=" * 25)

if hasattr(model, 'named_steps'):
    print("Pipeline detected:")
    for step_name, step in model.named_steps.items():
        print(f"  {step_name}: {type(step).__name__}")

    # Extract the estimator for SHAP analysis
    if 'rf' in model.named_steps:
        estimator = model.named_steps['rf']
        print(f"Estimator type: {type(estimator).__name__}")
    else:
        estimator = model
else:
    estimator = model
    print(f"Direct estimator: {type(estimator).__name__}")
```

## SHAP Explainer Setup

### Tree Explainer for Random Forest

```python
# Create SHAP explainer for tree-based model
print("Setting up SHAP TreeExplainer...")

# For pipeline, we need to transform the data first
if hasattr(model, 'named_steps'):
    # Transform training data through preprocessing steps
    X_train_transformed = X_train.copy()
    X_test_transformed = X_test.copy()

    # Apply preprocessing if it exists
    for step_name, step in model.named_steps.items():
        if step_name != 'rf':  # Skip the final estimator
            if hasattr(step, 'transform'):
                X_train_transformed = step.transform(X_train_transformed)
                X_test_transformed = step.transform(X_test_transformed)

    # Convert back to DataFrame if necessary
    if isinstance(X_train_transformed, np.ndarray):
        X_train_transformed = pd.DataFrame(X_train_transformed, columns=X.columns)
        X_test_transformed = pd.DataFrame(X_test_transformed, columns=X.columns)

    # Create explainer with the actual estimator
    explainer = shap.TreeExplainer(estimator, X_train_transformed)

else:
    # Direct model - no preprocessing
    explainer = shap.TreeExplainer(model, X_train)
    X_train_transformed = X_train
    X_test_transformed = X_test

print("SHAP TreeExplainer created successfully")
print(f"Explainer expected value: {explainer.expected_value}")
```

### Computing SHAP Values

```python
# Compute SHAP values for test set (sample for efficiency)
print("Computing SHAP values...")

# Use a sample for initial analysis
sample_size = min(100, len(X_test_transformed))
X_sample = X_test_transformed.iloc[:sample_size]
y_sample = y_test.iloc[:sample_size]

# Compute SHAP values
shap_values = explainer.shap_values(X_sample)

print(f"SHAP values computed for {sample_size} samples")
print(f"SHAP values shape: {len(shap_values)} classes x {shap_values[0].shape}")

# Get class names for multi-class
classes = model.classes_
print(f"Classes: {classes}")
```

## Global Feature Importance

### Mean Absolute SHAP Values

```python
# Calculate mean absolute SHAP values across all classes
mean_shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
global_importance = np.mean(mean_shap_values, axis=0)

# Create importance DataFrame
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': global_importance
}).sort_values('importance', ascending=False)

# Plot global feature importance
plt.figure(figsize=(10, 8))
top_features = importance_df.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Mean |SHAP Value|')
plt.title('Global Feature Importance (SHAP)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("Top 10 Most Important Features (Global SHAP):")
print(importance_df.head(10))
```

### Summary Plot

```python
# SHAP summary plot - shows feature importance and impact direction
plt.figure(figsize=(10, 8))

# For multi-class, we'll show the summary for the first class as example
class_idx = 0
class_name = classes[class_idx]

shap.summary_plot(shap_values[class_idx], X_sample,
                  feature_names=X.columns,
                  title=f'SHAP Summary Plot - {class_name}',
                  show=False)
plt.tight_layout()
plt.show()

# Beeswarm plot (alternative visualization)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values[class_idx], X_sample,
                  feature_names=X.columns,
                  plot_type="dot",
                  title=f'SHAP Beeswarm Plot - {class_name}',
                  show=False)
plt.tight_layout()
plt.show()
```

## Class-Specific Analysis

### Feature Importance by Class

```python
# Analyze feature importance for each class
class_importance = {}

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.ravel()

for i, class_name in enumerate(classes[:6]):  # Show first 6 classes
    # Calculate mean absolute SHAP values for this class
    class_shap = np.abs(shap_values[i]).mean(axis=0)

    # Create DataFrame
    class_df = pd.DataFrame({
        'feature': X.columns,
        'importance': class_shap
    }).sort_values('importance', ascending=False)

    class_importance[class_name] = class_df

    # Plot
    if i < len(axes):
        ax = axes[i]
        top_features = class_df.head(10)
        ax.barh(range(len(top_features)), top_features['importance'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Mean |SHAP Value|')
        ax.set_title(f'{class_name}')
        ax.invert_yaxis()

# Hide unused subplots
for j in range(len(classes), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()

# Print top features for each class
for class_name, df in class_importance.items():
    print(f"\nTop 5 features for {class_name}:")
    print(df.head(5)[['feature', 'importance']].round(4))
```

### Feature Contribution Heatmap

```python
# Create heatmap of feature contributions across classes
# Use top 20 features from global importance
top_20_features = importance_df.head(20)['feature'].tolist()

# Create contribution matrix
contribution_matrix = np.zeros((len(classes), len(top_20_features)))

for i, class_name in enumerate(classes):
    class_df = class_importance.get(class_name, pd.DataFrame())
    for j, feature in enumerate(top_20_features):
        feature_row = class_df[class_df['feature'] == feature]
        if len(feature_row) > 0:
            contribution_matrix[i, j] = feature_row['importance'].iloc[0]

# Plot heatmap
plt.figure(figsize=(15, 8))
sns.heatmap(contribution_matrix,
            xticklabels=top_20_features,
            yticklabels=classes,
            annot=True, fmt='.3f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Mean |SHAP Value|'})
plt.title('Feature Importance Heatmap Across Classes')
plt.xlabel('Features')
plt.ylabel('Localization Classes')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

## Local Explanations

### Individual Prediction Analysis

```python
# Analyze individual predictions
def analyze_prediction(sample_idx, show_all_classes=False):
    """Analyze a single prediction with SHAP explanations."""

    sample_data = X_sample.iloc[sample_idx:sample_idx+1]
    true_label = y_sample.iloc[sample_idx]

    # Get model prediction
    pred_proba = model.predict_proba(sample_data)[0]
    pred_label = model.predict(sample_data)[0]
    confidence = pred_proba.max()

    print(f"Sample Analysis - Index {sample_idx}")
    print("=" * 35)
    print(f"True label: {true_label}")
    print(f"Predicted: {pred_label} (confidence: {confidence:.3f})")
    print(f"Correct: {'✓' if pred_label == true_label else '✗'}")

    # Show top prediction probabilities
    prob_df = pd.DataFrame({
        'class': classes,
        'probability': pred_proba
    }).sort_values('probability', ascending=False)

    print("\nTop 3 Class Probabilities:")
    print(prob_df.head(3).round(3))

    # SHAP explanations for predicted class
    pred_class_idx = np.where(classes == pred_label)[0][0]
    sample_shap = shap_values[pred_class_idx][sample_idx]

    # Feature contributions
    contrib_df = pd.DataFrame({
        'feature': X.columns,
        'value': sample_data.iloc[0].values,
        'shap_value': sample_shap
    }).sort_values('shap_value', key=abs, ascending=False)

    print(f"\nTop Feature Contributions for {pred_label}:")
    print(contrib_df.head(10)[['feature', 'value', 'shap_value']].round(3))

    # Visualization
    plt.figure(figsize=(15, 6))

    # Bar plot of SHAP values
    plt.subplot(1, 2, 1)
    top_contrib = contrib_df.head(15)
    colors = ['red' if x < 0 else 'blue' for x in top_contrib['shap_value']]
    plt.barh(range(len(top_contrib)), top_contrib['shap_value'], color=colors)
    plt.yticks(range(len(top_contrib)), top_contrib['feature'])
    plt.xlabel('SHAP Value')
    plt.title(f'Feature Contributions\n{pred_label} (Expected: {explainer.expected_value[pred_class_idx]:.3f})')
    plt.gca().invert_yaxis()

    # Waterfall plot using SHAP
    plt.subplot(1, 2, 2)
    shap.waterfall_plot(
        shap.Explanation(values=sample_shap,
                        base_values=explainer.expected_value[pred_class_idx],
                        data=sample_data.iloc[0].values,
                        feature_names=X.columns),
        show=False
    )

    plt.tight_layout()
    plt.show()

    return contrib_df

# Analyze a few interesting cases
print("Analyzing individual predictions...")

# Correct high-confidence prediction
correct_high_conf = np.where((model.predict(X_sample) == y_sample) &
                            (model.predict_proba(X_sample).max(axis=1) > 0.9))[0]
if len(correct_high_conf) > 0:
    print("\n" + "="*50)
    print("HIGH CONFIDENCE CORRECT PREDICTION")
    print("="*50)
    analyze_prediction(correct_high_conf[0])

# Incorrect prediction
incorrect_preds = np.where(model.predict(X_sample) != y_sample)[0]
if len(incorrect_preds) > 0:
    print("\n" + "="*50)
    print("INCORRECT PREDICTION")
    print("="*50)
    analyze_prediction(incorrect_preds[0])
```

### Decision Path Analysis

```python
# Analyze decision paths for samples
def analyze_decision_boundary(feature1, feature2):
    """Analyze how two features interact in decision making."""

    # Select two important features
    if feature1 not in X.columns or feature2 not in X.columns:
        print(f"Features {feature1} or {feature2} not found")
        return

    # Get feature indices
    f1_idx = list(X.columns).index(feature1)
    f2_idx = list(X.columns).index(feature2)

    # Create scatter plot colored by SHAP values
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot for first class
    class_idx = 0
    class_name = classes[class_idx]

    scatter = axes[0].scatter(X_sample.iloc[:, f1_idx],
                             X_sample.iloc[:, f2_idx],
                             c=shap_values[class_idx][:, f1_idx],
                             cmap='RdBu_r', alpha=0.7)
    axes[0].set_xlabel(feature1)
    axes[0].set_ylabel(feature2)
    axes[0].set_title(f'SHAP Values for {feature1}\nClass: {class_name}')
    plt.colorbar(scatter, ax=axes[0])

    # Plot feature interaction
    scatter2 = axes[1].scatter(X_sample.iloc[:, f1_idx],
                              X_sample.iloc[:, f2_idx],
                              c=shap_values[class_idx][:, f2_idx],
                              cmap='RdBu_r', alpha=0.7)
    axes[1].set_xlabel(feature1)
    axes[1].set_ylabel(feature2)
    axes[1].set_title(f'SHAP Values for {feature2}\nClass: {class_name}')
    plt.colorbar(scatter2, ax=axes[1])

    plt.tight_layout()
    plt.show()

# Analyze top two features
top_2_features = importance_df.head(2)['feature'].tolist()
if len(top_2_features) >= 2:
    analyze_decision_boundary(top_2_features[0], top_2_features[1])
```

## Feature Interactions

### SHAP Interaction Values

```python
# Compute SHAP interaction values (computationally expensive)
print("Computing SHAP interaction values (this may take time)...")

# Use smaller sample for interaction analysis
interaction_sample_size = min(50, len(X_sample))
X_interaction = X_sample.iloc[:interaction_sample_size]

try:
    # Compute interaction values for one class
    interaction_values = explainer.shap_interaction_values(X_interaction)

    if isinstance(interaction_values, list):
        # Multi-class case - use first class
        interaction_values = interaction_values[0]

    # Average interaction values across samples
    avg_interactions = np.abs(interaction_values).mean(axis=0)

    # Create interaction heatmap
    plt.figure(figsize=(12, 10))

    # Use top 10 features for readability
    top_10_features = importance_df.head(10)['feature'].tolist()
    top_10_indices = [list(X.columns).index(f) for f in top_10_features]

    interaction_subset = avg_interactions[np.ix_(top_10_indices, top_10_indices)]

    sns.heatmap(interaction_subset,
                xticklabels=top_10_features,
                yticklabels=top_10_features,
                annot=True, fmt='.3f',
                cmap='YlOrRd',
                cbar_kws={'label': 'Mean |Interaction Value|'})
    plt.title('Feature Interaction Matrix (SHAP)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Find strongest interactions
    interaction_pairs = []
    for i in range(len(top_10_features)):
        for j in range(i+1, len(top_10_features)):
            interaction_pairs.append({
                'feature1': top_10_features[i],
                'feature2': top_10_features[j],
                'interaction': interaction_subset[i, j]
            })

    interaction_df = pd.DataFrame(interaction_pairs).sort_values('interaction', ascending=False)

    print("Strongest Feature Interactions:")
    print(interaction_df.head(10))

except Exception as e:
    print(f"Interaction value computation failed: {e}")
    print("Skipping interaction analysis...")
```

### Partial Dependence with SHAP

```python
# SHAP partial dependence plots
def create_partial_dependence_plot(feature_name, class_idx=0):
    """Create partial dependence plot using SHAP."""

    if feature_name not in X.columns:
        print(f"Feature {feature_name} not found")
        return

    feature_idx = list(X.columns).index(feature_name)
    class_name = classes[class_idx]

    # Create range of values for the feature
    feature_values = X_sample.iloc[:, feature_idx]
    value_range = np.linspace(feature_values.min(), feature_values.max(), 20)

    # Calculate SHAP values across the range
    shap_means = []

    for value in value_range:
        # Create modified dataset with fixed feature value
        X_modified = X_sample.copy()
        X_modified.iloc[:, feature_idx] = value

        # Compute SHAP values
        shap_vals = explainer.shap_values(X_modified)

        # Get mean SHAP value for this feature and class
        mean_shap = shap_vals[class_idx][:, feature_idx].mean()
        shap_means.append(mean_shap)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(value_range, shap_means, 'b-', linewidth=2)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.xlabel(f'{feature_name} Value')
    plt.ylabel(f'Mean SHAP Value')
    plt.title(f'Partial Dependence Plot\n{feature_name} for {class_name}')
    plt.grid(True, alpha=0.3)
    plt.show()

    return value_range, shap_means

# Create partial dependence plots for top features
top_3_features = importance_df.head(3)['feature'].tolist()

for feature in top_3_features:
    create_partial_dependence_plot(feature)
```

## Model Debugging and Validation

### SHAP-based Model Validation

```python
# Validate model using SHAP values
def validate_model_with_shap():
    """Validate model behavior using SHAP analysis."""

    print("Model Validation with SHAP")
    print("=" * 30)

    # 1. Check if SHAP values sum to prediction difference
    for i in range(min(5, len(X_sample))):
        sample_pred = model.predict_proba(X_sample.iloc[i:i+1])[0]

        total_shap_impact = {}
        for class_idx, class_name in enumerate(classes):
            base_value = explainer.expected_value[class_idx]
            shap_sum = shap_values[class_idx][i].sum()
            predicted_value = base_value + shap_sum
            actual_pred = sample_pred[class_idx]

            total_shap_impact[class_name] = {
                'base': base_value,
                'shap_sum': shap_sum,
                'predicted': predicted_value,
                'actual': actual_pred,
                'diff': abs(predicted_value - actual_pred)
            }

        print(f"\nSample {i}:")
        for class_name, values in total_shap_impact.items():
            print(f"  {class_name}: Base={values['base']:.3f}, "
                  f"SHAP_sum={values['shap_sum']:.3f}, "
                  f"Predicted={values['predicted']:.3f}, "
                  f"Actual={values['actual']:.3f}, "
                  f"Diff={values['diff']:.3f}")

    # 2. Feature consistency check
    print("\n" + "="*30)
    print("Feature Consistency Analysis")

    feature_directions = {}
    for class_idx, class_name in enumerate(classes):
        feature_directions[class_name] = {}

        for feature_idx, feature_name in enumerate(X.columns):
            shap_vals = shap_values[class_idx][:, feature_idx]

            # Calculate correlation between feature values and SHAP values
            feature_vals = X_sample.iloc[:, feature_idx]
            correlation = np.corrcoef(feature_vals, shap_vals)[0, 1]

            feature_directions[class_name][feature_name] = {
                'correlation': correlation,
                'mean_shap': shap_vals.mean(),
                'std_shap': shap_vals.std()
            }

    # Print most consistent features per class
    for class_name in classes[:3]:  # Show first 3 classes
        class_features = feature_directions[class_name]

        # Sort by absolute correlation
        sorted_features = sorted(class_features.items(),
                               key=lambda x: abs(x[1]['correlation']),
                               reverse=True)

        print(f"\nMost Consistent Features for {class_name}:")
        for feature, stats in sorted_features[:5]:
            print(f"  {feature}: corr={stats['correlation']:.3f}, "
                  f"mean_shap={stats['mean_shap']:.3f}")

validate_model_with_shap()
```

### Outlier Detection with SHAP

```python
# Identify outliers based on SHAP values
def find_shap_outliers(threshold=2):
    """Find samples with unusual SHAP patterns."""

    # Calculate total absolute SHAP impact per sample
    total_impacts = []

    for i in range(len(X_sample)):
        total_impact = 0
        for class_idx in range(len(classes)):
            total_impact += np.abs(shap_values[class_idx][i]).sum()
        total_impacts.append(total_impact)

    total_impacts = np.array(total_impacts)

    # Find outliers using z-score
    z_scores = np.abs((total_impacts - total_impacts.mean()) / total_impacts.std())
    outlier_indices = np.where(z_scores > threshold)[0]

    print(f"Found {len(outlier_indices)} SHAP outliers (threshold: {threshold} std devs)")

    if len(outlier_indices) > 0:
        # Analyze outliers
        outlier_analysis = []

        for idx in outlier_indices[:5]:  # Show first 5
            sample_pred = model.predict(X_sample.iloc[idx:idx+1])[0]
            sample_prob = model.predict_proba(X_sample.iloc[idx:idx+1])[0].max()
            true_label = y_sample.iloc[idx]

            outlier_analysis.append({
                'sample_idx': idx,
                'true_label': true_label,
                'predicted': sample_pred,
                'confidence': sample_prob,
                'total_shap_impact': total_impacts[idx],
                'z_score': z_scores[idx]
            })

        outlier_df = pd.DataFrame(outlier_analysis)
        print("\nSHAP Outliers Analysis:")
        print(outlier_df)

        # Visualize first outlier
        if len(outlier_indices) > 0:
            print(f"\nDetailed analysis of outlier {outlier_indices[0]}:")
            analyze_prediction(outlier_indices[0])

    return outlier_indices

outlier_indices = find_shap_outliers()
```

## Advanced SHAP Visualizations

### Force Plots

```python
# Create SHAP force plots for individual predictions
def create_force_plots(sample_indices, class_idx=0):
    """Create interactive force plots."""

    class_name = classes[class_idx]

    for i, sample_idx in enumerate(sample_indices[:3]):  # Show first 3
        print(f"\nForce Plot for Sample {sample_idx} - Class: {class_name}")

        # Create force plot
        shap.force_plot(
            explainer.expected_value[class_idx],
            shap_values[class_idx][sample_idx],
            X_sample.iloc[sample_idx],
            feature_names=X.columns,
            matplotlib=True,
            show=False
        )
        plt.title(f'SHAP Force Plot - Sample {sample_idx}\n'
                 f'Class: {class_name} | True: {y_sample.iloc[sample_idx]}')
        plt.tight_layout()
        plt.show()

# Create force plots for interesting samples
interesting_samples = [0, 1, 2]  # First few samples
if len(incorrect_preds) > 0:
    interesting_samples.extend(incorrect_preds[:2])

create_force_plots(interesting_samples)
```

### Clustering-based Analysis

```python
# Cluster samples based on SHAP values
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Flatten SHAP values for clustering
shap_features = []
for i in range(len(X_sample)):
    sample_shap = []
    for class_idx in range(len(classes)):
        sample_shap.extend(shap_values[class_idx][i])
    shap_features.append(sample_shap)

shap_features = np.array(shap_features)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
shap_pca = pca.fit_transform(shap_features)

# Cluster based on SHAP patterns
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(shap_features)

# Visualize clusters
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
scatter = plt.scatter(shap_pca[:, 0], shap_pca[:, 1], c=clusters, cmap='viridis')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('SHAP-based Sample Clusters')
plt.colorbar(scatter)

plt.subplot(1, 3, 2)
# Color by true labels
true_label_codes = pd.Categorical(y_sample).codes
scatter2 = plt.scatter(shap_pca[:, 0], shap_pca[:, 1], c=true_label_codes, cmap='tab10')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('Colored by True Labels')

plt.subplot(1, 3, 3)
# Color by prediction confidence
pred_confidence = model.predict_proba(X_sample).max(axis=1)
scatter3 = plt.scatter(shap_pca[:, 0], shap_pca[:, 1], c=pred_confidence, cmap='plasma')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('Colored by Prediction Confidence')
plt.colorbar(scatter3)

plt.tight_layout()
plt.show()

# Analyze cluster characteristics
print("Cluster Analysis:")
for cluster_id in sorted(clusters):
    cluster_mask = clusters == cluster_id
    cluster_samples = np.where(cluster_mask)[0]

    print(f"\nCluster {cluster_id} ({cluster_mask.sum()} samples):")

    # Most common true labels in cluster
    cluster_labels = y_sample.iloc[cluster_samples]
    label_counts = cluster_labels.value_counts()
    print(f"  Most common labels: {dict(label_counts.head(3))}")

    # Average prediction confidence
    cluster_confidence = pred_confidence[cluster_mask].mean()
    print(f"  Average confidence: {cluster_confidence:.3f}")
```

## Summary and Model Insights

```python
def generate_shap_insights():
    """Generate comprehensive SHAP analysis summary."""

    insights = {
        'global_importance': importance_df.head(10),
        'class_specific': {class_name: df.head(5) for class_name, df in class_importance.items()},
        'model_behavior': {},
        'recommendations': []
    }

    # Key insights
    top_feature = importance_df.iloc[0]['feature']
    top_importance = importance_df.iloc[0]['importance']

    print("SHAP Analysis Summary")
    print("=" * 25)
    print(f"Most Important Feature: {top_feature} (importance: {top_importance:.4f})")
    print(f"Number of samples analyzed: {len(X_sample)}")
    print(f"Number of classes: {len(classes)}")

    print("\nKey Insights:")
    print("✓ Feature importance rankings computed for all classes")
    print("✓ Local explanations available for individual predictions")
    print("✓ Model behavior validated through SHAP consistency")

    if len(outlier_indices) > 0:
        print(f"⚠ {len(outlier_indices)} outlier samples detected")

    print("\nRecommendations:")
    print("• Focus on top 10 features for model interpretation")
    print("• Investigate class-specific feature patterns")
    print("• Use local explanations for debugging misclassifications")
    print("• Consider feature engineering based on SHAP insights")

    return insights

insights = generate_shap_insights()
```

### Saving SHAP Results

```python
# Save SHAP analysis results
import pickle
from pathlib import Path

# Create SHAP results directory
shap_dir = Path("results/shap_analysis")
shap_dir.mkdir(parents=True, exist_ok=True)

# Save SHAP values
shap_results = {
    'shap_values': shap_values,
    'expected_values': explainer.expected_value,
    'feature_names': X.columns.tolist(),
    'class_names': classes.tolist(),
    'sample_indices': X_sample.index.tolist(),
    'global_importance': importance_df,
    'class_importance': class_importance
}

with open(shap_dir / 'shap_results.pkl', 'wb') as f:
    pickle.dump(shap_results, f)

# Save summary report
with open(shap_dir / 'shap_summary.txt', 'w') as f:
    f.write("SHAP Analysis Summary Report\n")
    f.write("=" * 35 + "\n\n")
    f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Model Type: {type(model).__name__}\n")
    f.write(f"Samples Analyzed: {len(X_sample)}\n")
    f.write(f"Features: {len(X.columns)}\n")
    f.write(f"Classes: {len(classes)}\n\n")

    f.write("Top 10 Global Features:\n")
    f.write("-" * 25 + "\n")
    for _, row in importance_df.head(10).iterrows():
        f.write(f"{row['feature']}: {row['importance']:.4f}\n")

print(f"SHAP results saved to: {shap_dir}")
```

## Best Practices and Guidelines

### SHAP Best Practices

1. **Computational Efficiency**: Use representative samples for large datasets
2. **Model Compatibility**: TreeExplainer for tree models, LinearExplainer for linear models
3. **Interpretation**: Combine global and local explanations
4. **Validation**: Check SHAP values sum to prediction differences
5. **Visualization**: Use appropriate plots for different analysis types

### Troubleshooting Common Issues

```python
def troubleshoot_shap():
    """Common SHAP troubleshooting tips."""

    print("SHAP Troubleshooting Guide")
    print("=" * 27)

    issues_solutions = {
        "Memory errors": [
            "Reduce sample size for analysis",
            "Use batched computation",
            "Consider approximate methods"
        ],
        "Slow computation": [
            "Use TreeExplainer for tree models",
            "Reduce background dataset size",
            "Use sampling for interaction values"
        ],
        "Pipeline compatibility": [
            "Extract final estimator from pipeline",
            "Transform data before SHAP analysis",
            "Use model-specific explainers"
        ],
        "Visualization issues": [
            "Update SHAP version",
            "Use matplotlib=True for static plots",
            "Check feature name compatibility"
        ]
    }

    for issue, solutions in issues_solutions.items():
        print(f"\n{issue}:")
        for solution in solutions:
            print(f"  • {solution}")

troubleshoot_shap()
```

## Next Steps

After completing SHAP analysis:

1. **[Model Comparison](comparison.md)**: Compare interpretability across models
2. **[Feature Engineering](feature_engineering.md)**: Use insights for new features
3. **[Production Monitoring](monitoring.md)**: Deploy SHAP for model monitoring
4. **[Advanced Methods](advanced_interpretation.md)**: Explore LIME, permutation importance

SHAP analysis provides crucial insights into model behavior and helps build trust in machine learning predictions for protein subcellular localization.
