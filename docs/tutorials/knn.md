# k-Nearest Neighbors Tutorial

This tutorial demonstrates how to train and evaluate k-Nearest Neighbors (k-NN) classifiers for protein subcellular localization using MAP-ExPLoc.

## Overview

The k-NN algorithm is a simple yet effective machine learning method that classifies proteins based on the similarity to their k nearest neighbors in feature space. This tutorial covers the complete workflow from data preparation to model evaluation and interpretation.

## Prerequisites

- Processed protein dataset with features
- Understanding of k-NN algorithm principles
- Basic knowledge of cross-validation and hyperparameter tuning

## Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Import MAP-ExPLoc modules
from mapexploc.models.knn import train_knn, knn_predict, knn_predict_proba, evaluate_knn
```

## Data Preparation

### Loading the Dataset

```python
# Load feature matrix
features_df = pd.read_csv('protein_features.csv')

print(f"Dataset shape: {features_df.shape}")
print(f"Features: {features_df.columns.tolist()}")

# Prepare features and targets
X = features_df.drop(['entry_name', 'localization'], axis=1)
y = features_df['localization']

print(f"Feature matrix shape: {X.shape}")
print(f"Target classes: {sorted(y.unique())}")
print(f"Class distribution:\n{y.value_counts()}")
```

### Train-Test Split

```python
# Split data for training and evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Verify class distribution is maintained
print("\nTraining set distribution:")
print(y_train.value_counts(normalize=True))
print("\nTest set distribution:")
print(y_test.value_counts(normalize=True))
```

## Basic k-NN Training

### Default Configuration

```python
# Train k-NN with default parameters
knn_result = train_knn(X_train, y_train)

print("k-NN Training Results")
print("=" * 25)
print(f"Best parameters: {knn_result['best_params']}")
print(f"Best CV score: {knn_result['best_cv_score']:.4f}")

# Access the trained model
best_model = knn_result['model']
print(f"Model type: {type(best_model)}")
```

### Understanding the Pipeline

The k-NN implementation includes automatic preprocessing:

```python
# Examine the pipeline components
pipeline = knn_result['model']
print("Pipeline steps:")
for step_name, step in pipeline.named_steps.items():
    print(f"  {step_name}: {step}")

# Get the actual k-NN classifier
knn_classifier = pipeline.named_steps['knn']
print(f"\nk-NN parameters:")
print(f"  n_neighbors: {knn_classifier.n_neighbors}")
print(f"  weights: {knn_classifier.weights}")
print(f"  metric: {knn_classifier.metric}")
```

## Hyperparameter Tuning

### Custom Parameter Grid

```python
# Define custom parameter grid
custom_param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9, 11, 15, 21],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan', 'minkowski']
}

# Train with custom parameters
knn_custom = train_knn(
    X_train, y_train,
    param_grid=custom_param_grid,
    cv=5,
    scoring='f1_macro'
)

print("Custom k-NN Training Results")
print("=" * 30)
print(f"Best parameters: {knn_custom['best_params']}")
print(f"Best CV score: {knn_custom['best_cv_score']:.4f}")
```

### Cross-Validation Results Analysis

```python
# Analyze cross-validation results
cv_results = pd.DataFrame(knn_result['cv_results'])

# Key columns to analyze
key_cols = ['mean_test_score', 'std_test_score', 'param_knn__n_neighbors',
            'param_knn__weights', 'param_knn__metric']
cv_summary = cv_results[key_cols].copy()

# Sort by mean test score
cv_summary = cv_summary.sort_values('mean_test_score', ascending=False)
print("Top 10 Parameter Combinations:")
print(cv_summary.head(10))

# Visualize k value vs performance
plt.figure(figsize=(12, 6))

# Group by k value and calculate mean performance
k_performance = cv_results.groupby('param_knn__n_neighbors')['mean_test_score'].agg(['mean', 'std'])

plt.errorbar(k_performance.index, k_performance['mean'],
             yerr=k_performance['std'], marker='o', capsize=5)
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Mean F1-macro Score')
plt.title('k-NN Performance vs k Value')
plt.grid(True, alpha=0.3)
plt.show()
```

## Model Evaluation

### Basic Performance Metrics

```python
# Make predictions on test set
y_pred = knn_predict(best_model, X_test)
y_pred_proba = knn_predict_proba(best_model, X_test)

# Calculate basic metrics
from sklearn.metrics import accuracy_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')

print("Test Set Performance")
print("=" * 20)
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-macro: {f1_macro:.4f}")
print(f"F1-weighted: {f1_weighted:.4f}")
```

### Detailed Classification Report

```python
# Generate detailed classification report
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

print("Classification Report:")
print(report_df.round(3))

# Visualize precision, recall, and F1-score by class
metrics_df = report_df.iloc[:-3, :3]  # Exclude avg rows and support column

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
metrics = ['precision', 'recall', 'f1-score']

for i, metric in enumerate(metrics):
    metrics_df[metric].plot(kind='bar', ax=axes[i])
    axes[i].set_title(f'{metric.capitalize()} by Localization')
    axes[i].set_xlabel('Localization')
    axes[i].set_ylabel(metric.capitalize())
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

### Confusion Matrix Analysis

```python
# Generate and visualize confusion matrix
cm = confusion_matrix(y_test, y_pred)
classes = sorted(y_test.unique())

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes,
            cmap='Blues', cbar_kws={'label': 'Count'})
plt.title('k-NN Confusion Matrix')
plt.xlabel('Predicted Localization')
plt.ylabel('True Localization')
plt.tight_layout()
plt.show()

# Calculate per-class accuracy
per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
accuracy_df = pd.DataFrame({
    'Localization': classes,
    'Accuracy': per_class_accuracy,
    'Support': cm.sum(axis=1)
}).sort_values('Accuracy', ascending=False)

print("Per-class Accuracy:")
print(accuracy_df.round(3))
```

## Advanced Analysis

### Distance Analysis

```python
# Analyze distances to nearest neighbors for different classes
from sklearn.neighbors import NearestNeighbors

# Extract features after preprocessing (scaling)
scaler = best_model.named_steps['scaler']
X_test_scaled = scaler.transform(X_test)

# Find nearest neighbors
knn_finder = NearestNeighbors(n_neighbors=knn_classifier.n_neighbors)
knn_finder.fit(scaler.transform(X_train))

distances, indices = knn_finder.kneighbors(X_test_scaled)

# Analyze average distances by true class
distance_by_class = pd.DataFrame({
    'true_class': y_test.values,
    'avg_distance': distances.mean(axis=1),
    'min_distance': distances.min(axis=1),
    'max_distance': distances.max(axis=1)
})

distance_summary = distance_by_class.groupby('true_class').agg({
    'avg_distance': ['mean', 'std'],
    'min_distance': ['mean', 'std'],
    'max_distance': ['mean', 'std']
}).round(3)

print("Distance Analysis by True Class:")
print(distance_summary)

# Visualize distance distributions
plt.figure(figsize=(12, 8))
classes_to_plot = sorted(y_test.unique())
for i, cls in enumerate(classes_to_plot):
    class_distances = distance_by_class[distance_by_class['true_class'] == cls]['avg_distance']
    plt.hist(class_distances, bins=20, alpha=0.7, label=cls, density=True)

plt.xlabel('Average Distance to k Nearest Neighbors')
plt.ylabel('Density')
plt.title('Distribution of Distances to Nearest Neighbors by Class')
plt.legend()
plt.show()
```

### Prediction Confidence Analysis

```python
# Analyze prediction confidence (probability of predicted class)
predicted_proba_max = y_pred_proba.max(axis=1)
correct_predictions = (y_pred == y_test)

confidence_analysis = pd.DataFrame({
    'confidence': predicted_proba_max,
    'correct': correct_predictions,
    'true_class': y_test.values,
    'pred_class': y_pred
})

# Overall confidence vs accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(predicted_proba_max[correct_predictions], bins=30, alpha=0.7,
         label='Correct', density=True)
plt.hist(predicted_proba_max[~correct_predictions], bins=30, alpha=0.7,
         label='Incorrect', density=True)
plt.xlabel('Prediction Confidence')
plt.ylabel('Density')
plt.title('Confidence Distribution: Correct vs Incorrect Predictions')
plt.legend()

plt.subplot(1, 2, 2)
# Confidence vs accuracy in bins
confidence_bins = np.linspace(0, 1, 11)
bin_accuracy = []
bin_centers = []

for i in range(len(confidence_bins)-1):
    mask = ((predicted_proba_max >= confidence_bins[i]) &
            (predicted_proba_max < confidence_bins[i+1]))
    if mask.sum() > 0:
        bin_accuracy.append(correct_predictions[mask].mean())
        bin_centers.append((confidence_bins[i] + confidence_bins[i+1]) / 2)

plt.plot(bin_centers, bin_accuracy, 'o-', linewidth=2, markersize=8)
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
plt.xlabel('Confidence')
plt.ylabel('Accuracy')
plt.title('Reliability Diagram (Calibration)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Error Analysis

```python
# Analyze misclassified samples
misclassified_mask = ~correct_predictions
misclassified_df = pd.DataFrame({
    'entry_name': X_test.index[misclassified_mask],
    'true_class': y_test.values[misclassified_mask],
    'pred_class': y_pred[misclassified_mask],
    'confidence': predicted_proba_max[misclassified_mask],
    'avg_distance': distances.mean(axis=1)[misclassified_mask]
})

print(f"Total misclassified samples: {len(misclassified_df)}")
print("\nMisclassification patterns:")
confusion_patterns = misclassified_df.groupby(['true_class', 'pred_class']).size()
print(confusion_patterns.sort_values(ascending=False))

# Most confident wrong predictions
print("\nMost confident incorrect predictions:")
most_confident_wrong = misclassified_df.nlargest(10, 'confidence')
print(most_confident_wrong[['true_class', 'pred_class', 'confidence']].round(3))
```

## Feature Importance Analysis

### Permutation Importance

```python
from sklearn.inspection import permutation_importance

# Calculate permutation importance
perm_importance = permutation_importance(
    best_model, X_test, y_test,
    n_repeats=10, random_state=42,
    scoring='f1_macro'
)

# Create importance dataframe
importance_df = pd.DataFrame({
    'feature': X_test.columns,
    'importance_mean': perm_importance.importances_mean,
    'importance_std': perm_importance.importances_std
}).sort_values('importance_mean', ascending=False)

# Plot top features
plt.figure(figsize=(10, 8))
top_features = importance_df.head(15)
plt.barh(range(len(top_features)), top_features['importance_mean'],
         xerr=top_features['importance_std'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Permutation Importance (F1-macro decrease)')
plt.title('Top 15 Most Important Features (k-NN)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("Top 10 Most Important Features:")
print(importance_df.head(10).round(4))
```

## Model Comparison

### Comparing Different k Values

```python
# Compare models with different k values
k_values = [1, 3, 5, 7, 11, 15, 21]
comparison_results = []

for k in k_values:
    # Train model with specific k
    param_grid = {
        'knn__n_neighbors': [k],
        'knn__weights': ['distance'],  # Use distance weighting
        'knn__metric': ['euclidean']
    }

    model_result = train_knn(X_train, y_train, param_grid=param_grid, cv=3)

    # Evaluate on test set
    y_pred_k = knn_predict(model_result['model'], X_test)
    accuracy_k = accuracy_score(y_test, y_pred_k)
    f1_k = f1_score(y_test, y_pred_k, average='macro')

    comparison_results.append({
        'k': k,
        'cv_score': model_result['best_cv_score'],
        'test_accuracy': accuracy_k,
        'test_f1_macro': f1_k
    })

# Create comparison dataframe
comparison_df = pd.DataFrame(comparison_results)
print("k-Value Comparison:")
print(comparison_df.round(4))

# Visualize comparison
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# CV Score vs k
axes[0].plot(comparison_df['k'], comparison_df['cv_score'], 'o-', linewidth=2)
axes[0].set_xlabel('k Value')
axes[0].set_ylabel('Cross-Validation F1-macro Score')
axes[0].set_title('Cross-Validation Performance vs k')
axes[0].grid(True, alpha=0.3)

# Test performance vs k
axes[1].plot(comparison_df['k'], comparison_df['test_accuracy'], 'o-',
             linewidth=2, label='Accuracy')
axes[1].plot(comparison_df['k'], comparison_df['test_f1_macro'], 's-',
             linewidth=2, label='F1-macro')
axes[1].set_xlabel('k Value')
axes[1].set_ylabel('Test Performance')
axes[1].set_title('Test Performance vs k')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Production Considerations

### Model Serialization

```python
import joblib
from pathlib import Path

# Save the best model
model_path = Path("models/knn_best_model.pkl")
model_path.parent.mkdir(exist_ok=True)
joblib.dump(best_model, model_path)

# Save preprocessing information
preprocessing_info = {
    'feature_columns': X_train.columns.tolist(),
    'classes': sorted(y_train.unique()),
    'scaler_params': best_model.named_steps['scaler'].get_params()
}

import json
with open(model_path.parent / "knn_preprocessing_info.json", "w") as f:
    json.dump(preprocessing_info, f, indent=2)

print(f"Model saved to: {model_path}")
print(f"Preprocessing info saved to: {model_path.parent}/knn_preprocessing_info.json")
```

### Prediction Function

```python
def predict_localization(sequence_features, model_path="models/knn_best_model.pkl"):
    """
    Predict subcellular localization for new protein sequences.

    Args:
        sequence_features: DataFrame with same features as training data
        model_path: Path to saved model

    Returns:
        predictions and probabilities
    """
    # Load model
    model = joblib.load(model_path)

    # Make predictions
    predictions = model.predict(sequence_features)
    probabilities = model.predict_proba(sequence_features)

    # Get class names
    classes = model.named_steps['knn'].classes_

    # Create results dataframe
    results = pd.DataFrame(probabilities, columns=classes)
    results['predicted_localization'] = predictions
    results['confidence'] = probabilities.max(axis=1)

    return results

# Example usage
# new_predictions = predict_localization(new_sequence_features)
```

## Summary and Best Practices

```python
def summarize_knn_results(knn_result, test_accuracy, test_f1):
    """Generate a summary of k-NN training results."""
    summary = f"""
k-NN Model Summary
==================
Best Parameters: {knn_result['best_params']}
Cross-validation F1-macro: {knn_result['best_cv_score']:.4f}
Test Accuracy: {test_accuracy:.4f}
Test F1-macro: {test_f1:.4f}

Model Components:
- Feature scaling: StandardScaler
- Algorithm: k-Nearest Neighbors
- Distance metric: {knn_result['best_params']['knn__metric']}
- Weighting: {knn_result['best_params']['knn__weights']}
- Number of neighbors: {knn_result['best_params']['knn__n_neighbors']}

Recommendations:
- k-NN is sensitive to feature scaling (handled automatically)
- Performance depends heavily on k value selection
- Distance weighting often improves performance
- Consider curse of dimensionality for high-dimensional features
- Suitable for datasets with clear local patterns
    """
    return summary

print(summarize_knn_results(knn_result, accuracy, f1_macro))
```

### Best Practices for k-NN in Protein Classification

1. **Feature Scaling**: Always use feature scaling (handled automatically)
2. **k Selection**: Use cross-validation to find optimal k value
3. **Distance Metrics**: Try different distance metrics (Euclidean, Manhattan)
4. **Weighting**: Consider distance-based weighting for better performance
5. **Curse of Dimensionality**: Be aware of performance degradation with many features
6. **Class Imbalance**: k-NN can be sensitive to imbalanced classes
7. **Computational Cost**: Consider efficiency for large datasets

## Next Steps

After completing k-NN analysis:

1. **[Random Forest Tutorial](rf.md)**: Compare with ensemble methods
2. **[SHAP Analysis](shap.md)**: Understand model predictions
3. **[Model Comparison](evaluation.md)**: Compare different algorithms
4. **Hyperparameter Optimization**: Fine-tune for production use
