# Random Forest Tutorial

This tutorial demonstrates how to train and evaluate Random Forest classifiers for protein subcellular localization using MAP-ExPLoc, including advanced techniques like SMOTE for handling class imbalance.

## Overview

Random Forest is an ensemble method that builds multiple decision trees and combines their predictions. This tutorial covers the complete workflow including hyperparameter tuning, class imbalance handling with SMOTE, feature importance analysis, and model interpretation.

## Prerequisites

- Processed protein dataset with features
- Understanding of ensemble methods and decision trees
- Knowledge of class imbalance techniques

## Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Import MAP-ExPLoc modules
from mapexploc.models.rf import train_random_forest
```

## Data Preparation

### Loading and Preparing the Dataset

```python
# Load feature matrix
features_df = pd.read_csv('protein_features.csv')

print(f"Dataset shape: {features_df.shape}")
print(f"Features: {list(features_df.columns)}")

# Prepare features and targets
X = features_df.drop(['entry_name', 'localization'], axis=1)
y = features_df['localization']

print(f"Feature matrix shape: {X.shape}")
print(f"Target classes: {sorted(y.unique())}")

# Analyze class distribution
class_counts = y.value_counts()
print(f"\nClass distribution:")
print(class_counts)

# Calculate imbalance ratio
imbalance_ratio = class_counts.max() / class_counts.min()
print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
```

### Train-Test Split with Stratification

```python
# Stratified split to maintain class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Verify stratification worked
print("\nClass distribution in training set:")
print(y_train.value_counts(normalize=True).sort_index())
print("\nClass distribution in test set:")
print(y_test.value_counts(normalize=True).sort_index())
```

## Basic Random Forest Training

### Default Configuration with SMOTE

```python
# Train Random Forest with default parameters and SMOTE
rf_result = train_random_forest(X_train, y_train, use_smote=True)

print("Random Forest Training Results")
print("=" * 35)
print(f"Best parameters: {rf_result['best_params']}")
print(f"Best CV score: {rf_result['best_cv_score']:.4f}")

# Access the trained model
best_model = rf_result['model']
print(f"Model type: {type(best_model)}")
```

### Understanding the Pipeline

```python
# Examine pipeline components
pipeline = rf_result['model']
print("Pipeline steps:")
for step_name, step in pipeline.named_steps.items():
    print(f"  {step_name}: {step}")

# Get the Random Forest classifier
if 'rf' in pipeline.named_steps:
    rf_classifier = pipeline.named_steps['rf']
    print(f"\nRandom Forest parameters:")
    print(f"  n_estimators: {rf_classifier.n_estimators}")
    print(f"  max_depth: {rf_classifier.max_depth}")
    print(f"  min_samples_split: {rf_classifier.min_samples_split}")
    print(f"  min_samples_leaf: {rf_classifier.min_samples_leaf}")
```

## Advanced Training with Custom Parameters

### Custom Parameter Grid

```python
# Define custom parameter grid for extensive search
custom_param_grid = {
    'rf__n_estimators': [100, 200, 300],
    'rf__max_depth': [10, 20, 30, None],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4],
    'rf__max_features': ['sqrt', 'log2', None]
}

# Train with custom parameters (using fewer iterations for demo)
rf_custom = train_random_forest(
    X_train, y_train,
    param_grid=custom_param_grid,
    search_type='randomized',
    n_iter=20,  # Reduced for demonstration
    cv=5,
    use_smote=True,
    random_state=42
)

print("Custom Random Forest Results")
print("=" * 30)
print(f"Best parameters: {rf_custom['best_params']}")
print(f"Best CV score: {rf_custom['best_cv_score']:.4f}")
```

### Comparison with and without SMOTE

```python
# Train models with and without SMOTE for comparison
print("Comparing SMOTE vs No SMOTE")
print("=" * 30)

# Without SMOTE
rf_no_smote = train_random_forest(
    X_train, y_train,
    use_smote=False,
    cv=3  # Reduced CV for faster demo
)

# With SMOTE (if classes are large enough)
try:
    rf_with_smote = train_random_forest(
        X_train, y_train,
        use_smote=True,
        cv=3
    )

    print(f"Without SMOTE CV score: {rf_no_smote['best_cv_score']:.4f}")
    print(f"With SMOTE CV score: {rf_with_smote['best_cv_score']:.4f}")

    # Use the better model for subsequent analysis
    if rf_with_smote['best_cv_score'] > rf_no_smote['best_cv_score']:
        best_rf = rf_with_smote
        smote_used = True
        print("SMOTE improved performance - using SMOTE model")
    else:
        best_rf = rf_no_smote
        smote_used = False
        print("SMOTE did not improve performance - using standard model")

except Exception as e:
    print(f"SMOTE training failed: {e}")
    print("Using model without SMOTE")
    best_rf = rf_no_smote
    smote_used = False
```

## Model Evaluation

### Test Set Performance

```python
# Make predictions
y_pred = best_rf['model'].predict(X_test)
y_pred_proba = best_rf['model'].predict_proba(X_test)

# Calculate performance metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')

print("Test Set Performance")
print("=" * 20)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (macro): {precision:.4f}")
print(f"Recall (macro): {recall:.4f}")
print(f"F1-macro: {f1_macro:.4f}")
print(f"F1-weighted: {f1_weighted:.4f}")
```

### Detailed Classification Report

```python
# Generate and visualize classification report
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

print("Classification Report:")
print(report_df.round(3))

# Visualize metrics by class
metrics_df = report_df.iloc[:-3, :3]  # Exclude summary rows

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

# Plot each metric
metrics = ['precision', 'recall', 'f1-score']
for i, metric in enumerate(metrics):
    ax = axes[i]
    bars = ax.bar(range(len(metrics_df)), metrics_df[metric])
    ax.set_title(f'{metric.capitalize()} by Localization')
    ax.set_xlabel('Localization')
    ax.set_ylabel(metric.capitalize())
    ax.set_xticks(range(len(metrics_df)))
    ax.set_xticklabels(metrics_df.index, rotation=45, ha='right')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')

# Support (number of samples)
ax = axes[3]
support_data = report_df.iloc[:-3, 3]  # Support column
bars = ax.bar(range(len(support_data)), support_data)
ax.set_title('Support (Number of Test Samples)')
ax.set_xlabel('Localization')
ax.set_ylabel('Count')
ax.set_xticks(range(len(support_data)))
ax.set_xticklabels(support_data.index, rotation=45, ha='right')

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{int(height)}', ha='center', va='bottom')

plt.tight_layout()
plt.show()
```

### Confusion Matrix Analysis

```python
# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
classes = sorted(y_test.unique())

# Plot confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes,
            cmap='Blues', cbar_kws={'label': 'Count'})
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted Localization')
plt.ylabel('True Localization')
plt.tight_layout()
plt.show()

# Calculate per-class metrics from confusion matrix
per_class_metrics = []
for i, class_name in enumerate(classes):
    tp = cm[i, i]  # True positives
    fp = cm[:, i].sum() - tp  # False positives
    fn = cm[i, :].sum() - tp  # False negatives
    tn = cm.sum() - tp - fp - fn  # True negatives

    precision_class = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_class = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    per_class_metrics.append({
        'Class': class_name,
        'Precision': precision_class,
        'Recall': recall_class,
        'Specificity': specificity,
        'Support': cm[i, :].sum()
    })

metrics_summary = pd.DataFrame(per_class_metrics)
print("Per-class Metrics Summary:")
print(metrics_summary.round(3))
```

## Feature Importance Analysis

### Built-in Feature Importance

```python
# Extract feature importance from Random Forest
if 'rf' in best_rf['model'].named_steps:
    feature_importance = best_rf['model'].named_steps['rf'].feature_importances_

    # Create feature importance dataframe
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    # Plot top features
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance (Gini Impurity)')
    plt.title('Top 15 Most Important Features (Random Forest)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    print("Top 10 Most Important Features:")
    print(importance_df.head(10))
```

### Permutation Importance

```python
from sklearn.inspection import permutation_importance

# Calculate permutation importance
print("Calculating permutation importance...")
perm_importance = permutation_importance(
    best_rf['model'], X_test, y_test,
    n_repeats=10, random_state=42,
    scoring='f1_macro'
)

# Create permutation importance dataframe
perm_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance_mean': perm_importance.importances_mean,
    'importance_std': perm_importance.importances_std
}).sort_values('importance_mean', ascending=False)

# Compare built-in vs permutation importance
comparison_df = pd.merge(
    importance_df[['feature', 'importance']].rename(columns={'importance': 'gini_importance'}),
    perm_importance_df[['feature', 'importance_mean']].rename(columns={'importance_mean': 'perm_importance'}),
    on='feature'
)

# Plot comparison
plt.figure(figsize=(10, 6))
plt.scatter(comparison_df['gini_importance'], comparison_df['perm_importance'], alpha=0.7)
plt.xlabel('Gini-based Importance')
plt.ylabel('Permutation Importance')
plt.title('Gini vs Permutation Feature Importance')

# Add diagonal line for reference
max_val = max(comparison_df['gini_importance'].max(), comparison_df['perm_importance'].max())
plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Perfect Agreement')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("Top 10 Features by Permutation Importance:")
print(perm_importance_df.head(10).round(4))
```

## Prediction Confidence and Calibration

### Prediction Confidence Analysis

```python
# Analyze prediction confidence
predicted_proba_max = y_pred_proba.max(axis=1)
correct_predictions = (y_pred == y_test)

# Confidence vs accuracy
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(predicted_proba_max[correct_predictions], bins=30, alpha=0.7,
         label='Correct', density=True)
plt.hist(predicted_proba_max[~correct_predictions], bins=30, alpha=0.7,
         label='Incorrect', density=True)
plt.xlabel('Prediction Confidence')
plt.ylabel('Density')
plt.title('Confidence Distribution')
plt.legend()

# Reliability diagram
plt.subplot(1, 3, 2)
from sklearn.calibration import calibration_curve

fraction_of_positives, mean_predicted_value = calibration_curve(
    (y_pred == y_test).astype(int), predicted_proba_max, n_bins=10
)

plt.plot(mean_predicted_value, fraction_of_positives, "s-", linewidth=2,
         label='Random Forest')
plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Plot')
plt.legend()

# Confidence by class
plt.subplot(1, 3, 3)
confidence_by_class = pd.DataFrame({
    'true_class': y_test.values,
    'confidence': predicted_proba_max,
    'correct': correct_predictions
})

class_confidence = confidence_by_class.groupby('true_class')['confidence'].mean().sort_values(ascending=False)
class_confidence.plot(kind='bar')
plt.title('Average Confidence by True Class')
plt.xlabel('True Class')
plt.ylabel('Average Confidence')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()

print("Confidence Statistics by Class:")
confidence_stats = confidence_by_class.groupby('true_class').agg({
    'confidence': ['mean', 'std', 'min', 'max'],
    'correct': 'mean'
}).round(3)
print(confidence_stats)
```

## Learning Curves and Model Analysis

### Learning Curves

```python
from sklearn.model_selection import learning_curve

# Generate learning curves
print("Generating learning curves...")
train_sizes, train_scores, val_scores = learning_curve(
    best_rf['model'], X_train, y_train,
    cv=3, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='f1_macro'
)

# Calculate means and standard deviations
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)
val_scores_std = np.std(val_scores, axis=1)

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                 val_scores_mean + val_scores_std, alpha=0.1, color="g")

plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Cross-validation score")

plt.xlabel('Training Set Size')
plt.ylabel('F1-macro Score')
plt.title('Learning Curves (Random Forest)')
plt.legend(loc="best")
plt.grid(True, alpha=0.3)
plt.show()

print(f"Final training score: {train_scores_mean[-1]:.4f} ± {train_scores_std[-1]:.4f}")
print(f"Final validation score: {val_scores_mean[-1]:.4f} ± {val_scores_std[-1]:.4f}")
```

### Tree Analysis

```python
# Analyze individual trees in the forest
if 'rf' in best_rf['model'].named_steps:
    rf_estimator = best_rf['model'].named_steps['rf']

    print("Random Forest Analysis")
    print("=" * 25)
    print(f"Number of trees: {rf_estimator.n_estimators}")
    print(f"Number of features per tree: {rf_estimator.max_features}")

    # Tree depth analysis
    tree_depths = [tree.tree_.max_depth for tree in rf_estimator.estimators_]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(tree_depths, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Tree Depth')
    plt.ylabel('Number of Trees')
    plt.title('Distribution of Tree Depths')
    plt.axvline(np.mean(tree_depths), color='red', linestyle='--',
                label=f'Mean: {np.mean(tree_depths):.1f}')
    plt.legend()

    # Number of leaves analysis
    tree_leaves = [tree.tree_.n_leaves for tree in rf_estimator.estimators_]

    plt.subplot(1, 2, 2)
    plt.hist(tree_leaves, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Number of Leaves')
    plt.ylabel('Number of Trees')
    plt.title('Distribution of Tree Sizes (Leaves)')
    plt.axvline(np.mean(tree_leaves), color='red', linestyle='--',
                label=f'Mean: {np.mean(tree_leaves):.0f}')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print(f"Average tree depth: {np.mean(tree_depths):.2f} ± {np.std(tree_depths):.2f}")
    print(f"Average number of leaves: {np.mean(tree_leaves):.0f} ± {np.std(tree_leaves):.0f}")
```

## Error Analysis

### Misclassification Analysis

```python
# Analyze misclassified samples
misclassified_mask = ~correct_predictions
misclassified_indices = X_test.index[misclassified_mask]

misclassified_analysis = pd.DataFrame({
    'entry_name': misclassified_indices,
    'true_class': y_test.values[misclassified_mask],
    'pred_class': y_pred[misclassified_mask],
    'confidence': predicted_proba_max[misclassified_mask]
})

print(f"Total misclassified samples: {len(misclassified_analysis)}")
print(f"Misclassification rate: {len(misclassified_analysis) / len(y_test):.1%}")

# Most common misclassification patterns
confusion_patterns = misclassified_analysis.groupby(['true_class', 'pred_class']).size()
print("\nMost common misclassification patterns:")
print(confusion_patterns.sort_values(ascending=False).head(10))

# High confidence misclassifications
high_conf_wrong = misclassified_analysis[misclassified_analysis['confidence'] > 0.8]
print(f"\nHigh confidence wrong predictions (>0.8): {len(high_conf_wrong)}")
if len(high_conf_wrong) > 0:
    print("Examples:")
    print(high_conf_wrong[['true_class', 'pred_class', 'confidence']].head())
```

## Model Serialization and Deployment

### Saving the Model

```python
import joblib
from pathlib import Path

# Create model directory
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

# Save the trained model
model_path = model_dir / "random_forest_best.pkl"
joblib.dump(best_rf['model'], model_path)

# Save training metadata
metadata = {
    'model_type': 'RandomForest',
    'features': X.columns.tolist(),
    'classes': sorted(y.unique()),
    'best_params': best_rf['best_params'],
    'cv_score': best_rf['best_cv_score'],
    'test_accuracy': accuracy,
    'test_f1_macro': f1_macro,
    'smote_used': smote_used,
    'training_samples': len(X_train),
    'test_samples': len(X_test)
}

import json
metadata_path = model_dir / "random_forest_metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Model saved to: {model_path}")
print(f"Metadata saved to: {metadata_path}")
```

### Prediction Pipeline

```python
def predict_localization_rf(features_df, model_path="models/random_forest_best.pkl"):
    """
    Predict protein subcellular localization using trained Random Forest.

    Args:
        features_df: DataFrame with protein features
        model_path: Path to saved model

    Returns:
        DataFrame with predictions and probabilities
    """
    # Load model
    model = joblib.load(model_path)

    # Ensure features are in correct order
    feature_cols = [col for col in features_df.columns
                   if col not in ['entry_name', 'localization']]
    X = features_df[feature_cols]

    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    # Get class names
    classes = model.classes_

    # Create results DataFrame
    results = pd.DataFrame(probabilities, columns=[f'prob_{cls}' for cls in classes])
    results['predicted_localization'] = predictions
    results['confidence'] = probabilities.max(axis=1)

    # Add entry names if available
    if 'entry_name' in features_df.columns:
        results['entry_name'] = features_df['entry_name'].values

    return results

# Example usage (commented out)
# new_predictions = predict_localization_rf(new_features_df)
# print(new_predictions.head())
```

## Summary and Best Practices

```python
def generate_rf_summary(rf_result, test_metrics, smote_used):
    """Generate comprehensive Random Forest training summary."""

    summary = f"""
Random Forest Model Summary
===========================
Training Configuration:
- SMOTE used: {smote_used}
- Cross-validation folds: 5
- Search type: {'Grid' if 'grid_search' in rf_result else 'Randomized'}

Best Parameters:
{json.dumps(rf_result['best_params'], indent=2)}

Performance Metrics:
- Cross-validation F1-macro: {rf_result['best_cv_score']:.4f}
- Test Accuracy: {test_metrics['accuracy']:.4f}
- Test F1-macro: {test_metrics['f1_macro']:.4f}
- Test F1-weighted: {test_metrics['f1_weighted']:.4f}

Model Characteristics:
- Ensemble method with multiple decision trees
- Built-in feature importance ranking
- Handles mixed data types well
- Robust to outliers
- Good performance with default parameters

Recommendations:
✓ Excellent baseline model for classification tasks
✓ Feature importance provides interpretability
✓ SMOTE helpful for imbalanced classes
✓ Consider hyperparameter tuning for optimal performance
✓ Monitor for overfitting with very deep trees
    """

    return summary

test_metrics = {
    'accuracy': accuracy,
    'f1_macro': f1_macro,
    'f1_weighted': f1_weighted
}

print(generate_rf_summary(best_rf, test_metrics, smote_used))
```

### Best Practices for Random Forest

1. **Feature Engineering**: Random Forest handles mixed data types well
2. **Hyperparameter Tuning**: Focus on n_estimators, max_depth, min_samples_split
3. **Class Imbalance**: Use SMOTE or class_weight for imbalanced datasets
4. **Feature Importance**: Use both Gini and permutation importance
5. **Cross-Validation**: Always use stratified CV for classification
6. **Overfitting**: Monitor training vs validation curves
7. **Interpretability**: Combine feature importance with SHAP analysis

## Next Steps

After completing Random Forest analysis:

1. **[SHAP Analysis Tutorial](shap.md)**: Deep dive into model interpretability
2. **[Model Comparison](evaluation.md)**: Compare Random Forest with k-NN
3. **[Ensemble Methods](ensemble.md)**: Combine multiple models
4. **Production Deployment**: Deploy model as API or service

The Random Forest model provides a strong baseline with built-in interpretability through feature importance analysis.
