# Model Explainability Guide

MAP-ExPLoc provides comprehensive model interpretability through multiple explanation methods. This guide covers how to choose appropriate explainers, configure them for your use case, and interpret the results effectively.

## Overview

Model explainability is crucial for understanding protein localization predictions, especially in scientific and clinical applications. MAP-ExPLoc supports several state-of-the-art explanation methods:

- **SHAP (SHapley Additive exPlanations)**: Game theory-based feature attributions
- **LIME (Local Interpretable Model-agnostic Explanations)**: Local surrogate models
- **Integrated Gradients**: Attribution method for neural networks
- **Attention Visualization**: For transformer-based models

## Explainer Selection

### Automatic Explainer Selection

The `ShapExplainer` automatically selects the most appropriate method based on your model:

```python
from mapexploc import ShapExplainer

# Automatic explainer selection
explainer = ShapExplainer(model_adapter)

# The explainer will automatically choose:
# - DeepExplainer: For neural networks with embedding layers
# - TreeExplainer: For tree-based models (Random Forest, XGBoost)
# - LinearExplainer: For linear models
# - KernelExplainer: Universal fallback for any model
```

### Manual Explainer Selection

For specific use cases, you can manually select explainer types:

```python
from mapexploc.explainers import (
    ShapTreeExplainer, ShapDeepExplainer,
    ShapKernelExplainer, LimeExplainer
)

# For tree-based models (fastest)
tree_explainer = ShapTreeExplainer(random_forest_adapter)

# For neural networks with embeddings
deep_explainer = ShapDeepExplainer(neural_network_adapter)

# Universal explainer (slower but works with any model)
kernel_explainer = ShapKernelExplainer(any_model_adapter)

# LIME for local explanations
lime_explainer = LimeExplainer(any_model_adapter)
```

## Background Dataset Configuration

### Importance of Representative Data

Provide a background dataset representative of your domain to obtain stable and meaningful attributions:

```python
import pandas as pd

# Load representative background data
background_sequences = [
    "MKTIIALSYIFCLVFADYKDDDDK",  # Cytoplasm example
    "MALWMRLLPLLALLALWGPGPGGA",  # Membrane example
    "MVLSEGEWQLVLHVWAKVEADVA",   # Nucleus example
    # ... more representative sequences
]

# Configure explainer with background
explainer = ShapExplainer(
    model_adapter,
    background_data=background_sequences,
    background_size=100  # Subsample if dataset is large
)
```

### Background Data Strategies

```python
# Strategy 1: Random sampling from training data
def create_random_background(training_data, size=100):
    return training_data.sample(n=size, random_state=42)['sequence'].tolist()

# Strategy 2: Stratified sampling by class
def create_stratified_background(training_data, size=100):
    background = []
    classes = training_data['localization'].unique()

    samples_per_class = size // len(classes)
    for class_name in classes:
        class_data = training_data[training_data['localization'] == class_name]
        samples = class_data.sample(n=min(samples_per_class, len(class_data)),
                                   random_state=42)
        background.extend(samples['sequence'].tolist())

    return background

# Strategy 3: K-means clustering for diversity
from sklearn.cluster import KMeans

def create_diverse_background(feature_matrix, sequences, size=100):
    # Cluster sequences and pick representatives
    kmeans = KMeans(n_clusters=size, random_state=42)
    clusters = kmeans.fit_predict(feature_matrix)

    background = []
    for cluster_id in range(size):
        cluster_indices = np.where(clusters == cluster_id)[0]
        if len(cluster_indices) > 0:
            # Pick sequence closest to cluster center
            distances = np.linalg.norm(
                feature_matrix[cluster_indices] - kmeans.cluster_centers_[cluster_id],
                axis=1
            )
            closest_idx = cluster_indices[np.argmin(distances)]
            background.append(sequences[closest_idx])

    return background
```

## Explainer Configuration

### SHAP Configuration

```python
# Comprehensive SHAP configuration
shap_explainer = ShapExplainer(
    model_adapter,
    background_data=background_sequences,
    background_size=100,
    max_evals=1000,  # Maximum evaluations for KernelExplainer
    link='identity',  # Link function: 'identity', 'logit'
    feature_perturbation='interventional'  # For TreeExplainer
)

# Advanced configuration for specific explainer types
if shap_explainer.explainer_type == 'tree':
    shap_explainer.configure_tree_explainer(
        model_output='probability',  # 'raw', 'probability', 'log-odds'
        approximate=False,  # Use approximate algorithm for speed
        check_additivity=True  # Verify SHAP values sum correctly
    )
elif shap_explainer.explainer_type == 'deep':
    shap_explainer.configure_deep_explainer(
        batch_size=32,
        interim_layer=None,  # Specify layer for attribution
        learning_phase_flags=None
    )
```

### LIME Configuration

```python
lime_explainer = LimeExplainer(
    model_adapter,
    mode='classification',
    training_data=background_sequences,
    discretize_continuous=False,
    sample_around_instance=True,
    kernel_width=25,  # Kernel width for weighting neighbors
    verbose=True
)

# Configure text-specific parameters for sequence data
lime_explainer.configure_text_explainer(
    bow=False,  # Don't use bag-of-words (preserve order)
    split_expression=r'(?<=.)(?=.)',  # Split into individual amino acids
    mask_string='X',  # Character to use for masking
    class_names=['Cytoplasm', 'Nucleus', 'Membrane', '...']
)
```

## Generating Explanations

### Single Sequence Explanations

```python
# Single sequence explanation
sequence = "MKTIIALSYIFCLVFADYKDDDDK"

# SHAP explanation
shap_explanation = shap_explainer.explain(sequence)

print(f"Predicted class: {shap_explanation.prediction}")
print(f"Confidence: {shap_explanation.confidence:.3f}")
print(f"Base value: {shap_explanation.base_value:.3f}")

# Feature attributions
for i, (amino_acid, attribution) in enumerate(zip(sequence, shap_explanation.attributions)):
    print(f"Position {i+1}: {amino_acid} -> {attribution:.4f}")

# LIME explanation
lime_explanation = lime_explainer.explain_instance(
    sequence,
    num_features=len(sequence),  # Explain all positions
    num_samples=5000  # Number of perturbed samples
)

print("LIME feature importance:")
for feature, importance in lime_explanation.as_list():
    print(f"{feature}: {importance:.4f}")
```

### Batch Explanations

```python
# Explain multiple sequences efficiently
sequences = [
    "MKTIIALSYIFCLVFADYKDDDDK",
    "MALWMRLLPLLALLALWGPGPGGA",
    "MVLSEGEWQLVLHVWAKVEADVA"
]

# Batch SHAP explanations
batch_explanations = shap_explainer.explain_batch(sequences)

for i, (seq, explanation) in enumerate(zip(sequences, batch_explanations)):
    print(f"\nSequence {i+1}: {seq[:20]}...")
    print(f"Prediction: {explanation.prediction}")
    print(f"Top 5 important positions:")

    # Get top contributing positions
    top_indices = np.argsort(np.abs(explanation.attributions))[-5:]
    for idx in reversed(top_indices):
        print(f"  Pos {idx+1}: {seq[idx]} -> {explanation.attributions[idx]:.4f}")
```

### Long Sequence Handling

For very long sequences, MAP-ExPLoc performs automatic subsampling and caching:

```python
# Configuration for long sequences
long_sequence_explainer = ShapExplainer(
    model_adapter,
    background_data=background_sequences,
    max_sequence_length=2000,  # Truncate longer sequences
    subsequence_length=500,   # Window size for analysis
    overlap=50,               # Overlap between windows
    aggregation_method='mean' # How to combine window results
)

# Explain very long sequence
long_sequence = "M" + "A" * 3000 + "K"  # 3002 amino acids
explanation = long_sequence_explainer.explain(long_sequence)

# Results include position mapping for original sequence
print(f"Original length: {len(long_sequence)}")
print(f"Analyzed windows: {explanation.num_windows}")
print(f"Aggregated attribution shape: {explanation.attributions.shape}")
```

## Visualization and Interpretation

### Attribution Plots

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_sequence_attributions(sequence, attributions, title="Feature Attributions"):
    """Plot amino acid attributions."""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))

    # Bar plot of attributions
    positions = range(1, len(sequence) + 1)
    colors = ['red' if attr < 0 else 'blue' for attr in attributions]

    ax1.bar(positions, attributions, color=colors, alpha=0.7)
    ax1.set_xlabel('Sequence Position')
    ax1.set_ylabel('SHAP Value')
    ax1.set_title(f'{title} - Attribution Magnitudes')
    ax1.grid(True, alpha=0.3)

    # Sequence with color-coded background
    ax2.scatter(positions, [1] * len(sequence),
               c=attributions, cmap='RdBu_r', s=100, alpha=0.8)

    # Add amino acid labels
    for i, aa in enumerate(sequence):
        ax2.text(i+1, 1, aa, ha='center', va='center', fontweight='bold')

    ax2.set_xlim(0, len(sequence) + 1)
    ax2.set_ylim(0.5, 1.5)
    ax2.set_xlabel('Sequence Position')
    ax2.set_title(f'{title} - Sequence View')
    ax2.set_yticks([])

    plt.tight_layout()
    plt.show()

# Use visualization
explanation = shap_explainer.explain("MKTIIALSYIFCLVFADYKDDDDK")
plot_sequence_attributions(
    "MKTIIALSYIFCLVFADYKDDDDK",
    explanation.attributions,
    "Protein Localization Prediction"
)
```

### Multi-Class Explanations

```python
def explain_all_classes(explainer, sequence):
    """Generate explanations for all possible classes."""

    # Get predictions for all classes
    predictions = explainer.model_adapter.predict_proba([sequence])[0]
    class_names = explainer.model_adapter.class_names

    # Explain top 3 most likely classes
    top_classes = np.argsort(predictions)[-3:]

    fig, axes = plt.subplots(len(top_classes), 1, figsize=(15, 4*len(top_classes)))
    if len(top_classes) == 1:
        axes = [axes]

    for i, class_idx in enumerate(reversed(top_classes)):
        class_name = class_names[class_idx]
        probability = predictions[class_idx]

        # Generate explanation for this specific class
        class_explanation = explainer.explain(sequence, target_class=class_idx)

        # Plot
        ax = axes[i]
        positions = range(1, len(sequence) + 1)
        colors = ['red' if attr < 0 else 'blue' for attr in class_explanation.attributions]

        ax.bar(positions, class_explanation.attributions, color=colors, alpha=0.7)
        ax.set_title(f'{class_name} (p={probability:.3f})')
        ax.set_xlabel('Sequence Position')
        ax.set_ylabel('Attribution')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Generate multi-class explanations
explain_all_classes(shap_explainer, "MKTIIALSYIFCLVFADYKDDDDK")
```

### Comparative Analysis

```python
def compare_explanations(sequence, explainers, explainer_names):
    """Compare explanations from different methods."""

    explanations = []
    for explainer in explainers:
        explanation = explainer.explain(sequence)
        explanations.append(explanation.attributions)

    # Create comparison plot
    fig, axes = plt.subplots(len(explainers), 1, figsize=(15, 4*len(explainers)))
    if len(explainers) == 1:
        axes = [axes]

    positions = range(1, len(sequence) + 1)

    for i, (attributions, name) in enumerate(zip(explanations, explainer_names)):
        ax = axes[i]
        colors = ['red' if attr < 0 else 'blue' for attr in attributions]
        ax.bar(positions, attributions, color=colors, alpha=0.7)
        ax.set_title(f'{name} Attributions')
        ax.set_xlabel('Sequence Position')
        ax.set_ylabel('Attribution')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Correlation analysis
    attribution_df = pd.DataFrame({
        name: attr for name, attr in zip(explainer_names, explanations)
    })

    correlation_matrix = attribution_df.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Attribution Correlation Between Methods')
    plt.show()

    return attribution_df

# Compare SHAP and LIME
shap_exp = ShapExplainer(model_adapter)
lime_exp = LimeExplainer(model_adapter)

comparison = compare_explanations(
    "MKTIIALSYIFCLVFADYKDDDDK",
    [shap_exp, lime_exp],
    ['SHAP', 'LIME']
)
```

## Advanced Explainability Features

### Interaction Effects

```python
# SHAP interaction values (for supported explainers)
if hasattr(shap_explainer, 'shap_interaction_values'):
    interaction_values = shap_explainer.explain_interactions("MKTIIALSYIFCLVFADYKDDDDK")

    # Visualize interaction matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(interaction_values,
                xticklabels=list("MKTIIALSYIFCLVFADYKDDDDK"),
                yticklabels=list("MKTIIALSYIFCLVFADYKDDDDK"),
                cmap='RdBu_r', center=0, annot=True, fmt='.3f')
    plt.title('Amino Acid Interaction Effects')
    plt.xlabel('Position j')
    plt.ylabel('Position i')
    plt.show()
```

### Counterfactual Explanations

```python
def generate_counterfactuals(explainer, sequence, target_class, num_changes=3):
    """Generate counterfactual explanations by minimal sequence modifications."""

    current_prediction = explainer.model_adapter.predict([sequence])[0]
    current_class = explainer.model_adapter.class_names[current_prediction]

    print(f"Original prediction: {current_class}")
    print(f"Target class: {target_class}")

    # Get attribution to guide changes
    explanation = explainer.explain(sequence)
    attributions = explanation.attributions

    # Find positions with strongest negative attribution (for target class)
    change_positions = np.argsort(attributions)[:num_changes]

    # Generate alternatives by changing amino acids
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    counterfactuals = []

    for pos in change_positions:
        original_aa = sequence[pos]
        for new_aa in amino_acids:
            if new_aa != original_aa:
                # Create modified sequence
                modified_seq = list(sequence)
                modified_seq[pos] = new_aa
                modified_seq = ''.join(modified_seq)

                # Check prediction
                pred = explainer.model_adapter.predict([modified_seq])[0]
                pred_class = explainer.model_adapter.class_names[pred]

                if pred_class == target_class:
                    confidence = explainer.model_adapter.predict_proba([modified_seq])[0][pred]
                    counterfactuals.append({
                        'sequence': modified_seq,
                        'change': f'{original_aa}{pos+1}{new_aa}',
                        'confidence': confidence
                    })
                    break  # Found successful change for this position

    return counterfactuals

# Generate counterfactuals
sequence = "MKTIIALSYIFCLVFADYKDDDDK"
counterfactuals = generate_counterfactuals(
    shap_explainer,
    sequence,
    target_class='Nucleus',
    num_changes=3
)

for cf in counterfactuals:
    print(f"Change {cf['change']}: {cf['sequence']} (conf: {cf['confidence']:.3f})")
```

## Performance Optimization

### Caching Strategies

```python
from functools import lru_cache
import pickle
import os

class CachedExplainer:
    """Explainer with result caching."""

    def __init__(self, base_explainer, cache_dir="explanation_cache"):
        self.base_explainer = base_explainer
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_path(self, sequence, method='shap'):
        """Generate cache file path."""
        sequence_hash = hashlib.md5(sequence.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{method}_{sequence_hash}.pkl")

    def explain(self, sequence, use_cache=True):
        """Explain with caching."""
        cache_path = self._get_cache_path(sequence)

        if use_cache and os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        # Generate explanation
        explanation = self.base_explainer.explain(sequence)

        # Cache result
        if use_cache:
            with open(cache_path, 'wb') as f:
                pickle.dump(explanation, f)

        return explanation

    def clear_cache(self):
        """Clear all cached explanations."""
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                os.remove(os.path.join(self.cache_dir, filename))

# Use cached explainer
cached_explainer = CachedExplainer(shap_explainer)
explanation = cached_explainer.explain("MKTIIALSYIFCLVFADYKDDDDK")  # Cached after first run
```

### Parallel Processing

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

def explain_sequence_batch_parallel(explainer, sequences, n_workers=None):
    """Parallel explanation generation."""

    if n_workers is None:
        n_workers = min(mp.cpu_count(), len(sequences))

    # Split sequences into chunks
    chunk_size = len(sequences) // n_workers + 1
    sequence_chunks = [sequences[i:i+chunk_size] for i in range(0, len(sequences), chunk_size)]

    def process_chunk(chunk):
        chunk_results = []
        for seq in chunk:
            explanation = explainer.explain(seq)
            chunk_results.append(explanation)
        return chunk_results

    # Process chunks in parallel
    all_explanations = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_chunk = {executor.submit(process_chunk, chunk): chunk
                          for chunk in sequence_chunks}

        for future in as_completed(future_to_chunk):
            chunk_explanations = future.result()
            all_explanations.extend(chunk_explanations)

    return all_explanations

# Parallel processing example
sequences = ["MKTIIALSYIFCLVFADYKDDDDK"] * 100  # Large batch
explanations = explain_sequence_batch_parallel(shap_explainer, sequences, n_workers=4)
```

## Best Practices and Troubleshooting

### Explanation Quality Assessment

```python
def assess_explanation_quality(explainer, sequences, num_samples=100):
    """Assess explanation quality and consistency."""

    quality_metrics = {
        'consistency': [],
        'completeness': [],
        'stability': []
    }

    sample_sequences = sequences[:num_samples]

    for seq in sample_sequences:
        # Consistency: Do SHAP values sum to prediction difference?
        explanation = explainer.explain(seq)
        base_value = explanation.base_value
        shap_sum = explanation.attributions.sum()
        prediction = explainer.model_adapter.predict_proba([seq])[0][explanation.predicted_class]

        consistency = abs((base_value + shap_sum) - prediction)
        quality_metrics['consistency'].append(consistency)

        # Completeness: Are all positions explained?
        completeness = len([attr for attr in explanation.attributions if attr != 0]) / len(seq)
        quality_metrics['completeness'].append(completeness)

        # Stability: Small perturbations should give similar explanations
        perturbed_seq = seq[:-1] + ('A' if seq[-1] != 'A' else 'G')
        perturbed_explanation = explainer.explain(perturbed_seq)

        if len(explanation.attributions) == len(perturbed_explanation.attributions):
            correlation = np.corrcoef(explanation.attributions, perturbed_explanation.attributions)[0, 1]
            quality_metrics['stability'].append(abs(correlation))
        else:
            quality_metrics['stability'].append(0.0)

    # Summary statistics
    summary = {}
    for metric, values in quality_metrics.items():
        summary[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'median': np.median(values)
        }

    return summary

# Assess explanation quality
quality = assess_explanation_quality(shap_explainer, test_sequences, num_samples=50)
for metric, stats in quality.items():
    print(f"{metric.capitalize()}: {stats['mean']:.3f} ± {stats['std']:.3f}")
```

### Common Issues and Solutions

```python
# Issue 1: Slow explanation generation
# Solution: Use TreeExplainer for tree models, reduce background size
fast_explainer = ShapTreeExplainer(tree_model_adapter, background_size=50)

# Issue 2: Memory issues with long sequences
# Solution: Use sequence chunking
chunked_explainer = ShapExplainer(
    model_adapter,
    max_sequence_length=1000,
    subsequence_length=200
)

# Issue 3: Inconsistent explanations
# Solution: Increase background data diversity and size
stable_explainer = ShapExplainer(
    model_adapter,
    background_data=diverse_background,
    background_size=200
)

# Issue 4: Poor explanation quality for rare classes
# Solution: Use stratified background sampling
def create_class_balanced_background(data, target_class, size=100):
    class_data = data[data['localization'] == target_class]
    other_data = data[data['localization'] != target_class]

    class_samples = min(size // 2, len(class_data))
    other_samples = size - class_samples

    background = []
    if class_samples > 0:
        background.extend(class_data.sample(n=class_samples)['sequence'].tolist())
    if other_samples > 0:
        background.extend(other_data.sample(n=other_samples)['sequence'].tolist())

    return background
```

## Integration with Analysis Pipelines

### Automated Explanation Reports

```python
def generate_explanation_report(explainer, sequences, output_file="explanation_report.html"):
    """Generate comprehensive explanation report."""

    explanations = []
    for seq in sequences:
        explanation = explainer.explain(seq)
        explanations.append({
            'sequence': seq,
            'prediction': explainer.model_adapter.class_names[explanation.predicted_class],
            'confidence': explanation.confidence,
            'top_contributions': sorted(
                [(i, aa, attr) for i, (aa, attr) in enumerate(zip(seq, explanation.attributions))],
                key=lambda x: abs(x[2]), reverse=True
            )[:5]
        })

    # Generate HTML report (simplified example)
    html_content = """
    <html>
    <head><title>Explanation Report</title></head>
    <body>
    <h1>Protein Localization Explanation Report</h1>
    """

    for i, exp in enumerate(explanations):
        html_content += f"""
        <h2>Sequence {i+1}</h2>
        <p><strong>Sequence:</strong> {exp['sequence'][:50]}{'...' if len(exp['sequence']) > 50 else ''}</p>
        <p><strong>Prediction:</strong> {exp['prediction']} (confidence: {exp['confidence']:.3f})</p>
        <p><strong>Top Contributing Positions:</strong></p>
        <ul>
        """

        for pos, aa, contrib in exp['top_contributions']:
            html_content += f"<li>Position {pos+1}: {aa} -> {contrib:.4f}</li>"

        html_content += "</ul>"

    html_content += "</body></html>"

    with open(output_file, 'w') as f:
        f.write(html_content)

    print(f"Report generated: {output_file}")

# Generate report
generate_explanation_report(shap_explainer, test_sequences[:10])
```

This comprehensive guide provides everything needed to effectively use MAP-ExPLoc's explainability features. For additional examples and advanced use cases, refer to the tutorials and API documentation.
