# Quickstart Guide

Get up and running with MAP-ExPLoc in minutes. This guide will walk you through installation, basic usage, and your first protein localization predictions.

## Installation

### Quick Install

Install MAP-ExPLoc using pip:

```bash
pip install mapexploc
```

### Verify Installation

Confirm the installation was successful:

```bash
mapexploc --version
mapexploc --help
```

## Basic Usage

### 1. Quick Prediction

Predict the subcellular localization of a single protein sequence:

```bash
# Single sequence prediction
mapexploc predict MKTIIALSYIFCLVFADYKDDDDK

# With detailed output
mapexploc predict MKTIIALSYIFCLVFADYKDDDDK --verbose
```

Expected output:

```
Protein: MKTIIALSYIFCLVFADYKDDDDK
Predicted Localization: Cytoplasm
Confidence: 0.87
```

### 2. Batch Predictions

Process multiple sequences from a file:

```bash
# From FASTA file
mapexploc predict-batch sequences.fasta --output predictions.csv

# From CSV file with sequence column
mapexploc predict-batch proteins.csv --sequence-column seq --output results.csv
```

### 3. Model Training

Train a custom model with your data:

```bash
# Using default configuration
mapexploc train --config config/default.yml

# Custom training parameters
mapexploc train --data data/custom_proteins.csv --model rf --output models/custom_model.pkl
```

### 4. Explainable Predictions

Generate explanations for model predictions:

```bash
# Single sequence explanation
mapexploc explain MKTIIALSYIFCLVFADYKDDDDK --method shap

# Batch explanations
mapexploc explain-batch sequences.fasta --method lime --output explanations/
```

## Python API Usage

### Basic Prediction

```python
from mapexploc import Predictor

# Initialize predictor
predictor = Predictor()

# Single sequence prediction
sequence = "MKTIIALSYIFCLVFADYKDDDDK"
result = predictor.predict(sequence)

print(f"Localization: {result.localization}")
print(f"Confidence: {result.confidence:.3f}")
```

### Batch Processing

```python
import pandas as pd
from mapexploc import BatchPredictor

# Initialize batch predictor
batch_predictor = BatchPredictor()

# Load sequences
sequences = [
    "MKTIIALSYIFCLVFADYKDDDDK",
    "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNGGHFLRILPDGTVDGTRDRSDQHIQLQLSAESVGEVYIKSTETGQYLAMDTSGLLYGSQTPNEECLFLERLEENHYNTYTSKKHAEKNWFVGLKKNGSCKRGPRTHYGQKAILFLPLPV",
    "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEHVKLVNELTEFAKTCVADESHAGCEKSLHTLFGDELCKVASLRETYGDMADCCEKQEPERNECFLSHKDDSPDLPKLKPDPNTLCDEFKADEKKFWGKYLYEIARRHPYFYAPELLYYANKYNGVFQECCQAEDKGACLLPKIETMREKVLASSARQRLRCASIQKFGERALKAWSVARLSQKFPKAEFVEVTKLVTDLTKVHKECCHGDLLECADDRADLAKYICDNQDTISSKLKECCDKPVNGFNVDSFADRVCACSQRTRRTRKQLQRSLELNRRQVGIPIRQQHRQQHGQQLRPPDHLPKSGQ"
]

# Batch prediction
results = batch_predictor.predict_batch(sequences)

# Display results
for i, result in enumerate(results):
    print(f"Sequence {i+1}: {result.localization} (conf: {result.confidence:.3f})")
```

### Training Custom Models

```python
from mapexploc import ModelTrainer
import pandas as pd

# Load training data
data = pd.read_csv('training_proteins.csv')

# Initialize trainer
trainer = ModelTrainer()

# Train model
model = trainer.train(
    sequences=data['sequence'],
    labels=data['localization'],
    model_type='random_forest',
    validation_split=0.2
)

# Save model
model.save('models/custom_model.pkl')

# Evaluate model
metrics = trainer.evaluate(model)
print(f"F1-macro: {metrics.f1_macro:.3f}")
print(f"Accuracy: {metrics.accuracy:.3f}")
```

## Configuration

### Default Configuration

MAP-ExPLoc uses sensible defaults, but you can customize behavior:

```python
from mapexploc import Config

# Load default configuration
config = Config.load_default()

# Modify settings
config.model.type = 'xgboost'
config.preprocessing.min_length = 10
config.prediction.confidence_threshold = 0.8

# Save custom configuration
config.save('config/custom.yml')
```

### Configuration File Example

Create a custom configuration file (`config/quickstart.yml`):

```yaml
model:
  type: random_forest
  parameters:
    n_estimators: 100
    max_depth: 20

preprocessing:
  min_length: 5
  max_length: 5000
  features:
    - amino_acid_composition
    - dipeptide_composition
    - molecular_weight
    - gravy_score

prediction:
  confidence_threshold: 0.7
  output_probabilities: true

training:
  validation_split: 0.2
  cross_validation: 5
  scoring: f1_macro
```

Use the custom configuration:

```bash
mapexploc train --config config/quickstart.yml
mapexploc predict SEQUENCE --config config/quickstart.yml
```

## Common Use Cases

### Research Applications

```bash
# Analyze proteome-wide localization
mapexploc predict-batch human_proteome.fasta --output human_localizations.csv

# Compare localization patterns
mapexploc compare-species species1.fasta species2.fasta --output comparison.html

# Generate publication-ready figures
mapexploc visualize results.csv --plot-type confusion_matrix --save figures/
```

### Biotech Applications

```python
from mapexploc import PipelineRunner

# High-throughput screening pipeline
pipeline = PipelineRunner()

# Process experimental results
results = pipeline.run_analysis(
    sequences_file='experimental_sequences.csv',
    reference_file='known_localizations.csv',
    output_dir='analysis_results/'
)

# Generate report
pipeline.generate_report(results, 'screening_report.html')
```

### Educational Use

```python
# Interactive learning mode
from mapexploc import InteractiveTutor

tutor = InteractiveTutor()

# Step-by-step prediction explanation
tutor.explain_prediction("MKTIIALSYIFCLVFADYKDDDDK")

# Feature importance visualization
tutor.show_features("SEQUENCE")

# Compare different models
tutor.model_comparison(["random_forest", "svm", "neural_network"])
```

## Troubleshooting

### Common Issues

**Installation Problems**:

```bash
# Update pip and try again
pip install --upgrade pip
pip install mapexploc

# Use conda if pip fails
conda install -c bioconda mapexploc
```

**Memory Issues**:

```python
# Use batch processing for large datasets
from mapexploc import BatchPredictor

predictor = BatchPredictor(batch_size=100)  # Process 100 sequences at a time
results = predictor.predict_file('large_dataset.fasta')
```

**Performance Optimization**:

```bash
# Use parallel processing
mapexploc predict-batch sequences.fasta --jobs 4 --output results.csv

# Enable GPU acceleration (if available)
mapexploc predict sequences.fasta --device gpu
```

### Getting Help

- **Documentation**: [Full documentation](index.md)
- **API Reference**: [API documentation](api.md)
- **Examples**: Check the `examples/` directory
- **Issues**: Report bugs on GitHub
- **Community**: Join our discussion forum

## Next Steps

Now that you've completed the quickstart:

1. **[Tutorial Series](tutorials/)**: Deep dive into specific techniques
2. **[API Documentation](api.md)**: Complete reference for all functions
3. **[Advanced Usage](explainer-guide.md)**: Explore advanced features
4. **[Model Training](tutorials/rf.md)**: Train custom models
5. **[Interpretability](tutorials/shap.md)**: Understand model decisions

## Example Workflows

### Complete Analysis Workflow

```bash
# 1. Prepare data
mapexploc preprocess raw_sequences.fasta --output processed_sequences.csv

# 2. Train model
mapexploc train --data processed_sequences.csv --output models/my_model.pkl

# 3. Make predictions
mapexploc predict-batch test_sequences.fasta --model models/my_model.pkl --output predictions.csv

# 4. Generate explanations
mapexploc explain-batch test_sequences.fasta --model models/my_model.pkl --output explanations/

# 5. Create visualizations
mapexploc visualize predictions.csv --output figures/
```

### Comparative Study

```python
from mapexploc import ComparativeAnalyzer

analyzer = ComparativeAnalyzer()

# Compare different organisms
results = analyzer.compare_organisms([
    'human_proteins.fasta',
    'yeast_proteins.fasta',
    'ecoli_proteins.fasta'
])

# Generate comparative report
analyzer.generate_report(results, 'comparative_analysis.html')

# Statistical analysis
stats = analyzer.statistical_analysis(results)
print(f"Significant differences: {stats.significant_pairs}")
```

You're now ready to explore the full capabilities of MAP-ExPLoc. For detailed information on any topic, refer to the comprehensive documentation and tutorials.
