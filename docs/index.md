# MAP-ExPLoc Documentation

Model-Agnostic Pipeline for Explainable Protein Subcellular Localization

## Overview

MAP-ExPLoc is a comprehensive machine learning pipeline designed for predicting protein subcellular localization from amino acid sequences. The package provides interpretable predictions through SHAP (SHapley Additive exPlanations) analysis and supports multiple machine learning algorithms optimized for protein classification tasks.

### Key Capabilities

- **Swiss-Prot Data Processing**: Automated extraction and cleaning of protein data from Swiss-Prot databases
- **Feature Engineering**: Generation of 31 physicochemical features from protein sequences using BioPython
- **Machine Learning Models**: k-Nearest Neighbors and Random Forest classifiers with automated hyperparameter optimization
- **Explainable AI**: Comprehensive SHAP analysis for model interpretation and feature importance
- **Production Deployment**: REST API and command-line interfaces for integration into larger systems
- **Evaluation Suite**: Complete model performance analysis with visualization tools

### Architecture

The pipeline follows a modular architecture with distinct components:

1. **Preprocessing Module**: Swiss-Prot data extraction and localization mapping
2. **Feature Engineering**: Physicochemical property calculation from sequences
3. **Model Training**: Automated hyperparameter tuning and cross-validation
4. **Explainability**: SHAP analysis and visualization generation
5. **Deployment**: API and CLI interfaces for prediction and explanation services

### Target Audience

- **Bioinformatics Researchers**: Scientists studying protein localization and function
- **Machine Learning Engineers**: Practitioners working on biological sequence analysis
- **Software Developers**: Engineers integrating protein analysis into applications
- **Students and Educators**: Learning about explainable AI in bioinformatics

## Quick Start

### Installation

Install the complete package with all features:

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from mapexploc.preprocessing import extract_protein_data
from mapexploc.features import build_feature_matrix
from mapexploc.models.rf import train_random_forest

# Extract protein data
proteins = extract_protein_data("uniprot_sprot.dat")

# Build feature matrix
features = build_feature_matrix("sequences.fasta", "annotations.csv")

# Train model
X = features.drop(['entry_name', 'localization'], axis=1)
y = features['localization']
model_result = train_random_forest(X, y)
```

### Command Line Interface

```bash
# Train a model
mapexploc train --config config/default.yml

# Make predictions
mapexploc predict MKTIIALSYIFCLVFADYKDDDDK

# Generate explanations
mapexploc explain MKTIIALSYIFCLVFADYKDDDDK
```

## Documentation Structure

- **[Quick Start Guide](quickstart.md)**: Get started with MAP-ExPLoc in minutes
- **[API Reference](api.md)**: Complete function and class documentation
- **[Tutorials](tutorials/)**: Step-by-step guides for each component
- **[Adapter Guide](adapter-guide.md)**: Custom model integration
- **[Explainer Guide](explainer-guide.md)**: SHAP analysis and interpretation
- **[UI Documentation](ui.md)**: Web interface usage

## Tutorials

- **[Preprocessing](tutorials/preprocessing.md)**: Swiss-Prot data extraction and cleaning
- **[Exploratory Analysis](tutorials/exploratory.md)**: Data exploration and visualization
- **[Feature Engineering](tutorials/features.md)**: Physicochemical feature calculation
- **[k-NN Classification](tutorials/knn.md)**: k-Nearest Neighbors model training
- **[Random Forest](tutorials/rf.md)**: Random Forest model with SMOTE
- **[SHAP Analysis](tutorials/shap.md)**: Model explainability and interpretation

## Contributing

We welcome contributions from the community. Please see our contributing guidelines and code of conduct for more information on how to get involved.

## Support

- **GitHub Issues**: Report bugs and request features
- **Documentation**: Comprehensive guides and API reference
- **Examples**: Complete workflow demonstrations in the notebooks/ directory
