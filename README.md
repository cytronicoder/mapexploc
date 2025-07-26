# Explainable Subcellular Localization Predictor

This is a lightweight, interpretable machine-learning pipeline designed to predict protein subcellular localization from primary amino-acid sequences and reveal the key sequence motifs driving each classification. We combined classical feature engineering with a Random Forest classifier and Shapley Additive exPlanations (SHAP) to generate transparent, biologically meaningful insights.

> [!NOTE]  
> This project has won the [2025 YBS Student Challenge](https://www.iscb.org/ybs2025/programme-agenda/student-challenge) for its innovative approach to AI in bioinformatics.

## Features

- Random Forest classifier trained on curated UniProtKB/Swiss-Prot sequences across 16 compartments
- SHAP values assign per-feature contributions for each protein prediction
- Minimal dependencies and fast inference on standard workstations
- Easily swap in new classifiers or add custom features

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/cytronicoder/explainable-localization-predictor.git
   cd explainable-localization-predictor
   ```

2. Create and activate a Conda environment (recommended):

   ```bash
   conda env create --file environment.yml
   conda activate eslp
   ```

3. To update or add dependencies, modify `environment.yml` and run:

   ```bash
   conda env update --file environment.yml --prune
   ```

## Requirements

- Python 3.8 or higher
- scikit-learn
- SHAP
- pandas
- numpy
- matplotlib (for optional plotting)

## Feature Engineering

- Amino-acid composition (20 dimensions)
- Dipeptide frequencies (400 dimensions)
- Physicochemical properties (molecular weight, isoelectric point)
- Entropy measures for disorder prediction

## Results

Sample performance on the held-out test set:

- Overall weighted F1: 0.88
- Compartment AUCs: 0.93–0.97
- Key biological insights:
  - Hydrophobicity (GRAVY) drives membrane vs. soluble distinction
  - Isoelectric point correlates with cytosolic adaptation
  - Sequence length penalties reveal targeting constraints

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please open issues for bug reports or feature requests, and submit pull requests against the `main` branch.
