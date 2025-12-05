# Preprocessing Tutorial

This tutorial demonstrates how to extract and process protein data from Swiss-Prot databases using MAP-ExPLoc's preprocessing module.

## Overview

The preprocessing module handles the extraction of protein sequences and their subcellular localization annotations from Swiss-Prot DAT files. It includes data cleaning, localization mapping, and filtering to prepare high-quality datasets for machine learning.

## Prerequisites

- BioPython library for Swiss-Prot parsing
- Swiss-Prot DAT file (downloadable from UniProt)
- Basic understanding of protein sequences and subcellular localization

## Installation

Ensure you have the required dependencies:

```bash
pip install biopython
# OR install complete requirements
pip install -r requirements.txt
```

## Basic Usage

### Importing the Module

```python
from mapexploc.preprocessing import extract_protein_data, ALLOWED_LOCS
import pandas as pd
```

### Extracting Protein Data

The main function `extract_protein_data` processes Swiss-Prot DAT files:

```python
# Extract protein data from Swiss-Prot DAT file
protein_data = extract_protein_data("uniprot_sprot.dat")

# Convert to DataFrame for analysis
df = pd.DataFrame(protein_data)
print(f"Extracted {len(df)} protein records")
print(df.head())
```

### Output Format

The function returns a list of dictionaries with the following structure:

```python
[
    {
        "entry_name": "PROTEIN_HUMAN",
        "localization": "Nucleus",
        "sequence": "MKTIIALSYIFCLVFADYKDDDDK..."
    },
    # ... more proteins
]
```

## Data Cleaning Process

### Localization Mapping

The preprocessing module includes a comprehensive mapping system for subcellular localizations:

```python
# View allowed localizations
print("Allowed subcellular localizations:")
for loc in sorted(ALLOWED_LOCS):
    print(f"  - {loc}")
```

The system maps various synonyms to standardized terms:

- "cell membrane", "plasma membrane" → "Membrane"
- "cytosol" → "Cytoplasm"
- "endoplasmic reticulum", "er" → "Endoplasmic Reticulum"

### Quality Filtering

The module automatically filters out low-quality entries:

1. **Evidence-based filtering**: Removes entries with non-experimental evidence
2. **Multi-compartment filtering**: Excludes proteins with ambiguous localizations
3. **Sequence validation**: Ensures valid amino acid sequences

### Advanced Usage

#### Custom Filtering

```python
# Process data with custom filtering
def filter_by_length(protein_data, min_length=50, max_length=2000):
    """Filter proteins by sequence length."""
    filtered = []
    for protein in protein_data:
        seq_len = len(protein['sequence'])
        if min_length <= seq_len <= max_length:
            filtered.append(protein)
    return filtered

# Extract and filter
raw_data = extract_protein_data("uniprot_sprot.dat")
filtered_data = filter_by_length(raw_data)

print(f"Original: {len(raw_data)} proteins")
print(f"Filtered: {len(filtered_data)} proteins")
```

#### Localization Distribution Analysis

```python
import matplotlib.pyplot as plt

# Analyze localization distribution
df = pd.DataFrame(protein_data)
loc_counts = df['localization'].value_counts()

# Create visualization
plt.figure(figsize=(12, 6))
loc_counts.plot(kind='bar')
plt.title('Distribution of Subcellular Localizations')
plt.xlabel('Localization')
plt.ylabel('Number of Proteins')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("Localization statistics:")
print(loc_counts)
```

## Working with Large Datasets

### Memory-Efficient Processing

For large Swiss-Prot files, consider processing in batches:

```python
def process_in_batches(dat_file, batch_size=10000):
    """Process Swiss-Prot file in batches."""
    from Bio import SwissProt

    batch = []
    all_proteins = []

    with open(dat_file, 'r') as handle:
        for i, record in enumerate(SwissProt.parse(handle)):
            # Process record (simplified version of extract_protein_data logic)
            if record.sequence:
                batch.append({
                    'entry_name': record.entry_name,
                    'sequence': record.sequence
                })

            # Process batch when full
            if len(batch) >= batch_size:
                print(f"Processing batch {i//batch_size + 1}")
                all_proteins.extend(batch)
                batch = []

    # Process remaining records
    if batch:
        all_proteins.extend(batch)

    return all_proteins
```

### Data Export

```python
# Save processed data for later use
df = pd.DataFrame(protein_data)

# Export to CSV
df.to_csv('processed_proteins.csv', index=False)

# Export to different formats
df.to_parquet('processed_proteins.parquet')  # More efficient for large datasets
df.to_pickle('processed_proteins.pkl')       # Preserves exact data types
```

## Integration with Feature Engineering

The preprocessing output integrates seamlessly with the feature engineering pipeline:

```python
from mapexploc.features import build_feature_matrix
from mapexploc.preprocessing import extract_protein_data

# Complete preprocessing to feature engineering workflow
protein_data = extract_protein_data("uniprot_sprot.dat")

# Create FASTA file for feature engineering
with open("sequences.fasta", "w") as f:
    for protein in protein_data:
        f.write(f">{protein['entry_name']}|{protein['localization']}\n")
        f.write(f"{protein['sequence']}\n")

# Create annotations file
df = pd.DataFrame(protein_data)
df[['entry_name', 'localization']].to_csv('annotations.csv', index=False)

# Build feature matrix
features = build_feature_matrix("sequences.fasta", "annotations.csv")
```

## Common Issues and Solutions

### Issue: BioPython Import Error

```python
# Solution: Install BioPython
# pip install biopython
```

### Issue: Empty Results

```python
# Check if DAT file is valid and contains SUBCELLULAR LOCATION comments
def validate_dat_file(dat_file, sample_size=10):
    """Validate DAT file format and content."""
    from Bio import SwissProt

    with open(dat_file, 'r') as handle:
        for i, record in enumerate(SwissProt.parse(handle)):
            if i >= sample_size:
                break

            print(f"Entry {i+1}: {record.entry_name}")
            print(f"Sequence length: {len(record.sequence)}")
            print(f"Comments: {len(record.comments)}")

            # Check for subcellular location comments
            subcell_comments = [c for c in record.comments
                              if c.startswith("SUBCELLULAR LOCATION")]
            print(f"Subcellular location comments: {len(subcell_comments)}")
            print("-" * 40)
```

### Issue: Memory Problems with Large Files

```python
# Use streaming approach for very large files
def count_proteins(dat_file):
    """Count proteins without loading all into memory."""
    from Bio import SwissProt

    count = 0
    with open(dat_file, 'r') as handle:
        for record in SwissProt.parse(handle):
            count += 1
            if count % 10000 == 0:
                print(f"Processed {count} records...")

    return count
```

## Best Practices

1. **Validate Input Files**: Always check DAT file integrity before processing
2. **Monitor Memory Usage**: Use batch processing for large datasets
3. **Save Intermediate Results**: Export processed data to avoid recomputation
4. **Document Filtering Criteria**: Keep track of quality filters applied
5. **Version Control Data**: Track Swiss-Prot release versions used

## Next Steps

After preprocessing, you can proceed to:

- **[Feature Engineering Tutorial](features.md)**: Generate physicochemical features
- **[Exploratory Analysis Tutorial](exploratory.md)**: Analyze the processed dataset
- **[Model Training Tutorials](knn.md)**: Train machine learning models
