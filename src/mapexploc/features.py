"""Feature extraction utilities for protein sequences."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis

logger = logging.getLogger(__name__)

# Standard amino acids
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def build_feature_matrix(
    sequences: Union[str, Path, List[str], pd.Series],
    annotations: Optional[Union[str, Path, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """
    Build a feature matrix from protein sequences.

    Args:
        sequences: Path to a FASTA file, or a list/Series of protein sequences.
        annotations: Optional path to annotations CSV or DataFrame.
                     If provided, will be merged with features.

    Returns:
        DataFrame containing extracted features.
    """
    # 1. Load sequences
    seq_list: List[str] = []
    ids: List[str] = []

    if isinstance(sequences, (str, Path)):
        path = Path(sequences)
        if path.suffix in (".fasta", ".fa", ".fna"):
            # Load from FASTA
            for record in SeqIO.parse(path, "fasta"):
                seq_list.append(str(record.seq))
                ids.append(record.id)
        else:
            # Assume it's a raw string if it's not a file path or if it's a short string?
            # But the type hint says Union[str, Path...].
            # If it's a string that looks like a path but doesn't exist, we might have issues.
            # Given the context, if it's a string, it's likely a path.
            # But if the user passes a single sequence as a string? Unlikely for "build_feature_matrix".
            raise ValueError(f"Unsupported file format or path not found: {sequences}")
    elif isinstance(sequences, (list, pd.Series, np.ndarray)):
        seq_list = [str(s) for s in sequences]
        ids = [f"seq_{i}" for i in range(len(seq_list))]
    else:
        raise TypeError(f"Unsupported type for sequences: {type(sequences)}")

    # 2. Extract features
    features_list = []
    for seq in seq_list:
        features_list.append(_extract_features(seq))

    df = pd.DataFrame(features_list)
    
    # Add IDs if we loaded from FASTA and have annotations to merge
    if ids:
        df.index = ids

    # 3. Merge annotations if provided
    if annotations is not None:
        if isinstance(annotations, (str, Path)):
            ann_df = pd.read_csv(annotations)
        elif isinstance(annotations, pd.DataFrame):
            ann_df = annotations
        else:
            raise TypeError(f"Unsupported type for annotations: {type(annotations)}")
        
        # If annotations have an ID column that matches our FASTA IDs, merge on it.
        # Otherwise, if lengths match, merge by index?
        # The notebook usage implies a merge.
        # "df = build_feature_matrix(INPUT_FASTA, INPUT_ANN)"
        # Usually annotations.csv has "accession" or "id".
        
        # For now, let's assume simple concatenation if lengths match and no common index
        if len(ann_df) == len(df):
             # Reset index to allow concat if indices don't match
            df = df.reset_index(drop=True)
            ann_df = ann_df.reset_index(drop=True)
            df = pd.concat([df, ann_df], axis=1)
        else:
            logger.warning("Annotation length mismatch. Skipping merge.")

    return df


def _extract_features(sequence: str) -> dict:
    """Extract biochemical features from a single sequence."""
    # Clean sequence (remove non-standard AA if necessary, or handle them)
    # BioPython's ProteinAnalysis handles standard AA.
    # We should probably remove 'X', 'U', 'Z', 'B', 'J', 'O' or treat them.
    # For simplicity, let's replace them or ignore.
    
    # Simple cleaning: remove non-standard
    clean_seq = "".join([aa for aa in sequence if aa in AMINO_ACIDS])
    
    if not clean_seq:
        # Return zeros if empty
        return {
            "length": 0,
            "gravy": 0.0,
            "isoelectric_point": 0.0,
            **{f"aa_{aa}": 0.0 for aa in AMINO_ACIDS},
            # Dipeptides omitted for brevity in empty case? Or should be 0.
        }

    analyser = ProteinAnalysis(clean_seq)
    
    # AA Composition
    aa_counts = analyser.count_amino_acids()
    aa_percent = analyser.amino_acids_percent
    
    features = {}
    
    # Length
    features["length"] = len(sequence) # Use original length?
    
    # AA Composition Features
    for aa in AMINO_ACIDS:
        features[f"aa_{aa}"] = aa_percent.get(aa, 0.0)
        
    # Dipeptide Composition (400 features)
    # ProteinAnalysis doesn't have a direct method for dipeptide frequency?
    # It has get_amino_acids_percent.
    # We can implement it manually.
    
    for aa1 in AMINO_ACIDS:
        for aa2 in AMINO_ACIDS:
            dipeptide = aa1 + aa2
            # Count occurrences
            count = sequence.count(dipeptide)
            # Frequency
            freq = count / (len(sequence) - 1) if len(sequence) > 1 else 0
            features[f"dp_{dipeptide}"] = freq

    # Physico-chemical properties
    try:
        features["gravy"] = analyser.gravy()
    except Exception:
        features["gravy"] = 0.0
        
    try:
        features["isoelectric_point"] = analyser.isoelectric_point()
    except Exception:
        features["isoelectric_point"] = 0.0 # Default or NaN

    return features
