"""Feature engineering convenience wrappers."""

from __future__ import annotations

import logging
from typing import Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_feature_matrix(fasta_file: str, annotations_file: str) -> pd.DataFrame:
    """Build feature matrix from FASTA sequences and annotations.

    Args:
        fasta_file: Path to FASTA file containing sequences
        annotations_file: Path to CSV file containing annotations

    Returns:
        DataFrame with features and metadata columns
    """
    try:
        from Bio import SeqIO
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
    except ImportError:
        logger.error(
            "BioPython is required but not installed. Install with: pip install biopython"
        )
        raise ImportError("BioPython is required for feature extraction")

    # Load annotations if provided
    ann_map = {}
    if annotations_file is not None:
        annotations = pd.read_csv(annotations_file)
        # Create a mapping from entry_name to localization
        ann_map = dict(zip(annotations["entry_name"], annotations["localization"]))

    # Process FASTA sequences
    sequences_data = []
    feature_data = []

    logger.info("Processing FASTA sequences from %s", fasta_file)

    with open(fasta_file, "r", encoding="utf-8") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            # Extract entry name from header (format: >ENTRY_NAME|LOCALIZATION)
            entry_parts = record.id.split("|")
            entry_name = entry_parts[0]

            # Get localization from annotations or header
            localization = ann_map.get(
                entry_name, entry_parts[1] if len(entry_parts) > 1 else "Other"
            )

            sequence = str(record.seq)

            # Skip short sequences
            if len(sequence) < 10:
                continue

            # Calculate protein features using ProtParam
            try:
                analysis = ProteinAnalysis(sequence)

                # Basic properties
                molecular_weight = analysis.molecular_weight()
                gravy = analysis.gravy()  # Hydropathy
                aromaticity = analysis.aromaticity()
                instability_index = analysis.instability_index()
                isoelectric_point = analysis.isoelectric_point()

                # Amino acid composition (20 standard amino acids)
                aa_composition = analysis.get_amino_acids_percent()

                # Secondary structure prediction
                sec_struct = analysis.secondary_structure_fraction()
                helix, turn, sheet = sec_struct

                # Sequence properties
                sequence_length = len(sequence)

                # Build feature vector
                features = {
                    "entry_name": entry_name,
                    "localization": localization,
                    "sequence_length": sequence_length,
                    "molecular_weight": molecular_weight,
                    "gravy": gravy,
                    "aromaticity": aromaticity,
                    "instability_index": instability_index,
                    "isoelectric_point": isoelectric_point,
                    "helix_fraction": helix,
                    "turn_fraction": turn,
                    "sheet_fraction": sheet,
                }

                # Add amino acid composition
                amino_acids = "ACDEFGHIKLMNPQRSTVWY"
                for aa in amino_acids:
                    features[f"aa_{aa}"] = aa_composition.get(aa, 0.0)

                feature_data.append(features)

            except Exception as e:
                logger.warning("Skipping sequence %s due to error: %s", entry_name, e)
                continue

    # Convert to DataFrame
    df = pd.DataFrame(feature_data)

    logger.info("Generated feature matrix with shape %s", df.shape)
    logger.info("Features: %d sequences processed", len(df))

    return df


__all__ = ["build_feature_matrix"]
