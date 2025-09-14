"""Sequence preprocessing and feature engineering."""

from __future__ import annotations

import logging
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Allowed localizations mapping (simplified terms)
ALLOWED_LOCS = {
    "Cell membrane",
    "Cell wall",
    "Chloroplast",
    "Cytoplasm",
    "Endoplasmic reticulum",
    "Extracellular",
    "Golgi apparatus",
    "Lysosome",
    "Membrane",
    "Mitochondrion",
    "Nucleus",
    "Periplasm",
    "Peroxisome",
    "Ribosome",
    "Secreted",
    "Vacuole",
    "Cell Surface",
    "Endoplasmic Reticulum",
    "Golgi Apparatus",
    "Plastid",
    "Virion",
    "Other",
}

# Synonym mapping for localization normalization
_SYNONYM_MAP = {
    "cell membrane": "Membrane",
    "plasma membrane": "Membrane",
    "cytoplasmic membrane": "Membrane",
    "cell wall": "Cell Surface",
    "cell surface": "Cell Surface",
    "chloroplast": "Plastid",
    "plastid": "Plastid",
    "cytoplasm": "Cytoplasm",
    "cytosol": "Cytoplasm",
    "endoplasmic reticulum": "Endoplasmic Reticulum",
    "er": "Endoplasmic Reticulum",
    "extracellular": "Secreted",
    "secreted": "Secreted",
    "golgi apparatus": "Golgi Apparatus",
    "golgi": "Golgi Apparatus",
    "lysosome": "Lysosome",
    "mitochondrion": "Mitochondrion",
    "mitochondria": "Mitochondrion",
    "nucleus": "Nucleus",
    "nuclear": "Nucleus",
    "periplasm": "Periplasm",
    "periplasmic": "Periplasm",
    "peroxisome": "Peroxisome",
    "ribosome": "Other",
    "ribosomal": "Other",
    "vacuole": "Vacuole",
    "vacuolar": "Vacuole",
    "virion": "Virion",
}


def _clean_and_primary(subcell_locs: List[str]) -> str:
    """Extract primary localization from Swiss-Prot subcellular location entries.

    This function:
    1. Filters out locations with non-experimental evidence codes
    2. Maps synonyms to standardized terms
    3. Returns the first valid location or 'Other' if none found
    4. Excludes multi-compartment entries (contains semicolon or comma)
    """
    if not subcell_locs:
        return "Other"

    # Join all locations and clean
    text = " ".join(subcell_locs).lower()

    # Skip multi-compartment entries
    if ";" in text or "," in text:
        return "Other"

    # Skip entries with non-experimental evidence
    skip_patterns = [
        "by similarity",
        "probable",
        "potential",
        "predicted",
        "inferred",
        "putative",
        "expected",
    ]
    if any(pattern in text for pattern in skip_patterns):
        return "Other"

    # Extract location name (before any parentheses or additional info)
    location = re.split(r"[({]", text)[0].strip()

    # Apply synonym mapping
    if location in _SYNONYM_MAP:
        location = _SYNONYM_MAP[location]
    else:
        # Capitalize first letter of each word for consistency
        location = " ".join(word.capitalize() for word in location.split())

    # Return if it's in allowed locations, otherwise 'Other'
    return location if location in ALLOWED_LOCS else "Other"


def extract_protein_data(dat_file_path: str) -> List[Dict[str, Any]]:
    """Extract protein data from Swiss-Prot DAT file.

    Returns list of dictionaries with keys:
    - entry_name: protein entry name
    - localization: cleaned primary subcellular localization
    - sequence: amino acid sequence
    """
    try:
        from Bio import SwissProt
    except ImportError:
        logger.error(
            "BioPython is required but not installed. Install with: pip install biopython"
        )
        raise ImportError("BioPython is required for Swiss-Prot parsing")

    protein_data = []

    logger.info("Extracting protein data from %s", dat_file_path)

    with open(dat_file_path, "r", encoding="utf-8") as handle:
        for record in SwissProt.parse(handle):
            # Extract subcellular localization
            subcell_locs = []
            for comment in record.comments:
                if comment.startswith("SUBCELLULAR LOCATION"):
                    subcell_locs.append(comment)

            # Clean and get primary localization
            primary_loc = _clean_and_primary(subcell_locs)

            # Skip if no valid sequence or localization
            if not record.sequence or primary_loc == "Other":
                continue

            protein_data.append(
                {
                    "entry_name": record.entry_name,
                    "localization": primary_loc,
                    "sequence": record.sequence,
                }
            )

    logger.info("Extracted %d protein records", len(protein_data))
    return protein_data


__all__ = ["extract_protein_data", "_clean_and_primary", "ALLOWED_LOCS", "_SYNONYM_MAP"]
