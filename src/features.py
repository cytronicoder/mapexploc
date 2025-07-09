import pandas as pd
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from sklearn.model_selection import train_test_split


INPUT_ANN = "data/processed/annotations.csv"
INPUT_FASTA = "data/processed/nonredundant.fasta"
OUTPUT_FEAT = "data/processed/features.csv"

OUTPUT_X_TRAIN = "data/processed/X/train.csv"
OUTPUT_X_TMP = "data/processed/X/tmp.csv"
OUTPUT_X_VAL = "data/processed/X/val.csv"
OUTPUT_X_TEST = "data/processed/X/test.csv"

OUTPUT_Y_TRAIN = "data/processed/Y/train.csv"
OUTPUT_Y_TMP = "data/processed/Y/tmp.csv"
OUTPUT_Y_VAL = "data/processed/Y/val.csv"
OUTPUT_Y_TEST = "data/processed/Y/test.csv"


def build_feature_matrix(fasta_path: str, ann_path: str) -> pd.DataFrame:
    """
    Parse a UniProt FASTA and annotation CSV to compute sequence features
    (amino-acid composition, dipeptide frequencies, GRAVY, pI).
    """
    # Load annotation map
    ann = pd.read_csv(ann_path).set_index("entry_name")["localization"].to_dict()

    # Standard amino acids & dipeptides
    standard_aas = list("ACDEFGHIKLMNPQRSTVWY")
    all_dipeps = [a + b for a in standard_aas for b in standard_aas]

    rows = []
    skipped = 0

    for rec in SeqIO.parse(fasta_path, "fasta"):
        # parse from the full description (captures spaces)
        try:
            entry, header_loc = rec.description.split("|", 1)
            header_loc = header_loc.strip()
        except ValueError:
            print(f"Warning: unexpected header format '{rec.description}'")
            continue

        loc = ann.get(entry)
        if loc is None:
            print(f"Warning: no annotation for {entry}")
            continue
        if loc != header_loc:
            print(f"Warning: header says {header_loc}, ann says {loc}")

        # sequence cleaning
        original_seq = str(rec.seq)
        cleaned = "".join([aa for aa in original_seq.upper() if aa in standard_aas])
        if len(cleaned) < 10:
            print(f"Skipping {entry}: cleaned length = {len(cleaned)}")
            skipped += 1
            continue

        # feature dict
        feats = {
            "entry": entry,
            "localization": loc,
            "sequence_length": len(cleaned),
            "original_length": len(original_seq),
        }

        # amino-acid composition + physico-chemical
        try:
            pa = ProteinAnalysis(cleaned)
            comp = {aa: pct / 100 for aa, pct in pa.amino_acids_percent.items()}
            feats.update(comp)
            feats["gravy"] = pa.gravy()
            feats["pI"] = pa.isoelectric_point()
        except Exception as e:
            print(f"Warning: feature calc failed for {entry}: {e}")
            for aa in standard_aas:
                feats[aa] = 0.0
            feats.update({"gravy": None, "pI": None})

        # dipeptide freqs (fixed key set)
        dipep = dict.fromkeys(all_dipeps, 0)
        for i in range(len(cleaned) - 1):
            dp = cleaned[i : i + 2]
            dipep[dp] += 1
        total = sum(dipep.values())
        if total > 0:
            for dp in dipep:
                dipep[dp] /= total

        feats.update({f"dp_{dp}": freq for dp, freq in dipep.items()})
        rows.append(feats)

    print(f"Processed {len(rows)} sequences, skipped {skipped}.")
    results_df = pd.DataFrame(rows).fillna(0)
    return results_df


df = build_feature_matrix(INPUT_FASTA, INPUT_ANN)
df.to_csv(OUTPUT_FEAT, index=False)
print(f"Wrote features to {OUTPUT_FEAT}")
print(f"Feature matrix shape: {df.shape}")
print(f"Features per sequence: {df.shape[1] - 2}")

df = pd.read_csv(OUTPUT_FEAT)
y = df.pop("localization")

X_train, X_tmp, y_train, y_tmp = train_test_split(
    df, y, test_size=0.30, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, random_state=42
)

print("Train dist:\n", y_train.value_counts(normalize=True), "\n")
print("Val dist (approx):\n", y_val.value_counts(normalize=True), "\n")
print("Test dist (approx):\n", y_test.value_counts(normalize=True))


X_train.to_csv(OUTPUT_X_TRAIN, index=False)
X_tmp.to_csv(OUTPUT_X_TMP, index=False)
X_val.to_csv(OUTPUT_X_VAL, index=False)
X_test.to_csv(OUTPUT_X_TEST, index=False)

y_train.to_csv(OUTPUT_Y_TRAIN, index=False)
y_tmp.to_csv(OUTPUT_Y_TMP, index=False)
y_val.to_csv(OUTPUT_Y_VAL, index=False)
y_test.to_csv(OUTPUT_Y_TEST, index=False)

print("Split data into train/val/test sets:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
print(f"Data written to {OUTPUT_X_TRAIN}, {OUTPUT_Y_TRAIN}, etc.")
