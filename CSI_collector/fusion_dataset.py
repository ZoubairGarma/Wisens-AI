"""
fusion_dataset.py - Fusionne tous les CSV WISENS-AI (data/*.csv) en un
                     seul dataset, en conservant une trace du fichier
                     source pour chaque ligne (colonne source_file).

Certains fichiers ont ete generes avec une version du script qui
ecrivait 13 valeurs par ligne (avec marker_state) mais un en-tete de
seulement 12 noms de colonnes (sans marker_state) -> decalage de
colonnes silencieux si on fait confiance a l'en-tete du fichier.

Ce script ignore l'en-tete de chaque fichier et determine le VRAI
nombre de colonnes en comptant les champs de la premiere ligne de
donnees (via le module csv, qui respecte les guillemets), puis assigne
les noms de colonnes corrects en consequence.

Usage:
    python fusion_dataset.py
"""

import csv
import glob
import os

import pandas as pd

DATA_DIR = "data"
OUTPUT_CSV = "dataset_fusionne.csv"
OUTPUT_PARQUET = "dataset_fusionne.parquet"

# Schema avec marker_state (13 colonnes) - fichiers generes apres
# l'ajout du bouton marqueur.
COLUMNS_13 = [
    "timestamp_us", "rssi", "channel", "mac", "csi_len", "csi_data",
    "marker_state", "experiment_id", "zone_id", "scenario",
    "distance_tx_rx_m", "ground_truth", "comment",
]

# Schema sans marker_state (12 colonnes) - fichiers generes avant
# l'ajout du bouton marqueur.
COLUMNS_12 = [
    "timestamp_us", "rssi", "channel", "mac", "csi_len", "csi_data",
    "experiment_id", "zone_id", "scenario",
    "distance_tx_rx_m", "ground_truth", "comment",
]


def detect_column_count(path: str) -> int:
    """Compte le nombre reel de champs dans la 1ere ligne de DONNEES
    (pas l'en-tete), en respectant les guillemets (module csv)."""
    with open(path, "r", newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        next(reader)          # ignore l'en-tete (potentiellement faux)
        first_data_row = next(reader)
    return len(first_data_row)


# Etape 1 : lister tous les fichiers CSV du dossier data/
csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
print(f"{len(csv_files)} fichiers trouves.")


# Etape 2 : charger chaque fichier en detectant le vrai schema de
#           colonnes (ignore l'en-tete du fichier, se base sur le
#           nombre reel de champs des donnees).
frames = []
for path in csv_files:
    filename = os.path.basename(path)
    session_id = os.path.splitext(filename)[0]

    n_cols = detect_column_count(path)

    if n_cols == len(COLUMNS_13):
        columns = COLUMNS_13
    elif n_cols == len(COLUMNS_12):
        columns = COLUMNS_12
    else:
        print(f"  [IGNORE] {filename}: {n_cols} colonnes detectees, "
              f"schema inconnu (attendu {len(COLUMNS_12)} ou "
              f"{len(COLUMNS_13)}) -> fichier saute, a inspecter manuellement.")
        continue

    df = pd.read_csv(path, header=None, skiprows=1, names=columns)

    if "marker_state" not in df.columns:
        df["marker_state"] = "empty"   # valeur par defaut, coherence
                                         # avec les fichiers plus recents

    df["source_file"] = filename
    df["session_id"] = session_id

    frames.append(df)
    print(f"  {filename}: {len(df)} lignes, schema {n_cols} colonnes")


# Etape 3 : fusionner tous les DataFrames en un seul
dataset = pd.concat(frames, ignore_index=True)
print(f"\nTotal fusionne: {len(dataset)} lignes, "
      f"{dataset['session_id'].nunique()} fichiers sources.")


# Etape 4 : forcer le typage numerique des colonnes qui doivent l'etre
#           (securite supplementaire, au cas ou d'autres anomalies
#           existeraient encore).
numeric_cols = ["distance_tx_rx_m", "timestamp_us", "rssi", "channel", "csi_len"]
for col in numeric_cols:
    if col not in dataset.columns:
        continue
    before_na = dataset[col].isna().sum()
    dataset[col] = pd.to_numeric(dataset[col], errors="coerce")
    after_na = dataset[col].isna().sum()
    n_corrupted = after_na - before_na
    if n_corrupted > 0:
        bad_sessions = dataset.loc[dataset[col].isna(), "session_id"].unique()
        print(f"ATTENTION: {n_corrupted} valeurs non numeriques restantes "
              f"dans '{col}', remplacees par NaN.")
        print(f"  Fichier(s) source concerne(s): {list(bad_sessions)}")


# Etape 5 : sauvegarder le resultat (deux formats)
dataset.to_csv(OUTPUT_CSV, index=False)
dataset.to_parquet(OUTPUT_PARQUET, index=False)

print(f"\nSauvegarde: {os.path.abspath(OUTPUT_CSV)}")
print(f"Sauvegarde: {os.path.abspath(OUTPUT_PARQUET)}")


# Etape 6 (exemple) : comment retrouver plus tard les lignes d'un
#                      fichier source precis, a partir du dataset fusionne
#
# dataset = pd.read_parquet("dataset_fusionne.parquet")
# lignes_S022 = dataset[dataset["session_id"] == "EXP_S022_mouvement_faible"]