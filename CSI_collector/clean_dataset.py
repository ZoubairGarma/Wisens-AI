"""
clean_dataset.py - WISENS-AI : Nettoyage complet du dataset CSI

Part du dataset DEJA fusionne (dataset_fusionne.parquet, produit par
fusion_dataset.py) et applique le pipeline de nettoyage complet :

  1. Suppression des lignes incompletes / csi_len invalide.
  2. Parsing du champ csi_data (string ";" -> liste d'entiers).
  3. Retrait des 2 premieres paires I/Q (4 valeurs placeholder fixes
     -110,-96,8,0 inserees par le driver Wi-Fi de l'ESP32, confirmees
     identiques sur toutes les sessions -> aucune info reelle,
     a exclure avant tout calcul de features).
  4. Detection (sans suppression) des RSSI hors plage realiste.
  5. Calcul du label effectif :
       - entree_zone / sortie_zone -> marker_state (empty/transition/stable)
       - autres scenarios -> scenario (constant sur tout le fichier)
  6. Sauvegarde du dataset nettoye, pret pour l'extraction de features.

Usage:
    python clean_dataset.py
"""

import os

import pandas as pd

INPUT_PARQUET = "dataset2_fusionne.parquet"    # sortie de fusion_dataset.py
OUTPUT_PARQUET = "dataset2_nettoye.parquet"
OUTPUT_CSV_PREVIEW = "dataset2_nettoye_preview.csv"   # extrait lisible, sans les vecteurs CSI

# ----------------------------------------------------------------------
# Constantes de nettoyage
# ----------------------------------------------------------------------

EXPECTED_CSI_LEN_RAW = 384
PLACEHOLDER_VALUES_COUNT = 4          # 2 paires I/Q placeholder en tete
EXPECTED_CSI_LEN_CLEAN = EXPECTED_CSI_LEN_RAW - PLACEHOLDER_VALUES_COUNT

RSSI_MIN_REALISTIC = -90
RSSI_MAX_REALISTIC = -20

TRANSITION_SCENARIOS = {"entree_zone", "sortie_zone"}


# ----------------------------------------------------------------------
# Etape 1 : chargement du dataset deja fusionne
# ----------------------------------------------------------------------

def load_fused_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' introuvable. Lance d'abord fusion_dataset.py "
            f"pour generer ce fichier a partir des CSV bruts."
        )
    df = pd.read_parquet(path)
    print(f"Dataset fusionne charge: {len(df)} lignes, "
          f"{df['session_id'].nunique()} sessions.\n")
    return df


# ----------------------------------------------------------------------
# Etape 2 : nettoyage
# ----------------------------------------------------------------------

def drop_incomplete_rows(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)
    required = ["timestamp_us", "rssi", "channel", "csi_len", "csi_data", "scenario"]
    df = df.dropna(subset=required)
    n_after = len(df)
    if n_after < n_before:
        print(f"{n_before - n_after} lignes supprimees (valeurs manquantes)")
    return df.reset_index(drop=True)


def check_csi_len(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)
    df["csi_len"] = pd.to_numeric(df["csi_len"], errors="coerce")
    bad_len = df["csi_len"] != EXPECTED_CSI_LEN_RAW
    n_bad = int(bad_len.sum())
    if n_bad > 0:
        print(f"{n_bad} lignes supprimees (csi_len != {EXPECTED_CSI_LEN_RAW})")
        df = df[~bad_len]
    return df.reset_index(drop=True)


def parse_and_clean_csi(df: pd.DataFrame) -> pd.DataFrame:
    """Parse csi_data en liste d'entiers, retire les valeurs placeholder."""
    def parse_row(raw):
        try:
            values = [int(x) for x in str(raw).split(";")]
        except (ValueError, AttributeError):
            return None
        if len(values) != EXPECTED_CSI_LEN_RAW:
            return None
        cleaned = values[PLACEHOLDER_VALUES_COUNT:]
        if len(cleaned) != EXPECTED_CSI_LEN_CLEAN:
            return None
        return cleaned

    df["csi_values"] = df["csi_data"].apply(parse_row)

    n_before = len(df)
    df = df.dropna(subset=["csi_values"])
    n_after = len(df)
    if n_after < n_before:
        print(f"{n_before - n_after} lignes supprimees (echec parsing csi_data)")

    df = df.drop(columns=["csi_data"])   # remplace par csi_values (nettoye)
    print(f"Parsing termine: {EXPECTED_CSI_LEN_CLEAN} valeurs CSI utiles "
          f"par mesure (apres retrait des {PLACEHOLDER_VALUES_COUNT} "
          f"valeurs placeholder).")
    return df.reset_index(drop=True)


def flag_rssi_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df["rssi"] = pd.to_numeric(df["rssi"], errors="coerce")
    anomaly = (df["rssi"] < RSSI_MIN_REALISTIC) | (df["rssi"] > RSSI_MAX_REALISTIC)
    n_anomaly = int(anomaly.sum())
    if n_anomaly > 0:
        print(f"{n_anomaly} lignes avec RSSI hors plage realiste "
              f"[{RSSI_MIN_REALISTIC}, {RSSI_MAX_REALISTIC}] dBm "
              f"(conservees, marquees rssi_anomaly=True)")
    df["rssi_anomaly"] = anomaly
    return df


def compute_effective_label(df: pd.DataFrame) -> pd.DataFrame:
    def label_for_row(row):
        if row["scenario"] in TRANSITION_SCENARIOS:
            return row["marker_state"]
        return row["scenario"]

    df["effective_label"] = df.apply(label_for_row, axis=1)
    return df


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    df = load_fused_dataset(INPUT_PARQUET)

    df = drop_incomplete_rows(df)
    df = check_csi_len(df)
    df = parse_and_clean_csi(df)
    df = flag_rssi_anomalies(df)
    df = compute_effective_label(df)

    print("\n" + "=" * 60)
    print("DISTRIBUTION DES CLASSES (label effectif)")
    print("=" * 60)
    counts = df["effective_label"].value_counts()
    total = len(df)
    for label, count in counts.items():
        pct = 100 * count / total
        print(f"  {label:20s} {count:6d} ({pct:5.1f}%)")
    print(f"\n  TOTAL: {total} echantillons\n")

    # Sauvegarde complete (avec les vecteurs CSI) en Parquet
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"Dataset nettoye sauvegarde: {os.path.abspath(OUTPUT_PARQUET)}")

    # Apercu lisible (sans les gros vecteurs CSI) en CSV, pour inspection rapide
    preview_cols = [c for c in df.columns if c not in ("csi_values",)]
    df[preview_cols].to_csv(OUTPUT_CSV_PREVIEW, index=False)
    print(f"Apercu (sans csi_values) sauvegarde: {os.path.abspath(OUTPUT_CSV_PREVIEW)}")


if __name__ == "__main__":
    main()
