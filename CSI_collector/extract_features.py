"""
extract_features.py - WISENS-AI : Alignement temporel, normalisation,
                       segmentation par fenetres, extraction des 9
                       caracteristiques (dossier projet, sections 8.1-8.2)

Pipeline (dans l'ordre du dossier projet) :

  ETAPE 3 - Alignement temporel
      Trie chaque session par timestamp_us, verifie l'absence de trous
      anormaux entre mesures consecutives.

  ETAPE 4 - Normalisation
      Convertit chaque mesure CSI (380 valeurs I/Q brutes) en UN SEUL
      signal d'amplitude moyenne par mesure : amplitude par sous-porteuse
      = sqrt(I^2 + Q^2), moyennee sur toutes les sous-porteuses.
      Ce signal d'amplitude est ensuite normalise (z-score) PAR SESSION
      (chaque session a sa propre moyenne/ecart-type de reference,
      pour ne pas melanger les echelles entre sessions differentes).

  ETAPE 5 - Segmentation par fenetres temporelles
      Chaque session est d'abord decoupee en blocs de label constant
      (une fenetre ne doit JAMAIS chevaucher un changement de label,
      ni chevaucher deux sessions differentes). A l'interieur de
      chaque bloc, fenetres glissantes de WINDOW_SIZE mesures avec un
      recouvrement de OVERLAP.

  ETAPE 8.2 - Calcul des 9 caracteristiques (par fenetre)
      Moyenne, variance, ecart-type, max, min, energie du signal,
      nombre de pics, variation moyenne, stabilite temporelle,
      amplitude moyenne CSI -- exactement la liste du dossier projet.

Dependances :
    pip install pandas numpy pyarrow

Usage :
    python extract_features.py
"""

import os

import numpy as np
import pandas as pd

INPUT_PARQUET = "dataset_nettoye.parquet"       # sortie de clean_dataset.py
OUTPUT_PARQUET = "dataset_features.parquet"
OUTPUT_CSV_PREVIEW = "dataset_features_preview.csv"

# ----------------------------------------------------------------------
# Parametres de fenetrage (ETAPE 5)
# ----------------------------------------------------------------------

WINDOW_SIZE = 5      # ~1 seconde a ~5 Hz (intervalle de ping ~200ms)
STRIDE = 3           # recouvrement ~40% -> plus de fenetres, utile pour
                      # les classes minoritaires (transition, stable)


# ========================================================================
# ETAPE 3 - Alignement temporel
# ========================================================================

def align_temporal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["session_id", "timestamp_us"]).reset_index(drop=True)

    # Verification des trous anormaux (> 2 secondes) entre mesures
    # consecutives d'une meme session -> signale, ne supprime rien
    # automatiquement (utile pour reperer des sessions interrompues).
    df["gap_us"] = df.groupby("session_id")["timestamp_us"].diff()
    big_gaps = df[df["gap_us"] > 2_000_000]   # > 2s
    if not big_gaps.empty:
        print(f"ATTENTION: {len(big_gaps)} trous > 2s detectes dans "
              f"{big_gaps['session_id'].nunique()} session(s). "
              f"Sessions concernees: {sorted(big_gaps['session_id'].unique())}")

    print(f"[Etape 3] Alignement temporel termine: {len(df)} lignes triees "
          f"par session/timestamp.\n")
    return df


# ========================================================================
# ETAPE 4 - Normalisation
# ========================================================================

def compute_mean_amplitude(csi_values: list) -> float:
    """Amplitude moyenne sur toutes les sous-porteuses pour UNE mesure.
    csi_values est une liste [I0,Q0,I1,Q1,...] (deja nettoyee, sans les
    valeurs placeholder -- voir clean_dataset.py)."""
    arr = np.asarray(csi_values, dtype=np.float64)
    I = arr[0::2]
    Q = arr[1::2]
    amplitudes = np.sqrt(I**2 + Q**2)
    return float(np.mean(amplitudes))


def normalize_per_session(df: pd.DataFrame) -> pd.DataFrame:
    # Un seul signal d'amplitude moyenne par mesure (reduction des 380
    # valeurs CSI brutes a UNE valeur representative du canal a cet
    # instant precis).
    df["amplitude_raw"] = df["csi_values"].apply(compute_mean_amplitude)

    # Normalisation z-score, calculee INDEPENDAMMENT pour chaque session
    # (chaque session garde sa propre reference de moyenne/ecart-type).
    def zscore(group):
        mean = group.mean()
        std = group.std()
        if std == 0 or pd.isna(std):
            return group * 0.0   # session totalement plate, evite division par 0
        return (group - mean) / std

    df["amplitude_norm"] = df.groupby("session_id")["amplitude_raw"].transform(zscore)

    print(f"[Etape 4] Normalisation terminee: amplitude moyenne calculee "
          f"et normalisee (z-score) pour chaque session.\n")
    return df


# ========================================================================
# ETAPE 5 - Segmentation par fenetres temporelles
# ========================================================================

def segment_into_label_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """Decoupe chaque session en blocs de label constant : une fenetre
    ne doit jamais chevaucher un changement de label (ex: transition
    -> stable) ni deux sessions differentes."""
    df["label_changed"] = (
        (df["session_id"] != df["session_id"].shift())
        | (df["effective_label"] != df["effective_label"].shift())
    )
    df["block_id"] = df["label_changed"].cumsum()
    return df


def make_windows(df: pd.DataFrame) -> pd.DataFrame:
    df = segment_into_label_blocks(df)

    windows = []
    for block_id, block in df.groupby("block_id"):
        block = block.reset_index(drop=True)
        n = len(block)

        if n < WINDOW_SIZE:
            continue   # bloc trop court pour former une seule fenetre complete

        start = 0
        while start + WINDOW_SIZE <= n:
            window_df = block.iloc[start:start + WINDOW_SIZE]
            windows.append(window_df)
            start += STRIDE

    print(f"[Etape 5] Segmentation terminee: {len(windows)} fenetres "
          f"generees ({WINDOW_SIZE} mesures/fenetre, stride={STRIDE}), "
          f"sans chevauchement de label ni de session.\n")
    return windows


# ========================================================================
# ETAPE 8.2 - Calcul des 9 caracteristiques par fenetre
# ========================================================================

# ========================================================================
# Detection de pics (implementation numpy pure, sans scipy)
# ========================================================================

def count_local_peaks(signal: np.ndarray) -> int:
    """Compte les maxima locaux : un point est un pic s'il est
    strictement superieur a ses deux voisins immediats. Equivalent
    simple a scipy.signal.find_peaks() pour ce cas d'usage (petites
    fenetres de quelques points), sans dependance externe."""
    if len(signal) < 3:
        return 0
    is_peak = (signal[1:-1] > signal[:-2]) & (signal[1:-1] > signal[2:])
    return int(np.sum(is_peak))


def extract_features_from_window(window_df: pd.DataFrame) -> dict:
    signal = window_df["amplitude_norm"].to_numpy()

    moyenne = float(np.mean(signal))
    variance = float(np.var(signal))
    ecart_type = float(np.std(signal))
    maximum = float(np.max(signal))
    minimum = float(np.min(signal))
    energie = float(np.sum(signal ** 2))

    # Nombre de pics : detection de maxima locaux (indicateur de
    # mouvement/perturbation dans la fenetre).
    nombre_de_pics = count_local_peaks(signal)

    # Variation moyenne : changement moyen absolu entre mesures successives.
    variation_moyenne = float(np.mean(np.abs(np.diff(signal)))) if len(signal) > 1 else 0.0

    # Stabilite temporelle : capacite a distinguer immobilite et
    # mouvement -- definie ici comme l'inverse de la dispersion
    # (1 = parfaitement stable, proche de 0 = tres instable).
    stabilite_temporelle = float(1.0 / (1.0 + ecart_type))

    # Amplitude moyenne CSI : amplitude brute moyenne (non normalisee),
    # gardee separement de "Moyenne" (qui porte sur le signal normalise)
    # pour respecter exactement l'intitule du dossier projet.
    amplitude_moyenne_csi = float(np.mean(window_df["amplitude_raw"].to_numpy()))

    return {
        "moyenne": moyenne,
        "variance": variance,
        "ecart_type": ecart_type,
        "maximum": maximum,
        "minimum": minimum,
        "energie_signal": energie,
        "nombre_de_pics": nombre_de_pics,
        "variation_moyenne": variation_moyenne,
        "stabilite_temporelle": stabilite_temporelle,
        "amplitude_moyenne_csi": amplitude_moyenne_csi,
    }


def build_feature_table(windows: list) -> pd.DataFrame:
    rows = []
    for window_df in windows:
        features = extract_features_from_window(window_df)

        features["session_id"] = window_df["session_id"].iloc[0]
        features["scenario"] = window_df["scenario"].iloc[0]
        features["effective_label"] = window_df["effective_label"].iloc[0]
        features["window_start_timestamp_us"] = window_df["timestamp_us"].iloc[0]
        features["window_end_timestamp_us"] = window_df["timestamp_us"].iloc[-1]
        features["n_samples_in_window"] = len(window_df)

        rows.append(features)

    feature_df = pd.DataFrame(rows)
    print(f"[Etape 8.2] Extraction des 9 caracteristiques terminee: "
          f"{len(feature_df)} fenetres x 9 features + metadonnees.\n")
    return feature_df


# ========================================================================
# MAIN
# ========================================================================

def main():
    if not os.path.exists(INPUT_PARQUET):
        raise FileNotFoundError(
            f"'{INPUT_PARQUET}' introuvable. Lance d'abord clean_dataset.py."
        )

    df = pd.read_parquet(INPUT_PARQUET)
    print(f"Dataset nettoye charge: {len(df)} lignes.\n")

    df = align_temporal(df)                # Etape 3
    df = normalize_per_session(df)         # Etape 4
    windows = make_windows(df)             # Etape 5
    feature_df = build_feature_table(windows)   # Etape 8.2

    print("=" * 60)
    print("DISTRIBUTION DES CLASSES (par fenetre)")
    print("=" * 60)
    counts = feature_df["effective_label"].value_counts()
    total = len(feature_df)
    for label, count in counts.items():
        pct = 100 * count / total
        print(f"  {label:20s} {count:6d} ({pct:5.1f}%)")
    print(f"\n  TOTAL: {total} fenetres\n")

    feature_df.to_parquet(OUTPUT_PARQUET, index=False)
    feature_df.to_csv(OUTPUT_CSV_PREVIEW, index=False)

    print(f"Dataset de features sauvegarde: {os.path.abspath(OUTPUT_PARQUET)}")
    print(f"Apercu CSV sauvegarde: {os.path.abspath(OUTPUT_CSV_PREVIEW)}")


if __name__ == "__main__":
    main()