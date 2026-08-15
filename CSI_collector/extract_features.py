"""
extract_features.py - WISENS-AI : Extraction de features CSI (v2)

Pipeline :
    dataset2_nettoye.parquet (clean_dataset.py)
        -> alignement temporel
        -> normalisation par session (reference de calibration, pas
           destructive : voir NORMALISATION ci-dessous)
        -> segmentation par fenetres (session + label constants)
        -> extraction de ~50 features en 6 groupes (A-F)
        -> dataset2_features.parquet

CHANGEMENT MAJEUR vs version precedente :
    L'ancienne version reduisait chaque mesure CSI (384 valeurs I/Q)
    a UN SEUL scalaire (amplitude moyenne sur toutes les sous-porteuses)
    avant tout calcul temporel. Cette reduction ecrasait l'information
    sur QUELLES sous-porteuses varient -- une cause probable de la
    confusion persistante entre mouvement faible (C1) et mouvement fort
    (C2/C3), observee identiquement sur 4 algorithmes differents.

    Cette version conserve la matrice complete (n_mesures x
    n_sous_porteuses) le plus longtemps possible dans le pipeline, et
    en extrait des features de DISTRIBUTION et de DIVERSITE entre
    sous-porteuses (groupes B et E), en plus des features globales et
    temporelles classiques (groupes A, C, D) et du RSSI (groupe F).

NORMALISATION :
    Les groupes A, B, D (amplitude brute, distribution par sous-
    porteuse, pics) restent en ECHELLE ABSOLUE (brute) -- l'information
    de distance/environnement y reste presente, mais ce n'est plus
    problematique ici puisque ce script ne sert qu'a UNE seule zone a
    la fois (contrairement a l'ancienne tentative de normalisation
    complete qui avait detruit du signal reel, voir historique).

    Le groupe C (dynamique temporelle, differences) est calcule sur une
    matrice mise a l'echelle par session (division par l'ecart-type de
    l'amplitude de la session -- PAS une soustraction de moyenne, pour
    eviter la degenerescence observee precedemment sur les sessions a
    faible variance). Ca rend les features de VARIATION comparables
    entre sessions de puissance de signal differente, sans supprimer
    l'information de variation elle-meme.

    Aucune normalisation n'utilise de statistique calculee sur plus
    d'une session, ni sur le jeu de test (prevention de fuite de
    donnees, voir ROBUSTESSE).

Dependances :
    pip install pandas numpy pyarrow

Usage :
    python extract_features.py
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

INPUT_PARQUET = "dataset2_nettoye.parquet"          # sortie de clean_dataset.py
OUTPUT_PARQUET = "dataset2_features.parquet"
OUTPUT_CSV_PREVIEW = "dataset2_features_preview.csv"

WINDOW_SIZE = 10
STRIDE = 5
GAP_THRESHOLD_US = 2_000_000   # 2 secondes

EPS = 1e-9   # evite les divisions par zero sans fausser les ratios


# ========================================================================
# ETAPE 3 - ALIGNEMENT TEMPOREL
# ========================================================================

def align_temporal(df: pd.DataFrame) -> pd.DataFrame:

    df = df.sort_values(
        ["session_id", "timestamp_us"]
    ).reset_index(drop=True)

    df["gap_us"] = df.groupby("session_id")["timestamp_us"].diff()
    big_gaps = df[df["gap_us"] > GAP_THRESHOLD_US]
    if not big_gaps.empty:
        print(f"ATTENTION: {len(big_gaps)} trous > {GAP_THRESHOLD_US/1e6:.0f}s "
              f"detectes dans {big_gaps['session_id'].nunique()} session(s). "
              f"Sessions concernees: {sorted(big_gaps['session_id'].unique())}")

    print(
        f"[Etape 3] Alignement terminé: "
        f"{len(df)} lignes."
    )

    return df


# ========================================================================
# Conversion CSI brut -> amplitudes par sous-porteuse
# ========================================================================

def compute_csi_amplitudes(csi_values) -> Optional[np.ndarray]:
    """Convertit une liste [I0,Q0,I1,Q1,...] en amplitudes par sous-porteuse
    (sqrt(I^2+Q^2)). Retourne None si le vecteur est malforme (longueur
    impaire, vide, valeurs non numeriques) -- la mesure sera alors ecartee
    plus loin, avec un avertissement, plutot que de faire planter le script."""
    try:
        arr = np.asarray(csi_values, dtype=np.float64)
    except (TypeError, ValueError):
        return None

    if arr.ndim != 1 or arr.size == 0 or arr.size % 2 != 0:
        return None

    I = arr[0::2]
    Q = arr[1::2]
    return np.sqrt(I ** 2 + Q ** 2)


# ========================================================================
# ETAPE 4 - Normalisation (reference de calibration par session)
# ========================================================================

def normalize_per_session(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les amplitudes par sous-porteuse pour chaque mesure, et une
    reference d'echelle PAR SESSION (ecart-type de l'amplitude moyenne de
    la session) -- utilisee plus tard uniquement pour mettre a l'echelle
    les features de VARIATION temporelle (groupe C), jamais pour ecraser
    les features de distribution/amplitude brutes (groupes A, B, D)."""

    df["amplitude_raw"] = df["csi_values"].apply(compute_csi_amplitudes)

    n_before = len(df)
    df = df[df["amplitude_raw"].notna()].reset_index(drop=True)
    n_after = len(df)
    if n_after < n_before:
        print(f"ATTENTION: {n_before - n_after} mesures ecartees "
              f"(vecteur CSI malforme).")

    df["amplitude_raw_mean"] = df["amplitude_raw"].apply(lambda a: float(np.mean(a)))

    # transform() aligne le resultat sur l'index d'origine (pas de risque
    # de desalignement, contrairement a un merge sur cle) -- calcule
    # STRICTEMENT a l'interieur de chaque session, jamais entre sessions.
    df["session_amp_std"] = df.groupby("session_id")["amplitude_raw_mean"].transform("std")
    df["session_amp_std"] = df["session_amp_std"].replace(0, np.nan)

    print(f"[Etape 4] Normalisation terminee: amplitudes par sous-porteuse "
          f"calculees, reference d'echelle par session etablie.\n")
    return df


# ========================================================================
# ETAPE 5 - SEGMENTATION
# ========================================================================

def segment_into_label_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """Une fenetre ne doit jamais chevaucher un changement de session ni
    de label (effective_label)."""
    df["label_changed"] = (
        (df["session_id"] != df["session_id"].shift())
        |
        (df["effective_label"] != df["effective_label"].shift())
    )

    df["block_id"] = df["label_changed"].cumsum()

    return df


def make_windows(df: pd.DataFrame) -> list[pd.DataFrame]:
    df = segment_into_label_blocks(df)

    windows = []

    for block_id, block in df.groupby("block_id"):

        block = block.reset_index(drop=True)

        n = len(block)

        if n < WINDOW_SIZE:
            continue
            continue

        start = 0

        while start + WINDOW_SIZE <= n:
            windows.append(block.iloc[start:start + WINDOW_SIZE])
            start += STRIDE

    print(f"[Etape 5] Segmentation terminee: {len(windows)} fenetres "
          f"candidates ({WINDOW_SIZE} mesures/fenetre, stride={STRIDE}), "
          f"sans chevauchement de label ni de session.\n")
    return windows


# ========================================================================
# Matrice d'amplitude par sous-porteuse pour une fenetre (avec validation)
# ========================================================================

def get_window_amplitude_matrix(window_df: pd.DataFrame) -> Optional[np.ndarray]:
    """Empile les amplitudes de chaque mesure de la fenetre en une matrice
    (n_mesures x n_sous_porteuses). Retourne None si les longueurs ne sont
    pas homogenes au sein de la fenetre (securite supplementaire, en plus
    du filtrage deja fait par clean_dataset.py)."""
    arrays = window_df["amplitude_raw"].tolist()
    lengths = {len(a) for a in arrays}
    if len(lengths) != 1:
        return None
    return np.stack(arrays)


# ========================================================================
# GROUPE A - Features globales d'amplitude CSI
# ========================================================================

def extract_global_features(amp_matrix: np.ndarray) -> dict:
    flat = amp_matrix.flatten()
    p25, p75 = np.percentile(flat, [25, 75])

    return {
        "amplitude_mean": float(np.mean(flat)),
        "amplitude_std": float(np.std(flat)),
        "amplitude_min": float(np.min(flat)),
        "amplitude_max": float(np.max(flat)),
        "amplitude_range": float(np.max(flat) - np.min(flat)),
        "amplitude_median": float(np.median(flat)),
        "amplitude_percentile_25": float(p25),
        "amplitude_percentile_75": float(p75),
        "amplitude_iqr": float(p75 - p25),
        "amplitude_energy": float(np.sum(flat ** 2)),
    }


# ========================================================================
# GROUPE B - Distribution par sous-porteuse
# GROUPE E - Diversite temporelle entre sous-porteuses
# (regroupes dans une seule fonction : E derive directement des memes
#  statistiques par sous-porteuse que B, evite les doublons de calcul)
# ========================================================================

def extract_subcarrier_features(amp_matrix: np.ndarray) -> dict:
    std_per_subcarrier = amp_matrix.std(axis=0)
    mean_per_subcarrier = amp_matrix.mean(axis=0)
    energy_per_subcarrier = np.sum(amp_matrix ** 2, axis=0)

    features = {
        # --- Groupe B : dispersion temporelle par sous-porteuse ---
        "subcarrier_std_mean": float(np.mean(std_per_subcarrier)),
        "subcarrier_std_max": float(np.max(std_per_subcarrier)),
        "subcarrier_std_std": float(np.std(std_per_subcarrier)),
        "subcarrier_std_range": float(np.max(std_per_subcarrier) - np.min(std_per_subcarrier)),
        # ratio max/mean demande a la fois en groupe B et E dans le
        # cahier des charges -- une seule feature, reutilisee (evite un
        # nom duplique, cf. contrainte "no duplicate feature names").
        "subcarrier_std_peakiness": float(
            np.max(std_per_subcarrier) / (np.mean(std_per_subcarrier) + EPS)
        ),

        # --- Groupe B : niveau moyen par sous-porteuse ---
        "subcarrier_mean_mean": float(np.mean(mean_per_subcarrier)),
        "subcarrier_mean_std": float(np.std(mean_per_subcarrier)),
        "subcarrier_mean_min": float(np.min(mean_per_subcarrier)),
        "subcarrier_mean_max": float(np.max(mean_per_subcarrier)),
        "subcarrier_mean_range": float(np.max(mean_per_subcarrier) - np.min(mean_per_subcarrier)),

        # --- Groupe B : energie par sous-porteuse ---
        "subcarrier_energy_mean": float(np.mean(energy_per_subcarrier)),
        "subcarrier_energy_std": float(np.std(energy_per_subcarrier)),
        "subcarrier_energy_min": float(np.min(energy_per_subcarrier)),
        "subcarrier_energy_max": float(np.max(energy_per_subcarrier)),

        # --- Groupe E : diversite / concentration des variations ---
        # Coefficient de variation de la dispersion elle-meme : distingue
        # "toutes les sous-porteuses varient un peu" (CV faible) de
        # "quelques sous-porteuses varient beaucoup" (CV eleve) --
        # exactement la distinction recherchee pour C1 vs C2.
        "subcarrier_diversity_cv": float(
            np.std(std_per_subcarrier) / (np.mean(std_per_subcarrier) + EPS)
        ),
        "subcarrier_std_iqr": float(
            np.percentile(std_per_subcarrier, 75) - np.percentile(std_per_subcarrier, 25)
        ),
        "subcarrier_upper_to_median_std_ratio": float(
            np.percentile(std_per_subcarrier, 90) / (np.median(std_per_subcarrier) + EPS)
        ),
        # Fraction de sous-porteuses "actives" (variation superieure a la
        # mediane de la fenetre) : un mouvement fort/etendu devrait
        # activer une plus grande fraction de sous-porteuses qu'un
        # mouvement faible/localise.
        "active_subcarrier_fraction": float(
            np.mean(std_per_subcarrier > np.median(std_per_subcarrier))
        ),
    }
    return features


# ========================================================================
# GROUPE D - Detection de pics (sur le signal global d'amplitude)
# ========================================================================

def extract_peak_features(signal: np.ndarray) -> dict:
    """Detection de pics avec un critere de proeminence robuste (base sur
    l'ecart-type du signal DANS la fenetre), pour eviter de compter
    chaque micro-fluctuation comme un evenement de mouvement."""
    n = len(signal)
    if n < 3:
        return {
            "n_peaks": 0, "peak_density": 0.0,
            "max_peak_magnitude": 0.0, "mean_peak_magnitude": 0.0,
        }

    is_local_max = (signal[1:-1] > signal[:-2]) & (signal[1:-1] > signal[2:])
    candidates = signal[1:-1][is_local_max]

    prominence_threshold = np.std(signal) * 0.5
    baseline = np.mean(signal)
    significant = candidates[(candidates - baseline) > prominence_threshold] if candidates.size else np.array([])

    n_peaks = int(significant.size)
    return {
        "n_peaks": n_peaks,
        "peak_density": float(n_peaks / n),
        "max_peak_magnitude": float(np.max(significant)) if n_peaks > 0 else 0.0,
        "mean_peak_magnitude": float(np.mean(significant)) if n_peaks > 0 else 0.0,
    }


# ========================================================================
# GROUPE C - Dynamique temporelle (differences 1er et 2eme ordre)
# ========================================================================

def extract_temporal_features(amp_matrix: np.ndarray, session_amp_std: float) -> dict:
    """Deux echelles complementaires :
    - features "fines" (pooled) : differences PAR sous-porteuse, mises a
      l'echelle par session (division par l'ecart-type de la session --
      pas de soustraction de moyenne, pour eviter la degenerescence sur
      les sessions a faible variance observee precedemment).
    - features "globales" (temporal_variation_*) : sur le signal
      d'amplitude moyenne par mesure, en ECHELLE BRUTE (complementaire,
      capture la dynamique d'ensemble sans dependre de la mise a
      l'echelle par session)."""

    if session_amp_std is None or pd.isna(session_amp_std) or session_amp_std == 0:
        scaled_matrix = amp_matrix
    else:
        scaled_matrix = amp_matrix / session_amp_std

    dx = np.diff(scaled_matrix, axis=0)
    abs_dx = np.abs(dx).flatten()

    if abs_dx.size:
        q1, q3 = np.percentile(abs_dx, [25, 75])
        threshold = np.median(abs_dx) + 1.5 * (q3 - q1)
        fraction_large_changes = float(np.mean(abs_dx > threshold))
    else:
        fraction_large_changes = 0.0

    d2x = np.diff(dx, axis=0).flatten() if dx.shape[0] > 1 else np.array([])

    global_signal = amp_matrix.mean(axis=1)   # brut, non mis a l'echelle
    global_dx = np.diff(global_signal)

    features = {
        # --- differences fines, mises a l'echelle par session ---
        "mean_absolute_difference": float(np.mean(abs_dx)) if abs_dx.size else 0.0,
        "std_absolute_difference": float(np.std(abs_dx)) if abs_dx.size else 0.0,
        "max_absolute_difference": float(np.max(abs_dx)) if abs_dx.size else 0.0,
        "fraction_large_changes": fraction_large_changes,
        "second_diff_mean": float(np.mean(d2x)) if d2x.size else 0.0,
        "second_diff_std": float(np.std(d2x)) if d2x.size else 0.0,
        "second_diff_energy": float(np.sum(d2x ** 2)) if d2x.size else 0.0,

        # --- dynamique globale, echelle brute ---
        "temporal_variation_mean": float(np.mean(np.abs(global_dx))) if global_dx.size else 0.0,
        "temporal_variation_std": float(np.std(np.abs(global_dx))) if global_dx.size else 0.0,
        "temporal_variation_max": float(np.max(np.abs(global_dx))) if global_dx.size else 0.0,
        "temporal_variation_energy": float(np.sum(global_dx ** 2)) if global_dx.size else 0.0,
    }

    features.update(extract_peak_features(global_signal))
    return features


# ========================================================================
# GROUPE F - RSSI
# ========================================================================

def extract_rssi_features(window_df: pd.DataFrame) -> dict:
    rssi = window_df["rssi"].to_numpy(dtype=np.float64)
    d = np.diff(rssi)

    return {
        "rssi_mean": float(np.mean(rssi)),
        "rssi_std": float(np.std(rssi)),
        "rssi_min": float(np.min(rssi)),
        "rssi_max": float(np.max(rssi)),
        "rssi_range": float(np.max(rssi) - np.min(rssi)),
        "rssi_median": float(np.median(rssi)),
        "rssi_variation_mean": float(np.mean(np.abs(d))) if d.size else 0.0,
        "rssi_variation_std": float(np.std(np.abs(d))) if d.size else 0.0,
    }


# ========================================================================
# Assemblage : une fenetre -> un vecteur de features complet
# ========================================================================

def extract_features_from_window(window_df: pd.DataFrame) -> Optional[dict]:
    amp_matrix = get_window_amplitude_matrix(window_df)
    if amp_matrix is None:
        return None

    session_amp_std = window_df["session_amp_std"].iloc[0]

    features = {}
    features.update(extract_global_features(amp_matrix))
    features.update(extract_subcarrier_features(amp_matrix))
    features.update(extract_temporal_features(amp_matrix, session_amp_std))
    features.update(extract_rssi_features(window_df))

    # Metadonnees (JAMAIS utilisees comme features ML -- voir train_model.py,
    # ou FEATURE_COLUMNS doit lister explicitement les colonnes numeriques
    # ci-dessus, jamais celles-ci).
    features["session_id"] = window_df["session_id"].iloc[0]
    features["scenario"] = window_df["scenario"].iloc[0]
    features["effective_label"] = window_df["effective_label"].iloc[0]
    features["window_start_timestamp_us"] = window_df["timestamp_us"].iloc[0]
    features["window_end_timestamp_us"] = window_df["timestamp_us"].iloc[-1]
    features["n_samples_in_window"] = len(window_df)

    return features


def build_feature_table(windows: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    n_discarded = 0

    for window_df in windows:
        features = extract_features_from_window(window_df)
        if features is None:
            n_discarded += 1
            continue
        rows.append(features)

    if n_discarded > 0:
        print(f"ATTENTION: {n_discarded} fenetre(s) ecartee(s) "
              f"(longueurs de vecteurs CSI incoherentes au sein de la fenetre).")

    feature_df = pd.DataFrame(rows)

    n_feature_cols = len([c for c in feature_df.columns if c not in (
        "session_id", "scenario", "effective_label",
        "window_start_timestamp_us", "window_end_timestamp_us", "n_samples_in_window",
    )])

    print(f"[Extraction] Termine: {len(feature_df)} fenetres valides "
          f"({n_discarded} ecartees), {n_feature_cols} features numeriques "
          f"+ 6 colonnes de metadonnees.\n")
    return feature_df


# ========================================================================
# MAIN
# ========================================================================

def main():

    if not os.path.exists(INPUT_PARQUET):

        raise FileNotFoundError(
            f"'{INPUT_PARQUET}' introuvable. "
            f"Lance d'abord clean_dataset.py."
        )

    df = pd.read_parquet(INPUT_PARQUET)
    print(f"Dataset nettoye charge: {len(df)} mesures.\n")

    df = align_temporal(df)
    df = normalize_per_session(df)
    windows = make_windows(df)
    feature_df = build_feature_table(windows)

    print("=" * 60)
    print("DISTRIBUTION DES CLASSES (par fenetre, effective_label)")
    print("=" * 60)
    counts = feature_df["effective_label"].value_counts()
    total = len(feature_df)

    for label, count in counts.items():
        pct = 100 * count / total if total else 0
        print(f"  {str(label):35s} {count:6d} ({pct:5.1f}%)")
    print(f"\n  TOTAL: {total} fenetres, "
          f"{feature_df['session_id'].nunique()} sessions\n")

    feature_columns = [c for c in feature_df.columns if c not in (
        "session_id", "scenario", "effective_label",
        "window_start_timestamp_us", "window_end_timestamp_us", "n_samples_in_window",
    )]
    print(f"Liste des {len(feature_columns)} features generees :")
    for name in feature_columns:
        print(f"  - {name}")
    print()

    feature_df.to_parquet(OUTPUT_PARQUET, index=False)
    feature_df.to_csv(OUTPUT_CSV_PREVIEW, index=False)

    print()
    print("=" * 70)
    print("SAUVEGARDE TERMINEE")
    print("=" * 70)

    print(
        f"Parquet: {os.path.abspath(OUTPUT_PARQUET)}"
    )

    print(
        f"CSV:     {os.path.abspath(OUTPUT_CSV)}"
    )


if __name__ == "__main__":
    main()