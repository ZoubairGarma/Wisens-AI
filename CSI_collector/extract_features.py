"""
extract_features_v2.py - WISENS-AI

Feature extraction améliorée pour la classification CSI.

Pipeline:
    Dataset nettoyé
        ↓
    Alignement temporel
        ↓
    Amplitude CSI par sous-porteuse
        ↓
    Normalisation par session
        ↓
    Fenêtrage
        ↓
    Feature engineering
        ↓
    Dataset de features

Les features comprennent:
    - statistiques temporelles
    - statistiques d'amplitude
    - statistiques de variation
    - diversité entre sous-porteuses
    - énergie des sous-porteuses
    - variation temporelle des sous-porteuses

IMPORTANT:
Les fenêtres ne traversent jamais:
    - deux sessions différentes
    - deux labels différents
"""

import os

import numpy as np
import pandas as pd


# ========================================================================
# CONFIGURATION
# ========================================================================

INPUT_PARQUET = "dataset2_nettoye.parquet"

OUTPUT_PARQUET = "dataset2_features_v2.parquet"
OUTPUT_CSV = "dataset2_features_v2.csv"

WINDOW_SIZE = 10
STRIDE = 5


# ========================================================================
# ETAPE 3 - ALIGNEMENT TEMPOREL
# ========================================================================

def align_temporal(df: pd.DataFrame) -> pd.DataFrame:

    df = df.sort_values(
        ["session_id", "timestamp_us"]
    ).reset_index(drop=True)

    df["gap_us"] = (
        df.groupby("session_id")["timestamp_us"].diff()
    )

    big_gaps = df[df["gap_us"] > 2_000_000]

    if not big_gaps.empty:
        print(
            f"ATTENTION: {len(big_gaps)} trous > 2s détectés "
            f"dans {big_gaps['session_id'].nunique()} session(s)."
        )

        print(
            "Sessions concernées:",
            sorted(big_gaps["session_id"].unique())
        )

    print(
        f"[Etape 3] Alignement terminé: "
        f"{len(df)} lignes."
    )

    return df


# ========================================================================
# ETAPE 4 - CALCUL AMPLITUDE CSI
# ========================================================================

def compute_amplitude_array(csi_values):

    arr = np.asarray(csi_values, dtype=np.float64)

    # I/Q entrelacés
    I = arr[0::2]
    Q = arr[1::2]

    amplitude = np.sqrt(I ** 2 + Q ** 2)

    return amplitude


def compute_mean_amplitude(csi_values):

    amplitude = compute_amplitude_array(csi_values)

    return float(np.mean(amplitude))


def normalize_per_session(df):

    df = df.copy()

    # Amplitude moyenne de chaque mesure
    df["amplitude_raw"] = df["csi_values"].apply(
        compute_mean_amplitude
    )

    def zscore(group):

        mean = group.mean()
        std = group.std()

        if std == 0 or pd.isna(std):
            return group * 0.0

        return (group - mean) / std

    # Normalisation temporelle par session
    df["amplitude_norm"] = (
        df.groupby("session_id")["amplitude_raw"]
        .transform(zscore)
    )

    print(
        "[Etape 4] Normalisation par session terminée."
    )

    return df


# ========================================================================
# ETAPE 5 - SEGMENTATION
# ========================================================================

def segment_into_label_blocks(df):

    df = df.copy()

    df["label_changed"] = (
        (df["session_id"] != df["session_id"].shift())
        |
        (df["effective_label"] != df["effective_label"].shift())
    )

    df["block_id"] = df["label_changed"].cumsum()

    return df


def make_windows(df):

    df = segment_into_label_blocks(df)

    windows = []

    for block_id, block in df.groupby("block_id"):

        block = block.reset_index(drop=True)

        n = len(block)

        if n < WINDOW_SIZE:
            continue

        start = 0

        while start + WINDOW_SIZE <= n:

            window_df = block.iloc[
                start:start + WINDOW_SIZE
            ]

            windows.append(window_df)

            start += STRIDE

    print(
        f"[Etape 5] {len(windows)} fenêtres générées "
        f"(window={WINDOW_SIZE}, stride={STRIDE})."
    )

    return windows


# ========================================================================
# FEATURES DE BASE
# ========================================================================

def count_local_peaks(signal):

    if len(signal) < 3:
        return 0

    is_peak = (
        (signal[1:-1] > signal[:-2])
        &
        (signal[1:-1] > signal[2:])
    )

    return int(np.sum(is_peak))


# ========================================================================
# MATRICE AMPLITUDE SOUS-PORTEUSES
# ========================================================================

def get_subcarrier_amplitude_matrix(window_df):

    matrix = np.stack(
        [
            np.asarray(v, dtype=np.float64)
            for v in window_df["csi_values"]
        ]
    )

    I = matrix[:, 0::2]
    Q = matrix[:, 1::2]

    amplitude = np.sqrt(I ** 2 + Q ** 2)

    return amplitude


# ========================================================================
# FEATURES SOUS-PORTEUSES
# ========================================================================

def extract_subcarrier_features(window_df):

    amp_matrix = get_subcarrier_amplitude_matrix(
        window_df
    )

    features = {}

    # ------------------------------------------------------------
    # 1. Variabilité temporelle de chaque sous-porteuse
    # ------------------------------------------------------------

    std_per_subcarrier = np.std(
        amp_matrix,
        axis=0
    )

    features["subcarrier_std_mean"] = float(
        np.mean(std_per_subcarrier)
    )

    features["subcarrier_std_max"] = float(
        np.max(std_per_subcarrier)
    )

    features["subcarrier_std_std"] = float(
        np.std(std_per_subcarrier)
    )

    features["subcarrier_std_range"] = float(
        np.max(std_per_subcarrier)
        -
        np.min(std_per_subcarrier)
    )

    features["subcarrier_std_peakiness"] = float(
        np.max(std_per_subcarrier)
        /
        (np.mean(std_per_subcarrier) + 1e-9)
    )

    # ------------------------------------------------------------
    # 2. Profil moyen des sous-porteuses
    # ------------------------------------------------------------

    mean_per_subcarrier = np.mean(
        amp_matrix,
        axis=0
    )

    features["subcarrier_mean_mean"] = float(
        np.mean(mean_per_subcarrier)
    )

    features["subcarrier_mean_std"] = float(
        np.std(mean_per_subcarrier)
    )

    features["subcarrier_mean_min"] = float(
        np.min(mean_per_subcarrier)
    )

    features["subcarrier_mean_max"] = float(
        np.max(mean_per_subcarrier)
    )

    features["subcarrier_mean_range"] = float(
        np.max(mean_per_subcarrier)
        -
        np.min(mean_per_subcarrier)
    )

    # ------------------------------------------------------------
    # 3. Energie des sous-porteuses
    # ------------------------------------------------------------

    energy_per_subcarrier = np.mean(
        amp_matrix ** 2,
        axis=0
    )

    features["subcarrier_energy_mean"] = float(
        np.mean(energy_per_subcarrier)
    )

    features["subcarrier_energy_std"] = float(
        np.std(energy_per_subcarrier)
    )

    features["subcarrier_energy_min"] = float(
        np.min(energy_per_subcarrier)
    )

    features["subcarrier_energy_max"] = float(
        np.max(energy_per_subcarrier)
    )

    # ------------------------------------------------------------
    # 4. Variation temporelle des sous-porteuses
    # ------------------------------------------------------------

    if amp_matrix.shape[0] > 1:

        temporal_diff = np.diff(
            amp_matrix,
            axis=0
        )

        abs_temporal_diff = np.abs(
            temporal_diff
        )

        features["temporal_variation_mean"] = float(
            np.mean(abs_temporal_diff)
        )

        features["temporal_variation_std"] = float(
            np.std(abs_temporal_diff)
        )

        features["temporal_variation_max"] = float(
            np.max(abs_temporal_diff)
        )

        features["temporal_variation_energy"] = float(
            np.mean(temporal_diff ** 2)
        )

    else:

        features["temporal_variation_mean"] = 0.0
        features["temporal_variation_std"] = 0.0
        features["temporal_variation_max"] = 0.0
        features["temporal_variation_energy"] = 0.0

    return features


# ========================================================================
# FEATURES TEMPORELLES
# ========================================================================

def extract_features_from_window(window_df):

    signal = (
        window_df["amplitude_norm"]
        .to_numpy(dtype=np.float64)
    )

    features = {}

    # ------------------------------------------------------------
    # Statistiques temporelles existantes
    # ------------------------------------------------------------

    features["moyenne"] = float(
        np.mean(signal)
    )

    features["variance"] = float(
        np.var(signal)
    )

    features["ecart_type"] = float(
        np.std(signal)
    )

    features["maximum"] = float(
        np.max(signal)
    )

    features["minimum"] = float(
        np.min(signal)
    )

    features["energie_signal"] = float(
        np.sum(signal ** 2)
    )

    features["nombre_de_pics"] = count_local_peaks(
        signal
    )

    if len(signal) > 1:

        diff = np.diff(signal)

        features["variation_moyenne"] = float(
            np.mean(np.abs(diff))
        )

        features["variation_std"] = float(
            np.std(diff)
        )

        features["variation_max"] = float(
            np.max(np.abs(diff))
        )

        features["variation_energy"] = float(
            np.mean(diff ** 2)
        )

    else:

        features["variation_moyenne"] = 0.0
        features["variation_std"] = 0.0
        features["variation_max"] = 0.0
        features["variation_energy"] = 0.0

    features["stabilite_temporelle"] = float(
        1.0 / (1.0 + features["ecart_type"])
    )

    # Amplitude CSI avant normalisation
    features["amplitude_moyenne_csi"] = float(
        np.mean(
            window_df["amplitude_raw"]
            .to_numpy(dtype=np.float64)
        )
    )

    # ------------------------------------------------------------
    # RSSI
    # ------------------------------------------------------------

    if "rssi" in window_df.columns:

        rssi = window_df["rssi"].to_numpy(
            dtype=np.float64
        )

        features["rssi_mean"] = float(
            np.mean(rssi)
        )

        features["rssi_std"] = float(
            np.std(rssi)
        )

        features["rssi_min"] = float(
            np.min(rssi)
        )

        features["rssi_max"] = float(
            np.max(rssi)
        )

        features["rssi_range"] = float(
            np.max(rssi) - np.min(rssi)
        )

    # ------------------------------------------------------------
    # Features sous-porteuses
    # ------------------------------------------------------------

    subcarrier_features = (
        extract_subcarrier_features(window_df)
    )

    features.update(
        subcarrier_features
    )

    return features


# ========================================================================
# CONSTRUCTION DU DATASET
# ========================================================================

def build_feature_table(windows):

    rows = []

    for window_df in windows:

        features = extract_features_from_window(
            window_df
        )

        # Métadonnées
        features["session_id"] = (
            window_df["session_id"].iloc[0]
        )

        features["scenario"] = (
            window_df["scenario"].iloc[0]
        )

        features["effective_label"] = (
            window_df["effective_label"].iloc[0]
        )

        features["window_start_timestamp_us"] = (
            window_df["timestamp_us"].iloc[0]
        )

        features["window_end_timestamp_us"] = (
            window_df["timestamp_us"].iloc[-1]
        )

        features["n_samples_in_window"] = (
            len(window_df)
        )

        rows.append(features)

    feature_df = pd.DataFrame(rows)

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

    print("=" * 70)
    print("WISENS-AI - FEATURE EXTRACTION V2")
    print("=" * 70)

    df = pd.read_parquet(
        INPUT_PARQUET
    )

    print(
        f"Dataset chargé: {len(df)} mesures."
    )

    # ------------------------------------------------------------
    # ETAPE 3
    # ------------------------------------------------------------

    df = align_temporal(df)

    # ------------------------------------------------------------
    # ETAPE 4
    # ------------------------------------------------------------

    df = normalize_per_session(df)

    # ------------------------------------------------------------
    # ETAPE 5
    # ------------------------------------------------------------

    windows = make_windows(df)

    # ------------------------------------------------------------
    # ETAPE 8
    # ------------------------------------------------------------

    feature_df = build_feature_table(
        windows
    )

    print(
        f"[Etape 8] {len(feature_df)} fenêtres "
        f"transformées en features."
    )

    # ------------------------------------------------------------
    # Distribution
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("DISTRIBUTION DES CLASSES")
    print("=" * 70)

    counts = (
        feature_df["effective_label"]
        .value_counts()
    )

    total = len(feature_df)

    for label, count in counts.items():

        percentage = (
            100 * count / total
        )

        print(
            f"{label:35s} "
            f"{count:6d} "
            f"({percentage:5.1f}%)"
        )

    print()
    print(
        f"TOTAL: {total} fenêtres"
    )

    # ------------------------------------------------------------
    # Nombre de features
    # ------------------------------------------------------------

    metadata_columns = [
        "session_id",
        "scenario",
        "effective_label",
        "window_start_timestamp_us",
        "window_end_timestamp_us",
        "n_samples_in_window",
    ]

    feature_columns = [
        col
        for col in feature_df.columns
        if col not in metadata_columns
    ]

    print()
    print(
        f"Nombre de features: "
        f"{len(feature_columns)}"
    )

    print()
    print("Features utilisées:")

    for feature in feature_columns:

        print(
            f"  - {feature}"
        )

    # ------------------------------------------------------------
    # Sauvegarde
    # ------------------------------------------------------------

    feature_df.to_parquet(
        OUTPUT_PARQUET,
        index=False
    )

    feature_df.to_csv(
        OUTPUT_CSV,
        index=False
    )

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