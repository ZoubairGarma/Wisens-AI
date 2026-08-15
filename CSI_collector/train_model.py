"""
train_model.py - WISENS-AI

Entraînement Random Forest sur le dataset de features V2.

IMPORTANT:
Le split est effectué par SESSION et non par fenêtre,
afin d'éviter la fuite de données due au chevauchement
des fenêtres.

Pipeline:

dataset2_features_v2.parquet
        ↓
GroupShuffleSplit par session
        ↓
Random Forest
        ↓
Accuracy
Precision
Recall
F1-score
Confusion Matrix
Feature Importance
"""

import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import (
    GroupShuffleSplit
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

from sklearn.impute import SimpleImputer


# ========================================================================
# CONFIGURATION
# ========================================================================

INPUT_FILE = "dataset2_features_v2.parquet"

TEST_SIZE = 0.20

RANDOM_STATE = 42

N_ESTIMATORS = 300

OUTPUT_IMPORTANCE = "feature_importance_v2.csv"


# ========================================================================
# MAIN
# ========================================================================

def main():

    print("=" * 70)
    print("WISENS-AI - RANDOM FOREST V2")
    print("=" * 70)

    # ------------------------------------------------------------
    # Chargement
    # ------------------------------------------------------------

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"Fichier introuvable: {INPUT_FILE}"
        )

    df = pd.read_parquet(
        INPUT_FILE
    )

    print(
        f"Dataset chargé: {len(df)} fenêtres."
    )

    # ------------------------------------------------------------
    # Vérification
    # ------------------------------------------------------------

    required_columns = [
        "session_id",
        "effective_label",
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Colonne obligatoire absente: {column}"
            )

    # ------------------------------------------------------------
    # Colonnes exclues
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
        column
        for column in df.columns
        if column not in metadata_columns
    ]

    print()
    print(
        f"Nombre de features: "
        f"{len(feature_columns)}"
    )

    # ------------------------------------------------------------
    # X / y / groups
    # ------------------------------------------------------------

    X = df[feature_columns].copy()

    y = df["effective_label"].copy()

    groups = df["session_id"].copy()

    # ------------------------------------------------------------
    # Conversion numérique
    # ------------------------------------------------------------

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # ------------------------------------------------------------
    # Gestion NaN
    # ------------------------------------------------------------

    imputer = SimpleImputer(
        strategy="median"
    )

    X = imputer.fit_transform(X)

    # ------------------------------------------------------------
    # Split par SESSION
    # ------------------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    train_indices, test_indices = next(
        splitter.split(
            X,
            y,
            groups=groups
        )
    )

    X_train = X[train_indices]
    X_test = X[test_indices]

    y_train = y.iloc[
        train_indices
    ]

    y_test = y.iloc[
        test_indices
    ]

    train_sessions = groups.iloc[
        train_indices
    ].unique()

    test_sessions = groups.iloc[
        test_indices
    ].unique()

    # ------------------------------------------------------------
    # Informations split
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("SPLIT PAR SESSION")
    print("=" * 70)

    print(
        f"Sessions TRAIN: {len(train_sessions)}"
    )

    print(
        f"Sessions TEST : {len(test_sessions)}"
    )

    print(
        f"Fenêtres TRAIN: {len(X_train)}"
    )

    print(
        f"Fenêtres TEST : {len(X_test)}"
    )

    # Vérification anti-leakage

    overlap = set(
        train_sessions
    ).intersection(
        set(test_sessions)
    )

    print(
        f"Sessions communes TRAIN/TEST: "
        f"{len(overlap)}"
    )

    if len(overlap) != 0:

        raise RuntimeError(
            "ERREUR: fuite de données détectée!"
        )

    # ------------------------------------------------------------
    # Random Forest
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("ENTRAINEMENT RANDOM FOREST")
    print("=" * 70)

    model = RandomForestClassifier(

        n_estimators=N_ESTIMATORS,

        random_state=RANDOM_STATE,

        n_jobs=-1,

        class_weight="balanced",

        max_features="sqrt",

        min_samples_leaf=2,
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "Entraînement terminé."
    )

    # ------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------

    y_pred = model.predict(
        X_test
    )

    # ------------------------------------------------------------
    # Accuracy
    # ------------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    # ------------------------------------------------------------
    # F1
    # ------------------------------------------------------------

    f1_macro = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    f1_weighted = f1_score(
        y_test,
        y_pred,
        average="weighted"
    )

    # ------------------------------------------------------------
    # Résultats
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("RESULTATS")
    print("=" * 70)

    print(
        f"Accuracy      : {accuracy:.4f}"
    )

    print(
        f"F1 macro      : {f1_macro:.4f}"
    )

    print(
        f"F1 weighted   : {f1_weighted:.4f}"
    )

    # ------------------------------------------------------------
    # Classification report
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    # ------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------

    labels = sorted(
        y.unique()
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )

    print()
    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print(
        pd.DataFrame(
            cm,
            index=labels,
            columns=labels
        )
    )

    # ------------------------------------------------------------
    # Affichage confusion matrix
    # ------------------------------------------------------------

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    disp.plot(
        xticks_rotation=45
    )

    plt.title(
        "WISENS-AI - Random Forest - Confusion Matrix"
    )

    plt.tight_layout()

    plt.show()

    # ------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------

    importance_df = pd.DataFrame({

        "feature": feature_columns,

        "importance":
            model.feature_importances_

    })

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------
    # Affichage
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("TOP 20 FEATURES")
    print("=" * 70)

    print(
        importance_df.head(20).to_string(
            index=False
        )
    )

    # ------------------------------------------------------------
    # Sauvegarde
    # ------------------------------------------------------------

    importance_df.to_csv(
        OUTPUT_IMPORTANCE,
        index=False
    )

    print()
    print(
        f"Importance sauvegardée: "
        f"{os.path.abspath(OUTPUT_IMPORTANCE)}"
    )


if __name__ == "__main__":
    main()