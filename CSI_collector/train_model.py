"""
train_model.py - WISENS-AI : Labellisation finale (C0-C5), split
                  train/test par session, entrainement du modele
                  baseline et evaluation (dossier projet, sections
                  8.3 a 8.5).

MODIFICATIONS APPORTEES :
  1. C4_complex_activity separee en deux classes distinctes
     (C4_multi_presence / C5_object_disturbance) : deux_presences et
     perturbation_objet sont des phenomenes radio tres differents,
     les fusionner ajoutait du bruit d'etiquetage.
  2. Ajout d'un diagnostic de confusion environnement/classe : verifie
     si le RSSI moyen varie de facon systematique et anormale entre
     classes, signe possible que le modele apprend l'environnement
     de capture (distance/routeur) plutot que le phenomene physique
     reel -- risque connu ici car la distance emetteur/recepteur a
     change en cours de collecte (voir historique du projet).

Usage:
    python train_model.py
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

INPUT_PARQUET = "dataset_features.parquet"   # sortie de extract_features.py
OUTPUT_MODEL = "model_random_forest.pkl"
OUTPUT_CONFUSION_CSV = "confusion_matrix.csv"

# ========================================================================
# ETAPE 8.3 - Mapping vers les classes finales
# ========================================================================

LABEL_TO_CLASS = {
    "piece_vide":         "C0_empty",
    "empty":              "C0_empty",

    "presence_immobile":  "C1_presence_stable",
    "stable":             "C1_presence_stable",

    "mouvement_faible":   "C2_low_motion",
    "transition":         "C2_low_motion",
    "deux_presences":     "C2_low_motion",   # protocole reel = mouvement
                                              # faible (deux personnes peu
                                              # mobiles), confirme par
                                              # observation directe -- pas
                                              # une classe distincte

    "mouvement_fort":     "C3_high_motion",

    "perturbation_objet": "C4_object_disturbance",
}

FEATURE_COLUMNS = [
    "moyenne", "variance", "ecart_type", "maximum", "minimum",
    "energie_signal", "nombre_de_pics", "variation_moyenne",
    "stabilite_temporelle", "amplitude_moyenne_csi",
    "subcarrier_std_mean", "subcarrier_std_max",
    "subcarrier_std_std", "subcarrier_std_peakiness",
]

TEST_SIZE = 0.25   # 25% des SESSIONS (pas des fenetres) en test
RANDOM_STATE = 42


def apply_class_mapping(df: pd.DataFrame) -> pd.DataFrame:
    df["final_class"] = df["effective_label"].map(LABEL_TO_CLASS)

    unmapped = df[df["final_class"].isna()]
    if not unmapped.empty:
        unknown_labels = unmapped["effective_label"].unique()
        raise ValueError(
            f"Labels non mappes trouves: {list(unknown_labels)}. "
            f"Ajoute-les dans LABEL_TO_CLASS avant de continuer."
        )

    print("[Etape 8.3] Mapping vers les classes finales termine.\n")
    return df


def check_environment_confound(df: pd.DataFrame):
    """Diagnostic : le RSSI absolu (amplitude_moyenne_csi) depend
    fortement de la distance emetteur/recepteur et du materiel reseau
    utilise -- pas du phenomene physique (mouvement/presence) qu'on veut
    detecter. Si une classe a un RSSI systematiquement tres different
    des autres, c'est un signe que le modele risque d'apprendre a
    reconnaitre les CONDITIONS DE CAPTURE (quelle session/quel jour/
    quelle distance) plutot que le vrai phenomene -- un biais que le
    split par session ne peut PAS corriger si le biais est constant
    sur toute une classe."""
    print("=" * 60)
    print("[Diagnostic] Confusion environnement / classe")
    print("=" * 60)

    stats = df.groupby("final_class")["amplitude_moyenne_csi"].agg(["mean", "std", "count"])
    print(stats.to_string())

    overall_mean = df["amplitude_moyenne_csi"].mean()
    overall_std = df["amplitude_moyenne_csi"].std()

    suspicious = stats[
        (stats["mean"] - overall_mean).abs() > 1.5 * overall_std
    ]

    if not suspicious.empty:
        print(f"\nATTENTION: {len(suspicious)} classe(s) avec une amplitude "
              f"CSI moyenne tres eloignee de la moyenne globale:")
        print(suspicious.to_string())
        print("\n-> Risque de confusion entre 'phenomene physique' et "
              "'conditions de capture' (distance, routeur, jour). "
              "A verifier: ces classes ont-elles ete enregistrees dans "
              "des conditions differentes (distance, routeur) des autres ?")
    else:
        print("\nAucune classe n'a une amplitude CSI moyenne anormalement "
              "eloignee des autres. Pas de signe evident de confusion "
              "environnement/classe sur cette feature.")
    print()


def cross_validate_by_group(df: pd.DataFrame, n_folds: int = 5):
    """Diagnostic complementaire : variance entre folds independants
    (different de evaluate_via_cross_validation, qui agrege TOUTES les
    predictions out-of-fold en un seul rapport -- ici on regarde plutot
    la stabilite fold par fold)."""
    X = df[FEATURE_COLUMNS]
    y = df["final_class"]
    groups = df["session_id"]

    min_sessions_per_class = df.groupby("final_class")["session_id"].nunique().min()
    n_folds = max(2, min(n_folds, min_sessions_per_class))

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    fold_accuracies = []
    fold_f1_macros = []

    print(f"[Diagnostic] Stabilite entre folds ({n_folds} folds)...\n")

    for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        fold_accuracies.append(acc)
        fold_f1_macros.append(f1_macro)

        test_sessions = df.iloc[test_idx]["session_id"].nunique()
        print(f"  Fold {fold_idx}: accuracy={acc:.3f} | f1_macro={f1_macro:.3f} "
              f"| {len(test_idx)} fenetres test ({test_sessions} sessions)")

    print(f"\n[Diagnostic] Accuracy moyenne: {np.mean(fold_accuracies):.3f} "
          f"(+/- {np.std(fold_accuracies):.3f})")
    print(f"[Diagnostic] F1-macro moyen:   {np.mean(fold_f1_macros):.3f} "
          f"(+/- {np.std(fold_f1_macros):.3f})\n")

    if np.std(fold_accuracies) > 0.10:
        print("INTERPRETATION: forte variance entre folds -> peu de sessions "
              "par classe rend chaque fold individuel peu fiable. C'est "
              "pourquoi EVALUATE_VIA_CROSS_VALIDATION (ci-dessous), qui "
              "agrege TOUTES les predictions, est la reference a utiliser "
              "pour juger la performance reelle du modele.\n")

    return fold_accuracies, fold_f1_macros


def evaluate_via_cross_validation(df: pd.DataFrame) -> RandomForestClassifier:
    """Evaluation par predictions 'out-of-fold' : chaque session sert de
    test EXACTEMENT une fois, sur l'integralite du dataset. Plus fiable
    qu'un split unique quand certaines classes n'ont que 3 sessions
    (un seul split malchanceux peut alors donner des metriques a 0.01
    de precision qui ne reflete rien de reel -- voir C0_empty, run
    precedent avec support=309 mais precision=0.01)."""

    X = df[FEATURE_COLUMNS]
    y = df["final_class"]
    groups = df["session_id"]

    min_sessions_per_class = df.groupby("final_class")["session_id"].nunique().min()
    n_splits = max(2, min(5, min_sessions_per_class))

    print(f"[Evaluation] Cross-validation out-of-fold ({n_splits} folds, "
          f"limite par la classe la moins fournie: {min_sessions_per_class} sessions)\n")

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    splits = list(sgkf.split(X, y, groups))

    model = RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )

    y_pred = cross_val_predict(model, X, y, cv=splits)

    accuracy = accuracy_score(y, y_pred)
    f1_macro = f1_score(y, y_pred, average="macro")
    f1_weighted = f1_score(y, y_pred, average="weighted")

    print("=" * 60)
    print("EVALUATION DU MODELE (out-of-fold, toutes sessions testees une fois)")
    print("=" * 60)
    print(f"Accuracy globale : {accuracy:.3f}")
    print(f"F1-score (macro) : {f1_macro:.3f}")
    print(f"F1-score (weighted): {f1_weighted:.3f}\n")
    print("Rapport detaille par classe (precision / recall / f1):")
    print(classification_report(y, y_pred, zero_division=0))

    labels = sorted(y.unique())
    cm = confusion_matrix(y, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print("Matrice de confusion (lignes = vrai, colonnes = predit):")
    print(cm_df.to_string())
    print()

    cm_df.to_csv(OUTPUT_CONFUSION_CSV)
    print(f"Matrice de confusion sauvegardee: {os.path.abspath(OUTPUT_CONFUSION_CSV)}")

    # Modele final entraine sur TOUT le dataset (pour le deploiement --
    # ce modele n'est pas celui evalue ci-dessus, les metriques
    # out-of-fold restent la reference honnete de sa performance attendue).
    final_model = RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    final_model.fit(X, y)

    importances = pd.Series(final_model.feature_importances_, index=FEATURE_COLUMNS)
    importances = importances.sort_values(ascending=False)
    print("\nImportance des features (Random Forest, modele final):")
    for feat, imp in importances.items():
        print(f"  {feat:25s} {imp:.3f}")

    return final_model, accuracy, f1_macro


def main():
    if not os.path.exists(INPUT_PARQUET):
        raise FileNotFoundError(
            f"'{INPUT_PARQUET}' introuvable. Lance d'abord extract_features.py."
        )

    df = pd.read_parquet(INPUT_PARQUET)
    print(f"Dataset de features charge: {len(df)} fenetres.\n")

    df = apply_class_mapping(df)

    print("Distribution des classes finales:")
    counts = df["final_class"].value_counts()
    total = len(df)
    for label, count in counts.items():
        pct = 100 * count / total
        print(f"  {label:25s} {count:6d} ({pct:5.1f}%)")
    print()

    check_environment_confound(df)

    cross_validate_by_group(df, n_folds=5)

    model, accuracy, f1_macro = evaluate_via_cross_validation(df)

    joblib.dump(model, OUTPUT_MODEL)
    print(f"\nModele sauvegarde (entraine sur 100% du dataset): {os.path.abspath(OUTPUT_MODEL)}")
    print(f"(accuracy out-of-fold={accuracy:.3f}, f1_macro out-of-fold={f1_macro:.3f})")


if __name__ == "__main__":
    main() 