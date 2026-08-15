"""
tune_knn.py - WISENS-AI : Recherche du meilleur reglage KNN.

KNN (k=5) s'est revele le meilleur des 4 algorithmes testes
(compare_models.py). Ce script explore plusieurs valeurs de k et deux
strategies de ponderation, sur EXACTEMENT le meme split train/test
definitif, pour trouver le meilleur reglage sans re-choisir un split
different (ce qui fausserait la comparaison).

Usage:
    python tune_knn.py
"""

import os

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

INPUT_PARQUET = "dataset2_features.parquet"

LABEL_TO_CLASS = {
    "piece_vide":         "C0_empty",
    "empty":              "C0_empty",

    "presence_immobile":  "C1_presence_stable",
    "stable":              "C1_presence_stable",
    "deux_presences_immobile": "C1_presence_stable",

    "mouvement_faible":   "C2_low_motion",
    "transition":          "C2_low_motion",
    "deux_presences":      "C2_low_motion",
    "deux_presences_mouvement_faible": "C2_low_motion",

    "mouvement_fort":      "C3_high_motion",
    "deux_presences_mouvement_fort": "C3_high_motion",

    "perturbation_objet":  "C4_object_disturbance",
}

FEATURE_COLUMNS = [
    "moyenne", "variance", "ecart_type", "maximum", "minimum",
    "energie_signal", "nombre_de_pics", "variation_moyenne",
    "stabilite_temporelle",
    "subcarrier_std_mean", "subcarrier_std_max",
    "subcarrier_std_std", "subcarrier_std_peakiness",
]

HELD_OUT_TEST_RATIO = 0.2
RANDOM_STATE = 42

# Grille testee : valeurs de k impaires (evite les egalites de vote pour
# un probleme a nombre de classes impair... ici 5 classes, moins critique,
# mais bonne pratique generale), et deux strategies de ponderation.
K_VALUES = [1, 3, 5, 7, 9, 11, 15, 21, 31, 41]
WEIGHTS_OPTIONS = ["uniform", "distance"]


def apply_class_mapping(df: pd.DataFrame) -> pd.DataFrame:
    df["final_class"] = df["effective_label"].map(LABEL_TO_CLASS)
    unmapped = df[df["final_class"].isna()]
    if not unmapped.empty:
        raise ValueError(
            f"Labels non mappes: {list(unmapped['effective_label'].unique())}."
        )
    return df


def split_held_out_test(df: pd.DataFrame):
    """IDENTIQUE a compare_models.py / train_model.py : meme graine
    aleatoire, meme logique -> reproduit EXACTEMENT le meme split, pour
    que cette recherche de k soit comparable au reste."""
    X = df[FEATURE_COLUMNS]
    y = df["final_class"]
    groups = df["session_id"]

    min_sessions_per_class = df.groupby("final_class")["session_id"].nunique().min()
    n_splits = max(2, round(1 / HELD_OUT_TEST_RATIO))
    n_splits = min(n_splits, min_sessions_per_class)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(sgkf.split(X, y, groups))

    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def main():
    if not os.path.exists(INPUT_PARQUET):
        raise FileNotFoundError(f"'{INPUT_PARQUET}' introuvable.")

    df = pd.read_parquet(INPUT_PARQUET)
    df = apply_class_mapping(df)

    train_df, test_df = split_held_out_test(df)
    print(f"Train: {len(train_df)} fenetres ({train_df['session_id'].nunique()} sessions)")
    print(f"Test:  {len(test_df)} fenetres ({test_df['session_id'].nunique()} sessions)\n")

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["final_class"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["final_class"]

    results = []

    for weights in WEIGHTS_OPTIONS:
        for k in K_VALUES:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=k, weights=weights)),
            ])
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average="macro")

            results.append({
                "k": k,
                "weights": weights,
                "accuracy": accuracy,
                "f1_macro": f1_macro,
            })
            print(f"k={k:3d} | weights={weights:9s} | "
                  f"accuracy={accuracy:.3f} | f1_macro={f1_macro:.3f}")

    results_df = pd.DataFrame(results).sort_values("f1_macro", ascending=False)

    print("\n" + "=" * 60)
    print("TOP 5 REGLAGES (trie par f1_macro)")
    print("=" * 60)
    print(results_df.head(5).to_string(index=False))

    results_df.to_csv("knn_tuning_results.csv", index=False)
    print(f"\nResultats complets sauvegardes: {os.path.abspath('knn_tuning_results.csv')}")

    best = results_df.iloc[0]
    print(f"\nMeilleur reglage: k={int(best['k'])}, weights={best['weights']} "
          f"(accuracy={best['accuracy']:.3f}, f1_macro={best['f1_macro']:.3f})")

    # Rapport detaille pour le meilleur reglage
    best_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=int(best["k"]), weights=best["weights"])),
    ])
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    print("\nRapport detaille (meilleur reglage):")
    print(classification_report(y_test, y_pred, zero_division=0))


if __name__ == "__main__":
    main()