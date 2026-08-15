"""
compare_models.py - WISENS-AI : Comparaison de plusieurs algorithmes de
                     classification (dossier projet, chapitre 8.4).

Compare KNN, SVM, Random Forest et Gradient Boosting, tous evalues sur
EXACTEMENT le meme split train/test definitif (par session, stratifie),
pour une comparaison equitable.

KNN et SVM sont sensibles a l'echelle des features (contrairement au
Random Forest / Gradient Boosting, bases sur des arbres) : un
StandardScaler est applique pour ces deux modeles, ajuste UNIQUEMENT
sur le train (jamais sur le test, pour eviter toute fuite d'information).

Usage:
    python compare_models.py
"""

import os

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

INPUT_PARQUET = "dataset2_features.parquet"   # zone2 -- dataset le plus fourni

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

# ------------------------------------------------------------------------
# Modeles a comparer. KNN et SVM passent par un Pipeline avec
# StandardScaler (fit sur train uniquement, applique sur test) --
# Random Forest et Gradient Boosting n'en ont pas besoin (bases sur des
# seuils par arbre, insensibles a l'echelle).
# ------------------------------------------------------------------------

def build_models():
    return {
        "KNN (k=5)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5
            )),
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, random_state=RANDOM_STATE,
        ),
    }


def apply_class_mapping(df: pd.DataFrame) -> pd.DataFrame:
    df["final_class"] = df["effective_label"].map(LABEL_TO_CLASS)
    unmapped = df[df["final_class"].isna()]
    if not unmapped.empty:
        raise ValueError(
            f"Labels non mappes: {list(unmapped['effective_label'].unique())}. "
            f"Ajoute-les dans LABEL_TO_CLASS."
        )
    return df


def split_held_out_test(df: pd.DataFrame):
    """Meme logique que train_model.py : split definitif par session,
    stratifie par classe, jamais reutilise avant l'evaluation finale."""
    X = df[FEATURE_COLUMNS]
    y = df["final_class"]
    groups = df["session_id"]

    min_sessions_per_class = df.groupby("final_class")["session_id"].nunique().min()
    n_splits = max(2, round(1 / HELD_OUT_TEST_RATIO))
    n_splits = min(n_splits, min_sessions_per_class)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(sgkf.split(X, y, groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    overlap = set(train_df["session_id"]) & set(test_df["session_id"])
    if overlap:
        raise RuntimeError(f"ERREUR: sessions partagees entre train et test: {overlap}")

    print(f"Train: {len(train_df)} fenetres ({train_df['session_id'].nunique()} sessions)")
    print(f"Test:  {len(test_df)} fenetres ({test_df['session_id'].nunique()} sessions)\n")

    return train_df, test_df


def main():
    if not os.path.exists(INPUT_PARQUET):
        raise FileNotFoundError(f"'{INPUT_PARQUET}' introuvable.")

    df = pd.read_parquet(INPUT_PARQUET)
    df = apply_class_mapping(df)
    print(f"Dataset charge: {len(df)} fenetres, {df['final_class'].nunique()} classes.\n")

    train_df, test_df = split_held_out_test(df)

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["final_class"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["final_class"]

    models = build_models()
    results = []

    for name, model in models.items():
        print("=" * 60)
        print(f"Modele : {name}")
        print("=" * 60)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")

        print(f"Accuracy : {accuracy:.3f}")
        print(f"F1-macro : {f1_macro:.3f}")
        print(f"F1-weighted : {f1_weighted:.3f}\n")
        print(classification_report(y_test, y_pred, zero_division=0))

        results.append({
            "modele": name,
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
        })

    results_df = pd.DataFrame(results).sort_values("f1_macro", ascending=False)

    print("=" * 60)
    print("TABLEAU COMPARATIF (trie par f1_macro decroissant)")
    print("=" * 60)
    print(results_df.to_string(index=False))

    results_df.to_csv("model_comparison_results.csv", index=False)
    print(f"\nResultats sauvegardes: {os.path.abspath('model_comparison_results.csv')}")

    best = results_df.iloc[0]
    print(f"\nMeilleur modele: {best['modele']} "
          f"(accuracy={best['accuracy']:.3f}, f1_macro={best['f1_macro']:.3f})")


if __name__ == "__main__":
    main()