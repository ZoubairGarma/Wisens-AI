"""
learning_curve.py - WISENS-AI : Courbe d'apprentissage empirique.

Question posee : combien de sessions supplementaires faudrait-il pour
depasser 0.8 d'accuracy ? Plutot que de deviner, on MESURE comment la
performance evolue en fonction du nombre de sessions d'entrainement
deja disponibles, et on extrapole a partir de la tendance reelle.

Principe :
    Le test reste FIXE (le meme split definitif que d'habitude). On fait
    varier UNIQUEMENT la quantite de sessions utilisees pour
    l'entrainement (25%, 50%, 75%, 100% des sessions de train
    disponibles), et on mesure le f1_macro a chaque etape.

    Si la courbe est encore clairement montante a 100% des donnees
    actuelles -> plus de sessions aidera probablement beaucoup.
    Si la courbe s'aplatit deja (plateau) -> plus de VOLUME n'aidera
    plus significativement, il faudra plutot ameliorer les features
    ou revoir la definition des classes (C1/C2, cf. diagnostic
    precedent).

Usage:
    python learning_curve.py
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
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

# Meilleur reglage trouve par tune_knn.py -- A AJUSTER si tune_knn.py a
# trouve un meilleur k que 5 (mettre a jour ici avant de lancer).
BEST_K = 5
BEST_WEIGHTS = "uniform"

# Fractions de SESSIONS de train testees (pas de fenetres -- cf. note
# en tete de fichier sur l'importance de raisonner en sessions).
SESSION_FRACTIONS = [0.25, 0.4, 0.55, 0.7, 0.85, 1.0]


def apply_class_mapping(df: pd.DataFrame) -> pd.DataFrame:
    df["final_class"] = df["effective_label"].map(LABEL_TO_CLASS)
    unmapped = df[df["final_class"].isna()]
    if not unmapped.empty:
        raise ValueError(f"Labels non mappes: {list(unmapped['effective_label'].unique())}.")
    return df


def split_held_out_test(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y = df["final_class"]
    groups = df["session_id"]

    min_sessions_per_class = df.groupby("final_class")["session_id"].nunique().min()
    n_splits = max(2, round(1 / HELD_OUT_TEST_RATIO))
    n_splits = min(n_splits, min_sessions_per_class)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(sgkf.split(X, y, groups))

    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def subsample_sessions(train_df: pd.DataFrame, fraction: float, rng) -> pd.DataFrame:
    """Sous-echantillonne le TRAIN par SESSION COMPLETE (pas par fenetre),
    en gardant un nombre proportionnel de sessions de CHAQUE classe --
    sinon reduire le volume pourrait faire disparaitre une classe rare
    entierement avant les autres, faussant la courbe."""
    kept_sessions = []
    for cls, group in train_df.groupby("final_class"):
        sessions = group["session_id"].unique()
        n_keep = max(1, round(len(sessions) * fraction))
        chosen = rng.choice(sessions, size=n_keep, replace=False)
        kept_sessions.extend(chosen)

    return train_df[train_df["session_id"].isin(kept_sessions)]


def main():
    if not os.path.exists(INPUT_PARQUET):
        raise FileNotFoundError(f"'{INPUT_PARQUET}' introuvable.")

    df = pd.read_parquet(INPUT_PARQUET)
    df = apply_class_mapping(df)

    train_df, test_df = split_held_out_test(df)
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["final_class"]

    n_total_train_sessions = train_df["session_id"].nunique()
    print(f"Sessions de train disponibles (100%): {n_total_train_sessions}")
    print(f"Sessions de test (fixe, jamais modifie): {test_df['session_id'].nunique()}\n")

    rng = np.random.default_rng(RANDOM_STATE)
    results = []
    N_REPEATS = 8   # plusieurs tirages par fraction, pour lisser le bruit
                     # (avec seulement 40 sessions, un seul tirage par
                     # fraction est tres sensible a quelle session precise
                     # tombe dedans -- voir l'accident au point 37/40)

    for fraction in SESSION_FRACTIONS:
        fraction_accuracies = []
        fraction_f1_macros = []
        n_sessions_used = None

        for repeat in range(N_REPEATS):
            subset = subsample_sessions(train_df, fraction, rng)
            n_sessions_used = subset["session_id"].nunique()

            X_train, y_train = subset[FEATURE_COLUMNS], subset["final_class"]

            model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=BEST_K, weights=BEST_WEIGHTS)),
            ])
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            fraction_accuracies.append(accuracy_score(y_test, y_pred))
            fraction_f1_macros.append(f1_score(y_test, y_pred, average="macro"))

        mean_acc = np.mean(fraction_accuracies)
        std_acc = np.std(fraction_accuracies)
        mean_f1 = np.mean(fraction_f1_macros)
        std_f1 = np.std(fraction_f1_macros)

        results.append({
            "fraction_train": fraction,
            "n_sessions_train": n_sessions_used,
            "accuracy_mean": mean_acc,
            "accuracy_std": std_acc,
            "f1_macro_mean": mean_f1,
            "f1_macro_std": std_f1,
        })

        print(f"~{n_sessions_used:3d} sessions -> "
              f"accuracy={mean_acc:.3f}(+/-{std_acc:.3f}) | "
              f"f1_macro={mean_f1:.3f}(+/-{std_f1:.3f})  "
              f"[{N_REPEATS} tirages]")

    results_df = pd.DataFrame(results)
    results_df.to_csv("learning_curve_results.csv", index=False)
    print(f"\nResultats sauvegardes: {os.path.abspath('learning_curve_results.csv')}")

    # Tendance sur l'ENSEMBLE des points (regression lineaire simple),
    # pas juste les deux derniers -- plus robuste au bruit residuel.
    x = results_df["n_sessions_train"].to_numpy()
    y = results_df["f1_macro_mean"].to_numpy()
    slope = np.polyfit(x, y, 1)[0]

    print("\n" + "=" * 60)
    print("INTERPRETATION (tendance sur l'ensemble des points)")
    print("=" * 60)
    print(f"Pente moyenne (regression lineaire): {slope:+.4f} f1_macro par session\n")

    plateau = results_df["f1_macro_mean"].iloc[-1] - results_df["f1_macro_mean"].iloc[len(results_df)//2]
    if slope > 0.005:
        print("-> Tendance globale clairement montante : ajouter des "
              "sessions devrait continuer a aider significativement.")
    elif slope > 0.001:
        print("-> Tendance legerement montante mais faible : des gains "
              "restent possibles mais seront modestes par session ajoutee.")
    else:
        print("-> PLATEAU : la performance ne progresse plus significativement "
              "avec le volume de sessions actuel, malgre le bruit visible "
              "point par point. Ajouter des sessions supplementaires du "
              "MEME type ne suffira probablement plus a lui seul pour "
              "atteindre 0.8. Le plafond observe est plus vraisemblablement "
              "du a la definition/chevauchement des classes (C1/C2 "
              "notamment, deja identifie dans les diagnostics precedents) "
              "qu'au manque de volume. Prioriser l'amelioration des "
              "features ou la revision des classes avant de collecter "
              "davantage.")

    print(f"\nRappel : objectif du dossier projet = 80% d'accuracy "
          f"(chapitre 14.2). Etat actuel (100% des sessions): "
          f"{results_df['accuracy_mean'].iloc[-1]:.1%}.")


if __name__ == "__main__":
    main()