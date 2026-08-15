"""
cross_zone_generalization.py - WISENS-AI : Test de generalisation
                                inter-zones (zone2 -> zone1).

Question posee : le modele entraine dans une zone physique (piece,
distance, routeur) fonctionne-t-il dans une AUTRE zone, jamais vue a
l'entrainement ? C'est le vrai test de robustesse d'un capteur destine
a etre deploye dans des conditions variees (dossier projet, chapitre 15
"Perspectives d'evolution" : amelioration de la robustesse entre
differentes pieces).

Principe : entrainement sur ZONE2 complete (dataset le plus fourni,
5 sessions/classe), test sur ZONE1 complete (jamais vue a
l'entrainement). Comparaison avec les performances INTRA-zone deja
mesurees, pour quantifier l'ecart de generalisation.

Prerequis :
    - dataset_features.parquet  (zone1, sortie de extract_features.py)
    - dataset2_features.parquet (zone2, sortie de extract_features.py)

Usage:
    python cross_zone_generalization.py
"""

import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score)

ZONE1_PARQUET = "dataset_features.parquet"
ZONE2_PARQUET = "dataset2_features.parquet"

# Mapping unifie : couvre les labels des DEUX zones (zone2 a des labels
# plus precis pour "deux_presences_*", zone1 a l'ancien label unique
# "deux_presences" -- les deux sont couverts ici pour que le meme
# schema de classes s'applique aux deux datasets).
LABEL_TO_CLASS = {
    "piece_vide":         "C0_empty",
    "empty":              "C0_empty",

    "presence_immobile":  "C1_presence_stable",
    "stable":              "C1_presence_stable",
    "deux_presences_immobile": "C1_presence_stable",

    "mouvement_faible":   "C2_low_motion",
    "transition":          "C2_low_motion",
    "deux_presences":      "C2_low_motion",   # ancien label zone1
    "deux_presences_mouvement_faible": "C2_low_motion",

    "mouvement_fort":      "C3_high_motion",
    "deux_presences_mouvement_fort": "C3_high_motion",

    "perturbation_objet":  "C4_object_disturbance",
}

FEATURE_COLUMNS = [
    "moyenne", "variance", "ecart_type", "maximum", "minimum",
    "energie_signal", "nombre_de_pics", "variation_moyenne",
    "stabilite_temporelle", "amplitude_moyenne_csi",
    "subcarrier_std_mean", "subcarrier_std_max",
    "subcarrier_std_std", "subcarrier_std_peakiness",
]

RANDOM_STATE = 42


def load_and_label(path: str, zone_name: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' introuvable. Lance d'abord extract_features.py "
            f"pour {zone_name}."
        )
    df = pd.read_parquet(path)
    df["final_class"] = df["effective_label"].map(LABEL_TO_CLASS)

    unmapped = df[df["final_class"].isna()]
    if not unmapped.empty:
        raise ValueError(
            f"[{zone_name}] Labels non mappes trouves: "
            f"{list(unmapped['effective_label'].unique())}. "
            f"Ajoute-les dans LABEL_TO_CLASS avant de continuer."
        )

    print(f"{zone_name}: {len(df)} fenetres, {df['session_id'].nunique()} sessions")
    return df


def train_on_one_zone_test_on_other(train_df, test_df, train_name, test_name):
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["final_class"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["final_class"]

    model = RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    print("=" * 60)
    print(f"ENTRAINEMENT sur {train_name} (complet) -> TEST sur {test_name} (complet, jamais vu)")
    print("=" * 60)
    print(f"Accuracy globale : {accuracy:.3f}")
    print(f"F1-score (macro) : {f1_macro:.3f}")
    print(f"F1-score (weighted): {f1_weighted:.3f}\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    labels = sorted(set(y_test.unique()) | set(y_train.unique()))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print("Matrice de confusion (lignes = vrai, colonnes = predit):")
    print(cm_df.to_string())
    print()

    out_csv = f"confusion_matrix_{train_name}_to_{test_name}.csv"
    cm_df.to_csv(out_csv)
    print(f"Matrice de confusion sauvegardee: {os.path.abspath(out_csv)}\n")

    return accuracy, f1_macro


def main():
    print("Chargement des deux datasets (zone1 et zone2)...\n")
    zone1 = load_and_label(ZONE1_PARQUET, "zone1")
    zone2 = load_and_label(ZONE2_PARQUET, "zone2")
    print()

    accuracy, f1_macro = train_on_one_zone_test_on_other(zone2, zone1, "zone2", "zone1")

    print("=" * 60)
    print("RESUME -- generalisation inter-zones (zone2 -> zone1)")
    print("=" * 60)
    print(f"Accuracy : {accuracy:.3f}")
    print(f"F1-macro : {f1_macro:.3f}")
    print()
    print("Pour comparaison, resultats INTRA-zone deja mesures precedemment:")
    print(f"  zone1 (out-of-fold, intra-zone)     : accuracy=0.417, f1_macro=0.332")
    print(f"  zone2 (split definitif, intra-zone) : accuracy=0.523, f1_macro=0.522")
    print(f"  zone2 (out-of-fold, intra-zone)      : accuracy=0.596, f1_macro=0.606")
    print()

    if accuracy < 0.417 - 0.05:
        print("INTERPRETATION: la performance inter-zones est nettement "
              "inferieure aux performances intra-zone -> le modele "
              "generalise mal a un environnement non vu.")
    else:
        print("INTERPRETATION: la performance inter-zones reste comparable "
              "aux performances intra-zone -> bon signe de generalisation.")


if __name__ == "__main__":
    main()