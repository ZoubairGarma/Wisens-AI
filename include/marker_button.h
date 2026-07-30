/**
 * @file    marker_button.h
 * @brief   Bouton marqueur pour labellisation en temps réel des scénarios
 *          de transition (S4 entrée, S5 sortie, S6 changement posture).
 *
 * Principe :
 *   Le bouton IO38 (déjà câblé sur le T-Beam, aucun câblage supplémentaire
 *   nécessaire) fait avancer un état courant dans un cycle fixe :
 *
 *       EMPTY -> TRANSITION -> STABLE -> EMPTY -> TRANSITION -> ...
 *
 *   Cet état courant est lu par acquisition_manager/main pour étiqueter
 *   CHAQUE mesure CSI au moment où elle est capturée, directement dans
 *   le firmware. Plus besoin de recouper des timestamps approximatifs
 *   en post-traitement : le label est exact, généré en temps réel.
 *
 * Protocole d'utilisation (exemple S4 - entrée dans la zone) :
 *   1. Avant de lancer la capture, appuyer sur IO38 jusqu'à revenir sur
 *      EMPTY (l'état de depart doit toujours etre EMPTY).
 *   2. Lancer collect_csi.py.
 *   3. Rester hors de la piece (etat EMPTY, ~10s).
 *   4. Appuyer sur IO38 -> etat passe a TRANSITION -> entrer dans la piece.
 *   5. Des que la position stable est atteinte, appuyer sur IO38 ->
 *      etat passe a STABLE -> rester immobile.
 *   6. A la fin du cycle (45s), appuyer sur IO38 -> retour a EMPTY pour
 *      le cycle suivant.
 */

#ifndef MARKER_BUTTON_H
#define MARKER_BUTTON_H

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MARKER_STATE_EMPTY = 0,
    MARKER_STATE_TRANSITION,
    MARKER_STATE_STABLE,
} marker_state_t;

/**
 * @brief Initialise le GPIO du bouton (pull-up + interruption front descendant)
 *        et l'état initial (MARKER_STATE_EMPTY).
 *
 * À appeler une seule fois, avant acquisition_manager_start().
 */
esp_err_t marker_button_init(void);

/**
 * @brief Retourne l'état courant (thread-safe, simple lecture volatile).
 *
 * À appeler depuis on_csi_sample() pour étiqueter chaque mesure.
 */
marker_state_t marker_button_get_state(void);

/**
 * @brief Convertit un état en chaîne lisible, pour l'écriture CSV.
 */
const char *marker_button_state_to_string(marker_state_t state);

#ifdef __cplusplus
}
#endif

#endif /* MARKER_BUTTON_H */