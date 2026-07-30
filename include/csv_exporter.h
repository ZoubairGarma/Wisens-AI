#ifndef CSV_EXPORTER_H
#define CSV_EXPORTER_H

#include "acquisition_manager.h"

#ifdef __cplusplus
extern "C" {
#endif


void csv_exporter_init(void);

/**
 * @brief Envoie une mesure CSI formatee sur le port serie.
 *
 * @param sample       Mesure CSI brute (voir acquisition_manager.h).
 * @param marker_state Etat courant du bouton marqueur ("empty",
 *                      "transition" ou "stable"), lu via
 *                      marker_button_get_state() +
 *                      marker_button_state_to_string() cote appelant.
 *                      Permet de labelliser chaque mesure en temps reel
 *                      pour les scenarios de transition (S4/S5/S6).
 */
void csv_exporter_send_sample(
        const wisens_csi_sample_t *sample,
        const char *marker_state);


#ifdef __cplusplus
}
#endif

#endif