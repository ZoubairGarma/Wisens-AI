/**
 * @file    acquisition_manager.h
 * @brief   API publique du module d'acquisition CSI/RSSI pour WISENS-AI.
 *
 * Rôle du module :
 *   - Activer la capture CSI côté driver Wi-Fi ESP-IDF.
 *   - Recevoir chaque trame CSI via le callback bas niveau du driver.
 *   - Exposer ces mesures à l'application sous forme d'une structure
 *     simple et horodatée, via un callback utilisateur.
 *
 * Ce module NE fait PAS :
 *   - de traitement de signal (filtrage, normalisation, features),
 *   - de stockage / export CSV-JSON,
 *   - de classification IA.
 * Il fournit uniquement des mesures brutes, propres et horodatées,
 * conformément à la structure décrite dans le dossier projet
 * (section "Structure d'une mesure").
 *
 * @warning Prérequis obligatoire : le Wi-Fi doit être connecté
 *          (wifi_manager_get_state() == WIFI_MGR_STATE_CONNECTED)
 *          avant d'appeler acquisition_manager_init(). La capture CSI
 *          s'appuie sur le driver Wi-Fi déjà démarré.
 */

#ifndef ACQUISITION_MANAGER_H
#define ACQUISITION_MANAGER_H

#include <stdint.h>
#include <stdbool.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Une mesure CSI brute, horodatée.
 *
 * @note Le pointeur csi_data n'est valide que pendant l'exécution du
 *       callback utilisateur enregistré via
 *       acquisition_manager_register_callback(). Si les données doivent
 *       être conservées au-delà (ex: pour traitement asynchrone), il faut
 *       les copier (memcpy) avant de retourner du callback : le driver
 *       Wi-Fi réutilise ce buffer immédiatement après.
 */
typedef struct {
    int64_t  timestamp_us;   /**< Horodatage local (esp_timer_get_time()). */
    int8_t   rssi;           /**< Puissance du signal reçu, en dBm. */
    uint8_t  channel;        /**< Canal Wi-Fi sur lequel la trame a été reçue. */
    uint8_t  mac[6];         /**< Adresse MAC source de la trame. */
    uint16_t csi_len;        /**< Nombre d'octets valides dans csi_data. */
    const int8_t *csi_data;  /**< Buffer brut CSI (paires I/Q entrelacées). */
} wisens_csi_sample_t;

/**
 * @brief Signature du callback utilisateur recevant chaque mesure CSI.
 *
 * Appelé depuis le contexte du driver Wi-Fi : garder ce callback court
 * et non bloquant (pas de log verbeux ni d'allocation lourde ici).
 */
typedef void (*acquisition_data_cb_t)(const wisens_csi_sample_t *sample);

/**
 * @brief Initialise la capture CSI (configuration du driver Wi-Fi).
 *
 * N'active pas encore la capture : voir acquisition_manager_start().
 * Doit être appelé une seule fois, après connexion Wi-Fi effective.
 *
 * @return ESP_OK en cas de succès, code d'erreur ESP-IDF sinon
 *         (notamment ESP_ERR_INVALID_STATE si le Wi-Fi n'est pas prêt,
 *         ou une erreur du driver si l'option CSI n'a pas été activée
 *         dans menuconfig : Component config -> Wi-Fi -> WiFi CSI).
 */
esp_err_t acquisition_manager_init(void);

/**
 * @brief Enregistre le callback utilisateur recevant les mesures.
 *
 * À appeler avant acquisition_manager_start().
 *
 * @param cb Fonction appelée pour chaque trame CSI reçue.
 */
void acquisition_manager_register_callback(acquisition_data_cb_t cb);

/**
 * @brief Démarre effectivement la capture CSI.
 *
 * @return ESP_OK si la capture a démarré correctement.
 */
esp_err_t acquisition_manager_start(void);

/**
 * @brief Arrête la capture CSI (sans désinitialiser le module).
 *
 * @return ESP_OK si l'arrêt a réussi.
 */
esp_err_t acquisition_manager_stop(void);

/**
 * @brief Indique si la capture est actuellement active.
 */
bool acquisition_manager_is_running(void);

#ifdef __cplusplus
}
#endif

#endif /* ACQUISITION_MANAGER_H */