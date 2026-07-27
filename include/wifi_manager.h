/**
 * @file    wifi_manager.h
 * @brief   API publique du module de gestion Wi-Fi pour WISENS-AI.
 *
 * Rôle du module :
 *   - Initialiser la pile Wi-Fi ESP-IDF (NVS, netif, event loop).
 *   - Se connecter en mode Station (STA) au réseau configuré.
 *   - Gérer automatiquement les tentatives de reconnexion.
 *   - Exposer un état clair pour les modules d'acquisition en aval
 *     (le module CSI/RSSI ne doit démarrer QUE si le Wi-Fi est connecté).
 *
 * Ce module NE fait PAS :
 *   - de traitement de signal,
 *   - d'acquisition RSSI/CSI,
 *   - de logique applicative.
 * Il fournit uniquement une connexion réseau fiable et un état exploitable.
 */

#ifndef WIFI_MANAGER_H
#define WIFI_MANAGER_H

#include <stdbool.h>
#include "esp_err.h"
#include "freertos/FreeRTOS.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief États possibles du module Wi-Fi.
 */
typedef enum {
    WIFI_MGR_STATE_UNINITIALIZED = 0, /**< init() n'a pas encore été appelé. */
    WIFI_MGR_STATE_CONNECTING,        /**< Tentative de connexion en cours. */
    WIFI_MGR_STATE_CONNECTED,         /**< Connecté, IP obtenue. */
    WIFI_MGR_STATE_FAILED              /**< Échec après épuisement des retries. */
} wifi_manager_state_t;

/**
 * @brief Initialise la pile réseau et Wi-Fi, puis lance la connexion STA.
 *
 * Effectue, dans l'ordre :
 *   1. nvs_flash_init()
 *   2. esp_netif_init() + création de la boucle d'événements par défaut
 *   3. création de l'interface netif STA par défaut
 *   4. configuration SSID/mot de passe depuis wisens_config.h
 *   5. démarrage de esp_wifi en mode STA
 *
 * @note Cette fonction est non bloquante : elle lance la connexion
 *       mais ne l'attend pas. Utiliser wifi_manager_wait_connected()
 *       pour bloquer jusqu'à confirmation.
 *
 * @return ESP_OK en cas de succès, code d'erreur ESP-IDF sinon.
 */
esp_err_t wifi_manager_init(void);

/**
 * @brief Bloque jusqu'à connexion effective (IP obtenue) ou échec définitif.
 *
 * @param timeout_ticks Délai maximum d'attente (portTICK_PERIOD_MS).
 *                      Utiliser portMAX_DELAY pour attendre indéfiniment.
 *
 * @return true  si connecté avec succès avant le timeout,
 *         false si échec, timeout, ou module non initialisé.
 */
bool wifi_manager_wait_connected(TickType_t timeout_ticks);

/**
 * @brief Retourne l'état courant du module Wi-Fi.
 *
 * Fonction thread-safe, utilisable depuis n'importe quelle tâche
 * (ex : le module d'acquisition peut vérifier l'état avant de lire le canal).
 */
wifi_manager_state_t wifi_manager_get_state(void);

/**
 * @brief Force une reconnexion (arrêt puis relance de la tentative STA).
 *
 * Utile si le module d'acquisition détecte une perte de qualité de
 * signal anormale et souhaite provoquer un cycle de reconnexion propre.
 *
 * @return ESP_OK si la demande a été acceptée.
 */
esp_err_t wifi_manager_reconnect(void);

#ifdef __cplusplus
}
#endif

#endif /* WIFI_MANAGER_H */