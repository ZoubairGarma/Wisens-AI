/**
 * @file    traffic_generator.h
 * @brief   API publique du générateur de trafic Wi-Fi pour WISENS-AI.
 *
 * Rôle du module :
 *   Le CSI ne se calcule que sur des trames Wi-Fi effectivement reçues
 *   par l'ESP32. Sans trafic réseau actif, aucune mesure n'est générée.
 *   Ce module fait pinguer en continu la passerelle réseau (routeur/
 *   hotspot) depuis le firmware lui-même, pour garantir un flux régulier
 *   de trames et donc de mesures CSI — sans dépendre d'un PC externe.
 *
 * Correspond à l'étape "Émission Wi-Fi contrôlée" de l'architecture
 * décrite dans le dossier projet (chapitre 4.1).
 *
 * Ce module NE fait PAS :
 *   - de capture CSI (voir acquisition_manager),
 *   - de traitement de signal.
 * Il se contente de provoquer du trafic réseau régulier.
 */

#ifndef TRAFFIC_GENERATOR_H
#define TRAFFIC_GENERATOR_H

#include <stdint.h>
#include <stdbool.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Démarre le ping continu vers la passerelle réseau.
 *
 * @note Nécessite que le Wi-Fi soit déjà connecté (l'IP de la passerelle
 *       doit être disponible via wifi_manager_get_gateway_ip()).
 *
 * @param interval_ms Intervalle entre deux pings, en millisecondes.
 *                     Une valeur plus faible donne un flux de mesures
 *                     CSI plus dense, au prix d'un peu plus de trafic
 *                     réseau et de charge CPU.
 *
 * @return ESP_OK si le générateur a démarré, ESP_ERR_INVALID_STATE si
 *         le Wi-Fi n'est pas connecté, ou une erreur du sous-système ping.
 */
esp_err_t traffic_generator_start(uint32_t interval_ms);

/**
 * @brief Arrête le générateur de trafic et libère la session de ping.
 *
 * @return ESP_OK si l'arrêt s'est bien passé.
 */
esp_err_t traffic_generator_stop(void);

/**
 * @brief Indique si le générateur de trafic est actuellement actif.
 */
bool traffic_generator_is_running(void);

#ifdef __cplusplus
}
#endif

#endif /* TRAFFIC_GENERATOR_H */