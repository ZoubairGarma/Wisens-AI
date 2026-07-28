/**
 * @file    wisens_config.h
 * @brief   Configuration centralisée du projet WISENS-AI.
 *
 * Ce fichier regroupe tous les paramètres modifiables du système :
 * identifiants réseau, paramètres de reconnexion, etc.
 * Il ne doit contenir AUCUNE logique, uniquement des constantes.
 *
 * ATTENTION : ne jamais committer ce fichier avec de vrais identifiants
 * dans un dépôt public. En version finale, préférer NVS ou un
 * provisioning Wi-Fi (SmartConfig / BLE) plutôt qu'un mot de passe en dur.
 */

#ifndef WISENS_CONFIG_H
#define WISENS_CONFIG_H

/* ---------------------------------------------------------------------- */
/*                          Paramètres Wi-Fi (STA)                        */
/* ---------------------------------------------------------------------- */

#define WISENS_WIFI_SSID          "Tunisie_Telecom-2.4G-2B3E"
#define WISENS_WIFI_PASSWORD      "W9ce5014e6"

/** Nombre max de tentatives de reconnexion avant abandon. */
#define WISENS_WIFI_MAX_RETRY     5

/** Canal Wi-Fi fixe recommandé pour des mesures RSSI/CSI reproductibles.
 *  0 = laisser l'AP choisir (déconseillé pour les mesures WISENS-AI). */
#define WISENS_WIFI_FIXED_CHANNEL 6

/* ---------------------------------------------------------------------- */
/*                          Paramètres système                            */
/* ---------------------------------------------------------------------- */

/** Tag utilisé pour les logs ESP_LOG du module Wi-Fi. */
#define WISENS_LOG_TAG_WIFI       "wisens_wifi"

/** Tag utilisé pour les logs ESP_LOG du module d'acquisition CSI. */
#define WISENS_LOG_TAG_ACQ        "wisens_acq"

/* ---------------------------------------------------------------------- */
/*                      Paramètres d'acquisition CSI                      */
/* ---------------------------------------------------------------------- */

/** Active la capture des trames LLTF (legacy), recommandé par défaut. */
#define WISENS_CSI_ENABLE_LLTF    1

/** Active le mode "HT" (802.11n), utile si l'AP supporte le 11n. */
#define WISENS_CSI_ENABLE_HT20    1

/** Intervalle entre deux pings du générateur de trafic (ms).
 *  Plus petit = flux CSI plus dense, mais plus de charge réseau/CPU. */
#define WISENS_TRAFFIC_PING_INTERVAL_MS  200

/** Fréquence (en nombre d'échantillons) d'affichage détaillé du RSSI. */
#define WISENS_LOG_SAMPLE_EVERY_N        5

/** Fréquence d'inspection des valeurs CSI brutes (I/Q + amplitude).
 *  Plus rare que le log RSSI car plus verbeux. */
#define WISENS_LOG_RAW_CSI_EVERY_N       25

/** Taille attendue (en octets) du buffer CSI pour une trame HT20 valide.
 *  Les mesures d'une autre taille (trames legacy 802.11b/g par exemple)
 *  sont rejetees, pour garantir un vecteur de features de taille fixe
 *  (indispensable pour l'entrainement des modeles IA : KNN, SVM,
 *  Random Forest, etc. qui exigent des lignes de meme dimension). */
#define WISENS_CSI_EXPECTED_LEN          384

#endif /* WISENS_CONFIG_H */