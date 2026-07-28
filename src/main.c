/**
 * @file    main.c
 * @brief   Point d'entree WISENS-AI - etape 1 : connexion Wi-Fi.
 *
 * Ce fichier ne fait volontairement QUE l'initialisation Wi-Fi pour
 * l'instant. Le module d'acquisition RSSI/CSI viendra se brancher
 * ici une fois la connexion confirmee (voir TODO en bas de fichier).
 */

#include "esp_log.h"
#include <inttypes.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "wifi_manager.h"
#include "acquisition_manager.h"
#include "traffic_generator.h"
#include "wisens_config.h"

static const char *TAG = "wisens_main";

/** Compteur de mesures reçues, mis à jour depuis le callback CSI.
 *  volatile car modifié dans un contexte différent de app_main(). */
static volatile uint32_t s_sample_count = 0;

/**
 * @brief Callback appelé pour chaque mesure CSI reçue.
 *
 * Affiche 1 mesure sur WISENS_LOG_SAMPLE_EVERY_N avec le RSSI (validation
 * rapide), et 1 mesure sur WISENS_LOG_RAW_CSI_EVERY_N avec les vraies
 * valeurs CSI brutes (paires I/Q + amplitude), pour vérifier que les
 * données ont un sens physique avant de passer a l'export.
 *
 * Reste volontairement léger : pas de traitement lourd ici. Dans une
 * prochaine étape, ce callback copiera l'échantillon vers une file
 * FreeRTOS (queue) pour un traitement/export dans une tâche séparée,
 * sans jamais bloquer le contexte du driver Wi-Fi.
 */
static void on_csi_sample(const wisens_csi_sample_t *sample)
{
    /* Filtre de taille : on ne garde que les mesures de taille fixe
     * attendue, pour garantir un vecteur de features homogene dans le
     * futur dataset (voir wisens_config.h : WISENS_CSI_EXPECTED_LEN). */
    if (sample->csi_len != WISENS_CSI_EXPECTED_LEN) {
        return;
    }

    s_sample_count++;

    if ((s_sample_count % WISENS_LOG_SAMPLE_EVERY_N) == 0) {
        ESP_LOGI(TAG,
                 "Mesure #%" PRIu32 " | t=%lld us | rssi=%d dBm | "
                 "canal=%u | csi_len=%u octets | mac=%02x:%02x:%02x:%02x:%02x:%02x",
                 s_sample_count,
                 (long long)sample->timestamp_us,
                 sample->rssi,
                 sample->channel,
                 sample->csi_len,
                 sample->mac[0], sample->mac[1], sample->mac[2],
                 sample->mac[3], sample->mac[4], sample->mac[5]);
    }

    /* Inspection des valeurs CSI brutes (moins frequent, plus verbeux). */
    if ((s_sample_count % WISENS_LOG_RAW_CSI_EVERY_N) == 0 && sample->csi_len >= 20) {
        char buf[160];
        int off = 0;
        off += snprintf(buf + off, sizeof(buf) - off, "CSI brut (10 premieres paires I/Q): ");
        for (int i = 0; i < 10 && (i * 2 + 1) < sample->csi_len; i++) {
            int8_t I = sample->csi_data[i * 2];
            int8_t Q = sample->csi_data[i * 2 + 1];
            off += snprintf(buf + off, sizeof(buf) - off, "(%d,%d) ", I, Q);
        }
        ESP_LOGI(TAG, "%s", buf);

        int8_t I0 = sample->csi_data[0];
        int8_t Q0 = sample->csi_data[1];
        float amplitude0 = sqrtf((float)(I0 * I0 + Q0 * Q0));
        ESP_LOGI(TAG, "Amplitude sous-porteuse 0 = %.2f", amplitude0);
    }
}

void app_main(void)
{
    ESP_LOGI(TAG, "=== Demarrage WISENS-AI ===");

    esp_err_t err = wifi_manager_init();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Echec critique d'initialisation Wi-Fi (%s), arret.",
                 esp_err_to_name(err));
        return;
    }

    ESP_LOGI(TAG, "Attente de connexion Wi-Fi...");
    bool connected = wifi_manager_wait_connected(pdMS_TO_TICKS(15000));

    if (!connected) {
        ESP_LOGE(TAG, "Impossible de se connecter au reseau Wi-Fi. "
                       "Verifie SSID/mot de passe dans wisens_config.h");
        return;
    }

    ESP_LOGI(TAG, "Wi-Fi connecte, etat = %d", wifi_manager_get_state());

    /* Generateur de trafic : demarre AVANT l'acquisition, car sans trafic
     * reseau actif, tres peu de trames sont recues et donc tres peu de
     * mesures CSI sont capturees (voir "Emission Wi-Fi controlee",
     * dossier projet chapitre 4.1). */
    err = traffic_generator_start(WISENS_TRAFFIC_PING_INTERVAL_MS);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "Echec demarrage generateur de trafic (%s). "
                       "L'acquisition continuera mais avec un flux "
                       "de mesures probablement tres faible.",
                 esp_err_to_name(err));
    }

    /* Acquisition CSI : uniquement une fois le Wi-Fi confirme connecte. */
    acquisition_manager_register_callback(on_csi_sample);

    err = acquisition_manager_init();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Echec init acquisition (%s). "
                       "Verifie l'option CSI dans menuconfig.",
                 esp_err_to_name(err));
        return;
    }

    err = acquisition_manager_start();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Echec demarrage acquisition (%s)", esp_err_to_name(err));
        return;
    }

    ESP_LOGI(TAG, "Acquisition CSI active, en attente de mesures...");

    /* -----------------------------------------------------------------
     * TODO (prochaine etape du stage) :
     *   - Copier chaque wisens_csi_sample_t vers une queue FreeRTOS
     *     depuis on_csi_sample() (sans bloquer le contexte Wi-Fi).
     *   - Une tache dediee consomme la queue et envoie les mesures
     *     vers le PC (UART ou reseau), au format CSV/JSON decrit
     *     dans le dossier projet (section "Structure d'une mesure").
     * ------------------------------------------------------------- */

    while (1) {
        ESP_LOGI(TAG, "WISENS-AI actif | wifi=%d | csi_running=%d | echantillons=%u",
                 wifi_manager_get_state(),
                 acquisition_manager_is_running(),
                 (unsigned int)s_sample_count);
        vTaskDelay(pdMS_TO_TICKS(5000));
    }
}