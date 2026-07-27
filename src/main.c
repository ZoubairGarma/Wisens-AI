/**
 * @file    main.c
 * @brief   Point d'entree WISENS-AI - etape 1 : connexion Wi-Fi.
 *
 * Ce fichier ne fait volontairement QUE l'initialisation Wi-Fi pour
 * l'instant. Le module d'acquisition RSSI/CSI viendra se brancher
 * ici une fois la connexion confirmee (voir TODO en bas de fichier).
 */

#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "wifi_manager.h"
#include "acquisition_manager.h"

static const char *TAG = "wisens_main";

/** Compteur de mesures reçues, mis à jour depuis le callback CSI.
 *  volatile car modifié dans un contexte différent de app_main(). */
static volatile uint32_t s_sample_count = 0;

/**
 * @brief Callback appelé pour chaque mesure CSI reçue.
 *
 * Reste volontairement très léger (juste un compteur) : dans une
 * prochaine étape, ce callback copiera l'échantillon vers une file
 * FreeRTOS (queue) pour un traitement/export dans une tâche séparée,
 * sans jamais bloquer le contexte du driver Wi-Fi.
 */
static void on_csi_sample(const wisens_csi_sample_t *sample)
{
    (void)sample;
    s_sample_count++;
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