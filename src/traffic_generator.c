/**
 * @file    traffic_generator.c
 * @brief   Implémentation du générateur de trafic Wi-Fi pour WISENS-AI.
 */

#include "traffic_generator.h"
#include "wifi_manager.h"
#include "wisens_config.h"

#include "ping/ping_sock.h"
#include "esp_log.h"
#include <inttypes.h>

/* ---------------------------------------------------------------------- */
/*                          Variables privées (static)                    */
/* ---------------------------------------------------------------------- */

static const char *TAG = "wisens_traffic";

static esp_ping_handle_t s_ping_handle = NULL;
static volatile bool s_running = false;

/* ---------------------------------------------------------------------- */
/*                          Implémentation publique                       */
/* ---------------------------------------------------------------------- */

esp_err_t traffic_generator_start(uint32_t interval_ms)
{
    if (s_running) {
        ESP_LOGW(TAG, "Generateur de trafic deja actif, appel ignore");
        return ESP_OK;
    }

    esp_ip4_addr_t gateway_ip;
    esp_err_t err = wifi_manager_get_gateway_ip(&gateway_ip);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Impossible de recuperer l'IP de la passerelle "
                       "(Wi-Fi connecte ?): %s", esp_err_to_name(err));
        return err;
    }

    esp_ping_config_t config = ESP_PING_DEFAULT_CONFIG();
    config.target_addr.type    = ESP_IPADDR_TYPE_V4;
   config.target_addr.u_addr.ip4.addr = gateway_ip.addr;
    config.interval_ms         = interval_ms;
    config.count               = ESP_PING_COUNT_INFINITE;

    /* Aucun callback necessaire : on ne s'interesse pas aux reponses du
     * ping lui-meme, seulement au fait qu'il provoque du trafic reseau
     * (les trames recues declenchent la capture CSI en parallele,
     * via acquisition_manager). */
    esp_ping_callbacks_t cbs = { 0 };

    err = esp_ping_new_session(&config, &cbs, &s_ping_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_ping_new_session a echoue: %s", esp_err_to_name(err));
        return err;
    }

    err = esp_ping_start(s_ping_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_ping_start a echoue: %s", esp_err_to_name(err));
        esp_ping_delete_session(s_ping_handle);
        s_ping_handle = NULL;
        return err;
    }

    s_running = true;
    ESP_LOGI(TAG, "Generateur de trafic demarre (ping passerelle "
                   "toutes les %" PRIu32 " ms)", interval_ms);

    return ESP_OK;
}

esp_err_t traffic_generator_stop(void)
{
    if (!s_running || s_ping_handle == NULL) {
        return ESP_OK;
    }

    esp_err_t err = esp_ping_stop(s_ping_handle);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "esp_ping_stop a echoue: %s", esp_err_to_name(err));
    }

    esp_ping_delete_session(s_ping_handle);
    s_ping_handle = NULL;
    s_running = false;

    ESP_LOGI(TAG, "Generateur de trafic arrete");

    return ESP_OK;
}

bool traffic_generator_is_running(void)
{
    return s_running;
}