/**
 * @file    acquisition_manager.c
 * @brief   ImplÃ©mentation du module d'acquisition CSI/RSSI pour WISENS-AI.
 */

#include "acquisition_manager.h"
#include "wisens_config.h"

#include <string.h>

#include "esp_wifi.h"
#include "esp_log.h"
#include "esp_timer.h"

/* ---------------------------------------------------------------------- */
/*                          Variables privÃ©es (static)                    */
/* ---------------------------------------------------------------------- */

static const char *TAG = WISENS_LOG_TAG_ACQ;

static acquisition_data_cb_t s_user_callback = NULL;
static volatile bool s_running = false;
static volatile bool s_initialized = false;

/* ---------------------------------------------------------------------- */
/*                          Prototypes privÃ©s                             */
/* ---------------------------------------------------------------------- */

static void wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info);

/* ---------------------------------------------------------------------- */
/*                          ImplÃ©mentation publique                       */
/* ---------------------------------------------------------------------- */

esp_err_t acquisition_manager_init(void)
{
    if (s_initialized) {
        ESP_LOGW(TAG, "acquisition_manager deja initialise, appel ignore");
        return ESP_OK;
    }

    /* Configuration de la capture CSI.
     * Les champs disponibles dÃ©pendent de la version d'ESP-IDF ;
     * ceux ci-dessous couvrent la configuration standard pour ESP32. */
    wifi_csi_config_t csi_config = {
        .lltf_en           = WISENS_CSI_ENABLE_LLTF,
        .htltf_en          = WISENS_CSI_ENABLE_HT20,
        .stbc_htltf2_en    = 1,
        .ltf_merge_en      = 1,
        .channel_filter_en = 1,
        .manu_scale        = 0,
        .shift             = 0,
    };

    esp_err_t err = esp_wifi_set_csi_config(&csi_config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_wifi_set_csi_config a echoue: %s. "
                       "Verifie que l'option CSI est activee dans menuconfig "
                       "(Component config -> Wi-Fi -> WiFi CSI).",
                 esp_err_to_name(err));
        return err;
    }

    err = esp_wifi_set_csi_rx_cb(wifi_csi_rx_cb, NULL);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_wifi_set_csi_rx_cb a echoue: %s", esp_err_to_name(err));
        return err;
    }

    s_initialized = true;
    ESP_LOGI(TAG, "Module d'acquisition CSI initialise");

    return ESP_OK;
}

void acquisition_manager_register_callback(acquisition_data_cb_t cb)
{
    s_user_callback = cb;
}

esp_err_t acquisition_manager_start(void)
{
    if (!s_initialized) {
        ESP_LOGE(TAG, "acquisition_manager_start: module non initialise "
                       "(appelle acquisition_manager_init() d'abord)");
        return ESP_ERR_INVALID_STATE;
    }

    esp_err_t err = esp_wifi_set_csi(true);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_wifi_set_csi(true) a echoue: %s", esp_err_to_name(err));
        return err;
    }

    s_running = true;
    ESP_LOGI(TAG, "Capture CSI demarree");

    return ESP_OK;
}

esp_err_t acquisition_manager_stop(void)
{
    if (!s_running) {
        return ESP_OK;
    }

    esp_err_t err = esp_wifi_set_csi(false);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_wifi_set_csi(false) a echoue: %s", esp_err_to_name(err));
        return err;
    }

    s_running = false;
    ESP_LOGI(TAG, "Capture CSI arretee");

    return ESP_OK;
}

bool acquisition_manager_is_running(void)
{
    return s_running;
}

/* ---------------------------------------------------------------------- */
/*                          ImplÃ©mentation privÃ©e                         */
/* ---------------------------------------------------------------------- */

/**
 * @brief Callback bas niveau appelÃ© par le driver Wi-Fi pour chaque trame CSI.
 *
 * @note PlacÃ© en IRAM (IRAM_ATTR) car appelÃ© dans un contexte sensible au
 *       timing par le driver Wi-Fi. Doit rester court : pas de log ESP_LOGI
 *       ici (trop lent / peut manquer de mÃ©moire flash au mauvais moment),
 *       uniquement la construction de la structure et l'appel utilisateur.
 */
static void IRAM_ATTR wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info)
{
    if (info == NULL || info->buf == NULL || s_user_callback == NULL) {
        return;
    }

    wisens_csi_sample_t sample = {
        .timestamp_us = esp_timer_get_time(),
        .rssi         = info->rx_ctrl.rssi,
        .channel      = info->rx_ctrl.channel,
        .csi_len      = info->len,
        .csi_data     = info->buf,
    };
    memcpy(sample.mac, info->mac, sizeof(sample.mac));

    s_user_callback(&sample);
}