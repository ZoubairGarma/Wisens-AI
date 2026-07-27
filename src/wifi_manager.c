/**
 * @file    wifi_manager.c
 * @brief   Implémentation du module de gestion Wi-Fi pour WISENS-AI.
 */

#include "wifi_manager.h"
#include "wisens_config.h"

#include <string.h>

#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"

/* ---------------------------------------------------------------------- */
/*                          Variables privées (static)                    */
/* ---------------------------------------------------------------------- */

static const char *TAG = WISENS_LOG_TAG_WIFI;

/** Group d'événements FreeRTOS pour synchroniser connexion / échec. */
static EventGroupHandle_t s_wifi_event_group = NULL;

#define WIFI_CONNECTED_BIT   BIT0
#define WIFI_FAIL_BIT        BIT1

/** Compteur de tentatives de reconnexion. */
static int s_retry_count = 0;

/** État exposé publiquement, protégé par le fait qu'il est écrit
 *  uniquement depuis le handler d'événements (contexte unique). */
static volatile wifi_manager_state_t s_state = WIFI_MGR_STATE_UNINITIALIZED;

static esp_netif_t *s_netif_sta = NULL;

/* ---------------------------------------------------------------------- */
/*                          Prototypes privés                             */
/* ---------------------------------------------------------------------- */

static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                                int32_t event_id, void *event_data);

/* ---------------------------------------------------------------------- */
/*                          Implémentation publique                       */
/* ---------------------------------------------------------------------- */

esp_err_t wifi_manager_init(void)
{
    esp_err_t err;

    /* 1. NVS : requis par le driver Wi-Fi pour stocker calibration/config. */
    err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_LOGW(TAG, "NVS corrompue ou version incompatible, effacement...");
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Echec nvs_flash_init: %s", esp_err_to_name(err));
        return err;
    }

    /* 2. Event group pour synchronisation connexion/échec. */
    s_wifi_event_group = xEventGroupCreate();
    if (s_wifi_event_group == NULL) {
        ESP_LOGE(TAG, "Echec creation event group");
        return ESP_ERR_NO_MEM;
    }

    /* 3. Pile réseau + boucle d'événements par défaut. */
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    s_netif_sta = esp_netif_create_default_wifi_sta();

    /* 4. Initialisation du driver Wi-Fi avec la config par défaut. */
    wifi_init_config_t init_cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&init_cfg));

    /* 5. Enregistrement des handlers d'événements Wi-Fi et IP. */
    ESP_ERROR_CHECK(esp_event_handler_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL));

    /* 6. Configuration STA : SSID / mot de passe / seuil de sécurité. */
    wifi_config_t wifi_config = { 0 };
    strncpy((char *)wifi_config.sta.ssid, WISENS_WIFI_SSID,
            sizeof(wifi_config.sta.ssid) - 1);
    strncpy((char *)wifi_config.sta.password, WISENS_WIFI_PASSWORD,
            sizeof(wifi_config.sta.password) - 1);
    wifi_config.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;
    wifi_config.sta.pmf_cfg.capable = true;
    wifi_config.sta.pmf_cfg.required = false;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));

    s_state = WIFI_MGR_STATE_CONNECTING;
    s_retry_count = 0;

    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "Initialisation terminee, connexion en cours vers SSID '%s'",
             WISENS_WIFI_SSID);

    return ESP_OK;
}

bool wifi_manager_wait_connected(TickType_t timeout_ticks)
{
    if (s_wifi_event_group == NULL) {
        ESP_LOGE(TAG, "wifi_manager_wait_connected: module non initialise");
        return false;
    }

    EventBits_t bits = xEventGroupWaitBits(
        s_wifi_event_group,
        WIFI_CONNECTED_BIT | WIFI_FAIL_BIT,
        pdFALSE,   /* ne pas effacer les bits : get_state() reste cohérent */
        pdFALSE,   /* OR logique : un seul des deux bits suffit */
        timeout_ticks);

    if (bits & WIFI_CONNECTED_BIT) {
        return true;
    }
    /* Soit WIFI_FAIL_BIT est levé, soit timeout : dans les deux cas échec. */
    return false;
}

wifi_manager_state_t wifi_manager_get_state(void)
{
    return s_state;
}

esp_err_t wifi_manager_reconnect(void)
{
    if (s_wifi_event_group == NULL) {
        ESP_LOGE(TAG, "wifi_manager_reconnect: module non initialise");
        return ESP_ERR_INVALID_STATE;
    }

    ESP_LOGI(TAG, "Reconnexion manuelle demandee");
    xEventGroupClearBits(s_wifi_event_group, WIFI_CONNECTED_BIT | WIFI_FAIL_BIT);
    s_retry_count = 0;
    s_state = WIFI_MGR_STATE_CONNECTING;

    return esp_wifi_connect();
}

/* ---------------------------------------------------------------------- */
/*                          Implémentation privée                         */
/* ---------------------------------------------------------------------- */

static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                                int32_t event_id, void *event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        ESP_LOGI(TAG, "Demarrage STA, premiere tentative de connexion");
        esp_wifi_connect();
        return;
    }

    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        if (s_retry_count < WISENS_WIFI_MAX_RETRY) {
            esp_wifi_connect();
            s_retry_count++;
            s_state = WIFI_MGR_STATE_CONNECTING;
            ESP_LOGW(TAG, "Deconnecte, tentative %d/%d",
                     s_retry_count, WISENS_WIFI_MAX_RETRY);
        } else {
            s_state = WIFI_MGR_STATE_FAILED;
            xEventGroupSetBits(s_wifi_event_group, WIFI_FAIL_BIT);
            ESP_LOGE(TAG, "Echec de connexion apres %d tentatives",
                     WISENS_WIFI_MAX_RETRY);
        }
        return;
    }

    if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *ip_evt = (ip_event_got_ip_t *)event_data;
        ESP_LOGI(TAG, "Connecte, IP obtenue: " IPSTR, IP2STR(&ip_evt->ip_info.ip));

        s_retry_count = 0;
        s_state = WIFI_MGR_STATE_CONNECTED;
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
        return;
    }
}