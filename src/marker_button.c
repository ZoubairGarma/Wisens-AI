/**
 * @file    marker_button.c
 * @brief   Implémentation du bouton marqueur pour WISENS-AI.
 */

#include "marker_button.h"

#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "wisens_marker";

/** IO38 : bouton déjà câblé sur le T-Beam (actif bas, GND quand pressé). */
#define MARKER_BUTTON_GPIO      GPIO_NUM_38

/** Anti-rebond : ignore les interruptions successives trop rapprochées
 *  (un vrai appui humain ne peut pas générer deux fronts utiles en
 *  moins de 200ms ; les rebonds mécaniques eux arrivent en <10ms). */
#define MARKER_DEBOUNCE_US      500000

static volatile marker_state_t s_state = MARKER_STATE_EMPTY;
static volatile int64_t s_last_press_us = 0;

/** Handle de la tâche à notifier pour affichage (hors contexte ISR). */
static TaskHandle_t s_notify_task_handle = NULL;

static void marker_button_isr_handler(void *arg);
static void marker_notify_task(void *arg);
static marker_state_t next_state(marker_state_t current);

esp_err_t marker_button_init(void)
{
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << MARKER_BUTTON_GPIO),
        .mode         = GPIO_MODE_INPUT,
        .pull_up_en   = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type    = GPIO_INTR_NEGEDGE,   /* front descendant = appui */
    };

    esp_err_t err = gpio_config(&io_conf);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "gpio_config a echoue: %s", esp_err_to_name(err));
        return err;
    }

    /* Service ISR partagé : peut déjà être installé par un autre module,
     * ESP_ERR_INVALID_STATE dans ce cas est normal et sans gravité. */
    err = gpio_install_isr_service(0);
    if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(TAG, "gpio_install_isr_service a echoue: %s", esp_err_to_name(err));
        return err;
    }

    err = gpio_isr_handler_add(MARKER_BUTTON_GPIO, marker_button_isr_handler, NULL);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "gpio_isr_handler_add a echoue: %s", esp_err_to_name(err));
        return err;
    }

    /* Tâche légère, uniquement pour afficher la confirmation du
     * changement d'état sur le moniteur série (utile a l'operateur
     * pendant la capture, aucun impact sur les donnees CSI elles-memes
     * puisque marker_button_get_state() est lu directement par
     * on_csi_sample()). */
    BaseType_t ok = xTaskCreate(marker_notify_task, "marker_notify",
                                 2048, NULL, 3, &s_notify_task_handle);
    if (ok != pdPASS) {
        ESP_LOGE(TAG, "Echec creation tache marker_notify");
        return ESP_ERR_NO_MEM;
    }

    s_state = MARKER_STATE_EMPTY;
    ESP_LOGI(TAG, "Bouton marqueur initialise sur GPIO%d, etat initial = EMPTY",
             MARKER_BUTTON_GPIO);

    return ESP_OK;
}

marker_state_t marker_button_get_state(void)
{
    return s_state;
}

const char *marker_button_state_to_string(marker_state_t state)
{
    switch (state) {
        case MARKER_STATE_EMPTY:      return "empty";
        case MARKER_STATE_TRANSITION: return "transition";
        case MARKER_STATE_STABLE:     return "stable";
        default:                      return "unknown";
    }
}

/* ---------------------------------------------------------------------- */
/*                          Implémentation privée                         */
/* ---------------------------------------------------------------------- */

static marker_state_t next_state(marker_state_t current)
{
    switch (current) {
        case MARKER_STATE_EMPTY:      return MARKER_STATE_TRANSITION;
        case MARKER_STATE_TRANSITION: return MARKER_STATE_STABLE;
        case MARKER_STATE_STABLE:     return MARKER_STATE_EMPTY;
        default:                      return MARKER_STATE_EMPTY;
    }
}

/**
 * @brief ISR déclenchée sur appui du bouton.
 *
 * @note Reste volontairement minimale (anti-rebond + changement d'état,
 *       pas de printf ici) car appelée en contexte d'interruption.
 *       La notification à la tâche d'affichage se fait via
 *       vTaskNotifyGiveFromISR, mécanisme sûr en ISR.
 */
static void IRAM_ATTR marker_button_isr_handler(void *arg)
{
    int64_t now_us = esp_timer_get_time();

    if ((now_us - s_last_press_us) < MARKER_DEBOUNCE_US) {
        return;   /* rebond ignore */
    }
    s_last_press_us = now_us;

    s_state = next_state(s_state);

    if (s_notify_task_handle != NULL) {
        BaseType_t higher_priority_woken = pdFALSE;
        vTaskNotifyGiveFromISR(s_notify_task_handle, &higher_priority_woken);
        if (higher_priority_woken) {
            portYIELD_FROM_ISR();
        }
    }
}

/**
 * @brief Tâche d'affichage : reveillee par l'ISR, imprime le nouvel etat
 *        sur le port serie pour que l'operateur ait une confirmation
 *        visuelle immediate pendant la capture.
 */
static void marker_notify_task(void *arg)
{
    while (1) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        ESP_LOGI(TAG, ">>> Etat marqueur change: %s <<<",
                 marker_button_state_to_string(marker_button_get_state()));
    }
}