#include "csv_exporter.h"

#include <stdio.h>
#include "esp_log.h"


static const char *TAG = "CSV_EXPORT";


void csv_exporter_init(void)
{
    /*
     * Une seule fois au démarrage.
     * Python pourra reconnaître cette ligne.
     */

   

    ESP_LOGI(TAG,"CSV serial exporter ready");
}



void csv_exporter_send_sample(
        const wisens_csi_sample_t *sample)
{

    if(sample == NULL)
        return;


    /*
     * Format :
     *
     * CSV,
     * timestamp,
     * RSSI,
     * channel,
     * MAC,
     * length,
     * CSI values
     */


    printf(
        "CSV,%lld,%d,%u,"
        "%02x:%02x:%02x:%02x:%02x:%02x,"
        "%u,",
        (long long)sample->timestamp_us,
        sample->rssi,
        sample->channel,

        sample->mac[0],
        sample->mac[1],
        sample->mac[2],
        sample->mac[3],
        sample->mac[4],
        sample->mac[5],

        sample->csi_len
    );


    for(int i=0;i<sample->csi_len;i++)
    {

        printf("%d",
               sample->csi_data[i]);


        if(i < sample->csi_len-1)
            printf(";");
    }


    printf("\n");
}