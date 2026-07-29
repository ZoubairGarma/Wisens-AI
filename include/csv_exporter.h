#ifndef CSV_EXPORTER_H
#define CSV_EXPORTER_H

#include "acquisition_manager.h"

#ifdef __cplusplus
extern "C" {
#endif


void csv_exporter_init(void);

void csv_exporter_send_sample(
        const wisens_csi_sample_t *sample);


#ifdef __cplusplus
}
#endif

#endif