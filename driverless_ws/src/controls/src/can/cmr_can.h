
#ifndef CMR_TX_H
#define CMR_TX_H

#ifdef __cplusplus
extern "C" { // Ensures compatibility with C++ compilers
#endif


#include <stdint.h>

int sendControlAction(int16_t frontTorque_mNm, int16_t rearTorque_mNm, uint8_t rackDisplacement_mm);
int sendFinishedCommand();

int cmr_can_rx(int channel, long id, bool verbose);


#ifdef __cplusplus
}
#endif

#endif //CMR_TX_H 


