/**
 * @file cmr_tx.c
 * @brief cmr_can_wrapper
 * 
 * @author Ayush Garg, Vikram Mani, Anthony Yip
 */



#include <canlib.h>
#include <canstat.h>
#include <stdio.h>
#include <signal.h>
#include <errno.h>
#include <unistd.h>
#include <stdbool.h>
#include <stdarg.h>
#include "cmr_can.h"
#include <math.h>
#include <assert.h>
#include <utils/paranoid_assert.h>


#define READ_WAIT_INFINITE (unsigned long)(-1)
static unsigned int msgCounter = 0;

static bool debug = false;

static void debug_printf(const char *format, ...) {
    if (debug) {
        va_list args;
        va_start(args, format);
        vprintf(format, args);
        va_end(args);
    }
}
static unsigned int can_tx_timeout = 30;

canHandle current_can_handle = -1;


static void check(char *id, canStatus stat)
{
    if (stat != canOK) {
        char buf[50];
        buf[0] = '\0';
        canGetErrorText(stat, buf, sizeof(buf));
        debug_printf("%s: failed, stat=%d (%s)\n", id, (int)stat, buf);
    }
}

static void printUsageAndExit(char *prgName)
{
    debug_printf("Usage: '%s <channel>'\n", prgName);
    exit(1);
}

static void sighand(int sig, siginfo_t *info, void *ucontext)
{
    (void)sig;
    (void)info;
    (void)ucontext;
 }

 static void exitOnSignal(int sig, siginfo_t *info, void *ucontext)
 {
     exit(1);
 }
 

static const char *busStatToStr(const unsigned long flag)
{
    switch (flag) {
    case canSTAT_ERROR_PASSIVE:
        return "canSTAT_ERROR_PASSIVE";

    case canSTAT_BUS_OFF:
        return "canSTAT_BUS_OFF";

    case canSTAT_ERROR_WARNING:
        return "canSTAT_ERROR_WARNING";

    case canSTAT_ERROR_ACTIVE:
        return "canSTAT_ERROR_ACTIVE";

    default:
        return "";
    }
}

void notifyCallback(canNotifyData *data)
{
    switch (data->eventType) {
    case canEVENT_STATUS:
        debug_printf("CAN Status Event: %s\n", busStatToStr(data->info.status.busStatus));
        break;

    case canEVENT_ERROR:
        debug_printf("CAN Error Event\n");
        break;

    case canEVENT_TX:
        debug_printf("CAN Tx Event\n");
        break;

    case canEVENT_RX:
        debug_printf("CAN Rx Event\n");
        break;
    }
    return;
}

uint16_t swangle_to_adc(float swangle)
{

    int modulus = 4096;
    float swangle_in_degrees = swangle * 180 / (float)M_PI;
    int zero_adc = 3100;
    int min_adc = 2019;
    int max_adc = modulus + 136;
    float min_deg = -22.10;
    float max_deg = 21.60;
    float adc_deg_ratio = ((float)(max_adc - min_adc)) / ((max_deg - min_deg));
    int desired_adc = (int)(swangle_in_degrees * adc_deg_ratio) + zero_adc;
    paranoid_assert(min_adc < desired_adc && desired_adc < max_adc);
    uint16_t desired_adc_modded = (uint16_t)(desired_adc % modulus);
    return desired_adc_modded;
}

static void cmr_can_error_exit(int signal) {
    canHandle hnd = current_can_handle; 
    canStatus stat;
    if (hnd >= 0) {
        stat = canBusOff(hnd);
        check("canBusOff", stat);
        usleep(50 * 1000); // Sleep just to get the last notification.
        stat = canClose(hnd);
        check("canClose", stat);
        stat = canUnloadLibrary();
        check("canUnloadLibrary", stat);
        return;
    }
}

static int cmr_can_init(int channel, bool verbose) {
    canHandle hnd;
    canStatus stat;

    if(verbose){
        debug_printf("Sending a CAN message on channel %d\n", channel);
    }

    canInitializeLibrary();

    /* Open channel, set parameters and go on bus */
    hnd = canOpenChannel(channel,
                         canOPEN_EXCLUSIVE /*| canOPEN_REQUIRE_EXTENDED*/ | canOPEN_ACCEPT_VIRTUAL);
    if (hnd < 0) {
        debug_printf("canOpenChannel %d", channel);
        check("", hnd);
        return -1;
    }

    stat = canSetBusParams(hnd, canBITRATE_500K, 0, 0, 0, 0, 0);
    check("canSetBusParams", stat);
    if (stat != canOK) {
        return -2;
    }

    if(verbose){
        stat = canSetNotify(hnd, notifyCallback,
            canNOTIFY_RX | canNOTIFY_TX | canNOTIFY_ERROR | canNOTIFY_STATUS |
            canNOTIFY_ENVVAR,
            (char *)0);
        check("canSetNotify", stat);
    }

    stat = canBusOn(hnd);
    check("canBusOn", stat);
    if (stat != canOK) {
        return -3;
    }
    current_can_handle = hnd;
    signal(SIGINT, cmr_can_error_exit);
    return 0;
 }

//CMR TX
//Channel should default to 0
static int cmr_can_tx(int channel, long id, void* msg, unsigned int msgLen, bool verbose){
    canHandle hnd = current_can_handle;
    canStatus stat;
    stat = canWrite(hnd, id, msg, msgLen, 0);
    check("canWrite", stat);
    if (stat != canOK) {
        return -1;
    }
    stat = canWriteSync(hnd, can_tx_timeout);
    check("canWriteSync", stat);
    if (stat != canOK) {
        return -1;
    }
    return 0;
}



int initializeCan(void) {
    return cmr_can_init(0, false);
}


/*
@breif sends control action message

@param frontTorque_mNm
@param rearTorque_mNm
@param rackDisplacement_mm

@returns 0 on success else error occured
*/
int sendControlAction(int16_t frontTorque_mNm, int16_t rearTorque_mNm, uint16_t velocity_rpm /*divide by ~13 for wheelspeed*/, uint16_t rackDisplacement_adc /*2450 -> 4050*/){
    const long controlActionID = 0x190;

    //defines our message
    unsigned char msg[8];
    msg[1] = (unsigned char) (frontTorque_mNm >> 8); //8 MSB of frontTorque
    msg[0] = (unsigned char) frontTorque_mNm; // 8 LSB of frontTorque
    msg[3] = (unsigned char) (rearTorque_mNm >> 8); // 8 MSB of rearTorque
    msg[2] = (unsigned char) rearTorque_mNm; // 8 LSB of rearTorque
    msg[5] = (unsigned char) (velocity_rpm >> 8);
    msg[4] = (unsigned char) velocity_rpm;
    msg[7] = (unsigned char) (rackDisplacement_adc >> 8);
    msg[6] = (unsigned char) rackDisplacement_adc;

    paranoid_assert(current_can_handle >= 0 && "can was not initialized when trying to send control action");

    return cmr_can_tx(0, controlActionID, &msg, 8, false);
}

int sendPIDConstants(float p, float feedforward) {
    long pidID = 0x548;

    //defines our message
    unsigned char msg[8];
    *(float*)msg = p;
    *(float*)(&msg[4]) = feedforward;

    return cmr_can_tx(0, pidID, &msg, 8, false);
}

/*
@breif sends finished mission message

@returns 0 on success else error occured
*/
int sendFinishedCommand(){ 
    long controlActionID = 0x777; // TO IMPLEMENT
    char msg = 0xFF; //finished command
    return cmr_can_tx(0, 0x777, &msg,1, false);
}


//CMR RX
//Channel should default to 0
//UNTESTED
int cmr_can_rx(int channel, long id, bool verbose)
{
    canHandle hnd;
    canStatus stat;
    stat == canOK;
    struct sigaction sigact;


    /* Use sighand and allow SIGINT to interrupt syscalls */
    sigact.sa_flags = SA_SIGINFO;
    sigemptyset(&sigact.sa_mask);
    sigact.sa_sigaction = sighand;
    if (sigaction(SIGINT, &sigact, NULL) != 0) {
        perror("sigaction SIGINT failed");
        return -1;
    }

    if(verbose){
        debug_printf("Reading CAN messages on channel %d\n", channel);
    }

    canInitializeLibrary();

    /* Open channel, set parameters and go on bus */
    hnd = canOpenChannel(channel,
                         canOPEN_EXCLUSIVE | canOPEN_REQUIRE_EXTENDED | canOPEN_ACCEPT_VIRTUAL);
   
    if(verbose) {
        if (hnd < 0) {
        debug_printf("canOpenChannel %d", channel);
        check("", hnd);
        return -1;
    }
    }


    stat = canSetNotify(hnd, notifyCallback,
                        canNOTIFY_RX | canNOTIFY_TX | canNOTIFY_ERROR | canNOTIFY_STATUS |
                            canNOTIFY_ENVVAR,
                        (char *)0);
    check("canSetNotify", stat);

    stat = canSetBusParams(hnd, canBITRATE_500K, 0, 0, 0, 0, 0);
    check("canSetBusParams", stat);
    if (stat != canOK) {
        goto ErrorExit;
    }

    stat = canBusOn(hnd);
    check("canBusOn", stat);
    if (stat != canOK) {
        goto ErrorExit;
    }
    
    bool seen = false;
    do {
        long curId;
        unsigned char msg[8];
        unsigned int dlc;
        unsigned int flag;
        unsigned long time;
    

        stat = canReadWait(hnd, &curId, &msg, &dlc, &flag, &time, READ_WAIT_INFINITE);

        if (stat == canOK) {
            msgCounter++;
            if (flag & canMSG_ERROR_FRAME) {
                debug_printf("(%u) ERROR FRAME", msgCounter);
            } else {
                unsigned j;

                if(curId == id) {
                    seen = true;
                    debug_printf("(%u) id:%ld dlc:%u data: ", msgCounter, id, dlc);
                    if (dlc > 8) {
                        dlc = 8;
                    }
                    for (j = 0; j < dlc; j++) {
                        debug_printf("%2.2x ", msg[j]);
                    }
                }

            }
            debug_printf(" flags:0x%x time:%lu\n", flag, time);
        } else {
            if (errno == 0) {
                check("\ncanReadWait", stat);
            } else {
                perror("\ncanReadWait error");
            }
        }

    } while ((stat == canOK) & seen!=true);

ErrorExit:

    stat = canBusOff(hnd);
    check("canBusOff", stat);
    usleep(50 * 1000); // Sleep just to get the last notification.
    stat = canClose(hnd);
    check("canClose", stat);
    stat = canUnloadLibrary();
    check("canUnloadLibrary", stat);

    return 0;
}
