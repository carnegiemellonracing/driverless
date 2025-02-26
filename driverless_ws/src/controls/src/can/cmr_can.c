/**
 * @file cmr_tx.c
 * @brief cmr_can_wrapper
 * 
 * @author Ayush Garg and Vikram Mani
 */



#include <canlib.h>
#include <canstat.h>
#include <stdio.h>
#include <signal.h>
#include <errno.h>
#include <unistd.h>
#include <stdbool.h>
#include "cmr_can.h"

#define READ_WAIT_INFINITE (unsigned long)(-1)
static unsigned int msgCounter = 0;



static void check(char *id, canStatus stat)
{
    if (stat != canOK) {
        char buf[50];
        buf[0] = '\0';
        canGetErrorText(stat, buf, sizeof(buf));
        printf("%s: failed, stat=%d (%s)\n", id, (int)stat, buf);
    }
}

static void printUsageAndExit(char *prgName)
{
    printf("Usage: '%s <channel>'\n", prgName);
    exit(1);
}

static void sighand(int sig, siginfo_t *info, void *ucontext)
{
    (void)sig;
    (void)info;
    (void)ucontext;
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
        printf("CAN Status Event: %s\n", busStatToStr(data->info.status.busStatus));
        break;

    case canEVENT_ERROR:
        printf("CAN Error Event\n");
        break;

    case canEVENT_TX:
        printf("CAN Tx Event\n");
        break;

    case canEVENT_RX:
        printf("CAN Rx Event\n");
        break;
    }
    return;
}

//CMR TX
//Channel should default to 0
static int cmr_can_tx(int channel, long id, void* msg, unsigned int msgLen, bool verbose){
    canHandle hnd;
    canStatus stat;
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
        printf("Sending a CAN message on channel %d\n", channel);
    }

    canInitializeLibrary();

    /* Open channel, set parameters and go on bus */
    hnd = canOpenChannel(channel,
                         canOPEN_EXCLUSIVE | canOPEN_REQUIRE_EXTENDED | canOPEN_ACCEPT_VIRTUAL);
    if (hnd < 0) {
        // printf("canOpenChannel %d", channel);
        check("", hnd);
        return -1;
    }

    stat = canSetBusParams(hnd, canBITRATE_500K, 0, 0, 0, 0, 0);
    check("canSetBusParams", stat);
    if (stat != canOK) {
        goto ErrorExit;
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
        goto ErrorExit;
    }

    stat = canWrite(hnd, id, msg, msgLen, 0);
    check("canWrite", stat);
    if (stat != canOK) {
        goto ErrorExit;
    }
    stat = canWriteSync(hnd, 1000);
    check("canWriteSync", stat);
    if (stat != canOK) {
        goto ErrorExit;
    }

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



/*
@breif sends control action message

@param frontTorque_mNm
@param rearTorque_mNm
@param rackDisplacement_mm

@returns 0 on success else error occured
*/
int sendControlAction(int16_t frontTorque_mNm, int16_t rearTorque_mNm, uint8_t rackDisplacement_mm){
    long controlActionID = 0x190;

    //defines our message
    unsigned char msg[5];
    msg[0] = (unsigned char) frontTorque_mNm >> 8; //8 MSB of frontTorque
    msg[1] = (unsigned char) frontTorque_mNm; // 8 LSB of frontTorque
    msg[2] = (unsigned char) rearTorque_mNm >> 8; // 8 MSB of rearTorque
    msg[3] = (unsigned char) rearTorque_mNm; // 8 LSB of rearTorque
    msg[4] = (unsigned char) rackDisplacement_mm;

    return cmr_can_tx(0, controlActionID, &msg, 5, false);
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
        printf("Reading CAN messages on channel %d\n", channel);
    }

    canInitializeLibrary();

    /* Open channel, set parameters and go on bus */
    hnd = canOpenChannel(channel,
                         canOPEN_EXCLUSIVE | canOPEN_REQUIRE_EXTENDED | canOPEN_ACCEPT_VIRTUAL);
   
    if(verbose) {
        if (hnd < 0) {
        // printf("canOpenChannel %d", channel);
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
                printf("(%u) ERROR FRAME", msgCounter);
            } else {
                unsigned j;

                if(curId == id) {
                    seen = true;
                    printf("(%u) id:%ld dlc:%u data: ", msgCounter, id, dlc);
                    if (dlc > 8) {
                        dlc = 8;
                    }
                    for (j = 0; j < dlc; j++) {
                        printf("%2.2x ", msg[j]);
                    }
                }

            }
            printf(" flags:0x%x time:%lu\n", flag, time);
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
