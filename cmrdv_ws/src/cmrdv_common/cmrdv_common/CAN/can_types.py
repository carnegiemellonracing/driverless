"""
from carnegiemellonracing/stm32f413-drivers:
/** @brief Standard CAN heartbeat. */
typedef struct {
    uint8_t state;          /**< @brief Board state. */
    uint8_t error[2];       /**< @brief Error matrix. */
    uint8_t warning[2];     /**< @brief Warning matrix. */
} cmr_canHeartbeat_t;
""" 

ALIVE_STATE = 2
WARNING_STATE = 3
ERROR_STATE = 4

""" Heartbeat timeout thresholds """
TIMEOUT_ERROR_S = 50/1000
TIMEOUT_WARN_S = 25/1000

""" CAN 10Hz Tick types """
TX10HZ_PRIORITY = 3                                     # CAN 10 Hz TX priority.
TX10HZ_PERIOD_MS = 100                                  # CAN 10 Hz TX period (milliseconds).
TX10HZ_PERIOD_S = TX10HZ_PERIOD_MS/1000                 # CAN 10 Hz TX period (seconds).

""" CAN 100Hz Tick types """
TX100HZ_PRIORITY = 5                                    # CAN 100 Hz TX priority.
TX100HZ_PERIOD_MS = 10                                  # CAN 100 Hz TX period (milliseconds).
TX100HZ_PERIOD_S = TX100HZ_PERIOD_MS/1000               # CAN 100 Hz TX period (seconds).

# """ CAN 100Hz Tick types """
# TX100HZ_PRIORITY = 5                                    # CAN 10 Hz TX priority.
# TX100HZ_PERIOD_MS = 10                                  # CAN 10 Hz TX period (milliseconds).
# TX100HZ_PERIOD_S = TX100HZ_PERIOD_MS/1000               # CAN 10 Hz TX period (seconds).

""" CAN IDs """
CMR_CANID_HEARTBEAT_AUTONOMOUS = 0x10B                  # Autonomous heartbeat.
CMR_CANID_HEARTBEAT_VSM = 0x100                         # VSM heartbeat. 

""" Node States """
class STATE:                                            # cmr_canState_t
    CMR_CAN_UNKNOWN = 0                                 # Current state unknown.
    CMR_CAN_GLV_ON = 1                                  # Grounded low voltage on.
    CMR_CAN_HV_EN = 2                                   # High voltage enabled.
    CMR_CAN_RTD = 3                                     # Ready to Drive.
    CMR_CAN_ERROR = 4                                   # Error has occurred.
    CMR_CAN_CLEAR_ERROR = 5                             # Request to clear error.\

""" Represents the car's current driving mode (gear). """
class GEAR: 
    CMR_CAN_GEAR_UNKNOWN = 0                            # Unknown Gear State
    CMR_CAN_GEAR_REVERSE = 1                            # Reverse mode
    CMR_CAN_GEAR_SLOW = 2                               # Slow mode
    CMR_CAN_GEAR_FAST = 3                               # Fast simple mode
    CMR_CAN_GEAR_ENDURANCE = 4                          # Endurance-event mode 
    CMR_CAN_GEAR_AUTOX = 5                              # Autocross-event mode
    CMR_CAN_GEAR_SKIDPAD = 6                            # Skidpad-event mode
    CMR_CAN_GEAR_ACCEL = 7                              # Acceleration-event mode
    CMR_CAN_GEAR_TEST = 8                               # Test mode (for experimentation)
    CMR_CAN_GEAR_AUTONOMOUS = 9                         # Autonomous mode (NEW)
    CMR_CAN_AUTO_INSP = 10,                             # Autonomous Inspection Mode (NEW)
    CMR_CAN_AUTO_BT = 11                                # Autonomous Brake Test Mode (NEW)
    CMR_CAN_AUTO_TRACKDRIVE = 12                        # Autonomous Trackdrive Mode (NEW)
    CMR_CAN_GEAR_LEN = 13

""" Heartbeat error matrix bit fields. """
class ERROR:
    CMR_CAN_ERROR_NONE = 0                              # No errors.
    CMR_CAN_ERROR_VSM_TIMEOUT = (1 << 0)                # No VSM heartbeat received for 50 ms.

    CMR_CAN_ERROR_VSM_MODULE_TIMEOUT = (1 << 15)        # Reception period surpassed error threshold.

    CMR_CAN_ERROR_VSM_MODULE_STATE = (1 << 14)          # At least one module is in the wrong state.
    CMR_CAN_ERROR_VSM_LATCHED_ERROR = (1 << 13)         # At least one of the error latches is active.
    CMR_CAN_ERROR_VSM_DCDC_FAULT = (1 << 12)            # VSM DCDC fault signal.
    CMR_CAN_ERROR_VSM_HALL_EFFECT = (1 << 11)           # VSM hall effect sensor out-of-range.
    CMR_CAN_ERROR_VSM_BPRES = (1 << 10)                 # @brief VSM brake pressure sensor out-of-range.

    CMR_CAN_ERROR_AFC_FANS_CURRENT = (1 << 15)          # AFC fan current out-of-range.
    CMR_CAN_ERROR_AFC_DRIVER1_TEMP = (1 << 14)          # AFC driver IC #1 temperature out-of-range.
    CMR_CAN_ERROR_AFC_DRIVER2_TEMP = (1 << 13)          # AFC driver IC #2 temperature out-of-range.
    CMR_CAN_ERROR_AFC_DRIVER3_TEMP = (1 << 12)          # AFC driver IC #3 temperature out-of-range.
    CMR_CAN_ERROR_AFC_DCDC1_TEMP = (1 << 11)            # AFC DCDC #1 temperature out-of-range.
    CMR_CAN_ERROR_AFC_DCDC2_TEMP = (1 << 10)            # AFC DCDC #2 temperature out-of-range.

    CMR_CAN_ERROR_PTC_FAN_CURRENT = (1 << 15)           # PTC fan current out-of-range.
    CMR_CAN_ERROR_PTC_DRIVERS_TEMP = (1 << 14)          # PTC fan/pump driver IC temperature out-of-range.  
    CMR_CAN_ERROR_PTC_WATER_TEMP = (1 << 13)            # PTC water temperature out-of-range.

""" Heartbeat warning matrix bit fields. """
class WARN:
    CMR_CAN_WARN_NONE = 0                               # No warnings.

    CMR_CAN_WARN_VSM_TIMEOUT = (1 << 0)                 # No VSM heartbeat received for 25 ms.
    CMR_CAN_WARN_BUS_VOLTAGE = (1 << 1)                 # Low-voltage bus voltage out-of-range.
    CMR_CAN_WARN_BUS_CURRENT = (1 << 2)                 # Low-voltage bus current out-of-range.

    CMR_CAN_WARN_VSM_HVC_TIMEOUT = (1 << 15)            # VSM hasn't received HVC heartbeat for 25 ms.
    CMR_CAN_WARN_VSM_CDC_TIMEOUT = (1 << 14)            # VSM hasn't received CDC heartbeat for 25 ms.
    CMR_CAN_WARN_VSM_FSM_TIMEOUT = (1 << 13)            # VSM hasn't received FSM heartbeat for 25 ms.
    CMR_CAN_WARN_VSM_PTC_TIMEOUT = (1 << 12)            # VSM hasn't received PTC heartbeat for 25 ms.
    CMR_CAN_WARN_VSM_DIM_TIMEOUT = (1 << 11)            # VSM hasn't received DIM heartbeat for 25 ms.
    CMR_CAN_WARN_VSM_AFC0_TIMEOUT = (1 << 10)           # VSM hasn't received AFC 0 heartbeat for 25 ms.
    CMR_CAN_WARN_VSM_AFC1_TIMEOUT = (1 << 9)            # VSM hasn't received AFC 1 heartbeat for 25 ms.
    CMR_CAN_WARN_VSM_APC_TIMEOUT = (1 << 8)             # VSM hasn't received APC heartbeat for 25 ms.
    CMR_CAN_WARN_VSM_DIM_REQ_NAK = (1 << 7)             # VSM is rejecting DIM state request.
    CMR_CAN_WARN_VSM_AUTONOMOUS_TIMEOUT = (1 << 6)      # VSM hasn't received Autonomous heartbeat for 25 ms.
                                                        # TODO: change to correct number of ms, depending on 

    CMR_CAN_WARN_FSM_TPOS_IMPLAUSIBLE = (1 << 15)       # FSM throttle position implausibility (L/R difference > 10%).
    CMR_CAN_WARN_FSM_BPP = (1 << 14)                    # FSM brake pedal plausibility fault.
    CMR_CAN_WARN_FSM_TPOS_R = (1 << 13)                 # FSM right throttle position out-of-range.
    CMR_CAN_WARN_FSM_TPOS_L = (1 << 12)                 # FSM left throttle position out-of-range.
    CMR_CAN_WARN_FSM_BPOS = (1 << 11)                   # FSM brake pedal position out-of-range.
    CMR_CAN_WARN_FSM_BPRES = (1 << 10)                  # FSM brake pressure sensor out-of-range.
    CMR_CAN_WARN_FSM_SWANGLE = (1 << 9)                 # FSM steering wheel angle out-of-range.
    