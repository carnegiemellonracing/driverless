#Brakes
BRAKES_CHANNEL_1 = 3
BRAKES_CHANNEL_2 = 4
BRAKES_STOP_VAL = 0 #?
MAX_BRAKES = 2**16-1

BRAKES_STATUS_TOPIC = "brakes_status"
BRAKES_TOPIC = "brakes"

EN_PIN = 'CAN1_DOUT'  #PIN 33
ENB_PIN = 'CAN0_DOUT' #pin 31

DELTA_THRESH = 0
MAX_EXTEND = .7

ADC_BUFFER = 200
MAX_ADC_VAL = 16000
MIN_ADC_VAL = 550 

#FSM
FSM_SENSOR1_CHANNEL = 0
FSM_SENSOR2_CHANNEL = 7
FSM_START_VAL = 0 
FSM_STOP_VAL = 0 
FSM_MAX_VAL = 20000 #range: 15-20k; 10% duty cycle → around 7500
FSM_MIN_VAL = 0

FSM_TOPIC = "throttle"

#STEER
STEER_CHANNEL = 7

STEER_TOPIC = "steer"

#1500-8000 -> 360 deg

#increments by 1000 currenty

#6000 theoretical full left
#2000 theoretical full right (mount slop)
STEER_MIN_VAL = 1638 
STEER_MAX_VAL = 8192
STEER_MIN_ANGLE = -10
STEER_MAX_ANGLE = 10
STEER_MID_VAL = (STEER_MAX_VAL + STEER_MIN_VAL) // 2

def angle_to_PWM(angle):
    return -(STEER_MAX_VAL - STEER_MIN_VAL) / (STEER_MAX_ANGLE - STEER_MIN_ANGLE) * angle + STEER_MID_VAL
