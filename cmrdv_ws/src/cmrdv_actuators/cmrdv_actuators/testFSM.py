from adafruit_servokit import ServoKit
from adafruit_pca9685 import PCA9685
from board import SCL, SDA
import busio


class DriverBoard:
  """
  Contains all connected actuators and communicates with the arduino
  Paramters
  ---------
  NONE
  Attributes
  ----------
  driverboard : 
      connected driverboard
  kit : servoKit
      servo kit
  """

  __instance = None
  @staticmethod
  def getInstance():
    """Static Access Method"""
    if DriverBoard.__instance == None:
      DriverBoard()
      return DriverBoard.__instance

  def __init__(self):
    """Virtual private constructor"""
    if DriverBoard.__instance != None:
      raise Exception("DriverBoard is a Singleton class")
    else:
      DriverBoard.__instance = self
      #self.arduino = serial.Serial('/dev/ttyACMO', baudrate=115200, timeout=1)
      self.i2c_bus = busio.I2C(SCL, SDA)
      self.pca = PCA9685(self.i2c_bus)
      self.pca.frequency = 1526 #applies to ALL channels

      #self.kit = ServoKit(channels=16, i2c=self.i2c_bus)


  def getServo(self, channel, continuous=False):
    """
    gets a servo from the driverboard at the specified channel
    
    Parameters
    ---------
      channel : int
        channel of desired servo
      continuous : bool
        true if the desired servo is a continuous servo
    
    Returns
    ---------
      a servo object
    """
    # if continuous: return self.kit.continuous_servo[channel]

    # return  self.kit.servo[channel]

    return None

  def getPWMChannel(self, channel):
    """
    gets a PWMchannel from the 
    
    Parameters
    ---------
      channel : int
        channel
    
    Returns
    ---------
      a pwmchannel object
    """

    return  self.pca.channels[channel]

  def read(self):
    """
    Reads and returns a line of data from the DriverBoard
    """

    try:
      data = self.arduino.readline()
      print(data)

    except:
      self.arduino.close()

    #TODO: clean data we get from arduino
    return data

  def kill(self):
    self.pca.deinit()


class FSMSensorController:
  """
  Controls two FSM sensors.
  """

  def __init__(self):

    self.FSM_SENSOR1_CHANNEL = 0 #TEMP
    self.FSM_SENSOR2_CHANNEL = 1 #TEMP

    self.START_VAL = 0 #TODO: Determine corret value
    self.STOP_VAL = 0 #TODO: Determine corret value
    self.MAX_VAL = 2**16 - 1 #TODO: Determine corret value
    self.MIN_VAL = 0 #TODO: Determine corret value

    self.driverBoard = DriverBoard.getInstance()
    #self.fsmSensor1 = self.driverBoard.getServo(self.FSM_SENSOR1_CHANNEL, True)
    #self.fsmSensor2 = self.driverBoard.getServo(self.FSM_SENSOR2_CHANNEL, True)

    self.fsmSensor1 = self.driverBoard.getPWMChannel(self.FSM_SENSOR1_CHANNEL)
    self.fsmSensor2 = self.driverBoard.getPWMChannel(self.FSM_SENSOR2_CHANNEL)

    #start pwm cycle
    self.updateDutyCycle(self.START_VAL)
    
  
  def updateDutyCycle(self, val : int):

      

    if val > self.MAX_VAL:
      print("Value was too high. Setting it to Max value")
      val = self.MAX_VAL

    if val < self.MIN_VAL:
      print("value was too low. Setting it to Min value")
      val = self.MIN_VAL
    
    try:
      print("setting sensor1 to: ", val)
      self.fsmSensor1.duty_cycle = val
      #TODO: account for voltage offset
      #self.fsmSensor1.throttle = val

      offset = 0
      print("setting sensor2 to: ", val)
      self.fsmSensor2.duty_cycle = val
      #self.fsmSensor2.throttle = val

      print("Offset was: ", offset)


    except Exception as e:
      print(e)
      print('Exception caught')
      self.terminateFSM()



  def terminateFSM(self):

    print("Terminating FSM")

    self.fsmSensor1.duty_cycle = self.STOP_VAL
    self.fsmSensor2.duty_cycle = self.STOP_VAL

    #self.fsmSensor1.throttle = self.STOP_VAL
    #self.fsmSensor2.throttle = self.STOP_VAL
    
    self.driverBoard.kill()
    print("FSM Stopped")


    




def main(args=None):

    controller = FSMSensorController()

    print("starting PWM Loop")

    while True:
      val = input("Input duty cycle. exit to terminate: ")

      if (val == 'exit'):
          controller.terminateFSM()
          print("stopping PWM Loop")
          
          break

      else:
        controller.updateDutyCycle(int(val))


    
    print("PWM Loop terminated")


if __name__ == "__main__":
  main()






    



