from adafruit_pca9685 import PCA9685
from board import SCL_1, SDA_1
import busio

import Jetson.GPIO as GPIO
import adafruit_ads1x15.ads1015 as ADS
from adafruit_ads1x15.analog_in import AnalogIn


class DriverBoard:
  """
  FSM driverboard 
  Paramters
  ---------
  NONE
  Attributes
  ----------
  driverboard : 
      connected driverboard
  """

  __instance = None
  @staticmethod
  def get_instance():
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
      i2c_bus = busio.I2C(SCL_1, SDA_1)

      self.pca = PCA9685(i2c_bus, address=0x41)
      self.pca_steer = PCA9685(i2c_bus)#, address=0x40)

      self.pca.frequency = 1526 #applies to ALL channels
      self.pca_steer.frequency = 50 #applies to ALL channels

      #self.kit = ServoKit(channels=16, i2c=self.i2c_bus)
      
      #GPIO.setmode(GPIO.BOARD)

      # Create the ADC object using the I2C bus
      #ads = ADS.ADS1015(i2c_bus, address=0x48)
      # Create single-ended input on channel 0
      #self.adc_chan = AnalogIn(ads, ADS.P0)


  def get_servo(self, channel, continuous=False):
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
    
  def get_adc_val(self):

    return self.adc_chan.value

  def jetson_setup_pin(self, pin, HIGH):

    if HIGH:
      GPIO.setup(pin, GPIO.OUT, initial=GPIO.HIGH)

    else:
      GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)


  def get_pwm_channel(self, channel):
    """
    gets a PWMchannel from the driverboard instance
    
    Parameters
    ---------
      channel : int
        channel
    
    Returns
    ---------
      a pwmchannel object
    """

    return  self.pca.channels[channel]

  def steering_get_pwm_channel(self, channel):
    """
    gets a PWMchannel from the driverboard instance
    
    Parameters
    ---------
      channel : int
        channel
    
    Returns
    ---------
      a pwmchannel object
    """

    return  self.pca_steering.channels[channel]
  
  def set_pwm_channel(self, channel, val):
    """
    sets a given PWMchannel to a given duty_cycle
    
    Parameters
    ---------
      channel : int
        channel

      val : int
        value 
    
    Returns
    ---------
      a pwmchannel object
    """
    try:
        self.pca.channels[channel].duty_cycle = val
    except BaseException as e:
        print(e)
        print("aborting")
        self.pca.channels[channel].duty_cycle = 0
    	
    	
  
  def steering_set_pwm_channel(self, channel, val):
    """
    sets a given PWMchannel to a given duty_cycle
    
    Parameters
    ---------
      channel : int
        channel

      val : int
        value 
    
    Returns
    ---------
      a pwmchannel object
    """


    try:
        self.pca_steer.channels[channel].duty_cycle = val
    except BaseException as e:
        print(e)
        print("aborting")
        self.pca_steer.channels[channel].duty_cycle = 0


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
    #self.pca.deinit()
    pass
