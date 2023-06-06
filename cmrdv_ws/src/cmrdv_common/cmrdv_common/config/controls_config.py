from numpy import array, ndarray

# Topics
CONTROL_ACTION_TOPIC = '/control_action'
CARROT_TOPIC = '/carrot'

# Constants
LOOKAHEAD = 1  # m
GAIN_MATRIX: ndarray = array([
    [0, 0, 0, 0, 0, 0],
    [7.5, 0, -0.1, 0, 0, 0]
])
GOAL_SPEED = 1
CARROT_AVG_HISTORY = 4
CONTROLLER_DEADZONE = 2