import my_optimizer
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn import svm
import time

class TestSuite:
    def __init__(self):
        self.s_curve_horizontal = []
        self.s_curve_vertical = []
        self.straight_horizontal = []
        self.straight_vertical = []
        self.hairpin_horizontal = []
        self.hairpin_vertical = []

        self.track_names = [
            "s_curve_horizontal",
            "s_curve_vertical",
            "straight_horizontal",
            "straight_vertical",
            "hairpin_horizontal",
            "hairpin_vertical"
        ]

        self.tracks = {
                0 : self.s_curve_horizontal,
                1 : self.s_curve_vertical,
                2 : self.straight_horizontal,
                3 : self.straight_vertical,
                4 : self.hairpin_horizontal,
                5 : self.hairpin_vertical,
                }


    def generate_tracks(self):
        """generate_tracks Generates the lists of tuples representing a track
                           track segment. Each tuple is in the form (x, y)
                           where x and y are both integers, representing the
                           coordinates of a cone on the track"""

        # Generating the S curve using a sin function
        blue_cones_x   = np.array([])
        blue_cones_y   = np.array([])
        yellow_cones_x = np.array([])
        yellow_cones_y = np.array([])

        for i in range(50):
            v = math.sin(i / 10)
            blue_cones_x = np.append(blue_cones_x, [i / 10])
            blue_cones_y = np.append(blue_cones_y, [v + 2])
            yellow_cones_x = np.append(yellow_cones_x, [i / 10])
            yellow_cones_y = np.append(yellow_cones_y, [v - 2])

        self.tracks[0] = [blue_cones_x, blue_cones_y, yellow_cones_x, yellow_cones_y]
        self.tracks[1] = [blue_cones_y, blue_cones_x, yellow_cones_y, yellow_cones_x]
        # Generating the straights
        blue_cones_y = np.array([])
        yellow_cones_y = np.array([])
        for i in range(50):

            blue_cones_y = np.append(blue_cones_y, [2])
            yellow_cones_y = np.append(yellow_cones_y, [-2])

        self.tracks[2] = [blue_cones_x, blue_cones_y, yellow_cones_x, yellow_cones_y]
        self.tracks[3] = [blue_cones_y, blue_cones_x, yellow_cones_y, yellow_cones_x]

        # Generating the hairpins
        blue_cones_x = np.array([])
        blue_cones_y = np.array([])
        yellow_cones_x = np.array([])
        yellow_cones_y = np.array([])
        for i in range(1,100):
            x = (-3.99 + (i/10))
            v_bottom_blue = -1 * math.log2(x + 4) - 12
            v_top_blue = math.log2(x + 4) + 5
            v_bottom_yellow = -1 * math.log2(i/10) - 7
            v_top_yellow = math.log2(i/10)
            blue_cones_x = np.append(blue_cones_x, [x, x])
            blue_cones_y = np.append(blue_cones_y, [v_bottom_blue, v_top_blue])
            yellow_cones_x = np.append(yellow_cones_x, [i/10, i/10])
            yellow_cones_y = np.append(yellow_cones_y, [v_bottom_yellow, v_top_yellow])

        for i in range(4):
            x = -3.995 + i * 0.01
            v_bottom_blue = -1 * math.log2(x + 4) - 12
            v_top_blue = math.log2(x + 4) + 5

            blue_cones_x = np.append(blue_cones_x, [x, x])
            blue_cones_y = np.append(blue_cones_y, [v_bottom_blue, v_top_blue])


        self.tracks[4] = [blue_cones_y, blue_cones_x, yellow_cones_y, yellow_cones_x]
        self.tracks[5] = [blue_cones_x, blue_cones_y, yellow_cones_x, yellow_cones_y]


TS = TestSuite()
TS.generate_tracks()
prompt_string = """Welcome to the Raceline Optimizer test suite!
1.) Name your raceline optimizer file: my_optimizer.py
When defining your raceline optimizer in my_optimizer.py, please define the
following functions:
###############################################################################

run_optimizer This function takes in cones and runs the optimizer on cones

:param blue_cones_x: A numpy array of x coordinates, representing x-coordinate
of blue cones in the racetrack segment your optimizer will be tested on

:param blue_cones_y: A numpy array of y coordinates, representing y-coordinate
of blue cones in the racetrack segment your optimizer will be tested on

:param yellow_cones_x: A numpy array of x coordinates, representing x-coordinate
of yellow cones in the racetrack segment your optimizer will be tested on

:param yellow_cones_y: A numpy array of y coordinates, representing y-coordinate
of yellow cones in the racetrack segment your optimizer will be tested on

:return: A list containing 2 numpy arrays. The 1st numpy array contains the
x coordinate of the raceline. The 2nd numpy array contains the y coordinate
of the raceline.
###############################################################################
2.) Select a track that you would like to test on
Below are options for different tracks that you can test you're optimizer on:\n"""

print(prompt_string)


for i in range(len(TS.track_names)):
    print("\t",i, " : ", TS.track_names[i])

while True:
    tracktype = input(("\nTo test on a specific track, PLEASE TYPE THE NUMBER" +
                    " to the left of the track (ex. 1). \nTo quit, type q." +
                    "To view the tracks, type t: "))
    if tracktype == 'q':
        break
    elif tracktype == 't':
        print()
        for i in range(len(TS.track_names)):
            print("\t",i, " : ", TS.track_names[i])

    elif int(tracktype) in TS.tracks.keys():
        print("Press the x on the top right corner to close",
                "of the visualization window")
        cur_track = TS.tracks[int(tracktype)]
        start_time = time.time()
        raceline = my_optimizer.run_optimizer(cur_track[0], cur_track[1],
                                                cur_track[2], cur_track[3])
        end_time = time.time()
        calc_time = end_time - start_time
        # raceline = [np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 5, 6])]
        plt.style.use('dark_background')
        plt.scatter(cur_track[0], cur_track[1], color='blue')
        plt.scatter(cur_track[2], cur_track[3], color='yellow')
        plt.scatter(raceline[0], raceline[1])
        print("\nRaceline generation time: ", calc_time, "seconds")
        plt.show()

    else:
        print("Error: please type in a number corresponding to a track")








