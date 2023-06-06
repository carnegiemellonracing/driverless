import numpy as np
import numpy.typing as npt
import cmrdv_ws.src.cmrdv_planning.planning_codebase.raceline.raceline as raceline
from typing import List

# bounding boxes and z axes are not needed since they are constant
# if depth is given later, that will be used
# one problem that i'm worried about is if the cone count doesn't match up in pairs
class MidpointGenerator:
    def __init__(self, interpolation_number=30):
        """
        Parameters
        ----------
        interpolation_number : int
            TODO
        """

        self.PERCEP_COLOR = 2 # TODO don't hardcode PERCEP_COLOR

        self.BLUE = 1 # TODO don't hardcode constants from perceptions
        self.YELLOW = 2 # TODO don't hardcode constants from perceptions
        self.ORANGE = 3

        self.interpolation_number = interpolation_number

        self.cumulated_splines: List[raceline.Spline] = []
        self.cumulated_lengths = []

    def sorted_by_norm(self, list):
        """
        Return a copy of a list sorted by the L2 norms of its vectors in ascending order

        Parameters
        ----------
        list : npt.NDArray[npt.NDArray[np.float64]]
            list of vectors


        Returns
        -------
        npt.NDArray[npt.NDArray[np.float64]]
            a copy of the input list, sorted by norms of vectors (ascending order)
        """
        norms = np.linalg.norm(list, axis=1)
        return list[np.argsort(norms)]

    def midpoint(self, inner, outer):
        '''
        Input 2 np arrays, output their midpoints

        TODO Complete documentation
        '''

        M = 1/2 * (inner + outer)
        return M

    def generate_splines(self, midpoints):
        """
        Generate the splines fitting the given points

        Parameters
        ----------
        midpoints : npt.NDArray[npt.NDArray[np.float64]]
            list of points


        Returns
        -------
        list[raceline.Spline]
            a list of splines represented as Spline objects
        """
        # Compute splines
        [splines, cumulative_lengths] = raceline.raceline_gen(midpoints, points_per_spline=len(midpoints), loop=False) # there is no loop to do as we don't generate an entire path

        self.cumulated_splines = [*self.cumulated_splines, *splines]

        if len(self.cumulated_lengths) == 0:
            self.cumulated_lengths = cumulative_lengths
        else:
            self.cumulated_lengths = np.hstack((self.cumulated_lengths, cumulative_lengths + self.cumulated_lengths[-1]))

        return splines

    def generate_points(self, perception_data):
        """
        Generate points used for predicting a good trajectory.
        Based on the quality of perception data, some virtual midpoints might be generated.

        Parameters
        ----------
        perception_data : npt.NDArray[npt.NDArray[np.float64]]
            perception_data including cone positions and colors in the frame of the car

        Returns
        -------
        npt.NDArray[npt.NDArray[np.float64]]
            a list of points to build a trajectory from
        """
        # check that perception_data is already numpy array
        # otherwise: perception_data = np.array(perception_data)

        perception_data = np.array(perception_data)
        print(perception_data)
        # separate data into inner and outer cones
        left_indices = perception_data[:, self.PERCEP_COLOR] == self.BLUE
        right_indices = perception_data[:, self.PERCEP_COLOR] == self.YELLOW

        left_points = perception_data[left_indices, :][:, :2]
        right_points = perception_data[right_indices, :][:, :2]

        # Sort points by distance to origin
        left_points = self.sorted_by_norm(left_points)
        right_points = self.sorted_by_norm(right_points)

        #interpolate points if we only see one of a single color
        if (len(left_points) == 0 and len(right_points) > 0):
            #interpolate the other side
            for i in range(len(right_points)):
                if(i == 0):
                    #just take first cone and flip x axis
                    temp = np.copy(right_points[0])
                    temp[0] = -1*temp[0] #flip axis
                    left_points = np.append(left_points, temp)
                    left_points = np.reshape(left_points, (1, 2))
                    # print("after first",left_points)
                else: #take linear transform
                    difference = [right_points[i][0]-right_points[i-1][0], right_points[i][1]-right_points[i-1][1]]
                    difference = np.array(difference, dtype=float)
                    # print("difference",difference)
                    # print("left_points[i-1]",left_points[i-1])
                    newCone = left_points[i-1]+difference
                    # print("newCone",newCone)
                    left_points = np.vstack((left_points, newCone))
            midpoints = self.midpoint(left_points, right_points)
        elif (len(left_points) > 0 and len(right_points) == 0):
            for i in range(len(left_points)):
                if(i == 0):
                    #just take first cone and flip x axis
                    temp = np.copy(left_points[0])
                    temp[0] = -1*temp[0] #flip axis
                    right_points = np.append(right_points, temp)
                    right_points = np.reshape(right_points, (1, 2))
                else: #take linear transform
                    difference = [left_points[i][0]-left_points[i-1][0], left_points[i][1]-left_points[i-1][1]]
                    difference = np.array(difference, dtype=float)
                    newCone = right_points[i-1]+difference
                    right_points = np.vstack((right_points, newCone))
            midpoints = self.midpoint(left_points, right_points)
        else:
            # Add origin to draw path from current position
            min_len = min(np.shape(left_points)[0], np.shape(right_points)[0])
            left_points = left_points[0:min_len]
            right_points = right_points[0:min_len]
            midpoints = self.midpoint(left_points, right_points)
            midpoints = np.vstack(([0, 0], midpoints))

        return midpoints

    def interpolate_cones(self, perception_data, interpolation_number=None):
        """
        TODO Complete documentation
        """
        if interpolation_number == None:
            interpolation_number = self.interpolation_number

        spline_next = self.spline_from_cones(perception_data)

        spline: raceline.Spline = spline_next

        return spline.interpolate(interpolation_number)

    def spline_from_cones(self, perception_data):
        """
        TODO Complete documentation
        """
        print('Executing PathPlanning Midpoint Spline...')
        # define what needs to be executed at every iteration of MCL

        # Determine (virtual) midpoints
        midpoints = self.generate_points(perception_data)
        
        assert(len(midpoints) > 1)

        # We only generate one spline
        splines = self.generate_splines(midpoints)

        #assert(len(splines) == 1)
        spline = splines[0]

        return spline
