from cmrdv_ws.src.cmrdv_planning.planning_codebase.midpoint.generator import MidpointGenerator
import numpy as np
from cmrdv_ws.src.cmrdv_planning.planning_codebase.raceline.raceline import reverse_transformation
from enum import Enum

Frame = Enum('Frame', ['car', 'world'])

# TODO: write function and use it whenever needed
## May be unnecessary
def switch_frame(points, input_frame: Frame, output_frame: Frame, reference = None):
    if input_frame == output_frame:
        return points
    
    if input_frame == Frame.car and output_frame == Frame.world:
        pass        
    elif input_frame == Frame.world and output_frame == Frame.car:
        pass

    raise NotImplementedError("Frame {input_frame} not implemented")

# OLD FUNCTION TO CHANGE FROM CAR FRAME TO WORLD FRAME
def change_frame(groups):
    # IDEA: get the cones in the frame of the car
    # Ideally orientation should be taken into account

    new_groups = 0 * groups

    for i in range(len(groups)):
        car = np.array([0.0, 0.0, 0.0]) # position of car + color
        if i > 0:
            last_group = groups[i-1]
            last_cones = last_group[-1:-3:-1]
            c1 = last_cones[0]
            c2 = last_cones[1]
            car = np.array([
                (c1[0] + c2[0])/2, # x average
                (c1[1] + c2[1])/2, # y average
                0.0                # neutral color
            ])

        new_groups[i] = groups[i] - car

    return new_groups

# OLD FUNCTION TO CHANGE FROM WORLD FRAME TO CAR FRAME
def change_frame_inverse(interpolation, index, cones):
    # IDEA: get the cones in the absolute frame
    # Ideally orientation should be taken into account

    car = np.array([0, 0]) # coordinates of the car

    if index > 0:
        last_group = cones[index-1]
        last_cones = last_group[-1:-3:-1]
        c1 = last_cones[0]
        c2 = last_cones[1]
        car = np.array([
            (c1[0] + c2[0])/2, # x average
            (c1[1] + c2[1])/2 # y average
        ])
        #copied from different function: generator and i both unreferenced
        #generator.cumulated_splines[i].update(translation=lambda t: t + car)# update translation vector

    return interpolation + car


def generate_centerline_from_cones(cones, interpolation_number=10):
    generator = MidpointGenerator(interpolation_number=interpolation_number)

    # def group_cones(lcones, rcones):
    #     per_group = 3

    #     n = len(lcones)
    #     n = n - n%per_group # we have groups of per_group (we don't deal with the case where we don't have enough cones)

    #     groups = []

    #     for i in range(n//per_group):
    #         indices = range((i*per_group), (i+1)*per_group)
            
    #         group_left_cones = lcones[indices]
    #         group_right_cones = rcones[indices]

    #         group = []
    #         for i in range(per_group):
    #             group.append(group_left_cones[i])
    #             group.append(group_right_cones[i])

    #         groups.append(np.array(group))

    #     groups = np.array(groups)

    #     return groups

    # cones = group_cones(lcones, rcones)

    # If force change of frame
    # Callee should use switch_frame instead
    # cones = change_frame(cones) if frame==Frame.car else cones

    interpolated_points = np.array([])

    for i in range(len(cones)):
        group = cones[i]
        interpolation = generator.interpolate_cones(group)

        # If force change of frame
        # Callee should use switch_frame instead
        # interpolation = change_frame_inverse(interpolation, i, cones) if frame==Frame.car else interpolation

        if len(interpolated_points) == 0:
            interpolated_points = interpolation
        else:
            interpolated_points = np.vstack((interpolated_points, interpolation))

    return interpolated_points, generator

def interpolate_raceline(path, cumulative_lengths, progress, previous_index=None, precision=20):
    '''
    Find a point on the raceline approximately corresponding to a given progress

    Inputs:
    -------------
    path: list[Spline]
        The path to work with

    cumulative_lengths: TODO

    progress: float
        Targeted progress on the raceline

    previous_index: int
        Previous result for the index, should be equal or lower than the next index

    precision: int
        Number of points used to get approximation for a specific spline

    Output:
    -------------
    TODO
    '''

    #if progress > cumulative_lengths[-1]:   
        #exit("Unreachable progress: the progress wanted is greater than the total length of the reference path")
    
    if progress < 0:
        exit("Unreachable progress: negative progress encountered during interpolation")

    index = previous_index
    if index is None:
        index = np.searchsorted(cumulative_lengths, progress)
    else:
        n = len(cumulative_lengths)

        while (index+1) < n and cumulative_lengths[index] < progress:
            index += 1

    spline = path[index]

    delta = progress if index == 0 else progress - cumulative_lengths[index-1]

    # local point is the point represented in the spline frame
    point, length, local_point, x = spline.along(delta, precision=precision)

    return (
        point,
        spline,
        local_point,
        progress - delta + length,
        index,
        x
    )

def state_to_point(progress, normal, reference_path, cumulative_lengths, last_index=None):
    no_deviation, spline, local_point, exact, index, x = interpolate_raceline(reference_path, cumulative_lengths, progress, last_index)

    deriv = spline.first_der(x) # gradient at the point
    tangent = [1, deriv]
    normal_direction = np.array([-tangent[1], tangent[0]])
    normal_direction /= np.linalg.norm(normal_direction)

    shifted = local_point + normal * normal_direction

    point = reverse_transformation(np.array([shifted]), spline.Q, spline.translation_vector)

    return point, no_deviation, exact, index

def states_to_points(states, reference_path, cumulative_lengths):
    points = []
    last_index = None
    progress_list = np.clip(states[:, 0], 0, None) # Make sure progress is non-negative
    n_list = states[:, 1] # Get normal deviation from reference path

    n = len(progress_list)

    for i in range(n): # for each point
        progress = progress_list[i]
        normal = n_list[i]

        # Convert point to (x, y) coordinates
        point, _, _, last_index = state_to_point(progress, normal, reference_path, cumulative_lengths, last_index)

        points.append(point)

    points = np.array(points)
