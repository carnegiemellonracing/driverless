from dataclasses import dataclass
# from functools import cached_property ; only python >= 3.8
import numpy as np
import numpy.typing as npt
from random import randint
import scipy.integrate as integrate
import sys
from typing import Tuple, List


preferred_degree = 3 # degree of spline polynomials
overlap = 0


@dataclass(order=False)
class Spline:
    """Spline class for storing splines and related data"""

    polynomial: np.poly1d
    
    
    points: npt.NDArray[np.float64]  # For testing frenet
    rotated_points: npt.NDArray[np.float64]
    
    
    Q: npt.NDArray[np.float64]
    translation_vector: npt.NDArray[np.float64]
    
    
    first_der: np.poly1d
    second_der: np.poly1d
    
    path_id: int
    sort_index: int

    #@cached_property only python >= 3.8
    @property
    def length(self):
        return arclength(self.polynomial, self.rotated_points[0, 0], self.rotated_points[-1, 0])

    def interpolate(self, number, bounds=None):
        return interpolate(self, number, bounds)

    def __eq__(self, other):
        if other.__class__ is not self.__class__ or other.path_id != self.path_id:
            return NotImplemented
        return self.sort_index == other.sort_index

    def __lt__(self, other):
        if other.__class__ is not self.__class__ or other.path_id != self.path_id:
            return NotImplemented
        return self.sort_index < other.sort_index
    
    def update(self, translation=(lambda t: t), rotation=(lambda Q: Q)):
        self.translation_vector = translation(self.translation_vector)
        self.Q = rotation(self.Q)
        self.points = reverse_transformation(self.rotated_points, self.Q, self.translation_vector)

    def along(self, progress, point_index=0, precision=20):
        assert(point_index >= 0 and point_index < len(self.points))
        
        length = self.length

        # Estimate the range needed for a good approximation

        first_x = self.rotated_points[0][0]
        last_x = self.rotated_points[-1][0]

        delta = last_x - first_x

        boundaries = [first_x, last_x]
        ratio = progress // length + 1

        def length_to(x):
            return arclength(self.polynomial, first_x, x), x

        if ratio >= 2: # if not on spline, we need to extrapolate
            shoot, x = length_to(first_x + delta * ratio)
            lower_bound, upper_bound = first_x + delta * (ratio - 1), first_x + delta * ratio

            if shoot < progress:
                while shoot < progress:
                    lower_bound = x
                    # add approximately one spline length to shoot
                    shoot, x = length_to(x + delta)
                upper_bound = x # upper bound is direct first overshoot (separated by delta from the lower bound)
            
            elif shoot >= progress: # equality not very important
                while shoot >= progress:
                    upper_bound = x
                    # remove approximately one splien length to shoot
                    shoot, x = length_to(x - delta)
                lower_bound = x # lower bound is direct first undershoot (separated by delta from the upper bound)

            boundaries = [lower_bound, upper_bound]

        # Perform a more precise search between the two computed bounds

        guesses = np.linspace(boundaries[0], boundaries[1], precision)

        # Evaluate progress along the (extrapolated) spline
        # As arclength is expensive and cannot take multiple points
        # at the same time, it is faster to use a for loop
        past = -1
        best_guess = -1
        best_length = -1
        for guess in guesses:
            guess_length = arclength(self.polynomial, first_x, guess)
            if abs(progress - guess_length) > abs(progress - past): # if we did worst than before
                break
            best_guess = guess
            best_length = guess_length
            past = guess_length

        rotated_point = np.array([best_guess, self.polynomial(best_guess)])
        point = reverse_transformation(np.array([rotated_point]), self.Q, self.translation_vector)

        return point, best_length, rotated_point, best_guess
    def getderiv(self,x):
                point = reverse_transformation(np.array([x]), self.Q, self.translation_vector)
                return self.first_der(point)




def interpolate(spline: Spline, number: int, bounds: Tuple[float, float] = None):
    """
    Interpolate received data by finding points on the given spline

    Parameters
    ----------
    spline : raceline.Spline
        a cubic spline

    number : int
        number of points to generate, including bounds

    bounds : tuple[float, float]
        the bounds of the evaluations of the spline

    Returns
    -------
    npt.NDArray[npt.NDArray[np.float64]]
        a list of points interpolated from the spline definition.
        they can be extrapolated depending on the bounds
    """

    if bounds == None:
        bounds = (spline.rotated_points[0][0], spline.rotated_points[-1][0])
    
    evaluation_range = np.linspace(bounds[0], bounds[1], number)

    points = np.vstack((evaluation_range, spline.polynomial(evaluation_range))).T
    return reverse_transformation(points, spline.Q, spline.translation_vector)


def rotation_matrix_gen(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Parameters
    ----------
    points : npt.NDArray[np.float64] (degree+1, 2)
        points to be rotated


    Returns
    -------
    npt.NDArray[np.float64]
        rotation matrix that will rotated points such that the first and second points are on the same axis

    """
    
    difference = points[-1] - points[0]

    distance = np.linalg.norm(difference, axis=0) # mesure distance between first and last point

    cos = difference[0] / distance
    sin = difference[1] / distance

    return np.array([
        [cos, sin],
        [-sin, cos]
    ])


def get_translation_vector(group: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """

    Parameters
    ----------
    group : npt.NDArray[np.float64]
        group of points that should be translated


    Returns
    -------
    npt.NDArray[np.float64]
        an array that would translate the first point of `group` to (0,0)

    """
    
    return group[0]


def transform_points(
    points: npt.NDArray[np.float64],
    Q: npt.NDArray[np.float64],
    translation_vector: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """

    Parameters
    ----------
    points : npt.NDArray[np.float64] (degree+1, 2)
        points to be transformed

    Q : npt.NDArray[np.float64] (2, 2)
        rotation matrix

    translation_vector : npt.NDArray[np.float64] (2,)
        translation vector


    Returns
    -------
    npt.NDArray[np.float64] (degree+1, 2)
        points that are rotated such that the first and last points are on the same axis and the first point is at (0,0)

    """

    return (points - translation_vector) @ Q.T

def reverse_transformation(
    points: npt.NDArray[np.float64],
    Q: npt.NDArray[np.float64],
    translation_vector: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Reverse the transformation of points used to deal with splines
    
    Parameters
    ----------
    points : npt.NDArray[np.float64] (degree+1, 2)
        points to be transformed

    Q : npt.NDArray[np.float64] (2, 2)
        rotation matrix

    translation_vector : npt.NDArray[np.float64] (2,)
        translation vector


    Returns
    -------
    npt.NDArray[np.float64] (degree+1, 2)
        points that are derotated and re-translated to revert transformation of points for splines
    """
    return points @ Q + translation_vector

def lagrange_gen(points: npt.NDArray[np.float64]) -> np.poly1d:
    """

    Parameters
    ----------
    points : npt.NDArray[np.float64] (degree+1, 2)
        points to make lagrange polynomial from


    Returns
    -------
    np.poly1d
        lagrange polynomial from points

    """
    lagrange_poly: np.poly1d = np.poly1d(0.0)

    for point in points:
        non_root: npt.NDArray[np.float64] = point[0]

        roots: npt.NDArray[np.float64] = points[:, 0][points[:, 0] != point[0]]
        
        poly_roots: np.poly1d = np.poly1d(c_or_r=roots, r=True)
        poly_div: np.poly1d = poly_roots / poly_roots(non_root)
        poly_mut: np.poly1d = poly_div * point[1]
        
        lagrange_poly: np.poly1d = lagrange_poly + poly_mut
    
    return lagrange_poly


def arclength(poly: np.poly1d, first_x: float, last_x: float) -> float:
    """

    Parameters
    ----------
    poly : np.poly1d
        polynomial to be integrateed

    first_x : float
        first bound of integration, x coordinate of first point of polynomial

    last_x : float
        second bound of integration, x coordinate of last point of polynomial


    Returns
    -------
    float
        arc length of `poly` with the bounds of `first_x` and `last_x`

    """
    poly_der: np.poly1d = np.polyder(poly)
    new_poly: np.poly1d = np.polymul(poly_der, poly_der) + 1
    
    result: float
    result, _ = integrate.quad(lambda t: np.sqrt(new_poly(t)), first_x, last_x)

    return result

def raceline_gen(
    res: npt.NDArray, path_id=randint(0, sys.maxsize), points_per_spline=(preferred_degree+1), loop=True
) -> Tuple[List[Spline], npt.NDArray[np.float64]]:
    """
    Generate a path following the given points.
    Parameters
    ----------
    res : npt.NDArray[np.float64]
        coordinates of points of the raceline in order
    path_id : int
        a identifier that can be used to mark the recognize splines belonging to the same path
    loop : boolean
        Determines if the path should loop if there are not enough points.
        If set to True and not enough points exist to form the last spline, it assumes a loop 
        and will take again points from the beginning to close the loop.
    Returns
    -------
    tuple[[Spline, ...], [float, ...]]
        [Spline, ...] : splines that make up the raceline in order
        [float, ...] : prefix sum of the lengths of each spline
    """

    #assert xres.shape == yres.shape
    #assert len(xres.shape) == 1
    
    #points: npt.NDArray[np.float64] = np.vstack((xres, yres)).T

    n: int = len(res)
    splines: List[Spline] = []
    points=res

    shift: int = points_per_spline - 1 - overlap # shift between two groups of points

    group_numbers: int = n // shift
    
    if loop:
        group_numbers += int(n % shift != 0) # integer ceil: last spline can overlap first one

    group_indices: npt.NDArray[np.int8] = (np.arange(points_per_spline)[None, :] + (np.arange(group_numbers) * shift)[:, None]) % n

    groups: npt.NDArray[np.float64] = points[group_indices]

    lengths: npt.NDArray[np.float64] = np.zeros((group_numbers))  # Prefix sum of lengths that is 0-indexed

    for i in range(group_numbers):
        group: npt.NDArray[np.float64] = groups[i]

        Q: npt.NDArray[np.float64] = rotation_matrix_gen(group)

        translation_vector: npt.NDArray[np.float64] = get_translation_vector(group)
        rotated_points: npt.NDArray[np.float64] = transform_points(group, Q, translation_vector)

        interpolation_poly: np.poly1d = lagrange_gen(rotated_points)
                
        spline: Spline = Spline(
            polynomial=interpolation_poly,
            
            points=group,
            rotated_points=rotated_points,
            Q=Q,
            translation_vector=translation_vector,

            first_der=np.polyder(interpolation_poly, 1),
            second_der=np.polyder(interpolation_poly, 2),
            path_id=path_id,
            sort_index=i
        )

        splines.append(spline)

        lengths[i] = spline.length

        '''
        Graphically checking interpolation:

        import matplotlib.pyplot as plt

        x = np.linspace(rotated_points[0], rotated_points[-1])
        y = np.polyval(interpolation_poly, x)

        plt.plot(x, y, "r")
        plt.scatter(rotated_points[:, 0], rotated_points[:, 1])
        plt.show()
        '''

    cumulative_lengths = np.cumsum(lengths)

    return (splines, cumulative_lengths)
