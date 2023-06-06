import math
import bisect
import numpy as np
import numpy.typing as npt
import cmrdv_ws.src.cmrdv_planning.planning_codebase.raceline.raceline as raceline
from dataclasses import dataclass
import torch as torch
from typing import Any, Callable, List, Tuple, Union

@dataclass
class Projection:
    """Projection class for storing result from minimization and related data"""

    progress: float
    min_index: int  # For testing
    min_distance: float  # For testing
    curvature: float
    velocity: float


#def get_curvature(spline: raceline.Spline, x: np.float64):
#    return get_curvature(spline.first_der, spline.second_der, x)

def get_curvature(
    poly_der_1: Callable[[Any], Any], poly_der_2: Callable[[Any], Any], min_x
):
    """
    Curvature at point(s) `min_x` based on 2d curvature equation https://mathworld.wolfram.com/Curvature.html

    Parameters
    ----------
    poly_der_1

    poly_der_2

    min_x


    Returns
    -------
    One or multiple floats corresponding to curvatures

    """
    #normalized first derivative
    curvature = poly_der_2(min_x) / (
        (1 + poly_der_1(min_x) ** 2) ** (3 / 2)
    )
    return curvature

def rotate_points(
    point: npt.NDArray[np.float64],
    poly_Qs: npt.NDArray[np.float64],
    poly_transMats: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Rotates points based on the rotation and transformation matrices

    Parameters
    ----------
    point : npt.NDArray[np.float64]

    poly_Qs : npt.NDArray[np.float64]

    poly_transMats : npt.NDArray[np.float64]


    Returns
    -------
    npt.NDArray[np.float64]

    """
    transform_point = point.T - poly_transMats
    rotated_point = poly_Qs @ transform_point[..., None]
    return rotated_point


def get_closest_distance(
    x_point: npt.NDArray[np.float64],
    y_point: npt.NDArray[np.float64],
    poly_coeffs: npt.NDArray[np.float64],
    poly_roots: npt.NDArray[np.float64],
    precision: int = 2,
    samples: int = 5,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Finds the point on each spline that minimizes the distance to the point.

    Uses Netwon's method for minimization

    https://drive.google.com/file/d/1MP-jhWPXpNb3WEztvSrkW7iTVYgKsy9l/view?usp=share_link for more details

    Parameters
    ----------
    x_point : npt.NDArray[np.float64]

    y_point : npt.NDArray[np.float64]

    poly_coeffs : npt.NDArray[np.float64]

    poly_roots : npt.NDArray[np.float64]


    Returns
    -------
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        npt.NDArray[np.float64] : the x-coordinates of the point that minimizes the distances
        npt.NDArray[np.float64] : the minimum distance from each spline to the point
    """
    n_coeffs: int = 7
    assert (
        poly_coeffs.shape[0]
        == poly_roots.shape[0]
        == x_point.shape[0]
        == y_point.shape[0]
    )
    a: npt.NDArray[np.float64] = np.array(poly_coeffs[:, 0, np.newaxis])

    b: npt.NDArray[np.float64] = np.array(poly_coeffs[:, 1, np.newaxis])
    c: npt.NDArray[np.float64] = np.array(poly_coeffs[:, 2, np.newaxis])
    d: npt.NDArray[np.float64] = np.array(poly_coeffs[:, 3, np.newaxis])
    c1: npt.NDArray[np.float64] = np.array(
        x_point**2 + y_point**2 - 2 * y_point * a + a**2
    )
    c2: npt.NDArray[np.float64] = 2 * (-x_point - y_point * b + b * a)
    c3: npt.NDArray[np.float64] = 1 - 2 * y_point * c + 2 * c * a + b**2
    c4: npt.NDArray[np.float64] = 2 * (d * a + b * c - y_point * d)
    c5: npt.NDArray[np.float64] = 2 * b * d + c**2
    c6: npt.NDArray[np.float64] = 2 * c * d
    c7: npt.NDArray[np.float64] = d**2

    distance_coeffs: npt.NDArray[np.float64] = np.array([c1, c2, c3, c4, c5, c6, c7])
    distance_deriv_coeffs: npt.NDArray[np.float64] = np.array([1, 2, 3, 4, 5, 6, 7])[
        :, np.newaxis, np.newaxis
    ] * np.array([c2, c3, c4, c5, c6, c7, np.zeros(c1.shape)])
    distance_double_deriv_coeffs: npt.NDArray[np.float64] = np.array(
        [2, 6, 12, 20, 30, 42, 0]
    )[:, np.newaxis, np.newaxis] * np.array(
        [c3, c4, c5, c6, c7, np.zeros(c1.shape), np.zeros(c2.shape)]
    )

    distance_coeffs: npt.NDArray[np.float64] = np.swapaxes(distance_coeffs, 0, 1)
    distance_deriv_coeffs: npt.NDArray[np.float64] = np.swapaxes(
        distance_deriv_coeffs, 0, 1
    )
    distance_double_deriv_coeffs: npt.NDArray[np.float64] = np.swapaxes(
        distance_double_deriv_coeffs, 0, 1
    )

    x: npt.NDArray[np.float64] = np.array(poly_roots[:, -1, np.newaxis])
    powers: npt.NDArray[np.float64 | np.signedinteger]
    assert len(poly_roots) > 0
    for i in range(len(poly_roots[0]) - 1):
        i: int
        between: npt.NDArray[np.float64] = np.linspace(
            poly_roots[:, i],
            poly_roots[:, i + 1],
            num=samples,
            endpoint=False,
            axis=1,
        )
        x = np.concatenate((x, between), axis=1)
    for i in range(precision):
        i: int
        powers = np.apply_along_axis(
            lambda x: np.vander(x, n_coeffs, increasing=True), 1, x
        )
        ddx: npt.NDArray[np.float64] = powers @ distance_deriv_coeffs
        dddx: npt.NDArray[np.float64] = powers @ distance_double_deriv_coeffs

        dddx[dddx == 0] = 0.001  # avoiding division by zero

        x = x - (ddx / dddx)[:, :, 0]

    x = np.apply_along_axis(
        lambda i: np.clip(i, poly_roots[:, 0], poly_roots[:, -1]), 0, x
    )
    powers = np.apply_along_axis(
        lambda x: np.vander(x, n_coeffs, increasing=True), 1, x
    )
    distances: npt.NDArray[np.float64] = powers @ distance_coeffs
    min_indices: npt.NDArray[np.int64] = np.argmin(distances, axis=1)[:, 0]
    di: npt.NDArray[np.signedinteger[typing.Any]] = np.arange(
        0,
        poly_roots.shape[0] * len(distances[0]),
        len(distances[0]),
    )
    min_x: npt.NDArray[np.float64] = np.take(x, di + min_indices)
    return (min_x, np.take(distances, di + min_indices))


def frenet(
    x: float,
    y: float,
    path: List[raceline.Spline],
    lengths: List[float],
    prev_progress: Union[float, None] = None,
    v_x: float = 0,
    v_y: float = 0,
) -> Projection:
    """
    Finds the progress (length) and curvature of point on a raceline generated from splines


    Parameters
    ----------
    x : float

    y : float

    path : tuple[raceline.Spline]

    lengths : tuple[float]

    prev_progress : float | None

    Returns
    -------
    Projection
    """
    assert len(path) == len(lengths)
    n: int = len(path)
    index_offset: int = 0
    indexes: List[int]
    if prev_progress:
        # Lengths must be sorted for bisect to work since its binary search
        assert all(lengths[i] <= lengths[i + 1] for i in range(len(lengths) - 1))
        index: int = bisect.bisect_left(
            lengths, prev_progress
        )  # get index where all elements in lengths[index:] are >= prev_progress
        indexes = [i % n for i in range(index, min(n, index + 30), 1)]
        index_offset = index
    else:
        indexes = list(range(len(path)))

    explore_space: Tuple[raceline.Spline] = tuple([path[i] for i in indexes])

    poly_coeffs: npt.NDArray = np.array(
        [np.flip(spline.polynomial.coef) for spline in explore_space], dtype=float
    )  # we need to flip coefficents here because the coefficients we use in get_closest_distance are in reverse order from the coefficients given by numpy.poly1d
    poly_roots: npt.NDArray = np.array(
        [poly.rotated_points[0, :] for poly in explore_space], dtype=float
    )
    poly_Qs: npt.NDArray = np.array([poly.Q for poly in explore_space])
    poly_transMat: npt.NDArray = np.array([poly.translation_vector for poly in explore_space])
    point: npt.NDArray = np.array([[x], [y]], dtype=float)
    rotated_points: npt.NDArray = rotate_points(
        point=point, poly_Qs=poly_Qs, poly_transMats=poly_transMat
    )
    opt_xs: npt.NDArray[np.float64]
    distances: npt.NDArray[np.float64]
    opt_xs, distances = get_closest_distance(
        x_point=rotated_points[:, 0],  # shape = (explore_space_len, 1)
        y_point=rotated_points[:, 1],  # shape = (explore_space_len, 1)
        poly_coeffs=poly_coeffs,
        poly_roots=poly_roots,
    )

    i: int = int(np.argmin(distances))
    min_index: int = (i + index_offset) % n
    min_polynomial: np.poly1d = path[i].polynomial
    min_x: np.float64 = opt_xs[i]
    curvature: float = float(
        get_curvature(
            poly_der_1=path[i].first_der,
            poly_der_2=path[i].second_der,
            min_x=min_x,
        )
    )
    assert min_index in indexes, "min_index is not in the range of indexes"
    extra_length: float = raceline.arclength(
        poly=min_polynomial, first_x=0, last_x=float(min_x)
    )
    mu = distances[i] # normal distance from curve
    velocity: float = (v_x * math.cos(mu) - v_y * math.sin(mu)) / (
        1 - distances[i] * curvature
    )
    result: Projection = Projection(
        progress=(0 if min_index == 0 else lengths[min_index - 1]) + extra_length,
        min_index=min_index,
        min_distance=distances[i],
        curvature=curvature,
        velocity=velocity,
    )
    return result
