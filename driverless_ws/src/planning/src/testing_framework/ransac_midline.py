import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import time

NUM_ITERS = 20
INLIER_THRESH = 75 # max error to be considered an inlier
CURVE_PADDING = 30

def ransac(points):
    best_poly = None
    best_inliers = 0
    best_inlier_dist = float('inf')
    all_inliers = []
    points_sampled = None

    for _ in range(NUM_ITERS):
        for s in range(4, 7):
            sampled_row_idxs = np.random.choice(points.shape[0], size=s, replace=False)
            sampled_rows = points[sampled_row_idxs, :]
            sample_x = sampled_rows[:, 0]
            sample_y = sampled_rows[:, 1]
            poly_coeffs = np.polyfit(sample_x, sample_y, 3)
            poly = np.poly1d(poly_coeffs)
            # poly = lagrange_gen(sampled_rows)
            
            # ---- OPTIMIZING Y POSITIONS ----
            inlier_list = []
            inliers = 0
            inlier_total_dist = 0
            for i in range(points.shape[0]):
                x0, y0 = points[i, 0], points[i, 1]
                distance_func = lambda x: np.abs(y0 - poly(x))
                result = minimize_scalar(distance_func)
                spline_x = result.x
                spline_y = poly(spline_x)
                dist = np.sqrt(np.abs(spline_x-x0)**2 + np.abs(spline_y-y0)**2)
                if dist < INLIER_THRESH:
                    inliers += 1
                    inlier_total_dist += dist
                    inlier_list.append((x0, y0))
        
            # ---- ATTEMPT AT USING ANTOINE CODE - DOESNT WORK ----
            # inliers = 0
            # for i in range(points.shape[0]):
            #     x, y = np.array([points[i, 0]]), np.array([points[i, 1]])
            #     poly_coeffs = np.array([lagrange_poly.coeffs])
            #     poly_roots = np.array([np.roots(lagrange_poly)])
            #     print(lagrange_poly.coeffs)
            #     res = get_closest_point(x, y, poly_coeffs, poly_roots)
            #     spline_x = res[0]
            #     spline_y = lagrange_poly(spline_x)
                
            #     dist = np.sqrt((spline_x - points[i, 0])**2 + (spline_y - points[i, 1])**2)
            #     print(dist)
            #     if dist < INLIER_THRESH:
            #         inliers += 1

            # ---- NAIVE APPROACH - JUST COMPARE Ys ----
            # inliers = 0
            # for i in range(points.shape[0]):
            #     pt = points[i, :]
            #     y_spline = lagrange_poly(pt[0])
            #     error = np.abs(y_spline - pt[1])
            #     if error < INLIER_THRESH:
            #         inliers += 1
        
            if inliers > best_inliers or (inliers == best_inliers and inlier_total_dist < best_inlier_dist):
                best_poly = poly
                best_inliers = inliers
                best_inlier_dist = inlier_total_dist
                points_sampled = sampled_rows
                all_inliers = inlier_list
    return best_poly, best_inliers, best_inlier_dist, points_sampled, all_inliers

def get_closest_point(
    x,
    y,
    poly_coeffs,
    poly_roots,
    precision=2,
    samples=5
):
    # retrieve coefficients
    a = poly_coeffs[:, 0, None]
    b = poly_coeffs[:, 1, None]
    c = poly_coeffs[:, 2, None]
    d = poly_coeffs[:, 3, None]

    # compute coefficients for the distance function
    c1 = x**2 + y**2 - 2*y*a + a**2
    c2 = 2*(-x - y*b + b*a)
    c3 = 1 - 2*y*c + 2*c*a + b**2
    c4 = 2*(d*a + b*c - y*d)
    c5 = 2*b*d + c**2
    c6 = 2*c*d
    c7 = d**2

    distance_coeffs = np.array([c1, c2, c3, c4, c5, c6, c7])
    distance_deriv_coeffs = np.array([1,2,3,4,5,6,7])[:, None, None] * np.array([c2, c3, c4, c5, c6, c7, np.zeros(c1.shape)])
    distance_double_deriv_coeffs = np.array([2,6,12,20,30,42,0])[:, None, None] * np.array([c3, c4, c5, c6, c7, np.zeros(c1.shape), np.zeros(c1.shape)])

    distance_coeffs = np.swapaxes(distance_coeffs, 0, 1)
    distance_deriv_coeffs = np.swapaxes(distance_deriv_coeffs, 0, 1)
    distance_double_deriv_coeffs = np.swapaxes(distance_double_deriv_coeffs, 0, 1)

    # initial guesses
    x = poly_roots[:, -1, None] # shape=(#polys, #guesses_per_poly)

    for i in range(len(poly_roots[0])-1):
        between = np.linspace(poly_roots[:, i], poly_roots[:, i+1], num=samples, endpoint=False, axis=1)
        x = np.concatenate((x, between), axis=1)

    print(x)

    for i in range(precision):
        powers = np.apply_along_axis(lambda x: np.vander(x, 7, increasing=True), 1, x) # shape=(#polys, #guesses, 7)

        # evaluate first and second derivatives with the guesses

        ddx = powers @ distance_deriv_coeffs # shape=(#polys, #guesses, 1)

        dddx = powers @ distance_double_deriv_coeffs # shape=(#polys, #guesses, 1)

        dddx[dddx == 0] = 0.001 # we want to avoid division by zero

        # iteration of Newton's method applied to derivative
        # x_{n+1} = x_n - f'(x_n)/f''(x_n)

        x = x - (ddx / dddx)[..., 0]

    x = np.apply_along_axis(lambda i: np.clip(i, poly_roots[:, 0], poly_roots[:, -1]), 0, x)

    powers = np.apply_along_axis(lambda x: np.vander(x, 7, increasing=True), 1, x) # shape=(#polys, #guesses, 7)

    distances = powers @ distance_coeffs # shape=(#polys, #guesses, 1)

    min_indices = np.argmin(distances, axis=1)[:, 0] # minimum for each polynom

    di = np.arange(0, poly_roots.shape[0] * len(distances[0]), len(distances[0]))

    min_x = np.take(x, di + min_indices)

    return (min_x,  np.take(distances, di + min_indices), x)

def lagrange_gen(coordinates):
    """
    Generate a Lagrange polynomial using the given coordinates.

    Parameters:
    - coordinates: Nx2 NumPy array representing 2D coordinates (x, y).

    Returns:
    - lagrange_poly: np.poly1d object representing the Lagrange polynomial.
    """

    # Extract x and y coordinates
    x_values = coordinates[:, 0]
    y_values = coordinates[:, 1]

    # Number of points
    num_points = len(x_values)

    # Initialize the Lagrange polynomial
    lagrange_poly = np.poly1d(0.0)

    for i in range(num_points):
        # Calculate the Lagrange basis polynomial for the current point
        basis_poly = np.poly1d(1.0)
        for j in range(num_points):
            if i != j:
                basis_poly *= np.poly1d([1, -x_values[j]]) / (x_values[i] - x_values[j])

        # Add the term to the Lagrange polynomial
        lagrange_poly += y_values[i] * basis_poly

    return lagrange_poly

# def lagrange_gen(points):
#     """
#     Parameters
#     ----------
#     points : npt.NDArray[np.float64]
#         points to make lagrange polynomial from
#     Returns
#     -------
#     np.poly1d
#         lagrange polynomial from points
#     """
#     lagrange_poly = np.poly1d(0.0)
#     n_row, n_col = points.shape
#     print(n_row, n_col)
#     for i in range(n_col):
#         non_root = points[i]
#         print(non_root)
#         roots = tuple(
#             [point for point in points[0, :] if point != non_root]
#         )
        
#         poly_roots = np.poly1d(c_or_r=roots, r=True)
#         poly_div = poly_roots / poly_roots(non_root)
#         poly_mut = poly_div * points[n_col + i]
        
#         lagrange_poly = lagrange_poly + poly_mut
            
#     return lagrange_poly

def read_points_from_file(file_path):
    points = np.loadtxt(file_path, delimiter=',')
    return points

def main():
    file_path = 'clicked_points.csv'
    clicked_points = read_points_from_file(file_path)
    clicked_points[:, [0, 1]] = clicked_points[:, [1, 0]]

    best_poly, best_inliers, best_inlier_dist, sampled_points, all_inliers = ransac(clicked_points)

    print(f"BEST POLY: {str(best_poly)}")
    print(f"INLIERS FOR BEST SPLINE: {best_inliers}")
    print(f"INLIERS DIST FOR BEST SPLINE: {best_inlier_dist}")

    x_all_pts = clicked_points[:, 0]
    y_all_pts = clicked_points[:, 1]

    x_sampled_pts = sampled_points[:, 0]
    y_sampled_pts = sampled_points[:, 1]
    
    min_x = np.min(x_all_pts)
    max_x = np.max(x_all_pts)
    
    min_y = np.min(y_all_pts)
    max_y = np.max(y_all_pts)

    inlier_x = [x for x, _ in all_inliers]
    inlier_y = [y for _, y in all_inliers]

    # best_poly = np.poly1d(np.polyfit(inlier_x, inlier_y, 3))

    x_curve = np.linspace(min_x, max_x, 1000)
    y_curve = best_poly(x_curve)

    for i in range(clicked_points.shape[0]):
        x0, y0 = clicked_points[i, 0], clicked_points[i, 1]
        distance_func = lambda x: np.abs(y0 - best_poly(x))
        result = minimize_scalar(distance_func)
        spline_x = result.x
        spline_y = best_poly(spline_x)
        dist = np.sqrt(np.abs(spline_x-x0)**2 + np.abs(spline_y-y0)**2)
        print(f"({x0}, {y0}) -> ({spline_x}, {spline_y}) | Dist: {dist}")

    plt.scatter(x_curve, y_curve, label="Curve of Best Fit", color="blue")
    plt.scatter(x_all_pts, y_all_pts, label="All Cones", color="red")
    plt.scatter(x_sampled_pts, y_sampled_pts, label="Sampled Cones", color="green")
    plt.scatter(inlier_x, inlier_y, label="Inliers", color="orange")

    plt.xlim([min_x-CURVE_PADDING, max_x+CURVE_PADDING])
    plt.ylim([min_y-CURVE_PADDING, max_y+CURVE_PADDING])
    plt.legend()
    plt.show()

main()
