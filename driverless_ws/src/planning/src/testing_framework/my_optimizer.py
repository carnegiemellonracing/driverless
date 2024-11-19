import numpy as np
import random

# Compute the derivative of the cubic spline at x
def derivative(spline, x):
    return 3 * spline[0] * (x ** 2) + 2 * spline[1] * x + spline[2]

# Least squares method to fit a cubic spline through 3 points
def least_squares(x1, y1, x2, y2, x3, y3):
    A = np.array([[x1 ** 3, x1 ** 2, x1, 1], [x2 ** 3, x2 ** 2, x2, 1], [x3 ** 3, x3 ** 2, x3, 1]])
    Y = np.array([y1, y2, y3])
    x_hat = np.linalg.solve(A.T @ A, A.T @ Y)
    return x_hat

# Calculate y value of the cubic spline at x
def y_value(cubic_spline, x):
    return cubic_spline[0] * (x ** 3) + cubic_spline[1] * (x ** 2) + cubic_spline[2] * x + cubic_spline[3]

# Calculate the derivative (slope) of the cubic spline at x
def spline_derivative(cubic_spline, x):
    return 3 * cubic_spline[0] * (x ** 2) + 2 * cubic_spline[1] * x + cubic_spline[2]

# Calculate the distance between two points (x1, y1) and (x2, y2)
def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Calculate arc length between two points x1 and x2 using numerical integration (approximating with sum of small segments)
def arc_length(cubic_spline, x1, x2, num_points=100):
    x_vals = np.linspace(x1, x2, num_points)
    #y_vals = [y_value(cubic_spline, x) for x in x_vals]
    derivatives = [spline_derivative(cubic_spline, x) for x in x_vals]
    length = 0.0
    for i in range(1, len(x_vals)):
        # Use the approximation for the arc length between two consecutive points
        dx = x_vals[i] - x_vals[i - 1]
        dy = derivatives[i] * dx
        ds = np.sqrt(dx**2 + dy**2)
        length += ds
    return length

# Total arc length for the entire path from start to end point
def total_arc_length(cubic_spline, start_x, end_x, num_points=100):
    arc_len = arc_length(cubic_spline, start_x, end_x, num_points)
    return arc_len

# Compute the cost based on arc length (objective function)
def compute_cost(cubic_spline, target_points, blue_cones, yellow_cones):
    # Total arc length (objective function)
    arc_len = total_arc_length(cubic_spline, target_points[0][0], target_points[-1][0])

    # Penalty for cone proximity
    cone_penalty = 0
    penalty_factor = 2
    for cone_x, cone_y in yellow_cones:
        spline_y = y_value(cubic_spline, cone_x)
        distance_to_cone = spline_y - cone_y
        if distance_to_cone > 0:
            cone_penalty += penalty_factor * distance_to_cone
        # elif distance_to_cone<0.5:
        #     cone_penalty += penalty_factor

    for cone_x, cone_y in blue_cones:
        spline_y = y_value(cubic_spline, cone_x)
        distance_to_cone = spline_y - cone_y
        if distance_to_cone < 0:
            cone_penalty += penalty_factor * abs(distance_to_cone)

    # Total cost is arc length + cone proximity penalty
    total_cost = arc_len + cone_penalty
    return total_cost

# Perturb the solution slightly (small random change to the spline coefficients)
def perturb_solution(spline, perturbation_factor=0.1):
    new_spline = spline + np.random.uniform(-perturbation_factor, perturbation_factor, len(spline))
    return new_spline

# Acceptance probability function for simulated annealing
def acceptance_probability(current_cost, new_cost, temperature):
    if new_cost < current_cost:
        return 1.0  # Always accept better solutions
    return np.exp((current_cost - new_cost) / temperature)  # Accept worse solutions with a probability

# Simulated Annealing approach
def simulated_annealing(cubic_spline, target_points, blue_cones, yellow_cones, initial_temp=1000, cooling_rate=0.99, max_iterations=1000):
    current_solution = cubic_spline
    current_cost = compute_cost(cubic_spline, target_points, blue_cones, yellow_cones)
    best_solution = current_solution
    best_cost = current_cost

    temperature = initial_temp
    iteration = 0

    while iteration < max_iterations:
        # Generate a neighbor solution by making a small change
        new_solution = perturb_solution(current_solution)
        new_cost = compute_cost(new_solution, target_points, blue_cones, yellow_cones)
        
        # Accept the new solution based on the cost difference and temperature
        if acceptance_probability(current_cost, new_cost, temperature) > random.random():
            current_solution = new_solution
            current_cost = new_cost

            # Update the best solution if the new solution is better
            if current_cost < best_cost:
                best_solution = current_solution
                best_cost = current_cost
        
        # Cool down the temperature
        temperature *= cooling_rate
        iteration += 1

    return best_solution, best_cost

# Gradient Descent with arc length-based cost function
def gradient_descent_update(cubic_spline, target_points, blue_cones, yellow_cones, learning_rate):
    base_cost = compute_cost(cubic_spline, target_points, blue_cones, yellow_cones)
    epsilon = 0.01
    gradients = []

    # Calculate gradients numerically
    for i in range(len(cubic_spline)):
        perturbed_spline = cubic_spline.copy()
        perturbed_spline[i] += epsilon
        perturbed_cost = compute_cost(perturbed_spline, target_points, blue_cones, yellow_cones)
        gradient = (perturbed_cost - base_cost) / epsilon
        gradients.append(gradient)

    # Update spline coefficients
    new_spline = [
        coeff - learning_rate * grad
        for coeff, grad in zip(cubic_spline, gradients)
    ]
    return new_spline

# Main optimizer function (using arc length as the objective function)
def run_optimizer(blue_cones_x, blue_cones_y, yellow_cones_x, yellow_cones_y, num_restarts=10):
    points_x = [-13, -3.5, 6]
    points_y = [8, -2, 8]
    
    target_points = list(zip(points_x, points_y))
    blue_cones = list(zip(blue_cones_x, blue_cones_y))
    yellow_cones = list(zip(yellow_cones_x, yellow_cones_y))

    best_solution = None
    best_cost = float('inf')

    for _ in range(num_restarts):
        cubic_spline = least_squares(points_x[0], points_y[0], points_x[1], points_y[1], points_x[2], points_y[2])
        
        # Run simulated annealing to optimize the spline
        optimized_solution, optimized_cost = simulated_annealing(cubic_spline, target_points, blue_cones, yellow_cones)

        # If the current solution is better, update the best solution
        if optimized_cost < best_cost:
            best_solution = optimized_solution
            best_cost = optimized_cost

    # Final interpolation and cropping of the optimized spline
    x_vals, y_vals = interpolate_points(best_solution, points_x[0], points_x[2], 25)
    cropped_points = crop(x_vals, y_vals, points_y[1] - 2, points_y[0] + 2)
    return np.array(cropped_points[0]), np.array(cropped_points[1])

# crop out points if outside the range y1, y2
def crop(x_coords, y_coords, y1, y2):
    cropped_x = [x for x, y in zip(x_coords, y_coords) if y1 <= y <= y2]
    cropped_y = [y for y in y_coords if y1 <= y <= y2]
    return cropped_x, cropped_y

# interpolate points on the cubic spline
def interpolate_points(cubic_spline, start_x, end_x, frequency=25):
    distance = abs(start_x - end_x)
    x_vals = [start_x + distance / frequency * i for i in range(frequency)]
    y_vals = [y_value(cubic_spline, x) for x in x_vals]
    return x_vals, y_vals
