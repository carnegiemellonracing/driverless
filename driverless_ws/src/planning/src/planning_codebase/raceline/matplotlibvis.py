import numpy as np
import matplotlib.pyplot as plt

def evaluate_polynomial(coeffs, t):
    """
    Evaluate a cubic polynomial given its coefficients and parameter t.
    coeffs: List of coefficients [a3, a2, a1, a0] for t^3, t^2, t, constant.
    t: Array of parameter values.
    """
    return coeffs[3] * t**3 + coeffs[2] * t**2 + coeffs[1] * t + coeffs[0]

def plot_parametric_cubic_polynomials(input_data, num_points=100):
    """
    Plot a list of parametric cubic polynomials (x(t), y(t)).
    input_data: List of tuples. Each tuple contains:
                - A list of tuples representing grouped parametric splines (x(t), y(t)).
                - A tuple (start_t, end_t) specifying the range of t for the group.
    num_points: Number of points to evaluate for each polynomial.
    """
    plt.figure(figsize=(12, 8))
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for group_index, (splines, t_range) in enumerate(input_data):
        group_size = len(splines)
        start_t, end_t = t_range
        group_color = colors[group_index % len(colors)]
        
        for spline_index, (x_coeffs, y_coeffs) in enumerate(splines):
            # Determine t_range for this spline
            if spline_index == 0:
                t_start, t_end = start_t, 1  # First spline gets group start_t
            elif spline_index == group_size - 1:
                t_start, t_end = 0, end_t  # Last spline gets group end_t
            else:
                t_start, t_end = 0, 1  # Intermediate splines default to [0, 1]
            
            t = np.linspace(t_start, t_end, num_points)
            
            # Evaluate x(t) and y(t)
            x_vals = evaluate_polynomial(x_coeffs, t)
            y_vals = evaluate_polynomial(y_coeffs, t)
            
            # Plot the parametric curve
            plt.plot(x_vals, y_vals, color=group_color, label=f'Group {group_index + 1}' if spline_index == 0 else None)
    
    plt.title("Parametric Cubic Polynomials")
    plt.xlabel("x(t)")
    plt.ylabel("y(t)")
    plt.grid(True)
    plt.legend()
    plt.show()

# Example Input Data
input_data = [([
([-4,1,-2,1],[0,0,4,-2])
,
([-4,0,0,0],[2,2,0,0])
,
([-4,0,0,0],[4,2,0,0])
,
([-4,0,0,0],[6,2,0,0])
,
([-4,0,0,0],[8,2,0,0])
,
([-4,0,0,0],[10,2,0,0])
,
([-4,0,0,0],[12,2,0,0])
,
([-4,0,0,0],[14,2,0,0])
,
([-4,0,0,0],[16,2,0,0])
,
([-4,0,0,0],[18,2,0,0])
,
([-4,0,0.133975,-0.133975],[20,2,0.5,-0.5])
],
 (0, 1)
),
([
([-4,-0.133975,-0.0358984,-0.0980762],[22,1.5,-0.866025,0.366025])
],
 (0, 1)
),
([
([-4.26795,-0.5,-0.330127,0.0980762],[23,0.866025,-0.0358984,-0.0980762])
],
 (0, 1)
),
([
([-5,-0.866025,0.232051,-0.366025],[23.7321,0.5,-0.330127,0.0980762])
,
([-6,-1.5,-1,0.5],[24,0.133975,-0.267949,0.133975])
,
([-8,-2,0,0],[24,0,0,0])
,
([-10,-2,0,0],[24,0,0,0])
,
([-12,-2,-0.5,0.5],[24,0,0.133975,-0.133975])
,
([-14,-1.5,0.866025,-0.366025],[24,-0.133975,-0.0358984,-0.0980762])
],
 (0, 1)
),
([
([-15,-0.866025,0.0358984,0.0980762],[23.7321,-0.5,-0.330127,0.0980762])
],
 (0, 1)
),
([
([-15.7321,-0.5,-0.279473,0.511524],[23,-0.866025,0.232051,-0.366025])
],
 (0, 1)
),
([
([-16,0.475625,1.79195,-1.04837],[22,-1.5,-1,0.5])
],
 (0, 1)
),
([
([-14.7808,0.9144,0.3048,-0.6096],[20,-2,0,0])
],
 (0, 1)
),
([
([-14.1712,-0.3048,-1.8288,0.9144],[18,-2,0,0])
],
 (0, 1)
),
([
([-15.3904,-1.2192,0,0],[16,-2,0,0])
],
 (0, 1)
),
([
([-16.6096,-1.2192,-0.9144,0.9144],[14,-2,0,0])
],
 (0, 1)
),
([
([-17.8288,-0.3048,1.524,-0.6096],[12,-2,0,0])
],
 (0, 1)
),
([
([-17.2192,0.9144,1.2192,-0.9144],[10,-2,0,0])
],
 (0, 1)
),
([
([-16,0.6096,-1.2192,0.6096],[8,-2,0,0])
,
([-16,0,-0.133975,0.133975],[6,-2,-0.5,0.5])
],
 (0, 1)
),
([
([-16,0.133975,0.0358984,0.0980762],[4,-1.5,1,-0.5])
],
 (0, 1)
),
([
([-15.7321,0.5,0.330127,-0.0980762],[3,-1,-0.5,0.5])
],
 (0, 1)
),
([
([-15,0.866025,-0.232051,0.366025],[2,-0.5,1,-0.5])
,
([-14,1.5,1,-0.5],[2,0,0,0])
,
([-12,2,0,0],[2,0,0,0])
,
([-10,2,0,0],[2,0,0,0])
,
([-8,2,0,0],[2,0,1,-1])
],
 (0, 1)
),
([
([-6,2,1,-1],[2,-1,-4,3])
],
 (0, 1)
)]

# Plot the parametric cubic polynomials
plot_parametric_cubic_polynomials(input_data)