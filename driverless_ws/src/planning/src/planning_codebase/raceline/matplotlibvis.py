import numpy as np
import matplotlib.pyplot as plt

def evaluate_polynomial(coeffs, t):
    """
    Evaluate a cubic polynomial given its coefficients and parameter t.
    coeffs: List of coefficients [a3, a2, a1, a0] for t^3, t^2, t, constant.
    t: Array of parameter values.
    """
    return coeffs[3] * t**3 + coeffs[2] * t**2 + coeffs[1] * t + coeffs[0]

def plot_parametric_cubic_polynomials(grouped_polynomials, t_range=(0, 1), num_points=100):
    """
    Plot a list of parametric cubic polynomials (x(t), y(t)).
    polynomials: List of tuples [(x_coeffs, y_coeffs), ...].
                 Each tuple contains the coefficients for x(t) and y(t).
    t_range: Tuple (start, end) for the parameter t.
    num_points: Number of points to evaluate for each polynomial.
    """
    t = np.linspace(t_range[0], t_range[1], num_points)
    
    plt.figure(figsize=(12, 8))
    
    colors = ['red', 'blue']
    for i, group in enumerate(grouped_polynomials):
        # Use a unique color for each group
        color = colors[i % len(colors)]
        if (i == 0):
            color = 'green'
        
        for x_coeffs, y_coeffs in group:
            # Evaluate x(t) and y(t)
            x_vals = evaluate_polynomial(x_coeffs, t)
            y_vals = evaluate_polynomial(y_coeffs, t)
            
            # Plot the parametric curve
            plt.plot(x_vals, y_vals, color=color)
    
    plt.title("Parametric Cubic Polynomials")
    plt.xlabel("x(t)")
    plt.ylabel("y(t)")
    plt.grid(True)
    plt.show()

# Example usage
# Define a list of polynomial coefficient pairs (x(t), y(t))
polynomials = [[
([-4,-2,4,-2],[0,1,2,-1])
,([-4,0,0,0],[2,2,0,0])
,([-4,0,0,0],[4,2,0,0])
,([-4,0,0,0],[6,2,0,0])
,([-4,0,0,0],[8,2,0,0])
,([-4,0,0,0],[10,2,0,0])
,([-4,0,0,0],[12,2,0,0])
,([-4,0,0,0],[14,2,0,0])
,([-4,0,0,0],[16,2,0,0])
,([-4,0,0,0],[18,2,0,0])
,([-4,0,0.133975,-0.133975],[20,2,0.5,-0.5])
],
[
([-4,-0.133975,-0.0358984,-0.0980762],[22,1.5,-0.866025,0.366025])
],
[
([-4.26795,-0.5,-0.330127,0.0980762],[23,0.866025,-0.0358984,-0.0980762])
],
[
([-5,-0.866025,0.232051,-0.366025],[23.7321,0.5,-0.330127,0.0980762])
,([-6,-1.5,-1,0.5],[24,0.133975,-0.267949,0.133975])
,([-8,-2,0,0],[24,0,0,0])
,([-10,-2,0,0],[24,0,0,0])
,([-12,-2,-0.5,0.5],[24,0,0.133975,-0.133975])
,([-14,-1.5,0.866025,-0.366025],[24,-0.133975,-0.0358984,-0.0980762])
],
[
([-15,-0.866025,0.0358984,0.0980762],[23.7321,-0.5,-0.330127,0.0980762])
],
[
([-15.7321,-0.5,-0.279473,0.511524],[23,-0.866025,0.232051,-0.366025])
],
[
([-16,0.475625,1.79195,-1.04837],[22,-1.5,-1,0.5])
],
[
([-14.7808,0.9144,0.3048,-0.6096],[20,-2,0,0])
],
[
([-14.1712,-0.3048,-1.8288,0.9144],[18,-2,0,0])
],
[
([-15.3904,-1.2192,0,0],[16,-2,0,0])
],
[
([-16.6096,-1.2192,-0.9144,0.9144],[14,-2,0,0])
],
[
([-17.8288,-0.3048,1.524,-0.6096],[12,-2,0,0])
],
[
([-17.2192,0.9144,1.2192,-0.9144],[10,-2,0,0])
],
[
([-16,0.6096,-1.2192,0.6096],[8,-2,0,0])
,([-16,0,-0.133975,0.133975],[6,-2,-0.5,0.5])
],
[
([-16,0.133975,0.0358984,0.0980762],[4,-1.5,1,-0.5])
],
[
([-15.7321,0.5,0.330127,-0.0980762],[3,-1,-0.5,0.5])
],
[
([-15,0.866025,-0.232051,0.366025],[2,-0.5,1,-0.5])
,([-14,1.5,1,-0.5],[2,0,0,0])
,([-12,2,0,0],[2,0,0,0])
,([-10,2,0,0],[2,0,0,0])
,([-8,2,0,0],[2,0,1,-1])
],
[
([-6,2,1,-1],[2,-1,-4,3])
]]

# Plot the polynomials
plot_parametric_cubic_polynomials(polynomials)