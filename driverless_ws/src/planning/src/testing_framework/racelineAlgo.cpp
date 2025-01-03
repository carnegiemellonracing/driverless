#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

using namespace Eigen;
using namespace std;

// Define a type alias for a 2D point represented as a pair of doubles (x, y)
typedef pair<double, double> Point;

// Calculate the cubic spline coefficients between two points using least squares
tuple<double, double, double, double> cubicSplineLeastSquares(const Point& point1, const Point& point2) {
    double x1 = point1.first, y1 = point1.second;
    double x2 = point2.first, y2 = point2.second;

    Matrix2d A;
    A << pow(x1, 3), pow(x1, 2), x1, 1,
         pow(x2, 3), pow(x2, 2), x2, 1;

    Vector2d y(y1, y2);

    // Solve the least squares problem to find the cubic coefficients
    Vector4d coefficients = A.bdcSvd(ComputeThinU | ComputeThinV).solve(y);

    return {coefficients[0], coefficients[1], coefficients[2], coefficients[3]};
}

// Compute the y-values for given x-values based on cubic spline coefficients
vector<double> interpolateCubic(const tuple<double, double, double, double>& coefficients, const vector<double>& x_values) {
    vector<double> y_values;
    auto [a, b, c, d] = coefficients;
    for (double x : x_values) {
        y_values.push_back(a * pow(x, 3) + b * pow(x, 2) + c * x + d);
    }
    return y_values;
}

// Compute a point along the line connecting two points, at a specified distance from one of them
Point pointAlongLine(const Point& blue, const Point& yellow, double distance) {
    Vector2d direction = Vector2d(blue.first, blue.second) - Vector2d(yellow.first, yellow.second);
    Vector2d unit_direction = direction / direction.norm();
    Vector2d target = Vector2d(yellow.first, yellow.second) + distance * unit_direction;
    return {target[0], target[1]};
}

// Calculate initial points for the optimizer based on the positions of blue and yellow cones
vector<Point> initialPoints(const vector<Point>& blueCones, const vector<Point>& yellowCones, double d1, double d2) {
    Point blueStart = blueCones.front();
    Point yellowStart = yellowCones.front();
    Point blueMid = blueCones[blueCones.size() / 2];
    Point yellowMid = yellowCones[yellowCones.size() / 2];
    Point blueEnd = blueCones.back();
    Point yellowEnd = yellowCones.back();

    vector<Point> points;
    points.push_back({(blueStart.first + yellowStart.first) / 2, (blueStart.second + yellowStart.second) / 2});
    points.push_back(pointAlongLine(blueMid, yellowMid, d1));
    points.push_back(pointAlongLine(blueEnd, yellowEnd, d2));

    return points;
}

// Optimize the race line through the cones and compute the cubic spline coefficients
pair<vector<double>, vector<double>> runOptimizer(const vector<double>& blueConesX, const vector<double>& blueConesY, const vector<double>& yellowConesX, const vector<double>& yellowConesY) {
    vector<Point> blueCones, yellowCones;

    // Combine x and y coordinates into Point vectors
    for (size_t i = 0; i < blueConesX.size(); ++i) {
        blueCones.emplace_back(blueConesX[i], blueConesY[i]);
    }

    for (size_t i = 0; i < yellowConesX.size(); ++i) {
        yellowCones.emplace_back(yellowConesX[i], yellowConesY[i]);
    }

    // Sort cones by x-coordinate
    sort(blueCones.begin(), blueCones.end());
    sort(yellowCones.begin(), yellowCones.end());

    // Compute initial points
    vector<Point> points = initialPoints(blueCones, yellowCones, 2.0, 2.5);

    // Extract x and y coordinates of the initial points
    double x1 = points[0].first, y1 = points[0].second;
    double x2 = points[1].first, y2 = points[1].second;
    double x3 = points[2].first, y3 = points[2].second;

    // Compute slopes of the start and end segments
    double k = (blueCones[0].second - blueCones[1].second) / (blueCones[0].first - blueCones[1].first);
    double l = (blueCones.back().second - blueCones[blueCones.size() - 2].second) / (blueCones.back().first - blueCones[blueCones.size() - 2].first);

    // Setup the right-hand side vectors for x and y coordinates
    VectorXd b_x(8), b_y(8);
    b_x << x1, x2, x2, x3, k, l, 0, 0;
    b_y << y1, y2, y2, y3, k, l, 0, 0;

    // Hardcoded inverse of the system matrix
    MatrixXd A_inverse(8, 8);
    A_inverse << 10, -10, -6, 6, 3, -1, 2, 0.25,
                -9, 9, 3, -3, -3.5, 0.5, -1, -0.125,
                 0, 0, 0, 0, 1, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0,
                -6, 6, 10, -10, -1, 3, -2, 0.25,
                 6, -6, -6, 6, 1, -1, 2, -0.25,
                -1.5, 1.5, -1.5, 1.5, -0.25, -0.25, -0.5, 0.0625,
                 0, 0, 1, 0, 0, 0, 0, 0;

    // Solve for the coefficients using the inverse matrix
    VectorXd X = A_inverse * b_x;
    VectorXd Y = A_inverse * b_y;

    // Generate 500 equally spaced points in the range [0, 0.5]
    vector<double> base;
    for (int i = 0; i < 500; ++i) base.push_back(i / 1000.0);

    // Compute cubic spline outputs for x and y coordinates
    auto spline_x1 = interpolateCubic({X[0], X[1], X[2], X[3]}, base);
    auto spline_x2 = interpolateCubic({X[4], X[5], X[6], X[7]}, base);
    auto spline_y1 = interpolateCubic({Y[0], Y[1], Y[2], Y[3]}, base);
    auto spline_y2 = interpolateCubic({Y[4], Y[5], Y[6], Y[7]}, base);

    // Combine the results from the two spline segments
    spline_x1.insert(spline_x1.end(), spline_x2.begin(), spline_x2.end());
    spline_y1.insert(spline_y1.end(), spline_y2.begin(), spline_y2.end());

    return {spline_x1, spline_y1};
}