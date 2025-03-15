#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>
#include <cassert>
#include "svm.h" // libSVM headers

#include "cones.h"

std::pair<size_t, double> getClosestPointIdx(const std::vector<std::pair<double, double>>& points, const std::pair<double, double>& curr_point) {
    assert(!points.empty());

    size_t closest_idx = 0;
    double min_dist_squared = std::numeric_limits<double>::max();

    for (size_t i = 0; i < points.size(); ++i) {
        double dx = points[i].first - curr_point.first;
        double dy = points[i].second - curr_point.second;
        double dist_squared = dx * dx + dy * dy; 

        if (dist_squared < min_dist_squared) {
            min_dist_squared = dist_squared;
            closest_idx = i;
        }
    }

    // return the closest index and the square root of the minimum distance 
    return {closest_idx, std::sqrt(min_dist_squared)};
}

size_t getSplineStartIdx(std::vector<std::pair<double, double>>& points) {
    // gets index of point with lowest y-axis value in points

    // first find minimum points
    double min_y = std::numeric_limits<double>::max();
    for (std::pair<double, double>& point : points) {
        if (point.second < min_y) {
            min_y = point.second;
        }
    }

    // find points with y == min_y
    std::vector<size_t> idxs;
    for (size_t i = 0; i < points.size(); ++i) {
        if (points[i].second == min_y) {
            idxs.push_back(i);
        }
    }

    // take point closest to x = 0
    size_t closest_x_idx = idxs[0]; 
    double min_abs_x = std::abs(points[idxs[0]].first);
    for (size_t i = 1; i < idxs.size(); ++i) {
        double abs_x = std::abs(points[idxs[i]].first);
        if (abs_x < min_abs_x) {
            min_abs_x = abs_x;
            closest_x_idx = idxs[i];
        }
    }

    return closest_x_idx;
}


std::vector<std::pair<double, double>> sortBoundaryPoints(std::vector<std::pair<double, double>> points, double max_spline_length=17.5) {
    // initialize spline length and sorted points
    double spline_length = 0;
    std::vector<std::pair<double, double>> sorted_points;

    // start from the lowest point along the y-axis
    size_t idx = getSplineStartIdx(points);
    std::pair<double, double> curr_point = points[idx];

    // remove the element at idx
    std::vector<std::pair<double, double>> rem_points = points;
    rem_points.erase(rem_points.begin() + idx);

    // add current point to sorted points
    sorted_points.push_back(curr_point);

    while (!rem_points.empty() && spline_length < max_spline_length) {

        // find closest point to curr_point
        double dist;
        std::pair<size_t, double> values = getClosestPointIdx(rem_points, curr_point);
        idx = values.first;
        dist = values.second;
        spline_length = spline_length + dist;

        // update iterates
        curr_point = rem_points[idx];
        rem_points.erase(rem_points.begin() + idx);

        // add closest point to sorted points
        sorted_points.push_back(curr_point);
    }

    return sorted_points;
}


std::vector<std::pair<double, double>> cones_to_midline(Cones cones) {
    // get blue and yellow cones
    Cones::ConeData data = cones.toStruct();
    std::vector<std::vector<double>> blue_cones = data.blue_cones;
    std::vector<std::vector<double>> yellow_cones = data.yellow_cones;

    if (blue_cones.empty() && yellow_cones.empty()) {
        return std::vector<std::pair<double, double>>(); 
    }

    // augment dataset to make it better for SVM training
    cones.supplementCones();
    cones = cones.augmentConesCircle(cones, 10, 1.2);
    
    std::pair<std::vector<std::vector<double>>, std::vector<double>> xy;
    std::vector<std::vector<double>> X;
    std::vector<double> y;

    xy = cones.conesToXY(cones);
    X = xy.first;
    y = xy.second;

    for (double label : y) {
        std::cout << "Label: " << label << "\n";
    }

    // prepare SVM data
    svm_problem prob;
    prob.l = X.size(); // number of training examples
    prob.y = new double[prob.l]; // labels
    prob.x = new svm_node*[prob.l]; // feature vectors

    for (int i = 0; i < prob.l; ++i) {
        prob.y[i] = y[i]; // set the label for each example

        // create the feature vector
        prob.x[i] = new svm_node[X[i].size() + 1]; // +1 for the end marker
        for (size_t j = 0; j < X[i].size(); ++j) {
            prob.x[i][j].index = j + 1; // 1-based indexing for libSVM
            prob.x[i][j].value = X[i][j];
        }
        prob.x[i][X[i].size()].index = -1; // End marker
    }

    // set up SVM parameters
    svm_parameter param;
    param.svm_type = C_SVC;
    param.kernel_type = POLY;
    param.degree = 3;
    param.C = 10;
    param.coef0 = 1.0;
    param.gamma = 0; // polynomial kernel

    // train the SVM model
    svm_model* model = svm_train(&prob, &param);

    // free allocated memory 
    for (int i = 0; i < prob.l; ++i) {
        delete[] prob.x[i];
    }
    delete[] prob.x;
    delete[] prob.y;

    // create meshgrid
    double x_min = X[0][0], x_max = X[0][0];
    double y_min = X[0][1], y_max = X[0][1];

    for (const auto& row : X) {
        x_min = std::min(x_min, row[0]);
        x_max = std::max(x_max, row[0]);
        y_min = std::min(y_min, row[1]);
        y_max = std::max(y_max, row[1]);
    }

    x_min -= 1.0;
    x_max += 1.0;
    y_min -= 1.0;
    y_max += 1.0;

    // ranges for x and y
    std::vector<double> x_range, y_range;
    for (double x = x_min; x <= x_max; x += 0.1) {
        x_range.push_back(x);
    }
    for (double y = y_min; y <= y_max; y += 0.1) {
        y_range.push_back(y);
    }

    // create the meshgrid
    std::vector<std::vector<double>> xx, yy;
    for (double y : y_range) {
        std::vector<double> x_row;
        std::vector<double> y_row;
        for (double x : x_range) {
            x_row.push_back(x);
            y_row.push_back(y);
        }
        xx.push_back(x_row);
        yy.push_back(y_row);
    }

    // flatten xx, yy
    std::vector<std::vector<double>> svm_input;
    for (size_t i = 0; i < xx.size(); ++i) {
        for (size_t j = 0; j < xx[i].size(); ++j) {
            svm_input.push_back({xx[i][j], yy[i][j]});
        }
    }

    // predict using the SVM Model
    std::vector<double> Z_flattened;
    for (const std::vector<double>& input : svm_input) {
        svm_node* node = new svm_node[input.size() + 1];
        for (size_t i = 0; i < input.size(); ++i) {
            node[i].index = i + 1;
            node[i].value = input[i];
        }
        node[input.size()].index = -1; 
        Z_flattened.push_back(svm_predict(model, node));
        delete[] node;
    }

    // reshape Z to match the shape of xx and yy
    std::vector<std::vector<double>> Z;
    size_t rows = xx.size();
    size_t cols = xx[0].size();
    for (size_t i = 0; i < rows; ++i) {
        Z.emplace_back(Z_flattened.begin() + i * cols, Z_flattened.begin() + (i + 1) * cols);
    }

    // boundary detection
    std::vector<std::pair<double, double>> boundary_points;
    for (size_t i = 0; i < rows - 1; ++i) {
        for (size_t j = 0; j < cols - 1; ++j) {
            double Z_TL = Z[i][j];
            double Z_BR = Z[i + 1][j + 1];
            double Z_TR = Z[i][j + 1];
            double Z_BL = Z[i + 1][j];

            if (Z_TL != Z_BR || Z_TL != Z_TR || Z_TL != Z_BL) {
                boundary_points.emplace_back(xx[i + 1][j + 1], yy[i + 1][j + 1]);
            }
        }
    }

    // sort boundary points 
    boundary_points = sortBoundaryPoints(boundary_points);

    // downsample boundary points
    std::vector<std::pair<double, double>> downsampled;
    double accumulated_dist = 0.0;

    for (size_t i = 1; i < boundary_points.size(); ++i) {
        auto& p0 = boundary_points[i - 1];
        auto& p1 = boundary_points[i];
        double dist = std::sqrt(std::pow(p1.first - p0.first, 2) + std::pow(p1.second - p0.second, 2));
        accumulated_dist += dist;

        if (std::abs(accumulated_dist - 0.5) < 0.1) {
            downsampled.push_back(p1);
            accumulated_dist = 0.0;
        }

        if (accumulated_dist > 0.55) {
            accumulated_dist = 0.0;
        }
    }

    // free the SVM model
    svm_free_and_destroy_model(&model);

    return downsampled;
}