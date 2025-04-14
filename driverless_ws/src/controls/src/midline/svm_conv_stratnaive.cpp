#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cassert>
#include <chrono>
#include <thread>
#include <cstring>
#include "svm.hpp" // libSVM headers

#include "cones.hpp"
#include "svm_conv.hpp"
#include <constants.hpp>
#define NUM_THREADS 4


namespace controls
{
namespace midline
{
namespace svm_naive {

typedef std::vector<std::pair<double, double>> conesList;


// predict the value of a node
double nodePredictor(const std::vector<double> &cone, const svm_model *model)
{
    svm_node *node = new svm_node[cone.size() + 1];
    for (size_t i = 0; i < cone.size(); ++i)
    {
        node[i].index = i + 1;
        node[i].value = cone[i];
    }
    node[cone.size()].index = -1;
    double value = svm_predict(model, node);
    delete[] node;
    return value;
}

/* take the flatten mesh and generate a vector of boundary points,
   using lazy evaluation
*/
conesList boundaryDetection(const std::vector<std::vector<double>> &xx, const std::vector<std::vector<double>> &yy,
    const svm_model *model) {
    size_t rows = xx.size();
    size_t cols = xx[0].size();
    std::set<std::pair<double, double>> boundary_points;
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            double x = xx[row][col];
            double y = yy[row][col];
            double val = nodePredictor({x, y}, model);
            size_t non_negative_row = row == 0 ? 0 : row - 1;
            size_t non_negative_col = col == 0 ? 0 : col - 1;
            // Ensure points are only added if they have a label of one
            if (abs(val - 1) < 0.1)
            {
                if (val != nodePredictor({xx[row][std::min(col + 1, cols - 1)], yy[row][std::min(col + 1, cols - 1)]}, model))
                    boundary_points.emplace(x, y);
                else if (val != nodePredictor({xx[row][non_negative_col], yy[row][non_negative_col]}, model))
                    boundary_points.emplace(x, y);
                else if (val != nodePredictor({xx[std::min(row + 1, rows - 1)][col], yy[std::min(row + 1, rows - 1)][col]}, model))
                    boundary_points.emplace(x, y);
                else if (val != nodePredictor({xx[non_negative_row][col], yy[non_negative_row][col]}, model))
                    boundary_points.emplace(x, y);
            }
        }
    }
    std::vector<std::pair<double, double>> boundary_points_out (boundary_points.begin(), boundary_points.end());
    return boundary_points_out;
}


}

}
}