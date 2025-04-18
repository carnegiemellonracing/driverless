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
#include <rclcpp/rclcpp.hpp>
#define NUM_THREADS 4


namespace controls
{
namespace midline
{
namespace svm_fast_double_binsearch {

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

//Takes vector to search over and adds edges to the set of boundary points
// void binarySearch(const std::vector<double> &x, const std::vector<double> &y, 
//                   double left_label, double right_label, const svm_model *model,
//                   std::set<std::pair<double, double>> &boundary_points)
// {
//     return;
// }

static constexpr bool is_colored_one(double label) {
    return abs(label - 1) <= 0.1;
}

/* take the flatten mesh and generate a vector of boundary points,
   using lazy evaluation
*/
conesList boundaryDetection(const std::vector<std::vector<double>> &xx, const std::vector<std::vector<double>> &yy,
                            const svm_model *model)
{

    size_t rows = xx.size();
    size_t cols = xx[0].size();
    std::set<std::pair<double, double>> boundary_points;
    std::optional<size_t> chosen_column = std::nullopt;
    bool zero_to_right = true;
    size_t number_of_rows_not_skipped = 0;
    for (size_t row = 0; row < rows; ++row)
    {
        // We check up to -2 and +2
        // also holy garbage code right here
        if (chosen_column.has_value()) { // TODO: add additional bounds checking
            size_t chosen_column_value = chosen_column.value();
            if (chosen_column_value < cols - 2 && chosen_column_value >= 2) {
                if (zero_to_right) {
                    /*
                    1 1 0 0
                    1 1 0 0

                    1 1 0 0 
                    1 1 1 0

                    1 1 0 0
                    1 0 0 0
                    */
                    double center_label = nodePredictor({xx[row][chosen_column_value], yy[row][chosen_column_value]}, model);
                    if (is_colored_one(center_label)) {
                        if (!is_colored_one(nodePredictor({xx[row][chosen_column_value + 1], yy[row][chosen_column_value + 1]}, model))) {
                            chosen_column = chosen_column_value;
                            continue;
                        } else if (!is_colored_one(nodePredictor({xx[row][chosen_column_value + 2], yy[row][chosen_column_value + 2]}, model))) {
                            chosen_column = chosen_column_value + 1;
                            continue;
                        }
                    } else {
                        if (is_colored_one(nodePredictor({xx[row][chosen_column_value - 1], yy[row][chosen_column_value - 1]}, model))) {
                            chosen_column = chosen_column_value - 1;
                            continue;
                        }
                    }
                } else {
                    /*
                    0 0 1 1
                    0 0 1 1

                    0 0 1 1
                    0 1 1 1
                    */
                // ^ Notice how the +'s and -'s are all swapped from the previous one, we can refactor this later
                    if (is_colored_one(nodePredictor({xx[row][chosen_column_value], yy[row][chosen_column_value]}, model))) {
                        if (!is_colored_one(nodePredictor({xx[row][chosen_column_value - 1], yy[row][chosen_column_value - 1]}, model))) {
                            chosen_column = chosen_column_value;
                            continue;
                        } else if (!is_colored_one(nodePredictor({xx[row][chosen_column_value - 2], yy[row][chosen_column_value - 2]}, model))) {
                            chosen_column = chosen_column_value - 1;
                            continue;
                        }
                    } else {
                        if (is_colored_one(nodePredictor({xx[row][chosen_column_value + 1], yy[row][chosen_column_value + 1]}, model))) {
                            chosen_column = chosen_column_value + 1;
                            continue;
                        }
                    }
                }
            }
        }
        // predict left and right labels
        number_of_rows_not_skipped++;
        std::vector<double> left_node = {xx[row][0], yy[row][0]};
        std::vector<double> right_node = {xx[row][cols - 1], yy[row][cols - 1]};
        double left_label = nodePredictor(left_node, model);
        double right_label = nodePredictor(right_node, model);

        // If the row is homogeneous (all cells have the same label) then linear search
        if (left_label != right_label) {

            size_t left = 0;
            size_t right = cols - 1;

            while (right - left > 1)
            {
                size_t mid = left + (right - left) / 2;
                double left_point = nodePredictor({xx[row][mid], yy[row][mid]}, model);
                double right_point = nodePredictor({xx[row][mid + 1], yy[row][mid + 1]}, model);

                if (left_point != right_point)
                {
                    if (is_colored_one(left_point)) {
                        chosen_column = mid;
                        zero_to_right = true;
                    } else {
                        chosen_column = mid + 1;
                        zero_to_right = false;
                    }
                    break;

                    right = left;
                }
                else
                {
                    if (left_point == left_label)
                    {
                        left = mid;
                    }
                    else
                    {
                        right = mid;
                    }
                }
            }
            boundary_points.emplace(xx[row][chosen_column], yy[row][chosen_column]);

        } else {
            // Reset chosen_column so we don't propagate error
            chosen_column = std::nullopt;
        }
    }

    std::cout << "Skipped: " << rows - number_of_rows_not_skipped << "/" << rows << std::endl;

    std::optional<size_t> chosen_row = std::nullopt;
    bool zero_below = true;
    size_t number_of_cols_not_skipped = 0;

    for (size_t col = 0; col < cols; ++col)
    {
        // predict left and right labels
        std::vector<double> top_node = {xx[0][col], yy[0][col]};
        std::vector<double> bottom_node = {xx[rows-1][col], yy[rows-1][col]};
        double top_label = nodePredictor(top_node, model);
        double bottom_label = nodePredictor(bottom_node, model);

        // If the row is homogeneous (all cells have the same label) then linear search
        if (top_label != bottom_label) {

            size_t top = 0;
            size_t bottom = rows - 1;

            while (bottom - top > 1)
            {
                size_t mid = top + (bottom - top) / 2;
                double top_point = nodePredictor({xx[mid][col], yy[mid][col]}, model);
                double bottom_point = nodePredictor({xx[mid+1][col], yy[mid+1][col]}, model);

                // std::cout << left_point << " " << right_point << std::endl;
                if (top_point != bottom_point)
                {
                    boundary_points.emplace(xx[mid+1][col], yy[mid+1][col]);
                    if(abs(top_point - 1) > 0.1)
                        boundary_points.emplace(xx[mid+1][col], yy[mid+1][col]);
                    else
                        boundary_points.emplace(xx[mid][col], yy[mid][col]);
                    bottom = top;
                }

                else
                {
                    if (top_point == top_label)
                    {
                        top = mid;
                    }
                    else
                    {
                        bottom = mid;
                    }
                }
            }
        }
    }

    //TODO: turn set back into vector
    std::vector<std::pair<double, double>> boundary_points_out (boundary_points.begin(), boundary_points.end());
    return boundary_points_out;
}


}

}
}