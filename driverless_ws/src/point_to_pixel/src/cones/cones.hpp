#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <geometry_msgs/msg/point.hpp>
#include "hsv.hpp"
#include "yolo.hpp"

// Forward declarations
namespace cones
{
    struct Cone
    {
        geometry_msgs::msg::Point point;
        double distance;
        Cone(const geometry_msgs::msg::Point &p) : point(p)
        {
            distance = std::sqrt(p.x * p.x + p.y * p.y);
        }
    };
    typedef std::vector<Cone> Cones;

    struct TrackBounds
    {
        Cones yellow;
        Cones blue;
    };

    /**
     * @brief Orders cones by their path direction
     * 
     * @param unordered_cones Vector of unordered cones
     * @return Cones Vector of ordered cones
     */

    Cones order_cones(const Cones& unordered_cones);

    /**
     * @brief Finds the next closest cone to the first cone in the vector
     * 
     * @param cones Vector of cones
     * @return Cone Closest cone
     */
    Cone find_closest_cone(const Cones& cones);

    /**
     * @brief Calculates the angle between two cones
     * 
     * @param from First cone
     * @param to Second cone
     * @return double Angle in radians
     */
    double calculate_angle(const Cone& from, const Cone& to);

    std::vector<double> cone_to_features(const Cone& cone);

    std::pair<std::vector<std::vector<double>>, std::vector<double>> cones_to_xy(const TrackBounds& track_bounds);

    void supplement_cones(TrackBounds &track_bounds);

    void augment_cones_circle(TrackBounds &track_bounds, int degrees, double radius);
}