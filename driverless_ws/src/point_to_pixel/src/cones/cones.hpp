#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <geometry_msgs/msg/point.hpp>

// Forward declarations
namespace point_to_pixel
{    // Cone types
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
     * @param max_distance_threshold Maximum distance between two cones before ordering clips detected cones
     * @return Cones Vector of ordered cones
     */

    Cones order_cones(const Cones& unordered_cones, double max_distance_threshold);

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
    
    /**
     * @brief Converts TrackBounds, a struct containing yellow cone vector and blue cone vector, to XY training data
     * 
     * @param track_bounds TrackBounds stuct
     * @return std::pair<std::vector<std::vector<double>>, std::vector<double>> Feature matrix and label vector
     */
    std::pair<std::vector<std::vector<double>>, std::vector<double>> cones_to_xy(const TrackBounds& track_bounds);

    /**
     * @brief Adds dummy cones to the side of the car, blue on left and yellow on right
     * 
     * @param track_bounds TrackBounds struct
     */
    void supplement_cones(TrackBounds &track_bounds);
} // namespace point_to_pixel