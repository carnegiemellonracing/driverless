#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <geometry_msgs/msg/point.hpp>
#include "hsv.hpp"
#include "yolo.hpp"

// Forward declarations
namespace cones

    /**
     * @brief LapCounter class uses a binary random variable with bayesian
     * update logic to detect then number of laps. It requires detection 
     * updates at each step and several input probabilities for updating.
     * 
     * @param init_prob sets the initial state of the random variable.
     * @param true_pos_rate is the rate of true positives of the sensors + 
     * algorithm that are detecting orange cones.
     * @param true_neg_rate is the rate of true negatives of the sensors +
     * algorithm that are detective orange cones.
     * @param decision_threshold is the cutoff for whether a cone is detected or not.
     */
    class LapCounter
    {
        private: 
            double cone_prob;
            bool cone_detected;
            double tp_rate;
            double tn_rate;
            double decision_thresh;
        
        public:
            int num_laps;

            // Update binary bayesian variable
            /**
             * @brief Bayesian update state for LapCounter random variable
             * 
             * @param detected is true if we have at least one orange cone 
             * within a certain (tunable) distance of the car\
             * @return boolean that is true if a new lap has been counted
             */
            bool update(bool detected);

            LapCounter(
                double init_prob = .1, 
                double true_pos_rate = .9, 
                double true_neg_rate = .9,
                double decision_threshold = .5
            )
            {
                num_laps = 0;
                cone_detected = false;
                cone_prob = init_prob;
                tp_rate = true_pos_rate;
                tn_rate = true_neg_rate;
                decision_thresh = decision_threshold;
            }
    
    };

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

    /**
     * @brief Augments data by adding cones in a circular pattern around the original cone, currently unused.
     * 
     * @param track_bounds TrackBounds struct
     * @param degrees Degrees to augment
     * @param radius Radius of the circle
     */
    void augment_cones_circle(TrackBounds &track_bounds, int degrees, double radius);
