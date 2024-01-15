#include "../raceline/raceline.hpp"
// #include "random.h"
#include <math.h>
#include <algorithm>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#ifndef MIDPOINTGENERATOR
#define MIDPOINTGENERATOR

struct perceptionsData{

    std::vector<std::pair<double,double>> bluecones,yellowcones,orangecones;
};

class MidpointGenerator
{
private:
    /* data */



public:
    int PERCEP_COLOR = 2;
    int BLUE = 1;
    int YELLOW = 2;
    int ORANGE = 3;
    int interpolation_number;
    std::vector<Spline> cumulated_splines;
    std::vector<double> cumulated_lengths;


    MidpointGenerator(int interpolation_number=30);

    std::vector<std::pair<double,double>> sorted_by_norm(std::vector<std::pair<double,double>> inp);
    // gsl_matrix *sorted_by_norm(gsl_matrix *list);
    

    std::vector<Spline> generate_splines(rclcpp::Logger logger, Eigen::MatrixXd& midpoints);    
    Eigen::MatrixXd generate_points(rclcpp::Logger logger, perceptionsData perceptions_data);  
    Eigen::MatrixXd interpolate_cones(rclcpp::Logger logger, perceptionsData perceptions_data, int interpolation_number = -1);
    Spline spline_from_cones(rclcpp::Logger logger, perceptionsData perceptions_data);
    Spline spline_from_curve(rclcpp::Logger logger, std::vector<std::pair<double,double>> side);

};

Eigen::MatrixXd midpoint(rclcpp::Logger logger, Eigen::MatrixXd& inner,Eigen::MatrixXd& outer);

#endif


