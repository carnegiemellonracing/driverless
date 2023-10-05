#include <iostream>
#include <Eigen/Dense>

// # DT = 0.1  # time tick [s]
// simulation time [s]
double SIM_TIME = 50.0

// maximum observation range
double MAX_RANGE = 20.0  
double M_DIFF_TH = 1.6
double M_DIST_TH = 2

// M_DIST_TH_FIRST = 0.25  # Threshold of Mahalanobis distance for data association.
double M_DIST_TH_ALL = 1
// State size [x,y,yaw]
double STATE_SIZE = 3  
// LM state size [x,y]
double LM_SIZE = 2

// Create Q_sim matrix
Eigen::Matrix2d Q_sim;
Q_sim << 0.2 * 0.2, 0.0, 0.0, (Eigen::deg2rad(1.0) * Eigen::deg2rad(1.0));

// Create R_sim matrix
Eigen::Matrix2d R_sim;
R_sim << 1.0 * 1.0, 0.0, 0.0, (Eigen::deg2rad(10.0) * Eigen::deg2rad(10.0));

// Create Cx matrix
Eigen::Matrix3d Cx;
Cx << 0.5 * 0.5, 0.0, 0.0, 0.0, 0.5 * 0.5, 0.0, 0.0, (Eigen::deg2rad(30.0) * Eigen::deg2rad(30.0))

// Create alphas
Eigen::VectorXd alphas(6);
alphas << 0.11, 0.01, 0.18, 0.08, 0.0, 0.0;


Eigen::Vector2d calcInput() {
    // Define v and yaw_rate
    double v = 1.0;         // [m/s]
    double yaw_rate = 0.1;  // [rad/s]

    // Create a 2x1 column vector u
    Eigen::Vector2d u;
    u << v, yaw_rate;

    return u;
}

// Function to calculate the motion model
Eigen::VectorXd motion_model(const Eigen::VectorXd& x, const Eigen::Vector2d& u, double dt) {
    Eigen::Matrix3d F;
    F << 1.0, 0.0, 0.0,
         0.0, 1.0, 0.0,
         0.0, 0.0, 1.0;

    Eigen::Matrix<double, 3, 2> B;
    B << dt * std::cos(x(2, 0)), 0.0,
         dt * std::sin(x(2, 0)), 0.0,
         0.0, dt;

    return (F * x) + (B * u);
}

// Function to calculate the number of landmarks
int calc_n_lm(const Eigen::VectorXd& x) {
    int n = static_cast<int>((x.size() - STATE_SIZE) / LM_SIZE);
    return n;
}

// Storing G and Fx matrices in the jacob_motion package
struct jacob_motion_package {

};



