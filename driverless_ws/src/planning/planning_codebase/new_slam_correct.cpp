#include <iostream>
#include <Eigen/Dense>
#include <cmath>

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
Eigen::MatrixXd Q_sim(2, 2);
Q_sim << 0.2 * 0.2, 0.0, 
         0.0, (Eigen::deg2rad(1.0) * Eigen::deg2rad(1.0));

// Create R_sim matrix
Eigen::MatrixXd R_sim(2,2);
R_sim << 1.0 * 1.0, 0.0, 
         0.0, (Eigen::deg2rad(10.0) * Eigen::deg2rad(10.0));

// Create Cx matrix
Eigen::MatrixXd Cx(3, 3);
Cx << 0.5 * 0.5, 0.0, 0.0, 
      0.0, 0.5 * 0.5, 0.0, 
      0.0, 0.0, (Eigen::deg2rad(30.0) * Eigen::deg2rad(30.0));

// Create alphas
Eigen::MatrixXd alphas(6, 1);
alphas << 0.11, 0.01, 0.18, 0.08, 0.0, 0.0;


Eigen::MatrixXd calcInput() {
    // Define v and yaw_rate
    double v = 1.0;         // [m/s]
    double yaw_rate = 0.1;  // [rad/s]

    // Create a 2x1 column vector u
    Eigen::MatrixXd u(2, 1);
    u << v, 
         yaw_rate;

    return u;
}

// Function to calculate the motion model
Eigen::MatrixXd motion_model(const Eigen::MatrixXd& x, const Eigen::MatrixXd& u, double dt) {
    Eigen::MatrixXd F(3, 3);
    F << 1.0, 0.0, 0.0,
         0.0, 1.0, 0.0,
         0.0, 0.0, 1.0;

    Eigen::MatrixXd B(3, 2);
    B << dt * std::cos(x(2, 0)), 0.0,
         dt * std::sin(x(2, 0)), 0.0,
         0.0, dt;

    return (F * x) + (B * u);
}

// Function to calculate the number of landmarks
int calc_n_lm(const Eigen::MatrixXd& x) {
    int n = static_cast<int>((x.rows() - STATE_SIZE) / LM_SIZE);
    return n;
}

// Storing G and Fx matrices in the jacob_motion package
struct jacob_motion_package {
    Eigen::MatrixXd Fx;
    Eigen::MatrixXd G;
};

// Function to calculate jacobian motion
jacob_motion_package jacob_motion(const Eigen::MatrixXd* x, const Eigen::MatrixXd& u, double dt) {


    // Creating Identity Matrix of Size 3 x 3
    Eigen::MatrixXd identityMatrix = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE);


    // Creating Zeroes Matrix
    int n_lm = calc_n_lm(x);
    Eigen::MatrixXd zerosMatrix(STATE_SIZE, LM_SIZE * n_lm);
    zerosMatrix.setZero();

    // Stacking Identity and Zeroes Matrix
    Eigen::MatrixXd Fx(STATE_SIZE, STATE_SIZE + (LM_SIZE * n_lm));
    Eigen::MatrixXd Fx << identityMatrix, zeroesMatrix;

    Eigen::MatrixXd jF;

    // Creating jF matrix
    jF << 0.0, 0.0, -dt * u(0, 0) * std::sin(x(2, 0)),
          0.0, 0.0, dt * u(0, 0) * std::cos(x(2, 0)),
          0.0, 0.0, 0.0;

    // Creating Fx transpose matrix
    Eigen::MatrixXd FxT = Fx.transpose();

    // Creating G matrix 
    Eigen::MatrixXd G = identityMatrix + (FxT * jF * Fx);

    // Constructing jacob motion package result
    jacob_motion_package result;
    result.Fx = Fx
    result.G = G

    return result
}


// Function to return (x, y) position of landmark stored as 2 x 1 matrix
Eigen::MatrixXd calc_landmark_position(const Eigen::MatrixXd& x, const Eigen::MatrixXd& z) {

    Eigen::MatrixXd zp(2, 1);
    zp.setZero();

    zp(0,0) = x(0, 0) + (z(0, 0) * std::cos(x(2, 0) + z(1, 0)));
    zp(1,0) = x(1, 0) + (z(0, 0) * std::sin(x(2, 0) + z(1, 0)));

    return zp;
}

Eigen::MatrixXd get_landmark_position_from_state(const Eigen::MatrixXd& x, int ind) {
    
    // Calculate the starting and ending row indices for the landmark position
    int start_row = STATE_SIZE + LM_SIZE * ind;
    int end_row = STATE_SIZE + LM_SIZE * (ind + 1);

    // Extract the landmark position from the state vector
    Eigen::MatrixXd lm = x.block(start_row, 0, LM_SIZE, 1);

    return lm;
}

// Function Calculates Jacobian Matrix H
Eigen::MatrixXd jacob_h(double q, const Eigen::MatrixXd& delta, const Eigen::MatrixXd& x, int i) {
    double sq = std::sqrt(q);

    Eigen::MatrixXd G(2, 5);
    G << -sq * delta(0, 0), -sq * delta(1, 0), 0.0, sq * delta(0, 0), sq * delta(1, 0),
         delta(1, 0), -1 * delta(0,0), -q, -1 * delta(1, 0), delta(0, 0);
    g /= q;

    // Calculate the number of landmarks
    int nLM = calc_n_lm(x);

    // Construct the F1 matrix
    Eigen::MatrixXd F1(3, 3 + 2 * nLM);
    F1 << Eigen::MatrixXd::Identity(3, 3), Eigen::MatrixXd::Zero(3, 2 * nLM);

    // Construct the F2 matrix
    Eigen::MatrixXd F2(2, 3 + 2 * nLM);
    F2 << Eigen::MatrixXd::Zero(2, 3), Eigen::MatrixXd::Zero(2, 2 * (i - 1)), Eigen::MatrixXd::Identity(2, 2), Eigen::MatrixXd::Zero(2, 2 * nLM - 2 * i);

    // Concatenate F1 and F2 to create F matrix
    Eigen::MatrixXd F(F1.rows() + F2.rows(), F1.cols());
    F << F1, F2;

    // Calculate the Jacobian H by multiplying G and F
    Eigen::MatrixXd H = G * F;

    return H;

}

double pi_2_pi(double angle) {
    return fmod(angle + M_PI, 2.0 * M_PI) - M_PI;
}

struct innovation_package {
    Eigen::MatrixXd y;
    Eigen::MatrixXd S;
    Eigen::MatrixXd H;
}

innovation_package calc_innovation(const Eigen::MatrixXd& lm, const Eigen::MatrixXd& xEst, 
                                   const Eigen::MatrixXd& PEst, const Eigen::MatrixXd& z, int LMid) {
    
    Eigen::MatrixXd delta = lm - xEst.topRows(2);

    // Calculate q
    double q = delta.squaredNorm();

    // Calculate z_angle
    double z_angle = std::atan2(delta(1, 0), delta(0, 0)) - xEst(2, 0);

    // Calculate zp
    Eigen::MatrixXd zp(1, 2);
    zp(0, 0) = std::sqrt(q);
    zp(0, 1) = pi_2_pi(z_angle);

    // Calculate y
    Eigen::MatrixXd y = (z - zp).transpose();
    y(1, 0) = pi_2_pi(y(1, 0));

    // Calculate H and S
    Eigen::MatrixXd H = jacob_h(q, delta, xEst, LMid + 1)
    Eigen::MatrixXd S = H * PEst * H.transpose() + Cx.block(0, 0, 2, 2);

    // Constructing Innovation Package Result
    innovation_package result;
    result.y = y
    result.S = S
    result.H = H

    return result

}








