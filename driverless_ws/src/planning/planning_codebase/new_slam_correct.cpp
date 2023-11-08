#include <iostream>
// #include <Eigen/Dense>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <vector>

// # DT = 0.1  # time tick [s]
// simulation time [s]
double SIM_TIME = 50.0;

// maximum observation range
double MAX_RANGE = 20.0;  
double M_DIFF_TH = 1.6;
double M_DIST_TH = 1.2;

// M_DIST_TH_FIRST = 0.25  # Threshold of Mahalanobis distance for data association.
double M_DIST_TH_ALL = 1;
// State size [x,y,yaw]
int STATE_SIZE = 3;  
// LM state size [x,y]
int LM_SIZE = 2;


// Create Cx matrix
Eigen::Matrix3d Cx = (Eigen::Matrix3d() << 0.5 * 0.5, 0.0, 0.0, 
        0.0, 0.5 * 0.5, 0.0, 
        0.0, 0.0, (std::cos(30.0) * std::cos(30.0))).finished(); // Convert 30.0 degrees to radians

// Create R_sim matrix
Eigen::Matrix2d R_sim = (Eigen::Matrix2d() << 1.0 * 1.0, 0.0, 
        0.0, (std::cos(10.0) * std::cos(10.0))).finished(); // Convert 10.0 degrees to radians

Eigen::MatrixXd calcInput() {
    // Define v and yaw_rate
    double v = 0.0;         // [m/s]
    double yaw_rate = 0.0;  // [rad/s]

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
jacob_motion_package jacob_motion(auto logger, const Eigen::MatrixXd& x, const Eigen::MatrixXd& u, double dt) {


    // Creating Identity Matrix of Size 3 x 3
    Eigen::MatrixXd identityMatrix = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE);


    // Creating Zeroes Matrix
    int n_lm = calc_n_lm(x);
    RCLCPP_INFO(logger, "NUM_LANDMARKS: %i", n_lm);
    Eigen::MatrixXd zeroesMatrix(STATE_SIZE, LM_SIZE * n_lm);
    zeroesMatrix.setZero();

    // Stacking Identity and Zeroes Matrix
    Eigen::MatrixXd Fx(STATE_SIZE, STATE_SIZE + (LM_SIZE * n_lm));
    Fx << identityMatrix, zeroesMatrix;

    Eigen::MatrixXd jF(3, 3);

    // Creating jF matrix
    jF << 0.0, 0.0, -dt * u(0, 0) * std::sin(x(2, 0)),
          0.0, 0.0, dt * u(0, 0) * std::cos(x(2, 0)),
          0.0, 0.0, 0.0;

    // Creating Fx transpose matrix
    Eigen::MatrixXd FxT = Fx.transpose().eval();

    // Creating G matrix 
    Eigen::MatrixXd G = identityMatrix + (FxT * jF * Fx);

    // Constructing jacob motion package result
    jacob_motion_package result;
    result.Fx = Fx;
    result.G = G;

    return result;
}


// Function to return (x, y) position of landmark stored as 2 x 1 matrix
Eigen::MatrixXd calc_landmark_position(auto logger, const Eigen::MatrixXd& x, const Eigen::MatrixXd& z) {

    Eigen::MatrixXd zp(1, 2);
    zp.setZero();

    zp(0,0) = x(0, 0) + (z(0, 0) * std::cos(x(2, 0) + z(0, 1)));
    zp(1,0) = x(1, 0) + (z(0, 0) * std::sin(x(2, 0) + z(0, 1)));

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
Eigen::MatrixXd jacob_h(auto logger, double q, const Eigen::MatrixXd& delta, const Eigen::MatrixXd& x, int i) {
    double sq = std::sqrt(q);

    Eigen::MatrixXd G(2, 5);
    G << -sq * delta(0, 0), -sq * delta(1, 0), 0.0, sq * delta(0, 0), sq * delta(1, 0),
         delta(1, 0), -1 * delta(0,0), -q, -1 * delta(1, 0), delta(0, 0);
    G /= q;

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
};

struct innovation_package calc_innovation(auto logger, const Eigen::MatrixXd& lm, const Eigen::MatrixXd& xEst, const Eigen::MatrixXd& PEst, const Eigen::MatrixXd& z, int LMid) {
    // Taking world frame back to car frame to compare to measurement
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
    // Eigen::MatrixXd y = (z - zp).transpose().eval();
    Eigen::MatrixXd y = (z - zp);
    y(1, 0) = pi_2_pi(y(1, 0));
    
    // Calculate H and S
    Eigen::MatrixXd H = jacob_h(logger, q, delta, xEst, LMid + 1);
    Eigen::MatrixXd S = H * PEst * H.transpose().eval() + Cx.block(0, 0, 2, 2);

    // Constructing Innovation Package Result
    innovation_package result;
    result.y = y;
    result.S = S;
    result.H = H;

    return result;

}

void print_matrix(auto logger, const Eigen::MatrixXd& xAug){
    for(int i = 0; i < xAug.rows(); i++){
        RCLCPP_INFO(logger, "%f", xAug(i, 0));
    }
}
int search_correspond_landmark_id(auto logger, const Eigen::MatrixXd& xAug, const Eigen::MatrixXd& PAug, const Eigen::MatrixXd& zi) {

    // Calculate the number of landmarks
    int nLM = calc_n_lm(xAug);

    // Vector that will store mahalanobis distances
    std::vector<double> min_dist;
    double r = zi(0, 0);
    double theta = zi(1, 0);

    double meas_x = r*std::cos(theta);
    double meas_y = r*std::sin(theta);

    RCLCPP_INFO(logger, "Measurement: (%f, %f)", meas_x, meas_y);
    // print_matrix(logger, xAug);
    for (int i = 0; i < nLM; ++i) {
        Eigen::MatrixXd lm = get_landmark_position_from_state(xAug, i);
        // Calculating y, S, and H matrices from innovation package
        struct innovation_package i_p = calc_innovation(logger, lm, xAug, PAug, zi, i);
        Eigen::MatrixXd y = i_p.y;
        Eigen::MatrixXd S = i_p.S;
        Eigen::MatrixXd H = i_p.H;

        // Calculating mahalanobis distance
        double mahalanobis = (y.transpose().eval() * S.inverse() * y)(0, 0);

        Eigen::MatrixXd lm_car_frame = lm - xAug.topRows(2);
        double euclidean = std::sqrt(pow(lm_car_frame(0, 0)-meas_x, 2) + pow(lm_car_frame(1, 0)-meas_y, 2));
        RCLCPP_INFO(logger, "   Landmark %i (Car Frame): (%f, %f) | Mahalanobis: %f | Euclidean: %f", i, lm_car_frame(0,0), lm_car_frame(1,0), mahalanobis, euclidean);

        // Adding mahalanobis distance to minimum distance vector
        min_dist.push_back(euclidean);
    }
    min_dist.push_back(M_DIST_TH); // Add M_DIST_TH for new landmark

    // Find the index of the minimum element in 'min_dist'
    int min_id = std::distance(min_dist.begin(), std::min_element(min_dist.begin(), min_dist.end()));

    return min_id;

}

struct ekfPackage {
    Eigen::MatrixXd x;
    Eigen::MatrixXd p;
    std::vector<Eigen::MatrixXd> cone;
};

struct ekfPackage ekf_slam(auto logger, Eigen::MatrixXd& xEst, Eigen::MatrixXd& PEst, Eigen::MatrixXd& u, Eigen::MatrixXd& z, double dt) {
    // Ensuring that z is a 2 x n matrix where every landmark is 2 x 1 matrix
    z = z.transpose().eval();
    std::vector<Eigen::MatrixXd> cones;
    int S = STATE_SIZE;
    struct jacob_motion_package j_m_p = jacob_motion(logger, xEst.topRows(S), u, dt);
    Eigen::MatrixXd G = j_m_p.G;
    Eigen::MatrixXd Fx = j_m_p.Fx;

    Eigen::MatrixXd M_t(2, 2);

    // Calculate the elements of M_t
    double element_1 = std::pow(0.11 * std::abs(u(0, 0)) + 0.01 * std::abs(u(1, 0)), 2);
    double element_2 = std::pow(0.18 * std::abs(u(0, 0)) + 0.08 * std::abs(u(1, 0)), 2);

    // Assign the elements to M_t
    M_t << element_1, 0,
           0, element_2;
    Eigen::MatrixXd x = xEst.topRows(S);

    Eigen::MatrixXd V_t(3, 2);

    double cos_x2 = std::cos(x(2, 0));
    double sin_x2 = std::sin(x(2, 0));

    // Calculate the elements of V_t
    V_t << cos_x2, -0.5 * sin_x2,
           sin_x2, 0.5 * cos_x2,
           0, 1;
    xEst.topRows(S) = motion_model(xEst.topRows(S), u, dt);
    PEst.block(0, 0, S, S) = G.transpose().eval() * PEst.block(0, 0, S, S) * G + Fx.transpose().eval() * Cx * Fx;
    Eigen::MatrixXd initP = Eigen::MatrixXd::Identity(2, 2);

    // Initializing landmark position
    Eigen::MatrixXd lm;
    
    for (int iz = 0; iz < z.rows(); ++iz) {
        int min_id = search_correspond_landmark_id(logger, xEst, PEst, z.row(iz));
        int nLM = calc_n_lm(xEst);
        if (min_id == nLM) {
            // Extend state and covariance matrix
            Eigen::MatrixXd xAug(xEst.rows() + LM_SIZE, xEst.cols());
            Eigen::MatrixXd zi = z.row(iz);
            Eigen::MatrixXd zi_in_world = calc_landmark_position(logger, xEst, zi);
            RCLCPP_INFO(logger, "ADDING: (%f, %f)", zi_in_world(0, 0), zi_in_world(0, 1));

            xAug << xEst, calc_landmark_position(logger, xEst, z.row(iz)).transpose().eval();

            Eigen::MatrixXd m1(PEst.rows(), PEst.cols() + LM_SIZE);
            Eigen::MatrixXd m1_zerosMatrix(xEst.rows(), LM_SIZE);
            m1_zerosMatrix.setZero();

            m1 << PEst, m1_zerosMatrix;

            Eigen::MatrixXd m2(LM_SIZE, xEst.rows() + initP.rows());
            Eigen::MatrixXd m2_zerosMatrix(LM_SIZE, xEst.rows());
            m2_zerosMatrix.setZero();

            m2 << m2_zerosMatrix, initP;

            Eigen::MatrixXd PAug(m1.rows() + m2.rows(), m1.cols());
            PAug << m1, m2;

            xEst = xAug;
            PEst = PAug;
        }

        lm = get_landmark_position_from_state(xEst, min_id);
        innovation_package i_p = calc_innovation(logger, lm, xEst, PEst, z.row(iz), min_id);
        Eigen::MatrixXd y = i_p.y;
        Eigen::MatrixXd S = i_p.S;
        Eigen::MatrixXd H = i_p.H;

        Eigen::MatrixXd K = PEst * H.transpose() * S.inverse();
        xEst.block(3, 0, xEst.rows() - 3, xEst.cols()) = xEst.block(3, 0, xEst.rows() - 3, xEst.cols()) + (K.block(3, 0, K.rows() - 3, K.cols()) * y);
        PEst = (Eigen::MatrixXd::Identity(PEst.rows(), PEst.cols()) - K * H) * PEst;
    }


    // xEst.row(2) = pi_2_pi(xEst.row(2));
    xEst(2, 0) = pi_2_pi(xEst(2, 0));

    // Constructing EKF SLAM Package Result
    ekfPackage result;
    result.x = xEst;
    result.p = PEst;
    result.cone = cones;

    return result;
}






