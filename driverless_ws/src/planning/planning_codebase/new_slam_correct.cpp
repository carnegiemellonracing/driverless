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








