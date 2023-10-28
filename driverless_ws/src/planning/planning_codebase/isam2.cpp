// #include "isam2.hpp"
#include <type_traits>

// Camera observations of landmarks will be stored as Point2 (x, y).
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Pose2.h>
#include <Eigen/Dense>
// Each variable in the system (sposes and landmarks) must be identified with a
// unique key. We can either use simple integer keys (1, 2, 3, ...) or symbols
// (X1, X2, L1). Here we will use Symbols
#include <gtsam/inference/Symbol.h>

// We want to use iSAM2 to solve the structure-from-motion problem
// incrementally, so include iSAM2 herel
#include <gtsam/nonlinear/ISAM2.h>

#include <gtsam/nonlinear/NonlinearISAM.h>


// iSAM2 requires as input a set of new factors to be added stored in a factor
// graph, and initial guesses for any new variables used in the added factors
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

// In GTSAM, measurement functions are represented as 'factors'. Several common
// factors have been provided with the library for solving robotics/SLAM/Bundle
// Adjustment problems. Here we will use Projection factors to model the
// camera's landmark observations. Also, we will initialize the robot at some
// location using a Prior factor.
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/BearingRangeFactor.h>

#include <vector>

// using namespace std;
using namespace gtsam;


static const float DT = 0.1;
static const float SIM_TIME = 50.0;
static const float MAX_RANGE = 10.0;
static const int M_DIST_TH = 1;
static const int LM_SIZE = 2;
static const int STATE_SIZE = 3;

static const int N_STEP = 100;

struct enumLm {
    int lm_id;
    gtsam::Point2 lm_pos;
};

// auto cmp [](enumLm lm1, enumLm lm2) {
//     return (lm1.lm_pos.x() > lm2.lm_pos.x() && lm1.lm_pos.y() > lm2.lm_pos.y());
// };

class Compare {
public:
    bool operator()(enumLm lm1, enumLm lm2) const {
        return (lm1.lm_pos.x() > lm2.lm_pos.x() && lm1.lm_pos.y() > lm2.lm_pos.y());
    }
};


class slamISAM {
private:
    // ISAM2Params parameters;
    // parameters.relinearizeThreshold = 0.01;
    // parameters.relinearizeSkip = 1;
    // ISAM2 isam2;
    NonlinearISAM isam;
    //Create a factor graph and values for new data
    NonlinearFactorGraph graph;
    Values values;

    int x;

    // std::vector<std::vector<int>> PEst;
    // std::vector<std::vector<int>> xDR;
    // std::vector<std::vector<int>> xEst;

    //Define empty set
    // using Cmp = std::integral_constant<decltype(&cmp), &cmp>;
    std::set<enumLm, const Compare> observed;

    gtsam::Symbol X(int robot_pose_id) {
        return Symbol('x', robot_pose_id);
    }

    gtsam::Symbol L(int cone_pose_id) {
        return Symbol('x', cone_pose_id);
    }

public:

    int n_landmarks;

    gtsam::Pose2 robot_est;
    std::vector<gtsam::Point2> landmark_est;


    slamISAM() {

        // isam2 = gtsam::ISAM2();
        isam = gtsam::NonlinearISAM();
        graph = gtsam::NonlinearFactorGraph();
        values = gtsam::Values();
        x = 0;
        n_landmarks = 0;
        robot_est = gtsam::Pose2(0, 0, 0);
        landmark_est = std::vector<gtsam::Point2>();

    }

    void step(gtsam::Pose2 global_odom, std::vector<Point2> &cone_obs) {

        Pose2 prev_robot_est;

        if (x==0) {
            noiseModel::Diagonal::shared_ptr prior_model = noiseModel::Diagonal::Sigmas(Eigen::Vector3d(0, 0, 0));
            gtsam::PriorFactor<Pose2> prior_factor = gtsam::PriorFactor<Pose2>(X(0), global_odom, prior_model);
            graph.add(prior_factor);
            values.insert(X(0), global_odom);
            prev_robot_est = Pose2(0, 0, 0);
        }
        else {
            noiseModel::Diagonal::shared_ptr odom_model = noiseModel::Diagonal::Sigmas(Eigen::Vector3d(0, 0, 0));
            Pose2 prev_pos = isam.estimate().at(X(x-1)).cast<Pose2>();
            gtsam::BetweenFactor<Pose2> odom_factor = gtsam::BetweenFactor<Pose2>(X(x - 1), X(x), Pose2(global_odom.x() - prev_pos.x(), global_odom.y() - prev_pos.y(), global_odom.theta() - prev_pos.theta()), odom_model);
            graph.add(odom_factor);
            values.insert(X(x), global_odom);
            prev_robot_est = prev_pos;
        }

        isam.update(graph, values);
        graph.resize(0);
        values.clear();
        Pose2 robot_est = isam.estimate().at(X(x)).cast<Pose2>();

        // DATA ASSOCIATION BEGIN
        for (Point2 cone : cone_obs) {
            Point2 global_cone(global_odom.x() + cone.x(), global_odom.y() + cone.y());
            const enumLm enum_cone{lm_id: n_landmarks, lm_pos: global_cone};
            if (observed.find(enum_cone) == observed.end()) {
                observed.insert(enum_cone);
                
                double range = std::sqrt(cone.x() * cone.x() + cone.y() * cone.y());
                double bearing = std::atan2(cone.y(), cone.x()) - global_odom.theta();
                graph.add(BearingRangeFactor<Pose2, Pose2, double, double>(X(x), L(n_landmarks), bearing, range, noiseModel::Diagonal::Sigmas(Eigen::Vector3d(0, 0, 0))));
                values.insert(L(n_landmarks), global_cone);
                n_landmarks++;
            } else {
                int associated_id = (*(observed.find(enum_cone))).lm_id;
                double range = std::sqrt(cone.x() * cone.x() + cone.y() * cone.y());
                double bearing = std::atan2(cone.y(), cone.x()) - global_odom.theta();
                graph.add(BearingRangeFactor<Pose2, Pose2, double, double>(X(x), L(associated_id), bearing, range, noiseModel::Diagonal::Sigmas(Eigen::Vector3d(0, 0, 0))));
            }
        }
        // DATA ASSOCIATION END

        isam.update(graph, values);
        graph.resize(0);
        values.clear();

        robot_est =  isam.estimate().at(X(x)).cast<Pose2>(); 
        x++;

        landmark_est.clear();
        for (int i = 0; i < n_landmarks; i++) {
            landmark_est.push_back(isam.estimate().at(L(i)).cast<gtsam::Point2>());
        }

    }

};


// int main(int argc, char* argv[]){
//     // Create an iSAM2 object. Unlike iSAM1, which performs periodic batch steps
//     // to maintain proper linearization and efficient variable ordering, iSAM2
//     // performs partial relinearization/reordering at each step. A parameter
//     // structure is available that allows the user to set various properties, such
//     // as the relinearization threshold and type of linear solver. For this
//     // example, we we set the relinearization threshold small so the iSAM2 result
//     // will approach the batch result.

//     ISAM2Params parameters;
//     parameters.RelinearizationThreshold = 0.01;
//     parameters.relinearizeSkip = 1;

//     // vector<vector<int>> xEst;
//     // vector<vector<int>> PEst;
//     // vector<vector<int>> xDR;

//     //Define empty set
//     // std::set<int> observed;

//     //for(x, (odom, obs) in enumerate(sim.step()): ) 
//     for(int i = 0; i < N_STEP; i++){
//         if(i == 0){
//         }

//     }





// }