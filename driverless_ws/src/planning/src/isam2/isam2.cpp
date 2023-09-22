#include "isam2.hpp"

// Camera observations of landmarks will be stored as Point2 (x, y).
#include <gtsam/geometry/Point2.h>

// Each variable in the system (poses and landmarks) must be identified with a
// unique key. We can either use simple integer keys (1, 2, 3, ...) or symbols
// (X1, X2, L1). Here we will use Symbols
#include <gtsam/inference/Symbol.h>

// We want to use iSAM2 to solve the structure-from-motion problem
// incrementally, so include iSAM2 here
#include <gtsam/nonlinear/ISAM2.h>

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

#include <vector>

using namespace std;
using namespace gtsam;


static const float DT = 0.1;
static const float SIM_TIME = 50.0;
static const float MAX_RANGE = 10.0;
static const int M_DIST_TH = 1;
static const int LM_SIZE = 2;
static const int STATE_SIZE = 3;

static const int N_STEP = 100;

int main(int argc, char* argv[]){
    // Create an iSAM2 object. Unlike iSAM1, which performs periodic batch steps
    // to maintain proper linearization and efficient variable ordering, iSAM2
    // performs partial relinearization/reordering at each step. A parameter
    // structure is available that allows the user to set various properties, such
    // as the relinearization threshold and type of linear solver. For this
    // example, we we set the relinearization threshold small so the iSAM2 result
    // will approach the batch result.

    ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    ISAM2 isam(parameters);
    //Create a factor graph and values for new data
    NonlinearFactorGraph graph;
    Values initialEstimate;

    vector<vector<int>> xEst;
    vector<vector<int>> PEst;
    vector<vector<int>> xDR;

    //Define empty set
    set<int> observed;

    //for(x, (odom, obs) in enumerate(sim.step()): ) 
    for(int i = 0; i < N_STEP; i++){
        if(i == 0){
            vector<>
        }

    }





}