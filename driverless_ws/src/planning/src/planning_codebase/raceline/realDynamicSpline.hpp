#include "frenetEigen.hpp"
#include "racelineEigen.hpp"

public:
    // used to transport info between optimizer and this file
    // include vector of progresses to optimize over for the optimizer to use
    struct segment {
        std::vector<double> progress; // in meters, signifies end of each segment
    };

    // section of track progress that share similar curvature
    struct bucket {
        double startProgress; // start of the bucket
        double endProgress; // end of the bucket
        double avgCurvature; // average curvature
    };

    

    