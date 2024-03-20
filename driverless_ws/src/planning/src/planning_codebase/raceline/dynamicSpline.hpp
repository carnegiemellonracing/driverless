// #include "frenetEigen.hpp"
#include "raceline.hpp"
#include "../midline/generator.hpp"


// section of track progress that share similar curvature
struct bucket {
    double startProgress; // start of the bucket
    double endProgress; // end of the bucket
    double sumCurvature; // sum of curvatures at sample points

    // below are fields to be used by optimizer
    double avgCurvature; // running avg of bucket
    std::vector<std::pair<double, double>> bluePoints; // points on the blue cone spline
    std::vector<std::pair<double, double>> yellowPoints; // points on the yellow cone spline
};

// std::vector<bucket>* bucketsVector; // contains bucket structs, which contain the start and end points
                                    // of each segment and points in between

/** function to make vector of splines from std::vector<std::pair<double,double>> 
    (blue or yellow cones) of perception data
*/
std::pair<std::vector<Spline>,std::vector<double>> makeSplinesVector (std::vector<std::pair<double,double>> cones);

/** update running average of current bucket, update splines vector
 * @arg curvature: curvature is added to the current average
 * @arg bucket: use numPointsInAvg in struct
*/
double calcRunningAvgCurve(double curvature, bucket b);
// spline_along to get single point (p) on spline (s)
// get_curvature, taking

// if running average of current bucket is significantly different from next cones
//bool checkStartNewBucket(double runningAvg, double newCurvature);
bool checkStartNewBucket(bucket b, double newCurvature);

// finds the progress points to split given bucket 
void progressSplits(bucket b);

/** @brief updates a given vector with the progress sections to optimize over
* @note about the usage of progressVector: vector of progresses that indicate sections to
* optimize over, should optimize as car is going
* @param progressVector can be accessed by the optimizer file as it is being updated
*/
void updateSegments(std::vector<double>* progressVector);
    