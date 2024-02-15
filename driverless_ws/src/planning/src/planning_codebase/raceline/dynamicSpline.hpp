#include "frenetEigen.hpp"
#include "racelineEigen.hpp"

public:
    std::vector<double>* progressVector; // in meters, signifies end points of each segment

    // section of track progress that share similar curvature
    struct bucket {
        double startProgress; // start of the bucket
        double endProgress; // end of the bucket
        double sumCurvature; // sum of curvatures at sample points
    };

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

    // how many times to split a bucket based on curvature
    int getNumBucketSplits(double curvature, bucket b);

    // finds the progress points to split given bucket 
    std::vector<double> progressSplits(bucket b);

    /** @brief updates a given vector with the progress sections to optimize over
    * @note about the usage of progressVector: vector of progresses that indicate sections to
    * optimize over, should optimize as car is going
    * @param progressVector can be accessed by the optimizer file as it is being updated
    */
    void updateSegments(std::vector<double>* progressVector);
    