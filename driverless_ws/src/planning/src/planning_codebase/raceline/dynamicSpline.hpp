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

    /** big wrapper function that calls each of the helpers
    * @param segment segment struct
    */
    // if checkStartNewBucket true, then reset running avg to 0 and start new bucket
    // else, update running avg
    void updateSegments(segment_t segment);
    