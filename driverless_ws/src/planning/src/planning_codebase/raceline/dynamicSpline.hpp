#include "frenetEigen.hpp"
#include "racelineEigen.hpp"

public:

    // segment is initialized in optimizer
    struct segment {
        perceptionsData perceptions_data;
        std::vector<double> progress; // in meters, signifies end of each segment
    } segment_t;

    struct bucket {
        std::vector<Spline> splines; // all of the splines in the bucket
        std::pair<double,double> startCone; // start cone index
        std::pair<double,double> endCone; // end cone index
        double length = 0; // length in meters of the bucket (?) HAS TO BE < 15
        double runningAvgCurvature = 0; // running average of curvature in the bucket
        double numPointsInAvg = 0; // number of points used to calculate running avg
    };
    

    /** look some distance ahead starting from start index and return end 
    * index of cones in that distance
    * @arg cones: vector (x, y) of blue cones
    * @arg startIndex: where in the cones vector we want to measure dist from
    * @return index of first cone that's outside of dist meters
    */
    int getConesWithinDist(std::vector<std::pair<double,double>> cones, int startIndex, int dist);

    /** update running average of current bucket, update splines vector
     * @arg curvature: curvature is added to the current average
     * @arg bucket: use numPointsInAvg in struct
    */
    double updateRunningAvgCurve(double curvature, bucket b);
    // spline_along to get single point (p) on spline (s)
    // get_curvature, taking

    // if running average of current bucket is significantly different from next cones
    //bool checkStartNewBucket(double runningAvg, double newCurvature);
    bool checkStartNewBucket(bucket b, double newCurvature);

    // how many times to split a bucket based on curvature
    int numBucketSplits(double curvature, bucket b);
    
    // finds the progress points to split given bucket 
    std::vector<double> generateSegmentsForBucket(bucket b);

    /** big wrapper function that calls each of the helpers
    * @arg perceptions data
    */
   // if checkStartNewBucket true, then reset running avg to 0 and start new bucket
   // else, update running avg
   void generateAllSegments(segment_t segment);


    



