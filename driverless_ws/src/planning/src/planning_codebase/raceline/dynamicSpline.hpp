#include "frenetEigen.hpp"
#include "racelineEigen.hpp"

public:

    struct bucket {
        std::vector<Spline> splines; // all of the splines in the bucket
        std::pair<double,double> startCone; // start cone index
        std::pair<double,double> endCone; // end cone index
        double length = 0; // length
        
    };

    /** look some distance ahead starting from start index and return end 
    * index of cones in that distance
    * @arg cones: vector (x, y) of blue cones
    * @arg startIndex: where in the cones vector we want to measure dist from
    * @return index of first cone that's outside of dist meters
    */
    int getConesWithinDist(std::vector<std::pair<double,double>> cones, int startIndex, int dist);

    /** running average of current bucket 
     * @arg spline: current spline we are looking at 
     * @arg point: curvature at point is added to the current average
    */
    double updateRunningAverageCurve(Spline spline, Eigen::VectorXd point);
    // spline_along to get single point (p) on spline (s)
    // get_curvature, taking

    // if running average of current bucket is significantly different from next cones
    bool checkStartNewBucket(double runningAvg, double newCurvature);

    // how many times to split a bucket based on curvature
    int numBucketSplits(double curvature, bucket b);

    // generates output: progress of start of section and end of section
    /** big wrapper function that calls each of the helpers
    * @arg perceptions data
    * @return tuple of floats (start and end of the section)
    */
   // if checkStartNewBucket true, then reset running avg to 0 and start new bucket
   // else, update running avg
   std::tuple<float, float> getSectionStartEnd(perceptionsData perceptions_data);


    



