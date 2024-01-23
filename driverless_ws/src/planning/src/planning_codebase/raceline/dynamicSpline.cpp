/**
 * @TODO: Victoria writes updateRunningAvgCurve
 * @TODO: Doris write checkStartNewBucket
 * 
*/

#include <cmath>
#include "dynamicSpline.hpp";

/** look some distance ahead starting from start index and return end 
* index of cones in that distance
* @param cones: vector (x, y) of blue cones
* @param startIndex: where in the cones vector we want to measure dist from
* @param dist distance in meters
* @return index of first cone that's outside of dist meters
*/
int getConesWithinDist(std::vector<std::pair<double,double>> cones, int startIndex, int dist){
    int accumDist = 0; //cumulative distance when going through cones
    int i = startIndex; // index of cones
    std::vector<std::pair<double,double>> totalCones = {cones[i]}; // vector of all cones within dist m

    // while cones are within dist m, keep adding them to totalCones
    while (accumDist < dist && i < (cones.size() - 1)) {
        float dX = cones[i+1].x - cones[i].x;
        float dY = cones[i+1].y - cones[i].y;
        float distBetweenCones = sqrt(pow(dX, 2) + pow(dY, 2));
        accumDist += distBetweenCones;
        totalCones.push_back(cones[i+1]);
    }
    return totalCones;
}

/** @brief updates running average of bucket by incorprating curvature into
 * current avg
 * @param curavture curvature of current point, to be added to running avg
 * @param b bucket struct
 * @param spline new spline to be added into bucket's vector of splines
*/
double updateRunningAvgCurve(double curvature, bucket b, Spline spline){
    b.numPointsInAvg++;
    b.splines.push_back(spline);
    return (b.runningAvgCurvature * (b.numPointsInAvg-1) + curvature) / b.numPointsInAvg;
}

/** @brief checks if running average of current bucket is significantly different 
 * from next cones return true, otherwise return false
 * @param b: bucket struct
 * @param newCurvature: curvature of new spline 
 * @return true to start new bucket, false to update running avg
*/
bool checkStartNewBucket(bucket_t b, double newCurvature) {
    double sigDiff = 1; // TODO: update the sigDiff after unit testing
    if (abs(b.runningAvgCurvature - newCurvature) >= sigDiff) {
        return true;
    }
    return false;
}

/** @brief calculate number of times to split bucket based on avg curvature of bucket
 * @param b bucket to split
 * @param c constant, find through testing
 * @param a constant, find through testing
*/
int numBucketSplits(bucket b, int c, int a){
    return c * b.runningAvgCurvature.pow(a);
}

std::vector<double> generateSegmentsForBucket(bucket b){

}

/** big wrapper function that calls each of the helpers
* @param perceptions_data includes blue and yellow cones, only need blue cones 
* need start and end of the section to gen progress (?)
*/
void generateAllSegments(segment_t segment) {
    std::vector<std::pair<double,double>> cones = perceptions_data.bluecones; // vector of outer cones
    int currConeIdx = 0; // index of the first cone that has not been added to a bucket yet

    // ignore first 4m, return 
    int coneIdx4m = getConesWithinDist(cones, currConeIdx, 4);
    currConeIdx = coneIdx4m + 1; // updating currCondeIdx to be first cone outside of 4m

    //look at next
    
    

    
}