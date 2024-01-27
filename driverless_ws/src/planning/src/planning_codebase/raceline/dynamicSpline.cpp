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

/** @brief calculate running average of bucket by incorprating curvature into
 * current avg
 * @param curavture curvature of current point, to be added to running avg
 * @param b bucket struct
 * @param spline new spline to be added into bucket's vector of splines
*/
double calcRunningAvgCurv(bucket b, double splineCurvature){
    b.numPointsInAvg += 4;
    return (b.runningAvgCurvature * (b.numPointsInAvg-4) + splineCurvature) / b.numPointsInAvg;
}

/** @brief checks if running average of current bucket is significantly different 
 * from next cones return true, otherwise return false
 * @param b: bucket struct
 * @param newCurvature: curvature of new spline 
 * @return true to start new bucket, false to update running avg
*/
bool checkStartNewBucket(bucket b, double newCurvature, double splineLength) {
    double sigDiff = 1; // TODO: update the sigDiff after unit testing
    if (abs(b.runningAvgCurvature - newCurvature) >= sigDiff && 
            (b.length + splineLength) < 15) {
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

/** @TODO: function to get length in meters from set oftcones
*/

/** @TODO: function to calculate curvature of a spline
*/
double calcSplineCurv (Spline spline) {}

/** @TODO: calculate how many times to split current bucket, based on avg curvature of bucket
    then add progress points to segment
*/
void addBucketToSegment(bucket b){}

std::vector<double> generateSegmentsForBucket(bucket b){

}

/** big wrapper function that calls each of the helpers, updates the segments field in optimizer's _ struct
* @param perceptions_data includes blue and yellow cones, only need blue cones 
* need start and end of the section to gen progress (?)
*/
void generateAllSegments(segment_t segment) {
    std::vector<std::pair<double,double>> cones = segment.perceptions_data.bluecones; // vector of outer cones
    int currConeIdx = 0; // index of the first cone that has not been added to a bucket yet

    // ignore first 4m, return 
    int coneIdx4m = getConesWithinDist(cones, currConeIdx, 4);
    //@TODO: add the progress at 4m into segment struct
    currConeIdx = coneIdx4m + 1; // updating currCondeIdx to be first cone outside of 4m

    // initialize a new bucket
    bucket* currBucket = new bucket;

    // generate 1 splines of 4 points as starting compare spline
    Eigen::MatrixXd first(2,4);
    for(int i = currConeIdx; i < 4; i++){
        first(0,i)=cones[i].first;
        first(1,i)=cones[i].second;
    }
    std::vector<Spline> prevSplines = generate_splines(logger, outer); // generates 1 spline
    Spline prevSpline = prevSplines[0];
    currConeIdx += 2;

    // initialize bucket fields
    currBucket->splines.push_back(prevSpline);
    currBucket->startCone = cones[currConeIdx-2];
    currBucket->endCone = cones[currConeIdx+2];
    currBucket->length = //sum of length splines - overlap???

     // calc curvature of spline
    double prevSplineCurv = calcSplineCurv(prevSplines);

    currBucket->runningAvgCurvature = calcRunningAvgCurv(currBucket, prevSplineCurv);
    
    while (currConeIdx < cones.size()) {
        // case on if can't generate another spline 
        // generate one spline that overlaps 2 points with previous spline, case on not enough points left
        Eigen::MatrixXd outer(2,4);
        for(int i = currConeIdx; i < 4; i++){
            outer(0,i)=cones[i].first;
            outer(1,i)=cones[i].second;
        }

        std::vector<Spline> currSplines = generate_splines(logger, outer); // generates 1 spline
        Spline currSpline = currSplines[0];
        currConeIdx += 2;

        // calc curvature of spline
        double currSplineCurv = calcSplineCurv(currSpline);

        // @TODO: compare to running avg curvature of the bucket UNIT TESTING TO FIND BEST THRESHOLD
        if (checkStartNewBucket(currBucket, currSplineCurv)) {
            // calculate how many times to split current bucket, based on avg curvature of bucket
            // then add progress points to segment (helper)
            addBucketToSegment(currBucket);
            // init new bucket
            currBucket = new bucket;

            // initialize bucket fields
            currBucket->splines.push_back(currSpline);
            currBucket->startCone = cones[currConeIdx-2];
            currBucket->endCone = cones[currConeIdx+2];
            currBucket->length = //sum of length splines - overlap???
            currBucket->runningAvgCurvature = calcRunningAvgCurv(currBucket, currSplineCurv);
            
        } else {
            // update bucket fields
            currBucket->splines.push_back(currSpline);
            currBucket->endCone = cones[currConeIdx+2];
            currBucket->length = //sum of length splines - overlap???
        }
        
    }
    
}