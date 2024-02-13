#include <math.h>
#include <realDynamicSpline.hpp>
#include "generator.hpp"

/** @brief Calculate running average of bucket by incorprating curvature into
 * current avg
 * Run after the endProgress is updated (to avoid possibly diving by 0)
 * @param b bucket struct
*/
double calcRunningAvgCurv(bucket b){
    return b.sumCurvature / (b.endProgress - b.startProgress);
}

/** @brief Checks if running average of current bucket is significantly different 
 * from next cones or if bucket is too long
 * @param b bucket struct
 * @param newCurvature curvature of new point
 * @return true to start new bucket, false to update running avg
*/
bool checkStartNewBucket(bucket b, double newCurvature) {
    double sigDiff = 1; /** @TODO: update the sigDiff after unit testing*/
    double avgCurve = calcRunningAvgCurv(b);
    if (abs(avgCurve - newCurvature) >= sigDiff && 
            (b.endProgress - b.startProgress) < 15) {
        return true;
    }
    return false;
}

/** @brief Calculate number of times to split bucket based on avg curvature of bucket
 * @param b bucket to split
*/
int getNumBucketSplits(bucket b){
    int c = 1; /** @TODO: find ideal params during testing */
    int a = 1;
    return c * calcRunningAvgCurv(b).pow(a);
}

/** @brief Returns a vector of progresses split based on getNumBucketSplits for bucket b
 * @param b bucket struct
*/
std::vector<double> progressSplits(bucket b){
    std::vector<double> res;
    double diff = b.endProgress - b.startProgress;
    int numSplits = getNumBucketSplits(b);
    double interval = diff / numSplits;

    for (int i = b.startProgress; i <= b.endProgress; i += interval){
        res.pushback(i);
    }
}

/** @brief updates a given vector with the progress sections to optimize over
* @note about the usage of progressVector: vector of progresses that indicate sections to
* optimize over, should optimize as car is going
* @param progressVector can be accessed by the optimizer file as it is being updated
*/
void updateSegments(std::vector<double>* progressVector) {
    // make vector of splines
    // vector of outer cones
    std::vector<std::pair<double,double>> cones = segment.perceptions_data.bluecones;

    Eigen::MatrixXd coneMatrix(1, cones.size());
    for(int i = 0; i < cones.size(); i++){
        assert(i < coneMatrix.cols());
        coneMatrix(0, i) = cones[i].first;
        coneMatrix(1, i) = cones[i].second;
    }

    // std::vector<Spline> racetrackSplines = generate_splines(logger, coneMatrix);
    // use raceline_gen to get splines and cumulative length
    std::pair<std::vector<Spline>,std::vector<double>> res = raceline_gen(logger, coneMatrix, std::rand(), 4, false);
    std::vector<Spline> racetrackSplines = res.first;
    std::vector<double> cumulativeLen = res.second;

    // init bucket
    bucket* currBucket = new bucket;
    currBucket->startProgress = 0;
    currBucket->startProgress = 0;
    currBucket->sumCurvature = 0;
    
    // loop through progress and use get curvature on each progress
    int increment = 1; // FLEXIBLE: change this if running too slow
    double totalProgress = cumulativeLen[cumulativeLen.size()-1]
    
    for (int currProgress = 1; currProgress <= totalProgress, currProgress += increment) {
        double curve = get_curvature(currProgress, racetrackSplines, cumulativeLen)
        // compare curve to avgCurvature of the curr bucket
        currBucket.endProgress = currProgress;
        if (!checkStartNewBucket(currBucket, curve)) {
            // if fits in curr bucket, update avg curve and end progress of curr bucket
            currBucket.sumCurvature += curve;
        }
        else { // if doesn't fit, new bucket, start new average
            std::vector<double> bucketProgresses = progressSplits(currBucket);
            progressVector.insert(progressVector.end(), 
                                    bucketProgresses.begin(), bucketProgresses.end());
            currBucket->startProgress = currProgress;
            currBucket->endProgress = currProgress;
            currBucket->sumCurvature = curve;
        }
    }
}