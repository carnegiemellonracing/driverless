/**
 * @TODO: Victoria writes updateRunningAvgCurve
 * @TODO: Doris write checkStartNewBucket
 * 
*/

#include "dynamicSpline.hpp";

/** @brief checks if running average of current bucket is significantly different 
 * from next cones return true, otherwise return false
 * @arg b: bucket struct
 * @arg newCurvature: curvature of new spline 
 * @return true to start new bucket, false to update running avg
*/
bool checkStartNewBucket(bucket_t b, double newCurvature) {
    double sigDiff = 1; // TODO: update the sigDiff after unit testing
    if (abs(b.runningAvgCurvature - newCurvature) >= sigDiff) {
        return true;
    }
    return false;
}


/** big wrapper function that calls each of the helpers
* @arg perceptions data 
* @return tuple of floats (start and end of the section)
*/