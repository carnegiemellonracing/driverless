#include <math.h>
#include "dynamicSpline.hpp"

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

/** @brief Update the blue and yellow point vectors in the bucket struct for the current bucket
 * @param b bucket struct
*/
void progressSplits(bucket* b, 
        std::pair<std::vector<Spline>,std::vector<double>> blueRaceline,
        std::pair<std::vector<Spline>,std::vector<double>> yellowRaceline){

    double interval = 0.5; // @TODO tunable param
    std::vector<double> cumulativeLenBlue = blueRaceline.second;
    std::vector<double> cumulativeLenYellow = yellowRaceline.second;

    // get the total length of both racelines
        double totalBlueSplinesLength = blueRaceline.second[cumulativeLenBlue.size()-1];
        double totalYellowSplinesLength = yellowRaceline.second[cumulativeLenYellow.size()-1];

    for (double percent = b->startProgress; percent <= b->endProgress; percent += interval){
        // convert percent prog into meter prog for both racelines
        double progMeterBlue = (percent*totalBlueSplinesLength)/100;
        double progMeterYellow = (percent*totalYellowSplinesLength)/100;
      
        std::pair<double, double> xyBlue = interpolate_raceline(progMeterBlue, blueRaceline.first, 
                                                        blueRaceline.second, 20);
        std::pair<double, double> xyYellow = interpolate_raceline(progMeterYellow, yellowRaceline.first, 
                                                        yellowRaceline.second, 20);

        b->bluePoints.push_back(xyBlue);
        b->yellowPoints.push_back(xyYellow);
    }
}

/** @brief function to make vector of splines from std::vector<std::pair<double,double>> 
    (blue or yellow cones) of perception data
    @param cones perceptions_data.bluecones or perceptions_data.yellowcones
    @return vector of splines, vector of cumulative lengths
*/
std::pair<std::vector<Spline>,std::vector<double>> makeSplinesVector(std::vector<std::pair<double,double>> cones) {
    Eigen::MatrixXd coneMatrix(2, cones.size());
    for(int i = 0; i < cones.size(); i++){
        assert(i < coneMatrix.cols());
        coneMatrix(0, i) = cones[i].first;
        coneMatrix(1, i) = cones[i].second;
    }

    std::cout << coneMatrix << std::endl;

    /*@TODO: how to make a dummy logger?*/
    auto dummy_logger = rclcpp::get_logger("du");
    std::pair<std::vector<Spline>,std::vector<double>> res = raceline_gen(dummy_logger, coneMatrix, std::rand(), 4, false);

    return res;
}


/** @brief updates a given vector with the progress sections to optimize over
* @note about the usage of progressVector: vector of progresses that indicate sections to
* optimize over, should optimize as car is going
* @param progressVector can be accessed by the optimizer file as it is being updated
*/
void updateSegments(std::vector<bucket> bucketVector, 
                    std::vector<std::pair<double,double>> blueCones,
                    std::vector<std::pair<double,double>> yellowCones) {
    // make vector of splines
    // vector of outer cones
    // std::vector<std::pair<double,double>> blueCones = perceptions_data.bluecones;
    // std::vector<std::pair<double,double>> yellowCones = perceptions_data.yellowcones;

    std::pair<std::vector<Spline>,std::vector<double>> blueRes = makeSplinesVector(blueCones);
    std::pair<std::vector<Spline>,std::vector<double>> yellowRes = makeSplinesVector(yellowCones);

    std::vector<Spline> racetrackSplines = blueRes.first;
    std::vector<double> cumulativeLen = blueRes.second;

    // init bucket
    bucket currBucket;
    currBucket.startProgress = 0;
    currBucket.startProgress = 0;
    currBucket.sumCurvature = 0;
    
    // loop through progress and use get curvature on each progress
    int increment = 1; // FLEXIBLE: change this if running too slow
    int totalProgress = 100;
    int totalBlueSplinesLength = cumulativeLen[cumulativeLen.size()-1];
    // double totalProgress = cumulativeLen[cumulativeLen.size()-1]
    
    for (int currPercentProgress = 0; currPercentProgress <= totalProgress; currPercentProgress += increment) {
        double currProgress = (currPercentProgress*totalBlueSplinesLength)/100; // progress in meters
        std::vector<double> currProgressVec;
        currProgressVec.push_back(currProgress);
        double curve = get_curvature_raceline(currProgressVec, racetrackSplines, cumulativeLen)[0];
        // compare curve to avgCurvature of the curr bucket
        currBucket.endProgress = currPercentProgress;
        if (!checkStartNewBucket(currBucket, curve)) {
            // if fits in curr bucket, update the sum curvature
            currBucket.sumCurvature += curve;
        }
        else { // if doesn't fit, new bucket, start new average
            // take care of splits for curr bucket
            progressSplits(&currBucket, blueRes, yellowRes); // fill in the current bucket's blue and yellow points vectors
            currBucket.avgCurvature = calcRunningAvgCurv(currBucket);
            bucketVector.push_back(currBucket);
            bucket currBucket;
            currBucket.startProgress = currPercentProgress;
            currBucket.endProgress = currPercentProgress;
            currBucket.sumCurvature = curve;
        }
    }
}

// int main () {
//     return 0;
// }