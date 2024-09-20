#include "racelineChunk.hpp"

#include <math.h>
#include "raceline.hpp"
#include "../midline/generator.hpp"

/**
 * Constructor for chunks.
 */
Chunk::Chunk() {
    startProgress = 0;
    endProgress = 0;
    sumCurvature = 0;
    avgCurvature = 0;
}

/** 
 * Calculate curvature running average.
 */
double Chunk::calcRunningAvgCurvature() {
    return sumCurvature / (endProgress - startProgress);
}

/** 
 * Checks if given chunk should be terminated, i.e. running average of
 * given chunk is significantly different from the given curvature
 * sample or the chunk is too long. 
 *
 * @param newCurvature Curvature of new point.
 * 
 * @return True if this chunk should be terminated and should not
 *         include given curvature point, false otherwise.
 */
bool Chunk::checkStopChunk(double newCurvature) {
    double sigDiff = 1; // TODO: tunable param
    double avgCurve = calcRunningAvgCurvature();
    if (abs(avgCurve - newCurvature) >= sigDiff && 
            (endProgress - startProgress) < 15) {
        return true;
    }
    return false;
}

/** 
 * Populate the point vectors in the chunk by interpolating points along
 * the given splines.
 *
 * @param blueRaceline Vectors of splines and cumulative lengths for blue cones.
 * @param yellowRaceline Vectors of splines and cumulative lengths for yellow
 *                       cones.
 */
void Chunk::generateConePoints(std::pair<std::vector<Spline>,std::vector<double>> blueRaceline,
                        std::pair<std::vector<Spline>,std::vector<double>> yellowRaceline) {
    double interval = 0.5; // @TODO tunable param
    std::vector<double> blueLengths = blueRaceline.second;
    std::vector<double> yellowLengths = yellowRaceline.second;

    // get the total length of both racelines
    double totalBlueLength = blueLengths[blueLengths.size()-1];
    double totalYellowLength = yellowLengths[yellowLengths.size()-1];

    for (double percent = startProgress; percent <= endProgress; percent += interval){
        // convert percent progress into meter progress for both racelines
        double progressBlue_m = (percent*totalBlueLength)/100;
        double progressYellow_m = (percent*totalYellowLength)/100;
      
        std::pair<double, double> xyBlue = interpolate_raceline(progressBlue_m, blueRaceline.first, 
                                                        blueRaceline.second, 20);
        std::pair<double, double> xyYellow = interpolate_raceline(progressYellow_m, yellowRaceline.first, 
                                                        yellowRaceline.second, 20);

        bluePoints.push_back(xyBlue);
        yellowPoints.push_back(xyYellow);
    }
}

/** 
 * Generates a vector of raceline chunks based on track boundaries.
 *
 * @param blueCones Vector of blue cone points.
 * @param yellowCones Vector of yellow cone points.
 * 
 * @return Vector of raceline chunks.
 */
std::vector<Chunk> generateChunks(std::vector<std::pair<double,double>> blueCones,
                                  std::vector<std::pair<double,double>> yellowCones) {

    // create chunk vector that stores chunks
    std::vector<Chunk> chunkVector;

    // make splines for track boundaries
    std::pair<std::vector<Spline>,std::vector<double>> blue = make_splines_vector(blueCones);
    std::pair<std::vector<Spline>,std::vector<double>> yellow = make_splines_vector(yellowCones);

    std::vector<Spline> racetrackSplines = blue.first;
    std::vector<double> cumulativeLen = blue.second;

    // create a chunk
    Chunk chunk;
    
    // loop through progress and sample curvature at each progress point
    int increment = 1; // TODO: tunable param
    int totalProgress = 100;
    int totalBlueLength = cumulativeLen[cumulativeLen.size()-1];
    
    for (int currPercentProgress = 0; currPercentProgress <= totalProgress; currPercentProgress += increment) {
        double currProgress = (currPercentProgress*totalBlueLength)/100; // progress in meters
        std::vector<double> currProgressVec;
        currProgressVec.push_back(currProgress);
        double curvature = get_curvature_raceline(currProgressVec, racetrackSplines, cumulativeLen)[0];
        // compare curvature to avgCurvature of the curr bucket
        chunk.endProgress = currPercentProgress;
        if (!chunk.checkStopChunk(curvature)) {
            // if curvature belongs in current chunk, updated sumCurvature
            chunk.sumCurvature += curvature;
        }
        else { 
            // if we need to stop current chunk, create a new chunk and update
            // previous chunk & add it to the chunk vector
            chunk.generateConePoints(blue, yellow); // fill in the current bucket's blue and yellow points vectors
            chunk.avgCurvature = chunk.calcRunningAvgCurvature();
            chunkVector.push_back(chunk);
            Chunk chunk;
            chunk.startProgress = currPercentProgress;
            chunk.endProgress = currPercentProgress;
            chunk.sumCurvature = curvature;
        }
    }

    return chunkVector;
}