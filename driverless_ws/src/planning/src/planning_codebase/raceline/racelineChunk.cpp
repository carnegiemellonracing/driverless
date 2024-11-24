
#include "racelineChunk.hpp"

#include <math.h>
#include "../midline/generator.hpp"


// tunable params for chunks
#define CHUNK_LEN_THRESH 1000
#define CHUNK_CURVE_THRESH 0.02

/**
 * Constructor for chunks.
 */
Chunk::Chunk() {
    startProgress = 0;
    endProgress = 0;
    curConcavitySign = Concavity::STRAIGHT;
    avgCurvature = 0;
}

// /** 
//  * Calculate curvature running average.
//  */
// double Chunk::calcRunningAvgCurvature() {
//     return sumCurvature / (endProgress - startProgress);
// }

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
bool Chunk::checkStopChunk(Concavity newConcavitySign) {
    return curConcavitySign != newConcavitySign || (endProgress - startProgress) > CHUNK_LEN_THRESH;
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

    for (double percent = startProgress; percent < endProgress; percent += interval){
        // convert percent progress into meter progress for both racelines
        double progressBlue_m = (percent*totalBlueLength)/100;
        double progressYellow_m = (percent*totalYellowLength)/100;

        // std::cout << "before interpolate" << std::endl;
        std::pair<double, double> xyBlue = interpolate_raceline(progressBlue_m, blueRaceline.first, blueRaceline.second, 200);
        std::pair<double, double> xyYellow = interpolate_raceline(progressYellow_m, yellowRaceline.first, yellowRaceline.second, 200);

        // std::cout << "after interpolate" << std::endl;
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

//TODO: should be returning by reference not value
std::vector<Chunk*>* generateChunks(std::vector<std::pair<double,double>> blueCones,
                                  std::vector<std::pair<double,double>> yellowCones) {

    // create chunk vector that stores chunks
    //TODO: use new keyword to create vector in heap not stack
    std::vector<Chunk*>* chunkVector = new std::vector<Chunk*>();

    // // std::cout << "init chunck vector" << std::endl;

    /* Getting the polynomials/splines for each track bound*/
    /* Pass in all of the blue and yellow cones, */
    std::pair<std::vector<Spline>,std::vector<double>> blue = make_splines_vector(blueCones);
    std::pair<std::vector<Spline>,std::vector<double>> yellow = make_splines_vector(yellowCones);

    std::vector<Spline> racetrackSplines = blue.first;
    std::vector<double> cumulativeLen = blue.second;

    // create a chunk
    Chunk* chunk = new Chunk();
    // // std::cout << "created new chunk" << std::endl;
    
    // loop through progress and sample curvature at each progress point
    int increment = 1; // TODO: tunable param
    int totalProgress = 100;
    int totalBlueLength = cumulativeLen[cumulativeLen.size()-1];
    
    for (int currPercentProgress = 0; currPercentProgress <= totalProgress; currPercentProgress += increment) {
        double currProgress = (currPercentProgress*totalBlueLength)/totalProgress; // progress in meters
        std::vector<double> currProgressVec;
        currProgressVec.push_back(currProgress);

        /* Get the concavity using the cubic spline interpolation from make_splines */
        Concavity cur_concavity_sign = get_curvature_raceline(currProgressVec, racetrackSplines, cumulativeLen);
        std::cout << concavity_to_string(cur_concavity_sign) << std::endl;

        /* Determine whether to split the chunk */
        chunk->endProgress = currPercentProgress;
        if (!chunk->checkStopChunk(cur_concavity_sign)) {
            // if curvature belongs in current chunk, updated sumCurvature
            //chunk->sumCurvature += curvature;
            std::cout << concavity_to_string(cur_concavity_sign) << std::endl;
            assert(cur_concavity_sign == chunk->curConcavitySign);
            // std::cout << "not created new chunk in loop" << std::endl;
        }
        else { 
            // if we need to stop current chunk, create a new chunk and update
            // previous chunk & add it to the chunk vector
            // std::cout << "new chunk" << std::endl;
            chunk->generateConePoints(blue, yellow); // fill in the current bucket's blue and yellow points vectors
            // std::cout << "new chunk 1" << std::endl;
            // chunk->avgCurvature = chunk->calcRunningAvgCurvature();
            // std::cout << "new chunk 2" << std::endl;
            //TODO: look into emplace_back
            chunkVector->emplace_back(chunk);
            // std::cout << "new chunk 3" << std::endl;
            chunk = new Chunk(); 
            // std::cout << "created new chunk in loop" << std::endl;
            chunk->startProgress = currPercentProgress;
            chunk->endProgress = currPercentProgress;
            chunk->curConcavitySign = cur_concavity_sign;
        }
    }
    chunk->generateConePoints(blue, yellow);
    // chunk->avgCurvature = chunk->calcRunningAvgCurvature();

    chunkVector->emplace_back(chunk);

    return chunkVector;
}