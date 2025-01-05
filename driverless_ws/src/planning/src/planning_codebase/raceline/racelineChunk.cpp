#include "racelineChunk.hpp"

#include <math.h>
// #include "../midline/generator.hpp"


// tunable params for chunks
// thresholds are arclength in meters
#define CHUNK_LEN_MAX_THRESH 1000
#define CHUNK_LEN_MIN_THRESH 0
#define CHUNK_CURVE_THRESH 0.02
#define CHUNK_FIRST_DER_THRESH 1
#define CHUNK_SECOND_DER_THRESH 1
#define CHUNK_THIRD_DER_THRESH 1

/**
 * Constructor for chunks.
 */
Chunk::Chunk() = default;

/** 
 * TODO handle infinity subtraction
 *
 * Checks if given chunk should be terminated, i.e. running average of
 * given chunk is significantly different from the given curvature
 * sample or the chunk is too long. 
 *
 * @param newCurvature Curvature of new point.
 * 
 * @return True if this chunk should be terminated and should not
 *         include given curvature point, false otherwise.
 */
bool Chunk::checkContinueChunk(ParameterizedSpline spline1, ParameterizedSpline spline2) {
    
    bool checkFirstDer = abs(spline1.get_first_der(1) - spline2.get_first_der(0)) < CHUNK_FIRST_DER_THRESH;
    bool checkSecondDer = abs(spline1.get_second_der(1) - spline2.get_second_der(0)) < CHUNK_SECOND_DER_THRESH;
    bool checkThirdDer = abs(spline1.get_third_der(1) - spline2.get_third_der(0)) < CHUNK_THIRD_DER_THRESH;
    return checkFirstDer && checkSecondDer && checkThirdDer;
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

    /* Getting the polynomials/splines for each track bound*/
    /* Pass in all of the blue and yellow cones, */
    std::cout << "make splines done" << std::endl;
    std::pair<std::vector<ParameterizedSpline>,std::vector<double>> blue = make_splines_vector(blueCones);
    std::cout << "make splines done" << std::endl;
    std::pair<std::vector<ParameterizedSpline>,std::vector<double>> yellow = make_splines_vector(yellowCones);

    std::cout << "make splines done" << std::endl;

    std::vector<ParameterizedSpline> blueRacetrackSplines = blue.first;
    std::cout << "make splines done" << std::endl;
    std::vector<double> blueCumulativeLen = blue.second;
    std::cout << "make splines done" << std::endl;

    std::vector<ParameterizedSpline> yellowRacetrackSplines = yellow.first;
    std::cout << "make splines done" << std::endl;
    std::vector<double> yellowCumulativeLen = yellow.second;
    std::cout << "make splines done" << std::endl;


    // create a chunk
    Chunk* chunk = new Chunk();
    
    // // loop through progress and sample curvature at each progress point
    // int increment = 1; // TODO: tunable param
    // int totalProgress = 100;
    // int totalBlueLength = cumulativeLen[cumulativeLen.size()-1];
    // // LOOP THROUGH SPLINES
    // // IN T VALUE, IF first last CHECK FIRST DERIVATIVE AND THIRD DERIVATIVE MATCH UP AND SECOND DERIVATIVE SIGN MATCH UP
    // // TODO
    
    double bluePercentProgress;
    int yellowSplineIdx = 0;
    for (int i = 1; i <= blueRacetrackSplines.size(); i++) {
        // add spline to chunk
        if (!chunk->checkContinueChunk(blueRacetrackSplines[i-1], blueRacetrackSplines[i]) && 
            i < blueRacetrackSplines.size()) {
            chunk->blueSplines.push_back(blueRacetrackSplines[i]);
        }
        // stop current chunk, add to vector, start new chunk
        else { 
            // TODO makevector for yellow
            bluePercentProgress = blueCumulativeLen[i - 1] / blueCumulativeLen[-1];
            
            // yellowindex is greater than yellowRacetrackSplines or 
            // cumsum is greater than cumsum of blue;yellowSplineIdx
            while ((yellowSplineIdx < yellowRacetrackSplines.size()) || 
                (yellowCumulativeLen[yellowSplineIdx]<= yellowCumulativeLen[-1] * bluePercentProgress)) {
                chunk->yellowSplines.push_back(yellowRacetrackSplines[yellowSplineIdx]);
                yellowSplineIdx++;
            }
            chunkVector->emplace_back(chunk);
            if (i != blueRacetrackSplines.size()) {
                chunk = new Chunk();
                // init chunk and add curr spline
                chunk->blueSplines.push_back(blueRacetrackSplines[i]);
            }
        }
    }

    return chunkVector;
}