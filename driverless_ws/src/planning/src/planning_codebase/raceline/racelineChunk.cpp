#include "racelineChunk.hpp"

#include <math.h>
// #include "../midline/generator.hpp"


// tunable params for chunks
// thresholds are arclength in meters
#define CHUNK_LEN_MAX_THRESH 1000
#define CHUNK_LEN_MIN_THRESH 0
#define CHUNK_FIRST_DER_THRESH 0.01
#define CHUNK_SECOND_DER_THRESH 1
#define CHUNK_THIRD_DER_THRESH 1
#define CHUNK_INFINITY_THRESH 500.0

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
 * @return True if this chunk should not be terminated and should not
 *         include given curvature point, false otherwise.
 */
bool Chunk::checkContinueChunk(ParameterizedSpline spline1, ParameterizedSpline spline2) {
    double spline1_first_der = spline1.get_first_der(1);
    double spline2_first_der = spline2.get_first_der(0);
    double spline1_second_der = spline1.get_second_der(1);
    double spline2_second_der = spline2.get_second_der(0);
    double spline1_third_der = spline1.get_third_der(1);
    double spline2_third_der = spline2.get_third_der(0);

    bool inffirst = false;
    if ((spline1_first_der == std::numeric_limits<double>::infinity() && abs(spline2_first_der) >= CHUNK_INFINITY_THRESH) || 
        (spline2_first_der == std::numeric_limits<double>::infinity()) && abs(spline1_first_der) >= CHUNK_INFINITY_THRESH){
            inffirst = true;
        }
    bool checkFirstDer = inffirst || (abs(spline1_first_der - spline2_first_der) <= CHUNK_FIRST_DER_THRESH);
    std::cout << "first der diff" << abs(spline1_first_der - spline2_first_der) << std::endl;

    bool infsec = false;
    if ((spline1_second_der == std::numeric_limits<double>::infinity() && abs(spline2_second_der) >= CHUNK_INFINITY_THRESH) || 
        (spline2_second_der == std::numeric_limits<double>::infinity()) && abs(spline1_second_der) >= CHUNK_INFINITY_THRESH){
            infsec = true;
        }
    bool checkSecondDer = infsec || (abs(spline1_second_der - spline2_second_der) <= CHUNK_SECOND_DER_THRESH);
    std::cout << "second der diff" << abs(spline1_second_der - spline2_second_der) << std::endl;

    bool infthird = false;
    if ((spline1_third_der == std::numeric_limits<double>::infinity() && abs(spline2_third_der) >= CHUNK_INFINITY_THRESH) || 
        (spline2_third_der == std::numeric_limits<double>::infinity()) && abs(spline1_third_der) >= CHUNK_INFINITY_THRESH){
            infthird = true;
        }
    bool checkThirdDer = infthird || (abs(spline1_third_der - spline2_third_der) <= CHUNK_THIRD_DER_THRESH);
    std::cout << "third der diff" << abs(spline1_third_der - spline2_third_der) << std::endl;

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
    std::pair<std::vector<ParameterizedSpline>,std::vector<double>> blue = make_splines_vector(blueCones);
    std::pair<std::vector<ParameterizedSpline>,std::vector<double>> yellow = make_splines_vector(yellowCones);


    std::vector<ParameterizedSpline> blueRacetrackSplines = blue.first;
    std::vector<double> blueCumulativeLen = blue.second;

    std::vector<ParameterizedSpline> yellowRacetrackSplines = yellow.first;
    std::vector<double> yellowCumulativeLen = yellow.second;


    // create a chunk
    Chunk* chunk = new Chunk();
    chunk->blueSplines.push_back(blueRacetrackSplines[0]);
    
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
        if (i < blueRacetrackSplines.size() && (chunk->checkContinueChunk(blueRacetrackSplines[i-1], blueRacetrackSplines[i]))) {
            chunk->blueSplines.push_back(blueRacetrackSplines[i]);
        }
        // stop current chunk, add to vector, start new chunk
        else { 
            // TODO makevector for yellow
            bluePercentProgress = blueCumulativeLen[i - 1] / blueCumulativeLen[-1];
            std::cout << i + 9 << "make splines done" << std::endl;
            
            // yellowindex is greater than yellowRacetrackSplines or 
            // cumsum is greater than cumsum of blue;yellowSplineIdx
            while ((yellowSplineIdx < yellowRacetrackSplines.size()) && 
                (yellowCumulativeLen[yellowSplineIdx]<= yellowCumulativeLen[-1] * bluePercentProgress)) {
                    std::cout << i + 9 << "make splines donel" << std::endl;
                chunk->yellowSplines.push_back(yellowRacetrackSplines[yellowSplineIdx]);
                yellowSplineIdx++;
            }
            chunkVector->emplace_back(chunk);
            std::cout << i + 9 << "make splines done" << std::endl;
            if (i != blueRacetrackSplines.size()) {
                chunk = new Chunk();
                // init chunk and add curr spline
                chunk->blueSplines.push_back(blueRacetrackSplines[i]);
            }
            std::cout << i + 9 << "make splines done" << std::endl;
        }
    }

    return chunkVector;
}