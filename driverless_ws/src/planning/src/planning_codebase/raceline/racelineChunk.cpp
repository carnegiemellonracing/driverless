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

    bool infsec = false;
    if ((spline1_second_der == std::numeric_limits<double>::infinity() && abs(spline2_second_der) >= CHUNK_INFINITY_THRESH) || 
        (spline2_second_der == std::numeric_limits<double>::infinity()) && abs(spline1_second_der) >= CHUNK_INFINITY_THRESH){
            infsec = true;
        }
    bool checkSecondDer = infsec || (abs(spline1_second_der - spline2_second_der) <= CHUNK_SECOND_DER_THRESH);

    bool infthird = false;
    if ((spline1_third_der == std::numeric_limits<double>::infinity() && abs(spline2_third_der) >= CHUNK_INFINITY_THRESH) || 
        (spline2_third_der == std::numeric_limits<double>::infinity()) && abs(spline1_third_der) >= CHUNK_INFINITY_THRESH){
            infthird = true;
        }
    bool checkThirdDer = infthird || (abs(spline1_third_der - spline2_third_der) <= CHUNK_THIRD_DER_THRESH);

    if (!(checkFirstDer && checkSecondDer && checkThirdDer)) {
        std::cout << "first der diff" << abs(spline1_first_der - spline2_first_der) << std::endl;
        std::cout << "second der diff" << abs(spline1_second_der - spline2_second_der) << std::endl;
        std::cout << "third der diff" << abs(spline1_third_der - spline2_third_der) << std::endl;
    }

    return checkFirstDer && checkSecondDer && checkThirdDer;
}

double ySplit(ParameterizedSpline spline, double targetArclength) {
    std::pair<polynomial, polynomial> splinePair = std::make_pair(spline.spline_x.spl_poly, spline.spline_y.spl_poly);
    double low = 0;
    double high = 1;
    double mid;
    double curArclength;
    while (high-low >= 0.001) {
        double mid = low + (high-low) / 2;
        double curArclength = arclength(splinePair, 0, mid);
        if (abs(curArclength - targetArclength) <= 0.001) {
            return mid;
        }
        else if (curArclength < targetArclength) {
            low = mid + 0.01;
        }
        else {
            high = mid;
        }
    }
    
    return low + (high-low) / 2;
}

void print_poly_1(Spline x, Spline y) {
    std::cout << "(["<< x.spl_poly.nums(0) << "," << x.spl_poly.nums(1) << ","
     << x.spl_poly.nums(2) << "," << x.spl_poly.nums(3) << "]," 
     << "[" << y.spl_poly.nums(0) << "," << y.spl_poly.nums(1) << ","
     << y.spl_poly.nums(2) << "," << y.spl_poly.nums(3) << "])"<< std::endl;
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

    std::cout << "YELLOW RACELINE START" << std::endl;

    for (int i = 0; i < yellowRacetrackSplines.size(); i++) {
        print_poly_1(yellowRacetrackSplines[i].spline_x, yellowRacetrackSplines[i].spline_y);
    }

    std::cout << "YELLOW RACELINE END" << std::endl;


    // create a chunk
    Chunk* chunk = new Chunk();
    chunk->blueSplines.push_back(blueRacetrackSplines[0]);
    chunk->tStart = 0;
    
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
            bluePercentProgress = blueCumulativeLen[i - 1] / blueCumulativeLen[blueCumulativeLen.size() - 1];
            std::cout << "blue full" << blueCumulativeLen[blueCumulativeLen.size() - 1] << std::endl;
            std::cout << "blue percent" << bluePercentProgress << std::endl;
            
            // yellowindex is greater than yellowRacetrackSplines or 
            // cumsum is greater than cumsum of blue;yellowSplineIdx
            while ((yellowSplineIdx < yellowRacetrackSplines.size()) && 
                (yellowCumulativeLen[yellowSplineIdx + 1] <= yellowCumulativeLen[yellowCumulativeLen.size() - 1] * bluePercentProgress)) {
                    std::cout << "yellow length" << yellowCumulativeLen[yellowSplineIdx] << std::endl;
                    std::cout << "length thresh" << yellowCumulativeLen[yellowCumulativeLen.size() - 1] * bluePercentProgress << std::endl;
                chunk->yellowSplines.push_back(yellowRacetrackSplines[yellowSplineIdx]);
                // print_poly_1(yellowRacetrackSplines[yellowSplineIdx].spline_x, yellowRacetrackSplines[yellowSplineIdx].spline_y);
                yellowSplineIdx++;
            }

            double nextTStart = 0;

            // split yellow if yellow spline included in chunk is longer than curr blue
            if (yellowSplineIdx < yellowRacetrackSplines.size()) {
                ParameterizedSpline splitSpline = yellowRacetrackSplines[yellowSplineIdx];
                
                // takes in spline, x-y, returns t value
                // x is the blue percent prog, y is yellow percent prog
                double splitT = ySplit(splitSpline, (bluePercentProgress * yellowCumulativeLen[yellowCumulativeLen.size() - 1]) - yellowCumulativeLen[yellowSplineIdx]);
                chunk->tEnd = splitT;
                if (splitT == 1) {
                    nextTStart = 0;
                    yellowSplineIdx++;
                }
                else {
                    nextTStart = splitT;
                }
                chunk->yellowSplines.push_back(yellowRacetrackSplines[yellowSplineIdx]);
                // print_poly_1(yellowRacetrackSplines[yellowSplineIdx].spline_x, yellowRacetrackSplines[yellowSplineIdx].spline_y);
            }

            chunkVector->emplace_back(chunk);
            if (i != blueRacetrackSplines.size()) {
                chunk = new Chunk();
                chunk->tStart = nextTStart;
                // init chunk and add curr spline
                chunk->blueSplines.push_back(blueRacetrackSplines[i]);
            }
        }
    }

    return chunkVector;
}