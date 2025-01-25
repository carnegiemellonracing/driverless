#include "racelineChunk.hpp"

#include <math.h>
// #include "../midline/generator.hpp"


// tunable params for chunks
// thresholds are arclength in meters
#define CHUNK_LEN_MAX_THRESH 1000
#define CHUNK_LEN_MIN_THRESH 15
#define CHUNK_FIRST_DER_THRESH 0.01
#define CHUNK_SECOND_DER_THRESH 1
#define CHUNK_THIRD_DER_THRESH 1
#define CHUNK_INFINITY_THRESH 500.0

/**
 * Constructor for chunks.
 */
Chunk::Chunk() = default;

/** 
 * Returns true if the thresholds for derivatives, otherwise returns false.
 * 
 * The 1st, 2nd, and 3rd derivatives for spline1 and spline2 should differ below
 * certain threshold as declared above.
 * Otherwise returns false.
 * 
 * @param spline1 the previous spline we check against
 * @param spline2 the current spline 
 *  
 */
bool Chunk::continueChunk(ParameterizedSpline spline1, ParameterizedSpline spline2) {
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

    return checkFirstDer && checkSecondDer && checkThirdDer;
}

/**
 * Return the best match t that corresponds to the arclength in the spline.
 * 
 * Arclength from start of spline to t should be targetArclength.
 * 
 * @param spline spline that t should reside in
 * @param targetArclength arclength from start of spline to t should be targetArclength
 */
double tInterpolate(ParameterizedSpline spline, double targetArclength) {
    std::pair<polynomial, polynomial> splinePair = std::make_pair(spline.spline_x.first_der, spline.spline_y.first_der);
    double low = 0;
    double high = 1;
    double mid;
    double curArclength;
    while (high-low >= 0.000001) {
        double mid = low + (high-low) / 2;
        double curArclength = arclength(splinePair, 0, mid);
        if (abs(curArclength - targetArclength) <= 0.000001) {
            return mid;
        }
        else if (curArclength < targetArclength) {
            low = mid + 0.00001;
        }
        else {
            high = mid;
        }
    }
    return low + (high-low) / 2;
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
    chunk->tStart = 0;
    chunk->blueArclengthStart = 0;
    
    double bluePercentProgress;
    int yellowSplineIdx = 0;

    int blueIdx = 0;
    int yellowIdx = 0;

    for (int i = 1; i <= blueRacetrackSplines.size(); i++) {
        // add spline to chunk
        if (i < blueRacetrackSplines.size() && 
            (chunk->continueChunk(blueRacetrackSplines[i-1], blueRacetrackSplines[i]))) {
            chunk->blueSplines.push_back(blueRacetrackSplines[i]);
        }
        // stop current chunk, add to vector, start new chunk
        else { 
            chunk->blueArclengthEnd = blueCumulativeLen[i - 1];
            bluePercentProgress = blueCumulativeLen[i - 1] / blueCumulativeLen[blueCumulativeLen.size() - 1];
            
            // GET CORRESPONDING YELLOW CHUNK
            // yellowindex is greater than yellowRacetrackSplines or 
            // cumsum is greater than cumsum of blue;yellowSplineIdx
            while ((yellowSplineIdx < yellowRacetrackSplines.size()) && 
                (yellowCumulativeLen[yellowSplineIdx] < yellowCumulativeLen[yellowCumulativeLen.size() - 1] * bluePercentProgress)) {
                chunk->yellowSplines.push_back(yellowRacetrackSplines[yellowSplineIdx]);
                yellowSplineIdx++;
            }

            double nextTStart = 0;

            // split yellow if yellow spline included in chunk is longer than curr blue
            if (yellowSplineIdx < yellowRacetrackSplines.size()) {
                ParameterizedSpline splitSpline = yellowRacetrackSplines[yellowSplineIdx];

                double yellowStartLen = 0;

                if (yellowSplineIdx != 0) {
                    yellowStartLen = yellowCumulativeLen[yellowSplineIdx - 1];
                }
                
                double splitT = tInterpolate(splitSpline, (bluePercentProgress * yellowCumulativeLen[yellowCumulativeLen.size() - 1]) - yellowStartLen);
                chunk->tEnd = splitT;

                chunk->yellowSplines.push_back(splitSpline);

                if (splitT == 1) {
                    nextTStart = 0;
                    yellowSplineIdx++;
                }
                else {
                    nextTStart = splitT;
                }
            }

            // FIND ALL FIELDS NEEDED FOR A CHUNK
            chunkVector->emplace_back(chunk);

            chunk->blueArclength = chunk->blueArclengthEnd - chunk->blueArclengthStart;
            chunk->yellowArclength = chunk->blueArclength * yellowCumulativeLen[yellowCumulativeLen.size() - 1] / blueCumulativeLen[blueCumulativeLen.size() - 1];

            chunk->blueFirstDerXStart = poly_eval(chunk->blueSplines[0].spline_x.first_der, 0);
            chunk->blueFirstDerXEnd = poly_eval(chunk->blueSplines[chunk->blueSplines.size() - 1].spline_x.first_der, 1);
            chunk->blueFirstDerYStart = poly_eval(chunk->blueSplines[0].spline_y.first_der, 0);
            chunk->blueFirstDerYEnd = poly_eval(chunk->blueSplines[chunk->blueSplines.size() - 1].spline_y.first_der, 1);

            chunk->yellowFirstDerXStart = poly_eval(chunk->yellowSplines[0].spline_x.first_der, chunk->tStart);
            chunk->yellowFirstDerXEnd = poly_eval(chunk->yellowSplines[chunk->yellowSplines.size() - 1].spline_x.first_der, chunk->tEnd);
            chunk->yellowFirstDerYStart = poly_eval(chunk->yellowSplines[0].spline_y.first_der, chunk->tStart);
            chunk->yellowFirstDerYEnd = poly_eval(chunk->yellowSplines[chunk->yellowSplines.size() - 1].spline_y.first_der, chunk->tEnd);

            double blueArcStart = chunk->blueArclengthEnd;

            // FIND MIDPOINTS
            while (blueCumulativeLen[blueIdx] < (chunk->blueArclength/2 + chunk->blueArclengthStart)) {
                blueIdx += 1;
            }

            double midFromMidSpline = (chunk->blueArclength/2 + chunk->blueArclengthStart);
            if (blueIdx != 0) {
                midFromMidSpline = midFromMidSpline - blueCumulativeLen[blueIdx-1];
            }

            // binary search from start of blueIdx spline 
            double midT = tInterpolate(blueRacetrackSplines[blueIdx], midFromMidSpline);

            chunk->blueMidX = poly_eval(blueRacetrackSplines[blueIdx].spline_x.spl_poly, midT);
            chunk->blueMidY = poly_eval(blueRacetrackSplines[blueIdx].spline_y.spl_poly, midT);
            chunk->blueStartX = poly_eval(chunk->blueSplines[0].spline_x.spl_poly, 0);
            chunk->blueStartY = poly_eval(chunk->blueSplines[0].spline_y.spl_poly, 0);
            chunk->blueEndX = poly_eval(chunk->blueSplines[chunk->blueSplines.size() - 1].spline_x.spl_poly, 1);
            chunk->blueEndY = poly_eval(chunk->blueSplines[chunk->blueSplines.size() - 1].spline_y.spl_poly, 1);
            chunk->blueFirstDerMidX = poly_eval(blueRacetrackSplines[blueIdx].spline_x.first_der, midT);
            chunk->blueFirstDerMidY = poly_eval(blueRacetrackSplines[blueIdx].spline_y.first_der, midT);

            double yellowEndLength = yellowCumulativeLen[yellowCumulativeLen.size() - 1];

            if (i != blueRacetrackSplines.size()) {
                // yellow midpoint and tangent
                ParameterizedSpline yellowSpline = yellowRacetrackSplines[yellowSplineIdx];
                yellowEndLength = arclength(std::make_pair(yellowSpline.spline_x.first_der, yellowSpline.spline_y.first_der), 0, chunk->tEnd);
                if (yellowSplineIdx > 0) {
                    yellowEndLength += yellowCumulativeLen[yellowSplineIdx - 1];
                }
            }

            while (yellowCumulativeLen[yellowIdx] < (yellowEndLength - chunk->yellowArclength/2)) {
                yellowIdx += 1;
            }

            // spline containing
            midFromMidSpline = (yellowEndLength - chunk->yellowArclength/2);
            if (yellowIdx != 0) {
                midFromMidSpline = midFromMidSpline - yellowCumulativeLen[yellowIdx-1];
            }
            
            // binary search from start of yellowIdx spline 
            midT = tInterpolate(yellowRacetrackSplines[yellowIdx], midFromMidSpline);

            chunk->yellowMidX = poly_eval(yellowRacetrackSplines[yellowIdx].spline_x.spl_poly, midT);
            chunk->yellowMidY = poly_eval(yellowRacetrackSplines[yellowIdx].spline_y.spl_poly, midT);
            chunk->yellowStartX = poly_eval(chunk->yellowSplines[0].spline_x.spl_poly, chunk->tStart);
            chunk->yellowStartY = poly_eval(chunk->yellowSplines[0].spline_y.spl_poly, chunk->tStart);
            chunk->yellowEndX = poly_eval(chunk->yellowSplines[chunk->yellowSplines.size() - 1].spline_x.spl_poly, chunk->tEnd);
            chunk->yellowEndY = poly_eval(chunk->yellowSplines[chunk->yellowSplines.size() - 1].spline_y.spl_poly, chunk->tEnd);
            chunk->yellowFirstDerMidX = poly_eval(yellowRacetrackSplines[yellowIdx].spline_x.first_der, midT);
            chunk->yellowFirstDerMidY = poly_eval(yellowRacetrackSplines[yellowIdx].spline_y.first_der, midT);
            

            if (i != blueRacetrackSplines.size()) {

                chunk = new Chunk();
                chunk->tStart = nextTStart;
                chunk->blueArclengthStart = blueArcStart;
                // init chunk and add curr spline
                chunk->blueSplines.push_back(blueRacetrackSplines[i]);
            }
        }

    }

    return chunkVector;
}